#ifndef _PHILOX_H_
#define _PHILOX_H_

// CPU/GPU implementation of DE Shaw's Philox 2x32 PRNG

#ifndef CUDAPREFIX
#ifdef __NVCC__
#define CUDAPREFIX __device__ __forceinline__
#else
#define CUDAPREFIX
#endif
#endif

// their paper recommends 10 rounds by default
template <int ROUNDS = 10>
class Philox_2x32 {
public:
  typedef unsigned u32;
  typedef unsigned long long u64;

  static const u32 PHILOX_M2x32_0x = 0xD256D193U;
  static const u32 PHILOX_W32_0x = 0x9E3779B9U;

  CUDAPREFIX
  static u64 rand_raw(u32 key, u32 ctr_hi, u32 ctr_lo)
  {
#ifdef __NVCC__
    #pragma unroll
#endif
    for(int i = 0; i < ROUNDS; i++) {
      u32 prod_hi, prod_lo;
#ifdef __NVCC__
      prod_hi = __umulhi(ctr_lo, PHILOX_M2x32_0x);
      prod_lo = ctr_lo * PHILOX_M2x32_0x;
#else
      u64 prod = ((u64)ctr_lo) * PHILOX_M2x32_0x;
      prod_hi = prod >> 32;
      prod_lo = prod;
#endif
      ctr_lo = ctr_hi ^ key ^ prod_hi;
      ctr_hi = prod_lo;
      key += PHILOX_W32_0x;
    }
    return (((u64)ctr_hi) << 32) + ctr_lo;
  }

  // returns an unsigned 32-bit integer in the range [0, n)
  CUDAPREFIX
  static u32 rand_int(u32 key, u32 ctr_hi, u32 ctr_lo, u32 n)
  {
    // need 32 random bits
    u32 bits = rand_raw(key, ctr_hi, ctr_lo);
    // now treat them as a 0.32 fixed-point value, multiply by n and truncate
#ifdef __NVCC__
    return __umulhi(bits, n);
#else
    return (((u64)n) * ((u64)bits)) >> 32;
#endif
  }

  // returns a float in the range [0.0, 1.0)
  CUDAPREFIX
  static float rand_float(u32 key, u32 ctr_hi, u32 ctr_lo)
  {
    // need 32 random bits (we probably lose a bunch when this gets converted to float)
    u32 bits = rand_raw(key, ctr_hi, ctr_lo);
    // would prefer 0x1p-32 here, but pedantic c++ doesn't get it until c++17
    const float scale = 1.0f / (1ULL << 32);  // 2^-32
    return (bits * scale);
  }
};

#endif
  
