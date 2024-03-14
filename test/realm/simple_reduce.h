#ifndef SIMPLE_REDUCE_H
#define SIMPLE_REDUCE_H

#include <cstdint>
#include <cstring>

class ReductionOpMixedAdd {
public:
  typedef double LHS;
  typedef int RHS;

  template <bool EXCL>
  static void apply(LHS& lhs, RHS rhs)
  {
    if(EXCL) {
      lhs += rhs;
    } else {
      // no FP64 atomics on cpu, so use compare_and_swap
      volatile uint64_t *tgtptr = reinterpret_cast<uint64_t *>(&lhs);
      while(true) {
        uint64_t origval = *tgtptr;
        LHS v;
        memcpy(&v, &origval, sizeof(LHS));
        v += rhs;
        uint64_t newval;
        memcpy(&newval, &v, sizeof(LHS));
        if(__sync_bool_compare_and_swap(tgtptr, origval, newval))
          break;
      }
    }
  }

  // both of these are optional
  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS& rhs1, RHS rhs2)
  {
    if(EXCL) {
      rhs1 += rhs2;
    } else {
      // non-exclusive fold is easier because we do have atomic integer add
      __sync_fetch_and_add(&rhs1, rhs2);
    }
  }

#if defined(REALM_USE_CUDA) && defined(__CUDACC__)
  static const bool has_cuda_reductions = true;

  // device methods for CUDA
  template <bool EXCL>
  static __device__ void apply_cuda(LHS& lhs, RHS rhs)
  {
    if(EXCL) {
      lhs += rhs;
    } else {
#if __CUDA_ARCH__ >= 600
      // sm_60 and up has native atomics on doubles
      atomicAdd(&lhs, (LHS)rhs);
#else
      // before sm_60, this requires a CAS - don't actually do an initial read,
      //  but guess that the value is 0 and the first CAS will serve as the
      //  read in the (common) case where we guessed wrong
      unsigned long long oldval = 0;
      while(true) {
        unsigned long long newval, chkval;
        newval = __double_as_longlong(__longlong_as_double(oldval) + rhs);
        chkval = atomicCAS(reinterpret_cast<unsigned long long *>(&lhs),
                           oldval, newval);
        if(chkval == oldval) break;
        oldval = chkval;
      }
#endif
    }
  }

  template <bool EXCL>
  static __device__ void fold_cuda(RHS& rhs1, RHS rhs2)
  {
    if(EXCL) {
      rhs1 += rhs2;
    } else {
      atomicAdd(&rhs1, rhs2);
    }
  }
#endif
  
#if defined(REALM_USE_HIP) && ( defined (__CUDACC__) || defined (__HIPCC__) )
  static const bool has_hip_reductions = true;

  // device methods for HIP
  template <bool EXCL>
  static __device__ void apply_hip(LHS& lhs, RHS rhs)
  {
    if(EXCL) {
      lhs += rhs;
    } else {
#if (__CUDA_ARCH__ >= 600) || defined(__HIP_DEVICE_COMPILE__)
      // sm_60 and up has native atomics on doubles
      atomicAdd(&lhs, (LHS)rhs);
#else
      // before sm_60, this requires a CAS - don't actually do an initial read,
      //  but guess that the value is 0 and the first CAS will serve as the
      //  read in the (common) case where we guessed wrong
      unsigned long long oldval = 0;
      while(true) {
        unsigned long long newval, chkval;
        newval = __double_as_longlong(__longlong_as_double(oldval) + rhs);
        chkval = atomicCAS(reinterpret_cast<unsigned long long *>(&lhs),
                           oldval, newval);
        if(chkval == oldval) break;
        oldval = chkval;
      }
#endif
    }
  }

  template <bool EXCL>
  static __device__ void fold_hip(RHS& rhs1, RHS rhs2)
  {
    if(EXCL) {
      rhs1 += rhs2;
    } else {
      atomicAdd(&rhs1, rhs2);
    }
  }
#endif
};

#endif
