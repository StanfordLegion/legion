/* Copyright 2019 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __HALF_H__
#define __HALF_H__

#include <stdint.h>
#include <cmath>

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

// Still need this for doing conversions from floats
static inline uint16_t __convert_float_to_halfint(float a)
{
  uint32_t ia = *reinterpret_cast<uint32_t*>(&a);
  uint16_t ir;

  ir = (ia >> 16) & 0x8000;

  if ((ia & 0x7f800000) == 0x7f800000)
  {
    if ((ia & 0x7fffffff) == 0x7f800000)
    {
      ir |= 0x7c00; /* infinity */
    }
    else
    {
      ir = 0x7fff; /* canonical NaN */
    }
  }
  else if ((ia & 0x7f800000) >= 0x33000000)
  {
    int32_t shift = (int32_t) ((ia >> 23) & 0xff) - 127;
    if (shift > 15)
    {
      ir |= 0x7c00; /* infinity */
    }
    else
    {
      ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
      if (shift < -14)
      { /* denormal */
        ir |= ia >> (-1 - shift);
        ia = ia << (32 - (-1 - shift));
      }
      else
      { /* normal */
        ir |= ia >> (24 - 11);
        ia = ia << (32 - (24 - 11));
        ir = ir + ((14 + shift) << 10);
      }
      /* IEEE-754 round to nearest of even */
      if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1)))
      {
        ir++;
      }
    }
  }

  return ir;
}

static inline float __convert_halfint_to_float(uint16_t __x)
{
  int sign        = ((__x >> 15) & 1);
  int exp         = ((__x >> 10) & 0x1f);
  int mantissa    = (__x & 0x3ff);
  uint32_t f      = 0;

  if (exp > 0 && exp < 31)
  {
    // normal
    exp += 112;
    f = (sign << 31) | (exp << 23) | (mantissa << 13);
  }
  else if (exp == 0)
  {
    if (mantissa)
    {
      // subnormal
      exp += 113;
      while ((mantissa & (1 << 10)) == 0)
      {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    }
    else
    {
      // zero
      f = 0;
    }
  }
  else if (exp == 31)
  {
    if (mantissa)
    {
      f = 0x7fffffff;     // not a number
    }
    else
    {
      f = (0xff << 23) | (sign << 31);    //  inf
    }
  }
  return *reinterpret_cast<float const *>(&f);
}

#ifdef __CUDACC__
// This brings in __half from 
#define __CUDA_NO_HALF_OPERATORS__
#include <cuda_fp16.h>

__CUDA_HD__
inline __half operator-(const __half &one)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hneg(one);
#else
  return __float2half(-__half2float(one));
#endif
#else
  return __half(-(float(one)));
#endif
}

__CUDA_HD__
inline __half operator+(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hadd(one, two);
#else
  return __float2half(__half2float(one) + __half2float(two));
#endif
#else
  return __half(float(one) + float(two));
#endif
}

__CUDA_HD__
inline __half operator-(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hsub(one, two);
#else
  return __float2half(__half2float(one) - __half2float(two));
#endif
#else
  return __half(float(one) - float(two));
#endif
}

__CUDA_HD__
inline __half operator*(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hmul(one, two);
#else
  return __float2half(__half2float(one) * __half2float(two));
#endif
#else
  return __half(float(one) * float(two));
#endif
}

__CUDA_HD__
inline __half operator/(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hdiv(one, two);
#else
  return __float2half(__half2float(one) / __half2float(two));
#endif
#else
  return __half(float(one) / float(two));
#endif
}

__CUDA_HD__
inline bool operator==(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __heq(one, two);
#else
  return (__half2float(one) == __half2float(two));
#endif
#else
  return (float(one) == float(two));
#endif
}

__CUDA_HD__
inline bool operator!=(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hne(one, two);
#else
  return (__half2float(one) != __half2float(two));
#endif
#else
  return (float(one) != float(two));
#endif
}

__CUDA_HD__
inline bool operator<(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hlt(one, two);
#else
  return (__half2float(one) < __half2float(two));
#endif
#else
  return (float(one) < float(two));
#endif
}

__CUDA_HD__
inline bool operator<=(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hle(one, two);
#else
  return (__half2float(one) <= __half2float(two));
#endif
#else
  return (float(one) <= float(two));
#endif
}

__CUDA_HD__
inline bool operator>(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hgt(one, two);
#else
  return (__half2float(one) > __half2float(two));
#endif
#else
  return (float(one) > float(two));
#endif
}

__CUDA_HD__
inline bool operator>=(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hge(one, two);
#else
  return (__half2float(one) >= __half2float(two));
#endif
#else
  return (float(one) >= float(two));
#endif
}
#else
struct __half
{
  uint16_t __x;

  inline __half(void)
  {
    __x = 0;
  }

  /// Constructor from uint16_t
  inline __half(short a, bool raw)
  {
    if (raw)
      __x = a;
    else
      __x = __convert_float_to_halfint(float(a));
  }

  /// Constructor from float
  inline __half(float a)
  {
    __x = __convert_float_to_halfint(a);
  }

  inline __half& operator=(const __half &rhs)
  {
    __x = rhs.__x;
    return *this;
  }

  inline __half& operator=(const float &rhs)
  {
    __x = __convert_float_to_halfint(rhs);
    return *this;
  }

  /// Cast to float
  inline operator float() const
  {
    return __convert_halfint_to_float(__x);
  }

  /// Get raw storage
  inline uint16_t raw() const
  {
    return this->__x;
  }

  inline void set_raw(uint16_t raw)
  {
    this->__x = raw; 
  }

  /// Increment
  inline __half& operator +=(const __half &rhs)
  {
    *this = __half(float(*this) + float(rhs));
    return *this;
  }
  
  /// Decrement
  inline __half& operator -=(const __half&rhs)
  {
    *this = __half(float(*this) - float(rhs));
    return *this;
  }

  /// Scale up
  inline __half& operator *=(const __half &rhs)
  {
    *this = __half(float(*this) * float(rhs));
    return *this;
  }

  /// Scale down
  inline __half& operator /=(const __half &rhs)
  {
    *this = __half(float(*this) / float(rhs));
    return *this;
  }

};

inline __half operator-(const __half &one)
{
  return __half(-(float(one)));
}

inline __half operator+(const __half &one, const __half &two)
{
  return __half(float(one) + float(two));
}

inline __half operator-(const __half &one, const __half &two)
{
  return __half(float(one) - float(two));
}

inline __half operator*(const __half &one, const __half &two)
{
  return __half(float(one) * float(two));
}

inline __half operator/(const __half &one, const __half &two)
{
  return __half(float(one) / float(two));
}

inline bool operator==(const __half &one, const __half &two)
{
  return (float(one) == float(two));
}

inline bool operator!=(const __half &one, const __half &two)
{
  return (float(one) != float(two));
}

inline bool operator<(const __half &one, const __half &two)
{
  return (float(one) < float(two));
}

inline bool operator<=(const __half &one, const __half &two)
{
  return (float(one) <= float(two));
}

inline bool operator>(const __half &one, const __half &two)
{
  return (float(one) > float(two));
}

inline bool operator>=(const __half &one, const __half &two)
{
  return (float(one) >= float(two));
}
#endif // Not nvcc

static inline __half __convert_float_to_half(float a)
{
  uint16_t result = __convert_float_to_halfint(a);
  return *reinterpret_cast<__half*>(&result);
}

namespace std
{

// put these functions in namespace std so that we can call the
// std:: versions of them uniformly on arithmetic types


static inline __half acos(__half a)
{
  return ::acosf(a);
}


static inline __half asin(__half a)
{
  return ::asinf(a);
}


static inline __half atan(__half a)
{
  return ::atanf(a);
}

__CUDA_HD__
static inline __half ceil(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return hceil(a);
#else
  return __float2half(ceilf(__half2float(a)));
#endif
#else
  return ::ceilf(a);
#endif
}


__CUDA_HD__
static inline __half cos(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return hcos(a);
#else
  return __float2half(cosf(__half2float(a)));
#endif
#else
  return ::cosf(a);
#endif
}

__CUDA_HD__
static inline __half exp(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return hexp(a);
#else
  return __float2half(expf(__half2float(a)));
#endif
#else
  return ::expf(a);
#endif
}

__CUDA_HD__
static inline __half fabs(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hgt(a, __float2half(0.f)) ? __hneg(a) : a;
#else
  return __float2half(fabs(__half2float(a)));
#endif
#else
  return ::fabsf(a);
#endif
}

__CUDA_HD__
static inline __half floor(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return hfloor(a);
#else
  return __float2half(floorf(__half2float(a)));
#endif
#else
  return ::floorf(a);
#endif
}

__CUDA_HD__
static inline bool isinf(__half a)
{
#ifdef __CUDA_ARCH__
  return __hisinf(a);
#else
  return std::isinf(static_cast<float>(a));
#endif
}

__CUDA_HD__
static inline bool isnan(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return __hisnan(a);
#else
  return ::isnan(__half2float(a));
#endif
#else
  return std::isnan(static_cast<float>(a));
#endif
}

__CUDA_HD__
static inline __half log(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return hlog(a);
#else
  return __float2half(logf(__half2float(a)));
#endif
#else
  return ::logf(a);
#endif
}


static inline __half pow(__half x, __half exponent)
{
  return ::powf(x, exponent);
}

__CUDA_HD__
static inline __half sin(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return hsin(a);
#else
  return __float2half(sinf(__half2float(a)));
#endif
#else
  return ::sinf(a);
#endif
}


static inline __half tan(__half a)
{
  return ::tanf(a);
}


static inline __half tanh(__half a)
{
  return ::tanhf(a);
}

__CUDA_HD__
static inline __half sqrt(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530
  return hsqrt(a);
#else
  return __float2half(sqrtf(__half2float(a)));
#endif
#else
  return ::sqrtf(a);
#endif
}


} // end std

#endif // __HALF_H__

