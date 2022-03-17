/* Copyright 2022 Stanford University, NVIDIA Corporation
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
#include <string.h> // memcpy
#include <cmath>

#ifndef __CUDA_HD__
#if defined (__CUDACC__) || defined (__HIPCC__)
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

// Still need this for doing conversions from floats
inline uint16_t __convert_float_to_halfint(float a)
{
  uint32_t ia = 0;
  static_assert(sizeof(ia) == sizeof(a), "half size mismatch");
  memcpy(&ia, &a, sizeof(a));
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

inline float __convert_halfint_to_float(uint16_t __x)
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
  float result = 0.f;
  static_assert(sizeof(float) == sizeof(f), "half size mismatch");
  memcpy(&result, &f, sizeof(f));
  return result;
}

#if defined (__CUDACC__) || defined (__HIPCC__)
// The CUDA Toolkit only provides device versions for half precision operators,
// so we have to provide custom implementations below.
#if defined(LEGION_USE_CUDA)
#define __CUDA_NO_HALF_OPERATORS__
#include <cuda_fp16.h>
#elif defined(LEGION_USE_HIP)
#ifdef __HIP_PLATFORM_NVCC__
#define __CUDA_NO_HALF_OPERATORS__
#include <cuda_fp16.h>
#else
#define __HIP_NO_HALF_OPERATORS__
#include <hip/hip_fp16.h>
#endif
#endif

__CUDA_HD__
inline __half operator-(const __half &one)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hneg(one);
#else
  return __float2half(-__half2float(one));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hneg(one);
#else
  return __half(-(float(one)));
#endif
}

__CUDA_HD__
inline __half operator+(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hadd(one, two);
#else
  return __float2half(__half2float(one) + __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hadd(one, two);
#else
  return __half(float(one) + float(two));
#endif
}

__CUDA_HD__
inline __half operator-(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hsub(one, two);
#else
  return __float2half(__half2float(one) - __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hsub(one, two);
#else
  return __half(float(one) - float(two));
#endif
}

__CUDA_HD__
inline __half operator*(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hmul(one, two);
#else
  return __float2half(__half2float(one) * __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hmul(one, two);
#else
  return __half(float(one) * float(two));
#endif
}

__CUDA_HD__
inline __half operator/(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ == 8
  return hdiv(one, two);
#elif __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 9
  return __hdiv(one, two);
#else
  return __float2half(__half2float(one) / __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hdiv(one, two);
#else
  return __half(float(one) / float(two));
#endif
}

__CUDA_HD__
inline bool operator==(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __heq(one, two);
#else
  return (__half2float(one) == __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __heq(one, two);
#else
  return (float(one) == float(two));
#endif
}

__CUDA_HD__
inline bool operator!=(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hne(one, two);
#else
  return (__half2float(one) != __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hne(one, two);
#else
  return (float(one) != float(two));
#endif
}

__CUDA_HD__
inline bool operator<(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hlt(one, two);
#else
  return (__half2float(one) < __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hlt(one, two);
#else
  return (float(one) < float(two));
#endif
}

__CUDA_HD__
inline bool operator<=(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hle(one, two);
#else
  return (__half2float(one) <= __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hle(one, two);
#else
  return (float(one) <= float(two));
#endif
}

__CUDA_HD__
inline bool operator>(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hgt(one, two);
#else
  return (__half2float(one) > __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hgt(one, two);
#else
  return (float(one) > float(two));
#endif
}

__CUDA_HD__
inline bool operator>=(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hge(one, two);
#else
  return (__half2float(one) >= __half2float(two));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hge(one, two);
#else
  return (float(one) >= float(two));
#endif
}

__CUDA_HD__
inline __half asin(const __half &one)
{
#ifdef __CUDA_ARCH__
  return (__float2half(asinf(__half2float(one))));
#elif defined(__HIP_DEVICE_COMPILE__)
  return (__float2half(asinf(__half2float(one))));
#else
  return (__float2half(std::asin(__half2float(one))));
#endif
}

__CUDA_HD__
inline __half atan(const __half &one)
{
#ifdef __CUDA_ARCH__
  return (__float2half(atanf(__half2float(one))));
#elif defined(__HIP_DEVICE_COMPILE__)
  return (__float2half(atanf(__half2float(one))));
#else
  return (__float2half(std::atan(__half2float(one))));
#endif
}

__CUDA_HD__
inline __half ceil(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return hceil(a);
#else
  return __float2half(ceilf(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return hceil(a);
#else
  return static_cast<__half>(::ceilf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline __half cos(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return hcos(a);
#else
  return __float2half(cosf(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return hcos(a);
#else
  return static_cast<__half>(::cosf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline __half exp(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return hexp(a);
#else
  return __float2half(expf(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return hexp(a);
#else
  return static_cast<__half>(::expf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline __half fabs(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && (__CUDACC_VER_MAJOR__ >= 11 || __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
  return __habs(a);
#else
  return __float2half(fabs(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __habs(a);
#else
  return static_cast<__half>(::fabsf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline __half floor(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return hfloor(a);
#else
  return __float2half(floorf(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return hfloor(a);
#else
  return static_cast<__half>(::floorf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline bool isinf(__half a)
{
#ifdef __CUDA_ARCH__
  return __hisinf(a);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hisinf(a);
#else
  return std::isinf(static_cast<float>(a));
#endif
}

__CUDA_HD__
inline bool isnan(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return __hisnan(a);
#else
  return ::isnan(__half2float(a));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __hisnan(a);
#else
  return std::isnan(static_cast<float>(a));
#endif
}

__CUDA_HD__
inline __half log(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return hlog(a);
#else
  return __float2half(logf(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return hlog(a);
#else
  return static_cast<__half>(::logf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline __half sin(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return hsin(a);
#else
  return __float2half(sinf(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return hsin(a);
#else
  return static_cast<__half>(::sinf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline __half sqrt(__half a)
{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 530 && __CUDACC_VER_MAJOR__ >= 8
  return hsqrt(a);
#else
  return __float2half(sqrtf(__half2float(a)));
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return hsqrt(a);
#else
  return static_cast<__half>(::sqrtf(static_cast<float>(a)));
#endif
}

__CUDA_HD__
inline __half pow(const __half &one, const __half &two)
{
#ifdef __CUDA_ARCH__
  return (__float2half(powf(__half2float(one), __half2float(two))));
#elif defined(__HIP_DEVICE_COMPILE__)
  return (__float2half(powf(__half2float(one), __half2float(two))));
#else
  return (__float2half(std::pow(__half2float(one), __half2float(two))));
#endif
}

__CUDA_HD__
inline __half tan(const __half &one)
{
#ifdef __CUDA_ARCH__
  return (__float2half(tanf(__half2float(one))));
#elif defined(__HIP_DEVICE_COMPILE__)
  return (__float2half(tanf(__half2float(one))));
#else
  return (__float2half(std::tan(__half2float(one))));
#endif
}

__CUDA_HD__
inline __half tanh(const __half &one)
{
#ifdef __CUDA_ARCH__
  return (__float2half(tanhf(__half2float(one))));
#elif defined(__HIP_DEVICE_COMPILE__)
  return (__float2half(tanhf(__half2float(one))));
#else
  return (__float2half(std::tanh(__half2float(one))));
#endif
}

__CUDA_HD__
inline __half acos(const __half &one)
{
#ifdef __CUDA_ARCH__
  return (__float2half(acosf(__half2float(one))));
#elif defined(__HIP_DEVICE_COMPILE__)
  return (__float2half(acosf(__half2float(one))));
#else
  return (__float2half(std::acos(__half2float(one))));
#endif
}

#else // not __CUDACC__ or __HIPCC__

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
  inline explicit __half(float a)
  {
    __x = __convert_float_to_halfint(a);
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

inline __half __convert_float_to_half(const float &a)
{
  uint16_t temp = __convert_float_to_halfint(a);
  __half result(0, true/*raw*/);
  result.set_raw(temp);
  return result;
}

inline __half floor(const __half &a)
{
  return static_cast<__half>(::floor(static_cast<float>(a)));
}

inline __half ceil(const __half &a)
{
  return static_cast<__half>(::ceil(static_cast<float>(a)));
}

inline __half exp(const __half &a)
{
  return static_cast<__half>(::exp(static_cast<float>(a)));
}

inline __half acos(const __half &a)
{
  return static_cast<__half>(::acos(static_cast<float>(a)));
}

inline __half cos(const __half &a)
{
  return static_cast<__half>(::cos(static_cast<float>(a)));
}

inline __half asin(const __half &a)
{
  return static_cast<__half>(::asin(static_cast<float>(a)));
}

inline __half sin(const __half &a)
{
  return static_cast<__half>(::sin(static_cast<float>(a)));
}

inline __half atan(const __half &a)
{
  return static_cast<__half>(::atan(static_cast<float>(a)));
}

inline __half tan(const __half &a)
{
  return static_cast<__half>(::tan(static_cast<float>(a)));
}

inline __half tanh(const __half &a)
{
  return static_cast<__half>(::tanh(static_cast<float>(a)));
}

inline __half fabs(const __half &a)
{
  return static_cast<__half>(::fabs(static_cast<float>(a)));
}

inline __half log(const __half &a)
{
  return static_cast<__half>(::log(static_cast<float>(a)));
}

inline __half pow(const __half &x, const __half &exponent)
{
  return static_cast<__half>(::pow(static_cast<float>(x), static_cast<float>(exponent)));
}

inline __half sqrt(const __half &a)
{
  return static_cast<__half>(::sqrt(static_cast<float>(a)));
}

#endif // Not nvcc or hipcc

#endif // __HALF_H__
