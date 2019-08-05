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
#include <cuda_fp16.h>
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

  /// Comparison 
  inline bool operator <(const __half &other) const
  {
    return (this->__x < other.__x);
  }

  /// Comparison 
  inline bool operator <=(const __half &other) const
  {
    return (this->__x <= other.__x);
  }

  /// Equality
  inline bool operator ==(const __half &other) const
  {
    return (this->__x == other.__x);
  }

  /// Inequality
  inline bool operator !=(const __half &other) const
  {
    return (this->__x != other.__x);
  }

  /// Comparison 
  inline bool operator >(const __half &other) const
  {
    return (this->__x > other.__x);
  }

  /// Comparison 
  inline bool operator >=(const __half &other) const
  {
    return (this->__x >= other.__x);
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

  /// Multiply
  inline __half operator*(const __half &other) const
  {
    return __half(float(*this) * float(other));
  }

  /// Addition 
  inline __half operator+(const __half &other) const
  {
    return __half(float(*this) + float(other));
  }

  /// Difference
  inline __half operator-(const __half &other) const
  {
    return __half(float(*this) - float(other));
  }

  /// Negate
  inline __half operator-(void) const
  {
    // xor with a sign bit flip
    uint16_t raw = this->__x ^ (1 << 15);
    return __half(raw, true/*raw*/);
  }

  /// Difference
  inline __half operator/(const __half &other) const
  {
    return __half(float(*this) / float(other));
  }

};
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


static inline __half ceil(__half a)
{
#ifdef __CUDA_ARCH__
  return hceil(a);
#else
  return ::ceilf(a);
#endif
}


static inline __half cos(__half a)
{
#ifdef __CUDA_ARCH__
  return hcos(a);
#else
  return ::cosf(a);
#endif
}


static inline __half exp(__half a)
{
#ifdef __CUDA_ARCH__
  return hexp(a);
#else
  return ::expf(a);
#endif
}


static inline __half fabs(__half a)
{
#ifdef __CUDA_ARCH__
  return (a < __half(0.f)) ? -a : a;
#else
  return ::fabsf(a);
#endif
}


static inline __half floor(__half a)
{
#ifdef __CUDA_ARCH__
  return hfloor(a);
#else
  return ::floorf(a);
#endif
}


static inline bool isinf(__half a)
{
#ifdef __CUDA_ARCH__
  return __hisinf(a);
#else
  return std::isinf(static_cast<float>(a));
#endif
}


static inline bool isnan(__half a)
{
#ifdef __CUDA_ARCH__
  return __hisnan(a);
#else
  return std::isnan(static_cast<float>(a));
#endif
}


static inline __half log(__half a)
{
#ifdef __CUDA_ARCH__
  return hlog(a);
#else
  return ::logf(a);
#endif
}


static inline __half pow(__half x, __half exponent)
{
  return ::powf(x, exponent);
}


static inline __half sin(__half a)
{
#ifdef __CUDA_ARCH__
  return hsin(a);
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


static inline __half sqrt(__half a)
{
#ifdef __CUDA_ARCH__
  return hsqrt(a);
#else
  return ::sqrtf(a);
#endif
}


} // end std

#endif // __HALF_H__

