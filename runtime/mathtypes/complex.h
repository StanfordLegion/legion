/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#if __cplusplus >= 201103L
#define __CEXPR__ constexpr
#else
#define __CEXPR__
#endif

#include <cmath>
#include <thrust/complex.h> 
#ifdef COMPLEX_HALF
#include "mathtypes/half.h"
#endif

template<typename T>
using complex = thrust::complex<T>;

// We need fabs for situations where we process complex, floating point, and
// integral types in the same generic call
// Overload for __half defined after complex<__half>
template<class T>
T fabs(const complex<T>& arg) {
  return abs(arg);
}

template <typename T>
struct convert_complex
{
};

template <>
struct convert_complex<double>
{
  __CUDA_HD__ static inline complex<double> from_int(unsigned long long val)
  {
    union
    {
      unsigned long long as_long;
      double array[2];
    } convert = {0};
    convert.as_long = val;
    complex<double> retval(convert.array[0], convert.array[1]);
    return retval;
  }
  // cast back to integer
  __CUDA_HD__ inline static unsigned long long as_int(complex<double> c)
  {
    union
    {
      unsigned long long as_long;
      double array[2];
    } convert = {0};
    convert.array[0] = c.real();
    convert.array[1] = c.imag();
    return convert.as_long;
  }
};

template <>
struct convert_complex<float>
{
  __CUDA_HD__ static inline complex<float> from_int(unsigned long long val)
  {
    union
    {
      unsigned long long as_long;
      float array[2];
    } convert = {0};
    convert.as_long = val;
    complex<float> retval(convert.array[0], convert.array[1]);
    return retval;
  }
  // cast back to integer
  __CUDA_HD__ inline static unsigned long long as_int(complex<float> c)
  {
    union
    {
      unsigned long long as_long;
      float array[2];
    } convert = {0};
    convert.array[0] = c.real();
    convert.array[1] = c.imag();
    return convert.as_long;
  }
};

// Need to put this in thrust namespace for ADL. The namespace has to be
// changed/removed if another implementation of complex is used
namespace thrust {
template<typename T> __CUDA_HD__ __CEXPR__
inline bool operator<(const complex<T>& c1, const complex<T>& c2) {
    return (c1.real() < c2.real()) || 
      (!(c2.real() < c1.real()) && (c1.imag() < c2.imag()));
}

template<typename T> __CUDA_HD__ __CEXPR__
inline bool operator>(const complex<T>& c1, const complex<T>& c2) {
    return (c1.real() > c2.real()) || 
      (!(c2.real() > c1.real()) && (c1.imag() > c2.imag()));
}

template<typename T> __CUDA_HD__ __CEXPR__
inline bool operator<=(const complex<T>& c1, const complex<T>& c2) {
    return (c1 == c2) || (c1.real() < c2.real()) || 
      (!(c2.real() < c1.real()) && (c1.imag() < c2.imag()));
}

template<typename T> __CUDA_HD__ __CEXPR__
inline bool operator>=(const complex<T>& c1, const complex<T>& c2) {
    return (c1 == c2) || (c1.real() > c2.real()) || 
      (!(c2.real() > c1.real()) && (c1.imag() > c2.imag()));
}
} // namespace thrust

#ifdef COMPLEX_HALF
template<>
class thrust::complex<__half> {
public:
  __CUDA_HD__
  complex(void) { } // empty default constructor for CUDA
  __CUDA_HD__
  complex(__half re, __half im = __half()) : _real(re), _imag(im) { }
  __CUDA_HD__
  complex(const complex<__half> &rhs) : _real(rhs.real()), _imag(rhs.imag()) { }
#ifdef __CUDACC__
  __device__ // Device only constructor
  complex(__half2 val) : _real(val.x), _imag(val.y) { }
#endif
public:
#ifdef __CUDACC__
  __device__
  operator __half2(void) const
  {
    __half2 result;
    result.x = _real;
    result.y = _imag;
    return result;
  }
#endif
public:
  __CUDA_HD__
  inline complex<__half>& operator=(const complex<__half> &rhs)
    {
      _real = rhs.real();
      _imag = rhs.imag();
      return *this;
    }
public:
  __CUDA_HD__
  inline __half real(void) const { return _real; }
  __CUDA_HD__
  inline __half imag(void) const { return _imag; }
public:
  __CUDA_HD__
  inline complex<__half>& operator+=(const complex<__half> &rhs)
    {
      _real = _real + rhs.real();
      _imag = _imag + rhs.imag();
      return *this;
    }
  __CUDA_HD__
  inline complex<__half>& operator-=(const complex<__half> &rhs)
    {
      _real = _real + rhs.real();
      _imag = _imag + rhs.imag();
      return *this;
    }
  __CUDA_HD__
  inline complex<__half>& operator*=(const complex<__half> &rhs)
    {
      const __half new_real = _real * rhs.real() - _imag * rhs.imag();
      const __half new_imag = _imag * rhs.real() + _real * rhs.imag();
      _real = new_real;
      _imag = new_imag;
      return *this;
    }
  __CUDA_HD__
  inline complex<__half>& operator/=(const complex<__half> &rhs)
    {
      // Note the plus because of conjugation
      const __half num_real = _real * rhs.real() + _imag * rhs.imag();       
      // Note the minus because of conjugation
      const __half num_imag = _imag * rhs.real() - _real * rhs.imag();
      const __half denom = rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
      _real = num_real / denom;
      _imag = num_imag / denom;
      return *this;
    }
protected:
  __half _real;
  __half _imag;
};

__CUDA_HD__
inline __half abs(const complex<__half>& z) {
#ifdef __CUDA_ARCH__
  return hypotf(z.real(), z.imag());
#elif __cplusplus >= 201103L
  return (__half)(std::hypotf(z.real(), z.imag()));
#else
  return (__half)(std::sqrt(z.real() * z.real() + z.imag() * z.imag()));
#endif
}

template <>
struct convert_complex<__half>
{
  __CUDA_HD__
  static inline complex<__half> from_int(int val)
  {
    union
    {
      int as_int;
      unsigned short array[2];
    } convert;
    convert.as_int = val;
    complex<__half> retval(
#ifdef __CUDA_ARCH__
    __short_as_half(convert.array[0]),
    __short_as_half(convert.array[1])
#else
    *(reinterpret_cast<const __half *>(&convert.array[0])),
    *(reinterpret_cast<const __half *>(&convert.array[1]))
#endif
    );
    return retval;
  }
  // cast back to integer
  __CUDA_HD__
  inline static int as_int(complex<__half> c)
  {
    union
    {
      int as_int;
      unsigned short array[2];
    } convert;
#ifdef __CUDA_ARCH__
    convert.array[0] = __half_as_short(c.real());
    convert.array[1] = __half_as_short(c.imag());
#else
    const __half real = c.real(), imag = c.imag();
    convert.array[0] = *(reinterpret_cast<const unsigned short *>(&real));
    convert.array[1] = *(reinterpret_cast<const unsigned short *>(&imag));
#endif
    return convert.as_int;
  }
};

__CUDA_HD__
inline complex<__half> operator+(const complex<__half> &one, const complex<__half> &two)
{
  return complex<__half>(one.real() + two.real(), one.imag() + two.imag());
}

__CUDA_HD__
inline complex<__half> operator-(const complex<__half> &one, const complex<__half> &two)
{
  return complex<__half>(one.real() - two.real(), one.imag() - two.imag());
}

__CUDA_HD__
inline complex<__half> operator*(const complex<__half> &one, const complex<__half> &two)
{
  return complex<__half>(one.real() * two.real() - one.imag() * two.imag(),
                    one.imag() * two.real() + one.real() * two.imag());
}

__CUDA_HD__
inline complex<__half> operator/(const complex<__half> &one, const complex<__half> &two)
{
  // Note the plus because of conjugation
  const __half num_real = one.real() * two.real() + one.imag() * two.imag();       
  // Note the minus because of conjugation
  const __half num_imag = one.imag() * two.real() - one.real() * two.imag();
  const __half denom = two.real() * two.real() + two.imag() * two.imag();
  return complex<__half>(num_real / denom, num_imag / denom);
}

#endif // COMPLEX_HALF

#undef __CEXPR__

#endif // complex_H__ 

