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

#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#ifndef __CUDA_HD__
#if defined (__CUDACC__) || defined (__HIPCC__)
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#include <cmath>
#if defined (LEGION_USE_CUDA)
#if __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2
#error "No complex number support for GPUs due to a Thrust bug in CUDA 9.2"
#elif __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 0
#error "No complex number support for GPUs due to a Thrust bug in CUDA 10.0"
#else
#include <thrust/complex.h>
#define COMPLEX_NAMESPACE thrust
#endif
#elif defined(LEGION_USE_HIP)
#include <thrust/complex.h>
#define COMPLEX_NAMESPACE thrust
#else
#include <complex>
#define COMPLEX_NAMESPACE std
#endif 
#ifdef COMPLEX_HALF
#include "mathtypes/half.h"
#endif

using COMPLEX_NAMESPACE::complex;

#if defined (LEGION_USE_CUDA) || defined (LEGION_USE_HIP)
// We need fabs for situations where we process complex, floating point, and
// integral types in the same generic call. This is only needed for the thrust
// version of complex as the std one already has fabs.
// Overload for __half defined after complex<__half>
namespace COMPLEX_NAMESPACE {
template<class T>
__CUDA_HD__ constexpr T fabs(const complex<T>& arg) {
  return abs(arg);
}
}
#endif

// Need to put this in COMPLEX_NAMESPACE namespace for ADL. The namespace has to
// be changed/removed if another implementation of complex is used
namespace COMPLEX_NAMESPACE {
template<typename T> __CUDA_HD__ constexpr
inline bool operator<(const complex<T>& c1, const complex<T>& c2) {
    return (c1.real() < c2.real()) || 
      (!(c2.real() < c1.real()) && (c1.imag() < c2.imag()));
}

template<typename T> __CUDA_HD__ constexpr
inline bool operator>(const complex<T>& c1, const complex<T>& c2) {
    return (c1.real() > c2.real()) || 
      (!(c2.real() > c1.real()) && (c1.imag() > c2.imag()));
}

template<typename T> __CUDA_HD__ constexpr 
inline bool operator<=(const complex<T>& c1, const complex<T>& c2) {
    return (c1 == c2) || (c1.real() < c2.real()) || 
      (!(c2.real() < c1.real()) && (c1.imag() < c2.imag()));
}

template<typename T> __CUDA_HD__ constexpr 
inline bool operator>=(const complex<T>& c1, const complex<T>& c2) {
    return (c1 == c2) || (c1.real() > c2.real()) || 
      (!(c2.real() > c1.real()) && (c1.imag() > c2.imag()));
}

} // namespace COMPLEX_NAMESPACE

#ifdef COMPLEX_HALF
template<>
class COMPLEX_NAMESPACE::complex<__half> {
public:
  __CUDA_HD__
  complex(void) { } // empty default constructor for CUDA
  __CUDA_HD__
  complex(__half re, __half im = __half()) : _real(re), _imag(im) { }
  complex(const complex<__half> &rhs) = default;
#if defined (__CUDACC__) || defined (__HIPCC__)
  __device__ // Device only constructor
  complex(__half2 val) : _real(val.x), _imag(val.y) { }
#endif
public:
#if defined (__CUDACC__) || defined (__HIPCC__)
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
  inline complex<__half>& operator=(const complex<__half> &rhs) = default;
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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return hypotf(z.real(), z.imag());
#elif __cplusplus >= 201103L
  return (__half)(std::hypotf(z.real(), z.imag()));
#else
  return (__half)(std::sqrt(z.real() * z.real() + z.imag() * z.imag()));
#endif
}

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

#endif // complex_H__ 

