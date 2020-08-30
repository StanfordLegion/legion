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

#ifdef COMPLEX_HALF
#include "mathtypes/half.h"
#endif

// Empty base version
// We support half and float versions for now
// TODO: double support
template<typename T>
class complex { };

#ifdef COMPLEX_HALF
template<>
class complex<__half> {
public:
  __CUDA_HD__
  complex(void) { } // empty default constructor for CUDA
  __CUDA_HD__
  complex(const __half val[2]) : _real(val[0]), _imag(val[1]) { }
  __CUDA_HD__
  complex(__half re, __half im) : _real(re), _imag(im) { }
  __CUDA_HD__
  complex(const complex<__half> &rhs) : _real(rhs.real()), _imag(rhs.imag()) { }
#ifdef __CUDACC__
  __device__ // Device only constructor
  complex(__half2 val) : _real(val.x), _imag(val.y) { }
#endif
public:
  // reinterpret cast from integer
  __CUDA_HD__ 
  static inline complex<__half> from_int(int val) 
    {
      union { int as_int; unsigned short array[2]; } convert;
      convert.as_int = val;
      complex<__half> retval;
#ifdef __CUDA_ARCH__
      retval._real = __short_as_half(convert.array[0]);
      retval._imag = __short_as_half(convert.array[1]);
#else
      retval._real = *(reinterpret_cast<const __half*>(&convert.array[0]));
      retval._imag = *(reinterpret_cast<const __half*>(&convert.array[1]));
#endif
      return retval;
    }
  // cast back to integer
  __CUDA_HD__
  inline int as_int(void) const
    {
      union { int as_int; unsigned short array[2]; } convert;
#ifdef __CUDA_ARCH__
      convert.array[0] = __half_as_short(_real);
      convert.array[1] = __half_as_short(_imag);
#else
      convert.array[0] = *(reinterpret_cast<const unsigned short*>(&_real));
      convert.array[1] = *(reinterpret_cast<const unsigned short*>(&_imag));
#endif
      return convert.as_int;
    }
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
#endif

template<>
class complex<float> {
public:
  __CUDA_HD__
  complex(void) { } // empty default constructor for CUDA
  __CUDA_HD__
  complex(const float val[2]) : _real(val[0]), _imag(val[1]) { }
  __CUDA_HD__
  complex(float re, float im) : _real(re), _imag(im) { }
  __CUDA_HD__
  complex(const complex<float> &rhs) : _real(rhs.real()), _imag(rhs.imag()) { }
#ifdef __CUDACC__
  __device__ // Device only constructor
  complex(float2 val) : _real(val.x), _imag(val.y) { }
#endif
public:
  // reinterpret cast from integer
  __CUDA_HD__ 
  static inline complex<float> from_int(unsigned long long val)
    {
      union { unsigned long long as_long; float array[2]; } convert;
      convert.as_long = val;
      complex<float> retval;
      retval._real = convert.array[0];
      retval._imag = convert.array[1];
      return retval;
    }
  // cast back to integer
  __CUDA_HD__
  inline unsigned long long as_int(void) const
    {
      union { unsigned long long as_long; float array[2]; } convert;
      convert.array[0] = _real;
      convert.array[1] = _imag;
      return convert.as_long;
    }
#ifdef __CUDACC__
  __device__
  operator float2(void) const
  {
    float2 result;
    result.x = _real;
    result.y = _imag;
    return result;
  }
#endif
public:
  __CUDA_HD__
  inline complex<float>& operator=(const complex<float> &rhs)
    {
      _real = rhs.real();
      _imag = rhs.imag();
      return *this;
    }
public:
  __CUDA_HD__
  inline float real(void) const { return _real; }
  __CUDA_HD__
  inline float imag(void) const { return _imag; }
public:
  __CUDA_HD__
  inline complex<float>& operator+=(const complex<float> &rhs)
    {
      _real += rhs.real();
      _imag += rhs.imag();
      return *this;
    }
  __CUDA_HD__
  inline complex<float>& operator-=(const complex<float> &rhs)
    {
      _real -= rhs.real();
      _imag -= rhs.imag();
      return *this;
    }
  __CUDA_HD__
  inline complex<float>& operator*=(const complex<float> &rhs)
    {
      const float new_real = _real * rhs.real() - _imag * rhs.imag();
      const float new_imag = _imag * rhs.real() + _real * rhs.imag();
      _real = new_real;
      _imag = new_imag;
      return *this;
    }
  __CUDA_HD__
  inline complex<float>& operator/=(const complex<float> &rhs)
    {
      // Note the plus because of conjugation
      const float num_real = _real * rhs.real() + _imag * rhs.imag();       
      // Note the minus because of conjugation
      const float num_imag = _imag * rhs.real() - _real * rhs.imag();
      const float denom = rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
      _real = num_real / denom;
      _imag = num_imag / denom;
      return *this;
    }
protected:
  float _real;
  float _imag;
};

template<>
class complex<double> {
public:
  __CUDA_HD__
  complex(void) { } // empty default constructor for CUDA
  __CUDA_HD__
  complex(const double val[2]) : _real(val[0]), _imag(val[1]) { }
  __CUDA_HD__
  complex(double re, double im) : _real(re), _imag(im) { }
  __CUDA_HD__
  complex(const complex<double> &rhs) : _real(rhs.real()), _imag(rhs.imag()) { }
#ifdef __CUDACC__
  __device__ // Device only constructor
  complex(double2 val) : _real(val.x), _imag(val.y) { }
#endif
public:
  // reinterpret cast from integer
  __CUDA_HD__ 
  static inline complex<double> from_int(unsigned long long val)
    {
      union { unsigned long long as_long; double array[2]; } convert;
      convert.as_long = val;
      complex<double> retval;
      retval._real = convert.array[0];
      retval._imag = convert.array[1];
      return retval;
    }
  // cast back to integer
  __CUDA_HD__
  inline unsigned long long as_int(void) const
    {
      union { unsigned long long as_long; double array[2]; } convert;
      convert.array[0] = _real;
      convert.array[1] = _imag;
      return convert.as_long;
    }
#ifdef __CUDACC__
  __device__
  operator double2(void) const
  {
    double2 result;
    result.x = _real;
    result.y = _imag;
    return result;
  }
#endif
public:
  __CUDA_HD__
  inline complex<double>& operator=(const complex<double> &rhs)
    {
      _real = rhs.real();
      _imag = rhs.imag();
      return *this;
    }
public:
  __CUDA_HD__
  inline double real(void) const { return _real; }
  __CUDA_HD__
  inline double imag(void) const { return _imag; }
public:
  __CUDA_HD__
  inline complex<double>& operator+=(const complex<double> &rhs)
    {
      _real += rhs.real();
      _imag += rhs.imag();
      return *this;
    }
  __CUDA_HD__
  inline complex<double>& operator-=(const complex<double> &rhs)
    {
      _real -= rhs.real();
      _imag -= rhs.imag();
      return *this;
    }
  __CUDA_HD__
  inline complex<double>& operator*=(const complex<double> &rhs)
    {
      const double new_real = _real * rhs.real() - _imag * rhs.imag();
      const double new_imag = _imag * rhs.real() + _real * rhs.imag();
      _real = new_real;
      _imag = new_imag;
      return *this;
    }
  __CUDA_HD__
  inline complex<double>& operator/=(const complex<double> &rhs)
    {
      // Note the plus because of conjugation
      const double num_real = _real * rhs.real() + _imag * rhs.imag();
      // Note the minus because of conjugation
      const double num_imag = _imag * rhs.real() - _real * rhs.imag();
      const double denom = rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
      _real = num_real / denom;
      _imag = num_imag / denom;
      return *this;
    }
protected:
  double _real;
  double _imag;
};

// These methods work on all types regardless of instatiations
template<typename T> __CUDA_HD__
inline complex<T> operator+(const complex<T> &rhs)
{
  return complex<T>(+rhs.real(), +rhs.imag());
}

template<typename T> __CUDA_HD__
inline complex<T> operator-(const complex<T> &rhs)
{
  return complex<T>(-rhs.real(), -rhs.imag());
}

template<typename T> __CUDA_HD__
inline complex<T> operator+(const complex<T> &one, const complex<T> &two)
{
  return complex<T>(one.real() + two.real(), one.imag() + two.imag());
}

template<typename T> __CUDA_HD__
inline complex<T> operator-(const complex<T> &one, const complex<T> &two)
{
  return complex<T>(one.real() - two.real(), one.imag() - two.imag());
}

template<typename T> __CUDA_HD__
inline complex<T> operator*(const complex<T> &one, const complex<T> &two)
{
  return complex<T>(one.real() * two.real() - one.imag() * two.imag(),
                    one.imag() * two.real() + one.real() * two.imag());
}

template<typename T> __CUDA_HD__
inline complex<T> operator/(const complex<T> &one, const complex<T> &two)
{
  // Note the plus because of conjugation
  const T num_real = one.real() * two.real() + one.imag() * two.imag();       
  // Note the minus because of conjugation
  const T num_imag = one.imag() * two.real() - one.real() * two.imag();
  const T denom = two.real() * two.real() + two.imag() * two.imag();
  return complex<T>(num_real / denom, num_imag / denom);
}

template<typename T> __CUDA_HD__
inline bool operator==(const complex<T> &one, const complex<T> &two)
{
  return (one.real() == two.real()) && (one.imag() == two.imag());
}

template<typename T> __CUDA_HD__
inline bool operator!=(const complex<T> &one, const complex<T> &two)
{
  return (one.real() != two.real()) || (one.imag() != two.imag());
}

template<typename T> __CUDA_HD__
inline bool operator<(const complex<T>& c1, const complex<T>& c2) {
    return (c1.real() < c2.real()) || 
      (!(c2.real() < c1.real()) && (c1.imag() < c2.imag()));
}

template<typename T> __CUDA_HD__
inline bool operator>(const complex<T>& c1, const complex<T>& c2) {
    return (c1.real() > c2.real()) || 
      (!(c2.real() > c1.real()) && (c1.imag() > c2.imag()));
}

template<typename T> __CUDA_HD__
inline bool operator<=(const complex<T>& c1, const complex<T>& c2) {
    return (c1 == c2) || (c1.real() < c2.real()) || 
      (!(c2.real() < c1.real()) && (c1.imag() < c2.imag()));
}

template<typename T> __CUDA_HD__
inline bool operator>=(const complex<T>& c1, const complex<T>& c2) {
    return (c1 == c2) || (c1.real() > c2.real()) || 
      (!(c2.real() > c1.real()) && (c1.imag() > c2.imag()));
}

// TODO: fill this out with full support for std::complex

#endif // complex_H__ 

