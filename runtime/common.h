/* Copyright 2014 Stanford University
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


#ifndef COMMON_H
#define COMMON_H

// This file contains declarations of objects
// that need to be globally visible to all layers
// of the program including the application code
// as well as both of the runtimes.

// At this point most of the code has been switched over to untyped pointers
#ifndef TYPED_POINTERS
struct ptr_t
{ 
public:
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(void) : value(0) { }
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(const ptr_t &p) : value(p.value) { }
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(unsigned v) : value(v) { }
public:
  unsigned value; 
public: 
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t& operator=(const ptr_t &ptr) { value = ptr.value; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator==(const ptr_t &ptr) const { return (ptr.value == this->value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator!=(const ptr_t &ptr) const { return (ptr.value != this->value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator< (const ptr_t &ptr) const { return (ptr.value <  this->value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline operator bool(void) const { return (value != (unsigned)-1); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline bool operator!(void) const { return (value == (unsigned)-1); }

  // Addition operation on pointers
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator+(const ptr_t &ptr) const { return ptr_t(value + ptr.value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator+(unsigned offset) const { return ptr_t(value + offset); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator+(int offset) const { return ptr_t(value + offset); }

  // Subtraction operation on pointers
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator-(const ptr_t &ptr) const { return ptr_t(value - ptr.value); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator-(unsigned offset) const { return ptr_t(value - offset); }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator-(int offset) const { return ptr_t(value - offset); }
  
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t& operator++(void) { value++; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator++(int) { value++; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t& operator--(void) { value--; return *this; }
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline ptr_t operator--(int) { value--; return *this; }

  // Thank you Eric for type cast operators!
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline operator unsigned(void) const { return value; } 
#ifdef __CUDACC__
  __host__ __device__ 
#endif
  inline operator int(void) const { return int(value); }

#ifdef __CUDACC__
  __host__ __device__
#endif
  inline bool is_null(void) const { return (value == ((unsigned)-1)); }

#ifdef __CUDACC__
  __host__ __device__ 
#endif
  static inline ptr_t nil(void) { ptr_t p; p.value = (unsigned)-1; return p; }
};

#else // TYPED_POINTERS

// Forware declaration
//template<typename T> struct ptr_t;

struct utptr_t
{ 
public:
  utptr_t(void) : value(0) { }
  utptr_t(const utptr_t &p) : value(p.value) { }
  utptr_t(unsigned v) : value(v) { }
public:
  unsigned value; 
public: 
#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  inline utptr_t& operator=(const utptr_t &ptr) { value = ptr.value; return *this; }

  template<typename T>
#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  inline utptr_t& operator=(const ptr_t<T> &ptr) { value = ptr.value; return *this; }

  inline bool operator==(const utptr_t &ptr) const { return (ptr.value == this->value); }
  inline bool operator!=(const utptr_t &ptr) const { return (ptr.value != this->value); }
  inline bool operator< (const utptr_t &ptr) const { return (ptr.value <  this->value); }
  inline operator bool(void) const { return (value != (unsigned)-1); }
  inline bool operator!(void) const { return (value == (unsigned)-1); }

  // Addition operation on pointers
  inline utptr_t operator+(const utptr_t &ptr) const { return utptr_t(value + ptr.value); }
  template<typename T>
  inline utptr_t operator+(const ptr_t<T> &ptr) const { return utptr_t(value + ptr.value); }
  inline utptr_t operator+(unsigned offset) const { return utptr_t(value + offset); }
  inline utptr_t operator+(int offset) const { return utptr_t(value + offset); }

  // Subtraction operation on pointers
  inline utptr_t operator-(const utptr_t &ptr) const { return utptr_t(value - ptr.value); }
  template<typename T>
  inline utptr_t operator-(const ptr_t<T> &ptr) const { return utptr_t(value - ptr.value); }
  inline utptr_t operator-(unsigned offset) const { return utptr_t(value - offset); }
  inline utptr_t operator-(int offset) const { return utptr_t(value - offset); }
  
  inline utptr_t& operator++(void) { value++; return *this; }
  inline utptr_t operator++(int) { value++; return *this; }
  inline utptr_t& operator--(void) { value--; return *this; }
  inline utptr_t operator--(int) { value--; return *this; }

  inline operator unsigned(void) const { return value; } 
  inline operator int(void) const { return int(value); }

  static inline utptr_t nil(void) { utptr_t p; p.value = (unsigned)-1; return p; }
};

template<typename T>
struct ptr_t 
{ 
public:
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(void) : value(0) { }
  explicit ptr_t(const utptr_t &p) : value(p.value) { }
#ifdef __CUDACC__
  __host__ __device__
#endif
  ptr_t(const ptr_t<T> &p) : value(p.value) { }
  ptr_t(unsigned v) : value(v) { }
public:
  unsigned value; 
public:
#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  inline ptr_t<T>& operator=(const ptr_t<T> &ptr) { value = ptr.value; return *this; }
#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  inline ptr_t<T>& operator=(const utptr_t &ptr)  { value = ptr.value; return *this; }
  inline bool operator==(const ptr_t<T> &ptr) const { return (ptr.value == this->value); }
  inline bool operator!=(const ptr_t<T> &ptr) const { return (ptr.value != this->value); }
  inline bool operator< (const ptr_t<T> &ptr) const { return (ptr.value <  this->value); }
  inline operator bool(void) const { return (value != (unsigned)-1); }
  inline bool operator!(void) const { return (value == (unsigned)-1); }
  inline operator utptr_t(void) const { utptr_t ptr; ptr.value = value; return ptr; }

  // Addition operation on pointers
  inline ptr_t<T> operator+(const ptr_t<T> &ptr) const { return ptr_t<T>(value + ptr.value); }
  inline ptr_t<T> operator+(const utptr_t &ptr) const { return ptr_t<T>(value + ptr.value); }
  inline ptr_t<T> operator+(unsigned offset) const { return ptr_t<T>(value + offset); }
  inline ptr_t<T> operator+(int offset) const { return ptr_t<T>(value + offset); }

  // Subtraction operation on pointers
  inline ptr_t<T> operator-(const ptr_t<T> &ptr) const { return ptr_t<T>(value - ptr.value); }
  inline ptr_t<T> operator-(const utptr_t &ptr) const { return ptr_t<T>(value - ptr.value); }
  inline ptr_t<T> operator-(unsigned offset) const { return ptr_t<T>(value - offset); }
  inline ptr_t<T> operator-(int offset) const { return ptr_t<T>(value - offset); }

  inline ptr_t<T>& operator++(void) { value++; return *this; }
  inline ptr_t<T> operator++(int) { value++; return *this; }
  inline ptr_t<T>& operator--(void) { value--; return *this; }
  inline ptr_t<T> operator--(int) { value--; return *this; }

  inline operator unsigned(void) const { return value; } 
  inline operator int(void) const { return int(value); }

  static inline ptr_t<T> nil(void) { ptr_t<T> p; p.value = (unsigned)-1; return p; }
};
#endif // TYPED_POINTERS

#endif // COMMON_H
