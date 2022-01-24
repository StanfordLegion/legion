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

// little helper utilities for Realm code
// none of this is Realm-specific, but it's put in the Realm namespace to
//  reduce the chance of conflicts

#ifndef REALM_UTILS_H
#define REALM_UTILS_H

#include "realm/realm_config.h"

#include <string>
#include <ostream>
#include <vector>
#include <map>
#include <cassert>
#include <sstream>

namespace Realm {
    
  // helpers for deleting contents STL containers of pointers-to-things

  template <typename T>
  void delete_container_contents(std::vector<T *>& v, bool clear_cont = true)
  {
    for(typename std::vector<T *>::iterator it = v.begin();
	it != v.end();
	it++)
      delete (*it);

    if(clear_cont)
      v.clear();
  }

  template <typename K, typename V>
  void delete_container_contents(std::map<K, V *>& m, bool clear_cont = true)
  {
    for(typename std::map<K, V *>::iterator it = m.begin();
	it != m.end();
	it++)
      delete it->second;

    if(clear_cont)
      m.clear();
  }

  template <typename K, typename V>
  void delete_container_contents_free(std::map<K, V *>& m, bool clear_cont = true)
  {
    for(typename std::map<K, V *>::iterator it = m.begin();
	it != m.end();
	it++)
      free(it->second);

    if(clear_cont)
      m.clear();
  }

  // streambuf that holds most messages in an internal buffer
  template <size_t _INTERNAL_BUFFER_SIZE, size_t _INITIAL_EXTERNAL_SIZE>
  class shortstringbuf : public std::streambuf {
  public:
    shortstringbuf();
    ~shortstringbuf();

    const char *data() const;
    size_t size() const;

  protected:
    virtual int_type overflow(int_type c);

    static const size_t INTERNAL_BUFFER_SIZE = _INTERNAL_BUFFER_SIZE;
    static const size_t INITIAL_EXTERNAL_BUFFER_SIZE = _INITIAL_EXTERNAL_SIZE;
    char internal_buffer[INTERNAL_BUFFER_SIZE];
    char *external_buffer;
    size_t external_buffer_size;
  };


  // helper class that lets you build a formatted std::string as a single expression:
  //  /*std::string s =*/ stringbuilder() << ... << ... << ...;

  class stringbuilder {
  public:
    stringbuilder() : os(&strbuf) {}
    operator std::string(void) const { return std::string(strbuf.data(),
							  strbuf.size()); }
    template <typename T>
    stringbuilder& operator<<(T data) { os << data; return *this; }
  protected:
    shortstringbuf<32, 64> strbuf;
    std::ostream os;
  };

  // little helper class that defines a default value for a member variable
  //  in the header rather than in the containing object's constructor
  //  implementation
  template <typename T, T _DEFAULT>
  struct WithDefault {
  public:
    static const T DEFAULT_VALUE = _DEFAULT;

    WithDefault(void) : val(_DEFAULT) {}
    WithDefault(T _val) : val(_val) {}

    operator T(void) const { return val; }
    WithDefault<T,_DEFAULT>& operator=(T newval) { val = newval; return *this; }

    T val;
  };

  // behaves like static_cast, but uses dynamic_cast+assert when DEBUG_REALM
  //  is defined
  template <typename T, typename T2>
  inline T checked_cast(T2 *ptr)
  {
#ifdef DEBUG_REALM
    T result = dynamic_cast<T>(ptr);
    assert(result != 0);
    return result;
#else
    return static_cast<T>(ptr);
#endif
  }


  // a wrapper class that defers construction of the underlying object until
  //  explicitly requested
  template <typename T>
  class DeferredConstructor {
  public:
    DeferredConstructor();
    ~DeferredConstructor();

    // zero and one argument constructors for now
    T *construct();

    template <typename T1>
    T *construct(T1 arg1);

    // object must have already been explicitly constructed to dereference
    T& operator*();
    T *operator->();

    const T& operator*() const;
    const T *operator->() const;

  protected:
    T *ptr;  // needed to avoid type-punning complaints
    typedef char Storage_unaligned[sizeof(T)];
    REALM_ALIGNED_TYPE_SAMEAS(Storage_aligned, Storage_unaligned, T);
    Storage_aligned raw_storage;
  };


  template <unsigned _BITS, unsigned _SHIFT>
  struct bitfield {
    static const unsigned BITS = _BITS;
    static const unsigned SHIFT = _SHIFT;

    template <typename T>
    static T extract(T source);

    template <typename T>
    static T insert(T target, T field);

    template <typename T>
    static T bit_or(T target, T field);
  };

  template <typename T>
  class bitpack {
  public:
    bitpack();  // no initialization
    bitpack(const bitpack<T>& copy_from);
    bitpack(T init_val);

    bitpack<T>& operator=(const bitpack<T>& copy_from);
    bitpack<T>& operator=(T new_val);

    operator T() const;

    template <typename BITFIELD>
    class bitsliceref {
    public:
      bitsliceref(T& _target);

      operator T() const;
      bitsliceref<BITFIELD>& operator=(T field);
      bitsliceref<BITFIELD>& operator|=(T field);

      static const T MAXVAL = ~(~T(0) << BITFIELD::BITS);

    protected:
      T& target;
    };

    template <typename BITFIELD>
    class constbitsliceref {
    public:
      constbitsliceref(const T& _target);

      operator T() const;

      static const T MAXVAL = ~(~T(0) << BITFIELD::BITS);

    protected:
      const T& target;
    };

    template <typename BITFIELD>
    bitsliceref<BITFIELD> slice();
    template <typename BITFIELD>
    constbitsliceref<BITFIELD> slice() const;

    template <typename BITFIELD>
    bitsliceref<BITFIELD> operator[](const BITFIELD& bitfield);
    template <typename BITFIELD>
    constbitsliceref<BITFIELD> operator[](const BITFIELD& bitfield) const;

  protected:
    T value;
  };


  // helpers to pretty-print containers

  template <typename T>
  class PrettyVector {
  public:
    explicit PrettyVector(const T *_data, size_t _size,
			  const char *_delim = ", ",
			  const char *_pfx = "[",
			  const char *_sfx = "]");
    explicit PrettyVector(const std::vector<T>& _v,
			  const char *_delim = ", ",
			  const char *_pfx = "[",
			  const char *_sfx = "]");

    void print(std::ostream& os) const;

  protected:
    const T *data;
    size_t size;
    const char *delim;
    const char *pfx;
    const char *sfx;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const PrettyVector<T>& pv);


  // metaprogramming stuff that's standard in c++11 and beyond
#if REALM_CXX_STANDARD >= 11
  using std::is_integral;
  using std::enable_if;
  using std::make_signed;
  using std::make_unsigned;
  using std::remove_const;
#else
  // roll our own...
  template <typename T> struct is_integral { static const bool value = false; };
  template <typename T> struct make_signed;
  template <typename T> struct make_unsigned;
  // we need to cover both const and non-const, so some macro helpers
#define IS_INTEGRAL(T) \
  template <> struct is_integral<T> { static const bool value = true; }; \
  template <> struct is_integral<const T> { static const bool value = true; }
#define IS_SIGNED_UNSIGNED_PAIR(T1, T2) \
  IS_INTEGRAL(T1); \
  IS_INTEGRAL(T2); \
  template <> struct make_signed<T1> { typedef T1 type; }; \
  template <> struct make_signed<T2> { typedef T1 type; }; \
  template <> struct make_unsigned<T1> { typedef T2 type; }; \
  template <> struct make_unsigned<T2> { typedef T2 type; };

  IS_INTEGRAL(bool);
  IS_SIGNED_UNSIGNED_PAIR(char, unsigned char);
  IS_SIGNED_UNSIGNED_PAIR(short, unsigned short);
  IS_SIGNED_UNSIGNED_PAIR(int, unsigned int);
  IS_SIGNED_UNSIGNED_PAIR(long, unsigned long);
  IS_SIGNED_UNSIGNED_PAIR(long long, unsigned long long);
#undef IS_INTEGRAL
#undef IS_SIGNED_UNSIGNED_PAIR

  template <bool B, typename T> struct enable_if {};
  template <typename T> struct enable_if<true, T> { typedef T type; };

  template< class T > struct remove_const          { typedef T type; };
  template< class T > struct remove_const<const T> { typedef T type; };
#endif

  // TODO: get this from <variant> for c++17 and up?
  struct monostate {};

  // TODO: actually use C++20 version if available
  const size_t dynamic_extent = size_t(-1);

  template <typename T, size_t Extent = dynamic_extent> class span;

  template <typename T>
  class span<T, dynamic_extent> {
  public:
    typedef typename remove_const<T>::type value_type;
    static const size_t extent = dynamic_extent;

    span() : base(0), length(0) {}
    span(T *_base, size_t _length) : base(_base), length(_length) {}

    // from another span
    template <size_t Extent2>
    span(span<T, Extent2> copy_from)
      : base(copy_from.data()), length(copy_from.size()) {}

    // from a vector
    span(const std::vector<typename remove_const<T>::type>& v)
      : base(v.data()), length(v.size()) {}

    // from a scalar
    span(T& v)
      : base(&v), length(1) {}

    T& operator[](size_t idx) const { return base[idx]; }

    T *data() const { return base; }
    size_t size() const { return length; }
    bool empty() const { return (length == 0); }

  protected:
    T *base;
    size_t length;
  };

  class empty_span {
  public:
    template <typename T>
    operator span<T, dynamic_extent>() const { return span<T, dynamic_extent>(); }
  };

  template <typename T>
  span<T, dynamic_extent> make_span(T *base, size_t length)
  {
    return span<T, dynamic_extent>(base, length);
  }

  // accumulates a crc32c checksum
  //   initialization (traditionally to 0xFFFFFFFF) and finalization (by
  //   inverting) the accumulator is left to the caller
  REALM_PUBLIC_API uint32_t crc32c_accumulate(uint32_t accum_in, const void *data, size_t len);

  // helper class to make something non-copy/moveable
  class noncopyable {
  protected:
    noncopyable() {}
    ~noncopyable() {}
  private:
#if REALM_CXX_STANDARD >= 11
    noncopyable(const noncopyable&) = delete;
    noncopyable(noncopyable&&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;
    noncopyable& operator=(noncopyable&&) = delete;
#else
    // pre C++-11 you have to define them, but they're private so can't be
    //  called
    noncopyable(const noncopyable&) {}
    noncopyable& operator=(const noncopyable&) { return *this; }
#endif
  };

  // explicitly calls the destructor on an object, working around issues
  //  with some compilers and typedefs
  template <typename T>
  void call_destructor(T *obj)
  {
    obj->~T();
  }

}; // namespace Realm

#include "realm/utils.inl"

#endif // ifndef REALM_UTILS_H
