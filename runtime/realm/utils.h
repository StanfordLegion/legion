/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include <string>
#include <ostream>
#include <vector>
#include <map>
#include <cassert>

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


}; // namespace Realm

#include "utils.inl"

#endif // ifndef REALM_UTILS_H
