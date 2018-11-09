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

// nop, but helps IDEs
#include "realm/utils.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

namespace Realm {
    
  ////////////////////////////////////////////////////////////////////////
  //
  // class shortstringbuf<I,E>

  template <size_t I, size_t E>
  inline shortstringbuf<I,E>::shortstringbuf()
    : external_buffer(0)
    , external_buffer_size(0)
  {
    setp(internal_buffer, internal_buffer + INTERNAL_BUFFER_SIZE);
  }

  template <size_t I, size_t E>
  inline shortstringbuf<I,E>::~shortstringbuf()
  {
    if(external_buffer)
      free(external_buffer);
  }

  template <size_t I, size_t E>
  inline const char *shortstringbuf<I,E>::data() const
  {
    return (external_buffer ? external_buffer : internal_buffer);
  }

  template <size_t I, size_t E>
  inline size_t shortstringbuf<I,E>::size() const
  {
    return pptr() - data();
  }

  template <size_t I, size_t E>
  inline typename shortstringbuf<I,E>::int_type shortstringbuf<I,E>::overflow(typename shortstringbuf<I,E>::int_type c)
  {
    size_t curlen;
    if(external_buffer) {
      // grow existing external buffer
      curlen = pptr() - external_buffer;
      external_buffer_size = curlen * 2;
      char *new_buffer = (char *)malloc(external_buffer_size);
      assert(new_buffer != 0);
      memcpy(new_buffer, external_buffer, curlen);
      free(external_buffer);
      external_buffer = new_buffer;
    } else {
      // switch from internal to external
      curlen = pptr() - internal_buffer;
      external_buffer_size = INITIAL_EXTERNAL_BUFFER_SIZE;
      external_buffer = (char *)malloc(external_buffer_size);
      assert(external_buffer != 0);
      memcpy(external_buffer, internal_buffer, curlen);
    }
    if(c >= 0)
      external_buffer[curlen++] = c;
    setp(external_buffer + curlen, external_buffer + external_buffer_size);
    return 0;
  }


}; // namespace Realm
