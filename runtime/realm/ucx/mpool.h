
/* Copyright 2024 NVIDIA Corporation
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

#ifndef MPOOL_H
#define MPOOL_H

#include <string>
#include <climits>
#include <cstdint>
#include <unordered_map>

/* A chunk looks like this:
 * +----------+-----+-----------+-----+------+---------+-----+
 * Chunk | pad | Elem |   obj   | pad | Elem |   obj   | ... |
 * +----------+-----+-----------+-----+------+---------+-----+
 *                    |<-->|
 *                         This location is aligned
 */

namespace Realm {

namespace UCP {

using chunk_alloc_t   = void* (*)(size_t bytes, void *arg);
using chunk_release_t = void  (*)(void *chunk, void *arg);
using obj_init_t      = void  (*)(void *obj, void *arg);
using obj_cleanup_t   = void  (*)(void *obj, void *arg);

class MPool {
public:
  MPool(std::string name_, bool leak_check_, size_t obj_size_,
      size_t alignment_, size_t alignment_offset_,
      size_t objs_per_chunk_ = 1024, size_t init_num_objs_ = 1024,
      size_t max_objs_ = UINT_MAX, size_t max_chunk_size_ = UINT_MAX,
      double expand_factor_ = 1.5, // new size = expand_factor * current size
      chunk_alloc_t chunk_alloc_ = &MPool::malloc_wrapper,
      void *chunk_alloc_arg_ = nullptr,
      chunk_release_t chunk_release_ = &MPool::free_wrapper,
      void *chunk_release_arg_ = nullptr,
      obj_init_t obj_init_ = nullptr,
      void *obj_init_arg_ = nullptr,
      obj_cleanup_t obj_cleanup_ = nullptr,
      void *obj_cleanup_arg_ = nullptr);

  MPool& operator=(const MPool&) = delete;
  MPool(const MPool&) = delete;
  ~MPool();

  void *get();
  static void put(void *obj);
  bool expand(size_t ext_objs);
  bool has(bool with_expand);

  static void *malloc_wrapper(size_t size, void *arg)
  {
    (void) arg;
    return malloc(size);
  }

  static void free_wrapper(void *chunk, void *arg)
  {
    (void) arg;
    free(chunk);
  }

private:
  union Elem {
    Elem  *next{nullptr}; // used when in freelist
    MPool *mp;            // used when not in freelist
    // obj
  };
  struct Chunk {
    size_t     num_objs;
    void       *elems;
    Chunk *next{nullptr};
    // elements
  };

  bool validate_config();
  void free_list_add(Elem *elem);
  void *chunk_elems(const Chunk *chunk);
  Elem *chunk_ith_elem(const Chunk *chunk, size_t i);

  std::string name;
  bool leak_check;
  size_t obj_size;
  size_t alignment;
  size_t alignment_offset;
  size_t objs_per_chunk;
  size_t init_num_objs;
  size_t max_objs;
  size_t max_chunk_size;
  double expand_factor; // new size = expand_factor * current size
  chunk_alloc_t chunk_alloc;
  void *chunk_alloc_arg;
  chunk_release_t chunk_release;
  void *chunk_release_arg;
  obj_init_t obj_init;
  void *obj_init_arg;
  obj_cleanup_t obj_cleanup;
  void *obj_cleanup_arg;

  size_t num_objs{0};
  size_t num_chunks{0};
  size_t obj_alloc_size;
  Elem  *free_list{nullptr};
  Chunk *chunks{nullptr};
};

class VMPool {
public:
  VMPool(std::string name_, bool leak_check_,
      size_t max_obj_size_, size_t alignment_,
      size_t objs_per_chunk_ = 128, size_t init_num_objs_ = 256,
      size_t max_objs_ = UINT_MAX, size_t max_chunk_size_ = UINT_MAX,
      double expand_factor_ = 1.5, // new size = expand_factor * current size
      chunk_alloc_t chunk_alloc_ = &MPool::malloc_wrapper,
      void *chunk_alloc_arg_ = nullptr,
      chunk_release_t chunk_release_ = &MPool::free_wrapper,
      void *chunk_release_arg_ = nullptr,
      obj_init_t obj_init_ = nullptr,
      void *obj_init_arg_ = nullptr,
      obj_cleanup_t obj_cleanup_ = nullptr,
      void *obj_cleanup_arg_ = nullptr);

  ~VMPool();

  void *get(size_t size);
  static void put(void *buf);
  bool expand(size_t ext_objs);

private:
  struct Elem {
    VMPool *vmp;
    uintptr_t parent_obj;
    // sub-allocation
  };

  static const size_t metadata_size{sizeof(Elem) + alignof(Elem)};
  uintptr_t obj_cached{0};
  uintptr_t addr_cached{0};
  size_t max_obj_size;
  size_t mpool_max_obj_size;
  size_t alignment;
  MPool *mp;
  std::unordered_map<uintptr_t, size_t> objs_map;
};

}; // namespace UCP

}; // namespace Realm

#endif
