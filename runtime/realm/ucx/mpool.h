
/* Copyright 2023 NVIDIA Corporation
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
  struct InitParams {
    size_t obj_size{0};
    size_t alignment{1};
    size_t alignment_offset{0};
    size_t objs_per_chunk{1024};
    size_t init_num_objs{1024};
    size_t max_objs{UINT_MAX};
    size_t max_chunk_size{UINT_MAX};
    double expand_factor{1.5}; // new size = expand_factor * current size
    std::string name;
    bool leak_check{false};
    chunk_alloc_t chunk_alloc{&MPool::malloc_wrapper};
    chunk_release_t chunk_release{&MPool::free_wrapper};
    obj_init_t obj_init{nullptr};
    obj_cleanup_t obj_cleanup{nullptr};
    void *chunk_alloc_arg{nullptr};
    void *chunk_release_arg{nullptr};
    void *obj_init_arg{nullptr};
    void *obj_cleanup_arg{nullptr};
  };
  MPool(const InitParams &init_params);
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

  InitParams init_params;
  size_t num_objs{0};
  size_t num_chunks{0};
  size_t obj_alloc_size;
  Elem  *free_list{nullptr};
  Chunk *chunks{nullptr};
};

class VMPool {
public:
  struct InitParams {
    size_t max_obj_size{0};
    size_t alignment{1};
    size_t objs_per_chunk{128};
    size_t init_num_objs{256};
    size_t max_objs{UINT_MAX};
    size_t max_chunk_size{UINT_MAX};
    double expand_factor{1.5}; // new size = expand_factor * current size
    std::string name;
    bool leak_check{false};
    chunk_alloc_t chunk_alloc{&MPool::malloc_wrapper};
    chunk_release_t chunk_release{&MPool::free_wrapper};
    obj_init_t obj_init{nullptr};
    obj_cleanup_t obj_cleanup{nullptr};
    void *chunk_alloc_arg{nullptr};
    void *chunk_release_arg{nullptr};
    void *obj_init_arg{nullptr};
    void *obj_cleanup_arg{nullptr};
  };

  VMPool(const InitParams &init_params);
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
  InitParams init_params;
  size_t max_obj_size;
  static const size_t metadata_size{sizeof(Elem) + alignof(Elem)};
  uintptr_t obj_cached{0};
  uintptr_t addr_cached{0};
  MPool *mp;
  std::unordered_map<uintptr_t, size_t> objs_map;
};

}; // namespace UCP

}; // namespace Realm

#endif
