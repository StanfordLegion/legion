/* Copyright 2015 Stanford University
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

#include "legion_terra_partitions.h"

#include <unordered_map>

#include "arrays.h"
#include "legion.h"
#include "legion_c_util.h"
#include "legion_utilities.h"
#include "lowlevel.h"
#include "utilities.h"

using namespace LegionRuntime;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::LowLevel;

#ifndef USE_TLS
// Mac OS X and GCC <= 4.6 do not support C++11 thread_local.
#if defined(__MACH__) || (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6)))
#define USE_TLS 0
#else
#define USE_TLS 1
#endif
#endif

struct CachedIndexIterator {
public:
  CachedIndexIterator(IndexSpace is, bool gl) : space(is), index(0), cached(false), global(gl) {}
  bool has_next() {
    if (!cached) cache();
    return index < spans.size();
  }
  ptr_t next_span(size_t *count) {
    if (!cached) cache();
    if (index >= spans.size()) {
      *count = 0;
      return ptr_t::nil();
    }
    std::pair<ptr_t, size_t> span = spans[index++];
    *count = span.second;
    return span.first;
  }
  void reset() {
    index = 0;
  }
private:
  void cache() {
    assert(!cached);

    if (global) {
#if !USE_TLS
      AutoLock guard(global_lock);
#endif
      std::unordered_map<IndexSpace::id_t, std::vector<std::pair<ptr_t, size_t> > >::iterator it =
        global_cache.find(space.id);
      if (it != global_cache.end()) {
        spans = it->second;
        cached = true;
        return;
      }
    }

    IndexIterator it(space);
    while (it.has_next()) {
      size_t count = 0;
      ptr_t start = it.next_span(count);
      assert(count && !start.is_null());
      spans.push_back(std::pair<ptr_t, size_t>(start, count));
    }

    if (global) {
#if USE_TLS
      global_cache[space.id] = spans;
#else
      AutoLock guard(global_lock);
      if (!global_cache.count(space.id)) {
        global_cache[space.id] = spans;
      }
#endif
    }

    cached = true;
  }
private:
  IndexSpace space;
  std::vector<std::pair<ptr_t, size_t> > spans;
  size_t index;
  bool cached;
  bool global;
private:
#if USE_TLS
  static thread_local
  std::unordered_map<IndexSpace::id_t, std::vector<std::pair<ptr_t, size_t> > > global_cache;
#else
  static std::unordered_map<IndexSpace::id_t, std::vector<std::pair<ptr_t, size_t> > > global_cache;
  static ImmovableLock global_lock;
#endif
};

#if USE_TLS
thread_local std::unordered_map<IndexSpace::id_t, std::vector<std::pair<ptr_t, size_t> > >
  CachedIndexIterator::global_cache;
#else
std::unordered_map<IndexSpace::id_t, std::vector<std::pair<ptr_t, size_t> > >
  CachedIndexIterator::global_cache;
ImmovableLock CachedIndexIterator::global_lock(true);
#endif

class TerraCObjectWrapper {
public:

#define NEW_OPAQUE_WRAPPER(T_, T)                                       \
      static T_ wrap(T t) {                                             \
        T_ t_;                                                          \
        t_.impl = static_cast<void *>(t);                               \
        return t_;                                                      \
      }                                                                 \
      static const T_ wrap_const(const T t) {                           \
        T_ t_;                                                          \
        t_.impl = const_cast<void *>(static_cast<const void *>(t));     \
        return t_;                                                      \
      }                                                                 \
      static T unwrap(T_ t_) {                                          \
        return static_cast<T>(t_.impl);                                 \
      }                                                                 \
      static const T unwrap_const(const T_ t_) {                        \
        return static_cast<const T>(t_.impl);                           \
      }

  NEW_OPAQUE_WRAPPER(legion_terra_cached_index_iterator_t, CachedIndexIterator *);
#undef NEW_OPAQUE_WRAPPER
};

legion_terra_index_cross_product_t
legion_terra_index_cross_product_create(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_index_partition_t lhs,
                                        legion_index_partition_t rhs)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  // FIXME: Validate: same index tree

  Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
  Domain rhs_colors = runtime->get_index_partition_color_space(ctx, rhs);

  legion_terra_index_cross_product_t prod;
  prod.partition = lhs;

  for (Domain::DomainPointIterator lhs_dp(lhs_colors); lhs_dp; lhs_dp++) {
    Color lhs_color = lhs_dp.p.get_point<1>()[0];
    IndexSpace lhs_space = runtime->get_index_subspace(ctx, lhs, lhs_color);

    std::set<ptr_t> lhs_points;
    for (IndexIterator lhs_it(lhs_space); lhs_it.has_next();) {
      lhs_points.insert(lhs_it.next());
    }

    Coloring lhs_coloring;

    for (Domain::DomainPointIterator rhs_dp(rhs_colors); rhs_dp; rhs_dp++) {
      Color rhs_color = rhs_dp.p.get_point<1>()[0];
      IndexSpace rhs_space = runtime->get_index_subspace(ctx, rhs, rhs_color);

      // Ensure the color exists.
      lhs_coloring[rhs_color];

      for (IndexIterator rhs_it(rhs_space); rhs_it.has_next();) {
        ptr_t rhs_ptr = rhs_it.next();

        if (lhs_points.count(rhs_ptr)) {
          lhs_coloring[rhs_color].points.insert(rhs_ptr);
        }
      }
    }

    runtime->create_index_partition(
      ctx, lhs_space, lhs_coloring,
      runtime->is_index_partition_disjoint(ctx, rhs),
      runtime->get_index_partition_color(ctx, lhs));
  }

  return prod;
}

legion_index_partition_t
legion_terra_index_cross_product_get_partition(
  legion_terra_index_cross_product_t prod)
{
  return prod.partition;
}

legion_index_partition_t
legion_terra_index_cross_product_get_subpartition_by_color(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_terra_index_cross_product_t prod,
  legion_color_t color)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  IndexSpace is = runtime->get_index_subspace(ctx, prod.partition, color);
  IndexPartition ip = runtime->get_index_partition(
    ctx, is, runtime->get_index_partition_color(ctx, prod.partition));
  return ip;
}

legion_terra_cached_index_iterator_t
legion_terra_cached_index_iterator_create(
  legion_index_space_t handle_)
{
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  CachedIndexIterator *result = new CachedIndexIterator(handle, true);
  return TerraCObjectWrapper::wrap(result);
}

void
legion_terra_cached_index_iterator_destroy(
  legion_terra_cached_index_iterator_t handle_)
{
  CachedIndexIterator *handle = TerraCObjectWrapper::unwrap(handle_);

  delete handle;
}

bool
legion_terra_cached_index_iterator_has_next(
  legion_terra_cached_index_iterator_t handle_)
{
  CachedIndexIterator *handle = TerraCObjectWrapper::unwrap(handle_);

  return handle->has_next();
}

legion_ptr_t
legion_terra_cached_index_iterator_next_span(
  legion_terra_cached_index_iterator_t handle_,
  size_t *count,
  size_t req_count /* must be -1 */)
{
  assert(req_count == size_t(-1));
  CachedIndexIterator *handle = TerraCObjectWrapper::unwrap(handle_);

  ptr_t result = handle->next_span(count);
  return CObjectWrapper::wrap(result);
}

void
legion_terra_cached_index_iterator_reset(
  legion_terra_cached_index_iterator_t handle_)
{
  CachedIndexIterator *handle = TerraCObjectWrapper::unwrap(handle_);

  handle->reset();
}
