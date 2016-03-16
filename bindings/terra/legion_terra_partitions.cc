/* Copyright 2016 Stanford University, NVIDIA Corporation
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
#include "legion_terra_partitions_cxx.h"

#include "arrays.h"
#include "legion.h"
#include "legion_c_util.h"
#include "legion_utilities.h"
#include "lowlevel.h"
#include "utilities.h"

#ifdef SHARED_LOWLEVEL
#define USE_LEGION_CROSS_PRODUCT 1
#else
// General LLR can't handle new partion API yet.
#define USE_LEGION_CROSS_PRODUCT 0
#endif

#ifndef USE_TLS
// Mac OS X and GCC <= 4.7 do not support C++11 thread_local.
#if __cplusplus < 201103L || defined(__MACH__) || (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 7)))
#define USE_TLS 0
#else
#define USE_TLS 1
#endif
#endif

struct CachedIndexIterator {
public:
  CachedIndexIterator(HighLevelRuntime *rt, Context ctx, IndexSpace is, bool gl)
    : runtime(rt), context(ctx), space(is), index(0), cached(false), global(gl)
  {
  }

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
      LegionRuntime::HighLevel::AutoLock guard(global_lock);
#endif
      std::map<IndexSpace, std::vector<std::pair<ptr_t, size_t> > >::iterator it =
        global_cache.find(space);
      if (it != global_cache.end()) {
        spans = it->second;
        cached = true;
        return;
      }
    }

    IndexIterator it(runtime, context, space);
    while (it.has_next()) {
      size_t count = 0;
      ptr_t start = it.next_span(count);
      assert(count && !start.is_null());
      spans.push_back(std::pair<ptr_t, size_t>(start, count));
    }

    if (global) {
#if USE_TLS
      global_cache[space] = spans;
#else
      LegionRuntime::HighLevel::AutoLock guard(global_lock);
      if (!global_cache.count(space)) {
        global_cache[space] = spans;
      }
#endif
    }

    cached = true;
  }

private:
  HighLevelRuntime *runtime;
  Context context;
  IndexSpace space;
  std::vector<std::pair<ptr_t, size_t> > spans;
  size_t index;
  bool cached;
  bool global;
private:
#if USE_TLS
  static thread_local
  std::map<IndexSpace, std::vector<std::pair<ptr_t, size_t> > > global_cache;
#else
  static std::map<IndexSpace, std::vector<std::pair<ptr_t, size_t> > > global_cache;
  static ImmovableLock global_lock;
#endif
};

#if USE_TLS
thread_local std::map<IndexSpace, std::vector<std::pair<ptr_t, size_t> > >
  CachedIndexIterator::global_cache;
#else
std::map<IndexSpace, std::vector<std::pair<ptr_t, size_t> > >
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

#if !USE_LEGION_CROSS_PRODUCT
static void
create_cross_product_coloring(HighLevelRuntime *runtime,
                              Context ctx,
                              IndexPartition lhs,
                              IndexPartition rhs,
                              std::map<Color, Coloring> &coloring)
{
  Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
  Domain rhs_colors = runtime->get_index_partition_color_space(ctx, rhs);

  for (Domain::DomainPointIterator lh_dp(lhs_colors); lh_dp; lh_dp++) {
    Color lh_color = lh_dp.p.get_point<1>()[0];
    IndexSpace lh_space = runtime->get_index_subspace(ctx, lhs, lh_color);

    for (Domain::DomainPointIterator rh_dp(rhs_colors); rh_dp; rh_dp++) {
      Color rh_color = rh_dp.p.get_point<1>()[0];
      IndexSpace rh_space = runtime->get_index_subspace(ctx, rhs, rh_color);

      for (IndexIterator lh_it(runtime, ctx, lh_space); lh_it.has_next();) {
        size_t lh_count = 0;
        ptr_t lh_ptr = lh_it.next_span(lh_count);
        ptr_t lh_end = lh_ptr.value + lh_count - 1;

        for (IndexIterator rh_it(runtime, ctx, rh_space, lh_ptr); rh_it.has_next();) {
          size_t rh_count = 0;
          ptr_t rh_ptr = rh_it.next_span(rh_count);
          ptr_t rh_end = rh_ptr.value + rh_count - 1;
          if (rh_ptr.value > lh_end.value) {
            break;
          }

          if (rh_end.value > lh_end.value) {
            coloring[lh_color][rh_color].ranges.insert(std::pair<ptr_t, ptr_t>(rh_ptr, lh_end));
            break;
          }

          coloring[lh_color][rh_color].ranges.insert(std::pair<ptr_t, ptr_t>(rh_ptr, rh_end));
        }
      }
    }
  }
}
#endif

Color
create_cross_product(HighLevelRuntime *runtime,
                     Context ctx,
                     IndexPartition lhs,
                     IndexPartition rhs,
                     Color rhs_color /* = -1 */)
{
  if (rhs_color == (Color)-1) {
    // Try to find a color that isn't already being used.
    Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
    Domain::DomainPointIterator lh_dp(lhs_colors);
    assert(lh_dp);
    Color lh_color = lh_dp.p.get_point<1>()[0];
    IndexSpace lh_subspace = runtime->get_index_subspace(ctx, lhs, lh_color);
    std::set<Color> colors;
    runtime->get_index_space_partition_colors(ctx, lh_subspace, colors);

    Color original_rhs_color = runtime->get_index_partition_color(ctx, rhs);
    for (Color c = original_rhs_color;; c++) {
      if (!colors.count(c)) {
        rhs_color = c;
        break;
      }
    }
  }

#if USE_LEGION_CROSS_PRODUCT
  std::map<DomainPoint, IndexPartition> handles;
  runtime->create_cross_product_partitions(
    ctx, lhs, rhs, handles,
    (runtime->is_index_partition_disjoint(ctx, rhs) ? DISJOINT_KIND : ALIASED_KIND),
    rhs_color, true);
#else
  // FIXME: Validate: same index tree

  Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
  Domain rhs_colors = runtime->get_index_partition_color_space(ctx, rhs);

  // The efficiency of this algorithm depends heavily on how many
  // spans are in lhs and rhs. Since it is *MUCH* better to have a
  // smaller number of spans on lhs, it is worth spending a little
  // time here estimating which will have fewer spans.

  bool flip = false;
  for (Domain::DomainPointIterator lh_dp(lhs_colors), rh_dp(rhs_colors); lh_dp && rh_dp;) {
    Color lh_color = lh_dp.p.get_point<1>()[0];
    IndexSpace lh_space = runtime->get_index_subspace(ctx, lhs, lh_color);
    Color rh_color = rh_dp.p.get_point<1>()[0];
    IndexSpace rh_space = runtime->get_index_subspace(ctx, rhs, rh_color);

    IndexIterator lh_it(runtime, ctx, lh_space);
    IndexIterator rh_it(runtime, ctx, rh_space);
    for (; lh_it.has_next() && rh_it.has_next();) {
      size_t lh_count = 0;
      lh_it.next_span(lh_count);
      size_t rh_count = 0;
      rh_it.next_span(rh_count);
    }

    flip = lh_it.has_next() && !rh_it.has_next();
    break;
  }

  std::map<Color, Coloring> coloring;
  if (flip) {
    create_cross_product_coloring(runtime, ctx, rhs, lhs, coloring);
  } else {
    create_cross_product_coloring(runtime, ctx, lhs, rhs, coloring);
  }

  for (Domain::DomainPointIterator lh_dp(lhs_colors); lh_dp; lh_dp++) {
    Color lh_color = lh_dp.p.get_point<1>()[0];
    IndexSpace lh_space = runtime->get_index_subspace(ctx, lhs, lh_color);

    Coloring empty; // Make sure this stays on the stack while we need it...
    Coloring &lh_coloring = flip ? empty : coloring[lh_color];
    for (Domain::DomainPointIterator rh_dp(rhs_colors); rh_dp; rh_dp++) {
      Color rh_color = rh_dp.p.get_point<1>()[0];

      // Flip order of coloring.
      if (flip) {
        lh_coloring[rh_color] = coloring[rh_color][lh_color];
      }

      // Ensure the color exists.
      lh_coloring[rh_color];
    }

    runtime->create_index_partition(
      ctx, lh_space, lh_coloring,
      runtime->is_index_partition_disjoint(ctx, rhs),
      rhs_color);
  }
#endif

  return rhs_color;
}

static void
create_cross_product_multi(HighLevelRuntime *runtime,
                           Context ctx,
                           size_t npartitions,
                           IndexPartition next,
                           std::vector<IndexPartition>::iterator rest,
                           std::vector<IndexPartition>::iterator end,
                           std::vector<Color> &result_colors,
                           int level)
{
  if (rest != end) {
    Color desired_color = result_colors[level];
    Color color = create_cross_product(runtime, ctx, next, *rest, desired_color);
    if (desired_color == Color(-1)) {
      result_colors[level] = color;
    } else {
      assert(color == desired_color);
    }

    Domain colors = runtime->get_index_partition_color_space(ctx, next);
    Color color2 = runtime->get_index_partition_color(ctx, *rest);
    for (Domain::DomainPointIterator dp(colors); dp; dp++) {
      Color color = dp.p.get_point<1>()[0];
      IndexSpace is = runtime->get_index_subspace(ctx, next, color);
      IndexPartition ip = runtime->get_index_partition(ctx, is, color2);
      create_cross_product_multi(
        runtime, ctx, npartitions - 1, ip, rest+1, end, result_colors, level + 1);
    }
  }
}

legion_terra_index_cross_product_t
legion_terra_index_cross_product_create(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_index_partition_t lhs_,
                                        legion_index_partition_t rhs_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition lhs = CObjectWrapper::unwrap(lhs_);
  IndexPartition rhs = CObjectWrapper::unwrap(rhs_);

  Color rhs_color = create_cross_product(runtime, ctx, lhs, rhs);

  legion_terra_index_cross_product_t result;
  result.partition = CObjectWrapper::wrap(lhs);
  result.other_color = rhs_color;
  return result;
}

legion_terra_index_cross_product_t
legion_terra_index_cross_product_create_multi(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t *partitions_, // input
  legion_color_t *colors_, // output
  size_t npartitions)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  std::vector<IndexPartition> partitions;
  for (size_t i = 0; i < npartitions; i++) {
    partitions.push_back(CObjectWrapper::unwrap(partitions_[i]));
  }
  std::vector<Color> colors(npartitions-1, Color(-1));

  assert(npartitions >= 2);
  create_cross_product_multi(runtime, ctx, npartitions,
                             partitions[0],
                             partitions.begin() + 1, partitions.end(), colors, 0);

  colors_[0] = runtime->get_index_partition_color(ctx, partitions[0]);
  std::copy(colors.begin(), colors.end(), colors_+1);

  legion_terra_index_cross_product_t result;
  result.partition = CObjectWrapper::wrap(partitions[0]);
  result.other_color = colors[0];
  return result;
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
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition partition = CObjectWrapper::unwrap(prod.partition);

  IndexSpace is = runtime->get_index_subspace(ctx, partition, color);
  IndexPartition ip = runtime->get_index_partition(ctx, is, prod.other_color);
  return CObjectWrapper::wrap(ip);
}

legion_terra_cached_index_iterator_t
legion_terra_cached_index_iterator_create(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  CachedIndexIterator *result = new CachedIndexIterator(runtime, ctx, handle, true);
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

unsigned
legion_terra_task_launcher_get_region_requirement_logical_region(
  legion_task_launcher_t launcher_,
  legion_logical_region_t region_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);

  unsigned idx = (unsigned)-1;
  for (unsigned i = 0; i < launcher->region_requirements.size(); i++) {
    if (launcher->region_requirements[i].handle_type == SINGULAR &&
        launcher->region_requirements[i].region == region) {
      idx = i;
      break;
    }
  }
  return idx;
}

bool
legion_terra_task_launcher_has_field(
  legion_task_launcher_t launcher_,
  unsigned idx,
  legion_field_id_t fid)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  const RegionRequirement &req = launcher->region_requirements[idx];
  return req.privilege_fields.count(fid) > 0;
}
