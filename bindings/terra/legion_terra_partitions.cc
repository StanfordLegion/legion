/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#if !USE_TLS
class AutoImmovableLock {
public:
  AutoImmovableLock(ImmovableLock& _lock)
    : lock(_lock)
  {
    lock.lock();
  }

  ~AutoImmovableLock(void)
  {
    lock.unlock();
  }

protected:
  ImmovableLock& lock;
};
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

  ptr_t next_span(size_t &count) {
    if (!cached) cache();
    if (index >= spans.size()) {
      count = 0;
      return ptr_t::nil();
    }
    std::pair<ptr_t, size_t> span = spans[index++];
    count = span.second;
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
      AutoImmovableLock guard(global_lock);
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
      AutoImmovableLock guard(global_lock);
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

#if 0
static legion_terra_index_space_list_list_t
create_list_list(legion_terra_index_space_list_t lhs_, size_t size1, size_t size2)
{
  legion_terra_index_space_list_list_t result;
  result.count = size1;
  result.sublists =
    (legion_terra_index_space_list_t*)calloc(size1,
        sizeof(legion_terra_index_space_list_t));

  for (size_t idx = 0; idx < size1; ++idx) {
    result.sublists[idx].count = size2;
    result.sublists[idx].subspaces =
      (legion_index_space_t*)calloc(size2, sizeof(legion_index_space_t));
    result.sublists[idx].space = lhs_.subspaces[idx];
  }
  return result;
}
#endif

static inline legion_terra_index_space_list_list_t
create_list_list(std::vector<IndexSpace> &spaces,
                 std::map<IndexSpace, std::vector<IndexSpace> > &product)
{
  legion_terra_index_space_list_list_t result;
  result.count = spaces.size();
  result.sublists = (legion_terra_index_space_list_t*)calloc(result.count,
      sizeof(legion_terra_index_space_list_t));
  for (size_t idx = 0; idx < result.count; ++idx) {
    IndexSpace& space = spaces[idx];
    std::vector<IndexSpace>& subspaces = product[space];
    size_t size = subspaces.size();
    result.sublists[idx].count = size;
    result.sublists[idx].subspaces =
      (legion_index_space_t*)calloc(size, sizeof(legion_index_space_t));
    result.sublists[idx].space = CObjectWrapper::wrap(space);
  }
  return result;
}

static inline void
assign_list(legion_terra_index_space_list_t& ls,
            size_t idx, legion_index_space_t is)
{
  ls.subspaces[idx] = is;
}

static inline void
assign_list_list(legion_terra_index_space_list_list_t& ls,
                 size_t idx1, size_t idx2, legion_index_space_t is)
{
  assign_list(ls.sublists[idx1], idx2, is);
}

// Returns true if the index space `is` is structured.
static bool
is_structured(HighLevelRuntime *runtime, Context ctx, IndexSpace is) {
  Domain is_domain = runtime->get_index_space_domain(ctx, is);
  return is_domain.get_dim() > 0;
}

// Returns true if the index space `ip` belongs to is structured.
static bool
is_structured(HighLevelRuntime *runtime, Context ctx, IndexPartition ip) {
  IndexSpace is = runtime->get_parent_index_space(ctx, ip);
  return is_structured(runtime, ctx, is);
}

// Returns true if `lhs` and `rhs` should be switched when we take their
// intersection to achieve better performance.  This is the case when the index
// space is unstructured and `rhs` is "smaller" than `lhs`.
static bool
should_flip_cross_product(HighLevelRuntime *runtime,
                          Context ctx,
                          IndexPartition lhs,
                          IndexPartition rhs,
                          const std::set<DomainPoint> *lhs_filter = NULL,
                          const std::set<DomainPoint> *rhs_filter = NULL)
{
  // The two partitions should belong to the same index tree.
  assert(lhs.get_tree_id() == rhs.get_tree_id());

  // Not necessary to flip in the structured case.
  if (is_structured(runtime, ctx, lhs)) { return false; }

  Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
  Domain rhs_colors = runtime->get_index_partition_color_space(ctx, rhs);

  size_t lhs_span_count = 0, rhs_span_count = 0;
  for (Domain::DomainPointIterator lh_dp(lhs_colors), rh_dp(rhs_colors);; lh_dp++, rh_dp++) {
    while (lh_dp && lhs_filter && !lhs_filter->count(lh_dp.p)) {
      lh_dp++;
    }
    if (!lh_dp) { break; }
    DomainPoint lh_color = lh_dp.p;
    IndexSpace lh_space = runtime->get_index_subspace(ctx, lhs, lh_color);

    while (rh_dp && rhs_filter && !rhs_filter->count(rh_dp.p)) {
      rh_dp++;
    }
    if (!rh_dp) { break; }
    DomainPoint rh_color = rh_dp.p;
    IndexSpace rh_space = runtime->get_index_subspace(ctx, rhs, rh_color);

    IndexIterator lh_it(runtime, ctx, lh_space), rh_it(runtime, ctx, rh_space);
    size_t lh_count = 0, rh_count = 0;
    for (; lh_it.has_next() && rh_it.has_next();) {
      lh_it.next_span(lh_count);
      rh_it.next_span(rh_count);
    }

    lhs_span_count += lh_it.has_next();
    rhs_span_count += rh_it.has_next();
  }
  return rhs_span_count < lhs_span_count;
}

#if !USE_LEGION_CROSS_PRODUCT
// Creates cross product between structured IndexPartition's.
// See documentation of `create_cross_product()` for details.
static Color
create_cross_product_structured(HighLevelRuntime *runtime,
                                Context ctx,
                                IndexPartition lhs,
                                IndexPartition rhs,
                                Color rhs_color,
                                bool consistent_ids,
                                std::map<IndexSpace, Color> *chosen_colors,
                                const std::set<DomainPoint> *lhs_filter,
                                const std::set<DomainPoint> *rhs_filter)
{
  // In the structured case, it makes no difference to "flip" `lhs` and `rhs`.

  Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
  Domain rhs_colors = runtime->get_index_partition_color_space(ctx, rhs);

  // Double for-loop: take rectangle intersection between each pair of
  // subspaces.
  for (Domain::DomainPointIterator lh_dp(lhs_colors); lh_dp; lh_dp++) {
    const DomainPoint &lh_color = lh_dp.p;
    if (lhs_filter && !lhs_filter->count(lh_color)) continue;

    DomainPointColoring lh_coloring; // Coloring for current lhs region.

    IndexSpace lh_space = runtime->get_index_subspace(ctx, lhs, lh_color);
    // Doesn't currently handle structured index spaces consisting of multiple
    // domains.
    assert(!runtime->has_multiple_domains(lh_space));
    Domain lh_domain = runtime->get_index_space_domain(lh_space);
    // Verify that index space is indeed structured.
    assert(lh_domain.get_dim() > 0);

    for (Domain::DomainPointIterator rh_dp(rhs_colors); rh_dp; rh_dp++) {
      const DomainPoint &rh_color = rh_dp.p;
      if (rhs_filter && !rhs_filter->count(rh_color)) continue;

      IndexSpace rh_space = runtime->get_index_subspace(ctx, rhs, rh_color);
      assert(!runtime->has_multiple_domains(rh_space));
      Domain rh_domain = runtime->get_index_space_domain(rh_space);
      assert(rh_domain.get_dim() > 0);

      // Take and store intersecton of the two rect domains.
      lh_coloring[rh_color] = lh_domain.intersection(rh_domain);
    }

    // Create cross-product partition for current region in `lhs`.  We simply
    // put `rhs_colors` as the color space, which may unfortunately include
    // indices that are filtered out.
    IndexPartition part = runtime->create_index_partition(
      ctx, lh_space, rhs_colors, lh_coloring,
      runtime->is_index_partition_disjoint(ctx, rhs) ? DISJOINT_KIND : ALIASED_KIND,
      rhs_color);
    if (chosen_colors) {
      (*chosen_colors)[lh_space] = runtime->get_index_partition_color(ctx, part);
    }
    if (rhs_color == Color(-1) && consistent_ids) {
      rhs_color = runtime->get_index_partition_color(ctx, part);
    }
  }

  return rhs_color;
}

// Creates and stores coloring for cross product in the `coloring` parameter,
// for unstructured IndexPartition's `lhs` and `rhs`.
//
// If `flip_coloring` is set, `coloring` will be created as a Map (rh_color =>
// lh_color => intersection); otherwise, it will be a Map (lh_color => rh_color
// => intersection).
static void
create_cross_product_coloring_unstructured(HighLevelRuntime *runtime,
                                           Context ctx,
                                           IndexPartition lhs,
                                           IndexPartition rhs,
                                           std::map<DomainPoint, PointColoring> &coloring,
                                           bool flip_coloring,
                                           const std::set<DomainPoint> *lhs_filter,
                                           const std::set<DomainPoint> *rhs_filter)
{
  Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
  Domain rhs_colors = runtime->get_index_partition_color_space(ctx, rhs);

  // Double for-loop: take intersection between each pair of subspaces.
  for (Domain::DomainPointIterator lh_dp(lhs_colors); lh_dp; lh_dp++) {
    const DomainPoint &lh_color = lh_dp.p;
    if (lhs_filter && !lhs_filter->count(lh_color)) continue;
    IndexSpace lh_space = runtime->get_index_subspace(ctx, lhs, lh_color);

    for (Domain::DomainPointIterator rh_dp(rhs_colors); rh_dp; rh_dp++) {
      const DomainPoint &rh_color = rh_dp.p;
      if (rhs_filter && !rhs_filter->count(rh_color)) continue;
      IndexSpace rh_space = runtime->get_index_subspace(ctx, rhs, rh_color);

      PointColoring::mapped_type &write_to = flip_coloring ?
        coloring[rh_color][lh_color] : coloring[lh_color][rh_color];

      // Take intersection between `lh_space` and `rh_space` by iterating
      // through spans in each with a double for-loop and taking intersection
      // between spans.
      for (IndexIterator lh_it(runtime, ctx, lh_space); lh_it.has_next(); ) {
        size_t lh_count = 0;
        ptr_t lh_ptr = lh_it.next_span(lh_count);
        ptr_t lh_end = lh_ptr.value + lh_count - 1;

        for (IndexIterator rh_it(runtime, ctx, rh_space, /* start = */ lh_ptr);
             rh_it.has_next(); ) {
          size_t rh_count = 0;
          ptr_t rh_ptr = rh_it.next_span(rh_count);
          ptr_t rh_end = rh_ptr.value + rh_count - 1;

          if (rh_ptr.value > lh_end.value) {
            // lh_ptr           lh_end
            //  |------------------|
            //                       |----------------|-->     |---...
            //                      rh_ptr         rh_end     rh_ptr'
            // RHS span entirely exceeds LHS span; there isn't, and won't be,
            // any overlap as the RHS span continues moving forward.
            break;
          }
          if (rh_end.value > lh_end.value) {
            // lh_ptr           lh_end
            //  |------------------|
            //            |----------------|-->      |---...
            //           rh_ptr         rh_end      rh_ptr'
            // This is the last time there's any overlap as RHS continues
            // moving forward.
            write_to.ranges.insert(std::pair<ptr_t, ptr_t>(rh_ptr, lh_end));
            break;
          }

          // lh_ptr           lh_end
          //  |------------------|
          //    |---------|-->
          //   rh_ptr  rh_end
          // More overlap is possible as RHS moves forward.
          write_to.ranges.insert(std::pair<ptr_t, ptr_t>(rh_ptr, rh_end));
        }
      }
    }
  }
}

// Creates cross product between unstructured IndexPartition's.
// See documentation of `create_cross_product()` for details.
static Color
create_cross_product_unstructured(HighLevelRuntime *runtime,
                                  Context ctx,
                                  IndexPartition lhs,
                                  IndexPartition rhs,
                                  Color rhs_color,
                                  bool consistent_ids,
                                  std::map<IndexSpace, Color> *chosen_colors,
                                  const std::set<DomainPoint> *lhs_filter,
                                  const std::set<DomainPoint> *rhs_filter)
{
  std::map<DomainPoint, PointColoring> coloring; // lh_color->rh_color->intersection

  // The efficiency of this algorithm depends heavily on how many
  // spans are in lhs and rhs. Since it is *MUCH* better to have a
  // smaller number of spans on lhs, it is worth spending a little
  // time here estimating which will have fewer spans.
  bool flip = should_flip_cross_product(runtime, ctx, lhs, rhs, lhs_filter, rhs_filter);
  if (flip) {
    // We switch `lhs` and `rhs` (and their filters) in the params and also
    // set `flip_coloring`, so that the output `map` be in the same format.
    create_cross_product_coloring_unstructured(runtime, ctx, rhs, lhs,
        coloring, /* flip_coloring = */ true,  rhs_filter, lhs_filter);
  } else {
    create_cross_product_coloring_unstructured(runtime, ctx, lhs, rhs,
        coloring, /* flip_coloring = */ false, lhs_filter, rhs_filter);
  }

  // Create index partitions for each LHS partition.
  Domain lhs_colors = runtime->get_index_partition_color_space(ctx, lhs);
  Domain rhs_colors = runtime->get_index_partition_color_space(ctx, rhs);
  bool allocable =
#ifdef ASSUME_UNALLOCABLE
    false
#else
    true
#endif
    ;

  for (Domain::DomainPointIterator lh_dp(lhs_colors); lh_dp; lh_dp++) {
    const DomainPoint &lh_color = lh_dp.p;
    if (lhs_filter && !lhs_filter->count(lh_color)) continue;
    IndexSpace lh_space = runtime->get_index_subspace(ctx, lhs, lh_color);

    IndexPartition part = runtime->create_index_partition(
        ctx, lh_space, rhs_colors, coloring[lh_color],
        runtime->is_index_partition_disjoint(ctx, rhs) ? DISJOINT_KIND : ALIASED_KIND,
        rhs_color, allocable);
    if (chosen_colors) {
      (*chosen_colors)[lh_space] = runtime->get_index_partition_color(ctx, part);
    }
    if (rhs_color == Color(-1) && consistent_ids) {
      rhs_color = runtime->get_index_partition_color(ctx, part);
    }
  }

  return rhs_color;
}
#endif

Color
create_cross_product(HighLevelRuntime *runtime,
                     Context ctx,
                     IndexPartition lhs,
                     IndexPartition rhs,
                     Color rhs_color /* = -1 */,
                     bool consistent_ids /* = true */,
                     std::map<IndexSpace, Color> *chosen_colors /* = NULL */,
                     const std::set<DomainPoint> *lhs_filter /* = NULL */,
                     const std::set<DomainPoint> *rhs_filter /* = NULL */)
{
#if USE_LEGION_CROSS_PRODUCT
  std::map<DomainPoint, IndexPartition> handles;
  runtime->create_cross_product_partitions(
    ctx, lhs, rhs, handles,
    (runtime->is_index_partition_disjoint(ctx, rhs) ? DISJOINT_KIND : ALIASED_KIND),
    rhs_color, true);
  return rhs_color;
#else
  // The two partitions should belong to the same index tree.
  assert(lhs.get_tree_id() == rhs.get_tree_id());
  
  if (is_structured(runtime, ctx, lhs)) {
    return create_cross_product_structured(runtime, ctx, lhs, rhs, rhs_color,
        consistent_ids, chosen_colors, lhs_filter, rhs_filter);
  } else {
    return create_cross_product_unstructured(runtime, ctx, lhs, rhs, rhs_color,
        consistent_ids, chosen_colors, lhs_filter, rhs_filter);
  }
#endif
}

// For each unstructured IndexSpace in `index_spaces`, stores its bounds (first & last
// element) as pairs in `bounds`.
static inline void
get_bounding_boxes(HighLevelRuntime *runtime,
                   Context ctx,
                   const std::vector<IndexSpace> &index_spaces,
                   std::vector<std::pair<ptr_t, ptr_t> >& bounds)
{
  bounds.reserve(index_spaces.size());
  for (std::vector<IndexSpace>::const_iterator isit = index_spaces.begin();
       isit != index_spaces.end(); ++isit) {
    const IndexSpace& space = *isit;

    bool is_first = true;
    ptr_t first, last;
    for (IndexIterator it(runtime, ctx, space); it.has_next();) {
      size_t count = 0;
      ptr_t ptr = it.next_span(count);
      ptr_t end = ptr.value + count - 1;
      if (is_first) {
        first = ptr;
        is_first = false;
      }
      last = end;
    }
    bounds.push_back(std::pair<ptr_t, ptr_t>(first, last));
  }
}

struct BoundComp {
  bool operator()(const std::pair<ptr_t, ptr_t>& b1,
                  const std::pair<ptr_t, ptr_t>& b2) const
  {
    return b1.first.value < b2.first.value;
  }
};

struct Leaf;
struct NonLeaf;
#define NODE_SIZE 256
#define NUM_LEAF_ENTRIES 15
#define NUM_NONLEAF_ENTRIES 10

struct Node {
  bool is_leaf() { return leaf; }
  Leaf* as_leaf() { return (Leaf*)this; }
  NonLeaf* as_nonleaf() { return (NonLeaf*)this; }
  bool leaf;
  int num_children;
};

struct Leaf {
  bool leaf;
  int num_children;
  long long int starts[NUM_LEAF_ENTRIES];
  long long int indices[NUM_LEAF_ENTRIES];
  Leaf* next;
};

struct NonLeaf {
  bool leaf;
  int num_children;
  long long int starts[NUM_NONLEAF_ENTRIES];
  long long int ends[NUM_NONLEAF_ENTRIES];
  Node* childs[NUM_NONLEAF_ENTRIES];
};

Node* create_interval_tree(std::vector<std::pair<ptr_t, ptr_t> >& lower_bounds,
                           std::vector<ptr_t>& upper_bounds)
{
  unsigned num_bounds = lower_bounds.size();
  unsigned num_leaves = (num_bounds + NUM_LEAF_ENTRIES - 1) / NUM_LEAF_ENTRIES;
  void* buffer_for_leaves = malloc(NODE_SIZE * num_leaves);

  std::vector<std::pair<ptr_t, ptr_t> >::iterator it = lower_bounds.begin();
  // make leaves
  char* ptr = (char*)buffer_for_leaves;
  unsigned remaining_bounds = num_bounds;
  while (remaining_bounds > 0)
  {
    Leaf* node = (Leaf*)ptr;
    unsigned num_children =
      std::min(remaining_bounds, (unsigned)NUM_LEAF_ENTRIES);
    node->leaf = 1;
    node->num_children = num_children;
    for (unsigned idx = 0; idx < num_children; ++idx)
    {
      node->starts[idx] = it->first.value;
      node->indices[idx] = it->second.value;
      it++;
    }
    ptr += NODE_SIZE;
    remaining_bounds -= num_children;
    node->next = remaining_bounds > 0 ? (Leaf*)ptr : 0;
  }

  // make non-leaf nodes
  int num_nonleaves =
    (num_leaves + NUM_NONLEAF_ENTRIES - 1) / NUM_NONLEAF_ENTRIES;
  unsigned num_prev_nodes = num_leaves;
  char* prev_ptr = (char*)buffer_for_leaves;
  while (num_nonleaves > 0)
  {
    void* buffer_for_nonleaves = malloc(NODE_SIZE * num_nonleaves);
    char* ptr = (char*)buffer_for_nonleaves;
    unsigned remaining_nodes = num_prev_nodes;
    while (remaining_nodes > 0)
    {
      NonLeaf* node = (NonLeaf*)ptr;
      unsigned num_children =
        std::min(remaining_nodes, (unsigned)NUM_NONLEAF_ENTRIES);
      node->leaf = 0;
      node->num_children = num_children;
      for (unsigned idx = 0; idx < num_children; ++idx)
      {
        Node* child = (Node*)prev_ptr;
        node->childs[idx] = child;
        if (child->is_leaf())
        {
          node->starts[idx] = child->as_leaf()->starts[0];
          node->ends[idx] =
            upper_bounds[child->as_leaf()->indices[child->num_children - 1]];
        }
        else
        {
          node->starts[idx] = child->as_nonleaf()->starts[0];
          node->ends[idx] =
            child->as_nonleaf()->ends[child->num_children - 1];
        }
        prev_ptr += NODE_SIZE;
      }
      ptr += NODE_SIZE;
      remaining_nodes -= num_children;
    }
    prev_ptr = (char*)buffer_for_nonleaves;
    num_prev_nodes = num_nonleaves;
    if (num_nonleaves == 1) break;
    num_nonleaves =
      (num_nonleaves + NUM_NONLEAF_ENTRIES - 1) / NUM_NONLEAF_ENTRIES;
  }
  return (Node*)prev_ptr;
}

void print_tree(Node* tree, unsigned level)
{
  printf("tree: %p\n", tree);
  if (tree->is_leaf())
  {
    Leaf* leaf = tree->as_leaf();
    for (unsigned idx = 0; idx < level; ++idx)
      printf("    ");
    for (int idx = 0; idx < leaf->num_children; ++idx)
      printf("start: %lld index: %lld, ", leaf->starts[idx], leaf->indices[idx]);
    printf("\n");
  }
  else
  {
    NonLeaf* nonleaf = tree->as_nonleaf();
    for (int idx = 0; idx < nonleaf->num_children; ++idx)
    {
      for (unsigned j = 0; j < level; ++j) printf("    ");
      printf("start: %lld end: %lld ptr: %p\n",
          nonleaf->starts[idx], nonleaf->ends[idx], nonleaf->childs[idx]);
      print_tree(nonleaf->childs[idx], level + 1);
    }
  }
}

static void
find_first_bounding_box(Node* root, unsigned key, Leaf*& leaf, int& idx)
{
  Node* node = root;
  while (!node->is_leaf())
  {
    bool found = false;
    NonLeaf* nonleaf = node->as_nonleaf();
    for (int idx = 0; idx < nonleaf->num_children; ++idx)
      if (nonleaf->starts[idx] <= key && key <= nonleaf->ends[idx])
      {
        node = nonleaf->childs[idx];
        found = true;
        break;
      }
    // not found any bounding box
    if (!found) return;
  }
  leaf = node->as_leaf();

  for (idx = 0; idx < leaf->num_children - 1; ++idx)
    if (leaf->starts[idx] <= key && key <= leaf->starts[idx + 1])
      return;
}

static void
create_cross_product_tree(HighLevelRuntime *runtime,
                          Context ctx,
                          bool flip,
                          const std::vector<IndexSpace> &lhs,
                          bool lhs_disjoint,
                          const std::vector<IndexSpace> &rhs,
                          bool rhs_disjoint,
                          legion_terra_logical_region_list_t *lhs_,
                          legion_terra_logical_region_list_t *rhs_,
                          legion_terra_logical_region_list_list_t *result_)
{
  assert(lhs_disjoint);
  std::vector<std::pair<ptr_t, ptr_t> > lhs_bounds;
  get_bounding_boxes(runtime, ctx, lhs, lhs_bounds);

  std::vector<std::pair<ptr_t, ptr_t> > rhs_bounds;
  get_bounding_boxes(runtime, ctx, rhs, rhs_bounds);

  std::vector<ptr_t> lhs_upper_bounds;
  lhs_upper_bounds.reserve(lhs_bounds.size());
  for (unsigned idx = 0; idx < lhs_bounds.size(); ++idx)
  {
    lhs_upper_bounds[idx] = lhs_bounds[idx].second;
    lhs_bounds[idx].second.value = idx;
  }
  BoundComp cmp;
  for (unsigned idx = 1; idx < lhs_bounds.size(); ++idx)
    if (!cmp(lhs_bounds[idx - 1], lhs_bounds[idx]))
    {
      std::sort(lhs_bounds.begin(), lhs_bounds.end(), cmp);
      break;
    }
  Node* root = create_interval_tree(lhs_bounds, lhs_upper_bounds);

  std::vector<unsigned> potentially_overlap;
  potentially_overlap.reserve(lhs_bounds.size());
  for (size_t rhs_idx = 0; rhs_idx < rhs_bounds.size(); rhs_idx++) {
    std::pair<ptr_t, ptr_t>& rhs_bound = rhs_bounds[rhs_idx];
    Leaf* leaf = 0;
    int idx = -1;
    find_first_bounding_box(root, rhs_bound.first.value, leaf, idx);
    if (leaf == 0) continue;

    while (leaf->starts[idx] <= rhs_bound.second.value) {
      int lhs_idx = leaf->indices[idx];
      if (rhs_bound.first.value <= lhs_upper_bounds[lhs_idx].value)
        potentially_overlap.push_back(lhs_idx);

      ++idx;
      if (idx >= leaf->num_children) {
        idx = 0;
        leaf = leaf->next;
      }
      if (leaf == 0) break;
    }

    const IndexSpace& rh_space = rhs[rhs_idx];
    for (unsigned idx = 0; idx < potentially_overlap.size(); ++idx) {
      int lhs_idx = potentially_overlap[idx];
      const IndexSpace& lh_space = lhs[lhs_idx];
      bool intersects = false;
      for (IndexIterator lh_it(runtime, ctx, lh_space); lh_it.has_next();) {
        size_t lh_count = 0;
        ptr_t lh_ptr = lh_it.next_span(lh_count);
        ptr_t lh_end = lh_ptr.value + lh_count - 1;

        IndexIterator rh_it(runtime, ctx, rh_space, lh_ptr);
        if (rh_it.has_next()) {
          size_t rh_count = 0;
          ptr_t rh_ptr = rh_it.next_span(rh_count);

          if (rh_ptr.value <= lh_end.value) {
            intersects = true;
            break;
          }
        }
      }
      if (intersects) {
        if (flip) {
          legion_terra_logical_region_list_t& sublist = result_->sublists[rhs_idx];
          size_t idx = sublist.count++;
          sublist.subregions[idx].index_space = CObjectWrapper::wrap(lh_space);
          sublist.subregions[idx].field_space = lhs_->subregions[lhs_idx].field_space;
          sublist.subregions[idx].tree_id = lhs_->subregions[lhs_idx].tree_id;
        }
        else {
          legion_terra_logical_region_list_t& sublist = result_->sublists[lhs_idx];
          size_t idx = sublist.count++;
          sublist.subregions[idx].index_space = CObjectWrapper::wrap(rh_space);
          sublist.subregions[idx].field_space = rhs_->subregions[rhs_idx].field_space;
          sublist.subregions[idx].tree_id = rhs_->subregions[rhs_idx].tree_id;
        }
      }
    }
    potentially_overlap.clear();
  }
}

// For each index space in `ispaces`, adds its domain's rectangle to `rects`.
// ASSUMES that each ispace is structured and only has one domain.
// `DIM` should match the dimensionality of the index spaces.
template <unsigned DIM>
static void
extract_ispace_domain_rects(HighLevelRuntime *runtime,
                            Context ctx,
                            const std::vector<IndexSpace> &ispaces,
                            std::vector<Rect<DIM> >& rects)
{
  assert(DIM > 0);
  assert(rects.empty());
  rects.reserve(ispaces.size());
  for (size_t i = 0; i < ispaces.size(); i++) {
    const IndexSpace &ispace = ispaces[i];
    // Doesn't currently handle structured index spaces with multiple domains.
    assert(!runtime->has_multiple_domains(ctx, ispace));
    Domain domain = runtime->get_index_space_domain(ctx, ispace);
    rects.push_back(domain.get_rect<DIM>());
  }
}

// Version of `create_cross_product_shallow_structured` templetized on the
// dimensionality of the structured index spaces.
template <unsigned DIM>
static void
create_cross_product_shallow_structured_spec(
    HighLevelRuntime *runtime,
    Context ctx,
    const std::vector<IndexSpace> &lhs,
    const std::vector<IndexSpace> &rhs,
    legion_terra_logical_region_list_t *rhs_,
    legion_terra_logical_region_list_list_t *result_)
{
  assert(DIM > 0); // Should be structured.

  std::vector<Rect<DIM> > lh_rects, rh_rects;
  extract_ispace_domain_rects<DIM>(runtime, ctx, lhs, lh_rects);
  extract_ispace_domain_rects<DIM>(runtime, ctx, rhs, rh_rects);

#define BLOCK_SIZE 512
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  for (size_t block_i = 0; block_i < lhs_size; block_i += BLOCK_SIZE) {
    for (size_t block_j = 0; block_j < rhs_size; block_j += BLOCK_SIZE) {
      size_t block_i_max = std::min(block_i + BLOCK_SIZE, lhs_size);
      size_t block_j_max = std::min(block_j + BLOCK_SIZE, rhs_size);
      for (size_t i = block_i; i < block_i_max; i++) {
        Rect<DIM> lh_rect = lh_rects[i];
        for (size_t j = block_j; j < block_j_max; j++) {
          Rect<DIM> rh_rect = rh_rects[j];
          if (lh_rect.overlaps(rh_rect)) {
            legion_terra_logical_region_list_t& sublist = result_->sublists[i];
            size_t idx = sublist.count++;
            sublist.subregions[idx].index_space = CObjectWrapper::wrap(rhs[j]);
            sublist.subregions[idx].field_space = rhs_->subregions[j].field_space;
            sublist.subregions[idx].tree_id = rhs_->subregions[j].tree_id;
          }
        }
      }
    }
  }
}

// Takes the "shallow" cross product between lists of structured index spaces
// `lhs` and `rhs`.  Specifically, if `lhs[i]` and `rhs[j]` intersect,
// `result[i][j]` is populated with `rhs[j]`.
static void
create_cross_product_shallow_structured(HighLevelRuntime *runtime,
                                        Context ctx,
                                        const std::vector<IndexSpace> &lhs,
                                        const std::vector<IndexSpace> &rhs,
                                        legion_terra_logical_region_list_t *rhs_,
                                        legion_terra_logical_region_list_list_t *result_)
{
  if (lhs.empty() || rhs.empty()) return;

  int dim = runtime->get_index_space_domain(ctx, lhs[0]).get_dim();
  switch (dim) {
    case 0:
      assert(false); // Index space should be structured.
    case 1:
      {
        create_cross_product_shallow_structured_spec<1>(
            runtime, ctx, lhs, rhs, rhs_, result_);
        break;
      }
    case 2:
      {
        create_cross_product_shallow_structured_spec<2>(
            runtime, ctx, lhs, rhs, rhs_, result_);
        break;
      }
    case 3:
      {
        create_cross_product_shallow_structured_spec<3>(
            runtime, ctx, lhs, rhs, rhs_, result_);
        break;
      }
    default:
      assert(false);
  }
}

// Takes the shallow cross product between lists of unstructured index spaces.
static void
create_cross_product_shallow_unstructured(HighLevelRuntime *runtime,
                                          Context ctx,
                                          bool flip,
                                          const std::vector<IndexSpace> &lhs,
                                          bool lhs_disjoint,
                                          const std::vector<IndexSpace> &rhs,
                                          bool rhs_disjoint,
                                          legion_terra_logical_region_list_t *lhs_,
                                          legion_terra_logical_region_list_t *rhs_,
                                          legion_terra_logical_region_list_list_t *result_)
{
  if (lhs_disjoint)
  {
    create_cross_product_tree(runtime, ctx, flip, lhs, lhs_disjoint, rhs, rhs_disjoint, lhs_, rhs_, result_);
    return;
  }
  else if (rhs_disjoint)
  {
    create_cross_product_tree(runtime, ctx, !flip, rhs, rhs_disjoint, lhs, lhs_disjoint, rhs_, lhs_, result_);
    return;
  }

  std::vector<std::pair<ptr_t, ptr_t> > lhs_bounds;
  std::vector<std::pair<ptr_t, ptr_t> > rhs_bounds;

  get_bounding_boxes(runtime, ctx, lhs, lhs_bounds);
  get_bounding_boxes(runtime, ctx, rhs, rhs_bounds);

  for (size_t i = 0; i < lhs.size(); i++) {
    IndexSpace lh_space = lhs[i];
    std::pair<ptr_t, ptr_t> lh_bound = lhs_bounds[i];
    for (size_t j = 0; j < rhs.size(); j++) {
      IndexSpace rh_space = rhs[j];
      std::pair<ptr_t, ptr_t> rh_bound = rhs_bounds[j];
      if (lh_bound.second.value < rh_bound.first.value ||
          rh_bound.second.value < lh_bound.first.value) {
        continue;
      }

      bool intersects = false;
      for (IndexIterator lh_it(runtime, ctx, lh_space); lh_it.has_next();) {
        size_t lh_count = 0;
        ptr_t lh_ptr = lh_it.next_span(lh_count);
        ptr_t lh_end = lh_ptr.value + lh_count - 1;

        IndexIterator rh_it(runtime, ctx, rh_space, lh_ptr);
        if (rh_it.has_next()) {
          size_t rh_count = 0;
          ptr_t rh_ptr = rh_it.next_span(rh_count);

          if (rh_ptr.value <= lh_end.value) {
            intersects = true;
            break;
          }
        }
      }
      if (intersects) {
        if (flip) {
          legion_terra_logical_region_list_t& sublist = result_->sublists[j];
          size_t idx = sublist.count++;
          sublist.subregions[idx].index_space = CObjectWrapper::wrap(lh_space);
          sublist.subregions[idx].field_space = lhs_->subregions[i].field_space;
          sublist.subregions[idx].tree_id = lhs_->subregions[i].tree_id;
        }
        else {
          legion_terra_logical_region_list_t& sublist = result_->sublists[i];
          size_t idx = sublist.count++;
          sublist.subregions[idx].index_space = CObjectWrapper::wrap(rh_space);
          sublist.subregions[idx].field_space = rhs_->subregions[j].field_space;
          sublist.subregions[idx].tree_id = rhs_->subregions[j].tree_id;
        }
      }
    }
  }
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
    assert(color != Color(-1));
    if (desired_color == Color(-1)) {
      assert(result_colors[level] == Color(-1));
      result_colors[level] = color;
    } else {
      assert(color == desired_color);
    }

    Domain colors = runtime->get_index_partition_color_space(ctx, next);
    for (Domain::DomainPointIterator dp(colors); dp; dp++) {
      const DomainPoint &next_color = dp.p;
      IndexSpace is = runtime->get_index_subspace(ctx, next, next_color);
      IndexPartition ip = runtime->get_index_partition(ctx, is, color);
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

static IndexSpace
unwrap_list(legion_terra_index_space_list_t list,
            std::vector<IndexSpace> &subspaces)
{
  subspaces.reserve(list.count);
  for (size_t i = 0; i < list.count; i++) {
    subspaces.push_back(CObjectWrapper::unwrap(list.subspaces[i]));
  }
  return CObjectWrapper::unwrap(list.space);
}

static void
unwrap_list(legion_terra_logical_region_list_t *list,
            std::vector<IndexSpace> &subspaces)
{
  subspaces.reserve(list->count);
  for (size_t i = 0; i < list->count; i++) {
    subspaces.push_back(
      CObjectWrapper::unwrap(list->subregions[i]).get_index_space());
  }
}

static void
unwrap_list_list(legion_terra_index_space_list_list_t list,
                 std::map<IndexSpace, std::vector<IndexSpace> > &product)
{
  for (size_t i = 0; i < list.count; i++) {
    std::vector<IndexSpace> subspaces;
    IndexSpace space = unwrap_list(list.sublists[i], subspaces);
    assert(space != IndexSpace::NO_SPACE);
    product[space] = subspaces;
  }
}

static void
wrap_list_list(std::vector<IndexSpace> &lhs,
               legion_terra_logical_region_list_t *rhs_,
               std::map<IndexSpace, std::vector<IndexSpace> > &product,
               legion_terra_logical_region_list_list_t *result)
{
  assert(result->count == lhs.size());
  for (size_t i = 0; i < lhs.size(); i++) {
    IndexSpace space = lhs[i];
    assert(product.count(space));
    std::vector<IndexSpace> &subspaces = product[space];

    legion_terra_logical_region_list_t& sublist = result->sublists[i];

    for (size_t j = 0; j < subspaces.size(); j++) {
      if (subspaces[j].exists()) {
        legion_index_space_t subspace = CObjectWrapper::wrap(subspaces[j]);
        size_t idx = sublist.count++;
        sublist.subregions[idx].index_space = subspace;
        sublist.subregions[idx].field_space = rhs_->subregions[j].field_space;
        sublist.subregions[idx].tree_id = rhs_->subregions[j].tree_id;
      }
    }
  }
}

static IndexPartition
partition_from_list(HighLevelRuntime *runtime, Context ctx,
                    const std::vector<IndexSpace> &subspaces)
{
  if (subspaces.empty()) return IndexPartition::NO_PART;
  assert(runtime->has_parent_index_partition(ctx, subspaces[0]));
  return runtime->get_parent_index_partition(ctx, subspaces[0]);
}

void
legion_terra_index_cross_product_create_list(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_terra_logical_region_list_t *lhs_,
  legion_terra_logical_region_list_t *rhs_,
  legion_terra_logical_region_list_list_t *result_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  std::vector<IndexSpace> lhs;
  unwrap_list(lhs_, lhs);
  std::vector<IndexSpace> rhs;
  unwrap_list(rhs_, rhs);

  IndexPartition lhs_part = partition_from_list(runtime, ctx, lhs);
  IndexPartition rhs_part = partition_from_list(runtime, ctx, rhs);

  Color sub_color = -1;
  if (lhs_part != IndexPartition::NO_PART && rhs_part != IndexPartition::NO_PART) {
    sub_color = create_cross_product(runtime, ctx, rhs_part, lhs_part);
  }

  std::map<IndexSpace, std::vector<IndexSpace> > product;
  for (std::vector<IndexSpace>::iterator it = rhs.begin(); it != rhs.end(); ++it) {
    IndexSpace rh_space = *it;

    for (std::vector<IndexSpace>::iterator it = lhs.begin(); it != lhs.end(); ++it) {
      IndexSpace lh_space = *it;
      Color lh_color = runtime->get_index_space_color(ctx, lh_space);
      IndexPartition rh_subpart = runtime->get_index_partition(ctx, rh_space, sub_color);
      IndexSpace rh_subspace = runtime->get_index_subspace(ctx, rh_subpart, lh_color);

      IndexIterator rh_it(runtime, ctx, rh_subspace);
      if (rh_it.has_next()) {
        product[lh_space].push_back(rh_subspace);
      } else {
        product[lh_space].push_back(IndexSpace::NO_SPACE);
      }
    }
  }

  wrap_list_list(lhs, rhs_, product, result_);
}

void
legion_terra_index_cross_product_create_list_shallow(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_terra_logical_region_list_t *lhs_,
  legion_terra_logical_region_list_t *rhs_,
  legion_terra_logical_region_list_list_t *result_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  std::vector<IndexSpace> lhs;
  unwrap_list(lhs_, lhs);
  std::vector<IndexSpace> rhs;
  unwrap_list(rhs_, rhs);

  for (size_t i = 0; i < result_->count; ++i) result_->sublists[i].count = 0;

  IndexPartition lhs_part = partition_from_list(runtime, ctx, lhs);
  IndexPartition rhs_part = partition_from_list(runtime, ctx, rhs);
  if ((lhs_part != IndexPartition::NO_PART && is_structured(runtime, ctx, lhs_part))
      || (rhs_part != IndexPartition::NO_PART && is_structured(runtime, ctx, rhs_part))) {
      // Structured index spaces.
    create_cross_product_shallow_structured(runtime, ctx, lhs, rhs, rhs_, result_);
  } else { // Unstructured index spaces.
    bool lhs_disjoint = runtime->is_index_partition_disjoint(ctx, lhs_part);
    bool rhs_disjoint = runtime->is_index_partition_disjoint(ctx, rhs_part);

    bool flip = false;
    if (lhs_part != IndexPartition::NO_PART && rhs_part != IndexPartition::NO_PART) {
      flip = should_flip_cross_product(runtime, ctx, lhs_part, rhs_part);
    }

    if (flip) {
      create_cross_product_shallow_unstructured(runtime, ctx, flip, rhs, rhs_disjoint, lhs, lhs_disjoint, rhs_, lhs_, result_);
    } else {
      create_cross_product_shallow_unstructured(runtime, ctx, flip, lhs, lhs_disjoint, rhs, rhs_disjoint, lhs_, rhs_, result_);
    }
  }
}

// "Completes" a shallow cross product between lists of structured index spaces.
// After a shallow cross product determines which index spaces intersect, this
// function actually creates partitions corresponding to the intersections.
//
// Specifically, this function partitions each "RHS" index space according to
// the LHS index spaces it intersects.
//
// Params:
//    lhs: list of left-hand side index spaces.
//    lhs_part_disjoint: true if the LHS partition is disjoint.
//    product: maps each LHS index space to the list of RHS index spaces that
//             intersect it.
//    result: `result[i][j]` is populated with the IndexSpace that is the
//            intersection of `lhs[i]` and `product[lhs[i]][j]`.  This
//            IndexSpace belongs to a Partition of the RHS IndexSpace.
static inline void
create_cross_product_complete_structured(
    HighLevelRuntime *runtime,
    Context ctx,
    std::vector<IndexSpace>& lhs,
    bool lhs_part_disjoint,
    std::map<IndexSpace, std::vector<IndexSpace> >& product,
    legion_terra_index_space_list_list_t &result)
{
  std::vector<DomainPoint> lhs_colors;
  lhs_colors.reserve(lhs.size());
  for (unsigned lhs_idx = 0; lhs_idx < lhs.size(); ++lhs_idx) {
    IndexSpace& lh_space = lhs[lhs_idx];
    DomainPoint lh_color = runtime->get_index_space_color_point(ctx, lh_space);
    lhs_colors.push_back(lh_color);
  }

  std::map<IndexSpace, DomainPointColoring> coloring; // Partition coloring for each RHS ispace.
  std::map<IndexSpace, Domain> color_spaces; // Color space for each partitioning.
  for (unsigned lhs_idx = 0; lhs_idx < lhs.size(); ++lhs_idx) {
    IndexSpace lh_space = lhs[lhs_idx];
    // Doesn't currently handle structured index spaces consisting of multiple
    // domains.
    assert(!runtime->has_multiple_domains(lh_space));
    Domain lh_domain = runtime->get_index_space_domain(lh_space);
    // Verify that index space is indeed structured.
    assert(lh_domain.get_dim() > 0);

    const std::vector<IndexSpace>& rh_spaces = product[lh_space];
    DomainPoint lh_color = lhs_colors[lhs_idx];

    for (unsigned rhs_idx = 0; rhs_idx < rh_spaces.size(); ++rhs_idx) {
      IndexSpace rh_space = rh_spaces[rhs_idx];
      assert(!runtime->has_multiple_domains(rh_space));
      Domain rh_domain = runtime->get_index_space_domain(rh_space);
      assert(rh_domain.get_dim() > 0);

      coloring[rh_space][lh_color] = rh_domain.intersection(lh_domain);
      if (color_spaces.count(rh_space) > 0) {
        color_spaces[rh_space] =
          color_spaces[rh_space].convex_hull(lh_color);
      } else {
        color_spaces[rh_space] = Domain::from_domain_point(lh_color);
      }
    }
  }

  std::map<IndexSpace, IndexPartition> rh_partitions;
  for (std::map<IndexSpace, DomainPointColoring>::iterator it = coloring.begin();
       it != coloring.end(); ++it) {

    IndexSpace rh_space = it->first;
    const DomainPointColoring& coloring = it->second;
    assert(color_spaces.count(rh_space) > 0);

    IndexPartition ip = runtime->create_index_partition(
        ctx, /* parent = */ rh_space, /* color_space = */ color_spaces[rh_space],
        coloring, lhs_part_disjoint ? DISJOINT_KIND : ALIASED_KIND);
    rh_partitions[rh_space] = ip;
  }

  for (unsigned lhs_idx = 0; lhs_idx < lhs.size(); ++lhs_idx) {
    IndexSpace lh_space = lhs[lhs_idx];
    std::vector<IndexSpace>& rh_spaces = product[lh_space];
    DomainPoint lh_color = lhs_colors[lhs_idx];

    for (unsigned rhs_idx = 0; rhs_idx < rh_spaces.size(); ++rhs_idx) {
      IndexSpace& rh_space = rh_spaces[rhs_idx];
      IndexPartition& rh_partition = rh_partitions[rh_space];
      IndexSpace is = runtime->get_index_subspace(ctx, rh_partition, lh_color);
      assert(lhs_idx >= 0 && lhs_idx < result.count);
      assert(rhs_idx >= 0 && rhs_idx < result.sublists[lhs_idx].count);
      assign_list_list(result, lhs_idx, rhs_idx, CObjectWrapper::wrap(is));
    }
  }
}

// Completes the shallow cross product between two lists of unstructured index spaces.
static inline void
create_cross_product_complete_unstructured(
    HighLevelRuntime *runtime,
    Context ctx,
    std::vector<IndexSpace>& lhs,
    bool lhs_part_disjoint,
    std::map<IndexSpace, std::vector<IndexSpace> >& product,
    legion_terra_index_space_list_list_t &result)
{
  std::vector<DomainPoint> lhs_colors;
  lhs_colors.reserve(lhs.size());
  for (unsigned lhs_idx = 0; lhs_idx < lhs.size(); ++lhs_idx) {
    IndexSpace& lh_space = lhs[lhs_idx];
    DomainPoint lh_color = runtime->get_index_space_color_point(ctx, lh_space);
    lhs_colors.push_back(lh_color);
  }

  std::map<IndexSpace, PointColoring> coloring;
  std::map<IndexSpace, Domain> color_spaces;
  for (unsigned lhs_idx = 0; lhs_idx < lhs.size(); ++lhs_idx) {
    IndexSpace& lh_space = lhs[lhs_idx];
    std::vector<IndexSpace>& rh_spaces = product[lh_space];
    DomainPoint lh_color = lhs_colors[lhs_idx];

    for (unsigned rhs_idx = 0; rhs_idx < rh_spaces.size(); ++rhs_idx) {
      IndexSpace& rh_space = rh_spaces[rhs_idx];

      coloring[rh_space][lh_color];
      for (IndexIterator rh_it(runtime, ctx, rh_space); rh_it.has_next();) {
        size_t rh_count = 0;
        ptr_t rh_ptr = rh_it.next_span(rh_count);
        ptr_t rh_end = rh_ptr.value + rh_count - 1;

        for (IndexIterator lh_it(runtime, ctx, lh_space, rh_ptr); lh_it.has_next();) {
          size_t lh_count = 0;
          ptr_t lh_ptr = lh_it.next_span(lh_count);
          ptr_t lh_end = lh_ptr.value + lh_count - 1;

          if (lh_ptr.value > rh_end.value) {
            break;
          }

          if (color_spaces.count(rh_space) > 0) {
            color_spaces[rh_space] =
              color_spaces[rh_space].convex_hull(lh_color);
          } else {
            color_spaces[rh_space] = Domain::from_domain_point(lh_color);
          }

          if (lh_end.value > rh_end.value) {
            coloring[rh_space][lh_color].ranges.insert(std::pair<ptr_t, ptr_t>(lh_ptr, rh_end));
            break;
          }

          coloring[rh_space][lh_color].ranges.insert(std::pair<ptr_t, ptr_t>(lh_ptr, lh_end));
        }
      }
    }
  }

  std::map<IndexSpace, IndexPartition> rh_partitions;
  for (std::map<IndexSpace, PointColoring>::iterator it = coloring.begin();
       it != coloring.end(); ++it) {
    IndexSpace rh_space = it->first;
    const PointColoring& coloring = it->second;
    assert(color_spaces.count(rh_space) > 0);

    IndexPartition ip = runtime->create_index_partition(
        ctx, /* parent = */ rh_space, /* color_space = */ color_spaces[rh_space],
        coloring, lhs_part_disjoint ? DISJOINT_KIND : ALIASED_KIND);
    rh_partitions[it->first] = ip;
  }

  for (unsigned lhs_idx = 0; lhs_idx < lhs.size(); ++lhs_idx) {
    IndexSpace& lh_space = lhs[lhs_idx];
    std::vector<IndexSpace>& rh_spaces = product[lh_space];
    DomainPoint lh_color = lhs_colors[lhs_idx];

    for (unsigned rhs_idx = 0; rhs_idx < rh_spaces.size(); ++rhs_idx) {
      IndexSpace& rh_space = rh_spaces[rhs_idx];
      IndexPartition& rh_partition = rh_partitions[rh_space];
      IndexSpace is = runtime->get_index_subspace(ctx, rh_partition, lh_color);
      assert(lhs_idx >= 0 && lhs_idx < result.count);
      assert(rhs_idx >= 0 && rhs_idx < result.sublists[lhs_idx].count);
      assign_list_list(result, lhs_idx, rhs_idx, CObjectWrapper::wrap(is));
    }
  }
}

legion_terra_index_space_list_list_t
legion_terra_index_cross_product_create_list_complete(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_terra_index_space_list_t lhs_,
  legion_terra_index_space_list_list_t product_,
  bool consistent_ids)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  std::vector<IndexSpace> lhs;
  unwrap_list(lhs_, lhs);
  std::map<IndexSpace, std::vector<IndexSpace> > product;
  unwrap_list_list(product_, product);

  legion_terra_index_space_list_list_t result = create_list_list(lhs, product);
  IndexPartition lhs_part = partition_from_list(runtime, ctx, lhs);
  bool lhs_part_disjoint = runtime->is_index_partition_disjoint(ctx, lhs_part);
  if (lhs_part != IndexPartition::NO_PART && is_structured(runtime, ctx, lhs_part)) {
    // Structured index spaces.
    create_cross_product_complete_structured(runtime, ctx, lhs, lhs_part_disjoint,
        product, result);
  } else { // Unstructured index spaces.
    create_cross_product_complete_unstructured(runtime, ctx, lhs, lhs_part_disjoint,
        product, result);
  }
  return result;
}

void
legion_terra_index_space_list_list_destroy(
  legion_terra_index_space_list_list_t list)
{
  for (size_t i = 0; i < list.count; i++) {
    free(list.sublists[i].subspaces);
  }
  free(list.sublists);
}

legion_index_partition_t
legion_terra_index_cross_product_get_partition(
  legion_terra_index_cross_product_t prod)
{
  return prod.partition;
}

legion_index_partition_t
legion_terra_index_cross_product_get_subpartition_by_color_domain_point(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_terra_index_cross_product_t prod,
  legion_domain_point_t color_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition partition = CObjectWrapper::unwrap(prod.partition);
  DomainPoint color = CObjectWrapper::unwrap(color_);

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

  assert(count);
  ptr_t result = handle->next_span(*count);
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
