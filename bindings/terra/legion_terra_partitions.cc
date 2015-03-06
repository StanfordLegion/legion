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

#include "arrays.h"
#include "legion.h"
#include "legion_c_util.h"
#include "lowlevel.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::LowLevel;

struct IndexCrossProduct {
  IndexPartition partition;
  std::map<Color, IndexPartition> subpartitions;
};

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

  NEW_OPAQUE_WRAPPER(legion_terra_index_cross_product_t, IndexCrossProduct *);
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

  IndexCrossProduct *prod = new IndexCrossProduct();
  prod->partition = lhs;
  std::map<Color, IndexPartition> &subpartitions = prod->subpartitions;

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

    IndexPartition lhs_subpartition = runtime->create_index_partition(
      ctx, lhs_space, lhs_coloring,
      runtime->is_index_partition_disjoint(ctx, rhs));
    subpartitions[lhs_color] = lhs_subpartition;
  }

  return TerraCObjectWrapper::wrap(prod);
}

legion_index_partition_t
legion_terra_index_cross_product_get_partition(
  legion_terra_index_cross_product_t prod_)
{
  IndexCrossProduct *prod = TerraCObjectWrapper::unwrap(prod_);
  return prod->partition;
}

legion_index_partition_t
legion_terra_index_cross_product_get_subpartition_by_color(
  legion_terra_index_cross_product_t prod_,
  legion_color_t color)
{
  IndexCrossProduct *prod = TerraCObjectWrapper::unwrap(prod_);
  return prod->subpartitions[color];
}
