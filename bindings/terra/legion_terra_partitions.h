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

#ifndef __LEGION_TERRA_PARTITIONS_H__
#define __LEGION_TERRA_PARTITIONS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "legion_c.h"

typedef struct legion_terra_index_cross_product_t {
  void *impl;
} legion_terra_index_cross_product_t;

legion_terra_index_cross_product_t
legion_terra_index_cross_product_create(
  legion_runtime_t runtime,
  legion_context_t ctx,
  legion_index_partition_t lhs,
  legion_index_partition_t rhs);

legion_index_partition_t
legion_terra_index_cross_product_get_partition(
  legion_terra_index_cross_product_t prod);

legion_index_partition_t
legion_terra_index_cross_product_get_subpartition_by_color(
  legion_terra_index_cross_product_t prod,
  legion_color_t color);

#ifdef __cplusplus
}
#endif

#endif // __LEGION_TERRA_PARTITIONS_H__
