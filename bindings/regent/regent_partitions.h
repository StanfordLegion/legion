/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef __REGENT_PARTITIONS_H__
#define __REGENT_PARTITIONS_H__

// C declarations for regent_partitions.cc

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++.
//
// ******************** IMPORTANT **************************

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct legion_terra_index_cross_product_t {
  legion_index_partition_t partition;
  legion_color_t other_color;
} legion_terra_index_cross_product_t;

typedef struct legion_terra_index_space_list_t {
  legion_index_space_t space;
  size_t count;
  legion_index_space_t *subspaces;
} legion_terra_index_space_list_t;

typedef struct legion_terra_index_space_list_list_t {
  size_t count;
  legion_terra_index_space_list_t *sublists;
} legion_terra_index_space_list_list_t;

typedef struct legion_terra_logical_region_list_t {
  size_t count;
  legion_logical_region_t *subregions;
} legion_terra_logical_region_list_t;

typedef struct legion_terra_logical_region_list_list_t {
  size_t count;
  legion_terra_logical_region_list_t *sublists;
} legion_terra_logical_region_list_list_t;

legion_terra_index_cross_product_t
legion_terra_index_cross_product_create(
  legion_runtime_t runtime,
  legion_context_t ctx,
  legion_index_partition_t lhs,
  legion_index_partition_t rhs);

legion_terra_index_cross_product_t
legion_terra_index_cross_product_create_multi(
  legion_runtime_t runtime,
  legion_context_t ctx,
  legion_index_partition_t *partitions, // input
  legion_color_t *colors, // output
  size_t npartitions);

legion_terra_index_cross_product_t
legion_terra_index_cross_product_import(
  legion_index_partition_t first_partition,
  legion_color_t second_color);

void
legion_terra_index_cross_product_create_list(
  legion_runtime_t runtime,
  legion_context_t ctx,
  legion_terra_logical_region_list_t *lhs,
  legion_terra_logical_region_list_t *rhs,
  legion_terra_logical_region_list_list_t* result);

void
legion_terra_index_cross_product_create_list_shallow(
  legion_runtime_t runtime,
  legion_context_t ctx,
  legion_terra_logical_region_list_t *lhs,
  legion_terra_logical_region_list_t *rhs,
  legion_terra_logical_region_list_list_t* result);

legion_terra_index_space_list_list_t
legion_terra_index_cross_product_create_list_complete(
  legion_runtime_t runtime,
  legion_context_t ctx,
  legion_terra_index_space_list_t lhs,
  legion_terra_index_space_list_list_t product,
  bool consistent_ids);

void
legion_terra_index_space_list_list_destroy(
  legion_terra_index_space_list_list_t list);

legion_index_partition_t
legion_terra_index_cross_product_get_partition(
  legion_terra_index_cross_product_t prod);

legion_index_partition_t
legion_terra_index_cross_product_get_subpartition_by_color_domain_point(
  legion_runtime_t runtime,
  legion_terra_index_cross_product_t prod,
  legion_domain_point_t color_);

unsigned
legion_terra_task_launcher_get_region_requirement_logical_region(
  legion_task_launcher_t launcher,
  legion_logical_region_t region);

bool
legion_terra_task_launcher_has_field(
  legion_task_launcher_t launcher,
  unsigned idx,
  legion_field_id_t fid);

#ifdef __cplusplus
}
#endif

#endif // __REGENT_PARTITIONS_H__
