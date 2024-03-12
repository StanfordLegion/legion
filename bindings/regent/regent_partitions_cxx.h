/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#ifndef __REGENT_PARTITIONS_CXX_H__
#define __REGENT_PARTITIONS_CXX_H__

// C++ declarations for regent_partitions.cc

#include "legion.h"

using namespace Legion;

/**
 * Creates the cross product between two IndexPartition's.
 * @param lhs the left-hand side partition, whose subspaces are to be
 *        sub-partitioned to represent the cross product
 * @param rhs the right-hand side partition, within the same index tree as lhs
 * @param rhs_color the Color to assign to partitions in the cross product; if
 *        equals AUTO_GENERATE_ID, suitable Color(s) will be chosen (see \p consistent_ids)
 * @param consistent_ids if set, all new partitions in the cross product will
 *        be assigned the same Color, even when `rhs_color` is AUTO_GENERATE_ID
 * @param[out] chosen_colors if non-NULL, will be populated so that each
 *        subspace in \p lhs is associated with the Color assigned to its
 *        corresponding partition in the cross product
 * @param lhs_filter,rhs_filter if non-NULL, only subspaces with Colors
 *        included in the filters will be included in the cross product
 * @return the Color assigned to the newly created partitions of the cross
 *         product, if all partitions are assigned the same Color (i.e. if
 *         \p rhs_color is AUTO_GENERATE_ID or \p consistent_ids is set); AUTO_GENERATE_ID otherwise
 */
Color
create_cross_product(Runtime *runtime,
                     Context ctx,
                     IndexPartition lhs,
                     IndexPartition rhs,
                     Color rhs_color = AUTO_GENERATE_ID,
                     bool consistent_ids = true,
                     std::map<IndexSpace, Color> *chosen_colors = NULL,
                     const std::set<DomainPoint> *lhs_filter = NULL,
                     const std::set<DomainPoint> *rhs_filter = NULL);

#endif // __REGENT_PARTITIONS_CXX_H__
