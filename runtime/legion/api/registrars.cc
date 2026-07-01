/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "legion/api/registrars.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // LayoutConstraintRegistrar
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  LayoutConstraintRegistrar::LayoutConstraintRegistrar(void)
    : handle(FieldSpace::NO_SPACE), layout_name(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  LayoutConstraintRegistrar::LayoutConstraintRegistrar(
      FieldSpace h, const char* layout /*= nullptr*/)
    : handle(h), layout_name(layout)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // TaskVariantRegistrar
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  TaskVariantRegistrar::TaskVariantRegistrar(void)
    : task_id(0), global_registration(true), task_variant_name(nullptr),
      leaf_variant(false), inner_variant(false), idempotent_variant(false),
      replicable_variant(false), concurrent_variant(false),
      concurrent_barrier(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  TaskVariantRegistrar::TaskVariantRegistrar(
      TaskID task_id, bool global, const char* variant_name)
    : task_id(task_id), global_registration(global),
      task_variant_name(variant_name), leaf_variant(false),
      inner_variant(false), idempotent_variant(false),
      replicable_variant(false), concurrent_variant(false),
      concurrent_barrier(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  TaskVariantRegistrar::TaskVariantRegistrar(
      TaskID task_id, const char* variant_name, bool global /*=true*/)
    : task_id(task_id), global_registration(global),
      task_variant_name(variant_name), leaf_variant(false),
      inner_variant(false), idempotent_variant(false),
      replicable_variant(false), concurrent_variant(false),
      concurrent_barrier(false)
  //--------------------------------------------------------------------------
  { }

}  // namespace Legion
