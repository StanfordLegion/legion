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

#ifndef __LEGION_DATA_IMPL_H__
#define __LEGION_DATA_IMPL_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/api/data.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FieldAllocatorImpl
     * The base implementation of a field allocator object. This
     * tracks how many outstanding copies of a field allocator
     * object there are for a task and once they've all been
     * destroyed it informs the context that there are no more
     * outstanding allocations.
     */
    class FieldAllocatorImpl : public Collectable {
    public:
      FieldAllocatorImpl(
          FieldSpaceNode* node, TaskContext* context, RtEvent ready);
      FieldAllocatorImpl(const FieldAllocatorImpl& rhs) = delete;
      ~FieldAllocatorImpl(void);
    public:
      FieldAllocatorImpl& operator=(const FieldAllocatorImpl& rhs) = delete;
    public:
      inline FieldSpace get_field_space(void) const { return field_space; }
    public:
      FieldID allocate_field(
          size_t field_size, FieldID desired_fieldid, CustomSerdezID serdez_id,
          bool local, Provenance* provenance);
      FieldID allocate_field(
          const Future& field_size, FieldID desired_fieldid,
          CustomSerdezID serdez_id, bool local, Provenance* provenance);
      void free_field(
          FieldID fid, const bool unordered, Provenance* provenance);
    public:
      void allocate_fields(
          const std::vector<size_t>& field_sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          bool local, Provenance* provenance);
      void allocate_fields(
          const std::vector<Future>& field_sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          bool local, Provenance* provenance);
      void free_fields(
          const std::set<FieldID>& to_free, const bool unordered,
          Provenance* provenance = nullptr);
    public:
      FieldSpace field_space;
      FieldSpaceNode* const node;
      TaskContext* const context;
      const RtEvent ready_event;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_DATA_IMPL_H__
