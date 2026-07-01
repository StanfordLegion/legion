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

#ifndef __LEGION_CREATION_H__
#define __LEGION_CREATION_H__

#include "legion/operations/operation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class CreationOp
     * A creation operation is used for deferring the creation of
     * an particular resource until some event has transpired such
     * as the resolution of a future.
     */
    class CreationOp : public Operation {
    public:
      enum CreationKind {
        INDEX_SPACE_CREATION,
        FIELD_ALLOCATION,
        FUTURE_MAP_CREATION,
      };
    public:
      CreationOp(void);
      CreationOp(const CreationOp& rhs) = delete;
      virtual ~CreationOp(void);
    public:
      CreationOp& operator=(const CreationOp& rhs) = delete;
    public:
      void initialize_index_space(
          InnerContext* ctx, IndexSpaceNode* node, const Future& future,
          Provenance* provenance, bool owner = true,
          const CollectiveMapping* mapping = nullptr);
      void initialize_field(
          InnerContext* ctx, FieldSpaceNode* node, FieldID fid,
          const Future& field_size, Provenance* provenance, bool owner = true);
      void initialize_fields(
          InnerContext* ctx, FieldSpaceNode* node,
          const std::vector<FieldID>& fids,
          const std::vector<Future>& field_sizes, Provenance* provenance,
          bool owner = true);
      void initialize_map(
          InnerContext* ctx, Provenance* provenance,
          const std::map<DomainPoint, Future>& futures);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        return false;
      }
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_execution(void) override;
    protected:
      CreationKind kind;
      IndexSpaceNode* index_space_node;
      FieldSpaceNode* field_space_node;
      std::vector<Future> futures;
      std::vector<FieldID> fields;
      const CollectiveMapping* mapping;
      bool owner;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_CREATION_H__
