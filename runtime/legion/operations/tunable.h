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

#ifndef __LEGION_TUNABLE_H__
#define __LEGION_TUNABLE_H__

#include "legion/operations/operation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TunableOp
     * Operation for performing tunable requests
     */
    class TunableOp : public Operation {
    public:
      TunableOp(void);
      TunableOp(const TunableOp& rhs) = delete;
      virtual ~TunableOp(void);
    public:
      TunableOp& operator=(const TunableOp& rhs) = delete;
    public:
      Future initialize(
          InnerContext* ctx, const TunableLauncher& launcher,
          Provenance* provenance);
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
      // virtual method for control replication
      virtual void process_result(
          MapperManager* mapper, void* buffer, size_t size) const
      { }
    protected:
      TunableID tunable_id;
      MapperID mapper_id;
      MappingTagID tag;
      void* arg;
      size_t argsize;
      uint64_t tunable_index;
      size_t return_type_size;
      Future result;
      FutureInstance* instance;
      std::vector<Future> futures;
      RtEvent futures_mapped;
    };

    /**
     * \class ReplTunableOp
     * A tunable operation that is aware that it is
     * being executed in a control replicated context
     */
    class ReplTunableOp : public TunableOp {
    public:
      ReplTunableOp(void);
      ReplTunableOp(const ReplTunableOp& rhs) = delete;
      virtual ~ReplTunableOp(void);
    public:
      ReplTunableOp& operator=(const ReplTunableOp& rhs) = delete;
    public:
      void initialize_replication(ReplicateContext* context);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void process_result(
          MapperManager* mapper, void* buffer, size_t size) const override;
    protected:
      BufferBroadcast* value_broadcast;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TUNABLE_H__
