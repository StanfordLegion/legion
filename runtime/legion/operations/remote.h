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

#ifndef __LEGION_REMOTE_OPERATION_H__
#define __LEGION_REMOTE_OPERATION_H__

#include "legion/operations/operation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RemoteOp
     * This operation is a shim for operations on remote nodes
     * and is used by remote physical analysis traversals to handle
     * any requests they might have of the original operation.
     */
    class RemoteOp : public Operation {
    public:
      struct DeferRemoteOpDeletionArgs
        : public LgTaskArgs<DeferRemoteOpDeletionArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_REMOTE_OP_DELETION_TASK_ID;
      public:
        DeferRemoteOpDeletionArgs(void) = default;
        DeferRemoteOpDeletionArgs(Operation* o)
          : LgTaskArgs<DeferRemoteOpDeletionArgs>(true, true), op(o)
        { }
        void execute(void) const;
      public:
        Operation* op;
      };
    public:
      RemoteOp(Operation* ptr, AddressSpaceID src);
      RemoteOp(const RemoteOp& rhs) = delete;
      virtual ~RemoteOp(void);
    public:
      RemoteOp& operator=(const RemoteOp& rhs) = delete;
    public:
      virtual void unpack(Deserializer& derez) = 0;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override = 0;
      virtual OpKind get_operation_kind(void) const override = 0;
      virtual Operation* get_origin_operation(void) override
      {
        std::abort();
      }  // should never be called on remote ops
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override;
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      virtual void report_uninitialized_usage(
          const unsigned index, const char* field_string,
          RtUserEvent reported) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override = 0;
    public:
      void defer_deletion(RtEvent precondition);
      void pack_remote_base(Serializer& rez) const;
      void unpack_remote_base(Deserializer& derez);
      void pack_profiling_requests(
          Serializer& rez, std::set<RtEvent>& applied) const;
      void unpack_profiling_requests(Deserializer& derez);
      // Caller takes ownership of this object and must delete it when done
      static RemoteOp* unpack_remote_operation(Deserializer& derez);
    public:
      virtual void record_completion_effect(ApEvent effect) override;
      virtual void record_completion_effect(
          ApEvent effect, std::set<RtEvent>& map_applied_events) override;
      virtual void record_completion_effects(
          const std::set<ApEvent>& effects) override;
      virtual void record_completion_effects(
          const std::vector<ApEvent>& effects) override;
    public:
      // This is a pointer to an operation on a remote node
      // it should never be dereferenced
      Operation* const remote_ptr;
      const AddressSpaceID source;
    protected:
      MapperManager* mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int profiling_priority;
      int copy_fill_priority;
      Processor profiling_target;
      RtUserEvent profiling_response;
      std::atomic<int> profiling_reports;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REMOTE_OPERATION_H__
