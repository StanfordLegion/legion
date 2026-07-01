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

#ifndef __LEGION_DISCARD_H__
#define __LEGION_DISCARD_H__

#include "legion/analysis/versioning.h"
#include "legion/operations/collective.h"
#include "legion/operations/remote.h"

namespace Legion {
  namespace Internal {

    /**
     * \class DiscardOp
     * Operation for reseting the state of fields back to an
     * uninitialized state like they were just created
     */
    class DiscardOp : public Operation {
    public:
      DiscardOp(void);
      DiscardOp(const DiscardOp& rhs) = delete;
      virtual ~DiscardOp(void);
    public:
      DiscardOp& operator=(const DiscardOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, const DiscardLauncher& launcher,
          Provenance* provenance);
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual unsigned find_parent_index(unsigned idx) override;
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual RtEvent finalize_complete_mapping(RtEvent event) { return event; }
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    protected:
      void discard_fields(const PhysicalTraceInfo& trace_info);
    public:
      RegionRequirement requirement;
      VersionInfo version_info;
      unsigned parent_req_index;
      std::set<RtEvent> map_applied_conditions;
    };

    /**
     * \class ReplDiscardOp
     * A discard operation that is aware that it is being
     * exected in a control replication context.
     */
    class ReplDiscardOp
      : public ReplCollectiveVersioning<CollectiveVersioning<DiscardOp> > {
    public:
      ReplDiscardOp(void);
      ReplDiscardOp(const ReplDiscardOp& rhs) = delete;
      virtual ~ReplDiscardOp(void);
    public:
      ReplDiscardOp& operator=(const ReplDiscardOp& rhs) = delete;
    public:
      void initialize_replication(ReplicateContext* ctx, bool is_first_local);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual RtEvent finalize_complete_mapping(RtEvent event) override;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
    protected:
      RtBarrier collective_map_barrier;
      bool is_first_local_shard;
    };

    /**
     * \class RemoteDiscardOp
     * This is a remote copy of a AttachOp to be used for
     * mapper calls and other operations
     */
    class RemoteDiscardOp
      : public RemoteOp,
        public Heapify<RemoteDiscardOp, OPERATION_LIFETIME> {
    public:
      RemoteDiscardOp(Operation* ptr, AddressSpaceID src);
      RemoteDiscardOp(const RemoteDiscardOp& rhs) = delete;
      virtual ~RemoteDiscardOp(void);
    public:
      RemoteDiscardOp& operator=(const RemoteDiscardOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual uint64_t get_context_index(void) const;
      virtual void set_context_index(uint64_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_DISCARD_H__
