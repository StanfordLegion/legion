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

#ifndef __LEGION_CLOSE_H__
#define __LEGION_CLOSE_H__

#include "legion/analysis/versioning.h"
#include "legion/api/mapping.h"
#include "legion/operations/internal.h"
#include "legion/operations/remote.h"
#include "legion/utilities/instance_set.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalClose
     * An extension of the external-facing Close to help
     * with packing and unpacking them
     */
    class ExternalClose : public Close,
                          public ExternalMappable {
    public:
      ExternalClose(void);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      void pack_external_close(Serializer& rez, AddressSpaceID target) const;
      void unpack_external_close(Deserializer& derez);
    };

    /**
     * \class CloseOp
     * Close operations are only visible internally inside
     * the runtime and are issued to help close up the
     * physical region tree. There are two types of close
     * operations that both inherit from this class:
     * InterCloseOp and PostCloseOp.
     */
    class CloseOp : public ExternalClose,
                    public InternalOp {
    public:
      CloseOp(void);
      CloseOp(const CloseOp& rhs) = delete;
      virtual ~CloseOp(void);
    public:
      CloseOp& operator=(const CloseOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
      virtual Mappable* get_mappable(void) override;
    public:
      // This is for post and virtual close ops
      void initialize_close(InnerContext* ctx, const RegionRequirement& req);
      // These is for internal close ops
      void initialize_close(
          Operation* creator, unsigned idx, const RegionRequirement& req);
      void perform_logging(Operation* creator, unsigned index, bool merge);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override = 0;
      virtual OpKind get_operation_kind(void) const override = 0;
      virtual size_t get_region_count(void) const override;
      virtual const FieldMask& get_internal_mask(void) const override;
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
    public:
      virtual void trigger_commit(void) override;
    };

    /**
     * \class MergeCloseOp
     * merge close operations are issued by the runtime
     * for closing up region trees as part of the normal execution
     * of an application.
     */
    class MergeCloseOp : public CloseOp {
    public:
      MergeCloseOp(void);
      MergeCloseOp(const MergeCloseOp& rhs) = delete;
      virtual ~MergeCloseOp(void);
    public:
      MergeCloseOp& operator=(const MergeCloseOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, const RegionRequirement& req, int close_idx,
          Operation* create_op);
      inline void update_close_mask(const FieldMask& mask)
      {
        close_mask |= mask;
      }
      inline const FieldMask& get_close_mask(void) const { return close_mask; }
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual const FieldMask& get_internal_mask(void) const override;
    public:
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
    protected:
      FieldMask close_mask;
      unsigned parent_req_index;
    };

    /**
     * \class PostCloseOp
     * Post close operations are issued by the runtime after a
     * task has finished executing and the region tree contexts
     * need to be closed up to the original physical instance
     * that was mapped by the parent task.
     */
    class PostCloseOp : public CloseOp {
    public:
      PostCloseOp(void);
      PostCloseOp(const PostCloseOp& rhs) = delete;
      virtual ~PostCloseOp(void);
    public:
      PostCloseOp& operator=(const PostCloseOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, unsigned index,
          const InstanceSet& target_instances);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_commit(void) override;
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override;
    protected:
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
      virtual void handle_profiling_update(int count) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
    protected:
      unsigned parent_idx;
      VersionInfo version_info;
      InstanceSet target_instances;
      std::map<PhysicalManager*, unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager* mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      RtUserEvent profiling_reported;
      int profiling_priority;
      std::atomic<int> outstanding_profiling_requests;
      std::atomic<int> outstanding_profiling_reported;
    };

    /**
     * \class ReplMergeCloseOp
     * A close operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplMergeCloseOp : public MergeCloseOp {
    public:
      ReplMergeCloseOp(void);
      ReplMergeCloseOp(const ReplMergeCloseOp& rhs) = delete;
      virtual ~ReplMergeCloseOp(void);
    public:
      ReplMergeCloseOp& operator=(const ReplMergeCloseOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      void set_repl_close_info(RtBarrier mapped_barrier);
      virtual void trigger_ready(void) override;
    protected:
      RtBarrier mapped_barrier;
    };

    /**
     * \class RemoteCloseOp
     * This is a remote copy of a CloseOp to be used
     * for mapper calls and other operations
     */
    class RemoteCloseOp : public ExternalClose,
                          public RemoteOp,
                          public Heapify<RemoteCloseOp, OPERATION_LIFETIME> {
    public:
      RemoteCloseOp(Operation* ptr, AddressSpaceID src);
      RemoteCloseOp(const RemoteCloseOp& rhs) = delete;
      virtual ~RemoteCloseOp(void);
    public:
      RemoteCloseOp& operator=(const RemoteCloseOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
    public:
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_CLOSE_H__
