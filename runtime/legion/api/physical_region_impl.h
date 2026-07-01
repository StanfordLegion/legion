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

#ifndef __LEGION_PHYSICAL_REGION_IMPL_H__
#define __LEGION_PHYSICAL_REGION_IMPL_H__

#include "legion/api/physical_region.h"
#include "legion/api/requirements.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/utilities/instance_set.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PhysicalRegionImpl
     * The base implementation of a physical region object.
     * Physical region objects are not allowed to move from the
     * node in which they are created.  Like other objects
     * available to both the user and runtime they are reference
     * counted to know when they can be deleted.
     *
     * Note that we don't need to protect physical region impls
     * with any kind of synchronization mechanism since they
     * will only be manipulated by a single task which is
     * guaranteed to only be running on one processor.
     */
    class PhysicalRegionImpl
      : public Collectable,
        public Heapify<PhysicalRegionImpl, CONTEXT_LIFETIME> {
    public:
      PhysicalRegionImpl(
          const RegionRequirement& req, RtEvent mapped_event,
          ApEvent ready_event, ApUserEvent term_event, bool mapped,
          TaskContext* ctx, MapperID mid, MappingTagID tag, bool leaf,
          bool virt, bool collective, uint64_t blocking, unsigned index);
      PhysicalRegionImpl(const PhysicalRegionImpl& rhs) = delete;
      ~PhysicalRegionImpl(void);
    public:
      PhysicalRegionImpl& operator=(const PhysicalRegionImpl& rhs) = delete;
    public:
      inline bool created_accessor(void) const { return made_accessor; }
      inline RtEvent get_mapped_event(void) const { return mapped_event; }
    public:
      void wait_until_valid(
          bool silence_warnings, const char* warning_string, bool warn = false,
          const char* src = nullptr);
      bool is_valid(void) const;
      bool is_mapped(void) const;
      LogicalRegion get_logical_region(void) const;
      PrivilegeMode get_privilege(void) const;
    public:
      void unmap_region(bool escaped);
      void report_usage(bool escaped);
      ApEvent remap_region(ApEvent new_ready_event, uint64_t blocking);
      const RegionRequirement& get_requirement(void) const;
      void add_padded_field(FieldID fid);
      void set_reference(const InstanceRef& references, bool safe = false);
      void set_references(const InstanceSet& instances, bool safe = false);
      bool has_references(void) const;
      void get_references(InstanceSet& instances) const;
      void get_memories(
          std::set<Memory>& memories, bool silence_warnings,
          const char* warning_string) const;
      void get_fields(std::vector<FieldID>& fields) const;
    public:
      void get_bounds(void* realm_is, TypeTag type_tag);
      PieceIteratorImpl* get_piece_iterator(
          FieldID fid, bool privilege_only, bool silence_warnings,
          const char* warning_string);
      PhysicalInstance get_instance_info(
          PrivilegeMode mode, FieldID fid, size_t field_size, void* realm_is,
          TypeTag type_tag, const char* warning_string, bool silence_warnings,
          bool generic_accessor, bool check_field_size, ReductionOpID redop);
      PhysicalInstance get_padding_info(
          FieldID fid, size_t field_size, Domain* inner, Domain& outer,
          const char* warning_string, bool silence_warnings,
          bool generic_accessor, bool check_field_size);
      void report_incompatible_accessor(
          const char* accessor_kind, PhysicalInstance instance, FieldID fid);
      void report_incompatible_multi_accessor(
          unsigned index, FieldID fid, PhysicalInstance inst1,
          PhysicalInstance inst2);
      void report_colocation_violation(
          const char* accessor_kind, FieldID fid, PhysicalInstance inst1,
          PhysicalInstance ins2, const PhysicalRegion& other, bool reduction);
      static void empty_colocation_regions(
          const char* accessor_kind, FieldID fid, bool reduction);
      static void fail_bounds_check(
          DomainPoint p, FieldID fid, PrivilegeMode mode, bool multi);
      static void fail_bounds_check(
          Domain d, FieldID fid, PrivilegeMode mode, bool multi);
      static void fail_privilege_check(
          DomainPoint p, FieldID fid, PrivilegeMode mode);
      static void fail_privilege_check(
          Domain d, FieldID fid, PrivilegeMode mode);
      static void fail_padding_check(DomainPoint d, FieldID fid);
    public:
      TaskContext* const context;
      const MapperID map_id;
      const MappingTagID tag;
      const unsigned requirement_index;  // region requirement index
      const bool leaf_region;
      const bool virtual_mapped;
      // Whether this physical region represents a collectively
      // created group of instances or not (e.g. ReplAttachOp)
      const bool collective;
      const bool replaying;
    private:
      const RegionRequirement req;
      // Event for when the 'references' are set by the producer op
      // can only be accessed in "application" side code
      // There should only be one of these triggered by the producer
      const RtEvent mapped_event;
      // Event for when it is safe to use the physical instances
      // can only be accessed in "application" side code
      // triggered by mapping stage code
      ApEvent ready_event;
      // Event for when the mapped application code is done accessing
      // the physical region, set in "application" side code
      // should only be accessed there as well
      ApUserEvent termination_event;
      // Physical instances for this mapping
      // written by the "mapping stage" code of whatever operation made this
      // can be accessed in "application" side code after 'mapped' triggers
      InstanceSet references;
      // Any fields which we have privileges on the padded space (sorted)
      // This enables us to access the padded space for this field
      std::vector<FieldID> padded_fields;
      // The blocking index for when this physical region was created
      uint64_t blocking_index;
      // For non-leaf tasks we need to time how long a task spends using
      // this physical region so we can record it properly for profiling
      std::optional<long long> start_time;
      // "appliciation side" state
      // whether it is currently mapped
      bool mapped;
      // whether it is currently valid -> mapped and ready_event has triggered
      bool valid;
      bool made_accessor;
#ifdef LEGION_BOUNDS_CHECKS
    private:
      Domain bounds;
#endif
    };

    /**
     * \class ExternalResourcesImpl
     * This class provides the backing data structure for a collection of
     * physical regions that represent external data that have been attached
     * to logical regions in the same region tree
     */
    class ExternalResourcesImpl
      : public Collectable,
        public Heapify<ExternalResourcesImpl, SHORT_LIFETIME> {
    public:
      ExternalResourcesImpl(
          InnerContext* context, size_t num_regions, RegionTreeNode* upper,
          IndexSpaceNode* launch, LogicalRegion parent,
          const std::set<FieldID>& privilege_fields);
      ExternalResourcesImpl(const ExternalResourcesImpl& rhs) = delete;
      ~ExternalResourcesImpl(void);
    public:
      ExternalResourcesImpl& operator=(const ExternalResourcesImpl& rhs) =
          delete;
    public:
      size_t size(void) const;
      void set_region(unsigned index, PhysicalRegionImpl* region);
      PhysicalRegion get_region(unsigned index) const;
      void set_projection(ProjectionID pid);
      inline ProjectionID get_projection(void) const { return pid; }
      Future detach(
          InnerContext* context, IndexDetachOp* op, const bool flush,
          const bool unordered, Provenance* provenance);
    public:
      InnerContext* const context;
      // Save these for when we go to do the detach
      RegionTreeNode* const upper_bound;
      IndexSpaceNode* const launch_bounds;
      const std::vector<FieldID> privilege_fields;
      const LogicalRegion parent;
    protected:
      std::vector<PhysicalRegion> regions;
      ProjectionID pid;
      bool detached;
    };

    /**
     * \class PieceIteratorImpl
     * This is an interface for iterating over pieces
     * which in this case are just a list of rectangles
     */
    class PieceIteratorImpl : public Collectable {
    public:
      virtual ~PieceIteratorImpl(void) { }
      virtual int get_next(int index, Domain& next_piece) = 0;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PHYSICAL_REGION_IMPL_H__
