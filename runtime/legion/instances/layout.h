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

#ifndef __LEGION_LAYOUT_H__
#define __LEGION_LAYOUT_H__

#include "legion/api/constraints.h"
#include "legion/api/data.h"
#include "legion/kernel/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LayoutDescription
     * This class is for deduplicating the meta-data
     * associated with describing the layouts of physical
     * instances. Often times this meta data is rather
     * large (~100K) and since we routinely create up
     * to 100K instances, it is important to deduplicate
     * the data.  Since many instances will have the
     * same layout then they can all share the same
     * description object.
     */
    class LayoutDescription : public Collectable,
                              public Heapify<LayoutDescription, LONG_LIFETIME> {
    public:
      LayoutDescription(
          FieldSpaceNode* owner, const FieldMask& mask,
          const unsigned total_dims, LayoutConstraints* constraints,
          const std::vector<unsigned>& mask_index_map,
          const std::vector<FieldID>& fids,
          const std::vector<size_t>& field_sizes,
          const std::vector<CustomSerdezID>& serdez);
      // Used only by the virtual manager
      LayoutDescription(const FieldMask& mask, LayoutConstraints* constraints);
      LayoutDescription(const LayoutDescription& rhs) = delete;
      ~LayoutDescription(void);
    public:
      LayoutDescription& operator=(const LayoutDescription& rhs) = delete;
    public:
      void log_instance_layout(LgEvent inst_event) const;
    public:
      void compute_copy_offsets(
          const FieldMask& copy_mask, const PhysicalInstance instance,
          std::vector<CopySrcDstField>& fields);
      void compute_copy_offsets(
          const std::vector<FieldID>& copy_fields,
          const PhysicalInstance instance,
          std::vector<CopySrcDstField>& fields);
    public:
      void get_fields(std::set<FieldID>& fields) const;
      bool has_field(FieldID fid) const;
      void has_fields(std::map<FieldID, bool>& fields) const;
      void remove_space_fields(std::set<FieldID>& fields) const;
    public:
      const CopySrcDstField& find_field_info(FieldID fid) const;
      size_t get_total_field_size(void) const;
      void get_fields(std::vector<FieldID>& fields) const;
      void compute_destroyed_fields(
          std::vector<PhysicalInstance::DestroyedField>& serdez_fields) const;
    public:
      bool match_layout(
          const LayoutConstraintSet& constraints, unsigned num_dims) const;
      bool match_layout(
          const LayoutDescription* layout, unsigned num_dims) const;
    public:
      void pack_layout_description(Serializer& rez, AddressSpaceID target);
      static LayoutDescription* handle_unpack_layout_description(
          LayoutConstraints* constraints, FieldSpaceNode* field_space,
          size_t total_dims);
    public:
      const FieldMask allocated_fields;
      LayoutConstraints* const constraints;
      FieldSpaceNode* const owner;
      const unsigned total_dims;
    protected:
      // In order by index of bit mask
      std::vector<CopySrcDstField> field_infos;
      // A mapping from FieldIDs to indexes into our field_infos
      std::map<FieldID, unsigned /*index*/> field_indexes;
    protected:
      mutable LocalLock layout_lock;
      std::map<
          LEGION_FIELD_MASK_FIELD_TYPE,
          lng::list<std::pair<FieldMask, FieldMask> > >
          comp_cache;
    };

    /**
     * \class LayoutConstraints
     * A class for tracking a long-lived set of constraints
     * These can be moved around the system and referred to in
     * variout places so we make it a distributed collectable
     */
    class LayoutConstraints
      : public LayoutConstraintSet,
        public DistributedCollectable,
        public Heapify<LayoutConstraints, RUNTIME_LIFETIME> {
    public:
      LayoutConstraints(
          LayoutConstraintID layout_id, FieldSpace handle, bool inter,
          DistributedID did = 0);
      LayoutConstraints(
          LayoutConstraintID layout_id,
          const LayoutConstraintRegistrar& registrar, bool inter,
          DistributedID did = 0,
          CollectiveMapping* collective_mapping = nullptr);
      LayoutConstraints(
          LayoutConstraintID layout_id, const LayoutConstraintSet& cons,
          FieldSpace handle, bool inter);
      LayoutConstraints(const LayoutConstraints& rhs) = delete;
      virtual ~LayoutConstraints(void);
    public:
      LayoutConstraints& operator=(const LayoutConstraints& rhs) = delete;
      bool operator==(const LayoutConstraints& rhs) const;
      bool operator==(const LayoutConstraintSet& rhs) const;
    public:
      virtual void notify_local(void) override;
    public:
      inline FieldSpace get_field_space(void) const { return handle; }
      inline const char* get_name(void) const { return constraints_name; }
    public:
      void send_constraint_response(
          AddressSpaceID source, RtUserEvent done_event);
      void update_constraints(Deserializer& derez);
    public:
      bool entails(
          LayoutConstraints* other_constraints, unsigned total_dims,
          const LayoutConstraint** failed_constraint, bool test_pointer = true);
      bool entails(
          const LayoutConstraintSet& other, unsigned total_dims,
          const LayoutConstraint** failed_constraint,
          bool test_pointer = true) const;
      bool conflicts(
          LayoutConstraints* other_constraints, unsigned total_dims,
          const LayoutConstraint** conflict_constraint);
      bool conflicts(
          const LayoutConstraintSet& other, unsigned total_dims,
          const LayoutConstraint** conflict_constraint) const;
    public:
      static AddressSpaceID get_owner_space(LayoutConstraintID layout_id);
    public:
      const LayoutConstraintID layout_id;
      const FieldSpace handle;
      // True if this layout constraint object was made by the runtime
      // False if it was made by the application or the mapper
      const bool internal;
    protected:
      char* constraints_name;
      mutable LocalLock layout_lock;
    protected:
      std::map<
          std::pair<LayoutConstraintID, unsigned /*total dims*/>,
          const LayoutConstraint*>
          conflict_cache;
      std::map<
          std::pair<LayoutConstraintID, unsigned /*total dims*/>,
          const LayoutConstraint*>
          entailment_cache;
      std::map<
          std::pair<LayoutConstraintID, unsigned /*total dims*/>,
          const LayoutConstraint*>
          no_pointer_entailment_cache;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_LAYOUT_H__
