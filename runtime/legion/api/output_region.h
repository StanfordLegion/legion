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

#ifndef __LEGION_OUTPUT_REGION_H__
#define __LEGION_OUTPUT_REGION_H__

#include "legion/api/constraints.h"

namespace Legion {

  /**
   * \class OutputRegion
   * An OutputRegion provides an interface for applications to specify
   * the output instances or allocations of memory to associate with
   * output region requirements.
   */
  class OutputRegion : public Unserializable {
  public:
    OutputRegion(void);
    OutputRegion(const OutputRegion& rhs);
    ~OutputRegion(void);
  private:
    Internal::OutputRegionImpl* impl;
  protected:
    FRIEND_ALL_RUNTIME_CLASSES
    explicit OutputRegion(Internal::OutputRegionImpl* impl);
  public:
    OutputRegion& operator=(const OutputRegion& rhs);
  public:
    Memory target_memory(void) const;
    // Returns the logical region of this output region.
    // The call is legal only when the output region is valid and
    // will raise an error otherwise.
    LogicalRegion get_logical_region(void) const;
    bool is_bounded_output_region(void) const;
  public:
    // Returns a deferred buffer that satisfies the layout constraints of
    // this output region. The caller still needs to pass this buffer to
    // a return_data call if the buffer needs to be bound to this output
    // region. The caller can optionally choose to bind the returned buffer
    // to the output region; such a call cannot be made more than once.
    template<
        typename T, int DIM, typename COORD_T = coord_t,
#ifdef LEGION_BOUNDS_CHECKS
        bool CHECK_BOUNDS = true>
#else
        bool CHECK_BOUNDS = false>
#endif
    DeferredBuffer<T, DIM, COORD_T, CHECK_BOUNDS> create_buffer(
        const Point<DIM, COORD_T>& extents, FieldID field_id,
        const T* initial_value = nullptr, bool return_buffer = false);
  private:
    void check_type_tag(TypeTag type_tag) const;
    void check_field_size(FieldID field_id, size_t field_size) const;
    void get_layout(
        FieldID field_id, std::vector<DimensionKind>& ordering,
        size_t& alignment) const;
  public:
    template<
        typename T, int DIM, typename COORD_T = coord_t,
#ifdef LEGION_BOUNDS_CHECKS
        bool CHECK_BOUNDS = true>
#else
        bool CHECK_BOUNDS = false>
#endif
    void return_data(
        const Point<DIM, COORD_T>& extents, FieldID field_id,
        DeferredBuffer<T, DIM, COORD_T, CHECK_BOUNDS>& buffer);
    void return_data(
        const DomainPoint& extents, FieldID field_id,
        Realm::RegionInstance instance, bool check_constraints = true);
  private:
    void return_data(
        const DomainPoint& extents, FieldID field_id,
        Realm::RegionInstance instance, const LayoutConstraintSet* constraints,
        bool check_constraints);
  };

}  // namespace Legion

#include "legion/api/output_region.inl"

#endif  // __LEGION_OUTPUT_REGION_H__
