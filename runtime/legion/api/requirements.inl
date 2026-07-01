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

// Included from requirements.h - do not include this directly

// Useful for IDEs
#include "legion/api/requirements.h"

namespace Legion {

  //--------------------------------------------------------------------------
  inline constexpr RegionFlags operator~(RegionFlags f)
  //--------------------------------------------------------------------------
  {
    return static_cast<RegionFlags>(~unsigned(f));
  }

  //--------------------------------------------------------------------------
  inline constexpr RegionFlags operator|(RegionFlags left, RegionFlags right)
  //--------------------------------------------------------------------------
  {
    return static_cast<RegionFlags>(unsigned(left) | unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr RegionFlags operator&(RegionFlags left, RegionFlags right)
  //--------------------------------------------------------------------------
  {
    return static_cast<RegionFlags>(unsigned(left) & unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr RegionFlags operator^(RegionFlags left, RegionFlags right)
  //--------------------------------------------------------------------------
  {
    return static_cast<RegionFlags>(unsigned(left) ^ unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr RegionFlags operator|=(RegionFlags& left, RegionFlags right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l |= r;
    return left = static_cast<RegionFlags>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr RegionFlags operator&=(RegionFlags& left, RegionFlags right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l &= r;
    return left = static_cast<RegionFlags>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr RegionFlags operator^=(RegionFlags& left, RegionFlags right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l ^= r;
    return left = static_cast<RegionFlags>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr PrivilegeMode operator~(PrivilegeMode p)
  //--------------------------------------------------------------------------
  {
    return static_cast<PrivilegeMode>(~unsigned(p));
  }

  //--------------------------------------------------------------------------
  inline constexpr PrivilegeMode operator|(
      PrivilegeMode left, PrivilegeMode right)
  //--------------------------------------------------------------------------
  {
    return static_cast<PrivilegeMode>(unsigned(left) | unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr PrivilegeMode operator&(
      PrivilegeMode left, PrivilegeMode right)
  //--------------------------------------------------------------------------
  {
    return static_cast<PrivilegeMode>(unsigned(left) & unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr PrivilegeMode operator^(
      PrivilegeMode left, PrivilegeMode right)
  //--------------------------------------------------------------------------
  {
    return static_cast<PrivilegeMode>(unsigned(left) ^ unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr PrivilegeMode operator|=(
      PrivilegeMode& left, PrivilegeMode right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l |= r;
    return left = static_cast<PrivilegeMode>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr PrivilegeMode operator&=(
      PrivilegeMode& left, PrivilegeMode right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l &= r;
    return left = static_cast<PrivilegeMode>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr PrivilegeMode operator^=(
      PrivilegeMode& left, PrivilegeMode right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l ^= r;
    return left = static_cast<PrivilegeMode>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr CoherenceProperty operator~(CoherenceProperty p)
  //--------------------------------------------------------------------------
  {
    return static_cast<CoherenceProperty>(~unsigned(p));
  }

  //--------------------------------------------------------------------------
  inline constexpr CoherenceProperty operator|(
      CoherenceProperty left, CoherenceProperty right)
  //--------------------------------------------------------------------------
  {
    return static_cast<CoherenceProperty>(unsigned(left) | unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr CoherenceProperty operator&(
      CoherenceProperty left, CoherenceProperty right)
  //--------------------------------------------------------------------------
  {
    return static_cast<CoherenceProperty>(unsigned(left) & unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr CoherenceProperty operator^(
      CoherenceProperty left, CoherenceProperty right)
  //--------------------------------------------------------------------------
  {
    return static_cast<CoherenceProperty>(unsigned(left) ^ unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr CoherenceProperty operator|=(
      CoherenceProperty& left, CoherenceProperty right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l |= r;
    return left = static_cast<CoherenceProperty>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr CoherenceProperty operator&=(
      CoherenceProperty& left, CoherenceProperty right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l &= r;
    return left = static_cast<CoherenceProperty>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr CoherenceProperty operator^=(
      CoherenceProperty& left, CoherenceProperty right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l ^= r;
    return left = static_cast<CoherenceProperty>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr AllocateMode operator~(AllocateMode a)
  //--------------------------------------------------------------------------
  {
    return static_cast<AllocateMode>(~unsigned(a));
  }

  //--------------------------------------------------------------------------
  inline constexpr AllocateMode operator|(AllocateMode left, AllocateMode right)
  //--------------------------------------------------------------------------
  {
    return static_cast<AllocateMode>(unsigned(left) | unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr AllocateMode operator&(AllocateMode left, AllocateMode right)
  //--------------------------------------------------------------------------
  {
    return static_cast<AllocateMode>(unsigned(left) & unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr AllocateMode operator^(AllocateMode left, AllocateMode right)
  //--------------------------------------------------------------------------
  {
    return static_cast<AllocateMode>(unsigned(left) ^ unsigned(right));
  }

  //--------------------------------------------------------------------------
  inline constexpr AllocateMode operator|=(
      AllocateMode& left, AllocateMode right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l |= r;
    return left = static_cast<AllocateMode>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr AllocateMode operator&=(
      AllocateMode& left, AllocateMode right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l &= r;
    return left = static_cast<AllocateMode>(l);
  }

  //--------------------------------------------------------------------------
  inline constexpr AllocateMode operator^=(
      AllocateMode& left, AllocateMode right)
  //--------------------------------------------------------------------------
  {
    unsigned l = static_cast<unsigned>(left);
    unsigned r = static_cast<unsigned>(right);
    l ^= r;
    return left = static_cast<AllocateMode>(l);
  }

  //--------------------------------------------------------------------------
  inline RegionRequirement& RegionRequirement::add_field(
      FieldID fid, bool instance /*= true*/)
  //--------------------------------------------------------------------------
  {
    privilege_fields.insert(fid);
    if (instance)
      instance_fields.emplace_back(fid);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline RegionRequirement& RegionRequirement::add_fields(
      const std::vector<FieldID>& fids, bool instance /*= true*/)
  //--------------------------------------------------------------------------
  {
    privilege_fields.insert(fids.begin(), fids.end());
    if (instance)
      instance_fields.insert(instance_fields.end(), fids.begin(), fids.end());
    return *this;
  }

  //--------------------------------------------------------------------------
  inline RegionRequirement& RegionRequirement::add_flags(RegionFlags new_flags)
  //--------------------------------------------------------------------------
  {
    flags |= new_flags;
    return *this;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename COORD_T>
  void OutputRequirement::set_type_tag()
  //--------------------------------------------------------------------------
  {
    type_tag = Internal::NT_TemplateHelper::encode_tag<DIM, COORD_T>();
  }

}  // namespace Legion
