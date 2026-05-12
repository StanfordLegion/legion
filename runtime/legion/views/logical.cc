/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/views/logical.h"
#include "legion/kernel/runtime.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // LogicalView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalView::LogicalView(
        DistributedID did, bool register_now, CollectiveMapping* map)
      : DistributedCollectable(did, register_now, map), valid_references(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LogicalView::~LogicalView(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(valid_references == 0);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ViewRequestMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      LogicalView* view = legion_safe_cast<LogicalView*>(dc);
      if (view->collective_mapping != nullptr)
      {
        // Check to see if this is a collective view, if the target
        // is in the replicated set, then there's nothing we need to do
        // We can just ignore this and the registration will be done later
        if (view->collective_mapping->contains(source))
          return;
        AddressSpaceID nearest = view->collective_mapping->find_nearest(source);
        if (nearest != runtime->address_space)
        {
          // Forward this on to the nearest space in the collective mapping
          ViewRequestMessage rez;
          {
            RezCheck z2(rez);
            rez.serialize(did);
            rez.serialize(source);
          }
          rez.dispatch(nearest);
          return;
        }
      }
      view->send_view(source);
    }

#ifdef LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    void LogicalView::add_base_valid_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      valid_references += cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_valid_references.find(source);
      if (finder == detailed_base_valid_references.end())
        detailed_base_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid();
    }

    //--------------------------------------------------------------------------
    void LogicalView::add_nested_valid_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      valid_references += cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_valid_references.find(source);
      if (finder == detailed_nested_valid_references.end())
        detailed_nested_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid();
    }

    //--------------------------------------------------------------------------
    bool LogicalView::remove_base_valid_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      legion_assert(valid_references >= cnt);
      valid_references -= cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_valid_references.find(source);
      legion_assert(finder != detailed_base_valid_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_base_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid();
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool LogicalView::remove_nested_valid_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      legion_assert(valid_references >= cnt);
      valid_references -= cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_valid_references.find(source);
      legion_assert(finder != detailed_nested_valid_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_nested_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid();
      else
        return false;
    }
#else   // LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    void LogicalView::add_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      // Cannot increment valid references until after the effects of
      // notify_valid are visible to avoid races
      if (valid_references.load() == 0)
        notify_valid();
      valid_references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    bool LogicalView::remove_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      legion_assert(valid_references.load() >= cnt);
      if (valid_references.fetch_sub(cnt) == cnt)
        return notify_invalid();
      else
        return false;
    }
#endif  // LEGION_DEBUG_GC

  }  // namespace Internal
}  // namespace Legion
