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

#include "legion/views/reduction.h"
#include "legion/instances/physical.h"
#include "legion/views/fill.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // ReductionView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(
        DistributedID did, AddressSpaceID log_own, PhysicalManager* man,
        bool register_now, CollectiveMapping* mapping)
      : IndividualView(
            encode_reduction_did(did), man, log_own, register_now, mapping),
        fill_view(runtime->find_or_create_reduction_fill_view(man->redop))
    //--------------------------------------------------------------------------
    {
      fill_view->add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Reduction View %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space,
          LEGION_DISTRIBUTED_ID_FILTER(manager->did));
#endif
    }

    //--------------------------------------------------------------------------
    ReductionView::~ReductionView(void)
    //--------------------------------------------------------------------------
    {
      if (fill_view->remove_nested_resource_ref(did))
        delete fill_view;
    }

    //--------------------------------------------------------------------------
    void ReductionView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Don't take the lock, it's alright to have duplicate sends
      ReductionViewMessage rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(manager->did);
        rez.serialize(logical_owner);
      }
      rez.dispatch(target);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    ReductionOpID ReductionView::get_redop(void) const
    //--------------------------------------------------------------------------
    {
      return manager->redop;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionViewMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      RtEvent man_ready, ctx_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(manager_did, man_ready);
      if (man_ready.exists() && !man_ready.has_triggered())
      {
        // Defer this until the manager is ready
        ReductionView::DeferReductionViewArgs args(did, manager, logical_owner);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_RESPONSE_PRIORITY, man_ready);
      }
      else
        ReductionView::create_remote_view(did, manager, logical_owner);
    }

    //--------------------------------------------------------------------------
    void ReductionView::DeferReductionViewArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      create_remote_view(did, manager, logical_owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::create_remote_view(
        DistributedID did, PhysicalManager* manager,
        AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
      legion_assert(manager->is_reduction_manager());
      void* location =
          runtime->find_or_create_pending_collectable_location<ReductionView>(
              did);
      ReductionView* view = new (location)
          ReductionView(did, logical_owner, manager, false /*register now*/);
      // Only register after construction
      view->register_with_runtime();
    }

  }  // namespace Internal
}  // namespace Legion
