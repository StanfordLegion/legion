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

#include "legion/views/replicate.h"
#include "legion/kernel/runtime.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // ReplicatedView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplicatedView::ReplicatedView(
        DistributedID id, DistributedID ctx_did,
        const std::vector<IndividualView*>& views,
        const std::vector<DistributedID>& insts, bool register_now,
        CollectiveMapping* mapping)
      : CollectiveView(
            encode_replicated_did(id), ctx_did, views, insts, register_now,
            mapping)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Replicated View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    ReplicatedView::~ReplicatedView(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplicatedView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      ReplicatedViewMessage rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(context_did);
        rez.serialize<size_t>(instances.size());
        rez.serialize(
            &instances.front(), instances.size() * sizeof(DistributedID));
        if (collective_mapping != nullptr)
          collective_mapping->pack(rez);
        else
          rez.serialize<size_t>(0);
      }
      rez.dispatch(target);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicatedViewMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did, ctx_did;
      derez.deserialize(did);
      derez.deserialize(ctx_did);
      size_t num_insts;
      derez.deserialize(num_insts);
      std::vector<DistributedID> instances(num_insts);
      derez.deserialize(&instances.front(), num_insts * sizeof(DistributedID));
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = nullptr;
      if (num_spaces > 0)
      {
        mapping = new CollectiveMapping(derez, num_spaces);
        mapping->add_reference();
      }
      void* location =
          runtime->find_or_create_pending_collectable_location<ReplicatedView>(
              did);
      std::vector<IndividualView*> no_views;
      ReplicatedView* view = new (location) ReplicatedView(
          did, ctx_did, no_views, instances, false /*register now*/, mapping);
      // Register only after construction
      view->register_with_runtime();
      if ((mapping != nullptr) && mapping->remove_reference())
        delete mapping;
    }

  }  // namespace Internal
}  // namespace Legion
