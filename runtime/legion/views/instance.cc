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

#include "legion/views/instance.h"
#include "legion/kernel/runtime.h"
#include "legion/utilities/privileges.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // InstanceView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(
        DistributedID did, bool register_now, CollectiveMapping* mapping)
      : LogicalView(did, register_now, mapping)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InstanceView::~InstanceView(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    /*static*/ void ViewRegisterUser::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView* view = runtime->find_or_request_logical_view(did, ready);
      DistributedID target_did;
      derez.deserialize(target_did);
      RtEvent target_ready;
      PhysicalManager* target =
          runtime->find_or_request_instance_manager(target_did, target_ready);

      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode* user_expr = runtime->get_node(handle);
      UniqueID op_id;
      derez.deserialize(op_id);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      ApEvent term_event;
      derez.deserialize(term_event);
      size_t local_collective_arrivals;
      derez.deserialize(local_collective_arrivals);
      ApUserEvent ready_event;
      derez.deserialize(ready_event);
      RtUserEvent registered_event, applied_event;
      derez.deserialize(registered_event);
      derez.deserialize(applied_event);
      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      if (target_ready.exists() && !target_ready.has_triggered())
        target_ready.wait();
      legion_assert(view->is_instance_view());
      InstanceView* inst_view = view->as_instance_view();
      std::vector<RtEvent> registered_events;
      ApEvent pre = inst_view->register_user(
          usage, user_mask, user_expr, op_id, op_ctx_index, index, term_event,
          target, nullptr /*no mapping*/, local_collective_arrivals,
          registered_events, applied_events, trace_info, source);
      if (ready_event.exists())
        Runtime::trigger_event(ready_event, pre, trace_info, applied_events);
      if (!registered_events.empty())
        Runtime::trigger_event(
            registered_event, Runtime::merge_events(registered_events));
      else
        Runtime::trigger_event(registered_event);
      if (!applied_events.empty())
        Runtime::trigger_event(
            applied_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

  }  // namespace Internal
}  // namespace Legion
