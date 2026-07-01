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

#include "legion/tasks/remote.h"
#include "legion/contexts/inner.h"
#include "legion/managers/mapper.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Remote Task Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTaskOp::RemoteTaskOp(Operation* ptr, AddressSpaceID src)
      : ExternalTask(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteTaskOp::~RemoteTaskOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteTaskOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteTaskOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteTaskOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    bool RemoteTaskOp::has_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return (get_depth() > 0);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteTaskOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return parent_ctx->get_task();
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteTaskOp::get_provenance_string(
        bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    const char* RemoteTaskOp::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl* impl = runtime->find_or_create_task_impl(task_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    Domain RemoteTaskOp::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
      // We're mapping a point task if we've made one of these
      return Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    ShardID RemoteTaskOp::get_shard_id(void) const
    //--------------------------------------------------------------------------
    {
      // We're mapping a point task if we've made one of these
      return 0;
    }

    //--------------------------------------------------------------------------
    size_t RemoteTaskOp::get_total_shards(void) const
    //--------------------------------------------------------------------------
    {
      // We're mapping a point task if we've made one of these
      return 1;
    }

    //--------------------------------------------------------------------------
    DomainPoint RemoteTaskOp::get_shard_point(void) const
    //--------------------------------------------------------------------------
    {
      return DomainPoint(0);
    }

    //--------------------------------------------------------------------------
    Domain RemoteTaskOp::get_shard_domain(void) const
    //--------------------------------------------------------------------------
    {
      return Domain(DomainPoint(0), DomainPoint(0));
    }

    //--------------------------------------------------------------------------
    const char* RemoteTaskOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TASK_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteTaskOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TASK_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking, points);
        return;
      }
      Mapper::SelectTaskSrcInput input;
      Mapper::SelectTaskSrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      input.region_req_index = index;
      if (mapper == nullptr)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_task_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_task(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_task(derez);
      unpack_profiling_requests(derez);
    }

  }  // namespace Internal
}  // namespace Legion
