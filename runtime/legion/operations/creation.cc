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

#include "legion/operations/creation.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/nodes/field.h"
#include "legion/nodes/index.h"
#include "legion/tools/spy.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Creation Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CreationOp::CreationOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CreationOp::~CreationOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_index_space(
        InnerContext* ctx, IndexSpaceNode* n, const Future& f,
        Provenance* provenance, bool own, const CollectiveMapping* map)
    //--------------------------------------------------------------------------
    {
      legion_assert(index_space_node == nullptr);
      legion_assert(futures.empty());
      initialize_operation(ctx, provenance);
      kind = INDEX_SPACE_CREATION;
      index_space_node = n;
      futures.emplace_back(f);
      mapping = map;
      owner = own;
      LegionSpy::log_creation_operation(
          parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_field(
        InnerContext* ctx, FieldSpaceNode* node, FieldID fid,
        const Future& field_size, Provenance* provenance, bool own)
    //--------------------------------------------------------------------------
    {
      legion_assert(field_space_node == nullptr);
      legion_assert(fields.empty());
      legion_assert(futures.empty());
      initialize_operation(ctx, provenance);
      kind = FIELD_ALLOCATION;
      field_space_node = node;
      fields.emplace_back(fid);
      futures.emplace_back(field_size);
      owner = own;
      LegionSpy::log_creation_operation(
          parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_fields(
        InnerContext* ctx, FieldSpaceNode* node,
        const std::vector<FieldID>& fids,
        const std::vector<Future>& field_sizes, Provenance* provenance,
        bool own)
    //--------------------------------------------------------------------------
    {
      legion_assert(field_space_node == nullptr);
      legion_assert(fields.empty());
      legion_assert(futures.empty());
      legion_assert(fids.size() == field_sizes.size());
      initialize_operation(ctx, provenance);
      kind = FIELD_ALLOCATION;
      field_space_node = node;
      fields = fids;
      futures = field_sizes;
      owner = own;
      LegionSpy::log_creation_operation(
          parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void CreationOp::initialize_map(
        InnerContext* ctx, Provenance* provenance,
        const std::map<DomainPoint, Future>& future_points)
    //--------------------------------------------------------------------------
    {
      legion_assert(futures.empty());
      initialize_operation(ctx, provenance);
      kind = FUTURE_MAP_CREATION;
      futures.resize(future_points.size());
      unsigned index = 0;
      for (const std::pair<const DomainPoint, Future>& it : future_points)
        futures[index++] = it.second;
      LegionSpy::log_creation_operation(
          parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void CreationOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      index_space_node = nullptr;
      field_space_node = nullptr;
      mapping = nullptr;
      owner = true;
    }

    //--------------------------------------------------------------------------
    void CreationOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      futures.clear();
      fields.clear();
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* CreationOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[CREATION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind CreationOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return CREATION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void CreationOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (const Future& future : futures)
      {
        legion_assert(future.impl != nullptr);
        // Register this operation as dependent on task that
        // generated the future
        future.impl->register_dependence(this);
      }
      // Record this with the context as an implicit dependence for all
      // later operations which may rely on this operation for the creation
      // Note that future map creations are exempt from this since the
      // resource that they are producing (a future map) will have downstream
      // operations explicitly recording mapping dependences on it
      if (kind != FUTURE_MAP_CREATION)
        parent_ctx->update_current_implicit_creation(this);
    }

    //--------------------------------------------------------------------------
    void CreationOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        case INDEX_SPACE_CREATION:
          {
            legion_assert(futures.size() == 1);
            if (owner)
            {
              // Have to request internal buffers before completing mapping
              // in case we have to make an instance as part of it
              FutureImpl* impl = futures[0].impl;
              impl->request_runtime_instance(this);
              complete_mapping();
              const RtEvent ready = impl->find_runtime_instance_ready();
              if (ready.exists() && !ready.has_triggered())
                parent_ctx->add_to_trigger_execution_queue(this, ready);
              else
                trigger_execution();
            }
            else
              trigger_execution();
            break;
          }
        case FIELD_ALLOCATION:
          {
            std::vector<RtEvent> mapped_events, ready_events;
            // Have to request internal buffers before completing mapping
            // in case we have to make an instance as part of it
            if (owner)
            {
              for (unsigned idx = 0; idx < futures.size(); idx++)
              {
                FutureImpl* impl = futures[idx].impl;
                impl->request_runtime_instance(this);
                const RtEvent subscribed = impl->find_runtime_instance_ready();
                if (subscribed.exists())
                  ready_events.emplace_back(subscribed);
              }
            }
            if (!mapped_events.empty())
              complete_mapping(Runtime::merge_events(mapped_events));
            else
              complete_mapping();
            if (!ready_events.empty())
            {
              const RtEvent ready = Runtime::merge_events(ready_events);
              if (ready.exists() && !ready.has_triggered())
                parent_ctx->add_to_trigger_execution_queue(this, ready);
              else
                trigger_execution();
            }
            else
              trigger_execution();
            break;
          }
        case FUTURE_MAP_CREATION:
          {
            complete_mapping();
            complete_execution();
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    void CreationOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> complete_preconditions;
      switch (kind)
      {
        case INDEX_SPACE_CREATION:
          {
            // Pull the pointer for the domain out of the future and assign
            // it to the index space node
            FutureImpl* impl = futures[0].impl;
            size_t future_size = 0;
            const Domain* domain = static_cast<const Domain*>(
                impl->find_runtime_buffer(parent_ctx, future_size));
            if (future_size != sizeof(Domain))
            {
              Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
              error << "Future for index space creation by " << *this
                    << " does not have the same size as sizeof(Domain) (e.g. "
                    << sizeof(Domain) << " bytes). The type of futures for "
                    << "index space domains must be a Domain.";
              error.raise();
            }
            if (owner &&
                index_space_node->set_domain(
                    *domain, ApEvent::NO_AP_EVENT, true /*take ownership*/))
              delete index_space_node;
            break;
          }
        case FIELD_ALLOCATION:
          {
            for (unsigned idx = 0; idx < futures.size(); idx++)
            {
              FutureImpl* impl = futures[idx].impl;
              size_t future_size = 0;
              const size_t* field_size = static_cast<const size_t*>(
                  impl->find_runtime_buffer(parent_ctx, future_size));
              if (future_size != sizeof(size_t))
              {
                Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
                error
                    << "Size of future passed into dynamic field allocation "
                       "for "
                    << "field " << fields[idx] << " is " << future_size
                    << " bytes which not the same as sizeof(size_t) ("
                    << sizeof(size_t) << " bytes). Futures passed into field "
                    << "allocation calls must contain data of the type size_t.";
                error.raise();
              }
              if (owner)
              {
                field_space_node->update_field_size(
                    fields[idx], *field_size, complete_preconditions,
                    runtime->address_space);
                LegionSpy::log_field_creation(
                    field_space_node->handle.get_id(), fields[idx], *field_size,
                    (get_provenance() == nullptr) ? std::string_view() :
                                                    get_provenance()->human);
              }
            }
            break;
          }
        default:
          std::abort();
      }
      if (!complete_preconditions.empty())
        complete_execution(Runtime::merge_events(complete_preconditions));
      else
        complete_execution();
    }

  }  // namespace Internal
}  // namespace Legion
