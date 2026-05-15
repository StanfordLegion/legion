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

#include "legion/nodes/index.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Index Tree Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTreeNode::IndexTreeNode(
        unsigned d, LegionColor c, DistributedID did, RtEvent init,
        CollectiveMapping* mapping, Provenance* prov, bool tree_valid)
      : ValidDistributedCollectable(
            did, false /*register*/, mapping, tree_valid),
        depth(d), color(c), provenance(prov), initialized(init)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    IndexTreeNode::~IndexTreeNode(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    void IndexTreeNode::attach_semantic_information(
        SemanticTag tag, AddressSpaceID source, const void* buffer, size_t size,
        bool is_mutable, bool local_only)
    //--------------------------------------------------------------------------
    {
      bool added = true;
      {
        AutoLock n_lock(node_lock);
        // See if it already exists
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          // First check to see if it is valid
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // It's not mutable so check to make
              // sure that the bits are the same
              if (size != finder->second.buffer.get_size())
              {
                Error err(LEGION_INTERFACE_EXCEPTION);
                err << "Inconsistent Semantic Tag value for tag " << tag
                    << " with different sizes of " << size << " and "
                    << finder->second.buffer.get_size()
                    << " for index tree node";
                err.raise();
              }
              // Otherwise do a bitwise comparison
              {
                const char* orig =
                    (const char*)finder->second.buffer.get_buffer();
                const char* next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                  {
                    Error err(LEGION_INTERFACE_EXCEPTION);
                    err << "Inconsistent Semantic Tag value for tag " << tag
                        << " with different values at byte " << idx
                        << " for index tree node, " << std::hex
                        << (int)orig[idx] << " != " << (int)next[idx];
                    err.raise();
                  }
                }
              }
              added = false;
            }
            else
            {
              // Mutable so overwrite the result
              finder->second.buffer.save_buffer(buffer, size);
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer.save_buffer(buffer, size);
            // Trigger will happen by the caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_info[tag] = SemanticInfo(buffer, size, is_mutable);
      }
      if (added)
      {
        AddressSpaceID owner_space = get_owner_space();
        // If we are not the owner and the message
        // didn't come from the owner, then send it
        if ((owner_space != runtime->address_space) &&
            (source != owner_space) && !local_only)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          send_semantic_info(owner_space, tag, buffer, size, is_mutable, done);
          if (!done.has_triggered())
            done.wait();
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTreeNode::retrieve_semantic_information(
        SemanticTag tag, const void*& result, size_t& size, bool can_fail,
        bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        lng::map<SemanticTag, SemanticInfo>::const_iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else  // can use the canonical event
              wait_on = finder->second.ready_event;
          }
          else if (wait_until)  // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        // Nothing to wait on so we have to do something
        if (can_fail)
          return false;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid semantic tag " << tag << " for index tree node.";
        error.raise();
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
          send_semantic_request(
              owner_space, tag, can_fail, wait_until, request);
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock, false /*exclusive*/);
      lng::map<SemanticTag, SemanticInfo>::const_iterator finder =
          semantic_info.find(tag);
      if (finder == semantic_info.end())
      {
        if (can_fail)
          return false;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid semantic tag " << tag << " for index tree node.";
        error.raise();
      }
      result = finder->second.buffer.get_buffer();
      size = finder->second.buffer.get_size();
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(
        IndexSpace h, IndexPartNode* par, LegionColor c,
        IndexSpaceExprID exp_id, RtEvent init, unsigned dep, Provenance* prov,
        CollectiveMapping* map, bool tree_valid)
      : IndexTreeNode(
            (dep == std::numeric_limits<unsigned>::max()) ?
                ((par == nullptr) ? 0 : par->depth + 1) :
                dep,
            c, h.did, init, map, prov, tree_valid),
        IndexSpaceExpression(
            h.type_tag,
            exp_id > 0 ? exp_id : runtime->get_unique_index_space_expr_id(),
            node_lock),
        handle(h), parent(par), next_uncollected_color(0),
        index_space_set(false), index_space_tight(false)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (parent == nullptr) ||
          (handle.get_type_tag() == parent->handle.get_type_tag()));
      if (parent != nullptr)
        parent->add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Index Space %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.did);
#endif
      if (is_owner())
        LegionSpy::log_index_space_expr(handle.get_id(), this->expr_id);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::~IndexSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      if ((parent != nullptr) && parent->remove_nested_resource_ref(did))
        delete parent;
      // Remove ourselves from the context
      if (registered_with_runtime)
        runtime->remove_node(handle);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here currently
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        runtime->unregister_remote_expression(expr_id);
      // Invalidate any derived operations
      invalidate_derived_operations(did);
      IndexSpaceExpression* canon = canonical.load();
      if (canon != nullptr)
      {
        if (canon == this)
          runtime->remove_canonical_expression(this);
        else if (canon->remove_canonical_reference(did))
          delete canon;
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::is_index_space_node(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

#ifdef LEGION_DEBUG
    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexSpaceNode::as_index_space_node(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexSpaceNode::as_index_part_node(void)
    //--------------------------------------------------------------------------
    {
      return nullptr;
    }
#endif

    //--------------------------------------------------------------------------
    AddressSpaceID IndexSpaceNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID IndexSpaceNode::get_owner_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return (handle.get_id() % runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* IndexSpaceNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_semantic_request(
        AddressSpaceID target, SemanticTag tag, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      IndexSpaceSemanticInfoRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_semantic_info(
        AddressSpaceID target, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      IndexSpaceSemanticInfoResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::process_semantic_request(
        SemanticTag tag, AddressSpaceID source, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(get_owner_space() == runtime->address_space);
      RtEvent precondition;
      const void* result = nullptr;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == nullptr)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::SemanticRequestArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->process_semantic_request(
          tag, source, false, false, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceSemanticInfoRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      IndexSpaceNode* node = runtime->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceSemanticInfoResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      runtime->get_node(handle)->attach_semantic_information(
          tag, source, buffer, size, is_mutable, false /*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* child = get_child(c, nullptr /*defer*/, true /*can fail*/);
      if (child == nullptr)
        return false;
      if (child->remove_base_resource_ref(REGION_TREE_REF))
        delete child;
      return true;
    }

    //--------------------------------------------------------------------------
    LegionColor IndexSpaceNode::generate_color(LegionColor suggestion)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        AutoLock n_lock(node_lock);
        // First check that we haven't recorded any children that don't have
        // any generated colors as it is illegal to generate colors if the
        // user has determined that they are specifying all the colors
        if (remote_colors.find(INVALID_COLOR) == remote_colors.end())
        {
          if (!color_map.empty() || !remote_colors.empty() ||
              (next_uncollected_color > 0))
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error
                << "Illegal request for Legion to generated a color for index "
                << "space " << handle.get_id()
                << " after a child was already registered with an "
                << "explicit color. Colors of partitions must either be "
                << "completely specified by the user or completely generated "
                << "by the runtime. Mixing of allocation modes is not allowed.";
            error.raise();
          }
          // If we made it here then there are no other children registered
          // so we record an empty entry to mark that we're generating colors
          remote_colors[INVALID_COLOR] = IndexPartition::NO_PART;
        }
        // If the user made a suggestion see if it was right
        if (suggestion != INVALID_COLOR)
        {
          // If someone already has it then they can't use it
          if ((next_uncollected_color <= suggestion) &&
              (color_map.find(suggestion) == color_map.end()))
          {
            color_map[suggestion] = nullptr;
            return suggestion;
          }
          else
            return INVALID_COLOR;
        }
        if (color_map.empty())
        {
          // save a space for later
          color_map[next_uncollected_color] = nullptr;
          return next_uncollected_color;
        }
        std::map<LegionColor, IndexPartNode*>::const_iterator next =
            color_map.begin();
        if (next->first > next_uncollected_color)
        {
          // save a space for later
          color_map[next_uncollected_color] = nullptr;
          return next_uncollected_color;
        }
        std::map<LegionColor, IndexPartNode*>::const_iterator prev = next++;
        while (next != color_map.end())
        {
          if (next->first != (prev->first + 1))
          {
            // save a space for later
            color_map[prev->first + 1] = nullptr;
            return prev->first + 1;
          }
          prev = next++;
        }
        color_map[prev->first + 1] = nullptr;
        return prev->first + 1;
      }
      else
      {
        // Send a message to the owner to pick a color and wait for the result
        std::atomic<LegionColor> result(suggestion);
        RtUserEvent ready = Runtime::create_rt_user_event();
        IndexSpaceGenerateColorRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(suggestion);
          rez.serialize(&result);
          rez.serialize(ready);
        }
        rez.dispatch(owner_space);
        if (!ready.has_triggered())
          ready.wait();
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::release_color(LegionColor color)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        AutoLock n_lock(node_lock);
        std::map<LegionColor, IndexPartNode*>::iterator finder =
            color_map.find(color);
        legion_assert(finder != color_map.end());
        legion_assert(finder->second == nullptr);
        color_map.erase(finder);
      }
      else
      {
        pack_valid_ref();
        IndexSpaceReleaseColor rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(color);
        }
        rez.dispatch(owner_space);
      }
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexSpaceNode::get_child(
        const LegionColor c, RtEvent* defer, bool can_fail)
    //--------------------------------------------------------------------------
    {
      // See if we have it locally if not go find it
      IndexPartition remote_handle = IndexPartition::NO_PART;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        std::map<LegionColor, IndexPartNode*>::const_iterator finder =
            color_map.find(c);
        if ((finder != color_map.end()) && (finder->second != nullptr))
        {
          if (can_fail)
            finder->second->add_base_resource_ref(REGION_TREE_REF);
          return finder->second;
        }
        std::map<LegionColor, IndexPartition>::const_iterator remote_finder =
            remote_colors.find(c);
        if (remote_finder != remote_colors.end())
          remote_handle = remote_finder->second;
      }
      // if we make it here, send a request
      AddressSpaceID owner_space = get_owner_space();
      if (owner_space == runtime->address_space)
      {
        if (remote_handle.exists())
        {
          IndexPartNode* result = runtime->get_node(remote_handle, defer);
          if (can_fail && (result != nullptr))
            result->add_base_resource_ref(REGION_TREE_REF);
          return result;
        }
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find entry for color " << c << " in "
              << "index space " << handle.get_id() << ".";
        error.raise();
      }
      RtUserEvent ready_event = Runtime::create_rt_user_event();

      DistributedID child_id = 0;
      IndexSpaceChildRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(c);
        if (defer == nullptr)
          rez.serialize(&child_id);
        else
          rez.serialize<DistributedID*>(nullptr);
        rez.serialize(ready_event);
      }
      rez.dispatch(owner_space);
      if (defer == nullptr)
      {
        ready_event.wait();
        if (child_id == 0)
        {
          if (can_fail)
            return nullptr;
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find entry for color " << c << " in "
                << "index space " << handle.get_id() << ".";
          error.raise();
        }
        IndexPartition child_handle(
            child_id, handle.get_tree_id(), handle.get_type_tag());
        IndexPartNode* result = runtime->get_node(child_handle);
        if (can_fail)
          result->add_base_resource_ref(REGION_TREE_REF);
        // Always unpack the global ref that got sent back with this
        result->unpack_global_ref();
        return result;
      }
      else
      {
        *defer = ready_event;
        return nullptr;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_child(IndexPartNode* child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // Can have a nullptr pointer
      legion_assert(
          (color_map.find(child->color) == color_map.end()) ||
          (color_map[child->color] == nullptr));
      if (is_owner() &&
          (remote_colors.find(INVALID_COLOR) != remote_colors.end()) &&
          (color_map.find(child->color) == color_map.end()))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal request for Legion to generated a color for index "
              << "space " << handle.get_id()
              << " after a child was already registered with an "
              << "explicit color. Colors of partitions must either be "
              << "completely specified by the user or completely generated "
              << "by the runtime. Mixing of allocation modes is not allowed.";
        error.raise();
      }
      color_map[child->color] = child;
      if (!remote_colors.empty())
        remote_colors.erase(child->color);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      std::map<LegionColor, IndexPartNode*>::iterator finder =
          color_map.find(c);
      legion_assert(finder != color_map.end());
      legion_assert(finder->second != nullptr);
      legion_assert(finder->second != ((IndexPartNode*)REMOVED_CHILD));
      finder->second = (IndexPartNode*)REMOVED_CHILD;
      while ((finder->first == next_uncollected_color) &&
             (finder->second == ((IndexPartNode*)REMOVED_CHILD)))
      {
        next_uncollected_color++;
        color_map.erase(finder);
        if (color_map.empty())
          break;
        finder = color_map.begin();
      }
    }

    //--------------------------------------------------------------------------
    size_t IndexSpaceNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock, false /*exclusive*/);
      return color_map.size();
    }

    //--------------------------------------------------------------------------
    RtEvent IndexSpaceNode::get_ready_event(void)
    //--------------------------------------------------------------------------
    {
      if (index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      AutoLock n_lock(node_lock);
      if (index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      if (!index_space_ready.exists())
        index_space_ready = Runtime::create_rt_user_event();
      return index_space_ready;
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::are_disjoint(LegionColor c1, LegionColor c2)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (c1 == c2)
        return false;
      if (c1 > c2)
        std::swap(c1, c2);
      const std::pair<LegionColor, LegionColor> key(c1, c2);
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        if (disjoint_subsets.find(key) != disjoint_subsets.end())
          return true;
        else if (aliased_subsets.find(key) != aliased_subsets.end())
          return false;
      }
      IndexPartNode* left = get_child(c1);
      IndexPartNode* right = get_child(c2);
      const bool intersects =
          left->intersects_with(right, !runtime->disable_independence_tests);
      AutoLock n_lock(node_lock);
      if (intersects)
      {
        aliased_subsets.insert(key);
        return false;
      }
      else
      {
        disjoint_subsets.insert(key);
        return true;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::record_remote_child(
        IndexPartition pid, LegionColor part_color)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      legion_assert(is_owner());
      legion_assert(
          (remote_colors.find(part_color) == remote_colors.end()) ||
          (remote_colors[part_color] == pid));
      // should only happen on the owner node
      legion_assert(get_owner_space() == runtime->address_space);
      if ((remote_colors.find(INVALID_COLOR) != remote_colors.end()) &&
          (color_map.find(part_color) == color_map.end()))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal request for Legion to generated a color for index "
              << "space " << handle.get_id()
              << " after a child was already registered with an "
              << "explicit color. Colors of partitions must either be "
              << "completely specified by the user or completely generated "
              << "by the runtime. Mixing of allocation modes is not allowed.";
        error.raise();
      }
      remote_colors[part_color] = pid;
    }

    //--------------------------------------------------------------------------
    LegionColor IndexSpaceNode::get_colors(std::vector<LegionColor>& colors)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, we need to request an up to date set of colors
      // since it can change arbitrarily
      AddressSpaceID owner_space = get_owner_space();
      if (owner_space != runtime->address_space)
      {
        LegionColor bound = INVALID_COLOR;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        IndexSpaceColorsRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(&colors);
          rez.serialize(&bound);
          rez.serialize(ready_event);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        legion_assert(bound != INVALID_COLOR);
        return bound;
      }
      else
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        for (const std::pair<const LegionColor, IndexPartNode*>& it : color_map)
        {
          // Can be nullptr in some cases of parallel partitioning
          if ((it.second != nullptr) &&
              (!it.second->initialized.exists() ||
               it.second->initialized.has_triggered()))
            colors.emplace_back(it.first);
        }
        return next_uncollected_color;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::IndexSpaceSetFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target == runtime->address_space)
        return;
      if (target == source)
        return;
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_node(
        AddressSpaceID target, bool recurse, bool valid)
    //--------------------------------------------------------------------------
    {
      // Quick out if we've already sent this
      if (has_remote_instance(target))
        return;
      // Send our parent first if necessary
      if (recurse && (parent != nullptr))
        parent->send_node(target, true /*recurse*/);
      // Only send it if we're the owner without a collective mapping
      // or the target is not in the collective mapping and we're the
      // closest node in the collective mapping to the target
      if ((is_owner() && (collective_mapping == nullptr)) ||
          ((collective_mapping != nullptr) &&
           !collective_mapping->contains(target) &&
           collective_mapping->contains(local_space) &&
           (local_space == collective_mapping->find_nearest(target))))
      {
        AutoLock n_lock(node_lock);
        legion_assert(is_global());
        legion_assert(is_valid() || !recurse);
        if (!has_remote_instance(target))
        {
          IndexSpaceResponse rez;
          pack_node(rez, target, recurse, valid);
          rez.dispatch(target);
          update_remote_instances(target);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::pack_node(
        Serializer& rez, AddressSpaceID target, bool recurse, bool valid)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(handle);
      if (recurse && (parent != nullptr))
        rez.serialize(parent->handle);
      else
        rez.serialize(IndexPartition::NO_PART);
      rez.serialize(color);
      rez.serialize(expr_id);
      rez.serialize(initialized);
      rez.serialize(depth);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      if (collective_mapping != nullptr)
        collective_mapping->pack(rez);
      else
        rez.serialize<size_t>(0);  // total spaces
      rez.serialize<bool>(valid);  // whether the tree is valid or not
      rez.serialize<size_t>(semantic_info.size());
      for (lng::map<SemanticTag, SemanticInfo>::iterator it =
               semantic_info.begin();
           it != semantic_info.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize<size_t>(it->second.buffer.get_size());
        rez.serialize(
            it->second.buffer.get_buffer(), it->second.buffer.get_size());
        rez.serialize(it->second.is_mutable);
      }
      if (index_space_set && ((collective_mapping == nullptr) ||
                              !collective_mapping->contains(target)))
      {
        rez.serialize<bool>(true);
        pack_index_space(rez, 1 /*reference count*/);
      }
      else
        rez.serialize<bool>(false);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::invalidate_root(
        AddressSpaceID source, std::set<RtEvent>& applied,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      legion_assert(parent == nullptr);
      bool need_broadcast = true;
      if (source == local_space)
      {
        // Entry point
        if (mapping != nullptr)
        {
          if ((collective_mapping != nullptr) &&
              ((mapping == collective_mapping) ||
               (*mapping == *collective_mapping)))
          {
            need_broadcast = false;
          }
          else if (mapping->contains(owner_space))
          {
            if (local_space != owner_space)
              return false;
          }
          else
          {
            // Find the one closest to the owner space
            const AddressSpaceID nearest = mapping->find_nearest(owner_space);
            if (nearest != local_space)
              return false;
            runtime->send_index_space_destruction(handle, owner_space, applied);
            // If we're part of the broadcast tree then we'll get sent back here
            // later so we don't need to do anything now
            if ((collective_mapping != nullptr) &&
                collective_mapping->contains(local_space))
              return false;
          }
        }
        else
        {
          // If we're not the owner space, send the message there
          if (!is_owner())
          {
            runtime->send_index_space_destruction(handle, owner_space, applied);
            return false;
          }
        }
      }
      if (need_broadcast && (collective_mapping != nullptr) &&
          collective_mapping->contains(local_space))
      {
        // Should be from our parent
        legion_assert(
            is_owner() || (source == collective_mapping->get_parent(
                                         owner_space, local_space)));
        // Keep broadcasting this out to all the children
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        for (const AddressSpaceID& child : children)
          runtime->send_index_space_destruction(handle, child, applied);
      }
      return remove_base_valid_ref(APPLICATION_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexPartition parent;
      derez.deserialize(parent);
      LegionColor color;
      derez.deserialize(color);
      IndexSpaceExprID expr_id;
      derez.deserialize(expr_id);
      RtEvent initialized;
      derez.deserialize(initialized);
      unsigned depth;
      derez.deserialize(depth);
      AutoProvenance provenance(
          Provenance::deserialize(derez), true /*has ref*/);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = nullptr;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);
      bool valid;
      derez.deserialize<bool>(valid);

      IndexPartNode* parent_node = nullptr;
      if (parent != IndexPartition::NO_PART)
        parent_node = runtime->get_node(parent);
      IndexSpaceNode* node = runtime->create_node(
          handle, Domain::NO_DOMAIN, true /*take ownership*/, parent_node,
          color, initialized, provenance, ApEvent::NO_AP_EVENT, expr_id,
          mapping, false /*add root reference*/, depth, valid);
      legion_assert(node != nullptr);
      size_t num_semantic;
      derez.deserialize(num_semantic);
      for (unsigned idx = 0; idx < num_semantic; idx++)
      {
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void* buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(
            tag, source, buffer, buffer_size, is_mutable, false /*local only*/);
      }
      bool has_index_space;
      derez.deserialize(has_index_space);
      if (has_index_space && node->unpack_index_space(derez, source))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      AddressSpaceID source;
      derez.deserialize(source);
      IndexSpaceNode* target =
          runtime->get_node(handle, nullptr, true /*can fail*/);
      bool valid = false;
      if (target != nullptr)
      {
        // If there is a collective mapping, check to see if we're on the
        // right node and if not forward it on to the right node
        if (target->collective_mapping != nullptr)
        {
          legion_assert(!target->collective_mapping->contains(source));
          legion_assert(
              target->collective_mapping->contains(target->local_space));
          if (target->is_owner())
          {
            const AddressSpaceID nearest =
                target->collective_mapping->find_nearest(source);
            // If we're not the nearest then forward it on to the
            // proper node to handle the request
            if (nearest != target->local_space)
            {
              IndexSpaceRequest rez;
              rez.serialize(handle);
              rez.serialize(to_trigger);
              rez.serialize(source);
              rez.dispatch(nearest);
              return;
            }
          }
          else
          {
            legion_assert(
                target->local_space ==
                target->collective_mapping->find_nearest(source));
          }
        }
        // See if we're going to be sending the whole tree or not
        bool recurse = false;
        if (target->parent == nullptr)
        {
          if (target->check_valid_and_increment(REGION_TREE_REF))
          {
            valid = true;
            target->pack_valid_ref();
            target->remove_base_valid_ref(REGION_TREE_REF);
          }
          else
            target->pack_global_ref();
        }
        else
        {
          // If we have a parent then we need to do the valid reference
          // check on the partition since that keeps this tree valid
          if (target->parent->check_valid_and_increment(REGION_TREE_REF))
          {
            valid = true;
            recurse = true;
            target->parent->pack_valid_ref();
            target->parent->remove_base_valid_ref(REGION_TREE_REF);
          }
          else
          {
            // We need the state to remain the same while we are in
            // transit so see if this can still be made valid
            if (target->check_valid_and_increment(REGION_TREE_REF))
            {
              valid = true;
              target->pack_valid_ref();
              target->remove_base_valid_ref(REGION_TREE_REF);
            }
            else
              target->pack_global_ref();
          }
        }
        target->send_node(source, recurse, valid);
        // Now send back the results
        IndexSpaceReturn rez;
        {
          RezCheck z(rez);
          rez.serialize(to_trigger);
          rez.serialize(handle);
          rez.serialize(valid);
          rez.serialize(recurse);
        }
        rez.dispatch(source);
      }
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceReturn::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode* node = runtime->get_node(handle);
      bool valid;
      derez.deserialize(valid);
      bool recurse;
      derez.deserialize(recurse);
      if (valid)
      {
        if (recurse)
          node->parent->unpack_valid_ref();
        else
          node->unpack_valid_ref();
      }
      else
        node->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceChildRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      LegionColor child_color;
      derez.deserialize(child_color);
      DistributedID* target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      IndexSpaceNode* parent = runtime->get_node(handle);
      RtEvent defer;
      IndexPartNode* child =
          parent->get_child(child_color, &defer, true /*can fail*/);
      if (defer.exists())
      {
        // Build a continuation and run it when the node is
        // ready, we have to do this in order to avoid blocking
        // the virtual channel for nested index tree requests
        IndexSpaceNode::DeferChildArgs args(
            parent, child_color, target, to_trigger, source);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, defer);
        return;
      }
      if (child != nullptr)
      {
        if (child->check_global_and_increment(REGION_TREE_REF))
        {
          IndexSpaceChildResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(child->handle);
            rez.serialize(target);
            rez.serialize(to_trigger);
            child->pack_global_ref();
          }
          rez.dispatch(source);
          if (child->remove_base_gc_ref(REGION_TREE_REF))
            delete child;
        }
        else  // can fail and unable to get a global reference
          Runtime::trigger_event(to_trigger);
        if (child->remove_base_resource_ref(REGION_TREE_REF))
          delete child;
      }
      else  // Failed so just trigger the result
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::DeferChildArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      IndexPartNode* child =
          proxy_this->get_child(child_color, nullptr, true /*can fail*/);
      if (child != nullptr)
      {
        if (child->check_global_and_increment(REGION_TREE_REF))
        {
          IndexSpaceChildResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(child->handle);
            rez.serialize(target);
            rez.serialize(to_trigger);
            child->pack_global_ref();
          }
          rez.dispatch(source);
          if (child->remove_base_gc_ref(REGION_TREE_REF))
            delete child;
        }
        else  // Unable to get a global reference
          Runtime::trigger_event(to_trigger);
        if (child->remove_base_resource_ref(REGION_TREE_REF))
          delete child;
      }
      else  // Failed so just trigger the result
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceChildResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      DistributedID* target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      if (target == nullptr)
      {
        // In this case we need to block here to make sure we can
        // unpack the global reference we added on the remote node
        // since there's nothing on the local node that is going to do it
        IndexPartNode* child = runtime->get_node(handle);
        child->unpack_global_ref();
        Runtime::trigger_event(to_trigger);
      }
      else
      {
        RtEvent defer;
        runtime->get_node(handle, &defer);
        // We'll update references and unpack the remote reference on
        // the requester here so there's no need to block waiting
        *target = handle.get_id(false /*filter*/);
        Runtime::trigger_event(to_trigger, defer);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceColorsRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      std::vector<LegionColor>* target;
      derez.deserialize(target);
      LegionColor* bound_target;
      derez.deserialize(bound_target);
      RtUserEvent ready;
      derez.deserialize(ready);
      IndexSpaceNode* node = runtime->get_node(handle);
      std::vector<LegionColor> results;
      LegionColor bound = node->get_colors(results);
      IndexSpaceColorsResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize<size_t>(results.size());
        for (const LegionColor& result : results) rez.serialize(result);
        rez.serialize(bound_target);
        rez.serialize(bound);
        rez.serialize(ready);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceColorsResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::vector<LegionColor>* target;
      derez.deserialize(target);
      size_t num_colors;
      derez.deserialize(num_colors);
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        LegionColor cp;
        derez.deserialize(cp);
        target->emplace_back(cp);
      }
      LegionColor* bound_target;
      derez.deserialize(bound_target);
      derez.deserialize(*bound_target);
      RtUserEvent ready;
      derez.deserialize(ready);
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceSet::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition parent_handle;
      derez.deserialize(parent_handle);
      if (parent_handle.exists())
      {
        LegionColor color;
        derez.deserialize(color);
        IndexPartNode* parent = runtime->get_node(parent_handle);
        IndexSpaceNode* child = parent->get_child(color);
        if (child->unpack_index_space(derez, source))
          delete child;
      }
      else
      {
        IndexSpace handle;
        derez.deserialize(handle);
        IndexSpaceNode* node = runtime->get_node(handle);
        if (node->unpack_index_space(derez, source))
          delete node;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceGenerateColorRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      LegionColor suggestion;
      derez.deserialize(suggestion);
      std::atomic<LegionColor>* target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      IndexSpaceNode* node = runtime->get_node(handle);
      LegionColor result = node->generate_color(suggestion);
      if (result != suggestion)
      {
        IndexSpaceGenerateColorResponse rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize(result);
          rez.serialize(done_event);
        }
        rez.dispatch(source);
      }
      else  // if we matched the suggestion we know the value is right
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceGenerateColorResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<LegionColor>* target;
      derez.deserialize(target);
      LegionColor result;
      derez.deserialize(result);
      target->store(result);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceReleaseColor::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      LegionColor color;
      derez.deserialize(color);

      IndexSpaceNode* node = runtime->get_node(handle);
      node->release_color(color);
      node->unpack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::record_index_space_user(ApEvent user)
    //--------------------------------------------------------------------------
    {
      if (user.exists() && !user.has_triggered_faultignorant())
      {
        AutoLock n_lock(node_lock);
        // Try popping entries off the front of the list
        while (!index_space_users.empty() &&
               index_space_users.front().has_triggered_faultignorant())
          index_space_users.pop_front();
        index_space_users.emplace_back(user);
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::pack_expression(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != runtime->address_space)
      {
        rez.serialize<bool>(false /*local*/);
        rez.serialize<bool>(true /*index space*/);
        rez.serialize(handle);
        pack_global_ref();
      }
      else
      {
        rez.serialize<bool>(true /*local*/);
        rez.serialize<IndexSpaceExpression*>(this);
        add_base_expression_reference(LIVE_EXPR_REF);
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::skip_unpack_expression(Deserializer& derez) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_canonical_reference(DistributedID source)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::try_add_live_reference(void)
    //--------------------------------------------------------------------------
    {
      if (check_global_and_increment(LIVE_EXPR_REF))
      {
        ImplicitReferenceTracker::record_live_expression(this);
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_base_expression_reference(
        ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_nested_expression_reference(
        DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_base_expression_reference(
        ReferenceSource source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_base_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_nested_expression_reference(
        DistributedID source, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_gc_ref(source, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::is_below_in_tree(
        IndexPartNode* partition, LegionColor& child) const
    //--------------------------------------------------------------------------
    {
      const IndexSpaceNode* node = this;
      while ((node->parent != nullptr) &&
             (node->parent->depth <= partition->depth))
      {
        if (node->parent == partition)
        {
          child = node->color;
          return true;
        }
        node = node->parent->parent;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_tree_expression_reference(
        DistributedID id, unsigned count)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(id, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::remove_tree_expression_reference(
        DistributedID id, unsigned count)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(id, count);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::intersects_with(IndexSpaceNode* rhs, bool compute)
    //--------------------------------------------------------------------------
    {
      legion_assert(rhs->handle.get_type_tag() == handle.get_type_tag());
      if (rhs == this)
        return true;
      // We're about to do something expensive so if these are both
      // in the same index space tree then walk up to a common partition
      // (if one exists) and see if it is disjoint
      if ((handle.get_tree_id() == rhs->handle.get_tree_id()) &&
          (parent != rhs->parent))
      {
        IndexSpaceNode* one = this;
        IndexSpaceNode* two = rhs;
        // Get them at the same depth
        while (one->depth > two->depth) one = one->parent->parent;
        while (one->depth < two->depth) two = two->parent->parent;
        // Handle the case where one dominates the other
        if (one == two)
          return true;
        // Now walk up until their parent is the same
        while (one->parent != two->parent)
        {
          one = one->parent->parent;
          two = two->parent->parent;
        }
        // If they have the same parent and it's not nullptr and
        // it is disjoint then they don't intersect if they are different
        if ((one->parent != nullptr) && (one != two) &&
            one->parent->is_disjoint())
          return false;
        // Otherwise fall through and do the expensive test
      }
      if (!compute)
        return true;
      IndexSpaceExpression* intersect =
          runtime->intersect_index_spaces(this, rhs);
      return !intersect->is_empty();
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::intersects_with(IndexPartNode* rhs, bool compute)
    //--------------------------------------------------------------------------
    {
      return rhs->intersects_with(this, compute);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::dominates(IndexSpaceNode* rhs)
    //--------------------------------------------------------------------------
    {
      legion_assert(rhs->handle.get_type_tag() == handle.get_type_tag());
      if (rhs == this)
        return true;
      // We're about to do something expensive, so use the region tree
      // as an acceleration data structure to try to make our tests
      // more efficient. If these are in the same tree, see if we can
      // walk up the tree from rhs and find ourself
      if (handle.get_tree_id() == rhs->handle.get_tree_id())
      {
        // If we're the root of the tree we also trivially dominate
        if (depth == 0)
          return true;
        if (rhs->depth > depth)
        {
          IndexSpaceNode* temp = rhs;
          while (depth < temp->depth) temp = temp->parent->parent;
          // If we find ourself at the same depth then we dominate
          if (temp == this)
            return true;
        }
        // Otherwise we fall through and do the expensive test
      }
      IndexSpaceExpression* diff = runtime->subtract_index_spaces(rhs, this);
      return diff->is_empty();
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(
        IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_sp,
        LegionColor c, bool dis, int comp, RtEvent init,
        CollectiveMapping* mapping, Provenance* prov)
      : IndexTreeNode(
            par->depth + 1, c, p.did, init, mapping, prov, true /*tree valid*/),
        handle(p), parent(par), color_space(color_sp),
        total_children(color_sp->get_volume()),
        max_linearized_color(color_sp->get_max_linearized_color()),
        total_children_volume(0), total_intersection_volume(0),
        has_disjoint(true), disjoint(dis), has_complete(comp >= 0),
        complete(comp != 0), first_entry(nullptr)
    //--------------------------------------------------------------------------
    {
      parent->add_nested_resource_ref(did);
      color_space->add_nested_resource_ref(did);
      legion_assert(handle.get_type_tag() == parent->handle.get_type_tag());
#ifdef LEGION_GC
      log_garbage.info(
          "GC Index Partition %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.did);
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(
        IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* color_sp,
        LegionColor c, int comp, RtEvent init, CollectiveMapping* map,
        Provenance* prov)
      : IndexTreeNode(
            par->depth + 1, c, p.did, init, map, prov, true /*tree valid*/),
        handle(p), parent(par), color_space(color_sp),
        total_children(color_sp->get_volume()),
        max_linearized_color(color_sp->get_max_linearized_color()),
        total_children_volume(0), total_intersection_volume(0),
        has_disjoint(false), disjoint(true), has_complete(comp >= 0),
        complete(comp != 0), first_entry(nullptr)
    //--------------------------------------------------------------------------
    {
      parent->add_nested_resource_ref(did);
      color_space->add_nested_resource_ref(did);
      legion_assert(handle.get_type_tag() == parent->handle.get_type_tag());
#ifdef LEGION_GC
      log_garbage.info(
          "GC Index Partition %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.did);
#endif
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::initialize_disjoint_complete_notifications(void)
    //--------------------------------------------------------------------------
    {
      // Figure out how many notifications we're waiting for
      if (is_owner() || ((collective_mapping != nullptr) &&
                         collective_mapping->contains(local_space)))
      {
        remaining_local_disjoint_complete_notifications = 0;
        // Count how many locat notifications we're going to get
        for (ColorSpaceIterator itr(this, true /*local only*/); itr; itr++)
          remaining_local_disjoint_complete_notifications++;
        // One for the disjointness task that will run
        if (remaining_local_disjoint_complete_notifications > 0)
          remaining_global_disjoint_complete_notifications = 1;
        else
          remaining_global_disjoint_complete_notifications = 0;
        // More notifications from any remote nodes
        if (collective_mapping != nullptr)
          remaining_global_disjoint_complete_notifications +=
              collective_mapping->count_children(owner_space, local_space);
        if (remaining_global_disjoint_complete_notifications == 0)
        {
          legion_assert(!is_owner());
          const AddressSpaceID target =
              collective_mapping->get_parent(owner_space, local_space);
          IndexPartitionDisjointUpdate rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<int>(1);  // up and compressed
            rez.serialize(total_children_volume);
            rez.serialize(total_intersection_volume);
          }
          rez.dispatch(target, initialized);
        }
      }
      else
        remaining_global_disjoint_complete_notifications = 0;
      // Add a reference to be removed only after both the disjointness
      // and the completeness is set
      add_base_valid_ref(REGION_TREE_REF);
    }

    //--------------------------------------------------------------------------
    IndexPartNode::~IndexPartNode(void)
    //--------------------------------------------------------------------------
    {
      // The reason we would be here is if we were leaked
      if (!partition_trackers.empty())
      {
        for (PartitionTracker* tracker : partition_trackers)
          if (tracker->remove_partition_reference())
            delete tracker;
        partition_trackers.clear();
      }
      // Lastly we can unregister ourselves with the context
      if (registered_with_runtime)
        runtime->remove_node(handle);
      if (parent->remove_nested_resource_ref(did))
        delete parent;
      if (color_space->remove_nested_resource_ref(did))
        delete color_space;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // Remove the valid reference that we hold on the color space
      if (color_space->parent != nullptr)
        color_space->parent->remove_nested_valid_ref(did);
      else
        color_space->remove_nested_valid_ref(did);
      // Remove valid ref on partition of parent if it exists, otherwise
      // our parent index space is a root so we remove the reference there
      if (parent->parent != nullptr)
      {
        if (parent->parent->remove_nested_valid_ref(did))
          delete parent->parent;
      }
      else
        parent->remove_nested_valid_ref(did);
      // Remove valid references on all owner children and any trackers
      // We should not need a lock at this point since nobody else should
      // be modifying the color map
      for (const std::pair<const LegionColor, IndexSpaceNode*>& it : color_map)
        // Remove the nested valid reference on this index space node
        if (it.second->remove_nested_valid_ref(did))
          std::abort();  // still holding resource ref so should never be hit
      if (!partition_trackers.empty())
      {
        for (PartitionTracker* tracker : partition_trackers)
          if (tracker->remove_partition_reference())
            delete tracker;
        partition_trackers.clear();
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      parent->remove_child(color);
      for (const std::pair<const LegionColor, IndexSpaceNode*>& it : color_map)
        if (it.second->remove_nested_gc_ref(did))
          delete it.second;
      color_map.clear();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_index_space_node(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

#ifdef LEGION_DEBUG
    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::as_index_space_node(void)
    //--------------------------------------------------------------------------
    {
      return nullptr;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexPartNode::as_index_part_node(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }
#endif

    //--------------------------------------------------------------------------
    AddressSpaceID IndexPartNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID IndexPartNode::get_owner_space(
        IndexPartition part)
    //--------------------------------------------------------------------------
    {
      return (part.get_id() % runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* IndexPartNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::send_semantic_request(
        AddressSpaceID target, SemanticTag tag, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      IndexPartSemanticInfoRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::send_semantic_info(
        AddressSpaceID target, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      IndexPartSemanticInfoResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::process_semantic_request(
        SemanticTag tag, AddressSpaceID source, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(get_owner_space() == runtime->address_space);
      RtEvent precondition;
      const void* result = nullptr;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == nullptr)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::SemanticRequestArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->process_semantic_request(
          tag, source, false, false, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartSemanticInfoRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      IndexPartNode* node = runtime->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartSemanticInfoResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      runtime->get_node(handle)->attach_semantic_information(
          tag, source, buffer, size, is_mutable, false /*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      return color_space->contains_color(c);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID IndexPartNode::find_color_creator_space(
        LegionColor color, CollectiveMapping*& child_mapping) const
    //--------------------------------------------------------------------------
    {
      legion_assert(child_mapping == nullptr);
      if (collective_mapping == nullptr)
      {
        if (is_owner())
          return local_space;
        else
          return owner_space;
      }
      else
      {
        // See whether the children are sharded or replicated
        if (((LegionColor)collective_mapping->size()) <= total_children)
        {
          // Sharded, so figure out which space to send the request to
          const size_t chunk =
              (max_linearized_color + collective_mapping->size() - 1) /
              collective_mapping->size();
          const unsigned offset = color / chunk;
          legion_assert(offset < collective_mapping->size());
          return (*collective_mapping)[offset];
        }
        else
        {
          // Replicated so find the child collective mapping
          std::vector<AddressSpaceID> child_spaces;
          const unsigned offset = color_space->compute_color_offset(color);
          legion_assert(offset < collective_mapping->size());
          for (unsigned idx = offset; idx < collective_mapping->size();
               idx += total_children)
            child_spaces.emplace_back((*collective_mapping)[idx]);
          legion_assert(!child_spaces.empty());
          child_mapping = new CollectiveMapping(
              child_spaces, runtime->legion_collective_radix);
          if (child_mapping->contains(local_space))
            return local_space;
          else
            return child_mapping->find_nearest(local_space);
        }
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::get_child(
        const LegionColor c, RtEvent* defer)
    //--------------------------------------------------------------------------
    {
      // First check to see if we can find it
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        std::map<LegionColor, IndexSpaceNode*>::const_iterator finder =
            color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      if (!color_space->contains_color(c, false /*report error*/))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid color space color for child " << c << " of partition "
              << handle.get_id();
        error.raise();
      }
      // Retake the lock and see if we're going to be the one responsible
      // for trying to make the child on this node
      RtUserEvent ready_event;
      {
        AutoLock n_lock(node_lock);
        // Make sure we didn't lose the race
        std::map<LegionColor, IndexSpaceNode*>::const_iterator finder =
            color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
        // See if we're the first ones to make this child
        std::map<LegionColor, RtUserEvent>::iterator pending_finder =
            pending_child_map.find(c);
        if (pending_finder != pending_child_map.end())
        {
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          ready_event = pending_finder->second;
        }
        else
          pending_child_map[c] = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (!ready_event.exists())
      {
        // See if we need to send a request to get the handle for this
        CollectiveMapping* child_mapping = nullptr;
        AddressSpaceID creator_space =
            find_color_creator_space(c, child_mapping);
        if (creator_space != local_space)
        {
          if (child_mapping != nullptr)
            delete child_mapping;
          // Find or get the ready event to wait on
          AutoLock n_lock(node_lock);
          std::map<LegionColor, IndexSpaceNode*>::const_iterator finder =
              color_map.find(c);
          if (finder != color_map.end())
            return finder->second;
          // If we get here then we need to send the request
          IndexPartitionChildRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(c);
          }
          rez.dispatch(creator_space);
          // Make sure we have to event to wait on for when the child is ready
          std::map<LegionColor, RtUserEvent>::iterator pending_finder =
              pending_child_map.find(c);
          legion_assert(pending_finder != pending_child_map.end());
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          ready_event = pending_finder->second;
        }
        else if (
            (child_mapping != nullptr) &&
            (local_space != child_mapping->get_origin()))
        {
          // We're not the origin that will make IDs, so retake the lock
          // and see if the child has appeared, if not record ourselves as
          // a pending waiter for it
          delete child_mapping;
          AutoLock n_lock(node_lock);
          std::map<LegionColor, IndexSpaceNode*>::const_iterator finder =
              color_map.find(c);
          if (finder != color_map.end())
            return finder->second;
          std::map<LegionColor, RtUserEvent>::iterator pending_finder =
              pending_child_map.find(c);
          legion_assert(pending_finder != pending_child_map.end());
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          ready_event = pending_finder->second;
        }
        else
        {
          // If we get here then we're the ones to actually make the name
          // of the index subspace and instantiate the node
          legion_assert(
              is_owner() || ((collective_mapping != nullptr) &&
                             collective_mapping->contains(local_space)));
          legion_assert(
              (child_mapping == nullptr) ||
              (local_space == child_mapping->get_origin()));
          IndexSpace is(
              runtime->get_unique_index_space_id(), handle.get_tree_id(),
              handle.get_type_tag());
          const IndexSpaceExprID expr_id =
              runtime->get_unique_index_space_expr_id();
          // Make a new index space node ready when the partition is ready
          IndexSpaceNode* result = runtime->create_node(
              is, *this, c, initialized, provenance, expr_id, child_mapping);
          if ((child_mapping != nullptr) && (child_mapping->size() > 1))
          {
            // We know other participants are nodes are going to need
            // these IDs so broadcast them up to the other nodes that
            // are also going to consider child as a local child
            std::vector<AddressSpaceID> children;
            child_mapping->get_children(local_space, local_space, children);
            legion_assert(!children.empty());
            IndexPartitionChildReplication rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(c);
              rez.serialize(is);
              rez.serialize(expr_id);
              child_mapping->pack(rez);
            }
            for (const AddressSpaceID& child : children) rez.dispatch(child);
          }
          LegionSpy::log_index_subspace(
              handle.get_id(), is.get_id(), runtime->address_space,
              result->get_domain_point_color());
          if (implicit_profiler != nullptr)
            implicit_profiler->register_index_subspace(
                handle.get_id(false /*filter*/), is.get_id(),
                result->get_domain_point_color());
          return result;
        }
      }
      legion_assert(ready_event.exists());
      if (defer == nullptr)
      {
        ready_event.wait();
        return get_child(c);
      }
      else
      {
        *defer = ready_event;
        return nullptr;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionChildReplication::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition parent_handle;
      derez.deserialize(parent_handle);
      LegionColor child_color;
      derez.deserialize(child_color);
      IndexSpace child_handle;
      derez.deserialize(child_handle);
      IndexSpaceExprID expr_id;
      derez.deserialize(expr_id);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      legion_assert(num_spaces > 0);
      CollectiveMapping* mapping = new CollectiveMapping(derez, num_spaces);

      IndexPartNode* parent = runtime->get_node(parent_handle);
      runtime->create_node(
          child_handle, *parent, child_color, parent->initialized,
          parent->provenance, expr_id, mapping);
      std::vector<AddressSpaceID> children;
      mapping->get_children(
          mapping->get_origin(), parent->local_space, children);
      if (!children.empty())
      {
        IndexPartitionChildReplication rez;
        {
          RezCheck z2(rez);
          rez.serialize(parent_handle);
          rez.serialize(child_color);
          rez.serialize(child_handle);
          rez.serialize(expr_id);
          mapping->pack(rez);
        }
        for (const AddressSpaceID& child : children) rez.dispatch(child);
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_child(IndexSpaceNode* child)
    //--------------------------------------------------------------------------
    {
      // This child should live as long as we are alive
      child->add_nested_gc_ref(did);
      child->add_nested_valid_ref(did);
      RtUserEvent to_trigger;
      {
        AutoLock n_lock(node_lock);
        legion_assert(is_valid());
        legion_assert(color_map.find(child->color) == color_map.end());
        color_map[child->color] = child;
        std::map<LegionColor, RtUserEvent>::iterator finder =
            pending_child_map.find(child->color);
        if (finder != pending_child_map.end())
        {
          if (finder->second.exists())
            to_trigger = finder->second;
          pending_child_map.erase(finder);
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::set_child(IndexSpaceNode* child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (!has_disjoint || !has_complete)
      {
        legion_assert(remaining_local_disjoint_complete_notifications > 0);
        legion_assert(remaining_global_disjoint_complete_notifications > 0);
        if (--remaining_local_disjoint_complete_notifications == 0)
        {
          // Launch the task to perform the local disjointness
          // and completeness tests
          DisjointnessArgs args(this);
          runtime->issue_runtime_meta_task(
              args, LG_THROUGHPUT_DEFERRED_PRIORITY, initialized);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_tracker(PartitionTracker* tracker)
    //--------------------------------------------------------------------------
    {
      std::vector<PartitionTracker*> to_prune;
      {
        AutoLock n_lock(node_lock);
        // To avoid leaks, see if there are any other trackers we can prune
        for (std::list<PartitionTracker*>::iterator it =
                 partition_trackers.begin();
             it != partition_trackers.end();
             /*nothing*/)
        {
          if ((*it)->can_prune())
          {
            to_prune.emplace_back(*it);
            it = partition_trackers.erase(it);
          }
          else
            it++;
        }
        partition_trackers.emplace_back(tracker);
      }
      for (PartitionTracker* tracker : to_prune)
        if (tracker->remove_reference())
          delete tracker;
    }

    //--------------------------------------------------------------------------
    size_t IndexPartNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return color_space->get_volume();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::needs_subspace_broadcast(void) const
    //--------------------------------------------------------------------------
    {
      // Only need to broadcast if the number of children is larger than
      // the number of participants creating this partition
      if (collective_mapping == nullptr)
        return true;
      else
        return (collective_mapping->size() < total_children);
    }

    //--------------------------------------------------------------------------
    IndexPartNode::RemoteDisjointnessFunctor::RemoteDisjointnessFunctor(
        IndexPartitionDisjointUpdate& r)
      : rez(r)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexPartNode::RemoteDisjointnessFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != runtime->address_space)
        rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::compute_disjointness_and_completeness(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          is_owner() || ((collective_mapping != nullptr) &&
                         collective_mapping->contains(local_space)));
      if (is_complete(false /*from app*/, true /*false if not ready*/) ||
          is_disjoint(false /*from app*/, true /*false if not ready*/))
      {
        // If we know we're complete, then we can check disjointness
        // simply by summing up the volume of all the children and
        // seeing if it equals the total volume of the parent. If it
        // does then we must be disjoint since any aliasing would result
        // in a volume that is larger than the volume of the parent.
        //
        // If we know we're disjoint, then we can check completeness
        // simply by summing up the volume of all the children and seeing
        // if it equals the the total volume of the parent. If it does then
        // we must be complete because there is no aliasing of the subspaces.
        //
        // If we have no collective mapping or the number of children is
        // larger than the collective mapping then we can eagerly sum
        // the volumes together, otherwise we need to keep them separte
        // so we can deduplicate the volumes across nodes
        if ((collective_mapping == nullptr) ||
            (((LegionColor)collective_mapping->size()) <= total_children))
        {
          // Children are sharded so no need to worry about uniqueness
          uint64_t children_volume = 0;
          for (ColorSpaceIterator itr(this, true /*local only*/); itr; itr++)
          {
            IndexSpaceNode* child = get_child(*itr);
            children_volume += child->get_volume();
          }
          return update_disjoint_complete_result(children_volume);
        }
        else
        {
          // Worry about uniqueness of children in this case
          std::map<LegionColor, uint64_t> children_volumes;
          for (ColorSpaceIterator itr(this, true /*local only*/); itr; itr++)
          {
            IndexSpaceNode* child = get_child(*itr);
            children_volumes[*itr] = child->get_volume();
          }
          return update_disjoint_complete_result(children_volumes);
        }
      }
      else
      {
        // In this case we don't know anything so we're computing both
        // disjointness and completeness at the same time.
        // To check for disjointness we look for any neighboring
        // children that alias. If we find any then we know that we
        // are not disjoint. To check for completeness, we count the
        // total volume of all the children and then subtract off the
        // volumes of the intersections from any interfering children
        // with a lower legion color to deduplicate counts. To compute
        // this difference for a given color C we first compute the union
        // of all the interfering children with colors <C and then subtract
        // that off the C to create a differende D, then we sum the
        // intersection of all the remaining interfering children with D
        // Try drawing yourself n-way venn diagrams to convince yourself
        // this is correct and will count all overlapping points exactly once.
        if ((collective_mapping == nullptr) ||
            (((LegionColor)collective_mapping->size()) <= total_children))
        {
          // Children are sharded so no need to worry about uniqueness
          uint64_t children_volume = 0;
          uint64_t intersection_volume = 0;
          for (ColorSpaceIterator itr(this, true /*local only*/); itr; itr++)
          {
            IndexSpaceNode* child = get_child(*itr);
            size_t child_volume = child->get_volume();
            if (child_volume == 0)
              continue;
            children_volume += child_volume;
            std::vector<LegionColor> interfering;
            if (!find_interfering_children_kd(child, interfering))
            {
              interfering.reserve(total_children);
              for (ColorSpaceIterator itr2(this); itr2; itr2++)
                interfering.push_back(*itr2);
              legion_assert(
                  std::is_sorted(interfering.begin(), interfering.end()));
            }
            else
              std::sort(interfering.begin(), interfering.end());
            std::vector<LegionColor>::const_iterator split =
                std::lower_bound(interfering.begin(), interfering.end(), *itr);
            // Better find our split value
            legion_assert(split != interfering.end());
            legion_assert(*split == *itr);
            // for cases where there is all-to-all interference we try to
            // avoid everyone stampeding on the colors in order, so we start
            // with loading the colors below our color in reverse order and
            // then we wrap around to do the colors above our split value
            // also in reverse order
            std::set<IndexSpaceExpression*> previous;
            std::vector<LegionColor>::const_iterator next = split;
            while (next != interfering.begin())
            {
              next = std::prev(next);
              IndexSpaceExpression* intersection =
                  runtime->intersect_index_spaces(child, get_child(*next));
              if (!intersection->is_empty())
                previous.insert(intersection);
            }
            IndexSpaceExpression* difference =
                previous.empty() ?
                    child :
                    runtime->subtract_index_spaces(
                        child, runtime->union_index_spaces(previous));
            // Now we can compute the intersection volume by wrapping
            // around to the end of the list of colors
            if (!difference->is_empty())
            {
              next = std::prev(interfering.end());
              while (next != split)
              {
                IndexSpaceNode* other = get_child(*next);
                IndexSpaceExpression* intersection =
                    runtime->intersect_index_spaces(difference, other);
                intersection_volume += intersection->get_volume();
                next = std::prev(next);
              }
            }
          }
          return update_disjoint_complete_result(
              children_volume, intersection_volume);
        }
        else
        {
          std::map<LegionColor, uint64_t> children_volumes;
          std::map<std::pair<LegionColor, LegionColor>, uint64_t>
              intersection_volumes;
          // Children are not sharded so we need to worry about aliasing
          // across the nodes for the same children
          for (ColorSpaceIterator itr(this, true /*local only*/); itr; itr++)
          {
            IndexSpaceNode* child = get_child(*itr);
            size_t child_volume = child->get_volume();
            children_volumes[*itr] = child_volume;
            if (child_volume == 0)
              continue;
            std::vector<LegionColor> interfering;
            if (!find_interfering_children_kd(child, interfering))
            {
              interfering.reserve(total_children);
              for (ColorSpaceIterator itr2(this); itr2; itr2++)
                interfering.push_back(*itr2);
              legion_assert(
                  std::is_sorted(interfering.begin(), interfering.end()));
            }
            else
              std::sort(interfering.begin(), interfering.end());
            std::vector<LegionColor>::const_iterator split =
                std::lower_bound(interfering.begin(), interfering.end(), *itr);
            // Better find our split value
            legion_assert(split != interfering.end());
            legion_assert(*split == *itr);
            // for cases where there is all-to-all interference we try to
            // avoid everyone stampeding on the colors in order, so we start
            // with loading the colors below our color in reverse order and
            // then we wrap around to do the colors above our split value
            // also in reverse order
            std::set<IndexSpaceExpression*> previous;
            std::vector<LegionColor>::const_iterator next = split;
            while (next != interfering.begin())
            {
              next = std::prev(next);
              IndexSpaceExpression* intersection =
                  runtime->intersect_index_spaces(child, get_child(*next));
              if (!intersection->is_empty())
                previous.insert(intersection);
            }
            IndexSpaceExpression* difference =
                previous.empty() ?
                    child :
                    runtime->subtract_index_spaces(
                        child, runtime->union_index_spaces(previous));
            if (!difference->is_empty())
            {
              next = std::prev(interfering.end());
              while (next != split)
              {
                IndexSpaceNode* other = get_child(*next);
                IndexSpaceExpression* intersection =
                    runtime->intersect_index_spaces(difference, other);
                if (!intersection->is_empty())
                  intersection_volumes.emplace(
                      std::make_pair(*itr, *next), intersection->get_volume());
                next = std::prev(next);
              }
            }
          }
          return update_disjoint_complete_result(
              children_volumes, &intersection_volumes);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::update_disjoint_complete_result(
        uint64_t children_volume, uint64_t intersection_volume)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      total_children_volume += children_volume;
      total_intersection_volume += intersection_volume;
      // Check to see if we've seen all our arrivals
      legion_assert(remaining_global_disjoint_complete_notifications > 0);
      if (--remaining_global_disjoint_complete_notifications == 0)
      {
        if (is_owner())
          return finalize_disjoint_complete();
        else
        {
          // Send the result up the tree
          const AddressSpaceID target =
              collective_mapping->get_parent(owner_space, local_space);
          IndexPartitionDisjointUpdate rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<int>(1);  // up and compressed
            rez.serialize(total_children_volume);
            rez.serialize(total_intersection_volume);
          }
          rez.dispatch(target, initialized);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::update_disjoint_complete_result(
        std::map<LegionColor, uint64_t>& children_volumes,
        std::map<std::pair<LegionColor, LegionColor>, uint64_t>*
            intersection_volumes)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (!total_children_volumes.empty())
      {
        for (const std::pair<const LegionColor, uint64_t>& it :
             children_volumes)
          total_children_volumes.insert(it);
      }
      else
        total_children_volumes.swap(children_volumes);
      if (intersection_volumes != nullptr)
      {
        if (!total_intersection_volumes.empty())
        {
          total_intersection_volumes.merge(*intersection_volumes);
          if (!intersection_volumes->empty())
          {
#ifdef LEGION_DEBUG
            for (const std::pair<
                     const std::pair<LegionColor, LegionColor>, uint64_t>& it :
                 *intersection_volumes)
              legion_assert(total_intersection_volumes[it.first] == it.second);
#endif
            intersection_volumes->clear();
          }
        }
        else
          total_intersection_volumes.swap(*intersection_volumes);
      }
      // Check to see if we've seen all our arrivals
      legion_assert(remaining_global_disjoint_complete_notifications > 0);
      if (--remaining_global_disjoint_complete_notifications == 0)
      {
        if (is_owner())
        {
          // We can now compute the final sums
          legion_assert(total_children_volume == 0);
          legion_assert(total_intersection_volume == 0);
          for (const std::pair<const LegionColor, uint64_t>& it :
               total_children_volumes)
            total_children_volume += it.second;
          total_children_volumes.clear();
          for (const std::pair<
                   const std::pair<LegionColor, LegionColor>, uint64_t>& it :
               total_intersection_volumes)
            total_intersection_volume += it.second;
          total_intersection_volumes.clear();
          return finalize_disjoint_complete();
        }
        else
        {
          // Send the result up the tree
          const AddressSpaceID target =
              collective_mapping->get_parent(owner_space, local_space);
          IndexPartitionDisjointUpdate rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<int>(-1);  // up and not compressed
            rez.serialize<size_t>(total_children_volumes.size());
            for (const std::pair<const LegionColor, uint64_t>& it :
                 total_children_volumes)
            {
              rez.serialize(it.first);
              rez.serialize(it.second);
            }
            rez.serialize<size_t>(total_intersection_volumes.size());
            for (const std::pair<
                     const std::pair<LegionColor, LegionColor>, uint64_t>& it :
                 total_intersection_volumes)
            {
              rez.serialize(it.first.first);
              rez.serialize(it.first.second);
              rez.serialize(it.second);
            }
          }
          rez.dispatch(target, initialized);
          total_children_volumes.clear();
          total_intersection_volumes.clear();
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::finalize_disjoint_complete(void)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        const size_t parent_volume = parent->get_volume();
        // We can now tell what our status is
        if (is_complete(false /*from app*/, true /*false if not ready*/))
        {
          legion_assert(parent_volume <= total_children_volume);
          disjoint.store((parent_volume == total_children_volume));
        }
        else if (is_disjoint(false /*from app*/, true /*false if not ready*/))
        {
          legion_assert(total_children_volume <= parent_volume);
          complete.store((parent_volume == total_children_volume));
        }
        else
        {
          legion_assert(
              (total_children_volume - total_intersection_volume) <=
              parent_volume);
          if (total_intersection_volume == 0)
          {
            disjoint.store(true);
            legion_assert((total_children_volume <= parent_volume));
            complete.store((total_children_volume == parent_volume));
          }
          else
          {
            disjoint.store(false);
            legion_assert(total_intersection_volume < total_children_volume);
            total_children_volume -= total_intersection_volume;
            legion_assert(total_children_volume <= parent_volume);
            complete.store((parent_volume == total_children_volume));
          }
        }
        if (implicit_profiler != nullptr)
          implicit_profiler->register_index_partition(
              parent->handle.get_id(), handle.get_id(), disjoint.load(), color);
      }
      has_disjoint.store(true);
      has_complete.store(true);
      if (disjoint_complete_ready.exists())
        Runtime::trigger_event(disjoint_complete_ready);
      if ((collective_mapping != nullptr) &&
          collective_mapping->contains(local_space))
      {
        // Broadcast the result out to the children
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        IndexPartitionDisjointUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<int>(0);  // down
          rez.serialize<bool>(disjoint.load());
          rez.serialize<bool>(complete.load());
        }
        for (const AddressSpaceID& child : children) rez.dispatch(child);
      }
      if (has_remote_instances())
      {
        IndexPartitionDisjointUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<int>(0);  // down
          rez.serialize<bool>(disjoint.load());
          rez.serialize<bool>(complete.load());
        }
        RemoteDisjointnessFunctor functor(rez);
        map_over_remote_instances(functor);
      }
      return remove_base_valid_ref(REGION_TREE_REF);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_disjoint(bool app_query, bool false_if_not_ready)
    //--------------------------------------------------------------------------
    {
      if (has_disjoint.load())
        return disjoint.load();
      if (false_if_not_ready)
        return false;
      RtEvent wait_on;
      {
        AutoLock n_lock(node_lock);
        if (has_disjoint.load())
          return disjoint.load();
        if (!disjoint_complete_ready.exists())
          disjoint_complete_ready = Runtime::create_rt_user_event();
        wait_on = disjoint_complete_ready;
      }
      wait_on.wait();
      legion_assert(has_disjoint.load());
      return disjoint.load();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::are_disjoint(
        LegionColor c1, LegionColor c2, bool force_compute)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return false;
      if (!force_compute && is_disjoint(false /*appy query*/))
        return true;
      if (c1 > c2)
        std::swap(c1, c2);
      const std::pair<LegionColor, LegionColor> key(c1, c2);
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        if (disjoint_subspaces.find(key) != disjoint_subspaces.end())
          return true;
        else if (aliased_subspaces.find(key) != aliased_subspaces.end())
          return false;
      }
      // Perform the test
      IndexSpaceNode* left = get_child(c1);
      IndexSpaceNode* right = get_child(c2);
      const bool intersects =
          left->intersects_with(right, !runtime->disable_independence_tests);
      AutoLock n_lock(node_lock);
      if (intersects)
      {
        aliased_subspaces.insert(key);
        return false;
      }
      else
      {
        disjoint_subspaces.insert(key);
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_complete(
        bool from_app /*=false*/, bool false_if_not_ready /*=false*/)
    //--------------------------------------------------------------------------
    {
      if (has_complete.load())
        return complete.load();
      if (false_if_not_ready)
        return false;
      RtEvent wait_on;
      {
        AutoLock n_lock(node_lock);
        if (has_complete.load())
          return complete.load();
        if (!disjoint_complete_ready.exists())
          disjoint_complete_ready = Runtime::create_rt_user_event();
        wait_on = disjoint_complete_ready;
      }
      wait_on.wait();
      legion_assert(has_complete.load());
      return complete.load();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::handle_disjointness_update(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      int mode;
      derez.deserialize(mode);
      if (mode < 0)
      {
        // up and not compressed
        std::map<LegionColor, uint64_t> children_volumes;
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          LegionColor color;
          derez.deserialize(color);
          derez.deserialize(children_volumes[color]);
        }
        size_t num_intersections;
        derez.deserialize(num_intersections);
        std::map<std::pair<LegionColor, LegionColor>, uint64_t>
            intersection_volumes;
        for (unsigned idx = 0; idx < num_intersections; idx++)
        {
          std::pair<LegionColor, LegionColor> key;
          derez.deserialize(key.first);
          derez.deserialize(key.second);
          derez.deserialize(intersection_volumes[key]);
        }
        return update_disjoint_complete_result(
            children_volumes, &intersection_volumes);
      }
      else if (mode > 0)
      {
        // up and already compressed
        uint64_t children_volume, intersection_volume;
        derez.deserialize(children_volume);
        derez.deserialize(intersection_volume);
        return update_disjoint_complete_result(
            children_volume, intersection_volume);
      }
      else
      {
        // sending back down to the children
        bool is_disjoint, is_complete;
        derez.deserialize(is_disjoint);
        derez.deserialize(is_complete);
        AutoLock n_lock(node_lock);
        legion_assert(remaining_global_disjoint_complete_notifications == 0);
        disjoint.store(is_disjoint);
        complete.store(is_complete);
        return finalize_disjoint_complete();
      }
    }

    //--------------------------------------------------------------------------
    LegionColor IndexPartNode::get_colors(std::vector<LegionColor>& colors)
    //--------------------------------------------------------------------------
    {
      color_space->instantiate_colors(colors);
      if (!colors.empty())
        return colors.front();
      else
        return 0;
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_equal_children(
        Operation* op, size_t granularity)
    //--------------------------------------------------------------------------
    {
      return parent->create_equal_children(op, this, granularity);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_weights(
        Operation* op, const std::map<DomainPoint, FutureImpl*>& weights,
        size_t granularity)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_weights(op, this, weights, granularity);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_union(
        Operation* op, IndexPartNode* left, IndexPartNode* right)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_union(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_intersection(
        Operation* op, IndexPartNode* left, IndexPartNode* right)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_intersection(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_intersection(
        Operation* op, IndexPartNode* original, const bool dominates)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_intersection(op, this, original, dominates);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_difference(
        Operation* op, IndexPartNode* left, IndexPartNode* right)
    //--------------------------------------------------------------------------
    {
      return parent->create_by_difference(op, this, left, right);
    }

    //--------------------------------------------------------------------------
    ApEvent IndexPartNode::create_by_restriction(
        const void* transform, const void* extent)
    //--------------------------------------------------------------------------
    {
      return color_space->create_by_restriction(
          this, transform, extent,
          NT_TemplateHelper::get_dim(handle.get_type_tag()));
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::DisjointnessArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (proxy_this->compute_disjointness_and_completeness())
        delete proxy_this;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::intersects_with(IndexSpaceNode* rhs, bool compute)
    //--------------------------------------------------------------------------
    {
      legion_assert(rhs->handle.get_type_tag() == handle.get_type_tag());
      // A very simple test but an obvious one
      if ((rhs->parent == this) || (parent == rhs))
        return true;
      // We're about to do something expensive so if these are both
      // in the same index space tree then walk up to a common partition
      // if one exists and see if it is disjoint
      if (handle.get_tree_id() == rhs->handle.get_tree_id())
      {
        IndexSpaceNode* one = parent;
        IndexSpaceNode* two = rhs;
        // Get them at the same depth
        while (one->depth > two->depth) one = one->parent->parent;
        while (one->depth < two->depth) two = two->parent->parent;
        // Handle the case where one dominates the other
        if (one == two)
          return true;
        // Now walk up until their parent is the same
        while (one->parent != two->parent)
        {
          one = one->parent->parent;
          two = two->parent->parent;
        }
        // If they have the same parent and it's not nullptr and
        // it is disjoint then they don't intersect if they are different
        if ((one->parent != nullptr) && (one != two) &&
            one->parent->is_disjoint())
          return false;
        // Otherwise fall through and do the expensive test
      }
      if (!compute)
        return true;
      std::vector<LegionColor> interfering;
      if (find_interfering_children_kd(rhs, interfering))
        return !interfering.empty();
      for (ColorSpaceIterator itr(this); itr; itr++)
      {
        IndexSpaceNode* child = get_child(*itr);
        IndexSpaceExpression* intersect =
            runtime->intersect_index_spaces(child, rhs);
        if (!intersect->is_empty())
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::intersects_with(IndexPartNode* rhs, bool compute)
    //--------------------------------------------------------------------------
    {
      legion_assert(rhs->handle.get_type_tag() == handle.get_type_tag());
      // A very simple but obvious test to do
      if (rhs == this)
        return true;
      // Another special case: if they both have the same parent and at least
      // one of them is complete then they do alias
      if ((parent == rhs->parent) && (is_complete() || rhs->is_complete()))
        return true;
      // We're about to do something expensive so see if we can use
      // the region tree as an acceleration data structure first
      if ((handle.get_tree_id() == rhs->handle.get_tree_id()) &&
          (parent != rhs->parent))
      {
        // Parent's are not the same, go up until we find
        // parents with a common partition
        IndexSpaceNode* one = parent;
        IndexSpaceNode* two = rhs->parent;
        // Get them at the same depth
        while (one->depth > two->depth) one = one->parent->parent;
        while (one->depth < two->depth) two = two->parent->parent;
        // Handle the case where one dominates the other
        if (one == two)
          return true;
        // Now walk up until their parent is the same
        while (one->parent != two->parent)
        {
          one = one->parent->parent;
          two = two->parent->parent;
        }
        // If they have the same parent and it's not nullptr and
        // it is dijsoint then they don't intersect if they are different
        if ((one->parent != nullptr) && (one != two) &&
            one->parent->is_disjoint())
          return false;
        // Otherwise we fall through and do the expensive test
      }
      if (!compute)
        return true;
      if (parent != rhs->parent)
      {
        IndexSpaceExpression* intersect =
            runtime->intersect_index_spaces(parent, rhs->parent);
        if (intersect->is_empty())
          return false;
      }
      // TODO::intersect KD-trees?
      return true;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::find_interfering_children(
        IndexSpaceExpression* expr, std::vector<LegionColor>& colors)
    //--------------------------------------------------------------------------
    {
      legion_assert(colors.empty());
      // Check to see if we have this in the cache
      {
        AutoLock n_lock(node_lock);
        std::map<IndexSpaceExprID, InterferenceEntry>::iterator finder =
            interference_cache.find(expr->expr_id);
        if (finder != interference_cache.end())
        {
          if (finder->second.expr_id != first_entry->expr_id)
          {
            InterferenceEntry* entry = &finder->second;
            // Remove it from its place in line
            if (entry->older != nullptr)
              entry->older->newer = entry->newer;
            if (entry->newer != nullptr)
              entry->newer->older = entry->older;
            // Move it to the front of the line
            entry->newer = nullptr;
            entry->older = first_entry;
            first_entry->newer = entry;
            first_entry = entry;
          }
          // Record the result
          colors = finder->second.colors;
          return;
        }
      }
      // Do a quick test to see if this expression is below us in the
      // index space tree which makes this computation simple
      LegionColor below_color = 0;
      if (!is_disjoint() || !expr->is_below_in_tree(this, below_color))
      {
        // We can only test this here after we've ruled out the symbolic check
        if (expr->is_empty())
          return;
        if (!find_interfering_children_kd(expr, colors))
        {
          for (ColorSpaceIterator itr(this); itr; itr++)
          {
            IndexSpaceNode* child = get_child(*itr);
            IndexSpaceExpression* intersection =
                runtime->intersect_index_spaces(expr, child);
            if (!intersection->is_empty())
              colors.emplace_back(*itr);
          }
        }
      }
      else
        colors.emplace_back(below_color);
      // Save the result in the cache for the future
      AutoLock n_lock(node_lock);
      // If someone else beat us to it then we are done
      if (interference_cache.find(expr->expr_id) != interference_cache.end())
        return;
      // Insert it at the front
      InterferenceEntry* entry = &interference_cache[expr->expr_id];
      entry->expr_id = expr->expr_id;
      entry->colors = colors;
      if (first_entry != nullptr)
        first_entry->newer = entry;
      entry->older = first_entry;
      first_entry = entry;
      if (interference_cache.size() > MAX_INTERFERENCE_CACHE_SIZE)
      {
        // Remove the oldest entry in the cache
        InterferenceEntry* last_entry = first_entry;
        while (last_entry->older != nullptr) last_entry = last_entry->older;
        if (last_entry->newer != nullptr)
          last_entry->newer->older = nullptr;
        interference_cache.erase(last_entry->expr_id);
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::send_node(AddressSpaceID target, bool recurse)
    //--------------------------------------------------------------------------
    {
      legion_assert(recurse);
      legion_assert(parent != nullptr);
      // Quick out if we've already sent this
      if (has_remote_instance(target))
        return;
      parent->send_node(target, true /*recurse*/);
      color_space->send_node(target, true /*recurse*/);
      // Only send it if we're the owner without a collective mapping
      // or the target is not in the collective mapping and we're the
      // closest node in the collective mapping to the target
      if ((is_owner() && (collective_mapping == nullptr)) ||
          ((collective_mapping != nullptr) &&
           !collective_mapping->contains(target) &&
           collective_mapping->contains(local_space) &&
           (local_space == collective_mapping->find_nearest(target))))
      {
        AutoLock n_lock(node_lock);
        legion_assert(is_valid());
        if (!has_remote_instance(target))
        {
          IndexPartitionResponse rez;
          pack_node(rez, target);
          rez.dispatch(target);
          update_remote_instances(target);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::pack_node(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have computed the disjointness result
      // If not we'll record that we need to do it and then when it
      // is computed we'll send out the result to all the remote copies
      RezCheck z(rez);
      rez.serialize(handle);
      rez.serialize(parent->handle);
      rez.serialize(color_space->handle);
      rez.serialize(color);
      rez.serialize<bool>(has_disjoint.load());
      rez.serialize<bool>(disjoint.load());
      if (has_complete)
      {
        if (complete)
          rez.serialize<int>(1);  // complete
        else
          rez.serialize<int>(0);  // not complete
      }
      else
        rez.serialize<int>(-1);  // we don't know yet
      rez.serialize(initialized);
      if (collective_mapping != nullptr)
        collective_mapping->pack(rez);
      else
        rez.serialize<size_t>(0);  // total spaces
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      rez.serialize<size_t>(semantic_info.size());
      for (lng::map<SemanticTag, SemanticInfo>::iterator it =
               semantic_info.begin();
           it != semantic_info.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize<size_t>(it->second.buffer.get_size());
        rez.serialize(
            it->second.buffer.get_buffer(), it->second.buffer.get_size());
        rez.serialize(it->second.is_mutable);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexSpace parent;
      derez.deserialize(parent);
      IndexSpace color_space;
      derez.deserialize(color_space);
      LegionColor color;
      derez.deserialize(color);
      bool has_disjoint, disjoint;
      derez.deserialize(has_disjoint);
      derez.deserialize(disjoint);
      int complete;
      derez.deserialize(complete);
      RtEvent initialized;
      derez.deserialize(initialized);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = nullptr;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);
      AutoProvenance provenance(
          Provenance::deserialize(derez), true /*has ref*/);
      IndexSpaceNode* parent_node = runtime->get_node(parent);
      IndexSpaceNode* color_space_node = runtime->get_node(color_space);
      legion_assert(parent_node != nullptr);
      legion_assert(color_space_node != nullptr);
      IndexPartNode* node =
          has_disjoint ?
              runtime->create_node(
                  handle, parent_node, color_space_node, color, disjoint,
                  complete, provenance, initialized, mapping) :
              runtime->create_node(
                  handle, parent_node, color_space_node, color, complete,
                  provenance, initialized, mapping);
      legion_assert(node != nullptr);
      size_t num_semantic;
      derez.deserialize(num_semantic);
      for (unsigned idx = 0; idx < num_semantic; idx++)
      {
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void* buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(
            tag, source, buffer, buffer_size, is_mutable, false /*local only*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      IndexPartition handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      AddressSpaceID source;
      derez.deserialize(source);
      IndexPartNode* target =
          runtime->get_node(handle, nullptr, true /*can fail*/);
      if (target != nullptr)
      {
        // If there is a collective mapping, check to see if we're on the
        // right node and if not forward it on to the right node
        if (target->collective_mapping != nullptr)
        {
          legion_assert(!target->collective_mapping->contains(source));
          legion_assert(
              target->collective_mapping->contains(target->local_space));
          if (target->is_owner())
          {
            const AddressSpaceID nearest =
                target->collective_mapping->find_nearest(source);
            // If we're not the nearest then forward it on to the
            // proper node to handle the request
            if (nearest != target->local_space)
            {
              IndexPartitionRequest rez;
              rez.serialize(handle);
              rez.serialize(to_trigger);
              rez.serialize(source);
              rez.dispatch(nearest);
              return;
            }
          }
          else
          {
            legion_assert(
                target->local_space ==
                target->collective_mapping->find_nearest(source));
          }
        }
        target->pack_valid_ref();
        target->send_node(source, true /*recurse*/);
        // Now send back the results
        IndexPartitionReturn rez;
        {
          RezCheck z(rez);
          rez.serialize(to_trigger);
          rez.serialize(handle);
        }
        rez.dispatch(source);
      }
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionReturn::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode* node = runtime->get_node(handle);
      node->unpack_valid_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionChildRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      LegionColor child_color;
      derez.deserialize(child_color);
      IndexPartNode* parent = runtime->get_node(handle);
      RtEvent defer;
      IndexSpaceNode* child = parent->get_child(child_color, &defer);
      // If we got a deferral event then we need to make a continuation
      // to avoid blocking the virtual channel for nested index tree requests
      if (defer.exists())
      {
        IndexPartNode::DeferChildArgs args(parent, child_color, source);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, defer);
      }
      else
      {
        IndexPartitionChildResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(child->handle);
        }
        rez.dispatch(source);
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::DeferChildArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* child = proxy_this->get_child(child_color);
      IndexPartitionChildResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(child->handle);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::DeferFindShardRects::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (proxy_this->find_local_shard_rects())
        delete proxy_this;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionChildResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      runtime->find_or_request_node(handle, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionDisjointUpdate::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode* node = runtime->get_node(handle);
      if (node->handle_disjointness_update(derez))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionNotification::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition pid;
      derez.deserialize(pid);
      IndexSpace parent;
      derez.deserialize(parent);
      LegionColor part_color;
      derez.deserialize(part_color);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      IndexSpaceNode* parent_node = runtime->get_node(parent);
      parent_node->record_remote_child(pid, part_color);
      // Now we can trigger the done event
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexPartNode::request_shard_rects(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      std::vector<AddressSpaceID> children;
      {
        AutoLock n_lock(node_lock);
        if (shard_rects_ready.exists())
          return shard_rects_ready;
        shard_rects_ready = Runtime::create_rt_user_event();
        // Add a reference to keep this node alive until this all done
        add_base_gc_ref(RUNTIME_REF);
        // Figure out how many downstream requests we have
        collective_mapping->get_children(owner_space, local_space, children);
        // Need to see all our children notifications plus our local rectangles
        remaining_rect_notifications = children.size() + 1;
        initialize_shard_rects();
      }
      if (!children.empty())
      {
        IndexPartitionShardRectsRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
        }
        for (const AddressSpaceID& child : children) rez.dispatch(child);
      }
      // Compute our local shard rectangles
      if (find_local_shard_rects())
        std::abort();  // should never delete ourselves
      return shard_rects_ready;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::process_shard_rects_response(
        Deserializer& derez, AddressSpace source)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      bool up;
      derez.deserialize<bool>(up);
      if (up)
      {
        bool need_local = false;
        std::vector<AddressSpaceID> children;
        AutoLock n_lock(node_lock);
        if (!shard_rects_ready.exists())
        {
          // Not initialized, so do the initialization
          shard_rects_ready = Runtime::create_rt_user_event();
          // Add a reference to keep this node alive until this all done
          add_base_gc_ref(RUNTIME_REF);
          // Figure out how many downstream requests we have
          collective_mapping->get_children(owner_space, local_space, children);
          legion_assert(!children.empty());
#ifdef LEGION_DEBUG
          bool found = false;
          for (const AddressSpaceID& child : children)
          {
            if (child != source)
              continue;
            found = true;
            break;
          }
          legion_assert(found);
#endif
          need_local = true;
          remaining_rect_notifications = children.size() + 1;
          IndexPartitionShardRectsRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
          }
          for (const AddressSpaceID& child : children)
            if (child != source)
              rez.dispatch(child);
          initialize_shard_rects();
        }
#ifdef LEGION_DEBUG
        else
        {
          collective_mapping->get_children(owner_space, local_space, children);
          legion_assert(!children.empty());
          bool found = false;
          for (const AddressSpaceID& child : children)
          {
            if (child != source)
              continue;
            found = true;
            break;
          }
          legion_assert(found);
        }
#endif
        unpack_shard_rects(derez);
        if (perform_shard_rects_notification())
        {
          legion_assert(!need_local);
          return true;
        }
        else if (!need_local)
          return false;
      }
      else
      {
        // Going down
        AutoLock n_lock(node_lock);
        unpack_shard_rects(derez);
        legion_assert(shard_rects_ready.exists());
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        if (!children.empty())
        {
          IndexPartitionShardRectsResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(false);  // going down
            pack_shard_rects(rez, false /*clear*/);
          }
          for (const AddressSpaceID& child : children) rez.dispatch(child);
        }
        // Only trigger this after we've packed the shard rects since the
        // local node is going to mutate it with its own values after this
        Runtime::trigger_event(shard_rects_ready);
        return remove_base_gc_ref(RUNTIME_REF);
      }
      // If we get here then we need to kick off the local analysis
      return find_local_shard_rects();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::perform_shard_rects_notification(void)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
      legion_assert(remaining_rect_notifications > 0);
      if (--remaining_rect_notifications == 0)
      {
        if (is_owner())
        {
          legion_assert(shard_rects_ready.exists());
          std::vector<AddressSpaceID> children;
          collective_mapping->get_children(owner_space, local_space, children);
          // We've got all the data now, so we can broadcast it back out
          IndexPartitionShardRectsResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(false);  // sending down the tree now
            pack_shard_rects(rez, false /*clear*/);
          }
          // Only trigger this after we've packed the shard rects since the
          // local node is going to mutate it with its own values after this
          Runtime::trigger_event(shard_rects_ready);
          for (const AddressSpaceID& child : children) rez.dispatch(child);
          return remove_base_gc_ref(RUNTIME_REF);
        }
        else
        {
          // Continue propagating it back up the tree
          IndexPartitionShardRectsResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<bool>(true);  // still going up
            pack_shard_rects(rez, true /*clear*/);
          }
          rez.dispatch(
              collective_mapping->get_parent(owner_space, local_space));
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    IndexPartNode::RemoteKDTracker::RemoteKDTracker(void)
      : done_event(RtUserEvent::NO_RT_USER_EVENT), remaining(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RtEvent IndexPartNode::RemoteKDTracker::find_remote_interfering(
        const std::set<AddressSpaceID>& targets, IndexPartition handle,
        IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      legion_assert(remaining.load() == 0);
      legion_assert(!targets.empty());
      remaining.store(targets.size());
      for (const AddressSpaceID& target : targets)
      {
        if (target == runtime->address_space)
        {
          legion_assert(remaining.load() > 0);
          if (remaining.fetch_sub(1) == 1)
            return RtEvent::NO_RT_EVENT;
          continue;
        }
        IndexPartitionRemoteInterferenceRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          expr->pack_expression(rez, target);
          rez.serialize(this);
        }
        rez.dispatch(target);
      }
      RtEvent wait_on;
      {
        AutoLock t_lock(tracker_lock);
        if (remaining.load() == 0)
          return RtEvent::NO_RT_EVENT;
        done_event = Runtime::create_rt_user_event();
        wait_on = done_event;
      }
      return wait_on;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::RemoteKDTracker::get_remote_interfering(
        std::set<LegionColor>& colors)
    //--------------------------------------------------------------------------
    {
      // No need for the lock since we're done at this point
      if (!remote_colors.empty())
      {
        if (colors.empty())
          colors.swap(remote_colors);
        else
          colors.insert(remote_colors.begin(), remote_colors.end());
      }
    }

    //--------------------------------------------------------------------------
    RtUserEvent
        IndexPartNode::RemoteKDTracker::process_remote_interfering_response(
            Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_colors;
      derez.deserialize(num_colors);
      AutoLock t_lock(tracker_lock);
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        LegionColor color;
        derez.deserialize(color);
        remote_colors.insert(color);
      }
      legion_assert(remaining.load() > 0);
      if ((remaining.fetch_sub(1) == 1) && done_event.exists())
        return done_event;
      else
        return RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionShardRectsRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode* node = runtime->get_node(handle);
      node->request_shard_rects();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionShardRectsResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexPartNode* node = runtime->get_node(handle);
      if (node->process_shard_rects_response(derez, source))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionRemoteInterferenceRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      IndexPartNode::RemoteKDTracker* tracker;
      derez.deserialize(tracker);

      IndexPartNode* node = runtime->get_node(handle);
      std::vector<LegionColor> local_colors;
      node->find_interfering_children_kd(
          expr, local_colors, true /*local only*/);
      IndexPartitionRemoteInterferenceResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(tracker);
        rez.serialize<size_t>(local_colors.size());
        for (const LegionColor& color : local_colors) rez.serialize(color);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionRemoteInterferenceResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartNode::RemoteKDTracker* tracker;
      derez.deserialize(tracker);
      const RtUserEvent to_trigger =
          tracker->process_remote_interfering_response(derez);
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    /////////////////////////////////////////////////////////////
    // Color Space Iterator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ColorSpaceIterator::ColorSpaceIterator(
        IndexPartNode* partition, bool local_only)
      : color_space(partition->color_space)
    //--------------------------------------------------------------------------
    {
      simple_step =
          (partition->total_children == partition->max_linearized_color);
      if (local_only && (partition->collective_mapping != nullptr))
      {
        if (partition->collective_mapping->contains(partition->local_space))
        {
          const unsigned index =
              partition->collective_mapping->find_index(partition->local_space);
          const LegionColor total_spaces =
              partition->collective_mapping->size();
          if (partition->total_children < total_spaces)
          {
            // Just a single color to handle here
            current = 0;
            end = partition->max_linearized_color;
            const unsigned offset = index % partition->total_children;
            for (unsigned idx = 0; idx < offset; idx++) step();
            legion_assert(current < end);
            end = current + 1;
          }
          else
          {
            const LegionColor chunk =
                compute_chunk(partition->max_linearized_color, total_spaces);
            current = index * chunk;
            end = ((current + chunk) < partition->max_linearized_color) ?
                      (current + chunk) :
                      partition->max_linearized_color;
            if (!simple_step && (current < end) &&
                !color_space->contains_color(current))
              step();
          }
        }
        else
        {
          // There are no local points
          end = partition->max_linearized_color;
          current = end;
        }
      }
      else
      {
        legion_assert(!local_only || partition->is_owner());
        current = 0;
        end = partition->max_linearized_color;
      }
    }

    //--------------------------------------------------------------------------
    ColorSpaceIterator::ColorSpaceIterator(
        IndexPartNode* partition, ShardID shard, size_t total_shards)
      : color_space(partition->color_space)
    //--------------------------------------------------------------------------
    {
      simple_step =
          (partition->total_children == partition->max_linearized_color);
      const LegionColor chunk =
          (partition->max_linearized_color + total_shards - 1) / total_shards;
      current = shard * chunk;
      end = ((current + chunk) < partition->max_linearized_color) ?
                (current + chunk) :
                partition->max_linearized_color;
      if (!simple_step && (current < end) &&
          !color_space->contains_color(current))
        step();
    }

    //--------------------------------------------------------------------------
    /*static*/ LegionColor ColorSpaceIterator::compute_chunk(
        LegionColor max_color, size_t total_spaces)
    //--------------------------------------------------------------------------
    {
      return (max_color + total_spaces - 1) / total_spaces;
    }

    //--------------------------------------------------------------------------
    ColorSpaceIterator::operator bool(void) const
    //--------------------------------------------------------------------------
    {
      return (current < end);
    }

    //--------------------------------------------------------------------------
    LegionColor ColorSpaceIterator::operator*(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(current < end);
      return current;
    }

    //--------------------------------------------------------------------------
    ColorSpaceIterator& ColorSpaceIterator::operator++(int)
    //--------------------------------------------------------------------------
    {
      step();
      return *this;
    }

    //--------------------------------------------------------------------------
    void ColorSpaceIterator::step(void)
    //--------------------------------------------------------------------------
    {
      current++;
      if (!simple_step)
      {
        while ((current < end) && !color_space->contains_color(current))
          current++;
      }
    }

  }  // namespace Internal
}  // namespace Legion
