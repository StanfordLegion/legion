/* Copyright 2018 Stanford University, NVIDIA Corporation
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

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Templated Index Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM,T>::IndexSpaceNodeT(RegionTreeForest *ctx, 
        IndexSpace handle, IndexPartNode *parent, LegionColor color,
        const Realm::IndexSpace<DIM,T> *is, DistributedID did, ApEvent ready)
      : IndexSpaceNode(ctx, handle, parent, color, did, ready), 
        linearization_ready(false)
    //--------------------------------------------------------------------------
    {
      if (is != NULL)
      {
        realm_index_space = *is;
        Runtime::trigger_event(realm_index_space_set);
        tighten_index_space();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM,T>::IndexSpaceNodeT(const IndexSpaceNodeT &rhs)
      : IndexSpaceNode(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM,T>::~IndexSpaceNodeT(void)
    //--------------------------------------------------------------------------
    { 
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      local_space.destroy();
      for (typename std::map<IndexTreeNode*,IntersectInfo>::iterator it =
            intersections.begin(); it != intersections.end(); it++)
      {
        if (it->second.has_intersection && it->second.intersection_valid)
          it->second.intersection.destroy();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM,T>& IndexSpaceNodeT<DIM,T>::operator=(
                                                     const IndexSpaceNodeT &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline ApEvent IndexSpaceNodeT<DIM,T>::get_realm_index_space(
                      Realm::IndexSpace<DIM,T> &result, bool need_tight_result)
    //--------------------------------------------------------------------------
    {
      if (!tight_index_space)
      {
        if (need_tight_result)
        {
          // Wait for the index space to be tight
          tight_index_space_set.wait();
          // Fall through and get the result when we're done
        }
        else
        {
          if (!realm_index_space_set.has_triggered())
            realm_index_space_set.wait();
          // Not tight yet so still subject to change so we need the lock
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          result = realm_index_space;
          return index_space_ready;
        }
      }
      // At this point we have a tight index space
      // That means it's already ready
      result = realm_index_space;
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline void IndexSpaceNodeT<DIM,T>::set_realm_index_space(
                  AddressSpaceID source, const Realm::IndexSpace<DIM,T> &value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!realm_index_space_set.has_triggered());
#endif
      // We can set this now but triggering the realm_index_space_set
      // event has to be done while holding the node_lock on the owner
      // node so that it is serialized with respect to queries from 
      // remote nodes for copies about the remote instance
      realm_index_space = value;
      // If we're not the owner, send a message back to the
      // owner specifying that it can set the index space value
      const AddressSpaceID owner_space = get_owner_space();
      if (owner_space != context->runtime->address_space)
      {
        // We're not the owner so we can trigger the event without the lock
        Runtime::trigger_event(realm_index_space_set);
        // We're not the owner, if this is not from the owner then
        // send a message there telling the owner that it is set
        if (source != owner_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            pack_index_space(rez, false/*include size*/);
          }
          context->runtime->send_index_space_set(owner_space, rez);
        }
        
      }
      else
      {
        // Log subspaces being set on the owner
        if (implicit_runtime->legion_spy_enabled && (parent != NULL))
          this->log_index_space_points(realm_index_space);
        // Hold the lock while walking over the node set
        AutoLock n_lock(node_lock);
        // Now we can trigger the event while holding the lock
        Runtime::trigger_event(realm_index_space_set);
        if (!remote_instances.empty())
        {
          // We're the owner, send messages to everyone else that we've 
          // sent this node to except the source
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            pack_index_space(rez, false/*include size*/);
          }
          IndexSpaceSetFunctor functor(context->runtime, source, rez);
          remote_instances.map(functor); 
        }
      }
      // Now we can tighten it
      tighten_index_space();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::tighten_index_space(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!tight_index_space);
      assert(!tight_index_space_set.has_triggered());
#endif
      if (!index_space_ready.has_triggered())
      {
        // If this index space isn't ready yet, then we have to defer this 
        TightenIndexSpaceArgs args;
        args.proxy_this = this;
        context->runtime->issue_runtime_meta_task(args,LG_LATENCY_WORK_PRIORITY,
                                     Runtime::protect_event(index_space_ready));
        return;
      }
      Realm::IndexSpace<DIM,T> tight_space = realm_index_space.tighten();
      Realm::IndexSpace<DIM,T> old_space;
      // Now take the lock and set everything
      {
        AutoLock n_lock(node_lock);
        old_space = realm_index_space;
        realm_index_space = tight_space;
        __sync_synchronize(); // small memory fence to propagate writes
        tight_index_space = true;
      }
      Runtime::trigger_event(tight_index_space_set);
      old_space.destroy();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::initialize_union_space(ApUserEvent to_trigger,
                             TaskOp *op, const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'union_index_spaces' "
                        "performed in task %s (UID %lld)",
                        op->get_task_name(), op->get_unique_id())
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        ApEvent ready = space->get_realm_index_space(spaces[idx], false);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                      op, DEP_PART_UNION_REDUCTION);
      Realm::IndexSpace<DIM,T> result_space;
      ApEvent done(Realm::IndexSpace<DIM,T>::compute_union(
            spaces, result_space, requests, precondition));
      set_realm_index_space(context->runtime->address_space, result_space);
      Runtime::trigger_event(to_trigger, done);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::initialize_intersection_space(
     ApUserEvent to_trigger, TaskOp *op, const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'intersect_index_spaces' "
                        "performed in task %s (UID %lld)",
                        op->get_task_name(), op->get_unique_id())
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        ApEvent ready = space->get_realm_index_space(spaces[idx], false);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                      op, DEP_PART_INTERSECTION_REDUCTION);
      Realm::IndexSpace<DIM,T> result_space;
      ApEvent done(Realm::IndexSpace<DIM,T>::compute_intersection(
            spaces, result_space, requests, precondition));
      set_realm_index_space(context->runtime->address_space, result_space);
      Runtime::trigger_event(to_trigger, done);
    }
    
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::initialize_difference_space(
          ApUserEvent to_trigger, TaskOp *op, IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      if (left.get_type_tag() != right.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                      "Dynamic type mismatch in 'subtract_index_spaces' "
                      "performed in task %s (UID %lld)",
                      op->get_task_name(), op->get_unique_id())
      IndexSpaceNodeT<DIM,T> *left_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(context->get_node(left));
      IndexSpaceNodeT<DIM,T> *right_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(context->get_node(right));
      Realm::IndexSpace<DIM,T> left_space, right_space;
      ApEvent left_ready = left_node->get_realm_index_space(left_space, false);
      ApEvent right_ready = right_node->get_realm_index_space(right_space, 
                                                              false);
      ApEvent precondition = Runtime::merge_events(left_ready, right_ready);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_DIFFERENCE);
      Realm::IndexSpace<DIM,T> result_space;
      ApEvent done(Realm::IndexSpace<DIM,T>::compute_difference(
           left_space, right_space, result_space, requests, precondition));
      set_realm_index_space(context->runtime->address_space, result_space);
      Runtime::trigger_event(to_trigger, done);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_index_space_points(void)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> tight_space;
      get_realm_index_space(tight_space, true/*tight*/);
      log_index_space_points(tight_space);
    }
      
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_index_space_points(
                              const Realm::IndexSpace<DIM,T> &tight_space) const
    //--------------------------------------------------------------------------
    {
      if (!tight_space.empty())
      {
        // Iterate over the rectangles and print them out 
        for (Realm::IndexSpaceIterator<DIM,T> itr(tight_space); 
              itr.valid; itr.step())
        {
          if (itr.rect.volume() == 1)
            LegionSpy::log_index_space_point(handle.get_id(), 
                                             Point<DIM,T>(itr.rect.lo));
          else
            LegionSpy::log_index_space_rect(handle.get_id(), 
                                            Rect<DIM,T>(itr.rect));
        }
      }
      else
        LegionSpy::log_empty_index_space(handle.get_id());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::compute_pending_space(Operation *op,
                          const std::vector<IndexSpace> &handles, bool is_union)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          TaskContext *ctx = op->get_context();
          if (is_union)
            REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                          "Dynamic type mismatch in 'create_index_space_union' "
                          "performed in task %s (UID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id())
          else
            REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                          "Dynamic type mismatch in "
                          "'create_index_space_intersection' performed in "
                          "task %s (UID %lld)", ctx->get_task_name(),
                          ctx->get_unique_id())
        }
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        ApEvent ready = space->get_realm_index_space(spaces[idx], false);
        if (ready.exists())
          preconditions.insert(ready);
      }
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::IndexSpace<DIM,T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_UNION_REDUCTION);
        ApEvent result(Realm::IndexSpace<DIM,T>::compute_union(
              spaces, result_space, requests, precondition));
        set_realm_index_space(context->runtime->address_space, result_space);
        return result;
      }
      else
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                op, DEP_PART_INTERSECTION_REDUCTION);
        ApEvent result(Realm::IndexSpace<DIM,T>::compute_intersection(
              spaces, result_space, requests, precondition));
        set_realm_index_space(context->runtime->address_space, result_space);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::compute_pending_space(Operation *op,
                                      IndexPartition part_handle, bool is_union)
    //--------------------------------------------------------------------------
    {
      if (part_handle.get_type_tag() != handle.get_type_tag())
      {
        TaskContext *ctx = op->get_context();
        if (is_union)
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'create_index_space_union' "
                        "performed in task %s (UID %lld)",
                        ctx->get_task_name(), ctx->get_unique_id())
        else
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in "
                        "'create_index_space_intersection' performed in "
                        "task %s (UID %lld)", ctx->get_task_name(),
                        ctx->get_unique_id())
      }
      IndexPartNode *partition = context->get_node(part_handle);
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM,T> > 
        spaces(partition->color_space->get_volume());
      unsigned subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
          ApEvent ready = child->get_realm_index_space(spaces[subspace_index++],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      else
      {
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
          ApEvent ready = child->get_realm_index_space(spaces[subspace_index++],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::IndexSpace<DIM,T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_UNION_REDUCTION);
        ApEvent result(Realm::IndexSpace<DIM,T>::compute_union(
              spaces, result_space, requests, precondition));
        set_realm_index_space(context->runtime->address_space, result_space);
        return result;
      }
      else
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                op, DEP_PART_INTERSECTION_REDUCTION);
        ApEvent result(Realm::IndexSpace<DIM,T>::compute_intersection(
              spaces, result_space, requests, precondition));
        set_realm_index_space(context->runtime->address_space, result_space);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::compute_pending_difference(Operation *op,
                        IndexSpace init, const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      if (init.get_type_tag() != handle.get_type_tag())
      {
        TaskContext *ctx = op->get_context();
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                      "Dynamic type mismatch in "
                      "'create_index_space_difference' performed in "
                      "task %s (%lld)", ctx->get_task_name(), 
                      ctx->get_unique_id())
      }
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          TaskContext *ctx = op->get_context();
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in "
                        "'create_index_space_difference' performed in "
                        "task %s (%lld)", ctx->get_task_name(), 
                        ctx->get_unique_id())
        }
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        ApEvent ready = space->get_realm_index_space(spaces[idx], 
                                                     false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
      } 
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::ProfilingRequestSet union_requests;
      Realm::ProfilingRequestSet diff_requests;
      if (context->runtime->profiler != NULL)
      {
        context->runtime->profiler->add_partition_request(union_requests,
                                            op, DEP_PART_UNION_REDUCTION);
        context->runtime->profiler->add_partition_request(diff_requests,
                                            op, DEP_PART_DIFFERENCE);
      }
      // Compute the union of the handles for the right-hand side
      Realm::IndexSpace<DIM,T> rhs_space;
      ApEvent rhs_ready(Realm::IndexSpace<DIM,T>::compute_union(
            spaces, rhs_space, union_requests, precondition));
      IndexSpaceNodeT<DIM,T> *lhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(context->get_node(init));
      Realm::IndexSpace<DIM,T> lhs_space, result_space;
      ApEvent lhs_ready = lhs_node->get_realm_index_space(lhs_space, false);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_difference(
            lhs_space, rhs_space, result_space, diff_requests,
            Runtime::merge_events(lhs_ready, rhs_ready)));
      set_realm_index_space(context->runtime->address_space, result_space);
      // Destroy the tempory rhs space once the computation is done
      rhs_space.destroy(result);
      return result;
    } 

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::get_index_space_domain(void *realm_is, 
                                                        TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (type_tag != handle.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
            "Dynamic type mismatch in 'get_index_space_domain'")
      Realm::IndexSpace<DIM,T> *target = 
        static_cast<Realm::IndexSpace<DIM,T>*>(realm_is);
      // No need to wait since we're waiting for it to be tight
      // which implies that it will be ready
      get_realm_index_space(*target, true/*tight*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::get_volume(void)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> volume_space;
      get_realm_index_space(volume_space, true/*tight*/);
      return volume_space.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::get_num_dims(void) const
    //--------------------------------------------------------------------------
    {
      return DIM;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_point(const void *realm_point, 
                                                TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (type_tag != handle.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
            "Dynamic type mismatch in 'safe_cast'")
      const Realm::Point<DIM,T> *point = 
        static_cast<const Realm::Point<DIM,T>*>(realm_point);
      Realm::IndexSpace<DIM,T> test_space;
      // Wait for a tight space on which to perform the test
      get_realm_index_space(test_space, true/*tight*/);
      return test_space.contains(*point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_point(const Realm::Point<DIM,T> &p)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> test_space;
      // Wait for a tight space on which to perform the test
      get_realm_index_space(test_space, true/*tight*/);
      return test_space.contains(p);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (destroyed)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_INDEX_SPACE_DELETION,
            "Duplicate deletion of Index Space %d", handle.get_id())
      destroyed = true;
      // If we're not the owner, send a message that we're removing
      // the application reference
      if (!is_owner())
      {
        runtime->send_index_space_destruction(handle, owner_space);
        return false;
      }
      else
        return remove_base_valid_ref(APPLICATION_REF, NULL/*mutator*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::get_max_linearized_color(void)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> color_bounds;
      get_realm_index_space(color_bounds, true/*tight*/);
      return color_bounds.bounds.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::compute_linearization_metadata(void)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> space;
      get_realm_index_space(space, true/*tight*/);
      // Don't need to wait for full index space since we just need bounds
      const Realm::Rect<DIM,T> &bounds = space.bounds;
      const long long volume = bounds.volume();
      if (volume > 0)
      {
        long long stride = 1;
        for (int idx = 0; idx < DIM; idx++)
        {
          offset[idx] = bounds.lo[idx];
          strides[idx] = stride;
          stride *= ((bounds.hi[idx] - bounds.lo[idx]) + 1);
        }
#ifdef DEBUG_LEGION
        assert(stride == volume);
#endif
      }
      else
      {
        for (int idx = 0; idx < DIM; idx++)
        {
          offset[idx] = 0;
          strides[idx] = 0;
        }
      }
      linearization_ready = true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::linearize_color(const void *realm_color,
                                                        TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(type_tag == handle.get_type_tag());
#endif
      if (!linearization_ready)
        compute_linearization_metadata();
      Realm::Point<DIM,T> point = 
        *(static_cast<const Realm::Point<DIM,T>*>(realm_color));
      // First subtract the offset to get to the origin
      point -= offset;
      LegionColor color = 0;
      for (int idx = 0; idx < DIM; idx++)
        color += point[idx] * strides[idx];
      return color;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::delinearize_color(LegionColor color,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(type_tag == handle.get_type_tag());
#endif
      if (!linearization_ready)
        compute_linearization_metadata();
      Realm::Point<DIM,T> &point = 
        *(static_cast<Realm::Point<DIM,T>*>(realm_color));
      for (int idx = DIM-1; idx >= 0; idx--)
      {
        point[idx] = color/strides[idx]; // truncates
        color -= point[idx] * strides[idx];
      }
      point += offset;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_color(LegionColor color, 
                                                bool report_error/*=false*/)
    //--------------------------------------------------------------------------
    {
      Realm::Point<DIM,T> point;
      delinearize_color(color, &point, handle.get_type_tag());
      Realm::IndexSpace<DIM,T> space;
      get_realm_index_space(space, true/*tight*/);
      if (!space.contains(point))
      {
        if (report_error)
          REPORT_LEGION_ERROR(ERROR_INVALID_INDEX_SPACE_COLOR,
              "Invalid color request")
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::instantiate_colors(
                                               std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      colors.resize(get_volume());
      unsigned idx = 0;
      Realm::IndexSpace<DIM,T> space;
      get_realm_index_space(space, true/*tight*/);
      for (Realm::IndexSpaceIterator<DIM,T> rect_itr(space); 
            rect_itr.valid; rect_itr.step())
      {
        for (Realm::PointInRectIterator<DIM,T> itr(rect_itr.rect);
              itr.valid; itr.step(), idx++)
          colors[idx] = linearize_color(&itr.p, handle.get_type_tag());
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Domain IndexSpaceNodeT<DIM,T>::get_color_space_domain(void)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> space;
      get_realm_index_space(space, true/*tight*/);
      return Domain(DomainT<DIM,T>(space));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainPoint IndexSpaceNodeT<DIM,T>::get_domain_point_color(void) const
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
        return DomainPoint(color);
      return parent->color_space->delinearize_color_to_point(color); 
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainPoint IndexSpaceNodeT<DIM,T>::delinearize_color_to_point(
                                                                  LegionColor c)
    //--------------------------------------------------------------------------
    {
      Realm::Point<DIM,T> color_point;
      delinearize_color(c, &color_point, handle.get_type_tag());
      return DomainPoint(Point<DIM,T>(color_point));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::intersects_with(IndexSpaceNode *rhs, 
                                                 bool compute)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      if (rhs == this)
        return true;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
          finder = intersections.find(rhs);
        // Only return the value if we either didn't want to compute
        // or we already have valid intersections
        if ((finder != intersections.end()) && 
            (!compute || finder->second.intersection_valid))
          return finder->second.has_intersection;
      }
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      if (!compute)
      {
        // If we just need the boolean result, do a quick test to
        // see if we do dominate it without needing the answer
        IndexSpaceNode *temp = rhs;
        while (temp->depth >= depth)
        {
          if (temp == this)
          {
            AutoLock n_lock(node_lock);
            intersections[rhs] = IntersectInfo(true/*result*/);
            return true;
          }
          if (temp->parent == NULL)
            break;
          temp = temp->parent->parent;
        }
        // Otherwise we fall through and do the expensive test
      }
      Realm::IndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent lhs_ready = get_realm_index_space(lhs_space, false);
      ApEvent rhs_ready = rhs_node->get_realm_index_space(rhs_space, false);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      ApEvent ready(Realm::IndexSpace<DIM,T>::compute_intersection(
        lhs_space, rhs_space, intersection, requests,
        Runtime::merge_events(lhs_ready, rhs_ready)));
      // Wait for the result to be ready
      if (!ready.has_triggered())
        ready.wait();
      // Always tighten these tests so that they are precise
      Realm::IndexSpace<DIM,T> tight_intersection = intersection.tighten();
      bool result = !tight_intersection.empty();
      AutoLock n_lock(node_lock);
      if (result)
      {
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder =
          intersections.find(rhs);
        // Check to make sure we didn't lose the race
        if ((finder == intersections.end()) || 
            (compute && !finder->second.intersection_valid))
          intersections[rhs] = IntersectInfo(tight_intersection);
        else
          tight_intersection.destroy(); // clean up spaces if we didn't save it
      }
      else
        intersections[rhs] = IntersectInfo(false/*result*/);
      intersection.destroy();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::intersects_with(IndexPartNode *rhs, 
                                                 bool compute)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
          finder = intersections.find(rhs);
        // Only return the value if we know we are valid and we didn't
        // want to compute anything or we already did compute it
        if ((finder != intersections.end()) &&
            (!compute || finder->second.intersection_valid))
          return finder->second.has_intersection;
      }
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      if (!compute)
      {
        // Before we do something expensive, let's do an easy test
        // just by walking the region tree
        IndexPartNode *temp = rhs;
        while ((temp != NULL) && (temp->parent->depth >= depth))
        {
          if (temp->parent == this)
          {
            AutoLock n_lock(node_lock);
            intersections[rhs] = IntersectInfo(true/*result*/);
            return true;
          }
          temp = temp->parent->parent;
        }
        // Otherwise we fall through and do the expensive test
      }
      Realm::IndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent lhs_ready = get_realm_index_space(lhs_space, false);
      ApEvent rhs_ready = rhs_node->get_union_index_space(rhs_space, false);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      ApEvent ready(Realm::IndexSpace<DIM,T>::compute_intersection(
            lhs_space, rhs_space, intersection, requests,
            Runtime::merge_events(lhs_ready, rhs_ready)));
      if (!ready.has_triggered())
        ready.wait();
      // Always tighten these tests so that they are precise
      Realm::IndexSpace<DIM,T> tight_intersection = intersection.tighten();
      bool result = !tight_intersection.empty();
      AutoLock n_lock(node_lock);
      if (result)
      {
        // Check to make sure we didn't lose the race
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder =
          intersections.find(rhs);
        if ((finder == intersections.end()) ||
            (compute && !finder->second.intersection_valid))
          intersections[rhs] = IntersectInfo(tight_intersection);
        else
          tight_intersection.destroy();
      }
      else
        intersections[rhs] = IntersectInfo(false/*result*/);
      intersection.destroy();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::dominates(IndexSpaceNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      if (rhs == this)
        return true;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(rhs);
        if (finder != dominators.end())
          return finder->second;
      }
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      // Before we do something expensive, let's do an easy test
      // just by walking the region tree
      IndexSpaceNode *temp = rhs; 
      while (temp->depth >= depth)
      {
        if (temp == this)
        {
          AutoLock n_lock(node_lock);
          dominators[rhs] = true;
          return true;
        }
        if (temp->parent == NULL)
          break;
        temp = temp->parent->parent;
      }
      // Otherwise we fall through and do the expensive test
      Realm::IndexSpace<DIM,T> local_space, rhs_space, difference; 
      get_realm_index_space(local_space, true/*tight*/);
      bool result = false;
      if (!local_space.dense())
      {
        ApEvent rhs_ready = rhs_node->get_realm_index_space(rhs_space, false);
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                        (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::IndexSpace<DIM,T>::compute_difference(
          rhs_space, local_space, difference, requests, rhs_ready));
        if (!ready.has_triggered())
          ready.wait();
        // Always tighten these tests so that they are precise
        Realm::IndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      }
      else // Fast path
      {
        rhs_node->get_realm_index_space(rhs_space, true/*tight*/);
        result = local_space.bounds.contains(rhs_space);
      }
      AutoLock n_lock(node_lock);
      dominators[rhs] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::dominates(IndexPartNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(rhs);
        if (finder != dominators.end())
          return finder->second;
      }
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      // Before we do something expensive, let's do an easy test
      // just by walking the region tree
      IndexPartNode *temp = rhs_node; 
      while ((temp != NULL) && (temp->parent->depth >= depth))
      {
        if (temp->parent == this)
        {
          AutoLock n_lock(node_lock);
          dominators[rhs] = true;
          return true;
        }
        temp = temp->parent->parent;
      }
      // Otherwise we fall through and do the expensive test
      Realm::IndexSpace<DIM,T> local_space, rhs_space, difference;
      get_realm_index_space(local_space, true/*tight*/);
      bool result = false;
      if (!local_space.dense())
      {
        ApEvent rhs_ready = rhs_node->get_union_index_space(rhs_space, false);
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                        (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::IndexSpace<DIM,T>::compute_difference(
              rhs_space, local_space, difference, requests, rhs_ready));
        if (!ready.has_triggered())
          ready.wait();
        // Always tighten these tests so that they are precise
        Realm::IndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      }
      else // Fast path
      {
        rhs_node->get_union_index_space(rhs_space, true/*tight*/);
        result = local_space.bounds.contains(rhs_space);
      }
      AutoLock n_lock(node_lock);
      dominators[rhs] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::pack_index_space(Serializer &rez,
                                                  bool include_size) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(realm_index_space_set.has_triggered());
#endif
      if (include_size)
        rez.serialize<size_t>(sizeof(realm_index_space));
      // No need for the lock, held by the caller
      rez.serialize(realm_index_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::unpack_index_space(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> result_space;
      derez.deserialize(result_space);
      set_realm_index_space(source, result_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_equal_children(Operation *op,
                                   IndexPartNode *partition, size_t granularity)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      const size_t count = partition->color_space->get_volume(); 
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                                op, DEP_PART_EQUAL);
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (op->has_execution_fence_event())
        ready = Runtime::merge_events(ready, op->get_execution_fence_event());
      ApEvent result(local_space.create_equal_subspaces(count, 
            granularity, subspaces, requests, ready));
#ifdef LEGION_SPY
      if (!result.exists() || (result == ready))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),handle,ready,result);
#endif
      // Enumerate the colors and assign the spaces
      unsigned subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      else
      {
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_union(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *left,
                                                    IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      const size_t count = partition->color_space->get_volume();
      std::vector<Realm::IndexSpace<DIM,T> > lhs_spaces(count);
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces(count);
      std::set<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      unsigned subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *left_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(left->get_child(color));
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent left_ready = 
            left_child->get_realm_index_space(lhs_spaces[subspace_index],
                                              false/*tight*/);
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (left_ready.exists())
            preconditions.insert(left_ready);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      else
      {
        for (LegionColor color = 0;
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *left_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent left_ready = 
            left_child->get_realm_index_space(lhs_spaces[subspace_index],
                                              false/*tight*/);
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (left_ready.exists())
            preconditions.insert(left_ready);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                              op, DEP_PART_UNIONS);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_unions(
            lhs_spaces, rhs_spaces, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    handle, precondition, result);
#endif
      // Now set the index spaces for the results
      subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      else
      {
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_intersection(Operation *op,
                                                      IndexPartNode *partition,
                                                      IndexPartNode *left,
                                                      IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      const size_t count = partition->color_space->get_volume();
      std::vector<Realm::IndexSpace<DIM,T> > lhs_spaces(count);
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces(count);
      std::set<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      unsigned subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *left_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(left->get_child(color));
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent left_ready = 
            left_child->get_realm_index_space(lhs_spaces[subspace_index],
                                              false/*tight*/);
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (left_ready.exists())
            preconditions.insert(left_ready);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      else
      {
        for (LegionColor color = 0;
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *left_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent left_ready = 
            left_child->get_realm_index_space(lhs_spaces[subspace_index],
                                              false/*tight*/);
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (left_ready.exists())
            preconditions.insert(left_ready);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_INTERSECTIONS);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_intersections(
            lhs_spaces, rhs_spaces, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    handle, precondition, result);
#endif
      // Now set the index spaces for the results
      subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      else
      {
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_intersection(Operation *op,
                                                      IndexPartNode *partition,
                                                      // Left is implicit "this"
                                                      IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      const size_t count = partition->color_space->get_volume();
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces(count);
      std::set<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      unsigned subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      else
      {
        for (LegionColor color = 0;
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_INTERSECTIONS);
      Realm::IndexSpace<DIM,T> lhs_space;
      ApEvent left_ready = get_realm_index_space(lhs_space, false/*tight*/);
      if (left_ready.exists())
        preconditions.insert(left_ready);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_intersections(
            lhs_space, rhs_spaces, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    handle, precondition, result);
#endif
      // Now set the index spaces for the results
      subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      else
      {
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_difference(Operation *op,
                                                      IndexPartNode *partition,
                                                      IndexPartNode *left,
                                                      IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      const size_t count = partition->color_space->get_volume();
      std::vector<Realm::IndexSpace<DIM,T> > lhs_spaces(count);
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces(count);
      std::set<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      unsigned subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *left_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(left->get_child(color));
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent left_ready = 
            left_child->get_realm_index_space(lhs_spaces[subspace_index],
                                              false/*tight*/);
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (left_ready.exists())
            preconditions.insert(left_ready);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      else
      {
        for (LegionColor color = 0;
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *left_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
          IndexSpaceNodeT<DIM,T> *right_child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < count);
#endif
          ApEvent left_ready = 
            left_child->get_realm_index_space(lhs_spaces[subspace_index],
                                              false/*tight*/);
          ApEvent right_ready = 
            right_child->get_realm_index_space(rhs_spaces[subspace_index++],
                                               false/*tight*/);
          if (left_ready.exists())
            preconditions.insert(left_ready);
          if (right_ready.exists())
            preconditions.insert(right_ready);
        }
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_DIFFERENCES);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_differences(
            lhs_spaces, rhs_spaces, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    handle, precondition, result);
#endif
      // Now set the index spaces for the results
      subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      else
      {
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!partition->color_space->contains_color(color))
            continue;
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_restriction(
                                                      IndexPartNode *partition,
                                                      const void *tran,
                                                      const void *ext,
                                                      int partition_dim)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // should be called on the color space
      assert(this == partition->color_space); 
#endif
      switch (partition_dim)
      {
        case 1:
          {
            const Realm::Matrix<1,DIM,T> *transform = 
              static_cast<const Realm::Matrix<1,DIM,T>*>(tran);
            const Realm::Rect<1,T> *extent = 
              static_cast<const Realm::Rect<1,T>*>(ext);
            return create_by_restriction_helper<1>(partition, 
                                                   *transform, *extent);
          }
        case 2:
          {
            const Realm::Matrix<2,DIM,T> *transform = 
              static_cast<const Realm::Matrix<2,DIM,T>*>(tran);
            const Realm::Rect<2,T> *extent = 
              static_cast<const Realm::Rect<2,T>*>(ext);
            return create_by_restriction_helper<2>(partition, 
                                                   *transform, *extent);
          }
        case 3:
          {
            const Realm::Matrix<3,DIM,T> *transform = 
              static_cast<const Realm::Matrix<3,DIM,T>*>(tran);
            const Realm::Rect<3,T> *extent = 
              static_cast<const Realm::Rect<3,T>*>(ext);
            return create_by_restriction_helper<3>(partition, 
                                                   *transform, *extent);
          }
        default:
          assert(false);
      }
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    template<int N, typename T> template<int M>
    ApEvent IndexSpaceNodeT<N,T>::create_by_restriction_helper(
                                        IndexPartNode *partition,
                                        const Realm::Matrix<M,N,T> &transform,
                                        const Realm::Rect<M,T> &extent)
    //--------------------------------------------------------------------------
    {
      // Get the parent index space in case it has a sparsity map
      IndexSpaceNodeT<M,T> *parent = 
                      static_cast<IndexSpaceNodeT<M,T>*>(partition->parent);
      // No need to wait since we'll just be messing with the bounds
      Realm::IndexSpace<M,T> parent_is;
      parent->get_realm_index_space(parent_is, true/*tight*/);
      Realm::IndexSpace<N,T> local_is;
      get_realm_index_space(local_is, true/*tight*/);
      // Iterate over our points (colors) and fill in the bounds
      for (Realm::IndexSpaceIterator<N,T> rect_itr(local_is); 
            rect_itr.valid; rect_itr.step())
      {
        for (Realm::PointInRectIterator<N,T> color_itr(rect_itr.rect); 
              color_itr.valid; color_itr.step())
        {
          // Copy the index space from the parent
          Realm::IndexSpace<M,T> child_is = parent_is;
          // Compute the new bounds and intersect it with the parent bounds
          child_is.bounds = parent_is.bounds.intersection(
                              extent + transform * color_itr.p);
          // Get the legion color
          LegionColor color = linearize_color(&color_itr.p, 
                                              handle.get_type_tag());
          // Get the appropriate child
          IndexSpaceNodeT<M,T> *child = 
            static_cast<IndexSpaceNodeT<M,T>*>(partition->get_child(color));
          // Then set the new index space
          child->set_realm_index_space(context->runtime->address_space, 
                                       child_is);
        }
      }
      // Our only precondition is that the parent index space is computed
      return parent->index_space_ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_field(Operation *op,
                                                    IndexPartNode *partition,
                              const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the color space type to do the actual operations 
      CreateByFieldHelper creator(this, op, partition, 
                                  instances, instances_ready);
      NT_TemplateHelper::demux<CreateByFieldHelper>(
                   partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T> template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_field_helper(Operation *op,
                                                      IndexPartNode *partition,
                             const std::vector<FieldDataDescriptor> &instances,
                                                       ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNodeT<COLOR_DIM,COLOR_T> *color_space = 
       static_cast<IndexSpaceNodeT<COLOR_DIM,COLOR_T>*>(partition->color_space);
      // Enumerate the color space
      Realm::IndexSpace<COLOR_DIM,COLOR_T> realm_color_space;
      color_space->get_realm_index_space(realm_color_space, true/*tight*/);
      const size_t num_colors = realm_color_space.volume();
      std::vector<Realm::Point<COLOR_DIM,COLOR_T> > colors(num_colors);
      unsigned index = 0;
      for (Realm::IndexSpaceIterator<COLOR_DIM,COLOR_T> 
            rect_iter(realm_color_space); rect_iter.valid; rect_iter.step())
      {
        for (Realm::PointInRectIterator<COLOR_DIM,COLOR_T> 
              itr(rect_iter.rect); itr.valid; itr.step())
        {
#ifdef DEBUG_LEGION
          assert(index < colors.size());
#endif
          colors[index++] = itr.p;
        }
      }
      // Translate the instances to realm field data descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM,T>,
                Realm::Point<COLOR_DIM,COLOR_T> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      std::set<ApEvent> preconditions; 
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM,T> *node = static_cast<IndexSpaceNodeT<DIM,T>*>(
                                          context->get_node(src.index_space));
        ApEvent ready = node->get_realm_index_space(dst.index_space, 
                                                    false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_FIELD);
      // Perform the operation
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (ready.exists())
        preconditions.insert(ready);
      preconditions.insert(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(local_space.create_subspaces_by_field(
            descriptors, colors, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),handle,
                                    precondition, result);
#endif
      // Update the children with the names of their subspaces 
      for (unsigned idx = 0; idx < colors.size(); idx++)
      {
        LegionColor child_color = color_space->linearize_color(&colors[idx],
                                        color_space->handle.get_type_tag());
        IndexSpaceNodeT<DIM,T> *child = static_cast<IndexSpaceNodeT<DIM,T>*>(
                                            partition->get_child(child_color));
        child->set_realm_index_space(context->runtime->address_space,
                                     subspaces[idx]);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_image(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByImageHelper creator(this, op, partition, projection,
                                  instances, instances_ready);
      NT_TemplateHelper::demux<CreateByImageHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1> template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1,T1>::create_by_image_helper(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Get the index spaces of the projection partition
      std::vector<Realm::IndexSpace<DIM2,T2> > 
                                sources(projection->color_space->get_volume());
      std::set<ApEvent> preconditions; 
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          ApEvent ready = child->get_realm_index_space(sources[color],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      else
      {
        unsigned index = 0;
        // Always use the partitions color space
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < sources.size());
#endif
          ApEvent ready = child->get_realm_index_space(sources[index++],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM2,T2>,
                                       Realm::Point<DIM1,T1> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM2,T2> *node = static_cast<IndexSpaceNodeT<DIM2,T2>*>(
                                          context->get_node(src.index_space));
        ApEvent ready = node->get_realm_index_space(dst.index_space,
                                                    false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_IMAGE);
      // Perform the operation
      std::vector<Realm::IndexSpace<DIM1,T1> > subspaces;
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (ready.exists())
        preconditions.insert(ready);
      preconditions.insert(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(local_space.create_subspaces_by_image(descriptors,
            sources, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),handle,
                                    precondition, result);
#endif
      // Update the child subspaces of the image
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[color]);
        }
      }
      else
      {
        unsigned index = 0;
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_image_range(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByImageRangeHelper creator(this, op, partition, projection,
                                       instances, instances_ready);
      NT_TemplateHelper::demux<CreateByImageRangeHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1> template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1,T1>::create_by_image_range_helper(
                                                    Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Get the index spaces of the projection partition
      std::vector<Realm::IndexSpace<DIM2,T2> > 
                                sources(projection->color_space->get_volume());
      std::set<ApEvent> preconditions;
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          ApEvent ready = child->get_realm_index_space(sources[color],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      else
      {
        unsigned index = 0;
        // Always use the partitions color space
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < sources.size());
#endif
          ApEvent ready = child->get_realm_index_space(sources[index++],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM2,T2>,
                                       Realm::Rect<DIM1,T1> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM2,T2> *node = static_cast<IndexSpaceNodeT<DIM2,T2>*>(
                                          context->get_node(src.index_space));
        ApEvent ready = node->get_realm_index_space(dst.index_space,
                                                    false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_IMAGE_RANGE);
      // Perform the operation
      std::vector<Realm::IndexSpace<DIM1,T1> > subspaces;
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (ready.exists())
        preconditions.insert(ready);
      preconditions.insert(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(local_space.create_subspaces_by_image(descriptors,
            sources, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),handle,
                                    precondition, result);
#endif
      // Update the child subspaces of the image
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[color]);
        }
      }
      else
      {
        unsigned index = 0;
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_preimage(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByPreimageHelper creator(this, op, partition, projection,
                                     instances, instances_ready);
      NT_TemplateHelper::demux<CreateByPreimageHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1> template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1,T1>::create_by_preimage_helper(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Get the index spaces of the projection partition
      std::vector<Realm::IndexSpace<DIM2,T2> > 
                                targets(projection->color_space->get_volume());
      std::set<ApEvent> preconditions;
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          ApEvent ready = child->get_realm_index_space(targets[color],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      else
      {
        unsigned index = 0;
        // Always use the partitions color space
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < targets.size());
#endif
          ApEvent ready = child->get_realm_index_space(targets[index++],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM1,T1>,
                                       Realm::Point<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM1,T1> *node = static_cast<IndexSpaceNodeT<DIM1,T1>*>(
                                          context->get_node(src.index_space));
        ApEvent ready = node->get_realm_index_space(dst.index_space,
                                                    false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_PREIMAGE);
      // Perform the operation
      std::vector<Realm::IndexSpace<DIM1,T1> > subspaces;
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (ready.exists())
        preconditions.insert(ready);
      preconditions.insert(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(local_space.create_subspaces_by_preimage(
            descriptors, targets, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),handle,
                                    precondition, result);
#endif
      // Update the child subspace of the preimage
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[color]);
        }
      }
      else
      {
        unsigned index = 0;
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_preimage_range(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByPreimageRangeHelper creator(this, op, partition, projection,
                                          instances, instances_ready);
      NT_TemplateHelper::demux<CreateByPreimageRangeHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1> template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1,T1>::create_by_preimage_range_helper(
                                                    Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Get the index spaces of the projection partition
      std::vector<Realm::IndexSpace<DIM2,T2> > 
                                targets(projection->color_space->get_volume());
      std::set<ApEvent> preconditions;
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          ApEvent ready = child->get_realm_index_space(targets[color],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      else
      {
        unsigned index = 0;
        // Always use the partitions color space
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < targets.size());
#endif
          ApEvent ready = child->get_realm_index_space(targets[index++],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM1,T1>,
                                       Realm::Rect<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM1,T1> *node = static_cast<IndexSpaceNodeT<DIM1,T1>*>(
                                          context->get_node(src.index_space));
        ApEvent ready = node->get_realm_index_space(dst.index_space,
                                                    false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_PREIMAGE_RANGE);
      // Perform the operation
      std::vector<Realm::IndexSpace<DIM1,T1> > subspaces;
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (ready.exists())
        preconditions.insert(ready);
      preconditions.insert(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(local_space.create_subspaces_by_preimage(
            descriptors, targets, subspaces, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),handle,
                                    precondition, result);
#endif
      // Update the child subspace of the preimage
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[color]);
        }
      }
      else
      {
        unsigned index = 0;
        for (LegionColor color = 0; 
              color < partition->max_linearized_color; color++)
        {
          if (!projection->color_space->contains_color(color))
            continue;
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          child->set_realm_index_space(context->runtime->address_space,
                                       subspaces[index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_association(Operation *op,
                                                       IndexSpaceNode *range,
                              const std::vector<FieldDataDescriptor> &instances,
                                                       ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Demux the range type to do the actual operation
      CreateAssociationHelper creator(this, op, range, 
                                      instances, instances_ready);
      NT_TemplateHelper::demux<CreateAssociationHelper>(
          range->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1> template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1,T1>::create_association_helper(Operation *op,
                                                      IndexSpaceNode *range,
                              const std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM1,T1>,
                                       Realm::Point<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      std::set<ApEvent> preconditions;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM1,T1> *node = static_cast<IndexSpaceNodeT<DIM1,T1>*>(
                                          context->get_node(src.index_space));
        ApEvent ready = node->get_realm_index_space(dst.index_space,
                                                    false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
      }
      // Get the range index space
      IndexSpaceNodeT<DIM2,T2> *range_node = 
        static_cast<IndexSpaceNodeT<DIM2,T2>*>(range);
      Realm::IndexSpace<DIM2,T2> range_space;
      ApEvent range_ready = range_node->get_realm_index_space(range_space,
                                                              false/*tight*/);
      if (range_ready.exists())
        preconditions.insert(range_ready);
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_ASSOCIATION);
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent local_ready = get_realm_index_space(local_space, false/*tight*/);
      if (local_ready.exists())
        preconditions.insert(local_ready);
      preconditions.insert(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      // Issue the operation
      ApEvent precondition = Runtime::merge_events(preconditions);
      ApEvent result(local_space.create_association(descriptors,
            range_space, requests, precondition));
#ifdef LEGION_SPY
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
      LegionSpy::log_deppart_events(op->get_unique_op_id(),handle,
                                    precondition, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::check_field_size(size_t field_size, bool range)
    //--------------------------------------------------------------------------
    {
      if (range)
        return (sizeof(Realm::Rect<DIM,T>) == field_size);
      else
        return (sizeof(Realm::Point<DIM,T>) == field_size);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::issue_copy(Operation *op,
#ifdef LEGION_SPY
                        const std::vector<Realm::CopySrcDstField> &src_fields,
                        const std::vector<Realm::CopySrcDstField> &dst_fields,
#else
                        const std::vector<CopySrcDstField> &src_fields,
                        const std::vector<CopySrcDstField> &dst_fields,
#endif
                        ApEvent precondition, PredEvent predicate_guard,
                        PhysicalTraceInfo &trace_info,
                        IndexTreeNode *intersect/*=NULL*/,
                        ReductionOpID redop /*=0*/,bool reduction_fold/*=true*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_ISSUE_COPY_CALL);
      Realm::ProfilingRequestSet requests;
      if (op != NULL)
        op->add_copy_profiling_request(requests); 
      if ((op != NULL) && op->has_execution_fence_event())
      {
        ApEvent old_precondition = precondition;
        precondition = Runtime::merge_events(precondition,
                        op->get_execution_fence_event());
        if (trace_info.recording)
        {
#ifdef DEBUG_LEGION
          assert(trace_info.tpl != NULL && trace_info.tpl->is_recording());
#endif
          trace_info.tpl->record_merge_events(precondition, old_precondition,
              op->get_execution_fence_event(), trace_info.op);
        }
      }
      ApEvent result;
      if ((intersect == NULL) || (intersect == this))
      {
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_copy_request(requests, op);
        // Include our event precondition if necessary
        if (index_space_ready.exists())
        {
          ApEvent old_precondition = precondition;
          precondition = Runtime::merge_events(precondition, index_space_ready);
          if (trace_info.recording)
            trace_info.tpl->record_merge_events(precondition, old_precondition,
                index_space_ready, trace_info.op);
        }
        Realm::IndexSpace<DIM,T> local_space;
        get_realm_index_space(local_space, true/*tight*/);
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
          if (trace_info.recording)
            trace_info.tpl->record_merge_events(pred_pre, precondition,
                ApEvent(predicate_guard), trace_info.op);
          result = Runtime::ignorefaults(local_space.copy(src_fields, 
                dst_fields, requests, pred_pre, redop, reduction_fold));
        }
        else
          result = ApEvent(local_space.copy(src_fields, dst_fields,
                        requests, precondition, redop, reduction_fold));
      }
      else
      {
        // This is a copy between the intersection of two nodes
        Realm::IndexSpace<DIM,T> intersection;
        if (intersect->is_index_space_node())
        {
          IndexSpaceNode *intersect_node = intersect->as_index_space_node();
          if (intersects_with(intersect_node))
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
              finder = intersections.find(intersect_node);
#ifdef DEBUG_LEGION
            assert(finder != intersections.end());
#endif
            intersection = finder->second.intersection;
          }
          else
          {
#ifdef LEGION_SPY
            ApUserEvent new_result = Runtime::create_ap_user_event();
            Runtime::trigger_event(new_result);
            return new_result;
#else
            return ApEvent::NO_AP_EVENT;
#endif
          }
        }
        else
        {
          IndexPartNode *intersect_node = intersect->as_index_part_node();
          if (intersects_with(intersect_node))
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
              finder = intersections.find(intersect_node);
#ifdef DEBUG_LEGION
            assert(finder != intersections.end());
#endif
            intersection = finder->second.intersection;
          }
          else
          {
#ifdef LEGION_SPY
            ApUserEvent new_result = Runtime::create_ap_user_event();
            Runtime::trigger_event(new_result);
            return new_result;
#else
            return ApEvent::NO_AP_EVENT;
#endif
          }
        }
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_copy_request(requests, op);
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
          if (trace_info.recording)
            trace_info.tpl->record_merge_events(pred_pre, precondition,
                ApEvent(predicate_guard), trace_info.op);
          result = Runtime::ignorefaults(intersection.copy(src_fields, 
                dst_fields, requests, pred_pre, redop, reduction_fold));
        }
        else
          result = ApEvent(intersection.copy(src_fields, dst_fields,
                        requests, precondition, redop, reduction_fold));
      }
#ifdef LEGION_SPY
      if (!result.exists())
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::issue_fill(Operation *op,
#ifdef LEGION_SPY
                        const std::vector<Realm::CopySrcDstField> &dst_fields,
#else
                        const std::vector<CopySrcDstField> &dst_fields,
#endif
                        const void *fill_value, size_t fill_size,
                        ApEvent precondition, PredEvent predicate_guard,
                        PhysicalTraceInfo &trace_info,
                        IndexTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_ISSUE_FILL_CALL);
      Realm::ProfilingRequestSet requests;
      if (op != NULL)
        op->add_copy_profiling_request(requests); 
      if ((op != NULL) && op->has_execution_fence_event())
      {
        ApEvent old_precondition = precondition;
        precondition = Runtime::merge_events(precondition,
                        op->get_execution_fence_event());
        if (trace_info.recording)
        {
#ifdef DEBUG_LEGION
          assert(trace_info.tpl != NULL && trace_info.tpl->is_recording());
#endif
          trace_info.tpl->record_merge_events(precondition, old_precondition,
              op->get_execution_fence_event(), trace_info.op);
        }
      }
      ApEvent result;
      if ((intersect == NULL) || (intersect == this))
      {
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_fill_request(requests, op);
        // Include our event precondition if necessary
        if (index_space_ready.exists())
        {
          ApEvent old_precondition = precondition;
          precondition = Runtime::merge_events(precondition, index_space_ready);
          if (trace_info.recording)
            trace_info.tpl->record_merge_events(precondition, old_precondition,
                index_space_ready, trace_info.op);
        }
        Realm::IndexSpace<DIM,T> local_space;
        get_realm_index_space(local_space, true/*tight*/);
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
          if (trace_info.recording)
            trace_info.tpl->record_merge_events(pred_pre, precondition,
                ApEvent(predicate_guard), trace_info.op);
          result = Runtime::ignorefaults(local_space.fill(dst_fields, 
                requests, fill_value, fill_size, pred_pre));
        }
        else
          result = ApEvent(local_space.fill(dst_fields, requests, 
                fill_value, fill_size, precondition));
      }
      else
      {
        // This is a copy between the intersection of two nodes
        Realm::IndexSpace<DIM,T> intersection;
        if (intersect->is_index_space_node())
        {
          IndexSpaceNode *intersect_node = intersect->as_index_space_node();
          if (intersects_with(intersect_node))
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
              finder = intersections.find(intersect_node);
#ifdef DEBUG_LEGION
            assert(finder != intersections.end());
#endif
            intersection = finder->second.intersection;
          }
#ifdef DEBUG_LEGION
          else
            assert(false);
#endif
        }
        else
        {
          IndexPartNode *intersect_node = intersect->as_index_part_node();
          if (intersects_with(intersect_node))
          {
            AutoLock n_lock(node_lock,1,false/*exclusive*/);
            typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
              finder = intersections.find(intersect_node);
#ifdef DEBUG_LEGION
            assert(finder != intersections.end());
#endif
            intersection = finder->second.intersection;
          }
#ifdef DEBUG_LEGION
          else
            assert(false);
#endif
        }
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_fill_request(requests, op);
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
          if (trace_info.recording)
            trace_info.tpl->record_merge_events(pred_pre, precondition,
                ApEvent(predicate_guard), trace_info.op);
          result = Runtime::ignorefaults(intersection.fill(dst_fields, 
                requests, fill_value, fill_size, pred_pre));
        }
        else
          result = ApEvent(intersection.fill(dst_fields, requests, 
                fill_value, fill_size, precondition));
      }
#ifdef LEGION_SPY
      if (!result.exists())
      {
        ApUserEvent new_result = Runtime::create_ap_user_event();
        Runtime::trigger_event(new_result);
        result = new_result;
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceNodeT<DIM,T>::create_layout(
                                    const Realm::InstanceLayoutConstraints &ilc,
                                    const OrderingConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_is;
      get_realm_index_space(local_is, true/*tight*/);
      int dim_order[DIM];
      // Construct the dimension ordering
      unsigned next_dim = 0;
      for (std::vector<DimensionKind>::const_iterator it = 
            constraint.ordering.begin(); it != constraint.ordering.end(); it++)
      {
        // Skip the field dimension we already handled it
        if ((*it) == DIM_F)
          continue;
        if ((*it) > DIM_F)
          assert(false); // TODO: handle split dimensions
        if ((*it) >= DIM) // Skip dimensions bigger than ours
          continue;
        dim_order[next_dim++] = *it;
      }
#ifdef DEBUG_LEGION
      assert(next_dim == DIM); // should have filled them all in
#endif
      return Realm::InstanceLayoutGeneric::choose_instance_layout(local_is,
                                                            ilc, dim_order);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_file_instance(
                                         const char *file_name,
                                         const std::vector<Realm::FieldID> &field_ids,
                                         const std::vector<size_t> &field_sizes,
                                         legion_file_mode_t file_mode,
                                         ApEvent &ready_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_CREATE_INSTANCE_CALL);
      // Have to wait for the index space to be ready if necessary
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      // No profiling for these kinds of instances currently
      Realm::ProfilingRequestSet requests;
      PhysicalInstance result;
      ready_event = ApEvent(PhysicalInstance::create_file_instance(result, 
          file_name, local_space, field_ids, field_sizes, file_mode, requests));
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_hdf5_instance(
                                    const char *file_name,
				    const std::vector<Realm::FieldID> &field_ids,
                                    const std::vector<size_t> &field_sizes,
                                    const std::vector<const char*> &field_files,
                                    bool read_only, ApEvent &ready_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_CREATE_INSTANCE_CALL);
      // Have to wait for the index space to be ready if necessary
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      // No profiling for these kinds of instances currently
      Realm::ProfilingRequestSet requests;
      PhysicalInstance result;
#ifdef USE_HDF
      ready_event = ApEvent(PhysicalInstance::create_hdf5_instance(result, 
                            file_name, local_space, field_ids, field_sizes,
		            field_files, read_only, requests));
#else
      assert(0 && "no HDF5 support");
      result = PhysicalInstance::NO_INST;
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_external_instance(
                                          Memory memory, uintptr_t base,
                                          Realm::InstanceLayoutGeneric *ilg,
                                          ApEvent &ready_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_CREATE_INSTANCE_CALL);
      // Have to wait for the index space to be ready if necessary
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      // No profiling for these kinds of instances currently
      Realm::ProfilingRequestSet requests;
      PhysicalInstance result;
      ready_event = ApEvent(PhysicalInstance::create_external(result,
                                        memory, base, ilg, requests));
      return result;
    }
    
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::get_launch_space_domain(Domain &launch_domain)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      launch_domain = local_space;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::validate_slicing(
                                  const std::vector<IndexSpace> &slice_spaces, 
                                  MultiTask *task, MapperManager *mapper)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpaceNodeT<DIM,T>*> slice_nodes(slice_spaces.size());
      for (unsigned idx = 0; idx < slice_spaces.size(); idx++)
      {
#ifdef DEBUG_LEGION
        assert(slice_spaces[idx].get_type_tag() == handle.get_type_tag());
#endif
        slice_nodes[idx] = static_cast<IndexSpaceNodeT<DIM,T>*>(
                            context->get_node(slice_spaces[idx]));
      }
      // Iterate over the points and make sure that they exist in exactly
      // one slice space, no more, no less
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      for (PointInDomainIterator<DIM,T> itr(local_space); itr(); itr++)
      {
        bool found = false;
        const Realm::Point<DIM,T> &point = *itr;
        for (unsigned idx = 0; idx < slice_nodes.size(); idx++)
        {
          if (!slice_nodes[idx]->contains_point(point))
            continue;
          if (found)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                    "Invalid mapper output from invocation of 'slice_task' "
                    "on mapper %s. Mapper returned multilple slices that "
                    "contained the same point for task %s (ID %lld)",
                    mapper->get_mapper_name(), task->get_task_name(),
                    task->get_unique_id())
          else
            found = true;
        }
        if (!found)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                    "Invalid mapper output from invocation of 'slice_task' "
                    "on mapper %s. Mapper returned no slices that "
                    "contained some point(s) for task %s (ID %lld)",
                    mapper->get_mapper_name(), task->get_task_name(),
                    task->get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_launch_space(UniqueID op_id)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      for (Realm::IndexSpaceIterator<DIM,T> itr(local_space); 
            itr.valid; itr.step())
        LegionSpy::log_launch_index_space_rect<DIM>(op_id, 
                                                    Rect<DIM,T>(itr.rect));
    }

    /////////////////////////////////////////////////////////////
    // Templated Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(RegionTreeForest *ctx, 
                                        IndexPartition p,
                                        IndexSpaceNode *par, IndexSpaceNode *cs,
                                        LegionColor c, bool disjoint, 
                                        DistributedID did,
                                        ApEvent partition_ready, 
                                        ApUserEvent pend)
      : IndexPartNode(ctx, p, par, cs, c, disjoint, did, partition_ready, pend),
        has_union_space(false), union_space_tight(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(RegionTreeForest *ctx, 
                                        IndexPartition p,
                                        IndexSpaceNode *par, IndexSpaceNode *cs,
                                        LegionColor c, RtEvent disjoint_event,
                                        DistributedID did,
                                        ApEvent partition_ready, 
                                        ApUserEvent pending)
      : IndexPartNode(ctx, p, par, cs, c, disjoint_event, did, 
                      partition_ready, pending),
        has_union_space(false), union_space_tight(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(const IndexPartNodeT &rhs)
      : IndexPartNode(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::~IndexPartNodeT(void)
    //--------------------------------------------------------------------------
    { 
      if (has_union_space && !partition_union_space.empty())
        partition_union_space.destroy();
      for (typename std::map<IndexTreeNode*,IntersectInfo>::iterator it = 
            intersections.begin(); it != intersections.end(); it++)
        if (it->second.has_intersection && it->second.intersection_valid)
          it->second.intersection.destroy();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>& IndexPartNodeT<DIM,T>::operator=(
                                                      const IndexPartNodeT &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::compute_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_disjoint());
#endif
      Realm::IndexSpace<DIM,T> parent_space, union_space, difference_space;
      get_union_index_space(union_space, true/*tight*/);
      ApEvent parent_ready = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(parent)->get_realm_index_space(
                                                parent_space, false/*tight*/);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                      (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
      ApEvent diff_ready(Realm::IndexSpace<DIM,T>::compute_difference(
          parent_space, union_space, difference_space, requests, parent_ready));
      if (!diff_ready.has_triggered())
        diff_ready.wait();
      // Always tighten these tests so that they are precise
      Realm::IndexSpace<DIM,T> tight_space = difference_space.tighten();
      bool complete = tight_space.empty();
      difference_space.destroy();
      tight_space.destroy();
      return complete;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::intersects_with(IndexSpaceNode *rhs, 
                                                bool compute)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
          finder = intersections.find(rhs);
        if ((finder != intersections.end()) &&
            (!compute || finder->second.intersection_valid))
          return finder->second.has_intersection;
      }
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      if (!compute)
      {
        // Before we do something expensive, let's do an easy test
        // just by walking the region tree
        IndexSpaceNode *temp = rhs;
        while ((temp->parent != NULL) && (temp->parent->depth >= depth))
        {
          if (temp->parent == this)
          {
            AutoLock n_lock(node_lock);
            intersections[rhs] = IntersectInfo(true/*result*/);
            return true;
          }
          temp = temp->parent->parent;
        }
        // Otherwise fall through and do the expensive test
      }
      Realm::IndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent union_precondition = get_union_index_space(lhs_space, false);
      ApEvent rhs_ready = rhs_node->get_realm_index_space(rhs_space, 
                                                          false/*tight*/);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      ApEvent ready(Realm::IndexSpace<DIM,T>::compute_intersection(
            lhs_space, rhs_space, intersection, requests,
            Runtime::merge_events(union_precondition, rhs_ready)));
      if (!ready.has_triggered())
        ready.wait();
      // Always tighten these tests so that they are precise
      Realm::IndexSpace<DIM,T> tight_intersection = intersection.tighten();
      bool result = !tight_intersection.empty();
      AutoLock n_lock(node_lock);
      if (result)
      {
        // Check to make sure we didn't lose the race
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder =
          intersections.find(rhs);
        if ((finder == intersections.end()) ||
            (compute && !finder->second.intersection_valid))
          intersections[rhs] = IntersectInfo(tight_intersection);
        else
          tight_intersection.destroy();
      }
      else
        intersections[rhs] = IntersectInfo(false/*result*/);
      intersection.destroy();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::intersects_with(IndexPartNode *rhs,bool compute)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      if (rhs == this)
        return true;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator 
          finder = intersections.find(rhs);
        // Only return the value if we know we are valid and we didn't
        // want to compute anything or we already did compute it
        if ((finder != intersections.end()) &&
            (!compute || finder->second.intersection_valid))
          return finder->second.has_intersection;
      }
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      if (!compute)
      {
        // Before we do an expensive test, let's do an easy test
        // just by walking the region tree
        IndexPartNode *temp = rhs;
        while ((temp != NULL) && (temp->depth >= depth))
        {
          if (temp == this)
          {
            AutoLock n_lock(node_lock);
            intersections[rhs] = IntersectInfo(true/*result*/);
            return true;
          }
          temp = temp->parent->parent;
        }
      }
      Realm::IndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent union_precondition = get_union_index_space(lhs_space, 
                                                         false/*tight*/);
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space, 
                                                         false/*tight*/);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      ApEvent ready(Realm::IndexSpace<DIM,T>::compute_intersection(
            lhs_space, rhs_space, intersection, requests,
            Runtime::merge_events(union_precondition, rhs_precondition)));
      if (!ready.has_triggered())
        ready.wait();
      // Always tighten these tests so that they are precise
      Realm::IndexSpace<DIM,T> tight_intersection = intersection.tighten();
      bool result = !tight_intersection.empty();
      AutoLock n_lock(node_lock);
      if (result)
      {
        // Check to make sure we didn't lose the race
        typename std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder =
          intersections.find(rhs);
        if ((finder == intersections.end()) ||
            (compute && !finder->second.intersection_valid))
          intersections[rhs] = IntersectInfo(tight_intersection);
        else
          tight_intersection.destroy();
      }
      else
        intersections[rhs] = IntersectInfo(false/*result*/);
      intersection.destroy();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::dominates(IndexSpaceNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(rhs);
        if (finder != dominators.end())
          return finder->second;
      }
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      // Before we do something expensive, let's do an easy test
      // just by walking the region tree
      IndexSpaceNode *temp = rhs;
      while ((temp->parent != NULL) && (temp->parent->depth >= depth))
      {
        if (temp->parent == this)
        {
          AutoLock n_lock(node_lock);
          dominators[rhs] = true;
          return true;
        }
        temp = temp->parent->parent;
      }
      // Otherwise fall through and do the expensive test
      Realm::IndexSpace<DIM,T> union_space, rhs_space, difference;
      get_union_index_space(union_space, true/*tight*/);
      bool result = false;
      if (!union_space.dense())
      {
        ApEvent rhs_ready = rhs_node->get_realm_index_space(rhs_space,
                                                            false/*tight*/);
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                          (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::IndexSpace<DIM,T>::compute_difference(
              rhs_space, union_space, difference, requests, rhs_ready));
        if (!ready.has_triggered())
          ready.wait();
        // Always tighten these tests so that they are precise
        Realm::IndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      }
      else // Fast path
      {
        rhs_node->get_realm_index_space(rhs_space, true/*tight*/);
        result = union_space.bounds.contains(rhs_space);
      }
      AutoLock n_lock(node_lock);
      dominators[rhs] = result;
      return result;
    }
    
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::dominates(IndexPartNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      if (rhs == this)
        return true;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(rhs);
        if (finder != dominators.end())
          return finder->second;
      }
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      // Before we do an expensive test, let's do an easy test
      // just by walking the region tree
      IndexPartNode *temp = rhs;
      while ((temp != NULL) && (temp->depth >= depth))
      {
        if (temp == this)
        {
          AutoLock n_lock(node_lock);
          dominators[rhs] = true;
          return true;
        }
        temp = temp->parent->parent;
      }
      // Otherwise we fall through and do the expensive test
      Realm::IndexSpace<DIM,T> union_space, rhs_space, difference;
      get_union_index_space(union_space, true/*tight*/);
      bool result = false;
      if (!union_space.dense())
      {
        ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space,
                                                              false/*tight*/);
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                        (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::IndexSpace<DIM,T>::compute_difference(
              rhs_space, union_space, difference, requests, rhs_precondition));
        if (!ready.has_triggered())
          ready.wait();
        // Always tighten these tests so that they are precise
        Realm::IndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      } 
      else // Fast path
      {
        rhs_node->get_union_index_space(rhs_space, true/*tight*/);
        result = union_space.bounds.contains(rhs_space);
      }
      AutoLock n_lock(node_lock);
      dominators[rhs] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexPartNodeT<DIM,T>::get_union_index_space(
                      Realm::IndexSpace<DIM,T> &result, bool need_tight_result)
    //--------------------------------------------------------------------------
    {
      if (!union_space_tight)
      {
        if (!has_union_space)
        {
          // Compute it and then check to see if we lost the race
          std::set<ApEvent> preconditions;
          std::vector<Realm::IndexSpace<DIM,T> > subspaces(
                                                    color_space->get_volume());
          unsigned subspace_index = 0;
          if (total_children == max_linearized_color)
          {
            for (LegionColor color = 0; color < total_children; color++)
            {
              IndexSpaceNodeT<DIM,T> *child = 
                static_cast<IndexSpaceNodeT<DIM,T>*>(get_child(color));
              ApEvent ready = 
                child->get_realm_index_space(subspaces[subspace_index++],
                                             false/*tight*/);
              if (ready.exists())
                preconditions.insert(ready);
            }
          }
          else
          {
            for (LegionColor color = 0; color < total_children; color++)
            {
              if (!color_space->contains_color(color))
                continue;
              IndexSpaceNodeT<DIM,T> *child = 
                static_cast<IndexSpaceNodeT<DIM,T>*>(get_child(color));
              ApEvent ready =
                child->get_realm_index_space(subspaces[subspace_index++],
                                             false/*tight*/);
              if (ready.exists())
                preconditions.insert(ready);
            }
          }
          Realm::ProfilingRequestSet requests;
          if (context->runtime->profiler != NULL)
            context->runtime->profiler->add_partition_request(requests,
                      (Operation*)NULL/*op*/, DEP_PART_UNION_REDUCTION);
          Realm::IndexSpace<DIM,T> union_space;
          ApEvent union_ready(Realm::IndexSpace<DIM,T>::compute_union(
                subspaces, union_space, requests, 
                Runtime::merge_events(preconditions)));
          bool delete_union_space = false;
          {
            AutoLock n_lock(node_lock);
            if (!has_union_space)
            {
              // Won the race
              partition_union_space = union_space;
              partition_union_ready = union_ready;
              __sync_synchronize();
              has_union_space = true;
              result = partition_union_space;
              // If we don't need it tight, we are done
              if (!need_tight_result)
                return partition_union_ready;
            }
            else
            {
              // Lost the race
              result = partition_union_space;
              delete_union_space = true;
            }
          }
          if (delete_union_space)
          {
            if (!union_ready.has_triggered())
              union_ready.wait();
            union_space.destroy(); 
          }
          if (!need_tight_result)
            return partition_union_ready;
        }
        else
        {
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          result = partition_union_space;
          if (union_space_tight) // was since tightened
            return ApEvent::NO_AP_EVENT;
          else if (!need_tight_result)
            return partition_union_ready;
        }
        // If we make it here we need to tighten our result
        if (!partition_union_ready.has_triggered())
          partition_union_ready.wait();
        Realm::IndexSpace<DIM,T> tight_space = result.tighten();
        // Retake the lock and see if we were the first to tighten
        Realm::IndexSpace<DIM,T> to_destroy;
        {
          AutoLock n_lock(node_lock);
          if (!union_space_tight)
          {
            // Won the race 
            to_destroy = partition_union_space;
            partition_union_space = tight_space;
            __sync_synchronize();
            union_space_tight = true;
          }
          else // Lost the race
            to_destroy = tight_space;
        }
        to_destroy.destroy();
      }
      // Once we get here we can just read it
      result = partition_union_space;
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::destroy_node(AddressSpaceID source) 
    //--------------------------------------------------------------------------
    {
      if (destroyed)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_INDEX_PARTITION_DELETION,
            "Duplicate deletion of Index Partition %d", handle.get_id())
      destroyed = true;
      // If we're not the owner send a message to do the destruction
      // otherwise we can do it here
      if (!is_owner())
      {
        runtime->send_index_partition_destruction(handle, owner_space);
        return false;
      }
      else
        return remove_base_valid_ref(APPLICATION_REF, NULL/*mutator*/);
    }

  }; // namespace Internal
}; // namespace Legion

