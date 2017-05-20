/* Copyright 2017 Stanford University, NVIDIA Corporation
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
        const Realm::ZIndexSpace<DIM,T> *is, ApEvent ready)
      : IndexSpaceNode(ctx, handle, parent, color, ready), 
        offset(0), linearization_ready(false)
    //--------------------------------------------------------------------------
    {
      if (is != NULL)
      {
        realm_index_space = *is;
        Runtime::trigger_event(realm_index_space_set);
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
      // Log subspaces when we are cleaning up
      if (Runtime::legion_spy_enabled && (parent != NULL) &&
          (get_owner_space() == context->runtime->address_space))
      {
        if (!index_space_ready.has_triggered())
          index_space_ready.lg_wait();
        log_index_space_points();
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
    void IndexSpaceNodeT<DIM,T>::get_realm_index_space(
                                        Realm::ZIndexSpace<DIM,T> &result) const
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      result = realm_index_space;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::set_realm_index_space(AddressSpaceID source,
                                         const Realm::ZIndexSpace<DIM,T> &value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!realm_index_space_set.has_triggered());
#endif
      realm_index_space = value;
      Runtime::trigger_event(realm_index_space_set);
      // If we're not the owner, send a message back to the
      // owner specifying that it can set the index space value
      const AddressSpaceID owner_space = get_owner_space();
      if (owner_space != context->runtime->address_space)
      {
        // We're not the owner, if this is not from the owner then
        // send a message there telling the owner that it is set
        if (source != owner_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            pack_index_space(rez);
          }
          context->runtime->send_index_space_set(owner_space, rez);
        }
      }
      else
      {
        // We're the owner, send messages to everyone else that we've 
        // sent this node to except the source
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          pack_index_space(rez);
        }
        IndexSpaceSetFunctor functor(context->runtime, source, rez);
        // Hold the lock while walking over the node set
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        creation_set.map(functor); 
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::initialize_union_space(ApUserEvent to_trigger,
                             TaskOp *op, const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> preconditions;
      std::vector<Realm::ZIndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          log_run.error("Dynamic type mismatch in 'union_index_spaces' "
                        "performed in task %s (UID %lld)",
                        op->get_task_name(), op->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_DYNAMIC_TYPE_MISMATCH);
        }
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        space->get_realm_index_space(spaces[idx]);
        if (!space->index_space_ready.has_triggered())
          preconditions.insert(space->index_space_ready);
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                      op, DEP_PART_UNION_REDUCTION);
      Realm::ZIndexSpace<DIM,T> result_space;
      ApEvent done(Realm::ZIndexSpace<DIM,T>::compute_union(
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
      std::vector<Realm::ZIndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          log_run.error("Dynamic type mismatch in 'intersect_index_spaces' "
                        "performed in task %s (UID %lld)",
                        op->get_task_name(), op->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_DYNAMIC_TYPE_MISMATCH);
        }
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        space->get_realm_index_space(spaces[idx]);
        if (!space->index_space_ready.has_triggered())
          preconditions.insert(space->index_space_ready);
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                      op, DEP_PART_INTERSECTION_REDUCTION);
      Realm::ZIndexSpace<DIM,T> result_space;
      ApEvent done(Realm::ZIndexSpace<DIM,T>::compute_intersection(
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
      {
        log_run.error("Dynamic type mismatch in 'subtract_index_spaces' "
                      "performed in task %s (UID %lld)",
                      op->get_task_name(), op->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DYNAMIC_TYPE_MISMATCH);
      }
      IndexSpaceNodeT<DIM,T> *left_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(context->get_node(left));
      IndexSpaceNodeT<DIM,T> *right_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(context->get_node(right));
      Realm::ZIndexSpace<DIM,T> left_space, right_space;
      left_node->get_realm_index_space(left_space);
      right_node->get_realm_index_space(right_space);
      ApEvent precondition = Runtime::merge_events(left_node->index_space_ready,
                                                 right_node->index_space_ready);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_DIFFERENCE);
      Realm::ZIndexSpace<DIM,T> result_space;
      ApEvent done(Realm::ZIndexSpace<DIM,T>::compute_difference(
           left_space, right_space, result_space, requests, precondition));
      set_realm_index_space(context->runtime->address_space, result_space);
      Runtime::trigger_event(to_trigger, done);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_index_space_points(void) const
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      Realm::ZIndexSpace<DIM,T> tight_space = realm_index_space.tighten();
      if (!tight_space.empty())
      {
        // Iterate over the rectangles and print them out 
        for (Realm::ZIndexSpaceIterator<DIM,T> itr(tight_space); 
              itr.valid; itr.step())
        {
          if (itr.rect.volume() == 1)
            LegionSpy::log_index_space_point(handle.get_id(), itr.rect.lo);
          else
            LegionSpy::log_index_space_rect(handle.get_id(), itr.rect);
        }
      }
      else
        LegionSpy::log_empty_index_space(handle.get_id());
      tight_space.destroy();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::compute_pending_space(Operation *op,
                          const std::vector<IndexSpace> &handles, bool is_union)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> preconditions;
      std::vector<Realm::ZIndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          TaskContext *ctx = op->get_context();
          if (is_union)
            log_run.error("Dynamic type mismatch in 'create_index_space_union' "
                          "performed in task %s (UID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
          else
            log_run.error("Dynamic type mismatch in "
                          "'create_index_space_intersection' performed in "
                          "task %s (UID %lld)", ctx->get_task_name(),
                          ctx->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_DYNAMIC_TYPE_MISMATCH);
        }
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        space->get_realm_index_space(spaces[idx]);
        if (!space->index_space_ready.has_triggered())
          preconditions.insert(space->index_space_ready);
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::ZIndexSpace<DIM,T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_UNION_REDUCTION);
        ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_union(
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
        ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_intersection(
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
          log_run.error("Dynamic type mismatch in 'create_index_space_union' "
                        "performed in task %s (UID %lld)",
                        ctx->get_task_name(), ctx->get_unique_id());
        else
          log_run.error("Dynamic type mismatch in "
                        "'create_index_space_intersection' performed in "
                        "task %s (UID %lld)", ctx->get_task_name(),
                        ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DYNAMIC_TYPE_MISMATCH);
      }
      IndexPartNode *partition = context->get_node(part_handle);
      std::set<ApEvent> preconditions;
      std::vector<Realm::ZIndexSpace<DIM,T> > 
        spaces(partition->color_space->get_volume());
      unsigned subspace_index = 0;
      if (partition->total_children == partition->max_linearized_color)
      {
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
          child->get_realm_index_space(spaces[subspace_index++]);
          if (!child->index_space_ready.has_triggered())
            preconditions.insert(child->index_space_ready);
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
          child->get_realm_index_space(spaces[subspace_index++]);
          if (!child->index_space_ready.has_triggered())
            preconditions.insert(child->index_space_ready);
        }
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(preconditions);
      Realm::ZIndexSpace<DIM,T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_UNION_REDUCTION);
        ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_union(
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
        ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_intersection(
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
        log_run.error("Dynamic type mismatch in "
                      "'create_index_space_difference' performed in "
                      "task %s (%lld)", ctx->get_task_name(), 
                      ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DYNAMIC_TYPE_MISMATCH);
      }
      std::set<ApEvent> preconditions;
      std::vector<Realm::ZIndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          TaskContext *ctx = op->get_context();
          log_run.error("Dynamic type mismatch in "
                        "'create_index_space_difference' performed in "
                        "task %s (%lld)", ctx->get_task_name(), 
                        ctx->get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_DYNAMIC_TYPE_MISMATCH);
        }
        IndexSpaceNodeT<DIM,T> *space = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(node);
        space->get_realm_index_space(spaces[idx]);
        if (!space->index_space_ready.has_triggered())
          preconditions.insert(space->index_space_ready);
      } 
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
      Realm::ZIndexSpace<DIM,T> rhs_space;
      ApEvent rhs_ready(Realm::ZIndexSpace<DIM,T>::compute_union(
            spaces, rhs_space, union_requests, precondition));
      IndexSpaceNodeT<DIM,T> *lhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(context->get_node(init));
      Realm::ZIndexSpace<DIM,T> lhs_space, result_space;
      lhs_node->get_realm_index_space(lhs_space);
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_difference(
            lhs_space, rhs_space, result_space, diff_requests,
            Runtime::merge_events(lhs_node->index_space_ready, rhs_ready)));
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
      {
        log_run.error("Dynamic type mismatch in 'get_index_space_domain'");
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DYNAMIC_TYPE_MISMATCH);
      }
      Realm::ZIndexSpace<DIM,T> *target = 
        static_cast<Realm::ZIndexSpace<DIM,T>*>(realm_is);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      *target = realm_index_space;
      // If the event isn't ready we have to wait 
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::get_volume(void) const
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      return realm_index_space.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_point(const void *realm_point, 
                                                TypeTag type_tag) const
    //--------------------------------------------------------------------------
    {
      if (type_tag != handle.get_type_tag())
      {
        log_run.error("Dynamic type mismatch in 'safe_cast'");
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DYNAMIC_TYPE_MISMATCH);
      }
      const Realm::ZPoint<DIM,T> *point = 
        static_cast<const Realm::ZPoint<DIM,T>*>(realm_point);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      return realm_index_space.contains(*point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceAllocator* IndexSpaceNodeT<DIM,T>::create_allocator(void) const
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      return new IndexSpaceAllocator(Domain(realm_index_space));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // If we've already been destroyed then we are done
      if (destroyed)
        return;
      std::set<RegionNode*> to_destroy;
      {
        AutoLock n_lock(node_lock);
        if (!destroyed)
        {
          destroyed = true;
          if (!creation_set.empty())
          {
            DestructionFunctor functor(handle, context->runtime);
            creation_set.map(functor);
          }
          to_destroy = logical_nodes;
        }
      }
      for (std::set<RegionNode*>::const_iterator it = to_destroy.begin();
            it != to_destroy.end(); it++)
      {
        (*it)->destroy_node(source);
      }
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (get_owner_space() == context->runtime->address_space)
        realm_index_space.destroy();
      for (typename std::map<IndexTreeNode*,IntersectInfo>::iterator it =
            intersections.begin(); it != intersections.end(); it++)
      {
        if (it->second.has_intersection && it->second.intersection_valid)
          it->second.intersection.destroy();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::get_max_linearized_color(void) const
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      return realm_index_space.bounds.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::compute_linearization_metadata(void)
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      // Don't need to wait for full index space since we just need bounds
      const Realm::ZRect<DIM,T> &bounds = realm_index_space.bounds;
      const size_t volume = bounds.volume();
      if (volume > 0)
      {
        size_t local_offset = 0;
        ptrdiff_t stride = 1;
        for (int idx = 0; idx < DIM; idx++)
        {
          local_offset += bounds.lo[idx] * stride;
          strides[idx] = stride;
          stride *= bounds.hi[idx] - bounds.lo[idx] + 1;
        }
        offset = local_offset;
#ifdef DEBUG_LEGION
        assert(stride == (ptrdiff_t)volume);
#endif
      }
      else
      {
        offset = 0;
        for (int idx = 0; idx < DIM; idx++)
          strides[idx] = 0;
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
      const Realm::ZPoint<DIM,T> &point = 
        *(static_cast<const Realm::ZPoint<DIM,T>*>(realm_color));
      LegionColor color = 0;
      for (int idx = 0; idx < DIM; idx++)
        color += point[idx] * strides[idx];
#ifdef DEBUG_LEGION
      assert(color >= offset);
#endif
      return (color - offset); 
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
      Realm::ZPoint<DIM,T> &point = 
        *(static_cast<Realm::ZPoint<DIM,T>*>(realm_color));
      color += offset;
      for (int idx = DIM-1; idx >= 0; idx--)
      {
        point[idx] = color/strides[idx]; // truncates
#ifdef DEBUG_LEGION
        assert(color >= (point[idx] * strides[idx]));
#endif
        color -= point[idx] * strides[idx];
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_color(LegionColor color, 
                                                bool report_error/*=false*/)
    //--------------------------------------------------------------------------
    {
      Realm::ZPoint<DIM,T> point;
      delinearize_color(color, &point, handle.get_type_tag());
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      if (!realm_index_space.contains(point))
      {
        if (report_error)
        {
          log_run.error("Invalid color request");
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_INDEX_SPACE_COLOR);
        }
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
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      colors.resize(get_volume());
      unsigned idx = 0;
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      for (Realm::ZIndexSpaceIterator<DIM,T> rect_itr(realm_index_space); 
            rect_itr.valid; rect_itr.step())
      {
        for (Realm::ZPointInRectIterator<DIM,T> itr(rect_itr.rect);
              itr.valid; itr.step(), idx++)
          colors[idx] = linearize_color(&itr.p, handle.get_type_tag());
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Domain IndexSpaceNodeT<DIM,T>::get_color_space_domain(void) const
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      return Domain(realm_index_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainPoint IndexSpaceNodeT<DIM,T>::get_domain_point_color(void) const
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
        return DomainPoint(color);
      Realm::ZPoint<DIM,coord_t> color_point;
      parent->color_space->delinearize_color(color, &color_point,
          NT_TemplateHelper::encode_tag<DIM,coord_t>());
      return DomainPoint(color_point);
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
      Realm::ZIndexSpace<DIM,T> rhs_space, intersection;
      rhs_node->get_realm_index_space(rhs_space);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
        realm_index_space, rhs_space, intersection, requests,
        Runtime::merge_events(index_space_ready, rhs_node->index_space_ready)));
      // Wait for the result to be ready
      if (!ready.has_triggered())
        ready.lg_wait();
      // Always tighten these tests so that they are precise
      Realm::ZIndexSpace<DIM,T> tight_intersection = intersection.tighten();
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
      Realm::ZIndexSpace<DIM,T> rhs_space, intersection;
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
            realm_index_space, rhs_space, intersection, requests,
            Runtime::merge_events(index_space_ready, rhs_precondition)));
      if (!ready.has_triggered())
        ready.lg_wait();
      // Always tighten these tests so that they are precise
      Realm::ZIndexSpace<DIM,T> tight_intersection = intersection.tighten();
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
      Realm::ZIndexSpace<DIM,T> rhs_space, difference; 
      rhs_node->get_realm_index_space(rhs_space);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      bool result = false;
      if (!realm_index_space.dense() || 
          !rhs_node->index_space_ready.has_triggered())
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                        (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
          rhs_space, realm_index_space, difference, requests,
          Runtime::merge_events(index_space_ready, 
                                rhs_node->index_space_ready)));
        if (!ready.has_triggered())
          ready.lg_wait();
        // Always tighten these tests so that they are precise
        Realm::ZIndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      }
      else // Fast path
        result = realm_index_space.bounds.contains(rhs_space);
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
      Realm::ZIndexSpace<DIM,T> rhs_space, difference;
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      bool result = false;
      if (!realm_index_space.dense() || !rhs_precondition.has_triggered())
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                        (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
              rhs_space, realm_index_space, difference, requests,
              Runtime::merge_events(index_space_ready, rhs_precondition)));
        if (!ready.has_triggered())
          ready.lg_wait();
        // Always tighten these tests so that they are precise
        Realm::ZIndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      }
      else // Fast path
        result = realm_index_space.bounds.contains(rhs_space);
      AutoLock n_lock(node_lock);
      dominators[rhs] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::pack_index_space(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      if (realm_index_space_set.has_triggered())
      {
        rez.serialize<size_t>(sizeof(realm_index_space));
        rez.serialize(realm_index_space);
      }
      else
        rez.serialize<size_t>(0); // not ready yet
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::unpack_index_space(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      size_t size;
      derez.deserialize(size);
      Realm::ZIndexSpace<DIM,T> result_space;
#ifdef DEBUG_LEGION
      assert(size == sizeof(result_space));
#endif
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
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(count);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                                op, DEP_PART_EQUAL);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent result(realm_index_space.create_equal_subspaces(count, 
            granularity, subspaces, requests, index_space_ready));
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
      std::vector<Realm::ZIndexSpace<DIM,T> > lhs_spaces(count);
      std::vector<Realm::ZIndexSpace<DIM,T> > rhs_spaces(count);
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
          left_child->get_realm_index_space(lhs_spaces[subspace_index]);
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!left_child->index_space_ready.has_triggered())
            preconditions.insert(left_child->index_space_ready);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
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
          left_child->get_realm_index_space(lhs_spaces[subspace_index]);
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!left_child->index_space_ready.has_triggered())
            preconditions.insert(left_child->index_space_ready);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
        }
      }
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(count);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                              op, DEP_PART_UNIONS);
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_unions(
            lhs_spaces, rhs_spaces, subspaces, requests,
            Runtime::merge_events(preconditions)));
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
      std::vector<Realm::ZIndexSpace<DIM,T> > lhs_spaces(count);
      std::vector<Realm::ZIndexSpace<DIM,T> > rhs_spaces(count);
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
          left_child->get_realm_index_space(lhs_spaces[subspace_index]);
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!left_child->index_space_ready.has_triggered())
            preconditions.insert(left_child->index_space_ready);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
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
          left_child->get_realm_index_space(lhs_spaces[subspace_index]);
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!left_child->index_space_ready.has_triggered())
            preconditions.insert(left_child->index_space_ready);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
        }
      }
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(count);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_INTERSECTIONS);
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_intersections(
            lhs_spaces, rhs_spaces, subspaces, requests,
            Runtime::merge_events(preconditions)));
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
      std::vector<Realm::ZIndexSpace<DIM,T> > rhs_spaces(count);
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
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
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
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
        }
      }
      if (!index_space_ready.has_triggered())
        preconditions.insert(index_space_ready);
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(count);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_INTERSECTIONS);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_intersections(
            realm_index_space, rhs_spaces, subspaces, requests,
            Runtime::merge_events(preconditions)));
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
      std::vector<Realm::ZIndexSpace<DIM,T> > lhs_spaces(count);
      std::vector<Realm::ZIndexSpace<DIM,T> > rhs_spaces(count);
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
          left_child->get_realm_index_space(lhs_spaces[subspace_index]);
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!left_child->index_space_ready.has_triggered())
            preconditions.insert(left_child->index_space_ready);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
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
          left_child->get_realm_index_space(lhs_spaces[subspace_index]);
          right_child->get_realm_index_space(rhs_spaces[subspace_index++]);
          if (!left_child->index_space_ready.has_triggered())
            preconditions.insert(left_child->index_space_ready);
          if (!right_child->index_space_ready.has_triggered())
            preconditions.insert(right_child->index_space_ready);
        }
      }
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(count);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_DIFFERENCES);
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_differences(
            lhs_spaces, rhs_spaces, subspaces, requests,
            Runtime::merge_events(preconditions)));
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
            const Realm::ZMatrix<1,DIM> *transform = 
              static_cast<const Realm::ZMatrix<1,DIM>*>(tran);
            const Realm::ZRect<1,T> *extent = 
              static_cast<const Realm::ZRect<1,T>*>(ext);
            return create_by_restriction_helper<1>(partition, 
                                                   *transform, *extent);
          }
        case 2:
          {
            const Realm::ZMatrix<2,DIM> *transform = 
              static_cast<const Realm::ZMatrix<2,DIM>*>(tran);
            const Realm::ZRect<2,T> *extent = 
              static_cast<const Realm::ZRect<2,T>*>(ext);
            return create_by_restriction_helper<2>(partition, 
                                                   *transform, *extent);
          }
        case 3:
          {
            const Realm::ZMatrix<3,DIM> *transform = 
              static_cast<const Realm::ZMatrix<3,DIM>*>(tran);
            const Realm::ZRect<3,T> *extent = 
              static_cast<const Realm::ZRect<3,T>*>(ext);
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
                                          const Realm::ZMatrix<M,N> &transform,
                                          const Realm::ZRect<M,T> &extent)
    //--------------------------------------------------------------------------
    {
      // Get the parent index space in case it has a sparsity map
      IndexSpaceNodeT<M,T> *parent = 
                      static_cast<IndexSpaceNodeT<M,T>*>(partition->parent);
      // No need to wait since we'll just be messing with the bounds
      Realm::ZIndexSpace<M,T> parent_is;
      parent->get_realm_index_space(parent_is);
      // Wait for our index space to be ready if necessary
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      // Iterate over our points (colors) and fill in the bounds
      for (Realm::ZIndexSpaceIterator<N,T> rect_itr(realm_index_space); 
            rect_itr.valid; rect_itr.step())
      {
        for (Realm::ZPointInRectIterator<N,T> color_itr(rect_itr.rect); 
              color_itr.valid; color_itr.step())
        {
          // Copy the index space from the parent
          Realm::ZIndexSpace<M,T> child_is = parent_is;
          // Compute the new bounds 
          child_is.bounds = extent + transform * color_itr.p;
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
      if (!color_space->index_space_ready.has_triggered())
        color_space->index_space_ready.lg_wait();
      Realm::ZIndexSpace<COLOR_DIM,COLOR_T> realm_color_space;
      color_space->get_realm_index_space(realm_color_space);
      const size_t num_colors = realm_color_space.volume();
      std::vector<Realm::ZPoint<COLOR_DIM,COLOR_T> > colors(num_colors);
      unsigned index = 0;
      for (Realm::ZIndexSpaceIterator<COLOR_DIM,COLOR_T> 
            rect_iter(realm_color_space); rect_iter.valid; rect_iter.step())
      {
        for (Realm::ZPointInRectIterator<COLOR_DIM,COLOR_T> 
              itr(rect_iter.rect); itr.valid; itr.step())
        {
#ifdef DEBUG_LEGION
          assert(index < colors.size());
#endif
          colors[index++] = itr.p;
        }
      }
      // Translate the instances to realm field data descriptors
      typedef Realm::FieldDataDescriptor<Realm::ZIndexSpace<DIM,T>,
                Realm::ZPoint<COLOR_DIM,COLOR_T> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM,T> *node = static_cast<IndexSpaceNodeT<DIM,T>*>(
                                          context->get_node(src.index_space));
        node->get_realm_index_space(dst.index_space);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_FIELD);
      // Perform the operation
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(colors.size());
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent result = ApEvent(realm_index_space.create_subspaces_by_field(
            descriptors, colors, subspaces, requests, 
            Runtime::merge_events(instances_ready, index_space_ready)));
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
      std::vector<Realm::ZIndexSpace<DIM2,T2> > 
                                sources(projection->color_space->get_volume());
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          child->get_realm_index_space(sources[color]);
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
          child->get_realm_index_space(sources[index++]);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::ZIndexSpace<DIM2,T2>,
                                       Realm::ZPoint<DIM1,T1> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM2,T2> *node = static_cast<IndexSpaceNodeT<DIM2,T2>*>(
                                          context->get_node(src.index_space));
        node->get_realm_index_space(dst.index_space);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_IMAGE);
      // Perform the operation
      std::vector<Realm::ZIndexSpace<DIM1,T1> > subspaces(sources.size());
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent result(realm_index_space.create_subspaces_by_image(descriptors,
            sources, subspaces, requests, 
            Runtime::merge_events(index_space_ready, instances_ready)));
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
      std::vector<Realm::ZIndexSpace<DIM2,T2> > 
                                sources(projection->color_space->get_volume());
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          child->get_realm_index_space(sources[color]);
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
          child->get_realm_index_space(sources[index++]);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::ZIndexSpace<DIM2,T2>,
                                       Realm::ZRect<DIM1,T1> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM2,T2> *node = static_cast<IndexSpaceNodeT<DIM2,T2>*>(
                                          context->get_node(src.index_space));
        node->get_realm_index_space(dst.index_space);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_IMAGE_RANGE);
      // Perform the operation
      std::vector<Realm::ZIndexSpace<DIM1,T1> > subspaces(sources.size());
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent result(realm_index_space.create_subspaces_by_image(descriptors,
            sources, subspaces, requests, 
            Runtime::merge_events(index_space_ready, instances_ready)));
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
      std::vector<Realm::ZIndexSpace<DIM2,T2> > 
                                targets(projection->color_space->get_volume());
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          child->get_realm_index_space(targets[color]);
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
          child->get_realm_index_space(targets[index++]);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::ZIndexSpace<DIM1,T1>,
                                       Realm::ZPoint<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM1,T1> *node = static_cast<IndexSpaceNodeT<DIM1,T1>*>(
                                          context->get_node(src.index_space));
        node->get_realm_index_space(dst.index_space);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_PREIMAGE);
      // Perform the operation
      std::vector<Realm::ZIndexSpace<DIM1,T1> > subspaces(targets.size());
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent result(realm_index_space.create_subspaces_by_preimage(
            descriptors, targets, subspaces, requests,
            Runtime::merge_events(index_space_ready, instances_ready)));
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
      std::vector<Realm::ZIndexSpace<DIM2,T2> > 
                                targets(projection->color_space->get_volume());
      if (partition->total_children == partition->max_linearized_color)
      {
        // Always use the partitions color space
        for (LegionColor color = 0; color < partition->total_children; color++)
        {
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM2,T2> *child = 
           static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(color));
          child->get_realm_index_space(targets[color]);
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
          child->get_realm_index_space(targets[index++]);
        }
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::ZIndexSpace<DIM1,T1>,
                                       Realm::ZRect<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM1,T1> *node = static_cast<IndexSpaceNodeT<DIM1,T1>*>(
                                          context->get_node(src.index_space));
        node->get_realm_index_space(dst.index_space);
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_PREIMAGE_RANGE);
      // Perform the operation
      std::vector<Realm::ZIndexSpace<DIM1,T1> > subspaces(targets.size());
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      ApEvent result(realm_index_space.create_subspaces_by_preimage(
            descriptors, targets, subspaces, requests,
            Runtime::merge_events(index_space_ready, instances_ready)));
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
      typedef Realm::FieldDataDescriptor<Realm::ZIndexSpace<DIM1,T1>,
                                       Realm::ZPoint<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.inst = src.inst;
        dst.field_offset = src.field_offset;
        IndexSpaceNodeT<DIM1,T1> *node = static_cast<IndexSpaceNodeT<DIM1,T1>*>(
                                          context->get_node(src.index_space));
        node->get_realm_index_space(dst.index_space);
      }
      // Get the range index space
      IndexSpaceNodeT<DIM2,T2> *range_node = 
        static_cast<IndexSpaceNodeT<DIM2,T2>*>(range);
      Realm::ZIndexSpace<DIM2,T2> range_space;
      range_node->get_realm_index_space(range_space);
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_ASSOCIATION);
      if (!realm_index_space_set.has_triggered())
        realm_index_space_set.lg_wait();
      // Issue the operation
      return ApEvent(realm_index_space.create_association(descriptors,
            range_space, requests, Runtime::merge_events(instances_ready,
              index_space_ready, range_node->index_space_ready)));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::check_field_size(size_t field_size, bool range)
    //--------------------------------------------------------------------------
    {
      if (range)
        return (sizeof(Realm::ZRect<DIM,T>) == field_size);
      else
        return (sizeof(Realm::ZPoint<DIM,T>) == field_size);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::issue_copy(Operation *op,
                        const std::vector<CopySrcDstField> &src_fields,
                        const std::vector<CopySrcDstField> &dst_fields,
                        ApEvent precondition, PredEvent predicate_guard,
                        IndexTreeNode *intersect/*=NULL*/,
                        ReductionOpID redop /*=0*/,bool reduction_fold/*=true*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_ISSUE_COPY_CALL);
      Realm::ProfilingRequestSet requests;
      op->add_copy_profiling_request(requests);
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_copy_request(requests, op);
      ApEvent result;
      if (intersect == NULL)
      {
        // Include our event precondition if necessary
        if (!index_space_ready.has_triggered())
          precondition = Runtime::merge_events(precondition, index_space_ready);
        if (!realm_index_space_set.has_triggered())
          realm_index_space_set.lg_wait();
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
          result = Runtime::ignorefaults(realm_index_space.copy(src_fields, 
                dst_fields, requests, pred_pre, redop, reduction_fold));
        }
        else
          result = ApEvent(realm_index_space.copy(src_fields, dst_fields,
                        requests, precondition, redop, reduction_fold));
      }
      else
      {
        // This is a copy between the intersection of two nodes
        Realm::ZIndexSpace<DIM,T> intersection;
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
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
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
                        const std::vector<CopySrcDstField> &dst_fields,
                        const void *fill_value, size_t fill_size,
                        ApEvent precondition, PredEvent predicate_guard,
                        IndexTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_ISSUE_FILL_CALL);
      Realm::ProfilingRequestSet requests;
      op->add_copy_profiling_request(requests);
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_fill_request(requests, op);
      ApEvent result;
      if (intersect == NULL)
      {
        // Include our event precondition if necessary
        if (!index_space_ready.has_triggered())
          precondition = Runtime::merge_events(precondition, index_space_ready);
        if (!realm_index_space_set.has_triggered())
          realm_index_space_set.lg_wait();
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
          result = Runtime::ignorefaults(realm_index_space.fill(dst_fields, 
                requests, fill_value, fill_size, pred_pre));
        }
        else
          result = ApEvent(realm_index_space.fill(dst_fields, requests, 
                fill_value, fill_size, precondition));
      }
      else
      {
        // This is a copy between the intersection of two nodes
        Realm::ZIndexSpace<DIM,T> intersection;
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
        // Have to protect against misspeculation
        if (predicate_guard.exists())
        {
          ApEvent pred_pre = Runtime::merge_events(precondition,
                                                   ApEvent(predicate_guard));
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
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_instance(Memory target,
                                       const std::vector<size_t> &field_sizes,
                                       size_t blocking_factor, UniqueID op_id)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_CREATE_INSTANCE_CALL);
      // Have to wait for the index space to be ready if necessary
      if (!realm_index_space_set.has_triggered())
          realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
      {
        context->runtime->profiler->add_inst_request(requests, op_id);
        PhysicalInstance result;
	LgEvent ready(PhysicalInstance::create_instance(result, target,
							realm_index_space,
							field_sizes, requests));
	// TODO
	ready.lg_wait();
        // If the result exists tell the profiler about it in case
        // it never gets deleted and we never see the profiling feedback
        if (result.exists())
        {
          unsigned long long creation_time = 
            Realm::Clock::current_time_in_nanoseconds();
          context->runtime->profiler->record_instance_creation(result, target, 
                                                        op_id, creation_time);
        }
        return result;
      }
      else
      {
        PhysicalInstance result;
	LgEvent ready(PhysicalInstance::create_instance(result, target,
							realm_index_space,
							field_sizes, requests));
	// TODO
	ready.lg_wait();
	return result;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_file_instance(
                                         const char *file_name,
                                         const std::vector<size_t> &field_sizes,
                                         legion_lowlevel_file_mode_t file_mode)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_CREATE_INSTANCE_CALL);
      // Have to wait for the index space to be ready if necessary
      if (!realm_index_space_set.has_triggered())
          realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      // No profiling for these kinds of instances currently
      Realm::ProfilingRequestSet requests;
      PhysicalInstance result;
      LgEvent ready(PhysicalInstance::create_file_instance(result, file_name, 
							   realm_index_space,
							   field_sizes,
							   file_mode, requests));
      // TODO
      ready.lg_wait();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_hdf5_instance(
                                    const char *file_name,
                                    const std::vector<size_t> &field_sizes,
                                    const std::vector<const char*> &field_files,
                                    bool read_only)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_CREATE_INSTANCE_CALL);
      // Have to wait for the index space to be ready if necessary
      if (!realm_index_space_set.has_triggered())
          realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      // No profiling for these kinds of instances currently
      Realm::ProfilingRequestSet requests;
      PhysicalInstance result;
      LgEvent ready(PhysicalInstance::create_hdf5_instance(result, file_name, 
							   realm_index_space,
							   field_sizes,
							   field_files,
							   read_only,
							   requests));
      // TODO
      ready.lg_wait();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::get_launch_space_domain(Domain &launch_domain)
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
          realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      launch_domain = realm_index_space;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_launch_space(UniqueID op_id)
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space_set.has_triggered())
          realm_index_space_set.lg_wait();
      if (!index_space_ready.has_triggered())
        index_space_ready.lg_wait();
      for (Realm::ZIndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
        LegionSpy::log_launch_index_space_rect<DIM>(op_id, itr.rect);
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
                                        ApEvent partition_ready, 
                                        ApUserEvent pend)
      : IndexPartNode(ctx, p, par, cs, c, disjoint, partition_ready, pend),
        has_union_space(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(RegionTreeForest *ctx, 
                                        IndexPartition p,
                                        IndexSpaceNode *par, IndexSpaceNode *cs,
                                        LegionColor c, RtEvent disjoint_event,
                                        ApEvent partition_ready, 
                                        ApUserEvent pending)
      : IndexPartNode(ctx, p, par, cs, c, disjoint_event, 
                      partition_ready, pending),
        has_union_space(false)
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
      Realm::ZIndexSpace<DIM,T> parent_space, union_space, difference_space;
      ApEvent union_ready = get_union_index_space(union_space);
      static_cast<IndexSpaceNodeT<DIM,T>*>(parent)->get_realm_index_space(
                                                              parent_space);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                      (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
      ApEvent diff_ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
            parent_space, union_space, difference_space, requests,
            Runtime::merge_events(parent->index_space_ready, union_ready)));
      if (!diff_ready.has_triggered())
        diff_ready.lg_wait();
      // Always tighten these tests so that they are precise
      Realm::ZIndexSpace<DIM,T> tight_space = difference_space.tighten();
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
      Realm::ZIndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent union_precondition = get_union_index_space(lhs_space);
      rhs_node->get_realm_index_space(rhs_space);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
            lhs_space, rhs_space, intersection, requests,
            Runtime::merge_events(union_precondition, 
                                  rhs_node->index_space_ready)));
      if (!ready.has_triggered())
        ready.lg_wait();
      // Always tighten these tests so that they are precise
      Realm::ZIndexSpace<DIM,T> tight_intersection = intersection.tighten();
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
      Realm::ZIndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent union_precondition = get_union_index_space(lhs_space);
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                    (Operation*)NULL/*op*/, DEP_PART_INTERSECTION);
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
            lhs_space, rhs_space, intersection, requests,
            Runtime::merge_events(union_precondition, rhs_precondition)));
      if (!ready.has_triggered())
        ready.lg_wait();
      // Always tighten these tests so that they are precise
      Realm::ZIndexSpace<DIM,T> tight_intersection = intersection.tighten();
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
      Realm::ZIndexSpace<DIM,T> union_space, rhs_space, difference;
      ApEvent union_precondition = get_union_index_space(union_space);
      rhs_node->get_realm_index_space(rhs_space);
      bool result = false;
      if (!union_precondition.has_triggered() || !union_space.dense() ||
          !rhs_node->index_space_ready.has_triggered())
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                          (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
              rhs_space, union_space, difference, requests,
              Runtime::merge_events(union_precondition, 
                                    rhs_node->index_space_ready)));
        if (!ready.has_triggered())
          ready.lg_wait();
        // Always tighten these tests so that they are precise
        Realm::ZIndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      }
      else // Fast path
        result = union_space.bounds.contains(rhs_space);
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
      Realm::ZIndexSpace<DIM,T> union_space, rhs_space, difference;
      ApEvent union_precondition = get_union_index_space(union_space);
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      bool result = false;
      if (!union_precondition.has_triggered() || !union_space.dense() ||
          !rhs_precondition.has_triggered())
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                        (Operation*)NULL/*op*/, DEP_PART_DIFFERENCE);
        ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
              rhs_space, union_space, difference, requests,
              Runtime::merge_events(union_precondition, rhs_precondition))); 
        if (!ready.has_triggered())
          ready.lg_wait();
        // Always tighten these tests so that they are precise
        Realm::ZIndexSpace<DIM,T> tight_difference = difference.tighten();
        result = tight_difference.empty();
        difference.destroy();
        tight_difference.destroy();
      } 
      else // Fast path
        result = union_space.bounds.contains(rhs_space);
      AutoLock n_lock(node_lock);
      dominators[rhs] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexPartNodeT<DIM,T>::get_union_index_space(
                                               Realm::ZIndexSpace<DIM,T> &space)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        if (has_union_space)
        {
          space = partition_union_space;
          return partition_union_ready;
        }
      }
      // Compute the space and then save it
      std::set<ApEvent> preconditions;
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(
                                                color_space->get_volume());
      unsigned subspace_index = 0;
      if (total_children == max_linearized_color)
      {
        for (LegionColor color = 0; color < total_children; color++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(get_child(color));
          if (!child->index_space_ready.has_triggered())
            preconditions.insert(child->index_space_ready);
          child->get_realm_index_space(subspaces[subspace_index++]);
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
          if (!child->index_space_ready.has_triggered())
            preconditions.insert(child->index_space_ready);
          child->get_realm_index_space(subspaces[subspace_index++]);
        }
      }
      AutoLock n_lock(node_lock);
      // See if we lost the race
      if (has_union_space)
      {
        space = partition_union_space;
        return partition_union_ready;
      }
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                  (Operation*)NULL/*op*/, DEP_PART_UNION_REDUCTION);
      partition_union_ready = 
        ApEvent(Realm::ZIndexSpace<DIM,T>::compute_union(subspaces,
              partition_union_space, requests, 
              Runtime::merge_events(preconditions)));
      space = partition_union_space;
      return partition_union_ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM,T>::destroy_node(AddressSpaceID source) 
    //--------------------------------------------------------------------------
    {
      // If we've already been destroyed then we are done
      if (destroyed)
        return;
      std::set<PartitionNode*> to_destroy;
      {
        AutoLock n_lock(node_lock);
        if (!destroyed)
        {
          destroyed = true;
          if (!creation_set.empty())
          {
            DestructionFunctor functor(handle, context->runtime);
            creation_set.map(functor);
          }
          to_destroy = logical_nodes;
          
        }
      }
      for (std::set<PartitionNode*>::const_iterator it = 
            to_destroy.begin(); it != to_destroy.end(); it++)
      {
        (*it)->destroy_node(source);
      }
      if (has_union_space && !partition_union_space.empty())
        partition_union_space.destroy();
      for (typename std::map<IndexTreeNode*,IntersectInfo>::iterator it = 
            intersections.begin(); it != intersections.end(); it++)
        if (it->second.has_intersection && it->second.intersection_valid)
          it->second.intersection.destroy();
    }

  }; // namespace Internal
}; // namespace Legion

