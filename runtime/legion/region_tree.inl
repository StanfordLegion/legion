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
        realm_index_space = *is;
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
      for (typename std::map<IndexTreeNode*,Realm::ZIndexSpace<DIM,T> >::
           iterator it = intersections.begin(); it != intersections.end(); it++)
        it->second.destroy();
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
    void* IndexSpaceNodeT<DIM,T>::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<IndexSpaceNodeT<DIM,T>,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_index_space_points(void) const
    //--------------------------------------------------------------------------
    {
      if (!realm_index_space.empty())
      {
        // Iterate over the rectangles and print them out 
        for (Realm::ZIndexSpaceIterator<DIM,T> itr(realm_index_space); 
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
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::compute_pending_space(
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
          if (is_union)
            log_run.error("Dynamic type mismatch in "
                          "'create_index_space_union'");
          else
            log_run.error("Dynamic type mismatch in "
                          "'create_index_space_intersection'");
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
      Realm::ProfilingRequestSet empty_requests;
      if (is_union)
        return ApEvent(Realm::ZIndexSpace<DIM,T>::compute_union(
              spaces, realm_index_space, empty_requests, precondition));
      else
        return ApEvent(Realm::ZIndexSpace<DIM,T>::compute_intersection(
              spaces, realm_index_space, empty_requests, precondition));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::compute_pending_difference(IndexSpace init,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> preconditions;
      std::vector<Realm::ZIndexSpace<DIM,T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        IndexSpaceNode *node = context->get_node(handles[idx]);
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          log_run.error("Dynamic type mismatch in "
                        "'create_index_space_difference'");
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
      Realm::ProfilingRequestSet empty_requests;
      // Compute the union of the handles for the right-hand side
      Realm::ZIndexSpace<DIM,T> rhs_space;
      ApEvent rhs_ready(Realm::ZIndexSpace<DIM,T>::compute_union(
            spaces, rhs_space, empty_requests, precondition));
      if (init.get_type_tag() != handle.get_type_tag())
      {
        log_run.error("Dynamic type mismatch in "
                      "'create_index_space_difference'");
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DYNAMIC_TYPE_MISMATCH);
      }
      IndexSpaceNodeT<DIM,T> *lhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(context->get_node(init));
      Realm::ZIndexSpace<DIM,T> lhs_space;
      lhs_node->get_realm_index_space(lhs_space);
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_difference(
            lhs_space, rhs_space, realm_index_space, empty_requests,
            Runtime::merge_events(lhs_node->index_space_ready, rhs_ready)));
      // Destroy the tempory rhs space once the computation is done
      rhs_space.destroy(result);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::compute_pending_space(
                                      IndexPartition part_handle, bool is_union)
    //--------------------------------------------------------------------------
    {
      if (part_handle.get_type_tag() != handle.get_type_tag())
      {
        if (is_union)
          log_run.error("Dynamic type mismatch in "
                        "'create_index_space_union'");
        else
          log_run.error("Dynamic type mismatch in "
                        "'create_index_space_intersection'");
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
      Realm::ProfilingRequestSet empty_requests;
      if (is_union)
        return ApEvent(Realm::ZIndexSpace<DIM,T>::compute_union(
              spaces, realm_index_space, empty_requests, precondition));
      else
        return ApEvent(Realm::ZIndexSpace<DIM,T>::compute_intersection(
              spaces, realm_index_space, empty_requests, precondition));
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
      *target = realm_index_space;
      // If the event isn't ready we have to wait 
      if (!index_space_ready.has_triggered())
        index_space_ready.wait();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::get_volume(void) const
    //--------------------------------------------------------------------------
    {
      if (!index_space_ready.has_triggered())
        index_space_ready.wait();
      return realm_index_space.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::get_max_linearized_color(void) const
    //--------------------------------------------------------------------------
    {
      return realm_index_space.bounds.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::compute_linearization_metadata(void)
    //--------------------------------------------------------------------------
    {
      if (!index_space_ready.has_triggered())
        index_space_ready.wait();
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
      if (!index_space_ready.has_triggered())
        index_space_ready.wait();
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
    void IndexSpaceNodeT<DIM,T>::instantiate_color_space(
                              IndexPartNode *partition, ApUserEvent instantiate)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instantiate.exists());
#endif
      if (!index_space_ready.has_triggered())
        index_space_ready.wait();
      std::set<ApEvent> preconditions;
      const IndexPartition pid = partition->handle;
      for (Realm::ZIndexSpaceIterator<DIM,T> rect_itr(realm_index_space); 
            rect_itr.valid; rect_itr.step())
      {
        for (Realm::ZPointInRectIterator<DIM,T> itr(rect_itr.rect);
              itr.valid; itr.step())
        {
          IndexSpace is = context->create_child_space_name(pid);
          LegionColor color = linearize_color(&itr.p, handle.get_type_tag());
          ApUserEvent child_ready = Runtime::create_ap_user_event();
          preconditions.insert(child_ready);
          context->create_node(is, NULL, partition, color, child_ready);
          partition->add_pending_child(color, child_ready);
        }
      }
      // Trigger the result once it is ready
      Runtime::trigger_event(instantiate, Runtime::merge_events(preconditions));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::instantiate_colors(
                                               std::vector<LegionColor> &colors)
    //--------------------------------------------------------------------------
    {
      if (!index_space_ready.has_triggered())
        index_space_ready.wait();
      colors.resize(get_volume());
      unsigned idx = 0;
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
      if (!index_space_ready.has_triggered())
        index_space_ready.wait();
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
    bool IndexSpaceNodeT<DIM,T>::intersects_with(IndexSpaceNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,Realm::ZIndexSpace<DIM,T> >::
          const_iterator finder = intersections.find(rhs);
        if (finder != intersections.end())
          return !finder->second.empty();
      }
      Realm::ZIndexSpace<DIM,T> rhs_space, intersection;
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      rhs_node->get_realm_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
        realm_index_space, rhs_space, intersection, empty_requests,
        Runtime::merge_events(index_space_ready, rhs_node->index_space_ready)));
      // Wait for the result to be ready
      if (!ready.has_triggered())
        ready.wait();
      bool result = !intersection.empty();
      AutoLock n_lock(node_lock);
      if (intersections.find(rhs) == intersections.end())
        intersections[rhs] = intersection;
      else
        intersection.destroy();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::intersects_with(IndexPartNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,Realm::ZIndexSpace<DIM,T> >::
          const_iterator finder = intersections.find(rhs);
        if (finder != intersections.end())
          return !finder->second.empty();
      }
      Realm::ZIndexSpace<DIM,T> rhs_space, intersection;
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
            realm_index_space, rhs_space, intersection, empty_requests,
            Runtime::merge_events(index_space_ready, rhs_precondition)));
      if (!ready.has_triggered())
        ready.wait();
      bool result = !intersection.empty();
      AutoLock n_lock(node_lock);
      if (intersections.find(rhs) == intersections.end())
        intersections[rhs] = intersection;
      else
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
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(rhs);
        if (finder != dominators.end())
          return finder->second;
      }
      Realm::ZIndexSpace<DIM,T> rhs_space, difference;
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      rhs_node->get_realm_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
        rhs_space, realm_index_space, difference, empty_requests,
        Runtime::merge_events(index_space_ready, rhs_node->index_space_ready)));
      if (!ready.has_triggered())
        ready.wait();
      bool result = difference.empty();
      if (!result)
        difference.destroy();
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
      Realm::ZIndexSpace<DIM,T> rhs_space, difference;
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
            rhs_space, realm_index_space, difference, empty_requests,
            Runtime::merge_events(index_space_ready, rhs_precondition)));
      if (!ready.has_triggered())
        ready.wait();
      bool result = difference.empty();
      if (!result)
        difference.destroy();
      AutoLock n_lock(node_lock);
      dominators[rhs] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::pack_index_space(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(sizeof(realm_index_space));
      rez.serialize(realm_index_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_equal_children(
                                   IndexPartNode *partition, size_t granularity)
    //--------------------------------------------------------------------------
    {
      const size_t count = partition->color_space->get_volume(); 
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(count);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent result(realm_index_space.create_equal_subspaces(count, 
            granularity, subspaces, empty_requests, index_space_ready));
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
          child->set_realm_index_space(subspaces[subspace_index++]);
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
          child->set_realm_index_space(subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_union(IndexPartNode *partition,
                                                    IndexPartNode *left,
                                                    IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
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
      Realm::ProfilingRequestSet empty_requests;
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_unions(
            lhs_spaces, rhs_spaces, subspaces, empty_requests,
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
          child->set_realm_index_space(subspaces[subspace_index++]);
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
          child->set_realm_index_space(subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_intersection(
                                                      IndexPartNode *partition,
                                                      IndexPartNode *left,
                                                      IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
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
      Realm::ProfilingRequestSet empty_requests;
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_intersections(
            lhs_spaces, rhs_spaces, subspaces, empty_requests,
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
          child->set_realm_index_space(subspaces[subspace_index++]);
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
          child->set_realm_index_space(subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_intersection(
                                                      IndexPartNode *partition,
                                                      IndexSpaceNode *left,
                                                      IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
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
      IndexSpaceNodeT<DIM,T> *left_child = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(left);
      Realm::ZIndexSpace<DIM,T> left_space;
      left_child->get_realm_index_space(left_space);
      if (!left_child->index_space_ready.has_triggered())
        preconditions.insert(left_child->index_space_ready);
      std::vector<Realm::ZIndexSpace<DIM,T> > subspaces(count);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_intersections(
            left_space, rhs_spaces, subspaces, empty_requests,
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
          child->set_realm_index_space(subspaces[subspace_index++]);
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
          child->set_realm_index_space(subspaces[subspace_index++]);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_difference(
                                                      IndexPartNode *partition,
                                                      IndexPartNode *left,
                                                      IndexPartNode *right)
    //--------------------------------------------------------------------------
    {
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
      Realm::ProfilingRequestSet empty_requests;
      ApEvent result(Realm::ZIndexSpace<DIM,T>::compute_differences(
            lhs_spaces, rhs_spaces, subspaces, empty_requests,
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
          child->set_realm_index_space(subspaces[subspace_index++]);
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
          child->set_realm_index_space(subspaces[subspace_index++]);
        }
      }
      return result;
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
                                        ApEvent partition_ready)
      : IndexPartNode(ctx, p, par, cs, c, disjoint, partition_ready),
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
                                        ApEvent partition_ready)
      : IndexPartNode(ctx, p, par, cs, c, disjoint_event, partition_ready),
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
      if (has_union_space && !partition_union_space.empty())
        partition_union_space.destroy();
      for (typename std::map<IndexTreeNode*,Realm::ZIndexSpace<DIM,T> >::
           iterator it = intersections.begin(); it != intersections.end(); it++)
        it->second.destroy();
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
    void* IndexPartNodeT<DIM,T>::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<IndexPartNodeT<DIM,T>,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM,T>::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
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
      Realm::ProfilingRequestSet empty_requests;
      ApEvent diff_ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
            parent_space, union_space, difference_space, empty_requests,
            Runtime::merge_events(parent->index_space_ready, union_ready)));
      if (!diff_ready.has_triggered())
        diff_ready.wait();
      bool complete = difference_space.empty();
      if (!complete)
        difference_space.destroy();
      return complete;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::intersects_with(IndexSpaceNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,Realm::ZIndexSpace<DIM,T> >::
          const_iterator finder = intersections.find(rhs);
        if (finder != intersections.end())
          return !finder->second.empty();
      }
      Realm::ZIndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent union_precondition = get_union_index_space(lhs_space);
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      rhs_node->get_realm_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
            lhs_space, rhs_space, intersection, empty_requests,
            Runtime::merge_events(union_precondition, 
                                  rhs_node->index_space_ready)));
      if (!ready.has_triggered())
        ready.wait();
      bool result = !intersection.empty();
      AutoLock n_lock(node_lock);
      if (intersections.find(rhs) == intersections.end())
        intersections[rhs] = intersection;
      else
        intersection.destroy();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::intersects_with(IndexPartNode *rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs->handle.get_type_tag() == handle.get_type_tag());
#endif
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,Realm::ZIndexSpace<DIM,T> >::
          const_iterator finder = intersections.find(rhs);
        if (finder != intersections.end())
          return !finder->second.empty();
      }
      Realm::ZIndexSpace<DIM,T> lhs_space, rhs_space, intersection;
      ApEvent union_precondition = get_union_index_space(lhs_space);
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_intersection(
            lhs_space, rhs_space, intersection, empty_requests,
            Runtime::merge_events(union_precondition, rhs_precondition)));
      if (!ready.has_triggered())
        ready.wait();
      bool result = !intersection.empty();
      AutoLock n_lock(node_lock);
      if (intersections.find(rhs) == intersections.end())
        intersections[rhs] = intersection;
      else
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
      Realm::ZIndexSpace<DIM,T> union_space, rhs_space, difference;
      ApEvent union_precondition = get_union_index_space(union_space);
      IndexSpaceNodeT<DIM,T> *rhs_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(rhs);
      rhs_node->get_realm_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
            rhs_space, union_space, difference, empty_requests,
            Runtime::merge_events(union_precondition, 
                                  rhs_node->index_space_ready)));
      if (!ready.has_triggered())
        ready.wait();
      bool result = difference.empty();
      if (!result)
        difference.destroy();
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
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        typename std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(rhs);
        if (finder != dominators.end())
          return finder->second;
      }
      Realm::ZIndexSpace<DIM,T> union_space, rhs_space, difference;
      ApEvent union_precondition = get_union_index_space(union_space);
      IndexPartNodeT<DIM,T> *rhs_node = 
        static_cast<IndexPartNodeT<DIM,T>*>(rhs);
      ApEvent rhs_precondition = rhs_node->get_union_index_space(rhs_space);
      Realm::ProfilingRequestSet empty_requests;
      ApEvent ready(Realm::ZIndexSpace<DIM,T>::compute_difference(
            rhs_space, union_space, difference, empty_requests,
            Runtime::merge_events(union_precondition, rhs_precondition))); 
      if (!ready.has_triggered())
        ready.wait();
      bool result = difference.empty();
      if (!result)
        difference.destroy();
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
      Realm::ProfilingRequestSet empty_requests;
      partition_union_ready = 
        ApEvent(Realm::ZIndexSpace<DIM,T>::compute_union(subspaces,
              partition_union_space, empty_requests, 
              Runtime::merge_events(preconditions)));
      space = partition_union_space;
      return partition_union_ready;
    }

  }; // namespace Internal
}; // namespace Legion

