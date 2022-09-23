/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// Useful for IDEs 
#include "legion/region_tree.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS 

#ifdef DEFINE_NT_TEMPLATES
    /////////////////////////////////////////////////////////////
    // PieceIteratorImplT
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PieceIteratorImplT<DIM,T>::PieceIteratorImplT(const void *piece_list,
                 size_t piece_list_size, IndexSpaceNodeT<DIM,T> *privilege_node)
      : PieceIteratorImpl()
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((piece_list_size % sizeof(Rect<DIM,T>)) == 0);
#endif
      const size_t num_pieces = piece_list_size / sizeof(Rect<DIM,T>);
      const Rect<DIM,T> *rects = static_cast<const Rect<DIM,T>*>(piece_list);
      if (privilege_node != NULL)
      {
        Realm::IndexSpace<DIM,T> privilege_space;
        const ApEvent ready = 
          privilege_node->get_realm_index_space(privilege_space, true/*tight*/);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (unsigned idx = 0; idx < num_pieces; idx++)
        {
          const Rect<DIM,T> &rect = rects[idx];
          for (Realm::IndexSpaceIterator<DIM,T> itr(privilege_space); 
                itr.valid; itr.step())
          {
            const Rect<DIM,T> overlap = rect.intersection(itr.rect);
            if (!overlap.empty())
              pieces.push_back(overlap);
          }
        }
      }
      else
      {
        pieces.resize(num_pieces);
        for (unsigned idx = 0; idx < num_pieces; idx++)
          pieces[idx] = rects[idx];
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    int PieceIteratorImplT<DIM,T>::get_next(int index, Domain &next_piece)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index >= -1);
#endif
      const unsigned next = index + 1;
      if (next < pieces.size())
      {
        next_piece = pieces[next];
        return int(next);
      }
      else
        return -1;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Expression 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceExpression::issue_fill_internal(
                                 RegionTreeForest *forest,
                                 const Realm::IndexSpace<DIM,T> &space,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                 UniqueID fill_uid,
                                 FieldSpace handle,
                                 RegionTreeID tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(forest->runtime, REALM_ISSUE_FILL_CALL);
#ifdef DEBUG_LEGION
      // We should only have empty spaces for fills that are indirections
      if (space.empty())
      {
        bool is_indirect = false;
        for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        {
          if (dst_fields[idx].indirect_index < 0)
            continue;
          is_indirect = true;
          break;
        }
        assert(is_indirect);
      }
#endif
      // Now that we know we're going to do this fill add any profiling requests
      Realm::ProfilingRequestSet requests;
      if (trace_info.op != NULL)
        trace_info.op->add_copy_profiling_request(trace_info, requests, true);
      if (forest->runtime->profiler != NULL)
        forest->runtime->profiler->add_fill_request(requests, trace_info.op);
#ifdef LEGION_SPY
      // Have to convert back to Realm data structures because C++ is dumb
      std::vector<Realm::CopySrcDstField> realm_dst_fields(dst_fields.size());
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        realm_dst_fields[idx] = dst_fields[idx];
#endif
      ApEvent result;
      if (pred_guard.exists())
      {
        ApEvent pred_pre = 
          Runtime::merge_events(&trace_info, precondition, ApEvent(pred_guard));
        if (trace_info.recording)
          trace_info.record_merge_events(pred_pre, precondition,
                                          ApEvent(pred_guard));
#ifdef LEGION_SPY
        result = Runtime::ignorefaults(space.fill(realm_dst_fields, requests, 
                                              fill_value, fill_size, pred_pre));
#else
        result = Runtime::ignorefaults(space.fill(dst_fields, requests, 
                                              fill_value, fill_size, pred_pre));
#endif                               
      }
      else
      {
#ifdef LEGION_SPY
        result = ApEvent(space.fill(realm_dst_fields, requests, 
                                    fill_value, fill_size, precondition));
#else
        result = ApEvent(space.fill(dst_fields, requests, 
                                    fill_value, fill_size, precondition));
#endif
      }
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists())
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      assert(trace_info.op != NULL);
      LegionSpy::log_fill_events(trace_info.op->get_unique_op_id(), 
          expr_id, handle, tree_id, precondition, result, fill_uid);
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        LegionSpy::log_fill_field(result, dst_fields[idx].field_id,
                                  dst_fields[idx].inst_event);
#endif
      if (trace_info.recording)
        trace_info.record_issue_fill(result, this, dst_fields,
                                     fill_value, fill_size,
#ifdef LEGION_SPY
                                     handle, tree_id,
#endif
                                     precondition, pred_guard);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceExpression::issue_copy_internal(
                                 RegionTreeForest *forest,
                                 const Realm::IndexSpace<DIM,T> &space,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const std::vector<CopySrcDstField> &src_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(forest->runtime, REALM_ISSUE_COPY_CALL);
#ifdef DEBUG_LEGION
      assert(!space.empty());
#endif
      // Now that we know we're going to do this copy add any profling requests
      Realm::ProfilingRequestSet requests;
      if (trace_info.op != NULL)
        trace_info.op->add_copy_profiling_request(trace_info, requests, false);
      if (forest->runtime->profiler != NULL)
        forest->runtime->profiler->add_copy_request(requests, trace_info.op);
#ifdef LEGION_SPY
      // Have to convert back to Realm structures because C++ is dumb  
      std::vector<Realm::CopySrcDstField> realm_src_fields(src_fields.size());
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        realm_src_fields[idx] = src_fields[idx];
      std::vector<Realm::CopySrcDstField> realm_dst_fields(dst_fields.size());
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        realm_dst_fields[idx] = dst_fields[idx];
#endif 
      ApEvent result;
      if (pred_guard.exists())
      {
        // No need for tracing to know about the precondition or reservations
        ApEvent pred_pre = precondition; 
          Runtime::merge_events(&trace_info, precondition, ApEvent(pred_guard));
        if (!reservations.empty())
        {
          // Need a protected version here to guarantee we always acquire
          // or release the lock regardless of poison
          pred_pre = Runtime::ignorefaults(precondition);
          for (std::vector<Reservation>::const_iterator it =
                reservations.begin(); it != reservations.end(); it++)
            pred_pre = 
              Runtime::acquire_ap_reservation(*it, true/*exclusive*/, pred_pre);
          // Tricky: now merge the predicate and precondition back in to get the 
          // effects of any poison since we protected against it above
          // Note you can't wait to acquire events until you know the full
          // precondition has triggered or poisoned including the predicate
          // or you risk deadlock which is why we need the double merge
          pred_pre =
            Runtime::merge_events(&trace_info, pred_pre, ApEvent(pred_guard));
        }
#ifdef LEGION_SPY
        result = Runtime::ignorefaults(space.copy(realm_src_fields, 
              realm_dst_fields, requests, pred_pre));
#else
        result = Runtime::ignorefaults(space.copy(src_fields, dst_fields, 
                                requests, pred_pre));
#endif
      }
      else
      {
        // No need for tracing to know about the reservations
        ApEvent copy_pre = precondition;
        for (std::vector<Reservation>::const_iterator it =
              reservations.begin(); it != reservations.end(); it++)
          copy_pre = Runtime::acquire_ap_reservation(*it, 
                                          true/*exclusive*/, copy_pre);
#ifdef LEGION_SPY
        result = ApEvent(space.copy(realm_src_fields, realm_dst_fields, 
                          requests, copy_pre));
#else
        result = ApEvent(space.copy(src_fields, dst_fields, requests, 
                          copy_pre));
#endif
      }
      // Release any reservations
      for (std::vector<Reservation>::const_iterator it =
            reservations.begin(); it != reservations.end(); it++)
        Runtime::release_reservation(*it, result);
      if (trace_info.recording)
        trace_info.record_issue_copy(result, this, src_fields, 
                                     dst_fields, reservations,
#ifdef LEGION_SPY
                                     src_tree_id, dst_tree_id,
#endif
                         precondition, pred_guard);
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists())
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      assert(trace_info.op != NULL);
      LegionSpy::log_copy_events(trace_info.op->get_unique_op_id(), 
          expr_id, src_tree_id, dst_tree_id, precondition, result);
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        LegionSpy::log_copy_field(result, src_fields[idx].field_id,
                                  src_fields[idx].inst_event,
                                  dst_fields[idx].field_id,
                                  dst_fields[idx].inst_event,
				  dst_fields[idx].redop_id);
#endif
      return result;
    }

    template <typename T, typename T2>
    inline T round_up(T val, T2 step)
    {
      T rem = val % step;
      if(rem == 0)
        return val;
      else
        return val + (step - rem);
    }

    template <typename T>
    inline T max(T a, T b)
    {
      return((a > b) ? a : b);
    }

    template <typename T>
    inline T gcd(T a, T b)
    {
      while(a != b) {
        if(a > b)
          a -= b;
        else
          b -= a;
      }
      return a;
    }

    template <typename T>
    inline T lcm(T a, T b)
    {
      // TODO: more efficient way?
      return(a * b / gcd(a, b));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceExpression::create_layout_internal(
                                 const Realm::IndexSpace<DIM,T> &space,
                                 const LayoutConstraintSet &constraints,
                                 const std::vector<FieldID> &field_ids,
                                 const std::vector<size_t> &field_sizes,
                                 bool compact, LayoutConstraintKind *unsat_kind,
                                 unsigned *unsat_index, void **piece_list,
                                 size_t *piece_list_size) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_ids.size() == field_sizes.size());
      assert(int(constraints.ordering_constraint.ordering.size()) == (DIM+1));
#endif
      Realm::InstanceLayout<DIM,T> *layout = new Realm::InstanceLayout<DIM,T>();
      layout->bytes_used = 0;
      // Start with 32-byte alignment for AVX instructions
      layout->alignment_reqd = 32;
      layout->space = space;
      std::vector<Rect<DIM,T> > piece_bounds;
      if (space.dense() || !compact)
      {
        if (!space.bounds.empty())
          piece_bounds.push_back(space.bounds);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(piece_list != NULL);
        assert((*piece_list) == NULL);
        assert(piece_list_size != NULL);
        assert((*piece_list_size) == 0);
#endif
        const SpecializedConstraint &spec = constraints.specialized_constraint;
        if (spec.max_overhead > 0)
        {
          std::vector<Realm::Rect<DIM,T> > covering;
          if (!space.compute_covering(spec.max_pieces, spec.max_overhead,
                                      covering))
          {
            if (unsat_kind != NULL)
              *unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
            if (unsat_index != NULL)
              *unsat_index = 0;
            return NULL;
          }
          // Container problem is stupid
          piece_bounds.resize(covering.size());
          for (unsigned idx = 0; idx < covering.size(); idx++)
            piece_bounds[idx] = covering[idx];
        }
        else
        {
          for (Realm::IndexSpaceIterator<DIM,T> itr(space); 
                itr.valid; itr.step())
            if (!itr.rect.empty())
              piece_bounds.push_back(itr.rect);
          if (spec.max_pieces < piece_bounds.size())
          {
            if (unsat_kind != NULL)
              *unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
            if (unsat_index != NULL)
              *unsat_index = 0;
            return NULL;
          }
        }
        if (!piece_bounds.empty())
        {
          *piece_list_size = piece_bounds.size() * sizeof(Rect<DIM,T>);
          *piece_list = malloc(*piece_list_size);
          Rect<DIM,T> *pieces = static_cast<Rect<DIM,T>*>(*piece_list);
          for (unsigned idx = 0; idx < piece_bounds.size(); idx++)
            pieces[idx] = piece_bounds[idx];
        }
      }

      // If the bounds are empty we can use the same piece list for all fields
      if (piece_bounds.empty())
      {
        layout->piece_lists.resize(1);
        for (unsigned idx = 0; idx < field_ids.size(); idx++)
        {
          const FieldID fid = field_ids[idx];
          Realm::InstanceLayoutGeneric::FieldLayout &fl = layout->fields[fid];
          fl.list_idx = 0;
          fl.rel_offset = 0;
          fl.size_in_bytes = field_sizes[idx];
        }
        return layout;
      }
      const OrderingConstraint &order = constraints.ordering_constraint;  
#ifdef DEBUG_LEGION
      assert(order.ordering.size() == (DIM+1));
#endif
      // Check if it is safe to re-use piece lists
      // It's only safe if fsize describes the size of a piece, which
      // is true if we only have a single piece or we're doing AOS
      const bool safe_reuse = 
       ((piece_bounds.size() == 1) || (order.ordering.front() == LEGION_DIM_F));
      // Get any alignment and offset constraints for individual fields
      std::map<FieldID,size_t> alignments;
      for (std::vector<AlignmentConstraint>::const_iterator it = 
            constraints.alignment_constraints.begin(); it !=
            constraints.alignment_constraints.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->eqk == LEGION_EQ_EK);
#endif
        alignments[it->fid] = it->alignment;
      }
      std::map<FieldID,off_t> offsets;
      for (std::vector<OffsetConstraint>::const_iterator it = 
            constraints.offset_constraints.begin(); it !=
            constraints.offset_constraints.end(); it++)
        offsets[it->fid] = it->offset;
      // Zip the fields with their sizes and sort them if we're allowed to
      std::set<size_t> unique_sizes;
      std::vector<std::pair<size_t,FieldID> > zip_fields;
      for (unsigned idx = 0; idx < field_ids.size(); idx++)
      {
        zip_fields.push_back(
            std::pair<size_t,FieldID>(field_sizes[idx], field_ids[idx]));
        if (safe_reuse)
          unique_sizes.insert(field_sizes[idx]);
      }
      if (!constraints.field_constraint.inorder)
      {
        // Sort them so the smallest fields are first
        std::stable_sort(zip_fields.begin(), zip_fields.end());
        // Reverse them so the biggest fields are first
        std::reverse(zip_fields.begin(), zip_fields.end());
        // Then reverse the field IDs back for the same size fields
        std::vector<std::pair<size_t,FieldID> >::iterator it1 = 
          zip_fields.begin(); 
        while (it1 != zip_fields.end())
        {
          std::vector<std::pair<size_t,FieldID> >::iterator it2 = it1;
          while ((it2 != zip_fields.end()) && (it1->first == it2->first))
            it2++;
          std::reverse(it1, it2);
          it1 = it2;
        }
      } 
      // Single affine piece or AOS on all pieces 
      // In this case we know fsize and falign are the same for
      // each of the pieces
      int field_index = -1;
      std::vector<size_t> elements_between_per_piece(piece_bounds.size(), 1);
      for (unsigned idx = 0; order.ordering.size(); idx++)
      {
        const DimensionKind dim = order.ordering[idx];
        if (dim == LEGION_DIM_F)
        {
          field_index = idx;
          break;
        }
#ifdef DEBUG_LEGION
        assert(int(dim) < DIM);
#endif
        for (unsigned pidx = 0; pidx < piece_bounds.size(); pidx++)
        {
          const Rect<DIM,T> &bounds = piece_bounds[pidx];
          elements_between_per_piece[pidx] *=
            (bounds.hi[dim] - bounds.lo[dim] + 1);
        }
      }
#ifdef DEBUG_LEGION
      assert(field_index >= 0);
#endif
      size_t elements_between_fields = elements_between_per_piece.front();
      for (unsigned idx = 1; idx < elements_between_per_piece.size(); idx++)
        elements_between_fields += elements_between_per_piece[idx];
      // This code borrows from choose_instance_layout but
      // there are subtle differences to handle Legion's layout constraints
      // What we want to compute is the size of the field dimension
      // in a way that guarantees that all fields maintain their alignments
      size_t fsize = 0;
      size_t falign = 1;
      // We can't make the piece lists yet because we don't know the 
      // extent of the field dimension needed to ensure alignment 
      std::map<FieldID, size_t> field_offsets;
      for (std::vector<std::pair<size_t,FieldID> >::const_iterator it = 
            zip_fields.begin(); it != zip_fields.end(); it++)
      {
        // if not specified, field goes at the end of all known fields
        // (or a bit past if alignment is a concern)
        size_t offset = fsize;
        std::map<FieldID,off_t>::const_iterator offset_finder = 
          offsets.find(it->second);
        if (offset_finder != offsets.end())
          offset += offset_finder->second;
        std::map<FieldID,size_t>::const_iterator alignment_finder = 
          alignments.find(it->second);
        const size_t field_alignment = (alignment_finder != alignments.end())
          ? alignment_finder->second : 1;
        if (field_alignment > 1)
        {
          offset = round_up(offset, field_alignment);
          if ((falign % field_alignment) != 0)
            falign = lcm(falign, field_alignment);
        }
        // increase size and alignment if needed
        fsize = max(fsize, offset + it->first * elements_between_fields);
        field_offsets[it->second] = offset;
      }
      if (falign > 1)
      {
        // group size needs to be rounded up to match group alignment
        fsize = round_up(fsize, falign);
        // overall instance alignment layout must be compatible with group
        layout->alignment_reqd = lcm(layout->alignment_reqd, falign);
      } 
      // compute the starting offsets for each piece
      std::vector<size_t> piece_offsets(piece_bounds.size());
      if (safe_reuse)
      {
        for (unsigned pidx = 0; pidx < piece_bounds.size(); pidx++)
        {
          const Rect<DIM,T> &bounds = piece_bounds[pidx];
          piece_offsets[pidx] = round_up(layout->bytes_used, falign);
          size_t piece_size = fsize;
          for (unsigned idx = field_index+1; idx < order.ordering.size(); idx++)
          {
            const DimensionKind dim = order.ordering[idx];
#ifdef DEBUG_LEGION
            assert(int(dim) < DIM);
#endif
            piece_size *= (bounds.hi[dim] - bounds.lo[dim] + 1);
          }
          layout->bytes_used = piece_offsets[pidx] + piece_size;
        }
      }
      // we've handled the offsets and alignment for every field across
      // all dimensions so we can just use the size of the field to 
      // determine the piece list
      std::map<size_t,unsigned> pl_indexes;
      layout->piece_lists.reserve(safe_reuse ? 
          unique_sizes.size() : zip_fields.size());
      for (std::vector<std::pair<size_t,FieldID> >::const_iterator it = 
            zip_fields.begin(); it != zip_fields.end(); it++)
      {
        unsigned li;
        std::map<size_t,unsigned>::const_iterator finder =
          safe_reuse ? pl_indexes.find(it->first) : pl_indexes.end();
        if (finder == pl_indexes.end())
        {
          li = layout->piece_lists.size();
#ifdef DEBUG_LEGION
          assert(li < (safe_reuse ? unique_sizes.size() : zip_fields.size()));
#endif
          layout->piece_lists.resize(li + 1);
          pl_indexes[it->first] = li;

          // create the piece list
          Realm::InstancePieceList<DIM,T>& pl = layout->piece_lists[li];
          pl.pieces.reserve(piece_bounds.size());

          size_t next_piece = safe_reuse ? 0 : field_offsets[it->second];
          for (unsigned pidx = 0; pidx < piece_bounds.size(); pidx++)
          {
            const Rect<DIM,T> &bounds = piece_bounds[pidx];
            // create the piece
            Realm::AffineLayoutPiece<DIM,T> *piece = 
              new Realm::AffineLayoutPiece<DIM,T>;
            piece->bounds = bounds; 
            size_t piece_start;
            if (safe_reuse)
              piece_start = piece_offsets[pidx];
            else
              piece_start = next_piece;
            piece->offset = piece_start;
            size_t stride = it->first;
            for (std::vector<DimensionKind>::const_iterator dit = 
                  order.ordering.begin(); dit != order.ordering.end(); dit++)
            {
              if ((*dit) != LEGION_DIM_F)
              {
#ifdef DEBUG_LEGION
                assert(int(*dit) < DIM);
#endif
                piece->strides[*dit] = stride;
                piece->offset -= bounds.lo[*dit] * stride;
                stride *= (bounds.hi[*dit] - bounds.lo[*dit] + 1);
              }
              else
              {
                // Update the location for the next piece to start
                if (!safe_reuse)
                  next_piece = piece_start + stride;
                // Reset the stride to the fsize for the next dimension
                // since it already incorporates everything prior to it
                stride = fsize;
              }
            }
            // Update the total bytes used for the last piece
            if (!safe_reuse && ((pidx+1) == piece_bounds.size()))
              layout->bytes_used = piece_start + stride;
            pl.pieces.push_back(piece);
          }
        }
        else
          li = finder->second;
#ifdef DEBUG_LEGION
        assert(layout->fields.count(it->second) == 0);
#endif
        Realm::InstanceLayoutGeneric::FieldLayout &fl = 
          layout->fields[it->second];
        fl.list_idx = li;
        fl.rel_offset = safe_reuse ? field_offsets[it->second] : 0;
        fl.size_in_bytes = it->first;
      }
#if 0
      // We have a two different implementations of how to compute the layout 
      // 1. In cases where we either have just one piece or we know we're
      //    doing AOS then we know the alignment and fsize will be the same
      //    across all the pieces, therefore we can share piece lists
      // 2. In more general cases, we can have SOA or hybrid with multiple
      //    pieces so we can't deduplicate piece lists safely given Realm's
      //    current encoding so instead we build a piece list per field and
      //    build up the representation for each piece individually

      // This is formerly case 2 that would lay out SOA and hybrid
      // layouts for each piece individually rather than grouping
      // pieces together for each field
      {
        // We have multiple pieces and we're not AOS so the per-field
        // offsets can be different in each piece dependent upon the
        // size of the piece and which dimensions are before the fields
        // In this case we're going to have one piece list for each field
        for (unsigned idx = 0; idx < zip_fields.size(); idx++)
        {
          layout->piece_lists[idx].pieces.reserve(piece_bounds.size());
          const std::pair<size_t,FieldID> &field = zip_fields[idx];
          Realm::InstanceLayoutGeneric::FieldLayout &fl = 
            layout->fields[field.second];
          fl.list_idx = idx;
          fl.rel_offset = 0;
          fl.size_in_bytes = field.first;
        }
        // We'll compute each piece infidivudaly and the piece list per field
        for (typename std::vector<Rect<DIM,T> >::const_iterator pit =
              piece_bounds.begin(); pit != piece_bounds.end(); pit++)
        {
          const Rect<DIM,T> &bounds = *pit;
          int field_index = -1;
          size_t elements_between_fields = 1;
          for (unsigned idx = 0; order.ordering.size(); idx++)
          {
            const DimensionKind dim = order.ordering[idx];
            if (dim == DIM_F)
            {
              field_index = idx;
              break;
            }
#ifdef DEBUG_LEGION
            assert(int(dim) < DIM);
#endif
            elements_between_fields *= (bounds.hi[dim] - bounds.lo[dim] + 1);
          }
#ifdef DEBUG_LEGION
          assert(field_index >= 0);
#endif
          // This code borrows from choose_instance_layout but
          // there are subtle differences to handle Legion's layout constraints
          // What we want to compute is the size of the field dimension
          // in a way that guarantees that all fields maintain their alignments
          size_t fsize = 0;
          size_t falign = 1;
          // We can't make the piece lists yet because we don't know the 
          // extent of the field dimension needed to ensure alignment 
          std::map<FieldID, size_t> field_offsets;
          for (std::vector<std::pair<size_t,FieldID> >::const_iterator it = 
                zip_fields.begin(); it != zip_fields.end(); it++)
          {
            // if not specified, field goes at the end of all known fields
            // (or a bit past if alignment is a concern)
            size_t offset = fsize;
            std::map<FieldID,off_t>::const_iterator offset_finder = 
              offsets.find(it->second);
            if (offset_finder != offsets.end())
              offset += offset_finder->second;
            std::map<FieldID,size_t>::const_iterator alignment_finder = 
              alignments.find(it->second);
            const size_t field_alignment = 
              (alignment_finder != alignments.end()) ? 
                alignment_finder->second : 1;
            if (field_alignment > 1)
            {
              offset = round_up(offset, field_alignment);
              if ((falign % field_alignment) != 0)
                falign = lcm(falign, field_alignment);
            }
            // increase size and alignment if needed
            fsize = max(fsize, offset + it->first * elements_between_fields);
            field_offsets[it->second] = offset;
          }
          if (falign > 1)
          {
            // group size needs to be rounded up to match group alignment
            fsize = round_up(fsize, falign);
            // overall instance alignment layout must be compatible with group
            layout->alignment_reqd = lcm(layout->alignment_reqd, falign);
          }
          // starting point for piece is first aligned location above
          // existing pieces
          const size_t piece_start = round_up(layout->bytes_used, falign);
          unsigned fidx = 0;
          for (std::vector<std::pair<size_t,FieldID> >::const_iterator it = 
                zip_fields.begin(); it != zip_fields.end(); it++, fidx++)
          {
            // create the piece
            Realm::AffineLayoutPiece<DIM,T> *piece = 
              new Realm::AffineLayoutPiece<DIM,T>;
            piece->bounds = bounds; 
            piece->offset = piece_start + field_offsets[it->second];
            size_t stride = it->first;
            for (std::vector<DimensionKind>::const_iterator dit = 
                  order.ordering.begin(); dit != order.ordering.end(); dit++)
            {
              if ((*dit) != DIM_F)
              {
#ifdef DEBUG_LEGION
                assert(int(*dit) < DIM);
#endif
                piece->strides[*dit] = stride;
                piece->offset -= bounds.lo[*dit] * stride;
                stride *= (bounds.hi[*dit] - bounds.lo[*dit] + 1);
              }
              else
                // Reset the stride to the fsize for the next dimension
                // since it already incorporates everything prior to it
                stride = fsize;
            }
            layout->piece_lists[fidx].pieces.push_back(piece);
          }
          // Lastly we need to update the bytes used for this piece
          size_t piece_bytes = fsize;
          for (unsigned idx = field_index+1; idx < order.ordering.size(); idx++)
          {
            const DimensionKind dim = order.ordering[idx];
#ifdef DEBUG_LEGION
            assert(int(dim) < DIM);
#endif
            piece_bytes *= (bounds.hi[dim] - bounds.lo[dim] + 1);
          }
          layout->bytes_used = piece_start + piece_bytes;
        }
      }
#endif
      return layout;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceExpression* 
            IndexSpaceExpression::create_layout_expression_internal(
                                     RegionTreeForest *context,
                                     const Realm::IndexSpace<DIM,T> &space,
                                     const Rect<DIM,T> *rects, size_t num_rects)
    //--------------------------------------------------------------------------
    {
      if (rects == NULL)
      {
        if (space.dense())
          return this;
        else
          // Make a new expression for the bounding box
          return new InstanceExpression<DIM,T>(&space.bounds,1/*size*/,context);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(num_rects > 0);
#endif
        // Make a realm expression from the rectangles
        return new InstanceExpression<DIM,T>(rects, num_rects, context);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline bool IndexSpaceExpression::meets_layout_expression_internal(
                          IndexSpaceExpression *space_expr, bool tight_bounds,
                          const Rect<DIM,T> *piece_list, size_t piece_list_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(type_tag == space_expr->type_tag);
#endif
      // See if this an convex hull or a piece list case
      if (piece_list == NULL)
      {
        // Get the bounds for each of them, can ignore ready events
        // since we're just going to be looking at the bounds
        Realm::IndexSpace<DIM,T> local, other;
        get_expr_index_space(&local, type_tag, true/*tight*/);
        space_expr->get_expr_index_space(&other, type_tag, true/*tight*/);
        // Check to see if we contain the space expression
        if (!local.bounds.contains(other.bounds))
          return false;
        // If tight, check to see if they are equivalent
        if (tight_bounds)
          return local.bounds == other.bounds;
        return true;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(piece_list_size > 0);
#endif
        // Iterate the rectangles in the space expr over the piece list
        // and compute the intersection volume summary
        // Note that this assumes that the rectangles in the piece list
        // are all non-overlapping with each other
        Realm::IndexSpace<DIM,T> other;
        const ApEvent ready =
          space_expr->get_expr_index_space(&other, type_tag, true/*tight*/);
        if (ready.exists() && !ready.has_triggered_faultignorant())
          ready.wait_faultignorant();
        size_t space_volume = 0; 
        size_t overlap_volume = 0;
        for (Realm::IndexSpaceIterator<DIM,T> itr(other); itr.valid; itr.step())
        {
          size_t local_volume = itr.rect.volume();
          space_volume += local_volume;
          for (unsigned idx = 0; idx < piece_list_size; idx++)
          {
            const Rect<DIM,T> overlap = piece_list[idx].intersection(itr.rect);
            size_t volume = overlap.volume();
            if (volume == 0)
              continue;
            overlap_volume += volume;
            local_volume -= volume;
            if (local_volume == 0)
              break;
          }
        }
#ifdef DEBUG_LEGION
        assert(overlap_volume <= space_volume);
#endif
        // If we didn't cover all the points in the space then we can't meet
        if (overlap_volume < space_volume)
          return false;
        if (tight_bounds)
        {
          // Check the total volume of all the pieces
          size_t piece_volume = 0;
          for (unsigned idx = 0; idx < piece_list_size; idx++)
            piece_volume += piece_list[idx].volume();
#ifdef DEBUG_LEGION
          assert(space_volume <= piece_volume);
#endif
          // Only meets if they have exactly the same points
          return (space_volume == piece_volume);
        }
        return true;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceExpression*
              IndexSpaceExpression::find_congruent_expression_internal(
                                   std::set<IndexSpaceExpression*> &expressions)
    //--------------------------------------------------------------------------
    {
      if (expressions.empty())
      {
        expressions.insert(this);
        return this;
      }
      Realm::IndexSpace<DIM,T> local_space;
      // No need to wait for the event, we know it is already triggered
      // because we called get_volume on this before we got here
      get_expr_index_space(&local_space, type_tag, true/*need tight result*/);
      const DistributedID local_did = get_distributed_id();
      for (std::set<IndexSpaceExpression*>::const_iterator it =
            expressions.begin(); it != expressions.end(); it++)
      {
        // We can get duplicates here
        if ((*it) == this)
          return this;
        Realm::IndexSpace<DIM,T> other_space;
        // No need to wait for the event here either, we know that if it is
        // in the 'expressions' data structure then wait has already been
        // called on it as well.
        (*it)->get_expr_index_space(&other_space, type_tag, true/*need tight*/);
        // See if the rectangles are the same
        if (local_space.bounds != other_space.bounds)
          continue;
        // See if the sparsity maps are the same
        if (local_space.sparsity == other_space.sparsity)
        {
          // We know that things are the same here
          // Try to add the expression reference, we can race with deletions
          // here though so handle the case we're we can't add a reference
          if ((*it)->try_add_canonical_reference(local_did))
            return (*it);
          else
            continue;
        }
        if (!local_space.sparsity.exists() || !other_space.sparsity.exists())
        {
          // Realm guarantees that tightening will remove a sparsity map if it
          // can so if one index space has a sparsity map and the other doesn't
          // then by definition they cannot be congruent (see issue #1020)
#ifdef DEBUG_LEGION
          // Should never hit this assertion as they should have equal sparsity
          // map IDs if the sparsity map does not exist for both of them
          assert(local_space.sparsity.exists() ||
                  other_space.sparsity.exists());
#endif
          continue;
        }
        else
        {
          // Both have sparsity maps
          // We know something important though here: we know that both
          // these sparsity maps contain the same number of points
          // Build lists of both sets of rectangles
          std::vector<Rect<DIM,T> > local_rects, other_rects;
          for (Realm::IndexSpaceIterator<DIM,T> itr(local_space);
                itr.valid; itr.step())
            local_rects.push_back(itr.rect);
          for (Realm::IndexSpaceIterator<DIM,T> itr(other_space);
                itr.valid; itr.step())
            other_rects.push_back(itr.rect);
          // We'll assume that the vector with more rectangles has
          // smaller ones and will therefore test the small ones
          // against the bigger ones, we might consider putting in
          // an acceleration data strucutre here if these vectors
          // are big enough
          if (local_rects.size() <= other_rects.size())
          {
            std::vector<size_t> remaining_local(local_rects.size());
            for (unsigned idx = 0; idx < local_rects.size(); idx++)
              remaining_local[idx] = local_rects[idx].volume();
            bool congruent = true;
            for (typename std::vector<Rect<DIM,T> >::const_iterator it =
                  other_rects.begin(); it != other_rects.end(); it++)
            {
              size_t remaining = it->volume();
              for (unsigned idx = 0; idx < local_rects.size(); idx++)
              {
                // Can skip local rects that are fully covered
                if (remaining_local[idx] == 0)
                  continue;
                const Rect<DIM,T> overlap = it->intersection(local_rects[idx]);
                const size_t volume = overlap.volume();
                if (volume == 0)
                  continue;
#ifdef DEBUG_LEGION
                assert(volume <= remaining);
                assert(volume <= remaining_local[idx]);
#endif
                remaining_local[idx] -= volume;
                remaining -= volume;
                if (remaining == 0)
                  break;
              }
              if (remaining != 0)
              {
                congruent = false;
                break;
              }
            }
            if (!congruent)
              continue;
#ifdef DEBUG_LEGION
            for (unsigned idx = 0; idx < local_rects.size(); idx++)
              assert(remaining_local[idx] == 0);
#endif
          }
          else
          {
            std::vector<size_t> remaining_other(other_rects.size());
            for (unsigned idx = 0; idx < other_rects.size(); idx++)
              remaining_other[idx] = other_rects[idx].volume();
            bool congruent = true;
            for (typename std::vector<Rect<DIM,T> >::const_iterator it =
                  local_rects.begin(); it != local_rects.end(); it++)
            {
              size_t remaining = it->volume();
              for (unsigned idx = 0; idx < other_rects.size(); idx++)
              {
                // Can skip local rects that are fully covered
                if (remaining_other[idx] == 0)
                  continue;
                const Rect<DIM,T> overlap = it->intersection(other_rects[idx]);
                const size_t volume = overlap.volume();
                if (volume == 0)
                  continue;
#ifdef DEBUG_LEGION
                assert(volume <= remaining);
                assert(volume <= remaining_other[idx]);
#endif
                remaining_other[idx] -= volume;
                remaining -= volume;
                if (remaining == 0)
                  break;
              }
              if (remaining != 0)
              {
                congruent = false;
                break;
              }
            }
            if (!congruent)
              continue;
#ifdef DEBUG_LEGION
            for (unsigned idx = 0; idx < other_rects.size(); idx++)
              assert(remaining_other[idx] == 0);
#endif
          }
        }
        // If we get here that means we are congruent
        // Try to add the expression reference, we can race with deletions
        // here though so handle the case we're we can't add a reference
        if ((*it)->try_add_canonical_reference(local_did))
          return (*it);
      }
      // Did not find any congruences so add ourself
      expressions.insert(this);
      return this;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Operations 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceOperationT<DIM,T>::IndexSpaceOperationT(OperationKind kind,
                                                      RegionTreeForest *ctx)
      : IndexSpaceOperation(NT_TemplateHelper::encode_tag<DIM,T>(),
                            kind, ctx), is_index_space_tight(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceOperationT<DIM,T>::IndexSpaceOperationT(RegionTreeForest *ctx, 
        IndexSpaceExprID eid, DistributedID did, AddressSpaceID owner,
        IndexSpaceOperation *origin, TypeTag tag, Deserializer &derez)
      : IndexSpaceOperation(tag, ctx, eid, did, owner, origin),
        is_index_space_tight(false)
    //--------------------------------------------------------------------------
    {
      // We can unpack the index space here directly
      derez.deserialize(this->realm_index_space);
      this->tight_index_space = this->realm_index_space;
      derez.deserialize(this->realm_index_space_ready);
      // Request that we make the valid index space valid
      this->tight_index_space_ready = 
        RtEvent(this->realm_index_space.make_valid());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceOperationT<DIM,T>::~IndexSpaceOperationT(void)
    //--------------------------------------------------------------------------
    {
      if (this->owner_space == this->context->runtime->address_space)
      {
        this->realm_index_space.destroy(realm_index_space_ready);
        this->tight_index_space.destroy(tight_index_space_ready);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM,T>::get_expr_index_space(void *result,
                                            TypeTag tag, bool need_tight_result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tag == type_tag);
#endif
      Realm::IndexSpace<DIM,T> *space = NULL;
      static_assert(sizeof(space) == sizeof(result), "Fuck c++");
      memcpy(&space, &result, sizeof(space));
      return get_realm_index_space(*space, need_tight_result);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Domain IndexSpaceOperationT<DIM,T>::get_domain(ApEvent &ready, bool tight)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> result;
      ready = get_realm_index_space(result, tight);
      return DomainT<DIM,T>(result);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM,T>::get_realm_index_space(
                        Realm::IndexSpace<DIM,T> &space, bool need_tight_result)
    //--------------------------------------------------------------------------
    {
      if (!is_index_space_tight)
      {
        if (need_tight_result)
        {
          // Wait for the index space to be tight
          if (tight_index_space_ready.exists() && 
              !tight_index_space_ready.has_triggered())
            tight_index_space_ready.wait();
          space = tight_index_space;
          return ApEvent::NO_AP_EVENT;
        }
        else
        {
          space = realm_index_space;
          return realm_index_space_ready;
        }
      }
      else
      {
        // Already tight so we can just return that
        space = tight_index_space;
        return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM,T>::tighten_index_space(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(realm_index_space.is_valid());
#endif
      tight_index_space = realm_index_space.tighten();
#ifdef DEBUG_LEGION
      assert(tight_index_space.is_valid());
#endif
      // Small memory fence to propagate writes before setting the flag
      __sync_synchronize();
      is_index_space_tight = true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceOperationT<DIM,T>::check_empty(void)
    //--------------------------------------------------------------------------
    {
      return (get_volume() == 0);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceOperationT<DIM,T>::get_volume(void)
    //--------------------------------------------------------------------------
    {
      if (has_volume)
        return volume;
      Realm::IndexSpace<DIM,T> temp;
      ApEvent ready = get_realm_index_space(temp, true/*tight*/);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      volume = temp.volume();
      __sync_synchronize();
      has_volume = true;
      return volume;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM,T>::pack_expression(Serializer &rez,
                                                      AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->is_valid());
#endif
      if (target == this->local_space)
      {
        rez.serialize<bool>(true/*local*/);
        rez.serialize(this);
        this->add_base_expression_reference(LIVE_EXPR_REF);
      }
      else if (target == this->owner_space)
      {
        rez.serialize<bool>(true/*local*/);
        rez.serialize(origin_expr);
        // Add a reference here that we'll remove after we've added a reference
        // on the target space expression
        this->add_base_expression_reference(REMOTE_DID_REF);
      }
      else
      {
        rez.serialize<bool>(false/*local*/);
        rez.serialize<bool>(false/*index space*/);
        rez.serialize(expr_id);
        rez.serialize(origin_expr);
        // Add a reference here that we'll remove after we've added a reference
        // on the target space expression
        this->add_base_expression_reference(REMOTE_DID_REF);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNode* IndexSpaceOperationT<DIM,T>::create_node(IndexSpace handle,
                             DistributedID did, RtEvent initialized,
                             Provenance *provenance, std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inter_lock, 1, false/*exclusive*/);
      if (is_index_space_tight)
        return context->create_node(handle, &tight_index_space, false/*domain*/,
                                    NULL/*parent*/, 0/*color*/, did, 
                                    initialized, provenance,
                                    realm_index_space_ready, expr_id, applied);
      else
        return context->create_node(handle, &realm_index_space, false/*domain*/,
                                    NULL/*parent*/, 0/*color*/, did,
                                    initialized, provenance,
                                    realm_index_space_ready, expr_id, applied);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PieceIteratorImpl* IndexSpaceOperationT<DIM,T>::create_piece_iterator(
      const void *piece_list, size_t piece_list_size, IndexSpaceNode *priv_node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      IndexSpaceNodeT<DIM,T> *privilege_node = 
        dynamic_cast<IndexSpaceNodeT<DIM,T>*>(priv_node);
      assert((privilege_node != NULL) || (priv_node == NULL));
#else
      IndexSpaceNodeT<DIM,T> *privilege_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(priv_node);
#endif
      if (piece_list == NULL)
      {
        Realm::IndexSpace<DIM,T> realm_space;
        get_realm_index_space(realm_space, true/*tight*/);
#ifdef DEBUG_LEGION
        // If there was no piece list it has to be because there
        // was just one piece which was a single dense rectangle
        assert(realm_space.dense());
#endif
        return new PieceIteratorImplT<DIM,T>(&realm_space.bounds,
                      sizeof(realm_space.bounds), privilege_node);
      }
      else
        return new PieceIteratorImplT<DIM,T>(piece_list, piece_list_size, 
                                             privilege_node);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM,T>::issue_fill(
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                 UniqueID fill_uid,
                                 FieldSpace handle,
                                 RegionTreeID tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (space_ready.exists() && precondition.exists())
        return issue_fill_internal(context, local_space, trace_info, 
            dst_fields, fill_value, fill_size, 
#ifdef LEGION_SPY
            fill_uid, handle, tree_id,
#endif
            Runtime::merge_events(&trace_info, space_ready, precondition),
            pred_guard);
      else if (space_ready.exists())
        return issue_fill_internal(context, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   space_ready, pred_guard);
      else
        return issue_fill_internal(context, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   precondition, pred_guard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM,T>::issue_copy(
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const std::vector<CopySrcDstField> &src_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (space_ready.exists() && precondition.exists())
        return issue_copy_internal(context, local_space, trace_info, 
            dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
            src_tree_id, dst_tree_id,
#endif
            Runtime::merge_events(&trace_info, precondition, space_ready),
            pred_guard);
      else if (space_ready.exists())
        return issue_copy_internal(context, local_space, trace_info, 
                dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                space_ready, pred_guard);
      else
        return issue_copy_internal(context, local_space, trace_info, 
                dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                precondition, pred_guard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructured* 
      IndexSpaceOperationT<DIM,T>::create_across_unstructured(
                                 const std::map<Reservation,bool> &reservations,
                                 const bool compute_preimages)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      return new CopyAcrossUnstructuredT<DIM,T>(context->runtime, this,
                   local_space, space_ready, reservations, compute_preimages);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceOperationT<DIM,T>::create_layout(
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<FieldID> &field_ids,
                                    const std::vector<size_t> &field_sizes,
                                    bool compact, 
                                    LayoutConstraintKind *unsat_kind,
                                    unsigned *unsat_index, void **piece_list,
                                    size_t *piece_list_size)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_is;
      ApEvent space_ready = get_realm_index_space(local_is, true/*tight*/);
      if (space_ready.exists())
        space_ready.wait();
      return create_layout_internal(local_is, constraints,field_ids,field_sizes,
                 compact, unsat_kind, unsat_index, piece_list, piece_list_size);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceOperationT<DIM,T>::create_layout_expression(
                                 const void *piece_list, size_t piece_list_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((piece_list_size % sizeof(Rect<DIM,T>)) == 0);
#endif
      Realm::IndexSpace<DIM,T> local_is;
      get_realm_index_space(local_is, true/*tight*/);
      // No need to wait for the index space to be ready since we
      // are never actually going to look at the sparsity map
      return create_layout_expression_internal(context, local_is,
                      static_cast<const Rect<DIM,T>*>(piece_list),
                      piece_list_size / sizeof(Rect<DIM,T>));
    }
    
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceOperationT<DIM,T>::meets_layout_expression(
                            IndexSpaceExpression *space_expr, bool tight_bounds,
                            const void *piece_list, size_t piece_list_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((piece_list_size % sizeof(Rect<DIM,T>)) == 0);
#endif
      return meets_layout_expression_internal<DIM,T>(space_expr, tight_bounds,
                                  static_cast<const Rect<DIM,T>*>(piece_list),
                                  piece_list_size / sizeof(Rect<DIM,T>));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* 
      IndexSpaceOperationT<DIM,T>::find_congruent_expression(
                                   std::set<IndexSpaceExpression*> &expressions)
    //--------------------------------------------------------------------------
    {
      return find_congruent_expression_internal<DIM,T>(expressions); 
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceUnion<DIM,T>::IndexSpaceUnion(
                            const std::vector<IndexSpaceExpression*> &to_union,
                            RegionTreeForest *ctx)
      : IndexSpaceOperationT<DIM,T>(IndexSpaceOperation::UNION_OP_KIND, ctx),
        sub_expressions(to_union)
    //--------------------------------------------------------------------------
    {
      // Add an resource ref that will be removed by the OperationCreator
      this->add_base_resource_ref(REGION_TREE_REF);
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM,T> > spaces(sub_expressions.size());
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
      {
        IndexSpaceExpression *sub = sub_expressions[idx];
#ifdef DEBUG_LEGION
        assert(sub->get_canonical_expression(this->context) == sub);
#endif
        // Add the parent and the reference
        sub->add_derived_operation(this);
        sub->add_tree_expression_reference(this->did);
        // Then get the realm index space expression
        ApEvent precondition = sub->get_expr_index_space(
            &spaces[idx], this->type_tag, false/*need tight result*/);
        if (precondition.exists())
          preconditions.insert(precondition);
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      Realm::ProfilingRequestSet requests;
      if (ctx->runtime->profiler != NULL)
        ctx->runtime->profiler->add_partition_request(requests,
                      implicit_provenance, DEP_PART_UNION_REDUCTION);
      this->realm_index_space_ready = ApEvent(
          Realm::IndexSpace<DIM,T>::compute_union(
              spaces, this->realm_index_space, requests, precondition));
      // Then launch the tighten call for it too since we know we're
      // going to want this eventually
      const RtEvent valid_event(this->realm_index_space.make_valid());
      // See if both the events needed for the tighten call are done
      if (!this->realm_index_space_ready.has_triggered() || 
          !valid_event.has_triggered())
      {
        IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
        if (!this->realm_index_space_ready.has_triggered())
        {
          if (!valid_event.has_triggered())
            this->tight_index_space_ready = 
              ctx->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_WORK_PRIORITY, Runtime::merge_events(valid_event,
                    Runtime::protect_event(this->realm_index_space_ready)));
          else
            this->tight_index_space_ready = 
              ctx->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_WORK_PRIORITY,
                  Runtime::protect_event(this->realm_index_space_ready));
        }
        else
          this->tight_index_space_ready = 
            ctx->runtime->issue_runtime_meta_task(args, 
                LG_LATENCY_WORK_PRIORITY, valid_event);
      }
      else // We can do the tighten call now
        this->tighten_index_space();
      if (ctx->runtime->legion_spy_enabled)
      {
        std::vector<IndexSpaceExprID> sources(this->sub_expressions.size()); 
        for (unsigned idx = 0; idx < this->sub_expressions.size(); idx++)
          sources[idx] = this->sub_expressions[idx]->expr_id;
        LegionSpy::log_index_space_union(this->expr_id, sources);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceUnion<DIM,T>::IndexSpaceUnion(const IndexSpaceUnion<DIM,T> &rhs)
      : IndexSpaceOperationT<DIM,T>(IndexSpaceOperation::UNION_OP_KIND, NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceUnion<DIM,T>::~IndexSpaceUnion(void)
    //--------------------------------------------------------------------------
    {
      // Remove references from our sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        if (sub_expressions[idx]->remove_tree_expression_reference(this->did))
          delete sub_expressions[idx];
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceUnion<DIM,T>& IndexSpaceUnion<DIM,T>::operator=(
                                              const IndexSpaceUnion<DIM,T> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceUnion<DIM,T>::pack_expression_value(Serializer &rez,
                                                       AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target != this->context->runtime->address_space);
#endif
      this->update_remote_instances(target);
      rez.serialize<bool>(false); // not an index space
      rez.serialize(this->type_tag); // unpacked by creator
      rez.serialize(this->expr_id); // unpacked by IndexSpaceOperation
      rez.serialize(this->did); // unpacked by IndexSpaceOperation
      rez.serialize(this->owner_space); // unpacked by IndexSpaceOperation
      rez.serialize(this->origin_expr); // unpacked by IndexSpaceOperation
      // unpacked by IndexSpaceOperationT
      Realm::IndexSpace<DIM,T> temp;
      ApEvent ready = this->get_realm_index_space(temp, true/*tight*/);
      rez.serialize(temp);
      rez.serialize(ready);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceUnion<DIM,T>::invalidate_operation(void)
    //--------------------------------------------------------------------------
    {
      // Make sure we only do this one time
      if (this->invalidated.fetch_add(1) > 0)
        return false;
      // Remove the parent operation from all the sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        sub_expressions[idx]->remove_derived_operation(this);
      // We were successfully removed
      return true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceUnion<DIM,T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      this->context->remove_union_operation(this, sub_expressions);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceIntersection<DIM,T>::IndexSpaceIntersection(
                            const std::vector<IndexSpaceExpression*> &to_inter,
                            RegionTreeForest *ctx)
      : IndexSpaceOperationT<DIM,T>(IndexSpaceOperation::INTERSECT_OP_KIND,ctx),
        sub_expressions(to_inter)
    //--------------------------------------------------------------------------
    {
      // Add an resource ref that will be removed by the OperationCreator
      this->add_base_resource_ref(REGION_TREE_REF);
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM,T> > spaces(sub_expressions.size());
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
      {
        IndexSpaceExpression *sub = sub_expressions[idx];
#ifdef DEBUG_LEGION
        assert(sub->get_canonical_expression(this->context) == sub);
#endif
        // Add the parent and the reference
        sub->add_derived_operation(this);
        sub->add_tree_expression_reference(this->did);
        ApEvent precondition = sub->get_expr_index_space(
            &spaces[idx], this->type_tag, false/*need tight result*/);
        if (precondition.exists())
          preconditions.insert(precondition);
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      Realm::ProfilingRequestSet requests;
      if (ctx->runtime->profiler != NULL)
        ctx->runtime->profiler->add_partition_request(requests,
                implicit_provenance, DEP_PART_INTERSECTION_REDUCTION);
      this->realm_index_space_ready = ApEvent(
          Realm::IndexSpace<DIM,T>::compute_intersection(
              spaces, this->realm_index_space, requests, precondition));
      // Then launch the tighten call for it too since we know we're
      // going to want this eventually
      const RtEvent valid_event(this->realm_index_space.make_valid());
      // See if both the events needed for the tighten call are done
      if (!this->realm_index_space_ready.has_triggered() || 
          !valid_event.has_triggered())
      {
        IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
        if (!this->realm_index_space_ready.has_triggered())
        {
          if (!valid_event.has_triggered())
            this->tight_index_space_ready = 
              ctx->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_WORK_PRIORITY, Runtime::merge_events(valid_event,
                    Runtime::protect_event(this->realm_index_space_ready)));
          else
            this->tight_index_space_ready = 
              ctx->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_WORK_PRIORITY,
                  Runtime::protect_event(this->realm_index_space_ready));
        }
        else
          this->tight_index_space_ready = 
            ctx->runtime->issue_runtime_meta_task(args, 
                LG_LATENCY_WORK_PRIORITY, valid_event);
      }
      else // We can do the tighten call now
        this->tighten_index_space();
      if (ctx->runtime->legion_spy_enabled)
      {
        std::vector<IndexSpaceExprID> sources(this->sub_expressions.size()); 
        for (unsigned idx = 0; idx < this->sub_expressions.size(); idx++)
          sources[idx] = this->sub_expressions[idx]->expr_id;
        LegionSpy::log_index_space_intersection(this->expr_id, sources);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceIntersection<DIM,T>::IndexSpaceIntersection(
                                      const IndexSpaceIntersection<DIM,T> &rhs)
      : IndexSpaceOperationT<DIM,T>(IndexSpaceOperation::INTERSECT_OP_KIND,NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceIntersection<DIM,T>::~IndexSpaceIntersection(void)
    //--------------------------------------------------------------------------
    {
      // Remove references from our sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        if (sub_expressions[idx]->remove_tree_expression_reference(this->did))
          delete sub_expressions[idx];
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceIntersection<DIM,T>& IndexSpaceIntersection<DIM,T>::operator=(
                                       const IndexSpaceIntersection<DIM,T> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceIntersection<DIM,T>::pack_expression_value(Serializer &rez, 
                                                          AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target != this->context->runtime->address_space);
#endif
      this->update_remote_instances(target);
      rez.serialize<bool>(false); // not an index space
      rez.serialize(this->type_tag); // unpacked by creator
      rez.serialize(this->expr_id); // unpacked by IndexSpaceOperation
      rez.serialize(this->did); // unpacked by IndexSpaceOperation
      rez.serialize(this->owner_space); // unpacked by IndexSpaceOperation
      rez.serialize(this->origin_expr); // unpacked by IndexSpaceOperation
      // unpacked by IndexSpaceOperationT
      Realm::IndexSpace<DIM,T> temp;
      ApEvent ready = this->get_realm_index_space(temp, true/*tight*/);
      rez.serialize(temp);
      rez.serialize(ready);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceIntersection<DIM,T>::invalidate_operation(void)
    //--------------------------------------------------------------------------
    {
      // Make sure we only do this one time
      if (this->invalidated.fetch_add(1) > 0)
        return false;
      // Remove the parent operation from all the sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        sub_expressions[idx]->remove_derived_operation(this);
      // We were successfully removed
      return true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceIntersection<DIM,T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      this->context->remove_intersection_operation(this, sub_expressions);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceDifference<DIM,T>::IndexSpaceDifference(IndexSpaceExpression *l,
                IndexSpaceExpression *r, RegionTreeForest *ctx) 
      : IndexSpaceOperationT<DIM,T>(IndexSpaceOperation::DIFFERENCE_OP_KIND,ctx)
        , lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
      // Add an resource ref that will be removed by the OperationCreator
      this->add_base_resource_ref(REGION_TREE_REF);
#ifdef DEBUG_LEGION
      assert(lhs->get_canonical_expression(this->context) == lhs);
      assert(rhs->get_canonical_expression(this->context) == rhs);
#endif
      if (lhs == rhs)
      {
        // Special case for when the expressions are the same
        lhs->add_derived_operation(this);
        lhs->add_tree_expression_reference(this->did);
        this->realm_index_space = Realm::IndexSpace<DIM,T>::make_empty();
        this->tight_index_space = Realm::IndexSpace<DIM,T>::make_empty();
        this->realm_index_space_ready = ApEvent::NO_AP_EVENT;
        this->tight_index_space_ready = RtEvent::NO_RT_EVENT;
      }
      else
      {
        Realm::IndexSpace<DIM,T> lhs_space, rhs_space;
        // Add the parent and the references
        lhs->add_derived_operation(this);
        rhs->add_derived_operation(this);
        lhs->add_tree_expression_reference(this->did);
        rhs->add_tree_expression_reference(this->did);
        ApEvent left_ready = 
          lhs->get_expr_index_space(&lhs_space, this->type_tag, false/*tight*/);
        ApEvent right_ready = 
          rhs->get_expr_index_space(&rhs_space, this->type_tag, false/*tight*/);
        ApEvent precondition = 
          Runtime::merge_events(NULL, left_ready, right_ready);
        Realm::ProfilingRequestSet requests;
        if (ctx->runtime->profiler != NULL)
          ctx->runtime->profiler->add_partition_request(requests,
                                implicit_provenance, DEP_PART_DIFFERENCE);
        this->realm_index_space_ready = ApEvent(
            Realm::IndexSpace<DIM,T>::compute_difference(lhs_space, rhs_space, 
                              this->realm_index_space, requests, precondition));
        // Then launch the tighten call for it too since we know we're
        // going to want this eventually
        const RtEvent valid_event(this->realm_index_space.make_valid());
        // See if both the events needed for the tighten call are done
        if (!this->realm_index_space_ready.has_triggered() || 
            !valid_event.has_triggered())
        {
          IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
          if (!this->realm_index_space_ready.has_triggered())
          {
            if (!valid_event.has_triggered())
              this->tight_index_space_ready = 
                ctx->runtime->issue_runtime_meta_task(args, 
                    LG_LATENCY_WORK_PRIORITY, Runtime::merge_events(valid_event,
                      Runtime::protect_event(this->realm_index_space_ready)));
            else
              this->tight_index_space_ready = 
                ctx->runtime->issue_runtime_meta_task(args, 
                    LG_LATENCY_WORK_PRIORITY,
                    Runtime::protect_event(this->realm_index_space_ready));
          }
          else
            this->tight_index_space_ready = 
              ctx->runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_WORK_PRIORITY, valid_event);
        }
        else // We can do the tighten call now
          this->tighten_index_space();
      }
      if (ctx->runtime->legion_spy_enabled)
        LegionSpy::log_index_space_difference(this->expr_id,
                                              lhs->expr_id, rhs->expr_id);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceDifference<DIM,T>::IndexSpaceDifference(
                                      const IndexSpaceDifference<DIM,T> &rhs)
     : IndexSpaceOperationT<DIM,T>(IndexSpaceOperation::DIFFERENCE_OP_KIND,
                                   NULL), lhs(NULL), rhs(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceDifference<DIM,T>::~IndexSpaceDifference(void)
    //--------------------------------------------------------------------------
    {
      if ((rhs != NULL) && (lhs != rhs) && 
          rhs->remove_tree_expression_reference(this->did))
        delete rhs;
      if ((lhs != NULL) && lhs->remove_tree_expression_reference(this->did))
        delete lhs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceDifference<DIM,T>& IndexSpaceDifference<DIM,T>::operator=(
                                         const IndexSpaceDifference<DIM,T> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceDifference<DIM,T>::pack_expression_value(Serializer &rez,
                                                          AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target != this->context->runtime->address_space);
#endif
      this->update_remote_instances(target);
      rez.serialize<bool>(false); // not an index space
      rez.serialize(this->type_tag); // unpacked by creator
      rez.serialize(this->expr_id); // unpacked by IndexSpaceOperation
      rez.serialize(this->did); // unpacked by IndexSpaceOperation
      rez.serialize(this->owner_space); // unpacked by IndexSpaceOperation
      rez.serialize(this->origin_expr); // unpacked by IndexSpaceOperation
      // unpacked by IndexSpaceOperationT
      Realm::IndexSpace<DIM,T> temp;
      ApEvent ready = this->get_realm_index_space(temp, true/*tight*/);
      rez.serialize(temp);
      rez.serialize(ready);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceDifference<DIM,T>::invalidate_operation(void)
    //--------------------------------------------------------------------------
    {
      // Make sure we only do this one time
      if (this->invalidated.fetch_add(1) > 0)
        return false;
      // Remove the parent operation from all the sub expressions
      if (lhs != NULL)
        lhs->remove_derived_operation(this);
      if ((rhs != NULL) && (lhs != rhs))
        rhs->remove_derived_operation(this);
      // We were successfully removed
      return true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceDifference<DIM,T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
       if ((lhs != NULL) && (rhs != NULL))
        this->context->remove_subtraction_operation(this, lhs, rhs);
    }

    /////////////////////////////////////////////////////////////
    // Instance Expression
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    InstanceExpression<DIM,T>::InstanceExpression(
           const Rect<DIM,T> *rects, size_t num_rects, RegionTreeForest *forest)
      : IndexSpaceOperationT<DIM,T>(
          IndexSpaceOperation::INSTANCE_EXPRESSION_KIND, forest)
    //--------------------------------------------------------------------------
    {
      // This is another kind of live expression made by the region tree
      this->add_base_expression_reference(LIVE_EXPR_REF);
      if (implicit_reference_tracker == NULL)
        implicit_reference_tracker = new ImplicitReferenceTracker;
      implicit_reference_tracker->record_live_expression(this);
#ifdef DEBUG_LEGION
      assert(num_rects > 0);
#endif
      if (num_rects > 1)
      {
        std::vector<Realm::Rect<DIM,T> > realm_rects(num_rects);
        for (unsigned idx = 0; idx < num_rects; idx++)
          realm_rects[idx] = rects[idx];
        this->realm_index_space = Realm::IndexSpace<DIM,T>(realm_rects); 
        const RtEvent valid_event(this->realm_index_space.make_valid());
        if (!valid_event.has_triggered())
        {
          IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
          this->tight_index_space_ready = 
            forest->runtime->issue_runtime_meta_task(args, 
                LG_LATENCY_WORK_PRIORITY, valid_event);
        }
        else // We can do the tighten call now
          this->tighten_index_space();
      }
      else
      {
        this->realm_index_space.bounds = rects[0];
        this->realm_index_space.sparsity.id = 0;
        this->tight_index_space = this->realm_index_space;
        this->is_index_space_tight = true;
      }
      if (forest->runtime->legion_spy_enabled)
      {
        // These index expressions cannot be computed, so we'll pretend
        // like they are index spaces to Legion Spy since these are 
        // effectively new "atom" index spaces for Legion Spy's analysis
        const IndexSpaceID fake_space_id = 
          forest->runtime->get_unique_index_space_id();
        LegionSpy::log_top_index_space(fake_space_id,
            forest->runtime->address_space, NULL/*provenance*/);
        LegionSpy::log_index_space_expr(fake_space_id, this->expr_id);
        bool all_empty = true;
        for (unsigned idx = 0; idx < num_rects; idx++)
        {
          const size_t volume = rects[idx].volume();
          if (volume == 0)
            continue;
          if (volume == 1)
            LegionSpy::log_index_space_point(fake_space_id, rects[idx].lo);
          else
            LegionSpy::log_index_space_rect(fake_space_id, rects[idx]);
          all_empty = false;
        }
        if (all_empty)
          LegionSpy::log_empty_index_space(fake_space_id);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    InstanceExpression<DIM,T>::InstanceExpression(
                                           const InstanceExpression<DIM,T> &rhs)
      : IndexSpaceOperationT<DIM,T>(
          IndexSpaceOperation::INSTANCE_EXPRESSION_KIND, NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    InstanceExpression<DIM,T>::~InstanceExpression(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    InstanceExpression<DIM,T>& InstanceExpression<DIM,T>::operator=(
                                           const InstanceExpression<DIM,T> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void InstanceExpression<DIM,T>::pack_expression_value(Serializer &rez,
                                                          AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target != this->context->runtime->address_space);
#endif
      this->update_remote_instances(target);
      rez.serialize<bool>(false); // not an index space
      rez.serialize(this->type_tag); // unpacked by creator
      rez.serialize(this->expr_id); // unpacked by IndexSpaceOperation
      rez.serialize(this->did); // unpacked by IndexSpaceOperation
      rez.serialize(this->owner_space); // unpacked by IndexSpaceOperation
      rez.serialize(this->origin_expr); // unpacked by IndexSpaceOperation
      // unpacked by IndexSpaceOperationT
      Realm::IndexSpace<DIM,T> temp;
      ApEvent ready = this->get_realm_index_space(temp, true/*tight*/);
      rez.serialize(temp);
      rez.serialize(ready);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool InstanceExpression<DIM,T>::invalidate_operation(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void InstanceExpression<DIM,T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here since we're not in the region tree
    }

    /////////////////////////////////////////////////////////////
    // Remote Expression
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RemoteExpression<DIM,T>::RemoteExpression(RegionTreeForest *forest,
        IndexSpaceExprID eid, DistributedID did, AddressSpaceID owner,
        IndexSpaceOperation *origin, TypeTag tag, Deserializer &derez)
      : IndexSpaceOperationT<DIM,T>(forest, eid, did, owner, origin, tag, derez)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RemoteExpression<DIM,T>::RemoteExpression(const RemoteExpression<DIM,T> &rs)
      : IndexSpaceOperationT<DIM,T>(
          IndexSpaceOperation::REMOTE_EXPRESSION_KIND, rs.context)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RemoteExpression<DIM,T>::~RemoteExpression(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RemoteExpression<DIM,T>& RemoteExpression<DIM,T>::operator=(
                                             const RemoteExpression<DIM,T> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void RemoteExpression<DIM,T>::pack_expression_value(Serializer &rez,
                                                        AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool RemoteExpression<DIM,T>::invalidate_operation(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void RemoteExpression<DIM,T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      // nothing to do here
    }

    /////////////////////////////////////////////////////////////
    // Templated Index Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM,T>::IndexSpaceNodeT(RegionTreeForest *ctx, 
        IndexSpace handle, IndexPartNode *parent, LegionColor color,
        const void *bounds, bool is_domain, DistributedID did, ApEvent ready, 
        IndexSpaceExprID expr_id, RtEvent init, unsigned dep, Provenance *prov)
      : IndexSpaceNode(ctx, handle, parent, color, did, ready, expr_id, init, 
          dep, prov), linearization_ready(false)
    //--------------------------------------------------------------------------
    {
      if (bounds != NULL)
      {
        if (is_domain)
        {
          const DomainT<DIM,T> temp_space = *static_cast<const Domain*>(bounds);
          realm_index_space = temp_space;
        }
        else
          realm_index_space = 
            *static_cast<const Realm::IndexSpace<DIM,T>*>(bounds);
        Runtime::trigger_event(realm_index_space_set);
      }
      else
        add_base_resource_ref(RUNTIME_REF);
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
      if (is_owner())
        realm_index_space.destroy(
            tight_index_space ? tight_index_space_set : realm_index_space_set);
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
    ApEvent IndexSpaceNodeT<DIM,T>::get_realm_index_space(
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
    bool IndexSpaceNodeT<DIM,T>::set_realm_index_space(
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
        // Hold the lock while walking over the node set
        AutoLock n_lock(node_lock);
        // Now we can trigger the event while holding the lock
        Runtime::trigger_event(realm_index_space_set);
        if (has_remote_instances())
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
          map_over_remote_instances(functor);
        }
      }
      // Now we can tighten it
      tighten_index_space();
      // Remove the reference we were holding until this was set
      return remove_base_resource_ref(RUNTIME_REF);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::get_expr_index_space(void *result,
                                            TypeTag tag, bool need_tight_result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(type_tag == handle.get_type_tag());
#endif
      Realm::IndexSpace<DIM,T> *space = NULL;
      static_assert(sizeof(space) == sizeof(result), "Fuck c++");
      memcpy(&space, &result, sizeof(space));
      return get_realm_index_space(*space, need_tight_result);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Domain IndexSpaceNodeT<DIM,T>::get_domain(ApEvent &ready, bool need_tight)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> result;
      ready = get_realm_index_space(result, need_tight);
      return DomainT<DIM,T>(result);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::set_domain(const Domain &domain, 
                                            AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      const DomainT<DIM,T> realm_space = domain;
      return set_realm_index_space(source, realm_space);
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
      const RtEvent valid_event(realm_index_space.make_valid());
      if (!index_space_ready.has_triggered() || !valid_event.has_triggered())
      {
        // If this index space isn't ready yet, then we have to defer this 
        TightenIndexSpaceArgs args(this, this);
        if (!index_space_ready.has_triggered())
        {
          if (!valid_event.has_triggered())
            context->runtime->issue_runtime_meta_task(args,
                LG_LATENCY_WORK_PRIORITY, Runtime::merge_events(valid_event,
                  Runtime::protect_event(index_space_ready)));
          else
            context->runtime->issue_runtime_meta_task(args,
                LG_LATENCY_WORK_PRIORITY,
                Runtime::protect_event(index_space_ready));
        }
        else
          context->runtime->issue_runtime_meta_task(args,
              LG_LATENCY_WORK_PRIORITY, valid_event);
        
        return;
      }
#ifdef DEBUG_LEGION
      assert(realm_index_space.is_valid());
#endif
      Realm::IndexSpace<DIM,T> tight_space = realm_index_space.tighten();
#ifdef DEBUG_LEGION
      assert(tight_space.is_valid());
#endif
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
      if (context->runtime->legion_spy_enabled || 
          (context->runtime->profiler != NULL))
      {
        // Log subspaces being set on the owner
        const AddressSpaceID owner_space = get_owner_space();
        if (owner_space == context->runtime->address_space)
        {
          if (context->runtime->legion_spy_enabled)
            this->log_index_space_points(tight_space);
          if (context->runtime->profiler != NULL)
            this->log_profiler_index_space_points(tight_space);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::check_empty(void)
    //--------------------------------------------------------------------------
    {
      return (get_volume() == 0);
    } 

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNode* IndexSpaceNodeT<DIM,T>::create_node(IndexSpace new_handle,
                             DistributedID did, RtEvent initialized, 
                             Provenance *provenance, std::set<RtEvent> *applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.get_type_tag() == new_handle.get_type_tag());
#endif
      Realm::IndexSpace<DIM,T> local_space;
      const ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      return context->create_node(new_handle, &local_space, false/*domain*/,
                                  NULL/*parent*/, 0/*color*/, did, initialized,
                                  provenance, ready, expr_id, applied);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PieceIteratorImpl* IndexSpaceNodeT<DIM,T>::create_piece_iterator(
      const void *piece_list, size_t piece_list_size, IndexSpaceNode *priv_node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      IndexSpaceNodeT<DIM,T> *privilege_node = 
        dynamic_cast<IndexSpaceNodeT<DIM,T>*>(priv_node);
      assert((privilege_node != NULL) || (priv_node == NULL));
#else
      IndexSpaceNodeT<DIM,T> *privilege_node = 
        static_cast<IndexSpaceNodeT<DIM,T>*>(priv_node);
#endif
      if (piece_list == NULL)
      {
        Realm::IndexSpace<DIM,T> realm_space;
        get_realm_index_space(realm_space, true/*tight*/);
#ifdef DEBUG_LEGION
        // If there was no piece list it has to be because there
        // was just one piece which was a single dense rectangle
        assert(realm_space.dense());
#endif
        return new PieceIteratorImplT<DIM,T>(&realm_space.bounds,
                      sizeof(realm_space.bounds), privilege_node);
      }
      else
        return new PieceIteratorImplT<DIM,T>(piece_list, piece_list_size, 
                                             privilege_node);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_index_space_points(
                              const Realm::IndexSpace<DIM,T> &tight_space) const
    //--------------------------------------------------------------------------
    {
      // Be careful, Realm can lie to us here
      if (!tight_space.empty())
      {
        bool logged = false;
        // Iterate over the rectangles and print them out 
        for (Realm::IndexSpaceIterator<DIM,T> itr(tight_space); 
              itr.valid; itr.step())
        {
          const size_t rect_volume = itr.rect.volume();
          if (rect_volume == 0)
            continue;
          logged = true;
          if (rect_volume == 1)
            LegionSpy::log_index_space_point(handle.get_id(), 
                                             Point<DIM,T>(itr.rect.lo));
          else
            LegionSpy::log_index_space_rect(handle.get_id(), 
                                            Rect<DIM,T>(itr.rect));
        }
        // Handle the case where Realm lied to us about being empty
        if (!logged)
          LegionSpy::log_empty_index_space(handle.get_id());
      }
      else
        LegionSpy::log_empty_index_space(handle.get_id());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::log_profiler_index_space_points(
                              const Realm::IndexSpace<DIM,T> &tight_space) const
    //--------------------------------------------------------------------------
    {
      if (!tight_space.empty())
      {
        bool is_dense = tight_space.dense();
        size_t dense_volume, sparse_volume;
        if (is_dense)
          dense_volume = sparse_volume = tight_space.volume();
        else
          {
            dense_volume = tight_space.bounds.volume();
            sparse_volume = tight_space.volume();
          }
        context->runtime->profiler->record_index_space_size(
                          handle.get_id(), dense_volume, sparse_volume, !is_dense);
        // Iterate over the rectangles and print them out
        for (Realm::IndexSpaceIterator<DIM,T> itr(tight_space);
              itr.valid; itr.step())
        {
          if (itr.rect.volume() == 1)
            context->runtime->profiler->record_index_space_point(
                handle.get_id(), Point<DIM,T>(itr.rect.lo));
          else
            context->runtime->profiler->record_index_space_rect(
                handle.get_id(), Rect<DIM,T>(itr.rect));
        }
      }
      else
        context->runtime->profiler->record_empty_index_space(handle.get_id());
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      Realm::IndexSpace<DIM,T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_UNION_REDUCTION);
        ApEvent result(Realm::IndexSpace<DIM,T>::compute_union(
              spaces, result_space, requests, precondition));
        if (set_realm_index_space(context->runtime->address_space,result_space))
          assert(false); // should never hit this
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
        if (set_realm_index_space(context->runtime->address_space,result_space))
          assert(false); // should never hit this
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
          ApEvent ready = child->get_realm_index_space(spaces[subspace_index++],
                                                       false/*tight*/);
          if (ready.exists())
            preconditions.insert(ready);
        }
        delete itr;
      }
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      Realm::IndexSpace<DIM,T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_UNION_REDUCTION);
        ApEvent result(Realm::IndexSpace<DIM,T>::compute_union(
              spaces, result_space, requests, precondition));
        if (set_realm_index_space(context->runtime->address_space,result_space))
          assert(false); // should never hit this
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
        if (set_realm_index_space(context->runtime->address_space,result_space))
          assert(false); // should never hit this
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
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
            Runtime::merge_events(NULL, lhs_ready, rhs_ready)));
      if (set_realm_index_space(context->runtime->address_space, result_space))
        assert(false); // should never hit this
      // Destroy the tempory rhs space once the computation is done
      rhs_space.destroy(result);
      return result;
    } 

    // This is a small helper class for converting realm index spaces when
    // the types don't naturally align with the underlying index space type
    template<int DIM, typename TYPELIST>
    struct RealmSpaceConverter {
      static inline void convert_to(const Domain &domain, void *realm_is, 
                                    const TypeTag type_tag, const char *context)
      {
        // Compute the type tag for this particular type with the same DIM
        const TypeTag tag =
          NT_TemplateHelper::encode_tag<DIM,typename TYPELIST::HEAD>();
        if (tag == type_tag)
        {
          Realm::IndexSpace<DIM,typename TYPELIST::HEAD> *target =
            static_cast<Realm::IndexSpace<DIM,typename TYPELIST::HEAD>*>(
                                                                realm_is);
          *target = domain;
        }
        else
          RealmSpaceConverter<DIM,typename TYPELIST::TAIL>::convert_to(domain,
                                                  realm_is, type_tag, context);
      }
    };

    // Specialization for end-of-list cases
    template<int DIM>
    struct RealmSpaceConverter<DIM,Realm::DynamicTemplates::TypeListTerm> {
      static inline void convert_to(const Domain &domain, void *realm_is, 
                                    const TypeTag type_tag, const char *context)
      {
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
          "Dynamic type mismatch in '%s'", context)
      }
    };

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::get_index_space_domain(void *realm_is, 
                                                        TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (type_tag == handle.get_type_tag())
      {
        Realm::IndexSpace<DIM,T> *target = 
          static_cast<Realm::IndexSpace<DIM,T>*>(realm_is);
        // No need to wait since we're waiting for it to be tight
        // which implies that it will be ready
        get_realm_index_space(*target, true/*tight*/);
      }
      else
      {
        Realm::IndexSpace<DIM,T> target;
        // No need to wait since we're waiting for it to be tight
        // which implies that it will be ready
        get_realm_index_space(target, true/*tight*/);
        const Domain domain(target);
        RealmSpaceConverter<DIM,Realm::DIMTYPES>::convert_to(
                  domain, realm_is, type_tag, "get_index_space_domain");
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::get_volume(void)
    //--------------------------------------------------------------------------
    {
      if (has_volume)
        return volume;
      Realm::IndexSpace<DIM,T> volume_space;
      get_realm_index_space(volume_space, true/*tight*/);
      volume = volume_space.volume();
      __sync_synchronize();
      has_volume = true;
      return volume;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::get_num_dims(void) const
    //--------------------------------------------------------------------------
    {
      return DIM;
    }

    // This is a small helper class for converting realm points when the 
    // types don't naturally align with the underling index space type
    template<int DIM, typename TYPELIST>
    struct RealmPointConverter {
      // Convert To
      static inline void convert_to(const DomainPoint &point, void *realm_point,
                                    const TypeTag type_tag, const char *context)
      {
        // Compute the type tag for this particular type with the same DIM
        const TypeTag tag =
          NT_TemplateHelper::template encode_tag<DIM,typename TYPELIST::HEAD>();
        if (tag == type_tag)
        {
          Realm::Point<DIM,typename TYPELIST::HEAD> *target =
           static_cast<Realm::Point<DIM,typename TYPELIST::HEAD>*>(realm_point);
          *target = point;
        }
        else
          RealmPointConverter<DIM,typename TYPELIST::TAIL>::convert_to(point,
                                               realm_point, type_tag, context);
      } 
      // Convert From
      static inline void convert_from(const void *realm_point, TypeTag type_tag,
                                      DomainPoint &point, const char *context)
      {
        // Compute the type tag for this particular type with the same DIM
        const TypeTag tag =
          NT_TemplateHelper::encode_tag<DIM,typename TYPELIST::HEAD>();
        if (tag == type_tag)
        {
          const Realm::Point<DIM,typename TYPELIST::HEAD> *source =
           static_cast<const Realm::Point<DIM,typename TYPELIST::HEAD>*>(
                                                              realm_point);
          point = *source;
        }
        else
          RealmPointConverter<DIM,typename TYPELIST::TAIL>::convert_from(
                                    realm_point, type_tag, point, context);
      } 
    };

    // Specialization for the end-of-list cases
    template<int DIM>
    struct RealmPointConverter<DIM,Realm::DynamicTemplates::TypeListTerm> {
      static inline void convert_to(const DomainPoint &point, void *realm_point,
                                    const TypeTag type_tag, const char *context)
      {
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
          "Dynamic type mismatch in '%s'", context)
      }
      static inline void convert_from(const void *realm_point, TypeTag type_tag,
                                      DomainPoint &point, const char *context)
      {
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
          "Dynamic type mismatch in '%s'", context)
      }
    };

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_point(const void *realm_point, 
                                                TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> test_space;
      // Wait for a tight space on which to perform the test
      get_realm_index_space(test_space, true/*tight*/);
      if (type_tag == handle.get_type_tag())
      {
        const Realm::Point<DIM,T> *point = 
          static_cast<const Realm::Point<DIM,T>*>(realm_point);
        return test_space.contains(*point);
      }
      else
      {
        DomainPoint point;
        RealmPointConverter<DIM,Realm::DIMTYPES>::convert_from(
            realm_point, type_tag, point, "safe_cast");
        return test_space.contains(Point<DIM,T>(point));
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      const Point<DIM,T> p = point;
      return contains_point(p); 
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
      // Need a memory fence here to make sure that writes propagate on 
      // non-total-store-ordered memory consistency machines like PowerPC
      __sync_synchronize();
      linearization_ready = true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::linearize_color(const void *realm_color,
                                                        TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (!linearization_ready)
        compute_linearization_metadata();
      Realm::Point<DIM,T> point;
      if (type_tag != handle.get_type_tag())
      {
        DomainPoint dp;
        RealmPointConverter<DIM,Realm::DIMTYPES>::convert_from(
            realm_color, type_tag, dp, "linearize_color");
        point = dp;
      }
      else
        point = *(static_cast<const Realm::Point<DIM,T>*>(realm_color));
      // First subtract the offset to get to the origin
      point -= offset;
      LegionColor color = 0;
      for (int idx = 0; idx < DIM; idx++)
        color += point[idx] * strides[idx];
      return color;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::linearize_color(Point<DIM,T> point)
    //--------------------------------------------------------------------------
    {
      if (!linearization_ready)
        compute_linearization_metadata();
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
      if (!linearization_ready)
        compute_linearization_metadata();
      if (type_tag == handle.get_type_tag())
      {
        Realm::Point<DIM,T> &point = 
          *(static_cast<Realm::Point<DIM,T>*>(realm_color));
        for (int idx = DIM-1; idx >= 0; idx--)
        {
          point[idx] = color/strides[idx]; // truncates
          color -= point[idx] * strides[idx];
        }
        point += offset;
      }
      else
      {
        Realm::Point<DIM,T> point;
        for (int idx = DIM-1; idx >= 0; idx--)
        {
          point[idx] = color/strides[idx]; // truncates
          color -= point[idx] * strides[idx];
        }
        point += offset;
        RealmPointConverter<DIM,Realm::DIMTYPES>::convert_to(
            DomainPoint(point), realm_color, type_tag, "delinearize_color");
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceIterator* 
                       IndexSpaceNodeT<DIM,T>::create_color_space_iterator(void)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> color_space;
      // Wait for a tight space on which to perform the test
      get_realm_index_space(color_space, true/*tight*/); 
      return new ColorSpaceIteratorT<DIM,T>(color_space, this);
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
    bool IndexSpaceNodeT<DIM,T>::unpack_index_space(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> result_space;
      derez.deserialize(result_space);
      return set_realm_index_space(source, result_space);
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
        ready = Runtime::merge_events(NULL, ready, 
                  op->get_execution_fence_event());
      ApEvent result(local_space.create_equal_subspaces(count, 
            granularity, subspaces, requests, ready));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == ready))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    expr_id, ready, result);
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color(); 
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
        delete itr;
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                              op, DEP_PART_UNIONS);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_unions(
            lhs_spaces, rhs_spaces, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    expr_id, precondition, result);
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
        delete itr;
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_INTERSECTIONS);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_intersections(
            lhs_spaces, rhs_spaces, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    expr_id, precondition, result);
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
        delete itr;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_intersection(Operation *op,
                                                      IndexPartNode *partition,
                                                      // Left is implicit "this"
                                                      IndexPartNode *right,
                                                      const bool dominates)
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
      }
      ApEvent result, precondition;
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      if (dominates)
      {
        // If we've been told that we dominate then there is no
        // need to event do the intersection tests at all
        subspaces.swap(rhs_spaces);
        result = Runtime::merge_events(NULL, preconditions);
      }
      else
      {
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
        precondition = Runtime::merge_events(NULL, preconditions);
        result = ApEvent(Realm::IndexSpace<DIM,T>::compute_intersections(
              lhs_space, rhs_spaces, subspaces, requests, precondition));  
      }
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    expr_id, precondition, result);
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
        delete itr;
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_DIFFERENCES);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(Realm::IndexSpace<DIM,T>::compute_differences(
            lhs_spaces, rhs_spaces, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    expr_id, precondition, result);
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
      }
      else
      {
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
        delete itr;
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
#define DIMFUNC(D2) \
        case D2: \
          { \
            const Realm::Matrix<D2,DIM,T> *transform = \
              static_cast<const Realm::Matrix<D2,DIM,T>*>(tran); \
            const Realm::Rect<D2,T> *extent = \
              static_cast<const Realm::Rect<D2,T>*>(ext); \
            return create_by_restriction_helper<D2>(partition, \
                                                   *transform, *extent); \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
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
          if (child->set_realm_index_space(context->runtime->address_space, 
                                           child_is))
            assert(false); // should never hit this
        }
      }
      // Our only precondition is that the parent index space is computed
      return parent->index_space_ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_domain(Operation *op,
                                                    IndexPartNode *partition,
                                                    FutureMapImpl *future_map,
                                                    bool perform_intersections)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the color space type to do the actual operations 
      CreateByDomainHelper creator(this, partition, op, 
                                    future_map, perform_intersections);
      NT_TemplateHelper::demux<CreateByDomainHelper>(
                   partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_weights(Operation *op,
                                                    IndexPartNode *partition,
                                                    FutureMapImpl *future_map,
                                                    size_t granularity)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the color space type to do the actual operations 
      CreateByWeightHelper creator(this, partition, op, future_map,granularity);
      NT_TemplateHelper::demux<CreateByWeightHelper>(
                   partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
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
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T> template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_domain_helper(Operation *op,
                          IndexPartNode *partition, FutureMapImpl *future_map,
                          bool perform_intersections)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNodeT<COLOR_DIM,COLOR_T> *color_space = 
       static_cast<IndexSpaceNodeT<COLOR_DIM,COLOR_T>*>(partition->color_space);
      // Enumerate the color space
      Realm::IndexSpace<COLOR_DIM,COLOR_T> realm_color_space;
      color_space->get_realm_index_space(realm_color_space, true/*tight*/);

      std::set<ApEvent> result_events;
      Realm::IndexSpace<DIM,T> parent_space;
      ApEvent parent_ready;
      if (perform_intersections)
      {
        parent_ready = get_realm_index_space(parent_space, false/*tight*/);
        if (op->has_execution_fence_event())
        {
          if (parent_ready.exists())
            parent_ready = Runtime::merge_events(NULL, parent_ready,
                                    op->get_execution_fence_event());
          else
            parent_ready = op->get_execution_fence_event();
        }
      }
      // Make all the entries for the color space
      for (Realm::IndexSpaceIterator<COLOR_DIM,COLOR_T> 
            rect_iter(realm_color_space); rect_iter.valid; rect_iter.step())
      {
        for (Realm::PointInRectIterator<COLOR_DIM,COLOR_T> 
              itr(rect_iter.rect); itr.valid; itr.step())
        {
          LegionColor child_color = color_space->linearize_color(&itr.p,
                                        color_space->handle.get_type_tag());
          IndexSpaceNodeT<DIM,T> *child = static_cast<IndexSpaceNodeT<DIM,T>*>(
                                            partition->get_child(child_color));
          Realm::IndexSpace<DIM,T> child_space;
          const DomainPoint key(Point<COLOR_DIM,COLOR_T>(itr.p));
          FutureImpl *future = future_map->find_future(key);
          if (future != NULL)
          {
            if (future->get_untyped_size(true/*internal*/) != 
                  sizeof(Domain))
              REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_BY_DOMAIN_VALUE,
                  "An invalid future size was found in a partition by domain "
                  "call. All futures must contain Domain objects.")
            const Domain *domain = static_cast<Domain*>(
                future->get_untyped_result(true, NULL, true/*internal*/));
            const DomainT<DIM,T> domaint = *domain;
            child_space = domaint;
            if (perform_intersections)
            {
              Realm::ProfilingRequestSet requests;
              if (context->runtime->profiler != NULL)
                context->runtime->profiler->add_partition_request(requests,
                                                op, DEP_PART_INTERSECTIONS);
              Realm::IndexSpace<DIM,T> result;
              ApEvent ready(Realm::IndexSpace<DIM,T>::compute_intersection(
                    parent_space, child_space, result, requests, parent_ready));
              child_space = result;
              if (ready.exists())
                result_events.insert(ready);
            }
          }
          else
            child_space = Realm::IndexSpace<DIM,T>::make_empty();
          if (child->set_realm_index_space(context->runtime->address_space,
                                           child_space))
            assert(false); // should never hit this
        }
      }
      if (result_events.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(NULL, result_events);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T> template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_weight_helper(Operation *op,
        IndexPartNode *partition, FutureMapImpl *future_map, size_t granularity)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNodeT<COLOR_DIM,COLOR_T> *color_space = 
       static_cast<IndexSpaceNodeT<COLOR_DIM,COLOR_T>*>(partition->color_space);
      // Enumerate the color space
      Realm::IndexSpace<COLOR_DIM,COLOR_T> realm_color_space;
      color_space->get_realm_index_space(realm_color_space, true/*tight*/); 
      const size_t count = realm_color_space.volume();
      // Unpack the futures and fill in the weights appropriately
      std::vector<int> weights;
      std::vector<size_t> long_weights;
      std::vector<LegionColor> child_colors(count);
      unsigned color_index = 0;
      // Make all the entries for the color space
      for (Realm::IndexSpaceIterator<COLOR_DIM,COLOR_T> 
            rect_iter(realm_color_space); rect_iter.valid; rect_iter.step())
      {
        for (Realm::PointInRectIterator<COLOR_DIM,COLOR_T> 
              itr(rect_iter.rect); itr.valid; itr.step())
        {
          const DomainPoint key(Point<COLOR_DIM,COLOR_T>(itr.p));
          FutureImpl *future = future_map->find_future(key);
          if (future == NULL)
            REPORT_LEGION_ERROR(ERROR_MISSING_PARTITION_BY_WEIGHT_COLOR,
                "A partition by weight call is missing an entry for a "
                "color in the color space. All colors must be present.")
          const size_t future_size = future->get_untyped_size(true/*internal*/);
          if (future_size == sizeof(int))
          {
            if (weights.empty())
            {
              if (!long_weights.empty())
                REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_BY_WEIGHT_VALUE,
                  "An invalid future size was found in a partition by weight "
                  "call. All futures must be consistent int or size_t values.")
              weights.resize(count);
            }
            weights[color_index] = *(static_cast<int*>(
              future->get_untyped_result(true, NULL, true/*internal*/)));
          }
          else if (future_size == sizeof(size_t))
          {
            if (long_weights.empty())
            {
              if (!weights.empty())
                REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_BY_WEIGHT_VALUE,
                  "An invalid future size was found in a partition by weight "
                  "call. All futures must be consistent int or size_t values.")
              long_weights.resize(count);
            }
            long_weights[color_index] = *(static_cast<size_t*>(
              future->get_untyped_result(true, NULL, true/*internal*/)));
          }
          else
            REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_BY_WEIGHT_VALUE,
                  "An invalid future size was found in a partition by weight "
                  "call. All futures must contain int or size_t values.")
          child_colors[color_index++] = color_space->linearize_color(&itr.p,
                                          color_space->handle.get_type_tag());
        }
      }
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                                op, DEP_PART_WEIGHTS);
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (op->has_execution_fence_event())
        ready = Runtime::merge_events(NULL, ready, 
                  op->get_execution_fence_event());
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      ApEvent result(weights.empty() ?
          local_space.create_weighted_subspaces(count,
            granularity, long_weights, subspaces, requests, ready) :
          local_space.create_weighted_subspaces(count,
            granularity, weights, subspaces, requests, ready));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == ready))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(),
                                    expr_id, ready, result);
#endif
      for (unsigned idx = 0; idx < count; idx++)
      {
        IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(
                partition->get_child(child_colors[idx]));
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[idx]))
            assert(false); // should never hit this
      }
      return result;
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(local_space.create_subspaces_by_field(
            descriptors, colors, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
                                    precondition, result);
#endif
      // Update the children with the names of their subspaces 
      for (unsigned idx = 0; idx < colors.size(); idx++)
      {
        LegionColor child_color = color_space->linearize_color(&colors[idx],
                                        color_space->handle.get_type_tag());
        IndexSpaceNodeT<DIM,T> *child = static_cast<IndexSpaceNodeT<DIM,T>*>(
                                            partition->get_child(child_color));
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[idx]))
          assert(false); // should never hit this
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
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
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES    
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(local_space.create_subspaces_by_image(descriptors,
            sources, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[color]))
            assert(false); // should never hit this
        }
      }
      else
      {
        unsigned index = 0;
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[index++]))
            assert(false); // should never hit this
        }
        delete itr;
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
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
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        // Always use the partitions color space
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(local_space.create_subspaces_by_image(descriptors,
            sources, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[color]))
            assert(false); // should never hit this
        }
      }
      else
      {
        unsigned index = 0;
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[index++]))
            assert(false); // should never hit this
        }
        delete itr;
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
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
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        // Always use the partitions color space
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(local_space.create_subspaces_by_preimage(
            descriptors, targets, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[color]))
            assert(false); // should never hit this
        }
      }
      else
      {
        unsigned index = 0;
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[index++]))
            assert(false); // should never hit this
        }
        delete itr;
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
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
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
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
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
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
        delete itr;
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(local_space.create_subspaces_by_preimage(
            descriptors, targets, subspaces, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
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
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[color]))
            assert(false); // should never hit this
        }
      }
      else
      {
        unsigned index = 0;
        ColorSpaceIterator *itr = 
          partition->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          // Get the child of the projection partition
          IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(color));
#ifdef DEBUG_LEGION
          assert(index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[index++]))
            assert(false); // should never hit this
        }
        delete itr;
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
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
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
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
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      ApEvent result(local_space.create_association(descriptors,
            range_space, requests, precondition));
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
                                    precondition, result);
#endif
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::get_coordinate_size(bool range) const
    //--------------------------------------------------------------------------
    {
      if (range)
        return sizeof(Realm::Rect<DIM,T>);
      else
        return sizeof(Realm::Point<DIM,T>);
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
      Realm::InstanceLayoutConstraints ilc(field_ids, field_sizes, 0 /*SOA*/);
      int dim_order[DIM];
      for (int i = 0; i < DIM; i++)
	dim_order[i] = i;
      Realm::InstanceLayoutGeneric *ilg;
      ilg = Realm::InstanceLayoutGeneric::choose_instance_layout(local_space,
							       ilc, dim_order);

      Realm::ExternalFileResource res(file_name, file_mode);
      // No profiling for these kinds of instances currently
      Realm::ProfilingRequestSet requests;
      PhysicalInstance result;
      ready_event = ApEvent(PhysicalInstance::create_external_instance(result, 
          res.suggested_memory(), ilg, res, requests));
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_hdf5_instance(
                                    const char *file_name,
				    const std::vector<Realm::FieldID> &field_ids,
                                    const std::vector<size_t> &field_sizes,
                                    const std::vector<const char*> &field_files,
                                    const OrderingConstraint &dimension_order,
                                    bool read_only, ApEvent &ready_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, REALM_CREATE_INSTANCE_CALL);
#ifdef DEBUG_LEGION
      assert(int(dimension_order.ordering.size()) == (DIM+1));
      assert(dimension_order.ordering.back() == LEGION_DIM_F);
#endif
      // Have to wait for the index space to be ready if necessary
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      // No profiling for these kinds of instances currently
      Realm::ProfilingRequestSet requests;
      PhysicalInstance result = PhysicalInstance::NO_INST;

#ifdef LEGION_USE_HDF5
      Realm::InstanceLayout<DIM,T> *layout = new Realm::InstanceLayout<DIM,T>;
      layout->bytes_used = 0;
      layout->alignment_reqd = 0;  // no allocation being made
      layout->space = local_space;
      layout->piece_lists.resize(field_ids.size());
      for (size_t i = 0; i < field_ids.size(); i++)
      {
	Realm::InstanceLayoutGeneric::FieldLayout& fl =
	  layout->fields[field_ids[i]];
	fl.list_idx = i;
	fl.rel_offset = 0;
	fl.size_in_bytes = field_sizes[i];

	// create a single piece (for non-empty index spaces)
	if(!local_space.empty()) {
	  Realm::HDF5LayoutPiece<DIM,T> *hlp = new Realm::HDF5LayoutPiece<DIM,T>;
	  hlp->bounds = local_space.bounds;
	  hlp->dsetname = field_files[i];
	  for (int j = 0; j < DIM; j++)	    
	    hlp->offset[j] = 0;
	  // Legion ordering constraints are listed from fastest to 
	  // slowest like fortran order, hdf5 is the opposite though
	  // so we want to list dimensions in order from slowest to fastest
	  for (unsigned idx = 0; idx < DIM; idx++)
	    hlp->dim_order[idx] = dimension_order.ordering[DIM - 1 - idx];
	  layout->piece_lists[i].pieces.push_back(hlp);
	}
      }

      Realm::ExternalHDF5Resource res(file_name, read_only);
      ready_event = ApEvent(PhysicalInstance::create_external_instance(result,
		            res.suggested_memory(), layout, res, requests));
#else
      assert(false); // should never get here
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::issue_fill(
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                 UniqueID fill_uid,
                                 FieldSpace handle,
                                 RegionTreeID tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (precondition.exists() && space_ready.exists())
        return issue_fill_internal(context, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
            Runtime::merge_events(&trace_info, space_ready, precondition),
            pred_guard);
      else if (space_ready.exists())
        return issue_fill_internal(context, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   space_ready, pred_guard);
      else
        return issue_fill_internal(context, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   precondition, pred_guard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::issue_copy(
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const std::vector<CopySrcDstField> &src_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (precondition.exists() && space_ready.exists())
        return issue_copy_internal(context, local_space, trace_info, dst_fields,
            src_fields, reservations,
#ifdef LEGION_SPY
            src_tree_id, dst_tree_id,
#endif
            Runtime::merge_events(&trace_info, space_ready, precondition),
            pred_guard);
      else if (space_ready.exists())
        return issue_copy_internal(context, local_space, trace_info, 
                dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                space_ready, pred_guard);
      else
        return issue_copy_internal(context, local_space, trace_info, 
                dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                precondition, pred_guard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructured* IndexSpaceNodeT<DIM,T>::create_across_unstructured(
                                 const std::map<Reservation,bool> &reservations,
                                 const bool compute_preimages)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      return new CopyAcrossUnstructuredT<DIM,T>(context->runtime, this,
                   local_space, space_ready, reservations, compute_preimages);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceNodeT<DIM,T>::create_layout(
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<FieldID> &field_ids,
                                    const std::vector<size_t> &field_sizes,
                                    bool compact, 
                                    LayoutConstraintKind *unsat_kind,
                                    unsigned *unsat_index, void **piece_list, 
                                    size_t *piece_list_size)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_is;
      ApEvent space_ready = get_realm_index_space(local_is, true/*tight*/);
      if (space_ready.exists())
        space_ready.wait();
      return create_layout_internal(local_is, constraints,field_ids,field_sizes,
                 compact, unsat_kind, unsat_index, piece_list, piece_list_size);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM,T>::create_layout_expression(
                                 const void *piece_list, size_t piece_list_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((piece_list_size % sizeof(Rect<DIM,T>)) == 0);
#endif
      Realm::IndexSpace<DIM,T> local_is;
      get_realm_index_space(local_is, true/*tight*/);
      // No need to wait for the index space to be ready since we
      // are never actually going to look at the sparsity map
      return create_layout_expression_internal(context, local_is,
                      static_cast<const Rect<DIM,T>*>(piece_list),
                      piece_list_size / sizeof(Rect<DIM,T>));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::meets_layout_expression(
                            IndexSpaceExpression *space_expr, bool tight_bounds,
                            const void *piece_list, size_t piece_list_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((piece_list_size % sizeof(Rect<DIM,T>)) == 0);
#endif
      return meets_layout_expression_internal<DIM,T>(space_expr, tight_bounds,
                                  static_cast<const Rect<DIM,T>*>(piece_list),
                                  piece_list_size / sizeof(Rect<DIM,T>));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* 
            IndexSpaceNodeT<DIM,T>::find_congruent_expression(
                                   std::set<IndexSpaceExpression*> &expressions)
    //--------------------------------------------------------------------------
    {
      return find_congruent_expression_internal<DIM,T>(expressions); 
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
    // Templated Color Space Iterator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceIteratorT<DIM,T>::ColorSpaceIteratorT(const DomainT<DIM,T> &d,
                                                    IndexSpaceNodeT<DIM,T> *cs)
      : ColorSpaceIterator(), PointInDomainIterator<DIM,T>(d), color_space(cs)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool ColorSpaceIteratorT<DIM,T>::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      return this->valid();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceIteratorT<DIM,T>::yield_color(void)
    //--------------------------------------------------------------------------
    {
      const LegionColor result = 
        color_space->linearize_color(*(this->point_itr));
      this->step();
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Templated Copy Across 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructuredT<DIM,T>::CopyAcrossUnstructuredT(Runtime *rt,
                IndexSpaceExpression *e, const DomainT<DIM,T> &domain, 
                ApEvent ready, const std::map<Reservation,bool> &rsrvs,
                const bool preimages)
      : CopyAcrossUnstructured(rt, preimages, rsrvs), expr(e),
        copy_domain(domain), copy_domain_ready(ready), 
        need_src_indirect_precondition(true),
        need_dst_indirect_precondition(true), 
        src_indirect_immutable_for_tracing(false),
        dst_indirect_immutable_for_tracing(false), has_empty_preimages(false)
    //--------------------------------------------------------------------------
    {
      expr->add_base_expression_reference(COPY_ACROSS_REF);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructuredT<DIM,T>::~CopyAcrossUnstructuredT(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_base_expression_reference(COPY_ACROSS_REF))
        delete expr;
#ifdef DEBUG_LEGION
      assert(src_preimages.empty());
      assert(dst_preimages.empty());
#endif
      // Clean up any preimages that we computed
      for (typename std::vector<DomainT<DIM,T> >::iterator it =
            current_src_preimages.begin(); it != 
            current_src_preimages.end(); it++)
        it->destroy(last_copy);
      for (typename std::vector<DomainT<DIM,T> >::iterator it =
            current_dst_preimages.begin(); it != 
            current_dst_preimages.end(); it++)
        it->destroy(last_copy);
      for (typename std::vector<const CopyIndirection*>::const_iterator it =
            indirections.begin(); it != indirections.end(); it++)
        delete (*it);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent CopyAcrossUnstructuredT<DIM,T>::execute(Operation *op, 
          PredEvent pred_guard, ApEvent copy_precondition, 
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          const PhysicalTraceInfo &trace_info, const bool recurrent_replay,
          const unsigned stage)
    //--------------------------------------------------------------------------
    {
      if (stage == 0)
      {
        RtEvent src_preimages_ready, dst_preimages_ready;
        if (!src_indirections.empty() && compute_preimages &&
            (!src_indirect_immutable_for_tracing || !recurrent_replay))
        {
          // Compute new preimages and add the to the back of the queue
          ComputePreimagesHelper helper(this, op, 
              src_indirect_precondition, true/*source*/);
          NT_TemplateHelper::demux<ComputePreimagesHelper>(
              src_indirect_type, &helper);
          if (helper.result.exists())
            src_preimages_ready = Runtime::protect_event(helper.result);
          AutoLock p_lock(preimage_lock);
          src_preimages.emplace_back(helper.new_preimages);
#ifdef LEGION_SPY
          src_preimage_preconditions.emplace_back(helper.result);
#endif
        }
        if (!dst_indirections.empty() && compute_preimages &&
            (!dst_indirect_immutable_for_tracing || !recurrent_replay))
        {
          // Compute new preimages and add them to the back of the queue
          ComputePreimagesHelper helper(this, op, 
              dst_indirect_precondition, false/*source*/);
          NT_TemplateHelper::demux<ComputePreimagesHelper>(
              dst_indirect_type, &helper);
          if (helper.result.exists())
            dst_preimages_ready = Runtime::protect_event(helper.result);
          AutoLock p_lock(preimage_lock);
          dst_preimages.emplace_back(helper.new_preimages);
#ifdef LEGION_SPY
          dst_preimage_preconditions.emplace_back(helper.result);
#endif
        }
        // Make sure that all the stage 1's are ordered 
        // by deferring execution if necessary
        if ((prev_done.exists() && !prev_done.has_triggered()) ||
            (src_preimages_ready.exists() && 
             !src_preimages_ready.has_triggered()) ||
            (dst_preimages_ready.exists() &&
             !dst_preimages_ready.has_triggered()))
        {
          const RtEvent defer = Runtime::merge_events(prev_done, 
              src_preimages_ready, dst_preimages_ready);
          DeferCopyAcrossArgs args(this, op, pred_guard, copy_precondition,
              src_indirect_precondition, dst_indirect_precondition,
              trace_info, recurrent_replay, stage);
          prev_done = runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, defer);
          return args.done_event;
        }
      }
      // Need to rebuild indirections in the first time through or if we
      // are computing preimages and not doing a recurrent replay
      if (indirections.empty() || (!recurrent_replay && compute_preimages))
      {
#ifdef LEGION_SPY
        // Make a unique indirections identifier if necessary
        unique_indirections_identifier =
          runtime->get_unique_indirections_id();
#endif
        // No need for the lock here, we know we are ordered
        if (!indirections.empty())
        {
          for (typename std::vector<const CopyIndirection*>::const_iterator it =
                indirections.begin(); it != indirections.end(); it++)
            delete (*it);
          indirections.clear();
          individual_field_indexes.clear();
        }
        has_empty_preimages = false;
        // Prune preimages if necessary
        if (!src_indirections.empty())
        {
          if (!current_src_preimages.empty())
          {
            // Destroy any previous source preimage spaces
            for (typename std::vector<DomainT<DIM,T> >::iterator it =
                  current_src_preimages.begin(); it != 
                  current_src_preimages.end(); it++)
              it->destroy(last_copy);
          }
          if (compute_preimages)
          {
            // Get the next batch of src preimages to use
            AutoLock p_lock(preimage_lock);
#ifdef DEBUG_LEGION
            assert(!src_preimages.empty());
#endif
            current_src_preimages.swap(src_preimages.front());
            src_preimages.pop_front();
#ifdef LEGION_SPY
            assert(!src_preimage_preconditions.empty());
            current_src_preimage_precondition =
              src_preimage_preconditions.front();
            src_preimage_preconditions.pop_front();
#endif
          }
          RebuildIndirectionsHelper helper(this, true/*sources*/);
          NT_TemplateHelper::demux<RebuildIndirectionsHelper>(
              src_indirect_type, &helper);
          if (helper.empty)
            has_empty_preimages = true;
        }
        if (!dst_indirections.empty())
        {
          if (!current_dst_preimages.empty())
          {
            // Destroy any previous destination preimage spaces
            for (typename std::vector<DomainT<DIM,T> >::iterator it =
                  current_dst_preimages.begin(); it != 
                  current_dst_preimages.end(); it++)
              it->destroy(last_copy);
          }
          if (compute_preimages)
          {
            // Get the next batch of dst preimages to use
            AutoLock p_lock(preimage_lock);
#ifdef DEBUG_LEGION
            assert(!dst_preimages.empty());
#endif
            current_dst_preimages.swap(dst_preimages.front());
            dst_preimages.pop_front();
#ifdef LEGION_SPY
            assert(!dst_preimage_preconditions.empty());
            current_dst_preimage_precondition =
              dst_preimage_preconditions.front();
            dst_preimage_preconditions.pop_front();
#endif
          }
          RebuildIndirectionsHelper helper(this, false/*sources*/);
          NT_TemplateHelper::demux<RebuildIndirectionsHelper>(
              dst_indirect_type, &helper);
          if (helper.empty)
            has_empty_preimages = true;
        }
#ifdef LEGION_SPY
        // Have to convert back to Realm structures because C++ is dumb  
        realm_src_fields.resize(src_fields.size());
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          realm_src_fields[idx] = src_fields[idx];
        realm_dst_fields.resize(dst_fields.size());
        for (unsigned idx = 0; idx < dst_fields.size(); idx++)
          realm_dst_fields[idx] = dst_fields[idx];
        // This part isn't necessary for correctness but it helps Legion Spy
        // see the dependences between the preimages and copy operations
        if (current_src_preimage_precondition.exists() ||
            current_dst_preimage_precondition.exists())
          copy_precondition = Runtime::merge_events(NULL, copy_precondition,
              current_src_preimage_precondition,
              current_dst_preimage_precondition);
#endif
      }
      if (has_empty_preimages)
      {
#ifdef LEGION_SPY
        ApUserEvent new_last_copy = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_last_copy);
        last_copy = new_last_copy;
        LegionSpy::log_indirect_events(op->get_unique_op_id(), expr->expr_id,
                unique_indirections_identifier, copy_precondition, last_copy);
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          LegionSpy::log_indirect_field(last_copy, src_fields[idx].field_id,
                                        src_fields[idx].inst_event,
                                        src_fields[idx].indirect_index,
                                        dst_fields[idx].field_id,
                                        dst_fields[idx].inst_event, 
                                        dst_fields[idx].indirect_index,
                                        dst_fields[idx].redop_id);
        return last_copy;
#else
        return ApEvent::NO_AP_EVENT;
#endif
      }
#ifdef DEBUG_LEGION
      assert(src_fields.size() == dst_fields.size());
#endif
      // Now that we know we're going to do this copy add any profling requests
      Realm::ProfilingRequestSet requests;
      const unsigned total_copies =
        individual_field_indexes.empty() ? 1 : individual_field_indexes.size();
      if (op != NULL)
        op->add_copy_profiling_request(trace_info, requests,
                                       false/*fill*/, total_copies);
      if (runtime->profiler != NULL)
        runtime->profiler->add_copy_request(requests, op, total_copies);
      if (pred_guard.exists())
      {
        // No need for tracing to know about the precondition or reservations
        ApEvent pred_pre = 
          Runtime::merge_events(NULL, copy_precondition, ApEvent(pred_guard));
        if (!reservations.empty())
        {
          // Need a protected version here to guarantee we always acquire
          // or release the lock regardless of poison
          pred_pre = Runtime::ignorefaults(copy_precondition);
          for (std::map<Reservation,bool>::const_iterator it =
                reservations.begin(); it != reservations.end(); it++)
            pred_pre = 
              Runtime::acquire_ap_reservation(it->first, it->second, pred_pre);
          // Tricky: now merge the predicate and precondition back in to get the 
          // effects of any poison since we protected against it above
          // Note you can't wait to acquire events until you know the full
          // precondition has triggered or poisoned including the predicate
          // or you risk deadlock which is why we need the double merge
          pred_pre =
            Runtime::merge_events(NULL, pred_pre, ApEvent(pred_guard));
        }
#ifdef LEGION_SPY
        if (!indirections.empty())
          last_copy = Runtime::ignorefaults(copy_domain.copy(realm_src_fields, 
                          realm_dst_fields, indirections, requests, pred_pre));
        else
          last_copy = Runtime::ignorefaults(copy_domain.copy(realm_src_fields,
                          realm_dst_fields, requests, pred_pre));
#else
        if (!indirections.empty())
        {
          if (!individual_field_indexes.empty())
            last_copy = Runtime::ignorefaults(
                issue_individual_copies(pred_pre, requests));
          else
            last_copy = Runtime::ignorefaults(copy_domain.copy(src_fields, 
                              dst_fields, indirections, requests, pred_pre));
        }
        else
          last_copy = Runtime::ignorefaults(copy_domain.copy(src_fields,
                            dst_fields, requests, pred_pre));
#endif
      }
      else
      {
        // No need for tracing to know about the reservations
        for (std::map<Reservation,bool>::const_iterator it =
              reservations.begin(); it != reservations.end(); it++)
          copy_precondition = Runtime::acquire_ap_reservation(it->first, 
                                          it->second, copy_precondition);
#ifdef LEGION_SPY
        if (!indirections.empty())
          last_copy = ApEvent(copy_domain.copy(realm_src_fields, 
                realm_dst_fields, indirections, requests, copy_precondition));
        else
          last_copy = ApEvent(copy_domain.copy(realm_src_fields,
                realm_dst_fields, requests, copy_precondition));
#else
        if (!indirections.empty())
        {
          if (!individual_field_indexes.empty())
            last_copy = issue_individual_copies(copy_precondition, requests);
          else
            last_copy = ApEvent(copy_domain.copy(src_fields, dst_fields, 
                  indirections, requests, copy_precondition));
        }
        else
          last_copy = ApEvent(copy_domain.copy(src_fields, dst_fields,
                requests, copy_precondition));
#endif
      }
      // Release any reservations
      if (!reservations.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it =
              reservations.begin(); it != reservations.end(); it++)
          Runtime::release_reservation(it->first, last_copy);
      }
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!last_copy.exists())
      {
        ApUserEvent new_last_copy = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_last_copy);
        last_copy = new_last_copy;
      }
#endif
#ifdef LEGION_SPY
      assert(op != NULL);
      if (src_indirections.empty() && dst_indirections.empty())
      {
        LegionSpy::log_copy_events(op->get_unique_op_id(), expr->expr_id,
                  src_tree_id, dst_tree_id, copy_precondition, last_copy);
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          LegionSpy::log_copy_field(last_copy, src_fields[idx].field_id,
                                    src_fields[idx].inst_event,
                                    dst_fields[idx].field_id,
                                    dst_fields[idx].inst_event, 
                                    dst_fields[idx].redop_id);
      }
      else
      {
        LegionSpy::log_indirect_events(op->get_unique_op_id(), expr->expr_id,
                unique_indirections_identifier, copy_precondition, last_copy);
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          LegionSpy::log_indirect_field(last_copy, src_fields[idx].field_id,
                                        src_fields[idx].inst_event,
                                        src_fields[idx].indirect_index,
                                        dst_fields[idx].field_id,
                                        dst_fields[idx].inst_event, 
                                        dst_fields[idx].indirect_index,
                                        dst_fields[idx].redop_id);
      }
#endif
      return last_copy;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void CopyAcrossUnstructuredT<DIM,T>::record_trace_immutable_indirection(
                                                                    bool source)
    //--------------------------------------------------------------------------
    {
      if (source)
        src_indirect_immutable_for_tracing = true;
      else
        dst_indirect_immutable_for_tracing = true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent CopyAcrossUnstructuredT<DIM,T>::issue_individual_copies(
                                     const ApEvent precondition,
                                     const Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(compute_preimages);
#endif
      // This is the case of separate gather/scatter copies for each 
      // of the individual preimages
      const bool gather = current_dst_preimages.empty();
#ifdef DEBUG_LEGION
      // Should be either a gather or a scatter, but not both
      assert(current_src_preimages.empty() != gather);
#endif
      // Issue separate copies for each preimage
      std::vector<DomainT<DIM,T> > &preimages = 
        gather ? current_src_preimages : current_dst_preimages;
      std::vector<CopySrcDstField> &fields = gather ? src_fields : dst_fields;
#ifdef DEBUG_LEGION
      assert(preimages.size() == individual_field_indexes.size());
#endif
      std::vector<ApEvent> postconditions;
      for (unsigned idx = 0; idx < preimages.size(); idx++)
      {
#ifdef DEBUG_LEGION
        assert(fields.size() == individual_field_indexes[idx].size());
#endif
        // Setup the indirect field indexes
        for (unsigned fidx = 0; fidx < fields.size(); fidx++)
          fields[fidx].indirect_index = individual_field_indexes[idx][fidx];
        const ApEvent post(preimages[idx].copy(src_fields, dst_fields, 
                                indirections, requests, precondition));
        if (post.exists())
          postconditions.push_back(post);
      }
      if (postconditions.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(NULL, postconditions);
    }
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int D1, typename T1> template<int D2, typename T2>
    ApEvent CopyAcrossUnstructuredT<D1,T1>::perform_compute_preimages(
                     std::vector<DomainT<D1,T1> > &preimages, 
                     Operation *op, ApEvent precondition, const bool source)
    //--------------------------------------------------------------------------
    {
      const std::vector<IndirectRecord> &indirect_records =
        source ? src_indirections : dst_indirections;
      std::vector<Realm::IndexSpace<D2,T2> > targets(indirect_records.size());
      for (unsigned idx = 0; idx < indirect_records.size(); idx++)
        targets[idx] = indirect_records[idx].domain;
      if (source ? need_src_indirect_precondition : 
          need_dst_indirect_precondition)
      {
        std::vector<ApEvent> preconditions;
        for (unsigned idx = 0; idx < indirect_records.size(); idx++)
        {
          const IndirectRecord &record = indirect_records[idx];
          ApEvent ready = record.domain_ready;
          if (ready.exists())
            preconditions.push_back(ready);
        }
        if (copy_domain_ready.exists())
          preconditions.push_back(copy_domain_ready);
        if (source)
        {
          // No need for tracing to know about this merge
          src_indirect_spaces_precondition = 
            Runtime::merge_events(NULL, preconditions);
          need_src_indirect_precondition = false;
        }
        else
        {
          dst_indirect_spaces_precondition = 
            Runtime::merge_events(NULL, preconditions);
          need_dst_indirect_precondition = false;
        }
      }
      if (source ? src_indirect_spaces_precondition.exists() :
          dst_indirect_spaces_precondition.exists())
      {
        if (precondition.exists())
          precondition = Runtime::merge_events(NULL, precondition, source ?
           src_indirect_spaces_precondition : dst_indirect_spaces_precondition);
        else
          precondition = source ? src_indirect_spaces_precondition : 
            dst_indirect_spaces_precondition;
      }
      ApEvent result;
      if (both_are_range)
      {
        // Range preimage
        typedef Realm::FieldDataDescriptor<Realm::IndexSpace<D1,T1>,
                                       Realm::Rect<D2,T2> > RealmDescriptor;
        std::vector<RealmDescriptor> descriptors(1);
        RealmDescriptor &descriptor = descriptors.back();
        descriptor.inst = 
          source ? src_indirect_instance : dst_indirect_instance;
        descriptor.field_offset =
          source ? src_indirect_field : dst_indirect_field;
        descriptor.index_space = copy_domain;
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != NULL)
          runtime->profiler->add_partition_request(requests, op, 
                                    DEP_PART_BY_PREIMAGE_RANGE);
        result = ApEvent(copy_domain.create_subspaces_by_preimage(
              descriptors, targets, preimages, requests, precondition));
      }
      else
      {
        // Point preimage
        typedef Realm::FieldDataDescriptor<Realm::IndexSpace<D1,T1>,
                                       Realm::Point<D2,T2> > RealmDescriptor;
        std::vector<RealmDescriptor> descriptors(1);
        RealmDescriptor &descriptor = descriptors.back();
        descriptor.inst = 
          source ? src_indirect_instance : dst_indirect_instance;
        descriptor.field_offset =
          source ? src_indirect_field : dst_indirect_field;
        descriptor.index_space = copy_domain;
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != NULL)
          runtime->profiler->add_partition_request(requests, op, 
                                          DEP_PART_BY_PREIMAGE);
        result = ApEvent(copy_domain.create_subspaces_by_preimage(
              descriptors, targets, preimages, requests, precondition));
      }
      std::vector<ApEvent> valid_events;
      // We also need to make sure that all the sparsity maps are valid
      // on this node before we test them
      for (unsigned idx = 0; idx < preimages.size(); idx++)
      {
        const ApEvent valid(preimages[idx].make_valid());
        if (valid.exists())
          valid_events.push_back(valid);
      }
      if (!valid_events.empty())
      {
        if (result.exists())
          valid_events.push_back(result);
        result = Runtime::merge_events(NULL, valid_events);
      }
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists() || (result == precondition))
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr->expr_id,
                                    precondition, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<int D1, typename T1> template<int D2, typename T2>
    bool CopyAcrossUnstructuredT<D1,T1>::rebuild_indirections(const bool source)
    //--------------------------------------------------------------------------
    {
      std::vector<CopySrcDstField> &fields = source ? src_fields : dst_fields;
      const std::vector<IndirectRecord> &indirect_records =
        source ? src_indirections : dst_indirections;
      std::vector<unsigned> nonempty_indexes;
      if (compute_preimages)
      {
        std::vector<DomainT<D1,T1> > &preimages =
          source ? current_src_preimages : current_dst_preimages;
        for (unsigned idx = 0; idx < preimages.size(); idx++)
        {
          DomainT<D1,T1> &preimage = preimages[idx];
          DomainT<D1,T1> tightened = preimage.tighten();
          if (tightened.empty())
          {
            // Reclaim any sparsity maps eagerly
            preimage.destroy();
            preimage = DomainT<D1,T1>::make_empty();
          }
          else
          {
            preimage = tightened;
            nonempty_indexes.push_back(idx);
          }
        }
      }
      else
      {
        nonempty_indexes.resize(indirect_records.size());
        for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
          nonempty_indexes[idx] = idx;
      }
      typedef typename Realm::CopyIndirection<D1,T1>::template 
                Unstructured<D2,T2> UnstructuredIndirection;
      // Legion Spy doesn't understand preimages, so go through and build
      // indirections for everything even if we are empty
#ifndef LEGION_SPY
      if (nonempty_indexes.empty())
        return true;
      if (compute_preimages && 
          (source ? dst_indirections.empty() : src_indirections.empty()))
      {
        // In the case that we've computed preimages, and we know we're just
        // doing a gather or a scatter (no full-indirections), then we 
        // instead want to compute separate indirections for each 
        // non-empty preimage because Realm's performance is better when
        // you have a single source or destination target for an indirection
        // Note we don't bother doing this with legion spy since it doesn't
        // know how to analyze these anyway
        individual_field_indexes.resize(nonempty_indexes.size());
        // We're also going to need to update preimages to match
        std::vector<DomainT<D1,T1> > &preimages =
          source ? current_src_preimages : current_dst_preimages;
        std::vector<DomainT<D1,T1> > new_preimages(nonempty_indexes.size());
        // Iterate over the non empty indexes and get instances for each field
        for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
        {
          const unsigned nonempty_index = nonempty_indexes[idx];
          // copy over the preimages to the set of dense non-empty preimages
          new_preimages[idx] = preimages[nonempty_index]; 
          std::vector<unsigned> &field_indexes = individual_field_indexes[idx];
          field_indexes.resize(fields.size());
          const unsigned offset = indirections.size();
          for (unsigned fidx = 0; fidx < fields.size(); fidx++)
          {
            const PhysicalInstance instance =
              indirect_records[nonempty_index].instances[fidx];
            // See if there is an unstructured index for this instance
            int indirect_index = -1;
            for (unsigned index = offset; index < indirections.size(); index++)
            {
              // It's safe to cast here because we know that the same types
              // made all these indirections as well
              const UnstructuredIndirection *unstructured = 
               static_cast<const UnstructuredIndirection*>(indirections[index]);
#ifdef DEBUG_LEGION
              assert(unstructured->inst == 
                  (source ? src_indirect_instance : dst_indirect_instance));
              assert(unsigned(unstructured->field_id) == 
                  (source ? src_indirect_field : dst_indirect_field));
              assert(unstructured->insts.size() == 1);
#endif
              if (unstructured->insts.back() != instance)
                continue;
              indirect_index = index;
              break;
            }
            if (indirect_index < 0)
            {
              // If we didn't make it then make it now
              UnstructuredIndirection *unstructured =
                new UnstructuredIndirection();
              unstructured->field_id = 
                source ? src_indirect_field : dst_indirect_field;
              unstructured->inst = 
                source ? src_indirect_instance : dst_indirect_instance;
              unstructured->is_ranges = both_are_range;
              unstructured->oor_possible = false; 
              unstructured->aliasing_possible =
                source ? false/*no aliasing*/ : possible_dst_aliasing;
              unstructured->subfield_offset = 0;
              unstructured->insts.push_back(instance);
              unstructured->spaces.resize(1);
              unstructured->spaces.back() =
                indirect_records[nonempty_index].domain;
              // No next indirections yet...
              unstructured->next_indirection = NULL;
              indirect_index = indirections.size();
              indirections.push_back(unstructured);
            }
            field_indexes[fidx] = indirect_index;
          }
        }
        // Now we can swap in the new preimages
        preimages.swap(new_preimages);
      }
      else
#else
      const unsigned offset = indirections.size();
#endif
      {
        // Now that we have the non-empty indexes we can go through and make
        // the indirections for each of the fields. We'll try to share 
        // indirections as much as possible wherever we can
#ifndef LEGION_SPY
        const unsigned offset = indirections.size(); 
#endif
        for (unsigned fidx = 0; fidx < fields.size(); fidx++)
        {
          // Compute our physical instances for this field
#ifdef LEGION_SPY
          std::vector<PhysicalInstance> instances(indirect_records.size());
          for (unsigned idx = 0; idx < indirect_records.size(); idx++)
            instances[idx] = indirect_records[idx].instances[fidx];
#else
          std::vector<PhysicalInstance> instances(nonempty_indexes.size());
          for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
            instances[idx] =
              indirect_records[nonempty_indexes[idx]].instances[fidx];
#endif
          // See if there is an unstructured index which already is what we want
          int indirect_index = -1;
          // Search through all the existing copy indirections starting from
          // the offset and check to see if we can reuse them
          for (unsigned index = offset; index < indirections.size(); index++)
          {
            // It's safe to cast here because we know that the same types
            // made all these indirections as well
            const UnstructuredIndirection *unstructured = 
              static_cast<const UnstructuredIndirection*>(indirections[index]);
#ifdef DEBUG_LEGION
            assert(unstructured->inst == 
                (source ? src_indirect_instance : dst_indirect_instance));
            assert(unsigned(unstructured->field_id) == 
                (source ? src_indirect_field : dst_indirect_field));
            assert(unstructured->insts.size() == instances.size());
#endif
            bool instances_match = true;
            for (unsigned idx = 0; idx < instances.size(); idx++)
            {
              if (unstructured->insts[idx] == instances[idx])
                continue;
              instances_match = false;
              break;
            }
            if (!instances_match)
              continue;
            // If we made it here we can reuse this indirection
            indirect_index = index;
            break;
          }
          if (indirect_index < 0)
          {
            // If we didn't make it then make it now
            UnstructuredIndirection *unstructured =
              new UnstructuredIndirection();
            unstructured->field_id = 
              source ? src_indirect_field : dst_indirect_field;
            unstructured->inst = 
              source ? src_indirect_instance : dst_indirect_instance;
            unstructured->is_ranges = both_are_range;
            unstructured->oor_possible = compute_preimages ? false :
              source ? possible_src_out_of_range : possible_dst_out_of_range;
            unstructured->aliasing_possible =
              source ? false/*no aliasing*/ : possible_dst_aliasing;
            unstructured->subfield_offset = 0;
            unstructured->insts.swap(instances);
            unstructured->spaces.resize(nonempty_indexes.size());
            for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
              unstructured->spaces[idx] =
                indirect_records[nonempty_indexes[idx]].domain;
            // No next indirections yet...
            unstructured->next_indirection = NULL;
            indirect_index = indirections.size();
            indirections.push_back(unstructured);
#ifdef LEGION_SPY
            // If we made a new indirection then log it with Legion Spy
            LegionSpy::log_indirect_instance(unique_indirections_identifier,
                indirect_index, source ? src_indirect_instance_event :
                dst_indirect_instance_event, unstructured->field_id);
            for (std::vector<IndirectRecord>::const_iterator it =
                  indirect_records.begin(); it != indirect_records.end(); it++)
              LegionSpy::log_indirect_group(unique_indirections_identifier,
                  indirect_index, it->instance_events[fidx], 
                  it->index_space.get_id());
#endif
          }
          fields[fidx].indirect_index = indirect_index;
        }
      }
#ifdef LEGION_SPY
      if (compute_preimages)
      {
        const size_t nonempty_size = nonempty_indexes.size();
        // Go through and fix-up all the indirections for execution
        for (typename std::vector<const CopyIndirection*>::const_iterator it =
              indirections.begin()+offset; it != indirections.end(); it++)
        {
          UnstructuredIndirection *unstructured = 
            const_cast<UnstructuredIndirection*>( 
              static_cast<const UnstructuredIndirection*>(*it));
          std::vector<PhysicalInstance> instances(nonempty_size);
          for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
            instances[idx] = unstructured->insts[nonempty_indexes[idx]];
          unstructured->insts.swap(instances);
        }
      }
      return nonempty_indexes.empty();
#else
      // Not empty
      return false;
#endif
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

    /////////////////////////////////////////////////////////////
    // Templated Index Partition Node 
    /////////////////////////////////////////////////////////////

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(RegionTreeForest *ctx, 
                                        IndexPartition p,
                                        IndexSpaceNode *par, IndexSpaceNode *cs,
                                        LegionColor c, bool disjoint, 
                                        int complete, DistributedID did,
                                        ApEvent part_ready, ApUserEvent pend,
                                        RtEvent init, Provenance *prov)
      : IndexPartNode(ctx, p, par, cs, c, disjoint, complete, did, part_ready,
                      pend, init, prov)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(RegionTreeForest *ctx, 
                                        IndexPartition p,
                                        IndexSpaceNode *par, IndexSpaceNode *cs,
                                        LegionColor c, RtEvent disjoint_event,
                                        int complete, DistributedID did,
                                        ApEvent partition_ready, 
                                        ApUserEvent pending, RtEvent init,
                                        Provenance *prov)
      : IndexPartNode(ctx, p, par, cs, c, disjoint_event, complete, did, 
                      partition_ready, pending, init, prov)
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
#endif // defined(DEFINE_NT_TEMPLATES)

  }; // namespace Internal
}; // namespace Legion

