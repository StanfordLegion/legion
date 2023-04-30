/* Copyright 2023 Stanford University, NVIDIA Corporation
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
                                 RegionTreeForest *forest, Operation *op,
                                 const Realm::IndexSpace<DIM,T> &space,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                 UniqueID fill_uid,
                                 FieldSpace handle,
                                 RegionTreeID tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent unique_event,
                                 CollectiveKind collective,
                                 int priority, bool replay)
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
      if (!replay)
        priority =
          op->add_copy_profiling_request(trace_info, requests, true/*fill*/);
      if (forest->runtime->profiler != NULL)
        forest->runtime->profiler->add_fill_request(requests, op, collective);
      ApEvent fill_pre;
      if (pred_guard.exists())
        // No need for tracing to know about the precondition
        fill_pre = Runtime::merge_events(NULL,precondition,ApEvent(pred_guard));
      else
        fill_pre = precondition;
      ApEvent result = ApEvent(space.fill(dst_fields, requests,
              fill_value, fill_size, fill_pre, priority));
      LgEvent fevent = result;
      if (pred_guard.exists())
      {
        result = Runtime::ignorefaults(result);
        // Need to merge back in the precondition so it is still reflected
        // in the completion event for this operation
        if (precondition.exists())
        {
          if (result.exists())
            result = Runtime::merge_events(NULL, result, precondition);
          else
            result = precondition;
          // Little catch here for tracing, make sure the result is unique
          if (trace_info.recording && result.exists() &&
              (result == precondition))
          {
            ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
            Runtime::trigger_event(NULL, new_result, precondition);
            result = new_result;
          }
        }
      }
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists())
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
      if (forest->runtime->profiler != NULL)
      {
        for (unsigned idx = 0; idx < dst_fields.size(); idx++)
          forest->runtime->profiler->record_fill_instance(
              dst_fields[idx].field_id, dst_fields[idx].inst,
              unique_event, fevent);
      }
#ifdef LEGION_SPY
      LegionSpy::log_fill_events(op->get_unique_op_id(), expr_id, handle,
          tree_id, precondition, result, fill_uid, collective);
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        LegionSpy::log_fill_field(result, dst_fields[idx].field_id,
                                  unique_event);
#endif
      if (trace_info.recording)
        trace_info.record_issue_fill(result, this, dst_fields,
                                     fill_value, fill_size,
#ifdef LEGION_SPY
                                     fill_uid, handle, tree_id,
#endif
                                     precondition, pred_guard,
                                     unique_event, priority, collective);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceExpression::issue_copy_internal(
                                 RegionTreeForest *forest, Operation *op,
                                 const Realm::IndexSpace<DIM,T> &space,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const std::vector<CopySrcDstField> &src_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent src_unique, LgEvent dst_unique,
                                 CollectiveKind collective,
                                 int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(forest->runtime, REALM_ISSUE_COPY_CALL);
#ifdef DEBUG_LEGION
      assert(!space.empty());
      // If we're doing any reductions with this copy then make sure they
      // are marked exclusive or we have some reservations
      for (std::vector<CopySrcDstField>::const_iterator it =
            dst_fields.begin(); it != dst_fields.end(); it++)
        assert((it->redop_id == 0) ||
                it->red_exclusive || !reservations.empty());
#endif
      // Now that we know we're going to do this copy add any profling requests
      Realm::ProfilingRequestSet requests;
      if (!replay)
        priority =
          op->add_copy_profiling_request(trace_info, requests, false/*fill*/);
      if (forest->runtime->profiler != NULL)
        forest->runtime->profiler->add_copy_request(requests, op, 1/*count*/,
                                                    collective);
      ApEvent copy_pre;
      if (pred_guard.exists())
        copy_pre = Runtime::merge_events(NULL,precondition,ApEvent(pred_guard));
      else
        copy_pre = precondition;
      for (std::vector<Reservation>::const_iterator it =
            reservations.begin(); it != reservations.end(); it++)
        copy_pre = Runtime::acquire_ap_reservation(*it, 
                                        true/*exclusive*/, copy_pre);
      ApEvent result = ApEvent(space.copy(src_fields, dst_fields, requests,
                                          copy_pre, priority));
      LgEvent fevent = result;
      // Release any reservations after the copy is done
      for (std::vector<Reservation>::const_iterator it =
            reservations.begin(); it != reservations.end(); it++)
        Runtime::release_reservation(*it, result);
      if (pred_guard.exists())
      {
        result = Runtime::ignorefaults(result);
        // Make sure to fold in the precondition back into the result
        // event in case this is poisoned to support transitive analysis
        if (precondition.exists())
        {
          if (result.exists())
            result = Runtime::merge_events(NULL, result, precondition);
          else
            result = precondition;
          // Little catch here for tracing, make sure the result is unique
          if (trace_info.recording && result.exists() &&
              (result == precondition))
          {
            ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
            Runtime::trigger_event(NULL, new_result, precondition);
            result = new_result;
          }
        }
      }
      if (trace_info.recording)
        trace_info.record_issue_copy(result, this, src_fields,
                                     dst_fields, reservations,
#ifdef LEGION_SPY
                                     src_tree_id, dst_tree_id,
#endif
                                     precondition, pred_guard,
                                     src_unique, dst_unique, 
                                     priority, collective);
#ifdef LEGION_DISABLE_EVENT_PRUNING
      if (!result.exists())
      {
        ApUserEvent new_result = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, new_result);
        result = new_result;
      }
#endif
      if (forest->runtime->profiler != NULL)
      {
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          forest->runtime->profiler->record_copy_instances(
              src_fields[idx].field_id, dst_fields[idx].field_id,
              src_fields[idx].inst, dst_fields[idx].inst,
              src_unique, dst_unique, fevent);
      }
#ifdef LEGION_SPY
      LegionSpy::log_copy_events(op->get_unique_op_id(), expr_id, src_tree_id,
                                 dst_tree_id, precondition, result, collective);
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        LegionSpy::log_copy_field(result, src_fields[idx].field_id,
                                  src_unique,
                                  dst_fields[idx].field_id,
                                  dst_unique,
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
                                   bool compact, void **piece_list,
                                   size_t *piece_list_size,
                                   size_t *num_pieces) const
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
      const SpecializedConstraint &spec = constraints.specialized_constraint;
      if (space.dense() || !compact)
      {
        if (!space.bounds.empty())
        {
          // Check to see if we have any tiling constraints
          if (!constraints.tiling_constraints.empty())
          {
#ifdef DEBUG_LEGION
            assert(piece_list != NULL);
            assert((*piece_list) == NULL);
            assert(piece_list_size != NULL);
            assert((*piece_list_size) == 0);
            assert(num_pieces != NULL);
            assert((*num_pieces) == 0);
#endif
            // First get the tile bounds
            Point<DIM,T> tile_size;
            for (int i = 0; i < DIM; i++)
              tile_size[i] = (space.bounds.hi[i] - space.bounds.lo[i]) + 1;
            for (std::vector<TilingConstraint>::const_iterator it =
                  constraints.tiling_constraints.begin(); it !=
                  constraints.tiling_constraints.end(); it++)
            {
#ifdef DEBUG_LEGION
              assert(it->dim < DIM);
#endif
              if (it->tiles)
                tile_size[it->dim] = 
                  (tile_size[it->dim] + it->value - 1) / it->value;
              else
                tile_size[it->dim] = it->value;
            }
            // Now we've got the tile size, walk over the dimensions 
            // in order to produce the tiles as pieces
            Point<DIM,T> offset = space.bounds.lo;
            // Iterate until we've tiled the entire space
            bool done = false;
            while (!done)
            {
              // Check to make sure the next tile is in bounds
              Rect<DIM,T> piece(offset, 
                  offset + tile_size - Point<DIM,T>::ONES());
              // Intersect with the original bounds to not overflow
              piece = space.bounds.intersection(piece);
#ifdef DEBUG_LEGION
              assert(!piece.empty());
#endif
              piece_bounds.push_back(piece);
              // Step the offset to the next location
              done = true;
              for (std::vector<TilingConstraint>::const_iterator it =
                    constraints.tiling_constraints.begin(); it !=
                    constraints.tiling_constraints.end(); it++)
              {
                offset[it->dim] += tile_size[it->dim];
                if (offset[it->dim] <= space.bounds.hi[it->dim])
                {
                  // Still in bounds so we can keep traversing
                  done = false;
                  break;
                }
                else // No longer in bounds, so ripple carry add
                  offset[it->dim] = space.bounds.lo[it->dim];
              }
            }
          }
          else
            piece_bounds.push_back(space.bounds);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(piece_list != NULL);
        assert((*piece_list) == NULL);
        assert(piece_list_size != NULL);
        assert((*piece_list_size) == 0);
        assert(num_pieces != NULL);
        assert((*num_pieces) == 0);
#endif
        if (spec.max_overhead > 0)
        {
          std::vector<Realm::Rect<DIM,T> > covering;
          if (space.compute_covering(spec.max_pieces, spec.max_overhead,
                                      covering))
          {
            // Container problem is stupid
            piece_bounds.resize(covering.size());
            for (unsigned idx = 0; idx < covering.size(); idx++)
              piece_bounds[idx] = covering[idx];
          }
          else
          {
            // Just fill in with the compact rectangles for now
            // This will likely fail the max pieces test later
            for (Realm::IndexSpaceIterator<DIM,T> itr(space); 
                  itr.valid; itr.step())
              if (!itr.rect.empty())
                piece_bounds.push_back(itr.rect);
          }
        }
        else
        {
          for (Realm::IndexSpaceIterator<DIM,T> itr(space); 
                itr.valid; itr.step())
            if (!itr.rect.empty())
              piece_bounds.push_back(itr.rect);
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
      else if (piece_bounds.size() > 1)
      {
        // Realm doesn't currently support padding on multiple pieces because
        // then we might have valid points in multiple pieces and its 
        // undefined which pieces Realm might copy to
        if (constraints.padding_constraint.delta.get_dim() > 0)
          REPORT_LEGION_FATAL(LEGION_FATAL_COMPACT_SPARSE_PADDING,
              "Legion does not currently support additional padding "
              "on compact sparse instances. Please open a github "
              "issue to request support.")
        *num_pieces = piece_bounds.size();
        *piece_list_size = piece_bounds.size() * sizeof(Rect<DIM,T>);
        *piece_list = malloc(*piece_list_size);
        Rect<DIM,T> *pieces = static_cast<Rect<DIM,T>*>(*piece_list);
        for (unsigned idx = 0; idx < piece_bounds.size(); idx++)
          pieces[idx] = piece_bounds[idx];
      }
      else if (constraints.padding_constraint.delta.get_dim() > 0)
      {
        // If the user requested any scratch padding on the instance apply it
        const Domain &delta = constraints.padding_constraint.delta;
        const Point<DIM> lo = delta.lo();
        const Point<DIM> hi = delta.hi();
#ifdef DEBUG_LEGION
        assert(!piece_bounds.empty());
        for (int i = 0; i < DIM; i++)
        {
          assert(lo[i] >= 0);
          assert(hi[i] >= 0);
        }
#endif
        for (typename std::vector<Rect<DIM,T> >::iterator it = 
              piece_bounds.begin(); it != piece_bounds.end(); it++)
        {
          it->lo -= lo;
          it->hi += hi;
        }
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
        // Hack to help out lazy users unwilling to specify alignment 
        // constraints that are necessary for correctness
        // If they haven't specified an alignment we align on the largest
        // power of two that divides the size of the field, for more
        // details see https://github.com/StanfordLegion/legion/issues/1384
        // Cap at a maximum of 128 byte alignment for GPUs
        const size_t field_alignment =
          (alignment_finder != alignments.end()) ? alignment_finder->second : 1;
          //std::min<size_t>(it->first & ~(it->first - 1), 128/*max alignment*/);
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
      else if (expressions.find(this) != expressions.end())
        return this;
      Realm::IndexSpace<DIM,T> local_space;
      // No need to wait for the event, we know it is already triggered
      // because we called get_volume on this before we got here
      get_expr_index_space(&local_space, type_tag, true/*need tight result*/);
      size_t local_rect_count = 0;
      KDNode<DIM,T,void> *local_tree = NULL;
      for (std::set<IndexSpaceExpression*>::const_iterator it =
            expressions.begin(); it != expressions.end(); it++)
      {
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
          // Check to see if they have the expression is still alive and
          // can be used as a canonical expression
          if ((*it)->try_add_live_reference())
          {
            if (local_tree != NULL)
              delete local_tree;
            return (*it);
          }
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
          KDNode<DIM,T> *other_tree = 
            (*it)->get_sparsity_map_kd_tree()->as_kdnode<DIM,T>();
          size_t other_rect_count = other_tree->count_rectangles();
          if (local_rect_count == 0)
          {
            // Count the number of rectangles in our sparsity map
            for (Realm::IndexSpaceIterator<DIM,T> itr(local_space);
                  itr.valid; itr.step())
              local_rect_count++;
#ifdef DEBUG_LEGION
            assert(local_rect_count > 0);
#endif
          }
          if (other_rect_count < local_rect_count)
          {
            // Build our KD tree if we haven't already
            if (local_tree == NULL)
            {
              std::vector<Rect<DIM,T> > local_rects;
              for (Realm::IndexSpaceIterator<DIM,T> itr(local_space);
                    itr.valid; itr.step())
                local_rects.push_back(itr.rect);
              local_tree = new KDNode<DIM,T>(local_space.bounds, local_rects);
            }
            // Iterate the other rectangles and see if they are covered
            bool congruent = true; 
            for (Realm::IndexSpaceIterator<DIM,T> itr(other_space);
                  itr.valid; itr.step())
            {
              const size_t intersecting_points = 
                local_tree->count_intersecting_points(itr.rect);
              if (intersecting_points == itr.rect.volume())
                continue;
              congruent = false;
              break;
            }
            if (!congruent)
              continue;
          }
          else
          {
            // Iterate our rectangles and see if they are all covered
            bool congruent = true; 
            for (Realm::IndexSpaceIterator<DIM,T> itr(local_space);
                  itr.valid; itr.step())
            {
              const size_t intersecting_points = 
                other_tree->count_intersecting_points(itr.rect);
              if (intersecting_points == itr.rect.volume())
                continue;
              congruent = false;
              break;
            }
            if (!congruent)
              continue;
          }
        }  
        // If we get here that means we are congruent
        // Try to add the expression reference, we can race with deletions
        // here though so handle the case we're we can't add a reference
        if ((*it)->try_add_live_reference())
        {
          if (local_tree != NULL)
            delete local_tree;
          return (*it);
        }
      }
      // Did not find any congruences so add ourself
      expressions.insert(this);
      // If we have a KD tree we can save it for later congruence tests
      if (local_tree != NULL)
      {
#ifdef DEBUG_LEGION
        assert(sparsity_map_kd_tree == NULL); // should not have a kd tree yet
#endif
        sparsity_map_kd_tree = local_tree;
      }
      return this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline KDTree* IndexSpaceExpression::get_sparsity_map_kd_tree_internal(void)
    //--------------------------------------------------------------------------
    {
      if (sparsity_map_kd_tree != NULL)
        return sparsity_map_kd_tree;
      Realm::IndexSpace<DIM,T> local_space;
      // No need to wait for the event, we know it is already triggered
      // because we called get_volume on this before we got here
      get_expr_index_space(&local_space, type_tag, true/*need tight result*/);
#ifdef DEBUG_LEGION
      assert(!local_space.dense());
#endif
      std::vector<Rect<DIM,T> > local_rects;
      for (Realm::IndexSpaceIterator<DIM,T> itr(local_space);
            itr.valid; itr.step())
        local_rects.push_back(itr.rect);
      sparsity_map_kd_tree = new KDNode<DIM,T>(local_space.bounds, local_rects);
      return sparsity_map_kd_tree;
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
        IndexSpaceExprID eid, DistributedID did,
        IndexSpaceOperation *origin, TypeTag tag, Deserializer &derez)
      : IndexSpaceOperation(tag, ctx, eid, did, origin),
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
        this->pack_global_ref();
      }
      else
      {
        rez.serialize<bool>(false/*local*/);
        rez.serialize<bool>(false/*index space*/);
        rez.serialize(expr_id);
        rez.serialize(origin_expr);
        // Add a reference here that we'll remove after we've added a reference
        // on the target space expression
        this->pack_global_ref();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNode* IndexSpaceOperationT<DIM,T>::create_node(IndexSpace handle,
                         DistributedID did, RtEvent initialized, 
                         Provenance *provenance,
                         CollectiveMapping *collective_mapping,
                         IndexSpaceExprID new_expr_id)
    //--------------------------------------------------------------------------
    {
      if (new_expr_id == 0)
        new_expr_id = expr_id;
      AutoLock i_lock(inter_lock, 1, false/*exclusive*/);
      if (is_index_space_tight)
        return context->create_node(handle, &tight_index_space, false/*domain*/,
                          NULL/*parent*/, 0/*color*/, did, initialized,
                          provenance, realm_index_space_ready, new_expr_id,
                          collective_mapping, true/*add root ref*/);
      else
        return context->create_node(handle, &realm_index_space, false/*domain*/,
                          NULL/*parent*/, 0/*color*/, did, initialized,
                          provenance, realm_index_space_ready, new_expr_id,
                          collective_mapping, true/*add root ref*/);
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
    ApEvent IndexSpaceOperationT<DIM,T>::issue_fill(Operation *op,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                 UniqueID fill_uid,
                                 FieldSpace handle,
                                 RegionTreeID tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent unique_event,
                                 CollectiveKind collective,
                                 int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (space_ready.exists() && precondition.exists())
        return issue_fill_internal(context, op, local_space, trace_info, 
            dst_fields, fill_value, fill_size, 
#ifdef LEGION_SPY
            fill_uid, handle, tree_id,
#endif
            Runtime::merge_events(&trace_info, space_ready, precondition),
            pred_guard, unique_event, collective, priority, replay);
      else if (space_ready.exists())
        return issue_fill_internal(context, op, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   space_ready, pred_guard, unique_event,
                                   collective, priority, replay);
      else
        return issue_fill_internal(context, op, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   precondition, pred_guard, unique_event,
                                   collective, priority, replay);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM,T>::issue_copy(Operation *op,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const std::vector<CopySrcDstField> &src_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent src_unique, LgEvent dst_unique,
                                 CollectiveKind collective,
                                 int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (space_ready.exists() && precondition.exists())
        return issue_copy_internal(context, op, local_space, trace_info,
            dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
            src_tree_id, dst_tree_id,
#endif
            Runtime::merge_events(&trace_info, precondition, space_ready),
            pred_guard, src_unique, dst_unique, collective, priority, replay);
      else if (space_ready.exists())
        return issue_copy_internal(context, op, local_space, trace_info,
                dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                space_ready, pred_guard, src_unique, dst_unique,
                collective, priority, replay);
      else
        return issue_copy_internal(context, op, local_space, trace_info,
                dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                precondition, pred_guard, src_unique, dst_unique,
                collective, priority, replay);
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
                                    bool compact, void **piece_list, 
                                    size_t *piece_list_size, size_t *num_pieces)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_is;
      ApEvent space_ready = get_realm_index_space(local_is, true/*tight*/);
      if (space_ready.exists())
        space_ready.wait();
      return create_layout_internal(local_is, constraints,field_ids,field_sizes,
                              compact, piece_list, piece_list_size, num_pieces);
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
    KDTree* IndexSpaceOperationT<DIM,T>::get_sparsity_map_kd_tree(void)
    //--------------------------------------------------------------------------
    {
      return get_sparsity_map_kd_tree_internal<DIM,T>();
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
      ImplicitReferenceTracker::record_live_expression(this);
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
        IndexSpaceExprID eid, DistributedID did,
        IndexSpaceOperation *origin, TypeTag tag, Deserializer &derez)
      : IndexSpaceOperationT<DIM,T>(forest, eid, did, origin, tag, derez)
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
        const void *bounds, bool is_domain, DistributedID did, 
        ApEvent ready, IndexSpaceExprID expr_id, RtEvent init, unsigned dep,
        Provenance *prov, CollectiveMapping *mapping, bool tree_valid)
      : IndexSpaceNode(ctx, handle, parent, color, did, ready, expr_id, init,
          dep, prov, mapping, tree_valid), linearization(NULL)
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
        index_space_set = true;
      }
      else
        add_base_resource_ref(RUNTIME_REF);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM,T>::~IndexSpaceNodeT(void)
    //--------------------------------------------------------------------------
    { 
      if (is_owner())
        realm_index_space.destroy(
            tight_index_space ? tight_index_space_set : realm_index_space_set);
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear != NULL)
        delete linear;
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
    bool IndexSpaceNodeT<DIM,T>::set_realm_index_space(AddressSpaceID source,
                                          const Realm::IndexSpace<DIM,T> &value,
                                          const CollectiveMapping *mapping,
                                                       RtEvent ready_event)
    //--------------------------------------------------------------------------
    {
      bool need_broadcast = true;
      const AddressSpaceID owner_space = get_owner_space();
      if (source == local_space)
      {
        if (mapping != NULL)
        {
          if ((collective_mapping != NULL) && ((mapping == collective_mapping)
                || (*mapping == *collective_mapping)))
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
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(value);
              rez.serialize(ready_event);
            }
            runtime->send_index_space_set(owner_space, rez);
            // If we're part of the broadcast tree then we'll get sent back here
            // later so we don't need to do anything now
            if ((collective_mapping != NULL) && 
                collective_mapping->contains(local_space))
              return false;
          }
        }
        else
        {
          // If we're not the owner space, send the message there
          if (!is_owner())
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(value);
              rez.serialize(ready_event);
            }
            runtime->send_index_space_set(owner_space, rez);
            // If we're part of the broadcast tree then we'll get sent back here
            // later so we don't need to do anything now
            if ((collective_mapping != NULL) && 
                collective_mapping->contains(local_space))
              return false;
            need_broadcast = false;
          }
        }
      }
      if (need_broadcast && (collective_mapping != NULL) &&
          collective_mapping->contains(local_space))
      {
#ifdef DEBUG_LEGION
        // Should be from our parent
        assert(is_owner() || (source == 
            collective_mapping->get_parent(owner_space, local_space)));
#endif
        // Keep broadcasting this out to all the children
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        if (!children.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(value);
            rez.serialize(realm_index_space_set);
          }
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
            runtime->send_index_space_set(*it, rez);
        }
      }
      // We can set this now and trigger the event but setting the
      // flag has to be done while holding the node_lock on the owner
      // node so that it is serialized with respect to queries from 
      // remote nodes for copies about the remote instance
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(!index_space_set);
        assert(!realm_index_space_set.has_triggered());
#endif
        realm_index_space = value;
        Runtime::trigger_event(realm_index_space_set, ready_event);
        index_space_set = true;
        if (is_owner() && has_remote_instances())
        {
          // We're the owner, send messages to everyone else that we've 
          // sent this node to except the source
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            pack_index_space(rez, false/*include size*/);
            rez.serialize(realm_index_space_set);
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
                        AddressSpaceID source, const CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      const DomainT<DIM,T> realm_space = domain;
      return set_realm_index_space(source, realm_space, mapping);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::set_output_union(
                         const std::map<DomainPoint,DomainPoint> &output_sizes,
                         AddressSpaceID space, const CollectiveMapping *mapping)
    //-------------------------------------------------------------------------- 
    {
      std::vector<Realm::Rect<DIM,T> > output_rects;
      output_rects.reserve(output_sizes.size());
      for (std::map<DomainPoint,DomainPoint>::const_iterator it =
            output_sizes.begin(); it != output_sizes.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert((it->first.get_dim()+it->second.dim) == DIM);
#endif
        int launch_ndim = DIM - it->second.dim;
        Point<DIM,T> lo, hi;
        for (int idx = 0; idx < launch_ndim; idx++)
        {
          lo[idx] = it->first[idx];
          hi[idx] = it->first[idx];
        }
        for (int idx = launch_ndim ; idx < DIM; idx++)
        {
          lo[idx] = 0;
          hi[idx] = it->second[idx - launch_ndim] - 1;
        }
        output_rects.push_back(Realm::Rect<DIM,T>(lo, hi));
      }
      const Realm::IndexSpace<DIM,T> output_space(output_rects);
      return set_realm_index_space(space, output_space, mapping);
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
                         Provenance *provenance,
                         CollectiveMapping *collective_mapping,
                         IndexSpaceExprID new_expr_id)
    //--------------------------------------------------------------------------
    {
      if (new_expr_id == 0)
        new_expr_id = expr_id;
#ifdef DEBUG_LEGION
      assert(handle.get_type_tag() == new_handle.get_type_tag());
#endif
      Realm::IndexSpace<DIM,T> local_space;
      const ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      return context->create_node(new_handle, &local_space, false/*domain*/,
                              NULL/*parent*/, 0/*color*/, did, initialized,
                              provenance, ready, new_expr_id,
                              collective_mapping, true/*add root reference*/);
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
      std::vector<Realm::IndexSpace<DIM,T> > spaces(partition->total_children);
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
        ApEvent ready = child->get_realm_index_space(spaces[subspace_index++],
                                                     false/*tight*/);
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
    bool IndexSpaceNodeT<DIM,T>::contains_point(const Point<DIM,T> &p)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> test_space;
      // Wait for a tight space on which to perform the test
      get_realm_index_space(test_space, true/*tight*/);
      return test_space.contains(p);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::get_max_linearized_color(void)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear == NULL)
        linear = compute_linearization_metadata();
      return linear->get_max_linearized_color();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceLinearizationT<DIM,T>*
                    IndexSpaceNodeT<DIM,T>::compute_linearization_metadata(void)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> space;
      get_realm_index_space(space, true/*tight*/);
      ColorSpaceLinearizationT<DIM,T> *result = 
        new ColorSpaceLinearizationT<DIM,T>(space);
      ColorSpaceLinearizationT<DIM,T> *expected = NULL;
      if (!linearization.compare_exchange_strong(expected, result))
      {
        delete result;
#ifdef DEBUG_LEGION
        assert(expected != NULL);
#endif
        result = expected;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::linearize_color(const DomainPoint &p)
    //--------------------------------------------------------------------------
    {
      const Point<DIM,T> point = p;
      return linearize_color(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::linearize_color(const void *realm_color,
                                                        TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear == NULL)
        linear = compute_linearization_metadata();
      Point<DIM,T> point;
      if (type_tag != handle.get_type_tag())
      {
        DomainPoint dp;
        RealmPointConverter<DIM,Realm::DIMTYPES>::convert_from(
            realm_color, type_tag, dp, "linearize_color");
        point = dp;
      }
      else
        point = *(static_cast<const Point<DIM,T>*>(realm_color));
      return linear->linearize(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM,T>::linearize_color(
                                                      const Point<DIM,T> &point)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear == NULL)
        linear = compute_linearization_metadata();
      return linear->linearize(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::delinearize_color(LegionColor color,
                                                   Point<DIM,T> &point)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear == NULL)
        linear = compute_linearization_metadata();
      linear->delinearize(color, point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::delinearize_color(LegionColor color,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear == NULL)
        linear = compute_linearization_metadata();
      Point<DIM,T> point;
      linear->delinearize(color, point);
      if (type_tag != handle.get_type_tag())
        RealmPointConverter<DIM,Realm::DIMTYPES>::convert_to(
            DomainPoint(point), realm_color, type_tag, "delinearize_color");
      else
        *(static_cast<Point<DIM,T>*>(realm_color)) = point;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM,T>::compute_color_offset(LegionColor color)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear == NULL)
        linear = compute_linearization_metadata();
      return linear->compute_color_offset(color);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::contains_color(LegionColor color, 
                                                bool report_error/*=false*/)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear == NULL)
        linear = compute_linearization_metadata();
      const bool result = linear->contains_color(color);
      if (!result && report_error)
        REPORT_LEGION_ERROR(ERROR_INVALID_INDEX_SPACE_COLOR,
              "Invalid color request")
      return result;
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
      Point<DIM,T> color_point;
      delinearize_color(c, color_point);
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
      RtEvent ready_event;
      derez.deserialize(ready_event);
      return set_realm_index_space(source, result_space, NULL, ready_event);
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
      const size_t count = partition->total_children;
      if (partition->is_owner() && (partition->collective_mapping == NULL))
      {
        // Common case is not control replication
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
        LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
                                      ready, result, DEP_PART_EQUAL);
#endif
        // Enumerate the colors and assign the spaces
        unsigned subspace_index = 0;
        for (ColorSpaceIterator itr(partition); itr; itr++)
        {
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
#ifdef DEBUG_LEGION
          assert(subspace_index < subspaces.size());
#endif
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspaces[subspace_index++]))
            assert(false); // should never hit this
        }
        return result;
      }
      else
      {
        const size_t count = partition->total_children;
        std::set<ApEvent> done_events;
        if (!realm_index_space_set.has_triggered())
          realm_index_space_set.wait();
        // In the case of control replication we do things 
        // one point at a time for the subspaces owned by this shard
        size_t color_offset = SIZE_MAX;
        for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
        {
          if (color_offset == SIZE_MAX)
            color_offset = partition->color_space->compute_color_offset(*itr);
          else
            color_offset++;
          Realm::ProfilingRequestSet requests;
          if (context->runtime->profiler != NULL)
            context->runtime->profiler->add_partition_request(requests,
                                                    op, DEP_PART_EQUAL);
          Realm::IndexSpace<DIM,T> subspace;
          ApEvent result(realm_index_space.create_equal_subspace(count, 
            granularity, color_offset, subspace, requests, 
            index_space_ready));
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
          if (child->set_realm_index_space(context->runtime->address_space,
                                           subspace))
            assert(false); // should never hit this
          done_events.insert(result);
        }
        if (!done_events.empty())
          return Runtime::merge_events(NULL, done_events);
        else
          return ApEvent::NO_AP_EVENT;
      }
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
      const size_t count = partition->total_children;
      std::vector<Realm::IndexSpace<DIM,T> > lhs_spaces(count);
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces(count);
      std::set<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *left_child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(left->get_child(*itr));
        IndexSpaceNodeT<DIM,T> *right_child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(*itr));
        lhs_spaces.resize(lhs_spaces.size() + 1);
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent left_ready = 
          left_child->get_realm_index_space(lhs_spaces.back(),
                                            false/*tight*/);
        ApEvent right_ready = 
          right_child->get_realm_index_space(rhs_spaces.back(),
                                             false/*tight*/);
        if (left_ready.exists())
          preconditions.insert(left_ready);
        if (right_ready.exists())
          preconditions.insert(right_ready);
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                              op, DEP_PART_UNIONS);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      const ApEvent precondition = Runtime::merge_events(NULL, preconditions);
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
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
                                    precondition, result, DEP_PART_UNIONS);
#endif
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(subspace_index < subspaces.size());
#endif
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[subspace_index++]))
          assert(false); // should never hit this
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
      std::vector<Realm::IndexSpace<DIM,T> > lhs_spaces;
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces;
      std::set<ApEvent> preconditions;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *left_child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(left->get_child(*itr));
        IndexSpaceNodeT<DIM,T> *right_child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(*itr));
        lhs_spaces.resize(lhs_spaces.size() + 1);
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent left_ready = 
          left_child->get_realm_index_space(lhs_spaces.back(),
                                            false/*tight*/);
        ApEvent right_ready = 
          right_child->get_realm_index_space(rhs_spaces.back(),
                                             false/*tight*/);
        if (left_ready.exists())
          preconditions.insert(left_ready);
        if (right_ready.exists())
          preconditions.insert(right_ready);
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_INTERSECTIONS);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      const ApEvent precondition = Runtime::merge_events(NULL, preconditions);
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
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
          precondition, result, DEP_PART_INTERSECTIONS);
#endif
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(subspace_index < subspaces.size());
#endif
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[subspace_index++]))
          assert(false); // should never hit this
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
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces;
      std::set<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *right_child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(*itr));
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent right_ready = 
          right_child->get_realm_index_space(rhs_spaces.back(),
                                             false/*tight*/);
        if (right_ready.exists())
          preconditions.insert(right_ready);
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
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
                      precondition, result, DEP_PART_INTERSECTIONS);
#endif
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(subspace_index < subspaces.size());
#endif
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[subspace_index++]))
          assert(false); // should never hit this
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
      std::vector<Realm::IndexSpace<DIM,T> > lhs_spaces;
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces;
      std::set<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *left_child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(left->get_child(*itr));
        IndexSpaceNodeT<DIM,T> *right_child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(right->get_child(*itr));
        lhs_spaces.resize(lhs_spaces.size() + 1);
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent left_ready = 
          left_child->get_realm_index_space(lhs_spaces.back(),
                                            false/*tight*/);
        ApEvent right_ready = 
          right_child->get_realm_index_space(rhs_spaces.back(),
                                             false/*tight*/);
        if (left_ready.exists())
          preconditions.insert(left_ready);
        if (right_ready.exists())
          preconditions.insert(right_ready);
      }
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_DIFFERENCES);
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      const ApEvent precondition = Runtime::merge_events(NULL, preconditions);
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
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id,
                        precondition, result, DEP_PART_DIFFERENCES);
#endif
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(subspace_index < subspaces.size());
#endif
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[subspace_index++]))
          assert(false); // should never hit this
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
#define DIMFUNC(D1) \
        case D1: \
          { \
            const Realm::Matrix<D1,DIM,T> *transform =  \
              static_cast<const Realm::Matrix<D1,DIM,T>*>(tran); \
            const Realm::Rect<D1,T> *extent = \
              static_cast<const Realm::Rect<D1,T>*>(ext); \
            return create_by_restriction_helper<D1>(partition, *transform, \
                                                    *extent); \
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
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        Point<N,T> color;
        delinearize_color(*itr, color);
        // Copy the index space from the parent
        Realm::IndexSpace<M,T> child_is = parent_is;
        // Compute the new bounds and intersect it with the parent bounds
        child_is.bounds =
          parent_is.bounds.intersection(extent + transform * color);
        // Get the appropriate child
        IndexSpaceNodeT<M,T> *child = 
          static_cast<IndexSpaceNodeT<M,T>*>(partition->get_child(*itr));
        // Then set the new index space
        if (child->set_realm_index_space(context->runtime->address_space, 
                                         child_is))
          assert(false); // should never hit this
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
      CreateByDomainHelper creator(this, partition, op, future_map, 
                                   perform_intersections);
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
      CreateByFieldHelper creator(this,op,partition,instances,instances_ready);
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
#ifdef DEBUG_LEGION
      assert(partition->color_space == this);
#endif
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
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        const DomainPoint color = delinearize_color_to_point(*itr);
        FutureImpl *future = future_map->find_local_future(color);
#ifdef DEBUG_LEGION
        assert(future != NULL);
#endif
        IndexSpaceNodeT<DIM,T> *child = 
          static_cast<IndexSpaceNodeT<DIM,T>*>(
              partition->get_child(*itr));
        Realm::IndexSpace<DIM,T> child_space;
        size_t future_size = 0;
        const Domain *domain = static_cast<const Domain*>(
            future->find_internal_buffer(op->get_context(), future_size));
        if (future_size != sizeof(Domain))
          REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_BY_DOMAIN_VALUE,
              "An invalid future size was found in a partition by domain "
              "call. All futures must contain Domain objects.")
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
        if (child->set_realm_index_space(context->runtime->address_space,
                                         child_space))
          assert(false); // should never hit this
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
      std::map<DomainPoint,FutureImpl*> futures;
      future_map->get_all_futures(futures);
      // Make all the entries for the color space
      for (Realm::IndexSpaceIterator<COLOR_DIM,COLOR_T> 
            rect_iter(realm_color_space); rect_iter.valid; rect_iter.step())
      {
        for (Realm::PointInRectIterator<COLOR_DIM,COLOR_T> 
              itr(rect_iter.rect); itr.valid; itr.step())
        {
          const DomainPoint key(Point<COLOR_DIM,COLOR_T>(itr.p));
          std::map<DomainPoint,FutureImpl*>::const_iterator finder = 
            futures.find(key);
          if (finder == futures.end())
            REPORT_LEGION_ERROR(ERROR_MISSING_PARTITION_BY_WEIGHT_COLOR,
                "A partition by weight call is missing an entry for a "
                "color in the color space. All colors must be present.")
          FutureImpl *future = finder->second;
          size_t future_size = 0;
          const void *data =
            future->find_internal_buffer(op->get_context(), future_size);
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
            weights[color_index] = *(static_cast<const int*>(data));
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
            long_weights[color_index] = *(static_cast<const size_t*>(data));
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
      LegionSpy::log_deppart_events(op->get_unique_op_id(), expr_id, ready,
                                    result, DEP_PART_WEIGHTS);
#endif
      // Iterate the local colors and destroy any that we don't use
      unsigned next = 0;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        // Find the color
        std::vector<LegionColor>::iterator finder =
          std::lower_bound(child_colors.begin(), child_colors.end(), *itr);
#ifdef DEBUG_LEGION
        assert(finder != child_colors.end());
        assert(*finder == *itr);
#endif
        const unsigned offset = std::distance(child_colors.begin(), finder);
#ifdef DEBUG_LEGION
        assert(next <= offset);
#endif
        while (next < offset)
          subspaces[next++].destroy();
        IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[next++]))
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
      std::vector<Realm::Point<COLOR_DIM,COLOR_T> > colors;
      std::vector<LegionColor> child_colors;
      const TypeTag color_type = color_space->handle.get_type_tag();
      const size_t num_colors = realm_color_space.volume();
      colors.resize(num_colors);
      child_colors.resize(num_colors);
      unsigned index = 0;
      for (Realm::IndexSpaceIterator<COLOR_DIM,COLOR_T> 
            rect_iter(realm_color_space); rect_iter.valid; rect_iter.step())
      {
        for (Realm::PointInRectIterator<COLOR_DIM,COLOR_T> 
              itr(rect_iter.rect); itr.valid; itr.step(), index++)
        {
#ifdef DEBUG_LEGION
          assert(index < colors.size());
#endif
          colors[index] = itr.p;
          child_colors[index] = color_space->linearize_color(&itr.p,color_type);
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
#ifdef DEBUG_LEGION
      assert(child_colors.size() == subspaces.size());
#endif
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
                                    precondition, result, DEP_PART_BY_FIELD);
#endif
      // Update the children with the names of their subspaces 
      for (unsigned idx = 0; idx < child_colors.size(); idx++)
      {
        IndexSpaceNodeT<DIM,T> *child = static_cast<IndexSpaceNodeT<DIM,T>*>(
                                  partition->get_child(child_colors[idx]));
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
                                                    ApEvent instances_ready,
                                                    ShardID shard,
                                                    size_t total_shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByImageHelper creator(this, op, partition, projection, instances, 
                                  instances_ready, shard, total_shards);
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
                                                    ApEvent instances_ready,
                                                    ShardID shard,
                                                    size_t total_shards)
    //--------------------------------------------------------------------------
    {
      std::vector<Realm::IndexSpace<DIM2,T2> > sources; 
      std::vector<LegionColor> child_colors;
      const size_t volume = projection->total_children;
      if (total_shards > 1)
      {
        const size_t max_children = (volume + total_shards - 1) / total_shards;
        sources.reserve(max_children);
        child_colors.reserve(max_children);
      }
      else
      {
        sources.reserve(volume);
        child_colors.reserve(volume);
      }
      // Get the index spaces of the projection partition
      std::set<ApEvent> preconditions; 
      // Always use the partitions color space
      for (ColorSpaceIterator itr(partition, shard, total_shards); itr; itr++)
      {
        child_colors.push_back(*itr);
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM2,T2> *child = 
         static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(*itr));
        sources.resize(sources.size() + 1);
        ApEvent ready = child->get_realm_index_space(sources.back(),
                                                     false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
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
#ifdef DEBUG_LEGION
      // This should be true after the call
      assert(child_colors.size() == subspaces.size());
#endif
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
                                    precondition, result, DEP_PART_BY_IMAGE);
#endif
      // Update the child subspaces of the image
      for (unsigned idx = 0; idx < child_colors.size(); idx++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM1,T1> *child = 
          static_cast<IndexSpaceNodeT<DIM1,T1>*>(
              partition->get_child(child_colors[idx]));
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
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_image_range(Operation *op,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready,
                                                    ShardID shard,
                                                    size_t total_shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByImageRangeHelper creator(this, op, partition, projection,
                       instances, instances_ready, shard, total_shards);
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
                                                    ApEvent instances_ready,
                                                    ShardID shard,
                                                    size_t total_shards)
    //--------------------------------------------------------------------------
    {
      std::vector<Realm::IndexSpace<DIM2,T2> > sources; 
      std::vector<LegionColor> child_colors;
      const size_t volume = projection->total_children;
      if (total_shards > 1)
      {
        const size_t max_children = (volume + total_shards - 1) / total_shards;
        sources.reserve(max_children);
        child_colors.reserve(max_children);
      }
      else
      {
        sources.reserve(volume);
        child_colors.reserve(volume);
      }
      // Get the index spaces of the projection partition
      std::set<ApEvent> preconditions;
      // Always use the partitions color space
      for (ColorSpaceIterator itr(partition, shard, total_shards); itr; itr++)
      {
        child_colors.push_back(*itr);
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM2,T2> *child = 
         static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(*itr));
        sources.resize(sources.size() + 1);
        ApEvent ready = child->get_realm_index_space(sources.back(),
                                                     false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
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
#ifdef DEBUG_LEGION
      // Should be true after the call
      assert(subspaces.size() == child_colors.size());
#endif
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
                      precondition, result, DEP_PART_BY_IMAGE_RANGE);
#endif
      // Update the child subspaces of the image
      for (unsigned idx = 0; idx < child_colors.size(); idx++)
      {
        IndexSpaceNodeT<DIM1,T1> *child = 
           static_cast<IndexSpaceNodeT<DIM1,T1>*>(
               partition->get_child(child_colors[idx]));
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
                                targets(projection->total_children);
      std::set<ApEvent> preconditions;
      // Always use the partitions color space
      unsigned index = 0;
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM2,T2> *child = 
         static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(index < targets.size());
#endif
        ApEvent ready = child->get_realm_index_space(targets[index++],
                                                     false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
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
                                    precondition, result, DEP_PART_BY_PREIMAGE);
#endif
      // Update the child subspace of the preimage
      index = 0;
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM1,T1> *child = 
         static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(index < subspaces.size());
#endif
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[index++]))
          assert(false); // should never hit this
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
                                targets(projection->total_children);
      std::set<ApEvent> preconditions;
      unsigned index = 0;
      // Always use the partitions color space
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM2,T2> *child = 
         static_cast<IndexSpaceNodeT<DIM2,T2>*>(projection->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(index < targets.size());
#endif
        ApEvent ready = child->get_realm_index_space(targets[index++],
                                                     false/*tight*/);
        if (ready.exists())
          preconditions.insert(ready);
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
                  precondition, result, DEP_PART_BY_PREIMAGE_RANGE);
#endif
      // Update the child subspace of the preimage
      index = 0;
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM1,T1> *child = 
         static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
#ifdef DEBUG_LEGION
        assert(index < subspaces.size());
#endif
        if (child->set_realm_index_space(context->runtime->address_space,
                                         subspaces[index++]))
          assert(false); // should never hit this
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
                                    precondition, result, DEP_PART_ASSOCIATION);
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
                                   const Realm::ProfilingRequestSet &requests,
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
      PhysicalInstance result;
      ready_event = ApEvent(PhysicalInstance::create_external_instance(result, 
          res.suggested_memory(), ilg, res, requests));
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance IndexSpaceNodeT<DIM,T>::create_hdf5_instance(
                                    const char *file_name,
                                    const Realm::ProfilingRequestSet &requests,
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
    ApEvent IndexSpaceNodeT<DIM,T>::issue_fill(Operation *op,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                                 UniqueID fill_uid,
                                 FieldSpace handle,
                                 RegionTreeID tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent unique_event,
                                 CollectiveKind collective,
                                 int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (precondition.exists() && space_ready.exists())
        return issue_fill_internal(context, op, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
            Runtime::merge_events(&trace_info, space_ready, precondition),
            pred_guard, unique_event, collective, priority, replay);
      else if (space_ready.exists())
        return issue_fill_internal(context, op, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   space_ready, pred_guard, unique_event,
                                   collective, priority, replay);
      else
        return issue_fill_internal(context, op, local_space, trace_info, 
                                   dst_fields, fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   precondition, pred_guard, unique_event,
                                   collective, priority, replay);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::issue_copy(Operation *op,
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &dst_fields,
                                 const std::vector<CopySrcDstField> &src_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                 RegionTreeID src_tree_id,
                                 RegionTreeID dst_tree_id,
#endif
                                 ApEvent precondition, PredEvent pred_guard,
                                 LgEvent src_unique, LgEvent dst_unique,
                                 CollectiveKind collective,
                                 int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent space_ready = get_realm_index_space(local_space, true/*tight*/);
      if (precondition.exists() && space_ready.exists())
        return issue_copy_internal(context, op, local_space, trace_info,
            dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
            src_tree_id, dst_tree_id,
#endif
            Runtime::merge_events(&trace_info, space_ready, precondition),
            pred_guard, src_unique, dst_unique, collective, priority, replay);
      else if (space_ready.exists())
        return issue_copy_internal(context, op, local_space, trace_info, 
                dst_fields, src_fields, reservations, 
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                space_ready, pred_guard, src_unique, dst_unique,
                collective, priority, replay);
      else
        return issue_copy_internal(context, op, local_space, trace_info, 
                dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
                src_tree_id, dst_tree_id,
#endif
                precondition, pred_guard, src_unique, dst_unique,
                collective, priority, replay);
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
                                    bool compact, void **piece_list,
                                    size_t *piece_list_size, size_t *num_pieces)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> local_is;
      ApEvent space_ready = get_realm_index_space(local_is, true/*tight*/);
      if (space_ready.exists())
        space_ready.wait();
      return create_layout_internal(local_is, constraints,field_ids,field_sizes,
                              compact, piece_list, piece_list_size, num_pieces);
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
    KDTree* IndexSpaceNodeT<DIM,T>::get_sparsity_map_kd_tree(void)
    //--------------------------------------------------------------------------
    {
      return get_sparsity_map_kd_tree_internal<DIM,T>();
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

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpace IndexSpaceNodeT<DIM,T>::create_shard_space(
       ShardingFunction *func, ShardID shard, IndexSpace shard_space,
       const Domain &shard_domain, const std::vector<DomainPoint> &shard_points,
       Provenance *provenance)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      Domain sharding_domain;
      if (shard_space != handle)
        context->find_launch_space_domain(shard_space, sharding_domain);
      else
        sharding_domain = local_space;
      std::vector<Realm::Point<DIM,T> > index_points; 
      if (!func->functor->is_invertible())
      {
        for (Realm::IndexSpaceIterator<DIM,T> rect_itr(local_space); 
              rect_itr.valid; rect_itr.step())
        {
          for (Realm::PointInRectIterator<DIM,T> itr(rect_itr.rect);
                itr.valid; itr.step())
          {
            const ShardID point_shard = 
             func->find_owner(DomainPoint(Point<DIM,T>(itr.p)),sharding_domain);
            if (point_shard == shard)
              index_points.push_back(itr.p);
          }
        }
      }
      else
      {
        std::vector<DomainPoint> domain_points;
        if (func->use_points)
          func->functor->invert_points(shard_points[shard], shard_points,
             shard_domain, Domain(local_space), sharding_domain, domain_points);
        else
          func->functor->invert(shard, sharding_domain, Domain(local_space),
                                shard_points.size(), domain_points);
        index_points.resize(domain_points.size());
        for (unsigned idx = 0; idx < domain_points.size(); idx++)
          index_points[idx] = Point<DIM,coord_t>(domain_points[idx]);
      }
      if (index_points.empty())
        return IndexSpace::NO_SPACE;
      // Another useful case is if all the points are in the shard then
      // we can return ourselves as the result
      if (index_points.size() == get_volume())
        return handle;
      Realm::IndexSpace<DIM,T> realm_is(index_points);
      const Domain domain((DomainT<DIM,T>(realm_is)));
      return context->runtime->find_or_create_index_slice_space(domain, 
                                    handle.get_type_tag(), provenance);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::compute_range_shards(ShardingFunction *func,
           IndexSpace shard_space, const std::vector<DomainPoint> &shard_points,
           const Domain &shard_domain, std::set<ShardID> &range_shards)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      Domain sharding_domain;
      if (shard_space.exists() && shard_space != handle)
        context->find_launch_space_domain(shard_space, sharding_domain);
      else
        sharding_domain = local_space;
      if (!func->functor->is_invertible())
      {
        const size_t max_size = get_volume();
        for (Realm::IndexSpaceIterator<DIM,T> rect_itr(local_space); 
              rect_itr.valid; rect_itr.step())
        {
          for (Realm::PointInRectIterator<DIM,T> itr(rect_itr.rect);
                itr.valid; itr.step())
          {
            const ShardID point_shard = 
             func->find_owner(DomainPoint(Point<DIM,T>(itr.p)),sharding_domain);
            if (range_shards.insert(point_shard).second && 
                (range_shards.size() == max_size))
              break;
          }
          if (range_shards.size() == max_size)
            break;
        }
      }
      else
      {
        for (ShardID shard = 0; shard < shard_points.size(); shard++)
        {
          std::vector<DomainPoint> domain_points;
          if (func->use_points)
            func->functor->invert_points(shard_points[shard], shard_points,
                shard_domain,Domain(local_space),sharding_domain,domain_points);
          else
            func->functor->invert(shard, Domain(local_space), sharding_domain,
                                  shard_points.size(), domain_points);
          if (!domain_points.empty())
            range_shards.insert(shard);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Templated Linearized Color Space
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceLinearizationT<DIM,T>::ColorSpaceLinearizationT(
                                                   const DomainT<DIM,T> &domain)
    //--------------------------------------------------------------------------
    {
      // Check for the common case of a dense color space that we can traverse
      // with a single Morton curve
      if (domain.dense())
      {
        unsigned interesting_count = 0;
        int interesting_dims[DIM] = { -1 };
        size_t largest_extent= 0;
        for (int i = 0; i < DIM; i++)
        {
          size_t extent = (domain.bounds.hi[i] - domain.bounds.lo[i]) + 1;
#ifdef DEBUG_LEGION
          assert(extent > 0);
          assert(extent < SIZE_MAX);
#endif
          if (extent == 1)
            continue;
          interesting_dims[interesting_count++] = i;
          if (largest_extent < extent)
            largest_extent = extent;
        }
        if ((interesting_count == 0) || (interesting_count == 1))
        {
          // This is a rectangle that represents a single point or a 
          // "pencil" in N-dimensions and therefore doesn't need a Morton curve
          morton_tiles.push_back(new MortonTile(domain.bounds,
                interesting_count, interesting_dims, 0/*order*/));
          kdtree = NULL;
          return;
        }
        // Find the least power of 2 >= extent
        unsigned power2 = largest_extent - 1;
        constexpr unsigned log2bits = STATIC_LOG2(8*sizeof(power2));
        for (unsigned idx = 0; idx < log2bits; idx++)
          power2 = power2 | (power2 >> (1 << idx));
        power2++;
        // Take the log to get the number of bits to represent it
        unsigned order = STATIC_LOG2(power2);
        // Check to see if we can fit this in the available bits
        const size_t max_morton = 8*sizeof(LegionColor) / interesting_count;
        if (order <= max_morton)
        {
          // It fits so we just need a single MortonTile
          morton_tiles.push_back(new MortonTile(domain.bounds,
                interesting_count, interesting_dims, order));
          kdtree = NULL;
          return;
        }
      }
      // Iterate over the rectangles of the domain
      std::vector<std::pair<Rect<DIM,T>,MortonTile*> > tiles;
      for (RectInDomainIterator<DIM,T> itr(domain); itr(); itr++)
      {
        // Find the extent of the smallest dimension of the rectangle
        // that is > 1. Any dimensions that have extent one are not interesting
        // and will be ignored by the Morton curve
        unsigned interesting_count = 0;
        int interesting_dims[DIM] = { -1 };
        size_t smallest_extent = SIZE_MAX;
        for (int i = 0; i < DIM; i++)
        {
          size_t extent = (itr->hi[i] - itr->lo[i]) + 1;
#ifdef DEBUG_LEGION
          assert(extent > 0);
          assert(extent < SIZE_MAX);
#endif
          if (extent == 1)
            continue;
          interesting_dims[interesting_count++] = i;
          if (extent < smallest_extent)
            smallest_extent = extent;
        }
        if ((interesting_count == 0) || (interesting_count == 1))
        {
          // This is a rectangle that represents a single point or a 
          // "pencil" in N-dimensions and therefore doesn't need a Morton curve
          tiles.push_back(std::make_pair(*itr, new MortonTile(*itr,
                  interesting_count, interesting_dims, 0/*order*/)));
          continue;
        }
        // Find the least power of 2 >= extent
        size_t power2 = smallest_extent - 1;
        constexpr unsigned log2bits = STATIC_LOG2(8*sizeof(power2));
        for (unsigned idx = 0; idx < log2bits; idx++)
          power2 = power2 | (power2 >> (1 << idx));
        power2++;
        // Take the log to get the number of bits to represent it
        unsigned order = STATIC_LOG2(power2);
        // For small dimensions over-approximating is not too bad, but in
        // larger dimensions it can become expensive as the amount of waste
        // is proportion to 2^DIM, so we deicde that for more than four 
        // dimensions we we'll look for the largest power of 2 <= extent
        if (DIM > 4)
        {
          // This is the least power of 2 >= extent, check if it is the
          // same as the extent, if not subtract by one to get the 
          // largest power of 2 <= the extent
#ifdef DEBUG_LEGION
          assert(smallest_extent <= (1ULL << order));
#endif
          if (smallest_extent != (1ULL << order))
            order--;
        }
        // If this is bigger than the largest order we support for the
        // given number of interesting dimensions then bound it
        const size_t max_morton = 8*sizeof(LegionColor) / interesting_count;
        if (order > max_morton)
          order = max_morton;
        // Tile the rectangle
        // We could do this in a Morton-ordered way too, but for now we're
        // just going to let the KD-tree figure out the right way to sort
        // things in the case that we have to make lots of tiles
        // The KD-tree sorting algorithm should be good enough to give us
        // locality where we actually need it
        Point<DIM,T> strides = Point<DIM,T>::ZEROES();
        for (unsigned idx = 0; idx < interesting_count; idx++)
          strides[interesting_dims[idx]] = (1 << order);
        Point<DIM,T> lower = itr->lo;
        bool done = false;
        while (!done)
        {
          Rect<DIM,T> next(lower, lower + strides);
          if (interesting_count < DIM)
          {
            for (unsigned idx = 0; idx < interesting_count; idx++)
              next.hi[interesting_dims[idx]] -= 1;
          }
          else
            next.hi -= Point<DIM,T>::ONES();
          next = itr->intersection(next);
#ifdef DEBUG_LEGION
          assert(next.volume() > 0);
#endif
          tiles.push_back(std::make_pair(next, new MortonTile(next,
                  interesting_count, interesting_dims, order)));
          done = true;
          for (unsigned idx = 0; idx < interesting_count; idx++)
          {
            int dim = interesting_dims[idx];
            lower[dim] += strides[dim];
            if (lower[dim] <= itr->hi[dim])
            {
              done = false;
              break;
            }
            // Otherwise reset this dimension and ripple-carry add
            lower[dim] = itr->lo[dim];
          }
        }
      }
      // Put the Morton Tiles in a KD-tree for fast lookups
      kdtree = new KDNode<DIM,T,MortonTile*>(domain.bounds, tiles);
      // Assign an order to the Morton Tiles based on their order in the
      // KD-tree which should give us good locality between the tiles
      kdtree->record_inorder_traversal(morton_tiles);
      // Now we can go through and compute the color offsets for each tile
      LegionColor offset = 0;
      color_offsets.resize(morton_tiles.size());
      for (unsigned idx = 0; idx < morton_tiles.size(); idx++)
      {
        color_offsets[idx] = offset;
        MortonTile *tile = morton_tiles[idx];
        tile->index = idx;
        LegionColor new_offset = offset;
        if (tile->morton_order == 0)
        {
          if (tile->interesting_count == 1)
          {
            int dim = tile->interesting_dims[0];
            new_offset += ((tile->bounds.hi[dim] - tile->bounds.lo[dim]) + 1);
          }
          else // single element
            new_offset++;
        }
        else
          new_offset += (1 << (tile->morton_order * tile->interesting_count));
        // Check for overflow which would be very bad
        if (new_offset <= offset)
          REPORT_LEGION_FATAL(LEGION_FATAL_MORTON_TILING_FAILURE,
              "Failure during Morton tiling of color space. Please "
              "report this issue as a bug and provide a reproducer.")
        offset = new_offset;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceLinearizationT<DIM,T>::~ColorSpaceLinearizationT(void)
    //--------------------------------------------------------------------------
    {
      if (kdtree != NULL)
        delete kdtree;
      for (unsigned idx = 0; idx < morton_tiles.size(); idx++)
        delete morton_tiles[idx];
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceLinearizationT<DIM,T>::get_max_linearized_color(void) 
                                                                           const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!morton_tiles.empty());
#endif
      MortonTile *last = morton_tiles.back();
      LegionColor max_color = last->get_max_linearized_color();
      if (!color_offsets.empty())
        max_color += color_offsets.back();
      return max_color;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceLinearizationT<DIM,T>::linearize(
                                                const Point<DIM,T> &point) const
    //--------------------------------------------------------------------------
    {
      if (morton_tiles.size() > 1)
      {
        // Find the Morton Tile that contains the point
        MortonTile *tile = kdtree->find(point);
#ifdef DEBUG_LEGION
        assert(tile != NULL);
#endif
        return color_offsets[tile->index] + tile->linearize(point);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!morton_tiles.empty());
#endif
        MortonTile *tile = morton_tiles.front();
        return tile->linearize(point);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void ColorSpaceLinearizationT<DIM,T>::delinearize(LegionColor color,
                                                      Point<DIM,T> &point) const
    //--------------------------------------------------------------------------
    {
      if ((morton_tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder = 
          std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
#ifdef DEBUG_LEGION
        assert(finder != color_offsets.begin());
#endif
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
#ifdef DEBUG_LEGION
        assert(index < morton_tiles.size());
        assert(index < color_offsets.size());
#endif
        color -= color_offsets[index];
        morton_tiles[index]->delinearize(color, point);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!morton_tiles.empty());
#endif
        MortonTile *tile = morton_tiles.front();
        tile->delinearize(color, point);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool ColorSpaceLinearizationT<DIM,T>::contains_color(LegionColor color)const
    //--------------------------------------------------------------------------
    {
      if ((morton_tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder = 
          std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
#ifdef DEBUG_LEGION
        assert(finder != color_offsets.begin());
#endif
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
#ifdef DEBUG_LEGION
        assert(index < morton_tiles.size());
        assert(index < color_offsets.size());
#endif
        color -= color_offsets[index];
        return morton_tiles[index]->contains_color(color);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!morton_tiles.empty());
#endif
        MortonTile *tile = morton_tiles.front();
        return tile->contains_color(color);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t ColorSpaceLinearizationT<DIM,T>::compute_color_offset(
                                                        LegionColor color) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(contains_color(color));
#endif
      if ((morton_tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder = 
          std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
#ifdef DEBUG_LEGION
        assert(finder != color_offsets.begin());
#endif
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
#ifdef DEBUG_LEGION
        assert(index < morton_tiles.size());
        assert(index < color_offsets.size());
#endif
        color -= color_offsets[index];
        size_t offset = morton_tiles[index]->compute_color_offset(color);
        // count all the points in the prior morton tiles
        for (unsigned idx = 0; idx < index; idx++)
          offset += morton_tiles[idx]->bounds.volume();
        return offset;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!morton_tiles.empty());
#endif
        MortonTile *tile = morton_tiles.front();
        return tile->compute_color_offset(color);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceLinearizationT<DIM,T>::MortonTile::
                                            get_max_linearized_color(void) const
    //--------------------------------------------------------------------------
    {
      if (interesting_count < 2)
        return bounds.volume();
      else
        return (1 << (morton_order * interesting_count));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceLinearizationT<DIM,T>::MortonTile::linearize(
                                                const Point<DIM,T> &point) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(bounds.contains(point));
#endif
      if (morton_order == 0)
      {
#ifdef DEBUG_LEGION
        assert((interesting_count == 0) || (interesting_count == 1));
#endif
        // No need for a Morton curve in these case of 0 or 1 interesting dims
        if (interesting_count == 0)
          return 0;
        return point[interesting_dims[0]] - bounds.lo[interesting_dims[0]];
      }
      else if (interesting_count < DIM)
      {
        // Slow path, not all dimensions are interesting
        // Pull them down to the localized dimensions
        unsigned coords[DIM];
        for (unsigned i = 0; i < interesting_count; i++)
          coords[i] = point[interesting_dims[i]]-bounds.lo[interesting_dims[i]];
        // Shift the bits for each of the coordinates
        // We could do this more efficiently by moving groups
        // of bits by the same offsets but that's more complicated
        // and error prone so we don't do it currently
        LegionColor codes[DIM] = { 0 };
        unsigned andbit = 1; unsigned shift = 0; 
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (unsigned i = 0; i < interesting_count; i++)
            codes[i] |= (LegionColor)(coords[i] & andbit) << shift;
          andbit <<= 1;
          shift += (interesting_count - 1);
        }
        // Interleave the bits from each coordinate
        LegionColor result = 0;
        for (unsigned i = 0; i < interesting_count; i++)
          result |= (codes[i] << i);
        return result;
      }
      else
      {
        // Fast path, all dimensions are interesting
        unsigned coords[DIM];
        for (int i = 0; i < DIM; i++)
          coords[i] = point[i] - bounds.lo[i];
        // Shift the bits for each of the coordinates
        // We could do this more efficiently by moving groups
        // of bits by the same offsets but that's more complicated
        // and error prone so we don't do it currently
        LegionColor codes[DIM] = { 0 };
        unsigned andbit = 1, shift = 0;
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (int i = 0; i < DIM; i++)
            codes[i] |= (LegionColor)(coords[i] & andbit) << shift;
          andbit <<= 1;
          shift += (DIM - 1);
        }
        // Interleave the bits from each coordinate
        LegionColor result = 0;
        for (int i = 0; i < DIM; i++)
          result |= (codes[i] << i);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void ColorSpaceLinearizationT<DIM,T>::MortonTile::delinearize(
                                   LegionColor color, Point<DIM,T> &point) const
    //--------------------------------------------------------------------------
    {
      point = Point<DIM,T>::ZEROES(); 
      if (morton_order == 0)
      {
#ifdef DEBUG_LEGION
        assert((interesting_count == 0) || (interesting_count == 1));
#endif
        if (interesting_count == 1)
          point[interesting_dims[0]] = color;
      }
      else if (interesting_count < DIM)
      {
        // Slow path, not all dimensions are interesting
        unsigned coords[DIM] = { 0 };
        unsigned selector = 0, shift = 0;
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (unsigned i = 0; i < interesting_count; i++)
            coords[i] |= (color & (1 << (selector + i))) >> (shift + i);
          selector += interesting_count;
          shift += (interesting_count - 1);
        }
        for (unsigned i = 0; i < interesting_count; i++)
          point[interesting_dims[i]] = coords[i];
      }
      else
      {
        unsigned coords[DIM] = { 0 };
        unsigned selector = 0, shift = 0;
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (int i = 0; i < DIM; i++)
            coords[i] |= (color & (1 << (selector + i))) >> (shift + i);
          selector += DIM;
          shift += (DIM-1);
        }
        for (int i = 0; i < DIM; i++)
          point[i] = coords[i];
      }
      point += bounds.lo;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool ColorSpaceLinearizationT<DIM,T>::MortonTile::contains_color(
                                                        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      if (get_max_linearized_color() <= color)
        return false;
      Point<DIM,T> point;
      delinearize(color, point);
      return bounds.contains(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t ColorSpaceLinearizationT<DIM,T>::MortonTile::compute_color_offset(
                                                        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      // Scan through all the points in this tile up to the color and check
      // that they are all in bounds
      size_t offset = 0;
      for (LegionColor c = 0; c < color; c++)
        if (contains_color(c))
          offset++;
      return offset;
    }

    /////////////////////////////////////////////////////////////
    // Templated Linearized Color Space (for DIM=1)
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    ColorSpaceLinearizationT<1,T>::ColorSpaceLinearizationT(
                                                     const DomainT<1,T> &domain)
    //--------------------------------------------------------------------------
    {
      if (!domain.dense())
      {
        std::map<T,size_t> tile_sizes;
        for (RectInDomainIterator<1,T> itr(domain); itr(); itr++)
          tile_sizes[itr->lo[0]] = (itr->hi[0] - itr->lo[0]) + 1;
        LegionColor offset = 0;
        tiles.reserve(tile_sizes.size());
        extents.reserve(tile_sizes.size());
        color_offsets.reserve(tiles.size());
        for (typename std::map<T,size_t>::const_iterator it =
              tile_sizes.begin(); it != tile_sizes.end(); it++)
        {
          tiles.push_back(it->first);
          extents.push_back(it->second);
          color_offsets.push_back(offset);
          offset += it->second;
        }
      }
      else
      {
        tiles.push_back(domain.bounds.lo[0]); 
        extents.push_back(domain.bounds.volume());
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    LegionColor ColorSpaceLinearizationT<1,T>::get_max_linearized_color(void)
                                                                           const
    //--------------------------------------------------------------------------
    {
      LegionColor max_color = extents.back();
      if (!color_offsets.empty())
        max_color += color_offsets.back();
      return max_color;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    LegionColor ColorSpaceLinearizationT<1,T>::linearize(
                                                  const Point<1,T> &point) const
    //--------------------------------------------------------------------------
    {
      if (tiles.size() > 1)
      {
        typename std::vector<T>::const_iterator finder = 
          std::upper_bound(tiles.begin(), tiles.end(), point[0]);
        if (finder != tiles.begin())
        {
          finder = std::prev(finder);
          unsigned index = std::distance(tiles.begin(), finder);
          return color_offsets[index] + (point[0] - tiles[index]);
        }
      }
#ifdef DEBUG_LEGION
      assert(!tiles.empty());
#endif
      return (point[0] - tiles.front());
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void ColorSpaceLinearizationT<1,T>::delinearize(
                                     LegionColor color, Point<1,T> &point) const
    //--------------------------------------------------------------------------
    {
      if ((tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder =
          std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
#ifdef DEBUG_LEGION
        assert(finder != color_offsets.begin());
#endif
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
        point[0] = tiles[index] + (color - *finder);
      }
      else
        point[0] = tiles.front() + color;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool ColorSpaceLinearizationT<1,T>::contains_color(LegionColor color) const
    //--------------------------------------------------------------------------
    {
      return (color < get_max_linearized_color());
    }

    //--------------------------------------------------------------------------
    template<typename T>
    size_t ColorSpaceLinearizationT<1,T>::compute_color_offset(
                                                        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      // Colors are dense here so colors are their own offsets
      return color;
    }

    /////////////////////////////////////////////////////////////
    // KD Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline KDNode<DIM,T>* KDTree::as_kdnode(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      KDNode<DIM,T> *result = dynamic_cast<KDNode<DIM,T>*>(this);
      assert(result != NULL);
      return result;
#else
      return static_cast<KDNode<DIM,T>*>(this);
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    KDNode<DIM,T,RT>::KDNode(const Rect<DIM,T> &b,
                             std::vector<std::pair<Rect<DIM,T>,RT> > &subrects)
      : bounds(b), left(NULL), right(NULL)
    //--------------------------------------------------------------------------
    {
      // This is the base case
      if (subrects.size() <= LEGION_MAX_BVH_FANOUT)
      {
        rects.swap(subrects);
        return;
      }
      // If we have sub-optimal bad sets we will track them here
      // so we can iterate through other dimensions to look for
      // better splitting planes
      int best_dim = -1;
      float best_cost = 2.f; // worst possible cost
      Rect<DIM,T> best_left_bounds, best_right_bounds;
      std::vector<std::pair<Rect<DIM,T>,RT> > best_left_set, best_right_set;
      for (int d = 0; d < DIM; d++)
      {
        // Try to compute a splitting plane for this dimension
        // Count how many rectangles start and end at each location
        std::map<std::pair<coord_t,bool/*stop*/>,unsigned> forward_lines;
        std::map<std::pair<coord_t,bool/*start*/>,unsigned> backward_lines;
        for (unsigned idx = 0; idx < subrects.size(); idx++)
        {
          const Rect<DIM,T> &subset_bounds = subrects[idx].first;
          // Start forward
          std::pair<coord_t,bool> start_key(subset_bounds.lo[d],false);
          std::map<std::pair<coord_t,bool>,unsigned>::iterator finder =
            forward_lines.find(start_key);
          if (finder == forward_lines.end())
            forward_lines[start_key] = 1;
          else
            finder->second++;
          // Start backward 
          start_key.second = true;
          finder = backward_lines.find(start_key);
          if (finder == backward_lines.end())
            backward_lines[start_key] = 1;
          else
            finder->second++;
          // Stop forward
          std::pair<coord_t,bool> stop_key(subset_bounds.hi[d],true);
          finder = forward_lines.find(stop_key);
          if (finder == forward_lines.end())
            forward_lines[stop_key] = 1;
          else
            finder->second += 1;
          // Stop backward 
          stop_key.second = false;
          finder = backward_lines.find(stop_key);
          if (finder == backward_lines.end())
            backward_lines[stop_key] = 1;
          else
            finder->second++;
        }
        // Construct two lists by scanning from left-to-right and
        // from right-to-left of the number of rectangles that would
        // be inlcuded on the left or right side by each splitting plane
        std::map<coord_t,unsigned> lower_inclusive, upper_exclusive;
        unsigned count = 0;
        for (typename std::map<std::pair<coord_t,bool>,unsigned>::const_iterator
              it = forward_lines.begin(); it != forward_lines.end(); it++)
        {
          // Increment first for starts for inclusivity
          if (!it->first.second)
            count += it->second;
          // Always record the count for all splits
          lower_inclusive[it->first.first] = count;
        }
        // If all the lines exist at the same value
        // then we'll never have a splitting plane
        if (lower_inclusive.size() == 1)
          continue;
        count = 0;
        for (typename std::map<
              std::pair<coord_t,bool>,unsigned>::const_reverse_iterator it = 
              backward_lines.rbegin(); it != backward_lines.rend(); it++)
        {
          // Always record the count for all splits
          upper_exclusive[it->first.first] = count;
          // Increment last for stops for exclusivity
          if (!it->first.second)
            count += it->second;
        }
#ifdef DEBUG_LEGION
        assert(lower_inclusive.size() == upper_exclusive.size());
#endif
        // We want to take the mini-max of the two numbers in order
        // to try to balance the splitting plane across the two sets
        T split = 0;
        unsigned split_max = subrects.size();
        for (std::map<coord_t,unsigned>::const_iterator it =
              lower_inclusive.begin(); it != lower_inclusive.end(); it++)
        {
          const unsigned lower = it->second;
          const unsigned upper = upper_exclusive[it->first];
          const unsigned max = (lower > upper) ? lower : upper;
          if (max < split_max)
          {
            split_max = max;
            split = it->first;
          }
        }
        // Check for the case where we can't find a splitting plane
        if (split_max == subrects.size())
          continue;
        // Sort the subsets into left and right
        Rect<DIM,T> left_bounds(bounds);
        Rect<DIM,T> right_bounds(bounds);
        left_bounds.hi[d] = split;
        right_bounds.lo[d] = split+1;
        std::vector<std::pair<Rect<DIM,T>,RT> > left_set, right_set;
        for (typename std::vector<std::pair<Rect<DIM,T>,RT> >::const_iterator
              it = subrects.begin(); it != subrects.end(); it++)
        {
          const Rect<DIM,T> left_rect = it->first.intersection(left_bounds);
          if (!left_rect.empty())
            left_set.push_back(std::make_pair(left_rect, it->second));
          const Rect<DIM,T> right_rect = it->first.intersection(right_bounds);
          if (!right_rect.empty())
            right_set.push_back(std::make_pair(right_rect, it->second));
        }
#ifdef DEBUG_LEGION
        assert(left_set.size() < subrects.size());
        assert(right_set.size() < subrects.size());
#endif
        // Compute the cost of this refinement
        // First get the percentage reductions of both sets
        float cost_left = float(left_set.size()) / float(subrects.size());
        float cost_right = float(right_set.size()) / float(subrects.size());
        // We want to give better scores to sets that are closer together
        // so we'll include the absolute value of the difference in the
        // two costs as part of computing the average cost
        // If the savings are identical then this will be zero extra cost
        // Note this cost metric should always produce values between
        // 1.0 and 2.0, with 1.0 being a perfect 50% reduction on each side
        float cost_diff = (cost_left < cost_right) ? 
          (cost_right - cost_left) : (cost_left - cost_right);
        float total_cost = (cost_left + cost_right + cost_diff);
#ifdef DEBUG_LEGION
        assert((1.f <= total_cost) && (total_cost <= 2.f));
#endif
        // Check to see if the cost is considered to be a "good" refinement
        // For now we'll say that this is a good cost if it is less than
        // or equal to 1.5, halfway between the range of costs from 1.0 to 2.0
        if ((total_cost <= 1.5f) && (total_cost < best_cost))
        {
          best_dim = d;
          best_cost = total_cost;
          best_left_set.swap(left_set);
          best_right_set.swap(right_set);
          best_left_bounds = left_bounds;
          best_right_bounds = right_bounds;
        }
      }
      // See if we had at least one good refinement
      if (best_dim >= 0)
      {
        // Always clear the old-subrects before recursing to reduce memory usage
        {
          std::vector<std::pair<Rect<DIM,T>,RT> > empty;
          empty.swap(subrects);
        }
        left = new KDNode<DIM,T,RT>(best_left_bounds, best_left_set); 
        right = new KDNode<DIM,T,RT>(best_right_bounds, best_right_set);
      }
      else
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_KDTREE_REFINEMENT_FAILED,
            "Failed to find a refinement for KD tree with %d dimensions "
            "and %zd rectangles. Please report your application to the "
            "Legion developers' mailing list.", DIM, subrects.size())
        // If we make it here then we couldn't find a splitting plane to refine
        // anymore so just record all the subrects as our rects
        rects.swap(subrects);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    KDNode<DIM,T,RT>::~KDNode(void)
    //--------------------------------------------------------------------------
    {
      if (left != NULL)
        delete left;
      if (right != NULL)
        delete right;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    void KDNode<DIM,T,RT>::find_interfering(const Rect<DIM,T> &test,
                                            std::set<RT> &interfering) const
    //--------------------------------------------------------------------------
    {
      if ((left != NULL) && left->bounds.overlaps(test))
        left->find_interfering(test, interfering);
      if ((right != NULL) && right->bounds.overlaps(test))
        right->find_interfering(test, interfering);
      for (typename std::vector<std::pair<Rect<DIM,T>,RT> >::
            const_iterator it = rects.begin(); it != rects.end(); it++)
        if (it->first.overlaps(test))
          interfering.insert(it->second);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    void KDNode<DIM,T,RT>::record_inorder_traversal(std::vector<RT> &out) const
    //--------------------------------------------------------------------------
    {
      if (left != NULL)
        left->record_inorder_traversal(out);
      for (typename std::vector<std::pair<Rect<DIM,T>,RT> >::
            const_iterator it = rects.begin(); it != rects.end(); it++)
        out.push_back(it->second);
      if (right != NULL)
        right->record_inorder_traversal(out);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, typename RT>
    RT KDNode<DIM,T,RT>::find(const Point<DIM,T> &point) const
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<std::pair<Rect<DIM,T>,RT> >::
            const_iterator it = rects.begin(); it != rects.end(); it++)
        if (it->first.contains(point))
          return it->second;
      if ((left != NULL) && left->bounds.contains(point))
        return left->find(point);
      if ((right != NULL) && right->bounds.contains(point))
        return right->find(point);
      // Should always find it currently
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    KDNode<DIM,T,void>::KDNode(const Rect<DIM,T> &b,
                               std::vector<Rect<DIM,T> > &subrects)
      : bounds(b), left(NULL), right(NULL)
    //--------------------------------------------------------------------------
    {
      // This is the base case
      if (subrects.size() <= LEGION_MAX_BVH_FANOUT)
      {
        rects.swap(subrects);
        return;
      }
      // If we have sub-optimal bad sets we will track them here
      // so we can iterate through other dimensions to look for
      // better splitting planes
      int best_dim = -1;
      float best_cost = 2.f; // worst possible cost
      Rect<DIM,T> best_left_bounds, best_right_bounds;
      std::vector<Rect<DIM,T> > best_left_set, best_right_set;
      for (int d = 0; d < DIM; d++)
      {
        // Try to compute a splitting plane for this dimension
        // Count how many rectangles start and end at each location
        std::map<std::pair<coord_t,bool/*stop*/>,unsigned> forward_lines;
        std::map<std::pair<coord_t,bool/*start*/>,unsigned> backward_lines;
        for (unsigned idx = 0; idx < subrects.size(); idx++)
        {
          const Rect<DIM,T> &subset_bounds = subrects[idx];
          // Start forward
          std::pair<coord_t,bool> start_key(subset_bounds.lo[d],false);
          std::map<std::pair<coord_t,bool>,unsigned>::iterator finder =
            forward_lines.find(start_key);
          if (finder == forward_lines.end())
            forward_lines[start_key] = 1;
          else
            finder->second++;
          // Start backward 
          start_key.second = true;
          finder = backward_lines.find(start_key);
          if (finder == backward_lines.end())
            backward_lines[start_key] = 1;
          else
            finder->second++;
          // Stop forward 
          std::pair<coord_t,bool> stop_key(subset_bounds.hi[d],true);
          finder = forward_lines.find(stop_key);
          if (finder == forward_lines.end())
            forward_lines[stop_key] = 1;
          else
            finder->second += 1;
          // Stop backward 
          stop_key.second = false;
          finder = backward_lines.find(stop_key);
          if (finder == backward_lines.end())
            backward_lines[stop_key] = 1;
          else
            finder->second++;
        }
        // Construct two lists by scanning from left-to-right and
        // from right-to-left of the number of rectangles that would
        // be inlcuded on the left or right side by each splitting plane
        std::map<coord_t,unsigned> lower_inclusive, upper_exclusive;
        unsigned count = 0;
        for (typename std::map<std::pair<coord_t,bool>,unsigned>::const_iterator
              it = forward_lines.begin(); it != forward_lines.end(); it++)
        {
          // Increment first for starts for inclusivity
          if (!it->first.second)
            count += it->second;
          // Always record the count for all splits
          lower_inclusive[it->first.first] = count;
        }
        // If all the lines exist at the same value
        // then we'll never have a splitting plane
        if (lower_inclusive.size() == 1)
          continue;
        count = 0;
        for (typename std::map<
              std::pair<coord_t,bool>,unsigned>::const_reverse_iterator it = 
              backward_lines.rbegin(); it != backward_lines.rend(); it++)
        {
          // Always record the count for all splits
          upper_exclusive[it->first.first] = count;
          // Increment last for stops for exclusivity
          if (!it->first.second)
            count += it->second;
        }
#ifdef DEBUG_LEGION
        assert(lower_inclusive.size() == upper_exclusive.size());
#endif
        // We want to take the mini-max of the two numbers in order
        // to try to balance the splitting plane across the two sets
        T split = 0;
        unsigned split_max = subrects.size();
        for (std::map<coord_t,unsigned>::const_iterator it = 
              lower_inclusive.begin(); it != lower_inclusive.end(); it++)
        {
          const unsigned left = it->second;
          const unsigned right = upper_exclusive[it->first];
          const unsigned max = (left > right) ? left : right;
          if (max < split_max)
          {
            split_max = max;
            split = it->first;
          }
        }
        // Check for the case where we can't find a splitting plane
        if (split_max == subrects.size())
          continue;
        // Sort the subsets into left and right
        Rect<DIM,T> left_bounds(bounds);
        Rect<DIM,T> right_bounds(bounds);
        left_bounds.hi[d] = split;
        right_bounds.lo[d] = split+1;
        std::vector<Rect<DIM,T> > left_set, right_set;
        for (typename std::vector<Rect<DIM,T> >::const_iterator it =
              subrects.begin(); it != subrects.end(); it++)
        {
          const Rect<DIM,T> left_rect = it->intersection(left_bounds);
          if (!left_rect.empty())
            left_set.push_back(left_rect);
          const Rect<DIM,T> right_rect = it->intersection(right_bounds);
          if (!right_rect.empty())
            right_set.push_back(right_rect);
        }
#ifdef DEBUG_LEGION
        assert(left_set.size() < subrects.size());
        assert(right_set.size() < subrects.size());
#endif
        // Compute the cost of this refinement
        // First get the percentage reductions of both sets
        float cost_left = float(left_set.size()) / float(subrects.size());
        float cost_right = float(right_set.size()) / float(subrects.size());
        // We want to give better scores to sets that are closer together
        // so we'll include the absolute value of the difference in the
        // two costs as part of computing the average cost
        // If the savings are identical then this will be zero extra cost
        // Note this cost metric should always produce values between
        // 1.0 and 2.0, with 1.0 being a perfect 50% reduction on each side
        float cost_diff = (cost_left < cost_right) ? 
          (cost_right - cost_left) : (cost_left - cost_right);
        float total_cost = (cost_left + cost_right + cost_diff);
#ifdef DEBUG_LEGION
        assert((1.f <= total_cost) && (total_cost <= 2.f));
#endif
        // Check to see if the cost is considered to be a "good" refinement
        // For now we'll say that this is a good cost if it is less than
        // or equal to 1.5, halfway between the range of costs from 1.0 to 2.0
        if ((total_cost <= 1.5f) && (total_cost < best_cost))
        {
          best_dim = d;
          best_cost = total_cost;
          best_left_set.swap(left_set);
          best_right_set.swap(right_set);
          best_left_bounds = left_bounds;
          best_right_bounds = right_bounds;
        }
      }
      // See if we had at least one good refinement
      if (best_dim >= 0)
      {
        // Always clear the old-subrects before recursing to reduce memory usage
        {
          std::vector<Rect<DIM,T> > empty;
          empty.swap(subrects);
        }
        left = new KDNode<DIM,T,void>(best_left_bounds, best_left_set); 
        right = new KDNode<DIM,T,void>(best_right_bounds, best_right_set);
      }
      else
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_KDTREE_REFINEMENT_FAILED,
            "Failed to find a refinement for KD tree with %d dimensions "
            "and %zd rectangles. Please report your application to the "
            "Legion developers' mailing list.", DIM, subrects.size())
        // If we make it here then we couldn't find a splitting plane to refine
        // anymore so just record all the subrects as our rects
        rects.swap(subrects);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    KDNode<DIM,T,void>::~KDNode(void)
    //--------------------------------------------------------------------------
    {
      if (left != NULL)
        delete left;
      if (right != NULL)
        delete right;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t KDNode<DIM,T,void>::count_rectangles(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = rects.size();
      if (left != NULL)
        result += left->count_rectangles();
      if (right != NULL)
        result += right->count_rectangles();
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t KDNode<DIM,T,void>::count_intersecting_points(
                                                  const Rect<DIM,T> &rect) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      for (typename std::vector<Rect<DIM,T> >::const_iterator it =
            rects.begin(); it != rects.end(); it++)
      {
        const Rect<DIM,T> overlap = it->intersection(rect);
        result += overlap.volume();
      }
      if (left != NULL)
      {
        Rect<DIM,T> left_overlap = rect.intersection(left->bounds);
        if (!left_overlap.empty())
          result += left->count_intersecting_points(left_overlap);
      }
      if (right != NULL)
      {
        Rect<DIM,T> right_overlap = rect.intersection(right->bounds);
        if (!right_overlap.empty())
          result += right->count_intersecting_points(right_overlap);
      }
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
          const PhysicalTraceInfo &trace_info, const bool replay,
          const bool recurrent_replay, const unsigned stage)
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
          // Note that for tracing, we can't actually defer this in 
          // the normal way because we need to actually get the real
          // finish event for the copy
          if (!trace_info.recording)
          {
            DeferCopyAcrossArgs args(this, op, pred_guard, copy_precondition,
                src_indirect_precondition, dst_indirect_precondition,
                trace_info, replay, recurrent_replay, stage);
            prev_done = runtime->issue_runtime_meta_task(args,
                LG_LATENCY_DEFERRED_PRIORITY, defer);
            return args.done_event;
          }
          else
            defer.wait();
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
                                        (idx < src_unique_events.size()) ? 
                                          src_unique_events[idx] :
                                          LgEvent::NO_LG_EVENT,
                                        src_fields[idx].indirect_index,
                                        dst_fields[idx].field_id,
                                        (idx < dst_unique_events.size()) ? 
                                          dst_unique_events[idx] : 
                                          LgEvent::NO_LG_EVENT,
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
      if (!replay)
        priority = op->add_copy_profiling_request(trace_info, requests,
                                          false/*fill*/, total_copies);
      // TODO: need to log unique IDs for instances here for copy indirections
      // The code right now is only correct for straight copy across
      if (runtime->profiler != NULL)
        runtime->profiler->add_copy_request(requests, op, total_copies);
      ApEvent copy_pre;
      if (pred_guard.exists())
        copy_pre =
          Runtime::merge_events(NULL, copy_precondition, ApEvent(pred_guard));
      else
        copy_pre = copy_precondition;
      // No need for tracing to know about the reservations
      for (std::map<Reservation,bool>::const_iterator it =
            reservations.begin(); it != reservations.end(); it++)
        copy_pre = Runtime::acquire_ap_reservation(it->first, 
                                        it->second, copy_pre);
      if (!indirections.empty())
      {
        if (individual_field_indexes.empty())
        {
          last_copy = ApEvent(copy_domain.copy(src_fields, dst_fields, 
                indirections, requests, copy_pre, priority));
          if (runtime->profiler != NULL)
            log_across_profiling(last_copy);
        }
        else
          last_copy = issue_individual_copies(copy_pre, requests);
          
      }
      else
      {
        last_copy = ApEvent(copy_domain.copy(src_fields, dst_fields,
              requests, copy_pre, priority));
        if (runtime->profiler != NULL)
          log_across_profiling(last_copy);
      }
      // Release any reservations
      for (std::map<Reservation,bool>::const_iterator it =
            reservations.begin(); it != reservations.end(); it++)
        Runtime::release_reservation(it->first, last_copy);
      if (pred_guard.exists())
      {
        // Protect against the poison from predication
        last_copy = Runtime::ignorefaults(last_copy);
        // Merge the preconditions into this result so they are still reflected
        // in the completion for this operation even if the operation ends up
        // being predicated out
        if (copy_precondition.exists())
        {
          if (last_copy.exists())
            last_copy = Runtime::merge_events(NULL,last_copy,copy_precondition);
          else
            last_copy = copy_precondition;
        }
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
                                   src_tree_id, dst_tree_id, copy_precondition,
                                   last_copy, COLLECTIVE_NONE);
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          LegionSpy::log_copy_field(last_copy, src_fields[idx].field_id,
                                    src_unique_events[idx],
                                    dst_fields[idx].field_id,
                                    dst_unique_events[idx],
                                    dst_fields[idx].redop_id);
      }
      else
      {
        LegionSpy::log_indirect_events(op->get_unique_op_id(), expr->expr_id,
                unique_indirections_identifier, copy_precondition, last_copy);
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          LegionSpy::log_indirect_field(last_copy, src_fields[idx].field_id,
                                        (idx < src_unique_events.size()) ?
                                          src_unique_events[idx] :
                                          LgEvent::NO_LG_EVENT,
                                        src_fields[idx].indirect_index,
                                        dst_fields[idx].field_id,
                                        (idx < dst_unique_events.size()) ?
                                          dst_unique_events[idx] :
                                          LgEvent::NO_LG_EVENT,
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
                            indirections, requests, precondition, priority));
        if (post.exists())
          postconditions.push_back(post);
        if (runtime->profiler != NULL)
          log_across_profiling(post, idx);
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
                                    precondition, result, DEP_PART_BY_PREIMAGE);
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
      nonempty_indexes.clear();
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
                                        ApEvent partition_ready, ApBarrier pend,
                                        RtEvent init, CollectiveMapping *map,
                                        Provenance *prov)
      : IndexPartNode(ctx, p, par, cs, c, disjoint, complete, did,
          partition_ready, pend, init, map, prov), kd_root(NULL),
        kd_remote(NULL), dense_shard_rects(NULL), sparse_shard_rects(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(RegionTreeForest *ctx, 
                                        IndexPartition p,
                                        IndexSpaceNode *par, IndexSpaceNode *cs,
                                        LegionColor c, RtEvent disjoint_event,
                                        int comp, DistributedID did,
                                        ApEvent partition_ready, ApBarrier pend,
                                        RtEvent init, CollectiveMapping *map,
                                        Provenance *prov)
      : IndexPartNode(ctx, p, par, cs, c, disjoint_event, comp, did,
          partition_ready, pend, init, map, prov), kd_root(NULL),
        kd_remote(NULL), dense_shard_rects(NULL), sparse_shard_rects(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::~IndexPartNodeT(void)
    //--------------------------------------------------------------------------
    { 
      if (kd_root != NULL)
        delete kd_root;
      if (kd_remote != NULL)
        delete kd_remote;
      if (dense_shard_rects != NULL)
        delete dense_shard_rects;
      if (sparse_shard_rects != NULL)
        delete sparse_shard_rects;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::find_interfering_children_kd(
       IndexSpaceExpression *expr, std::vector<LegionColor> &colors, bool local)
    //--------------------------------------------------------------------------
    {
      if (kd_root == NULL)
      {
        if (total_children <= LEGION_MAX_BVH_FANOUT)
          return false;
        DomainT<DIM,T> parent_space;
        const TypeTag type_tag = handle.get_type_tag();
        const ApEvent parent_ready = 
         parent->get_expr_index_space(&parent_space, type_tag, true/*tight*/); 
        if (collective_mapping == NULL)
        {
          // No shard mapping so we can build the full kd-tree here
          std::vector<std::pair<Rect<DIM,T>,LegionColor> > bounds;
          bounds.reserve(total_children);
          for (ColorSpaceIterator itr(this); itr; itr++)
          {
            IndexSpaceNode *child = get_child(*itr);
            DomainT<DIM,T> space;
            const ApEvent space_ready = 
              child->get_expr_index_space(&space, type_tag, true/*tight*/);
            if (space_ready.exists() && !space_ready.has_triggered())
              space_ready.wait();
            if (space.empty())
              continue;
            for (RectInDomainIterator<DIM,T> it(space); it(); it++)
              bounds.push_back(std::make_pair(*it, *itr));
          }
          if (parent_ready.exists() && !parent_ready.has_triggered())
            parent_ready.wait();
          KDNode<DIM,T,LegionColor> *root = 
           new KDNode<DIM,T,LegionColor>(parent_space.bounds, bounds);
          AutoLock n_lock(node_lock);
          if (kd_root == NULL)
            kd_root = root;
          else // Someone else beat us to it
            delete root;
        }
        else
        {
          // There is a shard-mapping so we're going to build two kd-trees
          // One for storing any local or dense rectangles from remote nodes
          // Another for upper bound rectanges of spaces from remote nodes
          // First check to see if we're the first ones here
          RtEvent wait_on;
          {
            AutoLock n_lock(node_lock);
            if (kd_remote_ready.exists() || (kd_remote != NULL))
              wait_on = kd_remote_ready;
            else
              kd_remote_ready = Runtime::create_rt_user_event();
          }
          if (!wait_on.exists() && (kd_remote == NULL))
          {
            const RtEvent rects_ready = request_shard_rects();
            // Grab our local children for later
            std::vector<IndexSpaceNode*> current_children;
            {
              AutoLock n_lock(node_lock,1,false/*exclusive*/);
              current_children.reserve(color_map.size());
              for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
                    color_map.begin(); it != color_map.end(); it++)
                current_children.push_back(it->second);
            }
            if (parent_ready.exists() && !parent_ready.has_triggered())
              parent_ready.wait();
            if (rects_ready.exists() && !rects_ready.has_triggered())
              rects_ready.wait();
            // Once we get the remote rectangles we can build the kd-trees
            if (!sparse_shard_rects->empty())
              kd_remote = new KDNode<DIM,T,AddressSpaceID>(
                  parent_space.bounds, *sparse_shard_rects);
            // Add any local sparse paces into the dense remote rects
            // All the local dense spaces are already included
            for (unsigned idx = 0; idx < current_children.size(); idx++)
            {
              IndexSpaceNode *child = current_children[idx];
              DomainT<DIM,T> space;
              const ApEvent space_ready = 
                child->get_expr_index_space(&space, type_tag, true/*tight*/);
              if (space_ready.exists() && !space_ready.has_triggered())
                space_ready.wait();
              if (space.empty() || space.dense())
                continue;
              for (RectInDomainIterator<DIM,T> it(space); it(); it++)
                dense_shard_rects->push_back(std::make_pair(*it, child->color));
            }
            KDNode<DIM,T,LegionColor> *root = new KDNode<DIM,T,LegionColor>(
                                    parent_space.bounds, *dense_shard_rects);
            AutoLock n_lock(node_lock);
            kd_root = root;
            Runtime::trigger_event(kd_remote_ready);
            kd_remote_ready = RtUserEvent::NO_RT_USER_EVENT;
          }
          else if (wait_on.exists() && !wait_on.has_triggered())
            wait_on.wait();
        }
      }
      DomainT<DIM,T> space;
      const ApEvent space_ready =
        expr->get_expr_index_space(&space, handle.get_type_tag(),true/*tight*/);
      if (space_ready.exists() && !space_ready.has_triggered())
        space_ready.wait();
      // If we have a remote kd tree then we need to query that to see if 
      // we have any remote colors to include
      std::set<LegionColor> color_set;
      if ((kd_remote != NULL) && !local)
      {
        std::set<AddressSpaceID> remote_spaces;
        for (RectInDomainIterator<DIM,T> itr(space); itr(); itr++)
          kd_remote->find_interfering(*itr, remote_spaces);
        if (!remote_spaces.empty())
        {
          RemoteKDTracker tracker(color_set, context->runtime);
          tracker.find_remote_interfering(remote_spaces, handle, expr); 
        }
      }
      for (RectInDomainIterator<DIM,T> itr(space); itr(); itr++)
        kd_root->find_interfering(*itr, color_set);
      if (!color_set.empty())
        colors.insert(colors.end(), color_set.begin(), color_set.end());
      return true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM,T>::initialize_shard_rects(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dense_shard_rects == NULL);
      assert(sparse_shard_rects == NULL);
#endif
      dense_shard_rects = 
        new std::vector<std::pair<Rect<DIM,T>,LegionColor> >();
      sparse_shard_rects =
        new std::vector<std::pair<Rect<DIM,T>,AddressSpaceID> >();
      const TypeTag type_tag = handle.get_type_tag();
      // No need for the lock here, it's being held the caller
      for (std::map<LegionColor,IndexSpaceNode*>::const_iterator it =
            color_map.begin(); it != color_map.end(); it++)
      {
        // Only handle children that we made so that we don't duplicate
        if (!it->second->is_owner())
          continue;
        DomainT<DIM,T> child_space;
        const ApEvent child_ready = 
          it->second->get_expr_index_space(&child_space,type_tag,true/*tight*/);
        if (child_ready.exists() && !child_ready.has_triggered())
          child_ready.wait();
        if (!child_space.dense())
        {
          // Scan through all the previous rectangles to make sure we
          // don't insert any duplicate bounding boxes
          bool found = false;
          for (typename std::vector<std::pair<Rect<DIM,T>,AddressSpaceID> >::
                const_iterator sit = sparse_shard_rects->begin(); sit !=
                sparse_shard_rects->end(); sit++)
          {
            if (sit->first != child_space.bounds)
              continue;
#ifdef DEBUG_LEGION
            assert(sit->second == local_space);
#endif
            found = true;
            break;
          }
          if (!found)
            sparse_shard_rects->push_back(
                std::make_pair(child_space.bounds, local_space));
        }
        else if (!child_space.bounds.empty())
          dense_shard_rects->push_back(
              std::make_pair(child_space.bounds, it->first));
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM,T>::pack_shard_rects(Serializer &rez, bool clear)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dense_shard_rects != NULL);
      assert(sparse_shard_rects != NULL);
#endif
      rez.serialize<size_t>(dense_shard_rects->size());
      for (typename std::vector<std::pair<Rect<DIM,T>,LegionColor> >::
            const_iterator it = dense_shard_rects->begin(); it !=
            dense_shard_rects->end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(sparse_shard_rects->size());
      for (typename std::vector<std::pair<Rect<DIM,T>,AddressSpaceID> >::
            const_iterator it = sparse_shard_rects->begin(); it !=
            sparse_shard_rects->end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      if (clear)
      {
        dense_shard_rects->clear();
        sparse_shard_rects->clear();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM,T>::unpack_shard_rects(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dense_shard_rects != NULL);
      assert(sparse_shard_rects != NULL);
#endif
      size_t num_dense;
      derez.deserialize(num_dense);
      if (num_dense > 0)
      {
        unsigned offset = dense_shard_rects->size();
        dense_shard_rects->resize(offset + num_dense);
        for (unsigned idx = 0; idx < num_dense; idx++)
        {
          std::pair<Rect<DIM,T>,LegionColor> &next = 
            (*dense_shard_rects)[offset + idx];
          derez.deserialize(next.first);
          derez.deserialize(next.second);
        }
      }
      size_t num_sparse;
      derez.deserialize(num_sparse);
      if (num_sparse > 0)
      {
        unsigned offset = sparse_shard_rects->size();
        sparse_shard_rects->resize(offset + num_sparse);
        for (unsigned idx = 0; idx < num_sparse; idx++)
        {
          std::pair<Rect<DIM,T>,AddressSpaceID> &next = 
            (*sparse_shard_rects)[offset + idx];
          derez.deserialize(next.first);
          derez.deserialize(next.second);
        }
      }
    }
#endif // defined(DEFINE_NT_TEMPLATES)

  }; // namespace Internal
}; // namespace Legion

