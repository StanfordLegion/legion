/* Copyright 2024 Stanford University, NVIDIA Corporation
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
        privilege_node->get_realm_index_space(privilege_space, true/*tight*/);
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
      {
        SmallNameClosure<1> *closure = new SmallNameClosure<1>();
        closure->record_instance_name(dst_fields.front().inst, unique_event);
        forest->runtime->profiler->add_fill_request(requests, closure, 
                                                    op, collective);
      }
      ApEvent fill_pre;
      if (pred_guard.exists())
        // No need for tracing to know about the precondition
        fill_pre = Runtime::merge_events(NULL,precondition,ApEvent(pred_guard));
      else
        fill_pre = precondition;
      ApEvent result = ApEvent(space.fill(dst_fields, requests,
              fill_value, fill_size, fill_pre, priority));
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
        assert((it->redop_id == 0) || !reservations.empty());
#endif
      // Now that we know we're going to do this copy add any profling requests
      Realm::ProfilingRequestSet requests;
      if (!replay)
        priority =
          op->add_copy_profiling_request(trace_info, requests, false/*fill*/);
      if (forest->runtime->profiler != NULL)
      {
        SmallNameClosure<2> *closure = new SmallNameClosure<2>();
        closure->record_instance_name(src_fields.front().inst, src_unique);
        closure->record_instance_name(dst_fields.front().inst, dst_unique);
        forest->runtime->profiler->add_copy_request(requests, closure,
                                                    op, 1/*count*/, collective);
      }
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
        {
          for (int dim = 0; dim < 
                constraints.padding_constraint.delta.get_dim(); dim++)
          {
#ifdef DEBUG_LEGION
            assert(constraints.padding_constraint.delta.lo()[dim] >= 0);
            assert(constraints.padding_constraint.delta.hi()[dim] >= 0);
#endif
            if ((constraints.padding_constraint.delta.lo()[dim] > 0) ||
                (constraints.padding_constraint.delta.hi()[dim] > 0))
              REPORT_LEGION_FATAL(LEGION_FATAL_COMPACT_SPARSE_PADDING,
                  "Legion does not currently support additional padding "
                  "on compact sparse instances. Please open a github "
                  "issue to request support.")
          }
        }
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
          return new InternalExpression<DIM,T>(&space.bounds,1/*size*/,context);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(num_rects > 0);
#endif
        // Make a realm expression from the rectangles
        return new InternalExpression<DIM,T>(rects, num_rects, context);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline bool IndexSpaceExpression::meets_layout_expression_internal(
                          IndexSpaceExpression *space_expr, bool tight_bounds,
                          const Rect<DIM,T> *piece_list, size_t piece_list_size,
                          const Domain *padding_delta)
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
        if ((padding_delta != NULL) && (padding_delta->get_dim() > 0))
        {
#ifdef DEBUG_LEGION
          assert(padding_delta->get_dim() == DIM);
#endif
          // We need to check that the dimensions are exactly matching for
          // any which have a non-trival padding
          for (int dim = 0; dim < DIM; dim++)
          {
            if ((padding_delta->lo()[dim] > 0) && 
                (local.bounds.lo[dim] != other.bounds.lo[dim]))
              return false;
            if ((padding_delta->hi()[dim] > 0) &&
                (local.bounds.hi[dim] != other.bounds.hi[dim]))
              return false;
          }
        }
        // If tight, check to see if they are equivalent
        if (tight_bounds)
          return local.bounds == other.bounds;
        return true;
      }
      else
      {
        // Padding is not supported for sparse layouts
        if ((padding_delta != NULL) && (padding_delta->get_dim() > 0))
          return false;
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
              IndexSpaceExpression::create_from_rectangles_internal(
                        RegionTreeForest *forest, const std::set<Domain> &rects)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!rects.empty());
#endif
      size_t total_volume = 0;
      std::vector<Rect<DIM,T> > rectangles;
      rectangles.reserve(rects.size());
      // We're just assuming that all the rectangles here are non-overlapping
      for (std::set<Domain>::const_iterator it =
            rects.begin(); it != rects.end(); it++)
      {
        Rect<DIM,T> rect = *it;
        total_volume += rect.volume();
        rectangles.push_back(rect);
      }
#ifdef DEBUG_LEGION
      assert(total_volume <= get_volume());
#endif
      // If all the points add up to the same as our volume then the 
      // expressions match and we can reuse this as the expression
      if (total_volume == get_volume())
        return this;
      InternalExpression<DIM,T> *result = new InternalExpression<DIM,T>(
          &rectangles.front(), rectangles.size(), forest);
      // Do a little test to see if there is already a canonical expression
      // that we know about that matches this expression if so we'll use that
      // Note that we don't need to explicitly delete it if it is not the
      // canonical expression since it has a live expression reference that
      // will be cleaned up after this meta-task is done running
      return result->get_canonical_expression(forest);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceExpression*
              IndexSpaceExpression::find_congruent_expression_internal(
                                std::vector<IndexSpaceExpression*> &expressions)
    //--------------------------------------------------------------------------
    {
      if (expressions.empty())
      {
        expressions.push_back(this);
        return this;
      }
      else if (std::binary_search(expressions.begin(), expressions.end(), this))
        return this;
      Realm::IndexSpace<DIM,T> local_space;
      // No need to wait for the event, we know it is already triggered
      // because we called get_volume on this before we got here
      get_expr_index_space(&local_space, type_tag, true/*need tight result*/);
      size_t local_rect_count = 0;
      KDNode<DIM,T,void> *local_tree = NULL;
      for (std::vector<IndexSpaceExpression*>::const_iterator it =
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
      expressions.push_back(this);
      // Keep the expressions sorted for searching
      std::sort(expressions.begin(), expressions.end());
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

    template<int DIM, typename T>
    bool IndexSpaceOperationT<DIM,T>::is_sparse()
    {
      return !realm_index_space.dense();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM,T>::get_domain(Domain &domain, bool tight)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> result;
      ApEvent ready = get_realm_index_space(result, tight);
      domain = result;
      return ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM,T>::get_realm_index_space(
                        Realm::IndexSpace<DIM,T> &space, bool need_tight_result)
    //--------------------------------------------------------------------------
    {
      if (!is_index_space_tight.load())
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
      is_index_space_tight.store(true);
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
      if (has_volume.load())
        return volume;
      Realm::IndexSpace<DIM,T> temp;
      get_realm_index_space(temp, true/*tight*/);
      volume = temp.volume();
      has_volume.store(true);
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
      if (is_index_space_tight.load())
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
    IndexSpaceExpression* IndexSpaceOperationT<DIM,T>::create_from_rectangles(
                                                  const std::set<Domain> &rects)
    //--------------------------------------------------------------------------
    {
      return create_from_rectangles_internal<DIM,T>(context, rects);
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
      get_realm_index_space(local_is, true/*tight*/);
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
                            const void *piece_list, size_t piece_list_size,
                            const Domain *padding_delta)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((piece_list_size % sizeof(Rect<DIM,T>)) == 0);
#endif
      return meets_layout_expression_internal<DIM,T>(space_expr, tight_bounds,
                                  static_cast<const Rect<DIM,T>*>(piece_list),
                                  piece_list_size / sizeof(Rect<DIM,T>),
                                  padding_delta);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* 
      IndexSpaceOperationT<DIM,T>::find_congruent_expression(
                                std::vector<IndexSpaceExpression*> &expressions)
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
    void IndexSpaceOperationT<DIM,T>::initialize_equivalence_set_kd_tree(
                     EqKDTree *tree, EquivalenceSet *set, const FieldMask &mask,
                     ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
      {
        const Rect<DIM,T> overlap = itr.rect.intersection(typed_tree->bounds);
        if (!overlap.empty())
          typed_tree->initialize_set(set, overlap, mask, local_shard, current);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM,T>::compute_equivalence_sets(
          EqKDTree *tree, LocalLock *tree_lock, const FieldMask &mask, 
          const std::vector<EqSetTracker*> &trackers,
          const std::vector<AddressSpaceID> &tracker_spaces,
          std::vector<unsigned> &new_tracker_references,
          FieldMaskSet<EquivalenceSet> &eq_sets,
          std::vector<RtEvent> &pending_sets,
          FieldMaskSet<EqKDTree> &subscriptions,
          FieldMaskSet<EqKDTree> &to_create,
          std::map<EqKDTree*,Domain> &creation_rects,
          std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock,1,false/*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
        typed_tree->compute_equivalence_sets(itr.rect, mask, trackers,
            tracker_spaces, new_tracker_references, eq_sets, pending_sets,
            subscriptions, to_create, creation_rects, creation_srcs,
            remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned IndexSpaceOperationT<DIM,T>::record_output_equivalence_set(
          EqKDTree *tree, LocalLock *tree_lock, EquivalenceSet *set, 
          const FieldMask &mask, EqSetTracker *tracker,
          AddressSpaceID tracker_space, FieldMaskSet<EqKDTree> &subscriptions,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      unsigned new_subs = 0;
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock,1,false/*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
        new_subs += typed_tree->record_output_equivalence_set(set, itr.rect,
            mask, tracker, tracker_space, subscriptions, remote_shard_rects,
            local_shard);
      return new_subs;
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
      if (this->realm_index_space_ready.exists() || 
          !valid_event.has_triggered())
      {
        IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
        if (this->realm_index_space_ready.exists())
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
      if (this->realm_index_space_ready.exists() ||
          !valid_event.has_triggered())
      {
        IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
        if (this->realm_index_space_ready.exists())
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
        if (this->realm_index_space_ready.exists() ||
            !valid_event.has_triggered())
        {
          IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
          if (this->realm_index_space_ready.exists())
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
    InternalExpression<DIM,T>::InternalExpression(
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
        this->is_index_space_tight.store(true);
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
    InternalExpression<DIM,T>::InternalExpression(
                                           const InternalExpression<DIM,T> &rhs)
      : IndexSpaceOperationT<DIM,T>(
          IndexSpaceOperation::INSTANCE_EXPRESSION_KIND, NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    InternalExpression<DIM,T>::~InternalExpression(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    InternalExpression<DIM,T>& InternalExpression<DIM,T>::operator=(
                                           const InternalExpression<DIM,T> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void InternalExpression<DIM,T>::pack_expression_value(Serializer &rez,
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
    bool InternalExpression<DIM,T>::invalidate_operation(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void InternalExpression<DIM,T>::remove_operation(void)
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
        DistributedID did, IndexSpaceExprID expr_id, RtEvent init, unsigned dep,
        Provenance *prov, CollectiveMapping *mapping, bool tree_valid)
      : IndexSpaceNode(ctx, handle, parent, color, did, expr_id, init,
          dep, prov, mapping, tree_valid), linearization(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM,T>::~IndexSpaceNodeT(void)
    //--------------------------------------------------------------------------
    { 
      if (is_owner() || ((collective_mapping != NULL) && 
            collective_mapping->contains(local_space)))
        realm_index_space.destroy(index_space_valid);
      ColorSpaceLinearizationT<DIM,T> *linear = linearization.load();
      if (linear != NULL)
        delete linear;
    }

    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::is_sparse()
    {
      return !realm_index_space.dense();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::get_realm_index_space(
                      Realm::IndexSpace<DIM,T> &result, bool need_tight_result)
    //--------------------------------------------------------------------------
    {
      if (!index_space_set.load())
      {
        RtEvent wait_on;
        {
          AutoLock n_lock(node_lock);
          if (!index_space_set.load())
          {
            if (!index_space_ready.exists())
              index_space_ready = Runtime::create_rt_user_event();
            wait_on = index_space_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
      if (need_tight_result && !index_space_tight.load())
      {
        RtEvent wait_on;
        {
          AutoLock n_lock(node_lock);
          if (!index_space_tight.load())
          {
            if (!index_space_ready.exists())
              index_space_ready = Runtime::create_rt_user_event();
            wait_on = index_space_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
      if (!need_tight_result)
      {
        // Need a reader lock to avoid racing with tightening
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        result = realm_index_space;
        return index_space_valid;
      }
      else
      {
        // Can read without the lock since there are no more modifications
        result = realm_index_space;
        return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::set_realm_index_space(
                       const Realm::IndexSpace<DIM,T> &value, ApEvent valid, 
                       bool initializing, bool broadcast, AddressSpaceID source)
    //--------------------------------------------------------------------------
    { 
      // If we're broadcasting, then send this out there to get it in flight
      if (broadcast)
      {
        if ((collective_mapping != NULL) && 
            collective_mapping->contains(local_space))
        {
          std::vector<AddressSpaceID> children;
          collective_mapping->get_children(owner_space, local_space, children);
          const AddressSpaceID parent_space = is_owner() ? source :
              collective_mapping->get_parent(owner_space, local_space);
          if (!children.empty() || (parent_space != source))
          {
            Serializer rez;
            {
              RezCheck z(rez);
              if (parent != NULL)
              {
                rez.serialize(parent->handle);
                rez.serialize(color);
              }
              else
              {
                rez.serialize(IndexPartition::NO_PART);
                rez.serialize(handle);
              }
              rez.serialize(value);
              rez.serialize(valid);
            }
            for (std::vector<AddressSpaceID>::const_iterator it =
                  children.begin(); it != children.end(); it++)
              if ((*it) != source)
                runtime->send_index_space_set(*it, rez);
            if (parent_space != source)
              runtime->send_index_space_set(parent_space, rez);
          }
        }
        else if (!is_owner() && (source == local_space))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            if (parent != NULL)
            {
              rez.serialize(parent->handle);
              rez.serialize(color);
            }
            else
            {
              rez.serialize(IndexPartition::NO_PART);
              rez.serialize(handle);
            }
            rez.serialize(value);
            rez.serialize(valid);
          }
          if (collective_mapping != NULL)
            runtime->send_index_space_set(
                collective_mapping->find_nearest(local_space), rez);
          else
            runtime->send_index_space_set(owner_space, rez);
        }
      }
      // We can set this now and trigger the event but setting the
      // flag has to be done while holding the node_lock on the owner
      // node so that it is serialized with respect to queries from 
      // remote nodes for copies about the remote instance
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
        assert(!index_space_set.load());
#endif
        realm_index_space = value;
        index_space_valid = valid;
        index_space_set.store(true);
        if (index_space_ready.exists())
        {
          Runtime::trigger_event(index_space_ready);
          index_space_ready = RtUserEvent::NO_RT_USER_EVENT;
        }
        if (has_remote_instances())
        {
          // We're the owner, send messages to everyone else that we've 
          // sent this node to except the source
          Serializer rez;
          {
            RezCheck z(rez);
            if (parent != NULL)
            {
              rez.serialize(parent->handle);
              rez.serialize(color);
            }
            else
            {
              rez.serialize(IndexPartition::NO_PART);
              rez.serialize(handle);
            }
            pack_index_space(rez, false/*include size*/);
            rez.serialize(valid);
          }
          IndexSpaceSetFunctor functor(context->runtime, source, rez);
          map_over_remote_instances(functor);
        }
      }
      // Now we can tighten it
      tighten_index_space();
      if (is_owner() || ((collective_mapping != NULL) && 
            collective_mapping->contains(local_space)))
      {
        if (parent != NULL)
          parent->set_child(this);
      }
      // Remove the reference we were holding until this was set
      if (initializing)
        return false;
      else if (parent == NULL)
        return remove_base_gc_ref(REGION_TREE_REF);
      if (parent->remove_base_gc_ref(REGION_TREE_REF))
        delete parent;
      return false;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RtEvent IndexSpaceNodeT<DIM,T>::get_realm_index_space_ready(bool need_tight)
    //--------------------------------------------------------------------------
    {
      if (index_space_tight.load())
        return RtEvent::NO_RT_EVENT;
      if (!need_tight && index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      AutoLock n_lock(node_lock);
      // See if we lost the race
      if (index_space_tight.load())
        return RtEvent::NO_RT_EVENT;
      if (!need_tight && index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      if (!index_space_ready.exists())
        index_space_ready = Runtime::create_rt_user_event();
      return index_space_ready;
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
    ApEvent IndexSpaceNodeT<DIM,T>::get_domain(Domain &domain, bool need_tight)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM,T> result;
      ApEvent ready = get_realm_index_space(result, need_tight);
      domain = result;
      return ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::set_domain(const Domain &domain,bool broadcast)
    //--------------------------------------------------------------------------
    {
      const DomainT<DIM,T> realm_space = domain;
      return set_realm_index_space(realm_space, ApEvent::NO_AP_EVENT,
          false/*init*/, broadcast, context->runtime->address_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::set_bounds(const void *bounds, bool is_domain,
                                          bool initialization, ApEvent is_ready)
    //--------------------------------------------------------------------------
    {
      if (is_domain)
      {
        const DomainT<DIM,T> temp_space = *static_cast<const Domain*>(bounds);
        return set_realm_index_space(temp_space, is_ready, initialization);
      }
      else
      {
        const DomainT<DIM,T> temp_space =
          *static_cast<const Realm::IndexSpace<DIM,T>*>(bounds);
        return set_realm_index_space(temp_space, is_ready, initialization);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::set_output_union(
                          const std::map<DomainPoint,DomainPoint> &output_sizes)
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
      return set_realm_index_space(output_space, ApEvent::NO_AP_EVENT,
          false/*init*/, false/*broadcast*/, context->runtime->address_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::tighten_index_space(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index_space_set.load());
      assert(!index_space_tight.load());
#endif
      const RtEvent valid_event(realm_index_space.make_valid());
      if (!valid_event.has_triggered() || index_space_valid.exists())
      {
        // If this index space isn't ready yet, then we have to defer this 
        if (!valid_event.has_triggered())
        {
          TightenIndexSpaceArgs args(this, this);
          if (index_space_valid.exists())
            context->runtime->issue_runtime_meta_task(args,
                LG_LATENCY_WORK_PRIORITY, Runtime::merge_events(valid_event,
                  Runtime::protect_event(index_space_valid)));
          else
            context->runtime->issue_runtime_meta_task(args,
                LG_LATENCY_WORK_PRIORITY, valid_event);
          return;
        }
        else
        {
          const RtEvent safe = Runtime::protect_event(index_space_valid);
          if (safe.exists() && !safe.has_triggered())
          {
            TightenIndexSpaceArgs args(this, this);
            context->runtime->issue_runtime_meta_task(args,
                  LG_LATENCY_WORK_PRIORITY, safe);
            return;
          }
        }
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
        index_space_tight.store(true);
        if (index_space_ready.exists())
        {
          Runtime::trigger_event(index_space_ready);
          index_space_ready = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      old_space.destroy(index_space_valid);
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
    IndexSpaceExpression* IndexSpaceNodeT<DIM,T>::create_from_rectangles(
                                                  const std::set<Domain> &rects)
    //--------------------------------------------------------------------------
    {
      return create_from_rectangles_internal<DIM,T>(context, rects);
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
        if (set_realm_index_space(result_space, result))
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
        if (set_realm_index_space(result_space, result))
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
        if (set_realm_index_space(result_space, result))
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
        if (set_realm_index_space(result_space, result))
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
      if (set_realm_index_space(result_space, result))
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
      if (has_volume.load())
        return volume;
      Realm::IndexSpace<DIM,T> volume_space;
      get_realm_index_space(volume_space, true/*tight*/);
      volume = volume_space.volume();
      has_volume.store(true);
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
      assert(index_space_set.load());
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
      ApEvent valid_event;
      derez.deserialize(valid_event);
      return set_realm_index_space(result_space, valid_event,
          false/*initialization*/, true/*broadcast*/, source);
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
          if (child->set_realm_index_space(subspaces[subspace_index++], result))
            delete child;
        }
        return result;
      }
      else
      {
        const size_t count = partition->total_children;
        std::set<ApEvent> done_events;
        Realm::IndexSpace<DIM,T> local_space;
        const ApEvent local_ready = 
          get_realm_index_space(local_space, false/*tight*/);
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
          ApEvent result(local_space.create_equal_subspace(count, 
            granularity, color_offset, subspace, requests, local_ready)); 
          IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
          if (child->set_realm_index_space(subspace, result))
            delete child;
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
      std::vector<Realm::IndexSpace<DIM,T> > lhs_spaces;
      std::vector<Realm::IndexSpace<DIM,T> > rhs_spaces;
      std::vector<ApEvent> preconditions;
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
          preconditions.push_back(left_ready);
        if (right_ready.exists())
          preconditions.push_back(right_ready);
      }
      if (lhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                              op, DEP_PART_UNIONS);
      if (op->has_execution_fence_event())
        preconditions.push_back(op->get_execution_fence_event());
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
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
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
      std::vector<ApEvent> preconditions;
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
          preconditions.push_back(left_ready);
        if (right_ready.exists())
          preconditions.push_back(right_ready);
      }
      if (lhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_INTERSECTIONS);
      if (op->has_execution_fence_event())
        preconditions.push_back(op->get_execution_fence_event());
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
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
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
      std::vector<ApEvent> preconditions;
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
          preconditions.push_back(right_ready);
      }
      if (rhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
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
          preconditions.push_back(left_ready);
        if (op->has_execution_fence_event())
          preconditions.push_back(op->get_execution_fence_event());
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
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
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
      std::vector<ApEvent> preconditions;
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
          preconditions.push_back(left_ready);
        if (right_ready.exists())
          preconditions.push_back(right_ready);
      }
      if (lhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_DIFFERENCES);
      if (op->has_execution_fence_event())
        preconditions.push_back(op->get_execution_fence_event());
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
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
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
      ApEvent ready = parent->get_realm_index_space(parent_is, false/*tight*/);
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
        if (child->set_realm_index_space(child_is, ready))
          delete child;
      }
      // Our only precondition is that the parent index space is computed
      return ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_domain(Operation *op,
                                                    IndexPartNode *partition,
                            const std::map<DomainPoint,FutureImpl*> &futures,
                                              const Domain &future_map_domain,
                                                    bool perform_intersections)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the color space type to do the actual operations 
      CreateByDomainHelper creator(this, partition, op, futures, 
                                   future_map_domain, perform_intersections);
      NT_TemplateHelper::demux<CreateByDomainHelper>(
                   partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_weights(Operation *op,
                                                    IndexPartNode *partition,
                              const std::map<DomainPoint,FutureImpl*> &weights,
                                                    size_t granularity)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the color space type to do the actual operations 
      CreateByWeightHelper creator(this, partition, op, weights, granularity);
      NT_TemplateHelper::demux<CreateByWeightHelper>(
                   partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_field(Operation *op, FieldID fid,
                                                    IndexPartNode *partition,
                              const std::vector<FieldDataDescriptor> &instances,
                                    std::vector<DeppartResult> *results,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the color space type to do the actual operations 
      CreateByFieldHelper creator(this, op, fid, partition,
                                  instances, results, instances_ready);
      NT_TemplateHelper::demux<CreateByFieldHelper>(
                   partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T> template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_domain_helper(Operation *op,
                          IndexPartNode *partition, 
                          const std::map<DomainPoint,FutureImpl*> &futures,
                          const Domain &future_map_domain,
                          bool perform_intersections) 
    //--------------------------------------------------------------------------
    {
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
        const DomainPoint color = 
          partition->color_space->delinearize_color_to_point(*itr);
        ApEvent child_ready;
        Realm::IndexSpace<DIM,T> child_space;
        if (future_map_domain.contains(color))
        {
          std::map<DomainPoint,FutureImpl*>::const_iterator finder =
            futures.find(color);
#ifdef DEBUG_LEGION
          assert(finder != futures.end());
#endif
          FutureImpl *future = finder->second;
          size_t future_size = 0;
          const Domain *domain = static_cast<const Domain*>(
              future->find_runtime_buffer(op->get_context(), future_size));
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
            child_ready = ApEvent(
                Realm::IndexSpace<DIM,T>::compute_intersection(
                  parent_space, child_space, result, requests, parent_ready));
            child_space = result;
            if (child_ready.exists())
              result_events.insert(child_ready);
          }
        }
        else
          child_space = Realm::IndexSpace<DIM,T>::make_empty();
        IndexSpaceNodeT<DIM,T> *child = 
            static_cast<IndexSpaceNodeT<DIM,T>*>(
                partition->get_child(*itr));
        if (child->set_realm_index_space(child_space, child_ready))
          delete child;
      }
      if (result_events.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(NULL, result_events);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T> template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_weight_helper(Operation *op,
        IndexPartNode *partition, 
        const std::map<DomainPoint,FutureImpl*> &futures, size_t granularity)
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
          std::map<DomainPoint,FutureImpl*>::const_iterator finder = 
            futures.find(key);
          if (finder == futures.end())
            REPORT_LEGION_ERROR(ERROR_MISSING_PARTITION_BY_WEIGHT_COLOR,
                "A partition by weight call is missing an entry for a "
                "color in the color space. All colors must be present.")
          FutureImpl *future = finder->second;
          size_t future_size = 0;
          const void *data =
            future->find_runtime_buffer(op->get_context(), future_size);
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
        if (child->set_realm_index_space(subspaces[next++], result))
          delete child;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T> template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_field_helper(Operation *op,
                                                      FieldID fid,
                                                      IndexPartNode *partition,
                             const std::vector<FieldDataDescriptor> &instances,
                                   std::vector<DeppartResult> *results,
                                                       ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID source_space = context->runtime->address_space;
      // If we already have results then we can just fill them in 
      if ((results != NULL) && !results->empty())
      {
        DeppartResult key;
        for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
        {
          IndexSpaceNodeT<DIM,T> *child =
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
          // Find the resulting color
          key.color = *itr;
          std::vector<DeppartResult>::const_iterator finder =
            std::lower_bound(results->begin(), results->end(), key);
#ifdef DEBUG_LEGION
          assert(finder != results->end());
          assert(finder->color == (*itr));
#endif
          Realm::IndexSpace<DIM,T> result = finder->domain;
          if (child->set_realm_index_space(result, instances_ready,
                false/*initialization*/, false/*broadcast*/, source_space))
            delete child;
        }
        return ApEvent::NO_AP_EVENT;
      }
      IndexSpaceNodeT<COLOR_DIM,COLOR_T> *color_space = 
       static_cast<IndexSpaceNodeT<COLOR_DIM,COLOR_T>*>(partition->color_space);
      unsigned index = 0;
      std::vector<Point<COLOR_DIM,COLOR_T> > colors(partition->total_children);
      if (results != NULL)
        results->resize(partition->total_children);
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
#ifdef DEBUG_LEGION
        assert(index < colors.size());
#endif
        if (results != NULL)
          results->at(index).color = *itr;
        color_space->delinearize_color(*itr, colors[index++]);
      }
      // Translate the instances to realm field data descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM,T>,
                Realm::Point<COLOR_DIM,COLOR_T> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_FIELD);
      // Perform the operation
      Realm::IndexSpace<DIM,T> local_space;
      ApEvent parent_ready = get_realm_index_space(local_space, false/*tight*/);
      std::vector<ApEvent> preconditions;
      if (parent_ready.exists())
        preconditions.push_back(parent_ready);
      if (instances_ready.exists())
        preconditions.push_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.push_back(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(NULL, preconditions);
      std::vector<Realm::IndexSpace<DIM,T> > subspaces;
      ApEvent result(local_space.create_subspaces_by_field(
            descriptors, colors, subspaces, requests, precondition));
#ifdef DEBUG_LEGION
      assert(colors.size() == subspaces.size());
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
      index = colors.size();
      // Set our local children results here
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        if (index == colors.size())
          // Compute the index offset the first time through
          index = color_space->compute_color_offset(*itr);
#ifdef DEBUG_LEGION
        assert(index < colors.size());
#endif
        IndexSpaceNodeT<DIM,T> *child =
            static_cast<IndexSpaceNodeT<DIM,T>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(subspaces[index++], result,
              false/*initialization*/, (results == NULL), source_space))
          delete child;
      }
      if (results != NULL)
      {
        // Save the results to be broadcast if necessary
        for (unsigned idx = 0; idx < subspaces.size(); idx++)
          results->at(idx).domain = subspaces[idx];
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_image(Operation *op,
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                                  std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByImageHelper creator(this, op, fid, partition, projection,
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
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                                  std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      ApEvent precondition;
      bool first_child = true;
      std::vector<ApEvent> results;
      Realm::IndexSpace<DIM1,T1> local_space;
      const AddressSpaceID source_space = context->runtime->address_space;
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM2,T2>,
                                       Realm::Point<DIM1,T1> > RealmDescriptor;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM1,T1> *child =
         static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
        // Partition by images are expensive to compute so we only want
        // to do it once so we only do it if we're the owner of the child
        // and then we'll broadcast it out to all the other copies
        if (!child->is_owner())
          continue;
        if (first_child)
        {
          std::vector<ApEvent> preconditions;
          ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
          if (ready.exists())
            preconditions.push_back(ready);
          if (instances_ready.exists())
            preconditions.push_back(instances_ready);
          if (op->has_execution_fence_event())
            preconditions.push_back(op->get_execution_fence_event());
          precondition = Runtime::merge_events(NULL, preconditions);
          // sort the instances so we can search for what we need
          std::sort(instances.begin(), instances.end());
          first_child = false;
        }
        std::vector<RealmDescriptor> descriptors;
        FieldDataDescriptor key;
        key.color = partition->color_space->delinearize_color_to_point(*itr);
        Realm::IndexSpace<DIM2,T2> source =
          Realm::IndexSpace<DIM2,T2>::make_empty();
        for (std::vector<FieldDataDescriptor>::const_iterator it =
              std::lower_bound(instances.begin(), instances.end(), key); it !=
              std::upper_bound(instances.begin(), instances.end(), key); it++)
        {
          descriptors.resize(descriptors.size() + 1);
          RealmDescriptor &dst = descriptors.back();
          dst.index_space = it->domain;
          source = dst.index_space;
          dst.inst = it->inst;
          dst.field_offset = fid;
        }
#ifdef DEBUG_LEGION
        // We should have exactly one of these here for each image
        assert(descriptors.size() == 1);
#endif
        // Get the profiling requests
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                              op, DEP_PART_BY_IMAGE);
        Realm::IndexSpace<DIM1,T1> subspace;
        ApEvent result(local_space.create_subspace_by_image(descriptors,
              source, subspace, requests, precondition));
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
        // Set the result and indicate that we're broadcasting it
        if (child->set_realm_index_space(subspace, result,
              false/*initialization*/, true/*broadcast*/, source_space))
          delete child;
        if (result.exists())
          results.push_back(result);
      }
      if (results.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(NULL, results);
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_image_range(Operation *op,
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                                  std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByImageRangeHelper creator(this, op, fid, partition, projection,
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
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                                  std::vector<FieldDataDescriptor> &instances,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      ApEvent precondition;
      bool first_child = true;
      std::vector<ApEvent> results;
      Realm::IndexSpace<DIM1,T1> local_space;
      const AddressSpaceID source_space = context->runtime->address_space;
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM2,T2>,
                                       Realm::Rect<DIM1,T1> > RealmDescriptor;
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM1,T1> *child = 
         static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
        // Partition by images are expensive to compute so we only want
        // to do it once so we only do it if we're the owner of the child
        // and then we'll broadcast it out to all the other copies
        if (!child->is_owner())
          continue;
        if (first_child)
        {
          std::vector<ApEvent> preconditions;
          ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
          if (ready.exists())
            preconditions.push_back(ready);
          if (instances_ready.exists())
            preconditions.push_back(instances_ready);
          if (op->has_execution_fence_event())
            preconditions.push_back(op->get_execution_fence_event());
          precondition = Runtime::merge_events(NULL, preconditions);
          // sort the instances so we can search for what we need
          std::sort(instances.begin(), instances.end());
          first_child = false;
        }
        std::vector<RealmDescriptor> descriptors;
        FieldDataDescriptor key;
        key.color = partition->color_space->delinearize_color_to_point(*itr);
        Realm::IndexSpace<DIM2,T2> source =
          Realm::IndexSpace<DIM2,T2>::make_empty();
        for (std::vector<FieldDataDescriptor>::const_iterator it =
              std::lower_bound(instances.begin(), instances.end(), key); it !=
              std::upper_bound(instances.begin(), instances.end(), key); it++)
        {
          descriptors.resize(descriptors.size() + 1);
          RealmDescriptor &dst = descriptors.back();
          dst.index_space = it->domain;
          source = dst.index_space;
          dst.inst = it->inst;
          dst.field_offset = fid;
        }
#ifdef DEBUG_LEGION
        // We should have exactly one of these here for each image
        assert(descriptors.size() == 1);
#endif
        // Get the profiling requests
        Realm::ProfilingRequestSet requests;
        if (context->runtime->profiler != NULL)
          context->runtime->profiler->add_partition_request(requests,
                                        op, DEP_PART_BY_IMAGE_RANGE);
        Realm::IndexSpace<DIM1,T1> subspace;
        ApEvent result(local_space.create_subspace_by_image(descriptors,
              source, subspace, requests, precondition));
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
        // Set the result and indicate that we're broadcasting it
        if (child->set_realm_index_space(subspace, result,
              false/*initialization*/, true/*broadcast*/, source_space))
          delete child;
        if (result.exists())
          results.push_back(result);
      }
      if (results.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(NULL, results);
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_preimage(Operation *op,
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                            const std::map<DomainPoint,Domain> *remote_targets,
                                  std::vector<DeppartResult> *results,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByPreimageHelper creator(this, op, fid, partition, projection,
                     instances, remote_targets, results, instances_ready);
      NT_TemplateHelper::demux<CreateByPreimageHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM1, typename T1> template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1,T1>::create_by_preimage_helper(Operation *op,
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                            const std::map<DomainPoint,Domain> *remote_targets,
                                  std::vector<DeppartResult> *results,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID source_space = context->runtime->address_space;
      // If we already have results then we can just fill them in 
      if ((results != NULL) && !results->empty())
      {
        DeppartResult key;
        for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
        {
          IndexSpaceNodeT<DIM1,T1> *child =
            static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
          // Find the resulting color
          key.color = *itr;
          std::vector<DeppartResult>::const_iterator finder =
            std::lower_bound(results->begin(), results->end(), key);
#ifdef DEBUG_LEGION
          assert(finder != results->end());
          assert(finder->color == (*itr));
#endif
          Realm::IndexSpace<DIM1,T1> result = finder->domain;
          if (child->set_realm_index_space(result, instances_ready,
                false/*initialization*/, false/*broadcast*/, source_space))
            delete child;
        }
        return ApEvent::NO_AP_EVENT;
      }
      // Get the target index spaces of the projection partition
      std::vector<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM2,T2> > 
        targets(partition->total_children);
      unsigned index = 0;
      if (results != NULL)
        results->resize(partition->total_children);
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
#ifdef DEBUG_LEGION
        assert(index < targets.size());
#endif
        if (results != NULL)
          results->at(index).color = *itr;
        const DomainPoint color =
          partition->color_space->delinearize_color_to_point(*itr);
        if (remote_targets != NULL)
        {
          std::map<DomainPoint,Domain>::const_iterator finder =
            remote_targets->find(color);
          if (finder != remote_targets->end())
          {
            targets[index++] = finder->second;
            continue;
          }
        }
        // Get the corresponding subspace for the targets
        const LegionColor target_color =
          projection->color_space->linearize_color(color);
        IndexSpaceNodeT<DIM2,T2> *target_child =
          static_cast<IndexSpaceNodeT<DIM2,T2>*>(
              projection->get_child(target_color));
        ApEvent ready = target_child->get_realm_index_space(
                            targets[index++], false/*tight*/);
        if (ready.exists())
          preconditions.push_back(ready);
      }
#ifdef DEBUG_LEGION
      assert(index == targets.size());
#endif
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM1,T1>,
                                       Realm::Point<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                            op, DEP_PART_BY_PREIMAGE);
      // Perform the operation
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (ready.exists())
        preconditions.push_back(ready);
      if (instances_ready.exists())
        preconditions.push_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.push_back(op->get_execution_fence_event());
      std::vector<Realm::IndexSpace<DIM1,T1> > subspaces;
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
      // Update any local children with their results
      index = subspaces.size();
      // Set our local children results here
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        if (index == subspaces.size())
          // Compute the index offset the first time through
          index = partition->color_space->compute_color_offset(*itr);
#ifdef DEBUG_LEGION
        assert(index < subspaces.size());
#endif
        IndexSpaceNodeT<DIM1,T1> *child =
            static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(subspaces[index++], result,
              false/*initialization*/, (results == NULL), source_space))
          delete child;
      }
      if (results != NULL)
      {
        // Save the results to be broadcast if necessary
        for (unsigned idx = 0; idx < subspaces.size(); idx++)
          results->at(idx).domain = subspaces[idx];
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_by_preimage_range(Operation *op,
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                            const std::map<DomainPoint,Domain> *remote_targets,
                                  std::vector<DeppartResult> *results,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->parent == this);
#endif
      // Demux the projection type to do the actual operations
      CreateByPreimageRangeHelper creator(this, op, fid, partition, projection,
                          instances, remote_targets, results, instances_ready);
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
                                                    FieldID fid,
                                                    IndexPartNode *partition,
                                                    IndexPartNode *projection,
                            const std::vector<FieldDataDescriptor> &instances,
                            const std::map<DomainPoint,Domain> *remote_targets,
                                  std::vector<DeppartResult> *results,
                                                    ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID source_space = context->runtime->address_space;
      // If we already have results then we can just fill them in 
      if ((results != NULL) && !results->empty())
      {
        DeppartResult key;
        for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
        {
          IndexSpaceNodeT<DIM1,T1> *child =
            static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
          // Find the resulting color
          key.color = *itr;
          std::vector<DeppartResult>::const_iterator finder =
            std::lower_bound(results->begin(), results->end(), key);
#ifdef DEBUG_LEGION
          assert(finder != results->end());
          assert(finder->color == (*itr));
#endif
          Realm::IndexSpace<DIM1,T1> result = finder->domain;
          if (child->set_realm_index_space(result, instances_ready,
                false/*initialization*/, false/*broadcast*/, source_space))
            delete child;
        }
        return ApEvent::NO_AP_EVENT;
      }

      // Get the target index spaces of the projection partition
      std::vector<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM2,T2> > 
        targets(partition->total_children);
      unsigned index = 0;
      if (results != NULL)
        results->resize(partition->total_children);
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
#ifdef DEBUG_LEGION
        assert(index < targets.size());
#endif
        if (results != NULL)
          results->at(index).color = *itr;
        const DomainPoint color =
          partition->color_space->delinearize_color_to_point(*itr);
        if (remote_targets != NULL)
        {
          std::map<DomainPoint,Domain>::const_iterator finder =
            remote_targets->find(color);
          if (finder != remote_targets->end())
          {
            targets[index++] = finder->second;
            continue;
          }
        }
        // Get the corresponding subspace for the targets
        const LegionColor target_color =
          projection->color_space->linearize_color(color);
        IndexSpaceNodeT<DIM2,T2> *target_child =
          static_cast<IndexSpaceNodeT<DIM2,T2>*>(
              projection->get_child(target_color));
        ApEvent ready = target_child->get_realm_index_space(
                            targets[index++], false/*tight*/);
        if (ready.exists())
          preconditions.push_back(ready);
      }
#ifdef DEBUG_LEGION
      assert(index == targets.size());
#endif
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM1,T1>,
                                       Realm::Rect<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
      }
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                    op, DEP_PART_BY_PREIMAGE_RANGE);
      // Perform the operation
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent ready = get_realm_index_space(local_space, false/*tight*/);
      if (ready.exists())
        preconditions.push_back(ready);
      if (instances_ready.exists())
        preconditions.push_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.push_back(op->get_execution_fence_event());
      std::vector<Realm::IndexSpace<DIM1,T1> > subspaces;
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
      // Update any local children with their results
      index = subspaces.size();
      // Set our local children results here
      for (ColorSpaceIterator itr(partition, true/*local only*/); itr; itr++)
      {
        if (index == subspaces.size())
          // Compute the index offset the first time through
          index = partition->color_space->compute_color_offset(*itr);
#ifdef DEBUG_LEGION
        assert(index < subspaces.size());
#endif
        IndexSpaceNodeT<DIM1,T1> *child =
            static_cast<IndexSpaceNodeT<DIM1,T1>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(subspaces[index++], result,
              false/*initialization*/, (results == NULL), source_space))
          delete child;
      }
      if (results != NULL)
      {
        // Save the results to be broadcast if necessary
        for (unsigned idx = 0; idx < subspaces.size(); idx++)
          results->at(idx).domain = subspaces[idx];
      }
      return result;
    }
#endif // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM,T>::create_association(Operation *op,
                                                       FieldID fid,
                                                       IndexSpaceNode *range,
                              const std::vector<FieldDataDescriptor> &instances,
                                                       ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Demux the range type to do the actual operation
      CreateAssociationHelper creator(this, op, fid, range, 
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
                                                      FieldID fid,
                                                      IndexSpaceNode *range,
                              const std::vector<FieldDataDescriptor> &instances,
                                                      ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<Realm::IndexSpace<DIM1,T1>,
                                       Realm::Point<DIM2,T2> > RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor &src = instances[idx];
        RealmDescriptor &dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
      }
      // Get the range index space
      IndexSpaceNodeT<DIM2,T2> *range_node = 
        static_cast<IndexSpaceNodeT<DIM2,T2>*>(range);
      Realm::IndexSpace<DIM2,T2> range_space;
      ApEvent range_ready = range_node->get_realm_index_space(range_space,
                                                              false/*tight*/);
      std::vector<ApEvent> preconditions;
      if (range_ready.exists())
        preconditions.push_back(range_ready);
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (context->runtime->profiler != NULL)
        context->runtime->profiler->add_partition_request(requests,
                                          op, DEP_PART_ASSOCIATION);
      Realm::IndexSpace<DIM1,T1> local_space;
      ApEvent local_ready = get_realm_index_space(local_space, false/*tight*/);
      if (local_ready.exists())
        preconditions.push_back(local_ready);
      if (instances_ready.exists())
        preconditions.push_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.push_back(op->get_execution_fence_event());
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
    Realm::InstanceLayoutGeneric* IndexSpaceNodeT<DIM,T>::create_hdf5_layout(
				    const std::vector<FieldID> &field_ids,
                                    const std::vector<size_t> &field_sizes,
                                    const std::vector<std::string> &field_files,
                                    const OrderingConstraint &dimension_order)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(int(dimension_order.ordering.size()) == (DIM+1));
      assert(dimension_order.ordering.back() == LEGION_DIM_F);
#endif
      // Have to wait for the index space to be ready if necessary
      Realm::IndexSpace<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
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
          layout->bytes_used += hlp->bounds.volume() * fl.size_in_bytes;
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
      return layout;
#else
      assert(false); // should never get here
      return NULL;
#endif
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
      get_realm_index_space(local_is, true/*tight*/);
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
                            const void *piece_list, size_t piece_list_size,
                            const Domain *padding_delta)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((piece_list_size % sizeof(Rect<DIM,T>)) == 0);
#endif
      return meets_layout_expression_internal<DIM,T>(space_expr, tight_bounds,
                                  static_cast<const Rect<DIM,T>*>(piece_list),
                                  piece_list_size / sizeof(Rect<DIM,T>),
                                  padding_delta);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* 
            IndexSpaceNodeT<DIM,T>::find_congruent_expression(
                                std::vector<IndexSpaceExpression*> &expressions)
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
    EqKDTree* IndexSpaceNodeT<DIM,T>::create_equivalence_set_kd_tree(
                                                            size_t total_shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(total_shards > 0);
#endif
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      if (total_shards == 1)
      {
        // Non replicated path
        // If it's dense we can just make a node and return it
        if (!realm_index_space.dense())
        {
          // If it's not dense then we need to make a sparse one to handle
          // all the rects associated with the sparsity map
          std::vector<Rect<DIM,T> > rects;
          for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
                itr.valid; itr.step())
            rects.push_back(itr.rect);
          return new EqKDSparse<DIM,T>(realm_index_space.bounds, rects);
        }
        else
          return new EqKDNode<DIM,T>(realm_index_space.bounds);
      }
      else
      {
        // Control replicated path
        if (!realm_index_space.dense())
        {
          std::vector<Rect<DIM,T> > rects;
          for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
                itr.valid; itr.step())
            rects.push_back(itr.rect);
          return new EqKDSparseSharded<DIM,T>(realm_index_space.bounds,
              0/*lower shard*/, total_shards - 1/*upper shard*/, rects);
        }
        else
          return new EqKDSharded<DIM,T>(realm_index_space.bounds, 
              0/*lower shard*/, total_shards - 1/*upper shard*/);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::initialize_equivalence_set_kd_tree(
                     EqKDTree *tree, EquivalenceSet *set, const FieldMask &mask,
                     ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      if (realm_index_space.empty())
      {
        // For backwards compatibility we handle the empty case which will
        // still store an equivalence set with names in it even though it
        // doesn't have to be updated ever
#ifdef DEBUG_LEGION
        assert(realm_index_space.bounds.empty());
        assert(typed_tree->bounds == realm_index_space.bounds);
#endif
        typed_tree->initialize_set(set, realm_index_space.bounds, mask,
                                   local_shard, current);
      }
      else
      {
        for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
              itr.valid; itr.step())
        {
          const Rect<DIM,T> overlap = itr.rect.intersection(typed_tree->bounds);
          if (!overlap.empty())
            typed_tree->initialize_set(set, overlap, mask, local_shard,current);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::compute_equivalence_sets(
          EqKDTree *tree, LocalLock *tree_lock, const FieldMask &mask,
          const std::vector<EqSetTracker*> &trackers,
          const std::vector<AddressSpaceID> &tracker_spaces,
          std::vector<unsigned> &new_tracker_references,
          FieldMaskSet<EquivalenceSet> &eq_sets,
          std::vector<RtEvent> &pending_sets,
          FieldMaskSet<EqKDTree> &subscriptions,
          FieldMaskSet<EqKDTree> &to_create,
          std::map<EqKDTree*,Domain> &creation_rects,
          std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock,1,false/*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
        typed_tree->compute_equivalence_sets(itr.rect, mask, trackers,
            tracker_spaces, new_tracker_references, eq_sets, pending_sets,
            subscriptions, to_create, creation_rects, creation_srcs,
            remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned IndexSpaceNodeT<DIM,T>::record_output_equivalence_set(
          EqKDTree *tree, LocalLock *tree_lock, EquivalenceSet *set,
          const FieldMask &mask, EqSetTracker *tracker,
          AddressSpaceID tracker_space, FieldMaskSet<EqKDTree> &subscriptions,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      unsigned new_subs = 0;
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock,1,false/*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
        new_subs += typed_tree->record_output_equivalence_set(set, itr.rect,
            mask, tracker, tracker_space, subscriptions, remote_shard_rects,
            local_shard);
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::invalidate_equivalence_set_kd_tree(
                   EqKDTree *tree, LocalLock *tree_lock, const FieldMask &mask,
                   std::vector<RtEvent> &invalidated, bool move_to_previous)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      // Need exclusive access to the tree for invalidations
      AutoLock t_lock(*tree_lock);
      for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
        typed_tree->invalidate_tree(itr.rect, mask, context->runtime,
                                    invalidated, move_to_previous);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM,T>::invalidate_shard_equivalence_set_kd_tree(
        EqKDTree *tree, LocalLock *tree_lock, const FieldMask &mask,
        std::vector<RtEvent> &invalidated,
        std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> realm_index_space;
      get_realm_index_space(realm_index_space, true/*tight*/);
      EqKDTreeT<DIM,T> *typed_tree = tree->as_eq_kd_tree<DIM,T>();
      // Need exclusive access to the tree for invalidations
      AutoLock t_lock(*tree_lock);
      for (Realm::IndexSpaceIterator<DIM,T> itr(realm_index_space); 
            itr.valid; itr.step())
        typed_tree->invalidate_shard_tree_remote(itr.rect, mask,
            context->runtime, invalidated, remote_shard_rects, local_shard);
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
        context->find_domain(shard_space, sharding_domain);
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
      if (shard_space.exists() && (shard_space != handle))
        context->find_domain(shard_space, sharding_domain);
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

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM,T>::has_shard_participants(ShardingFunction *func,
                                  ShardID shard, IndexSpace shard_space,
                                  const std::vector<DomainPoint> &shard_points,
                                  const Domain &shard_domain)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> local_space;
      get_realm_index_space(local_space, true/*tight*/);
      Domain sharding_domain;
      if (shard_space.exists() && (shard_space != handle))
        context->find_domain(shard_space, sharding_domain);
      else
        sharding_domain = local_space;
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
              return true;
          }
        }
        return false;
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
        return !domain_points.empty();
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
    template<int DIM, typename T, bool BY_RECTS>
    /*static*/ inline bool KDTree::compute_best_splitting_plane(
        const Rect<DIM,T> &bounds, const std::vector<Rect<DIM,T> > &rects,
        Rect<DIM,T> &best_left_bounds, Rect<DIM,T> &best_right_bounds,
        std::vector<Rect<DIM,T> > &best_left_set,
        std::vector<Rect<DIM,T> > &best_right_set)
    //--------------------------------------------------------------------------
    {
      // If we have sub-optimal bad sets we will track them here
      // so we can iterate through other dimensions to look for
      // better splitting planes
      int best_dim = -1;
      float best_cost = 2.f; // worst possible cost
      for (int d = 0; d < DIM; d++)
      {
        // Try to compute a splitting plane for this dimension
        // Count how many rectangles start and end at each location
        std::map<std::pair<T,bool/*stop*/>,uint64_t> forward_lines;
        std::map<std::pair<T,bool/*start*/>,uint64_t> backward_lines;
        for (unsigned idx = 0; idx < rects.size(); idx++)
        {
          const Rect<DIM,T> &subset_bounds = rects[idx];
          // Start forward
          std::pair<T,bool> start_key(subset_bounds.lo[d],false);
          typename std::map<std::pair<T,bool>,uint64_t>::iterator finder =
            forward_lines.find(start_key);
          if (finder == forward_lines.end())
          {
            if (BY_RECTS)
              forward_lines[start_key] = 1;
            else
              forward_lines[start_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second++;
            else
              finder->second += subset_bounds.volume();
          }
          // Start backward 
          start_key.second = true;
          finder = backward_lines.find(start_key);
          if (finder == backward_lines.end())
          {
            if (BY_RECTS)
              backward_lines[start_key] = 1;
            else
              backward_lines[start_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second++;
            else
              finder->second += subset_bounds.volume();
          }
          // Stop forward 
          std::pair<T,uint64_t> stop_key(subset_bounds.hi[d],true);
          finder = forward_lines.find(stop_key);
          if (finder == forward_lines.end())
          {
            if (BY_RECTS)
              forward_lines[stop_key] = 1;
            else
              forward_lines[stop_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second += 1;
            else
              finder->second += subset_bounds.volume();
          }
          // Stop backward 
          stop_key.second = false;
          finder = backward_lines.find(stop_key);
          if (finder == backward_lines.end())
          {
            if (BY_RECTS)
              backward_lines[stop_key] = 1;
            else
              backward_lines[stop_key] = subset_bounds.volume();
          }
          else
          {
            if (BY_RECTS)
              finder->second++;
            else
              finder->second += subset_bounds.volume();
          }
        }
        // Construct two lists by scanning from left-to-right and
        // from right-to-left of the number of rectangles that would
        // be inlcuded on the left or right side by each splitting plane
        std::map<T,uint64_t> lower_inclusive, upper_exclusive;
        uint64_t total = 0;
        for (typename std::map<std::pair<T,bool>,uint64_t>::const_iterator
              it = forward_lines.begin(); it != forward_lines.end(); it++)
        {
          // Increment first for starts for inclusivity
          if (!it->first.second)
            total += it->second;
          // Always record the count for all splits
          lower_inclusive[it->first.first] = total;
        }
        // If all the lines exist at the same value
        // then we'll never have a splitting plane
        if (lower_inclusive.size() == 1)
          continue;
        total = 0;
        for (typename std::map<
              std::pair<T,bool>,uint64_t>::const_reverse_iterator it = 
              backward_lines.rbegin(); it != backward_lines.rend(); it++)
        {
          // Always record the count for all splits
          upper_exclusive[it->first.first] = total;
          // Increment last for stops for exclusivity
          if (!it->first.second)
            total += it->second;
        }
#ifdef DEBUG_LEGION
        assert(lower_inclusive.size() == upper_exclusive.size());
#endif
        // We want to take the mini-max of the two numbers in order
        // to try to balance the splitting plane across the two sets
        T split = 0;
        if (!BY_RECTS)
        {
          total = 0;
          for (unsigned idx = 0; idx < rects.size(); idx++)
            total += rects[idx].volume();
        }
        else
          total = rects.size();
        uint64_t split_max = total;
        for (typename std::map<T,uint64_t>::const_iterator it = 
              lower_inclusive.begin(); it != lower_inclusive.end(); it++)
        {
          const uint64_t left = it->second;
          const uint64_t right = upper_exclusive[it->first];
          const uint64_t max = (left > right) ? left : right;
          if (max < split_max)
          {
            split_max = max;
            split = it->first;
          }
        }
        // Check for the case where we can't find a splitting plane
        if (split_max == total)
          continue;
        // Sort the subsets into left and right
        Rect<DIM,T> left_bounds(bounds);
        Rect<DIM,T> right_bounds(bounds);
        left_bounds.hi[d] = split;
        right_bounds.lo[d] = split+1;
        std::vector<Rect<DIM,T> > left_set, right_set;
        for (typename std::vector<Rect<DIM,T> >::const_iterator it =
              rects.begin(); it != rects.end(); it++)
        {
          const Rect<DIM,T> left_rect = it->intersection(left_bounds);
          if (!left_rect.empty())
            left_set.push_back(left_rect);
          const Rect<DIM,T> right_rect = it->intersection(right_bounds);
          if (!right_rect.empty())
            right_set.push_back(right_rect);
        }
#ifdef DEBUG_LEGION
        assert(left_set.size() < rects.size());
        assert(right_set.size() < rects.size());
#endif
        // Compute the cost of this refinement
        // First get the percentage reductions of both sets
        float cost_left, cost_right;
        if (BY_RECTS)
        {
          cost_left = float(left_set.size()) / float(rects.size());
          cost_right = float(right_set.size()) / float(rects.size());
        }
        else
        {
          uint64_t volume = 0;
          for (typename std::vector<Rect<DIM,T> >::const_iterator it =
                left_set.begin(); it != left_set.end(); it++)
            volume += it->volume();
          cost_left = float(volume) / float(total);
          volume = 0;
          for (typename std::vector<Rect<DIM,T> >::const_iterator it =
                right_set.begin(); it != right_set.end(); it++)
            volume += it->volume();
          cost_right = float(volume) / float(total);
        }
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
      return (best_dim >= 0);
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
      Rect<DIM,T> best_left_bounds, best_right_bounds;
      std::vector<Rect<DIM,T> > best_left_set, best_right_set;
      bool success = compute_best_splitting_plane<DIM,T>(bounds, subrects,
          best_left_bounds, best_right_bounds, best_left_set, best_right_set);
      // See if we had at least one good refinement
      if (success)
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
    // Equivalence Set KD Tree
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline EqKDTreeT<DIM,T>* EqKDTree::as_eq_kd_tree(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      EqKDTreeT<DIM,T> *result = dynamic_cast<EqKDTreeT<DIM,T>*>(this);
      assert(result != NULL);
      return result;
#else
      return static_cast<EqKDTreeT<DIM,T>*>(this);
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDTreeT<DIM,T>::EqKDTreeT(const Rect<DIM,T> &rect)
      : bounds(rect)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDTreeT<DIM,T>::compute_shard_equivalence_sets(
          const Domain &domain, const FieldMask &mask,
          const std::vector<EqSetTracker*> &trackers,
          const std::vector<AddressSpaceID> &tracker_spaces,
          std::vector<unsigned> &new_tracker_references,
          FieldMaskSet<EquivalenceSet> &eq_sets,
          std::vector<RtEvent> &pending_sets,
          FieldMaskSet<EqKDTree> &subscriptions,
          FieldMaskSet<EqKDTree> &to_create,
          std::map<EqKDTree*,Domain> &creation_rects,
          std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      const Rect<DIM,T> rect = domain;
      std::map<ShardID,LegionMap<Domain,FieldMask> > remote_shard_rects;
      compute_equivalence_sets(rect, mask, trackers, tracker_spaces,
          new_tracker_references, eq_sets, pending_sets, subscriptions,
          to_create, creation_rects, creation_srcs, remote_shard_rects,
          local_shard);
#ifdef DEBUG_LEGION
      // Should not have any of these at this point
      assert(remote_shard_rects.empty());
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDTreeT<DIM,T>::record_shard_output_equivalence_set(
        EquivalenceSet *set, const Domain &domain, const FieldMask &mask,
        EqSetTracker *tracker, AddressSpaceID tracker_space,
        FieldMaskSet<EqKDTree> &new_subscriptions, ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      const Rect<DIM,T> rect = domain;
      std::map<ShardID,LegionMap<Domain,FieldMask> > remote_shard_rects;
      unsigned references = record_output_equivalence_set(set, rect, mask, 
          tracker, tracker_space, new_subscriptions, remote_shard_rects, 
          local_shard);
#ifdef DEBUG_LEGION
      // Should not have any of these at this point
      assert(remote_shard_rects.empty());
#endif
      return references;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* EqKDTreeT<DIM,T>::create_from_rectangles(
               RegionTreeForest *forest, const std::vector<Domain> &rects) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!rects.empty());
#endif
      std::vector<Rect<DIM,T> > rectangles(rects.size());
      for (unsigned idx = 0; idx < rects.size(); idx++)
        rectangles[idx] = rects[idx];
      InternalExpression<DIM,T> *result = new InternalExpression<DIM,T>(
          &rectangles.front(), rectangles.size(), forest);
      // Do a little test to see if there is already a canonical expression
      // that we know about that matches this expression if so we'll use that
      // Note that we don't need to explicitly delete it if it is not the
      // canonical expression since it has a live expression reference that
      // will be cleaned up after this meta-task is done running
      return result->get_canonical_expression(forest);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDTreeT<DIM,T>::invalidate_shard_tree(const Domain &domain,
     const FieldMask &mask, Runtime *runtime, std::vector<RtEvent> &invalidated)
    //--------------------------------------------------------------------------
    {
      const Rect<DIM,T> rect = domain;
      invalidate_tree(rect, mask, runtime,invalidated,true/*move to previous*/);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDNode<DIM,T>::EqKDNode(const Rect<DIM,T> &rect)
      : EqKDTreeT<DIM,T>(rect), lefts(NULL), rights(NULL), current_sets(NULL),
        previous_sets(NULL), current_set_preconditions(NULL),
        pending_set_creations(NULL), pending_postconditions(NULL),
        subscriptions(NULL), child_previous_below(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDNode<DIM,T>::~EqKDNode(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(subscriptions == NULL);
      assert(pending_set_creations == NULL);
      assert(pending_postconditions == NULL);
#endif
      if (lefts != NULL)
      {
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
              lefts->begin(); it != lefts->end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        delete lefts;
      }
      if (rights != NULL)
      {
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
              rights->begin(); it != rights->end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        delete rights;
      }
      if (current_sets != NULL)
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              current_sets->begin(); it != current_sets->end(); it++)
          if (it->first->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
            delete it->first;
        delete current_sets;
      }
      if (previous_sets != NULL)
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              previous_sets->begin(); it != previous_sets->end(); it++)
          if (it->first->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
            delete it->first;
        delete previous_sets;
      }
      if (current_set_preconditions != NULL)
        delete current_set_preconditions;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::initialize_set(EquivalenceSet *set, 
                                 const Rect<DIM,T> &rect, const FieldMask &mask,
                                 ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<EqKDNode<DIM,T> > to_traverse;
      {
        FieldMask remaining, unrefined = mask;
        AutoLock n_lock(node_lock);
        if (lefts != NULL)
          unrefined -= lefts->get_valid_mask();
        if (!!unrefined)
        {
          if (rect == this->bounds)
          {
            if (current)
            {
              if (current_sets == NULL)
                current_sets = new FieldMaskSet<EquivalenceSet>();
              if (current_sets->insert(set, unrefined))
                set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
            }
            else
            {
              if (previous_sets == NULL)
                previous_sets = new FieldMaskSet<EquivalenceSet>();
              if (previous_sets->insert(set, unrefined))
                set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
            }
            remaining = mask - unrefined;
            if (!remaining)
              return;
          }
          else
          {
            // Refine for all the fields which aren't refined yet
            refine_node(rect, unrefined);
            if (!current)
              all_previous_below |= unrefined;
            remaining = mask;
          }
        }
        else
          remaining = mask;
        // If we get here, we're traversing refinements
        if (current && !!all_previous_below)
          all_previous_below -= remaining;
#ifdef DEBUG_LEGION
        assert(!!remaining);
        assert((lefts != NULL) && (rights != NULL));
#endif
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
              lefts->begin(); it != lefts->end(); it++)
        {
          const FieldMask overlap = remaining & it->second;
          if (!overlap)
            continue;
          Rect<DIM,T> intersection = rect.intersection(it->first->bounds);
          if (intersection.empty())
            continue;
          to_traverse.insert(it->first, overlap);
          if (intersection == rect)
          {
            remaining -= overlap;
            if (!remaining)
              break;
          }
        }
        if (!!remaining)
        {
          for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                rights->begin(); it != rights->end(); it++)
          {
            const FieldMask overlap = remaining & it->second;
            if (!overlap)
              continue;
#ifdef DEBUG_LEGION
            assert(rect.overlaps(it->first->bounds));
#endif
            to_traverse.insert(it->first, overlap);
            remaining -= overlap;
            if (!remaining)
              break;
          }
        }
      }
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        const Rect<DIM,T> overlap = rect.intersection(it->first->bounds);
#ifdef DEBUG_LEGION
        assert(!overlap.empty());
#endif
        it->first->initialize_set(set, overlap, it->second,
                                  local_shard, current);   
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::compute_equivalence_sets( 
          const Rect<DIM,T> &rect, const FieldMask &mask,
          const std::vector<EqSetTracker*> &trackers,
          const std::vector<AddressSpaceID> &tracker_spaces,
          std::vector<unsigned> &new_tracker_references,
          FieldMaskSet<EquivalenceSet> &eq_sets,
          std::vector<RtEvent> &pending_sets,
          FieldMaskSet<EqKDTree> &new_subscriptions,
          FieldMaskSet<EqKDTree> &to_create,
          std::map<EqKDTree*,Domain> &creation_rects,
          std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
#endif
      FieldMaskSet<EqKDNode<DIM,T> > 
        to_traverse, to_get_previous, to_invalidate_previous;
      {
        FieldMask remaining = mask;
        AutoLock n_lock(node_lock);
        // First check to see if we have any current equivalence sets 
        // here which means we can just record them
        if ((current_sets != NULL) && 
            !(remaining * current_sets->get_valid_mask()))
        {
          FieldMask check_preconditions;
          for (FieldMaskSet<EquivalenceSet>::const_iterator cit =
                current_sets->begin(); cit != current_sets->end(); cit++)
          {
            const FieldMask overlap = remaining & cit->second;
            if (!overlap)
              continue;
            eq_sets.insert(cit->first, overlap);
            remaining -= overlap;
            new_subscriptions.insert(this, overlap);
            for (unsigned idx = 0; idx < trackers.size(); idx++)
              new_tracker_references[idx] += record_subscription(
                  trackers[idx], tracker_spaces[idx], overlap);
            if (current_set_preconditions != NULL)
              check_preconditions |= overlap;
            if (!remaining)
              break;
          }
          if (!!check_preconditions)
          {
            // Check to see if there are any pending set creation events
            // still valid for this event, if so we still need to 
            // record them to make sure we don't try to use this set
            // until it is actually ready
            for (LegionMap<RtEvent,FieldMask>::iterator it =
                  current_set_preconditions->begin(); it !=
                  current_set_preconditions->end(); /*nothing*/)
            {
              if (!it->first.has_triggered())
              {
                if (!(check_preconditions * it->second))
                {
                  pending_sets.push_back(it->first);
                  check_preconditions -= it->second;
                  if (!check_preconditions)
                    break;
                }
                it++;
              }
              else
              {
                // Perform the previous invalidations now that the
                // event has triggered
                invalidate_previous_sets(it->second, to_invalidate_previous);
                LegionMap<RtEvent,FieldMask>::iterator to_delete = it++;
                current_set_preconditions->erase(to_delete);
              }
            }
            if (current_set_preconditions->empty())
            {
              delete current_set_preconditions;
              current_set_preconditions = NULL;
            }
          }
        }
        if (!!remaining)
        {
          // if we still have remaining fields, check for any pending
          // sets that might be in the process of being made
          if (pending_set_creations != NULL)
          {
            for (LegionMap<RtUserEvent,FieldMask>::const_iterator it =
                  pending_set_creations->begin(); it !=
                  pending_set_creations->end(); it++)
            {
              const FieldMask overlap = remaining & it->second;
              if (!overlap)
                continue;
              pending_sets.push_back(it->first);
              remaining -= overlap;
              new_subscriptions.insert(this, overlap);
              for (unsigned idx = 0; idx < trackers.size(); idx++)
                new_tracker_references[idx] += record_subscription(
                    trackers[idx], tracker_spaces[idx], overlap);
              if (!remaining)
                break;
            }
          }
          if (!!remaining)
          {
            // Next check to see if we have to traverse below any nodes
            // below because they have been refined. If they're all previous
            // below and we're trying to make equivalence sets here then we
            // can skip traversing below since we'll be able to coarsen
            FieldMask to_coarsen;
            if (!!all_previous_below && (rect == this->bounds))
            {
              to_coarsen = remaining & all_previous_below;
              remaining -= to_coarsen;
            }
            if ((lefts != NULL) && !(remaining * lefts->get_valid_mask()))
            {
              FieldMask right_mask;
              for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                    lefts->begin(); it != lefts->end(); it++)
              {
                const FieldMask overlap = it->second & remaining;
                if (!overlap)
                  continue;
                if (it->first->bounds.overlaps(rect))
                {
                  to_traverse.insert(it->first, overlap);
                  if (!it->first->bounds.contains(rect))
                    right_mask |= overlap;
                }
                else
                  right_mask |= overlap;
                remaining -= overlap;
                if (!remaining)
                  continue;
              }
              if (!!right_mask)
              {
                for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator
                      it = rights->begin(); it != rights->end(); it++)
                {
                  const FieldMask overlap = it->second & right_mask;
                  if (!overlap)
                    continue;
                  if (it->first->bounds.overlaps(rect))
                    to_traverse.insert(it->first, overlap);
                  right_mask -= overlap;
                  if (!right_mask)
                    break;
                }
              }
            }
            // Re-introduce the fields we want to try to refine here
            if (!!to_coarsen)
              remaining |= to_coarsen;
            if (!!remaining)
            {
              // if we still have remaining fields, then we're going to 
              // be the ones to make the equivalence set for these fields
              // check to see if the rect is the same as our bounds or
              // whether it is smaller
              if (rect == this->bounds)
              {
                // easy case, we can just record that we're making a
                // new equivalence set at this node
                if (pending_set_creations == NULL)
                  pending_set_creations =
                    new LegionMap<RtUserEvent,FieldMask>();
                const RtUserEvent ready = Runtime::create_rt_user_event();
                pending_set_creations->insert(
                    std::make_pair(ready, remaining));
                // Record the subscription now so we know whether to 
                // add a reference to the tracker or not
                for (unsigned idx = 0; idx < trackers.size(); idx++)
                  new_tracker_references[idx] += record_subscription(
                      trackers[idx], tracker_spaces[idx], remaining);
                to_create.insert(this, remaining);
                creation_rects[this] = Domain(rect);
                // Find any creation sources
                if ((previous_sets != NULL) && 
                    !(remaining * previous_sets->get_valid_mask()))
                {
                  for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                        previous_sets->begin(); it !=
                        previous_sets->end(); it++)
                  {
                    const FieldMask overlap = it->second & remaining;
                    if (!overlap)
                      continue;
                    creation_srcs[it->first][this->bounds] |= overlap;
                    remaining -= overlap;
                    if (!remaining)
                      break;
                  }
                }
                // Check for any previous sets below us to get as well
                if (!!remaining && !!all_previous_below)
                {
                  FieldMask all_prev_below = all_previous_below & remaining;
                  if (!!all_prev_below)
                    // These fields will no longer be all_prevous_below now
                    find_to_get_previous(all_prev_below, to_get_previous);
                }
              }
              else
              {
                refine_node(rect, remaining);
#ifdef DEBUG_LEGION
                assert((lefts != NULL) && (rights != NULL));
#endif
                for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator 
                      it = lefts->begin(); it != lefts->end(); it++)
                {
                  const FieldMask overlap = remaining & it->second;
                  if (!overlap)
                    continue;
                  const Rect<DIM,T> intersection = 
                    rect.intersection(it->first->bounds);
                  if (intersection.empty())
                    continue;
                  to_traverse.insert(it->first, overlap);
                  if (intersection == rect)
                  {
                    remaining -= overlap;
                    if (!remaining)
                      break;
                  }
                }
                if (!!remaining)
                {
                  for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator
                        it = rights->begin(); it != rights->end(); it++)
                  {
                    const FieldMask overlap = remaining & it->second;
                    if (!overlap)
                      continue;
#ifdef DEBUG_LEGION
                    assert(rect.overlaps(it->first->bounds));
#endif
                    to_traverse.insert(it->first, overlap);
                    remaining -= overlap;
                    if (!remaining)
                      break;
                  }
                }
              }
            }
          }
        }
        // If we're traversing for any fields then remove them from the set
        // of all previous below since we know what we'll no longer have
        // all previous below at this point
        if (!to_traverse.empty())
        {
          if (!!all_previous_below)
            all_previous_below -= to_traverse.get_valid_mask();
          // Also filter the individual child previous below
          if ((child_previous_below != NULL) &&
              !(to_traverse.get_valid_mask() * 
                child_previous_below->get_valid_mask()))
          {
            for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                  to_traverse.begin(); it != to_traverse.end(); it++)
            {
              typename FieldMaskSet<EqKDNode<DIM,T> >::iterator finder =
                child_previous_below->find(it->first);
              if (finder == child_previous_below->end())
                continue;
              finder.filter(it->second);
              if (!finder->second)
                child_previous_below->erase(finder);
            }
            if (child_previous_below->empty())
            {
              delete child_previous_below;
              child_previous_below = NULL;
            }
            else
              child_previous_below->tighten_valid_mask();
          }
        }
      }
#ifdef DEBUG_LEGION
      assert(to_traverse.get_valid_mask() * to_get_previous.get_valid_mask());
#endif
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        const Rect<DIM,T> overlap = rect.intersection(it->first->bounds);
#ifdef DEBUG_LEGION
        assert(!overlap.empty());
#endif
        it->first->compute_equivalence_sets(overlap, it->second, trackers, 
            tracker_spaces, new_tracker_references, eq_sets, pending_sets,
            new_subscriptions, to_create, creation_rects, creation_srcs,
            remote_shard_rects, local_shard);
      }
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_get_previous.begin(); it != to_get_previous.end(); it++)
        it->first->find_all_previous_sets(it->second, creation_srcs); 
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_invalidate_previous.begin(); it != 
            to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::find_all_previous_sets(FieldMask mask,
          std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs)
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<EqKDNode<DIM,T> > to_get_previous;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        if (previous_sets != NULL)
        {
          for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                previous_sets->begin(); it != previous_sets->end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            creation_srcs[it->first][this->bounds] |= overlap;
            mask -= overlap;
            if (!mask)
              return;
          }
        }
#ifdef DEBUG_LEGION
        assert(!(mask - all_previous_below));
#endif
        find_to_get_previous(mask, to_get_previous);
      }
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_get_previous.begin(); it != to_get_previous.end(); it++)
        it->first->find_all_previous_sets(it->second, creation_srcs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::find_to_get_previous(FieldMask &all_prev_below,
                          FieldMaskSet<EqKDNode<DIM,T> > &to_get_previous) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((lefts != NULL) && (rights != NULL));
#endif
      // We're going to pull these out of the lefts and rights
      // since we're just going to use them to get the sets
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            lefts->begin(); it != lefts->end(); it++)
      {
        const FieldMask overlap = all_prev_below & it->second;
        if (!overlap)
          continue;
        to_get_previous.insert(it->first, overlap);
      }
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            rights->begin(); it != rights->end(); it++)
      {
        const FieldMask overlap = all_prev_below & it->second;
        if (!overlap)
          continue;
        to_get_previous.insert(it->first, overlap);
        all_prev_below -= overlap;
        if (!all_prev_below)
          break;
      }
#ifdef DEBUG_LEGION
      assert(!all_prev_below);
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::refine_node(const Rect<DIM,T> &rect,  
                                      const FieldMask &mask,bool refine_current)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // We shouldn't have any existing refinements for these fields
      assert((lefts == NULL) || (mask * lefts->get_valid_mask()));
      assert((rights == NULL) || (mask * rights->get_valid_mask()));
      assert(mask * all_previous_below);
      assert((child_previous_below == NULL) || 
          (mask * child_previous_below->get_valid_mask()));
      // We shouldn't have any current sets either for these fields
      assert((current_sets == NULL) || 
          (mask * current_sets->get_valid_mask()) || refine_current);
      if (pending_set_creations != NULL)
      {
        // Invalidations should never be racing with pending set
        // creations, this should be guaranteed by the logical dependence
        // analysis which ensures refinements are serialized with respect
        // to all other operations
        for (LegionMap<RtUserEvent,FieldMask>::const_iterator it =
              pending_set_creations->begin(); it !=
              pending_set_creations->end(); it++)
          assert(mask * it->second);
      }
      if (subscriptions != NULL)
      {
        // We should never be refining something which has subscribers
        for (LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> >::
              const_iterator it = subscriptions->begin(); it !=
              subscriptions->end(); it++)
          assert(mask * it->second.get_valid_mask());
      }
#endif
      // Need to create a new refinement for these fields
      // to match the rectangle being requested
      // First check to see if we can find a dimension where
      // the half-way cutting plane puts the rectangle on 
      // one side or the other, if we can find such a
      // dimension then we'll just split on the biggest one
      // and continue on our way. If we can't find such a 
      // dimension we'll split on the dimension that where the
      // edge of the rectangle falls closest to the halfway point
      T split = 0;
      int dim = -1;
      T largest = 0;
      for (int d = 0; d < DIM; d++)
      {
        if (this->bounds.lo[d] == this->bounds.hi[d])
          continue;
        T diff = this->bounds.hi[d] - this->bounds.lo[d];
        T mid = this->bounds.lo[d] + (diff / 2);
        if ((rect.hi[d] <= mid) || (mid < rect.lo[d]))
        {
          if ((dim < 0) || (largest < diff))
          {
            dim = d;
            split = mid;
            largest = diff;
          }
        }
      }
      if (dim < 0)
      {
        // We couldn't find a nice splitting dimension, so
        // we're now going to find the one with an edge on
        // the rectangle that is closest to the middle splitting
        // point. We're guaranteed that such a split must exist
        // because the rect and the bounds are not equal
        T distance = 0;
        for (int d = 0; d < DIM; d++)
        {
          if (this->bounds.lo[d] == this->bounds.hi[d])
            continue;
          T diff = this->bounds.hi[d] - this->bounds.lo[d];
          T mid = this->bounds.lo[d] + (diff / 2);
          if (this->bounds.lo[d] < rect.lo[d])
          {
            T dist = ((rect.lo[d]-1) <= mid) ?
              mid - (rect.lo[d]-1) : (rect.lo[d]-1) - mid;
            if ((dim < 0) || (dist < distance))
            {
              dim = d;
              split = rect.lo[d]-1;
              distance = dist;
            }
          }
          if (rect.hi[d] < this->bounds.hi[d])
          {
            T dist = (rect.hi[d] <= mid) ?
              mid - rect.hi[d] : rect.hi[d] - mid;
            if ((dim < 0) || (dist < distance))
            {
              dim = d;
              split = rect.hi[d];
              distance = dist;
            }
          }
        }
#ifdef DEBUG_LEGION
        assert(dim >= 0);
#endif
      }
      Rect<DIM,T> left_bounds = this->bounds;
      Rect<DIM,T> right_bounds = this->bounds;
      left_bounds.hi[dim] = split;
      right_bounds.lo[dim] = split+1;
      // See if we can reuse any existing subnodes or whether we
      // need to make new ones
      if (lefts != NULL)
      {
        EqKDNode<DIM,T> *prior_left = NULL;
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::iterator it =
              lefts->begin(); it != lefts->end(); it++)
        {
          if (it->first->bounds != left_bounds)
            continue;
          prior_left = it->first;
          it.merge(mask);
          break;
        }
        if (prior_left != NULL)
        {
          EqKDNode<DIM,T> *prior_right = NULL;
          for (typename FieldMaskSet<EqKDNode<DIM,T> >::iterator it =
                rights->begin(); it != rights->end(); it++)
          {
            if (it->first->bounds != right_bounds)
              continue;
            prior_right = it->first;
            it.merge(mask);
            break;
          }
#ifdef DEBUG_LEGION
          assert(prior_right != NULL);
          assert(left_bounds.contains(rect) || 
              right_bounds.contains(rect));
#endif
          if (previous_sets != NULL)
          {
            if (!refine_current)
              all_previous_below |= mask & previous_sets->get_valid_mask();
            clone_sets(prior_left, prior_right, mask, previous_sets, false);
          }
          if (refine_current)
            clone_sets(prior_left, prior_right, mask, current_sets, true);
          return;
        }
      }
      // If we still have remaining fields, then we need to
      // make new left and right nodes
      EqKDNode<DIM,T> *new_left =
        new EqKDNode<DIM,T>(left_bounds);
      EqKDNode<DIM,T> *new_right =
        new EqKDNode<DIM,T>(right_bounds);
      if (lefts == NULL)
        lefts = new FieldMaskSet<EqKDNode<DIM,T> >();
      if (lefts->insert(new_left, mask))
        new_left->add_reference();
      if (rights == NULL)
        rights = new FieldMaskSet<EqKDNode<DIM,T> >();
      if (rights->insert(new_right, mask))
        new_right->add_reference();
      if (previous_sets != NULL)
      {
        if (!refine_current)
          all_previous_below |= mask & previous_sets->get_valid_mask();
        clone_sets(new_left, new_right, mask, previous_sets, false/*current*/);
      }
      if (refine_current)
        clone_sets(new_left, new_right, mask, current_sets, true/*current*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::record_equivalence_set(EquivalenceSet *set,
                                  const FieldMask &mask, RtEvent ready,
                                  const CollectiveMapping &creator_spaces,
                                  const std::vector<EqSetTracker*> &creators)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!creators.empty());
      assert(creator_spaces.size() == creators.size());
#endif
      FieldMaskSet<EqKDNode<DIM,T> > to_invalidate_previous;
      {
        AutoLock n_lock(node_lock);
        if (current_sets == NULL)
          current_sets = new FieldMaskSet<EquivalenceSet>();
#ifdef DEBUG_LEGION
        assert(mask * current_sets->get_valid_mask());
#endif
        if (current_sets->insert(set, mask))
          set->add_base_gc_ref(DISJOINT_COMPLETE_REF); 
        // Send notifications to all the subscriptions that are waiting for 
        // the set to be sent to them
        if (subscriptions != NULL)
        {
          Runtime *runtime = set->runtime;
          for (LegionMap<AddressSpaceID,
                         FieldMaskSet<EqSetTracker> >::const_iterator sit =
                subscriptions->begin(); sit != subscriptions->end(); sit++)
          {
            if (sit->second.get_valid_mask() * mask)
              continue;
            // See if there is a creator to ignore on this space
            EqSetTracker *creator = NULL;
            if (creator_spaces.contains(sit->first))
            {
              const unsigned index = creator_spaces.find_index(sit->first);
              creator = creators[index];
            }
            if (sit->first != runtime->address_space)
            {
              FieldMaskSet<EqSetTracker> to_notify;
              for (FieldMaskSet<EqSetTracker>::const_iterator it =
                    sit->second.begin(); it != sit->second.end(); it++)
              {
                // Skip the creator tracker since it made it
                if (it->first == creator)
                  continue;
                const FieldMask overlap = mask & it->second;
                if (!overlap)
                  continue;
                to_notify.insert(it->first, overlap);
              }
              if (!to_notify.empty())
              {
                // Create an event for when this is triggered
                const RtUserEvent recorded = Runtime::create_rt_user_event();
                Serializer rez;
                {
                  RezCheck z(rez);
                  rez.serialize(set->did);
                  rez.serialize<size_t>(to_notify.size());
                  for (FieldMaskSet<EqSetTracker>::const_iterator it =
                        to_notify.begin(); it != to_notify.end(); it++)
                  {
                    rez.serialize(it->first);
                    rez.serialize(it->second);
                  }
                  rez.serialize(recorded);
                }
                runtime->send_compute_equivalence_sets_pending(sit->first, rez);
                // Save this event as a postcondition for any pending creations
#ifdef DEBUG_LEGION
                assert(pending_set_creations != NULL);
#endif
                for (LegionMap<RtUserEvent,FieldMask>::const_iterator it =
                      pending_set_creations->begin(); it !=
                      pending_set_creations->end(); it++)
                {
                  if (it->second * to_notify.get_valid_mask())
                    continue;
                  if (pending_postconditions == NULL)
                    pending_postconditions =
                      new std::map<RtUserEvent,std::vector<RtEvent> >();
                  (*pending_postconditions)[it->first].push_back(recorded);
                }
              }
            }
            else
            {
              // Local case so we can notify these directly
              for (FieldMaskSet<EqSetTracker>::const_iterator it =
                    sit->second.begin(); it != sit->second.end(); it++)
              {
                // Skip the creator tracker since it made it
                if (it->first == creator)
                  continue;
                const FieldMask overlap = mask & it->second;
                if (!overlap)
                  continue;
                it->first->record_pending_equivalence_set(set, overlap);
              }
            }
          }
        }
#ifdef DEBUG_LEGION
        assert(pending_set_creations != NULL);
#endif
        // Filter out any pending set creation events
        for (LegionMap<RtUserEvent,FieldMask>::iterator it = 
              pending_set_creations->begin(); it !=
              pending_set_creations->end(); /*nothing*/)
        {
          it->second -= mask;
          if (!it->second)
          {
            // Removed all the fields so now we can trigger the event
            // See if it has any postconditions
            if (pending_postconditions != NULL)
            {
              std::map<RtUserEvent,std::vector<RtEvent> >::iterator finder =
                pending_postconditions->find(it->first);
              if (finder != pending_postconditions->end())
              {
                if (ready.exists())
                  finder->second.push_back(ready);
                Runtime::trigger_event(it->first, 
                    Runtime::merge_events(finder->second));
                pending_postconditions->erase(finder);
                if (pending_postconditions->empty())
                {
                  delete pending_postconditions;
                  pending_postconditions = NULL;
                }
              }
              else
                Runtime::trigger_event(it->first, ready);
            }
            else
              Runtime::trigger_event(it->first, ready);
            LegionMap<RtUserEvent,FieldMask>::iterator to_delete = it++;
            pending_set_creations->erase(to_delete);
          }
          else
          {
            if (ready.exists() && !ready.has_triggered())
            {
              if (pending_postconditions == NULL)
                pending_postconditions =
                  new std::map<RtUserEvent,std::vector<RtEvent> >();
              (*pending_postconditions)[it->first].push_back(ready);
            }
            it++;
          }
        }
        if (pending_set_creations->empty())
        {
          delete pending_set_creations;
          pending_set_creations = NULL;
        }
        // If the ready event hasn't triggered when need to keep around
        // this event so no one tries to use the new equivalence set or
        // invalidate any of the previous sets it depends on until it
        // is actually ready to be used and all the clones are done
        if (ready.exists() && !ready.has_triggered())
        {
          if (current_set_preconditions == NULL)
            current_set_preconditions = new LegionMap<RtEvent,FieldMask>();
          current_set_preconditions->insert(std::make_pair(ready, mask));
        }
        else // we can invalidate the previous sets now
          // we can only do this if the ready event has triggered which
          // indicates that all the clone operations from the previous
          // sets are done and it's safe to remove the references
          invalidate_previous_sets(mask, to_invalidate_previous);
      }
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_invalidate_previous.begin(); it != 
            to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDNode<DIM,T>::record_output_equivalence_set(EquivalenceSet *set,
        const Rect<DIM,T> &rect, const FieldMask &mask,
        EqSetTracker *tracker, AddressSpaceID tracker_space,
        FieldMaskSet<EqKDTree> &subscriptions,
        std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      unsigned new_subs = 0;
      FieldMaskSet<EqKDNode<DIM,T> > to_traverse;
      {
        FieldMask submask = mask;
        AutoLock n_lock(node_lock);
        if (rect == this->bounds)
        {
          FieldMask local_fields = mask;
          if (lefts != NULL)
            local_fields -= lefts->get_valid_mask();
          if (!!local_fields)
          {
            // Record the set and subscriptions here
#ifdef DEBUG_LEGION
            assert((current_sets == NULL) || 
                (local_fields * current_sets->get_valid_mask()));
            assert((previous_sets == NULL) ||
                (local_fields * previous_sets->get_valid_mask()));
#endif
            if (current_sets == NULL)
              current_sets = new FieldMaskSet<EquivalenceSet>();
            if (current_sets->insert(set, local_fields))
              set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
            subscriptions.insert(this, local_fields);
            new_subs += record_subscription(tracker,tracker_space,local_fields);
            submask -= local_fields;
          }
        }
        else
        {
          FieldMask to_refine = mask;
          if (lefts != NULL)
            to_refine -= lefts->get_valid_mask();
          if (!!to_refine)
            refine_node(rect, to_refine);
        }
        // Check to see if there are any already refined nodes to traverse
        if (!!submask)
        {
#ifdef DEBUG_LEGION
          assert((lefts != NULL) && (rights != NULL));
#endif
          for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                lefts->begin(); it != lefts->end(); it++)
          {
            const FieldMask overlap = submask & it->second;
            if (!overlap)
              continue;
            if (!it->first->bounds.overlaps(rect))
              continue;
            to_traverse.insert(it->first, overlap);
            if (it->first->bounds.contains(rect))
            {
              submask -= overlap;
              if (!submask)
                break;
            }
          }
          if (!!submask)
          {
            for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                  rights->begin(); it != rights->end(); it++)
            {
              const FieldMask overlap = submask & it->second;
              if (!overlap)
                continue;
#ifdef DEBUG_LEGION
              assert(it->first->bounds.overlaps(rect));
#endif
              to_traverse.insert(it->first, overlap);
              submask -= overlap;
              if (!submask)
                break;
            }
          }
        }
      }
      // Continue the traversal for anything below
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        const Rect<DIM,T> overlap = it->first->bounds.intersection(rect);
        new_subs += it->first->record_output_equivalence_set(set, overlap,
            it->second, tracker, tracker_space, subscriptions, 
            remote_shard_rects, local_shard);
      }
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::invalidate_all_previous_sets(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<EqKDNode<DIM,T> > to_invalidate_previous;
      {
        AutoLock n_lock(node_lock);
        invalidate_previous_sets(mask, to_invalidate_previous);
      }
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_invalidate_previous.begin(); it != 
            to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::invalidate_previous_sets(const FieldMask &mask,
                         FieldMaskSet<EqKDNode<DIM,T> > &to_invalidate_previous)
    //--------------------------------------------------------------------------
    {
      if ((previous_sets != NULL) && 
          !(mask * previous_sets->get_valid_mask()))
      {
        std::vector<EquivalenceSet*> to_delete;
        for (FieldMaskSet<EquivalenceSet>::iterator it =
              previous_sets->begin(); it != previous_sets->end(); it++)
        {
          it.filter(mask);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<EquivalenceSet*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          previous_sets->erase(*it);
          if ((*it)->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
            delete (*it);
        }
        if (previous_sets->empty())
        {
          delete previous_sets;
          previous_sets = NULL;
        }
        else
          previous_sets->tighten_valid_mask();
      }
      if (!(mask * all_previous_below))
      {
#ifdef DEBUG_LEGION
        assert((lefts != NULL) && (rights != NULL));
#endif
        all_previous_below -= mask;
        std::vector<EqKDNode<DIM,T>*> to_delete;
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::iterator
              it = lefts->begin(); it != lefts->end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          to_invalidate_previous.insert(it->first, overlap);
          it.filter(overlap);
          if (!it->second)
            // Don't remove the refernce, it's in to_invalidate_previous
            to_delete.push_back(it->first);
          else
            it->first->add_reference();
        }
        for (typename std::vector<EqKDNode<DIM,T>*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
          lefts->erase(*it);
        if (lefts->empty())
        {
          delete lefts;
          lefts = NULL;
        }
        else
          lefts->tighten_valid_mask();
        to_delete.clear();
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::iterator
              it = rights->begin(); it != rights->end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          to_invalidate_previous.insert(it->first, overlap);
          it.filter(overlap);
          if (!it->second)
            // Don't remove the refernce, it's in to_invalidate_previous
            to_delete.push_back(it->first);
          else
            it->first->add_reference();
        }
        for (typename std::vector<EqKDNode<DIM,T>*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
          rights->erase(*it);
        if (rights->empty())
        {
          delete rights;
          rights = NULL;
        }
        else
          rights->tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDNode<DIM,T>::record_subscription(EqSetTracker *tracker,
        AddressSpaceID tracker_space, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (subscriptions == NULL)
        subscriptions =
          new LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> >();
      FieldMaskSet<EqSetTracker> &trackers = (*subscriptions)[tracker_space];
      FieldMaskSet<EqSetTracker>::iterator finder = trackers.find(tracker);
      if (finder != trackers.end())
      {
        FieldMask new_fields = mask - finder->second;
        if (!new_fields)
          return 0;
        trackers.insert(tracker, new_fields);
        const unsigned total_fields = new_fields.pop_count();
        this->add_reference(total_fields);
        return total_fields;
      }
      else
      {
        trackers.insert(tracker, mask);
        // Add a reference for every field
        const unsigned total_fields = mask.pop_count();
        this->add_reference(total_fields);
        return total_fields;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::clone_sets(EqKDNode<DIM,T> *left, 
        EqKDNode<DIM,T> *right, FieldMask mask,
        FieldMaskSet<EquivalenceSet> *&sets, bool current)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sets != NULL);
#endif
      std::vector<EquivalenceSet*> to_delete;
      for (FieldMaskSet<EquivalenceSet>::iterator it =
            sets->begin(); it != sets->end(); it++)
      {
        const FieldMask overlap = it->second & mask;
        if (!overlap)
          continue;
        left->record_set(it->first, overlap, current);
        right->record_set(it->first, overlap, current);
        it.filter(overlap);
        if (!it->second)
          to_delete.push_back(it->first);
        mask -= overlap;
        if (!mask)
          break;
      }
      for (std::vector<EquivalenceSet*>::const_iterator it =
            to_delete.begin(); it != to_delete.end(); it++)
      {
        sets->erase(*it); 
        if ((*it)->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
          delete (*it);
      }
      if (sets->empty())
      {
        delete sets;
        sets = NULL;
      }
      else
        sets->tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::record_set(EquivalenceSet *set, 
                                     const FieldMask &mask, bool current)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (current)
      {
        if (current_sets == NULL)
          current_sets = new FieldMaskSet<EquivalenceSet>();
        if (current_sets->insert(set, mask))
          set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
      }
      else
      {
        if (previous_sets == NULL)
          previous_sets = new FieldMaskSet<EquivalenceSet>();
        if (previous_sets->insert(set, mask))
          set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::find_local_equivalence_sets(
        FieldMaskSet<EquivalenceSet> &eq_sets, ShardID local_shard) const
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since this should be done exclusively
      // while nothing else is modifying the state of this tree
      if (current_sets != NULL)
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              current_sets->begin(); it != current_sets->end(); it++)
          eq_sets.insert(it->first, it->second);
      }
      if (previous_sets != NULL)
      {
        // Only record previous sets if we didn't have current sets
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              previous_sets->begin(); it != previous_sets->end(); it++)
        {
          if (current_sets != NULL)
          {
            FieldMask mask = it->second - current_sets->get_valid_mask();
            if (!!mask)
              eq_sets.insert(it->first, mask);
          }
          else
            eq_sets.insert(it->first, it->second);
        }
      }
      if (lefts != NULL)
      {
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
              lefts->begin(); it != lefts->end(); it++)
          it->first->find_local_equivalence_sets(eq_sets, local_shard);
      }
      if (rights != NULL)
      {
        for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
              rights->begin(); it != rights->end(); it++)
          it->first->find_local_equivalence_sets(eq_sets, local_shard);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::find_shard_equivalence_sets(
                                std::map<ShardID,LegionMap<RegionNode*,
                                     FieldMaskSet<EquivalenceSet> > > &eq_sets,
                            ShardID source_shard, ShardID dst_lower_shard,
                            ShardID dst_upper_shard, RegionNode *region) const
    //--------------------------------------------------------------------------
    {
      if ((dst_lower_shard == dst_upper_shard) || 
          (this->bounds.volume() <= EqKDSharded<DIM,T>::MIN_SPLIT_SIZE))
        find_local_equivalence_sets(eq_sets[dst_lower_shard][region], 
                                    source_shard);
      else
        // We still need to break up this rectangle the same way that it will
        // be broken up by a EqKDSharded node for these shards
        find_shard_equivalence_sets(this->bounds, eq_sets,
            dst_lower_shard, dst_upper_shard, region);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::find_shard_equivalence_sets(const Rect<DIM,T> &rect,
                                std::map<ShardID,LegionMap<RegionNode*,
                                     FieldMaskSet<EquivalenceSet> > > &eq_sets,
                            ShardID dst_lower_shard,
                            ShardID dst_upper_shard, RegionNode *region) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dst_lower_shard < dst_upper_shard);
#endif
      // Split this the same way that EqKDSharded will split it to find 
      // the rectangle that we need to use to search for equivalence sets
      // Check to see if we hit the minimum size
      if (rect.volume() <= EqKDSharded<DIM,T>::MIN_SPLIT_SIZE)
      {
        find_rect_equivalence_sets(rect, eq_sets[dst_lower_shard][region]); 
        return;
      }
      // Find the largest dimension and split it in half
      // Note we cannot use the rectangle to guide our splitting plane here
      // like we do with the EqKDNode because this splitting needs to be
      // deterministic across the shards
      T split = 0;
      int dim = -1;
      T largest = 0;
      for (int d = 0; d < DIM; d++)
      {
        T diff = rect.hi[d] - rect.lo[d];
        if (diff <= largest)
          continue;
        largest = diff;
        dim = d;
        split = rect.lo[d] + (diff / 2);
      }
#ifdef DEBUG_LEGION
      assert(dim >= 0);
#endif
      Rect<DIM,T> left_bounds = rect;
      Rect<DIM,T> right_bounds = rect;
      left_bounds.hi[dim] = split;
      right_bounds.lo[dim] = split+1;
      // Find the splitting of the shards
      ShardID diff = dst_upper_shard - dst_lower_shard;
      ShardID mid = dst_lower_shard + (diff  / 2);
      if (dst_lower_shard == mid)
        find_rect_equivalence_sets(left_bounds, eq_sets[mid][region]);
      else
        find_shard_equivalence_sets(left_bounds, eq_sets,
            dst_lower_shard, mid, region);
      if ((mid+1) == dst_upper_shard)
        find_rect_equivalence_sets(right_bounds, eq_sets[mid+1][region]);
      else
        find_shard_equivalence_sets(right_bounds, eq_sets,
            mid+1, dst_upper_shard, region);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::find_rect_equivalence_sets(const Rect<DIM,T> &rect,
                                    FieldMaskSet<EquivalenceSet> &eq_sets) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
#endif
      std::vector<EqKDNode<DIM,T>*> to_traverse;
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        if (current_sets != NULL)
        {
          for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                current_sets->begin(); it != current_sets->end(); it++)
            eq_sets.insert(it->first, it->second);
        }
        if (previous_sets != NULL)
        {
          FieldMask remaining = previous_sets->get_valid_mask();
          if (current_sets != NULL)
            remaining -= current_sets->get_valid_mask();
          if (!!remaining)
          {
            for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                  current_sets->begin(); it != current_sets->end(); it++)
            {
              const FieldMask overlap = it->second & remaining;
              if (!overlap)
                continue;
              eq_sets.insert(it->first, overlap); 
              remaining -= overlap;
              if (!remaining)
                break;
            }
          }
        }
        FieldMask remaining;
        if (lefts != NULL)
        {
          for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                lefts->begin(); it != lefts->end(); it++)
          {
            const Rect<DIM,T> overlap = it->first->bounds.intersection(rect);
            if (!overlap.empty())
            {
              to_traverse.push_back(it->first);
              if (overlap != rect)
                remaining |= it->second;
            }
            else
              remaining |= it->second;
          }
        }
        if (!!remaining)
        {
          for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                rights->begin(); it != rights->end(); it++)
          {
            const FieldMask overlap = it->second & remaining;
            if (!overlap)
              continue;
#ifdef DEBUG_LEGION
            assert(it->first->bounds.overlaps(rect));
#endif
            to_traverse.push_back(it->first);
            remaining -= overlap;
            if (!remaining)
              break;
          }
        }
      }
      for (typename std::vector<EqKDNode<DIM,T>*>::const_iterator it =
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        const Rect<DIM,T> overlap = (*it)->bounds.intersection(rect);
#ifdef DEBUG_LEGION
        assert(!overlap.empty());
#endif
        (*it)->find_rect_equivalence_sets(overlap, eq_sets); 
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::invalidate_tree(const Rect<DIM,T> &rect,
                                          const FieldMask &mask,
                                          Runtime *runtime,
                                          std::vector<RtEvent> &invalidated,
                                          bool move_to_previous,
                                          FieldMask *parent_all_previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
#endif
      // This is very important: invalidations are protected from the root
      // of the equivalence set tree with an exclusive lock, that means that
      // any invalidation traversing the tree does not need to protect against
      // races with compute_equivalence_set traversals since.
      // This property is what allows us to safely propagate the information
      // about the all_previous_below safely up the tree. While it's expensive
      // to have a whole-tree lock like this, invalidations being done by
      // refinements should be pretty rare so it shouldn't impact performance.
      // Note we still need to take the node lock here in this node because
      // calls like record_equivalence_set or cancel_subscription can still
      // be coming back asynchronously to touch data structures in these nodes
      FieldMask remaining = mask;
      FieldMaskSet<EqKDNode<DIM,T> > to_invalidate_previous;
      LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> > to_invalidate;
      {
        // Take the lock to protect against data structures that might
        // be racing with recording equivalence sets or cancelling subscriptions
        AutoLock n_lock(node_lock);
        // First check to see if there are any current sets which
        // haven't had their previous sets filtered yet. We have to
        // do this first to ensure that the lefts and rights and data
        // structures only contain real lefts and rights and not ones
        // that we were just holding on to for previous reasons
        if (current_set_preconditions != NULL)
        {
          for (LegionMap<RtEvent,FieldMask>::iterator it =
                current_set_preconditions->begin(); it !=
                current_set_preconditions->end(); /*nothing*/)
          {
            if (!(it->second * mask))
            {
#ifdef DEBUG_LEGION
              // Better have triggered by the point we're doing
              // this invalidation or something is wrong with the
              // mapping dependences for this refinement operation
              assert(it->first.has_triggered());
#endif
              invalidate_previous_sets(it->second, to_invalidate_previous);
              LegionMap<RtEvent,FieldMask>::iterator to_delete = it++;
              current_set_preconditions->erase(to_delete);
            }
            else
              it++;
          }
          if (current_set_preconditions->empty())
          {
            delete current_set_preconditions;
            current_set_preconditions = NULL;
          } 
        }
        // Special case for where we're just invalidating everything so we
        // can filter things from the previous sets eagerly 
        if (!move_to_previous && (previous_sets != NULL) &&
            !(mask * previous_sets->get_valid_mask()))
        {
          std::vector<EquivalenceSet*> to_delete;
          for (FieldMaskSet<EquivalenceSet>::iterator it =
                previous_sets->begin(); it != previous_sets->end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          for (std::vector<EquivalenceSet*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            previous_sets->erase(*it);
            if ((*it)->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
              delete (*it);
          }
          if (previous_sets->empty())
          {
            delete previous_sets;
            previous_sets = NULL;
          }
          else
            previous_sets->tighten_valid_mask();
        }
        // Check to see which fields we have current equivalence sets
        // for on this node as these are the ones that will ultimately
        // have to be invalidated
        FieldMask current_mask;
        if (current_sets != NULL)
          current_mask = mask & current_sets->get_valid_mask();
        if (!!current_mask)
        {
          // No matter what we're going to do here we need to 
          // invalidate the subscriptions so do that first 
          if (subscriptions != NULL)
          {
            for (LegionMap<AddressSpaceID,
                           FieldMaskSet<EqSetTracker> >::iterator
                  sit = subscriptions->begin(); 
                  sit != subscriptions->end(); /*nothing*/)
            {
              if (sit->second.get_valid_mask() * current_mask)
              {
                sit++;
                continue;
              }
              FieldMaskSet<EqSetTracker> &invalidations = 
                to_invalidate[sit->first];
              if (!!(sit->second.get_valid_mask() - current_mask))
              {
                // Filter out specific subscriptions
                std::vector<EqSetTracker*> to_delete;
                for (FieldMaskSet<EqSetTracker>::iterator it =
                      sit->second.begin(); it != sit->second.end(); it++)
                {
                  const FieldMask overlap = current_mask & it->second;
                  if (!overlap)
                    continue;
                  invalidations.insert(it->first, overlap);
                  it.filter(overlap);
                  if (!it->second)
                    to_delete.push_back(it->first);
                }
                for (std::vector<EqSetTracker*>::const_iterator it =
                      to_delete.begin(); it != to_delete.end(); it++)
                  sit->second.erase(*it);
              }
              else // Invalidating the subscriptions for all fields
                invalidations.swap(sit->second);
              if (sit->second.empty())
              {
                LegionMap<AddressSpaceID,
                  FieldMaskSet<EqSetTracker> >::iterator to_delete = sit++;
                subscriptions->erase(to_delete);
              }
              else
              {
                sit->second.tighten_valid_mask();
                sit++;
              }
            }
            if (subscriptions->empty())
            {
              delete subscriptions;
              subscriptions = NULL;
            }
          }
          if (this->bounds == rect)
          {
            // We know these fields no longer need to be filtered
            // since we're going to filter them here
            remaining -= current_mask;
            // We have permissions to perform all the invalidations
            // here without needing to refine. First filter the 
            // previous sets for these fields since we know we're
            // going to have current sets to flush back to the 
            // previous sets for them
            if (move_to_previous)
            {
              if (previous_sets == NULL)
                previous_sets = new FieldMaskSet<EquivalenceSet>();
              else if (!(current_mask * previous_sets->get_valid_mask()))
              {
                // Very important that we only filter previous sets
                // if we actually have current sets to replace them with
                std::vector<EquivalenceSet*> to_delete;
                for (FieldMaskSet<EquivalenceSet>::iterator it =
                      previous_sets->begin(); it != previous_sets->end(); it++)
                {
                  it.filter(current_mask);
                  if (!it->second)
                    to_delete.push_back(it->first);
                }
                for (std::vector<EquivalenceSet*>::const_iterator it =
                      to_delete.begin(); it != to_delete.end(); it++)
                {
                  previous_sets->erase(*it);
                  if ((*it)->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
                    delete (*it);
                }
                // No need to delete previous sets or tighten its valid
                // mask since we know that there will be current sets
                // getting store into the previous sets for all those fields
              }
            }
            // Now we can invalidate the current sets and flush them
            // back to the previous sets if we're moving them
            std::vector<EquivalenceSet*> to_delete;
            for (FieldMaskSet<EquivalenceSet>::iterator it =
                  current_sets->begin(); it != current_sets->end(); it++)
            {
              const FieldMask overlap = it->second & current_mask;
              if (!overlap)
                continue;
              if (move_to_previous && previous_sets->insert(it->first, overlap))
                it->first->add_base_gc_ref(DISJOINT_COMPLETE_REF);
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              current_mask -= overlap;
              if (!current_mask)
                break;
            }
            for (std::vector<EquivalenceSet*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              current_sets->erase(*it);
              if ((*it)->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
                delete (*it);
            }
            if (current_sets->empty())
            {
              delete current_sets;
              current_sets = NULL;
            }
            else
              current_sets->tighten_valid_mask();
          }
          else
          {
            // It's unsound for us to partially invalidate this node
            // if the rect is only a subset of the bounds because if
            // we don't have permissions to perform the invalidations
            // for those points then we might end up trying to change
            // the equivalence sets while other operations are trying
            // to mutate those equivalence sets leading to races. To
            // avoid this we'll check to see if we have any current
            // equivalence sets
            if (lefts != NULL)
            {
              FieldMask refine = current_mask - lefts->get_valid_mask();
              if (!!refine)
                refine_node(rect, refine, true/*refine current*/);
            }
            else
              refine_node(rect, current_mask, true/*refine current*/);
          }
        }
      }
      if (!to_invalidate.empty())
        EqSetTracker::invalidate_subscriptions(runtime, this,
                                  to_invalidate, invalidated);
      for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
            to_invalidate_previous.begin(); it != 
            to_invalidate_previous.end(); it++)
      {
        it->first->invalidate_all_previous_sets(it->second);
        if (it->first->remove_reference())
          delete it->first;
      }
      // Now see if we need to continue the traversal for any remaining fields
      // Note that we know we don't need the lock here since we know that
      // the shape of the equivalence set KD tree can't be changing since 
      // we hold the tree lock at the root
      if (!!remaining)
      {
        // We can skip performing invalidations if we know that everything
        // below is already previous-only
        if (!!all_previous_below)
          remaining -= all_previous_below;
        // Find the nodes to traverse below
        if (!!remaining && (lefts != NULL) &&
            !(remaining * lefts->get_valid_mask()))
        {
          FieldMask right_mask;
          for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                lefts->begin(); it != lefts->end(); it++)
          {
            const FieldMask overlap = it->second & remaining;
            if (!overlap)
              continue;
            // Compute the overlap
            const Rect<DIM,T> intersection = 
              rect.intersection(it->first->bounds); 
            if (!intersection.empty())
            {
              // Invalidate the child and then record which fields it is
              // all previous below
              FieldMask child_previous;
              it->first->invalidate_tree(intersection, overlap, runtime,
                  invalidated, move_to_previous, &child_previous);
              if (!!child_previous)
                record_child_all_previous(it->first, child_previous);
              if (intersection != rect)
                right_mask |= overlap;
            }
            else
              right_mask |= overlap;
            remaining -= overlap;
            if (!remaining)
              break;
          }
          if (!!right_mask)
          {
            for (typename FieldMaskSet<EqKDNode<DIM,T> >::const_iterator it =
                  rights->begin(); it != rights->end(); it++)
            {
              const FieldMask overlap = it->second & right_mask;
              if (!overlap)
                continue;
              const Rect<DIM,T> intersection =
                rect.intersection(it->first->bounds);
#ifdef DEBUG_LEGION
              assert(!intersection.empty());
#endif
              FieldMask child_previous;
              it->first->invalidate_tree(intersection, overlap, runtime,
                  invalidated, move_to_previous, &child_previous);
              if (!!child_previous)
                record_child_all_previous(it->first, child_previous);
              right_mask -= overlap;
              if (!right_mask)
                break;
            }
#ifdef DEBUG_LEGION
            assert(!right_mask);
#endif
          }
        }
      }
      // Record the any all-previous fields at this child
      if (parent_all_previous != NULL)
      {
        *parent_all_previous = all_previous_below;
        if (previous_sets != NULL)
          *parent_all_previous |= previous_sets->get_valid_mask();
        // Only return fields that were invalidated
        *parent_all_previous &= mask;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::record_child_all_previous(EqKDNode<DIM,T> *child,
                                                    FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (!!all_previous_below)
      {
        // If the fields are already all-previous below than we're done
        mask -= all_previous_below;
        if (!mask)
          return;
      }
      if (child_previous_below != NULL)
      {
        // See if the other child is already all-previous for these fields
        if (!(mask * child_previous_below->get_valid_mask()))
        {
          std::vector<EqKDNode<DIM,T>*> to_delete;
          for (typename FieldMaskSet<EqKDNode<DIM,T> >::iterator it =
                child_previous_below->begin(); it != 
                child_previous_below->end(); it++)
          {
            if (it->first == child)
              continue;
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            // The other child is already all-previous for these fields
            // so we can record them as all previous now
            all_previous_below |= overlap;
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
            mask -= overlap;
            if (!mask)
              break;
          }
          for (typename std::vector<EqKDNode<DIM,T>*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
            child_previous_below->erase(*it);
          if (!mask)
          {
            if (child_previous_below->empty())
            {
              delete child_previous_below;
              child_previous_below = NULL;
            }
            return;
          }
        }
      }
      else
        child_previous_below = new FieldMaskSet<EqKDNode<DIM,T> >();
      child_previous_below->insert(child, mask);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDNode<DIM,T>::invalidate_shard_tree_remote(const Rect<DIM,T> &rect,
          const FieldMask &mask, Runtime *runtime, 
          std::vector<RtEvent> &invalidated,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      invalidate_tree(rect, mask, runtime,invalidated,true/*move to previous*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDNode<DIM,T>::cancel_subscription(EqSetTracker *tracker,
                                    AddressSpaceID space, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock); 
      if (subscriptions == NULL)
        return 0;
      LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> >::iterator
        subscription_finder = subscriptions->find(space);
      if (subscription_finder == subscriptions->end())
        return 0;
      FieldMaskSet<EqSetTracker>::iterator finder =
        subscription_finder->second.find(tracker);
      if (finder == subscription_finder->second.end())
        return 0;
      const FieldMask overlap = mask & finder->second;
      if (!overlap)
        return 0;
      finder.filter(overlap);
      if (!finder->second)
      {
        subscription_finder->second.erase(finder);
        if (subscription_finder->second.empty())
        {
          subscriptions->erase(subscription_finder);
          if (subscriptions->empty())
          {
            delete subscriptions;
            subscriptions = NULL;
          }
        }
        else
          subscription_finder->second.tighten_valid_mask();
      }
      else
        subscription_finder->second.tighten_valid_mask();
      return overlap.pop_count();
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Sparse
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparse<DIM,T>::EqKDSparse(const Rect<DIM,T> &bound,
                                  const std::vector<Rect<DIM,T> > &rects)
      : EqKDTreeT<DIM,T>(bound)
    //--------------------------------------------------------------------------
    {
      if (rects.size() <= LEGION_MAX_BVH_FANOUT)
      {
        // Base case of a small enough number of children
        children.reserve(rects.size());
        for (typename std::vector<Rect<DIM,T> >::const_iterator it =
              rects.begin(); it != rects.end(); it++)
        {
          EqKDNode<DIM,T> *child = new EqKDNode<DIM,T>(*it);
          child->add_reference();
          children.push_back(child);
        }
        return;
      }
      // Unlike some of our other KDNode implementations, we know that all of
      // these rectangles are non-overlapping with each other which means that
      // we should always be able to find good splitting planes
      Rect<DIM,T> best_left_bounds, best_right_bounds;
      std::vector<Rect<DIM,T> > best_left_set, best_right_set;
      bool success = KDTree::compute_best_splitting_plane<DIM,T>(bound, rects,
          best_left_bounds, best_right_bounds, best_left_set, best_right_set);
      // See if we had at least one good refinement
      if (success)
      {
        EqKDSparse<DIM,T> *left = 
          new EqKDSparse<DIM,T>(best_left_bounds, best_left_set); 
        left->add_reference();
        children.push_back(left);
        EqKDSparse<DIM,T> *right =
          new EqKDSparse<DIM,T>(best_right_bounds, best_right_set);
        right->add_reference();
        children.push_back(right);
      }
      else
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_KDTREE_REFINEMENT_FAILED,
            "Failed to find a refinement for Equivalence Set KD tree with %d " 
            "dimensions and %zd rectangles. Please report your application to "
            "the Legion developers' mailing list.", DIM, rects.size())
        // If we make it here then we couldn't find a splitting plane to refine
        // anymore so just record all the subrects as our rects
        children.reserve(rects.size());
        for (typename std::vector<Rect<DIM,T> >::const_iterator it =
              rects.begin(); it != rects.end(); it++)
        {
          EqKDNode<DIM,T> *child = new EqKDNode<DIM,T>(*it);
          child->add_reference();
          children.push_back(child);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparse<DIM,T>::~EqKDSparse(void)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM,T>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
        if ((*it)->remove_reference())
          delete (*it);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM,T>::initialize_set(EquivalenceSet *set,
                   const Rect<DIM,T> &rect, const FieldMask &mask,
                   ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM,T>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const Rect<DIM,T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          (*it)->initialize_set(set, overlap, mask, local_shard, current);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM,T>::compute_equivalence_sets(
        const Rect<DIM,T> &rect, const FieldMask &mask, 
        const std::vector<EqSetTracker*> &trackers,
        const std::vector<AddressSpaceID> &tracker_spaces,
        std::vector<unsigned> &new_tracker_references,
        FieldMaskSet<EquivalenceSet> &eq_sets,
        std::vector<RtEvent> &pending_sets,
        FieldMaskSet<EqKDTree> &subscriptions,
        FieldMaskSet<EqKDTree> &to_create,
        std::map<EqKDTree*,Domain> &creation_rects,
        std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs,
        std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM,T>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const Rect<DIM,T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          (*it)->compute_equivalence_sets(overlap, mask, trackers,
              tracker_spaces, new_tracker_references, eq_sets, pending_sets,
              subscriptions, to_create, creation_rects, creation_srcs,
              remote_shard_rects, local_shard);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM,T>::record_equivalence_set(EquivalenceSet *set,
        const FieldMask &mask, RtEvent ready, 
        const CollectiveMapping &creator_spaces, 
        const std::vector<EqSetTracker*> &creators)
    //--------------------------------------------------------------------------
    {
      // This should never be called on a sparse tree node
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSparse<DIM,T>::record_output_equivalence_set(
        EquivalenceSet *set, const Rect<DIM,T> &rect, const FieldMask &mask, 
        EqSetTracker *tracker, AddressSpaceID tracker_space, 
        FieldMaskSet<EqKDTree> &subscriptions,
        std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      unsigned new_subs = 0;
      for (typename std::vector<EqKDTreeT<DIM,T>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const Rect<DIM,T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          new_subs += (*it)->record_output_equivalence_set(set, overlap, mask,
              tracker, tracker_space, subscriptions, remote_shard_rects, 
              local_shard);
      }
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM,T>::find_local_equivalence_sets(
        FieldMaskSet<EquivalenceSet> &eq_sets, ShardID local_shard) const
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM,T>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
        (*it)->find_local_equivalence_sets(eq_sets, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM,T>::find_shard_equivalence_sets(
                                std::map<ShardID,LegionMap<RegionNode*,
                                     FieldMaskSet<EquivalenceSet> > > &eq_sets,
                            ShardID source_shard, ShardID dst_lower_shard,
                            ShardID dst_upper_shard, RegionNode *region) const
    //--------------------------------------------------------------------------
    {
      // TODO
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM,T>::invalidate_tree(const Rect<DIM,T> &rect,
                                            const FieldMask &mask,
                                            Runtime *runtime,
                                            std::vector<RtEvent> &invalidated,
                                            bool move_to_previous,
                                            FieldMask *parent_all_previous)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<EqKDTreeT<DIM,T>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const Rect<DIM,T> overlap = rect.intersection((*it)->bounds);
        if (!overlap.empty())
          (*it)->invalidate_tree(overlap, mask, runtime, invalidated,
              move_to_previous, parent_all_previous);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparse<DIM,T>::invalidate_shard_tree_remote(
          const Rect<DIM,T> &rect, const FieldMask &mask, Runtime *runtime, 
          std::vector<RtEvent> &invalidated,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      invalidate_tree(rect, mask, runtime,invalidated,true/*move to previous*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSparse<DIM,T>::cancel_subscription(EqSetTracker *tracker,
                                    AddressSpaceID space, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // should never be called on sparse nodes since they don't track
      assert(false);
      return 0;
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Sharded
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSharded<DIM,T>::EqKDSharded(const Rect<DIM,T> &rect, 
                                    ShardID low, ShardID high)
      : EqKDTreeT<DIM,T>(rect), lower(low), upper(high), left(NULL), right(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSharded<DIM,T>::~EqKDSharded(void)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM,T> *next = left.load();
      if ((next != NULL) && next->remove_reference())
        delete next;
      next = right.load();
      if ((next != NULL) && next->remove_reference())
        delete next;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::initialize_set(EquivalenceSet *set,
                               const Rect<DIM,T> &rect, const FieldMask &mask,
                               ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
      assert(lower <= local_shard);
      assert(local_shard <= upper);
#endif
      EqKDTreeT<DIM,T> *next = right.load();
      // Check to see if we've reached the bottom
      if (next == NULL)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM,T> *local = left.load();
            if (local == NULL)
              local = refine_local();
            local->initialize_set(set, rect, mask, local_shard, current);
          }
          return;
        }
        else // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
#ifdef DEBUG_LEGION
      assert(next != NULL);
      assert(lower != upper);
#endif
      // We only need to traverse down the child that has our local shard
      ShardID diff = upper - lower;
      ShardID mid = lower + (diff  / 2);
      if (local_shard <= mid)
        next = left.load();
      const Rect<DIM,T> overlap = next->bounds.intersection(rect);
      if (!overlap.empty())
        next->initialize_set(set, overlap, mask, local_shard, current);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::compute_equivalence_sets(
          const Rect<DIM,T> &rect, const FieldMask &mask,
          const std::vector<EqSetTracker*> &trackers,
          const std::vector<AddressSpaceID> &tracker_spaces,
          std::vector<unsigned> &new_tracker_references,
          FieldMaskSet<EquivalenceSet> &eq_sets,
          std::vector<RtEvent> &pending_sets,
          FieldMaskSet<EqKDTree> &subscriptions,
          FieldMaskSet<EqKDTree> &to_create,
          std::map<EqKDTree*,Domain> &creation_rects,
          std::map<EquivalenceSet*,LegionMap<Domain,FieldMask> > &creation_srcs,
          std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
          ShardID local_shard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
#endif
      EqKDTreeT<DIM,T> *next = right.load();
      // Check to see if we've reached the bottom
      if (next == NULL)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM,T> *local = left.load();
            if (local == NULL)
              local = refine_local();
            local->compute_equivalence_sets(rect, mask, trackers,
                tracker_spaces, new_tracker_references, eq_sets, pending_sets,
                subscriptions, to_create, creation_rects, creation_srcs,
                remote_shard_rects, local_shard);
          }
          else
            remote_shard_rects[lower][rect] |= mask;
          // We're done
          return;
        }
        else // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      const Rect<DIM,T> right_overlap = next->bounds.intersection(rect);
      if (!right_overlap.empty())
        next->compute_equivalence_sets(right_overlap, mask, trackers,
            tracker_spaces, new_tracker_references, eq_sets, pending_sets,
            subscriptions, to_create, creation_rects, creation_srcs,
            remote_shard_rects, local_shard);
      next = left.load();
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      const Rect<DIM,T> left_overlap = next->bounds.intersection(rect);
      if (!left_overlap.empty())
        next->compute_equivalence_sets(left_overlap, mask, trackers,
            tracker_spaces, new_tracker_references, eq_sets, pending_sets,
            subscriptions, to_create, creation_rects, creation_srcs,
            remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t EqKDSharded<DIM,T>::get_total_volume(void) const
    //--------------------------------------------------------------------------
    {
      return this->bounds.volume();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDTreeT<DIM,T>* EqKDSharded<DIM,T>::refine_local(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(right.load() == NULL);
#endif
      EqKDNode<DIM,T> *next = new EqKDNode<DIM,T>(this->bounds);
      EqKDTreeT<DIM,T> *expected = NULL;
      if (left.compare_exchange_strong(expected, next))
      {
        next->add_reference();
        return next;
      }
      else
      {
        delete next;
        return expected;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::refine_node(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lower < upper);
#endif
      // Find the largest dimension and split it in half
      // Note we cannot use the rectangle to guide our splitting plane here
      // like we do with the EqKDNode because this splitting needs to be
      // deterministic across the shards
      T split = 0;
      int dim = -1;
      T largest = 0;
      for (int d = 0; d < DIM; d++)
      {
        T diff = this->bounds.hi[d] - this->bounds.lo[d];
        if (diff <= largest)
          continue;
        largest = diff;
        dim = d;
        split = this->bounds.lo[d] + (diff / 2);
      }
#ifdef DEBUG_LEGION
      assert(dim >= 0);
#endif
      Rect<DIM,T> left_bounds = this->bounds;
      Rect<DIM,T> right_bounds = this->bounds;
      left_bounds.hi[dim] = split;
      right_bounds.lo[dim] = split+1;
      // Find the splitting of the shards
      ShardID diff = upper - lower;
      ShardID mid = lower + (diff  / 2);
      // Do left before right so that as soon as right is set then left is too
      EqKDSharded<DIM,T> *next = 
        new EqKDSharded<DIM,T>(left_bounds, lower, mid);
      EqKDTreeT<DIM,T> *expected = NULL;
      if (left.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
      next = new EqKDSharded<DIM,T>(right_bounds, mid+1, upper);
      expected = NULL;
      if (right.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::record_equivalence_set(EquivalenceSet *set,
        const FieldMask &mask, RtEvent ready,
        const CollectiveMapping &creator_spaces,
        const std::vector<EqSetTracker*> &creators)
    //--------------------------------------------------------------------------
    {
      // This should never be called on a sharded tree node
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSharded<DIM,T>::record_output_equivalence_set(
        EquivalenceSet *set, const Rect<DIM,T> &rect, const FieldMask &mask, 
        EqSetTracker *tracker, AddressSpaceID tracker_space, 
        FieldMaskSet<EqKDTree> &subscriptions,
        std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
#endif
      EqKDTreeT<DIM,T> *next = right.load();
      // Check to see if we've reached the bottom
      if (next == NULL)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM,T> *local = left.load();
            if (local == NULL)
              local = refine_local();
            return local->record_output_equivalence_set(set, rect, mask,
                tracker, tracker_space, subscriptions, remote_shard_rects,
                local_shard);
          }
          else
          {
            remote_shard_rects[lower][rect] |= mask;
            return 0;
          }
        }
        else // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
      unsigned new_subs = 0;
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      const Rect<DIM,T> right_overlap = next->bounds.intersection(rect);
      if (!right_overlap.empty())
        new_subs += next->record_output_equivalence_set(set, right_overlap,
            mask, tracker, tracker_space, subscriptions,
            remote_shard_rects, local_shard);
      next = left.load();
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      const Rect<DIM,T> left_overlap = next->bounds.intersection(rect);
      if (!left_overlap.empty())
        new_subs += next->record_output_equivalence_set(set, left_overlap,
            mask, tracker, tracker_space, subscriptions,
            remote_shard_rects, local_shard);
      return new_subs;  
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::find_local_equivalence_sets(
               FieldMaskSet<EquivalenceSet> &eq_sets, ShardID local_shard) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(lower <= local_shard);
      assert(local_shard <= upper);
#endif
      EqKDTreeT<DIM,T> *next = right.load();
      // Check to see if we've reached the bottom
      if (next == NULL)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM,T> *local = left.load();
            if (local != NULL)
              local->find_local_equivalence_sets(eq_sets, local_shard);
          }
        }
        return;
      }
#ifdef DEBUG_LEGION
      assert(next != NULL);
      assert(lower != upper);
#endif
      // We only need to traverse down the child that has our local shard
      ShardID diff = upper - lower;
      ShardID mid = lower + (diff  / 2);
      if (local_shard <= mid)
        next = left.load();
      next->find_local_equivalence_sets(eq_sets, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::find_shard_equivalence_sets(
                                std::map<ShardID,LegionMap<RegionNode*,
                                     FieldMaskSet<EquivalenceSet> > > &eq_sets,
                            ShardID source_shard, ShardID dst_lower_shard,
                            ShardID dst_upper_shard, RegionNode *region) const
    //--------------------------------------------------------------------------
    {
      // Keep going to the local shard and split the dst shards along the way 
#ifdef DEBUG_LEGION
      assert(lower <= source_shard);
      assert(source_shard <= upper);
#endif
      EqKDTreeT<DIM,T> *next = right.load();
      // Check to see if we've reached the bottom
      if (next == NULL)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == source_shard)
          {
            EqKDTreeT<DIM,T> *local = left.load();
            if (local != NULL)
              local->find_shard_equivalence_sets(eq_sets, source_shard,
                  dst_lower_shard, dst_upper_shard, region);
          }
        }
        return;
      }
#ifdef DEBUG_LEGION
      assert(next != NULL);
      assert(lower != upper);
#endif
      // We only need to traverse down the child that has our local shard
      ShardID src_diff = upper - lower;
      ShardID src_mid = lower + (src_diff  / 2);
      ShardID dst_diff = dst_upper_shard - dst_lower_shard;
      ShardID dst_mid = dst_lower_shard + (dst_diff / 2);
      if (source_shard <= src_mid)
      {
        next = left.load();
        dst_upper_shard = dst_mid;
      }
      else if (dst_lower_shard != dst_upper_shard)
        dst_lower_shard = dst_mid + 1;
      next->find_shard_equivalence_sets(eq_sets, source_shard,
          dst_lower_shard, dst_upper_shard, region);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::invalidate_tree(const Rect<DIM,T> &rect,
                                       const FieldMask &mask, Runtime *runtime,
                                       std::vector<RtEvent> &invalidated_events,
                                       bool move_to_previous,
                                       FieldMask *parent_all_previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
#endif
      // Just traverse everything that is open, no need to do any refinements 
      // since the same invalidation is being done on every shard
      EqKDTreeT<DIM,T> *next = left.load();
      if (next != NULL)
      {
        const Rect<DIM,T> overlap = next->bounds.intersection(rect);
        if (!overlap.empty())
          next->invalidate_tree(overlap, mask, runtime, 
              invalidated_events, move_to_previous, parent_all_previous);
      }
      next = right.load();
      if (next != NULL)
      {
        const Rect<DIM,T> overlap = next->bounds.intersection(rect);
        if (!overlap.empty())
          next->invalidate_tree(overlap, mask, runtime, 
              invalidated_events, move_to_previous, parent_all_previous);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSharded<DIM,T>::invalidate_shard_tree_remote(
                                           const Rect<DIM,T> &rect,
                                           const FieldMask &mask,
                                           Runtime *runtime,
                                           std::vector<RtEvent> &invalidated,
            std::map<ShardID,LegionMap<Domain,FieldMask> > &remote_shard_rects,
                                           ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      // This invalidation is only being done on the local shard so we need
      // to perform any needed refinements so we can send the updates to 
      // other shards to perform if necessary
#ifdef DEBUG_LEGION
      assert(this->bounds.contains(rect));
#endif
      EqKDTreeT<DIM,T> *next = right.load();
      // Check to see if we've reached the bottom
      if (next == NULL)
      {
        // No refinement yet, see if we need to make one
        if ((lower == upper) || (get_total_volume() <= MIN_SPLIT_SIZE))
        {
          // No more refinements, see if the local shard is the lower shard
          // and we can make a local node or node
          if (lower == local_shard)
          {
            EqKDTreeT<DIM,T> *local = left.load();
            // Only need to perform the refinement if it already exists
            if (local != NULL)
              local->invalidate_shard_tree_remote(rect, mask, runtime,
                  invalidated, remote_shard_rects, local_shard);
          }
          else
            remote_shard_rects[lower][rect] |= mask;
          return;
        }
        else // Create the refinement
        {
          refine_node();
          next = right.load();
        }
      }
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      const Rect<DIM,T> right_overlap = next->bounds.intersection(rect);
      if (!right_overlap.empty())
        next->invalidate_shard_tree_remote(right_overlap, mask, runtime,
            invalidated, remote_shard_rects, local_shard);
      next = left.load();
#ifdef DEBUG_LEGION
      assert(next != NULL);
#endif
      const Rect<DIM,T> left_overlap = next->bounds.intersection(rect);
      if (!left_overlap.empty())
        next->invalidate_shard_tree_remote(left_overlap, mask, runtime,
            invalidated, remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned EqKDSharded<DIM,T>::cancel_subscription(EqSetTracker *tracker,
                                    AddressSpaceID space, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // should never be called on sharded nodes since they don't track
      assert(false);
      return 0;
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set KD Sparse Sharded
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    /*static*/ inline bool EqKDSparseSharded<DIM,T>::sort_by_volume(
                                   const Rect<DIM,T> &r1, const Rect<DIM,T> &r2)
    //--------------------------------------------------------------------------
    {
      return (r1.volume() < r2.volume());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparseSharded<DIM,T>::EqKDSparseSharded(const Rect<DIM,T> &bound,
                    ShardID low, ShardID high, std::vector<Rect<DIM,T> > &rects)
      : EqKDSharded<DIM,T>(bound, low, high)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rects.size() > 1);
#endif
      rectangles.swap(rects);
      total_volume = 0;
      for (typename std::vector<Rect<DIM,T> >::const_iterator it =
            rectangles.begin(); it != rectangles.end(); it++)
        total_volume += it->volume();
      // If there's a chance that we might need to refine these then 
      // stable sort them so that refine_node can rely on them already
      // being sorted. Note the stable sort! Must maintain deterministic
      // order across the shards
      if (this->MIN_SPLIT_SIZE <= total_volume)
        std::stable_sort(rectangles.begin(), rectangles.end(), sort_by_volume);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDSparseSharded<DIM,T>::~EqKDSparseSharded(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t EqKDSparseSharded<DIM,T>::get_total_volume(void) const
    //--------------------------------------------------------------------------
    {
      return total_volume;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDTreeT<DIM,T>* EqKDSparseSharded<DIM,T>::refine_local(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->right.load() == NULL);
#endif
      EqKDSparse<DIM,T> *next = new EqKDSparse<DIM,T>(this->bounds, rectangles);
      EqKDTreeT<DIM,T> *expected = NULL;
      if (this->left.compare_exchange_strong(expected, next))
      {
        next->add_reference();
        return next;
      }
      else
      {
        delete next;
        return expected;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void EqKDSparseSharded<DIM,T>::refine_node(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this->lower < this->upper);
#endif
      // Note here that we don't want to evenly divide the rectangles across 
      // the shards, but instead we want to evenly split the points across the
      // shards. We have two ways to do this, the first way is to call 
      // compute_best_splitting_plane which will try to find a good splitting
      // plane that maintains the integrity of the spatial locality of all the
      // rectangles. Note that when we call compute_best_splitting_plane we 
      // ask it to sort based on the number of points and not by the number of
      // rectangles which should keep the total points balanced across shards
      Rect<DIM,T> left_bounds, right_bounds;
      std::vector<Rect<DIM,T> > left_set, right_set;
      if (!KDTree::compute_best_splitting_plane<DIM,T,false>(this->bounds,
            rectangles, left_bounds, right_bounds, left_set, right_set))
      {
        // If we get here, then compute_best_splitting_plane failed to find
        // a splitting plane to split the rectangles nicely, so now we're
        // going to fall back to a dumb and greedy heuristic which will still
        // give us a good distribution of points around the shards which is
        // just to go from the largest rectangles to the smallest and assign
        // them to either the right or the left set depending on which one
        // is larger to get a roughly evenly distributed set of points
        // Note that this is determinisitc because the stable sort done in
        // the constructor of this class maintains the ordering of rectangles
        // with equivalent volumes across the shards.
        uint64_t left_volume = 0, right_volume = 0;
        // Reverse iterator to go from largest to smallest
        for (typename std::vector<Rect<DIM,T> >::const_reverse_iterator it =
              rectangles.crbegin(); it != rectangles.crend(); it++)
        {
          if (left_volume <= right_volume)
          {
            left_set.push_back(*it);
            left_volume += it->volume();
            left_bounds = left_bounds.union_bbox(*it);
          }
          else
          {
            right_set.push_back(*it);
            right_volume += it->volume();
            right_bounds = right_bounds.union_bbox(*it);
          }
        }
      }
#ifdef DEBUG_LEGION
      assert(!left_set.empty());
      assert(!right_set.empty());
#endif
      // Find the splitting of the shards
      ShardID diff = this->upper - this->lower;
      ShardID mid = this->lower + (diff  / 2);
      EqKDSharded<DIM,T> *next = NULL;
      if (left_set.size() > 1)
        next = new EqKDSparseSharded(left_bounds, this->lower, mid, left_set);
      else
        next = new EqKDSharded<DIM,T>(left_set.back(), this->lower, mid);
      EqKDTreeT<DIM,T> *expected = NULL;
      if (this->left.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
      if (right_set.size() > 1)
        next =
          new EqKDSparseSharded(right_bounds, mid+1, this->upper, right_set);
      else
        next = new EqKDSharded<DIM,T>(right_set.back(), mid+1, this->upper);
      expected = NULL;
      if (this->right.compare_exchange_strong(expected, next))
        next->add_reference();
      else
        delete next;
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
      if (runtime->profiler != NULL)
        runtime->profiler->add_copy_request(requests, this, op, total_copies);
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
          last_copy = ApEvent(copy_domain.copy(src_fields, dst_fields, 
                indirections, requests, copy_pre, priority));
        else
          last_copy = issue_individual_copies(copy_pre, requests);
          
      }
      else
        last_copy = ApEvent(copy_domain.copy(src_fields, dst_fields,
              requests, copy_pre, priority));
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
                                        RtEvent init, CollectiveMapping *map,
                                        Provenance *prov)
      : IndexPartNode(ctx, p, par, cs, c, disjoint, complete, did,
                      init, map, prov), kd_root(NULL),
        kd_remote(NULL), dense_shard_rects(NULL), sparse_shard_rects(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM,T>::IndexPartNodeT(RegionTreeForest *ctx, 
                                        IndexPartition p, IndexSpaceNode *par,
                                        IndexSpaceNode *cs, LegionColor c, 
                                        int comp, DistributedID did,
                                        RtEvent init, CollectiveMapping *map,
                                        Provenance *prov)
      : IndexPartNode(ctx, p, par, cs, c, comp, did,
                      init, map, prov), kd_root(NULL),
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
            child->get_expr_index_space(&space, type_tag, true/*tight*/);
            if (space.empty())
              continue;
            for (RectInDomainIterator<DIM,T> it(space); it(); it++)
              bounds.push_back(std::make_pair(*it, *itr));
          }
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
            if (rects_ready.exists() && !rects_ready.has_triggered())
              rects_ready.wait();
            // Once we get the remote rectangles we can build the kd-trees
            if (!sparse_shard_rects->empty())
            {
              // Find the nearest address spaces for each color
              std::vector<std::pair<Rect<DIM,T>,AddressSpaceID> >
                sparse_shard_spaces;
              sparse_shard_spaces.reserve(sparse_shard_rects->size());
              LegionColor previous_color = INVALID_COLOR;
              for (typename std::vector<std::pair<Rect<DIM,T>,LegionColor> >::
                    const_iterator it = sparse_shard_rects->begin(); 
                    it != sparse_shard_rects->end(); it++)
              {
                if (it->second != previous_color)
                {
                  CollectiveMapping *child_mapping = NULL;
                  sparse_shard_spaces.emplace_back(std::make_pair(it->first,
                    this->find_color_creator_space(it->second, child_mapping)));
                  if (child_mapping != NULL)
                    delete child_mapping;
                  previous_color = it->second;
                }
                else // colors are the same so we know address space
                  sparse_shard_spaces.emplace_back(std::make_pair(it->first,
                        sparse_shard_spaces.back().second));
              }
              kd_remote = new KDNode<DIM,T,AddressSpaceID>(
                  parent_space.bounds, sparse_shard_spaces);
            }
            // Add any local sparse spaces into the dense remote rects
            // All the local dense spaces are already included
            for (ColorSpaceIterator itr(this, true/*local only*/); itr; itr++)
            {
              IndexSpaceNode *child = get_child(*itr);
              DomainT<DIM,T> space;
              child->get_expr_index_space(&space, type_tag, true/*tight*/);
              if (space.empty() || space.dense())
                continue;
              for (RectInDomainIterator<DIM,T> it(space); it(); it++)
                dense_shard_rects->push_back(std::make_pair(*it, *itr));
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
      expr->get_expr_index_space(&space, handle.get_type_tag(), true/*tight*/);
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
          RemoteKDTracker tracker(context->runtime);
          RtEvent remote_ready =
            tracker.find_remote_interfering(remote_spaces, handle, expr); 
          for (RectInDomainIterator<DIM,T> itr(space); itr(); itr++)
            kd_root->find_interfering(*itr, color_set);
          if (remote_ready.exists() && !remote_ready.has_triggered())
            remote_ready.wait();
          tracker.get_remote_interfering(color_set);
        }
        else
        {
          for (RectInDomainIterator<DIM,T> itr(space); itr(); itr++)
            kd_root->find_interfering(*itr, color_set);
        }
      }
      else
      {
        for (RectInDomainIterator<DIM,T> itr(space); itr(); itr++)
          kd_root->find_interfering(*itr, color_set);
      }
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
        new std::vector<std::pair<Rect<DIM,T>,LegionColor> >();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM,T>::find_local_shard_rects(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> ready_events;
      std::vector<IndexSpaceNodeT<DIM,T>*> children;
      for (ColorSpaceIterator itr(this, true/*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM,T> *child =
          static_cast<IndexSpaceNodeT<DIM,T>*>(get_child(*itr));
        // We're going to exchange these data structures between all the
        // copies of this partition so we only need to record our children
        // if they're actually the owner child
        if (!child->is_owner())
          continue;
        children.push_back(child);
        RtEvent ready = child->get_realm_index_space_ready(true/*tight*/);
        if (ready.exists())
          ready_events.push_back(ready);
      }
      if (!ready_events.empty())
      {
        const RtEvent ready = Runtime::merge_events(ready_events);
        if (ready.exists() && !ready.has_triggered())
        {
          // Defer this until they're all ready
          DeferFindShardRects args(this);
          context->runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, ready);
          return false;
        }
      }
      // All the children are ready so we can get their spaces safely
      AutoLock n_lock(node_lock);
#ifdef DEBUG_LEGION
      assert(dense_shard_rects != NULL);
      assert(sparse_shard_rects != NULL);
#endif
      unsigned logn_children = 0;
      for (typename std::vector<IndexSpaceNodeT<DIM,T>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        DomainT<DIM,T> child_space;
        (*it)->get_realm_index_space(child_space, true/*tight*/);
        const std::pair<Rect<DIM,T>,LegionColor> next(
                      child_space.bounds, (*it)->color);
        if (!child_space.dense())
        {
          // This is a sparse shard space and it's upper bound rectangle
          // can be arbitrarily big, so we want to give it some flexibility to
          // describe itself so others can prune out queries better. Therefore
          // we ask Realm to try to compute an apprxomate covering of the
          // index space. The number of rectangles we'll let it have will be
          // O(log N) in the number of subspaces in the partition, so the
          // more subspaces there are, there more rectangles we'll allow in
          // the approximation. This will be O(N log N) total cost across
          // all the sparse subspaces. We might consider making the number
          // of rectangles O(N^(1/2)) in the future but I'm scared of the 
          // scalability implications of that.
          if (logn_children == 0)
          {
            // Compute ceil(log(N)) of the total children to know how many
            // rectangles we can ask for in the approximation
            LegionColor power = 1;
            while (power < total_children)
            {
              logn_children++;
              power *= 2;
            }
          }
          std::vector<Rect<DIM,T> > covering;
          // Note we don't care about the overhead, it can't be worse
          // than the upper bound rectangle that we already have
          if ((logn_children > 1) && child_space.compute_covering(
                logn_children, INT_MAX/*overhead*/, covering))
          {
            for (typename std::vector<Rect<DIM,T> >::const_iterator cit =
                  covering.begin(); cit != covering.end(); cit++)
              sparse_shard_rects->push_back(std::make_pair(*cit, (*it)->color));
          }
          else // Can just add this as the covering failed
            sparse_shard_rects->push_back(next);
        }
        else if (!child_space.bounds.empty())
          dense_shard_rects->push_back(next);
      }
      return perform_shard_rects_notification();
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
      for (typename std::vector<std::pair<Rect<DIM,T>,LegionColor> >::
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
          std::pair<Rect<DIM,T>,LegionColor> &next = 
            (*sparse_shard_rects)[offset + idx];
          derez.deserialize(next.first);
          derez.deserialize(next.second);
        }
      }
    }
#endif // defined(DEFINE_NT_TEMPLATES)

  }; // namespace Internal
}; // namespace Legion

