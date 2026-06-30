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

// Included from expression.h - do not include this directly

// Useful for IDEs
#include "legion/nodes/expression.h"

namespace Legion {
  namespace Internal {

#ifdef DEFINE_NT_TEMPLATES
    /////////////////////////////////////////////////////////////
    // Index Space Expression
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    /*static*/ IndexSpaceExpression*
        IndexSpaceExpression::find_or_create_empty_expression(void)
    //--------------------------------------------------------------------------
    {
      static std::atomic<InternalExpression<DIM, T>*> empty_expr(nullptr);
      InternalExpression<DIM, T>* result = empty_expr.load();
      if (result != nullptr)
        return result;
      const Rect<DIM, T> empty = Rect<DIM, T>::make_empty();
      result = new InternalExpression<DIM, T>(&empty, 1);
      result->add_base_expression_reference(RUNTIME_REF);
      // See if we can swap it in
      InternalExpression<DIM, T>* actual = nullptr;
      if (empty_expr.compare_exchange_strong(actual, result))
      {
        runtime->record_empty_expression(result);
        return result;
      }
      else
      {
        legion_assert(actual != nullptr);
        if (result->remove_base_expression_reference(RUNTIME_REF))
          delete result;
        return actual;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceExpression::inline_union_internal(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      // Disable the fast path for Legion Spy to avoid creating too many
      // index space expressions for it to deal with
      if (spy_logging_level > NO_SPY_LOGGING)
        return nullptr;
      DomainT<DIM, T> domain = get_tight_domain();
      if (!domain.dense())
        return nullptr;
      Rect<DIM, T> bounds = domain.bounds;
      domain = rhs->get_tight_domain();
      if (!domain.dense())
        return nullptr;
      if (bounds.contains(domain.bounds))
        return this;
      if (domain.bounds.contains(bounds))
        return rhs;
      // The union bbox is the union if the volume of the union bbox is the
      // same as the volume of each rect added together and subtracted by
      // their overlap
      const Rect<DIM, T> result = bounds.union_bbox(domain.bounds);
      if (result.volume() == (bounds.volume() + domain.bounds.volume() -
                              bounds.intersection(domain.bounds).volume()))
        return new IndexSpaceUnion<DIM, T>(result);
      else
        return nullptr;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceExpression::inline_union_internal(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      // Disable the fast path for Legion Spy to avoid creating too many
      // index space expressions for it to deal with
      if (spy_logging_level > NO_SPY_LOGGING)
        return nullptr;
      if (exprs.size() == 2)
        return this->inline_union_internal<DIM, T>(*std::next(exprs.begin()));
      DomainT<DIM, T> domain = get_tight_domain();
      ;
      if (!domain.dense())
        return nullptr;
      Rect<DIM, T> result = domain.bounds;
      std::vector<Rect<DIM, T> > previous;
      previous.reserve(exprs.size());
      size_t total_volume = 0;
      // Try to use the union bbox, we'll only be able to use this if all
      // the rects are disjoint or contained by each other and then we can
      // sum up the points that we should expect to find
      for (SetView<IndexSpaceExpression*>::const_iterator eit = exprs.begin();
           eit != exprs.end(); eit++)
      {
        domain = (*eit)->get_tight_domain();
        if (!domain.dense())
          return nullptr;
        bool dominated = false;
        for (typename std::vector<Rect<DIM, T> >::iterator it =
                 previous.begin();
             it != previous.end();
             /*nothing*/)
        {
          if (!domain.bounds.overlaps(*it))
          {
            it++;
            continue;
          }
          if (it->contains(domain.bounds))
          {
            dominated = true;
            break;
          }
          else if (domain.bounds.contains(*it))
          {
            // subtract out the old volume since it no longer will count
            total_volume -= it->volume();
            // Remove the entry from the list since it's no longer needed
            it = previous.erase(it);
            // still need to keep going to check for overlaps
            continue;
          }
          // If we get here it overlaps without dominating which is bad
          // and we can't handle that case with union bbox at the moment
          return nullptr;
        }
        if (!dominated)
        {
          result = result.union_bbox(domain.bounds);
          previous.emplace_back(domain.bounds);
          total_volume += domain.bounds.volume();
        }
      }
      if (result.volume() == total_volume)
        return new IndexSpaceUnion<DIM, T>(result);
      else
        return nullptr;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceExpression::inline_intersection_internal(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      // Disable the fast path for Legion Spy to avoid creating too many
      // index space expressions for it to deal with
      if (spy_logging_level > NO_SPY_LOGGING)
        return nullptr;
      DomainT<DIM, T> left = get_tight_domain();
      DomainT<DIM, T> right = rhs->get_tight_domain();
      if (!left.bounds.overlaps(right.bounds))
        return find_or_create_empty_expression<DIM, T>();
      // A note on sparsity maps here: technically if we had just one sparsity
      // map then we could just tighten the bound on that sparsity map and
      // that would be good enough, but then we would create the illusion
      // of having created a new index space and the index space operation
      // would think it owned the sparsity map and would try to destroy it
      // when being cleaned up which would be wrong. We'll leave it up to
      // Realm to do that by falling back to the normal intersection path
      const Rect<DIM, T> intersection = left.bounds.intersection(right.bounds);
      if (!left.dense())
      {
        if (!right.dense())
        {
          // Only the same if they have the same sparsity map and bounds
          if (left.sparsity == right.sparsity)
          {
            if (left.bounds == intersection)
              return this;
            else if (right.bounds == intersection)
              return rhs;
            else
              return nullptr;
          }
          else
            return nullptr;
        }
        else
        {
          // See if tightening with right bounds changes bounding box
          if (left.bounds == intersection)
            return this;
          else
            return nullptr;
        }
      }
      else if (!right.dense())
      {
        // See if tightening with the left bounds changes bounding box
        if (right.bounds == intersection)
          return rhs;
        else
          return nullptr;
      }
      if (intersection == left.bounds)
        return this;
      if (intersection == right.bounds)
        return rhs;
      // Make a new Intersection space that is a value of this space
      return new IndexSpaceIntersection<DIM, T>(intersection);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceExpression::inline_intersection_internal(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      // Disable the fast path for Legion Spy to avoid creating too many
      // index space expressions for it to deal with
      if (spy_logging_level > NO_SPY_LOGGING)
        return nullptr;
      DomainT<DIM, T> domain = get_tight_domain();
      if (domain.empty())
        return this;
      Rect<DIM, T> result = domain.bounds;
      bool has_sparsity = !domain.dense();
      IndexSpaceExpression* smallest = nullptr;
      for (SetView<IndexSpaceExpression*>::const_iterator it = exprs.begin();
           it != exprs.end(); it++)
      {
        domain = (*it)->get_tight_domain();
        // Keep going anyway to see if the intersection proves empty
        if (!domain.dense())
          has_sparsity = true;
        const Rect<DIM, T> next = result.intersection(domain.bounds);
        if (next != result)
        {
          result = next;
          // If it becomes empty than we are done
          if (next.empty())
          {
            if (domain.empty())
              return *it;
            else
              return find_or_create_empty_expression<DIM, T>();
          }
          if (next == domain.bounds)
            smallest = (*it);
          else
            smallest = nullptr;
        }
      }
      if (has_sparsity)
        return nullptr;
      if (smallest != nullptr)
        return smallest;
      return new IndexSpaceIntersection<DIM, T>(result);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceExpression::inline_subtraction_internal(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      // Disable the fast path for Legion Spy to avoid creating too many
      // index space expressions for it to deal with
      if (spy_logging_level > NO_SPY_LOGGING)
        return nullptr;
      DomainT<DIM, T> left = get_tight_domain();
      DomainT<DIM, T> right = rhs->get_tight_domain();
      // If they don't overlap then this is easy
      if (!left.bounds.overlaps(right.bounds))
        return this;
      // If right is not dense we toss up our hands
      if (!right.dense())
        return nullptr;
      if (!left.dense())
      {
        // Can still do a test for containment here to see if we're empty
        if (right.bounds.contains(left.bounds))
          return find_or_create_empty_expression<DIM, T>();
        else
          return nullptr;
      }
      // We can find up to one non-dominating dimension and still easily
      // compute the difference, as soon as we have more than one then
      // this gets hard and we'll need to use Realm to compute all the
      // different rectangles and put them in a sparsity map
      // Note that even the non-dominating dimension still has to dominate
      // on one side for this to work at all
      int non_dominating_dim = -1;
      for (int i = 0; i < DIM; i++)
      {
        if (right.bounds.lo[i] <= left.bounds.lo[i])
        {
          // Check if dominated on both sides
          if (left.bounds.hi[i] <= right.bounds.hi[i])
            continue;
          // Just dominated on the low-side
          if (non_dominating_dim > -1)
            return nullptr;
          left.bounds.lo[i] = right.bounds.hi[i] + 1;
        }
        else if (left.bounds.hi[i] <= right.bounds.hi[i])
        {
          // Just dominated on the high side
          if (non_dominating_dim > -1)
            return nullptr;
          left.bounds.hi[i] = right.bounds.lo[i] - 1;
        }
        else  // Not dominated on either side
          return nullptr;
        non_dominating_dim = i;
      }
      // If all the dimensions were dominated then the result is empty
      if (non_dominating_dim == -1)
        return find_or_create_empty_expression<DIM, T>();
      else
        return new IndexSpaceDifference<DIM, T>(left.bounds);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    uint64_t IndexSpaceExpression::get_canonical_hash_internal(
        const DomainT<DIM, T>& realm_index_space) const
    //--------------------------------------------------------------------------
    {
      // Should never get this call for anything empty
      legion_assert(!realm_index_space.empty());
      // The promise of the canonical hash is that things that have the same
      // set of points should have the same hash value. We therefore hash first
      // on the type tag since things from different type tags don't need to
      // have the same hash value. Then we hash on the bounds of the tight
      // realm index space since things with the same sets of points will have
      // to have the same tight bounding box. Note that you cannot hash on the
      // sparsity map ID because two realm index spaces can have different
      // sparsity maps but contain the same set of points.
      Murmur3Hasher hasher;
      hasher.hash(type_tag);
      for (int d = 0; d < DIM; d++)
      {
        hasher.hash(realm_index_space.bounds.lo[d]);
        hasher.hash(realm_index_space.bounds.hi[d]);
      }
      // If there is a sparsity map then hash the volume to differentiate
      // things with sparsity maps from ones without sparsity maps. We know that
      // even if two index spaces have different sparsity maps then in order for
      // them to have the same points they should have the same volume.
      if (!realm_index_space.dense())
        hasher.hash(realm_index_space.volume());
      uint64_t hash[2];
      hasher.finalize(hash);
      return hash[0] ^ hash[1];
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceExpression::log_profiler_points(
        DistributedID did, const DomainT<DIM, T>& tight_space) const
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
        implicit_profiler->register_index_space_size(
            did, dense_volume, sparse_volume, !is_dense);
        // Iterate over the rectangles and print them out
        for (Realm::IndexSpaceIterator<DIM, T> itr(tight_space); itr.valid;
             itr.step())
        {
          if (itr.rect.volume() == 1)
            implicit_profiler->record_index_space_point(
                did, Point<DIM, T>(itr.rect.lo));
          else
            implicit_profiler->record_index_space_rect(
                did, Rect<DIM, T>(itr.rect));
        }
      }
      else
        implicit_profiler->register_empty_index_space(did);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceExpression::issue_fill_internal(
        Operation* op, const Realm::IndexSpace<DIM, T>& space,
        const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields, const void* fill_value,
        size_t fill_size, UniqueID fill_uid, FieldSpace handle,
        RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent unique_event, CollectiveKind collective, bool record_effect,
        int priority, bool replay)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
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
        legion_assert(is_indirect);
      }
#endif
      // Now that we know we're going to do this fill add any profiling requests
      Realm::ProfilingRequestSet requests;
      if (!replay)
        priority =
            op->add_copy_profiling_request(trace_info, requests, true /*fill*/);
      ApEvent fill_pre;
      if (pred_guard.exists())
        // No need for tracing to know about the precondition
        fill_pre =
            Runtime::merge_events(nullptr, precondition, ApEvent(pred_guard));
      else
        fill_pre = precondition;
      if (runtime->profiler != nullptr)
      {
        SmallNameClosure<1>* closure = new SmallNameClosure<1>(this);
        closure->record_instance_name(
            dst_fields.front().inst, unique_event, get_distributed_id());
        runtime->profiler->add_fill_request(
            requests, closure, op, fill_pre, collective);
      }
      ApEvent result = ApEvent(space.fill(
          dst_fields, requests, fill_value, fill_size, fill_pre, priority));
      if (pred_guard.exists())
      {
        result = Runtime::ignorefaults(result);
        // Need to merge back in the precondition so it is still reflected
        // in the completion event for this operation
        if (precondition.exists())
        {
          if (result.exists())
            result = Runtime::merge_events(nullptr, result, precondition);
          else
            result = precondition;
          // Little catch here for tracing, make sure the result is unique
          if (trace_info.recording && result.exists() &&
              (result == precondition))
            Runtime::rename_event(result);
        }
      }
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        if (!result.exists())
          Runtime::rename_event(result);
        LegionSpy::log_fill_events(
            op->get_unique_op_id(), expr_id, handle, tree_id, precondition,
            result, fill_uid, collective);
        for (unsigned idx = 0; idx < dst_fields.size(); idx++)
          LegionSpy::log_fill_field(
              result, dst_fields[idx].field_id, unique_event);
      }
      record_index_space_user(result);
      if (record_effect && result.exists())
        op->record_completion_effect(result);
      if (trace_info.recording)
        trace_info.record_issue_fill(
            result, this, dst_fields, fill_value, fill_size, fill_uid, handle,
            tree_id, precondition, pred_guard, unique_event, priority,
            collective, record_effect);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceExpression::issue_copy_internal(
        Operation* op, const Realm::IndexSpace<DIM, T>& space,
        const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<CopySrcDstField>& src_fields,
        const std::vector<Reservation>& reservations, RegionTreeID src_tree_id,
        RegionTreeID dst_tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent src_unique, LgEvent dst_unique, CollectiveKind collective,
        bool record_effect, int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(!space.empty());
#ifdef LEGION_DEBUG
      // If we're doing any reductions with this copy then make sure they
      // are marked exclusive or we have some reservations
      for (const CopySrcDstField& it : dst_fields)
        legion_assert((it.redop_id == 0) || !reservations.empty());
#endif
      // Now that we know we're going to do this copy add any profling requests
      Realm::ProfilingRequestSet requests;
      if (!replay)
        priority = op->add_copy_profiling_request(
            trace_info, requests, false /*fill*/);
      ApEvent copy_pre;
      if (pred_guard.exists())
        copy_pre =
            Runtime::merge_events(nullptr, precondition, ApEvent(pred_guard));
      else
        copy_pre = precondition;
      for (const Reservation& reservation : reservations)
        copy_pre = Runtime::acquire_ap_reservation(
            reservation, true /*exclusive*/, copy_pre);
      if (runtime->profiler != nullptr)
      {
        SmallNameClosure<2>* closure =
            new SmallNameClosure<2>(this, dst_fields.back().redop_id);
        const DistributedID did = get_distributed_id();
        closure->record_instance_name(src_fields.front().inst, src_unique, did);
        closure->record_instance_name(dst_fields.front().inst, dst_unique, did);
        runtime->profiler->add_copy_request(
            requests, closure, op, copy_pre, 1 /*count*/, collective);
      }
      ApEvent result = ApEvent(
          space.copy(src_fields, dst_fields, requests, copy_pre, priority));
      // Release any reservations after the copy is done
      for (const Reservation& reservation : reservations)
        Runtime::release_reservation(reservation, result);
      if (pred_guard.exists())
      {
        result = Runtime::ignorefaults(result);
        // Make sure to fold in the precondition back into the result
        // event in case this is poisoned to support transitive analysis
        if (precondition.exists())
        {
          if (result.exists())
            result = Runtime::merge_events(nullptr, result, precondition);
          else
            result = precondition;
          // Little catch here for tracing, make sure the result is unique
          if (trace_info.recording && result.exists() &&
              (result == precondition))
            Runtime::rename_event(result);
        }
      }
      if (record_effect && result.exists())
        op->record_completion_effect(result);
      if (trace_info.recording)
        trace_info.record_issue_copy(
            result, this, src_fields, dst_fields, reservations, src_tree_id,
            dst_tree_id, precondition, pred_guard, src_unique, dst_unique,
            priority, collective, record_effect);
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        if (!result.exists())
          Runtime::rename_event(result);
        LegionSpy::log_copy_events(
            op->get_unique_op_id(), expr_id, src_tree_id, dst_tree_id,
            precondition, result, collective);
        for (unsigned idx = 0; idx < src_fields.size(); idx++)
          LegionSpy::log_copy_field(
              result, src_fields[idx].field_id, src_unique,
              dst_fields[idx].field_id, dst_unique, dst_fields[idx].redop_id);
      }
      record_index_space_user(result);
      return result;
    }

    template<typename T, typename T2>
    inline T round_up(T val, T2 step)
    {
      T rem = val % step;
      if (rem == 0)
        return val;
      else
        return val + (step - rem);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceExpression::create_layout_internal(
        const Realm::IndexSpace<DIM, T>& space,
        const LayoutConstraintSet& constraints,
        const std::vector<FieldID>& field_ids,
        const std::vector<size_t>& field_sizes, bool compact, void** piece_list,
        size_t* piece_list_size, size_t* num_pieces,
        size_t base_alignment) const
    //--------------------------------------------------------------------------
    {
      legion_assert(base_alignment > 0);  // should be at least 1
      legion_assert(field_ids.size() == field_sizes.size());
      legion_assert(
          int(constraints.ordering_constraint.ordering.size()) == (DIM + 1));
      Realm::InstanceLayout<DIM, T>* layout =
          new Realm::InstanceLayout<DIM, T>();
      layout->bytes_used = 0;
      layout->alignment_reqd = base_alignment;
      layout->space = space;
      std::vector<Rect<DIM, T> > piece_bounds;
      const SpecializedConstraint& spec = constraints.specialized_constraint;
      if (space.dense() || !compact)
      {
        if (!space.bounds.empty())
        {
          // Check to see if we have any tiling constraints
          if (!constraints.tiling_constraints.empty())
          {
            legion_assert(piece_list != nullptr);
            legion_assert((*piece_list) == nullptr);
            legion_assert(piece_list_size != nullptr);
            legion_assert((*piece_list_size) == 0);
            legion_assert(num_pieces != nullptr);
            legion_assert((*num_pieces) == 0);
            // First get the tile bounds
            Point<DIM, T> tile_size;
            for (int i = 0; i < DIM; i++)
              tile_size[i] = (space.bounds.hi[i] - space.bounds.lo[i]) + 1;
            for (const TilingConstraint& constraint :
                 constraints.tiling_constraints)
            {
              legion_assert(constraint.dim < DIM);
              if (constraint.tiles)
                tile_size[constraint.dim] =
                    (tile_size[constraint.dim] + constraint.value - 1) /
                    constraint.value;
              else
                tile_size[constraint.dim] = constraint.value;
            }
            // Now we've got the tile size, walk over the dimensions
            // in order to produce the tiles as pieces
            Point<DIM, T> offset = space.bounds.lo;
            // Iterate until we've tiled the entire space
            bool done = false;
            while (!done)
            {
              // Check to make sure the next tile is in bounds
              Rect<DIM, T> piece(
                  offset, offset + tile_size - Point<DIM, T>::ONES());
              // Intersect with the original bounds to not overflow
              piece = space.bounds.intersection(piece);
              legion_assert(!piece.empty());
              piece_bounds.emplace_back(piece);
              // Step the offset to the next location
              done = true;
              for (const TilingConstraint& constraint :
                   constraints.tiling_constraints)
              {
                offset[constraint.dim] += tile_size[constraint.dim];
                if (offset[constraint.dim] <= space.bounds.hi[constraint.dim])
                {
                  // Still in bounds so we can keep traversing
                  done = false;
                  break;
                }
                else  // No longer in bounds, so ripple carry add
                  offset[constraint.dim] = space.bounds.lo[constraint.dim];
              }
            }
          }
          else
            piece_bounds.emplace_back(space.bounds);
        }
      }
      else
      {
        legion_assert(piece_list != nullptr);
        legion_assert((*piece_list) == nullptr);
        legion_assert(piece_list_size != nullptr);
        legion_assert((*piece_list_size) == 0);
        legion_assert(num_pieces != nullptr);
        legion_assert((*num_pieces) == 0);
        if (spec.max_overhead > 0)
        {
          std::vector<Realm::Rect<DIM, T> > covering;
          if (space.compute_covering(
                  spec.max_pieces, spec.max_overhead, covering))
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
            for (Realm::IndexSpaceIterator<DIM, T> itr(space); itr.valid;
                 itr.step())
              if (!itr.rect.empty())
                piece_bounds.emplace_back(itr.rect);
          }
        }
        else
        {
          for (Realm::IndexSpaceIterator<DIM, T> itr(space); itr.valid;
               itr.step())
            if (!itr.rect.empty())
              piece_bounds.emplace_back(itr.rect);
        }
      }

      // If the bounds are empty we can use the same piece list for all fields
      if (piece_bounds.empty())
      {
        layout->piece_lists.resize(1);
        for (unsigned idx = 0; idx < field_ids.size(); idx++)
        {
          const FieldID fid = field_ids[idx];
          Realm::InstanceLayoutGeneric::FieldLayout& fl = layout->fields[fid];
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
          for (int dim = 0;
               dim < constraints.padding_constraint.delta.get_dim(); dim++)
          {
            legion_assert(constraints.padding_constraint.delta.lo()[dim] >= 0);
            legion_assert(constraints.padding_constraint.delta.hi()[dim] >= 0);
            if ((constraints.padding_constraint.delta.lo()[dim] > 0) ||
                (constraints.padding_constraint.delta.hi()[dim] > 0))
            {
              Fatal fatal;
              fatal << "Legion does not currently support additional padding "
                    << "on compact sparse instances. Please open a github "
                    << "issue to request support.";
              fatal.raise();
            }
          }
        }
        *num_pieces = piece_bounds.size();
        *piece_list_size = piece_bounds.size() * sizeof(Rect<DIM, T>);
        *piece_list = malloc(*piece_list_size);
        Rect<DIM, T>* pieces = static_cast<Rect<DIM, T>*>(*piece_list);
        for (unsigned idx = 0; idx < piece_bounds.size(); idx++)
          pieces[idx] = piece_bounds[idx];
      }
      else if (constraints.padding_constraint.delta.get_dim() > 0)
      {
        // If the user requested any scratch padding on the instance apply it
        const Domain& delta = constraints.padding_constraint.delta;
        const Point<DIM> lo = delta.lo();
        const Point<DIM> hi = delta.hi();
        legion_assert(!piece_bounds.empty());
#ifdef LEGION_DEBUG
        for (int i = 0; i < DIM; i++)
        {
          legion_assert(lo[i] >= 0);
          legion_assert(hi[i] >= 0);
        }
#endif
        for (typename std::vector<Rect<DIM, T> >::iterator it =
                 piece_bounds.begin();
             it != piece_bounds.end(); it++)
        {
          it->lo -= lo;
          it->hi += hi;
        }
      }
      const OrderingConstraint& order = constraints.ordering_constraint;
      legion_assert(order.ordering.size() == (DIM + 1));
      // Check if it is safe to re-use piece lists
      // It's only safe if fsize describes the size of a piece, which
      // is true if we only have a single piece or we're doing AOS
      const bool safe_reuse =
          ((piece_bounds.size() == 1) ||
           (order.ordering.front() == LEGION_DIM_F));
      // Get any alignment and offset constraints for individual fields
      std::map<FieldID, size_t> alignments;
      for (const AlignmentConstraint& constraint :
           constraints.alignment_constraints)
      {
        legion_assert(constraint.eqk == LEGION_EQ_EK);
        alignments[constraint.fid] = constraint.alignment;
      }
      std::map<FieldID, off_t> offsets;
      for (const OffsetConstraint& constraint : constraints.offset_constraints)
        offsets[constraint.fid] = constraint.offset;
      // Zip the fields with their sizes and sort them if we're allowed to
      std::set<size_t> unique_sizes;
      std::vector<std::pair<size_t, FieldID> > zip_fields;
      for (unsigned idx = 0; idx < field_ids.size(); idx++)
      {
        zip_fields.emplace_back(
            std::pair<size_t, FieldID>(field_sizes[idx], field_ids[idx]));
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
        std::vector<std::pair<size_t, FieldID> >::iterator it1 =
            zip_fields.begin();
        while (it1 != zip_fields.end())
        {
          std::vector<std::pair<size_t, FieldID> >::iterator it2 = it1;
          while ((it2 != zip_fields.end()) && (it1->first == it2->first)) it2++;
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
        legion_assert(int(dim) < DIM);
        for (unsigned pidx = 0; pidx < piece_bounds.size(); pidx++)
        {
          const Rect<DIM, T>& bounds = piece_bounds[pidx];
          elements_between_per_piece[pidx] *=
              (bounds.hi[dim] - bounds.lo[dim] + 1);
        }
      }
      legion_assert(field_index >= 0);
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
      for (const std::pair<size_t, FieldID>& it : zip_fields)
      {
        // if not specified, field goes at the end of all known fields
        // (or a bit past if alignment is a concern)
        size_t offset = fsize;
        std::map<FieldID, off_t>::const_iterator offset_finder =
            offsets.find(it.second);
        if (offset_finder != offsets.end())
          offset += offset_finder->second;
        std::map<FieldID, size_t>::const_iterator alignment_finder =
            alignments.find(it.second);
        // Hack to help out lazy users unwilling to specify alignment
        // constraints that are necessary for correctness
        // If they haven't specified an alignment we align on the largest
        // power of two that divides the size of the field, for more
        // details see https://github.com/StanfordLegion/legion/issues/1384
        // Cap at a maximum of 128 byte alignment for GPUs
        const size_t field_alignment =
            (alignment_finder != alignments.end()) ?
                alignment_finder->second :
                std::min<size_t>(
                    it.first & ~(it.first - 1), 128 /*max alignment*/);
        if (field_alignment > 1)
        {
          offset = round_up(offset, field_alignment);
          if ((falign % field_alignment) != 0)
            falign = std::lcm(falign, field_alignment);
        }
        // increase size and alignment if needed
        fsize = std::max(fsize, offset + it.first * elements_between_fields);
        field_offsets[it.second] = offset;
      }
      if (falign > 1)
      {
        // round up the size of the field dimension if it is not the
        // last dimension in the layout to ensure alignment
        if (order.ordering.back() != LEGION_DIM_F)
          fsize = round_up(fsize, falign);
        // overall instance alignment layout must be compatible with group
        layout->alignment_reqd = std::lcm(layout->alignment_reqd, falign);
      }
      // compute the starting offsets for each piece
      std::vector<size_t> piece_offsets(piece_bounds.size());
      if (safe_reuse)
      {
        for (unsigned pidx = 0; pidx < piece_bounds.size(); pidx++)
        {
          const Rect<DIM, T>& bounds = piece_bounds[pidx];
          piece_offsets[pidx] = round_up(layout->bytes_used, falign);
          size_t piece_size = fsize;
          for (unsigned idx = field_index + 1; idx < order.ordering.size();
               idx++)
          {
            const DimensionKind dim = order.ordering[idx];
            legion_assert(int(dim) < DIM);
            piece_size *= (bounds.hi[dim] - bounds.lo[dim] + 1);
          }
          layout->bytes_used = piece_offsets[pidx] + piece_size;
        }
      }
      // we've handled the offsets and alignment for every field across
      // all dimensions so we can just use the size of the field to
      // determine the piece list
      std::map<size_t, unsigned> pl_indexes;
      layout->piece_lists.reserve(
          safe_reuse ? unique_sizes.size() : zip_fields.size());
      for (const std::pair<size_t, FieldID>& it : zip_fields)
      {
        unsigned li;
        std::map<size_t, unsigned>::const_iterator finder =
            safe_reuse ? pl_indexes.find(it.first) : pl_indexes.end();
        if (finder == pl_indexes.end())
        {
          li = layout->piece_lists.size();
          legion_assert(
              li < (safe_reuse ? unique_sizes.size() : zip_fields.size()));
          layout->piece_lists.resize(li + 1);
          pl_indexes[it.first] = li;

          // create the piece list
          Realm::InstancePieceList<DIM, T>& pl = layout->piece_lists[li];
          pl.pieces.reserve(piece_bounds.size());

          size_t next_piece = safe_reuse ? 0 : field_offsets[it.second];
          for (unsigned pidx = 0; pidx < piece_bounds.size(); pidx++)
          {
            const Rect<DIM, T>& bounds = piece_bounds[pidx];
            // create the piece
            Realm::AffineLayoutPiece<DIM, T>* piece =
                new Realm::AffineLayoutPiece<DIM, T>;
            piece->bounds = bounds;
            size_t piece_start;
            if (safe_reuse)
              piece_start = piece_offsets[pidx];
            else
              piece_start = next_piece;
            piece->offset = piece_start;
            size_t stride = it.first;
            for (const DimensionKind& dit : order.ordering)
            {
              if ((dit) != LEGION_DIM_F)
              {
                legion_assert(int(dit) < DIM);
                piece->strides[dit] = stride;
                piece->offset -= bounds.lo[dit] * stride;
                stride *= (bounds.hi[dit] - bounds.lo[dit] + 1);
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
            if (!safe_reuse && ((pidx + 1) == piece_bounds.size()))
              layout->bytes_used = piece_start + stride;
            pl.pieces.emplace_back(piece);
          }
        }
        else
          li = finder->second;
        legion_assert(layout->fields.count(it.second) == 0);
        Realm::InstanceLayoutGeneric::FieldLayout& fl =
            layout->fields[it.second];
        fl.list_idx = li;
        fl.rel_offset = safe_reuse ? field_offsets[it.second] : 0;
        fl.size_in_bytes = it.first;
      }
      return layout;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceExpression*
        IndexSpaceExpression::create_layout_expression_internal(
            const Realm::IndexSpace<DIM, T>& space, const Rect<DIM, T>* rects,
            size_t num_rects)
    //--------------------------------------------------------------------------
    {
      if (rects == nullptr)
      {
        if (space.dense())
          return this;
        else
          // Make a new expression for the bounding box
          return new InternalExpression<DIM, T>(&space.bounds, 1 /*size*/);
      }
      else
      {
        legion_assert(num_rects > 0);
        // Make a realm expression from the rectangles
        return new InternalExpression<DIM, T>(rects, num_rects);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline bool IndexSpaceExpression::meets_layout_expression_internal(
        IndexSpaceExpression* space_expr, bool tight_bounds,
        const Rect<DIM, T>* piece_list, size_t piece_list_size,
        const Domain* padding_delta)
    //--------------------------------------------------------------------------
    {
      legion_assert(type_tag == space_expr->type_tag);
      // See if this an convex hull or a piece list case
      if (piece_list == nullptr)
      {
        // Get the bounds for each of them, can ignore ready events
        // since we're just going to be looking at the bounds
        DomainT<DIM, T> local = get_tight_domain();
        DomainT<DIM, T> other = space_expr->get_tight_domain();
        // Check to see if we contain the space expression
        if (!local.bounds.contains(other.bounds))
          return false;
        if ((padding_delta != nullptr) && (padding_delta->get_dim() > 0))
        {
          legion_assert(padding_delta->get_dim() == DIM);
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
        if ((padding_delta != nullptr) && (padding_delta->get_dim() > 0))
          return false;
        legion_assert(piece_list_size > 0);
        // Iterate the rectangles in the space expr over the piece list
        // and compute the intersection volume summary
        // Note that this assumes that the rectangles in the piece list
        // are all non-overlapping with each other
        DomainT<DIM, T> other = space_expr->get_tight_domain();
        size_t space_volume = 0;
        size_t overlap_volume = 0;
        for (Realm::IndexSpaceIterator<DIM, T> itr(other); itr.valid;
             itr.step())
        {
          size_t local_volume = itr.rect.volume();
          space_volume += local_volume;
          for (unsigned idx = 0; idx < piece_list_size; idx++)
          {
            const Rect<DIM, T> overlap = piece_list[idx].intersection(itr.rect);
            size_t volume = overlap.volume();
            if (volume == 0)
              continue;
            overlap_volume += volume;
            local_volume -= volume;
            if (local_volume == 0)
              break;
          }
        }
        legion_assert(overlap_volume <= space_volume);
        // If we didn't cover all the points in the space then we can't meet
        if (overlap_volume < space_volume)
          return false;
        if (tight_bounds)
        {
          // Check the total volume of all the pieces
          size_t piece_volume = 0;
          for (unsigned idx = 0; idx < piece_list_size; idx++)
            piece_volume += piece_list[idx].volume();
          legion_assert(space_volume <= piece_volume);
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
            const local::set<Domain>& rects)
    //--------------------------------------------------------------------------
    {
      legion_assert(!rects.empty());
      size_t total_volume = 0;
      std::vector<Rect<DIM, T> > rectangles;
      rectangles.reserve(rects.size());
      // We're just assuming that all the rectangles here are non-overlapping
      for (local::set<Domain>::const_iterator it = rects.begin();
           it != rects.end(); it++)
      {
        Rect<DIM, T> rect = *it;
        total_volume += rect.volume();
        rectangles.emplace_back(rect);
      }
      legion_assert(total_volume <= get_volume());
      // If all the points add up to the same as our volume then the
      // expressions match and we can reuse this as the expression
      if (total_volume == get_volume())
        return this;
      InternalExpression<DIM, T>* result = new InternalExpression<DIM, T>(
          &rectangles.front(), rectangles.size());
      // Do a little test to see if there is already a canonical expression
      // that we know about that matches this expression if so we'll use that
      // Note that we don't need to explicitly delete it if it is not the
      // canonical expression since it has a live expression reference that
      // will be cleaned up after this meta-task is done running
      return result->get_canonical_expression();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceExpression*
        IndexSpaceExpression::find_congruent_expression_internal(
            SmallPointerVector<IndexSpaceExpression, true>& expressions)
    //--------------------------------------------------------------------------
    {
      if (expressions.empty())
      {
        expressions.insert(this);
        return this;
      }
      else if (expressions.contains(this))
        return this;
      DomainT<DIM, T> local_space = get_tight_domain();
      size_t local_rect_count = 0;
      KDNode<DIM, T, void>* local_tree = nullptr;
      for (unsigned idx = 0; idx < expressions.size(); idx++)
      {
        IndexSpaceExpression* expr = expressions[idx];
        DomainT<DIM, T> other_space = expr->get_tight_domain();
        // See if the rectangles are the same
        if (local_space.bounds != other_space.bounds)
          continue;
        // See if the sparsity maps are the same
        if (local_space.sparsity == other_space.sparsity)
        {
          // We know that things are the same here
          // Check to see if they have the expression is still alive and
          // can be used as a canonical expression
          if (expr->try_add_live_reference())
          {
            if (local_tree != nullptr)
              delete local_tree;
            return expr;
          }
          else
            continue;
        }
        if (!local_space.sparsity.exists() || !other_space.sparsity.exists())
        {
          // Realm guarantees that tightening will remove a sparsity map if it
          // can so if one index space has a sparsity map and the other doesn't
          // then by definition they cannot be congruent (see issue #1020)
          // Should never hit this assertion as they should have equal sparsity
          // map IDs if the sparsity map does not exist for both of them
          legion_assert(
              local_space.sparsity.exists() || other_space.sparsity.exists());
          continue;
        }
        else
        {
          // Both have sparsity maps
          // We know something important though here: we know that both
          // these sparsity maps contain the same number of points
          // Build lists of both sets of rectangles
          KDNode<DIM, T>* other_tree =
              expr->get_sparsity_map_kd_tree()->as_kdnode<DIM, T>();
          size_t other_rect_count = other_tree->count_rectangles();
          if (local_rect_count == 0)
          {
            // Count the number of rectangles in our sparsity map
            for (Realm::IndexSpaceIterator<DIM, T> itr(local_space); itr.valid;
                 itr.step())
              local_rect_count++;
            legion_assert(local_rect_count > 0);
          }
          if (other_rect_count < local_rect_count)
          {
            // Build our KD tree if we haven't already
            if (local_tree == nullptr)
            {
              std::vector<Rect<DIM, T> > local_rects;
              for (Realm::IndexSpaceIterator<DIM, T> itr(local_space);
                   itr.valid; itr.step())
                local_rects.emplace_back(itr.rect);
              local_tree = new KDNode<DIM, T>(local_space.bounds, local_rects);
            }
            // Iterate the other rectangles and see if they are covered
            bool congruent = true;
            for (Realm::IndexSpaceIterator<DIM, T> itr(other_space); itr.valid;
                 itr.step())
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
            for (Realm::IndexSpaceIterator<DIM, T> itr(local_space); itr.valid;
                 itr.step())
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
        if (expr->try_add_live_reference())
        {
          if (local_tree != nullptr)
            delete local_tree;
          return expr;
        }
      }
      // Did not find any congruences so add ourself
      expressions.insert(this);
      // If we have a KD tree we can save it for later congruence tests
      if (local_tree != nullptr)
      {
        legion_assert(
            sparsity_map_kd_tree == nullptr);  // should not have a kd tree yet
        sparsity_map_kd_tree = local_tree;
      }
      return this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline KDTree* IndexSpaceExpression::get_sparsity_map_kd_tree_internal(void)
    //--------------------------------------------------------------------------
    {
      if (sparsity_map_kd_tree != nullptr)
        return sparsity_map_kd_tree;
      DomainT<DIM, T> local_space = get_tight_domain();
      legion_assert(!local_space.dense());
      std::vector<Rect<DIM, T> > local_rects;
      for (Realm::IndexSpaceIterator<DIM, T> itr(local_space); itr.valid;
           itr.step())
        local_rects.emplace_back(itr.rect);
      sparsity_map_kd_tree =
          new KDNode<DIM, T>(local_space.bounds, local_rects);
      return sparsity_map_kd_tree;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Operations
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceOperationT<DIM, T>::IndexSpaceOperationT(OperationKind kind)
      : IndexSpaceOperation(NT_TemplateHelper::encode_tag<DIM, T>(), kind),
        realm_index_space(Realm::IndexSpace<DIM, T>::make_empty()),
        is_index_space_tight(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceOperationT<DIM, T>::IndexSpaceOperationT(
        IndexSpaceExprID eid, DistributedID did, IndexSpaceOperation* origin,
        TypeTag tag, Deserializer& derez)
      : IndexSpaceOperation(tag, eid, did, origin), is_index_space_tight(false)
    //--------------------------------------------------------------------------
    {
      // We can unpack the index space here directly
      derez.deserialize(this->realm_index_space);
      this->tight_index_space = this->realm_index_space;
      // Request that we make the valid index space valid
      this->tight_index_space_ready =
          RtEvent(this->realm_index_space.make_valid());
      if (implicit_profiler != nullptr)
        implicit_profiler->record_make_valid(
            this->tight_index_space_ready, this->did);
      if (!this->tight_index_space.dense())
      {
        ApEvent added;
        derez.deserialize(added);
        if (added.exists())
        {
          added.subscribe();
          index_space_users.emplace_back(added);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceOperationT<DIM, T>::~IndexSpaceOperationT(void)
    //--------------------------------------------------------------------------
    {
      if (!this->tight_index_space.dense())
      {
        // Check to see if we have any index space users that need to be
        // checked for deferring the destruction of the index space
        std::vector<ApEvent> preconditions;
        while (!index_space_users.empty())
        {
          if (!index_space_users.front().has_triggered_faultignorant())
            preconditions.emplace_back(index_space_users.front());
          index_space_users.pop_front();
        }
        if (!preconditions.empty())
        {
          if (!tight_index_space_ready.has_triggered())
            preconditions.emplace_back(ApEvent(tight_index_space_ready));
          ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
          // Protect it if necessary to make sure the deletion happens
          // even if one of the users was poisoned
          if (precondition.exists())
            tight_index_space_ready = Runtime::protect_event(precondition);
        }
        this->tight_index_space.destroy(tight_index_space_ready);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceOperationT<DIM, T>::is_sparse(void)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> result = get_tight_index_space();
      return !result.dense();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainT<DIM, T> IndexSpaceOperationT<DIM, T>::get_tight_index_space(void)
    //--------------------------------------------------------------------------
    {
      if (!is_index_space_tight.load())
      {
        // Wait for the index space to be tight
        if (tight_index_space_ready.exists() &&
            !tight_index_space_ready.has_triggered())
          tight_index_space_ready.wait();
        // In case the reason we had a tight event was because we are remote
        is_index_space_tight.store(true);
      }
      // Already tight so we can just return that
      return tight_index_space;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM, T>::get_loose_index_space(
        DomainT<DIM, T>& space, ApUserEvent& to_trigger)
    //--------------------------------------------------------------------------
    {
      if (!is_index_space_tight.load())
      {
        AutoLock i_lock(inter_lock);
        // Check to see if we lost the race
        if (!is_index_space_tight.load())
        {
          // Still not tight so record a user on the index space
          if (!to_trigger.exists())
            to_trigger = Runtime::create_ap_user_event(nullptr);
          while (!index_space_users.empty() &&
                 index_space_users.front().has_triggered_faultignorant())
            index_space_users.pop_front();
          index_space_users.emplace_back(to_trigger);
          space = realm_index_space;
          return realm_index_space_ready;
        }
      }
      // Already tight so we can just return that
      space = tight_index_space;
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Domain IndexSpaceOperationT<DIM, T>::get_tight_domain(void)
    //--------------------------------------------------------------------------
    {
      return Domain(get_tight_index_space());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM, T>::get_loose_domain(
        Domain& domain, ApUserEvent& to_trigger)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> index_space;
      const ApEvent result = get_loose_index_space(index_space, to_trigger);
      domain = index_space;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM, T>::record_index_space_user(ApEvent user)
    //--------------------------------------------------------------------------
    {
      if (!user.exists())
        return;
      // Can check some of these without the lock
      // Only need to save this if the sparsity map exists
      if (is_index_space_tight.load())
      {
        if (tight_index_space.dense())
          return;
        AutoLock i_lock(inter_lock);
        // Try popping entries off the front of the list
        while (!index_space_users.empty() &&
               index_space_users.front().has_triggered_faultignorant())
          index_space_users.pop_front();
        index_space_users.emplace_back(user);
      }
      else if (!realm_index_space.dense())
      {
        AutoLock i_lock(inter_lock);
        if (is_index_space_tight.load())
        {
          if (tight_index_space.dense())
            return;
        }
        else if (realm_index_space.dense())
          return;
        // Try popping entries off the front of the list
        while (!index_space_users.empty() &&
               index_space_users.front().has_triggered_faultignorant())
          index_space_users.pop_front();
        index_space_users.emplace_back(user);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM, T>::tighten_index_space(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(realm_index_space.is_valid());
      tight_index_space = realm_index_space.tighten();
      legion_assert(tight_index_space.is_valid());
      is_index_space_tight.store(true);
      if (!realm_index_space.dense() && tight_index_space.dense())
      {
        AutoLock i_lock(inter_lock);
        // Check to see if we have any index space users that need to be
        // checked for deferring the destruction of the index space
        std::vector<ApEvent> preconditions;
        while (!index_space_users.empty())
        {
          if (!index_space_users.front().has_triggered_faultignorant())
            preconditions.emplace_back(index_space_users.front());
          index_space_users.pop_front();
        }
        if (!preconditions.empty())
        {
          if (!tight_index_space_ready.has_triggered())
            preconditions.emplace_back(ApEvent(tight_index_space_ready));
          ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
          // Protect it if necessary to make sure the deletion happens
          // even if one of the users was poisoned
          if (precondition.exists())
            tight_index_space_ready = Runtime::protect_event(precondition);
        }
        realm_index_space.destroy(realm_index_space_ready);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceOperationT<DIM, T>::check_empty(void)
    //--------------------------------------------------------------------------
    {
      return (get_volume() == 0);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceOperationT<DIM, T>::get_volume(void)
    //--------------------------------------------------------------------------
    {
      if (has_volume.load())
        return volume;
      DomainT<DIM, T> temp = get_tight_index_space();
      volume = temp.volume();
      has_volume.store(true);
      return volume;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM, T>::pack_expression(
        Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->is_valid());
      if (target == this->local_space)
      {
        rez.serialize<bool>(true /*local*/);
        rez.serialize(this);
        // Caller-side increment only - the unpacker will record_live_expression
        if (!this->check_global_and_increment(LIVE_EXPR_REF))
          std::abort();
      }
      else if (target == this->owner_space)
      {
        rez.serialize<bool>(true /*local*/);
        rez.serialize(origin_expr);
        // Add a reference here that we'll remove after we've added a reference
        // on the target space expression
        this->pack_global_ref();
      }
      else
      {
        rez.serialize<bool>(false /*local*/);
        rez.serialize<bool>(false /*index space*/);
        rez.serialize(this->expr_id);
        rez.serialize(this->type_tag);
        rez.serialize(this->origin_expr);
        rez.serialize(this->did);
        DomainT<DIM, T> temp = this->get_tight_index_space();
        rez.serialize(temp);
        if (!temp.dense())
        {
          const ApEvent added(temp.sparsity.add_reference());
          rez.serialize(added);
        }
        // Record that we send a copy to the target address space
        if (this->is_owner())
          this->update_remote_instances(target);
        // Add a reference here that we'll remove after we've added a reference
        // on the target space expression
        this->pack_global_ref();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM, T>::skip_unpack_expression(
        Deserializer& derez) const
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      TypeTag tag;
      derez.deserialize(tag);
      legion_assert(tag == this->type_tag);
      IndexSpaceOperation* origin;
      derez.deserialize(origin);
      legion_assert(origin == this->origin_expr);
      DistributedID id;
      derez.deserialize(id);
      legion_assert(id == did);
#else
      derez.advance_pointer(
          sizeof(type_tag) + sizeof(origin_expr) + sizeof(did));
#endif
      Realm::IndexSpace<DIM, T> space;
      derez.deserialize(space);
      if (!space.dense())
      {
        ApEvent added;
        derez.deserialize(added);
        space.sparsity.destroy(added);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNode* IndexSpaceOperationT<DIM, T>::create_node(
        IndexSpace handle, RtEvent initialized, Provenance* provenance,
        CollectiveMapping* collective_mapping, IndexSpaceExprID new_expr_id)
    //--------------------------------------------------------------------------
    {
      if (new_expr_id == 0)
        new_expr_id = expr_id;
      AutoLock i_lock(inter_lock, false /*exclusive*/);
      if (is_index_space_tight.load())
        return runtime->create_node(
            handle, Domain(tight_index_space), false /*take ownership*/,
            nullptr /*parent*/, 0 /*color*/, initialized, provenance,
            realm_index_space_ready, new_expr_id, collective_mapping,
            true /*add root ref*/);
      else
        return runtime->create_node(
            handle, Domain(realm_index_space), false /*take ownership*/,
            nullptr /*parent*/, 0 /*color*/, initialized, provenance,
            realm_index_space_ready, new_expr_id, collective_mapping,
            true /*add root ref*/);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceOperationT<DIM, T>::create_from_rectangles(
        const local::set<Domain>& rects)
    //--------------------------------------------------------------------------
    {
      return create_from_rectangles_internal<DIM, T>(rects);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceOperationT<DIM, T>::inline_union(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      return inline_union_internal<DIM, T>(rhs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceOperationT<DIM, T>::inline_union(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      return inline_union_internal<DIM, T>(exprs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceOperationT<DIM, T>::inline_intersection(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      return inline_intersection_internal<DIM, T>(rhs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceOperationT<DIM, T>::inline_intersection(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      return inline_intersection_internal<DIM, T>(exprs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceOperationT<DIM, T>::inline_subtraction(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      return inline_subtraction_internal<DIM, T>(rhs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    uint64_t IndexSpaceOperationT<DIM, T>::get_canonical_hash(void)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> domain = get_tight_index_space();
      return get_canonical_hash_internal(domain);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DistributedID IndexSpaceOperationT<DIM, T>::record_profiler_expression(void)
    //--------------------------------------------------------------------------
    {
      if (!profiler_logged.exchange(true))
      {
        DomainT<DIM, T> tight_space = get_tight_index_space();
        log_profiler_points(did, tight_space);
      }
      return did;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM, T>::issue_fill(
        Operation* op, const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields, const void* fill_value,
        size_t fill_size, UniqueID fill_uid, FieldSpace handle,
        RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent unique_event, CollectiveKind collective, bool record_effect,
        int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      return issue_fill_internal(
          op, local_space, trace_info, dst_fields, fill_value, fill_size,
          fill_uid, handle, tree_id, precondition, pred_guard, unique_event,
          collective, priority, replay, record_effect);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceOperationT<DIM, T>::issue_copy(
        Operation* op, const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<CopySrcDstField>& src_fields,
        const std::vector<Reservation>& reservations, RegionTreeID src_tree_id,
        RegionTreeID dst_tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent src_unique, LgEvent dst_unique, CollectiveKind collective,
        bool record_effect, int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      return issue_copy_internal(
          op, local_space, trace_info, dst_fields, src_fields, reservations,
          src_tree_id, dst_tree_id, precondition, pred_guard, src_unique,
          dst_unique, collective, record_effect, priority, replay);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceOperationT<DIM, T>::create_layout(
        const LayoutConstraintSet& constraints,
        const std::vector<FieldID>& field_ids,
        const std::vector<size_t>& field_sizes, bool compact, void** piece_list,
        size_t* piece_list_size, size_t* num_pieces, size_t base_alignment)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_is = get_tight_index_space();
      return create_layout_internal(
          local_is, constraints, field_ids, field_sizes, compact, piece_list,
          piece_list_size, num_pieces, base_alignment);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression*
        IndexSpaceOperationT<DIM, T>::create_layout_expression(
            const void* piece_list, size_t piece_list_size)
    //--------------------------------------------------------------------------
    {
      legion_assert((piece_list_size % sizeof(Rect<DIM, T>)) == 0);
      DomainT<DIM, T> local_is = get_tight_index_space();
      // No need to wait for the index space to be ready since we
      // are never actually going to look at the sparsity map
      return create_layout_expression_internal(
          local_is, static_cast<const Rect<DIM, T>*>(piece_list),
          piece_list_size / sizeof(Rect<DIM, T>));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceOperationT<DIM, T>::meets_layout_expression(
        IndexSpaceExpression* space_expr, bool tight_bounds,
        const void* piece_list, size_t piece_list_size,
        const Domain* padding_delta)
    //--------------------------------------------------------------------------
    {
      legion_assert((piece_list_size % sizeof(Rect<DIM, T>)) == 0);
      return meets_layout_expression_internal<DIM, T>(
          space_expr, tight_bounds,
          static_cast<const Rect<DIM, T>*>(piece_list),
          piece_list_size / sizeof(Rect<DIM, T>), padding_delta);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression*
        IndexSpaceOperationT<DIM, T>::find_congruent_expression(
            SmallPointerVector<IndexSpaceExpression, true>& expressions)
    //--------------------------------------------------------------------------
    {
      return find_congruent_expression_internal<DIM, T>(expressions);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    KDTree* IndexSpaceOperationT<DIM, T>::get_sparsity_map_kd_tree(void)
    //--------------------------------------------------------------------------
    {
      return get_sparsity_map_kd_tree_internal<DIM, T>();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM, T>::initialize_equivalence_set_kd_tree(
        EqKDTree* tree, EquivalenceSet* set, const FieldMask& mask,
        ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
      {
        const Rect<DIM, T> overlap = itr.rect.intersection(typed_tree->bounds);
        if (!overlap.empty())
          typed_tree->initialize_set(set, overlap, mask, local_shard, current);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceOperationT<DIM, T>::compute_equivalence_sets(
        EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
        const std::vector<EqSetTracker*>& trackers,
        const std::vector<AddressSpaceID>& tracker_spaces,
        std::vector<unsigned>& new_tracker_references,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        std::vector<RtEvent>& pending_sets,
        op::FieldMaskMap<EqKDTree>& subscriptions,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock, false /*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        typed_tree->compute_equivalence_sets(
            itr.rect, mask, trackers, tracker_spaces, new_tracker_references,
            eq_sets, pending_sets, subscriptions, to_create, creation_rects,
            creation_srcs, remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned IndexSpaceOperationT<DIM, T>::record_output_equivalence_set(
        EqKDTree* tree, LocalLock* tree_lock, EquivalenceSet* set,
        const FieldMask& mask, EqSetTracker* tracker,
        AddressSpaceID tracker_space,
        local::FieldMaskMap<EqKDTree>& subscriptions,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      unsigned new_subs = 0;
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock, false /*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        new_subs += typed_tree->record_output_equivalence_set(
            set, itr.rect, mask, tracker, tracker_space, subscriptions,
            remote_shard_rects, local_shard);
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceUnion<DIM, T>::IndexSpaceUnion(
        const std::vector<IndexSpaceExpression*>& to_union)
      : IndexSpaceOperationT<DIM, T>(IndexSpaceOperation::UNION_OP_KIND),
        sub_expressions(to_union)
    //--------------------------------------------------------------------------
    {
      // Add an resource ref that will be removed by the OperationCreator
      this->add_base_resource_ref(REGION_TREE_REF);
      ApUserEvent to_trigger;
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM, T> > spaces(sub_expressions.size());
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
      {
        IndexSpaceExpression* sub = sub_expressions[idx];
        legion_assert(sub->get_canonical_expression() == sub);
        // Add the parent and the reference
        sub->add_derived_operation(this);
        sub->add_tree_expression_reference(this->did);
        // Then get the realm index space expression
        Domain domain;
        ApEvent precondition = sub->get_loose_domain(domain, to_trigger);
        if (precondition.exists())
          preconditions.insert(precondition);
        spaces[idx] = domain;
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, implicit_unique_op_id, DEP_PART_UNION_REDUCTION,
            precondition);
      this->realm_index_space_ready =
          ApEvent(Realm::IndexSpace<DIM, T>::compute_union(
              spaces, this->realm_index_space, requests, precondition));
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(
            to_trigger, this->realm_index_space_ready);
      // Then launch the tighten call for it too since we know we're
      // going to want this eventually
      const RtEvent valid_event(this->realm_index_space.make_valid());
      if (implicit_profiler != nullptr)
        implicit_profiler->record_make_valid(valid_event, this->did);
      // See if both the events needed for the tighten call are done
      if (this->realm_index_space_ready.exists() ||
          !valid_event.has_triggered())
      {
        IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
        if (this->realm_index_space_ready.exists())
        {
          if (!valid_event.has_triggered())
            this->tight_index_space_ready = runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY,
                Runtime::merge_events(
                    valid_event,
                    Runtime::protect_event(this->realm_index_space_ready)));
          else
            this->tight_index_space_ready = runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY,
                Runtime::protect_event(this->realm_index_space_ready));
        }
        else
          this->tight_index_space_ready = runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, valid_event);
      }
      else  // We can do the tighten call now
        this->tighten_index_space();
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        std::vector<IndexSpaceExprID> sources(this->sub_expressions.size());
        for (unsigned idx = 0; idx < this->sub_expressions.size(); idx++)
          sources[idx] = this->sub_expressions[idx]->expr_id;
        LegionSpy::log_index_space_union(this->expr_id, sources);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceUnion<DIM, T>::IndexSpaceUnion(const Rect<DIM, T>& bounds)
      : IndexSpaceOperationT<DIM, T>(IndexSpaceOperation::UNION_OP_KIND)
    //--------------------------------------------------------------------------
    {
      // Shouldn't be here if Legion Spy is enabled since we don't have
      // logging for this and we don't want to make too many index space
      // expressions for Legion Spy to deal with
      legion_assert(spy_logging_level == NO_SPY_LOGGING);
      this->realm_index_space = DomainT<DIM, T>(bounds);
      this->tight_index_space = this->realm_index_space;
      this->is_index_space_tight.store(true);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceUnion<DIM, T>::~IndexSpaceUnion(void)
    //--------------------------------------------------------------------------
    {
      // Remove references from our sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        if (sub_expressions[idx]->remove_tree_expression_reference(this->did))
          delete sub_expressions[idx];
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceUnion<DIM, T>::invalidate_operation(
        IndexSpaceExpression* source)
    //--------------------------------------------------------------------------
    {
      // Remove the parent operation from all the sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        if (sub_expressions[idx] != source)
          sub_expressions[idx]->remove_derived_operation(this);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceUnion<DIM, T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      if (!sub_expressions.empty())
        runtime->remove_union_operation(this, sub_expressions);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceIntersection<DIM, T>::IndexSpaceIntersection(
        const std::vector<IndexSpaceExpression*>& to_inter)
      : IndexSpaceOperationT<DIM, T>(IndexSpaceOperation::INTERSECT_OP_KIND),
        sub_expressions(to_inter)
    //--------------------------------------------------------------------------
    {
      // Add an resource ref that will be removed by the OperationCreator
      this->add_base_resource_ref(REGION_TREE_REF);
      ApUserEvent to_trigger;
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM, T> > spaces(sub_expressions.size());
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
      {
        IndexSpaceExpression* sub = sub_expressions[idx];
        legion_assert(sub->get_canonical_expression() == sub);
        // Add the parent and the reference
        sub->add_derived_operation(this);
        sub->add_tree_expression_reference(this->did);
        Domain domain;
        ApEvent precondition = sub->get_loose_domain(domain, to_trigger);
        if (precondition.exists())
          preconditions.insert(precondition);
        spaces[idx] = domain;
      }
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, implicit_unique_op_id, DEP_PART_INTERSECTION_REDUCTION,
            precondition);
      this->realm_index_space_ready =
          ApEvent(Realm::IndexSpace<DIM, T>::compute_intersection(
              spaces, this->realm_index_space, requests, precondition));
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(
            to_trigger, this->realm_index_space_ready);
      // Then launch the tighten call for it too since we know we're
      // going to want this eventually
      const RtEvent valid_event(this->realm_index_space.make_valid());
      if (implicit_profiler != nullptr)
        implicit_profiler->record_make_valid(valid_event, this->did);
      // See if both the events needed for the tighten call are done
      if (this->realm_index_space_ready.exists() ||
          !valid_event.has_triggered())
      {
        IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
        if (this->realm_index_space_ready.exists())
        {
          if (!valid_event.has_triggered())
            this->tight_index_space_ready = runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY,
                Runtime::merge_events(
                    valid_event,
                    Runtime::protect_event(this->realm_index_space_ready)));
          else
            this->tight_index_space_ready = runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY,
                Runtime::protect_event(this->realm_index_space_ready));
        }
        else
          this->tight_index_space_ready = runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, valid_event);
      }
      else  // We can do the tighten call now
        this->tighten_index_space();
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        std::vector<IndexSpaceExprID> sources(this->sub_expressions.size());
        for (unsigned idx = 0; idx < this->sub_expressions.size(); idx++)
          sources[idx] = this->sub_expressions[idx]->expr_id;
        LegionSpy::log_index_space_intersection(this->expr_id, sources);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceIntersection<DIM, T>::IndexSpaceIntersection(
        const Rect<DIM, T>& bounds)
      : IndexSpaceOperationT<DIM, T>(IndexSpaceOperation::INTERSECT_OP_KIND)
    //--------------------------------------------------------------------------
    {
      // Shouldn't be here if Legion Spy is enabled since we don't have
      // logging for this and we don't want to make too many index space
      // expressions for Legion Spy to deal with
      legion_assert(spy_logging_level == NO_SPY_LOGGING);
      this->realm_index_space = DomainT<DIM, T>(bounds);
      this->tight_index_space = this->realm_index_space;
      this->is_index_space_tight.store(true);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceIntersection<DIM, T>::~IndexSpaceIntersection(void)
    //--------------------------------------------------------------------------
    {
      // Remove references from our sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        if (sub_expressions[idx]->remove_tree_expression_reference(this->did))
          delete sub_expressions[idx];
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceIntersection<DIM, T>::invalidate_operation(
        IndexSpaceExpression* source)
    //--------------------------------------------------------------------------
    {
      // Remove the parent operation from all the sub expressions
      for (unsigned idx = 0; idx < sub_expressions.size(); idx++)
        if (sub_expressions[idx] != source)
          sub_expressions[idx]->remove_derived_operation(this);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceIntersection<DIM, T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      if (!sub_expressions.empty())
        runtime->remove_intersection_operation(this, sub_expressions);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceDifference<DIM, T>::IndexSpaceDifference(
        IndexSpaceExpression* l, IndexSpaceExpression* r)
      : IndexSpaceOperationT<DIM, T>(IndexSpaceOperation::DIFFERENCE_OP_KIND),
        lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
      // Add an resource ref that will be removed by the OperationCreator
      this->add_base_resource_ref(REGION_TREE_REF);
      legion_assert(lhs->get_canonical_expression() == lhs);
      legion_assert(rhs->get_canonical_expression() == rhs);
      if (lhs == rhs)
      {
        // Special case for when the expressions are the same
        lhs->add_derived_operation(this);
        lhs->add_tree_expression_reference(this->did);
        this->realm_index_space = Realm::IndexSpace<DIM, T>::make_empty();
        this->tight_index_space = Realm::IndexSpace<DIM, T>::make_empty();
        this->realm_index_space_ready = ApEvent::NO_AP_EVENT;
        this->tight_index_space_ready = RtEvent::NO_RT_EVENT;
      }
      else
      {
        // Add the parent and the references
        lhs->add_derived_operation(this);
        rhs->add_derived_operation(this);
        lhs->add_tree_expression_reference(this->did);
        rhs->add_tree_expression_reference(this->did);
        Domain domain;
        ApUserEvent to_trigger;
        ApEvent left_ready = lhs->get_loose_domain(domain, to_trigger);
        DomainT<DIM, T> lhs_space = domain;
        ApEvent right_ready = rhs->get_loose_domain(domain, to_trigger);
        DomainT<DIM, T> rhs_space = domain;
        ApEvent precondition =
            Runtime::merge_events(nullptr, left_ready, right_ready);
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, implicit_unique_op_id, DEP_PART_DIFFERENCE,
              precondition);
        this->realm_index_space_ready =
            ApEvent(Realm::IndexSpace<DIM, T>::compute_difference(
                lhs_space, rhs_space, this->realm_index_space, requests,
                precondition));
        if (to_trigger.exists())
          Runtime::trigger_event_untraced(
              to_trigger, this->realm_index_space_ready);
        // Then launch the tighten call for it too since we know we're
        // going to want this eventually
        const RtEvent valid_event(this->realm_index_space.make_valid());
        if (implicit_profiler != nullptr)
          implicit_profiler->record_make_valid(valid_event, this->did);
        // See if both the events needed for the tighten call are done
        if (this->realm_index_space_ready.exists() ||
            !valid_event.has_triggered())
        {
          IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
          if (this->realm_index_space_ready.exists())
          {
            if (!valid_event.has_triggered())
              this->tight_index_space_ready = runtime->issue_runtime_meta_task(
                  args, LG_LATENCY_WORK_PRIORITY,
                  Runtime::merge_events(
                      valid_event,
                      Runtime::protect_event(this->realm_index_space_ready)));
            else
              this->tight_index_space_ready = runtime->issue_runtime_meta_task(
                  args, LG_LATENCY_WORK_PRIORITY,
                  Runtime::protect_event(this->realm_index_space_ready));
          }
          else
            this->tight_index_space_ready = runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY, valid_event);
        }
        else  // We can do the tighten call now
          this->tighten_index_space();
      }
      if (spy_logging_level > NO_SPY_LOGGING)
        LegionSpy::log_index_space_difference(
            this->expr_id, lhs->expr_id, rhs->expr_id);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceDifference<DIM, T>::IndexSpaceDifference(
        const Rect<DIM, T>& bounds)
      : IndexSpaceOperationT<DIM, T>(IndexSpaceOperation::DIFFERENCE_OP_KIND),
        lhs(nullptr), rhs(nullptr)
    //--------------------------------------------------------------------------
    {
      // Shouldn't be here if Legion Spy is enabled since we don't have
      // logging for this and we don't want to make too many index space
      // expressions for Legion Spy to deal with
      legion_assert(spy_logging_level == NO_SPY_LOGGING);
      this->realm_index_space = DomainT<DIM, T>(bounds);
      this->tight_index_space = this->realm_index_space;
      this->is_index_space_tight.store(true);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceDifference<DIM, T>::~IndexSpaceDifference(void)
    //--------------------------------------------------------------------------
    {
      if ((rhs != nullptr) && (lhs != rhs) &&
          rhs->remove_tree_expression_reference(this->did))
        delete rhs;
      if ((lhs != nullptr) && lhs->remove_tree_expression_reference(this->did))
        delete lhs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceDifference<DIM, T>::invalidate_operation(
        IndexSpaceExpression* source)
    //--------------------------------------------------------------------------
    {
      // Remove the parent operation from all the sub expressions
      if ((lhs != nullptr) && (lhs != source))
        lhs->remove_derived_operation(this);
      if ((rhs != nullptr) && (lhs != rhs) && (rhs != source))
        rhs->remove_derived_operation(this);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceDifference<DIM, T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      if ((lhs != nullptr) && (rhs != nullptr))
        runtime->remove_subtraction_operation(this, lhs, rhs);
    }

    /////////////////////////////////////////////////////////////
    // Instance Expression
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    InternalExpression<DIM, T>::InternalExpression(
        const Rect<DIM, T>* rects, size_t num_rects)
      : IndexSpaceOperationT<DIM, T>(
            IndexSpaceOperation::INSTANCE_EXPRESSION_KIND)
    //--------------------------------------------------------------------------
    {
      // This is another kind of live expression made by the region tree
      if (!this->try_add_live_reference())
        std::abort();
      legion_assert(num_rects > 0);
      if (num_rects > 1)
      {
        std::vector<Realm::Rect<DIM, T> > realm_rects(num_rects);
        for (unsigned idx = 0; idx < num_rects; idx++)
          realm_rects[idx] = rects[idx];
        this->realm_index_space = Realm::IndexSpace<DIM, T>(realm_rects);
        const RtEvent valid_event(this->realm_index_space.make_valid());
        if (implicit_profiler != nullptr)
          implicit_profiler->record_make_valid(valid_event, this->did);
        if (!valid_event.has_triggered())
        {
          IndexSpaceExpression::TightenIndexSpaceArgs args(this, this);
          this->tight_index_space_ready = runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, valid_event);
        }
        else  // We can do the tighten call now
          this->tighten_index_space();
      }
      else
      {
        this->realm_index_space.bounds = rects[0];
        this->realm_index_space.sparsity.id = 0;
        this->tight_index_space = this->realm_index_space;
        this->is_index_space_tight.store(true);
      }
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        // These index expressions cannot be computed, so we'll pretend
        // like they are index spaces to Legion Spy since these are
        // effectively new "atom" index spaces for Legion Spy's analysis
        const DistributedID fake_space_id =
            runtime->get_unique_index_space_id();
        LegionSpy::log_top_index_space(
            fake_space_id, runtime->address_space, std::string_view());
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
    InternalExpression<DIM, T>::~InternalExpression(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void InternalExpression<DIM, T>::invalidate_operation(
        IndexSpaceExpression* source)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void InternalExpression<DIM, T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here since we're not in the region tree
    }

    // This is a bit out of place but needs to be here for instantiation to work

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* EqKDTreeT<DIM, T>::create_from_rectangles(
        const std::vector<Domain>& rects) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!rects.empty());
      std::vector<Rect<DIM, T> > rectangles(rects.size());
      for (unsigned idx = 0; idx < rects.size(); idx++)
        rectangles[idx] = rects[idx];
      InternalExpression<DIM, T>* result = new InternalExpression<DIM, T>(
          &rectangles.front(), rectangles.size());
      // Do a little test to see if there is already a canonical expression
      // that we know about that matches this expression if so we'll use that
      // Note that we don't need to explicitly delete it if it is not the
      // canonical expression since it has a live expression reference that
      // will be cleaned up after this meta-task is done running
      return result->get_canonical_expression();
    }

    /////////////////////////////////////////////////////////////
    // Remote Expression
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RemoteExpression<DIM, T>::RemoteExpression(
        IndexSpaceExprID eid, DistributedID did, IndexSpaceOperation* origin,
        TypeTag tag, Deserializer& derez)
      : IndexSpaceOperationT<DIM, T>(eid, did, origin, tag, derez)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RemoteExpression<DIM, T>::~RemoteExpression(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void RemoteExpression<DIM, T>::invalidate_operation(
        IndexSpaceExpression* source)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void RemoteExpression<DIM, T>::remove_operation(void)
    //--------------------------------------------------------------------------
    {
      // nothing to do here
    }
#endif  // DEFINE_NT_TEMPLATES

  }  // namespace Internal
}  // namespace Legion
