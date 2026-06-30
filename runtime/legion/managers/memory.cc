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

#include <sys/mman.h>  // munlock

#include "legion/api/config.h"  // need this for cuda/hip includes

#ifdef LEGION_USE_CUDA
#include <cuda.h>
#ifdef LEGION_MALLOC_INSTANCES
#include "realm/cuda/cuda_access.h"
#endif
#endif
#ifdef LEGION_USE_HIP
#include <hip/hip_runtime.h>
#ifdef LEGION_MALLOC_INSTANCES
#include "realm/hip/hip_access.h"
#endif
#endif

#include "legion/managers/memory.h"
#include "legion/contexts/context.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/builder.h"
#include "legion/instances/physical.h"
#include "legion/api/future_impl.h"
#include "legion/nodes/region.h"
#include "legion/tasks/single.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Memory Pool
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    /*static*/ void MemoryPool::pack_null_pool(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(Memory::NO_MEMORY);
    }

    //--------------------------------------------------------------------------
    /*static*/ MemoryPool* MemoryPool::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      Memory memory;
      derez.deserialize(memory);
      if (!memory.exists())
        return nullptr;
      MemoryManager* manager = runtime->find_memory_manager(memory);
      bool bounded;
      derez.deserialize<bool>(bounded);
      if (bounded)
      {
        size_t size;
        derez.deserialize(size);
        size_t alignment;
        derez.deserialize(alignment);
        PhysicalInstance instance;
        derez.deserialize(instance);
        RtEvent use_event;
        derez.deserialize(use_event);
        LgEvent unique_event;
        derez.deserialize(unique_event);
        return new ConcretePool(
            instance, size, alignment, use_event, unique_event, manager);
      }
      else
      {
        size_t max_free_bytes;
        derez.deserialize(max_free_bytes);
        UnboundPoolScope scope;
        derez.deserialize(scope);
        TaskTreeCoordinates coordinates;
        coordinates.deserialize(derez);
        UnboundPool* pool =
            new UnboundPool(manager, scope, coordinates, max_free_bytes);
        pool->unpack(derez);
        return pool;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ Realm::InstanceLayoutGeneric* MemoryPool::create_layout(
        size_t size, size_t alignment, size_t offset)
    //--------------------------------------------------------------------------
    {
      legion_assert(alignment > 0);
      const DomainT<1, coord_t> bounds =
          (size == 0) ? Rect<1>(1, 0) :
                        Rect<1>(offset, Point<1>(offset + size - 1));
      const std::vector<Realm::FieldID> field_ids(1, FID);
      const std::vector<size_t> field_sizes(1, sizeof(uint8_t));
      Realm::InstanceLayoutConstraints constraints(
          field_ids, field_sizes, 0 /*blocking*/);
      const int dim_order[] = {0};
      Realm::InstanceLayoutGeneric* layout =
          Realm::InstanceLayoutGeneric::choose_instance_layout(
              bounds, constraints, dim_order);
      layout->alignment_reqd = alignment;
      return layout;
    }

    //--------------------------------------------------------------------------
    ConcretePool::ConcretePool(
        PhysicalInstance inst, size_t remain, size_t alignment, RtEvent use,
        LgEvent unique, MemoryManager* man)
      : MemoryPool(alignment), manager(man), limit(remain),
        remaining_bytes(remain), first_unused_range(SENTINEL),
        ranges_initialized(false), released(false), finalized(false)
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      legion_assert(inst.exists() == (remain > 0));
      // Might not have an instance if this is a zero-sized pool
      if (inst.exists())
        backing_instances.emplace(
            std::make_pair(inst, std::make_pair(use, unique)));
    }

    //--------------------------------------------------------------------------
    ConcretePool::~ConcretePool(void)
    //--------------------------------------------------------------------------
    {
      if (!finalized)
        finalize_pool(RtEvent::NO_RT_EVENT);
    }

    //--------------------------------------------------------------------------
    ApEvent ConcretePool::get_ready_event(void) const
    //--------------------------------------------------------------------------
    {
      if (backing_instances.empty())
        return ApEvent::NO_AP_EVENT;
      legion_assert(backing_instances.size() == 1);
      return ApEvent(backing_instances.begin()->second.first);
    }

    //--------------------------------------------------------------------------
    size_t ConcretePool::query_memory_limit(void)
    //--------------------------------------------------------------------------
    {
      return limit;
    }

    //--------------------------------------------------------------------------
    size_t ConcretePool::query_available_memory(void)
    //--------------------------------------------------------------------------
    {
      if (released)
        return 0;
      else
        return remaining_bytes;
    }

    //--------------------------------------------------------------------------
    PoolBounds ConcretePool::get_bounds(void) const
    //--------------------------------------------------------------------------
    {
      return PoolBounds(limit, max_alignment);
    }

    //--------------------------------------------------------------------------
    void ConcretePool::capture_task_instances(
        const std::map<PhysicalManager*, unsigned>& instances)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here since we know we won't be trying to do deferred
      // deletions to satisfy allocations for this pool
    }

    //--------------------------------------------------------------------------
    FutureInstance* ConcretePool::allocate_future(
        UniqueID creator_uid, size_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      legion_assert(!released);
      if (remaining_bytes < size)
        return nullptr;
      size_t alignment = manager->compute_future_alignment(size);
      uintptr_t start = 0;
      const unsigned range_index = allocate(size, alignment, start);
      if (range_index == SENTINEL)
        return nullptr;
      // Create the layout for the future
      const std::vector<Realm::FieldID> fids(1, 0 /*field id*/);
      const std::vector<size_t> sizes(1, size);
      const int dim_order[1] = {0};
      const Realm::Point<1, coord_t> zero(0);
      const Realm::InstanceLayoutConstraints constraints(fids, sizes, 1);
      const Realm::IndexSpace<1, coord_t> rect_space(
          Realm::Rect<1, coord_t>(zero, zero));
      Realm::InstanceLayoutGeneric* layout =
          Realm::InstanceLayoutGeneric::choose_instance_layout<1, coord_t>(
              rect_space, constraints, dim_order);
      layout->alignment_reqd = alignment;
      // Futures are always going to escape out of this pool and ownership
      // will be taken by the caller so we escape them eagerly
      LgEvent unique_event;
      PhysicalInstance instance;
      RtEvent use_event = escape_range(
          range_index, 1 /*num_results*/, &instance, &unique_event,
          (const Realm::InstanceLayoutGeneric**)&layout, creator_uid);
      delete layout;
      legion_assert(instance.exists());
      return new FutureInstance(
          reinterpret_cast<const void*>(start), size, false /*external*/,
          true /*own allocation*/, unique_event, instance, Processor::NO_PROC,
          use_event);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance ConcretePool::allocate_instance(
        UniqueID creator_uid, LgEvent unique_event,
        const Realm::InstanceLayoutGeneric& layout, RtEvent& use_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(!released);
      // Should have been checked earlier
      legion_assert(layout.alignment_reqd <= max_alignment);
      if (layout.bytes_used == 0)
      {
        // Special case for empty instances
        Realm::InstanceLayoutGeneric* empty_layout = create_layout(0, 1);
        PhysicalInstance instance;
        const Realm::ProfilingRequestSet empty_requests;
        use_event = RtEvent(PhysicalInstance::create_instance(
            instance, manager->memory, *empty_layout, empty_requests));
        legion_assert(instance.exists());
        legion_assert(allocated.find(instance) == allocated.end());
        allocated[instance] = SENTINEL;
        delete empty_layout;
        return instance;
      }
      // Iterate over the free lists from smallest to largets looking
      // for a hole that is big enough to store the instance
      uintptr_t start = 0;
      unsigned range_index =
          allocate(layout.bytes_used, layout.alignment_reqd, start);
      if (range_index != SENTINEL)
      {
        // Allocation succeeded
        // Make an external instance for the data
        const size_t offset = start - ranges.front().first;
        const Point<1> start(offset);
        const Point<1> stop(offset + layout.bytes_used - 1);
        const Rect<1> bounds(start, stop);
        const Range& range = ranges[range_index];
        Realm::ExternalInstanceResource* external_resource =
            range.instance.generate_resource_info(
                Realm::IndexSpaceGeneric(bounds), FID, false /*read only*/);
        // We don't profile intermediate instances like this since no
        // Legion operations will happen on them until they're escaped
        PhysicalInstance instance;
        const Realm::ProfilingRequestSet empty_requests;
        // Normally Realm insists that we use suggested_memory() to
        // name the memory but since we know this pool is backed by
        // a normal Realm instance we can use that memory
        use_event = RtEvent(PhysicalInstance::create_external_instance(
            instance, manager->memory, layout, *external_resource,
            empty_requests, backing_instances[range.instance].first));
        // Free up the external resource that Realm created which is
        // bad for other reasons but necessary to avoid leaks
        delete external_resource;
        legion_assert(instance.exists());
        legion_assert(allocated.find(instance) == allocated.end());
        // Store it in the allocated data structure
        allocated[instance] = range_index;
        return instance;
      }
      else
        return PhysicalInstance::NO_INST;
    }

    //--------------------------------------------------------------------------
    bool ConcretePool::contains_instance(PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      return (allocated.find(instance) != allocated.end());
    }

    //--------------------------------------------------------------------------
    RtEvent ConcretePool::escape_task_local_instance(
        PhysicalInstance instance, RtEvent safe_effects, size_t num_results,
        PhysicalInstance* results, LgEvent* unique_events,
        const Realm::InstanceLayoutGeneric** layouts, UniqueID creator_uid)
    //--------------------------------------------------------------------------
    {
      legion_assert(instance.exists());
      std::map<PhysicalInstance, unsigned>::iterator finder =
          allocated.find(instance);
      if (finder == allocated.end())
      {
        Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        err << "Detected a use-after-free case for instance " << instance.id
            << " in memory " << manager->get_name() << " by task "
            << implicit_context->get_task_name() << " (UID "
            << implicit_context->get_unique_id() << ") while trying to "
            << "escape the instance for an output region.";
        err.raise();
      }
      const Realm::InstanceLayoutGeneric* layout = instance.get_layout();
      if (layouts == nullptr)
      {
        // Handle the case where we are escaping a deferred buffer/value
        // to become a future and we're just going to reuse the same layout
        legion_assert(num_results == 1);
        layouts = &layout;
      }
      RtEvent result;
      if (finder->second == SENTINEL)
      {
        // This is a zero-sized range, so we can just make new empty instances
        std::vector<RtEvent> done_events;
        const Realm::InstanceLayoutGeneric* layout = instance.get_layout();
        legion_assert(layout->bytes_used == 0);
        for (unsigned idx = 0; idx < num_results; idx++)
        {
          Realm::ProfilingRequestSet requests;
          if (runtime->profiler != NULL)
          {
            if (!unique_events[idx].exists())
              Runtime::rename_event(unique_events[idx]);
            runtime->profiler->add_inst_request(
                requests, creator_uid, unique_events[idx]);
          }
          // Don't even bother checking if this succeeded as we know that
          // it is an empty sized layout so it should always succeed
          result = RtEvent(PhysicalInstance::create_instance(
              results[idx], manager->memory, *layout, requests));
          if (result.exists())
          {
            if (implicit_profiler != NULL)
              implicit_profiler->record_instance_ready(
                  result, unique_events[idx]);
            done_events.push_back(result);
            result = RtEvent::NO_RT_EVENT;
          }
        }
        result = Runtime::merge_events(done_events);
      }
      else  // Non-zero-sized instance so can escape like normal
        result = escape_range(
            finder->second, num_results, results, unique_events, layouts,
            creator_uid);
      // Remove the allocated instance
      allocated.erase(finder);
      // Destroy the external instance that we created
      instance.destroy(safe_effects);
      return result;
    }

    //--------------------------------------------------------------------------
    void ConcretePool::free_instance(
        PhysicalInstance instance, RtEvent precondition, LgEvent unique_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(instance.exists());
      std::map<PhysicalInstance, unsigned>::iterator finder =
          allocated.find(instance);
      if (finder == allocated.end())
      {
        Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        err << "Detected a duplicate delete for instance " << instance.id
            << " in memory " << manager->get_name() << " by task "
            << implicit_context->get_task_name() << " (UID "
            << implicit_context->get_unique_id() << "). This most likely "
            << "means that you called 'destroy' on your deferred buffer "
            << " or deferred value twice which is illegal.";
        err.raise();
      }
      legion_assert(finder != allocated.end());
      if (released)
      {
        // If we're released then we can free the backing instance
        // immediately as we know that it is only for this instance
        const Range& range = ranges[finder->second];
        // Free up the back instance since we know that we're not
        // going to bother recycling this memory since we're released
        std::map<PhysicalInstance, std::pair<RtEvent, LgEvent>>::iterator
            backing_finder = backing_instances.find(range.instance);
        legion_assert(backing_finder != backing_instances.end());
        backing_finder->first.destroy(
            Runtime::merge_events(backing_finder->second.first, precondition));
        backing_instances.erase(backing_finder);
      }
      else if (finder->second != SENTINEL)
      {
        if (precondition.exists() && !precondition.has_triggered())
          pending_frees.emplace(std::make_pair(finder->second, precondition));
        else
          deallocate(finder->second);
      }
      allocated.erase(finder);
      // Finally destroy the external Realm instance that we made
      instance.destroy(precondition);
    }

    //--------------------------------------------------------------------------
    unsigned ConcretePool::allocate(
        size_t size, size_t alignment, uintptr_t& alloc_first)
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      // Make sure we've inserted the initial range if from the backing
      // instance if we haven't already done that
      if (!ranges_initialized)
      {
        if (!backing_instances.empty())
        {
          legion_assert(backing_instances.size() == 1);
          std::map<PhysicalInstance, std::pair<RtEvent, LgEvent>>::
              const_iterator it = backing_instances.begin();
          legion_assert(it->second.first.has_triggered());
          // If we're on the local node for the instance then set up the ranges
          const void* base = it->first.pointer_untyped(0, 0);
          if (base != NULL)
          {
            // Initialize the first range with the full allocated space
            Range& r = ranges.emplace_back(Range());
            r.first = reinterpret_cast<uintptr_t>(base);
            r.last = r.first + limit;
            r.prev = r.next = r.prev_free = r.next_free = SENTINEL;
            r.instance = it->first;
            const unsigned log2_size = floor_log2(limit);
            size_based_free_lists.resize(log2_size + 1, SENTINEL);
            size_based_free_lists[log2_size] = 0;
          }
        }
        ranges_initialized = true;
      }
      bool try_again = true;
      while (try_again)
      {
        // Walk the free lists from the smallest one big enough to have holes
        // up to the one with the largest holes to find the smallest hole
        // that we can use for this allocation
        for (unsigned idx = floor_log2(size);
             idx < size_based_free_lists.size(); idx++)
        {
          unsigned index = size_based_free_lists[idx];
          while (index != SENTINEL)
          {
            Range* r = &ranges[index];

            size_t offset = 0;
            if (alignment > 0)
            {
              size_t remainder = r->first % alignment;
              if (remainder > 0)
                offset = alignment - remainder;
            }
            // do we have enough space?
            if ((r->last - r->first) < (size + offset))
            {
              // No, keep going
              index = r->next_free;
              continue;
            }
            // We have enough space
            // Remove this from the current size free list
            remove_from_free_list(index, *r);
            // but we we may to chop things up to make the exact range
            alloc_first = r->first + offset;
            uintptr_t alloc_last = alloc_first + size;
            // do we need to carve off a new (free) block before us?
            if (offset > 0)
            {
              unsigned new_index =
                  alloc_range(r->first, alloc_first, r->instance);
              Range& new_prev = ranges[new_index];
              r = &ranges[index];  // alloc may have moved this!
              r->first = alloc_first;
              // insert into all-block dllist
              new_prev.prev = r->prev;
              new_prev.next = index;
              if (r->prev != SENTINEL)
                ranges[r->prev].next = new_index;
              r->prev = new_index;
              // Insert into the free list of the appropriate size
              add_to_free_list(new_index, new_prev);
            }
            // see if we have leftover space and need to make a new range
            // to represent the remainder
            if (alloc_last != r->last)
            {
              // case 2 - leftover at end - put in new range
              unsigned after_index =
                  alloc_range(alloc_last, r->last, r->instance);
              Range& r_after = ranges[after_index];
              r = &ranges[index];  // alloc may have moved this!
              r->last = alloc_last;
              // r_after goes after r in all block list
              r_after.prev = index;
              r_after.next = r->next;
              r->next = after_index;
              if (r_after.next != SENTINEL)
                ranges[r_after.next].prev = after_index;
              // Put r_after in the free list of the right size
              add_to_free_list(after_index, r_after);
            }
            // tie this off because we use it to detect allocated-ness
            r->prev_free = r->next_free = index;
            // Decrement the number of free bytes available
            legion_assert(size <= remaining_bytes);
            remaining_bytes -= size;
            return index;
          }
        }
        // If we made it here we couldn't find a hole, see if there
        // are any pending frees that we can wait for to free up more
        // space to use for doing the allocation
        try_again = false;
        unsigned smallest_index = ranges.size();
        size_t smallest_hole = std::numeric_limits<size_t>::max();
        // See if any of the pending frees are done and can be deallocated
        // and that will allow us to try again
        for (std::map<unsigned, RtEvent>::iterator it = pending_frees.begin();
             it != pending_frees.end();
             /*nothing*/)
        {
          if (it->second.has_triggered())
          {
            try_again = true;
            deallocate(it->first);
            std::map<unsigned, RtEvent>::iterator delete_it = it++;
            pending_frees.erase(delete_it);
          }
          else
          {
            // See if this has enough space for the allocation
            const Range& range = ranges[it->first];
            const size_t hole_size = range.last - range.first;
            if ((size <= hole_size) && (hole_size < smallest_hole))
            {
              smallest_hole = hole_size;
              smallest_index = it->first;
            }
            it++;
          }
        }
        if (!try_again)
        {
          if (smallest_index < ranges.size())
          {
            // Wait for the smallest hole to be ready
            try_again = true;
            std::map<unsigned, RtEvent>::iterator finder =
                pending_frees.find(smallest_index);
            legion_assert(finder != pending_frees.end());
            deallocate(finder->first);
            finder->second.wait();
            pending_frees.erase(finder);
          }
          else if (!pending_frees.empty())
          {
            // Wait for all the tiny holes to be ready and then try again
            try_again = true;
            std::vector<RtEvent> done_events;
            done_events.reserve(pending_frees.size());
            for (const std::pair<const unsigned, RtEvent>& entry :
                 pending_frees)
            {
              deallocate(entry.first);
              done_events.emplace_back(entry.second);
            }
            pending_frees.clear();
            Runtime::merge_events(done_events).wait();
          }
        }
      }
      // Failed to perform the allocation
      return SENTINEL;
    }

    //--------------------------------------------------------------------------
    void ConcretePool::deallocate(unsigned index)
    //--------------------------------------------------------------------------
    {
      Range& r = ranges[index];
      // Add these bytes back into the available pool
      remaining_bytes += (r.last - r.first);
      legion_assert(remaining_bytes <= limit);
      // See if the previous range is free so we can merge with it
      const unsigned pf_idx = r.prev;
      const bool merge_prev =
          (pf_idx != SENTINEL) && (ranges[pf_idx].prev_free != pf_idx);
      // See if the next range is free so we can merge with it
      const unsigned nf_idx = r.next;
      const bool merge_next =
          (nf_idx != SENTINEL) && (ranges[nf_idx].next_free != nf_idx);

      // four cases - ordered to match the allocation cases
      if (!merge_next)
      {
        if (!merge_prev)
        {
          // case 1 - no merging (exact match)
          // just add ourselves to the free list
          add_to_free_list(index, r);
        }
        else
        {
          // case 2 - merge before
          // merge ourselves into the range before
          Range& r_before = ranges[pf_idx];
          grow_hole(pf_idx, r_before, r.last, false /*before*/);
          r_before.next = r.next;
          if (r.next != SENTINEL)
            ranges[r.next].prev = pf_idx;
          free_range(index);
        }
      }
      else
      {
        if (!merge_prev)
        {
          // case 3 - merge after
          // merge ourselves into the range after
          Range& r_after = ranges[nf_idx];
          grow_hole(nf_idx, r_after, r.first, true /*before*/);
          r_after.prev = r.prev;
          if (r.prev != SENTINEL)
            ranges[r.prev].next = nf_idx;
          free_range(index);
        }
        else
        {
          // case 4 - merge both
          // merge both ourselves and range after into range before
          Range& r_before = ranges[pf_idx];
          Range& r_after = ranges[nf_idx];
          remove_from_free_list(nf_idx, r_after);
          grow_hole(pf_idx, r_before, r_after.last, false /*before*/);

          // adjust both normal list and free list
          r_before.next = r_after.next;
          if (r_after.next != SENTINEL)
            ranges[r_after.next].prev = pf_idx;
          free_range(index);
          free_range(nf_idx);
        }
      }
    }

    //--------------------------------------------------------------------------
    unsigned ConcretePool::alloc_range(
        uintptr_t first, uintptr_t last, PhysicalInstance instance)
    //--------------------------------------------------------------------------
    {
      legion_assert(first < last);  // should not be allocating empty ranges
      // find/make a free index in the range list for this range
      unsigned new_idx;
      if (first_unused_range != SENTINEL)
      {
        new_idx = first_unused_range;
        first_unused_range = ranges[new_idx].next;
      }
      else
      {
        new_idx = ranges.size();
        ranges.resize(new_idx + 1);
      }
      ranges[new_idx].first = first;
      ranges[new_idx].last = last;
      ranges[new_idx].instance = instance;
      return new_idx;
    }

    //--------------------------------------------------------------------------
    void ConcretePool::free_range(unsigned index)
    //--------------------------------------------------------------------------
    {
      ranges[index].next = first_unused_range;
      first_unused_range = index;
    }

    //--------------------------------------------------------------------------
    RtEvent ConcretePool::escape_range(
        unsigned index, size_t num_results, PhysicalInstance* results,
        LgEvent* unique_events, const Realm::InstanceLayoutGeneric** layouts,
        UniqueID creator_uid)
    //--------------------------------------------------------------------------
    {
      // Should have handled escaping detection earlier
      legion_assert(index < ranges.size());
      Range* range = &ranges[index];
      // We only need to update the ranges if we're not released because
      // releasing means we're never going to need to allocated again
      if (!released)
      {
        // First update the ranges
        // If there is any padding needed for alignment before the first
        // instance then we'll add that range to the free list
        unsigned prev_index = SENTINEL;
        unsigned next_index = SENTINEL;
        uintptr_t offset = range->first;
        if (layouts[0]->alignment_reqd > 0)
        {
          size_t remainder = offset % layouts[0]->alignment_reqd;
          if (remainder > 0)
          {
            offset += (layouts[0]->alignment_reqd - remainder);
            prev_index = alloc_range(range->first, offset, range->instance);
            // Update the original range to start later
            range = &ranges[index];  // vector might have resized
            range->first = offset;
            // Add the new previous range into the linked list
            Range& prev_range = ranges[prev_index];
            prev_range.prev = range->prev;
            prev_range.next = index;
            range->prev = prev_index;
            if (prev_range.prev != SENTINEL)
              ranges[prev_range.prev].next = prev_index;
            // Deallocate the previous range
            deallocate(prev_index);
          }
        }
        offset += layouts[0]->bytes_used;
        // Note this throws away padding bytes between instances, but those
        // were unlikely to matter to anyone anyway since it's unlikely they'll
        // be able to be used for anything
        for (unsigned idx = 1; idx < num_results; idx++)
        {
          if (layouts[idx]->alignment_reqd > 0)
          {
            size_t remainder = offset % layouts[idx]->alignment_reqd;
            if (remainder > 0)
              offset += (layouts[idx]->alignment_reqd - remainder);
          }
          offset += layouts[idx]->bytes_used;
        }
        // Do the same thing with any leftover bytes at the end of the range
        // after considering how the new instances will be laid out
        if (offset < range->last)
        {
          next_index = alloc_range(offset, range->last, range->instance);
          // Update the original range to start later
          range = &ranges[index];  // vector might have resized
          range->last = offset;
          // Add the new next range into the linked list
          Range& next_range = ranges[next_index];
          next_range.prev = index;
          next_range.next = range->next;
          range->next = next_index;
          if (next_range.next != SENTINEL)
            ranges[next_range.next].prev = next_index;
          // Deallocate the next range
          deallocate(next_index);
        }
      }
      // Find the first and last range with the same backing instance
      unsigned current = range->prev;
      unsigned prev_index = index;
      while (current != SENTINEL)
      {
        const Range& current_range = ranges[current];
        if (current_range.instance != range->instance)
          break;
        prev_index = current;
        current = current_range.prev;
      }
      current = range->next;
      unsigned next_index = index;
      while (current != SENTINEL)
      {
        const Range& current_range = ranges[current];
        if (current_range.instance != range->instance)
          break;
        next_index = current;
        current = current_range.next;
      }
      // Perform instance redistricting to create all the new instances
      std::map<PhysicalInstance, std::pair<RtEvent, LgEvent>>::iterator finder =
          backing_instances.find(range->instance);
      legion_assert(finder != backing_instances.end());
      RtEvent result;
#ifdef LEGION_DEBUG
      std::vector<MemoryManager::TaskLocalInstanceAllocator> allocators;
      allocators.reserve(num_results + 2);
      std::vector<ProfilingResponseBase> bases;
      bases.reserve(num_results + 2);
#endif
      if ((prev_index != index) || (index != next_index))
      {
        // Shouldn't be here if we're released because each allocated range
        // should have exactly one backing instance
        legion_assert(!released);
        std::vector<LgEvent> extra_unique_events;
        std::vector<const Realm::InstanceLayoutGeneric*> extra_layouts;
        if (prev_index != index)
        {
          const Range& prev_range = ranges[prev_index];
          size_t size = range->first - prev_range.first;
          size_t offset = prev_range.first - ranges.front().first;
          extra_layouts.emplace_back(
              create_layout(size, 1 /*alignment*/, offset));
          if (runtime->profiler != nullptr)
          {
            extra_unique_events.emplace_back(LgEvent::NO_LG_EVENT);
            Runtime::rename_event(extra_unique_events.back());
          }
          else
            extra_unique_events.emplace_back(LgEvent::NO_LG_EVENT);
        }
        for (unsigned idx = 0; idx < num_results; idx++)
        {
          extra_layouts.emplace_back(layouts[idx]);
          if ((runtime->profiler != nullptr) && !unique_events[idx].exists())
            Runtime::rename_event(unique_events[idx]);
          extra_unique_events.emplace_back(unique_events[idx]);
        }
        if (next_index != index)
        {
          const Range& next_range = ranges[next_index];
          size_t size = next_range.last - range->last;
          size_t offset = range->last - ranges.front().first;
          extra_layouts.emplace_back(
              create_layout(size, 1 /*alignment*/, offset));
          if (runtime->profiler != nullptr)
          {
            extra_unique_events.emplace_back(LgEvent::NO_LG_EVENT);
            Runtime::rename_event(extra_unique_events.back());
          }
          else
            extra_unique_events.emplace_back(LgEvent::NO_LG_EVENT);
        }
        std::vector<Realm::ProfilingRequestSet> requests(extra_layouts.size());
        for (unsigned idx = 0; idx < requests.size(); idx++)
        {
          if (runtime->profiler != nullptr)
            runtime->profiler->add_inst_request(
                requests[idx], creator_uid, extra_unique_events[idx]);
#ifdef LEGION_DEBUG
          allocators.emplace_back(MemoryManager::TaskLocalInstanceAllocator(
              extra_unique_events[idx]));
          bases.emplace_back(
              ProfilingResponseBase(&allocators[idx], creator_uid, false));
          Realm::ProfilingRequest& req = requests[idx].add_request(
              runtime->find_local_group(), LG_LEGION_PROFILING_ID, &bases[idx],
              sizeof(bases[idx]), LG_RESOURCE_PRIORITY);
          req.add_measurement<
              Realm::ProfilingMeasurements::InstanceAllocResult>();
#endif
        }
        std::vector<PhysicalInstance> extra_instances(extra_layouts.size());
        result = RtEvent(range->instance.redistrict(
            &extra_instances.front(), &extra_layouts.front(),
            extra_layouts.size(), &requests.front(), finder->second.first));
        // Report the profiling info if necessary
        if (result.exists() && (implicit_profiler != NULL))
        {
          for (unsigned idx = 0; idx < extra_unique_events.size(); idx++)
            implicit_profiler->record_instance_redistrict(
                result, finder->second.second, extra_unique_events[idx],
                finder->second.first);
        }
        if (prev_index != index)
        {
          // Update all the previous ranges with the new backing instance
          backing_instances.emplace(std::make_pair(
              extra_instances.front(),
              std::make_pair(result, extra_unique_events.front())));
          while (prev_index != index)
          {
            Range& prev_range = ranges[prev_index];
            prev_range.instance = extra_instances.front();
            prev_index = prev_range.next;
          }
          delete extra_layouts.front();
          // Copy over the names of the newly created instances
          for (unsigned idx = 0; idx < num_results; idx++)
            results[idx] = extra_instances[idx + 1];
        }
        else
        {
          // Copy over the names of the newly created instances
          for (unsigned idx = 0; idx < num_results; idx++)
            results[idx] = extra_instances[idx];
        }
        if (next_index != index)
        {
          // Update all the next ranges with the new backing instance
          backing_instances.emplace(std::make_pair(
              extra_instances.back(),
              std::make_pair(result, extra_unique_events.back())));
          while (next_index != index)
          {
            Range& next_range = ranges[next_index];
            next_range.instance = extra_instances.back();
            next_index = next_range.prev;
          }
          delete extra_layouts.back();
        }
      }
      else
      {
        // We're literally redistricting exactly the instance we want
        // with no space left over at the beginning or the end
        std::vector<Realm::ProfilingRequestSet> requests(num_results);
        for (unsigned idx = 0; idx < num_results; idx++)
        {
          if (runtime->profiler != nullptr)
          {
            if (!unique_events[idx].exists())
              Runtime::rename_event(unique_events[idx]);
            runtime->profiler->add_inst_request(
                requests[idx], creator_uid, unique_events[idx]);
          }
#ifdef LEGION_DEBUG
          allocators.emplace_back(
              MemoryManager::TaskLocalInstanceAllocator(unique_events[idx]));
          bases.emplace_back(
              ProfilingResponseBase(&allocators[idx], creator_uid, false));
          Realm::ProfilingRequest& req = requests[idx].add_request(
              runtime->find_local_group(), LG_LEGION_PROFILING_ID, &bases[idx],
              sizeof(bases[idx]), LG_RESOURCE_PRIORITY);
          req.add_measurement<
              Realm::ProfilingMeasurements::InstanceAllocResult>();
#endif
        }
        result = RtEvent(range->instance.redistrict(
            results, layouts, num_results, &requests.front(),
            finder->second.first));
        // Report the profiling info if necessary
        if (result.exists() && (implicit_profiler != NULL))
        {
          for (unsigned idx = 0; idx < num_results; idx++)
            implicit_profiler->record_instance_redistrict(
                result, finder->second.second, unique_events[idx],
                finder->second.first);
        }
        // Nothing to update here since we're just going to leave the
        // existing range in place as though it is allocated and it
        // will never be deallocated
      }
#ifdef LEGION_DEBUG
      for (unsigned idx = 0; idx < allocators.size(); idx++)
        legion_no_skip_assert(allocators[idx].succeeded());
#endif
      // Mark the current range as having escaped
      range->instance = PhysicalInstance::NO_INST;
      // Remove the backing instance from the map
      backing_instances.erase(finder);
      return result;
    }

    //--------------------------------------------------------------------------
    bool ConcretePool::is_released(void) const
    //--------------------------------------------------------------------------
    {
      return released;
    }

    //--------------------------------------------------------------------------
    void ConcretePool::release_pool(UniqueID creator)
    //--------------------------------------------------------------------------
    {
      if (!released)
      {
        released = true;
        // Release all the pending frees
        for (const std::pair<const unsigned, RtEvent>& entry : pending_frees)
          deallocate(entry.first);
        pending_frees.clear();
        // Iterate over all the existing allocations and escape their ranges
        // and replace their backing stores with the escaped instances
        std::map<PhysicalInstance, std::pair<RtEvent, LgEvent>>
            new_backing_instances;
        for (const std::pair<const PhysicalInstance, unsigned>& entry :
             allocated)
        {
          // If this is a zero-sized instance we can skip it as it is not
          // backed by any memory so there is nothing to escape
          if (entry.second == SENTINEL)
            continue;
          LgEvent unique_event;
          PhysicalInstance backing_instance;
          const Realm::InstanceLayoutGeneric* layout = entry.first.get_layout();
          const RtEvent ready = escape_range(
              entry.second, 1 /*num results*/, &backing_instance, &unique_event,
              &layout, creator);
          new_backing_instances.emplace(std::make_pair(
              backing_instance, std::make_pair(ready, unique_event)));
          Range& range = ranges[entry.second];
          range.instance = backing_instance;
        }
        // Then go through and delete the remaining backing stores
        for (const std::pair<
                 const PhysicalInstance, std::pair<RtEvent, LgEvent>>& entry :
             backing_instances)
        {
          manager->update_remaining_capacity(
              entry.first.get_layout()->bytes_used);
          entry.first.destroy(entry.second.first);
        }
        // Now the only remaining backing instances are the ones we made
        backing_instances.swap(new_backing_instances);
      }
    }

    //--------------------------------------------------------------------------
    void ConcretePool::finalize_pool(RtEvent done)
    //--------------------------------------------------------------------------
    {
      legion_assert(!finalized);
      manager->release_concrete_pool(done, backing_instances);
      finalized = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ unsigned ConcretePool::floor_log2(uint64_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      // Round down to the nearest power of two to figure out which range
      // to put it in using DeBruijin algorithm to compute integer log2
      // Taken from Hacker's Delight
      static const unsigned tab64[64] = {
          63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48, 27, 54, 33, 42, 3,
          61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4,
          62, 57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21,
          56, 45, 25, 31, 35, 16, 9,  12, 44, 24, 15, 8,  23, 7,  6,  5};
      uint64_t value = size;
      value |= value >> 1;
      value |= value >> 2;
      value |= value >> 4;
      value |= value >> 8;
      value |= value >> 16;
      value |= value >> 32;
      return tab64
          [((uint64_t)((value - (value >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
    }

    //--------------------------------------------------------------------------
    void ConcretePool::add_to_free_list(unsigned index, Range& range)
    //--------------------------------------------------------------------------
    {
      const size_t size = range.last - range.first;
      const unsigned log2_size = floor_log2(size);
      if (size_based_free_lists.size() <= log2_size)
        size_based_free_lists.resize(log2_size + 1, SENTINEL);
      // Insert the range into the list such that it maintains a sorted
      // list from smallest to largest
      unsigned prev = SENTINEL;
      unsigned next = size_based_free_lists[log2_size];
      while (next != SENTINEL)
      {
        Range& next_range = ranges[next];
        const size_t next_size = next_range.last - next_range.first;
        if (size <= next_size)
        {
          // We can insert this here and we're done
          range.prev_free = next_range.prev_free;
          if (range.prev_free == SENTINEL)
            size_based_free_lists[log2_size] = index;
          else
            ranges[range.prev_free].next_free = index;
          range.next_free = next;
          next_range.prev_free = index;
          return;
        }
        // Step to the next entry
        prev = next;
        next = next_range.next_free;
      }
      // If we get here then we're adding ourselves to the end of the list
      range.prev_free = prev;
      range.next_free = SENTINEL;
      if (prev == SENTINEL)
        size_based_free_lists[log2_size] = index;
      else
        ranges[prev].next_free = index;
    }

    //--------------------------------------------------------------------------
    void ConcretePool::remove_from_free_list(unsigned index, Range& range)
    //--------------------------------------------------------------------------
    {
      if (range.prev_free != SENTINEL)
      {
        if (range.next_free != SENTINEL)
        {
          // Remove an item in the middle of the list
          ranges[range.prev_free].next_free = range.next_free;
          ranges[range.next_free].prev_free = range.prev_free;
        }
        else  // last item in the list can just be removed
          ranges[range.prev_free].next_free = SENTINEL;
      }
      else
      {
        // We're the first item in the list
        const size_t size = range.last - range.first;
        const unsigned log2_size = floor_log2(size);
        legion_assert(log2_size < size_based_free_lists.size());
        legion_assert(size_based_free_lists[log2_size] == index);
        if (range.next_free != SENTINEL)
          ranges[range.next_free].prev_free = SENTINEL;
        size_based_free_lists[log2_size] = range.next_free;
      }
    }

    //--------------------------------------------------------------------------
    void ConcretePool::grow_hole(
        unsigned index, Range& range, uintptr_t bound, bool before)
    //--------------------------------------------------------------------------
    {
      unsigned old_bin = floor_log2(range.last - range.first);
      size_t new_size =
          (before ? range.last : bound) - (before ? bound : range.first);
      unsigned new_bin = floor_log2(new_size);
      if (old_bin == new_bin)
      {
        if (before)
          range.first = bound;
        else
          range.last = bound;
        // Bubble ourselves up the free list to keep things sorted
        while (range.next_free != SENTINEL)
        {
          unsigned next_index = range.next_free;
          Range& next_range = ranges[next_index];
          size_t next_size = next_range.last - next_range.first;
          if (new_size <= next_size)
            break;
          // Swap places with the next range
          if (range.prev_free != SENTINEL)
            ranges[range.prev_free].next_free = next_index;
          if (next_range.next_free != SENTINEL)
            ranges[next_range.next_free].prev_free = index;
          next_range.prev_free = range.prev_free;
          range.next_free = next_range.next_free;
          range.prev_free = next_index;
          next_range.next_free = index;
          // Make sure to handle the case where we're the first entry
          // in the free lists
          if (size_based_free_lists[old_bin] == index)
            size_based_free_lists[old_bin] = next_index;
        }
      }
      else
      {
        remove_from_free_list(index, range);
        if (before)
          range.first = bound;
        else
          range.last = bound;
        add_to_free_list(index, range);
      }
    }

    //--------------------------------------------------------------------------
    void ConcretePool::serialize(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(!finalized);
      rez.serialize(manager->memory);
      rez.serialize<bool>(true);  // bounded;
      rez.serialize(remaining_bytes);
      rez.serialize(max_alignment);
      if (!backing_instances.empty())
      {
        legion_assert(backing_instances.size() == 1);
        std::map<PhysicalInstance, std::pair<RtEvent, LgEvent>>::const_iterator
            finder = backing_instances.begin();
        rez.serialize(finder->first);
        rez.serialize(finder->second.first);
        rez.serialize(finder->second.second);
      }
      else
      {
        rez.serialize(PhysicalInstance::NO_INST);
        rez.serialize(RtEvent::NO_RT_EVENT);
        rez.serialize(LgEvent::NO_LG_EVENT);
      }
      finalized = true;  // was sent remotely which is the same as finalized
    }

    //--------------------------------------------------------------------------
    UnboundPool::UnboundPool(
        MemoryManager* man, UnboundPoolScope s, TaskTreeCoordinates& coords,
        size_t max_bytes)
      : MemoryPool(std::numeric_limits<size_t>::max()), manager(man),
        max_freed_bytes(max_bytes), freed_bytes(0), scope(s), released(false),
        finalized(false)
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      legion_assert(!coords.empty());
      coordinates.swap(coords);
    }

    //--------------------------------------------------------------------------
    UnboundPool::~UnboundPool(void)
    //--------------------------------------------------------------------------
    {
      if (!finalized)
        finalize_pool(RtEvent::NO_RT_EVENT);
    }

    //--------------------------------------------------------------------------
    ApEvent UnboundPool::get_ready_event(void) const
    //--------------------------------------------------------------------------
    {
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    size_t UnboundPool::query_memory_limit(void)
    //--------------------------------------------------------------------------
    {
      return manager->capacity;
    }

    //--------------------------------------------------------------------------
    size_t UnboundPool::query_available_memory(void)
    //--------------------------------------------------------------------------
    {
      if (released)
        return 0;
      else
        return manager->query_available_memory();
    }

    //--------------------------------------------------------------------------
    PoolBounds UnboundPool::get_bounds(void) const
    //--------------------------------------------------------------------------
    {
      return PoolBounds(scope);
    }

    //--------------------------------------------------------------------------
    void UnboundPool::capture_task_instances(
        const std::map<PhysicalManager*, unsigned>& instances)
    //--------------------------------------------------------------------------
    {
      const Memory memory = manager->memory;
      // Capture valid references on any instances in the same memory as
      // this unbounded pool to prevent them from being deferred deleted
      // to satisfy allocations done by this pool
      for (const std::pair<PhysicalManager* const, unsigned>& entry : instances)
      {
        if (memory != entry.first->get_memory())
          continue;
        entry.first->add_base_valid_ref(UNBOUNDED_POOL_REF);
        captured_instances.push_back(entry.first);
      }
    }

    //--------------------------------------------------------------------------
    FutureInstance* UnboundPool::allocate_future(
        UniqueID creator_uid, size_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      legion_assert(!released);
      // Check to see if we have a local freed instance that is big enough
      // to use for this instance
      if (!freed_instances.empty())
      {
        size_t previous_size = 0;
        RtEvent previous_done;
        LgEvent previous_unique;
        Realm::InstanceLayoutGeneric* layout = nullptr;
        PhysicalInstance previous = find_local_freed_hole(
            size, previous_size, previous_done, previous_unique);
        while (previous.exists())
        {
          // Redistrict the previously freed instance into a future instance
          if (layout == nullptr)
          {
            // Create the layout description for this instance
            const std::vector<Realm::FieldID> fids(1, 0 /*field id*/);
            const std::vector<size_t> sizes(1, size);
            const int dim_order[1] = {0};
            const Realm::Point<1, coord_t> zero(0);
            const Realm::InstanceLayoutConstraints constraints(fids, sizes, 1);
            const Realm::IndexSpace<1, coord_t> rect_space(
                Realm::Rect<1, coord_t>(zero, zero));
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout<
                1, coord_t>(rect_space, constraints, dim_order);
            layout->alignment_reqd = manager->compute_future_alignment(size);
          }
          LgEvent unique_event;
          if ((spy_logging_level > NO_SPY_LOGGING) ||
              (runtime->profiler != nullptr))
            Runtime::rename_event(unique_event);
          // Try to do the redistrict the previous instance into a new one
          MemoryManager::TaskLocalInstanceAllocator allocator(unique_event);
          ProfilingResponseBase base(&allocator, creator_uid, false);
          Realm::ProfilingRequestSet requests;
          Realm::ProfilingRequest& req = requests.add_request(
              runtime->find_local_group(), LG_LEGION_PROFILING_ID, &base,
              sizeof(base), LG_RESOURCE_PRIORITY);
          req.add_measurement<
              Realm::ProfilingMeasurements::InstanceAllocResult>();
          if (runtime->profiler != nullptr)
            runtime->profiler->add_inst_request(
                requests, creator_uid, unique_event);
          PhysicalInstance instance;
          RtEvent use_event(
              previous.redistrict(instance, layout, requests, previous_done));
          if (allocator.succeeded())
          {
            if (use_event.exists() && (implicit_profiler != NULL))
              implicit_profiler->record_instance_redistrict(
                  use_event, previous_unique, unique_event, previous_done);
            size_t bytes_used = instance.get_layout()->bytes_used;
            legion_assert(bytes_used <= previous_size);
            if (bytes_used < previous_size)
              manager->update_remaining_capacity(previous_size - bytes_used);
            delete layout;
            return new FutureInstance(
                nullptr /*data*/, size, false /*external*/,
                true /*own allocation*/, unique_event, instance,
                Processor::NO_PROC, use_event);
          }
          else
            manager->update_remaining_capacity(previous_size);
          previous = find_local_freed_hole(
              size, previous_size, previous_done, previous_unique);
        }
        if (layout != nullptr)
          delete layout;
        // Try to do the allocation normal way
        FutureInstance* instance = manager->create_future_instance(
            creator_uid, coordinates, size,
            nullptr /*safe_for_unbounded_pools*/);
        if (instance != nullptr)
          return instance;
        // If it doesn't work, free all our freed instances (which are all
        // smaller than the size or we would have found a hole to use) and
        // then try again
        for (const std::pair<const size_t, std::list<FreedInstance>>&
                 fit_entry : freed_instances)
          for (const FreedInstance& freed_instance : fit_entry.second)
            manager->free_task_local_instance(
                freed_instance.instance, freed_instance.precondition);
        freed_instances.clear();
        freed_bytes = 0;
      }
      // This is safe for unbounded pools because we are the unbounded pool :)
      return manager->create_future_instance(
          creator_uid, coordinates, size, nullptr /*safe_for_unbounded_pools*/);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance UnboundPool::allocate_instance(
        UniqueID creator_uid, LgEvent unique_event,
        const Realm::InstanceLayoutGeneric& layout, RtEvent& use_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(!released);
      if (layout.bytes_used == 0)
      {
        // Special case for empty instances, we don't need to talk to anyone
        Realm::InstanceLayoutGeneric* empty_layout = create_layout(0, 1);
        PhysicalInstance instance;
        const Realm::ProfilingRequestSet empty_requests;
        use_event = RtEvent(PhysicalInstance::create_instance(
            instance, manager->memory, *empty_layout, empty_requests));
        delete empty_layout;
        return instance;
      }
      if (!freed_instances.empty())
      {
        size_t previous_size = 0;
        RtEvent previous_done;
        LgEvent previous_unique;
        PhysicalInstance previous = find_local_freed_hole(
            layout.bytes_used, previous_size, previous_done, previous_unique);
        while (previous.exists())
        {
          // Redistrict the previously freed instance into a new instance
          MemoryManager::TaskLocalInstanceAllocator allocator(unique_event);
          ProfilingResponseBase base(&allocator, creator_uid, false);
          Realm::ProfilingRequestSet requests;
          Realm::ProfilingRequest& req = requests.add_request(
              runtime->find_local_group(), LG_LEGION_PROFILING_ID, &base,
              sizeof(base), LG_RESOURCE_PRIORITY);
          req.add_measurement<
              Realm::ProfilingMeasurements::InstanceAllocResult>();
          if (runtime->profiler != nullptr)
            runtime->profiler->add_inst_request(
                requests, creator_uid, unique_event);
          PhysicalInstance instance;
          use_event = RtEvent(
              previous.redistrict(instance, &layout, requests, previous_done));
          if (allocator.succeeded())
          {
            if (use_event.exists() && (implicit_profiler != NULL))
              implicit_profiler->record_instance_redistrict(
                  use_event, previous_unique, unique_event, previous_done);
            size_t bytes_used = instance.get_layout()->bytes_used;
            legion_assert(bytes_used <= previous_size);
            if (bytes_used < previous_size)
              manager->update_remaining_capacity(previous_size - bytes_used);
            return instance;
          }
          else
            manager->update_remaining_capacity(previous_size);
          previous = find_local_freed_hole(
              layout.bytes_used, previous_size, previous_done, previous_unique);
        }
        // Try to do the allocation the normal way
        PhysicalInstance instance = manager->create_task_local_instance(
            creator_uid, coordinates, unique_event, layout, use_event,
            nullptr /*safe_for_unbound_pools*/);
        if (instance.exists())
          return instance;
        // If it doesn't work, free all our freed instances (which are all
        // smaller than the size or we would have found a hole to use) and
        // then try again
        for (const std::pair<const size_t, std::list<FreedInstance>>&
                 fit_entry : freed_instances)
          for (const FreedInstance& freed_instance : fit_entry.second)
            manager->free_task_local_instance(
                freed_instance.instance, freed_instance.precondition);
        freed_instances.clear();
        freed_bytes = 0;
      }
      // We are the unbound pool so it is always safe for us
      return manager->create_task_local_instance(
          creator_uid, coordinates, unique_event, layout, use_event,
          nullptr /*safe_for_unbound_pools*/);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance UnboundPool::find_local_freed_hole(
        size_t size, size_t& previous_size, RtEvent& previous_done,
        LgEvent& previous_unique)
    //--------------------------------------------------------------------------
    {
      for (std::map<size_t, std::list<FreedInstance>>::iterator sit =
               freed_instances.lower_bound(size);
           sit != freed_instances.end(); sit++)
      {
        legion_assert(!sit->second.empty());
        legion_assert(sit->first <= freed_bytes);
        for (std::list<FreedInstance>::iterator it = sit->second.begin();
             it != sit->second.end(); it++)
        {
          // If the event hasn't triggered then skip it
          if (!it->precondition.has_triggered())
            continue;
          PhysicalInstance result = it->instance;
          freed_bytes -= sit->first;
          previous_size = sit->first;
          previous_done = it->precondition;
          previous_unique = it->unique_event;
          sit->second.erase(it);
          if (sit->second.empty())
            freed_instances.erase(sit);
          return result;
        }
      }
      return PhysicalInstance::NO_INST;
    }

    //--------------------------------------------------------------------------
    bool UnboundPool::contains_instance(PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      // We don't track any instances we made
      return false;
    }

    //--------------------------------------------------------------------------
    RtEvent UnboundPool::escape_task_local_instance(
        PhysicalInstance instance, RtEvent safe_effects, size_t num_results,
        PhysicalInstance* result, LgEvent* unique_events,
        const Realm::InstanceLayoutGeneric** layouts, UniqueID creator_uid)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void UnboundPool::free_instance(
        PhysicalInstance instance, RtEvent precondition, LgEvent unique_event)
    //--------------------------------------------------------------------------
    {
      if (!released && (max_freed_bytes > 0))
      {
        // Add it to the freed instances
        const size_t size = instance.get_layout()->bytes_used;
        // Special case for empty instances
        if (size == 0)
        {
          instance.destroy(precondition);
          return;
        }
        freed_instances[size].emplace_back(
            FreedInstance{instance, precondition, unique_event});
        freed_bytes += size;
        if (max_freed_bytes < freed_bytes)
        {
          // Start releasing the smallest instances to reduce fragmentation
          // until we're back under the limit of bytes we can buffer
          while (!freed_instances.empty())
          {
            std::map<size_t, std::list<FreedInstance>>::iterator it =
                freed_instances.begin();
            while (!it->second.empty())
            {
              manager->free_task_local_instance(
                  it->second.back().instance, it->second.back().precondition);
              it->second.pop_back();
              freed_bytes -= it->first;
              if (freed_bytes <= max_freed_bytes)
              {
                if (it->second.empty())
                  freed_instances.erase(it);
                return;
              }
            }
            freed_instances.erase(it);
          }
        }
      }
      else  // Release it right away
        manager->free_task_local_instance(instance, precondition);
    }

    //--------------------------------------------------------------------------
    bool UnboundPool::is_released(void) const
    //--------------------------------------------------------------------------
    {
      return released;
    }

    //--------------------------------------------------------------------------
    void UnboundPool::release_pool(UniqueID creator)
    //--------------------------------------------------------------------------
    {
      if (!released)
      {
        for (const std::pair<const size_t, std::list<FreedInstance>>&
                 fit_entry : freed_instances)
          for (const FreedInstance& freed_instance : fit_entry.second)
            manager->free_task_local_instance(
                freed_instance.instance, freed_instance.precondition);
        for (PhysicalManager* const & physical_manager : captured_instances)
          if (physical_manager->remove_base_valid_ref(UNBOUNDED_POOL_REF))
            delete physical_manager;
        manager->release_unbound_pool();
        freed_instances.clear();
        captured_instances.clear();
        freed_bytes = 0;
        released = true;
      }
    }

    //--------------------------------------------------------------------------
    void UnboundPool::finalize_pool(RtEvent done)
    //--------------------------------------------------------------------------
    {
      release_pool(0);
    }

    //--------------------------------------------------------------------------
    void UnboundPool::serialize(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(!finalized);
      legion_assert(manager != nullptr);
      rez.serialize(manager->memory);
      rez.serialize<bool>(false);  // bounded;
      rez.serialize(max_freed_bytes);
      rez.serialize(scope);
      coordinates.serialize(rez);
      rez.serialize<size_t>(captured_instances.size());
      for (PhysicalManager* const & physical_manager : captured_instances)
        rez.serialize(physical_manager->did);
      for (PhysicalManager* const & physical_manager : captured_instances)
      {
        physical_manager->pack_valid_ref(rez);
        if (physical_manager->remove_base_valid_ref(UNBOUNDED_POOL_REF))
          delete physical_manager;
      }
      captured_instances.clear();
      finalized = true;  // Send remotely counts as being finalized
    }

    //--------------------------------------------------------------------------
    void UnboundPool::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_instances;
      derez.deserialize(num_instances);
      std::vector<RtEvent> ready_events;
      captured_instances.reserve(num_instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        captured_instances.push_back(
            runtime->find_or_request_instance_manager(did, ready));
        if (ready.exists())
          ready_events.push_back(ready);
      }
      if (!ready_events.empty())
        Runtime::merge_events(ready_events).wait();
      for (PhysicalManager* const & physical_manager : captured_instances)
      {
#ifdef LEGION_DEBUG
        // This is a work-around for an overzealous assertion in the
        // physical manager valid reference addition here. We know we
        // hold a packed reference, but it doesn't, so we have to make
        // this look safe for it
        legion_no_skip_assert(
            physical_manager->acquire_instance(UNBOUNDED_POOL_REF));
#else
        physical_manager->add_base_valid_ref(UNBOUNDED_POOL_REF);
#endif
        physical_manager->unpack_valid_ref(derez);
      }
    }

    /////////////////////////////////////////////////////////////
    // Memory Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MemoryManager::MemoryManager(Memory m)
      : memory(m), owner_space(m.address_space()),
        is_owner(is_owner_memory(m, runtime->address_space)),
        capacity(m.capacity()), remaining_capacity(capacity),
        outstanding_task_local_allocations(0),
        outstanding_unbounded_allocations(0),
        unbounded_pool_scope(LEGION_BOUNDED_POOL), collective_lamport_clock(0),
        ready_collective_tasks(0), outstanding_collective_tasks(0)
    //--------------------------------------------------------------------------
    {
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
      if ((memory.kind() == Memory::GPU_FB_MEM) ||
          (memory.kind() == Memory::GPU_MANAGED_MEM) ||
          (memory.kind() == Memory::GPU_DYNAMIC_MEM))
      {
        Machine::ProcessorQuery finder(runtime->machine);
        finder.best_affinity_to(memory);
        finder.only_kind(Processor::TOC_PROC);
        legion_assert(finder.count() > 0);
        local_gpu = finder.first();
      }
      else if (memory.kind() == Memory::Z_COPY_MEM)
      {
        Machine::ProcessorQuery finder(runtime->machine);
        finder.has_affinity_to(memory);
        finder.only_kind(Processor::TOC_PROC);
        legion_assert(finder.count() > 0);
        local_gpu = finder.first();
      }
#endif
    }

    //--------------------------------------------------------------------------
    MemoryManager::~MemoryManager(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MemoryManager::find_shutdown_preconditions(
        std::set<ApEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // We only need to check this on the owner node instances and
      // in fact it's only safe for us to do it on the owner node
      // instance because we only are guaranteed to have references
      // to the owner node objects
      if (!is_owner)
        return;
      std::vector<PhysicalManager*> to_check;
      {
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        for (TreeFieldInstances::const_iterator cit = current_instances.begin();
             cit != current_instances.end(); cit++)
          for (TreeInstances::const_iterator it = cit->second.begin();
               it != cit->second.end(); it++)
          {
            it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
            to_check.emplace_back(it->first);
          }
      }
      for (PhysicalManager* const & physical_manager : to_check)
      {
        physical_manager->find_shutdown_preconditions(preconditions);
        if (physical_manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
          delete physical_manager;
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      // Only need to do things if we are the owner memory
      if (!is_owner)
        return;
      // This is a kind of deletion so make sure it is ordered
      AutoLock c_lock(collection_lock);
      // This a collection so make sure we're ordered with other collections
      std::vector<PhysicalManager*> to_delete;
      {
        AutoLock m_lock(manager_lock);
        for (TreeFieldInstances::iterator cit = current_instances.begin();
             cit != current_instances.end(); cit++)
        {
          for (TreeInstances::iterator it = cit->second.begin();
               it != cit->second.end(); it++)
          {
            if (it->first->is_external_instance())
              continue;
            if ((it->second == LEGION_GC_NEVER_PRIORITY) &&
                it->first->is_owner())
            {
              it->first->remove_base_valid_ref(NEVER_GC_REF);
              it->second = 0;
            }
            bool already_collected = false;
            if (it->first->can_collect(already_collected))
            {
              it->first->add_base_gc_ref(MEMORY_MANAGER_REF);
              to_delete.emplace_back(it->first);
            }
            else if (already_collected)
              remove_collectable(it->second, it->first);
          }
        }
      }
      if (!to_delete.empty())
        check_instance_deletions(to_delete);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::check_instance_deletions(
        const std::vector<PhysicalManager*>& to_delete)
    //--------------------------------------------------------------------------
    {
      for (PhysicalManager* const & physical_manager : to_delete)
      {
        legion_assert(!physical_manager->is_external_instance());
        RtEvent deletion_done;
        physical_manager->collect(deletion_done);
        if (physical_manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
          delete physical_manager;
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::remove_collectable(
        GCPriority priority, PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      if (priority != LEGION_GC_NEVER_PRIORITY)
      {
        std::map<GCPriority, std::set<PhysicalManager*>>::iterator finder =
            collectable_instances.find(priority);
        if (finder != collectable_instances.end())
        {
          finder->second.erase(manager);
          if (finder->second.empty())
            collectable_instances.erase(finder);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::finalize(void)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
        return;
      // No need for the lock, no one should be doing anything at this point
      // The only instances that are left here are the ones that were not
      // collected since we already waited for any pending collections to
      // finish and their meta tasks to run to prune them out of the
      // current_instances data structure
      for (TreeFieldInstances::const_iterator cit = current_instances.begin();
           cit != current_instances.end(); cit++)
        for (TreeInstances::const_iterator it = cit->second.begin();
             it != cit->second.end(); it++)
          it->first->force_deletion();
      current_instances.clear();
#ifdef LEGION_MALLOC_INSTANCES
      for (const std::pair<const RtEvent, PhysicalInstance> > &it :
           pending_collectables)
        free_legion_instance(it.first, it.second);
      pending_collectables.clear();
#endif
    }

    //--------------------------------------------------------------------------
    MemoryManager::TreeFieldKey::TreeFieldKey(PhysicalManager* manager)
      : std::pair<RegionTreeID, uint64_t>(
            manager->tree_id, manager->layout->allocated_fields.get_hash_key())
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MemoryManager::register_remote_instance(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner);
      const TreeFieldKey key(manager);
      AutoLock m_lock(manager_lock);
      TreeInstances& insts = current_instances[key];
      legion_assert(insts.find(manager) == insts.end());
      insts[manager] = LEGION_GC_NEVER_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::unregister_remote_instance(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner);
      const TreeFieldKey key(manager);
      AutoLock m_lock(manager_lock);
      TreeFieldInstances::iterator finder = current_instances.find(key);
      legion_assert(finder != current_instances.end());
      legion_assert(finder->second.find(manager) != finder->second.end());
      finder->second.erase(manager);
      if (finder->second.empty())
        current_instances.erase(finder);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::unregister_deleted_instance(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      {
        const TreeFieldKey key(manager);
        AutoLock m_lock(manager_lock);
        TreeFieldInstances::iterator tree_finder = current_instances.find(key);
        legion_assert(tree_finder != current_instances.end());
        TreeInstances::iterator finder = tree_finder->second.find(manager);
        legion_assert(finder != tree_finder->second.end());
        remove_collectable(finder->second, finder->first);
        tree_finder->second.erase(finder);
        if (tree_finder->second.empty())
          current_instances.erase(tree_finder);
      }
      if (manager->is_external_instance())
      {
        if (manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
          delete manager;
      }
      else
      {
        if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
          delete manager;
      }
    }

    //--------------------------------------------------------------------------
    MemoryManager::ManagerResult::~ManagerResult(void)
    //--------------------------------------------------------------------------
    {
      if (manager != nullptr)
        manager->unpack_global_ref(lamport_clock);
    }

    //--------------------------------------------------------------------------
    MemoryManager::ManagerResult::ManagerResult(ManagerResult&& rhs)
      : manager(rhs.manager), lamport_clock(rhs.lamport_clock)
    //--------------------------------------------------------------------------
    {
      rhs.manager = nullptr;
      rhs.lamport_clock = 0;
    }

    //--------------------------------------------------------------------------
    MemoryManager::ManagerResult& MemoryManager::ManagerResult::operator=(
        ManagerResult&& rhs)
    //--------------------------------------------------------------------------
    {
      if (manager != nullptr)
        manager->unpack_global_ref(lamport_clock);
      manager = rhs.manager;
      lamport_clock = rhs.lamport_clock;
      rhs.manager = nullptr;
      rhs.lamport_clock = 0;
      return *this;
    }

    //--------------------------------------------------------------------------
    MemoryManager::ManagerResult::operator MappingInstance(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      return MappingInstance(manager);
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::create_physical_instance(
        const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        Processor processor, bool acquire, GCPriority priority,
        bool tight_bounds, LayoutConstraintKind* unsat_kind,
        unsigned* unsat_index, size_t* footprint,
        RtEvent* safe_for_unbounded_pools, UniqueID creator_id, bool remote)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // Not the owner, send a meessage to the owner to request the creation
        InstanceRequest rez;
        ManagerResult target;
        std::atomic<bool> success(false);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(CREATE_INSTANCE_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          coordinates.serialize(rez);
          constraints.serialize(rez);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_bounds);
          rez.serialize(unsat_kind);
          rez.serialize(unsat_index);
          rez.serialize(footprint);
          rez.serialize(safe_for_unbounded_pools);
          rez.serialize(creator_id);
          rez.serialize(&target);
          rez.serialize(&success);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        if (!success.load())
          return false;
        result = target;
        if (acquire && !target.manager->acquire_instance(MAPPING_ACQUIRE_REF))
          return false;
        else
          return true;
      }
      else
      {
        // Create the builder and initialize it before getting
        // the allocation privilege to avoid deadlock scenario
        InstanceBuilder builder(regions, constraints, this, creator_id);
        if (!builder.initialize())
          return false;
        // Acquire allocation privilege before doing anything
        const RtEvent wait_on =
            acquire_allocation_privilege(coordinates, safe_for_unbounded_pools);
        if ((safe_for_unbounded_pools != nullptr) &&
            safe_for_unbounded_pools->exists())
          return false;
        if (wait_on.exists())
          wait_on.wait();
        // Try to make the result
        PhysicalManager* manager = allocate_physical_instance(
            builder, footprint, unsat_kind, unsat_index);
        bool success = false;
        if (manager != nullptr)
        {
          manager->log_instance_creation(creator_id, processor, regions);
          // Do this first to add a resource reference
          result = MappingInstance(manager);
          record_created_instance(manager, acquire, priority);
          success = true;
        }
        // Release our allocation privilege after doing the record
        release_allocation_privilege();
        return success;
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::create_physical_instance(
        LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        Processor processor, bool acquire, GCPriority priority,
        bool tight_bounds, LayoutConstraintKind* unsat_kind,
        unsigned* unsat_index, size_t* footprint,
        RtEvent* safe_for_unbounded_pools, UniqueID creator_id, bool remote)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // Not the owner, send a meessage to the owner to request the creation
        InstanceRequest rez;
        ManagerResult target;
        std::atomic<bool> success(false);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(CREATE_INSTANCE_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          coordinates.serialize(rez);
          rez.serialize(constraints->layout_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_bounds);
          rez.serialize(unsat_kind);
          rez.serialize(unsat_index);
          rez.serialize(footprint);
          rez.serialize(safe_for_unbounded_pools);
          rez.serialize(creator_id);
          rez.serialize(&target);
          rez.serialize(&success);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        if (!success.load())
          return false;
        result = target;
        if (acquire && !target.manager->acquire_instance(MAPPING_ACQUIRE_REF))
          return false;
        else
          return true;
      }
      else
      {
        // Create the builder and initialize it before getting
        // the allocation privilege to avoid deadlock scenario
        InstanceBuilder builder(regions, *constraints, this, creator_id);
        if (!builder.initialize())
          return false;
        // Acquire allocation privilege before doing anything
        const RtEvent wait_on =
            acquire_allocation_privilege(coordinates, safe_for_unbounded_pools);
        if ((safe_for_unbounded_pools != nullptr) &&
            safe_for_unbounded_pools->exists())
          return false;
        if (wait_on.exists())
          wait_on.wait();
        // Try to make the instance
        PhysicalManager* manager = allocate_physical_instance(
            builder, footprint, unsat_kind, unsat_index);
        bool success = false;
        if (manager != nullptr)
        {
          manager->log_instance_creation(creator_id, processor, regions);
          // Do this first to add a resource reference
          result = MappingInstance(manager);
          record_created_instance(manager, acquire, priority);
          success = true;
        }
        // Release our allocation privilege after doing the record
        release_allocation_privilege();
        return success;
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_or_create_physical_instance(
        const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        bool& created, Processor processor, bool acquire, GCPriority priority,
        bool tight_region_bounds, LayoutConstraintKind* unsat_kind,
        unsigned* unsat_index, size_t* footprint,
        RtEvent* safe_for_unbounded_pools, UniqueID creator_id, bool remote)
    //--------------------------------------------------------------------------
    {
      // Set created to default to false
      created = false;
      if (!is_owner)
      {
        // See if we can find a locally valid instance first
        if (find_valid_instance(
                constraints, regions, result, acquire, tight_region_bounds,
                remote))
          return true;
        // Not the owner, send a message to the owner to request creation
        InstanceRequest rez;
        ManagerResult target;
        std::atomic<bool> remote_created(created);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_OR_CREATE_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          coordinates.serialize(rez);
          constraints.serialize(rez);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(unsat_kind);
          rez.serialize(unsat_index);
          rez.serialize(footprint);
          rez.serialize(safe_for_unbounded_pools);
          rez.serialize(creator_id);
          rez.serialize(&target);
          rez.serialize(&remote_created);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        PhysicalManager* manager = target.manager;
        if (manager != nullptr)
        {
          result = target;
          created = remote_created.load();
          if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
            return false;
          else
            return true;
        }
        else
          return false;
      }
      else
      {
        // Create the builder and initialize it before getting
        // the allocation privilege to avoid deadlock scenario
        InstanceBuilder builder(regions, constraints, this, creator_id);
        if (!builder.initialize())
          return false;
        // First get our allocation privileges so we're the only
        // one trying to do any allocations
        const RtEvent wait_on =
            acquire_allocation_privilege(coordinates, safe_for_unbounded_pools);
        if ((safe_for_unbounded_pools != nullptr) &&
            safe_for_unbounded_pools->exists())
          return false;
        if (wait_on.exists())
          wait_on.wait();
        // Since this is find or acquire, first see if we can find
        // an instance that has already been makde that satisfies
        // our layout constraints
        bool success = find_satisfying_instance(
            constraints, regions, result, acquire, tight_region_bounds, remote);
        if (!success)
        {
          // If we couldn't find it, we have to make it
          PhysicalManager* manager = allocate_physical_instance(
              builder, footprint, unsat_kind, unsat_index);
          if (manager != nullptr)
          {
            success = true;
            manager->log_instance_creation(creator_id, processor, regions);
            // Do this first to add a resource reference
            result = MappingInstance(manager);
            record_created_instance(manager, acquire, priority);
            // We made this instance so mark that it was created
            created = true;
          }
        }
        else if (footprint != nullptr)
          *footprint = result.get_instance_size();
        // Release our allocation privilege after doing the record
        release_allocation_privilege();
        return success;
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_or_create_physical_instance(
        LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        bool& created, Processor processor, bool acquire, GCPriority priority,
        bool tight_region_bounds, LayoutConstraintKind* unsat_kind,
        unsigned* unsat_index, size_t* footprint,
        RtEvent* safe_for_unbounded_pools, UniqueID creator_id, bool remote)
    //--------------------------------------------------------------------------
    {
      // Set created to false in case we fail
      created = false;
      if (!is_owner)
      {
        // See if we can find it locally
        if (find_valid_instance(
                *constraints, regions, result, acquire, tight_region_bounds,
                remote))
          return true;
        // Not the owner, send a message to the owner to request creation
        InstanceRequest rez;
        ManagerResult target;
        std::atomic<bool> remote_created(created);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_OR_CREATE_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          coordinates.serialize(rez);
          rez.serialize(constraints->layout_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(unsat_kind);
          rez.serialize(unsat_index);
          rez.serialize(footprint);
          rez.serialize(safe_for_unbounded_pools);
          rez.serialize(creator_id);
          rez.serialize(&target);
          rez.serialize(&remote_created);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        PhysicalManager* manager = target.manager;
        if (manager != nullptr)
        {
          result = target;
          created = remote_created.load();
          if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
            return false;
          else
            return true;
        }
        else
          return false;
      }
      else
      {
        // Create the builder and initialize it before getting
        // the allocation privilege to avoid deadlock scenario
        InstanceBuilder builder(regions, *constraints, this, creator_id);
        if (!builder.initialize())
          return false;
        // First get our allocation privileges so we're the only
        // one trying to do any allocations
        const RtEvent wait_on =
            acquire_allocation_privilege(coordinates, safe_for_unbounded_pools);
        if ((safe_for_unbounded_pools != nullptr) &&
            safe_for_unbounded_pools->exists())
          return false;
        if (wait_on.exists())
          wait_on.wait();
        // Since this is find or acquire, first see if we can find
        // an instance that has already been makde that satisfies
        // our layout constraints
        // Try to find an instance first and then make one
        bool success = find_satisfying_instance(
            *constraints, regions, result, acquire, tight_region_bounds,
            remote);
        if (!success)
        {
          // If we couldn't find it, we have to make it
          PhysicalManager* manager = allocate_physical_instance(
              builder, footprint, unsat_kind, unsat_index);
          if (manager != nullptr)
          {
            success = true;
            manager->log_instance_creation(creator_id, processor, regions);
            // Do this first to add a resource reference
            result = MappingInstance(manager);
            record_created_instance(manager, acquire, priority);
            // We made this instance so mark that it was created
            created = true;
          }
        }
        else if (footprint != nullptr)
          *footprint = result.get_instance_size();
        // Release our allocation privilege after doing the record
        release_allocation_privilege();
        return success;
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::redistrict_physical_instance(
        MappingInstance& instance, const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions, Processor processor,
        bool acquire, GCPriority priority, bool tight_bounds,
        UniqueID creator_id)
    //--------------------------------------------------------------------------
    {
      PhysicalManager* old_manager = instance.impl->as_physical_manager();
      legion_assert(old_manager->memory_manager == this);
      if (!is_owner)
      {
        // Not the owner, send a meessage to the owner to request the redistrict
        InstanceRequest rez;
        ManagerResult target;
        std::atomic<bool> success(false);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(REDISTRICT_INSTANCE_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          // No need to pack a reference, held by the MapppingInstance
          rez.serialize(old_manager->did);
          constraints.serialize(rez);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_bounds);
          rez.serialize(creator_id);
          rez.serialize(&target);
          rez.serialize(&success);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        PhysicalManager* new_manager = target.manager;
        if (new_manager != nullptr)
        {
          instance = target;
          if (acquire && !new_manager->acquire_instance(MAPPING_ACQUIRE_REF))
            return false;
          else
            return true;
        }
        return success.load();
      }
      else
      {
        // Try to do the collection
        // We don't need to get the allocation privileges here because
        // we're redistricting an existing instance
        RtEvent collected;
        PhysicalInstance hole = PhysicalInstance::NO_INST;
        if (old_manager->collect(collected, &hole) && hole.exists())
        {
          // Create the builder and initialize it before getting
          // the allocation privilege to avoid deadlock scenario
          InstanceBuilder builder(regions, constraints, this, creator_id);
          if (!builder.initialize())
            return false;
          size_t footprint = 0;
          PhysicalManager* new_manager = builder.create_physical_instance(
              nullptr /*unsat kind*/, nullptr /*unset index*/, &footprint,
              collected, hole);
          if (new_manager != nullptr)
          {
            legion_assert(footprint <= new_manager->instance_footprint);
            new_manager->log_instance_creation(creator_id, processor, regions);
            instance = MappingInstance(new_manager);
            record_created_instance(new_manager, acquire, priority);
            // Update the footprint if necessary
            if (footprint < old_manager->instance_footprint)
            {
              const size_t diff = old_manager->instance_footprint - footprint;
#ifdef LEGION_DEBUG
              const size_t previous =
#endif
                  remaining_capacity.fetch_add(diff);
              legion_assert((previous + diff) <= capacity);
            }
            return true;
          }
          else
          {
            // The previous instance was deleted but we didn't reallocate
            // so we freed up all the space for it
#ifdef LEGION_DEBUG
            const size_t previous =
#endif
                remaining_capacity.fetch_add(old_manager->instance_footprint);
            legion_assert(
                (previous + old_manager->instance_footprint) <= capacity);
          }
        }
        return false;
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::redistrict_physical_instance(
        MappingInstance& instance, LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions, Processor processor,
        bool acquire, GCPriority priority, bool tight_bounds,
        UniqueID creator_id)
    //--------------------------------------------------------------------------
    {
      PhysicalManager* old_manager = instance.impl->as_physical_manager();
      legion_assert(old_manager->memory_manager == this);
      if (!is_owner)
      {
        // Not the owner, send a meessage to the owner to request the redistrict
        InstanceRequest rez;
        ManagerResult target;
        std::atomic<bool> success(false);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(REDISTRICT_INSTANCE_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          // No need to pack a reference, held by the MapppingInstance
          rez.serialize(old_manager->did);
          rez.serialize(constraints->layout_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_bounds);
          rez.serialize(creator_id);
          rez.serialize(&target);
          rez.serialize(&success);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        PhysicalManager* new_manager = target.manager;
        if (new_manager != nullptr)
        {
          instance = target;
          if (acquire && !new_manager->acquire_instance(MAPPING_ACQUIRE_REF))
            return false;
          else
            return true;
        }
        return success.load();
      }
      else
      {
        // Try to do the collection
        // We don't need to get the allocation privileges here because
        // we're redistricting an existing instance
        RtEvent collected;
        PhysicalInstance hole = PhysicalInstance::NO_INST;
        PhysicalManager* old_manager = instance.impl->as_physical_manager();
        if (old_manager->collect(collected, &hole) && hole.exists())
        {
          // Create the builder and initialize it before getting
          // the allocation privilege to avoid deadlock scenario
          InstanceBuilder builder(regions, *constraints, this, creator_id);
          if (!builder.initialize())
            return false;
          size_t footprint = 0;
          PhysicalManager* new_manager = builder.create_physical_instance(
              nullptr /*unsat kind*/, nullptr /*unset index*/, &footprint,
              collected, hole);
          if (new_manager != nullptr)
          {
            legion_assert(footprint <= old_manager->instance_footprint);
            new_manager->log_instance_creation(creator_id, processor, regions);
            instance = MappingInstance(new_manager);
            record_created_instance(new_manager, acquire, priority);
            // Update the footprint if necessary
            if (footprint < old_manager->instance_footprint)
            {
              const size_t diff = old_manager->instance_footprint - footprint;
#ifdef LEGION_DEBUG
              const size_t previous =
#endif
                  remaining_capacity.fetch_add(diff);
              legion_assert((previous + diff) <= capacity);
            }
            return true;
          }
          else
          {
            // The previous instance was deleted but we didn't reallocate
            // so we freed up all the space for it
#ifdef LEGION_DEBUG
            const size_t previous =
#endif
                remaining_capacity.fetch_add(old_manager->instance_footprint);
            legion_assert(
                (previous + old_manager->instance_footprint) <= capacity);
          }
        }
        return false;
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_physical_instance(
        const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions, MappingInstance& result,
        bool acquire, bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // See if we can find it locally
        if (find_valid_instance(
                constraints, regions, result, acquire, tight_region_bounds,
                remote))
          return true;
        // Not the owner, send a message to the owner to try and find it
        InstanceRequest rez;
        ManagerResult target;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_ONLY_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          constraints.serialize(rez);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(&target);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        PhysicalManager* manager = target.manager;
        if (manager != nullptr)
        {
          result = target;
          if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
            return false;
          else
            return true;
        }
        else
          return false;
      }
      else
      {
        // Try to find an instance
        return find_satisfying_instance(
            constraints, regions, result, acquire, tight_region_bounds, remote);
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_physical_instance(
        LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions, MappingInstance& result,
        bool acquire, bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // See if we can find a persistent instance
        if (find_valid_instance(
                *constraints, regions, result, acquire, tight_region_bounds,
                remote))
          return true;
        InstanceRequest rez;
        ManagerResult target;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_ONLY_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize(constraints->layout_id);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(&target);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        PhysicalManager* manager = target.manager;
        if (manager != nullptr)
        {
          result = target;
          if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
            return false;
          else
            return true;
        }
        else
          return false;
      }
      else
      {
        // Try to find an instance
        return find_satisfying_instance(
            *constraints, regions, result, acquire, tight_region_bounds,
            remote);
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::find_physical_instances(
        const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions,
        std::vector<MappingInstance>& results, bool acquire,
        bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // Not the owner, send a message to the owner to try and find it
        InstanceRequest rez;
        std::vector<ManagerResult> target;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_MANY_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          constraints.serialize(rez);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(&target);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        for (const ManagerResult& res : target)
        {
          PhysicalManager* manager = res.manager;
          if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
            continue;
          results.emplace_back(res);
        }
      }
      else
        find_satisfying_instances(
            constraints, regions, results, acquire, tight_region_bounds,
            remote);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::find_physical_instances(
        LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions,
        std::vector<MappingInstance>& results, bool acquire,
        bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        InstanceRequest rez;
        std::vector<ManagerResult> target;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_MANY_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize(constraints->layout_id);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(&target);
        }
        rez.dispatch(owner_space);
        ready_event.wait();
        for (const ManagerResult& res : target)
        {
          PhysicalManager* manager = res.manager;
          if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
            continue;
          results.emplace_back(res);
        }
      }
      else
        find_satisfying_instances(
            *constraints, regions, results, acquire, tight_region_bounds,
            remote);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_tree_instances(RegionTreeID tree_id)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, then there is nothing to do
      if (!is_owner)
        return;
      // Try to delete all the instances in the region tree
      // If any of them cannot be deleted yet, they'll have to
      // wait until we do a garbage collection
      // This is a collection so we need to order it with respect to
      // to other collections
      AutoLock c_lock(collection_lock);
      std::vector<PhysicalManager*> to_delete;
      {
        const TreeFieldKey lower_bound(tree_id, 0);
        const TreeFieldKey upper_bound(
            tree_id, std::numeric_limits<uint64_t>::max());
        AutoLock m_lock(manager_lock);
        for (TreeFieldInstances::iterator finder =
                 current_instances.lower_bound(lower_bound);
             finder != current_instances.upper_bound(upper_bound); finder++)
        {
          for (TreeInstances::iterator it = finder->second.begin();
               it != finder->second.end(); it++)
          {
            if (it->first->is_external_instance())
              continue;
            if ((it->second == LEGION_GC_NEVER_PRIORITY) &&
                it->first->is_owner())
            {
              it->first->remove_base_valid_ref(NEVER_GC_REF);
              it->second = 0;
            }
            bool already_collected = false;
            if (it->first->can_collect(already_collected))
            {
              it->first->add_base_gc_ref(MEMORY_MANAGER_REF);
              to_delete.emplace_back(it->first);
#ifdef LEGION_DEBUG
              const size_t previous =
#endif
                  remaining_capacity.fetch_add(it->first->instance_footprint);
              legion_assert(
                  (previous + it->first->instance_footprint) <= capacity);
            }
            else if (already_collected)
              remove_collectable(it->second, it->first);
          }
        }
      }
      if (!to_delete.empty())
        check_instance_deletions(to_delete);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::set_garbage_collection_priority(
        PhysicalManager* manager, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      const TreeFieldKey key(manager);
      AutoLock m_lock(manager_lock);
      TreeFieldInstances::iterator tree_finder = current_instances.find(key);
      legion_assert(tree_finder != current_instances.end());
      TreeInstances::iterator finder = tree_finder->second.find(manager);
      legion_assert(finder != tree_finder->second.end());
      remove_collectable(finder->second, manager);
      finder->second = priority;
      if (priority != LEGION_GC_NEVER_PRIORITY)
        collectable_instances[priority].insert(manager);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_instance_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      RequestKind kind;
      derez.deserialize(kind);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      size_t num_regions;
      derez.deserialize(num_regions);
      std::vector<LogicalRegion> regions(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
        derez.deserialize(regions[idx]);
      switch (kind)
      {
        case CREATE_INSTANCE_CONSTRAINTS:
          {
            TaskTreeCoordinates coordinates;
            coordinates.deserialize(derez);
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_region_bounds;
            derez.deserialize<bool>(tight_region_bounds);
            LayoutConstraintKind* remote_kind;
            derez.deserialize(remote_kind);
            unsigned* remote_index;
            derez.deserialize(remote_index);
            size_t* remote_footprint;  // warning: remote pointer
            derez.deserialize(remote_footprint);
            RtEvent* remote_safe_for_unbounded_pools;
            derez.deserialize(remote_safe_for_unbounded_pools);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            std::atomic<bool>* remote_success;
            derez.deserialize(remote_success);
            MappingInstance result;
            size_t local_footprint;
            LayoutConstraintKind local_kind;
            unsigned local_index;
            RtEvent safe_for_unbounded_pools;
            bool success = create_physical_instance(
                constraints, regions, coordinates, result, processor,
                false /*acquire*/, priority, tight_region_bounds, &local_kind,
                &local_index, &local_footprint,
                (remote_safe_for_unbounded_pools == nullptr) ?
                    nullptr :
                    &safe_for_unbounded_pools,
                creator_id, true /*remote*/);
            if (success || (remote_footprint != nullptr) ||
                (remote_kind != nullptr) || (remote_index != nullptr) ||
                (remote_safe_for_unbounded_pools != nullptr))
            {
              // Send back the response starting with the instance
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(success);
                if (success)
                {
                  InstanceManager* manager = result.impl;
                  rez.serialize(manager->did);
                  rez.serialize(remote_target);
                  manager->pack_global_ref(rez);
                  rez.serialize(remote_success);
                }
                rez.serialize(remote_kind);
                if (remote_kind != nullptr)
                  rez.serialize(local_kind);
                rez.serialize(remote_index);
                if (remote_index != nullptr)
                  rez.serialize(local_index);
                rez.serialize(remote_footprint);
                if (remote_footprint != nullptr)
                  rez.serialize(local_footprint);
                rez.serialize(remote_safe_for_unbounded_pools);
                if (remote_safe_for_unbounded_pools != nullptr)
                  rez.serialize(safe_for_unbounded_pools);
              }
              rez.dispatch(source);
            }
            else  // we can just trigger the done event since we failed
              Runtime::trigger_event(to_trigger);
            break;
          }
        case CREATE_INSTANCE_LAYOUT:
          {
            TaskTreeCoordinates coordinates;
            coordinates.deserialize(derez);
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_region_bounds;
            derez.deserialize<bool>(tight_region_bounds);
            LayoutConstraintKind* remote_kind;
            derez.deserialize(remote_kind);
            unsigned* remote_index;
            derez.deserialize(remote_index);
            size_t* remote_footprint;  // warning: remote pointer
            derez.deserialize(remote_footprint);
            RtEvent* remote_safe_for_unbounded_pools;
            derez.deserialize(remote_safe_for_unbounded_pools);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            std::atomic<bool>* remote_success;
            derez.deserialize(remote_success);
            LayoutConstraints* constraints =
                runtime->find_layout_constraints(layout_id);
            MappingInstance result;
            size_t local_footprint;
            LayoutConstraintKind local_kind;
            unsigned local_index;
            RtEvent safe_for_unbounded_pools;
            bool success = create_physical_instance(
                constraints, regions, coordinates, result, processor,
                false /*acquire*/, priority, tight_region_bounds, &local_kind,
                &local_index, &local_footprint,
                (remote_safe_for_unbounded_pools == nullptr) ?
                    nullptr :
                    &safe_for_unbounded_pools,
                creator_id, true /*remote*/);
            if (success || (remote_footprint != nullptr) ||
                (remote_kind != nullptr) || (remote_index != nullptr) ||
                (remote_safe_for_unbounded_pools != nullptr))
            {
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(success);
                InstanceManager* manager = result.impl;
                if (success)
                {
                  rez.serialize(manager->did);
                  rez.serialize(remote_target);
                  manager->pack_global_ref(rez);
                  rez.serialize(remote_success);
                }
                rez.serialize(remote_kind);
                if (remote_kind != nullptr)
                  rez.serialize(local_kind);
                rez.serialize(remote_index);
                if (remote_index != nullptr)
                  rez.serialize(local_index);
                rez.serialize(remote_footprint);
                if (remote_footprint != nullptr)
                  rez.serialize(local_footprint);
                rez.serialize(remote_safe_for_unbounded_pools);
                if (remote_safe_for_unbounded_pools != nullptr)
                  rez.serialize(safe_for_unbounded_pools);
              }
              rez.dispatch(source);
            }
            else  // if we failed, we can just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_OR_CREATE_CONSTRAINTS:
          {
            TaskTreeCoordinates coordinates;
            coordinates.deserialize(derez);
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            LayoutConstraintKind* remote_kind;
            derez.deserialize(remote_kind);
            unsigned* remote_index;
            derez.deserialize(remote_index);
            size_t* remote_footprint;  // warning: remote pointer
            derez.deserialize(remote_footprint);
            RtEvent* remote_safe_for_unbounded_pools;
            derez.deserialize(remote_safe_for_unbounded_pools);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            std::atomic<bool>* remote_created;
            derez.deserialize(remote_created);
            MappingInstance result;
            size_t local_footprint;
            LayoutConstraintKind local_kind;
            unsigned local_index;
            bool created;
            RtEvent safe_for_unbounded_pools;
            bool success = find_or_create_physical_instance(
                constraints, regions, coordinates, result, created, processor,
                false /*acquire*/, priority, tight_bounds, &local_kind,
                &local_index, &local_footprint,
                (remote_safe_for_unbounded_pools == nullptr) ?
                    nullptr :
                    &safe_for_unbounded_pools,
                creator_id, true /*remote*/);
            if (success || (remote_footprint != nullptr) ||
                (remote_kind != nullptr) || (remote_index != nullptr) ||
                (remote_safe_for_unbounded_pools != nullptr))
            {
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(success);
                InstanceManager* manager = result.impl;
                if (success)
                {
                  rez.serialize(manager->did);
                  rez.serialize(remote_target);
                  manager->pack_global_ref(rez);
                  rez.serialize(remote_created);
                  rez.serialize<bool>(created);
                }
                rez.serialize(remote_kind);
                if (remote_kind != nullptr)
                  rez.serialize(local_kind);
                rez.serialize(remote_index);
                if (remote_index != nullptr)
                  rez.serialize(local_index);
                rez.serialize(remote_footprint);
                if (remote_footprint != nullptr)
                  rez.serialize(local_footprint);
                rez.serialize(remote_safe_for_unbounded_pools);
                if (remote_safe_for_unbounded_pools != nullptr)
                  rez.serialize(safe_for_unbounded_pools);
              }
              rez.dispatch(source);
            }
            else  // if we failed, we can just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_OR_CREATE_LAYOUT:
          {
            TaskTreeCoordinates coordinates;
            coordinates.deserialize(derez);
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            LayoutConstraintKind* remote_kind;
            derez.deserialize(remote_kind);
            unsigned* remote_index;
            derez.deserialize(remote_index);
            size_t* remote_footprint;  // warning: remote pointer
            derez.deserialize(remote_footprint);
            RtEvent* remote_safe_for_unbounded_pools;
            derez.deserialize(remote_safe_for_unbounded_pools);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            std::atomic<bool>* remote_created;
            derez.deserialize(remote_created);
            LayoutConstraints* constraints =
                runtime->find_layout_constraints(layout_id);
            MappingInstance result;
            size_t local_footprint;
            LayoutConstraintKind local_kind;
            unsigned local_index;
            bool created;
            RtEvent safe_for_unbounded_pools;
            bool success = find_or_create_physical_instance(
                constraints, regions, coordinates, result, created, processor,
                false /*acquire*/, priority, tight_bounds, &local_kind,
                &local_index, &local_footprint,
                (remote_safe_for_unbounded_pools == nullptr) ?
                    nullptr :
                    &safe_for_unbounded_pools,
                creator_id, true /*remote*/);
            if (success || (remote_footprint != nullptr) ||
                (remote_kind != nullptr) || (remote_index != nullptr) ||
                (remote_safe_for_unbounded_pools != nullptr))
            {
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(success);
                InstanceManager* manager = result.impl;
                if (success)
                {
                  rez.serialize(manager->did);
                  rez.serialize(remote_target);
                  manager->pack_global_ref(rez);
                  rez.serialize(remote_created);
                  rez.serialize<bool>(created);
                }
                rez.serialize(remote_kind);
                if (remote_kind != nullptr)
                  rez.serialize(local_kind);
                rez.serialize(remote_index);
                if (remote_index != nullptr)
                  rez.serialize(local_index);
                rez.serialize(remote_footprint);
                if (remote_footprint != nullptr)
                  rez.serialize(local_footprint);
                rez.serialize(remote_safe_for_unbounded_pools);
                if (remote_safe_for_unbounded_pools != nullptr)
                  rez.serialize(safe_for_unbounded_pools);
              }
              rez.dispatch(source);
            }
            else  // we failed so just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case REDISTRICT_INSTANCE_CONSTRAINTS:
          {
            DistributedID did;
            derez.deserialize(did);
            RtEvent ready;
            PhysicalManager* manager =
                runtime->find_or_request_instance_manager(did, ready);
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_region_bounds;
            derez.deserialize<bool>(tight_region_bounds);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            std::atomic<bool>* remote_success;
            derez.deserialize(remote_success);
            ready.wait();
            MappingInstance result(manager);
            bool success = redistrict_physical_instance(
                result, constraints, regions, processor, false /*acquire*/,
                priority, tight_region_bounds, creator_id);
            if (success)
            {
              // Send back the response starting with the instance
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(true);
                InstanceManager* manager = result.impl;
                rez.serialize(manager->did);
                rez.serialize(remote_target);
                manager->pack_global_ref(rez);
                rez.serialize(remote_success);
                // No things for us to pass back here
                rez.serialize<LayoutConstraintKind*>(nullptr);
                rez.serialize<unsigned*>(nullptr);
                rez.serialize<size_t*>(nullptr);
                rez.serialize<RtEvent*>(nullptr);
              }
              rez.dispatch(source);
            }
            else  // we can just trigger the done event since we failed
              Runtime::trigger_event(to_trigger);
            break;
          }
        case REDISTRICT_INSTANCE_LAYOUT:
          {
            DistributedID did;
            derez.deserialize(did);
            RtEvent ready;
            PhysicalManager* manager =
                runtime->find_or_request_instance_manager(did, ready);
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_region_bounds;
            derez.deserialize<bool>(tight_region_bounds);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            std::atomic<bool>* remote_success;
            derez.deserialize(remote_success);
            LayoutConstraints* constraints =
                runtime->find_layout_constraints(layout_id);
            ready.wait();
            MappingInstance result(manager);
            bool success = redistrict_physical_instance(
                result, constraints, regions, processor, false /*acquire*/,
                priority, tight_region_bounds, creator_id);
            if (success)
            {
              // Send back the response starting with the instance
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(true);
                InstanceManager* manager = result.impl;
                rez.serialize(manager->did);
                rez.serialize(remote_target);
                manager->pack_global_ref(rez);
                rez.serialize(remote_success);
                // No things for us to pass back here
                rez.serialize<LayoutConstraintKind*>(nullptr);
                rez.serialize<unsigned*>(nullptr);
                rez.serialize<size_t*>(nullptr);
                rez.serialize<RtEvent*>(nullptr);
              }
              rez.dispatch(source);
            }
            else  // we can just trigger the done event since we failed
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_ONLY_CONSTRAINTS:
          {
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            MappingInstance result;
            bool success = find_physical_instance(
                constraints, regions, result, false /*acquire*/, tight_bounds,
                true /*remote*/);
            if (success)
            {
              InstanceManager* manager = result.impl;
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(true);  // success
                rez.serialize(manager->did);
                rez.serialize(remote_target);
                manager->pack_global_ref(rez);
                // No things for us to pass back here
                rez.serialize<LayoutConstraintKind*>(nullptr);
                rez.serialize<unsigned*>(nullptr);
                rez.serialize<size_t*>(nullptr);
                rez.serialize<RtEvent*>(nullptr);
              }
              rez.dispatch(source);
            }
            else  // we failed so we can just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_ONLY_LAYOUT:
          {
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            ManagerResult* remote_target;
            derez.deserialize(remote_target);
            LayoutConstraints* constraints =
                runtime->find_layout_constraints(layout_id);
            MappingInstance result;
            bool success = find_physical_instance(
                constraints, regions, result, false /*acquire*/, tight_bounds,
                true /*remote*/);
            if (success)
            {
              InstanceManager* manager = result.impl;
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(true);  // success
                rez.serialize(manager->did);
                rez.serialize(remote_target);
                manager->pack_global_ref(rez);
                // No things for us to pass back here
                rez.serialize<LayoutConstraintKind*>(nullptr);
                rez.serialize<unsigned*>(nullptr);
                rez.serialize<size_t*>(nullptr);
                rez.serialize<RtEvent*>(nullptr);
              }
              rez.dispatch(source);
            }
            else  // we failed so just trigger
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_MANY_CONSTRAINTS:
          {
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            std::vector<ManagerResult>* remote_target;
            derez.deserialize(remote_target);
            std::vector<MappingInstance> results;
            find_physical_instances(
                constraints, regions, results, false /*acquire*/, tight_bounds,
                true /*remote*/);
            if (!results.empty())
            {
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(false);  // success
                rez.serialize(remote_target);
                rez.serialize<size_t>(results.size());
                for (unsigned idx = 0; idx < results.size(); idx++)
                {
                  InstanceManager* manager = results[idx].impl;
                  rez.serialize(manager->did);
                  manager->pack_global_ref(rez);
                }
                // No things for us to pass back here
                rez.serialize<LayoutConstraintKind*>(nullptr);
                rez.serialize<unsigned*>(nullptr);
                rez.serialize<size_t*>(nullptr);
                rez.serialize<RtEvent*>(nullptr);
              }
              rez.dispatch(source);
            }
            else  // we failed so we can just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_MANY_LAYOUT:
          {
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            std::vector<ManagerResult>* remote_target;
            derez.deserialize(remote_target);
            LayoutConstraints* constraints =
                runtime->find_layout_constraints(layout_id);
            std::vector<MappingInstance> results;
            find_physical_instances(
                constraints, regions, results, false /*acquire*/, tight_bounds,
                true /*remote*/);
            if (!results.empty())
            {
              InstanceResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(kind);
                rez.serialize<bool>(false);  // success
                rez.serialize(remote_target);
                rez.serialize<size_t>(results.size());
                for (unsigned idx = 0; idx < results.size(); idx++)
                {
                  InstanceManager* manager = results[idx].impl;
                  rez.serialize(manager->did);
                  manager->pack_global_ref(rez);
                }
                // No things for us to pass back here
                rez.serialize<LayoutConstraintKind*>(nullptr);
                rez.serialize<unsigned*>(nullptr);
                rez.serialize<size_t*>(nullptr);
                rez.serialize<RtEvent*>(nullptr);
              }
              rez.dispatch(source);
            }
            else  // we failed so just trigger
              Runtime::trigger_event(to_trigger);
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_instance_response(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      RequestKind kind;
      derez.deserialize(kind);
      bool success;
      derez.deserialize<bool>(success);
      std::vector<RtEvent> preconditions;
      if (success)
      {
        DistributedID did;
        derez.deserialize(did);
        ManagerResult* target;
        derez.deserialize(target);
        legion_assert(
            (CREATE_INSTANCE_CONSTRAINTS <= kind) &&
            (kind <= FIND_ONLY_LAYOUT));
        if (did > 0)
        {
          RtEvent manager_ready = RtEvent::NO_RT_EVENT;
          target->manager =
              runtime->find_or_request_instance_manager(did, manager_ready);
          // If the manager isn't ready yet, then we need to wait for it
          if (manager_ready.exists())
            preconditions.emplace_back(manager_ready);
          target->lamport_clock =
              PhysicalManager::unpack_global_ref_clock(derez);
        }
        if ((kind == CREATE_INSTANCE_CONSTRAINTS) ||
            (kind == CREATE_INSTANCE_LAYOUT) ||
            (kind == REDISTRICT_INSTANCE_CONSTRAINTS) ||
            (kind == REDISTRICT_INSTANCE_LAYOUT))
        {
          std::atomic<bool>* remote_success;
          derez.deserialize(remote_success);
          remote_success->store(true);
        }
        else if (
            (kind == FIND_OR_CREATE_CONSTRAINTS) ||
            (kind == FIND_OR_CREATE_LAYOUT))
        {
          std::atomic<bool>* created_ptr;
          derez.deserialize(created_ptr);
          bool created;
          derez.deserialize(created);
          created_ptr->store(created);
        }
      }
      else
      {
        if ((kind == FIND_MANY_CONSTRAINTS) || (kind == FIND_MANY_LAYOUT))
        {
          std::vector<ManagerResult>* target;
          derez.deserialize(target);
          size_t num_insts;
          derez.deserialize(num_insts);
          target->reserve(num_insts);
          for (unsigned idx = 0; idx < num_insts; idx++)
          {
            DistributedID did;
            derez.deserialize(did);
            RtEvent manager_ready = RtEvent::NO_RT_EVENT;
            PhysicalManager* manager =
                runtime->find_or_request_instance_manager(did, manager_ready);
            // If the manager isn't ready yet, then we need to wait for it
            if (manager_ready.exists())
              preconditions.emplace_back(manager_ready);
            LamportClock lamport_clock =
                PhysicalManager::unpack_global_ref_clock(derez);
            target->emplace_back(ManagerResult(manager, lamport_clock));
          }
        }
      }
      // Unpack the constraint responses
      LayoutConstraintKind* local_kind;
      derez.deserialize(local_kind);
      if (local_kind != nullptr)
        derez.deserialize(*local_kind);
      unsigned* local_index;
      derez.deserialize(local_index);
      if (local_index != nullptr)
        derez.deserialize(*local_index);
      // Unpack the footprint and asign it if necessary
      size_t* local_footprint;
      derez.deserialize(local_footprint);
      if (local_footprint != nullptr)
        derez.deserialize(*local_footprint);
      RtEvent* local_safe_for_unbounded_pools;
      derez.deserialize(local_safe_for_unbounded_pools);
      if (local_safe_for_unbounded_pools != nullptr)
        derez.deserialize(*local_safe_for_unbounded_pools);
      // Trigger that we are done
      if (!preconditions.empty())
        Runtime::trigger_event(
            to_trigger, Runtime::merge_events(preconditions));
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_satisfying_instance(
        const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions, MappingInstance& result,
        bool acquire, bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tree_id = 0;
      for (const LogicalRegion& region : regions)
      {
        if (!region.exists())
          continue;
        tree_id = region.get_tree_id();
        break;
      }
      std::deque<PhysicalManager*> candidates;
      if (tree_id != 0)
      {
        FieldMask field_mask;
        TreeFieldKey lower_bound(tree_id, 0);
        if (!constraints.field_constraint.field_set.empty())
        {
          RegionNode* root = runtime->get_tree(tree_id);
          std::vector<unsigned> indexes(
              constraints.field_constraint.field_set.size());
          root->column_source->get_field_indexes(
              constraints.field_constraint.field_set, indexes);
          for (unsigned idx : indexes) field_mask.set_bit(idx);
          lower_bound.second = field_mask.get_hash_key();
        }
        const TreeFieldKey upper_bound(
            tree_id, std::numeric_limits<uint64_t>::max());
        // Hold the lock while searching here
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        for (TreeFieldInstances::const_iterator finder =
                 current_instances.lower_bound(lower_bound);
             finder != current_instances.upper_bound(upper_bound); finder++)
        {
          // Check to see if the field hash dominates
          if ((finder->first.second & lower_bound.second) != lower_bound.second)
            continue;
          for (TreeInstances::const_iterator it = finder->second.begin();
               it != finder->second.end(); it++)
          {
            if (it->first->is_collected())
              continue;
            if (!!(field_mask - it->first->layout->allocated_fields))
              continue;
            it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
            candidates.emplace_back(it->first);
          }
        }
      }
      else
      {
        // Just get all the instances since we don't care about regions
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        for (TreeFieldInstances::const_iterator rit = current_instances.begin();
             rit != current_instances.end(); rit++)
        {
          for (TreeInstances::const_iterator it = rit->second.begin();
               it != rit->second.end(); it++)
          {
            if (it->first->is_collected())
              continue;
            it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
            candidates.emplace_back(it->first);
          }
        }
      }
      // If we have any candidates check their constraints
      bool found = false;
      if (!candidates.empty())
      {
        if (tree_id != 0)
        {
          std::set<IndexSpaceExpression*> region_exprs;
          for (const LogicalRegion& region : regions)
          {
            // If the region tree IDs don't match that is bad
            if (tree_id != region.get_tree_id())
              return false;
            RegionNode* node = runtime->get_node(region);
            region_exprs.insert(node->row_source);
          }
          IndexSpaceExpression* space_expr =
              (region_exprs.size() == 1) ?
                  *(region_exprs.begin()) :
                  runtime->union_index_spaces(region_exprs);
          for (PhysicalManager* manager : candidates)
          {
            if (!manager->meets_expression(
                    space_expr, tight_region_bounds,
                    &constraints.padding_constraint.delta))
              continue;
            if (manager->entails(constraints, nullptr))
            {
              // Check to see if we need to acquire
              // If we fail to acquire then keep going
              if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
                continue;
              // If we make it here, we succeeded
              result = MappingInstance(manager);
              found = true;
              break;
            }
          }
        }
        else
        {
          // No region constraints, just check the base constraints
          for (PhysicalManager* manager : candidates)
          {
            if (manager->entails(constraints, nullptr))
            {
              // Check to see if we need to acquire
              // If we fail to acquire then keep going
              if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
                continue;
              // If we make it here, we succeeded
              result = MappingInstance(manager);
              found = true;
              break;
            }
          }
        }
        release_candidate_references(candidates);
      }
      return found;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::find_satisfying_instances(
        const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions,
        std::vector<MappingInstance>& results, bool acquire,
        bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tree_id = 0;
      for (const LogicalRegion& region : regions)
      {
        if (!region.exists())
          continue;
        tree_id = region.get_tree_id();
        break;
      }
      std::deque<PhysicalManager*> candidates;
      if (tree_id != 0)
      {
        FieldMask field_mask;
        TreeFieldKey lower_bound(tree_id, 0);
        if (!constraints.field_constraint.field_set.empty())
        {
          RegionNode* root = runtime->get_tree(tree_id);
          std::vector<unsigned> indexes(
              constraints.field_constraint.field_set.size());
          root->column_source->get_field_indexes(
              constraints.field_constraint.field_set, indexes);
          for (unsigned idx : indexes) field_mask.set_bit(idx);
          lower_bound.second = field_mask.get_hash_key();
        }
        const TreeFieldKey upper_bound(
            tree_id, std::numeric_limits<uint64_t>::max());
        // Hold the lock while searching here
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        for (TreeFieldInstances::const_iterator finder =
                 current_instances.lower_bound(lower_bound);
             finder != current_instances.upper_bound(upper_bound); finder++)
        {
          // Check to see if the field hash dominates
          if ((finder->first.second & lower_bound.second) != lower_bound.second)
            continue;
          for (TreeInstances::const_iterator it = finder->second.begin();
               it != finder->second.end(); it++)
          {
            if (it->first->is_collected())
              continue;
            if (!!(field_mask - it->first->layout->allocated_fields))
              continue;
            it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
            candidates.emplace_back(it->first);
          }
        }
      }
      else
      {
        // Just get all the instances since we don't care about regions
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        for (TreeFieldInstances::const_iterator rit = current_instances.begin();
             rit != current_instances.end(); rit++)
        {
          for (TreeInstances::const_iterator it = rit->second.begin();
               it != rit->second.end(); it++)
          {
            if (it->first->is_collected())
              continue;
            it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
            candidates.emplace_back(it->first);
          }
        }
      }
      // If we have any candidates check their constraints
      if (!candidates.empty())
      {
        if (tree_id != 0)
        {
          std::set<IndexSpaceExpression*> region_exprs;
          for (const LogicalRegion& region : regions)
          {
            // If the region tree IDs don't match that is bad
            if (tree_id != region.get_tree_id())
              return;
            RegionNode* node = runtime->get_node(region);
            region_exprs.insert(node->row_source);
          }
          IndexSpaceExpression* space_expr =
              (region_exprs.size() == 1) ?
                  *(region_exprs.begin()) :
                  runtime->union_index_spaces(region_exprs);
          for (PhysicalManager* manager : candidates)
          {
            if (!manager->meets_expression(
                    space_expr, tight_region_bounds,
                    &constraints.padding_constraint.delta))
              continue;
            if (manager->entails(constraints, nullptr))
            {
              // Check to see if we need to acquire
              // If we fail to acquire then keep going
              if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
                continue;
              // If we make it here, we succeeded
              results.emplace_back(MappingInstance(manager));
            }
          }
        }
        else
        {
          // No regions to care about here, just check constraints
          for (PhysicalManager* manager : candidates)
          {
            if (manager->entails(constraints, nullptr))
            {
              // Check to see if we need to acquire
              // If we fail to acquire then keep going
              if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
                continue;
              // If we make it here, we succeeded
              results.emplace_back(MappingInstance(manager));
            }
          }
        }
        release_candidate_references(candidates);
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_valid_instance(
        const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions, MappingInstance& result,
        bool acquire, bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      if (regions.empty())
        return false;
      RegionTreeID tree_id = 0;
      for (const LogicalRegion& region : regions)
      {
        if (!region.exists())
          continue;
        tree_id = region.get_tree_id();
        break;
      }
      if (tree_id == 0)
        return false;
      std::deque<PhysicalManager*> candidates;
      {
        FieldMask field_mask;
        TreeFieldKey lower_bound(tree_id, 0);
        if (!constraints.field_constraint.field_set.empty())
        {
          RegionNode* root = runtime->get_tree(tree_id);
          std::vector<unsigned> indexes(
              constraints.field_constraint.field_set.size());
          root->column_source->get_field_indexes(
              constraints.field_constraint.field_set, indexes);
          for (unsigned idx : indexes) field_mask.set_bit(idx);
          lower_bound.second = field_mask.get_hash_key();
        }
        const TreeFieldKey upper_bound(
            tree_id, std::numeric_limits<uint64_t>::max());
        // Hold the lock while searching here
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        for (TreeFieldInstances::const_iterator finder =
                 current_instances.lower_bound(lower_bound);
             finder != current_instances.upper_bound(upper_bound); finder++)
        {
          // Check to see if the field hash dominates
          if ((finder->first.second & lower_bound.second) != lower_bound.second)
            continue;
          for (TreeInstances::const_iterator it = finder->second.begin();
               it != finder->second.end(); it++)
          {
            if (it->first->is_collected())
              continue;
            if (!!(field_mask - it->first->layout->allocated_fields))
              continue;
            it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
            candidates.emplace_back(it->first);
          }
        }
      }
      // If we have any candidates check their constraints
      bool found = false;
      if (!candidates.empty())
      {
        std::set<IndexSpaceExpression*> region_exprs;
        for (const LogicalRegion& region : regions)
        {
          // If the region tree IDs don't match that is bad
          if (tree_id != region.get_tree_id())
            return false;
          RegionNode* node = runtime->get_node(region);
          region_exprs.insert(node->row_source);
        }
        IndexSpaceExpression* space_expr =
            (region_exprs.size() == 1) ?
                *(region_exprs.begin()) :
                runtime->union_index_spaces(region_exprs);
        for (PhysicalManager* manager : candidates)
        {
          if (!manager->meets_expression(
                  space_expr, tight_region_bounds,
                  &constraints.padding_constraint.delta))
            continue;
          if (manager->entails(constraints, nullptr))
          {
            // Check to see if we need to acquire
            // If we fail to acquire then keep going
            if (acquire && !manager->acquire_instance(MAPPING_ACQUIRE_REF))
              continue;
            // If we make it here, we succeeded
            result = MappingInstance(manager);
            found = true;
            break;
          }
        }
        release_candidate_references(candidates);
      }
      return found;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_candidate_references(
        const std::deque<PhysicalManager*>& candidates) const
    //--------------------------------------------------------------------------
    {
      for (PhysicalManager* manager : candidates)
        if (manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
          delete manager;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* MemoryManager::create_unbound_instance(
        LogicalRegion region, const LayoutConstraintSet& constraints,
        ApEvent producer_event, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // Send the request to the owner node for this memory to make
        // the manager and then get back to the response
        std::pair<PhysicalManager*, LamportClock> result(nullptr, 0);
        const RtUserEvent ready = Runtime::create_rt_user_event();
        CreateUnboundRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(region);
          constraints.serialize(rez);
          rez.serialize(producer_event);
          rez.serialize(priority);
          rez.serialize(&result);
          rez.serialize(ready);
        }
        rez.dispatch(owner_space);
        ready.wait();
        legion_assert(result.first != nullptr);
        legion_no_skip_assert(
            result.first->acquire_instance(MAPPING_ACQUIRE_REF));
        // Remove the packed valid reference that came back with the response
        result.first->unpack_valid_ref(result.second);
        return result.first;
      }
      // We don't need to acquire allocation privilege as this function
      // doesn't eagerly perform any instance collections.

      RegionNode* node = runtime->get_node(region);
      FieldSpaceNode* fspace_node = node->get_column_source();

      const std::vector<FieldID>& fields =
          constraints.field_constraint.field_set;
      FieldMask instance_mask;
      std::vector<size_t> field_sizes(fields.size());
      std::vector<unsigned> mask_index_map(fields.size());
      std::vector<CustomSerdezID> serdez(fields.size());
      fspace_node->compute_field_layout(
          fields, field_sizes, mask_index_map, serdez, instance_mask);

      LayoutDescription* layout =
          fspace_node->find_layout_description(instance_mask, 1, constraints);
      if (layout == nullptr)
      {
        LayoutConstraints* internal_constraints = runtime->register_layout(
            fspace_node->handle, constraints, true /*internal*/);
        layout = fspace_node->create_layout_description(
            instance_mask, 1, internal_constraints, mask_index_map, fields,
            field_sizes, serdez);
      }

      // Create an individual manager with a null instance
      DistributedID did = runtime->get_available_distributed_id();

      LgEvent unique_event;
      // When Legion Spy is enabled, we want the ready event to be unique.
      // So we create a fresh event and trigger it with the producer event
      if ((spy_logging_level > NO_SPY_LOGGING) ||
          (runtime->profiler != nullptr))
        Runtime::rename_event(unique_event);
      PhysicalManager* manager = new PhysicalManager(
          did, this, PhysicalInstance::NO_INST,
          node->get_row_source()->as_index_space_node(), nullptr /*piece_list*/,
          0 /*piece_list_size*/, fspace_node, region.get_tree_id(), layout,
          0 /*redop id*/, true /*register now*/, -1U /*instance_footprint*/,
          producer_event, unique_event, PhysicalManager::UNBOUND_INSTANCE_KIND,
          nullptr /*op*/, nullptr /*collective mapping*/, producer_event);
      if (implicit_profiler != nullptr)
      {
        implicit_profiler->register_physical_instance_region(
            unique_event, region);
        implicit_profiler->register_physical_instance_layout(
            unique_event, region.get_field_space(), constraints);
      }
      // Register the instance to make it visible to downstream tasks
      record_created_instance(manager, true /*acquire*/, priority);
      return manager;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreateUnboundRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      LogicalRegion region;
      derez.deserialize(region);
      LayoutConstraintSet constraints;
      constraints.deserialize(derez);
      ApEvent producer_event;
      derez.deserialize(producer_event);
      GCPriority priority;
      derez.deserialize(priority);
      std::pair<PhysicalManager*, LamportClock>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      MemoryManager* memory_manager = runtime->find_memory_manager(memory);
      PhysicalManager* manager = memory_manager->create_unbound_instance(
          region, constraints, producer_event, priority);
      CreateUnboundResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(target);
        rez.serialize(manager->did);
        rez.serialize(done);
        manager->pack_valid_ref(rez);
      }
      rez.dispatch(source);
      if (manager->remove_base_valid_ref(MAPPING_ACQUIRE_REF))
        delete manager;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreateUnboundResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::pair<PhysicalManager*, LamportClock>* target;
      derez.deserialize(target);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done;
      derez.deserialize(done);
      RtEvent ready;
      target->first = runtime->find_or_request_instance_manager(did, ready);
      derez.deserialize(target->second);
      Runtime::trigger_event(done, ready);
    }

    //--------------------------------------------------------------------------
    RtEvent MemoryManager::acquire_allocation_privilege(
        const TaskTreeCoordinates& coordinates,
        RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);  // should only happen on the owner
      AutoLock m_lock(manager_lock);
      if (!pending_allocation_attempts.empty())
      {
        // See if we're prepending or appending
        // We prepend if we're bypassing an unbounded pool allocation
        // which only occurs for strict and index task scopes
        if ((unbounded_pool_scope == LEGION_STRICT_UNBOUNDED_POOL) &&
            (unbounded_coordinates == coordinates))
        {
          // See if the front allocation has the same coordinates, if it
          // does then we insert ourselves after it, if not then we insert
          // ourselves in front of it
          std::pair<RtUserEvent, const TaskTreeCoordinates*>& front =
              pending_allocation_attempts.front();
          if (*front.second == unbounded_coordinates)
          {
            std::pair<RtUserEvent, const TaskTreeCoordinates*> current = front;
            pending_allocation_attempts.pop_front();
            const RtUserEvent wait_on = Runtime::create_rt_user_event();
            pending_allocation_attempts.emplace_front(
                std::make_pair(wait_on, &coordinates));
            pending_allocation_attempts.emplace_front(current);
            return wait_on;
          }
          else
          {
            // We're the first allocation bypassing, so just add
            // ourselves to the front of the list
            pending_allocation_attempts.emplace_front(
                std::make_pair(RtUserEvent::NO_RT_USER_EVENT, &coordinates));
            return RtEvent::NO_RT_EVENT;
          }
        }
        else if (
            (unbounded_pool_scope == LEGION_INDEX_TASK_UNBOUNDED_POOL) &&
            unbounded_coordinates.same_index_space(coordinates))
        {
          // See if the front allocation has the same index space task
          // coordinates, if it does then we insert ourselves after it,
          // if not then we insert ourselves in front of it
          std::pair<RtUserEvent, const TaskTreeCoordinates*>& front =
              pending_allocation_attempts.front();
          if (unbounded_coordinates.same_index_space(*front.second))
          {
            std::pair<RtUserEvent, const TaskTreeCoordinates*> current = front;
            pending_allocation_attempts.pop_front();
            const RtUserEvent wait_on = Runtime::create_rt_user_event();
            pending_allocation_attempts.emplace_front(
                std::make_pair(wait_on, &coordinates));
            pending_allocation_attempts.emplace_front(current);
            return wait_on;
          }
          else
          {
            // We're the first allocation bypassing, so just add
            // ourselves to the front of the list
            pending_allocation_attempts.emplace_front(
                std::make_pair(RtUserEvent::NO_RT_USER_EVENT, &coordinates));
            return RtEvent::NO_RT_EVENT;
          }
        }
        else
        {
          // Check to see if this is safe for unbounded pools
          if (safe_for_unbounded_pools != nullptr)
          {
            legion_assert(!safe_for_unbounded_pools->exists());
            // If there is an unbounded pool that we're going to block
            // on that is potentially unsafe
            if ((unbounded_pool_scope == LEGION_STRICT_UNBOUNDED_POOL) ||
                (unbounded_pool_scope == LEGION_INDEX_TASK_UNBOUNDED_POOL))
            {
              if (!unbounded_transition_event.exists())
                unbounded_transition_event = Runtime::create_rt_user_event();
              *safe_for_unbounded_pools = unbounded_transition_event;
              return RtEvent::NO_RT_EVENT;
            }
          }
          // Appending like normal to a list of pending allocations
          const RtUserEvent wait_on = Runtime::create_rt_user_event();
          pending_allocation_attempts.emplace_back(
              std::make_pair(wait_on, &coordinates));
          return wait_on;
        }
      }
      else
      {
        // We have no current pending allocations, see if we have an
        // unbounded pool that we need to bypass
        if ((unbounded_pool_scope == LEGION_STRICT_UNBOUNDED_POOL) &&
            (unbounded_coordinates != coordinates))
        {
          // Cannot bypass with different coordinates
          if (safe_for_unbounded_pools != nullptr)
          {
            legion_assert(!safe_for_unbounded_pools->exists());
            if (!unbounded_transition_event.exists())
              unbounded_transition_event = Runtime::create_rt_user_event();
            *safe_for_unbounded_pools = unbounded_transition_event;
            return RtEvent::NO_RT_EVENT;
          }
          const RtUserEvent wait_on = Runtime::create_rt_user_event();
          pending_allocation_attempts.emplace_back(
              std::make_pair(wait_on, &coordinates));
          return wait_on;
        }
        else if (
            (unbounded_pool_scope == LEGION_INDEX_TASK_UNBOUNDED_POOL) &&
            !unbounded_coordinates.same_index_space(coordinates))
        {
          // Cannot bypass without being in the same index space task
          if (safe_for_unbounded_pools != nullptr)
          {
            legion_assert(!safe_for_unbounded_pools->exists());
            if (!unbounded_transition_event.exists())
              unbounded_transition_event = Runtime::create_rt_user_event();
            *safe_for_unbounded_pools = unbounded_transition_event;
            return RtEvent::NO_RT_EVENT;
          }
          const RtUserEvent wait_on = Runtime::create_rt_user_event();
          pending_allocation_attempts.emplace_back(
              std::make_pair(wait_on, &coordinates));
          return wait_on;
        }
        else
        {
          // No unbounded pool or a permissive one so we can do our
          // allocation immediately, put in our guard allocation
          pending_allocation_attempts.emplace_back(
              std::make_pair(RtUserEvent::NO_RT_USER_EVENT, &coordinates));
          return RtEvent::NO_RT_EVENT;
        }
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_allocation_privilege(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);  // should only happen on the owner
      RtUserEvent to_trigger;
      {
        AutoLock m_lock(manager_lock);
        legion_assert(!pending_allocation_attempts.empty());
        // Pop the current pending allocation off the list
        pending_allocation_attempts.pop_front();
        if (!pending_allocation_attempts.empty())
        {
          const std::pair<RtUserEvent, const TaskTreeCoordinates*>& next =
              pending_allocation_attempts.front();
          legion_assert(next.first.exists());
          // If we're in an unbounded pool, see if the next allocation
          // can also bypass the current one, if we're not in one of those
          // scenarios then we can always start the next allocation
          switch (unbounded_pool_scope)
          {
            case LEGION_BOUNDED_POOL:
            case LEGION_PERMISSIVE_UNBOUNDED_POOL:
              {
                to_trigger = next.first;
                break;
              }
            case LEGION_STRICT_UNBOUNDED_POOL:
              {
                if (*next.second == unbounded_coordinates)
                  to_trigger = next.first;
                break;
              }
            case LEGION_INDEX_TASK_UNBOUNDED_POOL:
              {
                if (unbounded_coordinates.same_index_space(*next.second))
                  to_trigger = next.first;
                break;
              }
            default:
              std::abort();
          }
        }
        else if (
            unbounded_transition_event.exists() &&
            (outstanding_task_local_allocations == 0))
        {
          // Notify the unbounded pools that they can try again since
          // there are no more outstanding
          to_trigger = unbounded_transition_event;
          unbounded_transition_event = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    MemoryManager::GarbageCollector::GarbageCollector(
        LocalLock& c_lock, LocalLock& m_lock, AddressSpaceID local, Memory mem,
        size_t needed, size_t cap, std::atomic<size_t>& remaining,
        std::map<
            GCPriority, std::set<PhysicalManager*>, std::greater<GCPriority>>&
            collectables)
      : collection_lock(c_lock), manager_lock(m_lock),
        collectable_instances(collectables), memory(mem), local_space(local),
        needed_size(needed), capacity(cap), remaining_capacity(remaining)
    //--------------------------------------------------------------------------
    {
      AutoLock man_lock(manager_lock);
      if (!collectable_instances.empty())
      {
        current_priority = collectable_instances.begin()->first;
        sort_next_priority_holes(false /*advance*/);
      }
      else
        current_priority = LEGION_GC_NEVER_PRIORITY;
    }

    //--------------------------------------------------------------------------
    MemoryManager::GarbageCollector::~GarbageCollector(void)
    //--------------------------------------------------------------------------
    {
      // Remove any references to any holes that we are still holding
      for (PhysicalManager* manager : small_holes)
        if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
          delete manager;
      for (PhysicalManager* manager : perfect_holes)
        if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
          delete manager;
      for (const std::pair<const size_t, std::vector<PhysicalManager*>>& it :
           large_holes)
        for (PhysicalManager* manager : it.second)
          if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
            delete manager;
      for (const std::pair<const uintptr_t, Range>& it : ranges)
        for (PhysicalManager* manager : it.second.managers)
          if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
            delete manager;
    }

    //--------------------------------------------------------------------------
    MemoryManager::GarbageCollector::Range::Range(PhysicalManager* m)
      : size(m->instance_footprint)
    //--------------------------------------------------------------------------
    {
      managers.emplace_back(m);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::GarbageCollector::sort_next_priority_holes(bool advance)
    //--------------------------------------------------------------------------
    {
      if (!collectable_instances.empty())
      {
        std::map<
            GCPriority, std::set<PhysicalManager*>,
            std::greater<GCPriority>>::iterator next =
            collectable_instances.lower_bound(current_priority);
        if ((next->first == current_priority) && advance)
          next = std::next(next);
        if (next != collectable_instances.end())
        {
          current_priority = next->first;
          for (std::set<PhysicalManager*>::iterator it = next->second.begin();
               it != next->second.end();
               /*nothing*/)
          {
            bool already_collected = false;
            if ((*it)->can_collect(already_collected))
            {
              // Add a reference that ensures that this manager
              // won't be deleted out from under us when we do
              // the call to 'collect'
              (*it)->add_base_gc_ref(MEMORY_MANAGER_REF);
              if ((*it)->instance_footprint == needed_size)
                perfect_holes.emplace_back(*it);
              else if ((*it)->instance_footprint < needed_size)
                small_holes.emplace_back(*it);
              else
                large_holes[(*it)->instance_footprint].emplace_back(*it);
            }
            else if (already_collected)
            {
              // We can prune this out of the collected set immediately
              // since it has already been deleted so there is no need
              // to consider it again
              std::set<PhysicalManager*>::iterator to_delete = it++;
              next->second.erase(to_delete);
              continue;
            }
            it++;
          }
          if (next->second.empty())
            collectable_instances.erase(next);
        }
        else
          current_priority = LEGION_GC_NEVER_PRIORITY;
      }
      else
        current_priority = LEGION_GC_NEVER_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::GarbageCollector::update_capacity(size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      const size_t previous =
#endif
          remaining_capacity.fetch_add(size);
      legion_assert((previous + size) <= capacity);
    }

    //--------------------------------------------------------------------------
    RtEvent MemoryManager::GarbageCollector::perform_collection(
        PhysicalInstance& hole_instance, LgEvent& hole_unique_event)
    //--------------------------------------------------------------------------
    {
      while (!collection_complete())
      {
        // If we've run out of stuff for this priority then go on to the next
        if (small_holes.empty() && perfect_holes.empty() &&
            large_holes.empty() && ranges.empty())
        {
          AutoLock m_lock(manager_lock);
          sort_next_priority_holes();
          continue;
        }
        // Try to use any other perfectly sized instances first
        while (!perfect_holes.empty())
        {
          PhysicalManager* manager = perfect_holes.back();
          perfect_holes.pop_back();
          RtEvent collected;
          if (manager->collect(collected, &hole_instance))
          {
            if (hole_instance.exists())
              hole_unique_event = manager->get_unique_event();
            update_capacity(manager->instance_footprint);
            if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
              delete manager;
            return collected;
          }
          else if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
            delete manager;
        }
        // If that didn't work try to use any large holes starting from
        // the ones that are closest in size to the largest
        while (!large_holes.empty())
        {
          std::map<size_t, std::vector<PhysicalManager*>>::iterator sit =
              large_holes.begin();
          while (!sit->second.empty())
          {
            PhysicalManager* manager = sit->second.back();
            sit->second.pop_back();
            RtEvent collected;
            if (manager->collect(collected, &hole_instance))
            {
              if (hole_instance.exists())
                hole_unique_event = manager->get_unique_event();
              update_capacity(manager->instance_footprint);
              if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
                delete manager;
              return collected;
            }
            else if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
              delete manager;
          }
          large_holes.erase(sit);
        }
        // If we're down to just holes that are smaller than our desired
        // size then try grouping the small holes together into chunks that
        // are either as big as possible or as big as the hole we need and
        // try deleting them
        while (!small_holes.empty())
        {
          PhysicalManager* small_manager = small_holes.back();
          small_holes.pop_back();
          uintptr_t ptr = small_manager->get_instance_pointer();
          // Insert our range
          std::map<uintptr_t, Range>::iterator rit =
              ranges.insert(std::make_pair(ptr, Range(small_manager))).first;
          // Check if we can join it with the one before or after
          if (rit != ranges.begin())
          {
            std::map<uintptr_t, Range>::iterator prev = std::prev(rit);
            if ((prev->first + prev->second.size) == rit->first)
            {
              // Merge rit into prev
              prev->second.size += rit->second.size;
              prev->second.managers.insert(
                  prev->second.managers.end(), rit->second.managers.begin(),
                  rit->second.managers.end());
              ranges.erase(rit);
              rit = prev;
            }
          }
          std::map<uintptr_t, Range>::iterator next = std::next(rit);
          if (next != ranges.end())
          {
            if ((rit->first + rit->second.size) == next->first)
            {
              // Merge next into rit
              rit->second.size += next->second.size;
              rit->second.managers.insert(
                  rit->second.managers.end(), next->second.managers.begin(),
                  next->second.managers.end());
              ranges.erase(next);
            }
          }
          // See if it is is big enough to try an allocation
          if (needed_size <= rit->second.size)
          {
            std::vector<RtEvent> collected_events;
            for (PhysicalManager* manager : rit->second.managers)
            {
              RtEvent collected;
              if (manager->collect(collected))
              {
                update_capacity(manager->instance_footprint);
                if (collected.exists())
                  collected_events.emplace_back(collected);
              }
              if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
                delete manager;
            }
            ranges.erase(rit);
            if (!collected_events.empty())
              return Runtime::merge_events(collected_events);
            else
              return RtEvent::NO_RT_EVENT;
          }
        }
        // At this point, things look pretty hopeless, so just
        // go through and start deleting ranges until we've freed
        // up enough memory for the needed size until we run out
        // of stuff to delete
        size_t freed_size = 0;
        std::vector<RtEvent> collected_events;
        while (!ranges.empty())
        {
          std::map<uintptr_t, Range>::iterator rit = ranges.begin();
          for (PhysicalManager* manager : rit->second.managers)
          {
            RtEvent collected;
            if (manager->collect(collected))
            {
              update_capacity(manager->instance_footprint);
              if (collected.exists())
                collected_events.emplace_back(collected);
            }
            freed_size += manager->instance_footprint;
            if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
              delete manager;
          }
          ranges.erase(rit);
          if (needed_size <= freed_size)
          {
            if (!collected_events.empty())
              return Runtime::merge_events(collected_events);
            else
              return RtEvent::NO_RT_EVENT;
          }
        }
        // Can try one more collection at this level before going to the next
        if (freed_size > 0)
        {
          if (!collected_events.empty())
            return Runtime::merge_events(collected_events);
          else
            return RtEvent::NO_RT_EVENT;
        }
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* MemoryManager::allocate_physical_instance(
        InstanceBuilder& builder, size_t* footprint,
        LayoutConstraintKind* unsat_kind, unsigned* unsat_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      // First, just try to make the instance as is, if it works we are done
      size_t needed_size;
      PhysicalManager* result = builder.create_physical_instance(
          unsat_kind, unsat_index, &needed_size);
      if (footprint != nullptr)
        *footprint = needed_size;
      if ((result != nullptr) || (needed_size == 0))
      {
#ifdef LEGION_DEBUG
        size_t previous =
#endif
            remaining_capacity.fetch_sub(needed_size);
        legion_assert(needed_size <= previous);
        return result;
      }
      GarbageCollector collector(
          collection_lock, manager_lock, runtime->address_space, memory,
          needed_size, capacity, remaining_capacity, collectable_instances);
      while (!collector.collection_complete())
      {
        LgEvent hole_unique_event;
        PhysicalInstance hole_instance = PhysicalInstance::NO_INST;
        const RtEvent collection_done =
            collector.perform_collection(hole_instance, hole_unique_event);
        result = builder.create_physical_instance(
            unsat_kind, unsat_index, nullptr, collection_done, hole_instance,
            hole_unique_event);
        if (result != nullptr)
        {
#ifdef LEGION_DEBUG
          size_t previous =
#endif
              remaining_capacity.fetch_sub(needed_size);
          legion_assert(needed_size <= previous);
          break;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::record_created_instance(
        PhysicalManager* manager, bool acquire, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      // Add references first to prevent races with collection
      if (acquire)
        legion_no_skip_assert(manager->acquire_instance(MAPPING_ACQUIRE_REF));
      // Since we're going to put this in the table add a gc reference
      // which will keep this manager eligible for acquires until the
      // point where we actually end up deleting it
      manager->add_base_gc_ref(MEMORY_MANAGER_REF);
      // If we're setting the priority to min priority and this is the
      // owner then add the reference for the manager
      if ((priority == LEGION_GC_NEVER_PRIORITY) && manager->is_owner())
        manager->add_base_valid_ref(NEVER_GC_REF);
      // Record the manager here as being eligible for collection
      {
        const TreeFieldKey key(manager);
        AutoLock m_lock(manager_lock);
        TreeInstances& insts = current_instances[key];
        legion_assert(insts.find(manager) == insts.end());
        insts[manager] = priority;
        if (priority != LEGION_GC_NEVER_PRIORITY)
          collectable_instances[priority].insert(manager);
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::notify_collected_instances(
        const std::vector<PhysicalManager*>& instances)
    //--------------------------------------------------------------------------
    {
      if (is_owner)
      {
        AutoLock m_lock(manager_lock);
        for (PhysicalManager* manager : instances)
        {
          const TreeFieldKey key(manager);
          TreeFieldInstances::iterator current_finder =
              current_instances.find(key);
          if (current_finder == current_instances.end())
            continue;
          TreeInstances::iterator finder = current_finder->second.find(manager);
          if (finder == current_finder->second.end())
            continue;
          current_finder->second.erase(finder);
          if (current_finder->second.empty())
            current_instances.erase(current_finder);
          if (manager->remove_base_gc_ref(MEMORY_MANAGER_REF))
            delete manager;
        }
      }
      else
      {
        // Send the managers to the owner node to nodify them of the deletion
        NotifyCollectedInstances rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize<size_t>(instances.size());
          for (PhysicalManager* manager : instances)
            rez.serialize(manager->did);
          for (PhysicalManager* manager : instances)
            manager->pack_global_ref(rez);
        }
        rez.dispatch(owner_space);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void NotifyCollectedInstances::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      size_t num_instances;
      derez.deserialize(num_instances);
      std::vector<PhysicalManager*> instances(num_instances);
      std::vector<RtEvent> wait_for;
      for (unsigned idx = 0; idx < num_instances; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        instances[idx] = runtime->find_or_request_instance_manager(did, ready);
        if (ready.exists())
          wait_for.emplace_back(ready);
      }
      MemoryManager* manager = runtime->find_memory_manager(memory);
      if (!wait_for.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(wait_for);
        wait_on.wait();
      }
      manager->notify_collected_instances(instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
        instances[idx]->unpack_global_ref(derez);
    }

    //--------------------------------------------------------------------------
    size_t MemoryManager::query_available_memory(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      return remaining_capacity.load();
    }

    //--------------------------------------------------------------------------
    MemoryPool* MemoryManager::create_memory_pool(
        UniqueID creator_uid, TaskTreeCoordinates& coordinates,
        const PoolBounds& bounds, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        MemoryPool* result = nullptr;
        const RtEvent ready = Runtime::create_rt_user_event();
        CreatePoolRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(creator_uid);
          coordinates.serialize(rez);
          rez.serialize(bounds);
          rez.serialize(&result);
          rez.serialize(safe_for_unbounded_pools);
          rez.serialize(ready);
        }
        rez.dispatch(owner_space);
        ready.wait();
        return result;
      }
      if (bounds.is_bounded())
      {
        // Zero-sized pools are easy to make
        if (bounds.size == 0)
          return new ConcretePool(
              PhysicalInstance::NO_INST, bounds.size, bounds.alignment,
              RtEvent::NO_RT_EVENT, LgEvent::NO_LG_EVENT, this);
        // Creating a normal memory pool so create a task local instance
        // for the requested number of bytes and and then make a pool for it
        Realm::InstanceLayoutGeneric* layout =
            ConcretePool::create_layout(bounds.size, bounds.alignment);
        LgEvent unique_event;
        if (runtime->profiler != nullptr)
          Runtime::rename_event(unique_event);
        RtEvent use_event;
        PhysicalInstance instance = create_task_local_instance(
            creator_uid, coordinates, unique_event, *layout, use_event,
            safe_for_unbounded_pools);
        delete layout;
        if (!instance.exists())
          return nullptr;
        return new ConcretePool(
            instance, bounds.size, bounds.alignment, use_event, unique_event,
            this);
      }
      else
      {
        // Spin wait until we have can mark that there is an unbound pool
        // associated with these particular coordinates
        RtEvent wait_on;
        do {
          wait_on.wait();
          wait_on = RtEvent::NO_RT_EVENT;
          AutoLock m_lock(manager_lock);
          // First check to see if there is an unbounded pool that we
          // are consistent with, if so then we can break out early
          if (outstanding_unbounded_allocations > 0)
          {
            switch (unbounded_pool_scope)
            {
              case LEGION_STRICT_UNBOUNDED_POOL:
                {
                  if (coordinates != unbounded_coordinates)
                  {
                    if (!unbounded_transition_event.exists())
                      unbounded_transition_event =
                          Runtime::create_rt_user_event();
                    if (safe_for_unbounded_pools != nullptr)
                    {
                      legion_assert(!safe_for_unbounded_pools->exists());
                      *safe_for_unbounded_pools = unbounded_transition_event;
                      return nullptr;
                    }
                    wait_on = unbounded_transition_event;
                  }
                  else
                    outstanding_unbounded_allocations++;
                  break;
                }
              case LEGION_INDEX_TASK_UNBOUNDED_POOL:
                {
                  if (!coordinates.same_index_space(unbounded_coordinates) ||
                      (bounds.scope == LEGION_STRICT_UNBOUNDED_POOL))
                  {
                    if (!unbounded_transition_event.exists())
                      unbounded_transition_event =
                          Runtime::create_rt_user_event();
                    if (safe_for_unbounded_pools != nullptr)
                    {
                      legion_assert(!safe_for_unbounded_pools->exists());
                      *safe_for_unbounded_pools = unbounded_transition_event;
                      return nullptr;
                    }
                    wait_on = unbounded_transition_event;
                  }
                  else
                    outstanding_unbounded_allocations++;
                  break;
                }
              case LEGION_PERMISSIVE_UNBOUNDED_POOL:
                {
                  if (bounds.scope != LEGION_PERMISSIVE_UNBOUNDED_POOL)
                  {
                    if (!unbounded_transition_event.exists())
                      unbounded_transition_event =
                          Runtime::create_rt_user_event();
                    if (safe_for_unbounded_pools != nullptr)
                    {
                      legion_assert(!safe_for_unbounded_pools->exists());
                      *safe_for_unbounded_pools = unbounded_transition_event;
                      return nullptr;
                    }
                    wait_on = unbounded_transition_event;
                  }
                  else
                    outstanding_unbounded_allocations++;
                  break;
                }
              default:
                std::abort();
            }
          }
          else if (
              !pending_allocation_attempts.empty() ||
              (outstanding_task_local_allocations > 0))
          {
            // If there are other outstanding allocations we need to wait
            // for them to finish before we can start an unbounded allocation
            if (!unbounded_transition_event.exists())
              unbounded_transition_event = Runtime::create_rt_user_event();
            wait_on = unbounded_transition_event;
          }
          else
          {
            // If there are no outstanding allocations then we can start
            // a new unbounded allocation
            outstanding_unbounded_allocations++;
            unbounded_pool_scope = bounds.scope;
            unbounded_coordinates = coordinates;
          }
        } while (wait_on.exists());
        return new UnboundPool(this, bounds.scope, coordinates, bounds.size);
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_concrete_pool(
        RtEvent done,
        const std::map<PhysicalInstance, std::pair<RtEvent, LgEvent>>&
            backing_instances)
    //--------------------------------------------------------------------------
    {
      if (is_owner)
      {
        // Iterate over all the remaining backing stores delete them
        // once the done event is triggered
        size_t total_size = 0;
        for (const std::pair<
                 const PhysicalInstance, std::pair<RtEvent, LgEvent>>& it :
             backing_instances)
        {
          total_size += it.first.get_layout()->bytes_used;
          it.first.destroy(Runtime::merge_events(it.second.first, done));
        }
        update_remaining_capacity(total_size);
      }
      else
      {
        // Have to send a message to the owner node to do the update
        ReleasePoolRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize<bool>(true);  // concrete
          rez.serialize<size_t>(backing_instances.size());
          for (const std::pair<
                   const PhysicalInstance, std::pair<RtEvent, LgEvent>>& it :
               backing_instances)
          {
            rez.serialize(it.first);
            rez.serialize(Runtime::merge_events(it.second.first, done));
          }
        }
        rez.dispatch(owner_space);
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_unbound_pool(void)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // Have to send a message to the owner to do the update
        ReleasePoolRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize<bool>(false);  // not concrete
        }
        rez.dispatch(owner_space);
        return;
      }
      AutoLock m_lock(manager_lock);
      legion_assert(outstanding_unbounded_allocations > 0);
      if (--outstanding_unbounded_allocations > 0)
        return;
      // Wake up any waiting allocations attempts
      if (!pending_allocation_attempts.empty())
      {
        if (unbounded_pool_scope == LEGION_STRICT_UNBOUNDED_POOL)
        {
          // If the coordinates match it is already running so only
          // trigger the event if coordinates do not match
          const std::pair<RtUserEvent, const TaskTreeCoordinates*>& front =
              pending_allocation_attempts.front();
          if (*front.second != unbounded_coordinates)
          {
            legion_assert(front.first.exists());
            Runtime::trigger_event(front.first);
          }
        }
        else if (unbounded_pool_scope == LEGION_INDEX_TASK_UNBOUNDED_POOL)
        {
          // If the coordinates are from the same index space then it
          // is running already so only trigger if they are not
          const std::pair<RtUserEvent, const TaskTreeCoordinates*>& front =
              pending_allocation_attempts.front();
          if (!front.second->same_index_space(unbounded_coordinates))
          {
            legion_assert(front.first.exists());
            Runtime::trigger_event(front.first);
          }
        }
      }
      unbounded_pool_scope = LEGION_BOUNDED_POOL;
      unbounded_coordinates.clear();
      if (unbounded_transition_event.exists())
      {
        Runtime::trigger_event(unbounded_transition_event);
        unbounded_transition_event = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreatePoolRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      UniqueID creator_uid;
      derez.deserialize(creator_uid);
      TaskTreeCoordinates coordinates;
      coordinates.deserialize(derez);
      PoolBounds bounds;
      derez.deserialize(bounds);
      MemoryPool** result;
      derez.deserialize(result);
      RtEvent* remote_safe_for_unbounded_pools;
      derez.deserialize(remote_safe_for_unbounded_pools);
      RtUserEvent ready;
      derez.deserialize(ready);

      MemoryManager* manager = runtime->find_memory_manager(memory);
      RtEvent safe_for_unbounded_pools;
      MemoryPool* pool = manager->create_memory_pool(
          creator_uid, coordinates, bounds,
          (remote_safe_for_unbounded_pools == nullptr) ?
              nullptr :
              &safe_for_unbounded_pools);
      if ((pool != nullptr) || ((remote_safe_for_unbounded_pools != nullptr) &&
                                safe_for_unbounded_pools.exists()))
      {
        CreatePoolResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(result);
          if (pool != nullptr)
            pool->serialize(rez);
          else
            MemoryPool::pack_null_pool(rez);
          rez.serialize(remote_safe_for_unbounded_pools);
          if (remote_safe_for_unbounded_pools != nullptr)
            rez.serialize(safe_for_unbounded_pools);
          rez.serialize(ready);
        }
        rez.dispatch(source);
        delete pool;
      }
      else
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreatePoolResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      MemoryPool** result;
      derez.deserialize(result);
      *result = MemoryPool::deserialize(derez);
      RtEvent* safe_for_unbounded_pools;
      derez.deserialize(safe_for_unbounded_pools);
      if (safe_for_unbounded_pools != nullptr)
        derez.deserialize(*safe_for_unbounded_pools);
      RtUserEvent ready;
      derez.deserialize(ready);
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReleasePoolRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      MemoryManager* manager = runtime->find_memory_manager(memory);
      bool concrete;
      derez.deserialize<bool>(concrete);
      if (concrete)
      {
        size_t num_instances;
        derez.deserialize(num_instances);
        size_t total_size = 0;
        for (unsigned idx = 0; idx < num_instances; idx++)
        {
          PhysicalInstance instance;
          derez.deserialize(instance);
          total_size += instance.get_layout()->bytes_used;
          RtEvent done;
          derez.deserialize(done);
          instance.destroy(done);
        }
        manager->update_remaining_capacity(total_size);
      }
      else
        manager->release_unbound_pool();
    }

    //--------------------------------------------------------------------------
    uint64_t MemoryManager::order_collective_unbounded_pools(SingleTask* task)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      legion_assert(collective_tasks.find(task) == collective_tasks.end());
      const uint64_t lamport_clock = collective_lamport_clock++;
      collective_tasks.insert(
          std::make_pair(task, CollectiveState(lamport_clock)));
      return lamport_clock;
    }

    //--------------------------------------------------------------------------
    RtEvent MemoryManager::finalize_collective_unbounded_pools_order(
        SingleTask* task, uint64_t max_lamport_clock)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      std::map<SingleTask*, CollectiveState>::iterator finder =
          collective_tasks.find(task);
      legion_assert(finder != collective_tasks.end());
      legion_assert(!finder->second.max);
      legion_assert(finder->second.lamport_clock <= max_lamport_clock);
      if (collective_lamport_clock <= max_lamport_clock)
        collective_lamport_clock = max_lamport_clock + 1;
      finder->second.lamport_clock = max_lamport_clock;
      finder->second.max = true;
      ready_collective_tasks++;
      if (outstanding_collective_tasks == 0)
      {
        start_next_collective_unbounded_pools_task();
        // Check to see if it started
        finder = collective_tasks.find(task);
        if (finder == collective_tasks.end())
          return RtEvent::NO_RT_EVENT;
      }
      // Unable to start now so make an event to defer it
      legion_assert(!finder->second.ready_event.exists());
      finder->second.ready_event = Runtime::create_rt_user_event();
      return finder->second.ready_event;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::end_collective_unbounded_pools_task(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      legion_assert(outstanding_collective_tasks > 0);
      if ((--outstanding_collective_tasks == 0) && (ready_collective_tasks > 0))
        start_next_collective_unbounded_pools_task();
    }

    //--------------------------------------------------------------------------
    void MemoryManager::start_next_collective_unbounded_pools_task(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!collective_tasks.empty());
      legion_assert(outstanding_collective_tasks == 0);
      legion_assert(ready_collective_tasks > 0);
      // See if we can prove that there is a task that is safe to start
      uint64_t min_next = std::numeric_limits<uint64_t>::max();
      uint64_t min_pending = std::numeric_limits<uint64_t>::max();
      TaskTreeCoordinates next_coords;
      std::vector<SingleTask*> next_tasks;
      for (const std::pair<SingleTask* const, CollectiveState>& it :
           collective_tasks)
      {
        if (it.second.max)
        {
          if (!next_tasks.empty())
          {
            // Compare the lamport clocks
            if (it.second.lamport_clock < min_next)
            {
              next_tasks.clear();
              next_tasks.emplace_back(it.first);
              next_coords.clear();
              min_next = it.second.lamport_clock;
            }
            else if (min_next == it.second.lamport_clock)
            {
              // Very bad case, same min of max all-reduce of clocks
              // Resolve this conflict based on task tree coordinates
              TaskTreeCoordinates it_coords;
              if (next_coords.empty())
                next_tasks.back()->compute_task_tree_coordinates(next_coords);
              it.first->compute_task_tree_coordinates(it_coords);
              // See if these are the same index space
              if (next_coords.same_index_space(it_coords))
              {
                next_tasks.emplace_back(it.first);
                continue;
              }
              const size_t lower_bound =
                  std::min(next_coords.size(), it_coords.size());
              bool equal = true;
              for (unsigned idx = 0; idx < lower_bound; idx++)
              {
                const ContextCoordinate& c1 = next_coords[idx];
                const ContextCoordinate& c2 = it_coords[idx];
                if (c1.context_index == c2.context_index)
                {
                  if (c2.index_point < c1.index_point)
                  {
                    next_tasks.clear();
                    next_tasks.emplace_back(it.first);
                    next_coords.swap(it_coords);
                  }
                  else if (c1.index_point == c2.index_point)
                    continue;
                }
                else if (c2.context_index < c1.context_index)
                {
                  next_tasks.clear();
                  next_tasks.emplace_back(it.first);
                  next_coords.swap(it_coords);
                }
                equal = false;
                break;
              }
              if (equal)
              {
                legion_assert(next_coords.size() != it_coords.size());
                if (it_coords.size() < next_coords.size())
                {
                  next_tasks.clear();
                  next_tasks.emplace_back(it.first);
                  next_coords.swap(it_coords);
                }
              }
            }
          }
          else
          {
            next_tasks.emplace_back(it.first);
            min_next = it.second.lamport_clock;
          }
        }
        else if (it.second.lamport_clock < min_pending)
          min_pending = it.second.lamport_clock;
      }
      // If all the pending tasks with lamport clocks are
      // larger than our max lamport clock of the next task
      // to launch then we know they won't ever come before it
      // so we can issue our next task now, otherwise we'll need
      // to wait until those pending lamport clocks are done
      if (min_next < min_pending)
      {
        // Start all the next tasks
        for (SingleTask* next_task : next_tasks)
        {
          std::map<SingleTask*, CollectiveState>::iterator finder =
              collective_tasks.find(next_task);
          legion_assert(finder != collective_tasks.end());
          if (finder->second.ready_event.exists())
            Runtime::trigger_event(finder->second.ready_event);
          collective_tasks.erase(finder);
        }
        legion_assert(outstanding_collective_tasks == 0);
        legion_assert(next_tasks.size() <= ready_collective_tasks);
        ready_collective_tasks -= next_tasks.size();
        outstanding_collective_tasks = next_tasks.size();
      }
    }

    //--------------------------------------------------------------------------
    PhysicalInstance MemoryManager::create_task_local_instance(
        UniqueID creator_uid, const TaskTreeCoordinates& coordinates,
        LgEvent unique_event, const Realm::InstanceLayoutGeneric& layout,
        RtEvent& use_event, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      GarbageCollector* collector = nullptr;
      PhysicalInstance instance = PhysicalInstance::NO_INST;
      // We don't need to serialize this with respect to region allocations
      // because these instances are not subject to find_and_create calls
      // but we do need to make sure there are no outstanding unbounded
      // allocations occurring when we do this (unless it is coming from
      // an unbound pool in which case we can ignore this check
      RtEvent wait_on;
      {
        AutoLock m_lock(manager_lock);
        switch (unbounded_pool_scope)
        {
          case LEGION_BOUNDED_POOL:
          case LEGION_PERMISSIVE_UNBOUNDED_POOL:
            break;
          case LEGION_STRICT_UNBOUNDED_POOL:
            {
              if (coordinates != unbounded_coordinates)
              {
                if (!unbounded_transition_event.exists())
                  unbounded_transition_event = Runtime::create_rt_user_event();
                if (safe_for_unbounded_pools != nullptr)
                {
                  legion_assert(!safe_for_unbounded_pools->exists());
                  *safe_for_unbounded_pools = unbounded_transition_event;
                  return PhysicalInstance::NO_INST;
                }
                wait_on = unbounded_transition_event;
              }
              break;
            }
          case LEGION_INDEX_TASK_UNBOUNDED_POOL:
            {
              if (!unbounded_coordinates.same_index_space(coordinates))
              {
                if (!unbounded_transition_event.exists())
                  unbounded_transition_event = Runtime::create_rt_user_event();
                if (safe_for_unbounded_pools != nullptr)
                {
                  legion_assert(!safe_for_unbounded_pools->exists());
                  *safe_for_unbounded_pools = unbounded_transition_event;
                  return PhysicalInstance::NO_INST;
                }
                wait_on = unbounded_transition_event;
              }
              break;
            }
          default:
            std::abort();
        }
        outstanding_task_local_allocations++;
      }
      if (wait_on.exists())
        wait_on.wait();
      do {
        LgEvent hole_unique_event;
        RtEvent alloc_precondition;
        PhysicalInstance hole_instance = PhysicalInstance::NO_INST;
        if (collector != nullptr)
          alloc_precondition =
              collector->perform_collection(hole_instance, hole_unique_event);
        Realm::ProfilingRequestSet requests;
        legion_assert(!instance.exists());
#ifndef LEGION_MALLOC_INSTANCES
        TaskLocalInstanceAllocator allocator(unique_event);
        ProfilingResponseBase base(&allocator, creator_uid, false);
        Realm::ProfilingRequest& req = requests.add_request(
            runtime->find_local_group(), LG_LEGION_PROFILING_ID, &base,
            sizeof(base), LG_RESOURCE_PRIORITY);
        req.add_measurement<
            Realm::ProfilingMeasurements::InstanceAllocResult>();
        if (runtime->profiler != nullptr)
          runtime->profiler->add_inst_request(
              requests, creator_uid, unique_event);
        // Check to see if we have a hole instance, if we do then redistrict
        if (hole_instance.exists())
          use_event = RtEvent(hole_instance.redistrict(
              instance, &layout, requests, alloc_precondition));
        else
          use_event = RtEvent(PhysicalInstance::create_instance(
              instance, memory, layout, requests, alloc_precondition));
        if (allocator.succeeded())
        {
          // Only record this if we succeeded in the allocation
          if (implicit_profiler != nullptr)
          {
            if (hole_instance.exists())
              implicit_profiler->record_instance_redistrict(
                  use_event, hole_unique_event, unique_event,
                  alloc_precondition);
            else
              implicit_profiler->record_instance_ready(
                  use_event, unique_event, alloc_precondition);
          }
#ifdef LEGION_DEBUG
          size_t previous =
#endif
              remaining_capacity.fetch_sub(layout.bytes_used);
          legion_assert(layout.bytes_used <= previous);
          break;
        }
        else if (instance.exists())
        {
          instance.destroy();
          instance = PhysicalInstance::NO_INST;
        }
#else
        use_event = allocate_legion_instance(layout, requests, instance);
        if (instance.exists())
        {
          // Only record this if we succeeded in the allocation
          if (use_event.exists() && (implicit_profiler != NULL))
            implicit_profiler->record_instance_ready(
                use_event, unique_event, alloc_precondition);
#ifdef LEGION_DEBUG
          size_t previous =
#endif
              remaining_capacity.fetch_sub(layout.bytes_used);
          legion_assert(layout.bytes_used <= previous);
          break;
        }
#endif
        if (collector == nullptr)
          collector = new GarbageCollector(
              collection_lock, manager_lock, runtime->address_space, memory,
              layout.bytes_used, capacity, remaining_capacity,
              collectable_instances);
      } while (!collector->collection_complete());
      if (collector != nullptr)
        delete collector;
      // Retake the lock and mark that our allocation is done
      AutoLock m_lock(manager_lock);
      legion_assert(outstanding_task_local_allocations > 0);
      outstanding_task_local_allocations--;
      if (unbounded_transition_event.exists() &&
          pending_allocation_attempts.empty() &&
          (outstanding_task_local_allocations == 0))
      {
        Runtime::trigger_event(unbounded_transition_event);
        unbounded_transition_event = RtUserEvent::NO_RT_USER_EVENT;
      }
      return instance;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::free_task_local_instance(
        PhysicalInstance instance, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      const size_t size = instance.get_layout()->bytes_used;
#ifdef LEGION_DEBUG
      const size_t previous =
#endif
          remaining_capacity.fetch_add(size);
      legion_assert((previous + size) <= capacity);
      instance.destroy(precondition);
    }

    //--------------------------------------------------------------------------
    size_t MemoryManager::compute_future_alignment(size_t size) const
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      // Default max alignment is 32 bytes
      size_t max_alignment = 32;
      const Memory::Kind kind = memory.kind();
      // See if this is a GPU memory, if it is then we increase the maximum
      // alignment up to 128 bytes since GPUs tend to like that
      if ((kind == Memory::GPU_FB_MEM) || (kind == Memory::GPU_MANAGED_MEM) ||
          (kind == Memory::GPU_DYNAMIC_MEM))
        max_alignment = 128;
      static_assert((sizeof(size_t) == 4) || (sizeof(size_t) == 8));
      // Round up to the nearest power of 2
      size--;
      for (unsigned i = 1; i <= (4 * sizeof(size)); i <<= 1) size |= size >> i;
      size++;
      return std::min<size_t>(size, max_alignment);
    }

    //--------------------------------------------------------------------------
    FutureInstance* MemoryManager::create_future_instance(
        UniqueID creator_uid, const TaskTreeCoordinates& coordinates,
        size_t size, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        std::atomic<FutureInstance*> result(nullptr);
        // Send a message to the owner to do this and wait for the result
        const RtUserEvent wait_on = Runtime::create_rt_user_event();
        CreateFutureInstanceRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(&result);
          rez.serialize(wait_on);
          rez.serialize(creator_uid);
          coordinates.serialize(rez);
          rez.serialize(size);
          rez.serialize(safe_for_unbounded_pools);
        }
        rez.dispatch(owner_space);
        wait_on.wait();
        return result.load();
      }
      // Do a quick check to see if we can handle the easy case
      if ((size <= LEGION_MAX_RETURN_SIZE) &&
          (memory == runtime->runtime_system_memory))
      {
#ifdef __GNUC__
#if __GNUC__ >= 11
        // GCC is dumb and thinks we need to initialize the malloc buffer
        // before we pass it into the create local call, which we
        // obviously don't need to do, so tell the compiler to shut up
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
        // Special case where we can just allocate the buffer locally
        return new FutureInstance(
            malloc(size), size, true /*external*/, true /*own allocation*/);
#ifdef __GNUC__
#if __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif
#endif
      }
      // Create the layout description for this instance
      const std::vector<Realm::FieldID> fids(1, 0 /*field id*/);
      const std::vector<size_t> sizes(1, size);
      const int dim_order[1] = {0};
      const Realm::Point<1, coord_t> zero(0);
      const Realm::InstanceLayoutConstraints constraints(fids, sizes, 1);
      const Realm::IndexSpace<1, coord_t> rect_space(
          Realm::Rect<1, coord_t>(zero, zero));
      Realm::InstanceLayoutGeneric* ilg =
          Realm::InstanceLayoutGeneric::choose_instance_layout<1, coord_t>(
              rect_space, constraints, dim_order);
      // Create the layout for the future
      ilg->alignment_reqd = compute_future_alignment(size);
      LgEvent unique_event;
      if ((spy_logging_level > NO_SPY_LOGGING) ||
          (runtime->profiler != nullptr))
        Runtime::rename_event(unique_event);
      RtEvent use_event;
      PhysicalInstance instance = create_task_local_instance(
          creator_uid, coordinates, unique_event, *ilg, use_event,
          safe_for_unbounded_pools);
      delete ilg;
      if (!instance.exists())
        return nullptr;
      return new FutureInstance(
          nullptr /*data*/, size, false /*external*/, true /*own allocation*/,
          unique_event, instance, Processor::NO_PROC, use_event);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::free_future_instance(
        PhysicalInstance inst, size_t size, RtEvent free_event)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
      {
        // Send this to the owner node
        FreeFutureInstance rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(inst);
          rez.serialize(size);
          rez.serialize(free_event);
        }
        rez.dispatch(owner_space);
        return;
      }
      else
      {
        // perform the deferred deletion on this instance
#ifdef LEGION_DEBUG
        const size_t previous =
#endif
            remaining_capacity.fetch_add(size);
        legion_assert((previous + size) <= capacity);
        inst.destroy(free_event);
      }
    }

    //--------------------------------------------------------------------------
    MemoryManager::TaskLocalInstanceAllocator::TaskLocalInstanceAllocator(
        LgEvent unique)
      : ready(Runtime::create_rt_user_event()), unique_event(unique),
        caller_fevent(implicit_fevent), success(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MemoryManager::TaskLocalInstanceAllocator::TaskLocalInstanceAllocator(
        TaskLocalInstanceAllocator&& rhs) noexcept
      : ready(rhs.ready), unique_event(rhs.unique_event),
        caller_fevent(implicit_fevent), success(rhs.success)
    //--------------------------------------------------------------------------
    {
      rhs.ready = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::TaskLocalInstanceAllocator::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::InstanceAllocResult>());
      Realm::ProfilingMeasurements::InstanceAllocResult result;
      result.success = false;  // Need this to avoid compiler warnings
      legion_no_skip_assert(
          response.get_measurement<
              Realm::ProfilingMeasurements::InstanceAllocResult>(result));
      success = result.success;
      failed_alloc = !success;
      if (failed_alloc)
        fevent = caller_fevent;
      else
        fevent = unique_event;
      // Can't read anything after trigger the event as the object
      // might be deleted after we do that
      Runtime::trigger_event(ready);
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent MemoryManager::attach_external_instance(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      legion_assert(manager->is_external_instance());
      if (!is_owner)
      {
        // Send a message to the owner node to do the record
        RtUserEvent result = Runtime::create_rt_user_event();
        ExternalAttachRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(manager->did);
          rez.serialize(result);
        }
        rez.dispatch(manager->owner_space);
        return result;
      }
      legion_assert(is_owner);
      // Since we're going to put this in the table add a reference
      manager->add_base_resource_ref(MEMORY_MANAGER_REF);
      {
        const TreeFieldKey key(manager);
        AutoLock m_lock(manager_lock);
        TreeInstances& insts = current_instances[key];
        legion_assert(insts.find(manager) == insts.end());
        insts[manager] = LEGION_GC_NEVER_PRIORITY;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::detach_external_instance(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      legion_assert(manager->is_external_instance());
      if (!is_owner)
      {
        // Send a message to the owner node to do the deletion
        ExternalDetachRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(manager->did);
          manager->pack_valid_ref(rez);
        }
        rez.dispatch(manager->owner_space);
      }
      else
      {
        // Tell the manager that it can perform its deletion
        manager->perform_deletion(runtime->address_space);
      }
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::is_visible_memory(Memory other)
    //--------------------------------------------------------------------------
    {
      if (other == memory)
        return true;
      {
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        if (!visible_memories.empty())
          return (visible_memories.find(other) != visible_memories.end());
      }
      // Do the query while not holding the lock
      Machine::MemoryQuery vis_mems(runtime->machine);
      vis_mems.has_affinity_to(memory);
      AutoLock m_lock(manager_lock);
      if (visible_memories.empty())
        for (Machine::MemoryQuery::iterator it = vis_mems.begin();
             it != vis_mems.end(); it++)
          visible_memories.insert(*it);
      return (visible_memories.find(other) != visible_memories.end());
    }

    //--------------------------------------------------------------------------
    void MemoryManager::free_external_allocation(uintptr_t ptr, size_t size)
    //--------------------------------------------------------------------------
    {
      switch (memory.kind())
      {
        case Memory::SYSTEM_MEM:
        case Memory::SOCKET_MEM:
          {
            free((void*)ptr);
            break;
          }
        case Memory::REGDMA_MEM:
          {
            munlock((void*)ptr, size);
            free((void*)ptr);
            break;
          }
#ifdef LEGION_USE_CUDA
#define CHECK_CUDA(cmd)                                                \
  do {                                                                 \
    CUresult ret = (cmd);                                              \
    if (ret != CUDA_SUCCESS)                                           \
    {                                                                  \
      const char *name, *str;                                          \
      cuGetErrorName(ret, &name);                                      \
      cuGetErrorString(ret, &str);                                     \
      fprintf(stderr, "CU: %s = %d (%s): %s\n", #cmd, ret, name, str); \
      abort();                                                         \
    }                                                                  \
  } while (false)
        case Memory::GPU_FB_MEM:
          {
            CHECK_CUDA(cuMemFree((CUdeviceptr)ptr));
            break;
          }
        case Memory::Z_COPY_MEM:
          {
            CHECK_CUDA(cuMemFreeHost((void*)ptr));
            break;
          }
#undef CHECK_CUDA
#endif
#ifdef LEGION_USE_HIP
#define CHECK_HIP(cmd)                                                  \
  do {                                                                  \
    hipError_t ret = (cmd);                                             \
    if (ret != hipSuccess)                                              \
    {                                                                   \
      const char *name, *str;                                           \
      name = hipGetErrorName(ret);                                      \
      str = hipGetErrorString(ret);                                     \
      fprintf(stderr, "HIP: %s = %d (%s): %s\n", #cmd, ret, name, str); \
      abort();                                                          \
    }                                                                   \
  } while (false)
        case Memory::GPU_FB_MEM:
          {
            CHECK_HIP(hipFree((void*)ptr));
            break;
          }
        case Memory::Z_COPY_MEM:
          {
            CHECK_HIP(hipHostFree((void*)ptr));
            break;
          }
#undef CHECK_HIP
#endif
        default:
          {
            Fatal error;
            error << "Unsupported memory kind " << memory.kind();
            error.raise();
          }
      }
    }

#ifdef LEGION_MALLOC_INSTANCES
    //--------------------------------------------------------------------------
    RtEvent MemoryManager::allocate_legion_instance(
        const Realm::InstanceLayoutGeneric& layout,
        const Realm::ProfilingRequestSet& requests, PhysicalInstance& instance,
        LgEvent unique_event, bool needs_deferral)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      RtEvent result;
      const size_t footprint = layout.bytes_used;
      switch (memory.kind())
      {
        case Memory::SYSTEM_MEM:
        case Memory::SOCKET_MEM:
          {
            void* ptr = nullptr;
            if (footprint > 0)
            {
              if (posix_memalign(&ptr, 32 /*alignment*/, footprint))
                return result;  // failed
            }
            const Realm::ExternalMemoryResource resource(
                (uintptr_t)ptr, footprint, false /*read only*/);
            result = RtEvent(PhysicalInstance::create_external_instance(
                instance, resource.suggested_memory(), layout, resource,
                requests));
            break;
          }
        case Memory::REGDMA_MEM:
          {
            void* ptr = nullptr;
            if (footprint > 0)
            {
              if (posix_memalign(&ptr, 32 /*alignment*/, footprint))
                return result;  // failed
              mlock(ptr, footprint);
            }
            const Realm::ExternalMemoryResource resource(
                (uintptr_t)ptr, footprint, false /*read only*/);
            result = RtEvent(PhysicalInstance::create_external_instance(
                instance, resource.suggested_memory(), layout, resource,
                requests));
            break;
          }
#ifdef LEGION_USE_CUDA
        case Memory::Z_COPY_MEM:
        case Memory::GPU_FB_MEM:
          {
            if (needs_deferral)
            {
              MallocInstanceArgs args(
                  this, layout.clone(), &requests, &instance, unique_event);
              const RtEvent wait_on = runtime->issue_application_processor_task(
                  args, LG_LATENCY_WORK_PRIORITY, local_gpu);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
              return result;
            }
            else
            {
              // Use the driver API here to avoid the CUDA hijack
              if (memory.kind() == Memory::GPU_FB_MEM)
              {
                CUdeviceptr ptr;
                if ((footprint > 0) &&
                    (cuMemAlloc(&ptr, footprint) != CUDA_SUCCESS))
                  return result;
                CUdevice device;
                if (cuCtxGetDevice(&device) != CUDA_SUCCESS)
                  return result;
                const Realm::ExternalCudaMemoryResource resource(
                    device, (uintptr_t)ptr, footprint, false /*read only*/);
                result = RtEvent(PhysicalInstance::create_external_instance(
                    instance, resource.suggested_memory(), layout, resource,
                    requests));
              }
              else
              {
                void* ptr = nullptr;
                if ((footprint > 0) &&
                    (cuMemHostAlloc(
                         &ptr, footprint,
                         CU_MEMHOSTALLOC_PORTABLE |
                             CU_MEMHOSTALLOC_DEVICEMAP) != CUDA_SUCCESS))
                  return result;
                // Check that the device pointer is the same as the host
                CUdeviceptr gpuptr;
                if (cuMemHostGetDevicePointer(&gpuptr, ptr, 0) != CUDA_SUCCESS)
                  return result;
                if (ptr != (void*)gpuptr)
                  return result;
                const Realm::ExternalCudaPinnedHostResource resource(
                    (uintptr_t)ptr, footprint, false /*read only*/);
                result = RtEvent(PhysicalInstance::create_external_instance(
                    instance, resource.suggested_memory(), layout, resource,
                    requests));
              }
            }
            break;
          }
#endif
#ifdef LEGION_USE_HIP
        case Memory::Z_COPY_MEM:
        case Memory::GPU_FB_MEM:
          {
            if (needs_deferral)
            {
              MallocInstanceArgs args(
                  this, layout.clone(), &requests, &instance, unique_event);
              const RtEvent wait_on = runtime->issue_application_processor_task(
                  args, LG_LATENCY_WORK_PRIORITY, local_gpu);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
              return result;
            }
            else
            {
              // Use the driver API here to avoid the CUDA hijack
              if (memory.kind() == Memory::GPU_FB_MEM)
              {
                hipDeviceptr_t ptr;
                if ((footprint > 0) &&
                    (hipMalloc((void**)&ptr, footprint) != hipSuccess))
                  return result;
                int device;
                if (hipGetDevice(&device) != hipSuccess)
                  return result;
                const Realm::ExternalHipMemoryResource resource(
                    device, (uintptr_t)ptr, footprint, false /*read only*/);
                result = RtEvent(PhysicalInstance::create_external_instance(
                    instance, resource.suggested_memory(), layout, resource,
                    requests));
              }
              else
              {
                void* ptr = nullptr;
                if ((footprint > 0) &&
                    (hipHostMalloc(
                         &ptr, footprint,
                         hipHostMallocPortable | hipHostMallocMapped) !=
                     hipSuccess))
                  return result;
                hipDeviceptr_t gpuptr;
                if (hipHostGetDevicePointer((void**)&gpuptr, ptr, 0) !=
                    hipSuccess)
                  return result;
                if (ptr != (void*)gpuptr)
                  return result;
                const Realm::ExternalHipPinnedHostResource resource(
                    (uintptr_t)ptr, footprint, false /*read only*/);
                result = RtEvent(PhysicalInstance::create_external_instance(
                    instance, resource.suggested_memory(), layout, resource,
                    requests));
              }
            }
            break;
          }
#endif
        default:
          {
            Fatal error;
            error << "Unsupported memory kind for LEGION_MALLOC_INSTANCES "
                  << memory.kind();
            error.raise();
          }
      }
      if (instance.exists())
      {
        if (result.exists() && (implicit_profiler != nullptr))
          implicit_profiler->record_instance_ready(result, unique_event);
        AutoLock m_lock(manager_lock);
        legion_assert(allocations.find(instance) == allocations.end());
        allocations[instance] = footprint;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::record_legion_instance(
        InstanceManager* man, PhysicalInstance instance)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      AutoLock m_lock(manager_lock);
      legion_assert(legion_instances.find(man) == legion_instances.end());
      legion_instances[man] = instance;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::free_legion_instance(
        InstanceManager* man, RtEvent defer)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      PhysicalInstance instance;
      {
        AutoLock m_lock(manager_lock);
        std::map<InstanceManager*, PhysicalInstance>::iterator finder =
            legion_instances.find(man);
        legion_assert(finder != legion_instances.end());
        instance = finder->second;
        legion_instances.erase(finder);
      }
      free_legion_instance(defer, instance);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::free_legion_instance(
        RtEvent defer, PhysicalInstance instance, bool needs_defer)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner);
      legion_assert(instance.exists());
      size_t size;
      {
        AutoLock m_lock(manager_lock);
        if (defer.exists() && !defer.has_triggered())
        {
          std::map<RtEvent, PhysicalInstance>::iterator finder =
              pending_collectables.find(defer);
          if (finder == pending_collectables.end())
          {
            FreeInstanceArgs args(this, instance);
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
            if (local_gpu.exists())
              runtime->issue_application_processor_task(
                  args, LG_LOW_PRIORITY, local_gpu, defer);
            else
              runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY, defer);
#else
            runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY, defer);
#endif
          }
          else
            finder->second = instance;
          return;
        }
        std::map<PhysicalInstance, size_t>::iterator finder =
            allocations.find(instance);
        legion_assert(finder != allocations.end());
        size = finder->second;
        allocations.erase(finder);
      }
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
      if (needs_defer && (size > 0) &&
          ((memory.kind() == Memory::Z_COPY_MEM) ||
           (memory.kind() == Memory::GPU_FB_MEM)))
      {
        // Put the allocation back in for when we go to look
        // for it on the second pass
        {
          AutoLock m_lock(manager_lock);
          legion_assert(allocations.find(instance) == allocations.end());
          allocations[instance] = size;
        }
        FreeInstanceArgs args(this, instance);
        runtime->issue_application_processor_task(
            args, LG_LOW_PRIORITY, local_gpu, defer);
        return;
      }
#endif
      if (size > 0)
        free_external_allocation(
            (uintptr_t)instance.pointer_untyped(0, 0), size);
      instance.destroy(defer);
    }
#endif

    //--------------------------------------------------------------------------
    void MemoryManager::MallocInstanceArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_MALLOC_INSTANCES
      const RtEvent ready = manager->allocate_legion_instance(
          layout, *requests, *instance, unique_event, false /*needs defer*/);
      delete layout;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#else
      std::abort();
#endif
    }

    //--------------------------------------------------------------------------
    void MemoryManager::FreeInstanceArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_MALLOC_INSTANCES
      manager->free_legion_instance(
          RtEvent::NO_RT_EVENT, instance, false /*needs defer*/);
#else
      std::abort();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager* manager = runtime->find_memory_manager(target_memory);
      manager->process_instance_request(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager* manager = runtime->find_memory_manager(target_memory);
      manager->process_instance_response(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalAttachRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      DistributedID did;
      derez.deserialize(did);
      RtEvent manager_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(did, manager_ready);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      MemoryManager* memory_manager =
          runtime->find_memory_manager(target_memory);
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();
      RtEvent local_done = memory_manager->attach_external_instance(manager);
      Runtime::trigger_event(done_event, local_done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalDetachRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      DistributedID did;
      derez.deserialize(did);
      RtEvent manager_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(did, manager_ready);
      MemoryManager* memory_manager =
          runtime->find_memory_manager(target_memory);
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();
      memory_manager->detach_external_instance(manager);
      manager->unpack_valid_ref(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreateFutureInstanceRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      std::atomic<FutureInstance*>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      UniqueID uid;
      derez.deserialize(uid);
      TaskTreeCoordinates coordinates;
      coordinates.deserialize(derez);
      size_t size;
      derez.deserialize(size);
      RtEvent* remote_safe_for_unbounded_pools;
      derez.deserialize(remote_safe_for_unbounded_pools);

      MemoryManager* manager = runtime->find_memory_manager(memory);
      RtEvent safe_for_unbounded_pools;
      FutureInstance* result = manager->create_future_instance(
          uid, coordinates, size,
          (remote_safe_for_unbounded_pools == nullptr) ?
              nullptr :
              &safe_for_unbounded_pools);
      if ((result != nullptr) ||
          ((remote_safe_for_unbounded_pools != nullptr) &&
           safe_for_unbounded_pools.exists()))
      {
        CreateFutureInstanceResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          if (result != nullptr)
            result->pack_instance(
                rez, ApEvent::NO_AP_EVENT, true /*pack ownership*/,
                false /*allow by value*/);
          else
            FutureInstance::pack_null(rez);
          rez.serialize(remote_safe_for_unbounded_pools);
          if (remote_safe_for_unbounded_pools != nullptr)
            rez.serialize(safe_for_unbounded_pools);
          rez.serialize(done);
        }
        rez.dispatch(source);
        delete result;
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreateFutureInstanceResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<FutureInstance*>* target;
      derez.deserialize(target);
      target->store(FutureInstance::unpack_instance(derez));
      RtEvent* safe_for_unbounded_pools;
      derez.deserialize(safe_for_unbounded_pools);
      if (safe_for_unbounded_pools != nullptr)
        derez.deserialize(*safe_for_unbounded_pools);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FreeFutureInstance::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory memory;
      derez.deserialize(memory);
      PhysicalInstance instance;
      derez.deserialize(instance);
      size_t size;
      derez.deserialize(size);
      RtEvent free_event;
      derez.deserialize(free_event);
      MemoryManager* manager = runtime->find_memory_manager(memory);
      manager->free_future_instance(instance, size, free_event);
    }

  }  // namespace Internal
}  // namespace Legion
