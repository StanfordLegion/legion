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

#ifndef __LEGION_MEMORY_MANAGER_H__
#define __LEGION_MEMORY_MANAGER_H__

#include "legion/tools/profiler.h"
#include "legion/utilities/coordinates.h"

namespace Legion {
  namespace Internal {

    /**
     * \class MemoryPool
     * A memory pool abstracts an interface for performing immediate
     * memory allocations without going through Realm's instance allocation
     * pathway. This is necessary for performing allocations that happen
     * during the execution state of the pipeline and therefore we cannot
     * have such allocations racing with mapping allocations.
     */
    class MemoryPool {
    public:
      MemoryPool(size_t alignment) : max_alignment(alignment) { }
      virtual ~MemoryPool(void) { }
      virtual ApEvent get_ready_event(void) const = 0;
      virtual size_t query_memory_limit(void) = 0;
      virtual size_t query_available_memory(void) = 0;
      virtual PoolBounds get_bounds(void) const = 0;
      virtual void capture_task_instances(
          const std::map<PhysicalManager*, unsigned>& instances) = 0;
      virtual FutureInstance* allocate_future(
          UniqueID creator_uid, size_t size) = 0;
      virtual PhysicalInstance allocate_instance(
          UniqueID creator_uid, LgEvent unique_event,
          const Realm::InstanceLayoutGeneric& layout, RtEvent& use_event) = 0;
      virtual bool contains_instance(PhysicalInstance instance) const = 0;
      virtual RtEvent escape_task_local_instance(
          PhysicalInstance instance, RtEvent safe_effects, size_t num_results,
          PhysicalInstance* result, LgEvent* unique_events,
          const Realm::InstanceLayoutGeneric** layouts, UniqueID creator) = 0;
      virtual void free_instance(
          PhysicalInstance instance, RtEvent precondition,
          LgEvent unique_event) = 0;
      virtual bool is_released(void) const = 0;
      virtual void release_pool(UniqueID creator) = 0;
      virtual void finalize_pool(RtEvent done) = 0;
    public:
      virtual void serialize(Serializer& rez) = 0;
      static void pack_null_pool(Serializer& rez);
      static MemoryPool* deserialize(Deserializer& derez);
    public:
      static constexpr FieldID FID = 0;
      static Realm::InstanceLayoutGeneric* create_layout(
          size_t size, size_t alignment, size_t offset = 0);
    public:
      const size_t max_alignment;
    };

    /**
     * \class ConcretePool
     * A concrete pool has a specific amount of memory available for
     * dynamic allocations and it will progressively shrink during
     * the execution of the task. This pool is backed by a Realm
     * instance that we will redistrict to split off into new instances
     */
    class ConcretePool : public MemoryPool {
    private:
      struct Range {
        uintptr_t first, last;  // half-open range: [first, last)
        unsigned prev, next;    // double-linked list of all ranges (by index)
        unsigned prev_free,
            next_free;  // double-linked list of just free ranges
        PhysicalInstance instance;
      };
    public:
      ConcretePool(
          PhysicalInstance instance, size_t size, size_t alignment,
          RtEvent use_event, LgEvent unique_event, MemoryManager* manager);
      virtual ~ConcretePool(void) override;
      virtual ApEvent get_ready_event(void) const override;
      virtual size_t query_memory_limit(void) override;
      virtual size_t query_available_memory(void) override;
      virtual PoolBounds get_bounds(void) const override;
      virtual void capture_task_instances(
          const std::map<PhysicalManager*, unsigned>& instances) override;
      virtual FutureInstance* allocate_future(
          UniqueID creator_uid, size_t size) override;
      virtual PhysicalInstance allocate_instance(
          UniqueID creator_uid, LgEvent unique_event,
          const Realm::InstanceLayoutGeneric& layout,
          RtEvent& use_event) override;
      virtual bool contains_instance(PhysicalInstance instance) const override;
      virtual RtEvent escape_task_local_instance(
          PhysicalInstance instance, RtEvent safe_effects, size_t num_results,
          PhysicalInstance* result, LgEvent* unique_events,
          const Realm::InstanceLayoutGeneric** layouts, UniqueID uid) override;
      virtual void free_instance(
          PhysicalInstance instance, RtEvent precondition,
          LgEvent unique_event) override;
      virtual bool is_released(void) const override;
      virtual void release_pool(UniqueID creator) override;
      virtual void finalize_pool(RtEvent done) override;
      virtual void serialize(Serializer& rez) override;
    private:
      unsigned allocate(size_t size, size_t alignment, uintptr_t& start);
      void deallocate(unsigned index);
      unsigned alloc_range(
          uintptr_t first, uintptr_t last, PhysicalInstance backing);
      void free_range(unsigned index);
      void add_to_free_list(unsigned index, Range& r);
      void remove_from_free_list(unsigned index, Range& r);
      void grow_hole(unsigned index, Range& r, uintptr_t bound, bool before);
      RtEvent escape_range(
          unsigned index, size_t num_results, PhysicalInstance* results,
          LgEvent* unique_events, const Realm::InstanceLayoutGeneric** layouts,
          UniqueID creator);
      static unsigned floor_log2(uint64_t size);
    private:
      MemoryManager* const manager;
      const size_t limit;
      size_t remaining_bytes;
    private:
      static constexpr unsigned SENTINEL = std::numeric_limits<unsigned>::max();
      std::vector<Range> ranges;
      // Each external instance has a range that it corresponds to
      std::map<PhysicalInstance, unsigned> allocated;
      // Each backing instance has a start range and use event
      std::map<PhysicalInstance, std::pair<RtEvent, LgEvent> >
          backing_instances;
      // Instances that are freed with event preconditions
      std::map<unsigned, RtEvent> pending_frees;
      // Free lists associated with a specific sizes by powers of 2
      // entry[0] = sizes from [2^0,2^1)
      // entry[1] = sizes from [2^1,2^2)
      // entry[2] = sizes from [2^2,2^3)
      // ...
      std::vector<unsigned> size_based_free_lists;
      // Linked list of ranges not currently be used
      unsigned first_unused_range;
      // Whether the ranges have been initialized
      bool ranges_initialized;
      // Whether this pool has been released
      bool released;
      // Whether this pool has been finalized
      bool finalized;
    };

    /**
     * \class UnboundPool
     * An unbound pool is a place holder for being able to do allocations
     * with an unbounded amount of memory (different from infinite since
     * we can still run out). This will forward all allocation requests
     * through to the actual memory manager, which is only safe because
     * as long as this object is alive it blocks the memory manager
     * from doing any additional allocations.
     */
    class UnboundPool : public MemoryPool {
    public:
      UnboundPool(
          MemoryManager* manager, UnboundPoolScope scope,
          TaskTreeCoordinates& coordinates, size_t max_free_bytes);
      virtual ~UnboundPool(void) override;
      virtual ApEvent get_ready_event(void) const override;
      virtual size_t query_memory_limit(void) override;
      virtual size_t query_available_memory(void) override;
      virtual PoolBounds get_bounds(void) const override;
      virtual void capture_task_instances(
          const std::map<PhysicalManager*, unsigned>& instances) override;
      virtual FutureInstance* allocate_future(
          UniqueID creator_uid, size_t size) override;
      virtual PhysicalInstance allocate_instance(
          UniqueID creator_uid, LgEvent unique_event,
          const Realm::InstanceLayoutGeneric& layout,
          RtEvent& use_event) override;
      virtual bool contains_instance(PhysicalInstance instance) const override;
      virtual RtEvent escape_task_local_instance(
          PhysicalInstance instance, RtEvent safe_effects, size_t num_results,
          PhysicalInstance* result, LgEvent* unique_events,
          const Realm::InstanceLayoutGeneric** layouts, UniqueID uid) override;
      virtual void free_instance(
          PhysicalInstance instance, RtEvent precondition,
          LgEvent unique_event) override;
      virtual bool is_released(void) const override;
      virtual void release_pool(UniqueID creator) override;
      virtual void finalize_pool(RtEvent done) override;
      virtual void serialize(Serializer& rez) override;
      void unpack(Deserializer& derez);
    private:
      PhysicalInstance find_local_freed_hole(
          size_t size, size_t& prev_size, RtEvent& previous_done,
          LgEvent& previous_unique);
    private:
      TaskTreeCoordinates coordinates;
      struct FreedInstance {
        PhysicalInstance instance;
        RtEvent precondition;
        LgEvent unique_event;
      };
      std::map<size_t, std::list<FreedInstance> > freed_instances;
      std::vector<PhysicalManager*> captured_instances;
      MemoryManager* const manager;
      const size_t max_freed_bytes;
      size_t freed_bytes;
      const UnboundPoolScope scope;
      bool released;
      bool finalized;
    };

    /**
     * \class MemoryManager
     * The goal of the memory manager is to keep track of all of
     * the physical instances that the runtime knows about in various
     * memories throughout the system.  This will then allow for
     * feedback when mapping to know when memories are nearing
     * their capacity.
     */
    class MemoryManager {
    public:
      enum RequestKind {
        CREATE_INSTANCE_CONSTRAINTS,
        CREATE_INSTANCE_LAYOUT,
        FIND_OR_CREATE_CONSTRAINTS,
        FIND_OR_CREATE_LAYOUT,
        REDISTRICT_INSTANCE_CONSTRAINTS,
        REDISTRICT_INSTANCE_LAYOUT,
        FIND_ONLY_CONSTRAINTS,
        FIND_ONLY_LAYOUT,
        FIND_MANY_CONSTRAINTS,
        FIND_MANY_LAYOUT,
      };
    public:
      class TaskLocalInstanceAllocator : public ProfilingResponseHandler {
      public:
        TaskLocalInstanceAllocator(void) = delete;
        TaskLocalInstanceAllocator(LgEvent unique_event);
        TaskLocalInstanceAllocator(const TaskLocalInstanceAllocator&) = delete;
        TaskLocalInstanceAllocator(TaskLocalInstanceAllocator&& rhs) noexcept;
        virtual ~TaskLocalInstanceAllocator(void) { ready.wait(); }
      public:
        TaskLocalInstanceAllocator& operator=(
            const TaskLocalInstanceAllocator&) = delete;
        TaskLocalInstanceAllocator& operator=(TaskLocalInstanceAllocator&&) =
            delete;
      public:
        virtual bool handle_profiling_response(
            const Realm::ProfilingResponse& response, const void* orig,
            size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
        inline bool succeeded(void) const
        {
          ready.wait();
          return success;
        }
      private:
        RtUserEvent ready;
        LgEvent unique_event;
        LgEvent caller_fevent;
        bool success;
      };
    public:
      struct MallocInstanceArgs : public LgTaskArgs<MallocInstanceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MALLOC_INSTANCE_TASK_ID;
      public:
        MallocInstanceArgs(
            MemoryManager* m, const Realm::InstanceLayoutGeneric* l,
            const Realm::ProfilingRequestSet* r, PhysicalInstance* i, LgEvent u)
          : LgTaskArgs<MallocInstanceArgs>(false, false), manager(m),
            layout(l->clone()), requests(r), instance(i), unique_event(u)
        { }
        void execute(void) const;
      public:
        MemoryManager* manager;
        Realm::InstanceLayoutGeneric* layout;
        const Realm::ProfilingRequestSet* requests;
        PhysicalInstance* instance;
        LgEvent unique_event;
      };
      struct FreeInstanceArgs : public LgTaskArgs<FreeInstanceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_FREE_INSTANCE_TASK_ID;
      public:
        FreeInstanceArgs(MemoryManager* m, PhysicalInstance i)
          : LgTaskArgs<FreeInstanceArgs>(true, true), manager(m), instance(i)
        { }
        void execute(void) const;
      public:
        MemoryManager* manager;
        PhysicalInstance instance;
      };
    public:
      MemoryManager(Memory mem);
      MemoryManager(const MemoryManager& rhs) = delete;
      ~MemoryManager(void);
    public:
      MemoryManager& operator=(const MemoryManager& rhs) = delete;
    public:
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
      inline Processor get_local_gpu(void) const { return local_gpu; }
#endif
      static inline bool is_owner_memory(Memory m, AddressSpace space)
      {
        if (m.address_space() == space)
          return true;
        const Memory::Kind kind = m.kind();
        // File system memories are "local" everywhere
        return ((kind == Memory::HDF_MEM) || (kind == Memory::FILE_MEM));
      }
      inline const char* get_name(void) const
      {
        const char* mem_names[] = {
#define MEM_NAMES(name, desc) #name,
            REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
        };
        return mem_names[memory.kind()];
      }
      inline void update_remaining_capacity(size_t size)
      {
        legion_assert(is_owner);
        remaining_capacity.fetch_add(size);
      }
    public:
      void find_shutdown_preconditions(std::set<ApEvent>& preconditions);
      void prepare_for_shutdown(void);
      void finalize(void);
    public:
      void register_remote_instance(PhysicalManager* manager);
      void unregister_remote_instance(PhysicalManager* manager);
      void unregister_deleted_instance(PhysicalManager* manager);
    public:
      bool create_physical_instance(
          const LayoutConstraintSet& contraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          Processor processor, bool acquire, GCPriority priority,
          bool tight_bounds, LayoutConstraintKind* unsat_kind,
          unsigned* unsat_index, size_t* footprint,
          RtEvent* safe_for_unbounded_pools, UniqueID creator_id,
          bool remote = false);
      bool create_physical_instance(
          LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          Processor processor, bool acquire, GCPriority priority,
          bool tight_bounds, LayoutConstraintKind* unsat_kind,
          unsigned* unsat_index, size_t* footprint,
          RtEvent* safe_for_unbounded_pools, UniqueID creator_id,
          bool remote = false);
      bool find_or_create_physical_instance(
          const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          bool& created, Processor processor, bool acquire, GCPriority priority,
          bool tight_region_bounds, LayoutConstraintKind* unsat_kind,
          unsigned* unsat_index, size_t* footprint,
          RtEvent* safe_for_unbounded_pools, UniqueID creator_id,
          bool remote = false);
      bool find_or_create_physical_instance(
          LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions,
          const TaskTreeCoordinates& coordinates, MappingInstance& result,
          bool& created, Processor processor, bool acquire, GCPriority priority,
          bool tight_region_bounds, LayoutConstraintKind* unsat_kind,
          unsigned* unsat_index, size_t* footprint,
          RtEvent* safe_for_unbounded_pools, UniqueID creator_id,
          bool remote = false);
      bool redistrict_physical_instance(
          MappingInstance& instance, const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions, Processor processor,
          bool acquire, GCPriority priority, bool tight_bounds,
          UniqueID creator_id);
      bool redistrict_physical_instance(
          MappingInstance& instance, LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions, Processor processor,
          bool acquire, GCPriority priority, bool tight_bounds,
          UniqueID creator_id);
      bool find_physical_instance(
          const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions, MappingInstance& result,
          bool acquire, bool tight_bounds, bool remote = false);
      bool find_physical_instance(
          LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions, MappingInstance& result,
          bool acquire, bool tight_bounds, bool remote = false);
      void find_physical_instances(
          const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions,
          std::vector<MappingInstance>& results, bool acquire,
          bool tight_bounds, bool remote = false);
      void find_physical_instances(
          LayoutConstraints* constraints,
          const std::vector<LogicalRegion>& regions,
          std::vector<MappingInstance>& results, bool acquire,
          bool tight_bounds, bool remote = false);
      void release_tree_instances(RegionTreeID tid);
      void set_garbage_collection_priority(
          PhysicalManager* manager, GCPriority priority);
      void record_created_instance(
          PhysicalManager* manager, bool acquire, GCPriority priority);
      void notify_collected_instances(
          const std::vector<PhysicalManager*>& instances);
    public:
      size_t compute_future_alignment(size_t size) const;
      FutureInstance* create_future_instance(
          UniqueID creator_id, const TaskTreeCoordinates& coordinates,
          size_t size, RtEvent* safe_for_unbounded_pools);
      void free_future_instance(
          PhysicalInstance inst, size_t size, RtEvent free_event);
      PhysicalInstance create_task_local_instance(
          UniqueID creator_uid, const TaskTreeCoordinates& coordinates,
          LgEvent unique_event, const Realm::InstanceLayoutGeneric& layout,
          RtEvent& use_event, RtEvent* safe_for_unbounded_pools);
      void free_task_local_instance(
          PhysicalInstance instance,
          RtEvent precondition = RtEvent::NO_RT_EVENT);
      size_t query_available_memory(void);
      MemoryPool* create_memory_pool(
          UniqueID creator_uid, TaskTreeCoordinates& coordinates,
          const PoolBounds& bounds, RtEvent* safe_for_unbounded_pools);
      void release_concrete_pool(
          RtEvent done,
          const std::map<PhysicalInstance, std::pair<RtEvent, LgEvent> >&
              backing_instances);
      void release_unbound_pool(void);
      uint64_t order_collective_unbounded_pools(SingleTask* task);
      RtEvent finalize_collective_unbounded_pools_order(
          SingleTask* task, uint64_t max_lamport_clock);
      void end_collective_unbounded_pools_task(void);
    protected:
      void start_next_collective_unbounded_pools_task(void);
    public:
      void process_instance_request(Deserializer& derez, AddressSpaceID source);
      void process_instance_response(
          Deserializer& derez, AddressSpaceID source);
    protected:
      bool find_satisfying_instance(
          const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions, MappingInstance& result,
          bool acquire, bool tight_region_bounds, bool remote);
      void find_satisfying_instances(
          const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions,
          std::vector<MappingInstance>& results, bool acquire,
          bool tight_region_bounds, bool remote);
      bool find_valid_instance(
          const LayoutConstraintSet& constraints,
          const std::vector<LogicalRegion>& regions, MappingInstance& result,
          bool acquire, bool tight_region_bounds, bool remote);
      void release_candidate_references(
          const std::deque<PhysicalManager*>& candidates) const;
    public:
      PhysicalManager* create_unbound_instance(
          LogicalRegion region, const LayoutConstraintSet& constraints,
          ApEvent ready_event, GCPriority priority);
      void check_instance_deletions(const std::vector<PhysicalManager*>& del);
    protected:
      // We serialize all allocation attempts in a memory in order to
      // ensure find_and_create calls will remain atomic and are also
      // ordered with respect to any unbound pool allocations
      RtEvent acquire_allocation_privilege(
          const TaskTreeCoordinates& coords, RtEvent* safe_for_unbounded_pools);
      void release_allocation_privilege(void);
      PhysicalManager* allocate_physical_instance(
          InstanceBuilder& builder, size_t* footprint,
          LayoutConstraintKind* unsat_kind, unsigned* unsat_index);
    public:
      void remove_collectable(GCPriority priority, PhysicalManager* manager);
    public:
      RtEvent attach_external_instance(PhysicalManager* manager);
      void detach_external_instance(PhysicalManager* manager);
    public:
      bool is_visible_memory(Memory other);
    public:
      void free_external_allocation(uintptr_t ptr, size_t size);
#ifdef LEGION_MALLOC_INSTANCES
    public:
      RtEvent allocate_legion_instance(
          const Realm::InstanceLayoutGeneric& layout,
          const Realm::ProfilingRequestSet& requests, PhysicalInstance& inst,
          LgEvent unique_event, bool needs_defer = true);
      void record_legion_instance(
          InstanceManager* manager, PhysicalInstance instance);
      void free_legion_instance(InstanceManager* manager, RtEvent deferred);
      void free_legion_instance(
          RtEvent deferred, PhysicalInstance inst, bool needs_defer = true);
#endif
    public:
      // The memory that we are managing
      const Memory memory;
      // The owner address space
      const AddressSpaceID owner_space;
      // Is this the owner memory or not
      const bool is_owner;
      // The capacity in bytes of this memory
      const size_t capacity;
      // The remaining capacity in this memory
      std::atomic<size_t> remaining_capacity;
    protected:
      // Lock for controlling access to the data
      // structures in this memory manager
      mutable LocalLock manager_lock;
      // Lock for ordering garbage collection
      // This lock should always be taken before the manager lock
      mutable LocalLock collection_lock;
      // We maintain several sets of instances here
      // This is a generic list that tracks all the allocated instances
      // For collectable instances they have non-nullptr GCHole that
      // represents a range of memory that can be collected
      // This data structure is protected by the manager_lock
      typedef lng::map<PhysicalManager*, GCPriority> TreeInstances;
      // We use a two-part key here that is sorted three different ways
      // First we split on region tree ID since we only need to do finds
      // for matching instances in particular region tree and then we
      // sort by the field mask hash as a summary of which fields a
      // particular instance has. This field mask has is sorted by
      // popcount first since we want to easily discount instances
      // that have fewer fields than what we're searching for.
      struct TreeFieldKey : public std::pair<RegionTreeID, uint64_t> {
      public:
        TreeFieldKey(void) = default;
        TreeFieldKey(RegionTreeID tid, uint64_t hash)
          : std::pair<RegionTreeID, uint64_t>(tid, hash)
        { }
        TreeFieldKey(PhysicalManager* manager);
      };
      struct TreeFieldComparator {
      public:
        inline int popcount(uint64_t value) const
        {
#if __cplusplus >= 202002L
          return std::popcount(value);
#elif defined(__clang__) || defined(__GCC__)
          return __builtin_popcountll(value);
#else
          // Adapted from Hacker's Delight
          // put count of each 2 bits into those 2 bits
          value -= (value >> 1) & 0x5555555555555555ULL;
          // put count of each 4 bits into those 4 bits
          value = (value & 0x3333333333333333ULL) +
                  ((value >> 2) & 0x3333333333333333ULL);
          // put count of each 8 bits into those 8 bits
          value = (value + (value >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
          // sum all bytes into top byte
          return int((value * 0x0101010101010101ULL) >> 56);
#endif
        }
        inline bool operator()(
            const TreeFieldKey& lhs, const TreeFieldKey& rhs) const
        {
          if (lhs.first < rhs.first)
            return true;
          if (lhs.first > rhs.first)
            return false;
          const int lhs_popcount = popcount(lhs.second);
          const int rhs_popcount = popcount(rhs.second);
          if (lhs_popcount < rhs_popcount)
            return true;
          if (lhs_popcount > rhs_popcount)
            return false;
          return lhs.second < rhs.second;
        }
      };
      typedef lng::map<TreeFieldKey, TreeInstances, TreeFieldComparator>
          TreeFieldInstances;
      TreeFieldInstances current_instances;
      // Keep track of all groupings of instances based on their
      // garbage collection priorities and placement in memory
      std::map<
          GCPriority, std::set<PhysicalManager*>, std::greater<GCPriority> >
          collectable_instances;
      // Keep track of outstanding requests for allocations which
      // will be tried in the order that they arrive
      std::deque<std::pair<RtUserEvent, const TaskTreeCoordinates*> >
          pending_allocation_attempts;
      // Track how many outstanding local instance allocations there are
      unsigned outstanding_task_local_allocations;
      // Track how many outstanding unbounded allocators there are
      unsigned outstanding_unbounded_allocations;
      // Keep track of the current unbounded pool scope
      UnboundPoolScope unbounded_pool_scope;
      // If we have an unbounded pool then track the task tree coordinates
      TaskTreeCoordinates unbounded_coordinates;
      // Allocation transition event for switching between bounded
      // and unbounded modes
      RtUserEvent unbounded_transition_event;
      // Data structures for helping to order collectively mapped tasks
      // with unbounded memory pools
      struct CollectiveState {
      public:
        CollectiveState(uint64_t clock) : lamport_clock(clock), max(false) { }
      public:
        uint64_t lamport_clock;
        RtUserEvent ready_event;
        bool max;  // whether the lamport clock is the max all-reduce or not
      };
      std::map<SingleTask*, CollectiveState> collective_tasks;
      uint64_t collective_lamport_clock;
      uint32_t ready_collective_tasks;
      uint32_t outstanding_collective_tasks;
    protected:
      std::set<Memory> visible_memories;
    protected:
#ifdef LEGION_MALLOC_INSTANCES
      std::map<InstanceManager*, PhysicalInstance> legion_instances;
      std::map<PhysicalInstance, size_t> allocations;
      std::map<RtEvent, PhysicalInstance> pending_collectables;
#endif
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
      Processor local_gpu;
#endif
    protected:
      class GarbageCollector {
      public:
        GarbageCollector(
            LocalLock& collection_lock, LocalLock& manager_lock,
            AddressSpaceID local, Memory memory, size_t needed, size_t capacity,
            std::atomic<size_t>& remaining,
            std::map<
                GCPriority, std::set<PhysicalManager*>,
                std::greater<GCPriority> >& collectables);
        GarbageCollector(const GarbageCollector& rhs) = delete;
        ~GarbageCollector(void);
      public:
        GarbageCollector& operator=(const GarbageCollector& rhs) = delete;
      public:
        RtEvent perform_collection(
            PhysicalInstance& hole_instance, LgEvent& hole_unique_event);
        inline bool collection_complete(void) const
        {
          return (current_priority == LEGION_GC_NEVER_PRIORITY);
        }
      protected:
        void sort_next_priority_holes(bool advance = true);
        void update_capacity(size_t size);
      protected:
        struct Range {
        public:
          Range(void) : size(0) { }
          Range(PhysicalManager* m);
          std::vector<PhysicalManager*> managers;
          size_t size;
        };
      protected:
        // Note this makes sure there is only one collection at a time
        AutoLock collection_lock;
        LocalLock& manager_lock;
        std::map<
            GCPriority, std::set<PhysicalManager*>, std::greater<GCPriority> >&
            collectable_instances;
        const Memory memory;
        const AddressSpaceID local_space;
        const size_t needed_size;
        const size_t capacity;
        std::atomic<size_t>& remaining_capacity;
      protected:
        std::vector<PhysicalManager*> small_holes, perfect_holes;
        std::map<size_t, std::vector<PhysicalManager*> > large_holes;
        std::map<uintptr_t, Range> ranges;
        GCPriority current_priority;
      };
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_MEMORY_MANAGER_H__
