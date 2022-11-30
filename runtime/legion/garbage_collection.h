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


#ifndef __LEGION_GARBAGE_COLLECTION__
#define __LEGION_GARBAGE_COLLECTION__

#include "legion/legion_types.h"
#include "legion/bitmask.h"

namespace Legion {
  namespace Internal {

    enum DistCollectableType {
      PHYSICAL_MANAGER_DC = 0x1,
      INDEX_SPACE_NODE_DC = 0x2,
      INDEX_PART_NODE_DC = 0x3,
      MATERIALIZED_VIEW_DC = 0x4,
      REDUCTION_VIEW_DC = 0x5,
      FILL_VIEW_DC = 0x6,
      PHI_VIEW_DC = 0x7,
      SHARDED_VIEW_DC = 0x8,
      FUTURE_DC = 0x9,
      FUTURE_MAP_DC = 0xA,
      INDEX_EXPR_NODE_DC = 0xB,
      FIELD_SPACE_DC = 0xC,
      REGION_TREE_NODE_DC = 0xD,
      EQUIVALENCE_SET_DC = 0xE,
      // be careful making this last one bigger than 0x10! see instance encoding
      DIST_TYPE_LAST_DC = 0xF,  // must be last
    };

    enum ReferenceSource {
      INTERNAL_VALID_REF = 0,
      DEFERRED_TASK_REF = 1,
      VERSION_MANAGER_REF = 2,
      PHYSICAL_ANALYIS_REF = 3,
      PENDING_UNBOUND_REF = 4,
      PHYSICAL_REGION_REF = 5,
      //PENDING_GC_REF = 6,
      REMOTE_DID_REF = 7,
      PENDING_COLLECTIVE_REF = 8,
      MEMORY_MANAGER_REF = 9,
      PENDING_REFINEMENT_REF = 10,
      FIELD_ALLOCATOR_REF = 11,
      REMOTE_CREATE_REF = 12,
      INSTANCE_MAPPER_REF = 13,
      APPLICATION_REF = 14,
      MAPPING_ACQUIRE_REF = 15,
      NEVER_GC_REF = 16,
      CONTEXT_REF = 17,
      RESTRICTED_REF = 18,
      META_TASK_REF = 19,
      PHYSICAL_USER_REF = 20,
      INSTANCE_BUILDER_REF = 21,
      REGION_TREE_REF = 22,
      LAYOUT_DESC_REF = 23,
      RUNTIME_REF = 24,
      LIVE_EXPR_REF = 25,
      TRACE_REF = 26,
      AGGREGATOR_REF = 27,
      FIELD_STATE_REF = 28,
      COPY_ACROSS_REF = 29,
      CANONICAL_REF = 30,
      DISJOINT_COMPLETE_REF = 31,
      REPLICATION_REF = 32,
      PHYSICAL_ANALYSIS_REF = 33,
      LAST_SOURCE_REF = 34,
    };

    enum ReferenceKind {
      GC_REF_KIND = 0,
      VALID_REF_KIND = 1,
      RESOURCE_REF_KIND = 2,
    };

#define REFERENCE_NAMES_ARRAY(names)                \
    const char *const names[LAST_SOURCE_REF] = {    \
      "Internal Nested Valid Reference",            \
      "Deferred Task Reference",                    \
      "Version Manager Reference",                  \
      "Physical Analysis Reference",                \
      "Pending Unbound Reference",                  \
      "Physical Region Reference",                  \
      "Pending GC Reference",                       \
      "Remote Distributed ID Reference",            \
      "Pending Collective Reference",               \
      "Memory Manager Reference",                   \
      "Pending Refinement Reference",               \
      "Field Allocator Reference",                  \
      "Remote Creation Reference",                  \
      "Instance Mapper Reference",                  \
      "Application Reference",                      \
      "Mapping Acquire Reference",                  \
      "Never GC Reference",                         \
      "Context Reference",                          \
      "Restricted Reference",                       \
      "Meta-Task Reference",                        \
      "Physical User Reference",                    \
      "Instance Builder Reference",                 \
      "Region Tree Reference",                      \
      "Layout Description Reference",               \
      "Runtime Reference",                          \
      "Live Index Space Expression Reference",      \
      "Physical Trace Reference",                   \
      "Aggregator Reference",                       \
      "Field State Reference",                      \
      "Copy Across Executor Reference",             \
      "Canonical Index Space Expression Reference", \
      "Disjoint Complete Reference",                \
      "Replication Reference",                      \
      "Physical Analysis Reference",                \
    }

    extern Realm::Logger log_garbage;

    /**
     * \class Collectable
     * We'll use this class for reference
     * counting objects to know when they 
     * can be garbage collected.  We rely
     * on GCC built-in atomics to do this
     * efficiently without needing locks.
     * If remove reference returns true
     * then that was the last reference and
     * the object should be garbage
     * collected.  This kind of collectable
     * is only valid on a single node.  Below
     * we define other kinds of collectables
     * that work in a distributed environment.
     */
    class Collectable {
    public:
      Collectable(unsigned init = 0)
        : references(init) { }
    public:
      inline void add_reference(unsigned cnt = 1);
      inline bool remove_reference(unsigned cnt = 1);
    protected:
      std::atomic<unsigned int> references;
    };

    /**
     * \interface Notifiable
     * This is for registering waiters on predicate
     * objects.  Notifiable objects will be notified
     * when the predicate values have been set.  Note
     * that a notifiable must be a collectable so
     * we can add and remove references to them
     * before and after doing notify operations.
     */
    class Notifiable : public Collectable {
    public:
      virtual ~Notifiable(void) = 0;
    public:
      virtual void notify(bool result, int key) = 0;
      virtual void reset(int key) = 0;
    };

    /**
     * \class ImplicitReferenceTracker
     * This class tracks implicit references that are held either by
     * an application runtime API call or a meta-task. At the end of the
     * runtime API call or meta-task the references are updated.
     */
    class ImplicitReferenceTracker {
    public:
      ImplicitReferenceTracker(void) { }
      ImplicitReferenceTracker(const ImplicitReferenceTracker&) = delete;
      ~ImplicitReferenceTracker(void);
    public:
      ImplicitReferenceTracker& operator=(
                               const ImplicitReferenceTracker&) = delete;
    public:
      static inline void record_live_expression(IndexSpaceExpression *expr)
      {
        if (implicit_reference_tracker == NULL)
          implicit_reference_tracker = new ImplicitReferenceTracker;
        implicit_reference_tracker->live_expressions.push_back(expr);
      }
    private:
      std::vector<IndexSpaceExpression*> live_expressions;
    };

    /**
     * \class Distributed Collectable
     * This is the base class for handling all the reference
     * counting logic for any objects in the runtime which 
     * have distributed copies of themselves across multiple
     * nodes and therefore have to have a distributed 
     * reference counting scheme to determine when they 
     * can be collected. Distributed collectable objects
     * are only collected once it is safe to collect all of
     * them. There are three "levels" they can be at. The
     * objects will descend down through the levels collectively
     * across all their copies and monotonically. The last level
     * only guarantees that the object is kept alive locally.
     */
    class DistributedCollectable {
    public:
      enum State {
        DELETED_REF_STATE,
        LOCAL_REF_STATE,
        GLOBAL_REF_STATE,
        VALID_REF_STATE, // a second global ref state
      };
    public:
      DistributedCollectable(Runtime *rt, DistributedID did,
                             bool register_with_runtime = true,
                             CollectiveMapping *mapping = NULL,
                             State initial_state = GLOBAL_REF_STATE);
      DistributedCollectable(const DistributedCollectable &rhs);
      virtual ~DistributedCollectable(void);
    public:
      inline void add_base_gc_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_gc_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_gc_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_gc_ref(DistributedID source, int cnt = 1); 
    public:
      inline void add_base_resource_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_resource_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_resource_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_resource_ref(DistributedID source, int cnt = 1);
    public:
      void pack_global_ref(unsigned cnt = 1);
      void unpack_global_ref(unsigned cnt = 1); 
    public:
      template<bool NEED_LOCK=true>
      bool is_global(void) const;
      // Atomic check and increment operations 
      inline bool check_global_and_increment(ReferenceSource src, int cnt = 1);
      inline bool check_global_and_increment(DistributedID source, int cnt = 1);
#ifndef DEBUG_LEGION_GC
    private:
      void add_gc_reference(int cnt);
      bool remove_gc_reference(int cnt);
      bool acquire_global(int cnt);
      bool acquire_global_remote(AddressSpaceID &forward);
    private:
      void add_resource_reference(int cnt);
      bool remove_resource_reference(int cnt);
#else
    private:
      void add_gc_reference(int cnt);
      void add_base_gc_ref_internal(ReferenceSource source, int cnt);
      void add_nested_gc_ref_internal(DistributedID source, int cnt);
      bool remove_base_gc_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_gc_ref_internal(DistributedID source, int cnt);
      template<typename T>
      bool acquire_global(int cnt, T source, std::map<T,int> &gc_references);
      bool acquire_global_remote(AddressSpaceID &forward);
    public:
      void add_base_resource_ref_internal(ReferenceSource source, int cnt); 
      void add_nested_resource_ref_internal(DistributedID source, int cnt); 
      bool remove_base_resource_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_resource_ref_internal(DistributedID source, int cnt); 
#endif
    public:
      // Notify that this is no long globally available
      virtual void notify_local(void) = 0;
    public:
      inline bool is_owner(void) const { return (owner_space == local_space); }
      inline bool is_registered(void) const { return registered_with_runtime; }
      bool has_remote_instance(AddressSpaceID remote_space) const;
      void update_remote_instances(AddressSpaceID remote_space); 
      void filter_remote_instances(AddressSpaceID remote_space);
    public:
      inline bool has_remote_instances(void) const;
      inline size_t count_remote_instances(void) const;
      template<typename FUNCTOR>
      inline void map_over_remote_instances(FUNCTOR &functor);
    public:
      void register_with_runtime(void);
    public:
      // This for remote nodes only
      void unregister_collectable(std::set<RtEvent> &done_events);
      static void handle_unregister_collectable(Runtime *runtime,
                                                Deserializer &derez);
    public:
      RtEvent send_remote_registration(void);
      static void handle_did_remote_registration(Runtime *runtime,
                                                 Deserializer &derez,
                                                 AddressSpaceID source);
    protected:
      bool can_delete(AutoLock &gc);
      virtual bool can_downgrade(void) const;
      virtual bool perform_downgrade(AutoLock &gc);
      virtual void process_downgrade_update(void);
      virtual void initialize_downgrade_state(AddressSpaceID owner);
      virtual void update_instances_internal(AddressSpaceID remote_inst);
      void check_for_downgrade(AddressSpaceID downgrade_owner, 
                               bool need_lock = true);
      bool process_downgrade_response(AddressSpaceID notready,
                                              uint64_t total_sent,
                                              uint64_t total_received);
      void send_downgrade_notifications(void);
      bool process_downgrade_success(void);
      AddressSpaceID get_downgrade_target(AddressSpaceID owner) const;
    public:
      static void handle_downgrade_request(Runtime *runtime,
                         Deserializer &derez, AddressSpaceID source);
      static void handle_downgrade_response(Runtime *runtime,
                                            Deserializer &derez);
      static void handle_downgrade_success(Runtime *runtime,
                                           Deserializer &derez);
      static void handle_downgrade_update(Runtime *runtime,
                                          Deserializer &derez);
      static void handle_global_acquire_request(Runtime *runtime,
                                                Deserializer &derez);
      static void handle_global_acquire_response(Deserializer &derez);
    public:
      Runtime *const runtime;
      const DistributedID did;
      const AddressSpaceID owner_space;
      const AddressSpaceID local_space;
      CollectiveMapping *const collective_mapping;
    protected:
      mutable LocalLock gc_lock;
    protected:
      State current_state;
    protected:
#ifdef DEBUG_LEGION_GC
      int gc_references;
      int resource_references;
#else
      std::atomic<int> gc_references;
      std::atomic<int> resource_references;
#endif
#ifdef DEBUG_LEGION_GC
    protected:
      std::map<ReferenceSource,int> detailed_base_gc_references;
      std::map<DistributedID,int> detailed_nested_gc_references;
      std::map<ReferenceSource,int> detailed_base_resource_references;
      std::map<DistributedID,int> detailed_nested_resource_references;
#endif
    protected:
      // Track all the remote instances (relative to ourselves) we know about
      NodeSet                  remote_instances;
    protected:
      AddressSpaceID           downgrade_owner, notready_owner;
      uint64_t  sent_global_references, received_global_references;
      uint64_t  total_sent_references, total_received_references;
      unsigned  remaining_responses;
    protected:
      mutable bool registered_with_runtime;
    };

    /**
     * \class ValidDistributedCollectable
     * The valid distributed collectable class is a distributed collectable
     * that also happens to be have an additional state that it goes through
     * before it can be deleted
     */
    class ValidDistributedCollectable : public DistributedCollectable {
    public:
      ValidDistributedCollectable(Runtime *rt, DistributedID did,
                                  bool register_with_runtime = true,
                                  CollectiveMapping *mapping = NULL);
      ValidDistributedCollectable(const ValidDistributedCollectable &rhs);
      virtual ~ValidDistributedCollectable(void);     
    public:
      inline void add_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, int cnt = 1);
    public:
      template<bool NEED_LOCK=true>
      bool is_valid(void) const;
      bool check_valid_and_increment(ReferenceSource source,int cnt = 1);
      bool check_valid_and_increment(DistributedID source, int cnt = 1);
#ifndef DEBUG_LEGION_GC
    private:
      void add_valid_reference(int cnt);
      bool remove_valid_reference( int cnt);
      bool acquire_valid(int cnt);
      bool acquire_valid_remote(AddressSpaceID &forward);
#else
    public:
      void add_valid_reference(int cnt);
      void add_base_valid_ref_internal(ReferenceSource source, int cnt);
      void add_nested_valid_ref_internal(DistributedID source, int cnt);
      bool remove_base_valid_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_valid_ref_internal(DistributedID source, int cnt);
      template<typename T>
      bool acquire_valid(int cnt, T source, std::map<T,int> &valid_references);
      bool acquire_valid_remote(AddressSpaceID &forward);
#endif
    public:
      void pack_valid_ref(unsigned cnt = 1);
      void unpack_valid_ref(unsigned cnt = 1);
    protected:
      virtual bool can_downgrade(void) const;
      virtual bool perform_downgrade(AutoLock &gc);
      virtual void process_downgrade_update(void);
      virtual void initialize_downgrade_state(AddressSpaceID owner);
      virtual void update_instances_internal(AddressSpaceID remote_inst);
    public:
      // Notify that this is no longer globally valid
      virtual void notify_invalid(void) = 0;
    public:
      static void handle_valid_acquire_request(Runtime *runtime,
                                               Deserializer &derez);
      static void handle_valid_acquire_response(Deserializer &derez);
    protected:
#ifdef DEBUG_LEGION_GC
      int valid_references;
#else
      std::atomic<int> valid_references;
#endif
#ifdef DEBUG_LEGION_GC
    protected:
      std::map<ReferenceSource,int> detailed_base_valid_references;
      std::map<DistributedID,int> detailed_nested_valid_references;
#endif
    protected:
      uint64_t  sent_valid_references, received_valid_references;
    };

    //--------------------------------------------------------------------------
    // Give some implementations here so things get inlined
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    template<bool ADD>
    static inline void log_base_ref(ReferenceKind kind, DistributedID did,
                  AddressSpaceID local_space, ReferenceSource src, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      did = LEGION_DISTRIBUTED_ID_FILTER(did);
      if (ADD)
        log_garbage.info("GC Add Base Ref %d %lld %d %d %d",
                          kind, did, local_space, src, cnt);
      else
        log_garbage.info("GC Remove Base Ref %d %lld %d %d %d",
                          kind, did, local_space, src, cnt);
    }

    //--------------------------------------------------------------------------
    template<bool ADD>
    static inline void log_nested_ref(ReferenceKind kind, DistributedID did, 
                    AddressSpaceID local_space, DistributedID src, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      did = LEGION_DISTRIBUTED_ID_FILTER(did);
      src = LEGION_DISTRIBUTED_ID_FILTER(src);
      if (ADD)
        log_garbage.info("GC Add Nested Ref %d %lld %d %lld %d",
                          kind, did, local_space, src, cnt);
      else
        log_garbage.info("GC Remove Nested Ref %d %lld %d %lld %d",
                          kind, did, local_space, src, cnt);
    }

    //--------------------------------------------------------------------------
    inline void Collectable::add_reference(unsigned cnt /*= 1*/)
    //--------------------------------------------------------------------------
    {
      references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool Collectable::remove_reference(unsigned cnt /*= 1*/)
    //--------------------------------------------------------------------------
    {
      unsigned prev = references.fetch_sub(cnt);
#ifdef DEBUG_LEGION
      assert(prev >= cnt); // check for underflow
#endif
      // If previous is equal to count, the value is now
      // zero so it is safe to reclaim this object
      return (prev == cnt);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::has_remote_instances(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      return !remote_instances.empty();
    }

    //--------------------------------------------------------------------------
    inline size_t DistributedCollectable::count_remote_instances(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      return remote_instances.pop_count();
    }

    //--------------------------------------------------------------------------
    template<typename FUNCTOR>
    void DistributedCollectable::map_over_remote_instances(FUNCTOR &functor)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock,1,false/*exclusive*/); 
      remote_instances.map(functor); 
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_gc_ref(ReferenceSource source,
                                                        int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_base_gc_ref_internal(source, cnt); 
#else
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return;
      }
      add_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_gc_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_nested_gc_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return;
      }
      add_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_gc_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<false>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_base_gc_ref_internal(source, cnt);
#else
      int current = gc_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_gc_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<false>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_nested_gc_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = gc_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_resource_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<true>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_base_resource_ref_internal(source, cnt);
#else
      int current = resource_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return;
      }
      add_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_resource_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<true>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_nested_resource_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = resource_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return;
      }
      add_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_resource_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<false>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_base_resource_ref_internal(source, cnt);
#else
      int current = resource_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_resource_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<false>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_nested_resource_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = resource_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::check_global_and_increment(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt > 0);
#endif
#ifndef DEBUG_LEGION_GC
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
      bool result = acquire_global(cnt);
#else
      bool result = acquire_global(cnt, source, detailed_base_gc_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::check_global_and_increment(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt > 0);
#endif
#ifndef DEBUG_LEGION_GC
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
      bool result = acquire_global(cnt);
#else
      bool result = acquire_global(cnt, source, detailed_nested_gc_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline void ValidDistributedCollectable::add_base_valid_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void ValidDistributedCollectable::add_nested_valid_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      add_nested_valid_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), 
                                    cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::remove_base_valid_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::remove_nested_valid_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      return remove_nested_valid_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = valid_references.load();
#ifdef DEBUG_LEGION
      assert(current >= cnt);
#endif
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::check_valid_and_increment(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt > 0);
#endif
#ifndef DEBUG_LEGION_GC
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
      bool result = acquire_valid(cnt);
#else
      bool result = acquire_valid(cnt, source, detailed_base_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::check_valid_and_increment(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt > 0);
#endif
#ifndef DEBUG_LEGION_GC
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
      bool result = acquire_valid(cnt);
#else
      bool result = acquire_valid(cnt, source,detailed_nested_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_GARBAGE_COLLECTION__

// EOF

