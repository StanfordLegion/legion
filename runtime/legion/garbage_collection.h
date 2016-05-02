/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "legion_types.h"

// This is a macro for enabling the use of remote references
// on distributed collectable objects. Remote references 
// account for the transitory nature of references that are
// in flight but not necessarily known to the owner node.
// For example, a copy of a distributed collectable on node A
// sends a messages to node B to create a copy of itself and
// add a reference. Remote references are a notfication to 
// the owner node that this creation and reference are in flight
// in case the copy on node A has all its references removed
// before the instance on node B can be created and send its
// own references to the owner node.
// #define USE_REMOTE_REFERENCES

namespace Legion {
  namespace Internal {

    enum ReferenceSource {
      FUTURE_HANDLE_REF = 0,
      DEFERRED_TASK_REF = 1,
      CURRENT_STATE_REF = 2,
      VERSION_INFO_REF = 3,
      PHYSICAL_STATE_REF = 4,
      PHYSICAL_REGION_REF = 5,
      VERSION_MANAGER_REF = 6,
      PENDING_GC_REF = 7,
      REMOTE_DID_REF = 8,
      PENDING_COLLECTIVE_REF = 9,
      MEMORY_MANAGER_REF = 10,
      COMPOSITE_NODE_REF = 11,
      COMPOSITE_HANDLE_REF = 12,
      PERSISTENCE_REF = 13,
      REMOTE_CREATE_REF = 14,
      INSTANCE_MAPPER_REF = 15,
      APPLICATION_REF = 16,
      MAPPING_ACQUIRE_REF = 17,
      NEVER_GC_REF = 18,
      CONTEXT_REF = 19,
      LAST_SOURCE_REF = 20,
    };

    enum ReferenceKind {
      GC_REF_KIND = 0,
      VALID_REF_KIND = 1,
      RESOURCE_REF_KIND = 2,
    };

#define REFERENCE_NAMES_ARRAY(names)                \
    const char *const names[LAST_SOURCE_REF] = {    \
      "Future Handle Reference",                    \
      "Deferred Task Reference",                    \
      "Current State Reference",                    \
      "Version Info Reference",                     \
      "Physical State Reference",                   \
      "Physical Region Reference",                  \
      "Version Manager Reference",                  \
      "Pending GC Reference",                       \
      "Remote Distributed ID Reference",            \
      "Pending Collective Reference",               \
      "Memory Manager Reference",                   \
      "Composite Node Reference",                   \
      "Composite Handle Reference",                 \
      "Persistent Reference",                       \
      "Remote Creation Reference",                  \
      "Instance Mapper Reference",                  \
      "Application Reference",                      \
      "Mapping Acquire Reference",                  \
      "Never GC Reference",                         \
      "Context Reference",                          \
    }

    extern LegionRuntime::Logger::Category log_garbage;

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
      unsigned int references;
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

    class DistributedCollectable {
    public:
      enum State {
        INACTIVE_STATE,
        ACTIVE_INVALID_STATE,
        ACTIVE_DELETED_STATE,
        VALID_STATE,
        DELETED_STATE,
        PENDING_ACTIVE_STATE,
        PENDING_INACTIVE_STATE,
        PENDING_VALID_STATE,
        PENDING_INVALID_STATE,
        PENDING_ACTIVE_VALID_STATE,
        PENDING_ACTIVE_DELETED_STATE,
        PENDING_INVALID_DELETED_STATE,
        PENDING_INACTIVE_DELETED_STATE,
      };
    public:
      template<ReferenceKind REF_KIND, bool ADD>
      class UpdateReferenceFunctor {
      public:
        UpdateReferenceFunctor(DistributedCollectable *dc, unsigned cnt = 1)
          : source(dc), count(cnt) { }
      public:
        inline void apply(AddressSpaceID target);
      protected:
        DistributedCollectable *source;
        unsigned count;
      };
    public:
      DistributedCollectable(Runtime *rt, DistributedID did,
                             AddressSpaceID owner_space,
                             AddressSpaceID local_space,
                             bool register_with_runtime = true);
      virtual ~DistributedCollectable(void);
    public:
      inline void add_base_gc_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_gc_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_gc_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_gc_ref(DistributedID source, int cnt = 1);
    public:
      inline void add_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_valid_ref(ReferenceSource source, 
                                        int cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, 
                                          int cnt = 1);
    public:
      inline void add_base_resource_ref(ReferenceSource source, 
                                        int cnt = 1);
      inline void add_nested_resource_ref(DistributedID source, 
                                          int cnt = 1);
      inline bool remove_base_resource_ref(ReferenceSource source, 
                                           int cnt = 1);
      inline bool remove_nested_resource_ref(DistributedID source, 
                                             int cnt = 1);
    public: // some help for manaing physical instances 
      inline bool try_add_base_valid_ref(ReferenceSource source,
                                         bool must_be_valid,
                                         int cnt = 1);
      bool try_active_deletion(void);
    private:
      void add_gc_reference(void);
      bool remove_gc_reference(void);
    private:
      void add_valid_reference(void);
      bool remove_valid_reference(void);
      bool try_add_valid_reference(bool must_be_valid, int cnt);
    private:
      void add_resource_reference(void);
      bool remove_resource_reference(void);
#ifdef USE_REMOTE_REFERENCES
    private:
      bool add_create_reference(AddressSpaceID source,
                                AddressSpaceID target, ReferenceKind kind);
      bool remove_create_reference(AddressSpaceID source,
                                   AddressSpaceID target, ReferenceKind kind);
#endif
    public:
      // Methods for changing state
      virtual void notify_active(void) = 0;
      virtual void notify_inactive(void) = 0;
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      inline bool is_owner(void) const { return (owner_space == local_space); }
      inline Event get_destruction_event(void) const 
        { return destruction_event; }
      bool has_remote_instance(AddressSpaceID remote_space) const;
      void update_remote_instances(AddressSpaceID remote_space);
    public:
      template<typename FUNCTOR>
      inline void map_over_remote_instances(FUNCTOR &functor);
    public:
      // This is for the owner node only
      void register_remote_instance(AddressSpaceID source, Event destroy_event);
      void register_with_runtime(void);
    public:
      virtual void send_remote_registration(void);
      void send_remote_valid_update(AddressSpaceID target, 
                                    unsigned count, bool add);
      void send_remote_gc_update(AddressSpaceID target,
                                 unsigned count, bool add);
      void send_remote_resource_update(AddressSpaceID target,
                                       unsigned count, bool add);
#ifdef USE_REMOTE_REFERENCES
    public:
      ReferenceKind send_create_reference(AddressSpaceID target);
      void post_create_reference(ReferenceKind kind, 
                                 AddressSpaceID target, bool flush);
#endif
    public:
      static void handle_did_remote_registration(Runtime *runtime,
                                                 Deserializer &derez,
                                                 AddressSpaceID source);
      static void handle_did_remote_valid_update(Runtime *runtime,
                                                 Deserializer &derez);
      static void handle_did_remote_gc_update(Runtime *runtime,
                                              Deserializer &derez);
      static void handle_did_remote_resource_update(Runtime *runtime,
                                                    Deserializer &derez);
    public:
      static void handle_did_add_create(Runtime *runtime, 
                                        Deserializer &derez);
      static void handle_did_remove_create(Runtime *runtime, 
                                           Deserializer &derez);
    protected:
      bool update_state(bool &need_activate, bool &need_validate,
                        bool &need_invalidate, bool &need_deactivate,
                        bool &do_deletion);
      bool can_delete(void);
    public:
      Runtime *const runtime;
      const DistributedID did;
      const AddressSpaceID owner_space;
      const AddressSpaceID local_space;
    protected: // derived users can get the gc lock
      Reservation gc_lock;
    private: // derived users can't see the state information
      State current_state;
      bool has_gc_references;
      bool has_valid_references;
      bool has_resource_references;
    private: // derived users can't see the references
      int gc_references;
      int valid_references;
      int resource_references;
#ifdef USE_REMOTE_REFERENCES
    protected:
      // These are only valid on the owner node
      std::map<std::pair<AddressSpaceID/*src*/,
                         AddressSpaceID/*dst*/>,int> create_gc_refs;
      std::map<std::pair<AddressSpaceID/*src*/,
                         AddressSpaceID/*dst*/>,int> create_valid_refs;
#endif
    protected:
      // Track all the remote instances (relative to ourselves) we know about
      NodeSet                  remote_instances;
    protected:
      // Only valid on owner
      std::set<Event>          recycle_events;
    protected:
      // Only matter on remote nodes
      UserEvent destruction_event;
    protected:
      bool registered_with_runtime;
    };

    //--------------------------------------------------------------------------
    // Give some implementations here so things get inlined
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    template<bool ADD>
    static inline void log_base_ref(ReferenceKind kind, DistributedID did,
                                    ReferenceSource src, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      did = LEGION_DISTRIBUTED_ID_FILTER(did);
      if (ADD)
        log_garbage.info("GC Add Base Ref %d %ld %d %d",
                          kind, did, src, cnt);
      else
        log_garbage.info("GC Remove Base Ref %d %ld %d %d",
                          kind, did, src, cnt);
    }

    //--------------------------------------------------------------------------
    template<bool ADD>
    static inline void log_nested_ref(ReferenceKind kind, DistributedID did, 
                                      DistributedID src, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      did = LEGION_DISTRIBUTED_ID_FILTER(did);
      src = LEGION_DISTRIBUTED_ID_FILTER(src);
      if (ADD)
        log_garbage.info("GC Add Nested Ref %d %ld %ld %d",
                          kind, did, src, cnt);
      else
        log_garbage.info("GC Remove Nested Ref %d %ld %ld %d",
                          kind, did, src, cnt);
    }

    //--------------------------------------------------------------------------
    inline void Collectable::add_reference(unsigned cnt /*= 1*/)
    //--------------------------------------------------------------------------
    {
      __sync_add_and_fetch(&references,cnt);
    }

    //--------------------------------------------------------------------------
    inline bool Collectable::remove_reference(unsigned cnt /*= 1*/)
    //--------------------------------------------------------------------------
    {
      unsigned prev = __sync_fetch_and_sub(&references,cnt);
#ifdef DEBUG_LEGION
      assert(prev >= cnt); // check for underflow
#endif
      // If previous is equal to count, the value is now
      // zero so it is safe to reclaim this object
      return (prev == cnt);
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
    template<ReferenceKind REF_KIND, bool ADD>
    void DistributedCollectable::UpdateReferenceFunctor<REF_KIND,ADD>::apply(
                                                          AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      switch (REF_KIND)
      {
        case GC_REF_KIND:
          {
            source->send_remote_gc_update(target, count, ADD);
            break;
          }
        case VALID_REF_KIND:
          {
            source->send_remote_valid_update(target, count, ADD);
            break;
          }
        case RESOURCE_REF_KIND:
          {
            source->send_remote_resource_update(target, count, ADD);
            break;
          }
        default:
          assert(false); // should never get here
      }
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
      log_base_ref<true>(GC_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&gc_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_gc_reference();
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
      log_nested_ref<true>(GC_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&gc_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_gc_reference();
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
      log_base_ref<false>(GC_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&gc_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_gc_reference();
      return false;
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
      log_nested_ref<false>(GC_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&gc_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_gc_reference();
      return false;
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_valid_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&valid_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_valid_reference();
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_valid_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&valid_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_valid_reference();
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_valid_ref(
                                         ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<false>(VALID_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&valid_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_valid_reference();
      return false;
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_valid_ref(
                                           DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<false>(VALID_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&valid_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_valid_reference();
      return false;
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
      log_base_ref<true>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&resource_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_resource_reference();
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
      log_nested_ref<true>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&resource_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_resource_reference();
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
      log_base_ref<false>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&resource_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_resource_reference();
      return false;
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
      log_nested_ref<false>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      int previous = __sync_fetch_and_add(&resource_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_resource_reference();
      return false;
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::try_add_base_valid_ref(
                      ReferenceSource source, bool must_be_valid, int cnt/*=1*/) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      bool result = try_add_valid_reference(must_be_valid, cnt);
      if (result)
        log_base_ref<true>(VALID_REF_KIND, did, source, cnt);
      return result; 
#else
      return try_add_valid_reference(must_be_valid, cnt);
#endif
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_GARBAGE_COLLECTION__

// EOF

