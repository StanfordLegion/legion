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

namespace LegionRuntime {
  namespace HighLevel {

    enum ReferenceSource {
      FUTURE_HANDLE_REF,
      DEFERRED_TASK_REF,
      CURRENT_STATE_REF,
      PHYSICAL_STATE_REF,
      PHYSICAL_REGION_REF,
      VERSION_MANAGER_REF,
      PENDING_GC_REF,
      REMOTE_DID_REF,
      PENDING_COLLECTIVE_REF,
      MEMORY_MANAGER_REF,
      COMPOSITE_NODE_REF,
      COMPOSITE_HANDLE_REF,
      PERSISTENCE_REF,
      INITIAL_CREATION_REF,
      REMOTE_CREATE_REF,
      LAST_SOURCE_REF,
    };

    enum ReferenceKind {
      GC_REF_KIND,
      VALID_REF_KIND,
      RESOURCE_REF_KIND,
    };

#define REFERENCE_NAMES_ARRAY(names)                \
    const char *const names[LAST_SOURCE_REF] = {    \
      "Future Handle Reference",                    \
      "Deferred Task Reference",                    \
      "Current State Reference",                    \
      "Physical State Reference",                   \
      "Physical Region Reference",                  \
      "Version Manager Reference",                  \
      "Pending GC Reference",                       \
      "Remote Distributed ID Reference",            \
      "Pending Collective Reference",               \
      "Memory Manager Reference"                    \
      "Composite Node Reference",                   \
      "Composite Handle Reference",                 \
      "Persistent Reference",                       \
      "Initial Creation Reference",                 \
      "Remote Creation Reference",                  \
    }

    extern Logger::Category log_garbage;

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
        VALID_STATE,
        PENDING_ACTIVE_STATE,
        PENDING_INACTIVE_STATE,
        PENDING_VALID_STATE,
        PENDING_INVALID_STATE,
        DELETED_STATE,
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
      DistributedCollectable(Internal *rt, DistributedID did,
                             AddressSpaceID owner_space,
                             AddressSpaceID local_space,
                             bool register_with_runtime = true);
      virtual ~DistributedCollectable(void);
    public:
      inline void add_base_gc_ref(ReferenceSource source, unsigned cnt = 1);
      inline void add_nested_gc_ref(DistributedID source, unsigned cnt = 1);
      inline bool remove_base_gc_ref(ReferenceSource source, unsigned cnt = 1);
      inline bool remove_nested_gc_ref(DistributedID source, unsigned cnt = 1);
    public:
      inline void add_base_valid_ref(ReferenceSource source, unsigned cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, unsigned cnt = 1);
      inline bool remove_base_valid_ref(ReferenceSource source, 
                                        unsigned cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, 
                                          unsigned cnt = 1);
    public:
      inline void add_base_resource_ref(ReferenceSource source, 
                                        unsigned cnt = 1);
      inline void add_nested_resource_ref(DistributedID source, 
                                          unsigned cnt = 1);
      inline bool remove_base_resource_ref(ReferenceSource source, 
                                           unsigned cnt = 1);
      inline bool remove_nested_resource_ref(DistributedID source, 
                                             unsigned cnt = 1);
    private:
      void add_gc_reference(unsigned cnt);
      bool remove_gc_reference(unsigned cnt);
    private:
      void add_valid_reference(unsigned cnt);
      bool remove_valid_reference(unsigned cnt);
    private:
      void add_resource_reference(unsigned cnt);
      bool remove_resource_reference(unsigned cnt);
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
      static void handle_did_remote_registration(Internal *runtime,
                                                 Deserializer &derez,
                                                 AddressSpaceID source);
      static void handle_did_remote_valid_update(Internal *runtime,
                                                 Deserializer &derez);
      static void handle_did_remote_gc_update(Internal *runtime,
                                              Deserializer &derez);
      static void handle_did_remote_resource_update(Internal *runtime,
                                                    Deserializer &derez);
    public:
      static void handle_did_add_create(Internal *runtime, 
                                        Deserializer &derez);
      static void handle_did_remove_create(Internal *runtime, 
                                           Deserializer &derez);
    protected:
      bool update_state(bool &need_activate, bool &need_validate,
                        bool &need_invalidate, bool &need_deactivate,
                        bool &do_deletion);
      bool can_delete(void);
    public:
      Internal *const runtime;
      const DistributedID did;
      const AddressSpaceID owner_space;
      const AddressSpaceID local_space;
    protected:
      State current_state;
      Reservation gc_lock;
      unsigned gc_references;
      unsigned valid_references;
      unsigned resource_references;
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
#ifdef DEBUG_HIGH_LEVEL
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
                                                        unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_base_ref<true>(GC_REF_KIND, did, source, cnt);
#endif
      add_gc_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_gc_ref(
                                      DistributedID source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_nested_ref<true>(GC_REF_KIND, did, source, cnt);
#endif
      add_gc_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_gc_ref(
                                    ReferenceSource source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_base_ref<false>(GC_REF_KIND, did, source, cnt);
#endif
      return remove_gc_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_gc_ref(
                                      DistributedID source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_nested_ref<false>(GC_REF_KIND, did, source, cnt);
#endif
      return remove_gc_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_valid_ref(
                                    ReferenceSource source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, source, cnt);
#endif
      add_valid_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_valid_ref(
                                      DistributedID source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, source, cnt);
#endif
      add_valid_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_valid_ref(
                                    ReferenceSource source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_base_ref<false>(VALID_REF_KIND, did, source, cnt);
#endif
      return remove_valid_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_valid_ref(
                                      DistributedID source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_nested_ref<false>(VALID_REF_KIND, did, source, cnt);
#endif
      return remove_valid_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_resource_ref(
                                    ReferenceSource source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_base_ref<true>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      add_resource_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_resource_ref(
                                      DistributedID source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_nested_ref<true>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      add_resource_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_resource_ref(
                                    ReferenceSource source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_base_ref<false>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      return remove_resource_reference(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_resource_ref(
                                      DistributedID source, unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_nested_ref<false>(RESOURCE_REF_KIND, did, source, cnt);
#endif
      return remove_resource_reference(cnt);
    }

  }; // namespace HighLevel 
}; // namespace LegionRuntime

#endif // __LEGION_GARBAGE_COLLECTION__

// EOF

