/* Copyright 2018 Stanford University, NVIDIA Corporation
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

    enum DistCollectableType {
      INSTANCE_MANAGER_DC = 0x1,
      REDUCTION_FOLD_DC = 0x2,
      REDUCTION_LIST_DC = 0x3,
      MATERIALIZED_VIEW_DC = 0x4,
      REDUCTION_VIEW_DC = 0x5,
      COMPOSITE_VIEW_DC = 0x6,
      FILL_VIEW_DC = 0x7,
      PHI_VIEW_DC = 0x8,
      VERSION_STATE_DC = 0x9,
      FUTURE_DC = 0xA,
      FUTURE_MAP_DC = 0xB,
      INDEX_TREE_NODE_DC = 0xC,
      FIELD_SPACE_DC = 0xD,
      REGION_TREE_NODE_DC = 0xE,
      DIST_TYPE_LAST_DC,
    };

    enum ReferenceSource {
      FUTURE_HANDLE_REF = 0,
      DEFERRED_TASK_REF = 1,
      VERSION_MANAGER_REF = 2,
      VERSION_INFO_REF = 3,
      PHYSICAL_STATE_REF = 4,
      PHYSICAL_REGION_REF = 5,
      PENDING_GC_REF = 6,
      REMOTE_DID_REF = 7,
      PENDING_COLLECTIVE_REF = 8,
      MEMORY_MANAGER_REF = 9,
      COMPOSITE_NODE_REF = 10,
      COMPOSITE_HANDLE_REF = 11,
      REMOTE_CREATE_REF = 12,
      INSTANCE_MAPPER_REF = 13,
      APPLICATION_REF = 14,
      MAPPING_ACQUIRE_REF = 15,
      NEVER_GC_REF = 16,
      CONTEXT_REF = 17,
      RESTRICTED_REF = 18,
      VERSION_STATE_TREE_REF = 19,
      PHYSICAL_MANAGER_REF = 20,
      LOGICAL_VIEW_REF = 21,
      REGION_TREE_REF = 22,
      LAST_SOURCE_REF = 23,
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
      "Version Manager Reference",                  \
      "Version Info Reference",                     \
      "Physical State Reference",                   \
      "Physical Region Reference",                  \
      "Pending GC Reference",                       \
      "Remote Distributed ID Reference",            \
      "Pending Collective Reference",               \
      "Memory Manager Reference",                   \
      "Composite Node Reference",                   \
      "Composite Handle Reference",                 \
      "Remote Creation Reference",                  \
      "Instance Mapper Reference",                  \
      "Application Reference",                      \
      "Mapping Acquire Reference",                  \
      "Never GC Reference",                         \
      "Context Reference",                          \
      "Restricted Reference",                       \
      "Version State Tree Reference",               \
      "Physical Manager Reference",                 \
      "Logical View Reference",                     \
      "Region Tree Reference",                      \
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

    /**
     * \interface ReferenceMutator
     * This interface is used for capturing the effects
     * of mutating the state of references for distributed
     * collectable objects such as a message needing 
     * to be send to another node as a consequence of 
     * adding another reference.
     */
    class ReferenceMutator {
    public:
      virtual void record_reference_mutation_effect(RtEvent event) = 0;
    };

    /**
     * \class LocalReferenceMutator 
     * This is an implementation of the ReferenceMutator
     * interface for handling local effects that should 
     * be waiting on locally.
     */
    class LocalReferenceMutator : public ReferenceMutator {
    public:
      LocalReferenceMutator(void) { }
      LocalReferenceMutator(const LocalReferenceMutator &rhs);
      ~LocalReferenceMutator(void);
    public:
      LocalReferenceMutator& operator=(const LocalReferenceMutator &rhs);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      RtEvent get_done_event(void);
    private:
      std::set<RtEvent> mutation_effects;
    };

    /**
     * \class WrapperReferenceMutator
     * This wraps a target set of runtime events to update
     */
    class WrapperReferenceMutator : public ReferenceMutator {
    public:
      WrapperReferenceMutator(std::set<RtEvent> &targets)
        : mutation_effects(targets) { }
      WrapperReferenceMutator(const WrapperReferenceMutator &rhs);
      ~WrapperReferenceMutator(void) { }
    public:
      WrapperReferenceMutator& operator=(const WrapperReferenceMutator &rhs);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    private:
      std::set<RtEvent> &mutation_effects;
    };

    /**
     * \class IgnoreReferenceMutator
     * This will ignore any reference effects
     */
    class IgnoreReferenceMutator : public ReferenceMutator {
    public:
      virtual void record_reference_mutation_effect(RtEvent event) { }
    };

    /**
     * \class Distributed Collectable
     * This is the base class for handling all the reference
     * counting logic for any objects in the runtime which 
     * have distributed copies of themselves across multiple
     * nodes and therefore have to have a distributed 
     * reference counting scheme to determine when they 
     * can be collected.
     */
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
        UpdateReferenceFunctor(DistributedCollectable *dc, 
                               ReferenceMutator *m,
                               unsigned cnt = 1)
          : source(dc), mutator(m), count(cnt) { }
      public:
        inline void apply(AddressSpaceID target);
      protected:
        DistributedCollectable *const source;
        ReferenceMutator *const mutator;
        unsigned count;
      };
      class RemoteInvalidateFunctor {
      public:
        RemoteInvalidateFunctor(DistributedCollectable *dc,
                                ReferenceMutator *m)
          : source(dc), mutator(m) { }
      public:
        inline void apply(AddressSpaceID target);
      protected:
        DistributedCollectable *const source;
        ReferenceMutator *const mutator;
      };
      class RemoteDeactivateFunctor {
      public:
        RemoteDeactivateFunctor(DistributedCollectable *dc,
                                ReferenceMutator *m)
          : source(dc), mutator(m) { }
      public:
        inline void apply(AddressSpaceID target);
      protected:
        DistributedCollectable *const source;
        ReferenceMutator *const mutator;
      };
      class UnregisterFunctor {
      public:
        UnregisterFunctor(Runtime *rt, const DistributedID d,
                          VirtualChannelKind v, std::set<RtEvent> &done)
          : runtime(rt), did(d), vc(v), done_events(done) { }
      public:
        void apply(AddressSpaceID target);
      protected:
        Runtime *const runtime;
        const DistributedID did;
        const VirtualChannelKind vc;
        std::set<RtEvent> &done_events;
      };
    public:
      DistributedCollectable(Runtime *rt, DistributedID did,
                             AddressSpaceID owner_space,
                             bool register_with_runtime = true);
      DistributedCollectable(const DistributedCollectable &rhs);
      virtual ~DistributedCollectable(void);
    public:
      inline void add_base_gc_ref(ReferenceSource source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
      inline void add_nested_gc_ref(DistributedID source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
      inline bool remove_base_gc_ref(ReferenceSource source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
      inline bool remove_nested_gc_ref(DistributedID source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
    public:
      inline void add_base_valid_ref(ReferenceSource source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
      inline bool remove_base_valid_ref(ReferenceSource source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, 
                                  ReferenceMutator *mutator = NULL,int cnt = 1);
    public:
      inline void add_base_resource_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_resource_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_resource_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_resource_ref(DistributedID source, int cnt = 1);
    public: // some help for managing physical instances 
      inline bool try_add_base_valid_ref(ReferenceSource source,
                                         ReferenceMutator *mutator,
                                         bool must_be_valid,
                                         int cnt = 1);
      bool try_active_deletion(void);
    private:
      void add_gc_reference(ReferenceMutator *mutator);
      bool remove_gc_reference(ReferenceMutator *mutator);
    private:
      void add_valid_reference(ReferenceMutator *mutator);
      bool remove_valid_reference(ReferenceMutator *mutator);
      bool try_add_valid_reference(ReferenceMutator *mutator,
                                   bool must_be_valid, int cnt);
    private:
      void add_resource_reference(void);
      bool remove_resource_reference(void);
#ifdef USE_REMOTE_REFERENCES
    private:
      bool add_create_reference(AddressSpaceID source, 
          ReferenceMutator *mutator, AddressSpaceID target, ReferenceKind kind);
      bool remove_create_reference(AddressSpaceID source,
          ReferenceMutator *mutator, AddressSpaceID target, ReferenceKind kind);
#endif
#ifdef DEBUG_LEGION_GC
    private:
      void add_base_gc_ref_internal(ReferenceSource source, 
                                    ReferenceMutator *mutator, int cnt);
      void add_nested_gc_ref_internal(DistributedID source, 
                                    ReferenceMutator *mutator, int cnt);
      bool remove_base_gc_ref_internal(ReferenceSource source, 
                                    ReferenceMutator *mutator, int cnt);
      bool remove_nested_gc_ref_internal(DistributedID source, 
                                    ReferenceMutator *mutator, int cnt);
    public:
      void add_base_valid_ref_internal(ReferenceSource source, 
                                    ReferenceMutator *mutator, int cnt);
      void add_nested_valid_ref_internal(DistributedID source, 
                                    ReferenceMutator *mutator, int cnt);
      bool remove_base_valid_ref_internal(ReferenceSource source, 
                                    ReferenceMutator *mutator, int cnt);
      bool remove_nested_valid_ref_internal(DistributedID source, 
                                    ReferenceMutator *mutator, int cnt);
      bool try_add_valid_reference_internal(ReferenceSource source, 
                                            ReferenceMutator *mutator, 
                                            bool must_be_valid, int cnt);
    public:
      void add_base_resource_ref_internal(ReferenceSource source, int cnt); 
      void add_nested_resource_ref_internal(DistributedID source, int cnt); 
      bool remove_base_resource_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_resource_ref_internal(DistributedID source, int cnt); 
#endif
    public:
      // Methods for changing state
      virtual void notify_active(ReferenceMutator *mutator) = 0;
      virtual void notify_inactive(ReferenceMutator *mutator) = 0;
      virtual void notify_valid(ReferenceMutator *mutator) = 0;
      virtual void notify_invalid(ReferenceMutator *mutator) = 0;
      // For explicit remote messages
      virtual void notify_remote_inactive(ReferenceMutator *mutator);
      virtual void notify_remote_invalid(ReferenceMutator *mutator);
    public:
      inline bool is_owner(void) const { return (owner_space == local_space); }
      inline bool is_registered(void) const { return registered_with_runtime; }
      bool has_remote_instance(AddressSpaceID remote_space) const;
      void update_remote_instances(AddressSpaceID remote_space);
    public:
      inline bool has_remote_instances(void) const;
      template<typename FUNCTOR>
      inline void map_over_remote_instances(FUNCTOR &functor);
    public:
      // This is for the owner node only
      void register_with_runtime(ReferenceMutator *mutator);
    protected:
      void unregister_with_runtime(void) const;
      RtEvent send_unregister_messages(VirtualChannelKind vc) const;
    public:
      // This for remote nodes only
      void unregister_collectable(void);
      static void handle_unregister_collectable(Runtime *runtime,
                                                Deserializer &derez);
    public:
      virtual void send_remote_registration(ReferenceMutator *mutator);
      void send_remote_valid_update(AddressSpaceID target, 
                                    ReferenceMutator *mutator,
                                    unsigned count, bool add);
      void send_remote_gc_update(AddressSpaceID target,
                                 ReferenceMutator *mutator,
                                 unsigned count, bool add);
      void send_remote_resource_update(AddressSpaceID target,
                                       unsigned count, bool add);
      void send_remote_invalidate(AddressSpaceID target,
                                  ReferenceMutator *mutator);
      void send_remote_deactivate(AddressSpaceID target,
                                  ReferenceMutator *mutator);
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
      static void handle_did_remote_invalidate(Runtime *runtime,
                                               Deserializer &derez);
      static void handle_did_remote_deactivate(Runtime *runtime,
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
      mutable LocalLock gc_lock;
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
#ifdef DEBUG_LEGION_GC
    protected:
      std::map<ReferenceSource,int> detailed_base_gc_references;
      std::map<DistributedID,int> detailed_nested_gc_references;
      std::map<ReferenceSource,int> detailed_base_valid_references;
      std::map<DistributedID,int> detailed_nested_valid_references;
      std::map<ReferenceSource,int> detailed_base_resource_references;
      std::map<DistributedID,int> detailed_nested_resource_references;
#endif
    protected:
      // Track all the remote instances (relative to ourselves) we know about
      NodeSet                  remote_instances;
    protected:
      mutable bool registered_with_runtime;
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
    inline bool DistributedCollectable::has_remote_instances(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      return !remote_instances.empty();
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
    inline void DistributedCollectable::RemoteInvalidateFunctor::apply(
                                                          AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      source->send_remote_invalidate(target, mutator);
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::RemoteDeactivateFunctor::apply(
                                                          AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      source->send_remote_deactivate(target, mutator);
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
            source->send_remote_gc_update(target, mutator, count, ADD);
            break;
          }
        case VALID_REF_KIND:
          {
            source->send_remote_valid_update(target, mutator, count, ADD);
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
                                      ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&gc_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_gc_reference(mutator);
#else
      add_base_gc_ref_internal(source, mutator, cnt); 
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_gc_ref(
                DistributedID source, ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&gc_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_gc_reference(mutator);
#else
      add_nested_gc_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), 
                                 mutator, cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_gc_ref(
              ReferenceSource source, ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<false>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&gc_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_gc_reference(mutator);
      return false;
#else
      return remove_base_gc_ref_internal(source, mutator, cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_gc_ref(
                DistributedID source, ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<false>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&gc_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_gc_reference(mutator);
      return false;
#else
      return remove_nested_gc_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), mutator, cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_valid_ref(
              ReferenceSource source, ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&valid_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_valid_reference(mutator);
#else
      add_base_valid_ref_internal(source, mutator, cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_valid_ref(
                DistributedID source, ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&valid_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_valid_reference(mutator);
#else
      add_nested_valid_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), 
                                    mutator, cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_valid_ref(
              ReferenceSource source, ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_base_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&valid_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_valid_reference(mutator);
      return false;
#else
      return remove_base_valid_ref_internal(source, mutator, cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_valid_ref(
                DistributedID source, ReferenceMutator *mutator, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
      log_nested_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&valid_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_valid_reference(mutator);
      return false;
#else
      return remove_nested_valid_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), mutator, cnt);
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
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&resource_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_resource_reference();
#else
      add_base_resource_ref_internal(source, cnt);
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
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&resource_references, cnt);
#ifdef DEBUG_LEGION
      assert(previous >= 0);
#endif
      if (previous == 0)
        add_resource_reference();
#else
      add_nested_resource_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
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
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&resource_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_resource_reference();
      return false;
#else
      return remove_base_resource_ref_internal(source, cnt);
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
#ifndef DEBUG_LEGION_GC
      int previous = __sync_fetch_and_add(&resource_references, -cnt);
#ifdef DEBUG_LEGION
      assert(previous >= cnt);
#endif
      if (previous == cnt)
        return remove_resource_reference();
      return false;
#else
      return remove_nested_resource_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::try_add_base_valid_ref(
                            ReferenceSource source, ReferenceMutator *mutator, 
                            bool must_be_valid, int cnt/*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifdef LEGION_GC
#ifdef DEBUG_LEGION_GC
      bool result = try_add_valid_reference_internal(source, mutator, 
                                                     must_be_valid, cnt);
#else
      bool result = try_add_valid_reference(mutator, must_be_valid, cnt);
#endif
      if (result)
        log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
      return result; 
#else
#ifdef DEBUG_LEGION_GC
      return try_add_valid_reference_internal(source,mutator,must_be_valid,cnt);
#else
      return try_add_valid_reference(mutator, must_be_valid, cnt);
#endif
#endif
    }

  }; // namespace Internal 
}; // namespace Legion 

#endif // __LEGION_GARBAGE_COLLECTION__

// EOF

