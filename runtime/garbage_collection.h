/* Copyright 2014 Stanford University
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

namespace LegionRuntime {
  namespace HighLevel {

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
     * This is a base class for both Distributed and Hierarchical
     * collectable classes that implements the state machine for
     * knowing when to invoke the various notify methods indicating
     * that a change of state has occurred.  It is oblivious to 
     * the ABA problem and relies on the client to avoid cases
     * where ABA races could result in errors.
     */
    class CollectableState {
    public:
      enum State {
        INACTIVE_STATE,
        ACTIVE_INVALID_STATE,
        VALID_STATE,
        PENDING_ACTIVE_STATE,
        PENDING_INACTIVE_STATE,
        PENDING_VALID_STATE,
        PENDING_INVALID_STATE,
      };
    public:
      CollectableState(void);
      virtual ~CollectableState(void);
    protected:
      bool update_state(bool has_gc_references, 
                        bool has_remote_references,
                        bool has_valid_references,
                        bool has_resource_references,
                        bool &need_activate, bool &need_validate,
                        bool &need_invalidate, bool &need_deactivate,
                        bool &do_delete);
      bool can_delete(bool has_gc_references,
                      bool has_remote_references,
                      bool has_valid_references,
                      bool has_resource_references);
    protected:
      State current_state;
    };
    
    /**
     * \class DistributedCollectable
     * This class implements a distributed reference counting
     * scheme for garbage collection and deletion of resources.
     * It relies on a distributed protocol with the notion of
     * an owner instance and remote instances.  Removal of
     * reference counts include the name of the node on which
     * the removal came from in order to guarantee that there
     * are no premature deletions of the resource.  This class
     * is currently used for garbage collecting physical managers
     * and the instances that they own.
     */
    class DistributedCollectable : public CollectableState {
    public:
      DistributedCollectable(Runtime *rt, DistributedID did,
                             AddressSpaceID owner_space,
                             AddressSpaceID local_space);
      virtual ~DistributedCollectable(void);
    public:
      void add_gc_reference(unsigned cnt = 1);
      bool remove_gc_reference(unsigned cnt = 1);
    public:
      void add_valid_reference(unsigned cnt = 1);
      bool remove_valid_reference(unsigned cnt = 1);
    public:
      void add_resource_reference(unsigned cnt = 1);
      bool remove_resource_reference(unsigned cnt = 1);
    public:
      bool add_remote_reference(AddressSpaceID sid, unsigned cnt = 1);
      bool remove_remote_reference(AddressSpaceID sid, Event dest_event,
                                   unsigned cnt = 1);
    public:
      void add_held_remote_reference(unsigned cnt = 1);
      // Notify the owner of a remote reference sent somewhere else
      // Return true if the reference actually needs to be sent
      bool send_remote_reference(AddressSpaceID sid, unsigned cnt = 1);
      // Update the people who we know have this collectable
      void update_remote_spaces(AddressSpaceID sid);
    protected:
      // Must be called while holding the gc lock
      void return_held_references(void); 
    public:
      virtual void notify_activate(void) = 0;
      virtual void garbage_collect(void) = 0;
    public:
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      // Will only be called on the owner
      virtual void notify_new_remote(AddressSpaceID sid) = 0;
    public:
      static void process_remove_resource_reference(Runtime *rt,
                                                    Deserializer &derez);
      static void process_remove_remote_reference(Runtime *rt,
                                                  AddressSpaceID source,
                                                  Deserializer &derez);
      static void process_add_remote_reference(Runtime *rt,
                                               Deserializer &derez);
    public:
      Runtime *const runtime;
      const DistributedID did;
      const AddressSpaceID owner_space;
      const AddressSpaceID local_space;
      const bool owner;
    protected:
      Reservation gc_lock;
      unsigned gc_references;
      unsigned valid_references;
      unsigned resource_references;
      // Places where we know there are remote instances
      std::set<AddressSpaceID> remote_spaces;
    protected:
      // Only matters on the remote nodes
      unsigned held_remote_references;
      UserEvent destruction_event;
    protected:
      // These only matter on the owner node
      std::map<AddressSpaceID,int> remote_references;
      std::set<Event> recycle_events;
    };

    /**
     * \class HierarchicalCollectable 
     * This class implements the basis for a hierarchical reference
     * counting scheme for garbage collection and deletion
     * of resources.  It is used for implementing garbage collection of 
     * views in Legion's distributed system.  In conjunction with
     * Legion's natural task hierarchy, the views of particular physical
     * instance from a given logical region's perspective maintain
     * their own heirarchy and only once all of the references have been
     * removed from the view at the top of the hierarchy, can the reference
     * to the manager be removed.
     *
     * There are three kinds of references:
     *  - garbage collection references
     *  - remote references
     *  - resource references
     *  The first two references are used for knowing when the resource
     *  being managed by the object can garbage collected.  All three
     *  are used for knowing when this object can be deleted.  The
     *  virtual garbage collection algorithm will be called every
     *  time a remove operation is performed and both the gc
     *  and remote references counts are zero after the remove
     *  finishes.  True will be returned every time a remove
     *  operation is performed and all three sets of references
     *  are zero.
     */
    class HierarchicalCollectable : public CollectableState {
    public:
      // Create an owner collectable
      HierarchicalCollectable(Runtime *rt, DistributedID did,
                              AddressSpaceID owner_addr, 
                              DistributedID owner_did);
      virtual ~HierarchicalCollectable(void);
    public:
      void add_gc_reference(unsigned cnt = 1);
      bool remove_gc_reference(unsigned cnt = 1);
    public:
      void add_valid_reference(unsigned cnt = 1);
      bool remove_valid_reference(unsigned cnt = 1);
    public:
      void add_resource_reference(unsigned cnt = 1);
      bool remove_resource_reference(unsigned cnt = 1);
    public:
      void add_remote_reference(unsigned cnt = 1);
      bool remove_remote_reference(unsigned cnt = 1);
    public:
      void add_subscriber(AddressSpaceID target, 
                          DistributedID subscriber_did);
      void add_held_remote_reference(unsigned cnt = 1);
    public:
      DistributedID find_distributed_id(AddressSpaceID target) const;
      void set_no_free_did(void);
    protected:
      // Must be called while holding the gc lock
      void return_held_references(void);
    public:
      virtual void notify_activate(void) = 0;
      virtual void garbage_collect(void) = 0;
    public:
      virtual void notify_valid(void) = 0;
      virtual void notify_invalid(void) = 0;
    public:
      static void process_remove_resource_reference(Runtime *rt,
                                                    Deserializer &derez);
      static void process_remove_remote_reference(Runtime *rt,
                                                  Deserializer &derez);
    public:
      Runtime *const runtime;
      const DistributedID did;
    protected:
      Reservation gc_lock;
      unsigned gc_references;
      unsigned valid_references;
      unsigned remote_references;
      unsigned resource_references;
    protected:
      // Fields for owner collectables
      std::map<AddressSpaceID,DistributedID> subscribers;
    protected:
      // Fields for remote collectables
      AddressSpaceID owner_addr;
      DistributedID owner_did;
      unsigned held_remote_references;
    protected:
      // Free distributed ID on destruction
      bool free_distributed_id;
    };


    //--------------------------------------------------------------------------
    // Give some implementations here so things get inlined
    //--------------------------------------------------------------------------

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

  }; // namespace HighLevel 
}; // namespace LegionRuntime

#endif // __LEGION_GARBAGE_COLLECTION__

// EOF

