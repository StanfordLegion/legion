/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_SYNC_IMPL_H__
#define __LEGION_SYNC_IMPL_H__

#include "legion/api/sync.h"
#include "legion/contexts/context.h"
#include "legion/kernel/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \class GrantImpl
     * This is the base implementation of a grant object.
     * The grant implementation remembers the locks that
     * must be acquired and gives out an precondition event
     * for acquiring the locks whenever a user attempts
     * to register as using the grant.  Registering requires
     * providing a completion event for the operation which
     * the grant object then knows to use when releasing the
     * locks.  Grants continues accepting registrations
     * until the runtime marks that it is no longer active.
     */
    class GrantImpl : public Collectable,
                      public Heapify<GrantImpl, SHORT_LIFETIME> {
    public:
      struct ReservationRequest {
      public:
        ReservationRequest(void)
          : reservation(Reservation::NO_RESERVATION), mode(0), exclusive(true)
        { }
        ReservationRequest(Reservation r, unsigned m, bool e)
          : reservation(r), mode(m), exclusive(e)
        { }
      public:
        Reservation reservation;
        unsigned mode;
        bool exclusive;
      };
    public:
      GrantImpl(void);
      GrantImpl(const std::vector<ReservationRequest>& requests);
      GrantImpl(const GrantImpl& rhs) = delete;
      ~GrantImpl(void);
    public:
      GrantImpl& operator=(const GrantImpl& rhs) = delete;
    public:
      void register_operation(ApEvent completion_event);
      ApEvent acquire_grant(void);
      void release_grant(void);
    public:
      void pack_grant(Serializer& rez);
      void unpack_grant(Deserializer& derez);
    private:
      std::vector<ReservationRequest> requests;
      bool acquired;
      ApEvent grant_event;
      std::set<ApEvent> completion_events;
      mutable LocalLock grant_lock;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SYNC_IMPL_H__
