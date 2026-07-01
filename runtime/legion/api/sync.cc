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

#include "legion/api/sync_impl.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Lock
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  Lock::Lock(void) : reservation_lock(Reservation::NO_RESERVATION)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Lock::Lock(Reservation r) : reservation_lock(r)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  bool Lock::operator<(const Lock& rhs) const
  //--------------------------------------------------------------------------
  {
    return (reservation_lock < rhs.reservation_lock);
  }

  //--------------------------------------------------------------------------
  bool Lock::operator==(const Lock& rhs) const
  //--------------------------------------------------------------------------
  {
    return (reservation_lock == rhs.reservation_lock);
  }

  //--------------------------------------------------------------------------
  void Lock::acquire(unsigned mode /*=0*/, bool exclusive /*=true*/)
  //--------------------------------------------------------------------------
  {
    legion_assert(reservation_lock.exists());
    Internal::ApEvent lock_event(reservation_lock.acquire(mode, exclusive));
    bool poisoned = false;
    lock_event.wait_faultaware(poisoned);
    if (poisoned)
      Internal::implicit_context->raise_poison_exception();
  }

  //--------------------------------------------------------------------------
  void Lock::release(void)
  //--------------------------------------------------------------------------
  {
    legion_assert(reservation_lock.exists());
    reservation_lock.release();
  }

  /////////////////////////////////////////////////////////////
  // Lock Request
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  LockRequest::LockRequest(Lock l, unsigned m, bool excl)
    : lock(l), mode(m), exclusive(excl)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // Grant
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  Grant::Grant(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Grant::Grant(Internal::GrantImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  Grant::Grant(const Grant& rhs) : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  Grant::~Grant(void)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
      impl = nullptr;
    }
  }

  //--------------------------------------------------------------------------
  Grant& Grant::operator=(const Grant& rhs)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
    }
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  /////////////////////////////////////////////////////////////
  // Phase Barrier
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  PhaseBarrier::PhaseBarrier(void)
    : phase_barrier(Internal::ApBarrier::NO_AP_BARRIER)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  PhaseBarrier::PhaseBarrier(Internal::ApBarrier b) : phase_barrier(b)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  bool PhaseBarrier::operator<(const PhaseBarrier& rhs) const
  //--------------------------------------------------------------------------
  {
    return (phase_barrier < rhs.phase_barrier);
  }

  //--------------------------------------------------------------------------
  bool PhaseBarrier::operator==(const PhaseBarrier& rhs) const
  //--------------------------------------------------------------------------
  {
    return (phase_barrier == rhs.phase_barrier);
  }

  //--------------------------------------------------------------------------
  bool PhaseBarrier::operator!=(const PhaseBarrier& rhs) const
  //--------------------------------------------------------------------------
  {
    return (phase_barrier != rhs.phase_barrier);
  }

  //--------------------------------------------------------------------------
  void PhaseBarrier::arrive(unsigned count /*=1*/)
  //--------------------------------------------------------------------------
  {
    legion_assert(phase_barrier.exists());
    Internal::runtime->phase_barrier_arrive(*this, count);
  }

  //--------------------------------------------------------------------------
  void PhaseBarrier::wait(void)
  //--------------------------------------------------------------------------
  {
    legion_assert(phase_barrier.exists());
    Internal::ApEvent e = Internal::Runtime::get_previous_phase(*this);
    bool poisoned = false;
    e.wait_faultaware(poisoned);
    if (poisoned)
      Internal::implicit_context->raise_poison_exception();
  }

  //--------------------------------------------------------------------------
  void PhaseBarrier::alter_arrival_count(int delta)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::alter_arrival_count(*this, delta);
  }

  //--------------------------------------------------------------------------
  bool PhaseBarrier::exists(void) const
  //--------------------------------------------------------------------------
  {
    return phase_barrier.exists();
  }

  /////////////////////////////////////////////////////////////
  // Dynamic Collective
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  DynamicCollective::DynamicCollective(void) : PhaseBarrier(), redop(0)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  DynamicCollective::DynamicCollective(Internal::ApBarrier b, ReductionOpID r)
    : PhaseBarrier(b), redop(r)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  void DynamicCollective::arrive(
      const void* value, size_t size, unsigned count /*=1*/)
  //--------------------------------------------------------------------------
  {
    Internal::runtime->phase_barrier_arrive(
        *this, count, Internal::ApEvent::NO_AP_EVENT, value, size);
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Grant Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GrantImpl::GrantImpl(void) : acquired(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    GrantImpl::GrantImpl(const std::vector<ReservationRequest>& reqs)
      : requests(reqs), acquired(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    GrantImpl::~GrantImpl(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void GrantImpl::register_operation(ApEvent completion_event)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      completion_events.insert(completion_event);
    }

    //--------------------------------------------------------------------------
    ApEvent GrantImpl::acquire_grant(void)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      if (!acquired)
      {
        grant_event = ApEvent::NO_AP_EVENT;
        for (const ReservationRequest& request : requests)
        {
          grant_event = ApEvent(request.reservation.acquire(
              request.mode, request.exclusive, grant_event));
        }
        acquired = true;
      }
      return grant_event;
    }

    //--------------------------------------------------------------------------
    void GrantImpl::release_grant(void)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      ApEvent deferred_release =
          Runtime::merge_events(nullptr, completion_events);
      for (const ReservationRequest& request : requests)
      {
        request.reservation.release(deferred_release);
      }
    }

    //--------------------------------------------------------------------------
    void GrantImpl::pack_grant(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      ApEvent pack_event = acquire_grant();
      rez.serialize(pack_event);
    }

    //--------------------------------------------------------------------------
    void GrantImpl::unpack_grant(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ApEvent unpack_event;
      derez.deserialize(unpack_event);
      AutoLock g_lock(grant_lock);
      legion_assert(!acquired);
      grant_event = unpack_event;
      acquired = true;
    }

  }  // namespace Internal
}  // namespace Legion
