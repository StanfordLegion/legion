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

#include "mapper_manager.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Mapper Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperManager::MapperManager(Runtime *rt, Mapping::Mapper *mp)
      : runtime(rt), mapper(mp), mapper_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapperManager::~MapperManager(void)
    //--------------------------------------------------------------------------
    {
      // We can now delete our mapper
      delete mapper;
      mapper_lock.destroy_reservation();
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperManager::get_index_partition(IndexSpace parent, 
                                               Color color) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_index_subspace(IndexPartition p,Color c) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_subspace(p, c);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_index_subspace(IndexPartition p, 
                                                 const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_subspace(p, color);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_multiple_domains(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    Domain MapperManager::get_index_space_domain(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_domain(handle);
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_index_space_domains(IndexSpace handle,
                                         std::vector<Domain> &domains) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain MapperManager::get_index_partition_color_space(IndexPartition p)const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_index_space_partition_colors(
                               IndexSpace handle, std::set<Color> &colors) const
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_partition_colors(handle, colors);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_index_partition_disjoint(IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return runtime->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_index_space_color(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_color(handle);
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_index_partition_color(IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_parent_index_space(IndexPartition handle)const
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_parent_index_partition(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperManager::get_parent_index_partition(
                                                        IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    size_t MapperManager::get_field_size(FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_field_space_fields(FieldSpace handle, 
                                               std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      runtime->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition(LogicalRegion parent,
                                                   IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition_by_color(
                                           LogicalRegion par, Color color) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition_by_tree(
                                                        IndexPartition part,
                                                        FieldSpace fspace, 
                                                        RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_tree(part, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion(LogicalPartition parent,
                                                       IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion_by_color(
                                        LogicalPartition par, Color color) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion_by_tree(
                   IndexSpace handle, FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_logical_region_color(LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_region_color(handle);
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_logical_partition_color(
                                                  LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_parent_logical_region(
                                                    LogicalPartition part) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_region(part);
    }
    
    //--------------------------------------------------------------------------
    bool MapperManager::has_parent_logical_partition(LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_parent_logical_partition(
                                                          LogicalRegion r) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_partition(r);
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* MapperManager::allocate_call_info(MappingCallKind kind,
                                                       bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock m_lock(mapper_lock);
        return allocate_call_info(kind, false/*need lock*/);
      }
      if (!available_infos.empty())
      {
        MappingCallInfo *result = available_infos.back();
        available_infos.pop_back();
        result->kind = kind;
        return result;
      }
      return new MappingCallInfo(this, kind);
    }

    //--------------------------------------------------------------------------
    void MapperManager::free_call_info(MappingCallInfo *info, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock m_lock(mapper_lock);
        free_call_info(info, false/*need lock*/);
      }
      info->resume = UserEvent::NO_USER_EVENT;
      available_infos.push_back(info);
    }

    /////////////////////////////////////////////////////////////
    // Serializing Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SerializingManager::SerializingManager(Runtime *rt, Mapping::Mapper *mp,
                                           bool init_reentrant)
      : MapperManager(rt, mp), permit_reentrant(init_reentrant), 
        executing_call(NULL), paused_calls(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SerializingManager::SerializingManager(const SerializingManager &rhs)
      : MapperManager(NULL,NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    SerializingManager::~SerializingManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SerializingManager& SerializingManager::operator=(
                                                  const SerializingManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool SerializingManager::is_locked(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Serializing managers are always effectively locked
      return true;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::lock_mapper(MappingCallInfo *info, bool read_only)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void SerializingManager::unlock_mapper(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void SerializingManager::is_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      // No need to hold the lock here since we are exclusive
      return permit_reentrant;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::enable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      // No need to hold the lock since we know we are exclusive 
      permit_reentrant = true;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::disable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      // No need to hold the lock since we know we are exclusive
      if (permit_reentrant)
      {
        // If there are paused calls, we need to wait for them all 
        // to finish before we can 
        if (paused_calls > 0)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!info->resume.exists());
#endif
          UserEvent *ready_event = UserEvent::create_user_event();
          info->resume = ready_event;
          non_reentrant_calls.push_back(info);
          ready_event.wait();
          // When we wake up, we should be non-reentrant
#ifdef DEBUG_HIGH_LEVEL
          assert(!permit_reentrant);
#endif
        }
        else
          permit_reentrant = false;
      }
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* SerializingManager::begin_mapper_call(MappingCallKind kind)
    //--------------------------------------------------------------------------
    {
      Event wait_on = Event::NO_EVENT;  
      MappingCallInfo *result = NULL;
      {
        AutoLock m_lock(mapper_lock);
        result = allocate_call_info(kind, false/*need lock*/);
        // See if we are ready to run this or not
        if (executing_call != NULL)
        {
          // Put this on the list of pending calls
          info->resume = UserEvent::create_user_event();
          wait_on = info->resume;
          pending_calls.push_back(info);
        }
        else
          executing_call = result;
      }
      // If we have an event to wait on, then wait until we can execute
      if (wait_on.exists())
        wait_on.wait();
      return result;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::pause_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      // We definitely know we can't start any non_reentrant calls
      // Screw fairness, we care about throughput, see if there are any
      // pending calls to wake up, and then go to sleep ourself
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock m_lock(mapper_lock);
        // Increment the count of the paused mapper calls
        paused_calls++;
        if (!pending_calls.empty())
        {
          // Get the next available call to handle
          executing_call = pending_calls.front();
          pending_calls.pop_front();
          to_trigger = executing_call->resume; 
        }
        else // No one to wake up so just go to sleep
          executing_call = NULL;
      }
      if (to_trigger.exists())
        to_trigger.trigger();
    }

    //--------------------------------------------------------------------------
    void SerializingManager::resume_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // See if we are ready to be woken up
      Event wait_on = Event::NO_EVENT;
      {
        AutoLock m_lock(mapper_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(paused_calls > 0);
#endif
        paused_calls--;
        if (executing_call != NULL)
        {
          info->resume = UserEvent::create_user_event();
          wait_on = info->resume;
          ready_calls.push_back(info);
        }
        else
          executing_call = info;
      }
      if (wait_on.exists())
        wait_on.wait();
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
    }

    //--------------------------------------------------------------------------
    void SerializingManager::finish_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock m_lock(mapper_lock);
        // See if can start a non-reentrant task
        if (!non_reentrant_calls.empty() && (paused_calls == 0) && 
            ready_calls.empty())
        {
          // Mark that we are now not permitting re-entrant
          permit_reentrant = false;
          executing_call = non_reentrant_calls.front();
          non_reentrant_calls.pop_front();
          to_trigger = executing_call->resume;
        }
        else if (!ready_calls.empty())
        {
          executing_call = ready_calls.front();
          ready_calls.pop_front();
          to_trigger = executing_call->resume;
        }
        else if (!pending_calls.empty())
        {
          executing_call = pending_calls.front();
          pending_calls.pop_front();
          to_trigger = executing_call->resume;
        }
        else
          executing_call = NULL;
        // Return our call info
        free_call_info(info, false/*need lock*/);
      }
      // Wake up the next task if necessary
      if (to_trigger.exists())
        to_trigger.trigger();
    }

    /////////////////////////////////////////////////////////////
    // Concurrent Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConcurrentManager::ConcurrentManager(Runtime *rt, Mapping::Mapper *mp)
      : MapperManager(rt, mp), lock_state(UNLOCKED_STATE), current_holders(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ConcurrentManager::ConcurrentManager(const ConcurrentManager &rhs)
      : MapperManager(NULL,NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ConcurrentManager::~ConcurrentManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ConcurrentManager& ConcurrentManager::operator=(
                                                   const ConcurrentManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ConcurrentManager::is_locked(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Can read this without holding the lock
      return (lock_state != UNLOCKED_STATE);  
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::lock_mapper(MappingCallInfo *info, bool read_only)
    //--------------------------------------------------------------------------
    {
      Event wait_on = Event::NO_EVENT;
      {
        AutoLock m_lock(mapper_lock); 
        if (current_holders.find(info) != current_holders.end())
        {
          // TODO: error message for duplicate acquire
          assert(false);
        }
        switch (lock_state)
        {
          case UNLOCKED_STATE:
            {
              // Grant the lock immediately
              current_holders.insert(info);
              if (read_only)
                lock_state = READ_ONLY_STATE;
              else
                lock_state = EXCLUSIVE_STATE;
              break;
            }
          case READ_ONLY_STATE:
            {
              if (!read_only)
              {
                info->resume = UserEvent::create_user_event();
                wait_on = info->resume;
                exclusive_waiters.push_back(info);
              }
              else // add it to the set of current holders
                current_holders.insert(info);
              break;
            }
          case EXCLUSIVE_STATE:
            {
              // Have to wait no matter what
              info->resume = UserEvent::create_user_event();
              wait_on = info->resume;
              if (read_only)
                read_only_waiters.push_back(info);
              else
                exclusive_waiters.push_back(info);
              break;
            }
          default:
            assert(false);
        }
      }
      if (wait_on.exists())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::unlock_mapper(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      std::vector<UserEvent> to_trigger;
      {
        AutoLock m_lock(mapper_lock);
        std::set<MappingCallInfo*>::iterator finder = 
          current_holders.find(info);
        if (finder == current_holders.end())
        {
          // Really bad if we can't find it in the set of current holders
          // TODO: put in an error message here
          assert(false);
        }
        current_holders.erase(finder);
        // See if we can now give the lock to someone else
        if (current_holders.empty())
        {
          switch (lock_state)
          {
            case READ_ONLY_STATE:
              {
                if (!exclusive_waiters.empty())
                {
                  // Pull off the first exlusive waiter
                  to_trigger.push_back(exclusive_waiters.front()->resume);
                  exclusive_waiters.pop_front();
                  lock_state = EXCLUSIVE_STATE;
                }
                else
                  lock_state = UNLOCKED_STATE;
                break;
              }
            case EXCLUSIVE_STATE:
              {
                if (!read_only_waiters.empty())
                {
                  to_trigger.resize(read_only_waiters.size());
                  for (unsigned idx = 0; idx < read_only_waiters.size(); idx++)
                    to_trigger[idx] = read_only_waiters[idx]->resume;
                  ready_only_waiters.clear();
                  lock_state = READ_ONLY_STATE;
                }
                else
                  lock_state = UNLOCKED_STATE;
                break;
              }
            default:
              assert(false);
          }
        }
      }
      if (!to_trigger.empty())
      {
        for (std::vector<UserEvent>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
          it->trigger();
      }
    }

    //--------------------------------------------------------------------------
    bool ConcurrentManager::is_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Always reentrant for the concurrent manager
      return true;
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::enable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do 
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::disable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* ConcurrentManager::begin_mapper_call(MappingCallKind kind)
    //--------------------------------------------------------------------------
    {
      return allocate_call_info(kind, true/*need lock*/);
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::pause_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::resume_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::finish_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      free_call_info(info, true/*need lock*/);
    }

  };
}; // namespace Legion
