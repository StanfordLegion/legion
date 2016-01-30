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

#ifndef __MAPPER_MANAGER_H__
#define __MAPPER_MANAGER_H__

#include "legion_types.h"

namespace Legion {
  namespace Internal {

    enum MappingCallKind {

    };

    struct MappingCallInfo {
    public:
      MappingCallInfo(MapperManager *man, MappingCallKind k)
        : manager(man), resume(UserEvent::NO_USER_EVENT), kind(k) { }
    public:
      MapperManager *const manager;
      UserEvent            resume;
      MappingCallKind      kind;
    };

    /**
     * \class MapperManager
     * This is the base class for a bunch different kinds of mapper
     * managers. Some calls into this manager from the mapper will
     * be handled right away, while other we may need to defer and
     * possibly preempt.  This later class of calls are the ones that
     * are made virtual so that the 
     */
    class MapperManager {
    public:
      MapperManager(Runtime *runtime, Mapping::Mapper *mapper);
      virtual ~MapperManager(void);
    public:
      virtual bool is_locked(MappingCallInfo *info) = 0;
      virtual void lock_mapper(MappingCallInfo *info, bool read_only) = 0;
      virtual void unlock_mapper(MappingCallInfo *info) = 0;
    public:
      virtual bool is_reentrant(MappingCallInfo *info) = 0;
      virtual void enable_reentrant(MappingCallInfo *info) = 0;
      virtual void disable_reentrant(MappingCallInfo *info) = 0;
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind) = 0;
      virtual void pause_mapper_call(MappingCallInfo *info) = 0;
      virtual void resume_mapper_call(MappingCallInfo *info) = 0;
      virtual void finish_mapper_call(MappingCallInfo *info) = 0;
    public:
      void send_message(MappingCallInfo *info, Processor target, 
                        const void *message, size_t message_size);
      void broadcast(MappingCallInfo *info, const void *message, 
                     size_t message_size, int radix);
    public:
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      IndexSpace get_index_subspace(IndexPartition p, Color c);
      IndexSpace get_index_subspace(IndexPartition p, 
                                    const DomainPoint &color);
      bool has_multiple_domains(IndexSpace handle);
      Domain get_index_space_domain(IndexSpace handle);
      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_space_partition_colors(IndexSpace sp, 
                                            std::set<Color> &colors);
      bool is_index_partition_disjoint(IndexPartition p);
      template<unsigned DIM>
      IndexSpace get_index_subspace(IndexPartition p, 
                                    ColorPoint &color_point);
      Color get_index_space_color(IndexSpace handle);
      Color get_index_partition_color(IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
    public:
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(FieldSpace handle, 
                                  std::set<FieldID> &fields);
    protected:
      // Both these must be called while holding the lock
      MappingCallInfo* allocate_call_info(MappingCallKind kind, bool need_lock);
      void free_call_info(MappingCallInfo *info, bool need_lock);
    protected:
      Runtime *const runtime;
      Mapping::Mapper *const mapper;
    protected:
      const Reservation mapper_lock;
    protected:
      std::vector<MappingCallInfo*> available_infos;
    };

    /**
     * \class SerializingManager
     * In this class at most one mapper call can be running at 
     * a time. Mapper calls that invoke expensive runtime operations
     * can be pre-empted and it is up to the mapper to control
     * whether additional mapper calls when the call is blocked.
     */
    class SerializingManager : public MapperManager {
    public:
      SerializingManager(Runtime *runtime, Mapping::Mapper *mapper,
                         bool reentrant);
      SerializingManager(const SerializingManager &rhs);
      virtual ~SerializingManager(void);
    public:
      SerializingManager& operator=(const SerializingManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info);
      virtual void finish_mapper_call(MappingCallInfo *info);
    protected:
      bool permit_reentrant;
    protected:
      // The one executing call if any otherwise NULL
      MappingCallInfo *executing_call; 
      // Calls yet to start running
      std::deque<MappingCallInfo*> pending_calls; 
      // Number of calls paused due to runtime work
      unsigned paused_calls;
      // Calls that are ready to resume after runtime work
      std::deque<MappingCallInfo*> ready_calls;
      // Calls that are waiting for diabling of reentrancy
      std::deque<MappingCallInfo*> non_reentrant_calls;
    };

    /**
     * \class ConcurrentManager
     * In this class many mapper calls can be running concurrently.
     * It is upper to the mapper to lock itself when necessary to 
     * protect internal state. Mappers can be locked in exclusive
     * or non-exclusive modes.
     */
    class ConcurrentManager : public MapperManager {
    public:
      enum LockState {
        UNLOCKED_STATE,
        READ_ONLY_STATE,
        EXCLUSIVE_STATE,
      };
    public:
      ConcurrentManager(Runtime *runtime, Mapping::Mapper *mapper);
      ConcurrentManager(const ConcurrentManager &rhs);
      virtual ~ConcurrentManager(void);
    public:
      ConcurrentManager& operator=(const ConcurrentManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info);
      virtual void finish_mapper_call(MappingCallInfo *info);
    protected:
      LockState lock_state;
      std::set<MappingCallInfo*> current_holders;
      std::deque<MappingCallInfo*> read_only_waiters;
      std::deque<MappingCallInfo*> exclusive_waiters;
    };

  };
};

#endif // __MAPPER_MANAGER_H__
