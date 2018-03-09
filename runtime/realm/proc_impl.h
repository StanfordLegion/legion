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

// Processor/ProcessorGroup implementations for Realm

#ifndef REALM_PROC_IMPL_H
#define REALM_PROC_IMPL_H

#include "realm/processor.h"
#include "realm/id.h"

#include "realm/activemsg.h"
#include "realm/operation.h"
#include "realm/profiling.h"
#include "realm/sampling.h"

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"

#include "realm/tasks.h"
#include "realm/threads.h"
#include "realm/codedesc.h"

namespace Realm {

    class ProcessorGroup;

    class ProcessorImpl {
    public:
      ProcessorImpl(Processor _me, Processor::Kind _kind, int _num_cores=1);

      virtual ~ProcessorImpl(void);

      virtual void enqueue_task(Task *task) = 0;

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority) = 0;

      // starts worker threads and performs any per-processor initialization
      virtual void start_threads(void);

      // blocks until things are cleaned up
      virtual void shutdown(void);

      virtual void add_to_group(ProcessorGroup *group) = 0;

      virtual void register_task(Processor::TaskFuncID func_id,
				 CodeDescriptor& codedesc,
				 const ByteArrayRef& user_data);

    protected:
      friend class Task;

      virtual void execute_task(Processor::TaskFuncID func_id,
				const ByteArrayRef& task_args);

    public:
      Processor me;
      Processor::Kind kind;
      int num_cores;
    }; 

    // generic local task processor - subclasses must create and configure a task
    // scheduler and pass in with the set_scheduler() method
    class LocalTaskProcessor : public ProcessorImpl {
    public:
      LocalTaskProcessor(Processor _me, Processor::Kind _kind, int num_cores=1);
      virtual ~LocalTaskProcessor(void);

      virtual void enqueue_task(Task *task);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);

      virtual void register_task(Processor::TaskFuncID func_id,
				 CodeDescriptor& codedesc,
				 const ByteArrayRef& user_data);

      // starts worker threads and performs any per-processor initialization
      virtual void start_threads(void);

      // blocks until things are cleaned up
      virtual void shutdown(void);

      virtual void add_to_group(ProcessorGroup *group);

    protected:
      void set_scheduler(ThreadedTaskScheduler *_sched);

      ThreadedTaskScheduler *sched;
      PriorityQueue<Task *, GASNetHSL> task_queue;
      ProfilingGauges::AbsoluteRangeGauge<int> ready_task_count;

      struct TaskTableEntry {
	Processor::TaskFuncPtr fnptr;
	ByteArray user_data;
      };

      std::map<Processor::TaskFuncID, TaskTableEntry> task_table;

      virtual void execute_task(Processor::TaskFuncID func_id,
				const ByteArrayRef& task_args);
    };

    // three simple subclasses for:
    // a) "CPU" processors, which request a dedicated core and use user threads
    //      when possible
    // b) "utility" processors, which also use user threads but share cores with
    //      other runtime threads
    // c) "IO" processors, which use kernel threads so that blocking IO calls
    //      are permitted
    //
    // each of these is implemented just by supplying the right kind of scheduler to
    //  LocalTaskProcessor in the constructor

    class LocalCPUProcessor : public LocalTaskProcessor {
    public:
      LocalCPUProcessor(Processor _me, CoreReservationSet& crs,
			size_t _stack_size, bool _force_kthreads);
      virtual ~LocalCPUProcessor(void);
    protected:
      CoreReservation *core_rsrv;
    };

    class LocalUtilityProcessor : public LocalTaskProcessor {
    public:
      LocalUtilityProcessor(Processor _me, CoreReservationSet& crs,
			    size_t _stack_size, bool _force_kthreads);
      virtual ~LocalUtilityProcessor(void);
    protected:
      CoreReservation *core_rsrv;
    };

    class LocalIOProcessor : public LocalTaskProcessor {
    public:
      LocalIOProcessor(Processor _me, CoreReservationSet& crs, size_t _stack_size,
		       int _concurrent_io_threads);
      virtual ~LocalIOProcessor(void);
    protected:
      CoreReservation *core_rsrv;
    };

    class RemoteProcessor : public ProcessorImpl {
    public:
      RemoteProcessor(Processor _me, Processor::Kind _kind, int _num_cores=1);
      virtual ~RemoteProcessor(void);

      virtual void enqueue_task(Task *task);

      virtual void add_to_group(ProcessorGroup *group);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);
    };

    class ProcessorGroup : public ProcessorImpl {
    public:
      ProcessorGroup(void);

      virtual ~ProcessorGroup(void);

      static const ID::ID_Types ID_TYPE = ID::ID_PROCGROUP;

      void init(Processor _me, int _owner);

      void set_group_members(const std::vector<Processor>& member_list);

      void get_group_members(std::vector<Processor>& member_list);

      virtual void enqueue_task(Task *task);

      virtual void add_to_group(ProcessorGroup *group);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);

    public: //protected:
      bool members_valid;
      bool members_requested;
      std::vector<ProcessorImpl *> members;
      ReservationImpl lock;
      ProcessorGroup *next_free;

      void request_group_members(void);

      PriorityQueue<Task *, GASNetHSL> task_queue;
      ProfilingGauges::AbsoluteRangeGauge<int> *ready_task_count;
    };
    
    // this is generally useful to all processor implementations, so put it here
    class DeferredTaskSpawn : public EventWaiter {
    public:
      DeferredTaskSpawn(ProcessorImpl *_proc, Task *_task) 
        : proc(_proc), task(_task) {}

      virtual ~DeferredTaskSpawn(void)
      {
        // we do _NOT_ own the task - do not free it
      }

      virtual bool event_triggered(Event e, bool poisoned);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      ProcessorImpl *proc;
      Task *task;
    };

    // a task registration can take a while if remote processors and/or JITs are
    //  involved
    class TaskRegistration : public Operation {
    public:
      TaskRegistration(const CodeDescriptor& _codedesc,
		       const ByteArrayRef& _userdata,
		       Event _finish_event, const ProfilingRequestSet &_requests);

    protected:
      // deletion performed when reference count goes to zero
      virtual ~TaskRegistration(void);

    public:
      virtual void print(std::ostream& os) const;

      CodeDescriptor codedesc;
      ByteArray userdata;
    };

    class RemoteTaskRegistration : public Operation::AsyncWorkItem {
    public:
      RemoteTaskRegistration(TaskRegistration *reg_op, int _target_node);

      virtual void request_cancellation(void);

      virtual void print(std::ostream& os) const;

    protected:
      int target_node;
    };

    // active messages

    struct SpawnTaskMessage {
      // Employ some fancy struct packing here to fit in 64 bytes
      struct RequestArgs : public BaseMedium {
	Processor proc;
	Event start_event;
	Event finish_event;
	size_t user_arglen;
	int priority;
	Processor::TaskFuncID func_id;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<SPAWN_TASK_MSGID,
 	                                 RequestArgs,
 	                                 handle_request> Message;

      static void send_request(NodeID target, Processor proc,
			       Processor::TaskFuncID func_id,
			       const void *args, size_t arglen,
			       const ProfilingRequestSet *prs,
			       Event start_event, Event finish_event,
			       int priority);
    };
    
    struct RegisterTaskMessage {
      struct RequestArgs : public BaseMedium {
	NodeID sender;
	Processor::TaskFuncID func_id;
	Processor::Kind kind;
	RemoteTaskRegistration *reg_op;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<REGISTER_TASK_MSGID,
 	                                 RequestArgs,
 	                                 handle_request> Message;

      static void send_request(NodeID target,
			       Processor::TaskFuncID func_id,
			       Processor::Kind kind,
			       const std::vector<Processor>& procs,
			       const CodeDescriptor& codedesc,
			       const void *userdata, size_t userlen,
			       RemoteTaskRegistration *reg_op);
    };
    
    struct RegisterTaskCompleteMessage {
      struct RequestArgs {
	NodeID sender;
	RemoteTaskRegistration *reg_op;
	bool successful;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<REGISTER_TASK_COMPLETE_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(NodeID target,
			       RemoteTaskRegistration *reg_op,
			       bool successful);
    };

}; // namespace Realm

#endif // ifndef REALM_PROC_IMPL_H
