/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "processor.h"
#include "id.h"

#include "activemsg.h"
#include "operation.h"
#include "profiling.h"

#include "event_impl.h"
#include "rsrv_impl.h"

#include "threads.h"
#include "pri_queue.h"

#include <greenlet>

namespace Realm {

    // information for a task launch
    class Task : public Operation {
    public:
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
	   Event _finish_event, int _priority,
           int expected_count);
      Task(Processor _proc,
	   Processor::TaskFuncID _func_id,
	   const void *_args, size_t _arglen,
           const ProfilingRequestSet &reqs,
	   Event _finish_event, int _priority,
           int expected_count);

      virtual ~Task(void);

      void execute_on_processor(Processor p);

      Processor proc;
      Processor::TaskFuncID func_id;
      void *args;
      size_t arglen;
      Event finish_event;
      int priority;
      int run_count, finish_count;
      bool capture_proc;
    };

    // TODO: get rid of this class
    template <class T>
    class Atomic {
    public:
      Atomic(T _value) : value(_value)
      {
	//printf("%d: atomic %p = %d\n", gasnet_mynode(), this, value);
      }

      T get(void) const { return (*((volatile T*)(&value))); }

      void decrement(void)
      {
	AutoHSLLock a(mutex);
	//T old_value(value);
	value--;
	//printf("%d: atomic %p %d -> %d\n", gasnet_mynode(), this, old_value, value);
      }

    protected:
      T value;
      GASNetHSL mutex;
    };

    class ProcessorImpl {
    public:
      ProcessorImpl(Processor _me, Processor::Kind _kind);

      virtual ~ProcessorImpl(void);

      void run(Atomic<int> *_run_counter)
      {
	run_counter = _run_counter;
      }

      virtual void start_processor(void) = 0;
      virtual void shutdown_processor(void) = 0;
      virtual void initialize_processor(void) = 0;
      virtual void finalize_processor(void) = 0;

      virtual void enqueue_task(Task *task) = 0;

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority) = 0;

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority) = 0;

      void finished(void)
      {
	if(run_counter)
	  run_counter->decrement();
      }

    public:
      Processor me;
      Processor::Kind kind;
      Atomic<int> *run_counter;
    }; 

    // a task scheduler in which one or more worker threads execute tasks from one
    //  or more task queues
    // once given a task, a worker must complete it before taking on new work
    // if a worker needs to suspend, a new worker may be spun up to start a new task
    // this parent version tries to be agnostic to whether the threads are
    //  user or kernel threads
    class ThreadedTaskScheduler : public ThreadScheduler {
    public:
      ThreadedTaskScheduler(Processor _proc);

      virtual ~ThreadedTaskScheduler(void);

      typedef PriorityQueue<Task *, GASNetHSL> TaskQueue;

      virtual void add_task_queue(TaskQueue *queue);

      // called when thread status changes
      virtual void thread_blocking(Thread *thread);
      virtual void thread_ready(Thread *thread);

    public:
      // the main scheduler loop - lock should be held before calling
      void scheduler_loop(void);
      // an entry point that takes the scheduler lock explicitly
      void scheduler_loop_wlock(void);

    protected:

      virtual Thread *worker_create(bool make_active) = 0;
      virtual void worker_sleep(Thread *switch_to) = 0;
      virtual void worker_wake(Thread *to_wake) = 0;
      virtual void worker_terminate(Thread *switch_to) = 0;
      virtual void idle_thread_yield(void) = 0;

      Processor proc;
      GASNetHSL lock;
      std::vector<TaskQueue *> task_queues;
      std::vector<Thread *> idle_workers;
      PriorityQueue<Thread *, DummyLock> resumable_workers;
      std::map<Thread *, int> worker_priorities;
      bool shutdown_flag;
      int active_worker_count;  // workers that are awake (i.e. using a core)
      int unassigned_worker_count;  // awake but unassigned workers

      // helper for tracking/sanity-checking worker counts
      void update_worker_count(int active_delta, int unassigned_delta, bool check = true);

    public:
      // various configurable settings
      bool cfg_reuse_workers;
      int cfg_max_idle_workers;
      int cfg_min_active_workers;
      int cfg_max_active_workers;
    };

    // an implementation of ThreadedTaskScheduler that uses kernel threads
    //  for workers
    class KernelThreadTaskScheduler : public ThreadedTaskScheduler {
    public:
      KernelThreadTaskScheduler(Processor _proc, CoreReservation& _core_rsrv);

      virtual ~KernelThreadTaskScheduler(void);

      virtual void add_task_queue(TaskQueue *queue);

      void start(void);
      void shutdown(void);

      virtual void thread_starting(Thread *thread);

      virtual void thread_terminating(Thread *thread);

    protected:
      virtual Thread *worker_create(bool make_active);
      virtual void worker_sleep(Thread *switch_to);
      virtual void worker_wake(Thread *to_wake);
      virtual void worker_terminate(Thread *switch_to);
      virtual void idle_thread_yield(void);

      CoreReservation &core_rsrv;

      std::set<Thread *> all_workers;
      std::set<Thread *> active_workers;
      std::map<Thread *, GASNetCondVar *> sleeping_threads;
      GASNetCondVar shutdown_condvar;
    };

    // an implementation of ThreadedTaskScheduler that uses user threads
    //  for workers (and one or more kernel threads for hosts
    class UserThreadTaskScheduler : public ThreadedTaskScheduler {
    public:
      UserThreadTaskScheduler(Processor _proc, CoreReservation& _core_rsrv);

      virtual ~UserThreadTaskScheduler(void);

      virtual void add_task_queue(TaskQueue *queue);

      void start(void);
      void shutdown(void);

      virtual void thread_starting(Thread *thread);

      virtual void thread_terminating(Thread *thread);

    protected:
      void host_thread(void);
      
      // you can't delete a user thread until you've switched off of it, so
      //  use TLS to mark when that should happen
      void request_user_thread_cleanup(Thread *thread);
      void do_user_thread_cleanup(void);
      
      virtual Thread *worker_create(bool make_active);
      virtual void worker_sleep(Thread *switch_to);
      virtual void worker_wake(Thread *to_wake);
      virtual void worker_terminate(Thread *switch_to);
      virtual void idle_thread_yield(void);

      CoreReservation &core_rsrv;

      std::set<Thread *> all_hosts;
      std::set<Thread *> all_workers;

    public:
      int cfg_num_host_threads;
    };

    class NewLocalProcessor : public ProcessorImpl {
    public:
      NewLocalProcessor(Processor _me, Processor::Kind _kind, 
                     size_t stack_size, const char *name,
                     int core_id = -1);
      virtual ~NewLocalProcessor(void);
    public:
      // Make these virtual so they can be modified if necessary
      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);

      virtual void enqueue_task(Task *task);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);
    protected:
      CoreReservation core_rsrv;
      UserThreadTaskScheduler *sched;
      PriorityQueue<Task *, GASNetHSL> task_queue;
    };

    class RemoteProcessor : public ProcessorImpl {
    public:
      RemoteProcessor(Processor _me, Processor::Kind _kind);
      virtual ~RemoteProcessor(void);

      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);

      virtual void enqueue_task(Task *task);

      virtual void tasks_available(int priority);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);

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

      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);

      virtual void enqueue_task(Task *task);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);

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

      virtual bool event_triggered(void);
      virtual void print_info(FILE *f);

    protected:
      ProcessorImpl *proc;
      Task *task;
    };

    // active messages

    struct SpawnTaskMessage {
      // Employ some fancy struct packing here to fit in 64 bytes
      struct RequestArgs : public BaseMedium {
	Processor proc;
	Event::id_t start_id;
	Event::id_t finish_id;
	size_t user_arglen;
	int priority;
	Processor::TaskFuncID func_id;
	Event::gen_t start_gen;
	Event::gen_t finish_gen;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<SPAWN_TASK_MSGID,
 	                                 RequestArgs,
 	                                 handle_request> Message;

      static void send_request(gasnet_node_t target, Processor proc,
			       Processor::TaskFuncID func_id,
			       const void *args, size_t arglen,
			       const ProfilingRequestSet *prs,
			       Event start_event, Event finish_event,
			       int priority);
    };


    ///////////////////////////////////////////////////////////////////////////
    //
    // SPAGHETTI CODE BELOW THIS POINT
    
    class ProcessorAssignment {
    public:
      ProcessorAssignment(int _num_local_procs);

      // binds a thread to the right set of cores based (-1 = not a local proc)
      void bind_thread(int core_id, pthread_attr_t *attr, const char *debug_name = 0);

    protected:
      // physical configuration of processors
      typedef std::map<int, std::vector<int> > NodeProcMap;
      typedef std::map<int, NodeProcMap> SystemProcMap;

      int num_local_procs;
      bool valid;
      std::vector<int> local_proc_assignments;
#ifndef __MACH__
      cpu_set_t leftover_procs;
#endif
    };
    extern ProcessorAssignment *proc_assignment;

    extern Processor::TaskIDTable task_id_table;

    // generic way of keeping a prioritized queue of stuff to do
    // Needs to be protected by owner lock
    template <typename JOBTYPE>
    class JobQueue {
    public:
      JobQueue(void);

      bool empty(void) const;

      void insert(JOBTYPE *job, int priority);

      JOBTYPE *pop(void);

      struct WaitingJob : public EventWaiter {
	JOBTYPE *job;
	int priority;
	JobQueue *queue;

	virtual bool event_triggered(void);
	virtual void print_info(FILE *f);
      };

      std::map<int, std::deque<JOBTYPE*> > ready;
    };

    template <typename JOBTYPE>
    JobQueue<JOBTYPE>::JobQueue(void)
    {
    }

    template<typename JOBTYPES>
    bool JobQueue<JOBTYPES>::empty(void) const
    {
      return ready.empty();
    }

    template <typename JOBTYPE>
    void JobQueue<JOBTYPE>::insert(JOBTYPE *job, int priority)
    {
      std::deque<JOBTYPE *>& dq = ready[-priority];
      dq.push_back(job);
    }

    template <typename JOBTYPE>
    JOBTYPE *JobQueue<JOBTYPE>::pop(void)
    {
      if(ready.empty()) return 0;

      // get the sublist with the highest priority (remember, we negate before lookup)
      typename std::map<int, std::deque<JOBTYPE *> >::iterator it = ready.begin();

      // any deque that's present better be non-empty
      assert(!(it->second.empty()));
      JOBTYPE *job = it->second.front();
      it->second.pop_front();

      // if the list is now empty, remove it and update the new max priority
      if(it->second.empty()) {
	ready.erase(it);
      }

      return job;
    }

    class PreemptableThread {
    public:
      PreemptableThread(void) 
      {
#ifdef EVENT_GRAPH_TRACE
        enclosing_stack.push_back(Event::NO_EVENT);
#endif
      }
      virtual ~PreemptableThread(void) {}

      void start_thread(size_t stack_size, int core_id, const char *debug_name);

      static bool preemptable_sleep(Event wait_for);

      virtual Processor get_processor(void) const = 0;

      void run_task(Task *task, Processor actual_proc = Processor::NO_PROC);

#ifdef EVENT_GRAPH_TRACE
      inline Event find_enclosing(void) 
      { assert(!enclosing_stack.empty()); return enclosing_stack.back(); }
      inline void start_enclosing(const Event &term_event)
      { enclosing_stack.push_back(term_event); }
      inline void finish_enclosing(void)
      { assert(enclosing_stack.size() > 1); enclosing_stack.pop_back(); }
#endif

    protected:
      static void *thread_entry(void *data);

      virtual void thread_main(void) = 0;

      virtual void sleep_on_event(Event wait_for) = 0;

      pthread_t thread;

#ifdef EVENT_GRAPH_TRACE
      std::deque<Event> enclosing_stack; 
#endif
    };

    // Forward declaration
    class LocalProcessor;

    class LocalThread : public PreemptableThread, EventWaiter {
    public:
      enum ThreadState {
        RUNNING_STATE,
        PAUSED_STATE,
        RESUMABLE_STATE,
        SLEEPING_STATE, // about to sleep
        SLEEP_STATE,
      };
    public:
      LocalThread(LocalProcessor *proc);
      virtual ~LocalThread(void);
    public:
      inline void do_initialize(void) { initialize = true; }
      inline void do_finalize(void) { finalize = true; }
    public:
      virtual Processor get_processor(void) const;
    protected:
      virtual void thread_main(void);
      virtual void sleep_on_event(Event wait_for);
      virtual bool event_triggered(void);
      virtual void print_info(FILE *f);
    public:
      void awake(void);
      void sleep(void);
      void prepare_to_sleep(void);
      void resume(void);
      void shutdown(void);
    public:
      LocalProcessor *const proc;
    protected:
      ThreadState state;
      GASNetHSL thread_mutex;
      GASNetCondVar thread_cond;
      bool initialize;
      bool finalize;
    };

    class LocalProcessor : public ProcessorImpl {
    public:
      LocalProcessor(Processor _me, Processor::Kind _kind, 
                     size_t stack_size, const char *name,
                     int core_id = -1);
      virtual ~LocalProcessor(void);
    public:
      // Make these virtual so they can be modified if necessary
      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);
      virtual LocalThread* create_new_thread(void);
    public:
      bool execute_task(LocalThread *thread);
      void pause_thread(LocalThread *thread);
      void resume_thread(LocalThread *thread);
    public:
      virtual void enqueue_task(Task *task);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstance> instances_needed,
			      Event start_event, Event finish_event,
                              int priority);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);
    protected:
      const int core_id;
      const size_t stack_size;
      const char *const processor_name;
      GASNetHSL mutex;
      GASNetCondVar condvar;
      bool done_initialization;
      JobQueue<Task> task_queue;
      bool shutdown, shutdown_trigger;
    protected:
      LocalThread               *running_thread;
      std::set<LocalThread*>    paused_threads;
      std::deque<LocalThread*>  resumable_threads;
      std::vector<LocalThread*> available_threads;
    };

    // Forward declarations
    class GreenletThread;
    class GreenletProcessor;

    class GreenletTask : public greenlet, public EventWaiter {
    public:
      GreenletTask(Task *task, GreenletProcessor *proc,
                   void *stack, long *stack_size);
      virtual ~GreenletTask(void);
    public:
      virtual bool event_triggered(void);
      virtual void print_info(FILE *f);
    public:
      virtual void* run(void *arg);
    protected:
      Task *const task;
      GreenletProcessor *const proc;
    };

    class GreenletThread : public PreemptableThread {
    public:
      GreenletThread(GreenletProcessor *proc);
      virtual ~GreenletThread(void);
    public:
      virtual Processor get_processor(void) const;
    public:
      virtual void thread_main(void);
      virtual void sleep_on_event(Event wait_for);
    public:
      void start_task(GreenletTask *task);
      void resume_task(GreenletTask *task);
      void return_to_root(void);
      void wait_for_shutdown(void);
    public:
      GreenletProcessor *const proc;
    protected:
      GreenletTask *current_task;
    };

    class GreenletProcessor : public ProcessorImpl {
    public:
      enum GreenletState {
        GREENLET_IDLE,
        GREENLET_RUNNING,
      };
      struct GreenletStack {
      public:
        void *stack;
        long stack_size;
      };
    public:
      GreenletProcessor(Processor _me, Processor::Kind _kind,
                        size_t stack_size, int init_stack_count,
                        const char *name, int core_id = -1);
      virtual ~GreenletProcessor(void);
    public:
      virtual void start_processor(void);
      virtual void shutdown_processor(void);
      virtual void initialize_processor(void);
      virtual void finalize_processor(void);
    public:
      virtual void enqueue_task(Task *task);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event,
                              int priority);
      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
                              const ProfilingRequestSet &reqs,
			      Event start_event, Event finish_event,
                              int priority);
    public:
      bool execute_task(void);
      void pause_task(GreenletTask *paused_task);
      void unpause_task(GreenletTask *paused_task);
    public:
      bool allocate_stack(GreenletStack &stack);
      void create_stack(GreenletStack &stack);
      void complete_greenlet(GreenletTask *greenlet); 
    public:
      const int core_id;
      const size_t proc_stack_size;
      const char *const processor_name;
    protected:
      GASNetHSL mutex;
      GASNetCondVar condvar;
      bool done_initialization;
      JobQueue<Task> task_queue;
      bool shutdown, shutdown_trigger;
    protected:
      GreenletThread             *greenlet_thread;
      GreenletState              thread_state;
      std::set<GreenletTask*>    paused_tasks; 
      std::list<GreenletTask*>   resumable_tasks;
      std::vector<GreenletStack> greenlet_stacks;
      std::vector<GreenletTask*> complete_greenlets;
    };

    
}; // namespace Realm

#endif // ifndef REALM_PROC_IMPL_H
