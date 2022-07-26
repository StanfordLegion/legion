/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#ifndef REALM_PYTHON_INTERNAL_H
#define REALM_PYTHON_INTERNAL_H

#include "realm/proc_impl.h"

#include "realm/python/python_source.h"

#ifdef REALM_USE_OPENMP
#include "realm/openmp/openmp_threadpool.h"
#endif

namespace Realm {

#define USE_PYGILSTATE_CALLS

  // these are all defined in Python.h, which we currently do not include
  struct PyObject;
  struct PyInterpreterState;
  struct PyThreadState {
#if REALM_PYTHON_VERSION_MAJOR >= 3
    struct PyThreadState *prev;
#endif
    struct PyThreadState *next;
    struct PyInterpreterState *interp;
    // lots more stuff here
  };
  typedef ssize_t Py_ssize_t;
#ifdef USE_PYGILSTATE_CALLS
  enum PyGILState_STATE {PyGILState_LOCKED, PyGILState_UNLOCKED};
#endif

  // This class contains interpreter-specific instances of Python API calls.
  class PythonAPI {
  public:
    PythonAPI(void *handle);

  protected:
    template<typename T>
    void get_symbol(T &fn, const char *symbol, bool missing_ok = false);

  protected:
    void *handle;

  public:
    // Python API calls
    void (*Py_DecRef)(PyObject *); // non-macro version of PyDECREF
    void (*Py_Finalize)(void);
    void (*Py_InitializeEx)(int);

    PyObject *(*PyByteArray_FromStringAndSize)(const char *, Py_ssize_t);

    void (*PyEval_InitThreads)(void);

#ifdef USE_PYGILSTATE_CALLS
    PyGILState_STATE (*PyGILState_Ensure)(void);
    void (*PyGILState_Release)(PyGILState_STATE);
#else
    PyThreadState *(*PyThreadState_New)(PyInterpreterState *);
    void (*PyThreadState_Clear)(PyThreadState *);
    void (*PyThreadState_Delete)(PyThreadState *);
#endif
    void (*PyEval_RestoreThread)(PyThreadState *);
    PyThreadState *(*PyEval_SaveThread)(void);

    PyThreadState *(*PyThreadState_Swap)(PyThreadState *);
    PyThreadState *(*PyThreadState_Get)(void);
    int (*PyGILState_Check)(void);

    void (*PyErr_PrintEx)(int set_sys_last_vars);

    PyObject *(*PyImport_ImportModule)(const char *);

    PyObject *(*PyModule_GetDict)(PyObject *);

    PyObject *(*PyLong_FromUnsignedLong)(unsigned long);

    PyObject *(*PyObject_CallFunction)(PyObject *, const char *, ...);
    PyObject* (*PyObject_CallObject)(PyObject *callable, PyObject *args);
    PyObject *(*PyObject_GetAttrString)(PyObject *, const char *);
    int (*PyObject_Print)(PyObject *, FILE *, int);

    void (*PyRun_SimpleString)(const char *);
    PyObject *(*PyRun_String)(const char *, int, PyObject *, PyObject *);

    PyObject *(*PyTuple_New)(Py_ssize_t len);
    int (*PyTuple_SetItem)(PyObject *p, Py_ssize_t pos, PyObject *o);
  };

  class PythonInterpreter {
  public:
    PythonInterpreter();
    ~PythonInterpreter();

    PyObject *find_or_import_function(const PythonSourceImplementation *psi);

    void import_module(const std::string& module_name);
    void run_string(const std::string& script_text);

  protected:
    void *handle;
#ifdef REALM_USE_DLMOPEN
    void *dlmproxy_handle;
#endif
    
  public:
    PythonAPI *api;
  };

  class PythonThreadTaskScheduler;
  
  // this is nearly identical to a LocalCPUProcessor, but it asks for its thread(s)
  //  to run on the specified numa domain
  class LocalPythonProcessor : public ProcessorImpl {
  public:
    LocalPythonProcessor(Processor _me, int _numa_node,
                         CoreReservationSet& crs, size_t _stack_size,
#ifdef REALM_USE_OPENMP
			 int _omp_workers,
#endif
			 const std::vector<std::string>& _import_modules,
			 const std::vector<std::string>& _init_scripts);
    virtual ~LocalPythonProcessor(void);

    virtual void enqueue_task(Task *task);
    virtual void enqueue_tasks(Task::TaskList& tasks, size_t num_tasks);

    virtual void spawn_task(Processor::TaskFuncID func_id,
			    const void *args, size_t arglen,
			    const ProfilingRequestSet &reqs,
			    Event start_event,
			    GenEventImpl *finish_event,
			    EventImpl::gen_t finish_gen,
			    int priority);
    
    virtual void execute_task(Processor::TaskFuncID func_id,
			      const ByteArrayRef& task_args);

    // starts worker threads and performs any per-processor initialization
    virtual void start_threads(void);

    virtual void shutdown(void);

    virtual void add_to_group(ProcessorGroupImpl *group);
    
    virtual void remove_from_group(ProcessorGroupImpl *group);
    
    virtual bool register_task(Processor::TaskFuncID func_id,
                               CodeDescriptor& codedesc,
                               const ByteArrayRef& user_data);

    //void worker_main(void);

    class TaskRegistration : public InternalTask {
    public:
      virtual ~TaskRegistration() {}

      virtual void execute_on_processor(Processor p)
      {
	proc->perform_task_registration(this);
      }
      
      LocalPythonProcessor *proc;
      Processor::TaskFuncID func_id;
      CodeDescriptor *codedesc;
      ByteArray user_data;
    };


  protected:
    friend class PythonThreadTaskScheduler;
    void create_interpreter(void);
    void destroy_interpreter(void);
    bool perform_task_registration(TaskRegistration *treg);

    int numa_node;
    CoreReservation *core_rsrv;
#ifdef REALM_USE_OPENMP
    ThreadPool *omp_threadpool;
#endif
    const std::vector<std::string>& import_modules;
    const std::vector<std::string>& init_scripts;

    PythonThreadTaskScheduler *sched;
    PythonInterpreter *interpreter;
    PyThreadState *master_thread;

    struct TaskTableEntry {
      PyObject *python_fnptr;
      Processor::TaskFuncPtr cpp_fnptr;
      ByteArray user_data;
    };

    std::map<Processor::TaskFuncID, TaskTableEntry> task_table;

    TaskQueue task_queue; // ready tasks
    ProfilingGauges::AbsoluteRangeGauge<int> ready_task_count;
    DeferredSpawnCache deferred_spawn_cache;
  };

  // based on KernelThreadTaskScheduler, deals with the python GIL and thread
  //  state changes as well

  class PythonThreadTaskScheduler : public KernelThreadTaskScheduler {
  public:
    PythonThreadTaskScheduler(LocalPythonProcessor *_pyproc,
			      CoreReservation& _core_rsrv);

    // entry point for python worker threads - falls through to scheduler_loop
    void python_scheduler_loop(void);

    // called by a worker thread when it needs to wait for something (and we
    //   should release the GIL)
    virtual void thread_blocking(Thread *thread);

    virtual void thread_ready(Thread *thread);

  protected:
    // both real and internal tasks need to be wrapped with acquires of the GIL
    virtual bool execute_task(Task *task);
    virtual void execute_internal_task(InternalTask *task);
    
    virtual Thread *worker_create(bool make_active);
    virtual void worker_terminate(Thread *switch_to);

    LocalPythonProcessor *pyproc;
    bool interpreter_ready;
    std::map<Thread *, PyThreadState *> pythreads;
  };

}; // namespace Realm

#endif // defined REALM_PYTHON_INTERNAL_H
