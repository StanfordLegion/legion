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

#ifndef REALM_PYTHON_INTERNAL_H
#define REALM_PYTHON_INTERNAL_H

#include "realm/proc_impl.h"

#include "realm/python/python_source.h"

namespace Realm {

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
    void (*PyEval_RestoreThread)(PyThreadState *);
    PyThreadState *(*PyEval_SaveThread)(void);

    PyThreadState *(*PyThreadState_New)(PyInterpreterState *);
    void (*PyThreadState_Clear)(PyThreadState *);
    void (*PyThreadState_Delete)(PyThreadState *);
    PyThreadState *(*PyThreadState_Get)(void);
    PyThreadState *(*PyThreadState_Swap)(PyThreadState *);

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
			 const std::vector<std::string>& _import_modules,
			 const std::vector<std::string>& _init_scripts);
    virtual ~LocalPythonProcessor(void);

    virtual void enqueue_task(Task *task);

    virtual void spawn_task(Processor::TaskFuncID func_id,
			    const void *args, size_t arglen,
			    const ProfilingRequestSet &reqs,
			    Event start_event, Event finish_event,
			    int priority);
    
    virtual void execute_task(Processor::TaskFuncID func_id,
			      const ByteArrayRef& task_args);

    // starts worker threads and performs any per-processor initialization
    virtual void start_threads(void);

    virtual void shutdown(void);

    virtual void add_to_group(ProcessorGroup *group);
    
    virtual void register_task(Processor::TaskFuncID func_id,
                               CodeDescriptor& codedesc,
                               const ByteArrayRef& user_data);

    //void worker_main(void);

    struct TaskRegistration {
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
    const std::vector<std::string>& import_modules;
    const std::vector<std::string>& init_scripts;

    PythonThreadTaskScheduler *sched;
    PythonInterpreter *interpreter;
    PyThreadState *master_thread;

    struct TaskTableEntry {
      PyObject *fnptr;
      ByteArray user_data;
    };

    std::map<Processor::TaskFuncID, TaskTableEntry> task_table;

    PriorityQueue<Task *, GASNetHSL> task_queue;
    ProfilingGauges::AbsoluteRangeGauge<int> ready_task_count;
  };

  // based on KernelThreadTaskScheduler, deals with the python GIL and thread
  //  state changes as well

  class PythonThreadTaskScheduler : public KernelThreadTaskScheduler {
  public:
    PythonThreadTaskScheduler(LocalPythonProcessor *_pyproc,
			      CoreReservation& _core_rsrv);

    void enqueue_taskreg(LocalPythonProcessor::TaskRegistration *treg);

    // entry point for python worker threads - falls through to scheduler_loop
    void python_scheduler_loop(void);

    // called by a worker thread when it needs to wait for something (and we
    //   should release the GIL)
    virtual void thread_blocking(Thread *thread);

    virtual void thread_ready(Thread *thread);

  protected:
    virtual Thread *worker_create(bool make_active);
    virtual void worker_terminate(Thread *switch_to);

    LocalPythonProcessor *pyproc;
    bool interpreter_ready;
    std::list<LocalPythonProcessor::TaskRegistration *> taskreg_queue;
    std::map<Thread *, PyThreadState *> pythreads;
  };

}; // namespace Realm

#endif // defined REALM_PYTHON_INTERNAL_H
