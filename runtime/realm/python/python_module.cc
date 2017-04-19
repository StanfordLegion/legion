/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include "python_module.h"

#include "python_source.h"

#include "../numa/numasysif.h"
#include "logging.h"
#include "cmdline.h"
#include "proc_impl.h"
#include "threads.h"
#include "runtime_impl.h"
#include "utils.h"

#include <dlfcn.h>

#include <list>

struct PyObject;
struct PyThreadState;
typedef ssize_t Py_ssize_t;

namespace Realm {

  Logger log_py("python");

  ////////////////////////////////////////////////////////////////////////
  //
  // class PythonAPI

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
    void (*Py_Initialize)(void);

    PyObject *(*PyByteArray_FromStringAndSize)(const char *, Py_ssize_t);

    void (*PyEval_AcquireLock)(void);
    void (*PyEval_InitThreads)(void);
    void (*PyEval_RestoreThread)(PyThreadState *);
    PyThreadState *(*PyEval_SaveThread)(void);

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

  PythonAPI::PythonAPI(void *_handle)
    : handle(_handle)
  {
    get_symbol(this->Py_DecRef, "Py_DecRef");
    get_symbol(this->Py_Finalize, "Py_Finalize");
    get_symbol(this->Py_Initialize, "Py_Initialize");

    get_symbol(this->PyByteArray_FromStringAndSize, "PyByteArray_FromStringAndSize");

    get_symbol(this->PyEval_AcquireLock, "PyEval_AcquireLock");
    get_symbol(this->PyEval_InitThreads, "PyEval_InitThreads");
    get_symbol(this->PyEval_RestoreThread, "PyEval_RestoreThread");
    get_symbol(this->PyEval_SaveThread, "PyEval_SaveThread");

    get_symbol(this->PyErr_PrintEx, "PyErr_PrintEx");

    get_symbol(this->PyImport_ImportModule, "PyImport_ImportModule");
    get_symbol(this->PyModule_GetDict, "PyModule_GetDict");

    get_symbol(this->PyLong_FromUnsignedLong, "PyLong_FromUnsignedLong");

    get_symbol(this->PyObject_CallFunction, "PyObject_CallFunction");
    get_symbol(this->PyObject_CallObject, "PyObject_CallObject");
    get_symbol(this->PyObject_GetAttrString, "PyObject_GetAttrString");
    get_symbol(this->PyObject_Print, "PyObject_Print");

    get_symbol(this->PyRun_SimpleString, "PyRun_SimpleString");
    get_symbol(this->PyRun_String, "PyRun_String");

    get_symbol(this->PyTuple_New, "PyTuple_New");
    get_symbol(this->PyTuple_SetItem, "PyTuple_SetItem");
  }

  template<typename T>
  void PythonAPI::get_symbol(T &fn, const char *symbol,
                             bool missing_ok /*= false*/)
  {
    fn = reinterpret_cast<T>(dlsym(handle, symbol));
    if(!fn && !missing_ok) {
      const char *error = dlerror();
      log_py.fatal() << "failed to find symbol '" << symbol << "': " << error;
      assert(false);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class PythonInterpreter

  class PythonInterpreter {
  public:
    PythonInterpreter();
    ~PythonInterpreter();

    PyObject *find_or_import_function(const PythonSourceImplementation *psi);

    void import_module(const std::string& module_name);
    void run_string(const std::string& script_text);

  protected:
    void *handle;

  public:
    PythonAPI *api;
  };

  PythonInterpreter::PythonInterpreter() 
  {
#ifdef REALM_USE_DLMOPEN
    handle = dlmopen(LM_ID_NEWLM, "libpython2.7.so", RTLD_DEEPBIND | RTLD_LOCAL | RTLD_LAZY);
#else
    handle = dlopen("libpython2.7.so", RTLD_GLOBAL | RTLD_LAZY);
#endif
    if (!handle) {
      const char *error = dlerror();
      log_py.fatal() << error;
      assert(false);
    }

    api = new PythonAPI(handle);

    (api->Py_Initialize)();
    //(api->PyEval_InitThreads)();
    //(api->Py_Finalize)();

    //PyThreadState *state;
    //state = (api->PyEval_SaveThread)();
    //(api->PyEval_RestoreThread)(state);

    //(api->PyRun_SimpleString)("print 'hello Python world!'");

    //PythonSourceImplementation psi("taskreg_helper", "task1");
    //find_or_import_function(&psi);
  }

  PythonInterpreter::~PythonInterpreter()
  {
    (api->Py_Finalize)();

    delete api;

#if 0
    if (dlclose(handle)) {
      const char *error = dlerror();
      log_py.fatal() << error;
      assert(false);
    }
#endif
  }

  PyObject *PythonInterpreter::find_or_import_function(const PythonSourceImplementation *psi)
  {
    //log_py.print() << "attempting to acquire python lock";
    //(api->PyEval_AcquireLock)();
    //log_py.print() << "lock acquired";

    // not calling PythonInterpreter::import_module here because we want the
    //  PyObject result
    log_py.debug() << "attempting to import module: " << psi->module_name;
    PyObject *module = (api->PyImport_ImportModule)(psi->module_name.c_str());
    if (!module) {
      log_py.fatal() << "unable to import Python module " << psi->module_name;
      (api->PyErr_PrintEx)(0);
      assert(0);
    }
    //(api->PyObject_Print)(module, stdout, 0); printf("\n");

    log_py.debug() << "finding attribute '" << psi->function_name << "' in module '" << psi->module_name << "'";
    PyObject *function = (api->PyObject_GetAttrString)(module, psi->function_name.c_str());
    if (!function) {
      log_py.fatal() << "unable to import Python function " << psi->function_name << " from module" << psi->module_name;
      (api->PyErr_PrintEx)(0);
      assert(0);
    }
    //(api->PyObject_Print)(function, stdout, 0); printf("\n");

    //(api->PyObject_CallFunction)(function, "iii", 1, 2, 3);

    (api->Py_DecRef)(module);

    return function;
  }

  void PythonInterpreter::import_module(const std::string& module_name)
  {
    log_py.debug() << "attempting to import module: " << module_name;
    PyObject *module = (api->PyImport_ImportModule)(module_name.c_str());
    if (!module) {
      log_py.fatal() << "unable to import Python module " << module_name;
      (api->PyErr_PrintEx)(0);
      assert(0);
    }
    (api->Py_DecRef)(module);
  }

  void PythonInterpreter::run_string(const std::string& script_text)
  {
    // from Python.h
    const int Py_file_input = 257;

    log_py.debug() << "running python string: " << script_text;
    PyObject *mainmod = (api->PyImport_ImportModule)("__main__");
    assert(mainmod != 0);
    PyObject *globals = (api->PyModule_GetDict)(mainmod);
    assert(globals != 0);
    PyObject *res = (api->PyRun_String)(script_text.c_str(),
					Py_file_input,
					globals,
					globals);
    if(!res) {
      log_py.fatal() << "unable to run python string:" << script_text;
      (api->PyErr_PrintEx)(0);
      assert(0);
    }
    (api->Py_DecRef)(res);
    (api->Py_DecRef)(globals);
    (api->Py_DecRef)(mainmod);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalPythonProcessor

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

    virtual void shutdown(void);

    virtual void add_to_group(ProcessorGroup *group);

    virtual void register_task(Processor::TaskFuncID func_id,
                               CodeDescriptor& codedesc,
                               const ByteArrayRef& user_data);

    void worker_main(void);

  protected:
    int numa_node;
    CoreReservation *core_rsrv;
    const std::vector<std::string>& import_modules;
    const std::vector<std::string>& init_scripts;

    Thread *worker_thread;
    GASNetHSL mutex;
    GASNetCondVar condvar;
    PythonInterpreter *interpreter;
    bool shutdown_requested;

    struct TaskTableEntry {
      PyObject *fnptr;
      ByteArray user_data;
    };

    struct TaskRegistration {
      Processor::TaskFuncID func_id;
      CodeDescriptor *codedesc;
      ByteArray user_data;
    };

    std::map<Processor::TaskFuncID, TaskTableEntry> task_table;

    std::list<TaskRegistration *> taskreg_queue;
    PriorityQueue<Task *, DummyLock> task_queue; // protected by 'mutex' above
    ProfilingGauges::AbsoluteRangeGauge<int> ready_task_count;
  };

  LocalPythonProcessor::LocalPythonProcessor(Processor _me, int _numa_node,
                                             CoreReservationSet& crs,
                                             size_t _stack_size,
					     const std::vector<std::string>& _import_modules,
					     const std::vector<std::string>& _init_scripts)
    : ProcessorImpl(_me, Processor::PY_PROC)
    , numa_node(_numa_node)
    , import_modules(_import_modules)
    , init_scripts(_init_scripts)
    , condvar(mutex)
    , interpreter(0)
    , shutdown_requested(false)
    , ready_task_count(stringbuilder() << "realm/proc " << me << "/ready tasks")
  {
    task_queue.set_gauge(&ready_task_count);

    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_numa_domain(numa_node);
    params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "Python" << numa_node << " proc " << _me;

    core_rsrv = new CoreReservation(name, crs, params);
    ThreadLaunchParameters tlp;
    worker_thread = Thread::create_kernel_thread<LocalPythonProcessor,
						 &LocalPythonProcessor::worker_main>(this,
										     tlp,
										     *core_rsrv);
  }

  LocalPythonProcessor::~LocalPythonProcessor(void)
  {
    delete core_rsrv;
  }

  void LocalPythonProcessor::shutdown(void)
  {
    log_py.info() << "shutting down";

    {
      AutoHSLLock al(mutex);
      shutdown_requested = true;
      condvar.signal();
    }
    worker_thread->join();
    delete worker_thread;
  }

  void LocalPythonProcessor::worker_main(void)
  {
    log_py.info() << "worker thread started";

    // create a python interpreter that stays entirely within this thread
    interpreter = new PythonInterpreter;

    for(std::vector<std::string>::const_iterator it = import_modules.begin();
	it != import_modules.end();
	++it)
      interpreter->import_module(*it);

    for(std::vector<std::string>::const_iterator it = init_scripts.begin();
	it != init_scripts.end();
	++it)
      interpreter->run_string(*it);

    while(!shutdown_requested) {
      TaskRegistration *todo_reg = 0;
      Task *todo_task = 0;
      {
	AutoHSLLock al(mutex);
	if(!taskreg_queue.empty()) {
	  todo_reg = taskreg_queue.front();
	  taskreg_queue.pop_front();
	} else {
	  if(!task_queue.empty()) {
	    todo_task = task_queue.get(0);
	  } else {
	    log_py.debug() << "no work - sleeping";
	    condvar.wait();
	    log_py.debug() << "awake again";
	  }
	}
      }
      if(todo_reg) {
	// first, make sure we haven't seen this task id before
	if(task_table.count(todo_reg->func_id) > 0) {
	  log_py.fatal() << "duplicate task registration: proc=" << me << " func=" << todo_reg->func_id;
	  assert(0);
	}

	// next, see if we have a Python function to register
	const PythonSourceImplementation *psi = todo_reg->codedesc->find_impl<PythonSourceImplementation>();
	if(!psi) {
	  log_py.fatal() << "invalid code descriptor for python proc: " << *(todo_reg->codedesc);
	  assert(0);
	}
	PyObject *fnptr = interpreter->find_or_import_function(psi);
	assert(fnptr != 0);

	log_py.info() << "task " << todo_reg->func_id << " registered on " << me << ": " << *(todo_reg->codedesc);

	TaskTableEntry &tte = task_table[todo_reg->func_id];
	tte.fnptr = fnptr;
	tte.user_data.swap(todo_reg->user_data);

	delete todo_reg->codedesc;
	delete todo_reg;
      }
      if(todo_task) {
	log_py.debug() << "running task";
	todo_task->execute_on_processor(me);
	log_py.debug() << "done running task";
      }
    }

    log_py.info() << "worker thread shutting down";

    delete interpreter;
    interpreter = 0;

    log_py.info() << "cleanup complete";
  }

  void LocalPythonProcessor::enqueue_task(Task *task)
  {
    // just jam it into the task queue, signal worker if needed
    if(task->mark_ready()) {
      AutoHSLLock al(mutex);
      bool was_empty = taskreg_queue.empty() && task_queue.empty();
      task_queue.put(task, task->priority);
      if(was_empty)
	condvar.signal();
    } else
      task->mark_finished(false /*!successful*/);
  }

  void LocalPythonProcessor::spawn_task(Processor::TaskFuncID func_id,
					const void *args, size_t arglen,
					const ProfilingRequestSet &reqs,
					Event start_event, Event finish_event,
					int priority)
  {
    // create a task object for this
    Task *task = new Task(me, func_id, args, arglen, reqs,
			  start_event, finish_event, priority);
    get_runtime()->optable.add_local_operation(finish_event, task);

    // if the start event has already triggered, we can enqueue right away
    bool poisoned = false;
    if (start_event.has_triggered_faultaware(poisoned)) {
      if(poisoned) {
	log_poison.info() << "cancelling poisoned task - task=" << task << " after=" << task->get_finish_event();
	task->handle_poisoned_precondition(start_event);
      } else
	enqueue_task(task);
    } else {
      EventImpl::add_waiter(start_event, new DeferredTaskSpawn(this, task));
    }
  }

  void LocalPythonProcessor::add_to_group(ProcessorGroup *group)
  {
    assert(0);
  }

  void LocalPythonProcessor::register_task(Processor::TaskFuncID func_id,
                                           CodeDescriptor& codedesc,
                                           const ByteArrayRef& user_data)
  {
    TaskRegistration *treg = new TaskRegistration;
    treg->func_id = func_id;
    treg->codedesc = new CodeDescriptor(codedesc);
    treg->user_data = user_data;
    {
      AutoHSLLock al(mutex);
      bool was_empty = taskreg_queue.empty() && task_queue.empty();
      taskreg_queue.push_back(treg);
      if(was_empty)
	condvar.signal();
    }
#if 0
    // first, make sure we haven't seen this task id before
    if(task_table.count(func_id) > 0) {
      log_py.fatal() << "duplicate task registration: proc=" << me << " func=" << func_id;
      assert(0);
    }

    // next, get see if we have a Python function to register
    const PythonSourceImplementation *psi = codedesc.find_impl<PythonSourceImplementation>();
    assert(psi != 0);

    PyObject *fnptr = interpreter->find_or_import_function(psi);

    log_py.info() << "task " << func_id << " registered on " << me << ": " << codedesc;

    TaskTableEntry &tte = task_table[func_id];
    tte.fnptr = fnptr;
    tte.user_data = user_data;
#endif
  }

  void LocalPythonProcessor::execute_task(Processor::TaskFuncID func_id,
					  const ByteArrayRef& task_args)
  {
    std::map<Processor::TaskFuncID, TaskTableEntry>::const_iterator it = task_table.find(func_id);
    if(it == task_table.end()) {
      // TODO: remove this hack once the tools are available to the HLR to call these directly
      if(func_id < Processor::TASK_ID_FIRST_AVAILABLE) {
	log_py.info() << "task " << func_id << " not registered on " << me << ": ignoring missing legacy setup/shutdown task";
	return;
      }
      log_py.fatal() << "task " << func_id << " not registered on " << me;
      assert(0);
    }

    const TaskTableEntry& tte = it->second;

    log_py.debug() << "task " << func_id << " executing on " << me << ": " << ((void *)(tte.fnptr));

    PyObject *arg1 = (interpreter->api->PyByteArray_FromStringAndSize)(
                                                   (const char *)task_args.base(),
						   task_args.size());
    assert(arg1 != 0);
    PyObject *arg2 = (interpreter->api->PyByteArray_FromStringAndSize)(
                                                   (const char *)tte.user_data.base(),
						   tte.user_data.size());
    assert(arg2 != 0);
    // TODO: make into a Python realm.Processor object
    PyObject *arg3 = (interpreter->api->PyLong_FromUnsignedLong)(me.id);
    assert(arg3 != 0);

    PyObject *args = (interpreter->api->PyTuple_New)(3);
    assert(args != 0);
    (interpreter->api->PyTuple_SetItem)(args, 0, arg1);
    (interpreter->api->PyTuple_SetItem)(args, 1, arg2);
    (interpreter->api->PyTuple_SetItem)(args, 2, arg3);

    //printf("args = "); (interpreter->api->PyObject_Print)(args, stdout, 0); printf("\n");

    PyObject *res = (interpreter->api->PyObject_CallObject)(tte.fnptr, args);

    (interpreter->api->Py_DecRef)(args);

    //printf("res = "); PyObject_Print(res, stdout, 0); printf("\n");
    if(res != 0) {
      (interpreter->api->Py_DecRef)(res);
    } else {
      log_py.fatal() << "python exception occurred within task:";
      (interpreter->api->PyErr_PrintEx)(0);
      assert(0);
    }

  }

  namespace Python {

    ////////////////////////////////////////////////////////////////////////
    //
    // class PythonModule

    PythonModule::PythonModule(void)
      : Module("python")
      , cfg_num_python_cpus(0)
      , cfg_use_numa(false)
      , cfg_stack_size_in_mb(2)
    {
    }

    PythonModule::~PythonModule(void)
    {}

    /*static*/ Module *PythonModule::create_module(RuntimeImpl *runtime,
                                                 std::vector<std::string>& cmdline)
    {
      // create a module to fill in with stuff - we'll delete it if numa is
      //  disabled
      PythonModule *m = new PythonModule;

      // first order of business - read command line parameters
      {
        CommandLineParser cp;

        cp.add_option_int("-ll:py", m->cfg_num_python_cpus)
	  .add_option_int("-ll:pynuma", m->cfg_use_numa)
	  .add_option_int("-ll:pystack", m->cfg_stack_size_in_mb)
	  .add_option_stringlist("-ll:pyimport", m->cfg_import_modules)
	  .add_option_stringlist("-ll:pyinit", m->cfg_init_scripts);

        bool ok = cp.parse_command_line(cmdline);
        if(!ok) {
          log_py.fatal() << "error reading Python command line parameters";
          assert(false);
        }
      }

      // if no cpus were requested, there's no point
      if(m->cfg_num_python_cpus == 0) {
        log_py.debug() << "no Python cpus requested";
        delete m;
        return 0;
      }

#ifndef REALM_USE_DLMOPEN
      // Multiple CPUs are only allowed if we're using dlmopen.
      if(m->cfg_num_python_cpus > 1) {
        log_py.fatal() << "support for multiple Python CPUs is not available: recompile with USE_DLMOPEN";
        assert(false);
      }
#endif

      // get number/sizes of NUMA nodes -
      //   disable (with a warning) numa binding if support not found
      if(m->cfg_use_numa) {
        std::map<int, NumaNodeCpuInfo> cpuinfo;
        if(numasysif_numa_available() &&
           numasysif_get_cpu_info(cpuinfo) &&
           !cpuinfo.empty()) {
          // filter out any numa domains with insufficient core counts
          int cores_needed = m->cfg_num_python_cpus;
          for(std::map<int, NumaNodeCpuInfo>::const_iterator it = cpuinfo.begin();
              it != cpuinfo.end();
              ++it) {
            const NumaNodeCpuInfo& ci = it->second;
            if(ci.cores_available >= cores_needed) {
              m->active_numa_domains.insert(ci.node_id);
            } else {
              log_py.warning() << "not enough cores in NUMA domain " << ci.node_id << " (" << ci.cores_available << " < " << cores_needed << ")";
            }
          }
        } else {
          log_py.warning() << "numa support not found (or not working)";
          m->cfg_use_numa = false;
        }
      }

      // if we don't end up with any active numa domains,
      //  use NUMA_DOMAIN_DONTCARE
      // actually, use the value (-1) since it seems to cause link errors!?
      if(m->active_numa_domains.empty())
        m->active_numa_domains.insert(-1 /*CoreReservationParameters::NUMA_DOMAIN_DONTCARE*/);

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void PythonModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void PythonModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);

      for(std::set<int>::const_iterator it = active_numa_domains.begin();
          it != active_numa_domains.end();
          ++it) {
        int cpu_node = *it;
        for(int i = 0; i < cfg_num_python_cpus; i++) {
          Processor p = runtime->next_local_processor_id();
          ProcessorImpl *pi = new LocalPythonProcessor(p, cpu_node,
                                                       runtime->core_reservation_set(),
                                                       cfg_stack_size_in_mb << 20,
						       cfg_import_modules,
						       cfg_init_scripts);
          runtime->add_processor(pi);

          // create affinities between this processor and system/reg memories
          // if the memory is one we created, use the kernel-reported distance
          // to adjust the answer
          std::vector<MemoryImpl *>& local_mems = runtime->nodes[gasnet_mynode()].memories;
          for(std::vector<MemoryImpl *>::iterator it2 = local_mems.begin();
              it2 != local_mems.end();
              ++it2) {
            Memory::Kind kind = (*it2)->get_kind();
            if((kind != Memory::SYSTEM_MEM) && (kind != Memory::REGDMA_MEM))
              continue;

            Machine::ProcessorMemoryAffinity pma;
            pma.p = p;
            pma.m = (*it2)->me;

            // use the same made-up numbers as in
            //  runtime_impl.cc
            if(kind == Memory::SYSTEM_MEM) {
              pma.bandwidth = 100;  // "large"
              pma.latency = 5;      // "small"
            } else {
              pma.bandwidth = 80;   // "large"
              pma.latency = 10;     // "small"
            }

            runtime->add_proc_mem_affinity(pma);
          }
        }
      }
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void PythonModule::cleanup(void)
    {
      Module::cleanup();
    }

  }; // namespace Python

}; // namespace Realm
