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

#include "realm/python/python_module.h"
#include "realm/python/python_internal.h"

#include "realm/numa/numasysif.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/proc_impl.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"

#include <dlfcn.h>
#include <link.h>

#include <list>

namespace Realm {

  Logger log_py("python");

  ////////////////////////////////////////////////////////////////////////
  //
  // class PythonAPI

  PythonAPI::PythonAPI(void *_handle)
    : handle(_handle)
  {
    get_symbol(this->Py_DecRef, "Py_DecRef");
    get_symbol(this->Py_Finalize, "Py_Finalize");
    get_symbol(this->Py_InitializeEx, "Py_InitializeEx");

    get_symbol(this->PyByteArray_FromStringAndSize, "PyByteArray_FromStringAndSize");

    get_symbol(this->PyEval_InitThreads, "PyEval_InitThreads");
    get_symbol(this->PyEval_RestoreThread, "PyEval_RestoreThread");
    get_symbol(this->PyEval_SaveThread, "PyEval_SaveThread");

    get_symbol(this->PyThreadState_New, "PyThreadState_New");
    get_symbol(this->PyThreadState_Clear, "PyThreadState_Clear");
    get_symbol(this->PyThreadState_Delete, "PyThreadState_Delete");
    get_symbol(this->PyThreadState_Get, "PyThreadState_Get");
    get_symbol(this->PyThreadState_Swap, "PyThreadState_Swap");

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

#ifdef REALM_USE_DLMOPEN
  // dlmproxy symbol lookups have to happen in a function we define so that
  //  dl[v]sym searches in the right place
  static void *dlmproxy_lookup(const char *symname, const char *symver)
  {
    
    void *handle = 0;
    void *sym = (symver ?
		   dlvsym(handle, symname, symver) :
		   dlsym(handle, symname));
    if(sym)
      log_py.debug() << "found symbol: name=" << symname << " ver=" << (symver ? symver : "(none)") << " ptr=" << sym;
    else
      log_py.warning() << "missing symbol: name=" << symname << " ver=" << (symver ? symver : "(none)");
    return sym;
  }
#endif

  PythonInterpreter::PythonInterpreter() 
  {
#ifdef REALM_PYTHON_LIB
    const char *python_lib = REALM_PYTHON_LIB;
#else
    const char *python_lib = "libpython2.7.so";
#endif

#ifdef REALM_USE_DLMOPEN
    // loading libpython into its own namespace will cause it to try to bring
    //   in a second copy of libpthread.so.0, which is fairly disastrous
    // we deal with it by loading a "dlmproxy" of pthreads that tunnels all 
    //   pthreads calls back to the (only) version in the main executable
    const char *dlmproxy_filename = getenv("DLMPROXY_LIBPTHREAD");
    if(!dlmproxy_filename)
      dlmproxy_filename = "dlmproxy_libpthread.so.0";
    dlmproxy_handle = dlmopen(LM_ID_NEWLM,
			      dlmproxy_filename,
			      RTLD_DEEPBIND | RTLD_GLOBAL | RTLD_LAZY);
    if(!dlmproxy_handle) {
      const char *error = dlerror();
      log_py.fatal() << "HELP!  Use of dlmopen for python requires dlmproxy for pthreads!  Failed to\n"
		     << "  load: " << dlmproxy_filename << "\n"
		     << "  error: " << error;
      assert(false);
    }

    // now that the proxy is loaded, we need to tell it where the real
    //  libpthreads functions are
    {
      void *sym = dlsym(dlmproxy_handle, "dlmproxy_load_symbols");
      assert(sym != 0);
      ((void (*)(void *(*)(const char *, const char *)))sym)(dlmproxy_lookup);
    }

    // now we can load libpython, but make sure we do it in the new namespace
    Lmid_t lmid;
    int ret = dlinfo(dlmproxy_handle, RTLD_DI_LMID, &lmid);
    assert(ret == 0);

    handle = dlmopen(lmid, python_lib, RTLD_DEEPBIND | RTLD_GLOBAL | RTLD_NOW);
#else
    // life is so much easier if we use dlopen (but we only get one copy then)
    handle = dlopen(python_lib, RTLD_GLOBAL | RTLD_LAZY);
#endif
    if (!handle) {
      const char *error = dlerror();
      log_py.fatal() << error;
      assert(false);
    }

    api = new PythonAPI(handle);

    (api->Py_InitializeEx)(0 /*!initsigs*/);
    (api->PyEval_InitThreads)();
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

    if (dlclose(handle)) {
      const char *error = dlerror();
      log_py.fatal() << "libpython dlclose error: " << error;
      assert(false);
    }

#ifdef REALM_USE_DLMOPEN
    if (dlclose(dlmproxy_handle)) {
      const char *error = dlerror();
      log_py.fatal() << "dlmproxy dlclose error: " << error;
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

    PyObject *function = module;
    for (std::vector<std::string>::const_iterator it = psi->function_name.begin(),
           ie = psi->function_name.end(); function && it != ie; ++it) {
      function = (api->PyObject_GetAttrString)(function, it->c_str());
    }
    if (!function) {
      {
        LoggerMessage m = log_py.fatal();
        m << "unable to import Python function " << psi->module_name;
        for (std::vector<std::string>::const_iterator it = psi->function_name.begin(),
               ie = psi->function_name.begin(); it != ie; ++it) {
          m << "." << *it;
        }
      }
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
  // class PythonThreadTaskScheduler

  PythonThreadTaskScheduler::PythonThreadTaskScheduler(LocalPythonProcessor *_pyproc,
						       CoreReservation& _core_rsrv)
    : KernelThreadTaskScheduler(_pyproc->me, _core_rsrv)
    , pyproc(_pyproc)
    , interpreter_ready(false)
  {}

  void PythonThreadTaskScheduler::enqueue_taskreg(LocalPythonProcessor::TaskRegistration *treg)
  {
    AutoHSLLock al(lock);
    taskreg_queue.push_back(treg);
    // we've added work to the system
    work_counter.increment_counter();
  }

  void PythonThreadTaskScheduler::python_scheduler_loop(void)
  {
    // global startup of python interpreter if needed
    if(!interpreter_ready) {
      log_py.info() << "creating interpreter";
      pyproc->create_interpreter();
      interpreter_ready = true;
    }

    // always create and remember our own python thread - does NOT require GIL
    PyThreadState *pythread = (pyproc->interpreter->api->PyThreadState_New)(pyproc->master_thread->interp);
    log_py.debug() << "created python thread: " << pythread;
    
    assert(pythread != 0);
    assert(pythreads.count(Thread::self()) == 0);
    pythreads[Thread::self()] = pythread;

    // now go into main scheduler loop, holding scheduler lock for whole thing
    AutoHSLLock al(lock);
    while(true) {
      // remember the work counter value before we start so that we don't iterate
      //   unnecessarily
      long long old_work_counter = work_counter.read_counter();

      // first priority - task registration
      while(!taskreg_queue.empty()) {
	LocalPythonProcessor::TaskRegistration *treg = taskreg_queue.front();
	taskreg_queue.pop_front();
	
	// one fewer unassigned worker
	update_worker_count(0, -1);
	
	// we'll run the task after letting go of the lock, but update this thread's
	//  priority here
	worker_priorities[Thread::self()] = TaskQueue::PRI_POS_INF;

	// release the lock while we run the task
	lock.unlock();

#ifndef NDEBUG
	bool ok =
#endif
	  pyproc->perform_task_registration(treg);
	assert(ok);  // no fault recovery yet

	lock.lock();

	worker_priorities.erase(Thread::self());

	// and we're back to being unassigned
	update_worker_count(0, +1);
      }

      // if we have both resumable and new ready tasks, we want the one that
      //  is the highest priority, with ties going to resumable tasks - we
      //  can do this cleanly by taking advantage of the fact that the
      //  resumable_workers queue uses the scheduler lock, so can't change
      //  during this call
      // peek at the top thing (if any) in that queue, and then try to find
      //  a ready task with higher priority
      int resumable_priority = ResumableQueue::PRI_NEG_INF;
      resumable_workers.peek(&resumable_priority);

      // try to get a new task then
      // remember where a task has come from in case we want to put it back
      Task *task = 0;
      TaskQueue *task_source = 0;
      int task_priority = resumable_priority;
      for(std::vector<TaskQueue *>::const_iterator it = task_queues.begin();
	  it != task_queues.end();
	  it++) {
	int new_priority;
	Task *new_task = (*it)->get(&new_priority, task_priority);
	if(new_task) {
	  // if we got something better, put back the old thing (if any)
	  if(task)
	    task_source->put(task, task_priority, false); // back on front of list
	  
	  task = new_task;
	  task_source = *it;
	  task_priority = new_priority;
	}
      }

      // did we find work to do?
      if(task) {
	// one fewer unassigned worker
	update_worker_count(0, -1);

	// we'll run the task after letting go of the lock, but update this thread's
	//  priority here
	worker_priorities[Thread::self()] = task_priority;

	// release the lock while we run the task
	lock.unlock();

	// make our python thread state active, acquiring the GIL
	assert((pyproc->interpreter->api->PyThreadState_Swap)(0) == 0);
	log_py.debug() << "RestoreThread <- " << pythread;
	(pyproc->interpreter->api->PyEval_RestoreThread)(pythread);

#ifndef NDEBUG
	bool ok =
#endif
	  execute_task(task);
	assert(ok);  // no fault recovery yet

	// release the GIL
	PyThreadState *saved = (pyproc->interpreter->api->PyEval_SaveThread)();
	log_py.debug() << "SaveThread -> " << saved;
	assert(saved == pythread);

	lock.lock();

	worker_priorities.erase(Thread::self());

	// and we're back to being unassigned
	update_worker_count(0, +1);
	continue;
      }

      // having checked for higher-priority ready tasks, we can always
      //  take the highest-priority resumable task, if any, and run it
      if(!resumable_workers.empty()) {
	Thread *yield_to = resumable_workers.get(0); // priority is irrelevant
	assert(yield_to != Thread::self());

	// this should only happen if we're at the max active worker count (otherwise
	//  somebody should have just woken this guy up earlier), and reduces the 
	// unassigned worker count by one
	update_worker_count(0, -1);

	idle_workers.push_back(Thread::self());
	worker_sleep(yield_to);

	// loop around and check both queues again
	continue;
      }

      {
	// no ready or resumable tasks?  thumb twiddling time

	// are we shutting down?
	if(shutdown_flag) {
	  // yes, we can terminate - wake up an idler (if any) first though
	  if(!idle_workers.empty()) {
	    Thread *to_wake = idle_workers.back();
	    idle_workers.pop_back();
	    // no net change in worker counts
	    worker_terminate(to_wake);
	  } else {
	    // nobody to wake, so -1 active/unassigned worker
	    update_worker_count(-1, -1, false); // ok to drop below mins
	    worker_terminate(0);
	  }
	  return;
	}

	// do we have more unassigned and idle tasks than we need?
	int total_idle_count = (unassigned_worker_count +
				(int)(idle_workers.size()));
	if(total_idle_count > cfg_max_idle_workers) {
	  // if there are sleeping idlers, terminate in favor of one of those - keeps
	  //  worker counts constant
	  if(!idle_workers.empty()) {
	    Thread *to_wake = idle_workers.back();
	    assert(to_wake != Thread::self());
	    idle_workers.pop_back();
	    // no net change in worker counts
	    worker_terminate(to_wake);
	    return;
	  }
	}

	// no, stay awake but suspend until there's a chance that the next iteration
	//  of this loop would turn out different
	wait_for_work(old_work_counter);
      }
    }

    // should never get here
    assert(0);
  }

  Thread *PythonThreadTaskScheduler::worker_create(bool make_active)
  {
    // lock is held by caller
    ThreadLaunchParameters tlp;
    Thread *t = Thread::create_kernel_thread<PythonThreadTaskScheduler,
					     &PythonThreadTaskScheduler::python_scheduler_loop>(this,
												tlp,
												core_rsrv,
												this);
    all_workers.insert(t);
    if(make_active)
      active_workers.insert(t);
    return t;
  }
 
  // called by a worker thread when it needs to wait for something (and we
  //   should release the GIL)
  void PythonThreadTaskScheduler::thread_blocking(Thread *thread)
  {
    // if this gets called before we're done initializing the interpreter,
    //  we need a simple blocking wait
    if(!interpreter_ready) {
      AutoHSLLock al(lock);

      log_py.debug() << "waiting during initialization";
      bool really_blocked = try_update_thread_state(thread,
						    Thread::STATE_BLOCKING,
						    Thread::STATE_BLOCKED);
      if(!really_blocked) return;

      while(true) {
	long long old_work_counter = work_counter.read_counter();

	if(!resumable_workers.empty()) {
	  Thread *t = resumable_workers.get(0);
	  assert(t == thread);
	  log_py.debug() << "awake again";
	  return;
	}

	wait_for_work(old_work_counter);
      }
    }

    // if we got here through a cffi call, the GIL has already been released,
    //  so try to handle that case here - a call PyEval_SaveThread
    //  if the GIL is not held will assert-fail, and while a call to
    //  PyThreadState_Swap is technically illegal (and unsafe if python-created
    //  threads exist), it does what we want for now
    PyThreadState *saved = (pyproc->interpreter->api->PyThreadState_Swap)(0);
    if(saved != 0) {
      log_py.info() << "python worker sleeping - releasing GIL";
      // put it back so we can save it properly
      (pyproc->interpreter->api->PyThreadState_Swap)(saved);
      // would like to sanity-check that this returns the expected thread state,
      //  but that would require taking the PythonThreadTaskScheduler's lock
      (pyproc->interpreter->api->PyEval_SaveThread)();
      log_py.debug() << "SaveThread -> " << saved;
    } else
      log_py.info() << "python worker sleeping - GIL already released";
    
    KernelThreadTaskScheduler::thread_blocking(thread);

    if(saved) {
      log_py.info() << "python worker awake - acquiring GIL";
      log_py.debug() << "RestoreThread <- " << saved;
      (pyproc->interpreter->api->PyEval_RestoreThread)(saved);
    } else
      log_py.info() << "python worker awake - not acquiring GIL";
  }

  void PythonThreadTaskScheduler::thread_ready(Thread *thread)
  {
    // handle the wakening of the initialization thread specially
    if(!interpreter_ready) {
      AutoHSLLock al(lock);
      resumable_workers.put(thread, 0);
    } else {
      KernelThreadTaskScheduler::thread_ready(thread);
    }
  }

  void PythonThreadTaskScheduler::worker_terminate(Thread *switch_to)
  {
    // before we can kill the kernel thread, we need to tear down the python thread
    std::map<Thread *, PyThreadState *>::iterator it = pythreads.find(Thread::self());
    assert(it != pythreads.end());
    PyThreadState *pythread = it->second;
    pythreads.erase(it);

    log_py.debug() << "destroying python thread: " << pythread;
    
    // our thread should not be active
    assert((pyproc->interpreter->api->PyThreadState_Swap)(0) == 0);

    // switch to the master thread, retaining the GIL
    log_py.debug() << "RestoreThread <- " << pyproc->master_thread;
    (pyproc->interpreter->api->PyEval_RestoreThread)(pyproc->master_thread);

    // clear and delete the worker thread
    (pyproc->interpreter->api->PyThreadState_Clear)(pythread);
    (pyproc->interpreter->api->PyThreadState_Delete)(pythread);

    // release the GIL
    PyThreadState *saved = (pyproc->interpreter->api->PyEval_SaveThread)();
    log_py.debug() << "SaveThread -> " << saved;
    assert(saved == pyproc->master_thread);

    // TODO: tear down interpreter if last thread
    if(shutdown_flag && pythreads.empty())
      pyproc->destroy_interpreter();

    KernelThreadTaskScheduler::worker_terminate(switch_to);
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalPythonProcessor

  LocalPythonProcessor::LocalPythonProcessor(Processor _me, int _numa_node,
                                             CoreReservationSet& crs,
                                             size_t _stack_size,
					     const std::vector<std::string>& _import_modules,
					     const std::vector<std::string>& _init_scripts)
    : ProcessorImpl(_me, Processor::PY_PROC)
    , numa_node(_numa_node)
    , import_modules(_import_modules)
    , init_scripts(_init_scripts)
    , interpreter(0)
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

    sched = new PythonThreadTaskScheduler(this, *core_rsrv);
    sched->add_task_queue(&task_queue);
  }

  LocalPythonProcessor::~LocalPythonProcessor(void)
  {
    delete core_rsrv;
    delete sched;
  }

  // starts worker threads and performs any per-processor initialization
  void LocalPythonProcessor::start_threads(void)
  {
    // finally, fire up the scheduler
    sched->start();
  }

  void LocalPythonProcessor::shutdown(void)
  {
    log_py.info() << "shutting down";

    sched->shutdown();
  }

  void LocalPythonProcessor::create_interpreter(void)
  {
    assert(interpreter == 0);
  
    // create a python interpreter that stays entirely within this thread
    interpreter = new PythonInterpreter;
    master_thread = (interpreter->api->PyThreadState_Get)();

    // always need the python threading module
    interpreter->import_module("threading");
    
    // perform requested initialization
    for(std::vector<std::string>::const_iterator it = import_modules.begin();
	it != import_modules.end();
	++it)
      interpreter->import_module(*it);

    for(std::vector<std::string>::const_iterator it = init_scripts.begin();
	it != init_scripts.end();
	++it)
      interpreter->run_string(*it);

    // default state is GIL _released_
    PyThreadState *saved = (interpreter->api->PyEval_SaveThread)();
    log_py.debug() << "SaveThread -> " << saved;
    assert(saved == master_thread);
  }

  void LocalPythonProcessor::destroy_interpreter(void)
  {
    assert(interpreter != 0);

    // take GIL with master thread
    assert((interpreter->api->PyThreadState_Swap)(0) == 0);
    log_py.debug() << "RestoreThread <- " << master_thread;
    (interpreter->api->PyEval_RestoreThread)(master_thread);

    // during shutdown, the threading module tries to remove the Thread object
    //  associated with this kernel thread - if that doesn't exist (because we're
    //  shutting down from a different thread that we initialized the interpreter
    //  _and_ nobody called threading.current_thread() from this kernel thread),
    //  we'll get a KeyError in threading.py
    // resolve this by calling threading.current_thread() here, using __import__
    //  to deal with the case where 'import threading' never got called
    (interpreter->api->PyRun_SimpleString)("__import__('threading').current_thread()");

    delete interpreter;
    interpreter = 0;
    master_thread = 0;
  }
  
  bool LocalPythonProcessor::perform_task_registration(LocalPythonProcessor::TaskRegistration *treg)
  {
    // first, make sure we haven't seen this task id before
    if(task_table.count(treg->func_id) > 0) {
      log_py.fatal() << "duplicate task registration: proc=" << me << " func=" << treg->func_id;
      assert(0);
    }

    // next, see if we have a Python function to register
    const PythonSourceImplementation *psi = treg->codedesc->find_impl<PythonSourceImplementation>();
    if(!psi) {
      log_py.fatal() << "invalid code descriptor for python proc: " << *(treg->codedesc);
      assert(0);
    }

    // perform import/compile on master thread
    assert((interpreter->api->PyThreadState_Swap)(0) == 0);
    log_py.debug() << "RestoreThread <- " << master_thread;
    (interpreter->api->PyEval_RestoreThread)(master_thread);
    
    PyObject *fnptr = interpreter->find_or_import_function(psi);
    assert(fnptr != 0);

    PyThreadState *saved = (interpreter->api->PyEval_SaveThread)();
    log_py.debug() << "SaveThread -> " << saved;
    assert(saved == master_thread);

    log_py.info() << "task " << treg->func_id << " registered on " << me << ": " << *(treg->codedesc);

    TaskTableEntry &tte = task_table[treg->func_id];
    tte.fnptr = fnptr;
    tte.user_data.swap(treg->user_data);

    delete treg->codedesc;
    delete treg;

    return true;
  }

  void LocalPythonProcessor::enqueue_task(Task *task)
  {
    // just jam it into the task queue, scheduler will take care of the rest
    if(task->mark_ready())
      task_queue.put(task, task->priority);
    else
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
    // add the group's task queue to our scheduler too
    sched->add_task_queue(&group->task_queue);
  }

  void LocalPythonProcessor::register_task(Processor::TaskFuncID func_id,
                                           CodeDescriptor& codedesc,
                                           const ByteArrayRef& user_data)
  {
    TaskRegistration *treg = new TaskRegistration;
    treg->func_id = func_id;
    treg->codedesc = new CodeDescriptor(codedesc);
    treg->user_data = user_data;
    sched->enqueue_taskreg(treg);
#if 0
    {
      AutoHSLLock al(mutex);
      bool was_empty = taskreg_queue.empty() && task_queue.empty();
      taskreg_queue.push_back(treg);
      if(was_empty)
	condvar.signal();
    }
#endif
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

    /*static*/ std::vector<std::string> PythonModule::extra_import_modules;

    PythonModule::PythonModule(void)
      : Module("python")
      , cfg_num_python_cpus(0)
      , cfg_use_numa(false)
      , cfg_stack_size_in_mb(2)
    {
    }

    PythonModule::~PythonModule(void)
    {}

    /*static*/ void PythonModule::import_python_module(const char *module_name)
    {
      extra_import_modules.push_back(module_name);
    }

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

      // add extra module imports requested by the application
      m->cfg_import_modules.insert(m->cfg_import_modules.end(),
                                   extra_import_modules.begin(),
                                   extra_import_modules.end());

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
          std::vector<MemoryImpl *>& local_mems = runtime->nodes[my_node_id].memories;
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
