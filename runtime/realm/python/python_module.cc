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

namespace Realm {

  Logger log_py("python");

  class PyObject; // FIXME: Should probably get a definition of this from python.h

  ////////////////////////////////////////////////////////////////////////
  //
  // class PythonInterpreter

  class PythonInterpreter {
  public:
    PythonInterpreter();
    ~PythonInterpreter();

    PyObject *find_or_import_function(const PythonSourceImplementation *psi);

  protected:
    void *handle;

    template<typename T>
    void get_symbol(T &fn, const char *symbol);
  };

  template<typename T>
  void PythonInterpreter::get_symbol(T &fn, const char *symbol)
  {
    fn = reinterpret_cast<T>(dlsym(handle, symbol));
    if (!fn) {
      const char *error = dlerror();
      log_py.fatal() << error;
      assert(false);
    }
  }

  PythonInterpreter::PythonInterpreter() {
    handle = dlmopen(LM_ID_NEWLM, "libpython2.7.so", RTLD_DEEPBIND | RTLD_LOCAL | RTLD_LAZY);
    if (!handle) {
      const char *error = dlerror();
      log_py.fatal() << error;
      assert(false);
    }

    void (*Py_Initialize)(void);
    get_symbol(Py_Initialize, "Py_Initialize");
    Py_Initialize();


    void (*PyRun_SimpleString)(const char *);
    get_symbol(PyRun_SimpleString, "PyRun_SimpleString");
    PyRun_SimpleString("print 'hello Python world!'");
  }

  PythonInterpreter::~PythonInterpreter() {
    void (*Py_Finalize)(void);
    get_symbol(Py_Finalize, "Py_Finalize");
    Py_Finalize();

    if (dlclose(handle)) {
      const char *error = dlerror();
      log_py.fatal() << error;
      assert(false);
    }
  }

  PyObject *PythonInterpreter::find_or_import_function(const PythonSourceImplementation *psi)
  {
    PyObject *(*PyImport_ImportModule)(const char *);
    get_symbol(PyImport_ImportModule, "PyImport_ImportModule");
    PyObject *module = PyImport_ImportModule(psi->module_name.c_str());
    if (!module) {
      // FIXME: Check exception for message
      log_py.fatal() << "unable to import Python module " << psi->module_name;
      assert(0);
    }

    PyObject *(*PyObject_GetAttr)(PyObject *, const char *);
    get_symbol(PyObject_GetAttr, "PyObject_GetAttr");
    PyObject *function = PyObject_GetAttr(module, psi->function_name.c_str());
    if (!function) {
      // FIXME: Check exception for message
      log_py.fatal() << "unable to import Python function " << psi->function_name << " from module" << psi->module_name;
      assert(0);
    }

    return function;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalPythonProcessor

  // this is nearly identical to a LocalCPUProcessor, but it asks for its thread(s)
  //  to run on the specified numa domain

  class LocalPythonProcessor : public LocalTaskProcessor {
  public:
    LocalPythonProcessor(Processor _me, int _numa_node,
                         CoreReservationSet& crs, size_t _stack_size);
    virtual ~LocalPythonProcessor(void);

    virtual void shutdown(void);

    virtual void register_task(Processor::TaskFuncID func_id,
                               CodeDescriptor& codedesc,
                               const ByteArrayRef& user_data);

    virtual void execute_task(Processor::TaskFuncID func_id,
                              const ByteArrayRef& task_args);

  protected:
    int numa_node;
    CoreReservation *core_rsrv;
    PythonInterpreter *interpreter;

    struct TaskTableEntry {
      PyObject *fnptr;
      ByteArray user_data;
    };

    std::map<Processor::TaskFuncID, TaskTableEntry> task_table;
  };

  LocalPythonProcessor::LocalPythonProcessor(Processor _me, int _numa_node,
                                             CoreReservationSet& crs,
                                             size_t _stack_size)
    : LocalTaskProcessor(_me, Processor::PY_PROC)
    , numa_node(_numa_node)
  {
    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_numa_domain(numa_node);
    params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "Python" << numa_node << " proc " << _me;

    core_rsrv = new CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS
    UserThreadTaskScheduler *sched = new UserThreadTaskScheduler(me, *core_rsrv);
    // no config settings we want to tweak yet
#else
    KernelThreadTaskScheduler *sched = new KernelThreadTaskScheduler(me, *core_rsrv);
    sched->cfg_max_idle_workers = 3; // keep a few idle threads around
#endif
    set_scheduler(sched);

    interpreter = new PythonInterpreter();
  }

  LocalPythonProcessor::~LocalPythonProcessor(void)
  {
    delete interpreter;
    delete core_rsrv;
  }

  void LocalPythonProcessor::shutdown(void)
  {
    log_py.info() << "shutting down";

    LocalTaskProcessor::shutdown();
  }

  void LocalPythonProcessor::register_task(Processor::TaskFuncID func_id,
                                           CodeDescriptor& codedesc,
                                           const ByteArrayRef& user_data)
  {
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
  }

  void LocalPythonProcessor::execute_task(Processor::TaskFuncID func_id,
                                          const ByteArrayRef& task_args)
  {
    LocalTaskProcessor::execute_task(func_id, task_args);
  }


  namespace Python {

    ////////////////////////////////////////////////////////////////////////
    //
    // class PythonModule

    PythonModule::PythonModule(void)
      : Module("python")
      , cfg_num_python_cpus(0)
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
	  .add_option_int("-ll:pystack", m->cfg_stack_size_in_mb);

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
                                                       cfg_stack_size_in_mb << 20);
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
