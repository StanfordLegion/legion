#include "realm.h"
#include "realm/cmdline.h"
#include "realm/network.h"
#include "realm/cuda/cuda_module.h"
#include "realm/hip/hip_module.h"
#include "realm/numa/numa_module.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

#ifdef REALM_ON_WINDOWS
#include <malloc.h>
#else
#include <alloca.h>
#endif

using namespace Realm;

Logger log_app("app");

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

namespace TestConfig {
  // core module
  int num_cpu_procs = 4;
  int num_util_procs = 2;
  int num_io_procs = 1;
  size_t sysmem_size = 16 * 1024 * 1024;
  size_t stack_size = 4 * 1024 * 1024;
  bool pin_util_procs = true;
  bool use_ext_sysmem = false;
  size_t regmem_size = 2 * 1024 * 1024;

  // numa module
  size_t numa_mem_size = 2 * 1024 * 1024;
  size_t numa_nocpu_mem_size = 1 * 1024 * 1024;
  int num_numa_cpus = 1;
  bool numa_pin_memory = true;

  // cuda/hip module
  int num_gpus = 1;
  size_t fb_mem_size = 1024 * 1024 * 1024;
  size_t zc_mem_size = 8 * 1024 * 1024;
  size_t fb_ib_size = 16 * 1024 * 1024;
  size_t zc_ib_size = 32 * 1024 * 1024;
  size_t uvm_mem_size = 64 * 1024 * 1024;
  bool use_dynamic_fb = true;
  size_t dynfb_max_size = 128 * 1024 * 1024;
  unsigned task_streams = 2;
  unsigned d2d_streams = 2;

  // openmp module
  int num_openmp_cpus = 1;
  int num_threads_per_cpu = 2;
  bool openmp_use_numa = false;
  size_t openmp_stack_size = 4 * 1024 * 1024;

  // python module
  int num_python_cpus = 1;
  size_t py_stack_size = 4 * 1024 * 1024;
};

void top_level_task(const void *args, size_t arglen, 
		                const void *userdata, size_t userlen, Processor p)
{
  Runtime rt = Runtime::get_runtime();
  int wrong_config = 0;
  bool ret_value = false;

  // core module
  {
    ModuleConfig *core_config = rt.get_module_config("core");
    assert(core_config != NULL);
    int num_cpu_procs = 0;
    int num_util_procs = 0;
    int num_io_procs = 0;
    size_t sysmem_size = 0;
    size_t regmem_size = 0;
    size_t stack_size = 0;
    bool pin_util_procs = false;
    bool use_ext_sysmem = true;
    ret_value = core_config->get_property<int>("cpu", num_cpu_procs);
    assert(ret_value == true);
    assert(num_cpu_procs == TestConfig::num_cpu_procs);
    ret_value = core_config->get_property<int>("util", num_util_procs);
    assert(ret_value == true);
    assert(num_util_procs == TestConfig::num_util_procs);
    ret_value = core_config->get_property<int>("io", num_io_procs);
    assert(ret_value == true);
    assert(num_io_procs == TestConfig::num_io_procs);
    ret_value = core_config->get_property<size_t>("sysmem", sysmem_size);
    assert(ret_value == true);
    assert(sysmem_size == TestConfig::sysmem_size);
    ret_value = core_config->get_property<size_t>("stack_size", stack_size);
    assert(ret_value == true);
    assert(stack_size == TestConfig::stack_size);
    ret_value = core_config->get_property<bool>("pin_util_procs", pin_util_procs);
    assert(ret_value == true);
    assert(pin_util_procs == TestConfig::pin_util_procs);
    ret_value = core_config->get_property<bool>("use_ext_sysmem", use_ext_sysmem);
    assert(ret_value == true);
    assert(use_ext_sysmem == TestConfig::use_ext_sysmem);
    ret_value = core_config->get_property<size_t>("regmem", regmem_size);
    assert(ret_value == true);
    assert(regmem_size == TestConfig::regmem_size);
    // test wrong property
    ret_value = core_config->get_property<int>("get_error_core", wrong_config);
    assert(ret_value == false);
    log_app.print("cpus %d, utils %d, ios %d, sysmem %zu, regmem %zu", 
      num_cpu_procs, num_util_procs, num_io_procs, sysmem_size, regmem_size);
    // test stack size
#ifdef REALM_ON_WINDOWS
    void *stack_ptr = _alloca(TestConfig::stack_size * 2 / 3);
#else
    void *stack_ptr = alloca(TestConfig::stack_size * 2 / 3);
#endif
    assert(stack_ptr != nullptr);
  }

  // numa module
  // ModuleConfig *numa_config = rt.get_module_config("numa");
  // if (numa_config) {
  //   size_t numa_mem_size = 0;
  //   size_t numa_nocpu_mem_size = 0;
  //   int num_numa_cpus = 0;
  //   bool numa_pin_memory = false;
  //   ret_value = numa_config->get_property("numamem", numa_mem_size);
  //   assert(ret_value == true);
  //   assert(numa_mem_size == TestConfig::numa_mem_size);
  //   ret_value = numa_config->get_property("numa_nocpumem", numa_nocpu_mem_size);
  //   assert(ret_value == true);
  //   assert(numa_nocpu_mem_size == TestConfig::numa_nocpu_mem_size);
  //   ret_value = numa_config->get_property("numacpus", num_numa_cpus);
  //   assert(ret_value == true);
  //   assert(num_numa_cpus == TestConfig::num_numa_cpus);
  //   ret_value = numa_config->get_property("pin_memory", numa_pin_memory);
  //   assert(ret_value == true);
  //   assert(numa_pin_memory == TestConfig::numa_pin_memory);
  //   // test wrong property
  //   ret_value = numa_config->get_property("get_error_numa", wrong_config);
  //   assert(ret_value == false);
  //   log_app.print("numa numamem %zu, numa_nocpumem %zu, numacpsus %d, numa_pin_memory %d", 
  //     numa_mem_size, numa_nocpu_mem_size, num_numa_cpus, numa_pin_memory);
  // } else {
  //   log_app.print("numa is not loaded");
  // }

  // cuda/hip module
  ModuleConfig *cuda_config = rt.get_module_config("cuda");
  ModuleConfig *hip_config = rt.get_module_config("hip");
  ModuleConfig *gpu_config = nullptr;
  if (cuda_config) {
    gpu_config = cuda_config;
  } else if (hip_config) {
    gpu_config = hip_config;
  }
  if (gpu_config) {
    int num_gpus = 0;
    size_t fb_mem_size = 0;
    size_t zc_mem_size = 0;
    size_t fb_ib_size = 0;
    size_t zc_ib_size = 0;
    size_t uvm_mem_size = 0;
    bool use_dynamic_fb = false;
    size_t dynfb_max_size = 0;
    unsigned task_streams = 0;
    unsigned d2d_streams = 0;
    ret_value = gpu_config->get_property("gpu", num_gpus);
    assert(ret_value == true);
    assert(num_gpus == TestConfig::num_gpus);
    ret_value = gpu_config->get_property("fbmem", fb_mem_size);
    assert(ret_value == true);
    assert(fb_mem_size == TestConfig::fb_mem_size);
    ret_value = gpu_config->get_property("zcmem", zc_mem_size);
    assert(ret_value == true);
    assert(zc_mem_size == TestConfig::zc_mem_size);
    ret_value = gpu_config->get_property("ib_fbmem", fb_ib_size);
    assert(ret_value == true);
    assert(fb_ib_size == TestConfig::fb_ib_size);
    ret_value = gpu_config->get_property("ib_zcmem", zc_ib_size);
    assert(ret_value == true);
    assert(zc_ib_size == TestConfig::zc_ib_size);
    // uvm is only available in cuda
    if (cuda_config) {
      ret_value = gpu_config->get_property("uvmem", uvm_mem_size);
      assert(ret_value == true);
      assert(uvm_mem_size == TestConfig::uvm_mem_size);
    }
    ret_value = gpu_config->get_property("use_dynamic_fb", use_dynamic_fb);
    assert(ret_value == true);
    assert(use_dynamic_fb == TestConfig::use_dynamic_fb);
    ret_value = gpu_config->get_property("dynfb_max_size", dynfb_max_size);
    assert(ret_value == true);
    assert(dynfb_max_size == TestConfig::dynfb_max_size);
    ret_value = gpu_config->get_property("task_streams", task_streams);
    assert(ret_value == true);
    assert(task_streams == TestConfig::task_streams);
    ret_value = gpu_config->get_property("d2d_streams", d2d_streams);
    assert(ret_value == true);
    assert(d2d_streams == TestConfig::d2d_streams);
    // test wrong property
    ret_value = gpu_config->get_property("get_error_cuda", wrong_config);
    assert(ret_value == false);
    log_app.print("cuda gpus %d, fbmem %zu, zcmem %zu, fb_ib %zu, zc_ib %zu, uvm %zu, use_dynamic_fb %d, dynfb_max_size %zu, task_streams %u, d2d_streams %u", 
      num_gpus, fb_mem_size, zc_mem_size, fb_ib_size, zc_ib_size, uvm_mem_size, use_dynamic_fb, dynfb_max_size, task_streams, d2d_streams);
  } else {
    log_app.print("cuda/hip is not loaded");
  }

  // openmp module
  ModuleConfig *openmp_config = rt.get_module_config("openmp");
  if (openmp_config) {
    int num_openmp_cpus = 0;
    int num_threads_per_cpu = 0;
    bool openmp_use_numa = false;
    size_t openmp_stack_size = 0;
    ret_value = openmp_config->get_property("ocpu", num_openmp_cpus);
    assert(ret_value == true);
    assert(num_openmp_cpus == TestConfig::num_openmp_cpus);
    ret_value = openmp_config->get_property("othr", num_threads_per_cpu);
    assert(ret_value == true);
    assert(num_threads_per_cpu == TestConfig::num_threads_per_cpu);
    ret_value = openmp_config->get_property("onuma", openmp_use_numa);
    assert(ret_value == true);
    assert(openmp_use_numa == TestConfig::openmp_use_numa);
    ret_value = openmp_config->get_property("ostack", openmp_stack_size);
    assert(ret_value == true);
    assert(openmp_stack_size == TestConfig::openmp_stack_size);
    // test wrong property
    ret_value = openmp_config->get_property("get_error_openmp", wrong_config);
    assert(ret_value == false);
    log_app.print("ocpus %d, othr %d, use_numa %d, stack_size %zu", num_openmp_cpus, num_threads_per_cpu, openmp_use_numa, openmp_stack_size);
  } else {
    log_app.print("openmp is not loaded");
  }

  // python module
  ModuleConfig *python_config = rt.get_module_config("python");
  if (python_config) {
    int num_python_cpus = 0;
    size_t py_stack_size = 0;
    ret_value = python_config->get_property("pyproc", num_python_cpus);
    assert(ret_value == true);
    assert(num_python_cpus == TestConfig::num_python_cpus);
    ret_value = python_config->get_property("pystack", py_stack_size);
    assert(ret_value == true);
    assert(py_stack_size == TestConfig::openmp_stack_size);
    // test wrong property
    ret_value = python_config->get_property("get_error_python", wrong_config);
    assert(ret_value == false);
    log_app.print("python procs %d, stack_size %zu", num_python_cpus, py_stack_size);
  } else {
    log_app.print("python is not loaded");
  }

  if (Network::my_node_id == 0) {
    int num_cpu_procs = 0;
    int num_util_procs = 0;
    int num_io_procs = 0;
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    int num_gpus = 0;
#endif
#ifdef REALM_USE_OPENMP
    int num_openmp_cpus = 0;
#endif
#ifdef REALM_USE_PYTHON
    int num_python_cpus = 0;
#endif
    Machine machine = Machine::get_machine();
    int num_nodes = static_cast<int>(machine.get_address_space_count());
    // iterate all processors within the machine.
    for(Machine::ProcessorQuery::iterator it = Machine::ProcessorQuery(machine).begin(); it; ++it) {
      Processor p = *it;
      Processor::Kind kind = p.kind();
      switch (kind)
      {
        case Processor::LOC_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is CPU.", p.address_space(), p.id);
          num_cpu_procs ++;
          break;
        }
        case Processor::UTIL_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is utility.", p.address_space(), p.id);
          num_util_procs ++;
          break;
        }
        case Processor::IO_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is I/O Proc.", p.address_space(), p.id);
          num_io_procs ++;
          break;
        }
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
        case Processor::TOC_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is GPU.", p.address_space(), p.id);
          num_gpus ++;
          break;
        }
#endif
#ifdef REALM_USE_OPENMP
        case Processor::OMP_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is OMP.", p.address_space(), p.id);
          num_openmp_cpus ++;
          break;
        }
#endif
#ifdef REALM_USE_PYTHON
        case Processor::PY_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is Python.", p.address_space(), p.id);
          num_python_cpus ++;
          break;
        }
#endif
        default:
        {
          break;
        }
      }
    }
    Numa::NumaModule *numa_module = rt.get_module<Numa::NumaModule>("numa");
    if (numa_module) {
      int num_numas = numa_module->numa_cpu_counts.size();
      assert(num_cpu_procs == (TestConfig::num_cpu_procs + TestConfig::num_numa_cpus * num_numas) * num_nodes);
    } else {
      assert(num_cpu_procs == TestConfig::num_cpu_procs * num_nodes);
    }
    assert(num_util_procs == TestConfig::num_util_procs * num_nodes);
    assert(num_io_procs == TestConfig::num_io_procs * num_nodes);
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    assert(num_gpus == TestConfig::num_gpus * num_nodes);
#endif
#ifdef REALM_USE_OPENMP
    assert(num_openmp_cpus == TestConfig::num_openmp_cpus * num_nodes);
#endif
#ifdef REALM_USE_PYTHON
    assert(num_python_cpus == TestConfig::num_python_cpus * num_nodes);
#endif

    size_t sysmem_size = 0;
    size_t regmem_size = 0;
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    size_t fb_mem_size = 0;
    size_t zc_mem_size = 0;
    size_t uvm_mem_size = 0;
    size_t dynfb_max_size = 0;
#endif
    // iterate all memories within the machine.
    for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it; ++it) {
      Memory p = *it;
      Memory::Kind kind = p.kind();
      switch (kind)
      {
        case Memory::SYSTEM_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is SYSTEM MEM, size %zu.", p.address_space(), p.id, p.capacity());
          sysmem_size += p.capacity();
          break;
        }
        case Memory::REGDMA_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is REGDMA MEM, size %zu.", p.address_space(), p.id, p.capacity());
          regmem_size += p.capacity();
          break;
        }
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
        case Memory::GPU_FB_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is FB MEM, size %zu.", p.address_space(), p.id, p.capacity());
          fb_mem_size += p.capacity();
          break;
        }
        case Memory::Z_COPY_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is ZC MEM, size %zu.", p.address_space(), p.id, p.capacity());
          zc_mem_size += p.capacity();
          break;
        }
        case Memory::GPU_MANAGED_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is GPU MANAGED MEM, size %zu.", p.address_space(), p.id, p.capacity());
          uvm_mem_size += p.capacity();
          break;
        }
        case Memory::GPU_DYNAMIC_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is GPU DYNAMIC FB MEM, size %zu.", p.address_space(), p.id, p.capacity());
          dynfb_max_size += p.capacity();
          break;
        }
#endif
        default:
        {
          break;
        }
      }
    }
    assert(sysmem_size == TestConfig::sysmem_size * num_nodes);
    assert(regmem_size == TestConfig::regmem_size * num_nodes);
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    assert(fb_mem_size == TestConfig::fb_mem_size * num_nodes);
    assert(zc_mem_size == TestConfig::zc_mem_size * num_nodes);
#ifdef REALM_USE_CUDA
    // uvm is only available in cuda
    assert(uvm_mem_size == TestConfig::uvm_mem_size * num_nodes);
#endif
    assert(dynfb_max_size == TestConfig::dynfb_max_size * num_nodes);
#endif
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.network_init(&argc, &argv);

  rt.create_configs(argc, argv);

  std::vector<std::string> cmdline;
  cmdline.reserve(argc);
  for(int i = 1; i < argc; i++) {
    cmdline.push_back(argv[i]);
  }

  // rt.parse_command_line(cmdline, true);
  CommandLineParser cp;
  int test_args = 0;
  cp.add_option_int("-test_args", test_args);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);


  if(test_args == 0) {
    // core module
    ModuleConfig* core_config = rt.get_module_config("core");
    assert(core_config != NULL);
    bool ret_value = false;
    {
      int ncores = 0;
      size_t sysmem = 0;
      ret_value = core_config->get_resource("cpu", ncores);
      assert(ret_value == true);
      ret_value = core_config->get_resource("sysmem", sysmem);
      assert(ret_value == true);
      log_app.print("Discover number of CPU cores %d, sysmem %zu", ncores, sysmem);
      ret_value = core_config->set_property<int>("cpu", TestConfig::num_cpu_procs);
      assert(ret_value == true);
      ret_value = core_config->set_property<int>("util", TestConfig::num_util_procs);
      assert(ret_value == true);
      ret_value = core_config->set_property<int>("io", TestConfig::num_io_procs);
      assert(ret_value == true);
      ret_value = core_config->set_property<size_t>("sysmem", TestConfig::sysmem_size);
      assert(ret_value == true);
      ret_value = core_config->set_property<size_t>("stack_size", TestConfig::stack_size);
      assert(ret_value == true);
      ret_value = core_config->set_property<bool>("pin_util_procs", TestConfig::pin_util_procs);
      assert(ret_value == true);
      ret_value = core_config->set_property<bool>("use_ext_sysmem", TestConfig::use_ext_sysmem);
      assert(ret_value == true);
      ret_value = core_config->set_property<size_t>("regmem", TestConfig::regmem_size);
      assert(ret_value == true);
      // test wrong config
      ret_value = core_config->set_property("set_error_core", TestConfig::sysmem_size);
      assert(ret_value == false);
    }

    // numa module
    // ModuleConfig* numa_config = rt.get_module_config("numa");
    // if (numa_config) {
    //   ret_value = numa_config->set_property<size_t>("numamem", TestConfig::numa_mem_size);
    //   assert(ret_value == true);
    //   ret_value = numa_config->set_property<size_t>("numa_nocpumem", TestConfig::numa_nocpu_mem_size);
    //   assert(ret_value == true);
    //   ret_value = numa_config->set_property<int>("numacpus", TestConfig::num_numa_cpus);
    //   assert(ret_value == true);
    //   ret_value = numa_config->set_property<bool>("pin_memory", TestConfig::numa_pin_memory);
    //   assert(ret_value == true);
    //   // test wrong config
    //   ret_value = numa_config->set_property("set_error_numa", TestConfig::numa_mem_size);
    //   assert(ret_value == false);
    // } else {
    //   log_app.print("numa is not loaded");
    // }

    // cuda/hip module
    ModuleConfig* cuda_config = rt.get_module_config("cuda");
    ModuleConfig* hip_config = rt.get_module_config("hip");
    ModuleConfig* gpu_config = nullptr;
    if (cuda_config) {
      gpu_config = cuda_config;
    } else if (hip_config)
    {
      gpu_config = hip_config;
    }
    if (gpu_config) {
      int ngpus = 0;
      size_t fbmem = 0;
      ret_value = gpu_config->get_resource("gpu", ngpus);
      assert(ret_value == true);
      ret_value = gpu_config->get_resource("fbmem", fbmem);
      assert(ret_value == true);
      log_app.print("number of cuda GPUs %d, fbmem size %zu", ngpus, fbmem);
      ret_value = gpu_config->set_property<int>("gpu", TestConfig::num_gpus);
      assert(ret_value == true);
      ret_value = gpu_config->set_property<size_t>("zcmem", TestConfig::zc_mem_size);
      assert(ret_value == true);
      ret_value = gpu_config->set_property<size_t>("fbmem", TestConfig::fb_mem_size);
      assert(ret_value == true);
      ret_value = gpu_config->set_property<size_t>("ib_fbmem", TestConfig::fb_ib_size);
      assert(ret_value == true);
      ret_value = gpu_config->set_property<size_t>("ib_zcmem", TestConfig::zc_ib_size);
      assert(ret_value == true);
      // uvm is only available in cuda
      if (cuda_config) {
        ret_value = gpu_config->set_property<size_t>("uvmem", TestConfig::uvm_mem_size);
        assert(ret_value == true);
      }
      ret_value = gpu_config->set_property<bool>("use_dynamic_fb", TestConfig::use_dynamic_fb);
      assert(ret_value == true);
      ret_value = gpu_config->set_property<size_t>("dynfb_max_size", TestConfig::dynfb_max_size);
      assert(ret_value == true);
      ret_value = gpu_config->set_property<unsigned>("task_streams", TestConfig::task_streams);
      assert(ret_value == true);
      ret_value = gpu_config->set_property<unsigned>("d2d_streams", TestConfig::d2d_streams);
      assert(ret_value == true);
      // test wrong config
      ret_value = gpu_config->set_property("set_error_cuda", TestConfig::fb_mem_size);
      assert(ret_value == false);
    } else {
      log_app.print("cuda/hip is not loaded");
    }

    // openmp module
    ModuleConfig* openmp_config = rt.get_module_config("openmp");
    if (openmp_config) {
      ret_value = openmp_config->set_property<int>("ocpu", TestConfig::num_openmp_cpus);
      assert(ret_value == true);
      ret_value = openmp_config->set_property<int>("othr", TestConfig::num_threads_per_cpu);
      assert(ret_value == true);
      ret_value = openmp_config->set_property<bool>("onuma", TestConfig::openmp_use_numa);
      assert(ret_value == true);
      ret_value = openmp_config->set_property<size_t>("ostack", TestConfig::openmp_stack_size);
      assert(ret_value == true);
      // test wrong config
      ret_value = openmp_config->set_property("set_error_openmp", TestConfig::num_openmp_cpus);
      assert(ret_value == false);
    } else {
      log_app.print("openmp is not loaded");
    }

    // python module
    ModuleConfig* python_config = rt.get_module_config("python");
    if (python_config) {
      ret_value = python_config->set_property<int>("pyproc", TestConfig::num_python_cpus);
      assert(ret_value == true);
      ret_value = python_config->set_property<size_t>("pystack", TestConfig::py_stack_size);
      assert(ret_value == true);
      // test wrong config
      ret_value = python_config->set_property("set_error_openmp", TestConfig::num_openmp_cpus);
      assert(ret_value == false);
    } else {
      log_app.print("python is not loaded");
    }
  }

  rt.parse_command_line(cmdline, true);
  rt.finish_configure();
  rt.start();

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // Collective launch a single task on each CPU processor of each node, that
  // just means this test will be done once on each node
  Event e = rt.collective_spawn_by_kind(Processor::LOC_PROC, TOP_LEVEL_TASK,
                      NULL, 0, true/*one per node*/);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}