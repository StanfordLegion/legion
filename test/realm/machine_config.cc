#include "realm.h"
#include "realm/cmdline.h"
#include "realm/network.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace Realm;

Logger log_app("app");

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

namespace TestConfig {
  int ncpus = 4;
  int nutils = 2;
  int nios = 1;
  size_t sysmem = 16*1024*1024;
  int ngpus = 1;
  size_t zcmem = 32*1024*1024;
  size_t fbmem = 128*1024*1024;
#ifdef REALM_USE_KOKKOS
  int nocpus = 1;
#else
  int nocpus = 2;
#endif
  int nothr = 2;
};

void top_level_task(const void *args, size_t arglen, 
		                const void *userdata, size_t userlen, Processor p)
{
  Runtime rt = Runtime::get_runtime();
  ModuleConfig *core_config = rt.get_module_config("core");
  assert(core_config != NULL);
  int wrong_config = 0;
  bool ret_value = false;
  int ncpus = 0;
  int nutils = 0;
  int nios = 0;
  size_t sysmem = 0;
  ret_value = core_config->get_property<int>("cpu", ncpus);
  assert(ret_value == true);
  ret_value = core_config->get_property<int>("util", nutils);
  assert(ret_value == true);
  ret_value = core_config->get_property<int>("io", nios);
  assert(ret_value == true);
  ret_value = core_config->get_property<size_t>("sysmem", sysmem);
  assert(ret_value == true);
  // test wrong property
  ret_value = core_config->get_property<int>("get_error_core", wrong_config);
  assert(ret_value == false);
  log_app.print("cpus %d, utils %d, ios %d, sysmem %zu", 
    ncpus, nutils, nios, sysmem);

  {
    int ngpus = 0;
    size_t fbmem = 0;
    size_t zcmem = 0;
    ModuleConfig *cuda_config = rt.get_module_config("cuda");
    if (cuda_config) {
      ret_value = cuda_config->get_property("gpu", ngpus);
      assert(ret_value == true);
      ret_value = cuda_config->get_property("fbmem", fbmem);
      assert(ret_value == true);
      ret_value = cuda_config->get_property("zcmem", zcmem);
      assert(ret_value == true);
      // test wrong property
      ret_value = cuda_config->get_property("get_error_cuda", wrong_config);
      assert(ret_value == false);
      log_app.print("cuda gpus %d, fbmem %zu, zcmem %zu", 
        ngpus, fbmem, zcmem);
    } else {
      log_app.print("cuda is not loaded");
    }
  }

  {
    int ngpus = 0;
    size_t fbmem = 0;
    size_t zcmem = 0;
    ModuleConfig *hip_config = rt.get_module_config("hip");
    if (hip_config) {
      ret_value = hip_config->get_property("gpu", ngpus);
      assert(ret_value == true);
      ret_value = hip_config->get_property("fbmem", fbmem);
      assert(ret_value == true);
      ret_value = hip_config->get_property("zcmem", zcmem);
      assert(ret_value == true);
      // test wrong property
      ret_value = hip_config->get_property("get_error_hip", wrong_config);
      assert(ret_value == false);
      log_app.print("hip gpus %d, fbmem %zu, zcmem %zu", 
        ngpus, fbmem, zcmem);
    } else {
      log_app.print("hip is not loaded");
    }
  }

  int ocpus = 0;
  int othr = 0;
  ModuleConfig *openmp_config = rt.get_module_config("openmp");
  if (openmp_config) {
    ret_value = openmp_config->get_property("ocpu", ocpus);
    assert(ret_value == true);
    ret_value = openmp_config->get_property("othr", othr);
    assert(ret_value == true);
    // test wrong property
    ret_value = openmp_config->get_property("get_error_openmp", wrong_config);
    assert(ret_value == false);
    log_app.print("ocpus %d, othr %d", 
      ocpus, othr);
  } else {
    log_app.print("openmp is not loaded");
  }

  if (Network::my_node_id == 0) {
    Machine machine = Machine::get_machine();
    // iterate all processors within the machine.
    for(Machine::ProcessorQuery::iterator it = Machine::ProcessorQuery(machine).begin(); it; ++it) {
      Processor p = *it;
      Processor::Kind kind = p.kind();
      switch (kind)
      {
        case Processor::LOC_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is CPU.", p.address_space(), p.id); 
          break;
        }
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
        case Processor::TOC_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is GPU.", p.address_space(), p.id);
          break;
        }
#endif
#ifdef REALM_USE_OPENMP
        case Processor::OMP_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is OMP.", p.address_space(), p.id);
          break;
        }
#endif
        case Processor::IO_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is I/O Proc.", p.address_space(), p.id);
          break;
        }
        case Processor::UTIL_PROC:
        {
          log_app.print("Rank %u, Processor ID " IDFMT " is utility.", p.address_space(), p.id);
          break;
        }
        default:
        {
          break;
        }
      }
    }

    // iterate all memories within the machine.
    for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it; ++it) {
      Memory p = *it;
      Memory::Kind kind = p.kind();
      switch (kind)
      {
        case Memory::SYSTEM_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is SYSTEM MEM, size %zu.", p.address_space(), p.id, p.capacity());
          break;
        }
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
        case Memory::GPU_FB_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is FB MEM, size %zu.", p.address_space(), p.id, p.capacity());
          break;
        }
        case Memory::Z_COPY_MEM:
        {
          log_app.print("Rank %u, Memory ID " IDFMT " is ZC MEM, size %zu.", p.address_space(), p.id, p.capacity());
          break;
        }
#endif
        default:
        {
          break;
        }
      }
    }
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
  ModuleConfig* core_config = rt.get_module_config("core");
  assert(core_config != NULL);
  bool ret_value = false;
  int ncores = 0;
  size_t sysmem = 0;
  ret_value = core_config->get_resource("cpu", ncores);
  assert(ret_value == true);
  ret_value = core_config->get_resource("sysmem", sysmem);
  assert(ret_value == true);
  log_app.print("number of CPU cores %d, sysmem %zu", ncores, sysmem);
  ret_value = core_config->set_property<int>("cpu", TestConfig::ncpus);
  assert(ret_value == true);
  ret_value = core_config->set_property<int>("util", TestConfig::nutils);
  assert(ret_value == true);
  ret_value = core_config->set_property<int>("io", TestConfig::nios);
  assert(ret_value == true);
  ret_value = core_config->set_property<size_t>("sysmem", TestConfig::sysmem);
  assert(ret_value == true);
  // test wrong config
  ret_value = core_config->set_property("set_error_core", TestConfig::sysmem);
  assert(ret_value == false);

  ModuleConfig* cuda_config = rt.get_module_config("cuda");
  if (cuda_config) {
    int ngpus = 0;
    size_t fbmem = 0;
    ret_value = cuda_config->get_resource("gpu", ngpus);
    assert(ret_value == true);
    ret_value = cuda_config->get_resource("fbmem", fbmem);
    assert(ret_value == true);
    log_app.print("number of cuda GPUs %d, fbmem size %zu", ngpus, fbmem);
    ret_value = cuda_config->set_property<int>("gpu", TestConfig::ngpus);
    assert(ret_value == true);
    ret_value = cuda_config->set_property<size_t>("zcmem", TestConfig::zcmem);
    assert(ret_value == true);
    ret_value = cuda_config->set_property<size_t>("fbmem", TestConfig::fbmem);
    assert(ret_value == true);
    // test wrong config
    ret_value = cuda_config->set_property("set_error_cuda", TestConfig::fbmem);
    assert(ret_value == false);
  } else {
    log_app.print("cuda is not loaded");
  }

  ModuleConfig* hip_config = rt.get_module_config("hip");
  if (hip_config) {
    int ngpus = 0;
    size_t fbmem = 0;
    ret_value = hip_config->get_resource("gpu", ngpus);
    assert(ret_value == true);
    ret_value = hip_config->get_resource("fbmem", fbmem);
    assert(ret_value == true);
    log_app.print("number of hip GPUs %d, fbmem size %zu", ngpus, fbmem);
    ret_value = hip_config->set_property<int>("gpu", TestConfig::ngpus);
    assert(ret_value == true);
    ret_value = hip_config->set_property<size_t>("zcmem", TestConfig::zcmem);
    assert(ret_value == true);
    ret_value = hip_config->set_property<size_t>("fbmem", TestConfig::fbmem);
    assert(ret_value == true);
    // test wrong config
    ret_value = hip_config->set_property("set_error_hip", TestConfig::fbmem);
    assert(ret_value == false);
  } else {
    log_app.print("hip is not loaded");
  }

  ModuleConfig* openmp_config = rt.get_module_config("openmp");
  if (openmp_config) {
    ret_value = openmp_config->set_property("ocpu", TestConfig::nocpus);
    assert(ret_value == true);
    ret_value = openmp_config->set_property("othr", TestConfig::nothr);
    assert(ret_value == true);
    // test wrong config
    ret_value = openmp_config->set_property("set_error_openmp", TestConfig::nothr);
    assert(ret_value == false);
  } else {
    log_app.print("openmp is not loaded");
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