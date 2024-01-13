/* Copyright 2023 Stanford University, Los Alamos National Laboratory
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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
using namespace Legion;

Logger log_app("app");

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

namespace TestConfig {
  int ncpus = 4;
  int nutils = 2;
  int nios = 1;
  size_t sysmem = 16*1024*1024;

  size_t numa_mem_size = 2 * 1024 * 1024;
  size_t numa_nocpu_mem_size = 1 * 1024 * 1024;
  int num_numa_cpus = 1;

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

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  Realm::Runtime rt = Realm::Runtime::get_runtime();
  Realm::ModuleConfig *core_config = rt.get_module_config("core");
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
    Realm::ModuleConfig *numa_config = rt.get_module_config("numa");
    if (numa_config) {
      bool numa_avail = false;
      ret_value = numa_config->get_resource("numa", numa_avail);
      assert(ret_value == true);
      if (numa_avail) {
        size_t numa_mem_size = 0;
        size_t numa_nocpu_mem_size = 0;
        int num_numa_cpus = 0;
        ret_value = numa_config->get_property("numamem", numa_mem_size);
        assert(ret_value == true);
        ret_value = numa_config->get_property("numa_nocpumem", numa_nocpu_mem_size);
        assert(ret_value == true);
        ret_value = numa_config->get_property("numacpus", num_numa_cpus);
        assert(ret_value == true);
        // test wrong property
        ret_value = numa_config->get_property("get_error_numa", wrong_config);
        assert(ret_value == false);
        log_app.print("numa numamem %zu, numa_nocpumem %zu, numacpsus %d", 
          numa_mem_size, numa_nocpu_mem_size, num_numa_cpus);
      } else {
        log_app.warning("numa is not available");
      }
    } else {
      log_app.print("numa is not loaded");
    }
  }

  {
    int ngpus = 0;
    size_t fbmem = 0;
    size_t zcmem = 0;
    Realm::ModuleConfig *cuda_config = rt.get_module_config("cuda");
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
    Realm::ModuleConfig *hip_config = rt.get_module_config("hip");
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
  Realm::ModuleConfig *openmp_config = rt.get_module_config("openmp");
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

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  Runtime::initialize(&argc, &argv, false, false);
  Realm::Runtime rt = Realm::Runtime::get_runtime();

  Realm::ModuleConfig* core_config = rt.get_module_config("core");
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

  Realm::ModuleConfig* numa_config = rt.get_module_config("numa");
  if (numa_config) {
    bool numa_avail = false;
    ret_value = numa_config->get_resource("numa", numa_avail);
    assert(ret_value == true);
    if (numa_avail) {
      ret_value = numa_config->set_property<size_t>("numamem", TestConfig::numa_mem_size);
      assert(ret_value == true);
      ret_value = numa_config->set_property<size_t>("numa_nocpumem", TestConfig::numa_nocpu_mem_size);
      assert(ret_value == true);
      ret_value = numa_config->set_property<int>("numacpus", TestConfig::num_numa_cpus);
      assert(ret_value == true);
      // test wrong config
      ret_value = numa_config->set_property("set_error_numa", TestConfig::numa_mem_size);
      assert(ret_value == false);
    } else {
      log_app.warning("numa is not available");
    }
  } else {
    log_app.print("numa is not loaded");
  }

  Realm::ModuleConfig* cuda_config = rt.get_module_config("cuda");
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

  Realm::ModuleConfig* hip_config = rt.get_module_config("hip");
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

  Realm::ModuleConfig* openmp_config = rt.get_module_config("openmp");
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

  return Runtime::start(argc, argv);
}
