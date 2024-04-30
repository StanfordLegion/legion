/* Copyright 2024 NVIDIA Corporation
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

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/id.h"
#include <assert.h>

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};


void main_task(const void *args, size_t arglen, 
               const void *userdata, size_t userlen, Processor)
{
  Machine machine = Machine::get_machine();
  // iterate all processors within the machine.
  for(Machine::ProcessorQuery::iterator it = Machine::ProcessorQuery(machine).begin(); it; ++it) {
    Processor p = *it;
    ID proc_id = ID(p.id);
    assert (proc_id.is_processor() || proc_id.is_procgroup());
    Processor::Kind kind = p.kind();
    switch (kind)
    {
      case Processor::LOC_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is CPU.", p.address_space(), p.id); 
        break;
      }
      case Processor::TOC_PROC:
      {
        log_app.print("Rank %u, Processor ID " IDFMT " is GPU.", p.address_space(), p.id);
        break;
      }
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
        log_app.print("Rank %u, Processor " IDFMT " is unknown (kind=%d)", p.address_space(), p.id, p.kind());
        break;
      }
    }

    // query the memories that have affinity with the processor
    log_app.print("Has Affinity with:");
    Machine::MemoryQuery mq = Machine::MemoryQuery(machine).has_affinity_to(p, 0, 0);
    for(Machine::MemoryQuery::iterator it = mq.begin(); it; ++it) {
      Memory m = *it;
      ID mem_id = ID(m.id);
      assert (mem_id.is_memory() || mem_id.is_ib_memory());
      size_t memory_size_in_kb = m.capacity() >> 10;
      std::vector<Machine::ProcessorMemoryAffinity> pm_affinity;
      machine.get_proc_mem_affinity(pm_affinity, p, m, true/*local_only*/);
      assert(pm_affinity.size() == 1);
      unsigned bandwidth = pm_affinity[0].bandwidth;
      unsigned latency = pm_affinity[0].latency;
      Memory::Kind kind = m.kind();
      switch (kind)
      {
        case Memory::GLOBAL_MEM:
        {
          log_app.print("\tGASNet Global Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d.", 
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::SYSTEM_MEM:
        {
          log_app.print("\tSystem Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::REGDMA_MEM:
        {
          log_app.print("\tPinned Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::SOCKET_MEM:
        {
          log_app.print("\tSocket Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::Z_COPY_MEM:
        {
          log_app.print("\tZero-Copy Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::GPU_FB_MEM:
        {
          log_app.print("\tGPU Frame Buffer Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::GPU_MANAGED_MEM:
        {
          log_app.print("\tGPU Managed Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::GPU_DYNAMIC_MEM:
        {
          log_app.print("\tGPU Dynamic-allocated Frame Buffer Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::DISK_MEM:
        {
          log_app.print("\tDisk Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::HDF_MEM:
        {
          log_app.print("\tHDF Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::FILE_MEM:
        {
          log_app.print("\tFile Memory ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::LEVEL3_CACHE:
        {
          log_app.print("\tLevel 3 Cache ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::LEVEL2_CACHE:
        {
          log_app.print("\tLevel 2 Cache ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        case Memory::LEVEL1_CACHE:
        {
          log_app.print("\tLevel 1 Cache ID " IDFMT " has %zd KB, bandwidth %u, latency %u, is IB_Mem %d",
                  m.id, memory_size_in_kb, bandwidth, latency, mem_id.is_ib_memory());
          break;
        }
        default:
        {
          log_app.print("\tMemory " IDFMT " is unknown (kind=%d).", it->id, it->kind());
          break;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet()).external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);

  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
