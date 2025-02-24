/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "realm/hardware_topology.h"
#include "realm/logging.h"
#include "realm/realm_c.h"

#include <list>
#include <algorithm>

#ifdef REALM_USE_HWLOC
#include <hwloc.h>
#endif

#ifdef REALM_ON_LINUX
#include <sys/sysinfo.h>
#include <filesystem>
#include <fstream>
#include <regex>
#endif

#ifdef REALM_ON_MACOS
#include <sys/sysctl.h>
#endif

#ifdef REALM_ON_WINDOWS
#include <windows.h>
#endif

namespace Realm {

  Logger log_topo("topology");

  // this function is templated on the map key, since we don't really care
  //  what was used to make the equivalence classes
  template <typename K>
  static void update_core_sharing(std::vector<HardwareTopology::Proc> &all_procs,
                                  const std::map<K, std::set<size_t>> &core_sets,
                                  bool share_alu, bool share_fpu, bool share_ldst)
  {
    for(const auto &[_, cset] : core_sets) {
      if(cset.size() == 1)
        continue; // singleton set - no sharing

      // all pairs dependencies
      for(const size_t proc1 : cset) {
        for(const size_t proc2 : cset) {
          if(proc1 != proc2) {
            HardwareTopology::Proc &p = all_procs[proc1];
            if(share_alu) {
              p.shares_alu.insert(all_procs[proc2].id);
            }
            if(share_fpu) {
              p.shares_fpu.insert(all_procs[proc2].id);
            }
            if(share_ldst) {
              p.shares_ldst.insert(all_procs[proc2].id);
            }
          }
        }
      }
    }
  }

#ifdef REALM_ON_LINUX
  static RealmStatus
  discover_coremap_from_linux(std::vector<HardwareTopology::Proc> &logical_cores,
                              const std::filesystem::path &cpu_dir_path)
  {
    cpu_set_t cset;
    int ret = sched_getaffinity(0, sizeof(cset), &cset);
    if(ret < 0) {
      log_topo.warning() << "failed to get affinity info";
      return REALM_TOPOLOGY_ERROR_NO_AFFINITY;
    }

    logical_cores.clear();

    std::map<std::pair<int, int>, std::set<size_t>> ht_sets;
    std::map<std::string, std::set<size_t>> sibling_sets;

    for(const std::filesystem::directory_entry &cpu_entry :
        std::filesystem::directory_iterator(cpu_dir_path)) {
      if(cpu_entry.is_directory()) {
        std::string name = cpu_entry.path().filename().string();
        if(name.compare(0, 3, "cpu") != 0) {
          continue;
        }

        // Extract CPU ID
        char *endptr;
        long cpu_id = std::strtol(name.c_str() + 3, &endptr, 10);
        if(*endptr != '\0' || cpu_id < 0) {
          continue;
        }

        if(!CPU_ISSET(cpu_id, &cset)) {
          continue;
        }

        HardwareTopology::Proc &proc = logical_cores.emplace_back();
        proc.domain = 0;
        proc.id = static_cast<int>(cpu_id);
        proc.kernel_proc_ids.insert(static_cast<int>(cpu_id));

        int core_id = cpu_id;

        // Determine NUMA node
        std::filesystem::path per_cpu_dir = cpu_entry.path();
        for(const auto &entry : std::filesystem::directory_iterator(per_cpu_dir)) {
          if(entry.is_directory()) {
            std::string node_name = entry.path().filename().string();
            if(node_name.compare(0, 4, "node") == 0) {
              char *endptr;
              long domain_id = std::strtol(node_name.c_str() + 4, &endptr, 10);
              if(*endptr == '\0' && domain_id >= 0) {
                proc.domain = static_cast<int>(domain_id);
                break;
              }
            }
          }
        }

#if !(defined(__arm__) || defined(__aarch64__))
        std::filesystem::path core_id_path = per_cpu_dir / "topology/core_id";
        if(std::filesystem::exists(core_id_path)) {
          std::ifstream core_id_file(core_id_path);
          if(core_id_file) {
            core_id_file >> core_id;
          }
        }
#endif
        size_t proc_idx = logical_cores.size() - 1;
        ht_sets[std::make_pair(proc.domain, core_id)].insert(proc_idx);

        std::filesystem::path sibling_path = per_cpu_dir / "topology/thread_siblings";
        if(std::filesystem::exists(sibling_path)) {
          std::ifstream sibling_file(sibling_path);
          if(sibling_file) {
            std::string sibling_mask;
            std::getline(sibling_file, sibling_mask);
            sibling_sets[sibling_mask].insert(proc_idx);
          }
        }
      }
    }

    update_core_sharing(logical_cores, ht_sets, true /*alu*/, true /*fpu*/,
                        true /*ldst*/);
    update_core_sharing(logical_cores, sibling_sets, false /*!alu*/, true /*fpu*/,
                        false /*!ldst*/);
    return REALM_SUCCESS;
  }

  static RealmStatus
  discover_memory_from_linux(std::vector<HardwareTopology::MemoryInfo> &memories,
                             size_t &host_memory,
                             const std::filesystem::path &numa_node_path)
  {
    memories.clear();

    // Detect NUMA memory
    for(const std::filesystem::directory_entry &entry :
        std::filesystem::directory_iterator(numa_node_path)) {
      if(entry.is_directory()) {
        std::string dirname = entry.path().filename().string();

        // Check if the directory name starts with "node" and extract the NUMA node ID
        if(dirname.compare(0, 4, "node") == 0) {
          char *endptr;
          long numa_node = std::strtol(dirname.c_str() + 4, &endptr, 10);
          if(*endptr != '\0' || numa_node < 0) {
            continue;
          }

          std::string meminfo_path = entry.path().string() + "/meminfo";

          // Read memory size from meminfo
          std::ifstream meminfo_file(meminfo_path);
          if(meminfo_file.is_open()) {
            std::string line;
            while(std::getline(meminfo_file, line)) {
              if(line.find("MemTotal:") != std::string::npos) {
                size_t memory_size_kB = 0;
                int mem_node_id = -1;
                if(sscanf(line.c_str(), "Node %d MemTotal: %zu kB", &mem_node_id,
                          &memory_size_kB) == 2) {
                  // Ignore NUMA nodes with 0 memory
                  if(memory_size_kB > 0) {
                    memories.emplace_back(HardwareTopology::MemoryInfo{
                        memory_size_kB << 10, static_cast<int>(numa_node)});
                    assert(static_cast<int>(numa_node) == mem_node_id);
                  }
                }
                break;
              }
            }
            meminfo_file.close();
          }
        }
      }
    }

    // System memory
    struct sysinfo memInfo;
    sysinfo(&memInfo);

    // Cross-check with NUMA memory
    // On 32-bit machines, memInfo may be less than the actual memory if it exceeds 32-bit
    // bounds
#if defined(__x86_64__) || defined(_M_X64)
    size_t system_mem = 0;
    for(const HardwareTopology::MemoryInfo &memory : memories) {
      system_mem += memory.bytes;
    }
    assert(system_mem == memInfo.totalram);
#endif

    host_memory = memInfo.totalram;

    return REALM_SUCCESS;
  }

  static RealmStatus
  create_topology_from_linux(std::vector<HardwareTopology::Proc> &logical_cores,
                             std::vector<HardwareTopology::MemoryInfo> &memories,
                             size_t &host_memory)
  {
    const std::filesystem::path cpu_dir_path = "/sys/devices/system/cpu";
    if(!std::filesystem::exists(cpu_dir_path) ||
       !std::filesystem::is_directory(cpu_dir_path)) {
      log_topo.warning() << "can't open /sys/devices/system/cpu";
      return REALM_TOPOLOGY_ERROR_LINUX_NO_CPU_DIR;
    }

    const std::filesystem::path numa_node_path = "/sys/devices/system/node/";
    if(!std::filesystem::exists(numa_node_path) ||
       !std::filesystem::is_directory(numa_node_path)) {
      log_topo.warning() << "can't open /sys/devices/system/node/";
      return REALM_TOPOLOGY_ERROR_LINUX_NO_NUMA_DIR;
    }

    RealmStatus status = discover_coremap_from_linux(logical_cores, cpu_dir_path);
    if(status == REALM_SUCCESS) {
      status = discover_memory_from_linux(memories, host_memory, numa_node_path);
    }
    return status;
  }
#endif

#ifdef REALM_ON_MACOS
  static RealmStatus
  create_topology_from_macos(std::vector<HardwareTopology::Proc> &logical_cores,
                             std::vector<HardwareTopology::MemoryInfo> &memories,
                             size_t &host_memory)
  {
    // We are not able to query the affinity on Mac, so ignore it

    logical_cores.clear();
    memories.clear();

    // assume mac only have one numa
    // system memory
    size_t sysmem_size = 0;
    size_t buflen = sizeof(size_t);
    sysctlbyname("hw.memsize", &sysmem_size, &buflen, NULL, 0);
    memories.emplace_back(HardwareTopology::MemoryInfo{sysmem_size, 0});
    host_memory = sysmem_size;
    // phyical cores
    buflen = sizeof(int);
    int num_physical_cores = 0;
    sysctlbyname("hw.physicalcpu", &num_physical_cores, &buflen, NULL, 0);
    // logical cores
    int num_logical_cores = 0;
    sysctlbyname("hw.logicalcpu", &num_logical_cores, &buflen, NULL, 0);
    // we can only handle the case that there is no hyper threads
    assert(num_physical_cores == num_logical_cores);
    for(int i = 0; i < num_physical_cores; i++) {
      HardwareTopology::Proc &proc = logical_cores.emplace_back();
      // assume mac only have one numa
      proc.domain = 0;
      proc.id = i;
      proc.kernel_proc_ids.insert(i);
    }
    return REALM_SUCCESS;
  }
#endif

#ifdef REALM_ON_WINDOWS
  static RealmStatus
  create_topology_from_windows(std::vector<HardwareTopology::Proc> &logical_cores,
                               std::vector<HardwareTopology::MemoryInfo> &memories,
                               size_t &host_memory)
  {
    DWORD_PTR process_mask, system_mask;
    GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);
    if(process_mask == 0) {
      log_topo.warning() << "process affinity mask is empty? (system = " << system_mask
                         << ")";
      return REALM_TOPOLOGY_ERROR_NO_AFFINITY;
    }
    log_topo.debug() << "affinity_mask = " << process_mask << " system=" << system_mask;

    logical_cores.clear();
    memories.clear();

    // ------ Step 1: core map
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION proc_info = NULL;
    DWORD proc_info_size = 0;
    DWORD rc;
    rc = GetLogicalProcessorInformation(proc_info, &proc_info_size);
    if((rc == TRUE) || (GetLastError() != ERROR_INSUFFICIENT_BUFFER) ||
       (proc_info_size == 0)) {
      log_topo.warning() << "unable to query processor info size";
      return REALM_TOPOLOGY_ERROR_WIN32_NO_PROC_INFO;
    }
    proc_info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(proc_info_size);
    assert(proc_info != 0);
    rc = GetLogicalProcessorInformation(proc_info, &proc_info_size);
    assert(rc == TRUE);

    // populate _all_procs map
    for(int i = 0; (i < sizeof(DWORD_PTR) * 8) && ((DWORD_PTR(1) << i) <= process_mask);
        i++) {
      if((process_mask & (DWORD_PTR(1) << i)) != 0) {
        HardwareTopology::Proc &proc = logical_cores.emplace_back();
        proc.id = i;
        proc.domain = -1; // fill in below
        proc.kernel_proc_ids.insert(i);
      }
    }

    bool numa_enabled = FALSE;
    size_t num_infos = proc_info_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for(size_t i = 0; i < num_infos; i++) {
      DWORD_PTR eff_mask = process_mask & proc_info[i].ProcessorMask;
      if(eff_mask == 0)
        continue;

      switch(proc_info[i].Relationship) {
      case RelationNumaNode:
      {
        log_topo.debug() << "info[" << i << "]: eff_mask=" << proc_info[i].ProcessorMask
                         << " numa node=" << proc_info[i].NumaNode.NodeNumber;
        ULONGLONG availableMemory = 0;
        if(GetNumaAvailableMemoryNode(proc_info[i].NumaNode.NodeNumber,
                                      &availableMemory)) {
          numa_enabled = TRUE;
          memories.emplace_back(HardwareTopology::MemoryInfo{
              availableMemory, (int)(proc_info[i].NumaNode.NodeNumber)});
        }
        for(int i = 0; (i < sizeof(DWORD_PTR) * 8) && ((DWORD_PTR(1) << i) <= eff_mask);
            i++)
          if((eff_mask & (DWORD_PTR(1) << i)) != 0) {
            HardwareTopology::Proc &p = logical_cores[i];
            p.domain = proc_info[i].NumaNode.NodeNumber;
          }
        break;
      }

      case RelationProcessorCore:
      {
        log_topo.debug() << "info[" << i << "]: eff_mask=" << proc_info[i].ProcessorMask
                         << " hyperthreads";

        // these are hyperthreads - do we care?
        for(int i = 0; (i < sizeof(DWORD_PTR) * 8) && ((DWORD_PTR(1) << i) <= eff_mask);
            i++) {
          if((eff_mask & (DWORD_PTR(1) << i)) != 0) {
            HardwareTopology::Proc &p1 = logical_cores[i];
            for(int j = i + 1;
                (i < sizeof(DWORD_PTR) * 8) && ((DWORD_PTR(1) << j) <= eff_mask); j++)
              if((eff_mask & (DWORD_PTR(1) << j)) != 0) {
                HardwareTopology::Proc &p2 = logical_cores[j];
                p1.shares_alu.insert(p2.id);
                p1.shares_fpu.insert(p2.id);
                p1.shares_ldst.insert(p2.id);

                p2.shares_alu.insert(p1.id);
                p2.shares_fpu.insert(p1.id);
                p2.shares_ldst.insert(p1.id);
              }
          }
        }
        break;
      }

      default:
      {
        log_topo.debug() << "info[" << i << "]: eff_mask=" << proc_info[i].ProcessorMask
                         << " rel=" << proc_info[i].Relationship;
        break;
      }
      }
    }

    free(proc_info);

    // ------ Step 2: system memory
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    // cross check with numa memory
    if(numa_enabled) {
      size_t system_mem = 0;
      for(const HardwareTopology::MemoryInfo &memory : memories) {
        system_mem += memory.bytes;
      }
      assert(system_mem == memInfo.ullTotalPageFile);
    } else {
      memories.emplace_back(HardwareTopology::MemoryInfo{memInfo.ullTotalPageFile, 0});
    }
    host_memory = static_cast<size_t>(memInfo.ullTotalPageFile);

    return REALM_SUCCESS;
  }
#endif

#ifdef REALM_USE_HWLOC

#if 0 // This is only for debugging
  static void print_topology(hwloc_topology_t topology)
  {
    char string[128];
    for(int depth = 0; depth < hwloc_topology_get_depth(topology); ++depth) {
      log_topo.print("*** Objects at level %d", depth);
      for(unsigned i = 0; i < hwloc_get_nbobjs_by_depth(topology, depth); i++) {
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, depth, i);
        hwloc_obj_type_snprintf(string, sizeof(string), obj, 0);
        log_topo.print("Index %u: %s, %p, os_index %d, logical_index %d", i, string, obj,
                       obj->os_index, obj->logical_index);
      }
    }
    log_topo.print("*** Objects at NUMA depth");
    for(unsigned i = 0;
        i < hwloc_get_nbobjs_by_depth(topology, HWLOC_TYPE_DEPTH_NUMANODE); i++) {
      hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, HWLOC_TYPE_DEPTH_NUMANODE, i);
      hwloc_obj_type_snprintf(string, sizeof(string), obj, 0);
      log_topo.print("Index %u: %s, %p, os_index %d, logical_index %d", i, string, obj,
                     obj->os_index, obj->logical_index);
    }
  }
#endif

  static RealmStatus
  discover_coremap_from_hwloc(hwloc_topology_t &topo,
                              std::vector<HardwareTopology::Proc> &logical_cores)
  {
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    if(hwloc_get_cpubind(topo, cpuset, HWLOC_CPUBIND_PROCESS) != 0) {
      log_topo.warning() << "Failed to get CPU binding info";
      hwloc_bitmap_free(cpuset);
      return REALM_TOPOLOGY_ERROR_NO_AFFINITY;
    }

    // Get the depth of PU (Processing Unit, i.e., logical core)
    int pu_depth = hwloc_get_type_depth(topo, HWLOC_OBJ_PU);
    if(pu_depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
      log_topo.warning() << "Unable to get PU depth";
      hwloc_topology_destroy(topo);
      return REALM_TOPOLOGY_ERROR_HWLOC_TYPE_DEPTH_UNKNOWN;
    }

    logical_cores.clear();
    // hyperthreading sets are cores with the same node ID and physical core ID
    //  they share ALU, FPU, and LDST units
    std::map<std::pair<int, int>, std::set<size_t>> ht_sets;

    // bulldozer specific sets
    std::map<int, std::set<size_t>> bd_sets;

    for(unsigned i = 0; i < hwloc_get_nbobjs_by_depth(topo, pu_depth); i++) {
      hwloc_obj_t pu = hwloc_get_obj_by_depth(topo, pu_depth, i);
      // Not an accessible cpu
      unsigned cpu_id = pu->os_index;
      if(!hwloc_bitmap_isset(cpuset, cpu_id)) {
        continue;
      }
      hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu);
      if(core) {
        HardwareTopology::Proc &proc = logical_cores.emplace_back();
        proc.id = cpu_id;
        int numa_node = hwloc_bitmap_first(pu->nodeset);
        proc.domain = numa_node;
        proc.kernel_proc_ids.insert(cpu_id);
        int core_id = core->os_index;
        size_t proc_idx = logical_cores.size() - 1;
        ht_sets[std::make_pair(numa_node, core_id)].insert(proc_idx);

#ifdef REALM_ON_LINUX
        // add bulldozer sets
        std::set<int> sibling_ids;
        for(unsigned i = 0; i < core->arity; ++i) {
          if(cpu_id != core->children[i]->os_index) {
            sibling_ids.insert(core->children[i]->os_index);
          }
        }
        if(!sibling_ids.empty()) {
          bd_sets[cpu_id].insert(proc_idx);
          for(std::set<int>::const_iterator it = sibling_ids.begin();
              it != sibling_ids.end(); ++it) {
            bd_sets[*it].insert(proc_idx);
          }
        }
#endif
      }
    }

    hwloc_bitmap_free(cpuset);

    update_core_sharing(logical_cores, ht_sets, true /*alu*/, true /*fpu*/,
                        true /*ldst*/);
    update_core_sharing(logical_cores, bd_sets, false /*!alu*/, true /*fpu*/,
                        false /*!ldst*/);

    return REALM_SUCCESS;
  }

  static RealmStatus
  discover_memory_from_hwloc(hwloc_topology_t &topo,
                             std::vector<HardwareTopology::MemoryInfo> &memories,
                             size_t &host_memory)
  {
    memories.clear();

    // numa memory
    unsigned num_nodes = hwloc_get_nbobjs_by_depth(topo, HWLOC_TYPE_DEPTH_NUMANODE);
    for(unsigned i = 0; i < num_nodes; ++i) {
      hwloc_obj_t numa_node = hwloc_get_obj_by_depth(topo, HWLOC_TYPE_DEPTH_NUMANODE, i);
      if(numa_node) {
        memories.emplace_back(
            HardwareTopology::MemoryInfo{numa_node->total_memory, static_cast<int>(i)});
      }
    }

    //  system memory
    hwloc_obj_t root_obj = hwloc_get_root_obj(topo);
    assert(root_obj != nullptr);
    // cross check with numa memory
    size_t system_mem = 0;
    for(const HardwareTopology::MemoryInfo &memory : memories) {
      system_mem += memory.bytes;
    }
    assert(system_mem == root_obj->total_memory);
    host_memory = root_obj->total_memory;

    return REALM_SUCCESS;
  }

  static RealmStatus
  create_topology_from_hwloc(std::vector<HardwareTopology::Proc> &logical_cores,
                             std::vector<HardwareTopology::MemoryInfo> &memories,
                             size_t &host_memory)
  {
    hwloc_topology_t topo;
    if(hwloc_topology_init(&topo) != 0) {
      log_topo.warning() << "Failed to initialize hwloc topology";
      return REALM_TOPOLOGY_ERROR_HWLOC_INIT_FAILED;
    }
    if(hwloc_topology_load(topo) != 0) {
      log_topo.warning() << "Failed to load hwloc topology";
      hwloc_topology_destroy(topo);
      return REALM_TOPOLOGY_ERROR_HWLOC_LOAD_TOPO_FAILED;
    }

    RealmStatus status = discover_coremap_from_hwloc(topo, logical_cores);
    if(status == REALM_SUCCESS) {
      status = discover_memory_from_hwloc(topo, memories, host_memory);
    }

    hwloc_topology_destroy(topo);

    return status;
  }
#endif

  static void create_synthetic_topology(
      unsigned num_domains, unsigned cores_per_domain, unsigned num_hyperthreads,
      unsigned fp_cluster_size, size_t memory_per_domain,
      std::vector<HardwareTopology::Proc> &logical_cores,
      std::vector<HardwareTopology::MemoryInfo> &memories, size_t &host_memory)
  {
    // processor ids will just be monotonically increasing
    int next_id = 0;

    for(unsigned domain = 0; domain < num_domains; domain++) {
      for(unsigned core = 0; core < cores_per_domain; core++) {
        std::set<size_t> fp_procs;

        for(unsigned fp = 0; fp < fp_cluster_size; fp++) {
          std::set<size_t> ht_procs;

          for(unsigned hyterthread = 0; hyterthread < num_hyperthreads; hyterthread++) {
            int id = next_id++;

            HardwareTopology::Proc &proc = logical_cores.emplace_back();

            proc.id = id;
            proc.domain = domain;
            // kernel proc id list is empty - this is synthetic

            memories.emplace_back(HardwareTopology::MemoryInfo{memory_per_domain,
                                                               static_cast<int>(domain)});

            ht_procs.insert(id);
            fp_procs.insert(id);
          }

          // ALU and LD/ST shared with all other hyperthreads
          for(std::set<size_t>::iterator it1 = ht_procs.begin(); it1 != ht_procs.end();
              it1++) {
            for(std::set<size_t>::iterator it2 = ht_procs.begin(); it2 != ht_procs.end();
                it2++) {
              if(it1 != it2) {
                logical_cores[*it1].shares_alu.insert(logical_cores[*it2].id);
                logical_cores[*it1].shares_ldst.insert(logical_cores[*it2].id);
              }
            }
          }
        }

        // FPU shared with all other procs in cluster
        for(std::set<size_t>::iterator it1 = fp_procs.begin(); it1 != fp_procs.end();
            it1++) {
          for(std::set<size_t>::iterator it2 = fp_procs.begin(); it2 != fp_procs.end();
              it2++) {
            if(it1 != it2) {
              logical_cores[*it1].shares_fpu.insert(logical_cores[*it2].id);
            }
          }
        }
      }
    }

    host_memory = memory_per_domain * num_domains;
  }

  static RealmStatus parse_topology_from_env(int &num_domains, int &cores_per_domain,
                                             int &num_hyperthreads, int &fp_cluster_size,
                                             size_t &memory_per_domain)
  {
    if(getenv("REALM_SYNTHETIC_CORE_MAP")) {
      const char *p = getenv("REALM_SYNTHETIC_CORE_MAP");
      while(true) {
        if(!(p[0] && (p[1] == '=') && isdigit(p[2])))
          break;

        const char *p2;
        int x = strtol(p + 2, (char **)&p2, 10);
        if(x == 0) {
          p += 2;
          break;
        } // zero of anything is bad
        if(p[0] == 'd')
          num_domains = x;
        else if(p[0] == 'c')
          cores_per_domain = x;
        else if(p[0] == 'h')
          num_hyperthreads = x;
        else if(p[0] == 'f')
          fp_cluster_size = x;
        else if(p[0] == 'm')
          memory_per_domain = x;
        else
          break;
        p = p2;

        // now we want a comma (to continue) or end-of-string
        if(*p != ',')
          break;
        p++;
      }
      // if parsing reached the end of string, we're good
      if(*p == 0) {
        return REALM_SUCCESS;
      } else {
        const char *orig = getenv("REALM_SYNTHETIC_CORE_MAP");
        // We can not use log because it is printed before runtime initialization
        fprintf(stderr, "Error parsing REALM_SYNTHETIC_CORE_MAP: '%.*s(^)%s'\n",
                (int)(p - orig), orig, p);
        abort();
      }
    }
    return REALM_TOPOLOGY_ERROR_ENV_LOAD_FAILED;
  }

  // ------------ HardwareTopology -----------------

  /* static */ HardwareTopology HardwareTopology::create_topology(void)
  {
    RealmStatus created = REALM_ERROR;

    std::vector<HardwareTopology::Proc> logical_cores;
    std::vector<HardwareTopology::MemoryInfo> memories;
    size_t host_memory = 0;

    // we'll try a number of different strategies to discover the local cores:
    // 1) a user-defined synthetic map, if REALM_SYNTHETIC_CORE_MAP is set
    {
      int num_domains = 1;
      int cores_per_domain = 1;
      int num_hyperthreads = 1;
      int fp_cluster_size = 1;
      size_t memsize_per_numa = 128 << 20;
      created = parse_topology_from_env(num_domains, cores_per_domain, num_hyperthreads,
                                        fp_cluster_size, memsize_per_numa);
      if(created == REALM_SUCCESS) {
        create_synthetic_topology(num_domains, cores_per_domain, num_hyperthreads,
                                  fp_cluster_size, memsize_per_numa, logical_cores,
                                  memories, host_memory);
      }
    }

    // 2) extracted from hwloc information
#ifdef REALM_USE_HWLOC
    if(created != REALM_SUCCESS) {
      created = create_topology_from_hwloc(logical_cores, memories, host_memory);
    }
#endif

    // 3) extracted from Linux's /sys
#ifdef REALM_ON_LINUX
    if(created != REALM_SUCCESS) {
      created = create_topology_from_linux(logical_cores, memories, host_memory);
    }
#endif

    // 4) windows has an API for this
#ifdef REALM_ON_WINDOWS
    if(created != REALM_SUCCESS) {
      created = create_topology_from_windows(logical_cores, memories, host_memory);
    }
#endif

    // 5) extracted from mac os
#ifdef REALM_ON_MACOS
    if(created != REALM_SUCCESS) {
      created = create_topology_from_macos(logical_cores, memories, host_memory);
    }
#endif

    // 5) as a final fallback a single-core synthetic map
    if(created != REALM_SUCCESS) {
      create_synthetic_topology(1, 1, 1, 1, 32 << 20, logical_cores, memories,
                                host_memory);
    }

    return HardwareTopology(logical_cores, memories, host_memory);
  }

  HardwareTopology::HardwareTopology(void) {}

  HardwareTopology::HardwareTopology(const std::vector<Proc> &logical_cores,
                                     const std::vector<MemoryInfo> &memories,
                                     const size_t host_memory)
  {
    assert(logical_cores.size() > 0);
    assert(memories.size() > 0);
    // ------ Step 1: construct numa memory
    for(const MemoryInfo &memory : memories) {
      by_domain[memory.domain].memory_size = memory.bytes;
    }

    // ------ Step 2: construct proc
    // Copy all Procs
    for(const Proc &proc : logical_cores) {
      Proc new_proc = proc;
      all_procs[new_proc.id] = new_proc;
      by_domain[new_proc.domain].logical_cores.insert(new_proc.id);
    }

    // ------ Step 4: num_logical_cores_per_physical_core
    for(const auto &[_, proc] : all_procs) {
      if(num_logical_cores_per_physical_core == 0) {
        num_logical_cores_per_physical_core = proc.shares_fpu.size() + 1;
      } else {
        assert(num_logical_cores_per_physical_core == proc.shares_fpu.size() + 1);
      }
    }

    // ------ Step 5: system memory size
    sys_memory_size = host_memory;
  }

  HardwareTopology::~HardwareTopology(void)
  {
    // now clear out the two maps
    all_procs.clear();
    by_domain.clear();
  }

  void HardwareTopology::remove_hyperthreads(void)
  {
    for(auto &[_, proc] : all_procs) {
      proc.shares_alu.clear();
      proc.shares_fpu.clear();
      proc.shares_ldst.clear();
    }
  }

  size_t HardwareTopology::system_memory(void) const { return sys_memory_size; }

  unsigned HardwareTopology::num_logical_cores(void) const { return all_procs.size(); }

  unsigned HardwareTopology::num_physical_cores(void) const
  {
    return all_procs.size() / num_logical_cores_per_physical_core;
  }

  unsigned HardwareTopology::num_numa_domains(void) const { return by_domain.size(); }

  void HardwareTopology::get_logical_cores(std::set<ProcID> &cores) const
  {
    for(const auto &[core_id, _] : all_procs) {
      cores.insert(core_id);
    }
  }

  bool HardwareTopology::has_processor(ProcID proc_id) const
  {
    ProcMap::const_iterator it = all_procs.find(proc_id);
    if(it == all_procs.cend()) {
      return false;
    } else {
      return true;
    }
  }

  bool HardwareTopology::numa_domain_has_processors(int domain_id) const
  {
    DomainMap::const_iterator it = by_domain.find(domain_id);
    if(it == by_domain.cend()) {
      return false;
    } else {
      return !it->second.logical_cores.empty();
    }
  }

  const std::set<HardwareTopology::ProcID> &
  HardwareTopology::get_processors_by_domain(int domain_id) const
  {
    const DomainInfo &domain = by_domain.at(domain_id);
    return domain.logical_cores;
  }

  const std::set<HardwareTopology::ProcID> &
  HardwareTopology::get_processors_share_alu(HardwareTopology::ProcID proc_id) const
  {
    const HardwareTopology::Proc &p = all_procs.at(proc_id);
    return p.shares_alu;
  }

  const std::set<HardwareTopology::ProcID> &
  HardwareTopology::get_processors_share_fpu(HardwareTopology::ProcID proc_id) const
  {
    const HardwareTopology::Proc &p = all_procs.at(proc_id);
    return p.shares_fpu;
  }

  const std::set<HardwareTopology::ProcID> &
  HardwareTopology::get_processors_share_ldst(HardwareTopology::ProcID proc_id) const
  {
    const HardwareTopology::Proc &p = all_procs.at(proc_id);
    return p.shares_ldst;
  }

  const std::set<HardwareTopology::ProcID> &
  HardwareTopology::get_kernel_processor_ids(HardwareTopology::ProcID proc_id) const
  {
    const HardwareTopology::Proc &p = all_procs.at(proc_id);
    return p.kernel_proc_ids;
  }

  std::vector<HardwareTopology::ProcID>
  HardwareTopology::distribute_processors_across_domains(void) const
  {
    std::vector<ProcID> pm;
    std::list<std::pair<const std::set<ProcID> *, std::set<ProcID>::const_iterator>> rr;

    for(DomainMap::const_iterator domain_it = by_domain.begin();
        domain_it != by_domain.end(); ++domain_it) {
      if(!domain_it->second.logical_cores.empty()) {
        rr.push_back(std::make_pair(&(domain_it->second.logical_cores),
                                    domain_it->second.logical_cores.begin()));
      }
    }

    while(!rr.empty()) {
      std::pair<const std::set<int> *, std::set<int>::const_iterator> x = rr.front();
      rr.pop_front();
      pm.push_back(*x.second);
      ++x.second;
      if(x.second != x.first->end()) {
        rr.push_back(x);
      }
    }
    assert(pm.size() == all_procs.size());
    return pm;
  }

  static void show_share_set(std::ostream &os, const char *name,
                             const std::set<HardwareTopology::ProcID> &sset)
  {
    if(sset.empty())
      return;

    os << ' ' << name << "=<";
    std::set<HardwareTopology::ProcID>::const_iterator it = sset.begin();
    while(true) {
      os << (*it);
      if(++it == sset.end())
        break;
      os << ',';
    }
    os << ">";
  }

  /*friend*/ std::ostream &operator<<(std::ostream &os, const HardwareTopology &topo)
  {
    os << "Topology {" << std::endl;
    for(const auto &[numa_node, numa_domain] : topo.by_domain) {
      os << "  domain " << numa_node << ", memory size:" << numa_domain.memory_size
         << " {" << std::endl;
      for(const HardwareTopology::ProcID &proc_id : numa_domain.logical_cores) {
        os << "    core " << proc_id << " {";
        const HardwareTopology::Proc &p = topo.all_procs.at(proc_id);
        if(!p.kernel_proc_ids.empty()) {
          os << " ids=<";
          std::set<int>::const_iterator it3 = p.kernel_proc_ids.begin();
          while(true) {
            os << *it3;
            if(++it3 == p.kernel_proc_ids.end())
              break;
            os << ',';
          }
          os << ">";
        }

        show_share_set(os, "alu", p.shares_alu);
        show_share_set(os, "fpu", p.shares_fpu);
        show_share_set(os, "ldst", p.shares_ldst);

        os << " }" << std::endl;
      }
      os << "  }" << std::endl;
    }
    os << "}";
    return os;
  }

}; /* namespace Realm*/
