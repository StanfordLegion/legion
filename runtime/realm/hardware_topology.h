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

#ifndef REALM_TOPOLOGY_H
#define REALM_TOPOLOGY_H

#include "realm/realm_config.h"
#include <stddef.h>
#include <memory>
#include <set>
#include <map>
#include <vector>

namespace Realm {

  /**
   * @class HardwareTopology
   * @brief Represents the topology of the host processor cores and memory.
   *
   * This class provides an abstraction for enumerating processor cores and the ways in
   * which they share components (e.g., ALU, FPU, LD/ST). It also describes NUMA domains
   * and memory information for optimal resource allocation.
   */
  class HardwareTopology {

  public:
    /**
     * @typedef ProcID
     * @brief A unique integer identifier for a processor.
     */
    typedef int ProcID;

    /**
     * @struct Proc
     * @brief Represents a processor core with sharing relationships.
     */
    struct Proc {
      ProcID id;                        ///< Unique processor identifier.
      int domain;                       ///< NUMA domain the processor belongs to.
      std::set<ProcID> kernel_proc_ids; ///< Set of kernel processor IDs (may be empty).
      std::set<ProcID>
          shares_alu; ///< Set of processors sharing an ALU with this processor.
      std::set<ProcID>
          shares_fpu; ///< Set of processors sharing an FPU with this processor.
      std::set<ProcID>
          shares_ldst; ///< Set of processors sharing LD/ST paths with this processor.
    };

    /**
     * @struct MemoryInfo
     * @brief Represents memory information for a NUMA domain.
     */
    struct MemoryInfo {
      size_t bytes; ///< Size of memory in bytes.
      int domain;   ///< NUMA domain associated with the memory.
    };

    /**
     * @brief Creates a topology instance based on the system's hardware.
     * @return A unique pointer to the created HardwareTopology instance.
     */
    static HardwareTopology create_topology(void);

    /**
     * @brief Constructs an empty HardwareTopology instance.
     */
    HardwareTopology(void);

    /**
     * @brief Constructs a HardwareTopology instance.
     * @param logical_cores Vector of logical cores in the system.
     * @param memories Vector of memory information for the system.
     * @param host_memory Total host memory size.
     */
    HardwareTopology(const std::vector<Proc> &logical_cores,
                     const std::vector<MemoryInfo> &memories, const size_t host_memory);

    /**
     * @brief Destructor for HardwareTopology.
     */
    ~HardwareTopology(void);

    HardwareTopology &operator=(const HardwareTopology &rhs) = default;

    /**
     * @brief Removes hyperthreads from the topology.
     */
    void remove_hyperthreads(void);

    /**
     * @brief Returns the size of system memory in bytes.
     * @return Size of system memory.
     */
    size_t system_memory(void) const;

    /**
     * @brief Returns the number of logical cores.
     * @return Number of logical cores.
     */
    unsigned num_logical_cores(void) const;

    /**
     * @brief Returns the number of physical cores.
     * @return Number of physical cores.
     */
    unsigned num_physical_cores(void) const;

    /**
     * @brief Returns the number of NUMA domains.
     * @return Number of NUMA domains.
     */
    unsigned num_numa_domains(void) const;

    /**
     * @brief Retrieves the set of logical core IDs.
     * @param cores Output parameter to store the set of logical core IDs.
     */
    void get_logical_cores(std::set<ProcID> &cores) const;

    /**
     * @brief Checks if a processor exists in the topology.
     * @param proc_id Processor ID to check.
     * @return True if the processor exists, false otherwise.
     */
    bool has_processor(ProcID proc_id) const;

    /**
     * @brief Checks if a NUMA domain exists in the topology and has processors.
     * @param domain_id NUMA domain ID to check.
     * @return True if the domain exists and has processors, false otherwise.
     */
    bool numa_domain_has_processors(int domain_id) const;

    /**
     * @brief Gets the processors in a specific NUMA domain.
     * @param domain_id NUMA domain ID.
     * @return A set of processor IDs in the domain.
     */
    const std::set<ProcID> &get_processors_by_domain(int domain_id) const;

    /**
     * @brief Gets processors sharing an ALU with a specified processor.
     * @param proc_id Processor ID.
     * @return A set of processor IDs sharing the ALU.
     */
    const std::set<ProcID> &get_processors_share_alu(ProcID proc_id) const;

    /**
     * @brief Gets processors sharing an FPU with a specified processor.
     * @param proc_id Processor ID.
     * @return A set of processor IDs sharing the FPU.
     */
    const std::set<ProcID> &get_processors_share_fpu(ProcID proc_id) const;

    /**
     * @brief Gets processors sharing LD/ST with a specified processor.
     * @param proc_id Processor ID.
     * @return A set of processor IDs sharing LD/ST.
     */
    const std::set<ProcID> &get_processors_share_ldst(ProcID proc_id) const;

    /**
     * @brief Gets kernel processor IDs (usually assigned by OS) associated with a
     * specified processor.
     * @param proc_id Processor ID.
     * @return A set of kernel processor IDs.
     */
    const std::set<ProcID> &get_kernel_processor_ids(ProcID proc_id) const;

    /**
     * @brief Distributes processors across NUMA domains.
     * @return A vector of processor IDs representing the distribution.
     */
    std::vector<ProcID> distribute_processors_across_domains(void) const;

    /**
     * @brief Outputs the topology information to a stream.
     * @param os Output stream.
     * @param topo HardwareTopology instance.
     * @return The output stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const HardwareTopology &topo);

  private:
    size_t sys_memory_size{0};                       ///< Total system memory size.
    unsigned physical_cores{0};                      ///< Number of physical cores.

    typedef std::map<ProcID, Proc> ProcMap; ///< Map of processors by ID.

    /**
     * @struct DomainInfo
     * @brief Contains information about a NUMA domain.
     */
    struct DomainInfo {
      size_t memory_size;             ///< Size of memory in the domain.
      std::set<ProcID> logical_cores; ///< Set of logical cores in the domain.
    };

    typedef std::map<int, DomainInfo> DomainMap; ///< Map of NUMA domains by ID.

    ProcMap all_procs;   ///< Map of all processors.
    DomainMap by_domain; ///< Map of domains with their information.
  };

}; /* namespace Realm*/

#endif /* REALM_TOPOLOGY_H */