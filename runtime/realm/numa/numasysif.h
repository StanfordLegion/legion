/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// portability wrapper over NUMA system interfaces

// NOTE: this is nowhere near a full libnuma-style interface - it's just the
//  calls that Realm's NUMA module needs

#ifndef NUMASYSIF_H
#define NUMASYSIF_H

#include "realm/realm_config.h"

#include <stdlib.h>
#include <map>

namespace Realm {

  struct NumaNodeMemInfo {
    int node_id;
    size_t bytes_available;
  };

  struct NumaNodeCpuInfo {
    int node_id;
    int cores_available;
  };

  // is NUMA support available in the system?
  bool numasysif_numa_available(void);

  // return info on the memory and cpu in each NUMA node
  // default is to restrict to only those nodes enabled in the current affinity mask
  bool numasysif_get_mem_info(std::map<int, NumaNodeMemInfo>& info,
			      bool only_available = true);
  bool numasysif_get_cpu_info(std::map<int, NumaNodeCpuInfo>& info,
			      bool only_available = true);

  // return the "distance" between two nodes - try to normalize to Linux's model of
  //  10 being the same node and the cost for other nodes increasing by roughly 10
  //  per hop
  int numasysif_get_distance(int node1, int node2);

  // allocate memory on a given NUMA node - pin if requested
  void *numasysif_alloc_mem(int node, size_t bytes, bool pin);

  // free memory allocated on a given NUMA node
  bool numasysif_free_mem(int node, void *base, size_t bytes);

  // bind already-allocated memory to a given node - pin if requested
  // may fail if the memory has already been touched
  bool numasysif_bind_mem(int node, void *base, size_t bytes, bool pin);

};

#endif
