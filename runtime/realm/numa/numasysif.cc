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

// portability wrapper over NUMA system interfaces

// NOTE: this is nowhere near a full libnuma-style interface - it's just the
//  calls that Realm's NUMA module needs

#include "realm/numa/numasysif.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include <vector>

#ifdef __linux__
#include <alloca.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <dirent.h>
#include <sched.h>
#include <ctype.h>
#include <sys/mman.h>

namespace {
  long get_mempolicy(int *policy, unsigned long *nmask,
		     unsigned long maxnode, void *addr, int flags)
  {
    return syscall(__NR_get_mempolicy, policy, nmask,
		   maxnode, addr, flags);
  }

  long mbind(void *start, unsigned long len, int mode,
	     const unsigned long *nmask, unsigned long maxnode, unsigned flags)
  {
    return syscall(__NR_mbind, (long)start, len, mode, (long)nmask,
		   maxnode, flags);
  }

#if 0
  long set_mempolicy(int mode, const unsigned long *nmask,
		     unsigned long maxnode)
  {
    return syscall(__NR_set_mempolicy, mode, nmask, maxnode);
  }
#endif
};
#endif

namespace Realm {

  // as soon as we get more than one real version of these, split them out into
  //  separate files

#ifdef __linux__
  namespace {
    // Linux wants you to guess how many nodes there are, and if you're wrong,
    //  it just tells you to try again - save the answer here so we only have
    //  to do it once
    int detected_node_count = 8 * sizeof(unsigned long);
    const int max_supported_node_count = 1024 * sizeof(unsigned long);
  };

  static bool mask_nonempty(const unsigned char *nmask, int max_count)
  {
    for(int i = 0; i < (max_count >> 3); i++)
      if(nmask[i] != 0)
	return true;
    return false;
  }
#endif

  // is NUMA support available in the system?
  bool numasysif_numa_available(void)
  {
#ifdef __linux__
    int policy;
    unsigned char *nmask = (unsigned char *)alloca(max_supported_node_count >> 3);
    while(1) {
      errno = 0;
      int ret = get_mempolicy(&policy,
			      (unsigned long *)nmask, detected_node_count,
			      0, MPOL_F_MEMS_ALLOWED);
      if(ret == 0) break;

      // EINVAL maybe means our mask isn't big enough
      if((errno == EINVAL) &&
	 (detected_node_count < max_supported_node_count)) {
	detected_node_count <<= 1;
      } else {
	// otherwise we're out of luck
        //fprintf(stderr, "get_mempolicy() returned: ret=%d errno=%d\n", ret, errno);
	return false;
      }
    }

    // also check that we have at least one node set in the mask - if not, 
    //  assume numa support is disabled
    if(!mask_nonempty(nmask, detected_node_count)) {
      //fprintf(stderr, "get_mempolicy() returned empty node mask!\n");
      return false;
    }

    return true;
#else
    return false;
#endif
  }

  // return info on the memory and cpu in each NUMA node
  // default is to restrict to only those nodes enabled in the current affinity mask
  bool numasysif_get_mem_info(std::map<int, NumaNodeMemInfo>& info,
			      bool only_available /*= true*/)
  {
#ifdef __linux__
    int policy = -1;
    unsigned char *nmask = (unsigned char *)alloca(detected_node_count >> 3);
    for(int i = 0; i < detected_node_count >> 3; i++)
      nmask[i] = 0;
    int ret = -1;
    if(only_available) {
      // first, ask for the default policy for the thread to detect binding via 
      //  numactl, mpirun, etc.
      errno = 0;
      ret = get_mempolicy(&policy,
			  (unsigned long *)nmask, detected_node_count, 0, 0);
    }
    if((ret != 0) || (policy != MPOL_BIND) ||
       !mask_nonempty(nmask, detected_node_count)) {
      // not a reasonable-looking bound state, so ask for all nodes
      errno = 0;
      ret = get_mempolicy(&policy,
			  (unsigned long *)nmask, detected_node_count,
			  0, MPOL_F_MEMS_ALLOWED);
      if((ret != 0) || !mask_nonempty(nmask, detected_node_count)) {
	// this really shouldn't fail, since we made the same call above in
	//  numasysif_numa_available()
	fprintf(stderr, "mems_allowed: ret=%d errno=%d mask=%08lx count=%d\n",
		ret, errno, *(unsigned long *)nmask, detected_node_count);
	return false;
      }
    }

    // for each bit set in the mask, try to query the free memory
    for(int i = 0; i < detected_node_count; i++)
      if(((nmask[i >> 3] >> (i & 7)) & 1) != 0) {
	// free information comes from /sys...
	char fname[80];
	sprintf(fname, "/sys/devices/system/node/node%d/meminfo", i);
	FILE *f = fopen(fname, "r");
	if(!f) {
	  fprintf(stderr, "can't read '%s': %s\n", fname, strerror(errno));
	  continue;
	}
	char line[256];
	while(fgets(line, 256, f)) {
	  const char *s = strstr(line, "MemFree");
	  if(!s) continue;
	  const char *endptr;
	  errno = 0;
	  long long sz = strtoll(s+9, (char **)&endptr, 10);
	  if((errno != 0) || strcmp(endptr, " kB\n")) {
	    fprintf(stderr, "ill-formed line: '%s' '%s'\n", s, endptr);
	    continue;
	  }
	  // success - add this to the list and stop reading
	  NumaNodeMemInfo& mi = info[i];
	  mi.node_id = i;
	  mi.bytes_available = (sz << 10);
	  break;
	}
	// if we get all the way through the file without finding the size,
	//   we just don't add anything to the info
	fclose(f);
      }

    // as long as we got at least one valid node, assume we're successful
    return !info.empty();
#else
    return false;
#endif
  }

  bool numasysif_get_cpu_info(std::map<int, NumaNodeCpuInfo>& info,
			      bool only_available /*= true*/)
  {
#ifdef __linux__
    // if we're restricting to what's been made available, find what's been 
    //  made available
    cpu_set_t avail_cpus;
    if(only_available) {
      int ret = sched_getaffinity(0, sizeof(avail_cpus), &avail_cpus);
      if(ret != 0) {
	fprintf(stderr, "sched_getaffinity failed: %s\n", strerror(errno));
	return false;
      }
    } else
      CPU_ZERO(&avail_cpus);

    // now enumerate cpus via /sys and determine which nodes they belong to
    std::map<int, int> cpu_counts;
    DIR *cpudir = opendir("/sys/devices/system/cpu");
    if(!cpudir) {
      fprintf(stderr, "couldn't read /sys/devices/system/cpu: %s\n", strerror(errno));
      return false;
    }
    struct dirent *de;
    while((de = readdir(cpudir)) != 0)
      if(!strncmp(de->d_name, "cpu", 3)) {
	int cpu_index = atoi(de->d_name + 3);
	if(only_available && !CPU_ISSET(cpu_index, &avail_cpus))
	  continue;
	// find the node symlink to determine the node
	char path2[256];
	sprintf(path2, "/sys/devices/system/cpu/%.16s", de->d_name);
	DIR *d2 = opendir(path2);
	if(!d2) {
	  fprintf(stderr, "couldn't read '%s': %s\n", path2, strerror(errno));
	  continue;
	}
	struct dirent *de2;
	while((de2 = readdir(d2)) != 0)
	  if(!strncmp(de2->d_name, "node", 4)) {
	    int node_index = atoi(de2->d_name + 4);
	    cpu_counts[node_index]++;
	    break;
	  }
	closedir(d2);
      }

    // any matches is "success"
    if(!cpu_counts.empty()) {
      for(std::map<int,int>::const_iterator it = cpu_counts.begin();
	  it != cpu_counts.end();
	  ++it) {
	NumaNodeCpuInfo& ci = info[it->first];
	ci.node_id = it->first;
	ci.cores_available = it->second;
      }
      return true;
    } else
      return false;
#else
    return false;
#endif
  }

  // return the "distance" between two nodes - try to normalize to Linux's model of
  //  10 being the same node and the cost for other nodes increasing by roughly 10
  //  per hop
  int numasysif_get_distance(int node1, int node2)
  {
#ifdef __linux__
    static std::map<int, std::vector<int> > saved_distances;

    std::map<int, std::vector<int> >::iterator it = saved_distances.find(node1);
    if(it == saved_distances.end()) {
      // not one we've already looked up, so do it now

      // if we break out early, we'll end up with an empty vector, which means
      //  we'll return -1 for all future queries
      std::vector<int>& v = saved_distances[node1];

      char fname[256];
      sprintf(fname, "/sys/devices/system/node/node%d/distance", node1);
      FILE *f = fopen(fname, "r");
      if(!f) {
	fprintf(stderr, "can't read '%s': %s\n", fname, strerror(errno));
	saved_distances[node1].clear();
	return -1;
      }
      char line[256];
      if(fgets(line, 256, f)) {
	char *p = line;
	while(isdigit(*p)) {
	  errno = 0;
	  int d = strtol(p, &p, 10);
	  if(errno != 0) break;
	  v.push_back(d);
	  while(isspace(*p)) p++;
	}
      }
      fclose(f);
      if((node2 >= 0) && (node2 < (int)v.size()))
	return v[node2];
      else
	return -1;
    } else {
      const std::vector<int>& v = it->second;
      if((node2 >= 0) && (node2 < (int)v.size()))
	return v[node2];
      else
	return -1;
    }
#else
    return -1;
#endif
  }

  // allocate memory on a given NUMA node - pin if requested
  void *numasysif_alloc_mem(int node, size_t bytes, bool pin)
  {
#ifdef __linux__
    // get memory from mmap
    // TODO: hugetlbfs, if possible
    void *base = mmap(0,
		      bytes, 
		      PROT_READ | PROT_WRITE,
		      MAP_PRIVATE | MAP_ANONYMOUS,
		      -1,
		      0);
    if(!base) return 0;

    // use the bind call for the rest
    if(numasysif_bind_mem(node, base, bytes, pin))
      return base;

    // if not, clean up and return failure
    numasysif_free_mem(node, base, bytes);
    return 0;
#else
    return 0;
#endif
  }

  // free memory allocated on a given NUMA node
  bool numasysif_free_mem(int node, void *base, size_t bytes)
  {
#ifdef __linux__
    int ret = munmap(base, bytes);
    return(ret == 0);
#else
    return false;
#endif
  }

  // bind already-allocated memory to a given node - pin if requested
  // may fail if the memory has already been touched
  bool numasysif_bind_mem(int node, void *base, size_t bytes, bool pin)
  {
#ifdef __linux__
    int policy = MPOL_BIND;
    if((node < 0) || (node >= detected_node_count)) {
      fprintf(stderr, "bind request for node out of range: %d\n", node);
      return false;
    }
    unsigned char *nmask = (unsigned char *)alloca(detected_node_count >> 3);
    for(int i = 0; i < detected_node_count >> 3; i++)
      nmask[i] = 0;
    nmask[(node >> 3)] = (1 << (node & 7));
    int ret = mbind(base, bytes,
		    policy,
		    (const unsigned long *)nmask, detected_node_count,
		    MPOL_MF_STRICT | MPOL_MF_MOVE);
    if(ret != 0) {
      fprintf(stderr, "failed to bind memory for node %d: %s\n", node, strerror(errno));
      return false;
    }

    // attempt to pin the memory if requested
    if(pin) {
      int ret = mlock(base, bytes);
      if(ret != 0) {
	fprintf(stderr, "mlock failed for memory on node %d: %s\n", node, strerror(errno));
	return false;
      }
    }

    return true;
#else
    return false;
#endif
  }

};
