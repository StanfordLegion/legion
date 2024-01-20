// Copyright 2023 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// test for Realm's IDs

#include "realm.h"
#include "realm/nodeset.h"

#include <stdio.h>
#include <string.h>

#include <set>
#include <cassert>

#include "osdep.h"

using namespace Realm;

int num_nodesets = 32;
int num_nodes = 1024;
int seed = 12345;
int num_steps = 100000;
bool verbose = false;
bool twolevel = false;

NodeID random_nodeid()
{
  return lrand48() % num_nodes;
}

class NodeSetWrapper {
public:
  NodeSetWrapper()
  {
    nodeset = new NodeSet;
  }

  ~NodeSetWrapper()
  {
    delete nodeset;
  }

  void add(NodeID id)
  {
    nodeset->add(id);
    expected.insert(id);
  }

  void remove(NodeID id)
  {
    nodeset->remove(id);
    expected.erase(id);
  }

  void add_range(NodeID lo, NodeID hi)
  {
    nodeset->add_range(lo, hi);
    for(NodeID i = lo; i <= hi; i++)
      expected.insert(i);
  }

  void remove_range(NodeID lo, NodeID hi)
  {
    nodeset->remove_range(lo, hi);
    for(NodeID i = lo; i <= hi; i++)
      expected.erase(i);
  }

  bool contains(NodeID id) const
  {
    bool exp = (expected.count(id) > 0);
    bool act = nodeset->contains(id);
    assert(exp == act);
    return act;
  }

  void clear()
  {
    nodeset->clear();
    expected.clear();
  }

  void swap(NodeSetWrapper *swap_with)
  {
    nodeset->swap(*(swap_with->nodeset));
    expected.swap(swap_with->expected);
  }

  void copy(const NodeSetWrapper *copy_from, bool reconstruct)
  {
    if(reconstruct) {
      delete nodeset;
      nodeset = new NodeSet(*(copy_from->nodeset));
    } else {
      *nodeset = *(copy_from->nodeset);
    }
    expected = copy_from->expected;
  }

  void reconstruct()
  {
    delete nodeset;
    nodeset = new NodeSet;
    expected.clear();
  }

  void validate() const
  {
    assert(nodeset->empty() == expected.empty());
    assert(nodeset->size() == expected.size());
    size_t c1 = 0;
    for(NodeSet::const_iterator it = nodeset->begin();
	it != nodeset->end();
	++it) {
      assert(expected.count(*it) > 0);
      c1++;
    }
    assert(c1 == nodeset->size());
    for(std::set<NodeID>::const_iterator it = expected.begin();
	it != expected.end();
	++it)
      assert(nodeset->contains(*it));
  }

  NodeID random_nodeid_in_set()
  {
    size_t count = expected.size();
    if(count == 0)
      return -1;
    size_t skip = lrand48() % count;
    std::set<NodeID>::const_iterator it = expected.begin();
    while(skip-- > 0) ++it;
    return *it;
  }

protected:
  NodeSet *nodeset;
  std::set<NodeID> expected;
};

NodeSetWrapper *sets = 0;

NodeSetWrapper *random_nodeset()
{
  return &sets[lrand48() % num_nodesets];
}

enum {
  ACT_ADD,
  ACT_ADD_EXISTING,
  ACT_REMOVE,
  ACT_REMOVE_EXISTING,
  ACT_ADD_RANGE,
  ACT_ADD_RANGE_SMALL,
  ACT_REMOVE_RANGE,
  ACT_REMOVE_RANGE_SMALL,
  ACT_CONTAINS,
  ACT_CLEAR,
  ACT_SWAP,
  ACT_COPY,
  ACT_COPY_RECON,
  ACT_RECON,
  ACT_TOTAL_ACTIONS
};

int main(int argc, const char *argv[])
{
  // parse args
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-s")) {
      seed = atoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-n")) {
      num_nodesets = atoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-m")) {
      num_nodes = atoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-i")) {
      num_steps = atoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-v")) {
      verbose = true;
      continue;
    }
    if(!strcmp(argv[i], "-t")) {
      twolevel = true;
      continue;
    }
  }

  NodeSetBitmask::configure_allocator(num_nodes - 1,
				      4,
				      twolevel);

  srand48(seed);

  sets = new NodeSetWrapper[num_nodesets];

  for(int i = 0; i < num_steps; i++) {
    // choose an action
    int act = lrand48() % ACT_TOTAL_ACTIONS;

    switch(act) {
    case ACT_ADD:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = random_nodeid();
	if(verbose)
          printf("ADD(%p, %d)\n", static_cast<void *>(s1), id1);
        s1->add(id1);
	s1->validate();
	break;
      }

    case ACT_ADD_EXISTING:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = s1->random_nodeid_in_set();
	if(id1 == -1) break;
	if(verbose)
          printf("ADD_EXIST(%p, %d)\n", static_cast<void *>(s1), id1);
        s1->add(id1);
	s1->validate();
	break;
      }

    case ACT_REMOVE:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = random_nodeid();
	if(verbose)
          printf("REMOVE(%p, %d)\n", static_cast<void *>(s1), id1);
        s1->remove(id1);
	s1->validate();
	break;
      }

    case ACT_REMOVE_EXISTING:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = s1->random_nodeid_in_set();
	if(id1 == -1) break;
	if(verbose)
          printf("REMOVE_EXIST(%p, %d)\n", static_cast<void *>(s1), id1);
        s1->remove(id1);
	s1->validate();
	break;
      }

    case ACT_ADD_RANGE:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = random_nodeid();
	NodeID id2 = random_nodeid();
	if(verbose)
          printf("ADD_RANGE(%p, %d, %d)\n", static_cast<void *>(s1), id1, id2);
        s1->add_range(id1, id2);
	s1->validate();
	break;
      }

    case ACT_ADD_RANGE_SMALL:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = random_nodeid();
	NodeID id2 = id1 + (lrand48() % 32);
	if(id2 >= num_nodes) break;
	if(verbose)
          printf("ADD_RANGE_SMALL(%p, %d, %d)\n", static_cast<void *>(s1), id1, id2);
        s1->add_range(id1, id2);
	s1->validate();
	break;
      }

    case ACT_REMOVE_RANGE:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = random_nodeid();
	NodeID id2 = random_nodeid();
	if(verbose)
          printf("REMOVE_RANGE(%p, %d, %d)\n", static_cast<void *>(s1), id1, id2);
        s1->remove_range(id1, id2);
	s1->validate();
	break;
      }

    case ACT_REMOVE_RANGE_SMALL:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = random_nodeid();
	NodeID id2 = id1 + (lrand48() % 32);
	if(id2 >= num_nodes) break;
	if(verbose)
          printf("REMOVE_RANGE_SMALL(%p, %d, %d)\n", static_cast<void *>(s1), id1, id2);
        s1->remove_range(id1, id2);
	s1->validate();
	break;
      }

    case ACT_CONTAINS:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeID id1 = random_nodeid();
	if(verbose) {
          printf("CONTAINS(%p, %d) = ", static_cast<void *>(s1), id1);
          fflush(stdout);
	}
	bool present = s1->contains(id1);
	if(verbose)
	  printf("%d\n", present);
	s1->validate();
	break;
      }

    case ACT_CLEAR:
      {
	NodeSetWrapper *s1 = random_nodeset();
	if(verbose)
          printf("CLEAR(%p)\n", static_cast<void *>(s1));
        s1->clear();
	s1->validate();
	break;
      }

    case ACT_SWAP:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeSetWrapper *s2 = random_nodeset();
	if(verbose)
          printf("SWAP(%p, %p)\n", static_cast<void *>(s1), static_cast<void *>(s2));
        s1->swap(s2);
	s1->validate();
	s2->validate();
	break;
      }

    case ACT_COPY:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeSetWrapper *s2 = random_nodeset();
	if(verbose)
          printf("COPY(%p, %p)\n", static_cast<void *>(s1), static_cast<void *>(s2));
        s1->copy(s2, false /*!reconstruct*/);
	s1->validate();
	s2->validate();
	break;
      }

    case ACT_COPY_RECON:
      {
	NodeSetWrapper *s1 = random_nodeset();
	NodeSetWrapper *s2 = random_nodeset();
	if(s1 == s2) break;
	if(verbose)
          printf("COPY_RECON(%p, %p)\n", static_cast<void *>(s1),
                 static_cast<void *>(s2));
        s1->copy(s2, true /*reconstruct*/);
	s1->validate();
	s2->validate();
	break;
      }

    case ACT_RECON:
      {
	NodeSetWrapper *s1 = random_nodeset();
	if(verbose)
          printf("RECON(%p)\n", static_cast<void *>(s1));
        s1->reconstruct();
	s1->validate();
	break;
      }

    default:
      //assert(0);
      break;
    }
  }

  delete[] sets;

  NodeSetBitmask::free_allocations();
}
