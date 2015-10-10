#include "realm/realm.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>
#include <unistd.h>

using namespace Realm;

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  INIT_NODES_TASK,
  INIT_EDGES_TASK,
};

// we're going to use alarm() as a watchdog to detect deadlocks
void sigalrm_handler(int sig)
{
  fprintf(stderr, "HELP!  Alarm triggered - likely deadlock!\n");
  exit(1);
}

static int num_nodes = 100;
static int num_edges = 10;
static int num_pieces = 2;

void top_level_task(const void *args, size_t arglen, Processor p)
{
  int errors = 0;

  printf("Realm dependent partitioning test - %d nodes, %d edges, %d pieces\n",
	 num_nodes, num_edges, num_pieces);

  // find all the system memories - we'll stride our data across them
  std::vector<Memory> sysmems;

  Machine machine = Machine::get_machine();
  {
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(std::set<Memory>::const_iterator it = all_memories.begin();
	it != all_memories.end();
	it++) {
      Memory m = *it;
      if(m.kind() == Memory::SYSTEM_MEM)
	sysmems.push_back(m);
    }
  }
  assert(sysmems.size() > 0);

  // now create index spaces for nodes and edges
  ZIndexSpace<1> is_nodes(ZRect<1>(0, num_nodes - 1));

  if(errors > 0) {
    printf("Exiting with errors\n");
    exit(1);
  }

  printf("all done!\n");
  sleep(1);

  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-n")) {
      num_nodes = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-e")) {
      num_edges = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-p")) {
      num_pieces = atoi(argv[++i]);
      continue;
    }
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  signal(SIGALRM, sigalrm_handler);

  // Start the machine running
  // Control never returns from this call
  // Note we only run the top level task on one processor
  // You can also run the top level task on all processors or one processor per node
  rt.run(TOP_LEVEL_TASK, Runtime::ONE_TASK_ONLY);

  //rt.shutdown();
  return 0;
}
