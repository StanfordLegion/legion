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
  INIT_DATA_TASK,
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
//static int random_seed = 12345;
//static bool random_colors = false;

struct InitDataArgs {
  int index;
  RegionInstance ri_nodes, ri_edges;
};

Logger log_app("app");

void init_data_task(const void *args, size_t arglen, Processor p)
{
  const InitDataArgs& i_args = *(const InitDataArgs *)args;

  log_app.print() << "init task #" << i_args.index << " (ri_nodes=" << i_args.ri_nodes << ", ri_edges=" << i_args.ri_edges << ")";

  ZIndexSpace<1> is_nodes = i_args.ri_nodes.get_indexspace<1>();
  ZIndexSpace<1> is_edges = i_args.ri_edges.get_indexspace<1>();

  log_app.print() << "N: " << is_nodes;
  log_app.print() << "E: " << is_edges;
}

void top_level_task(const void *args, size_t arglen, Processor p)
{
  int errors = 0;

  printf("Realm dependent partitioning test - %d nodes, %d edges, %d pieces\n",
	 num_nodes, num_edges, num_pieces);

  // find all the system memories - we'll stride our data across them
  // for each memory, we'll need one CPU that can do the initialization of the data
  std::vector<Memory> sysmems;
  std::vector<Processor> procs;

  Machine machine = Machine::get_machine();
  {
    std::set<Memory> all_memories;
    machine.get_all_memories(all_memories);
    for(std::set<Memory>::const_iterator it = all_memories.begin();
	it != all_memories.end();
	it++) {
      Memory m = *it;
      if(m.kind() == Memory::SYSTEM_MEM) {
	sysmems.push_back(m);
	std::set<Processor> pset;
	machine.get_shared_processors(m, pset);
	Processor p = Processor::NO_PROC;
	for(std::set<Processor>::const_iterator it2 = pset.begin();
	    it2 != pset.end();
	    it2++) {
	  if(it2->kind() == Processor::LOC_PROC) {
	    p = *it2;
	    break;
	  }
	}
	assert(p.exists());
	procs.push_back(p);
      }
    }
  }
  assert(sysmems.size() > 0);

  // now create index spaces for nodes and edges
  ZIndexSpace<1> is_nodes(ZRect<1>(0, num_nodes - 1));
  ZIndexSpace<1> is_edges(ZRect<1>(0, num_edges - 1));

  // equal partition is used to do initial population of edges and nodes
  std::vector<ZIndexSpace<1> > ss_nodes_eq;
  std::vector<ZIndexSpace<1> > ss_edges_eq;

  is_nodes.create_equal_subspaces(num_pieces, 1, ss_nodes_eq, Realm::ProfilingRequestSet()).wait();
  is_edges.create_equal_subspaces(num_pieces, 1, ss_edges_eq, Realm::ProfilingRequestSet()).wait();

  std::cout << "Initial partitions:" << std::endl;
  for(size_t i = 0; i < ss_nodes_eq.size(); i++)
    std::cout << " Nodes #" << i << ": " << ss_nodes_eq[i] << std::endl;
  for(size_t i = 0; i < ss_edges_eq.size(); i++)
    std::cout << " Edges #" << i << ": " << ss_edges_eq[i] << std::endl;

  // create instances for each of these subspaces
  std::vector<size_t> node_fields, edge_fields;
  node_fields.push_back(sizeof(int));  // subckt_id
  edge_fields.push_back(sizeof(int));  // in_node
  edge_fields.push_back(sizeof(int));  // out_node

  std::vector<RegionInstance> ri_nodes, ri_edges;

  for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
    RegionInstance ri = ss_nodes_eq[i].create_instance(sysmems[i % sysmems.size()],
						       node_fields,
						       1,
						       Realm::ProfilingRequestSet());
    ri_nodes.push_back(ri);
  }

  for(size_t i = 0; i < ss_edges_eq.size(); i++) {
    RegionInstance ri = ss_edges_eq[i].create_instance(sysmems[i % sysmems.size()],
						       edge_fields,
						       1,
						       Realm::ProfilingRequestSet());
    ri_edges.push_back(ri);
  }

  // fire off tasks to initialize data
  std::set<Event> events;
  for(int i = 0; i < num_pieces; i++) {
    Processor p = procs[i % sysmems.size()];
    InitDataArgs args;
    args.index = i;
    args.ri_nodes = ri_nodes[i];
    args.ri_edges = ri_edges[i];
    Event e = p.spawn(INIT_DATA_TASK, &args, sizeof(args));
    events.insert(e);
  }
  Event::merge_events(events).wait();

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
  rt.register_task(INIT_DATA_TASK, init_data_task);

  signal(SIGALRM, sigalrm_handler);

  // Start the machine running
  // Control never returns from this call
  // Note we only run the top level task on one processor
  // You can also run the top level task on all processors or one processor per node
  rt.run(TOP_LEVEL_TASK, Runtime::ONE_TASK_ONLY);

  //rt.shutdown();
  return 0;
}
