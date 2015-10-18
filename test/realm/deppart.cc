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
static int pct_wire_in_piece = 50;
static int random_seed = 12345;
static bool random_colors = false;
static bool show_graph = true;
static bool wait_on_events = false;

struct InitDataArgs {
  int index;
  RegionInstance ri_nodes, ri_edges;
};

Logger log_app("app");

static ZPoint<1> random_node(const ZIndexSpace<1>& is_nodes, unsigned short *rngstate, bool in_piece)
{
  if(in_piece)
    return (is_nodes.bounds.lo + (nrand48(rngstate) % (is_nodes.bounds.hi.x - is_nodes.bounds.lo.x + 1)));
  else
    return (nrand48(rngstate) % num_nodes);
}

void init_data_task(const void *args, size_t arglen, Processor p)
{
  const InitDataArgs& i_args = *(const InitDataArgs *)args;

  log_app.print() << "init task #" << i_args.index << " (ri_nodes=" << i_args.ri_nodes << ", ri_edges=" << i_args.ri_edges << ")";

  ZIndexSpace<1> is_nodes = i_args.ri_nodes.get_indexspace<1>();
  ZIndexSpace<1> is_edges = i_args.ri_edges.get_indexspace<1>();

  log_app.print() << "N: " << is_nodes;
  log_app.print() << "E: " << is_edges;

  unsigned short rngstate[3];
  rngstate[0] = random_seed;
  rngstate[1] = is_nodes.bounds.lo;
  rngstate[2] = 0;
  for(int i = 0; i < 20; i++) nrand48(rngstate);

  {
    AffineAccessor<int,1> a_subckt_id(i_args.ri_nodes, 0 /* offset */);

    for(int i = is_nodes.bounds.lo; i <= is_nodes.bounds.hi; i++) {
      int color;
      if(random_colors)
	color = nrand48(rngstate) % num_pieces;
      else
	color = i_args.index;
      a_subckt_id.write(i, color);
    }
  }

  {
    AffineAccessor<ZPoint<1>,1> a_in_node(i_args.ri_edges, 0 * sizeof(ZPoint<1>) /* offset */);
    AffineAccessor<ZPoint<1>,1> a_out_node(i_args.ri_edges, 1 * sizeof(ZPoint<1>) /* offset */);

    for(int i = is_edges.bounds.lo; i <= is_edges.bounds.hi; i++) {
      int in_node = random_node(is_nodes, rngstate,
				!random_colors);
      int out_node = random_node(is_nodes, rngstate,
				 !random_colors && ((nrand48(rngstate) % 100) < pct_wire_in_piece));
      a_in_node.write(i, in_node);
      a_out_node.write(i, out_node);
    }
  }

  if(show_graph) {
    AffineAccessor<int,1> a_subckt_id(i_args.ri_nodes, 0 /* offset */);

    for(int i = is_nodes.bounds.lo; i <= is_nodes.bounds.hi; i++)
      std::cout << "subckt_id[" << i << "] = " << a_subckt_id.read(i) << std::endl;

    AffineAccessor<ZPoint<1>,1> a_in_node(i_args.ri_edges, 0 * sizeof(ZPoint<1>) /* offset */);

    for(int i = is_edges.bounds.lo; i <= is_edges.bounds.hi; i++)
      std::cout << "in_node[" << i << "] = " << a_in_node.read(i) << std::endl;

    AffineAccessor<ZPoint<1>,1> a_out_node(i_args.ri_edges, 1 * sizeof(ZPoint<1>) /* offset */);

    for(int i = is_edges.bounds.lo; i <= is_edges.bounds.hi; i++)
      std::cout << "out_node[" << i << "] = " << a_out_node.read(i) << std::endl;
  }
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
  assert(sizeof(int) == sizeof(ZPoint<1>));
  edge_fields.push_back(sizeof(ZPoint<1>));  // in_node
  edge_fields.push_back(sizeof(ZPoint<1>));  // out_node

  std::vector<RegionInstance> ri_nodes(num_pieces);
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, int> > subckt_field_data(num_pieces);

  for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
    RegionInstance ri = ss_nodes_eq[i].create_instance(sysmems[i % sysmems.size()],
						       node_fields,
						       1,
						       Realm::ProfilingRequestSet());
    ri_nodes[i] = ri;

    subckt_field_data[i].index_space = ss_nodes_eq[i];
    subckt_field_data[i].inst = ri_nodes[i];
    subckt_field_data[i].field_offset = 0;
  }

  std::vector<RegionInstance> ri_edges(num_pieces);
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, ZPoint<1> > > in_node_field_data(num_pieces);
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, ZPoint<1> > > out_node_field_data(num_pieces);

  for(size_t i = 0; i < ss_edges_eq.size(); i++) {
    RegionInstance ri = ss_edges_eq[i].create_instance(sysmems[i % sysmems.size()],
						       edge_fields,
						       1,
						       Realm::ProfilingRequestSet());
    ri_edges[i] = ri;

    in_node_field_data[i].index_space = ss_edges_eq[i];
    in_node_field_data[i].inst = ri_edges[i];
    in_node_field_data[i].field_offset = 0 * sizeof(ZPoint<1>);
      
    out_node_field_data[i].index_space = ss_edges_eq[i];
    out_node_field_data[i].inst = ri_edges[i];
    out_node_field_data[i].field_offset = 1 * sizeof(ZPoint<1>);
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
  // we're going to time the next stuff, so always wait here
  Event::merge_events(events).wait();

  // the outputs of our partitioning will be:
  //  is_private, is_shared - subsets of is_nodes based on private/shared
  //  p_pvt, p_shr, p_ghost - subsets of the above split by subckt
  //  p_edges               - subsets of is_edges for each subckt

  ZIndexSpace<1> is_shared, is_private;
  std::vector<ZIndexSpace<1> > p_pvt, p_shr, p_ghost;
  std::vector<ZIndexSpace<1> > p_edges;

  // now actual partitioning work
  {
    Realm::TimeStamp ts("dependent partitioning work", true, &log_app);

    // first partition nodes by subckt id (this is the independent partition,
    //  but not actually used by the app)
    std::vector<ZIndexSpace<1> > p_nodes;

    std::vector<int> colors(num_pieces);
    for(int i = 0; i < num_pieces; i++)
      colors[i] = i;

    Event e1 = is_nodes.create_subspaces_by_field(subckt_field_data,
						  colors,
						  p_nodes,
						  Realm::ProfilingRequestSet());
    if(wait_on_events) e1.wait();

    // now compute p_edges based on the color of their in_node (i.e. a preimage)
    Event e2 = is_edges.create_subspaces_by_preimage(in_node_field_data,
						     p_nodes,
						     p_edges,
						     Realm::ProfilingRequestSet(),
						     e1);
    if(wait_on_events) e2.wait();

    // an image of p_edges through out_node gives us all the shared nodes, along
    //  with some private nodes
    std::vector<ZIndexSpace<1> > p_extra_nodes;

    Event e3 = is_nodes.create_subspaces_by_image(out_node_field_data,
						  p_edges,
						  p_extra_nodes,
						  Realm::ProfilingRequestSet(),
						  e2);
    if(wait_on_events) e3.wait();
  
    // subtracting out those private nodes gives us p_ghost
    Event e4 = ZIndexSpace<1>::compute_differences(p_extra_nodes,
						   p_nodes,
						   p_ghost,
						   Realm::ProfilingRequestSet(),
						   e3);
    if(wait_on_events) e4.wait();

    // the union of everybody's ghost nodes is is_shared
    Event e5 = ZIndexSpace<1>::compute_union(p_ghost, is_shared,
					     Realm::ProfilingRequestSet(),
					     e4);
    if(wait_on_events) e5.wait();

    // and is_private is just the nodes of is_nodes that aren't in is_shared
    Event e6 = ZIndexSpace<1>::compute_difference(is_nodes, is_shared, is_private,
						  Realm::ProfilingRequestSet(),
						  e5);
    if(wait_on_events) e6.wait();

    // the intersection of the original p_nodes with is_shared gives us p_shr
    // (note that we can do this in parallel with the computation of is_private)
    Event e7 = ZIndexSpace<1>::compute_intersections(p_nodes, is_shared, p_shr,
						     Realm::ProfilingRequestSet(),
						     e5);
    if(wait_on_events) e7.wait();

    // and finally, the intersection of p_nodes with is_private gives us p_pvt
    Event e8 = ZIndexSpace<1>::compute_intersections(p_nodes, is_private, p_pvt,
						     Realm::ProfilingRequestSet(),
						     e6);
    if(wait_on_events) e8.wait();

    // all done - wait on e7 and e8, which dominate every other operation
    e7.wait();
    e8.wait();
  }

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
