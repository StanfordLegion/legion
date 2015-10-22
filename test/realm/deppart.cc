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
  INIT_CIRCUIT_DATA_TASK,
  INIT_PENNANT_DATA_TASK,
};

// we're going to use alarm() as a watchdog to detect deadlocks
void sigalrm_handler(int sig)
{
  fprintf(stderr, "HELP!  Alarm triggered - likely deadlock!\n");
  exit(1);
}

template <typename T, T DEFVAL>
class WithDefault {
public:
  WithDefault(void) : val(DEFVAL) {}
  WithDefault(T _val) : val(_val) {}
  WithDefault<T,DEFVAL>& operator=(T _val) { val = _val; return *this; }
  operator T(void) const { return val; }
protected:
  T val;
};

Logger log_app("app");

class TestInterface {
public:
  virtual ~TestInterface(void) {}

  virtual void print_info(void) = 0;

  virtual Event initialize_data(const std::vector<Memory>& memories,
				const std::vector<Processor>& procs) = 0;

  virtual Event perform_partitioning(void) = 0;

  virtual int check_partitioning(void) = 0;
};

// generic configuration settings
namespace {
  int random_seed = 12345;
  bool random_colors = false;
  bool wait_on_events = false;
  bool show_graph = true;
  TestInterface *testcfg = 0;
};

class CircuitTest : public TestInterface {
public:
  // graph config parameters
  WithDefault<int, 100> num_nodes;
  WithDefault<int,  10> num_edges;
  WithDefault<int,   2> num_pieces;
  WithDefault<int,  50> pct_wire_in_piece;

  CircuitTest(int argc, const char *argv[])
  {
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
  }

  struct InitDataArgs {
    int index;
    RegionInstance ri_nodes, ri_edges;
  };

  ZPoint<1> random_node(const ZIndexSpace<1>& is_nodes, unsigned short *rngstate, bool in_piece)
  {
    if(in_piece)
      return (is_nodes.bounds.lo + (nrand48(rngstate) % (is_nodes.bounds.hi.x - is_nodes.bounds.lo.x + 1)));
    else
      return (nrand48(rngstate) % num_nodes);
  }

  static void init_data_task_wrapper(const void *args, size_t arglen, Processor p)
  {
    CircuitTest *me = (CircuitTest *)testcfg;
    me->init_data_task(args, arglen, p);
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

  ZIndexSpace<1> is_nodes, is_edges;
  std::vector<RegionInstance> ri_nodes;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, int> > subckt_field_data;
  std::vector<RegionInstance> ri_edges;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, ZPoint<1> > > in_node_field_data;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, ZPoint<1> > > out_node_field_data;

  virtual void print_info(void)
  {
    printf("Realm dependent partitioning test - circuit: %d nodes, %d edges, %d pieces\n",
	   (int)num_nodes, (int)num_edges, (int)num_pieces);
  }

  virtual Event initialize_data(const std::vector<Memory>& memories,
				const std::vector<Processor>& procs)
  {
    // now create index spaces for nodes and edges
    is_nodes = ZRect<1>(0, num_nodes - 1);
    is_edges = ZRect<1>(0, num_edges - 1);

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

    ri_nodes.resize(num_pieces);
    subckt_field_data.resize(num_pieces);

    for(size_t i = 0; i < ss_nodes_eq.size(); i++) {
      RegionInstance ri = ss_nodes_eq[i].create_instance(memories[i % memories.size()],
							 node_fields,
							 1,
							 Realm::ProfilingRequestSet());
      ri_nodes[i] = ri;
    
      subckt_field_data[i].index_space = ss_nodes_eq[i];
      subckt_field_data[i].inst = ri_nodes[i];
      subckt_field_data[i].field_offset = 0;
    }

    ri_edges.resize(num_pieces);
    in_node_field_data.resize(num_pieces);
    out_node_field_data.resize(num_pieces);

    for(size_t i = 0; i < ss_edges_eq.size(); i++) {
      RegionInstance ri = ss_edges_eq[i].create_instance(memories[i % memories.size()],
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
      Processor p = procs[i % memories.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_nodes = ri_nodes[i];
      args.ri_edges = ri_edges[i];
      Event e = p.spawn(INIT_CIRCUIT_DATA_TASK, &args, sizeof(args));
      events.insert(e);
    }

    return Event::merge_events(events);
  }

  // the outputs of our partitioning will be:
  //  is_private, is_shared - subsets of is_nodes based on private/shared
  //  p_pvt, p_shr, p_ghost - subsets of the above split by subckt
  //  p_edges               - subsets of is_edges for each subckt

  ZIndexSpace<1> is_shared, is_private;
  std::vector<ZIndexSpace<1> > p_pvt, p_shr, p_ghost;
  std::vector<ZIndexSpace<1> > p_edges;

  virtual Event perform_partitioning(void)
  {
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
    return Event::merge_events(e7, e8);
  }

  virtual int check_partitioning(void)
  {
    // TODO
    return 0;
  }
};

class PennantTest : public TestInterface {
public:
public:
  // graph config parameters
  enum MeshType {
    RectangularMesh,
  };
  WithDefault<MeshType, RectangularMesh> mesh_type;
  WithDefault<int,  10> nzx;  // number of zones in x
  WithDefault<int,  10> nzy;  // number of zones in y
  WithDefault<int,   2> numpcx;  // number of submeshes in x
  WithDefault<int,   2> numpcy;  // number of submeshes in y

  int npx, npy;      // number of points in each dimension
  int nz, ns, np, numpc;  // total number of zones, sides, points, and pieces
  std::vector<int> zxbound, zybound; // x and y split points between submeshes
  std::vector<int> lz, ls, lp;  // number of zones, sides, and points in each submesh

  PennantTest(int argc, const char *argv[])
  {
#define INT_ARG(s, v) if(!strcmp(argv[i], s)) { v = atoi(argv[++i]); continue; }
    for(int i = 1; i < argc; i++) {
      INT_ARG("-nzx",    nzx)
      INT_ARG("-nzy",    nzy)
      INT_ARG("-numpcx", numpcx)
      INT_ARG("-numpcy", numpcy)
    }

    switch(mesh_type) {
    case RectangularMesh:
      {
	npx = nzx + 1;
	npy = nzy + 1;
	numpc = numpcx * numpcy;

	zxbound.resize(numpcx + 1);
	for(int i = 0; i <= numpcx; i++)
	  zxbound[i] = (i * nzx) / numpcx;

	zybound.resize(numpcy + 1);
	for(int i = 0; i <= numpcy; i++)
	  zybound[i] = (i * nzy) / numpcy;

	nz = ns = np = 0;
	for(int pcy = 0; pcy < numpcy; pcy++) {
	  for(int pcx = 0; pcx < numpcx; pcx++) {
	    int lx = zxbound[pcx + 1] - zxbound[pcx];
	    int ly = zybound[pcy + 1] - zybound[pcy];

	    int zones = lx * ly;
	    int sides = zones * 4;
	    // points are a little funny - shared edges go to the lower numbered piece
	    int points = ((pcx == 0) ? (lx + 1) : lx) * ((pcy == 0) ? (ly + 1) : ly);

	    lz.push_back(zones);
	    ls.push_back(sides);
	    lp.push_back(points);
	    nz += zones;
	    ns += sides;
	    np += points;
	  }
	}

	assert(nz == (nzx * nzy));
	assert(ns == (4 * nzx * nzy));
	assert(np == (npx * npy));

	break;
      }
    }
  }

  virtual void print_info(void)
  {
    printf("Realm dependent partitioning test - pennant: %d x %d zones, %d x %d pieces\n",
	   (int)nzx, (int)nzy, (int)numpcx, (int)numpcy);
  }

  ZIndexSpace<1> is_zones, is_sides, is_points;
  std::vector<RegionInstance> ri_zones;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, int> > zone_color_field_data;
  std::vector<RegionInstance> ri_sides;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, ZPoint<1> > > side_mapsz_field_data;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, ZPoint<1> > > side_mapss3_field_data;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, ZPoint<1> > > side_mapsp1_field_data;
  std::vector<FieldDataDescriptor<ZIndexSpace<1>, bool > > side_ok_field_data;

  struct InitDataArgs {
    int index;
    RegionInstance ri_zones, ri_sides;
  };

  virtual Event initialize_data(const std::vector<Memory>& memories,
				const std::vector<Processor>& procs)
  {
    // top level index spaces
    is_zones = ZRect<1>(0, nz - 1);
    is_sides = ZRect<1>(0, ns - 1);
    is_points = ZRect<1>(0, np - 1);

    // weighted partitions based on the distribution we already computed
    std::vector<ZIndexSpace<1> > ss_zones_w;
    std::vector<ZIndexSpace<1> > ss_sides_w;
    std::vector<ZIndexSpace<1> > ss_points_w;

    is_zones.create_weighted_subspaces(numpc, 1, lz, ss_zones_w, Realm::ProfilingRequestSet()).wait();
    is_sides.create_weighted_subspaces(numpc, 1, ls, ss_sides_w, Realm::ProfilingRequestSet()).wait();
    is_points.create_weighted_subspaces(numpc, 1, lp, ss_points_w, Realm::ProfilingRequestSet()).wait();

    std::cout << "Initial partitions:" << std::endl;
    for(size_t i = 0; i < ss_zones_w.size(); i++)
      std::cout << " Zones #" << i << ": " << ss_zones_w[i] << std::endl;
    for(size_t i = 0; i < ss_sides_w.size(); i++)
      std::cout << " Sides #" << i << ": " << ss_sides_w[i] << std::endl;
    for(size_t i = 0; i < ss_points_w.size(); i++)
      std::cout << " Points #" << i << ": " << ss_points_w[i] << std::endl;

    // create instances for each of these subspaces
    std::vector<size_t> zone_fields, side_fields;
    zone_fields.push_back(sizeof(int));  // color
    assert(sizeof(int) == sizeof(ZPoint<1>));
    side_fields.push_back(sizeof(ZPoint<1>));  // mapsz
    side_fields.push_back(sizeof(ZPoint<1>));  // mapss3
    side_fields.push_back(sizeof(ZPoint<1>));  // mapsp1
    side_fields.push_back(sizeof(bool));  // ok

    ri_zones.resize(numpc);
    zone_color_field_data.resize(numpc);

    for(size_t i = 0; i < ss_zones_w.size(); i++) {
      RegionInstance ri = ss_zones_w[i].create_instance(memories[i % memories.size()],
							 zone_fields,
							 1,
							 Realm::ProfilingRequestSet());
      ri_zones[i] = ri;
    
      zone_color_field_data[i].index_space = ss_zones_w[i];
      zone_color_field_data[i].inst = ri_zones[i];
      zone_color_field_data[i].field_offset = 0;
    }

    ri_sides.resize(numpc);
    side_mapsz_field_data.resize(numpc);
    side_mapss3_field_data.resize(numpc);
    side_mapsp1_field_data.resize(numpc);
    side_ok_field_data.resize(numpc);

    for(size_t i = 0; i < ss_sides_w.size(); i++) {
      RegionInstance ri = ss_sides_w[i].create_instance(memories[i % memories.size()],
							 side_fields,
							 1,
							 Realm::ProfilingRequestSet());
      ri_sides[i] = ri;

      side_mapsz_field_data[i].index_space = ss_sides_w[i];
      side_mapsz_field_data[i].inst = ri_sides[i];
      side_mapsz_field_data[i].field_offset = 0 * sizeof(ZPoint<1>);
      
      side_mapss3_field_data[i].index_space = ss_sides_w[i];
      side_mapss3_field_data[i].inst = ri_sides[i];
      side_mapss3_field_data[i].field_offset = 1 * sizeof(ZPoint<1>);

      side_mapsp1_field_data[i].index_space = ss_sides_w[i];
      side_mapsp1_field_data[i].inst = ri_sides[i];
      side_mapsp1_field_data[i].field_offset = 2 * sizeof(ZPoint<1>);
      
      side_ok_field_data[i].index_space = ss_sides_w[i];
      side_ok_field_data[i].inst = ri_sides[i];
      side_ok_field_data[i].field_offset = 3 * sizeof(ZPoint<1>);
    }

    // fire off tasks to initialize data
    std::set<Event> events;
    for(int i = 0; i < numpc; i++) {
      Processor p = procs[i % memories.size()];
      InitDataArgs args;
      args.index = i;
      args.ri_zones = ri_zones[i];
      args.ri_sides = ri_sides[i];
      Event e = p.spawn(INIT_PENNANT_DATA_TASK, &args, sizeof(args));
      events.insert(e);
    }

    return Event::merge_events(events);
  }

  static void init_data_task_wrapper(const void *args, size_t arglen, Processor p)
  {
    PennantTest *me = (PennantTest *)testcfg;
    me->init_data_task(args, arglen, p);
  }

  ZPoint<1> global_point_pointer(int py, int px) const
  {
    int pp = 0;

    // start by steping over whole y slabs - again be careful that the extra slab belongs to pcy == 0
    int dy;
    if(py > zybound[1]) {
      int pcy = 1;
      while(py > zybound[pcy + 1]) pcy++;
      int slabs = zybound[pcy] + 1;
      pp += npx * slabs;
      py -= slabs;
      dy = zybound[pcy + 1] - zybound[pcy];
    } else {
      dy = zybound[1] + 1;
    }

    // now chunks in x, using just the y width of this row of chunks
    int dx;
    if(px > zxbound[1]) {
      int pcx = 1;
      while(px > zxbound[pcx + 1]) pcx++;
      int strips = zxbound[pcx] + 1;
      pp += dy * strips;
      px -= strips;
      dx = zxbound[pcx + 1] - zxbound[pcx];
    } else {
      dx = zxbound[1] + 1;
    }

    // finally, px and py are now local and are handled easily
    pp += py * dx + px;

    return pp;
  }

  void init_data_task(const void *args, size_t arglen, Processor p)
  {
    const InitDataArgs& i_args = *(const InitDataArgs *)args;

    log_app.print() << "init task #" << i_args.index << " (ri_zones=" << i_args.ri_zones << ", ri_sides=" << i_args.ri_sides << ")";

    ZIndexSpace<1> is_zones = i_args.ri_zones.get_indexspace<1>();
    ZIndexSpace<1> is_sides = i_args.ri_sides.get_indexspace<1>();

    log_app.print() << "Z: " << is_zones;
    log_app.print() << "S: " << is_sides;
    
    int pcx = i_args.index % numpcx;
    int pcy = i_args.index / numpcx;

    int zxlo = zxbound[pcx];
    int zxhi = zxbound[pcx + 1];
    int zylo = zybound[pcy];
    int zyhi = zybound[pcy + 1];

    {
      AffineAccessor<int,1> a_zone_color(i_args.ri_zones, 0 /* offset */);
      AffineAccessor<ZPoint<1>,1> a_side_mapsz(i_args.ri_sides, 0 * sizeof(ZPoint<1>) /* offset */);
      AffineAccessor<ZPoint<1>,1> a_side_mapss3(i_args.ri_sides, 1 * sizeof(ZPoint<1>) /* offset */);
      AffineAccessor<ZPoint<1>,1> a_side_mapsp1(i_args.ri_sides, 2 * sizeof(ZPoint<1>) /* offset */);
      AffineAccessor<bool,1> a_side_ok(i_args.ri_sides, 3 * sizeof(ZPoint<1>) /* offset */);
      
      ZPoint<1> pz = is_zones.bounds.lo;
      ZPoint<1> ps = is_sides.bounds.lo;

      for(int zy = zylo; zy < zyhi; zy++) {
	for(int zx = zxlo; zx < zxhi; zx++) {
	  // get 4 side pointers
	  ZPoint<1> ps0 = ps; ps.x++;
	  ZPoint<1> ps1 = ps; ps.x++;
	  ZPoint<1> ps2 = ps; ps.x++;
	  ZPoint<1> ps3 = ps; ps.x++;

	  // point pointers are ugly because they can be in neighbors - use a helper
	  ZPoint<1> pp0 = global_point_pointer(zy, zx); // go CCW
	  ZPoint<1> pp1 = global_point_pointer(zy+1, zx);
	  ZPoint<1> pp2 = global_point_pointer(zy+1, zx+1);
	  ZPoint<1> pp3 = global_point_pointer(zy, zx+1);

	  a_zone_color.write(pz, i_args.index);

	  a_side_mapsz.write(ps0, pz);
	  a_side_mapsz.write(ps1, pz);
	  a_side_mapsz.write(ps2, pz);
	  a_side_mapsz.write(ps3, pz);

	  a_side_mapss3.write(ps0, ps1);
	  a_side_mapss3.write(ps1, ps2);
	  a_side_mapss3.write(ps2, ps3);
	  a_side_mapss3.write(ps3, ps0);

	  a_side_mapsp1.write(ps0, pp0);
	  a_side_mapsp1.write(ps1, pp1);
	  a_side_mapsp1.write(ps2, pp2);
	  a_side_mapsp1.write(ps3, pp3);

	  a_side_ok.write(ps0, true);
	  a_side_ok.write(ps1, true);
	  a_side_ok.write(ps2, true);
	  a_side_ok.write(ps3, true);

	  pz.x++;
	}
      }
      assert(pz.x == is_zones.bounds.hi.x + 1);
      assert(ps.x == is_sides.bounds.hi.x + 1);
    }
    
    if(show_graph) {
      AffineAccessor<int,1> a_zone_color(i_args.ri_zones, 0 /* offset */);

      for(int i = is_zones.bounds.lo; i <= is_zones.bounds.hi; i++)
	std::cout << "Z[" << i << "]: color=" << a_zone_color.read(i) << std::endl;

      AffineAccessor<ZPoint<1>,1> a_side_mapsz(i_args.ri_sides, 0 * sizeof(ZPoint<1>) /* offset */);
      AffineAccessor<ZPoint<1>,1> a_side_mapss3(i_args.ri_sides, 1 * sizeof(ZPoint<1>) /* offset */);
      AffineAccessor<ZPoint<1>,1> a_side_mapsp1(i_args.ri_sides, 2 * sizeof(ZPoint<1>) /* offset */);
      AffineAccessor<bool,1> a_side_ok(i_args.ri_sides, 3 * sizeof(ZPoint<1>) /* offset */);

      for(int i = is_sides.bounds.lo; i <= is_sides.bounds.hi; i++)
	std::cout << "S[" << i << "]:"
		  << " mapsz=" << a_side_mapsz.read(i)
		  << " mapss3=" << a_side_mapss3.read(i)
		  << " mapsp1=" << a_side_mapsp1.read(i)
		  << " ok=" << a_side_ok.read(i)
		  << std::endl;
    }
  }

  // the outputs of our partitioning will be:
  //  p_zones               - subsets of is_zones split by piece
  //  p_sides               - subsets of is_sides split by piece (with bad sides removed)
  //  p_points              - subsets of is_points by piece (aliased)

  std::vector<ZIndexSpace<1> > p_zones;
  std::vector<ZIndexSpace<1> > p_sides;
  std::vector<ZIndexSpace<1> > p_points;

  virtual Event perform_partitioning(void)
  {
    // first get the set of bad sides (i.e. ok == false)
    ZIndexSpace<1> bad_sides;

    Event e1 = is_sides.create_subspace_by_field(side_ok_field_data,
						 false,
						 bad_sides,
						 Realm::ProfilingRequestSet());
    if(wait_on_events) e1.wait();

    // map the bad sides through to bad zones
    ZIndexSpace<1> bad_zones;
    Event e2 = is_zones.create_subspace_by_image(side_mapsz_field_data,
						 bad_sides,
						 bad_zones,
						 Realm::ProfilingRequestSet(),
						 e1);
    if(wait_on_events) e2.wait();

    // subtract bad zones to get good zones
    ZIndexSpace<1> good_zones;
    Event e3 = ZIndexSpace<1>::compute_difference(is_zones, bad_zones, good_zones,
						  Realm::ProfilingRequestSet(),
						  e2);
    if(wait_on_events) e3.wait();

    // now do actual partitions with just good zones
    std::vector<int> colors(numpc);
    for(int i = 0; i < numpc; i++)
      colors[i] = i;

    Event e4 = good_zones.create_subspaces_by_field(zone_color_field_data,
						    colors,
						    p_zones,
						    Realm::ProfilingRequestSet(),
						    e3);
    if(wait_on_events) e4.wait();

    // preimage of zones is sides
    Event e5 = is_sides.create_subspaces_by_preimage(side_mapsz_field_data,
						     p_zones,
						     p_sides,
						     Realm::ProfilingRequestSet(),
						     e4);
    if(wait_on_events) e5.wait();

    // and image of sides->mapsp1 is points
    Event e6 = is_points.create_subspaces_by_image(side_mapsp1_field_data,
						   p_sides,
						   p_points,
						   Realm::ProfilingRequestSet(),
						   e5);
    if(wait_on_events) e6.wait();

    return e6;
  }

  virtual int check_partitioning(void)
  {
    int errors = 0;

    for(int pcy = 0; pcy < numpcy; pcy++) {
      for(int pcx = 0; pcx < numpcx; pcx++) {
	int idx = pcy * numpcx + pcx;

	int lx = zxbound[pcx + 1] - zxbound[pcx];
	int ly = zybound[pcy + 1] - zybound[pcy];

	int exp_zones = lx * ly;
	int exp_sides = exp_zones * 4;
	int exp_points = (lx + 1) * (ly + 1); // easier because of aliasing

	int act_zones = p_zones[idx].volume();
	int act_sides = p_sides[idx].volume();
	int act_points = p_points[idx].volume();

	if(exp_zones != act_zones) {
	  log_app.error() << "Piece #" << idx << ": zone count mismatch: exp = " << exp_zones << ", act = " << act_zones;
	  errors++;
	}
	if(exp_sides != act_sides) {
	  log_app.error() << "Piece #" << idx << ": side count mismatch: exp = " << exp_sides << ", act = " << act_sides;
	  errors++;
	}
	if(exp_points != act_points) {
	  log_app.error() << "Piece #" << idx << ": point count mismatch: exp = " << exp_points << ", act = " << act_points;
	  errors++;
	}
      }
    }
    
    return errors;
  }
};


void top_level_task(const void *args, size_t arglen, Processor p)
{
  int errors = 0;

  testcfg->print_info();

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

  Event e = testcfg->initialize_data(sysmems, procs);
  // wait for all initialization to be done
  e.wait();

  // now actual partitioning work
  {
    Realm::TimeStamp ts("dependent partitioning work", true, &log_app);

    Event e = testcfg->perform_partitioning();

    e.wait();
  }

  errors += testcfg->check_partitioning();

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

  //testcfg = new CircuitTest(argc, (const char **)argv);
  testcfg = new PennantTest(argc, (const char **)argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(INIT_CIRCUIT_DATA_TASK, CircuitTest::init_data_task_wrapper);
  rt.register_task(INIT_PENNANT_DATA_TASK, PennantTest::init_data_task_wrapper);

  signal(SIGALRM, sigalrm_handler);

  // Start the machine running
  // Control never returns from this call
  // Note we only run the top level task on one processor
  // You can also run the top level task on all processors or one processor per node
  rt.run(TOP_LEVEL_TASK, Runtime::ONE_TASK_ONLY);

  //rt.shutdown();
  return 0;
}
