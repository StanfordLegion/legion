/* Copyright 2012 Stanford University
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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <time.h>

#include "circuit.h"
#include "circuit_mapper.h"
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

LegionRuntime::Logger::Category log_circuit("circuit");

// Utility functions (forward declarations)
void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed,
		      int &sync);

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        HighLevelRuntime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed);

void registration_func(Machine *machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs);

template<typename T>
FieldID allocate_field(Context ctx, HighLevelRuntime *runtime, FieldSpace space);

// Top level task

void region_main(const void *args, size_t arglen,
                 const std::vector<RegionRequirement> &logical_regions,
                 const std::vector<PhysicalRegion> &physical_regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  int num_loops = 2;
  int num_pieces = 4;
  int nodes_per_piece = 2;
  int wires_per_piece = 4;
  int pct_wire_in_piece = 95;
  int random_seed = 12345;
  int sync = 0;
  {
    InputArgs *inputs = (InputArgs*)args;
    char **argv = inputs->argv;
    int argc = inputs->argc;

    parse_input_args(argv, argc, num_loops, num_pieces, nodes_per_piece, 
		     wires_per_piece, pct_wire_in_piece, random_seed,
		     sync);

    log_circuit(LEVEL_PRINT,"circuit settings: loops=%d pieces=%d nodes/piece=%d wires/piece=%d pct_in_piece=%d seed=%d",
       num_loops, num_pieces, nodes_per_piece, wires_per_piece,
       pct_wire_in_piece, random_seed);
  }

  Circuit circuit;
  {
    int num_circuit_nodes = num_pieces * nodes_per_piece;
    int num_circuit_wires = num_pieces * wires_per_piece;
    // Make index spaces
    IndexSpace node_index_space = runtime->create_index_space(ctx,num_circuit_nodes);
    IndexSpace wire_index_space = runtime->create_index_space(ctx,num_circuit_wires);
    // Make field spaces
    FieldSpace node_field_space = runtime->create_field_space(ctx);
    FieldSpace wire_field_space = runtime->create_field_space(ctx);
    FieldSpace locator_field_space = runtime->create_field_space(ctx);
    // Allocate fields
    circuit.node_field = allocate_field<CircuitNode>(ctx,runtime,node_field_space);
    circuit.wire_field = allocate_field<CircuitWire>(ctx,runtime,wire_field_space);
    circuit.locator_field = allocate_field<PointerLocation>(ctx,runtime,locator_field_space);
    // Make logical regions
    circuit.all_nodes = runtime->create_logical_region(ctx,node_index_space,node_field_space);
    circuit.all_wires = runtime->create_logical_region(ctx,wire_index_space,wire_field_space);
    circuit.node_locator = runtime->create_logical_region(ctx,node_index_space,locator_field_space);
  }

  // Load the circuit
  std::vector<CircuitPiece> pieces(num_pieces);
  Partitions parts = load_circuit(circuit, pieces, ctx, runtime, num_pieces, nodes_per_piece,
                                  wires_per_piece, pct_wire_in_piece, random_seed);

  // Start the simulation
  LegionRuntime::LowLevel::DetailedTimer::clear_timers();
  printf("Starting main simulation loop\n");
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  // Build the region requirements for each task
  std::vector<RegionRequirement> cnc_regions;
  cnc_regions.push_back(RegionRequirement(parts.pvt_wires, 0/*identity colorize function*/,
                        READ_WRITE, EXCLUSIVE, circuit.all_wires));
  cnc_regions.back().add_field(circuit.wire_field);
  cnc_regions.push_back(RegionRequirement(parts.pvt_nodes, 0/*identity*/,
                        READ_ONLY, EXCLUSIVE, circuit.all_nodes));
  cnc_regions.back().add_field(circuit.node_field);
  cnc_regions.push_back(RegionRequirement(parts.shr_nodes, 0/*identity*/,
                        READ_ONLY, EXCLUSIVE, circuit.all_nodes));
  cnc_regions.back().add_field(circuit.node_field);
  cnc_regions.push_back(RegionRequirement(parts.ghost_nodes, 0/*identity*/,
                        READ_ONLY, EXCLUSIVE, circuit.all_nodes));
  cnc_regions.back().add_field(circuit.node_field);

  std::vector<RegionRequirement> dsc_regions;
  dsc_regions.push_back(RegionRequirement(parts.pvt_wires, 0/*identity*/,
                        READ_ONLY, EXCLUSIVE, circuit.all_wires));
  dsc_regions.back().add_field(circuit.wire_field);
  dsc_regions.push_back(RegionRequirement(parts.pvt_nodes, 0/*identity*/,
                        READ_WRITE, EXCLUSIVE, circuit.all_nodes));
  dsc_regions.back().add_field(circuit.node_field);
  dsc_regions.push_back(RegionRequirement(parts.shr_nodes, 0/*identity*/,
                        REDUCE_ID, EXCLUSIVE, circuit.all_nodes));
  dsc_regions.back().add_field(circuit.node_field);
  dsc_regions.push_back(RegionRequirement(parts.ghost_nodes, 0/*identity*/,
                        REDUCE_ID, EXCLUSIVE, circuit.all_nodes));
  dsc_regions.back().add_field(circuit.node_field);

  std::vector<RegionRequirement> upv_regions;
  upv_regions.push_back(RegionRequirement(parts.pvt_nodes, 0/*identity*/,
                        READ_WRITE, EXCLUSIVE, circuit.all_nodes));
  upv_regions.back().add_field(circuit.node_field);
  upv_regions.push_back(RegionRequirement(parts.shr_nodes, 0/*identity*/,
                        READ_WRITE, EXCLUSIVE, circuit.all_nodes));
  upv_regions.back().add_field(circuit.node_field);
  upv_regions.push_back(RegionRequirement(parts.node_locations, 0/*identity*/,
                        READ_ONLY, EXCLUSIVE, circuit.node_locator));
  upv_regions.back().add_field(circuit.locator_field);

  IndexSpace task_space = runtime->create_index_space(ctx, num_pieces);
  {
    IndexAllocator allocator = runtime->create_index_allocator(ctx, task_space);
    allocator.alloc(num_pieces);
  }

  TaskArgument global_arg;
  std::vector<IndexSpaceRequirement> index_space_reqs;
  std::vector<FieldSpaceRequirement> field_space_reqs;
  ArgumentMap local_args = runtime->create_argument_map(ctx);
  for (int idx = 0; idx < num_pieces; idx++)
  {
    int point[1];
    point[0] = idx;
    local_args.set_point_arg<int,1>(point, TaskArgument(&(pieces[idx]),sizeof(CircuitPiece)));
  }

  FutureMap last;
  for (int i = 0; i < num_loops; i++)
  {
    log_circuit(LEVEL_PRINT,"starting loop %d of %d", i, num_loops);

    // Calculate new currents
    runtime->execute_index_space(ctx, CALC_NEW_CURRENTS, task_space,
                                  index_space_reqs, field_space_reqs, cnc_regions,
                                  global_arg, local_args);

    // Distribute charge
    runtime->execute_index_space(ctx, DISTRIBUTE_CHARGE, task_space,
                                  index_space_reqs, field_space_reqs, dsc_regions,
                                  global_arg, local_args);

    // Update voltages
    last = runtime->execute_index_space(ctx, UPDATE_VOLTAGES, task_space,
                                  index_space_reqs, field_space_reqs, upv_regions,
                                  global_arg, local_args);
  }

  log_circuit(LEVEL_PRINT,"waiting for all simulation tasks to complete");
  last.wait_all_results();

  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  log_circuit(LEVEL_PRINT,"SUCCESS!");
  {
    double sim_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                       (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
    printf("ELAPSED TIME = %7.3f s\n", sim_time);

    // Compute the floating point operations per second
    long num_circuit_nodes = num_pieces * nodes_per_piece;
    long num_circuit_wires = num_pieces * wires_per_piece;
    // calculate currents
    long operations = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * STEPS;
    // distribute charge
    operations += (num_circuit_wires * 4);
    // update voltages
    operations += (num_circuit_nodes * 4);
    // multiply by the number of loops
    operations *= num_loops;

    // Compute the number of gflops
    double gflops = (1e-9*operations)/sim_time;
    printf("GFLOPS = %7.3f GFLOPS\n", gflops);
  }
  LegionRuntime::LowLevel::DetailedTimer::report_timers();

  log_circuit(LEVEL_PRINT,"simulation complete - destroying regions");

  // Now we can destroy all the things that we made
  {
    runtime->destroy_logical_region(ctx,circuit.all_nodes);
    runtime->destroy_logical_region(ctx,circuit.all_wires);
    runtime->destroy_logical_region(ctx,circuit.node_locator);
    runtime->destroy_field_space(ctx,circuit.all_nodes.get_field_space());
    runtime->destroy_field_space(ctx,circuit.all_wires.get_field_space());
    runtime->destroy_field_space(ctx,circuit.node_locator.get_field_space());
    runtime->destroy_index_space(ctx,circuit.all_nodes.get_index_space());
    runtime->destroy_index_space(ctx,circuit.all_wires.get_index_space());
    runtime->destroy_index_space(ctx,task_space);
  }
}

// CPU wrappers

void calculate_currents_task_cpu(const void *global_args, size_t global_arglen,
                                 const void *local_args, size_t local_arglen,
                                 const unsigned point[1],
                                 const std::vector<RegionRequirement> &logical_regions,
                                 const std::vector<PhysicalRegion> &physical_regions,
                                 Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_PRINT,"CPU calculate currents for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  calc_new_currents_cpu(p, physical_regions);
}

void distribute_charge_task_cpu(const void *global_args, size_t global_arglen,
                                const void *local_args, size_t local_arglen,
                                const unsigned point[1],
                                const std::vector<RegionRequirement> &logical_regions,
                                const std::vector<PhysicalRegion> &physical_regions,
                                Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_PRINT,"CPU distribute charge for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  distribute_charge_cpu(p, physical_regions);
}

void update_voltages_task_cpu(const void *global_args, size_t global_arglen,
                              const void *local_args, size_t local_arglen,
                              const unsigned point[1],
                              const std::vector<RegionRequirement> &logical_regions,
                              const std::vector<PhysicalRegion> &physical_regions,
                              Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_PRINT,"CPU update voltages for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  update_voltages_cpu(p, physical_regions);
}

// GPU wrappers
#ifndef USING_SHARED
void calculate_currents_task_gpu(const void *global_args, size_t global_arglen,
                                 const void *local_args, size_t local_arglen,
                                 const unsigned point[1],
                                 const std::vector<RegionRequirement> &logical_regions,
                                 const std::vector<PhysicalRegion> &physical_regions,
                                 Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_PRINT,"GPU calculate currents for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  // Call the __host__ function in circuit_gpu.cc that launches the kernel
  calc_new_currents_gpu(p, physical_regions);
}

void distribute_charge_task_gpu(const void *global_args, size_t global_arglen,
                                const void *local_args, size_t local_arglen,
                                const unsigned point[1],
                                const std::vector<RegionRequirement> &logical_regions,
                                const std::vector<PhysicalRegion> &physical_regions,
                                Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_PRINT,"GPU distribute charge for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  distribute_charge_gpu(p, physical_regions);
}

void update_voltages_task_gpu(const void *global_args, size_t global_arglen,
                              const void *local_args, size_t local_arglen,
                              const unsigned point[1],
                              const std::vector<RegionRequirement> &logical_regions,
                              const std::vector<PhysicalRegion> &physical_regions,
                              Context ctx, HighLevelRuntime *runtime)
{
  log_circuit(LEVEL_PRINT,"GPU update voltages for point %d",point[0]);
  CircuitPiece *p = (CircuitPiece*)local_args;
  update_voltages_gpu(p, physical_regions);
}
#endif

// Call back function for running on the GPUs
void mapper_callback_function(Machine *machine, HighLevelRuntime *rt, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new CircuitMapper(machine, rt, *it), *it); 
  }
}

/// Start-up 

int main(int argc, char **argv)
{
  HighLevelRuntime::set_registration_callback(registration_func);
  HighLevelRuntime::set_top_level_task_id(REGION_MAIN);
  // CPU variants
  HighLevelRuntime::register_single_task<region_main>
          (REGION_MAIN, Processor::LOC_PROC, false/*leaf*/, "region_main");
  HighLevelRuntime::register_index_task<INDEX_TYPE,INDEX_DIM,calculate_currents_task_cpu>
          (CALC_NEW_CURRENTS, Processor::LOC_PROC, true/*leaf*/, "calc_new_currents");
  HighLevelRuntime::register_index_task<INDEX_TYPE,INDEX_DIM,distribute_charge_task_cpu>
          (DISTRIBUTE_CHARGE, Processor::LOC_PROC, true/*leaf*/, "distribute_charge");
  HighLevelRuntime::register_index_task<INDEX_TYPE,INDEX_DIM,update_voltages_task_cpu>
          (UPDATE_VOLTAGES, Processor::LOC_PROC, true/*leaf*/, "update_voltages");
#ifndef USING_SHARED
  // GPU variants
  HighLevelRuntime::register_index_task<INDEX_TYPE,INDEX_DIM,calculate_currents_task_gpu>
          (CALC_NEW_CURRENTS, Processor::TOC_PROC, true/*leaf*/, "calc_new_currents");
  HighLevelRuntime::register_index_task<INDEX_TYPE,INDEX_DIM,distribute_charge_task_gpu>
          (DISTRIBUTE_CHARGE, Processor::TOC_PROC, true/*leaf*/, "distribute_charge");
  HighLevelRuntime::register_index_task<INDEX_TYPE,INDEX_DIM,update_voltages_task_gpu>
          (UPDATE_VOLTAGES, Processor::TOC_PROC, true/*leaf*/, "update_voltages");

  // Register the callback function for using the custom CircuitMapper
  HighLevelRuntime::set_registration_callback(mapper_callback_function);
#endif

  // Register reduction op
  HighLevelRuntime::register_reduction_op<AccumulateCharge>(REDUCE_ID);

  return HighLevelRuntime::start(argc, argv);
}

// Utility function implementations

void registration_func(Machine *machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{

}

void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed,
		      int &sync)
{
  for (int i = 1; i < argc; i++) 
  {
    if (!strcmp(argv[i], "-l")) 
    {
      num_loops = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-p")) 
    {
      num_pieces = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-npp")) 
    {
      nodes_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-wpp")) 
    {
      wires_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-pct")) 
    {
      pct_wire_in_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-s")) 
    {
      random_seed = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-sync")) 
    {
      sync = atoi(argv[++i]);
      continue;
    }
  }
}

template<typename T>
FieldID allocate_field(Context ctx, HighLevelRuntime *runtime, FieldSpace space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, space);
  return allocator.allocate_field(sizeof(T));
}

template<typename T>
static T random_element(const std::set<T> &set)
{
  int index = int(drand48() * set.size());
  typename std::set<T>::const_iterator it = set.begin();
  while (index-- > 0) it++;
  return *it;
}

PointerLocation find_location(ptr_t ptr, const std::set<ptr_t> &private_nodes,
                              const std::set<ptr_t> &shared_nodes, const std::set<ptr_t> &ghost_nodes)
{
  if (private_nodes.find(ptr) != private_nodes.end())
  {
    return PRIVATE_PTR;
  }
  else if (shared_nodes.find(ptr) != shared_nodes.end())
  {
    return SHARED_PTR;
  }
  else if (ghost_nodes.find(ptr) != ghost_nodes.end())
  {
    return GHOST_PTR;
  }
  // Should never make it here, if we do something bad happened
  assert(false);
  return PRIVATE_PTR;
}

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        HighLevelRuntime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed)
{
  log_circuit(LEVEL_PRINT,"Initializing circuit simulation...");
  // inline map physical instances for the nodes and wire regions
  RegionRequirement wires_req(ckt.all_wires, READ_WRITE, EXCLUSIVE, ckt.all_wires);
  wires_req.add_field(ckt.wire_field);
  RegionRequirement nodes_req(ckt.all_nodes, READ_WRITE, EXCLUSIVE, ckt.all_nodes);
  nodes_req.add_field(ckt.node_field);
  RegionRequirement locator_req(ckt.node_locator, READ_WRITE, EXCLUSIVE, ckt.node_locator);
  locator_req.add_field(ckt.locator_field);
  PhysicalRegion wires = runtime->map_region(ctx, wires_req);
  PhysicalRegion nodes = runtime->map_region(ctx, nodes_req);
  PhysicalRegion locator = runtime->map_region(ctx, locator_req);

  Coloring wire_owner_map;
  Coloring private_node_map;
  Coloring shared_node_map;
  Coloring ghost_node_map;
  Coloring locator_node_map;

  Coloring privacy_map;

  srand48(random_seed);

  nodes.wait_until_valid();
  RegionAccessor<AccessorType::Generic, CircuitNode> nodes_acc = nodes.get_accessor().typeify<CircuitNode>(); 
  locator.wait_until_valid();
  RegionAccessor<AccessorType::Generic, PointerLocation> locator_acc = locator.get_accessor().typeify<PointerLocation>();
  ptr_t *first_nodes = new ptr_t[num_pieces];
  {
    IndexAllocator node_allocator = runtime->create_index_allocator(ctx, ckt.all_nodes.get_index_space());
    node_allocator.alloc(num_pieces * nodes_per_piece);
  }
  {
    IndexIterator itr(ckt.all_nodes.get_index_space());
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        assert(itr.has_next());
        ptr_t node_ptr = itr.next();
        if (i == 0)
          first_nodes[n] = node_ptr;
        CircuitNode node;
        node.charge = 0.f;
        node.voltage = 2*drand48() - 1;
        node.capacitance = drand48() + 1;
        node.leakage = 0.1f * drand48();

        nodes_acc.write(node_ptr, node);

        // Just put everything in everyones private map at the moment       
        // We'll pull pointers out of here later as nodes get tied to 
        // wires that are non-local
        private_node_map[n].points.insert(node_ptr);
        privacy_map[0].points.insert(node_ptr);
        locator_node_map[n].points.insert(node_ptr);
      }
    }
  }

  wires.wait_until_valid();
  RegionAccessor<AccessorType::Generic, CircuitWire> wires_acc = wires.get_accessor().typeify<CircuitWire>();
  ptr_t *first_wires = new ptr_t[num_pieces];
  // Allocate all the wires
  {
    IndexAllocator wire_allocator = runtime->create_index_allocator(ctx, ckt.all_wires.get_index_space());
    wire_allocator.alloc(num_pieces * wires_per_piece);
  }
  {
    IndexIterator itr(ckt.all_wires.get_index_space());
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < wires_per_piece; i++)
      {
        assert(itr.has_next());
        ptr_t wire_ptr = itr.next();
        // Record the first wire pointer for this piece
        if (i == 0)
          first_wires[n] = wire_ptr;
        CircuitWire wire;
        for (int j = 0; j < WIRE_SEGMENTS; j++) wire.current[j] = 0.f;
        for (int j = 0; j < WIRE_SEGMENTS-1; j++) wire.voltage[j] = 0.f;

        wire.resistance = drand48() * 10 + 1;
        wire.inductance = drand48() * 0.01 + 0.1;
        wire.capacitance = drand48() * 0.1;

        wire.in_ptr = random_element(private_node_map[n].points);

        if ((100 * drand48()) < pct_wire_in_piece)
        {
          wire.out_ptr = random_element(private_node_map[n].points);
        }
        else
        {
          // pick a random other piece and a node from there
          int nn = int(drand48() * (num_pieces - 1));
          if(nn >= n) nn++;

          wire.out_ptr = random_element(private_node_map[nn].points); 
          // This node is no longer private
          privacy_map[0].points.erase(wire.out_ptr);
          privacy_map[1].points.insert(wire.out_ptr);
          ghost_node_map[n].points.insert(wire.out_ptr);
        }

        // Write the wire
        wires_acc.write(wire_ptr, wire);

        wire_owner_map[n].points.insert(wire_ptr);
      }
    }
  }

  // Second pass: make some random fraction of the private nodes shared
  {
    IndexIterator itr(ckt.all_nodes.get_index_space()); 
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        assert(itr.has_next());
        ptr_t node_ptr = itr.next();
        if (privacy_map[0].points.find(node_ptr) == privacy_map[0].points.end())
        {
          private_node_map[n].points.erase(node_ptr);
          // node is now shared
          shared_node_map[n].points.insert(node_ptr);
          locator_acc.write(node_ptr,SHARED_PTR); // node is shared 
        }
        else
        {
          locator_acc.write(node_ptr,PRIVATE_PTR); // node is private 
        }
      }
    }
  }
  // Second pass (part 2): go through the wires and update the locations
  {
    IndexIterator itr(ckt.all_wires.get_index_space());
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < wires_per_piece; i++)
      {
        assert(itr.has_next());
        ptr_t wire_ptr = itr.next();
        CircuitWire wire = wires_acc.read(wire_ptr);

        wire.in_loc = find_location(wire.in_ptr, private_node_map[n].points, shared_node_map[n].points, ghost_node_map[n].points);     
        wire.out_loc = find_location(wire.out_ptr, private_node_map[n].points, shared_node_map[n].points, ghost_node_map[n].points);

        // Write the wire back
        wires_acc.write(wire_ptr, wire);
      }
    }
  }

  runtime->unmap_region(ctx, wires);
  runtime->unmap_region(ctx, nodes);
  runtime->unmap_region(ctx, locator);

  // Now we can create our partitions and update the circuit pieces

  // first create the privacy partition that splits all the nodes into either shared or private
  IndexPartition privacy_part = runtime->create_index_partition(ctx, ckt.all_nodes.get_index_space(), privacy_map, true/*disjoint*/);
  
  IndexSpace all_private = runtime->get_index_subspace(ctx, privacy_part, 0);
  IndexSpace all_shared  = runtime->get_index_subspace(ctx, privacy_part, 1);
  

  // Now create partitions for each of the subregions
  Partitions result;
  IndexPartition priv = runtime->create_index_partition(ctx, all_private, private_node_map, true/*disjoint*/);
  result.pvt_nodes = runtime->get_logical_partition_by_tree(ctx, priv, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  IndexPartition shared = runtime->create_index_partition(ctx, all_shared, shared_node_map, true/*disjoint*/);
  result.shr_nodes = runtime->get_logical_partition_by_tree(ctx, shared, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  IndexPartition ghost = runtime->create_index_partition(ctx, all_shared, ghost_node_map, false/*disjoint*/);
  result.ghost_nodes = runtime->get_logical_partition_by_tree(ctx, ghost, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());

  IndexPartition pvt_wires = runtime->create_index_partition(ctx, ckt.all_wires.get_index_space(), wire_owner_map, true/*disjoint*/);
  result.pvt_wires = runtime->get_logical_partition_by_tree(ctx, pvt_wires, ckt.all_wires.get_field_space(), ckt.all_wires.get_tree_id()); 

  IndexPartition locs = runtime->create_index_partition(ctx, ckt.node_locator.get_index_space(), locator_node_map, true/*disjoint*/);
  result.node_locations = runtime->get_logical_partition_by_tree(ctx, locs, ckt.node_locator.get_field_space(), ckt.node_locator.get_tree_id());

  // Build the pieces
  for (int n = 0; n < num_pieces; n++)
  {
    pieces[n].pvt_nodes = runtime->get_logical_subregion_by_color(ctx, result.pvt_nodes, n);
    pieces[n].shr_nodes = runtime->get_logical_subregion_by_color(ctx, result.shr_nodes, n);
    pieces[n].ghost_nodes = runtime->get_logical_subregion_by_color(ctx, result.ghost_nodes, n);
    pieces[n].pvt_wires = runtime->get_logical_subregion_by_color(ctx, result.pvt_wires, n);
    pieces[n].num_wires = wires_per_piece;
    pieces[n].first_wire = first_wires[n];
    pieces[n].num_nodes = nodes_per_piece;
    pieces[n].first_node = first_nodes[n];
  }

  delete [] first_wires;
  delete [] first_nodes;

  log_circuit(LEVEL_PRINT,"Finished initializing simulation...");

  return result;
}

// EOF

