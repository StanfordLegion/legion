/* Copyright 2017 Stanford University
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

Logger log_circuit("circuit");

// Utility functions (forward declarations)
void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed,
                      int &steps, int &sync, bool &perform_checks, bool &dump_values);

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        Runtime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed,
			int steps);

void allocate_node_fields(Context ctx, Runtime *runtime, FieldSpace node_space);
void allocate_wire_fields(Context ctx, Runtime *runtime, FieldSpace wire_space);
void allocate_locator_fields(Context ctx, Runtime *runtime, FieldSpace locator_space);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_loops = 2;
  int num_pieces = 4;
  int nodes_per_piece = 2;
  int wires_per_piece = 4;
  int pct_wire_in_piece = 95;
  int random_seed = 12345;
  int steps = STEPS;
  int sync = 0;
  bool perform_checks = false;
  bool dump_values = false;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;

    parse_input_args(argv, argc, num_loops, num_pieces, nodes_per_piece, 
		     wires_per_piece, pct_wire_in_piece, random_seed,
		     steps, sync, perform_checks, dump_values);

    log_circuit.print("circuit settings: loops=%d pieces=%d nodes/piece=%d "
                            "wires/piece=%d pct_in_piece=%d seed=%d",
       num_loops, num_pieces, nodes_per_piece, wires_per_piece,
       pct_wire_in_piece, random_seed);
  }

  Circuit circuit;
  {
    int num_circuit_nodes = num_pieces * nodes_per_piece;
    int num_circuit_wires = num_pieces * wires_per_piece;
    // Make index spaces
    IndexSpace node_index_space = runtime->create_index_space(ctx,num_circuit_nodes);
    runtime->attach_name(node_index_space, "node_index_space");
    IndexSpace wire_index_space = runtime->create_index_space(ctx,num_circuit_wires);
    runtime->attach_name(wire_index_space, "wire_index_space");
    // Make field spaces
    FieldSpace node_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(node_field_space, "node_field_space");
    FieldSpace wire_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(wire_field_space, "wire_field_space");
    FieldSpace locator_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(locator_field_space, "locator_field_space");
    // Allocate fields
    allocate_node_fields(ctx, runtime, node_field_space);
    allocate_wire_fields(ctx, runtime, wire_field_space);
    allocate_locator_fields(ctx, runtime, locator_field_space);
    // Make logical regions
    circuit.all_nodes = runtime->create_logical_region(ctx,node_index_space,node_field_space);
    runtime->attach_name(circuit.all_nodes, "all_nodes");
    circuit.all_wires = runtime->create_logical_region(ctx,wire_index_space,wire_field_space);
    runtime->attach_name(circuit.all_wires, "all_wires");
    circuit.node_locator = runtime->create_logical_region(ctx,node_index_space,locator_field_space);
    runtime->attach_name(circuit.node_locator, "node_locator");
  }

  // Load the circuit
  std::vector<CircuitPiece> pieces(num_pieces);
  Partitions parts = load_circuit(circuit, pieces, ctx, runtime, num_pieces, nodes_per_piece,
                                  wires_per_piece, pct_wire_in_piece, random_seed, steps);

  // Arguments for each point
  ArgumentMap local_args;
  for (int idx = 0; idx < num_pieces; idx++)
  {
    DomainPoint point(idx);
    local_args.set_point(point, TaskArgument(&(pieces[idx]),sizeof(CircuitPiece)));
  }

  // Make the launchers
  const Rect<1> launch_rect(0, num_pieces-1); 
  CalcNewCurrentsTask cnc_launcher(parts.pvt_wires, parts.pvt_nodes, parts.shr_nodes, parts.ghost_nodes,
                                   circuit.all_wires, circuit.all_nodes, launch_rect, local_args);

  DistributeChargeTask dsc_launcher(parts.pvt_wires, parts.pvt_nodes, parts.shr_nodes, parts.ghost_nodes,
                                    circuit.all_wires, circuit.all_nodes, launch_rect, local_args);

  UpdateVoltagesTask upv_launcher(parts.pvt_nodes, parts.shr_nodes, parts.node_locations,
                                 circuit.all_nodes, circuit.node_locator, launch_rect, local_args);

  printf("Starting main simulation loop\n");
  //struct timespec ts_start, ts_end;
  //clock_gettime(CLOCK_MONOTONIC, &ts_start);
  double ts_start, ts_end;
  ts_start = Realm::Clock::current_time_in_microseconds();
  // Run the main loop
  bool simulation_success = true;
  for (int i = 0; i < num_loops; i++)
  {
    TaskHelper::dispatch_task<CalcNewCurrentsTask>(cnc_launcher, ctx, runtime, 
                                                   perform_checks, simulation_success);
    TaskHelper::dispatch_task<DistributeChargeTask>(dsc_launcher, ctx, runtime, 
                                                    perform_checks, simulation_success);
    TaskHelper::dispatch_task<UpdateVoltagesTask>(upv_launcher, ctx, runtime, 
                                                  perform_checks, simulation_success,
                                                  ((i+1)==num_loops));
  }
  ts_end = Realm::Clock::current_time_in_microseconds();
  if (simulation_success)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
  {
    double sim_time = 1e-6 * (ts_end - ts_start);
    printf("ELAPSED TIME = %7.3f s\n", sim_time);

    // Compute the floating point operations per second
    long num_circuit_nodes = num_pieces * nodes_per_piece;
    long num_circuit_wires = num_pieces * wires_per_piece;
    // calculate currents
    long operations = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * steps;
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
  log_circuit.print("simulation complete - destroying regions");

  if (dump_values)
  {
    RegionRequirement wires_req(circuit.all_wires, READ_ONLY, EXCLUSIVE, circuit.all_wires);
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      wires_req.add_field(FID_CURRENT+i);
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      wires_req.add_field(FID_WIRE_VOLTAGE+i);
    PhysicalRegion wires = runtime->map_region(ctx, wires_req);
    wires.wait_until_valid();
    FieldAccessor<READ_ONLY,float,1> fa_wire_currents[WIRE_SEGMENTS];
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      fa_wire_currents[i] = FieldAccessor<READ_ONLY,float,1>(wires, FID_CURRENT+i);
    FieldAccessor<READ_ONLY,float,1> fa_wire_voltages[WIRE_SEGMENTS-1];
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_wire_voltages[i] = FieldAccessor<READ_ONLY,float,1>(wires, FID_WIRE_VOLTAGE+i);

    for (int i = 0; i < (num_pieces * wires_per_piece); i++)
    {
      const Point<1> wire_ptr(i);
      for (int i = 0; i < WIRE_SEGMENTS; ++i)
        printf(" %.5g", fa_wire_currents[i][wire_ptr]);
      for (int i = 0; i < WIRE_SEGMENTS - 1; ++i)
        printf(" %.5g", fa_wire_voltages[i][wire_ptr]);
      printf("\n");
    }
    runtime->unmap_region(ctx, wires);
  }

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
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  TaskHelper::register_hybrid_variants<CalcNewCurrentsTask>();
  TaskHelper::register_hybrid_variants<DistributeChargeTask>();
  TaskHelper::register_hybrid_variants<UpdateVoltagesTask>();
  CheckTask::register_task();
  Runtime::register_reduction_op<AccumulateCharge>(REDUCE_ID);
  Runtime::add_registration_callback(update_mappers);

  return Runtime::start(argc, argv);
}

void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed,
                      int &steps, int &sync, bool &perform_checks,
                      bool &dump_values)
{
  for (int i = 1; i < argc; i++) 
  {
    if (!strcmp(argv[i], "-l")) 
    {
      num_loops = atoi(argv[++i]);
      continue;
    }

    if (!strcmp(argv[i], "-i")) 
    {
      steps = atoi(argv[++i]);
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

    if(!strcmp(argv[i], "-checks"))
    {
      perform_checks = true;
      continue;
    }

    if(!strcmp(argv[i], "-dump"))
    {
      dump_values = true;
      continue;
    }
  }
}

void allocate_node_fields(Context ctx, Runtime *runtime, FieldSpace node_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, node_space);
  allocator.allocate_field(sizeof(float), FID_NODE_CAP);
  runtime->attach_name(node_space, FID_NODE_CAP, "node capacitance");
  allocator.allocate_field(sizeof(float), FID_LEAKAGE);
  runtime->attach_name(node_space, FID_LEAKAGE, "leakage");
  allocator.allocate_field(sizeof(float), FID_CHARGE);
  runtime->attach_name(node_space, FID_CHARGE, "charge");
  allocator.allocate_field(sizeof(float), FID_NODE_VOLTAGE);
  runtime->attach_name(node_space, FID_NODE_VOLTAGE, "node voltage");
  allocator.allocate_field(sizeof(Point<1>), FID_PIECE_COLOR);
  runtime->attach_name(node_space, FID_PIECE_COLOR, "piece color");
}

void allocate_wire_fields(Context ctx, Runtime *runtime, FieldSpace wire_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, wire_space);
  allocator.allocate_field(sizeof(Point<1>), FID_IN_PTR);
  runtime->attach_name(wire_space, FID_IN_PTR, "in_ptr");
  allocator.allocate_field(sizeof(Point<1>), FID_OUT_PTR);
  runtime->attach_name(wire_space, FID_OUT_PTR, "out_ptr");
  allocator.allocate_field(sizeof(PointerLocation), FID_IN_LOC);
  runtime->attach_name(wire_space, FID_IN_LOC, "in_loc");
  allocator.allocate_field(sizeof(PointerLocation), FID_OUT_LOC);
  runtime->attach_name(wire_space, FID_OUT_LOC, "out_loc");
  allocator.allocate_field(sizeof(float), FID_INDUCTANCE);
  runtime->attach_name(wire_space, FID_INDUCTANCE, "inductance");
  allocator.allocate_field(sizeof(float), FID_RESISTANCE);
  runtime->attach_name(wire_space, FID_RESISTANCE, "resistance");
  allocator.allocate_field(sizeof(float), FID_WIRE_CAP);
  runtime->attach_name(wire_space, FID_WIRE_CAP, "wire capacitance");
  for (int i = 0; i < WIRE_SEGMENTS; i++)
  {
    char field_name[10];
    allocator.allocate_field(sizeof(float), FID_CURRENT+i);
    sprintf(field_name, "current_%d", i);
    runtime->attach_name(wire_space, FID_CURRENT+i, field_name);
  }
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
  {
    char field_name[15];
    allocator.allocate_field(sizeof(float), FID_WIRE_VOLTAGE+i);
    sprintf(field_name, "wire_voltage_%d", i);
    runtime->attach_name(wire_space, FID_WIRE_VOLTAGE+i, field_name);
  }
}

void allocate_locator_fields(Context ctx, Runtime *runtime, FieldSpace locator_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, locator_space);
  allocator.allocate_field(sizeof(PointerLocation), FID_LOCATOR);
  runtime->attach_name(locator_space, FID_LOCATOR, "locator");
}

PointerLocation find_location(Point<1> ptr, const std::set<ptr_t> &private_nodes,
                              const std::set<ptr_t> &shared_nodes, 
                              const std::set<ptr_t> &ghost_nodes)
{
  if (private_nodes.find(ptr_t(ptr)) != private_nodes.end())
  {
    return PRIVATE_PTR;
  }
  else if (shared_nodes.find(ptr_t(ptr)) != shared_nodes.end())
  {
    return SHARED_PTR;
  }
  else if (ghost_nodes.find(ptr_t(ptr)) != ghost_nodes.end())
  {
    return GHOST_PTR;
  }
  // Should never make it here, if we do something bad happened
  assert(false);
  return PRIVATE_PTR;
}

template<typename T>
static T random_element(const std::set<T> &set)
{
  int index = int(drand48() * set.size());
  typename std::set<T>::const_iterator it = set.begin();
  while (index-- > 0) it++;
  return *it;
}

template<typename T>
static T random_element(const std::vector<T> &vec)
{
  int index = int(drand48() * vec.size());
  return vec[index];
}

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        Runtime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed,
			int steps)
{
  log_circuit.print("Initializing circuit simulation...");
  // inline map physical instances for the nodes and wire regions
  RegionRequirement wires_req(ckt.all_wires, READ_WRITE, EXCLUSIVE, ckt.all_wires);
  wires_req.add_field(FID_IN_PTR);
  wires_req.add_field(FID_OUT_PTR);
  wires_req.add_field(FID_IN_LOC);
  wires_req.add_field(FID_OUT_LOC);
  wires_req.add_field(FID_INDUCTANCE);
  wires_req.add_field(FID_RESISTANCE);
  wires_req.add_field(FID_WIRE_CAP);
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    wires_req.add_field(FID_CURRENT+i);
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    wires_req.add_field(FID_WIRE_VOLTAGE+i);
  RegionRequirement nodes_req(ckt.all_nodes, READ_WRITE, EXCLUSIVE, ckt.all_nodes);
  nodes_req.add_field(FID_NODE_CAP);
  nodes_req.add_field(FID_LEAKAGE);
  nodes_req.add_field(FID_CHARGE);
  nodes_req.add_field(FID_NODE_VOLTAGE);
  nodes_req.add_field(FID_PIECE_COLOR);
  RegionRequirement locator_req(ckt.node_locator, READ_WRITE, EXCLUSIVE, ckt.node_locator);
  locator_req.add_field(FID_LOCATOR);
  PhysicalRegion wires = runtime->map_region(ctx, wires_req);
  PhysicalRegion nodes = runtime->map_region(ctx, nodes_req);
  PhysicalRegion locator = runtime->map_region(ctx, locator_req);

  // keep a O(1) indexable list of nodes in each piece for connecting wires
  std::vector<std::vector<Point<1> > > piece_node_ptrs(num_pieces);
  std::vector<int> piece_shared_nodes(num_pieces, 0);

  srand48(random_seed);

  nodes.wait_until_valid();
  const FieldAccessor<READ_WRITE,float,1> fa_node_cap(nodes, FID_NODE_CAP);
  const FieldAccessor<READ_WRITE,float,1> fa_node_leakage(nodes, FID_LEAKAGE);
  const FieldAccessor<READ_WRITE,float,1> fa_node_charge(nodes, FID_CHARGE);
  const FieldAccessor<READ_WRITE,float,1> fa_node_voltage(nodes, FID_NODE_VOLTAGE);
  const FieldAccessor<READ_WRITE,Point<1>,1> fa_node_color(nodes, FID_PIECE_COLOR); 
  Point<1> *first_nodes = new Point<1>[num_pieces];
  {
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        const Point<1> node_ptr(n * nodes_per_piece + i);
        if (i == 0)
          first_nodes[n] = node_ptr;
        float capacitance = drand48() + 1.f;
        fa_node_cap[node_ptr] = capacitance;
        float leakage = 0.1f * drand48();
        fa_node_leakage[node_ptr] = leakage;
        fa_node_charge[node_ptr] = 0.f;
        fa_node_voltage[node_ptr] = 2*drand48() - 1.f;
        fa_node_color[node_ptr] = n;
	piece_node_ptrs[n].push_back(node_ptr);
      }
    }
  }

  wires.wait_until_valid();
  FieldAccessor<READ_WRITE,float,1> fa_wire_currents[WIRE_SEGMENTS];
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_wire_currents[i] = FieldAccessor<READ_WRITE,float,1>(wires, FID_CURRENT+i);
  FieldAccessor<READ_WRITE,float,1> fa_wire_voltages[WIRE_SEGMENTS-1];
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_wire_voltages[i] = FieldAccessor<READ_WRITE,float,1>(wires, FID_WIRE_VOLTAGE+i);
  const FieldAccessor<READ_WRITE,Point<1>,1> fa_wire_in_ptr(wires, FID_IN_PTR);
  const FieldAccessor<READ_WRITE,Point<1>,1> fa_wire_out_ptr(wires, FID_OUT_PTR);
  const FieldAccessor<READ_WRITE,PointerLocation,1> fa_wire_in_loc(wires, FID_IN_LOC);
  const FieldAccessor<READ_WRITE,PointerLocation,1> fa_wire_out_loc(wires, FID_OUT_LOC);
  const FieldAccessor<READ_WRITE,float,1> fa_wire_inductance(wires, FID_INDUCTANCE);
  const FieldAccessor<READ_WRITE,float,1> fa_wire_resistance(wires, FID_RESISTANCE);
  const FieldAccessor<READ_WRITE,float,1> fa_wire_cap(wires, FID_WIRE_CAP);
  Point<1> *first_wires = new Point<1>[num_pieces];
  {
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < wires_per_piece; i++)
      {
        const Point<1> wire_ptr(n * wires_per_piece + i);
        // Record the first wire pointer for this piece
        if (i == 0)
          first_wires[n] = wire_ptr;
        for (int j = 0; j < WIRE_SEGMENTS; j++)
          fa_wire_currents[j][wire_ptr] = 0.f;
        for (int j = 0; j < WIRE_SEGMENTS-1; j++) 
          fa_wire_voltages[j][wire_ptr] = 0.f;

        fa_wire_resistance[wire_ptr] = drand48() * 10.0 + 1.0;
        // Keep inductance on the order of 1e-3 * dt to avoid resonance problems
        fa_wire_inductance[wire_ptr] = (drand48() + 0.1) * DELTAT * 1e-3;
        fa_wire_cap[wire_ptr] = drand48() * 0.1;

        fa_wire_in_ptr[wire_ptr] = random_element(piece_node_ptrs[n]); //private_node_map[n].points));

        if ((100 * drand48()) < pct_wire_in_piece)
        {
          fa_wire_out_ptr[wire_ptr] = random_element(piece_node_ptrs[n]); //private_node_map[n].points));
        }
        else
        {
          // pick a random other piece and a node from there
          int nn = int(drand48() * (num_pieces - 1));
          if(nn >= n) nn++;

	  // pick an arbitrary node, except that if it's one that didn't used to be shared, make the 
	  //  sequentially next pointer shared instead so that each node's shared pointers stay compact
	  int idx = int(drand48() * piece_node_ptrs[nn].size());
	  if(idx >= piece_shared_nodes[nn])
	    idx = piece_shared_nodes[nn]++;
	  const Point<1> out_ptr = piece_node_ptrs[nn][idx];

          fa_wire_out_ptr[wire_ptr] = out_ptr; 
        }
      }
    }
  }
  // Now we have to update the pointer locations
  for (int n = 0; n < num_pieces; n++)
  {
    for (int i = 0; i < wires_per_piece; i++)
    {
      const Point<1> wire_ptr(n * wires_per_piece + i);
      const Point<1> in_ptr = fa_wire_in_ptr[wire_ptr];
      const Point<1> out_ptr = fa_wire_out_ptr[wire_ptr];

      // In pointers always are local so see if it is shared or not
      assert((in_ptr[0] / nodes_per_piece) == n);
      const int local_in_node = in_ptr[0] % nodes_per_piece;
      if (local_in_node < piece_shared_nodes[n])
        fa_wire_in_loc[wire_ptr] = SHARED_PTR;
      else
        fa_wire_in_loc[wire_ptr] = PRIVATE_PTR;
      // Out pointers can be anything
      // Figure out which piece it is in
      const int nn = out_ptr[0] / nodes_per_piece;
      if (nn == n) {
        const int local_out_node = out_ptr[0] % nodes_per_piece;
        if (local_out_node < piece_shared_nodes[nn])
          fa_wire_out_loc[wire_ptr] = SHARED_PTR;
        else
          fa_wire_out_loc[wire_ptr] = PRIVATE_PTR;
      } else { // another piece so definitely a ghost pointer
        fa_wire_out_loc[wire_ptr] = GHOST_PTR; 
      }
    }
  }

  runtime->unmap_region(ctx, wires);
  runtime->unmap_region(ctx, nodes);

  // Now we can create our partitions and update the circuit pieces

  // First a partition based on the color field, we generated the field
  // in this case but it could also have been computed by a partitioner 
  // such as metis
  IndexSpace piece_is = runtime->create_index_space(ctx, Rect<1>(0, num_pieces-1));
  IndexPartition owned_ip = runtime->create_partition_by_field(ctx, ckt.all_nodes,
                                                               ckt.all_nodes,
                                                               FID_PIECE_COLOR,
                                                               piece_is);
  runtime->attach_name(owned_ip, "owned");

  // Partition the wires by ownership of the in node
  IndexPartition wire_ip = runtime->create_partition_by_preimage(ctx, owned_ip,
                                                                 ckt.all_wires,
                                                                 ckt.all_wires,
                                                                 FID_IN_PTR,
                                                                 piece_is);

  // Ghost nodes are connected to our wires, but not owned by us
  IndexPartition extended_ip = runtime->create_partition_by_image(ctx, 
                              ckt.all_nodes.get_index_space(), 
                              runtime->get_logical_partition(ckt.all_wires, wire_ip),
                              ckt.all_wires, FID_OUT_PTR, piece_is);
  runtime->attach_name(extended_ip, "extended");
  IndexPartition temp_ghost_ip = runtime->create_partition_by_difference(ctx, 
                              ckt.all_nodes.get_index_space(), extended_ip, 
                              owned_ip, piece_is);
  runtime->attach_name(temp_ghost_ip, "temp ghost");

  // Shared nodes are those that are ghost for somebody
  IndexSpace privacy_color_is = runtime->create_index_space(ctx, Rect<1>(0, 1));
  IndexPartition privacy_ip = runtime->create_pending_partition(ctx, 
                                ckt.all_nodes.get_index_space(), privacy_color_is);
  runtime->attach_name(privacy_ip, "privacy partition");
  IndexSpace all_shared_is = runtime->create_index_space_union(ctx, privacy_ip,
                                                     1/*color*/, temp_ghost_ip);
  runtime->attach_name(all_shared_is, "all shared");
  std::vector<IndexSpace> diff_spaces(1, all_shared_is);
  IndexSpace all_private_is = runtime->create_index_space_difference(ctx, privacy_ip,
                            0/*color*/, ckt.all_nodes.get_index_space(), diff_spaces);
  runtime->attach_name(all_private_is, "all private");

  // Create the private and shared partitions with a cross-product partition
  // There are only two handles so get them back right away
  std::map<IndexSpace,IndexPartition> partition_handles;
  partition_handles[all_private_is] = IndexPartition::NO_PART;
  partition_handles[all_shared_is] = IndexPartition::NO_PART;
  runtime->create_cross_product_partitions(ctx, privacy_ip, owned_ip, partition_handles);
  assert(partition_handles[all_private_is].exists());
  IndexPartition private_ip = partition_handles[all_private_is];
  runtime->attach_name(private_ip, "private partition");
  assert(partition_handles[all_shared_is].exists());
  IndexPartition shared_ip = partition_handles[all_shared_is];
  runtime->attach_name(shared_ip, "shared partition");

  // Finally compute the ghost sub partition of the all_shared piece
  IndexPartition ghost_ip = runtime->create_partition_by_difference(ctx, all_shared_is,
                                                       extended_ip, owned_ip, piece_is); 
  runtime->attach_name(ghost_ip, "ghost partition");
                                                              
  // Fill in the result partitions
  Partitions result;                                                               
  result.pvt_wires = runtime->get_logical_partition(ckt.all_wires, wire_ip); 
  result.pvt_nodes = runtime->get_logical_partition_by_tree(private_ip,
      ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  result.shr_nodes = runtime->get_logical_partition_by_tree(shared_ip,
      ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  result.ghost_nodes = runtime->get_logical_partition_by_tree(ghost_ip,
      ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  result.node_locations = runtime->get_logical_partition_by_tree(owned_ip,
      ckt.node_locator.get_field_space(), ckt.node_locator.get_tree_id());
  // Clean up the partitions that we don't need
  runtime->destroy_index_partition(ctx, extended_ip);
  runtime->destroy_index_partition(ctx, temp_ghost_ip);

  // Finally we need to figure out where everything ended up
  locator.wait_until_valid();
  const FieldAccessor<READ_WRITE,PointerLocation,1> locator_acc(locator, FID_LOCATOR);
  for (int n = 0; n < num_pieces; n++)
  {
    LogicalRegionT<1> private_lr( 
      runtime->get_logical_subregion_by_color(result.pvt_nodes, n));
    for (int i = 0; i < nodes_per_piece; i++)
    {
      const Point<1> node_ptr(n * nodes_per_piece + i);
      if (runtime->safe_cast(ctx, node_ptr, private_lr))
        locator_acc[node_ptr] = PRIVATE_PTR;
      else
        locator_acc[node_ptr] = SHARED_PTR;
    }
  }
  runtime->unmap_region(ctx, locator);

  char buf[100];
  // Build the pieces
  for (int n = 0; n < num_pieces; n++)
  {
    pieces[n].pvt_nodes = runtime->get_logical_subregion_by_color(ctx, result.pvt_nodes, n);
    sprintf(buf, "private_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].pvt_nodes, buf);
    pieces[n].shr_nodes = runtime->get_logical_subregion_by_color(ctx, result.shr_nodes, n);
    sprintf(buf, "shared_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].shr_nodes, buf);
    pieces[n].ghost_nodes = runtime->get_logical_subregion_by_color(ctx, result.ghost_nodes, n);
    sprintf(buf, "ghost_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].ghost_nodes, buf);
    pieces[n].pvt_wires = runtime->get_logical_subregion_by_color(ctx, result.pvt_wires, n);
    sprintf(buf, "private_wires_of_piece_%d", n);
    runtime->attach_name(pieces[n].pvt_wires, buf);
    pieces[n].num_wires = wires_per_piece;
    pieces[n].first_wire = first_wires[n];
    pieces[n].num_nodes = nodes_per_piece;
    pieces[n].first_node = first_nodes[n];

    pieces[n].dt = DELTAT;
    pieces[n].steps = steps;
  }

  delete [] first_wires;
  delete [] first_nodes;

  log_circuit.print("Finished initializing simulation...");

  return result;
}

