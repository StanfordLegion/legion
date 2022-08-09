/* Copyright 2022 Stanford University
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
  int nodes_per_piece = 1024;
  int wires_per_piece = 1024;
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
    IndexSpace node_index_space = runtime->create_index_space(ctx,
        Rect<1>(0, num_circuit_nodes-1));
    runtime->attach_name(node_index_space, "node_index_space");
    IndexSpace wire_index_space = runtime->create_index_space(ctx,
        Rect<1>(0, num_circuit_wires-1));
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
  log_circuit.print("Initializing circuit simulation...");
  Partitions parts = load_circuit(circuit, pieces, ctx, runtime, num_pieces, nodes_per_piece,
                                  wires_per_piece, pct_wire_in_piece, random_seed, steps);
  log_circuit.print("Finished initializing simulation...");

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
  Future f_start = runtime->get_current_time_in_microseconds(ctx);
  double ts_start = f_start.get_result<long long>();
  // Run the main loop
  bool simulation_success = true;
  for (int i = 0; i < num_loops; i++)
  {
    TaskHelper::dispatch_task<CalcNewCurrentsTask>(cnc_launcher, ctx, runtime, 
                                                   perform_checks, simulation_success);
    TaskHelper::dispatch_task<DistributeChargeTask>(dsc_launcher, ctx, runtime, 
                                                    perform_checks, simulation_success);
    TaskHelper::dispatch_task<UpdateVoltagesTask>(upv_launcher, ctx, runtime, 
                                                  perform_checks, simulation_success);
  }
  // Execution fence to wait for all prior operations to be done before getting our timing result
  runtime->issue_execution_fence(ctx);
  Future f_end = runtime->get_current_time_in_microseconds(ctx);
  double ts_end = f_end.get_result<long long>();
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
    AccessorROfloat fa_wire_currents[WIRE_SEGMENTS];
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      fa_wire_currents[i] = AccessorROfloat(wires, FID_CURRENT+i);
    AccessorROfloat fa_wire_voltages[WIRE_SEGMENTS-1];
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_wire_voltages[i] = AccessorROfloat(wires, FID_WIRE_VOLTAGE+i);

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
#ifndef SEQUENTIAL_LOAD_CIRCUIT
    registrar.set_replicable();
#endif
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // Create an aligned layout constraint for any vector architectures
#if defined(__AVX512F__)
  LayoutConstraintRegistrar layout_constraints;
  layout_constraints.add_constraint(
      AlignmentConstraint(0/*all fields*/, LEGION_EQ_EK, 64/*bytes*/));
  const LayoutConstraintID id = Runtime::preregister_layout(layout_constraints);
#elif defined(__AVX__)
  LayoutConstraintRegistrar layout_constraints;
  layout_constraints.add_constraint(
      AlignmentConstraint(0/*all fields*/, LEGION_EQ_EK, 32/*bytes*/));
  const LayoutConstraintID id = Runtime::preregister_layout(layout_constraints);
#elif defined(__SSE__)
  LayoutConstraintRegistrar layout_constraints;
  layout_constraints.add_constraint(
      AlignmentConstraint(0/*all fields*/, LEGION_EQ_EK, 16/*bytes*/));
  const LayoutConstraintID id = Runtime::preregister_layout(layout_constraints);
#else
  const LayoutConstraintID id = 0;
#endif
  TaskHelper::register_hybrid_variants<CalcNewCurrentsTask>(id);
  TaskHelper::register_hybrid_variants<DistributeChargeTask>(0/*no need for alignments on this task*/);
  TaskHelper::register_hybrid_variants<UpdateVoltagesTask>(0/*no need for alignments on this task*/);
  CheckTask::register_task();
#ifndef SEQUENTIAL_LOAD_CIRCUIT
  InitNodesTask::register_task();
  InitWiresTask::register_task();
  InitLocationTask::register_task();
#endif
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
    char field_name[64];
    allocator.allocate_field(sizeof(float), FID_CURRENT+i);
    snprintf(field_name, 64, "current_%d", i);
    runtime->attach_name(wire_space, FID_CURRENT+i, field_name);
  }
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
  {
    char field_name[64];
    allocator.allocate_field(sizeof(float), FID_WIRE_VOLTAGE+i);
    snprintf(field_name, 64, "wire_voltage_%d", i);
    runtime->attach_name(wire_space, FID_WIRE_VOLTAGE+i, field_name);
  }
}

void allocate_locator_fields(Context ctx, Runtime *runtime, FieldSpace locator_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, locator_space);
  allocator.allocate_field(sizeof(PointerLocation), FID_LOCATOR);
  runtime->attach_name(locator_space, FID_LOCATOR, "locator");
}

