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

#include "circuit.h"

#ifndef SEQUENTIAL_LOAD_CIRCUIT

InitNodesTask::InitNodesTask(LogicalRegion lr_all_nodes,
                             LogicalPartition lp_equal_nodes,
                             IndexSpace launch_domain)
  : IndexLauncher(InitNodesTask::TASK_ID, launch_domain, TaskArgument(),
                  ArgumentMap(), Predicate::TRUE_PRED, false/*must*/,
                  InitNodesTask::MAPPER_ID)
{
  RegionRequirement rr_nodes(lp_equal_nodes, 0/*identity*/,
                             WRITE_DISCARD, EXCLUSIVE, lr_all_nodes);
  rr_nodes.add_field(FID_NODE_CAP);
  rr_nodes.add_field(FID_LEAKAGE);
  rr_nodes.add_field(FID_CHARGE);
  rr_nodes.add_field(FID_NODE_VOLTAGE);
  rr_nodes.add_field(FID_PIECE_COLOR);
  add_region_requirement(rr_nodes);
}

/*static*/ const char * const InitNodesTask::TASK_NAME = "Initialize Nodes";

/*static*/
void InitNodesTask::cpu_base_impl(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime)
{
  const AccessorWOfloat fa_node_cap(regions[0], FID_NODE_CAP);
  const AccessorWOfloat fa_node_leakage(regions[0], FID_LEAKAGE);
  const AccessorWOfloat fa_node_charge(regions[0], FID_CHARGE);
  const AccessorWOfloat fa_node_voltage(regions[0], FID_NODE_VOLTAGE);
  const AccessorWOpoint fa_node_color(regions[0], FID_PIECE_COLOR); 

  DomainT<1> dom = runtime->get_index_space_domain(ctx,
      IndexSpaceT<1>(task->regions[0].region.get_index_space()));
  // Keep our initialization deterministic
  unsigned short int xsubi[3];
  for (int i = 0; i < 3; i++)
    xsubi[i] = task->index_point[0];
  for (PointInDomainIterator<1> itr(dom); itr(); itr++)
  {
    fa_node_cap[*itr] = erand48(xsubi) + 1.f;
    fa_node_leakage[*itr] = 0.1f * erand48(xsubi);
    fa_node_charge[*itr] = 0.f;
    fa_node_voltage[*itr] = 2.f * erand48(xsubi) - 1.f;
    fa_node_color[*itr] = task->index_point;
  }
}

/*static*/
void InitNodesTask::register_task(void)
{
  TaskVariantRegistrar registrar(InitNodesTask::TASK_ID, InitNodesTask::TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(InitNodesTask::LEAF);
  Runtime::preregister_task_variant<cpu_base_impl>(registrar, InitNodesTask::TASK_NAME);
}

InitWiresTask::InitWiresTask(LogicalRegion lr_all_wires,
                             LogicalPartition lp_equal_wires,
                             IndexSpace launch_domain,
                             int num_pieces,
                             int nodes_per_piece,
                             int pct_wire_in_piece)
  : IndexLauncher(InitWiresTask::TASK_ID, launch_domain, 
                  TaskArgument(&args, sizeof(args)),
                  ArgumentMap(), Predicate::TRUE_PRED, false/*must*/,
                  InitWiresTask::MAPPER_ID),
    args(Args(num_pieces, nodes_per_piece,  pct_wire_in_piece))
{
  RegionRequirement rr_wires(lp_equal_wires, 0/*identity*/,
                             WRITE_DISCARD, EXCLUSIVE, lr_all_wires);
  rr_wires.add_field(FID_IN_PTR);
  rr_wires.add_field(FID_OUT_PTR);
  rr_wires.add_field(FID_INDUCTANCE);
  rr_wires.add_field(FID_RESISTANCE);
  rr_wires.add_field(FID_WIRE_CAP);
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    rr_wires.add_field(FID_CURRENT+i);
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    rr_wires.add_field(FID_WIRE_VOLTAGE+i);
  add_region_requirement(rr_wires);
}

/*static*/ const char * const InitWiresTask::TASK_NAME = "Initialize Wires";

/*static*/
void InitWiresTask::cpu_base_impl(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime)
{
  const Args *args = (const Args*)task->args;
  const int num_pieces = args->num_pieces;
  const int nodes_per_piece = args->nodes_per_piece;
  const int pct_wire_in_piece = args->pct_wire_in_piece;
  const PhysicalRegion wires = regions[0];
  AccessorWOfloat fa_wire_currents[WIRE_SEGMENTS];
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_wire_currents[i] = AccessorWOfloat(wires, FID_CURRENT+i);
  AccessorWOfloat fa_wire_voltages[WIRE_SEGMENTS-1];
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_wire_voltages[i] = AccessorWOfloat(wires, FID_WIRE_VOLTAGE+i);
  const AccessorWOpoint fa_wire_in_ptr(wires, FID_IN_PTR);
  const AccessorWOpoint fa_wire_out_ptr(wires, FID_OUT_PTR);
  const AccessorWOfloat fa_wire_inductance(wires, FID_INDUCTANCE);
  const AccessorWOfloat fa_wire_resistance(wires, FID_RESISTANCE);
  const AccessorWOfloat fa_wire_cap(wires, FID_WIRE_CAP);

  DomainT<1> dom = runtime->get_index_space_domain(ctx,
      IndexSpaceT<1>(task->regions[0].region.get_index_space()));
  // Keep our initialization deterministic
  unsigned short int xsubi[3];
  for (int i = 0; i < 3; i++)
    xsubi[i] = task->index_point[0];
  for (PointInDomainIterator<1> itr(dom); itr(); itr++)
  {
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      fa_wire_currents[i][*itr] = 0.f;
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_wire_voltages[i][*itr] = 0.f;

    fa_wire_resistance[*itr] = erand48(xsubi) * 10.0 + 1.f;
    fa_wire_inductance[*itr] = (erand48(xsubi) + 0.1) * DELTAT * 1e-3;
    fa_wire_cap[*itr] = erand48(xsubi) * 0.1;
    // Pick a random node within our piece
    int in_index = int(erand48(xsubi) * nodes_per_piece);
    fa_wire_in_ptr[*itr] = Point<1>(task->index_point[0] * nodes_per_piece + in_index);
    if ((100 * erand48(xsubi)) < pct_wire_in_piece)
    {
      // Stay within our piece
      int out_index = int(erand48(xsubi) * nodes_per_piece);
      fa_wire_out_ptr[*itr] = Point<1>(task->index_point[0] * nodes_per_piece + out_index);
    }
    else
    {
      // Pick one from a different piece
      int nn = int(erand48(xsubi) * (num_pieces - 1));
      if (nn >= task->index_point[0]) nn++;
      // Not going to guarantee compactness in the parallel case 
      int out_index = int(erand48(xsubi) * nodes_per_piece);
      fa_wire_out_ptr[*itr] = Point<1>(nn * nodes_per_piece + out_index);
    }
  }
}

/*static*/
void InitWiresTask::register_task(void)
{
  TaskVariantRegistrar registrar(InitWiresTask::TASK_ID, InitWiresTask::TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(InitWiresTask::LEAF);
  Runtime::preregister_task_variant<cpu_base_impl>(registrar, InitWiresTask::TASK_NAME);
}

InitLocationTask::InitLocationTask(LogicalRegion lr_location,
                                   LogicalPartition lp_equal_location,
                                   LogicalRegion lr_all_wires,
                                   LogicalPartition lp_equal_wires,
                                   IndexSpace launch_domain,
                                   LogicalPartition lp_private,
                                   LogicalPartition lp_shared)
  : IndexLauncher(InitLocationTask::TASK_ID, launch_domain, 
                  TaskArgument(&args, sizeof(args)),
                  ArgumentMap(), Predicate::TRUE_PRED, false/*must*/,
                  InitLocationTask::MAPPER_ID),
    args(Args(lp_private, lp_shared))
{
  RegionRequirement rr_loc(lp_equal_location, 0/*identity*/,
                           WRITE_DISCARD, EXCLUSIVE, lr_location);
  rr_loc.add_field(FID_LOCATOR);
  add_region_requirement(rr_loc);

  RegionRequirement rr_wires_out(lp_equal_wires, 0/*identity*/,
                                 WRITE_DISCARD, EXCLUSIVE, lr_all_wires);
  rr_wires_out.add_field(FID_IN_LOC);
  rr_wires_out.add_field(FID_OUT_LOC);
  add_region_requirement(rr_wires_out);

  RegionRequirement rr_wires_in(lp_equal_wires, 0/*identity*/,
                                READ_ONLY, EXCLUSIVE, lr_all_wires);
  rr_wires_in.add_field(FID_IN_PTR);
  rr_wires_in.add_field(FID_OUT_PTR);
  add_region_requirement(rr_wires_in);
}

/*static*/ const char * const InitLocationTask::TASK_NAME = "Init Locations";

/*static*/
void InitLocationTask::cpu_base_impl(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
  // Get our sub-regions
  const Args *args = (const Args*)task->args;
  const LogicalPartition lp_private = args->lp_private;
  const LogicalPartition lp_shared = args->lp_shared;
  // Get our specific subregions
  const LogicalRegionT<1> lr_private(
      runtime->get_logical_subregion_by_color(lp_private, task->index_point));
  const LogicalRegionT<1> lr_shared(
      runtime->get_logical_subregion_by_color(lp_shared, task->index_point));

  // Update the node locations first
  const AccessorWOloc locator_acc(regions[0], FID_LOCATOR);
  DomainT<1> node_dom = runtime->get_index_space_domain(ctx,
      IndexSpaceT<1>(task->regions[0].region.get_index_space()));
  for (PointInDomainIterator<1> itr(node_dom); itr(); itr++)
  {
    if (runtime->safe_cast(ctx, *itr, lr_private)) {
      locator_acc[*itr] = PRIVATE_PTR;
    } else {
      assert(runtime->safe_cast(ctx, *itr, lr_shared));
      locator_acc[*itr] = SHARED_PTR;
    }
  }

  // Then do the wire locations
  const AccessorWOloc fa_wire_in_loc(regions[1], FID_IN_LOC);
  const AccessorWOloc fa_wire_out_loc(regions[1], FID_OUT_LOC);
  const AccessorROpoint fa_wire_in_ptr(regions[2], FID_IN_PTR);
  const AccessorROpoint fa_wire_out_ptr(regions[2], FID_OUT_PTR);
  DomainT<1> wire_dom = runtime->get_index_space_domain(ctx,
      IndexSpaceT<1>(task->regions[1].region.get_index_space()));
  for (PointInDomainIterator<1> itr(wire_dom); itr(); itr++)
  {
    Point<1> in_ptr = fa_wire_in_ptr[*itr];
    if (runtime->safe_cast(ctx, in_ptr, lr_private)) {
      fa_wire_in_loc[*itr] = PRIVATE_PTR;
    } else {
      assert(runtime->safe_cast(ctx, in_ptr, lr_shared));
      fa_wire_in_loc[*itr] = SHARED_PTR;
    }
    Point<1> out_ptr = fa_wire_out_ptr[*itr];
    if (runtime->safe_cast(ctx, out_ptr, lr_private))
      fa_wire_out_loc[*itr] = PRIVATE_PTR;
    else if (runtime->safe_cast(ctx, out_ptr, lr_shared))
      fa_wire_out_loc[*itr] = SHARED_PTR;
    else
      fa_wire_out_loc[*itr] = GHOST_PTR;
  }
}

/*static*/
void InitLocationTask::register_task(void)
{
  TaskVariantRegistrar registrar(InitLocationTask::TASK_ID, InitLocationTask::TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(InitLocationTask::LEAF);
  Runtime::preregister_task_variant<cpu_base_impl>(registrar, InitLocationTask::TASK_NAME);
}

#endif // !SEQUENTIAL_LOAD_CIRCUIT

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
  IndexSpace piece_is = runtime->create_index_space(ctx, Rect<1>(0, num_pieces-1));
#ifdef SEQUENTIAL_LOAD_CIRCUIT
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
  const AccessorRWfloat fa_node_cap(nodes, FID_NODE_CAP);
  const AccessorRWfloat fa_node_leakage(nodes, FID_LEAKAGE);
  const AccessorRWfloat fa_node_charge(nodes, FID_CHARGE);
  const AccessorRWfloat fa_node_voltage(nodes, FID_NODE_VOLTAGE);
  const AccessorRWpoint fa_node_color(nodes, FID_PIECE_COLOR); 
  {
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        const Point<1> node_ptr(n * nodes_per_piece + i);
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
  AccessorRWfloat fa_wire_currents[WIRE_SEGMENTS];
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_wire_currents[i] = AccessorRWfloat(wires, FID_CURRENT+i);
  AccessorRWfloat fa_wire_voltages[WIRE_SEGMENTS-1];
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_wire_voltages[i] = AccessorRWfloat(wires, FID_WIRE_VOLTAGE+i);
  const AccessorRWpoint fa_wire_in_ptr(wires, FID_IN_PTR);
  const AccessorRWpoint fa_wire_out_ptr(wires, FID_OUT_PTR);
  const AccessorRWloc fa_wire_in_loc(wires, FID_IN_LOC);
  const AccessorRWloc fa_wire_out_loc(wires, FID_OUT_LOC);
  const AccessorRWfloat fa_wire_inductance(wires, FID_INDUCTANCE);
  const AccessorRWfloat fa_wire_resistance(wires, FID_RESISTANCE);
  const AccessorRWfloat fa_wire_cap(wires, FID_WIRE_CAP);
  {
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < wires_per_piece; i++)
      {
        const Point<1> wire_ptr(n * wires_per_piece + i);
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
#else // SEQUENTIAL_LOAD_CIRCUIT
  // Make equal partitions of our node and wire subregions
  IndexPartition node_equal_ip = 
    runtime->create_equal_partition(ctx, ckt.all_nodes.get_index_space(), piece_is);
  IndexPartition wire_equal_ip = 
    runtime->create_equal_partition(ctx, ckt.all_wires.get_index_space(), piece_is);
  // Launch tasks to initialize the nodes and wires
  InitNodesTask init_nodes_launcher(ckt.all_nodes,
      runtime->get_logical_partition(ckt.all_nodes, node_equal_ip), piece_is);
  FutureMap fm_nodes_initialized = runtime->execute_index_space(ctx, init_nodes_launcher);

  InitWiresTask init_wires_launcher(ckt.all_wires,
      runtime->get_logical_partition(ckt.all_wires, wire_equal_ip), piece_is,
      num_pieces, nodes_per_piece, pct_wire_in_piece);
  FutureMap fm_wires_initialized = runtime->execute_index_space(ctx, init_wires_launcher);
#endif // !SEQUENTIAL_LOAD_CIRCUIT

  // Now we can create our partitions and update the circuit pieces

  // First a partition based on the color field, we generated the field
  // in this case but it could also have been computed by a partitioner 
  // such as metis
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
  runtime->destroy_index_space(ctx, privacy_color_is);
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
#ifdef SEQUENTIAL_LOAD_CIRCUIT
  locator.wait_until_valid();
  const AccessorRWloc locator_acc(locator, FID_LOCATOR);
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
#else // SEQUENTIAL_LOAD_CIRCUIT
  IndexPartition locator_equal_ip = 
    runtime->create_equal_partition(ctx, ckt.node_locator.get_index_space(), piece_is);
  InitLocationTask init_location_launcher(ckt.node_locator,
      runtime->get_logical_partition(ckt.node_locator, locator_equal_ip), ckt.all_wires,
      runtime->get_logical_partition(ckt.all_wires, wire_equal_ip), piece_is,
      runtime->get_logical_partition_by_tree(private_ip, 
        ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id()),
      runtime->get_logical_partition_by_tree(shared_ip,
        ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id()));
  FutureMap fm_locations_initialized = 
    runtime->execute_index_space(ctx, init_location_launcher);
  // Destroy our equal partitions since we don't need them anymore
  runtime->destroy_index_partition(ctx, node_equal_ip);
  runtime->destroy_index_partition(ctx, wire_equal_ip);
  runtime->destroy_index_partition(ctx, locator_equal_ip);
#endif // !SEQUENTIAL_LOAD_CIRCUIT
  runtime->destroy_index_space(ctx, piece_is);

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
    pieces[n].first_wire = Point<1>(n * wires_per_piece);
    pieces[n].num_nodes = nodes_per_piece;
    pieces[n].first_node = Point<1>(n * nodes_per_piece);

    pieces[n].dt = DELTAT;
    pieces[n].steps = steps;
  }

#ifndef SEQUENTIAL_LOAD_CIRCUIT
  // Wait for everything to be ready to get timing right
  fm_nodes_initialized.wait_all_results();
  fm_wires_initialized.wait_all_results();
  fm_locations_initialized.wait_all_results();
#endif

  return result;
}

