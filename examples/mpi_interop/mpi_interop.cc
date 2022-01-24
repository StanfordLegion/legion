/* Copyright 2022 Stanford University, NVIDIA Corporation
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

////////////////////////////////////////////////////////////
//
// This example must be built with a Realm network layer
// that is compatible with MPI (e.g. a GASNet conduit that
// supports (and is built with) --enable-mpi-compat, or the
// MPI network layer).
//
// Any network layer that uses MPI for any communication
// during the application (rather than just for bootstrapping)
// additionally requires that the MPI implementation support
// MPI_THREAD_MULTIPLE.
//
////////////////////////////////////////////////////////////

#include <cstdio>
#include <stdlib.h>
// Need MPI header file
#include <mpi.h>

#include "legion.h"

using namespace Legion;

enum TaskID
{
  TOP_LEVEL_TASK_ID,
  MPI_INTEROP_TASK_ID,
  WORKER_TASK_ID,
};

// Here is our global MPI-Legion handshake
// You can have as many of these as you 
// want but the common case is just to
// have one per Legion-MPI rank pair
MPILegionHandshake handshake;

// Have a global static number of iterations for
// this example, but you can easily configure it
// from command line arguments which get passed 
// to both MPI and Legion
const int total_iterations = 10;

void worker_task(const Task *task, 
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  printf("Legion Doing Work in Rank %lld\n", 
          task->parent_task->index_point[0]);
}

void mpi_interop_task(const Task *task, 
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  printf("Hello from Legion MPI-Interop Task %lld\n", task->index_point[0]);
  for (int i = 0; i < total_iterations; i++)
  {
    // Legion can interop with MPI in blocking and non-blocking
    // ways. You can use the calls to 'legion_wait_on_mpi' and
    // 'legion_handoff_to_mpi' in the same way as the MPI thread
    // does. Alternatively, you can get a phase barrier associated
    // with a LegionMPIHandshake object which will allow you to
    // continue launching more sub-tasks without blocking. 
    // For deferred execution we prefer the later style, but
    // both will work correctly.
    if (i < (total_iterations/2))
    {
      // This is the blocking way of using handshakes, it
      // is not the ideal way, but it works correctly
      // Wait for MPI to give us control to run our worker
      // This is a blocking call
      handshake.legion_wait_on_mpi();
      // Launch our worker task
      TaskLauncher worker_launcher(WORKER_TASK_ID, TaskArgument(NULL,0));
      Future f = runtime->execute_task(ctx, worker_launcher);
      // Have to wait for the result before signaling MPI
      f.get_void_result();
      // Perform a non-blocking call to signal
      // MPI that we are giving it control back
      handshake.legion_handoff_to_mpi();
    }
    else
    {
      // This is the preferred way of using handshakes in Legion
      TaskLauncher worker_launcher(WORKER_TASK_ID, TaskArgument(NULL,0));
      // We can user our handshake as a phase barrier
      // Record that we will wait on this handshake
      worker_launcher.add_wait_handshake(handshake);
      // Advance the handshake to the next version
      handshake.advance_legion_handshake();
      // Then record that we will arrive on this versions
      worker_launcher.add_arrival_handshake(handshake);
      // Launch our worker task
      // No need to wait for anything
      runtime->execute_task(ctx, worker_launcher);
    }
  }
}

void top_level_task(const Task *task, 
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  printf("Hello from Legion Top-Level Task\n");
  // Both the application and Legion mappers have access to
  // the mappings between MPI Ranks and Legion address spaces
  // The reverse mapping goes the other way
  const std::map<int,AddressSpace> &forward_mapping = 
    runtime->find_forward_MPI_mapping();
  for (std::map<int,AddressSpace>::const_iterator it = 
        forward_mapping.begin(); it != forward_mapping.end(); it++)
    printf("MPI Rank %d maps to Legion Address Space %d\n", 
            it->first, it->second);

  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Do a must epoch launch to align with the number of MPI ranks
  MustEpochLauncher must_epoch_launcher;
  Rect<1> launch_bounds(0,size - 1);
  ArgumentMap args_map;
  IndexLauncher index_launcher(MPI_INTEROP_TASK_ID, launch_bounds, 
                               TaskArgument(NULL, 0), args_map);
  must_epoch_launcher.add_index_task(index_launcher);
  runtime->execute_must_epoch(ctx, must_epoch_launcher);
}

int main(int argc, char **argv)
{
#if defined(GASNET_CONDUIT_IBV) || defined(GASNET_CONDUIT_UCX)
  // work around GASNet issues during application cleanup:
  //  ibv conduit (debug build only): https://gasnet-bugs.lbl.gov/bugzilla/show_bug.cgi?id=4166
  //  ucx conduit (debug and release): https://upc-bugs.lbl.gov/bugzilla/show_bug.cgi?id=4172
  setenv("GASNET_CATCH_EXIT", "0", 0 /*!overwrite*/);
#endif
#if defined(GASNET_CONDUIT_MPI) || defined(REALM_USE_MPI)
  // The GASNet MPI conduit and/or the Realm MPI network layer
  // require that MPI be initialized for multiple threads
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  // If you fail this assertion, then your version of MPI
  // does not support calls from multiple threads and you 
  // cannot use the GASNet MPI conduit
  if (provided < MPI_THREAD_MULTIPLE)
    printf("ERROR: Your implementation of MPI does not support "
           "MPI_THREAD_MULTIPLE which is required for use of the "
           "GASNet MPI conduit or the Realm MPI network layer "
           "with the Legion-MPI Interop!\n");
  assert(provided == MPI_THREAD_MULTIPLE);
#else
  // Perform MPI start-up like normal for most GASNet conduits
  MPI_Init(&argc, &argv);
#endif

  int rank = -1, size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Hello from MPI process %d of %d\n", rank, size);

  // Configure the Legion runtime with the rank of this process
  Runtime::configure_MPI_interoperability(rank);
  // Register our task variants
  {
    TaskVariantRegistrar top_level_registrar(TOP_LEVEL_TASK_ID);
    top_level_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(top_level_registrar, 
                                                      "Top Level Task");
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  }
  {
    TaskVariantRegistrar mpi_interop_registrar(MPI_INTEROP_TASK_ID);
    mpi_interop_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<mpi_interop_task>(mpi_interop_registrar,
                                                        "MPI Interop Task");
  }
  {
    TaskVariantRegistrar worker_task_registrar(WORKER_TASK_ID);
    worker_task_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<worker_task>(worker_task_registrar,
                                                   "Worker Task");
  }
  // Create a handshake for passing control between Legion and MPI
  // Indicate that MPI has initial control and that there is one
  // participant on each side
  handshake = Runtime::create_handshake(true/*MPI initial control*/,
                                        1/*MPI participants*/,
                                        1/*Legion participants*/);
  // Start the Legion runtime in background mode
  // This call will return immediately
  Runtime::start(argc, argv, true/*background*/);
  // Run your MPI program like normal
  // If you want strict bulk-synchronous execution include
  // the barriers protected by this variable, otherwise
  // you can elide them, they are not required for correctness
  const bool strict_bulk_synchronous_execution = true;
  for (int i = 0; i < total_iterations; i++)
  {
    printf("MPI Doing Work on rank %d\n", rank);
    if (strict_bulk_synchronous_execution)
      MPI_Barrier(MPI_COMM_WORLD);
    // Perform a handoff to Legion, this call is
    // asynchronous and will return immediately
    handshake.mpi_handoff_to_legion();
    // You can put additional work in here if you like
    // but it may interfere with Legion work

    // Wait for Legion to hand control back,
    // This call will block until a Legion task
    // running in this same process hands control back
    handshake.mpi_wait_on_legion();
    if (strict_bulk_synchronous_execution)
      MPI_Barrier(MPI_COMM_WORLD);
  }
  // When you're done wait for the Legion runtime to shutdown
  Runtime::wait_for_shutdown();
#ifndef GASNET_CONDUIT_MPI
  // Then finalize MPI like normal
  // Exception for the MPI conduit which does its own finalization
  MPI_Finalize();
#endif

  return 0;
}
