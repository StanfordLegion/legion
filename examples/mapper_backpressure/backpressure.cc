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

/*
 * This example shows how to use a custom mapper to apply backpressure when
 * mapping tasks. This is useful when a set of tasks can be executed in parallel,
 * but will use too much of a resource (for example memory) if they do. In such a
 * case, executing only some number of the tasks at a time can be more efficient
 * or even allow for successful completion of the program. We accomplish this through
 * a series of mapper calls:
 *  * select_tasks_to_map: ensure that only a fixed number of tasks execute at a time
 *  * map_task: mark target tasks to return profiling results on completion
 *  * report_profiling: notify select_tasks_to_map when tasks complete to schedule more
 *
 * To execute this program without backpressure, use
 *   ./backpressure -ll:csize 4000 -lg:eager_alloc_percentage 50
 * Under this configuration, the output will show that all worker tasks begin executing
 * concurrently.
 *
 * Then, execute the program with backpressure by using
 *   ./backpressure -ll:csize 4000 -lg:eager_alloc_percentage 50 -backpressure
 * which ensures that only a fixed number (default 3) tasks execute at a time.
 *
 * The number of tasks that execute at a time can be controlled with -maxTasksInFlight.
 */

#include "legion.h"
#include "mappers/default_mapper.h"
#include <unistd.h>
#include <chrono>

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_WORKER,
};

// BackPressureMapper will backpressure executions of TID_WORKER so that there
// at most a fixed number of executions of TID_WORKER on a single processor live
// at the same time.
class BackPressureMapper : public Mapping::DefaultMapper {
public:
  BackPressureMapper(MapperRuntime* rt, Machine machine, Processor local) : DefaultMapper(rt, machine, local) {
    int argc = Legion::HighLevelRuntime::get_input_args().argc;
    char **argv = Legion::HighLevelRuntime::get_input_args().argv;
    // Parse some command line parameters.
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-backpressure") == 0) {
        this->enableBackPressure = true;
        continue;
      }
      if (strcmp(argv[i], "-maxTasksInFlight") == 0) {
        this->maxInFlightTasks = atoi(argv[++i]);
        continue;
      }
    }
  }

  void select_tasks_to_map(const MapperContext ctx,
                           const SelectMappingInput& input,
                                 SelectMappingOutput& output) override {
    // Record when we are scheduling tasks.
    auto schedTime = std::chrono::high_resolution_clock::now();

    // Maintain an event that we can return in case we don't schedule anything.
    // This event will be used by the runtime to determine when it should ask us
    // to schedule more tasks for mapping. We also maintain a timestamp with the
    // return event. This is so that we choose the earliest scheduled task to wait
    // on, so that we can keep the processors busy.
    MapperEvent returnEvent;
    auto returnTime = std::chrono::high_resolution_clock::time_point::max();

    // Schedule all of our available tasks, except tasks with TID_WORKER,
    // to which we'll backpressure.
    for (auto task : input.ready_tasks) {
      bool schedule = true;
      if (task->task_id == TID_WORKER && this->enableBackPressure) {
        // See how many tasks we have in flight.
        auto inflight = this->queue[task->target_proc];
        if (inflight.size() == this->maxInFlightTasks) {
          // We've hit the cap, so we can't schedule any more tasks.
          schedule = false;
          // As a heuristic, we'll wait on the first mapper event to
          // finish, as it's likely that one will finish first. We'll also
          // try to get a task that will complete before the current best.
          auto front = inflight.front();
          if (front.schedTime < returnTime) {
            returnEvent = front.event;
            returnTime = front.schedTime;
          }
        } else {
          // Otherwise, we can schedule the task. Create a new event
          // and queue it up on the processor.
          this->queue[task->target_proc].push_back({
            .id = task->get_unique_id(),
            .event = this->runtime->create_mapper_event(ctx),
            .schedTime = schedTime,
          });
        }
      }
      // Schedule tasks that passed the check.
      if (schedule) {
        output.map_tasks.insert(task);
      }
    }
    // If we don't schedule any tasks for mapping, the runtime needs to know
    // when to ask us again to schedule more things. Return the MapperEvent we
    // selected earlier.
    if (output.map_tasks.size() == 0) {
      assert(returnEvent.exists());
      output.deferral_event = returnEvent;
    }
  }

  void map_task(const MapperContext ctx,
                const Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output) override {
    DefaultMapper::map_task(ctx, task, input, output);
    // We need to know when the TID_WORKER tasks complete, so we'll ask the runtime
    // to give us profiling information when they complete.
    if (task.task_id == TID_WORKER && this->enableBackPressure) {
      output.task_prof_requests.add_measurement<ProfilingMeasurements::OperationStatus>();
    }
  }

  void report_profiling(const MapperContext ctx,
                        const Task& task,
                        const TaskProfilingInfo& input) override {
    // Only TID_WORKER tasks should have profiling information.
    assert(task.task_id == TID_WORKER);
    // We expect all of our tasks to complete successfully.
    auto prof = input.profiling_responses.get_measurement<ProfilingMeasurements::OperationStatus>();
    assert(prof->result == Realm::ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY);
    // Clean up after ourselves.
    delete prof;
    // Iterate through the queue and find the event for this task instance.
    std::deque<InFlightTask>& inflight = this->queue[task.target_proc];
    MapperEvent event;
    for (auto it = inflight.begin(); it != inflight.end(); it++) {
      if (it->id == task.get_unique_id()) {
        event = it->event;
        inflight.erase(it);
        break;
      }
    }
    assert(event.exists());
    // Trigger the event so that the runtime knows it's time to schedule
    // some more tasks to map.
    this->runtime->trigger_mapper_event(ctx, event);
  }

  // We use the non-reentrant serialized model as we want to ensure sole access to
  // mapper data structures. If we use the reentrant model, we would have to lock
  // accesses to the queue, since we use it interchanged with calls into the runtime.
  MapperSyncModel get_mapper_sync_model() const override {
    return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
  }

  // InFlightTask represents a currently executing task.
  struct InFlightTask {
    // A unique identifier for an instance of a task.
    UniqueID id;
    // An event that we will trigger when the task completes.
    MapperEvent event;
    // The point in time when we scheduled the task.
    std::chrono::high_resolution_clock::time_point schedTime;
  };
  // queue maintains the current tasks executing on each processor.
  std::map<Processor, std::deque<InFlightTask>> queue;
  // maxInFlightTasks controls how many tasks can execute at a time
  // on a single processor.
  size_t maxInFlightTasks = 3;
  bool enableBackPressure = false;
};

void worker(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto workerIdx = *(int32_t*)task->args;
  std::cout << "starting worker " << workerIdx << std::endl;
  // Allocate a deferred buffer with an initial value. This allows the runtime to
  // deschedule this task while the fill is occurring and start execution of another
  // parallel task.
  double initVal = 0.f;
  DeferredBuffer<double, 1> buf(Memory::Kind::SYSTEM_MEM, DomainT<1>(Rect<1>(0, 500000)), &initVal);
  // Sleep for a little bit to see some interesting interleavings.
  usleep(500 * 1000);
  std::cout << "finishing worker " << workerIdx << std::endl;
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Launch several tasks that can execute in parallel (i.e. have no dependencies).
  for (int i = 0; i < 12; i++) {
    int32_t idx = i;
    TaskLauncher launcher(TID_WORKER, TaskArgument(&idx, sizeof(int32_t)));
    runtime->execute_task(ctx, launcher);
  }
}

void register_mapper(Machine machine, Runtime* runtime, const std::set<Processor> &local_procs) {
  // Replace the DefaultMapper with a single BackPressureMapper.
  auto m = new BackPressureMapper(runtime->get_mapper_runtime(), machine, *local_procs.begin());
  runtime->replace_default_mapper(m);
}

int main(int argc, char** argv) {
  // Register tasks and start the runtime.
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_WORKER, "worker");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<worker>(registrar, "worker");
  }
  Runtime::add_registration_callback(register_mapper);
  return Runtime::start(argc, argv);
}
