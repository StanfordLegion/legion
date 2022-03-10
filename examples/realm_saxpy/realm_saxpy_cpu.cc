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

#include "realm_saxpy.h"
#include "realm/cmdline.h"
#include "realm/profiling.h"

#include <math.h>

#ifdef _MSC_VER
static double drand48(void) { return rand() / (double(RAND_MAX) + 1); }
#endif

// NOTE: all logging messages in this example use the 'print' level, which is
//  on by default, so that you can see stuff is happening - normally most
//  application logging should be at the 'info' level so that it has to be
//  enabled by an explicit request from the command line (i.e. '-level app=2')
Logger log_app("app");

// Other experiments:
// - Use partitioning API

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  INIT_VECTOR_TASK = Processor::TASK_ID_FIRST_AVAILABLE+1,
  CPU_SAXPY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+2,
  GPU_SAXPY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+3,
  CHECK_RESULT_TASK = Processor::TASK_ID_FIRST_AVAILABLE+4,
  PROFILE_TASK = Processor::TASK_ID_FIRST_AVAILABLE+5,
};

// global configuration parameters
namespace TestConfig {
  size_t num_elements = 32768;
  int iterations = 2;
  bool prefetch = false;
};

void profiling_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  ProfilingResponse pr(args, arglen);

  // expecting the iteration number in the _profile_response_'s user data
  assert(pr.user_data_size() == sizeof(int));
  int iteration = *static_cast<const int *>(pr.user_data());

  using namespace Realm::ProfilingMeasurements;
  OperationTimeline timeline;
  if(pr.get_measurement(timeline)) {
    // use the 'complete_time' instead of 'end_time' to account for deferred
    //  kernel execution on the GPU
    long long elapsed_ns = timeline.complete_time - timeline.start_time;
    // each element reads x/y/z, writes z
    double bw_gbs = double(TestConfig::num_elements * sizeof(float) * 4) / elapsed_ns;
    log_app.print() << "iteration " << iteration << ": elapsed="
                    << elapsed_ns << " ns, bw=" << bw_gbs << " GB/s";
  }
}

void find_processors(Processor &first_cpu, Processor &first_gpu)
{
  // Print out our processors and their kinds
  // Remember the first CPU and GPU processor
  Machine machine = Machine::get_machine();
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    Processor::Kind kind = it->kind();  
    switch (kind)
    {
      case Processor::LOC_PROC:
        {
          if (!first_cpu.exists())
            first_cpu = *it;
          log_app.print() << "CPU processor: " << *it;
          break;
        }
      case Processor::TOC_PROC:
        {
          if (!first_gpu.exists())
            first_gpu = *it;
          log_app.print() << "GPU processor: " << *it;
          break;
        }
      case Processor::UTIL_PROC:
        {
          log_app.print() << "utility processor: " << *it;
          break;
        }
      case Processor::IO_PROC:
        {
          log_app.print() << "I/O processor: " << *it;
          break;
        }
      case Processor::PROC_GROUP:
        {
          log_app.print() << "processor group: " << *it;
          break;
        }
      default:
	{
          log_app.print() << "unknown processor: " << *it;
	  break;
	}
    }
  }
}

void find_memories(Processor cpu, Processor gpu,
                   Memory &system, Memory &framebuffer,
                   Memory &managed, Memory &zerocopy)
{
  Machine machine = Machine::get_machine();
  std::set<Memory> visible_mems;
  machine.get_visible_memories(cpu, visible_mems);
  for (std::set<Memory>::const_iterator it = visible_mems.begin();
        it != visible_mems.end(); it++)
  {
    // skip memories with no capacity for creating instances
    if(it->capacity() == 0)
      continue;

    Memory::Kind kind = it->kind();
    switch (kind)
    {
      case Memory::SYSTEM_MEM:
      case Memory::SOCKET_MEM:
      case Memory::REGDMA_MEM:
        {
          system = *it;
          log_app.print() << "system memory: " << *it << " capacity="
                          << (it->capacity() >> 20) << " MB";
          break;
        }
      default:
        break;
    }
  }

  if (gpu.exists())
  {
    visible_mems.clear();
    machine.get_visible_memories(gpu, visible_mems);
    for (std::set<Memory>::const_iterator it = visible_mems.begin();
          it != visible_mems.end(); it++)
    {
      Memory::Kind kind = it->kind();
      switch (kind)
      {
        case Memory::GPU_FB_MEM:
          {
            framebuffer = *it;
            log_app.print() << "framebuffer memory: " << *it << " capacity="
                            << (it->capacity() >> 20) << " MB";
            break;
          }
        case Memory::Z_COPY_MEM:
          {
            zerocopy = *it;
            log_app.print() << "zero-copy memory: " << *it << " capacity="
                            << (it->capacity() >> 20) << " MB";
            break;
          }
        case Memory::GPU_MANAGED_MEM:
          {
            managed = *it;
            log_app.print() << "managed memory: " << *it << " capacity="
                            << (it->capacity() >> 20) << " MB";
            break;
          }
        default:
          break;
      }
    }
  }
}

void top_level_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{ 
  Processor first_cpu = Processor::NO_PROC;
  Processor first_gpu = Processor::NO_PROC;
  find_processors(first_cpu, first_gpu);

  Memory system_mem = Memory::NO_MEMORY;
  Memory framebuffer_mem = Memory::NO_MEMORY;
  Memory managed_mem = Memory::NO_MEMORY;
  Memory zerocopy_mem = Memory::NO_MEMORY;
  find_memories(first_cpu, first_gpu, system_mem, framebuffer_mem,
                managed_mem, zerocopy_mem);

  Rect<1> bounds(0, TestConfig::num_elements-1);

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_X] = sizeof(float);
  field_sizes[FID_Y] = sizeof(float);
  field_sizes[FID_Z] = sizeof(float);

  RegionInstance cpu_inst;
  RegionInstance::create_instance(cpu_inst, system_mem,
                                  bounds, field_sizes,
                                  0 /*SOA*/, ProfilingRequestSet()).wait();

  log_app.print() << "created system memory instance: " << cpu_inst;

  CopySrcDstField cpu_x_field, cpu_y_field, cpu_z_field;
  cpu_x_field.inst = cpu_inst;
  cpu_x_field.field_id = FID_X;
  cpu_x_field.size = sizeof(float);

  cpu_y_field.inst = cpu_inst;
  cpu_y_field.field_id = FID_Y;
  cpu_y_field.size = sizeof(float);

  cpu_z_field.inst = cpu_inst;
  cpu_z_field.field_id = FID_Z;
  cpu_z_field.size = sizeof(float);

  float init_x_value = drand48();
  float init_y_value = drand48();

  Event fill_x;
  {
    std::vector<CopySrcDstField> fill_vec;
    fill_vec.push_back(cpu_x_field);
    fill_x = bounds.fill(fill_vec, ProfilingRequestSet(),
			 &init_x_value, sizeof(init_x_value));
  }
  Event fill_y;
  {
    std::vector<CopySrcDstField> fill_vec;
    fill_vec.push_back(cpu_y_field);
    fill_y = bounds.fill(fill_vec, ProfilingRequestSet(),
			 &init_y_value, sizeof(init_y_value));
  }

  Event z_ready = Event::NO_EVENT;
  SaxpyArgs saxpy_args;
  saxpy_args.x_inst = cpu_inst;
  saxpy_args.y_inst = cpu_inst;
  saxpy_args.z_inst = cpu_inst;
  saxpy_args.alpha = drand48();
  saxpy_args.bounds = bounds;
  if (first_gpu.exists())
  {
    // Run the computation on the GPU
    // Make instances on the GPU
    RegionInstance gpu_inst;
    while(true) {
      // plan A: framebuffer memory
      if(framebuffer_mem.exists()) {
        Event e = RegionInstance::create_instance(gpu_inst, framebuffer_mem,
                                                  bounds, field_sizes,
                                                  0 /*SOA*/,
                                                  ProfilingRequestSet());
        // a failed allocation will result in a "poisoned" completion event
        bool poisoned;
        e.wait_faultaware(poisoned);
        if(!poisoned) {
          log_app.print() << "created framebuffer memory instance: " << cpu_inst;
          break;
        } else {
          log_app.print() << "unsufficient framebuffer memory for instance";
        }
      }

      // plan B: managed memory
      if(managed_mem.exists()) {
        Event e = RegionInstance::create_instance(gpu_inst, managed_mem,
                                                  bounds, field_sizes,
                                                  0 /*SOA*/,
                                                  ProfilingRequestSet());
        // a failed allocation will result in a "poisoned" completion event
        bool poisoned;
        e.wait_faultaware(poisoned);
        if(!poisoned) {
          log_app.print() << "created managed memory instance: " << cpu_inst;
          break;
        } else {
          log_app.print() << "unsufficient managed memory for instance";
        }
      }

      // plan C: zerocopy memory (not cached by GPU)
      if(zerocopy_mem.exists()) {
        Event e = RegionInstance::create_instance(gpu_inst, zerocopy_mem,
                                                  bounds, field_sizes,
                                                  0 /*SOA*/,
                                                  ProfilingRequestSet());
        // a failed allocation will result in a "poisoned" completion event
        bool poisoned;
        e.wait_faultaware(poisoned);
        if(!poisoned) {
          log_app.print() << "created zerocopy memory instance: " << cpu_inst;
          break;
        } else {
          log_app.print() << "unsufficient zerocopy memory for instance";
        }
      }

      // if we fall all the way through to here, it's a fatal error
      log_app.fatal() << "could not allocate instances for GPU - aborting!";
      Runtime::get_runtime().shutdown(Event::NO_EVENT, 1 /*exit code*/);
      return;
    }

    CopySrcDstField gpu_x_field, gpu_y_field, gpu_z_field;
    gpu_x_field.inst = gpu_inst;
    gpu_x_field.field_id = FID_X;
    gpu_x_field.size = sizeof(float);

    gpu_y_field.inst = gpu_inst;
    gpu_y_field.field_id = FID_Y;
    gpu_y_field.size = sizeof(float);

    gpu_z_field.inst = gpu_inst;
    gpu_z_field.field_id = FID_Z;
    gpu_z_field.size = sizeof(float);

    // Copy down
    Event copy_x;
    {
      std::vector<CopySrcDstField> srcs, dsts;
      srcs.push_back(cpu_x_field);
      dsts.push_back(gpu_x_field);
      copy_x = bounds.copy(srcs, dsts, ProfilingRequestSet(), fill_x);
    }
    Event copy_y;
    {
      std::vector<CopySrcDstField> srcs, dsts;
      srcs.push_back(cpu_y_field);
      dsts.push_back(gpu_y_field);
      copy_y = bounds.copy(srcs, dsts, ProfilingRequestSet(), fill_y);
    }
    Event fill_z;
    {
      std::vector<CopySrcDstField> dsts(1, gpu_z_field);
      float fill_value = 1.0f;
      fill_z = bounds.fill(dsts, ProfilingRequestSet(),
			   &fill_value, sizeof(fill_value));
    }

    SaxpyArgs gpu_args;
    gpu_args.x_inst = gpu_inst;
    gpu_args.y_inst = gpu_inst;
    gpu_args.z_inst = gpu_inst;
    gpu_args.alpha  = saxpy_args.alpha;
    gpu_args.bounds = bounds;

    Event e = Event::merge_events(copy_x, copy_y, fill_z);
    for(int i = 0; i < TestConfig::iterations; i++) {
      ProfilingRequestSet prs;
      prs.add_request(p, PROFILE_TASK, &i, sizeof(i))
        .add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      e = first_gpu.spawn(GPU_SAXPY_TASK, &gpu_args, sizeof(gpu_args), prs, e);
    }

    // Copy back
    {
      std::vector<CopySrcDstField> srcs, dsts;
      srcs.push_back(gpu_z_field);
      dsts.push_back(cpu_z_field);
      z_ready = bounds.copy(srcs, dsts, ProfilingRequestSet(), e);
    }
  }
  else
  {
    // Run the computation on the CPU
    Event fill_z;
    {
      std::vector<CopySrcDstField> dsts(1, cpu_z_field);
      float fill_value = 1.0f;
      fill_z = bounds.fill(dsts, ProfilingRequestSet(),
			   &fill_value, sizeof(fill_value));
    }

    Event e = Event::merge_events(fill_x, fill_y, fill_z);
    for(int i = 0; i < TestConfig::iterations; i++) {
      ProfilingRequestSet prs;
      prs.add_request(p, PROFILE_TASK, &i, sizeof(i))
        .add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      e = first_cpu.spawn(CPU_SAXPY_TASK, &saxpy_args, sizeof(saxpy_args), prs, e);
    }
    z_ready = e;
  }

  // Run our checker task
  Event done = first_cpu.spawn(CHECK_RESULT_TASK, &saxpy_args, 
                                sizeof(saxpy_args), z_ready);
  log_app.print() << "all work issued - done event=" << done;
  done.wait();
}

void cpu_saxpy_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(SaxpyArgs));
  const SaxpyArgs *saxpy_args = (const SaxpyArgs*)args;

  log_app.print() << "executing CPU saxpy task";

  // get affine accessors for each of our three instances
  AffineAccessor<float, 1> ra_x = AffineAccessor<float, 1>(saxpy_args->x_inst,
							   FID_X);
  AffineAccessor<float, 1> ra_y = AffineAccessor<float, 1>(saxpy_args->y_inst,
							   FID_Y);
  AffineAccessor<float, 1> ra_z = AffineAccessor<float, 1>(saxpy_args->z_inst,
							   FID_Z);

  for(int i = saxpy_args->bounds.lo; i <= saxpy_args->bounds.hi; i++)
    ra_z[i] += saxpy_args->alpha * ra_x[i] + ra_y[i];
}

void check_result_task(const void *args, size_t arglen,
                       const void *userdata, size_t userlen, Processor)
{
  assert(arglen == sizeof(SaxpyArgs));
  const SaxpyArgs *saxpy_args = (const SaxpyArgs*)args;

  log_app.print() << "executing checking task";

  // get affine accessors for each of our three instances
  AffineAccessor<float, 1> ra_x = AffineAccessor<float, 1>(saxpy_args->x_inst,
							   FID_X);
  AffineAccessor<float, 1> ra_y = AffineAccessor<float, 1>(saxpy_args->y_inst,
							   FID_Y);
  AffineAccessor<float, 1> ra_z = AffineAccessor<float, 1>(saxpy_args->z_inst,
							   FID_Z);

  size_t errors = 0;
  for(int i = saxpy_args->bounds.lo; i <= saxpy_args->bounds.hi; i++) {
    float expected = (TestConfig::iterations *
                      (saxpy_args->alpha * ra_x[i] + ra_y[i])) + 1.0;
    float actual = ra_z[i];

    // FMAs are too accurate
    float relative = (actual - expected) / expected;
    if (fabsf(relative) < 1e-6) {
      // ok
    } else {
      // only print the first 10 or so errors
      if(errors++ <= 10)
        log_app.warning() << "mismatch at index " << i
                          << ": expected=" << expected
                          << " actual=" << actual;
    }
  }
  if (errors == 0) {
    log_app.print() << "success - no mismatches detected";
    Runtime::get_runtime().shutdown(Event::NO_EVENT);
  } else {
    log_app.error() << "failure - " << errors << " mismatch(es) detected";
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 1 /*exit code*/);
  }
}

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
extern void gpu_saxpy_task(const void *args, size_t arglen,
                           const void *userdata, size_t userlen, Processor p);
#endif

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  bool ok = Realm::CommandLineParser()
    .add_option_int("-n", TestConfig::num_elements)
    .add_option_int("-i", TestConfig::iterations)
    .add_option_int("-p", TestConfig::prefetch)
    .parse_command_line(argc, argv);

  if(!ok) {
    log_app.fatal() << "error parsing command line arguments";
    exit(1);
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CPU_SAXPY_TASK, cpu_saxpy_task);
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  rt.register_task(GPU_SAXPY_TASK, gpu_saxpy_task);
#endif
  rt.register_task(CHECK_RESULT_TASK, check_result_task);
  rt.register_task(PROFILE_TASK, profiling_task);
  
  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    Machine::ProcessorQuery query(Machine::get_machine());
    query.only_kind(Processor::LOC_PROC);
    p = query.first();
  }
  assert(p.exists());

  // collective launch of a single task - checking task will request shutdown
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // now sleep this thread until that shutdown actually happens
  return rt.wait_for_shutdown();
}
