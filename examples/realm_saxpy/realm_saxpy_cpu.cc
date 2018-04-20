/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// Other experiments:
// - Run computation out of Zero-Copy memory
// - Use partitioning API
// - Add profiling

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  INIT_VECTOR_TASK = Processor::TASK_ID_FIRST_AVAILABLE+1,
  CPU_SAXPY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+2,
  GPU_SAXPY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+3,
  CHECK_RESULT_TASK = Processor::TASK_ID_FIRST_AVAILABLE+4,
};

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
      case LOC_PROC:
        {
          if (!first_cpu.exists())
            first_cpu = *it;
          printf("CPU Processor " IDFMT "\n", it->id);
          break;
        }
      case TOC_PROC:
        {
          if (!first_gpu.exists())
            first_gpu = *it;
          printf("GPU Processor " IDFMT "\n", it->id);
          break;
        }
      case UTIL_PROC:
        {
          printf("Utility Processor " IDFMT "\n", it->id);
          break;
        }
      case IO_PROC:
        {
          printf("I/O Processor " IDFMT "\n", it->id);
          break;
        }
      case PROC_GROUP:
        {
          printf("Processor Group " IDFMT "\n", it->id);
          break;
        }
      default:
        assert(false);
    }
  }
  printf("\n");
}

void find_memories(Processor cpu, Processor gpu,
                   Memory &system, Memory &framebuffer)
{
  Machine machine = Machine::get_machine();
  std::set<Memory> visible_mems;
  machine.get_visible_memories(cpu, visible_mems);
  for (std::set<Memory>::const_iterator it = visible_mems.begin();
        it != visible_mems.end(); it++)
  {
    Memory::Kind kind = it->kind();
    switch (kind)
    {
      case Memory::SYSTEM_MEM:
      case Memory::SOCKET_MEM:
      case Memory::REGDMA_MEM:
        {
          system = *it;
          printf("System Memory " IDFMT " for CPU Processor " IDFMT 
                 " has capacity %zd MB\n", it->id, cpu.id,
                 (it->capacity() >> 20));
          break;
        }
      case Memory::Z_COPY_MEM:
        {
          printf("Zero-Copy Memory " IDFMT " for CPU Processor " IDFMT 
                 " has capacity %zd MB\n", it->id, cpu.id,
                 (it->capacity() >> 20));
          break;
        }
      default:
        printf("Unknown Memory Kind for CPU: %d\n", kind);
    }
  }
  printf("\n");
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
            printf("Framebuffer Memory " IDFMT " for GPU Processor " IDFMT 
                   " has capacity %zd MB\n", it->id, gpu.id,
                   (it->capacity() >> 20));
            break;
          }
        case Memory::Z_COPY_MEM:
          {
            printf("Zero-Copy Memory " IDFMT " for GPU Processor " IDFMT 
                   " has capacity %zd MB\n", it->id, gpu.id,
                   (it->capacity() >> 20));
            break;
          }
        default:
          printf("Unknown Memory Kind for GPU: %d\n", kind);
      }
    }
    printf("\n");
  }
}

void top_level_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{ 
  printf("HELLO WORLD from Processor " IDFMT "!\n\n", p.id);
  Processor first_cpu = Processor::NO_PROC;
  Processor first_gpu = Processor::NO_PROC;
  find_processors(first_cpu, first_gpu);

  Memory system_mem = Memory::NO_MEMORY;
  Memory framebuffer_mem = Memory::NO_MEMORY;
  find_memories(first_cpu, first_gpu, system_mem, framebuffer_mem);

  Rect<1> bounds(0, 16383);

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_X] = sizeof(float);
  field_sizes[FID_Y] = sizeof(float);
  field_sizes[FID_Z] = sizeof(float);

  RegionInstance cpu_inst;
  RegionInstance::create_instance(cpu_inst, system_mem,
                                  bounds, field_sizes,
                                  0 /*SOA*/, ProfilingRequestSet()).wait();

  printf("Created System Memory Instance: " IDFMT "\n\n",
      cpu_inst.id);

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
    RegionInstance::create_instance(gpu_inst, framebuffer_mem,
				    bounds, field_sizes,
				    0 /*SOA*/, ProfilingRequestSet()).wait();

    printf("Created Framebuffer Memory Instances: " IDFMT "\n\n",
      gpu_inst.id);

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

    Event precondition = Event::merge_events(copy_x, copy_y, fill_z);
    Event gpu_done = first_gpu.spawn(GPU_SAXPY_TASK, &gpu_args,
                                     sizeof(gpu_args), precondition);
    // Copy back
    {
      std::vector<CopySrcDstField> srcs, dsts;
      srcs.push_back(gpu_z_field);
      dsts.push_back(cpu_z_field);
      z_ready = bounds.copy(srcs, dsts, ProfilingRequestSet(), gpu_done);
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

    Event precondition = Event::merge_events(fill_x, fill_y, fill_z);
    z_ready = first_cpu.spawn(CPU_SAXPY_TASK, &saxpy_args, 
                              sizeof(saxpy_args), precondition);
  }

  // Run our checker task
  Event done = first_cpu.spawn(CHECK_RESULT_TASK, &saxpy_args, 
                                sizeof(saxpy_args), z_ready); 
  printf("Done Event is (" IDFMT ")\n\n", done.id);
  done.wait();
}

void cpu_saxpy_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(SaxpyArgs));
  const SaxpyArgs *saxpy_args = (const SaxpyArgs*)args;
  printf("Running CPU Saxpy Task\n\n");

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
  printf("Running Checking Task...");

  // get affine accessors for each of our three instances
  AffineAccessor<float, 1> ra_x = AffineAccessor<float, 1>(saxpy_args->x_inst,
							   FID_X);
  AffineAccessor<float, 1> ra_y = AffineAccessor<float, 1>(saxpy_args->y_inst,
							   FID_Y);
  AffineAccessor<float, 1> ra_z = AffineAccessor<float, 1>(saxpy_args->z_inst,
							   FID_Z);

  bool success = true;
  for(int i = saxpy_args->bounds.lo; i <= saxpy_args->bounds.hi; i++) {
    float expected = saxpy_args->alpha * ra_x[i] + ra_y[i] + 1.0;
    float actual = ra_z[i];

    // FMAs are too accurate
    float diff = (actual >= expected) ? actual - expected : expected - actual;
    float relative = diff / expected;
    if (relative < 1e-6) {
      // ok
    } else {
      printf("Index: %d Expected: %.8g Actual: %.8g\n", i, expected, actual);
      success = false;
      break;
    }
  }
  if (success)
    printf("SUCCESS!\n\n");
  else
    printf("FAILURE!\n\n");
}

#ifdef USE_CUDA
extern void gpu_saxpy_task(const void *args, size_t arglen,
                           const void *userdata, size_t userlen, Processor p);
#endif

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CPU_SAXPY_TASK, cpu_saxpy_task);
#ifdef USE_CUDA
  rt.register_task(GPU_SAXPY_TASK, gpu_saxpy_task);
#endif
  rt.register_task(CHECK_RESULT_TASK, check_result_task);
  
  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    Machine::ProcessorQuery query(Machine::get_machine());
    query.only_kind(Processor::LOC_PROC);
    p = query.first();
  }
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}
