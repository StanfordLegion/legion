#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>  //sleep

#include "realm.h"
#include "realm/fpga/fpga_utils.h"

#include "realm/fpga/xcl2.hpp"

using namespace Realm;

#define DEFAULT_COMP_UNITS 1
#define DATA_SIZE 12

enum
{
    FID_X = 101,
    FID_Y = 102,
    FID_Z = 103,
};

// execute a task on FPGA Processor
Logger log_app("app");

enum
{
    TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
    FPGA_TASK,
};

struct FPGAArgs
{
    RegionInstance x_inst, y_inst, z_inst;
    Rect<1> bounds;
};

void fpga_task(const void *args, size_t arglen,
               const void *userdata, size_t userlen, Processor p)
{
    log_app.print() << "fpga task started";
    const FPGAArgs &local_args = *(const FPGAArgs *)args;

    // get affine accessors for each of our three instances
    AffineAccessor<int, 1> ra_x = AffineAccessor<int, 1>(local_args.x_inst,
                                                         FID_X);
    AffineAccessor<int, 1> ra_y = AffineAccessor<int, 1>(local_args.y_inst,
                                                         FID_Y);
    AffineAccessor<int, 1> ra_z = AffineAccessor<int, 1>(local_args.z_inst,
                                                         FID_Z);
    size_t data_size = local_args.bounds.volume();

    int num_cu = DEFAULT_COMP_UNITS;
    std::vector<cl::Kernel> krnls(num_cu);
    cl::Program program = FPGAGetCurrentProgram();
    cl_int err;
    // Creating Kernel objects
    for (int i = 0; i < num_cu; i++)
    {
        OCL_CHECK(err, krnls[i] = cl::Kernel(program, "vadd", &err));
    }

    // Creating sub-buffers
    cl::Buffer device_buff = FPGAGetCurrentBuffer();
    void *base_ptr_sys = FPGAGetBasePtrSys();
    auto chunk_size = data_size / num_cu;
    size_t vector_size_bytes = sizeof(int) * chunk_size;
    std::vector<cl::Buffer> buffer_in1(num_cu);
    std::vector<cl::Buffer> buffer_in2(num_cu);
    std::vector<cl::Buffer> buffer_output(num_cu);

    for (int i = 0; i < num_cu; i++) {
        cl_buffer_region buffer_in1_info = {(uint64_t)ra_x.ptr(0) - (uint64_t)base_ptr_sys + i * vector_size_bytes, vector_size_bytes};
        OCL_CHECK(err, buffer_in1[i] = device_buff.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in1_info, &err));
        cl_buffer_region buffer_in2_info = {(uint64_t)ra_y.ptr(0) - (uint64_t)base_ptr_sys + i * vector_size_bytes, vector_size_bytes};
        OCL_CHECK(err, buffer_in2[i] = device_buff.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in2_info, &err));
        cl_buffer_region buffer_output_info = {(uint64_t)ra_z.ptr(0) - (uint64_t)base_ptr_sys + i * vector_size_bytes, vector_size_bytes};
        OCL_CHECK(err, buffer_output[i] = device_buff.createSubBuffer(CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_output_info, &err));
        log_app.info() << "buffer_output_info " << buffer_output_info.origin << " " << buffer_output_info.size;
    }

    for (int i = 0; i < num_cu; i++) {
        int narg = 0;

        // Setting kernel arguments
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in1[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in2[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_output[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, (int)chunk_size));
    }

    cl::Event task_events[num_cu];
    cl::CommandQueue command_queue = FPGAGetCurrentCommandQueue();
    for (int i = 0; i < num_cu; i++) {
        // Launch the kernel
        OCL_CHECK(err, err = command_queue.enqueueTask(krnls[i], nullptr, &task_events[i]));
    }

    std::vector<cl::Event> wait_events[num_cu];
    // Copy result from device global memory to host local memory
    for (int i = 0; i < num_cu; i++) {
        wait_events[i].push_back(task_events[i]);
        OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects({buffer_output[i]}, CL_MIGRATE_MEM_OBJECT_HOST, &wait_events[i], nullptr));
    }
    // OCL_CHECK(err, err = command_queue.finish());
    OCL_CHECK(err, err = command_queue.flush());

    log_app.print() << "fpga kernels flushed";
}

void top_level_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
    log_app.print() << "top task running on " << p;
    Machine machine = Machine::get_machine();
    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for (std::set<Processor>::const_iterator it = all_processors.begin();
         it != all_processors.end();
         it++)
    {
        Processor pp = (*it);
        if (pp.kind() == Processor::FPGA_PROC)
        {
            Memory cpu_mem = Memory::NO_MEMORY;
            Memory fpga_mem = Memory::NO_MEMORY;
            std::set<Memory> visible_mems;
            machine.get_visible_memories(pp, visible_mems);
            for (std::set<Memory>::const_iterator it = visible_mems.begin();
                 it != visible_mems.end(); it++)
            {
                if (it->kind() == Memory::FPGA_MEM)
                {
                    fpga_mem = *it;
                    log_app.print() << "fpga memory: " << *it << " capacity="
                                    << (it->capacity() >> 20) << " MB";
                }
                if (it->kind() == Memory::SYSTEM_MEM)
                {
                    cpu_mem = *it;
                    log_app.print() << "sys memory: " << *it << " capacity="
                                    << (it->capacity() >> 20) << " MB";
                }
            }

            int init_x_value = 1;
            int init_y_value = 2;
            int init_z_value = 9;

            Rect<1> bounds(0, DATA_SIZE - 1);

            std::map<FieldID, size_t> field_sizes;
            field_sizes[FID_X] = sizeof(int);
            field_sizes[FID_Y] = sizeof(int);
            field_sizes[FID_Z] = sizeof(int);

            RegionInstance cpu_inst;
            RegionInstance::create_instance(cpu_inst, cpu_mem,
                                            bounds, field_sizes,
                                            0 /*SOA*/, ProfilingRequestSet())
                .wait();
            log_app.print() << "created cpu memory instance: " << cpu_inst;

            CopySrcDstField cpu_x_field, cpu_y_field, cpu_z_field;
            cpu_x_field.inst = cpu_inst;
            cpu_x_field.field_id = FID_X;
            cpu_x_field.size = sizeof(int);

            cpu_y_field.inst = cpu_inst;
            cpu_y_field.field_id = FID_Y;
            cpu_y_field.size = sizeof(int);

            cpu_z_field.inst = cpu_inst;
            cpu_z_field.field_id = FID_Z;
            cpu_z_field.size = sizeof(int);

            RegionInstance fpga_inst;
            RegionInstance::create_instance(fpga_inst, fpga_mem,
                                            bounds, field_sizes,
                                            0 /*SOA*/, ProfilingRequestSet())
                .wait();
            log_app.print() << "created fpga memory instance: " << fpga_inst;

            CopySrcDstField fpga_x_field, fpga_y_field, fpga_z_field;
            fpga_x_field.inst = fpga_inst;
            fpga_x_field.field_id = FID_X;
            fpga_x_field.size = sizeof(int);

            fpga_y_field.inst = fpga_inst;
            fpga_y_field.field_id = FID_Y;
            fpga_y_field.size = sizeof(int);

            fpga_z_field.inst = fpga_inst;
            fpga_z_field.field_id = FID_Z;
            fpga_z_field.size = sizeof(int);

            AffineAccessor<int, 1> fpga_ra_x = AffineAccessor<int, 1>(fpga_inst, FID_X);
            AffineAccessor<int, 1> cpu_ra_y = AffineAccessor<int, 1>(cpu_inst, FID_Y);
            AffineAccessor<int, 1> fpga_ra_y = AffineAccessor<int, 1>(fpga_inst, FID_Y);

            //Test fill: fill fpga memory directly
            Event fill_x;
            {
                std::vector<CopySrcDstField> fill_vec;
                fill_vec.push_back(fpga_x_field);
                fill_x = bounds.fill(fill_vec, ProfilingRequestSet(),
                                     &init_x_value, sizeof(init_x_value));
            }
            fill_x.wait();

            Event fill_z;
            {
                std::vector<CopySrcDstField> fill_vec;
                fill_vec.push_back(fpga_z_field);
                fill_z = bounds.fill(fill_vec, ProfilingRequestSet(),
                                     &init_z_value, sizeof(init_z_value));
            }
            fill_z.wait();

            printf("fpga_ra_x:\n");
            for (int i = bounds.lo; i <= bounds.hi; i++)
            {
                printf("%d ", fpga_ra_x[i]);
            }
            printf("\n");

            // fill cpu mem and copy to fpga mem
            Event fill_y_cpu;
            {
                std::vector<CopySrcDstField> fill_vec;
                fill_vec.push_back(cpu_y_field);
                fill_y_cpu = bounds.fill(fill_vec, ProfilingRequestSet(),
                                         &init_y_value, sizeof(init_y_value));
            }
            fill_y_cpu.wait();

            Event copy_y;
            {
                std::vector<CopySrcDstField> srcs, dsts;
                srcs.push_back(cpu_y_field);
                dsts.push_back(fpga_y_field);
                copy_y = bounds.copy(srcs, dsts, ProfilingRequestSet());
            }
            copy_y.wait();

            printf("cpu_ra_y:\n");
            for (int i = bounds.lo; i <= bounds.hi; i++)
            {
                printf("%d ", cpu_ra_y[i]);
            }
            printf("\n");
            printf("fpga_ra_y:\n");
            for (int i = bounds.lo; i <= bounds.hi; i++)
            {
                printf("%d ", fpga_ra_y[i]);
            }
            printf("\n");

            FPGAArgs fpga_args;
            fpga_args.x_inst = fpga_inst;
            fpga_args.y_inst = fpga_inst;
            fpga_args.z_inst = fpga_inst;
            fpga_args.bounds = bounds;
            Event e = pp.spawn(FPGA_TASK, &fpga_args, sizeof(fpga_args));

            // Copy back
            Event z_ready;
            {
                std::vector<CopySrcDstField> srcs, dsts;
                srcs.push_back(fpga_z_field);
                dsts.push_back(cpu_z_field);
                z_ready = bounds.copy(srcs, dsts, ProfilingRequestSet(), e);
            }
            z_ready.wait();

            AffineAccessor<int, 1> ra_z = AffineAccessor<int, 1>(cpu_inst, FID_Z);
            for (int i = bounds.lo; i <= bounds.hi; i++)
            {
                printf("%d ", ra_z[i]);
            }
            printf("\n");
        }
    }

    log_app.print() << "all done!";
}

int main(int argc, char **argv)
{
    // sleep(20);
    Runtime rt;
    rt.init(&argc, &argv);
    rt.register_task(TOP_LEVEL_TASK, top_level_task);

    Processor::register_task_by_kind(Processor::FPGA_PROC, false /*!global*/,
                                     FPGA_TASK,
                                     CodeDescriptor(fpga_task),
                                     ProfilingRequestSet())
        .wait();

    // select a processor to run the top level task on
    Processor p = Machine::ProcessorQuery(Machine::get_machine())
                      .only_kind(Processor::LOC_PROC)
                      .first();
    assert(p.exists());

    // collective launch of a single task - everybody gets the same finish event
    Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

    // request shutdown once that task is complete
    rt.shutdown(e);

    // now sleep this thread until that shutdown actually happens
    rt.wait_for_shutdown();

    return 0;
}
