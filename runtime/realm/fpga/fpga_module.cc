#include "realm/fpga/fpga_module.h"

#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/utils.h"

#include <sys/stat.h>
#include <sys/mman.h>

namespace Realm
{
    namespace FPGA
    {

        namespace ThreadLocal
        {
            static REALM_THREAD_LOCAL FPGAProcessor *current_fpga_proc = NULL;
        }

        Logger log_fpga("fpga");

        // need types with various powers-of-2 size/alignment - we have up to
        //  uint64_t as builtins, but we need trivially-copyable 16B and 32B things
        struct dummy_16b_t
        {
            uint64_t a, b;
        };
        struct dummy_32b_t
        {
            uint64_t a, b, c, d;
        };
        REALM_ALIGNED_TYPE_CONST(aligned_16b_t, dummy_16b_t, 16);
        REALM_ALIGNED_TYPE_CONST(aligned_32b_t, dummy_32b_t, 32);

        template <typename T>
        static void fpga_memcpy_2d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
                                         uintptr_t src_base, uintptr_t src_lstride,
                                         size_t bytes, size_t lines)
        {
            for (size_t i = 0; i < lines; i++)
            {
                std::copy(reinterpret_cast<const T *>(src_base),
                          reinterpret_cast<const T *>(src_base + bytes),
                          reinterpret_cast<T *>(dst_base));
                // manual strength reduction
                src_base += src_lstride;
                dst_base += dst_lstride;
            }
        }

        static void fpga_memcpy_2d(uintptr_t dst_base, uintptr_t dst_lstride,
                                   uintptr_t src_base, uintptr_t src_lstride,
                                   size_t bytes, size_t lines)
        {
            // by subtracting 1 from bases, strides, and lengths, we get LSBs set
            //  based on the common alignment of every parameter in the copy
            unsigned alignment = ((dst_base - 1) & (dst_lstride - 1) &
                                  (src_base - 1) & (src_lstride - 1) &
                                  (bytes - 1));
            // TODO: consider jump table approach?
            if ((alignment & 31) == 31)
                fpga_memcpy_2d_typed<aligned_32b_t>(dst_base, dst_lstride,
                                                    src_base, src_lstride,
                                                    bytes, lines);
            else if ((alignment & 15) == 15)
                fpga_memcpy_2d_typed<aligned_16b_t>(dst_base, dst_lstride,
                                                    src_base, src_lstride,
                                                    bytes, lines);
            else if ((alignment & 7) == 7)
                fpga_memcpy_2d_typed<uint64_t>(dst_base, dst_lstride, src_base, src_lstride,
                                               bytes, lines);
            else if ((alignment & 3) == 3)
                fpga_memcpy_2d_typed<uint32_t>(dst_base, dst_lstride, src_base, src_lstride,
                                               bytes, lines);
            else if ((alignment & 1) == 1)
                fpga_memcpy_2d_typed<uint16_t>(dst_base, dst_lstride, src_base, src_lstride,
                                               bytes, lines);
            else
                fpga_memcpy_2d_typed<uint8_t>(dst_base, dst_lstride, src_base, src_lstride,
                                              bytes, lines);
        }

        /**
         * FPGAWorker: it is responsible for making progress on one or more Command Queues.
         * This may be done directly by an FPGAProcessor or in a background thread spawned for the purpose.
         */
        FPGAWorker::FPGAWorker(void)
            : BackgroundWorkItem("FPGA device worker"),
              condvar(lock),
              core_rsrv(0),
              worker_thread(0),
              thread_sleeping(false),
              worker_shutdown_requested(false)
        {
        }

        FPGAWorker::~FPGAWorker(void)
        {
            // shutdown should have already been called
            assert(worker_thread == 0);
        }

        void FPGAWorker::start_background_thread(
            Realm::CoreReservationSet &crs,
            size_t stack_size)
        {
            assert(manager == 0);
            core_rsrv = new Realm::CoreReservation("A worker thread", crs,
                                                   Realm::CoreReservationParameters());
            Realm::ThreadLaunchParameters tlp;
            worker_thread = Realm::Thread::create_kernel_thread<FPGAWorker, &FPGAWorker::thread_main>(this, tlp, *core_rsrv, 0);
        }

        void FPGAWorker::shutdown_background_thread(void)
        {
            {
                AutoLock<> al(lock);
                worker_shutdown_requested.store(true);
                if (thread_sleeping)
                {
                    thread_sleeping = false;
                    condvar.broadcast();
                }
            }

            worker_thread->join();
            delete worker_thread;
            worker_thread = 0;

            delete core_rsrv;
            core_rsrv = 0;
        }

        void FPGAWorker::add_queue(FPGAQueue *queue)
        {
            bool was_empty = false;
            {
                AutoLock<> al(lock);
#ifdef DEBUG_REALM
                // insist that the caller de-duplicate these
                for (ActiveQueue::iterator it = active_queues.begin();
                     it != active_queues.end();
                     ++it)
                    assert(*it != queue);
#endif
                was_empty = active_queues.empty();
                active_queues.push_back(queue);
                if (thread_sleeping)
                {
                    thread_sleeping = false;
                    condvar.broadcast();
                }
            }
            // if we're a background work item, request attention if needed
            if (was_empty && (manager != 0))
                make_active();
        }

        bool FPGAWorker::do_work(TimeLimit work_until)
        {
            // pop the first queue off the list and immediately become re-activ if more queues remain
            FPGAQueue *queue = 0;
            bool still_not_empty = false;
            {
                AutoLock<> al(lock);

                assert(!active_queues.empty());
                queue = active_queues.front();
                active_queues.pop_front();
                still_not_empty = !active_queues.empty();
            }
            if (still_not_empty)
                make_active();
            // do work for the queue we popped, paying attention to the cutoff time
            bool requeue_q = false;

            if (queue->reap_events(work_until))
            {
                // still work (e.g. copies) to do
                if (work_until.is_expired())
                {
                    // out of time - save it for later
                    requeue_q = true;
                }
                else if (queue->issue_copies(work_until))
                    requeue_q = true;
            }

            bool was_empty = false;
            if (requeue_q)
            {
                AutoLock<> al(lock);
                was_empty = active_queues.empty();
                active_queues.push_back(queue);
            }
            // note that we can need requeueing even if we called make_active above!
            return was_empty;
        }

        bool FPGAWorker::process_queues(bool sleep_on_empty)
        {
            FPGAQueue *cur_queue = 0;
            FPGAQueue *first_queue = 0;
            bool requeue_queue = false;
            while (true)
            {
                // grab the front queue in the list
                {
                    AutoLock<> al(lock);
                    // if we didn't finish work on the queue from the previous
                    // iteration, add it back to the end
                    if (requeue_queue)
                        active_queues.push_back(cur_queue);

                    while (active_queues.empty())
                    {
                        // sleep only if this was the first attempt to get a queue
                        if (sleep_on_empty && (first_queue == 0) &&
                            !worker_shutdown_requested.load())
                        {
                            thread_sleeping = true;
                            condvar.wait();
                        }
                        else
                            return false;
                    }
                    cur_queue = active_queues.front();
                    // did we wrap around?  if so, stop for now
                    if (cur_queue == first_queue)
                        return true;

                    active_queues.pop_front();
                    if (!first_queue)
                        first_queue = cur_queue;
                }
                // and do some work for it
                requeue_queue = false;
                // reap_events report whether any kind of work
                if (!cur_queue->reap_events(TimeLimit()))
                    continue;
                if (!cur_queue->issue_copies(TimeLimit()))
                    continue;
                // if we fall, the queues never went empty at any time, so it's up to us to requeue
                requeue_queue = true;
            }
        }

        void FPGAWorker::thread_main(void)
        {
            // TODO: consider busy-waiting in some cases to reduce latency?
            while (!worker_shutdown_requested.load())
            {
                bool work_left = process_queues(true);
                // if there was work left, yield our thread
                // for now to avoid a tight spin loop
                if (work_left)
                    Realm::Thread::yield();
            }
        }

        /**
         * FPGAWorkFence: used to determine when a device kernel completes execution
         */
        FPGAWorkFence::FPGAWorkFence(Realm::Operation *op)
            : Realm::Operation::AsyncWorkItem(op)
        {
        }

        void FPGAWorkFence::request_cancellation(void)
        {
            // ignored - no way to shoot down FPGA work
        }

        void FPGAWorkFence::print(std::ostream &os) const
        {
            os << "FPGAWorkFence";
        }

        void FPGAWorkFence::enqueue(FPGAQueue *queue)
        {
            queue->add_fence(this);
        }

        /**
         * FPGAQueue: device command queue
         */
        FPGAQueue::FPGAQueue(FPGADevice *fpga_device, FPGAWorker *fpga_worker, cl::CommandQueue &command_queue)
            : fpga_device(fpga_device), fpga_worker(fpga_device->fpga_worker), command_queue(command_queue)
        {
            log_fpga.info() << "Create FPGAQueue ";
            pending_events.clear();
            pending_copies.clear();
        }

        FPGAQueue::~FPGAQueue(void)
        {
        }

        cl::CommandQueue &FPGAQueue::get_command_queue() const
        {
            return command_queue;
        }

        void FPGAQueue::add_fence(FPGAWorkFence *fence)
        {
            cl::Event opencl_event;
            cl_int err = 0;
            OCL_CHECK(err, err = command_queue.enqueueMarkerWithWaitList(nullptr, &opencl_event));
            add_event(opencl_event, fence, 0);
        }

        void FPGAQueue::add_notification(FPGACompletionNotification *notification)
        {
            cl::Event opencl_event;
            cl_int err = 0;
            OCL_CHECK(err, err = command_queue.enqueueMarkerWithWaitList(nullptr, &opencl_event));

            add_event(opencl_event, 0, notification);
        }

        // add event to worker so it can be progressed
        void FPGAQueue::add_event(cl::Event opencl_event,
                                  FPGAWorkFence *fence,
                                  FPGACompletionNotification *n)

        {
            bool add_to_worker = false;
            // assert(opencl_event != nullptr);
            {
                AutoLock<> al(mutex);
                // remember to add ourselves
                // to the worker if we didn't already have work
                add_to_worker = pending_events.empty() &&
                                pending_copies.empty();
                PendingEvent e;
                e.opencl_event = opencl_event;
                e.fence = fence;
                e.notification = n;
                pending_events.push_back(e);
            }
            if (add_to_worker)
                fpga_worker->add_queue(this);
        }

        bool FPGAQueue::has_work(void) const
        {
            return (!pending_events.empty() ||
                    !pending_copies.empty());
        }

        bool FPGAQueue::reap_events(TimeLimit work_until)
        {
            // peek at the first event
            cl::Event opencl_event;
            FPGACompletionNotification *notification = 0;
            bool event_valid = false;
            {
                AutoLock<> al(mutex);

                if (pending_events.empty())
                    // no events left, but command queue
                    // might have other work left
                    return has_work();
                opencl_event = pending_events.front().opencl_event;
                notification = pending_events.front().notification;
                event_valid = true;
            }
            // we'll keep looking at events
            // until we find one that hasn't triggered
            bool work_left = true;
            while (event_valid)
            {
                cl_int err = 0;
                cl_int status;
                OCL_CHECK(err, err = opencl_event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));
                if (status == CL_QUEUED || status == CL_SUBMITTED || status == CL_RUNNING)
                {
                    // event is not finished - check again later
                    return true;
                }
                else if (status != CL_COMPLETE)
                {
                    log_fpga.fatal() << "Error reported on FPGA " << fpga_device->name;
                }

                // this event has triggered
                FPGAWorkFence *fence = 0;
                {
                    AutoLock<> al(mutex);

                    const PendingEvent &e = pending_events.front();
                    assert(e.opencl_event == opencl_event);
                    fence = e.fence;
                    notification = e.notification;
                    pending_events.pop_front();

                    if (pending_events.empty())
                    {
                        event_valid = false;
                        work_left = has_work();
                    }
                    else
                    {
                        opencl_event = pending_events.front().opencl_event;
                    }
                }
                if (fence)
                {
                    fence->mark_finished(true /*successful*/); // set preconditions for next tasks
                }
                if (notification)
                {
                    notification->request_completed();
                }
                if (event_valid && work_until.is_expired())
                    return true;
            }
            // if we get here, we ran out of events, but there might have been
            // other kinds of work that we need to let the caller know about
            return work_left;
        }

        void FPGAQueue::add_copy(FPGADeviceMemcpy *copy)
        {
            bool add_to_worker = false;
            {
                AutoLock<> al(mutex);
                // add if we haven't been added yet
                add_to_worker =
                    pending_copies.empty() && pending_events.empty();
                pending_copies.push_back(copy);
            }
            if (add_to_worker)
                fpga_worker->add_queue(this);
        }

        bool FPGAQueue::issue_copies(TimeLimit work_until)
        {
            while (true)
            {
                // if we cause the list to go empty,
                // we stop even if more copies show
                // up because we don't want to requeue ourselves twice
                bool list_exhausted = false;
                FPGADeviceMemcpy *copy = 0;
                {
                    AutoLock<> al(mutex);
                    if (pending_copies.empty())
                        // no copies left,
                        // but queue might have other work left
                        return has_work();
                    copy = pending_copies.front();
                    pending_copies.pop_front();
                    list_exhausted = !has_work();
                }
                copy->execute(this);
                delete copy;
                // if the list was exhausted, let the caller know
                if (list_exhausted)
                    return false;

                // if we still have work, but time's up, return also
                if (work_until.is_expired())
                    return true;
            }
            return false; // should never reach here
        }

        FPGADevice::FPGADevice(cl::Device &device, std::string name, std::string xclbin, FPGAWorker *fpga_worker, size_t fpga_coprocessor_num_cu, std::string fpga_coprocessor_kernel)
            : name(name), fpga_worker(fpga_worker), fpga_coprocessor_num_cu(fpga_coprocessor_num_cu), fpga_coprocessor_kernel(fpga_coprocessor_kernel)
        {
            cl_int err;
            this->device = device;
            // Create FPGA context and command queue for each device
            OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
            OCL_CHECK(err, command_queue = cl::CommandQueue(context, device,
                                                            CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

            // Program the device
            auto fileBuf = xcl::read_binary_file(xclbin);
            cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
            log_fpga.info() << "Trying to program device " << device.getInfo<CL_DEVICE_NAME>();
            OCL_CHECK(err, program = cl::Program(context, {device}, bins, nullptr, &err));

            fpga_mem = nullptr;
            local_sysmem = nullptr;
            local_ibmem = nullptr;
            fpga_queue = nullptr;
            create_fpga_queues();
        }

        FPGADevice::~FPGADevice()
        {
            if (fpga_queue != nullptr)
            {
                delete fpga_queue;
                fpga_queue = nullptr;
            }
            command_queue.finish();
        }

        void FPGADevice::create_dma_channels(RuntimeImpl *runtime)
        {
            if (!fpga_mem)
            {
                return;
            }

            const std::vector<MemoryImpl *> &local_mems = runtime->nodes[Network::my_node_id].memories;
            for (std::vector<Realm::MemoryImpl *>::const_iterator it = local_mems.begin();
                 it != local_mems.end();
                 it++)
            {
                if ((*it)->lowlevel_kind == Memory::SYSTEM_MEM)
                {
                    this->local_sysmem = *it;
                    log_fpga.info() << "local_sysmem " << std::hex << (*it)->me.id << std::dec << " kind: " << (*it)->kind << " low-level kind: " << (*it)->lowlevel_kind;
                    break;
                }
            }

            const std::vector<IBMemory *> &local_ib_mems = runtime->nodes[Network::my_node_id].ib_memories;
            for (std::vector<Realm::IBMemory *>::const_iterator it = local_ib_mems.begin();
                 it != local_ib_mems.end();
                 it++)
            {
                if ((*it)->lowlevel_kind == Memory::REGDMA_MEM)
                {
                    this->local_ibmem = *it;
                    log_fpga.info() << "local_ibmem " << std::hex << (*it)->me.id << std::dec << " kind: " << (*it)->kind << " low-level kind: " << (*it)->lowlevel_kind;
                    break;
                }
            }

            runtime->add_dma_channel(new FPGAfillChannel(this, &runtime->bgwork));
            runtime->add_dma_channel(new FPGAChannel(this, XFER_FPGA_IN_DEV, &runtime->bgwork));
            runtime->add_dma_channel(new FPGAChannel(this, XFER_FPGA_TO_DEV, &runtime->bgwork));
            runtime->add_dma_channel(new FPGAChannel(this, XFER_FPGA_FROM_DEV, &runtime->bgwork));
            runtime->add_dma_channel(new FPGAChannel(this, XFER_FPGA_COMP, &runtime->bgwork));

            Machine::MemoryMemoryAffinity mma;
            mma.m1 = fpga_mem->me;
            mma.m2 = local_sysmem->me;
            mma.bandwidth = 20; // TODO
            mma.latency = 200;
            runtime->add_mem_mem_affinity(mma);

            mma.m1 = fpga_mem->me;
            mma.m2 = local_ibmem->me;
            mma.bandwidth = 20; // TODO
            mma.latency = 200;
            runtime->add_mem_mem_affinity(mma);

            // TODO: create p2p channel
            // runtime->add_dma_channel(new FPGAChannel(this, XFER_FPGA_PEER_DEV, &runtime->bgwork));
        }

        void FPGADevice::create_fpga_mem(RuntimeImpl *runtime, size_t size)
        {
            // TODO: only use membank 0 for now
            cl_int err;
            void *base_ptr_sys = nullptr;
            posix_memalign((void **)&base_ptr_sys, 4096, size);
            OCL_CHECK(err, buff = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, base_ptr_sys, &err));
            Memory m = runtime->next_local_memory_id();
            this->fpga_mem = new FPGADeviceMemory(m, this, base_ptr_sys, size);
            runtime->add_memory(fpga_mem);
            log_fpga.info() << "create_fpga_mem: "
                            << device.getInfo<CL_DEVICE_NAME>()
                            << ", size = "
                            << (size >> 20) << " MB"
                            << ", base_ptr_sys = " << base_ptr_sys;
        }

        void FPGADevice::create_fpga_ib(RuntimeImpl *runtime, size_t size)
        {
            // TODO: only use membank 0 for now
            cl_int err;
            void *base_ptr_sys = nullptr;
            posix_memalign((void **)&base_ptr_sys, 4096, size);
            OCL_CHECK(err, ib_buff = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, base_ptr_sys, &err));
            Memory m = runtime->next_local_ib_memory_id();
            IBMemory *ib_mem;
            ib_mem = new IBMemory(m, size,
                                  MemoryImpl::MKIND_FPGA, Memory::FPGA_MEM,
                                  base_ptr_sys, 0);
            this->fpga_ib = ib_mem;
            runtime->add_ib_memory(ib_mem);
            log_fpga.info() << "create_fpga_ib: "
                            << device.getInfo<CL_DEVICE_NAME>()
                            << ", size = "
                            << (size >> 20) << " MB"
                            << ", base_ptr_sys = " << base_ptr_sys;
        }

        void FPGADevice::create_fpga_queues()
        {
            fpga_queue = new FPGAQueue(this, fpga_worker, command_queue);
        }

        void FPGADevice::copy_to_fpga(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification)
        {
            log_fpga.info() << "copy_to_fpga: src = " << src << " dst = " << dst
                            << " src_offset = " << src_offset << "dst_offset = " << dst_offset
                            << " bytes = " << bytes << " notification = " << notification;

            FPGADeviceMemcpy *copy = new FPGADeviceMemcpy1D(this,
                                                            dst,
                                                            src,
                                                            bytes,
                                                            dst_offset,
                                                            FPGA_MEMCPY_HOST_TO_DEVICE,
                                                            notification);
            fpga_queue->add_copy(copy);
        }

        void FPGADevice::copy_from_fpga(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification)
        {
            log_fpga.info() << "copy_from_fpga: src = " << src << " dst = " << dst
                            << " src_offset = " << src_offset << "dst_offset = " << dst_offset
                            << " bytes = " << bytes << " notification = " << notification;

            FPGADeviceMemcpy *copy = new FPGADeviceMemcpy1D(this,
                                                            dst,
                                                            src,
                                                            bytes,
                                                            src_offset,
                                                            FPGA_MEMCPY_DEVICE_TO_HOST,
                                                            notification);
            fpga_queue->add_copy(copy);
        }

        void FPGADevice::copy_within_fpga(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification)
        {
            log_fpga.info() << "copy_within_fpga: src = " << src << " dst = " << dst
                            << " src_offset = " << src_offset << "dst_offset = " << dst_offset
                            << " bytes = " << bytes << " notification = " << notification;
            FPGADeviceMemcpy *copy = new FPGADeviceMemcpy1D(this,
                                                            dst,
                                                            src,
                                                            bytes,
                                                            dst_offset,
                                                            FPGA_MEMCPY_DEVICE_TO_DEVICE,
                                                            notification);
            fpga_queue->add_copy(copy);
        }

        void FPGADevice::copy_to_peer(FPGADevice *dst_dev, void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification)
        {
            log_fpga.info() << "copy_to_peer(not implemented!): dst_dev = " << dst_dev
                            << "src = " << src << " dst = " << dst
                            << " src_offset = " << src_offset << "dst_offset = " << dst_offset
                            << " bytes = " << bytes << " notification = " << notification;
            assert(0);
        }

        // TODO: FPGA coprocessor kernels are invoked in this function
        void FPGADevice::comp(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification)
        {
            log_fpga.info() << "comp: src = " << src << " dst = " << dst
                            << " src_offset = " << src_offset << "dst_offset = " << dst_offset
                            << " bytes = " << bytes << " notification = " << notification;
            // An example of invoking an FPGA coprocessor kernel
            // program device
            int num_cu = fpga_coprocessor_num_cu;
            std::vector<cl::Kernel> krnls(num_cu);
            cl_int err;
            // Creating Kernel objects
            for (int i = 0; i < num_cu; i++)
            {
                OCL_CHECK(err, krnls[i] = cl::Kernel(program, fpga_coprocessor_kernel.c_str(), &err));
            }
            // Creating sub-buffers
            auto chunk_size = bytes / num_cu;
            size_t vector_size_bytes = sizeof(int) * chunk_size;
            std::vector<cl::Buffer> buffer_in1(num_cu);
            std::vector<cl::Buffer> buffer_in2(num_cu);
            std::vector<cl::Buffer> buffer_output(num_cu);

            // I/O data vectors
            std::vector<int, aligned_allocator<int>> source_in2(bytes, 1);

            for (int i = 0; i < num_cu; i++)
            {
                OCL_CHECK(err, buffer_in2[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
                                                          source_in2.data() + i * chunk_size, &err));
            }
            for (int i = 0; i < num_cu; i++)
            {
                cl_buffer_region buffer_in1_info = {src_offset + i * vector_size_bytes, vector_size_bytes};
                OCL_CHECK(err, buffer_in1[i] = ib_buff.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in1_info, &err));
                cl_buffer_region buffer_output_info = {dst_offset + i * vector_size_bytes, vector_size_bytes};
                OCL_CHECK(err, buffer_output[i] = ib_buff.createSubBuffer(CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_output_info, &err));
            }

            for (int i = 0; i < num_cu; i++)
            {
                int narg = 0;

                // Setting kernel arguments
                OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in1[i]));
                OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in2[i]));
                OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_output[i]));
                OCL_CHECK(err, err = krnls[i].setArg(narg++, (int)chunk_size));
            }

            cl::Event task_events[num_cu];
            for (int i = 0; i < num_cu; i++)
            {
                // Launch the kernel
                OCL_CHECK(err, err = command_queue.enqueueTask(krnls[i], nullptr, &task_events[i]));
            }

            std::vector<cl::Event> wait_events[num_cu];
            // Copy result from device global memory to host local memory
            for (int i = 0; i < num_cu; i++)
            {
                wait_events[i].push_back(task_events[i]);
                OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects({buffer_output[i]}, CL_MIGRATE_MEM_OBJECT_HOST, &wait_events[i], nullptr));
            }

            OCL_CHECK(err, err = command_queue.flush());
            fpga_queue->add_notification(notification);
        }

        bool FPGADevice::is_in_buff(void *ptr)
        {
            uint64_t base_ptr = (uint64_t)(fpga_mem->base_ptr_sys);
            if ((uint64_t)ptr >= base_ptr && (uint64_t)ptr <= base_ptr + fpga_mem->size)
            {
                return true;
            }
            return false;
        }

        bool FPGADevice::is_in_ib_buff(void *ptr)
        {
            uint64_t base_ptr = (uint64_t)(fpga_ib->get_direct_ptr(0, 0));
            if ((uint64_t)ptr >= base_ptr && (uint64_t)ptr <= base_ptr + fpga_ib->size)
            {
                return true;
            }
            return false;
        }

        /**
         * Device Memory Copy Operations
         */
        FPGADeviceMemcpy::FPGADeviceMemcpy(FPGADevice *fpga_device,
                                           FPGAMemcpyKind kind,
                                           FPGACompletionNotification *notification)
            : fpga_device(fpga_device), kind(kind), notification(notification)
        {
        }

        /**
         * 1D Memory Copy Operation
         */
        FPGADeviceMemcpy1D::FPGADeviceMemcpy1D(FPGADevice *fpga_device,
                                               void *dst,
                                               const void *src,
                                               size_t bytes,
                                               off_t buff_offset,
                                               FPGAMemcpyKind kind,
                                               FPGACompletionNotification *notification)
            : FPGADeviceMemcpy(fpga_device, kind, notification),
              dst(dst), src(src), elmt_size(bytes), buff_offset(buff_offset)
        {
        }

        FPGADeviceMemcpy1D::~FPGADeviceMemcpy1D(void)
        {
        }

        void FPGADeviceMemcpy1D::do_span(off_t pos, size_t len)
        {
            off_t span_start = pos * elmt_size;
            size_t span_bytes = len * elmt_size;
            void *dstptr = ((uint8_t *)dst) + span_start;
            void *srcptr = ((uint8_t *)src) + span_start;
            size_t size = span_bytes;
            cl_int err = 0;
            log_fpga.debug() << "do_span: buff_offset " << buff_offset << " size " << size << " srcptr " << srcptr << " dstptr " << dstptr;
            cl::Buffer *temp_buff = nullptr;
            if (kind == FPGA_MEMCPY_HOST_TO_DEVICE)
            {
                if (fpga_device->is_in_ib_buff(dstptr))
                    temp_buff = &(fpga_device->ib_buff);
                else if (fpga_device->is_in_buff(dstptr))
                    temp_buff = &(fpga_device->buff);
                else
                {
                    log_fpga.error() << "dstptr is not in buffer or ib_buffer";
                    assert(0);
                }
                OCL_CHECK(err, err = fpga_queue->get_command_queue().enqueueWriteBuffer(*temp_buff,     // buffer on the FPGA
                                                                                        CL_FALSE,       // blocking call
                                                                                        buff_offset,    // buffer offset in bytes
                                                                                        size,           // Size in bytes
                                                                                        (void *)srcptr, // Pointer to the data to copy
                                                                                        nullptr, nullptr));
            }
            else if (kind == FPGA_MEMCPY_DEVICE_TO_HOST)
            {
                if (fpga_device->is_in_ib_buff(srcptr))
                    temp_buff = &(fpga_device->ib_buff);
                else if (fpga_device->is_in_buff(srcptr))
                    temp_buff = &(fpga_device->buff);
                else
                {
                    log_fpga.error() << "srcptr is not in buffer or ib_buffer";
                    assert(0);
                }
                OCL_CHECK(err, err = fpga_queue->get_command_queue().enqueueReadBuffer(*temp_buff,     // buffer on the FPGA
                                                                                       CL_FALSE,       // blocking call
                                                                                       buff_offset,    // buffer offset in bytes
                                                                                       size,           // Size in bytes
                                                                                       (void *)dstptr, // Pointer to the data to copy
                                                                                       nullptr, nullptr));
            }
            else if (kind == FPGA_MEMCPY_DEVICE_TO_DEVICE)
            {
                if (fpga_device->is_in_ib_buff(dstptr))
                    temp_buff = &(fpga_device->ib_buff);
                else if (fpga_device->is_in_buff(dstptr))
                    temp_buff = &(fpga_device->buff);
                else
                {
                    log_fpga.error() << "dstptr is not in buffer or ib_buffer";
                    assert(0);
                }
                OCL_CHECK(err, err = fpga_queue->get_command_queue().enqueueWriteBuffer(*temp_buff,     // buffer on the FPGA
                                                                                        CL_FALSE,       // blocking call
                                                                                        buff_offset,    // buffer offset in bytes
                                                                                        size,           // Size in bytes
                                                                                        (void *)srcptr, // Pointer to the data to copy
                                                                                        nullptr, nullptr));
            }
            else if (kind == FPGA_MEMCPY_PEER_TO_PEER)
            {
                log_fpga.error() << "FPGA_MEMCPY_PEER_TO_PEER not implemented";
                assert(0);
            }
            else
            {
                log_fpga.error() << "FPGADeviceMemcpy kind error";
                assert(0);
            }
            OCL_CHECK(err, err = fpga_queue->get_command_queue().flush());
        }

        void FPGADeviceMemcpy1D::execute(FPGAQueue *queue)
        {
            log_fpga.info("FPGADevice memcpy: dst=%p src=%p bytes=%zd kind=%d",
                          dst, src, elmt_size, kind);
            // save queue into local variable
            // for do_span (which may be called indirectly by ElementMask::forall_ranges)
            fpga_queue = queue;
            do_span(0, 1);
            if (notification)
            {
                fpga_queue->add_notification(notification);
            }
            log_fpga.info("fpga memcpy 1d issued: dst=%p src=%p bytes=%zd kind=%d",
                          dst, src, elmt_size, kind);
        }

        FPGAModule::FPGAModule()
            : Module("fpga"), cfg_num_fpgas(0), cfg_use_worker_threads(false),
              cfg_use_shared_worker(true), cfg_fpga_mem_size(4 << 20), cfg_fpga_ib_size(4 << 20),
              cfg_fpga_coprocessor_num_cu(1)
        {
            shared_worker = nullptr;
            cfg_fpga_xclbin = "";
            fpga_devices.clear();
            dedicated_workers.clear();
            fpga_procs_.clear();
            cfg_fpga_coprocessor_kernel = "";
        }

        FPGAModule::~FPGAModule(void)
        {
            if (!this->fpga_devices.empty())
            {
                for (size_t i = 0; i < fpga_devices.size(); i++)
                {

                    // xclClose(fpga_devices[i]->dev_handle);
                    delete this->fpga_devices[i];
                }
            }
        }

        Module *FPGAModule::create_module(RuntimeImpl *runtime, std::vector<std::string> &cmdline)
        {
            FPGAModule *m = new FPGAModule;
            log_fpga.info() << "create_module";
            // first order of business - read command line parameters
            {
                Realm::CommandLineParser cp;
                cp.add_option_int("-ll:fpga", m->cfg_num_fpgas);
                cp.add_option_bool("-ll:fpga_work_thread", m->cfg_use_worker_threads);
                cp.add_option_bool("-ll:fpga_shared_worker", m->cfg_use_shared_worker);
                cp.add_option_int_units("-ll:fpga_size", m->cfg_fpga_mem_size, 'm');
                cp.add_option_int_units("-ll:fpga_ib_size", m->cfg_fpga_ib_size, 'm');
                cp.add_option_string("-ll:fpga_xclbin", m->cfg_fpga_xclbin);
                cp.add_option_int("-ll:fpga_coprocessor_num_cu", m->cfg_fpga_coprocessor_num_cu);
                cp.add_option_string("-ll:fpga_coprocessor_kernel", m->cfg_fpga_coprocessor_kernel);

                bool ok = cp.parse_command_line(cmdline);
                if (!ok)
                {
                    log_fpga.error() << "error reading fpga parameters";
                    exit(1);
                }
            }
            return m;
        }

        // do any general initialization - this is called after all configuration is complete
        void FPGAModule::initialize(RuntimeImpl *runtime)
        {
            log_fpga.info() << "initialize";
            Module::initialize(runtime);

            std::vector<cl::Device> devices = xcl::get_xil_devices();
            if (cfg_num_fpgas > devices.size())
            {
                log_fpga.error() << cfg_num_fpgas << " FPGA Processors requested, but only " << devices.size() << " available!";
                exit(1);
            }

            // if we are using a shared worker, create that next
            if (cfg_use_shared_worker)
            {
                shared_worker = new FPGAWorker;
                if (cfg_use_worker_threads)
                {
                    shared_worker->start_background_thread(
                        runtime->core_reservation_set(),
                        1 << 20); // hardcoded worker stack size
                }
                else
                    shared_worker->add_to_manager(&(runtime->bgwork));
            }

            // set up fpga_devices
            for (unsigned int i = 0; i < cfg_num_fpgas; i++)
            {
                // for each device record the FPGAWorker
                FPGAWorker *worker;
                if (cfg_use_shared_worker)
                {
                    worker = shared_worker;
                }
                else
                {
                    worker = new FPGAWorker;
                    if (cfg_use_worker_threads)
                        worker->start_background_thread(
                            runtime->core_reservation_set(),
                            1 << 20); // hardcoded worker stack size
                    else
                        worker->add_to_manager(&(runtime->bgwork));
                }

                FPGADevice *fpga_device = new FPGADevice(devices[i], "fpga" + std::to_string(i), cfg_fpga_xclbin, worker, cfg_fpga_coprocessor_num_cu, cfg_fpga_coprocessor_kernel);
                fpga_devices.push_back(fpga_device);

                if (!cfg_use_shared_worker)
                {
                    log_fpga.info() << "add to dedicated workers " << worker;
                    dedicated_workers[fpga_device] = worker;
                }
            }
        }

        // create any memories provided by this module (default == do nothing)
        //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
        void FPGAModule::create_memories(RuntimeImpl *runtime)
        {
            log_fpga.info() << "create_memories";
            Module::create_memories(runtime);
            if (cfg_fpga_mem_size > 0)
            {
                for (size_t i = 0; i < cfg_num_fpgas; i++)
                {
                    fpga_devices[i]->create_fpga_mem(runtime, cfg_fpga_mem_size);
                }
            }
            if (cfg_fpga_ib_size > 0)
            {
                for (size_t i = 0; i < cfg_num_fpgas; i++)
                {
                    fpga_devices[i]->create_fpga_ib(runtime, cfg_fpga_ib_size);
                }
            }
        }

        // create any processors provided by the module (default == do nothing)
        //  (each new ProcessorImpl should use a Processor from RuntimeImpl::next_local_processor_id)
        void FPGAModule::create_processors(RuntimeImpl *runtime)
        {
            Module::create_processors(runtime);
            for (size_t i = 0; i < cfg_num_fpgas; i++)
            {
                Processor p = runtime->next_local_processor_id();
                FPGAProcessor *proc = new FPGAProcessor(fpga_devices[i], p, runtime->core_reservation_set());
                fpga_procs_.push_back(proc);
                runtime->add_processor(proc);
                log_fpga.info() << "create fpga processor " << i;

                // create mem affinities to add a proc to machine model
                // create affinities between this processor and system/reg memories
                // if the memory is one we created, use the kernel-reported distance
                // to adjust the answer
                std::vector<MemoryImpl *> &local_mems = runtime->nodes[Network::my_node_id].memories;
                for (std::vector<MemoryImpl *>::iterator it = local_mems.begin();
                     it != local_mems.end();
                     ++it)
                {
                    Memory::Kind kind = (*it)->get_kind();
                    if (kind == Memory::SYSTEM_MEM or kind == Memory::FPGA_MEM)
                    {
                        Machine::ProcessorMemoryAffinity pma;
                        pma.p = p;
                        pma.m = (*it)->me;

                        // use the same made-up numbers as in
                        //  runtime_impl.cc
                        if (kind == Memory::SYSTEM_MEM)
                        {
                            pma.bandwidth = 100; // "large"
                            pma.latency = 5;     // "small"
                        }
                        else if (kind == Memory::FPGA_MEM)
                        {
                            pma.bandwidth = 200; // "large"
                            pma.latency = 10;    // "small"
                        }
                        else
                        {
                            assert(0 && "wrong memory kind");
                        }

                        runtime->add_proc_mem_affinity(pma);
                    }
                }
            }
        }

        // create any DMA channels provided by the module (default == do nothing)
        void FPGAModule::create_dma_channels(RuntimeImpl *runtime)
        {
            log_fpga.info() << "create_dma_channels";
            for (std::vector<FPGADevice *>::iterator it = fpga_devices.begin(); it != fpga_devices.end(); it++)
            {
                (*it)->create_dma_channels(runtime);
            }

            Module::create_dma_channels(runtime);
        }

        // create any code translators provided by the module (default == do nothing)
        void FPGAModule::create_code_translators(RuntimeImpl *runtime)
        {
            log_fpga.info() << "create_code_translators";
            Module::create_code_translators(runtime);
        }

        // clean up any common resources created by the module - this will be called
        //  after all memories/processors/etc. have been shut down and destroyed
        void FPGAModule::cleanup(void)
        {
            log_fpga.info() << "cleanup";
            // clean up worker(s)
            if (shared_worker)
            {
#ifdef DEBUG_REALM
                shared_worker->shutdown_work_item();
#endif
                if (cfg_use_worker_threads)
                    shared_worker->shutdown_background_thread();

                delete shared_worker;
                shared_worker = 0;
            }
            for (std::map<FPGADevice *, FPGAWorker *>::iterator it = dedicated_workers.begin();
                 it != dedicated_workers.end();
                 it++)
            {
                log_fpga.info() << "shutdown worker in cleanup";
                FPGAWorker *worker = it->second;
#ifdef DEBUG_REALM
                worker->shutdown_work_item();
#endif
                if (cfg_use_worker_threads)
                    worker->shutdown_background_thread();

                delete worker;
            }
            dedicated_workers.clear();
            Module::cleanup();
        }

        template <typename T>
        class FPGATaskScheduler : public T
        {
        public:
            FPGATaskScheduler(Processor proc, Realm::CoreReservation &core_rsrv, FPGAProcessor *fpga_proc);
            virtual ~FPGATaskScheduler(void);

        protected:
            virtual bool execute_task(Task *task);
            virtual void execute_internal_task(InternalTask *task);
            FPGAProcessor *fpga_proc_;
        };

        template <typename T>
        FPGATaskScheduler<T>::FPGATaskScheduler(Processor proc,
                                                Realm::CoreReservation &core_rsrv,
                                                FPGAProcessor *fpga_proc) : T(proc, core_rsrv), fpga_proc_(fpga_proc)
        {
        }

        template <typename T>
        FPGATaskScheduler<T>::~FPGATaskScheduler(void)
        {
        }

        template <typename T>
        bool FPGATaskScheduler<T>::execute_task(Task *task)
        {
            assert(ThreadLocal::current_fpga_proc == NULL);
            ThreadLocal::current_fpga_proc = fpga_proc_;
            FPGAQueue *queue = fpga_proc_->fpga_device->fpga_queue;
            log_fpga.info() << "execute_task " << task;

            // we'll use a "work fence" to track when the kernels launched by this task actually
            //  finish - this must be added to the task _BEFORE_ we execute
            FPGAWorkFence *fence = new FPGAWorkFence(task);
            task->add_async_work_item(fence);
            bool ok = T::execute_task(task);
            fence->enqueue(queue);

            assert(ThreadLocal::current_fpga_proc == fpga_proc_);
            ThreadLocal::current_fpga_proc = NULL;
            return ok;
        }

        template <typename T>
        void FPGATaskScheduler<T>::execute_internal_task(InternalTask *task)
        {
            assert(ThreadLocal::current_fpga_proc == NULL);
            ThreadLocal::current_fpga_proc = fpga_proc_;
            log_fpga.info() << "execute_internal_task";
            T::execute_internal_task(task);
            assert(ThreadLocal::current_fpga_proc == fpga_proc_);
            ThreadLocal::current_fpga_proc = NULL;
        }

        FPGAProcessor::FPGAProcessor(FPGADevice *fpga_device, Processor me, Realm::CoreReservationSet &crs)
            : LocalTaskProcessor(me, Processor::FPGA_PROC)
        {
            log_fpga.info() << "FPGAProcessor()";
            this->fpga_device = fpga_device;
            Realm::CoreReservationParameters params;
            params.set_num_cores(1);
            params.set_alu_usage(params.CORE_USAGE_SHARED);
            params.set_fpu_usage(params.CORE_USAGE_SHARED);
            params.set_ldst_usage(params.CORE_USAGE_SHARED);
            params.set_max_stack_size(2 << 20);
            std::string name = stringbuilder() << "fpga proc " << me;
            core_rsrv_ = new Realm::CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS
            UserThreadTaskScheduler *sched = new FPGATaskScheduler<UserThreadTaskScheduler>(me, *core_rsrv_, this);
#else
            KernelThreadTaskScheduler *sched = new FPGATaskScheduler<KernelThreadTaskScheduler>(me, *core_rsrv_, this);
#endif
            set_scheduler(sched);
        }

        FPGAProcessor::~FPGAProcessor(void)
        {
            delete core_rsrv_;
        }

        FPGAProcessor *FPGAProcessor::get_current_fpga_proc(void)
        {
            return ThreadLocal::current_fpga_proc;
        }

        FPGADeviceMemory::FPGADeviceMemory(Memory memory, FPGADevice *device, void *base_ptr_sys, size_t size)
            : LocalManagedMemory(memory, size, MKIND_FPGA, 512, Memory::FPGA_MEM, NULL), base_ptr_sys(base_ptr_sys)
        {
        }

        FPGADeviceMemory::~FPGADeviceMemory(void)
        {
            // this function is invoked before ~FPGADevice
            if (base_ptr_sys != nullptr)
            {
                free(base_ptr_sys);
                base_ptr_sys = nullptr;
            }
        }

        void FPGADeviceMemory::get_bytes(off_t src_offset, void *dst, size_t size)
        {

            FPGACompletionEvent n; // TODO: fix me
            void *src = (void *)((uint8_t *)(base_ptr_sys) + src_offset);
            off_t dst_offset = (uint8_t *)dst - (uint8_t *)(base_ptr_sys);
            get_device()->copy_from_fpga(dst, src, dst_offset, src_offset, size, &n);
            n.request_completed();
        }

        void FPGADeviceMemory::put_bytes(off_t dst_offset, const void *src, size_t size)
        {
            FPGACompletionEvent n; // TODO: fix me
            void *dst = (void *)((uint8_t *)(base_ptr_sys) + dst_offset);
            off_t src_offset = (uint8_t *)src - (uint8_t *)(base_ptr_sys);
            get_device()->copy_to_fpga(dst, src, dst_offset, src_offset, size, &n);
            n.request_completed();
        }

        void *FPGADeviceMemory::get_direct_ptr(off_t offset, size_t size)
        {
            return (void *)((uint8_t *)base_ptr_sys + offset);
        }

        void FPGACompletionEvent::request_completed(void)
        {
            log_fpga.info() << "request_completed " << req;
            req->xd->notify_request_read_done(req);
            req->xd->notify_request_write_done(req);
        }

        FPGAXferDes::FPGAXferDes(uintptr_t _dma_op, Channel *_channel,
                                 NodeID _launch_node, XferDesID _guid,
                                 const std::vector<XferDesPortInfo> &inputs_info,
                                 const std::vector<XferDesPortInfo> &outputs_info,
                                 int _priority, XferDesKind kind)
            : XferDes(_dma_op, _channel, _launch_node, _guid,
                      inputs_info, outputs_info,
                      _priority, 0, 0)
        {
            if ((inputs_info.size() >= 1) &&
                (input_ports[0].mem->kind == MemoryImpl::MKIND_FPGA))
            {
                // all input ports should agree on which fpga they target
                src_fpga = ((FPGADeviceMemory *)(input_ports[0].mem))->device;
                for (size_t i = 1; i < input_ports.size(); i++)
                {
                    // exception: control and indirect ports should be readable from cpu
                    if ((int(i) == input_control.control_port_idx) ||
                        (int(i) == output_control.control_port_idx) ||
                        input_ports[i].is_indirect_port)
                    {
                        assert((input_ports[i].mem->kind == MemoryImpl::MKIND_SYSMEM));
                        continue;
                    }
                    assert(input_ports[i].mem == input_ports[0].mem);
                }
            }
            else
            {
                src_fpga = 0;
            }

            if ((outputs_info.size() >= 1) &&
                (output_ports[0].mem->kind == MemoryImpl::MKIND_FPGA))
            {
                // all output ports should agree on which adev they target
                dst_fpga = ((FPGADeviceMemory *)(output_ports[0].mem))->device;
                for (size_t i = 1; i < output_ports.size(); i++)
                    assert(output_ports[i].mem == output_ports[0].mem);
            }
            else
            {
                dst_fpga = 0;
            }

            // if we're doing a multi-hop copy, we'll dial down the request
            //  sizes to improve pipelining
            bool multihop_copy = false;
            for (size_t i = 1; i < input_ports.size(); i++)
                if (input_ports[i].peer_guid != XFERDES_NO_GUID)
                    multihop_copy = true;
            for (size_t i = 1; i < output_ports.size(); i++)
                if (output_ports[i].peer_guid != XFERDES_NO_GUID)
                    multihop_copy = true;

            log_fpga.info() << "create FPGAXferDes " << kind;
            this->kind = kind;
            switch (kind)
            {
            case XFER_FPGA_TO_DEV:
                if (multihop_copy)
                    max_req_size = 4 << 20;
                break;
            case XFER_FPGA_FROM_DEV:
                if (multihop_copy)
                    max_req_size = 4 << 20;
                break;
            case XFER_FPGA_IN_DEV:
                max_req_size = 1 << 30;
                break;
            case XFER_FPGA_PEER_DEV:
                max_req_size = 256 << 20;
                break;
            case XFER_FPGA_COMP:
                if (multihop_copy)
                    max_req_size = 4 << 20;
                break;
            default:
                break;
            }
            const int max_nr = 10; // TODO:FIXME
            for (int i = 0; i < max_nr; i++)
            {
                FPGARequest *fpga_req = new FPGARequest;
                fpga_req->xd = this;
                fpga_req->event.req = fpga_req;
                available_reqs.push(fpga_req);
            }
        }

        long FPGAXferDes::get_requests(Request **requests, long nr)
        {
            FPGARequest **reqs = (FPGARequest **)requests;
            // no do allow 2D and 3D copies
            // unsigned flags = (TransferIterator::LINES_OK |
            //                   TransferIterator::PLANES_OK);
            unsigned flags = 0;
            long new_nr = default_get_requests(requests, nr, flags);
            for (long i = 0; i < new_nr; i++)
            {
                switch (kind)
                {
                case XFER_FPGA_TO_DEV:
                {
                    reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off, reqs[i]->nbytes);
                    reqs[i]->dst_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off, reqs[i]->nbytes);
                    assert(reqs[i]->src_base != 0);
                    assert(reqs[i]->dst_base != 0);
                    break;
                }
                case XFER_FPGA_FROM_DEV:
                {
                    reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off, reqs[i]->nbytes);
                    reqs[i]->dst_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off, reqs[i]->nbytes);
                    assert(reqs[i]->src_base != 0);
                    assert(reqs[i]->dst_base != 0);
                    break;
                }
                case XFER_FPGA_IN_DEV:
                {
                    reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off, reqs[i]->nbytes);
                    reqs[i]->dst_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off, reqs[i]->nbytes);
                    assert(reqs[i]->src_base != 0);
                    assert(reqs[i]->dst_base != 0);
                    break;
                }
                case XFER_FPGA_PEER_DEV:
                {
                    reqs[i]->dst_fpga = dst_fpga;
                    break;
                }
                case XFER_FPGA_COMP:
                {
                    reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off, reqs[i]->nbytes);
                    reqs[i]->dst_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off, reqs[i]->nbytes);
                    assert(reqs[i]->src_base != 0);
                    assert(reqs[i]->dst_base != 0);
                    break;
                }
                default:
                    assert(0);
                }
            }
            return new_nr;
        }

        bool FPGAXferDes::progress_xd(FPGAChannel *channel, TimeLimit work_until)
        {
            Request *rq;
            bool did_work = false;
            do
            {
                long count = get_requests(&rq, 1);
                if (count > 0)
                {
                    channel->submit(&rq, count);
                    did_work = true;
                }
                else
                    break;
            } while (!work_until.is_expired());
            return did_work;
        }

        void FPGAXferDes::notify_request_read_done(Request *req)
        {
            default_notify_request_read_done(req);
        }

        void FPGAXferDes::notify_request_write_done(Request *req)
        {
            default_notify_request_write_done(req);
        }

        void FPGAXferDes::flush()
        {
        }

        FPGAChannel::FPGAChannel(FPGADevice *_src_fpga, XferDesKind _kind, BackgroundWorkManager *bgwork)
            : SingleXDQChannel<FPGAChannel, FPGAXferDes>(bgwork, _kind, "FPGA channel")
        {
            log_fpga.info() << "FPGAChannel(): " << (int)_kind;
            src_fpga = _src_fpga;

            Memory temp_fpga_mem = src_fpga->fpga_mem->me;
            Memory temp_fpga_ib_mem = src_fpga->fpga_ib->me;
            Memory temp_sys_mem = src_fpga->local_sysmem->me;
            Memory temp_rdma_mem = src_fpga->local_ibmem->me;

            switch (_kind)
            {
            case XFER_FPGA_TO_DEV:
            {
                unsigned bw = 10; // TODO
                unsigned latency = 0;
                unsigned frag_overhead = 0;
                add_path(temp_sys_mem, temp_fpga_mem, bw, latency, frag_overhead, XFER_FPGA_TO_DEV);
                add_path(temp_rdma_mem, temp_fpga_mem, bw, latency, frag_overhead, XFER_FPGA_TO_DEV);
                add_path(temp_sys_mem, temp_fpga_ib_mem, bw, latency, frag_overhead, XFER_FPGA_TO_DEV);
                break;
            }
            case XFER_FPGA_FROM_DEV:
            {
                unsigned bw = 10; // TODO
                unsigned latency = 0;
                unsigned frag_overhead = 0;
                add_path(temp_fpga_mem, temp_sys_mem, bw, latency, frag_overhead, XFER_FPGA_FROM_DEV);
                add_path(temp_fpga_mem, temp_rdma_mem, bw, latency, frag_overhead, XFER_FPGA_FROM_DEV);
                add_path(temp_fpga_ib_mem, temp_sys_mem, bw, latency, frag_overhead, XFER_FPGA_FROM_DEV);
                break;
            }
            case XFER_FPGA_IN_DEV:
            {
                // self-path
                unsigned bw = 10; // TODO
                unsigned latency = 0;
                unsigned frag_overhead = 0;
                add_path(temp_fpga_mem, temp_fpga_mem, bw, latency, frag_overhead, XFER_FPGA_IN_DEV);
                add_path(temp_fpga_ib_mem, temp_fpga_ib_mem, bw, latency, frag_overhead, XFER_FPGA_IN_DEV);
                break;
            }
            case XFER_FPGA_PEER_DEV:
            {
                // just do paths to peers - they'll do the other side
                assert(0 && "not implemented");
                break;
            }
            case XFER_FPGA_COMP:
            {
                unsigned bw = 10; // TODO
                unsigned latency = 1000;
                unsigned frag_overhead = 0;
                add_path(temp_fpga_mem, temp_fpga_mem, bw, latency, frag_overhead, XFER_FPGA_COMP);
                add_path(temp_fpga_ib_mem, temp_fpga_ib_mem, bw, latency, frag_overhead, XFER_FPGA_COMP);
                break;
            }
            default:
                assert(0);
            }
        }

        FPGAChannel::~FPGAChannel()
        {
        }

        XferDes *FPGAChannel::create_xfer_des(uintptr_t dma_op,
                                              NodeID launch_node,
                                              XferDesID guid,
                                              const std::vector<XferDesPortInfo> &inputs_info,
                                              const std::vector<XferDesPortInfo> &outputs_info,
                                              int priority,
                                              XferDesRedopInfo redop_info,
                                              const void *fill_data, size_t fill_size)
        {
            assert(redop_info.id == 0);
            assert(fill_size == 0);
            return new FPGAXferDes(dma_op, this, launch_node, guid,
                                   inputs_info, outputs_info,
                                   priority, kind);
        }

        long FPGAChannel::submit(Request **requests, long nr)
        {
            for (long i = 0; i < nr; i++)
            {
                FPGARequest *req = (FPGARequest *)requests[i];
                // no serdez support
                assert(req->xd->input_ports[req->src_port_idx].serdez_op == 0);
                assert(req->xd->output_ports[req->dst_port_idx].serdez_op == 0);

                // empty transfers don't need to bounce off the ADevice
                if (req->nbytes == 0)
                {
                    req->xd->notify_request_read_done(req);
                    req->xd->notify_request_write_done(req);
                    continue;
                }

                switch (req->dim)
                {
                case Request::DIM_1D:
                {
                    switch (kind)
                    {
                    case XFER_FPGA_TO_DEV:
                        src_fpga->copy_to_fpga(req->dst_base, req->src_base,
                                               req->dst_off, req->src_off,
                                               req->nbytes, &req->event);
                        break;
                    case XFER_FPGA_FROM_DEV:
                        src_fpga->copy_from_fpga(req->dst_base, req->src_base,
                                                 req->dst_off, req->src_off,
                                                 req->nbytes, &req->event);
                        break;
                    case XFER_FPGA_IN_DEV:
                        src_fpga->copy_within_fpga(req->dst_base, req->src_base,
                                                   req->dst_off, req->src_off,
                                                   req->nbytes, &req->event);
                        break;
                    case XFER_FPGA_PEER_DEV:
                        src_fpga->copy_to_peer(req->dst_fpga,
                                               req->dst_base, req->src_base,
                                               req->dst_off, req->src_off,
                                               req->nbytes, &req->event);
                        break;
                    case XFER_FPGA_COMP:
                        src_fpga->comp(req->dst_base, req->src_base,
                                       req->dst_off, req->src_off,
                                       req->nbytes, &req->event);
                        break;
                    default:
                        assert(0);
                    }
                    break;
                }

                case Request::DIM_2D:
                {
                    switch (kind)
                    {
                    case XFER_FPGA_TO_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_FROM_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_IN_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_PEER_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_COMP:
                        assert(0 && "not implemented");
                        break;
                    default:
                        assert(0);
                    }
                    break;
                }

                case Request::DIM_3D:
                {
                    switch (kind)
                    {
                    case XFER_FPGA_TO_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_FROM_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_IN_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_PEER_DEV:
                        assert(0 && "not implemented");
                        break;
                    case XFER_FPGA_COMP:
                        assert(0 && "not implemented");
                        break;
                    default:
                        assert(0);
                    }
                    break;
                }

                default:
                    assert(0);
                }
            }

            return nr;
        }

        FPGAfillXferDes::FPGAfillXferDes(uintptr_t _dma_op, Channel *_channel,
                                         NodeID _launch_node, XferDesID _guid,
                                         const std::vector<XferDesPortInfo> &inputs_info,
                                         const std::vector<XferDesPortInfo> &outputs_info,
                                         int _priority,
                                         const void *_fill_data, size_t _fill_size)
            : XferDes(_dma_op, _channel, _launch_node, _guid,
                      inputs_info, outputs_info,
                      _priority, _fill_data, _fill_size)
        {
            kind = XFER_FPGA_IN_DEV;

            // no direct input data for us
            assert(input_control.control_port_idx == -1);
            input_control.current_io_port = -1;
        }

        long FPGAfillXferDes::get_requests(Request **requests, long nr)
        {
            // unused
            assert(0);
            return 0;
        }

        bool FPGAfillXferDes::progress_xd(FPGAfillChannel *channel,
                                          TimeLimit work_until)
        {
            bool did_work = false;
            ReadSequenceCache rseqcache(this, 2 << 20);
            WriteSequenceCache wseqcache(this, 2 << 20);

            while (true)
            {
                size_t min_xfer_size = 4096; // TODO: make controllable
                size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
                if (max_bytes == 0)
                    break;

                XferPort *out_port = 0;
                size_t out_span_start = 0;
                if (output_control.current_io_port >= 0)
                {
                    out_port = &output_ports[output_control.current_io_port];
                    out_span_start = out_port->local_bytes_total;
                }

                size_t total_bytes = 0;
                if (out_port != 0)
                {
                    // input and output both exist - transfer what we can
                    log_fpga.info() << "memfill chunk: min=" << min_xfer_size
                                    << " max=" << max_bytes;

                    uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));
                    uintptr_t initial_out_offset = out_port->addrcursor.get_offset();
                    while (total_bytes < max_bytes)
                    {
                        AddressListCursor &out_alc = out_port->addrcursor;

                        uintptr_t out_offset = out_alc.get_offset();

                        // the reported dim is reduced for partially consumed address
                        //  ranges - whatever we get can be assumed to be regular
                        int out_dim = out_alc.get_dim();

                        size_t bytes = 0;
                        size_t bytes_left = max_bytes - total_bytes;
                        // memfills don't need to be particularly big to achieve
                        //  peak efficiency, so trim to something that takes
                        //  10's of us to be responsive to the time limit
                        // NOTE: have to be a little careful and make sure the limit
                        //  is a multiple of the fill size - we'll make it a power-of-2
                        const size_t TARGET_CHUNK_SIZE = 256 << 10; // 256KB
                        if (bytes_left > TARGET_CHUNK_SIZE)
                        {
                            size_t max_chunk = fill_size;
                            while (max_chunk < TARGET_CHUNK_SIZE)
                                max_chunk <<= 1;
                            bytes_left = std::min(bytes_left, max_chunk);
                        }

                        if (out_dim > 0)
                        {
                            size_t ocount = out_alc.remaining(0);

                            // contig bytes is always the first dimension
                            size_t contig_bytes = std::min(ocount, bytes_left);

                            // catch simple 1D case first
                            if ((contig_bytes == bytes_left) ||
                                ((contig_bytes == ocount) && (out_dim == 1)))
                            {
                                bytes = contig_bytes;
                                // we only have one element worth of data, so fill
                                //  multiple elements by using a "2d" copy with a
                                //  source stride of 0
                                size_t repeat_count = contig_bytes / fill_size;
#ifdef DEBUG_REALM
                                assert((contig_bytes % fill_size) == 0);
#endif
                                fpga_memcpy_2d(out_base + out_offset, fill_size,
                                               reinterpret_cast<uintptr_t>(fill_data), 0,
                                               fill_size, repeat_count);
                                out_alc.advance(0, bytes);
                            }
                            else
                            {
                                // grow to a 2D fill
                                assert(0 && "FPGA 2D fill not implemented");
                            }
                        }
                        else
                        {
                            // scatter adddress list
                            assert(0);
                        }

#ifdef DEBUG_REALM
                        assert(bytes <= bytes_left);
#endif
                        total_bytes += bytes;

                        // stop if it's been too long, but make sure we do at least the
                        //  minimum number of bytes
                        if ((total_bytes >= min_xfer_size) && work_until.is_expired())
                            break;
                    }
                    for (size_t i = 0; i < 50; i++)
                    {
                        printf("%d ", ((int *)(channel->fpga->fpga_mem->base_ptr_sys))[i]);
                    }
                    printf("\n");

                    log_fpga.info() << "before write buffer, initial_out_offset " << initial_out_offset << " total_bytes " << total_bytes;

                    cl_int err = 0;
                    OCL_CHECK(err, err = channel->fpga->command_queue.enqueueWriteBuffer(channel->fpga->buff,                                                              // buffer on the FPGA
                                                                                         CL_TRUE,                                                                          // blocking call
                                                                                         initial_out_offset,                                                               // buffer offset in bytes
                                                                                         total_bytes,                                                                      // Size in bytes
                                                                                         (void *)((uint64_t)(channel->fpga->fpga_mem->base_ptr_sys) + initial_out_offset), // Pointer to the data to copy
                                                                                         nullptr, nullptr));
                    log_fpga.info() << "after write buffer";
                }
                else
                {
                    // fill with no output, so just count the bytes
                    total_bytes = max_bytes;
                }

                // mem fill is always immediate, so handle both skip and copy with
                //  the same code
                wseqcache.add_span(output_control.current_io_port,
                                   out_span_start, total_bytes);
                out_span_start += total_bytes;

                bool done = record_address_consumption(total_bytes, total_bytes);

                did_work = true;

                if (done || work_until.is_expired())
                    break;
            }

            rseqcache.flush();
            wseqcache.flush();

            return did_work;
        }

        FPGAfillChannel::FPGAfillChannel(FPGADevice *_fpga, BackgroundWorkManager *bgwork)
            : SingleXDQChannel<FPGAfillChannel, FPGAfillXferDes>(bgwork,
                                                                 XFER_GPU_IN_FB,
                                                                 "FPGA fill channel"),
              fpga(_fpga)
        {
            Memory temp_fpga_mem = fpga->fpga_mem->me;

            unsigned bw = 10; // TODO
            unsigned latency = 0;
            unsigned frag_overhead = 0;
            add_path(Memory::NO_MEMORY, temp_fpga_mem,
                     bw, latency, frag_overhead, XFER_FPGA_IN_DEV);

            xdq.add_to_manager(bgwork);
        }

        XferDes *FPGAfillChannel::create_xfer_des(uintptr_t dma_op,
                                                  NodeID launch_node,
                                                  XferDesID guid,
                                                  const std::vector<XferDesPortInfo> &inputs_info,
                                                  const std::vector<XferDesPortInfo> &outputs_info,
                                                  int priority,
                                                  XferDesRedopInfo redop_info,
                                                  const void *fill_data, size_t fill_size)
        {
            assert(redop_info.id == 0);
            return new FPGAfillXferDes(dma_op, this, launch_node, guid,
                                       inputs_info, outputs_info,
                                       priority,
                                       fill_data, fill_size);
        }

        long FPGAfillChannel::submit(Request **requests, long nr)
        {
            // unused
            assert(0);
            return 0;
        }

    }; // namespace FPGA
};     // namespace Realm
