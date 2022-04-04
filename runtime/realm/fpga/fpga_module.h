#ifndef REALM_FPGA_H
#define REALM_FPGA_H

#include "realm/module.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/runtime_impl.h"
#include "realm/transfer/channel.h"
#include "realm/circ_queue.h"
#include "realm/transfer/ib_memory.h"

// OpenCL utility layer
#include "xcl2.hpp"

namespace Realm
{
    namespace FPGA
    {
        class FPGAQueue;
        class FPGADevice;
        class FPGADeviceMemcpy;
        class FPGARequest;

        class FPGACompletionNotification
        {
        public:
            virtual ~FPGACompletionNotification(void) {}

            virtual void request_completed(void) = 0;
        };


        class FPGACompletionEvent : public FPGACompletionNotification
        {
        public:
            void request_completed(void);

            FPGARequest *req;
        };

        class FPGAWorker : public BackgroundWorkItem
        {
        public:
            FPGAWorker(void);
            virtual ~FPGAWorker(void);
            // adds a stream that has work to be done
            void add_queue(FPGAQueue *queue);
            // processes work on queues,
            // optionally sleeping for work to show up
            // returns true if work remains to be done
            bool process_queues(bool sleep_on_empty);
            void start_background_thread(Realm::CoreReservationSet &
                                             crs,
                                         size_t stack_size);
            void shutdown_background_thread(void);
            bool do_work(TimeLimit work_until);

        public:
            void thread_main(void);

        protected:
            Mutex lock;
            Mutex::CondVar condvar;
            typedef Realm::CircularQueue<FPGAQueue *, 16> ActiveQueue;
            ActiveQueue active_queues;
            // used by the background thread (if any)
            Realm::CoreReservation *core_rsrv;
            Realm::Thread *worker_thread;
            bool thread_sleeping;
            atomic<bool> worker_shutdown_requested;
        };


        class FPGAWorkFence : public Realm::Operation::AsyncWorkItem
        {
        public:
            FPGAWorkFence(Realm::Operation *op);
            virtual void request_cancellation(void);
            void enqueue(FPGAQueue *queue);
            virtual void print(std::ostream &os) const;
        };

        class FPGAQueue
        {
        public:
            FPGAQueue(FPGADevice *fpga_device, FPGAWorker *fpga_worker, cl::CommandQueue &command_queue);
            ~FPGAQueue(void);
            cl::CommandQueue &get_command_queue() const;
            void add_fence(FPGAWorkFence *fence);
            void add_notification(FPGACompletionNotification *notification);
            bool reap_events(TimeLimit work_until);
            void add_copy(FPGADeviceMemcpy *copy);
            bool issue_copies(TimeLimit work_until);
            void add_event(cl::Event opencl_event,
                           FPGAWorkFence *fence,
                           FPGACompletionNotification *n = 0);

        protected:
            // may only be tested with lock held
            bool has_work(void) const;
            FPGADevice *fpga_device;
            FPGAWorker *fpga_worker;
            cl::CommandQueue &command_queue;
            Mutex mutex;
            struct PendingEvent
            {
                cl::Event opencl_event;
                FPGAWorkFence *fence;
                FPGACompletionNotification *notification;
            };
            std::deque<PendingEvent> pending_events;
            std::deque<FPGADeviceMemcpy *> pending_copies;
        };

        class FPGADeviceMemory;

        class FPGADevice
        {
        public:
            std::string name;
            cl::Device device;
            cl::Buffer buff;
            cl::Buffer ib_buff;
            cl::Context context;
            cl::CommandQueue command_queue;
            cl::Program program;
            FPGADevice(cl::Device &device, std::string name, std::string xclbin, FPGAWorker *fpga_worker, size_t fpga_coprocessor_num_cu, std::string fpga_coprocessor_kernel);
            ~FPGADevice();
            void create_fpga_mem(RuntimeImpl *runtime, size_t size);
            void create_fpga_ib(RuntimeImpl *runtime, size_t size);
            void create_dma_channels(RuntimeImpl *runtime);
            void create_fpga_queues();
            void copy_to_fpga(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification);
            void copy_from_fpga(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification);
            void copy_within_fpga(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification);
            void copy_to_peer(FPGADevice *dst_dev, void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification);
            void comp(void *dst, const void *src, off_t dst_offset, off_t src_offset, size_t bytes, FPGACompletionNotification *notification);
            bool is_in_buff(void *ptr);
            bool is_in_ib_buff(void *ptr);
            FPGADeviceMemory *fpga_mem;
            IBMemory *fpga_ib;
            MemoryImpl *local_sysmem;
            IBMemory *local_ibmem;
            FPGAWorker *fpga_worker;
            FPGAQueue *fpga_queue;
            size_t fpga_coprocessor_num_cu;
            std::string fpga_coprocessor_kernel;
        };

        enum FPGAMemcpyKind
        {
            FPGA_MEMCPY_HOST_TO_DEVICE,
            FPGA_MEMCPY_DEVICE_TO_HOST,
            FPGA_MEMCPY_DEVICE_TO_DEVICE,
            FPGA_MEMCPY_PEER_TO_PEER,
        };

        // An abstract base class for all FPGA memcpy operations
        class FPGADeviceMemcpy
        {
        public:
            FPGADeviceMemcpy(FPGADevice *fpga_device, FPGAMemcpyKind kind, FPGACompletionNotification *notification);
            virtual ~FPGADeviceMemcpy(void) {}

        public:
            virtual void execute(FPGAQueue *queue) = 0;

        public:
            FPGADevice *const fpga_device;

        protected:
            FPGAMemcpyKind kind;
            FPGACompletionNotification *notification;
        };

        class FPGADeviceMemcpy1D : public FPGADeviceMemcpy
        {
        public:
            FPGADeviceMemcpy1D(FPGADevice *fpga_device,
                               void *dst,
                               const void *src,
                               size_t bytes,
                               off_t buff_offset,
                               FPGAMemcpyKind kind,
                               FPGACompletionNotification *notification);

            virtual ~FPGADeviceMemcpy1D(void);

        public:
            virtual void execute(FPGAQueue *q);

        protected:
            void *dst;
            const void *src;
            size_t elmt_size;
            off_t buff_offset;

        private:
            void do_span(off_t pos, size_t len);
            FPGAQueue *fpga_queue;
        };

        class FPGAProcessor : public LocalTaskProcessor
        {
        public:
            FPGAProcessor(FPGADevice *fpga_device, Processor me, Realm::CoreReservationSet &crs);
            virtual ~FPGAProcessor(void);
            static FPGAProcessor *get_current_fpga_proc(void);
            FPGADevice *fpga_device;

        protected:
            Realm::CoreReservation *core_rsrv_;
        };

        class FPGADeviceMemory : public LocalManagedMemory
        {
        public:
            FPGADeviceMemory(Memory memory, FPGADevice *device, void *base_ptr_sys, size_t size);
            virtual ~FPGADeviceMemory(void);
            virtual void get_bytes(off_t offset, void *dst, size_t size);
            virtual void put_bytes(off_t offset, const void *src, size_t size);
            virtual void *get_direct_ptr(off_t offset, size_t size);

            FPGADevice *get_device() const { return device; };
            void *get_mem_base_sys() const { return base_ptr_sys; };
            FPGADevice *device;
            void *base_ptr_sys;
        };

        class FPGARequest : public Request
        {
        public:
            const void *src_base;
            void *dst_base;
            FPGADevice *dst_fpga;
            FPGACompletionEvent event;
        };

        class FPGAChannel;

        class FPGAXferDes : public XferDes
        {
        public:
            FPGAXferDes(uintptr_t _dma_op, Channel *_channel,
                        NodeID _launch_node, XferDesID _guid,
                        const std::vector<XferDesPortInfo> &inputs_info,
                        const std::vector<XferDesPortInfo> &outputs_info,
                        int _priority, XferDesKind kind);

            ~FPGAXferDes()
            {
                while (!available_reqs.empty())
                {
                    FPGARequest *fpga_req = (FPGARequest *)available_reqs.front();
                    available_reqs.pop();
                    delete fpga_req;
                }
            }

            long default_get_requests_tentative(Request **requests, long nr, unsigned flags);
            long get_requests(Request **requests, long nr);
            void notify_request_read_done(Request *req);
            void notify_request_write_done(Request *req);
            void flush();

            bool progress_xd(FPGAChannel *channel, TimeLimit work_until);

        private:
            FPGADevice *src_fpga;
            FPGADevice *dst_fpga;
        };

        class FPGAChannel : public SingleXDQChannel<FPGAChannel, FPGAXferDes>
        {
        public:
            FPGAChannel(FPGADevice *_src_fpga, XferDesKind _kind,
                        BackgroundWorkManager *bgwork);
            ~FPGAChannel();

            // TODO: multiple concurrent copies not ok for now
            static const bool is_ordered = true;

            virtual XferDes *create_xfer_des(uintptr_t dma_op,
                                             NodeID launch_node,
                                             XferDesID guid,
                                             const std::vector<XferDesPortInfo> &inputs_info,
                                             const std::vector<XferDesPortInfo> &outputs_info,
                                             int priority,
                                             XferDesRedopInfo redop_info,
                                             const void *fill_data, size_t fill_size);

            long submit(Request **requests, long nr);

        private:
            FPGADevice *src_fpga;
        };

        class FPGAfillChannel;

        class FPGAfillXferDes : public XferDes
        {
        public:
            FPGAfillXferDes(uintptr_t _dma_op, Channel *_channel,
                            NodeID _launch_node, XferDesID _guid,
                            const std::vector<XferDesPortInfo> &inputs_info,
                            const std::vector<XferDesPortInfo> &outputs_info,
                            int _priority,
                            const void *_fill_data, size_t _fill_size);

            long get_requests(Request **requests, long nr);

            bool progress_xd(FPGAfillChannel *channel, TimeLimit work_until);

        protected:
            size_t reduced_fill_size;
        };

        class FPGAfillChannel : public SingleXDQChannel<FPGAfillChannel, FPGAfillXferDes>
        {
        public:
            FPGAfillChannel(FPGADevice *_fpga, BackgroundWorkManager *bgwork);

            // TODO: multiple concurrent fills not ok for now
            static const bool is_ordered = true;

            virtual XferDes *create_xfer_des(uintptr_t dma_op,
                                             NodeID launch_node,
                                             XferDesID guid,
                                             const std::vector<XferDesPortInfo> &inputs_info,
                                             const std::vector<XferDesPortInfo> &outputs_info,
                                             int priority,
                                             XferDesRedopInfo redop_info,
                                             const void *fill_data, size_t fill_size);

            long submit(Request **requests, long nr);

        protected:
            friend class FPGAfillXferDes;
            FPGADevice *fpga;
        };

        class FPGAModule : public Module
        {
        protected:
            FPGAModule(void);

        public:
            virtual ~FPGAModule(void);

            static Module *create_module(RuntimeImpl *runtime, std::vector<std::string> &cmdline);

            // do any general initialization - this is called after all configuration is
            //  complete
            virtual void initialize(RuntimeImpl *runtime);

            // create any memories provided by this module (default == do nothing)
            //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
            virtual void create_memories(RuntimeImpl *runtime);

            // create any processors provided by the module (default == do nothing)
            //  (each new ProcessorImpl should use a Processor from
            //   RuntimeImpl::next_local_processor_id)
            virtual void create_processors(RuntimeImpl *runtime);

            // create any DMA channels provided by the module (default == do nothing)
            virtual void create_dma_channels(RuntimeImpl *runtime);

            // create any code translators provided by the module (default == do nothing)
            virtual void create_code_translators(RuntimeImpl *runtime);

            // clean up any common resources created by the module - this will be called
            //  after all memories/processors/etc. have been shut down and destroyed
            virtual void cleanup(void);

        public:
            size_t cfg_num_fpgas;
            bool cfg_use_worker_threads;
            bool cfg_use_shared_worker;
            size_t cfg_fpga_mem_size;
            size_t cfg_fpga_ib_size;
            FPGAWorker *shared_worker;
            std::map<FPGADevice *, FPGAWorker *> dedicated_workers;
            std::string cfg_fpga_xclbin;
            std::vector<FPGADevice *> fpga_devices;
            size_t cfg_fpga_coprocessor_num_cu;
            std::string cfg_fpga_coprocessor_kernel;

        protected:
            std::vector<FPGAProcessor *> fpga_procs_;
        };

    }; // namespace FPGA
};     // namespace Realm

#endif
