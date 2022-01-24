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

// index space partitioning for Realm

#ifndef REALM_PARTITIONS_H
#define REALM_PARTITIONS_H

#include "realm/indexspace.h"
#include "realm/sparsity.h"
#include "realm/activemsg.h"
#include "realm/id.h"
#include "realm/operation.h"
#include "realm/threads.h"
#include "realm/cmdline.h"
#include "realm/pri_queue.h"
#include "realm/nodeset.h"
#include "realm/interval_tree.h"
#include "realm/dynamic_templates.h"
#include "realm/deppart/sparsity_impl.h"
#include "realm/deppart/inst_helper.h"
#include "realm/bgwork.h"

namespace Realm {

  class PartitioningMicroOp;
  class PartitioningOperation;


  template <int N, typename T>
  class OverlapTester {
  public:
    OverlapTester(void);
    ~OverlapTester(void);

    void add_index_space(int label, const IndexSpace<N,T>& space, bool use_approx = true);

    void construct(void);

    void test_overlap(const Rect<N,T>* rects, size_t count, std::set<int>& overlaps);
    void test_overlap(const IndexSpace<N,T>& space, std::set<int>& overlaps, bool approx);
    void test_overlap(const SparsityMapImpl<N,T> *sparsity, std::set<int>& overlaps, bool approx);

  protected:
    std::vector<int> labels;
    std::vector<IndexSpace<N,T> > spaces;
    std::vector<bool> approxs;
  };

  template <typename T>
  class OverlapTester<1,T> {
  public:
    OverlapTester(void);
    ~OverlapTester(void);

    void add_index_space(int label, const IndexSpace<1,T>& space, bool use_approx = true);

    void construct(void);

    void test_overlap(const Rect<1,T>* rects, size_t count, std::set<int>& overlaps);
    void test_overlap(const IndexSpace<1,T>& space, std::set<int>& overlaps, bool approx);
    void test_overlap(const SparsityMapImpl<1,T> *sparsity, std::set<int>& overlaps, bool approx);

  protected:
    IntervalTree<T,int> interval_tree;
  };


  /////////////////////////////////////////////////////////////////////////

  class AsyncMicroOp : public Operation::AsyncWorkItem {
  public:
    AsyncMicroOp(Operation *_op, PartitioningMicroOp *_uop);
    ~AsyncMicroOp();
    
    virtual void request_cancellation(void);
      
    virtual void print(std::ostream& os) const;

  protected:
    PartitioningMicroOp *uop;
  };

  class PartitioningMicroOp {
  public:
    PartitioningMicroOp(void);
    virtual ~PartitioningMicroOp(void);

    virtual void execute(void) = 0;

    void mark_started(void);
    void mark_finished(void);

    template <int N, typename T>
    void sparsity_map_ready(SparsityMapImpl<N,T> *sparsity, bool precise);

    IntrusiveListLink<PartitioningMicroOp> uop_link;
    REALM_PMTA_DEFN(PartitioningMicroOp,IntrusiveListLink<PartitioningMicroOp>,uop_link);
    typedef IntrusiveList<PartitioningMicroOp, REALM_PMTA_USE(PartitioningMicroOp,uop_link), DummyLock> MicroOpList;

  protected:
    PartitioningMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop);

    void finish_dispatch(PartitioningOperation *op, bool inline_ok);

    atomic<int> wait_count;  // how many sparsity maps are we still waiting for?
    NodeID requestor;
    AsyncMicroOp *async_microop;

    // helper code to ship a microop to another node
    template <typename T>
    static void forward_microop(NodeID target,
				PartitioningOperation *op, T *microop);
  };

  template <int N, typename T>
  class ComputeOverlapMicroOp : public PartitioningMicroOp {
  public:
    // tied to the ImageOperation * - cannot be moved around the system
    ComputeOverlapMicroOp(PartitioningOperation *_op);
    virtual ~ComputeOverlapMicroOp(void);

    void add_input_space(const IndexSpace<N,T>& input_space);
    void add_extra_dependency(const IndexSpace<N,T>& dep_space);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    PartitioningOperation *op;
    std::vector<IndexSpace<N,T> > input_spaces;
    std::vector<SparsityMapImpl<N,T> *> extra_deps;
  };

  ////////////////////////////////////////
  //
  
  class PartitioningOperation : public Operation {
  public:
    PartitioningOperation(const ProfilingRequestSet &reqs,
			  GenEventImpl *_finish_event,
			  EventImpl::gen_t _finish_gen);

    virtual void execute(void) = 0;

    // the type of 'tester' depends on which operation it is, so erase the type here...
    virtual void set_overlap_tester(void *tester);

    void launch(Event wait_for);

    // some partitioning operations are handled inline for simple cases
    // these cases must still supply all the requested profiling responses
    static void do_inline_profiling(const ProfilingRequestSet &reqs,
				    long long inline_start_time);

    IntrusiveListLink<PartitioningOperation> op_link;
    REALM_PMTA_DEFN(PartitioningOperation,IntrusiveListLink<PartitioningOperation>,op_link);
    typedef IntrusiveList<PartitioningOperation, REALM_PMTA_USE(PartitioningOperation,op_link), DummyLock> OpList;

    class DeferredLaunch : public EventWaiter {
    public:
      void defer(PartitioningOperation *_op, Event wait_on);

      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      PartitioningOperation *op;
    };
    DeferredLaunch deferred_launch;
  };


  ////////////////////////////////////////
  //

  class PartitioningOpQueue : public BackgroundWorkItem {
  public:
    PartitioningOpQueue(CoreReservation *_rsrv,
			BackgroundWorkManager *_bgwork);
    virtual ~PartitioningOpQueue(void);

    static void configure_from_cmdline(std::vector<std::string>& cmdline);
    static void start_worker_threads(CoreReservationSet& crs,
				     BackgroundWorkManager *_bgwork);
    static void stop_worker_threads(void);

    void enqueue_partitioning_operation(PartitioningOperation *op);
    void enqueue_partitioning_microop(PartitioningMicroOp *uop);

    void worker_thread_loop(void);

    // called by BackgroundWorkers
    bool do_work(TimeLimit work_until);

  protected:
    atomic<bool> shutdown_flag;
    CoreReservation *rsrv;
    PartitioningOperation::OpList op_list;
    PartitioningMicroOp::MicroOpList uop_list;
    Mutex mutex;
    Mutex::CondVar condvar;
    std::vector<Thread *> workers;
    bool work_advertised;
  };


  ////////////////////////////////////////
  //
  // active messages


  template <typename T>
  struct RemoteMicroOpMessage {
    PartitioningOperation *operation;
    AsyncMicroOp *async_microop;

    static void handle_message(NodeID sender,
			       const RemoteMicroOpMessage<T> &msg,
			       const void *data, size_t datalen)
    {
      Serialization::FixedBufferDeserializer fbd(data, datalen);
      T *uop = new T(sender, msg.async_microop, fbd);
      uop->dispatch(msg.operation, false /*not ok to run in this thread*/);
    }
  };


  struct RemoteMicroOpCompleteMessage {
    AsyncMicroOp *async_microop;

    static void handle_message(NodeID sender,
			       const RemoteMicroOpCompleteMessage &msg,
			       const void *data, size_t datalen);

    static ActiveMessageHandlerReg<RemoteMicroOpCompleteMessage> areg;
  };


};

#include "realm/deppart/partitions.inl"

#endif // REALM_PARTITIONS_H

