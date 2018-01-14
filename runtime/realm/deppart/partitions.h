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

    enum Opcode {
      UOPCODE_INVALID,
      UOPCODE_BY_FIELD,
      UOPCODE_IMAGE,
      UOPCODE_PREIMAGE,
      UOPCODE_UNION,
      UOPCODE_INTERSECTION,
      UOPCODE_DIFFERENCE,
    };

  protected:
    PartitioningMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop);

    void finish_dispatch(PartitioningOperation *op, bool inline_ok);

    int wait_count;  // how many sparsity maps are we still waiting for?
    NodeID requestor;
    AsyncMicroOp *async_microop;
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
			  Event _finish_event);

    virtual void execute(void) = 0;

    // the type of 'tester' depends on which operation it is, so erase the type here...
    virtual void set_overlap_tester(void *tester);

    void deferred_launch(Event wait_for);

    // some partitioning operations are handled inline for simple cases
    // these cases must still supply all the requested profiling responses
    static void do_inline_profiling(const ProfilingRequestSet &reqs,
				    long long inline_start_time);
  };


  ////////////////////////////////////////
  //

  class PartitioningOpQueue {
  public:
    PartitioningOpQueue(CoreReservation *_rsrv);
    ~PartitioningOpQueue(void);

    static void configure_from_cmdline(std::vector<std::string>& cmdline);
    static void start_worker_threads(CoreReservationSet& crs);
    static void stop_worker_threads(void);

    enum {
      OPERATION_PRIORITY = 1,
      MICROOP_PRIORITY = 0
    };

    void enqueue_partitioning_operation(PartitioningOperation *op);
    void enqueue_partitioning_microop(PartitioningMicroOp *uop);

    void worker_thread_loop(void);

  protected:
    bool shutdown_flag;
    CoreReservation *rsrv;
    PriorityQueue<void *, DummyLock> queued_ops;
    GASNetHSL mutex;
    GASNetCondVar condvar;
    std::vector<Thread *> workers;
  };


  ////////////////////////////////////////
  //
  // active messages


  struct RemoteMicroOpMessage {
    struct RequestArgs : public BaseMedium {
      NodeID sender;
      DynamicTemplates::TagType type_tag;
      PartitioningMicroOp::Opcode opcode;
      PartitioningOperation *operation;
      AsyncMicroOp *async_microop;
    };

    // each different type of microop needs a different demux helper because they
    //  use differing numbers of template arguments
    struct ByFieldDecoder {
      template <typename NT, typename T, typename FT>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };
    struct ImageDecoder {
      template <typename NT, typename T, typename N2T, typename T2>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };
    struct PreimageDecoder {
      template <typename NT, typename T, typename N2T, typename T2>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };
    struct UnionDecoder {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };
    struct IntersectionDecoder {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };
    struct DifferenceDecoder {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<REMOTE_MICROOP_MSGID,
                                       RequestArgs,
                                       handle_request> Message;

    template <typename T>
    static void send_request(NodeID target, PartitioningOperation *operation,
			     const T& microop);
  };

  template <typename T>
  /*static*/ void RemoteMicroOpMessage::send_request(NodeID target, 
						     PartitioningOperation *operation,
						     const T& microop)
  {
    RequestArgs args;

    args.sender = my_node_id;
    args.type_tag = T::type_tag();
    args.opcode = T::OPCODE;
    args.operation = operation;
    args.async_microop = microop.async_microop;

    Serialization::DynamicBufferSerializer dbs(256);
    microop.serialize_params(dbs);
    ByteArray b = dbs.detach_bytearray();

    Message::request(target, args, b.base(), b.size(), PAYLOAD_FREE);
    b.detach();
  }

  struct RemoteMicroOpCompleteMessage {
    struct RequestArgs {
      AsyncMicroOp *async_microop;
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<REMOTE_MICROOP_COMPLETE_MSGID,
                                      RequestArgs,
                                      handle_request> Message;

    static void send_request(NodeID target, AsyncMicroOp *async_microop);
  };


};

#endif // REALM_PARTITIONS_H

