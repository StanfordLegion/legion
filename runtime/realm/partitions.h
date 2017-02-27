/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "indexspace.h"
#include "sparsity.h"
#include "activemsg.h"
#include "id.h"
#include "operation.h"
#include "threads.h"
#include "cmdline.h"
#include "pri_queue.h"
#include "nodeset.h"
#include "interval_tree.h"
#include "dynamic_templates.h"

namespace Realm {

  class PartitioningMicroOp;
  class PartitioningOperation;

  // although partitioning operations eventually generate SparsityMap's, we work with
  //  various intermediates that try to keep things from turning into one big bitmask

  // the CoverageCounter just counts the number of points that get added to it
  // it's not even smart enough to eliminate duplicates
  template <int N, typename T>
  class CoverageCounter {
  public:
    CoverageCounter(void);

    void add_point(const ZPoint<N,T>& p);

    void add_rect(const ZRect<N,T>& r);

    size_t get_count(void) const;

  protected:
    size_t count;
  };

  template <int N, typename T>
  class DenseRectangleList {
  public:
    DenseRectangleList(size_t _max_rects = 0);

    void add_point(const ZPoint<N,T>& p);

    void add_rect(const ZRect<N,T>& r);

    void merge_rects(size_t upper_bound);

    std::vector<ZRect<N,T> > rects;
    size_t max_rects;
  };

  template <int N, typename T>
  class HybridRectangleList {
  public:
    static const size_t HIGH_WATER_MARK = 64;
    static const size_t LOW_WATER_MARK = 16;

    HybridRectangleList(void);

    void add_point(const ZPoint<N,T>& p);

    void add_rect(const ZRect<N,T>& r);

    const std::vector<ZRect<N,T> >& convert_to_vector(void);

    std::vector<ZRect<N,T> > as_vector;
    //std::multimap<T, ZRect<N,T> > as_mmap;
  };

  template <typename T>
  class HybridRectangleList<1,T> : public DenseRectangleList<1,T> {
  public:
    static const size_t HIGH_WATER_MARK = 64;
    static const size_t LOW_WATER_MARK = 16;

    HybridRectangleList(void);

    void add_point(const ZPoint<1,T>& p);

    void add_rect(const ZRect<1,T>& r);

    const std::vector<ZRect<1,T> >& convert_to_vector(void);
    void convert_to_map(void);

    bool is_vector;
    std::map<T, T> as_map;
  };


  /////////////////////////////////////////////////////////////////////////

  template <int N, typename T>
  class SparsityMapImpl : public SparsityMapPublicImpl<N,T> {
  public:
    SparsityMapImpl(SparsityMap<N,T> _me);

    // actual implementation - SparsityMapPublicImpl's version just calls this one
    Event make_valid(bool precise = true);

    static SparsityMapImpl<N,T> *lookup(SparsityMap<N,T> sparsity);

    // methods used in the population of a sparsity map

    // when we plan out a partitioning operation, we'll know how many
    //  different uops are going to contribute something (or nothing) to
    //  the sparsity map - once all of those contributions arrive, we can
    //  finalize the sparsity map
    void set_contributor_count(int count);

    void contribute_nothing(void);
    void contribute_dense_rect_list(const std::vector<ZRect<N,T> >& rects);
    void contribute_raw_rects(const ZRect<N,T>* rects, size_t count, bool last);

    // adds a microop as a waiter for valid sparsity map data - returns true
    //  if the uop is added to the list (i.e. will be getting a callback at some point),
    //  or false if the sparsity map became valid before this call (i.e. no callback)
    bool add_waiter(PartitioningMicroOp *uop, bool precise);

    void remote_data_request(gasnet_node_t requestor, bool send_precise, bool send_approx);
    void remote_data_reply(gasnet_node_t requestor, bool send_precise, bool send_approx);

    SparsityMap<N,T> me;

  protected:
    void finalize(void);
    
    int remaining_contributor_count;
    GASNetHSL mutex;
    std::vector<PartitioningMicroOp *> approx_waiters, precise_waiters;
    bool precise_requested, approx_requested;
    Event precise_ready_event, approx_ready_event;
    NodeSet remote_precise_waiters, remote_approx_waiters;
    NodeSet remote_sharers;
    size_t sizeof_precise;
  };

  // we need a type-erased wrapper to store in the runtime's lookup table
  class SparsityMapImplWrapper {
  public:
    static const ID::ID_Types ID_TYPE = ID::ID_SPARSITY;

    SparsityMapImplWrapper(void);

    void init(ID _me, unsigned _init_owner);

    ID me;
    unsigned owner;
    SparsityMapImplWrapper *next_free;
    DynamicTemplates::TagType type_tag;
    void *map_impl;  // actual implementation

    template <int N, typename T>
    SparsityMapImpl<N,T> *get_or_create(SparsityMap<N,T> me);

    void destroy(void);
  };

  template <int N, typename T>
  class OverlapTester {
  public:
    OverlapTester(void);
    ~OverlapTester(void);

    void add_index_space(int label, const ZIndexSpace<N,T>& space, bool use_approx = true);

    void construct(void);

    void test_overlap(const ZRect<N,T>* rects, size_t count, std::set<int>& overlaps);
    void test_overlap(const ZIndexSpace<N,T>& space, std::set<int>& overlaps, bool approx);
    void test_overlap(const SparsityMapImpl<N,T> *sparsity, std::set<int>& overlaps, bool approx);

  protected:
    std::vector<int> labels;
    std::vector<ZIndexSpace<N,T> > spaces;
    std::vector<bool> approxs;
  };

  template <typename T>
  class OverlapTester<1,T> {
  public:
    OverlapTester(void);
    ~OverlapTester(void);

    void add_index_space(int label, const ZIndexSpace<1,T>& space, bool use_approx = true);

    void construct(void);

    void test_overlap(const ZRect<1,T>* rects, size_t count, std::set<int>& overlaps);
    void test_overlap(const ZIndexSpace<1,T>& space, std::set<int>& overlaps, bool approx);
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
    PartitioningMicroOp(gasnet_node_t _requestor, AsyncMicroOp *_async_microop);

    void finish_dispatch(PartitioningOperation *op, bool inline_ok);

    int wait_count;  // how many sparsity maps are we still waiting for?
    gasnet_node_t requestor;
    AsyncMicroOp *async_microop;
  };

  template <int N, typename T, typename FT>
  class ByFieldMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;
    typedef FT FIELDTYPE;

    static const Opcode OPCODE = UOPCODE_BY_FIELD;

    static DynamicTemplates::TagType type_tag(void);

    ByFieldMicroOp(ZIndexSpace<N,T> _parent_space, ZIndexSpace<N,T> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~ByFieldMicroOp(void);

    void set_value_range(FT _lo, FT _hi);
    void set_value_set(const std::vector<FT>& _value_set);
    void add_sparsity_output(FT _val, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage;
    template <typename S>
    bool serialize_params(S& s) const;

    // construct from received packet
    template <typename S>
    ByFieldMicroOp(gasnet_node_t _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmasks(std::map<FT, BM *>& bitmasks);

    ZIndexSpace<N,T> parent_space, inst_space;
    RegionInstance inst;
    size_t field_offset;
    bool value_range_valid, value_set_valid;
    FT range_lo, range_hi;
    std::set<FT> value_set;
    std::map<FT, SparsityMap<N,T> > sparsity_outputs;
  };

  template <int N, typename T>
  class ComputeOverlapMicroOp : public PartitioningMicroOp {
  public:
    // tied to the ImageOperation * - cannot be moved around the system
    ComputeOverlapMicroOp(PartitioningOperation *_op);
    virtual ~ComputeOverlapMicroOp(void);

    void add_input_space(const ZIndexSpace<N,T>& input_space);
    void add_extra_dependency(const ZIndexSpace<N,T>& dep_space);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    PartitioningOperation *op;
    std::vector<ZIndexSpace<N,T> > input_spaces;
    std::vector<SparsityMapImpl<N,T> *> extra_deps;
  };

  template <int N, typename T, int N2, typename T2>
  class ImageMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;
    static const int DIM2 = N2;
    typedef T2 IDXTYPE2;

    static const Opcode OPCODE = UOPCODE_IMAGE;

    static DynamicTemplates::TagType type_tag(void);

    ImageMicroOp(ZIndexSpace<N,T> _parent_space, ZIndexSpace<N2,T2> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~ImageMicroOp(void);

    void add_sparsity_output(ZIndexSpace<N2,T2> _source, SparsityMap<N,T> _sparsity);
    void add_sparsity_output_with_difference(ZIndexSpace<N2,T2> _source,
                                             ZIndexSpace<N,T> _diff_rhs,
                                             SparsityMap<N,T> _sparsity);
    void add_approx_output(int index, PartitioningOperation *op);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage;
    template <typename S>
    bool serialize_params(S& s) const;

    // construct from received packet
    template <typename S>
    ImageMicroOp(gasnet_node_t _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmasks(std::map<int, BM *>& bitmasks);

    template <typename BM>
    void populate_approx_bitmask(BM& bitmask);

    ZIndexSpace<N,T> parent_space;
    ZIndexSpace<N2,T2> inst_space;
    RegionInstance inst;
    size_t field_offset;
    std::vector<ZIndexSpace<N2,T2> > sources;
    std::vector<ZIndexSpace<N,T> > diff_rhss;
    std::vector<SparsityMap<N,T> > sparsity_outputs;
    int approx_output_index;
    intptr_t approx_output_op;
  };

  template <int N, typename T, int N2, typename T2>
  class PreimageMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;
    static const int DIM2 = N2;
    typedef T2 IDXTYPE2;

    static const Opcode OPCODE = UOPCODE_PREIMAGE;

    static DynamicTemplates::TagType type_tag(void);

    PreimageMicroOp(ZIndexSpace<N,T> _parent_space, ZIndexSpace<N,T> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~PreimageMicroOp(void);

    void add_sparsity_output(ZIndexSpace<N2,T2> _target, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage;
    template <typename S>
    bool serialize_params(S& s) const;

    // construct from received packet
    template <typename S>
    PreimageMicroOp(gasnet_node_t _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmasks(std::map<int, BM *>& bitmasks);

    ZIndexSpace<N,T> parent_space, inst_space;
    RegionInstance inst;
    size_t field_offset;
    std::vector<ZIndexSpace<N2,T2> > targets;
    std::vector<SparsityMap<N,T> > sparsity_outputs;
  };

  template <int N, typename T>
  class UnionMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;

    static const Opcode OPCODE = UOPCODE_UNION;

    static DynamicTemplates::TagType type_tag(void);

    UnionMicroOp(const std::vector<ZIndexSpace<N,T> >& _inputs);
    UnionMicroOp(ZIndexSpace<N,T> _lhs, ZIndexSpace<N,T> _rhs);
    virtual ~UnionMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage;
    template <typename S>
    bool serialize_params(S& s) const;

    // construct from received packet
    template <typename S>
    UnionMicroOp(gasnet_node_t _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmask(BM& bitmask);

    std::vector<ZIndexSpace<N,T> > inputs;
    SparsityMap<N,T> sparsity_output;
  };

  template <int N, typename T>
  class IntersectionMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;

    static const Opcode OPCODE = UOPCODE_INTERSECTION;

    static DynamicTemplates::TagType type_tag(void);

    IntersectionMicroOp(const std::vector<ZIndexSpace<N,T> >& _inputs);
    IntersectionMicroOp(ZIndexSpace<N,T> _lhs, ZIndexSpace<N,T> _rhs);
    virtual ~IntersectionMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage;
    template <typename S>
    bool serialize_params(S& s) const;

    // construct from received packet
    template <typename S>
    IntersectionMicroOp(gasnet_node_t _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmask(BM& bitmask);

    std::vector<ZIndexSpace<N,T> > inputs;
    SparsityMap<N,T> sparsity_output;
  };

  template <int N, typename T>
  class DifferenceMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;

    static const Opcode OPCODE = UOPCODE_DIFFERENCE;

    static DynamicTemplates::TagType type_tag(void);

    DifferenceMicroOp(ZIndexSpace<N,T> _lhs, ZIndexSpace<N,T> _rhs);
    virtual ~DifferenceMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage;
    template <typename S>
    bool serialize_params(S& s) const;

    // construct from received packet
    template <typename S>
    DifferenceMicroOp(gasnet_node_t _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmask(BM& bitmask);

    ZIndexSpace<N,T> lhs, rhs;
    SparsityMap<N,T> sparsity_output;
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
  };


  template <int N, typename T, typename FT>
  class ByFieldOperation : public PartitioningOperation {
  public:
    ByFieldOperation(const ZIndexSpace<N,T>& _parent,
		     const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& _field_data,
		     const ProfilingRequestSet &reqs,
		     Event _finish_event);

    virtual ~ByFieldOperation(void);

    ZIndexSpace<N,T> add_color(FT color);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    ZIndexSpace<N,T> parent;
    std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> > field_data;
    std::vector<FT> colors;
    std::vector<SparsityMap<N,T> > subspaces;
  };

  template <int N, typename T, int N2, typename T2>
  class ImageOperation : public PartitioningOperation {
  public:
    ImageOperation(const ZIndexSpace<N,T>& _parent,
		   const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& _field_data,
		   const ProfilingRequestSet &reqs,
		   Event _finish_event);

    virtual ~ImageOperation(void);

    ZIndexSpace<N,T> add_source(const ZIndexSpace<N2,T2>& source);
    ZIndexSpace<N,T> add_source_with_difference(const ZIndexSpace<N2,T2>& source,
                                                const ZIndexSpace<N,T>& diff_rhs);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

    virtual void set_overlap_tester(void *tester);

  protected:
    ZIndexSpace<N,T> parent;
    std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > > field_data;
    std::vector<ZIndexSpace<N2,T2> > sources;
    std::vector<ZIndexSpace<N,T> > diff_rhss;
    std::vector<SparsityMap<N,T> > images;
  };

  template <int N, typename T, int N2, typename T2>
  class PreimageOperation : public PartitioningOperation {
  public:
    PreimageOperation(const ZIndexSpace<N,T>& _parent,
		      const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > >& _field_data,
		      const ProfilingRequestSet &reqs,
		      Event _finish_event);

    virtual ~PreimageOperation(void);

    ZIndexSpace<N,T> add_target(const ZIndexSpace<N2,T2>& target);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

    virtual void set_overlap_tester(void *tester);

    void provide_sparse_image(int index, const ZRect<N2,T2> *rects, size_t count);

  protected:
    ZIndexSpace<N,T> parent;
    std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > > field_data;
    std::vector<ZIndexSpace<N2,T2> > targets;
    std::vector<SparsityMap<N,T> > preimages;
    GASNetHSL mutex;
    OverlapTester<N2,T2> *overlap_tester;
    std::map<int, std::vector<ZRect<N2,T2> > > pending_sparse_images;
    int remaining_sparse_images;
    std::vector<int> contrib_counts;
    AsyncMicroOp *dummy_overlap_uop;
  };

  template <int N, typename T>
  class UnionOperation : public PartitioningOperation {
  public:
    UnionOperation(const ProfilingRequestSet& reqs,
		   Event _finish_event);

    virtual ~UnionOperation(void);

    ZIndexSpace<N,T> add_union(const ZIndexSpace<N,T>& lhs, const ZIndexSpace<N,T>& rhs);
    ZIndexSpace<N,T> add_union(const std::vector<ZIndexSpace<N,T> >& ops);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    std::vector<std::vector<ZIndexSpace<N,T> > > inputs;
    std::vector<SparsityMap<N,T> > outputs;
  };

  template <int N, typename T>
  class IntersectionOperation : public PartitioningOperation {
  public:
    IntersectionOperation(const ProfilingRequestSet& reqs,
			  Event _finish_event);

    virtual ~IntersectionOperation(void);

    ZIndexSpace<N,T> add_intersection(const ZIndexSpace<N,T>& lhs, const ZIndexSpace<N,T>& rhs);
    ZIndexSpace<N,T> add_intersection(const std::vector<ZIndexSpace<N,T> >& ops);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    std::vector<std::vector<ZIndexSpace<N,T> > > inputs;
    std::vector<SparsityMap<N,T> > outputs;
  };

  template <int N, typename T>
  class DifferenceOperation : public PartitioningOperation {
  public:
    DifferenceOperation(const ProfilingRequestSet& reqs,
			Event _finish_event);

    virtual ~DifferenceOperation(void);

    ZIndexSpace<N,T> add_difference(const ZIndexSpace<N,T>& lhs, const ZIndexSpace<N,T>& rhs);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    std::vector<ZIndexSpace<N,T> > lhss, rhss;
    std::vector<SparsityMap<N,T> > outputs;
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

  class FragmentAssembler {
  public:
    FragmentAssembler(void);
    ~FragmentAssembler(void);

    // returns a sequence ID that may not be unique, but hasn't been used in a 
    //   long time
    int get_sequence_id(void);

    // adds a fragment to the list, returning true if this is the last one from
    //  a sequence
    bool add_fragment(gasnet_node_t sender, int sequence_id, int sequence_count);

  protected:
    int next_sequence_id;
    GASNetHSL mutex; // protects the fragments map
    std::map<gasnet_node_t, std::map<int, int> > fragments;
  };

  struct RemoteMicroOpMessage {
    struct RequestArgs : public BaseMedium {
      gasnet_node_t sender;
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
    static void send_request(gasnet_node_t target, PartitioningOperation *operation,
			     const T& microop);
  };

  struct RemoteMicroOpCompleteMessage {
    struct RequestArgs {
      AsyncMicroOp *async_microop;
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<REMOTE_MICROOP_COMPLETE_MSGID,
                                      RequestArgs,
                                      handle_request> Message;

    static void send_request(gasnet_node_t target, AsyncMicroOp *async_microop);
  };

  struct RemoteSparsityContribMessage {
    struct RequestArgs : public BaseMedium {
      gasnet_node_t sender;
      DynamicTemplates::TagType type_tag;
      ID::IDType sparsity_id;
      int sequence_id;
      int sequence_count;
    };

    struct DecodeHelper {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<REMOTE_SPARSITY_CONTRIB_MSGID,
                                       RequestArgs,
                                       handle_request> Message;

    template <int N, typename T>
    static void send_request(gasnet_node_t target, SparsityMap<N,T> sparsity,
			     int sequence_id, int sequence_count,
			     const ZRect<N,T> *rects, size_t count);
  };

  struct RemoteSparsityRequestMessage {
    struct RequestArgs {
      gasnet_node_t sender;
      DynamicTemplates::TagType type_tag;
      ID::IDType sparsity_id;
      bool send_precise;
      bool send_approx;
    };

    struct DecodeHelper {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args);
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<REMOTE_SPARSITY_REQUEST_MSGID,
                                      RequestArgs,
                                      handle_request> Message;

    template <int N, typename T>
    static void send_request(gasnet_node_t target, SparsityMap<N,T> sparsity,
			     bool send_precise, bool send_approx);
  };

  struct ApproxImageResponseMessage {
    struct RequestArgs : public BaseMedium {
      DynamicTemplates::TagType type_tag;
      intptr_t approx_output_op;
      int approx_output_index;
    };

    struct DecodeHelper {
      template <typename NT, typename T, typename N2T, typename T2>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<APPROX_IMAGE_RESPONSE_MSGID,
                                       RequestArgs,
                                       handle_request> Message;

    template <int N, typename T, int N2, typename T2>
    static void send_request(gasnet_node_t target, intptr_t output_op, int output_index,
			     const ZRect<N2,T2> *rects, size_t count);
  };
    
  struct SetContribCountMessage {
    struct RequestArgs {
      DynamicTemplates::TagType type_tag;
      ID::IDType sparsity_id;
      int count;
    };

    struct DecodeHelper {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args);
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<SET_CONTRIB_COUNT_MSGID,
                                      RequestArgs,
                                      handle_request> Message;

    template <int N, typename T>
    static void send_request(gasnet_node_t target, SparsityMap<N,T> sparsity, int count);
  };
    
};

#endif // REALM_PARTITIONS_H

