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

// NOTE: all these interfaces are templated, which means partitions.cc is going
//  to have to somehow know which ones to instantiate - we'll try to have a 
//  Makefile-based way to control this, but right now it's hardcoded at the
//  bottom of partitions.cc, so go there if you get link errors

namespace Realm {

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
    DenseRectangleList(void);

    void add_point(const ZPoint<N,T>& p);

    void add_rect(const ZRect<N,T>& r);

    std::vector<ZRect<N,T> > rects;
  };

  /////////////////////////////////////////////////////////////////////////

  template <int N, typename T>
  class SparsityMapImpl : public SparsityMapPublicImpl<N,T> {
  public:
    SparsityMapImpl(void);

    // actual implementation - SparsityMapPublicImpl's version just calls this one
    Event make_valid(bool precise = true);

    static SparsityMapImpl<N,T> *lookup(SparsityMap<N,T> sparsity);

    // methods used in the population of a sparsity map

    // when we plan out a partitioning operation, we'll know how many
    //  different uops are going to contribute something (or nothing) to
    //  the sparsity map - once all of those contributions arrive, we can
    //  finalize the sparsity map
    void update_contributor_count(int delta = 1);

    void contribute_nothing(void);
    void contribute_dense_rect_list(const DenseRectangleList<N,T>& rects);

  protected:
    void finalize(void);
    
    int remaining_contributor_count;
    GASNetHSL mutex;
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
    int dim;
    int idxtype; // captured via sizeof(T) right now
    void *map_impl;  // actual implementation

    template <int N, typename T>
    SparsityMapImpl<N,T> *get_or_create(void);

    void destroy(void);
  };


  /////////////////////////////////////////////////////////////////////////

  class PartitioningMicroOp {
  public:
    virtual ~PartitioningMicroOp(void);

    virtual void execute(void) = 0;
  };

  template <int N, typename T, typename FT>
  class ByFieldMicroOp : public PartitioningMicroOp {
  public:
    ByFieldMicroOp(ZIndexSpace<N,T> _parent_space, ZIndexSpace<N,T> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~ByFieldMicroOp(void);

    void set_value_range(FT _lo, FT _hi);
    void set_value_set(const std::vector<FT>& _value_set);
    void add_sparsity_output(FT _val, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

  protected:
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

  template <int N, typename T, int N2, typename T2>
  class ImageMicroOp : public PartitioningMicroOp {
  public:
    ImageMicroOp(ZIndexSpace<N,T> _parent_space, ZIndexSpace<N2,T2> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~ImageMicroOp(void);

    void add_sparsity_output(ZIndexSpace<N2,T2> _source, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(Operation *op);

  protected:
    template <typename BM>
    void populate_bitmasks(std::map<int, BM *>& bitmasks);

    ZIndexSpace<N,T> parent_space;
    ZIndexSpace<N2,T2> inst_space;
    RegionInstance inst;
    size_t field_offset;
    std::vector<ZIndexSpace<N2,T2> > sources;
    std::vector<SparsityMap<N,T> > sparsity_outputs;
  };

  template <int N, typename T, int N2, typename T2>
  class PreimageMicroOp : public PartitioningMicroOp {
  public:
    PreimageMicroOp(ZIndexSpace<N,T> _parent_space, ZIndexSpace<N,T> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~PreimageMicroOp(void);

    void add_sparsity_output(ZIndexSpace<N2,T2> _target, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

  protected:
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
    UnionMicroOp(const std::vector<ZIndexSpace<N,T> >& _inputs);
    UnionMicroOp(ZIndexSpace<N,T> _lhs, ZIndexSpace<N,T> _rhs);
    virtual ~UnionMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

  protected:
    template <typename BM>
    void populate_bitmask(BM& bitmask);

    std::vector<ZIndexSpace<N,T> > inputs;
    SparsityMap<N,T> sparsity_output;
  };

  template <int N, typename T>
  class IntersectionMicroOp : public PartitioningMicroOp {
  public:
    IntersectionMicroOp(const std::vector<ZIndexSpace<N,T> >& _inputs);
    IntersectionMicroOp(ZIndexSpace<N,T> _lhs, ZIndexSpace<N,T> _rhs);
    virtual ~IntersectionMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

  protected:
    template <typename BM>
    void populate_bitmask(BM& bitmask);

    std::vector<ZIndexSpace<N,T> > inputs;
    SparsityMap<N,T> sparsity_output;
  };

  template <int N, typename T>
  class DifferenceMicroOp : public PartitioningMicroOp {
  public:
    DifferenceMicroOp(ZIndexSpace<N,T> _lhs, ZIndexSpace<N,T> _rhs);
    virtual ~DifferenceMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

  protected:
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

    void deferred_launch(Event wait_for);
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

    virtual void execute(void);

  protected:
    ZIndexSpace<N,T> parent;
    std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > > field_data;
    std::vector<ZIndexSpace<N2,T2> > sources;
    std::vector<SparsityMap<N,T> > images;
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


};

#endif // REALM_PARTITIONS_H

