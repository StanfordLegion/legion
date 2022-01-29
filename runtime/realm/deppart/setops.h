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

// set operations for Realm dependent partitioning

#ifndef REALM_DEPPART_SETOPS_H
#define REALM_DEPPART_SETOPS_H

#include "realm/deppart/partitions.h"

namespace Realm {

  template <int N, typename T>
  class UnionMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;

    UnionMicroOp(const std::vector<IndexSpace<N,T> >& _inputs);
    UnionMicroOp(IndexSpace<N,T> _lhs, IndexSpace<N,T> _rhs);
    virtual ~UnionMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage<UnionMicroOp<N,T> >;
    static ActiveMessageHandlerReg<RemoteMicroOpMessage<UnionMicroOp<N,T> > > areg;

    friend class PartitioningMicroOp;
    template <typename S>
    REALM_ATTR_WARN_UNUSED(bool serialize_params(S& s) const);

    // construct from received packet
    template <typename S>
    UnionMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmask(BM& bitmask);

    std::vector<IndexSpace<N,T> > inputs;
    SparsityMap<N,T> sparsity_output;
  };

  template <int N, typename T>
  class IntersectionMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;

    IntersectionMicroOp(const std::vector<IndexSpace<N,T> >& _inputs);
    IntersectionMicroOp(IndexSpace<N,T> _lhs, IndexSpace<N,T> _rhs);
    virtual ~IntersectionMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage<IntersectionMicroOp<N,T> >;
    static ActiveMessageHandlerReg<RemoteMicroOpMessage<IntersectionMicroOp<N,T> > > areg;

    friend class PartitioningMicroOp;
    template <typename S>
    REALM_ATTR_WARN_UNUSED(bool serialize_params(S& s) const);

    // construct from received packet
    template <typename S>
    IntersectionMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmask(BM& bitmask);

    std::vector<IndexSpace<N,T> > inputs;
    SparsityMap<N,T> sparsity_output;
  };

  template <int N, typename T>
  class DifferenceMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;

    DifferenceMicroOp(IndexSpace<N,T> _lhs, IndexSpace<N,T> _rhs);
    virtual ~DifferenceMicroOp(void);

    void add_sparsity_output(SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage<DifferenceMicroOp<N,T> >;
    static ActiveMessageHandlerReg<RemoteMicroOpMessage<DifferenceMicroOp<N,T> > > areg;

    friend class PartitioningMicroOp;
    template <typename S>
    REALM_ATTR_WARN_UNUSED(bool serialize_params(S& s) const);

    // construct from received packet
    template <typename S>
    DifferenceMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmask(BM& bitmask);

    IndexSpace<N,T> lhs, rhs;
    SparsityMap<N,T> sparsity_output;
  };

  template <int N, typename T>
  class UnionOperation : public PartitioningOperation {
  public:
    UnionOperation(const ProfilingRequestSet& reqs,
		   GenEventImpl *_finish_event, EventImpl::gen_t _finish_gen);

    virtual ~UnionOperation(void);

    IndexSpace<N,T> add_union(const IndexSpace<N,T>& lhs, const IndexSpace<N,T>& rhs);
    IndexSpace<N,T> add_union(const std::vector<IndexSpace<N,T> >& ops);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    std::vector<std::vector<IndexSpace<N,T> > > inputs;
    std::vector<SparsityMap<N,T> > outputs;
  };

  template <int N, typename T>
  class IntersectionOperation : public PartitioningOperation {
  public:
    IntersectionOperation(const ProfilingRequestSet& reqs,
			  GenEventImpl *_finish_event, EventImpl::gen_t _finish_gen);

    virtual ~IntersectionOperation(void);

    IndexSpace<N,T> add_intersection(const IndexSpace<N,T>& lhs, const IndexSpace<N,T>& rhs);
    IndexSpace<N,T> add_intersection(const std::vector<IndexSpace<N,T> >& ops);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    std::vector<std::vector<IndexSpace<N,T> > > inputs;
    std::vector<SparsityMap<N,T> > outputs;
  };

  template <int N, typename T>
  class DifferenceOperation : public PartitioningOperation {
  public:
    DifferenceOperation(const ProfilingRequestSet& reqs,
			GenEventImpl *_finish_event, EventImpl::gen_t _finish_gen);

    virtual ~DifferenceOperation(void);

    IndexSpace<N,T> add_difference(const IndexSpace<N,T>& lhs, const IndexSpace<N,T>& rhs);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    std::vector<IndexSpace<N,T> > lhss, rhss;
    std::vector<SparsityMap<N,T> > outputs;
  };

};

#endif // REALM_DEPPART_SETOPS_H
