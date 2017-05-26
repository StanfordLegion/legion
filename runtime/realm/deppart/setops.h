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

// set operations for Realm dependent partitioning

#ifndef REALM_DEPPART_SETOPS_H
#define REALM_DEPPART_SETOPS_H

#include "partitions.h"

namespace Realm {

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

};

#endif // REALM_DEPPART_SETOPS_H
