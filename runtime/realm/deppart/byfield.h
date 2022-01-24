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

// byfield (filter) operations for Realm dependent partitioning

#ifndef REALM_DEPPART_BYFIELD_H
#define REALM_DEPPART_BYFIELD_H

#include "realm/deppart/partitions.h"

namespace Realm {

  template <int N, typename T, typename FT>
  class ByFieldMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;
    typedef FT FIELDTYPE;

    ByFieldMicroOp(IndexSpace<N,T> _parent_space, IndexSpace<N,T> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~ByFieldMicroOp(void);

    void set_value_range(FT _lo, FT _hi);
    void set_value_set(const std::vector<FT>& _value_set);
    void add_sparsity_output(FT _val, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage<ByFieldMicroOp<N,T,FT> >;
    static ActiveMessageHandlerReg<RemoteMicroOpMessage<ByFieldMicroOp<N,T,FT> > > areg;

    friend class PartitioningMicroOp;
    template <typename S>
    REALM_ATTR_WARN_UNUSED(bool serialize_params(S& s) const);

    // construct from received packet
    template <typename S>
    ByFieldMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmasks(std::map<FT, BM *>& bitmasks);

    IndexSpace<N,T> parent_space, inst_space;
    RegionInstance inst;
    size_t field_offset;
    bool value_range_valid, value_set_valid;
    FT range_lo, range_hi;
    std::set<FT> value_set;
    std::map<FT, SparsityMap<N,T> > sparsity_outputs;
  };

  template <int N, typename T, typename FT>
  class ByFieldOperation : public PartitioningOperation {
  public:
    ByFieldOperation(const IndexSpace<N,T>& _parent,
		     const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& _field_data,
		     const ProfilingRequestSet &reqs,
		     GenEventImpl *_finish_event, EventImpl::gen_t _finish_gen);

    virtual ~ByFieldOperation(void);

    IndexSpace<N,T> add_color(FT color);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

  protected:
    IndexSpace<N,T> parent;
    std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> > field_data;
    std::vector<FT> colors;
    std::vector<SparsityMap<N,T> > subspaces;
  };
    
};

#endif // REALM_DEPPART_BYFIELD_H
