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

// image operations for Realm dependent partitioning

#ifndef REALM_DEPPART_IMAGE_H
#define REALM_DEPPART_IMAGE_H

#include "partitions.h"

namespace Realm {

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
		 RegionInstance _inst, size_t _field_offset, bool _is_ranged);
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
    void populate_bitmasks_ptrs(std::map<int, BM *>& bitmasks);

    template <typename BM>
    void populate_bitmasks_ranges(std::map<int, BM *>& bitmasks);

    template <typename BM>
    void populate_approx_bitmask_ptrs(BM& bitmask);

    template <typename BM>
    void populate_approx_bitmask_ranges(BM& bitmask);

    ZIndexSpace<N,T> parent_space;
    ZIndexSpace<N2,T2> inst_space;
    RegionInstance inst;
    size_t field_offset;
    bool is_ranged;
    std::vector<ZIndexSpace<N2,T2> > sources;
    std::vector<ZIndexSpace<N,T> > diff_rhss;
    std::vector<SparsityMap<N,T> > sparsity_outputs;
    int approx_output_index;
    intptr_t approx_output_op;
  };

  template <int N, typename T, int N2, typename T2>
  class ImageOperation : public PartitioningOperation {
  public:
    ImageOperation(const ZIndexSpace<N,T>& _parent,
		   const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& _field_data,
		   const ProfilingRequestSet &reqs,
		   Event _finish_event);

    ImageOperation(const ZIndexSpace<N,T>& _parent,
		   const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZRect<N,T> > >& _field_data,
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
    std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > > ptr_data;
    std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZRect<N,T> > > range_data;
    std::vector<ZIndexSpace<N2,T2> > sources;
    std::vector<ZIndexSpace<N,T> > diff_rhss;
    std::vector<SparsityMap<N,T> > images;
  };
    
};

#endif // REALM_DEPPART_IMAGE_H
