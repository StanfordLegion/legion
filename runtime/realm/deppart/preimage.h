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

// preimage operations for Realm dependent partitioning

#ifndef REALM_DEPPART_PREIMAGE_H
#define REALM_DEPPART_PREIMAGE_H

#include "realm/deppart/partitions.h"

namespace Realm {

  template <int N, typename T, int N2, typename T2>
  class PreimageMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;
    static const int DIM2 = N2;
    typedef T2 IDXTYPE2;

    PreimageMicroOp(IndexSpace<N,T> _parent_space, IndexSpace<N,T> _inst_space,
		    RegionInstance _inst, size_t _field_offset, bool _is_ranged);
    virtual ~PreimageMicroOp(void);

    void add_sparsity_output(IndexSpace<N2,T2> _target, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    friend struct RemoteMicroOpMessage<PreimageMicroOp<N,T,N2,T2> >;
    static ActiveMessageHandlerReg<RemoteMicroOpMessage<PreimageMicroOp<N,T,N2,T2> > > areg;

    friend class PartitioningMicroOp;
    template <typename S>
    REALM_ATTR_WARN_UNUSED(bool serialize_params(S& s) const);

    // construct from received packet
    template <typename S>
    PreimageMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop, S& s);

    template <typename BM>
    void populate_bitmasks_ptrs(std::map<int, BM *>& bitmasks);

    template <typename BM>
    void populate_bitmasks_ranges(std::map<int, BM *>& bitmasks);

    IndexSpace<N,T> parent_space, inst_space;
    RegionInstance inst;
    size_t field_offset;
    bool is_ranged;
    std::vector<IndexSpace<N2,T2> > targets;
    std::vector<SparsityMap<N,T> > sparsity_outputs;
  };

  template <typename T>
  struct ApproxImageResponseMessage;

  template <int N, typename T, int N2, typename T2>
  class PreimageOperation : public PartitioningOperation {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;
    static const int DIM2 = N2;
    typedef T2 IDXTYPE2;

    PreimageOperation(const IndexSpace<N, T> &_parent,
                      const DomainTransform<N2, T2, N, T> &_domain_transform,
                      const ProfilingRequestSet &reqs,
                      GenEventImpl *_finish_event,
                      EventImpl::gen_t _finish_gen);

    virtual ~PreimageOperation(void);

    IndexSpace<N,T> add_target(const IndexSpace<N2,T2>& target);

    virtual void execute(void);

    virtual void print(std::ostream& os) const;

    virtual void set_overlap_tester(void *tester);

    void provide_sparse_image(int index, const Rect<N2,T2> *rects, size_t count);

  protected:
    static ActiveMessageHandlerReg<ApproxImageResponseMessage<PreimageOperation<N,T,N2,T2> > > areg;

    IndexSpace<N, T> parent;
    DomainTransform<N2, T2, N, T> domain_transform;
    std::vector<IndexSpace<N2, T2> > targets;
    std::vector<SparsityMap<N, T> > preimages;
    Mutex mutex;
    OverlapTester<N2,T2> *overlap_tester;
    std::map<int, std::vector<Rect<N2,T2> > > pending_sparse_images;
    atomic<int> remaining_sparse_images;
    std::vector<atomic<int> > contrib_counts;
    AsyncMicroOp *dummy_overlap_uop;
  };

  template <typename T>
  struct ApproxImageResponseMessage {
    intptr_t approx_output_op;
    int approx_output_index;

    static void handle_message(NodeID sender,
			       const ApproxImageResponseMessage<T> &msg,
			       const void *data, size_t datalen);
  };

  template <int N, typename T, int N2, typename T2>
  class StructuredPreimageMicroOp : public PartitioningMicroOp {
  public:
    static const int DIM = N;
    typedef T IDXTYPE;
    static const int DIM2 = N2;
    typedef T2 IDXTYPE2;

    StructuredPreimageMicroOp(const StructuredTransform<N2, T2, N, T> &_transform,
                              IndexSpace<N, T> _parent_space);

    virtual ~StructuredPreimageMicroOp(void);

    void add_sparsity_output(IndexSpace<N2,T2> _target, SparsityMap<N,T> _sparsity);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:

   template <typename BM>
   void populate_bitmasks(std::map<int, BM *> &bitmasks);

   StructuredTransform<N2, T2, N, T> transform;
   IndexSpace<N, T> parent_space;
   std::vector<IndexSpace<N2, T2> > targets;
   std::vector<SparsityMap<N, T> > sparsity_outputs;
  };

  };  // namespace Realm

#endif // REALM_DEPPART_PREIMAGE_H
