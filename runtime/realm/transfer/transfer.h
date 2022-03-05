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

// data transfer (a.k.a. dma) engine for Realm

#ifndef REALM_TRANSFER_H
#define REALM_TRANSFER_H

#include "realm/event.h"
#include "realm/memory.h"
#include "realm/indexspace.h"
#include "realm/atomics.h"
#include "realm/network.h"
#include "realm/operation.h"
#include "realm/transfer/channel.h"
#include "realm/profiling.h"

namespace Realm {

  // the data transfer engine has too much code to have it all be templated on the
  //  type of IndexSpace that is driving the transfer, so we need a widget that
  //  can hold an arbitrary IndexSpace and dispatch based on its type

  class XferDes;
  class AddressList;

  class TransferIterator {
  public:
    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);

    virtual ~TransferIterator(void);

    // must be called (and waited on) before iteration is possible
    virtual Event request_metadata(void);

    // specify the xd port used for indirect address flow control, if any
    virtual void set_indirect_input_port(XferDes *xd, int port_idx,
					 TransferIterator *inner_iter);

    virtual void reset(void) = 0;
    virtual bool done(void) = 0;

    // flag bits to control iterators
    enum {
      SRC_PARTIAL_OK   = (1 << 0),
      SRC_LINES_OK     = (1 << 1),
      SRC_PLANES_OK    = (1 << 2),
      SRC_FLAGMASK     = 0xff,

      DST_PARTIAL_OK   = (1 << 8),
      DST_LINES_OK     = (1 << 9),
      DST_PLANES_OK    = (1 << 10),
      DST_FLAGMASK     = 0xff00,

      PARTIAL_OK       = SRC_PARTIAL_OK | DST_PARTIAL_OK,
      LINES_OK         = SRC_LINES_OK   | DST_LINES_OK,
      PLANES_OK        = SRC_PLANES_OK  | DST_PLANES_OK,
    };

    struct AddressInfo {
      size_t base_offset;
      size_t bytes_per_chunk; // multiple of sizeof(T) unless PARTIAL_OK
      size_t num_lines;   // guaranteed to be 1 unless LINES_OK (i.e. 2D)
      size_t line_stride;
      size_t num_planes;  // guaranteed to be 1 unless PLANES_OK (i.e. 3D)
      size_t plane_stride;
    };

    // a custom address info interface for cases where linearized
    //  addresses are not suitable
    class AddressInfoCustom {
    protected:
      virtual ~AddressInfoCustom() {}

    public:
      // offers an N-D rectangle from a given piece of a given instance,
      //  along with a specified dimension order
      // return value is how many dimensions are accepted (0 = single
      //  point), which can be less than the input if the target has
      //  strict ordering rules
      virtual int set_rect(const RegionInstanceImpl *inst,
                           const InstanceLayoutPieceBase *piece,
                           size_t field_size, size_t field_offset,
                           int ndims,
                           const int64_t lo[/*ndims*/],
                           const int64_t hi[/*ndims*/],
                           const int order[/*ndims*/]) = 0;
    };

    // if a step is tentative, it must either be confirmed or cancelled before
    //  another one is possible
    virtual size_t step(size_t max_bytes, AddressInfo& info, unsigned flags,
			bool tentative = false) = 0;
    virtual size_t step_custom(size_t max_bytes, AddressInfoCustom& info,
                               bool tentative = false) = 0;

    virtual void confirm_step(void) = 0;
    virtual void cancel_step(void) = 0;

    virtual bool get_addresses(AddressList &addrlist,
                               const InstanceLayoutPieceBase *&nonaffine) = 0;
  };

  class TransferDomain {
  protected:
    TransferDomain(void);

  public:
    template <typename S>
    static TransferDomain *deserialize_new(S& deserializer);
    
    template <int N, typename T>
    static TransferDomain *construct(const IndexSpace<N,T>& is);

    virtual TransferDomain *clone(void) const = 0;

    virtual ~TransferDomain(void);

    virtual Event request_metadata(void) = 0;

    virtual bool empty(void) const = 0;
    virtual size_t volume(void) const = 0;

    virtual void choose_dim_order(std::vector<int>& dim_order,
				  const std::vector<CopySrcDstField>& srcs,
				  const std::vector<CopySrcDstField>& dsts,
				  const std::vector<IndirectionInfo *>& indirects,
				  bool force_fortran_order,
				  size_t max_stride) const = 0;

    virtual void count_fragments(RegionInstance inst,
                                 const std::vector<int>& dim_order,
                                 const std::vector<FieldID>& fields,
                                 const std::vector<size_t>& fld_sizes,
                                 std::vector<size_t>& fragments) const = 0;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      const std::vector<int>& dim_order,
					      const std::vector<FieldID>& fields,
					      const std::vector<size_t>& fld_offsets,
					      const std::vector<size_t>& fld_sizes) const = 0;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      RegionInstance peer,
					      const std::vector<FieldID>& fields,
					      const std::vector<size_t>& fld_offsets,
					      const std::vector<size_t>& fld_sizes) const = 0;

    virtual void print(std::ostream& os) const = 0;
  };

  class TransferOperation;

  // copies with generalized scatter and gather have a DAG that describes
  //  the overall transfer: nodes are transfer descriptors and edges are
  //  intermediate buffers
  struct TransferGraph {
    struct XDTemplate {
      NodeID target_node;
      //XferDesKind kind;
      XferDesFactory *factory;
      int gather_control_input;
      int scatter_control_input;
      XferDesRedopInfo redop;

      enum IOType {
	IO_INST,
	IO_INDIRECT_INST,
	IO_EDGE,
	IO_FILL_DATA
      };
      struct IO {
	IOType iotype;
	union {
	  struct {
	    RegionInstance inst;
	    unsigned fld_start;
	    unsigned fld_count;
	  } inst;
	  struct {
	    unsigned ind_idx;
	    unsigned port;
	    RegionInstance inst;
	    unsigned fld_start;
	    unsigned fld_count;
	  } indirect;
	  unsigned edge;
	  struct {
	    unsigned fill_start;
	    unsigned fill_size;
            size_t fill_total;
	  } fill;
	};
      };
      static IO mk_inst(RegionInstance _inst,
			unsigned _fld_start, unsigned _fld_count);
      static IO mk_indirect(unsigned _ind_idx, unsigned _port,
			    RegionInstance _inst,
			    unsigned _fld_start, unsigned _fld_count);
      static IO mk_edge(unsigned _edge);
      static IO mk_fill(unsigned _fill_start, unsigned _fill_size,
                        size_t _fill_total);

      std::vector<IO> inputs;  // TODO: short vectors
      std::vector<IO> outputs;

      // helper functions for initializing these things
      void set_simple(Channel *channel,
		      int in_edge, int out_edge);
    };
    struct IBInfo {
      Memory memory;
      size_t size;
    };
    std::vector<XDTemplate> xd_nodes;
    std::vector<IBInfo> ib_edges;
    std::vector<unsigned> ib_alloc_order;
  };

  class TransferDesc {
  public:
    template <int N, typename T>
    TransferDesc(IndexSpace<N,T> _is,
		 const std::vector<CopySrcDstField> &_srcs,
		 const std::vector<CopySrcDstField> &_dsts,
		 const std::vector<const typename CopyIndirection<N,T>::Base *> &_indirects,
		 const ProfilingRequestSet &requests);

  protected:
    // reference-counted - do not delete directly
    ~TransferDesc();

  public:
    void add_reference();
    void remove_reference();

    // returns true if the analysis is complete, and ib allocation can proceed
    // if the analysis isn't, returns false and op->allocate_ibs() will be
    //   called once it is
    bool request_analysis(TransferOperation *op);

    struct FieldInfo {
      FieldID id;
      size_t offset, size;
      CustomSerdezID serdez_id;
    };

  protected:
    atomic<int> refcount;

    void check_analysis_preconditions();
    void perform_analysis();

    class DeferredAnalysis : public EventWaiter {
    public:
      DeferredAnalysis(TransferDesc *_desc);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      TransferDesc *desc;
    };
    DeferredAnalysis deferred_analysis;

    friend class TransferOperation;

    TransferDomain *domain;
    std::vector<CopySrcDstField> srcs, dsts;
    std::vector<IndirectionInfo *> indirects;
    ProfilingRequestSet prs;

    Mutex mutex;
    atomic<bool> analysis_complete;
    std::vector<TransferOperation *> pending_ops;
    TransferGraph graph;
    std::vector<int> dim_order;
    std::vector<FieldInfo> src_fields, dst_fields;
    void *fill_data;
    size_t fill_size;
    ProfilingMeasurements::OperationMemoryUsage prof_usage;
    ProfilingMeasurements::OperationCopyInfo prof_cpinfo;
  };
               
  class IndirectionInfo {
  public:
    virtual ~IndirectionInfo(void) {}
    virtual Event request_metadata(void) = 0;

    virtual void generate_gather_paths(Memory dst_mem,
				       TransferGraph::XDTemplate::IO dst_edge,
				       unsigned indirect_idx,
				       unsigned src_fld_start,
				       unsigned src_fld_count,
				       size_t bytes_per_element,
				       CustomSerdezID serdez_id,
				       std::vector<TransferGraph::XDTemplate>& xd_nodes,
				       std::vector<TransferGraph::IBInfo>& ib_edges,
				       std::vector<TransferDesc::FieldInfo>& src_fields) = 0;

    virtual void generate_scatter_paths(Memory src_mem,
					TransferGraph::XDTemplate::IO src_edge,
					unsigned indirect_idx,
					unsigned dst_fld_start,
					unsigned dst_fld_count,
					size_t bytes_per_element,
					CustomSerdezID serdez_id,
					std::vector<TransferGraph::XDTemplate>& xd_nodes,
					std::vector<TransferGraph::IBInfo>& ib_edges,
					std::vector<TransferDesc::FieldInfo>& src_fields) = 0;

    virtual RegionInstance get_pointer_instance(void) const = 0;

    virtual TransferIterator *create_address_iterator(RegionInstance peer) const = 0;

    virtual TransferIterator *create_indirect_iterator(Memory addrs_mem,
						       RegionInstance inst,
						       const std::vector<FieldID>& fields,
						       const std::vector<size_t>& fld_offsets,
						       const std::vector<size_t>& fld_sizes) const = 0;

    virtual void print(std::ostream& os) const = 0;
  };

  std::ostream& operator<<(std::ostream& os, const IndirectionInfo& ii);


  // a TransferOperation is an application-requested copy/fill/reduce
  class TransferOperation : public Operation {
  public:
    TransferOperation(TransferDesc& _desc,
		      Event _precondition,
		      GenEventImpl *_finish_event,
		      EventImpl::gen_t _finish_gen);

    ~TransferOperation();

    virtual void print(std::ostream& os) const;

    void start_or_defer(void);

    virtual bool mark_ready(void);
    virtual bool mark_started(void);

    void allocate_ibs();
    void create_xds();

    void notify_ib_allocation(unsigned ib_index, off_t ib_offset);
    void notify_ib_allocations(unsigned count, unsigned first_index,
                               const off_t *offsets);
    void notify_xd_completion(XferDesID xd_id);

    class XDLifetimeTracker : public Operation::AsyncWorkItem {
    public:
      XDLifetimeTracker(TransferOperation *_op,
			XferDesID _xd_id);
      virtual void request_cancellation(void);
      virtual void print(std::ostream& os) const;

    protected:
      XferDesID xd_id;
    };

  protected:
    virtual void mark_completed(void);

    class DeferredStart : public EventWaiter {
    public:
      DeferredStart(TransferOperation *_op);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      TransferOperation *op;
    };
    DeferredStart deferred_start;

    TransferDesc& desc;
    Event precondition;
    std::vector<XferDesID> xd_ids;
    std::vector<XDLifetimeTracker *> xd_trackers;
    std::vector<off_t> ib_offsets;
    atomic<unsigned> ib_responses_needed;
    int priority;
  };

}; // namespace Realm

#include "realm/transfer/transfer.inl"

#endif // ifndef REALM_TRANSFER_H
