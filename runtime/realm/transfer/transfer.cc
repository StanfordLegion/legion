/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include <realm/transfer/transfer.h>

#include <realm/transfer/lowlevel_dma.h>
#include <realm/mem_impl.h>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDomain
  //

  TransferDomain::TransferDomain(void)
  {}

  TransferDomain::~TransferDomain(void)
  {}

  class TransferDomainIndexSpace : public TransferDomain {
  public:
    TransferDomainIndexSpace(IndexSpace _is);

    //protected:
    IndexSpace is;
  };

  TransferDomainIndexSpace::TransferDomainIndexSpace(IndexSpace _is)
    : is(_is)
  {}

  template <unsigned DIM>
  class TransferDomainRect : public TransferDomain {
  public:
    TransferDomainRect(LegionRuntime::Arrays::Rect<DIM> _r);

    //protected:
    LegionRuntime::Arrays::Rect<DIM> r;
  };

  template <unsigned DIM>
  TransferDomainRect<DIM>::TransferDomainRect(LegionRuntime::Arrays::Rect<DIM> _r)
    : r(_r)
  {}

  /*static*/ TransferDomain *TransferDomain::construct(Domain d)
  {
    switch(d.get_dim()) {
    case 0: return new TransferDomainIndexSpace(d.get_index_space());
    case 1: return new TransferDomainRect<1>(d.get_rect<1>());
    case 2: return new TransferDomainRect<2>(d.get_rect<2>());
    case 3: return new TransferDomainRect<3>(d.get_rect<3>());
    }
    assert(0);
    return 0;
  }

  // HACK!
  Domain cvt(const TransferDomain *td)
  {
    const TransferDomainIndexSpace *tdis = dynamic_cast<const TransferDomainIndexSpace *>(td);
    if(tdis)
      return Domain(tdis->is);

    const TransferDomainRect<1> *tdr1 = dynamic_cast<const TransferDomainRect<1> *>(td);
    if(tdr1)
      return Domain::from_rect<1>(tdr1->r);

    const TransferDomainRect<2> *tdr2 = dynamic_cast<const TransferDomainRect<2> *>(td);
    if(tdr2)
      return Domain::from_rect<2>(tdr2->r);

    const TransferDomainRect<3> *tdr3 = dynamic_cast<const TransferDomainRect<3> *>(td);
    if(tdr3)
      return Domain::from_rect<3>(tdr3->r);

    assert(0);
    return Domain(IndexSpace::NO_SPACE);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferPlan
  //

  TransferPlan::TransferPlan(void)
  {}

  TransferPlan::~TransferPlan(void)
  {}

  class TransferPlanCopy : public TransferPlan {
  public:
    TransferPlanCopy(OASByInst *_oas_by_inst);
    virtual ~TransferPlanCopy(void);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority);

  protected:
    OASByInst *oas_by_inst;
  };

  TransferPlanCopy::TransferPlanCopy(OASByInst *_oas_by_inst)
    : oas_by_inst(_oas_by_inst)
  {}

  TransferPlanCopy::~TransferPlanCopy(void)
  {
    delete oas_by_inst;
  }

  static gasnet_node_t select_dma_node(Memory src_mem, Memory dst_mem,
				       ReductionOpID redop_id, bool red_fold)
  {
    gasnet_node_t src_node = ID(src_mem).memory.owner_node;
    gasnet_node_t dst_node = ID(dst_mem).memory.owner_node;

    bool src_is_rdma = get_runtime()->get_memory_impl(src_mem)->kind == MemoryImpl::MKIND_GLOBAL;
    bool dst_is_rdma = get_runtime()->get_memory_impl(dst_mem)->kind == MemoryImpl::MKIND_GLOBAL;

    if(src_is_rdma) {
      if(dst_is_rdma) {
	// gasnet -> gasnet - blech
	log_dma.warning("WARNING: gasnet->gasnet copy being serialized on local node (%d)", gasnet_mynode());
	return gasnet_mynode();
      } else {
	// gathers by the receiver
	return dst_node;
      }
    } else {
      if(dst_is_rdma) {
	// writing to gasnet is also best done by the sender
	return src_node;
      } else {
	// if neither side is gasnet, favor the sender (which may be the same as the target)
	return src_node;
      }
    }
  }

  Event TransferPlanCopy::execute_plan(const TransferDomain *td,
				       const ProfilingRequestSet& requests,
				       Event wait_on, int priority)
  {
    Event ev = GenEventImpl::create_genevent()->current_event();

#if 0
    int priority = 0;
    if (get_runtime()->get_memory_impl(src_mem)->kind == MemoryImpl::MKIND_GPUFB)
      priority = 1;
    else if (get_runtime()->get_memory_impl(dst_mem)->kind == MemoryImpl::MKIND_GPUFB)
      priority = 1;
#endif

    // ask which node should perform the copy
    Memory src_mem, dst_mem;
    {
      assert(!oas_by_inst->empty());
      OASByInst::const_iterator it = oas_by_inst->begin();
      src_mem = it->first.first.get_location();
      dst_mem = it->first.second.get_location();
    }
    gasnet_node_t dma_node = select_dma_node(src_mem, dst_mem, 0, false);
    log_dma.debug() << "copy: srcmem=" << src_mem << " dstmem=" << dst_mem
		    << " node=" << dma_node;

    CopyRequest *r = new CopyRequest(cvt(td), oas_by_inst, 
				     wait_on, ev, priority, requests);
    // we've given oas_by_inst to the copyrequest, so forget it
    assert(oas_by_inst != 0);
    oas_by_inst = 0;

    if(dma_node == gasnet_mynode()) {
      log_dma.debug("performing copy on local node");

      get_runtime()->optable.add_local_operation(ev, r);
      r->check_readiness(false, dma_queue);
    } else {
      RemoteCopyArgs args;
      args.redop_id = 0;
      args.red_fold = false;
      args.before_copy = wait_on;
      args.after_copy = ev;
      args.priority = priority;

      size_t msglen = r->compute_size();
      void *msgdata = malloc(msglen);

      r->serialize(msgdata);

      log_dma.debug("performing copy on remote node (%d), event=" IDFMT, dma_node, args.after_copy.id);
      get_runtime()->optable.add_remote_operation(ev, dma_node);
      RemoteCopyMessage::request(dma_node, args, msgdata, msglen, PAYLOAD_FREE);

      // done with the local copy of the request
      r->remove_reference();
    }

    return ev;
  }

  class TransferPlanReduce : public TransferPlan {
  public:
    TransferPlanReduce(const std::vector<CopySrcDstField>& _srcs,
		       const CopySrcDstField& _dst,
		       ReductionOpID _redop_id, bool _red_fold);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority);

  protected:
    std::vector<CopySrcDstField> srcs;
    CopySrcDstField dst;
    ReductionOpID redop_id;
    bool red_fold;
  };

  TransferPlanReduce::TransferPlanReduce(const std::vector<CopySrcDstField>& _srcs,
					 const CopySrcDstField& _dst,
					 ReductionOpID _redop_id, bool _red_fold)
    : srcs(_srcs)
    , dst(_dst)
    , redop_id(_redop_id)
    , red_fold(_red_fold)
  {}

  Event TransferPlanReduce::execute_plan(const TransferDomain *td,
					 const ProfilingRequestSet& requests,
					 Event wait_on, int priority)
  {
    Event ev = GenEventImpl::create_genevent()->current_event();

    // TODO
    bool inst_lock_needed = false;

    ReduceRequest *r = new ReduceRequest(cvt(td),
					 srcs, dst,
					 inst_lock_needed,
					 redop_id, red_fold,
					 wait_on, ev,
					 0 /*priority*/, requests);

    gasnet_node_t src_node = ID(srcs[0].inst).instance.owner_node;
    if(src_node == gasnet_mynode()) {
      log_dma.debug("performing reduction on local node");

      get_runtime()->optable.add_local_operation(ev, r);	  
      r->check_readiness(false, dma_queue);
    } else {
      RemoteCopyArgs args;
      args.redop_id = redop_id;
      args.red_fold = red_fold;
      args.before_copy = wait_on;
      args.after_copy = ev;
      args.priority = 0 /*priority*/;

      size_t msglen = r->compute_size();
      void *msgdata = malloc(msglen);
      r->serialize(msgdata);

      log_dma.debug("performing reduction on remote node (%d), event=" IDFMT,
		    src_node, args.after_copy.id);
      get_runtime()->optable.add_remote_operation(ev, src_node);
      RemoteCopyMessage::request(src_node, args, msgdata, msglen, PAYLOAD_FREE);
      // done with the local copy of the request
      r->remove_reference();
    }
    return ev;
  }

  class TransferPlanFill : public TransferPlan {
  public:
    TransferPlanFill(const void *_data, size_t _size,
		     RegionInstance _inst, unsigned _offset);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority);

  protected:
    ByteArray data;
    RegionInstance inst;
    unsigned offset;
  };

  TransferPlanFill::TransferPlanFill(const void *_data, size_t _size,
				     RegionInstance _inst, unsigned _offset)
    : data(_data, _size)
    , inst(_inst)
    , offset(_offset)
  {}

  Event TransferPlanFill::execute_plan(const TransferDomain *td,
				       const ProfilingRequestSet& requests,
				       Event wait_on, int priority)
  {
    CopySrcDstField f;
    f.inst = inst;
    f.offset = offset;
    f.size = data.size();

    Event ev = GenEventImpl::create_genevent()->current_event();
    FillRequest *r = new FillRequest(cvt(td), f, data.base(), data.size(),
				     wait_on, ev, priority, requests);

    gasnet_node_t tgt_node = ID(inst).instance.owner_node;
    if(tgt_node == gasnet_mynode()) {
      get_runtime()->optable.add_local_operation(ev, r);
      r->check_readiness(false, dma_queue);
    } else {
      RemoteFillArgs args;
      args.inst = inst;
      args.offset = offset;
      args.size = data.size();
      args.before_fill = wait_on;
      args.after_fill = ev;
      //args.priority = 0;

      size_t msglen = r->compute_size();
      void *msgdata = malloc(msglen);

      r->serialize(msgdata);

      get_runtime()->optable.add_remote_operation(ev, tgt_node);

      RemoteFillMessage::request(tgt_node, args, msgdata, msglen, PAYLOAD_FREE);

      // release local copy of operation
      r->remove_reference();
    }

    return ev;
  }

  /*static*/ bool TransferPlan::plan_copy(std::vector<TransferPlan *>& plans,
					  const std::vector<CopySrcDstField> &srcs,
					  const std::vector<CopySrcDstField> &dsts,
					  ReductionOpID redop_id /*= 0*/,
					  bool red_fold /*= false*/)
  {
    if(redop_id == 0) {
      // not a reduction, so sort fields by src/dst mem pairs
      //log_new_dma.info("Performing copy op");

      OASByMem oas_by_mem;

      std::vector<CopySrcDstField>::const_iterator src_it = srcs.begin();
      std::vector<CopySrcDstField>::const_iterator dst_it = dsts.begin();
      unsigned src_suboffset = 0;
      unsigned dst_suboffset = 0;

      while((src_it != srcs.end()) && (dst_it != dsts.end())) {
	InstPair ip(src_it->inst, dst_it->inst);
	MemPair mp(get_runtime()->get_instance_impl(src_it->inst)->memory,
		   get_runtime()->get_instance_impl(dst_it->inst)->memory);

	// printf("I:(%x/%x) M:(%x/%x) sub:(%d/%d) src=(%d/%d) dst=(%d/%d)\n",
	//        ip.first.id, ip.second.id, mp.first.id, mp.second.id,
	//        src_suboffset, dst_suboffset,
	//        src_it->offset, src_it->size, 
	//        dst_it->offset, dst_it->size);

	OffsetsAndSize oas;
	oas.src_offset = src_it->offset + src_suboffset;
	oas.dst_offset = dst_it->offset + dst_suboffset;
	oas.size = min(src_it->size - src_suboffset, dst_it->size - dst_suboffset);
	oas.serdez_id = src_it->serdez_id;

	// This is a little bit of hack: if serdez_id != 0 we directly create a
	// separate copy plan instead of inserting it into ''oasvec''
	if (oas.serdez_id != 0) {
	  OASByInst* oas_by_inst = new OASByInst;
	  (*oas_by_inst)[ip].push_back(oas);
	  TransferPlanCopy *p = new TransferPlanCopy(oas_by_inst);
	  plans.push_back(p);
	} else {
	  // </SERDEZ_DMA>
	  OASByInst *oas_by_inst;
	  OASByMem::iterator it = oas_by_mem.find(mp);
	  if(it != oas_by_mem.end()) {
	    oas_by_inst = it->second;
	  } else {
	    oas_by_inst = new OASByInst;
	    oas_by_mem[mp] = oas_by_inst;
	  }
	  (*oas_by_inst)[ip].push_back(oas);
	}
	src_suboffset += oas.size;
	assert(src_suboffset <= src_it->size);
	if(src_suboffset == src_it->size) {
	  src_it++;
	  src_suboffset = 0;
	}
	dst_suboffset += oas.size;
	assert(dst_suboffset <= dst_it->size);
	if(dst_suboffset == dst_it->size) {
	  dst_it++;
	  dst_suboffset = 0;
	}
      }
      // make sure we used up both
      assert(src_it == srcs.end());
      assert(dst_it == dsts.end());

      log_dma.debug() << "copy: " << oas_by_mem.size() << " distinct src/dst mem pairs";

      for(OASByMem::const_iterator it = oas_by_mem.begin(); it != oas_by_mem.end(); it++) {
	OASByInst *oas_by_inst = it->second;
	// TODO: teach new DMA code to handle multiple instances in the same memory
	for(OASByInst::const_iterator it2 = oas_by_inst->begin();
	    it2 != oas_by_inst->end();
	    ++it2) {
	  OASByInst *new_oas_by_inst = new OASByInst;
	  (*new_oas_by_inst)[it2->first] = it2->second;
	  TransferPlanCopy *p = new TransferPlanCopy(new_oas_by_inst);
	  plans.push_back(p);
	}
	// done with original oas_by_inst
	delete oas_by_inst;
      }
    } else {
      // reduction op case

      // sanity checks:
      // 1) all sources in same node
      for(size_t i = 1; i < srcs.size(); i++)
	assert(ID(srcs[i].inst).instance.owner_node == ID(srcs[0].inst).instance.owner_node);
      // 2) single destination field
      assert(dsts.size() == 1);

      TransferPlanReduce *p = new TransferPlanReduce(srcs, dsts[0],
						     redop_id, red_fold);
      plans.push_back(p);
    }

    return true;
  }

  /*static*/ bool TransferPlan::plan_fill(std::vector<TransferPlan *>& plans,
					  const std::vector<CopySrcDstField> &dsts,
					  const void *fill_value,
					  size_t fill_value_size)
  {
    // when 'dsts' contains multiple fields, the 'fill_value' should look
    // like a packed struct with a fill value for each field in order -
    // track the offset and complain if we run out of data
    size_t fill_ofs = 0;
    for(std::vector<CopySrcDstField>::const_iterator it = dsts.begin();
	it != dsts.end();
	++it) {
      if((fill_ofs + it->size) > fill_value_size) {
	log_dma.fatal() << "insufficient data for fill - need at least "
			<< (fill_ofs + it->size) << " bytes, but have only " << fill_value_size;
	assert(0);
      }
      TransferPlan *p = new TransferPlanFill(((const char *)fill_value) + fill_ofs,
					     it->size,
					     it->inst,
					     it->offset);
      plans.push_back(p);

      // special case: if a field uses all of the fill value, the next
      //  field (if any) is allowed to use the same value
      if((fill_ofs > 0) || (it->size != fill_value_size))
	fill_ofs += it->size;
    }

    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Domain
  //

  Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		     const std::vector<CopySrcDstField>& dsts,
		     Event wait_on,
		     ReductionOpID redop_id, bool red_fold) const
  {
    Realm::ProfilingRequestSet reqs;
    return Domain::copy(srcs, dsts, reqs, wait_on, redop_id, red_fold);
  }

  Event Domain::copy(const std::vector<CopySrcDstField>& srcs,
		     const std::vector<CopySrcDstField>& dsts,
		     const Realm::ProfilingRequestSet &requests,
		     Event wait_on,
		     ReductionOpID redop_id, bool red_fold) const
  {
    TransferDomain *td = TransferDomain::construct(*this);
    std::vector<TransferPlan *> plans;
    bool ok = TransferPlan::plan_copy(plans, srcs, dsts, redop_id, red_fold);
    assert(ok);
    std::set<Event> finish_events;
    for(std::vector<TransferPlan *>::iterator it = plans.begin();
	it != plans.end();
	++it) {
      Event e = (*it)->execute_plan(td, requests, wait_on, 0 /*priority*/);
      finish_events.insert(e);
      delete *it;
    }
    return Event::merge_events(finish_events);
  }

  Event Domain::fill(const std::vector<CopySrcDstField> &dsts,
		     const void *fill_value, size_t fill_value_size,
		     Event wait_on /*= Event::NO_EVENT*/) const
  {
    Realm::ProfilingRequestSet reqs;
    return Domain::fill(dsts, reqs, fill_value, fill_value_size, wait_on);
  }

  Event Domain::fill(const std::vector<CopySrcDstField> &dsts,
		     const Realm::ProfilingRequestSet &requests,
		     const void *fill_value, size_t fill_value_size,
		     Event wait_on /*= Event::NO_EVENT*/) const
  {
    TransferDomain *td = TransferDomain::construct(*this);
    std::vector<TransferPlan *> plans;
    bool ok = TransferPlan::plan_fill(plans, dsts, fill_value, fill_value_size);
    assert(ok);
    std::set<Event> finish_events;
    for(std::vector<TransferPlan *>::iterator it = plans.begin();
	it != plans.end();
	++it) {
      Event e = (*it)->execute_plan(td, requests, wait_on, 0 /*priority*/);
      finish_events.insert(e);
      delete *it;
    }
    return Event::merge_events(finish_events);
  }

}; // namespace Realm
