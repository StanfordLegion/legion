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

// Realm subgraph implementation

#include "realm/subgraph_impl.h"
#include "realm/runtime_impl.h"

namespace Realm {

  Logger log_subgraph("subgraph");


  ////////////////////////////////////////////////////////////////////////
  //
  // class Subgraph

  /*static*/ const Subgraph Subgraph::NO_SUBGRAPH = { 0 };

  /*static*/ Event Subgraph::create_subgraph(Subgraph& subgraph,
					     const SubgraphDefinition& defn,
					     const ProfilingRequestSet& prs,
					     Event wait_on /*= Event::NO_EVENT*/)
  {
    NodeID target_node = Network::my_node_id;
    SubgraphImpl *impl = get_runtime()->local_subgraph_free_lists[target_node]->alloc_entry();
    impl->me.subgraph_creator_node() = Network::my_node_id;
    subgraph = impl->me.convert<Subgraph>();

    impl->defn = new SubgraphDefinition(defn);

    // no handling of preconditions or profiling yet
    assert(wait_on.has_triggered());
    assert(prs.empty());

    if(impl->compile()) {
      log_subgraph.info() << "created: subgraph=" << subgraph << " ops=" << impl->schedule.size();
      return Event::NO_EVENT;
    } else {
      // fatal error for now - once we have profiling, return a poisoned event
      //  if there was a profiling request for OperationStatus
      log_subgraph.fatal() << "subgraph compilation failed";
      abort();
    }
  }

  void Subgraph::destroy(Event wait_on /*= Event::NO_EVENT*/) const
  {
    NodeID owner = ID(*this).subgraph_owner_node();

    log_subgraph.info() << "destroy: subgraph=" << *this << " wait_on=" << wait_on;

    if(owner == Network::my_node_id) {
      SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(*this);

      if(wait_on.has_triggered())
	subgraph->destroy();
      else
	subgraph->deferred_destroy.defer(subgraph, wait_on);
    } else {
      ActiveMessage<SubgraphDestroyMessage> amsg(owner);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg.commit();
    }
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
			      const ProfilingRequestSet& prs,
			      Event wait_on /*= Event::NO_EVENT*/,
			      int priority_adjust /*= 0*/) const
  {
    NodeID target_node = ID(*this).subgraph_owner_node();

    Event finish_event = GenEventImpl::create_genevent()->current_event();

    log_subgraph.info() << "instantiate: subgraph=" << *this
			 << " before=" << wait_on << " after=" << finish_event;

    if(target_node == Network::my_node_id) {
      SubgraphImpl *impl = get_runtime()->get_subgraph_impl(*this);
      impl->instantiate(args, arglen, prs,
			empty_span() /*preconditions*/,
			empty_span() /*postconditions*/,
			wait_on, finish_event,
			priority_adjust);
    } else {
      Serialization::ByteCountSerializer bcs;
      {
	bool ok = (bcs.append_bytes(args, arglen) &&
		   (bcs << span<const Event>()) &&
		   (bcs << span<const Event>()) &&
		   (bcs << prs));
	assert(ok);
      }
      size_t msglen = bcs.bytes_used();
      ActiveMessage<SubgraphInstantiateMessage> amsg(target_node, msglen);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg->finish_event = finish_event;
      amsg->arglen = arglen;
      amsg->priority_adjust = priority_adjust;
      {
	amsg.add_payload(args, arglen);
	bool ok = ((amsg << span<const Event>()) &&
		   (amsg << span<const Event>()) &&
		   (amsg << prs));
	assert(ok);
      }
      amsg.commit();
    }
    return finish_event;
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
			      const ProfilingRequestSet& prs,
			      const std::vector<Event>& preconditions,
			      std::vector<Event>& postconditions,
			      Event wait_on /*= Event::NO_EVENT*/,
			      int priority_adjust /*= 0*/) const
  {
    NodeID target_node = ID(*this).subgraph_owner_node();

    Event finish_event = GenEventImpl::create_genevent()->current_event();

    // need to pre-create all the postcondition events too
    for(size_t i = 0; i < postconditions.size(); i++)
      postconditions[i] = GenEventImpl::create_genevent()->current_event();

    log_subgraph.info() << "instantiate: subgraph=" << *this
			<< " before=" << wait_on << " after=" << finish_event
			<< " preconds=" << PrettyVector<Event>(preconditions)
			<< " postconds=" << PrettyVector<Event>(postconditions);

    if(target_node == Network::my_node_id) {
      SubgraphImpl *impl = get_runtime()->get_subgraph_impl(*this);
      impl->instantiate(args, arglen, prs, 
			preconditions,
			postconditions,
			wait_on, finish_event,
			priority_adjust);
    } else {
      Serialization::ByteCountSerializer bcs;
      {
	bool ok = (bcs.append_bytes(args, arglen) &&
		   (bcs << preconditions) &&
		   (bcs << postconditions) &&
		   (bcs << prs));
	assert(ok);
      }
      size_t msglen = bcs.bytes_used();
      ActiveMessage<SubgraphInstantiateMessage> amsg(target_node, msglen);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg->finish_event = finish_event;
      amsg->arglen = arglen;
      amsg->priority_adjust = priority_adjust;
      {
	amsg.add_payload(args, arglen);
	bool ok = ((amsg << preconditions) &&
		   (amsg << postconditions) &&
		   (amsg << prs));
	assert(ok);
      }
      amsg.commit();
    }
    return finish_event;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphImpl

  SubgraphImpl::SubgraphImpl()
    : me(Subgraph::NO_SUBGRAPH)
  {}

  SubgraphImpl::~SubgraphImpl()
  {}

  void SubgraphImpl::init(ID _me, int _owner)
  {
    me = _me;
    assert(NodeID(me.subgraph_owner_node()) == NodeID(_owner));
  }

  static bool has_interpolation(const std::vector<SubgraphDefinition::Interpolation>& interpolations,
				unsigned first_interp, unsigned num_interps,
				SubgraphDefinition::Interpolation::TargetKind target_kind,
				unsigned target_index)
  {
    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation& it = interpolations[first_interp + i];
      if((it.target_kind == target_kind) &&
	 (it.target_index == target_index))
	return true;
    }
    return false;
  }

  class InterpolationScratchHelper {
  public:
    template <unsigned N>
    InterpolationScratchHelper(char (&prealloc)[N], size_t _needed)
      : needed(_needed), used(0)
    {
      if(needed > N) {
	need_free = true;
	base = static_cast<char *>(malloc(N));
	assert(base != 0);
      } else {
	need_free = false;
	base = prealloc;
      }
    }

    ~InterpolationScratchHelper()
    {
      if(need_free)
	free(base);
    }

    void *next(size_t bytes)
    {
      void *p = base + used;
      used += bytes;
      assert(used <= needed);
      return p;
    }

  protected:
    size_t needed, used;
    bool need_free;
    char *base;
  };
  
  // performs any necessary interpolations, making a copy of the destination
  //  in the supplied scratch memory if needed, and returns a pointer to either
  //  the original if no changes were made or the scratch if the copy was
  //  performed
  static const void *do_interpolation(const std::vector<SubgraphDefinition::Interpolation>& interpolations,
				      unsigned first_interp, unsigned num_interps,
				      SubgraphDefinition::Interpolation::TargetKind target_kind,
				      unsigned target_index,
				      const void *srcdata, size_t srclen,
				      const void *dstdata, size_t dstlen,
				      InterpolationScratchHelper& scratch_helper)
  {
    void *scratch_buffer = 0;
    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation& it = interpolations[first_interp + i];
      if((it.target_kind != target_kind) ||
	 (it.target_index != target_index)) continue;

      // match - make the copy if we haven't already
      if(scratch_buffer == 0) {
	scratch_buffer = scratch_helper.next(dstlen);
	memcpy(scratch_buffer, dstdata, dstlen);
      }

      assert((it.offset + it.bytes) <= srclen);
      if(it.redop_id == 0) {
	// overwrite
	assert((it.target_offset + it.bytes) <= dstlen);
	memcpy(reinterpret_cast<char *>(scratch_buffer) + it.target_offset,
	       reinterpret_cast<const char *>(srcdata) + it.offset,
	       it.bytes);
      } else {
	const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(it.redop_id, 0);
	assert((it.target_offset + redop->sizeof_lhs) <= dstlen);
	(redop->cpu_apply_excl_fn)(reinterpret_cast<char *>(scratch_buffer) + it.target_offset,
                                   0,
                                   reinterpret_cast<const char *>(srcdata) + it.offset,
                                   0,
                                   1 /*count*/,
                                   redop->userdata);
      }
    }

    return ((scratch_buffer != 0) ? scratch_buffer : dstdata);
  }

  // a typed version for interpolating small values
  template <typename T>
  static T do_interpolation(const std::vector<SubgraphDefinition::Interpolation>& interpolations,
			    unsigned first_interp, unsigned num_interps,
			    SubgraphDefinition::Interpolation::TargetKind target_kind,
			    unsigned target_index,
			    const void *srcdata, size_t srclen,
			    T dstdata)
  {
    T val = dstdata;

    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation& it = interpolations[first_interp + i];
      if((it.target_kind != target_kind) ||
	 (it.target_index != target_index)) continue;

      assert((it.offset + it.bytes) <= srclen);
      if(it.redop_id == 0) {
	// overwrite
	assert((it.target_offset + it.bytes) <= sizeof(T));
	memcpy(reinterpret_cast<char *>(&val) + it.target_offset,
	       reinterpret_cast<const char *>(srcdata) + it.offset,
	       it.bytes);
      } else {
	const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(it.redop_id, 0);
	assert((it.target_offset + redop->sizeof_lhs) <= sizeof(T));
	(redop->cpu_apply_excl_fn)(reinterpret_cast<char *>(&val) + it.target_offset,
                                   0,
                                   reinterpret_cast<const char *>(srcdata) + it.offset,
                                   0,
                                   1 /*count*/,
                                   redop->userdata);
      }
    }

    return val;
  }

  class SortInterpolationsByKindAndIndex {
  public:
    bool operator()(const SubgraphDefinition::Interpolation& a,
		    const SubgraphDefinition::Interpolation& b) const
    {
      // ignore bottom 8 bits of interpolation kinds so we're just looking
      //  at the operation kind
      unsigned a_opkind = a.target_kind >> 8;
      unsigned b_opkind = b.target_kind >> 8;
      return ((a_opkind < b_opkind) ||
	      ((a_opkind == b_opkind) && (a.target_index < b.target_index)));
    }
  };

  bool SubgraphImpl::compile(void)
  {
    typedef std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, unsigned> TopoMap;
    TopoMap toposort;

    unsigned nextval = 0;
    for(unsigned i = 0; i < defn->tasks.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_TASK, i)] = nextval++;
    for(unsigned i = 0; i < defn->copies.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_COPY, i)] = nextval++;
    for(unsigned i = 0; i < defn->arrivals.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_ARRIVAL, i)] = nextval++;
    for(unsigned i = 0; i < defn->instantiations.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_INSTANTIATION, i)] = nextval++;
    for(unsigned i = 0; i < defn->acquires.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_ACQUIRE, i)] = nextval++;
    for(unsigned i = 0; i < defn->releases.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_RELEASE, i)] = nextval++;
    unsigned total_ops = nextval;

    // for subgraph instantiations, we need to do a pass over the dependencies
    //  to see which ports are used
    std::vector<unsigned> inst_pre_max_port(defn->instantiations.size(), 0);
    std::vector<unsigned> inst_post_max_port(defn->instantiations.size(), 0);

    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it = defn->dependencies.begin();
	it != defn->dependencies.end();
	++it) {
      if(it->src_op_kind == SubgraphDefinition::OPKIND_INSTANTIATION) {
	inst_post_max_port[it->src_op_index] = std::max(inst_post_max_port[it->src_op_index],
							it->src_op_port);
      } else
	assert(it->src_op_port == 0);

      if(it->tgt_op_kind == SubgraphDefinition::OPKIND_INSTANTIATION) {
	inst_pre_max_port[it->tgt_op_index] = std::max(inst_pre_max_port[it->tgt_op_index],
						       it->tgt_op_port);
      } else
	assert(it->tgt_op_port == 0);
    }
    
    // sort by performing passes over dependency list...
    // any dependency whose target is before the source is resolved by
    //  moving the target to be after everybody
    // takes at most depth (<= N) passes unless there are loops
    bool converged = false;
    for(unsigned i = 0; !converged && (i < total_ops); i++) {
      converged = true;
      for(std::vector<SubgraphDefinition::Dependency>::const_iterator it = defn->dependencies.begin();
	  it != defn->dependencies.end();
	  ++it) {
	// external pre/post-conditions are always satisfied
	if(it->src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND) continue;
	if(it->tgt_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND) continue;

	TopoMap::const_iterator src = toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
	assert(src != toposort.end());

	TopoMap::iterator tgt = toposort.find(std::make_pair(it->tgt_op_kind, it->tgt_op_index));
	assert(tgt != toposort.end());

	if(src->second > tgt->second) {
	  tgt->second = nextval++;
	  converged = false;
	}
      }
    }
    if(!converged) {
      log_subgraph.error() << "subgraph sort did not converge - has a cycle?";
      return false;
    }

    // re-compact the ordering indices
    unsigned curval = 0;
    while(curval < total_ops) {
      TopoMap::iterator best = toposort.end();
      for(TopoMap::iterator it = toposort.begin(); it != toposort.end(); ++it)
	if((it->second >= curval) && ((best == toposort.end()) ||
				      (best->second > it->second)))
	  best = it;
      assert(best != toposort.end());
      best->second = curval++;
    }

    // if there are any external postconditions, add them to the end of the
    //  toposort
    unsigned num_ext_postcond = 0;
    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it = defn->dependencies.begin();
	it != defn->dependencies.end();
	++it) {
      if(it->tgt_op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND) continue;
      if(it->tgt_op_index >= num_ext_postcond)
	num_ext_postcond = it->tgt_op_index + 1;
    }
    for(unsigned i = 0; i < num_ext_postcond; i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_EXT_POSTCOND, i)] = total_ops++;

    schedule.resize(total_ops);
    for(TopoMap::const_iterator it = toposort.begin();
	it != toposort.end();
	++it) {
      schedule[it->second].op_kind = it->first.first;
      schedule[it->second].op_index = it->first.second;
    }

    // count number of intermediate events - instantiations can produce more
    //  than one
    num_intermediate_events = 0;
    num_final_events = 0;
    for(std::vector<SubgraphScheduleEntry>::iterator it = schedule.begin();
	it != schedule.end();
	++it) {
      if(it->op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND) {
	// we'll clear this later if we find our contribution to the final
	//  event is done transitively
	it->is_final_event = true;
	num_final_events++;
      } else
	it->is_final_event = false;

      it->intermediate_event_base = num_intermediate_events;

      if(it->op_kind == SubgraphDefinition::OPKIND_INSTANTIATION)
	it->intermediate_event_count = inst_post_max_port[it->op_index] + 1;
      else if(it->op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND)
	it->intermediate_event_count = 1;
      else
	it->intermediate_event_count = 0;

      num_intermediate_events += it->intermediate_event_count;
    }

    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it = defn->dependencies.begin();
	it != defn->dependencies.end();
	++it) {
      TopoMap::const_iterator tgt = toposort.find(std::make_pair(it->tgt_op_kind, it->tgt_op_index));
      assert(tgt != toposort.end());

      switch(it->src_op_kind) {
      case SubgraphDefinition::OPKIND_EXT_PRECOND:
	{
	  // external preconditions are encoded as negative indices
	  int idx = -1 - (int)(it->src_op_index);
	  schedule[tgt->second].preconditions.push_back(std::make_pair(it->tgt_op_port, idx));
	  break;
	}

      default:
	{
	  TopoMap::const_iterator src = toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
	  assert(src != toposort.end());
	  unsigned ev_idx = schedule[src->second].intermediate_event_base + it->src_op_port;
	  schedule[tgt->second].preconditions.push_back(std::make_pair(it->tgt_op_port, ev_idx));
	  // if we are depending on port 0 of another node and we're not an
	  //  external postcondition, then the preceeding node is not final
	  if((it->src_op_port == 0) &&
	     (it->tgt_op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND) &&
	     (schedule[src->second].is_final_event)) {
	    schedule[src->second].is_final_event = false;
	    num_final_events--;
	  }
	  break;
	}
      }
    }

    // now sort the preconditions for each entry - allows us to group by port
    //  and also notice duplicates
    max_preconditions = 1;  // have to count global precondition when needed
    for(std::vector<SubgraphScheduleEntry>::iterator it = schedule.begin();
	it != schedule.end();
	++it) {
      if(it->preconditions.empty()) continue;
      
      std::sort(it->preconditions.begin(), it->preconditions.end());
      // look for duplicates past the first event
      size_t num_unique = 1;
      for(size_t i = 1; i < it->preconditions.size(); i++)
	if(it->preconditions[i] != it->preconditions[num_unique - 1]) {
	  if(num_unique < i)
	    it->preconditions[num_unique] = it->preconditions[i];
	  num_unique++;
	}
      if(num_unique < it->preconditions.size())
	it->preconditions.resize(num_unique);
      if(num_unique >= max_preconditions)
	max_preconditions = num_unique + 1;
    }
    
    // sort the interpolations so that each operation has a compact range
    //  to iterate through
    std::sort(defn->interpolations.begin(), defn->interpolations.end(),
	      SortInterpolationsByKindAndIndex());
    for(std::vector<SubgraphScheduleEntry>::iterator it = schedule.begin();
	it != schedule.end();
	++it) {
      // binary search to find an interpolation for this operation
      unsigned lo = 0;
      unsigned hi = defn->interpolations.size();
      while(true) {
	if(lo >= hi) {
	  // search failed - no interpolations
	  it->first_interp = it->num_interps = 0;
	  break;
	}
	unsigned mid = (lo + hi) >> 1;
	int mid_opkind = defn->interpolations[mid].target_kind >> 8;
	if(it->op_kind < mid_opkind) {
	  hi = mid;
	} else if(it->op_kind > mid_opkind) {
	  lo = mid + 1;
	} else {
	  if(it->op_index < defn->interpolations[mid].target_index) {
	    hi = mid;
	  } else if(it->op_index > defn->interpolations[mid].target_index) {
	    lo = mid + 1;
	  } else {
	    // found a value - now scan linearly up and down for full range
	    lo = mid;
	    while((lo > 0) &&
		  ((defn->interpolations[lo - 1].target_kind >> 8) == it->op_kind) &&
		  (defn->interpolations[lo - 1].target_index == it->op_index))
	      lo--;
	    hi = mid + 1;
	    while((hi < defn->interpolations.size()) &&
		  ((defn->interpolations[hi].target_kind >> 8) == it->op_kind) &&
		  (defn->interpolations[hi].target_index == it->op_index))
	      hi++;
	    it->first_interp = lo;
	    it->num_interps = hi - lo;
	    //log_subgraph.print() << "search (" << it->op_kind << "," << it->op_index << ") -> " << lo << " .. " << hi;
	    break;
	  }
	}
      }
    }

    // also sanity-check that any interpolation using a reduction op has it
    //  defined and sizes match up
    for(std::vector<SubgraphDefinition::Interpolation>::iterator it = defn->interpolations.begin();
	it != defn->interpolations.end();
	++it) {
      if(it->redop_id != 0) {
	const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(it->redop_id, 0);
	if(redop == 0) {
	  log_subgraph.error() << "no reduction op registered for ID " << it->redop_id;
	  return false;
	}
	if(redop->sizeof_rhs != it->bytes) {
	  log_subgraph.error() << "reduction op size mismatch";
	  return false;
	}
      }
    }

    return true;
  }

  void SubgraphImpl::instantiate(const void *args, size_t arglen,
				 const ProfilingRequestSet& prs,
				 span<const Event> preconditions,
				 span<const Event> postconditions,
				 Event start_event, Event finish_event,
				 int priority_adjust)
  {
    // we precomputed the number of intermediate events we need, so put them
    //  on the stack
    Event *intermediate_events = static_cast<Event *>(alloca(num_intermediate_events *
							     sizeof(Event)));
    size_t cur_intermediate_events = 0;

    // we've also computed how many events will contribute to the finish
    //  event, so we can arm the merger as we go
    GenEventImpl *event_impl = 0;
    if(num_final_events > 0) {
      event_impl = get_genevent_impl(finish_event);
      event_impl->merger.prepare_merger(finish_event, false /*!ignore_faults*/,
					num_final_events);
    }

    Event *preconds = static_cast<Event *>(alloca(max_preconditions *
						  sizeof(Event)));
    
    for(std::vector<SubgraphScheduleEntry>::const_iterator it = schedule.begin();
	it != schedule.end();
	++it) {
      // assemble precondition
      size_t num_preconds = 0;
      bool need_global_precond = start_event.exists();

      size_t pc_idx = 0;
      while(pc_idx < it->preconditions.size()) {
	// if we see something for a nonzero port, save those for later
	if(it->preconditions[pc_idx].first != 0)
	  break;

	if(it->preconditions[pc_idx].second >= 0) {
	  // this is a dependency on another operation
	  assert(unsigned(it->preconditions[pc_idx].second) < cur_intermediate_events);
	  preconds[num_preconds++] = intermediate_events[it->preconditions[pc_idx].second];
	  // we get the global precondition transitively...
	  need_global_precond = false;
	} else {
	  // external precondition
	  int idx = -1 - it->preconditions[pc_idx].second;
	  if((idx < int(preconditions.size())) && preconditions[idx].exists())
	    preconds[num_preconds++] = preconditions[idx];
	}

	pc_idx++;
      }
      if(need_global_precond)
	preconds[num_preconds++] = start_event;

      assert(num_preconds <= max_preconditions);

      // for external postconditions, merge the preconditions directly into the
      //  returned event
      if(it->op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND) {
	// only bother if the caller wanted the event
	if(it->op_index < postconditions.size()) {
	  Event post_event = postconditions[it->op_index];
	  if(num_preconds > 0) {
	    GenEventImpl *post_impl = get_genevent_impl(post_event);
	    post_impl->merger.prepare_merger(post_event,
					     false /*!ignore_faults*/,
					     num_preconds);
	    for(size_t i = 0; i < num_preconds; i++)
	      post_impl->merger.add_precondition(preconds[i]);
	    post_impl->merger.arm_merger();
	  } else
	    GenEventImpl::trigger(post_event, false /*!poisoned*/);
	}
	continue;
      }

      span<const Event> s(preconds, num_preconds);
      Event pre = GenEventImpl::merge_events(s, false);
#if 0
      Event pre = GenEventImpl::merge_events(make_span<const Event>(preconds,
								    num_preconds),
					     false /*!ignore_faults*/);
#endif
      // scratch buffer used for interpolations
      const size_t SCRATCH_SIZE = 1024;
      char interp_scratch[SCRATCH_SIZE];

      Event e = Event::NO_EVENT;

      switch(it->op_kind) {
      case SubgraphDefinition::OPKIND_TASK:
	{
	  const SubgraphDefinition::TaskDesc& td = defn->tasks[it->op_index];
	  Processor proc = td.proc;
	  Processor::TaskFuncID task_id = td.task_id;
	  int priority = td.priority;

	  size_t scratch_needed = 0;
	  if(has_interpolation(defn->interpolations,
			       it->first_interp, it->num_interps,
			       SubgraphDefinition::Interpolation::TARGET_TASK_ARGS,
			       it->op_index))
	    scratch_needed += td.args.size();
						   
	  InterpolationScratchHelper ish(interp_scratch, scratch_needed);

	  const void *task_args = do_interpolation(defn->interpolations,
						   it->first_interp, it->num_interps,
						   SubgraphDefinition::Interpolation::TARGET_TASK_ARGS,
						   it->op_index,
						   args, arglen,
						   td.args.base(), td.args.size(),
						   ish);

	  e = proc.spawn(task_id, task_args, td.args.size(),
			 td.prs,
			 pre,
			 priority + priority_adjust);
	  intermediate_events[cur_intermediate_events++] = e;
	  break;
	}

      case SubgraphDefinition::OPKIND_COPY:
	{
	  const SubgraphDefinition::CopyDesc& cd = defn->copies[it->op_index];
	  e = cd.space.copy(cd.srcs,
			    cd.dsts,
			    cd.prs,
			    pre);
	  intermediate_events[cur_intermediate_events++] = e;
	  break;
	}

      case SubgraphDefinition::OPKIND_ARRIVAL:
	{
	  const SubgraphDefinition::ArrivalDesc& ad = defn->arrivals[it->op_index];

	  InterpolationScratchHelper ish(interp_scratch,
					 ad.reduce_value.size());

	  Barrier b = do_interpolation(defn->interpolations,
				       it->first_interp, it->num_interps,
				       SubgraphDefinition::Interpolation::TARGET_ARRIVAL_BARRIER,
				       it->op_index,
				       args, arglen,
				       ad.barrier);
	  const void *red_val = do_interpolation(defn->interpolations,
						 it->first_interp, it->num_interps,
						 SubgraphDefinition::Interpolation::TARGET_ARRIVAL_VALUE,
						 it->op_index,
						 args, arglen,
						 ad.reduce_value.base(),
						 ad.reduce_value.size(),
						 ish);
	  unsigned count = ad.count;
	  b.arrive(count, pre, red_val, ad.reduce_value.size());

	  // "finish event" is precondition
	  intermediate_events[cur_intermediate_events++] = e = pre;
	  break;
	}

      case SubgraphDefinition::OPKIND_ACQUIRE:
	{
	  const SubgraphDefinition::AcquireDesc& ad = defn->acquires[it->op_index];
	  Reservation rsrv = ad.rsrv;
	  unsigned mode = ad.mode;
	  bool excl = ad.exclusive;
	  e = rsrv.acquire(mode, excl, pre);
	  intermediate_events[cur_intermediate_events++] = e;
	  break;
	}

      case SubgraphDefinition::OPKIND_RELEASE:
	{
	  const SubgraphDefinition::ReleaseDesc& rd = defn->releases[it->op_index];
	  Reservation rsrv = rd.rsrv;
	  rsrv.release(pre);
	  // "finish event" is precondition
	  intermediate_events[cur_intermediate_events++] = e = pre;
	  break;
	}

      case SubgraphDefinition::OPKIND_INSTANTIATION:
	{
	  const SubgraphDefinition::InstantiationDesc& id = defn->instantiations[it->op_index];
	  Subgraph sg_inner = id.subgraph;
	  int priority_adjust = id.priority_adjust;

	  size_t scratch_needed = 0;
	  if(has_interpolation(defn->interpolations,
			       it->first_interp, it->num_interps,
			       SubgraphDefinition::Interpolation::TARGET_INSTANCE_ARGS,
			       it->op_index))
	    scratch_needed += id.args.size();

	  InterpolationScratchHelper ish(interp_scratch, scratch_needed);

	  const void *inst_args = do_interpolation(defn->interpolations,
						   it->first_interp, it->num_interps,
						   SubgraphDefinition::Interpolation::TARGET_INSTANCE_ARGS,
						   it->op_index,
						   args, arglen,
						   id.args.base(), id.args.size(),
						   ish);

	  // TODO: avoid dynamic allocation?
	  std::vector<Event> inst_preconds, inst_postconds;

	  // how many preconditions do we need to form?
	  unsigned num_inst_preconds = (it->preconditions.empty() ?
					  0 :
					  it->preconditions.rbegin()->first);
	  //log_subgraph.print() << "inst_preconds = " << num_inst_preconds;
	  if(num_inst_preconds > 0) {
	    inst_preconds.resize(num_inst_preconds);
	    for(unsigned i = 0; i < num_inst_preconds; i++) {
	      std::vector<Event> evs;
	      // continue scanning preconditions where the previous scan(s) stopped
	      while((pc_idx < it->preconditions.size()) &&
		    (it->preconditions[pc_idx].first == (i + 1))) {
		if(it->preconditions[pc_idx].second >= 0) {
		  // this is a dependency on another operation
		  assert(unsigned(it->preconditions[pc_idx].second) < cur_intermediate_events);
		  evs.push_back(intermediate_events[it->preconditions[pc_idx].second]);
		} else {
		  // external precondition
		  int idx = -1 - it->preconditions[pc_idx].second;
		  if((idx < int(preconditions.size())) && preconditions[idx].exists())
		    evs.push_back(preconditions[idx]);
		}

		pc_idx++;
	      }

	      inst_preconds[i] = GenEventImpl::merge_events(evs,
							    false /*!ignore_faults*/);
	    }
	  }

	  if(it->intermediate_event_count > 1) {
	    //log_subgraph.print() << "inst_postconds = " << (it->intermediate_event_count - 1);
	    inst_postconds.resize(it->intermediate_event_count - 1);
	  }

	  e = sg_inner.instantiate(inst_args, id.args.size(),
				   id.prs,
				   inst_preconds,
				   inst_postconds,
				   pre,
				   priority_adjust);

	  intermediate_events[cur_intermediate_events] = e;
	  if(it->intermediate_event_count > 1)
	    memcpy(&intermediate_events[cur_intermediate_events + 1],
		   inst_postconds.data(),
		   (it->intermediate_event_count - 1) * sizeof(Event));
	  cur_intermediate_events += it->intermediate_event_count;
	  break;
	}

      default:
	assert(0);
      }

      // contribute to the final event if we need to
      if(it->is_final_event)
	event_impl->merger.add_precondition(e);
    }

    // sanity-check that we counted right
    assert(cur_intermediate_events == num_intermediate_events);

    if(num_final_events > 0) {
      event_impl->merger.arm_merger();
    } else {
      GenEventImpl::trigger(finish_event, false /*!poisoned*/);
    }
  }

  void SubgraphImpl::destroy(void)
  {
    delete defn;
    schedule.clear();

    // TODO: when we create subgraphs on remote nodes, send a message to the
    //  creator node so they can add it to their free list
    NodeID creator_node = ID(me).subgraph_creator_node();
    assert(creator_node == Network::my_node_id);
    NodeID owner_node = ID(me).subgraph_owner_node();
    assert(owner_node == Network::my_node_id);

    get_runtime()->local_subgraph_free_lists[owner_node]->free_entry(this);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphImpl::DeferredDestroy
  //

  void SubgraphImpl::DeferredDestroy::defer(SubgraphImpl *_subgraph, Event wait_on)
  {
    subgraph = _subgraph;
    EventImpl::add_waiter(wait_on, this);
  }

  void SubgraphImpl::DeferredDestroy::event_triggered(bool poisoned,
						      TimeLimit work_until)
  {
    assert(!poisoned);
    subgraph->destroy();
  }

  void SubgraphImpl::DeferredDestroy::print(std::ostream& os) const
  {
    os << "deferred subgraph destruction: subgraph=" << subgraph->me;
  }

  Event SubgraphImpl::DeferredDestroy::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphInstantiateMessage

  /*static*/ void SubgraphInstantiateMessage::handle_message(NodeID sender,
							 const SubgraphInstantiateMessage &msg,
							 const void *data, size_t datalen)
  {
    SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(msg.subgraph);
    span<const Event> preconditions, postconditions;
    ProfilingRequestSet prs;

    Serialization::FixedBufferDeserializer fbd(data, datalen);
    fbd.extract_bytes(0, msg.arglen);  // skip over instantiation args - we'll access those directly
    bool ok = ((fbd >> preconditions) &&
	       (fbd >> postconditions));
    if(ok && (fbd.bytes_left() > 0))
      ok = (fbd >> prs);
    assert(ok);

    subgraph->instantiate(data, msg.arglen, prs,
			  preconditions,
			  postconditions,
			  msg.wait_on, msg.finish_event,
			  msg.priority_adjust);
  }

  ActiveMessageHandlerReg<SubgraphInstantiateMessage> subgraph_instantiate_message_handler;


  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphDestroyMessage

  /*static*/ void SubgraphDestroyMessage::handle_message(NodeID sender,
							 const SubgraphDestroyMessage &msg,
							 const void *data, size_t datalen)
  {
    SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(msg.subgraph);

    if(msg.wait_on.has_triggered())
      subgraph->destroy();
    else
      subgraph->deferred_destroy.defer(subgraph, msg.wait_on);
  }

  ActiveMessageHandlerReg<SubgraphDestroyMessage> subgraph_destroy_message_handler;

}; // namespace Realm
