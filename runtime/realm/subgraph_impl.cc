/* Copyright 2019 Stanford University, NVIDIA Corporation
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

    typedef std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, unsigned> TopoMap;
    TopoMap toposort;

    unsigned nextval = 0;
    for(unsigned i = 0; i < defn.tasks.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_TASK, i)] = nextval++;
    for(unsigned i = 0; i < defn.copies.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_COPY, i)] = nextval++;
    for(unsigned i = 0; i < defn.arrivals.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_ARRIVAL, i)] = nextval++;
    for(unsigned i = 0; i < defn.instantiations.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_INSTANTIATION, i)] = nextval++;
    for(unsigned i = 0; i < defn.acquires.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_ACQUIRE, i)] = nextval++;
    for(unsigned i = 0; i < defn.releases.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_RELEASE, i)] = nextval++;
    unsigned total_ops = nextval;

    // sort by performing passes over dependency list...
    // any dependency whose target is before the source is resolved by
    //  moving the target to be after everybody
    // takes at most depth (<= N) passes unless there are loops
    bool converged = false;
    for(unsigned i = 0; !converged && (i < total_ops); i++) {
      converged = true;
      for(std::vector<SubgraphDefinition::Dependency>::const_iterator it = defn.dependencies.begin();
	  it != defn.dependencies.end();
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
    assert(converged);

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
    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it = defn.dependencies.begin();
	it != defn.dependencies.end();
	++it) {
      if(it->tgt_op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND) continue;
      if(it->tgt_op_index >= num_ext_postcond)
	num_ext_postcond = it->tgt_op_index + 1;
    }
    for(unsigned i = 0; i < num_ext_postcond; i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_EXT_POSTCOND, i)] = total_ops++;

    impl->schedule.resize(total_ops);
    for(TopoMap::const_iterator it = toposort.begin();
	it != toposort.end();
	++it) {
      impl->schedule[it->second].op_kind = it->first.first;
      impl->schedule[it->second].op_index = it->first.second;
    }

    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it = defn.dependencies.begin();
	it != defn.dependencies.end();
	++it) {
      TopoMap::const_iterator tgt = toposort.find(std::make_pair(it->tgt_op_kind, it->tgt_op_index));
      assert(tgt != toposort.end());

      switch(it->src_op_kind) {
      case SubgraphDefinition::OPKIND_EXT_PRECOND:
	{
	  // external preconditions are encoded as negative indices
	  int idx = -1 - it->src_op_index;
	  impl->schedule[tgt->second].preconditions.push_back(idx);
	  break;
	}

      default:
	{
	  TopoMap::const_iterator src = toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
	  assert(src != toposort.end());
	  impl->schedule[tgt->second].preconditions.push_back(src->second);
	  break;
	}
      }
    }

    return Event::NO_EVENT;
  }

  void Subgraph::destroy(Event wait_on /*= Event::NO_EVENT*/) const
  {
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
			      const ProfilingRequestSet& prs,
			      Event wait_on /*= Event::NO_EVENT*/,
			      int priority_adjust /*= 0*/) const
  {
    NodeID target_node = ID(*this).subgraph_owner_node();

    Event finish_event = GenEventImpl::create_genevent()->current_event();

    if(target_node == Network::my_node_id) {
      std::vector<Event> preconditions;
      std::vector<Event> postconditions;
      SubgraphImpl *impl = get_runtime()->get_subgraph_impl(*this);
      impl->instantiate(args, arglen, prs, 
			preconditions,
			postconditions,
			wait_on, finish_event,
			priority_adjust);
    } else {
      assert(0);
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

    if(target_node == Network::my_node_id) {
      SubgraphImpl *impl = get_runtime()->get_subgraph_impl(*this);
      impl->instantiate(args, arglen, prs, 
			preconditions,
			postconditions,
			wait_on, finish_event,
			priority_adjust);
    } else {
      assert(0);
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
				SubgraphDefinition::Interpolation::TargetKind target_kind,
				unsigned target_index)
  {
    for(std::vector<SubgraphDefinition::Interpolation>::const_iterator it = interpolations.begin();
	it != interpolations.end();
	++it)
      if((it->target_kind == target_kind) &&
	 (it->target_index == target_index))
	return true;
    return false;
  }

  static void do_interpolation(const std::vector<SubgraphDefinition::Interpolation>& interpolations,
			       SubgraphDefinition::Interpolation::TargetKind target_kind,
			       unsigned target_index,
			       const void *srcdata, size_t srclen,
			       void *dstdata, size_t dstlen)
  {
    for(std::vector<SubgraphDefinition::Interpolation>::const_iterator it = interpolations.begin();
	it != interpolations.end();
	++it) {
      if((it->target_kind != target_kind) ||
	 (it->target_index != target_index)) continue;

      assert((it->offset + it->bytes) <= srclen);
      if(it->redop_id == 0) {
	// overwrite
	assert((it->target_offset + it->bytes) <= dstlen);
	memcpy(reinterpret_cast<char *>(dstdata) + it->target_offset,
	       reinterpret_cast<const char *>(srcdata) + it->offset,
	       it->bytes);
      } else {
	assert(0);
      }
    }
  }

  void SubgraphImpl::instantiate(const void *args, size_t arglen,
				 const ProfilingRequestSet& prs,
				 const std::vector<Event>& preconditions,
				 std::vector<Event>& postconditions,
				 Event start_event, Event finish_event,
				 int priority_adjust)
  {
    std::vector<Event> events;

    for(std::vector<SubgraphScheduleEntry>::const_iterator it = schedule.begin();
	it != schedule.end();
	++it) {
      // assemble precondition
      std::set<Event> preconds;
      bool need_global_precond = start_event.exists();
      for(std::vector<int>::const_iterator it2 = it->preconditions.begin();
	  it2 != it->preconditions.end();
	  ++it2) {
	if(*it2 >= 0) {
	  assert(*it2 < int(events.size()));
	  preconds.insert(events[*it2]);
	  // we get the global precondition transitively...
	  need_global_precond = false;
	} else {
	  int idx = -1 - *it2;
	  if((idx < int(preconditions.size())) && preconditions[idx].exists())
	    preconds.insert(preconditions[idx]);
	}
      }
      if(need_global_precond)
	preconds.insert(start_event);

      Event pre = Event::merge_events(preconds);

      switch(it->op_kind) {
      case SubgraphDefinition::OPKIND_TASK:
	{
	  const SubgraphDefinition::TaskDesc& td = defn->tasks[it->op_index];
	  Processor proc = td.proc;
	  Processor::TaskFuncID task_id = td.task_id;
	  int priority = td.priority;
	  // TODO: avoid copy if no interpolation is needed
	  ByteArray taskargs(td.args);
	  do_interpolation(defn->interpolations,
			   SubgraphDefinition::Interpolation::TARGET_TASK_ARGS,
			   it->op_index,
			   args, arglen,
			   taskargs.base(), taskargs.size());
	  Event e = proc.spawn(task_id, taskargs.base(), taskargs.size(),
			       td.prs,
			       pre,
			       priority + priority_adjust);
	  events.push_back(e);
	  break;
	}

      case SubgraphDefinition::OPKIND_COPY:
	{
	  const SubgraphDefinition::CopyDesc& cd = defn->copies[it->op_index];
	  Event e = cd.space.copy(cd.srcs,
				  cd.dsts,
				  cd.prs,
				  pre);
	  events.push_back(e);
	  break;
	}

      case SubgraphDefinition::OPKIND_ARRIVAL:
	{
	  const SubgraphDefinition::ArrivalDesc& ad = defn->arrivals[it->op_index];
	  Barrier b = ad.barrier;
	  assert(!has_interpolation(defn->interpolations,
				    SubgraphDefinition::Interpolation::TARGET_ARRIVAL_BARRIER,
				    it->op_index));
	  unsigned count = ad.count;
	  b.arrive(count, pre,
		   ad.reduce_value.base(),
		   ad.reduce_value.size());
	  // "finish event" is precondition
	  events.push_back(pre);
	  break;
	}

      case SubgraphDefinition::OPKIND_ACQUIRE:
	{
	  const SubgraphDefinition::AcquireDesc& ad = defn->acquires[it->op_index];
	  Reservation rsrv = ad.rsrv;
	  unsigned mode = ad.mode;
	  bool excl = ad.exclusive;
	  Event e = rsrv.acquire(mode, excl, pre);
	  events.push_back(e);
	  break;
	}

      case SubgraphDefinition::OPKIND_RELEASE:
	{
	  const SubgraphDefinition::ReleaseDesc& rd = defn->releases[it->op_index];
	  Reservation rsrv = rd.rsrv;
	  rsrv.release(pre);
	  // "finish event" is precondition
	  events.push_back(pre);
	  break;
	}

      case SubgraphDefinition::OPKIND_EXT_POSTCOND:
	{
	  // only write postconditions the caller cares about
	  if(it->op_index < postconditions.size())
	    postconditions[it->op_index] = pre;
	  break;
	}

      default:
	assert(0);
      }
    }

    if(!events.empty()) {
      // use the event's merger to wait for completion of all nodes
      GenEventImpl *event_impl = get_genevent_impl(finish_event);
      event_impl->merger.prepare_merger(finish_event, false /*!ignore_faults*/,
					events.size());
      for(std::vector<Event>::const_iterator it = events.begin();
	  it != events.end();
	  ++it)
	event_impl->merger.add_precondition(*it);
      event_impl->merger.arm_merger();
    } else {
      GenEventImpl::trigger(finish_event, false /*!poisoned*/);
    }
  }


}; // namespace Realm
