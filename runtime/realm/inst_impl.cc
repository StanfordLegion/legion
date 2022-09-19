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

#include "realm/inst_impl.h"

#include "realm/event_impl.h"
#include "realm/mem_impl.h"
#include "realm/logging.h"
#include "realm/runtime_impl.h"
#include "realm/deppart/inst_helper.h"

TYPE_IS_SERIALIZABLE(Realm::InstanceLayoutGeneric::FieldLayout);

namespace Realm {

  Logger log_inst("inst");

  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstanceImpl::DeferredCreate
  //

  void RegionInstanceImpl::DeferredCreate::defer(RegionInstanceImpl *_inst,
						 MemoryImpl *_mem,
						 bool _need_alloc_result,
 						 Event wait_on)
  {
    inst = _inst;
    mem = _mem;
    need_alloc_result = _need_alloc_result;
    EventImpl::add_waiter(wait_on, this);
  }

  void RegionInstanceImpl::DeferredCreate::event_triggered(bool poisoned,
							   TimeLimit work_until)
  {
    if(poisoned)
      log_poison.info() << "poisoned deferred instance creation skipped - inst=" << inst;
    
    mem->allocate_storage_immediate(inst,
				    need_alloc_result, poisoned, work_until);
  }

  void RegionInstanceImpl::DeferredCreate::print(std::ostream& os) const
  {
    os << "deferred instance creation";
  }

  Event RegionInstanceImpl::DeferredCreate::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstanceImpl::DeferredDestroy
  //

  void RegionInstanceImpl::DeferredDestroy::defer(RegionInstanceImpl *_inst,
						  MemoryImpl *_mem,
						  Event wait_on)
  {
    inst = _inst;
    mem = _mem;
    EventImpl::add_waiter(wait_on, this);
  }

  void RegionInstanceImpl::DeferredDestroy::event_triggered(bool poisoned,
							    TimeLimit work_until)
  {
    if(poisoned)
      log_poison.info() << "poisoned deferred instance destruction skipped - POSSIBLE LEAK - inst=" << inst;
    
    mem->release_storage_immediate(inst, poisoned, work_until);
  }

  void RegionInstanceImpl::DeferredDestroy::print(std::ostream& os) const
  {
    os << "deferred instance destruction";
  }

  Event RegionInstanceImpl::DeferredDestroy::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class CompiledInstanceLayout
  //

  CompiledInstanceLayout::CompiledInstanceLayout()
    : program_base(0), program_size(0)
  {}

  CompiledInstanceLayout::~CompiledInstanceLayout()
  {
     reset();
  }

  void *CompiledInstanceLayout::allocate_memory(size_t bytes)
  {
    // TODO: allocate where GPUs can see it
    assert(program_base == 0);
#ifdef REALM_ON_WINDOWS
    program_base = _aligned_malloc(bytes, 16);
    assert(program_base != 0);
#else
    int ret = posix_memalign(&program_base, 16, bytes);
    assert(ret == 0);
#endif
    program_size = bytes;
    return program_base;
  }

  void CompiledInstanceLayout::reset()
  {
    if(program_base) {
#ifdef REALM_ON_WINDOWS
      _aligned_free(program_base);
#else
      free(program_base);
#endif
    }
    program_base = 0;
    program_size = 0;
    fields.clear();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayout<N,T>
  //

  static size_t roundup(size_t n, size_t mult)
  {
    n += mult - 1;
    n -= (n % mult);
    return n;
  }

  template <typename T>
  struct PieceSplitNode {
    std::vector<int> presplit_pieces;
    int split_dim;
    T split_plane;
    PieceSplitNode<T> *low_child, *high_child;
    size_t total_splits;

    PieceSplitNode() : low_child(0), high_child(0), total_splits(0) {}
    ~PieceSplitNode() { if (low_child) delete low_child; if (high_child) delete high_child; }

    template <int N>
    static PieceSplitNode<T> *compute_split(const std::vector<InstanceLayoutPiece<N,T> *>& pieces,
					    std::vector<int>& idxs)
    {
      PieceSplitNode<T> *n = new PieceSplitNode<T>;

      while(true) {
	// base case
	if(idxs.size() == 1) {
	  n->presplit_pieces.push_back(idxs[0]);
	  return n;
	}

	// we can either split in one of the dimensions as long as it doesn't
	//  cut through any given piece, or we can take the largest piece and
	//  test for it explicitly
	int best_dim = -1;
	T best_plane = 0;
	int best_idx = idxs[0];
	size_t best_vol = pieces[idxs[0]]->bounds.volume();
	size_t total_vol = best_vol;
	for(size_t i = 1; i < idxs.size(); i++) {
	  size_t v = pieces[idxs[i]]->bounds.volume();
	  total_vol += v;
	  if(v > best_vol) {
	    best_idx = idxs[i];
	    best_vol = v;
	  }
	}

	for(int dim = 0; dim < N; dim++) {
	  // make a list of the start/stop points for each piece in the
	  //  current dimension
	  std::vector<std::pair<T, int> > ss;
	  ss.reserve(idxs.size() * 2);
	  for(size_t i = 0; i < idxs.size(); i++) {
	    ss.push_back(std::make_pair(pieces[idxs[i]]->bounds.lo[dim],
					-1 - idxs[i]));
	    ss.push_back(std::make_pair(pieces[idxs[i]]->bounds.hi[dim],
					idxs[i]));
	  }

	  // sort list from lowest to highest (with starts before stops)
	  std::sort(ss.begin(), ss.end());

	  // now walk and consider any place with zero open intervals as a
	  //  cut point
	  int opens = 0;
	  size_t cur_vol = 0;
	  for(size_t i = 0; i < ss.size(); i++) {
	    if(opens == 0) {
	      // what's the smaller volume of this split?
	      size_t v = std::min(cur_vol, total_vol - cur_vol);
	      if(v > best_vol) {
		best_dim = dim;
		best_plane = ss[i].first;
		best_vol = v;
	      }
	    }
	    if(ss[i].second < 0) {
	      // beginning of an interval
	      opens++;
	      cur_vol += pieces[(-1 - ss[i].second)]->bounds.volume();
	    } else {
	      assert(opens > 0);
	      opens--;
	    }
	  }
	  assert(cur_vol == total_vol);
	  assert(opens == 0);
	}

	if(best_dim >= 0) {
	  n->split_dim = best_dim;
	  n->split_plane = best_plane;

	  // build two new idx lists
	  std::vector<int> lo_idxs, hi_idxs;
	  for(size_t i = 0; i < idxs.size(); i++)
	    if(pieces[idxs[i]]->bounds.hi[best_dim] < best_plane)
	      lo_idxs.push_back(idxs[i]);
	    else
	      hi_idxs.push_back(idxs[i]);

	  // and recursively divide them
	  n->low_child = compute_split(pieces, lo_idxs);
	  n->high_child = compute_split(pieces, hi_idxs);
	  n->total_splits = (1 +
			     n->low_child->total_splits +
			     n->high_child->total_splits);
	  return n;
	} else {
	  // best was to just grab the biggest piece, so do that
	  n->presplit_pieces.push_back(best_idx);
	  // find and remove best_idx from idxs
	  for(size_t i = 0; i < idxs.size() - 1; i++)
	    if(idxs[i] == best_idx) {
	      idxs[i] = idxs[idxs.size() - 1];
	      break;
	    }
	  idxs.resize(idxs.size() - 1);
	  // wrap around and try again
	}
      }
    }

    template <int N>
    void print(const std::vector<InstanceLayoutPiece<N,T> *>& pieces, int indent)
    {
      for(size_t i = 0; i < presplit_pieces.size(); i++) {
	for(int j = 0; j < indent; j++) std::cout << ' ';
	std::cout << "piece[" << presplit_pieces[i] << "]: " << pieces[presplit_pieces[i]]->bounds << "\n";
      }
      if(total_splits > 0) {
	for(int j = 0; j < indent; j++) std::cout << ' ';
	std::cout << "split " << split_dim << " " << split_plane << "\n";
	low_child->print(pieces, indent + 2);
	for(int j = 0; j < indent; j++) std::cout << ' ';
	std::cout << "---\n";
	high_child->print(pieces, indent + 2);
      }
    }

    template <int N>
    char *generate_instructions(const std::vector<InstanceLayoutPiece<N,T> *>& pieces, char *next_inst, unsigned& usage_mask)
    {
      // generate tests against presplit pieces
      for(size_t i = 0; i < presplit_pieces.size(); i++) {
	size_t bytes = roundup(pieces[presplit_pieces[i]]->lookup_inst_size(), 16);

	// unless we're the last piece AND there's not subsequent split,
	//  we point at the next instruction
	unsigned next_delta = (((i < (presplit_pieces.size() - 1)) ||
				(total_splits > 0)) ?
			         (bytes >> 4) :
			         0);
	PieceLookup::Instruction *inst = pieces[presplit_pieces[i]]->create_lookup_inst(next_inst, next_delta);
	usage_mask |= (1U << inst->opcode());
	next_inst += roundup(bytes, 16);
      }

      if(total_splits > 0) {
	usage_mask |= PieceLookup::ALLOW_SPLIT1;
	size_t size = roundup(sizeof(PieceLookup::SplitPlane<N,T>), 16);
	char *cur_inst = next_inst;
	PieceLookup::SplitPlane<N,T> *sp =
	  new(next_inst) PieceLookup::SplitPlane<N,T>(split_dim,
						      split_plane,
						      0 /*we'll fix delta below*/);
	next_inst += size;

	// generate low half of tree then record delta for high tree
	next_inst = low_child->generate_instructions(pieces, next_inst,
						     usage_mask);
	size_t delta_bytes = next_inst - cur_inst;
	assert((delta_bytes & 15) == 0);
	assert(delta_bytes < (1 << 20));
	sp->set_delta(delta_bytes >> 4);

	next_inst = high_child->generate_instructions(pieces, next_inst,
						      usage_mask);
      }

      return next_inst;
    }

  };

  template <int N, typename T>
  void InstanceLayout<N,T>::compile_lookup_program(PieceLookup::CompiledProgram& p) const
  {
    // first, count up how many bytes we're going to need

    size_t total_bytes = 0;

    // each piece list that's used will turn into a program
    std::map<int, size_t> piece_list_starts;
    std::map<int, PieceSplitNode<T> *> piece_list_plans;
    for(std::map<FieldID, FieldLayout>::const_iterator it = fields.begin();
	it != fields.end();
	++it) {
      // did we already do this piece list?
      if(piece_list_starts.count(it->second.list_idx) > 0)
	continue;

      piece_list_starts[it->second.list_idx] = total_bytes;

      const InstancePieceList<N,T>& pl = piece_lists[it->second.list_idx];

      if(pl.pieces.empty()) {
	// need room for one abort instruction
	total_bytes += 16;
      } else {
	// each piece will need a corresponding instruction
	for(typename std::vector<InstanceLayoutPiece<N,T> *>::const_iterator it2 = pl.pieces.begin();
	    it2 != pl.pieces.end();
	    ++it2) {
	  size_t bytes = (*it2)->lookup_inst_size();
	  total_bytes += roundup(bytes, 16);
	}

	std::vector<int> idxs(pl.pieces.size());
	for(size_t i = 0; i < pl.pieces.size(); i++)
	  idxs[i] = i;
	PieceSplitNode<T> *plan = PieceSplitNode<T>::compute_split(pl.pieces,
								   idxs);
	piece_list_plans[it->second.list_idx] = plan;
	//plan->print(pl.pieces, 2);
	total_bytes += (plan->total_splits *
			roundup(sizeof(PieceLookup::SplitPlane<N,T>), 16));
      }
    }

    void *base = p.allocate_memory(total_bytes);
    // zero things out for sanity
    memset(base, 0, total_bytes);

    std::map<int, unsigned> piece_list_masks;
    // now generate programs
    for(std::map<int, size_t>::const_iterator it = piece_list_starts.begin();
	it != piece_list_starts.end();
	++it) {
      const InstancePieceList<N,T>& pl = piece_lists[it->first];
      char *next_inst = static_cast<char *>(base) + it->second;

      unsigned usage_mask = 0;

      if(pl.pieces.empty()) {
	// all zeros is ok for now
      } else {
	PieceSplitNode<T> *plan = piece_list_plans[it->first];
	plan->generate_instructions(pl.pieces, next_inst, usage_mask);
	delete plan;
      }

      piece_list_masks[it->first] = usage_mask;
    }

    // fill in per field info
    for(std::map<FieldID, FieldLayout>::const_iterator it = fields.begin();
	it != fields.end();
	++it) {
      PieceLookup::CompiledProgram::PerField& pf = p.fields[it->first];
      pf.start_inst = reinterpret_cast<const PieceLookup::Instruction *>(reinterpret_cast<uintptr_t>(base) + piece_list_starts[it->second.list_idx]);
      pf.inst_usage_mask = piece_list_masks[it->second.list_idx];
      pf.field_offset = it->second.rel_offset;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance
  //

    AddressSpace RegionInstance::address_space(void) const
    {
      return ID(id).instance_owner_node();
    }

    Memory RegionInstance::get_location(void) const
    {
      return (exists() ?
	        ID::make_memory(ID(id).instance_owner_node(),
				ID(id).instance_mem_idx()).convert<Memory>() :
	        Memory::NO_MEMORY);
    }

    /*static*/ Event RegionInstance::create_instance(RegionInstance& inst,
						     Memory memory,
						     InstanceLayoutGeneric *ilg,
						     const ProfilingRequestSet& prs,
						     Event wait_on)
    {
      return RegionInstanceImpl::create_instance(inst, memory, ilg, 0,
						 prs, wait_on);
    }

    /*static*/ Event RegionInstance::create_external_instance(RegionInstance& inst,
							      Memory memory,
							      InstanceLayoutGeneric *ilg,
							      const ExternalInstanceResource& res,
							      const ProfilingRequestSet& prs,
							      Event wait_on)
    {
      return RegionInstanceImpl::create_instance(inst, memory, ilg, &res,
						 prs, wait_on);
    }

    void RegionInstance::destroy(Event wait_on /*= Event::NO_EVENT*/) const
    {
      // we can immediately turn this into a (possibly-preconditioned) request to
      //  deallocate the instance's storage - the eventual callback from that
      //  will be what actually destroys the instance
      log_inst.info() << "instance destroyed: inst=" << *this << " wait_on=" << wait_on;

      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(*this);
      RegionInstanceImpl *inst_impl = mem_impl->get_instance(*this);
      mem_impl->release_storage_deferrable(inst_impl, wait_on);
    }

    void RegionInstance::destroy(const std::vector<DestroyedField>& destroyed_fields,
				 Event wait_on /*= Event::NO_EVENT*/) const
    {
      // TODO: actually call destructor
      if(!destroyed_fields.empty()) {
	log_inst.warning() << "WARNING: field destructors ignored - inst=" << *this;
      }
      destroy(wait_on);
    }

    // it is sometimes useful to re-register an existing instance (in whole or
    //  in part) as an "external" instance (e.g. to provide a different view
    //  on the same bits) - this hopefully gives an ExternalInstanceResource *
    //  (which must be deleted by the caller) that corresponds to this instance
    //  but may return a null pointer for instances that do not support
    //  re-registration
    ExternalInstanceResource *RegionInstance::generate_resource_info(bool read_only) const
    {
      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(*this);
      RegionInstanceImpl *inst_impl = mem_impl->get_instance(*this);
      return mem_impl->generate_resource_info(inst_impl,
					      0, span<const FieldID>(),
					      read_only);
    }

    ExternalInstanceResource *RegionInstance::generate_resource_info(const IndexSpaceGeneric& space,
								     span<const FieldID> fields,
								     bool read_only) const
    {
      MemoryImpl *mem_impl = get_runtime()->get_memory_impl(*this);
      RegionInstanceImpl *inst_impl = mem_impl->get_instance(*this);
      return mem_impl->generate_resource_info(inst_impl,
					      &space, fields,
					      read_only);
    }

    /*static*/ const RegionInstance RegionInstance::NO_INST = { 0 };

    // before you can get an instance's index space or construct an accessor for
    //  a given processor, the necessary metadata for the instance must be
    //  available on to that processor
    // this can require network communication and/or completion of the actual
    //  allocation, so an event is returned and (as always) the application
    //  must decide when/where to handle this precondition
    Event RegionInstance::fetch_metadata(Processor target) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);

      NodeID target_node = ID(target).proc_owner_node();
      if(target_node == Network::my_node_id) {
	// local metadata request
	return r_impl->request_metadata();
      } else {
	// prefetch on other node's behalf
	return r_impl->prefetch_metadata(target_node);
      }
    }

    const InstanceLayoutGeneric *RegionInstance::get_layout(void) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      return r_impl->metadata.layout;
    }

    // gets a compiled piece lookup program for a given field
    template <int N, typename T>
    const PieceLookup::Instruction *RegionInstance::get_lookup_program(FieldID field_id,
								       unsigned allowed_mask,
								       uintptr_t& field_offset)
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      std::map<FieldID, PieceLookup::CompiledProgram::PerField>::const_iterator it;
      it = r_impl->metadata.lookup_program.fields.find(field_id);
      assert(it != r_impl->metadata.lookup_program.fields.end());

      // bail out if the program requires unsupported instructions
      if((it->second.inst_usage_mask & ~allowed_mask) != 0)
	return 0;

      // the "field offset" picks up both the actual per-field offset but also
      //  the base of the instance itself
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      void *ptr = mem->get_inst_ptr(r_impl, 0,
				    r_impl->metadata.layout->bytes_used);
      assert(ptr != 0);
      field_offset = (reinterpret_cast<uintptr_t>(ptr) +
		      it->second.field_offset);

      return it->second.start_inst;
    }

    template <int N, typename T>
    const PieceLookup::Instruction *RegionInstance::get_lookup_program(FieldID field_id,
								       const Rect<N,T>& subrect,
								       unsigned allowed_mask,
								       size_t& field_offset)
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      std::map<FieldID, PieceLookup::CompiledProgram::PerField>::const_iterator it;
      it = r_impl->metadata.lookup_program.fields.find(field_id);
      assert(it != r_impl->metadata.lookup_program.fields.end());

      // bail out if the program requires unsupported instructions
      if((it->second.inst_usage_mask & ~allowed_mask) != 0)
	return 0;

      // the "field offset" picks up both the actual per-field offset but also
      //  the base of the instance itself
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      void *ptr = mem->get_inst_ptr(r_impl, 0,
				    r_impl->metadata.layout->bytes_used);
      assert(ptr != 0);
      field_offset = (reinterpret_cast<uintptr_t>(ptr) +
		      it->second.field_offset);

      // try to pre-execute part of the program based on the subrect given
      const PieceLookup::Instruction *i = it->second.start_inst;
      while(true) {
	if(i->opcode() == PieceLookup::Opcodes::OP_SPLIT1) {
	  const PieceLookup::SplitPlane<N,T> *sp = static_cast<const PieceLookup::SplitPlane<N,T> *>(i);
	  // if our subrect straddles the split plane, we have to stop here
	  if(sp->splits_rect(subrect))
	    break;

	  // otherwise all points in the rect go the same way and we can do
	  //  that now
	  i = sp->next(subrect.lo);
	} else
	  break;
      }

      return i;
    }

    void RegionInstance::read_untyped(size_t offset, void *data, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      mem->get_bytes(r_impl->metadata.inst_offset + offset, data, datalen);
    }

    void RegionInstance::write_untyped(size_t offset, const void *data, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      mem->put_bytes(r_impl->metadata.inst_offset + offset, data, datalen);
    }

    void RegionInstance::reduce_apply_untyped(size_t offset, ReductionOpID redop_id,
					      const void *data, size_t datalen,
					      bool exclusive /*= false*/) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      if(redop == 0) {
	log_inst.fatal() << "no reduction op registered for ID " << redop_id;
	abort();
      }
      // data should match RHS size
      assert(datalen == redop->sizeof_rhs);
      // can we run the reduction op directly on the memory location?
      void *ptr = mem->get_inst_ptr(r_impl, offset,
				    redop->sizeof_lhs);
      if(ptr) {
        if(exclusive)
          (redop->cpu_apply_excl_fn)(ptr, 0, data, 0, 1, redop->userdata);
        else
          (redop->cpu_apply_nonexcl_fn)(ptr, 0, data, 0, 1, redop->userdata);
      } else {
	// we have to do separate get/put, which means we cannot supply
	//  atomicity in the !exclusive case
	assert(exclusive);
	void *lhs_copy = alloca(redop->sizeof_lhs);
	mem->get_bytes(r_impl->metadata.inst_offset + offset,
		       lhs_copy, redop->sizeof_lhs);
        (redop->cpu_apply_excl_fn)(lhs_copy, 0, data, 0, 1, redop->userdata);
	mem->put_bytes(r_impl->metadata.inst_offset + offset,
		       lhs_copy, redop->sizeof_lhs);
      }
    }

    void RegionInstance::reduce_fold_untyped(size_t offset, ReductionOpID redop_id,
					     const void *data, size_t datalen,
					     bool exclusive /*= false*/) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      const ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      if(redop == 0) {
	log_inst.fatal() << "no reduction op registered for ID " << redop_id;
	abort();
      }
      // data should match RHS size
      assert(datalen == redop->sizeof_rhs);
      // can we run the reduction op directly on the memory location?
      void *ptr = mem->get_inst_ptr(r_impl, offset,
				    redop->sizeof_rhs);
      if(ptr) {
        if(exclusive)
          (redop->cpu_fold_excl_fn)(ptr, 0, data, 0, 1, redop->userdata);
        else
          (redop->cpu_fold_nonexcl_fn)(ptr, 0, data, 0, 1, redop->userdata);
      } else {
	// we have to do separate get/put, which means we cannot supply
	//  atomicity in the !exclusive case
	assert(exclusive);
	void *rhs1_copy = alloca(redop->sizeof_rhs);
	mem->get_bytes(r_impl->metadata.inst_offset + offset,
		       rhs1_copy, redop->sizeof_rhs);
        (redop->cpu_fold_excl_fn)(rhs1_copy, 0, data, 0, 1, redop->userdata);
	mem->put_bytes(r_impl->metadata.inst_offset + offset,
		       rhs1_copy, redop->sizeof_rhs);
      }
    }

    // returns a null pointer if the instance storage cannot be directly
    //  accessed via load/store instructions
    void *RegionInstance::pointer_untyped(size_t offset, size_t datalen) const
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);
      // metadata must already be available
      assert(r_impl->metadata.is_valid() &&
	     "instance metadata must be valid before accesses are performed");
      assert(r_impl->metadata.layout);
      MemoryImpl *mem = get_runtime()->get_memory_impl(r_impl->memory);
      void *ptr = mem->get_inst_ptr(r_impl, offset, datalen);
      return ptr;
    }

    void RegionInstance::get_strided_access_parameters(size_t start, size_t count,
						       ptrdiff_t field_offset, size_t field_size,
						       intptr_t& base, ptrdiff_t& stride)
    {
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(*this);

      // TODO: make sure we're in range

      void *orig_base = 0;
      size_t orig_stride = 0;
      bool ok = r_impl->get_strided_parameters(orig_base, orig_stride, field_offset);
      assert(ok);
      base = reinterpret_cast<intptr_t>(orig_base);
      stride = orig_stride;
    }

    void RegionInstance::report_instance_fault(int reason,
					       const void *reason_data,
					       size_t reason_size) const
    {
      assert(0);
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstanceImpl
  //

    RegionInstanceImpl::RegionInstanceImpl(RegionInstance _me, Memory _memory)
      : me(_me), memory(_memory) //, lis(0)
    {
      lock.init(ID(me).convert<Reservation>(), ID(me).instance_creator_node());
      lock.in_use = true;

      metadata.inst_offset = INSTOFFSET_UNALLOCATED;
      metadata.ready_event = Event::NO_EVENT;
      metadata.layout = 0;
      metadata.ext_resource = 0;
      metadata.mem_specific = 0;
      
      // Initialize this in case the user asks for profiling information
      timeline.instance = _me;
    }

    RegionInstanceImpl::~RegionInstanceImpl(void)
    {
      // clean up metadata if needed, but it's too late to send messages
      if(metadata.is_valid())
        metadata.initiate_cleanup(me.id, true /*local only*/);
    }

    /*static*/ Event RegionInstanceImpl::create_instance(RegionInstance& inst,
							 Memory memory,
							 InstanceLayoutGeneric *ilg,
							 const ExternalInstanceResource *res,
							 const ProfilingRequestSet& prs,
							 Event wait_on)
    {
      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
      RegionInstanceImpl *impl = m_impl->new_instance();
      // we can fail to get a valid pointer if we are out of instance slots
      if(!impl) {
	inst = RegionInstance::NO_INST;
	// import the profiling requests to see if anybody is paying attention to
	//  failure
	ProfilingMeasurementCollection pmc;
	pmc.import_requests(prs);
	bool reported = false;
	if(pmc.wants_measurement<ProfilingMeasurements::InstanceStatus>()) {
	  ProfilingMeasurements::InstanceStatus stat;
	  stat.result = ProfilingMeasurements::InstanceStatus::INSTANCE_COUNT_EXCEEDED;
	  stat.error_code = 0;
	  pmc.add_measurement(stat);
	  reported = true;
	}
	if(pmc.wants_measurement<ProfilingMeasurements::InstanceAbnormalStatus>()) {
	  ProfilingMeasurements::InstanceAbnormalStatus stat;
	  stat.result = ProfilingMeasurements::InstanceStatus::INSTANCE_COUNT_EXCEEDED;
	  stat.error_code = 0;
	  pmc.add_measurement(stat);
	  reported = true;
	}
	if(pmc.wants_measurement<ProfilingMeasurements::InstanceAllocResult>()) {
	  ProfilingMeasurements::InstanceAllocResult result;
	  result.success = false;
	  pmc.add_measurement(result);
	}
	if(!reported) {
	  // fatal error
	  log_inst.fatal() << "FATAL: instance count exceeded for memory " << memory;
	  assert(0);
	}
	// generate a poisoned event for completion
	GenEventImpl *ev = GenEventImpl::create_genevent();
	Event ready_event = ev->current_event();
	GenEventImpl::trigger(ready_event, true /*poisoned*/);
	return ready_event;
      }

      // set this handle before we do anything that can result in a
      //  profiling callback containing this instance handle
      inst = impl->me;

      impl->metadata.layout = ilg;
      if(res)
	impl->metadata.ext_resource = res->clone();
      else
	impl->metadata.ext_resource = 0;
      ilg->compile_lookup_program(impl->metadata.lookup_program);

      bool need_alloc_result = false;
      if (!prs.empty()) {
        impl->requests = prs;
        impl->measurements.import_requests(impl->requests);
        if(impl->measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>())
          impl->timeline.record_create_time();
	need_alloc_result = impl->measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>();
      }

      impl->metadata.need_alloc_result = need_alloc_result;
      impl->metadata.need_notify_dealloc = false;

      log_inst.debug() << "instance layout: inst=" << inst << " layout=" << *ilg;

      // request allocation of storage - note that due to the asynchronous
      //  nature of any profiling responses, it is not safe to refer to the
      //  instance metadata (whether the allocation succeeded or not) after
      //  this point)
      Event ready_event;
      switch(m_impl->allocate_storage_deferrable(impl,
						 need_alloc_result,
						 wait_on)) {
      case MemoryImpl::ALLOC_INSTANT_SUCCESS:
	{
	  // successful allocation
	  assert(impl->metadata.inst_offset <= RegionInstanceImpl::INSTOFFSET_MAXVALID);
	  ready_event = Event::NO_EVENT;
	  break;
	}

      case MemoryImpl::ALLOC_INSTANT_FAILURE:
      case MemoryImpl::ALLOC_CANCELLED:
	{
	  // generate a poisoned event for completion
	  // NOTE: it is unsafe to look at the impl->metadata or the 
	  //  passed-in instance layout at this point due to the possibility
	  //  of an asynchronous destruction of the instance in a profiling
	  //  handler
	  GenEventImpl *ev = GenEventImpl::create_genevent();
	  ready_event = ev->current_event();
	  GenEventImpl::trigger(ready_event, true /*poisoned*/);
	  break;
	}

      case MemoryImpl::ALLOC_DEFERRED:
	{
	  // we will probably need an event to track when it is ready
	  GenEventImpl *ev = GenEventImpl::create_genevent();
	  ready_event = ev->current_event();
	  bool alloc_done, alloc_successful;
	  // use mutex to avoid race on allocation callback
	  {
	    AutoLock<> al(impl->mutex);
	    switch(impl->metadata.inst_offset) {
	    case RegionInstanceImpl::INSTOFFSET_UNALLOCATED:
	    case RegionInstanceImpl::INSTOFFSET_DELAYEDALLOC:
	    case RegionInstanceImpl::INSTOFFSET_DELAYEDDESTROY:
	      {
		alloc_done = false;
		alloc_successful = false;
		impl->metadata.ready_event = ready_event;
		break;
	      }
	    case RegionInstanceImpl::INSTOFFSET_FAILED:
	      {
		alloc_done = true;
		alloc_successful = false;
		break;
	      }
	    default:
	      {
		alloc_done = true;
		alloc_successful = true;
		break;
	      }
	    }
	  }
	  if(alloc_done) {
	    // lost the race to the notification callback, so we trigger the
	    //  ready event ourselves
	    if(alloc_successful) {
	      GenEventImpl::trigger(ready_event, false /*!poisoned*/);
	      ready_event = Event::NO_EVENT;
	    } else {
	      // poison the ready event and still return it
	      GenEventImpl::trigger(ready_event, true /*poisoned*/);
	    }
	  }
	  break;
	}

      case MemoryImpl::ALLOC_EVENTUAL_SUCCESS:
      case MemoryImpl::ALLOC_EVENTUAL_FAILURE:
	// should not occur
	assert(0);
      }

      if(res)
	log_inst.info() << "instance created: inst=" << inst << " external=" << *res << " ready=" << ready_event;
      else
	log_inst.info() << "instance created: inst=" << inst << " bytes=" << ilg->bytes_used << " ready=" << ready_event;
      return ready_event;
    }

    void RegionInstanceImpl::send_metadata(const NodeSet& early_reqs)
    {
      log_inst.debug() << "sending instance metadata to early requestors: isnt=" << me;
      Serialization::DynamicBufferSerializer dbs(4096);
      metadata.serialize_msg(dbs);

      // fragment serialized metadata if needed
      size_t offset = 0;
      size_t total_bytes = dbs.bytes_used();

      while(offset < total_bytes) {
        size_t to_send = std::min(total_bytes - offset,
                                  ActiveMessage<MetadataResponseMessage>::recommended_max_payload(early_reqs,
                                                                                                  false /*without congestion*/));

        ActiveMessage<MetadataResponseMessage> amsg(early_reqs, to_send);
        amsg->id = ID(me).id;
        amsg->offset = offset;
        amsg->total_bytes = total_bytes;
        amsg.add_payload(static_cast<const char *>(dbs.get_buffer()) + offset,
                         to_send);
        amsg.commit();

        offset += to_send;
      }
    }

    void RegionInstanceImpl::notify_allocation(MemoryImpl::AllocationResult result,
					       size_t offset, TimeLimit work_until)
    {
      // response needs to be handled by the instance's creator node, so forward
      //  there if it's not us
      NodeID creator_node = ID(me).instance_creator_node();
      if(creator_node != Network::my_node_id) {
	// update our local metadata as well - TODO: clean this up
	switch(result) {
	case MemoryImpl::ALLOC_INSTANT_SUCCESS:
	case MemoryImpl::ALLOC_EVENTUAL_SUCCESS:
	  {	    
	    metadata.inst_offset = offset;
	    break;
	  }

	case MemoryImpl::ALLOC_DEFERRED:
	  {
	    // no change here - it was done atomically earlier
	    break;
	  }

	case MemoryImpl::ALLOC_CANCELLED:
	case MemoryImpl::ALLOC_INSTANT_FAILURE:
	case MemoryImpl::ALLOC_EVENTUAL_FAILURE:
	  {
	    metadata.inst_offset = INSTOFFSET_FAILED;
	    break;
	  }

	default:
	  assert(0);
	}
	  
	ActiveMessage<MemStorageAllocResponse> amsg(creator_node);
	amsg->inst = me;
	amsg->offset = offset;
	amsg->result = result;
	amsg.commit();

	return;
      }
      
      using namespace ProfilingMeasurements;

      if((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ||
	 (result == MemoryImpl::ALLOC_EVENTUAL_FAILURE) ||
	 (result == MemoryImpl::ALLOC_CANCELLED)) {
	// if somebody is listening to profiling measurements, we report
	//  a failed allocation through that channel - if not, we explode
	// exception: InstanceAllocResult is not enough for EVENTUAL_FAILURE,
	//  since we would have already said we thought it would succeed
	bool report_failure = (measurements.wants_measurement<InstanceStatus>() ||
			       measurements.wants_measurement<InstanceAbnormalStatus>() ||
			       (measurements.wants_measurement<InstanceAllocResult>() &&
				(result != MemoryImpl::ALLOC_EVENTUAL_FAILURE)));
	if(!report_failure) {
	  if((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ||
	     (result == MemoryImpl::ALLOC_EVENTUAL_FAILURE)) {
	    log_inst.fatal() << "instance allocation failed - out of memory in mem " << memory;
	    abort();
	  }

	  // exception: allocations that were cancelled would have had some
	  //  error response reported further up the chain, so let this
	  //  one slide
	  assert(result == MemoryImpl::ALLOC_CANCELLED);
	}
	
	log_inst.info() << "allocation failed: inst=" << me;

        // mark metadata valid before we return any profiling responses
        NodeSet early_reqs;
        metadata.inst_offset = (size_t)-2;
        metadata.mark_valid(early_reqs);
        if(!early_reqs.empty())
          send_metadata(early_reqs);

	// poison the completion event, if it exists
	Event ready_event = Event::NO_EVENT;
	{
	  AutoLock<> al(mutex);
	  ready_event = metadata.ready_event;
	  metadata.ready_event = Event::NO_EVENT;

          // adding measurements is not thread safe w.r.t. a deferral
          //  message, so do it with lock held
	  if(measurements.wants_measurement<InstanceStatus>()) {
	    InstanceStatus stat;
	    stat.result = ((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ?
                           InstanceStatus::FAILED_ALLOCATION :
			   InstanceStatus::CANCELLED_ALLOCATION);
	    stat.error_code = 0;
	    measurements.add_measurement(stat);
	  }

	  if(measurements.wants_measurement<InstanceAbnormalStatus>()) {
	    InstanceAbnormalStatus stat;
	    stat.result = ((result == MemoryImpl::ALLOC_INSTANT_FAILURE) ?
                             InstanceStatus::FAILED_ALLOCATION :
			     InstanceStatus::CANCELLED_ALLOCATION);
	    stat.error_code = 0;
	    measurements.add_measurement(stat);
	  }

	  if(metadata.need_alloc_result) {
#ifdef DEBUG_REALM
	    assert(measurements.wants_measurement<InstanceAllocResult>());
	    assert(!metadata.need_notify_dealloc);
#endif

	    // this is either the only result we will get or has raced ahead of
	    //  the deferral message
	    metadata.need_alloc_result = false;

	    InstanceAllocResult result;
	    result.success = false;
	    measurements.add_measurement(result);
	  }
	  
          // send any remaining incomplete profiling responses
          measurements.send_responses(requests);

          // clear the measurements after we send the response
          // TODO: move this to a single place on the instance destruction
          //  code path instead
          measurements.clear();
	}

	if(ready_event.exists())
	  GenEventImpl::trigger(ready_event, true /*poisoned*/, work_until);
	return;
      }

      if(result == MemoryImpl::ALLOC_DEFERRED) {
	// this should only be received if an InstanceAllocRequest measurement
	//  was requested, but we have to be careful about recording the
	//  expected-future-success because it may race with the actual success
	//  (or unexpected failure), so use the mutex
	bool need_notify_dealloc = false;
	{
	  AutoLock<> al(mutex);

	  if(metadata.need_alloc_result) {
#ifdef DEBUG_REALM
	    assert(measurements.wants_measurement<ProfilingMeasurements::InstanceAllocResult>());
#endif
	    ProfilingMeasurements::InstanceAllocResult result;
	    result.success = true;
	    measurements.add_measurement(result);

	    metadata.need_alloc_result = false;

	    // if we were super-slow, notification of the subsequent
	    //  deallocation may have been delayed
	    need_notify_dealloc = metadata.need_notify_dealloc;
	    metadata.need_notify_dealloc = false;
	  }
	}

	if(need_notify_dealloc)
	  notify_deallocation();

	return;
      }

      log_inst.debug() << "allocation completed: inst=" << me << " offset=" << offset;

      // before we publish the offset, we need to update the layout
      // SJT: or not?  that might be part of RegionInstance::get_base_address?
      //metadata.layout->relocate(offset);

      // update must be performed with the metadata mutex held to make sure there
      //  are no races between it and getting the ready event 
      Event ready_event;
      {
	AutoLock<> al(mutex);
	ready_event = metadata.ready_event;
	metadata.ready_event = Event::NO_EVENT;
	metadata.inst_offset = offset;

	// adding measurements is not thread safe w.r.t. a deferral
	//  message, so do it with lock held
	if(metadata.need_alloc_result) {
#ifdef DEBUG_REALM
	  assert(measurements.wants_measurement<InstanceAllocResult>());
	  assert(!metadata.need_notify_dealloc);
#endif

	  // this is either the only result we will get or has raced ahead of
	  //  the deferral message
	  metadata.need_alloc_result = false;

	  ProfilingMeasurements::InstanceAllocResult result;
	  result.success = true;
	  measurements.add_measurement(result);
	}

	// the InstanceMemoryUsage measurement is added at creation time for
	//  profilers that want that before instance deletion occurs
	if(measurements.wants_measurement<ProfilingMeasurements::InstanceMemoryUsage>()) {
	  ProfilingMeasurements::InstanceMemoryUsage usage;
	  usage.instance = me;
	  usage.memory = memory;
	  usage.bytes = metadata.layout->bytes_used;
	  measurements.add_measurement(usage);
	}
      }
      if(ready_event.exists())
	GenEventImpl::trigger(ready_event, false /*!poisoned*/, work_until);

      // metadata is now valid and can be shared
      NodeSet early_reqs;
      metadata.mark_valid(early_reqs);
      if(!early_reqs.empty())
        send_metadata(early_reqs);

      if(measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>()) {
	timeline.record_ready_time();
      }

    }

    void RegionInstanceImpl::notify_deallocation(void)
    {
      // response needs to be handled by the instance's creator node, so forward
      //  there if it's not us
      NodeID creator_node = ID(me).instance_creator_node();
      if(creator_node != Network::my_node_id) {
	ActiveMessage<MemStorageReleaseResponse> amsg(creator_node);
	amsg->inst = me;
	amsg.commit();

	return;
      }

      // handle race with a slow DEFERRED notification
      bool notification_delayed = false;
      {
	AutoLock<> al(mutex);
	if(metadata.need_alloc_result) {
	  metadata.need_notify_dealloc = true;
	  notification_delayed = true;
	}
      }
      if(notification_delayed) return;

      log_inst.debug() << "deallocation completed: inst=" << me;

      // our instance better not be in the unallocated state...
      assert(metadata.inst_offset != INSTOFFSET_UNALLOCATED);
      assert(metadata.inst_offset != INSTOFFSET_DELAYEDALLOC);
      assert(metadata.inst_offset != INSTOFFSET_DELAYEDDESTROY);

      // was this a successfully allocatated instance?
      if(metadata.inst_offset != INSTOFFSET_FAILED) {
	if (measurements.wants_measurement<ProfilingMeasurements::InstanceStatus>()) {
	  ProfilingMeasurements::InstanceStatus stat;
	  stat.result = ProfilingMeasurements::InstanceStatus::DESTROYED_SUCCESSFULLY;
	  stat.error_code = 0;
	  measurements.add_measurement(stat);
	}

	if (measurements.wants_measurement<ProfilingMeasurements::InstanceTimeline>()) {
	  timeline.record_delete_time();
	  measurements.add_measurement(timeline);
	}

	// send any remaining incomplete profiling responses
	measurements.send_responses(requests);
      }

      // flush the remote prefetch cache
      {
        AutoLock<> al(mutex);
        prefetch_events.clear();
      }

      // send any required invalidation messages for metadata
      bool recycle_now = metadata.initiate_cleanup(me.id);
      if(recycle_now)
        recycle_instance();
    }

    void RegionInstanceImpl::recycle_instance(void)
    {
      measurements.clear();

      MemoryImpl *m_impl = get_runtime()->get_memory_impl(memory);
      m_impl->release_instance(me);
    }

    Event RegionInstanceImpl::prefetch_metadata(NodeID target_node)
    {
      assert(target_node != Network::my_node_id);

      Event e = Event::NO_EVENT;
      {
	AutoLock<> al(mutex);
	std::map<NodeID, Event>::iterator it = prefetch_events.find(target_node);
	if(it != prefetch_events.end())
	  return it->second;

	// have to make a new one
	e = GenEventImpl::create_genevent()->current_event();
	prefetch_events.insert(std::make_pair(target_node, e));
      }

      // send a message to the target node to fetch metadata
      // TODO: save a hop by sending request to owner directly?
      ActiveMessage<InstanceMetadataPrefetchRequest> amsg(target_node);
      amsg->inst = me;
      amsg->valid_event = e;
      amsg.commit();

      return e;
    }

    /*static*/ void InstanceMetadataPrefetchRequest::handle_message(NodeID sender,
								    const InstanceMetadataPrefetchRequest& msg,
								    const void *data,
								    size_t datalen,
								    TimeLimit work_until)
    {
      // make a local request and trigger the remote event based on the local one
      RegionInstanceImpl *r_impl = get_runtime()->get_instance_impl(msg.inst);
      Event e = r_impl->request_metadata();
      log_inst.info() << "metadata prefetch: inst=" << msg.inst
		      << " local=" << e << " remote=" << msg.valid_event;

      if(e.exists()) {
	GenEventImpl *e_impl = get_runtime()->get_genevent_impl(msg.valid_event);
	EventMerger *m = &(e_impl->merger);
	m->prepare_merger(msg.valid_event, false/*!ignore faults*/, 1);
	m->add_precondition(e);
	m->arm_merger();
      } else {
	GenEventImpl::trigger(msg.valid_event, false /*!poisoned*/, work_until);
      }	
    }

    ActiveMessageHandlerReg<InstanceMetadataPrefetchRequest> inst_prefetch_msg_handler;

    bool RegionInstanceImpl::get_strided_parameters(void *&base, size_t &stride,
						      off_t field_offset)
    {
      MemoryImpl *mem = get_runtime()->get_memory_impl(memory);

      // this exists for compatibility and assumes N=1, T=long long
      const InstanceLayout<1,long long> *inst_layout = dynamic_cast<const InstanceLayout<1,long long> *>(metadata.layout);
      assert(inst_layout != 0);

      // look up the right field
      std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = inst_layout->fields.find(field_offset);
      assert(it != inst_layout->fields.end());

      // hand out a null pointer for empty instances (stride can be whatever
      //  the caller wants)
      if(inst_layout->piece_lists[it->second.list_idx].pieces.empty()) {
	base = 0;
	return true;
      }

      // also only works for a single piece
      assert(inst_layout->piece_lists[it->second.list_idx].pieces.size() == 1);
      const InstanceLayoutPiece<1,long long> *piece = inst_layout->piece_lists[it->second.list_idx].pieces[0];
      assert((piece->layout_type == PieceLayoutTypes::AffineLayoutType));
      const AffineLayoutPiece<1,long long> *affine = static_cast<const AffineLayoutPiece<1,long long> *>(piece);

      // if the caller wants a particular stride and we differ (and have more
      //  than one element), fail
      if(stride != 0) {
        if((affine->bounds.hi[0] > affine->bounds.lo[0]) &&
           (affine->strides[0] != stride))
          return false;
      } else {
        stride = affine->strides[0];
      }

      // find the offset of the first and last elements and then try to
      //  turn that into a direct memory pointer
      size_t start_offset = (metadata.inst_offset +
                             affine->offset +
                             affine->strides.dot(affine->bounds.lo) +
                             it->second.rel_offset);
      size_t total_bytes = (it->second.size_in_bytes + 
                            affine->strides[0] * (affine->bounds.hi -
                                                  affine->bounds.lo));
 
      base = mem->get_direct_ptr(start_offset, total_bytes);
      if (!base) return false;

      // now adjust the base pointer so that we can use absolute indexing
      //  again
      // careful - have to use 'stride' instead of 'affine->strides' in
      //  case we agreed to the caller's incorrect stride when size == 1
      base = ((char *)base) - (stride * affine->bounds.lo[0]);
     
      return true;
    }

    RegionInstanceImpl::Metadata::Metadata()
      : inst_offset(INSTOFFSET_UNALLOCATED)
      , ready_event(Event::NO_EVENT)
      , layout(0)
      , ext_resource(0)
      , mem_specific(0)
    {}

    void *RegionInstanceImpl::Metadata::serialize(size_t& out_size) const
    {
      Serialization::DynamicBufferSerializer dbs(128);

      bool ok = ((dbs << inst_offset) &&
		 (dbs << *layout));
      assert(ok);

      out_size = dbs.bytes_used();
      return dbs.detach_buffer(0 /*trim*/);
    }

    void RegionInstanceImpl::Metadata::deserialize(const void *in_data, size_t in_size)
    {
      Serialization::FixedBufferDeserializer fbd(in_data, in_size);

      bool ok = (fbd >> inst_offset);
      if(ok) {
	layout = InstanceLayoutGeneric::deserialize_new(fbd);
	layout->compile_lookup_program(lookup_program);
      }
      assert(ok && (layout != 0) && (fbd.bytes_left() == 0));
    }

    void RegionInstanceImpl::Metadata::do_invalidate(void)
    {
      // delete an existing layout, if present
      if(layout) {
	delete layout;
	layout = 0;
	lookup_program.reset();
      }

      if(ext_resource) {
	delete ext_resource;
	ext_resource = 0;
      }

      // clean up chain of mem-specific info
      while(mem_specific) {
        MemSpecificInfo *next = mem_specific->next;
        delete mem_specific;
        mem_specific = next;
      }

      // set the offset back to the "unallocated" value
      inst_offset = INSTOFFSET_UNALLOCATED;
    }

    void RegionInstanceImpl::Metadata::add_mem_specific(MemSpecificInfo *info)
    {
      info->next = mem_specific;
      mem_specific = info;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalInstanceResource

  ExternalInstanceResource::ExternalInstanceResource()
  {}

  ExternalInstanceResource::~ExternalInstanceResource()
  {}

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalMemoryResource

  ExternalMemoryResource::ExternalMemoryResource()
  {}
  
  ExternalMemoryResource::ExternalMemoryResource(uintptr_t _base,
						 size_t _size_in_bytes,
						 bool _read_only)
    : base(_base)
    , size_in_bytes(_size_in_bytes)
    , read_only(_read_only)
  {}

  ExternalMemoryResource::ExternalMemoryResource(void *_base,
						 size_t _size_in_bytes)
    : base(reinterpret_cast<uintptr_t>(_base))
    , size_in_bytes(_size_in_bytes)
    , read_only(false)
  {}

  ExternalMemoryResource::ExternalMemoryResource(const void *_base,
						 size_t _size_in_bytes)
    : base(reinterpret_cast<uintptr_t>(_base))
    , size_in_bytes(_size_in_bytes)
    , read_only(true)
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalMemoryResource::suggested_memory() const
  {
    // TODO: some way to ask for external memory resources on other ranks?
    CoreModule *mod = get_runtime()->get_module<CoreModule>("core");
    assert(mod);
    return mod->ext_sysmem->me;
  }

  ExternalInstanceResource *ExternalMemoryResource::clone(void) const
  {
    return new ExternalMemoryResource(base, size_in_bytes, read_only);
  }

  void ExternalMemoryResource::print(std::ostream& os) const
  {
    os << "memory(base=" << std::hex << base << std::dec;
    os << ", size=" << size_in_bytes;
    if(read_only)
      os << ", readonly";
    os << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalMemoryResource> ExternalMemoryResource::serdez_subclass;


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalFileResource

  ExternalFileResource::ExternalFileResource()
  {}
  
  ExternalFileResource::ExternalFileResource(const std::string& _filename,
					     realm_file_mode_t _mode,
					     size_t _offset /*= 0*/)
    : filename(_filename)
    , offset(_offset)
    , mode(_mode)
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalFileResource::suggested_memory() const
  {
    // TODO: support [rank=nnn] syntax here too
    Memory memory = Machine::MemoryQuery(Machine::get_machine())
      .local_address_space()
      .only_kind(Memory::FILE_MEM)
      .first();
    return memory;
  }

  ExternalInstanceResource *ExternalFileResource::clone(void) const
  {
    return new ExternalFileResource(filename, mode, offset);
  }

  void ExternalFileResource::print(std::ostream& os) const
  {
    os << "file(name='" << filename << "', mode=" << int(mode) << ", offset=" << offset << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalFileResource> ExternalFileResource::serdez_subclass;


  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, AffineLayoutPiece<N,T> > AffineLayoutPiece<N,T>::serdez_subclass;

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutGeneric, InstanceLayout<N,T> > InstanceLayout<N,T>::serdez_subclass;

#define DOIT(N,T) \
  template class AffineLayoutPiece<N,T>; \
  template class InstanceLayout<N,T>; \
  template const PieceLookup::Instruction *RegionInstance::get_lookup_program<N,T>(FieldID, unsigned, uintptr_t&); \
  template const PieceLookup::Instruction *RegionInstance::get_lookup_program<N,T>(FieldID, const Rect<N,T>&, unsigned, uintptr_t&);
  FOREACH_NT(DOIT)
#undef DOIT

}; // namespace Realm
