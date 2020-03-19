/* Copyright 2020 Stanford University, NVIDIA Corporation
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

// implementation of sparsity maps

#include "realm/deppart/sparsity_impl.h"

#include "realm/runtime_impl.h"
#include "realm/deppart/partitions.h"
#include "realm/deppart/deppart_config.h"
#include "realm/deppart/rectlist.h"
#include "realm/deppart/inst_helper.h"
#include "realm/logging.h"

namespace Realm {

  extern Logger log_part;
  extern Logger log_dpops;

  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMap<N,T>

  // looks up the public subset of the implementation object
  template <int N, typename T>
  SparsityMapPublicImpl<N,T> *SparsityMap<N,T>::impl(void) const
  {
    SparsityMapImplWrapper *wrapper = get_runtime()->get_sparsity_impl(*this);
    return wrapper->get_or_create<N,T>(*this);
  }

  // if 'always_create' is false and the points/rects completely fill their
  //  bounding box, returns NO_SPACE (i.e. id == 0)
  
  template <int N, typename T>
  /*static*/ SparsityMap<N,T> SparsityMap<N,T>::construct(const std::vector<Point<N,T> >& points,
							  bool always_create)
  {
    HybridRectangleList<N,T> hrl;
    for(typename std::vector<Point<N,T> >::const_iterator it = points.begin();
	it != points.end();
	++it)
      hrl.add_point(*it);
    const std::vector<Rect<N,T> >& dense = hrl.convert_to_vector();

    // are we allow to leave early for dense collections?
    if(!always_create && (dense.size() <= 1)) {
      SparsityMap<N,T> sparsity;
      sparsity.id = 0;
      return sparsity;
    }

    // construct and fill in a sparsity map
    SparsityMapImplWrapper *wrap = get_runtime()->get_available_sparsity_impl(Network::my_node_id);
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    SparsityMapImpl<N,T> *impl = wrap->get_or_create<N,T>(sparsity);
    impl->set_contributor_count(1);
    impl->contribute_dense_rect_list(dense);
    return sparsity;
  }

  template <int N, typename T>
  /*static*/ SparsityMap<N,T> SparsityMap<N,T>::construct(const std::vector<Rect<N,T> >& rects,
							  bool always_create)
  {
    HybridRectangleList<N,T> hrl;
    for(typename std::vector<Rect<N,T> >::const_iterator it = rects.begin();
	it != rects.end();
	++it)
      hrl.add_rect(*it);
    const std::vector<Rect<N,T> >& dense = hrl.convert_to_vector();

    // are we allow to leave early for dense collections?
    if(!always_create && (dense.size() <= 1)) {
      SparsityMap<N,T> sparsity;
      sparsity.id = 0;
      return sparsity;
    }

    // construct and fill in a sparsity map
    SparsityMapImplWrapper *wrap = get_runtime()->get_available_sparsity_impl(Network::my_node_id);
    SparsityMap<N,T> sparsity = wrap->me.convert<SparsityMap<N,T> >();
    SparsityMapImpl<N,T> *impl = wrap->get_or_create<N,T>(sparsity);
    impl->set_contributor_count(1);
    impl->contribute_dense_rect_list(dense);
    return sparsity;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImplWrapper

  SparsityMapImplWrapper::SparsityMapImplWrapper(void)
    : me((ID::IDType)-1), owner(-1), type_tag(0), map_impl(0)
  {}

  void SparsityMapImplWrapper::init(ID _me, unsigned _init_owner)
  {
    me = _me;
    owner = _init_owner;
  }

  template <int N, typename T>
  /*static*/ SparsityMapImpl<N,T> *SparsityMapImplWrapper::get_or_create(SparsityMap<N,T> me)
  {
    DynamicTemplates::TagType new_tag = NT_TemplateHelper::encode_tag<N,T>();
    assert(new_tag != 0);

    // try set the tag for this entry - if it's 0, we may be the first to get here
    DynamicTemplates::TagType old_tag = 0;
    if(!type_tag.compare_exchange(old_tag, new_tag)) {
      // if it was already set, it better not mismatch
      assert(old_tag == new_tag);
    }

    // now see if the pointer is valid - the validity of the old_tag is no guarantee
    void *impl = map_impl.load_acquire();
    if(impl)
      return static_cast<SparsityMapImpl<N,T> *>(impl);

    // create one and try to swap it in
    SparsityMapImpl<N,T> *new_impl = new SparsityMapImpl<N,T>(me);
    if(map_impl.compare_exchange(impl, new_impl)) {
      // ours is the winner - return it
      return new_impl;
    } else {
      // we lost the race - free the one we made and return the winner
      delete new_impl;
      return static_cast<SparsityMapImpl<N,T> *>(impl);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapPublicImpl<N,T>

  template <int N, typename T>
  SparsityMapPublicImpl<N,T>::SparsityMapPublicImpl(void)
    : entries_valid(false), approx_valid(false)
  {}

  // call actual implementation - inlining makes this cheaper than a virtual method
  template <int N, typename T>
  Event SparsityMapPublicImpl<N,T>::make_valid(bool precise /*= true*/)
  {
    return static_cast<SparsityMapImpl<N,T> *>(this)->make_valid(precise);
  }

  // membership test between two (presumably-different) sparsity maps are not
  //  cheap - try bounds-based checks first (see IndexSpace::overlaps)
  template <int N, typename T>
  bool SparsityMapPublicImpl<N,T>::overlaps(SparsityMapPublicImpl<N,T> *other,
					    const Rect<N,T>& bounds,
					    bool approx)
  {
    // full cross-product test for now - for larger rectangle lists, consider
    //  an acceleration structure?
    if(approx) {
      const std::vector<Rect<N,T> >& rects1 = get_approx_rects();
      const std::vector<Rect<N,T> >& rects2 = other->get_approx_rects();
      for(typename std::vector<Rect<N,T> >::const_iterator it1 = rects1.begin();
	  it1 != rects1.end();
	  it1++) {
	Rect<N,T> isect = it1->intersection(bounds);
	if(isect.empty())
	  continue;
	for(typename std::vector<Rect<N,T> >::const_iterator it2 = rects2.begin();
	    it2 != rects2.end();
	    it2++) {
	  if(it2->overlaps(isect))
	    return true;
	}
      }
    } else {
      const std::vector<SparsityMapEntry<N,T> >& entries1 = get_entries();
      const std::vector<SparsityMapEntry<N,T> >& entries2 = other->get_entries();
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it1 = entries1.begin();
	  it1 != entries1.end();
	  it1++) {
	Rect<N,T> isect = it1->bounds.intersection(bounds);
	if(isect.empty())
	  continue;
	for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it2 = entries2.begin();
	    it2 != entries2.end();
	    it2++) {
	  if(!it2->bounds.overlaps(isect)) continue;
	  // TODO: handle further sparsity in either side
	  assert(!it1->sparsity.exists() && (it1->bitmap == 0) &&
		 !it2->sparsity.exists() && (it2->bitmap == 0));
	  return true;
	}
      }
    }

    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImpl<N,T>

  template <int N, typename T>
  SparsityMapImpl<N,T>::SparsityMapImpl(SparsityMap<N,T> _me)
    : me(_me), remaining_contributor_count(0)
    , total_piece_count(0)
    , remaining_piece_count(0)
    , precise_requested(false), approx_requested(false)
    , precise_ready_event(Event::NO_EVENT), approx_ready_event(Event::NO_EVENT)
    , sizeof_precise(0)
  {}

  template <int N, typename T>
  inline /*static*/ SparsityMapImpl<N,T> *SparsityMapImpl<N,T>::lookup(SparsityMap<N,T> sparsity)
  {
    SparsityMapImplWrapper *wrapper = get_runtime()->get_sparsity_impl(sparsity);
    return wrapper->get_or_create<N,T>(sparsity);
  }

  // actual implementation - SparsityMapPublicImpl's version just calls this one
  template <int N, typename T>
  Event SparsityMapImpl<N,T>::make_valid(bool precise /*= true*/)
  {
    // early out
    if(precise ? this->entries_valid : this->approx_valid)
      return Event::NO_EVENT;

    // take lock to get/create event cleanly
    bool request_approx = false;
    bool request_precise = false;
    Event e = Event::NO_EVENT;
    {
      AutoLock<> al(mutex);

      if(precise) {
	if(!this->entries_valid) {
	  // do we need to request the data?
	  if((NodeID(ID(me).sparsity_creator_node()) != Network::my_node_id) && !precise_requested) {
	    request_precise = true;
	    precise_requested = true;
	    // also get approx while we're at it
	    request_approx = !(this->approx_valid || approx_requested);
	    approx_requested = true;
	    // can't use set_contributor_count, so directly set
	    //  remaining_contributor_count to the 1 contribution we expect
	    // (store is ok because we haven't sent the request that will
	    //  do the async decrement)
	    remaining_contributor_count.store(1);
	  }
	  // do we have a finish event?
	  if(precise_ready_event.exists()) {
	    e = precise_ready_event;
	  } else {
	    e = GenEventImpl::create_genevent()->current_event();
	    precise_ready_event = e;
	  }
	}
      } else {
	if(!this->approx_valid) {
	  // do we need to request the data?
	  if((NodeID(ID(me).sparsity_creator_node()) != Network::my_node_id) && !approx_requested) {
	    request_approx = true;
	    approx_requested = true;
	  }
	  // do we have a finish event?
	  if(approx_ready_event.exists()) {
	    e = approx_ready_event;
	  } else {
	    e = GenEventImpl::create_genevent()->current_event();
	    approx_ready_event = e;
	  }
	}
      }
    }
    
    if(request_approx || request_precise) {
      ActiveMessage<RemoteSparsityRequest> amsg(ID(me).sparsity_creator_node(), 0);
      amsg->sparsity = me;
      amsg->send_precise = request_precise;
      amsg->send_approx = request_approx;
      amsg.commit();
    }

    return e;
  }


  // methods used in the population of a sparsity map

  // when we plan out a partitioning operation, we'll know how many
  //  different uops are going to contribute something (or nothing) to
  //  the sparsity map - once all of those contributions arrive, we can
  //  finalize the sparsity map
  template <int N, typename T>
  void SparsityMapImpl<N,T>::set_contributor_count(int count)
  {
    if(NodeID(ID(me).sparsity_creator_node()) == Network::my_node_id) {
      // increment the count atomically - if it brings the total up to 0
      //  (which covers count == 0), immediately propagate the total piece
      //  count and finalize if needed - the contributions (or at least the
      //  portions that know how many pieces there were) happened before we
      //  got here
      int v = remaining_contributor_count.fetch_add(count) + count;
      if(v == 0) {
	int pcount = total_piece_count.load();
	bool have_all_pieces = ((pcount == 0) ||
				((remaining_piece_count.fetch_add(pcount) + pcount) == 0));
	if(have_all_pieces)
	  finalize();
      }
    } else {
      // send the contributor count to the owner node
      ActiveMessage<SetContribCountMessage> amsg(ID(me).sparsity_creator_node(), 0);
      amsg->sparsity = me;
      amsg->count = count;
      amsg.commit();
    }
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::contribute_nothing(void)
  {
    NodeID owner = ID(me).sparsity_creator_node();

    if(owner != Network::my_node_id) {
      // send (the lack of) data to the owner to collect
      ActiveMessage<RemoteSparsityContrib> amsg(owner, 0);
      amsg->sparsity = me;
      amsg->piece_count = 1;
      amsg.commit();

      return;
    }

    // count is allowed to go negative if we get contributions before we know the total expected
    int left = remaining_contributor_count.fetch_sub(1) - 1;
    if(left == 0) {
      // we didn't have anything to contribute to the total piece count, but
      //  it's our job to incorporate it into the remaining piece count and
      //  finalize if needed
      int pcount = total_piece_count.load();
      bool have_all_pieces = ((pcount == 0) ||
			      ((remaining_piece_count.fetch_add(pcount) + pcount) == 0));
      if(have_all_pieces)
	finalize();
    }
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::contribute_dense_rect_list(const std::vector<Rect<N,T> >& rects)
  {
    NodeID owner = ID(me).sparsity_creator_node();

    if(owner != Network::my_node_id) {
      // send the data to the owner to collect
      const size_t max_to_send = DeppartConfig::cfg_max_bytes_per_packet / sizeof(Rect<N,T>);
      const Rect<N,T> *rdata = (rects.empty() ? 0 : &rects[0]);
      size_t num_pieces = 0;
      size_t remaining = rects.size();
      // send partial messages first
      while(remaining > max_to_send) {
	size_t bytes = max_to_send * sizeof(Rect<N,T>);
	ActiveMessage<RemoteSparsityContrib> amsg(owner, bytes);
	amsg->sparsity = me;
	amsg->piece_count = 0;
	amsg.add_payload(rdata, bytes, PAYLOAD_COPY);
	amsg.commit();

	num_pieces++;
	remaining -= max_to_send;
	rdata += max_to_send;
      }

      // final message includes the count of all messages (including this one!)
      size_t bytes = remaining * sizeof(Rect<N,T>);
      ActiveMessage<RemoteSparsityContrib> amsg(owner, bytes);
      amsg->sparsity = me;
      amsg->piece_count = num_pieces + 1;
      amsg.add_payload(rdata, bytes, PAYLOAD_COPY);
      amsg.commit();

      return;
    }

    // local contribution is done as a single piece
    contribute_raw_rects((rects.empty() ? 0 : &rects[0]), rects.size(), 1);
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::contribute_raw_rects(const Rect<N,T>* rects,
						  size_t count,
						  size_t piece_count)
  {
    if(count > 0) {
      AutoLock<> al(mutex);

      if(N == 1) {
	// demand that our input data is sorted
	for(size_t i = 1; i < count; i++)
	  assert(rects[i-1].hi.x < (rects[i].lo.x - 1));

	// fast case - all these rectangles are after all the ones we have now
	if(this->entries.empty() || (this->entries.rbegin()->bounds.hi.x < rects[0].lo.x)) {
	  // special case when merging occurs with the last entry from before
	  size_t n = this->entries.size();
	  if((n > 0) && (this->entries.rbegin()->bounds.hi.x == (rects[0].lo.x - 1))) {
	    this->entries.resize(n + count - 1);
	    assert(!this->entries[n - 1].sparsity.exists());
	    assert(this->entries[n - 1].bitmap == 0);
	    this->entries[n - 1].bounds.hi = rects[0].hi;
	    for(size_t i = 1; i < count; i++) {
	      this->entries[n - 1 + i].bounds = rects[i];
	      this->entries[n - 1 + i].sparsity.id = 0; // no sparsity map
	      this->entries[n - 1 + i].bitmap = 0;
	    }
	  } else {
	    this->entries.resize(n + count);
	    for(size_t i = 0; i < count; i++) {
	      this->entries[n + i].bounds = rects[i];
	      this->entries[n + i].sparsity.id = 0; // no sparsity map
	      this->entries[n + i].bitmap = 0;
	    }
	  }
	} else {
	  // do a merge of the new data with the old
	  std::vector<SparsityMapEntry<N,T> > old_data;
	  old_data.swap(this->entries);
	  size_t i = 0;
	  size_t n = 0;
	  typename std::vector<SparsityMapEntry<N,T> >::const_iterator old_it = old_data.begin();
	  while((i < count) && (old_it != old_data.end())) {
	    if(rects[i].hi.x < (old_it->bounds.lo.x - 1)) {
	      this->entries.resize(n + 1);
	      this->entries[n].bounds = rects[i];
	      this->entries[n].sparsity.id = 0; // no sparsity map
	      this->entries[n].bitmap = 0;
	      n++;
	      i++;
	      continue;
	    }

	    if(old_it->bounds.hi.x < (rects[i].lo.x - 1)) {
	      this->entries.push_back(*old_it);
	      n++;
	      old_it++;
	      continue;
	    }

	    Rect<N,T> u = rects[i].union_bbox(old_it->bounds);
	    // step rects, but not old_it - want sanity checks below to be done
	    i++;
	    while(true) {
	      if((i < count) && (rects[i].lo.x <= (u.hi.x + 1))) {
		u.hi.x = std::max(u.hi.x, rects[i].hi.x);
		i++;
		continue;
	      }
	      if((old_it != old_data.end()) && (old_it->bounds.lo.x <= (u.hi.x + 1))) {
		assert(!old_it->sparsity.exists());
		assert(old_it->bitmap == 0);
		u.hi.x = std::max(u.hi.x, old_it->bounds.hi.x);
		old_it++;
		continue;
	      }
	      // if neither test passed, the chain is broken
	      break;
	    }
	    this->entries.resize(n + 1);
	    this->entries[n].bounds = u;
	    this->entries[n].sparsity.id = 0; // no sparsity map
	    this->entries[n].bitmap = 0;
	    n++;
	  }

	  // leftovers...
	  while(i < count) {
	    this->entries.resize(n + 1);
	    this->entries[n].bounds = rects[i];
	    this->entries[n].sparsity.id = 0; // no sparsity map
	    this->entries[n].bitmap = 0;
	    n++;
	    i++;
	  }

	  while(old_it != old_data.end()) {
	    this->entries.push_back(*old_it);
	    old_it++;
	  }
	}
      } else {
	// each new rectangle has to be tested against existing ones for containment, overlap,
	//  or mergeability
	// can't use iterators on entry list, since push_back invalidates end()
	size_t orig_count = this->entries.size();

	for(size_t i = 0; i < count; i++) {
	  const Rect<N,T>& r = rects[i];

	  // index is declared outside for loop so we can detect early exits
	  size_t idx;
	  for(idx = 0; idx < orig_count; idx++) {
	    SparsityMapEntry<N,T>& e = this->entries[idx];
	    if(e.bounds.contains(r)) {
	      // existing entry contains us - still three cases though
	      if(e.sparsity.exists()) {
		assert(0);
	      } else if(e.bitmap != 0) {
		assert(0);
	      } else {
		// dense entry containing new one - nothing to do
		break;
	      }
	    }
	    if(e.bounds.overlaps(r)) {
	      assert(0);
	      break;
	    }
	    // only worth merging against a dense rectangle
	    if(can_merge(e.bounds, r) && !e.sparsity.exists() && (e.bitmap == 0)) {
	      e.bounds = e.bounds.union_bbox(r);
	      break;
	    }
	  }
	  if(idx == orig_count) {
	    // no matches against existing stuff, so add a new entry
	    idx = this->entries.size();
	    this->entries.resize(idx + 1);
	    this->entries[idx].bounds = r;
	    this->entries[idx].sparsity.id = 0; //SparsityMap<N,T>::NO_SPACE;
	    this->entries[idx].bitmap = 0;
	  }
	}
      }
    }

    bool have_all_pieces = false;
    if(piece_count > 0) {
      // this is the last (or only) piece of a contribution, so add our
      //  piece count to the total piece count across all contributions and
      //  then see if we are the final contributor
      total_piece_count.fetch_add(piece_count);

      bool last_contrib = (remaining_contributor_count.fetch_sub(1) == 1);
      if(last_contrib) {
	// refetch the total piece count and add it to the remaining count,
	//  minus one for this piece we're doing now - if the count comes out
	//  to zero, we're the last count
	int pcount = total_piece_count.load() - 1;
	have_all_pieces = ((pcount == 0) ||
			   ((remaining_piece_count.fetch_add(pcount) + pcount) == 0));
      } else {
	// decrement remaining count for our piece - if count goes to zero,
	//  we were the last piece
	have_all_pieces = (remaining_piece_count.fetch_sub(1) == 1);
      }
    } else {
      // decrement remaining count for our piece - if count goes to zero,
      //  we were the last piece
      have_all_pieces = (remaining_piece_count.fetch_sub(1) == 1);
    }
    
    if(have_all_pieces) {
      if(NodeID(ID(me).sparsity_creator_node()) != Network::my_node_id) {
	// this is a remote sparsity map, so sanity check that we requested the data
	assert(precise_requested);
      }

      finalize();
    }
  }

  // adds a microop as a waiter for valid sparsity map data - returns true
  //  if the uop is added to the list (i.e. will be getting a callback at some point),
  //  or false if the sparsity map became valid before this call (i.e. no callback)
  template <int N, typename T>
  bool SparsityMapImpl<N,T>::add_waiter(PartitioningMicroOp *uop, bool precise)
  {
    // early out
    if(precise ? this->entries_valid : this->approx_valid)
      return false;

    // take lock and retest, and register if not ready
    bool registered = false;
    bool request_approx = false;
    bool request_precise = false;
    {
      AutoLock<> al(mutex);

      if(precise) {
	if(!this->entries_valid) {
	  precise_waiters.push_back(uop);
	  registered = true;
	  // do we need to request the data?
	  if((NodeID(ID(me).sparsity_creator_node()) != Network::my_node_id) && !precise_requested) {
	    request_precise = true;
	    precise_requested = true;
	    // also get approx while we're at it
	    request_approx = !(this->approx_valid || approx_requested);
	    approx_requested = true;
	    // can't use set_contributor_count, so directly set
	    //  remaining_contributor_count to the 1 contribution we expect
	    // (store is ok because we haven't sent the request that will
	    //  do the async decrement)
	    remaining_contributor_count.store(1);
	  }
	}
      } else {
	if(!this->approx_valid) {
	  approx_waiters.push_back(uop);
	  registered = true;
	  // do we need to request the data?
	  if((NodeID(ID(me).sparsity_creator_node()) != Network::my_node_id) && !approx_requested) {
	    request_approx = true;
	    approx_requested = true;
	  }
	}
      }
    }

    if(request_approx || request_precise) {
      ActiveMessage<RemoteSparsityRequest> amsg(ID(me).sparsity_creator_node(), 0);
      amsg->sparsity = me;
      amsg->send_precise = request_precise;
      amsg->send_approx = request_approx;
      amsg.commit();
    }

    return registered;
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::remote_data_request(NodeID requestor, bool send_precise, bool send_approx)
  {
    // first sanity check - we should be the owner of the data
    assert(NodeID(ID(me).sparsity_creator_node()) == Network::my_node_id);

    // take the long to determine atomically if we can send data or if we need to register as a listener
    bool reply_precise = false;
    bool reply_approx = false;
    {
      AutoLock<> al(mutex);

      // always add the requestor to the sharer list
      remote_sharers.add(requestor);

      if(send_precise) {
	if(this->entries_valid)
	  reply_precise = true;
	else
	  remote_precise_waiters.add(requestor);
      }

      if(send_approx) {
	if(this->approx_valid)
	  reply_approx = true;
	else
	  remote_approx_waiters.add(requestor);
      }
    }

    if(reply_approx || reply_precise)
      remote_data_reply(requestor, reply_precise, reply_approx);
  }
  
  template <int N, typename T>
  void SparsityMapImpl<N,T>::remote_data_reply(NodeID requestor, bool reply_precise, bool reply_approx)
  {
    if(reply_approx) {
      // TODO
      assert(this->approx_valid);
    }

    if(reply_precise) {
      log_part.info() << "sending precise data: sparsity=" << me << " target=" << requestor;
      
      // scan the entry list, sending bitmaps first and making a list of rects
      std::vector<Rect<N,T> > rects;
      for(typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = this->entries.begin();
	  it != this->entries.end();
	  it++) {
	if(it->bitmap) {
	  // TODO: send bitmap
	  assert(0);
	}
	else if(it->sparsity.exists()) {
	  // TODO: ?
	  assert(0);
	} else {
	  rects.push_back(it->bounds);
	}
      }
	
      const Rect<N,T> *rdata = &rects[0];
      size_t remaining = rects.size();
      const size_t max_to_send = DeppartConfig::cfg_max_bytes_per_packet / sizeof(Rect<N,T>);
      size_t num_pieces = 0;
      // send partial messages first
      while(remaining > max_to_send) {
	size_t bytes = max_to_send * sizeof(Rect<N,T>);
	ActiveMessage<RemoteSparsityContrib> amsg(requestor, bytes);
	amsg->sparsity = me;
	amsg->piece_count = 0;
	amsg.add_payload(rdata, bytes, PAYLOAD_COPY);
	amsg.commit();

	num_pieces++;
	remaining -= max_to_send;
	rdata += max_to_send;
      }

      // final message includes the count of all messages (including this one!)
      size_t bytes = remaining * sizeof(Rect<N,T>);
      ActiveMessage<RemoteSparsityContrib> amsg(requestor, bytes);
      amsg->sparsity = me;
      amsg->piece_count = num_pieces + 1;
      amsg.add_payload(rdata, bytes, PAYLOAD_COPY);
      amsg.commit();
    }
  }
  
  template <int N, typename T>
  static inline bool non_overlapping_bounds_1d_comp(const SparsityMapEntry<N,T>& lhs,
						    const SparsityMapEntry<N,T>& rhs)
  {
    return lhs.bounds.lo.x < rhs.bounds.lo.x;
  }

  template <int N, typename T>
  static void compute_approximation(const std::vector<SparsityMapEntry<N,T> >& entries,
				    std::vector<Rect<N,T> >& approx_rects,
				    int max_rects)
  {
    size_t n = entries.size();
    // ignore max rects for now and just copy bounds over
    if(n <= (size_t)max_rects) {
      approx_rects.resize(n);
      for(size_t i = 0; i < n; i++)
	approx_rects[i] = entries[i].bounds;
      return;
    }

    // TODO: partial k-d tree?
    // for now, just approximate with the bounding box
    Rect<N,T> bbox = entries[0].bounds;
    for(size_t i = 1; i < n; i++)
      bbox = bbox.union_bbox(entries[i].bounds);
    approx_rects.resize(1);
    approx_rects[0] = bbox;
  }

  template <typename T>
  static void compute_approximation(const std::vector<SparsityMapEntry<1,T> >& entries,
				    std::vector<Rect<1,T> >& approx_rects,
				    int max_rects)
  {
    int n = entries.size();
    // if we have few enough entries, just copy things over
    if(n <= max_rects) {
      approx_rects.resize(n);
      for(int i = 0; i < n; i++)
	approx_rects[i] = entries[i].bounds;
      return;
    }

    // if not, then do a scan through the entries and remember the max_rects-1 largest gaps - those
    //  are the ones we'll keep
    std::vector<T> gap_sizes(max_rects - 1, 0);
    std::vector<int> gap_idxs(max_rects - 1, -1);
    for(int i = 1; i < n; i++) {
      T gap = entries[i].bounds.lo.x - entries[i - 1].bounds.hi.x;
      if(gap <= gap_sizes[0])
	continue;
      // the smallest gap is discarded and we insertion-sort this new value in
      int j = 0;
      while((j < (max_rects - 2) && (gap > gap_sizes[j+1]))) {
	gap_sizes[j] = gap_sizes[j+1];
	gap_idxs[j] = gap_idxs[j+1];
	j++;
      }
      gap_sizes[j] = gap;
      gap_idxs[j] = i;
    }
    // std::cout << "[[[";
    // for(size_t i = 0; i < gap_sizes.size(); i++)
    //   std::cout << " " << gap_idxs[i] << "=" << gap_sizes[i] << ":" << entries[gap_idxs[i]-1].bounds << "," << entries[gap_idxs[i]].bounds;
    // std::cout << " ]]]\n";
    // now just sort the gap indices so we can emit the right rectangles
    std::sort(gap_idxs.begin(), gap_idxs.end());
    approx_rects.resize(max_rects);
    approx_rects[0].lo = entries[0].bounds.lo;
    for(int i = 0; i < max_rects - 1; i++) {
      approx_rects[i].hi = entries[gap_idxs[i] - 1].bounds.hi;
      approx_rects[i+1].lo = entries[gap_idxs[i]].bounds.lo;
    }
    approx_rects[max_rects - 1].hi = entries[n - 1].bounds.hi;
    // std::cout << "[[[";
    // for(size_t i = 0; i < approx_rects.size(); i++)
    //   std::cout << " " << approx_rects[i];
    // std::cout << " ]]]\n";
  }

  template <int N, typename T>
  void SparsityMapImpl<N,T>::finalize(void)
  {
    {
      LoggerMessage msg = log_part.info();
      if(msg.is_active()) {
	msg << "finalizing " << me << "(" << this << "), " << this->entries.size() << " entries";
	for(size_t i = 0; i < this->entries.size(); i++)
	  msg << "\n  [" << i
	      << "]: bounds=" << this->entries[i].bounds
	      << " sparsity=" << this->entries[i].sparsity
	      << " bitmap=" << this->entries[i].bitmap;
      }
    }

    // first step is to organize the data a little better - for N=1, this means sorting
    //  the entries list
    if(N == 1) {
      std::sort(this->entries.begin(), this->entries.end(), non_overlapping_bounds_1d_comp<N,T>);
      for(size_t i = 1; i < this->entries.size(); i++)
	assert(this->entries[i-1].bounds.hi.x < (this->entries[i].bounds.lo.x - 1));
    }

    // now that we've got our entries nice and tidy, build a bounded approximation of them
    if(true /*ID(me).sparsity_creator_node() == Network::my_node_id*/) {
      assert(!this->approx_valid);
      compute_approximation(this->entries, this->approx_rects, DeppartConfig::cfg_max_rects_in_approximation);
      this->approx_valid = true;
    }

#ifdef DEBUG_PARTITIONING
    std::cout << "finalizing " << this << ", " << this->entries.size() << " entries" << std::endl;
    for(size_t i = 0; i < this->entries.size(); i++)
      std::cout << "  [" << i
		<< "]: bounds=" << this->entries[i].bounds
		<< " sparsity=" << this->entries[i].sparsity
		<< " bitmap=" << this->entries[i].bitmap
		<< std::endl;
#endif

    NodeSet sendto_precise, sendto_approx;
    Event trigger_precise = Event::NO_EVENT;
    Event trigger_approx = Event::NO_EVENT;
    std::vector<PartitioningMicroOp *> precise_waiters_copy, approx_waiters_copy;
    {
      AutoLock<> al(mutex);

      assert(!this->entries_valid);
      this->entries_valid = true;
      precise_requested = false;
      if(precise_ready_event.exists()) {
	trigger_precise = precise_ready_event;
	precise_ready_event = Event::NO_EVENT;
      }

      precise_waiters_copy.swap(precise_waiters);
      approx_waiters_copy.swap(approx_waiters);

      sendto_precise = remote_precise_waiters;
      remote_precise_waiters.clear();
    }

    for(std::vector<PartitioningMicroOp *>::const_iterator it = precise_waiters_copy.begin();
	it != precise_waiters_copy.end();
	it++)
      (*it)->sparsity_map_ready(this, true);

    for(std::vector<PartitioningMicroOp *>::const_iterator it = approx_waiters_copy.begin();
	it != approx_waiters_copy.end();
	it++)
      (*it)->sparsity_map_ready(this, false);

    if(!sendto_approx.empty()) {
      for(NodeID i = 0; (i <= Network::max_node_id) && !sendto_approx.empty(); i++)
	if(sendto_approx.contains(i)) {
	  bool also_precise = sendto_precise.contains(i);
	  if(also_precise)
	    sendto_precise.remove(i);
	  remote_data_reply(i, also_precise, true);
	  sendto_approx.remove(i);
	}
    }

    if(!sendto_precise.empty()) {
      for(NodeID i = 0; (i <= Network::max_node_id) && !sendto_precise.empty(); i++)
	if(sendto_precise.contains(i)) {
	  remote_data_reply(i, true, false);
	  sendto_precise.remove(i);
	}
    }

    if(trigger_approx.exists())
      GenEventImpl::trigger(trigger_approx, false /*!poisoned*/);

    if(trigger_precise.exists())
      GenEventImpl::trigger(trigger_precise, false /*!poisoned*/);
  }

  template <int N, typename T>
  /*static*/ ActiveMessageHandlerReg<typename SparsityMapImpl<N,T>::RemoteSparsityRequest> SparsityMapImpl<N,T>::remote_sparsity_request_reg;
  template <int N, typename T>
  /*static*/ ActiveMessageHandlerReg<typename SparsityMapImpl<N,T>::RemoteSparsityContrib> SparsityMapImpl<N,T>::remote_sparsity_contrib_reg;
  template <int N, typename T>
  /*static*/ ActiveMessageHandlerReg<typename SparsityMapImpl<N,T>::SetContribCountMessage> SparsityMapImpl<N,T>::set_contrib_count_msg_reg;


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImpl<N,T>::RemoteSparsityRequest

  template <int N, typename T>
  inline /*static*/ void SparsityMapImpl<N,T>::RemoteSparsityRequest::handle_message(NodeID sender,
										     const SparsityMapImpl<N,T>::RemoteSparsityRequest &msg,
										     const void *data, size_t datalen)
  {
    log_part.info() << "received sparsity request: sparsity=" << msg.sparsity << " precise=" << msg.send_precise << " approx=" << msg.send_approx;
    SparsityMapImpl<N,T>::lookup(msg.sparsity)->remote_data_request(sender,
								    msg.send_precise,
								    msg.send_approx);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImpl<N,T>::RemoteSparsityContrib

  template <int N, typename T>
  inline /*static*/ void SparsityMapImpl<N,T>::RemoteSparsityContrib::handle_message(NodeID sender,
										     const SparsityMapImpl<N,T>::RemoteSparsityContrib &msg,
										     const void *data, size_t datalen)
  {
    log_part.info() << "received remote contribution: sparsity=" << msg.sparsity << " len=" << datalen;
    size_t count = datalen / sizeof(Rect<N,T>);
    assert((datalen % sizeof(Rect<N,T>)) == 0);
    SparsityMapImpl<N,T>::lookup(msg.sparsity)->contribute_raw_rects((const Rect<N,T> *)data,
								     count,
								     msg.piece_count);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class SparsityMapImpl<N,T>::SetContribCountMessage

  template <int N, typename T>
  inline /*static*/ void SparsityMapImpl<N,T>::SetContribCountMessage::handle_message(NodeID sender,
										     const SparsityMapImpl<N,T>::SetContribCountMessage &msg,
										     const void *data, size_t datalen)
  {
    log_part.info() << "received contributor count: sparsity=" << msg.sparsity << " count=" << msg.count;
    SparsityMapImpl<N,T>::lookup(msg.sparsity)->set_contributor_count(msg.count);
  }

#define DOIT(N,T) \
  template class SparsityMapPublicImpl<N,T>; \
  template class SparsityMapImpl<N,T>; \
  template class SparsityMap<N,T>;
  FOREACH_NT(DOIT)

}; // namespace Realm
