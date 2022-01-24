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

// dynamic node set implementation for Realm

// NOP, but useful for IDEs
#include "realm/nodeset.h"

#ifdef DEBUG_REALM_NODESET
#include <cassert>
#endif

// need std::swap
#if REALM_CXX_STANDARD >= 11
#include <utility>
#else
#include <algorithm>
#endif

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class NodeSet

  inline NodeSet::NodeSet()
    : count(0)
    , enc_format(ENC_EMPTY)
    , range_count(0)
  {}

  inline NodeSet::~NodeSet()
  {
    if((count > 0) && (enc_format == ENC_BITMASK))
      NodeSetBitmask::release_bitmask(data.bitmask, false /*!already_empty*/);
  }

  inline NodeSet::NodeSet(const NodeSet& copy_from)
    : count(copy_from.count)
    , enc_format(copy_from.enc_format)
    , range_count(copy_from.range_count)
  {
    if(count > 0) {
      if(enc_format == ENC_BITMASK)
	data.bitmask = NodeSetBitmask::clone_bitmask(copy_from.data.bitmask);
      else
	data = copy_from.data;
    }
#ifdef DEBUG_REALM_NODESET
    reference_set = copy_from.reference_set;
#endif
  }

  inline NodeSet& NodeSet::operator=(const NodeSet& copy_from)
  {
    if(this == &copy_from) return *this;

    if((count > 0) && (enc_format == ENC_BITMASK)) {
      if((copy_from.count > 0) && (copy_from.enc_format == ENC_BITMASK)) {
	data.bitmask->copy(copy_from.data.bitmask);
      } else {
	NodeSetBitmask::release_bitmask(data.bitmask, false /*!already_empty*/);
	if(copy_from.count > 0)
	  data = copy_from.data;
      }
    } else {
      if(copy_from.count > 0) {
	if(copy_from.enc_format == ENC_BITMASK)
	  data.bitmask = NodeSetBitmask::clone_bitmask(copy_from.data.bitmask);
	else
	  data = copy_from.data;
      }
    }

    count = copy_from.count;
    enc_format = copy_from.enc_format;
    range_count = copy_from.range_count;
#ifdef DEBUG_REALM_NODESET
    reference_set = copy_from.reference_set;
#endif
    return *this;
  }

  inline void NodeSet::swap(NodeSet& swap_with)
  {
    if(this == &swap_with) return;

    std::swap(count, swap_with.count);
    std::swap(enc_format, swap_with.enc_format);
    std::swap(range_count, swap_with.range_count);
    std::swap(data, swap_with.data);
#ifdef DEBUG_REALM_NODESET
    std::swap(reference_set, swap_with.reference_set);
#endif
  }

  inline bool NodeSet::empty() const
  {
#ifdef DEBUG_REALM_NODESET
    assert((count == 0) == reference_set.empty());
#endif
    return (count == 0);
  }

  inline size_t NodeSet::size() const
  {
#ifdef DEBUG_REALM_NODESET
    assert(count == reference_set.size());
#endif
    return count;
  }

  inline void NodeSet::add(NodeID id)
  {
#ifdef DEBUG_REALM_NODESET
    reference_set.insert(id);
#endif

    if(count == 0) {
      // convert to value list - populate first entry
      enc_format = ENC_VALS;
      data.values[0] = id;
      count = 1;
      return;
    }

    switch(enc_format) {
    case ENC_VALS:
      {
	// see if it's already there - if so, nothing to do
	for(short i = 0; i < short(count); i++)
	  if(data.values[i] == id)
	    return;

	// no, can we add it?
	if(count < MAX_VALUES) {
	  data.values[count++] = id;
	  return;
	}

	// if not, convert to bitmask and then add
	convert_to_bitmask();
	data.bitmask->set_bit(id); // we know the bit is not currently set
	count++;
	break;
      }

    case ENC_RANGES:
      {
	// see if it's already in a range - if so, nothing to do
	for(short i = 0; i < range_count; i++)
	  if((data.ranges[i].lo <= id) && (id <= data.ranges[i].hi))
	    return;

	// can we grow an existing range?
	for(short i = 0; i < range_count; i++) {
	  if(data.ranges[i].lo == (id + 1)) {
	    data.ranges[i].lo = id;
	    count++;
	    return;
	  }
	  if(data.ranges[i].hi == (id - 1)) {
	    data.ranges[i].hi = id;
	    count++;
	    return;
	  }
	}

	// no, can we add it?
	if(range_count < MAX_RANGES) {
	  data.ranges[range_count].lo = data.ranges[range_count].hi = id;
	  range_count++;
	  count++;
	  return;
	}

	// if not, convert to bitmask and then add
	convert_to_bitmask();
	data.bitmask->set_bit(id); // we know the bit is not currently set
	count++;
	break;
      }

    case ENC_BITMASK:
      {
	count += unsigned(data.bitmask->set_bit(id));
	break;
      }
    }
  }

  inline void NodeSet::add_range(NodeID lo, NodeID hi)
  {
    if(lo > hi) return; // empty range

#ifdef DEBUG_REALM_NODESET
    for(NodeID id = lo; id <= hi; id++)
      reference_set.insert(id);
#endif

    if(count == 0) {
      // convert to range list - populate first entry
      enc_format = ENC_RANGES;
      data.ranges[0].lo = lo;
      data.ranges[0].hi = hi;
      range_count = 1;
      count = (hi - lo + 1);
      return;
    }

    switch(enc_format) {
    case ENC_VALS:
      {
	// pessimistic check - can we add all entries assuming none are
	//  already there?
	if((count + (hi - lo + 1)) <= MAX_VALUES) {
	  // yes, now add each id that isn't already there
	  for(NodeID id = lo; id <= hi; id++) {
	    bool found = false;
	    for(short i = 0; (i < short(count)) && !found; i++)
	      found = (data.values[i] == id);
	    if(!found)
	      data.values[count++] = id;
	  }
	} else {
	  // no, switch to a bitmask
	  convert_to_bitmask();
	  count += unsigned(data.bitmask->set_range(lo, hi));
	}
	break;
      }

    case ENC_RANGES:
      {
	// first check - did we overlap with existing ranges?
	for(short i = 0; i < range_count; i++)
	  if((data.ranges[i].lo <= hi) && (lo <= data.ranges[i].hi)) {
	    if((data.ranges[i].lo <= lo) && (hi <= data.ranges[i].hi)) {
	      // fully contained - add is a nop
	    } else {
	      // partial overlap - punt to a bitmask representation
	      convert_to_bitmask();
	      count += unsigned(data.bitmask->set_range(lo, hi));
	    }
	    return;
	  }

	// next try - can we grow an existing range?
	for(short i = 0; i < range_count; i++) {
	  if(data.ranges[i].hi == (lo - 1)) {
	    data.ranges[i].hi = hi;
	    count += (hi - lo + 1);
	    return;
	  }
	  if(data.ranges[i].lo == (hi + 1)) {
	    data.ranges[i].lo = lo;
	    count += (hi - lo + 1);
	    return;
	  }
	}

	// room for another range?
	if(range_count < MAX_RANGES) {
	  data.ranges[range_count].lo = lo;
	  data.ranges[range_count].hi = hi;
	  range_count++;
	  count += (hi - lo + 1);
	  return;
	}

	// if not, convert to bitmask and then add
	convert_to_bitmask();
	count += unsigned(data.bitmask->set_range(lo, hi));
	break;
      }

    case ENC_BITMASK:
      {
	count += unsigned(data.bitmask->set_range(lo, hi));
	break;
      }
    }
  }

  inline void NodeSet::clear()
  {
    if((count > 0) && (enc_format == ENC_BITMASK))
      NodeSetBitmask::release_bitmask(data.bitmask, false /*!already_empty*/);

    count = 0;

#ifdef DEBUG_REALM_NODESET
    reference_set.clear();
#endif
  }

  inline bool NodeSet::contains(NodeID id) const
  {
    if(count == 0) {
#ifdef DEBUG_REALM_NODESET
      assert(reference_set.count(id) == 0);
#endif
      return false;
    }

    switch(enc_format) {
    case ENC_VALS:
      {
	for(short i = 0; i < short(count); i++)
	  if(data.values[i] == id) {
#ifdef DEBUG_REALM_NODESET
	    assert(reference_set.count(id) > 0);
#endif
	    return true;
	  }
	break;
      }

    case ENC_RANGES:
      {
	for(short i = 0; i < range_count; i++)
	  if((data.ranges[i].lo <= id) && (id <= data.ranges[i].hi)) {
#ifdef DEBUG_REALM_NODESET
	    assert(reference_set.count(id) > 0);
#endif
	    return true;
	  }
	break;
      }

    case ENC_BITMASK:
      {
	if(data.bitmask->is_set(id)) {
#ifdef DEBUG_REALM_NODESET
	  assert(reference_set.count(id) > 0);
#endif
	  return true;
	}
	break;
      }
    }

#ifdef DEBUG_REALM_NODESET
    assert(reference_set.count(id) == 0);
#endif
    return false;
  }

  inline NodeSet::const_iterator NodeSet::begin() const
  {
    return NodeSetIterator(*this);
  }

  inline NodeSet::const_iterator NodeSet::end() const
  {
    return NodeSetIterator();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class NodeSetIterator

  inline NodeSetIterator::NodeSetIterator()
    : nodeset(0)
    , cur_node(-1)
  {}

  inline NodeSetIterator::NodeSetIterator(const NodeSet& _nodeset)
    : nodeset(&_nodeset)
  {
    if(nodeset->count > 0) {
      switch(nodeset->enc_format) {
      case NodeSet::ENC_VALS:
	{
	  cur_node = nodeset->data.values[0];
	  iter_pos = 0;
	  break;
	}
      case NodeSet::ENC_RANGES:
	{
	  cur_node = nodeset->data.ranges[0].lo;
	  iter_pos = 0;
	  break;
	}
      case NodeSet::ENC_BITMASK:
	{
	  cur_node = nodeset->data.bitmask->first_set();
	  break;
	}
      }
    } else {
      cur_node = -1;
    }
  }

  inline bool NodeSetIterator::operator==(const NodeSetIterator& compare_to) const
  {
    if(cur_node == -1) {
      // only cur_node must match for a finished iterator
      return(compare_to.cur_node == -1);
    } else {
      // otherwise both node set and current node must match
      return((nodeset == compare_to.nodeset) &&
	     (cur_node == compare_to.cur_node));
    }
  }

  inline bool NodeSetIterator::operator!=(const NodeSetIterator& compare_to) const
  {
    if(cur_node == -1) {
      // only cur_node must match for a finished iterator
      return(compare_to.cur_node != -1);
    } else {
      // otherwise both node set and current node must match
      return((nodeset != compare_to.nodeset) ||
	     (cur_node != compare_to.cur_node));
    }
  }

  inline NodeID NodeSetIterator::operator*() const
  {
#ifdef DEBUG_REALM_NODESET
    assert(nodeset->reference_set.count(cur_node) > 0);
#endif
    return cur_node;
  }

  inline const NodeID *NodeSetIterator::operator->() const
  {
#ifdef DEBUG_REALM_NODESET
    assert(nodeset->reference_set.count(cur_node) > 0);
#endif
    return &cur_node;
  }

  inline NodeSetIterator& NodeSetIterator::operator++(/*prefix*/)
  {
    if(cur_node != -1) {
      switch(nodeset->enc_format) {
      case NodeSet::ENC_VALS:
	{
	  iter_pos++;
	  if(iter_pos < short(nodeset->count))
	    cur_node = nodeset->data.values[iter_pos];
	  else
	    cur_node = -1;
	  break;
	}
      case NodeSet::ENC_RANGES:
	{
	  if(cur_node < nodeset->data.ranges[iter_pos].hi) {
	    // step within range
	    cur_node++;
	  } else {
	    // go to next range (if it exists)
	    iter_pos++;
	    if(iter_pos < nodeset->range_count)
	      cur_node = nodeset->data.ranges[iter_pos].lo;
	    else
	      cur_node = -1;
	  }
	  break;
	}
      case NodeSet::ENC_BITMASK:
	{
	  cur_node = nodeset->data.bitmask->next_set(cur_node);
	  break;
	}
      }
    }
    return *this;
  }

  inline NodeSetIterator NodeSetIterator::operator++(int /*postfix*/)
  {
    NodeSetIterator orig = *this;
    ++(*this);
    return orig;
  }


};
