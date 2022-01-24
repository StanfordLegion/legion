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

// rectangle lists for Realm partitioning

#ifndef REALM_DEPPART_RECTLIST_INL
#define REALM_DEPPART_RECTLIST_INL

#include "realm/deppart/rectlist.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CoverageCounter<N,T>

  template <int N, typename T>
  inline CoverageCounter<N,T>::CoverageCounter(void)
    : count(0)
  {}

  template <int N, typename T>
  inline void CoverageCounter<N,T>::add_point(const Point<N,T>& p)
  {
    count++;
  }

  template <int N, typename T>
  inline void CoverageCounter<N,T>::add_rect(const Rect<N,T>& r)
  {
    count += r.volume();
  }

  template <int N, typename T>
  inline size_t CoverageCounter<N,T>::get_count(void) const
  {
    return count;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class DenseRectangleList<N,T>

  template <int N, typename T>
  inline DenseRectangleList<N,T>::DenseRectangleList(size_t _max_rects /*= 0*/)
    : max_rects(_max_rects)
    , merge_dim(-1)
  {}

  template <int N, typename T>
  inline void DenseRectangleList<N,T>::merge_rects(size_t upper_bound)
  {
    assert(upper_bound > 0);
    while(rects.size() > upper_bound) {
      // scan the rectangles to decide which to merge - want the smallest gap
      size_t best_idx = 0;
      T best_gap = rects[1].lo.x - rects[0].hi.x;
      for(size_t i = 1; i < max_rects; i++) {
	T gap = rects[i + 1].lo.x - rects[i].hi.x;
	if(gap < best_gap) {
	  best_gap = gap;
	  best_idx = i;
	}
      }
      //std::cout << "merging " << rects[best_idx] << " and " << rects[best_idx + 1] << "\n";
      rects[best_idx].hi.x = rects[best_idx + 1].hi.x;
      rects.erase(rects.begin() + best_idx + 1);
    }
  }

  template <int N, typename T>
  inline void DenseRectangleList<N,T>::add_point(const Point<N,T>& p)
  {
    if(rects.empty()) {
      rects.push_back(Rect<N,T>(p, p));
      return;
    }

    if(N == 1) {
      // optimize for sorted insertion (i.e. stuff at end)
      {
	Rect<N,T> &lr = *rects.rbegin();
	if(p.x == (lr.hi.x + 1)) {
	  lr.hi.x = p.x;
	  return;
	}
	if(p.x > (lr.hi.x + 1)) {
	  rects.push_back(Rect<N,T>(p, p));
	  if((max_rects > 0) && (rects.size() > (size_t)max_rects)) {
	    //std::cout << "too big " << rects.size() << " > " << max_rects << "\n";
	    merge_rects(max_rects);
	  }
	  return;
	}
      }

      // maintain sorted order, even at the cost of copying stuff (for lists
      //  that will get big and aren't sorted well (e.g. images), the HybridRectangleList
      //  is a better choice)

      // std::cout << "{{";
      // for(size_t i = 0; i < rects.size(); i++) std::cout << " " << rects[i];
      // std::cout << " }} <- " << p << "\n";
      // binary search to find the rectangles above and below our point
      int lo = 0;
      int hi = rects.size();
      while(lo < hi) {
	int mid = (lo + hi) >> 1;
	if(p.x < rects[mid].lo.x)
	  hi = mid;
	else if(p.x > rects[mid].hi.x)
	  lo = mid + 1;
	else {
	  // we landed right on an existing rectangle - we're done
	  // std::cout << "{{";
	  // for(size_t i = 0; i < rects.size(); i++) std::cout << " " << rects[i];
	  // std::cout << " }} INCLUDED\n";
	  return;
	}
      }
      // when we get here, 'lo' is the first rectangle above us, so check for a merge below first
      if((lo > 0) && (rects[lo - 1].hi.x == (p.x - 1))) {
	// merging low
	if((lo < (int)rects.size()) && rects[lo].lo.x == (p.x + 1)) {
	  // merging high too
	  rects[lo - 1].hi.x = rects[lo].hi.x;
	  rects.erase(rects.begin() + lo);
	} else {
	  // just low
	  rects[lo - 1].hi.x = p.x;
	}
      } else {
	if((lo < (int)rects.size()) && rects[lo].lo.x == (p.x + 1)) {
	  // merging just high
	  rects[lo].lo.x = p.x;
	} else {
	  // no merge - must insert
	  rects.insert(rects.begin() + lo, Rect<N,T>(p, p));
	  if((max_rects > 0) && (rects.size() > (size_t)max_rects)) {
	    //std::cout << "too big " << rects.size() << " > " << max_rects << "\n";
	    merge_rects(max_rects);
	  }
	}
      }
      // std::cout << "{{";
      // for(size_t i = 0; i < rects.size(); i++) std::cout << " " << rects[i];
      // std::cout << " }}\n";
    } else {
      // just treat it as a small rectangle
      add_rect(Rect<N,T>(p,p));
    }
  }

  template <int N, typename T>
  inline bool can_merge(const Rect<N,T>& r1, const Rect<N,T>& r2)
  {
    // N-1 dimensions must match exactly and 1 may be adjacent
    int idx = 0;
    while((idx < N) && (r1.lo[idx] == r2.lo[idx]) && (r1.hi[idx] == r2.hi[idx]))
      idx++;

    // if we get all the way through, the rectangles are equal and can be "merged"
    if(idx >= N) return true;

    // if not, this has to be the dimension that is adjacent or overlaps
    if(((r1.hi[idx] + 1) < r2.lo[idx]) ||
       ((r2.hi[idx] + 1) < r1.lo[idx]))
      return false;

    // and the rest of the dimensions have to match too
    while(++idx < N)
      if((r1.lo[idx] != r2.lo[idx]) || (r1.hi[idx] != r2.hi[idx]))
	return false;

    return true;
  }

#ifdef REALM_DEBUG_RECT_MERGING
  extern Logger log_part;

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const std::vector<Rect<N,T> >& v)
  {
    if(v.empty()) {
      os << "[]";
    } else {
      os << "[ ";
      typename std::vector<Rect<N,T> >::const_iterator it = v.begin();
      os << *it;
      while(++it != v.end())
	os << ", " << *it;
      os << " ]";
    }
    return os;
  }
#endif

  template <int N, typename T>
  inline void DenseRectangleList<N,T>::add_rect(const Rect<N,T>& _r)
  {
    // never add an empty rectangle
    if(_r.empty())
      return;

    if(rects.empty()) {
      rects.push_back(_r);
      return;
    }

    if(N == 1) {
      // try to optimize for sorted insertion (i.e. stuff at end)
      Rect<N,T> &lr = *rects.rbegin();
      if(_r.lo.x == (lr.hi.x + 1)) {
	lr.hi.x = _r.hi.x;
	return;
      }
      if(_r.lo.x > (lr.hi.x + 1)) {
	rects.push_back(_r);
	if((max_rects > 0) && (rects.size() > (size_t)max_rects)) {
          merge_rects(max_rects);
	}
	return;
      }

      // maintain sorted order, even at the cost of copying stuff (for lists
      //  that will get big and aren't sorted well (e.g. images), the HybridRectangleList
      //  is a better choice)
      // use a binary search to skip over all rectangles that are strictly
      //  below the new rectangle (i.e. all r s.t. r.hi.x + 1 < _r.lo.x)
      int lo = 0;
      int hi = rects.size();
      while(lo < hi) {
	int mid = (lo + hi) >> 1;
	if(rects[mid].hi.x + 1 < _r.lo.x)
	  lo = mid + 1;
	else
	  hi = mid;
      }
      // because of the early out tests above, 'lo' should always point at a
      //  valid index
      assert(lo < (int)rects.size());
      Rect<N,T> &mr = rects[lo];

      // if the new rect fits entirely below the existing one, insert the new
      //  one here and we're done
      if(_r.hi.x + 1 < mr.lo.x) {
	rects.insert(rects.begin()+lo, _r);
	return;
      }

      // last case: merge _r with mr and possibly rects above it
      assert(can_merge(_r, mr));
      mr = mr.union_bbox(_r);
      int dlo = lo + 1;
      int dhi = dlo;
      while((dhi < (int)rects.size()) &&
	    ((mr.hi.x + 1) >= rects[dhi].lo.x)) {
	mr.hi.x = std::max(mr.hi.x, rects[dhi].hi.x);
	dhi++;
      }
      if(dhi > dlo)
	rects.erase(rects.begin()+dlo, rects.begin()+dhi);
      return;
    }

    // for 2+ dimensions, there's no easy way to keep the data sorted, so we
    //  don't bother - we'll try to extend the most-recently-added rectangle
    //  if that works
#ifdef __PGI
    // suppress "dynamic initialization in unreachable code" warning
#pragma diag_suppress initialization_not_reachable
#endif

    {
      Rect<N,T>& last = rects[rects.size() - 1];

      if(merge_dim == -1) {
        // no merging has occurred, so we're free to pick any dimension that is
        //  possible
        int candidate_dim = 0;
        int matching_dims = 0;
        for(int i = 0; i < N; i++)
          if((last.lo[i] == _r.lo[i]) && (last.hi[i] == _r.hi[i]))
            matching_dims++;
          else
            candidate_dim += i;  // if more than one adds here, matching count will be wrong
        if(matching_dims == (N - 1)) {
          if((last.hi[candidate_dim] + 1) == _r.lo[candidate_dim]) {
            merge_dim = candidate_dim;
            last.hi[candidate_dim] = _r.hi[candidate_dim];
            return;
          }
          if((_r.hi[candidate_dim] + 1) == last.lo[candidate_dim]) {
            merge_dim = candidate_dim;
            last.lo[candidate_dim] = _r.lo[candidate_dim];
            return;
          }
        }
      } else {
        // only merge in the same dimension as previous merges (and only if
        //  possible)
        bool ok = true;
        for(int i = 0; i < N; i++)
          if((i != merge_dim) &&
             ((last.lo[i] != _r.lo[i]) || (last.hi[i] != _r.hi[i]))) {
            ok = false;
            break;
          }
        if(ok) {
          if((last.hi[merge_dim] + 1) == _r.lo[merge_dim]) {
            last.hi[merge_dim] = _r.hi[merge_dim];
            return;
          }
          if((_r.hi[merge_dim] + 1) == last.lo[merge_dim]) {
            last.lo[merge_dim] = _r.lo[merge_dim];
            return;
          }
        }
      }
    }

    // if we can't extend, just add another rectangle if we're not at the
    //  limit
    if((max_rects == 0) || (rects.size() < max_rects)) {
      rects.push_back(_r);
      return;
    }

    // as a last resort, scan the (hopefully short, since it's bounded) list of
    //  current rectangles, and see which one suffers the least from being
    //  union'd with this new rectangle
    {
      size_t best_idx = 0;
      Rect<N,T> best_bbox = rects[0].union_bbox(_r);
      size_t best_volume = best_bbox.volume();
      for(size_t i = 1; i < rects.size(); i++) {
        Rect<N,T> bbox = rects[i].union_bbox(_r);
        size_t volume = bbox.volume();
        if(volume < best_volume) {
          best_idx = i;
          best_bbox = bbox;
          best_volume = volume;
        }
      }
      // swap the union'd bbox to the end if it's not already there
      if(best_idx < (rects.size() - 1))
        rects[best_idx] = rects[rects.size() - 1];
      rects[rects.size() - 1] = best_bbox;
    }
#if 0
    //std::cout << "slow path!\n";
    // our rectangle may break into multiple pieces that we have to 
    //  iteratively add
    std::vector<Rect<N,T> > to_add(1, _r);
#ifdef REALM_DEBUG_RECT_MERGING
    std::vector<Rect<N,T> > orig_rects(rects);
#endif

    while(!to_add.empty()) {
      Rect<N,T> r = to_add.back();
      to_add.pop_back();

      // scan through rectangles, looking for containment (really good),
      //   mergability (also good), or overlap (bad)
      std::vector<int> absorbed;
      int count = rects.size();
      int i = 0;
      while(i < count) {
	// old rect containing new means we can drop the new rectangle
	if(rects[i].contains(r)) break;

	// new rect containing old absorbs it and continues
	if(r.contains(rects[i])) {
	  absorbed.push_back(i);
	  i++;
	  continue;
	}

	// mergeability absorbs the old rect into the new one and starts
	//  it over in case it can now merge with something bigger
	if(can_merge(rects[i], r)) {
	  absorbed.push_back(i);
	  to_add.push_back(r.union_bbox(rects[i]));
	  break;
	}

	// disjoint?  continue on
	if(!rects[i].overlaps(r)) {
	  i++;
	  continue;
	}

	// if we get here, rects[i] and r overlap partially
	// nasty case - figure out the up to 2N-1 rectangles that describe
	//  r - rects[i] and start each of them over from the beginning
	for(int j = 0; j < N; j++) {
	  if(r.lo[j] < rects[i].lo[j]) {
	    // leftover "below"
	    Rect<N,T> subr = r;
	    subr.hi[j] = rects[i].lo[j] - 1;
	    r.lo[j] = rects[i].lo[j];
	    to_add.push_back(subr);
	  }

	  if(r.hi[j] > rects[i].hi[j]) {
	    // leftover "above"
	    Rect<N,T> subr = r;
	    subr.lo[j] = rects[i].hi[j] + 1;
	    r.hi[j] = rects[i].hi[j];
	    to_add.push_back(subr);
	  }
	}
	break;
      }

      // did we actually get to the end?  if so, add our (possibly-grown)
      //  rectangle, using the first absorbed slot if it exists
      if(i == count) {
	if(absorbed.empty()) {
	  rects.push_back(r);
	  count++;
	} else {
	  rects[absorbed.back()] = r;
	  absorbed.pop_back();
	}
      }

      // any other absorbed rectangled need to be deleted - work from the
      //  highest to lowest to avoid moving any absorbed entries
      if(!absorbed.empty()) {
	int i = absorbed.size();
	while(i-- > 0) {
	  if(absorbed[i] < (count - 1))
	    std::swap(rects[absorbed[i]], rects[count - 1]);
	  count--;
	}
	rects.resize(count);
      }
    }

#ifdef REALM_DEBUG_RECT_MERGING
    log_part.print() << "add: " << _r << " + " << orig_rects << " = " << rects;

    // sanity-check: no two rectangles should overlap or be mergeable
    for(size_t i = 0; i < rects.size(); i++)
      for(size_t j = i + 1; j < rects.size(); j++) {
	assert(!rects[i].overlaps(rects[j]));
	assert(!can_merge(rects[i], rects[j]));
      }
#endif
#endif
  }

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const DenseRectangleList<N,T>& drl)
  {
    os << "drl";
    if(drl.rects.empty()) {
      os << "{}";
    } else {
      os << "{";
      for(typename std::vector<Rect<N,T> >::const_iterator it = drl.rects.begin();
	  it != drl.rects.end();
	  ++it)
	os << " " << *it;
      os << " }";
    }
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class HybridRectangleList<N,T>

  template <int N, typename T>
  HybridRectangleList<N,T>::HybridRectangleList(void)
  {}

  template <int N, typename T>
  inline void HybridRectangleList<N,T>::add_point(const Point<N,T>& p)
  {
    as_vector.add_point(p);
    //as_vector.push_back(Rect<N,T>(p, p));
  }

  template <int N, typename T>
  inline void HybridRectangleList<N,T>::add_rect(const Rect<N,T>& r)
  {
    as_vector.add_rect(r);
    //as_vector.push_back(r);
  }

  template <int N, typename T>
  inline const std::vector<Rect<N,T> >& HybridRectangleList<N,T>::convert_to_vector(void)
  {
    return as_vector.rects;
  }

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const HybridRectangleList<N,T>& hrl)
  {
    os << "hrl[]";
    os << "hrl";
    if(hrl.as_vector.rects.empty()) {
      os << "{}";
    } else {
      os << "{ (vec)";
      for(typename std::vector<Rect<N,T> >::const_iterator it = hrl.as_vector.rects.begin();
	  it != hrl.as_vector.rects.end();
	  ++it)
	os << " " << *it;
      os << " }";
    }
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class HybridRectangleList<1,T>

  template <typename T>
  class HybridRectangleList<1,T> : public DenseRectangleList<1,T> {
  public:
    static const size_t HIGH_WATER_MARK = 64;
    static const size_t LOW_WATER_MARK = 16;

    HybridRectangleList(void);

    void add_point(const Point<1,T>& p);

    void add_rect(const Rect<1,T>& r);

    const std::vector<Rect<1,T> >& convert_to_vector(void);
    void convert_to_map(void);

    bool is_vector;
    std::map<T, T> as_map;
  };

  template <typename T>
  HybridRectangleList<1,T>::HybridRectangleList(void)
    : is_vector(true)
  {}

  template <typename T>
  void HybridRectangleList<1,T>::add_point(const Point<1,T>& p)
  {
    if(is_vector) {
      DenseRectangleList<1,T>::add_point(p);
      if(this->rects.size() > HIGH_WATER_MARK)
	convert_to_map();
      return;
    }

    // otherwise add to the map
    assert(!as_map.empty());
    typename std::map<T, T>::iterator it = as_map.lower_bound(p.x);
    if(it == as_map.end()) {
      //std::cout << "add " << p << " BIGGER " << as_map.rbegin()->first << "," << as_map.rbegin()->second << "\n";
      // bigger than everything - see if we can merge with the last guy
      T& last = as_map.rbegin()->second;
      if(last == (p.x - 1))
	last = p.x;
      else if(last < (p.x - 1))
	as_map[p.x] = p.x;
    } 
    else if(it->first == p.x) {
      //std::cout << "add " << p << " OVERLAP1 " << it->first << "," << it->second << "\n";
      // we're the beginning of an existing range - nothing to do
    } else if(it == as_map.begin()) {
      //std::cout << "add " << p << " FIRST " << it->first << "," << it->second << "\n";
      // we're before everything - see if we can merge with the first guy
      if(it->first == (p.x + 1)) {
	T last = it->second;
	as_map.erase(it);
	as_map[p.x] = last;
      } else {
	as_map[p.x] = p.x;
      }
    } else {
      typename std::map<T, T>::iterator it2 = it; --it2;
      //std::cout << "add " << p << " BETWEEN " << it->first << "," << it->second << " / " << it2->first << "," << it2->second << "\n";
      if(it2->second >= p.x) {
	// range below us includes us - nothing to do
      } else {
	bool merge_above = it->first == (p.x + 1);
	bool merge_below = it2->second == (p.x - 1);

	if(merge_below) {
	  if(merge_above) {
	    it2->second = it->second;
	    as_map.erase(it);
	  } else
	    it2->second = p.x;
	} else {
	  T last;
	  if(merge_above) {
	    last = it->second;
	    as_map.erase(it);
	  } else
	    last = p.x;
	  as_map[p.x] = last;
	}
      }
    }
    // mergers can cause us to drop below LWM
    if(as_map.size() < LOW_WATER_MARK)
      convert_to_vector();
  }

  template <typename T>
  void HybridRectangleList<1,T>::add_rect(const Rect<1,T>& r)
  {
    // never add an empty rectangle
    if(r.empty())
      return;

    if(is_vector) {
      DenseRectangleList<1,T>::add_rect(r);
      if(this->rects.size() > HIGH_WATER_MARK)
	convert_to_map();
      return;
    }

    // otherwise add to the map
    assert(!as_map.empty());
    typename std::map<T, T>::iterator it = as_map.lower_bound(r.lo.x);
    if(it == as_map.end()) {
      //std::cout << "add " << p << " BIGGER " << as_map.rbegin()->first << "," << as_map.rbegin()->second << "\n";
      // bigger than everything - see if we can merge with the last guy
      T& last = as_map.rbegin()->second;
      if(last == (r.lo.x - 1))
	last = r.hi.x;
      else if(last < (r.lo.x - 1))
	as_map[r.lo.x] = r.hi.x;
    } else {
      // if the interval we found isn't the first, we may need to back up one to
      //  find the one that overlaps the start of our range
      if(it != as_map.begin()) {
	typename std::map<T, T>::iterator it2 = it;
	--it2;
	if(it2->second >= (r.lo.x - 1))
	  it = it2;
      }

      if(it->first <= r.lo.x) {
	assert((it->second + 1) >= r.lo.x); // it had better overlap or just touch

	if(it->second < r.hi.x)
	  it->second = r.hi.x;
      } else {
	// we are the low end of a range (but may absorb other ranges)
	it = as_map.insert(std::make_pair(r.lo.x, r.hi.x)).first;
      }

      // have we subsumed or merged with anything?
      typename std::map<T, T>::iterator it2 = it;
      ++it2;
      while((it2 != as_map.end()) && (it2->first <= (it->second + 1))) {
	if(it2->second > it->second) it->second = it2->second;
	typename std::map<T, T>::iterator it3 = it2;
	++it2;
	as_map.erase(it3);
      }
    }
    // mergers can cause us to drop below LWM
    if(as_map.size() < LOW_WATER_MARK)
      convert_to_vector();
  }

  template <typename T>
  void HybridRectangleList<1,T>::convert_to_map(void)
  {
    if(!is_vector) return;
    assert(as_map.empty());
    for(typename std::vector<Rect<1,T> >::iterator it = this->rects.begin();
	it != this->rects.end();
	it++)
      as_map[it->lo.x] = it->hi.x;
    this->rects.clear();
    is_vector = false;
  }

  template <typename T>
  const std::vector<Rect<1,T> >& HybridRectangleList<1,T>::convert_to_vector(void)
  {
    if(!is_vector) {
      assert(this->rects.empty());
      for(typename std::map<T, T>::iterator it = as_map.begin();
	  it != as_map.end();
	  it++) {
	Rect<1,T> r;
	r.lo.x = it->first;
	r.hi.x = it->second;
	this->rects.push_back(r);
      }
      for(size_t i = 1; i < this->rects.size(); i++)
	assert(this->rects[i-1].hi.x < (this->rects[i].lo.x - 1));
      as_map.clear();
      is_vector = true;
    }
    return this->rects;
  }

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const HybridRectangleList<1,T>& hrl)
  {
    os << "hrl";
    if(hrl.is_vector) {
      if(hrl.rects.empty()) {
	os << "{}";
      } else {
	os << "{ (vec)";
	for(typename std::vector<Rect<1,T> >::const_iterator it = hrl.rects.begin();
	    it != hrl.rects.end();
	    ++it)
	  os << " " << *it;
	os << " }";
      }
    } else {
      os << "{ (map)";
      for(typename std::map<T,T>::const_iterator it = hrl.as_map.begin();
	  it != hrl.as_map.end();
	  ++it)
	os << " " << Rect<1,T>(it->first, it->second);
      os << " }";
    }
    return os;
  }
    
};

#endif // REALM_DEPPART_RECTLIST_INL
