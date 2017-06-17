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

// rectangle lists for Realm partitioning

#ifndef REALM_DEPPART_RECTLIST_INL
#define REALM_DEPPART_RECTLIST_INL

#include "rectlist.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CoverageCounter<N,T>

  template <int N, typename T>
  inline CoverageCounter<N,T>::CoverageCounter(void)
    : count(0)
  {}

  template <int N, typename T>
  inline void CoverageCounter<N,T>::add_point(const ZPoint<N,T>& p)
  {
    count++;
  }

  template <int N, typename T>
  inline void CoverageCounter<N,T>::add_rect(const ZRect<N,T>& r)
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
  inline void DenseRectangleList<N,T>::add_point(const ZPoint<N,T>& p)
  {
    if(rects.empty()) {
      rects.push_back(ZRect<N,T>(p, p));
      return;
    }

    if(N == 1) {
      // optimize for sorted insertion (i.e. stuff at end)
      {
	ZRect<N,T> &lr = *rects.rbegin();
	if(p.x == (lr.hi.x + 1)) {
	  lr.hi.x = p.x;
	  return;
	}
	if(p.x > (lr.hi.x + 1)) {
	  rects.push_back(ZRect<N,T>(p, p));
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
	  rects.insert(rects.begin() + lo, ZRect<N,T>(p, p));
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
      add_rect(ZRect<N,T>(p,p));
    }
  }

  template <int N, typename T>
  inline bool can_merge(const ZRect<N,T>& r1, const ZRect<N,T>& r2)
  {
    // N-1 dimensions must match exactly and 1 may be adjacent
    int idx = 0;
    while((idx < N) && (r1.lo[idx] == r2.lo[idx]) && (r1.hi[idx] == r2.hi[idx]))
      idx++;

    // if we get all the way through, the rectangles are equal and can be "merged"
    if(idx >= N) return true;

    // if not, this has to be the dimension that is adjacent
    if((r1.lo[idx] != (r2.hi[idx] + 1)) && (r2.lo[idx] != (r1.hi[idx] + 1)))
      return false;

    // and the rest of the dimensions have to match too
    while(++idx < N)
      if((r1.lo[idx] != r2.lo[idx]) || (r1.hi[idx] != r2.hi[idx]))
	return false;

    return true;
  }

  template <int N, typename T>
  inline void DenseRectangleList<N,T>::add_rect(const ZRect<N,T>& _r)
  {
    if(rects.empty()) {
      rects.push_back(_r);
      return;
    }

    if(N == 1) {
      // try to optimize for sorted insertion (i.e. stuff at end)
      ZRect<N,T> &lr = *rects.rbegin();
      if(_r.lo.x == (lr.hi.x + 1)) {
	lr.hi.x = _r.hi.x;
	return;
      }
      if(_r.lo.x > (lr.hi.x + 1)) {
	rects.push_back(_r);
	if((max_rects > 0) && (rects.size() > (size_t)max_rects)) {
	  std::cout << "need better compression\n";
	  rects[max_rects-1].hi = rects[max_rects].hi;
	  rects.resize(max_rects);
	}
	return;
      }
    }

    //std::cout << "slow path!\n";
    ZRect<N,T> r = _r;

    // scan through rectangles, looking for containment (really good),
    //   mergability (also good), or overlap (bad)
    int merge_with = -1;
    std::vector<int> absorbed;
    int count = rects.size();
    for(int i = 0; i < count; i++) {
      if(rects[i].contains(r)) return;
      if(rects[i].overlaps(r)) {
        assert(N == 1);  // TODO: splitting for 2+-D
        r = r.union_bbox(rects[i]);
        absorbed.push_back(i);
        continue;
      }
      if((merge_with == -1) && can_merge(rects[i], r))
	merge_with = i;
    }

    if(merge_with == -1) {
      if(absorbed.empty()) {
        // no merge candidates and nothing absorbed, just add the new rectangle
        rects.push_back(r);
      } else {
        // replace the first absorbed rectangle, delete the others (if any)
        rects[absorbed[0]] = r;
        for(size_t i = 1; i < absorbed.size(); i++) {
          if(absorbed[i] < (count - 1))
            std::swap(rects[absorbed[i]], rects[count - 1]);
          count--;
        }
        rects.resize(count);
      }
      return;
    }

#ifdef DEBUG_PARTITIONING
    std::cout << "merge: " << rects[merge_with] << " and " << r << std::endl;
#endif
    rects[merge_with] = rects[merge_with].union_bbox(r);

    // this may trigger a cascade merge, so look again
    int last_merged = merge_with;
    while(true) {
      merge_with = -1;
      for(int i = 0; i < (int)rects.size(); i++) {
	if((i != last_merged) && can_merge(rects[i], rects[last_merged])) {
	  merge_with = i;
	  break;
	}
      }
      if(merge_with == -1)
	return;  // all done

      // merge downward in case one of these is the last one
      if(merge_with > last_merged)
	std::swap(merge_with, last_merged);

#ifdef DEBUG_PARTITIONING
      std::cout << "merge: " << rects[merge_with] << " and " << rects[last_merged] << std::endl;
#endif
      rects[merge_with] = rects[merge_with].union_bbox(rects[last_merged]);

      // can delete last merged
      int last = rects.size() - 1;
      if(last != last_merged)
	std::swap(rects[last_merged], rects[last]);
      rects.resize(last);

      last_merged = merge_with;
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class HybridRectangleList<N,T>

  template <int N, typename T>
  HybridRectangleList<N,T>::HybridRectangleList(void)
  {}

  template <int N, typename T>
  inline void HybridRectangleList<N,T>::add_point(const ZPoint<N,T>& p)
  {
    as_vector.push_back(ZRect<N,T>(p, p));
  }

  template <int N, typename T>
  inline void HybridRectangleList<N,T>::add_rect(const ZRect<N,T>& r)
  {
    as_vector.push_back(r);
  }

  template <int N, typename T>
  inline const std::vector<ZRect<N,T> >& HybridRectangleList<N,T>::convert_to_vector(void)
  {
    return as_vector;
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

    void add_point(const ZPoint<1,T>& p);

    void add_rect(const ZRect<1,T>& r);

    const std::vector<ZRect<1,T> >& convert_to_vector(void);
    void convert_to_map(void);

    bool is_vector;
    std::map<T, T> as_map;
  };

  template <typename T>
  HybridRectangleList<1,T>::HybridRectangleList(void)
    : is_vector(true)
  {}

  template <typename T>
  void HybridRectangleList<1,T>::add_point(const ZPoint<1,T>& p)
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
  void HybridRectangleList<1,T>::add_rect(const ZRect<1,T>& r)
  {
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
	assert(it->second >= r.lo.x); // it had better overlap

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
    for(typename std::vector<ZRect<1,T> >::iterator it = this->rects.begin();
	it != this->rects.end();
	it++)
      as_map[it->lo.x] = it->hi.x;
    this->rects.clear();
    is_vector = false;
  }

  template <typename T>
  const std::vector<ZRect<1,T> >& HybridRectangleList<1,T>::convert_to_vector(void)
  {
    if(!is_vector) {
      assert(this->rects.empty());
      for(typename std::map<T, T>::iterator it = as_map.begin();
	  it != as_map.end();
	  it++) {
	ZRect<1,T> r;
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

    
};

#endif // REALM_DEPPART_RECTLIST_INL
