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

#ifndef REALM_DEPPART_RECTLIST_H
#define REALM_DEPPART_RECTLIST_H

#include "realm/indexspace.h"

namespace Realm {

  // although partitioning operations eventually generate SparsityMap's, we work with
  //  various intermediates that try to keep things from turning into one big bitmask

  // the CoverageCounter just counts the number of points that get added to it
  // it's not even smart enough to eliminate duplicates
  template <int N, typename T>
  class CoverageCounter {
  public:
    CoverageCounter(void);

    void add_point(const Point<N,T>& p);

    void add_rect(const Rect<N,T>& r);

    size_t get_count(void) const;

  protected:
    size_t count;
  };

  // NOTE: the DenseRectangleList does NOT guarantee to remove all duplicate
  //  entries - some merging is done opportunistically to keep the list size
  //  down, but we don't pay for perfect de-duplication here
  template <int N, typename T>
  class DenseRectangleList {
  public:
    DenseRectangleList(size_t _max_rects = 0);

    void add_point(const Point<N,T>& p);

    void add_rect(const Rect<N,T>& r);

    void merge_rects(size_t upper_bound);

    std::vector<Rect<N,T> > rects;
    size_t max_rects;
    int merge_dim;
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const DenseRectangleList<N,T>& drl);

  template <int N, typename T>
  class HybridRectangleList {
  public:
    static const size_t HIGH_WATER_MARK = 64;
    static const size_t LOW_WATER_MARK = 16;

    HybridRectangleList(void);

    void add_point(const Point<N,T>& p);

    void add_rect(const Rect<N,T>& r);

    const std::vector<Rect<N,T> >& convert_to_vector(void);

    //std::vector<Rect<N,T> > as_vector;
    DenseRectangleList<N,T> as_vector;
    //std::multimap<T, Rect<N,T> > as_mmap;
  };
    
  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const HybridRectangleList<N,T>& hrl);

};

#endif // REALM_DEPPART_RECTLIST_H

#include "realm/deppart/rectlist.inl"
