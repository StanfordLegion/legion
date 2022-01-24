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

// sparsity maps for Realm

// NOTE: SparsityMap's are not intended to be manipulated directly by Realm
//  applications (including higher-level runtimes), but they make heavy use of
//  templating and inlining for performance, so the headers are "reachable" from
//  the external parts of the Realm API

#ifndef REALM_SPARSITY_H
#define REALM_SPARSITY_H

#include "realm/realm_config.h"
#include "realm/event.h"
#include "realm/atomics.h"

#include <iostream>
#include <vector>

namespace Realm {

  template <int N, typename T /*= int*/> struct Point;
  template <int N, typename T /*= int*/> struct Rect;
  template <int N, typename T = int> class HierarchicalBitMap;

  // a SparsityMap is a Realm handle to sparsity data for one or more index spaces - all
  //  SparsityMap's use the same ID namespace (i.e. regardless of N and T), but the
  //  template parameters are kept to avoid losing dimensionality information upon iteration/etc.

  // there are three layers to the SparsityMap onion:
  // a) SparsityMap is the Realm handle that (like all other handles) can be copied/stored
  //     whereever and represents a name for a distributed object with a known "creator" node,
  //     with valid data for the object existing on one or more nodes (which may or may not
  //     include the creator node) - methods on this "object" simply forward to the actual
  //     implementation object, described next
  // b) SparsityMapPublicImpl is the public subset of the storage and functionality of the actual
  //     sparsity map implementation - this should be sufficient for all the needs of user code,
  //     but not for Realm internals (e.g. the part that actually computes new sparsity maps) -
  //     these objects are not allocated directly
  // c) SparsityMapImpl is the actual dynamically allocated object that exists on each "interested"
  //     node for a given SparsityMap - it inherits from SparsityMapPublicImpl and adds the "private"
  //     storage and functionality - this separation is primarily to avoid the installed version of
  //     of Realm having to include all the internal .h files

  template <int N, typename T> class SparsityMapPublicImpl;

  template <int N, typename T>
  class REALM_PUBLIC_API SparsityMap {
  public:
    typedef ::realm_id_t id_t;
    id_t id;
    bool operator<(const SparsityMap<N,T> &rhs) const;
    bool operator==(const SparsityMap<N,T> &rhs) const;
    bool operator!=(const SparsityMap<N,T> &rhs) const;

    //static const SparsityMap<N,T> NO_SPACE;
    REALM_CUDA_HD
    bool exists(void) const;

    // looks up the public subset of the implementation object
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    SparsityMapPublicImpl<N,T> *impl(void) const;

    // if 'always_create' is false and the points/rects completely fill their
    //  bounding box, returns NO_SPACE (i.e. id == 0)
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    static SparsityMap<N,T> construct(const std::vector<Point<N,T> >& points,
				      bool always_create, bool disjoint);
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    static SparsityMap<N,T> construct(const std::vector<Rect<N,T> >& rects,
				      bool always_create, bool disjoint);
  };

  template <int N, typename T>
  REALM_PUBLIC_API
  std::ostream& operator<<(std::ostream& os, SparsityMap<N,T> s);

  template <int N, typename T>
  struct SparsityMapEntry {
    Rect<N,T> bounds;
    SparsityMap<N,T> sparsity;
    HierarchicalBitMap<N,T> *bitmap;
  };

  template <int N, typename T>
  REALM_PUBLIC_API
  std::ostream& operator<<(std::ostream& os, const SparsityMapEntry<N,T>& entry);

  template <int N, typename T>
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE SparsityMapPublicImpl {
  protected:
    // cannot be constructed directly
    SparsityMapPublicImpl(void);

  public:
    // application side code should only ever look at "completed" sparsity maps (i.e. ones
    //  that have reached their steady-state immutable value - this is computed in a deferred
    //  fashion and fetched by other nodes on demand, so the application code needs to call
    //  make_valid() before attempting to use the contents and either wait on the event or 
    //  otherwise defer the actual use until the event has triggered
    Event make_valid(bool precise = true);
    bool is_valid(bool precise = true);

    // a sparsity map entry is similar to an IndexSpace - it's a rectangle and optionally a
    //  reference to another SparsityMap OR a pointer to a HierarchicalBitMap, which is a 
    //  dense array of bits describing the validity of each point in the rectangle

    const std::vector<SparsityMapEntry<N,T> >& get_entries(void);
    
    // a sparsity map can exist in an approximate form as well - this is a bounded list of rectangles
    //  that are guaranteed to cover all actual entries

    const std::vector<Rect<N,T> >& get_approx_rects(void);

    // membership test between two (presumably-different) sparsity maps are not
    //  cheap - try bounds-based checks first (see IndexSpace::overlaps)
    bool overlaps(SparsityMapPublicImpl<N,T> *other,
		  const Rect<N,T>& bounds, bool approx);

    // see IndexSpace<N,T>::compute_covering for a description of this
    bool compute_covering(const Rect<N,T> &bounds,
			  size_t max_rects, int max_overhead,
			  std::vector<Rect<N,T> >& covering);

  protected:
    atomic<bool> entries_valid, approx_valid;
    std::vector<SparsityMapEntry<N,T> > entries;
    std::vector<Rect<N,T> > approx_rects;
  };

}; // namespace Realm

#include "realm/sparsity.inl"

#endif // ifndef REALM_SPARSITY_H

