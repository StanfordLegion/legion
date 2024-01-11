/* Copyright 2023 Stanford University, NVIDIA Corporation
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

/**
 * \file sparsity.h
 * This file provides the interface for sparsity maps, which are used to
 * represent the sparsity of index spaces.
 */

namespace Realm {

  template <int N, typename T /*= int*/> struct Point;
  template <int N, typename T /*= int*/> struct Rect;
  template <int N, typename T = int> class HierarchicalBitMap;

  template <int N, typename T> class SparsityMapPublicImpl;

  /**
   * \class SparistyMap
   * SparsityMap is the Realm handle that (like all other handles) can be
   * copied/stored whereever and represents a name for a distributed object with
   * a known "creator" node, with valid data for the object existing on one or
   * more nodes (which may or may not include the creator node) - methods on
   * this "object" simply forward to the actual implementation object.
   *
   * There are three layers to the SparsityMap onion: SpartisyMap,
   * SparsityMapPublicImpl, and SparsityMapImpl.
   */
  template <int N, typename T>
  class REALM_PUBLIC_API SparsityMap {
  public:
    typedef ::realm_id_t id_t;
    id_t id;
    bool operator<(const SparsityMap<N,T> &rhs) const;
    bool operator==(const SparsityMap<N,T> &rhs) const;
    bool operator!=(const SparsityMap<N,T> &rhs) const;

    //static const SparsityMap<N,T> NO_SPACE;

    /**
     * Check if this sparsity map exists.
     * @return true if this sparsity map exists, false otherwise
     */
    REALM_CUDA_HD
    bool exists(void) const;

    /**
     * Destroy the sparsity map.
     * @param wait_on a precondition event
     */
    void destroy(Event wait_on = Event::NO_EVENT);

    /**
     * Add one reference to the sparsity map.
     */
    void add_reference(void);

    /**
     * Remove references from the sparsity map.
     * @param count a number of references to remove
     */
    void remove_references(int count = 1);

    /**
     * Lookup the public implementation object for this sparsity map.
     * @return the public implementation object for this sparsity map
     */
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    SparsityMapPublicImpl<N,T> *impl(void) const;

    ///@{
    /**
     * Construct a sparsity map from a set of points or rectangles.
     * @param points/rects a vector of points/rects.
     * @param always_create if true, always create a sparsity map, even if the points
     *                     completely fill their bounding box
     *                     if false, return NO_SPACE if the points completely fill their
     *                     bounding box (i.e. id == 0)
     * @param disjoint if true, the points are assumed to be disjoint
     * @return a sparsity map
     */
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    static SparsityMap<N,T> construct(const std::vector<Point<N,T> >& points,
				      bool always_create, bool disjoint);
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    static SparsityMap<N,T> construct(const std::vector<Rect<N,T> >& rects,
				      bool always_create, bool disjoint);
    ///@}

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

  /**
   * \class SparsityMapPublicImpl
   *
   * SparsityMapPublicImpl is the public subset of the storage and functionality
   * of the actual sparsity map implementation - this should be sufficient for
   * all the needs of user code, but not for Realm internals (e.g. the part that
   * actually computes new sparsity maps) - these objects are not allocated
   * directly.
   *
   */
  template <int N, typename T>
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE SparsityMapPublicImpl {
  protected:
    // cannot be constructed directly
    SparsityMapPublicImpl(void);

  public:
   /**
    * Make this sparsity map valid.
    * Applications should call this method before attempting to use the contents
    * of a sparsity map or otherwise defer the actual use until the returned
    * event has triggered.
    *
    * The valid sparsity map is called "completed" and is computed
    * in a deferred fashion and fetched by other nodes on demand.
    * @param precise if true, the sparsity map is computed precisely
    * @return an event that triggers when the sparsity map is valid
    */
   Event make_valid(bool precise = true);

   /**
    * Check if this sparsity map is valid.
    * @param precise if true, the sparsity map is computed precisely
    * @return true if the sparsity map is valid, false otherwise
    */
   bool is_valid(bool precise = true);

   /**
    * Get the entries of this sparsity map.
    * A sparsity map entry is similar to an IndexSpace - it's a rectangle and
    * optionally a reference to another SparsityMap OR a pointer to a
    * HierarchicalBitMap, which is a dense array of bits describing the
    * validity of each point in the rectangle.
    * @return the entries of this sparsity map
    */
   const std::vector<SparsityMapEntry<N, T> >& get_entries(void);

   /**
    * Get the approximate rectangles of this sparsity map.
    * A sparsity map can exist in an approximate form as well - this is a
    * bounded list of rectangles that are guaranteed to cover all actual
    * entries.
    * @return the approximate rectangles of this sparsity map
    */
   const std::vector<Rect<N, T> >& get_approx_rects(void);

   /**
    * Check if this sparsity map overlaps another sparsity map.
    * This method is not cheap - try bounds-based checks first (see
    * IndexSpace::overlaps).
    * @param other the other sparsity map
    * @param bounds the bounds of the other sparsity map
    * @param approx if true, use the approximate rectangles of this sparsity map
    * @return true if this sparsity map overlaps the other sparsity map, false
    * otherwise
    */
   bool overlaps(SparsityMapPublicImpl<N, T>* other, const Rect<N, T>& bounds,
                 bool approx);

   /**
    * Attempt to compute a set of covering rectangles.
    * See IndexSpace<N,T>::compute_covering for a description of this.
    * @param bounds the bounds of the rectangle
    * @param max_rects Maximum number of rectangles to use (0 = no limit).
    * @param max_overhead Maximum relative storage overhead (0 = no limit).
    * @param covering Vector to fill in with covering rectangles.
    * @return true if the covering was computed, false otherwise
    */
   bool compute_covering(const Rect<N, T>& bounds, size_t max_rects,
                         int max_overhead, std::vector<Rect<N, T> >& covering);

  protected:
    atomic<bool> entries_valid, approx_valid;
    std::vector<SparsityMapEntry<N,T> > entries;
    std::vector<Rect<N,T> > approx_rects;
  };

}; // namespace Realm

#include "realm/sparsity.inl"

#endif // ifndef REALM_SPARSITY_H

