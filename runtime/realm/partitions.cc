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

// index space partitioning for Realm

#include "partitions.h"

#include "profiling.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ZIndexSpace<N,T>

  template <int N, typename T>
  inline Event ZIndexSpace<N,T>::create_equal_subspaces(size_t count, size_t granularity,
							std::vector<ZIndexSpace<N,T> >& subspaces,
							const ProfilingRequestSet &reqs,
							Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());
    //assert(reqs.empty());

    // dense case is easy(er)
    if(dense()) {
      // always split in x dimension for now
      assert(count >= 1);
      T total_x = std::max(bounds.hi.x - bounds.lo.x + 1, 0);
      subspaces.reserve(count);
      T px = bounds.lo.x;
      for(size_t i = 0; i < count; i++) {
	ZIndexSpace<N,T> ss(*this);
	T nx = bounds.lo.x + (total_x * (i + 1) / count);
	ss.bounds.lo.x = px;
	ss.bounds.hi.x = nx - 1;
	subspaces.push_back(ss);
	px = nx;
      }
      return Event::NO_EVENT;
    }

    // TODO: sparse case
    assert(0);
    return Event::NO_EVENT;
  }

  template <int N, typename T>
  inline Event ZIndexSpace<N,T>::create_weighted_subspaces(size_t count, size_t granularity,
							   const std::vector<int>& weights,
							   std::vector<ZIndexSpace<N,T> >& subspaces,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());
    //assert(reqs.empty());

    // determine the total weight
    size_t total_weight = 0;
    assert(weights.size() == count);
    for(size_t i = 0; i < count; i++)
      total_weight += weights[i];

    // dense case is easy(er)
    if(dense()) {
      // always split in x dimension for now
      assert(count >= 1);
      T total_x = std::max(bounds.hi.x - bounds.lo.x + 1, 0);
      subspaces.reserve(count);
      T px = bounds.lo.x;
      size_t cum_weight = 0;
      for(size_t i = 0; i < count; i++) {
	ZIndexSpace<N,T> ss(*this);
	cum_weight += weights[i];
	T nx = bounds.lo.x + (total_x * cum_weight / total_weight);
	ss.bounds.lo.x = px;
	ss.bounds.hi.x = nx - 1;
	subspaces.push_back(ss);
	px = nx;
      }
      return Event::NO_EVENT;
    }

    // TODO: sparse case
    assert(0);
    return Event::NO_EVENT;
  }

  template <int N, typename T>
  template <typename FT>
  inline Event ZIndexSpace<N,T>::create_subspaces_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
							   const std::vector<FT>& colors,
							   std::vector<ZIndexSpace<N,T> >& subspaces,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // just give copies of ourselves for now
    size_t n = colors.size();
    subspaces.resize(n);
    for(size_t i = 0; i < n; i++)
      subspaces[i] = *this;

    return Event::NO_EVENT;
  }


  template <int N, typename T>
  template <int N2, typename T2>
  inline Event ZIndexSpace<N,T>::create_subspaces_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& field_data,
							   const std::vector<ZIndexSpace<N2,T2> >& sources,
							   std::vector<ZIndexSpace<N,T> >& images,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // just give copies of ourselves for now
    size_t n = sources.size();
    images.resize(n);
    for(size_t i = 0; i < n; i++)
      images[i] = *this;

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  template <int N2, typename T2>
  Event ZIndexSpace<N,T>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > >& field_data,
						       const std::vector<ZIndexSpace<N2,T2> >& targets,
						       std::vector<ZIndexSpace<N,T> >& preimages,
						       const ProfilingRequestSet &reqs,
						       Event wait_on /*= Event::NO_EVENT*/) const
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // just give copies of ourselves for now
    size_t n = targets.size();
    preimages.resize(n);
    for(size_t i = 0; i < n; i++)
      preimages[i] = *this;

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_unions(const std::vector<ZIndexSpace<N,T> >& lhss,
						    const std::vector<ZIndexSpace<N,T> >& rhss,
						    std::vector<ZIndexSpace<N,T> >& results,
						    const ProfilingRequestSet &reqs,
						    Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // just give copies of lhss for now
    size_t n = lhss.size();
    results.resize(n);
    for(size_t i = 0; i < n; i++)
      results[i] = lhss[i];

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_intersections(const std::vector<ZIndexSpace<N,T> >& lhss,
							   const std::vector<ZIndexSpace<N,T> >& rhss,
							   std::vector<ZIndexSpace<N,T> >& results,
							   const ProfilingRequestSet &reqs,
							   Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // just give copies of lhss for now
    size_t n = lhss.size();
    results.resize(n);
    for(size_t i = 0; i < n; i++)
      results[i] = lhss[i];

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_differences(const std::vector<ZIndexSpace<N,T> >& lhss,
							 const std::vector<ZIndexSpace<N,T> >& rhss,
							 std::vector<ZIndexSpace<N,T> >& results,
							 const ProfilingRequestSet &reqs,
							 Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    // just give copies of lhss for now
    assert(lhss.size() == rhss.size());
    size_t n = lhss.size();
    results.resize(n);
    for(size_t i = 0; i < n; i++)
      results[i] = lhss[i];

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_union(const std::vector<ZIndexSpace<N,T> >& subspaces,
						   ZIndexSpace<N,T>& result,
						   const ProfilingRequestSet &reqs,
						   Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    assert(!subspaces.empty());
    // WRONG
    result = subspaces[0];

    return Event::NO_EVENT;
  }

  template <int N, typename T>
  /*static*/ Event ZIndexSpace<N,T>::compute_intersection(const std::vector<ZIndexSpace<N,T> >& subspaces,
							  ZIndexSpace<N,T>& result,
							  const ProfilingRequestSet &reqs,
							  Event wait_on /*= Event::NO_EVENT*/)
  {
    // no support for deferring yet
    assert(wait_on.has_triggered());

    assert(!subspaces.empty());
    // WRONG
    result = subspaces[0];

    return Event::NO_EVENT;
  }


  namespace {
    template <int N, typename T>
    class InstantiatePartitioningStuff {
    public:
      typedef ZIndexSpace<N,T> IS;
      template <typename FT>
      static void inst_field(void)
      {
	ZIndexSpace<N,T> i;
	std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> > field_data;
	std::vector<FT> colors;
	std::vector<ZIndexSpace<N,T> > subspaces;
	i.create_subspaces_by_field(field_data, colors, subspaces,
				    Realm::ProfilingRequestSet());
      }
      template <int N2, typename T2>
      static void inst_image_and_preimage(void)
      {
	std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,ZPoint<N2,T2> > > field_data;
	std::vector<ZIndexSpace<N, T> > list1;
	std::vector<ZIndexSpace<N2, T2> > list2;
	list1[0].create_subspaces_by_preimage(field_data, list2, list1,
					      Realm::ProfilingRequestSet());
	list2[0].create_subspaces_by_image(field_data, list1, list2,
					   Realm::ProfilingRequestSet());
      }
      static void inst_stuff(void)
      {
	inst_field<int>();
	inst_field<bool>();
	inst_image_and_preimage<1,int>();

	ZIndexSpace<N,T> i;
	std::vector<int> weights;
	std::vector<ZIndexSpace<N,T> > list;
	i.create_equal_subspaces(0, 0, list, Realm::ProfilingRequestSet());
	i.create_weighted_subspaces(0, 0, weights, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_unions(list, list, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_intersections(list, list, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_differences(list, list, list, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_union(list, i, Realm::ProfilingRequestSet());
	ZIndexSpace<N,T>::compute_intersection(list, i, Realm::ProfilingRequestSet());
      }
    };

    //
  };

  void (*dummy)(void) __attribute__((unused)) = &InstantiatePartitioningStuff<1,int>::inst_stuff;
  //InstantiatePartitioningStuff<1,int> foo __attribute__((unused));
};

