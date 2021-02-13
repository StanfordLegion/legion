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

// Realm index space implementation

#include "realm/idx_impl.h"

#include "realm/deppart/inst_helper.h"
#include "realm/instance.h"

#include <iomanip>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // struct CopySrcDstField

  std::ostream& operator<<(std::ostream& os, const CopySrcDstField& sd)
  {
    if(sd.field_id >= 0) {
      os << "field(" << sd.field_id;
      if(sd.indirect_index >= 0)
	os << ", ind=" << sd.indirect_index;
      else
	os << ", inst=" << sd.inst;
      if(sd.redop_id != 0)
	os << ", redop=" << sd.redop_id << (sd.red_fold ? "(fold)" : "(apply)");
      if(sd.serdez_id != 0)
	os << ", serdez=" << sd.serdez_id;
      os << ", size=" << sd.size;
      if(sd.subfield_offset != 0)
	os << "+" << sd.subfield_offset;
      os << ")";
    } else {
      os << "fill(";
      if(sd.size <= CopySrcDstField::MAX_DIRECT_SIZE) {
	os << std::hex << std::setfill('0');
	// show data in little-endian order
	for(size_t i = 0; i < sd.size; i++)
	  os << std::setw(2)
             << (int)(unsigned char)(sd.fill_data.direct[sd.size - 1 - i]);
	os << std::dec << ")";
      } else
	os << "size=" << sd.size << ")";
    }
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct IndexSpaceGeneric

  IndexSpaceGeneric::IndexSpaceGeneric()
    : impl(0)
  {}

  IndexSpaceGeneric::IndexSpaceGeneric(const IndexSpaceGeneric& copy_from)
  {
    if(copy_from.impl)
      impl = copy_from.impl->clone_at(raw_storage);
    else
      impl = 0;
  }

  template <int N, typename T>
  IndexSpaceGeneric::IndexSpaceGeneric(const IndexSpace<N,T>& copy_from)
  {
    assert(STORAGE_BYTES >= sizeof(IndexSpaceGenericImplTyped<N,T>));
    impl = new(raw_storage) IndexSpaceGenericImplTyped<N,T>(copy_from);
  }

  IndexSpaceGeneric::~IndexSpaceGeneric()
  {
    if(impl)
      impl->~IndexSpaceGenericImpl();
  }

  IndexSpaceGeneric& IndexSpaceGeneric::operator=(const IndexSpaceGeneric& copy_from)
  {
    if(&copy_from != this) {
      if(impl)
	impl->~IndexSpaceGenericImpl();
      if(copy_from.impl)
	impl = copy_from.impl->clone_at(raw_storage);
      else
	impl = 0;
    }
    return *this;
  }

  template <int N, typename T>
  IndexSpaceGeneric& IndexSpaceGeneric::operator=(const IndexSpace<N,T>& copy_from)
  {
    assert(STORAGE_BYTES >= sizeof(IndexSpaceGenericImplTyped<N,T>));
    if(impl)
      impl->~IndexSpaceGenericImpl();
    impl = new(raw_storage) IndexSpaceGenericImplTyped<N,T>(copy_from);
    return *this;
  }

  template <int N, typename T>
  const IndexSpace<N,T>& IndexSpaceGeneric::as_index_space() const
  {
    IndexSpaceGenericImplTyped<N,T> *typed = dynamic_cast<IndexSpaceGenericImplTyped<N,T> *>(impl);
    assert(typed != 0);
    return typed->space;
  }

  // only IndexSpace method exposed directly is copy
  Event IndexSpaceGeneric::copy(const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				const ProfilingRequestSet &requests,
				Event wait_on /*= Event::NO_EVENT*/) const
  {
    return impl->copy(srcs, dsts, 0, 0, requests, wait_on);
  }

  template <int N, typename T>
  Event IndexSpaceGeneric::copy(const std::vector<CopySrcDstField> &srcs,
				const std::vector<CopySrcDstField> &dsts,
				const std::vector<const typename CopyIndirection<N,T>::Base *> &indirects,
				const ProfilingRequestSet &requests,
				Event wait_on /*= Event::NO_EVENT*/) const
  {
    return impl->copy(srcs, dsts,
		      &indirects[0],
		      indirects.size(),
		      requests,
		      wait_on);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct IndexSpaceGenericImpl

  IndexSpaceGenericImpl::~IndexSpaceGenericImpl()
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct IndexSpaceGenericImplTyped<N,T>

  template <int N, typename T>
  IndexSpaceGenericImplTyped<N,T>::IndexSpaceGenericImplTyped(const IndexSpace<N,T>& _space)
    : space(_space)
  {}

  template <int N, typename T>
  IndexSpaceGenericImpl *IndexSpaceGenericImplTyped<N,T>::clone_at(void *dst) const
  {
    return new(dst) IndexSpaceGenericImplTyped<N,T>(*this);
  }

  template <int N, typename T>
  Event IndexSpaceGenericImplTyped<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
					      const std::vector<CopySrcDstField> &dsts,
					      const void *indirects_data,
					      size_t indirect_len,
					      const ProfilingRequestSet &requests,
					      Event wait_on) const
  {
    // TODO: move to transfer.cc for indirection goodness
    assert(indirect_len == 0);
    return space.copy(srcs, dsts, requests, wait_on);
  }


  // explicit template instantiation

#define DOIT(N,T) \
  template IndexSpaceGeneric::IndexSpaceGeneric(const IndexSpace<N,T>& copy_from); \
  template IndexSpaceGeneric& IndexSpaceGeneric::operator=<N,T>(const IndexSpace<N,T>&); \
  template const IndexSpace<N,T>& IndexSpaceGeneric::as_index_space<N,T>() const; \
  template Event IndexSpaceGeneric::copy<N,T>(const std::vector<CopySrcDstField>&, \
                                              const std::vector<CopySrcDstField>&, \
					      const std::vector<const CopyIndirection<N,T>::Base *>&, \
					      const ProfilingRequestSet&, \
					      Event) const; \
  template class IndexSpaceGenericImplTyped<N,T>;

  FOREACH_NT(DOIT)

}; // namespace Realm
