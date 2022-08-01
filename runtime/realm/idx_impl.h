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

// Realm index space implementation

#ifndef REALM_IDX_IMPL_H
#define REALM_IDX_IMPL_H

#include "realm/indexspace.h"

namespace Realm {

  class IndexSpaceGenericImpl {
  public:
    virtual ~IndexSpaceGenericImpl();

    virtual IndexSpaceGenericImpl *clone_at(void *dst) const = 0;

    virtual Event copy(const std::vector<CopySrcDstField> &srcs,
		       const std::vector<CopySrcDstField> &dsts,
		       const void *indirects_data,
		       size_t indirect_len,
		       const ProfilingRequestSet &requests,
		       Event wait_on) const = 0;

    // given an instance layout, attempts to provide bounds (start relative to
    //  the base of the instance and relative limit - i.e. first nonaccessibly
    //  offset) on affine accesses via elements in this index space - returns
    //  true if successful, false if not
    virtual bool compute_affine_bounds(const InstanceLayoutGeneric *ilg,
                                       FieldID fid,
                                       uintptr_t& rel_base,
                                       uintptr_t& limit) const = 0;
  };

  template <int N, typename T>
  class IndexSpaceGenericImplTyped : public IndexSpaceGenericImpl {
  public:
    IndexSpaceGenericImplTyped(const IndexSpace<N,T>& _space);

    virtual IndexSpaceGenericImpl *clone_at(void *dst) const;

    virtual Event copy(const std::vector<CopySrcDstField> &srcs,
		       const std::vector<CopySrcDstField> &dsts,
		       const void *indirects_data,
		       size_t indirect_len,
		       const ProfilingRequestSet &requests,
		       Event wait_on) const;

    virtual bool compute_affine_bounds(const InstanceLayoutGeneric *ilg,
                                       FieldID fid,
                                       uintptr_t& rel_base,
                                       uintptr_t& limit) const;

    IndexSpace<N,T> space;
  };

};

#endif
