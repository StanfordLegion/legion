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

// per-dimension instantiator for image.cc

#define REALM_TEMPLATES_ONLY
#include "./image.cc"

#ifndef INST_N1
  #error INST_N1 must be defined!
#endif
#ifndef INST_N2
  #error INST_N2 must be defined!
#endif

#define FOREACH_TT(__func__) \
  __func__(int,int) \
  __func__(int,unsigned) \
  __func__(int,long long) \
  __func__(unsigned,int) \
  __func__(unsigned,unsigned) \
  __func__(unsigned,long long) \
  __func__(long long,int) \
  __func__(long long,unsigned) \
  __func__(long long,long long)


namespace Realm {

#define N1 INST_N1
#define N2 INST_N2

#define DOIT(T1,T2)			                                                                                             \
  template class StructuredImageMicroOp<N1,T1,N2,T2>;                                                                \
  template class ImageMicroOp<N1,T1,N2,T2>;                                                                          \
  template class ImageOperation<N1,T1,N2,T2>;                                                                        \
  template ImageMicroOp<N1,T1,N2,T2>::ImageMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template Event IndexSpace<N1, T1>::create_subspaces_by_image(                                                      \
      const DomainTransform<N1, T1, N2, T2> &, const std::vector<IndexSpace<N2, T2> > &,                             \
      std::vector<IndexSpace<N1, T1> > &, const ProfilingRequestSet &, Event)                                        \
      const;                                                                                                         \
  template Event IndexSpace<N1,T1>::create_subspaces_by_image_with_difference(                                       \
      const DomainTransform<N1, T1, N2, T2> &,                                                                       \
									       const std::vector<IndexSpace<N2,T2> >&,                                                     \
									       const std::vector<IndexSpace<N1,T1> >&,                                                     \
									       std::vector<IndexSpace<N1,T1> >&,                                                           \
									       const ProfilingRequestSet&,                                                                 \
									       Event) const;

  FOREACH_TT(DOIT)

  };  // namespace Realm
