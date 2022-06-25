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

// per-dimension instantiator for preimage.cc

#define REALM_TEMPLATES_ONLY
#include "./preimage.cc"

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

#define COMMA ,
#define FOREACH_TT1(__func__) \
  __func__(int,int,AffineTransform<INST_N2 COMMA INST_N1 COMMA int>) \
  __func__(int,unsigned,AffineTransform<INST_N2 COMMA INST_N1 COMMA unsigned>) \
  __func__(int,unsigned,AffineTransform<INST_N2 COMMA INST_N1 COMMA int>) \
  __func__(int,long long,AffineTransform<INST_N2 COMMA INST_N1 COMMA unsigned>) \
  __func__(int,long long,AffineTransform<INST_N2 COMMA INST_N1 COMMA long long>) \
  __func__(int,long long,AffineTransform<INST_N2 COMMA INST_N1 COMMA int>) \
  __func__(unsigned,int,AffineTransform<INST_N2 COMMA INST_N1 COMMA int>) \
  __func__(unsigned,int,AffineTransform<INST_N2 COMMA INST_N1 COMMA unsigned>) \
  __func__(unsigned,unsigned,AffineTransform<INST_N2 COMMA INST_N1 COMMA unsigned>) \
  __func__(unsigned,long long,AffineTransform<INST_N2 COMMA INST_N1 COMMA long long>) \
  __func__(unsigned,long long,AffineTransform<INST_N2 COMMA INST_N1 COMMA unsigned>) \
  __func__(long long,int,AffineTransform<INST_N2 COMMA INST_N1 COMMA int>) \
  __func__(long long,int,AffineTransform<INST_N2 COMMA INST_N1 COMMA long long>) \
  __func__(long long,int,AffineTransform<INST_N2 COMMA INST_N1 COMMA unsigned>) \
  __func__(long long,unsigned,AffineTransform<INST_N2 COMMA INST_N1 COMMA unsigned>) \
  __func__(long long,unsigned,AffineTransform<INST_N2 COMMA INST_N1 COMMA long long>) \
  __func__(long long,long long,AffineTransform<INST_N2 COMMA INST_N1 COMMA long long>)

namespace Realm {

#define N1 INST_N1
#define N2 INST_N2

#define DOIT(T1,T2)			    \
  template class PreimageMicroOp<N1,T1,N2,T2>; \
  template class PreimageOperation<N1,T1,N2,T2>; \
  template PreimageMicroOp<N1,T1,N2,T2>::PreimageMicroOp(NodeID, AsyncMicroOp *, Serialization::FixedBufferDeserializer&); \
  template Event IndexSpace<N1,T1>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N1,T1>,Point<N2,T2> > >&, \
								  const std::vector<IndexSpace<N2,T2> >&, \
								  std::vector<IndexSpace<N1,T1> >&, \
								  const ProfilingRequestSet &, \
								  Event) const; \
  template Event IndexSpace<N1,T1>::create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N1,T1>,Rect<N2,T2> > >&, \
								  const std::vector<IndexSpace<N2,T2> >&, \
								  std::vector<IndexSpace<N1,T1> >&, \
								  const ProfilingRequestSet &, \
								  Event) const;

  FOREACH_TT(DOIT)

#define DOIT1(T1, T2, TRANSFORM)                                              \
  template class StructuredPreimageMicroOp<N1, T1, N2, T2, TRANSFORM>;        \
  template class StructuredPreimageOperation<N1, T1, N2, T2, TRANSFORM>;      \
  template Event IndexSpace<N1, T1>::create_subspaces_by_preimage(            \
      const TRANSFORM &, const std::vector<IndexSpace<N2, T2> > &,            \
      std::vector<IndexSpace<N1, T1> > &, const ProfilingRequestSet &, Event) \
      const;

  FOREACH_TT1(DOIT1)

};
