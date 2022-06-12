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

// per-dimension instantiator for region_tree.inl

#include "legion.h"
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/region_tree.h"
#include "legion/legion_spy.h"
#include "legion/legion_context.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_analysis.h"
#include "legion/legion_trace.h"

#ifndef INST_N1
  #error INST_N1 must be defined!
#endif

#ifdef INST_N2
  #define DEFINE_NTNT_TEMPLATES
#else
  #define DEFINE_NT_TEMPLATES
#endif

#include "legion/region_tree.inl"

#define FOREACH_T(__func__) \
  __func__(int) \
  __func__(unsigned) \
  __func__(long long)

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

namespace Legion {
  namespace Internal {

#define N1 INST_N1
#define N2 INST_N2

#ifdef INST_N2
  // NTNT templates (helper methods on IndexSpaceNodeT)
#define DOIT_TT(T1,T2) \
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_by_domain_helper<INST_N2,T2>(Operation *,    \
                                     IndexPartNode *,     \
                                     FutureMapImpl *,     \
                                     bool); \
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_by_weight_helper<INST_N2,T2>(Operation *,    \
                                     IndexPartNode *,   \
                                     FutureMapImpl *,   \
                                     size_t); \
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_by_field_helper<INST_N2,T2>(Operation *,	\
				     IndexPartNode *,	  \
				     const std::vector<FieldDataDescriptor> &, \
				     ApEvent); \
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_by_image_helper<INST_N2,T2>(Operation *, \
				     IndexPartNode *, \
				     IndexPartNode *, \
				     const std::vector<FieldDataDescriptor> &, \
				     ApEvent); \
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_by_image_range_helper<INST_N2,T2>(Operation *, \
				     IndexPartNode *,	\
				     IndexPartNode *,		\
				     const std::vector<FieldDataDescriptor> &, \
				     ApEvent); \
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_by_preimage_helper<INST_N2,T2>(Operation *, \
				     IndexPartNode *,	\
				     IndexPartNode *,		\
				     const std::vector<FieldDataDescriptor> &, \
				     ApEvent);		\
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_by_preimage_range_helper<INST_N2,T2>(Operation *, \
				     IndexPartNode *,	\
				     IndexPartNode *,	\
				     const std::vector<FieldDataDescriptor> &, \
				     ApEvent); \
  template ApEvent IndexSpaceNodeT<INST_N1,T1>:: \
    create_association_helper<INST_N2,T2>(Operation *, \
				     IndexSpaceNode *, \
				     const std::vector<FieldDataDescriptor> &, \
				     ApEvent); \
  template ApEvent CopyAcrossUnstructuredT<INST_N1,T1>:: \
    perform_compute_preimages<INST_N2,T2>(std::vector<DomainT<INST_N1,T1> >&, \
                                     Operation*, \
                                     ApEvent, \
                                     const bool); \
  template bool CopyAcrossUnstructuredT<INST_N1,T1>:: \
    rebuild_indirections<INST_N2,T2>(const bool);

  FOREACH_TT(DOIT_TT)
#else
  // just the NT template
#define DOIT_T(T) \
  template class Internal::IndexSpaceNodeT<INST_N1,T>; \
  template class Internal::IndexPartNodeT<INST_N1,T>; \
  template class Internal::IndexSpaceUnion<INST_N1,T>; \
  template class Internal::IndexSpaceIntersection<INST_N1,T>; \
  template class Internal::IndexSpaceDifference<INST_N1,T>; \
  template class Internal::InstanceExpression<INST_N1,T>; \
  template class Internal::RemoteExpression<INST_N1,T>;
  

  FOREACH_T(DOIT_T)
#endif

  };
};
