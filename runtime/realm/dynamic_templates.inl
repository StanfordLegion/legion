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

// helper templates for dynamically packing/unpacking template parameters

// nop, but helps IDEs
#include "realm/dynamic_templates.h"

#include <assert.h>

namespace Realm {

  namespace DynamicTemplates {

    ////////////////////////////////////////////////////////////////////////
    //
    // class TypeListElem<_HEAD,_TAIL>

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, int N>
    template <typename T1>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::DemuxHelper<TARGET,N>::demux(int index, T1 arg1)
    {
      if(index == N)
	TARGET::template demux<_HEAD>(arg1);
      else
	_TAIL::template DemuxHelper<TARGET, N+1>::template demux<T1>(index, arg1);
    }

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, int N>
    template <typename T1, typename T2>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::DemuxHelper<TARGET,N>::demux(int index, T1 arg1, T2 arg2)
    {
      if(index == N)
	TARGET::template demux<_HEAD>(arg1, arg2);
      else
	_TAIL::template DemuxHelper<TARGET, N+1>::template demux<T1, T2>(index, arg1, arg2);
    }

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, int N>
    template <typename T1, typename T2, typename T3>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::DemuxHelper<TARGET,N>::demux(int index, T1 arg1, T2 arg2, T3 arg3)
    {
      if(index == N)
	TARGET::template demux<_HEAD>(arg1, arg2, arg3);
      else
	_TAIL::template DemuxHelper<TARGET, N+1>::template demux<T1, T2, T3>(index, arg1, arg2, arg3);
    }

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, int N>
    template <typename T1, typename T2, typename T3, typename T4>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::DemuxHelper<TARGET,N>::demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
    {
      if(index == N)
	TARGET::template demux<_HEAD>(arg1, arg2, arg3, arg4);
      else
	_TAIL::template DemuxHelper<TARGET, N+1>::template demux<T1, T2, T3, T4>(index, arg1, arg2, arg3, arg4);
    }

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, typename T1>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::demux(int index, T1 arg1)
    {
      return DemuxHelper<TARGET, 0>::demux(index, arg1);
    }

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, typename T1, typename T2>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::demux(int index, T1 arg1, T2 arg2)
    {
      return DemuxHelper<TARGET, 0>::demux(index, arg1, arg2);
    }

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, typename T1, typename T2, typename T3>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::demux(int index, T1 arg1, T2 arg2, T3 arg3)
    {
      return DemuxHelper<TARGET, 0>::demux(index, arg1, arg2, arg3);
    }

    template <typename _HEAD, typename _TAIL>
    template <typename TARGET, typename T1, typename T2, typename T3, typename T4>
    inline /*static*/ void TypeListElem<_HEAD,_TAIL>::demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
    {
      return DemuxHelper<TARGET, 0>::demux(index, arg1, arg2, arg3, arg4);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class TypeListTerm

    template <typename TARGET, int N>
    template <typename T1>
    inline /*static*/ void TypeListTerm::DemuxHelper<TARGET,N>::demux(int index, T1 arg1)
    {
      assert(0);
    }

    template <typename TARGET, int N>
    template <typename T1, typename T2>
    inline /*static*/ void TypeListTerm::DemuxHelper<TARGET,N>::demux(int index, T1 arg1, T2 arg2)
    {
      assert(0);
    }

    template <typename TARGET, int N>
    template <typename T1, typename T2, typename T3>
    inline /*static*/ void TypeListTerm::DemuxHelper<TARGET,N>::demux(int index, T1 arg1, T2 arg2, T3 arg3)
    {
      assert(0);
    }

    template <typename TARGET, int N>
    template <typename T1, typename T2, typename T3, typename T4>
    inline /*static*/ void TypeListTerm::DemuxHelper<TARGET,N>::demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
    {
      assert(0);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class IntList<MIN,MAX>

    template <int MIN, int MAX>
    template <typename TARGET, typename T1>
    inline /*static*/ void IntList<MIN,MAX>::demux(int index, T1 arg1)
    {
      assert((MIN <= index) && (index <= MAX));
      IntDemuxHelper<TARGET,MIN,MAX-MIN>::template demux<T1>(index, arg1);
    }

    template <int MIN, int MAX>
    template <typename TARGET, typename T1, typename T2>
    inline /*static*/ void IntList<MIN,MAX>::demux(int index, T1 arg1, T2 arg2)
    {
      assert((MIN <= index) && (index <= MAX));
      IntDemuxHelper<TARGET,MIN,MAX-MIN>::template demux<T1,T2>(index, arg1, arg2);
    }

    template <int MIN, int MAX>
    template <typename TARGET, typename T1, typename T2, typename T3>
    inline /*static*/ void IntList<MIN,MAX>::demux(int index, T1 arg1, T2 arg2, T3 arg3)
    {
      assert((MIN <= index) && (index <= MAX));
      IntDemuxHelper<TARGET,MIN,MAX-MIN>::template demux<T1,T2,T3>(index, arg1, arg2, arg3);
    }

    template <int MIN, int MAX>
    template <typename TARGET, typename T1, typename T2, typename T3, typename T4>
    inline /*static*/ void IntList<MIN,MAX>::demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
    {
      assert((MIN <= index) && (index <= MAX));
      IntDemuxHelper<TARGET,MIN,MAX-MIN>::template demux<T1,T2,T3,T4>(index, arg1, arg2, arg3, arg4);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class ListProduct2<L1,L2>

    template <typename L1, typename L2>
    template <typename T1, typename T2> REALM_CUDA_HD
    inline /*static*/ constexpr TagType ListProduct2<L1,L2>::encode_tag(void)
    {
      return (L1::template TypeToIndex<T1>::INDEX << 8) + (L2::template TypeToIndex<T2>::INDEX);
    }

    template <typename L1, typename L2>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1>
    inline /*static*/ void ListProduct2<L1,L2>::DemuxHelper2<TARGET,T1>::demux(A1 arg1)
    {
      TARGET::template demux<T1, T2>(arg1);
    }

    template <typename L1, typename L2>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1, typename A2>
    inline /*static*/ void ListProduct2<L1,L2>::DemuxHelper2<TARGET,T1>::demux(A1 arg1, A2 arg2)
    {
      TARGET::template demux<T1, T2>(arg1, arg2);
    }

    template <typename L1, typename L2>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct2<L1,L2>::DemuxHelper2<TARGET,T1>::demux(A1 arg1, A2 arg2, A3 arg3)
    {
      TARGET::template demux<T1, T2>(arg1, arg2, arg3);
    }

    template <typename L1, typename L2>
    template <typename TARGET>
    template <typename T1, typename A1>
    inline /*static*/ void ListProduct2<L1,L2>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1)
    {
      TagType tag2 = (tag >> 0) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, A1>(tag2, arg1);
    }

    template <typename L1, typename L2>
    template <typename TARGET>
    template <typename T1, typename A1, typename A2>
    inline /*static*/ void ListProduct2<L1,L2>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag2 = (tag >> 0) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, A1, A2>(tag2, arg1, arg2);
    }

    template <typename L1, typename L2>
    template <typename TARGET>
    template <typename T1, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct2<L1,L2>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag2 = (tag >> 0) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, A1, A2, A3>(tag2, arg1, arg2, arg3);
    }

    template <typename L1, typename L2>
    template <typename TARGET, typename A1>
    inline /*static*/ void ListProduct2<L1,L2>::demux(TagType tag, A1 arg1)
    {
      TagType tag1 = (tag >> 8) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1>(tag1, tag, arg1);
    }

    template <typename L1, typename L2>
    template <typename TARGET, typename A1, typename A2>
    inline /*static*/ void ListProduct2<L1,L2>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag1 = (tag >> 8) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1, A2>(tag1, tag, arg1, arg2);
    }

    template <typename L1, typename L2>
    template <typename TARGET, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct2<L1,L2>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag1 = (tag >> 8) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1, A2, A3>(tag1, tag, arg1, arg2, arg3);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class ListProduct3<L1,L2,L3>

    template <typename L1, typename L2, typename L3>
    template <typename T1, typename T2, typename T3> REALM_CUDA_HD
    inline /*static*/ constexpr TagType ListProduct3<L1,L2,L3>::encode_tag(void)
    {
      return ((L1::template TypeToIndex<T1>::INDEX << 16) +
	      (L2::template TypeToIndex<T2>::INDEX << 8) +
	      (L3::template TypeToIndex<T3>::INDEX));
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename T1, typename T2>
    template <typename T3, typename A1>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper3<TARGET,T1,T2>::demux(A1 arg1)
    {
      TARGET::template demux<T1, T2, T3>(arg1);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename T1, typename T2>
    template <typename T3, typename A1, typename A2>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper3<TARGET,T1,T2>::demux(A1 arg1, A2 arg2)
    {
      TARGET::template demux<T1, T2, T3>(arg1, arg2);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename T1, typename T2>
    template <typename T3, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper3<TARGET,T1,T2>::demux(A1 arg1, A2 arg2, A3 arg3)
    {
      TARGET::template demux<T1, T2, T3>(arg1, arg2, arg3);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper2<TARGET,T1>::demux(TagType tag, A1 arg1)
    {
      TagType tag3 = (tag >> 0) & 0xff;
      L3::template demux<DemuxHelper3<TARGET, T1, T2>, A1>(tag3, arg1);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1, typename A2>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper2<TARGET,T1>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag3 = (tag >> 0) & 0xff;
      L3::template demux<DemuxHelper3<TARGET, T1, T2>, A1, A2>(tag3, arg1, arg2);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper2<TARGET,T1>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag3 = (tag >> 0) & 0xff;
      L3::template demux<DemuxHelper3<TARGET, T1, T2>, A1, A2, A3>(tag3, arg1, arg2, arg3);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET>
    template <typename T1, typename A1>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1)
    {
      TagType tag2 = (tag >> 8) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, TagType, A1>(tag2, tag, arg1);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET>
    template <typename T1, typename A1, typename A2>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag2 = (tag >> 8) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, TagType, A1, A2>(tag2, tag, arg1, arg2);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET>
    template <typename T1, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct3<L1,L2,L3>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag2 = (tag >> 8) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, TagType, A1, A2, A3>(tag2, tag, arg1, arg2, arg3);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename A1>
    inline /*static*/ void ListProduct3<L1,L2,L3>::demux(TagType tag, A1 arg1)
    {
      TagType tag1 = (tag >> 16) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1>(tag1, tag, arg1);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename A1, typename A2>
    inline /*static*/ void ListProduct3<L1,L2,L3>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag1 = (tag >> 16) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1, A2>(tag1, tag, arg1, arg2);
    }

    template <typename L1, typename L2, typename L3>
    template <typename TARGET, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct3<L1,L2,L3>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag1 = (tag >> 16) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1, A2, A3>(tag1, tag, arg1, arg2, arg3);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class ListProduct4<L1,L2,L3,L4>

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename T1, typename T2, typename T3, typename T4> REALM_CUDA_HD
    inline /*static*/ constexpr TagType ListProduct4<L1,L2,L3,L4>::encode_tag(void)
    {
      return ((L1::template TypeToIndex<T1>::INDEX << 24) +
	      (L2::template TypeToIndex<T2>::INDEX << 16) +
	      (L3::template TypeToIndex<T3>::INDEX << 8) +
	      (L4::template TypeToIndex<T4>::INDEX));
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1, typename T2, typename T3>
    template <typename T4, typename A1>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper4<TARGET,T1,T2,T3>::demux(A1 arg1)
    {
      TARGET::template demux<T1, T2, T3, T4>(arg1);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1, typename T2, typename T3>
    template <typename T4, typename A1, typename A2>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper4<TARGET,T1,T2,T3>::demux(A1 arg1, A2 arg2)
    {
      TARGET::template demux<T1, T2, T3, T4>(arg1, arg2);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1, typename T2, typename T3>
    template <typename T4, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper4<TARGET,T1,T2,T3>::demux(A1 arg1, A2 arg2, A3 arg3)
    {
      TARGET::template demux<T1, T2, T3, T4>(arg1, arg2, arg3);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1, typename T2>
    template <typename T3, typename A1>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper3<TARGET,T1,T2>::demux(TagType tag, A1 arg1)
    {
      TagType tag4 = (tag >> 0) & 0xff;
      L4::template demux<DemuxHelper4<TARGET, T1, T2, T3>, A1>(tag4, arg1);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1, typename T2>
    template <typename T3, typename A1, typename A2>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper3<TARGET,T1,T2>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag4 = (tag >> 0) & 0xff;
      L4::template demux<DemuxHelper4<TARGET, T1, T2, T3>, A1, A2>(tag4, arg1, arg2);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1, typename T2>
    template <typename T3, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper3<TARGET,T1,T2>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag4 = (tag >> 0) & 0xff;
      L4::template demux<DemuxHelper4<TARGET, T1, T2, T3>, A1, A2, A3>(tag4, arg1, arg2, arg3);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper2<TARGET,T1>::demux(TagType tag, A1 arg1)
    {
      TagType tag3 = (tag >> 8) & 0xff;
      L3::template demux<DemuxHelper3<TARGET, T1, T2>, TagType, A1>(tag3, tag, arg1);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1, typename A2>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper2<TARGET,T1>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag3 = (tag >> 8) & 0xff;
      L3::template demux<DemuxHelper3<TARGET, T1, T2>, TagType, A1, A2>(tag3, tag, arg1, arg2);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename T1>
    template <typename T2, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper2<TARGET,T1>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag3 = (tag >> 8) & 0xff;
      L3::template demux<DemuxHelper3<TARGET, T1, T2>, TagType, A1, A2, A3>(tag3, tag, arg1, arg2, arg3);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET>
    template <typename T1, typename A1>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1)
    {
      TagType tag2 = (tag >> 16) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, TagType, A1>(tag2, tag, arg1);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET>
    template <typename T1, typename A1, typename A2>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag2 = (tag >> 16) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, TagType, A1, A2>(tag2, tag, arg1, arg2);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET>
    template <typename T1, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::DemuxHelper1<TARGET>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag2 = (tag >> 16) & 0xff;
      L2::template demux<DemuxHelper2<TARGET, T1>, TagType, A1, A2, A3>(tag2, tag, arg1, arg2, arg3);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename A1>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::demux(TagType tag, A1 arg1)
    {
      TagType tag1 = (tag >> 24) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1>(tag1, tag, arg1);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename A1, typename A2>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::demux(TagType tag, A1 arg1, A2 arg2)
    {
      TagType tag1 = (tag >> 24) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1, A2>(tag1, tag, arg1, arg2);
    }

    template <typename L1, typename L2, typename L3, typename L4>
    template <typename TARGET, typename A1, typename A2, typename A3>
    inline /*static*/ void ListProduct4<L1,L2,L3,L4>::demux(TagType tag, A1 arg1, A2 arg2, A3 arg3)
    {
      TagType tag1 = (tag >> 24) & 0xff;
      L1::template demux<DemuxHelper1<TARGET>, TagType, A1, A2, A3>(tag1, tag, arg1, arg2, arg3);
    }


  }; // namespace DynamicTemplates

}; // namespace Realm

