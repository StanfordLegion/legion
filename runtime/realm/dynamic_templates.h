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

#ifndef REALM_DYNAMIC_TEMPLATES_H
#define REALM_DYNAMIC_TEMPLATES_H

#include "compiler_support.h"

namespace Realm {
  namespace DynamicTemplates {

    // a TypeList is a linked list of templated types that captures a list of types
    // and supports the following template-land operations:
    // a) TL::TypePresent<T>::value - returns a boolean indicating if T is in the list
    // b) TL::TypeToIndex<T>::INDEX - returns the 0-based index of a type in the list
    //                                    (compile error if not present)
    // c) TL::IndexToType<N>::TYPE - returns the type in position N in the list
    //                                    (compile error if not present)
    //
    // and a family of dynamic operations:
    // d) TL::demux<TARGET>(i, args...) - calls TARGET::demux<T>(args...) where T
    //          is the i'th type in the list

    template <typename _HEAD, typename _TAIL> struct TypeListElem;
    struct TypeListTerm;

    // variadic templates from C++11 would be more graceful
    template <typename T1 = void, typename T2 = void, typename T3 = void,
	      typename T4 = void, typename T5 = void, typename T6 = void>
    struct TypeList {
      typedef TypeListElem<T1, typename TypeList<T2, T3, T4, T5, T6>::TL> TL;
    };

    template <>
    struct TypeList<void, void, void, void, void, void> {
      typedef TypeListTerm TL;
    };

    template <typename _HEAD, typename _TAIL> 
    struct TypeListElem {
      typedef _HEAD HEAD;
      typedef _TAIL TAIL;

      // TypePresent support
      template <typename T, typename ORIG = TypeListElem<HEAD,TAIL> >
      struct TypePresent {
	static const bool value = _TAIL::template TypePresent<T,ORIG>::value;
      };
      template <typename ORIG>
      struct TypePresent<HEAD, ORIG> {
	static const bool value = true;
      };

      // TypeToIndex support
      template <typename T, typename ORIG = TypeListElem<HEAD,TAIL> >
      struct TypeToIndex {
	static const int INDEX = 1 + _TAIL::template TypeToIndex<T,ORIG>::INDEX;
      };
      template <typename ORIG>
      struct TypeToIndex<HEAD, ORIG> {
	static const int INDEX = 0;
      };

      // IndexToType support
      template <int N, typename ORIG = TypeListElem<HEAD,TAIL> >
      struct IndexToType {
	typedef typename _TAIL::template IndexToType<N-1,ORIG>::TYPE TYPE;
      };
      template <typename ORIG>
      struct IndexToType<0, ORIG> {
	typedef _HEAD TYPE;
      };

      // MaxSize support
      struct MaxSize {
        // Unforuntately std::max is not constexpr until c++14
        static constexpr size_t value = sizeof(_HEAD) < _TAIL::MaxSize::value ?
          _TAIL::MaxSize::value : sizeof(_HEAD);
      };

      // MaxSizeType support
      template <size_t SIZE>
      struct MaxSizeType {
        typedef typename std::conditional<SIZE == sizeof(_HEAD),_HEAD,
                  typename _TAIL::template MaxSizeType<SIZE> >::type TYPE;
      };

      // demux support
      template <typename TARGET, int N>
      struct DemuxHelper {
	template <typename T1>
	static void demux(int index, T1 arg1);

	template <typename T1, typename T2>
	static void demux(int index, T1 arg1, T2 arg2);

	template <typename T1, typename T2, typename T3>
	static void demux(int index, T1 arg1, T2 arg2, T3 arg3);

	template <typename T1, typename T2, typename T3, typename T4>
	static void demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
      };

      template <typename TARGET, typename T1>
      static void demux(int index, T1 arg1);

      template <typename TARGET, typename T1, typename T2>
      static void demux(int index, T1 arg1, T2 arg2);

      template <typename TARGET, typename T1, typename T2, typename T3>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3);

      template <typename TARGET, typename T1, typename T2, typename T3, typename T4>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
    };

    struct TypeListTerm {
      // TypePresent support
      template <typename T, typename ORIG>
      struct TypePresent {
	static const bool value = false;
      };

      // TypeToIndex support
      template <typename T, typename ORIG>
      struct ERROR_TYPE_NOT_IN_LIST {};

      template <typename T, typename ORIG>
      struct TypeToIndex {
	static const int INDEX = ERROR_TYPE_NOT_IN_LIST<T,ORIG>::INDEX;
      };

      // IndexToType support
      template <typename ORIG>
      struct ERROR_INDEX_NOT_IN_LIST {};
      template <int N, typename ORIG>
      struct IndexToType {
	static const int INDEX = ERROR_INDEX_NOT_IN_LIST<ORIG>::TYPE;
      };

      // MaxSize support
      struct MaxSize {
        static constexpr size_t value = 0;
      };

      struct ERROR_SIZE_NOT_IN_LIST { };

      template <size_t SIZE>
      struct MaxSizeType {
        typedef ERROR_SIZE_NOT_IN_LIST TYPE;
      };

      // demux support
      template <typename TARGET, int N>
      struct DemuxHelper {
	template <typename T1>
	static void demux(int index, T1 arg1);

	template <typename T1, typename T2>
	static void demux(int index, T1 arg1, T2 arg2);

	template <typename T1, typename T2, typename T3>
	static void demux(int index, T1 arg1, T2 arg2, T3 arg3);

	template <typename T1, typename T2, typename T3, typename T4>
	static void demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
      };
    };


    // an IntList is like a TypeList, but it conceptually holds a contiguous range of
    //  integers.  For uniformity, it actually holds Int<N>'s and supports the same operations:
    // a) TL::TypePresent<N>::value - returns a boolean indicating if N is in the list
    // b) TL::TypeToIndex<Int<N>>::INDEX - returns N if it is in the list
    //                                    (compile error if not present)
    // c) TL::IndexToType<N>::TYPE - returns Int<N> if N is in the list
    //                                    (compile error if not present)
    //
    // and a family of dynamic operations:
    // d) TL::demux<TARGET>(i, args...) - calls TARGET::demux<Int<i>>(args...) if i is in the list

    template <int _N> struct Int { static const int N = _N; };

    template <typename TARGET, int BASE, int DELTA>
    struct IntDemuxHelper {
      template <typename T1>
      static void demux(int index, T1 arg1)
      {
        if(index == (BASE + DELTA))
          TARGET::template demux<Int<BASE + DELTA> >(arg1);
        else
          IntDemuxHelper<TARGET, BASE, DELTA-1>::template demux<T1>(index, arg1);
      }

      template <typename T1, typename T2>
      static void demux(int index, T1 arg1, T2 arg2)
      {
        if(index == (BASE + DELTA))
          TARGET::template demux<Int<BASE + DELTA> >(arg1, arg2);
        else
          IntDemuxHelper<TARGET, BASE, DELTA-1>::template demux<T1,T2>(index, arg1, arg2);
      }

      template <typename T1, typename T2, typename T3>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3)
      {
        if(index == (BASE + DELTA))
          TARGET::template demux<Int<BASE + DELTA> >(arg1, arg2, arg3);
        else
          IntDemuxHelper<TARGET, BASE, DELTA-1>::template demux<T1,T2,T3>(index, arg1, arg2, arg3);
      }

      template <typename T1, typename T2, typename T3, typename T4>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
      {
        if(index == (BASE + DELTA))
          TARGET::template demux<Int<BASE + DELTA> >(arg1, arg2, arg3, arg4);
        else
          IntDemuxHelper<TARGET, BASE, DELTA-1>::template demux<T1,T2,T3,T4>(index, arg1, arg2, arg3, arg4);
      }
    };

    template <typename TARGET, int BASE>
    struct IntDemuxHelper<TARGET, BASE, 0> {
      template <typename T1>
      static void demux(int index, T1 arg1)
      {
        TARGET::template demux<Int<BASE> >(arg1);
      }

      template <typename T1, typename T2>
      static void demux(int index, T1 arg1, T2 arg2)
      {
        TARGET::template demux<Int<BASE> >(arg1, arg2);
      }

      template <typename T1, typename T2, typename T3>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3)
      {
        TARGET::template demux<Int<BASE> >(arg1, arg2, arg3);
      }

      template <typename T1, typename T2, typename T3, typename T4>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
      {
        TARGET::template demux<Int<BASE> >(arg1, arg2, arg3, arg4);
      }
    };

    template <int MIN, int MAX>
    struct IntList {
      template <int N>
      struct TypePresent {
	static const bool value = (MIN <= N) && (N <= MAX);
      };

      template <int N>
      struct ERROR_INT_NOT_IN_LIST {};

      template <int N, bool OK>
      struct TypeIndexHelper {
	static const int INDEX = ERROR_INT_NOT_IN_LIST<N>::INDEX;
	typedef typename ERROR_INT_NOT_IN_LIST<N>::TYPE TYPE;
      };
      template <int N>
      struct TypeIndexHelper<N, true> {
	static const int INDEX = N;
	typedef Int<N> TYPE;
      };

      template <typename T>
      struct TypeToIndex {
	static const int INDEX = TypeIndexHelper<T::N, TypePresent<T::N>::value>::INDEX;
      };

      template <int N>
      struct IndexToType {
	typedef typename TypeIndexHelper<N, TypePresent<N>::value>::TYPE TYPE;
      };

      template <typename TARGET, typename T1>
      static void demux(int index, T1 arg1);

      template <typename TARGET, typename T1, typename T2>
      static void demux(int index, T1 arg1, T2 arg2);

      template <typename TARGET, typename T1, typename T2, typename T3>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3);

      template <typename TARGET, typename T1, typename T2, typename T3, typename T4>
      static void demux(int index, T1 arg1, T2 arg2, T3 arg3, T4 arg4);
    };
    
    // templates with multiple arguments are supported via the following helpers

    // nested demuxes require the use of several helper classes (each), in 
    //  order to successively capture:
    // a) the TARGET of the demux
    // b) each looked up type
    // each helper class has several overloaded versions of the demux method to
    // support different argument lists on the TARGET

    typedef unsigned TagType;

    template <typename L1, typename L2>
    struct ListProduct2 {
      template <typename T1, typename T2> REALM_CUDA_HD
      static constexpr TagType encode_tag(void);

      template <typename TARGET, typename T1>
      struct DemuxHelper2 {
	template <typename T2, typename A1>
	static void demux(A1 arg1);

	template <typename T2, typename A1, typename A2>
	static void demux(A1 arg1, A2 arg2);

	template <typename T2, typename A1, typename A2, typename A3>
	static void demux(A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET>
      struct DemuxHelper1 {
	template <typename T1, typename A1>
	static void demux(TagType tag, A1 arg1);

	template <typename T1, typename A1, typename A2>
	static void demux(TagType tag, A1 arg1, A2 arg2);

	template <typename T1, typename A1, typename A2, typename A3>
	static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET, typename A1>
      static void demux(TagType tag, A1 arg1);

      template <typename TARGET, typename A1, typename A2>
      static void demux(TagType tag, A1 arg1, A2 arg2);

      template <typename TARGET, typename A1, typename A2, typename A3>
      static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);

    };

    template <typename L1, typename L2, typename L3>
    struct ListProduct3 {
      template <typename T1, typename T2, typename T3> REALM_CUDA_HD
      static constexpr TagType encode_tag(void);

      template <typename TARGET, typename T1, typename T2>
      struct DemuxHelper3 {
	template <typename T3, typename A1>
	static void demux(A1 arg1);

	template <typename T3, typename A1, typename A2>
	static void demux(A1 arg1, A2 arg2);

	template <typename T3, typename A1, typename A2, typename A3>
	static void demux(A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET, typename T1>
      struct DemuxHelper2 {
	template <typename T2, typename A1>
	static void demux(TagType tag, A1 arg1);

	template <typename T2, typename A1, typename A2>
	static void demux(TagType tag, A1 arg1, A2 arg2);

	template <typename T2, typename A1, typename A2, typename A3>
	static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET>
      struct DemuxHelper1 {
	template <typename T1, typename A1>
	static void demux(TagType tag, A1 arg1);

	template <typename T1, typename A1, typename A2>
	static void demux(TagType tag, A1 arg1, A2 arg2);

	template <typename T1, typename A1, typename A2, typename A3>
	static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET, typename A1>
      static void demux(TagType tag, A1 arg1);

      template <typename TARGET, typename A1, typename A2>
      static void demux(TagType tag, A1 arg1, A2 arg2);

      template <typename TARGET, typename A1, typename A2, typename A3>
      static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);

    };

    template <typename L1, typename L2, typename L3, typename L4>
    struct ListProduct4 {
      template <typename T1, typename T2, typename T3, typename T4> REALM_CUDA_HD
      static constexpr TagType encode_tag(void);

      template <typename TARGET, typename T1, typename T2, typename T3>
      struct DemuxHelper4 {
	template <typename T4, typename A1>
	static void demux(A1 arg1);

	template <typename T4, typename A1, typename A2>
	static void demux(A1 arg1, A2 arg2);

	template <typename T4, typename A1, typename A2, typename A3>
	static void demux(A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET, typename T1, typename T2>
      struct DemuxHelper3 {
	template <typename T3, typename A1>
	static void demux(TagType tag, A1 arg1);

	template <typename T3, typename A1, typename A2>
	static void demux(TagType tag, A1 arg1, A2 arg2);

	template <typename T3, typename A1, typename A2, typename A3>
	static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET, typename T1>
      struct DemuxHelper2 {
	template <typename T2, typename A1>
	static void demux(TagType tag, A1 arg1);

	template <typename T2, typename A1, typename A2>
	static void demux(TagType tag, A1 arg1, A2 arg2);

	template <typename T2, typename A1, typename A2, typename A3>
	static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET>
      struct DemuxHelper1 {
	template <typename T1, typename A1>
	static void demux(TagType tag, A1 arg1);

	template <typename T1, typename A1, typename A2>
	static void demux(TagType tag, A1 arg1, A2 arg2);

	template <typename T1, typename A1, typename A2, typename A3>
	static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);
      };

      template <typename TARGET, typename A1>
      static void demux(TagType tag, A1 arg1);

      template <typename TARGET, typename A1, typename A2>
      static void demux(TagType tag, A1 arg1, A2 arg2);

      template <typename TARGET, typename A1, typename A2, typename A3>
      static void demux(TagType tag, A1 arg1, A2 arg2, A3 arg3);

    };

  }; // namespace DynamicTemplates

}; // namespace Realm

#include "realm/dynamic_templates.inl"

#endif
