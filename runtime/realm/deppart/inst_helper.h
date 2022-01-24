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

// template instantiation helpers for deppart

#ifndef REALM_DEPPART_INST_HELPER_H
#define REALM_DEPPART_INST_HELPER_H

#include "realm/indexspace.h"
#include "realm/dynamic_templates.h"

namespace Realm {

  struct NT_TemplateHelper : public DynamicTemplates::ListProduct2<DIMCOUNTS, DIMTYPES> {
    typedef DynamicTemplates::ListProduct2<DIMCOUNTS, DIMTYPES> SUPER;
    template <int N, typename T>
    static DynamicTemplates::TagType encode_tag(void) {
      return SUPER::template encode_tag<DynamicTemplates::Int<N>, T>();
    }
  };

  // we need an augmented field list that includes Point<N,T> and Rect<N,T>
  //  for all DIMCOUNTS/DIMTYPES
  struct FLDTYPES_AUG {
    template <typename T>
    struct TypePresent {
      static const bool value = FLDTYPES::TypePresent<T>::value;
    };

    template <int N, typename T>
    struct TypePresent<Point<N,T> > {
      static const bool value = false;
    };

    template <typename T>
    struct TypeToIndex {
      static const int INDEX = FLDTYPES::TypeToIndex<T>::INDEX << 2;
    };

    template <int N, typename T>
    struct TypeToIndex<Point<N,T> > {
      static const int INDEX = ((N << 5) + 
				(DIMTYPES::TypeToIndex<T>::INDEX << 2) +
				1);
    };

#ifdef ZRECT_AS_FIELD_TYPE
    template <int N, typename T>
    struct TypeToIndex<Rect<N,T> > {
      static const int INDEX = ((N << 5) +
				(DIMTYPES::TypeToIndex<T>::INDEX << 2) +
				3);
    };
#endif

    template <typename TARGET, typename T1>
    struct PointDemux1 {
      template <typename NT, typename T>
      static void demux(T1 arg1)
      {
	TARGET::template demux<Point<NT::N, T> >(arg1);
      }
    };

    template <typename TARGET, typename T1, typename T2>
    struct PointDemux2 {
      template <typename NT, typename T>
      static void demux(T1 arg1, T2 arg2)
      {
	TARGET::template demux<Point<NT::N, T> >(arg1, arg2);
      }
    };

    template <typename TARGET, typename T1, typename T2, typename T3>
    struct PointDemux3 {
      template <typename NT, typename T>
      static void demux(T1 arg1, T2 arg2, T3 arg3)
      {
	TARGET::template demux<Point<NT::N, T> >(arg1, arg2, arg3);
      }
    };

#ifdef ZRECT_AS_FIELD_TYPE
    template <typename TARGET, typename T1>
    struct RectDemux1 {
      template <typename NT, typename T>
      static void demux(T1 arg1)
      {
	TARGET::template demux<Rect<NT::N, T> >(arg1);
      }
    };

    template <typename TARGET, typename T1, typename T2>
    struct RectDemux2 {
      template <typename NT, typename T>
      static void demux(T1 arg1, T2 arg2)
      {
	TARGET::template demux<Rect<NT::N, T> >(arg1, arg2);
      }
    };

    template <typename TARGET, typename T1, typename T2, typename T3>
    struct RectDemux3 {
      template <typename NT, typename T>
      static void demux(T1 arg1, T2 arg2, T3 arg3)
      {
	TARGET::template demux<Rect<NT::N, T> >(arg1, arg2, arg3);
      }
    };
#endif

    template <typename TARGET, typename T1>
    static void demux(int index, T1 arg1)
    {
      switch(index & 3) {
      case 0: // fall through to base type list
	{
	  FLDTYPES::demux<TARGET,T1>(index >> 2, arg1);
	  break;
	}

      case 1: // index encodes N,T for Point<N,T>
	{
	  // space the index back out into 8-bit fields
	  int full_index = (((index & 0xe0) << 3) +
			    ((index & 0x1c) >> 2));
	  NT_TemplateHelper::demux<PointDemux1<TARGET,T1> >(full_index, arg1);
	  break;
	}

#ifdef ZRECT_AS_FIELD_TYPE
      case 3: // index encodes N,T for Rect<N,T>
	{
	  // space the index back out into 8-bit fields
	  int full_index = (((index & 0xe0) << 3) +
			    ((index & 0x1c) >> 2));
	  NT_TemplateHelper::demux<RectDemux1<TARGET,T1> >(full_index, arg1);
	  break;
	}
#endif
      }
    }

    template <typename TARGET, typename T1, typename T2>
    static void demux(int index, T1 arg1, T2 arg2)
    {
      switch(index & 3) {
      case 0: // fall through to base type list
	{
	  FLDTYPES::demux<TARGET,T1,T2>(index >> 2, arg1, arg2);
	  break;
	}

      case 1: // index encodes N,T for Point<N,T>
	{
	  // space the index back out into 8-bit fields
	  int full_index = (((index & 0xe0) << 3) +
			    ((index & 0x1c) >> 2));
	  NT_TemplateHelper::demux<PointDemux2<TARGET,T1,T2> >(full_index, arg1, arg2);
	  break;
	}

#ifdef ZRECT_AS_FIELD_TYPE
      case 3: // index encodes N,T for Rect<N,T>
	{
	  // space the index back out into 8-bit fields
	  int full_index = (((index & 0xe0) << 3) +
			    ((index & 0x1c) >> 2));
	  NT_TemplateHelper::demux<RectDemux2<TARGET,T1,T2> >(full_index, arg1, arg2);
	  break;
	}
#endif
      }
    }

    template <typename TARGET, typename T1, typename T2, typename T3>
    static void demux(int index, T1 arg1, T2 arg2, T3 arg3)
    {
      switch(index & 3) {
      case 0: // fall through to base type list
	{
	  FLDTYPES::demux<TARGET,T1,T2,T3>(index >> 2, arg1, arg2, arg3);
	  break;
	}

      case 1: // index encodes N,T for Point<N,T>
	{
	  // space the index back out into 8-bit fields
	  int full_index = (((index & 0xe0) << 3) +
			    ((index & 0x1c) >> 2));
	  NT_TemplateHelper::demux<PointDemux3<TARGET,T1,T2,T3> >(full_index, arg1, arg2, arg3);
	  break;
	}

#ifdef ZRECT_AS_FIELD_TYPE
      case 3: // index encodes N,T for Rect<N,T>
	{
	  // space the index back out into 8-bit fields
	  int full_index = (((index & 0xe0) << 3) +
			    ((index & 0x1c) >> 2));
	  NT_TemplateHelper::demux<RectDemux3<TARGET,T1,T2,T3> >(full_index, arg1, arg2, arg3);
	  break;
	}
#endif
      }
    }
  };

  struct NTF_TemplateHelper : public DynamicTemplates::ListProduct3<DIMCOUNTS, DIMTYPES, FLDTYPES_AUG> {
    typedef DynamicTemplates::ListProduct3<DIMCOUNTS, DIMTYPES, FLDTYPES_AUG> SUPER;
    template <int N, typename T, typename FT>
    static DynamicTemplates::TagType encode_tag(void) {
      return SUPER::template encode_tag<DynamicTemplates::Int<N>, T, FT>();
    }
  };

  struct NTNT_TemplateHelper : public DynamicTemplates::ListProduct4<DIMCOUNTS, DIMTYPES, DIMCOUNTS, DIMTYPES> {
    typedef DynamicTemplates::ListProduct4<DIMCOUNTS, DIMTYPES, DIMCOUNTS, DIMTYPES>  SUPER;
    template <int N, typename T, int N2, typename T2>
    static DynamicTemplates::TagType encode_tag(void) {
      return SUPER::template encode_tag<DynamicTemplates::Int<N>, T,
 	                                DynamicTemplates::Int<N2>, T2>();
    }
  };

}; // namespace Realm

#if REALM_MAX_DIM == 1

#define FOREACH_N(__func__) \
  __func__(1) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \

#elif REALM_MAX_DIM == 2

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \

#elif REALM_MAX_DIM == 3

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \
  __func__(3,int) \
  __func__(3,unsigned) \
  __func__(3,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \
  __func__(3,int,int) \
  __func__(3,int,bool) \
  __func__(3,unsigned,int) \
  __func__(3,unsigned,bool) \
  __func__(3,long long,int) \
  __func__(3,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,int,3,int) \
  __func__(1,int,3,unsigned) \
  __func__(1,int,3,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,unsigned,3,int) \
  __func__(1,unsigned,3,unsigned) \
  __func__(1,unsigned,3,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
  __func__(1,long long,3,int) \
  __func__(1,long long,3,unsigned) \
  __func__(1,long long,3,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,int,3,int) \
  __func__(2,int,3,unsigned) \
  __func__(2,int,3,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,unsigned,3,int) \
  __func__(2,unsigned,3,unsigned) \
  __func__(2,unsigned,3,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \
  __func__(2,long long,3,int) \
  __func__(2,long long,3,unsigned) \
  __func__(2,long long,3,long long) \
\
  __func__(3,int,1,int) \
  __func__(3,int,1,unsigned) \
  __func__(3,int,1,long long) \
  __func__(3,int,2,int) \
  __func__(3,int,2,unsigned) \
  __func__(3,int,2,long long) \
  __func__(3,int,3,int) \
  __func__(3,int,3,unsigned) \
  __func__(3,int,3,long long) \
  __func__(3,unsigned,1,int) \
  __func__(3,unsigned,1,unsigned) \
  __func__(3,unsigned,1,long long) \
  __func__(3,unsigned,2,int) \
  __func__(3,unsigned,2,unsigned) \
  __func__(3,unsigned,2,long long) \
  __func__(3,unsigned,3,int) \
  __func__(3,unsigned,3,unsigned) \
  __func__(3,unsigned,3,long long) \
  __func__(3,long long,1,int) \
  __func__(3,long long,1,unsigned) \
  __func__(3,long long,1,long long) \
  __func__(3,long long,2,int) \
  __func__(3,long long,2,unsigned) \
  __func__(3,long long,2,long long) \
  __func__(3,long long,3,int) \
  __func__(3,long long,3,unsigned) \
  __func__(3,long long,3,long long) \

#elif REALM_MAX_DIM == 4

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \
  __func__(3,int) \
  __func__(3,unsigned) \
  __func__(3,long long) \
  __func__(4,int) \
  __func__(4,unsigned) \
  __func__(4,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \
  __func__(3,int,int) \
  __func__(3,int,bool) \
  __func__(3,unsigned,int) \
  __func__(3,unsigned,bool) \
  __func__(3,long long,int) \
  __func__(3,long long,bool) \
  __func__(4,int,int) \
  __func__(4,int,bool) \
  __func__(4,unsigned,int) \
  __func__(4,unsigned,bool) \
  __func__(4,long long,int) \
  __func__(4,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,int,3,int) \
  __func__(1,int,3,unsigned) \
  __func__(1,int,3,long long) \
  __func__(1,int,4,int) \
  __func__(1,int,4,unsigned) \
  __func__(1,int,4,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,unsigned,3,int) \
  __func__(1,unsigned,3,unsigned) \
  __func__(1,unsigned,3,long long) \
  __func__(1,unsigned,4,int) \
  __func__(1,unsigned,4,unsigned) \
  __func__(1,unsigned,4,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
  __func__(1,long long,3,int) \
  __func__(1,long long,3,unsigned) \
  __func__(1,long long,3,long long) \
  __func__(1,long long,4,int) \
  __func__(1,long long,4,unsigned) \
  __func__(1,long long,4,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,int,3,int) \
  __func__(2,int,3,unsigned) \
  __func__(2,int,3,long long) \
  __func__(2,int,4,int) \
  __func__(2,int,4,unsigned) \
  __func__(2,int,4,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,unsigned,3,int) \
  __func__(2,unsigned,3,unsigned) \
  __func__(2,unsigned,3,long long) \
  __func__(2,unsigned,4,int) \
  __func__(2,unsigned,4,unsigned) \
  __func__(2,unsigned,4,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \
  __func__(2,long long,3,int) \
  __func__(2,long long,3,unsigned) \
  __func__(2,long long,3,long long) \
  __func__(2,long long,4,int) \
  __func__(2,long long,4,unsigned) \
  __func__(2,long long,4,long long) \
\
  __func__(3,int,1,int) \
  __func__(3,int,1,unsigned) \
  __func__(3,int,1,long long) \
  __func__(3,int,2,int) \
  __func__(3,int,2,unsigned) \
  __func__(3,int,2,long long) \
  __func__(3,int,3,int) \
  __func__(3,int,3,unsigned) \
  __func__(3,int,3,long long) \
  __func__(3,int,4,int) \
  __func__(3,int,4,unsigned) \
  __func__(3,int,4,long long) \
  __func__(3,unsigned,1,int) \
  __func__(3,unsigned,1,unsigned) \
  __func__(3,unsigned,1,long long) \
  __func__(3,unsigned,2,int) \
  __func__(3,unsigned,2,unsigned) \
  __func__(3,unsigned,2,long long) \
  __func__(3,unsigned,3,int) \
  __func__(3,unsigned,3,unsigned) \
  __func__(3,unsigned,3,long long) \
  __func__(3,unsigned,4,int) \
  __func__(3,unsigned,4,unsigned) \
  __func__(3,unsigned,4,long long) \
  __func__(3,long long,1,int) \
  __func__(3,long long,1,unsigned) \
  __func__(3,long long,1,long long) \
  __func__(3,long long,2,int) \
  __func__(3,long long,2,unsigned) \
  __func__(3,long long,2,long long) \
  __func__(3,long long,3,int) \
  __func__(3,long long,3,unsigned) \
  __func__(3,long long,3,long long) \
  __func__(3,long long,4,int) \
  __func__(3,long long,4,unsigned) \
  __func__(3,long long,4,long long) \
\
  __func__(4,int,1,int) \
  __func__(4,int,1,unsigned) \
  __func__(4,int,1,long long) \
  __func__(4,int,2,int) \
  __func__(4,int,2,unsigned) \
  __func__(4,int,2,long long) \
  __func__(4,int,3,int) \
  __func__(4,int,3,unsigned) \
  __func__(4,int,3,long long) \
  __func__(4,int,4,int) \
  __func__(4,int,4,unsigned) \
  __func__(4,int,4,long long) \
  __func__(4,unsigned,1,int) \
  __func__(4,unsigned,1,unsigned) \
  __func__(4,unsigned,1,long long) \
  __func__(4,unsigned,2,int) \
  __func__(4,unsigned,2,unsigned) \
  __func__(4,unsigned,2,long long) \
  __func__(4,unsigned,3,int) \
  __func__(4,unsigned,3,unsigned) \
  __func__(4,unsigned,3,long long) \
  __func__(4,unsigned,4,int) \
  __func__(4,unsigned,4,unsigned) \
  __func__(4,unsigned,4,long long) \
  __func__(4,long long,1,int) \
  __func__(4,long long,1,unsigned) \
  __func__(4,long long,1,long long) \
  __func__(4,long long,2,int) \
  __func__(4,long long,2,unsigned) \
  __func__(4,long long,2,long long) \
  __func__(4,long long,3,int) \
  __func__(4,long long,3,unsigned) \
  __func__(4,long long,3,long long) \
  __func__(4,long long,4,int) \
  __func__(4,long long,4,unsigned) \
  __func__(4,long long,4,long long) \

#elif REALM_MAX_DIM == 5

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \
  __func__(3,int) \
  __func__(3,unsigned) \
  __func__(3,long long) \
  __func__(4,int) \
  __func__(4,unsigned) \
  __func__(4,long long) \
  __func__(5,int) \
  __func__(5,unsigned) \
  __func__(5,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \
  __func__(3,int,int) \
  __func__(3,int,bool) \
  __func__(3,unsigned,int) \
  __func__(3,unsigned,bool) \
  __func__(3,long long,int) \
  __func__(3,long long,bool) \
  __func__(4,int,int) \
  __func__(4,int,bool) \
  __func__(4,unsigned,int) \
  __func__(4,unsigned,bool) \
  __func__(4,long long,int) \
  __func__(4,long long,bool) \
  __func__(5,int,int) \
  __func__(5,int,bool) \
  __func__(5,unsigned,int) \
  __func__(5,unsigned,bool) \
  __func__(5,long long,int) \
  __func__(5,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,int,3,int) \
  __func__(1,int,3,unsigned) \
  __func__(1,int,3,long long) \
  __func__(1,int,4,int) \
  __func__(1,int,4,unsigned) \
  __func__(1,int,4,long long) \
  __func__(1,int,5,int) \
  __func__(1,int,5,unsigned) \
  __func__(1,int,5,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,unsigned,3,int) \
  __func__(1,unsigned,3,unsigned) \
  __func__(1,unsigned,3,long long) \
  __func__(1,unsigned,4,int) \
  __func__(1,unsigned,4,unsigned) \
  __func__(1,unsigned,4,long long) \
  __func__(1,unsigned,5,int) \
  __func__(1,unsigned,5,unsigned) \
  __func__(1,unsigned,5,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
  __func__(1,long long,3,int) \
  __func__(1,long long,3,unsigned) \
  __func__(1,long long,3,long long) \
  __func__(1,long long,4,int) \
  __func__(1,long long,4,unsigned) \
  __func__(1,long long,4,long long) \
  __func__(1,long long,5,int) \
  __func__(1,long long,5,unsigned) \
  __func__(1,long long,5,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,int,3,int) \
  __func__(2,int,3,unsigned) \
  __func__(2,int,3,long long) \
  __func__(2,int,4,int) \
  __func__(2,int,4,unsigned) \
  __func__(2,int,4,long long) \
  __func__(2,int,5,int) \
  __func__(2,int,5,unsigned) \
  __func__(2,int,5,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,unsigned,3,int) \
  __func__(2,unsigned,3,unsigned) \
  __func__(2,unsigned,3,long long) \
  __func__(2,unsigned,4,int) \
  __func__(2,unsigned,4,unsigned) \
  __func__(2,unsigned,4,long long) \
  __func__(2,unsigned,5,int) \
  __func__(2,unsigned,5,unsigned) \
  __func__(2,unsigned,5,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \
  __func__(2,long long,3,int) \
  __func__(2,long long,3,unsigned) \
  __func__(2,long long,3,long long) \
  __func__(2,long long,4,int) \
  __func__(2,long long,4,unsigned) \
  __func__(2,long long,4,long long) \
  __func__(2,long long,5,int) \
  __func__(2,long long,5,unsigned) \
  __func__(2,long long,5,long long) \
\
  __func__(3,int,1,int) \
  __func__(3,int,1,unsigned) \
  __func__(3,int,1,long long) \
  __func__(3,int,2,int) \
  __func__(3,int,2,unsigned) \
  __func__(3,int,2,long long) \
  __func__(3,int,3,int) \
  __func__(3,int,3,unsigned) \
  __func__(3,int,3,long long) \
  __func__(3,int,4,int) \
  __func__(3,int,4,unsigned) \
  __func__(3,int,4,long long) \
  __func__(3,int,5,int) \
  __func__(3,int,5,unsigned) \
  __func__(3,int,5,long long) \
  __func__(3,unsigned,1,int) \
  __func__(3,unsigned,1,unsigned) \
  __func__(3,unsigned,1,long long) \
  __func__(3,unsigned,2,int) \
  __func__(3,unsigned,2,unsigned) \
  __func__(3,unsigned,2,long long) \
  __func__(3,unsigned,3,int) \
  __func__(3,unsigned,3,unsigned) \
  __func__(3,unsigned,3,long long) \
  __func__(3,unsigned,4,int) \
  __func__(3,unsigned,4,unsigned) \
  __func__(3,unsigned,4,long long) \
  __func__(3,unsigned,5,int) \
  __func__(3,unsigned,5,unsigned) \
  __func__(3,unsigned,5,long long) \
  __func__(3,long long,1,int) \
  __func__(3,long long,1,unsigned) \
  __func__(3,long long,1,long long) \
  __func__(3,long long,2,int) \
  __func__(3,long long,2,unsigned) \
  __func__(3,long long,2,long long) \
  __func__(3,long long,3,int) \
  __func__(3,long long,3,unsigned) \
  __func__(3,long long,3,long long) \
  __func__(3,long long,4,int) \
  __func__(3,long long,4,unsigned) \
  __func__(3,long long,4,long long) \
  __func__(3,long long,5,int) \
  __func__(3,long long,5,unsigned) \
  __func__(3,long long,5,long long) \
\
  __func__(4,int,1,int) \
  __func__(4,int,1,unsigned) \
  __func__(4,int,1,long long) \
  __func__(4,int,2,int) \
  __func__(4,int,2,unsigned) \
  __func__(4,int,2,long long) \
  __func__(4,int,3,int) \
  __func__(4,int,3,unsigned) \
  __func__(4,int,3,long long) \
  __func__(4,int,4,int) \
  __func__(4,int,4,unsigned) \
  __func__(4,int,4,long long) \
  __func__(4,int,5,int) \
  __func__(4,int,5,unsigned) \
  __func__(4,int,5,long long) \
  __func__(4,unsigned,1,int) \
  __func__(4,unsigned,1,unsigned) \
  __func__(4,unsigned,1,long long) \
  __func__(4,unsigned,2,int) \
  __func__(4,unsigned,2,unsigned) \
  __func__(4,unsigned,2,long long) \
  __func__(4,unsigned,3,int) \
  __func__(4,unsigned,3,unsigned) \
  __func__(4,unsigned,3,long long) \
  __func__(4,unsigned,4,int) \
  __func__(4,unsigned,4,unsigned) \
  __func__(4,unsigned,4,long long) \
  __func__(4,unsigned,5,int) \
  __func__(4,unsigned,5,unsigned) \
  __func__(4,unsigned,5,long long) \
  __func__(4,long long,1,int) \
  __func__(4,long long,1,unsigned) \
  __func__(4,long long,1,long long) \
  __func__(4,long long,2,int) \
  __func__(4,long long,2,unsigned) \
  __func__(4,long long,2,long long) \
  __func__(4,long long,3,int) \
  __func__(4,long long,3,unsigned) \
  __func__(4,long long,3,long long) \
  __func__(4,long long,4,int) \
  __func__(4,long long,4,unsigned) \
  __func__(4,long long,4,long long) \
  __func__(4,long long,5,int) \
  __func__(4,long long,5,unsigned) \
  __func__(4,long long,5,long long) \
\
  __func__(5,int,1,int) \
  __func__(5,int,1,unsigned) \
  __func__(5,int,1,long long) \
  __func__(5,int,2,int) \
  __func__(5,int,2,unsigned) \
  __func__(5,int,2,long long) \
  __func__(5,int,3,int) \
  __func__(5,int,3,unsigned) \
  __func__(5,int,3,long long) \
  __func__(5,int,4,int) \
  __func__(5,int,4,unsigned) \
  __func__(5,int,4,long long) \
  __func__(5,int,5,int) \
  __func__(5,int,5,unsigned) \
  __func__(5,int,5,long long) \
  __func__(5,unsigned,1,int) \
  __func__(5,unsigned,1,unsigned) \
  __func__(5,unsigned,1,long long) \
  __func__(5,unsigned,2,int) \
  __func__(5,unsigned,2,unsigned) \
  __func__(5,unsigned,2,long long) \
  __func__(5,unsigned,3,int) \
  __func__(5,unsigned,3,unsigned) \
  __func__(5,unsigned,3,long long) \
  __func__(5,unsigned,4,int) \
  __func__(5,unsigned,4,unsigned) \
  __func__(5,unsigned,4,long long) \
  __func__(5,unsigned,5,int) \
  __func__(5,unsigned,5,unsigned) \
  __func__(5,unsigned,5,long long) \
  __func__(5,long long,1,int) \
  __func__(5,long long,1,unsigned) \
  __func__(5,long long,1,long long) \
  __func__(5,long long,2,int) \
  __func__(5,long long,2,unsigned) \
  __func__(5,long long,2,long long) \
  __func__(5,long long,3,int) \
  __func__(5,long long,3,unsigned) \
  __func__(5,long long,3,long long) \
  __func__(5,long long,4,int) \
  __func__(5,long long,4,unsigned) \
  __func__(5,long long,4,long long) \
  __func__(5,long long,5,int) \
  __func__(5,long long,5,unsigned) \
  __func__(5,long long,5,long long) \

#elif REALM_MAX_DIM == 6

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \
  __func__(3,int) \
  __func__(3,unsigned) \
  __func__(3,long long) \
  __func__(4,int) \
  __func__(4,unsigned) \
  __func__(4,long long) \
  __func__(5,int) \
  __func__(5,unsigned) \
  __func__(5,long long) \
  __func__(6,int) \
  __func__(6,unsigned) \
  __func__(6,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \
  __func__(3,int,int) \
  __func__(3,int,bool) \
  __func__(3,unsigned,int) \
  __func__(3,unsigned,bool) \
  __func__(3,long long,int) \
  __func__(3,long long,bool) \
  __func__(4,int,int) \
  __func__(4,int,bool) \
  __func__(4,unsigned,int) \
  __func__(4,unsigned,bool) \
  __func__(4,long long,int) \
  __func__(4,long long,bool) \
  __func__(5,int,int) \
  __func__(5,int,bool) \
  __func__(5,unsigned,int) \
  __func__(5,unsigned,bool) \
  __func__(5,long long,int) \
  __func__(5,long long,bool) \
  __func__(6,int,int) \
  __func__(6,int,bool) \
  __func__(6,unsigned,int) \
  __func__(6,unsigned,bool) \
  __func__(6,long long,int) \
  __func__(6,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,int,3,int) \
  __func__(1,int,3,unsigned) \
  __func__(1,int,3,long long) \
  __func__(1,int,4,int) \
  __func__(1,int,4,unsigned) \
  __func__(1,int,4,long long) \
  __func__(1,int,5,int) \
  __func__(1,int,5,unsigned) \
  __func__(1,int,5,long long) \
  __func__(1,int,6,int) \
  __func__(1,int,6,unsigned) \
  __func__(1,int,6,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,unsigned,3,int) \
  __func__(1,unsigned,3,unsigned) \
  __func__(1,unsigned,3,long long) \
  __func__(1,unsigned,4,int) \
  __func__(1,unsigned,4,unsigned) \
  __func__(1,unsigned,4,long long) \
  __func__(1,unsigned,5,int) \
  __func__(1,unsigned,5,unsigned) \
  __func__(1,unsigned,5,long long) \
  __func__(1,unsigned,6,int) \
  __func__(1,unsigned,6,unsigned) \
  __func__(1,unsigned,6,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
  __func__(1,long long,3,int) \
  __func__(1,long long,3,unsigned) \
  __func__(1,long long,3,long long) \
  __func__(1,long long,4,int) \
  __func__(1,long long,4,unsigned) \
  __func__(1,long long,4,long long) \
  __func__(1,long long,5,int) \
  __func__(1,long long,5,unsigned) \
  __func__(1,long long,5,long long) \
  __func__(1,long long,6,int) \
  __func__(1,long long,6,unsigned) \
  __func__(1,long long,6,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,int,3,int) \
  __func__(2,int,3,unsigned) \
  __func__(2,int,3,long long) \
  __func__(2,int,4,int) \
  __func__(2,int,4,unsigned) \
  __func__(2,int,4,long long) \
  __func__(2,int,5,int) \
  __func__(2,int,5,unsigned) \
  __func__(2,int,5,long long) \
  __func__(2,int,6,int) \
  __func__(2,int,6,unsigned) \
  __func__(2,int,6,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,unsigned,3,int) \
  __func__(2,unsigned,3,unsigned) \
  __func__(2,unsigned,3,long long) \
  __func__(2,unsigned,4,int) \
  __func__(2,unsigned,4,unsigned) \
  __func__(2,unsigned,4,long long) \
  __func__(2,unsigned,5,int) \
  __func__(2,unsigned,5,unsigned) \
  __func__(2,unsigned,5,long long) \
  __func__(2,unsigned,6,int) \
  __func__(2,unsigned,6,unsigned) \
  __func__(2,unsigned,6,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \
  __func__(2,long long,3,int) \
  __func__(2,long long,3,unsigned) \
  __func__(2,long long,3,long long) \
  __func__(2,long long,4,int) \
  __func__(2,long long,4,unsigned) \
  __func__(2,long long,4,long long) \
  __func__(2,long long,5,int) \
  __func__(2,long long,5,unsigned) \
  __func__(2,long long,5,long long) \
  __func__(2,long long,6,int) \
  __func__(2,long long,6,unsigned) \
  __func__(2,long long,6,long long) \
\
  __func__(3,int,1,int) \
  __func__(3,int,1,unsigned) \
  __func__(3,int,1,long long) \
  __func__(3,int,2,int) \
  __func__(3,int,2,unsigned) \
  __func__(3,int,2,long long) \
  __func__(3,int,3,int) \
  __func__(3,int,3,unsigned) \
  __func__(3,int,3,long long) \
  __func__(3,int,4,int) \
  __func__(3,int,4,unsigned) \
  __func__(3,int,4,long long) \
  __func__(3,int,5,int) \
  __func__(3,int,5,unsigned) \
  __func__(3,int,5,long long) \
  __func__(3,int,6,int) \
  __func__(3,int,6,unsigned) \
  __func__(3,int,6,long long) \
  __func__(3,unsigned,1,int) \
  __func__(3,unsigned,1,unsigned) \
  __func__(3,unsigned,1,long long) \
  __func__(3,unsigned,2,int) \
  __func__(3,unsigned,2,unsigned) \
  __func__(3,unsigned,2,long long) \
  __func__(3,unsigned,3,int) \
  __func__(3,unsigned,3,unsigned) \
  __func__(3,unsigned,3,long long) \
  __func__(3,unsigned,4,int) \
  __func__(3,unsigned,4,unsigned) \
  __func__(3,unsigned,4,long long) \
  __func__(3,unsigned,5,int) \
  __func__(3,unsigned,5,unsigned) \
  __func__(3,unsigned,5,long long) \
  __func__(3,unsigned,6,int) \
  __func__(3,unsigned,6,unsigned) \
  __func__(3,unsigned,6,long long) \
  __func__(3,long long,1,int) \
  __func__(3,long long,1,unsigned) \
  __func__(3,long long,1,long long) \
  __func__(3,long long,2,int) \
  __func__(3,long long,2,unsigned) \
  __func__(3,long long,2,long long) \
  __func__(3,long long,3,int) \
  __func__(3,long long,3,unsigned) \
  __func__(3,long long,3,long long) \
  __func__(3,long long,4,int) \
  __func__(3,long long,4,unsigned) \
  __func__(3,long long,4,long long) \
  __func__(3,long long,5,int) \
  __func__(3,long long,5,unsigned) \
  __func__(3,long long,5,long long) \
  __func__(3,long long,6,int) \
  __func__(3,long long,6,unsigned) \
  __func__(3,long long,6,long long) \
\
  __func__(4,int,1,int) \
  __func__(4,int,1,unsigned) \
  __func__(4,int,1,long long) \
  __func__(4,int,2,int) \
  __func__(4,int,2,unsigned) \
  __func__(4,int,2,long long) \
  __func__(4,int,3,int) \
  __func__(4,int,3,unsigned) \
  __func__(4,int,3,long long) \
  __func__(4,int,4,int) \
  __func__(4,int,4,unsigned) \
  __func__(4,int,4,long long) \
  __func__(4,int,5,int) \
  __func__(4,int,5,unsigned) \
  __func__(4,int,5,long long) \
  __func__(4,int,6,int) \
  __func__(4,int,6,unsigned) \
  __func__(4,int,6,long long) \
  __func__(4,unsigned,1,int) \
  __func__(4,unsigned,1,unsigned) \
  __func__(4,unsigned,1,long long) \
  __func__(4,unsigned,2,int) \
  __func__(4,unsigned,2,unsigned) \
  __func__(4,unsigned,2,long long) \
  __func__(4,unsigned,3,int) \
  __func__(4,unsigned,3,unsigned) \
  __func__(4,unsigned,3,long long) \
  __func__(4,unsigned,4,int) \
  __func__(4,unsigned,4,unsigned) \
  __func__(4,unsigned,4,long long) \
  __func__(4,unsigned,5,int) \
  __func__(4,unsigned,5,unsigned) \
  __func__(4,unsigned,5,long long) \
  __func__(4,unsigned,6,int) \
  __func__(4,unsigned,6,unsigned) \
  __func__(4,unsigned,6,long long) \
  __func__(4,long long,1,int) \
  __func__(4,long long,1,unsigned) \
  __func__(4,long long,1,long long) \
  __func__(4,long long,2,int) \
  __func__(4,long long,2,unsigned) \
  __func__(4,long long,2,long long) \
  __func__(4,long long,3,int) \
  __func__(4,long long,3,unsigned) \
  __func__(4,long long,3,long long) \
  __func__(4,long long,4,int) \
  __func__(4,long long,4,unsigned) \
  __func__(4,long long,4,long long) \
  __func__(4,long long,5,int) \
  __func__(4,long long,5,unsigned) \
  __func__(4,long long,5,long long) \
  __func__(4,long long,6,int) \
  __func__(4,long long,6,unsigned) \
  __func__(4,long long,6,long long) \
\
  __func__(5,int,1,int) \
  __func__(5,int,1,unsigned) \
  __func__(5,int,1,long long) \
  __func__(5,int,2,int) \
  __func__(5,int,2,unsigned) \
  __func__(5,int,2,long long) \
  __func__(5,int,3,int) \
  __func__(5,int,3,unsigned) \
  __func__(5,int,3,long long) \
  __func__(5,int,4,int) \
  __func__(5,int,4,unsigned) \
  __func__(5,int,4,long long) \
  __func__(5,int,5,int) \
  __func__(5,int,5,unsigned) \
  __func__(5,int,5,long long) \
  __func__(5,int,6,int) \
  __func__(5,int,6,unsigned) \
  __func__(5,int,6,long long) \
  __func__(5,unsigned,1,int) \
  __func__(5,unsigned,1,unsigned) \
  __func__(5,unsigned,1,long long) \
  __func__(5,unsigned,2,int) \
  __func__(5,unsigned,2,unsigned) \
  __func__(5,unsigned,2,long long) \
  __func__(5,unsigned,3,int) \
  __func__(5,unsigned,3,unsigned) \
  __func__(5,unsigned,3,long long) \
  __func__(5,unsigned,4,int) \
  __func__(5,unsigned,4,unsigned) \
  __func__(5,unsigned,4,long long) \
  __func__(5,unsigned,5,int) \
  __func__(5,unsigned,5,unsigned) \
  __func__(5,unsigned,5,long long) \
  __func__(5,unsigned,6,int) \
  __func__(5,unsigned,6,unsigned) \
  __func__(5,unsigned,6,long long) \
  __func__(5,long long,1,int) \
  __func__(5,long long,1,unsigned) \
  __func__(5,long long,1,long long) \
  __func__(5,long long,2,int) \
  __func__(5,long long,2,unsigned) \
  __func__(5,long long,2,long long) \
  __func__(5,long long,3,int) \
  __func__(5,long long,3,unsigned) \
  __func__(5,long long,3,long long) \
  __func__(5,long long,4,int) \
  __func__(5,long long,4,unsigned) \
  __func__(5,long long,4,long long) \
  __func__(5,long long,5,int) \
  __func__(5,long long,5,unsigned) \
  __func__(5,long long,5,long long) \
  __func__(5,long long,6,int) \
  __func__(5,long long,6,unsigned) \
  __func__(5,long long,6,long long) \
\
  __func__(6,int,1,int) \
  __func__(6,int,1,unsigned) \
  __func__(6,int,1,long long) \
  __func__(6,int,2,int) \
  __func__(6,int,2,unsigned) \
  __func__(6,int,2,long long) \
  __func__(6,int,3,int) \
  __func__(6,int,3,unsigned) \
  __func__(6,int,3,long long) \
  __func__(6,int,4,int) \
  __func__(6,int,4,unsigned) \
  __func__(6,int,4,long long) \
  __func__(6,int,5,int) \
  __func__(6,int,5,unsigned) \
  __func__(6,int,5,long long) \
  __func__(6,int,6,int) \
  __func__(6,int,6,unsigned) \
  __func__(6,int,6,long long) \
  __func__(6,unsigned,1,int) \
  __func__(6,unsigned,1,unsigned) \
  __func__(6,unsigned,1,long long) \
  __func__(6,unsigned,2,int) \
  __func__(6,unsigned,2,unsigned) \
  __func__(6,unsigned,2,long long) \
  __func__(6,unsigned,3,int) \
  __func__(6,unsigned,3,unsigned) \
  __func__(6,unsigned,3,long long) \
  __func__(6,unsigned,4,int) \
  __func__(6,unsigned,4,unsigned) \
  __func__(6,unsigned,4,long long) \
  __func__(6,unsigned,5,int) \
  __func__(6,unsigned,5,unsigned) \
  __func__(6,unsigned,5,long long) \
  __func__(6,unsigned,6,int) \
  __func__(6,unsigned,6,unsigned) \
  __func__(6,unsigned,6,long long) \
  __func__(6,long long,1,int) \
  __func__(6,long long,1,unsigned) \
  __func__(6,long long,1,long long) \
  __func__(6,long long,2,int) \
  __func__(6,long long,2,unsigned) \
  __func__(6,long long,2,long long) \
  __func__(6,long long,3,int) \
  __func__(6,long long,3,unsigned) \
  __func__(6,long long,3,long long) \
  __func__(6,long long,4,int) \
  __func__(6,long long,4,unsigned) \
  __func__(6,long long,4,long long) \
  __func__(6,long long,5,int) \
  __func__(6,long long,5,unsigned) \
  __func__(6,long long,5,long long) \
  __func__(6,long long,6,int) \
  __func__(6,long long,6,unsigned) \
  __func__(6,long long,6,long long) \

#elif REALM_MAX_DIM == 7

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6) \
  __func__(7) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \
  __func__(3,int) \
  __func__(3,unsigned) \
  __func__(3,long long) \
  __func__(4,int) \
  __func__(4,unsigned) \
  __func__(4,long long) \
  __func__(5,int) \
  __func__(5,unsigned) \
  __func__(5,long long) \
  __func__(6,int) \
  __func__(6,unsigned) \
  __func__(6,long long) \
  __func__(7,int) \
  __func__(7,unsigned) \
  __func__(7,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \
  __func__(3,int,int) \
  __func__(3,int,bool) \
  __func__(3,unsigned,int) \
  __func__(3,unsigned,bool) \
  __func__(3,long long,int) \
  __func__(3,long long,bool) \
  __func__(4,int,int) \
  __func__(4,int,bool) \
  __func__(4,unsigned,int) \
  __func__(4,unsigned,bool) \
  __func__(4,long long,int) \
  __func__(4,long long,bool) \
  __func__(5,int,int) \
  __func__(5,int,bool) \
  __func__(5,unsigned,int) \
  __func__(5,unsigned,bool) \
  __func__(5,long long,int) \
  __func__(5,long long,bool) \
  __func__(6,int,int) \
  __func__(6,int,bool) \
  __func__(6,unsigned,int) \
  __func__(6,unsigned,bool) \
  __func__(6,long long,int) \
  __func__(6,long long,bool) \
  __func__(7,int,int) \
  __func__(7,int,bool) \
  __func__(7,unsigned,int) \
  __func__(7,unsigned,bool) \
  __func__(7,long long,int) \
  __func__(7,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,int,3,int) \
  __func__(1,int,3,unsigned) \
  __func__(1,int,3,long long) \
  __func__(1,int,4,int) \
  __func__(1,int,4,unsigned) \
  __func__(1,int,4,long long) \
  __func__(1,int,5,int) \
  __func__(1,int,5,unsigned) \
  __func__(1,int,5,long long) \
  __func__(1,int,6,int) \
  __func__(1,int,6,unsigned) \
  __func__(1,int,6,long long) \
  __func__(1,int,7,int) \
  __func__(1,int,7,unsigned) \
  __func__(1,int,7,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,unsigned,3,int) \
  __func__(1,unsigned,3,unsigned) \
  __func__(1,unsigned,3,long long) \
  __func__(1,unsigned,4,int) \
  __func__(1,unsigned,4,unsigned) \
  __func__(1,unsigned,4,long long) \
  __func__(1,unsigned,5,int) \
  __func__(1,unsigned,5,unsigned) \
  __func__(1,unsigned,5,long long) \
  __func__(1,unsigned,6,int) \
  __func__(1,unsigned,6,unsigned) \
  __func__(1,unsigned,6,long long) \
  __func__(1,unsigned,7,int) \
  __func__(1,unsigned,7,unsigned) \
  __func__(1,unsigned,7,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
  __func__(1,long long,3,int) \
  __func__(1,long long,3,unsigned) \
  __func__(1,long long,3,long long) \
  __func__(1,long long,4,int) \
  __func__(1,long long,4,unsigned) \
  __func__(1,long long,4,long long) \
  __func__(1,long long,5,int) \
  __func__(1,long long,5,unsigned) \
  __func__(1,long long,5,long long) \
  __func__(1,long long,6,int) \
  __func__(1,long long,6,unsigned) \
  __func__(1,long long,6,long long) \
  __func__(1,long long,7,int) \
  __func__(1,long long,7,unsigned) \
  __func__(1,long long,7,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,int,3,int) \
  __func__(2,int,3,unsigned) \
  __func__(2,int,3,long long) \
  __func__(2,int,4,int) \
  __func__(2,int,4,unsigned) \
  __func__(2,int,4,long long) \
  __func__(2,int,5,int) \
  __func__(2,int,5,unsigned) \
  __func__(2,int,5,long long) \
  __func__(2,int,6,int) \
  __func__(2,int,6,unsigned) \
  __func__(2,int,6,long long) \
  __func__(2,int,7,int) \
  __func__(2,int,7,unsigned) \
  __func__(2,int,7,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,unsigned,3,int) \
  __func__(2,unsigned,3,unsigned) \
  __func__(2,unsigned,3,long long) \
  __func__(2,unsigned,4,int) \
  __func__(2,unsigned,4,unsigned) \
  __func__(2,unsigned,4,long long) \
  __func__(2,unsigned,5,int) \
  __func__(2,unsigned,5,unsigned) \
  __func__(2,unsigned,5,long long) \
  __func__(2,unsigned,6,int) \
  __func__(2,unsigned,6,unsigned) \
  __func__(2,unsigned,6,long long) \
  __func__(2,unsigned,7,int) \
  __func__(2,unsigned,7,unsigned) \
  __func__(2,unsigned,7,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \
  __func__(2,long long,3,int) \
  __func__(2,long long,3,unsigned) \
  __func__(2,long long,3,long long) \
  __func__(2,long long,4,int) \
  __func__(2,long long,4,unsigned) \
  __func__(2,long long,4,long long) \
  __func__(2,long long,5,int) \
  __func__(2,long long,5,unsigned) \
  __func__(2,long long,5,long long) \
  __func__(2,long long,6,int) \
  __func__(2,long long,6,unsigned) \
  __func__(2,long long,6,long long) \
  __func__(2,long long,7,int) \
  __func__(2,long long,7,unsigned) \
  __func__(2,long long,7,long long) \
\
  __func__(3,int,1,int) \
  __func__(3,int,1,unsigned) \
  __func__(3,int,1,long long) \
  __func__(3,int,2,int) \
  __func__(3,int,2,unsigned) \
  __func__(3,int,2,long long) \
  __func__(3,int,3,int) \
  __func__(3,int,3,unsigned) \
  __func__(3,int,3,long long) \
  __func__(3,int,4,int) \
  __func__(3,int,4,unsigned) \
  __func__(3,int,4,long long) \
  __func__(3,int,5,int) \
  __func__(3,int,5,unsigned) \
  __func__(3,int,5,long long) \
  __func__(3,int,6,int) \
  __func__(3,int,6,unsigned) \
  __func__(3,int,6,long long) \
  __func__(3,int,7,int) \
  __func__(3,int,7,unsigned) \
  __func__(3,int,7,long long) \
  __func__(3,unsigned,1,int) \
  __func__(3,unsigned,1,unsigned) \
  __func__(3,unsigned,1,long long) \
  __func__(3,unsigned,2,int) \
  __func__(3,unsigned,2,unsigned) \
  __func__(3,unsigned,2,long long) \
  __func__(3,unsigned,3,int) \
  __func__(3,unsigned,3,unsigned) \
  __func__(3,unsigned,3,long long) \
  __func__(3,unsigned,4,int) \
  __func__(3,unsigned,4,unsigned) \
  __func__(3,unsigned,4,long long) \
  __func__(3,unsigned,5,int) \
  __func__(3,unsigned,5,unsigned) \
  __func__(3,unsigned,5,long long) \
  __func__(3,unsigned,6,int) \
  __func__(3,unsigned,6,unsigned) \
  __func__(3,unsigned,6,long long) \
  __func__(3,unsigned,7,int) \
  __func__(3,unsigned,7,unsigned) \
  __func__(3,unsigned,7,long long) \
  __func__(3,long long,1,int) \
  __func__(3,long long,1,unsigned) \
  __func__(3,long long,1,long long) \
  __func__(3,long long,2,int) \
  __func__(3,long long,2,unsigned) \
  __func__(3,long long,2,long long) \
  __func__(3,long long,3,int) \
  __func__(3,long long,3,unsigned) \
  __func__(3,long long,3,long long) \
  __func__(3,long long,4,int) \
  __func__(3,long long,4,unsigned) \
  __func__(3,long long,4,long long) \
  __func__(3,long long,5,int) \
  __func__(3,long long,5,unsigned) \
  __func__(3,long long,5,long long) \
  __func__(3,long long,6,int) \
  __func__(3,long long,6,unsigned) \
  __func__(3,long long,6,long long) \
  __func__(3,long long,7,int) \
  __func__(3,long long,7,unsigned) \
  __func__(3,long long,7,long long) \
\
  __func__(4,int,1,int) \
  __func__(4,int,1,unsigned) \
  __func__(4,int,1,long long) \
  __func__(4,int,2,int) \
  __func__(4,int,2,unsigned) \
  __func__(4,int,2,long long) \
  __func__(4,int,3,int) \
  __func__(4,int,3,unsigned) \
  __func__(4,int,3,long long) \
  __func__(4,int,4,int) \
  __func__(4,int,4,unsigned) \
  __func__(4,int,4,long long) \
  __func__(4,int,5,int) \
  __func__(4,int,5,unsigned) \
  __func__(4,int,5,long long) \
  __func__(4,int,6,int) \
  __func__(4,int,6,unsigned) \
  __func__(4,int,6,long long) \
  __func__(4,int,7,int) \
  __func__(4,int,7,unsigned) \
  __func__(4,int,7,long long) \
  __func__(4,unsigned,1,int) \
  __func__(4,unsigned,1,unsigned) \
  __func__(4,unsigned,1,long long) \
  __func__(4,unsigned,2,int) \
  __func__(4,unsigned,2,unsigned) \
  __func__(4,unsigned,2,long long) \
  __func__(4,unsigned,3,int) \
  __func__(4,unsigned,3,unsigned) \
  __func__(4,unsigned,3,long long) \
  __func__(4,unsigned,4,int) \
  __func__(4,unsigned,4,unsigned) \
  __func__(4,unsigned,4,long long) \
  __func__(4,unsigned,5,int) \
  __func__(4,unsigned,5,unsigned) \
  __func__(4,unsigned,5,long long) \
  __func__(4,unsigned,6,int) \
  __func__(4,unsigned,6,unsigned) \
  __func__(4,unsigned,6,long long) \
  __func__(4,unsigned,7,int) \
  __func__(4,unsigned,7,unsigned) \
  __func__(4,unsigned,7,long long) \
  __func__(4,long long,1,int) \
  __func__(4,long long,1,unsigned) \
  __func__(4,long long,1,long long) \
  __func__(4,long long,2,int) \
  __func__(4,long long,2,unsigned) \
  __func__(4,long long,2,long long) \
  __func__(4,long long,3,int) \
  __func__(4,long long,3,unsigned) \
  __func__(4,long long,3,long long) \
  __func__(4,long long,4,int) \
  __func__(4,long long,4,unsigned) \
  __func__(4,long long,4,long long) \
  __func__(4,long long,5,int) \
  __func__(4,long long,5,unsigned) \
  __func__(4,long long,5,long long) \
  __func__(4,long long,6,int) \
  __func__(4,long long,6,unsigned) \
  __func__(4,long long,6,long long) \
  __func__(4,long long,7,int) \
  __func__(4,long long,7,unsigned) \
  __func__(4,long long,7,long long) \
\
  __func__(5,int,1,int) \
  __func__(5,int,1,unsigned) \
  __func__(5,int,1,long long) \
  __func__(5,int,2,int) \
  __func__(5,int,2,unsigned) \
  __func__(5,int,2,long long) \
  __func__(5,int,3,int) \
  __func__(5,int,3,unsigned) \
  __func__(5,int,3,long long) \
  __func__(5,int,4,int) \
  __func__(5,int,4,unsigned) \
  __func__(5,int,4,long long) \
  __func__(5,int,5,int) \
  __func__(5,int,5,unsigned) \
  __func__(5,int,5,long long) \
  __func__(5,int,6,int) \
  __func__(5,int,6,unsigned) \
  __func__(5,int,6,long long) \
  __func__(5,int,7,int) \
  __func__(5,int,7,unsigned) \
  __func__(5,int,7,long long) \
  __func__(5,unsigned,1,int) \
  __func__(5,unsigned,1,unsigned) \
  __func__(5,unsigned,1,long long) \
  __func__(5,unsigned,2,int) \
  __func__(5,unsigned,2,unsigned) \
  __func__(5,unsigned,2,long long) \
  __func__(5,unsigned,3,int) \
  __func__(5,unsigned,3,unsigned) \
  __func__(5,unsigned,3,long long) \
  __func__(5,unsigned,4,int) \
  __func__(5,unsigned,4,unsigned) \
  __func__(5,unsigned,4,long long) \
  __func__(5,unsigned,5,int) \
  __func__(5,unsigned,5,unsigned) \
  __func__(5,unsigned,5,long long) \
  __func__(5,unsigned,6,int) \
  __func__(5,unsigned,6,unsigned) \
  __func__(5,unsigned,6,long long) \
  __func__(5,unsigned,7,int) \
  __func__(5,unsigned,7,unsigned) \
  __func__(5,unsigned,7,long long) \
  __func__(5,long long,1,int) \
  __func__(5,long long,1,unsigned) \
  __func__(5,long long,1,long long) \
  __func__(5,long long,2,int) \
  __func__(5,long long,2,unsigned) \
  __func__(5,long long,2,long long) \
  __func__(5,long long,3,int) \
  __func__(5,long long,3,unsigned) \
  __func__(5,long long,3,long long) \
  __func__(5,long long,4,int) \
  __func__(5,long long,4,unsigned) \
  __func__(5,long long,4,long long) \
  __func__(5,long long,5,int) \
  __func__(5,long long,5,unsigned) \
  __func__(5,long long,5,long long) \
  __func__(5,long long,6,int) \
  __func__(5,long long,6,unsigned) \
  __func__(5,long long,6,long long) \
  __func__(5,long long,7,int) \
  __func__(5,long long,7,unsigned) \
  __func__(5,long long,7,long long) \
\
  __func__(6,int,1,int) \
  __func__(6,int,1,unsigned) \
  __func__(6,int,1,long long) \
  __func__(6,int,2,int) \
  __func__(6,int,2,unsigned) \
  __func__(6,int,2,long long) \
  __func__(6,int,3,int) \
  __func__(6,int,3,unsigned) \
  __func__(6,int,3,long long) \
  __func__(6,int,4,int) \
  __func__(6,int,4,unsigned) \
  __func__(6,int,4,long long) \
  __func__(6,int,5,int) \
  __func__(6,int,5,unsigned) \
  __func__(6,int,5,long long) \
  __func__(6,int,6,int) \
  __func__(6,int,6,unsigned) \
  __func__(6,int,6,long long) \
  __func__(6,int,7,int) \
  __func__(6,int,7,unsigned) \
  __func__(6,int,7,long long) \
  __func__(6,unsigned,1,int) \
  __func__(6,unsigned,1,unsigned) \
  __func__(6,unsigned,1,long long) \
  __func__(6,unsigned,2,int) \
  __func__(6,unsigned,2,unsigned) \
  __func__(6,unsigned,2,long long) \
  __func__(6,unsigned,3,int) \
  __func__(6,unsigned,3,unsigned) \
  __func__(6,unsigned,3,long long) \
  __func__(6,unsigned,4,int) \
  __func__(6,unsigned,4,unsigned) \
  __func__(6,unsigned,4,long long) \
  __func__(6,unsigned,5,int) \
  __func__(6,unsigned,5,unsigned) \
  __func__(6,unsigned,5,long long) \
  __func__(6,unsigned,6,int) \
  __func__(6,unsigned,6,unsigned) \
  __func__(6,unsigned,6,long long) \
  __func__(6,unsigned,7,int) \
  __func__(6,unsigned,7,unsigned) \
  __func__(6,unsigned,7,long long) \
  __func__(6,long long,1,int) \
  __func__(6,long long,1,unsigned) \
  __func__(6,long long,1,long long) \
  __func__(6,long long,2,int) \
  __func__(6,long long,2,unsigned) \
  __func__(6,long long,2,long long) \
  __func__(6,long long,3,int) \
  __func__(6,long long,3,unsigned) \
  __func__(6,long long,3,long long) \
  __func__(6,long long,4,int) \
  __func__(6,long long,4,unsigned) \
  __func__(6,long long,4,long long) \
  __func__(6,long long,5,int) \
  __func__(6,long long,5,unsigned) \
  __func__(6,long long,5,long long) \
  __func__(6,long long,6,int) \
  __func__(6,long long,6,unsigned) \
  __func__(6,long long,6,long long) \
  __func__(6,long long,7,int) \
  __func__(6,long long,7,unsigned) \
  __func__(6,long long,7,long long) \
\
  __func__(7,int,1,int) \
  __func__(7,int,1,unsigned) \
  __func__(7,int,1,long long) \
  __func__(7,int,2,int) \
  __func__(7,int,2,unsigned) \
  __func__(7,int,2,long long) \
  __func__(7,int,3,int) \
  __func__(7,int,3,unsigned) \
  __func__(7,int,3,long long) \
  __func__(7,int,4,int) \
  __func__(7,int,4,unsigned) \
  __func__(7,int,4,long long) \
  __func__(7,int,5,int) \
  __func__(7,int,5,unsigned) \
  __func__(7,int,5,long long) \
  __func__(7,int,6,int) \
  __func__(7,int,6,unsigned) \
  __func__(7,int,6,long long) \
  __func__(7,int,7,int) \
  __func__(7,int,7,unsigned) \
  __func__(7,int,7,long long) \
  __func__(7,unsigned,1,int) \
  __func__(7,unsigned,1,unsigned) \
  __func__(7,unsigned,1,long long) \
  __func__(7,unsigned,2,int) \
  __func__(7,unsigned,2,unsigned) \
  __func__(7,unsigned,2,long long) \
  __func__(7,unsigned,3,int) \
  __func__(7,unsigned,3,unsigned) \
  __func__(7,unsigned,3,long long) \
  __func__(7,unsigned,4,int) \
  __func__(7,unsigned,4,unsigned) \
  __func__(7,unsigned,4,long long) \
  __func__(7,unsigned,5,int) \
  __func__(7,unsigned,5,unsigned) \
  __func__(7,unsigned,5,long long) \
  __func__(7,unsigned,6,int) \
  __func__(7,unsigned,6,unsigned) \
  __func__(7,unsigned,6,long long) \
  __func__(7,unsigned,7,int) \
  __func__(7,unsigned,7,unsigned) \
  __func__(7,unsigned,7,long long) \
  __func__(7,long long,1,int) \
  __func__(7,long long,1,unsigned) \
  __func__(7,long long,1,long long) \
  __func__(7,long long,2,int) \
  __func__(7,long long,2,unsigned) \
  __func__(7,long long,2,long long) \
  __func__(7,long long,3,int) \
  __func__(7,long long,3,unsigned) \
  __func__(7,long long,3,long long) \
  __func__(7,long long,4,int) \
  __func__(7,long long,4,unsigned) \
  __func__(7,long long,4,long long) \
  __func__(7,long long,5,int) \
  __func__(7,long long,5,unsigned) \
  __func__(7,long long,5,long long) \
  __func__(7,long long,6,int) \
  __func__(7,long long,6,unsigned) \
  __func__(7,long long,6,long long) \
  __func__(7,long long,7,int) \
  __func__(7,long long,7,unsigned) \
  __func__(7,long long,7,long long) \

#elif REALM_MAX_DIM == 8

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6) \
  __func__(7) \
  __func__(8) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \
  __func__(3,int) \
  __func__(3,unsigned) \
  __func__(3,long long) \
  __func__(4,int) \
  __func__(4,unsigned) \
  __func__(4,long long) \
  __func__(5,int) \
  __func__(5,unsigned) \
  __func__(5,long long) \
  __func__(6,int) \
  __func__(6,unsigned) \
  __func__(6,long long) \
  __func__(7,int) \
  __func__(7,unsigned) \
  __func__(7,long long) \
  __func__(8,int) \
  __func__(8,unsigned) \
  __func__(8,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \
  __func__(3,int,int) \
  __func__(3,int,bool) \
  __func__(3,unsigned,int) \
  __func__(3,unsigned,bool) \
  __func__(3,long long,int) \
  __func__(3,long long,bool) \
  __func__(4,int,int) \
  __func__(4,int,bool) \
  __func__(4,unsigned,int) \
  __func__(4,unsigned,bool) \
  __func__(4,long long,int) \
  __func__(4,long long,bool) \
  __func__(5,int,int) \
  __func__(5,int,bool) \
  __func__(5,unsigned,int) \
  __func__(5,unsigned,bool) \
  __func__(5,long long,int) \
  __func__(5,long long,bool) \
  __func__(6,int,int) \
  __func__(6,int,bool) \
  __func__(6,unsigned,int) \
  __func__(6,unsigned,bool) \
  __func__(6,long long,int) \
  __func__(6,long long,bool) \
  __func__(7,int,int) \
  __func__(7,int,bool) \
  __func__(7,unsigned,int) \
  __func__(7,unsigned,bool) \
  __func__(7,long long,int) \
  __func__(7,long long,bool) \
  __func__(8,int,int) \
  __func__(8,int,bool) \
  __func__(8,unsigned,int) \
  __func__(8,unsigned,bool) \
  __func__(8,long long,int) \
  __func__(8,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,int,3,int) \
  __func__(1,int,3,unsigned) \
  __func__(1,int,3,long long) \
  __func__(1,int,4,int) \
  __func__(1,int,4,unsigned) \
  __func__(1,int,4,long long) \
  __func__(1,int,5,int) \
  __func__(1,int,5,unsigned) \
  __func__(1,int,5,long long) \
  __func__(1,int,6,int) \
  __func__(1,int,6,unsigned) \
  __func__(1,int,6,long long) \
  __func__(1,int,7,int) \
  __func__(1,int,7,unsigned) \
  __func__(1,int,7,long long) \
  __func__(1,int,8,int) \
  __func__(1,int,8,unsigned) \
  __func__(1,int,8,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,unsigned,3,int) \
  __func__(1,unsigned,3,unsigned) \
  __func__(1,unsigned,3,long long) \
  __func__(1,unsigned,4,int) \
  __func__(1,unsigned,4,unsigned) \
  __func__(1,unsigned,4,long long) \
  __func__(1,unsigned,5,int) \
  __func__(1,unsigned,5,unsigned) \
  __func__(1,unsigned,5,long long) \
  __func__(1,unsigned,6,int) \
  __func__(1,unsigned,6,unsigned) \
  __func__(1,unsigned,6,long long) \
  __func__(1,unsigned,7,int) \
  __func__(1,unsigned,7,unsigned) \
  __func__(1,unsigned,7,long long) \
  __func__(1,unsigned,8,int) \
  __func__(1,unsigned,8,unsigned) \
  __func__(1,unsigned,8,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
  __func__(1,long long,3,int) \
  __func__(1,long long,3,unsigned) \
  __func__(1,long long,3,long long) \
  __func__(1,long long,4,int) \
  __func__(1,long long,4,unsigned) \
  __func__(1,long long,4,long long) \
  __func__(1,long long,5,int) \
  __func__(1,long long,5,unsigned) \
  __func__(1,long long,5,long long) \
  __func__(1,long long,6,int) \
  __func__(1,long long,6,unsigned) \
  __func__(1,long long,6,long long) \
  __func__(1,long long,7,int) \
  __func__(1,long long,7,unsigned) \
  __func__(1,long long,7,long long) \
  __func__(1,long long,8,int) \
  __func__(1,long long,8,unsigned) \
  __func__(1,long long,8,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,int,3,int) \
  __func__(2,int,3,unsigned) \
  __func__(2,int,3,long long) \
  __func__(2,int,4,int) \
  __func__(2,int,4,unsigned) \
  __func__(2,int,4,long long) \
  __func__(2,int,5,int) \
  __func__(2,int,5,unsigned) \
  __func__(2,int,5,long long) \
  __func__(2,int,6,int) \
  __func__(2,int,6,unsigned) \
  __func__(2,int,6,long long) \
  __func__(2,int,7,int) \
  __func__(2,int,7,unsigned) \
  __func__(2,int,7,long long) \
  __func__(2,int,8,int) \
  __func__(2,int,8,unsigned) \
  __func__(2,int,8,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,unsigned,3,int) \
  __func__(2,unsigned,3,unsigned) \
  __func__(2,unsigned,3,long long) \
  __func__(2,unsigned,4,int) \
  __func__(2,unsigned,4,unsigned) \
  __func__(2,unsigned,4,long long) \
  __func__(2,unsigned,5,int) \
  __func__(2,unsigned,5,unsigned) \
  __func__(2,unsigned,5,long long) \
  __func__(2,unsigned,6,int) \
  __func__(2,unsigned,6,unsigned) \
  __func__(2,unsigned,6,long long) \
  __func__(2,unsigned,7,int) \
  __func__(2,unsigned,7,unsigned) \
  __func__(2,unsigned,7,long long) \
  __func__(2,unsigned,8,int) \
  __func__(2,unsigned,8,unsigned) \
  __func__(2,unsigned,8,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \
  __func__(2,long long,3,int) \
  __func__(2,long long,3,unsigned) \
  __func__(2,long long,3,long long) \
  __func__(2,long long,4,int) \
  __func__(2,long long,4,unsigned) \
  __func__(2,long long,4,long long) \
  __func__(2,long long,5,int) \
  __func__(2,long long,5,unsigned) \
  __func__(2,long long,5,long long) \
  __func__(2,long long,6,int) \
  __func__(2,long long,6,unsigned) \
  __func__(2,long long,6,long long) \
  __func__(2,long long,7,int) \
  __func__(2,long long,7,unsigned) \
  __func__(2,long long,7,long long) \
  __func__(2,long long,8,int) \
  __func__(2,long long,8,unsigned) \
  __func__(2,long long,8,long long) \
\
  __func__(3,int,1,int) \
  __func__(3,int,1,unsigned) \
  __func__(3,int,1,long long) \
  __func__(3,int,2,int) \
  __func__(3,int,2,unsigned) \
  __func__(3,int,2,long long) \
  __func__(3,int,3,int) \
  __func__(3,int,3,unsigned) \
  __func__(3,int,3,long long) \
  __func__(3,int,4,int) \
  __func__(3,int,4,unsigned) \
  __func__(3,int,4,long long) \
  __func__(3,int,5,int) \
  __func__(3,int,5,unsigned) \
  __func__(3,int,5,long long) \
  __func__(3,int,6,int) \
  __func__(3,int,6,unsigned) \
  __func__(3,int,6,long long) \
  __func__(3,int,7,int) \
  __func__(3,int,7,unsigned) \
  __func__(3,int,7,long long) \
  __func__(3,int,8,int) \
  __func__(3,int,8,unsigned) \
  __func__(3,int,8,long long) \
  __func__(3,unsigned,1,int) \
  __func__(3,unsigned,1,unsigned) \
  __func__(3,unsigned,1,long long) \
  __func__(3,unsigned,2,int) \
  __func__(3,unsigned,2,unsigned) \
  __func__(3,unsigned,2,long long) \
  __func__(3,unsigned,3,int) \
  __func__(3,unsigned,3,unsigned) \
  __func__(3,unsigned,3,long long) \
  __func__(3,unsigned,4,int) \
  __func__(3,unsigned,4,unsigned) \
  __func__(3,unsigned,4,long long) \
  __func__(3,unsigned,5,int) \
  __func__(3,unsigned,5,unsigned) \
  __func__(3,unsigned,5,long long) \
  __func__(3,unsigned,6,int) \
  __func__(3,unsigned,6,unsigned) \
  __func__(3,unsigned,6,long long) \
  __func__(3,unsigned,7,int) \
  __func__(3,unsigned,7,unsigned) \
  __func__(3,unsigned,7,long long) \
  __func__(3,unsigned,8,int) \
  __func__(3,unsigned,8,unsigned) \
  __func__(3,unsigned,8,long long) \
  __func__(3,long long,1,int) \
  __func__(3,long long,1,unsigned) \
  __func__(3,long long,1,long long) \
  __func__(3,long long,2,int) \
  __func__(3,long long,2,unsigned) \
  __func__(3,long long,2,long long) \
  __func__(3,long long,3,int) \
  __func__(3,long long,3,unsigned) \
  __func__(3,long long,3,long long) \
  __func__(3,long long,4,int) \
  __func__(3,long long,4,unsigned) \
  __func__(3,long long,4,long long) \
  __func__(3,long long,5,int) \
  __func__(3,long long,5,unsigned) \
  __func__(3,long long,5,long long) \
  __func__(3,long long,6,int) \
  __func__(3,long long,6,unsigned) \
  __func__(3,long long,6,long long) \
  __func__(3,long long,7,int) \
  __func__(3,long long,7,unsigned) \
  __func__(3,long long,7,long long) \
  __func__(3,long long,8,int) \
  __func__(3,long long,8,unsigned) \
  __func__(3,long long,8,long long) \
\
  __func__(4,int,1,int) \
  __func__(4,int,1,unsigned) \
  __func__(4,int,1,long long) \
  __func__(4,int,2,int) \
  __func__(4,int,2,unsigned) \
  __func__(4,int,2,long long) \
  __func__(4,int,3,int) \
  __func__(4,int,3,unsigned) \
  __func__(4,int,3,long long) \
  __func__(4,int,4,int) \
  __func__(4,int,4,unsigned) \
  __func__(4,int,4,long long) \
  __func__(4,int,5,int) \
  __func__(4,int,5,unsigned) \
  __func__(4,int,5,long long) \
  __func__(4,int,6,int) \
  __func__(4,int,6,unsigned) \
  __func__(4,int,6,long long) \
  __func__(4,int,7,int) \
  __func__(4,int,7,unsigned) \
  __func__(4,int,7,long long) \
  __func__(4,int,8,int) \
  __func__(4,int,8,unsigned) \
  __func__(4,int,8,long long) \
  __func__(4,unsigned,1,int) \
  __func__(4,unsigned,1,unsigned) \
  __func__(4,unsigned,1,long long) \
  __func__(4,unsigned,2,int) \
  __func__(4,unsigned,2,unsigned) \
  __func__(4,unsigned,2,long long) \
  __func__(4,unsigned,3,int) \
  __func__(4,unsigned,3,unsigned) \
  __func__(4,unsigned,3,long long) \
  __func__(4,unsigned,4,int) \
  __func__(4,unsigned,4,unsigned) \
  __func__(4,unsigned,4,long long) \
  __func__(4,unsigned,5,int) \
  __func__(4,unsigned,5,unsigned) \
  __func__(4,unsigned,5,long long) \
  __func__(4,unsigned,6,int) \
  __func__(4,unsigned,6,unsigned) \
  __func__(4,unsigned,6,long long) \
  __func__(4,unsigned,7,int) \
  __func__(4,unsigned,7,unsigned) \
  __func__(4,unsigned,7,long long) \
  __func__(4,unsigned,8,int) \
  __func__(4,unsigned,8,unsigned) \
  __func__(4,unsigned,8,long long) \
  __func__(4,long long,1,int) \
  __func__(4,long long,1,unsigned) \
  __func__(4,long long,1,long long) \
  __func__(4,long long,2,int) \
  __func__(4,long long,2,unsigned) \
  __func__(4,long long,2,long long) \
  __func__(4,long long,3,int) \
  __func__(4,long long,3,unsigned) \
  __func__(4,long long,3,long long) \
  __func__(4,long long,4,int) \
  __func__(4,long long,4,unsigned) \
  __func__(4,long long,4,long long) \
  __func__(4,long long,5,int) \
  __func__(4,long long,5,unsigned) \
  __func__(4,long long,5,long long) \
  __func__(4,long long,6,int) \
  __func__(4,long long,6,unsigned) \
  __func__(4,long long,6,long long) \
  __func__(4,long long,7,int) \
  __func__(4,long long,7,unsigned) \
  __func__(4,long long,7,long long) \
  __func__(4,long long,8,int) \
  __func__(4,long long,8,unsigned) \
  __func__(4,long long,8,long long) \
\
  __func__(5,int,1,int) \
  __func__(5,int,1,unsigned) \
  __func__(5,int,1,long long) \
  __func__(5,int,2,int) \
  __func__(5,int,2,unsigned) \
  __func__(5,int,2,long long) \
  __func__(5,int,3,int) \
  __func__(5,int,3,unsigned) \
  __func__(5,int,3,long long) \
  __func__(5,int,4,int) \
  __func__(5,int,4,unsigned) \
  __func__(5,int,4,long long) \
  __func__(5,int,5,int) \
  __func__(5,int,5,unsigned) \
  __func__(5,int,5,long long) \
  __func__(5,int,6,int) \
  __func__(5,int,6,unsigned) \
  __func__(5,int,6,long long) \
  __func__(5,int,7,int) \
  __func__(5,int,7,unsigned) \
  __func__(5,int,7,long long) \
  __func__(5,int,8,int) \
  __func__(5,int,8,unsigned) \
  __func__(5,int,8,long long) \
  __func__(5,unsigned,1,int) \
  __func__(5,unsigned,1,unsigned) \
  __func__(5,unsigned,1,long long) \
  __func__(5,unsigned,2,int) \
  __func__(5,unsigned,2,unsigned) \
  __func__(5,unsigned,2,long long) \
  __func__(5,unsigned,3,int) \
  __func__(5,unsigned,3,unsigned) \
  __func__(5,unsigned,3,long long) \
  __func__(5,unsigned,4,int) \
  __func__(5,unsigned,4,unsigned) \
  __func__(5,unsigned,4,long long) \
  __func__(5,unsigned,5,int) \
  __func__(5,unsigned,5,unsigned) \
  __func__(5,unsigned,5,long long) \
  __func__(5,unsigned,6,int) \
  __func__(5,unsigned,6,unsigned) \
  __func__(5,unsigned,6,long long) \
  __func__(5,unsigned,7,int) \
  __func__(5,unsigned,7,unsigned) \
  __func__(5,unsigned,7,long long) \
  __func__(5,unsigned,8,int) \
  __func__(5,unsigned,8,unsigned) \
  __func__(5,unsigned,8,long long) \
  __func__(5,long long,1,int) \
  __func__(5,long long,1,unsigned) \
  __func__(5,long long,1,long long) \
  __func__(5,long long,2,int) \
  __func__(5,long long,2,unsigned) \
  __func__(5,long long,2,long long) \
  __func__(5,long long,3,int) \
  __func__(5,long long,3,unsigned) \
  __func__(5,long long,3,long long) \
  __func__(5,long long,4,int) \
  __func__(5,long long,4,unsigned) \
  __func__(5,long long,4,long long) \
  __func__(5,long long,5,int) \
  __func__(5,long long,5,unsigned) \
  __func__(5,long long,5,long long) \
  __func__(5,long long,6,int) \
  __func__(5,long long,6,unsigned) \
  __func__(5,long long,6,long long) \
  __func__(5,long long,7,int) \
  __func__(5,long long,7,unsigned) \
  __func__(5,long long,7,long long) \
  __func__(5,long long,8,int) \
  __func__(5,long long,8,unsigned) \
  __func__(5,long long,8,long long) \
\
  __func__(6,int,1,int) \
  __func__(6,int,1,unsigned) \
  __func__(6,int,1,long long) \
  __func__(6,int,2,int) \
  __func__(6,int,2,unsigned) \
  __func__(6,int,2,long long) \
  __func__(6,int,3,int) \
  __func__(6,int,3,unsigned) \
  __func__(6,int,3,long long) \
  __func__(6,int,4,int) \
  __func__(6,int,4,unsigned) \
  __func__(6,int,4,long long) \
  __func__(6,int,5,int) \
  __func__(6,int,5,unsigned) \
  __func__(6,int,5,long long) \
  __func__(6,int,6,int) \
  __func__(6,int,6,unsigned) \
  __func__(6,int,6,long long) \
  __func__(6,int,7,int) \
  __func__(6,int,7,unsigned) \
  __func__(6,int,7,long long) \
  __func__(6,int,8,int) \
  __func__(6,int,8,unsigned) \
  __func__(6,int,8,long long) \
  __func__(6,unsigned,1,int) \
  __func__(6,unsigned,1,unsigned) \
  __func__(6,unsigned,1,long long) \
  __func__(6,unsigned,2,int) \
  __func__(6,unsigned,2,unsigned) \
  __func__(6,unsigned,2,long long) \
  __func__(6,unsigned,3,int) \
  __func__(6,unsigned,3,unsigned) \
  __func__(6,unsigned,3,long long) \
  __func__(6,unsigned,4,int) \
  __func__(6,unsigned,4,unsigned) \
  __func__(6,unsigned,4,long long) \
  __func__(6,unsigned,5,int) \
  __func__(6,unsigned,5,unsigned) \
  __func__(6,unsigned,5,long long) \
  __func__(6,unsigned,6,int) \
  __func__(6,unsigned,6,unsigned) \
  __func__(6,unsigned,6,long long) \
  __func__(6,unsigned,7,int) \
  __func__(6,unsigned,7,unsigned) \
  __func__(6,unsigned,7,long long) \
  __func__(6,unsigned,8,int) \
  __func__(6,unsigned,8,unsigned) \
  __func__(6,unsigned,8,long long) \
  __func__(6,long long,1,int) \
  __func__(6,long long,1,unsigned) \
  __func__(6,long long,1,long long) \
  __func__(6,long long,2,int) \
  __func__(6,long long,2,unsigned) \
  __func__(6,long long,2,long long) \
  __func__(6,long long,3,int) \
  __func__(6,long long,3,unsigned) \
  __func__(6,long long,3,long long) \
  __func__(6,long long,4,int) \
  __func__(6,long long,4,unsigned) \
  __func__(6,long long,4,long long) \
  __func__(6,long long,5,int) \
  __func__(6,long long,5,unsigned) \
  __func__(6,long long,5,long long) \
  __func__(6,long long,6,int) \
  __func__(6,long long,6,unsigned) \
  __func__(6,long long,6,long long) \
  __func__(6,long long,7,int) \
  __func__(6,long long,7,unsigned) \
  __func__(6,long long,7,long long) \
  __func__(6,long long,8,int) \
  __func__(6,long long,8,unsigned) \
  __func__(6,long long,8,long long) \
\
  __func__(7,int,1,int) \
  __func__(7,int,1,unsigned) \
  __func__(7,int,1,long long) \
  __func__(7,int,2,int) \
  __func__(7,int,2,unsigned) \
  __func__(7,int,2,long long) \
  __func__(7,int,3,int) \
  __func__(7,int,3,unsigned) \
  __func__(7,int,3,long long) \
  __func__(7,int,4,int) \
  __func__(7,int,4,unsigned) \
  __func__(7,int,4,long long) \
  __func__(7,int,5,int) \
  __func__(7,int,5,unsigned) \
  __func__(7,int,5,long long) \
  __func__(7,int,6,int) \
  __func__(7,int,6,unsigned) \
  __func__(7,int,6,long long) \
  __func__(7,int,7,int) \
  __func__(7,int,7,unsigned) \
  __func__(7,int,7,long long) \
  __func__(7,int,8,int) \
  __func__(7,int,8,unsigned) \
  __func__(7,int,8,long long) \
  __func__(7,unsigned,1,int) \
  __func__(7,unsigned,1,unsigned) \
  __func__(7,unsigned,1,long long) \
  __func__(7,unsigned,2,int) \
  __func__(7,unsigned,2,unsigned) \
  __func__(7,unsigned,2,long long) \
  __func__(7,unsigned,3,int) \
  __func__(7,unsigned,3,unsigned) \
  __func__(7,unsigned,3,long long) \
  __func__(7,unsigned,4,int) \
  __func__(7,unsigned,4,unsigned) \
  __func__(7,unsigned,4,long long) \
  __func__(7,unsigned,5,int) \
  __func__(7,unsigned,5,unsigned) \
  __func__(7,unsigned,5,long long) \
  __func__(7,unsigned,6,int) \
  __func__(7,unsigned,6,unsigned) \
  __func__(7,unsigned,6,long long) \
  __func__(7,unsigned,7,int) \
  __func__(7,unsigned,7,unsigned) \
  __func__(7,unsigned,7,long long) \
  __func__(7,unsigned,8,int) \
  __func__(7,unsigned,8,unsigned) \
  __func__(7,unsigned,8,long long) \
  __func__(7,long long,1,int) \
  __func__(7,long long,1,unsigned) \
  __func__(7,long long,1,long long) \
  __func__(7,long long,2,int) \
  __func__(7,long long,2,unsigned) \
  __func__(7,long long,2,long long) \
  __func__(7,long long,3,int) \
  __func__(7,long long,3,unsigned) \
  __func__(7,long long,3,long long) \
  __func__(7,long long,4,int) \
  __func__(7,long long,4,unsigned) \
  __func__(7,long long,4,long long) \
  __func__(7,long long,5,int) \
  __func__(7,long long,5,unsigned) \
  __func__(7,long long,5,long long) \
  __func__(7,long long,6,int) \
  __func__(7,long long,6,unsigned) \
  __func__(7,long long,6,long long) \
  __func__(7,long long,7,int) \
  __func__(7,long long,7,unsigned) \
  __func__(7,long long,7,long long) \
  __func__(7,long long,8,int) \
  __func__(7,long long,8,unsigned) \
  __func__(7,long long,8,long long) \
\
  __func__(8,int,1,int) \
  __func__(8,int,1,unsigned) \
  __func__(8,int,1,long long) \
  __func__(8,int,2,int) \
  __func__(8,int,2,unsigned) \
  __func__(8,int,2,long long) \
  __func__(8,int,3,int) \
  __func__(8,int,3,unsigned) \
  __func__(8,int,3,long long) \
  __func__(8,int,4,int) \
  __func__(8,int,4,unsigned) \
  __func__(8,int,4,long long) \
  __func__(8,int,5,int) \
  __func__(8,int,5,unsigned) \
  __func__(8,int,5,long long) \
  __func__(8,int,6,int) \
  __func__(8,int,6,unsigned) \
  __func__(8,int,6,long long) \
  __func__(8,int,7,int) \
  __func__(8,int,7,unsigned) \
  __func__(8,int,7,long long) \
  __func__(8,int,8,int) \
  __func__(8,int,8,unsigned) \
  __func__(8,int,8,long long) \
  __func__(8,unsigned,1,int) \
  __func__(8,unsigned,1,unsigned) \
  __func__(8,unsigned,1,long long) \
  __func__(8,unsigned,2,int) \
  __func__(8,unsigned,2,unsigned) \
  __func__(8,unsigned,2,long long) \
  __func__(8,unsigned,3,int) \
  __func__(8,unsigned,3,unsigned) \
  __func__(8,unsigned,3,long long) \
  __func__(8,unsigned,4,int) \
  __func__(8,unsigned,4,unsigned) \
  __func__(8,unsigned,4,long long) \
  __func__(8,unsigned,5,int) \
  __func__(8,unsigned,5,unsigned) \
  __func__(8,unsigned,5,long long) \
  __func__(8,unsigned,6,int) \
  __func__(8,unsigned,6,unsigned) \
  __func__(8,unsigned,6,long long) \
  __func__(8,unsigned,7,int) \
  __func__(8,unsigned,7,unsigned) \
  __func__(8,unsigned,7,long long) \
  __func__(8,unsigned,8,int) \
  __func__(8,unsigned,8,unsigned) \
  __func__(8,unsigned,8,long long) \
  __func__(8,long long,1,int) \
  __func__(8,long long,1,unsigned) \
  __func__(8,long long,1,long long) \
  __func__(8,long long,2,int) \
  __func__(8,long long,2,unsigned) \
  __func__(8,long long,2,long long) \
  __func__(8,long long,3,int) \
  __func__(8,long long,3,unsigned) \
  __func__(8,long long,3,long long) \
  __func__(8,long long,4,int) \
  __func__(8,long long,4,unsigned) \
  __func__(8,long long,4,long long) \
  __func__(8,long long,5,int) \
  __func__(8,long long,5,unsigned) \
  __func__(8,long long,5,long long) \
  __func__(8,long long,6,int) \
  __func__(8,long long,6,unsigned) \
  __func__(8,long long,6,long long) \
  __func__(8,long long,7,int) \
  __func__(8,long long,7,unsigned) \
  __func__(8,long long,7,long long) \
  __func__(8,long long,8,int) \
  __func__(8,long long,8,unsigned) \
  __func__(8,long long,8,long long) \

#elif REALM_MAX_DIM == 9

#define FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6) \
  __func__(7) \
  __func__(8) \
  __func__(9) \

#define FOREACH_NT(__func__) \
  __func__(1,int) \
  __func__(1,unsigned) \
  __func__(1,long long) \
  __func__(2,int) \
  __func__(2,unsigned) \
  __func__(2,long long) \
  __func__(3,int) \
  __func__(3,unsigned) \
  __func__(3,long long) \
  __func__(4,int) \
  __func__(4,unsigned) \
  __func__(4,long long) \
  __func__(5,int) \
  __func__(5,unsigned) \
  __func__(5,long long) \
  __func__(6,int) \
  __func__(6,unsigned) \
  __func__(6,long long) \
  __func__(7,int) \
  __func__(7,unsigned) \
  __func__(7,long long) \
  __func__(8,int) \
  __func__(8,unsigned) \
  __func__(8,long long) \
  __func__(9,int) \
  __func__(9,unsigned) \
  __func__(9,long long) \

#define FOREACH_NTF(__func__) \
  __func__(1,int,int) \
  __func__(1,int,bool) \
  __func__(1,unsigned,int) \
  __func__(1,unsigned,bool) \
  __func__(1,long long,int) \
  __func__(1,long long,bool) \
  __func__(2,int,int) \
  __func__(2,int,bool) \
  __func__(2,unsigned,int) \
  __func__(2,unsigned,bool) \
  __func__(2,long long,int) \
  __func__(2,long long,bool) \
  __func__(3,int,int) \
  __func__(3,int,bool) \
  __func__(3,unsigned,int) \
  __func__(3,unsigned,bool) \
  __func__(3,long long,int) \
  __func__(3,long long,bool) \
  __func__(4,int,int) \
  __func__(4,int,bool) \
  __func__(4,unsigned,int) \
  __func__(4,unsigned,bool) \
  __func__(4,long long,int) \
  __func__(4,long long,bool) \
  __func__(5,int,int) \
  __func__(5,int,bool) \
  __func__(5,unsigned,int) \
  __func__(5,unsigned,bool) \
  __func__(5,long long,int) \
  __func__(5,long long,bool) \
  __func__(6,int,int) \
  __func__(6,int,bool) \
  __func__(6,unsigned,int) \
  __func__(6,unsigned,bool) \
  __func__(6,long long,int) \
  __func__(6,long long,bool) \
  __func__(7,int,int) \
  __func__(7,int,bool) \
  __func__(7,unsigned,int) \
  __func__(7,unsigned,bool) \
  __func__(7,long long,int) \
  __func__(7,long long,bool) \
  __func__(8,int,int) \
  __func__(8,int,bool) \
  __func__(8,unsigned,int) \
  __func__(8,unsigned,bool) \
  __func__(8,long long,int) \
  __func__(8,long long,bool) \
  __func__(9,int,int) \
  __func__(9,int,bool) \
  __func__(9,unsigned,int) \
  __func__(9,unsigned,bool) \
  __func__(9,long long,int) \
  __func__(9,long long,bool) \

#define FOREACH_NTNT(__func__) \
  __func__(1,int,1,int) \
  __func__(1,int,1,unsigned) \
  __func__(1,int,1,long long) \
  __func__(1,int,2,int) \
  __func__(1,int,2,unsigned) \
  __func__(1,int,2,long long) \
  __func__(1,int,3,int) \
  __func__(1,int,3,unsigned) \
  __func__(1,int,3,long long) \
  __func__(1,int,4,int) \
  __func__(1,int,4,unsigned) \
  __func__(1,int,4,long long) \
  __func__(1,int,5,int) \
  __func__(1,int,5,unsigned) \
  __func__(1,int,5,long long) \
  __func__(1,int,6,int) \
  __func__(1,int,6,unsigned) \
  __func__(1,int,6,long long) \
  __func__(1,int,7,int) \
  __func__(1,int,7,unsigned) \
  __func__(1,int,7,long long) \
  __func__(1,int,8,int) \
  __func__(1,int,8,unsigned) \
  __func__(1,int,8,long long) \
  __func__(1,int,9,int) \
  __func__(1,int,9,unsigned) \
  __func__(1,int,9,long long) \
  __func__(1,unsigned,1,int) \
  __func__(1,unsigned,1,unsigned) \
  __func__(1,unsigned,1,long long) \
  __func__(1,unsigned,2,int) \
  __func__(1,unsigned,2,unsigned) \
  __func__(1,unsigned,2,long long) \
  __func__(1,unsigned,3,int) \
  __func__(1,unsigned,3,unsigned) \
  __func__(1,unsigned,3,long long) \
  __func__(1,unsigned,4,int) \
  __func__(1,unsigned,4,unsigned) \
  __func__(1,unsigned,4,long long) \
  __func__(1,unsigned,5,int) \
  __func__(1,unsigned,5,unsigned) \
  __func__(1,unsigned,5,long long) \
  __func__(1,unsigned,6,int) \
  __func__(1,unsigned,6,unsigned) \
  __func__(1,unsigned,6,long long) \
  __func__(1,unsigned,7,int) \
  __func__(1,unsigned,7,unsigned) \
  __func__(1,unsigned,7,long long) \
  __func__(1,unsigned,8,int) \
  __func__(1,unsigned,8,unsigned) \
  __func__(1,unsigned,8,long long) \
  __func__(1,unsigned,9,int) \
  __func__(1,unsigned,9,unsigned) \
  __func__(1,unsigned,9,long long) \
  __func__(1,long long,1,int) \
  __func__(1,long long,1,unsigned) \
  __func__(1,long long,1,long long) \
  __func__(1,long long,2,int) \
  __func__(1,long long,2,unsigned) \
  __func__(1,long long,2,long long) \
  __func__(1,long long,3,int) \
  __func__(1,long long,3,unsigned) \
  __func__(1,long long,3,long long) \
  __func__(1,long long,4,int) \
  __func__(1,long long,4,unsigned) \
  __func__(1,long long,4,long long) \
  __func__(1,long long,5,int) \
  __func__(1,long long,5,unsigned) \
  __func__(1,long long,5,long long) \
  __func__(1,long long,6,int) \
  __func__(1,long long,6,unsigned) \
  __func__(1,long long,6,long long) \
  __func__(1,long long,7,int) \
  __func__(1,long long,7,unsigned) \
  __func__(1,long long,7,long long) \
  __func__(1,long long,8,int) \
  __func__(1,long long,8,unsigned) \
  __func__(1,long long,8,long long) \
  __func__(1,long long,9,int) \
  __func__(1,long long,9,unsigned) \
  __func__(1,long long,9,long long) \
\
  __func__(2,int,1,int) \
  __func__(2,int,1,unsigned) \
  __func__(2,int,1,long long) \
  __func__(2,int,2,int) \
  __func__(2,int,2,unsigned) \
  __func__(2,int,2,long long) \
  __func__(2,int,3,int) \
  __func__(2,int,3,unsigned) \
  __func__(2,int,3,long long) \
  __func__(2,int,4,int) \
  __func__(2,int,4,unsigned) \
  __func__(2,int,4,long long) \
  __func__(2,int,5,int) \
  __func__(2,int,5,unsigned) \
  __func__(2,int,5,long long) \
  __func__(2,int,6,int) \
  __func__(2,int,6,unsigned) \
  __func__(2,int,6,long long) \
  __func__(2,int,7,int) \
  __func__(2,int,7,unsigned) \
  __func__(2,int,7,long long) \
  __func__(2,int,8,int) \
  __func__(2,int,8,unsigned) \
  __func__(2,int,8,long long) \
  __func__(2,int,9,int) \
  __func__(2,int,9,unsigned) \
  __func__(2,int,9,long long) \
  __func__(2,unsigned,1,int) \
  __func__(2,unsigned,1,unsigned) \
  __func__(2,unsigned,1,long long) \
  __func__(2,unsigned,2,int) \
  __func__(2,unsigned,2,unsigned) \
  __func__(2,unsigned,2,long long) \
  __func__(2,unsigned,3,int) \
  __func__(2,unsigned,3,unsigned) \
  __func__(2,unsigned,3,long long) \
  __func__(2,unsigned,4,int) \
  __func__(2,unsigned,4,unsigned) \
  __func__(2,unsigned,4,long long) \
  __func__(2,unsigned,5,int) \
  __func__(2,unsigned,5,unsigned) \
  __func__(2,unsigned,5,long long) \
  __func__(2,unsigned,6,int) \
  __func__(2,unsigned,6,unsigned) \
  __func__(2,unsigned,6,long long) \
  __func__(2,unsigned,7,int) \
  __func__(2,unsigned,7,unsigned) \
  __func__(2,unsigned,7,long long) \
  __func__(2,unsigned,8,int) \
  __func__(2,unsigned,8,unsigned) \
  __func__(2,unsigned,8,long long) \
  __func__(2,unsigned,9,int) \
  __func__(2,unsigned,9,unsigned) \
  __func__(2,unsigned,9,long long) \
  __func__(2,long long,1,int) \
  __func__(2,long long,1,unsigned) \
  __func__(2,long long,1,long long) \
  __func__(2,long long,2,int) \
  __func__(2,long long,2,unsigned) \
  __func__(2,long long,2,long long) \
  __func__(2,long long,3,int) \
  __func__(2,long long,3,unsigned) \
  __func__(2,long long,3,long long) \
  __func__(2,long long,4,int) \
  __func__(2,long long,4,unsigned) \
  __func__(2,long long,4,long long) \
  __func__(2,long long,5,int) \
  __func__(2,long long,5,unsigned) \
  __func__(2,long long,5,long long) \
  __func__(2,long long,6,int) \
  __func__(2,long long,6,unsigned) \
  __func__(2,long long,6,long long) \
  __func__(2,long long,7,int) \
  __func__(2,long long,7,unsigned) \
  __func__(2,long long,7,long long) \
  __func__(2,long long,8,int) \
  __func__(2,long long,8,unsigned) \
  __func__(2,long long,8,long long) \
  __func__(2,long long,9,int) \
  __func__(2,long long,9,unsigned) \
  __func__(2,long long,9,long long) \
\
  __func__(3,int,1,int) \
  __func__(3,int,1,unsigned) \
  __func__(3,int,1,long long) \
  __func__(3,int,2,int) \
  __func__(3,int,2,unsigned) \
  __func__(3,int,2,long long) \
  __func__(3,int,3,int) \
  __func__(3,int,3,unsigned) \
  __func__(3,int,3,long long) \
  __func__(3,int,4,int) \
  __func__(3,int,4,unsigned) \
  __func__(3,int,4,long long) \
  __func__(3,int,5,int) \
  __func__(3,int,5,unsigned) \
  __func__(3,int,5,long long) \
  __func__(3,int,6,int) \
  __func__(3,int,6,unsigned) \
  __func__(3,int,6,long long) \
  __func__(3,int,7,int) \
  __func__(3,int,7,unsigned) \
  __func__(3,int,7,long long) \
  __func__(3,int,8,int) \
  __func__(3,int,8,unsigned) \
  __func__(3,int,8,long long) \
  __func__(3,int,9,int) \
  __func__(3,int,9,unsigned) \
  __func__(3,int,9,long long) \
  __func__(3,unsigned,1,int) \
  __func__(3,unsigned,1,unsigned) \
  __func__(3,unsigned,1,long long) \
  __func__(3,unsigned,2,int) \
  __func__(3,unsigned,2,unsigned) \
  __func__(3,unsigned,2,long long) \
  __func__(3,unsigned,3,int) \
  __func__(3,unsigned,3,unsigned) \
  __func__(3,unsigned,3,long long) \
  __func__(3,unsigned,4,int) \
  __func__(3,unsigned,4,unsigned) \
  __func__(3,unsigned,4,long long) \
  __func__(3,unsigned,5,int) \
  __func__(3,unsigned,5,unsigned) \
  __func__(3,unsigned,5,long long) \
  __func__(3,unsigned,6,int) \
  __func__(3,unsigned,6,unsigned) \
  __func__(3,unsigned,6,long long) \
  __func__(3,unsigned,7,int) \
  __func__(3,unsigned,7,unsigned) \
  __func__(3,unsigned,7,long long) \
  __func__(3,unsigned,8,int) \
  __func__(3,unsigned,8,unsigned) \
  __func__(3,unsigned,8,long long) \
  __func__(3,unsigned,9,int) \
  __func__(3,unsigned,9,unsigned) \
  __func__(3,unsigned,9,long long) \
  __func__(3,long long,1,int) \
  __func__(3,long long,1,unsigned) \
  __func__(3,long long,1,long long) \
  __func__(3,long long,2,int) \
  __func__(3,long long,2,unsigned) \
  __func__(3,long long,2,long long) \
  __func__(3,long long,3,int) \
  __func__(3,long long,3,unsigned) \
  __func__(3,long long,3,long long) \
  __func__(3,long long,4,int) \
  __func__(3,long long,4,unsigned) \
  __func__(3,long long,4,long long) \
  __func__(3,long long,5,int) \
  __func__(3,long long,5,unsigned) \
  __func__(3,long long,5,long long) \
  __func__(3,long long,6,int) \
  __func__(3,long long,6,unsigned) \
  __func__(3,long long,6,long long) \
  __func__(3,long long,7,int) \
  __func__(3,long long,7,unsigned) \
  __func__(3,long long,7,long long) \
  __func__(3,long long,8,int) \
  __func__(3,long long,8,unsigned) \
  __func__(3,long long,8,long long) \
  __func__(3,long long,9,int) \
  __func__(3,long long,9,unsigned) \
  __func__(3,long long,9,long long) \
\
  __func__(4,int,1,int) \
  __func__(4,int,1,unsigned) \
  __func__(4,int,1,long long) \
  __func__(4,int,2,int) \
  __func__(4,int,2,unsigned) \
  __func__(4,int,2,long long) \
  __func__(4,int,3,int) \
  __func__(4,int,3,unsigned) \
  __func__(4,int,3,long long) \
  __func__(4,int,4,int) \
  __func__(4,int,4,unsigned) \
  __func__(4,int,4,long long) \
  __func__(4,int,5,int) \
  __func__(4,int,5,unsigned) \
  __func__(4,int,5,long long) \
  __func__(4,int,6,int) \
  __func__(4,int,6,unsigned) \
  __func__(4,int,6,long long) \
  __func__(4,int,7,int) \
  __func__(4,int,7,unsigned) \
  __func__(4,int,7,long long) \
  __func__(4,int,8,int) \
  __func__(4,int,8,unsigned) \
  __func__(4,int,8,long long) \
  __func__(4,int,9,int) \
  __func__(4,int,9,unsigned) \
  __func__(4,int,9,long long) \
  __func__(4,unsigned,1,int) \
  __func__(4,unsigned,1,unsigned) \
  __func__(4,unsigned,1,long long) \
  __func__(4,unsigned,2,int) \
  __func__(4,unsigned,2,unsigned) \
  __func__(4,unsigned,2,long long) \
  __func__(4,unsigned,3,int) \
  __func__(4,unsigned,3,unsigned) \
  __func__(4,unsigned,3,long long) \
  __func__(4,unsigned,4,int) \
  __func__(4,unsigned,4,unsigned) \
  __func__(4,unsigned,4,long long) \
  __func__(4,unsigned,5,int) \
  __func__(4,unsigned,5,unsigned) \
  __func__(4,unsigned,5,long long) \
  __func__(4,unsigned,6,int) \
  __func__(4,unsigned,6,unsigned) \
  __func__(4,unsigned,6,long long) \
  __func__(4,unsigned,7,int) \
  __func__(4,unsigned,7,unsigned) \
  __func__(4,unsigned,7,long long) \
  __func__(4,unsigned,8,int) \
  __func__(4,unsigned,8,unsigned) \
  __func__(4,unsigned,8,long long) \
  __func__(4,unsigned,9,int) \
  __func__(4,unsigned,9,unsigned) \
  __func__(4,unsigned,9,long long) \
  __func__(4,long long,1,int) \
  __func__(4,long long,1,unsigned) \
  __func__(4,long long,1,long long) \
  __func__(4,long long,2,int) \
  __func__(4,long long,2,unsigned) \
  __func__(4,long long,2,long long) \
  __func__(4,long long,3,int) \
  __func__(4,long long,3,unsigned) \
  __func__(4,long long,3,long long) \
  __func__(4,long long,4,int) \
  __func__(4,long long,4,unsigned) \
  __func__(4,long long,4,long long) \
  __func__(4,long long,5,int) \
  __func__(4,long long,5,unsigned) \
  __func__(4,long long,5,long long) \
  __func__(4,long long,6,int) \
  __func__(4,long long,6,unsigned) \
  __func__(4,long long,6,long long) \
  __func__(4,long long,7,int) \
  __func__(4,long long,7,unsigned) \
  __func__(4,long long,7,long long) \
  __func__(4,long long,8,int) \
  __func__(4,long long,8,unsigned) \
  __func__(4,long long,8,long long) \
  __func__(4,long long,9,int) \
  __func__(4,long long,9,unsigned) \
  __func__(4,long long,9,long long) \
\
  __func__(5,int,1,int) \
  __func__(5,int,1,unsigned) \
  __func__(5,int,1,long long) \
  __func__(5,int,2,int) \
  __func__(5,int,2,unsigned) \
  __func__(5,int,2,long long) \
  __func__(5,int,3,int) \
  __func__(5,int,3,unsigned) \
  __func__(5,int,3,long long) \
  __func__(5,int,4,int) \
  __func__(5,int,4,unsigned) \
  __func__(5,int,4,long long) \
  __func__(5,int,5,int) \
  __func__(5,int,5,unsigned) \
  __func__(5,int,5,long long) \
  __func__(5,int,6,int) \
  __func__(5,int,6,unsigned) \
  __func__(5,int,6,long long) \
  __func__(5,int,7,int) \
  __func__(5,int,7,unsigned) \
  __func__(5,int,7,long long) \
  __func__(5,int,8,int) \
  __func__(5,int,8,unsigned) \
  __func__(5,int,8,long long) \
  __func__(5,int,9,int) \
  __func__(5,int,9,unsigned) \
  __func__(5,int,9,long long) \
  __func__(5,unsigned,1,int) \
  __func__(5,unsigned,1,unsigned) \
  __func__(5,unsigned,1,long long) \
  __func__(5,unsigned,2,int) \
  __func__(5,unsigned,2,unsigned) \
  __func__(5,unsigned,2,long long) \
  __func__(5,unsigned,3,int) \
  __func__(5,unsigned,3,unsigned) \
  __func__(5,unsigned,3,long long) \
  __func__(5,unsigned,4,int) \
  __func__(5,unsigned,4,unsigned) \
  __func__(5,unsigned,4,long long) \
  __func__(5,unsigned,5,int) \
  __func__(5,unsigned,5,unsigned) \
  __func__(5,unsigned,5,long long) \
  __func__(5,unsigned,6,int) \
  __func__(5,unsigned,6,unsigned) \
  __func__(5,unsigned,6,long long) \
  __func__(5,unsigned,7,int) \
  __func__(5,unsigned,7,unsigned) \
  __func__(5,unsigned,7,long long) \
  __func__(5,unsigned,8,int) \
  __func__(5,unsigned,8,unsigned) \
  __func__(5,unsigned,8,long long) \
  __func__(5,unsigned,9,int) \
  __func__(5,unsigned,9,unsigned) \
  __func__(5,unsigned,9,long long) \
  __func__(5,long long,1,int) \
  __func__(5,long long,1,unsigned) \
  __func__(5,long long,1,long long) \
  __func__(5,long long,2,int) \
  __func__(5,long long,2,unsigned) \
  __func__(5,long long,2,long long) \
  __func__(5,long long,3,int) \
  __func__(5,long long,3,unsigned) \
  __func__(5,long long,3,long long) \
  __func__(5,long long,4,int) \
  __func__(5,long long,4,unsigned) \
  __func__(5,long long,4,long long) \
  __func__(5,long long,5,int) \
  __func__(5,long long,5,unsigned) \
  __func__(5,long long,5,long long) \
  __func__(5,long long,6,int) \
  __func__(5,long long,6,unsigned) \
  __func__(5,long long,6,long long) \
  __func__(5,long long,7,int) \
  __func__(5,long long,7,unsigned) \
  __func__(5,long long,7,long long) \
  __func__(5,long long,8,int) \
  __func__(5,long long,8,unsigned) \
  __func__(5,long long,8,long long) \
  __func__(5,long long,9,int) \
  __func__(5,long long,9,unsigned) \
  __func__(5,long long,9,long long) \
\
  __func__(6,int,1,int) \
  __func__(6,int,1,unsigned) \
  __func__(6,int,1,long long) \
  __func__(6,int,2,int) \
  __func__(6,int,2,unsigned) \
  __func__(6,int,2,long long) \
  __func__(6,int,3,int) \
  __func__(6,int,3,unsigned) \
  __func__(6,int,3,long long) \
  __func__(6,int,4,int) \
  __func__(6,int,4,unsigned) \
  __func__(6,int,4,long long) \
  __func__(6,int,5,int) \
  __func__(6,int,5,unsigned) \
  __func__(6,int,5,long long) \
  __func__(6,int,6,int) \
  __func__(6,int,6,unsigned) \
  __func__(6,int,6,long long) \
  __func__(6,int,7,int) \
  __func__(6,int,7,unsigned) \
  __func__(6,int,7,long long) \
  __func__(6,int,8,int) \
  __func__(6,int,8,unsigned) \
  __func__(6,int,8,long long) \
  __func__(6,int,9,int) \
  __func__(6,int,9,unsigned) \
  __func__(6,int,9,long long) \
  __func__(6,unsigned,1,int) \
  __func__(6,unsigned,1,unsigned) \
  __func__(6,unsigned,1,long long) \
  __func__(6,unsigned,2,int) \
  __func__(6,unsigned,2,unsigned) \
  __func__(6,unsigned,2,long long) \
  __func__(6,unsigned,3,int) \
  __func__(6,unsigned,3,unsigned) \
  __func__(6,unsigned,3,long long) \
  __func__(6,unsigned,4,int) \
  __func__(6,unsigned,4,unsigned) \
  __func__(6,unsigned,4,long long) \
  __func__(6,unsigned,5,int) \
  __func__(6,unsigned,5,unsigned) \
  __func__(6,unsigned,5,long long) \
  __func__(6,unsigned,6,int) \
  __func__(6,unsigned,6,unsigned) \
  __func__(6,unsigned,6,long long) \
  __func__(6,unsigned,7,int) \
  __func__(6,unsigned,7,unsigned) \
  __func__(6,unsigned,7,long long) \
  __func__(6,unsigned,8,int) \
  __func__(6,unsigned,8,unsigned) \
  __func__(6,unsigned,8,long long) \
  __func__(6,unsigned,9,int) \
  __func__(6,unsigned,9,unsigned) \
  __func__(6,unsigned,9,long long) \
  __func__(6,long long,1,int) \
  __func__(6,long long,1,unsigned) \
  __func__(6,long long,1,long long) \
  __func__(6,long long,2,int) \
  __func__(6,long long,2,unsigned) \
  __func__(6,long long,2,long long) \
  __func__(6,long long,3,int) \
  __func__(6,long long,3,unsigned) \
  __func__(6,long long,3,long long) \
  __func__(6,long long,4,int) \
  __func__(6,long long,4,unsigned) \
  __func__(6,long long,4,long long) \
  __func__(6,long long,5,int) \
  __func__(6,long long,5,unsigned) \
  __func__(6,long long,5,long long) \
  __func__(6,long long,6,int) \
  __func__(6,long long,6,unsigned) \
  __func__(6,long long,6,long long) \
  __func__(6,long long,7,int) \
  __func__(6,long long,7,unsigned) \
  __func__(6,long long,7,long long) \
  __func__(6,long long,8,int) \
  __func__(6,long long,8,unsigned) \
  __func__(6,long long,8,long long) \
  __func__(6,long long,9,int) \
  __func__(6,long long,9,unsigned) \
  __func__(6,long long,9,long long) \
\
  __func__(7,int,1,int) \
  __func__(7,int,1,unsigned) \
  __func__(7,int,1,long long) \
  __func__(7,int,2,int) \
  __func__(7,int,2,unsigned) \
  __func__(7,int,2,long long) \
  __func__(7,int,3,int) \
  __func__(7,int,3,unsigned) \
  __func__(7,int,3,long long) \
  __func__(7,int,4,int) \
  __func__(7,int,4,unsigned) \
  __func__(7,int,4,long long) \
  __func__(7,int,5,int) \
  __func__(7,int,5,unsigned) \
  __func__(7,int,5,long long) \
  __func__(7,int,6,int) \
  __func__(7,int,6,unsigned) \
  __func__(7,int,6,long long) \
  __func__(7,int,7,int) \
  __func__(7,int,7,unsigned) \
  __func__(7,int,7,long long) \
  __func__(7,int,8,int) \
  __func__(7,int,8,unsigned) \
  __func__(7,int,8,long long) \
  __func__(7,int,9,int) \
  __func__(7,int,9,unsigned) \
  __func__(7,int,9,long long) \
  __func__(7,unsigned,1,int) \
  __func__(7,unsigned,1,unsigned) \
  __func__(7,unsigned,1,long long) \
  __func__(7,unsigned,2,int) \
  __func__(7,unsigned,2,unsigned) \
  __func__(7,unsigned,2,long long) \
  __func__(7,unsigned,3,int) \
  __func__(7,unsigned,3,unsigned) \
  __func__(7,unsigned,3,long long) \
  __func__(7,unsigned,4,int) \
  __func__(7,unsigned,4,unsigned) \
  __func__(7,unsigned,4,long long) \
  __func__(7,unsigned,5,int) \
  __func__(7,unsigned,5,unsigned) \
  __func__(7,unsigned,5,long long) \
  __func__(7,unsigned,6,int) \
  __func__(7,unsigned,6,unsigned) \
  __func__(7,unsigned,6,long long) \
  __func__(7,unsigned,7,int) \
  __func__(7,unsigned,7,unsigned) \
  __func__(7,unsigned,7,long long) \
  __func__(7,unsigned,8,int) \
  __func__(7,unsigned,8,unsigned) \
  __func__(7,unsigned,8,long long) \
  __func__(7,unsigned,9,int) \
  __func__(7,unsigned,9,unsigned) \
  __func__(7,unsigned,9,long long) \
  __func__(7,long long,1,int) \
  __func__(7,long long,1,unsigned) \
  __func__(7,long long,1,long long) \
  __func__(7,long long,2,int) \
  __func__(7,long long,2,unsigned) \
  __func__(7,long long,2,long long) \
  __func__(7,long long,3,int) \
  __func__(7,long long,3,unsigned) \
  __func__(7,long long,3,long long) \
  __func__(7,long long,4,int) \
  __func__(7,long long,4,unsigned) \
  __func__(7,long long,4,long long) \
  __func__(7,long long,5,int) \
  __func__(7,long long,5,unsigned) \
  __func__(7,long long,5,long long) \
  __func__(7,long long,6,int) \
  __func__(7,long long,6,unsigned) \
  __func__(7,long long,6,long long) \
  __func__(7,long long,7,int) \
  __func__(7,long long,7,unsigned) \
  __func__(7,long long,7,long long) \
  __func__(7,long long,8,int) \
  __func__(7,long long,8,unsigned) \
  __func__(7,long long,8,long long) \
  __func__(7,long long,9,int) \
  __func__(7,long long,9,unsigned) \
  __func__(7,long long,9,long long) \
\
  __func__(8,int,1,int) \
  __func__(8,int,1,unsigned) \
  __func__(8,int,1,long long) \
  __func__(8,int,2,int) \
  __func__(8,int,2,unsigned) \
  __func__(8,int,2,long long) \
  __func__(8,int,3,int) \
  __func__(8,int,3,unsigned) \
  __func__(8,int,3,long long) \
  __func__(8,int,4,int) \
  __func__(8,int,4,unsigned) \
  __func__(8,int,4,long long) \
  __func__(8,int,5,int) \
  __func__(8,int,5,unsigned) \
  __func__(8,int,5,long long) \
  __func__(8,int,6,int) \
  __func__(8,int,6,unsigned) \
  __func__(8,int,6,long long) \
  __func__(8,int,7,int) \
  __func__(8,int,7,unsigned) \
  __func__(8,int,7,long long) \
  __func__(8,int,8,int) \
  __func__(8,int,8,unsigned) \
  __func__(8,int,8,long long) \
  __func__(8,int,9,int) \
  __func__(8,int,9,unsigned) \
  __func__(8,int,9,long long) \
  __func__(8,unsigned,1,int) \
  __func__(8,unsigned,1,unsigned) \
  __func__(8,unsigned,1,long long) \
  __func__(8,unsigned,2,int) \
  __func__(8,unsigned,2,unsigned) \
  __func__(8,unsigned,2,long long) \
  __func__(8,unsigned,3,int) \
  __func__(8,unsigned,3,unsigned) \
  __func__(8,unsigned,3,long long) \
  __func__(8,unsigned,4,int) \
  __func__(8,unsigned,4,unsigned) \
  __func__(8,unsigned,4,long long) \
  __func__(8,unsigned,5,int) \
  __func__(8,unsigned,5,unsigned) \
  __func__(8,unsigned,5,long long) \
  __func__(8,unsigned,6,int) \
  __func__(8,unsigned,6,unsigned) \
  __func__(8,unsigned,6,long long) \
  __func__(8,unsigned,7,int) \
  __func__(8,unsigned,7,unsigned) \
  __func__(8,unsigned,7,long long) \
  __func__(8,unsigned,8,int) \
  __func__(8,unsigned,8,unsigned) \
  __func__(8,unsigned,8,long long) \
  __func__(8,unsigned,9,int) \
  __func__(8,unsigned,9,unsigned) \
  __func__(8,unsigned,9,long long) \
  __func__(8,long long,1,int) \
  __func__(8,long long,1,unsigned) \
  __func__(8,long long,1,long long) \
  __func__(8,long long,2,int) \
  __func__(8,long long,2,unsigned) \
  __func__(8,long long,2,long long) \
  __func__(8,long long,3,int) \
  __func__(8,long long,3,unsigned) \
  __func__(8,long long,3,long long) \
  __func__(8,long long,4,int) \
  __func__(8,long long,4,unsigned) \
  __func__(8,long long,4,long long) \
  __func__(8,long long,5,int) \
  __func__(8,long long,5,unsigned) \
  __func__(8,long long,5,long long) \
  __func__(8,long long,6,int) \
  __func__(8,long long,6,unsigned) \
  __func__(8,long long,6,long long) \
  __func__(8,long long,7,int) \
  __func__(8,long long,7,unsigned) \
  __func__(8,long long,7,long long) \
  __func__(8,long long,8,int) \
  __func__(8,long long,8,unsigned) \
  __func__(8,long long,8,long long) \
  __func__(8,long long,9,int) \
  __func__(8,long long,9,unsigned) \
  __func__(8,long long,9,long long) \
\
  __func__(9,int,1,int) \
  __func__(9,int,1,unsigned) \
  __func__(9,int,1,long long) \
  __func__(9,int,2,int) \
  __func__(9,int,2,unsigned) \
  __func__(9,int,2,long long) \
  __func__(9,int,3,int) \
  __func__(9,int,3,unsigned) \
  __func__(9,int,3,long long) \
  __func__(9,int,4,int) \
  __func__(9,int,4,unsigned) \
  __func__(9,int,4,long long) \
  __func__(9,int,5,int) \
  __func__(9,int,5,unsigned) \
  __func__(9,int,5,long long) \
  __func__(9,int,6,int) \
  __func__(9,int,6,unsigned) \
  __func__(9,int,6,long long) \
  __func__(9,int,7,int) \
  __func__(9,int,7,unsigned) \
  __func__(9,int,7,long long) \
  __func__(9,int,8,int) \
  __func__(9,int,8,unsigned) \
  __func__(9,int,8,long long) \
  __func__(9,int,9,int) \
  __func__(9,int,9,unsigned) \
  __func__(9,int,9,long long) \
  __func__(9,unsigned,1,int) \
  __func__(9,unsigned,1,unsigned) \
  __func__(9,unsigned,1,long long) \
  __func__(9,unsigned,2,int) \
  __func__(9,unsigned,2,unsigned) \
  __func__(9,unsigned,2,long long) \
  __func__(9,unsigned,3,int) \
  __func__(9,unsigned,3,unsigned) \
  __func__(9,unsigned,3,long long) \
  __func__(9,unsigned,4,int) \
  __func__(9,unsigned,4,unsigned) \
  __func__(9,unsigned,4,long long) \
  __func__(9,unsigned,5,int) \
  __func__(9,unsigned,5,unsigned) \
  __func__(9,unsigned,5,long long) \
  __func__(9,unsigned,6,int) \
  __func__(9,unsigned,6,unsigned) \
  __func__(9,unsigned,6,long long) \
  __func__(9,unsigned,7,int) \
  __func__(9,unsigned,7,unsigned) \
  __func__(9,unsigned,7,long long) \
  __func__(9,unsigned,8,int) \
  __func__(9,unsigned,8,unsigned) \
  __func__(9,unsigned,8,long long) \
  __func__(9,unsigned,9,int) \
  __func__(9,unsigned,9,unsigned) \
  __func__(9,unsigned,9,long long) \
  __func__(9,long long,1,int) \
  __func__(9,long long,1,unsigned) \
  __func__(9,long long,1,long long) \
  __func__(9,long long,2,int) \
  __func__(9,long long,2,unsigned) \
  __func__(9,long long,2,long long) \
  __func__(9,long long,3,int) \
  __func__(9,long long,3,unsigned) \
  __func__(9,long long,3,long long) \
  __func__(9,long long,4,int) \
  __func__(9,long long,4,unsigned) \
  __func__(9,long long,4,long long) \
  __func__(9,long long,5,int) \
  __func__(9,long long,5,unsigned) \
  __func__(9,long long,5,long long) \
  __func__(9,long long,6,int) \
  __func__(9,long long,6,unsigned) \
  __func__(9,long long,6,long long) \
  __func__(9,long long,7,int) \
  __func__(9,long long,7,unsigned) \
  __func__(9,long long,7,long long) \
  __func__(9,long long,8,int) \
  __func__(9,long long,8,unsigned) \
  __func__(9,long long,8,long long) \
  __func__(9,long long,9,int) \
  __func__(9,long long,9,unsigned) \
  __func__(9,long long,9,long long) \

#else
#error "Illegal value of REALM_MAX_DIM"
#endif

#endif // defined REALM_DEPPART_INST_HELPER_H
