/* Copyright 2021 Stanford University, NVIDIA Corporation
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

// reduction ops for Realm

#ifndef REALM_REDOP_H
#define REALM_REDOP_H

#include <sys/types.h>

namespace Realm {

    // a reduction op needs to look like this
#ifdef NOT_REALLY_CODE
    class MyReductionOp {
    public:
      typedef int LHS;
      typedef int RHS;

      static void apply(LHS& lhs, RHS rhs);

      // both of these are optional
      static const RHS identity;
      static void fold(RHS& rhs1, RHS rhs2);
    };
#endif

    class ReductionOpUntyped {
    public:
      size_t sizeof_lhs;
      size_t sizeof_rhs;
      size_t sizeof_list_entry;
      bool has_identity;
      bool is_foldable;

      template <class REDOP>
	static ReductionOpUntyped *create_reduction_op(void);

      virtual void apply(void *lhs_ptr, const void *rhs_ptr, size_t count,
			 bool exclusive = false) const = 0;
      virtual void apply_strided(void *lhs_ptr, const void *rhs_ptr,
				 off_t lhs_stride, off_t rhs_stride, size_t count,
				 bool exclusive = false) const = 0;
      virtual void fold(void *rhs1_ptr, const void *rhs2_ptr, size_t count,
			bool exclusive = false) const = 0;
      virtual void fold_strided(void *lhs_ptr, const void *rhs_ptr,
				off_t lhs_stride, off_t rhs_stride, size_t count,
				bool exclusive = false) const = 0;
      virtual void init(void *rhs_ptr, size_t count) const = 0;

#ifdef NEED_TO_FIX_REDUCTION_LISTS_FOR_DEPPART
      virtual void apply_list_entry(void *lhs_ptr, const void *entry_ptr, size_t count,
				    off_t ptr_offset, bool exclusive = false) const = 0;
      virtual void fold_list_entry(void *rhs_ptr, const void *entry_ptr, size_t count,
                                    off_t ptr_offset, bool exclusive = false) const = 0;
      virtual void get_list_pointers(unsigned *ptrs, const void *entry_ptr, size_t count) const = 0;
#endif

      virtual ~ReductionOpUntyped() {}

      virtual ReductionOpUntyped *clone(void) const = 0;

    protected:
      ReductionOpUntyped(size_t _sizeof_lhs, size_t _sizeof_rhs,
			 size_t _sizeof_list_entry,
			 bool _has_identity, bool _is_foldable)
	: sizeof_lhs(_sizeof_lhs), sizeof_rhs(_sizeof_rhs),
	  sizeof_list_entry(_sizeof_list_entry),
  	  has_identity(_has_identity), is_foldable(_is_foldable) {}
    };

#ifdef NEED_TO_FIX_REDUCTION_LISTS_FOR_DEPPART
    template <class LHS, class RHS>
    struct ReductionListEntry {
      ptr_t ptr;
      RHS rhs;
    };
#endif

    template <class REDOP>
    class ReductionOp : public ReductionOpUntyped {
    public:
      // TODO: don't assume identity and fold are available - use scary
      //  template-fu to figure it out
      ReductionOp(void)
	: ReductionOpUntyped(sizeof(typename REDOP::LHS), sizeof(typename REDOP::RHS),
#ifdef NEED_TO_FIX_REDUCTION_LISTS_FOR_DEPPART
			     sizeof(ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS>),
#else
			     0,
#endif
			     true, true) {}

      virtual ReductionOpUntyped *clone(void) const
      {
	return new ReductionOp<REDOP>;
      }

      virtual void apply(void *lhs_ptr, const void *rhs_ptr, size_t count,
			 bool exclusive = false) const
      {
	typename REDOP::LHS *lhs = static_cast<typename REDOP::LHS *>(lhs_ptr);
	const typename REDOP::RHS *rhs = static_cast<const typename REDOP::RHS *>(rhs_ptr);
	if(exclusive) {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<true>(lhs[i], rhs[i]);
	} else {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<false>(lhs[i], rhs[i]);
	}
      }

      virtual void apply_strided(void *lhs_ptr, const void *rhs_ptr,
				 off_t lhs_stride, off_t rhs_stride, size_t count,
				 bool exclusive = false) const
      {
	if(exclusive) {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template apply<true>(*static_cast<typename REDOP::LHS *>(lhs_ptr),
					*static_cast<const typename REDOP::RHS *>(rhs_ptr));
	    lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
	    rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
	  }
	} else {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template apply<false>(*static_cast<typename REDOP::LHS *>(lhs_ptr),
					 *static_cast<const typename REDOP::RHS *>(rhs_ptr));
	    lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
	    rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
	  }
	}
      }

      virtual void fold(void *rhs1_ptr, const void *rhs2_ptr, size_t count,
			bool exclusive = false) const
      {
	typename REDOP::RHS *rhs1 = static_cast<typename REDOP::RHS *>(rhs1_ptr);
	const typename REDOP::RHS *rhs2 = static_cast<const typename REDOP::RHS *>(rhs2_ptr);
	if(exclusive) {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template fold<true>(rhs1[i], rhs2[i]);
	} else {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template fold<false>(rhs1[i], rhs2[i]);
	}
      }

      virtual void fold_strided(void *lhs_ptr, const void *rhs_ptr,
				off_t lhs_stride, off_t rhs_stride, size_t count,
				bool exclusive = false) const
      {
	if(exclusive) {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template fold<true>(*static_cast<typename REDOP::RHS *>(lhs_ptr),
				       *static_cast<const typename REDOP::RHS *>(rhs_ptr));
	    lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
	    rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
	  }
	} else {
	  for(size_t i = 0; i < count; i++) {
	    REDOP::template fold<false>(*static_cast<typename REDOP::RHS *>(lhs_ptr),
					*static_cast<const typename REDOP::RHS *>(rhs_ptr));
	    lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
	    rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
	  }
	}
      }

      virtual void init(void *ptr, size_t count) const
      {
        typename REDOP::RHS *rhs_ptr = static_cast<typename REDOP::RHS *>(ptr);
        for (size_t i = 0; i < count; i++)
          *rhs_ptr++ = REDOP::identity;
      }

#ifdef NEED_TO_FIX_REDUCTION_LISTS_FOR_DEPPART
      virtual void apply_list_entry(void *lhs_ptr, const void *entry_ptr, size_t count,
				    off_t ptr_offset, bool exclusive = false) const
      {
	typename REDOP::LHS *lhs = (typename REDOP::LHS *)lhs_ptr;
	const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *entry = (const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *)entry_ptr;
	if(exclusive) {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<true>(lhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
	} else {
	  for(size_t i = 0; i < count; i++)
	    REDOP::template apply<false>(lhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
	}
      }

      virtual void fold_list_entry(void *rhs_ptr, const void *entry_ptr, size_t count,
                                    off_t ptr_offset, bool exclusive = false) const
      {
        typename REDOP::RHS *rhs = (typename REDOP::RHS*)rhs_ptr;
        const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *entry = (const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *)entry_ptr;
        if (exclusive)
        {
          for (size_t i = 0; i < count; i++)
            REDOP::template fold<true>(rhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
        }
        else
        {
          for (size_t i = 0; i < count; i++)
            REDOP::template fold<false>(rhs[entry[i].ptr.value - ptr_offset], entry[i].rhs);
        }
      }

      virtual void get_list_pointers(unsigned *ptrs, const void *entry_ptr, size_t count) const
      {
	const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *entry = (const ReductionListEntry<typename REDOP::LHS,typename REDOP::RHS> *)entry_ptr;
	for(size_t i = 0; i < count; i++) {
	  ptrs[i] = entry[i].ptr.value;
	  //printf("%d=%d\n", i, ptrs[i]);
	}
      }
#endif
    };

    template <class REDOP>
    ReductionOpUntyped *ReductionOpUntyped::create_reduction_op(void)
    {
      ReductionOp<REDOP> *redop = new ReductionOp<REDOP>();
      return redop;
    }

}; // namespace Realm

//include "redop.inl"

#endif // ifndef REALM_REDOP_H


