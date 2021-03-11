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

#include "realm/realm_config.h"

namespace Realm {

    // a reduction op needs to look like this
#ifdef NOT_REALLY_CODE
    class MyReductionOp {
    public:
      typedef int LHS;
      typedef int RHS;

      void apply(LHS& lhs, RHS rhs) const;

      // both of these are optional
      static const RHS identity;
      void fold(RHS& rhs1, RHS rhs2) const;
    };
#endif

    template <typename REDOP>
    struct ReductionOp;

    struct ReductionOpUntyped {
      size_t sizeof_this;  // includes any identity val or user data after struct
      size_t sizeof_lhs;
      size_t sizeof_rhs;
      size_t sizeof_userdata;  // extra data supplied to apply/fold
      void *identity;          // if non-null, points into same object
      void *userdata;          // if non-null, points into same object

      // CPU apply/fold functions - tolerate strided src/dst
      void (*cpu_apply_excl_fn)(void *lhs_ptr, size_t lhs_stride,
                                const void *rhs_ptr, size_t rhs_stride,
                                size_t count, const void *userdata);
      void (*cpu_apply_nonexcl_fn)(void *lhs_ptr, size_t lhs_stride,
                                   const void *rhs_ptr, size_t rhs_stride,
                                   size_t count, const void *userdata);
      void (*cpu_fold_excl_fn)(void *rhs1_ptr, size_t rhs1_stride,
                               const void *rhs2_ptr, size_t rhs2_stride,
                               size_t count, const void *userdata);
      void (*cpu_fold_nonexcl_fn)(void *rhs1_ptr, size_t rhs1_stride,
                                  const void *rhs2_ptr, size_t rhs2_stride,
                                  size_t count, const void *userdata);

      template <class REDOP>
      static ReductionOpUntyped *create_reduction_op(void)
      {
        ReductionOpUntyped *redop = new ReductionOp<REDOP>;
        return redop;
      }

      static ReductionOpUntyped *clone_reduction_op(const ReductionOpUntyped *redop);
    };

    namespace ReductionKernels {
      template <typename REDOP, bool EXCL>
      void cpu_apply_wrapper(void *lhs_ptr, size_t lhs_stride,
                             const void *rhs_ptr, size_t rhs_stride,
                             size_t count, const void *userdata)
      {
        const REDOP *redop = static_cast<const REDOP *>(userdata);
        for(size_t i = 0; i < count; i++) {
          redop->template apply<EXCL>(*static_cast<typename REDOP::LHS *>(lhs_ptr),
                                      *static_cast<const typename REDOP::RHS *>(rhs_ptr));
          lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
          rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
        }
      }

      template <typename REDOP, bool EXCL>
      void cpu_fold_wrapper(void *rhs1_ptr, size_t rhs1_stride,
                            const void *rhs2_ptr, size_t rhs2_stride,
                            size_t count, const void *userdata)
      {
        const REDOP *redop = static_cast<const REDOP *>(userdata);
        for(size_t i = 0; i < count; i++) {
          redop->template fold<EXCL>(*static_cast<typename REDOP::RHS *>(rhs1_ptr),
                                     *static_cast<const typename REDOP::RHS *>(rhs2_ptr));
          rhs1_ptr = static_cast<char *>(rhs1_ptr) + rhs1_stride;
          rhs2_ptr = static_cast<const char *>(rhs2_ptr) + rhs2_stride;
        }
      }
    };

    template <typename REDOP>
    struct ReductionOp : public ReductionOpUntyped {
      // tacked on to end of ReductionOpUntyped struct
      typename REDOP::RHS identity_val;
      REDOP userdata_val;

      ReductionOp()
        : identity_val(REDOP::identity)
        , userdata_val()
      {
        sizeof_this = sizeof(ReductionOp<REDOP>);
        sizeof_lhs = sizeof(typename REDOP::LHS);
        sizeof_rhs = sizeof(typename REDOP::RHS);
        sizeof_userdata = sizeof(REDOP);
        identity = &identity_val;
        userdata = &userdata_val;
        cpu_apply_excl_fn = &ReductionKernels::cpu_apply_wrapper<REDOP, true>;
        cpu_apply_nonexcl_fn = &ReductionKernels::cpu_apply_wrapper<REDOP, false>;
        cpu_fold_excl_fn = &ReductionKernels::cpu_fold_wrapper<REDOP, true>;
        cpu_fold_nonexcl_fn = &ReductionKernels::cpu_fold_wrapper<REDOP, false>;
      }

    protected:
    };

}; // namespace Realm

//include "redop.inl"

#endif // ifndef REALM_REDOP_H


