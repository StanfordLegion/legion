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

// reduction ops for Realm

#ifndef REALM_REDOP_H
#define REALM_REDOP_H

#include "realm/realm_config.h"

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_redop.h"
#endif

#ifdef REALM_USE_HIP
#include "realm/hip/hip_redop.h"
#endif

#include <cstddef>

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

#ifdef REALM_USE_CUDA
      // CUDA kernels for apply/fold - these are not actually the functions,
      //  but just information (e.g. host wrapper fnptr) that can be used
      //  to look up the actual kernels
      void *cuda_apply_excl_fn, *cuda_apply_nonexcl_fn;
      void *cuda_fold_excl_fn, *cuda_fold_nonexcl_fn;

      // These function pointers make the connection to the app's runtime
      // instance in order to properly translate and capture the correct
      // function to launch.
      // The runtime function pointer to launch these shadow symbols
      void *cudaLaunchKernel_fn;
      // The runtime function pointer to translate the host shadow symbol to a driver function
      void *cudaGetFuncBySymbol_fn;
#endif
#ifdef REALM_USE_HIP
      // HIP kernels for apply/fold - these are not actually the functions,
      //  but just information (e.g. host wrapper fnptr) that can be used
      //  to look up the actual kernels
      void *hip_apply_excl_fn, *hip_apply_nonexcl_fn;
      void *hip_fold_excl_fn, *hip_fold_nonexcl_fn;
#endif

      ReductionOpUntyped()
      : sizeof_this(sizeof(ReductionOpUntyped))
      , sizeof_lhs(0)
      , sizeof_rhs(0)
      , sizeof_userdata(0)
      , identity(0)
      , userdata(0)
      , cpu_apply_excl_fn(0)
      , cpu_apply_nonexcl_fn(0)
      , cpu_fold_excl_fn(0)
      , cpu_fold_nonexcl_fn(0)
#ifdef REALM_USE_CUDA
      , cuda_apply_excl_fn(0)
      , cuda_apply_nonexcl_fn(0)
      , cuda_fold_excl_fn(0)
      , cuda_fold_nonexcl_fn(0)
      , cudaLaunchKernel_fn(0)
      , cudaGetFuncBySymbol_fn(0)
#endif
#ifdef REALM_USE_HIP
      , hip_apply_excl_fn(0)
      , hip_apply_nonexcl_fn(0)
      , hip_fold_excl_fn(0)
      , hip_fold_nonexcl_fn(0)
#endif
      {}

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

#if defined(REALM_USE_CUDA) && defined(__CUDACC__)
    // with a cuda-capable compiler, we'll automatically add cuda reduction
    //  kernels if the REDOP class defines has_cuda_reductions AND it's true
    // this requires a bunch of SFINAE template-fu
    template <typename T>
    struct HasHasCudaReductions {
      struct YES { char dummy[1]; };
      struct NO { char dummy[2]; };
      struct AltnerativeDefinition { static const bool has_cuda_reductions = false; };
      template <typename T2> struct Combined : public T2, public AltnerativeDefinition {};
      template <typename T2, T2> struct CheckAmbiguous {};
      template <typename T2> static NO has_member(CheckAmbiguous<const bool *, &Combined<T2>::has_cuda_reductions> *);
      template <typename T2> static YES has_member(...);
      const static bool value = sizeof(has_member<T>(0)) == sizeof(YES);
    };

    template <typename T, bool OK> struct MaybeAddCudaReductions;
    template <typename T>
    struct MaybeAddCudaReductions<T, false> {
      static void if_member_exists(ReductionOpUntyped *redop) {};
      static void if_member_is_true(ReductionOpUntyped *redop) {};
    };
    template <typename T>
    struct MaybeAddCudaReductions<T, true> {
      static void if_member_exists(ReductionOpUntyped *redop) { MaybeAddCudaReductions<T, T::has_cuda_reductions>::if_member_is_true(redop); }
      static void if_member_is_true(ReductionOpUntyped *redop) { Cuda::add_cuda_redop_kernels<T>(redop); }
    };
#endif
	
#if defined(REALM_USE_HIP) && ( defined (__CUDACC__) || defined (__HIPCC__) )
    // with a hip-capable compiler, we'll automatically add hip reduction
    //  kernels if the REDOP class defines has_hip_reductions AND it's true
    // this requires a bunch of SFINAE template-fu
    template <typename T>
    struct HasHasHipReductions {
      struct YES { char dummy[1]; };
      struct NO { char dummy[2]; };
      struct AltnerativeDefinition { static const bool has_hip_reductions = false; };
      template <typename T2> struct Combined : public T2, public AltnerativeDefinition {};
      template <typename T2, T2> struct CheckAmbiguous {};
      template <typename T2> static NO has_member(CheckAmbiguous<const bool *, &Combined<T2>::has_hip_reductions> *);
      template <typename T2> static YES has_member(...);
      const static bool value = sizeof(has_member<T>(0)) == sizeof(YES);
    };

    template <typename T, bool OK> struct MaybeAddHipReductions;
    template <typename T>
    struct MaybeAddHipReductions<T, false> {
      static void if_member_exists(ReductionOpUntyped *redop) {};
      static void if_member_is_true(ReductionOpUntyped *redop) {};
    };
    template <typename T>
    struct MaybeAddHipReductions<T, true> {
      static void if_member_exists(ReductionOpUntyped *redop) { MaybeAddHipReductions<T, T::has_hip_reductions>::if_member_is_true(redop); }
      static void if_member_is_true(ReductionOpUntyped *redop) { Hip::add_hip_redop_kernels<T>(redop); }
    };
#endif

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
#if defined(REALM_USE_CUDA) && defined(__CUDACC__)
        // if REDOP defines/sets 'has_cuda_reductions' to true, try to
        //  automatically build wrappers for apply_cuda<> and fold_cuda<>
        MaybeAddCudaReductions<REDOP, HasHasCudaReductions<REDOP>::value>::if_member_exists(this);
#endif
#if defined(REALM_USE_HIP) && ( defined (__CUDACC__) || defined (__HIPCC__) )
        // if REDOP defines/sets 'has_hip_reductions' to true, try to
        //  automatically build wrappers for apply_hip<> and fold_hip<>
        MaybeAddHipReductions<REDOP, HasHasHipReductions<REDOP>::value>::if_member_exists(this);
#endif
      }

    protected:
    };

}; // namespace Realm

//include "redop.inl"

#endif // ifndef REALM_REDOP_H
