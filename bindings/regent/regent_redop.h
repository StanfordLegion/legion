/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef __REGENT_REDUCTIONS_H__
#define __REGENT_REDUCTIONS_H__

// Regent uses Legion's built-in reductions for base types, but needs to define
//  its own for fields that are arrays of base types

template <class ELEM_REDOP>
struct ArrayReduction {
  unsigned N;

  typedef typename ELEM_REDOP::LHS LHS;
  typedef typename ELEM_REDOP::RHS RHS;

  ArrayReduction(unsigned _N) : N(_N) {}

  // these methods take references to the rhs so that we can access the
  //  other array elements behind them - the Realm wrappers are ok with this
  template <bool EXCL>
  void apply(LHS& lhs, const RHS& rhs) const
  {
    for (unsigned i = 0; i < N; i++)
      ELEM_REDOP::template apply<EXCL>((&lhs)[i], (&rhs)[i]);
  }

  template <bool EXCL>
  void fold(RHS& rhs1, const RHS& rhs2) const
  {
    for (unsigned i = 0; i < N; i++)
      ELEM_REDOP::template fold<EXCL>((&rhs1)[i], (&rhs2)[i]);
  }

#ifdef __CUDACC__
  template <bool EXCL>
  __device__ void apply_cuda(LHS& lhs, const RHS& rhs) const
  {
    for (unsigned i = 0; i < N; i++)
      ELEM_REDOP::template apply<EXCL>((&lhs)[i], (&rhs)[i]);
  }

  template <bool EXCL>
  __device__ void fold_cuda(RHS& rhs1, const RHS& rhs2) const
  {
    for (unsigned i = 0; i < N; i++)
      ELEM_REDOP::template fold<EXCL>((&rhs1)[i], (&rhs2)[i]);
  }
#endif

#ifdef __HIPCC__
  template <bool EXCL>
  __device__ void apply_hip(LHS& lhs, const RHS& rhs) const
  {
    for (unsigned i = 0; i < N; i++)
      ELEM_REDOP::template apply<EXCL>((&lhs)[i], (&rhs)[i]);
  }

  template <bool EXCL>
  __device__ void fold_hip(RHS& rhs1, const RHS& rhs2) const
  {
    for (unsigned i = 0; i < N; i++)
      ELEM_REDOP::template fold<EXCL>((&rhs1)[i], (&rhs2)[i]);
  }
#endif
};

template <typename ELEM_REDOP>
struct ArrayReductionOp : public Realm::ReductionOpUntyped {
  // tacked on to end of ReductionOpUntyped struct
  ArrayReduction<ELEM_REDOP> userdata_val;
  typename ELEM_REDOP::RHS identity_val[1 /*really N*/];

protected:
  ArrayReductionOp(unsigned N)
    : userdata_val(N)
  {
    for (unsigned i = 0; i < N; i++)
      identity_val[i] = ELEM_REDOP::identity;

    sizeof_this = (sizeof(ArrayReductionOp<ELEM_REDOP>) +
                   ((N - 1) * sizeof(typename ELEM_REDOP::RHS)));
    sizeof_lhs = sizeof(typename ELEM_REDOP::LHS) * N;
    sizeof_rhs = sizeof(typename ELEM_REDOP::RHS) * N;
    sizeof_userdata = sizeof(ArrayReduction<ELEM_REDOP>);
    identity = identity_val;
    userdata = &userdata_val;
    cpu_apply_excl_fn = &Realm::ReductionKernels::cpu_apply_wrapper<ArrayReduction<ELEM_REDOP>, true>;
    cpu_apply_nonexcl_fn = &Realm::ReductionKernels::cpu_apply_wrapper<ArrayReduction<ELEM_REDOP>, false>;
    cpu_fold_excl_fn = &Realm::ReductionKernels::cpu_fold_wrapper<ArrayReduction<ELEM_REDOP>, true>;
    cpu_fold_nonexcl_fn = &Realm::ReductionKernels::cpu_fold_wrapper<ArrayReduction<ELEM_REDOP>, false>;
#if defined(LEGION_USE_CUDA) && defined(__CUDACC__)
    Realm::Cuda::add_cuda_redop_kernels<ArrayReduction<ELEM_REDOP> >(this);
#elif defined(LEGION_USE_HIP) && defined(__HIPCC__)
    Realm::Hip::add_hip_redop_kernels<ArrayReduction<ELEM_REDOP> >(this);
#endif
  }

public:
  static ArrayReductionOp<ELEM_REDOP> *create_reduction_op(unsigned N)
  {
    size_t bytes = (sizeof(ArrayReductionOp<ELEM_REDOP>) +
                    ((N - 1) * sizeof(typename ELEM_REDOP::RHS)));
    void *ptr = malloc(bytes);
    assert(ptr);
    return new(ptr) ArrayReductionOp<ELEM_REDOP>(N);
  }
};

#define REGENT_ARRAY_REDUCE_LIST_BASE(__op__) \
  __op__(register_array_reduction_plus_float  , SumReduction<float>) \
  __op__(register_array_reduction_plus_double , SumReduction<double>) \
  __op__(register_array_reduction_plus_int16  , SumReduction<int16_t>) \
  __op__(register_array_reduction_plus_int32  , SumReduction<int32_t>) \
  __op__(register_array_reduction_plus_int64  , SumReduction<int64_t>) \
  __op__(register_array_reduction_plus_uint16 , SumReduction<uint16_t>) \
  __op__(register_array_reduction_plus_uint32 , SumReduction<uint32_t>) \
  __op__(register_array_reduction_plus_uint64 , SumReduction<uint64_t>) \
  __op__(register_array_reduction_minus_float  , DiffReduction<float>) \
  __op__(register_array_reduction_minus_double , DiffReduction<double>) \
  __op__(register_array_reduction_minus_int16  , DiffReduction<int16_t>) \
  __op__(register_array_reduction_minus_int32  , DiffReduction<int32_t>) \
  __op__(register_array_reduction_minus_int64  , DiffReduction<int64_t>) \
  __op__(register_array_reduction_minus_uint16 , DiffReduction<uint16_t>) \
  __op__(register_array_reduction_minus_uint32 , DiffReduction<uint32_t>) \
  __op__(register_array_reduction_minus_uint64 , DiffReduction<uint64_t>) \
  __op__(register_array_reduction_times_float  , ProdReduction<float>) \
  __op__(register_array_reduction_times_double , ProdReduction<double>) \
  __op__(register_array_reduction_times_int16  , ProdReduction<int16_t>) \
  __op__(register_array_reduction_times_int32  , ProdReduction<int32_t>) \
  __op__(register_array_reduction_times_int64  , ProdReduction<int64_t>) \
  __op__(register_array_reduction_times_uint16 , ProdReduction<uint16_t>) \
  __op__(register_array_reduction_times_uint32 , ProdReduction<uint32_t>) \
  __op__(register_array_reduction_times_uint64 , ProdReduction<uint64_t>) \
  __op__(register_array_reduction_divide_float  , DivReduction<float>) \
  __op__(register_array_reduction_divide_double , DivReduction<double>) \
  __op__(register_array_reduction_divide_int16  , DivReduction<int16_t>) \
  __op__(register_array_reduction_divide_int32  , DivReduction<int32_t>) \
  __op__(register_array_reduction_divide_int64  , DivReduction<int64_t>) \
  __op__(register_array_reduction_divide_uint16 , DivReduction<uint16_t>) \
  __op__(register_array_reduction_divide_uint32 , DivReduction<uint32_t>) \
  __op__(register_array_reduction_divide_uint64 , DivReduction<uint64_t>) \
  __op__(register_array_reduction_max_float  , MaxReduction<float>) \
  __op__(register_array_reduction_max_double , MaxReduction<double>) \
  __op__(register_array_reduction_max_int16  , MaxReduction<int16_t>) \
  __op__(register_array_reduction_max_int32  , MaxReduction<int32_t>) \
  __op__(register_array_reduction_max_int64  , MaxReduction<int64_t>) \
  __op__(register_array_reduction_max_uint16 , MaxReduction<uint16_t>) \
  __op__(register_array_reduction_max_uint32 , MaxReduction<uint32_t>) \
  __op__(register_array_reduction_max_uint64 , MaxReduction<uint64_t>) \
  __op__(register_array_reduction_min_float  , MinReduction<float>) \
  __op__(register_array_reduction_min_double , MinReduction<double>) \
  __op__(register_array_reduction_min_int16  , MinReduction<int16_t>) \
  __op__(register_array_reduction_min_int32  , MinReduction<int32_t>) \
  __op__(register_array_reduction_min_int64  , MinReduction<int64_t>) \
  __op__(register_array_reduction_min_uint16 , MinReduction<uint16_t>) \
  __op__(register_array_reduction_min_uint32 , MinReduction<uint32_t>) \
  __op__(register_array_reduction_min_uint64 , MinReduction<uint64_t>)

#ifdef LEGION_REDOP_COMPLEX
  #define REGENT_ARRAY_REDUCE_LIST_COMPLEX(__op__) \
  __op__(register_array_reduction_plus_complex64 , SumReduction<complex<float> >) \
  __op__(register_array_reduction_minus_complex64 , DiffReduction<complex<float> >) \
  __op__(register_array_reduction_times_complex64 , ProdReduction<complex<float> >) \
  __op__(register_array_reduction_divide_complex64 , DivReduction<complex<float> >)
#else
  #define REGENT_ARRAY_REDUCE_LIST_COMPLEX(__op__)
#endif

#define REGENT_ARRAY_REDUCE_LIST(__op__) \
  REGENT_ARRAY_REDUCE_LIST_BASE(__op__) \
  REGENT_ARRAY_REDUCE_LIST_COMPLEX(__op__)

#endif
