#ifndef SIMPLE_REDUCE_H
#define SIMPLE_REDUCE_H

#include <cstdint>
#include <cstring>
#if defined(REALM_ON_WINDOWS)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

class ReductionOpMixedAdd {
public:
  typedef uint64_t LHS;
  typedef uint32_t RHS;

  // You can specify user data here that can be applied to all elements as data members
  // here.
  RHS offset = 1;
  // The identity for fold operations
  static const RHS identity = 0;

  // The reduction operation definitions all processors will use
  template <bool EXCL>
  REALM_CUDA_HD void apply(LHS &lhs, const RHS &rhs) const
  {
    if(EXCL) {
      lhs += rhs * offset;
    } else {
      atomic_add(lhs, rhs * offset);
    }
  }

  template <bool EXCL>
  REALM_CUDA_HD void fold(RHS &rhs1, const RHS &rhs2) const
  {
    if(EXCL) {
      rhs1 += rhs2 * offset;
    } else {
      atomic_add(rhs1, rhs2 * offset);
    }
  }

  // Provide a platform agnostic version of atomically incrementing an element by some
  // given amount
  static REALM_CUDA_HD void atomic_add(LHS &lhs, const RHS &rhs)
  {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicAdd((unsigned long long int *)&lhs, (unsigned long long)rhs);
#elif defined(_MSC_VER)
    InterlockedAdd64((volatile int64_t *)&lhs, (int64_t)rhs);
#else
// Would be nice to use atomic_ref here...
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Watomic-alignment"
#endif
    __atomic_add_fetch(&lhs, (LHS)rhs, __ATOMIC_SEQ_CST);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
  }

  static REALM_CUDA_HD void atomic_add(RHS &rhs1, const RHS &rhs2)
  {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicAdd(&rhs1, rhs2);
#elif defined(_MSC_VER)
    InterlockedAdd((volatile LONG *)&rhs1, (LONG)rhs2);
#else
// Would be nice to use atomic_ref here...
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Watomic-alignment"
#endif
    __atomic_add_fetch(&rhs1, rhs2, __ATOMIC_SEQ_CST);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
  }

#if defined(REALM_USE_CUDA) && defined(__CUDACC__)
  // We don't actually want to register this the normal way for cuda in order to test the
  // auxilary way for registration but if CUDART hijack is enabled, we need to use it
  // since we can't call runtime functions outside of a task... :sigh:
#if defined(REALM_USE_CUDART_HIJACK)
  static const bool has_cuda_reductions = true;
#else
  static const bool has_cuda_reductions = false;
#endif
  template <bool EXCL>
  __device__ void apply_cuda(LHS &lhs, const RHS &rhs) const
  {
    apply<EXCL>(lhs, rhs);
  }
  template <bool EXCL>
  __device__ void fold_cuda(RHS &rhs1, const RHS &rhs2) const
  {
    fold<EXCL>(rhs1, rhs2);
  }
#endif

#if defined(REALM_USE_HIP) && defined(__HIPCC__)
  // HIP implementations that redirect to the __host__ __device__ implementations defined
  // earlier
  static const bool has_hip_reductions = true;

  template <bool EXCL>
  __device__ void apply_hip(LHS &lhs, const RHS &rhs) const
  {
    apply<EXCL>(lhs, rhs);
  }
  template <bool EXCL>
  __device__ void fold_hip(RHS &rhs1, const RHS &rhs2) const
  {
    fold<EXCL>(rhs1, rhs2);
  }
#endif
};

#endif
