/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#ifndef __MAX__
#define __MAX__(x,y) (((x) > (y)) ? (x) : (y))
#endif

#ifndef __MIN__
#define __MIN__(x,y) (((x) < (y)) ? (x) : (y))
#endif

namespace Legion {

  template<> __CUDA_HD__ inline
  void SumReduction<bool>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs = lhs || rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<bool>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] || rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&lhs;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool || rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<bool>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 = rhs1 || rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<bool>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] || rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&rhs1;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool || rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F += rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F += rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x += rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y += rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&lhs, rhs);
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&rhs1, rhs2);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64bit int atomic yet
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed += rhs;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64bit int atomic yet
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed += rhs2;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F += rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F += rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline 
  void SumReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x += rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y += rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&lhs, rhs); 
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&rhs1, rhs2);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64 bit int atomic yet
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval += rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    __sync_fetch_and_add(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64 bit int atomic yet
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval += rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void SumReduction<__half>::apply<true>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    lhs = lhs + rhs;
#else
    lhs = __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&lhs)) + 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<__half>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) + rhs);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&lhs;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) + 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<__half>::fold<true>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    rhs1 = rhs1 + rhs2;
#else
    rhs1= __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs1)) + 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<__half>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) + rhs2);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&rhs1;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) + 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void SumReduction<float>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<float>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&lhs, rhs);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<float>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<float>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&rhs1, rhs2);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<double>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<double>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 600
    atomicAdd(&lhs, rhs);
#else
    unsigned long long *target = (unsigned long long *)&lhs;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float += rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#endif
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<double>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<double>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 600
    atomicAdd(&rhs1, rhs2);
#else
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float += rhs2; 
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#endif
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void SumReduction<complex<__half> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<__half> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int newval, oldval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) + rhs);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) + rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<__half> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<__half> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) + rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) + rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void SumReduction<complex<float> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<float> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) + rhs);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) + rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<float> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<float> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) + rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) + rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<double> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<double> >::apply<false>(LHS &lhs, RHS rhs)
  {
    double *l = reinterpret_cast<double*>(&lhs);
    double *r = reinterpret_cast<double*>(&rhs);
#ifdef __CUDA_ARCH__
    for (unsigned i = 0; i < 2; ++i) {
#if __CUDA_ARCH__ >= 600
      atomicAdd(&l[i], r[i]);
#else
      unsigned long long *target = (unsigned long long *)&l[i];
      union { unsigned long long as_int; double as_float; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_float += r[i];
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
#endif
    }
#else
    for (unsigned i = 0; i < 2; ++i) {
      volatile long long *target = (volatile long long*)&l[i];
      union { long long as_int; double as_float; } oldval, newval;
      do {
        oldval.as_int = *target;
        newval.as_float = oldval.as_float + r[i];
      } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
    }
#endif
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<double> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void SumReduction<complex<double> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
    double *r1 = reinterpret_cast<double*>(&rhs1);
    double *r2 = reinterpret_cast<double*>(&rhs2);
#ifdef __CUDA_ARCH__
    for (unsigned i = 0; i < 2; ++i) {
#if __CUDA_ARCH__ >= 600
      atomicAdd(&r1[i], r2[i]);
#else
      unsigned long long *target = (unsigned long long *)&r1[i];
      union { unsigned long long as_int; double as_float; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_float += r2[i];
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
#endif
    }
#else
    for (unsigned i = 0; i < 2; ++i) {
      volatile long long *target = (volatile long long*)&r1[i];
      union { long long as_int; double as_float; } oldval, newval;
      do {
        oldval.as_int = *target;
        newval.as_float = oldval.as_float + r2[i];
      } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
    }
#endif
  }
#endif // LEGION_REDOP_COMPLEX

  template<> __CUDA_HD__ inline
  void DiffReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F -= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F += rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x -= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y -= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    atomicSub(&lhs, rhs);
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&rhs1, rhs2);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64bit int atomic yet
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed -= rhs;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64bit int atomic yet
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed += rhs2;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F -= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F += rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline 
  void DiffReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x -= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y -= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y += rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    atomicSub(&lhs, rhs); 
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&rhs1, rhs2);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64 bit int atomic yet
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval -= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    __sync_fetch_and_sub(&lhs, rhs);
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // Apparently there is no signed 64 bit int atomic yet
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval += rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    __sync_fetch_and_add(&rhs1, rhs2);
#endif
  }

#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void DiffReduction<__half>::apply<true>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    lhs = lhs - rhs;
#else
    lhs = __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&lhs)) - 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<__half>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) - rhs);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&lhs;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) - 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<__half>::fold<true>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    rhs1 = rhs1 + rhs2;
#else
    rhs1= __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs1)) + 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<__half>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) + rhs2);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&rhs1;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) + 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void DiffReduction<float>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<float>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int*)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float -= rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float - rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<float>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<float>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    atomicAdd(&rhs1, rhs2);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<double>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<double>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float -= rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float - rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<double>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<double>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 600
    atomicAdd(&rhs1, rhs2);
#else
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float += rhs2; 
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#endif
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void DiffReduction<complex<__half> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<complex<__half> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) - rhs);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) - rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<complex<__half> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<complex<__half> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) - rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) - rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void DiffReduction<complex<float> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs -= rhs;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<complex<float> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) - rhs);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) - rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<complex<float> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<> __CUDA_HD__ inline
  void DiffReduction<complex<float> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) - rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) - rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_COMPLEX

  template<> __CUDA_HD__ inline
  void ProdReduction<bool>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs = lhs && rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<bool>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] && rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&lhs;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool && rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<bool>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 = rhs1 && rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<bool>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] && rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&rhs1;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool && rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F *= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&lhs;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F *= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&rhs1;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x *= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y *= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&lhs;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&rhs1;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed *= rhs;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed *= rhs2;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F *= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&lhs;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F *= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&rhs1;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline 
  void ProdReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x *= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y *= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&lhs;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&rhs1;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&lhs;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&lhs;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&rhs1;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&rhs1;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void ProdReduction<__half>::apply<true>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    lhs = lhs * rhs;
#else
    lhs = __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&lhs)) * 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<__half>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) * rhs);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&lhs;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) * 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<__half>::fold<true>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    rhs1 = rhs1 * rhs2;
#else
    rhs1= __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs1)) * 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<__half>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) * rhs2);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&rhs1;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) * 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void ProdReduction<float>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<float>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float *= rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<float>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<float>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float *= rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<double>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<double>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float *= rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<double>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<double>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float *= rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void ProdReduction<complex<__half> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<complex<__half> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) * rhs);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) * rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<complex<__half> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<complex<__half> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) * rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) * rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void ProdReduction<complex<float> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs *= rhs;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<complex<float> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) * rhs);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) * rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<complex<float> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void ProdReduction<complex<float> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) * rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) * rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_COMPLEX

  template<> __CUDA_HD__ inline
  void DivReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F /= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&lhs;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F *= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&rhs1;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x /= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y /= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&lhs;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&rhs1;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval /= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed /= rhs;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed *= rhs2;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F /= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&lhs;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F *= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&rhs1;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline 
  void DivReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x /= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y /= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&lhs;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y *= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&rhs1;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&lhs;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval /= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&lhs;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&rhs1;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&rhs1;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval /= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval *= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void DivReduction<__half>::apply<true>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    lhs = lhs / rhs;
#else
    lhs = __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&lhs)) / 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<__half>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) / rhs);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&lhs;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) / 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<__half>::fold<true>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    rhs1 = rhs1 * rhs2;
#else
    rhs1= __convert_float_to_half(
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs1)) * 
        __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<__half>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(
          __short_as_half(newval.as_short[offset]) * rhs2);
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&rhs1;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(
          __convert_halfint_to_float(oldval.as_short) * 
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2)));
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void DivReduction<float>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<float>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float /= rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<float>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<float>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float *= rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<double>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<double>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float /= rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float / rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<double>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<double>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float *= rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void DivReduction<complex<__half> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<complex<__half> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) / rhs);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) / rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<complex<__half> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<complex<__half> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(newval) / rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(convert_complex<__half>::from_int(oldval) / rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void DivReduction<complex<float> >::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs /= rhs;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<complex<float> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) / rhs);
      newval  = atomicCAS(target, oldval , newval );
    } while (oldval  != newval );
#else
    volatile unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) / rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void DivReduction<complex<float> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 *= rhs2;
  }

  template<> __CUDA_HD__ inline
  void DivReduction<complex<float> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(convert_complex<float>::from_int(newval) / rhs2);
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(convert_complex<float>::from_int(oldval) / rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_COMPLEX

  template<> __CUDA_HD__ inline
  void MaxReduction<bool>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (!lhs && rhs)
      lhs = true;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<bool>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] || rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&lhs;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool || rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<bool>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (!rhs1 && rhs2)
      rhs1 = true;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<bool>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    union { int as_int; bool as_bool; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool = oldval.as_bool || rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool != newval.as_bool);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&rhs1;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool || rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MAX__(oldval.as_char.F, rhs);               \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&lhs;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MAX__(oldval.as_char.F, rhs2);              \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \


    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&rhs1;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MAX__(oldval.as_short.x, rhs);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MAX__(oldval.as_short.y, rhs);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&lhs;
    short oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MAX__(oldval.as_short.x, rhs2);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MAX__(oldval.as_short.y, rhs2);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&rhs1;
    short oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed = __MAX__(oldval.as_signed, rhs);
      if (newval.as_signed == oldval.as_signed)
        break;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed = __MAX__(oldval.as_signed, rhs2);
      if (newval.as_signed == oldval.as_signed)
        break;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MAX__(oldval.as_char.F, rhs);               \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&lhs;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MAX__(oldval.as_char.F, rhs2);              \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \


    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&rhs1;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline 
  void MaxReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MAX__(oldval.as_short.x, rhs);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MAX__(oldval.as_short.y, rhs);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&lhs;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MAX__(oldval.as_short.x, rhs2);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MAX__(oldval.as_short.y, rhs2);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&rhs1;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&lhs;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&lhs;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&rhs1;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&rhs1;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = __MAX__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void MaxReduction<__half>::apply<true>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    if (rhs > lhs)
      lhs = rhs;
#else
    if ((*reinterpret_cast<uint16_t*>(&rhs)) > (*reinterpret_cast<uint16_t*>(&lhs)))
      lhs = rhs;
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<__half>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(__MAX__(
          __short_as_half(newval.as_short[offset]), rhs));
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&lhs;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(__MAX__(
          __convert_halfint_to_float(oldval.as_short),
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs))));
      if (newval.as_short == oldval.as_short)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<__half>::fold<true>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    if (rhs2 > rhs1)
      rhs1 = rhs2;
#else
    if ((*reinterpret_cast<uint16_t*>(&rhs2)) > (*reinterpret_cast<uint16_t*>(&rhs1)))
      rhs1 = rhs2;
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<__half>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(__MAX__(
          __short_as_half(newval.as_short[offset]), rhs2));
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&rhs1;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(__MAX__(
          __convert_halfint_to_float(oldval.as_short),
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2))));
      if (newval.as_short == oldval.as_short)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void MaxReduction<float>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<float>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float = __MAX__(oldval.as_float, rhs);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MAX__(oldval.as_float, rhs);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<float>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<float>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float = __MAX__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MAX__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<double>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<double>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float = __MAX__(oldval.as_float, rhs);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MAX__(oldval.as_float, rhs);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<double>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<double>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int; 
      newval.as_float = __MAX__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MAX__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void MaxReduction<complex<__half> >::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<complex<__half> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(__MAX__(convert_complex<__half>::from_int(newval), rhs));
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(__MAX__(convert_complex<__half>::from_int(oldval), rhs));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<complex<__half> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<complex<__half> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(__MAX__(convert_complex<__half>::from_int(newval), rhs2));
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(__MAX__(convert_complex<__half>::from_int(oldval), rhs2));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void MaxReduction<complex<float> >::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs > lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<complex<float> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(__MAX__(convert_complex<float>::from_int(newval), rhs));
      newval  = atomicCAS(target, oldval , newval );
    } while (oldval  != newval );
#else
    volatile unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(__MAX__(convert_complex<float>::from_int(oldval), rhs));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<complex<float> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 > rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MaxReduction<complex<float> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long  oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<float>::as_int(__MAX__(convert_complex<float>::from_int(newval), rhs2));
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(__MAX__(convert_complex<float>::from_int(oldval), rhs2));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_COMPLEX

  template<> __CUDA_HD__ inline
  void MinReduction<bool>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (lhs && !rhs)
      lhs = false;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<bool>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] && rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&lhs;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool && rhs;
      if (newval.as_bool == oldval.as_bool)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<bool>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs1 && !rhs2)
      rhs1 = false;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<bool>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] && rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&rhs1;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool && rhs2;
      if (newval.as_bool == oldval.as_bool)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MIN__(oldval.as_char.F, rhs);               \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&lhs;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MIN__(oldval.as_char.F, rhs2);              \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \


    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&rhs1;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MIN__(oldval.as_short.x, rhs);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MIN__(oldval.as_short.y, rhs);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&lhs;
    short oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MIN__(oldval.as_short.x, rhs2);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MIN__(oldval.as_short.y, rhs2);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&rhs1;
    short oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed = __MIN__(oldval.as_signed, rhs);
      if (newval.as_signed == oldval.as_signed)
        break;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed = __MIN__(oldval.as_signed, rhs2);
      if (newval.as_signed == oldval.as_signed)
        break;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs2);
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MIN__(oldval.as_char.F, rhs);               \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&lhs;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F = __MIN__(oldval.as_char.F, rhs2);              \
          if (newval.as_char.F == oldval.as_char.F)                        \
            break;                                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \


    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&rhs1;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval * rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline 
  void MinReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MIN__(oldval.as_short.x, rhs);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MIN__(oldval.as_short.y, rhs);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&lhs;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x = __MIN__(oldval.as_short.x, rhs2);
        if (newval.as_short.x == oldval.as_short.x)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y = __MIN__(oldval.as_short.y, rhs2);
        if (newval.as_short.y == oldval.as_short.y)
          break;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&rhs1;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&lhs;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&lhs;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&rhs1;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&rhs1;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = __MIN__(oldval, rhs2);
      if (newval == oldval)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void MinReduction<__half>::apply<true>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    if (rhs < lhs)
      lhs = rhs;
#else
    if ((*reinterpret_cast<uint16_t*>(&rhs)) < (*reinterpret_cast<uint16_t*>(&lhs)))
      lhs = rhs;
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<__half>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(__MIN__(
          __short_as_half(newval.as_short[offset]), rhs));
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&lhs;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(__MIN__(
          __convert_halfint_to_float(oldval.as_short),
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs))));
      if (newval.as_short == oldval.as_short)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<__half>::fold<true>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    if (rhs2 < rhs1)
      rhs1 = rhs2;
#else
    if ((*reinterpret_cast<uint16_t*>(&rhs2)) < (*reinterpret_cast<uint16_t*>(&rhs1)))
      rhs1 = rhs2;
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<__half>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = (((unsigned long long)ptr) % 4) / sizeof(__half);
    union { int as_int; short as_short[2]; } oldval, newval;
    int *target = (int *)(ptr - (offset * sizeof(__half)));
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_short[offset] = __half_as_short(__MIN__(
          __short_as_half(newval.as_short[offset]), rhs2));
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int*)&rhs1;
    union { int as_int; uint16_t as_short; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_short = __convert_float_to_halfint(__MIN__(
          __convert_halfint_to_float(oldval.as_short),
          __convert_halfint_to_float(*reinterpret_cast<uint16_t*>(&rhs2))));
      if (newval.as_short == oldval.as_short)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void MinReduction<float>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<float>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float = __MIN__(oldval.as_float, rhs);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&lhs;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MIN__(oldval.as_float, rhs);
      if (oldval.as_float == newval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<float>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<float>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float = __MIN__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile int *target = (volatile int *)&rhs1;
    union { int as_int; float as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MIN__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<double>::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<double>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float = __MIN__(oldval.as_float, rhs);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MIN__(oldval.as_float, rhs);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<double>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<double>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { unsigned long long as_int; double as_float; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_float = __MIN__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_int != newval.as_int);
#else
    // No atomic floating point operations so use compare and swap 
    volatile long long *target = (volatile long long *)&rhs1;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = __MIN__(oldval.as_float, rhs2);
      if (newval.as_float == oldval.as_float)
        break;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<> __CUDA_HD__ inline
  void MinReduction<complex<__half> >::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<complex<__half> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(__MIN__(convert_complex<__half>::from_int(newval), rhs));
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(__MIN__(convert_complex<__half>::from_int(oldval), rhs));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<complex<__half> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<complex<__half> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<__half>::as_int(__MIN__(convert_complex<__half>::from_int(newval), rhs2));
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int *)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<__half>::as_int(__MIN__(convert_complex<__half>::from_int(oldval), rhs2));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_HALF

  template<> __CUDA_HD__ inline
  void MinReduction<complex<float> >::apply<true>(LHS &lhs, RHS rhs)
  {
    if (rhs < lhs)
      lhs = rhs;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<complex<float> >::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval= convert_complex<float>::as_int(__MIN__(convert_complex<float>::from_int(newval), rhs));
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(__MIN__(convert_complex<float>::from_int(oldval), rhs));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void MinReduction<complex<float> >::fold<true>(RHS &rhs1, RHS rhs2)
  {
    if (rhs2 < rhs1)
      rhs1 = rhs2;
  }

  template<> __CUDA_HD__ inline
  void MinReduction<complex<float> >::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval = convert_complex<float>::as_int(__MIN__(convert_complex<float>::from_int(newval), rhs2));
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (unsigned long long*)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = convert_complex<float>::as_int(__MIN__(convert_complex<float>::from_int(oldval), rhs2));
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }
#endif // LEGION_REDOP_COMPLEX

  template<> __CUDA_HD__ inline
  void OrReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F |= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&lhs;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F |= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&rhs1;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x |= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y |= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&lhs;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x |= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y |= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&rhs1;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval |= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval |= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed |= rhs;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed |= rhs2;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F |= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&lhs;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F |= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&rhs1;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline 
  void OrReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x |= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y |= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&lhs;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x |= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y |= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&rhs1;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&lhs;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval |= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&lhs;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&rhs1;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval |= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&rhs1;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs |= rhs;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval |= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 |= rhs2;
  }

  template<> __CUDA_HD__ inline
  void OrReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval |= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval | rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F &= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&lhs;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F &= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&rhs1;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x &= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y &= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&lhs;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x &= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y &= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&rhs1;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval &= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval &= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed &= rhs;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed &= rhs2;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F &= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&lhs;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F &= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&rhs1;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline 
  void AndReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x &= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y &= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&lhs;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x &= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y &= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&rhs1;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&lhs;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval &= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&lhs;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&rhs1;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval &= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&rhs1;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs &= rhs;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval &= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 &= rhs2;
  }

  template<> __CUDA_HD__ inline
  void AndReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval &= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval & rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<bool>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs = lhs ^ rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<bool>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] ^ rhs;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&lhs;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<bool>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 = rhs1 ^ rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<bool>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    const unsigned offset = ((unsigned long long)ptr) % 4;
    int *target = (int *)(ptr - offset);
    union { int as_int; bool as_bool[4]; } oldval, newval;
    newval.as_int = *target;
    do {
      oldval.as_int = newval.as_int;
      newval.as_bool[offset] = newval.as_bool[offset] ^ rhs2;
      newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
    } while (oldval.as_bool[offset] != newval.as_bool[offset]);
#else
    // No atomic logical operations so use compare and swap
    volatile int8_t *target = (volatile int8_t *)&rhs1;
    union { int8_t as_int; bool as_bool; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_bool = oldval.as_bool ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F ^= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&lhs;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(int8_t)*IDX);                     \
        union { int as_int; char4 as_char; } oldval, newval;               \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F ^= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile int8_t *target = (volatile int8_t*)&rhs1;
    int8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x ^= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y ^= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&lhs;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x ^= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(short));
      union { int as_int; short2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y ^= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile short *target = (volatile short*)&rhs1;
    short oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&lhs;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval ^= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&lhs;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    int *target = (int *)&rhs1;
    int oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval ^= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile int *target = (int*)&rhs1;
    int oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed ^= rhs;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&lhs;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<int64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    union { long long as_signed; unsigned long long as_unsigned; } oldval, newval;
    newval.as_unsigned = *target;
    do {
      oldval.as_signed = newval.as_signed;
      newval.as_signed ^= rhs2;
      newval.as_unsigned = atomicCAS(target, oldval.as_unsigned, newval.as_unsigned);
    } while (oldval.as_signed != newval.as_signed);
#else
    volatile long long *target = (volatile long long *)&rhs1;
    long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint8_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint8_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F ^= rhs;                                         \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&lhs;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint8_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint8_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
#define CASE(F, IDX)                                                       \
      case IDX : {                                                         \
        int *target = (int *)(ptr-sizeof(uint8_t)*IDX);                    \
        union { int as_int; uchar4 as_char; } oldval, newval;              \
        newval.as_int = *target;                                           \
        do {                                                               \
          oldval.as_int = newval.as_int;                                   \
          newval.as_char.F ^= rhs2;                                        \
          newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int); \
        } while (oldval.as_int != newval.as_int);                          \
        break;                                                             \
      }                                                                    \

    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    switch (((unsigned long long)ptr) % 4) {
      CASE(x, 0)
      CASE(y, 1)
      CASE(z, 2)
      CASE(w, 3)
      default :{ assert(false); break; }
    }
#undef CASE
#else
    volatile uint8_t *target = (volatile uint8_t*)&rhs1;
    uint8_t oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint16_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline 
  void XorReduction<uint16_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&lhs;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x ^= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y ^= rhs;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&lhs;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint16_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint16_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    // GPU atomics need 4 byte alignment
    char *ptr = (char*)&rhs1;
    if ((((unsigned long long)ptr) % 4) == 0) {
      // Aligned case
      int *target = (int *)ptr;
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.x ^= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    } else {
      // Unaligned case
      int *target = (int *)(ptr-sizeof(unsigned short));
      union { int as_int; ushort2 as_short; } oldval, newval;
      newval.as_int = *target;
      do {
        oldval.as_int = newval.as_int;
        newval.as_short.y ^= rhs2;
        newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
      } while (oldval.as_int != newval.as_int);
    }
#else
    volatile unsigned short *target = (volatile unsigned short*)&rhs1;
    unsigned short oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint32_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint32_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&lhs;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval ^= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&lhs;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint32_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint32_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned *target = (unsigned *)&rhs1;
    unsigned oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval ^= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned *target = (volatile unsigned *)&rhs1;
    unsigned oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint64_t>::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs ^= rhs;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint64_t>::apply<false>(LHS &lhs, RHS rhs)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval ^= rhs;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&lhs;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint64_t>::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 ^= rhs2;
  }

  template<> __CUDA_HD__ inline
  void XorReduction<uint64_t>::fold<false>(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    unsigned long long *target = (unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    newval = *target;
    do {
      oldval = newval;
      newval ^= rhs2;
      newval = atomicCAS(target, oldval, newval);
    } while (oldval != newval);
#else
    volatile unsigned long long *target = (volatile unsigned long long *)&rhs1;
    unsigned long long oldval, newval;
    do {
      oldval = *target;
      newval = oldval ^ rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval, newval));
#endif
  }

}; // namespace Legion

#undef __MAX__
#undef __MIN__

