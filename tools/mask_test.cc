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

#include <set>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

#include "../runtime/legion/bitmask.h"
#include "../runtime/legion/legion_allocation.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#else
#include <time.h>
#endif

using namespace Legion;

enum OpKind {
  EQ_OP,
  NEG_OP,
  OR_OP,
  AND_OP,
  XOR_OP,
  ORA_OP,
  ANDA_OP,
  XORA_OP,
  DIS_OP,
  DIFF_OP,
  DIFFA_OP,
  EMPTY_OP,
  SL_OP,
  SR_OP,
  SLA_OP,
  SRA_OP,
};

class BaseMask {
public:
  BaseMask(int max);
  BaseMask(const BaseMask &rhs);
  ~BaseMask(void);
public:
  inline BaseMask& operator=(const BaseMask &rhs);
public:
  inline void set_bit(unsigned bit);
public:
  inline bool operator==(const BaseMask &rhs) const;
public:
  inline BaseMask operator~(void) const;
  inline BaseMask operator|(const BaseMask &rhs) const;
  inline BaseMask operator&(const BaseMask &rhs) const;
  inline BaseMask operator^(const BaseMask &rhs) const;
public:
  inline BaseMask& operator|=(const BaseMask &rhs);
  inline BaseMask& operator&=(const BaseMask &rhs);
  inline BaseMask& operator^=(const BaseMask &rhs);
public:
  inline bool operator*(const BaseMask &rhs) const;
  inline BaseMask operator-(const BaseMask &rhs) const;
  inline BaseMask& operator-=(const BaseMask &rhs);
  inline bool operator!(void) const;
public:
  inline BaseMask operator<<(unsigned shift) const;
  inline BaseMask operator>>(unsigned shift) const;
public:
  inline BaseMask& operator<<=(unsigned shift);
  inline BaseMask& operator>>=(unsigned shift);
public:
  template<typename T>
  inline bool equals(const T &mask) const;
  inline void print(const char *name) const;
private:
  const int max;
  std::set<unsigned> values;
};

BaseMask::BaseMask(int m)
  : max(m)
{
}

BaseMask::BaseMask(const BaseMask &rhs)
  : max(rhs.max), values(rhs.values)
{
}

BaseMask::~BaseMask(void)
{
}

BaseMask& BaseMask::operator=(const BaseMask &rhs)
{
  assert(max == rhs.max);
  values = rhs.values;
  return *this;
}

void BaseMask::set_bit(unsigned bit)
{
  assert(bit < max);
  values.insert(bit);
}

bool BaseMask::operator==(const BaseMask &rhs) const
{
  if (max != rhs.max)
    return false;
  return (values == rhs.values);
}

BaseMask BaseMask::operator~(void) const
{
  BaseMask result(max);
  for (int i = 0; i < max; i++)
    if (values.find(i) == values.end())
      result.set_bit(i);
  return result;
}

BaseMask BaseMask::operator|(const BaseMask &rhs) const
{
  BaseMask result(max);
  result.values = values;
  result.values.insert(rhs.values.begin(), rhs.values.end());
  return result;
}

BaseMask BaseMask::operator&(const BaseMask &rhs) const
{
  BaseMask result(max);
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    if (rhs.values.find(*it) != rhs.values.end())
      result.set_bit(*it);
  }
  return result;
}

BaseMask BaseMask::operator^(const BaseMask &rhs) const
{
  BaseMask result(max);
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    if (rhs.values.find(*it) == rhs.values.end())
      result.set_bit(*it);
  }
  for (std::set<unsigned>::const_iterator it = rhs.values.begin();
        it != rhs.values.end(); it++)
  {
    if (values.find(*it) == values.end())
      result.set_bit(*it);
  }
  return result;
}

BaseMask& BaseMask::operator|=(const BaseMask &rhs)
{
  values.insert(rhs.values.begin(), rhs.values.end());
  return *this;
}

BaseMask& BaseMask::operator&=(const BaseMask &rhs)
{
  std::set<unsigned> next;
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    if (rhs.values.find(*it) != rhs.values.end())
      next.insert(*it);
  }
  values = next;
  return *this;
}

BaseMask& BaseMask::operator^=(const BaseMask &rhs)
{
  std::set<unsigned> next = rhs.values;
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    std::set<unsigned>::iterator finder = next.find(*it);
    if (finder != next.end())
      next.erase(finder);
    else
      next.insert(*it);
  }
  values = next;
  return *this;
}

bool BaseMask::operator*(const BaseMask &rhs) const
{
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    if (rhs.values.find(*it) != rhs.values.end())
      return false;
  }
  return true;
}

BaseMask BaseMask::operator-(const BaseMask &rhs) const
{
  BaseMask result(max);
  result.values = values;
  for (std::set<unsigned>::const_iterator it = rhs.values.begin();
        it != rhs.values.end(); it++)
  {
    result.values.erase(*it);
  }
  return result;
}

BaseMask& BaseMask::operator-=(const BaseMask &rhs)
{
  for (std::set<unsigned>::const_iterator it = rhs.values.begin();
        it != rhs.values.end(); it++)
  {
    values.erase(*it);
  }
  return *this;
}

bool BaseMask::operator!(void) const
{
  return values.empty();
}

BaseMask BaseMask::operator<<(unsigned shift) const
{
  BaseMask result(max);
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    unsigned next = *it + shift;
    if (next < max)
      result.set_bit(next);
  }
  return result;
}

BaseMask BaseMask::operator>>(unsigned shift) const
{
  BaseMask result(max);
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    if (*it >= shift)
      result.set_bit(*it - shift);
  }
  return result;
}

BaseMask& BaseMask::operator<<=(unsigned shift)
{
  std::set<unsigned> next;
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    unsigned next_bit = *it + shift;
    if (next_bit < max)
      next.insert(next_bit);
  }
  values = next;
  return *this;
}

BaseMask& BaseMask::operator>>=(unsigned shift)
{
  std::set<unsigned> next;
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    if (*it >= shift)
      next.insert(*it - shift);
  }
  values = next;
  return *this;
}

template<typename T>
bool BaseMask::equals(const T &mask) const
{
  if (T::pop_count(mask) != values.size())
    return false;
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    if (!mask.is_set(*it))
      return false;
  }
  return true;
}

void BaseMask::print(const char *name) const
{
  printf("    %s:", name);
  for (std::set<unsigned>::const_iterator it = values.begin();
        it != values.end(); it++)
  {
    printf(" %d", *it);
  }
  printf("\n");
}

template<typename BITMASK>
void print_mask(const char *name, BITMASK &rhs, const int max)
{
  printf("    %s:", name);
  for (int i = 0; i < max; i++)
    if (rhs.is_set(i))
      printf(" %d", i);
  printf("\n");
}

template<typename BITMASK, int MAX>
inline void initialize_random_mask(BITMASK &mask, BaseMask &base)
{
  const int num_set = lrand48() % MAX;
  for (int i = 0; i < num_set; i++)
  {
    int bit = lrand48() % MAX;
    mask.set_bit(bit);
    base.set_bit(bit);
  }
}

template<typename BITMASK, int MAX>
void test_equality(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing == for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left; 
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    bool expected = (left_base == right_base);
    bool actual = (left == right);
    if (expected != actual) {
      printf("FAILURE!\n"); 
      left_base.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_negation(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing ~ for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK mask;
    BaseMask base_mask(MAX);
    initialize_random_mask<BITMASK,MAX>(mask, base_mask);
    BITMASK result_mask = ~mask;
    BaseMask result_base = ~base_mask;
    if (!result_base.equals(result_mask)) {
      printf("FAILURE!\n");
      base_mask.print("base");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_or(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing | for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left; 
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BITMASK result = left | right;
    BaseMask result_base = left_base | right_base;
    if (!result_base.equals(result)) {
      printf("FAILURE!\n");
      left_base.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_and(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing & for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left; 
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BITMASK result = left & right;
    BaseMask result_base = left_base & right_base;
    if (!result_base.equals(result)) {
      printf("FAILURE!\n");
      left_base.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_xor(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing ^ for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left; 
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BITMASK result = left ^ right;
    BaseMask result_base = left_base ^ right_base;
    if (!result_base.equals(result)) {
      printf("FAILURE!\n");
      left_base.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_or_assign(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing |= for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left; 
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BaseMask copy = left_base;
    left |= right;
    left_base |= right_base;
    if (!left_base.equals(left)) {
      printf("FAILURE!\n");
      copy.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_and_assign(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing &= for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left; 
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BaseMask copy = left_base;
    left &= right;
    left_base &= right_base;
    if (!left_base.equals(left)) {
      printf("FAILURE!\n");
      copy.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_xor_assign(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing ^= for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left; 
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BaseMask copy = left_base;
    left ^= right;
    left_base ^= right_base;
    if (!left_base.equals(left)) {
      printf("FAILURE!\n");
      copy.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_disjoint(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing * for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left;
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    bool expected = left_base * right_base;
    bool actual = left * right;
    if (expected != actual) {
      printf("FAILURE!\n");
      left_base.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_diff(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing - for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left;
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BITMASK result = left - right;
    BaseMask result_base = left_base - right_base;
    if (!result_base.equals(result)) {
      printf("FAILURE!\n");
      left_base.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_diff_assign(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing -= for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK left;
    BITMASK right;
    BaseMask left_base(MAX), right_base(MAX);
    initialize_random_mask<BITMASK,MAX>(left, left_base);
    initialize_random_mask<BITMASK,MAX>(right, right_base);
    BaseMask copy = left_base;
    left -= right;
    left_base -= right_base;
    if (!left_base.equals(left)) {
      printf("FAILURE!\n");
      copy.print("left");
      right_base.print("right");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_empty(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing ! for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK mask;
    BaseMask base_mask(MAX);
    initialize_random_mask<BITMASK,MAX>(mask, base_mask);
    bool actual = !mask;
    bool expected = !base_mask;
    if (actual != expected) {
      printf("FAILURE!\n");
      base_mask.print("base");
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_shift_left(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing << for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK mask;
    BaseMask base_mask(MAX);
    initialize_random_mask<BITMASK,MAX>(mask, base_mask);
    const unsigned shift = lrand48() % MAX;
    BITMASK result_mask = mask << shift;
    BaseMask result_base = base_mask << shift;
    if (!result_base.equals(result_mask)) {
      printf("FAILURE!\n");
      printf("    Shift %d\n", shift);
      base_mask.print("base");
      result_base.print("result");
      print_mask("actual", result_mask, MAX);
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_shift_right(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing >> for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK mask;
    BaseMask base_mask(MAX);
    initialize_random_mask<BITMASK,MAX>(mask, base_mask);
    const unsigned shift = lrand48() % MAX;
    BITMASK result_mask = mask >> shift;
    BaseMask result_base = base_mask >> shift;
    if (!result_base.equals(result_mask)) {
      printf("FAILURE!\n");
      printf("    Shift %d\n", shift);
      base_mask.print("base");
      result_base.print("result");
      print_mask("actual", mask, MAX);
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_shift_left_assign(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing <<= for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK mask;
    BaseMask base_mask(MAX);
    initialize_random_mask<BITMASK,MAX>(mask, base_mask);
    const unsigned shift = lrand48() % MAX;
    mask <<= shift;
    BaseMask copy = base_mask;
    base_mask <<= shift;
    if (!base_mask.equals(mask)) {
      printf("FAILURE!\n");
      printf("    Shift %d\n", shift);
      copy.print("base");
      base_mask.print("result");
      print_mask("actual", mask, MAX);
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK, int MAX>
void test_shift_right_assign(const int num_iterations, const char *name)
{
  fprintf(stdout,"  Testing >>= for %s... ", name);
  fflush(stdout);
  for (int i = 0; i < num_iterations; i++)
  {
    BITMASK mask;
    BaseMask base_mask(MAX);
    initialize_random_mask<BITMASK,MAX>(mask, base_mask);
    const unsigned shift = lrand48() % MAX;
    mask >>= shift;
    BaseMask copy = base_mask;
    base_mask >>= shift;
    if (!base_mask.equals(mask)) {
      printf("FAILURE!\n");
      printf("    Shift %d\n", shift);
      copy.print("base");
      base_mask.print("result");
      print_mask("actual", mask, MAX);
      return;
    }
  }
  printf("SUCCESS!\n");
}

template<typename BITMASK>
void test_mask(const int num_iterations, const char *name)
{
  printf("Running tests for mask %s...\n", name);
  const int MAX = BITMASK::BIT_ELMTS * BITMASK::ELEMENT_SIZE;
  test_equality<BITMASK,MAX>(num_iterations, name);
  test_negation<BITMASK,MAX>(num_iterations, name);
  test_or<BITMASK,MAX>(num_iterations, name);
  test_and<BITMASK,MAX>(num_iterations, name);
  test_xor<BITMASK,MAX>(num_iterations, name);
  test_or_assign<BITMASK,MAX>(num_iterations, name);
  test_and_assign<BITMASK,MAX>(num_iterations, name);
  test_xor_assign<BITMASK,MAX>(num_iterations, name);
  test_disjoint<BITMASK,MAX>(num_iterations, name);
  test_diff<BITMASK,MAX>(num_iterations, name);
  test_diff_assign<BITMASK,MAX>(num_iterations, name);
  test_empty<BITMASK,MAX>(num_iterations, name);
  test_shift_left<BITMASK,MAX>(num_iterations, name);
  test_shift_right<BITMASK,MAX>(num_iterations, name);
  test_shift_left_assign<BITMASK,MAX>(num_iterations, name);
  test_shift_right_assign<BITMASK,MAX>(num_iterations, name);
}

template<int MAX, int SCALE, typename BITMASK>
void initialize_perf_masks(BITMASK *masks, const int num_masks)
{
  for (int idx = 0; idx < num_masks; idx++)
  {
    new (masks+idx) BITMASK();
    const int num_bits = lrand48() % (MAX/SCALE); 
    for (int i = 0; i < num_bits; i++)
      masks[idx].set_bit(lrand48() % MAX);
  }
}

template<typename BITMASK>
void delete_perf_masks(BITMASK *masks, const int num_masks)
{
  for (int idx = 0; idx < num_masks; idx++)
    masks[idx].~BITMASK();
}

template<int MAX>
void initialize_int_array(int *array, const int num_elements)
{
  for (int idx = 0; idx < num_elements; idx++)
    array[idx] = lrand48() % MAX;
}

inline unsigned long long current_time_in_nanoseconds(void)
{
#ifdef __MACH__
  mach_timespec_t ts;
  clock_serv_t cclock;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &ts);
  mach_port_deallocate(mach_task_self(), cclock);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
  long long t = (1000000000LL * ts.tv_sec) + ts.tv_nsec;
  return t;
}

template<int MAX, int SCALE, OpKind OP, typename BITMASK>
void test_mask_operation(const int num_iterations, const char *mask_name)
{
  unsigned long long start=0, stop=0;
  switch (OP)
  {
    case EQ_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
                alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 2*num_iterations);
        int counter = 0;
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          (masks[2*idx] == masks[2*idx+1]) ? counter++ : counter--;
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        break;
      }
    case NEG_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
                alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 2*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[2*idx+1] = ~masks[2*idx];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        break;
      }
    case OR_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
                alignof(BITMASK), false>(3*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 3*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[3*idx+2] = masks[3*idx] | masks[3*idx+1];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 3*num_iterations);
        free(masks);
        break;
      }
    case AND_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
                alignof(BITMASK), false>(3*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 3*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[3*idx+2] = masks[3*idx] & masks[3*idx+1];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 3*num_iterations);
        free(masks);
        break;
      }
    case XOR_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
                alignof(BITMASK), false>(3*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 3*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[3*idx+2] = masks[3*idx] ^ masks[3*idx+1];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 3*num_iterations);
        free(masks);
        break;
      }
    case ORA_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE>(masks, 2*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[2*idx+1] |= masks[2*idx];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        break;
      }
    case ANDA_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE>(masks, 2*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[2*idx+1] &= masks[2*idx];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        break;
      }
    case XORA_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE>(masks, 2*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[2*idx+1] ^= masks[2*idx];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        break;
      }
    case DIS_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 2*num_iterations);
        int counter = 0;
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          (masks[2*idx] * masks[2*idx+1]) ? counter++ : counter--;
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        break;
      }
    case DIFF_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(3*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 3*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[3*idx+2] = masks[3*idx] - masks[3*idx+1];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 3*num_iterations);
        free(masks);
        break;
      }
    case DIFFA_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 2*num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[2*idx+1] -= masks[2*idx];
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        break;
      }
    case EMPTY_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, num_iterations);
        int counter = 0;
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          (!masks[idx]) ? counter++ : counter--;
        stop = current_time_in_nanoseconds();
        delete_perf_masks<BITMASK>(masks, num_iterations);
        free(masks);
        break;
      }
    case SL_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 2*num_iterations);
        int *shift = (int*)malloc(num_iterations*sizeof(int));
        initialize_int_array<MAX>(shift, num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[2*idx+1] = masks[2*idx] << shift[idx];
        stop = current_time_in_nanoseconds(); 
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        free(shift);
        break;
      }
    case SR_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(2*num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, 2*num_iterations);
        int *shift = (int*)malloc(num_iterations*sizeof(int));
        initialize_int_array<MAX>(shift, num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[2*idx+1] = masks[2*idx] >> shift[idx];
        stop = current_time_in_nanoseconds(); 
        delete_perf_masks<BITMASK>(masks, 2*num_iterations);
        free(masks);
        free(shift);
        break;
      }
    case SLA_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, num_iterations);
        int *shift = (int*)malloc(num_iterations*sizeof(int));
        initialize_int_array<MAX>(shift, num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[idx] <<= shift[idx];
        stop = current_time_in_nanoseconds(); 
        delete_perf_masks<BITMASK>(masks, num_iterations);
        free(masks);
        free(shift);
        break;
      }
    case SRA_OP:
      {
        BITMASK *masks = (BITMASK*)Internal::legion_alloc_aligned<sizeof(BITMASK), 
            alignof(BITMASK), false>(num_iterations);
        initialize_perf_masks<MAX,SCALE,BITMASK>(masks, num_iterations);
        int *shift = (int*)malloc(num_iterations*sizeof(int));
        initialize_int_array<MAX>(shift, num_iterations);
        start = current_time_in_nanoseconds();
        for (int idx = 0; idx < num_iterations; idx++)
          masks[idx] >>= shift[idx];
        stop = current_time_in_nanoseconds(); 
        delete_perf_masks<BITMASK>(masks, num_iterations);
        free(masks);
        free(shift);
        break;
      }
    default:
      assert(false);
  }
  unsigned long long total = stop - start;
  unsigned long long avg = total / num_iterations; 
  printf("    Mask %s: %lld ns (total=%lld)\n", mask_name, avg, total);
}

template<OpKind OP>
void print_operation_prefix(void)
{
  switch (OP)
  {
    case EQ_OP:
      {
        printf("  Perf of == operator:\n");
        break;
      }
    case NEG_OP:
      {
        printf("  Perf of ~ operator:\n");
        break;
      }
    case OR_OP:
      {
        printf("  Perf of | operator:\n");
        break;
      }
    case AND_OP:
      {
        printf("  Perf of & operator:\n");
        break;
      }
    case XOR_OP:
      {
        printf("  Perf of ^ operator:\n");
        break;
      }
    case ORA_OP:
      {
        printf("  Perf of |= operator:\n");
        break;
      }
    case ANDA_OP:
      {
        printf("  Perf of &= operator:\n");
        break;
      }
    case XORA_OP:
      {
        printf("  Perf of ^= operator:\n");
        break;
      }
    case DIS_OP:
      {
        printf("  Perf of * operator:\n");
        break;
      }
    case DIFF_OP:
      {
        printf("  Perf of - operator:\n");
        break;
      }
    case DIFFA_OP:
      {
        printf("  Perf of -= operator:\n");
        break;
      }
    case EMPTY_OP:
      {
        printf("  Perf of ! operator:\n");
        break;
      }
    case SL_OP:
      {
        printf("  Perf of << operator:\n");
        break;
      }
    case SR_OP:
      {
        printf("  Perf of >> operator:\n");
        break;
      }
    case SLA_OP:
      {
        printf("  Perf of <<= operator:\n");
        break;
      }
    case SRA_OP:
      {
        printf("  Perf of >>= operator:\n");
        break;
      }
    default:
      assert(false);
  }
}

template<int SCALE, OpKind OP>
void test_operation_64(const int num_iterations)
{
  print_operation_prefix<OP>();  
  const int MAX = 64;
  test_mask_operation<MAX,SCALE,OP,
    BitMask<uint64_t,MAX,6,0x3F> >(num_iterations, "BitMask");
  test_mask_operation<MAX,SCALE,OP,
    TLBitMask<uint64_t,MAX,6,0x3F> >(num_iterations, "TLBitMask");

  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<6> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<8> >");

  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<6> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<8> >");
}

template<int SCALE, OpKind OP>
void test_operation_128(const int num_iterations)
{
  print_operation_prefix<OP>();  
  const int MAX = 128;
  test_mask_operation<MAX,SCALE,OP,
    BitMask<uint64_t,MAX,6,0x3F> >(num_iterations, "BitMask");
  test_mask_operation<MAX,SCALE,OP,
    TLBitMask<uint64_t,MAX,6,0x3F> >(num_iterations, "TLBitMask");
#ifdef __SSE2__
  test_mask_operation<MAX,SCALE,OP,SSEBitMask<MAX> >(num_iterations, "SSEBitMask");
  test_mask_operation<MAX,SCALE,OP,SSETLBitMask<MAX> >(num_iterations, "SSETLBitMask");
#endif
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<6> >");
    test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<8> >");

  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<6> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<8> >");

#ifdef __SSE2__
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<6> >");
    test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<8> >");

  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<6> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<8> >");
#endif
}

template<int MAX, int SCALE, OpKind OP>
void test_operation(const int num_iterations)
{
  print_operation_prefix<OP>();  
  test_mask_operation<MAX,SCALE,OP,
    BitMask<uint64_t,MAX,6,0x3F> >(num_iterations, "BitMask");
  test_mask_operation<MAX,SCALE,OP,
    TLBitMask<uint64_t,MAX,6,0x3F> >(num_iterations, "TLBitMask");
#ifdef __SSE2__
  test_mask_operation<MAX,SCALE,OP,SSEBitMask<MAX> >(num_iterations, "SSEBitMask");
  test_mask_operation<MAX,SCALE,OP,SSETLBitMask<MAX> >(num_iterations, "SSETLBitMask");
#endif
#ifdef __AVX__
  test_mask_operation<MAX,SCALE,OP,AVXBitMask<MAX> >(num_iterations, "AVXBitMask");
  test_mask_operation<MAX,SCALE,OP,AVXTLBitMask<MAX> >(num_iterations, "AVXTLBitMask");
#endif
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<6> >");
    test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<BitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<BitMask<8> >");

  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<6> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<TLBitMask<uint64_t,MAX,6,0x3F> > >(
        num_iterations, "CompoundBitMask<TLBitMask<8> >");

#ifdef __SSE2__
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<6> >");
    test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSEBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSEBitMask<8> >");

  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<6> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<SSETLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<SSETLBitMask<8> >");
#endif
#ifdef __AVX__
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXBitMask<6> >");
    test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXBitMask<8> >");

  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXTLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXTLBitMask<2> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXTLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXTLBitMask<4> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXTLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXTLBitMask<6> >");
  test_mask_operation<MAX,SCALE,OP,
    CompoundBitMask<AVXTLBitMask<MAX> > >(
        num_iterations, "CompoundBitMask<AVXTLBitMask<8> >");
#endif
}

template<int SCALE>
void test_perf_64(const int num_iterations)
{
  printf("Running perf for MAX=64,SCALE=%d...\n", SCALE);
  test_operation_64<SCALE,EQ_OP>(num_iterations);
  test_operation_64<SCALE,NEG_OP>(num_iterations);
  test_operation_64<SCALE,OR_OP>(num_iterations);
  test_operation_64<SCALE,AND_OP>(num_iterations);
  test_operation_64<SCALE,XOR_OP>(num_iterations);
  test_operation_64<SCALE,ORA_OP>(num_iterations);
  test_operation_64<SCALE,ANDA_OP>(num_iterations);
  test_operation_64<SCALE,XORA_OP>(num_iterations);
  test_operation_64<SCALE,DIS_OP>(num_iterations);
  test_operation_64<SCALE,DIFF_OP>(num_iterations);
  test_operation_64<SCALE,DIFFA_OP>(num_iterations);
  test_operation_64<SCALE,EMPTY_OP>(num_iterations);
  test_operation_64<SCALE,SL_OP>(num_iterations);
  test_operation_64<SCALE,SR_OP>(num_iterations);
  test_operation_64<SCALE,SLA_OP>(num_iterations);
  test_operation_64<SCALE,SRA_OP>(num_iterations);
}

template<int SCALE>
void test_perf_128(const int num_iterations)
{
  printf("Running perf for MAX=128,SCALE=%d...\n", SCALE);
  test_operation_128<SCALE,EQ_OP>(num_iterations);
  test_operation_128<SCALE,NEG_OP>(num_iterations);
  test_operation_128<SCALE,OR_OP>(num_iterations);
  test_operation_128<SCALE,AND_OP>(num_iterations);
  test_operation_128<SCALE,XOR_OP>(num_iterations);
  test_operation_128<SCALE,ORA_OP>(num_iterations);
  test_operation_128<SCALE,ANDA_OP>(num_iterations);
  test_operation_128<SCALE,XORA_OP>(num_iterations);
  test_operation_128<SCALE,DIS_OP>(num_iterations);
  test_operation_128<SCALE,DIFF_OP>(num_iterations);
  test_operation_128<SCALE,DIFFA_OP>(num_iterations);
  test_operation_128<SCALE,EMPTY_OP>(num_iterations);
  test_operation_128<SCALE,SL_OP>(num_iterations);
  test_operation_128<SCALE,SR_OP>(num_iterations);
  test_operation_128<SCALE,SLA_OP>(num_iterations);
  test_operation_128<SCALE,SRA_OP>(num_iterations);
}

template<int MAX, int SCALE>
void test_perf(const int num_iterations)
{
  printf("Running perf for MAX=%d,SCALE=%d...\n", MAX, SCALE);
  test_operation<MAX,SCALE,EQ_OP>(num_iterations);
  test_operation<MAX,SCALE,NEG_OP>(num_iterations);
  test_operation<MAX,SCALE,OR_OP>(num_iterations);
  test_operation<MAX,SCALE,AND_OP>(num_iterations);
  test_operation<MAX,SCALE,XOR_OP>(num_iterations);
  test_operation<MAX,SCALE,ORA_OP>(num_iterations);
  test_operation<MAX,SCALE,ANDA_OP>(num_iterations);
  test_operation<MAX,SCALE,XORA_OP>(num_iterations);
  test_operation<MAX,SCALE,DIS_OP>(num_iterations);
  test_operation<MAX,SCALE,DIFF_OP>(num_iterations);
  test_operation<MAX,SCALE,DIFFA_OP>(num_iterations);
  test_operation<MAX,SCALE,EMPTY_OP>(num_iterations);
  test_operation<MAX,SCALE,SL_OP>(num_iterations);
  test_operation<MAX,SCALE,SR_OP>(num_iterations);
  test_operation<MAX,SCALE,SLA_OP>(num_iterations);
  test_operation<MAX,SCALE,SRA_OP>(num_iterations);
}

int main(int argc, const char **argv)
{
  int num_iterations = 1024;
  if (argc > 1)
    num_iterations = atoi(argv[1]);
  printf("Iteration Count: %d\n", num_iterations);

  printf("\nBitMask Tests\n");
  test_mask<BitMask<uint64_t,64,6,0x3F> >(num_iterations,"BitMask<64>");
  test_mask<BitMask<uint64_t,128,6,0x3F> >(num_iterations,"BitMask<128>");
  test_mask<BitMask<uint64_t,192,6,0x3F> >(num_iterations,"BitMask<192>");
  test_mask<BitMask<uint64_t,256,6,0X3F> >(num_iterations,"BitMask<256>");
  test_mask<BitMask<uint64_t,384,6,0x3F> >(num_iterations,"BitMask<384>");
  test_mask<BitMask<uint64_t,512,6,0X3F> >(num_iterations,"BitMask<512>");
  test_mask<BitMask<uint64_t,768,6,0x3F> >(num_iterations,"BitMask<768>");
  test_mask<BitMask<uint64_t,1024,6,0x3F> >(num_iterations,"BitMask<1024>");
  test_mask<BitMask<uint64_t,1536,6,0x3F> >(num_iterations,"BitMask<1536>");
  test_mask<BitMask<uint64_t,2048,6,0x3F> >(num_iterations,"BitMask<2048>");

  printf("\nTLBitMask Tests\n");
  test_mask<TLBitMask<uint64_t,64,6,0x3F> >(num_iterations,"TLBitMask<64>");
  test_mask<TLBitMask<uint64_t,128,6,0x3F> >(num_iterations,"TLBitMask<128>");
  test_mask<TLBitMask<uint64_t,192,6,0x3F> >(num_iterations,"TLBitMask<192>");
  test_mask<TLBitMask<uint64_t,256,6,0X3F> >(num_iterations,"TLBitMask<256>");
  test_mask<TLBitMask<uint64_t,384,6,0x3F> >(num_iterations,"TLBitMask<384>");
  test_mask<TLBitMask<uint64_t,512,6,0X3F> >(num_iterations,"TLBitMask<512>");
  test_mask<TLBitMask<uint64_t,768,6,0x3F> >(num_iterations,"TLBitMask<768>");
  test_mask<TLBitMask<uint64_t,1024,6,0x3F> >(num_iterations,"TLBitMask<1024>");
  test_mask<TLBitMask<uint64_t,1536,6,0x3F> >(num_iterations,"TLBitMask<1536>");
  test_mask<TLBitMask<uint64_t,2048,6,0x3F> >(num_iterations,"TLBitMask<2048>");

#ifdef __SSE2__
  printf("\nSSEBitMask Tests\n");
  test_mask<SSEBitMask<128> >(num_iterations,"SSEBitMask<128>");
  test_mask<SSEBitMask<256> >(num_iterations,"SSEBitMask<256>");
  test_mask<SSEBitMask<384> >(num_iterations,"SSEBitMask<384>");
  test_mask<SSEBitMask<512> >(num_iterations,"SSEBitMask<512>");
  test_mask<SSEBitMask<768> >(num_iterations,"SSEBitMask<768>");
  test_mask<SSEBitMask<1024> >(num_iterations,"SSEBitMask<1024>");
  test_mask<SSEBitMask<1536> >(num_iterations,"SSEBitMask<1536>");
  test_mask<SSEBitMask<2048> >(num_iterations,"SSEBitMask<2048>");

  printf("\nSSETLBitMask Tests\n");
  test_mask<SSETLBitMask<128> >(num_iterations,"SSETLBitMask<128>");
  test_mask<SSETLBitMask<256> >(num_iterations,"SSETLBitMask<256>");
  test_mask<SSETLBitMask<384> >(num_iterations,"SSETLBitMask<384>");
  test_mask<SSETLBitMask<512> >(num_iterations,"SSETLBitMask<512>");
  test_mask<SSETLBitMask<768> >(num_iterations,"SSETLBitMask<768>");
  test_mask<SSETLBitMask<1024> >(num_iterations,"SSETLBitMask<1024>");
  test_mask<SSETLBitMask<1536> >(num_iterations,"SSETLBitMask<1536>");
  test_mask<SSETLBitMask<2048> >(num_iterations,"SSETLBitMask<2048>");
#endif

#ifdef __AVX__
  printf("\nAVXBitMask Tests\n");
  test_mask<AVXBitMask<256> >(num_iterations,"AVXBitMask<256>");
  test_mask<AVXBitMask<512> >(num_iterations,"AVXBitMask<512>");
  test_mask<AVXBitMask<768> >(num_iterations,"AVXBitMask<768>");
  test_mask<AVXBitMask<1024> >(num_iterations,"AVXBitMask<1024>");
  test_mask<AVXBitMask<1536> >(num_iterations,"AVXBitMask<1536>");
  test_mask<AVXBitMask<2048> >(num_iterations,"AVXBitMask<2048>");

  printf("\nAVXTLBitMask Tests\n");
  test_mask<AVXTLBitMask<256> >(num_iterations,"AVXTLBitMask<256>");
  test_mask<AVXTLBitMask<512> >(num_iterations,"AVXTLBitMask<512>");
  test_mask<AVXTLBitMask<768> >(num_iterations,"AVXTLBitMask<768>");
  test_mask<AVXTLBitMask<1024> >(num_iterations,"AVXTLBitMask<1024>");
  test_mask<AVXTLBitMask<1536> >(num_iterations,"AVXTLBitMask<1536>");
  test_mask<AVXTLBitMask<2048> >(num_iterations,"AVXTLBitMask<2048>");
#endif

  printf("\nCompoundBitMask Tests\n");
  test_mask<CompoundBitMask<BitMask<uint64_t,64,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<64,2>");
  test_mask<CompoundBitMask<BitMask<uint64_t,64,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<64,3>");
  test_mask<CompoundBitMask<BitMask<uint64_t,64,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<64,4>");
  test_mask<CompoundBitMask<BitMask<uint64_t,64,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<64,5>");
  test_mask<CompoundBitMask<BitMask<uint64_t,64,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<64,6>");
  test_mask<CompoundBitMask<BitMask<uint64_t,64,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<64,7>");
  test_mask<CompoundBitMask<BitMask<uint64_t,64,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<64,8>");

  test_mask<CompoundBitMask<BitMask<uint64_t,128,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<128,2>");
  test_mask<CompoundBitMask<BitMask<uint64_t,128,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<128,3>");
  test_mask<CompoundBitMask<BitMask<uint64_t,128,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<128,4>");
  test_mask<CompoundBitMask<BitMask<uint64_t,128,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<128,5>");
  test_mask<CompoundBitMask<BitMask<uint64_t,128,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<128,6>");
  test_mask<CompoundBitMask<BitMask<uint64_t,128,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<128,7>");
  test_mask<CompoundBitMask<BitMask<uint64_t,128,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<128,8>");

  test_mask<CompoundBitMask<BitMask<uint64_t,192,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<192,2>");
  test_mask<CompoundBitMask<BitMask<uint64_t,192,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<192,3>");
  test_mask<CompoundBitMask<BitMask<uint64_t,192,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<192,4>");
  test_mask<CompoundBitMask<BitMask<uint64_t,192,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<192,5>");
  test_mask<CompoundBitMask<BitMask<uint64_t,192,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<192,6>");
  test_mask<CompoundBitMask<BitMask<uint64_t,192,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<192,7>");
  test_mask<CompoundBitMask<BitMask<uint64_t,192,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<192,8>");

  test_mask<CompoundBitMask<BitMask<uint64_t,256,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<256,2>");
  test_mask<CompoundBitMask<BitMask<uint64_t,256,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<256,3>");
  test_mask<CompoundBitMask<BitMask<uint64_t,256,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<256,4>");
  test_mask<CompoundBitMask<BitMask<uint64_t,256,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<256,5>");
  test_mask<CompoundBitMask<BitMask<uint64_t,256,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<256,6>");
  test_mask<CompoundBitMask<BitMask<uint64_t,256,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<256,7>");
  test_mask<CompoundBitMask<BitMask<uint64_t,256,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<256,8>");

  test_mask<CompoundBitMask<BitMask<uint64_t,512,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<512,2>");
  test_mask<CompoundBitMask<BitMask<uint64_t,512,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<512,3>");
  test_mask<CompoundBitMask<BitMask<uint64_t,512,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<512,4>");
  test_mask<CompoundBitMask<BitMask<uint64_t,512,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<512,5>");
  test_mask<CompoundBitMask<BitMask<uint64_t,512,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<512,6>");
  test_mask<CompoundBitMask<BitMask<uint64_t,512,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<512,7>");
  test_mask<CompoundBitMask<BitMask<uint64_t,512,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<512,8>");

  test_mask<CompoundBitMask<BitMask<uint64_t,1024,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<1024,2>");
  test_mask<CompoundBitMask<BitMask<uint64_t,1024,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<1024,3>");
  test_mask<CompoundBitMask<BitMask<uint64_t,1024,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<1024,4>");
  test_mask<CompoundBitMask<BitMask<uint64_t,1024,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<1024,5>");
  test_mask<CompoundBitMask<BitMask<uint64_t,1024,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<1024,6>");
  test_mask<CompoundBitMask<BitMask<uint64_t,1024,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<1024,7>");
  test_mask<CompoundBitMask<BitMask<uint64_t,1024,6,0x3F> > >(
                              num_iterations,"CompoundBitMask<1024,8>");

#if 0
  test_perf_64<8>(num_iterations);
  test_perf_64<4>(num_iterations);
  test_perf_64<2>(num_iterations);
#endif
  test_perf_64<1>(num_iterations);

#if 0
  test_perf_128<8>(num_iterations);
  test_perf_128<4>(num_iterations);
  test_perf_128<2>(num_iterations);
#endif
  test_perf_128<1>(num_iterations);

#if 0
  test_perf<256,8>(num_iterations);
  test_perf<256,4>(num_iterations);
  test_perf<256,2>(num_iterations);
#endif
  test_perf<256,1>(num_iterations);

#if 0
  test_perf<512,8>(num_iterations);
  test_perf<512,4>(num_iterations);
  test_perf<512,2>(num_iterations);
#endif
  test_perf<512,1>(num_iterations);

#if 0
  test_perf<1024,8>(num_iterations);
  test_perf<1024,4>(num_iterations);
  test_perf<1024,2>(num_iterations);
#endif
  test_perf<1024,1>(num_iterations);

#if 0
  test_perf<2048,8>(num_iterations);
  test_perf<2048,4>(num_iterations);
  test_perf<2048,2>(num_iterations);
#endif
  test_perf<2048,1>(num_iterations);

  return 0;
}
