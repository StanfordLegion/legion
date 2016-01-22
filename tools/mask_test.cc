/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "legion_utilities.h"

using namespace LegionRuntime::HighLevel;

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
  if (T::pop_count(mask) != (int)values.size())
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
  const int MAX = BITMASK::ELEMENTS * BITMASK::ELEMENT_SIZE;
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

  return 0;
}
