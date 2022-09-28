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

#include <assert.h>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <string.h>

using namespace std;
typedef std::default_random_engine RNG;

#if __cplusplus < 201103L
#error This test requires C++11 or better.
#endif

#include "legion/legion_redop.h"

using namespace Legion;

// Define all the 'functors' that will be used to test the implementation
template <typename Redop> struct RedopTest;

template <typename T> struct RedopTest<SumReduction<T> > {
  typedef typename SumReduction<T>::LHS LHS;
  typedef typename SumReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return x + y; }
};
template <typename T> struct RedopTest<DiffReduction<T> > {
  typedef typename DiffReduction<T>::LHS LHS;
  typedef typename DiffReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return x - y; }
};
template <typename T> struct RedopTest<ProdReduction<T> > {
  typedef typename ProdReduction<T>::LHS LHS;
  typedef typename ProdReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return x * y; }
};
template <typename T> struct RedopTest<DivReduction<T> > {
  typedef typename DivReduction<T>::LHS LHS;
  typedef typename DivReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return x / y; }
};

template <typename T> struct RedopTest<MaxReduction<T> > {
  typedef typename MaxReduction<T>::LHS LHS;
  typedef typename MaxReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return std::max(x, y); }
};
template <typename T> struct RedopTest<MinReduction<T> > {
  typedef typename MinReduction<T>::LHS LHS;
  typedef typename MinReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return std::min(x, y); }
};
template <typename T> struct RedopTest<OrReduction<T> > {
  typedef typename OrReduction<T>::LHS LHS;
  typedef typename OrReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return x | y; }
};
template <typename T> struct RedopTest<AndReduction<T> > {
  typedef typename AndReduction<T>::LHS LHS;
  typedef typename AndReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return x & y; }
};
template <typename T> struct RedopTest<XorReduction<T> > {
  typedef typename XorReduction<T>::LHS LHS;
  typedef typename XorReduction<T>::RHS RHS;
  static LHS invoke(const LHS &x, const RHS &y) { return x ^ y; }
};

template <> struct RedopTest<SumReduction<bool> > {
  typedef typename SumReduction<bool>::LHS LHS;
  typedef typename SumReduction<bool>::RHS RHS;
  static bool invoke(const bool &x, const bool &y) { return x || y; }
};
template <> struct RedopTest<ProdReduction<bool> > {
  typedef typename ProdReduction<bool>::LHS LHS;
  typedef typename ProdReduction<bool>::RHS RHS;
  static bool invoke(const bool &x, const bool &y) { return x && y; }
};

template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value, void>::type
get_rand(T &v, RNG &rng) {
  static std::uniform_real_distribution<T> dist;
  v = dist(rng);
}

template <typename T> static void get_rand(complex<T> &v, RNG &rng) {
  T r, i;
  get_rand<T>(r, rng);
  get_rand<T>(i, rng);
  v = complex<T>(r, i);
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, void>::type
get_rand(T &v, RNG &rng) {
  static std::uniform_int_distribution<T> dist;
  v = dist(rng);
}

template <> void get_rand<bool>(bool &v, RNG &rng) {
  unsigned char x = 0;
  get_rand<unsigned char>(x, rng);
  v = !!x;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
check_value(const T &a, const T &b, const char *name, const char *file,
            int line) {
  if (a != b) {
    std::cout << std::boolalpha << "Comparision failed at " << file << ':'
              << line << '(' << name << ")! Expected " << a << ", got " << b
              << std::endl;
    assert(a == b);
  }
}

// Floating point operations, especially complicated ones, can lead to rounding
// errors, so just check that result is within two epsilons to allow for some
// inaccuracy
template <typename T>
static inline bool fuzzyCompare(const T &a, const T &b,
                                          const T &eps) {
  if (a == b)
    return true;
  const T norm = std::min(std::abs(a + b), std::numeric_limits<T>::max());
  return std::abs(a - b) < std::max(std::numeric_limits<T>::min(), eps * norm);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
check_value(const T &a, const T &b, const char *name, const char *file,
            int line) {
  const T epsilon = 2 * std::numeric_limits<T>::epsilon();
  if (!fuzzyCompare(a, b, epsilon)) {
    std::cout << std::setprecision(10) << "Comparision failed at " << file
              << ':' << line << '(' << name << ")! Expected " << (unsigned)a
              << ", got " << (unsigned)b << std::endl;
    assert(a == b);
  }
}

template <typename T>
void check_value(const std::complex<T> &a, const std::complex<T> &b,
                 const char *name, const char *file, int line) {
  const T epsilon = 2 * std::numeric_limits<T>::epsilon();
  if (!(fuzzyCompare(a.real(), b.real(), epsilon) &&
        fuzzyCompare(a.imag(), b.imag(), epsilon))) {
    std::cout << std::setprecision(10) << "Comparision failed at " << file
              << ':' << line << '(' << name << ")! Expected " << a << ", got "
              << b << std::endl;
    assert(a == b);
  }
}

// Just to get a printable version of the value for character types
template <>
void check_value<uint8_t>(const uint8_t &a, const uint8_t &b, const char *name,
                          const char *file, int line) {
  if (a != b) {
    std::cout << "Comparision failed at " << file << ':' << line << '(' << name
              << ")! Expected " << (unsigned)a << ", got " << (unsigned)b
              << std::endl;
    assert(a == b);
  }
}

template <>
void check_value<int8_t>(const int8_t &a, const int8_t &b, const char *name,
                         const char *file, int line) {
  if (a != b) {
    std::cout << "Comparision failed at " << file << ':' << line << '(' << name
              << ")! Expected " << (int)a << ", got " << (int)b << std::endl;
    assert(a == b);
  }
}

#define CHECK_VALUE(a, b, name) check_value(a, b, name, __FILE__, __LINE__)

template <typename Redop> static void test_redop(const char *name, RNG &rng) {

  // Run the test over several different values
  for (size_t i = 0; i < 10; i++) {
    typename Redop::LHS test_lhs, gold_value, test_lhs_copy;
    typename Redop::RHS test_rhs;

    get_rand(test_lhs, rng);
    get_rand(test_rhs, rng);

    gold_value = RedopTest<Redop>::invoke(test_lhs, test_rhs);
    test_lhs_copy = test_lhs;

    // Verify each case sets test_lhs to the gold value
    Redop::template apply<true>(test_lhs, test_rhs);
    CHECK_VALUE(gold_value, test_lhs, name);
    test_lhs = test_lhs_copy;

    Redop::template apply<false>(test_lhs, test_rhs);
    CHECK_VALUE(gold_value, test_lhs, name);
    test_lhs = test_lhs_copy;

    Redop::template fold<true>(test_lhs, test_rhs);
    CHECK_VALUE(gold_value, test_lhs, name);
    test_lhs = test_lhs_copy;

    Redop::template fold<false>(test_lhs, test_rhs);
    CHECK_VALUE(gold_value, test_lhs, name);
    test_lhs = test_lhs_copy;
  }
}

int main() {
  RNG rng;

#define RUN_TEST(id, redop)                                                    \
  assert(redop::REDOP_ID == id);                                               \
  test_redop<redop>(#redop, rng);
  LEGION_REDOP_LIST(RUN_TEST)

  return 0;
}
