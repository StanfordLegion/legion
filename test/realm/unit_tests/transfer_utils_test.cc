#include "realm/transfer/transfer_utils.h"
#include <variant>
#include <vector>
#include <numeric>
#include <gtest/gtest.h>

using namespace Realm;

template <typename PointType>
struct PointTraits;

template <int N, typename T>
struct PointTraits<Realm::Point<N, T>> {
  static constexpr int DIM = N;
  using value_type = T;
};

template <typename PointType>
class NextRectTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;

  void SetUp() override
  {
    dim_order.resize(N);
    std::iota(dim_order.begin(), dim_order.end(), 0);
  }

  std::vector<int> dim_order;
};

TYPED_TEST_SUITE_P(NextRectTest);

TYPED_TEST_P(NextRectTest, EmptyDomain)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;
  Rect<N, T> subrect;
  Rect<N, T> domain = Rect<N, T>(TypeParam(1), TypeParam(0));
  TypeParam next_start(0);

  EXPECT_TRUE(next_subrect(domain, TypeParam(1), domain, this->dim_order.data(), subrect,
                           next_start));

  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, TypeParam(0));
}

TYPED_TEST_P(NextRectTest, ContainsFullSubrect)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;
  Rect<N, T> subrect;
  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(8));
  TypeParam next_start(0);

  EXPECT_TRUE(next_subrect(domain, TypeParam(0), domain, this->dim_order.data(), subrect,
                           next_start));

  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, TypeParam(0));
}

TYPED_TEST_P(NextRectTest, ContainsPartialSubrect)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;
  const T size = 4;
  Rect<N, T> subrect;
  Rect<N, T> bounds = Rect<N, T>(TypeParam(0), TypeParam(size));
  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(size * 2));
  TypeParam exp_lo = TypeParam(0);
  exp_lo.x() = size;
  TypeParam exp_next = TypeParam(0);
  exp_next.x() = size + 1;

  TypeParam next_start(0);
  bool done = next_subrect(domain, TypeParam(0), bounds, this->dim_order.data(), subrect,
                           next_start);

  EXPECT_FALSE(done);
  EXPECT_EQ(subrect.lo, TypeParam(0));
  EXPECT_EQ(subrect.hi, exp_lo);
  EXPECT_EQ(next_start, exp_next);
}

REGISTER_TYPED_TEST_SUITE_P(NextRectTest, EmptyDomain, ContainsFullSubrect,
                            ContainsPartialSubrect);

template <typename T, int... Ns>
auto GeneratePointTypes(std::integer_sequence<int, Ns...>)
{
  return ::testing::Types<Realm::Point<Ns + 1, T>...>{};
}

template <typename T>
auto GeneratePointTypesForAllDims()
{
  return GeneratePointTypes<T>(std::make_integer_sequence<int, REALM_MAX_DIM>{});
}

#define INSTANTIATE_TEST_TYPES(BASE_TYPE, SUFFIX)                                        \
  using N##SUFFIX = decltype(GeneratePointTypesForAllDims<BASE_TYPE>());                 \
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, NextRectTest, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int);
INSTANTIATE_TEST_TYPES(long long, LongLong);

template <int N>
struct ComputeTargetSubrectTestCase {
  std::vector<int> dim_order;
  Rect<N> bounds;
  Rect<N> cur_rect;
  Point<N> cur_point;
  std::vector<Rect<N>> expected_rects;
};

// Define aliases for specific dimensions
using ComputeTargetSubrectTestCase1D = ComputeTargetSubrectTestCase<1>;
using ComputeTargetSubrectTestCase2D = ComputeTargetSubrectTestCase<2>;
using ComputeTargetSubrectTestCase3D = ComputeTargetSubrectTestCase<3>;

// Unified test case variant
using ComputeTargetSubrectTestVariant =
    std::variant<ComputeTargetSubrectTestCase1D, ComputeTargetSubrectTestCase2D,
                 ComputeTargetSubrectTestCase3D>;

class ComputeTargetSubrectTest
  : public ::testing::TestWithParam<ComputeTargetSubrectTestVariant> {
protected:
  void TearDown() override {}

  template <int N>
  void RunTest(const ComputeTargetSubrectTestCase<N> &test_case)
  {
    std::vector<Rect<N>> rects;
    Point<N> cur_point = test_case.cur_point;

    bool not_done = false;
    do {
      Rect<N> next_subrect;
      not_done = compute_target_subrect(test_case.bounds, test_case.cur_rect, cur_point,
                                        next_subrect, test_case.dim_order.data());
      rects.push_back(next_subrect);
    } while(not_done);

    ASSERT_EQ(test_case.expected_rects.size(), rects.size());
    for(size_t i = 0; i < test_case.expected_rects.size(); i++) {
      EXPECT_EQ(test_case.expected_rects[i].lo, rects[i].lo);
      EXPECT_EQ(test_case.expected_rects[i].hi, rects[i].hi);
    }
  }
};

TEST_P(ComputeTargetSubrectTest, NextTargetSubrectEmpty)
{
  std::visit([this](auto &&test_case) { RunTest(test_case); }, GetParam());
}

// 1D test cases
const static ComputeTargetSubrectTestCase1D kComputeSubrectTestCases1D[] = {
    // Simple full span in 1D
    {{0},
     Rect<1>(Point<1>(0), Point<1>(10)),
     Rect<1>(Point<1>(0), Point<1>(10)),
     Point<1>(0),
     {Rect<1>(Point<1>(0), Point<1>(10))}},
    // Partial span in 1D
    {{0},
     Rect<1>(Point<1>(0), Point<1>(10)),
     Rect<1>(Point<1>(3), Point<1>(7)),
     Point<1>(3),
     {Rect<1>(Point<1>(3), Point<1>(7))}},
    // Out of bounds starting point in 1D
    /*{{0},
     Rect<1>(Point<1>(0), Point<1>(10)),
     Rect<1>(Point<1>(11), Point<1>(15)),
     Point<1>(11),
     {}},*/
};

#if REALM_MAX_DIM > 1
// 2D test cases
const static ComputeTargetSubrectTestCase2D kComputeSubrectTestCases2D[] = {
    // Full cover in 2D
    {{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Point<2>(0, 0),
     {Rect<2>(Point<2>(0, 0), Point<2>(10, 10))}},
    // Partial span in 2D
    {{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(0, 0), Point<2>(10, 5)),
     Point<2>(0, 0),
     {Rect<2>(Point<2>(0, 0), Point<2>(10, 5))}},
    // Middle start point in 2D
    {{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(0, 0), Point<2>(10, 5)),
     Point<2>(4, 4),
     {Rect<2>(Point<2>(4, 4), Point<2>(10, 4)),
      Rect<2>(Point<2>(0, 5), Point<2>(10, 5))}},
    // Out of bounds in 2D
    /*{{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(11, 11), Point<2>(12, 12)),
     Point<2>(11, 11),
     {}},*/
};
#endif

#if REALM_MAX_DIM > 2
// 3D test cases
const static ComputeTargetSubrectTestCase3D kComputeSubrectTestCases3D[] = {
    // Full cover in 3D
    {{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Point<3>(0, 0, 0),
     {Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10))}},
    // Partial span in 3D
    {{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 5, 5)),
     Point<3>(0, 0, 0),
     {Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 5, 5))}},
    // Middle start in 3D
    {{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 5, 5)),
     Point<3>(4, 4, 4),
     {Rect<3>(Point<3>(4, 4, 4), Point<3>(10, 4, 4)),
      Rect<3>(Point<3>(0, 5, 4), Point<3>(10, 5, 4)),
      Rect<3>(Point<3>(0, 0, 5), Point<3>(10, 5, 5))}},
    // Out of bounds in 3D
    /*{{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(11, 11, 11), Point<3>(12, 12, 12)),
     Point<3>(11, 11, 11),
     {}},*/
};
#endif

// Concatenate all test cases into one vector
std::vector<ComputeTargetSubrectTestVariant> allTestCases()
{
  std::vector<ComputeTargetSubrectTestVariant> cases;
  cases.insert(cases.end(), std::begin(kComputeSubrectTestCases1D),
               std::end(kComputeSubrectTestCases1D));

#if REALM_MAX_DIM > 1
  cases.insert(cases.end(), std::begin(kComputeSubrectTestCases2D),
               std::end(kComputeSubrectTestCases2D));
#endif
#if REALM_MAX_DIM > 2
  cases.insert(cases.end(), std::begin(kComputeSubrectTestCases3D),
               std::end(kComputeSubrectTestCases3D));
#endif

  return cases;
}

INSTANTIATE_TEST_SUITE_P(TestAllDimensions, ComputeTargetSubrectTest,
                         testing::ValuesIn(allTestCases()));
