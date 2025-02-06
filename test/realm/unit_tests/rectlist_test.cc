#include "realm/deppart/rectlist.h"
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
class RectListTest : public ::testing::Test {
public:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;
};

TYPED_TEST_SUITE_P(RectListTest);

TYPED_TEST_P(RectListTest, HybridAddDisjointRects)
{
  constexpr size_t max_rects =
      HybridRectangleList<TestFixture::N, typename TestFixture::T>::HIGH_WATER_MARK + 10;
  HybridRectangleList<TestFixture::N, typename TestFixture::T> rectlist;
  std::vector<Rect<TestFixture::N, typename TestFixture::T>> rects;

  size_t shift = 0;
  for(size_t i = 0; i < max_rects; i++) {
    rects.push_back(Rect<TestFixture::N, typename TestFixture::T>(TypeParam(shift),
                                                                  TypeParam(shift + 1)));
    rectlist.add_rect(rects.back());
    shift += 3;
  }

  auto rect_vector = rectlist.convert_to_vector();
  EXPECT_EQ(rect_vector.size(), max_rects);
  for(size_t i = 0; i < max_rects; i++) {
    EXPECT_EQ(rects[i].lo, rect_vector[i].lo);
    EXPECT_EQ(rects[i].hi, rect_vector[i].hi);
  }
}

TYPED_TEST_P(RectListTest, HybridAddPoints)
{
  constexpr size_t max_rects =
      HybridRectangleList<TestFixture::N, typename TestFixture::T>::HIGH_WATER_MARK + 1;
  HybridRectangleList<TestFixture::N, typename TestFixture::T> rectlist;
  std::vector<TypeParam> points;

  for(size_t i = 0; i < max_rects; i++) {
    TypeParam point(0);
    point.x() = i;
    points.push_back(point);
    rectlist.add_point(points.back());
  }

  auto rect_vector = rectlist.convert_to_vector();
  EXPECT_EQ(rect_vector.size(), 1);
  EXPECT_EQ(TypeParam(0), rect_vector.back().lo);
  EXPECT_EQ(points.back(), rect_vector.back().hi);
}

TYPED_TEST_P(RectListTest, HybridAddDisjointPoints)
{
  constexpr size_t max_rects =
      HybridRectangleList<TestFixture::N, typename TestFixture::T>::HIGH_WATER_MARK + 10;
  HybridRectangleList<TestFixture::N, typename TestFixture::T> rectlist;
  std::vector<TypeParam> points;

  size_t shift = 0;
  for(size_t i = 0; i < max_rects; i++) {
    TypeParam point(shift);
    shift += 2;
    points.push_back(point);
    rectlist.add_point(points.back());
  }

  auto rect_vector = rectlist.convert_to_vector();
  EXPECT_EQ(rect_vector.size(), max_rects);
  for(size_t i = 0; i < max_rects; i++) {
    EXPECT_EQ(points[i], rect_vector[i].lo);
    EXPECT_EQ(points[i], rect_vector[i].hi);
  }
}

REGISTER_TYPED_TEST_SUITE_P(RectListTest, HybridAddDisjointRects, HybridAddPoints,
                            HybridAddDisjointPoints);

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

#define INSTANTIATE_TEST_TYPES(BASE_TYPE, SUFFIX, SUITE)                                 \
  using N##SUFFIX = decltype(GeneratePointTypesForAllDims<BASE_TYPE>());                 \
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, SUITE, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int, RectListTest);
INSTANTIATE_TEST_TYPES(long long, LongLong, RectListTest);

template <int DIM>
struct TypeWrapper {
  static constexpr int value = DIM;
};

template <typename TypeWrapper>
class DenseAddPointTest : public ::testing::Test {
public:
  static constexpr int DIM = TypeWrapper::value;
};

template <int DIM>
struct AddPointTestCase {
  size_t max_rects;
  std::vector<Point<DIM>> points;
  std::vector<Rect<DIM>> rects;
};

template <int DIM>
std::vector<AddPointTestCase<DIM>> GetAddPointTestCases()
{
  if constexpr(DIM == 1) {
    return {
        // Case 1: empty
        {
            /*max_rects=*/1,
            /*points=*/{},
            /*rects=*/{},
        },

        // Case 1: zero max_rects
        {
            /*max_rects=*/0,
            /*points=*/{Point<1>(0), Point<1>(1), Point<1>(2)},
            /*rects=*/{Rect<1>(0, 2)},
        },

        // Case 1: all points are mergeable
        {
            /*max_rects=*/2,
            /*points=*/{Point<1>(0), Point<1>(1), Point<1>(2)},
            /*rects=*/{Rect<1>(0, 2)},
        },

        // Case 2: all points are disjoint
        {
            /*max_rects=*/3,
            /*points=*/{Point<1>(0), Point<1>(2), Point<1>(4)},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
        },

        // Case 3: all points are disjoint and limited by max_rects
        {
            /*max_rects=*/2,
            /*points=*/{Point<1>(0), Point<1>(2), Point<1>(4)},
            /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 4)},
        },
    };
  } else if constexpr(DIM == 2) {
    return {

        // Case 7: fully mergable along x dimension
        {
            /*max_rects=*/1,
            /*points=*/{Point<2>(0, 0), Point<2>(0, 1), Point<2>(0, 2)},
            /*rects=*/{Rect<2>({0, 0}, {0, 2})},
        },
    };
  } else if constexpr(DIM == 3) {
    return {
        // Case 11: disjoint 3D
        {
            /*max_rects=*/1,
            /*points=*/{Point<3>(0, 0, 0), Point<3>(2, 2, 2), Point<3>(4, 4, 4)},
            /*rects=*/{Rect<3>({0, 0, 0}, {4, 4, 4})},
        },
    };
  }
  return {};
}

TYPED_TEST_SUITE_P(DenseAddPointTest);

TYPED_TEST_P(DenseAddPointTest, Base)
{
  using T = int;
  constexpr int N = TypeParam::value;

  auto test_cases = GetAddPointTestCases<N>();
  for(const auto &test_case : test_cases) {
    DenseRectangleList<N, T> rectlist(test_case.max_rects);
    for(const auto p : test_case.points) {
      rectlist.add_point(p);
    }
    EXPECT_EQ(rectlist.rects.size(), test_case.rects.size());
    for(size_t i = 0; i < test_case.rects.size(); i++) {
      EXPECT_EQ(rectlist.rects[i].lo, test_case.rects[i].lo);
      EXPECT_EQ(rectlist.rects[i].hi, test_case.rects[i].hi);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(DenseAddPointTest, Base);

template <typename Seq>
struct WrapTypes;

template <std::size_t... Ns>
struct WrapTypes<std::index_sequence<Ns...>> {
  using type = ::testing::Types<TypeWrapper<Ns + 1>...>;
};

using TestTypes = typename WrapTypes<std::make_index_sequence<REALM_MAX_DIM>>::type;

INSTANTIATE_TYPED_TEST_SUITE_P(AllDimensions, DenseAddPointTest, TestTypes);

template <typename TypeWrapper>
class DenseAddRectTest : public ::testing::Test {
public:
  static constexpr int DIM = TypeWrapper::value;
};

template <int DIM>
struct AddRectCase {
  size_t max_rects;
  std::vector<Rect<DIM>> rects;
  std::vector<Rect<DIM>> expected;
};

template <int DIM>
std::vector<AddRectCase<DIM>> GetAddRectTestCases()
{
  if constexpr(DIM == 1) {
    return {
        // Case 1: empty
        {
            /*max_rects=*/1,
            /*rects=*/{},
            /*expected=*/{},
        },

        // Case 1: zero max_rects
        {
            /*max_rects=*/0,
            /*rects=*/{Rect<1>(0, 0), Rect<1>(1, 1), Rect<1>(2, 2)},
            /*expected=*/{Rect<1>(0, 2)},
        },

        // Case 1: all rects are mergeable
        {
            /*max_rects=*/2,
            /*rects=*/{Rect<1>(0, 0), Rect<1>(1, 1), Rect<1>(2, 2)},
            /*expected=*/{Rect<1>(0, 2)},
        },

        // Case 2: all rects are disjoint
        {
            /*max_rects=*/3,
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
        },

        // Case 3: all rects are disjoint and limited by max_rects
        {
            /*max_rects=*/2,
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
            /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 4)},
        },
    };
  } else if constexpr(DIM == 2) {
    return {

        // Case 7: fully mergable along x dimension
        {
            /*max_rects=*/1,
            /*rects=*/
            {Rect<2>({0, 0}, {0, 0}), Rect<2>({0, 0}, {1, 1}), Rect<2>({0, 0}, {2, 2})},
            /*expected=*/{Rect<2>({0, 0}, {2, 2})},
        },
    };
  } else if constexpr(DIM == 3) {
    return {
        // Case 11: disjoint 3D
        {
            /*max_rects=*/1,
            {Rect<3>({0, 0, 0}, {0, 0, 0}), Rect<3>({2, 2, 2}, {2, 2, 2}),
             Rect<3>({4, 4, 4}, {4, 4, 4})},
            /*expected=*/{Rect<3>({0, 0, 0}, {4, 4, 4})},
        },
    };
  }
  return {};
}

TYPED_TEST_SUITE_P(DenseAddRectTest);

TYPED_TEST_P(DenseAddRectTest, Base)
{
  using T = int;
  constexpr int N = TypeParam::value;

  auto test_cases = GetAddRectTestCases<N>();
  for(const auto &test_case : test_cases) {
    DenseRectangleList<N, T> rectlist(test_case.max_rects);
    for(const auto r : test_case.rects) {
      rectlist.add_rect(r);
    }
    EXPECT_EQ(rectlist.rects.size(), test_case.expected.size());
    for(size_t i = 0; i < test_case.expected.size(); i++) {
      EXPECT_EQ(rectlist.rects[i].lo, test_case.expected[i].lo);
      EXPECT_EQ(rectlist.rects[i].hi, test_case.expected[i].hi);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(DenseAddRectTest, Base);

INSTANTIATE_TYPED_TEST_SUITE_P(AllDimensions, DenseAddRectTest, TestTypes);

template <typename TypeWrapper>
class CoverageCounterTest : public ::testing::Test {
public:
  static constexpr int DIM = TypeWrapper::value;
};

template <int DIM>
struct CoverageCounterTestCase {
  std::vector<Rect<DIM>> rects;
  std::vector<Point<DIM>> points;
  size_t expected;
};

template <int DIM>
std::vector<CoverageCounterTestCase<DIM>> GetCoverageCounterTestCases()
{
  if constexpr(DIM == 1) {
    return {
        // Case empty
        {/*rects=*/{}, /*points=*/{}, /*expected=*/0},

        // Case empty rects
        {/*rects=*/{Rect<DIM>({1}, {0}), Rect<DIM>({3}, {2})}, /*points=*/{},
         /*expected=*/0},

        // Case normal rects
        {/*rects=*/{Rect<DIM>({0}, {1}), Rect<DIM>({3}, {4})}, /*points=*/{},
         /*expected=*/4},

        // Case normal rects and points
        {/*rects=*/{Rect<DIM>({0}, {1}), Rect<DIM>({3}, {4})},
         /*points=*/{Point<DIM>(0), Point<DIM>(1)}, /*expected=*/6},

    };
  } else if constexpr(DIM == 2) {
    // Case normal 2D rects and points
    return {
        {/*rects=*/{Rect<DIM>({0, 0}, {1, 1}), Rect<DIM>({3, 3}, {4, 4})},
         /*points=*/{Point<DIM>(0), Point<DIM>(1)}, /*expected=*/10},
    };
  }
  return {};
}

TYPED_TEST_SUITE_P(CoverageCounterTest);

TYPED_TEST_P(CoverageCounterTest, Base)
{
  using T = int;
  constexpr int N = TypeParam::value;

  auto test_cases = GetCoverageCounterTestCases<N>();
  for(const auto &test_case : test_cases) {
    Realm::CoverageCounter<N, T> counter;

    for(const auto &r : test_case.rects) {
      counter.add_rect(r);
    }

    for(const auto &p : test_case.points) {
      counter.add_point(p);
    }

    EXPECT_EQ(counter.get_count(), test_case.expected);
  }
}

REGISTER_TYPED_TEST_SUITE_P(CoverageCounterTest, Base);
INSTANTIATE_TYPED_TEST_SUITE_P(AllDimensions, CoverageCounterTest, TestTypes);
