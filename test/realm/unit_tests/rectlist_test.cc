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
class DenseRectListTest : public ::testing::Test {
public:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;
};

TYPED_TEST_SUITE_P(DenseRectListTest);

TYPED_TEST_P(DenseRectListTest, AddPoints)
{
  const size_t max_rects = 3;
  DenseRectangleList<TestFixture::N, typename TestFixture::T> rectlist(max_rects);
  std::vector<TypeParam> points;

  for(size_t i = 0; i < max_rects; i++) {
    TypeParam point(0);
    point.x() = i;
    points.emplace_back(point);
    rectlist.add_point(points.back());
  }

  EXPECT_EQ(rectlist.rects.size(), 1);
  EXPECT_EQ(TypeParam(0), rectlist.rects[0].lo);
  EXPECT_EQ(points.back(), rectlist.rects[0].hi);
}

TYPED_TEST_P(DenseRectListTest, AddMaxDisjointRects)
{
  const size_t max_rects = 3;
  DenseRectangleList<TestFixture::N, typename TestFixture::T> rectlist(max_rects);
  std::vector<Rect<TestFixture::N, typename TestFixture::T>> rects;

  size_t shift = 0;
  for(size_t i = 0; i < max_rects; i++) {
    rects.emplace_back(Rect<TestFixture::N, typename TestFixture::T>(
        TypeParam(shift), TypeParam(shift + 1)));
    rectlist.add_rect(rects.back());
    shift += 3;
  }

  EXPECT_EQ(rectlist.rects.size(), max_rects);
  for(size_t i = 0; i < max_rects; i++) {
    EXPECT_EQ(rects[i].lo, rectlist.rects[i].lo);
    EXPECT_EQ(rects[i].hi, rectlist.rects[i].hi);
  }
}

REGISTER_TYPED_TEST_SUITE_P(DenseRectListTest, AddMaxDisjointRects, AddPoints);

#define TEST_POINT_TYPES(T) GeneratePointTypesForAllDims<T>()

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
  using N##SUFFIX = decltype(TEST_POINT_TYPES(BASE_TYPE));                               \
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, SUITE, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int, DenseRectListTest);
INSTANTIATE_TEST_TYPES(long long, LongLong, DenseRectListTest);
