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

// TODO(apryakhin@):
// 1. Consider merging tests
// 2. Add edge cases

TYPED_TEST_P(RectListTest, AddPoints)
{
  constexpr size_t max_rects = 3;
  DenseRectangleList<TestFixture::N, typename TestFixture::T> rectlist(max_rects);
  std::vector<TypeParam> points;

  for(size_t i = 0; i < max_rects; i++) {
    TypeParam point(0);
    point.x() = i;
    points.emplace_back(point);
    rectlist.add_point(points.back());
  }

  EXPECT_EQ(rectlist.rects.size(), 1);
  EXPECT_EQ(TypeParam(0), rectlist.rects.back().lo);
  EXPECT_EQ(points.back(), rectlist.rects.back().hi);
}

TYPED_TEST_P(RectListTest, AddDisjointPoints)
{
  constexpr size_t max_rects = 3;
  DenseRectangleList<TestFixture::N, typename TestFixture::T> rectlist(max_rects);
  std::vector<TypeParam> points;

  size_t shift = 0;
  for(size_t i = 0; i < max_rects; i++) {
    TypeParam point(shift);
    shift += 2;
    points.emplace_back(point);
    rectlist.add_point(points.back());
  }

  EXPECT_EQ(rectlist.rects.size(), max_rects);
  for(size_t i = 0; i < max_rects; i++) {
    EXPECT_EQ(points[i], rectlist.rects[i].lo);
    EXPECT_EQ(points[i], rectlist.rects[i].hi);
  }
}

TYPED_TEST_P(RectListTest, AddDisjointRects)
{
  constexpr size_t max_rects = 3;
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

TYPED_TEST_P(RectListTest, HybridAddDisjointRects)
{
  constexpr size_t max_rects =
      HybridRectangleList<TestFixture::N, typename TestFixture::T>::HIGH_WATER_MARK + 10;
  HybridRectangleList<TestFixture::N, typename TestFixture::T> rectlist;
  std::vector<Rect<TestFixture::N, typename TestFixture::T>> rects;

  size_t shift = 0;
  for(size_t i = 0; i < max_rects; i++) {
    rects.emplace_back(Rect<TestFixture::N, typename TestFixture::T>(
        TypeParam(shift), TypeParam(shift + 1)));
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
    points.emplace_back(point);
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
    points.emplace_back(point);
    rectlist.add_point(points.back());
  }

  auto rect_vector = rectlist.convert_to_vector();
  EXPECT_EQ(rect_vector.size(), max_rects);
  for(size_t i = 0; i < max_rects; i++) {
    EXPECT_EQ(points[i], rect_vector[i].lo);
    EXPECT_EQ(points[i], rect_vector[i].hi);
  }
}

TYPED_TEST_P(RectListTest, CoverageCounter)
{
  constexpr size_t max_elements = 3;
  size_t elements = 0;
  Realm::CoverageCounter<TestFixture::N, typename TestFixture::T> counter;

  for(size_t i = 0; i < max_elements; i++) {
    auto rect =
        Rect<TestFixture::N, typename TestFixture::T>(TypeParam(i), TypeParam(i + 1));
    counter.add_rect(rect);
    elements += rect.volume();
  }

  for(size_t i = 0; i < max_elements; i++) {
    TypeParam point(i);
    counter.add_point(point);
  }
  elements += max_elements;

  EXPECT_EQ(counter.get_count(), elements);
}

REGISTER_TYPED_TEST_SUITE_P(RectListTest, AddPoints, AddDisjointPoints, AddDisjointRects,
                            HybridAddDisjointRects, HybridAddPoints,
                            HybridAddDisjointPoints, CoverageCounter);

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
