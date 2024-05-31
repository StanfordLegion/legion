#include "realm/point.h"

#include <tuple>
#include <type_traits>
#include <gtest/gtest.h>

using namespace Realm;

template <int N, typename T>
struct ValueAndType {
  static constexpr int value = N;
  using type = T;
};

template <typename T>
class PointTest : public ::testing::Test {};

using test_types =
    ::testing::Types<ValueAndType<1, int>, ValueAndType<2, int>, ValueAndType<3, int>,
                     ValueAndType<1, long long>, ValueAndType<2, long long>,
                     ValueAndType<2, long long>>;

TYPED_TEST_SUITE_P(PointTest);

TYPED_TEST_P(PointTest, BaseAccess)
{
  using T = typename TypeParam::type;
  constexpr int N = TypeParam::value;
  T values[N];
  for(int i = 0; i < N; i++)
    values[i] = 42 + i;

  Point<N, T> point(values);

  for(int i = 0; i < N; i++)
    EXPECT_EQ(point[i], values[i]);
}

TYPED_TEST_P(PointTest, Equality)
{
  using T = typename TypeParam::type;
  constexpr int N = TypeParam::value;
  T values1[N];
  T values2[N];
  for(int i = 0; i < N; i++) {
    values1[i] = 42 + i;
    values2[i] = 43 + i;
  }

  Point<N, T> point1(values1);
  Point<N, T> point2(values1);
  Point<N, T> point3(values2);

  EXPECT_TRUE(point1 == point2);
  EXPECT_FALSE(point1 == point3);
}

REGISTER_TYPED_TEST_SUITE_P(PointTest, BaseAccess, Equality);
INSTANTIATE_TYPED_TEST_SUITE_P(Instantiation, PointTest, test_types);
