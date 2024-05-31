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
class PointTest : public ::testing::Test {
protected:
  using T_ = typename T::type;

  void SetUp() override
  {
    for(int i = 0; i < T::value; i++) {
      values1[i] = i + 1;
      values2[i] = i * 2;
    }
  }

  T_ values1[T::value];
  T_ values2[T::value];
};

TYPED_TEST_SUITE_P(PointTest);

TYPED_TEST_P(PointTest, Zeroes)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point = Point<N, T>::ZEROES();
  for(int i = 0; i < N; i++)
    EXPECT_EQ(point[i], 0);
}

TYPED_TEST_P(PointTest, Ones)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point = Point<N, T>::ONES();
  for(int i = 0; i < N; i++)
    EXPECT_EQ(point[i], 1);
}

TYPED_TEST_P(PointTest, BaseAccess)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point(this->values1);

  for(int i = 0; i < N; i++)
    EXPECT_EQ(point[i], this->values1[i]);
}

TYPED_TEST_P(PointTest, Equality)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point1(this->values1);
  Point<N, T> point2(this->values1);
  Point<N, T> point3(this->values2);

  EXPECT_TRUE(point1 == point2);
  EXPECT_FALSE(point1 == point3);
}

TYPED_TEST_P(PointTest, Add)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point1(this->values1);
  Point<N, T> point2(this->values2);
  Point<N, T> result = point1 + point2;

  for(int i = 0; i < N; i++)
    EXPECT_EQ(result[i], this->values1[i] + this->values2[i]);
}

TYPED_TEST_P(PointTest, Subtract)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point1(this->values2);
  Point<N, T> point2(this->values1);
  Point<N, T> result = point1 - point2;

  for(int i = 0; i < N; i++)
    EXPECT_EQ(result[i], this->values2[i] - this->values1[i]);
}

TYPED_TEST_P(PointTest, Multiply)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point1(this->values2);
  Point<N, T> point2(this->values1);
  Point<N, T> result = point1 * point2;

  for(int i = 0; i < N; i++)
    EXPECT_EQ(result[i], this->values2[i] * this->values1[i]);
}

TYPED_TEST_P(PointTest, Divide)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point1(this->values2);
  Point<N, T> point2(this->values1);
  Point<N, T> result = point1 / point2;

  for(int i = 0; i < N; i++)
    EXPECT_EQ(result[i], this->values2[i] / this->values1[i]);
}

TYPED_TEST_P(PointTest, Modulo)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  Point<N, T> point1(this->values2);
  Point<N, T> point2(this->values1);
  Point<N, T> result = point1 % point2;

  for(int i = 0; i < N; i++)
    EXPECT_EQ(result[i], this->values2[i] % this->values1[i]);
}

TYPED_TEST_P(PointTest, Dot)
{
  constexpr int N = TypeParam::value;
  using T = typename TypeParam::type;
  T product = 0;
  for(int i = 0; i < N; i++) {
    product += this->values1[i] * this->values2[i];
  }

  Point<N, T> point1(this->values1);
  Point<N, T> point2(this->values2);
  T dot = point1.dot(point2);

  EXPECT_EQ(dot, product);
}

REGISTER_TYPED_TEST_SUITE_P(PointTest, BaseAccess, Equality, Dot, Zeroes, Ones, Add,
                            Subtract, Multiply, Divide, Modulo);

using test_types =
    ::testing::Types<ValueAndType<1, int>, ValueAndType<2, int>, ValueAndType<3, int>,
                     ValueAndType<1, long long>, ValueAndType<2, long long>,
                     ValueAndType<2, long long>>;
INSTANTIATE_TYPED_TEST_SUITE_P(Instantiation, PointTest, test_types);
