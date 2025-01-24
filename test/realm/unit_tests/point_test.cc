#include "realm/point.h"
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
class PointTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;

  void SetUp() override
  {
    for(int i = 0; i < N; i++) {
      values1[i] = i + 1;
      values2[i] = i * 2;
    }
  }

  T values1[N];
  T values2[N];
};

TYPED_TEST_SUITE_P(PointTest);

TYPED_TEST_P(PointTest, Zeroes)
{
  using T = typename TestFixture::T;
  TypeParam point = TypeParam::ZEROES();
  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point[i], static_cast<T>(0));
  }
}

TYPED_TEST_P(PointTest, Ones)
{
  using T = typename TestFixture::T;
  TypeParam point = TypeParam::ONES();
  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point[i], static_cast<T>(1));
  }
}

TYPED_TEST_P(PointTest, BaseAccess)
{
  TypeParam point(this->values1);
  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(point[i], this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Equality)
{
  TypeParam point1(this->values1);
  TypeParam point2(this->values1);
  TypeParam point3(this->values2);

  EXPECT_TRUE(point1 == point2);
  EXPECT_FALSE(point1 == point3);
}

TYPED_TEST_P(PointTest, Add)
{
  TypeParam point1(this->values1);
  TypeParam point2(this->values2);
  TypeParam result = point1 + point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values1[i] + this->values2[i]);
  }

  EXPECT_EQ(result.x(), this->values1[0] + this->values2[0]);

  if constexpr(TestFixture::N > 1) {
    EXPECT_EQ(result.y(), this->values1[1] + this->values2[1]);
  }

  if constexpr(TestFixture::N > 2) {
    EXPECT_EQ(result.z(), this->values1[2] + this->values2[2]);
  }

  if constexpr(TestFixture::N > 3) {
    EXPECT_EQ(result.w(), this->values1[3] + this->values2[3]);
  }
}

TYPED_TEST_P(PointTest, Subtract)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 - point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] - this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Multiply)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 * point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] * this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Divide)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 / point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] / this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Modulo)
{
  TypeParam point1(this->values2);
  TypeParam point2(this->values1);
  TypeParam result = point1 % point2;

  for(int i = 0; i < TestFixture::N; i++) {
    EXPECT_EQ(result[i], this->values2[i] % this->values1[i]);
  }
}

TYPED_TEST_P(PointTest, Dot)
{
  using T = typename TestFixture::T;
  T product = 0;
  for(int i = 0; i < TestFixture::N; i++) {
    product += this->values1[i] * this->values2[i];
  }

  TypeParam point1(this->values1);
  TypeParam point2(this->values2);
  T dot = point1.dot(point2);

  EXPECT_EQ(dot, product);
}

TYPED_TEST_P(PointTest, Conversion)
{
  constexpr int N = TestFixture::N;

  using PointUnsigned = Point<N, unsigned>;
  PointUnsigned point_unsigned;
  for(int i = 0; i < N; i++) {
    point_unsigned[i] = 2u;
  }

  using PointInt = Point<N, int>;
  PointInt point_int = point_unsigned;

  for(int i = 0; i < N; i++) {
    EXPECT_EQ(point_int[i], static_cast<int>(point_unsigned[i]));
  }
}

// Register the Dot test
REGISTER_TYPED_TEST_SUITE_P(PointTest, BaseAccess, Equality, Dot, Zeroes, Ones, Add,
                            Subtract, Multiply, Divide, Modulo, Conversion);

template <typename T, int... Ns>
auto GeneratePointTypes(std::integer_sequence<int, Ns...>)
{
  return ::testing::Types<Realm::Point<Ns + 1, T>...>{};
}

using TestTypesInt =
    decltype(GeneratePointTypes<int>(std::make_integer_sequence<int, REALM_MAX_DIM>{}));
using TestTypesLongLong = decltype(GeneratePointTypes<long long>(
    std::make_integer_sequence<int, REALM_MAX_DIM>{}));

INSTANTIATE_TYPED_TEST_SUITE_P(IntInstantiation, PointTest, TestTypesInt);
INSTANTIATE_TYPED_TEST_SUITE_P(LongLongInstantiation, PointTest, TestTypesLongLong);
