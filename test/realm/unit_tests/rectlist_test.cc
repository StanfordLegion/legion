#include "realm/deppart/rectlist.h"
#include <gtest/gtest.h>

using namespace Realm;

// ---------------------------- Base Test Case Structure ----------------------------
struct BaseTestCaseData {
  virtual ~BaseTestCaseData() = default;
  virtual int get_dim() const = 0;
};

template <int N>
struct DenseRectListTestCase {
  size_t max_rects;
  std::vector<Point<N>> points;
  std::vector<Rect<N>> rects;
  std::vector<Rect<N>> expected;
};

template <int N>
struct HybridRectListTestCase {
  size_t max_rects;
  std::vector<Point<N>> points;
  std::vector<Rect<N>> rects;
  std::vector<Rect<N>> expected;
};

template <int N>
struct CoverageCounterTestCase {
  std::vector<Rect<N>> rects;
  std::vector<Point<N>> points;
  size_t expected_count;
};

template <int N>
struct WrappedDenseTestCaseData : public BaseTestCaseData {
  DenseRectListTestCase<N> dense_data;

  explicit WrappedDenseTestCaseData(DenseRectListTestCase<N> d)
    : dense_data(std::move(d))
  {}

  int get_dim() const override { return N; }
};

template <int N>
struct WrappedHybridTestCaseData : public BaseTestCaseData {
  HybridRectListTestCase<N> hybrid_data;

  explicit WrappedHybridTestCaseData(HybridRectListTestCase<N> h)
    : hybrid_data(std::move(h))
  {}

  int get_dim() const override { return N; }
};

template <int N>
struct WrappedCounterTestCaseData : public BaseTestCaseData {
  CoverageCounterTestCase<N> coverage_data;

  explicit WrappedCounterTestCaseData(CoverageCounterTestCase<N> c)
    : coverage_data(std::move(c))
  {}

  int get_dim() const override { return N; }
};

template <typename Func, size_t... Is>
void dispatch_for_dimension(int dim, Func &&func, std::index_sequence<Is...>)
{
  (
      [&] {
        if(dim == static_cast<int>(Is + 1)) {
          func(std::integral_constant<int, Is + 1>{});
        }
      }(),
      ...);
}

// ---------------------------- DenseRectangleList Tests ----------------------------
class DenseRectListTest : public ::testing::TestWithParam<BaseTestCaseData *> {
protected:
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_dense_test_case(const DenseRectListTestCase<N> &test_case)
{
  DenseRectangleList<N, int> rectlist(test_case.max_rects);

  for(const Rect<N> &rect : test_case.rects) {
    rectlist.add_rect(rect);
  }

  for(const Point<N> &point : test_case.points) {
    rectlist.add_point(point);
  }

  EXPECT_EQ(rectlist.rects.size(), test_case.expected.size());
  for(size_t i = 0; i < test_case.expected.size(); i++) {
    EXPECT_EQ(rectlist.rects[i].lo, test_case.expected[i].lo);
    EXPECT_EQ(rectlist.rects[i].hi, test_case.expected[i].hi);
  }
}

TEST_P(DenseRectListTest, Add)
{
  BaseTestCaseData *base_test_case = GetParam();
  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        run_dense_test_case(
            static_cast<const WrappedDenseTestCaseData<Dim.value> *>(base_test_case)
                ->dense_data);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

// ---------------------------- HybridRectangleList Tests ----------------------------
class HybridRectListTest : public ::testing::TestWithParam<BaseTestCaseData *> {
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_hybrid_test_case(const HybridRectListTestCase<N> &test_case)
{
  HybridRectangleList<N, int> rectlist;

  for(const Point<N> &point : test_case.points) {
    rectlist.add_point(point);
  }

  for(const Rect<N> &rect : test_case.rects) {
    rectlist.add_rect(rect);
  }

  std::vector<Rect<N>> rect_vector = rectlist.convert_to_vector();
  EXPECT_EQ(rect_vector.size(), test_case.expected.size());
  for(size_t i = 0; i < test_case.expected.size(); i++) {
    EXPECT_EQ(rect_vector[i].lo, test_case.expected[i].lo);
    EXPECT_EQ(rect_vector[i].hi, test_case.expected[i].hi);
  }
}

TEST_P(HybridRectListTest, Add)
{
  BaseTestCaseData *base_test_case = GetParam();
  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        run_hybrid_test_case(
            static_cast<const WrappedHybridTestCaseData<Dim.value> *>(base_test_case)
                ->hybrid_data);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

// ---------------------------- CoverageCounter Tests ----------------------------
class CoverageCounterTest : public ::testing::TestWithParam<BaseTestCaseData *> {
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_coverage_test_case(const CoverageCounterTestCase<N> &test_case)
{
  CoverageCounter<N, int> counter;

  for(const Rect<N> &rect : test_case.rects) {
    counter.add_rect(rect);
  }
  for(const Point<N> &point : test_case.points) {
    counter.add_point(point);
  }

  EXPECT_EQ(counter.get_count(), test_case.expected_count);
}

TEST_P(CoverageCounterTest, RunTest)
{
  BaseTestCaseData *base_test_case = GetParam();
  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        run_coverage_test_case(
            static_cast<const WrappedCounterTestCaseData<Dim.value> *>(base_test_case)
                ->coverage_data);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(
    DenseRectangleTests, DenseRectListTest,
    ::testing::ValuesIn(std::vector<BaseTestCaseData *>{
        // Empty
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/1,
            /*points=*/{},
            /*rects=*/{},
        }),

        // Zero max rects
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/0,
            /*points=*/{Point<1>(0), Point<1>(1), Point<1>(2)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 2)},
        }),
        // All points are mergeable
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{Point<1>(0), Point<1>(1), Point<1>(2)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 2)},
        }),

        // All points are disjoint
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/3,
            /*points=*/{Point<1>(0), Point<1>(2), Point<1>(4)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
        }),

        // All points are disjoint and limited by max_rects
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{Point<1>(0), Point<1>(2), Point<1>(4)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 2), Rect<1>(4, 4)},
        }),

        // Fully mergeable along x dimension
        new WrappedDenseTestCaseData<2>({
            /*max_rects=*/1,
            /*points=*/{Point<2>(0, 0), Point<2>(0, 1), Point<2>(0, 2)},
            /*rects=*/{},
            /*expected=*/{Rect<2>({0, 0}, {0, 2})},
        }),

        // Disjoint 3D
        new WrappedDenseTestCaseData<3>({
            /*max_rects=*/1,
            /*points=*/{Point<3>(0, 0, 0), Point<3>(2, 2, 2), Point<3>(4, 4, 4)},
            /*rects=*/{},
            /*expected=*/{Rect<3>({0, 0, 0}, {4, 4, 4})},
        }),

        // All rects are mergeable
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(1, 1), Rect<1>(2, 2)},
            /*expected=*/{Rect<1>(0, 2)},

        }),

        // All rects are disjoint
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/3,
            /*points=*/{},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
            /*expected=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},

        }),

        // All rects are disjoint and limited
        new WrappedDenseTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
            /*expected=*/{Rect<1>(0, 2), Rect<1>(4, 4)},

        }),

        // Fully mergable along x dimension
        new WrappedDenseTestCaseData<2>({
            /*max_rects=*/1,
            /*points=*/{},
            /*rects=*/
            {Rect<2>({0, 0}, {0, 0}), Rect<2>({0, 0}, {1, 1}), Rect<2>({0, 0}, {2, 2})},
            /*expected=*/{Rect<2>({0, 0}, {2, 2})},
        }),

        // Disjoint 3D
        new WrappedDenseTestCaseData<3>({
            /*max_rects=*/1,
            /*points=*/{},
            {Rect<3>({0, 0, 0}, {0, 0, 0}), Rect<3>({2, 2, 2}, {2, 2, 2}),
             Rect<3>({4, 4, 4}, {4, 4, 4})},
            /*expected=*/{Rect<3>({0, 0, 0}, {4, 4, 4})},
        }),

    }));

INSTANTIATE_TEST_SUITE_P(
    HybridRectangleTests, HybridRectListTest,
    ::testing::ValuesIn(std::vector<BaseTestCaseData *>{
        // Empty
        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/1,
            /*points=*/{},
            /*rects=*/{},
        }),

        // Zero max rects
        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/0,
            /*points=*/{Point<1>(0), Point<1>(1), Point<1>(2)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 2)},
        }),
        // All points are mergeable
        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{Point<1>(0), Point<1>(1), Point<1>(2)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 2)},
        }),

        // All points are disjoint
        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/3,
            /*points=*/{Point<1>(0), Point<1>(2), Point<1>(4)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
        }),

        // All points are disjoint and limited by max_rects
        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{Point<1>(0), Point<1>(2), Point<1>(4)},
            /*rects=*/{},
            /*expected=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
        }),

        // Fully mergeable along x dimension
        new WrappedHybridTestCaseData<2>({
            /*max_rects=*/1,
            /*points=*/{Point<2>(0, 0), Point<2>(0, 1), Point<2>(0, 2)},
            /*rects=*/{},
            /*expected=*/{Rect<2>({0, 0}, {0, 2})},
        }),

        // Disjoint 3D
        new WrappedHybridTestCaseData<3>(
            {/*max_rects=*/1,
             /*points=*/{Point<3>(0, 0, 0), Point<3>(2, 2, 2), Point<3>(4, 4, 4)},
             /*rects=*/{},
             /*expected=*/
             {Rect<3>({0, 0, 0}, {0, 0, 0}), Rect<3>({2, 2, 2}, {2, 2, 2}),
              Rect<3>({4, 4, 4}, {4, 4, 4})}}),

        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(1, 1), Rect<1>(2, 2)},
            /*expected=*/{Rect<1>(0, 2)},

        }),

        // All rects are disjoint
        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/3,
            /*points=*/{},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
            /*expected=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},

        }),

        // All rects are disjoint and limited
        new WrappedHybridTestCaseData<1>({
            /*max_rects=*/2,
            /*points=*/{},
            /*rects=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},
            /*expected=*/{Rect<1>(0, 0), Rect<1>(2, 2), Rect<1>(4, 4)},

        }),

        // Fully mergable along x dimension
        new WrappedHybridTestCaseData<2>({
            /*max_rects=*/1,
            /*points=*/{},
            /*rects=*/
            {Rect<2>({0, 0}, {0, 0}), Rect<2>({0, 0}, {1, 1}), Rect<2>({0, 0}, {2, 2})},
            /*expected=*/
            {Rect<2>({0, 0}, {0, 0}), Rect<2>({0, 0}, {1, 1}), Rect<2>({0, 0}, {2, 2})},
        }),

        // Disjoint 3D
        new WrappedHybridTestCaseData<3>({
            /*max_rects=*/1,
            /*points=*/{},
            /*rects=*/
            {Rect<3>({0, 0, 0}, {0, 0, 0}), Rect<3>({2, 2, 2}, {2, 2, 2}),
             Rect<3>({4, 4, 4}, {4, 4, 4})},
            /*expected=*/
            {Rect<3>({0, 0, 0}, {0, 0, 0}), Rect<3>({2, 2, 2}, {2, 2, 2}),
             Rect<3>({4, 4, 4}, {4, 4, 4})},
        }),

    }));

INSTANTIATE_TEST_SUITE_P(CounterTests, CoverageCounterTest,
                         ::testing::ValuesIn(std::vector<BaseTestCaseData *>{
                             // Empty
                             new WrappedCounterTestCaseData<1>({
                                 /*points=*/{},
                                 /*rects=*/{},
                             }),

                             // Empty rects
                             new WrappedCounterTestCaseData<1>({
                                 /*rects=*/{Rect<1>(1, 0)},
                                 /*points=*/{},
                                 /*expected=*/0,
                             }),

                             // Normal rects and points
                             new WrappedCounterTestCaseData<1>({
                                 /*rects=*/{Rect<1>(0, 1), Rect<1>(3, 4)},
                                 /*points=*/{Point<1>(0), Point<1>(1)},
                                 /*expected=*/6,
                             }),

                             // Normal rects
                             new WrappedCounterTestCaseData<1>({
                                 /*rects=*/{Rect<1>(0, 1), Rect<1>(3, 4)},
                                 /*points=*/{},
                                 /*expected=*/4,
                             })}));
