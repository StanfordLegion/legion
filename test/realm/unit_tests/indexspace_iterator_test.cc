#include "realm/transfer/transfer.h"
#include <memory>
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
class IndexSpaceIteratorParamTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;
  constexpr static size_t elem_size = 8;
};

TYPED_TEST_SUITE_P(IndexSpaceIteratorParamTest);

template <int N, typename T>
static std::vector<Rect<N, T>> create_rects(int num_rects, int gap, int start = 0)
{
  std::vector<Rect<N, T>> rect_list;
  int index = start;
  for(int i = 0; i < num_rects; i++) {
    Point<N, T> lo_point = Point<N, T>(index);
    Point<N, T> hi_point = Point<N, T>(index + 1);
    index += gap;
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }
  return rect_list;
}

// TODO(apryakhin@): Consider removing or consolidatiing. This covers base cases for all
// dims and below parameterized test covers edge cases.
TYPED_TEST_P(IndexSpaceIteratorParamTest, StepSparseRealmMaxDims)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int gap = 3;
  constexpr size_t num_rects = 3;
  NodeSet subscribers;
  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(16));
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  std::vector<Rect<N, T>> rects = create_rects<N, T>(num_rects, gap);
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers);
  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(rects, /*disjoint=*/true);
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  size_t index = 0;
  for(IndexSpaceIterator<N, T> it(domain, domain, public_impl); it.valid; it.step()) {
    ASSERT_TRUE(index < rects.size());
    ASSERT_EQ(it.rect.lo, rects[index].lo);
    ASSERT_EQ(it.rect.hi, rects[index].hi);
    index++;
  }

  ASSERT_EQ(index, rects.size());
}

REGISTER_TYPED_TEST_SUITE_P(IndexSpaceIteratorParamTest, StepSparseRealmMaxDims);

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
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, IndexSpaceIteratorParamTest, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int);
// TODO(apryakhin@): Consider enabling if needed
// INSTANTIATE_TEST_TYPES(long long, LongLong);

template <int N>
struct IndexSpaceIteratorTestCase {
  Rect<N> domain;
  Rect<N> restrictions;
  std::vector<Rect<N>> rects;
  std::vector<Rect<N>> expected_rects;
};

using IndexSpaceIteratorTestCase1D = IndexSpaceIteratorTestCase<1>;
using IndexSpaceIteratorTestCase2D = IndexSpaceIteratorTestCase<2>;
using IndexSpaceIteratorTestCase3D = IndexSpaceIteratorTestCase<3>;

using IndexSpaceIteratorTestVariant =
    std::variant<IndexSpaceIteratorTestCase1D, IndexSpaceIteratorTestCase2D,
                 IndexSpaceIteratorTestCase3D>;

class IndexSpaceIteratorTest
  : public ::testing::TestWithParam<IndexSpaceIteratorTestVariant> {
protected:
  void TearDown() override {}

  template <int N>
  void RunTest(const IndexSpaceIteratorTestCase<N> &test_case)
  {
    using T = int;
    NodeSet subscribers;
    std::unique_ptr<SparsityMapImpl<N, T>> impl;
    if(!test_case.rects.empty()) {
      SparsityMap<N, T> handle =
          (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
      impl = std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers);
      impl->set_contributor_count(1);
      impl->contribute_dense_rect_list(test_case.rects, true);
    }

    SparsityMapPublicImpl<N, T> *public_impl = impl.get();

    size_t index = 0;
    for(IndexSpaceIterator<N, T> it(test_case.domain, test_case.restrictions,
                                    public_impl);
        it.valid; it.step()) {
      ASSERT_TRUE(index < test_case.expected_rects.size());
      ASSERT_EQ(it.rect.lo, test_case.expected_rects[index].lo);
      ASSERT_EQ(it.rect.hi, test_case.expected_rects[index].hi);
      index++;
    }

    ASSERT_EQ(index, test_case.expected_rects.size());
  }
};

TEST_P(IndexSpaceIteratorTest, NextTargetSubrectEmpty)
{
  std::visit([this](auto &&test_case) { RunTest(test_case); }, GetParam());
}

// 1D test cases
const static IndexSpaceIteratorTestCase1D kIndexSpaceItTestCases1D[] = {
    // Full 1D no rects
    {/*bounds=*/{Rect<1>(0, 10)},
     /*restrictions=*/{Rect<1>(0, 10)},
     /*rects=*/{},
     /*exp_rects=*/{Rect<1>(0, 10)}},

    // Full 1D iteration
    {/*bounds=*/{Rect<1>(0, 10)},
     /*restrictions=*/{Rect<1>(0, 10)},
     /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
     /*exp_rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)}},

    // Full 1D iteration with restrictions
    {/*bounds=*/{Rect<1>(0, 10)},
     /*restrictions=*/{Rect<1>(4, 8)},
     /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
     /*exp_rects=*/{Rect<1>(4, 6), Rect<1>(8, 8)}},

    // 1D Empty bounds
    {/*bounds=*/{Rect<1>(1, 0)},
     /*restrictions=*/{Rect<1>(4, 8)},
     /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
     /*exp_rects=*/{}},

    // 1D empty restrictions
    {/*bounds=*/{Rect<1>(0, 10)},
     /*restrictions=*/{Rect<1>(1, 0)},
     /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
     /*exp_rects=*/{}},
};

std::vector<IndexSpaceIteratorTestVariant> AllItTestCases()
{
  std::vector<IndexSpaceIteratorTestVariant> cases;
  cases.insert(cases.end(), std::begin(kIndexSpaceItTestCases1D),
               std::end(kIndexSpaceItTestCases1D));

  /*#if REALM_MAX_DIM > 1
    cases.insert(cases.end(), std::begin(kIndexSpaceItTestCases2D),
                 std::end(kIndexSpaceItTestCases2D));
  #endif
  #if REALM_MAX_DIM > 2
    cases.insert(cases.end(), std::begin(kIndexSpaceItTestCases3D),
                 std::end(kIndexSpaceItTestCases3D));
  #endif*/

  return cases;
}

INSTANTIATE_TEST_SUITE_P(TestAllDims, IndexSpaceIteratorTest,
                         testing::ValuesIn(AllItTestCases()));
