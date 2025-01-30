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

TYPED_TEST_P(IndexSpaceIteratorParamTest, StepSparse)
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
}

TYPED_TEST_P(IndexSpaceIteratorParamTest, StepDense)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;
  Rect<N, T> domain = Rect<N, T>(TypeParam(4), TypeParam(16));
  Rect<N, T> restrictions = Rect<N, T>(TypeParam(4), TypeParam(14));
  std::vector<Rect<N, T>> rects{domain};

  size_t index = 0;
  for(IndexSpaceIterator<N, T> it(domain, restrictions); it.valid; it.step()) {
    ASSERT_TRUE(index < rects.size());
    ASSERT_EQ(it.rect.lo, restrictions.lo);
    ASSERT_EQ(it.rect.hi, restrictions.hi);
    index++;
  }
}

REGISTER_TYPED_TEST_SUITE_P(IndexSpaceIteratorParamTest, StepDense, StepSparse);

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
