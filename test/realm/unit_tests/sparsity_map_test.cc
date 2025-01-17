#include <gtest/gtest.h>
#include "realm/deppart/sparsity_impl.h"

using namespace Realm;

TEST(SparistyMapTest, Construct) {}

template <typename PointType>
struct PointTraits;

template <int N, typename T>
struct PointTraits<Realm::Point<N, T>> {
  static constexpr int DIM = N;
  using value_type = T;
};

template <typename PointType>
class SparsityMapTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;
};

TYPED_TEST_SUITE_P(SparsityMapTest);

// TODO: Implement mock communicator
TYPED_TEST_P(SparsityMapTest, SetContributorCountRemote)
{
  return; // DISABLED
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const int num_rects = 1;

  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(i);
    TypeParam hi_point = TypeParam(i + 1);
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();

  NodeSet node;
  SparsityMapImpl<N, T> *impl = new SparsityMapImpl<N, T>(handle, node);
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), rect_list.size(),
                             /*disjoint=*/false, 0);
  impl->set_contributor_count(1);
}

// TODO: Implement mock communicator
TYPED_TEST_P(SparsityMapTest, ContributeNothingRemote)
{
  return; // DISABLED
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const int num_rects = 1;

  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(i);
    TypeParam hi_point = TypeParam(i + 1);
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();

  NodeSet node;
  SparsityMapImpl<N, T> *impl = new SparsityMapImpl<N, T>(handle, node);
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), rect_list.size(),
                             /*disjoint=*/false, 0);
  impl->contribute_nothing();
}

TYPED_TEST_P(SparsityMapTest, ContributeDenseJointRects)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const int num_rects = 1;

  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(i);
    TypeParam hi_point = TypeParam(i + 1);
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();

  NodeSet node;
  SparsityMapImpl<N, T> *impl = new SparsityMapImpl<N, T>(handle, node);
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), rect_list.size(),
                             /*disjoint=*/false, 0);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl);

  auto entries = public_impl->get_entries();
  EXPECT_EQ(entries.size(), 1);
  EXPECT_EQ(entries.size(), rect_list.size());
  for(size_t i = 0; i < entries.size(); i++) {
    EXPECT_EQ(entries[i].bounds.lo, rect_list[i].lo);
    EXPECT_EQ(entries[i].bounds.hi, rect_list[i].hi);
  }
}

TYPED_TEST_P(SparsityMapTest, ContributeDenseDisjointRects)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const int num_rects = 2;
  const int gap = 3;

  int index = 0;
  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(index);
    TypeParam hi_point = TypeParam(index + 1);
    index += gap;
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  SparsityMap<N, T> handle;
  handle.id = 0;

  NodeSet node;
  SparsityMapImpl<N, T> *impl = new SparsityMapImpl<N, T>(handle, node);
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);

  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl);

  auto entries = public_impl->get_entries();
  EXPECT_EQ(entries.size(), rect_list.size());
  for(size_t i = 0; i < entries.size(); i++) {
    EXPECT_EQ(entries[i].bounds.lo, rect_list[i].lo);
    EXPECT_EQ(entries[i].bounds.hi, rect_list[i].hi);
  }
}

TYPED_TEST_P(SparsityMapTest, ComputeCoveringForOneRect)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const int num_rects = 2;
  const int gap = 3;

  int index = 0;
  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(index);
    TypeParam hi_point = TypeParam(index + 1);
    index += gap;
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  SparsityMap<N, T> handle;
  handle.id = 0;

  NodeSet node;
  SparsityMapImpl<N, T> *impl = new SparsityMapImpl<N, T>(handle, node);
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);

  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl);

  TypeParam lo_point = TypeParam(0);
  TypeParam hi_point = TypeParam(10);
  Rect<N, T> bounds(lo_point, hi_point);
  std::vector<Rect<N, T>> covering;
  bool ok = public_impl->compute_covering(bounds, /*max_rects=*/1, /*max_overhead=*/1000,
                                          covering);

  EXPECT_TRUE(ok);
  EXPECT_EQ(covering.size(), 1);
  EXPECT_EQ(covering.front().lo, TypeParam(0));
  EXPECT_EQ(covering.front().hi, TypeParam(index - gap + 1));
}

REGISTER_TYPED_TEST_SUITE_P(SparsityMapTest, ContributeDenseJointRects,
                            ContributeDenseDisjointRects, SetContributorCountRemote,
                            ContributeNothingRemote, ComputeCoveringForOneRect);

template <typename T, int... Ns>
auto GeneratePointTypes(std::integer_sequence<int, Ns...>)
{
  return ::testing::Types<Realm::Point<Ns + 1, T>...>{};
}

using TestTypesInt =
    decltype(GeneratePointTypes<int>(std::make_integer_sequence<int, REALM_MAX_DIM>{}));
using TestTypesLongLong = decltype(GeneratePointTypes<long long>(
    std::make_integer_sequence<int, REALM_MAX_DIM>{}));

INSTANTIATE_TYPED_TEST_SUITE_P(IntInstantiation, SparsityMapTest, TestTypesInt);
INSTANTIATE_TYPED_TEST_SUITE_P(LongLongInstantiation, SparsityMapTest, TestTypesLongLong);
