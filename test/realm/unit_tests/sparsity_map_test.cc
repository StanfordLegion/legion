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

template <int N, typename T>
class MockSparsityMapCommunicator : public SparsityMapCommunicator<N, T> {
public:
  virtual ~MockSparsityMapCommunicator() = default;

  virtual void send_request(SparsityMap<N, T> me, bool request_precise,
                            bool request_approx)
  {
    sent_requests++;
  }

  virtual void send_contribute(SparsityMap<N, T> me, size_t piece_count,
                               size_t total_count, bool disjoint, const void *data,
                               size_t datalen)
  {
    sent_contributions++;
    sent_piece_count += piece_count;
    sent_bytes += datalen;
  }

  virtual size_t recommend_max_payload(NodeID owner, bool with_congestion)
  {
    return sizeof(Rect<N, T>);
  }

  int sent_requests = 0;
  int sent_contributions = 0;
  size_t sent_piece_count = 0;
  size_t sent_bytes = 0;
};

TYPED_TEST_P(SparsityMapTest, AddRemoteWaiter)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();

  NodeSet node;
  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));

  bool ok = impl->add_waiter(/*uop=*/nullptr, true);

  EXPECT_TRUE(ok);
  EXPECT_EQ(sparsity_comm->sent_requests, 1);
}

TYPED_TEST_P(SparsityMapTest, RemoteDataReply)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const size_t num_rects = 3;
  const size_t max_rects = 4;
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
  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);

  impl->set_contributor_count(1);
  impl->contribute_nothing();
  impl->remote_data_reply(2, true, false);

  EXPECT_EQ(sparsity_comm->sent_contributions, num_rects);
  EXPECT_EQ(sparsity_comm->sent_piece_count, num_rects);
  EXPECT_EQ(sparsity_comm->sent_bytes, num_rects * sizeof(T) * N * 2);
}

TYPED_TEST_P(SparsityMapTest, ContributeDenseRectListRemote)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const int num_rects = 3;

  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(i);
    TypeParam hi_point = TypeParam(i + 1);
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();

  NodeSet node;

  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_dense_rect_list(rect_list,
                                   /*disjoint=*/false);

  EXPECT_EQ(sparsity_comm->sent_contributions, num_rects);
  EXPECT_EQ(sparsity_comm->sent_piece_count, num_rects);
  EXPECT_EQ(sparsity_comm->sent_bytes, num_rects * sizeof(T) * N * 2);
}

TYPED_TEST_P(SparsityMapTest, SetContributorCountRemote)
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

  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();

  NodeSet node;

  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), rect_list.size(),
                             /*disjoint=*/false, 0);
  impl->set_contributor_count(1);

  EXPECT_EQ(sparsity_comm->sent_contributions, 1);
}

TYPED_TEST_P(SparsityMapTest, ContributeNothingRemote)
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

  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();
  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();

  NodeSet node;
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), rect_list.size(),
                             /*disjoint=*/false, 0);
  impl->contribute_nothing();

  EXPECT_EQ(sparsity_comm->sent_contributions, 1);
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

  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));

  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), rect_list.size(),
                             /*disjoint=*/false, 0);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

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
  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);

  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

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
  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);

  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

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

TYPED_TEST_P(SparsityMapTest, ComputeCoveringForNRect)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const size_t num_rects = 3;
  const size_t max_rects = 4;
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
  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);

  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

  TypeParam lo_point = TypeParam(0);
  TypeParam hi_point = TypeParam(10);
  Rect<N, T> bounds(lo_point, hi_point);
  std::vector<Rect<N, T>> covering;
  bool ok =
      public_impl->compute_covering(bounds, max_rects, /*max_overhead=*/1000, covering);

  EXPECT_TRUE(ok);
  EXPECT_EQ(covering.size(), num_rects);
  EXPECT_EQ(covering[0].lo, TypeParam(0));
  EXPECT_EQ(covering[0].hi, TypeParam(1));
  EXPECT_EQ(covering[1].lo, TypeParam(3));
  EXPECT_EQ(covering[1].hi, TypeParam(4));
}

TYPED_TEST_P(SparsityMapTest, ComputeOverlapPassApprox)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  std::vector<Rect<N, T>> rect_list;
  const size_t num_rects = 3;
  const size_t max_rects = 2;
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

  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

  auto *other_sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto other_impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(other_sparsity_comm));
  other_impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0,
                                   /*disjoint=*/true, 0);
  other_impl->set_contributor_count(1);
  other_impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *other_public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

  TypeParam lo_point = TypeParam(0);
  TypeParam hi_point = TypeParam(10);
  Rect<N, T> bounds(lo_point, hi_point);
  bool ok = public_impl->overlaps(other_public_impl, bounds, true);

  EXPECT_TRUE(ok);
}

TYPED_TEST_P(SparsityMapTest, ComputeOverlapFail)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  const size_t num_rects = 3;
  const size_t max_rects = 2;
  const int gap = 3;

  int index = 0;
  std::vector<Rect<N, T>> rect_list;
  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(index);
    TypeParam hi_point = TypeParam(index + 1);
    index += gap;
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  SparsityMap<N, T> handle;
  handle.id = 0;
  NodeSet node;

  auto *sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node, reinterpret_cast<SparsityMapCommunicator<N, T> *>(sparsity_comm));
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

  index = 100;
  std::vector<Rect<N, T>> other_rect_list;
  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(index);
    TypeParam hi_point = TypeParam(index + 1);
    index += gap;
    other_rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }

  auto *other_sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto other_impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(other_sparsity_comm));
  other_impl->contribute_raw_rects(other_rect_list.data(), other_rect_list.size(), 0,
                                   /*disjoint=*/true, 0);
  other_impl->set_contributor_count(1);
  other_impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *other_public_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

  TypeParam lo_point = TypeParam(100);
  TypeParam hi_point = TypeParam(1000);
  Rect<N, T> bounds(lo_point, hi_point);
  bool ok = public_impl->overlaps(other_public_impl, bounds, false);

  EXPECT_FALSE(ok);
}

REGISTER_TYPED_TEST_SUITE_P(SparsityMapTest, AddRemoteWaiter, RemoteDataReply,
                            ContributeDenseRectListRemote, ContributeDenseJointRects,
                            ContributeDenseDisjointRects, SetContributorCountRemote,
                            ContributeNothingRemote, ComputeCoveringForOneRect,
                            ComputeCoveringForNRect, ComputeOverlapPassApprox,
                            ComputeOverlapFail);

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
