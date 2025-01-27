#include <gtest/gtest.h>
#include "realm/deppart/sparsity_impl.h"

using namespace Realm;

class MockSparsityWrapperCommunicator : public SparsityWrapperCommunicator {
public:
  virtual void unsubscribe(SparsityMapImplWrapper *impl, NodeID sender, ID id)
  {
    unsubscribers.add(sender);
  }
  NodeSet unsubscribers;
};

// TOOD(apryakhin@): Add test SparsityMapRefCounter

TEST(SparistyMapImplWrapperTest, UnsubscribeWithoutRecycling)
{
  NodeSet subscribers, removed;
  MockSparsityWrapperCommunicator *comm = new MockSparsityWrapperCommunicator();
  SparsityMapImplWrapper *wrapper = new SparsityMapImplWrapper(comm);
  SparsityMap<1, int> handle =
      (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<1, int>>();

  subscribers.add(7);
  subscribers.add(9);
  subscribers.add(11);
  wrapper->init(ID::make_sparsity(0, 0, 0), 0);
  auto impl = wrapper->get_or_create(handle);
  wrapper->add_references(subscribers.size() + 1);
  for(const auto node : subscribers) {
    impl->record_remote_contributor(node);
  }

  removed = subscribers;
  for(const auto node : subscribers) {
    wrapper->unsubscribe(node);
    removed.remove(node);
    if(removed.size() <= 1) {
      break;
    }
  }

  for(const auto node : removed) {
    EXPECT_TRUE(wrapper->subscribers.contains(node));
  }
}

TEST(SparistyMapImplWrapperTest, AddReferences)
{
  constexpr int unsigned count = 2;
  auto wrapper = std::make_unique<SparsityMapImplWrapper>();

  wrapper->add_references(count);

  EXPECT_EQ(wrapper->references.load(), count);
}

TEST(SparistyMapImplWrapperTest, RemoveReferences)
{
  NodeSet subscribers;
  MockSparsityWrapperCommunicator *comm = new MockSparsityWrapperCommunicator();
  SparsityMapImplWrapper *wrapper = new SparsityMapImplWrapper(comm);
  SparsityMap<1, int> handle =
      (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<1, int>>();
  subscribers.add(7);
  subscribers.add(9);

  wrapper->init(ID::make_sparsity(0, 0, 0), 0);
  auto impl = wrapper->get_or_create(handle);
  wrapper->add_references(2);
  for(const auto node : subscribers) {
    impl->record_remote_contributor(node);
  }
  wrapper->remove_references(2, Event::NO_EVENT);

  EXPECT_NE(impl, nullptr);
  for(const auto node : subscribers) {
    EXPECT_TRUE(comm->unsubscribers.contains(node));
  }
}

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
    send_contribute(NodeID(ID(me).sparsity_creator_node()), me, piece_count, total_count,
                    disjoint, data, datalen);
  }

  virtual void send_contribute(NodeID target, SparsityMap<N, T> me, size_t piece_count,
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

template <typename PointType>
struct PointTraits;

template <int N, typename T>
struct PointTraits<Realm::Point<N, T>> {
  static constexpr int DIM = N;
  using value_type = T;
};

template <typename PointType>
class SparsityMapImplTest : public ::testing::Test {
public:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;

  void SetUp() override { sparsity_comm = new MockSparsityMapCommunicator<N, T>(); }

  std::vector<Rect<N, T>> create_rects(int num_rects, int gap, int start = 0)
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
  // constexpr static int num_rects = 3;
  MockSparsityMapCommunicator<N, T> *sparsity_comm;
};

TYPED_TEST_SUITE_P(SparsityMapImplTest);

TYPED_TEST_P(SparsityMapImplTest, AddRemoteWaiter)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  NodeSet subscribers;
  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();
  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);

  bool ok = impl->add_waiter(/*uop=*/nullptr, true);

  EXPECT_TRUE(ok);
  EXPECT_EQ(this->sparsity_comm->sent_requests, 1);
}

TYPED_TEST_P(SparsityMapImplTest, RemoteDataReply)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int gap = 3;
  constexpr size_t num_rects = 3;
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list = this->create_rects(num_rects, gap);
  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);

  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);
  impl->set_contributor_count(1);
  impl->remote_data_reply(/*requestor=*/2, /*reply_precise=*/true,
                          /*reply_approx=*/false);

  EXPECT_EQ(this->sparsity_comm->sent_contributions, num_rects);
  EXPECT_EQ(this->sparsity_comm->sent_piece_count, num_rects);
  EXPECT_EQ(this->sparsity_comm->sent_bytes, num_rects * sizeof(T) * N * 2);
}

TYPED_TEST_P(SparsityMapImplTest, ContributeDenseRectListRemote)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int num_rects = 3, gap = 3;
  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list = this->create_rects(num_rects, gap);
  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();
  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);

  impl->contribute_dense_rect_list(rect_list,
                                   /*disjoint=*/false);

  EXPECT_EQ(this->sparsity_comm->sent_contributions, num_rects);
  EXPECT_EQ(this->sparsity_comm->sent_piece_count, num_rects);
  EXPECT_EQ(this->sparsity_comm->sent_bytes, num_rects * sizeof(T) * N * 2);
}

TYPED_TEST_P(SparsityMapImplTest, SetContributorCountRemote)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;

  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list{Rect<N, T>(TypeParam(0), TypeParam(1))};
  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, subscribers,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(this->sparsity_comm));

  impl->set_contributor_count(2);

  EXPECT_EQ(this->sparsity_comm->sent_contributions, 1);
}

TYPED_TEST_P(SparsityMapImplTest, ContributeNothingRemote)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list{Rect<N, T>(TypeParam(0), TypeParam(1))};
  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();
  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);

  impl->contribute_nothing();

  EXPECT_EQ(this->sparsity_comm->sent_contributions, 1);
}

TYPED_TEST_P(SparsityMapImplTest, ContributeDenseNotDisjoint)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int num_rects = 3, gap = 3;
  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list = this->create_rects(num_rects, gap);
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(rect_list,
                                   /*disjoint=*/false);

  auto entries = public_impl->get_entries();
  EXPECT_TRUE(public_impl->is_valid());
  EXPECT_EQ(entries.size(), num_rects);
  EXPECT_EQ(entries.size(), rect_list.size());
  for(size_t i = 0; i < entries.size(); i++) {
    EXPECT_EQ(entries[i].bounds.lo, rect_list[i].lo);
    EXPECT_EQ(entries[i].bounds.hi, rect_list[i].hi);
  }
}

TYPED_TEST_P(SparsityMapImplTest, ContributeDenseDisjointRects)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int num_rects = 2, gap = 3;
  NodeSet subscribers;
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  std::vector<Rect<N, T>> rect_list = this->create_rects(num_rects, gap);
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, subscribers,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(this->sparsity_comm));

  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);

  SparsityMapPublicImpl<N, T> *public_impl = impl.get();
  auto entries = public_impl->get_entries();
  EXPECT_EQ(entries.size(), rect_list.size());
  for(size_t i = 0; i < entries.size(); i++) {
    EXPECT_EQ(entries[i].bounds.lo, rect_list[i].lo);
    EXPECT_EQ(entries[i].bounds.hi, rect_list[i].hi);
  }
}

TYPED_TEST_P(SparsityMapImplTest, ComputeCoveringForOneRect)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int num_rects = 2, gap = 3, max_overhead = 1000;
  NodeSet subscribers;
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  std::vector<Rect<N, T>> rect_list;
  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();
  Rect<N, T> bounds(TypeParam(0), TypeParam(10));
  std::vector<Rect<N, T>> covering;

  int offset = 0;
  for(int i = 0; i < num_rects; i++) {
    TypeParam lo_point = TypeParam(offset);
    TypeParam hi_point = TypeParam(offset + 1);
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
    offset += gap;
  }

  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);
  impl->set_contributor_count(1);
  bool ok =
      public_impl->compute_covering(bounds, /*max_rects=*/1, max_overhead, covering);

  EXPECT_TRUE(ok);
  EXPECT_EQ(covering.size(), 1);
  EXPECT_EQ(covering.front().lo, TypeParam(0));
  EXPECT_EQ(covering.front().hi, TypeParam(offset - gap + 1));
}

// TODO(apryakhin@): There are possible inputs for ::compute_covering, so consider making
// a standlone paratemerie test if needed.
TYPED_TEST_P(SparsityMapImplTest, ComputeCoveringForNRect)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr size_t num_rects = 4, max_rects = 3, gap = 3, max_overhead = 1000;
  std::vector<Rect<N, T>> rect_list = this->create_rects(num_rects, gap);
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  NodeSet subscribers;
  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();
  Rect<N, T> bounds(TypeParam(0), TypeParam(10));
  std::vector<Rect<N, T>> covering;

  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  bool ok = public_impl->compute_covering(bounds, max_rects, max_overhead, covering);

  EXPECT_TRUE(ok);
  EXPECT_EQ(covering.size(), max_rects);
  EXPECT_EQ(covering[0].lo, TypeParam(0));
  EXPECT_EQ(covering[0].hi, TypeParam(1));
  EXPECT_EQ(covering[1].lo, TypeParam(3));
  EXPECT_EQ(covering[1].hi, TypeParam(4));
}

TYPED_TEST_P(SparsityMapImplTest, ComputeOverlapPassApprox)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  constexpr size_t num_rects = 3;
  constexpr int gap = 3;
  std::vector<Rect<N, T>> rect_list = this->create_rects(num_rects, gap);
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  NodeSet subscribers;

  auto impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  auto *other_sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto other_impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, other_sparsity_comm);
  other_impl->contribute_dense_rect_list(rect_list,
                                         /*disjoint=*/true);
  other_impl->set_contributor_count(1);
  other_impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *other_public_impl = other_impl.get();

  Rect<N, T> bounds(TypeParam(0), TypeParam(10));
  bool ok = public_impl->overlaps(other_public_impl, bounds, /*approx=*/true);

  EXPECT_TRUE(ok);
}

TYPED_TEST_P(SparsityMapImplTest, ComputeOverlapFail)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  constexpr size_t num_rects = 3;
  constexpr int gap = 3;

  std::vector<Rect<N, T>> rect_list = this->create_rects(num_rects, gap);
  std::vector<Rect<N, T>> other_rect_list = this->create_rects(num_rects, gap, 100);
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  NodeSet node;

  auto impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(this->sparsity_comm));
  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  auto *other_sparsity_comm = new MockSparsityMapCommunicator<N, T>();
  auto other_impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(other_sparsity_comm));
  other_impl->contribute_raw_rects(other_rect_list.data(), other_rect_list.size(), 0,
                                   /*disjoint=*/true, 0);
  other_impl->set_contributor_count(1);
  other_impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *other_public_impl = other_impl.get();

  Rect<N, T> bounds(TypeParam(0), TypeParam(1000));
  bool ok = public_impl->overlaps(other_public_impl, bounds, /*approx=*/false);

  EXPECT_FALSE(ok);
}

REGISTER_TYPED_TEST_SUITE_P(SparsityMapImplTest, AddRemoteWaiter, RemoteDataReply,
                            ContributeDenseRectListRemote, ContributeDenseNotDisjoint,
                            ContributeDenseDisjointRects, SetContributorCountRemote,
                            ContributeNothingRemote, ComputeCoveringForOneRect,
                            ComputeCoveringForNRect, ComputeOverlapPassApprox,
                            ComputeOverlapFail);

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

#define INSTANTIATE_TEST_TYPES(BASE_TYPE, SUFFIX, FIXTURE)                               \
  using N##SUFFIX = decltype(GeneratePointTypesForAllDims<BASE_TYPE>());                 \
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, FIXTURE, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int, SparsityMapImplTest);
INSTANTIATE_TEST_TYPES(long long, LongLong, SparsityMapImplTest);
