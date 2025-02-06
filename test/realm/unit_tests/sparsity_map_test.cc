#include <gtest/gtest.h>
#include "realm/deppart/sparsity_impl.h"
#include "realm/inst_impl.h"

using namespace Realm;

class MockSparsityWrapperCommunicator : public SparsityWrapperCommunicator {
public:
  void unsubscribe(SparsityMapImplWrapper *impl, NodeID sender, ID id) override
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
  std::unique_ptr<SparsityMapImplWrapper> wrapper =
      std::make_unique<SparsityMapImplWrapper>(comm, /*report_leaks=*/false);
  SparsityMap<1, int> handle =
      (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<1, int>>();

  subscribers.add(7);
  subscribers.add(9);
  subscribers.add(11);
  wrapper->init(ID::make_sparsity(0, 0, 0), 0);
  SparsityMapImpl<1, int> *impl = wrapper->get_or_create(handle);
  wrapper->add_references(subscribers.size() + 1);
  for(const auto &node : subscribers) {
    impl->record_remote_contributor(node);
  }

  removed = subscribers;
  for(const auto &node : subscribers) {
    wrapper->unsubscribe(node);
    removed.remove(node);
    if(removed.size() <= 1) {
      break;
    }
  }

  for(const auto node : removed) {
    ASSERT_TRUE(wrapper->subscribers.contains(node));
  }
}

TEST(SparistyMapImplWrapperTest, AddReferences)
{
  constexpr int unsigned count = 2;
  std::unique_ptr<SparsityMapImplWrapper> wrapper =
      std::make_unique<SparsityMapImplWrapper>(/*comm=*/nullptr, /*report_leaks=*/false);

  wrapper->add_references(count);

  ASSERT_EQ(wrapper->references.load(), count);
}

TEST(SparistyMapImplWrapperTest, RemoveReferences)
{
  NodeSet subscribers;
  MockSparsityWrapperCommunicator *comm = new MockSparsityWrapperCommunicator();
  std::unique_ptr<SparsityMapImplWrapper> wrapper =
      std::make_unique<SparsityMapImplWrapper>(comm, /*report_leaks=*/false);
  SparsityMap<1, int> handle =
      (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<1, int>>();
  subscribers.add(7);
  subscribers.add(9);

  wrapper->init(ID::make_sparsity(0, 0, 0), 0);
  SparsityMapImpl<1, int> *impl = wrapper->get_or_create(handle);
  wrapper->add_references(2);
  for(const auto node : subscribers) {
    impl->record_remote_contributor(node);
  }
  wrapper->remove_references(2, Event::NO_EVENT);

  ASSERT_NE(impl, nullptr);
  for(const auto &node : subscribers) {
    ASSERT_TRUE(comm->unsubscribers.contains(node));
  }
}

template <int N, typename T = int>
class MockSparsityMapCommunicator : public SparsityMapCommunicator<N, T> {
public:
  virtual ~MockSparsityMapCommunicator() = default;

  void send_request(SparsityMap<N, T> me, bool request_precise,
                    bool request_approx) override
  {
    sent_requests++;
  }

  void send_contribute(SparsityMap<N, T> me, size_t piece_count, size_t total_count,
                       bool disjoint, const void *data, size_t datalen) override
  {
    send_contribute(NodeID(ID(me).sparsity_creator_node()), me, piece_count, total_count,
                    disjoint, data, datalen);
  }

  void send_contribute(NodeID target, SparsityMap<N, T> me, size_t piece_count,
                       size_t total_count, bool disjoint, const void *data,
                       size_t datalen) override
  {
    sent_contributions++;
    sent_piece_count += piece_count;
    sent_bytes += datalen;
  }

  size_t recommend_max_payload(NodeID owner, bool with_congestion) override
  {
    return sizeof(Rect<N, T>);
  }

  int sent_requests = 0;
  int sent_contributions = 0;
  size_t sent_piece_count = 0;
  size_t sent_bytes = 0;
};

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

TEST(SparsityMapImplTest, AddRemoteWaiter)
{
  NodeSet subscribers;
  SparsityMap<1> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<1>>();
  MockSparsityMapCommunicator<1> *sparsity_comm = new MockSparsityMapCommunicator<1>();

  std::unique_ptr<SparsityMapImpl<1, int>> impl =
      std::make_unique<SparsityMapImpl<1, int>>(handle, subscribers, sparsity_comm);

  bool ok = impl->add_waiter(/*uop=*/nullptr, true);

  ASSERT_TRUE(ok);
  ASSERT_EQ(sparsity_comm->sent_requests, 1);
}

TEST(SparsityMapImplTest, SetContributorCountRemote)
{
  NodeSet subscribers;
  std::vector<Rect<1>> rect_list{Rect<1>(Point<1>(0), Point<1>(1))};
  MockSparsityMapCommunicator<1> *sparsity_comm = new MockSparsityMapCommunicator<1>();
  SparsityMap<1> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<1>>();
  std::unique_ptr<SparsityMapImpl<1, int>> impl =
      std::make_unique<SparsityMapImpl<1, int>>(
          handle, subscribers,
          reinterpret_cast<SparsityMapCommunicator<1, int> *>(sparsity_comm));

  impl->set_contributor_count(2);

  ASSERT_EQ(sparsity_comm->sent_contributions, 1);
}

TEST(SparsityMapImplTest, ContributeNothingRemote)
{
  NodeSet subscribers;
  std::vector<Rect<1>> rect_list{Rect<1>(Point<1>(0), Point<1>(1))};
  MockSparsityMapCommunicator<1> *sparsity_comm = new MockSparsityMapCommunicator<1>();
  SparsityMap<1> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<1>>();
  std::unique_ptr<SparsityMapImpl<1, int>> impl =
      std::make_unique<SparsityMapImpl<1, int>>(handle, subscribers, sparsity_comm);

  impl->contribute_nothing();

  ASSERT_EQ(sparsity_comm->sent_contributions, 1);
}

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

  // constexpr static int num_rects = 3;
  MockSparsityMapCommunicator<N, T> *sparsity_comm;
};

TYPED_TEST_SUITE_P(SparsityMapImplTest);

TYPED_TEST_P(SparsityMapImplTest, RemoteDataReply)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int gap = 3;
  constexpr size_t num_rects = 3;
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list = create_rects<N, T>(num_rects, gap);
  std::unique_ptr<SparsityMapImpl<N, T>> impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);

  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);
  impl->set_contributor_count(1);
  impl->remote_data_reply(/*requestor=*/2, /*reply_precise=*/true,
                          /*reply_approx=*/false);

  ASSERT_EQ(this->sparsity_comm->sent_contributions, num_rects);
  ASSERT_EQ(this->sparsity_comm->sent_piece_count, num_rects);
  ASSERT_EQ(this->sparsity_comm->sent_bytes, num_rects * sizeof(T) * N * 2);
}

TYPED_TEST_P(SparsityMapImplTest, ContributeDenseRectListRemote)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int num_rects = 3, gap = 3;
  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list = create_rects<N, T>(num_rects, gap);
  SparsityMap<N, T> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<N, T>>();
  std::unique_ptr<SparsityMapImpl<N, T>> impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);

  impl->contribute_dense_rect_list(rect_list,
                                   /*disjoint=*/false);

  ASSERT_EQ(this->sparsity_comm->sent_contributions, num_rects);
  ASSERT_EQ(this->sparsity_comm->sent_piece_count, num_rects);
  ASSERT_EQ(this->sparsity_comm->sent_bytes, num_rects * sizeof(T) * N * 2);
}

TYPED_TEST_P(SparsityMapImplTest, ContributeDenseNotDisjoint)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int num_rects = 3, gap = 3;
  NodeSet subscribers;
  std::vector<Rect<N, T>> rect_list = create_rects<N, T>(num_rects, gap);
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  std::unique_ptr<SparsityMapImpl<N, T>> impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(rect_list,
                                   /*disjoint=*/false);

  auto entries = public_impl->get_entries();
  ASSERT_TRUE(public_impl->is_valid());
  ASSERT_EQ(entries.size(), num_rects);
  ASSERT_EQ(entries.size(), rect_list.size());
  for(size_t i = 0; i < entries.size(); i++) {
    ASSERT_EQ(entries[i].bounds.lo, rect_list[i].lo);
    ASSERT_EQ(entries[i].bounds.hi, rect_list[i].hi);
  }
}

TYPED_TEST_P(SparsityMapImplTest, ContributeDenseDisjointRects)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int num_rects = 2, gap = 3;
  NodeSet subscribers;
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  std::vector<Rect<N, T>> rect_list = create_rects<N, T>(num_rects, gap);
  std::unique_ptr<SparsityMapImpl<N, T>> impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, subscribers,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(this->sparsity_comm));

  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);

  SparsityMapPublicImpl<N, T> *public_impl = impl.get();
  auto entries = public_impl->get_entries();
  ASSERT_EQ(entries.size(), rect_list.size());
  for(size_t i = 0; i < entries.size(); i++) {
    ASSERT_EQ(entries[i].bounds.lo, rect_list[i].lo);
    ASSERT_EQ(entries[i].bounds.hi, rect_list[i].hi);
  }
}

TYPED_TEST_P(SparsityMapImplTest, ComputeOverlapPassApprox)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  constexpr size_t num_rects = 3;
  constexpr int gap = 3;
  std::vector<Rect<N, T>> rect_list = create_rects<N, T>(num_rects, gap);
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  NodeSet subscribers;

  std::unique_ptr<SparsityMapImpl<N, T>> impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, this->sparsity_comm);
  impl->contribute_raw_rects(rect_list.data(), rect_list.size(), 0, /*disjoint=*/true, 0);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  MockSparsityMapCommunicator<N, T> *other_sparsity_comm =
      new MockSparsityMapCommunicator<N, T>();
  std::unique_ptr<SparsityMapImpl<N, T>> other_impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, other_sparsity_comm);
  other_impl->contribute_dense_rect_list(rect_list,
                                         /*disjoint=*/true);
  other_impl->set_contributor_count(1);
  other_impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *other_public_impl = other_impl.get();

  Rect<N, T> bounds(TypeParam(0), TypeParam(10));
  bool ok = public_impl->overlaps(other_public_impl, bounds, /*approx=*/true);

  ASSERT_TRUE(ok);
}

TYPED_TEST_P(SparsityMapImplTest, ComputeOverlapFail)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;

  constexpr size_t num_rects = 3;
  constexpr int gap = 3;

  std::vector<Rect<N, T>> rect_list = create_rects<N, T>(num_rects, gap);
  std::vector<Rect<N, T>> other_rect_list = create_rects<N, T>(num_rects, gap, 100);
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  NodeSet node;

  std::unique_ptr<SparsityMapImpl<N, T>> impl = std::make_unique<SparsityMapImpl<N, T>>(
      handle, node,
      reinterpret_cast<SparsityMapCommunicator<N, T> *>(this->sparsity_comm));
  impl->contribute_dense_rect_list(rect_list, /*disjoint=*/true);
  impl->set_contributor_count(1);
  impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  MockSparsityMapCommunicator<N, T> *other_sparsity_comm =
      new MockSparsityMapCommunicator<N, T>();
  std::unique_ptr<SparsityMapImpl<N, T>> other_impl =
      std::make_unique<SparsityMapImpl<N, T>>(
          handle, node,
          reinterpret_cast<SparsityMapCommunicator<N, T> *>(other_sparsity_comm));
  other_impl->contribute_raw_rects(other_rect_list.data(), other_rect_list.size(), 0,
                                   /*disjoint=*/true, 0);
  other_impl->set_contributor_count(1);
  other_impl->contribute_nothing();
  SparsityMapPublicImpl<N, T> *other_public_impl = other_impl.get();

  Rect<N, T> bounds(TypeParam(0), TypeParam(1000));
  bool ok = public_impl->overlaps(other_public_impl, bounds, /*approx=*/false);

  ASSERT_FALSE(ok);
}

REGISTER_TYPED_TEST_SUITE_P(SparsityMapImplTest, RemoteDataReply,
                            ContributeDenseRectListRemote, ContributeDenseNotDisjoint,
                            ContributeDenseDisjointRects, ComputeOverlapPassApprox,
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

template <int N, typename T>
static InstanceLayout<N, T> *create_layout(Rect<N, T> bounds,
                                           const std::vector<int> &dim_order,
                                           size_t bytes_per_element = 8)
{
  InstanceLayout<N, T> *inst_layout = new InstanceLayout<N, T>();
  InstanceLayoutGeneric::FieldLayout field_layout;
  field_layout.list_idx = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = bytes_per_element;

  AffineLayoutPiece<N, T> *affine_piece = new AffineLayoutPiece<N, T>();
  affine_piece->bounds = bounds;
  affine_piece->offset = 0;
  affine_piece->strides[dim_order[0]] = bytes_per_element;
  size_t mult = affine_piece->strides[dim_order[0]];
  for(int i = 1; i < N; i++) {
    int d = dim_order[i];
    affine_piece->strides[d] = (bounds.hi[i - 1] - bounds.lo[i - 1] + 1) * mult;
    mult *= (bounds.hi[i - 1] - bounds.lo[i - 1] + 1);
  }

  inst_layout->space = bounds;
  inst_layout->fields[0] = field_layout;
  inst_layout->piece_lists.resize(1);
  inst_layout->piece_lists[0].pieces.push_back(affine_piece);

  return inst_layout;
}

// TODO(apryakhin@): Move to utils
static inline RegionInstance make_inst(int owner = 0, int creator = 0, int mem_idx = 0,
                                       int inst_idx = 0)
{
  return ID::make_instance(owner, creator, mem_idx, inst_idx).convert<RegionInstance>();
}

template <int N, typename T>
static RegionInstanceImpl *
create_inst(Rect<N, T> bounds, const std::vector<int> &dim_order,
            size_t bytes_per_element = 8, RegionInstance inst = make_inst())
{
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, dim_order, bytes_per_element);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  NodeSet ns;
  impl->metadata.mark_valid(ns);
  return impl;
}

template <int DIM>
struct TypeWrapper {
  static constexpr int value = DIM;
};

template <typename TypeWrapper>
class ComputeCoveringTest : public ::testing::Test {
public:
  static constexpr int DIM = TypeWrapper::value;
};

template <int DIM>
struct ComputeCoveringCase {
  Rect<DIM> bounds;
  std::vector<Rect<DIM>> rects;
  bool disjoint;
  size_t max_rects;
  int max_overhead;
  std::vector<Rect<DIM>> expected;
  bool status = true;
};

template <int DIM>
std::vector<ComputeCoveringCase<DIM>> ComputeCoveringTestCases()
{
  if constexpr(DIM == 1) {
    return {
        {// Case: Empty input (no rects)
         /*bounds=*/Rect<DIM>{{1}, {0}},
         /*rects=*/{},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/10,
         /*expected=*/{}, true},

        {// Case: Max overhead = -1 (Unlimited overhead, should always return a valid
         // covering)
         /*bounds=*/Rect<DIM>{{0}, {5}},
         /*rects=*/{Rect<DIM>{{1}, {2}}, Rect<DIM>{{3}, {4}}},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/-1,
         /*expected=*/{Rect<DIM>{{1}, {4}}}},

        {// Case : compute covering zero max overhead
         /*bounds=*/Rect<DIM>{{0}, {6}},
         /*rects=*/{Rect<DIM>{{0}, {0}}, Rect<DIM>{{2}, {2}}, Rect<DIM>{{4}, {4}}},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/0,
         /*expected=*/{},
         /*status=*/false},

        {// Case: All rects exactly fit within bounds (disjoint)
         /*bounds=*/Rect<DIM>{{0}, {5}},
         /*rects=*/{Rect<DIM>{{0}, {0}}, Rect<DIM>{{3}, {3}}, Rect<DIM>{{5}, {5}}},
         /*disjoint=*/true,
         /*max_rects=*/3,
         /*max_overhead=*/10,
         /*expected=*/{Rect<DIM>{{0}, {0}}, Rect<DIM>{{3}, {3}}, Rect<DIM>{{5}, {5}}}},

        {// Case: Single rect much larger than bounds (Should be cropped correctly)
         /*bounds=*/Rect<DIM>{{1}, {3}},
         /*rects=*/{Rect<DIM>{{0}, {5}}},
         /*disjoint=*/true,
         /*max_rects=*/1,
         /*max_overhead=*/10,
         /*expected=*/{Rect<DIM>{{1}, {3}}}},

        {// Case: Two rects touching (hi[j] + 1 == lo[j+1], should merge)
         /*bounds=*/Rect<DIM>{{0}, {5}},
         /*rects=*/{Rect<DIM>{{0}, {2}}, Rect<DIM>{{3}, {5}}},
         /*disjoint=*/true,
         /*max_rects=*/1,
         /*max_overhead=*/10,
         /*expected=*/{Rect<DIM>{{0}, {5}}}},
    };
  } else if constexpr(DIM == 2) {
    return {
        {// Case: Empty input (no rects)
         /*bounds=*/Rect<DIM>{{1, 1}, {0, 0}},
         /*rects=*/{},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/10,
         /*expected=*/{}},

        {// Case: Max overhead = -1 (Should always return a valid covering)
         /*bounds=*/Rect<DIM>{{0, 0}, {4, 4}},
         /*rects=*/{Rect<DIM>{{0, 0}, {1, 1}}, Rect<DIM>{{2, 2}, {3, 3}}},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/-1,
         /*expected=*/{Rect<DIM>{{0, 0}, {1, 1}}, Rect<DIM>{{2, 2}, {3, 3}}}},

        {// Case: Rectangles exactly fit bounds
         /*bounds=*/Rect<DIM>{{0, 0}, {4, 4}},
         /*rects=*/{Rect<DIM>{{0, 0}, {2, 2}}, Rect<DIM>{{3, 3}, {4, 4}}},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/10,
         /*expected=*/{Rect<DIM>{{0, 0}, {2, 2}}, Rect<DIM>{{3, 3}, {4, 4}}}},

        {// Case: Large rectangle covering bounds
         /*bounds=*/Rect<DIM>{{1, 1}, {3, 3}},
         /*rects=*/{Rect<DIM>{{0, 0}, {4, 4}}},
         /*disjoint=*/true,
         /*max_rects=*/1,
         /*max_overhead=*/10,
         /*expected=*/{Rect<DIM>{{1, 1}, {3, 3}}}},
    };
  } else if constexpr(DIM == 3) {
    return {
        {// Case: Empty input (no rects)
         /*bounds=*/Rect<DIM>{{1, 1, 1}, {0, 0, 0}},
         /*rects=*/{},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/10,
         /*expected=*/{}},

        {// Case: Max overhead = -1 (Should always return a valid covering)
         /*bounds=*/Rect<DIM>{{0, 0, 0}, {4, 4, 4}},
         /*rects=*/{Rect<DIM>{{0, 0, 0}, {1, 1, 1}}, Rect<DIM>{{2, 2, 2}, {3, 3, 3}}},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/-1,
         /*expected=*/{Rect<DIM>{{0, 0, 0}, {1, 1, 1}}, Rect<DIM>{{2, 2, 2}, {3, 3, 3}}}},

        {// Case: Rectangles exactly fit bounds
         /*bounds=*/Rect<DIM>{{0, 0, 0}, {4, 4, 4}},
         /*rects=*/{Rect<DIM>{{0, 0, 0}, {2, 2, 2}}, Rect<DIM>{{3, 3, 3}, {4, 4, 4}}},
         /*disjoint=*/true,
         /*max_rects=*/2,
         /*max_overhead=*/10,
         /*expected=*/{Rect<DIM>{{0, 0, 0}, {2, 2, 2}}, Rect<DIM>{{3, 3, 3}, {4, 4, 4}}}},

        {// Case: Large rectangle covering bounds
         /*bounds=*/Rect<DIM>{{1, 1, 1}, {3, 3, 3}},
         /*rects=*/{Rect<DIM>{{0, 0, 0}, {4, 4, 4}}},
         /*disjoint=*/true,
         /*max_rects=*/1,
         /*max_overhead=*/10,
         /*expected=*/{Rect<DIM>{{1, 1, 1}, {3, 3, 3}}}},
    };
  }
  return {};
}

TYPED_TEST_SUITE_P(ComputeCoveringTest);

TYPED_TEST_P(ComputeCoveringTest, Base)
{
  using T = int;
  constexpr int N = TypeParam::value;

  auto test_cases = ComputeCoveringTestCases<N>();
  for(const auto &test_case : test_cases) {
    MockSparsityMapCommunicator<N, T> *sparsity_comm =
        new MockSparsityMapCommunicator<N, T>();
    SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
    NodeSet subscribers;
    std::unique_ptr<SparsityMapImpl<N, T>> impl =
        std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers, sparsity_comm);
    SparsityMapPublicImpl<N, T> *public_impl = impl.get();

    std::vector<Rect<N, T>> covering;

    impl->contribute_dense_rect_list(test_case.rects, test_case.disjoint);
    impl->set_contributor_count(1);
    impl->contribute_nothing();
    bool ok = public_impl->compute_covering(test_case.bounds, test_case.max_rects,
                                            test_case.max_overhead, covering);
    EXPECT_EQ(ok, test_case.status);
    EXPECT_EQ(covering.size(), test_case.expected.size());
    for(size_t i = 0; i < test_case.expected.size(); i++) {
      EXPECT_EQ(covering[i].lo, test_case.expected[i].lo);
      EXPECT_EQ(covering[i].hi, test_case.expected[i].hi);
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(ComputeCoveringTest, Base);

template <typename Seq>
struct WrapTypes;

template <std::size_t... Ns>
struct WrapTypes<std::index_sequence<Ns...>> {
  using type = ::testing::Types<TypeWrapper<Ns + 1>...>;
};

using TestTypes = typename WrapTypes<std::make_index_sequence<REALM_MAX_DIM>>::type;

INSTANTIATE_TYPED_TEST_SUITE_P(AllDimensions, ComputeCoveringTest, TestTypes);
