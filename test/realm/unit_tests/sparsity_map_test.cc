#include <gtest/gtest.h>
#include "realm/deppart/sparsity_impl.h"
#include "realm/transfer/transfer.h"
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
  for(const NodeID node : subscribers) {
    impl->record_remote_contributor(node);
  }

  removed = subscribers;
  for(const NodeID node : subscribers) {
    wrapper->unsubscribe(node);
    removed.remove(node);
    if(removed.size() <= 1) {
      break;
    }
  }

  for(const NodeID node : removed) {
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
  for(const NodeID node : subscribers) {
    impl->record_remote_contributor(node);
  }
  wrapper->remove_references(2, Event::NO_EVENT);

  ASSERT_NE(impl, nullptr);
  for(const NodeID node : subscribers) {
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
  constexpr int count = 2;
  std::vector<Rect<1>> rect_list{Rect<1>(Point<1>(0), Point<1>(1))};
  MockSparsityMapCommunicator<1> *sparsity_comm = new MockSparsityMapCommunicator<1>();
  SparsityMap<1> handle = (ID::make_sparsity(1, 1, 0)).convert<SparsityMap<1>>();
  std::unique_ptr<SparsityMapImpl<1, int>> impl =
      std::make_unique<SparsityMapImpl<1, int>>(
          handle, subscribers,
          reinterpret_cast<SparsityMapCommunicator<1, int> *>(sparsity_comm));

  impl->set_contributor_count(count);

  ASSERT_EQ(sparsity_comm->sent_contributions, 1);
  ASSERT_EQ(sparsity_comm->sent_piece_count, count);
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

  std::vector<SparsityMapEntry<N, T>> entries = public_impl->get_entries();
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
  std::vector<SparsityMapEntry<N, T>> entries = public_impl->get_entries();
  ASSERT_EQ(entries.size(), rect_list.size());
  for(size_t i = 0; i < entries.size(); i++) {
    ASSERT_EQ(entries[i].bounds.lo, rect_list[i].lo);
    ASSERT_EQ(entries[i].bounds.hi, rect_list[i].hi);
  }
}

REGISTER_TYPED_TEST_SUITE_P(SparsityMapImplTest, RemoteDataReply,
                            ContributeDenseRectListRemote, ContributeDenseNotDisjoint,
                            ContributeDenseDisjointRects);

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

template <int N, typename T>
static RegionInstanceImpl *create_inst(Rect<N, T> bounds,
                                       const std::vector<int> &dim_order,
                                       size_t bytes_per_element = 8)
{
  RegionInstance inst =
      ID::make_instance(/*owner=*/0, /*creator=*/0, /*idx=*/0, /*inst_idx=*/0)
          .convert<RegionInstance>();
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, dim_order, bytes_per_element);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  NodeSet ns;
  impl->metadata.mark_valid(ns);
  return impl;
}

template <int N>
struct ComputeCoveringTestData {
  Rect<N> bounds;
  std::vector<Rect<N>> rects;
  bool disjoint;
  size_t max_rects;
  int max_overhead;
  std::vector<Rect<N>> expected;
  bool status = true;
};

struct BaseTestData {
  virtual ~BaseTestData() = default;
  virtual int get_dim() const = 0;
};

template <int N>
struct WrappedComputeCoveringData : public BaseTestData {
  ComputeCoveringTestData<N> data;
  explicit WrappedComputeCoveringData(ComputeCoveringTestData<N> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class ComputeCoveringTest : public ::testing::TestWithParam<BaseTestData *> {
protected:
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_test_case(const ComputeCoveringTestData<N> &test_case)
{
  using T = int;
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

TEST_P(ComputeCoveringTest, Base)
{
  const BaseTestData *base_test_case = GetParam();

  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        constexpr int N = Dim;
        auto &test_case =
            static_cast<const WrappedComputeCoveringData<N> *>(base_test_case)->data;
        run_test_case(test_case);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(
    ComputeCoveringCases, ComputeCoveringTest,
    ::testing::Values(
        // Case 1: All points are mergeable
        new WrappedComputeCoveringData<1>({// Case: Empty input (no rects)
                                           /*bounds=*/Rect<1>{{1}, {0}},
                                           /*rects=*/{},
                                           /*disjoint=*/true,
                                           /*max_rects=*/2,
                                           /*max_overhead=*/10,
                                           /*expected=*/{}, true}),

        new WrappedComputeCoveringData<1>(
            {// Case: Max overhead = -1 (Unlimited overhead, should always return a valid
             // covering)
             /*bounds=*/Rect<1>{{0}, {5}},
             /*rects=*/{Rect<1>{{1}, {2}}, Rect<1>{{3}, {4}}},
             /*disjoint=*/true,
             /*max_rects=*/2,
             /*max_overhead=*/-1,
             /*expected=*/{Rect<1>{{1}, {4}}}}),

        new WrappedComputeCoveringData<1>(
            {// Case : compute covering zero max overhead
             /*bounds=*/Rect<1>{{0}, {6}},
             /*rects=*/{Rect<1>{{0}, {0}}, Rect<1>{{2}, {2}}, Rect<1>{{4}, {4}}},
             /*disjoint=*/true,
             /*max_rects=*/2,
             /*max_overhead=*/0,
             /*expected=*/{},
             /*status=*/false}),

        new WrappedComputeCoveringData<1>(
            {// Case: All rects exactly fit within bounds (disjoint)
             /*bounds=*/Rect<1>{{0}, {5}},
             /*rects=*/{Rect<1>{{0}, {0}}, Rect<1>{{3}, {3}}, Rect<1>{{5}, {5}}},
             /*disjoint=*/true,
             /*max_rects=*/3,
             /*max_overhead=*/10,
             /*expected=*/{Rect<1>{{0}, {0}}, Rect<1>{{3}, {3}}, Rect<1>{{5}, {5}}}}),

        new WrappedComputeCoveringData<1>(
            {// Case: Single rect much larger than bounds (Should be cropped correctly)
             /*bounds=*/Rect<1>{{1}, {3}},
             /*rects=*/{Rect<1>{{0}, {5}}},
             /*disjoint=*/true,
             /*max_rects=*/1,
             /*max_overhead=*/10,
             /*expected=*/{Rect<1>{{1}, {3}}}}),

        new WrappedComputeCoveringData<1>(
            {// Case: Two rects touching (hi[j] + 1 == lo[j+1], should merge)
             /*bounds=*/Rect<1>{{0}, {5}},
             /*rects=*/{Rect<1>{{0}, {2}}, Rect<1>{{3}, {5}}},
             /*disjoint=*/true,
             /*max_rects=*/1,
             /*max_overhead=*/10,
             /*expected=*/{Rect<1>{{0}, {5}}}}),

        new WrappedComputeCoveringData<2>({// Case: Empty input (no rects)
                                           /*bounds=*/Rect<2>{{1, 1}, {0, 0}},
                                           /*rects=*/{},
                                           /*disjoint=*/true,
                                           /*max_rects=*/2,
                                           /*max_overhead=*/10,
                                           /*expected=*/{}}),

        new WrappedComputeCoveringData<2>(
            {// Case: Max overhead = -1 (Should always return a valid covering)
             /*bounds=*/Rect<2>{{0, 0}, {4, 4}},
             /*rects=*/{Rect<2>{{0, 0}, {1, 1}}, Rect<2>{{2, 2}, {3, 3}}},
             /*disjoint=*/true,
             /*max_rects=*/2,
             /*max_overhead=*/-1,
             /*expected=*/{Rect<2>{{0, 0}, {1, 1}}, Rect<2>{{2, 2}, {3, 3}}}}),

        new WrappedComputeCoveringData<2>(
            {// Case: Rectangles exactly fit bounds
             /*bounds=*/Rect<2>{{0, 0}, {4, 4}},
             /*rects=*/{Rect<2>{{0, 0}, {2, 2}}, Rect<2>{{3, 3}, {4, 4}}},
             /*disjoint=*/true,
             /*max_rects=*/2,
             /*max_overhead=*/10,
             /*expected=*/{Rect<2>{{0, 0}, {2, 2}}, Rect<2>{{3, 3}, {4, 4}}}}),

        new WrappedComputeCoveringData<2>({// Case: Large rectangle covering bounds
                                           /*bounds=*/Rect<2>{{1, 1}, {3, 3}},
                                           /*rects=*/{Rect<2>{{0, 0}, {4, 4}}},
                                           /*disjoint=*/true,
                                           /*max_rects=*/1,
                                           /*max_overhead=*/10,
                                           /*expected=*/{Rect<2>{{1, 1}, {3, 3}}}}),

        new WrappedComputeCoveringData<3>({// Case: Empty input (no rects)
                                           /*bounds=*/Rect<3>{{1, 1, 1}, {0, 0, 0}},
                                           /*rects=*/{},
                                           /*disjoint=*/true,
                                           /*max_rects=*/2,
                                           /*max_overhead=*/10,
                                           /*expected=*/{}}),

        new WrappedComputeCoveringData<3>(
            {// Case: Max overhead = -1 (Should always return a valid covering)
             /*bounds=*/Rect<3>{{0, 0, 0}, {4, 4, 4}},
             /*rects=*/{Rect<3>{{0, 0, 0}, {1, 1, 1}}, Rect<3>{{2, 2, 2}, {3, 3, 3}}},
             /*disjoint=*/true,
             /*max_rects=*/2,
             /*max_overhead=*/-1,
             /*expected=*/
             {Rect<3>{{0, 0, 0}, {1, 1, 1}}, Rect<3>{{2, 2, 2}, {3, 3, 3}}}}),

        new WrappedComputeCoveringData<3>(
            {// Case: Rectangles exactly fit bounds
             /*bounds=*/Rect<3>{{0, 0, 0}, {4, 4, 4}},
             /*rects=*/{Rect<3>{{0, 0, 0}, {2, 2, 2}}, Rect<3>{{3, 3, 3}, {4, 4, 4}}},
             /*disjoint=*/true,
             /*max_rects=*/2,
             /*max_overhead=*/10,
             /*expected=*/
             {Rect<3>{{0, 0, 0}, {2, 2, 2}}, Rect<3>{{3, 3, 3}, {4, 4, 4}}}}),

        new WrappedComputeCoveringData<3>(
            {// Case: Large rectangle covering bounds
             /*bounds=*/Rect<3>{{1, 1, 1}, {3, 3, 3}},
             /*rects=*/{Rect<3>{{0, 0, 0}, {4, 4, 4}}},
             /*disjoint=*/true,
             /*max_rects=*/1,
             /*max_overhead=*/10,
             /*expected=*/{Rect<3>{{1, 1, 1}, {3, 3, 3}}}})));

template <int N>
struct OverlapTestData {
  Rect<N> bounds;
  std::vector<Rect<N>> rects1;
  std::vector<Rect<N>> rects2;
  bool approx;
  bool expected;
};

template <int N>
struct WrappedOverlapTestData : public BaseTestData {
  OverlapTestData<N> data;
  explicit WrappedOverlapTestData(OverlapTestData<N> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class OverlapTest : public ::testing::TestWithParam<BaseTestData *> {
protected:
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_test_case(const OverlapTestData<N> &test_case)
{
  using T = int;
  SparsityMap<N, T> handle1 = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  SparsityMap<N, T> handle2 = (ID::make_sparsity(0, 0, 1)).convert<SparsityMap<N, T>>();

  NodeSet subscribers;

  std::unique_ptr<SparsityMapImpl<N, T>> impl1 =
      std::make_unique<SparsityMapImpl<N, T>>(handle1, subscribers);
  std::unique_ptr<SparsityMapImpl<N, T>> impl2 =
      std::make_unique<SparsityMapImpl<N, T>>(handle2, subscribers);

  SparsityMapPublicImpl<N, T> *public_impl1 = impl1.get();
  SparsityMapPublicImpl<N, T> *public_impl2 = impl2.get();

  impl1->contribute_dense_rect_list(test_case.rects1, /*disjoint=*/true);
  impl2->contribute_dense_rect_list(test_case.rects2, /*disjoint=*/true);
  impl1->set_contributor_count(1);
  impl2->set_contributor_count(1);
  impl1->contribute_nothing();
  impl2->contribute_nothing();

  bool overlaps =
      public_impl1->overlaps(public_impl2, test_case.bounds, test_case.approx);

  ASSERT_EQ(overlaps, test_case.expected);
}

TEST_P(OverlapTest, Base)
{
  const BaseTestData *base_test_case = GetParam();

  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        constexpr int N = Dim;
        auto &test_case =
            static_cast<const WrappedOverlapTestData<N> *>(base_test_case)->data;
        run_test_case(test_case);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(
    OverlapCases, OverlapTest,
    ::testing::Values(

        new WrappedOverlapTestData<1>({// Case: No overlap (disjoint regions)
                                       /*bounds=*/Rect<1>{{0}, {5}},
                                       /*rects1=*/{Rect<1>{{0}, {2}}},
                                       /*rects2=*/{Rect<1>{{3}, {5}}},
                                       /*approx=*/false,
                                       /*expected=*/false}),

        new WrappedOverlapTestData<1>({// Case: Exact overlap (rects are identical)
                                       /*bounds=*/Rect<1>{{0}, {5}},
                                       /*rects1=*/{Rect<1>{{1}, {4}}},
                                       /*rects2=*/{Rect<1>{{1}, {4}}},
                                       /*approx=*/false,
                                       /*expected=*/true}),

        new WrappedOverlapTestData<1>(
            {// Case: Partial overlap (rects partially intersect)
             /*bounds=*/Rect<1>{{0}, {5}},
             /*rects1=*/{Rect<1>{{1}, {4}}},
             /*rects2=*/{Rect<1>{{3}, {6}}},
             /*approx=*/false,
             /*expected=*/true}),

        new WrappedOverlapTestData<1>({// Case: Overlapping at a single point
                                       /*bounds=*/Rect<1>{{0}, {5}},
                                       /*rects1=*/{Rect<1>{{1}, {3}}},
                                       /*rects2=*/{Rect<1>{{3}, {4}}},
                                       /*approx=*/false,
                                       /*expected=*/true}),

        new WrappedOverlapTestData<1>(
            {// Case: Approximate mode (loose check, should return true)
             /*bounds=*/Rect<1>{{0}, {5}},
             /*rects1=*/{Rect<1>{{1}, {3}}},
             /*rects2=*/{Rect<1>{{3}, {6}}},
             /*approx=*/true,
             /*expected=*/true}),

        new WrappedOverlapTestData<1>({// Case: No overlap even in approximate mode
                                       /*bounds=*/Rect<1>{{0}, {5}},
                                       /*rects1=*/{Rect<1>{{1}, {2}}},
                                       /*rects2=*/{Rect<1>{{4}, {6}}},
                                       /*approx=*/true,
                                       /*expected=*/false}),

        new WrappedOverlapTestData<1>({// Case: Sparse regions with no overlap
                                       /*bounds=*/Rect<1>{{0}, {10}},
                                       /*rects1=*/{Rect<1>{{0}, {1}}, Rect<1>{{3}, {4}}},
                                       /*rects2=*/{Rect<1>{{5}, {6}}, Rect<1>{{8}, {9}}},
                                       /*approx=*/false,
                                       /*expected=*/false}),

        new WrappedOverlapTestData<2>({// Case: No overlap (completely separate regions)
                                       /*bounds=*/Rect<2>{{0, 0}, {5, 5}},
                                       /*rects1=*/{Rect<2>{{0, 0}, {2, 2}}},
                                       /*rects2=*/{Rect<2>{{3, 3}, {5, 5}}},
                                       /*approx=*/false,
                                       /*expected=*/false}),

        new WrappedOverlapTestData<2>({// Case: Full overlap
                                       /*bounds=*/Rect<2>{{0, 0}, {5, 5}},
                                       /*rects1=*/{Rect<2>{{1, 1}, {4, 4}}},
                                       /*rects2=*/{Rect<2>{{1, 1}, {4, 4}}},
                                       /*approx=*/false,
                                       /*expected=*/true}),

        new WrappedOverlapTestData<2>({// Case: Partial overlap
                                       /*bounds=*/Rect<2>{{0, 0}, {5, 5}},
                                       /*rects1=*/{Rect<2>{{1, 1}, {3, 3}}},
                                       /*rects2=*/{Rect<2>{{2, 2}, {4, 4}}},
                                       /*approx=*/false,
                                       /*expected=*/true}),

        new WrappedOverlapTestData<2>({// Case: Overlapping at a single point
                                       /*bounds=*/Rect<2>{{0, 0}, {5, 5}},
                                       /*rects1=*/{Rect<2>{{1, 1}, {3, 3}}},
                                       /*rects2=*/{Rect<2>{{3, 3}, {4, 4}}},
                                       /*approx=*/false,
                                       /*expected=*/true}),

        new WrappedOverlapTestData<2>({// Case: Approximate mode (loose check)
                                       /*bounds=*/Rect<2>{{0, 0}, {5, 5}},
                                       /*rects1=*/{Rect<2>{{1, 1}, {3, 3}}},
                                       /*rects2=*/{Rect<2>{{3, 3}, {6, 6}}},
                                       /*approx=*/true,
                                       /*expected=*/true}),

        new WrappedOverlapTestData<2>({// Case: No overlap with approximation
                                       /*bounds=*/Rect<2>{{0, 0}, {5, 5}},
                                       /*rects1=*/{Rect<2>{{1, 1}, {2, 2}}},
                                       /*rects2=*/{Rect<2>{{4, 4}, {6, 6}}},
                                       /*approx=*/true,
                                       /*expected=*/false}),

        new WrappedOverlapTestData<2>(
            {// Case: Sparse non-overlapping regions
             /*bounds=*/Rect<2>{{0, 0}, {10, 10}},
             /*rects1=*/{Rect<2>{{0, 0}, {1, 1}}, Rect<2>{{3, 3}, {4, 4}}},
             /*rects2=*/{Rect<2>{{5, 5}, {6, 6}}, Rect<2>{{8, 8}, {9, 9}}},
             /*approx=*/false,
             /*expected=*/false})));
