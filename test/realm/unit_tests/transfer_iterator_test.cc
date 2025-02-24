#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

template <int N, typename T>
static InstanceLayout<N, T> *create_layout(Rect<N, T> bounds,
                                           const std::vector<FieldID> &field_ids,
                                           const std::vector<size_t> &field_sizes)
{
  InstanceLayout<N, T> *inst_layout = new InstanceLayout<N, T>();

  inst_layout->piece_lists.resize(field_ids.size());

  for(int i = 0; i < field_ids.size(); i++) {
    InstanceLayoutGeneric::FieldLayout field_layout;
    field_layout.list_idx = i;
    field_layout.rel_offset = 0;
    field_layout.size_in_bytes = field_sizes[i];

    AffineLayoutPiece<N, T> *affine_piece = new AffineLayoutPiece<N, T>();
    affine_piece->bounds = bounds;
    affine_piece->offset = 0;
    affine_piece->strides[0] = field_sizes[i];
    size_t mult = affine_piece->strides[0];
    for(int i = 1; i < N; i++) {
      affine_piece->strides[i] = (bounds.hi[i - 1] - bounds.lo[i - 1] + 1) * mult;
      mult *= (bounds.hi[i - 1] - bounds.lo[i - 1] + 1);
    }

    inst_layout->space = bounds;
    inst_layout->fields[field_ids[i]] = field_layout;
    inst_layout->piece_lists[i].pieces.push_back(affine_piece);
  }

  return inst_layout;
}

// TODO(apryakhin@): Move to utils
static inline RegionInstance make_inst(int owner = 0, int creator = 0, int mem_idx = 0,
                                       int inst_idx = 0)
{
  return ID::make_instance(owner, creator, mem_idx, inst_idx).convert<RegionInstance>();
}

template <int N, typename T>
static RegionInstanceImpl *create_inst(Rect<N, T> bounds,
                                       const std::vector<FieldID> &field_ids,
                                       const std::vector<size_t> &field_sizes)
{
  RegionInstance inst = ID::make_instance(0, 0, 0, 0).convert<RegionInstance>();
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, field_ids, field_sizes);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  return impl;
}

struct BaseTrasferItTestCaseData {
  virtual ~BaseTrasferItTestCaseData() = default;
  virtual int get_dim() const = 0;
};

template <int N>
struct TrasferItTestCaseData {
  Rect<N> domain;
  std::vector<Rect<N>> rects;
  std::vector<Rect<N>> expected;
  std::vector<int> dim_order;
  std::vector<FieldID> field_ids;
  std::vector<size_t> field_offsets;
  std::vector<size_t> field_sizes;
};

template <int N>
struct WrappedTrasferItTestCaseData : public BaseTrasferItTestCaseData {
  TrasferItTestCaseData<N> data;
  explicit WrappedTrasferItTestCaseData(TrasferItTestCaseData<N> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class TransferIteratorGetAddressesTest
  : public ::testing::TestWithParam<BaseTrasferItTestCaseData *> {
protected:
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_test_case(const TrasferItTestCaseData<N> &test_case)
{
  using T = int;
  NodeSet subscribers;

  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  std::unique_ptr<SparsityMapImpl<N, T>> impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers);
  SparsityMapPublicImpl<N, T> *local_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(test_case.rects, true);
  IndexSpace<N, T> domain = test_case.domain;

  std::unique_ptr<TransferIteratorIndexSpace<N, T>> it =
      std::make_unique<TransferIteratorIndexSpace<N, T>>(
          test_case.dim_order.data(), test_case.field_ids, test_case.field_offsets,
          test_case.field_sizes,
          create_inst<N, T>(test_case.domain, test_case.field_ids, test_case.field_sizes),
          domain, local_impl);
  const InstanceLayoutPieceBase *nonaffine;
  AddressList addrlist;
  AddressListCursor cursor;

  bool ok = it->get_addresses(addrlist, nonaffine);

  ASSERT_TRUE(ok);
  ASSERT_TRUE(it->done());

  cursor.set_addrlist(&addrlist);
  size_t total_volume = 0;
  for(const auto &rect : test_case.expected) {
    total_volume += rect.volume();
  }

  size_t bytes_pending = 0;
  for(const size_t size : test_case.field_sizes) {
    bytes_pending += total_volume * size;
  }

  ASSERT_EQ(addrlist.bytes_pending(), bytes_pending);

  if(bytes_pending > 0 && cursor.get_dim() == 1) {
    // TODO(apryakhin:@): Find better way to analyze the adddress list
    // ASSERT_EQ(cursor.get_dim(), 1);
    for(const size_t field_size : test_case.field_sizes) {
      for(const auto &rect : test_case.expected) {
        int dim = cursor.get_dim();
        ASSERT_EQ(cursor.remaining(dim - 1), rect.volume() * field_size);
        cursor.advance(dim - 1, cursor.remaining(dim - 1));
      }
    }

    ASSERT_EQ(addrlist.bytes_pending(), 0);
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

TEST_P(TransferIteratorGetAddressesTest, Base)
{
  const BaseTrasferItTestCaseData *base_test_case = GetParam();

  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        constexpr int N = Dim;
        auto &test_case =
            static_cast<const WrappedTrasferItTestCaseData<N> *>(base_test_case)->data;
        run_test_case(test_case);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(
    TransferIteratorGetAddressesCases, TransferIteratorGetAddressesTest,
    ::testing::Values(

        new WrappedTrasferItTestCaseData<1>(

            // Empty 1D domain
            {
                /*domain=*/{Rect<1>(1, 0)},
                /*rects=*/{},
                /*expected=*/{},
                /*dim_order=*/{0},
                /*field_ids=*/{0, 1},
                /*field_offsets=*/{0},
                /*field_sizes=*/{sizeof(int)},
            }),

        new WrappedTrasferItTestCaseData<1>(
            // Dense 1D rects multifield
            {
                /*domain=*/{Rect<1>(0, 14)},
                /*rects=*/{Rect<1>(0, 14)},
                /*expected=*/{Rect<1>(0, 14)},
                /*dim_order=*/{0},
                /*field_ids=*/{0, 1},
                /*field_offsets=*/{0, 0},
                /*field_sizes=*/{sizeof(int), sizeof(long long)},
            }),

        new WrappedTrasferItTestCaseData<1>(
            // Dense 1D rects
            {
                /*domain=*/{Rect<1>(0, 14)},
                /*rects=*/{Rect<1>(0, 14)},
                /*expected=*/{Rect<1>(0, 14)},
                /*dim_order=*/{0},
                /*field_ids=*/{0},
                /*field_offsets=*/{0},
                /*field_sizes=*/{sizeof(int)},
            }),

        new WrappedTrasferItTestCaseData<1>(
            // Sparse 1D rects
            {
                /*domain=*/{Rect<1>(0, 14)},
                /*rects=*/{Rect<1>(2, 4), Rect<1>(6, 8), Rect<1>(10, 12)},
                /*expected=*/{Rect<1>(2, 4), Rect<1>(6, 8), Rect<1>(10, 12)},
                /*dim_order=*/{0},
                /*field_ids=*/{0},
                /*field_offsets=*/{0},
                /*field_sizes=*/{sizeof(int)},
            }),

        new WrappedTrasferItTestCaseData<2>(
            // Full 2D dense
            {
                /*domain=*/Rect<2>({0, 0}, {10, 10}),
                /*rects*/ {Rect<2>({0, 0}, {10, 10})},
                /*expected=*/{Rect<2>({0, 0}, {10, 10})},
                /*dim_order=*/{0, 1},
                /*field_ids=*/{0},
                /*field_offsets=*/{0},
                /*field_sizes=*/{sizeof(int)},
            }),

        new WrappedTrasferItTestCaseData<2>(
            // Full 2D sparse
            {
                /*domain=*/Rect<2>({0, 0}, {10, 10}),
                /*rects*/ {Rect<2>({0, 0}, {2, 2}), Rect<2>({4, 4}, {8, 8})},
                /*expected=*/{Rect<2>({0, 0}, {2, 2}), Rect<2>({4, 4}, {8, 8})},
                /*dim_order=*/{0, 1},
                /*field_ids=*/{0},
                /*field_offsets=*/{0},
                /*field_sizes=*/{sizeof(int)},
            }),

        new WrappedTrasferItTestCaseData<2>(
            // Full 2D dense reverse dims
            {
                /*domain=*/Rect<2>({0, 0}, {10, 10}),
                /*rects*/ {Rect<2>({0, 0}, {10, 10})},
                /*expected=*/{Rect<2>({0, 0}, {10, 10})},
                /*dim_order=*/{1, 0},
                /*field_ids=*/{0},
                /*field_offsets=*/{0},
                /*field_sizes=*/{sizeof(int)},
            }),

        new WrappedTrasferItTestCaseData<3>({
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*expected=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*dim_order=*/{0, 1, 2},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        })));

constexpr static size_t kByteSize = sizeof(int);

struct IteratorStepTestCase {
  TransferIterator *it;
  std::vector<TransferIterator::AddressInfo> infos;
  std::vector<size_t> max_bytes;
  std::vector<size_t> exp_bytes;
  int num_steps;
};

class TransferIteratorStepTest : public ::testing::TestWithParam<IteratorStepTestCase> {};

TEST_P(TransferIteratorStepTest, Base)
{
  IteratorStepTestCase test_case = GetParam();

  for(int i = 0; i < test_case.num_steps; i++) {
    TransferIterator::AddressInfo info;
    size_t bytes = test_case.it->step(test_case.max_bytes[i], info, 0, 0);

    ASSERT_EQ(bytes, test_case.exp_bytes[i]);

    if(!test_case.infos.empty()) {
      ASSERT_EQ(info.base_offset, test_case.infos[i].base_offset);
      ASSERT_EQ(info.bytes_per_chunk, test_case.infos[i].bytes_per_chunk);
      ASSERT_EQ(info.num_lines, test_case.infos[i].num_lines);
      ASSERT_EQ(info.line_stride, test_case.infos[i].line_stride);
      ASSERT_EQ(info.num_planes, test_case.infos[i].num_planes);
    }
  }
}

const static IteratorStepTestCase kIteratorStepTestCases[] = {
    // Case 0: step through 1D layout with 2 elements
    IteratorStepTestCase{
        .it = new TransferIteratorIndexSpace<1, int>(
            0, {0}, {0},
            /*field_sizes=*/{kByteSize},
            create_inst<1, int>(Rect<1, int>(0, 1), {0}, {kByteSize}),
            Rect<1, int>(0, 1)),
        .infos = {TransferIterator::AddressInfo{/*offset=*/0,
                                                /*bytes_per_el=*/kByteSize,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0},
                  TransferIterator::AddressInfo{/*offset=*/kByteSize,
                                                /*bytes_per_el=*/kByteSize,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0}},
        .max_bytes = {kByteSize, kByteSize},
        .exp_bytes = {kByteSize, kByteSize},
        .num_steps = 2,
    },
// Case 1: step through 2D layout with 4 elements
#if REALM_MAX_DIM > 1
    IteratorStepTestCase{
        .it = new TransferIteratorIndexSpace<2, int>(
            0, {0}, {0},
            /*field_sizes=*/{kByteSize},
            create_inst<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(1)), {0},
                                {kByteSize}),
            Rect<2, int>(Point<2, int>(0), Point<2, int>(1))),

        .infos = {TransferIterator::AddressInfo{/*offset=*/0,
                                                /*bytes_per_el=*/kByteSize * 2,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0},
                  TransferIterator::AddressInfo{/*offset=*/kByteSize * 2,
                                                /*bytes_per_el=*/kByteSize * 2,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0}},

        .max_bytes = {kByteSize * 2, kByteSize * 2},
        .exp_bytes = {kByteSize * 2, kByteSize * 2},
        .num_steps = 2,
    },

    // Case 3: Partial steps through 2D layout
    IteratorStepTestCase{
        .it = new TransferIteratorIndexSpace<2, int>(
            0, {0}, {0}, /*field_sizes=*/{kByteSize},
            create_inst<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(3)), {0},
                                {kByteSize}),
            Rect<2, int>(Point<2, int>(0), Point<2, int>(1))),
        .infos = {TransferIterator::AddressInfo{/*offset=*/0,
                                                /*bytes_per_el=*/kByteSize * 2,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0},

                  TransferIterator::AddressInfo{/*offset=*/kByteSize * 4,
                                                /*bytes_per_el=*/kByteSize * 2,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0}},

        .max_bytes = {kByteSize * 4, kByteSize * 4},
        .exp_bytes = {kByteSize * 2, kByteSize * 2},
        .num_steps = 2,
    },

    // Case 4: step through 2D layout at once
    IteratorStepTestCase{
        .it = new TransferIteratorIndexSpace<2, int>(
            0, {0}, {0}, /*field_sizes=*/{kByteSize},
            create_inst<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(1)), {0},
                                {kByteSize}),
            Rect<2, int>(Point<2, int>(0), Point<2, int>(1))),
        .infos = {TransferIterator::AddressInfo{/*offset=*/0,
                                                /*bytes_per_el=*/kByteSize * 4,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0}},
        .max_bytes = {kByteSize * 4},
        .exp_bytes = {kByteSize * 4},
        .num_steps = 1,
    },

#endif

#if REALM_MAX_DIM > 2
    // Case 5: step through 3D layout at once
    IteratorStepTestCase{
        .it = new TransferIteratorIndexSpace<3, int>(
            0, {0}, {0}, /*field_sizes=*/{kByteSize},
            create_inst<3, int>(Rect<3, int>(Point<3, int>(0), Point<3, int>(1)), {0},
                                {kByteSize}),
            Rect<3, int>(Point<3, int>(0), Point<3, int>(1))),
        .infos = {TransferIterator::AddressInfo{/*offset=*/0,
                                                /*bytes_per_el=*/kByteSize * 8,
                                                /*num_lines=*/1,
                                                /*line_stride=*/0,
                                                /*num_planes=*/1,
                                                /*plane_stride=*/0}},
        .max_bytes = {kByteSize * 8},
        .exp_bytes = {kByteSize * 8},
        .num_steps = 1,
    },
#endif

    // Case 6: step with empty rect
    IteratorStepTestCase{.it = new TransferIteratorIndexSpace<1, int>(
                             0, {0}, {0},
                             /*field_sizes=*/{kByteSize},
                             create_inst<1, int>(Rect<1, int>(0, 1), {0}, {kByteSize}),
                             Rect<1, int>::make_empty()),
                         .max_bytes = {0},
                         .exp_bytes = {0},
                         .num_steps = 1},

    // TODO(apryakhin): This currently hits an assert which should be
    // converted into an error.
    // Case 7: step with non-overlapping rectangle
    /*IteratorStepTestCase{.it = new TransferIteratorIndexSpace<1, int>(
                         Rect<1, int>(2, 3),
                         create_inst<1, int>(Rect<1, int>(0, 1), kByteSize), 0, 0, {0},
                         {0},
                         {kByteSize}, 0),
                     .max_bytes = {0},
                     .exp_bytes = {0},
                     .num_steps = 1},*/

    // TODO(apryakhin): Add more test cases
    //
    // 1. Step through multiple fileds
    // 2. Step with inverted dimension order
    // 3. Step through instance layout with multiple affine pieces
};

INSTANTIATE_TEST_SUITE_P(Foo, TransferIteratorStepTest,
                         testing::ValuesIn(kIteratorStepTestCases));
