#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

template <int N, typename T>
static InstanceLayout<N, T> *
create_layout(Rect<N, T> bounds, const std::vector<FieldID> &field_ids,
              const std::vector<size_t> &field_sizes, size_t bytes_per_element = 8)
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
static RegionInstanceImpl *
create_inst(Rect<N, T> bounds, const std::vector<FieldID> &field_ids,
            const std::vector<size_t> &field_sizes, size_t bytes_per_element = 8,
            RegionInstance inst = make_inst())
{
  InstanceLayout<N, T> *inst_layout =
      create_layout(bounds, field_ids, field_sizes, bytes_per_element);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  return impl;
}

template <int DIM>
struct TypeWrapper {
  static constexpr int value = DIM;
};

template <typename TypeWrapper>
class GetAddressesTest : public ::testing::Test {
public:
  static constexpr int DIM = TypeWrapper::value;
};

template <int DIM>
struct GetAddressesTestCase {
  Rect<DIM> domain;
  std::vector<Rect<DIM>> rects;
  std::vector<Rect<DIM>> expected;
  std::vector<int> dim_order;
  std::vector<FieldID> field_ids;
  std::vector<size_t> field_offsets;
  std::vector<size_t> field_sizes;
  size_t elem_size;
};

template <int DIM>
std::vector<GetAddressesTestCase<DIM>> GetTestCases()
{
  if constexpr(DIM == 1) {
    return {
        // Dense 1D rects multifield
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*expected=*/{Rect<1>(0, 14)},
            /*dim_order=*/{0},
            /*field_ids=*/{0, 1},
            /*field_offsets=*/{0, 0},
            /*field_sizes=*/{sizeof(int), sizeof(long long)},
            /*elem_size=*/sizeof(int),
        },

        // Dense 1D rects
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*expected=*/{Rect<1>(0, 14)},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
            /*elem_size=*/sizeof(int),
        },

        // Sparse 1D rects
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(2, 4), Rect<1>(6, 8), Rect<1>(10, 12)},
            /*expected=*/{Rect<1>(2, 4), Rect<1>(6, 8), Rect<1>(10, 12)},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
            /*elem_size=*/sizeof(int),
        },
    };
  } else if constexpr(DIM == 2) {
    return {

        // Full 2D dense
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*rects*/ {Rect<2>({0, 0}, {10, 10})},
            /*expected=*/{Rect<2>({0, 0}, {10, 10})},
            /*dim_order=*/{0, 1},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
            /*elem_size=*/sizeof(int),
        },

        // Full 2D sparse
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*rects*/ {Rect<2>({0, 0}, {2, 2}), Rect<2>({4, 4}, {8, 8})},
            /*expected=*/{Rect<2>({0, 0}, {2, 2}), Rect<2>({4, 4}, {8, 8})},
            /*dim_order=*/{0, 1},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
            /*elem_size=*/sizeof(int),
        },

        // Full 2D dense reverse dims
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*rects*/ {Rect<2>({0, 0}, {10, 10})},
            /*expected=*/{Rect<2>({0, 0}, {10, 10})},
            /*dim_order=*/{1, 0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
            /*elem_size=*/sizeof(int),
        },
    };
  } else if constexpr(DIM == 3) {
    return {
        // Full 3D domain
        {
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*expected=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*dim_order=*/{0, 1, 2},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
            /*elem_size=*/sizeof(int),
        },

    };
  }
  return {};
}

TYPED_TEST_SUITE_P(GetAddressesTest);

TYPED_TEST_P(GetAddressesTest, Base)
{
  using T = int;
  constexpr int N = TypeParam::value;
  auto test_cases = GetTestCases<N>();
  for(const auto &test_case : test_cases) {
    NodeSet subscribers;

    SparsityMapPublicImpl<N, T> *local_impl = nullptr;
    SparsityMap<N, T>::ImplLookup::get_impl_ptr =
        [&local_impl](const SparsityMap<N, T> &map) -> SparsityMapPublicImpl<N, T> * {
      if(local_impl == nullptr) {
        SparsityMap<N, T> handle =
            (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
        NodeSet subscribers;
        local_impl = new SparsityMapImpl<N, T>(handle, subscribers);
      }
      return local_impl;
    };

    SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
    auto impl = reinterpret_cast<SparsityMapImpl<N, T> *>(handle.impl());
    impl->set_contributor_count(1);
    impl->contribute_dense_rect_list(test_case.rects, true);
    IndexSpace<N, T> domain = test_case.domain;
    if(!test_case.rects.empty()) {
      domain.sparsity = handle;
    }

    auto it = std::make_unique<TransferIteratorIndexSpace<N, T>>(
        domain,
        create_inst<N, T>(test_case.domain, test_case.field_ids, test_case.field_sizes),
        test_case.dim_order.data(), test_case.field_ids, test_case.field_offsets,
        test_case.field_sizes);
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

    if(cursor.get_dim() == 1) {
      // TODO(apryakhin:@): Find better way to analyze the adddress list
      // ASSERT_EQ(cursor.get_dim(), 1);
      for(const size_t field_size : test_case.field_sizes) {
        for(const auto &rect : test_case.expected) {
          int dim = cursor.get_dim();
          size_t rem = cursor.remaining(dim - 1);
          ASSERT_EQ(cursor.remaining(dim - 1), rect.volume() * field_size);
          cursor.advance(dim - 1, cursor.remaining(dim - 1));
        }
      }

      ASSERT_EQ(addrlist.bytes_pending(), 0);
    }

    delete impl;
  }
}

REGISTER_TYPED_TEST_SUITE_P(GetAddressesTest, Base);

using TestTypes = ::testing::Types<TypeWrapper<1>
#if REALM_MAX_DIM > 1
                                   ,
                                   TypeWrapper<2>
#endif
#if REALM_MAX_DIM > 2
                                   ,
                                   TypeWrapper<3>
#endif
                                   >;

INSTANTIATE_TYPED_TEST_SUITE_P(AllDimensions, GetAddressesTest, TestTypes);

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
// Case 1: step through 2D layout with 4 elements
#if REALM_MAX_DIM > 1
    IteratorStepTestCase{
        .it = new TransferIteratorIndexSpace<2, int>(
            Rect<2, int>(Point<2, int>(0), Point<2, int>(1)),
            create_inst<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(1)), {0},
                                {kByteSize}),
            0, {0}, {0}, /*field_sizes=*/{kByteSize}),
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
            Rect<2, int>(Point<2, int>(0), Point<2, int>(1)),
            create_inst<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(3)), {0},
                                {kByteSize}),
            0, {0}, {0}, /*field_sizes=*/{kByteSize}),
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
#endif

    // Case 6: step with empty rect
    IteratorStepTestCase{.it = new TransferIteratorIndexSpace<1, int>(
                             Rect<1, int>::make_empty(),
                             create_inst<1, int>(Rect<1, int>(0, 1), {0}, {kByteSize}), 0,
                             {0}, {0},
                             /*field_sizes=*/{kByteSize}),
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
