#include "realm/transfer/channel.h"
#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
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
static RegionInstanceImpl *
create_inst(Rect<N, T> bounds, const std::vector<FieldID> &field_ids,
            const std::vector<size_t> &field_sizes, RegionInstance inst = make_inst())
{
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, field_ids, field_sizes);
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
class IndirectGetAddresses : public ::testing::Test {
public:
  static constexpr int DIM = TypeWrapper::value;
};

template <int DIM>
struct IndirectGetAddressesCase {
  Rect<DIM> domain;
  std::vector<Rect<DIM>> rects;
  std::vector<Point<DIM>> indirection;
  std::vector<size_t> expected;
  std::vector<int> dim_order;
  std::vector<FieldID> field_ids;
  std::vector<size_t> field_offsets;
  std::vector<size_t> field_sizes;
};

template <int DIM>
std::vector<IndirectGetAddressesCase<DIM>> GetTestCases()
{
  if constexpr(DIM == 1) {
    return {
        // Case 1: all points are mergeable
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*indirection=*/{Point<1>(0), Point<1>(1), Point<1>(2)},
            /*expected=*/{0},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 2: merge breaks then allows to a certain point
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*indirection=*/{Point<1>(0), Point<1>(3), Point<1>(4)},
            /*expected=*/{0, 12},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 3: merge allows to a certain point then breaks
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*indirection=*/{Point<1>(0), Point<1>(1), Point<1>(3)},
            /*expected=*/{0, 12},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 4: disjoint points
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*indirection=*/{Point<1>(0), Point<1>(2), Point<1>(4)},
            /*expected=*/{0, 8, 16},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 5: disjoint points descending order
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*indirection=*/{Point<1>(4), Point<1>(2), Point<1>(0)},
            /*expected=*/{16, 8, 0},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 6: out of range point
        // TODO(apryakhin): Iterator needs fixing

        // Case 6: multiple continuous merges
        {
            /*domain=*/{Rect<1>(0, 14)},
            /*rects=*/{Rect<1>(0, 14)},
            /*indirection=*/{Point<1>(0), Point<1>(1), Point<1>(3), Point<1>(4)},
            /*expected=*/{0, 12},
            /*dim_order=*/{0},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },
    };
  } else if constexpr(DIM == 2) {
    return {

        // Case 7: full 2D merge fails, early exit
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*rects*/ {Rect<2>({0, 0}, {10, 10})},
            /*indirection=*/{Point<2>(1, 1), Point<2>(1, 2), Point<2>(2, 4)},
            /*expected=*/{48, 92, 184},
            /*dim_order=*/{0, 1},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 8: disjoint 2D
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*rects*/ {Rect<2>({0, 0}, {10, 10})},
            /*indirection=*/{Point<2>(0, 0), Point<2>(2, 2), Point<2>(4, 4)},
            /*expected=*/
            {0, 96, 192},
            /*dim_order=*/{0, 1},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },
    };
  } else if constexpr(DIM == 3) {
    return {
        // Case 9: start and merges along z, fails to merge in Z, trigger early exit
        {
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*indirection=*/{Point<3>(1, 1, 1), Point<3>(1, 1, 2), Point<3>(1, 1, 4)},
            /*expected=*/{532, 1016, 1984},
            /*dim_order=*/{0, 1, 2},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 10: start and merges along z, fails to merge in Y, trigger early exit
        {
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*indirection=*/{Point<3>(1, 1, 1), Point<3>(1, 1, 2), Point<3>(1, 2, 2)},
            /*expected=*/{532, 1016, 1060},
            /*dim_order=*/{0, 1, 2},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

        // Case 11: disjoint 3D
        {
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*indirection=*/{Point<3>(0, 0, 0), Point<3>(2, 2, 2), Point<3>(4, 4, 4)},
            /*expected=*/{0, 1064, 2128},
            /*dim_order=*/{0, 1, 2},
            /*field_ids=*/{0},
            /*field_offsets=*/{0},
            /*field_sizes=*/{sizeof(int)},
        },

    };
  }
  return {};
}

TYPED_TEST_SUITE_P(IndirectGetAddresses);

TYPED_TEST_P(IndirectGetAddresses, Base)
{
  using T = int;
  constexpr int N = TypeParam::value;

  auto test_cases = GetTestCases<N>();
  for(const auto &test_case : test_cases) {

    AddressList addrlist;
    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);
    std::vector<XferDesPortInfo> inputs_info;
    std::vector<XferDesPortInfo> outputs_info;
    constexpr size_t elem_size = sizeof(int);
    const InstanceLayoutPieceBase *nonaffine;
    const size_t bytes = sizeof(Point<N, T>) * test_case.indirection.size();
    std::vector<Point<N, T>> buffer = test_case.indirection;

    int dim_order[N];
    for(int i = 0; i < N; i++) {
      dim_order[i] = test_case.dim_order[i];
    }

    std::unique_ptr<LocalCPUMemory> input_mem = std::make_unique<LocalCPUMemory>(
        Memory::NO_MEMORY, bytes, 0, Memory::SYSTEM_MEM, buffer.data());

    std::unique_ptr<MemcpyXferDes> xd = std::make_unique<MemcpyXferDes>(
        /*addrs_mem=*/0, /*channel=*/nullptr,
        /*launch_node=*/0, /*guid=*/0, inputs_info, outputs_info, /*priority=*/0);

    xd->input_ports.resize(1);
    xd->input_ports[0].mem = input_mem.get();

    std::unique_ptr<TransferIteratorIndirect<N, T>> it =
        std::make_unique<TransferIteratorIndirect<N, T>>(
            Memory::NO_MEMORY,
            create_inst<N, T>(test_case.domain, test_case.field_ids,
                              test_case.field_sizes),
            test_case.field_ids, test_case.field_offsets, test_case.field_sizes);

    Rect<1, T> addr_domain = Rect<1, T>(Point<1, T>(0), Point<1, T>(buffer.size() - 1));
    std::vector<FieldID> indirect_fields{0};
    std::vector<size_t> indirect_field_sizes{sizeof(Point<N, T>)};
    std::unique_ptr<TransferIteratorIndexSpace<1, T>> addr_it =
        std::make_unique<TransferIteratorIndexSpace<1, T>>(
            addr_domain,
            create_inst<1, T>(addr_domain, indirect_fields, indirect_field_sizes),
            dim_order, indirect_fields, test_case.field_offsets, indirect_field_sizes,
            /*extra_elems=*/0);
    it->set_indirect_input_port(xd.get(), /*indirect_port_idx=*/0, addr_it.get());

    bool done_early = it->done();
    bool ok = it->get_addresses(addrlist, nonaffine);
    bool done_later = it->done();

    ASSERT_FALSE(done_early);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(done_later);
    ASSERT_EQ(nonaffine, nullptr);
    ASSERT_EQ(addrlist.bytes_pending(), buffer.size() * elem_size);

    for(size_t offset : test_case.expected) {
      EXPECT_EQ(offset, cursor.get_offset());
      cursor.advance(0, cursor.remaining(0));
    }

    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }
}

REGISTER_TYPED_TEST_SUITE_P(IndirectGetAddresses, Base);

template <typename Seq>
struct WrapTypes;

template <std::size_t... Ns>
struct WrapTypes<std::index_sequence<Ns...>> {
  using type = ::testing::Types<std::integral_constant<int, Ns + 1>...>;
};

using TestTypes = typename WrapTypes<std::make_index_sequence<REALM_MAX_DIM>>::type;

INSTANTIATE_TYPED_TEST_SUITE_P(AllDimensions, IndirectGetAddresses, TestTypes);
