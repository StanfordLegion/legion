#include "realm/transfer/channel.h"
#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include <numeric>
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

template <int N, typename T>
static InstanceLayout<N, T> *create_layout(Rect<N, T> bounds,
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
  affine_piece->strides[0] = bytes_per_element;
  size_t mult = affine_piece->strides[0];
  for(int i = 1; i < N; i++) {
    affine_piece->strides[i] = (bounds.hi[i - 1] - bounds.lo[i - 1] + 1) * mult;
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
static RegionInstanceImpl *create_inst(Rect<N, T> bounds, size_t bytes_per_element = 8,
                                       RegionInstance inst = make_inst())
{
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, bytes_per_element);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  return impl;
}

template <typename PointType>
struct PointTraits;

template <int N, typename T>
struct PointTraits<Realm::Point<N, T>> {
  static constexpr int DIM = N;
  using value_type = T;
};

template <typename PointType>
class IndirectIteratorTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;

  void SetUp() override { std::iota(dim_order, dim_order + N, 0); }
  int dim_order[N];
};

TYPED_TEST_SUITE_P(IndirectIteratorTest);

TYPED_TEST_P(IndirectIteratorTest, GetAddresses)
{
  constexpr int N = TestFixture::N;
  constexpr size_t elem_size = 8;
  constexpr size_t max_bytes = elem_size * 2;
  using T = typename TestFixture::T;
  const InstanceLayoutPieceBase *nonaffine;
  AddressList addrlist;
  AddressListCursor cursor;
  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;
  std::vector<FieldID> field_ids{0};
  std::vector<size_t> field_sizes{elem_size};
  std::vector<size_t> field_offsets{0};

  T index = 0;
  constexpr int num_points = 4;
  constexpr int step = 2;
  constexpr size_t bytes = sizeof(TypeParam) * num_points * 1;
  std::vector<TypeParam> buffer;
  for(int i = 0; i < num_points; i++) {
    TypeParam point(0);
    point.x() = index;
    index += step;
    buffer.emplace_back(point);
  }

  /*for(int i = 0; i < num_points; i++) {
     buffer.emplace_back(TypeParam(index++));
  }*/
  auto input_mem = std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, bytes, 0,
                                                    Memory::SYSTEM_MEM, buffer.data());

  auto xd = std::make_unique<MemcpyXferDes>(/*addrs_mem=*/0, /*channel=*/nullptr,
                                            /*launch_node=*/0, /*guid=*/0, inputs_info,
                                            outputs_info, /*priority=*/0);
  xd->input_ports.resize(1);
  xd->input_ports[0].mem = input_mem.get();

  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(index));
  auto it = std::make_unique<TransferIteratorIndirect<N, T>>(
      Memory::NO_MEMORY, create_inst<N, T>(domain, elem_size), field_ids, field_offsets,
      field_sizes);

  Rect<1, T> addr_domain = Rect<1, T>(Point<1, T>(0), Point<1, T>(buffer.size() - 1));
  std::vector<FieldID> indirect_fields{0};
  std::vector<size_t> indirect_field_sizes{sizeof(TypeParam)};
  auto addr_it = std::make_unique<TransferIteratorIndexSpace<1, T>>(
      addr_domain, create_inst<1, T>(addr_domain, sizeof(TypeParam)), this->dim_order,
      indirect_fields, field_offsets, indirect_field_sizes, /*extra_elems=*/0);

  it->set_indirect_input_port(xd.get(), /*indirect_port_idx=*/0, addr_it.get());
  bool ok = it->get_addresses(addrlist, nonaffine);
  cursor.set_addrlist(&addrlist);

  ASSERT_TRUE(ok);
  ASSERT_TRUE(it->done());
  ASSERT_EQ(nonaffine, nullptr);
  ASSERT_EQ(addrlist.bytes_pending(), buffer.size() * elem_size);
  EXPECT_EQ(cursor.remaining(0), elem_size);

  size_t offset = 0;
  for(int i = 0; i < buffer.size(); i++) {
    EXPECT_EQ(cursor.get_offset(), offset);
    cursor.advance(0, elem_size);
    offset += elem_size * step;
  }
}

REGISTER_TYPED_TEST_SUITE_P(IndirectIteratorTest, GetAddresses);

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
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, IndirectIteratorTest, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int);
// TODO(apryakhin@): Consider enabling if needed
// INSTANTIATE_TEST_TYPES(long long, LongLong);
