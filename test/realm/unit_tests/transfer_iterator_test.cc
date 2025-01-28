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
class IndexSpaceIteratorTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;

  void SetUp() override { std::iota(dim_order, dim_order + N, 0); }

  constexpr static size_t elem_size = 8;
  std::vector<FieldID> field_ids{0};
  std::vector<size_t> field_sizes{elem_size};
  std::vector<size_t> field_offsets{0};
  int dim_order[N];
};

TYPED_TEST_SUITE_P(IndexSpaceIteratorTest);

TYPED_TEST_P(IndexSpaceIteratorTest, GetAddressesDenseInvertedDims)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;
  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(4));
  AddressList addrlist;
  AddressListCursor cursor;
  const InstanceLayoutPieceBase *nonaffine;

  int inverted_dim_order[N];
  for(int i = N - 1; i >= 0; i--) {
    inverted_dim_order[N - 1 - i] = i;
  }

  auto it = std::make_unique<TransferIteratorIndexSpace<N, T>>(
      domain, create_inst<N, T>(domain, this->elem_size), inverted_dim_order,
      this->field_ids, this->field_offsets, this->field_sizes);

  bool ok = it->get_addresses(addrlist, nonaffine);

  cursor.set_addrlist(&addrlist);
  ASSERT_TRUE(ok);
  ASSERT_TRUE(it->done());
  ASSERT_EQ(cursor.remaining(0),
            N > 1 ? this->elem_size : this->elem_size * domain.volume());
  for(int i = 1; i < N; i++) {
    int d = inverted_dim_order[i];
    size_t count = (domain.hi[d] - domain.lo[d] + 1);
    ASSERT_EQ(cursor.remaining(i), count);
  }
}

TYPED_TEST_P(IndexSpaceIteratorTest, GetAddressesDense)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;
  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(4));

  auto it = std::make_unique<TransferIteratorIndexSpace<N, T>>(
      domain, create_inst<N, T>(domain, this->elem_size), this->dim_order,
      this->field_ids, this->field_offsets, this->field_sizes);
  const InstanceLayoutPieceBase *nonaffine;
  AddressList addrlist;

  bool ok = it->get_addresses(addrlist, nonaffine);

  AddressListCursor cursor;
  cursor.set_addrlist(&addrlist);

  ASSERT_TRUE(ok);
  ASSERT_TRUE(it->done());
  ASSERT_EQ(nonaffine, nullptr);
  ASSERT_EQ(addrlist.bytes_pending(), domain.volume() * this->elem_size);
  ASSERT_EQ(cursor.remaining(0), domain.volume() * this->elem_size);
  ASSERT_EQ(cursor.get_offset(), 0);
  ASSERT_EQ(cursor.get_dim(), 1);
}

TYPED_TEST_P(IndexSpaceIteratorTest, StepDense)
{
  constexpr int N = TestFixture::N;
  const size_t max_bytes = this->elem_size * 2;
  using T = typename TestFixture::T;
  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(3));

  auto it = std::make_unique<TransferIteratorIndexSpace<N, T>>(
      domain, create_inst<N, T>(domain, this->elem_size), this->dim_order,
      this->field_ids, this->field_offsets, this->field_sizes);

  size_t offset = 0;
  for(int i = 0; i < domain.volume() / 2; i++) {
    TransferIterator::AddressInfo info;
    size_t ret_bytes = it->step(max_bytes, info, 0, 0);
    ASSERT_EQ(ret_bytes, max_bytes);
    ASSERT_EQ(info.base_offset, offset);
    ASSERT_EQ(info.bytes_per_chunk, max_bytes);
    ASSERT_EQ(info.num_lines, 1);
    ASSERT_EQ(info.num_planes, 1);
    offset += max_bytes;
  }

  EXPECT_TRUE(it->done());
}

REGISTER_TYPED_TEST_SUITE_P(IndexSpaceIteratorTest, GetAddressesDense,
                            GetAddressesDenseInvertedDims, StepDense);

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
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, IndexSpaceIteratorTest, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int);
// TODO(apryakhin@): Consider enabling if needed
// INSTANTIATE_TEST_TYPES(long long, LongLong);

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
            create_inst<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(1)),
                                kByteSize),
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
            create_inst<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(3)),
                                kByteSize),
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
                             create_inst<1, int>(Rect<1, int>(0, 1), kByteSize), 0, {0},
                             {0},
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
