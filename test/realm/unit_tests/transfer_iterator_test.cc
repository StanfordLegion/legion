#include "realm/transfer/transfer_utils.h"
#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

struct IteratorTestCase {
  TransferIterator *it;
  std::vector<TransferIterator::AddressInfo> infos;
  std::vector<size_t> max_bytes;
  std::vector<size_t> exp_bytes;
  int num_steps;
};

class TransferIteratorParamTest : public ::testing::TestWithParam<IteratorTestCase> {};

TEST_P(TransferIteratorParamTest, Base)
{
  IteratorTestCase test_case = GetParam();

  for(int i = 0; i < test_case.num_steps; i++) {
    TransferIterator::AddressInfo info;
    size_t bytes = test_case.it->step(test_case.max_bytes[i], info, 0, 0);

    EXPECT_EQ(bytes, test_case.exp_bytes[i]);

    if(!test_case.infos.empty()) {
      EXPECT_EQ(info.base_offset, test_case.infos[i].base_offset);
      EXPECT_EQ(info.bytes_per_chunk, test_case.infos[i].bytes_per_chunk);
      EXPECT_EQ(info.num_lines, test_case.infos[i].num_lines);
      EXPECT_EQ(info.line_stride, test_case.infos[i].line_stride);
      EXPECT_EQ(info.num_planes, test_case.infos[i].num_planes);
      // EXPECT_FALSE(it.done());
    }
  }
}

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

const static size_t kByteSize = sizeof(int);

const static IteratorTestCase kIteratorTestCases[] = {
    // Case 0: step through 1D layout with 2 elements
    IteratorTestCase{
        .it = new TransferIteratorIndexSpace<1, int>(
            Rect<1, int>(0, 1), create_layout<1, int>(Rect<1, int>(0, 1), kByteSize), 0,
            0, {0}, {0}, /*field_sizes=*/{kByteSize}, 0),
        .infos = {TransferIterator::AddressInfo{/*offset=*/0, /*bytes_per_el=*/kByteSize,
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
    IteratorTestCase{
        .it = new TransferIteratorIndexSpace<2, int>(
            Rect<2, int>(Point<2, int>(0), Point<2, int>(1)),
            create_layout<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(1)),
                                  kByteSize),
            0, 0, {0}, {0}, /*field_sizes=*/{kByteSize}, 0),
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

    // Case 2: step through 2D layout at once
    IteratorTestCase{
        .it = new TransferIteratorIndexSpace<2, int>(
            Rect<2, int>(Point<2, int>(0), Point<2, int>(1)),
            create_layout<2, int>(Rect<2, int>(Point<2, int>(0), Point<2, int>(1)),
                                  kByteSize),
            0, 0, {0}, {0}, /*field_sizes=*/{kByteSize}, 0),
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

    // Case 3: step with empty rect
    IteratorTestCase{.it = new TransferIteratorIndexSpace<1, int>(
                         Rect<1, int>::make_empty(),
                         create_layout<1, int>(Rect<1, int>(0, 1), kByteSize), 0, 0, {0},
                         {0},
                         /*field_sizes=*/{kByteSize}, 0),
                     .max_bytes = {0},
                     .exp_bytes = {0},
                     .num_steps = 1},
};

INSTANTIATE_TEST_SUITE_P(Foo, TransferIteratorParamTest,
                         testing::ValuesIn(kIteratorTestCases));
