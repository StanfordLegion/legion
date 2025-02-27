#include "realm/transfer/transfer_utils.h"
#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

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

constexpr static size_t kByteSize = sizeof(int);

template <int N, typename T>
static InstanceLayout<N, T> *create_layout(Rect<N, T> bounds,
                                           const std::vector<FieldID> &field_ids,
                                           const std::vector<size_t> &field_sizes)
{
  InstanceLayout<N, T> *inst_layout = new InstanceLayout<N, T>();

  inst_layout->piece_lists.resize(field_ids.size());

  for(size_t i = 0; i < field_ids.size(); i++) {
    InstanceLayoutGeneric::FieldLayout field_layout;
    field_layout.list_idx = i;
    field_layout.rel_offset = 0;
    field_layout.size_in_bytes = field_sizes[i];

    AffineLayoutPiece<N, T> *affine_piece = new AffineLayoutPiece<N, T>();
    affine_piece->bounds = bounds;
    affine_piece->offset = 0;
    affine_piece->strides[0] = field_sizes[i];
    size_t mult = affine_piece->strides[0];
    for(int d = 1; d < N; d++) {
      affine_piece->strides[d] = (bounds.hi[d - 1] - bounds.lo[d - 1] + 1) * mult;
      mult *= (bounds.hi[d - 1] - bounds.lo[d - 1] + 1);
    }

    inst_layout->space = bounds;
    inst_layout->fields[field_ids[i]] = field_layout;
    inst_layout->piece_lists[i].pieces.push_back(affine_piece);
  }

  return inst_layout;
}

template <int N, typename T>
RegionInstanceImpl *create_inst(Rect<N, T> bounds, const std::vector<FieldID> &field_ids,
                                const std::vector<size_t> &field_sizes)
{
  RegionInstance inst = ID::make_instance(0, 0, 0, 0).convert<RegionInstance>();
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, field_ids, field_sizes);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  return impl;
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
