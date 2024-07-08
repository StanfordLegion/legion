#include "realm/transfer/transfer_utils.h"
#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

// TODO(apryakhin@): Consider parameterizing even more..
class TransferIteratorIndexSpaceTest
  : public ::testing::TestWithParam<std::tuple<unsigned, Rect<3, int>>> {
protected:
  void SetUp()
  {
    const unsigned bytes_per_element = std::get<0>(GetParam());
    IndexSpace<3, int> is(std::get<1>(GetParam()));

    InstanceLayoutGeneric::FieldLayout field_layout;
    field_layout.list_idx = 0;
    field_layout.rel_offset = 0;
    field_layout.size_in_bytes = bytes_per_element;

    AffineLayoutPiece<3, int> *affine_piece = new AffineLayoutPiece<3, int>();
    affine_piece->offset = 0;
    affine_piece->strides[0] = bytes_per_element;
    affine_piece->strides[1] =
        bytes_per_element * (is.bounds.hi[0] - is.bounds.lo[0] + 1);
    affine_piece->strides[2] = bytes_per_element *
                               (is.bounds.hi[0] - is.bounds.lo[0] + 1) *
                               (is.bounds.hi[1] - is.bounds.lo[1] + 1);
    affine_piece->bounds = is.bounds;

    inst_layout = new InstanceLayout<3, int>();
    inst_layout->space = is.bounds;
    inst_layout->fields[0] = field_layout;
    inst_layout->piece_lists.resize(1);
    inst_layout->piece_lists[0].pieces.push_back(affine_piece);
  }

  void TearDown() { delete inst_layout; }

  InstanceLayout<3, int> *inst_layout = nullptr;
};

TEST_P(TransferIteratorIndexSpaceTest, Step3DEmpty)
{
  const unsigned bytes_per_element = std::get<0>(GetParam());
  IndexSpace<3, int> is(std::get<1>(GetParam()));
  const std::vector<FieldID> fields{0};
  const std::vector<size_t> fld_offsets{0};
  const std::vector<size_t> fld_sizes{bytes_per_element};
  TransferIteratorIndexSpace<3, int> it(is, inst_layout, 0, 0, fields, fld_offsets,
                                        fld_sizes, 0);

  size_t max_bytes = 0;
  TransferIterator::AddressInfo info{0, 0, 0, 0, 0};
  size_t bytes = it.step(max_bytes, info, 0, 0);

  EXPECT_EQ(bytes, max_bytes);
  EXPECT_EQ(info.base_offset, 0);
  EXPECT_EQ(info.bytes_per_chunk, 0);
  EXPECT_EQ(info.num_lines, 0);
  EXPECT_EQ(info.line_stride, 0);
  EXPECT_EQ(info.num_planes, 0);
  EXPECT_FALSE(it.done());
}

TEST_P(TransferIteratorIndexSpaceTest, Step3D)
{
  const unsigned bytes_per_element = std::get<0>(GetParam());
  IndexSpace<3, int> is(std::get<1>(GetParam()));
  const std::vector<FieldID> fields{0};
  const std::vector<size_t> fld_offsets{0};
  const std::vector<size_t> fld_sizes{bytes_per_element};
  TransferIteratorIndexSpace<3, int> it(is, inst_layout, 0, 0, fields, fld_offsets,
                                        fld_sizes, 0);

  size_t offset = 0;
  size_t max_bytes = is.bounds.volume() * bytes_per_element;
  for(int i = 0; i < (is.bounds.volume() * bytes_per_element) / max_bytes; i++) {
    TransferIterator::AddressInfo info{0, 0, 0, 0, 0};
    size_t bytes = it.step(max_bytes, info, 0, 0);
    EXPECT_EQ(bytes, max_bytes);
    EXPECT_EQ(info.base_offset, offset);
    EXPECT_EQ(info.bytes_per_chunk, max_bytes);
    EXPECT_EQ(info.num_lines, 1);
    EXPECT_EQ(info.line_stride, 0);
    EXPECT_EQ(info.num_planes, 1);
    offset += max_bytes;
  }

  EXPECT_TRUE(it.done());
}

TEST_P(TransferIteratorIndexSpaceTest, GetAddressesIsCoversEntirePiece)
{
  const unsigned bytes_per_element = std::get<0>(GetParam());
  // TODO(apryakhin@): Add more tests in releation
  IndexSpace<3, int> is(std::get<1>(GetParam()));
  const std::vector<FieldID> fields{0};
  const std::vector<size_t> fld_offsets{0};
  const std::vector<size_t> fld_sizes{bytes_per_element};
  TransferIteratorIndexSpace<3, int> it(is, inst_layout, 0, 0, fields, fld_offsets,
                                        fld_sizes, 0);

  AddressList addrlist;
  const InstanceLayoutPieceBase *nonaffine;
  bool done = it.get_addresses(addrlist, nonaffine);

  EXPECT_EQ(done, true);
  EXPECT_EQ(addrlist.bytes_pending(), is.bounds.volume() * bytes_per_element);
  AddressListCursor cursor;
  cursor.set_addrlist(&addrlist);
  // EXPECT_EQ(cursor.get_dim(), 1);
  EXPECT_EQ(cursor.get_offset(), 0);
  EXPECT_EQ(cursor.remaining(0), is.bounds.volume() * bytes_per_element);
}

INSTANTIATE_TEST_SUITE_P(
    IteratorTest, TransferIteratorIndexSpaceTest,
    testing::Values(std::make_tuple(8, Rect<3, int>({0, 0, 0}, {3, 0, 0})),
                    std::make_tuple(8, Rect<3, int>({0, 0, 0}, {3, 3, 0})),
                    std::make_tuple(8, Rect<3, int>({0, 0, 0}, {3, 3, 3})),
                    std::make_tuple(16, Rect<3, int>({0, 0, 0}, {3, 3, 3}))));
