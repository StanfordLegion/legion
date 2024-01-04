#include "realm/transfer/address_list.h"

#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class AddressListTestsWithParams : public ::testing::TestWithParam<std::tuple<int, int>> {
};

// TODO(apryakhin): Consider merging tests and testing for edge cases
TEST_P(AddressListTestsWithParams, Create1DEntry)
{
  const size_t dim = 1;
  const size_t stride = std::get<0>(GetParam());
  const size_t bytes = std::get<1>(GetParam());
  assert(stride <= bytes);

  AddressList addrlist;
  size_t *addr_data = addrlist.begin_nd_entry(dim);
  addr_data[0] = (bytes << 4) + dim;
  addrlist.commit_nd_entry(dim, bytes);

  AddressListCursor addrcursor;
  addrcursor.set_addrlist(&addrlist);

  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes);
  EXPECT_EQ(addrcursor.get_dim(), dim);
  EXPECT_EQ(addrcursor.get_offset(), 0);

  addrcursor.advance(dim - 1, stride);
  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes - stride);

  addrcursor.skip_bytes(stride);
  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes - 2 * stride);

  addrcursor.advance(dim - 1, bytes - 2 * stride);
  EXPECT_EQ(addrlist.bytes_pending(), 0);
}

TEST_P(AddressListTestsWithParams, Create3DEntry)
{
  AddressList addrlist;
  // TODO(apryakhin): parameterize dimensions
  const size_t dim = 3;
  const size_t stride = std::get<0>(GetParam());
  const size_t bytes = std::get<1>(GetParam());
  const std::vector<size_t> strides{stride, stride * stride, stride * stride * stride};
  const size_t total_bytes = strides[0];
  size_t *addr_data = addrlist.begin_nd_entry(dim);

  size_t cur_dim = 1;
  for(int i = 0; i < strides.size() - 1; i++) {
    addr_data[cur_dim * 2] = bytes / strides[i];
    addr_data[cur_dim * 2 + 1] = strides[i];
    cur_dim++;
  }

  addr_data[0] = (strides[0] << 4) + cur_dim;
  addrlist.commit_nd_entry(cur_dim, bytes);

  AddressListCursor addrcursor;
  addrcursor.set_addrlist(&addrlist);
  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes / strides[1]);
  EXPECT_EQ(addrcursor.remaining(dim - 2), bytes / strides[0]);
  EXPECT_EQ(addrcursor.remaining(dim - 3), strides[0]);

  addrcursor.advance(dim - 1, 1);
  EXPECT_EQ(addrlist.bytes_pending(), 0);
}

TEST_P(AddressListTestsWithParams, CommitMax1DEntries)
{
  const size_t dim = 1;
  const size_t stride = std::get<0>(GetParam());
  const size_t bytes = std::get<1>(GetParam());
  const size_t max_entries = 499;
  assert(stride <= bytes);

  AddressList addrlist;
  for(size_t i = 0; i < max_entries; i++) {
    size_t *addr_data = addrlist.begin_nd_entry(dim);
    EXPECT_NE(addr_data, nullptr);
    addr_data[0] = (bytes << 4) + dim;
    addrlist.commit_nd_entry(dim, bytes);
  }

  AddressListCursor addrcursor;
  addrcursor.set_addrlist(&addrlist);

  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes);
  EXPECT_EQ(addrcursor.get_dim(), dim);
  EXPECT_EQ(addrcursor.get_offset(), 8);

  addrcursor.advance(dim - 1, stride);
  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes - stride);

  // advance rest of first entry
  addrcursor.advance(dim - 1, bytes - stride);
  EXPECT_EQ(addrlist.bytes_pending(), bytes * (max_entries - 1));
}

INSTANTIATE_TEST_CASE_P(AddressListTest, AddressListTestsWithParams,
                        ::testing::Values(std::make_tuple(8, 1024),
                                          std::make_tuple(8, 2048)));
