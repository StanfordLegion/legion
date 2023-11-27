#include "realm.h"
#include "realm/transfer/channel.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>
#include <tuple>

#include <time.h>
#include <gtest/gtest.h>

using namespace Realm;

class AddressListTestsWithParams : public ::testing::TestWithParam<std::tuple<int>> {};

TEST_P(AddressListTestsWithParams, AdvanceBase1D)
{
  AddressList addrlist;
  size_t dim = 1;
  size_t bytes = std::get<0>(GetParam());
  size_t *addr_data = addrlist.begin_nd_entry(dim);
  addr_data[0] = (bytes << 4) + dim;
  addrlist.commit_nd_entry(dim, bytes);
  AddressListCursor addrcursor;
  addrcursor.set_addrlist(&addrlist);
  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes);
  EXPECT_EQ(addrcursor.get_dim(), dim);
  EXPECT_EQ(addrcursor.get_offset(), 0);
  addrcursor.advance(dim - 1, bytes);
  EXPECT_EQ(addrlist.bytes_pending(), 0);
}

TEST_P(AddressListTestsWithParams, AdvanceBase2D)
{
  AddressList addrlist;
  // TODO(apryakhin): parameterize this
  size_t dim = 2;
  size_t stride = 8;
  size_t bytes = std::get<0>(GetParam());
  size_t total_bytes = stride;
  size_t *addr_data = addrlist.begin_nd_entry(dim);

  size_t cur_dim = 1;
  addr_data[cur_dim * 2] = bytes / stride; // count
  addr_data[cur_dim * 2 + 1] = stride; // stride
  total_bytes *= bytes / stride;
  cur_dim++;

  addr_data[0] = (stride << 4) + cur_dim;
  addrlist.commit_nd_entry(cur_dim, total_bytes);

  AddressListCursor addrcursor;
  addrcursor.set_addrlist(&addrlist);
  EXPECT_EQ(addrcursor.remaining(dim - 1), bytes / stride);
  EXPECT_EQ(addrcursor.remaining(dim - 2), stride);

  EXPECT_EQ(addrcursor.get_dim(), dim);
  EXPECT_EQ(addrcursor.get_offset(), 0);
  // advance first entry
  addrcursor.advance(dim - 1, 1);
  EXPECT_EQ(addrlist.bytes_pending(), bytes - stride);
  // advance remaining
  addrcursor.advance(dim - 1, bytes / stride - 1);
  EXPECT_EQ(addrlist.bytes_pending(), 0);
}

INSTANTIATE_TEST_CASE_P(AddressListTest, AddressListTestsWithParams,
                        ::testing::Values(std::make_tuple(16), std::make_tuple(512),
                                          std::make_tuple(1024)));
