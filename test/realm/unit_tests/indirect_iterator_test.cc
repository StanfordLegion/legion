#include "realm/transfer/channel.h"
#include "realm/transfer/memcpy_channel.h"
#include "realm/transfer/transfer.h"
#include "test_common.h"
#include <gtest/gtest.h>

using namespace Realm;

template <int N>
struct TestCaseData {
  Rect<N, int> domain;
  std::vector<Rect<N, int>> rects;
  std::vector<Point<N, int>> indirection;
  std::vector<size_t> expected;
  std::vector<int> dim_order;
  std::vector<FieldID> field_ids;
  std::vector<size_t> field_offsets;
  std::vector<size_t> field_sizes;
};

struct BaseTestCaseData {
  virtual ~BaseTestCaseData() = default;
  virtual int get_dim() const = 0;
};

template <int N>
struct WrappedTestCaseData : public BaseTestCaseData {
  TestCaseData<N> data;
  explicit WrappedTestCaseData(TestCaseData<N> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class IndirectGetAddressesTest : public ::testing::TestWithParam<BaseTestCaseData *> {};

template <int N>
void run_test_case(const TestCaseData<N> &test_case)
{
  using T = int;
  AddressList addrlist;
  AddressListCursor cursor;
  cursor.set_addrlist(&addrlist);
  constexpr size_t elem_size = sizeof(int);
  const InstanceLayoutPieceBase *nonaffine;
  const size_t bytes = sizeof(Point<N, T>) * test_case.indirection.size();
  std::vector<Point<N, T>> buffer = test_case.indirection;

  std::unique_ptr<LocalCPUMemory> input_mem = std::make_unique<LocalCPUMemory>(
      Memory::NO_MEMORY, bytes, 0, Memory::SYSTEM_MEM, buffer.data());

  std::unique_ptr<MemcpyXferDes> xd = std::make_unique<MemcpyXferDes>(
      /*addrs_mem=*/0, /*channel=*/nullptr,
      /*launch_node=*/0, /*guid=*/0, std::vector<XferDesPortInfo>(),
      std::vector<XferDesPortInfo>(), /*priority=*/0);

  xd->input_ports.resize(1);
  xd->input_ports[0].mem = input_mem.get();

  std::unique_ptr<TransferIteratorIndirect<N, T>> it =
      std::make_unique<TransferIteratorIndirect<N, T>>(
          Memory::NO_MEMORY,
          create_inst<N, T>(test_case.domain, test_case.field_ids, test_case.field_sizes),
          test_case.field_ids, test_case.field_offsets, test_case.field_sizes);

  Rect<1, T> addr_domain = Rect<1, T>(Point<1, T>(0), Point<1, T>(buffer.size() - 1));
  std::vector<FieldID> indirect_fields{0};
  std::vector<size_t> indirect_field_sizes{sizeof(Point<N, T>)};
  std::unique_ptr<TransferIteratorIndexSpace<1, T>> addr_it =
      std::make_unique<TransferIteratorIndexSpace<1, T>>(
          test_case.dim_order.data(), indirect_fields, test_case.field_offsets,
          indirect_field_sizes,
          create_inst<1, T>(addr_domain, indirect_fields, indirect_field_sizes),
          addr_domain);

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

TEST_P(IndirectGetAddressesTest, Base)
{
  const BaseTestCaseData *base_test_case = GetParam();

  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        constexpr int N = Dim;
        auto &test_case =
            static_cast<const WrappedTestCaseData<N> *>(base_test_case)->data;
        run_test_case(test_case);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(
    IndirectGetAddressesCases, IndirectGetAddressesTest,
    ::testing::Values(
        // Case 1: All points are mergeable
        new WrappedTestCaseData<1>(
            {/*domain=*/Rect<1, int>(0, 14),
             /*rects=*/{Rect<1, int>(0, 14)},
             /*indirection=*/{Point<1, int>(0), Point<1, int>(1), Point<1, int>(2)},
             /*expected=*/{0},
             /*dim_order=*/{0},
             /*field_ids=*/{0},
             /*field_offsets=*/{0},
             /*field_sizes=*/{sizeof(int)}}),

        // Case 2: Merge breaks then allows to a certain point
        new WrappedTestCaseData<1>(
            {/*domain=*/Rect<1, int>(0, 14),
             /*rects=*/{Rect<1, int>(0, 14)},
             /*indirection=*/{Point<1, int>(0), Point<1, int>(3), Point<1, int>(4)},
             /*expected=*/{0, 12},
             /*dim_order=*/{0},
             /*field_ids=*/{0},
             /*field_offsets=*/{0},
             /*field_sizes=*/{sizeof(int)}}),

        // Case 3: Merge allows to a certain point then breaks
        new WrappedTestCaseData<1>(
            {/*domain=*/Rect<1, int>(0, 14),
             /*rects=*/{Rect<1, int>(0, 14)},
             /*indirection=*/{Point<1, int>(0), Point<1, int>(1), Point<1, int>(3)},
             /*expected=*/{0, 12},
             /*dim_order=*/{0},
             /*field_ids=*/{0},
             /*field_offsets=*/{0},
             /*field_sizes=*/{sizeof(int)}}),

        // Case 4: Disjoint points
        new WrappedTestCaseData<1>(
            {/*domain=*/Rect<1, int>(0, 14),
             /*rects=*/{Rect<1, int>(0, 14)},
             /*indirection=*/{Point<1, int>(0), Point<1, int>(2), Point<1, int>(4)},
             /*expected=*/{0, 8, 16},
             /*dim_order=*/{0},
             /*field_ids=*/{0},
             /*field_offsets=*/{0},
             /*field_sizes=*/{sizeof(int)}}),

        // Case 5: Disjoint points in descending order
        new WrappedTestCaseData<1>(
            {/*domain=*/Rect<1, int>(0, 14),
             /*rects=*/{Rect<1, int>(0, 14)},
             /*indirection=*/{Point<1, int>(4), Point<1, int>(2), Point<1, int>(0)},
             /*expected=*/{16, 8, 0},
             /*dim_order=*/{0},
             /*field_ids=*/{0},
             /*field_offsets=*/{0},
             /*field_sizes=*/{sizeof(int)}}),

        // Case 6: Multiple continuous merges
        new WrappedTestCaseData<1>({/*domain=*/Rect<1, int>(0, 14),
                                    /*rects=*/{Rect<1, int>(0, 14)},
                                    /*indirection=*/
                                    {Point<1, int>(0), Point<1, int>(1), Point<1, int>(3),
                                     Point<1, int>(4)},
                                    /*expected=*/{0, 12},
                                    /*dim_order=*/{0},
                                    /*field_ids=*/{0},
                                    /*field_offsets=*/{0},
                                    /*field_sizes=*/{sizeof(int)}}),

        // Case 7: Full 2D mrg efails, early exit
        new WrappedTestCaseData<2>({/*domain=*/Rect<2, int>({0, 0}, {10, 10}),
                                    /*rects=*/{Rect<2, int>({0, 0}, {10, 10})},
                                    /*indirection=*/
                                    {Point<2, int>(1, 1), Point<2, int>(1, 2),
                                     Point<2, int>(2, 4)},
                                    /*expected=*/{48, 92, 184},
                                    /*dim_order=*/{0, 1},
                                    /*field_ids=*/{0},
                                    /*field_offsets=*/{0},
                                    /*field_sizes=*/{sizeof(int)}}),

        // Case 8: Disjoint 2D points
        new WrappedTestCaseData<2>({/*domain=*/Rect<2, int>({0, 0}, {10, 10}),
                                    /*rects=*/{Rect<2, int>({0, 0}, {10, 10})},
                                    /*indirection=*/
                                    {Point<2, int>(0, 0), Point<2, int>(2, 2),
                                     Point<2, int>(4, 4)},
                                    /*expected=*/{0, 96, 192},
                                    /*dim_order=*/{0, 1},
                                    /*field_ids=*/{0},
                                    /*field_offsets=*/{0},
                                    /*field_sizes=*/{sizeof(int)}}),

        // Case 9: Start merging along Z, fails to merge in Z, trigger early exit
        new WrappedTestCaseData<3>({/*domain=*/Rect<3, int>({0, 0, 0}, {10, 10, 10}),
                                    /*rects=*/{Rect<3, int>({0, 0, 0}, {10, 10, 10})},
                                    /*indirection=*/
                                    {Point<3, int>(1, 1, 1), Point<3, int>(1, 1, 2),
                                     Point<3, int>(1, 1, 4)},
                                    /*expected=*/{532, 1016, 1984},
                                    /*dim_order=*/{0, 1, 2},
                                    /*field_ids=*/{0},
                                    /*field_offsets=*/{0},
                                    /*field_sizes=*/{sizeof(int)}}),

        // Case 10: Start merging along Z, fails to merge in Y, trigger early exit
        new WrappedTestCaseData<3>({/*domain=*/Rect<3, int>({0, 0, 0}, {10, 10, 10}),
                                    /*rects=*/{Rect<3, int>({0, 0, 0}, {10, 10, 10})},
                                    /*indirection=*/
                                    {Point<3, int>(1, 1, 1), Point<3, int>(1, 1, 2),
                                     Point<3, int>(1, 2, 2)},
                                    /*expected=*/{532, 1016, 1060},
                                    /*dim_order=*/{0, 1, 2},
                                    /*field_ids=*/{0},
                                    /*field_offsets=*/{0},
                                    /*field_sizes=*/{sizeof(int)}}),

        // Case 11: Disjoint 3D points
        new WrappedTestCaseData<3>({/*domain=*/Rect<3, int>({0, 0, 0}, {10, 10, 10}),
                                    /*rects=*/{Rect<3, int>({0, 0, 0}, {10, 10, 10})},
                                    /*indirection=*/
                                    {Point<3, int>(0, 0, 0), Point<3, int>(2, 2, 2),
                                     Point<3, int>(4, 4, 4)},
                                    /*expected=*/{0, 1064, 2128},
                                    /*dim_order=*/{0, 1, 2},
                                    /*field_ids=*/{0},
                                    /*field_offsets=*/{0},
                                    /*field_sizes=*/{sizeof(int)}})));
