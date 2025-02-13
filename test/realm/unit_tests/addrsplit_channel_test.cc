#include "realm/transfer/addrsplit_channel.h"
#include <tuple>
#include <gtest/gtest.h>
#include <cstring>
#include <vector>

using namespace Realm;

class MockAddressSplitChannel : public AddressSplitChannel {
public:
  MockAddressSplitChannel(BackgroundWorkManager *bgwork)
    : AddressSplitChannel(bgwork)
  {}

  void enqueue_ready_xd(XferDes *xd) override { num_xds++; }

  int num_xds = 0;
};

template <int DIM, typename T>
class MockIterator : public TransferIterator {
public:
  MockIterator(int _max_iterations, size_t _total_bytes = 256)
    : max_iterations(_max_iterations)
    , total_bytes(_total_bytes)
  {}

  Event request_metadata(void) override { return Event::NO_EVENT; }

  void reset(void) override {}

  bool done(void) override { return iterations >= max_iterations; }
  size_t step(size_t max_bytes, AddressInfo &info, unsigned flags,
              bool tentative = false) override
  {
    info.base_offset = offset;
    size_t bytes = std::min(max_bytes, total_bytes);
    offset += bytes;
    iterations++;
    return bytes;
  }

  size_t step_custom(size_t max_bytes, AddressInfoCustom &info,
                     bool tentative = false) override
  {
    assert(0);
    return 0;
  }

  void confirm_step(void) override {}
  void cancel_step(void) override {}

  size_t get_base_offset(void) const override { return 0; }

  bool get_addresses(AddressList &addrlist,
                     const InstanceLayoutPieceBase *&nonaffine) override
  {
    nonaffine = 0;
    return false;
  }

  size_t max_iterations = 0;
  size_t iterations = 0;
  size_t offset = 0;
  size_t total_bytes = 0;
};

template <int N, typename T>
struct AddressSplitXferDescTestCase {
  int expected_iterations = 1;
  size_t bytes_per_element = 4;
  std::vector<IndexSpace<N, T>> spaces;
  std::vector<Point<N, T>> src_points;
  std::vector<std::vector<Point<N, T>>> exp_points;
};

struct BaseTestCaseData {
  virtual ~BaseTestCaseData() = default;
  virtual int get_dim() const = 0;
};

template <int N, typename T>
struct WrappedTestCaseData : public BaseTestCaseData {
  AddressSplitXferDescTestCase<N, T> data;
  explicit WrappedTestCaseData(AddressSplitXferDescTestCase<N, T> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class AddressSplitTest : public ::testing::TestWithParam<BaseTestCaseData *> {};

template <int N, typename T>
void run_test_case(const AddressSplitXferDescTestCase<N, T> &test_case)
{
  std::unique_ptr<BackgroundWorkManager> bgwork =
      std::make_unique<BackgroundWorkManager>();
  std::unique_ptr<MockAddressSplitChannel> addrsplit_channel =
      std::make_unique<MockAddressSplitChannel>(bgwork.get());

  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;

  XferDesRedopInfo redop_info;
  XferDesID guid = 0;
  int priority = 0;
  NodeID owner = 0;
  Node node_data;

  AddressSplitXferDes<N, T> xfer_des(0, addrsplit_channel.get(), owner, guid, inputs_info,
                                     outputs_info, priority, sizeof(Point<N, T>),
                                     test_case.spaces);

  xfer_des.input_ports.resize(1);
  XferDes::XferPort &input_port = xfer_des.input_ports[0];

  const size_t src_points = test_case.src_points.size();
  size_t total_src_bytes = sizeof(Point<N, T>) * src_points;
  Point<N, T> *src_buffer = new Point<N, T>[src_points];
  std::memcpy(src_buffer, test_case.src_points.data(), total_src_bytes);
  auto input_mem = std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, total_src_bytes, 0,
                                                    Memory::SYSTEM_MEM, src_buffer);
  input_port.mem = input_mem.get();
  input_port.peer_port_idx = 0;

  input_port.iter =
      new MockIterator<1, int>(test_case.expected_iterations, total_src_bytes);

  input_port.peer_guid = XferDes::XFERDES_NO_GUID;

  size_t num_spaces = test_case.spaces.size();

  std::vector<std::unique_ptr<LocalCPUMemory>> buffs;
  xfer_des.output_ports.resize(num_spaces + 1);
  for(size_t i = 0; i < num_spaces; i++) {
    XferDes::XferPort &output_port = xfer_des.output_ports[i];
    const size_t dst_points = src_points;
    size_t total_dst_bytes = sizeof(Point<N, T>) * dst_points;
    Point<N, T> *dst_buffer_one = new Point<N, T>[dst_points];

    buffs.emplace_back(std::make_unique<LocalCPUMemory>(
        Memory::NO_MEMORY, total_dst_bytes, 0, Memory::SYSTEM_MEM, dst_buffer_one));

    output_port.mem = buffs.back().get();
    output_port.peer_port_idx = 0;
    output_port.seq_remote.add_span(0, total_src_bytes);
    output_port.iter =
        new MockIterator<1, int>(test_case.expected_iterations, total_src_bytes);
  }

  XferDes::XferPort &output_port = xfer_des.output_ports[num_spaces];
  Point<N, T> *dst_buffer_two = new Point<N, T>[src_points];
  auto output_mem_two = std::make_unique<LocalCPUMemory>(
      Memory::NO_MEMORY, total_src_bytes, 0, Memory::SYSTEM_MEM, dst_buffer_two);
  output_port.mem = output_mem_two.get();
  output_port.peer_port_idx = 0;
  output_port.seq_remote.add_span(0, total_src_bytes);
  output_port.iter =
      new MockIterator<1, int>(test_case.expected_iterations, total_src_bytes);

  while(!xfer_des.transfer_completed.load()) {
    xfer_des.progress_xd(addrsplit_channel.get(), TimeLimit::relative(1000000));
  }

  for(size_t i = 0; i < test_case.exp_points.size(); i++) {
    const Point<N, T> *buffer =
        reinterpret_cast<const Point<N, T> *>(buffs[i]->get_direct_ptr(
            0, test_case.exp_points[i].size() * sizeof(Point<N, T>)));
    for(size_t j = 0; j < test_case.exp_points[i].size(); j++) {
      EXPECT_EQ(buffer[j], test_case.exp_points[i][j]);
    }
  }

  addrsplit_channel->shutdown();
}

template <typename Func, size_t... Is>
void dispatch_for_dimension(int dim, Func &&func, std::index_sequence<Is...>)
{
  (
      [&] {
        if(dim == static_cast<int>(Is + 1)) {
          func(std::integral_constant<int, Is + 1>{});
        }
      }(),
      ...);
}

TEST_P(AddressSplitTest, ProgressXD)
{
  const BaseTestCaseData *base_test_case = GetParam();

  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto DimConstant) {
        constexpr int N = DimConstant.value;
        auto &test_case =
            static_cast<const WrappedTestCaseData<N, int> *>(base_test_case)->data;
        run_test_case(test_case);
      },
      std::make_index_sequence<3>{});
}

INSTANTIATE_TEST_SUITE_P(
    AddressSplitCases, AddressSplitTest,
    ::testing::Values(
        // Case 0: All points inside one space
        new WrappedTestCaseData<1, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/{IndexSpace<1, int>{Rect<1>{Point<1>(0), Point<1>(3)}}},
             /*src_points=*/
             {Point<1, int>(0), Point<1, int>(1), Point<1, int>(2), Point<1, int>(3)},
             /*exp_points=*/
             {{Point<1, int>(0), Point<1, int>(1), Point<1, int>(2), Point<1, int>(3)}}}),

        // Case 1: Partial overlap with a single index space
        new WrappedTestCaseData<1, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/{IndexSpace<1, int>{Rect<1>{Point<1>(2), Point<1>(6)}}},
             /*src_points=*/
             {Point<1, int>(0), Point<1, int>(1), Point<1, int>(2), Point<1, int>(3)},
             /*exp_points=*/{{Point<1, int>(2), Point<1, int>(3)}}}),

        // Case 2: Multiple disjoint index spaces
        new WrappedTestCaseData<1, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/
             {IndexSpace<1, int>{Rect<1>{Point<1>(0), Point<1>(2)}},
              IndexSpace<1, int>{Rect<1>{Point<1>(3), Point<1>(6)}}},
             /*src_points=*/
             {Point<1, int>(1), Point<1, int>(2), Point<1, int>(3), Point<1, int>(4)},
             /*exp_points=*/
             {{Point<1, int>(1), Point<1, int>(2)},
              {Point<1, int>(3), Point<1, int>(4)}}}),

        // Case 3: All input points outside index spaces (should produce no output)
        new WrappedTestCaseData<1, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/{IndexSpace<1, int>{Rect<1>{Point<1>(10), Point<1>(20)}}},
             /*src_points=*/{Point<1, int>(0), Point<1, int>(1), Point<1, int>(2)},
             /*exp_points=*/{{}}}),

        // Case 4: Partially overlapping spaces
        new WrappedTestCaseData<1, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/
             {IndexSpace<1, int>{Rect<1>{Point<1>(0), Point<1>(4)}},
              IndexSpace<1, int>{Rect<1>{Point<1>(3), Point<1>(6)}}},
             /*src_points=*/{Point<1, int>(2), Point<1, int>(3), Point<1, int>(4)},
             /*exp_points=*/
             {
                 {Point<1, int>(2), Point<1, int>(3), Point<1, int>(4)},
             }}),

        // Case 5: 2D test with a single row
        new WrappedTestCaseData<2, int>({/*expected_iterations=*/1,
                                         /*bytes_per_element=*/4,
                                         /*spaces=*/
                                         {IndexSpace<2, int>{Rect<2, int>{
                                             Point<2, int>(0, 0), Point<2, int>(3, 0)}}},
                                         /*src_points=*/
                                         {Point<2, int>(0, 0), Point<2, int>(1, 0),
                                          Point<2, int>(2, 0), Point<2, int>(3, 0)},
                                         /*exp_points=*/
                                         {{Point<2, int>(0, 0), Point<2, int>(1, 0),
                                           Point<2, int>(2, 0), Point<2, int>(3, 0)}}}),

        // Case 6: 2D test with multiple rows
        new WrappedTestCaseData<2, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/
             {IndexSpace<2, int>{Rect<2, int>{Point<2, int>(0, 0), Point<2, int>(2, 2)}}},
             /*src_points=*/
             {Point<2, int>(0, 0), Point<2, int>(1, 1), Point<2, int>(2, 2),
              Point<2, int>(3, 3)},
             /*exp_points=*/
             {{Point<2, int>(0, 0), Point<2, int>(1, 1), Point<2, int>(2, 2)}}}),

        // Case 7: 2D test with all points outside (should produce no output)
        new WrappedTestCaseData<2, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/
             {IndexSpace<2, int>{
                 Rect<2, int>{Point<2, int>(10, 10), Point<2, int>(15, 15)}}},
             /*src_points=*/{Point<2, int>(1, 1), Point<2, int>(2, 2)},
             /*exp_points=*/{{}}}),

        // Case 8: 2D test with overlapping spaces
        new WrappedTestCaseData<2, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/
             {IndexSpace<2, int>{Rect<2, int>{Point<2, int>(0, 0), Point<2, int>(2, 2)}},
              IndexSpace<2, int>{Rect<2, int>{Point<2, int>(1, 1), Point<2, int>(3, 3)}}},
             /*src_points=*/{Point<2, int>(1, 1), Point<2, int>(2, 2)},
             /*exp_points=*/
             {
                 {Point<2, int>(1, 1), Point<2, int>(2, 2)},
             }}),

        // Case 9: 3D test with merging along Z, fails in Z
        new WrappedTestCaseData<3, int>(
            {/*expected_iterations=*/1,
             /*bytes_per_element=*/4,
             /*spaces=*/
             {IndexSpace<3, int>{
                 Rect<3, int>{Point<3, int>(0, 0, 0), Point<3, int>(10, 10, 10)}}},
             /*src_points=*/
             {Point<3, int>(1, 1, 1), Point<3, int>(1, 1, 2), Point<3, int>(1, 1, 4)},
             /*exp_points=*/
             {{Point<3, int>(1, 1, 1), Point<3, int>(1, 1, 2),
               Point<3, int>(1, 1, 4)}}})));
