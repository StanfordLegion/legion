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

  virtual void enqueue_ready_xd(XferDes *xd) { num_xds++; }

  int num_xds = 0;
};

struct AddressSplitFactoryTest : public ::testing::Test {};

TEST_F(AddressSplitFactoryTest, CreateXferDesLocal)
{
  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;
  XferDesRedopInfo redop_info;
  XferDesID guid = 0;
  NodeID launch_node = 0;
  NodeID target_node = 0;
  Node node_data;
  size_t bytes_per_element = 4;

  auto bgwork = std::make_unique<BackgroundWorkManager>();
  auto addrsplit_channel = new MockAddressSplitChannel(bgwork.get());
  std::vector<IndexSpace<1>> spaces(1);
  auto factory = new AddressSplitXferDesFactory<1, int>(bytes_per_element, spaces,
                                                        addrsplit_channel);
  factory->create_xfer_des(0, launch_node, target_node, guid, inputs_info, outputs_info,
                           0, redop_info, nullptr, 0, 0);
  factory->release();
}

TEST_F(AddressSplitFactoryTest, DISABLED_CreateXferDesRemote)
{
  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;
  XferDesRedopInfo redop_info;
  XferDesID guid = 0;
  NodeID launch_node = 1;
  NodeID target_node = 1;
  Node node_data;
  size_t bytes_per_element = 4;

  auto bgwork = std::make_unique<BackgroundWorkManager>();
  auto addrsplit_channel = new MockAddressSplitChannel(bgwork.get());
  std::vector<IndexSpace<1>> spaces(1);
  auto factory = new AddressSplitXferDesFactory<1, int>(bytes_per_element, spaces,
                                                        addrsplit_channel);
  factory->create_xfer_des(0, launch_node, target_node, guid, inputs_info, outputs_info,
                           0, redop_info, nullptr, 0, 0);
  factory->release();
}

template <int N, typename T>
struct AddressSplitXferDescTestCase {
  int expected_iterations = 1;
  size_t bytes_per_element = 4;
  std::vector<IndexSpace<N, T>> spaces;
  std::vector<Point<N, T>> src_points;
  std::vector<std::vector<Point<N, T>>> exp_points;
};

template <typename Param>
struct AddressSplitTest : public ::testing::Test {
  static std::vector<AddressSplitXferDescTestCase<std::tuple_element_t<0, Param>::value,
                                                  std::tuple_element_t<1, Param>>>
      _test_cases_;
};

TYPED_TEST_SUITE_P(AddressSplitTest);

template <int DIM, typename T>
class MockIterator : public TransferIterator {
public:
  MockIterator(int _max_iterations, size_t _total_bytes = 256)
    : max_iterations(_max_iterations)
    , total_bytes(_total_bytes)
  {}

  virtual Event request_metadata(void) { return Event::NO_EVENT; }

  virtual void reset(void) {}

  virtual bool done(void) { return iterations >= max_iterations; }
  virtual size_t step(size_t max_bytes, AddressInfo &info, unsigned flags,
                      bool tentative = false)
  {
    info.base_offset = offset;
    size_t bytes = std::min(max_bytes, total_bytes);
    offset += bytes;
    iterations++;
    return bytes;
  }

  virtual size_t step_custom(size_t max_bytes, AddressInfoCustom &info,
                             bool tentative = false)
  {
    assert(0);
    return 0;
  }

  virtual void confirm_step(void) {}
  virtual void cancel_step(void) {}

  virtual size_t get_base_offset(void) const { return 0; }

  virtual bool get_addresses(AddressList &addrlist,
                             const InstanceLayoutPieceBase *&nonaffine)
  {
    nonaffine = 0;
    return false;
  }

  virtual bool get_next_rect(Rect<DIM, T> &r, FieldID &fid, size_t &offset, size_t &fsize)
  {
    return false;
  }

  size_t max_iterations = 0;
  size_t iterations = 0;
  size_t offset = 0;
  size_t total_bytes = 0;
};

TYPED_TEST_P(AddressSplitTest, ProgressXD)
{
  constexpr int N = std::tuple_element_t<0, TypeParam>::value;
  using T = std::tuple_element_t<1, TypeParam>;

  for(const auto &test_case : AddressSplitTest<TypeParam>::_test_cases_) {
    auto bgwork = std::make_unique<BackgroundWorkManager>();
    auto addrsplit_channel = std::make_unique<MockAddressSplitChannel>(bgwork.get());

    std::vector<XferDesPortInfo> inputs_info;
    std::vector<XferDesPortInfo> outputs_info;

    XferDesRedopInfo redop_info;
    XferDesID guid = 0;
    int priority = 0;
    NodeID owner = 0;
    Node node_data;

    AddressSplitXferDes<N, T> xfer_des(0, addrsplit_channel.get(), owner, guid,
                                       inputs_info, outputs_info, priority,
                                       sizeof(Point<N, T>), test_case.spaces);

    xfer_des.input_ports.resize(1);
    XferDes::XferPort &input_port = xfer_des.input_ports[0];

    const size_t src_points = test_case.src_points.size();
    size_t total_src_bytes = sizeof(Point<N, T>) * src_points;
    Point<N, T> *src_buffer = new Point<N, T>[src_points];
    std::memcpy(src_buffer, test_case.src_points.data(), total_src_bytes);
    auto input_mem = std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, total_src_bytes,
                                                      0, Memory::SYSTEM_MEM, src_buffer);
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
      const size_t dst_points = test_case.exp_points[i].size();
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
      auto buffer = reinterpret_cast<Point<N, T> *>(buffs[i]->get_direct_ptr(
          0, test_case.exp_points[i].size() * sizeof(Point<N, T>)));
      for(size_t j = 0; j < test_case.exp_points[i].size(); j++) {
        EXPECT_EQ(buffer[j], test_case.exp_points[i][j]);
      }
    }

    addrsplit_channel->shutdown();
  }
}

REGISTER_TYPED_TEST_SUITE_P(AddressSplitTest, ProgressXD);

typedef ::testing::Types<std::tuple<std::integral_constant<int, 1>, int>,
                         std::tuple<std::integral_constant<int, 2>, long long>>
    MyTypes;

INSTANTIATE_TYPED_TEST_SUITE_P(My, AddressSplitTest, MyTypes);

template <>
std::vector<AddressSplitXferDescTestCase<1, int>>
    AddressSplitTest<std::tuple<std::integral_constant<int, 1>, int>>::_test_cases_ =
        {

            // Case 0
            AddressSplitXferDescTestCase<1, int>{
                .spaces = {IndexSpace<1, int>{Rect<1>{Point<1>(0), Point<1>(3)}}},
                .src_points = {Point<1, int>(0), Point<1, int>(1), Point<1, int>(2),
                               Point<1, int>(3)},
                .exp_points = {{Point<1, int>(0), Point<1, int>(1), Point<1, int>(2),
                                Point<1, int>(3)}},
            },

            // Case 1 - partial overlap with a single index space
            AddressSplitXferDescTestCase<1, int>{
                .spaces = {IndexSpace<1, int>{Rect<1>{Point<1>(2), Point<1>(6)}}},
                .src_points = {Point<1, int>(0), Point<1, int>(1), Point<1, int>(2),
                               Point<1, int>(3)},
                .exp_points = {{Point<1, int>(2), Point<1, int>(3)}},
            },

            // Case 2
            AddressSplitXferDescTestCase<1, int>{
                .spaces =
                    {
                        IndexSpace<1, int>{Rect<1>{Point<1>(0), Point<1>(2)}},
                        IndexSpace<1, int>{Rect<1>{Point<1>(3), Point<1>(6)}},
                    },
                .src_points = {Point<1, int>(1), Point<1, int>(2), Point<1, int>(3),
                               Point<1, int>(4)},
                .exp_points =
                    {
                        {Point<1, int>(1), Point<1, int>(2)},
                        {Point<1, int>(3), Point<1, int>(4)},
                    },
            },

};

template <>
std::vector<AddressSplitXferDescTestCase<2, long long>> AddressSplitTest<
    std::tuple<std::integral_constant<int, 2>, long long>>::_test_cases_ = {

    // Case 3
    AddressSplitXferDescTestCase<2, long long>{
        .spaces = {IndexSpace<2, long long>{
            Rect<2, long long>{Point<2, long long>(0, 0), Point<2, long long>(3, 0)}}},
        .src_points = {Point<2, long long>(0, 0), Point<2, long long>(1, 0),
                       Point<2, long long>(2, 0), Point<2, long long>(3, 0)},
        .exp_points = {{Point<2, long long>(0, 0), Point<2, long long>(1, 0),
                        Point<2, long long>(2, 0), Point<2, long long>(3, 0)}},
    }};
