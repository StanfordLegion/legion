#include "realm/transfer/memcpy_channel.h"
#include <tuple>
#include <gtest/gtest.h>
#include <cstring>

using namespace Realm;

TEST(MemcpyChannelTest, CreateMemcpyChannel)
{
  NodeID owner = 0;
  Node node;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  BackgroundWorkManager *bgwork = new BackgroundWorkManager();

  std::unique_ptr<Channel> channel(
      new MemcpyChannel(bgwork, &node, remote_shared_memory_mappings, owner));
  NodeID channel_owner = channel->node;
  XferDesKind channel_kind = channel->kind;
  auto paths = channel->get_paths();

  EXPECT_EQ(channel_kind, XferDesKind::XFER_MEM_CPY);
  EXPECT_EQ(channel_owner, owner);
  EXPECT_TRUE(paths.empty());
  channel->shutdown();
}

/*struct ChannelTestCase {
  std::vector<Memory> memories;
};

class MemcpyChannelParamTest : public ::testing::TestWithParam<ChannelTestCase> {};

TEST_P(MemcpyChannelParamTest, MemcpyChannelCreateXferDes)
{
  // ChannelTestCase test_case = GetParam();
  NodeID owner = 0;
  Node node;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  BackgroundWorkManager *bgwork = new BackgroundWorkManager();
  MemcpyChannel channel(bgwork, &node, remote_shared_memory_mappings, owner);

  channel.shutdown();
}

const static ChannelTestCase kMemcpyChannelTestCases[] = {
    ChannelTestCase{},
};

INSTANTIATE_TEST_SUITE_P(Foo, MemcpyChannelParamTest,
                         testing::ValuesIn(kMemcpyChannelTestCases));*/

// TODO(apryakhin): Move under test utils
template <int N, typename T>
class TransferIteratorMock : public TransferIterator {
public:
  TransferIteratorMock(void) {}

  virtual Event request_metadata(void) { return Event::NO_EVENT; }

  virtual void reset(void) {}

  virtual bool done(void) { return true; }
  virtual size_t step(size_t max_bytes, AddressInfo &info, unsigned flags,
                      bool tentative = false)
  {
    return 0;
  }

  virtual size_t step_custom(size_t max_bytes, AddressInfoCustom &info,
                             bool tentative = false)
  {
    return 0;
  }

  virtual void confirm_step(void) {}
  virtual void cancel_step(void) {}

  virtual size_t get_base_offset(void) const { return 0; }

  virtual bool get_addresses(AddressList &addrlist,
                             const InstanceLayoutPieceBase *&nonaffine)
  {
    nonaffine = 0;
    return true;
  }

  virtual bool get_next_rect(Rect<N, T> &r, FieldID &fid, size_t &offset, size_t &fsize)
  {
    return false;
  }
};

struct MemcpyXferDescTestCase {
  std::vector<size_t> src_strides;
  std::vector<size_t> src_extents;
  std::vector<size_t> dst_strides;
  std::vector<size_t> dst_extents;
  int expected_iterations = 0;
};

class MemcpyXferDescParamTest : public ::testing::TestWithParam<MemcpyXferDescTestCase> {
public:
  size_t configure_port(XferDes::XferPort &port, const std::vector<size_t> &strides,
                        const std::vector<size_t> &extents)
  {
    size_t bytes = strides[0] * extents[0];
    size_t *addr_data = port.addrlist.begin_nd_entry(strides.size());

    for(int dim = 1; dim < strides.size(); dim++) {
      size_t count = extents[dim];
      addr_data[dim * 2] = count;
      addr_data[dim * 2 + 1] = strides[dim];
      bytes *= count;
    }

    addr_data[0] = (bytes << 4) + strides.size();
    port.addrlist.commit_nd_entry(strides.size(), bytes);
    port.addrcursor.set_addrlist(&port.addrlist);

    return bytes;
  }
};

TEST_P(MemcpyXferDescParamTest, ProgresXD)
{
  auto test_case = GetParam();

  NodeID owner = 0;
  Node node_data;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  // TODO(apryakhin@): Consider mocking bgwork
  auto bgwork = std::make_unique<BackgroundWorkManager>();
  MemcpyChannel channel(bgwork.get(), &node_data, remote_shared_memory_mappings, owner);

  XferDesID guid = 0;
  int priority = 0;
  XferDesRedopInfo redop_info;
  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;

  auto xfer_desc = std::unique_ptr<MemcpyXferDes>(dynamic_cast<MemcpyXferDes *>(
      channel.create_xfer_des(0, owner, guid, inputs_info, outputs_info, priority,
                              redop_info, nullptr, 0, 0)));

  xfer_desc->input_ports.resize(1);
  XferDes::XferPort &input_port = xfer_desc->input_ports[0];

  size_t src_bytes =
      configure_port(input_port, test_case.src_strides, test_case.src_extents);
  char *in_buffer = new char[src_bytes];
  std::memset(in_buffer, 7, src_bytes);
  auto input_mem = std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, src_bytes, 0,
                                                    Memory::SYSTEM_MEM, in_buffer);
  input_port.mem = input_mem.get();
  input_port.peer_port_idx = 0;
  input_port.iter = new TransferIteratorMock<1, int>();
  input_port.peer_guid = XferDes::XFERDES_NO_GUID;

  xfer_desc->output_ports.resize(1);
  XferDes::XferPort &output_port = xfer_desc->output_ports[0];

  size_t dst_bytes =
      configure_port(output_port, test_case.dst_strides, test_case.dst_extents);
  char *out_buffer = new char[dst_bytes];
  std::memset(out_buffer, 1, dst_bytes);
  auto output_mem = std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, dst_bytes, 0,
                                                     Memory::SYSTEM_MEM, out_buffer);
  output_port.mem = output_mem.get();
  output_port.peer_port_idx = 0;
  output_port.iter = new TransferIteratorMock<1, int>();

  int iterations = 0;
  while(!xfer_desc->progress_xd(&channel, TimeLimit::responsive())) {
    iterations++;
  }

  EXPECT_EQ(test_case.expected_iterations, iterations);
  for(size_t i = 0; i < src_bytes; i++) {
    EXPECT_EQ(in_buffer[i], out_buffer[i]);
  }

  channel.shutdown();
}

const static MemcpyXferDescTestCase kMemcpyXferDescTestCases[] = {
    // Case 1 - 1D
    MemcpyXferDescTestCase{
        .src_strides = {4}, .src_extents = {4}, .dst_strides = {4}, .dst_extents = {4}},
    // Case 2 - 2D
    MemcpyXferDescTestCase{
        .src_strides = {4, 16},
        .src_extents = {4, 4},
        .dst_strides = {4, 4},
        .dst_extents = {4, 16},
    }};

INSTANTIATE_TEST_SUITE_P(Foo, MemcpyXferDescParamTest,
                         testing::ValuesIn(kMemcpyXferDescTestCases));
