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

struct MemcpyXferDescTestCase {
  std::vector<size_t> src_strides;
  std::vector<size_t> src_extents;
  std::vector<size_t> dst_strides;
  std::vector<size_t> dst_extents;
  int expected_iterations = 1;
};

size_t fill_address_list(const std::vector<size_t> &strides,
                         const std::vector<size_t> &extents, AddressList &addrlist)
{
  size_t num_dims = strides.size();
  size_t *addr_data = addrlist.begin_nd_entry(num_dims);
  size_t bytes = strides[0];
  int cur_dim = 1, di = 0;
  for(; di < num_dims; di++) {
    int d = di;
    if(strides[d] != bytes)
      break;
    bytes *= extents[d];
  }

  size_t total_bytes = bytes;
  while(di < num_dims) {
    size_t total_count = 1;
    size_t stride = strides[di];

    for(; di < num_dims; di++) {
      int d = di;
      size_t count = extents[d];
      if(strides[d] != (stride * total_count))
        break;
      total_count *= count;
    }

    addr_data[cur_dim * 2] = total_count;
    addr_data[cur_dim * 2 + 1] = stride;
    total_bytes *= total_count;
    cur_dim++;
  }

  addr_data[0] = (bytes << 4) + cur_dim;
  addrlist.commit_nd_entry(cur_dim, total_bytes);

  return bytes;
}

// TODO(apryakhin): Move under test utils
template <int DIM, typename T>
class TransferIteratorMock : public TransferIterator {
public:
  TransferIteratorMock(const std::vector<size_t> &_strides,
                       const std::vector<size_t> &_extents, int _max_iterations)
    : strides(_strides)
    , extents(_extents)
    , max_iterations(_max_iterations)
  {}

  virtual Event request_metadata(void) { return Event::NO_EVENT; }

  virtual void reset(void) {}

  virtual bool done(void) { return iterations >= max_iterations; }
  virtual size_t step(size_t max_bytes, AddressInfo &info, unsigned flags,
                      bool tentative = false)
  {
    assert(0);
    return 0;
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
    fill_address_list(strides, extents, addrlist);
    return (++iterations >= max_iterations);
  }

  virtual bool get_next_rect(Rect<DIM, T> &r, FieldID &fid, size_t &offset, size_t &fsize)
  {
    return false;
  }

  std::vector<size_t> strides;
  std::vector<size_t> extents;
  size_t max_iterations = 0;
  size_t iterations = 0;
  size_t offset = 0;
  size_t total_bytes = 0;
};

class MemcpyXferDescParamTest : public ::testing::TestWithParam<MemcpyXferDescTestCase> {
protected:
  void SetUp() override
  {
    bgwork = new BackgroundWorkManager();
    channel = new MemcpyChannel(bgwork, &node_data, remote_shared_memory_mappings, owner);
  }

  void TearDown() override
  {
    channel->shutdown();
    delete channel;
    delete bgwork;
  }

  size_t configure_port(XferDes::XferPort &port, const std::vector<size_t> &strides,
                        const std::vector<size_t> &extents, size_t offset)
  {
    size_t bytes = strides[0] * extents[0];
    for(int dim = 1; dim < strides.size(); dim++) {
      size_t count = extents[dim];
      bytes *= count;
    }
    return bytes;
  }

  XferDesID guid = 0;
  int priority = 0;
  NodeID owner = 0;
  Node node_data;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  XferDesRedopInfo redop_info;
  BackgroundWorkManager *bgwork;
  MemcpyChannel *channel;
};

TEST_P(MemcpyXferDescParamTest, ProgresXD)
{
  auto test_case = GetParam();

  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;

  auto xfer_desc = std::unique_ptr<MemcpyXferDes>(dynamic_cast<MemcpyXferDes *>(
      channel->create_xfer_des(0, owner, guid, inputs_info, outputs_info, priority,
                               redop_info, nullptr, 0, 0)));

  xfer_desc->input_ports.resize(1);
  XferDes::XferPort &input_port = xfer_desc->input_ports[0];

  size_t total_src_bytes = 0;
  for(int i = 0; i < test_case.expected_iterations; i++) {
    size_t src_bytes = configure_port(input_port, test_case.src_strides,
                                      test_case.src_extents, total_src_bytes);
    total_src_bytes += src_bytes;
  }

  char *in_buffer = new char[total_src_bytes];
  std::memset(in_buffer, 7, total_src_bytes);
  auto input_mem = std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, total_src_bytes, 0,
                                                    Memory::SYSTEM_MEM, in_buffer);
  input_port.mem = input_mem.get();
  input_port.peer_port_idx = 0;
  input_port.iter = new TransferIteratorMock<1, int>(
      test_case.src_strides, test_case.src_extents, test_case.expected_iterations);
  input_port.peer_guid = XferDes::XFERDES_NO_GUID;
  input_port.addrcursor.set_addrlist(&input_port.addrlist);

  xfer_desc->output_ports.resize(1);
  XferDes::XferPort &output_port = xfer_desc->output_ports[0];

  size_t total_dst_bytes = 0;
  for(int i = 0; i < test_case.expected_iterations; i++) {
    size_t dst_bytes = configure_port(output_port, test_case.dst_strides,
                                      test_case.dst_extents, total_dst_bytes);
    total_dst_bytes += dst_bytes;
  }

  char *out_buffer = new char[total_dst_bytes];
  std::memset(out_buffer, 1, total_dst_bytes);
  auto output_mem = std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, total_dst_bytes,
                                                     0, Memory::SYSTEM_MEM, out_buffer);
  output_port.mem = output_mem.get();
  output_port.peer_port_idx = 0;
  output_port.iter = new TransferIteratorMock<1, int>(
      test_case.dst_strides, test_case.dst_extents, test_case.expected_iterations);
  output_port.addrcursor.set_addrlist(&output_port.addrlist);

  int iterations = 0;
  while(xfer_desc->progress_xd(channel, TimeLimit::responsive())) {
    iterations++;
  }

  EXPECT_EQ(iterations, test_case.expected_iterations);

  // TODO:(apryakhin@:): find a better way to populate the address list
  // and check all bytes
  AddressList addrlist;
  size_t check_bytes =
      fill_address_list(test_case.dst_strides, test_case.dst_extents, addrlist);

  for(size_t i = 0; i < check_bytes; i++) {
    EXPECT_EQ(in_buffer[i], out_buffer[i]);
  }
}

const static MemcpyXferDescTestCase kMemcpyXferDescTestCases[] = {
    // Case 1 - 1D
    MemcpyXferDescTestCase{
        .src_strides = {4}, .src_extents = {4}, .dst_strides = {4}, .dst_extents = {4}},

    // Case 2 - 2D
    MemcpyXferDescTestCase{.src_strides = {4, 16},
                           .src_extents = {4, 4},
                           .dst_strides = {4, 16},
                           .dst_extents = {4, 4}},

    // Case 3 - 2D
    MemcpyXferDescTestCase{.src_strides = {16, 4},
                           .src_extents = {4, 4},
                           .dst_strides = {16, 4},
                           .dst_extents = {4, 4}},

    // Case 3 - 3D
    MemcpyXferDescTestCase{.src_strides = {256, 16, 4},
                           .src_extents = {4, 4, 4},
                           .dst_strides = {256, 16, 4},
                           .dst_extents = {4, 4, 4}},
};

INSTANTIATE_TEST_SUITE_P(Foo, MemcpyXferDescParamTest,
                         testing::ValuesIn(kMemcpyXferDescTestCases));
