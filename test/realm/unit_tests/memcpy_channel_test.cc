#include "realm/transfer/channel.h"
#include <tuple>
#include <gtest/gtest.h>

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

struct ChannelTestCase {
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
                         testing::ValuesIn(kMemcpyChannelTestCases));

struct MemcpyXferDescTestCase {
  std::vector<size_t> src_strides;
  std::vector<size_t> src_extents;

  std::vector<size_t> dst_strides;
  std::vector<size_t> dst_extents;
};

class MemcpyXferDescParamTest : public ::testing::TestWithParam<MemcpyXferDescTestCase> {
public:
  size_t set_port(XferDes::XferPort &port, const std::vector<size_t> &strides,
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

template <int N, typename T>
class TransferIteratorMock : public TransferIterator {
public:
  TransferIteratorMock(void) {}
  TransferIteratorMock(RegionInstanceImpl *_inst_impl, const int _dim_order[N]) {}

  virtual Event request_metadata(void) { return Event::NO_EVENT; }

  virtual void reset(void) {}

  virtual bool done(void) { return true; }
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
    return true;
  }

protected:
  virtual bool get_next_rect(Rect<N, T> &r, FieldID &fid, size_t &offset, size_t &fsize)
  {
    assert(0);
    return false;
  }
};

TEST_P(MemcpyXferDescParamTest, ProgresXD)
{
  auto test_case = GetParam();

  NodeID owner = 0;
  Node node;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  BackgroundWorkManager *bgwork = new BackgroundWorkManager();
  MemcpyChannel channel(bgwork, &node, remote_shared_memory_mappings, owner);

  NodeID launch_node = owner;
  XferDesID guid = 0;
  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;
  int priority = 0;
  XferDesRedopInfo redop_info;

  XferDes *xfer_desc =
      channel.create_xfer_des(0, launch_node, guid, inputs_info, outputs_info, priority,
                              redop_info, nullptr, 0, 0);

  xfer_desc->input_ports.resize(1);
  XferDes::XferPort &input_port = xfer_desc->input_ports[0];

  size_t src_bytes = set_port(input_port, test_case.src_strides, test_case.src_extents);
  std::vector<char> in_buffer(src_bytes, 7);
  void *input_alloc_base = in_buffer.data();
  MemoryImpl *input_mem = new LocalCPUMemory(Memory::NO_MEMORY, src_bytes, 0,
                                             Memory::SYSTEM_MEM, input_alloc_base);

  input_port.mem = input_mem;
  input_port.peer_port_idx = 0;

  input_port.iter = new TransferIteratorMock<1, int>();
  input_port.peer_guid = XferDes::XFERDES_NO_GUID;

  xfer_desc->output_ports.resize(1);
  XferDes::XferPort &output_port = xfer_desc->output_ports[0];

  size_t dst_bytes = set_port(output_port, test_case.dst_strides, test_case.dst_extents);
  std::vector<char> out_buffer(dst_bytes, 7);
  void *output_alloc_base = out_buffer.data();
  MemoryImpl *output_mem = new LocalCPUMemory(Memory::NO_MEMORY, dst_bytes, 0,
                                              Memory::SYSTEM_MEM, output_alloc_base);

  output_port.mem = output_mem;
  output_port.peer_port_idx = 0;
  output_port.iter = new TransferIteratorMock<1, int>();

  auto memcpy_xfer_desc = reinterpret_cast<MemcpyXferDes *>(xfer_desc);
  memcpy_xfer_desc->progress_xd(&channel, TimeLimit::responsive());

  for(size_t i = 0; i < in_buffer.size(); i++) {
    EXPECT_EQ(in_buffer[i], out_buffer[i]);
  }

  channel.shutdown();
}

const static MemcpyXferDescTestCase kMemcpyXferDescTestCases[] = {
    // Case 1
    MemcpyXferDescTestCase{
        .src_strides = {4}, .src_extents = {4}, .dst_strides = {4}, .dst_extents = {4}}};

INSTANTIATE_TEST_SUITE_P(Foo, MemcpyXferDescParamTest,
                         testing::ValuesIn(kMemcpyXferDescTestCases));
