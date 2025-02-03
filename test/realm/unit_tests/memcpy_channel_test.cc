#include "realm/transfer/channel.h"
#include "realm/transfer/memcpy_channel.h"
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
      new MemcpyChannel(bgwork, &node, remote_shared_memory_mappings));
  NodeID channel_owner = channel->node;
  XferDesKind channel_kind = channel->kind;
  auto paths = channel->get_paths();

  ASSERT_EQ(channel_kind, XferDesKind::XFER_MEM_CPY);
  ASSERT_EQ(channel_owner, owner);
  ASSERT_TRUE(paths.empty());
  channel->shutdown();
}

static inline Memory make_mem(int idx, int node_id)
{
  return ID::make_memory(idx, node_id).convert<Memory>();
}

TEST(MemcpyChannelTest, SupportsPathRemoteSharedMemories)
{
  Node node;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  BackgroundWorkManager *bgwork = new BackgroundWorkManager();
  constexpr size_t bytes = 16;
  std::vector<std::byte> buffer(bytes);
  auto src_mem =
      new LocalCPUMemory(make_mem(0, 0), bytes, 0, Memory::SYSTEM_MEM, buffer.data());
  node.memories.push_back(src_mem);

  std::vector<std::byte> buffer1(bytes);
  auto dst_mem_1 =
      new LocalCPUMemory(make_mem(0, 1), bytes, 0, Memory::SYSTEM_MEM, buffer1.data());

  std::vector<std::byte> buffer2(bytes);
  auto dst_mem_2 =
      new LocalCPUMemory(make_mem(0, 2), bytes, 0, Memory::SYSTEM_MEM, buffer2.data());

  std::vector<std::byte> buffer3(bytes);
  auto dst_mem_3 =
      new LocalCPUMemory(make_mem(0, 3), bytes, 0, Memory::SYSTEM_MEM, buffer3.data());

  remote_shared_memory_mappings.insert({dst_mem_1->me.id, SharedMemoryInfo()});
  remote_shared_memory_mappings.insert({dst_mem_2->me.id, SharedMemoryInfo()});

  uint64_t cost_1 = 0, cost_2 = 0, cost_3 = 0;

  std::unique_ptr<Channel> channel(
      new MemcpyChannel(bgwork, &node, remote_shared_memory_mappings));

  cost_1 =
      channel->supports_path(ChannelCopyInfo(src_mem->me, dst_mem_1->me),
                             /*src_serdez_id=*/0, /*dst_serdez_id=*/0,
                             /*redop_id=*/0, bytes, /*src_frangs=*/0, /*dst_frags=*/0);

  cost_2 =
      channel->supports_path(ChannelCopyInfo(src_mem->me, dst_mem_2->me),
                             /*src_serdez_id=*/0, /*dst_serdez_id=*/0,
                             /*redop_id=*/0, bytes, /*src_frangs=*/0, /*dst_frags=*/0);

  cost_3 =
      channel->supports_path(ChannelCopyInfo(src_mem->me, dst_mem_3->me),
                             /*src_serdez_id=*/0, /*dst_serdez_id=*/0,
                             /*redop_id=*/0, bytes, /*src_frangs=*/0, /*dst_frags=*/0);

  ASSERT_EQ(cost_1, 100);
  ASSERT_EQ(cost_2, 100);
  ASSERT_EQ(cost_3, 0);
  ASSERT_FALSE(channel->get_paths().empty());

  delete dst_mem_1;
  delete dst_mem_2;
  delete dst_mem_3;

  channel->shutdown();
}

TEST(MemcpyChannelTest, SupportsPathLocalMemories)
{
  Node node;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  BackgroundWorkManager *bgwork = new BackgroundWorkManager();
  constexpr size_t bytes = 16;
  std::vector<std::byte> buffer(bytes);
  auto src_mem =
      new LocalCPUMemory(make_mem(0, 0), bytes, 0, Memory::SYSTEM_MEM, buffer.data());

  std::vector<std::byte> buffer1(bytes);
  auto dst_mem_1 =
      new LocalCPUMemory(make_mem(1, 1), bytes, 0, Memory::SYSTEM_MEM, buffer1.data());

  std::vector<std::byte> buffer2(bytes);
  auto dst_mem_2 =
      new LocalCPUMemory(make_mem(2, 2), bytes, 0, Memory::SYSTEM_MEM, buffer2.data());

  std::vector<std::byte> buffer3(bytes);
  auto dst_mem_3 =
      new LocalCPUMemory(make_mem(3, 3), bytes, 0, Memory::SYSTEM_MEM, buffer3.data());

  node.memories.push_back(src_mem);
  node.memories.push_back(dst_mem_1);
  node.memories.push_back(dst_mem_2);
  uint64_t cost_1 = 0, cost_2 = 0, cost_3 = 0;

  std::unique_ptr<Channel> channel(
      new MemcpyChannel(bgwork, &node, remote_shared_memory_mappings));

  cost_1 =
      channel->supports_path(ChannelCopyInfo(src_mem->me, dst_mem_1->me),
                             /*src_serdez_id=*/0, /*dst_serdez_id=*/0,
                             /*redop_id=*/0, bytes, /*src_frangs=*/0, /*dst_frags=*/0);

  cost_2 =
      channel->supports_path(ChannelCopyInfo(src_mem->me, dst_mem_2->me),
                             /*src_serdez_id=*/0, /*dst_serdez_id=*/0,
                             /*redop_id=*/0, bytes, /*src_frangs=*/0, /*dst_frags=*/0);

  cost_3 =
      channel->supports_path(ChannelCopyInfo(src_mem->me, dst_mem_3->me),
                             /*src_serdez_id=*/0, /*dst_serdez_id=*/0,
                             /*redop_id=*/0, bytes, /*src_frangs=*/0, /*dst_frags=*/0);

  ASSERT_EQ(cost_1, 100);
  ASSERT_EQ(cost_2, 100);
  ASSERT_EQ(cost_3, 0);
  ASSERT_FALSE(channel->get_paths().empty());

  // we only need to delete memories are not part of the 'node' since pointers will be
  // deleted for us
  delete dst_mem_3;
  channel->shutdown();
}

struct MemcpyXferDescTestCase {
  std::vector<size_t> src_strides;
  std::vector<size_t> src_extents;
  std::vector<size_t> dst_strides;
  std::vector<size_t> dst_extents;
  int max_entries = 1;
};

size_t fill_address_list(const std::vector<size_t> &strides,
                         const std::vector<size_t> &extents, AddressList &addrlist,
                         size_t offset = 0)
{
  int num_dims = static_cast<int>(strides.size());
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

  // offset of initial entry is easy to compute
  addr_data[1] = offset;
  addr_data[0] = (bytes << 4) + cur_dim;
  addrlist.commit_nd_entry(cur_dim, total_bytes);

  return bytes;
}

// TODO(apryakhin): Move under test utils
template <int DIM, typename T>
class MockIterator : public TransferIterator {
public:
  MockIterator(const std::vector<size_t> &_strides, const std::vector<size_t> &_extents,
               int _max_entries)
    : strides(_strides)
    , extents(_extents)
    , max_entries(_max_entries)
  {}

  virtual Event request_metadata(void) { return Event::NO_EVENT; }

  virtual void reset(void) {}

  virtual bool done(void) { return entries >= max_entries; }
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
    size_t offset = 0;
    while(!done()) {
      offset += fill_address_list(strides, extents, addrlist, offset);
      entries++;
    }
    return true;
  }

  virtual bool get_next_rect(Rect<DIM, T> &r, FieldID &fid, size_t &offset, size_t &fsize)
  {
    return false;
  }

  std::vector<size_t> strides;
  std::vector<size_t> extents;
  int entries = 0;
  int max_entries = 1;
  size_t offset = 0;
  size_t total_bytes = 0;
};

class MemcpyXferDescParamTest : public ::testing::TestWithParam<MemcpyXferDescTestCase> {
protected:
  void SetUp() override
  {
    bgwork = new BackgroundWorkManager();
    channel = new MemcpyChannel(bgwork, &node_data, remote_shared_memory_mappings);
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
    for(size_t dim = 1; dim < strides.size(); dim++) {
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

  // TODO:(apryakhin@:): find a better way to populate the address list
  // and check all bytes
  AddressList addrlist;
  size_t check_bytes =
      fill_address_list(test_case.dst_strides, test_case.dst_extents, addrlist);

  xfer_desc->input_ports.resize(1);
  XferDes::XferPort &input_port = xfer_desc->input_ports[0];

  size_t total_src_bytes = 0;
  for(int i = 0; i < test_case.max_entries; i++) {
    size_t src_bytes = configure_port(input_port, test_case.src_strides,
                                      test_case.src_extents, total_src_bytes);
    total_src_bytes += src_bytes;
  }

  std::vector<std::byte> src_buffer(total_src_bytes, std::byte(7));
  auto input_mem = std::make_unique<LocalCPUMemory>(
      Memory::NO_MEMORY, total_src_bytes, 0, Memory::SYSTEM_MEM, src_buffer.data());
  input_port.mem = input_mem.get();
  input_port.peer_port_idx = 0;
  input_port.iter = new MockIterator<1, int>(test_case.src_strides, test_case.src_extents,
                                             test_case.max_entries);
  input_port.peer_guid = XferDes::XFERDES_NO_GUID;
  input_port.addrcursor.set_addrlist(&input_port.addrlist);

  xfer_desc->output_ports.resize(1);
  XferDes::XferPort &output_port = xfer_desc->output_ports[0];

  size_t total_dst_bytes = 0;
  for(int i = 0; i < test_case.max_entries; i++) {
    size_t dst_bytes = configure_port(output_port, test_case.dst_strides,
                                      test_case.dst_extents, total_dst_bytes);
    total_dst_bytes += dst_bytes;
  }

  std::vector<std::byte> dst_buffer(total_dst_bytes, std::byte(1));
  auto output_mem = std::make_unique<LocalCPUMemory>(
      Memory::NO_MEMORY, total_dst_bytes, 0, Memory::SYSTEM_MEM, dst_buffer.data());
  output_port.mem = output_mem.get();
  output_port.peer_port_idx = 0;
  output_port.iter = new MockIterator<1, int>(
      test_case.dst_strides, test_case.dst_extents, test_case.max_entries);
  output_port.addrcursor.set_addrlist(&output_port.addrlist);

  while(xfer_desc->progress_xd(channel, TimeLimit::relative(10000000))) {
  }

  for(size_t i = 0; i < check_bytes * test_case.max_entries; i++) {
    ASSERT_EQ(src_buffer[i], dst_buffer[i]);
  }
}

const static MemcpyXferDescTestCase kMemcpyXferDescTestCases[] = {
    // Case 1: 1D regular copy
    {.src_strides = {4},
     .src_extents = {4},
     .dst_strides = {4},
     .dst_extents = {4},
     .max_entries = 1},
    // Case 2: 2D regular copy with uniform strides
    {.src_strides = {16, 4},
     .src_extents = {4, 4},
     .dst_strides = {16, 4},
     .dst_extents = {4, 4},
     .max_entries = 1},
    // Case 3: 2D irregular copy with non-uniform strides
    {.src_strides = {20, 5},
     .src_extents = {4, 5},
     .dst_strides = {20, 5},
     .dst_extents = {4, 5},
     .max_entries = 1},
    // case 3a: 2D transpose
    {.src_strides = {4, 16},
     .src_extents = {4, 4},
     .dst_strides = {16, 4},
     .dst_extents = {4, 4}},
    // Case 4: 3D regular copy with uniform strides
    {.src_strides = {64, 16, 4},
     .src_extents = {4, 4, 4},
     .dst_strides = {64, 16, 4},
     .dst_extents = {4, 4, 4},
     .max_entries = 1},
    // Case 5: 3D irregular copy with non-uniform strides
    {.src_strides = {70, 17, 5},
     .src_extents = {4, 5, 6},
     .dst_strides = {70, 17, 5},
     .dst_extents = {4, 5, 6},
     .max_entries = 1},
    // Case 6: Sparse copy with gaps in strides
    {.src_strides = {128, 32},
     .src_extents = {4, 8},
     .dst_strides = {132, 34},
     .dst_extents = {4, 8},
     .max_entries = 2},
    // Case 7: Copy with small extents and high stride
    {.src_strides = {64},
     .src_extents = {2},
     .dst_strides = {64},
     .dst_extents = {2},
     .max_entries = 1},
    // Case 8: Multi-entry copy
    {.src_strides = {4},
     .src_extents = {4},
     .dst_strides = {4},
     .dst_extents = {4},
     .max_entries = 4}};

INSTANTIATE_TEST_SUITE_P(Foo, MemcpyXferDescParamTest,
                         testing::ValuesIn(kMemcpyXferDescTestCases));
