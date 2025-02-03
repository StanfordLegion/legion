#include "realm/transfer/channel.h"
#include "realm/transfer/memcpy_channel.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace Realm;

static inline Memory make_mem(int idx, int node_id)
{
  return ID::make_memory(idx, node_id).convert<Memory>();
}

struct SupportsPathTestCase {
  int src_mem_id;
  int src_node_id;
  int dst_mem_id;
  int dst_node_id;
  bool is_remote_shared;
  bool is_local;
  uint64_t expected_cost;
  ReductionOpID redop_id = 0;
  bool src_serdez_id = 0;
  bool dst_serdez_id = 0;
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
};

class SupportsPathTest : public ::testing::TestWithParam<SupportsPathTestCase> {
protected:
  MemoryImpl *create_memory(int id, int node_id, size_t bytes)
  {
    return new LocalCPUMemory(make_mem(id, node_id), bytes, 0, Memory::SYSTEM_MEM,
                              nullptr);
  }

  Node node;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
};

TEST_P(SupportsPathTest, CheckSupportsPath)
{
  auto test_case = GetParam();
  constexpr size_t bytes = 16;

  auto src_mem = create_memory(test_case.src_mem_id, test_case.src_node_id, bytes);
  auto dst_mem = create_memory(test_case.dst_mem_id, test_case.dst_node_id, bytes);

  node.memories.push_back(src_mem);

  if(test_case.is_remote_shared) {
    remote_shared_memory_mappings.insert({dst_mem->me.id, SharedMemoryInfo()});
  }

  if(test_case.is_local) {
    node.memories.push_back(dst_mem);
  }

  BackgroundWorkManager *bgwork = new BackgroundWorkManager();
  std::unique_ptr<Channel> channel(
      new MemcpyChannel(bgwork, &node, remote_shared_memory_mappings));

  uint64_t cost = channel->supports_path(
      ChannelCopyInfo(src_mem->me, dst_mem->me), test_case.src_serdez_id,
      test_case.dst_serdez_id, test_case.redop_id, bytes,
      (test_case.src_frags.empty() ? nullptr : &test_case.src_frags),
      (test_case.dst_frags.empty() ? nullptr : &test_case.dst_frags));

  ASSERT_EQ(cost, test_case.expected_cost);

  if(!test_case.is_local) {
    delete dst_mem;
  }

  channel->shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    SupportsPathTests, SupportsPathTest,
    ::testing::Values(
        // Case 1: Local memory copy (expected cost: 100)
        SupportsPathTestCase{0, 0, 1, 0, false, true, 100},

        // Case 2: Remote shared memory copy (expected cost: 100)
        SupportsPathTestCase{0, 0, 1, 1, true, false, 100},

        // Case 3: Unreachable memory (expected cost: 0)
        SupportsPathTestCase{0, 0, 2, 2, false, false, 0},

        // Case 4: Another local memory (expected cost: 100)
        SupportsPathTestCase{0, 0, 3, 0, false, true, 100},

        // Case 5: Another remote shared memory (expected cost: 100)
        SupportsPathTestCase{0, 0, 4, 1, true, false, 100},

        // Case 6: Completely disconnected memory (expected cost: 0)
        SupportsPathTestCase{0, 0, 5, 2, false, false, 0},

        // Case 7: Local memory with reduction
        SupportsPathTestCase{0, 0, 1, 0, false, true, 0,
                             /*redop_id=*/1},

        // Case 8: Local memory with both src serdez
        SupportsPathTestCase{0, 0, 1, 0, false, true, 100,
                             /*redop_id=*/0, /*src_serdez_id=*/1,
                             /*dst_serdez_id=*/0},

        // Case 9: Local memory with dst serdez
        SupportsPathTestCase{0, 0, 1, 0, false, true, 100,
                             /*redop_id=*/0, /*src_serdez_id=*/0,
                             /*dst_serdez_id=*/1},

        // Case 10: Local memory with both src/dst serdez
        SupportsPathTestCase{0, 0, 1, 0, false, true, 0,
                             /*redop_id=*/0, /*src_serdez_id=*/1,
                             /*dst_serdez_id=*/1},

        // Case 11: Local memory with src frags
        SupportsPathTestCase{0, 0, 1, 0, false, true, 1000,
                             /*redop_id=*/0, /*src_serdez_id=*/0,
                             /*dst_serdez_id=*/0, /*src_frags=*/{10}},

        // Case 12: Local memory with src and dst frags
        SupportsPathTestCase{0, 0, 1, 0, false, true, 2000,
                             /*redop_id=*/0, /*src_serdez_id=*/0,
                             /*dst_serdez_id=*/0, /*src_frags=*/{10}, /*dst_frags=*/{20}}

        ));

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
