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

  EXPECT_EQ(channel_kind, XferDesKind::XFER_MEM_CPY);
  EXPECT_EQ(channel_owner, owner);
  EXPECT_TRUE(paths.empty());
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

  int iterations = 0;
  while(xfer_desc->progress_xd(channel, TimeLimit::relative(0))) {
    iterations++;
  }

  EXPECT_EQ(iterations, 1);

  for(size_t i = 0; i < check_bytes * test_case.max_entries; i++) {
    EXPECT_EQ(src_buffer[i], dst_buffer[i]);
  }
}

const static MemcpyXferDescTestCase kMemcpyXferDescTestCases[] = {
    MemcpyXferDescTestCase{
        .src_strides = {4}, .src_extents = {4}, .dst_strides = {4}, .dst_extents = {4}},

    MemcpyXferDescTestCase{.src_strides = {4},
                           .src_extents = {4},
                           .dst_strides = {4},
                           .dst_extents = {4},
                           .max_entries = 3},

    MemcpyXferDescTestCase{.src_strides = {4, 16},
                           .src_extents = {4, 4},
                           .dst_strides = {4, 16},
                           .dst_extents = {4, 4}},

    MemcpyXferDescTestCase{.src_strides = {16, 4},
                           .src_extents = {4, 4},
                           .dst_strides = {16, 4},
                           .dst_extents = {4, 4}},

    MemcpyXferDescTestCase{.src_strides = {256, 16, 4},
                           .src_extents = {4, 4, 4},
                           .dst_strides = {256, 16, 4},
                           .dst_extents = {4, 4, 4}},
};

INSTANTIATE_TEST_SUITE_P(Foo, MemcpyXferDescParamTest,
                         testing::ValuesIn(kMemcpyXferDescTestCases));
