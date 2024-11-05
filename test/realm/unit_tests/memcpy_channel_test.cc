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
  std::vector<Memory> memories;
};

class MemcpyXferDescParamTest : public ::testing::TestWithParam<MemcpyXferDescTestCase> {
};

template <int N, typename T>
class TransferIteratorMock : public TransferIterator {
public:
  TransferIteratorMock(void) {}
  TransferIteratorMock(RegionInstanceImpl *_inst_impl, const int _dim_order[N]) {}

  virtual Event request_metadata(void) { return Event::NO_EVENT; }

  virtual void reset(void) {}

  virtual bool done(void) { return false; }
  virtual size_t step(size_t max_bytes, AddressInfo &info, unsigned flags,
                      bool tentative = false)
  {
    assert(0);
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
    return false;
  }

protected:
  virtual bool get_next_rect(Rect<N, T> &r, FieldID &fid, size_t &offset, size_t &fsize)
  {
    assert(0);
    return false;
  }

  bool have_rect, is_done;
  Rect<N, T> cur_rect;
  FieldID cur_field_id;
  size_t cur_field_offset, cur_field_size;
  Point<N, T> cur_point, next_point;
  bool carry;

  RegionInstanceImpl *inst_impl;
  // InstanceLayout<N, T> *inst_layout;
  size_t inst_offset;
  bool tentative_valid;
  int dim_order[N];
};

TEST_P(MemcpyXferDescParamTest, ProgresXD)
{
  // ChannelTestCase test_case = GetParam();
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

  size_t alloc_bytes = 16;
  void *input_alloc_base = malloc(alloc_bytes);
  MemoryImpl *input_mem = new LocalCPUMemory(Memory::NO_MEMORY, alloc_bytes, 0,
                                             Memory::SYSTEM_MEM, input_alloc_base);

  void *output_alloc_base = malloc(alloc_bytes);
  MemoryImpl *output_mem = new LocalCPUMemory(Memory::NO_MEMORY, alloc_bytes, 0,
                                              Memory::SYSTEM_MEM, output_alloc_base);

  const size_t dim = 1;
  const size_t bytes = alloc_bytes;

  AddressList inaddrlist;
  size_t *in_addr_data = inaddrlist.begin_nd_entry(dim);
  in_addr_data[0] = (bytes << 4) + dim;
  inaddrlist.commit_nd_entry(dim, bytes);

  AddressListCursor in_addrcursor;
  in_addrcursor.set_addrlist(&inaddrlist);

  XferDes::XferPort input_port;
  input_port.mem = input_mem;
  input_port.peer_port_idx = 0;
  input_port.addrlist = inaddrlist;
  input_port.addrcursor = in_addrcursor;
  input_port.iter = new TransferIteratorMock<1, int>();
  xfer_desc->input_ports.push_back(input_port);

  AddressList outaddrlist;
  size_t *out_addr_data = outaddrlist.begin_nd_entry(dim);
  out_addr_data[0] = (bytes << 4) + dim;
  outaddrlist.commit_nd_entry(dim, bytes);

  AddressListCursor out_addrcursor;
  out_addrcursor.set_addrlist(&outaddrlist);

  XferDes::XferPort output_port;
  output_port.mem = output_mem;
  output_port.peer_port_idx = 0;
  output_port.iter = new TransferIteratorMock<1, int>();
  output_port.addrlist = outaddrlist;
  output_port.addrcursor = out_addrcursor;
  output_port.peer_guid = XferDes::XFERDES_NO_GUID;
  xfer_desc->output_ports.push_back(output_port);


  //xfer_desc->output_control.current_io_port = -1;

  auto memcpy_xfer_desc = reinterpret_cast<MemcpyXferDes *>(xfer_desc);
  memcpy_xfer_desc->progress_xd(&channel, TimeLimit::responsive());

  channel.shutdown();
}

const static MemcpyXferDescTestCase kMemcpyXferDescTestCases[] = {
    MemcpyXferDescTestCase{},
};

INSTANTIATE_TEST_SUITE_P(Foo, MemcpyXferDescParamTest,
                         testing::ValuesIn(kMemcpyXferDescTestCases));
