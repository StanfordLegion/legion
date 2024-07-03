#include "realm/transfer/transfer.h"

#include <tuple>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace Realm;

class MockChannel : public Channel {
public:
  MockChannel(XferDesKind _kind, int node)
    : Channel(_kind)
  {
    this->node = node;
  }

  MOCK_METHOD(uint64_t, supports_path,
              (ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
               CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
               const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
               XferDesKind *kind_ret, unsigned *bw_ret, unsigned *lat_ret),
              ());

  MOCK_METHOD(Memory, suggest_ib_memories, (Memory mem), (const));
  MOCK_METHOD(long, available, (), ());
  MOCK_METHOD(long, submit, (Request * *requests, long nr), ());
  MOCK_METHOD(void, pull, (), ());
  MOCK_METHOD(void, wakeup_xd, (XferDes * xd), ());
  MOCK_METHOD(void, enqueue_ready_xd, (XferDes * xd), ());
  MOCK_METHOD(XferDesFactory *, get_factory, (), ());
};

class MockAddressSplitChannel : public Channel {
public:
  MockAddressSplitChannel(XferDesKind _kind, int node)
    : Channel(_kind)
  {
    this->node = node;
  }

  MOCK_METHOD(uint64_t, supports_path,
              (ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
               CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
               const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
               XferDesKind *kind_ret, unsigned *bw_ret, unsigned *lat_ret),
              ());

  MOCK_METHOD(Memory, suggest_ib_memories, (Memory mem), (const));
  MOCK_METHOD(Memory, suggest_ib_memories_for_node, (NodeID node_id), (const));
  MOCK_METHOD(long, available, (), ());
  MOCK_METHOD(long, submit, (Request * *requests, long nr), ());
  MOCK_METHOD(void, pull, (), ());
  MOCK_METHOD(void, wakeup_xd, (XferDes * xd), ());
  MOCK_METHOD(void, enqueue_ready_xd, (XferDes * xd), ());
  MOCK_METHOD(XferDesFactory *, get_factory, (), ());
};

class GatherTest : public ::testing::Test {
protected:
  static MockAddressSplitChannel *mock_addrsplit_channel;

  static void SetUpTestSuite()
  {
    mock_addrsplit_channel = new MockAddressSplitChannel(XferDesKind::XFER_ADDR_SPLIT, 0);
  }

  static void TearDownTestSuite() { delete mock_addrsplit_channel; }

  void mock_supports_path(MockChannel *channel, Memory src, Memory dst, Memory ind,
                          bool oor_possible, uint64_t cost, size_t addr_size = 4,
                          size_t total_bytes = 16, bool direct = true)
  {
    ChannelCopyInfo info{src, dst, ind};
    info.is_direct = direct;
    info.addr_size = addr_size;
    info.oor_possible = oor_possible;
    EXPECT_CALL(*channel, supports_path(info, /*src_serdez_id=*/0, /*dst_serdez_id*/ 0,
                                        /*redop_id=*/0, total_bytes, testing::_,
                                        testing::_, testing::_, testing::_, testing::_))
        .Times(1)
        .WillRepeatedly(::testing::Return(cost));
  }
};

MockAddressSplitChannel *GatherTest::mock_addrsplit_channel;

TEST_F(GatherTest, GatherSingleSource1DSameNode)
{
  // ARRANGE

  const unsigned indirect_idx = 2;
  const unsigned src_field_start = 8;
  const unsigned src_field_count = 8;
  const size_t bytes_per_element = 8;

  // TODO: set actually some meaningfull costs
  const uint64_t exp_cost = 4;

  // Set instances, memories and node info
  RegionInstance src_inst =
      ID::make_instance(/*owner=*/0, /*creator=*/0, /*mem_idx=*/0, /*inst_idx=*/0)
          .convert<RegionInstance>();

  RegionInstance dst_inst =
      ID::make_instance(/*owner=*/0, /*creator=*/0, /*mem_idx=*/1, /*inst_idx=*/1)
          .convert<RegionInstance>();

  RegionInstance ind_inst =
      ID::make_instance(/*owner=*/0, /*creator=*/0, /*mem_idx=*/2, /*inst_idx=*/2)
          .convert<RegionInstance>();

  Memory src_mem = src_inst.get_location();
  Memory dst_mem = dst_inst.get_location();
  Memory ind_mem = ind_inst.get_location();
  Memory ind_ib_mem = ID::make_memory(0, 3).convert<Memory>();

  std::vector<Node> nodes(1);
  MockChannel *channel_a = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0);
  nodes[0].dma_channels.push_back(channel_a);

  // Set indirection
  IndexSpace<1, int> is(Rect<1>(0, 1));
  typename CopyIndirection<1, int>::template Unstructured<1, int> indirect;
  indirect.field_id = 0;
  indirect.inst = ind_inst;
  indirect.insts.push_back(src_inst);
  indirect.spaces.push_back(is);
  indirect.oor_possible = false;
  indirect.is_ranges = false;
  IndirectionInfo *ind =
      new IndirectionInfoTyped<1, int, 1, int>(is, indirect, mock_addrsplit_channel);
  Memory mem = Memory::NO_MEMORY;

  // Set input for generate_gather_paths
  TransferGraph::XDTemplate::IO dst_edge;
  dst_edge.iotype = TransferGraph::XDTemplate::IO_INST;
  dst_edge.inst.inst = dst_inst;
  dst_edge.inst.fld_start = 2;
  dst_edge.inst.fld_count = 3;
  std::vector<TransferGraph::XDTemplate> xd_nodes;
  std::vector<TransferGraph::IBInfo> ib_edges;
  std::vector<TransferDesc::FieldInfo> src_fields;

  EXPECT_CALL(*channel_a, suggest_ib_memories(ind_mem))
      .Times(1)
      .WillOnce(::testing::Return(ind_ib_mem));
  EXPECT_CALL(*channel_a, get_factory())
      .Times(2)
      .WillRepeatedly(::testing::Return(nullptr));
  mock_supports_path(channel_a, src_mem, dst_mem, ind_mem, 0, exp_cost);
  mock_supports_path(channel_a, ind_mem, ind_ib_mem, Memory::NO_MEMORY, 0, exp_cost, 0,
                     0);

  // ACT
  ind->generate_gather_paths(nodes.data(), dst_mem, dst_edge, indirect_idx,
                             src_field_start, src_field_count, bytes_per_element,
                             /*serdez_id=*/0, xd_nodes, ib_edges, src_fields);

  // ASSERT
  EXPECT_EQ(xd_nodes.size(), 2);
  EXPECT_EQ(ib_edges.size(), 1);
  EXPECT_EQ(src_fields.size(), 1);

  // Bring addresses into local cpu accessible memory
  // addresses --(copy)--> edge(0, IB)
  EXPECT_EQ(xd_nodes[0].gather_control_input, -1);
  EXPECT_EQ(xd_nodes[0].scatter_control_input, -1);
  EXPECT_EQ(xd_nodes[0].target_node, 0); // TODO
  EXPECT_NE(xd_nodes[0].channel, nullptr);
  EXPECT_EQ(xd_nodes[0].channel->kind, XferDesKind::XFER_MEM_CPY);
  // address edge input
  EXPECT_EQ(xd_nodes[0].inputs.size(), 1);
  EXPECT_EQ(xd_nodes[0].inputs[0].iotype, TransferGraph::XDTemplate::IO_INST);
  EXPECT_EQ(xd_nodes[0].inputs[0].inst.fld_start, 0);
  EXPECT_EQ(xd_nodes[0].inputs[0].inst.fld_count, 1);
  EXPECT_EQ(xd_nodes[0].inputs[0].inst.inst, ind_inst);
  // address edge output
  EXPECT_EQ(xd_nodes[0].outputs.size(), 1);
  EXPECT_EQ(xd_nodes[0].outputs[0].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[0].outputs[0].edge, 0);

  //
  // Do an actual gather
  // [SRC_INST(), edge(0, IB)] --(gather)--> DST_INST
  EXPECT_EQ(xd_nodes[1].gather_control_input, -1);
  EXPECT_EQ(xd_nodes[1].scatter_control_input, -1);
  EXPECT_EQ(xd_nodes[1].target_node, 0); // TODO
  EXPECT_NE(xd_nodes[1].channel, nullptr);
  EXPECT_EQ(xd_nodes[1].channel->kind, XferDesKind::XFER_MEM_CPY);
  // local gather 2 inputs source and addresses
  EXPECT_EQ(xd_nodes[1].inputs.size(), 2);
  EXPECT_EQ(xd_nodes[1].inputs[0].indirect.port, 1);
  EXPECT_EQ(xd_nodes[1].inputs[0].indirect.ind_idx, indirect_idx);
  EXPECT_EQ(xd_nodes[1].inputs[0].indirect.fld_start, src_field_start);
  EXPECT_EQ(xd_nodes[1].inputs[0].indirect.fld_count, src_field_count);
  EXPECT_EQ(xd_nodes[1].inputs[0].indirect.inst, src_inst);
  EXPECT_EQ(xd_nodes[1].inputs[0].iotype, TransferGraph::XDTemplate::IO_INDIRECT_INST);
  EXPECT_EQ(xd_nodes[1].inputs[1].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[1].inputs[1].edge, 0);
  // local gather output edge
  EXPECT_EQ(xd_nodes[1].outputs.size(), 1);
  EXPECT_EQ(xd_nodes[1].outputs[0].iotype, TransferGraph::XDTemplate::IO_INST);
  EXPECT_EQ(xd_nodes[1].outputs[0].inst.fld_start, 2);
  EXPECT_EQ(xd_nodes[1].outputs[0].inst.fld_count, 3);
  EXPECT_EQ(xd_nodes[1].outputs[0].inst.inst, dst_inst);
}

TEST_F(GatherTest, GatherSingleSource1DSameNode_OOREnabled)
{
  // ARRANGE
  const uint64_t exp_cost_a = 4;
  const unsigned indirect_idx = 2;
  const unsigned src_field_start = 8;
  const unsigned src_field_count = 8;
  const size_t bytes_per_element = 8;

  RegionInstance src_inst =
      ID::make_instance(/*owner=*/0, /*creator=*/0, /*mem_idx=*/0, /*inst_idx=*/0)
          .convert<RegionInstance>();

  RegionInstance dst_inst =
      ID::make_instance(/*owner=*/0, /*creator=*/0, /*mem_idx=*/1, /*inst_idx=*/1)
          .convert<RegionInstance>();

  RegionInstance ind_inst =
      ID::make_instance(/*owner=*/0, /*creator=*/0, /*mem_idx=*/2, /*inst_idx=*/2)
          .convert<RegionInstance>();

  Memory src_mem = src_inst.get_location(); // ID::make_memory(0, 0).convert<Memory>();
  Memory dst_mem = dst_inst.get_location(); // ID::make_memory(0, 1).convert<Memory>();
  Memory ind_mem = ind_inst.get_location(); // ID::make_memory(0, 2).convert<Memory>();
  Memory ind_ib_mem = ID::make_memory(0, 3).convert<Memory>();
  Memory src_ib_mem = ID::make_memory(0, 4).convert<Memory>();
  Memory dst_ib_mem = ID::make_memory(0, 5).convert<Memory>();

  std::vector<Node> nodes(1);
  MockChannel *channel_a = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0);
  nodes[0].dma_channels.push_back(channel_a);

  IndexSpace<1, int> is(Rect<1>(0, 1));
  typename CopyIndirection<1, int>::template Unstructured<1, int> indirect;
  indirect.field_id = 0;
  indirect.inst = ind_inst;
  indirect.insts.push_back(src_inst);
  indirect.spaces.push_back(is);
  indirect.oor_possible = true;
  indirect.is_ranges = false;
  IndirectionInfo *ind =
      new IndirectionInfoTyped<1, int, 1, int>(is, indirect, mock_addrsplit_channel);

  TransferGraph::XDTemplate::IO dst_edge;
  dst_edge.iotype = TransferGraph::XDTemplate::IO_INST;
  dst_edge.inst.inst = dst_inst;
  dst_edge.inst.fld_start = 2;
  dst_edge.inst.fld_count = 3;

  std::vector<TransferGraph::XDTemplate> xd_nodes;
  std::vector<TransferGraph::IBInfo> ib_edges;
  std::vector<TransferDesc::FieldInfo> src_fields;

  EXPECT_CALL(*channel_a, suggest_ib_memories(src_mem))
      .Times(1)
      .WillRepeatedly(::testing::Return(ind_ib_mem)); // TODO

  EXPECT_CALL(*channel_a, suggest_ib_memories(dst_mem))
      .Times(1)
      .WillRepeatedly(::testing::Return(ind_ib_mem));

  EXPECT_CALL(*mock_addrsplit_channel, suggest_ib_memories_for_node(0))
      .Times(1)
      .WillOnce(::testing::Return(ind_ib_mem));

  EXPECT_CALL(*channel_a, get_factory())
      .Times(2)
      .WillRepeatedly(::testing::Return(nullptr));

  mock_supports_path(channel_a, src_mem, dst_mem, ind_mem, 1, exp_cost_a);
  mock_supports_path(channel_a, ind_mem, ind_ib_mem, Memory::NO_MEMORY, 0, exp_cost_a, 0,
                     0);

  // ACT
  ind->generate_gather_paths(nodes.data(), dst_mem, dst_edge, indirect_idx,
                             src_field_start, src_field_count, bytes_per_element,
                             /*serdez_id=*/0, xd_nodes, ib_edges, src_fields);

  // ASSERT
  EXPECT_EQ(xd_nodes.size(), 3);
  EXPECT_EQ(ib_edges.size(), 3);
  EXPECT_EQ(src_fields.size(), 1);

  // Bring addresses into local cpu accessible memory
  // addresses --(copy)--> edge(0, IB)
  EXPECT_EQ(xd_nodes[0].gather_control_input, -1);
  EXPECT_EQ(xd_nodes[0].scatter_control_input, -1);
  EXPECT_EQ(xd_nodes[0].target_node, 0);
  EXPECT_NE(xd_nodes[0].channel, nullptr);
  EXPECT_EQ(xd_nodes[0].channel->kind, XferDesKind::XFER_MEM_CPY);
  // address input
  EXPECT_EQ(xd_nodes[0].inputs.size(), 1);
  EXPECT_EQ(xd_nodes[0].inputs[0].iotype, TransferGraph::XDTemplate::IO_INST);
  EXPECT_EQ(xd_nodes[0].inputs[0].inst.fld_start, 0);
  EXPECT_EQ(xd_nodes[0].inputs[0].inst.fld_count, 1);
  // address output
  EXPECT_EQ(xd_nodes[0].outputs.size(), 1);
  EXPECT_EQ(xd_nodes[0].outputs[0].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[0].outputs[0].edge, 0);

  //
  // Run addresses through "address splitter"
  // edge(0, IB) --(address_splitter)--> [edge(1, IB), ctrl_edge(2, IB)]
  EXPECT_EQ(xd_nodes[1].gather_control_input, -1);
  EXPECT_EQ(xd_nodes[1].scatter_control_input, -1);
  EXPECT_EQ(xd_nodes[1].target_node, 0);
  EXPECT_NE(xd_nodes[1].channel, mock_addrsplit_channel);
  // TODO
  // EXPECT_EQ(xd_nodes[1].channel->kind, XferDesKind::XFER_ADDR_SPLIT);

  // input edge(0, IB)
  EXPECT_EQ(xd_nodes[1].inputs.size(), 1);
  EXPECT_EQ(xd_nodes[1].inputs[0].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[1].inputs[0].edge, 0);

  EXPECT_EQ(xd_nodes[1].outputs.size(), 2);
  // (address_splitter)--> edge(1, IB)
  EXPECT_EQ(xd_nodes[1].outputs[0].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[1].outputs[0].edge, 1);
  // (address_splitter)--> ctrl_edge(2, IB)
  EXPECT_EQ(xd_nodes[1].outputs[1].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[1].outputs[1].edge, 2);

  //
  // Do an actual gather
  // [SRC_INST(), edge(1, IB), ctrl_edge(2, IB)] --(gather)--> DST_INST()
  EXPECT_EQ(xd_nodes[2].inputs.size(), 3);
  /// gather input source data
  EXPECT_EQ(xd_nodes[2].inputs[0].iotype, TransferGraph::XDTemplate::IO_INDIRECT_INST);

  EXPECT_EQ(xd_nodes[2].inputs[0].indirect.port, 2);
  EXPECT_EQ(xd_nodes[2].inputs[0].indirect.ind_idx, indirect_idx);
  EXPECT_EQ(xd_nodes[2].inputs[0].indirect.fld_start, src_field_start);
  EXPECT_EQ(xd_nodes[2].inputs[0].indirect.fld_count, src_field_count);
  EXPECT_EQ(xd_nodes[2].inputs[0].indirect.inst, src_inst);
  /// gather input ctrl edge
  EXPECT_EQ(xd_nodes[2].inputs[1].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[2].inputs[1].edge, 2);
  /// gather input addresses
  EXPECT_EQ(xd_nodes[2].inputs[2].iotype, TransferGraph::XDTemplate::IO_EDGE);
  EXPECT_EQ(xd_nodes[2].inputs[2].edge, 1);
  // gather output
  EXPECT_EQ(xd_nodes[2].outputs.size(), 1);
  EXPECT_EQ(xd_nodes[2].outputs[0].iotype, TransferGraph::XDTemplate::IO_INST);
  EXPECT_EQ(xd_nodes[2].outputs[0].inst.fld_start, 2);
  EXPECT_EQ(xd_nodes[2].outputs[0].inst.fld_count, 3);
}

// TODO
TEST(GatherScatterTest, Scatter) {}
