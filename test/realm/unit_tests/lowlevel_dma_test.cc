#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/ib_memory.h"
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

TEST(FindBestChannelTest, BestChannelForMemPairNoChannels)
{
  std::vector<Node> nodes(1);
  ChannelCopyInfo channel_info{ID::make_memory(0, 0).convert<Memory>(),
                               ID::make_memory(0, 1).convert<Memory>()};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;
  uint64_t best_cost;
  Channel *best_channel;
  XferDesKind best_kind = XFER_NONE;

  bool ok = find_best_channel_for_memories(nodes.data(), channel_info, 0, 0, 0,
                                           total_bytes, &src_frags, &dst_frags, best_cost,
                                           best_channel, best_kind);

  EXPECT_FALSE(ok);
}

TEST(FindBestChannelTest, BestChannelForMemPairSameMemory)
{
  const uint64_t exp_cost_a = 7;
  const uint64_t exp_cost_b = 5;
  std::vector<Node> nodes(1);
  MockChannel *channel_a = new MockChannel(XferDesKind::XFER_MEM_CPY, 0);
  nodes[0].dma_channels.push_back(channel_a);
  MockChannel *channel_b = new MockChannel(XferDesKind::XFER_MEM_CPY, 0);
  nodes[0].dma_channels.push_back(channel_b);
  ChannelCopyInfo channel_info{ID::make_memory(0, 0).convert<Memory>(),
                               ID::make_memory(0, 1).convert<Memory>()};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;
  uint64_t best_cost;
  Channel *best_channel;
  XferDesKind best_kind = XFER_NONE;
  // TODO(apryakhin): Consider wrapping into a for-loop
  EXPECT_CALL(*channel_a, supports_path(channel_info, 0, 0, 0, total_bytes, &src_frags,
                                        &dst_frags, testing::_, nullptr, nullptr))
      .Times(1)
      .WillRepeatedly(::testing::Return(exp_cost_a));
  EXPECT_CALL(*channel_b, supports_path(channel_info, 0, 0, 0, total_bytes, &src_frags,
                                        &dst_frags, testing::_, nullptr, nullptr))
      .Times(1)
      .WillRepeatedly(::testing::Return(exp_cost_b));

  bool ok = find_best_channel_for_memories(nodes.data(), channel_info, 0, 0, 0,
                                           total_bytes, &src_frags, &dst_frags, best_cost,
                                           best_channel, best_kind);

  EXPECT_TRUE(ok);
  EXPECT_EQ(best_cost, exp_cost_b);
  EXPECT_EQ(best_channel, channel_b);
}

TEST(FindBestChannelTest, BestChannelForMemPairDifferntMemory)
{
  const uint64_t exp_cost_src = 7;
  const uint64_t exp_cost_dst = 10;
  std::vector<Node> nodes(2);
  MockChannel *src_channel = new MockChannel(XferDesKind::XFER_MEM_CPY, 0);
  nodes[0].dma_channels.push_back(src_channel);
  MockChannel *dst_channel = new MockChannel(XferDesKind::XFER_MEM_CPY, 1);
  nodes[1].dma_channels.push_back(dst_channel);

  ChannelCopyInfo channel_info{ID::make_memory(0, 0).convert<Memory>(),
                               ID::make_memory(1, 1).convert<Memory>()};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;
  uint64_t best_cost;
  Channel *best_channel;
  XferDesKind best_kind = XFER_NONE;

  EXPECT_CALL(*src_channel, supports_path(channel_info, 0, 0, 0, total_bytes, &src_frags,
                                          &dst_frags, testing::_, nullptr, nullptr))
      .Times(1)
      .WillRepeatedly(::testing::Return(exp_cost_src));
  EXPECT_CALL(*dst_channel, supports_path(channel_info, 0, 0, 0, total_bytes, &src_frags,
                                          &dst_frags, testing::_, nullptr, nullptr))
      .Times(1)
      .WillRepeatedly(::testing::Return(exp_cost_dst));

  bool ok = find_best_channel_for_memories(nodes.data(), channel_info, 0, 0, 0,
                                           total_bytes, &src_frags, &dst_frags, best_cost,
                                           best_channel, best_kind);

  EXPECT_TRUE(ok);
  EXPECT_EQ(best_cost, exp_cost_src);
  EXPECT_EQ(best_channel, src_channel);
}

TEST(FindBestChannelTest, BestChannelForMemPairNoSrcPath)
{
  const uint64_t exp_cost_dst = 10;
  std::vector<Node> nodes(2);
  MockChannel *src_channel = new MockChannel(XferDesKind::XFER_MEM_CPY, 0);
  nodes[0].dma_channels.push_back(src_channel);
  MockChannel *dst_channel = new MockChannel(XferDesKind::XFER_MEM_CPY, 1);
  nodes[1].dma_channels.push_back(dst_channel);

  ChannelCopyInfo channel_info{ID::make_memory(0, 0).convert<Memory>(),
                               ID::make_memory(1, 1).convert<Memory>()};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;
  uint64_t best_cost;
  Channel *best_channel;
  XferDesKind best_kind = XFER_NONE;
  EXPECT_CALL(*src_channel, supports_path(channel_info, 0, 0, 0, total_bytes, &src_frags,
                                          &dst_frags, testing::_, nullptr, nullptr))
      .Times(1)
      .WillRepeatedly(::testing::Return(0));
  EXPECT_CALL(*dst_channel, supports_path(channel_info, 0, 0, 0, total_bytes, &src_frags,
                                          &dst_frags, testing::_, nullptr, nullptr))
      .Times(1)
      .WillRepeatedly(::testing::Return(exp_cost_dst));

  bool ok = find_best_channel_for_memories(nodes.data(), channel_info, 0, 0, 0,
                                           total_bytes, &src_frags, &dst_frags, best_cost,
                                           best_channel, best_kind);

  EXPECT_TRUE(ok);
  EXPECT_EQ(best_cost, exp_cost_dst);
  EXPECT_EQ(best_channel, dst_channel);
}

class FFPTest : public ::testing::Test {
protected:
  virtual void SetUp() {}

  virtual void TearDown() {}

  void mock_supports_path(MockChannel *channel, Memory src, Memory dst,
                          std::vector<size_t> *src_frags, std::vector<size_t> *dst_frags,
                          uint64_t cost, bool direct = true, size_t total_bytes = 16)
  {
    ChannelCopyInfo info{src, dst};
    info.is_direct = direct;
    EXPECT_CALL(*channel, supports_path(info, /*src_serdez_id=*/0, /*dst_serdez_id*/ 0,
                                        /*redop_id=*/0, total_bytes, src_frags, dst_frags,
                                        testing::_, nullptr, nullptr))
        .Times(1)
        .WillRepeatedly(::testing::Return(cost));
  }
};

// TODO(apryakhin): Add tests with path cache

TEST_F(FFPTest, FFPNoChannels)
{
  std::vector<Node> nodes(1);
  ChannelCopyInfo channel_info{ID::make_memory(0, 0).convert<Memory>(),
                               ID::make_memory(0, 1).convert<Memory>()};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;

  MemPathInfo info;
  std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *> path_cache;
  bool ok = find_fastest_path(nodes.data(), path_cache, channel_info, /*serdez_id=*/0,
                              /*redop_id=*/0, total_bytes, &src_frags, &dst_frags, info);

  EXPECT_FALSE(ok);
  EXPECT_TRUE(info.xd_channels.empty());
  EXPECT_TRUE(info.path.empty());
}

TEST_F(FFPTest, FFPSrcDstDirectNoPath)
{
  const uint64_t exp_cost_a = 0;
  const uint64_t exp_cost_b = 0;
  std::vector<Node> nodes(1);
  MockChannel *channel_a = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0);
  MockChannel *channel_b = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1);
  nodes[0].dma_channels.push_back(channel_a);
  nodes[0].dma_channels.push_back(channel_b);
  Memory src_mem = ID::make_memory(0, 0).convert<Memory>();
  Memory dst_mem = ID::make_memory(0, 1).convert<Memory>();

  ChannelCopyInfo channel_info{src_mem, dst_mem};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;

  mock_supports_path(channel_a, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_a);
  mock_supports_path(channel_b, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_b);

  MemPathInfo info;
  std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *> path_cache;
  bool ok = find_fastest_path(nodes.data(), path_cache, channel_info, /*serdez_id=*/0,
                              /*redop_id=*/0, total_bytes, &src_frags, &dst_frags, info);

  EXPECT_FALSE(ok);
  EXPECT_TRUE(info.xd_channels.empty());
  EXPECT_TRUE(info.path.empty());
}

// TODO(aprakhin): There are more channel/cost combinations we can
// test. Consider making parameterized tests in a future.

TEST_F(FFPTest, FFPSrcDstDirect)
{
  const uint64_t exp_cost_a = 7;
  const uint64_t exp_cost_b = 5;
  std::vector<Node> nodes(1);
  MockChannel *channel_a = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0);
  MockChannel *channel_b = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1);
  nodes[0].dma_channels.push_back(channel_a);
  nodes[0].dma_channels.push_back(channel_b);
  Memory src_mem = ID::make_memory(0, 0).convert<Memory>();
  Memory dst_mem = ID::make_memory(0, 1).convert<Memory>();

  ChannelCopyInfo channel_info{src_mem, dst_mem};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;

  mock_supports_path(channel_a, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_a);
  mock_supports_path(channel_b, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_b);

  MemPathInfo info;
  std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *> path_cache;
  bool ok = find_fastest_path(nodes.data(), path_cache, channel_info, /*serdez_id=*/0,
                              /*redop_id=*/0, total_bytes, &src_frags, &dst_frags, info);

  EXPECT_TRUE(ok);
  EXPECT_EQ(info.xd_channels.size(), 1);
  EXPECT_EQ(info.xd_channels[0], channel_b);
  EXPECT_EQ(info.path.size(), 2);
  EXPECT_EQ(info.path[0], src_mem);
  EXPECT_EQ(info.path[1], dst_mem);
}

TEST_F(FFPTest, FFPSrcIBDst)
{
  const uint64_t exp_cost_a = 12;
  const uint64_t exp_cost_b = 12;
  std::vector<Node> nodes(2);
  MockChannel *channel_a = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0);
  MockChannel *channel_b = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1);
  nodes[0].dma_channels.push_back(channel_a);
  nodes[1].dma_channels.push_back(channel_b);
  Memory src_mem = ID::make_memory(0, 0).convert<Memory>();
  Memory dst_mem = ID::make_memory(1, 1).convert<Memory>();
  IBMemory *src_ib_mem =
      new IBMemory(Memory(ID::make_memory(0, 3).convert<Memory>()),
                   /*size=*/16, MemoryImpl::MKIND_SYSMEM, Memory::SYSTEM_MEM,
                   /*prealloc_base=*/0, /*_segment=*/0);
  nodes[0].ib_memories.push_back(src_ib_mem);

  ChannelCopyInfo channel_info{src_mem, dst_mem};
  channel_info.is_direct = true;
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;

  // Channels setup - direct path
  mock_supports_path(channel_a, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_a);
  mock_supports_path(channel_b, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_b);

  // Channels setup - path via single edge
  mock_supports_path(channel_a, src_mem, src_ib_mem->me, &src_frags, nullptr, 2, false);
  mock_supports_path(channel_a, src_ib_mem->me, dst_mem, nullptr, &dst_frags, 4, false);
  mock_supports_path(channel_b, src_ib_mem->me, dst_mem, nullptr, &dst_frags, 4, false);

  MemPathInfo info;
  std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *> path_cache;
  bool ok = find_fastest_path(nodes.data(), path_cache, channel_info, /*serdez_id=*/0,
                              /*redop_id=*/0, total_bytes, &src_frags, &dst_frags, info);

  EXPECT_TRUE(ok);
  EXPECT_EQ(info.xd_channels.size(), 2);
  EXPECT_EQ(info.xd_channels[0], channel_a);
  EXPECT_EQ(info.xd_channels[1], channel_a);
  EXPECT_EQ(info.path.size(), 3);
  EXPECT_EQ(info.path[0], src_mem);
  EXPECT_EQ(info.path[1], src_ib_mem->me);
  EXPECT_EQ(info.path[2], dst_mem);
}

TEST_F(FFPTest, FFPSrcIBToIBDst)
{
  const uint64_t exp_cost_a = 12;
  const uint64_t exp_cost_b = 12;
  std::vector<Node> nodes(2);
  MockChannel *channel_a = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0);
  MockChannel *channel_b = new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1);
  nodes[0].dma_channels.push_back(channel_a);
  nodes[1].dma_channels.push_back(channel_b);
  Memory src_mem = ID::make_memory(0, 0).convert<Memory>();
  Memory dst_mem = ID::make_memory(1, 3).convert<Memory>();
  IBMemory *src_ib_mem =
      new IBMemory(Memory(ID::make_memory(0, 1).convert<Memory>()),
                   /*size=*/16, MemoryImpl::MKIND_SYSMEM, Memory::SYSTEM_MEM,
                   /*prealloc_base=*/0, /*_segment=*/0);
  IBMemory *dst_ib_mem =
      new IBMemory(Memory(ID::make_memory(1, 2).convert<Memory>()),
                   /*size=*/16, MemoryImpl::MKIND_SYSMEM, Memory::SYSTEM_MEM,
                   /*prealloc_base=*/0, /*_segment=*/0);
  nodes[0].ib_memories.push_back(src_ib_mem);
  nodes[1].ib_memories.push_back(dst_ib_mem);

  ChannelCopyInfo channel_info{src_mem, dst_mem};
  channel_info.is_direct = true;
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  size_t total_bytes = 16;

  // TODO(apryakhin): Consider doing a smarter setup and name
  // variables.

  // Channels setup - direct path
  mock_supports_path(channel_a, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_a);
  mock_supports_path(channel_b, src_mem, dst_mem, &src_frags, &dst_frags, exp_cost_b);

  // Channels setup - first edge
  mock_supports_path(channel_a, src_mem, src_ib_mem->me, &src_frags, nullptr, 2, false);
  mock_supports_path(channel_a, src_mem, dst_ib_mem->me, &src_frags, nullptr, 0, false);
  mock_supports_path(channel_b, src_mem, dst_ib_mem->me, &src_frags, nullptr, 0, false);

  // Channels setup - intermediate edges
  mock_supports_path(channel_a, src_ib_mem->me, dst_ib_mem->me, nullptr, nullptr, 2,
                     false);
  mock_supports_path(channel_b, src_ib_mem->me, dst_ib_mem->me, nullptr, nullptr, 2,
                     false);

  // Channels setup - final step
  mock_supports_path(channel_a, src_ib_mem->me, dst_mem, nullptr, &dst_frags, 0, false);
  mock_supports_path(channel_b, src_ib_mem->me, dst_mem, nullptr, &dst_frags, 8, false);
  mock_supports_path(channel_b, dst_ib_mem->me, dst_mem, nullptr, &dst_frags, 4, false);

  MemPathInfo info;
  std::map<std::pair<realm_id_t, realm_id_t>, PathLRU *> path_cache;
  bool ok = find_fastest_path(nodes.data(), path_cache, channel_info, /*serdez_id=*/0,
                              /*redop_id=*/0, total_bytes, &src_frags, &dst_frags, info);

  EXPECT_TRUE(ok);
  EXPECT_EQ(info.xd_channels.size(), 3);
  EXPECT_EQ(info.path.size(), 4);
  EXPECT_EQ(info.xd_channels[0], channel_a);
  EXPECT_EQ(info.xd_channels[1], channel_a);
  EXPECT_EQ(info.xd_channels[2], channel_b);
  EXPECT_EQ(info.path[0], src_mem);
  EXPECT_EQ(info.path[1], src_ib_mem->me);
  EXPECT_EQ(info.path[2], dst_ib_mem->me);
  EXPECT_EQ(info.path[3], dst_mem);
}
