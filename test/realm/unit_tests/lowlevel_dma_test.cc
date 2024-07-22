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

class FFPMockChannel : public Channel {
public:
  typedef std::map<std::pair<Memory, Memory>, size_t> PathMap;
  PathMap paths;
  // Node *node;
  FFPMockChannel(XferDesKind _kind, NodeID _node, const PathMap &_paths)
    : Channel(_kind)
    , paths(_paths)
  {
    this->node = _node;
  }

  uint64_t supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                         CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                         size_t total_bytes, const std::vector<size_t> *src_frags,
                         const std::vector<size_t> *dst_frags, XferDesKind *kind_ret,
                         unsigned *bw_ret, unsigned *lat_ret)
  {
    PathMap::iterator it =
        paths.find(std::make_pair(channel_copy_info.src_mem, channel_copy_info.dst_mem));
    *kind_ret = this->kind;
    return it != paths.end() ? it->second : 0;
  }

  MOCK_METHOD(Memory, suggest_ib_memories, (Memory mem), (const));
  MOCK_METHOD(long, available, (), ());
  MOCK_METHOD(long, submit, (Request * *requests, long nr), ());
  MOCK_METHOD(void, pull, (), ());
  MOCK_METHOD(void, wakeup_xd, (XferDes * xd), ());
  MOCK_METHOD(void, enqueue_ready_xd, (XferDes * xd), ());
  MOCK_METHOD(XferDesFactory *, get_factory, (), ());
};

struct TestCase {
  Memory src;
  Memory dst;
  // TODO: ind_mem
  std::vector<FFPMockChannel *> channels;
  std::vector<IBMemory *> ib_mems;
  size_t total_bytes;
  MemPathInfo info;
  bool status;
};

class FFPTest : public ::testing::TestWithParam<TestCase> {
protected:
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  PathCache path_cache;
};

TEST_P(FFPTest, Base)
{
  TestCase test_case = GetParam();

  NodeID max_node_id = 0;
  for(auto *ch : test_case.channels) {
    max_node_id = std::max(max_node_id, ch->node);
  }

  std::vector<Node> nodes(max_node_id + 1);
  for(auto *ch : test_case.channels) {
    nodes[ch->node].dma_channels.push_back(ch);
  }

  for(auto *ib_mem : test_case.ib_mems) {
    NodeID node = NodeID(ID(ib_mem->me).memory_owner_node());
    nodes[node].ib_memories.push_back(ib_mem);
  }

  ChannelCopyInfo channel_info{test_case.src, test_case.dst};
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;

  MemPathInfo info;
  EXPECT_EQ(find_fastest_path(nodes.data(), path_cache, channel_info, /*serdez_id=*/0,
                              /*redop_id=*/0, test_case.total_bytes, &src_frags,
                              &dst_frags, info),
            test_case.status);
  for(size_t i = 0; i < test_case.info.path.size(); i++) {
    EXPECT_EQ(info.path[i], test_case.info.path[i]);
  }
  // EXPECT_EQ(info.xd_channels.size(), test_case.info.xd_channels.size());
}

static inline Memory make_mem(int idx, int node_id)
{
  return ID::make_memory(idx, node_id).convert<Memory>();
}

INSTANTIATE_TEST_SUITE_P(
    FindFastestPath, FFPTest,
    testing::Values(

        // Case 0: no path available
        TestCase{.src = make_mem(0, 0),
                 .dst = make_mem(1, 1),
                 .channels{new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                              {// src->dst
                                               {{make_mem(0, 0), make_mem(1, 1)},
                                                /*cost=*/0}}),
                           new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                              {{{make_mem(0, 0), make_mem(1, 1)},
                                                /*cost=*/0}})},
                 .total_bytes = 16,
                 .status = 0},

        // Case 1: src(0) --> dst(1)
        TestCase{.src = make_mem(0, 0),
                 .dst = make_mem(1, 1),
                 .channels{new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                              {// src->dst
                                               {{make_mem(0, 0), make_mem(1, 1)},
                                                /*cost=*/12}}),
                           new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                              {{{make_mem(0, 0), make_mem(1, 1)},
                                                /*cost=*/6}})},
                 .total_bytes = 16,
                 .info =
                     {
                         .path = {make_mem(0, 0), make_mem(1, 1)},
                     },
                 .status = 1},

        // Case 2: src(0) -> src_ib(0) --> dst(1)
        TestCase{.src = make_mem(0, 0),
                 .dst = make_mem(1, 1),
                 .channels{new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                              {// src->dst
                                               {{make_mem(0, 0), make_mem(1, 1)},
                                                /*cost=*/12},
                                               // src->src_ib
                                               {{make_mem(0, 0), make_mem(0, 3)},
                                                /*cost=*/2},
                                               // src_ib->dst
                                               {{make_mem(0, 3), make_mem(1, 1)},
                                                /*cost=*/4}}),
                           new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                              {{{make_mem(0, 0), make_mem(1, 1)},
                                                /*cost=*/12}})},
                 .ib_mems{new IBMemory(make_mem(0, 3),
                                       /*size=*/16, MemoryImpl::MKIND_SYSMEM,
                                       Memory::SYSTEM_MEM,
                                       /*prealloc_base=*/0, /*_segment=*/0)},
                 .total_bytes = 16,
                 .info =
                     {
                         .path = {make_mem(0, 0), make_mem(0, 3), make_mem(1, 1)},
                     },
                 .status = 1},

        // Case 3: src(0) -> src_ib(0) --> dst_ib(1) -> dst
        TestCase{
            .src = make_mem(0, 0),
            .dst = make_mem(1, 1),
            .channels{new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                         {// src->src_ib
                                          {{make_mem(0, 0), make_mem(0, 3)},
                                           /*cost=*/2},
                                          // src_ib->dst_ib
                                          {{make_mem(0, 3), make_mem(1, 3)},
                                           /*cost=*/2}}),
                      new FFPMockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                         {// dst_ib->dst
                                          {{make_mem(1, 3), make_mem(1, 1)},
                                           /*cost=*/2}})},
            .ib_mems{
                new IBMemory(make_mem(0, 3),
                             /*size=*/16, MemoryImpl::MKIND_SYSMEM, Memory::SYSTEM_MEM,
                             /*prealloc_base=*/0,
                             /*_segment=*/0),
                new IBMemory(make_mem(1, 3),
                             /*size=*/16, MemoryImpl::MKIND_SYSMEM, Memory::SYSTEM_MEM,
                             /*prealloc_base=*/0,
                             /*_segment=*/0),
            },
            .total_bytes = 16,
            .info =
                {
                    .path = {make_mem(0, 0), make_mem(0, 3), make_mem(1, 3),
                             make_mem(1, 1)},
                },
            .status = 1}));

