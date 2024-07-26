#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/ib_memory.h"
#include <tuple>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace Realm;

class MockChannel : public Channel {
public:
  typedef std::map<std::pair<Memory, Memory>, size_t> PathMap;
  PathMap paths;
  // Node *node;
  MockChannel(XferDesKind _kind, NodeID _node, const PathMap &_paths)
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

// TODO(apryakhin@): Add operator<<
struct TestCase {
  Memory src;
  Memory dst;
  std::vector<MockChannel *> channels;
  size_t total_bytes;
  uint64_t best_cost;
  int best_ch_node;
  int best_ch_idx;
  XferDesKind best_kind;
  bool status;

  friend std::ostream &operator<<(std::ostream &os, const TestCase &tc)
  {
    os << "src: " << tc.src << "\n"
       << "dst: " << tc.dst << "\n"
       << "total_bytes: " << tc.total_bytes << "\n"
       << "best_cost: " << tc.best_cost << "\n"
       << "best_ch_node: " << tc.best_ch_node << "\n"
       << "best_ch_idx: " << tc.best_ch_idx << "\n"
       << "best_kind: " << tc.best_kind << "\n"
       << "status: " << (tc.status ? "true" : "false") << "\n";
    for(size_t i = 0; i < tc.channels.size(); ++i) {
      os << "channel[" << i << "]: " << *(tc.channels[i]) << "\n";
    }
    return os;
  }
};

class FindBestChannelTest : public ::testing::TestWithParam<TestCase> {
protected:
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
};

TEST_P(FindBestChannelTest, Base)
{
  TestCase test_case = GetParam();

  NodeID max_node_id = 0;
  for(MockChannel *ch : test_case.channels) {
    max_node_id = std::max(max_node_id, ch->node);
  }

  std::vector<Node> nodes(max_node_id + 1);
  for(MockChannel *ch : test_case.channels) {
    nodes[ch->node].dma_channels.push_back(ch);
  }

  ChannelCopyInfo channel_info{test_case.src, test_case.dst};
  uint64_t best_cost;
  Channel *best_channel;
  XferDesKind best_kind = XFER_NONE;

  bool ok = find_best_channel_for_memories(
      nodes.data(), channel_info, /*src_serdez_id=*/0, /*dst_serdez_id=*/0,
      /*redop_id=*/0, test_case.total_bytes, &src_frags, &dst_frags, best_cost,
      best_channel, best_kind);

  EXPECT_EQ(ok, test_case.status);
  EXPECT_EQ(best_cost, test_case.best_cost);
  EXPECT_EQ(best_kind, test_case.best_kind);
  if(test_case.best_ch_node != -1) {
    EXPECT_EQ(best_channel,
              nodes[test_case.best_ch_node].dma_channels[test_case.best_ch_idx])
        << test_case;
  }
}

static inline Memory make_mem(int idx, int node_id)
{
  return ID::make_memory(idx, node_id).convert<Memory>();
}

const static TestCase kTestCases[] = {
    // Case 0: No channels
    TestCase{.src = make_mem(0, 0),
             .dst = make_mem(0, 1),
             .total_bytes = 16,
             .best_cost = 0,
             .best_ch_node = -1,
             .best_kind = XFER_NONE,
             .status = 0},

    // Case 1: Channels don't support path
    TestCase{.src = make_mem(0, 0),
             .dst = make_mem(0, 1),
             .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                       {{{make_mem(0, 0), make_mem(0, 1)},
                                         /*cost=*/0}}),
                       new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                       {{{make_mem(0, 0), make_mem(0, 1)},
                                         /*cost=*/0}})},
             .total_bytes = 16,
             .best_cost = 0,
             .best_ch_node = -1,
             .best_kind = XFER_NONE,
             .status = 0},

    // Case 2: Same node test, best second channel
    TestCase{.src = make_mem(0, 0),
             .dst = make_mem(0, 1),
             .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                       {{{make_mem(0, 0), make_mem(0, 1)},
                                         /*cost=*/7}}),
                       new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                       {{{make_mem(0, 0), make_mem(0, 1)},
                                         /*cost=*/5}})},
             .total_bytes = 16,
             .best_cost = 5,
             .best_ch_node = 0,
             .best_ch_idx = 1,
             .best_kind = XferDesKind::XFER_MEM_CPY,
             .status = 1},

    // Case 3: Different node test
    TestCase{.src = make_mem(0, 0),
             .dst = make_mem(1, 1),
             .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                       {{{make_mem(0, 0), make_mem(1, 1)},
                                         /*cost=*/7}}),
                       new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                       {{{make_mem(0, 0), make_mem(1, 1)},
                                         /*cost=*/10}})},
             .total_bytes = 16,
             .best_cost = 7,
             .best_ch_node = 0,
             .best_ch_idx = 0,
             .best_kind = XferDesKind::XFER_MEM_CPY,
             .status = 1},

    // Case 4: Different node test, no path from souce channel
    TestCase{.src = make_mem(0, 0),
             .dst = make_mem(1, 1),
             .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                       {{{make_mem(0, 0), make_mem(1, 1)},
                                         /*cost=*/0}}),
                       new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                       {{{make_mem(0, 0), make_mem(1, 1)},
                                         /*cost=*/10}})},
             .total_bytes = 16,
             .best_cost = 10,
             .best_ch_node = 1,
             .best_ch_idx = 0,
             .best_kind = XferDesKind::XFER_MEM_CPY,
             .status = 1}};

INSTANTIATE_TEST_SUITE_P(FindBestChannel, FindBestChannelTest,
                         testing::ValuesIn(kTestCases));

struct FFPTestCase {
  Memory src;
  Memory dst;
  // TODO: ind_mem
  std::vector<MockChannel *> channels;
  std::vector<IBMemory *> ib_mems;
  size_t total_bytes;
  MemPathInfo info;
  bool status;

  friend std::ostream &operator<<(std::ostream &os, const FFPTestCase &ftc)
  {
    os << "src: " << ftc.src << "\n"
       << "dst: " << ftc.dst << "\n"
       << "total_bytes: " << ftc.total_bytes << "\n"
       << "status: " << (ftc.status ? "true" : "false") << "\n";
    for(size_t i = 0; i < ftc.channels.size(); ++i) {
      os << "input channel[" << i << "]: " << *(ftc.channels[i]) << "\n";
    }

    for(size_t i = 0; i < ftc.info.path.size(); ++i) {
      os << "path[" << i << "]: " << (ftc.info.path[i]) << "\n";
    }
    return os;
  }
};

class FFPTest : public ::testing::TestWithParam<FFPTestCase> {
protected:
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
  PathCache path_cache;
};

TEST_P(FFPTest, Base)
{
  FFPTestCase test_case = GetParam();

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
  // TODO(apryakhin@): Also compare info channels.
  // EXPECT_EQ(info.xd_channels.size(), test_case.info.xd_channels.size());
}

const static FFPTestCase kFFPTestCases[] = {
    // Case 0: no path available
    FFPTestCase{.src = make_mem(0, 0),
                .dst = make_mem(1, 1),
                .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                          {// src->dst
                                           {{make_mem(0, 0), make_mem(1, 1)},
                                            /*cost=*/0}}),
                          new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                          {{{make_mem(0, 0), make_mem(1, 1)},
                                            /*cost=*/0}})},
                .total_bytes = 16,
                .status = 0},

    // Case 1: src(0) --> dst(1)
    FFPTestCase{.src = make_mem(0, 0),
                .dst = make_mem(1, 1),
                .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                          {// src->dst
                                           {{make_mem(0, 0), make_mem(1, 1)},
                                            /*cost=*/12}}),
                          new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                          {{{make_mem(0, 0), make_mem(1, 1)},
                                            /*cost=*/6}})},
                .total_bytes = 16,
                .info =
                    {
                        .path = {make_mem(0, 0), make_mem(1, 1)},
                    },
                .status = 1},

    // Case 2: src(0) -> src_ib(0) --> dst(1)
    FFPTestCase{
        .src = make_mem(0, 0),
        .dst = make_mem(1, 1),
        .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                  {// src->dst
                                   {{make_mem(0, 0), make_mem(1, 1)},
                                    /*cost=*/12},
                                   // src->src_ib
                                   {{make_mem(0, 0), make_mem(0, 3)},
                                    /*cost=*/2},
                                   // src_ib->dst
                                   {{make_mem(0, 3), make_mem(1, 1)},
                                    /*cost=*/4}}),
                  new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                  {{{make_mem(0, 0), make_mem(1, 1)},
                                    /*cost=*/12}})},
        .ib_mems{new IBMemory(make_mem(0, 3),
                              /*size=*/16, MemoryImpl::MKIND_SYSMEM, Memory::SYSTEM_MEM,
                              /*prealloc_base=*/0, /*_segment=*/0)},
        .total_bytes = 16,
        .info =
            {
                .path = {make_mem(0, 0), make_mem(0, 3), make_mem(1, 1)},
            },
        .status = 1},

    // Case 3: src(0) -> src_ib(0) --> dst_ib(1) -> dst
    FFPTestCase{
        .src = make_mem(0, 0),
        .dst = make_mem(1, 1),
        .channels{new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                  {// src->src_ib
                                   {{make_mem(0, 0), make_mem(0, 3)},
                                    /*cost=*/2},
                                   // src_ib->dst_ib
                                   {{make_mem(0, 3), make_mem(1, 3)},
                                    /*cost=*/2}}),
                  new MockChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
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
                .path = {make_mem(0, 0), make_mem(0, 3), make_mem(1, 3), make_mem(1, 1)},
            },
        .status = 1}

};

INSTANTIATE_TEST_SUITE_P(FindFastestPath, FFPTest, testing::ValuesIn(kFFPTestCases));
