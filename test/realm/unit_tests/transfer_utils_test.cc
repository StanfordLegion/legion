#include "realm/transfer/transfer_utils.h"
#include <tuple>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace Realm;

TEST(TransferUtilsTest, EmptyDomainAndBounds)
{
  Rect<1> subrect, domain = Rect<1>(Point<1>(1), Point<1>(0));
  Point<1> next_start(0);
  std::vector<int> dim_order{0};
  EXPECT_TRUE(
      next_subrect(domain, Point<1>(1), domain, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, Point<1>(0));
}

TEST(TransferUtilsTest, EmptyDomain)
{
  Rect<1> subrect, domain = Rect<1>(Point<1>(1), Point<1>(0));
  Point<1> next_start(0);
  std::vector<int> dim_order{0};
  EXPECT_TRUE(
      next_subrect(domain, Point<1>(1), domain, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, Point<1>(0));
}

TEST(TransferUtilsTest, BoundsContainFullSubrect)
{
  Rect<1> subrect, domain = Rect<1>(Point<1>(0), Point<1>(10));
  Point<1> next_start(0);
  std::vector<int> dim_order{0};
  EXPECT_TRUE(
      next_subrect(domain, Point<1>(0), domain, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, Point<1>(0));
}

TEST(TransferUtilsTest, StartNotFullSpanDimension)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<2> subrect, domain = Rect<2>(Point<2>(0, 0), Point<2>(8, 8));
  Point<2> next_start(0, 0);
  std::vector<int> dim_order{0, 1};

  EXPECT_FALSE(next_subrect(domain, Point<2>(1, 0), bounds, dim_order.data(), subrect,
                            next_start));
  EXPECT_EQ(subrect.lo, Point<2>(1, 0));
  EXPECT_EQ(subrect.hi, Point<2>(8, 0));
  EXPECT_EQ(next_start, Point<2>(0, 1));

  EXPECT_TRUE(
      next_subrect(domain, next_start, bounds, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, Point<2>(0, 1));
  EXPECT_EQ(subrect.hi, Point<2>(8, 8));
  EXPECT_EQ(next_start, Point<2>(0, 0));
}

TEST(TransferUtilsTest, HigherDomainBounds)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<2> subrect, domain = Rect<2>(Point<2>(1, 1), Point<2>(11, 11));
  Point<2> next_start(0, 0);
  std::vector<int> dim_order{0, 1};

  for(int next_y = 1; next_y < 10; next_y++) {
    EXPECT_FALSE(next_subrect(domain, Point<2>(1, next_y), bounds, dim_order.data(),
                              subrect, next_start));
    EXPECT_EQ(subrect.lo, Point<2>(1, next_y));
    EXPECT_EQ(subrect.hi, Point<2>(10, next_y));
    EXPECT_EQ(next_start, Point<2>(11, next_y));
  }
}

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
  const uint64_t exp_cost_a = 7;
  const uint64_t exp_cost_b = 5;
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
}
