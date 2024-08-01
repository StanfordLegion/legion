#include "realm/transfer/transfer.h"

#include "realm/transfer/ib_memory.h"
#include <tuple>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace Realm;

static inline Memory make_mem(int idx, int node_id)
{
  return ID::make_memory(idx, node_id).convert<Memory>();
}

static inline IBMemory *make_ib_mem(Memory memory, size_t size = 16)
{
  return new IBMemory(memory, size, MemoryImpl::MKIND_SYSMEM, Memory::SYSTEM_MEM,
                      /*prealloc_base=*/0,
                      /*_segment=*/0);
}

constexpr int kIBMemIdx = 3;

class MockGatherChannel : public Channel {
public:
  typedef std::map<std::pair<Memory, Memory>, size_t> PathMap;
  PathMap paths;
  MockGatherChannel(XferDesKind _kind, NodeID _node, const PathMap &_paths)
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

  Memory suggest_ib_memories(Memory memory) const { return make_mem(node, kIBMemIdx); }

  Memory suggest_ib_memories_for_node(NodeID node) const
  {
    return make_mem(node, kIBMemIdx);
  }

  XferDesFactory *get_factory() { return nullptr; }

  MOCK_METHOD(long, available, (), ());
  MOCK_METHOD(long, submit, (Request * *requests, long nr), ());
  MOCK_METHOD(void, pull, (), ());
  MOCK_METHOD(void, wakeup_xd, (XferDes * xd), ());
  MOCK_METHOD(void, enqueue_ready_xd, (XferDes * xd), ());
};

struct GatherTestCase {
  std::vector<MockGatherChannel *> channels;
  std::vector<IBMemory *> ib_mems;
  IndirectionInfo *indirection;
  RegionInstance dst_inst;
  // expected
  std::vector<TransferGraph::XDTemplate> xd_nodes;
};

class GatherScatterTest : public ::testing::TestWithParam<GatherTestCase> {
  void TearDown() { GatherTestCase test_case = GetParam(); }
};

TEST_P(GatherScatterTest, Base)
{
  GatherTestCase test_case = GetParam();

  const unsigned indirect_idx = 2;
  const unsigned src_field_start = 8;
  const unsigned src_field_count = 8;
  const size_t bytes_per_element = 8;

  Memory dst_mem = test_case.dst_inst.get_location();

  std::vector<TransferGraph::XDTemplate> xd_nodes;
  std::vector<TransferGraph::IBInfo> ib_edges;
  std::vector<TransferDesc::FieldInfo> src_fields;

  TransferGraph::XDTemplate::IO dst_edge;
  dst_edge.iotype = TransferGraph::XDTemplate::IO_INST;
  dst_edge.inst.inst = test_case.dst_inst;
  dst_edge.inst.fld_start = 2;
  dst_edge.inst.fld_count = 3;

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

  test_case.indirection->generate_gather_paths(
      nodes.data(), dst_mem, dst_edge, indirect_idx, src_field_start, src_field_count,
      bytes_per_element,
      /*serdez_id=*/0, xd_nodes, ib_edges, src_fields);

  for(size_t i = 0; i < test_case.xd_nodes.size(); i++) {
    EXPECT_EQ(xd_nodes[i].inputs.size(), test_case.xd_nodes[i].inputs.size())
        << " xd_node:" << i;

    // TODO: compare all fields
    for(size_t j = 0; j < test_case.xd_nodes[i].inputs.size(); j++) {

      EXPECT_EQ(xd_nodes[i].inputs[j].iotype, test_case.xd_nodes[i].inputs[j].iotype)
          << "xd_node:" << i << " index:" << j;

      if(test_case.xd_nodes[i].inputs[j].iotype ==
         TransferGraph::XDTemplate::IO_INDIRECT_INST) {
        EXPECT_EQ(xd_nodes[i].inputs[j].indirect.ind_idx,
                  test_case.xd_nodes[i].inputs[j].indirect.ind_idx);

        EXPECT_EQ(xd_nodes[i].inputs[j].indirect.inst,
                  test_case.xd_nodes[i].inputs[j].indirect.inst);

        EXPECT_EQ(xd_nodes[i].inputs[j].indirect.port,
                  test_case.xd_nodes[i].inputs[j].indirect.port);
      }

      else if(test_case.xd_nodes[i].inputs[j].iotype ==
              TransferGraph::XDTemplate::IO_INST) {
        EXPECT_EQ(xd_nodes[i].inputs[j].inst.inst,
                  test_case.xd_nodes[i].inputs[j].inst.inst);
      } else {
        EXPECT_EQ(xd_nodes[i].inputs[j].edge, test_case.xd_nodes[i].inputs[j].edge);
      }
    }

    EXPECT_EQ(xd_nodes.size(), test_case.xd_nodes.size());

    EXPECT_EQ(xd_nodes[i].outputs.size(), test_case.xd_nodes[i].outputs.size())
        << " xd_node:" << i;

    for(size_t j = 0; j < test_case.xd_nodes[i].outputs.size(); j++) {
      EXPECT_EQ(xd_nodes[i].outputs[j].iotype, test_case.xd_nodes[i].outputs[j].iotype)
          << "xd_node:" << i << " index:" << j;

      if(test_case.xd_nodes[i].outputs[j].iotype == TransferGraph::XDTemplate::IO_INST) {
        EXPECT_EQ(xd_nodes[i].outputs[j].inst.inst,
                  test_case.xd_nodes[i].outputs[j].inst.inst)
            << "xd_node:" << i << " index:" << j;
      } else if(test_case.xd_nodes[i].outputs[j].iotype ==
                TransferGraph::XDTemplate::IO_EDGE) {
        EXPECT_EQ(xd_nodes[i].outputs[j].edge, test_case.xd_nodes[i].outputs[j].edge);
      }
    }
  }
}

static inline RegionInstance make_inst(int owner, int creator, int mem_idx, int inst_idx)
{
  return ID::make_instance(owner, creator, mem_idx, inst_idx).convert<RegionInstance>();
}

static TransferGraph::XDTemplate::IO inline mk_indirect(RegionInstance inst,
                                                        unsigned ind_idx = 2,
                                                        unsigned port = 1,
                                                        unsigned fld_start = 0,
                                                        unsigned fld_count = 0)
{
  return TransferGraph::XDTemplate::mk_indirect(ind_idx, port, inst, fld_start,
                                                fld_count);
}

static TransferGraph::XDTemplate::IO inline mk_inst(RegionInstance inst,
                                                    unsigned fld_start = 0,
                                                    unsigned fld_count = 0)
{
  return TransferGraph::XDTemplate::mk_inst(inst, fld_start, fld_count);
}

static inline TransferGraph::XDTemplate::IO mk_edge(unsigned edge)
{
  return TransferGraph::XDTemplate::mk_edge(edge);
}

struct TestInstances {
  RegionInstance src;
  RegionInstance dst;
  RegionInstance ind;
};

// 0: src, dst, ind (same node)
// 1: src, dst, ind (different nodes)
const static TestInstances kInst[] = {{.src = make_inst(0, 0, 0, 0),
                                       .dst = make_inst(0, 0, 1, 1),
                                       .ind = make_inst(0, 0, 2, 2)},
                                      {.src = make_inst(0, 0, 0, 0),
                                       .dst = make_inst(1, 1, 1, 1),
                                       .ind = make_inst(0, 0, 2, 2)}};

const static Memory kIBMem[] = {make_mem(0, 3), make_mem(0, 4), make_mem(1, 4)};

const static GatherTestCase kTestCases[] = {
    // Case 0: Same node gather
    // dst(0) <-- src(0)[ind(0)]
    GatherTestCase{
        .channels{new MockGatherChannel(
            XferDesKind::XFER_MEM_CPY, /*node=*/0,
            {// src --> dst
             {{kInst[0].src.get_location(), kInst[0].dst.get_location()},
              /*cost=*/2},
             // ind_mem -> ind_ib_mem
             {{kInst[0].ind.get_location(), kIBMem[0]},
              /*cost=*/2}})},

        .ib_mems{make_ib_mem(kIBMem[0])},

        .indirection = new IndirectionInfoTyped<1, int, 1, int>(
            Rect<1, int>(0, 1),
            typename CopyIndirection<1, int>::template Unstructured<1, int>(
                kInst[0].ind, {Rect<1, int>(0, 1)}, {kInst[0].src}, 0),
            nullptr),
        .dst_inst = kInst[0].dst,

        .xd_nodes = {{.inputs = {mk_indirect(kInst[0].src), mk_inst(kInst[0].ind)},
                      .outputs = {mk_inst(kInst[0].dst)}}}},

    // Case 1: Different nodes gather
    // dst(1) <-- ib(1) <-- ib(0) <-- src(0)[ind(0)]
    // TODO
    GatherTestCase{
        .channels{
            new MockGatherChannel(XferDesKind::XFER_MEM_CPY, /*node=*/0,
                                  {
                                      // src -> src_ib
                                      {{kInst[1].src.get_location(), kIBMem[1]},
                                       /*cost=*/2},
                                      // ind_mem -> ind_ib_mem
                                      {{kInst[1].ind.get_location(), kIBMem[0]},
                                       /*cost=*/2},
                                      // src_ib -> dst_ib
                                      {{kIBMem[1], kIBMem[2]},
                                       /*cost=*/4},
                                  }),

            new MockGatherChannel(XferDesKind::XFER_MEM_CPY, /*node=*/1,
                                  {// dst_ib -> dst
                                   {{kIBMem[2], kInst[1].dst.get_location()},
                                    /*cost=*/2},
                                   // ind_mem -> ind_ib_mem
                                   {{kInst[1].ind.get_location(), kIBMem[0]},
                                    /*cost=*/0}}),
        },

        .ib_mems{make_ib_mem(kIBMem[0]), make_ib_mem(kIBMem[1]), make_ib_mem(kIBMem[2])},

        .indirection = new IndirectionInfoTyped<1, int, 1, int>(
            Rect<1, int>(0, 1),
            typename CopyIndirection<1, int>::template Unstructured<1, int>(
                kInst[1].ind, {Rect<1, int>(0, 1)}, {kInst[1].src}, 0),
            nullptr),

        .dst_inst = kInst[1].dst,

        .xd_nodes =
            {
                {.inputs = {mk_indirect(kInst[1].src), mk_inst(kInst[1].ind)},
                 .outputs = {mk_edge(0)}},
                {.inputs = {mk_edge(0)}, .outputs = {mk_edge(1)}},
                {.inputs = {mk_edge(1)}, .outputs = {mk_inst(kInst[1].dst)}},
            },

    },

    // Case 2: Same node gather with oorflag
    // dst(0) <-- src(0)[ind(0)]
    GatherTestCase{.channels{new MockGatherChannel(
                       XferDesKind::XFER_MEM_CPY, /*node=*/0,
                       {// src --> dst
                        {{kInst[0].src.get_location(), kInst[0].dst.get_location()},
                         /*cost=*/2},
                        // ind_mem -> ind_ib_mem
                        {{kInst[0].ind.get_location(), kIBMem[0]},
                         /*cost=*/2}})},

                   .ib_mems{make_ib_mem(kIBMem[0])},

                   .indirection = new IndirectionInfoTyped<1, int, 1, int>(
                       Rect<1, int>(0, 1),
                       typename CopyIndirection<1, int>::template Unstructured<1, int>(
                           kInst[0].ind, {Rect<1, int>(0, 1)}, {kInst[0].src},
                           /*field_id=*/0, /*subfield_offset=*/0,
                           /*is_ranges=*/0, /*oor=*/1),
                       new MockGatherChannel(XferDesKind::XFER_ADDR_SPLIT, 0, {})),
                   // TODO: Don't leak address split channel

                   .dst_inst = kInst[0].dst,

                   .xd_nodes = {{
                                    .inputs = {mk_inst(kInst[0].ind)},
                                    .outputs = {mk_edge(0), mk_edge(1)},
                                },

                                {.inputs = {mk_indirect(kInst[0].src, 2, 2), mk_edge(1),
                                            mk_edge(0)},
                                 .outputs = {mk_inst(kInst[0].dst)}}}}};

INSTANTIATE_TEST_SUITE_P(Foo, GatherScatterTest, testing::ValuesIn(kTestCases));
