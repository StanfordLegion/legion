#include "realm/nodeset.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>

using namespace Realm;

enum
{
  EMPTY,
  VALS,    // one or more distinct values
  RANGES,  // one or more non-overlapping ranges
  BITMASK, // full (externally-allocated) bitmask
};

#define NULL_NODE std::nullopt

// we need to mock the NodeSet in order to access the proteced member
class MockNodeSet : public NodeSet {
public:
  bool check_format(int format) { return enc_format == format; }
};

class NodeSetTestBase : public ::testing::Test {
protected:
  virtual void SetUp()
  {
    NodeSetBitmask::configure_allocator(max_node_id, bitsets_per_chunk, use_twolevel);
  }

  virtual void TearDown()
  {
    nodeset.clear();
    NodeSetBitmask::free_allocations();
  }

  MockNodeSet nodeset;
  int max_node_id = 512;
  size_t bitsets_per_chunk = 1024;
  bool use_twolevel = true;
};

// 1. test add and remove
// in arrage, we add nodes using add_range or add
// after arrange, the nodeset is in the format of ENC_EMPTY, ENC_VALS, ENC_RANGE or
// ENC_BITMASK

struct AddRemoveTestParam {
  std::vector<std::pair<NodeID, std::optional<NodeID>>>
      arrage_nodes; // if second is -1, we use add(first), otherwise, we use
                    // add_range(first, second)
  std::vector<NodeID> action_nodes;
  int format_pior_action;
  int format_after_action;
  size_t nb_nodes;
};

// arrange function used by both test 1 and 2
static inline void
add_nodes(MockNodeSet &nodeset,
          std::vector<std::pair<NodeID, std::optional<NodeID>>> &arrage_nodes, int format)
{
  for(std::pair<NodeID, std::optional<NodeID>> &range : arrage_nodes) {
    if(range.second == NULL_NODE) {
      nodeset.add(range.first);
    } else {
      nodeset.add_range(range.first, range.second.value());
    }
  }
  assert(nodeset.check_format(format));
}

class NodeSetAddRemoveTestBase
  : public NodeSetTestBase,
    public ::testing::WithParamInterface<AddRemoveTestParam> {
protected:
  virtual void SetUp()
  {
    // action
    NodeSetTestBase::SetUp();
    param = GetParam();
    add_nodes(nodeset, param.arrage_nodes, param.format_pior_action);
  }

public:
  AddRemoveTestParam param;
};

class NodeSetAddTest : public NodeSetAddRemoveTestBase {};

TEST_P(NodeSetAddTest, AddNode)
{
  std::vector<NodeID> &action_nodes = param.action_nodes;
  for(NodeID &node_id : action_nodes) {
    nodeset.add(node_id);
  }

  EXPECT_FALSE(nodeset.empty());
  EXPECT_EQ(nodeset.size(), param.nb_nodes);
  EXPECT_TRUE(nodeset.check_format(param.format_after_action));
  for(NodeID &node_id : action_nodes) {
    EXPECT_TRUE(nodeset.contains(node_id));
  }
}

INSTANTIATE_TEST_SUITE_P(
    NodeSetTestAdd, NodeSetAddTest,
    ::testing::Values(
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{},
                           std::vector<NodeID>{0}, EMPTY, VALS, 1}, // empty to vals
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<NodeID>{1, 2}, VALS, VALS, 3}, // vals to vals, multiple adds
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<NodeID>{1, 1}, VALS, VALS, 2}, // vals to vals, duplicated adds
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<NodeID>{2, 3, 4, 5, 6}, VALS, BITMASK, 6}, // vals to bitmask
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}, {2, 3}},
            std::vector<NodeID>{1}, VALS, VALS, 4}, // vals to vals
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{1, NULL_NODE},
                                                                  {2, NULL_NODE},
                                                                  {3, NULL_NODE},
                                                                  {4, NULL_NODE},
                                                                  {5, NULL_NODE}},
            std::vector<NodeID>{6, 8}, BITMASK, BITMASK, 7}, // bitmask to bitmask
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}},
                           std::vector<NodeID>{0}, RANGES, RANGES,
                           2}, // ranges to ranges, duplicated adds
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}},
                           std::vector<NodeID>{2}, RANGES, RANGES, 3}, // grow range hi
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{1, 2}},
                           std::vector<NodeID>{0}, RANGES, RANGES, 3}, // grow range lo
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}},
                           std::vector<NodeID>{8, 9, 10}, RANGES, RANGES,
                           5}, // range to range
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}},
                           std::vector<NodeID>{8, 9, 11}, RANGES, BITMASK,
                           5}, // range to bitmask
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}, {3, 4}, {6, 7}},
            std::vector<NodeID>{9}, BITMASK, BITMASK, 7} // bitmask to bitmask
        ));

class NodeSetRemoveTest : public NodeSetAddRemoveTestBase {};

TEST_P(NodeSetRemoveTest, RemoveNode)
{
  std::vector<NodeID> &action_nodes = param.action_nodes;
  for(NodeID &node_id : action_nodes) {
    nodeset.remove(node_id);
  }

  EXPECT_EQ(nodeset.size(), param.nb_nodes);
  EXPECT_TRUE(nodeset.check_format(param.format_after_action));
  for(NodeID &node_id : action_nodes) {
    EXPECT_FALSE(nodeset.contains(node_id));
  }
}

INSTANTIATE_TEST_SUITE_P(
    NodeSetTestRemove, NodeSetRemoveTest,
    ::testing::Values(
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<NodeID>{0}, VALS, VALS, 0}, // vals to vals
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<NodeID>{0, 0}, VALS, VALS, 0}, // vals to vals, duplicated removes
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<NodeID>{1}, VALS, VALS, 1}, // vals, wrong
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 9}},
                           std::vector<NodeID>{11}, RANGES, RANGES, 10}, // ranges, wrong
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                               {0, NULL_NODE}, {1, NULL_NODE}, {2, NULL_NODE}},
                           std::vector<NodeID>{1, 2}, VALS, VALS,
                           1}, // vals to vals, multiple removes
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE},
                                                                  {1, NULL_NODE},
                                                                  {2, NULL_NODE},
                                                                  {3, NULL_NODE},
                                                                  {4, NULL_NODE}},
            std::vector<NodeID>{1, 2}, BITMASK, BITMASK, 3}, // bitmask to bitmask
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}},
                           std::vector<NodeID>{0, 1}, RANGES, RANGES,
                           0}, // range to range, trim lo, empty
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 9}},
                           std::vector<NodeID>{9}, RANGES, RANGES,
                           9}, // range to range, trim the hi
        AddRemoveTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 9}},
                           std::vector<NodeID>{5}, RANGES, RANGES,
                           9}, // range to range, break the range to two ranges
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}, {3, 4}},
            std::vector<NodeID>{0, 1}, RANGES, RANGES,
            2}, // range to range, remove the 1st range
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 9}, {11, 19}},
            std::vector<NodeID>{15}, RANGES, BITMASK, 18}, // range to bitmask
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}, {3, 4}, {6, 7}},
            std::vector<NodeID>{6, 7}, BITMASK, BITMASK, 4}, // bitmask to bitmask
        AddRemoveTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}, {3, 4}, {6, 7}},
            std::vector<NodeID>{0, 1, 3, 4, 6, 7}, BITMASK, BITMASK, 0}
        // bitmask to bitmask, empty
        ));

// 2. test add/remove range
// in arrage, we add nodes using add_range or add
// after arrange, the nodeset is in the format of ENC_EMPTY, ENC_VALS, ENC_RANGE or
// ENC_BITMASK

struct AddRemoveRangeTestParam {
  std::vector<std::pair<NodeID, std::optional<NodeID>>>
      arrage_nodes; // if second is -1, we use add(first), otherwise, we use
                    // add_range(first, second)
  std::vector<std::pair<NodeID, NodeID>> action_nodes;
  int format_pior_action;
  int format_after_action;
  size_t nb_nodes;
};

class NodeSetAddRemoveRangeTestBase
  : public NodeSetTestBase,
    public ::testing::WithParamInterface<AddRemoveRangeTestParam> {
protected:
  virtual void SetUp()
  {
    // action
    NodeSetTestBase::SetUp();
    param = GetParam();
    add_nodes(nodeset, param.arrage_nodes, param.format_pior_action);
  }

public:
  AddRemoveRangeTestParam param;
};

class NodeSetAddRangeTest : public NodeSetAddRemoveRangeTestBase {};

TEST_P(NodeSetAddRangeTest, AddRange)
{
  std::vector<std::pair<NodeID, NodeID>> &action_nodes = param.action_nodes;
  for(std::pair<NodeID, NodeID> &range : action_nodes) {
    nodeset.add_range(range.first, range.second);
  }

  EXPECT_FALSE(nodeset.empty());
  EXPECT_EQ(nodeset.size(), param.nb_nodes);
  EXPECT_TRUE(nodeset.check_format(param.format_after_action));
  for(std::pair<NodeID, NodeID> &range : action_nodes) {
    for(int i = range.first; i <= range.second; i++) {
      EXPECT_TRUE(nodeset.contains(i));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    NodeSetTestAddRange, NodeSetAddRangeTest,
    ::testing::Values(
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<std::pair<NodeID, NodeID>>{{0, 1}}, VALS, VALS,
            2}, // vals to vals
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<std::pair<NodeID, NodeID>>{{0, 0}}, VALS, VALS,
            1}, // vals to vals, lo=hi
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<std::pair<NodeID, NodeID>>{{2, 10}}, VALS, BITMASK,
            10}, // vals to bitmask
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}},
            std::vector<std::pair<NodeID, NodeID>>{{7, 8}}, RANGES, RANGES,
            4}, // rangs to ranges
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}},
            std::vector<std::pair<NodeID, NodeID>>{{2, 3}}, RANGES, RANGES,
            4}, // rangs to ranges, grow hi
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{2, 3}},
            std::vector<std::pair<NodeID, NodeID>>{{0, 1}}, RANGES, RANGES,
            4}, // rangs to ranges, grow lo
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}, {3, 4}},
            std::vector<std::pair<NodeID, NodeID>>{{7, 8}}, RANGES, BITMASK,
            6}, // rangs to bitmask
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}, {3, 4}, {7, 8}},
            std::vector<std::pair<NodeID, NodeID>>{{10, 11}}, BITMASK, BITMASK,
            8}, // bitmask to bitmask
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 10}},
            std::vector<std::pair<NodeID, NodeID>>{{6, 15}}, RANGES, BITMASK,
            16}, // ranges to bitmask, overlap
        AddRemoveRangeTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                                    {0, 65}, {67, 128}, {130, 256}},
                                std::vector<std::pair<NodeID, NodeID>>{{6, 250}}, BITMASK,
                                BITMASK, 257} // bitmask to bitmask, overlap
        ));

class NodeSetRemoveRangeTest : public NodeSetAddRemoveRangeTestBase {};

TEST_P(NodeSetRemoveRangeTest, RemoveRange)
{
  std::vector<std::pair<NodeID, NodeID>> &action_nodes = param.action_nodes;
  for(std::pair<NodeID, NodeID> &range : action_nodes) {
    nodeset.remove_range(range.first, range.second);
  }

  EXPECT_EQ(nodeset.size(), param.nb_nodes);
  EXPECT_TRUE(nodeset.check_format(param.format_after_action));
  for(std::pair<NodeID, NodeID> &range : action_nodes) {
    for(int i = range.first; i <= range.second; i++) {
      EXPECT_FALSE(nodeset.contains(i));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    NodeSetTestRemoveRange, NodeSetRemoveRangeTest,
    ::testing::Values(
        AddRemoveRangeTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{},
                                std::vector<std::pair<NodeID, NodeID>>{{0, 0}}, EMPTY,
                                EMPTY, 0}, // nothing to remove
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, NULL_NODE}},
            std::vector<std::pair<NodeID, NodeID>>{{0, 0}}, VALS, VALS,
            0}, // vals to vals, lo=hi
        AddRemoveRangeTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                                    {0, NULL_NODE}, {1, NULL_NODE}, {2, NULL_NODE}},
                                std::vector<std::pair<NodeID, NodeID>>{{0, 1}}, VALS,
                                VALS, 1}, // vals to vals
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 1}, {4, 5}},
            std::vector<std::pair<NodeID, NodeID>>{{0, 1}}, RANGES, RANGES,
            2}, // ranges to ranges
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 9}},
            std::vector<std::pair<NodeID, NodeID>>{{3, 4}}, RANGES, RANGES,
            8}, // ranges to ranges, overlapped
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 9}, {20, 29}},
            std::vector<std::pair<NodeID, NodeID>>{{3, 4}}, RANGES, BITMASK,
            18}, // ranges to bitmask, overlapped
        AddRemoveRangeTestParam{
            std::vector<std::pair<NodeID, std::optional<NodeID>>>{{0, 2}, {6, 8}},
            std::vector<std::pair<NodeID, NodeID>>{{1, 7}}, RANGES, RANGES,
            2}, // ranges to ranges, overlapped across two rangs
        AddRemoveRangeTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                                    {0, 65}, {67, 128}, {130, 256}},
                                std::vector<std::pair<NodeID, NodeID>>{{9, 250}}, BITMASK,
                                BITMASK, 15}, // bitmask to bitmask, overlapped
        AddRemoveRangeTestParam{std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                                    {0, 9}, {11, 19}, {21, 30}},
                                std::vector<std::pair<NodeID, NodeID>>{{0, 30}}, BITMASK,
                                BITMASK, 0} // bitmask to bitmask, empty
        ));

// 3. test copy, swap and iterator

struct NodeSetCopyAndIteratorTestParam {
  std::vector<std::pair<NodeID, std::optional<NodeID>>>
      arrage_nodes; // if second is -1, we use add(first), otherwise, we use
                    // add_range(first, second)
  int format;
};

class NodeSetCopyAndIteratorTest
  : public NodeSetTestBase,
    public ::testing::WithParamInterface<NodeSetCopyAndIteratorTestParam> {
protected:
  virtual void SetUp()
  {
    // action
    NodeSetTestBase::SetUp();
    param = GetParam();
    add_nodes(nodeset, param.arrage_nodes, param.format);

    // put all nodes into a set for verification
    for(std::pair<NodeID, std::optional<NodeID>> &range : param.arrage_nodes) {
      if(range.second == NULL_NODE) {
        verification_nodes.insert(range.first);
      } else {
        for(int i = range.first; i <= range.second; i++) {
          verification_nodes.insert(i);
        }
      }
    }
  }

public:
  NodeSetCopyAndIteratorTestParam param;
  std::set<NodeID> verification_nodes;
};

TEST_P(NodeSetCopyAndIteratorTest, NodeSetCopyConstructor)
{
  NodeSet new_nodeset(nodeset);

  EXPECT_EQ(new_nodeset.size(), nodeset.size());
  for(const NodeID &node : verification_nodes) {
    EXPECT_TRUE(new_nodeset.contains(node));
  }
}

TEST_P(NodeSetCopyAndIteratorTest, NodeSetCopyOperator)
{
  NodeSet new_nodeset;
  new_nodeset = nodeset;

  EXPECT_EQ(new_nodeset.size(), nodeset.size());
  for(const NodeID &node : verification_nodes) {
    EXPECT_TRUE(new_nodeset.contains(node));
  }
}

TEST_P(NodeSetCopyAndIteratorTest, NodeSetSwap)
{
  NodeSet new_nodeset;
  nodeset.swap(new_nodeset);

  EXPECT_TRUE(nodeset.empty());
  for(const NodeID &node : verification_nodes) {
    EXPECT_TRUE(new_nodeset.contains(node));
  }
  new_nodeset.clear();
}

TEST_P(NodeSetCopyAndIteratorTest, IterateNode)
{
  for(NodeSetIterator it = nodeset.begin(); it != nodeset.end(); it++) {
    std::set<NodeID>::iterator nodes_it = verification_nodes.find(*it);
    EXPECT_TRUE(nodes_it != verification_nodes.end());
  }
}

INSTANTIATE_TEST_SUITE_P(NodeSetCopyAndIterator, NodeSetCopyAndIteratorTest,
                         ::testing::Values(
                             NodeSetCopyAndIteratorTestParam{
                                 std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                                     {0, NULL_NODE}, {1, NULL_NODE}, {2, NULL_NODE}},
                                 VALS}, // vals
                             NodeSetCopyAndIteratorTestParam{
                                 std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                                     {0, 9}},
                                 RANGES}, // ranges
                             NodeSetCopyAndIteratorTestParam{
                                 std::vector<std::pair<NodeID, std::optional<NodeID>>>{
                                     {0, 2}, {5, 9}, {11, 16}},
                                 BITMASK} // bitmask
                             ));