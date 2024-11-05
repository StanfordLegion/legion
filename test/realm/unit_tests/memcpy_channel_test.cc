#include "realm/transfer/channel.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

struct ChannelTestCase {};

class MemcpyChannelParamTest : public ::testing::TestWithParam<ChannelTestCase> {};

TEST_P(MemcpyChannelParamTest, Base)
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
