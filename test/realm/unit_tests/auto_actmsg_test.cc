#include <gtest/gtest.h>
#include "realm/activemsg.h"
#include <vector>
#include <set>

using namespace Realm;

struct TestMessage {
  int data_field{0};
};

template <typename T>
class MockActiveMessage {
public:
  MockActiveMessage(NodeID /*target*/, size_t /*max_payload_size*/) {}

  void commit()
  {
    sent_messages.push_back({header_, payload_});
    payload_.clear();
  }
  void add_payload(const void *data, size_t datalen)
  {
    payload_.insert(payload_.end(), static_cast<const char *>(data),
                    static_cast<const char *>(data) + datalen);
  }

  const std::vector<char> &last_payload() const { return payload_; }

  static void reset() { sent_messages.clear(); }

  T *operator->() { return &header_; }
  const T *operator->() const { return &header_; }

  static inline std::vector<std::pair<T, std::vector<char>>> sent_messages;

private:
  T header_{};
  std::vector<char> payload_;
};

/**********************************************************************
 *                         Unit Tests                                 *
 **********************************************************************/

class ActiveMessageAutoTest : public ::testing::Test {
protected:
  NodeID target_node = 2;
  size_t max_payload = 16; // small for test purposes

  std::vector<char> make_payload(size_t n)
  {
    std::vector<char> p(n);
    for(size_t i = 0; i < n; ++i) {
      p[i] = static_cast<char>(i & 0xFF);
    }
    return p;
  }

  void SetUp() override { MockActiveMessage<WrappedWithFragInfo<TestMessage>>::reset(); }
};

template <typename Hdr>
using MockAmBuilder = MockActiveMessage<WrappedWithFragInfo<Hdr>>;

TEST_F(ActiveMessageAutoTest, NoFragmentForSmallPayload)
{
  ActiveMessageAuto<TestMessage, MockAmBuilder> msg(target_node, max_payload);
  auto data = make_payload(max_payload);
  msg.add_payload(data.data(), data.size());
  msg.commit();

  ASSERT_EQ(MockActiveMessage<WrappedWithFragInfo<TestMessage>>::sent_messages.size(), 1);
  const auto &header =
      MockActiveMessage<WrappedWithFragInfo<TestMessage>>::sent_messages[0].first;
  EXPECT_EQ(header.frag_info.chunk_id, 0u);
  EXPECT_EQ(header.frag_info.total_chunks, 1u);
}

TEST_F(ActiveMessageAutoTest, FragmentForOversizePayload)
{
  const size_t payload_size = 45;
  ActiveMessageAuto<TestMessage, MockAmBuilder> msg(target_node, max_payload);
  auto data = make_payload(payload_size);
  msg.add_payload(data.data(), data.size());
  msg.commit();

  size_t expected_chunks = (payload_size + max_payload - 1) / max_payload;
  ASSERT_EQ(MockActiveMessage<WrappedWithFragInfo<TestMessage>>::sent_messages.size(),
            expected_chunks);
  for(size_t i = 0; i < expected_chunks; ++i) {
    const auto &header =
        MockActiveMessage<WrappedWithFragInfo<TestMessage>>::sent_messages[i].first;
    EXPECT_EQ(header.frag_info.chunk_id, i);
    EXPECT_EQ(header.frag_info.total_chunks, expected_chunks);
  }
}

TEST_F(ActiveMessageAutoTest, ReassemblePayload)
{
  const size_t payload_size = 60;
  ActiveMessageAuto<TestMessage, MockAmBuilder> msg(target_node, max_payload);
  auto data = make_payload(payload_size);
  msg.add_payload(data.data(), data.size());
  msg.commit();

  // Reassemble using frag_info
  std::map<uint64_t, std::vector<char>> buffers;
  for(const auto &m :
      MockActiveMessage<WrappedWithFragInfo<TestMessage>>::sent_messages) {
    auto header = m.first;
    auto payload = m.second;
    uint64_t id = header.frag_info.msg_id;
    uint32_t offset = header.frag_info.chunk_id * max_payload;
    if(buffers[id].size() < payload_size)
      buffers[id].resize(payload_size);
    std::copy(payload.begin(), payload.end(), buffers[id].begin() + offset);
  }
  ASSERT_EQ(buffers.size(), 1u);
  auto reconstructed = buffers.begin()->second;
  EXPECT_EQ(reconstructed, data);
}
