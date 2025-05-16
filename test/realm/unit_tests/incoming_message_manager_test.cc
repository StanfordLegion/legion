#include <gtest/gtest.h>
#include "realm/activemsg.h"
#include "realm/timers.h"
#include "realm/threads.h"
#include "realm/activemsg.h"
#include <vector>
#include <atomic>

using namespace Realm;

namespace {

  struct RxMessage {
    FragmentInfo frag_info;
    int dummy{0};

    // inline handler that simply records the payload we receive
    static std::vector<std::vector<char>> received_payloads;
    static std::atomic<int> call_count;

    static bool handle_inline(NodeID /*sender*/, const RxMessage & /*hdr*/,
                              const void *payload, size_t payload_size, TimeLimit /*tl*/)
    {
      const char *c = static_cast<const char *>(payload);
      RxMessage::received_payloads.emplace_back(c, c + payload_size);
      call_count.fetch_add(1, std::memory_order_relaxed);
      return true; // handled inline, so add_incoming_message returns true
    }

    // Provide a non-inline handler to satisfy ActiveMessageHandlerTable invariant
    static void handle_message(NodeID sender, const RxMessage &hdr, const void *payload,
                               size_t payload_size)
    {
      (void)sender;
      (void)hdr;
      handle_inline(sender, hdr, payload, payload_size, TimeLimit());
    }
  };
  std::vector<std::vector<char>> RxMessage::received_payloads;
  std::atomic<int> RxMessage::call_count{0};

  // force registration at static initialization time
  static ActiveMessageHandlerReg<RxMessage> rxmsg_reg;

  class IncomingMessageManagerTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      RxMessage::received_payloads.clear();
      RxMessage::call_count.store(0);

      activemsg_handler_table.construct_handler_table();
    }
  };

  TEST_F(IncomingMessageManagerTest, FragmentReassemblyViaAddIncoming)
  {
    CoreReservationSet crs(nullptr);
    const int nodes = 2;
    IncomingMessageManager mgr(nodes, /*dedicated_threads=*/0, crs);

    // prepare data to send
    const size_t max_payload = 10;
    std::vector<char> full_msg(37);
    for(size_t i = 0; i < full_msg.size(); ++i)
      full_msg[i] = static_cast<char>(i);

    // split into chunks
    size_t total_chunks = (full_msg.size() + max_payload - 1) / max_payload;
    uint64_t msg_id = 0xABCD1234ULL; // arbitrary message id
    NodeID sender = 1;

    unsigned short msgid = activemsg_handler_table.lookup_message_id<RxMessage>();

    size_t offset = 0;
    for(uint32_t chunk_id = 0; chunk_id < total_chunks; ++chunk_id) {
      size_t chunk_size = std::min(max_payload, full_msg.size() - offset);

      RxMessage hdr;
      hdr.frag_info.chunk_id = chunk_id;
      hdr.frag_info.total_chunks = static_cast<uint32_t>(total_chunks);
      hdr.frag_info.msg_id = msg_id;

      const void *payload_ptr = full_msg.data() + offset;

      bool handled = mgr.add_incoming_message(sender, msgid, &hdr, sizeof(hdr),
                                              PAYLOAD_COPY, payload_ptr, chunk_size,
                                              PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
      if(chunk_id < total_chunks - 1) {
        EXPECT_FALSE(handled) << "Intermediate fragment should not be handled.";
      } else {
        EXPECT_TRUE(handled) << "Last fragment should be handled inline.";
      }

      offset += chunk_size;
    }

    // Exactly one inline call should have occurred
    ASSERT_EQ(RxMessage::call_count.load(), 1);

    // The payload passed to the handler must equal the entire original message
    ASSERT_EQ(RxMessage::received_payloads.size(), 1u);
    const auto &payload0 = RxMessage::received_payloads.front();
    EXPECT_EQ(payload0.size(), full_msg.size());
    EXPECT_EQ(payload0, full_msg);
    mgr.shutdown();
  }

} // anonymous namespace
