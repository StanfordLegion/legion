/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Tests for ActiveMessage automatic payload fragmentation.
//
// The actual chunked send path (init_chunked_data / commit_chunked) requires
// a real network backend, so these tests verify the supporting machinery:
//   - is_wrapped_with_frag_info trait
//   - WrappedWithFragInfo header wrapping
//   - Dual handler registration (T and WrappedWithFragInfo<T>)
//   - End-to-end reassembly of wrapped fragments through IncomingMessageManager

#include <gtest/gtest.h>
#include "realm/activemsg.h"
#include "realm/timers.h"
#include "realm/threads.h"
#include <vector>
#include <atomic>

using namespace Realm;

namespace {

  // A test message type whose handler records what it receives
  struct FragTestMessage {
    int value{0};

    static std::vector<std::vector<char>> received_payloads;
    static std::vector<int> received_values;
    static std::atomic<int> call_count;

    static bool handle_inline(NodeID sender, const FragTestMessage &hdr,
                              const void *payload, size_t payload_size,
                              TimeLimit /*work_until*/)
    {
      (void)sender;
      const char *c = static_cast<const char *>(payload);
      received_payloads.emplace_back(c, c + payload_size);
      received_values.push_back(hdr.value);
      call_count.fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    static void handle_message(NodeID sender, const FragTestMessage &hdr,
                               const void *payload, size_t payload_size)
    {
      (void)sender;
      handle_inline(sender, hdr, payload, payload_size, TimeLimit());
    }
  };
  std::vector<std::vector<char>> FragTestMessage::received_payloads;
  std::vector<int> FragTestMessage::received_values;
  std::atomic<int> FragTestMessage::call_count{0};

  // Register the handler - this also auto-registers WrappedWithFragInfo<FragTestMessage>
  static ActiveMessageHandlerReg<FragTestMessage> frag_test_reg;

  class ActMsgFragmentationTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      FragTestMessage::received_payloads.clear();
      FragTestMessage::received_values.clear();
      FragTestMessage::call_count.store(0);
      activemsg_handler_table.construct_handler_table();
    }
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // Trait tests
  //

  TEST_F(ActMsgFragmentationTest, IsWrappedWithFragInfoTrait)
  {
    // Plain types should not match
    EXPECT_FALSE(is_wrapped_with_frag_info<FragTestMessage>::value);
    EXPECT_FALSE(is_wrapped_with_frag_info<int>::value);

    // Wrapped types should match
    EXPECT_TRUE(is_wrapped_with_frag_info<WrappedWithFragInfo<FragTestMessage>>::value);
    EXPECT_TRUE(is_wrapped_with_frag_info<WrappedWithFragInfo<int>>::value);

    // Double-wrapping should still match (outer layer is WrappedWithFragInfo)
    EXPECT_TRUE(is_wrapped_with_frag_info<
                WrappedWithFragInfo<WrappedWithFragInfo<FragTestMessage>>>::value);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // WrappedWithFragInfo structure tests
  //

  TEST_F(ActMsgFragmentationTest, WrappedHeaderAccess)
  {
    WrappedWithFragInfo<FragTestMessage> wrapped;
    wrapped.frag_info = {2, 5, 0xDEADBEEF};
    wrapped.user.value = 42;

    // operator-> and operator* should give access to user header
    EXPECT_EQ(wrapped->value, 42);
    EXPECT_EQ((*wrapped).value, 42);

    // frag_info should be independently accessible
    EXPECT_EQ(wrapped.frag_info.chunk_id, 2u);
    EXPECT_EQ(wrapped.frag_info.total_chunks, 5u);
    EXPECT_EQ(wrapped.frag_info.msg_id, 0xDEADBEEFULL);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // Dual handler registration tests
  //

  TEST_F(ActMsgFragmentationTest, DualHandlerRegistration)
  {
    // Both the plain type and the wrapped type should have valid message IDs
    // (lookup_message_id asserts internally if not found)
    unsigned short plain_id =
        activemsg_handler_table.lookup_message_id<FragTestMessage>();
    unsigned short wrapped_id =
        activemsg_handler_table.lookup_message_id<WrappedWithFragInfo<FragTestMessage>>();

    // They should be different IDs
    EXPECT_NE(plain_id, wrapped_id);

    // Both should have valid handler entries
    auto *plain_entry = activemsg_handler_table.lookup_message_handler(plain_id);
    auto *wrapped_entry = activemsg_handler_table.lookup_message_handler(wrapped_id);
    EXPECT_NE(plain_entry, nullptr);
    EXPECT_NE(wrapped_entry, nullptr);

    // The wrapped entry should have extract_frag_info set
    EXPECT_TRUE(wrapped_entry->extract_frag_info.has_value());
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // End-to-end reassembly tests
  //
  // These simulate what commit_chunked does: split a payload into chunks,
  // wrap each with WrappedWithFragInfo<T>, and feed them through
  // IncomingMessageManager. The handler should receive the complete
  // reassembled payload with the correct user header.
  //

  TEST_F(ActMsgFragmentationTest, SingleChunkReassembly)
  {
    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, /*dedicated_threads=*/0, crs);

    std::vector<char> data(20);
    for(size_t i = 0; i < data.size(); i++) {
      data[i] = static_cast<char>(i);
    }

    WrappedWithFragInfo<FragTestMessage> hdr;
    hdr.frag_info = {0, 1, 0x1234};
    hdr.user.value = 99;

    unsigned short msgid =
        activemsg_handler_table.lookup_message_id<WrappedWithFragInfo<FragTestMessage>>();

    bool handled = mgr.add_incoming_message(
        /*sender=*/1, msgid, &hdr, sizeof(hdr), PAYLOAD_COPY, data.data(), data.size(),
        PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());
    EXPECT_TRUE(handled);

    ASSERT_EQ(FragTestMessage::call_count.load(), 1);
    ASSERT_EQ(FragTestMessage::received_payloads.size(), 1u);
    EXPECT_EQ(FragTestMessage::received_payloads[0], data);
    EXPECT_EQ(FragTestMessage::received_values[0], 99);
    mgr.shutdown();
  }

  TEST_F(ActMsgFragmentationTest, MultiChunkReassembly)
  {
    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, /*dedicated_threads=*/0, crs);

    // Create a payload that will be split into 4 chunks of 16 bytes each
    const size_t max_chunk = 16;
    std::vector<char> full_data(60);
    for(size_t i = 0; i < full_data.size(); i++) {
      full_data[i] = static_cast<char>(i & 0xFF);
    }

    uint32_t total_chunks =
        static_cast<uint32_t>((full_data.size() + max_chunk - 1) / max_chunk);
    ASSERT_EQ(total_chunks, 4u);

    uint64_t msg_id = 0xCAFEBABE;
    NodeID sender = 1;
    unsigned short msgid =
        activemsg_handler_table.lookup_message_id<WrappedWithFragInfo<FragTestMessage>>();

    size_t offset = 0;
    for(uint32_t chunk_id = 0; chunk_id < total_chunks; chunk_id++) {
      size_t chunk_size = std::min(max_chunk, full_data.size() - offset);

      WrappedWithFragInfo<FragTestMessage> hdr;
      hdr.frag_info = {chunk_id, total_chunks, msg_id};
      hdr.user.value = 42;

      bool handled = mgr.add_incoming_message(
          sender, msgid, &hdr, sizeof(hdr), PAYLOAD_COPY, full_data.data() + offset,
          chunk_size, PAYLOAD_COPY, nullptr, 0, 0, TimeLimit());

      if(chunk_id < total_chunks - 1) {
        EXPECT_FALSE(handled) << "Intermediate chunk " << chunk_id
                              << " should not trigger handler";
      } else {
        EXPECT_TRUE(handled) << "Final chunk should trigger handler";
      }

      offset += chunk_size;
    }

    // Handler should have been called exactly once with the full payload
    ASSERT_EQ(FragTestMessage::call_count.load(), 1);
    ASSERT_EQ(FragTestMessage::received_payloads.size(), 1u);
    EXPECT_EQ(FragTestMessage::received_payloads[0], full_data);
    EXPECT_EQ(FragTestMessage::received_values[0], 42);
    mgr.shutdown();
  }

  TEST_F(ActMsgFragmentationTest, OutOfOrderChunkReassembly)
  {
    CoreReservationSet crs(nullptr);
    IncomingMessageManager mgr(2, /*dedicated_threads=*/0, crs);

    const size_t max_chunk = 10;
    std::vector<char> full_data(35);
    for(size_t i = 0; i < full_data.size(); i++) {
      full_data[i] = static_cast<char>('A' + (i % 26));
    }

    uint32_t total_chunks =
        static_cast<uint32_t>((full_data.size() + max_chunk - 1) / max_chunk);
    ASSERT_EQ(total_chunks, 4u);

    uint64_t msg_id = 0x12345678;
    NodeID sender = 1;
    unsigned short msgid =
        activemsg_handler_table.lookup_message_id<WrappedWithFragInfo<FragTestMessage>>();

    // Send chunks in reverse order: 3, 2, 1, 0
    uint32_t send_order[] = {3, 2, 1, 0};
    for(uint32_t chunk_id : send_order) {
      size_t offset = chunk_id * max_chunk;
      size_t chunk_size = std::min(max_chunk, full_data.size() - offset);

      WrappedWithFragInfo<FragTestMessage> hdr;
      hdr.frag_info = {chunk_id, total_chunks, msg_id};
      hdr.user.value = 7;

      mgr.add_incoming_message(sender, msgid, &hdr, sizeof(hdr), PAYLOAD_COPY,
                               full_data.data() + offset, chunk_size, PAYLOAD_COPY,
                               nullptr, 0, 0, TimeLimit());
    }

    ASSERT_EQ(FragTestMessage::call_count.load(), 1);
    ASSERT_EQ(FragTestMessage::received_payloads.size(), 1u);
    EXPECT_EQ(FragTestMessage::received_payloads[0], full_data);
    EXPECT_EQ(FragTestMessage::received_values[0], 7);
    mgr.shutdown();
  }

} // anonymous namespace
