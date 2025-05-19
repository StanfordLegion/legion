#include <gtest/gtest.h>
#include "realm/fragmented_message.h"

using namespace Realm;

class FragmentedMessageTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    total_chunks = 3;
    message = std::make_unique<FragmentedMessage>(total_chunks);
  }

  uint32_t total_chunks;
  std::unique_ptr<FragmentedMessage> message;
};

TEST_F(FragmentedMessageTest, AddChunkWorksCorrectly)
{
  const char chunk1[] = "Data";
  EXPECT_TRUE(message->add_chunk(0, chunk1, sizeof(chunk1) - 1));
  EXPECT_EQ(message->size(), sizeof(chunk1) - 1);
}

TEST_F(FragmentedMessageTest, AddDuplicateChunkIgnored)
{
  const char chunk1[] = "Data";
  message->add_chunk(0, chunk1, sizeof(chunk1));
  EXPECT_EQ(message->size(), sizeof(chunk1));
  message->add_chunk(0, chunk1, sizeof(chunk1)); // duplicate
  EXPECT_EQ(message->size(), sizeof(chunk1));    // unchanged
}

TEST_F(FragmentedMessageTest, IsCompleteReturnsCorrectly)
{
  const char chunk1[] = "One";
  const char chunk2[] = "Two";

  EXPECT_FALSE(message->is_complete());

  message->add_chunk(0, chunk1, sizeof(chunk1));
  EXPECT_FALSE(message->is_complete());

  message->add_chunk(1, chunk2, sizeof(chunk2));
  EXPECT_FALSE(message->is_complete());

  message->add_chunk(2, chunk1, sizeof(chunk1));
  EXPECT_TRUE(message->is_complete());
}

TEST_F(FragmentedMessageTest, ReassembleThrowsOnIncompleteMessage)
{
  const char chunk1[] = "Data";
  message->add_chunk(0, chunk1, sizeof(chunk1));
  std::vector<char> msg = message->reassemble();
  EXPECT_TRUE(msg.empty());
}

TEST_F(FragmentedMessageTest, ReassembleWorksCorrectly)
{
  const char chunk1[] = "Hello ";
  const char chunk2[] = "World";
  const char chunk3[] = "!";

  message->add_chunk(0, chunk1, sizeof(chunk1) - 1);
  message->add_chunk(1, chunk2, sizeof(chunk2) - 1);
  message->add_chunk(2, chunk3, sizeof(chunk3) - 1);

  ASSERT_TRUE(message->is_complete());

  std::vector<char> result = message->reassemble();
  std::string result_str(result.begin(), result.end());

  EXPECT_EQ(result_str, "Hello World!");
}

TEST_F(FragmentedMessageTest, ReassembleThrowsWhenIncomplete)
{
  const char chunk1[] = "Incomplete";
  EXPECT_TRUE(message->add_chunk(0, chunk1, sizeof(chunk1)));
  std::vector<char> msg = message->reassemble();
  EXPECT_TRUE(msg.empty());
}

TEST_F(FragmentedMessageTest, SizeCalculation)
{
  const char chunk1[] = "123";
  const char chunk2[] = "4567";

  EXPECT_EQ(message->size(), 0u);

  message->add_chunk(0, chunk1, 3);
  EXPECT_EQ(message->size(), 3u);

  message->add_chunk(1, chunk2, 4);
  EXPECT_EQ(message->size(), 7u);
}

TEST_F(FragmentedMessageTest, AddChunkOutOfBounds)
{
  const char chunk[] = "OutOfBounds";
  EXPECT_FALSE(message->add_chunk(5, chunk, sizeof(chunk)));
}

TEST_F(FragmentedMessageTest, EmptyReassemble)
{
  std::vector<char> msg = message->reassemble();
  EXPECT_TRUE(msg.empty());
}

TEST_F(FragmentedMessageTest, HandlesLargeChunks)
{
  std::vector<char> large_chunk(1024, 'A');
  message = std::make_unique<FragmentedMessage>(1);
  message->add_chunk(0, large_chunk.data(), large_chunk.size());

  ASSERT_TRUE(message->is_complete());
  std::vector<char> result = message->reassemble();
  EXPECT_EQ(result.size(), 1024u);
  EXPECT_EQ(result[0], 'A');
  EXPECT_EQ(result[1023], 'A');
}

TEST_F(FragmentedMessageTest, HandlesOutOfOrderChunks)
{
  const char chunk1[] = "Part 1 ";
  const char chunk2[] = "Part 2 ";
  const char chunk3[] = "Part 3";

  // Add chunks out of order
  message->add_chunk(2, chunk3, sizeof(chunk3) - 1);
  message->add_chunk(0, chunk1, sizeof(chunk1) - 1);
  message->add_chunk(1, chunk2, sizeof(chunk2) - 1);

  ASSERT_TRUE(message->is_complete());

  std::vector<char> result = message->reassemble();
  std::string result_str(result.begin(), result.end());

  EXPECT_EQ(result_str, "Part 1 Part 2 Part 3");
}
