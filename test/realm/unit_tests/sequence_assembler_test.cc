#include "realm/transfer/channel.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

TEST(WrappingFIFOIteratorTest, EmptySpan)
{
  SequenceAssembler assembler;
  EXPECT_TRUE(assembler.empty());
  EXPECT_EQ(assembler.add_span(0, 0), 0);
  EXPECT_EQ(assembler.span_exists(0, 0), 0);
}

TEST(WrappingFIFOIteratorTest, SingleSpan)
{
  SequenceAssembler assembler;
  EXPECT_TRUE(assembler.empty());

  const size_t count = 16;
  const size_t start = 4;

  EXPECT_EQ(assembler.add_span(0, count), count);

  // exists
  EXPECT_EQ(assembler.span_exists(0, count - 1), count - 1);
  EXPECT_EQ(assembler.span_exists(start, count - 1), count - start);
  EXPECT_EQ(assembler.span_exists(0, count), count);
  EXPECT_EQ(assembler.span_exists(0, count + 1), count);
  EXPECT_EQ(assembler.span_exists(0, count * 2), count);

  // doesn't exist
  EXPECT_EQ(assembler.span_exists(0, 0), 0);
  EXPECT_EQ(assembler.span_exists(count, count * 2), 0);
}

TEST(WrappingFIFOIteratorTest, NocontigOutOfOrderSpans)
{
  SequenceAssembler assembler;

  const size_t beg_span_a = 0;
  const size_t end_span_a = 16;
  const size_t beg_span_b = 17;
  const size_t end_span_b = 24;

  EXPECT_EQ(assembler.add_span(beg_span_b, end_span_b), 0);
  EXPECT_EQ(assembler.add_span(beg_span_a, end_span_a), end_span_a - beg_span_a);
  
  EXPECT_EQ(assembler.span_exists(beg_span_a, end_span_a), end_span_a);
  EXPECT_EQ(assembler.span_exists(beg_span_b, end_span_b), end_span_b);
  EXPECT_EQ(assembler.span_exists(0, end_span_b), end_span_a);

  EXPECT_FALSE(assembler.empty());
}

TEST(WrappingFIFOIteratorTest, ContigSpans)
{
  SequenceAssembler assembler;

  const size_t beg_span_a = 0;
  const size_t end_span_a = 16;
  const size_t beg_span_b = 16;
  const size_t end_span_b = 24;

  EXPECT_EQ(assembler.add_span(beg_span_a, end_span_a), end_span_a);
  EXPECT_EQ(assembler.add_span(beg_span_b, end_span_b), end_span_b);

  EXPECT_EQ(assembler.span_exists(beg_span_a, end_span_a), end_span_a);
  EXPECT_EQ(assembler.span_exists(beg_span_b, end_span_b), end_span_b);
  EXPECT_FALSE(assembler.empty());
}
