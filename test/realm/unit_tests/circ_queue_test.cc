#include "realm/circ_queue.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <deque>

using namespace Realm;

struct Entry {
  Entry(int _value = 0)
    : value(_value)
  {}
  int value;
};

// use templated test for testing empty circ_queue
template <typename T>
class EmptyCirQueueTest : public testing::Test {};

template <unsigned INTSIZE, unsigned INIT_CAPACITY>
struct EmptyCircQueueParam {
  static constexpr unsigned intsize = INTSIZE;
  static constexpr unsigned init_capacity = INIT_CAPACITY;
};

using test_types =
    ::testing::Types<EmptyCircQueueParam<4, 0>, EmptyCircQueueParam<8, 16>>;

TYPED_TEST_SUITE(EmptyCirQueueTest, test_types);

TYPED_TEST(EmptyCirQueueTest, TestEmptyQueue)
{
  static constexpr std::size_t intsize = TypeParam::intsize;
  static constexpr std::size_t init_capacity = TypeParam::init_capacity;
  CircularQueue<Entry, intsize> circ_queue{init_capacity};

  ASSERT_EQ(circ_queue.size(), 0);
  ASSERT_TRUE(circ_queue.empty());
  ASSERT_EQ(circ_queue.capacity(), intsize);
}

// gtest does not support parameter and template, so we pick the
// default INTSIZE and then use parameter to test other methods

struct CircQueueBaseParam {
  size_t init_capacity;
  int growth_factor;
  std::vector<Entry> arrange_entries;
};

class CircQueueTestBase : public ::testing::Test {
protected:
  virtual void SetUp()
  {
    circ_queue =
        new CircularQueue<Entry>(base_param->init_capacity, base_param->growth_factor);
    for(Entry &entry : base_param->arrange_entries) {
      circ_queue->push_back(entry);
    }
  }

  virtual void TearDown() { delete circ_queue; }

  bool verify_with_deque(void)
  {
    // first, check if two sizes are equal
    if(circ_queue->size() != verification_deque.size()) {
      return false;
    }
    CircularQueueIterator<Entry, 4> it_cq = circ_queue->begin();
    std::deque<Entry>::iterator it_dq = verification_deque.begin();
    while(it_cq != circ_queue->end() && it_dq != verification_deque.end()) {
      if(it_cq->value != it_dq->value) {
        return false;
      }
      it_cq++;
      it_dq++;
    }
    return true;
  }

protected:
  CircQueueBaseParam *base_param;
  CircularQueue<Entry, 4> *circ_queue;
  std::deque<Entry> verification_deque; // deque used for verification with CircularQueue
};

// push_front and push_back
struct CircQueuePushParam {
  CircQueueBaseParam base_param;
  std::vector<Entry> action_entries;
};

class CircQueuePushTest : public CircQueueTestBase,
                          public ::testing::WithParamInterface<CircQueuePushParam> {
protected:
  virtual void SetUp()
  {
    param = GetParam();
    base_param = &param.base_param;
    CircQueueTestBase::SetUp();
    for(Entry &entry : base_param->arrange_entries) {
      verification_deque.push_back(entry);
    }
  }

protected:
  CircQueuePushParam param;
};

TEST_P(CircQueuePushTest, TestPushFront)
{
  // arrange, setup verification deque
  for(Entry &entry : param.action_entries) {
    verification_deque.push_front(entry);
  }

  // action
  for(Entry &entry : param.action_entries) {
    circ_queue->push_front(entry);
  }

  // assert
  Entry &front = circ_queue->front();
  ASSERT_EQ(front.value, param.action_entries.back().value);
  ASSERT_TRUE(verify_with_deque());
}

TEST_P(CircQueuePushTest, TestPushBack)
{
  // arrange, setup verification deque
  for(Entry &entry : param.action_entries) {
    verification_deque.push_back(entry);
  }

  // action
  for(Entry &entry : param.action_entries) {
    circ_queue->push_back(entry);
  }

  // assert
  Entry &back = circ_queue->back();
  ASSERT_EQ(back.value, param.action_entries.back().value);
  ASSERT_TRUE(verify_with_deque());
}

INSTANTIATE_TEST_SUITE_P(
    CircQueuePush, CircQueuePushTest,
    ::testing::Values(
        // negative growth_factor
        CircQueuePushParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
            std::vector<Entry>{Entry(2), Entry(3)}}, // within the initial size
        CircQueuePushParam{{4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
                           std::vector<Entry>{Entry(2), Entry(3), Entry(4), Entry(5),
                                              Entry(6)}}, // outside the initial size
        // positive growth_factor
        CircQueuePushParam{
            {4, 2, std::vector<Entry>{Entry(0), Entry(1)}},
            std::vector<Entry>{Entry(2), Entry(3)}}, // within the initial size
        CircQueuePushParam{{4, 2, std::vector<Entry>{Entry(0), Entry(1)}},
                           std::vector<Entry>{Entry(2), Entry(3), Entry(4), Entry(5),
                                              Entry(6)}}) // outside the initial size
);

// pop_front and pop_back
struct CircQueuePopParam {
  CircQueueBaseParam base_param;
  size_t num_action_entries;
};

class CircQueuePopTest : public CircQueueTestBase,
                         public ::testing::WithParamInterface<CircQueuePopParam> {
protected:
  virtual void SetUp()
  {
    param = GetParam();
    base_param = &param.base_param;
    CircQueueTestBase::SetUp();
    for(Entry &entry : base_param->arrange_entries) {
      verification_deque.push_back(entry);
    }
  }

protected:
  CircQueuePopParam param;
};

TEST_P(CircQueuePopTest, TestPopFront)
{
  // arrange, setup verification deque
  for(size_t i = 0; i < param.num_action_entries; i++) {
    verification_deque.pop_front();
  }

  // action
  for(size_t i = 0; i < param.num_action_entries; i++) {
    circ_queue->pop_front();
  }

  // assert
  if(circ_queue->size() > 0) {
    Entry &front = circ_queue->front();
    ASSERT_EQ(front.value, verification_deque.front().value);
  }
  ASSERT_TRUE(verify_with_deque());
}

TEST_P(CircQueuePopTest, TestPopBack)
{
  // arrange, setup verification deque
  for(size_t i = 0; i < param.num_action_entries; i++) {
    verification_deque.pop_back();
  }

  // action
  for(size_t i = 0; i < param.num_action_entries; i++) {
    circ_queue->pop_back();
  }

  // assert
  if(circ_queue->size() > 0) {
    Entry &back = circ_queue->back();
    ASSERT_EQ(back.value, verification_deque.back().value);
  }
  ASSERT_TRUE(verify_with_deque());
}

INSTANTIATE_TEST_SUITE_P(
    CircQueuePop, CircQueuePopTest,
    ::testing::Values(
        // negative growth_factor
        CircQueuePopParam{{4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
                          1}, // within the initial size, remove 1
        CircQueuePopParam{{4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
                          2}, // within the initial size, remove all
        CircQueuePopParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1), Entry(2), Entry(3), Entry(4)}},
            2}, // within the initial size, remove 2
        CircQueuePopParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1), Entry(2), Entry(3), Entry(4)}},
            5}, // within the initial size, remove all
        // postive growth_factor
        CircQueuePopParam{{4, 2, std::vector<Entry>{Entry(0), Entry(1)}},
                          1}, // within the initial size, remove 1
        CircQueuePopParam{{4, 2, std::vector<Entry>{Entry(0), Entry(1)}},
                          2}, // within the initial size, remove all
        CircQueuePopParam{
            {4, 2, std::vector<Entry>{Entry(0), Entry(1), Entry(2), Entry(3), Entry(4)}},
            2}, // within the initial size, remove 2
        CircQueuePopParam{
            {4, 2, std::vector<Entry>{Entry(0), Entry(1), Entry(2), Entry(3), Entry(4)}},
            5}) // within the initial size, remove all
);

// reserve
struct CircQueueReserveParam {
  CircQueueBaseParam base_param;
  size_t num_action_entries;
  size_t capacity;
};

class CircQueueReserveTest : public CircQueueTestBase,
                             public ::testing::WithParamInterface<CircQueueReserveParam> {
protected:
  virtual void SetUp()
  {
    param = GetParam();
    base_param = &param.base_param;
    CircQueueTestBase::SetUp();
  }

protected:
  CircQueueReserveParam param;
};

TEST_P(CircQueueReserveTest, TestReserve)
{
  // action
  circ_queue->reserve(param.num_action_entries);

  // assert
  ASSERT_EQ(circ_queue->capacity(), param.capacity);
}

INSTANTIATE_TEST_SUITE_P(
    CircQueueReserve, CircQueueReserveTest,
    ::testing::Values(
        CircQueueReserveParam{{4, -2, std::vector<Entry>{}}, 6, 6}, // empty
        CircQueueReserveParam{{4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
                              2,
                              4}, // new_capacity <= max_size
        CircQueueReserveParam{{4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
                              6,
                              6}, // new_capacity > max_size
        CircQueueReserveParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1), Entry(2), Entry(3), Entry(4)}},
            8,
            8}) // new_capacity > max_size, has external buffer
);

// swap
struct CircQueueSwapParam {
  CircQueueBaseParam base_param;
  std::vector<Entry> action_entries;
};

class CircQueueSwapTest : public CircQueueTestBase,
                          public ::testing::WithParamInterface<CircQueueSwapParam> {
protected:
  virtual void SetUp()
  {
    param = GetParam();
    base_param = &param.base_param;
    CircQueueTestBase::SetUp();
    for(Entry &entry : param.action_entries) {
      verification_deque.push_back(entry);
    }
  }

protected:
  CircQueueSwapParam param;
};

TEST_P(CircQueueSwapTest, TestSwap)
{
  // arrange
  CircularQueue<Entry, 4> swap_queue{param.base_param.init_capacity,
                                     param.base_param.growth_factor};
  for(Entry &entry : param.action_entries) {
    swap_queue.push_back(entry);
  }

  // action
  circ_queue->swap(swap_queue);

  // assert
  ASSERT_TRUE(verify_with_deque());
}

INSTANTIATE_TEST_SUITE_P(
    CircQueueSwap, CircQueueSwapTest,
    ::testing::Values(
        CircQueueSwapParam{
            {4, -2, std::vector<Entry>{}},
            std::vector<Entry>{Entry(2),
                               Entry(3)}}, // NOT (current_size > 0) && (external_buffer
                                           // == 0) AND (swap_with.current_size > 0) &&
                                           // (swap_with.external_buffer == 0)
        CircQueueSwapParam{
            {4, -2, std::vector<Entry>{}},
            std::vector<Entry>{Entry(2), Entry(3), Entry(4), Entry(5),
                               Entry(6)}}, // NOT (current_size > 0) && (external_buffer
                                           // == 0) AND NOT (swap_with.current_size > 0)
                                           // && (swap_with.external_buffer == 0)
        // CircQueueSwapParam{{4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
        // std::vector<Entry>{Entry(2), Entry(3)}}, // failed (current_size > 0) &&
        // (external_buffer == 0) AND (swap_with.current_size > 0) &&
        // (swap_with.external_buffer == 0)
        CircQueueSwapParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
            std::vector<Entry>{}}, // (current_size > 0) && (external_buffer == 0) AND NOT
                                   // (swap_with.current_size > 0) &&
                                   // (swap_with.external_buffer == 0)
        CircQueueSwapParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1)}},
            std::vector<Entry>{Entry(2), Entry(3), Entry(4), Entry(5),
                               Entry(6)}}) // (current_size > 0) && (external_buffer == 0)
                                           // AND NOT (swap_with.current_size > 0) &&
                                           // (swap_with.external_buffer == 0)
);

// clear
struct CircQueueClearParam {
  CircQueueBaseParam base_param;
};

class CircQueueClearTest : public CircQueueTestBase,
                           public ::testing::WithParamInterface<CircQueueClearParam> {
protected:
  virtual void SetUp()
  {
    param = GetParam();
    base_param = &param.base_param;
    CircQueueTestBase::SetUp();
  }

protected:
  CircQueueClearParam param;
};

TEST_P(CircQueueClearTest, TestClear)
{
  // action
  circ_queue->clear();

  // assert
  ASSERT_TRUE(circ_queue->empty());
  ASSERT_EQ(circ_queue->size(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    CircQueueClear, CircQueueClearTest,
    ::testing::Values(
        CircQueueClearParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1)}}}, // no external buffer
        CircQueueClearParam{
            {4, -2,
             std::vector<Entry>{Entry(0), Entry(1), Entry(2), Entry(3), Entry(4),
                                Entry(5)}}}) // has external buffer
);

// iterator
struct CircQueueIteratorParam {
  CircQueueBaseParam base_param;
};

class CircQueueIteratorTest
  : public CircQueueTestBase,
    public ::testing::WithParamInterface<CircQueueIteratorParam> {
protected:
  virtual void SetUp()
  {
    param = GetParam();
    base_param = &param.base_param;
    CircQueueTestBase::SetUp();
    for(Entry &entry : base_param->arrange_entries) {
      verification_deque.push_back(entry);
    }
  }

protected:
  CircQueueIteratorParam param;
};

TEST_P(CircQueueIteratorTest, TestPostfix)
{
  // assert
  ASSERT_EQ(circ_queue->size(), verification_deque.size());
  CircularQueueIterator<Entry, 4> it_cq = circ_queue->begin();
  std::deque<Entry>::iterator it_dq = verification_deque.begin();
  while(it_cq != circ_queue->end() && it_dq != verification_deque.end()) {
    // test -> and *
    Entry entry = *it_cq;
    ASSERT_EQ(it_cq->value, it_dq->value);
    ASSERT_EQ(entry.value, it_dq->value);
    it_cq++;
    it_dq++;
  }
}

TEST_P(CircQueueIteratorTest, TestPrefix)
{
  // assert
  ASSERT_EQ(circ_queue->size(), verification_deque.size());
  CircularQueueIterator<Entry, 4> it_cq = circ_queue->begin();
  std::deque<Entry>::iterator it_dq = verification_deque.begin();
  while(it_cq != circ_queue->end() && it_dq != verification_deque.end()) {
    // test -> and *
    Entry entry = *it_cq;
    ASSERT_EQ(it_cq->value, it_dq->value);
    ASSERT_EQ(entry.value, it_dq->value);
    ++it_cq;
    it_dq++;
  }
}

INSTANTIATE_TEST_SUITE_P(
    CircQueueIterator, CircQueueIteratorTest,
    ::testing::Values(
        CircQueueIteratorParam{
            {4, -2, std::vector<Entry>{Entry(0), Entry(1)}}}, // no external buffer
        CircQueueIteratorParam{
            {4, -2,
             std::vector<Entry>{Entry(0), Entry(1), Entry(2), Entry(3), Entry(4),
                                Entry(5)}}}) // has external buffer
);