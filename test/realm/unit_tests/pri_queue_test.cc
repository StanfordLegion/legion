#include "realm/pri_queue.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace Realm;

struct Entry {
  Entry(int _value = 0)
    : value(_value)
  {}
  int value;
};

class DummyLock {
public:
  void lock(void) {}
  void unlock(void) {}
};

using testing::_;
using testing::MockFunction;
using testing::Return;

TEST(PriorityQueueTest, CheckEmptyQueue)
{
  PriorityQueue<Entry *, DummyLock> queue;
  EXPECT_TRUE(queue.empty());

  {
    int priority = 0;
    Entry *entry = queue.get(&priority, 3);
    EXPECT_EQ(entry, nullptr);
    EXPECT_EQ(priority, 0);
  }
}

TEST(PriorityQueueTest, SingleEntryPutGet)
{
  PriorityQueue<Entry *, DummyLock> queue;

  Entry entry1(42);
  queue.put(&entry1, 0);

  {
    Entry *entry = queue.get(0);
    EXPECT_EQ(entry->value, entry1.value);
  }

  EXPECT_TRUE(queue.empty());
}

TEST(PriorityQueueTest, SingleEntryPutPeek)
{
  PriorityQueue<Entry *, DummyLock> queue;

  Entry entry1(42);
  queue.put(&entry1, 0);

  {
    Entry *entry = queue.peek(0);
    EXPECT_EQ(entry->value, entry1.value);
  }

  EXPECT_FALSE(queue.empty());
}

TEST(PriorityQueueTest, MultipEntriesSamePriority)
{
  PriorityQueue<Entry *, DummyLock> queue;

  Entry entry1(42);
  queue.put(&entry1, 0);

  Entry entry2(43);
  queue.put(&entry2, 0);

  {
    Entry *entry = queue.get(0);
    EXPECT_EQ(entry->value, entry1.value);
  }

  {
    Entry *entry = queue.get(0);
    EXPECT_EQ(entry->value, entry2.value);
  }

  EXPECT_TRUE(queue.empty());
}

TEST(PriorityQueueTest, MultipEntriesDifferentPriority)
{
  PriorityQueue<Entry *, DummyLock> queue;

  Entry entry1(42);
  queue.put(&entry1, 2);

  Entry entry2(43);
  queue.put(&entry2, 1);

  {
    int priority = 0;
    Entry *entry = queue.get(&priority);
    EXPECT_EQ(entry->value, entry1.value);
    EXPECT_EQ(priority, 2);
  }

  {
    int priority = 0;
    Entry *entry = queue.get(&priority);
    EXPECT_EQ(entry->value, entry2.value);
    EXPECT_EQ(priority, 1);
  }

  EXPECT_TRUE(queue.empty());
}

TEST(PriorityQueueTest, MultipEntriesFilterByPriority)
{
  PriorityQueue<Entry *, DummyLock> queue;

  Entry entry1(42);
  queue.put(&entry1, 1);

  Entry entry2(43);
  queue.put(&entry2, 2);

  {
    int priority = 0;
    Entry *entry = queue.get(&priority, 3);
    EXPECT_EQ(entry, nullptr);
    EXPECT_EQ(priority, 0);
  }

  {
    int priority = 0;
    Entry *entry = queue.get(&priority, 1);
    EXPECT_EQ(entry->value, entry2.value);
    EXPECT_EQ(priority, 2);
  }

  EXPECT_FALSE(queue.empty());
}

TEST(PriorityQueueTest, MultipEntriesFilterByPriorityByPeek)
{
  PriorityQueue<Entry *, DummyLock> queue;

  Entry entry1(42);
  queue.put(&entry1, 1);

  Entry entry2(43);
  queue.put(&entry2, 2);

  {
    int priority = 0;
    Entry *entry = queue.peek(&priority, 3);
    EXPECT_EQ(entry, nullptr);
    EXPECT_EQ(priority, 0);
  }

  {
    int priority = 0;
    Entry *entry = queue.peek(&priority, 1);
    EXPECT_EQ(entry->value, entry2.value);
    EXPECT_EQ(priority, 2);
  }

  EXPECT_FALSE(queue.empty());
}

class NotificationCallback
  : public PriorityQueue<Entry *, DummyLock>::NotificationCallback {
public:
  MOCK_METHOD(bool, item_available, (Entry * item, int item_priority), (override));
};

TEST(PriorityQueueTest, InvokeTwoNotificationCallbacks)
{
  PriorityQueue<Entry *, DummyLock> queue;
  NotificationCallback callback;
  queue.add_subscription(&callback);

  Entry entry1(42);
  EXPECT_CALL(callback, item_available(&entry1, 1))
      .Times(1)
      .WillOnce(::testing::Return(false));

  Entry entry2(43);
  EXPECT_CALL(callback, item_available(&entry2, 2))
      .Times(1)
      .WillOnce(::testing::Return(false));

  queue.put(&entry1, 1);
  queue.put(&entry2, 2);

  {
    int priority = 0;
    Entry *entry = queue.get(&priority, 0);
    EXPECT_EQ(entry->value, entry2.value);
    EXPECT_EQ(priority, 2);
  }

  {
    int priority = 0;
    Entry *entry = queue.peek(&priority, 0);
    EXPECT_EQ(entry->value, entry1.value);
    EXPECT_EQ(priority, 1);
  }

  queue.remove_subscription(&callback);

  {
    int priority = 0;
    Entry *entry = queue.get(&priority, 0);
    EXPECT_EQ(entry->value, entry1.value);
    EXPECT_EQ(priority, 1);
  }

  EXPECT_TRUE(queue.empty());
}
