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
}

TEST(PriorityQueueTest, GetEntryFromEmptyQueue)
{
  PriorityQueue<Entry *, DummyLock> queue;

  int priority = 0;
  Entry *entry = queue.get(&priority, 3);

  EXPECT_EQ(entry, nullptr);
  EXPECT_EQ(priority, 0);
}

TEST(PriorityQueueTest, SingleEntryPutGet)
{
  PriorityQueue<Entry *, DummyLock> queue;
  Entry entry1(42);

  queue.put(&entry1, 0);
  Entry *entry = queue.get(0);

  EXPECT_EQ(entry->value, entry1.value);
}

TEST(PriorityQueueTest, SingleEntryPutPeek)
{
  PriorityQueue<Entry *, DummyLock> queue;
  Entry entry1(42);

  queue.put(&entry1, 0);
  Entry *entry = queue.peek(0);

  EXPECT_EQ(entry->value, entry1.value);
  EXPECT_FALSE(queue.empty());
}

TEST(PriorityQueueTest, MultipEntriesSamePriority)
{
  PriorityQueue<Entry *, DummyLock> queue;
  Entry entry1(42);
  Entry entry2(43);

  queue.put(&entry1, 0);
  queue.put(&entry2, 0);
  Entry *res_entry1 = queue.get(0);
  Entry *res_entry2 = queue.get(0);

  EXPECT_EQ(res_entry1->value, entry1.value);
  EXPECT_EQ(res_entry2->value, entry2.value);
  EXPECT_TRUE(queue.empty());
}

TEST(PriorityQueueTest, MultipEntriesDifferentPriority)
{
  PriorityQueue<Entry *, DummyLock> queue;
  Entry entry1(42);
  Entry entry2(43);
  int priority1 = 0;
  int priority2 = 0;

  queue.put(&entry2, 1);
  queue.put(&entry1, 2);
  Entry *res_entry1 = queue.get(&priority1);
  Entry *res_entry2 = queue.get(&priority2);

  EXPECT_EQ(res_entry1->value, entry1.value);
  EXPECT_EQ(priority1, 2);
  EXPECT_EQ(res_entry2->value, entry2.value);
  EXPECT_EQ(priority2, 1);
  EXPECT_TRUE(queue.empty());
}

TEST(PriorityQueueTest, MultipEntriesFilterByPriority)
{
  PriorityQueue<Entry *, DummyLock> queue;
  Entry entry1(42);
  Entry entry2(43);
  int priority1 = 0;
  int priority2 = 0;

  queue.put(&entry1, 1);
  queue.put(&entry2, 2);
  Entry *res_entry1 = queue.get(&priority1, 3);
  Entry *res_entry2 = queue.get(&priority2, 1);

  EXPECT_EQ(res_entry1, nullptr);
  EXPECT_EQ(priority1, 0);
  EXPECT_EQ(res_entry2->value, entry2.value);
  EXPECT_EQ(priority2, 2);
}

TEST(PriorityQueueTest, MultipEntriesFilterByPriorityByPeek)
{
  PriorityQueue<Entry *, DummyLock> queue;
  Entry entry1(42);
  Entry entry2(43);
  int priority1 = 0;
  int priority2 = 0;

  queue.put(&entry1, 1);
  queue.put(&entry2, 2);
  Entry *res_entry1 = queue.peek(&priority1, 3);
  Entry *res_entry2 = queue.peek(&priority2, 1);

  EXPECT_EQ(res_entry1, nullptr);
  EXPECT_EQ(priority1, 0);
  EXPECT_EQ(res_entry2->value, entry2.value);
  EXPECT_EQ(priority2, 2);
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
  Entry entry1(42);
  Entry entry2(43);
  Entry entry3(44);
  int priority1 = 0;
  int priority2 = 0;
  int priority3 = 0;

  EXPECT_CALL(callback, item_available(&entry1, 1))
      .Times(1)
      .WillOnce(::testing::Return(false));
  EXPECT_CALL(callback, item_available(&entry2, 2))
      .Times(1)
      .WillOnce(::testing::Return(false));

  queue.add_subscription(&callback);
  queue.put(&entry1, 1);
  queue.put(&entry2, 2);
  Entry *res_entry1 = queue.get(&priority1, 0);
  Entry *res_entry2 = queue.peek(&priority2, 0);

  queue.remove_subscription(&callback);
  queue.put(&entry3, 2);
  Entry *res_entry3 = queue.get(&priority3, 0);

  EXPECT_EQ(res_entry1->value, entry2.value);
  EXPECT_EQ(priority1, 2);
  EXPECT_EQ(res_entry2->value, entry1.value);
  EXPECT_EQ(priority2, 1);
  EXPECT_EQ(res_entry3->value, entry3.value);
  EXPECT_EQ(priority3, 2);
}
