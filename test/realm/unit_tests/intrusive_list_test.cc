#include <cstddef>
#include <ostream>
#include "realm/lists.h"

#include <gtest/gtest.h>

using namespace Realm;

struct TestEntryLock {
  void lock(void) {}
  void unlock(void) {}
};

struct TestEntry {
  TestEntry(int _value = 0)
    : value(_value)
  {}

  int value;
  IntrusiveListLink<TestEntry> ew_list_link;
  REALM_PMTA_DEFN(TestEntry, IntrusiveListLink<TestEntry>, ew_list_link);
  typedef IntrusiveList<TestEntry, REALM_PMTA_USE(TestEntry, ew_list_link), TestEntryLock>
      TestEntryList;
};

TEST(IntrusiveListTest, CheckEmptyList)
{
  TestEntry::TestEntryList list;

  TestEntry *entry = list.pop_front();

  EXPECT_EQ(entry, nullptr);
  EXPECT_TRUE(list.empty());
}

TEST(IntrusiveListTest, CheckNonEmptyList)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);

  list.push_back(&object);

  EXPECT_FALSE(list.empty());
}

TEST(IntrusiveListTest, SingleEntryPushBackPopFront)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);

  list.push_back(&object);
  TestEntry *entry = list.pop_front();

  EXPECT_EQ(entry->value, object.value);
}

TEST(IntrusiveListTest, SingleEntryPushFrontPopFront)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);

  list.push_front(&object);
  TestEntry *entry = list.pop_front();

  EXPECT_EQ(entry->value, object.value);
}

TEST(IntrusiveListTest, PushFrontSameEntry)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);

  list.push_back(&object);
  list.push_back(&object);
  TestEntry *entry1 = list.pop_front();
  TestEntry *entry2 = list.pop_front();

  EXPECT_EQ(entry1->value, object.value);
  EXPECT_EQ(entry2->value, object.value);
}

TEST(IntrusiveListTest, EraseEntry)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);

  list.push_front(&object);
  list.erase(&object);

  EXPECT_TRUE(list.empty());
}

TEST(IntrusiveListTest, SwapLists)
{
  TestEntry::TestEntryList list;
  TestEntry::TestEntryList new_list;
  TestEntry object1(42);
  TestEntry object2(43);
  TestEntry object3(44);

  list.push_back(&object1);
  list.push_back(&object2);
  new_list.push_back(&object3);
  new_list.swap(list);
  TestEntry *entry1 = new_list.pop_front();
  TestEntry *entry2 = new_list.pop_front();
  TestEntry *entry3 = list.pop_front();

  EXPECT_EQ(entry1->value, object1.value);
  EXPECT_EQ(entry2->value, object2.value);
  EXPECT_EQ(entry3->value, object3.value);
}

TEST(IntrusiveListTest, AbsortAppend)
{
  TestEntry::TestEntryList list;
  TestEntry::TestEntryList new_list;

  TestEntry object1(42);
  TestEntry object2(43);

  list.push_back(&object1);
  list.push_back(&object2);
  new_list.absorb_append(list);
  TestEntry *entry1 = new_list.pop_front();
  TestEntry *entry2 = new_list.pop_front();

  EXPECT_EQ(entry1->value, object1.value);
  EXPECT_EQ(entry2->value, object2.value);
  EXPECT_TRUE(list.empty());
}
