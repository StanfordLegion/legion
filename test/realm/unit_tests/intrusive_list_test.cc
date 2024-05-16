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
  EXPECT_TRUE(list.empty());
  TestEntry *entry = list.pop_front();
  EXPECT_EQ(entry, nullptr);
}

TEST(IntrusiveListTest, SingleEntryPushBackPopFront)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);
  list.push_back(&object);
  EXPECT_FALSE(list.empty());
  {
    TestEntry *entry = list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object.value);
  }
  EXPECT_TRUE(list.empty());

  {
    TestEntry *entry = list.pop_front();
    EXPECT_EQ(entry, nullptr);
  }
}

TEST(IntrusiveListTest, SingleEntryPushFrontPopFront)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);
  list.push_front(&object);
  EXPECT_FALSE(list.empty());
  {
    TestEntry *entry = list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object.value);
  }
  EXPECT_TRUE(list.empty());

  {
    TestEntry *entry = list.pop_front();
    EXPECT_EQ(entry, nullptr);
  }
}

TEST(IntrusiveListTest, PushFrontSameEntry)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);
  list.push_back(&object);
  list.push_back(&object);

  EXPECT_FALSE(list.empty());

  {
    TestEntry *entry = list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object.value);
  }

  EXPECT_FALSE(list.empty());

  {
    TestEntry *entry = list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object.value);
  }

  EXPECT_TRUE(list.empty());
}

TEST(IntrusiveListTest, EraseEntry)
{
  TestEntry::TestEntryList list;
  TestEntry object(42);
  list.push_front(&object);
  EXPECT_FALSE(list.empty());

  list.erase(&object);
  EXPECT_TRUE(list.empty());
  // erase same object again
  list.erase(&object);
  EXPECT_TRUE(list.empty());
}

TEST(IntrusiveListTest, SwapLists)
{
  TestEntry::TestEntryList list;
  TestEntry object1(42);
  list.push_back(&object1);
  TestEntry object2(43);
  list.push_back(&object2);
  EXPECT_FALSE(list.empty());

  TestEntry::TestEntryList new_list;
  TestEntry object3(44);
  new_list.push_back(&object3);

  new_list.swap(list);
  EXPECT_FALSE(new_list.empty());
  EXPECT_FALSE(list.empty());

  {
    TestEntry *entry = new_list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object1.value);
  }

  {
    TestEntry *entry = new_list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object2.value);
  }

  {
    TestEntry *entry = list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object3.value);
  }
}

TEST(IntrusiveListTest, AbsortAppend)
{
  TestEntry::TestEntryList list;
  TestEntry object1(42);
  list.push_back(&object1);
  TestEntry object2(43);
  list.push_back(&object2);
  EXPECT_FALSE(list.empty());

  TestEntry::TestEntryList new_list;
  new_list.absorb_append(list);

  EXPECT_FALSE(new_list.empty());
  EXPECT_TRUE(list.empty());

  {
    TestEntry *entry = new_list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object1.value);
  }

  {
    TestEntry *entry = new_list.pop_front();
    EXPECT_NE(entry, nullptr);
    EXPECT_EQ(entry->value, object2.value);
  }
}
