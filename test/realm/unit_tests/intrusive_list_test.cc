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

class IntrusiveListTestWithParams
  : public testing::TestWithParam<std::vector<TestEntry>> {};

TEST_P(IntrusiveListTestWithParams, SwapLists)
{
  TestEntry::TestEntryList list;
  TestEntry::TestEntryList new_list;
  std::vector<TestEntry> orig_entries = GetParam();
  std::vector<TestEntry *> list_entries;
  for(size_t i = 0; i < orig_entries.size() - 1; i++) {
    list.push_back(&orig_entries[i]);
  }
  new_list.push_back(&orig_entries.back());

  new_list.swap(list);
  for(size_t i = 0; i < orig_entries.size() - 1; i++) {
    list_entries.push_back(new_list.pop_front());
  }
  TestEntry *last_entry = list.pop_front();

  for(size_t i = 0; i < orig_entries.size() - 1; i++) {
    EXPECT_EQ(list_entries[i]->value, orig_entries[i].value);
  }
  EXPECT_EQ(last_entry->value, orig_entries.back().value);
}

TEST_P(IntrusiveListTestWithParams, AbsortAppend)
{
  TestEntry::TestEntryList list;
  TestEntry::TestEntryList new_list;
  std::vector<TestEntry> orig_entries = GetParam();
  std::vector<TestEntry *> list_entries;
  for(size_t i = 0; i < orig_entries.size(); i++) {
    list.push_back(&orig_entries[i]);
  }

  new_list.absorb_append(list);
  for(size_t i = 0; i < orig_entries.size(); i++) {
    list_entries.push_back(new_list.pop_front());
  }

  for(size_t i = 0; i < orig_entries.size() - 1; i++) {
    EXPECT_EQ(list_entries[i]->value, orig_entries[i].value);
  }
  EXPECT_TRUE(list.empty());
}

INSTANTIATE_TEST_SUITE_P(
    ListTest, IntrusiveListTestWithParams,
    ::testing::Values(std::vector<TestEntry>{TestEntry(42), TestEntry(43), TestEntry(44)},
                      std::vector<TestEntry>{TestEntry(44), TestEntry(43)}));
