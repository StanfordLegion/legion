#include "realm/repl_heap.h"
#include <gtest/gtest.h>

using namespace Realm;

// 1. test listener
class FakeListener : public ReplicatedHeap::Listener {
public:
  FakeListener(void) {}

  virtual void chunk_created(void *base, size_t bytes)
  {
    chunks.insert(std::pair<void *, size_t>(base, bytes));
    nb_chunk_created_calls++;
  }
  virtual void chunk_destroyed(void *base, size_t bytes)
  {
    chunks.erase(base);
    nb_chunk_destroyed_calls++;
  }

  virtual void data_updated(void *base, size_t bytes)
  {
    commit_write_chunks.insert(std::pair<void *, size_t>(base, bytes));
  }

  // the following two function is used for testing if chunk_created
  // chunk_destroyed and data_updated are called by the ReplicatedHeap
  bool validate_chunks(void)
  {
    if(chunks.empty()) {
      return false;
    }
    bool results = true;
    for(std::map<void *, size_t>::iterator it = chunks.begin(); it != chunks.end();
        it++) {
      results = results & has_chunk(it->first, it->second);
    }
    return results;
  }

  bool has_chunk(void *base, size_t bytes)
  {
    std::map<void *, size_t>::iterator it = chunks.find(base);
    if(it != chunks.end()) {
      return it->second == bytes;
    } else {
      return false;
    }
  }

  bool is_data_updated_called(void *base)
  {
    std::map<void *, size_t>::iterator it = commit_write_chunks.find(base);
    return it != commit_write_chunks.end();
  }

public:
  std::map<void *, size_t> chunks;
  std::map<void *, size_t> commit_write_chunks;
  int nb_chunk_created_calls = 0;
  int nb_chunk_destroyed_calls = 0;
};

class ReplicatedHeapListernerTest : public testing::Test {
protected:
  virtual void SetUp() { repl_heap.init(chunk_size, 128); }
  virtual void TearDown()
  {
    if(listener.chunks.size() > 0) {
      repl_heap.cleanup();
    }
  }
  FakeListener listener;
  ReplicatedHeap repl_heap;
  size_t chunk_size = 1024 * 10;
};

TEST_F(ReplicatedHeapListernerTest, AddListener)
{
  repl_heap.add_listener(&listener);

  EXPECT_TRUE(listener.validate_chunks());
  EXPECT_EQ(listener.nb_chunk_created_calls, 1);
}

TEST_F(ReplicatedHeapListernerTest, RemoveListener)
{
  repl_heap.add_listener(&listener);

  repl_heap.remove_listener(&listener);

  EXPECT_FALSE(listener.validate_chunks());
  EXPECT_EQ(listener.nb_chunk_destroyed_calls, 1);
}

TEST_F(ReplicatedHeapListernerTest, ReplHeapCleanup)
{
  repl_heap.add_listener(&listener);

  repl_heap.cleanup();

  EXPECT_FALSE(listener.validate_chunks());
  EXPECT_EQ(listener.nb_chunk_destroyed_calls, 1);
}

TEST_F(ReplicatedHeapListernerTest, DoubleRemove)
{
  repl_heap.add_listener(&listener);
  repl_heap.remove_listener(&listener);

  repl_heap.cleanup();

  // the remove_listener call removed the listener, so the cleanup call
  // won't trigger the callback again
  EXPECT_EQ(listener.nb_chunk_destroyed_calls, 1);
}

TEST_F(ReplicatedHeapListernerTest, CommitWrites)
{
  repl_heap.add_listener(&listener);

  repl_heap.commit_writes((void *)0xDEADBEEF, 0);

  EXPECT_TRUE(listener.is_data_updated_called((void *)0xDEADBEEF));
  EXPECT_FALSE(listener.is_data_updated_called(nullptr));
}

// 2. test allocate objects

// we need mock ReplicatedHeap because we need to access the protected members for
// verifications
class MockReplicatedHeap : public ReplicatedHeap {

public:
  uint64_t size_available(void)
  {
    uint64_t size = 0;
    for(const std::pair<const uint64_t, uint64_t> &kv : free_by_start) {
      size += kv.second;
    }
    return size;
  }

  static size_t calculate_actual_size(size_t bytes)
  {
    return sizeof(ObjectHeader) * (((bytes - 1) / sizeof(ObjectHeader)) + 2);
  }
};

struct ObjParam {
  size_t bytes;
  size_t alignment;
};

struct AllocateObjParam {
  std::vector<ObjParam> action_objs;
};

class ReplicatedHeapObjTestBase : public ::testing::Test {
protected:
  virtual void SetUp() { repl_heap.init(chunk_size, 128); }

  virtual void TearDown() { repl_heap.cleanup(); }
  MockReplicatedHeap repl_heap;
  static constexpr size_t chunk_size = 1024 * 10;
};

class ReplicatedHeapAllocateObjTest
  : public ReplicatedHeapObjTestBase,
    public ::testing::WithParamInterface<AllocateObjParam> {};

TEST_P(ReplicatedHeapAllocateObjTest, AllocateObj)
{
  AllocateObjParam param = GetParam();
  size_t actual_sizes = 0;
  for(ObjParam &obj : param.action_objs) {
    repl_heap.alloc_obj(obj.bytes, obj.alignment);
    actual_sizes += MockReplicatedHeap::calculate_actual_size(obj.bytes);
  }
  EXPECT_EQ(repl_heap.size_available(), chunk_size - actual_sizes);
}

INSTANTIATE_TEST_SUITE_P(
    ReplicatedHeapAllocateObj, ReplicatedHeapAllocateObjTest,
    ::testing::Values(AllocateObjParam{std::vector<ObjParam>{
                          {1024, sizeof(int)}}}, // single element
                      AllocateObjParam{std::vector<ObjParam>{
                          {1024, sizeof(int)},
                          {1024, sizeof(int)},
                          {1024, sizeof(int)}}}, // multiple same elements
                      AllocateObjParam{std::vector<ObjParam>{
                          {512, sizeof(int)}, {1024, sizeof(int)}, {2048, sizeof(int)}}}
                      // multiple different elements
                      ));

// 3. test free objects
struct FreeObjParam {
  std::vector<ObjParam> arrange_objs;
  std::vector<int> action_objs; // index of arrange objs
};

class ReplicatedHeapFreeObjTest : public ReplicatedHeapObjTestBase,
                                  public ::testing::WithParamInterface<FreeObjParam> {
protected:
  virtual void SetUp()
  {
    ReplicatedHeapObjTestBase::SetUp();
    param = GetParam();
    for(ObjParam &obj : param.arrange_objs) {
      void *ptr = repl_heap.alloc_obj(obj.bytes, obj.alignment);
      size_t allocated_size = MockReplicatedHeap::calculate_actual_size(obj.bytes);
      allocated_buffers.push_back(std::make_pair(ptr, allocated_size));
      total_allocated_size += allocated_size;
    }
    assert(allocated_buffers.size() >= param.action_objs.size());
  }

  FreeObjParam param;
  size_t total_allocated_size = 0;
  std::vector<std::pair<void *, size_t>> allocated_buffers;
};

TEST_P(ReplicatedHeapFreeObjTest, FreeObj)
{
  size_t deallocated_sizes = 0;
  for(const int &idx : param.action_objs) {
    repl_heap.free_obj(allocated_buffers[idx].first);
    deallocated_sizes += allocated_buffers[idx].second;
  }
  EXPECT_EQ(repl_heap.size_available(),
            chunk_size - total_allocated_size + deallocated_sizes);
}

INSTANTIATE_TEST_SUITE_P(
    ReplicatedHeapFreeObj, ReplicatedHeapFreeObjTest,
    ::testing::Values(
        FreeObjParam{std::vector<ObjParam>{{1024, sizeof(int)}},
                     std::vector<int>{0}}, // 1 alloc, 1 free
        FreeObjParam{std::vector<ObjParam>{
                         {512, sizeof(int)}, {1024, sizeof(int)}, {2048, sizeof(int)}},
                     std::vector<int>{0}}, // 3 alloc, remove the 1st
        FreeObjParam{std::vector<ObjParam>{
                         {512, sizeof(int)}, {1024, sizeof(int)}, {2048, sizeof(int)}},
                     std::vector<int>{1}}, // 3 alloc, remove the 2nd
        FreeObjParam{std::vector<ObjParam>{
                         {512, sizeof(int)}, {1024, sizeof(int)}, {2048, sizeof(int)}},
                     std::vector<int>{2}}, // 3 alloc, remove the 3rd
        FreeObjParam{std::vector<ObjParam>{
                         {512, sizeof(int)}, {1024, sizeof(int)}, {2048, sizeof(int)}},
                     std::vector<int>{1, 2}} // 3 alloc, remove 2
        ));