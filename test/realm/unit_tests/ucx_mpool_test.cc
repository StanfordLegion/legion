#include <cstddef>
#include "realm/id.h"
#include "realm/mutex.h"

#include "realm/ucx/mpool.h"

#include <gtest/gtest.h>

using namespace Realm;
using namespace Realm::UCP;

class UCXMPoolTest : public ::testing::Test {
public:
  void SetUp() override {}

  void TearDown() override {}

  size_t obj_size = 16;
  size_t alignment = 16;
  size_t alignment_offset = 0;
};

TEST_F(UCXMPoolTest, GetLessObjects)
{
  const size_t obj_per_chunks = 4;
  const size_t init_number_objects = 4;
  const size_t max_objects = 4;
  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects);

  std::vector<void *> objects;
  for(size_t i = 0; i < init_number_objects - 1; i++) {
    objects.push_back(mpool.get());
  }

  for(size_t i = 0; i < init_number_objects - 1; i++) {
    EXPECT_NE(objects[i], nullptr);
  }

  EXPECT_TRUE(mpool.has(false));
}

TEST_F(UCXMPoolTest, GetMoreObjects)
{
  const size_t obj_per_chunks = 1;
  const size_t init_number_objects = 4;
  const size_t max_objects = 4;
  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects);

  std::vector<void *> objects;
  for(size_t i = 0; i < init_number_objects + 1; i++) {
    objects.push_back(mpool.get());
  }

  for(size_t i = 0; i < init_number_objects; i++) {
    EXPECT_NE(objects[i], nullptr);
  }

  EXPECT_EQ(objects[init_number_objects], nullptr);
  EXPECT_FALSE(mpool.has(false));
  EXPECT_FALSE(mpool.expand(1));
}

TEST_F(UCXMPoolTest, GetObjectsInitAndCheckContent)
{
  const size_t obj_per_chunks = 4;
  const size_t init_number_objects = 4;
  const size_t max_objects = 5;
  const size_t max_chunk_size = 16;

  auto obj_init_function = [](void *obj, void *arg) {
    int size = *static_cast<int *>(arg);
    int *object = static_cast<int *>(obj);
    for(int i = 0; i < size / sizeof(int); ++i) {
      object[i] = 7;
    }
  };

  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects, max_chunk_size, 1,
              &MPool::malloc_wrapper, nullptr, &MPool::free_wrapper, nullptr,
              obj_init_function, static_cast<void *>(&obj_size));

  std::vector<void *> objects;
  for(size_t i = 0; i < init_number_objects; i++) {
    objects.push_back(mpool.get());
  }

  for(size_t i = 0; i < init_number_objects; i++) {
    EXPECT_NE(objects[i], nullptr);
    int *object = static_cast<int *>(objects[i]);
    for(int i = 0; i < obj_size / sizeof(int); ++i) {
      EXPECT_EQ(object[i], 7);
    }
  }
  EXPECT_FALSE(mpool.has(false));
  EXPECT_TRUE(mpool.expand(1));
}

TEST_F(UCXMPoolTest, Expand)
{
  const size_t obj_per_chunks = 4;
  const size_t init_number_objects = 0;
  const size_t max_objects = 4;
  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects);

  EXPECT_TRUE(mpool.expand(max_objects));
}

TEST_F(UCXMPoolTest, DISABLED_PutObject)
{
  const size_t obj_per_chunks = 1;
  const size_t init_number_objects = 4;
  const size_t max_objects = 4;
  MPool mpool("test", /*leak_check_=*/false, obj_size, alignment, alignment_offset,
              obj_per_chunks, init_number_objects, max_objects);
}
