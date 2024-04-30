#include "realm.h"
#include <tuple>
#include <vector>
#include <string>
#include <assert.h>
#include <gtest/gtest.h>

using namespace Realm;

class MemoryQueryTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    std::vector<const char *> cmdline_argv;
    const char dummy_args[] = "test"; // a dummy one represents the program itself
    const char csize_cmd[] = "-ll:csize";
    std::string csize = std::to_string(csize_);
    cmdline_argv.push_back(dummy_args);
    cmdline_argv.push_back(csize_cmd);
    cmdline_argv.push_back(csize.c_str());
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
    const char gpu_cmd[] = "-ll:gpu";
    const char gpu[] = "1";
    cmdline_argv.push_back(gpu_cmd);
    cmdline_argv.push_back(gpu);
    const char fbsize_cmd[] = "-ll:fsize";
    std::string fbsize = std::to_string(fbsize_);
    cmdline_argv.push_back(fbsize_cmd);
    cmdline_argv.push_back(fbsize.c_str());
    const char zcsize_cmd[] = "-ll:zsize";
    std::string zcsize = std::to_string(zcsize_);
    cmdline_argv.push_back(zcsize_cmd);
    cmdline_argv.push_back(zcsize.c_str());
    const char dyfbsize_cmd[] = "-cuda:dynfb_max";
    std::string dyfbsize = std::to_string(dyfbsize_);
    cmdline_argv.push_back(dyfbsize_cmd);
    cmdline_argv.push_back(dyfbsize.c_str());
#ifdef REALM_USE_CUDA
    const char msize_cmd[] = "-ll:msize";
    std::string msize = std::to_string(msize_);
    cmdline_argv.push_back(msize_cmd);
    cmdline_argv.push_back(msize.c_str());
#endif
#endif
    int argc = cmdline_argv.size();
    char **argv = const_cast<char **>(cmdline_argv.data());

    runtime_ = new Runtime();
    runtime_->init(&argc, &argv);
  }

  static void TearDownTestSuite()
  {
    runtime_->shutdown();
    runtime_->wait_for_shutdown();
    delete runtime_;
  }

  static Runtime *runtime_;
  static constexpr size_t csize_ = 32;
  static constexpr size_t fbsize_ = 8;
  static constexpr size_t zcsize_ = 16;
  static constexpr size_t msize_ = 32;
  static constexpr size_t dyfbsize_ = 64;
};

Runtime *MemoryQueryTest::runtime_ = nullptr;

TEST_F(MemoryQueryTest, MemoryQueryCapacity)
{
  Machine machine = Machine::get_machine();
  Machine::MemoryQuery mq(machine);

  size_t num_all_memories = mq.count();
  mq.has_capacity(1);
  size_t num_nonempty_memories = mq.count();
  // there is an external system memory whose size is 0, so
  // num_all_memories should be always greater than num_nonempty_memories
  EXPECT_GT(num_all_memories, num_nonempty_memories);
}

TEST_F(MemoryQueryTest, MemoryQueryIterator)
{
  Machine machine = Machine::get_machine();
  Machine::MemoryQuery mq(machine);

  size_t num_expected_memories = mq.count();
  size_t num_actual_memories = 0;
  for(Machine::MemoryQuery::iterator it = mq.begin(); it; ++it) {
    Memory m = *it;
    Memory::Kind kind = m.kind();
    switch(kind) {
    case Memory::SYSTEM_MEM:
    {
      EXPECT_TRUE(m.capacity() == 0 ||
                  m.capacity() == MemoryQueryTest::csize_ * 1024 * 1024);
      num_actual_memories++;
      break;
    }
    case Memory::Z_COPY_MEM:
    {
      EXPECT_EQ(m.capacity(), MemoryQueryTest::zcsize_ * 1024 * 1024);
      num_actual_memories++;
      break;
    }
    case Memory::GPU_FB_MEM:
    {
      EXPECT_EQ(m.capacity(), MemoryQueryTest::fbsize_ * 1024 * 1024);
      num_actual_memories++;
      break;
    }
    case Memory::GPU_MANAGED_MEM:
    {
      EXPECT_EQ(m.capacity(), MemoryQueryTest::msize_ * 1024 * 1024);
      num_actual_memories++;
      break;
    }
    case Memory::GPU_DYNAMIC_MEM:
    {
      EXPECT_EQ(m.capacity(), MemoryQueryTest::dyfbsize_ * 1024 * 1024);
      num_actual_memories++;
      break;
    }
    case Memory::FILE_MEM:
    {
      EXPECT_EQ(m.capacity(), 0);
      num_actual_memories++;
      break;
    }
    default:
    {
      assert(0);
      break;
    }
    }
  }

  EXPECT_EQ(num_expected_memories, num_actual_memories);
}
