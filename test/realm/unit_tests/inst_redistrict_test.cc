#include "realm.h"
#include "realm/transfer/transfer.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class InstanceRedistrictTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    runtime_ = new Runtime();
    const char *argv[] = {"", "-ll:csize", "1"};
    char **cmds = const_cast<char **>(argv);
    int argc = 3;
    runtime_->init(&argc, &cmds);
  }

  static void TearDownTestSuite()
  {
    runtime_->shutdown();
    runtime_->wait_for_shutdown();
    // wait for shutdown
    delete runtime_;
  }

  void SetUp() override
  {
    Machine::MemoryQuery mq(Machine::get_machine());
    mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
    memories.assign(mq.begin(), mq.end());
    assert(!memories.empty());
  }

  InstanceLayoutGeneric *create_layout(IndexSpace<1> space)
  {
    std::map<FieldID, size_t> fields;
    fields[0] = sizeof(int);
    InstanceLayoutConstraints ilc(fields, 1);
    int dim_order[1];
    dim_order[0] = 0;
    InstanceLayoutGeneric *ilg =
        InstanceLayoutGeneric::choose_instance_layout<1, int>(space, ilc, dim_order);
    return ilg;
  }

  std::vector<Memory> memories;
  static Runtime *runtime_;
};

Runtime *InstanceRedistrictTest::runtime_ = nullptr;

TEST_F(InstanceRedistrictTest, EmptyLayouts)
{
  IndexSpace<1> space(Rect<1>(Point<1>(0), Point<1>(7)));
  RegionInstance inst;
  RegionInstance::create_instance(inst, memories[0], create_layout(space),
                                  ProfilingRequestSet())
      .wait();
  Event e = inst.redistrict(nullptr, nullptr, 0, nullptr);
  bool poisoned = false;
  e.wait_faultaware(poisoned);
  EXPECT_FALSE(poisoned);
}

#ifdef DEBUG_REALM
TEST_F(InstanceRedistrictTest, PendingRelease)
{
  IndexSpace<1> space(Rect<1>(Point<1>(0), Point<1>(7)));
  RegionInstance inst;
  RegionInstance::create_instance(inst, memories[0], create_layout(space),
                                  ProfilingRequestSet())
      .wait();
  UserEvent event = UserEvent::create_user_event();
  inst.destroy(event);
  std::vector<RegionInstance> insts(2);
  InstanceLayoutGeneric *ilg_a = create_layout(space);
  InstanceLayoutGeneric *ilg_b = create_layout(space);
  std::vector<InstanceLayoutGeneric *> layouts{ilg_a, ilg_b};
  std::vector<ProfilingRequestSet> prs(2);
  Event e = inst.redistrict(insts.data(), layouts.data(), 2, prs.data());
  bool poisoned = false;
  e.wait_faultaware(poisoned);
  EXPECT_TRUE(poisoned);
  event.trigger();
}
#endif
