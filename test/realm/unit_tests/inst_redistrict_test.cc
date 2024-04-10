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
  std::vector<RegionInstance> insts(2);
  Event e = inst.redistrict(nullptr, nullptr, 0, ProfilingRequestSet());
  bool poisoned = false;
  e.wait_faultaware(poisoned);
  EXPECT_TRUE(poisoned);
}

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
  Event e = inst.redistrict(insts.data(), layouts.data(), 2, ProfilingRequestSet());
  bool poisoned = false;
  e.wait_faultaware(poisoned);
  EXPECT_TRUE(poisoned);
  event.trigger();
}

TEST_F(InstanceRedistrictTest, RedistrictEvenlySameLayout)
{
  size_t num_elemnts = (1048576 / sizeof(int)) - 32;
  IndexSpace<1> space(Rect<1>(Point<1>(0), num_elemnts));
  RegionInstance inst;
  RegionInstance::create_instance(inst, memories[0], create_layout(space),
                                  ProfilingRequestSet())
      .wait();

  std::vector<int> data;
  {
    int index = 0;
    AffineAccessor<int, 1, int> acc(inst, 0);
    IndexSpaceIterator<1, int> it(space);
    while(it.valid) {
      PointInRectIterator<1, int> pit(it.rect);
      while(pit.valid) {
        data.push_back(index);
        acc[pit.p] = index++;
        pit.step();
      }
      it.step();
    }
  }

  IndexSpace<1> child_space_a(Rect<1>(Point<1>(0), num_elemnts / 2 - 1));
  IndexSpace<1> child_space_b(Rect<1>(Point<1>(0), num_elemnts / 2 - 1));

  std::vector<RegionInstance> insts(2);
  InstanceLayoutGeneric *ilg_a = create_layout(child_space_a);
  InstanceLayoutGeneric *ilg_b = create_layout(child_space_b);
  std::vector<InstanceLayoutGeneric *> layouts{ilg_a, ilg_b};
  Event e = inst.redistrict(insts.data(), layouts.data(), 2, ProfilingRequestSet());
  bool poisoned = false;
  e.wait_faultaware(poisoned);
  EXPECT_FALSE(poisoned);

  insts[0].destroy();

  int index = child_space_a.volume();
  for(size_t i = 1; i < insts.size(); i++) {
    RegionInstanceImpl *child_impl = get_runtime()->get_instance_impl(insts[i]);
    child_impl->request_metadata().wait();
    AffineAccessor<int, 1, int> acc(insts[i], 0);
    IndexSpaceIterator<1, int> it(child_space_a);
    while(it.valid) {
      PointInRectIterator<1, int> pit(it.rect);
      while(pit.valid) {
        int val = acc[pit.p];
        assert(val == data[index++]);
        pit.step();
      }
      it.step();
    }
  }
}
