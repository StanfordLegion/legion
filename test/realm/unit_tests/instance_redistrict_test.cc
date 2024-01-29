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
    runtime_->init(0, 0);
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

  std::vector<Memory> memories;
  static Runtime *runtime_;
};

Runtime *InstanceRedistrictTest::runtime_ = nullptr;

TEST_F(InstanceRedistrictTest, Dummy)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(7)));
  IndexSpace<1> space(rects);
  typedef std::map<FieldID, size_t> FieldMap;
  FieldMap fields;
  fields[0] = sizeof(int);
  InstanceLayoutConstraints ilc(fields, 1);
  int dim_order[1];
  dim_order[0] = 0;
  RegionInstance inst;

  {
    InstanceLayoutGeneric *ilg =
        InstanceLayoutGeneric::choose_instance_layout<1, int>(space, ilc, dim_order);
    RegionInstance::create_instance(inst, memories[0], ilg, ProfilingRequestSet()).wait();
  }

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

  IndexSpace<1> child_space_a(Rect<1>(Point<1>(0), Point<1>(3)));
  IndexSpace<1> child_space_b(Rect<1>(Point<1>(0), Point<1>(3)));

  RegionInstance child_inst_a;
  RegionInstance child_inst_b;
  std::vector<RegionInstance> insts{child_inst_a, child_inst_b};
  {
    InstanceLayoutGeneric *ilg_a =
        InstanceLayoutGeneric::choose_instance_layout<1, int>(child_space_a, ilc, dim_order);
    InstanceLayoutGeneric *ilg_b =
        InstanceLayoutGeneric::choose_instance_layout<1, int>(child_space_b, ilc, dim_order);

    std::vector<InstanceLayoutGeneric*> layouts{ilg_a, ilg_b};
    inst.redistrict(insts.data(), layouts.data(), 2, ProfilingRequestSet());
  }

  int index = 0;
  for(size_t i = 0; i < insts.size(); i++) {
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
