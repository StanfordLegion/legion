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

  static Runtime *runtime_;
};

Runtime *InstanceRedistrictTest::runtime_ = nullptr;

TEST_F(InstanceRedistrictTest, Dummy)
{}
