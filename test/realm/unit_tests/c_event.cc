#include "realm/realm_c.h"
#include "test_mock.h"
#include "test_common.h"
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <map>
#include <set>
#include <gtest/gtest.h>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
};

// test event without parameters

class CEventTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl = std::make_unique<MockRuntimeImplWithEventFreeList>();
    runtime_impl->init();
  }

  void TearDown() override { runtime_impl->finalize(); }

  std::unique_ptr<MockRuntimeImplWithEventFreeList> runtime_impl{nullptr};
};

TEST_F(CEventTest, CreateUserEventNullRuntime)
{
  realm_user_event_t event;
  realm_status_t status = realm_user_event_create(nullptr, &event);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CEventTest, CreateUserEventNullEvent)
{
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_user_event_create(runtime, nullptr);
  EXPECT_EQ(status, REALM_EVENT_ERROR_INVALID_EVENT);
}

// TODO(wei): Fix this once get_runtime() is removed from GenEventImpl::GenEventImpl
TEST_F(CEventTest, DISABLED_CreateUserEventSuccess)
{
  realm_user_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_user_event_create(runtime, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_TRUE(ID(event).is_event());
}

TEST_F(CEventTest, MergeEventsNullRuntime)
{
  realm_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_merge(nullptr, nullptr, 0, &event);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

TEST_F(CEventTest, MergeEventsNullWaitFor)
{
  realm_event_t event = REALM_NO_EVENT;
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_event_merge(runtime, nullptr, 1, &event);
  EXPECT_EQ(status, REALM_EVENT_ERROR_INVALID_EVENT);
}

TEST_F(CEventTest, MergeEventsNullEvent)
{
  realm_user_event_t wait_for_events[2] = {REALM_NO_EVENT, REALM_NO_EVENT};
  realm_runtime_t runtime = *runtime_impl;
  realm_status_t status = realm_event_merge(runtime, wait_for_events, 1, nullptr);
  EXPECT_EQ(status, REALM_EVENT_ERROR_INVALID_EVENT);
}

TEST_F(CEventTest, MergeEventsZeroWaitFor)
{
  realm_user_event_t wait_for_events[1] = {REALM_NO_EVENT};
  realm_runtime_t runtime = *runtime_impl;
  realm_user_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_merge(runtime, wait_for_events, 0, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, REALM_NO_EVENT);
}

TEST_F(CEventTest, MergeEventsNoEvents)
{
  realm_user_event_t wait_for_events[2] = {REALM_NO_EVENT, REALM_NO_EVENT};
  realm_runtime_t runtime = *runtime_impl;
  realm_user_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_merge(runtime, wait_for_events, 2, &event);
  EXPECT_EQ(status, REALM_SUCCESS);
  EXPECT_EQ(event, REALM_NO_EVENT);
}

// TODO: remove the get_runtime in the has_triggered function to unblock the test, merge
// calls it. TEST_F(CEventTest, MergeEventsSuccess)
// {
//   int num_events = 2;
//   realm_user_event_t wait_for_events[num_events];
//   realm_runtime_t runtime = *runtime_impl;
//   for (int i = 0; i < num_events; i++) {
//     realm_status_t status = realm_user_event_create(runtime, &wait_for_events[i]);
//     ASSERT_REALM(status);
//   }
//   realm_user_event_t event = REALM_NO_EVENT;
//   realm_status_t status = realm_event_merge(runtime, wait_for_events, num_events,
//   &event); EXPECT_EQ(status, REALM_SUCCESS); EXPECT_TRUE(ID(event).is_event());
// }

TEST_F(CEventTest, EventWaitNullRuntime)
{
  realm_event_t event = REALM_NO_EVENT;
  realm_status_t status = realm_event_wait(nullptr, event);
  EXPECT_EQ(status, REALM_RUNTIME_ERROR_NOT_INITIALIZED);
}

// TODO: finish this test once we remove the get_runtime and threading
// TEST_F(CEventTest, EventWaitTriggeredEvent)
// {
//   realm_user_event_t event = REALM_NO_EVENT;
//   realm_runtime_t runtime = *runtime_impl;
//   ASSERT_REALM(realm_user_event_create(runtime, &event));
//   ASSERT_REALM(realm_user_event_trigger(runtime, event));

//   realm_status_t status = realm_event_wait(runtime, event);
//   EXPECT_EQ(status, REALM_SUCCESS);
// }

// TEST_F(CEventTest, EventWaitNotTriggeredEvent)
// {
//   realm_user_event_t event = REALM_NO_EVENT;
//   realm_runtime_t runtime = *runtime_impl;
//   ASSERT_REALM(realm_user_event_create(runtime, &event));

//   realm_status_t status = realm_event_wait(runtime, event);
//   EXPECT_EQ(status, REALM_SUCCESS);
// }

// // an event id with maybe a higher generation than is triggered should return an error
// TEST_F(CEventTest, EventWaitInvalidEvent)
// {
// }

// TODO: remove the get_runtime in the trigger function
// TEST_F(CEventTest, UserEventTriggerSuccess)
// {
//   realm_user_event_t event = REALM_NO_EVENT;
//   realm_runtime_t runtime = *runtime_impl;
//   realm_status_t status = realm_user_event_create(runtime, &event);
//   ASSERT_REALM(status);

//   status = realm_user_event_trigger(runtime, event);
//   EXPECT_EQ(status, REALM_SUCCESS);
//   EventImpl *e = runtime_impl->get_event_impl(Event(event));
//   bool poisoned = false;
//   EXPECT_TRUE(e->has_triggered(ID(event).event_generation(), poisoned));
// }