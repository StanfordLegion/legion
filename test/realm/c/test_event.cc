#include "common.h"
#include "realm/realm_c.h"
#include <stdio.h>

enum
{
  TOP_LEVEL_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
  EVENT_TASK,
};

struct event_task_args_t {
  realm_user_event_t user_event;
};

void REALM_FNPTR event_task(const void *args, size_t arglen, const void *userdata,
                            size_t userlen, realm_processor_t proc)
{
  printf("event_task on proc %llx\n", proc);
  event_task_args_t *task_args = (event_task_args_t *)args;
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_get_runtime(&runtime);
  assert(status == REALM_SUCCESS);
  status = realm_user_event_trigger(runtime, task_args->user_event);
  assert(status == REALM_SUCCESS);
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc)
{
  printf("top_level_task on proc %llx\n", proc);
  realm_user_event_t user_events[10];
  realm_event_t task_events[10];
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_get_runtime(&runtime);
  assert(status == REALM_SUCCESS);
  for(int i = 0; i < 10; i++) {
    status = realm_user_event_create(runtime, &user_events[i]);
    assert(status == REALM_SUCCESS);
    event_task_args_t args;
    args.user_event = user_events[i];
    status = realm_processor_spawn(runtime, proc, EVENT_TASK, &args, sizeof(args), NULL,
                                   0, 0, &task_events[i]);
    assert(status == REALM_SUCCESS);
  }

  realm_event_t merged_event;
  status = realm_event_merge(runtime, task_events, 10, &merged_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, merged_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_merge(runtime, user_events, 10, &merged_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, merged_event);
  assert(status == REALM_SUCCESS);
}

int main(int argc, char **argv)
{
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_create(&runtime);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_init(runtime, &argc, &argv);
  assert(status == REALM_SUCCESS);

  realm_event_t register_task_event;

  status = realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, TOP_LEVEL_TASK, top_level_task, 0,
      0, &register_task_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, register_task_event);
  assert(status == REALM_SUCCESS);

  status = realm_processor_register_task_by_kind(runtime, LOC_PROC,
                                                 REALM_REGISTER_TASK_DEFAULT, EVENT_TASK,
                                                 event_task, 0, 0, &register_task_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, register_task_event);
  assert(status == REALM_SUCCESS);

  realm_processor_query_t proc_query;
  status = realm_processor_query_create(runtime, &proc_query);
  assert(status == REALM_SUCCESS);
  status = realm_processor_query_restrict_to_kind(proc_query, LOC_PROC);
  assert(status == REALM_SUCCESS);
  realm_processor_t proc;
  realm_processor_query_first(proc_query, &proc);
  status = realm_processor_query_destroy(proc_query);
  assert(status == REALM_SUCCESS);
  assert(proc != REALM_NO_PROC);

  realm_event_t e;
  status = realm_runtime_collective_spawn(runtime, proc, TOP_LEVEL_TASK, 0, 0, 0, 0, &e);
  assert(status == REALM_SUCCESS);

  status = realm_runtime_signal_shutdown(runtime, e, 0);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_wait_for_shutdown(runtime);
  assert(status == REALM_SUCCESS);
  status = realm_runtime_destroy(runtime);
  assert(status == REALM_SUCCESS);

  return 0;
}