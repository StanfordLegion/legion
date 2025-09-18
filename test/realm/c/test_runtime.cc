#include "common.h"

#include "realm.h"

#include <stdio.h>
#include <unistd.h>

Realm::Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
  HELLO_TASK,
};

struct append_process_args_t {
  std::vector<realm_processor_t> procs;
};

static realm_status_t REALM_FNPTR append_process(realm_processor_t p, void *user_data)
{
  append_process_args_t *args = reinterpret_cast<append_process_args_t *>(user_data);
  args->procs.push_back(p);
  return REALM_SUCCESS;
}

void REALM_FNPTR hello_task(const void *args, size_t arglen, const void *userdata,
                            size_t userlen, realm_processor_t proc)
{
  log_app.info("hello_task on proc " IDFMT, proc);
  sleep(1);
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc)
{
  log_app.info("top_level_task on proc " IDFMT, proc);
  realm_event_t event;
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_get_runtime(&runtime);
  assert(status == REALM_SUCCESS);

  // get all cpu procs
  realm_processor_query_t proc_query;
  status = realm_processor_query_create(runtime, &proc_query);
  assert(status == REALM_SUCCESS);
  // restrict to LOC_PROC
  status = realm_processor_query_restrict_to_kind(proc_query, LOC_PROC);
  assert(status == REALM_SUCCESS);
  // query all LOC_PROC processors
  append_process_args_t cpu_proc_query_args;
  status = realm_processor_query_iter(proc_query, append_process, &cpu_proc_query_args,
                                      SIZE_MAX);
  assert(status == REALM_SUCCESS);
  status = realm_processor_query_destroy(proc_query);
  assert(status == REALM_SUCCESS);
  // spawn tasks on all cpu procs
  size_t num_cpus = cpu_proc_query_args.procs.size();
  realm_event_t events[num_cpus];
  for(size_t i = 0; i < num_cpus; i++) {
    status = realm_processor_spawn(runtime, cpu_proc_query_args.procs[i], HELLO_TASK, 0,
                                   0, NULL, 0, 0, &events[i]);
    assert(status == REALM_SUCCESS);
  }
  status = realm_event_merge(runtime, events, num_cpus, &event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, event);
  assert(status == REALM_SUCCESS);

#ifdef REALM_USE_CUDA
  status = realm_processor_query_create(runtime, &proc_query);
  assert(status == REALM_SUCCESS);
  status = realm_processor_query_restrict_to_kind(proc_query, TOC_PROC);
  assert(status == REALM_SUCCESS);

  realm_processor_t gpu_proc;
  realm_processor_query_first(proc_query, &gpu_proc);
  status = realm_processor_query_destroy(proc_query);
  assert(status == REALM_SUCCESS);
  assert(gpu_proc != REALM_NO_PROC);

  status = realm_processor_spawn(runtime, gpu_proc, HELLO_TASK, 0, 0, NULL, 0, 0, &event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, event);
  assert(status == REALM_SUCCESS);
#endif
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
                                                 REALM_REGISTER_TASK_DEFAULT, HELLO_TASK,
                                                 hello_task, 0, 0, &register_task_event);
  assert(status == REALM_SUCCESS);
  status = realm_event_wait(runtime, register_task_event);
  assert(status == REALM_SUCCESS);

  status = realm_processor_register_task_by_kind(runtime, TOC_PROC,
                                                 REALM_REGISTER_TASK_DEFAULT, HELLO_TASK,
                                                 hello_task, 0, 0, &register_task_event);
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