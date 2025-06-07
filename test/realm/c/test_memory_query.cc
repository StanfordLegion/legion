#include "common.h"

#include "realm.h"

#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>

Realm::Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
};

struct append_memory_args_t {
  std::vector<realm_memory_t> mems;
};

static realm_status_t REALM_FNPTR append_memory(realm_memory_t m, void *user_data)
{
  append_memory_args_t *args = reinterpret_cast<append_memory_args_t *>(user_data);
  args->mems.push_back(m);
  return REALM_SUCCESS;
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc)
{
  printf("top_level_task on proc " IDFMT, proc);
  realm_runtime_t runtime;
  realm_status_t status;
  status = realm_runtime_get_runtime(&runtime);
  assert(status == REALM_SUCCESS);

  // Iterate over all CPU memories, and print their attributes
  realm_memory_query_t cpu_mem_query;
  realm_memory_query_create(runtime, &cpu_mem_query);
  // restrict to SYSTEM_MEM
  status = realm_memory_query_restrict_to_kind(cpu_mem_query, SYSTEM_MEM);
  assert(status == REALM_SUCCESS);
  append_memory_args_t cpu_mem_query_args;
  // query all system memories
  status = realm_memory_query_iter(cpu_mem_query, append_memory, &cpu_mem_query_args,
                                   SIZE_MAX);
  assert(status == REALM_SUCCESS);
  // print the attributes of the memories
  for(realm_memory_t mem : cpu_mem_query_args.mems) {
    realm_memory_attr_t attrs[3] = {REALM_MEMORY_ATTR_KIND,
                                    REALM_MEMORY_ATTR_ADDRESS_SPACE,
                                    REALM_MEMORY_ATTR_CAPACITY};
    uint64_t values[3];
    status = realm_memory_get_attributes(runtime, mem, attrs, values, 3);
    assert(status == REALM_SUCCESS);
    log_app.info("Memory " IDFMT " kind: %" PRIu64 ", address space: %" PRIu64
                 ", size: %" PRIu64,
                 mem, values[0], values[1], values[2]);
  }
  status = realm_memory_query_destroy(cpu_mem_query);
  assert(status == REALM_SUCCESS);

  // Iterate over all GPU FB memories, and print their attributes
  realm_memory_query_t gpu_mem_query;
  status = realm_memory_query_create(runtime, &gpu_mem_query);
  assert(status == REALM_SUCCESS);
  status = realm_memory_query_restrict_to_kind(gpu_mem_query, GPU_FB_MEM);
  assert(status == REALM_SUCCESS);
  // query all GPU FB memories
  append_memory_args_t gpu_mem_query_args;
  status = realm_memory_query_iter(gpu_mem_query, append_memory, &gpu_mem_query_args,
                                   SIZE_MAX);
  assert(status == REALM_SUCCESS);
  // print the attributes of the memories
  for(realm_memory_t mem : gpu_mem_query_args.mems) {
    realm_memory_attr_t attrs[3] = {REALM_MEMORY_ATTR_KIND,
                                    REALM_MEMORY_ATTR_ADDRESS_SPACE,
                                    REALM_MEMORY_ATTR_CAPACITY};
    uint64_t values[3];
    status = realm_memory_get_attributes(runtime, mem, attrs, values, 3);
    assert(status == REALM_SUCCESS);
    log_app.info("Memory " IDFMT " kind: %" PRIu64 ", address space: %" PRIu64
                 ", size: %" PRIu64,
                 mem, values[0], values[1], values[2]);
  }
  status = realm_memory_query_destroy(gpu_mem_query);
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