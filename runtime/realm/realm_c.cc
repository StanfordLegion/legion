/* Copyright 2025 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm/realm_c.h"

#include "realm/runtime_impl.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/mutex.h"
#include "realm/utils.h"
#include <cassert>

namespace Realm {

  extern RuntimeImpl *runtime_singleton;

  Mutex runtime_singleton_mutex;

  bool enable_unit_tests = false;

  class ProcessorQueryImplWrapper {
  public:
    explicit ProcessorQueryImplWrapper(const MachineImpl *machine_impl)
    {
      impl = new ProcessorQueryImpl(machine_impl);
    }
    ~ProcessorQueryImplWrapper(void) { impl->remove_reference(); }
    operator ProcessorQueryImpl *&() { return impl; }

  protected:
    ProcessorQueryImpl *impl;
  };

  class MemoryQueryImplWrapper {
  public:
    explicit MemoryQueryImplWrapper(const MachineImpl *machine_impl)
    {
      impl = new MemoryQueryImpl(machine_impl);
    }
    ~MemoryQueryImplWrapper(void) { impl->remove_reference(); }
    operator MemoryQueryImpl *&() { return impl; }

  protected:
    MemoryQueryImpl *impl;
  };

}; // namespace Realm

Realm::Logger log_realm_c("realmc");

static inline realm_status_t check_runtime_validity_and_assign(realm_runtime_t runtime,
                                                               Realm::RuntimeImpl *&impl)
{
  if(runtime == nullptr) {
    return REALM_RUNTIME_ERROR_NOT_INITIALIZED;
  }
  if(!Realm::enable_unit_tests &&
     runtime != reinterpret_cast<realm_runtime_t>(Realm::runtime_singleton)) {
    return REALM_RUNTIME_ERROR_INVALID_RUNTIME;
  }
  impl = reinterpret_cast<Realm::RuntimeImpl *>(runtime);
  return REALM_SUCCESS;
}

static inline realm_status_t check_processor_validity(realm_processor_t proc)
{
  if(proc == REALM_NO_PROC) {
    return REALM_PROCESSOR_ERROR_INVALID_PROCESSOR;
  }
  return Realm::ID(proc).is_processor() ? REALM_SUCCESS
                                        : REALM_PROCESSOR_ERROR_INVALID_PROCESSOR;
}

static inline realm_status_t check_processor_kind_validity(realm_processor_kind_t kind)
{
  if(kind >= TOC_PROC && kind <= PY_PROC) {
    return REALM_SUCCESS;
  }
  return REALM_PROCESSOR_ERROR_INVALID_PROCESSOR_KIND;
}

static inline realm_status_t check_event_validity(realm_event_t event)
{
  if(event == REALM_NO_EVENT) {
    return REALM_SUCCESS;
  }
  return Realm::ID(event).is_event() ? REALM_SUCCESS : REALM_EVENT_ERROR_INVALID_EVENT;
}

static inline realm_status_t check_memory_validity(realm_memory_t mem)
{
  if(mem == REALM_NO_MEM) {
    return REALM_MEMORY_ERROR_INVALID_MEMORY;
  }
  return Realm::ID(mem).is_memory() ? REALM_SUCCESS : REALM_MEMORY_ERROR_INVALID_MEMORY;
}

static inline realm_status_t check_memory_kind_validity(realm_memory_kind_t kind)
{
  if(kind >= GLOBAL_MEM && kind <= GPU_DYNAMIC_MEM) {
    return REALM_SUCCESS;
  }
  return REALM_MEMORY_ERROR_INVALID_MEMORY_KIND;
}

/* Runtime API */

realm_status_t realm_runtime_create(realm_runtime_t *runtime)
{
  {
    Realm::AutoLock<> lock(Realm::runtime_singleton_mutex);
    if(Realm::runtime_singleton == nullptr) {
      Realm::runtime_singleton = new Realm::RuntimeImpl;
    }
  }
  *runtime = reinterpret_cast<realm_runtime_t>(Realm::runtime_singleton);
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_destroy(realm_runtime_t runtime)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  Realm::AutoLock<> lock(Realm::runtime_singleton_mutex);
  delete runtime_impl;
  Realm::runtime_singleton = nullptr;
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_get_runtime(realm_runtime_t *runtime)
{
  *runtime = reinterpret_cast<realm_runtime_t>(Realm::runtime_singleton);
  return (*runtime == nullptr) ? REALM_RUNTIME_ERROR_NOT_INITIALIZED : REALM_SUCCESS;
}

realm_status_t realm_runtime_init(realm_runtime_t runtime, int *argc, char ***argv)
{
  REALM_ENTRY_EXIT(log_realm_c);
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }

  // if we get null pointers for argc and argv, use a local version so
  //  any changes from network_init are seen in configure_from_command_line
  int my_argc = 0;
  char **my_argv = nullptr;
  if(argc == nullptr) {
    argc = &my_argc;
  }
  if(argv == nullptr) {
    argv = &my_argv;
  }

  // TODO: we need to let each of these functions to return a specific error code
  if(!runtime_impl->network_init(argc, argv)) {
    return REALM_ERROR;
  }
  if(!runtime_impl->create_configs(*argc, *argv)) {
    return REALM_ERROR;
  }

  // TODO: do not create a vector here, just pass the array
  std::vector<std::string> cmdline;
  cmdline.reserve(*argc);
  for(int i = 1; i < *argc; i++) {
    cmdline.push_back((*argv)[i]);
  }
  if(!runtime_impl->configure_from_command_line(cmdline)) {
    return REALM_ERROR;
  }
  runtime_impl->start();
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_signal_shutdown(realm_runtime_t runtime,
                                             realm_event_t wait_on, int result_code)
{

  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  runtime_impl->shutdown(Realm::Event(wait_on), result_code);
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_wait_for_shutdown(realm_runtime_t runtime)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  runtime_impl->wait_for_shutdown();
  return REALM_SUCCESS;
}

realm_status_t realm_runtime_collective_spawn(realm_runtime_t runtime,
                                              realm_processor_t target_proc,
                                              realm_task_func_id_t task_id,
                                              const void *args, size_t arglen,
                                              realm_event_t wait_on, int priority,
                                              realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(target_proc);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  // TODO: check the validation of the task id if target_proc is local, if it is not
  // local, we will poison the event.
  *event = runtime_impl->collective_spawn(Realm::Processor(target_proc), task_id, args,
                                          arglen, Realm::Event(wait_on), priority);
  return REALM_SUCCESS;
}

/* Processor API */

realm_status_t realm_processor_register_task_by_kind(
    realm_runtime_t runtime, realm_processor_kind_t target_kind,
    realm_register_task_flags_t flags, realm_task_func_id_t task_id,
    realm_task_pointer_t func, void *user_data, size_t user_data_len,
    realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_kind_validity(target_kind);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(func == nullptr) {
    return REALM_PROCESSOR_ERROR_INVALID_TASK_FUNCTION;
  }
  bool global = (flags & REALM_REGISTER_TASK_GLOBAL) != 0;
  Realm::CodeDescriptor code_desc(
      Realm::Type::from_cpp_type<Realm::Processor::TaskFuncPtr>());
  code_desc.add_implementation(
      new Realm::FunctionPointerImplementation(reinterpret_cast<void (*)()>(func)));
  *event = Realm::Processor::register_task_by_kind(
      static_cast<Realm::Processor::Kind>(target_kind), global, task_id, code_desc,
      Realm::ProfilingRequestSet(), user_data, user_data_len);
  return REALM_SUCCESS;
}

realm_status_t realm_processor_spawn(realm_runtime_t runtime,
                                     realm_processor_t target_proc,
                                     realm_task_func_id_t task_id, const void *args,
                                     size_t arglen, realm_profiling_request_set_t prs,
                                     realm_event_t wait_on, int priority,
                                     realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(target_proc);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(wait_on);
  if(status != REALM_SUCCESS) {
    return status;
  }
  // TODO: check the validation of the task id for local processor
  Realm::ProcessorImpl *proc_impl =
      runtime_impl->get_processor_impl(Realm::Processor(target_proc));

  Realm::GenEventImpl *finish_event = Realm::GenEventImpl::create_genevent(runtime_impl);
  Realm::Event cxx_event = finish_event->current_event();
  if(prs == nullptr) {
    proc_impl->spawn_task(task_id, args, arglen, Realm::ProfilingRequestSet(),
                          Realm::Event(wait_on), finish_event,
                          Realm::ID(cxx_event).event_generation(), priority);
  } else {
    proc_impl->spawn_task(task_id, args, arglen,
                          *reinterpret_cast<Realm::ProfilingRequestSet *>(prs),
                          Realm::Event(wait_on), finish_event,
                          Realm::ID(cxx_event).event_generation(), priority);
  }
  *event = cxx_event;
  return REALM_SUCCESS;
}

realm_status_t realm_processor_get_attributes(realm_runtime_t runtime,
                                              realm_processor_t proc,
                                              realm_processor_attr_t *attrs,
                                              uint64_t *values, size_t num)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_processor_validity(proc);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(num == 0) {
    return REALM_SUCCESS;
  }
  if((attrs == nullptr) || (values == nullptr)) {
    return REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE;
  }

  Realm::ProcessorImpl *proc_impl =
      runtime_impl->get_processor_impl(Realm::Processor(proc));
  if(proc_impl == nullptr) {
    return REALM_PROCESSOR_ERROR_INVALID_PROCESSOR;
  }
  for(size_t attr_idx = 0; attr_idx < num; attr_idx++) {
    switch(attrs[attr_idx]) {
    case REALM_PROCESSOR_ATTR_KIND:
      values[attr_idx] = static_cast<uint64_t>(proc_impl->kind);
      break;
    case REALM_PROCESSOR_ATTR_ADDRESS_SPACE:
      values[attr_idx] = Realm::ID(proc).proc_owner_node();
      break;
    default:
      return REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE;
    }
  }
  return REALM_SUCCESS;
}

/* ProcessorQuery API */

realm_status_t realm_processor_query_create(realm_runtime_t runtime,
                                            realm_processor_query_t *query)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  *query = reinterpret_cast<realm_processor_query_t>(
      new Realm::ProcessorQueryImplWrapper(runtime_impl->machine));
  return REALM_SUCCESS;
}

realm_status_t realm_processor_query_destroy(realm_processor_query_t query)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::ProcessorQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query);
  delete query_impl_wrapper;
  return REALM_SUCCESS;
}

realm_status_t realm_processor_query_restrict_to_kind(realm_processor_query_t query,
                                                      realm_processor_kind_t kind)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::ProcessorQueryImpl *query_impl =
      *(reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query));
  realm_status_t status = check_processor_kind_validity(kind);
  if(status != REALM_SUCCESS) {
    return status;
  }
  query_impl = query_impl->writeable_reference();
  query_impl->restrict_to_kind(static_cast<Realm::Processor::Kind>(kind));
  return REALM_SUCCESS;
}

realm_status_t
realm_processor_query_restrict_to_address_space(realm_processor_query_t query,
                                                realm_address_space_t address_space)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::ProcessorQueryImpl *query_impl =
      *(reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query));
  query_impl = query_impl->writeable_reference();
  query_impl->restrict_to_node(address_space);
  return REALM_SUCCESS;
}

static Realm::Processor realm_processor_query_next(Realm::ProcessorQueryImpl *query_impl,
                                                   Realm::Processor after)
{
  Realm::Processor proc;
  if(Realm::Config::use_machine_query_cache) {
    proc = query_impl->cache_next(after);
  } else {
    proc = query_impl->next_match(after);
  }
  return proc;
}

realm_status_t realm_processor_query_iter(realm_processor_query_t query,
                                          realm_processor_query_cb_t cb, void *user_data,
                                          size_t max_queries)
{
  if(query == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY;
  }
  if(cb == nullptr) {
    return REALM_PROCESSOR_QUERY_ERROR_INVALID_CALLBACK;
  }
  Realm::ProcessorQueryImpl *query_impl =
      *(reinterpret_cast<Realm::ProcessorQueryImplWrapper *>(query));
  size_t num_queries = 0;
  Realm::Processor proc = query_impl->first_match();
  while(num_queries < max_queries && proc != Realm::Processor::NO_PROC) {
    realm_status_t status = cb(proc, user_data);
    if(status != REALM_SUCCESS) {
      return status;
    }
    proc = realm_processor_query_next(query_impl, proc);
    num_queries++;
  }
  return REALM_SUCCESS;
}

/* Memory API */

realm_status_t realm_memory_get_attributes(realm_runtime_t runtime, realm_memory_t mem,
                                           realm_memory_attr_t *attrs, uint64_t *values,
                                           size_t num)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_memory_validity(mem);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(num == 0) {
    return REALM_SUCCESS;
  }
  if(attrs == nullptr || values == nullptr) {
    return REALM_MEMORY_ERROR_INVALID_ATTRIBUTE;
  }

  Realm::MemoryImpl *mem_impl = runtime_impl->get_memory_impl(Realm::Memory(mem));
  if(mem_impl == nullptr) {
    return REALM_MEMORY_ERROR_INVALID_MEMORY;
  }
  for(size_t attr_idx = 0; attr_idx < num; attr_idx++) {
    switch(attrs[attr_idx]) {
    case REALM_MEMORY_ATTR_KIND:
      values[attr_idx] = static_cast<uint64_t>(mem_impl->get_kind());
      break;
    case REALM_MEMORY_ATTR_ADDRESS_SPACE:
      values[attr_idx] = Realm::ID(mem).memory_owner_node();
      break;
    case REALM_MEMORY_ATTR_CAPACITY:
      values[attr_idx] = mem_impl->size;
      break;
    default:
      return REALM_MEMORY_ERROR_INVALID_ATTRIBUTE;
    }
  }
  return REALM_SUCCESS;
}

/* MemoryQuery API */

realm_status_t realm_memory_query_create(realm_runtime_t runtime,
                                         realm_memory_query_t *query)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  *query = reinterpret_cast<realm_memory_query_t>(
      new Realm::MemoryQueryImplWrapper(runtime_impl->machine));
  return REALM_SUCCESS;
}

realm_status_t realm_memory_query_destroy(realm_memory_query_t query)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::MemoryQueryImplWrapper *query_impl_wrapper =
      reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query);
  delete query_impl_wrapper;
  return REALM_SUCCESS;
}

realm_status_t realm_memory_query_restrict_to_kind(realm_memory_query_t query,
                                                   realm_memory_kind_t kind)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::MemoryQueryImpl *query_impl =
      *(reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query));
  realm_status_t status = check_memory_kind_validity(kind);
  if(status != REALM_SUCCESS) {
    return status;
  }
  query_impl = query_impl->writeable_reference();
  query_impl->restrict_to_kind(static_cast<Realm::Memory::Kind>(kind));
  return REALM_SUCCESS;
}

realm_status_t
realm_memory_query_restrict_to_address_space(realm_memory_query_t query,
                                             realm_address_space_t address_space)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::MemoryQueryImpl *query_impl =
      *(reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query));
  query_impl = query_impl->writeable_reference();
  query_impl->restrict_to_node(address_space);
  return REALM_SUCCESS;
}

realm_status_t realm_memory_query_restrict_by_capacity(realm_memory_query_t query,
                                                       size_t min_bytes)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  Realm::MemoryQueryImpl *query_impl =
      *(reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query));
  query_impl = query_impl->writeable_reference();
  query_impl->restrict_by_capacity(min_bytes);
  return REALM_SUCCESS;
}

static Realm::Memory realm_memory_query_next(Realm::MemoryQueryImpl *query_impl,
                                             Realm::Memory after)
{
  Realm::Memory m;
  if(Realm::Config::use_machine_query_cache) {
    m = query_impl->cache_next(after);
  } else {
    m = query_impl->next_match(after);
  }
  return m;
}

realm_status_t realm_memory_query_iter(realm_memory_query_t query,
                                       realm_memory_query_cb_t cb, void *user_data,
                                       size_t max_queries)
{
  if(query == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_QUERY;
  }
  if(cb == nullptr) {
    return REALM_MEMORY_QUERY_ERROR_INVALID_CALLBACK;
  }
  Realm::MemoryQueryImpl *query_impl =
      *(reinterpret_cast<Realm::MemoryQueryImplWrapper *>(query));
  size_t num_queries = 0;
  Realm::Memory mem = query_impl->first_match();
  while(num_queries < max_queries && mem != Realm::Memory::NO_MEMORY) {
    realm_status_t status = cb(mem, user_data);
    if(status != REALM_SUCCESS) {
      return status;
    }
    mem = realm_memory_query_next(query_impl, mem);
    num_queries++;
  }
  return REALM_SUCCESS;
}

/* Event API */

realm_status_t realm_event_wait(realm_runtime_t runtime, realm_event_t event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(event);
  if(status != REALM_SUCCESS) {
    return status;
  }
  Realm::Event cxx_event = Realm::Event(event);
  cxx_event.wait();
  return REALM_SUCCESS;
}

realm_status_t realm_event_merge(realm_runtime_t runtime, const realm_event_t *wait_for,
                                 size_t num_events, realm_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(wait_for == nullptr || event == nullptr) {
    return REALM_EVENT_ERROR_INVALID_EVENT;
  }
  Realm::Event *event_array =
      const_cast<Realm::Event *>(reinterpret_cast<const Realm::Event *>(wait_for));
  *event = Realm::Event::merge_events(
      Realm::span<const Realm::Event>(event_array, num_events));
  return REALM_SUCCESS;
}

/* UserEvent API */

realm_status_t realm_user_event_create(realm_runtime_t runtime, realm_user_event_t *event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  if(event == nullptr) {
    return REALM_EVENT_ERROR_INVALID_EVENT;
  }

  Realm::Event cxx_event =
      Realm::GenEventImpl::create_genevent(runtime_impl)->current_event();
  assert(cxx_event.id != 0);
  *event = cxx_event;

  return REALM_SUCCESS;
}

realm_status_t realm_user_event_trigger(realm_runtime_t runtime, realm_user_event_t event)
{
  Realm::RuntimeImpl *runtime_impl = nullptr;
  realm_status_t status = check_runtime_validity_and_assign(runtime, runtime_impl);
  if(status != REALM_SUCCESS) {
    return status;
  }
  status = check_event_validity(event);
  if(status != REALM_SUCCESS) {
    return status;
  }
  Realm::UserEvent(event).trigger();
  return REALM_SUCCESS;
}
