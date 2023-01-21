/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#include "bishop_c.h"
#include "bishop_mapper.h"
#include "legion.h"
#include "legion/legion_c_util.h"

#include <vector>
#include <set>
#include <map>
#include <cstdlib>

using namespace std;
using namespace Legion;
using namespace Legion::Mapping;
using namespace Legion::Mapping::Utilities;

static vector<bishop_mapper_impl_t> mapper_impls;
static vector<bishop_transition_fn_t> transitions;
static bishop_mapper_state_init_fn_t mapper_init;

namespace Legion {
  namespace Mapping {
    extern LegionRuntime::Logger::Category log_bishop;
  }
}

static void
bishop_mapper_registration_callback(Machine machine, Runtime *runtime,
                              			const set<Processor> &local_procs)
{
  MapperRuntime* rt = runtime->get_mapper_runtime();
  for (set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++)
  {
    BishopMapper* mapper =
      new BishopMapper(mapper_impls, transitions, mapper_init, rt, machine,
                       *it);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void
register_bishop_mappers(bishop_mapper_impl_t* _mapper_impls,
                        unsigned _num_mapper_impls,
                        bishop_transition_fn_t* _transitions,
                        unsigned _num_transitions,
                        bishop_mapper_state_init_fn_t _mapper_init)
{
  for (unsigned i = 0; i < _num_mapper_impls; ++i)
    mapper_impls.push_back(_mapper_impls[i]);
  for (unsigned i = 0; i < _num_transitions; ++i)
    transitions.push_back(_transitions[i]);
  mapper_init = _mapper_init;

  Runtime::add_registration_callback(
      bishop_mapper_registration_callback);
}

#define LIST_OP(NAME, TYPE, BASE)                        \
static TYPE                                              \
bishop_create_##NAME##_list(unsigned size)               \
{                                                        \
  TYPE l;                                                \
  l.size = size;                                         \
  if (size > 0)                                          \
    l.list = (BASE*)malloc(sizeof(BASE) * size);         \
  l.persistent = 0;                                      \
  return l;                                              \
}                                                        \
void                                                     \
bishop_delete_##NAME##_list(TYPE l)                      \
{                                                        \
  if (!l.persistent && l.size > 0) free(l.list);         \
}                                                        \

LIST_OP(processor, bishop_processor_list_t, legion_processor_t)
LIST_OP(memory, bishop_memory_list_t, legion_memory_t)
LIST_OP(field, bishop_field_list_t, legion_field_id_t)

bishop_processor_list_t
bishop_all_processors()
{
  Machine m = Machine::get_machine();
  set<Processor> procs;
  m.get_all_processors(procs);

  bishop_processor_list_t procs_ = bishop_create_processor_list(procs.size());
  int idx = 0;
  for (set<Processor>::iterator it = procs.begin(); it != procs.end(); ++it)
  {
    // FIXME: need to change this if we add more processors useful for
    // mapping purporses
    if (it->kind() != Processor::UTIL_PROC)
      procs_.list[idx++] = CObjectWrapper::wrap(*it);
  }
  procs_.size = idx;
  return procs_;
}

bishop_memory_list_t
bishop_all_memories()
{
  Machine m = Machine::get_machine();
  set<Memory> mems;
  m.get_all_memories(mems);

  bishop_memory_list_t mems_ = bishop_create_memory_list(mems.size());
  int idx = 0;
  for (set<Memory>::iterator it = mems.begin(); it != mems.end(); ++it)
    mems_.list[idx++] = CObjectWrapper::wrap(*it);
  return mems_;
}

legion_processor_t NO_PROC = CObjectWrapper::wrap(Processor::NO_PROC);

legion_processor_t
bishop_get_no_processor()
{
  return NO_PROC;
}

legion_memory_t
bishop_get_no_memory()
{
  return CObjectWrapper::wrap(Memory::NO_MEMORY);
}

bishop_processor_list_t
bishop_filter_processors_by_isa(bishop_processor_list_t source,
                                bishop_isa_t isa)
{
  vector<legion_processor_t> result;

  for (unsigned i = 0; i < source.size; ++i)
  {
    legion_processor_t proc_ = source.list[i];
    Processor proc = CObjectWrapper::unwrap(proc_);
    switch (isa)
    {
      case X86_ISA:
        {
          if (proc.kind() == Processor::LOC_PROC) result.push_back(proc_);
          break;
        }
      case CUDA_ISA:
        {
          if (proc.kind() == Processor::TOC_PROC) result.push_back(proc_);
          break;
        }
      default:
        {
          break;
        }
    }
  }

  bishop_processor_list_t result_ = bishop_create_processor_list(result.size());
  for (unsigned i = 0; i < result.size(); ++i)
    result_.list[i] = result[i];
  return result_;
}

bishop_processor_list_t
bishop_filter_processors_by_kind(bishop_processor_list_t source,
                                 legion_processor_kind_t kind_)
{
  vector<legion_processor_t> result;

  Processor::Kind kind = CObjectWrapper::unwrap(kind_);
  for (unsigned i = 0; i < source.size; ++i)
  {
    legion_processor_t proc_ = source.list[i];
    Processor proc = CObjectWrapper::unwrap(proc_);
    if (proc.kind() == kind) result.push_back(proc_);
  }

  bishop_processor_list_t result_ = bishop_create_processor_list(result.size());
  for (unsigned i = 0; i < result.size(); ++i)
    result_.list[i] = result[i];
  return result_;
}

bishop_memory_list_t
bishop_filter_memories_by_visibility(legion_processor_t proc_)
{
  Machine m = Machine::get_machine();
  vector<Memory> memories;
  Processor proc = CObjectWrapper::unwrap(proc_);
  MachineQueryInterface::find_memory_stack(m, proc, memories,
                                           proc.kind() == Processor::LOC_PROC);

  bishop_memory_list_t memories_ = bishop_create_memory_list(memories.size());
  int idx = 0;
  for (vector<Memory>::iterator it = memories.begin();
       it != memories.end(); ++it)
    memories_.list[idx++] = CObjectWrapper::wrap(*it);
  return memories_;
}

bishop_memory_list_t
bishop_filter_memories_by_kind(bishop_memory_list_t source,
                               legion_memory_kind_t kind_)
{
  Memory::Kind kind = CObjectWrapper::unwrap(kind_);

  vector<legion_memory_t> result;

  for (unsigned i = 0; i < source.size; ++i)
  {
    legion_memory_t memory_ = source.list[i];
    Memory memory = CObjectWrapper::unwrap(memory_);
    if (memory.kind() == kind) result.push_back(memory_);
  }

  bishop_memory_list_t result_ = bishop_create_memory_list(result.size());
  for (unsigned i = 0; i < result.size(); ++i)
    result_.list[i] = result[i];
  return result_;
}

bishop_isa_t
bishop_processor_get_isa(legion_processor_t proc_)
{
  Processor proc = CObjectWrapper::unwrap(proc_);
  switch (proc.kind())
  {
    case Processor::LOC_PROC:
      {
        return X86_ISA;
      }
    case Processor::TOC_PROC:
      {
        return CUDA_ISA;
      }
    default:
      {
        assert(false);
        return X86_ISA; // unreachable
      }
  }
}

legion_memory_t
bishop_physical_region_get_memory(legion_physical_region_t pr_)
{
  set<Memory> memories;
  PhysicalRegion* pr = CObjectWrapper::unwrap(pr_);
  pr->get_memories(memories);
  assert(memories.size() > 0);
  return CObjectWrapper::wrap(*memories.begin());
}

bishop_memory_list_t
bishop_physical_region_get_memories(legion_physical_region_t pr_)
{
  set<Memory> memories;
  PhysicalRegion* pr = CObjectWrapper::unwrap(pr_);
  pr->get_memories(memories);
  bishop_memory_list_t memories_ = bishop_create_memory_list(memories.size());
  unsigned idx = 0;
  for (set<Memory>::const_iterator it = memories.begin();
       it != memories.end(); it++)
    memories_.list[idx++] = CObjectWrapper::wrap(*it);
  return memories_;
}

bishop_field_list_t
bishop_physical_region_get_fields(legion_physical_region_t pr_)
{
  vector<FieldID> fields;
  PhysicalRegion* pr = CObjectWrapper::unwrap(pr_);
  pr->get_fields(fields);
  bishop_field_list_t fields_ = bishop_create_field_list(fields.size());
  unsigned idx = 0;
  for (vector<FieldID>::const_iterator it = fields.begin();
       it != fields.end(); it++)
    fields_.list[idx++] = *it;
  return fields_;
}

typedef std::pair<size_t, std::pair<LogicalRegion, Memory> > InstanceCacheKey;
typedef std::map<InstanceCacheKey,
                 legion_physical_instance_t*> InstanceCache;

bishop_instance_cache_t
bishop_instance_cache_create()
{
  bishop_instance_cache_t cache_;
  cache_.impl = new InstanceCache;
  return cache_;
}

legion_physical_instance_t*
bishop_instance_cache_get_cached_instances(bishop_instance_cache_t cache_,
                                           size_t idx,
                                           legion_logical_region_t lr_,
                                           legion_memory_t memory_)
{
  InstanceCache *cache = (InstanceCache*)cache_.impl;

  LogicalRegion lr = CObjectWrapper::unwrap(lr_);
  Memory memory = CObjectWrapper::unwrap(memory_);
  InstanceCacheKey key(idx, std::make_pair(lr, memory));
  InstanceCache::iterator finder = cache->find(key);
  if (finder == cache->end())
    return NULL;
  else
  // TODO: Some layout constraints may require multiple instances
    return finder->second;
}

bool
bishop_instance_cache_register_instances(bishop_instance_cache_t cache_,
                                         size_t idx,
                                         legion_logical_region_t lr_,
                                         legion_memory_t memory_,
                                         legion_physical_instance_t *instances)
{
  InstanceCache *cache = (InstanceCache*)cache_.impl;

  LogicalRegion lr = CObjectWrapper::unwrap(lr_);
  Memory memory = CObjectWrapper::unwrap(memory_);
  InstanceCacheKey key(idx, std::make_pair(lr, memory));
  InstanceCache::iterator finder = cache->find(key);
  if (finder != cache->end())
  {
#ifdef DEBUG_LEGION
    assert(CObjectWrapper::unwrap(*finder->second)->get_instance_id() ==
        CObjectWrapper::unwrap(*instances)->get_instance_id());
#endif
    return false;
  }
  else
  {
    // TODO: Some layout constraints may require multiple instances
    (*cache)[key] = instances;
    return true;
  }
}

typedef std::map<Domain, std::vector<Mapper::TaskSlice> > SliceCache;

bishop_slice_cache_t
bishop_slice_cache_create()
{
  bishop_slice_cache_t cache_;
  cache_.impl = new SliceCache;
  return cache_;
}

bool
bishop_slice_cache_has_cached_slices(bishop_slice_cache_t cache_,
                                     legion_domain_t domain_)
{
  SliceCache *cache = (SliceCache*)cache_.impl;
  return cache->find(CObjectWrapper::unwrap(domain_)) != cache->end();
}

void
bishop_slice_cache_copy_cached_slices(bishop_slice_cache_t cache_,
                                      legion_domain_t domain_,
                                      legion_slice_task_output_t output_)
{
  SliceCache *cache = (SliceCache*)cache_.impl;
  SliceCache::iterator finder = cache->find(CObjectWrapper::unwrap(domain_));
#ifdef DEBUG_LEGION
  assert(finder != cache->end());
#endif
  Mapper::SliceTaskOutput* output = CObjectWrapper::unwrap(output_);
  output->slices.reserve(finder->second.size());
  output->slices = finder->second;
}

void
bishop_slice_cache_add_entry(bishop_slice_cache_t cache_,
                             legion_domain_t domain_,
                             legion_slice_task_output_t output_)
{
  SliceCache *cache = (SliceCache*)cache_.impl;
  Mapper::SliceTaskOutput* output = CObjectWrapper::unwrap(output_);
  (*cache)[CObjectWrapper::unwrap(domain_)] = output->slices;
}

coord_t
bishop_domain_point_linearize(legion_domain_point_t point_,
                              legion_domain_t domain_,
                              legion_task_t task_)
{
  DomainPoint point = CObjectWrapper::unwrap(point_);
  Domain domain = CObjectWrapper::unwrap(domain_);
  Task *task = CObjectWrapper::unwrap(task_);

  if (!task->is_index_space || point.get_dim() == 0) return 0;

#ifdef DEBUG_LEGION
  assert(domain.dense());
  assert(point.get_dim() == domain.get_dim());
#endif

  int dim = domain.get_dim();
  coord_t linearized = 0;
  std::vector<coord_t> sizes(dim);
  DomainPoint lo = domain.lo();
  DomainPoint hi = domain.hi();

  for (int i = 0; i < dim; ++i)
    sizes[i] = hi[i] - lo[i] + 1;
  linearized = point[dim - 1];
  for (int i = dim - 2; i >= 0; --i)
    linearized = linearized * sizes[i] + point[i];
  return linearized;
}

void
bishop_logger_info(const char* msg, ...)
{
  va_list args;
  va_start(args, msg);
  log_bishop.info().vprintf(msg, args);
  va_end(args);
}

void
bishop_logger_warning(const char* msg, ...)
{
  va_list args;
  va_start(args, msg);
  log_bishop.warning().vprintf(msg, args);
  va_end(args);
}

void
bishop_logger_debug(const char* msg, ...)
{
  va_list args;
  va_start(args, msg);
  log_bishop.debug().vprintf(msg, args);
  va_end(args);
}
