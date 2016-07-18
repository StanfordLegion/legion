// mapper for SPMD CG solver

#include "cgmapper.h"

LegionRuntime::Logger::Category log_cgmap("cgmapper");

// thanks to the wonders of ADL, this template has to be in either the Realm or std
//  namespace to be found...
namespace Realm {
  template <typename T>
  std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
  {
    switch(v.size()) {
    case 0: 
      {
	os << "[]";
	break;
      }
    case 1:
      {
	os << "[ " << v[0] << " ]";
	break;
      }
    default:
      {
	os << "[ " << v[0];
	for(size_t i = 1; i < v.size(); i++)
	  os << ", " << v[i];
	os << " ]";
      }
    }
    return os;
  }
};

CGMapper::CGMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : ShimMapper(machine, rt, rt->get_mapper_runtime(), local)
  , shard_per_proc(false)
  , runtime(rt)
{
  // check to see if there any input arguments to parse
  {
    int argc = HighLevelRuntime::get_input_args().argc;
    const char **argv = (const char **)HighLevelRuntime::get_input_args().argv;

    for(int i=1; i < argc; i++) {
      if(!strcmp(argv[i], "-perproc")) {
	shard_per_proc = true;
	continue;
      }
    }
  }

  // we're going to do a SPMD distribution with one shard per "system memory"
  // (there's one of these per node right now, but we might have more with NUMA
  // eventually)
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  for(Machine::MemoryQuery::iterator it = mq.begin();
      it != mq.end();
      it++) {
    Memory m = *it;

    Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine)
      .only_kind(Processor::LOC_PROC)
      .best_affinity_to(m);
    if(shard_per_proc) {
      // create an entry for each proc
      for(Machine::ProcessorQuery::iterator it2 = pq.begin(); it2; ++it2) {
	Processor p = *it2;
	sysmems.push_back(m);
	procs.push_back(std::vector<Processor>(1, p));
	proc_to_shard[p] = sysmems.size() - 1;
	log_cgmap.debug() << "sysmem=" << m << " proc=" << p;
      }
    } else {
      // get one representative CPU processor associated with the memory
      std::vector<Processor> ps(pq.begin(), pq.end());
      assert(!ps.empty());

      sysmems.push_back(m);
      procs.push_back(ps);
      for(std::vector<Processor>::const_iterator it2 = ps.begin(); it2 != ps.end(); it2++)
	proc_to_shard[*it2] = sysmems.size() - 1;
      log_cgmap.debug() << "sysmem=" << m << " proc=" << ps;
    }
  }

#if 0
  std::set<Memory> all_mems;
  machine.get_all_memories(all_mems);
  for(std::set<Memory>::const_iterator it = all_mems.begin();
      it != all_mems.end();
      it++)
    if(it->kind() == Memory::SYSTEM_MEM) {
      sysmems.push_back(*it);

      std::set<Processor> shared_procs;
      machine.get_shared_processors(*it, shared_procs);
      while(!shared_procs.empty() && (shared_procs.begin()->kind() != Processor::LOC_PROC))
	shared_procs.erase(shared_procs.begin());
      assert(!shared_procs.empty());
      procs.push_back(*(shared_procs.begin()));
      Processor p1 = *shared_procs.begin();
      Processor p2 = Machine::ProcessorQuery(machine)
	.only_kind(Processor::LOC_PROC)
	.best_affinity_to(*it)
	.first();
      Machine::ProcessorQuery *q = new Machine::ProcessorQuery(machine);
      q->only_kind(Processor::LOC_PROC);
      q->best_affinity_to(*it);
      std::cout << "count = " << q->count() << "\n";
      std::cout << "first = " << q->first() << "\n";
      std::cout << "random = " << q->random() << "\n";
      std::vector<Processor> v(q->begin(), q->end());
      std::cout << "count = " << v.size() << "\n";
      std::cout << "first = " << v[0] << "\n";
      delete q;
      assert(p1 == p2);
    }
#endif

  if(sysmems.empty()) {
    log_cgmap.fatal() << "HELP!  No system memories found!?";
    assert(false);
  }
}

CGMapper::~CGMapper(void)
{
}


void CGMapper::select_task_options(Task *task)
{
  log_cgmap.print() << "select_task_options: id=" << task->task_id << " tag=" << task->tag;

  // is this a sharded task?
  if(task->tag >= TAG_SHARD_BASE) {
    int shard = task->tag - TAG_SHARD_BASE;
    assert(shard < (int)(procs.size()));
    
    task->inline_task = false;
    task->spawn_task = false;
    task->map_locally = true; 
    task->profile_task = false;
    task->task_priority = 0;
    task->target_proc = procs[shard][0];
    return;
  }

  if(task->tag == TAG_LOCAL_SHARD) {
    int shard = proc_to_shard[task->orig_proc];

    task->inline_task = false;
    task->spawn_task = false;
    task->map_locally = true; 
    task->profile_task = false;
    task->task_priority = 0;
    task->target_proc = procs[shard][0];
    task->additional_procs.insert(procs[shard].begin(), procs[shard].end());
    return;
  }

  // fall through to default mapper's logic
  ShimMapper::select_task_options(task);
}

bool CGMapper::pre_map_task(Task *task)
{
  // assume that all must_early_map regions have an existing instance and just use that
  for(unsigned idx = 0; idx < task->regions.size(); idx++)
    if(task->regions[idx].early_map || (task->regions[idx].prop == SIMULTANEOUS)) {
      log_cgmap.print() << "pre_map_task needs early map: id " << task->task_id << " tag=" << task->tag 
			<< ": #" << idx << ": " << task->regions[idx].region << " (" << task->regions[idx].current_instances.size() << " current)";
      task->regions[idx].virtual_map = false;
      task->regions[idx].early_map = true;
      task->regions[idx].enable_WAR_optimization = false;
      task->regions[idx].reduction_list = false;
      task->regions[idx].make_persistent = false;
      task->regions[idx].blocking_factor = task->regions[idx].max_blocking_factor;

      if(task->regions[idx].tag >= TAG_SHARD_BASE) {
	int shard = task->regions[idx].tag - TAG_SHARD_BASE;
	assert(shard < (int)(procs.size()));

	if(!(task->regions[idx].current_instances.empty())) {
	  assert(task->regions[idx].current_instances.size() == 1);
	  Memory m = (task->regions[idx].current_instances.begin())->first;
	  assert(m == sysmems[shard]);
	}
	task->regions[idx].target_ranking.push_back(sysmems[shard]);
      } else {
	assert(0);
      }
    }

  return true;
}

bool CGMapper::map_must_epoch(const std::vector<Task*> &tasks,
			      const std::vector<MappingConstraint> &constraints,
			      MappingTagID tag)
{
  // just map all the tasks, and rely on them satisfying the constraints
  for(std::vector<Task*>::const_iterator it = tasks.begin(); it != tasks.end(); it++)
    map_task(*it);

  return false;
}

void CGMapper::notify_mapping_result(const Mappable *mappable)
{
  const Task *task = mappable->as_mappable_task();
  const char *name = "(unknown)";
  runtime->retrieve_name(task->task_id, name);
  log_cgmap.print() << "task " << task->task_id << "(" << name << ") mapped on " << task->target_proc;
  for(unsigned idx = 0; idx < task->regions.size(); idx++) {
    const RegionRequirement& rr = task->regions[idx];
    log_cgmap.print() << " region #" << idx << ": " << rr.region << " (" << rr.privilege << "," << rr.prop
		      << ") mapped on " << task->regions[idx].selected_memory
		      << ", fields=" << task->regions[idx].instance_fields;
  }
}

int CGMapper::get_tunable_value(const Task *task, 
				TunableID tid,
				MappingTagID tag)
{
  switch(tid) {
  case TID_NUM_SHARDS:
    {
      return sysmems.size();
    }
  }
  // Unknown tunable value
  assert(false);
  return 0;
}
