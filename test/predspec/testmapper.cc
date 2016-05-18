#include "testmapper.h"

LegionRuntime::Logger::Category log_testmap("testmapper");

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

TestMapper::TestMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
  , runtime(rt)
{
  int seed = 12345;

  // check to see if there any input arguments to parse
  {
    int argc = HighLevelRuntime::get_input_args().argc;
    const char **argv = (const char **)HighLevelRuntime::get_input_args().argv;

    for(int i=1; i < argc; i++) {
      if(!strcmp(argv[i], "-seed")) {
	seed = atoi(argv[++i]);
	continue;
      }
    }
  }

  // initialize random state and scramble it a bit
  rstate[0] = seed > 16;
  rstate[1] = seed;
  rstate[2] = local.id % 60317; // this is a prime ~<= 2^16
  for(int i = 0; i < 10; i++)
    jrand48(rstate);

  // approximate the "nodes" by looking at sysmems and the procs with best
  //   affinity to them
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  for(Machine::MemoryQuery::iterator it = mq.begin();
      it != mq.end();
      it++) {
    Memory m = *it;

    size_t idx = sysmems.size();
    sysmems.push_back(m);
    procs.resize(idx + 1);
    std::vector<Processor>& ps = procs[idx];

    Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine)
      .only_kind(Processor::LOC_PROC)
      .best_affinity_to(m);

    for(Machine::ProcessorQuery::iterator it2 = pq.begin(); it2; ++it2) {
      ps.push_back(*it2);
      proc_to_node[*it2] = idx;
    }

    log_testmap.debug() << "sysmem=" << m << " proc=" << ps;
  }

  if(sysmems.empty()) {
    log_testmap.fatal() << "HELP!  No system memories found!?";
    assert(false);
  }
}

TestMapper::~TestMapper(void)
{
}


void TestMapper::select_task_options(Task *task)
{
  log_testmap.print() << "select_task_options: id=" << task->task_id << " tag=" << task->tag;

  int node;
  if((task->tag & TAGOPT_RANDOM_NODE) != 0) {
    node = jrand48(rstate) % sysmems.size();
  } else {
    // keep on local node
    node = proc_to_node[task->orig_proc];
  }

  task->inline_task = false;
  task->spawn_task = false;
  task->map_locally = true; 
  task->profile_task = false;
  task->task_priority = 0;
  task->target_proc = procs[node][0];
  task->additional_procs.insert(procs[node].begin(), procs[node].end());
  return;
}

bool TestMapper::speculate_on_predicate(const Mappable *mappable,
					bool &spec_value)
{
  if((mappable->tag & TAGOPT_SPECULATE_FALSE) != 0) {
    log_testmap.print() << "speculating false: mappable=" << (void *)mappable;
    spec_value = false;
    return true;
  }

  if((mappable->tag & TAGOPT_SPECULATE_TRUE) != 0) {
    log_testmap.print() << "speculating true: mappable=" << (void *)mappable;
    spec_value = true;
    return true;
  }

  return false;
}


void TestMapper::notify_mapping_result(const Mappable *mappable)
{
  const Task *task = mappable->as_mappable_task();
  const char *name = "(unknown)";
  runtime->retrieve_name(task->task_id, name);
  log_testmap.print() << "task " << task->task_id << "(" << name << ") mapped on " << task->target_proc;
  for(unsigned idx = 0; idx < task->regions.size(); idx++) {
    const RegionRequirement& rr = task->regions[idx];
    log_testmap.print() << " region #" << idx << ": " << rr.region << " (" << rr.privilege << "," << rr.prop
		      << ") mapped on " << task->regions[idx].selected_memory
		      << ", fields=" << task->regions[idx].instance_fields;
  }
}
