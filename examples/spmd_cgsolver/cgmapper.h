// mapper for SPMD CG solver

#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

class CGMapper : public DefaultMapper {
public:
  CGMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual ~CGMapper(void);

  // supported tunables
  enum {
    TID_NUM_SHARDS = 11,
  };

  // mapper tags to pick specific shards
  enum {
    // tasks tagged with LOCAL_SHARD are sent to the proc group for the local shard
    TAG_LOCAL_SHARD = 100,

    TAG_SHARD_BASE = 1000,
    TAG_SHARD_END = TAG_SHARD_BASE + 65536,
  };

  static MappingTagID SHARD_TAG(int shard) { return TAG_SHARD_BASE + shard; }

  virtual void select_task_options(Task *task);

  virtual bool pre_map_task(Task *task);

  virtual bool map_must_epoch(const std::vector<Task*> &tasks,
			      const std::vector<MappingConstraint> &constraints,
			      MappingTagID tag);

  virtual void notify_mapping_result(const Mappable *mappable);

  virtual int get_tunable_value(const Task *task, 
				TunableID tid,
				MappingTagID tag);

protected:
  bool shard_per_proc;
  std::vector<Memory> sysmems;
  std::vector<std::vector<Processor> > procs;
  std::map<Processor, int> proc_to_shard;
  HighLevelRuntime *runtime;
};
