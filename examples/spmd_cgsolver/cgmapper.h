// mapper for SPMD CG solver

#include "legion.h"
#include "shim_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class CGMapper : public ShimMapper {
public:
  CGMapper(Machine machine, Runtime *rt, Processor local);
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

  using ShimMapper::select_task_options;
  virtual void select_task_options(Task *task);

  virtual bool pre_map_task(Task *task);

  virtual void notify_mapping_result(const Mappable *mappable);

  virtual int get_tunable_value(const Task *task, 
				TunableID tid,
				MappingTagID tag);

  // override DefaultMapper's policy for choosing locations for
  // instances constrained by a must epoch launch
  virtual Memory default_policy_select_constrained_instance_constraints(
				    MapperContext ctx,
				    const std::vector<const Legion::Task *> &tasks,
				    const std::vector<unsigned> &req_indexes,
				    const std::vector<Processor> &target_procs,
				    const std::set<LogicalRegion> &needed_regions,
				    const std::set<FieldID> &needed_fields,
                                    LayoutConstraintSet &constraints);

protected:
  bool shard_per_proc;
  std::vector<Memory> sysmems;
  std::vector<std::vector<Processor> > procs;
  std::map<Processor, int> proc_to_shard;
  Runtime *runtime;
};
