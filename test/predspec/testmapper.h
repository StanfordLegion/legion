#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

class TestMapper : public DefaultMapper {
public:
  TestMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual ~TestMapper(void);

  // application tasks can be tagged with one or more options applied by the mapper
  // mapper tags to pick specific shards
  enum {
    TAGOPT_SPECULATE_FALSE = (1 << 0),
    TAGOPT_SPECULATE_TRUE = (1 << 1),

    TAGOPT_RANDOM_NODE = (1 << 2),
  };

  virtual void select_task_options(Task *task);

  virtual bool speculate_on_predicate(const Mappable *mappable,
				      bool &spec_value);

  virtual void notify_mapping_result(const Mappable *mappable);

protected:
  std::vector<Memory> sysmems;
  std::vector<std::vector<Processor> > procs;
  std::map<Processor, int> proc_to_node;
  HighLevelRuntime *runtime;
  unsigned short rstate[3];
};
