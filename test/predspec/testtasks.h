// helper tasks for tests

#ifndef TESTTASKS_H
#define TESTTASKS_H

#include "runtests.h"

// sets one field of one entry to the specified value
class SetEntry {
public:
  struct Args {
    int idx;
    int newval;
  };

  class Launcher : public TaskLauncher {
  public:
    Launcher(LogicalRegion region, int idx, FieldID fid, int newval,
	     Predicate pred = Predicate::TRUE_PRED,
	     MappingTagID tag = 0);
  };

  static TaskID taskid;

  static Future run(Runtime *runtime, Context ctx,
		    LogicalRegion region, int idx, FieldID fid, int newval,
		    Predicate pred = Predicate::TRUE_PRED,
		    MappingTagID tag = 0);

  static void preregister_tasks(void);

  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};

// checks that one field of one entry has the desired value
class CheckEntry {
public:
  struct Args {
    int idx;
    int checkval;
  };

  class Launcher : public TaskLauncher {
  public:
    Launcher(LogicalRegion region, int idx, FieldID fid, int checkval,
	     Predicate pred = Predicate::TRUE_PRED,
	     MappingTagID tag = 0);
  };

  static TaskID taskid;

  static Future run(Runtime *runtime, Context ctx,
		    LogicalRegion region, int idx, FieldID fid, int checkval,
		    Predicate pred = Predicate::TRUE_PRED,
		    MappingTagID tag = 0);

  static void preregister_tasks(void);

  static bool cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};


#endif
