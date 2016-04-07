// simple test harness

#ifndef RUNTESTS_H
#define RUNTESTS_H

#include <legion.h>

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

extern LegionRuntime::Logger::Category log_app;

enum FieldIDs {
  FID_X = 10000,
  FID_Y,
  FID_Z,
};

#define FID(n) (FID_X + (n))

struct Globals {
  IndexSpace ispace;
  FieldSpace fspace;
  std::vector<LogicalRegion> regions;
};

enum TestResult {
  RESULT_UNKNOWN,
  RESULT_PASS,
  RESULT_FAIL,
  RESULT_SKIP,
};

typedef TestResult (*TestEntryFn)(Runtime *runtime, Context ctx, const Globals& g);  

// declare one of these for each of your tests as a global variable - the constructor adds itself to
//  the global test list, which is iterated on by the main task
class TestInformation {
public:
  TestInformation(const char *_name, TestEntryFn _entry_fn,
		  int _num_regions, int _num_elements, int _num_fields);

  char name[40];
  TestEntryFn entry_fn;
  int num_regions;
  int num_elements;
  int num_fields;
  TestInformation *next;
};

class DelayedPredicate {
public:
  DelayedPredicate(Runtime *_runtime, Context _ctx);
  operator Predicate(void) const;
  DelayedPredicate& operator=(bool newval);
  DelayedPredicate& operator=(Future f);
protected:
  Runtime *runtime;
  Context ctx;
  DynamicCollective dc;
  Predicate p;
};

#endif
