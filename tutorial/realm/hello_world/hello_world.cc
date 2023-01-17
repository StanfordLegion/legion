#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "realm.h"
#include "realm/cmdline.h"

using namespace Realm;

Logger log_app("app");

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  const int *value = (const int *)args;
  assert(*value == 7);
  log_app.info() << "Hello! value=" << (*value);
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, &argv);
  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  int value = 7;
  Event e =
      rt.collective_spawn_by_kind(Processor::LOC_PROC, TOP_LEVEL_TASK, &value,
                                  sizeof(int), /*one_per_node=*/true);

  // request shutdown once that task is complete
  rt.shutdown(e);
  rt.wait_for_shutdown();

  return 0;
}

