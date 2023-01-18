#include "realm.h"
#include "realm/cmdline.h"

using namespace Realm;

Logger log_app("app");

enum {
  HELLO_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

void hello_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  log_app.info() << "Hello World!";
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, &argv);
  rt.register_task(HELLO_TASK, hello_task);

  Event e = rt.collective_spawn_by_kind(Processor::LOC_PROC, HELLO_TASK,
                                        /*args=*/nullptr,
                                        /*arglen=*/0, /*one_per_node=*/true);
  rt.shutdown(e);
  rt.wait_for_shutdown();

  return 0;
}

