#include "realm.h"
#include "realm/id.h"
#include "realm/cmdline.h"

#include <cassert>

using namespace Realm;

Logger log_app("app");

namespace TestConfig {
  size_t size = 15;
};

void run_set_operations() {
  const size_t size = TestConfig::size;
  std::vector<IndexSpace<2>> subspaces;
  for (size_t y = 0; y <= size; y++) {
    subspaces.push_back(
        IndexSpace<2>(Rect<2>(Point<2>(0, y), Point<2>(size, y))));
  }

  IndexSpace<2> center_is(Rect<2>(Point<2>(size / 2 - 1, size / 2 - 1),
                                  Point<2>(size / 2 + 1, size / 2 + 1)));

  IndexSpace<2> union_is;
  Event event1 =
      IndexSpace<2>::compute_union(subspaces, union_is, ProfilingRequestSet());

  IndexSpace<2> diffs_is;
  Event event2 = IndexSpace<2>::compute_difference(
      union_is, center_is, diffs_is, ProfilingRequestSet(), event1);

  IndexSpace<2> isect_is;
  Event event3 = IndexSpace<2>::compute_intersection(
      diffs_is, union_is, isect_is, ProfilingRequestSet(), event2);
  event3.wait();

  assert(union_is.dense() == false);
  assert(union_is.tighten(/*precise=*/true).dense() == true);

  IndexSpaceIterator<2> diffs_it(diffs_is), isect_it(isect_is);
  while (diffs_it.valid && isect_it.valid) {
    if (diffs_it.rect != isect_it.rect) {
      log_app.error() << "rects don't match: " << diffs_it.rect
                      << " != " << isect_it.rect;
    }
    diffs_it.step();
    isect_it.step();
  }
  if (diffs_it.valid || isect_it.valid) {
    log_app.error() << "At least one iterator is invalid";
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-size", TestConfig::size);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  run_set_operations();

  rt.shutdown(Event::NO_EVENT);
  rt.wait_for_shutdown();
  return 0;
}


