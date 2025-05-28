#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <time.h>
#include <thread>

#include "realm.h"

#include "osdep.h"

using namespace Realm;

// Task IDs, some IDs are reserved so start at first available number
enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  CHILD_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 1,
  CHECK_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 2,
};

enum
{
  REDOP_ADD = 1
};

template <typename T, size_t BYTES>
struct Pad {
  static_assert(BYTES >= sizeof(T), "Padding size must be at least the size of T");

  T val;
  char padding[BYTES - sizeof(T)];

  Pad() {}
  Pad(T _val)
    : val(_val)
  {}

  operator T() const { return val; }

  Pad &operator+=(const T &rhs)
  {
    val += rhs;
    return *this;
  }

  Pad &operator*=(const T &rhs)
  {
    val *= rhs;
    return *this;
  }
};

template <typename T, size_t BYTES>
std::ostream &operator<<(std::ostream &, const Pad<T, BYTES> &);

template <typename T, size_t BYTES>
std::ostream &operator<<(std::ostream &os, const Pad<T, BYTES> &pad)
{
  os << pad.val;
  return os;
}

static constexpr int BYTES = 64;

class ReductionOpIntAdd {
public:
  typedef Pad<float, BYTES> LHS;
  typedef Pad<float, BYTES> RHS;

  template <bool EXCL>
  static void apply(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  // both of these are optional
  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }
};

const ReductionOpIntAdd::RHS ReductionOpIntAdd::identity = 0;

struct ChildTaskArgs {
  size_t num_iters;
  size_t index;
  Barrier b;
};

static const Pad<float, BYTES> BARRIER_INITIAL_VALUE = 42;

static int errors = 0;

// we're going to use alarm() as a watchdog to detect deadlocks
void sigalrm_handler(int sig)
{
  fprintf(stderr, "HELP!  Alarm triggered - likely hang!\n");
  exit(1);
}

void child_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                Processor p)
{
  assert(arglen == sizeof(ChildTaskArgs));
  const ChildTaskArgs &child_args = *(const ChildTaskArgs *)args;

  printf("starting child task %zd on processor " IDFMT "\n", child_args.index, p.id);
  Barrier b = child_args.b; // so we can advance it
  for(size_t i = 0; i < child_args.num_iters; i++) {
    // make one task slower than all the others
    if(i != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    Pad<float, BYTES> reduce_val = (i + 1) * (child_args.index + 1);
    b.arrive(1, Event::NO_EVENT, &reduce_val, sizeof(reduce_val));

    // is it our turn to wait on the barrier?
    if(i == child_args.index) {
      Pad<float, BYTES> result;
      bool ready = b.get_result(&result, sizeof(result));
      if(!ready) {
        // wait on barrier to be ready and then ask for result again
        b.wait();
        ready = b.get_result(&result, sizeof(result));
        if(!ready) {
          printf("child %zd: iter %zd still not ready after explicit wait!?\n",
                 child_args.index, i);
          errors++;
        }
      }
      if(ready) {
        Pad<float, BYTES> exp_result =
            BARRIER_INITIAL_VALUE +
            (i + 1) * child_args.num_iters * (child_args.num_iters + 1) / 2;
        if(result == exp_result) {
          printf("child %zd: iter %zd = %f (%d) OK\n", child_args.index, i, result.val,
                 ready);
        } else {
          printf("child %zd: iter %zd = %f (%d) ERROR (expected %f)\n", child_args.index,
                 i, result.val, ready, exp_result.val);
          errors++;
        }
      }
    }

    b = b.advance_barrier();
  }

  printf("ending child task %zd on processor " IDFMT "\n", child_args.index, p.id);
}

void check_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                Processor p)
{
  assert(arglen == sizeof(ChildTaskArgs));
  const ChildTaskArgs &child_args = *(const ChildTaskArgs *)args;

  Pad<float, BYTES> result;
  bool ready = child_args.b.get_result(&result, sizeof(result));
  if(!ready) {
    printf("check %zd: barrier data wasn't ready!?\n", child_args.index);
    errors++;
  } else {
    Pad<float, BYTES> exp_result =
        BARRIER_INITIAL_VALUE +
        (child_args.index + 1) * child_args.num_iters * (child_args.num_iters + 1) / 2;
    if(result == exp_result) {
      printf("check %zd = %f OK\n", child_args.index, result.val);
    } else {
      printf("check %zd = %f ERROR (expected %f)\n", child_args.index, result.val,
             exp_result.val);
      errors++;
    }
  }
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  printf("top level task - getting machine and list of CPUs\n");

  Machine machine = Machine::get_machine();
  std::vector<Processor> all_cpus;
  {
    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
        it != all_processors.end(); it++)
      if((*it).kind() == Processor::LOC_PROC)
        all_cpus.push_back(*it);
  }

  printf("top level task - creating barrier\n");

  Barrier b = Barrier::create_barrier(all_cpus.size(), REDOP_ADD, &BARRIER_INITIAL_VALUE,
                                      sizeof(BARRIER_INITIAL_VALUE));

  std::set<Event> task_events;

  // spawn the check tasks before the arriving tasks
  {
    Barrier check_barrier = b;
    for(size_t i = 0; i < all_cpus.size(); i++) {
      ChildTaskArgs args;
      args.num_iters = all_cpus.size();
      args.index = i;
      args.b = check_barrier;

      Event e = all_cpus[i].spawn(CHECK_TASK, &args, sizeof(args), ProfilingRequestSet(),
                                  check_barrier);
      task_events.insert(e);

      check_barrier = check_barrier.advance_barrier();
    }
  }

  for(size_t i = 0; i < all_cpus.size(); i++) {
    ChildTaskArgs args;
    args.num_iters = all_cpus.size();
    args.index = i;
    args.b = b;

    Event e = all_cpus[i].spawn(CHILD_TASK, &args, sizeof(args));
    task_events.insert(e);
  }
  printf("%zd tasks launched\n", task_events.size());
  ;

  // now wait on each generation of the barrier and report the result
  for(size_t i = 0; i < all_cpus.size(); i++) {
    Pad<float, BYTES> result;
    bool ready = b.get_result(&result, sizeof(result));
    if(!ready) {
      // set an alarm so that we turn hangs into error messages.
      // this reset the alarm for the next wait
      alarm(all_cpus.size() * 2 + 10);
      // wait on barrier to be ready and then ask for result again
      b.wait();
      // We made progress, cancel any pending alarm to avoid it triggering spuriously
      alarm(0);
      bool ready2 = b.get_result(&result, sizeof(result));
      if(!ready2) {
        printf("parent: iter %zd still not ready after explicit wait!?\n", i);
        errors++;
        continue;
      }
    }
    Pad<float, BYTES> exp_result =
        BARRIER_INITIAL_VALUE + (i + 1) * all_cpus.size() * (all_cpus.size() + 1) / 2;
    if(result == exp_result) {
      printf("parent: iter %zd = %f.val (%d) OK\n", i, result.val, ready);
    } else {
      printf("parent: iter %zd = %f (%d) ERROR (expected %f)\n", i, result.val, ready,
             exp_result.val);
      errors++;
    }

    b = b.advance_barrier();
  }

  // wait on all child tasks to finish before destroying barrier
  Event merged = Event::merge_events(task_events);
  printf("merged event ID is " IDFMT " - waiting on it...\n", merged.id);
  merged.wait();

  b.destroy_barrier();

  if(errors > 0) {
    printf("Exiting with errors.\n");
    exit(1);
  }

  alarm(0);

  printf("done!\n");
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CHILD_TASK, child_task);
  rt.register_task(CHECK_TASK, check_task);

  rt.register_reduction<ReductionOpIntAdd>(REDOP_ADD);

  signal(SIGALRM, sigalrm_handler);

  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    std::set<Processor> all_procs;
    Machine::get_machine().get_all_processors(all_procs);
    for(std::set<Processor>::const_iterator it = all_procs.begin(); it != all_procs.end();
        it++)
      if(it->kind() == Processor::LOC_PROC) {
        p = *it;
        break;
      }
  }
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  return 0;
}
