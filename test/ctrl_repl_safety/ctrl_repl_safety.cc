/* Copyright 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sys/wait.h>
#include <unistd.h>

#include <stdio.h>
#include <legion.h>

using namespace Legion;

template<typename T,
         T (FN)(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx,
                Runtime *runtime)>
void regLocTask(TaskID task_id, const char *name)
{
  TaskVariantRegistrar reg(task_id, name);
  reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  reg.set_replicable();
  Runtime::preregister_task_variant<T, FN>(reg, name);
}

struct Test
{
  enum ID
  {
    TASK_TOP_LEVEL,
    TASK_POSITIVE,
    TASK_NEGATIVE,
    REDOP_INTEGER_ADD,
  };

  struct Params
  {
    int is_always_positive;
    int is_check_enabled;
  };

  struct Result
  {
    size_t shard_count;
    int sum;
    bool is_complete;
  };

  class IntegerAdd
  {
  public:
    typedef int RHS;
    typedef int LHS;
    static const RHS identity = 0;
    template<bool EXCL> void apply(LHS &acc, RHS cur) const { acc += cur; }
    template<bool EXCL> void fold(RHS &a, RHS b) const { a += b; }
  };

  static const int TASK_COUNT = 4;

  static void register_ops()
  {
    Runtime::register_reduction_op<IntegerAdd>(REDOP_INTEGER_ADD);
    regLocTask<int, positive>(TASK_POSITIVE, "positive");
    regLocTask<int, negative>(TASK_NEGATIVE, "negative");
    regLocTask<Result, top_level>(TASK_TOP_LEVEL, "top_level");
  }

  static Result top_level(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx,
                          Runtime *rt)
  {
    const Params *p = (Params *)task->args;
    bool pick_positive = p->is_always_positive ||
                         rt->get_shard_id(ctx, true) == 0;

    IndexTaskLauncher launcher(pick_positive ? TASK_POSITIVE : TASK_NEGATIVE,
                               Rect<1>(0, TASK_COUNT - 1),
                               UntypedBuffer(),
                               ArgumentMap());

    Future f = rt->execute_index_space(ctx, launcher, REDOP_INTEGER_ADD);

    return Result{rt->get_num_shards(ctx, true), f.get_result<int>()};
  }

  static int positive(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx,
                      Runtime *rt)
  {
    return task->index_point[0] + 1;
  }

  static int negative(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx,
                      Runtime *rt)
  {
    return -positive(task, regions, ctx, rt);
  }

  static Result launch(const Params *params)
  {
    Runtime *rt = Runtime::get_runtime();
    TaskLauncher task(Test::TASK_TOP_LEVEL,
                      UntypedBuffer(params, sizeof *params));

    return rt->launch_top_level_task(task).get_result<Result>();
  }

  static Params parse(char **argv)
  {
    Params p{};
    // default values
    p.is_always_positive = true;

    for (const char *arg = argv[0]; arg != NULL; arg++)
    {
      if (arg[0] == ':') {
        sscanf(arg,
               ":%d:%d",
               &p.is_always_positive,
               &p.is_check_enabled);
        break;
      }
    }

    return p;
  }

  static bool is_expected(const Result *r, const Params *p)
  {
    int exp_sum = 0;
    for (int i = 1; i <= TASK_COUNT; i++) {
      exp_sum += i;
    }

    bool actually_correct = (exp_sum == r->sum);
    bool expected_correct = (p->is_always_positive || r->shard_count == 1);
    bool expected_complete = (!p->is_check_enabled || expected_correct);

    return actually_correct == expected_correct &&
           expected_complete == r->is_complete;
  }
};

int main(int argc, char **argv)
{
  ssize_t ret = 0;
  int result_pipe[2];
  ret = pipe(result_pipe);
  if (ret != 0) {
    return -1;
  }

  Test::Params params = Test::parse(argv);
  Test::Result result{0};

  pid_t child = fork();
  if (child == 0) {
    close(result_pipe[0]);

    // Run legion in the child
    Test::register_ops();
    Runtime::start(argc, argv, /* background = */ true);
    result = Test::launch(&params);

    // Seems we haven't crashed, write result to the pipe
    result.is_complete = true;
    ret = write(result_pipe[1], &result, sizeof result);
    if (ret != sizeof result) {
      exit(-1);
    }
    exit(Runtime::wait_for_shutdown());
  }

  close(result_pipe[1]);
  // if the child crashes, it won't actually write to the pipe
  ret = read(result_pipe[0], &result, sizeof result);
  if (ret != 0 && ret != sizeof result) {
    return -1;
  }

  int status = 0;
  pid_t finished = waitpid(child, &status, 0);
  assert(finished == child);

  printf("TASK_COUNT = %d\n", Test::TASK_COUNT);

  printf("Params(is_always_positive = %d, is_check_enabled = %d)\n",
         params.is_always_positive,
         params.is_check_enabled);

  printf("Result(shard_count = %zd, sum = %d, is_complete = %d)\n",
         result.shard_count,
         result.sum,
         result.is_complete);

  bool passed = Test::is_expected(&result, &params);
  printf("Test %s\n", passed ? "passed" : "failed");
  return !passed;
}
