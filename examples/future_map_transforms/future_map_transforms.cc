#include <legion.h>

using namespace Legion;

enum TaskIDs {
  TID_TOP_LEVEL,
};

class DelinearizeFunctor : public PointTransformFunctor {
public:
  virtual DomainPoint transform_point(const DomainPoint &point,
                                      const Domain &domain,
                                      const Domain &range)
  {
    Point<1> p = point;
    return DomainPoint(Point<2>(p[0] % 2, p[0] / 2));
  }
};

class TransposeFunctor : public PointTransformFunctor {
public:
  virtual DomainPoint transform_point(const DomainPoint &point,
                                      const Domain &domain,
                                      const Domain &range)
  {
    Point<2> p = point;
    return DomainPoint(Point<2>(p[1], p[0]));
  }
};

DomainPoint linearize_function(const DomainPoint &point,
                               const Domain &domain,
                               const Domain &range)
{
  Point<2> p = point;
  coord_t x = p[1] * 2 + p[0];
  return DomainPoint(Point<1>(x));
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime *runtime) {
  const Rect<1> bounds1d(Point<1>(0), Point<1>(3));
  IndexSpace space1d = runtime->create_index_space(ctx, Domain(bounds1d));
  const Rect<2> bounds2d(Point<2>(0,0),Point<2>(1,1));
  IndexSpace space2d = runtime->create_index_space(ctx, Domain(bounds2d));

  std::vector<unsigned> data(4);
  for (unsigned idx = 0; idx < 4; idx++)
    data[idx] = idx;

  std::map<DomainPoint,TaskArgument> args;
  for (unsigned idx = 0; idx < 4; idx++)
    args[DomainPoint(Point<1>(idx))] = TaskArgument(&data[idx], sizeof(data[idx]));

  FutureMap f1 = runtime->construct_future_map(ctx, space1d, args);

  FutureMap f2 = runtime->transform_future_map(ctx, f1, space2d, linearize_function);

  FutureMap f3 = runtime->transform_future_map(ctx, f2, space2d,
                                new TransposeFunctor(), true/*own*/);

  FutureMap f4 = runtime->transform_future_map(ctx, f3, space1d,
                                new DelinearizeFunctor(), true/*own*/);

  Future zero = f4.get_future(DomainPoint(Point<1>(0)));
  Future one = f4.get_future(DomainPoint(Point<1>(1)));
  Future two = f4.get_future(DomainPoint(Point<1>(2)));
  Future three = f4.get_future(DomainPoint(Point<1>(3)));

  assert(zero.get_result<unsigned>() == 0);
  assert(one.get_result<unsigned>() == 2);
  assert(two.get_result<unsigned>() == 1);
  assert(three.get_result<unsigned>() == 3);
  
  runtime->destroy_index_space(ctx, space1d);
  runtime->destroy_index_space(ctx, space2d);
  printf("SUCCESS\n");
}

int main(int argc, char **argv) {
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  return Runtime::start(argc, argv);
}
