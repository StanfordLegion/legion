// tests the ability of realm to clumpify pieces of a sparse index space
//  (i.e. find points/subrectangles that can be grouped into larger
//  subrectangles) - the number of subrectangles needed to describe any
//  given sparse index space has a large impact on the performance of
//  operations (e.g. copies, deppart ops) that use that index space
//
//  NOTE: the most aggressive clumpification strategies are too computationally
//  expensive, so we're mostly focused on getting common cases right

#include "realm.h"
#include "realm/cmdline.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>
#include "osdep.h"
#include "philox.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  DYNAMIC_TASK_START
};

namespace TestConfig {
  unsigned dim_mask = 7; // i.e. 1-D, 2-D, 3-D
  unsigned type_mask = 3; // i.e. int, long long
  int log2_maxgrid = 8;
  int random_tests = 200;
  int random_seed = 12345;
  int max_holes = 3;
  bool verbose = false;
};

class PRNG {
public:
  typedef Philox_2x32<> PRNGBase;

  PRNG(uint32_t _seed, uint32_t _stream)
    : seed(_seed), stream(_stream), counter(0)
  {}

  uint32_t rand_int(uint32_t n)
  {
    return PRNGBase::rand_int(seed, stream, counter++, n);
  }

  uint64_t rand_raw()
  {
    return PRNGBase::rand_raw(seed, stream, counter++);
  }

protected:
  uint32_t seed, stream, counter;
};

template <int N, typename T, typename T2>
bool check_space(int seed, int subcase, const Rect<N,T>& grid_size,
                 const std::vector<T2>& pts_or_rects, IndexSpace<N,T> is,
                 size_t exp_volume, const Rect<N,T>& exp_bounds, size_t max_pieces)
{
  bool ok = true;

  size_t act_volume = is.volume();
  if(act_volume != exp_volume) {
    log_app.error() << "volume mismatch: seed=" << seed << " subcase=" << subcase
                    << " is=" << is << " expected=" << exp_volume
                    << " actual=" << act_volume;
    ok = false;
  }

  Rect<N,T> act_bounds = is.bounds;
  if(act_bounds != exp_bounds) {
    log_app.error() << "bounds mismatch: seed=" << seed << " subcase=" << subcase
                    << " is=" << is << " expected=" << exp_bounds
                    << " actual=" << act_bounds;
    ok = false;
  }

  size_t act_pieces = 0;
  for(IndexSpaceIterator<N,T> it(is); it.valid; it.step())
    act_pieces++;
  if(act_pieces > max_pieces) {
    log_app.error() << "pieces mismatch: seed=" << seed << " subcase=" << subcase
                    << " is=" << is << " maximum=" << max_pieces
                    << " actual=" << act_pieces;
    ok = false;
  }

  if(!ok && TestConfig::verbose) {
    log_app.warning() << " computed rects: ";
    for(IndexSpaceIterator<N,T> it(is); it.valid; it.step())
      log_app.warning() << "  " << it.rect;
    
    log_app.warning() << " original pts/rects: " << PrettyVector<T2>(pts_or_rects);
  }

  return ok;
}

template <typename T>
void random_permute(std::vector<T>& v, PRNG& prng)
{
  size_t n = v.size();
  for(size_t i = 0; i < n; i++) {
    size_t i1 = prng.rand_raw() % n;
    size_t i2 = prng.rand_raw() % n;
    if(i1 != i2) {
      using std::swap;
      swap(v[i1], v[i2]);
    }
  }
}

template <typename T>
bool in_list(const T& val, const std::vector<T>& vec)
{
  for(size_t i = 0; i < vec.size(); i++)
    if(val == vec[i])
      return true;
  return false;
}

template <int N, typename T>
bool test_case(int seed)
{
  // figure out the grid we'll operate on
  PRNG prng_grid(seed, 0);
  Rect<N,T> grid_size;
  for(int i = 0; i < N; i++) {
    grid_size.lo[i] = 0;
    grid_size.hi[i] = prng_grid.rand_int(1 << ((TestConfig::log2_maxgrid / N) +
                                               ((i < (TestConfig::log2_maxgrid % N)) ? 1 : 0)));
  }

  // each grid point corresponds to a subrectangle in T^N
  std::vector<T> grid_coords[N];
  for(int i = 0; i < N; i++) {
    grid_coords[i].resize(2 + grid_size.hi[i]);
    while(true) {
      for(int j = 0; j < (grid_size.hi[i] + 2); j++) {
        uint64_t v = prng_grid.rand_raw();
        memcpy(&grid_coords[i][j], &v, sizeof(T));
      }
      // sort the coordinates, and in the (fairly unlikely) case of a duplicate,
      //   try again
      std::sort(grid_coords[i].begin(), grid_coords[i].end());
      bool ok = true;
      for(int j = 0; j < (grid_size.hi[i] + 1); j++)
        if(grid_coords[i][j] == grid_coords[i][j+1]) ok = false;
      if(ok) break;
    }
  }

  Rect<N,T> exp_bounds;
  for(int i = 0; i < N; i++) {
    exp_bounds.lo[i] = grid_coords[i][0];
    exp_bounds.hi[i] = grid_coords[i][grid_size.hi[i] + 1] - 1;
  }

  size_t exp_pt_volume = grid_size.volume();
  size_t exp_rect_volume = exp_bounds.volume();

  // now decide on holes
  size_t num_holes = prng_grid.rand_int(TestConfig::max_holes);
  std::vector<Point<N,T> > holes;  // using grid indices, not coords!
  holes.reserve(num_holes);
  for(size_t i = 0; i < num_holes; i++) {
    // pick a random point that isn't the lo, hi, or one we've already chosen
    int tries = 0;
    static const int MAX_TRIES = 10;
    while(tries++ < MAX_TRIES) {
      Point<N, T> p;
      for(int j = 0; j < N; j++)
        p[j] = prng_grid.rand_int(grid_size.hi[j] + 1);
      if((p != grid_size.lo) && (p != grid_size.hi) && !in_list(p, holes)) {
        holes.push_back(p);

        exp_pt_volume -= 1;
        {
          Rect<N,T> r;
          for(int k = 0; k < N; k++) {
            r.lo[k] = grid_coords[k][p[k]];
            r.hi[k] = grid_coords[k][p[k] + 1] - 1;
          }
          exp_rect_volume -= r.volume();
        }

        break;
      }
    }
    if(tries >= MAX_TRIES) {
      // stop here, and reduce the holes to however many we managed to create
      num_holes = holes.size();
      break;
    }
  }

  // first test: individual points, no duplicates
  {
    PRNG prng_pts(seed, 1);
    std::vector<Point<N,T> > pts;
    for(PointInRectIterator<N,T> it(grid_size); it.valid; it.step())
      if(!in_list(it.p, holes)) {
        Point<N,T> p;
        for(int i = 0; i < N; i++)
          p[i] = /*grid_coords[i][0] +*/ it.p[i];
        pts.push_back(p);
      }
    random_permute(pts, prng_pts);
    IndexSpace<N,T> is(pts, true /*disjoint*/);
    // each hole should require at most 2N-1 more subrectangles to describe
    //  its absense (i.e. one subrectangle for each of the 2N planes, minus
    //  the one we were already planning to use)
    size_t max_pieces = 1 + num_holes * (2 * N - 1);
    if(!check_space(seed, 1, grid_size, pts, is,
                    exp_pt_volume, grid_size, max_pieces)) return false;
    is.destroy();
  }

  // second test: individual rectangles, no duplicates
  {
    PRNG prng_pts(seed, 2);
    std::vector<Rect<N,T> > rects;
    for(PointInRectIterator<N,T> it(grid_size); it.valid; it.step())
      if(!in_list(it.p, holes)) {
        Rect<N,T> r;
        for(int i = 0; i < N; i++) {
          r.lo[i] = grid_coords[i][it.p[i]];
          r.hi[i] = grid_coords[i][it.p[i] + 1] - 1;
        }
        rects.push_back(r);
      }
    random_permute(rects, prng_pts);
    IndexSpace<N,T> is(rects, true /*disjoint*/);
    size_t max_pieces;
    if(N == 1) {
      // each hole should require at most 2N-1 more subrectangles to describe
      //  its absense (i.e. one subrectangle for each of the 2N planes, minus
      //  the one we were already planning to use)
      max_pieces = 1 + num_holes * (2 * N - 1);
    } else {
      // due to the randomness of the speculative merging, we can't actually
      //  know that everything will coalesce into the minimal number of rectangles
      max_pieces = rects.size();
    }
    if(!check_space(seed, 1, grid_size, rects, is,
                    exp_rect_volume, exp_bounds, max_pieces)) return false;
    is.destroy();
  }

  return true;
}

template <int N, typename T>
bool test_dim_and_type(int& seed)
{
  for(int i = 0; i < TestConfig::random_tests; i++) {
    if(!test_case<N,T>(seed++)) return false;
  }

  return true;
}

template <int N>
bool test_dim(int& seed)
{
  if(((TestConfig::type_mask & 1) != 0) && !test_dim_and_type<N,int>(seed))
    return false;
  if(((TestConfig::type_mask & 2) != 0) && !test_dim_and_type<N,long long>(seed))
    return false;
  return true;
}

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Realm sparse index space construction test";

  bool ok = true;

  int seed = TestConfig::random_seed;

  if(ok && ((TestConfig::dim_mask & 1) != 0) && !test_dim<1>(seed))
    ok = false;
  if(ok && ((TestConfig::dim_mask & 2) != 0) && !test_dim<2>(seed))
    ok = false;
  if(ok && ((TestConfig::dim_mask & 4) != 0) && !test_dim<3>(seed))
    ok = false;

  if(ok)
    log_app.info() << "sparse_construct test finished successfully";
  else
    log_app.error() << "sparse_construct test finished with errors!";

  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(),
				  ok ? 0 : 1);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-dims", TestConfig::dim_mask);
  cp.add_option_int("-types", TestConfig::type_mask);
  cp.add_option_int("-rand", TestConfig::random_tests);
  cp.add_option_int("-seed", TestConfig::random_seed);
  cp.add_option_int("-grid", TestConfig::log2_maxgrid);
  cp.add_option_bool("-verbose", TestConfig::verbose);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
  
  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // shutdown will be requested by main task

  // now sleep this thread until that shutdown actually happens
  int result = rt.wait_for_shutdown();
  
  return result;
}
