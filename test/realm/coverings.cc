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

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  DYNAMIC_TASK_START
};

enum {
  FID_DATA = 100,
  FID_A,
  FID_B,
};

namespace TestConfig {
  unsigned dim_mask = 7; // i.e. 1-D, 2-D, 3-D
  unsigned type_mask = 3; // i.e. int, long long
  int random_tests = 20;
  int random_seed = 12345;
  bool warn_as_error = false;
};

template <int N, typename T, typename FT, typename LAMBDA>
class FillerTask {
public:
  struct Args {
    Args(LAMBDA _filler) : filler(_filler) {}
    IndexSpace<N,T> space;
    RegionInstance inst;
    FieldID field_id;
    LAMBDA filler;
  };

  static void task_body(const void *argdata, size_t arglen,
			const void *userdata, size_t userlen, Processor p)
  {
    assert(sizeof(Args) == arglen);
    const Args& args = *reinterpret_cast<const Args *>(argdata);
    log_app.info() << "filler: is=" << args.space << " inst=" << args.inst;

    args.inst.fetch_metadata(p).wait();			     
    AffineAccessor<FT,N,T> acc(args.inst, args.field_id);
    IndexSpaceIterator<N, T> it(args.space);
    while(it.valid) {
      PointInRectIterator<N,T> pit(it.rect);
      while(pit.valid) {
	FT val = args.filler(pit.p);
	log_app.debug() << "  [" << pit.p << "] = " << val;
	acc[pit.p] = val;
	pit.step();
      }
      it.step();
    }
    
  }
};

Processor::TaskFuncID next_func_id = DYNAMIC_TASK_START;
std::map<const char *, Processor::TaskFuncID> task_ids;

template <typename T>
static Processor::TaskFuncID lookup_task_id()
{
  const char *key = typeid(T).name();
  std::map<const char *, Processor::TaskFuncID>::const_iterator it = task_ids.find(key);
  if(it != task_ids.end())
    return it->second;

  Processor::TaskFuncID id = next_func_id++;
  Event e = Processor::register_task_by_kind(Processor::LOC_PROC, true /*global*/,
					     id,
					     CodeDescriptor(&T::task_body),
					     ProfilingRequestSet());
  e.wait();
  task_ids[key] = id;
  return id;
}					     

#if 0
template <int N, typename T>
template <typename FT, typename LAMBDA>
Event DistributedData<N,T>::fill(IndexSpace<N,T> is, FieldID fid, LAMBDA filler,
				 Event wait_on)
{
  typename FillerTask<N,T,FT,LAMBDA>::Args args(filler);
  args.field_id = fid;
  Processor::TaskFuncID id = lookup_task_id<FillerTask<N,T,FT,LAMBDA> >();
  std::vector<Event> events;
  for(typename std::vector<Piece>::iterator it = pieces.begin();
      it != pieces.end();
      ++it) {
    IndexSpace<N,T> isect;
    IndexSpace<N,T>::compute_intersection(is, it->space, isect, ProfilingRequestSet()).wait();
    args.space = isect;
    args.inst = it->inst;
    Event e = it->proc.spawn(id, &args, sizeof(args), ProfilingRequestSet(), wait_on);
    events.push_back(e);
  }

  // update reference data
  std::map<Point<N,T>, Maybe<FT> >& ref = get_ref_data<FT>(fid);
  IndexSpaceIterator<N, T> it(is);
  while(it.valid) {
    PointInRectIterator<N,T> pit(it.rect);
    while(pit.valid) {
      ref[pit.p] = filler(pit.p);
      pit.step();
    }
    it.step();
  }  

  return Event::merge_events(events);
}
#endif

template <int N, typename T>
bool test_case(const char *name,
	       IndexSpace<N,T> is, size_t volume,
	       const std::vector<Rect<N,T> >& input_rects,
	       size_t max_rects, int max_overhead, bool must_succeed,
	       std::vector<Rect<N,T> > *result = 0)
{
  std::vector<Rect<N,T> > covering;

  log_app.info() << name << ": max_rects=" << max_rects << " max_overhead=" << max_overhead << " must_succeed=" << must_succeed;

  bool ok = is.compute_covering(max_rects, max_overhead, covering);

  log_app.info() << "result: ok=" << ok << " covering=" << PrettyVector<Rect<N,T> >(covering);
  
  if(!ok) {
    if(must_succeed) {
      log_app.error() << name << ": should not have failed: max_rects=" << max_rects << " max_overhead=" << max_overhead;
      return false;
    }
    return true;
  }

  // verify containment
  for(size_t i = 0; i < input_rects.size(); i++) {
    bool found = false;
    // note that compute_covering doesn't officially guarantee rectangles
    //  will stay together, which would require testing this for every point,
    //  but the current implementation does, so do the faster version
    for(size_t j = 0; j < covering.size(); j++)
      if(covering[j].contains(input_rects[i])) {
	found = true;
	break;
      }
    if(!found) {
      log_app.error() << name << ": missing coverage: " << input_rects[i] << " not in " << PrettyVector<Rect<N,T> >(covering);
      return false;
    }
  }

  // verify non-overlapping
  for(size_t i = 0; i < covering.size(); i++)
    for(size_t j = i + 1; j < covering.size(); j++)
      if(covering[i].overlaps(covering[j])) {
	log_app.error() << name << ": overlap found: " << covering[i] << " and " << covering[j];
	return false;
      }

  if((max_rects > 0) && (covering.size() > max_rects)) {
    log_app.error() << name << ": too many rects: max_rects=" << max_rects << " actual=" << PrettyVector<Rect<N,T> >(covering);
    return false;
  }

  // even if our overhead limit was optimistic, we should never get a
  //  "successful" covering that violates what we asked for
  if(max_overhead >= 0) {
    size_t act_volume = 0;
    for(size_t i = 0; i < covering.size(); i++)
      act_volume += covering[i].volume();
    assert(act_volume >= volume);
    int overhead = ((act_volume * 100) / volume) - 100;
    if(overhead > max_overhead) {
      log_app.error() << name << ": too much overhead: max=" << max_overhead << " actual=" << overhead << " covering=" << PrettyVector<Rect<N,T> >(covering);
      return false;
    }
  }

  if(result)
    *result = covering;
  return true;
}

template <int N, typename T, typename FT>
bool check_and_update(IndexSpace<N,T> is,
		      const std::vector<Rect<N,T> >& rects,
		      RegionInstance inst,
		      FieldID fid,
		      bool is_varying,
		      int exp_offset,
		      int new_offset)
{
  size_t errors = 0;

  // can we do an affine instance for the whole thing?
  if(AffineAccessor<FT,N,T>::is_compatible(inst, fid)) {
    AffineAccessor<FT,N,T> acc(inst, fid);
    for(IndexSpaceIterator<N,T> it(is); it.valid; it.step()) {
      for(PointInRectIterator<N,T> pir(it.rect); pir.valid; pir.step()) {
	T expval = exp_offset;
	if(is_varying) {
	  T scale = 100;
	  for(int i = 0; i < N; i++, scale *= 100)
	    expval += pir.p[i] * scale;
	}
	T actval = acc[pir.p];
	if(actval != expval) {
	  if(errors++ < 10)
	    log_app.error() << "mismatch: inst=" << inst
			    << " point=" << pir.p
			    << " expected=" << expval
			    << " actual=" << actval;
	} else
	  log_app.debug() << pir.p << " " << expval << " " << actval;
	T newval = new_offset;
	T scale = 100;
	for(int i = 0; i < N; i++, scale *= 100)
	  newval += pir.p[i] * scale;
	acc[pir.p] = newval;
      }
    }
  } else {
    MultiAffineAccessor<FT,N,T> ma1(inst, fid);
    // const version isn't allowed to cache the current rectangle
    const MultiAffineAccessor<FT,N,T> ma2(ma1);

    for(size_t i = 0; i < rects.size(); i++) {
      if(AffineAccessor<FT,N,T>::is_compatible(inst, fid, rects[i])) {
	AffineAccessor<FT,N,T> acc(inst, fid, rects[i]);
	for(IndexSpaceIterator<N,T> it(is, rects[i]); it.valid; it.step()) {
	  for(PointInRectIterator<N,T> pir(it.rect); pir.valid; pir.step()) {
	    T expval = exp_offset;
	    if(is_varying) {
	      T scale = 100;
	      for(int i = 0; i < N; i++, scale *= 100)
		expval += pir.p[i] * scale;
	    }
	    FT* p1 = ma1.ptr(pir.p);
	    FT* p2 = ma2.ptr(pir.p);
	    FT* p3 = acc.ptr(pir.p);
	    if((p1 != p3) || (p2 != p3)) {
	      if(errors++ < 10)
		log_app.error() << "multi-affine pointer mismatch: ref="
				<< p3 << " nonconst=" << p1
				<< " const=" << p2;
	    }
	    T actval = acc[pir.p];
	    if(actval != expval) {
	      if(errors++ < 10)
		log_app.error() << "mismatch: inst=" << inst
				<< " point=" << pir.p
				<< " expected=" << expval
				<< " actual=" << actval;
	    } else
	      log_app.debug() << pir.p << " " << expval << " " << actval;
	    T newval = new_offset;
	    T scale = 100;
	    for(int i = 0; i < N; i++, scale *= 100)
	      newval += pir.p[i] * scale;
	    acc[pir.p] = newval;
	  }
	}
      } else {
	log_app.error() << "can't get affine accessor for subrect: inst=" << inst << " rect=" << rects[i];
	errors++;
      }
    }
  }

  if(errors > 0) {
    log_app.error() << errors << " total errors on inst=" << inst;
    return false;
  } else
    return true;
}

static int gcd(int a, int b)
{
  while(true) {
    if(a > b) a -= b; else
    if(b > a) b -= a; else
      return a;
  }
}

template <int N, typename T>
bool test_copies(Memory m,
		 IndexSpace<N,T> is,
		 const std::vector<Rect<N,T> >& input_rects,
		 const std::vector<Rect<N,T> >& covering1,
		 const std::vector<Rect<N,T> >& covering2)
{
  int num_layouts = 0;
  InstanceLayoutGeneric *ilc[8];
  RegionInstance inst[8];

  InstanceLayoutConstraints::FieldInfo fi_a, fi_b;
  fi_a.field_id = FID_A;
  fi_a.fixed_offset = false;
  fi_a.size = sizeof(int);
  fi_a.alignment = sizeof(int);
  fi_b.field_id = FID_B;
  fi_b.fixed_offset = false;
  fi_b.size = sizeof(long long);
  fi_b.alignment = sizeof(long long);
  InstanceLayoutConstraints ilc_soa, ilc_aos;
  ilc_soa.field_groups.resize(2);
  ilc_soa.field_groups[0].push_back(fi_a);
  ilc_soa.field_groups[1].push_back(fi_b);
  ilc_aos.field_groups.resize(1);
  ilc_aos.field_groups[0].push_back(fi_a);
  ilc_aos.field_groups[0].push_back(fi_b);
  
  int dim_order[N];
  for(int i = 0; i < N; i++) dim_order[i] = i;

  // dense instances
  ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, ilc_soa, dim_order);
  ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, ilc_aos, dim_order);

  // full sparse instances
  ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, input_rects, ilc_soa, dim_order);
  ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, input_rects, ilc_aos, dim_order);

  // custom covering1
  if(!covering1.empty()) {
    ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, covering1, ilc_soa, dim_order);
    ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, covering1, ilc_aos, dim_order);
  }

  // custom covering2
  if(!covering2.empty()) {
    ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, covering2, ilc_soa, dim_order);
    ilc[num_layouts++] = InstanceLayoutGeneric::choose_instance_layout<N,T>(is, covering2, ilc_aos, dim_order);
  }

  // create instances
  for(int i = 0; i < num_layouts; i++)
    RegionInstance::create_instance(inst[i], m, ilc[i], ProfilingRequestSet()).wait();
  
  // fill insts
  for(int i = 0; i < num_layouts; i++) {
    std::vector<CopySrcDstField> fill_src(2), fill_dst(2);
    int fv_a = 2*i + 1;
    long long fv_b = 2*i + 2;
    fill_src[0].set_fill(fv_a);
    fill_src[1].set_fill(fv_b);
    fill_dst[0].set_field(inst[i], FID_A, sizeof(int));
    fill_dst[1].set_field(inst[i], FID_B, sizeof(long long));
    is.copy(fill_src, fill_dst, ProfilingRequestSet()).wait();
  }

  // test every src->dst pair
  if(!check_and_update<N,T,int>(is, input_rects, inst[0], FID_A, false, 1, 10)) return false;
  if(!check_and_update<N,T,long long>(is, input_rects, inst[0], FID_B, false, 2, 11)) return false;

  std::vector<bool> done(num_layouts, false);
  int cur_offset = 10;
  for(int delta1 = 1; delta1 < num_layouts; delta1++) {
    if(done[delta1]) continue;

    // if delta1 and num_layouts are not relatively prime, we need to
    //  alternate it
    //  and another delta we haven't done in order to avoid coming back
    //  to the first instance too early
    int delta2 = 0;
    if(gcd(delta1, num_layouts) > 1) {
      delta2 = delta1 + 1;
      while(((delta2 < num_layouts) && done[delta2]) ||
	    (gcd(delta1 + delta2, num_layouts) > 1)) {
	delta2++;
	assert(delta2 < num_layouts + delta1);
      }
      if(delta2 < num_layouts)
	done[delta2] = true;
    }

    int from = 0;
    for(int i = 0; i < num_layouts; i++) {
      assert((i == 0) || (from != 0));
      int to = (from + delta1) % num_layouts;
      {
	std::vector<CopySrcDstField> fill_src(2), fill_dst(2);
	fill_src[0].set_field(inst[from], FID_A, sizeof(int));
	fill_src[1].set_field(inst[from], FID_B, sizeof(long long));
	fill_dst[0].set_field(inst[to], FID_A, sizeof(int));
	fill_dst[1].set_field(inst[to], FID_B, sizeof(long long));
	is.copy(fill_src, fill_dst, ProfilingRequestSet()).wait();
      }
      if(!check_and_update<N,T,int>(is, input_rects, inst[to],
				    FID_A, true,
				    cur_offset,
				    cur_offset + 2)) return false;
      if(!check_and_update<N,T,long long>(is, input_rects, inst[to],
					  FID_B, true,
					  cur_offset + 1,
					  cur_offset + 3)) return false;
      cur_offset += 2;

      if(delta2 == 0) {
	from = to;
      } else {
	int to2 = (to + delta2) % num_layouts;
	{
	  std::vector<CopySrcDstField> fill_src(2), fill_dst(2);
	  fill_src[0].set_field(inst[to], FID_A, sizeof(int));
	  fill_src[1].set_field(inst[to], FID_B, sizeof(long long));
	  fill_dst[0].set_field(inst[to2], FID_A, sizeof(int));
	  fill_dst[1].set_field(inst[to2], FID_B, sizeof(long long));
	  is.copy(fill_src, fill_dst, ProfilingRequestSet()).wait();
	}
	if(!check_and_update<N,T,int>(is, input_rects, inst[to2],
				      FID_A, true,
				      cur_offset,
				      cur_offset + 2)) return false;
	if(!check_and_update<N,T,long long>(is, input_rects, inst[to2],
					    FID_B, true,
					    cur_offset + 1,
					    cur_offset + 3)) return false;
	cur_offset += 2;
	from = to2;
      }
    }
  }

  // destroy instances
  for(int i = 0; i < num_layouts; i++)
    inst[i].destroy();

  return true;
}

template <int N, typename T>
bool test_input(Memory m, const std::vector<Rect<N,T> >& input_rects,
		size_t num_clumps, int clump_overhead, bool warn_heuristic,
		bool do_copies)
{
  IndexSpace<N,T> is(input_rects);

  log_app.info() << "testing input: input=" << PrettyVector<Rect<N,T> >(input_rects);

  // compute volume from original rects
  size_t vol = 0;
  for(size_t i = 0; i < input_rects.size(); i++)
    vol += input_rects[i].volume();
  // sanity-check
  assert(vol == is.volume());

  // test 1: a one-rect covering should always match the bounds
  {
    std::vector<Rect<N,T> > covering;
    if(!test_case("test 1",
		  is, vol, input_rects, 1, -1, true, &covering)) return false;
    if(covering[0] != is.bounds) {
      log_app.error() << "fail test 1: bounds=" << is.bounds << " covering=" << PrettyVector<Rect<N,T> >(covering);
      return false;
    }
  }

  // test 2: covering with no max_rect constraint should be no worse than
  //  what we started with
  {
    std::vector<Rect<N,T> > covering;
    if(!test_case("test 2",
		  is, vol, input_rects, 0, 0, true, &covering)) return false;
    if(covering.size() > input_rects.size()) {
      log_app.error() << "fail test 2: too many rects: covering=" << PrettyVector<Rect<N,T> >(covering);
      return false;
    }
  }

  // test 3: same as test 2 with the bound specified
  {
    std::vector<Rect<N,T> > covering;
    if(!test_case("test 3",
		  is, vol, input_rects, input_rects.size(), 0, true, &covering)) return false;
    if(covering.size() > input_rects.size()) {
      log_app.error() << "fail test 3: too many rects: covering=" << PrettyVector<Rect<N,T> >(covering);
      return false;
    }
  }

  // test the ability to squish into fewer rectangles
  for(size_t target = 2; (target <= 10) && (target < input_rects.size()); target++) {
    // this is not guaranteed to succeed for N>1, even with no coverage
    //  limit
    bool required = (N == 1);
    std::vector<Rect<N,T> > covering;
    if(!test_case("test 4",
		  is, vol, input_rects, target, -1,
		  required, &covering)) return false;
    // but we'll warn if it fails when it "should" work
    if(covering.empty() && warn_heuristic) {
      log_app.warning() << "unconstrained clumping failed: target=" << target << " input=" << PrettyVector<Rect<N,T> >(input_rects);
      if(TestConfig::warn_as_error)
	return false;
    }
  }

  // test the ability to squish into fewer rectangles with the known clump
  //  overhead
  for(size_t target = 2; (target <= 10) && (target < input_rects.size()); target++) {
    // this is not guaranteed to succeed for N>1, even if we're above the
    //  number of clumps
    bool required = (N == 1) && (target >= num_clumps);
    // can only expect overhead to be hit above clump size
    int max_overhead = ((target >= num_clumps) ? clump_overhead : -1);
    if(N > 1) max_overhead *= 2;
    std::vector<Rect<N,T> > covering;
    if(!test_case("test 5",
		  is, vol, input_rects, target, max_overhead,
		  required, &covering)) return false;
    // but we'll warn if it fails when it "should" work
    if((target >= num_clumps) && covering.empty() && warn_heuristic) {
      log_app.warning() << "clumping failed: target=" << target << " num_clumps=" << num_clumps << " input=" << PrettyVector<Rect<N,T> >(input_rects);
      if(TestConfig::warn_as_error)
	return false;
    }
  }

  if(do_copies) {
    std::vector<Rect<N,T> > covering1, covering2;

    is.compute_covering(2, -1, covering1);

    // if we've got enough input rects, aim for something in between 2 and
    //  all
    if(input_rects.size() > 5)
      is.compute_covering((input_rects.size() + 2) >> 1, -1, covering2);
    
    if(!test_copies(m, is, input_rects, covering1, covering2))
      return false;
  }

  return true;
}

template <int N, typename T>
bool test_directed(Memory m)
{
  return true;
}

template <int N, typename T>
size_t random_nonoverlapping_rects(size_t max_rects,
				   const Rect<N,T>& bounds,
				   std::vector<Rect<N,T> >& rects)
{
  rects.clear();
  rects.reserve(max_rects);
  for(size_t i = 0; i < max_rects; i++) {
    // randomly choose a rectangle that doesn't overlap with any existing one
    int failures = 0;
    while(true) {
      Rect<N,T> r;
      for(int j = 0; j < N; j++) {
	T v1 = bounds.lo[j] + (lrand48() % (bounds.hi[j] - bounds.lo[j] + 1));
	T v2 = bounds.lo[j] + (lrand48() % (bounds.hi[j] - bounds.lo[j] + 1));
	if(v1 <= v2) {
	  r.lo[j] = v1;
	  r.hi[j] = v2;
	} else {
	  r.lo[j] = v2;
	  r.hi[j] = v1;
	}
      }
      bool ok = true;
      for(size_t j = 0; j < i; j++)
	if(rects[j].overlaps(r)) {
	  ok = false;
	  break;
	}
      if(!ok) {
	if(failures++ > 100)
	  return i; // stop early
	else
	  continue; // try again
      }
      rects.push_back(r);
      break;
    } 
  }

  return max_rects; // got 'em all
}

template <int N, typename T>
size_t random_clumpy_rects(size_t clump_target,
			   size_t max_rects,
			   const Rect<N,T>& bounds,
			   std::vector<Rect<N,T> >& clumps,
			   std::vector<Rect<N,T> >& rects)
{
  size_t num_clumps = random_nonoverlapping_rects(clump_target,
						  bounds,
						  clumps);

  for(size_t i = 0; i < max_rects; i++) {
    // randomly choose a rectangle that doesn't overlap with any existing one
    int failures = 0;
    while(true) {
      size_t clump_idx = lrand48() % num_clumps;
      std::vector<Rect<N,T> > r;
      size_t count = random_nonoverlapping_rects(1 /*just want one*/,
						 clumps[clump_idx],
						 r);
      bool ok;
      if(count == 1) {
	ok = true;
	for(size_t j = 0; j < i; j++)
	  if(rects[j].overlaps(r[0])) {
	    ok = false;
	    break;
	  }
      } else
	ok = false;
      if(!ok) {
	if(failures++ < 100) continue;
	log_app.info() << "could not find non-overlapping rect - stopping";
	return i;
      }
      rects.push_back(r[0]);
      break;
    }
  }
  return max_rects;
}

template <int N, typename T>
bool test_random(Memory m, size_t max_rects, T min_coord, T max_coord)
{
  // first decide how many clumps we want
  size_t clump_target = (lrand48() % 10) + 1;
  Rect<N,T> overall_bounds;
  for(int i = 0; i < N; i++) {
    overall_bounds.lo[i] = min_coord;
    overall_bounds.hi[i] = max_coord;
  }
  std::vector<Rect<N,T> > clumps;
  std::vector<Rect<N,T> > rects;
  size_t num_rects = random_clumpy_rects(clump_target,
					 max_rects,
					 overall_bounds,
					 clumps,
					 rects);
  log_app.info() << "using clumps: " << PrettyVector<Rect<N,T> >(clumps);

  // try each subset of the rectangle list we've chosen
  size_t clump_vol = 0;
  std::vector<Rect<N,T> > clumps_used(clumps.size(), Rect<N,T>::make_empty());
  std::vector<Rect<N,T> > input_rects;
  size_t vol = 0;
  for(size_t i = 0; i < num_rects; i++) {
    const Rect<N,T>& r = rects[i];

    input_rects.push_back(r);
    vol += r.volume();

    // track the volume at the granularity of clumps too
    // which clump did this come from?
    bool found = false;
    for(size_t j = 0; j < clumps.size(); j++)
      if(clumps[j].contains(r)) {
	clump_vol -= clumps_used[j].volume();
	clumps_used[j] = clumps_used[j].union_bbox(r);
	clump_vol += clumps_used[j].volume();
	found = true;
	break;
      }
    assert(found);

    // given the clumps, we can compute a maximum overhead that should be
    //  needed for any covering with at least that many rectangles
    int clump_ratio = (clump_vol * 100 + (vol - 1)) / vol;
    assert(clump_ratio >= 100);
    //log_app.debug() << "clump_ratio=" << clump_ratio << " idxs=" << PrettyVector<size_t>(clump_idxs) << " used=" << PrettyVector<Rect<N,T> >(clumps_used);
    // don't print warnings on heuristic failures since they're not that
    //  uncommon
    bool test_copy = (i == (num_rects - 1));
    if(!test_input(m, input_rects, clumps.size(), clump_ratio - 100,
		   false, test_copy)) {
      log_app.error() << "failure for " << PrettyVector<Rect<N,T> >(input_rects);
      return false;
    }	
  }

  return true;
}

template <int N, typename T>
bool test_dim_and_type(Memory m)
{
  if(!test_directed<N,T>(m)) return false;

  for(int i = 0; i < TestConfig::random_tests; i++) {
    if(!test_random<N,T>(m, 32, 0, 100)) return false;
  }

  return true;
}

template <int N>
bool test_dim(Memory m)
{
  if(((TestConfig::type_mask & 1) != 0) && !test_dim_and_type<N,int>(m))
    return false;
  if(((TestConfig::type_mask & 2) != 0) && !test_dim_and_type<N,long long>(m))
    return false;
  return true;
}

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Realm coverings test";

  Memory m = Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::SYSTEM_MEM).has_affinity_to(p).first();
  assert(m.exists());

  srand48(TestConfig::random_seed);

  bool ok = true;

  if(((TestConfig::dim_mask & 1) != 0) && !test_dim<1>(m))
    ok = false;
  if(((TestConfig::dim_mask & 2) != 0) && !test_dim<2>(m))
    ok = false;
  if(((TestConfig::dim_mask & 4) != 0) && !test_dim<3>(m))
    ok = false;

  if(ok)
    log_app.info() << "coverings test finished successfully";
  else
    log_app.error() << "coverings test finished with errors!";

  // HACK: there's a shutdown race condition related to instance destruction
  usleep(100000);
  
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
