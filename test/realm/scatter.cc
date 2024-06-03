#include "realm.h"
#include "realm/cmdline.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  INDIRECT_PROF_TASK,
  DYNAMIC_TASK_START,
};

enum {
  FID_PTR1 = 100,
  FID_DATA1 = 200,
  FID_DATA2,
  FID_RANGE1 = 300,
  FID_RANGE2,
};

enum {
  SERDEZ_WRAP_FLOAT = 444,
};

template <typename T>
struct WrappingSerdez {
  typedef T FIELD_TYPE;
  static const size_t MAX_SERIALIZED_SIZE = sizeof(T) + sizeof(size_t);

  static size_t serialized_size(const T& val)
  {
    return MAX_SERIALIZED_SIZE;
  }

  static size_t serialize(const T& val, void *buffer)
  {
    size_t size = sizeof(T);
    memcpy(buffer, &size, sizeof(size_t));
    memcpy(static_cast<char *>(buffer)+sizeof(size_t), &val, sizeof(T));
    return MAX_SERIALIZED_SIZE;
  }

  static size_t deserialize(T& val, const void *buffer)
  {
    size_t size;
    memcpy(&size, buffer, sizeof(size_t));
    assert(size == sizeof(T));
    memcpy(&val, static_cast<const char *>(buffer)+sizeof(size_t), sizeof(T));
    return MAX_SERIALIZED_SIZE;
  }

  static void destroy(T& val) {}
};

namespace TestConfig {
  size_t size1 = 10;
  size_t size2 = 8;
  size_t pieces1 = 1;
  size_t pieces2 = 1;
  bool skipfirst = false;
  bool splitcopies = false;
  bool do_gather = true;
  bool do_scatter = true;
};

struct SpeedTestArgs {
  Memory mem;
  RegionInstance inst;
  size_t elements;
  int reps;
  Machine::AffinityDetails affinity;
};

#define MAX_INSTS 32
struct IndirectCopyProfResult {
  UserEvent profile_done_event;
  RegionInstance src_insts[MAX_INSTS];
  RegionInstance dst_insts[MAX_INSTS];
  size_t src_insts_size;
  size_t dst_insts_size;
  FieldID src_fid;
  FieldID dst_fid;
  RegionInstance src_indirection_inst = RegionInstance::NO_INST;
  RegionInstance dst_indirection_inst = RegionInstance::NO_INST;
  FieldID src_indirect_fid = 0;
  FieldID dst_indirect_fid = 0;
  int copy_type; // 0: gather, 1: scatter, 2: range_copy
};

typedef std::map<FieldID, size_t> FieldMap;

// maybe type - used to handle cases where expected value is not known
//  (e.g. due to aliased scatters)
template <typename T>
class Maybe {
public:
  Maybe() : valid(false) {}
  Maybe(T _val) : valid(true), value(_val) {}
  bool has_value() const { return valid; }
  T get_value() const { assert(valid); return value; }
protected:
  bool valid;
  T value;
};

template <int N, typename T>
class DistributedData {
public:
  static const int _N = N;
  typedef T _T;

  ~DistributedData();
  
  void add_subspaces(IndexSpace<N,T> is, size_t count);
  void add_subspaces(const std::vector<IndexSpace<N,T> >& subspaces);
  
  template <typename LAMBDA>
  Event create_instances(const FieldMap& fields, LAMBDA mem_picker);

  void destroy_instances(Event wait_on);

  template <typename FT, typename LAMBDA>
  Event fill(IndexSpace<N,T> is, FieldID fid, LAMBDA filler, Event wait_on);

  template <typename FT, typename SRC>
  Event gather(IndexSpace<N,T> is, FieldID ptr_id, const SRC& src,
	       FieldID src_id, FieldID dst_id,
	       bool oor_possible,
	       CustomSerdezID serdez_id,
	       Event wait_on,
	       Processor p);

  template <typename FT, typename DST>
  Event scatter(IndexSpace<N,T> is, FieldID ptr_id, DST& dst,
		FieldID src_id, FieldID dst_id,
		bool oor_possible, bool aliasing_possible,
		CustomSerdezID serdez_id,
		Event wait_on,
		Processor p) const;

  template <typename FT, typename SRC, typename DST>
  Event range_copy(IndexSpace<N,T> is, FieldID srcptr_id,
		   const SRC& src, FieldID src_id,
		   bool src_oor_possible, bool src_aliasing_possible,
		   const DistributedData<N,T>& dstptr, FieldID dstptr_id,
		   DST& dst, FieldID dst_id,
		   bool dst_oor_possible, bool dst_aliasing_possible,
		   Event wait_on,
		   Processor p) const;

  template <typename FT>
  bool verify(IndexSpace<N,T> is, FieldID fid, Event wait_on);

protected:
  template <int N2, typename T2>
  friend class DistributedData;
  
  struct Piece {
    IndexSpace<N,T> space;
    Processor proc;
    RegionInstance inst, cpu_inst;
  };
  std::vector<Piece> pieces;

  struct RefDataEntry {
    void *data;
    void (*deleter)(void *);
  };
  std::map<FieldID, RefDataEntry> ref_data;
  template <typename FT>
  std::map<Point<N,T>, Maybe<FT> >& get_ref_data(FieldID field_id);
  template <typename FT>
  const std::map<Point<N,T>, Maybe<FT> >& get_ref_data(FieldID field_id) const;
};

template <int N, typename T>
DistributedData<N,T>::~DistributedData()
{
  for(typename std::map<FieldID, RefDataEntry>::const_iterator it = ref_data.begin();
      it != ref_data.end();
      ++it)
    (it->second.deleter)(it->second.data);
  ref_data.clear();
}

template <int N, typename T>
void DistributedData<N,T>::add_subspaces(IndexSpace<N,T> is, size_t count)
{
  if(count == 1) {
    size_t idx = pieces.size();
    pieces.resize(idx + 1);
    pieces[idx].space = is;
    pieces[idx].proc = Processor::NO_PROC;
    pieces[idx].inst = RegionInstance::NO_INST;
   } else {
    std::vector<IndexSpace<N,T> > subspaces;
    is.create_equal_subspaces(count, 1, subspaces, ProfilingRequestSet()).wait();
    add_subspaces(subspaces);
  }
}

template <int N, typename T>
void DistributedData<N,T>::add_subspaces(const std::vector<IndexSpace<N,T> >& subspaces)
{
  size_t base = pieces.size();
  pieces.resize(base + subspaces.size());
  for(size_t i = 0; i < subspaces.size(); i++) {
    pieces[base + i].space = subspaces[i];
    pieces[base + i].proc = Processor::NO_PROC;
    pieces[base + i].inst = RegionInstance::NO_INST;
  }
}

template <int N, typename T>
template <typename LAMBDA>
Event DistributedData<N,T>::create_instances(const FieldMap& fields, LAMBDA mem_picker)
{
  std::vector<Event> events;
  for(size_t i = 0; i < pieces.size(); i++) {
    Memory m = mem_picker(i, pieces[i].space);
    Processor p = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC).has_affinity_to(m).first();

    // if no processor has affinity, at least pick one in same address space
    Memory cpu_mem = Memory::NO_MEMORY;
    if(!p.exists()) {
      p = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC).same_address_space_as(m).first();
      assert(p.exists());
      cpu_mem = Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::SYSTEM_MEM).has_affinity_to(p).has_capacity(1).first();
      assert(cpu_mem.exists());
    }

    pieces[i].proc = p;
    {
      Event e = RegionInstance::create_instance(pieces[i].inst,
                                                m,
                                                pieces[i].space,
                                                fields,
                                                0 /* SOA */,
                                                ProfilingRequestSet());
      events.push_back(e);
    }

    if(cpu_mem.exists()) {
      Event e = RegionInstance::create_instance(pieces[i].cpu_inst,
                                                cpu_mem,
                                                pieces[i].space,
                                                fields,
                                                0 /* SOA */,
                                                ProfilingRequestSet());
      events.push_back(e);
    } else
      pieces[i].cpu_inst = RegionInstance::NO_INST;
  }
  return Event::merge_events(events);
}

template <int N, typename T>
void DistributedData<N,T>::destroy_instances(Event wait_on)
{
  for(typename std::vector<Piece>::iterator it = pieces.begin();
      it != pieces.end();
      ++it) {
    it->inst.destroy(wait_on);
    it->inst = RegionInstance::NO_INST;
    if(it->cpu_inst.exists()) {
      it->cpu_inst.destroy(wait_on);
      it->cpu_inst = RegionInstance::NO_INST;
    }
    it->proc = Processor::NO_PROC;
  }
}

template <typename T>
static void delete_object(void *obj)
{
  delete reinterpret_cast<T *>(obj);
}

template <int N, typename T>
template <typename FT>
std::map<Point<N,T>, Maybe<FT> >& DistributedData<N,T>::get_ref_data(FieldID field_id)
{
  typename std::map<FieldID, RefDataEntry>::const_iterator it = ref_data.find(field_id);
  if(it != ref_data.end()) {
    return *reinterpret_cast<std::map<Point<N,T>, Maybe<FT> > *>(it->second.data);
  } else {
    std::map<Point<N,T>, Maybe<FT> > *newmap = new std::map<Point<N,T>, Maybe<FT> >;
    RefDataEntry &e = ref_data[field_id];
    e.data = reinterpret_cast<void *>(newmap);
    e.deleter = &delete_object<std::map<Point<N,T>, Maybe<FT> > >;
    return *newmap;
  }
}

template <int N, typename T>
template <typename FT>
const std::map<Point<N,T>, Maybe<FT> >& DistributedData<N,T>::get_ref_data(FieldID field_id) const
{
  typename std::map<FieldID, RefDataEntry>::const_iterator it = ref_data.find(field_id);
  assert(it != ref_data.end());
  return *reinterpret_cast<const std::map<Point<N,T>, Maybe<FT> > *>(it->second.data);
}

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
  Event e = Processor::register_task_by_kind(Processor::LOC_PROC,
#ifdef _MSC_VER
    // no portable task registration for windows yet
                                             false /*!global*/,
#else
                                             true /*global*/,
#endif
					     id,
					     CodeDescriptor(&T::task_body),
					     ProfilingRequestSet());
  e.wait();
  task_ids[key] = id;
  return id;
}					     

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
    args.inst = it->cpu_inst.exists() ? it->cpu_inst : it->inst;
    Event e = it->proc.spawn(id, &args, sizeof(args), ProfilingRequestSet(), wait_on);

    // do a copy if we're using a proxy cpu instance
    if(it->cpu_inst.exists()) {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(it->cpu_inst, fid, sizeof(FT));
      dsts[0].set_field(it->inst, fid, sizeof(FT));
      e = isect.copy(srcs, dsts, ProfilingRequestSet(), e);
    }

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

void indirect_prof_task(const void *args, size_t arglen, 
			const void *userdata, size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(IndirectCopyProfResult));
  const IndirectCopyProfResult *result = static_cast<const IndirectCopyProfResult *>(resp.user_data());

  ProfilingMeasurements::OperationCopyInfo copy_info;
  if(resp.get_measurement(copy_info)) {
    assert(result->src_fid == copy_info.inst_info[0].src_fields[0]);
    assert(result->dst_fid == copy_info.inst_info[0].dst_fields[0]);
    assert(result->src_insts_size == copy_info.inst_info[0].src_insts.size());
    for (size_t i = 0; i < result->src_insts_size; i++) {
      assert(result->src_insts[i] == copy_info.inst_info[0].src_insts[i]);
    }
    assert(result->dst_insts_size == copy_info.inst_info[0].dst_insts.size());
    for (size_t i = 0; i < result->dst_insts_size; i++) {
      assert(result->dst_insts[i] == copy_info.inst_info[0].dst_insts[i]);
    }
    assert(result->src_indirection_inst == copy_info.inst_info[0].src_indirection_inst);
    assert(result->dst_indirection_inst == copy_info.inst_info[0].dst_indirection_inst);
    assert(result->src_indirect_fid == copy_info.inst_info[0].src_indirection_field);
    assert(result->dst_indirect_fid == copy_info.inst_info[0].dst_indirection_field);
    log_app.print() << "copy type " << result->copy_type
                    << ", src_insts (" << PrettyVector<RegionInstance>(copy_info.inst_info[0].src_insts) << ") size " << copy_info.inst_info[0].src_insts.size()
                    << ", dst_insts (" << PrettyVector<RegionInstance>(copy_info.inst_info[0].dst_insts) << ") size " << copy_info.inst_info[0].dst_insts.size()
                    << ", src_fid " << copy_info.inst_info[0].src_fields[0]
                    << ", dst_fid " << copy_info.inst_info[0].dst_fields[0]
                    << ", src_indirect_inst " << copy_info.inst_info[0].src_indirection_inst << " fid " << copy_info.inst_info[0].src_indirection_field
                    << ", dst_indirect_inst " << copy_info.inst_info[0].dst_indirection_inst << " fid " << copy_info.inst_info[0].dst_indirection_field;
    result->profile_done_event.trigger();
  }
}

template <int N, typename T>
template <typename FT, typename SRC>
Event DistributedData<N,T>::gather(IndexSpace<N,T> is, FieldID ptr_id, const SRC& src,
				   FieldID src_id, FieldID dst_id,
				   bool oor_possible,
				   CustomSerdezID serdez_id,
				   Event wait_on,
				   Processor p)
{
  std::vector<Event> events;
  for(typename std::vector<Piece>::const_iterator it = pieces.begin();
      it != pieces.end();
      ++it) {
    IndexSpace<N,T> isect;
    IndexSpace<N,T>::compute_intersection(is, it->space, isect, ProfilingRequestSet()).wait();

    typename CopyIndirection<N,T>::template Unstructured<SRC::_N,typename SRC::_T> indirect;
    indirect.field_id = ptr_id;
    indirect.inst = it->inst;
    indirect.is_ranges = false;
    indirect.subfield_offset = 0;
    indirect.oor_possible = oor_possible;
    indirect.aliasing_possible = true; // doesn't matter for gather perf, so be sound
    indirect.next_indirection = nullptr;

    std::vector<CopySrcDstField> srcs, dsts;
    srcs.resize(1);
    dsts.resize(1);
    srcs[0].set_indirect(0, src_id, sizeof(FT));
    dsts[0].set_field(it->inst, dst_id, sizeof(FT));
    if(serdez_id != 0) {
      srcs[0].set_serdez(serdez_id);
      dsts[0].set_serdez(serdez_id);
    }

    if(TestConfig::splitcopies) {
      indirect.spaces.resize(1);
      indirect.insts.resize(1);

      if(src.pieces.size() > 1)
	indirect.oor_possible = true;
      
      for(typename std::vector<typename SRC::Piece>::const_iterator it2 = src.pieces.begin();
	  it2 != src.pieces.end();
	  ++it2) {
	indirect.spaces[0] = it2->space;
	indirect.insts[0] = it2->inst;

	// if we had preimages, we could intersect against those
        IndirectCopyProfResult result;
        UserEvent profile_done_event = UserEvent::create_user_event();
        result.profile_done_event = profile_done_event;
        result.src_insts[0] = indirect.insts[0];
        result.src_insts_size = 1;
        result.dst_insts[0] = it->inst;
        result.dst_insts_size = 1;
        result.src_fid = src_id;
        result.dst_fid = dst_id;
        result.src_indirection_inst = it->inst;
        result.src_indirect_fid = ptr_id;
        result.copy_type = 0;
        ProfilingRequestSet prs;
        prs.add_request(p, INDIRECT_PROF_TASK, &result, sizeof(IndirectCopyProfResult))
             .add_measurement<ProfilingMeasurements::OperationCopyInfo>();
	Event e = isect.copy(srcs, dsts, 
			     std::vector<const typename CopyIndirection<N,T>::Base *>(1, &indirect),
			     prs, wait_on);
	events.push_back(e);
	events.push_back(profile_done_event);
      }
    } else {
      for(typename std::vector<typename SRC::Piece>::const_iterator it2 = src.pieces.begin();
	  it2 != src.pieces.end();
	  ++it2) {
	indirect.spaces.push_back(it2->space);
	indirect.insts.push_back(it2->inst);
      }

      assert(indirect.insts.size() <= MAX_INSTS);
      IndirectCopyProfResult result;
      UserEvent profile_done_event = UserEvent::create_user_event();
      result.profile_done_event = profile_done_event;
      for (size_t i = 0; i < indirect.insts.size(); i++) {
        result.src_insts[i] = indirect.insts[i];
      }
      result.src_insts_size = indirect.insts.size();
      result.dst_insts[0] = it->inst;
      result.dst_insts_size = 1;
      result.src_fid = src_id;
      result.dst_fid = dst_id;
      result.src_indirection_inst = it->inst;
      result.src_indirect_fid = ptr_id;
      result.copy_type = 0;
      ProfilingRequestSet prs;
      prs.add_request(p, INDIRECT_PROF_TASK, &result, sizeof(IndirectCopyProfResult))
           .add_measurement<ProfilingMeasurements::OperationCopyInfo>();
      Event e = isect.copy(srcs, dsts, 
			   std::vector<const typename CopyIndirection<N,T>::Base *>(1, &indirect),
			   prs, wait_on);
      events.push_back(e);
      events.push_back(profile_done_event);
    }
  }

  // update reference data
  const std::map<Point<N,T>, Maybe<Point<SRC::_N,typename SRC::_T> > >& ptrref = get_ref_data<Point<SRC::_N,typename SRC::_T> >(ptr_id);
  const std::map<Point<SRC::_N,typename SRC::_T>, Maybe<FT> >& srcref = src.template get_ref_data<FT>(src_id);
  std::map<Point<N,T>, Maybe<FT> >& dstref = get_ref_data<FT>(dst_id);
  IndexSpaceIterator<N, T> it(is);
  while(it.valid) {
    PointInRectIterator<N,T> pit(it.rect);
    while(pit.valid) {
      Point<SRC::_N,typename SRC::_T> p2 = ptrref.at(pit.p).get_value();
      if(srcref.count(p2) > 0)
	dstref[pit.p] = srcref.at(p2);
      else
	assert(oor_possible);  // make sure we didn't lie to Realm
      pit.step();
    }
    it.step();
  }  

  return Event::merge_events(events);
}

template <int N, typename T>
template <typename FT, typename DST>
Event DistributedData<N,T>::scatter(IndexSpace<N,T> is, FieldID ptr_id, DST& dst,
				    FieldID src_id, FieldID dst_id,
				    bool oor_possible, bool aliasing_possible,
				    CustomSerdezID serdez_id,
				    Event wait_on,
				    Processor p) const
{
  std::vector<Event> events;
  for(typename std::vector<Piece>::const_iterator it = pieces.begin();
      it != pieces.end();
      ++it) {
    IndexSpace<N,T> isect;
    IndexSpace<N,T>::compute_intersection(is, it->space, isect, ProfilingRequestSet()).wait();

    typename CopyIndirection<N,T>::template Unstructured<DST::_N,typename DST::_T> indirect;
    indirect.field_id = ptr_id;
    indirect.inst = it->inst;
    indirect.is_ranges = false;
    indirect.subfield_offset = 0;
    indirect.oor_possible = oor_possible;
    indirect.aliasing_possible = aliasing_possible;
    indirect.next_indirection = nullptr;

    std::vector<CopySrcDstField> srcs, dsts;
    srcs.resize(1);
    dsts.resize(1);
    srcs[0].set_field(it->inst, src_id, sizeof(FT));
    dsts[0].set_indirect(0, dst_id, sizeof(FT));
    if(serdez_id != 0) {
      srcs[0].set_serdez(serdez_id);
      dsts[0].set_serdez(serdez_id);
    }

    if(TestConfig::splitcopies) {
      indirect.spaces.resize(1);
      indirect.insts.resize(1);

      if(dst.pieces.size() > 1)
	indirect.oor_possible = true;
      
      for(typename std::vector<typename DST::Piece>::const_iterator it2 = dst.pieces.begin();
	  it2 != dst.pieces.end();
	  ++it2) {
	indirect.spaces[0] = it2->space;
	indirect.insts[0] = it2->inst;

	// if we had preimages, we could intersect against those
	IndirectCopyProfResult result;
        UserEvent profile_done_event = UserEvent::create_user_event();
        result.profile_done_event = profile_done_event;
        result.src_insts[0] = it->inst;
        result.src_insts_size = 1;
        result.dst_insts[0] = indirect.insts[0];
        result.dst_insts_size = 1;
        result.src_fid = src_id;
        result.dst_fid = dst_id;
        result.dst_indirection_inst = it->inst;
        result.dst_indirect_fid = ptr_id;
        result.copy_type = 0;
        ProfilingRequestSet prs;
        prs.add_request(p, INDIRECT_PROF_TASK, &result, sizeof(IndirectCopyProfResult))
             .add_measurement<ProfilingMeasurements::OperationCopyInfo>();
	Event e = isect.copy(srcs, dsts, 
			     std::vector<const typename CopyIndirection<N,T>::Base *>(1, &indirect),
			     prs, wait_on);
	events.push_back(e);
	events.push_back(profile_done_event);
      }
    } else {
      for(typename std::vector<typename DST::Piece>::const_iterator it2 = dst.pieces.begin();
	  it2 != dst.pieces.end();
	  ++it2) {
	indirect.spaces.push_back(it2->space);
	indirect.insts.push_back(it2->inst);
      }

      assert(indirect.insts.size() <= MAX_INSTS);
      IndirectCopyProfResult result;
      UserEvent profile_done_event = UserEvent::create_user_event();
      result.profile_done_event = profile_done_event;
      for (size_t i = 0; i < indirect.insts.size(); i++) {
        result.dst_insts[i] = indirect.insts[i];
      }
      result.dst_insts_size = indirect.insts.size();
      result.src_insts[0] = it->inst;
      result.src_insts_size = 1;
      result.src_fid = src_id;
      result.dst_fid = dst_id;
      result.dst_indirection_inst = it->inst;
      result.dst_indirect_fid = ptr_id;
      result.copy_type = 1;
      ProfilingRequestSet prs;
      prs.add_request(p, INDIRECT_PROF_TASK, &result, sizeof(IndirectCopyProfResult))
           .add_measurement<ProfilingMeasurements::OperationCopyInfo>();

      Event e = isect.copy(srcs, dsts, 
			   std::vector<const typename CopyIndirection<N,T>::Base *>(1, &indirect),
			   prs, wait_on);
      events.push_back(e);
      events.push_back(profile_done_event);
    }
  }

  // update reference data
  const std::map<Point<N,T>, Maybe<Point<DST::_N,typename DST::_T> > >& ptrref = get_ref_data<Point<DST::_N,typename DST::_T> >(ptr_id);
  const std::map<Point<N,T>, Maybe<FT> >& srcref = get_ref_data<FT>(src_id);
  std::map<Point<DST::_N,typename DST::_T>, Maybe<FT> >& dstref = dst.template get_ref_data<FT>(dst_id);
  std::set<Point<DST::_N,typename DST::_T> > touched;  // to detect aliasing
  IndexSpaceIterator<N, T> it(is);
  while(it.valid) {
    PointInRectIterator<N,T> pit(it.rect);
    while(pit.valid) {
      Point<DST::_N,typename DST::_T> p2 = ptrref.at(pit.p).get_value();
      if(dstref.count(p2) > 0) {
	if(touched.count(p2) > 0) {
	  assert(aliasing_possible);
	  dstref[p2] = Maybe<FT>();
	} else {
	  dstref[p2] = srcref.at(pit.p);
	  touched.insert(p2);
	}
      } else
	assert(oor_possible);  // make sure we didn't lie to Realm
      pit.step();
    }
    it.step();
  }  

  return Event::merge_events(events);
}

template <int N, typename T>
template <typename FT, typename SRC, typename DST>
Event DistributedData<N,T>::range_copy(IndexSpace<N,T> is, FieldID srcptr_id,
				       const SRC& src, FieldID src_id,
				       bool src_oor_possible, bool src_aliasing_possible,
				       const DistributedData<N,T>& dstptr, FieldID dstptr_id,
				       DST& dst, FieldID dst_id,
				       bool dst_oor_possible, bool dst_aliasing_possible,
				       Event wait_on,
				       Processor p) const
{
  std::vector<Event> events;
  for(typename std::vector<Piece>::const_iterator it = pieces.begin();
      it != pieces.end();
      ++it) {
    IndexSpace<N,T> isect;
    IndexSpace<N,T>::compute_intersection(is, it->space, isect, ProfilingRequestSet()).wait();

    for(typename std::vector<Piece>::const_iterator it2 = dstptr.pieces.begin();
	it2 != dstptr.pieces.end();
	++it2) {
      IndexSpace<N,T> isect2;
      IndexSpace<N,T>::compute_intersection(isect, it2->space, isect2,
					    ProfilingRequestSet()).wait();
      isect2.make_valid().wait();
      isect2 = isect2.tighten();
      if(isect2.empty()) continue;

      typename CopyIndirection<N,T>::template Unstructured<SRC::_N,typename SRC::_T> src_ind;
      src_ind.field_id = srcptr_id;
      src_ind.inst = it->inst;
      src_ind.is_ranges = true;
      src_ind.subfield_offset = 0;
      src_ind.oor_possible = src_oor_possible;
      src_ind.aliasing_possible = src_aliasing_possible;
      src_ind.next_indirection = nullptr;

      typename CopyIndirection<N,T>::template Unstructured<DST::_N,typename DST::_T> dst_ind;
      dst_ind.field_id = dstptr_id;
      dst_ind.inst = it2->inst;
      dst_ind.is_ranges = true;
      dst_ind.subfield_offset = 0;
      dst_ind.oor_possible = dst_oor_possible;
      dst_ind.aliasing_possible = dst_aliasing_possible;
      dst_ind.next_indirection = nullptr;

      std::vector<CopySrcDstField> srcs, dsts;
      srcs.resize(1);
      dsts.resize(1);
      srcs[0].set_indirect(0, src_id, sizeof(FT));
      dsts[0].set_indirect(1, dst_id, sizeof(FT));

      std::vector<const typename CopyIndirection<N,T>::Base *> indirects;
      indirects.push_back(&src_ind);
      indirects.push_back(&dst_ind);

      // TODO: splitcopies could get really expensive for double-indirect?
      for(typename std::vector<typename SRC::Piece>::const_iterator it3 = src.pieces.begin();
	  it3 != src.pieces.end();
	  ++it3) {
	src_ind.spaces.push_back(it3->space);
	src_ind.insts.push_back(it3->inst);
      }

      for(typename std::vector<typename DST::Piece>::const_iterator it3 = dst.pieces.begin();
	  it3 != dst.pieces.end();
	  ++it3) {
	dst_ind.spaces.push_back(it3->space);
	dst_ind.insts.push_back(it3->inst);
      }

      assert(src_ind.insts.size() <= MAX_INSTS && dst_ind.insts.size() <= MAX_INSTS);
      IndirectCopyProfResult result;
      UserEvent profile_done_event = UserEvent::create_user_event();
      result.profile_done_event = profile_done_event;
      for (size_t i = 0; i < src_ind.insts.size(); i++) {
        result.src_insts[i] = src_ind.insts[i];
      }
      result.src_insts_size = src_ind.insts.size();
      for (size_t i = 0; i < dst_ind.insts.size(); i++) {
        result.dst_insts[i] = dst_ind.insts[i];
      }
      result.dst_insts_size = dst_ind.insts.size();
      result.src_fid = src_id;
      result.dst_fid = dst_id;
      result.src_indirection_inst = it->inst;
      result.dst_indirection_inst = it2->inst;
      result.src_indirect_fid = srcptr_id;
      result.dst_indirect_fid = dstptr_id;
      result.copy_type = 2;
      ProfilingRequestSet prs;
      prs.add_request(p, INDIRECT_PROF_TASK, &result, sizeof(IndirectCopyProfResult))
           .add_measurement<ProfilingMeasurements::OperationCopyInfo>();

      Event e = isect2.copy(srcs, dsts, indirects,
			    prs, wait_on);
      events.push_back(e);
      events.push_back(profile_done_event);
    }
  }

  // update reference data
  const std::map<Point<N,T>, Maybe<Rect<SRC::_N,typename SRC::_T> > >& srcptrref = get_ref_data<Rect<SRC::_N,typename SRC::_T> >(srcptr_id);
  const std::map<Point<N,T>, Maybe<Rect<DST::_N,typename DST::_T> > >& dstptrref = dstptr.template get_ref_data<Rect<DST::_N,typename DST::_T> >(dstptr_id);
  const std::map<Point<SRC::_N,typename SRC::_T>, Maybe<FT> >& srcref = src.template get_ref_data<FT>(src_id);
  std::map<Point<DST::_N,typename DST::_T>, Maybe<FT> >& dstref = dst.template get_ref_data<FT>(dst_id);
  std::set<Point<DST::_N,typename DST::_T> > touched;  // to detect aliasing
  IndexSpaceIterator<N, T> it(is);
  while(it.valid) {
    PointInRectIterator<N,T> pit(it.rect);
    while(pit.valid) {
      Rect<SRC::_N,typename SRC::_T> srect = srcptrref.at(pit.p).get_value();
      Rect<DST::_N,typename DST::_T> drect = dstptrref.at(pit.p).get_value();
      PointInRectIterator<SRC::_N,typename SRC::_T> sit(srect);
      PointInRectIterator<DST::_N,typename DST::_T> dit(drect);
      while(sit.valid && dit.valid) {
	if(srcref.count(sit.p) > 0) {
	  if(dstref.count(dit.p) > 0) {
	    if(touched.count(dit.p) > 0) {
	      assert(dst_aliasing_possible);
	      dstref[dit.p] = Maybe<FT>();
	    } else {
	      dstref[dit.p] = srcref.at(sit.p);
	      touched.insert(dit.p);
	    }
	  } else {
	    assert(dst_oor_possible);
	    assert(0);
	  }
	} else {
	  assert(src_oor_possible);
	  assert(0);
	}
	sit.step();
	dit.step();
      }
      assert(!sit.valid && !dit.valid);
      pit.step();
    }
    it.step();
  }  

  return Event::merge_events(events);
}

template <int N, typename T>
template <typename FT>
bool DistributedData<N,T>::verify(IndexSpace<N,T> is, FieldID fid, Event wait_on)
{
  wait_on.wait();

  const std::map<Point<N,T>, Maybe<FT> >& ref = get_ref_data<FT>(fid);
  
  int errors = 0;
  for(typename std::vector<Piece>::iterator it = pieces.begin();
      it != pieces.end();
      ++it) {
    IndexSpace<N,T> isect;
    IndexSpace<N,T>::compute_intersection(is, it->space, isect, ProfilingRequestSet()).wait();

    AffineAccessor<FT,N,T> acc;
    RegionInstance tmp_inst = RegionInstance::NO_INST;
    if(Machine::get_machine().has_affinity(Processor::get_executing_processor(),
                                           it->inst.get_location())) {
      // good, access this instance directly
      acc.reset(it->inst, fid);
    } else {
      // need to make a temporary instance in a memory we can access
      Memory m = Machine::MemoryQuery(Machine::get_machine()).has_affinity_to(Processor::get_executing_processor()).has_capacity(1).first(); // TODO: best!
      assert(m.exists());
      std::map<FieldID, size_t> tmp_fields;
      tmp_fields[fid] = sizeof(FT);
      RegionInstance::create_instance(tmp_inst, m, isect,
				      tmp_fields, 0 /*SOA*/,
				      ProfilingRequestSet()).wait();
      std::vector<CopySrcDstField> srcs, dsts;
      srcs.resize(1);
      dsts.resize(1);
      srcs[0].set_field(it->inst, fid, sizeof(FT));
      dsts[0].set_field(tmp_inst, fid, sizeof(FT));
      isect.copy(srcs, dsts, ProfilingRequestSet()).wait();
      acc.reset(tmp_inst, fid);
    }

    IndexSpaceIterator<N, T> iit(isect);
    while(iit.valid) {
      PointInRectIterator<N,T> pit(iit.rect);
      while(pit.valid) {
	Maybe<FT> exp = ref.at(pit.p);
	FT act = acc[pit.p];
	if(exp.has_value()) {
	  if(exp.get_value() == act) {
	    // good
	    log_app.debug() << "  match at [" << pit.p << "]: exp=" << exp.get_value() << " act=" << act;
	  } else {
	    if(errors++ < 10)
	      log_app.error() << "  mismatch at [" << pit.p << "]: exp=" << exp.get_value() << " act=" << act;
	  }
	} else {
	  log_app.debug() << "  cannot check at [" << pit.p << "]: exp=??? act=" << act;
	}
	pit.step();
      }
      iit.step();
    }

    if(tmp_inst.exists())
      tmp_inst.destroy();
  }

  return (errors == 0);
}

template <int N, typename T>
class RoundRobinPicker {
public:
  RoundRobinPicker(const std::vector<Memory>& _mems, bool _reverse = false)
    : mems(_mems), reverse(_reverse) {}
  Memory operator()(size_t i, IndexSpace<N,T> is)
  {
    if(reverse)
      return mems[mems.size() - 1 - (i % mems.size())];
    else
      return mems[i % mems.size()];
  }
protected:
  const std::vector<Memory>& mems;
  bool reverse;
};

template <int N, typename T, typename DT>
void dump_field(RegionInstance inst, FieldID fid, IndexSpace<N,T> is)
{
  AffineAccessor<DT, N, T> acc(inst, fid);
  for(IndexSpaceIterator<N,T> it(is); it.valid; it.step())
    for(PointInRectIterator<N,T> it2(it.rect); it2.valid; it2.step()) {
      DT v = acc[it2.p];
      std::cout << it2.p << ": " << v << "\n";
    }
}

template <typename T, size_t BYTES>
struct Pad {
  T val;
  char padding[BYTES - sizeof(T)];
  Pad() {}
  Pad(T _val) : val(_val) {}
  operator T() const { return val; }
};

template <int N, typename T, int N2, typename T2, typename DT>
bool scatter_gather_test(const std::vector<Memory> &mems, T size1, T2 size2, int pieces1,
                         int pieces2, Processor p, CustomSerdezID serdez_id = 0,
                         bool oor_possible = false)
{
  Rect<N,T> r1;
  Rect<N2,T2> r2;
  for(int i = 0; i < N; i++) r1.lo[i] = 0;
  for(int i = 0; i < N; i++) r1.hi[i] = size1 - 1;
  for(int i = 0; i < N2; i++) r2.lo[i] = 0;
  for(int i = 0; i < N2; i++) r2.hi[i] = size2 - 1;
  IndexSpace<N,T> is1(r1);
  IndexSpace<N2,T2> is2(r2);

  FieldMap fields1;
  fields1[FID_PTR1] = sizeof(Point<N2,T2>);
  fields1[FID_DATA1] = sizeof(DT);
  fields1[FID_DATA2] = sizeof(DT);

  std::map<FieldID, size_t> fields2;
  fields2[FID_DATA1] = sizeof(DT);
  fields2[FID_DATA2] = sizeof(DT);

  DistributedData<N,T> region1;
  region1.add_subspaces(is1, pieces1);
  region1.create_instances(fields1, RoundRobinPicker<N,T>(mems)).wait();

  DistributedData<N2,T2> region2;
  region2.add_subspaces(is2, pieces2);
  region2.create_instances(fields2, RoundRobinPicker<N2,T2>(mems)).wait();

  region1.template fill<DT>(is1, FID_DATA1, [](Point<N,T> p) -> DT { return DT(p.x); },
			    Event::NO_EVENT).wait();
  region1.template fill<DT>(is1, FID_DATA2, [](Point<N,T> p) -> DT { return DT(p.x + 100); },
			    Event::NO_EVENT).wait();
  
  region2.template fill<DT>(is2, FID_DATA1, [](Point<N2,T2> p) -> DT { return DT(200 + p.x + 10*p.y); },
			    Event::NO_EVENT).wait();
  region2.template fill<DT>(is2, FID_DATA2, [](Point<N2,T2> p) -> DT { return DT(300 + p.x + 10*p.y); },
			    Event::NO_EVENT).wait();

  region1.template fill<Point<N2,T2> >(is1, FID_PTR1, [=](Point<N,T> p) -> Point<N2,T2> { return Point<N2,T2>(p.x % size2); },
				       Event::NO_EVENT).wait();

  if(TestConfig::do_gather) {
    region1
        .template gather<DT>(is1, FID_PTR1, region2, FID_DATA1, FID_DATA1,
                             oor_possible /*!oor_possible*/, serdez_id, Event::NO_EVENT,
                             p)
        .wait();

    if(!region1.template verify<DT>(is1, FID_DATA1, Event::NO_EVENT))
      return false;
  }

  if(TestConfig::do_scatter) {
    region1
        .template scatter<DT>(is1, FID_PTR1, region2, FID_DATA2, FID_DATA2,
                              oor_possible /*!oor_possible*/, true /*aliasing_possible*/,
                              serdez_id, Event::NO_EVENT, p)
        .wait();

    if(!region2.template verify<DT>(is2, FID_DATA2, Event::NO_EVENT))
      return false;
  }

  region1.destroy_instances(Event::NO_EVENT);
  region2.destroy_instances(Event::NO_EVENT);

  return true;
#if 0
  Rect<N,T> r1;
  Rect<N2,T2> r2;
  for(int i = 0; i < N; i++) r1.lo[i] = 0;
  for(int i = 0; i < N; i++) r1.hi[i] = size1 - 1;
  for(int i = 0; i < N2; i++) r2.lo[i] = 0;
  for(int i = 0; i < N2; i++) r2.hi[i] = size2 - 1;
  IndexSpace<N,T> is1(r1);
  IndexSpace<N2,T2> is2(r2);

  RegionInstance inst1, inst2a, inst2b;

  std::map<FieldID, size_t> fields1;
  fields1[FID_PTR1] = sizeof(Point<N2,T2>);
  fields1[FID_DATA1] = sizeof(DT);
  fields1[FID_DATA2] = sizeof(DT);
  RegionInstance::create_instance(inst1, m, is1, fields1,
				  0 /*SOA*/, ProfilingRequestSet()).wait();

  std::map<FieldID, size_t> fields2;
  fields2[FID_DATA1] = sizeof(DT);
  fields2[FID_DATA2] = sizeof(DT);
  RegionInstance::create_instance(inst2a, m, is2, fields2,
				  0 /*SOA*/, ProfilingRequestSet()).wait();
  RegionInstance::create_instance(inst2b, m, is2, fields2,
				  0 /*SOA*/, ProfilingRequestSet()).wait();

  // fill the new instance
  {
    AffineAccessor<Point<N2, T2>, N, T> acc_ptr1(inst1, FID_PTR1);
    AffineAccessor<DT, N, T> acc_data1(inst1, FID_DATA1);

    DT count = 0;
    IndexSpaceIterator<N, T> it1(is1);
    IndexSpaceIterator<N2, T2> it2(is2);
    assert(it1.valid);
    assert(it2.valid);
    PointInRectIterator<N, T> pit1(it1.rect);
    PointInRectIterator<N2, T2> pit2(it2.rect);
    while(true) {
      acc_ptr1[pit1.p] = pit2.p;
      acc_data1[pit1.p] = count;
      count += 1;
      if(!pit1.step()) {
	if(!it1.step()) break;
	pit1.reset(it1.rect);
      }
      if(!pit2.step()) {
	if(!it2.step()) it2.reset(is2);
	pit2.reset(it2.rect);
      }
    }
  }

  dump_field<N, T, Point<N2, T2> >(inst1, FID_PTR1, is1);
  dump_field<N, T, DT >(inst1, FID_DATA1, is1);

  // affine copy within inst1
  if(0) {
    Matrix<N, N, T> xform;
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++)
	xform.rows[i][j] = (i == j) ? -1 : 0;
    Point<N, T> offset(r1.hi);
    typename CopyIndirection<N,T>::template Affine<N,T> indirect;
    indirect.transform = xform;
    indirect.offset_lo = offset;
    indirect.offset_hi = offset;
    for(int i = 0; i < N; i++) indirect.divisor[i] = 1;
    indirect.spaces.push_back(is2);
    indirect.insts.push_back(inst2a);

    std::vector<CopySrcDstField> srcs, dsts;
    srcs.resize(1);
    dsts.resize(1);
#ifdef ACTUALLY_TEST_GATHER
    srcs[0].set_indirect(0, FID_DATA1, sizeof(DT));
#else
    srcs[0].set_field(inst1, FID_DATA1, sizeof(DT));
#endif
    //srcs[0].template set_fill<DT>(2.5);
    dsts[0].set_field(inst1, FID_DATA2, sizeof(DT));

    is1.copy(srcs, dsts, 
	     std::vector<const typename CopyIndirection<N,T>::Base *>(1, &indirect),
	     ProfilingRequestSet()).wait();
    
    dump_field<N, T, DT >(inst1, FID_DATA2, is1);
  }

  // fill the new instance
  {
    AffineAccessor<DT, N2, T2> acc_data1(inst2a, FID_DATA1);

    DT count = 100;
    IndexSpaceIterator<N2, T2> it2(is2);
    assert(it2.valid);
    PointInRectIterator<N2, T2> pit2(it2.rect);
    while(true) {
      acc_data1[pit2.p] = count;
      count += 1;
      if(!pit2.step()) {
	if(!it2.step()) break;
	pit2.reset(it2.rect);
      }
    }

    dump_field<N2, T2, DT>(inst2a, FID_DATA1, is2);
  }

  // indirect gather from inst2 to inst1
  if(1) {
    typename CopyIndirection<N,T>::template Unstructured<N2,T2> indirect;
    indirect.field_id = FID_PTR1;
    indirect.inst = inst1;
    indirect.is_ranges = false;
    indirect.subfield_offset = 0;
    indirect.spaces.push_back(is2);
    indirect.insts.push_back(inst2a);

    std::vector<CopySrcDstField> srcs, dsts;
    srcs.resize(1);
    dsts.resize(1);
    srcs[0].set_indirect(0, FID_DATA1, sizeof(DT));
    dsts[0].set_field(inst1, FID_DATA2, sizeof(DT));

    is1.copy(srcs, dsts, 
	     std::vector<const typename CopyIndirection<N,T>::Base *>(1, &indirect),
	     ProfilingRequestSet()).wait();
    
    dump_field<N, T, DT >(inst1, FID_DATA2, is1);
  }

  inst1.destroy();
  inst2a.destroy();
  inst2b.destroy();
  return true;
#endif
}

template <typename DT>
class ConstantFiller {
public:
  ConstantFiller(DT _cval)
    : cval(_cval)
  {}

  template <int N, typename T>
  DT operator()(Point<N,T> p) const { return cval; }

protected:
  DT cval;
};

template <typename DT>
class RegularFiller {
public:
  RegularFiller(DT _base, DT _step0 = 1, DT _step1 = 10, DT _step2 = 100)
    : base(_base), step0(_step0), step1(_step1), step2(_step2)
  {}

  template <typename T>
  DT operator()(Point<1,T> p) const { return base + step0 * p.x; }

  template <typename T>
  DT operator()(Point<2,T> p) const { return base + step0 * p.x + step1 * p.y; }

  template <typename T>
  DT operator()(Point<3,T> p) const { return base + step0 * p.x + step1 * p.y + step2 * p.z; }

protected:
  DT base, step0, step1, step2;
};

template <int N, typename T, int N2, typename T2, int N3, typename T3, typename DT>
bool range_copy_test(const std::vector<Memory>& mems,
		     T size1, T2 size2, T3 size3,
		     int pieces1, int pieces2, int pieces3,
		     Processor p)
{
  Rect<N,T> r1;
  Rect<N2,T2> r2;
  Rect<N3,T3> r3;
  for(int i = 0; i < N; i++) r1.lo[i] = 0;
  for(int i = 0; i < N; i++) r1.hi[i] = size1 - 1;
  for(int i = 0; i < N2; i++) r2.lo[i] = 0;
  for(int i = 0; i < N2; i++) r2.hi[i] = size2 - 1;
  for(int i = 0; i < N3; i++) r3.lo[i] = 0;
  for(int i = 0; i < N3; i++) r3.hi[i] = size3 - 1;
  IndexSpace<N,T> is1(r1);
  IndexSpace<N2,T2> is2(r2);
  IndexSpace<N3,T3> is3(r3);

  FieldMap fields1;
  fields1[FID_RANGE1] = sizeof(Rect<N2,T2>);
  fields1[FID_RANGE2] = sizeof(Rect<N3,T3>);

  std::map<FieldID, size_t> fields2;
  fields2[FID_DATA1] = sizeof(DT);
  fields2[FID_DATA2] = sizeof(DT);

  std::map<FieldID, size_t> fields3;
  fields3[FID_DATA1] = sizeof(DT);
  fields3[FID_DATA2] = sizeof(DT);

  DistributedData<N,T> region1;
  region1.add_subspaces(is1, pieces1);
  region1.create_instances(fields1, RoundRobinPicker<N,T>(mems)).wait();

  DistributedData<N2,T2> region2;
  region2.add_subspaces(is2, pieces2);
  region2.create_instances(fields2, RoundRobinPicker<N2,T2>(mems)).wait();

  DistributedData<N3,T3> region3;
  region3.add_subspaces(is3, pieces3);
  region3.create_instances(fields3, RoundRobinPicker<N3,T3>(mems, true)).wait();

  region1.template fill<Rect<N2,T2> >(is1, FID_RANGE1, [=](Point<N,T> p) -> Rect<N2,T2> { return Rect<N2,T2>::make_empty(); }, Event::NO_EVENT).wait();
  //region1.template fill<Rect<N3,T3> >(is1, FID_RANGE2, [=](Point<N,T> p) -> Rect<N3,T3> { return Rect<N3,T3>::make_empty(); }, Event::NO_EVENT).wait();
  region1.template fill<Rect<N3,T3> >(is1, FID_RANGE2,
				      ConstantFiller<Rect<N3,T3> >(Rect<N3,T3>::make_empty()),
				      Event::NO_EVENT).wait();
  region1.template fill<Rect<N2,T2> >(Rect<N,T>(Point<N,T>(1),
						Point<N,T>(1)), FID_RANGE1,
				      ConstantFiller<Rect<N2,T2> >(Rect<N2,T2>(Point<N2,T2>(1,0),
									       Point<N2,T2>(1,1))),
				      Event::NO_EVENT).wait();
  region1.template fill<Rect<N2,T2> >(Rect<N,T>(Point<N,T>(3),
						Point<N,T>(3)), FID_RANGE1,
				      ConstantFiller<Rect<N2,T2> >(Rect<N2,T2>(Point<N2,T2>(1,2),
									       Point<N2,T2>(1,3))),
				      Event::NO_EVENT).wait();
  region1.template fill<Rect<N3,T3> >(Rect<N,T>(Point<N,T>(1),
						Point<N,T>(1)), FID_RANGE2,
				      ConstantFiller<Rect<N3,T3> >(Rect<N3,T3>(Point<N3,T3>(1,1,1),
									       Point<N3,T3>(1,2,1))),
				      Event::NO_EVENT).wait();
  region1.template fill<Rect<N3,T3> >(Rect<N,T>(Point<N,T>(3),
						Point<N,T>(3)), FID_RANGE2,
				      ConstantFiller<Rect<N3,T3> >(Rect<N3,T3>(Point<N3,T3>(2,1,1),
									       Point<N3,T3>(2,2,1))),
				      Event::NO_EVENT).wait();

  region2.template fill<DT>(is2, FID_DATA1, RegularFiller<DT>(1000),
    Event::NO_EVENT).wait();

  region3.template fill<DT>(is3, FID_DATA1, RegularFiller<DT>(2000),
    Event::NO_EVENT).wait();
#if 0
  region1.template fill<DT>(is1, FID_DATA1, [](Point<N,T> p) -> DT { return DT(p.x); },
			    Event::NO_EVENT).wait();
  region1.template fill<DT>(is1, FID_DATA2, [](Point<N,T> p) -> DT { return DT(p.x + 100); },
			    Event::NO_EVENT).wait();
  
  region2.template fill<DT>(is2, FID_DATA1, [](Point<N2,T2> p) -> DT { return DT(200 + p.x + 10*p.y); },
			    Event::NO_EVENT).wait();
  region2.template fill<DT>(is2, FID_DATA2, [](Point<N2,T2> p) -> DT { return DT(300 + p.x + 10*p.y); },
			    Event::NO_EVENT).wait();

  region1.template fill<Point<N2,T2> >(is1, FID_PTR1, [=](Point<N,T> p) -> Point<N2,T2> { return Point<N2,T2>(p.x % size2); },
				       Event::NO_EVENT).wait();
#endif

  region1.template range_copy<DT>(is1, FID_RANGE1, region2, FID_DATA1,
    false /*!src_oor_possible*/, true /*src_aliasing_possible*/,
    region1, FID_RANGE2, region3, FID_DATA1,
    false /*!dst_oor_possible*/, true /*dst_aliasing_possible*/,
    Event::NO_EVENT, p).wait();

  if(!region3.template verify<DT>(is3, FID_DATA1, Event::NO_EVENT))
    return false;

  return true;
}

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Realm scatter/gather test";

  std::vector<Memory> mems;
  bool do_serdez = false;

  // first try: use fb memories, if available
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::GPU_FB_MEM).has_capacity(1);
  mems.assign(mq.begin(), mq.end());

  // second try: system memories
  if(mems.empty()) {
    Machine::MemoryQuery mq(Machine::get_machine());
    mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
    mems.assign(mq.begin(), mq.end());
    assert(!mems.empty());
    do_serdez = true;
  }

  if(TestConfig::skipfirst && (mems.size() > 1))
    mems.erase(mems.begin());

  bool ok = true;

  // normal-sized data
  if(!scatter_gather_test<1, int, 2, int, float>(
         mems, TestConfig::size1, TestConfig::size2, TestConfig::pieces1,
         TestConfig::pieces2, p, /*serdez_id=*/0, /*oor_possible=*/true))
    ok = false;

  // normal-sized data
  if(!scatter_gather_test<1, int, 2, int, float>(mems,
						 TestConfig::size1,
						 TestConfig::size2,
						 TestConfig::pieces1,
						 TestConfig::pieces2,
						 p))
    ok = false;

  // really big (non-power-of-2) fields
  typedef Pad<float, 2000> BigFloat;
  if(!scatter_gather_test<1, int, 2, int, BigFloat>(mems,
						    TestConfig::size1,
						    TestConfig::size2,
						    TestConfig::pieces1,
						    TestConfig::pieces2,
						    p))
    ok = false;

  // serdez
  if(do_serdez &&
     !scatter_gather_test<1, int, 2, int, float>(mems,
						 TestConfig::size1,
						 TestConfig::size2,
						 TestConfig::pieces1,
						 TestConfig::pieces2,
						 p,
						 SERDEZ_WRAP_FLOAT))
    ok = false;

  if(!range_copy_test<1, int, 2, int, 3, int, float>(mems,
						     4, 4, 4,
						     1,
						     1 /*TestConfig::pieces1*/,
						     1 /*TestConfig::pieces2*/,
						     p))
    ok = false;

  if(ok)
    log_app.info() << "scatter/gather test finished successfully";
  else
    log_app.error() << "scatter/gather test finished with errors!";

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
  cp.add_option_int("-s1", TestConfig::size1);
  cp.add_option_int("-s2", TestConfig::size2);
  cp.add_option_int("-p1", TestConfig::pieces1);
  cp.add_option_int("-p2", TestConfig::pieces2);
  cp.add_option_bool("-skipfirst", TestConfig::skipfirst);
  cp.add_option_bool("-split", TestConfig::splitcopies);
  cp.add_option_int("-gather", TestConfig::do_gather);
  cp.add_option_int("-scatter", TestConfig::do_scatter);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
  
#if 0
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-b")) {
      buffer_size = strtoll(argv[++i], 0, 10);
      continue;
    }

  }
#endif

  rt.register_custom_serdez<WrappingSerdez<float> >(SERDEZ_WRAP_FLOAT);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
				   INDIRECT_PROF_TASK,
				   CodeDescriptor(indirect_prof_task),
				   ProfilingRequestSet(),
				   0, 0).wait();

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
