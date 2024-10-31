/* Copyright 2024 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace PRealm {

  class ThreadProfiler {
  public:
    ThreadProfiler(Processor p);
    ThreadProfiler(const ThreadProfiler &rhs) = delete;
    ThreadProfiler& operator=(const ThreadProfiler &rhs) = delete;
  public:
    void add_fill_request(ProfilingRequestSet &requests,
        const std::vector<CopySrcDstField> &dsts, Event critical);
    void add_copy_request(ProfilingRequestSet &requests,
        const std::vector<CopySrcDstField> &srcs,
        const std::vector<CopySrcDstField> &dsts, Event critical);
    void add_task_request(ProfilingRequestSet &requests,
        Processor target, Event critical);
    Event add_inst_request(ProfilingRequestSet &requests,
        const InstanceLayoutGeneric *ilg, Event critical);
  public:
    void record_event_wait(Event wait_on, Backtrace &bt);
    void record_event_trigger(Event result, Event precondition);
    void record_event_poison(Event result);
    void record_barrier_arrival(Event result, Event precondition);
    void record_event_merger(Event result, const Event *preconditions, size_t num_events);
    void record_reservation_acquire(Reservation r, Event result, Event precondition);
    void record_instance_usage(RegionInstance inst, FieldID field_id);
  
    static ThreadProfiler& get_thread_profiler(void);
  private:
    const Processor local_proc;
  };

  inline CopySrcDstField::CopySrcDstField(void)
    : inst(RegionInstance::NO_INST)
    , field_id(FieldID(-1))
    , size(0)
    , redop_id(0)
    , red_fold(false)
    , red_exclusive(false)
    , serdez_id(0)
    , subfield_offset(0)
    , indirect_index(-1)
  {
    fill_data.indirect = 0;
  }

  inline CopySrcDstField::CopySrcDstField(const CopySrcDstField& copy_from)
    : inst(copy_from.inst)
    , field_id(copy_from.field_id)
    , size(copy_from.size)
    , redop_id(copy_from.redop_id)
    , red_fold(copy_from.red_fold)
    , red_exclusive(copy_from.red_exclusive)
    , serdez_id(copy_from.serdez_id)
    , subfield_offset(copy_from.subfield_offset)
    , indirect_index(copy_from.indirect_index)
  {
    // we know there's a fill value if the field ID is -1
    if(copy_from.field_id == FieldID(-1)) {
      if(size <= MAX_DIRECT_SIZE) {
	// copy whole buffer to make sure indirect is initialized too
	memcpy(fill_data.direct, copy_from.fill_data.direct, MAX_DIRECT_SIZE);
      } else {
	if(copy_from.fill_data.indirect) {
	  fill_data.indirect = malloc(size);
	  memcpy(fill_data.indirect, copy_from.fill_data.indirect, size);
	} else
	  fill_data.indirect = 0;
      }
    } else
      fill_data.indirect = 0;
  }

  inline CopySrcDstField& CopySrcDstField::operator=(const CopySrcDstField& copy_from)
  {
    if(this == &copy_from) return *this; // self-assignment
    if((field_id != FieldID(-1)) && (size > MAX_DIRECT_SIZE) && fill_data.indirect)
      free(fill_data.indirect);

    inst = copy_from.inst;
    field_id = copy_from.field_id;
    size = copy_from.size;
    redop_id = copy_from.redop_id;
    red_fold = copy_from.red_fold;
    red_exclusive = copy_from.red_exclusive;
    serdez_id = copy_from.serdez_id;
    subfield_offset = copy_from.subfield_offset;
    indirect_index = copy_from.indirect_index;

    // we know there's a fill value if the field ID is -1
    if(copy_from.field_id == FieldID(-1)) {
      if(size <= MAX_DIRECT_SIZE) {
	// copy whole buffer to make sure indirect is initialized too
	memcpy(fill_data.direct, copy_from.fill_data.direct, MAX_DIRECT_SIZE);
      } else {
	if(copy_from.fill_data.indirect) {
	  fill_data.indirect = malloc(size);
	  memcpy(fill_data.indirect, copy_from.fill_data.indirect, size);
	} else
	  fill_data.indirect = 0;
      }
    } else
      fill_data.indirect = 0;

    return *this;
  }

  inline CopySrcDstField::~CopySrcDstField(void)
  {
    if((field_id == FieldID(-1)) && (size > MAX_DIRECT_SIZE)) {
      free(fill_data.indirect);
    }
  }

  inline CopySrcDstField &CopySrcDstField::set_field(RegionInstance _inst,
						     FieldID _field_id,
						     size_t _size,
						     size_t _subfield_offset /*= 0*/)
  {
    inst = _inst;
    field_id = _field_id;
    size = _size;
    subfield_offset = _subfield_offset;
    return *this;
  }

  inline CopySrcDstField &CopySrcDstField::set_indirect(int _indirect_index,
							FieldID _field_id,
							size_t _size,
							size_t _subfield_offset /*= 0*/)
  {
    indirect_index = _indirect_index;
    field_id = _field_id;
    size = _size;
    subfield_offset = _subfield_offset;
    return *this;
  }

  inline CopySrcDstField &CopySrcDstField::set_redop(ReductionOpID _redop_id,
                                                     bool _is_fold,
                                                     bool _is_excl)
  {
    redop_id = _redop_id;
    red_fold = _is_fold;
    red_exclusive = _is_excl;
    return *this;
  }

  inline CopySrcDstField &CopySrcDstField::set_serdez(CustomSerdezID _serdez_id)
  {
    serdez_id = _serdez_id;
    return *this;
  }
  
  inline CopySrcDstField &CopySrcDstField::set_fill(const void *_data, size_t _size)
  {
    field_id = -1;
    size = _size;
    if(size <= MAX_DIRECT_SIZE) {
      memcpy(&fill_data.direct, _data, size);
    } else {
      fill_data.indirect = malloc(size);
      memcpy(fill_data.indirect, _data, size);
    }
    return *this;
  }

  template <typename T>
  inline CopySrcDstField &CopySrcDstField::set_fill(T value)
  {
    return set_fill(&value, sizeof(T));
  }

  CopySrcDstField::operator Realm::CopySrcDstField(void) const
  {
    Realm::CopySrcDstField result;
    result.inst = inst;
    result.field_id = field_id;
    result.size = size;
    result.redop_id = redop_id;
    result.red_fold = red_fold;
    result.red_exclusive = red_exclusive;
    result.serdez_id = serdez_id;
    result.subfield_offset = subfield_offset;
    result.indirect_index = indirect_index;
    static_assert(sizeof(result.fill_data) == sizeof(fill_data));
    memcpy(&result.fill_data, &fill_data, sizeof(fill_data));
    return result;
  } 

  inline void Event::wait(void) const
  {
    if (!exists()) return;
    Backtrace bt;
    bt.capture_backtrace();
    ThreadProfiler::get_thread_profiler().record_event_wait(*this, bt);
    Realm::Event::wait();
  }

  inline void Event::wait_faultaware(bool &poisoned) const
  {
    if (!exists()) return;
    Backtrace bt;
    bt.capture_backtrace();
    ThreadProfiler::get_thread_profiler().record_event_wait(*this, bt);
    Realm::Event::wait_faultaware(poisoned);
  }

  /*static*/ inline Event Event::merge_events(const Event *wait_for, size_t num_events)
  {
    const Event result = Realm::Event::merge_events(wait_for, num_events);
    ThreadProfiler::get_thread_profiler().record_event_merger(result, wait_for, num_events);
    return result;
  }

  /*static*/ inline Event Event::merge_events(Event ev1, Event ev2, Event ev3, Event ev4, Event ev5, Event ev6)
  {
    size_t num_events = 2;
    if (ev3.exists()) num_events++;
    if (ev4.exists()) num_events++;
    if (ev5.exists()) num_events++;
    if (ev6.exists()) num_events++;
    const Event events[] = { ev1, ev2, ev3, ev4, ev5, ev6 };
    const Event result = Realm::Event::merge_events(events, num_events);
    ThreadProfiler::get_thread_profiler().record_event_merger(result, events, num_events);
    return result;
  }

  /*static*/ inline Event Event::merge_events(const std::set<Event>& wait_for)
  {
    const std::vector<Event> events(wait_for.begin(), wait_for.end());
    const Event result = Realm::Event::merge_events(&events.front(), events.size());
    ThreadProfiler::get_thread_profiler().record_event_merger(result, &events.front(), events.size());
    return result;
  }

  /*static*/ inline Event Event::merge_events(const span<const Event>& wait_for)
  {
    const Event result = Realm::Event::merge_events(wait_for.data(), wait_for.size());
    ThreadProfiler::get_thread_profiler().record_event_merger(result, wait_for.data(), wait_for.size());
    return result;
  }

  /*static*/ inline Event Event::merge_events_ignorefaults(const Event *wait_for, size_t num_events)
  {
    const Event result = Realm::Event::merge_events_ignorefaults(wait_for, num_events);
    ThreadProfiler::get_thread_profiler().record_event_merger(result, wait_for, num_events);
    return result;
  }

  /*static*/ inline Event Event::merge_events_ignorefaults(const std::set<Event>& wait_for)
  {
    const std::vector<Event> events(wait_for.begin(), wait_for.end());
    const Event result = Realm::Event::merge_events_ignorefaults(&events.front(), events.size());
    ThreadProfiler::get_thread_profiler().record_event_merger(result, &events.front(), events.size());
    return result;
  }

  /*static*/ inline Event Event::merge_events_ignorefaults(const span<const Event>& wait_for)
  {
    const Event result = Realm::Event::merge_events_ignorefaults(wait_for.data(), wait_for.size());
    ThreadProfiler::get_thread_profiler().record_event_merger(result, wait_for.data(), wait_for.size());
    return result;
  }

  /*static*/ inline Event Event::ignorefaults(Event wait_for)
  {
    const Event result = Realm::Event::ignorefaults(wait_for);
    ThreadProfiler::get_thread_profiler().record_event_merger(result, &wait_for, 1);
    return result;
  }

  inline void UserEvent::trigger(Event wait_on, bool ignore_faults) const
  {
    ThreadProfiler::get_thread_profiler().record_event_trigger(*this, wait_on);
    Realm::UserEvent copy;
    copy.id = id;
    copy.trigger(wait_on, ignore_faults);
  }

  inline void UserEvent::cancel(void) const
  {
    ThreadProfiler::get_thread_profiler().record_event_poison(*this);
    Realm::UserEvent copy;
    copy.id = id;
    copy.cancel();
  }

  inline void Barrier::arrive(unsigned count, Event wait_on, const void *value, size_t size) const
  {
    ThreadProfiler::get_thread_profiler().record_barrier_arrival(*this, wait_on);
    Realm::Barrier copy;
    copy.id = id;
    copy.timestamp = timestamp;
    copy.arrive(count, wait_on, value, size);
  }

  inline Event Reservation::acquire(unsigned mode, bool exclusive, Event wait_on) const
  {
    const Event result = Realm::Reservation::acquire(mode, exclusive, wait_on);
    ThreadProfiler::get_thread_profiler().record_reservation_acquire(*this, result, wait_on);
    return result;
  }

  inline Event Reservation::try_acquire(bool retry, unsigned mode, bool exclusive, Event wait_on) const
  {
    const Event result = Realm::Reservation::try_acquire(retry, mode, exclusive, wait_on);
    ThreadProfiler::get_thread_profiler().record_reservation_acquire(*this, result, wait_on);
    return result;
  }

  inline Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
		  Event wait_on, int priority) const
  {
    ProfilingRequestSet requests;
    ThreadProfiler::get_thread_profiler().add_task_request(requests, *this, wait_on);
    return Realm::Processor::spawn(func_id, args, arglen, requests, wait_on, priority);
  }

  inline Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
                const ProfilingRequestSet &requests, Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_task_request(alt_requests, *this, wait_on);
    return Realm::Processor::spawn(func_id, args, arglen, alt_requests, wait_on, priority);
  }

  /*static*/ inline Event RegionInstance::create_instance(RegionInstance &inst,
      Memory memory, InstanceLayoutGeneric *ilg, const ProfilingRequestSet &requests, Event wait_on)
  {
    ProfilingRequestSet alt_requests = requests;
    inst.unique_event = ThreadProfiler::get_thread_profiler().add_inst_request(alt_requests, ilg, wait_on);
    return Realm::RegionInstance::create_instance(inst, memory, ilg, alt_requests, wait_on);
  }

  /*static*/ inline Event RegionInstance::create_external_instance(RegionInstance &inst,
      Memory memory, InstanceLayoutGeneric *ilg, const ExternalInstanceResource& resource,
      const ProfilingRequestSet &requests, Event wait_on)
  {
    ProfilingRequestSet alt_requests = requests;
    inst.unique_event = ThreadProfiler::get_thread_profiler().add_inst_request(alt_requests, ilg, wait_on);
    return Realm::RegionInstance::create_external_instance(inst, memory, ilg, resource, alt_requests, wait_on);
  }

  /*static*/ inline Event RegionInstance::create_external(RegionInstance &inst,
      Memory memory, uintptr_t base, InstanceLayoutGeneric *ilg,
      const ProfilingRequestSet &requests, Event wait_on)
  {
    // this interface doesn't give us a size or read-only ness, so get the size
    //  from the layout and assume it's read/write
    ExternalMemoryResource res(reinterpret_cast<void *>(base), ilg->bytes_used);
    return create_external_instance(inst, memory, ilg, res, requests, wait_on);
  }

  template <int N, typename T>
  /*static*/ inline Event RegionInstance::create_instance(RegionInstance& inst, Memory memory,
      const IndexSpace<N,T>& space, const std::vector<size_t>& field_sizes, size_t block_size,
      const ProfilingRequestSet& reqs, Event wait_on)
  {
    // smoosh hybrid block sizes back to SOA for now
    if(block_size > 1)
      block_size = 0;
    InstanceLayoutConstraints ilc(field_sizes, block_size);
    // We use fortran order here
    int dim_order[N];
    for (int i = 0; i < N; i++)
      dim_order[i] = i;
    InstanceLayoutGeneric *layout = InstanceLayoutGeneric::choose_instance_layout<N,T>(space, ilc, dim_order);
    return create_instance(inst, memory, layout, reqs, wait_on);
  }

  template <int N, typename T>
  /*static*/ inline Event RegionInstance::create_instance(RegionInstance& inst, Memory memory,
      const IndexSpace<N,T>& space, const std::map<FieldID, size_t> &field_sizes, size_t block_size,
      const ProfilingRequestSet& reqs, Event wait_on)
  {
    // smoosh hybrid block sizes back to SOA for now
    if(block_size > 1)
      block_size = 0;
    InstanceLayoutConstraints ilc(field_sizes, block_size);
    // We use fortran order here
    int dim_order[N];
    for (int i = 0; i < N; i++)
      dim_order[i] = i;
    InstanceLayoutGeneric *layout = InstanceLayoutGeneric::choose_instance_layout<N,T>(space, ilc, dim_order);
    return create_instance(inst, memory, layout, reqs, wait_on);
  }

  template <int N, typename T>
  /*static*/ inline Event RegionInstance::create_instance(RegionInstance& inst, Memory memory,
      const Rect<N,T>& rect, const std::vector<size_t>& field_sizes, size_t block_size,
      const ProfilingRequestSet& prs, Event wait_on)
  {
    return RegionInstance::create_instance<N,T>(inst, memory, IndexSpace<N,T>(rect),
        field_sizes, block_size, prs, wait_on);
  }

  template <int N, typename T>
  /*static*/ inline Event RegionInstance::create_instance(RegionInstance& inst, Memory memory,
      const Rect<N,T>& rect, const std::map<FieldID, size_t> &field_sizes, size_t block_size,
      const ProfilingRequestSet& prs, Event wait_on)
  {
    return RegionInstance::create_instance<N,T>(inst, memory, IndexSpace<N,T>(rect),
        field_sizes, block_size, prs, wait_on);
  }

  template <int N, typename T>
  /*static*/ inline Event RegionInstance::create_file_instance(RegionInstance& inst,
      const char *file_name, const IndexSpace<N,T>& space, const std::vector<FieldID> &field_ids,
      const std::vector<size_t> &field_sizes, realm_file_mode_t file_mode,
      const ProfilingRequestSet& prs, Event wait_on)
  {
    // this old interface assumes an SOA layout of fields in memory, starting at
    //  the beginning of the file
    InstanceLayoutConstraints ilc(field_ids, field_sizes, 0 /*SOA*/);
    int dim_order[N];
    for (int i = 0; i < N; i++)
      dim_order[i] = i;
    InstanceLayoutGeneric *ilg;
    ilg = InstanceLayoutGeneric::choose_instance_layout(space, ilc, dim_order);

    ExternalFileResource res(file_name, file_mode);
    return create_external_instance(inst, res.suggested_memory(), ilg, res, prs, wait_on);
  }

  template <typename FT, int N, typename T>
  inline GenericAccessor<FT,N,T>::GenericAccessor(RegionInstance inst,
		   FieldID field_id, size_t subfield_offset)
    : Realm::GenericAccessor<FT,N,T>(inst, field_id, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <typename FT, int N, typename T>
  inline GenericAccessor<FT,N,T>::GenericAccessor(RegionInstance inst,
		   FieldID field_id, const Rect<N,T>& subrect, size_t subfield_offset)
    : Realm::GenericAccessor<FT,N,T>(inst, field_id, subrect, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <typename FT, int N, typename T>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
		   FieldID field_id, size_t subfield_offset)
    : Realm::AffineAccessor<FT,N,T>(inst, field_id, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <typename FT, int N, typename T>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
		   FieldID field_id, const Rect<N,T>& subrect, size_t subfield_offset)
    : Realm::AffineAccessor<FT,N,T>(inst, field_id, subrect, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <typename FT, int N, typename T> template<int N2, typename T2>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
      const Matrix<N2,N,T2>& transform, const Point<N2,T2>& offset,
      FieldID field_id, size_t subfield_offset)
    : Realm::AffineAccessor<FT,N,T>(inst, transform, offset, field_id, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <typename FT, int N, typename T> template<int N2, typename T2>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
      const Matrix<N2,N,T2>& transform, const Point<N2,T2>& offset,
      FieldID field_id, const Rect<N,T>& subrect, size_t subfield_offset)
    : Realm::AffineAccessor<FT,N,T>(inst, transform, offset, field_id, subrect, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <typename FT, int N, typename T>
  inline MultiAffineAccessor<FT,N,T>::MultiAffineAccessor(RegionInstance inst,
		   FieldID field_id, size_t subfield_offset)
    : Realm::MultiAffineAccessor<FT,N,T>(inst, field_id, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <typename FT, int N, typename T>
  inline MultiAffineAccessor<FT,N,T>::MultiAffineAccessor(RegionInstance inst,
		   FieldID field_id, const Rect<N,T>& subrect, size_t subfield_offset)
    : Realm::MultiAffineAccessor<FT,N,T>(inst, field_id, subrect, subfield_offset)
  {
    ThreadProfiler::get_thread_profiler().record_instance_usage(inst, field_id);
  }

  template <int N, typename T>
  inline Event Rect<N,T>::fill(const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               const void *fill_value, size_t fill_value_size,
               Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_fill_request(alt_requests, dsts, wait_on);  
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::Rect<N,T>::fill(alt_dsts, alt_requests, fill_value, fill_value_size, wait_on, priority);
  }

  template <int N, typename T>
  inline Event Rect<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_copy_request(alt_requests, srcs, dsts, wait_on);
    std::vector<Realm::CopySrcDstField> alt_srcs(srcs.size());
    for (unsigned idx = 0; idx < srcs.size(); idx++)
      alt_srcs[idx] = srcs[idx];
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::Rect<N,T>::copy(alt_srcs, alt_dsts, alt_requests, wait_on, priority);
  }

  template <int N, typename T>
  inline Event Rect<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const IndexSpace<N,T> &mask,
               const ProfilingRequestSet &requests,
               Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_copy_request(alt_requests, srcs, dsts, wait_on);
    std::vector<Realm::CopySrcDstField> alt_srcs(srcs.size());
    for (unsigned idx = 0; idx < srcs.size(); idx++)
      alt_srcs[idx] = srcs[idx];
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::Rect<N,T>::copy(alt_srcs, alt_dsts, mask, alt_requests, wait_on, priority);
  }
  
  template <int N, typename T>
  inline Event IndexSpace<N,T>::fill(const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               const void *fill_value, size_t fill_value_size,
               Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_fill_request(alt_requests, dsts, wait_on);  
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::IndexSpace<N,T>::fill(alt_dsts, alt_requests, fill_value, fill_value_size, wait_on, priority);
  }

  template <int N, typename T>
  inline Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_copy_request(alt_requests, srcs, dsts, wait_on);
    std::vector<Realm::CopySrcDstField> alt_srcs(srcs.size());
    for (unsigned idx = 0; idx < srcs.size(); idx++)
      alt_srcs[idx] = srcs[idx];
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::IndexSpace<N,T>::copy(alt_srcs, alt_dsts, alt_requests, wait_on, priority);
  }

  template <int N, typename T>
  inline Event IndexSpace<N,T>::copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const std::vector<const typename CopyIndirection<N,T>::Base*> &indirects,
               const ProfilingRequestSet &requests,
               Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_copy_request(alt_requests, srcs, dsts, wait_on);
    std::vector<Realm::CopySrcDstField> alt_srcs(srcs.size());
    for (unsigned idx = 0; idx < srcs.size(); idx++)
      alt_srcs[idx] = srcs[idx];
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::IndexSpace<N,T>::copy(alt_srcs, alt_dsts, indirects, alt_requests, wait_on, priority);
  }

  inline Event IndexSpaceGeneric::copy(const std::vector<CopySrcDstField> &srcs,
	       const std::vector<CopySrcDstField> &dsts,
	       const ProfilingRequestSet &requests,
	       Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_copy_request(alt_requests, srcs, dsts, wait_on);
    std::vector<Realm::CopySrcDstField> alt_srcs(srcs.size());
    for (unsigned idx = 0; idx < srcs.size(); idx++)
      alt_srcs[idx] = srcs[idx];
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::IndexSpaceGeneric::copy(alt_srcs, alt_dsts, alt_requests, wait_on, priority);
  }

  template <int N, typename T>
  inline Event IndexSpaceGeneric::copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const std::vector<const typename CopyIndirection<N,T>::Base*> &indirects,
               const ProfilingRequestSet &requests,
               Event wait_on, int priority) const
  {
    ProfilingRequestSet alt_requests = requests;
    ThreadProfiler::get_thread_profiler().add_copy_request(alt_requests, srcs, dsts, wait_on);
    std::vector<Realm::CopySrcDstField> alt_srcs(srcs.size());
    for (unsigned idx = 0; idx < srcs.size(); idx++)
      alt_srcs[idx] = srcs[idx];
    std::vector<Realm::CopySrcDstField> alt_dsts(dsts.size());
    for (unsigned idx = 0; idx < dsts.size(); idx++)
      alt_dsts[idx] = dsts[idx];
    return Realm::IndexSpaceGeneric::copy<N,T>(alt_srcs, alt_dsts, indirects, alt_requests, wait_on, priority);
  }

  inline void Machine::get_all_processors(std::set<Processor>& pset) const
  {
    // Container type problem is dumb
    Realm::Machine::get_all_processors(*reinterpret_cast<std::set<Realm::Processor>*>(&pset));
  }

  inline void Machine::get_local_processors(std::set<Processor>& pset) const
  {
    // Container type problem is dumb
    Realm::Machine::get_local_processors(*reinterpret_cast<std::set<Realm::Processor>*>(&pset));
  }
  
  inline void Machine::get_local_processors_by_kind(std::set<Processor>& pset, Processor::Kind kind) const
  {
    // Container type problem is dumb
    Realm::Machine::get_local_processors_by_kind(*reinterpret_cast<std::set<Realm::Processor>*>(&pset), kind);
  }

  inline void Machine::get_shared_processors(Memory m, std::set<Processor>& pset, bool local_only) const
  {
    // Container type problem is dumb
    Realm::Machine::get_shared_processors(m, *reinterpret_cast<std::set<Realm::Processor>*>(&pset), local_only);
  }
}
