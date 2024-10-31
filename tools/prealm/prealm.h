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

#ifndef PREALM_H
#define PREALM_H

#include "realm.h"
#include "realm/cmdline.h"

namespace PRealm {
  // Import a bunch of types directly from the realm public interface
  // but overload ones that we need to profile

  // forward declarations
  using Realm::Logger;
  using Realm::CodeDescriptor;
  using Realm::ProfilingRequestSet;
  using Realm::CommandLineParser;
  template<int N, typename T>
  struct Rect;
  template<int N, typename T>
  class IndexSpace;

  // from utils.h
  using Realm::dynamic_extent;
  template<typename T, size_t Extent = dynamic_extent>
  using span = Realm::span<T,Extent>;

  // from faults.h
  namespace Faults {
    using namespace Realm::Faults;
  }
  using Realm::Backtrace;
  using Realm::ExecutionException;
  using Realm::CancellationException;
  using Realm::PoisonedEventException;
  using Realm::ApplicationException;

  // from redop.h
  using Realm::ReductionOpID;
  using Realm::ReductionOpUntyped;
  namespace ReductionKernels {
    using namespace Realm::ReductionKernels;
  }
#if defined(REALM_USE_CUDA) && defined(__CUDACC__)
  template<typename T>
  using HasHasCudaReductions = Realm::HasHasCudaReductions<T>;
#endif
#if defined(REALM_USE_HIP) && ( defined (__CUDACC__) || defined (__HIPCC__) )
  template<typename T>
  using HasHasHipReductions = Realm::HasHasHipReductions<T>;
#endif
  template<typename REDOP>
  using ReductionOp = Realm::ReductionOp<REDOP>;

  // from custom_serdez.h
  using Realm::CustomSerdezID;
  template<typename T>
  using SerdezObject = Realm::SerdezObject<T>;
  template <typename T, size_t MAX_SER_SIZE = 4096>
  using SimpleSerdez = Realm::SimpleSerdez<T,MAX_SER_SIZE>;
  using Realm::CustomSerdezUntyped;

  // from event.h
  class Event : public Realm::Event {
  public:
    Event(void) { id = 0; }
    Event(const Realm::Event &e) : Realm::Event(e) { }
    Event(const Event &rhs) = default;
    Event(Event &&rhs) = default;
    Event& operator=(const Realm::Event &e) { id = e.id; return *this; }
    Event& operator=(const Event &e) = default;
    Event& operator=(Event &&rhs) = default;
  public:
    // Don't care about external waits
    void wait(void) const;
    void wait_faultaware(bool &poisoned) const;
  public:
    static Event merge_events(const Event *wait_for, size_t num_events);
    static Event merge_events(Event ev1, Event ev2,
                              Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
                              Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);
    static Event merge_events(const std::set<Event>& wait_for);
    static Event merge_events(const span<const Event>& wait_for);
  public:
    static Event merge_events_ignorefaults(const Event *wait_for, size_t num_events);
    static Event merge_events_ignorefaults(const span<const Event>& wait_for);
    static Event merge_events_ignorefaults(const std::set<Event>& wait_for);
    static Event ignorefaults(Event wait_for);
  public:
    static const Event NO_EVENT;
  };
  static_assert(sizeof(Event) == sizeof(Realm::Event));

  class UserEvent : public Event {
  public:
    UserEvent(void) { id = 0; }
    UserEvent(const Realm::UserEvent &e) { id = e.id; }
    UserEvent(const UserEvent &rhs) = default;
    UserEvent(UserEvent &&rhs) = default;
    UserEvent& operator=(const Realm::UserEvent &e) { id = e.id; return *this; }
    UserEvent& operator=(const UserEvent &rhs) = default;
    UserEvent& operator=(UserEvent &&rhs) = default;
  public:
    void trigger(Event wait_on = Event::NO_EVENT,
                   bool ignore_faults = false) const;
    void cancel(void) const;

    static const UserEvent NO_USER_EVENT;
  };
  static_assert(sizeof(UserEvent) == sizeof(Realm::UserEvent));

  class Barrier : public Event {
  public:
    Barrier(void) { id = 0; timestamp = 0; }
    Barrier(const Realm::Barrier &b) { id = b.id; timestamp = b.timestamp; }
    Barrier(const Barrier &b) = default;
    Barrier(Barrier &&b) = default;
    Barrier& operator=(const Realm::Barrier &b) { id = b.id; timestamp = b.timestamp; return *this; }
    Barrier& operator=(const Barrier &b) = default;
    Barrier& operator=(Barrier &&b) = default;
  public:
    typedef ::realm_barrier_timestamp_t timestamp_t;
    timestamp_t timestamp;

    static const Barrier NO_BARRIER;

    void arrive(unsigned count = 1, Event wait_on = Event::NO_EVENT,
                  const void* reduce_value = 0,
                  size_t reduce_value_size = 0) const;
  };
  static_assert(sizeof(Barrier) == sizeof(Realm::Barrier));

  class CompletionQueue : public Realm::CompletionQueue {
  public:
    CompletionQueue(void) { id = 0; }
    CompletionQueue(Realm::CompletionQueue q) : Realm::CompletionQueue(q) {}
    CompletionQueue(const CompletionQueue &q) = default;
    CompletionQueue(CompletionQueue &&q) = default;
    CompletionQueue& operator=(Realm::CompletionQueue q) { this->id = q.id; return *this; }
    CompletionQueue& operator=(const CompletionQueue &q) = default;
    CompletionQueue& operator=(CompletionQueue &&q) = default;
    // TODO: tracking completion queue events is hard because you need to follow this
    // through to the point where we pop the events out the queue
    Event get_nonempty_event(void);
    size_t pop_events(Event *events, size_t max_events);
  public:
    static const CompletionQueue NO_QUEUE;
  };
  static_assert(sizeof(CompletionQueue) == sizeof(Realm::CompletionQueue));

  // from reservation.h
  class Reservation : public Realm::Reservation {
  public:
    Reservation(void) { id = 0; }
    Reservation(Realm::Reservation r) : Realm::Reservation(r) { }
    Reservation(const Reservation &r) = default;
    Reservation(Reservation &&r) = default;
    Reservation& operator=(Realm::Reservation r) { this->id = r.id; return *this; }
    Reservation& operator=(const Reservation &r) = default;
    Reservation& operator=(Reservation &&r) = default;
    Event acquire(unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT) const;
    Event try_acquire(bool retry, unsigned mode = 0, bool exclusive = true,
			Event wait_on = Event::NO_EVENT) const;
  public:
    static const Reservation NO_RESERVATION;
  };
  static_assert(sizeof(Reservation) == sizeof(Realm::Reservation));
  
  class FastReservation : public Realm::FastReservation {
  public:
    // TODO: do we need to profile these, right now Legion Prof won't use them
    Event lock(WaitMode mode = SPIN); // synonym for wrlock()
    Event wrlock(WaitMode mode = SPIN);
    Event rdlock(WaitMode mode = SPIN);
  };
  static_assert(sizeof(FastReservation) == sizeof(Realm::FastReservation));

  // from memory.h
  using Realm::Memory;

  // from processor.h
  using Realm::AddressSpace;

  class Processor : public Realm::Processor {
  public:
    Processor(void) { id = 0; }
    Processor(Realm::Processor p) : Realm::Processor(p) { }
    Processor(const Processor &p) = default;
    Processor(Processor &&p) = default;
    Processor& operator=(Realm::Processor p) { this->id = p.id; return *this; }
    Processor& operator=(const Processor &p) = default;
    Processor& operator=(Processor &&p) = default;
    typedef void (*TaskFuncPtr)(const void *args, size_t arglen,
				  const void *user_data, size_t user_data_len,
				  Processor proc);
    Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
		  Event wait_on = Event::NO_EVENT, int priority = 0) const;

    // Same as the above but with requests for profiling
    Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
                const ProfilingRequestSet &requests,
                Event wait_on = Event::NO_EVENT, int priority = 0) const;

    // TODO: do we need to profile these? right now Legion prof won't use them
    Event register_task(TaskFuncID func_id,
                        const CodeDescriptor& codedesc,
                        const ProfilingRequestSet& prs,
                        const void *user_data = 0, size_t user_data_len = 0) const;

    static Event register_task_by_kind(Kind target_kind, bool global,
                                       TaskFuncID func_id,
                                       const CodeDescriptor& codedesc,
                                       const ProfilingRequestSet& prs,
                                       const void *user_data = 0, size_t user_data_len = 0);
    static const Processor NO_PROC;
  };
  static_assert(sizeof(Processor) == sizeof(Realm::Processor));

  class ProcessorGroup : public Processor {
  public:
    ProcessorGroup(void) { id = 0; }
    ProcessorGroup(Realm::ProcessorGroup g) : Processor(g) { }
    ProcessorGroup(const ProcessorGroup &g) = default;
    ProcessorGroup(ProcessorGroup &&g) = default;
    ProcessorGroup& operator=(Realm::ProcessorGroup g) { this->id = g.id; return *this; }
    ProcessorGroup& operator=(const ProcessorGroup &g) = default;
    ProcessorGroup& operator=(ProcessorGroup &&g) = default;

    static const ProcessorGroup NO_PROC_GROUP;
  };
  static_assert(sizeof(ProcessorGroup) == sizeof(Realm::ProcessorGroup));

  // from inst_layout.h
  using Realm::InstanceLayoutConstraints;
  namespace PieceLookup {
    using namespace Realm::PieceLookup;
  }
  using Realm::InstanceLayoutGeneric;
  using Realm::InstanceLayoutOpaque;
  namespace PieceLayoutTypes {
    using namespace Realm::PieceLayoutTypes;
  }
  using Realm::InstanceLayoutPieceBase;
  template <int N, typename T = int>
  using InstanceLayoutPiece = Realm::InstanceLayoutPiece<N,T>;
  template <int N, typename T = int>
  using AffineLayoutPiece = Realm::AffineLayoutPiece<N,T>;
  template <int N, typename T = int>
  using InstancePieceList = Realm::InstancePieceList<N,T>;
  template <int N, typename T = int>
  using InstanceLayout = Realm::InstanceLayout<N,T>;

  // from instance.h
  using Realm::FieldID;
  using Realm::ExternalInstanceResource;
  using Realm::ExternalMemoryResource;
  using Realm::ExternalFileResource;
  class RegionInstance : public Realm::RegionInstance {
  public:
    RegionInstance(void) { id = 0; unique_event = Event::NO_EVENT; }
    // No implicit conversion from realm region instances because of unique event
    RegionInstance(const RegionInstance &i) = default;
    RegionInstance(RegionInstance &&i) = default;
    // No implicit conversion from realm region instances because of unique event
    RegionInstance& operator=(const RegionInstance &i) = default;
    RegionInstance& operator=(RegionInstance &&i) = default;
    // TODO: handle redistricting cases
    Event redistrict(RegionInstance &instance, const InstanceLayoutGeneric *layout,
                     const ProfilingRequestSet &prs, Event wait_on = Event::NO_EVENT);
    Event redistrict(RegionInstance *instances, const InstanceLayoutGeneric **layouts,
                     size_t num_layouts, const ProfilingRequestSet *prs,
                     Event wait_on = Event::NO_EVENT);
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 InstanceLayoutGeneric *ilg,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);
    static Event create_external_instance(RegionInstance& inst,
					  Memory memory,
					  InstanceLayoutGeneric *ilg,
					  const ExternalInstanceResource& resource,
					  const ProfilingRequestSet& prs,
					  Event wait_on = Event::NO_EVENT);
    static Event create_external(RegionInstance& inst,
				 Memory memory, uintptr_t base,
				 InstanceLayoutGeneric *ilg,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);
    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const IndexSpace<N,T>& space,
				 const std::vector<size_t>& field_sizes,
				 size_t block_size,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);
    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const IndexSpace<N,T>& space,
				 const std::map<FieldID, size_t>& field_sizes,
				 size_t block_size,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);
    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const Rect<N,T>& rect,
				 const std::vector<size_t>& field_sizes,
				 size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);
    template <int N, typename T>
    static Event create_instance(RegionInstance& inst,
				 Memory memory,
				 const Rect<N,T>& rect,
				 const std::map<FieldID, size_t>& field_sizes,
				 size_t block_size, // 0=SOA, 1=AOS, 2+=hybrid
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);
    template <int N, typename T>
    static Event create_file_instance(RegionInstance& inst,
				      const char *file_name,
				      const IndexSpace<N,T>& space,
				      const std::vector<FieldID> &field_ids,
				      const std::vector<size_t> &field_sizes,
				      realm_file_mode_t file_mode,
				      const ProfilingRequestSet& prs,
				      Event wait_on = Event::NO_EVENT);
    // TODO: Profile this? Legion prof won't recognize where the event came from
    Event fetch_metadata(Processor target) const;
    static const RegionInstance NO_INST;
  public:
    // A unique event that acts as a handle for describing this instance for profiling
    Event unique_event;
  };
  using Realm::ExternalMemoryResource;
  using Realm::ExternalFileResource;

  // Need a special copy src dst field that records RegionInstances with unique names
  struct CopySrcDstField {
  public:
    CopySrcDstField(void);
    CopySrcDstField(const CopySrcDstField& copy_from);
    CopySrcDstField& operator=(const CopySrcDstField& copy_from);
    ~CopySrcDstField(void);
    CopySrcDstField &set_field(RegionInstance _inst, FieldID _field_id,
			       size_t _size, size_t _subfield_offset = 0);
    CopySrcDstField &set_indirect(int _indirect_index, FieldID _field_id,
				  size_t _size, size_t _subfield_offset = 0);
    CopySrcDstField &set_redop(ReductionOpID _redop_id, bool _is_fold, bool exclusive = false);
    CopySrcDstField &set_serdez(CustomSerdezID _serdez_id);
    CopySrcDstField &set_fill(const void *_data, size_t _size);
    template <typename T>
    CopySrcDstField &set_fill(T value);
    
    // Implicit conversion to Realm::CopySrcDstField
    operator Realm::CopySrcDstField(void) const;

  public:
    RegionInstance inst;
    FieldID field_id;
    size_t size;
    ReductionOpID redop_id;
    bool red_fold;
    bool red_exclusive;
    CustomSerdezID serdez_id;
    size_t subfield_offset;
    int indirect_index;
    static constexpr size_t MAX_DIRECT_SIZE = 8;
    union {
      char direct[8];
      void *indirect;
    } fill_data;
  };
  
  // from point.h
  template<int N, typename T = int>
  using Point = Realm::Point<N,T>;
  template<int N, typename T = int>
  struct Rect : public Realm::Rect<N,T> {
    using Realm::Rect<N,T>::Rect;
    REALM_CUDA_HD
    Rect(void) { }
    template<int N2, typename T2> REALM_CUDA_HD
    Rect(const Realm::Rect<N2,T2> &rhs) : Realm::Rect<N,T>(rhs) { }
    template<int N2, typename T2> REALM_CUDA_HD
    Rect(const Rect<N2,T2> &rhs) : Realm::Rect<N,T>(rhs) { }
    template<int N2, typename T2> REALM_CUDA_HD
    Rect<N,T>& operator=(const Realm::Rect<N2,T2> &rhs) { Realm::Rect<N,T>::operator=(rhs); return *this; }
    template<int N2, typename T2> REALM_CUDA_HD
    Rect<N,T>& operator=(const Rect<N2,T2> &rhs) { Realm::Rect<N,T>::operator=(rhs); return *this; }

    Event fill(const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               const void *fill_value, size_t fill_value_size,
               Event wait_on = Event::NO_EVENT,
               int priority = 0) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               int priority = 0) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const IndexSpace<N,T> &mask,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               int priority = 0) const;
  };
  template<int N, typename T = int>
  using PointInRectIterator = Realm::PointInRectIterator<N,T>;
  template<int M, int N, typename T = int>
  using Matrix = Realm::Matrix<M,N,T>; 

  // from profiling.h
  using Realm::ProfilingMeasurementID;
  namespace ProfilingMeasurements {
    using namespace Realm::ProfilingMeasurements;
  }
  using Realm::ProfilingRequest;
  using Realm::ProfilingMeasurementCollection;
  using Realm::ProfilingResponse;  

  // from machine.h
  class Machine : public Realm::Machine {
  public:
    using Realm::Machine::Machine;
    Machine(const Realm::Machine &m) : Realm::Machine(m) { }
    Machine(const Machine &m) = default;
    Machine(Machine &&m) = default;
    Machine& operator=(const Realm::Machine &m) { Realm::Machine::operator=(m); return *this; }
    Machine& operator=(const Machine &m) = default;
    Machine& operator=(Machine &&m) = default;

    void get_all_processors(std::set<Processor>& pset) const;

    void get_local_processors(std::set<Processor>& pset) const;
    void get_local_processors_by_kind(std::set<Processor>& pset,
                                      Processor::Kind kind) const;
    void get_shared_processors(Memory m, std::set<Processor>& pset,
				 bool local_only = true) const;
  };
  static_assert(sizeof(Machine) == sizeof(Realm::Machine));

  template<typename QT, typename RT>
  using MachineQueryIterator = Realm::MachineQueryIterator<QT,RT>;
  
  // from runtime.h
  class Runtime : public Realm::Runtime {
  public:
    using Realm::Runtime::Runtime;
    Runtime(const Realm::Runtime &r) : Realm::Runtime(r) { }
    Runtime(const Runtime &r) = default;
    Runtime(Runtime &&r) = default;
    Runtime& operator=(const Realm::Runtime &r) { Realm::Runtime::operator=(r); return *this; }
    Runtime& operator=(const Runtime &r) = default;
    Runtime& operator=(Runtime &&r) = default;
    void parse_command_line(int argc, char **argv);
    void parse_command_line(std::vector<std::string> &cmdline,
                            bool remove_realm_args = false);
    bool configure_from_command_line(int argc, char **argv);
    bool configure_from_command_line(std::vector<std::string> &cmdline,
                                     bool remove_realm_args = false);
    void start(void);
    bool init(int *argc, char ***argv);
    bool register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr);
    void shutdown(Event wait_on = Event::NO_EVENT, int result_code = 0);
    int wait_for_shutdown(void);
  };
  static_assert(sizeof(Runtime) == sizeof(Realm::Runtime));

  template <typename FT, int N, typename T = int>
  class GenericAccessor : public Realm::GenericAccessor<FT,N,T> {
  public:
    GenericAccessor(void) { }
    GenericAccessor(RegionInstance inst,
		   FieldID field_id, size_t subfield_offset = 0);
    GenericAccessor(RegionInstance inst,
		   FieldID field_id, const Rect<N,T>& subrect,
		   size_t subfield_offset = 0);
  };
  template <typename FT, int N, typename T = int>
  class AffineAccessor : public Realm::AffineAccessor<FT,N,T> {
  public:
    REALM_CUDA_HD
    AffineAccessor(void) { }
    AffineAccessor(RegionInstance inst,
		   FieldID field_id, size_t subfield_offset = 0);
    AffineAccessor(RegionInstance inst,
		   FieldID field_id, const Rect<N,T>& subrect,
		   size_t subfield_offset = 0);
    template <int N2, typename T2>
    AffineAccessor(RegionInstance inst,
		   const Matrix<N2, N, T2>& transform,
		   const Point<N2, T2>& offset,
		   FieldID field_id, size_t subfield_offset = 0);
    template <int N2, typename T2>
    AffineAccessor(RegionInstance inst,
		   const Matrix<N2, N, T2>& transform,
		   const Point<N2, T2>& offset,
		   FieldID field_id, const Rect<N,T>& subrect,
		   size_t subfield_offset = 0);
    REALM_CUDA_HD
    ~AffineAccessor(void) { }

    AffineAccessor(const AffineAccessor &) = default;
    AffineAccessor &operator=(const AffineAccessor &) = default;
    AffineAccessor(AffineAccessor &&) noexcept = default;
    AffineAccessor &operator=(AffineAccessor &&) noexcept = default;
  };
  template <typename FT, int N, typename T>
  class MultiAffineAccessor : public Realm::MultiAffineAccessor<FT,N,T> {
  public:
    REALM_CUDA_HD
    MultiAffineAccessor(void) { }
    MultiAffineAccessor(RegionInstance inst,
			FieldID field_id, size_t subfield_offset = 0);
    MultiAffineAccessor(RegionInstance inst,
			FieldID field_id, const Rect<N,T>& subrect,
			size_t subfield_offset = 0);
    REALM_CUDA_HD
    ~MultiAffineAccessor(void) { }

    MultiAffineAccessor(const MultiAffineAccessor &) = default;
    MultiAffineAccessor &operator=(const MultiAffineAccessor &) = default;
    MultiAffineAccessor(MultiAffineAccessor &&) noexcept = default;
    MultiAffineAccessor &operator=(MultiAffineAccessor &&) noexcept = default;
  }; 

  // from indexspace.h
  template <typename IS, typename FT>
  using FieldDataDescriptor = Realm::FieldDataDescriptor<IS,FT>;
  template <int N, typename T = int>
  using TranslationTransform = Realm::TranslationTransform<N,T>;
  template <int M, int N, typename T = int>
  using AffineTransform = Realm::AffineTransform<M,N,T>;
  template <int N, typename T, int N2, typename T2>
  using StructuredTransform = Realm::StructuredTransform<N,T,N2,T2>;
  template <int N, typename T, int N2, typename T2>
  using DomainTransform = Realm::DomainTransform<N,T,N2,T2>;
  template <int N, typename T = int>
  using CopyIndirection = Realm::CopyIndirection<N,T>;
  template<int N, typename T>
  class IndexSpace : public Realm::IndexSpace<N,T> {
  public:
    using Realm::IndexSpace<N,T>::IndexSpace;
    IndexSpace(void) { }
    IndexSpace(const Realm::IndexSpace<N,T> &i) : Realm::IndexSpace<N,T>(i) { }
    IndexSpace(const IndexSpace<N,T> &i) = default;
    IndexSpace(IndexSpace<N,T> &&i) = default;
    IndexSpace& operator=(const Realm::IndexSpace<N,T> &i) { Realm::IndexSpace<N,T>::operator=(i); return *this; }
    IndexSpace& operator=(const IndexSpace<N,T> &i) = default;
    IndexSpace& operator=(IndexSpace<N,T> &&i) = default;

    Event make_valid(bool precise = true) const;
    Event fill(const std::vector<CopySrcDstField>& dsts,
               const ProfilingRequestSet& requests, const void* fill_value,
               size_t fill_value_size, Event wait_on = Event::NO_EVENT,
               int priority = 0) const;
    Event copy(const std::vector<CopySrcDstField>& srcs,
               const std::vector<CopySrcDstField>& dsts,
               const ProfilingRequestSet& requests,
               Event wait_on = Event::NO_EVENT, int priority = 0) const;
    Event copy(const std::vector<CopySrcDstField>& srcs,
               const std::vector<CopySrcDstField>& dsts,
               const std::vector<const typename CopyIndirection<N, T>::Base*>&
                   indirects,
               const ProfilingRequestSet& requests,
               Event wait_on = Event::NO_EVENT, int priority = 0) const;
    // TODO: fill in the dependent partition methods
    Event create_equal_subspace(size_t count, size_t granularity,
                                unsigned index, IndexSpace<N,T> &subspace,
                                const ProfilingRequestSet &reqs,
                                Event wait_on = Event::NO_EVENT) const;
    Event create_equal_subspaces(size_t count, size_t granularity,
				 std::vector<IndexSpace<N,T> >& subspaces,
				 const ProfilingRequestSet &reqs,
				 Event wait_on = Event::NO_EVENT) const;
    Event create_weighted_subspaces(size_t count, size_t granularity,
				    const std::vector<int>& weights,
				    std::vector<IndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;
    Event create_weighted_subspaces(size_t count, size_t granularity,
				    const std::vector<size_t>& weights,
				    std::vector<IndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;
    template <typename FT>
    Event create_subspace_by_field(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT>> &field_data,
        FT color, IndexSpace<N, T> &subspace, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;

    template <typename FT>
    Event create_subspaces_by_field(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT>> &field_data,
        const std::vector<FT> &colors, std::vector<IndexSpace<N, T>> &subspaces,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;
    template <typename FT, typename FT2>
    Event create_subspace_by_field(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT>> &field_data,
        const CodeDescriptor &codedesc, FT2 color, IndexSpace<N, T> &subspace,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;
    template <typename FT, typename FT2>
    Event create_subspaces_by_field(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT>> &field_data,
        const CodeDescriptor &codedesc, const std::vector<FT2> &colors,
        std::vector<IndexSpace<N, T>> &subspaces, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspace_by_image(
        const TRANSFORM &transform, const IndexSpace<N2, T2> &source,
        const IndexSpace<N, T> &image, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspaces_by_image(
        const TRANSFORM &transform, const std::vector<IndexSpace<N2, T2>> &sources,
        std::vector<IndexSpace<N, T>> &images, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_image(
        const DomainTransform<N, T, N2, T2> &domain_transform,
        const std::vector<IndexSpace<N2, T2>> &sources,
        std::vector<IndexSpace<N, T>> &images, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspace_by_image(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>
            &field_data,
        const IndexSpace<N2, T2> &source, IndexSpace<N, T> &image,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_image(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>
            &field_data,
        const std::vector<IndexSpace<N2, T2>> &sources,
        std::vector<IndexSpace<N, T>> &images, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspace_by_image(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Rect<N, T>>>
            &field_data,
        const IndexSpace<N2, T2> &source, IndexSpace<N, T> &image,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_image(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Rect<N, T>>>
            &field_data,
        const std::vector<IndexSpace<N2, T2>> &sources,
        std::vector<IndexSpace<N, T>> &images, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_image_with_difference(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>
            &field_data,
        const std::vector<IndexSpace<N2, T2>> &sources,
        const std::vector<IndexSpace<N, T>> &diff_rhs,
        std::vector<IndexSpace<N, T>> &images, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_image_with_difference(
        const DomainTransform<N, T, N2, T2> &domain_transform,
        const std::vector<IndexSpace<N2, T2>> &sources,
        const std::vector<IndexSpace<N, T>> &diff_rhs,
        std::vector<IndexSpace<N, T>> &images, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspace_by_preimage(
        const TRANSFORM &transform, const IndexSpace<N2, T2> &target,
        IndexSpace<N, T> &preimage, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspaces_by_preimage(
        const TRANSFORM &transform, const std::vector<IndexSpace<N2, T2>> &targets,
        std::vector<IndexSpace<N, T>> &preimages, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_preimage(
        const DomainTransform<N2, T2, N, T> &domain_transform,
        const std::vector<IndexSpace<N2, T2>> &targets,
        std::vector<IndexSpace<N, T>> &preimages, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspace_by_preimage(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Point<N2, T2>>>
            &field_data,
        const IndexSpace<N2, T2> &target, IndexSpace<N, T> &preimage,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_preimage(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Point<N2, T2>>>
            &field_data,
        const std::vector<IndexSpace<N2, T2>> &targets,
        std::vector<IndexSpace<N, T>> &preimages, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspace_by_preimage(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Rect<N2, T2>>>
            &field_data,
        const IndexSpace<N2, T2> &target, IndexSpace<N, T> &preimage,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;
    template <int N2, typename T2>
    Event create_subspaces_by_preimage(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Rect<N2, T2>>>
            &field_data,
        const std::vector<IndexSpace<N2, T2>> &targets,
        std::vector<IndexSpace<N, T>> &preimages, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    static Event compute_union(const IndexSpace<N,T>& lhs,
				    const IndexSpace<N,T>& rhs,
				    IndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);
    static Event compute_unions(const std::vector<IndexSpace<N,T> >& lhss,
        const std::vector<IndexSpace<N,T> >& rhss,
        std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_unions(const IndexSpace<N,T>& lhs,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_unions(const std::vector<IndexSpace<N,T> >& lhss,
				     const IndexSpace<N,T>& rhs,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_intersection(const IndexSpace<N,T>& lhs,
				    const IndexSpace<N,T>& rhs,
				    IndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);
    static Event compute_intersections(const std::vector<IndexSpace<N,T> >& lhss,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_intersections(const IndexSpace<N,T>& lhs,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_intersections(const std::vector<IndexSpace<N,T> >& lhss,
				     const IndexSpace<N,T>& rhs,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_difference(const IndexSpace<N,T>& lhs,
				    const IndexSpace<N,T>& rhs,
				    IndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);
    static Event compute_differences(const std::vector<IndexSpace<N,T> >& lhss,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_differences(const IndexSpace<N,T>& lhs,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_differences(const std::vector<IndexSpace<N,T> >& lhss,
				     const IndexSpace<N,T>& rhs,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    static Event compute_union(const std::vector<IndexSpace<N,T> >& subspaces,
			       IndexSpace<N,T>& result,
			       const ProfilingRequestSet &reqs,
			       Event wait_on = Event::NO_EVENT);
    static Event compute_intersection(const std::vector<IndexSpace<N,T> >& subspaces,
				      IndexSpace<N,T>& result,
				      const ProfilingRequestSet &reqs,
				      Event wait_on = Event::NO_EVENT);
  };

  class IndexSpaceGeneric : public Realm::IndexSpaceGeneric {
  public:
    using Realm::IndexSpaceGeneric::IndexSpaceGeneric;
    IndexSpaceGeneric(void) { }
    IndexSpaceGeneric(const Realm::IndexSpaceGeneric &i) : Realm::IndexSpaceGeneric(i) { }
    IndexSpaceGeneric(const IndexSpaceGeneric &i) = default;
    IndexSpaceGeneric(IndexSpaceGeneric &&i) = default;
    IndexSpaceGeneric& operator=(const Realm::IndexSpaceGeneric &i) { Realm::IndexSpaceGeneric::operator=(i); return *this; }
    IndexSpaceGeneric& operator=(const IndexSpaceGeneric &i) = default;
    IndexSpaceGeneric& operator=(IndexSpaceGeneric &&i) = default;
    Event copy(const std::vector<CopySrcDstField> &srcs,
	       const std::vector<CopySrcDstField> &dsts,
	       const ProfilingRequestSet &requests,
	       Event wait_on = Event::NO_EVENT,
	       int priority = 0) const;

    template <int N, typename T>
    Event copy(const std::vector<CopySrcDstField> &srcs,
	       const std::vector<CopySrcDstField> &dsts,
	       const std::vector<const typename CopyIndirection<N,T>::Base *> &indirects,
	       const ProfilingRequestSet &requests,
	       Event wait_on = Event::NO_EVENT,
	       int priority = 0) const;
  };
  static_assert(sizeof(IndexSpaceGeneric) == sizeof(Realm::IndexSpaceGeneric));

  // from codedesc.h
  using Realm::Type;
  using Realm::OpaqueType;
  using Realm::IntegerType;
  using Realm::PointerType;
  using Realm::FunctionPointerType;
  namespace TypeConv {
    using namespace Realm::TypeConv;
  }
  using Realm::CodeImplementation;
  using Realm::CodeProperty;
  using Realm::CodeTranslator;
  using Realm::FunctionPointerImplementation;
#ifdef REALM_USE_DLFCN
  using Realm::DSOReferenceImplementation;
  using Realm::DSOCodeTranslator;
#endif

  // from subgraph.h
  using Realm::Subgraph;
  using Realm::SubgraphDefinition;

  // from module_config.h
  using Realm::ModuleConfig;
  
}

#include "prealm.inl"

#endif // ifndef PREALM_H
