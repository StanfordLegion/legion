/* Copyright 2018 Stanford University
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


#ifndef __LEGION_C_UTIL_H__
#define __LEGION_C_UTIL_H__

/**
 * \file legion_c_util.h
 * Legion C API: C++ Conversion Utilities
 */

#include "legion.h"
#include "legion/legion_c.h"
#include "legion/legion_mapping.h"
#include "mappers/mapping_utilities.h"

#include <stdlib.h>
#include <string.h>
#include <algorithm>

namespace Legion {

    class CContext;

    class CObjectWrapper {
    public:
      typedef Legion::UnsafeFieldAccessor<char,1,coord_t,
                Realm::AffineAccessor<char,1,coord_t> >   ArrayAccessor1D;
      typedef Legion::UnsafeFieldAccessor<char,2,coord_t,
                Realm::AffineAccessor<char,2,coord_t> >   ArrayAccessor2D;
      typedef Legion::UnsafeFieldAccessor<char,3,coord_t,
                Realm::AffineAccessor<char,3,coord_t> >   ArrayAccessor3D;

#ifdef __ICC
// icpc complains about "error #858: type qualifier on return type is meaningless"
// but it's pretty annoying to get this macro to handle all the cases right
#pragma warning (push)
#pragma warning (disable: 858)
#endif
#define NEW_OPAQUE_WRAPPER(T_, T)                                       \
      static T_ wrap(T t) {                                             \
        T_ t_;                                                          \
        t_.impl = static_cast<void *>(t);                               \
        return t_;                                                      \
      }                                                                 \
      static const T_ wrap_const(const T t) {                           \
        T_ t_;                                                          \
        t_.impl = const_cast<void *>(static_cast<const void *>(t));     \
        return t_;                                                      \
      }                                                                 \
      static T unwrap(T_ t_) {                                          \
        return static_cast<T>(t_.impl);                                 \
      }                                                                 \
      static const T unwrap_const(const T_ t_) {                        \
        return static_cast<const T>(t_.impl);                           \
      }

      NEW_OPAQUE_WRAPPER(legion_runtime_t, Runtime *);
      NEW_OPAQUE_WRAPPER(legion_context_t, CContext *);
      NEW_OPAQUE_WRAPPER(legion_domain_point_iterator_t, Domain::DomainPointIterator *);
      NEW_OPAQUE_WRAPPER(legion_coloring_t, Coloring *);
      NEW_OPAQUE_WRAPPER(legion_domain_coloring_t, DomainColoring *);
      NEW_OPAQUE_WRAPPER(legion_point_coloring_t, PointColoring *);
      NEW_OPAQUE_WRAPPER(legion_domain_point_coloring_t, DomainPointColoring *);
      NEW_OPAQUE_WRAPPER(legion_multi_domain_point_coloring_t, MultiDomainPointColoring *);
      NEW_OPAQUE_WRAPPER(legion_index_space_allocator_t, IndexSpaceAllocator *);
      NEW_OPAQUE_WRAPPER(legion_field_allocator_t, FieldAllocator *);
      NEW_OPAQUE_WRAPPER(legion_argument_map_t, ArgumentMap *);
      NEW_OPAQUE_WRAPPER(legion_predicate_t, Predicate *);
      NEW_OPAQUE_WRAPPER(legion_future_t, Future *);
      NEW_OPAQUE_WRAPPER(legion_future_map_t, FutureMap *);
      NEW_OPAQUE_WRAPPER(legion_task_launcher_t, TaskLauncher *);
      NEW_OPAQUE_WRAPPER(legion_index_launcher_t, IndexTaskLauncher *);
      NEW_OPAQUE_WRAPPER(legion_inline_launcher_t, InlineLauncher *);
      NEW_OPAQUE_WRAPPER(legion_copy_launcher_t, CopyLauncher *);
      NEW_OPAQUE_WRAPPER(legion_acquire_launcher_t, AcquireLauncher *);
      NEW_OPAQUE_WRAPPER(legion_release_launcher_t, ReleaseLauncher *);
      NEW_OPAQUE_WRAPPER(legion_attach_launcher_t, AttachLauncher *);
      NEW_OPAQUE_WRAPPER(legion_must_epoch_launcher_t, MustEpochLauncher *);
      NEW_OPAQUE_WRAPPER(legion_physical_region_t, PhysicalRegion *);
      NEW_OPAQUE_WRAPPER(legion_accessor_array_1d_t, ArrayAccessor1D *);
      NEW_OPAQUE_WRAPPER(legion_accessor_array_2d_t, ArrayAccessor2D *);
      NEW_OPAQUE_WRAPPER(legion_accessor_array_3d_t, ArrayAccessor3D *);
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
      NEW_OPAQUE_WRAPPER(legion_index_iterator_t, IndexIterator *);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
      NEW_OPAQUE_WRAPPER(legion_task_t, Task *);
      NEW_OPAQUE_WRAPPER(legion_inline_t, InlineMapping *);
      NEW_OPAQUE_WRAPPER(legion_mappable_t, Mappable *);
      NEW_OPAQUE_WRAPPER(legion_region_requirement_t , RegionRequirement *);
      NEW_OPAQUE_WRAPPER(legion_machine_t, Machine *);
      NEW_OPAQUE_WRAPPER(legion_mapper_t, Mapping::Mapper *);
      NEW_OPAQUE_WRAPPER(legion_processor_query_t, Machine::ProcessorQuery *);
      NEW_OPAQUE_WRAPPER(legion_memory_query_t, Machine::MemoryQuery *);
      NEW_OPAQUE_WRAPPER(legion_machine_query_interface_t,
                         Mapping::Utilities::MachineQueryInterface *);
      NEW_OPAQUE_WRAPPER(legion_default_mapper_t, Mapping::DefaultMapper *);
      NEW_OPAQUE_WRAPPER(legion_execution_constraint_set_t, ExecutionConstraintSet *);
      NEW_OPAQUE_WRAPPER(legion_layout_constraint_set_t, LayoutConstraintSet *);
      NEW_OPAQUE_WRAPPER(legion_task_layout_constraint_set_t, TaskLayoutConstraintSet *);
      NEW_OPAQUE_WRAPPER(legion_map_task_input_t, Mapping::Mapper::MapTaskInput *);
      NEW_OPAQUE_WRAPPER(legion_map_task_output_t, Mapping::Mapper::MapTaskOutput *);
      NEW_OPAQUE_WRAPPER(legion_slice_task_output_t, Mapping::Mapper::SliceTaskOutput *);
      NEW_OPAQUE_WRAPPER(legion_physical_instance_t, Mapping::PhysicalInstance *);
      NEW_OPAQUE_WRAPPER(legion_mapper_runtime_t, Mapping::MapperRuntime *);
      NEW_OPAQUE_WRAPPER(legion_mapper_context_t, Mapping::MapperContext);
      typedef std::map<FieldID, const char *> FieldMap;
      NEW_OPAQUE_WRAPPER(legion_field_map_t, FieldMap *);
#undef NEW_OPAQUE_WRAPPER
#ifdef __ICC
// icpc complains about "error #858: type qualifier on return type is meaningless"
// but it's pretty annoying to get this macro to handle all the cases right
#pragma warning (pop)
#endif

      static legion_ptr_t
      wrap(ptr_t ptr)
      {
        legion_ptr_t ptr_;
        ptr_.value = ptr.value;
        return ptr_;
      }

      static ptr_t
      unwrap(legion_ptr_t ptr_)
      {
        ptr_t ptr;
        ptr.value = ptr_.value;
        return ptr;
      }

#define NEW_POINT_WRAPPER(T_, T, DIM)                \
      static T_ wrap(T t) {                          \
        T_ t_;                                       \
        for (int i = 0; i < DIM; i++)                \
          t_.x[i] = t[i];                            \
        return t_;                                   \
      }                                              \
      static T unwrap(T_ t_) {                       \
        T t;                                         \
        for (int i = 0; i < DIM; i++)                \
          t[i] = t_.x[i];                            \
        return t;                                    \
      }

      typedef Point<1,coord_t> Point1D;
      typedef Point<2,coord_t> Point2D;
      typedef Point<3,coord_t> Point3D;
      NEW_POINT_WRAPPER(legion_point_1d_t, Point1D, 1);
      NEW_POINT_WRAPPER(legion_point_2d_t, Point2D, 2);
      NEW_POINT_WRAPPER(legion_point_3d_t, Point3D, 3);
#undef NEW_POINT_WRAPPER

#define NEW_RECT_WRAPPER(T_, T, PT)                     \
      static T_ wrap(T t) {                             \
        T_ t_;                                          \
        t_.lo = wrap(PT(t.lo));                         \
        t_.hi = wrap(PT(t.hi));                         \
        return t_;                                      \
      }                                                 \
      static T unwrap(T_ t_) {                          \
        T t(unwrap(t_.lo), unwrap(t_.hi));              \
        return t;                                       \
      }

      typedef Rect<1,coord_t> Rect1D;
      typedef Rect<2,coord_t> Rect2D;
      typedef Rect<3,coord_t> Rect3D;
      NEW_RECT_WRAPPER(legion_rect_1d_t, Rect1D, Point1D);
      NEW_RECT_WRAPPER(legion_rect_2d_t, Rect2D, Point2D);
      NEW_RECT_WRAPPER(legion_rect_3d_t, Rect3D, Point3D);
#undef NEW_RECT_WRAPPER 

#define NEW_BLOCKIFY_WRAPPER(T_, T)                     \
      static T unwrap(T_ t_) {                          \
        T t(unwrap(t_.block_size), unwrap(t_.offset));  \
        return t;                                       \
      }
      template<int DIM>
      struct Blockify {
      public:
        Blockify(Point<DIM,coord_t> b, 
                 Point<DIM,coord_t> o)
          : block_size(b), offset(o) { }
      public:
        Point<DIM,coord_t> block_size, offset;
      };

      NEW_BLOCKIFY_WRAPPER(legion_blockify_1d_t, Blockify<1>);
      NEW_BLOCKIFY_WRAPPER(legion_blockify_2d_t, Blockify<2>);
      NEW_BLOCKIFY_WRAPPER(legion_blockify_3d_t, Blockify<3>);
#undef NEW_RECT_WRAPPER
      static legion_domain_t
      wrap(Domain domain) {
        legion_domain_t domain_;
        domain_.is_id = domain.is_id;
        domain_.dim = domain.dim;
        std::copy(domain.rect_data, domain.rect_data + 2 * MAX_RECT_DIM, domain_.rect_data);
        return domain_;
      }

      static Domain
      unwrap(legion_domain_t domain_) {
        Domain domain;
        domain.is_id = domain_.is_id;
        domain.dim = domain_.dim;
        std::copy(domain_.rect_data, domain_.rect_data + 2 * MAX_RECT_DIM, domain.rect_data);
        return domain;
      }

      static legion_domain_point_t
      wrap(DomainPoint dp) {
        legion_domain_point_t dp_;
        dp_.dim = dp.dim;
        std::copy(dp.point_data, dp.point_data + MAX_POINT_DIM, dp_.point_data);
        return dp_;
      }

      static DomainPoint
      unwrap(legion_domain_point_t dp_) {
        DomainPoint dp;
        dp.dim = dp_.dim;
        std::copy(dp_.point_data, dp_.point_data + MAX_POINT_DIM, dp.point_data);
        return dp;
      }

      static legion_domain_transform_t
      wrap(DomainTransform transform) {
        legion_domain_transform_t transform_;
        transform_.m = transform.m;
        transform_.n = transform.n;
        std::copy(transform.matrix, transform.matrix + MAX_POINT_DIM * MAX_POINT_DIM, transform_.matrix);
        return transform_;
      }

      static DomainTransform
      unwrap(legion_domain_transform_t transform_) {
        DomainTransform transform;
        transform.m = transform_.m;
        transform.n = transform_.n;
        std::copy(transform_.matrix, transform_.matrix + MAX_POINT_DIM * MAX_POINT_DIM, transform.matrix);
        return transform;
      }

      static legion_domain_affine_transform_t
      wrap(DomainAffineTransform transform) {
        legion_domain_affine_transform_t transform_;
        transform_.transform = wrap(transform.transform);
        transform_.offset = wrap(transform.offset);
        return transform_;
      }

      static DomainAffineTransform
      unwrap(legion_domain_affine_transform_t transform_) {
        DomainAffineTransform transform;
        transform.transform = unwrap(transform_.transform);
        transform.offset = unwrap(transform_.offset);
        return transform;
      }

      static legion_index_space_t
      wrap(IndexSpace is)
      {
        legion_index_space_t is_;
        is_.id = is.id;
        is_.tid = is.tid;
        is_.type_tag = is.type_tag;
        return is_;
      }

      static IndexSpace
      unwrap(legion_index_space_t is_)
      {
        IndexSpace is;
        is.id = is_.id;
        is.tid = is_.tid;
        is.type_tag = is_.type_tag;
        return is;
      }

      static legion_index_partition_t
      wrap(IndexPartition ip)
      {
        legion_index_partition_t ip_;
        ip_.id = ip.id;
        ip_.tid = ip.tid;
        ip_.type_tag = ip.type_tag;
        return ip_;
      }

      static IndexPartition
      unwrap(legion_index_partition_t ip_)
      {
        IndexPartition ip;
        ip.id = ip_.id;
        ip.tid = ip_.tid;
        ip.type_tag = ip_.type_tag;
        return ip;
      }

      static legion_field_space_t
      wrap(FieldSpace fs)
      {
        legion_field_space_t fs_;
        fs_.id = fs.id;
        return fs_;
      }

      static FieldSpace
      unwrap(legion_field_space_t fs_)
      {
        FieldSpace fs(fs_.id);
        return fs;
      }

      static legion_logical_region_t
      wrap(LogicalRegion r)
      {
        legion_logical_region_t r_;
        r_.tree_id = r.tree_id;
        r_.index_space = wrap(r.index_space);
        r_.field_space = wrap(r.field_space);
        return r_;
      }

      static LogicalRegion
      unwrap(legion_logical_region_t r_)
      {
        LogicalRegion r(r_.tree_id,
                        unwrap(r_.index_space),
                        unwrap(r_.field_space));
        return r;
      }

      static legion_logical_partition_t
      wrap(LogicalPartition r)
      {
        legion_logical_partition_t r_;
        r_.tree_id = r.tree_id;
        r_.index_partition = wrap(r.index_partition);
        r_.field_space = wrap(r.field_space);
        return r_;
      }

      static LogicalPartition
      unwrap(legion_logical_partition_t r_)
      {
        LogicalPartition r(r_.tree_id,
                           unwrap(r_.index_partition),
                           unwrap(r_.field_space));
        return r;
      }

      static legion_task_argument_t
      wrap(TaskArgument arg)
      {
        legion_task_argument_t arg_;
        arg_.args = arg.get_ptr();
        arg_.arglen = arg.get_size();
        return arg_;
      }

      static TaskArgument
      unwrap(legion_task_argument_t arg_)
      {
        return TaskArgument(arg_.args, arg_.arglen);
      }

      static const legion_byte_offset_t
      wrap(const ptrdiff_t offset)
      {
        legion_byte_offset_t offset_;
        offset_.offset = offset;
        return offset_;
      }

      static ptrdiff_t
      unwrap(const legion_byte_offset_t offset_)
      {
        return offset_.offset;
      }

      static const legion_input_args_t
      wrap_const(const InputArgs arg)
      {
        legion_input_args_t arg_;
        arg_.argv = arg.argv;
        arg_.argc = arg.argc;
        return arg_;
      }

      static const InputArgs
      unwrap_const(const legion_input_args_t args_)
      {
        InputArgs args;
        args.argv = args_.argv;
        args.argc = args_.argc;
        return args;
      }

      static legion_task_config_options_t
      wrap(TaskConfigOptions options)
      {
        legion_task_config_options_t options_;
        options_.leaf = options.leaf;
        options_.inner = options.inner;
        options_.idempotent = options.idempotent;
        return options_;
      }

      static TaskConfigOptions
      unwrap(legion_task_config_options_t options_)
      {
        TaskConfigOptions options(options_.leaf,
                                  options_.inner,
                                  options_.idempotent);
        return options;
      }

      static legion_processor_t
      wrap(Processor proc)
      {
        legion_processor_t proc_;
        proc_.id = proc.id;
        return proc_;
      }

      static Processor
      unwrap(legion_processor_t proc_)
      {
        Processor proc;
        proc.id = proc_.id;
        return proc;
      }

      static legion_processor_kind_t
      wrap(Processor::Kind options)
      {
        return static_cast<legion_processor_kind_t>(options);
      }

      static Processor::Kind
      unwrap(legion_processor_kind_t options_)
      {
        return static_cast<Processor::Kind>(options_);
      }

      static legion_memory_t
      wrap(Memory mem)
      {
        legion_memory_t mem_;
        mem_.id = mem.id;
        return mem_;
      }

      static Memory
      unwrap(legion_memory_t mem_)
      {
        Memory mem;
        mem.id = mem_.id;
        return mem;
      }

      static legion_memory_kind_t
      wrap(Memory::Kind options)
      {
        return static_cast<legion_memory_kind_t>(options);
      }

      static Memory::Kind
      unwrap(legion_memory_kind_t options_)
      {
        return static_cast<Memory::Kind>(options_);
      }

      static legion_task_slice_t
      wrap(Mapping::Mapper::TaskSlice task_slice) {
        legion_task_slice_t task_slice_;
        task_slice_.domain = wrap(task_slice.domain);
        task_slice_.proc = wrap(task_slice.proc);
        task_slice_.recurse = task_slice.recurse;
        task_slice_.stealable = task_slice.stealable;
        return task_slice_;
      }

      static Mapping::Mapper::TaskSlice
      unwrap(legion_task_slice_t task_slice_) {
        Mapping::Mapper::TaskSlice task_slice;
            task_slice.domain = unwrap(task_slice_.domain);
            task_slice.proc = unwrap(task_slice_.proc);
            task_slice.recurse = task_slice_.recurse;
            task_slice.stealable = task_slice_.stealable;
        return task_slice;
      }

      static legion_phase_barrier_t
      wrap(PhaseBarrier barrier) {
        legion_phase_barrier_t barrier_;
        barrier_.id = barrier.get_barrier().id;
        barrier_.timestamp = barrier.get_barrier().timestamp;
        return barrier_;
      }

      static PhaseBarrier
      unwrap(legion_phase_barrier_t barrier_) {
        PhaseBarrier barrier;
        barrier.phase_barrier.id = barrier_.id;
        barrier.phase_barrier.timestamp = barrier_.timestamp;
        return barrier;
      }

      static legion_dynamic_collective_t
      wrap(DynamicCollective collective) {
        legion_dynamic_collective_t collective_;
        collective_.id = collective.get_barrier().id;
        collective_.timestamp = collective.get_barrier().timestamp;
        collective_.redop = collective.redop;
        return collective_;
      }

      static DynamicCollective
      unwrap(legion_dynamic_collective_t collective_) {
        DynamicCollective collective;
        collective.phase_barrier.id = collective_.id;
        collective.phase_barrier.timestamp = collective_.timestamp;
        collective.redop = collective_.redop;
        return collective;
      }

      static legion_task_options_t
      wrap(Mapping::Mapper::TaskOptions& options) {
        legion_task_options_t options_;
        options_.initial_proc = CObjectWrapper::wrap(options.initial_proc);
        options_.inline_task = options.inline_task;
        options_.stealable = options.stealable;
        options_.map_locally = options.map_locally;
        options_.parent_priority = options.parent_priority;
        return options_;
      }

      static Mapping::Mapper::TaskOptions
      unwrap(legion_task_options_t& options_) {
        Mapping::Mapper::TaskOptions options;
        options.initial_proc = CObjectWrapper::unwrap(options_.initial_proc);
        options.inline_task = options_.inline_task;
        options.stealable = options_.stealable;
        options.map_locally = options_.map_locally;
        options.parent_priority = options_.parent_priority;
        return options;
      }

      static legion_slice_task_input_t
      wrap(Mapping::Mapper::SliceTaskInput& input) {
        legion_slice_task_input_t input_;
        input_.domain = CObjectWrapper::wrap(input.domain);
        return input_;
      }

      static legion_slice_task_input_t
      wrap_const(const Mapping::Mapper::SliceTaskInput& input) {
        legion_slice_task_input_t input_;
        input_.domain = CObjectWrapper::wrap(input.domain);
        return input_;
      }

      static Mapping::Mapper::SliceTaskInput
      unwrap(legion_slice_task_input_t& input_) {
        Mapping::Mapper::SliceTaskInput input;
        input.domain = CObjectWrapper::unwrap(input_.domain);
        return input;
      }
    };

    class CContext {
    public:
      CContext(Context _ctx)
	: ctx(_ctx)
      {}

      CContext(Context _ctx, const std::vector<PhysicalRegion>& _physical_regions)
	: ctx(_ctx)
	, physical_regions(_physical_regions.size())
      {
	for (size_t i = 0; i < _physical_regions.size(); i++) {
	  physical_regions[i] =
            CObjectWrapper::wrap(new PhysicalRegion(_physical_regions[i]));
	}
      }

      ~CContext(void)
      {
	for (size_t i = 0; i < physical_regions.size(); i++) {
          delete CObjectWrapper::unwrap(physical_regions[i]);
	}
      }

      Context context(void) const
      {
	return ctx;
      }

      const legion_physical_region_t *regions(void) const
      {
	if(physical_regions.empty())
	  return 0;
	else
	  return &physical_regions[0];
      }

      size_t num_regions(void) const
      {
	return physical_regions.size();
      }

    protected:
      Context ctx;
      std::vector<legion_physical_region_t> physical_regions;
    };

};

#endif // __LEGION_C_UTIL_H__
