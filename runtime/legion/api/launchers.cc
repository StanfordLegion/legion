/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "legion/api/launchers.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // StaticDependence
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  StaticDependence::StaticDependence(void)
    : previous_offset(0), previous_req_index(0), current_req_index(0),
      dependence_type(LEGION_NO_DEPENDENCE), validates(false), shard_only(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  StaticDependence::StaticDependence(
      unsigned prev, unsigned prev_req, unsigned current_req,
      DependenceType dtype, bool val, bool shard)
    : previous_offset(prev), previous_req_index(prev_req),
      current_req_index(current_req), dependence_type(dtype), validates(val),
      shard_only(shard)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // TaskLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  TaskLauncher::TaskLauncher(void)
    : task_id(0), argument(UntypedBuffer()), predicate(Predicate::TRUE_PRED),
      map_id(0), tag(0), point(DomainPoint(0)),
      sharding_space(IndexSpace::NO_SPACE), static_dependences(nullptr),
      enable_inlining(false), local_function_task(false),
      independent_requirements(false), elide_future_return(false),
      silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  TaskLauncher::TaskLauncher(
      TaskID tid, UntypedBuffer arg, Predicate pred /*= Predicate::TRUE_PRED*/,
      MapperID mid /*=0*/, MappingTagID t /*=0*/,
      UntypedBuffer marg /*=UntypedBuffer*/,
      const char* prov /*=UntypedBuffer*/)
    : task_id(tid), argument(arg), predicate(pred), map_id(mid), tag(t),
      map_arg(marg), point(DomainPoint(0)),
      sharding_space(IndexSpace::NO_SPACE), provenance(prov),
      static_dependences(nullptr), enable_inlining(false),
      local_function_task(false), independent_requirements(false),
      elide_future_return(false), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // IndexTaskLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  IndexTaskLauncher::IndexTaskLauncher(void)
    : task_id(0), launch_domain(Domain::NO_DOMAIN),
      launch_space(IndexSpace::NO_SPACE), sharding_space(IndexSpace::NO_SPACE),
      global_arg(UntypedBuffer()), argument_map(ArgumentMap()),
      predicate(Predicate::TRUE_PRED), concurrent_functor(0), concurrent(false),
      must_parallelism(false), map_id(0), tag(0), static_dependences(nullptr),
      enable_inlining(false), independent_requirements(false),
      silence_warnings(false), elide_future_return(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexTaskLauncher::IndexTaskLauncher(
      TaskID tid, Domain dom, UntypedBuffer global, ArgumentMap map,
      Predicate pred /*= Predicate::TRUE_PRED*/, bool must /*=false*/,
      MapperID mid /*=0*/, MappingTagID t /*=0*/, UntypedBuffer marg,
      const char* prov)
    : task_id(tid), launch_domain(dom), launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE), global_arg(global),
      argument_map(map), predicate(pred), concurrent_functor(0),
      concurrent(false), must_parallelism(must), map_id(mid), tag(t),
      map_arg(marg), provenance(prov), static_dependences(nullptr),
      enable_inlining(false), independent_requirements(false),
      silence_warnings(false), elide_future_return(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexTaskLauncher::IndexTaskLauncher(
      TaskID tid, IndexSpace space, UntypedBuffer global, ArgumentMap map,
      Predicate pred /*= Predicate::TRUE_PRED*/, bool must /*=false*/,
      MapperID mid /*=0*/, MappingTagID t /*=0*/, UntypedBuffer marg,
      const char* prov)
    : task_id(tid), launch_domain(Domain::NO_DOMAIN), launch_space(space),
      sharding_space(IndexSpace::NO_SPACE), global_arg(global),
      argument_map(map), predicate(pred), concurrent_functor(0),
      concurrent(false), must_parallelism(must), map_id(mid), tag(t),
      map_arg(marg), provenance(prov), static_dependences(nullptr),
      enable_inlining(false), independent_requirements(false),
      silence_warnings(false), elide_future_return(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // InlineLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  InlineLauncher::InlineLauncher(void)
    : map_id(0), tag(0), layout_constraint_id(0), static_dependences(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  InlineLauncher::InlineLauncher(
      const RegionRequirement& req, MapperID mid /*=0*/, MappingTagID t /*=0*/,
      LayoutConstraintID lay_id /*=0*/,
      UntypedBuffer marg /*= UntypedBuffer()*/,
      const char* prov /*= UntypedBuffer()*/)
    : requirement(req), map_id(mid), tag(t), map_arg(marg),
      layout_constraint_id(lay_id), provenance(prov),
      static_dependences(nullptr)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // CopyLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  CopyLauncher::CopyLauncher(
      Predicate pred /*= Predicate::TRUE_PRED*/, MapperID mid /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=nullptr*/)
    : predicate(pred), map_id(mid), tag(t), map_arg(marg),
      point(DomainPoint(0)), sharding_space(IndexSpace::NO_SPACE),
      provenance(prov), static_dependences(nullptr),
      possible_src_indirect_out_of_range(true),
      possible_dst_indirect_out_of_range(true),
      possible_dst_indirect_aliasing(true), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // IndexCopyLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  IndexCopyLauncher::IndexCopyLauncher(void)
    : launch_domain(Domain::NO_DOMAIN), launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE), predicate(Predicate::TRUE_PRED),
      map_id(0), tag(0), static_dependences(nullptr),
      possible_src_indirect_out_of_range(true),
      possible_dst_indirect_out_of_range(true),
      possible_dst_indirect_aliasing(true),
      collective_src_indirect_points(true),
      collective_dst_indirect_points(true), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexCopyLauncher::IndexCopyLauncher(
      Domain dom, Predicate pred /*= Predicate::TRUE_PRED*/,
      MapperID mid /*=0*/, MappingTagID t /*=0*/,
      UntypedBuffer marg /*=UntypedBuffer()*/, const char* prov /*=nullptr*/)
    : launch_domain(dom), launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE), predicate(pred), map_id(mid),
      tag(t), map_arg(marg), provenance(prov), static_dependences(nullptr),
      possible_src_indirect_out_of_range(true),
      possible_dst_indirect_out_of_range(true),
      possible_dst_indirect_aliasing(true),
      collective_src_indirect_points(true),
      collective_dst_indirect_points(true), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexCopyLauncher::IndexCopyLauncher(
      IndexSpace space, Predicate pred /*= Predicate::TRUE_PRED*/,
      MapperID mid /*=0*/, MappingTagID t /*=0*/,
      UntypedBuffer marg /*=UntypedBuffer()*/, const char* prov /*=nullptr*/)
    : launch_domain(Domain::NO_DOMAIN), launch_space(space),
      sharding_space(IndexSpace::NO_SPACE), predicate(pred), map_id(mid),
      tag(t), map_arg(marg), provenance(prov), static_dependences(nullptr),
      possible_src_indirect_out_of_range(true),
      possible_dst_indirect_out_of_range(true),
      possible_dst_indirect_aliasing(true),
      collective_src_indirect_points(true),
      collective_dst_indirect_points(true), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // AcquireLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  AcquireLauncher::AcquireLauncher(
      LogicalRegion reg, LogicalRegion par, PhysicalRegion phy,
      Predicate pred /*= Predicate::TRUE_PRED*/, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=UntypedBuffer()*/)
    : logical_region(reg), parent_region(par), physical_region(phy),
      predicate(pred), map_id(id), tag(t), map_arg(marg), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // ReleaseLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  ReleaseLauncher::ReleaseLauncher(
      LogicalRegion reg, LogicalRegion par, PhysicalRegion phy,
      Predicate pred /*= Predicate::TRUE_PRED*/, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=UntypedBuffer()*/)
    : logical_region(reg), parent_region(par), physical_region(phy),
      predicate(pred), map_id(id), tag(t), map_arg(marg), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // FillLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  FillLauncher::FillLauncher(void)
    : handle(LogicalRegion::NO_REGION), parent(LogicalRegion::NO_REGION),
      map_id(0), tag(0), point(DomainPoint(0)), static_dependences(nullptr),
      silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  FillLauncher::FillLauncher(
      LogicalRegion h, LogicalRegion p, UntypedBuffer arg,
      Predicate pred /*= Predicate::TRUE_PRED*/, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=UntypedBuffer()*/)
    : handle(h), parent(p), argument(arg), predicate(pred), map_id(id), tag(t),
      map_arg(marg), point(DomainPoint(0)), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  FillLauncher::FillLauncher(
      LogicalRegion h, LogicalRegion p, Future f,
      Predicate pred /*= Predicate::TRUE_PRED*/, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=UntypedBuffer()*/)
    : handle(h), parent(p), future(f), predicate(pred), map_id(id), tag(t),
      map_arg(marg), point(DomainPoint(0)), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // IndexFillLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(void)
    : launch_domain(Domain::NO_DOMAIN), launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE), region(LogicalRegion::NO_REGION),
      partition(LogicalPartition::NO_PART), projection(0), map_id(0), tag(0),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      Domain dom, LogicalRegion h, LogicalRegion p, UntypedBuffer arg,
      ProjectionID proj, Predicate pred, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=nullptr*/)
    : launch_domain(dom), launch_space(IndexSpace::NO_SPACE),
      partition(LogicalPartition::NO_PART), parent(p), projection(proj),
      argument(arg), predicate(pred), map_id(id), tag(t), map_arg(marg),
      provenance(prov), static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      Domain dom, LogicalRegion h, LogicalRegion p, Future f, ProjectionID proj,
      Predicate pred, MapperID id /*=0*/, MappingTagID t /*=0*/,
      UntypedBuffer marg /*=UntypedBuffer()*/, const char* prov /*=nullptr*/)
    : launch_domain(dom), launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE), region(h),
      partition(LogicalPartition::NO_PART), parent(p), projection(proj),
      future(f), predicate(pred), map_id(id), tag(t), map_arg(marg),
      provenance(prov), static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      IndexSpace space, LogicalRegion h, LogicalRegion p, UntypedBuffer arg,
      ProjectionID proj, Predicate pred, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=nullptr*/)
    : launch_domain(Domain::NO_DOMAIN), launch_space(space),
      sharding_space(IndexSpace::NO_SPACE), region(h),
      partition(LogicalPartition::NO_PART), parent(p), projection(proj),
      argument(arg), predicate(pred), map_id(id), tag(t), map_arg(marg),
      provenance(prov), static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      IndexSpace space, LogicalRegion h, LogicalRegion p, Future f,
      ProjectionID proj, Predicate pred, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=nullptr*/)
    : launch_domain(Domain::NO_DOMAIN), launch_space(space),
      sharding_space(IndexSpace::NO_SPACE), region(h),
      partition(LogicalPartition::NO_PART), parent(p), projection(proj),
      future(f), predicate(pred), map_id(id), tag(t), map_arg(marg),
      provenance(prov), static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      Domain dom, LogicalPartition h, LogicalRegion p, UntypedBuffer arg,
      ProjectionID proj, Predicate pred, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=nullptr*/)
    : launch_domain(dom), launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE), region(LogicalRegion::NO_REGION),
      partition(h), parent(p), projection(proj), argument(arg), predicate(pred),
      map_id(id), tag(t), map_arg(marg), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      Domain dom, LogicalPartition h, LogicalRegion p, Future f,
      ProjectionID proj, Predicate pred, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=UntypedBuffer()*/)
    : launch_domain(dom), launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE), region(LogicalRegion::NO_REGION),
      partition(h), parent(p), projection(proj), future(f), predicate(pred),
      map_id(id), tag(t), map_arg(marg), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      IndexSpace space, LogicalPartition h, LogicalRegion p, UntypedBuffer arg,
      ProjectionID proj, Predicate pred, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=nullptr*/)
    : launch_domain(Domain::NO_DOMAIN), launch_space(space),
      sharding_space(IndexSpace::NO_SPACE), region(LogicalRegion::NO_REGION),
      partition(h), parent(p), projection(proj), argument(arg), predicate(pred),
      map_id(id), tag(t), map_arg(marg), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexFillLauncher::IndexFillLauncher(
      IndexSpace space, LogicalPartition h, LogicalRegion p, Future f,
      ProjectionID proj, Predicate pred, MapperID id /*=0*/,
      MappingTagID t /*=0*/, UntypedBuffer marg /*=UntypedBuffer()*/,
      const char* prov /*=nullptr*/)
    : launch_domain(Domain::NO_DOMAIN), launch_space(space),
      sharding_space(IndexSpace::NO_SPACE), region(LogicalRegion::NO_REGION),
      partition(h), parent(p), projection(proj), future(f), predicate(pred),
      map_id(id), tag(t), map_arg(marg), provenance(prov),
      static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // DiscardLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  DiscardLauncher::DiscardLauncher(LogicalRegion h, LogicalRegion p)
    : handle(h), parent(p), static_dependences(nullptr), silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

  LEGION_DISABLE_DEPRECATED_WARNINGS

  /////////////////////////////////////////////////////////////
  // AttachLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  AttachLauncher::AttachLauncher(
      ExternalResource r, LogicalRegion h, LogicalRegion p,
      const bool restr /*= true*/, const bool map /*= true*/)
    : resource(r), parent(p), handle(h), external_resource(nullptr),
      restricted(restr), mapped(map),
      collective((r == LEGION_EXTERNAL_INSTANCE) ? true : false),
      deduplicate_across_shards(false), file_name(nullptr),
      mode(LEGION_FILE_READ_ONLY), footprint(0), static_dependences(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  AttachLauncher::~AttachLauncher(void)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // IndexAttachLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  IndexAttachLauncher::IndexAttachLauncher(
      ExternalResource r, LogicalRegion p, const bool restr)
    : resource(r), parent(p), restricted(restr),
      deduplicate_across_shards(false), mode(LEGION_FILE_READ_ONLY),
      static_dependences(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexAttachLauncher::~IndexAttachLauncher(void)
  //--------------------------------------------------------------------------
  { }

  LEGION_REENABLE_DEPRECATED_WARNINGS

  /////////////////////////////////////////////////////////////
  // PredicateLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  PredicateLauncher::PredicateLauncher(bool and_) : and_op(and_)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // TimingLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  TimingLauncher::TimingLauncher(TimingMeasurement m) : measurement(m)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // TunableLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  TunableLauncher::TunableLauncher(
      TunableID tid, MapperID m, MappingTagID t, size_t return_size)
    : tunable(tid), mapper(m), tag(t), return_type_size(return_size)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // MustEpochLauncher
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  MustEpochLauncher::MustEpochLauncher(
      MapperID id /*= 0*/, MappingTagID tag /*= 0*/)
    : map_id(id), mapping_tag(tag), launch_domain(Domain::NO_DOMAIN),
      launch_space(IndexSpace::NO_SPACE), sharding_space(IndexSpace::NO_SPACE),
      silence_warnings(false)
  //--------------------------------------------------------------------------
  { }

}  // namespace Legion
