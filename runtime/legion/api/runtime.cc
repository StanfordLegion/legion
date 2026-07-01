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

#include "legion/api/runtime.h"
#include "legion/api/argument_map_impl.h"
#include "legion/api/future_impl.h"
#include "legion/contexts/context.h"
#include "legion/kernel/runtime.h"
#include "legion/tasks/single.h"
#include "legion/utilities/provenance.h"

namespace Legion {

  // Cache static type tags so we don't need to recompute them all the time
#define DIMFUNC(DIM)                       \
  static const TypeTag TYPE_TAG_##DIM##D = \
      Internal::NT_TemplateHelper::encode_tag<DIM, coord_t>();
  LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC

  /////////////////////////////////////////////////////////////
  // Auto Call
  /////////////////////////////////////////////////////////////

  /**
   * This class helps with recording runtime calls for profiling
   * and also will check that the user has passed in the correct
   * context when they pass in a non-nullptr one.
   */
  template<Internal::RuntimeCallKind KIND>
  class AutoCall : public Internal::AutoProvenance,
                   public Internal::ImplicitReferenceTracker {
  public:
    // no string versions
    inline AutoCall(void)
      : AutoProvenance(),
        start(
            (Internal::implicit_context == nullptr) ?
                0 :
            Internal::implicit_context->begin_runtime_call(KIND, *this) ?
                Realm::Clock::current_time_in_nanoseconds() :
                0),
        previous_provenance(Internal::implicit_provenance)
    { }
    inline AutoCall(Internal::TaskContext* ctx, const char* func)
      : AutoProvenance(),
        start(
            (Internal::implicit_context == nullptr) ?
                0 :
            Internal::implicit_context->begin_runtime_call(KIND, *this) ?
                Realm::Clock::current_time_in_nanoseconds() :
                0),
        previous_provenance(Internal::implicit_provenance)
    {
      legion_assert(Internal::implicit_provenance == 0);
      if (provenance != nullptr)
        Internal::implicit_provenance = provenance->pid;
      if (ctx != Internal::implicit_context)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid task context passed to runtime call " << func;
        error.raise();
      }
    }
    // C string versions
    inline AutoCall(const char* prov)
      : AutoProvenance(prov),
        start(
            (Internal::implicit_context == nullptr) ?
                0 :
            Internal::implicit_context->begin_runtime_call(KIND, *this) ?
                Realm::Clock::current_time_in_nanoseconds() :
                0),
        previous_provenance(Internal::implicit_provenance)
    { }
    inline AutoCall(
        const char* prov, Internal::TaskContext* ctx, const char* func)
      : AutoProvenance(prov),
        start(
            (Internal::implicit_context == nullptr) ?
                0 :
            Internal::implicit_context->begin_runtime_call(KIND, *this) ?
                Realm::Clock::current_time_in_nanoseconds() :
                0),
        previous_provenance(Internal::implicit_provenance)
    {
      legion_assert(Internal::implicit_provenance == 0);
      if (provenance != nullptr)
        Internal::implicit_provenance = provenance->pid;
      if (ctx != Internal::implicit_context)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid task context passed to runtime call " << func;
        error.raise();
      }
    }
    // std::string versions
    inline AutoCall(const std::string& prov)
      : AutoProvenance(prov),
        start(
            (Internal::implicit_context == nullptr) ?
                0 :
            Internal::implicit_context->begin_runtime_call(KIND, *this) ?
                Realm::Clock::current_time_in_nanoseconds() :
                0),
        previous_provenance(Internal::implicit_provenance)
    { }
    inline AutoCall(
        const std::string& prov, Internal::TaskContext* ctx, const char* func)
      : AutoProvenance(prov),
        start(
            (Internal::implicit_context == nullptr) ?
                0 :
            Internal::implicit_context->begin_runtime_call(KIND, *this) ?
                Realm::Clock::current_time_in_nanoseconds() :
                0),
        previous_provenance(Internal::implicit_provenance)
    {
      if (provenance != nullptr)
        Internal::implicit_provenance = provenance->pid;
      if (ctx != Internal::implicit_context)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid task context passed to runtime call " << func;
        error.raise();
      }
    }
    inline ~AutoCall(void)
    {
      if (Internal::implicit_context != nullptr)
      {
        if (start == 0)
          Internal::implicit_context->end_runtime_call(KIND, *this, start, 0);
        else
          Internal::implicit_context->end_runtime_call(
              KIND, *this, start, Realm::Clock::current_time_in_nanoseconds());
      }
      // Reset the implicit provenance back to the way it was
      Internal::implicit_provenance = previous_provenance;
    }
  private:
    const unsigned long long start;
    const ProvenanceID previous_provenance;
  };

  /////////////////////////////////////////////////////////////
  // Task Config Options
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  TaskConfigOptions::TaskConfigOptions(
      bool l /*=false*/, bool in /*=false*/, bool idem /*=false*/)
    : leaf(l), inner(in), idempotent(idem)
  //--------------------------------------------------------------------------
  { }

  /////////////////////////////////////////////////////////////
  // Legion Runtime
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  Runtime::Runtime(Internal::Runtime* rt) : runtime(rt)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space(Context ctx, size_t max_num_elmts)
  //--------------------------------------------------------------------------
  {
    const Rect<1, coord_t> bounds(
        (Point<1, coord_t>(0)), (Point<1, coord_t>(max_num_elmts - 1)));
    const Domain domain(bounds);
    return create_index_space(ctx, domain, TYPE_TAG_1D, nullptr /*provenance*/);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space(
      Context ctx, const Domain& domain, TypeTag type_tag, const char* prov,
      bool take_ownership)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_CALL> call(
        prov, ctx, __func__);
    if (type_tag == 0)
    {
      switch (domain.get_dim())
      {
#define DIMFUNC(DIM)                \
  case DIM:                         \
    {                               \
      type_tag = TYPE_TAG_##DIM##D; \
      break;                        \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error
                << "Unsupported dimension " << domain.get_dim()
                << " for Runtime::create_index_space. This probably means you "
                << "need to build Legion with support for more dimensions.";
            error.raise();
          }
      }
    }
    return ctx->create_index_space(domain, take_ownership, type_tag, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space(
      Context ctx, size_t dimensions, const Future& future, TypeTag type_tag,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_CALL> call(
        prov, ctx, __func__);
    if (type_tag == 0)
    {
      switch (dimensions)
      {
#define DIMFUNC(DIM)                \
  case DIM:                         \
    {                               \
      type_tag = TYPE_TAG_##DIM##D; \
      break;                        \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error
                << "Unsupported dimension " << dimensions
                << " for Runtime::create_index_space. This probably means you "
                << "need to build Legion with support for more dimensions.";
            error.raise();
          }
      }
    }
    return ctx->create_index_space(future, type_tag, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space(
      Context ctx, const std::set<Domain>& domains)
  //--------------------------------------------------------------------------
  {
    std::vector<Domain> rects(domains.begin(), domains.end());
    return create_index_space(ctx, rects, nullptr /*provenance*/);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space(
      Context ctx, const std::vector<DomainPoint>& points, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_index_space(points, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space(
      Context ctx, const std::vector<Domain>& rects, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_index_space(rects, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::union_index_spaces(
      Context ctx, const std::vector<IndexSpace>& spaces, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_UNION_INDEX_SPACES_CALL> call(
        prov, ctx, __func__);
    return ctx->union_index_spaces(spaces, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::intersect_index_spaces(
      Context ctx, const std::vector<IndexSpace>& spaces, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_INTERSECT_INDEX_SPACES_CALL> call(
        prov, ctx, __func__);
    return ctx->intersect_index_spaces(spaces, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::subtract_index_spaces(
      Context ctx, IndexSpace left, IndexSpace right, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_SUBTRACT_INDEX_SPACES_CALL> call(
        prov, ctx, __func__);
    return ctx->subtract_index_spaces(left, right, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::create_shared_ownership(Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_SHARED_OWNERSHIP_CALL> call(
        ctx, __func__);
    ctx->create_shared_ownership(handle);
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_index_space(
      Context ctx, IndexSpace handle, const bool unordered, const bool recurse,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DESTROY_INDEX_SPACE_CALL> call(
        prov, ctx, __func__);
    ctx->destroy_index_space(handle, unordered, recurse, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::create_shared_ownership(Context ctx, IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_SHARED_OWNERSHIP_CALL> call(
        ctx, __func__);
    ctx->create_shared_ownership(handle);
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_index_partition(
      Context ctx, IndexPartition handle, const bool unordered,
      const bool recurse, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DESTROY_INDEX_SPACE_CALL> call(
        prov, ctx, __func__);
    ctx->destroy_index_partition(handle, unordered, recurse, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_equal_partition(
      Context ctx, IndexSpace parent, IndexSpace color_space,
      size_t granularity, Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_EQUAL_PARTITION_CALL> call(
        prov, ctx, __func__);
    return ctx->create_equal_partition(
        parent, color_space, granularity, color, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_weights(
      Context ctx, IndexSpace parent, const std::map<DomainPoint, int>& weights,
      IndexSpace color_space, size_t granularity, Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_WEIGHTS_CALL> call(
        prov, ctx, __func__);
    std::map<DomainPoint, UntypedBuffer> data;
    for (const std::pair<const DomainPoint, int>& weight_pair : weights)
      data.emplace(std::make_pair(
          weight_pair.first,
          UntypedBuffer(&weight_pair.second, sizeof(weight_pair.second))));
    FutureMap future_map = construct_future_map(
        ctx, color_space, data, false /*collective*/, 0 /*sid*/,
        false /*implicit*/, prov);
    return ctx->create_partition_by_weights(
        parent, future_map, color_space, granularity, color, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_weights(
      Context ctx, IndexSpace parent,
      const std::map<DomainPoint, size_t>& weights, IndexSpace color_space,
      size_t granularity, Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_WEIGHTS_CALL> call(
        prov, ctx, __func__);
    std::map<DomainPoint, UntypedBuffer> data;
    for (const std::pair<const DomainPoint, size_t>& weight_pair : weights)
      data.emplace(std::make_pair(
          weight_pair.first,
          UntypedBuffer(&weight_pair.second, sizeof(weight_pair.second))));
    FutureMap future_map = construct_future_map(
        ctx, color_space, data, false /*collective*/, 0 /*sid*/,
        false /*implicit*/, prov);
    return ctx->create_partition_by_weights(
        parent, future_map, color_space, granularity, color, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_weights(
      Context ctx, IndexSpace parent, const FutureMap& weights,
      IndexSpace color_space, size_t granularity, Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_WEIGHTS_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_weights(
        parent, weights, color_space, granularity, color, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_union(
      Context ctx, IndexSpace parent, IndexPartition handle1,
      IndexPartition handle2, IndexSpace color_space, PartitionKind kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_UNION_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_union(
        parent, handle1, handle2, color_space, kind, color, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_intersection(
      Context ctx, IndexSpace parent, IndexPartition handle1,
      IndexPartition handle2, IndexSpace color_space, PartitionKind kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_INTERSECTION_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_intersection(
        parent, handle1, handle2, color_space, kind, color, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_intersection(
      Context ctx, IndexSpace parent, IndexPartition partition,
      PartitionKind part_kind, Color color, bool dominates, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_INTERSECTION_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_intersection(
        parent, partition, part_kind, color, dominates, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_difference(
      Context ctx, IndexSpace parent, IndexPartition handle1,
      IndexPartition handle2, IndexSpace color_space, PartitionKind kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_DIFFERENCE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_difference(
        parent, handle1, handle2, color_space, kind, color, call);
  }

  //--------------------------------------------------------------------------
  Color Runtime::create_cross_product_partitions(
      Context ctx, IndexPartition handle1, IndexPartition handle2,
      std::map<IndexSpace, IndexPartition>& handles, PartitionKind kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_CROSS_PRODUCT_PARTITIONS_CALL> call(
        prov, ctx, __func__);
    return ctx->create_cross_product_partitions(
        handle1, handle2, handles, kind, color, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::create_association(
      Context ctx, LogicalRegion domain, LogicalRegion domain_parent,
      FieldID domain_fid, IndexSpace range, MapperID id, MappingTagID tag,
      UntypedBuffer marg, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_ASSOCIATION_CALL> call(
        prov, ctx, __func__);
    ctx->create_association(
        domain, domain_parent, domain_fid, range, id, tag, marg, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::create_bidirectional_association(
      Context ctx, LogicalRegion domain, LogicalRegion domain_parent,
      FieldID domain_fid, LogicalRegion range, LogicalRegion range_parent,
      FieldID range_fid, MapperID id, MappingTagID tag, UntypedBuffer marg,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    // Realm guarantees that creating association in either direction
    // will produce the same result, so we can do these separately
    create_association(
        ctx, domain, domain_parent, domain_fid, range.get_index_space(), id,
        tag, marg, provenance);
    create_association(
        ctx, range, range_parent, range_fid, domain.get_index_space(), id, tag,
        marg, provenance);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_restriction(
      Context ctx, IndexSpace par, IndexSpace cs, DomainTransform tran,
      Domain ext, PartitionKind part_kind, Color color, const char* provenance)
  //--------------------------------------------------------------------------
  {
    switch ((ext.get_dim() - 1) * LEGION_MAX_DIM + (tran.n - 1))
    {
#define DIMFUNC(D1, D2)                                                  \
  case (D1 - 1) * LEGION_MAX_DIM + (D2 - 1):                             \
    {                                                                    \
      const IndexSpaceT<D1, coord_t> parent(par);                        \
      const Rect<D1, coord_t> extent(ext);                               \
      const Transform<D1, D2> transform(tran);                           \
      const IndexSpaceT<D2, coord_t> color_space(cs);                    \
      return create_partition_by_restriction<D1, D2, coord_t>(           \
          ctx, parent, color_space, transform, extent, part_kind, color, \
          provenance);                                                   \
    }
      LEGION_FOREACH_NN(DIMFUNC)
#undef DIMFUNC
    }
    return IndexPartition::NO_PART;
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_blockify(
      Context ctx, IndexSpace par, DomainPoint bf, Color color,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    switch (bf.get_dim())
    {
#define DIMFUNC(DIM)                                        \
  case DIM:                                                 \
    {                                                       \
      const IndexSpaceT<DIM, coord_t> parent(par);          \
      const Point<DIM, coord_t> blocking_factor(bf);        \
      return create_partition_by_blockify<DIM, coord_t>(    \
          ctx, parent, blocking_factor, color, provenance); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << bf.get_dim() << " for "
                << "Runtime::create_partition_by_blockify. This probably means "
                << "you need to build Legion with support for more dimensions.";
          if (provenance != nullptr)
            error << "\nProvenance: " << provenance;
          error.raise();
        }
    }
    return IndexPartition::NO_PART;
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_blockify(
      Context ctx, IndexSpace par, DomainPoint bf, DomainPoint orig,
      Color color, const char* provenance)
  //--------------------------------------------------------------------------
  {
    switch (bf.get_dim())
    {
#define DIMFUNC(DIM)                                                \
  case DIM:                                                         \
    {                                                               \
      const IndexSpaceT<DIM, coord_t> parent(par);                  \
      const Point<DIM, coord_t> blocking_factor(bf);                \
      const Point<DIM, coord_t> origin(orig);                       \
      return create_partition_by_blockify<DIM, coord_t>(            \
          ctx, parent, blocking_factor, origin, color, provenance); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << bf.get_dim() << " for "
                << "Runtime::create_partition_by_blockify. This probably means "
                << "you need to build Legion with support for more dimensions.";
          if (provenance != nullptr)
            error << "\nProvenance: " << provenance;
          error.raise();
        }
    }
    return IndexPartition::NO_PART;
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_restricted_partition(
      Context ctx, IndexSpace parent, IndexSpace color_space,
      const void* transform, size_t transform_size, const void* extent,
      size_t extent_size, PartitionKind part_kind, Color color,
      const char* prov, const char* func, bool blockify)
  //--------------------------------------------------------------------------
  {
    if (blockify)
    {
      AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_BLOCKIFY_CALL> call(
          prov, ctx, func);
      return ctx->create_restricted_partition(
          parent, color_space, transform, transform_size, extent, extent_size,
          part_kind, color, call);
    }
    else
    {
      AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_RESTRICTION_CALL> call(
          prov, ctx, func);
      return ctx->create_restricted_partition(
          parent, color_space, transform, transform_size, extent, extent_size,
          part_kind, color, call);
    }
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_domain(
      Context ctx, IndexSpace parent,
      const std::map<DomainPoint, Domain>& domains, IndexSpace color_space,
      bool perform_intersections, PartitionKind part_kind, Color color,
      const char* prov, bool take_ownership)
  //--------------------------------------------------------------------------
  {
    // Convert this into a future map and call that version of this method
    std::map<DomainPoint, Future> futures;
    for (const std::pair<const DomainPoint, Domain>& domain_pair : domains)
      futures[domain_pair.first] =
          Future::from_domain(domain_pair.second, take_ownership);
    FutureMap fm = construct_future_map(ctx, color_space, futures);
    return create_partition_by_domain(
        ctx, parent, fm, color_space, perform_intersections, part_kind, color,
        prov);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_domain(
      Context ctx, IndexSpace parent, const FutureMap& domains,
      IndexSpace color_space, bool perform_intersections,
      PartitionKind part_kind, Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_DOMAIN_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_domain(
        parent, domains, color_space, perform_intersections, part_kind, color,
        call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_field(
      Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
      IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
      PartitionKind part_kind, UntypedBuffer marg, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_FIELD_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_field(
        handle, parent, fid, color_space, color, id, tag, part_kind, marg,
        call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_image(
      Context ctx, IndexSpace handle, LogicalPartition projection,
      LogicalRegion parent, FieldID fid, IndexSpace color_space,
      PartitionKind part_kind, Color color, MapperID id, MappingTagID tag,
      UntypedBuffer marg, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_IMAGE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_image(
        handle, projection, parent, fid, color_space, part_kind, color, id, tag,
        marg, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_image_range(
      Context ctx, IndexSpace handle, LogicalPartition projection,
      LogicalRegion parent, FieldID fid, IndexSpace color_space,
      PartitionKind part_kind, Color color, MapperID id, MappingTagID tag,
      UntypedBuffer marg, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_IMAGE_RANGE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_image_range(
        handle, projection, parent, fid, color_space, part_kind, color, id, tag,
        marg, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_preimage(
      Context ctx, IndexPartition projection, LogicalRegion handle,
      LogicalRegion parent, FieldID fid, IndexSpace color_space,
      PartitionKind part_kind, Color color, MapperID id, MappingTagID tag,
      UntypedBuffer marg, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_PREIMAGE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_preimage(
        projection, handle, parent, fid, color_space, part_kind, color, id, tag,
        marg, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_partition_by_preimage_range(
      Context ctx, IndexPartition projection, LogicalRegion handle,
      LogicalRegion parent, FieldID fid, IndexSpace color_space,
      PartitionKind part_kind, Color color, MapperID id, MappingTagID tag,
      UntypedBuffer marg, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PARTITION_BY_PREIMAGE_RANGE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_partition_by_preimage_range(
        projection, handle, parent, fid, color_space, part_kind, color, id, tag,
        marg, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::create_pending_partition(
      Context ctx, IndexSpace parent, IndexSpace color_space,
      PartitionKind part_kind, Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PENDING_PARTITION_CALL> call(
        prov, ctx, __func__);
    return ctx->create_pending_partition(
        parent, color_space, part_kind, color, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_union(
      Context ctx, IndexPartition parent, const DomainPoint& color,
      const std::vector<IndexSpace>& handles, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_UNION_CALL> call(
        prov, ctx, __func__);
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                        \
  case DIM:                                                                 \
    {                                                                       \
      Point<DIM, coord_t> point = color;                                    \
      return ctx->create_index_space_union(                                 \
          parent, &point, sizeof(point), TYPE_TAG_##DIM##D, handles, call); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::create_index_space_union. This probably means you "
                << "need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return IndexSpace::NO_SPACE;
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_union_internal(
      Context ctx, IndexPartition parent, const void* color, size_t color_size,
      TypeTag type_tag, const char* prov, const char* func_name,
      const std::vector<IndexSpace>& handles)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_UNION_CALL> call(
        prov, ctx, func_name);
    return ctx->create_index_space_union(
        parent, color, color_size, type_tag, handles, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_union(
      Context ctx, IndexPartition parent, const DomainPoint& color,
      IndexPartition handle, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_UNION_CALL> call(
        prov, ctx, __func__);
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                       \
  case DIM:                                                                \
    {                                                                      \
      Point<DIM, coord_t> point = color;                                   \
      return ctx->create_index_space_union(                                \
          parent, &point, sizeof(point), TYPE_TAG_##DIM##D, handle, call); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::create_index_space_union. This probably means you "
                << "need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return IndexSpace::NO_SPACE;
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_union_internal(
      Context ctx, IndexPartition parent, const void* realm_color, size_t size,
      TypeTag type_tag, const char* prov, const char* func,
      IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_UNION_CALL> call(
        prov, ctx, func);
    return ctx->create_index_space_union(
        parent, realm_color, size, type_tag, handle, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_intersection(
      Context ctx, IndexPartition parent, const DomainPoint& color,
      const std::vector<IndexSpace>& handles, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_INTERSECTION_CALL> call(
        prov, ctx, __func__);
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                        \
  case DIM:                                                                 \
    {                                                                       \
      Point<DIM, coord_t> point = color;                                    \
      return ctx->create_index_space_intersection(                          \
          parent, &point, sizeof(point), TYPE_TAG_##DIM##D, handles, call); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::create_index_space_intersection. This probably "
                   "means "
                << "you need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return IndexSpace::NO_SPACE;
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_intersection_internal(
      Context ctx, IndexPartition parent, const void* color, size_t color_size,
      TypeTag type_tag, const char* prov, const char* func,
      const std::vector<IndexSpace>& handles)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_INTERSECTION_CALL> call(
        prov, ctx, func);
    return ctx->create_index_space_intersection(
        parent, color, color_size, type_tag, handles, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_intersection(
      Context ctx, IndexPartition parent, const DomainPoint& color,
      IndexPartition handle, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_INTERSECTION_CALL> call(
        prov, ctx, __func__);
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                       \
  case DIM:                                                                \
    {                                                                      \
      Point<DIM, coord_t> point = color;                                   \
      return ctx->create_index_space_intersection(                         \
          parent, &point, sizeof(point), TYPE_TAG_##DIM##D, handle, call); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::create_index_space_intersection. This probably "
                   "means "
                << "you need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return IndexSpace::NO_SPACE;
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_intersection_internal(
      Context ctx, IndexPartition parent, const void* realm_color,
      size_t color_size, TypeTag type_tag, const char* prov, const char* func,
      IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_INTERSECTION_CALL> call(
        prov, ctx, func);
    return ctx->create_index_space_intersection(
        parent, realm_color, color_size, type_tag, handle, call);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_difference(
      Context ctx, IndexPartition parent, const DomainPoint& color,
      IndexSpace initial, const std::vector<IndexSpace>& handles,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_DIFFERENCE_CALL> call(
        prov, ctx, __func__);
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                          \
  case DIM:                                                                   \
    {                                                                         \
      Point<DIM, coord_t> point = color;                                      \
      return ctx->create_index_space_difference(                              \
          parent, &point, sizeof(point), TYPE_TAG_##DIM##D, initial, handles, \
          call);                                                              \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "Unsupported dimension " << color.get_dim() << " for "
              << "Runtime::create_index_space_difference. This probably means "
              << "you need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return IndexSpace::NO_SPACE;
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::create_index_space_difference_internal(
      Context ctx, IndexPartition parent, const void* realm_color,
      size_t color_size, TypeTag type_tag, const char* prov, const char* func,
      IndexSpace initial, const std::vector<IndexSpace>& handles)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_INDEX_SPACE_DIFFERENCE_CALL> call(
        prov, ctx, func);
    return ctx->create_index_space_difference(
        parent, realm_color, color_size, type_tag, initial, handles, call);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::get_index_partition(
      Context ctx, IndexSpace parent, Color color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_CALL> call(ctx, __func__);
    return runtime->get_index_partition(parent, color);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::get_index_partition(
      Context ctx, IndexSpace parent, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_CALL> call(ctx, __func__);
    return runtime->get_index_partition(parent, color.get_color());
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::get_index_partition(IndexSpace parent, Color color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_CALL> call;
    return runtime->get_index_partition(parent, color);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::get_index_partition(
      IndexSpace parent, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_CALL> call;
    return get_index_partition(parent, color.get_color());
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_index_partition(Context ctx, IndexSpace parent, Color c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_INDEX_PARTITION_CALL> call(ctx, __func__);
    return runtime->has_index_partition(parent, c);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_index_partition(
      Context ctx, IndexSpace parent, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_INDEX_PARTITION_CALL> call(ctx, __func__);
    return runtime->has_index_partition(parent, color.get_color());
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_index_partition(IndexSpace parent, Color c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_INDEX_PARTITION_CALL> call;
    return runtime->has_index_partition(parent, c);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_index_partition(IndexSpace parent, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_INDEX_PARTITION_CALL> call;
    return runtime->has_index_partition(parent, color.get_color());
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_index_subspace(
      Context ctx, IndexPartition p, Color color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SUBSPACE_CALL> call(ctx, __func__);
    Point<1, coord_t> point = color;
    return runtime->get_index_subspace(p, &point, TYPE_TAG_1D);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_index_subspace(
      Context ctx, IndexPartition p, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SUBSPACE_CALL> call(ctx, __func__);
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                    \
  case DIM:                                                             \
    {                                                                   \
      Point<DIM, coord_t> point = color;                                \
      return runtime->get_index_subspace(p, &point, TYPE_TAG_##DIM##D); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::get_index_subspace. This probably means you need "
                << "to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return IndexSpace::NO_SPACE;
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_index_subspace(IndexPartition p, Color color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SUBSPACE_CALL> call;
    Point<1, coord_t> point = color;
    return runtime->get_index_subspace(p, &point, TYPE_TAG_1D);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_index_subspace(
      IndexPartition p, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SUBSPACE_CALL> call;
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                    \
  case DIM:                                                             \
    {                                                                   \
      Point<DIM, coord_t> point = color;                                \
      return runtime->get_index_subspace(p, &point, TYPE_TAG_##DIM##D); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::get_index_subspace. This probably means you need "
                << "to build Legion with support for ore dimensions.";
          error.raise();
        }
    }
    return IndexSpace::NO_SPACE;
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_index_subspace_internal(
      IndexPartition p, const void* realm_color, TypeTag type_tag)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SUBSPACE_CALL> call;
    return runtime->get_index_subspace(p, realm_color, type_tag);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_index_subspace(
      Context ctx, IndexPartition p, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_INDEX_SUBSPACE_CALL> call(ctx, __func__);
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                    \
  case DIM:                                                             \
    {                                                                   \
      Point<DIM, coord_t> point = color;                                \
      return runtime->has_index_subspace(p, &point, TYPE_TAG_##DIM##D); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::has_index_subspace. This probably means you need "
                << "to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return false;
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_index_subspace(IndexPartition p, const DomainPoint& color)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_INDEX_SUBSPACE_CALL> call;
    switch (color.get_dim())
    {
#define DIMFUNC(DIM)                                                    \
  case DIM:                                                             \
    {                                                                   \
      Point<DIM, coord_t> point = color;                                \
      return runtime->has_index_subspace(p, &point, TYPE_TAG_##DIM##D); \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << color.get_dim() << " for "
                << "Runtime::has_index_subspace. This probably means you need "
                << "to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return false;
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_index_subspace_internal(
      IndexPartition p, const void* realm_color, TypeTag type_tag)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_INDEX_SUBSPACE_CALL> call;
    return runtime->has_index_subspace(p, realm_color, type_tag);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_multiple_domains(Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    // Multiple domains supported implicitly
    return false;
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_multiple_domains(IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    // Multiple domains supported implicitly
    return false;
  }

  //--------------------------------------------------------------------------
  Domain Runtime::get_index_space_domain(Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_DOMAIN_CALL> call(ctx, __func__);
    const TypeTag type_tag = handle.get_type_tag();
    const int dim = Internal::NT_TemplateHelper::get_dim(type_tag);
    switch (dim)
    {
#define DIMFUNC(DIM)                                                \
  case DIM:                                                         \
    {                                                               \
      DomainT<DIM, coord_t> realm_is;                               \
      runtime->get_index_space_domain(                              \
          handle, &realm_is,                                        \
          Internal::NT_TemplateHelper::encode_tag<DIM, coord_t>()); \
      return Domain(realm_is);                                      \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << dim << " for "
                << "Runtime::get_index_space_domain. This probably means you "
                << "need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return Domain::NO_DOMAIN;
  }

  //--------------------------------------------------------------------------
  Domain Runtime::get_index_space_domain(IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_DOMAIN_CALL> call;
    const TypeTag type_tag = handle.get_type_tag();
    const int dim = Internal::NT_TemplateHelper::get_dim(type_tag);
    switch (dim)
    {
#define DIMFUNC(DIM)                                                \
  case DIM:                                                         \
    {                                                               \
      DomainT<DIM, coord_t> realm_is;                               \
      runtime->get_index_space_domain(                              \
          handle, &realm_is,                                        \
          Internal::NT_TemplateHelper::encode_tag<DIM, coord_t>()); \
      return Domain(realm_is);                                      \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << dim << " for "
                << "Runtime::get_index_space_domain. This probably means you "
                << "need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return Domain::NO_DOMAIN;
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_domain_internal(
      IndexSpace handle, void* realm_is, TypeTag type_tag)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_DOMAIN_CALL> call;
    runtime->get_index_space_domain(handle, realm_is, type_tag);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_domains(
      Context ctx, IndexSpace handle, std::vector<Domain>& domains)
  //--------------------------------------------------------------------------
  {
    domains.emplace_back(get_index_space_domain(ctx, handle));
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_domains(
      IndexSpace handle, std::vector<Domain>& domains)
  //--------------------------------------------------------------------------
  {
    domains.emplace_back(get_index_space_domain(handle));
  }

  //--------------------------------------------------------------------------
  Domain Runtime::get_index_partition_color_space(Context ctx, IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_SPACE_CALL> call(
        ctx, __func__);
    return runtime->get_index_partition_color_space(p);
  }

  //--------------------------------------------------------------------------
  Domain Runtime::get_index_partition_color_space(IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_SPACE_CALL> call;
    return runtime->get_index_partition_color_space(p);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_partition_color_space_internal(
      IndexPartition p, void* realm_is, TypeTag type_tag)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_SPACE_CALL> call;
    runtime->get_index_partition_color_space(p, realm_is, type_tag);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_index_partition_color_space_name(
      Context ctx, IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_SPACE_NAME_CALL> call(
        ctx, __func__);
    return runtime->get_index_partition_color_space_name(p);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_index_partition_color_space_name(IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_SPACE_NAME_CALL> call;
    return runtime->get_index_partition_color_space_name(p);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_partition_colors(
      Context ctx, IndexSpace sp, std::set<Color>& colors)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_PARTITION_COLORS_CALL> call(
        ctx, __func__);
    runtime->get_index_space_partition_colors(sp, colors);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_partition_colors(
      Context ctx, IndexSpace sp, std::set<DomainPoint>& colors)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_PARTITION_COLORS_CALL> call(
        ctx, __func__);
    std::set<Color> temp_colors;
    runtime->get_index_space_partition_colors(sp, temp_colors);
    for (const Color& color : temp_colors) colors.insert(DomainPoint(color));
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_partition_colors(
      IndexSpace sp, std::set<Color>& colors)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_PARTITION_COLORS_CALL> call;
    runtime->get_index_space_partition_colors(sp, colors);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_partition_colors(
      IndexSpace sp, std::set<DomainPoint>& colors)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_PARTITION_COLORS_CALL> call;
    std::set<Color> temp_colors;
    runtime->get_index_space_partition_colors(sp, temp_colors);
    for (const Color& color : temp_colors) colors.insert(DomainPoint(color));
  }

  //--------------------------------------------------------------------------
  bool Runtime::is_index_partition_disjoint(Context ctx, IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_IS_INDEX_PARTITION_DISJOINT_CALL> call(
        ctx, __func__);
    return runtime->is_index_partition_disjoint(p);
  }

  //--------------------------------------------------------------------------
  bool Runtime::is_index_partition_disjoint(IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_IS_INDEX_PARTITION_DISJOINT_CALL> call;
    return runtime->is_index_partition_disjoint(p);
  }

  //--------------------------------------------------------------------------
  bool Runtime::is_index_partition_complete(Context ctx, IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_IS_INDEX_PARTITION_COMPLETE_CALL> call(
        ctx, __func__);
    return runtime->is_index_partition_complete(p);
  }

  //--------------------------------------------------------------------------
  bool Runtime::is_index_partition_complete(IndexPartition p)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_IS_INDEX_PARTITION_COMPLETE_CALL> call;
    return runtime->is_index_partition_complete(p);
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_index_space_color(Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_COLOR_CALL> call(ctx, __func__);
    Point<1, coord_t> point;
    runtime->get_index_space_color_point(handle, &point, TYPE_TAG_1D);
    return point[0];
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_index_space_color(IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_COLOR_CALL> call;
    Point<1, coord_t> point;
    runtime->get_index_space_color_point(handle, &point, TYPE_TAG_1D);
    return point[0];
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_index_space_color_point(
      Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_COLOR_POINT_CALL> call(
        ctx, __func__);
    return runtime->get_index_space_color_point(handle);
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_index_space_color_point(IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_COLOR_POINT_CALL> call;
    return runtime->get_index_space_color_point(handle);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_index_space_color_internal(
      IndexSpace handle, void* realm_color, TypeTag type_tag)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_COLOR_POINT_CALL> call;
    runtime->get_index_space_color_point(handle, realm_color, type_tag);
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_index_partition_color(Context ctx, IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_CALL> call(
        ctx, __func__);
    return runtime->get_index_partition_color(handle);
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_index_partition_color(IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_CALL> call;
    return runtime->get_index_partition_color(handle);
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_index_partition_color_point(
      Context ctx, IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_POINT_CALL> call(
        ctx, __func__);
    return DomainPoint(runtime->get_index_partition_color(handle));
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_index_partition_color_point(IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_COLOR_POINT_CALL> call;
    return DomainPoint(runtime->get_index_partition_color(handle));
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_parent_index_space(Context ctx, IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_INDEX_SPACE_CALL> call(ctx, __func__);
    return runtime->get_parent_index_space(handle);
  }

  //--------------------------------------------------------------------------
  IndexSpace Runtime::get_parent_index_space(IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_INDEX_SPACE_CALL> call;
    return runtime->get_parent_index_space(handle);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_parent_index_partition(Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_PARENT_INDEX_PARTITION_CALL> call(
        ctx, __func__);
    return runtime->has_parent_index_partition(handle);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_parent_index_partition(IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_PARENT_INDEX_PARTITION_CALL> call;
    return runtime->has_parent_index_partition(handle);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::get_parent_index_partition(
      Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_INDEX_PARTITION_CALL> call(
        ctx, __func__);
    return runtime->get_parent_index_partition(handle);
  }

  //--------------------------------------------------------------------------
  IndexPartition Runtime::get_parent_index_partition(IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_INDEX_PARTITION_CALL> call;
    return runtime->get_parent_index_partition(handle);
  }

  //--------------------------------------------------------------------------
  unsigned Runtime::get_index_space_depth(Context ctx, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_DEPTH_CALL> call(ctx, __func__);
    return runtime->get_index_space_depth(handle);
  }

  //--------------------------------------------------------------------------
  unsigned Runtime::get_index_space_depth(IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_SPACE_DEPTH_CALL> call;
    return runtime->get_index_space_depth(handle);
  }

  //--------------------------------------------------------------------------
  unsigned Runtime::get_index_partition_depth(
      Context ctx, IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_DEPTH_CALL> call(
        ctx, __func__);
    return runtime->get_index_partition_depth(handle);
  }

  //--------------------------------------------------------------------------
  unsigned Runtime::get_index_partition_depth(IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_INDEX_PARTITION_DEPTH_CALL> call;
    return runtime->get_index_partition_depth(handle);
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::safe_cast(
      Context ctx, DomainPoint point, LogicalRegion region)
  //--------------------------------------------------------------------------
  {
    // Don't check against implicit_context here because this method might
    // be called from OpenMP processors which don't have implicit_context set
    if ((ctx == nullptr) || ((Internal::implicit_context != nullptr) &&
                             (Internal::implicit_context != ctx)))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Invalid task context passed to Runtime::safe_cast";
      error.raise();
    }
    unsigned long long start = 0;
    Internal::AutoProvenance prov;
    // Only do timing on the main thread
    if ((Internal::implicit_context != nullptr) &&
        ctx->begin_runtime_call(Internal::RUNTIME_SAFE_CAST_CALL, prov))
      start = Realm::Clock::current_time_in_nanoseconds();
    switch (point.get_dim())
    {
#define DIMFUNC(DIM)                                                         \
  case DIM:                                                                  \
    {                                                                        \
      Point<DIM, coord_t> p(point);                                          \
      if (ctx->safe_cast(region.get_index_space(), &p, TYPE_TAG_##DIM##D))   \
      {                                                                      \
        ctx->end_runtime_call(                                               \
            Internal::RUNTIME_SAFE_CAST_CALL, prov, start,                   \
            (start == 0) ? 0 : Realm::Clock::current_time_in_nanoseconds()); \
        return point;                                                        \
      }                                                                      \
      break;                                                                 \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Unsupported dimension " << point.get_dim() << " for "
                << "Runtime::safe_cast. This probably means you need to build "
                << "Legion with support for more dimensions.";
          error.raise();
        }
    }
    if (Internal::implicit_context != nullptr)
      ctx->end_runtime_call(
          Internal::RUNTIME_SAFE_CAST_CALL, prov, start,
          (start == 0) ? 0 : Realm::Clock::current_time_in_nanoseconds());
    return DomainPoint::nil();
  }

  //--------------------------------------------------------------------------
  bool Runtime::safe_cast_internal(
      Context ctx, LogicalRegion region, const void* realm_point,
      TypeTag type_tag, const char* func)
  //--------------------------------------------------------------------------
  {
    // Don't check against implicit_context here because this method might
    // be called from OpenMP processors which don't have implicit_context set
    if ((ctx == nullptr) || ((Internal::implicit_context != nullptr) &&
                             (Internal::implicit_context != ctx)))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Invalid task context passed to Runtime::safe_cast";
      error.raise();
    }
    unsigned long long start = 0;
    Internal::AutoProvenance prov;
    // Only do timing on the main thread
    if ((Internal::implicit_context != nullptr) &&
        ctx->begin_runtime_call(Internal::RUNTIME_SAFE_CAST_CALL, prov))
      start = Realm::Clock::current_time_in_nanoseconds();
    const bool result =
        ctx->safe_cast(region.get_index_space(), realm_point, type_tag);
    if (Internal::implicit_context != nullptr)
      ctx->end_runtime_call(
          Internal::RUNTIME_SAFE_CAST_CALL, prov, start,
          (start == 0) ? 0 : Realm::Clock::current_time_in_nanoseconds());
    return result;
  }

  //--------------------------------------------------------------------------
  FieldSpace Runtime::create_field_space(Context ctx, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_FIELD_SPACE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_field_space(call);
  }

  //--------------------------------------------------------------------------
  FieldSpace Runtime::create_field_space(
      Context ctx, const std::vector<size_t>& field_sizes,
      std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_FIELD_SPACE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_field_space(
        field_sizes, resulting_fields, serdez_id, call);
  }

  //--------------------------------------------------------------------------
  FieldSpace Runtime::create_field_space(
      Context ctx, const std::vector<Future>& field_sizes,
      std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_FIELD_SPACE_CALL> call(
        prov, ctx, __func__);
    return ctx->create_field_space(
        field_sizes, resulting_fields, serdez_id, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::create_shared_ownership(Context ctx, FieldSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_SHARED_OWNERSHIP_CALL> call(
        ctx, __func__);
    ctx->create_shared_ownership(handle);
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_field_space(
      Context ctx, FieldSpace handle, const bool unordered, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DESTROY_FIELD_SPACE_CALL> call(
        prov, ctx, __func__);
    ctx->destroy_field_space(handle, unordered, call);
  }

  //--------------------------------------------------------------------------
  size_t Runtime::get_field_size(Context ctx, FieldSpace handle, FieldID fid)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_FIELD_SIZE_CALL> call(ctx, __func__);
    return runtime->get_field_size(handle, fid);
  }

  //--------------------------------------------------------------------------
  size_t Runtime::get_field_size(FieldSpace handle, FieldID fid)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_FIELD_SIZE_CALL> call;
    return runtime->get_field_size(handle, fid);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_field_space_fields(
      Context ctx, FieldSpace handle, std::vector<FieldID>& fields)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_FIELD_SPACE_FIELDS_CALL> call(ctx, __func__);
    runtime->get_field_space_fields(handle, fields);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_field_space_fields(
      FieldSpace handle, std::vector<FieldID>& fields)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_FIELD_SPACE_FIELDS_CALL> call;
    runtime->get_field_space_fields(handle, fields);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_field_space_fields(
      Context ctx, FieldSpace handle, std::set<FieldID>& fields)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_FIELD_SPACE_FIELDS_CALL> call(ctx, __func__);
    std::vector<FieldID> local;
    runtime->get_field_space_fields(handle, local);
    fields.insert(local.begin(), local.end());
  }

  //--------------------------------------------------------------------------
  void Runtime::get_field_space_fields(
      FieldSpace handle, std::set<FieldID>& fields)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_FIELD_SPACE_FIELDS_CALL> call;
    std::vector<FieldID> local;
    runtime->get_field_space_fields(handle, local);
    fields.insert(local.begin(), local.end());
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::create_logical_region(
      Context ctx, IndexSpace index, FieldSpace fields, bool task_local,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_LOGICAL_REGION_CALL> call(
        prov, ctx, __func__);
    return ctx->create_logical_region(index, fields, task_local, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::create_shared_ownership(Context ctx, LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_SHARED_OWNERSHIP_CALL> call(
        ctx, __func__);
    ctx->create_shared_ownership(handle);
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_logical_region(
      Context ctx, LogicalRegion handle, const bool unordered, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DESTROY_LOGICAL_REGION_CALL> call(ctx, __func__);
    ctx->destroy_logical_region(handle, unordered, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_logical_partition(
      Context ctx, LogicalPartition handle, const bool unordered)
  //--------------------------------------------------------------------------
  {
    // This is a no-op now
  }

  //--------------------------------------------------------------------------
  void Runtime::reset_equivalence_sets(
      Context ctx, LogicalRegion parent, LogicalRegion region,
      const std::set<FieldID>& fields)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_RESET_EQUIVALENCE_SETS_CALL> call(ctx, __func__);
    ctx->reset_equivalence_sets(parent, region, fields);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition(
      Context ctx, LogicalRegion parent, IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_CALL> call(ctx, __func__);
    return runtime->get_logical_partition(parent, handle);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition(
      LogicalRegion parent, IndexPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_CALL> call;
    return runtime->get_logical_partition(parent, handle);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition_by_color(
      Context ctx, LogicalRegion parent, Color c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_BY_COLOR_CALL> call(
        ctx, __func__);
    return runtime->get_logical_partition_by_color(parent, c);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition_by_color(
      Context ctx, LogicalRegion parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_BY_COLOR_CALL> call(
        ctx, __func__);
    return runtime->get_logical_partition_by_color(parent, c.get_color());
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition_by_color(
      LogicalRegion parent, Color c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_BY_COLOR_CALL> call;
    return runtime->get_logical_partition_by_color(parent, c);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition_by_color(
      LogicalRegion parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_BY_COLOR_CALL> call;
    return runtime->get_logical_partition_by_color(parent, c.get_color());
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_logical_partition_by_color(
      Context ctx, LogicalRegion parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_LOGICAL_PARTITION_BY_COLOR_CALL> call(
        ctx, __func__);
    return runtime->has_logical_partition_by_color(parent, c.get_color());
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_logical_partition_by_color(
      LogicalRegion parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_LOGICAL_PARTITION_BY_COLOR_CALL> call;
    return runtime->has_logical_partition_by_color(parent, c.get_color());
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition_by_tree(
      Context ctx, IndexPartition handle, FieldSpace fspace, RegionTreeID tid)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_BY_TREE_CALL> call(
        ctx, __func__);
    return runtime->get_logical_partition_by_tree(handle, fspace, tid);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_logical_partition_by_tree(
      IndexPartition handle, FieldSpace fspace, RegionTreeID tid)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_BY_TREE_CALL> call;
    return runtime->get_logical_partition_by_tree(handle, fspace, tid);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion(
      Context ctx, LogicalPartition parent, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_CALL> call(ctx, __func__);
    return runtime->get_logical_subregion(parent, handle);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion(
      LogicalPartition parent, IndexSpace handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_CALL> call;
    return runtime->get_logical_subregion(parent, handle);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion_by_color(
      Context ctx, LogicalPartition parent, Color c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_BY_COLOR_CALL> call(
        ctx, __func__);
    Point<1, coord_t> point(c);
    return runtime->get_logical_subregion_by_color(parent, &point, TYPE_TAG_1D);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion_by_color(
      Context ctx, LogicalPartition parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_BY_COLOR_CALL> call(
        ctx, __func__);
    switch (c.get_dim())
    {
#define DIMFUNC(DIM)                                  \
  case DIM:                                           \
    {                                                 \
      Point<DIM, coord_t> point(c);                   \
      return runtime->get_logical_subregion_by_color( \
          parent, &point, TYPE_TAG_##DIM##D);         \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "Unsupported dimension " << c.get_dim() << " for "
              << "Runtime::get_logical_subregion_by_color. This probably means "
              << "you need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return LogicalRegion::NO_REGION;
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion_by_color(
      LogicalPartition parent, Color c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_BY_COLOR_CALL> call;
    Point<1, coord_t> point(c);
    return runtime->get_logical_subregion_by_color(parent, &point, TYPE_TAG_1D);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion_by_color(
      LogicalPartition parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_BY_COLOR_CALL> call;
    switch (c.get_dim())
    {
#define DIMFUNC(DIM)                                  \
  case DIM:                                           \
    {                                                 \
      Point<DIM, coord_t> point(c);                   \
      return runtime->get_logical_subregion_by_color( \
          parent, &point, TYPE_TAG_##DIM##D);         \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "Unsupported dimension " << c.get_dim() << " for "
              << "Runtime::get_logical_subregion_by_color. This probably means "
              << "you need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return LogicalRegion::NO_REGION;
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion_by_color_internal(
      LogicalPartition parent, const void* realm_color, TypeTag type_tag)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_BY_COLOR_CALL> call;
    return runtime->get_logical_subregion_by_color(
        parent, realm_color, type_tag);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_logical_subregion_by_color(
      Context ctx, LogicalPartition parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_LOGICAL_SUBREGION_BY_COLOR_CALL> call(
        ctx, __func__);
    switch (c.get_dim())
    {
#define DIMFUNC(DIM)                                  \
  case DIM:                                           \
    {                                                 \
      Point<DIM, coord_t> point(c);                   \
      return runtime->has_logical_subregion_by_color( \
          parent, &point, TYPE_TAG_##DIM##D);         \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "Unsupported dimension " << c.get_dim() << " for "
              << "Runtime::has_logical_subregion_by_color. This probably means "
              << "you need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return false;
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_logical_subregion_by_color(
      LogicalPartition parent, const DomainPoint& c)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_LOGICAL_SUBREGION_BY_COLOR_CALL> call;
    switch (c.get_dim())
    {
#define DIMFUNC(DIM)                                  \
  case DIM:                                           \
    {                                                 \
      Point<DIM, coord_t> point(c);                   \
      return runtime->has_logical_subregion_by_color( \
          parent, &point, TYPE_TAG_##DIM##D);         \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "Unsupported dimension " << c.get_dim() << " for "
              << "Runtime::has_logical_subregion_by_color. This probably means "
              << "you need to build Legion with support for more dimensions.";
          error.raise();
        }
    }
    return false;
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_logical_subregion_by_color_internal(
      LogicalPartition parent, const void* realm_color, TypeTag type_tag)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_LOGICAL_SUBREGION_BY_COLOR_CALL> call;
    return runtime->has_logical_subregion_by_color(
        parent, realm_color, type_tag);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion_by_tree(
      Context ctx, IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_BY_TREE_CALL> call(
        ctx, __func__);
    return runtime->get_logical_subregion_by_tree(handle, fspace, tid);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_logical_subregion_by_tree(
      IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_SUBREGION_BY_TREE_CALL> call;
    return runtime->get_logical_subregion_by_tree(handle, fspace, tid);
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_logical_region_color(Context ctx, LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_REGION_COLOR_CALL> call(
        ctx, __func__);
    Point<1, coord_t> point;
    runtime->get_logical_region_color(handle, &point, TYPE_TAG_1D);
    return point[0];
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_logical_region_color_point(
      Context ctx, LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_REGION_COLOR_POINT_CALL> call(
        ctx, __func__);
    return runtime->get_logical_region_color_point(handle);
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_logical_region_color(LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_REGION_COLOR_CALL> call;
    Point<1, coord_t> point;
    runtime->get_logical_region_color(handle, &point, TYPE_TAG_1D);
    return point[0];
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_logical_region_color_point(LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_REGION_COLOR_POINT_CALL> call;
    return runtime->get_logical_region_color_point(handle);
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_logical_partition_color(
      Context ctx, LogicalPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_COLOR_CALL> call(
        ctx, __func__);
    return runtime->get_logical_partition_color(handle);
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_logical_partition_color_point(
      Context ctx, LogicalPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_COLOR_CALL> call(
        ctx, __func__);
    return DomainPoint(runtime->get_logical_partition_color(handle));
  }

  //--------------------------------------------------------------------------
  Color Runtime::get_logical_partition_color(LogicalPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_COLOR_CALL> call;
    return runtime->get_logical_partition_color(handle);
  }

  //--------------------------------------------------------------------------
  DomainPoint Runtime::get_logical_partition_color_point(
      LogicalPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOGICAL_PARTITION_COLOR_POINT_CALL> call;
    return DomainPoint(runtime->get_logical_partition_color(handle));
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_parent_logical_region(
      Context ctx, LogicalPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_LOGICAL_REGION_CALL> call(
        ctx, __func__);
    return runtime->get_parent_logical_region(handle);
  }

  //--------------------------------------------------------------------------
  LogicalRegion Runtime::get_parent_logical_region(LogicalPartition handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_LOGICAL_REGION_CALL> call;
    return runtime->get_parent_logical_region(handle);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_parent_logical_partition(Context ctx, LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_PARENT_LOGICAL_PARTITION_CALL> call(
        ctx, __func__);
    return runtime->has_parent_logical_partition(handle);
  }

  //--------------------------------------------------------------------------
  bool Runtime::has_parent_logical_partition(LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_HAS_PARENT_LOGICAL_PARTITION_CALL> call;
    return runtime->has_parent_logical_partition(handle);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_parent_logical_partition(
      Context ctx, LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_LOGICAL_REGION_CALL> call(
        ctx, __func__);
    return runtime->get_parent_logical_partition(handle);
  }

  //--------------------------------------------------------------------------
  LogicalPartition Runtime::get_parent_logical_partition(LogicalRegion handle)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_LOGICAL_REGION_CALL> call;
    return runtime->get_parent_logical_partition(handle);
  }

  //--------------------------------------------------------------------------
  FieldAllocator Runtime::create_field_allocator(Context ctx, FieldSpace space)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_PARENT_LOGICAL_REGION_CALL> call(
        ctx, __func__);
    return FieldAllocator(
        ctx->create_field_allocator(space, false /*unordered*/));
  }

  //--------------------------------------------------------------------------
  ArgumentMap Runtime::create_argument_map(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_ARGUMENT_MAP_CALL> call(ctx, __func__);
    return ArgumentMap(new Internal::ArgumentMapImpl());
  }

  //--------------------------------------------------------------------------
  Future Runtime::execute_task(
      Context ctx, const TaskLauncher& launcher,
      std::vector<OutputRequirement>* outputs /*= nullptr*/)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_EXECUTE_TASK_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->execute_task(launcher, outputs, call);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::execute_index_space(
      Context ctx, const IndexTaskLauncher& launcher,
      std::vector<OutputRequirement>* outputs /*= nullptr*/)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_EXECUTE_INDEX_SPACE_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->execute_index_space(launcher, outputs, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::execute_index_space(
      Context ctx, const IndexTaskLauncher& launcher, ReductionOpID redop,
      bool deterministic, std::vector<OutputRequirement>* outputs /*= nullptr*/)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_EXECUTE_INDEX_SPACE_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->execute_index_space(
        launcher, redop, deterministic, outputs, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::reduce_future_map(
      Context ctx, const FutureMap& future_map, ReductionOpID redop,
      bool deterministic, MapperID map, MappingTagID tag, const char* prov,
      Future initial_value)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_REDUCE_FUTURE_MAP_CALL> call(
        prov, ctx, __func__);
    return ctx->reduce_future_map(
        future_map, redop, deterministic, map, tag, call, initial_value);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::construct_future_map(
      Context ctx, IndexSpace domain,
      const std::map<DomainPoint, UntypedBuffer>& data, bool collective,
      ShardingID sid, bool implicit, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CONSTRUCT_FUTURE_MAP_CALL> call(
        prov, ctx, __func__);
    return ctx->construct_future_map(
        domain, data, call, collective, sid, implicit);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::construct_future_map(
      Context ctx, const Domain& domain,
      const std::map<DomainPoint, UntypedBuffer>& data, bool collective,
      ShardingID sid, bool implicit)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CONSTRUCT_FUTURE_MAP_CALL> call(ctx, __func__);
    return ctx->construct_future_map(domain, data, collective, sid, implicit);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::construct_future_map(
      Context ctx, IndexSpace domain,
      const std::map<DomainPoint, Future>& futures, bool collective,
      ShardingID sid, bool implicit, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CONSTRUCT_FUTURE_MAP_CALL> call(
        prov, ctx, __func__);
    return ctx->construct_future_map(
        domain, futures, call, false, collective, sid, implicit);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::construct_future_map(
      Context ctx, const Domain& domain,
      const std::map<DomainPoint, Future>& futures, bool collective,
      ShardingID sid, bool implicit)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CONSTRUCT_FUTURE_MAP_CALL> call(ctx, __func__);
    return ctx->construct_future_map(
        domain, futures, collective, sid, implicit);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::transform_future_map(
      Context ctx, const FutureMap& fm, IndexSpace new_domain,
      PointTransformFunc func, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_TRANSFORM_FUTURE_MAP_CALL> call(
        prov, ctx, __func__);
    class PointTransformWrapper : public PointTransformFunctor {
    public:
      PointTransformWrapper(PointTransformFunc f) : func(f) { }
      virtual ~PointTransformWrapper(void) { }
    public:
      virtual bool is_invertible(void) const override { return false; }
      virtual DomainPoint transform_point(
          const DomainPoint& point, const Domain& domain,
          const Domain& range) override
      {
        return func(point, domain, range);
      }
    public:
      const PointTransformFunc func;
    };
    return ctx->transform_future_map(
        fm, new_domain, new PointTransformWrapper(func), true /*own*/, call);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::transform_future_map(
      Context ctx, const FutureMap& fm, IndexSpace new_domain,
      PointTransformFunctor* functor, bool own, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_TRANSFORM_FUTURE_MAP_CALL> call(
        prov, ctx, __func__);
    return ctx->transform_future_map(fm, new_domain, functor, own, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::execute_task(
      Context ctx, TaskID task_id,
      const std::vector<IndexSpaceRequirement>& indexes,
      const std::vector<FieldSpaceRequirement>& fields,
      const std::vector<RegionRequirement>& regions, const UntypedBuffer& arg,
      const Predicate& predicate, MapperID id, MappingTagID tag)
  //--------------------------------------------------------------------------
  {
    TaskLauncher launcher(task_id, arg, predicate, id, tag);
    launcher.index_requirements = indexes;
    launcher.region_requirements = regions;
    return execute_task(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::execute_index_space(
      Context ctx, TaskID task_id, const Domain domain,
      const std::vector<IndexSpaceRequirement>& indexes,
      const std::vector<FieldSpaceRequirement>& fields,
      const std::vector<RegionRequirement>& regions,
      const UntypedBuffer& global_arg, const ArgumentMap& arg_map,
      const Predicate& predicate, bool must_parallelism, MapperID id,
      MappingTagID tag)
  //--------------------------------------------------------------------------
  {
    IndexTaskLauncher launcher(
        task_id, domain, global_arg, arg_map, predicate, must_parallelism, id,
        tag);
    launcher.index_requirements = indexes;
    launcher.region_requirements = regions;
    return execute_index_space(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  Future Runtime::execute_index_space(
      Context ctx, TaskID task_id, const Domain domain,
      const std::vector<IndexSpaceRequirement>& indexes,
      const std::vector<FieldSpaceRequirement>& fields,
      const std::vector<RegionRequirement>& regions,
      const UntypedBuffer& global_arg, const ArgumentMap& arg_map,
      ReductionOpID reduction, const UntypedBuffer& initial_value,
      const Predicate& predicate, bool must_parallelism, MapperID id,
      MappingTagID tag)
  //--------------------------------------------------------------------------
  {
    IndexTaskLauncher launcher(
        task_id, domain, global_arg, arg_map, predicate, must_parallelism, id,
        tag);
    launcher.index_requirements = indexes;
    launcher.region_requirements = regions;
    return execute_index_space(ctx, launcher, reduction, true);
  }

  //--------------------------------------------------------------------------
  PhysicalRegion Runtime::map_region(
      Context ctx, const InlineLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_MAP_REGION_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->map_region(launcher, call);
  }

  //--------------------------------------------------------------------------
  PhysicalRegion Runtime::map_region(
      Context ctx, const RegionRequirement& req, MapperID id, MappingTagID tag,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    InlineLauncher launcher(req, id, tag);
    if (provenance != nullptr)
      launcher.provenance = provenance;
    return map_region(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  PhysicalRegion Runtime::map_region(
      Context ctx, unsigned idx, MapperID id, MappingTagID tag,
      const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_REMAP_REGION_CALL> call(prov, ctx, __func__);
    PhysicalRegion region = ctx->get_physical_region(idx);
    ctx->remap_region(region, call);
    return region;
  }

  //--------------------------------------------------------------------------
  void Runtime::remap_region(
      Context ctx, PhysicalRegion region, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_REMAP_REGION_CALL> call(prov, ctx, __func__);
    ctx->remap_region(region, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::unmap_region(Context ctx, PhysicalRegion region)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_UNMAP_REGION_CALL> call(ctx, __func__);
    ctx->unmap_region(region);
  }

  //--------------------------------------------------------------------------
  void Runtime::unmap_all_regions(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_UNMAP_ALL_REGIONS_CALL> call(ctx, __func__);
    ctx->unmap_all_regions();
  }

  //--------------------------------------------------------------------------
  OutputRegion Runtime::get_output_region(Context ctx, unsigned index)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_OUTPUT_REGION_CALL> call(ctx, __func__);
    return ctx->get_output_region(index);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_output_regions(
      Context ctx, std::vector<OutputRegion>& regions)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_OUTPUT_REGIONS_CALL> call(ctx, __func__);
    regions = ctx->get_output_regions();
  }

  //--------------------------------------------------------------------------
  void Runtime::fill_field(
      Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
      const void* value, size_t size, Predicate pred)
  //--------------------------------------------------------------------------
  {
    FillLauncher launcher(handle, parent, UntypedBuffer(value, size), pred);
    launcher.add_field(fid);
    fill_fields(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  void Runtime::fill_field(
      Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
      Future f, Predicate pred)
  //--------------------------------------------------------------------------
  {
    FillLauncher launcher(handle, parent, UntypedBuffer(), pred);
    launcher.set_future(f);
    launcher.add_field(fid);
    fill_fields(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  void Runtime::fill_fields(
      Context ctx, LogicalRegion handle, LogicalRegion parent,
      const std::set<FieldID>& fields, const void* value, size_t size,
      Predicate pred)
  //--------------------------------------------------------------------------
  {
    FillLauncher launcher(handle, parent, UntypedBuffer(value, size), pred);
    launcher.fields = fields;
    fill_fields(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  void Runtime::fill_fields(
      Context ctx, LogicalRegion handle, LogicalRegion parent,
      const std::set<FieldID>& fields, Future f, Predicate pred)
  //--------------------------------------------------------------------------
  {
    FillLauncher launcher(handle, parent, UntypedBuffer(), pred);
    launcher.set_future(f);
    launcher.fields = fields;
    fill_fields(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  void Runtime::fill_fields(Context ctx, const FillLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_FILL_FIELDS_CALL> call(
        launcher.provenance, ctx, __func__);
    ctx->fill_fields(launcher, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::fill_fields(Context ctx, const IndexFillLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_FILL_FIELDS_CALL> call(
        launcher.provenance, ctx, __func__);
    ctx->fill_fields(launcher, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::discard_fields(Context ctx, const DiscardLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DISCARD_FIELDS_CALL> call(
        launcher.provenance, ctx, __func__);
    ctx->discard_fields(launcher, call);
  }

  //--------------------------------------------------------------------------
  PhysicalRegion Runtime::attach_external_resource(
      Context ctx, const AttachLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ATTACH_EXTERNAL_RESOURCE_CALL> call(
        launcher.provenance, ctx, __func__);
    if (launcher.mapped)
    {
      PhysicalRegion region = ctx->attach_resource(launcher, call);
      ctx->remap_region(region, call);
      return region;
    }
    else
      return ctx->attach_resource(launcher, call);
  }

  //--------------------------------------------------------------------------
  ExternalResources Runtime::attach_external_resources(
      Context ctx, const IndexAttachLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ATTACH_EXTERNAL_RESOURCES_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->attach_resources(launcher, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::detach_external_resource(
      Context ctx, PhysicalRegion region, const bool flush /*= true*/,
      const bool unordered /*= false*/, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DETACH_EXTERNAL_RESOURCE_CALL> call(
        prov, ctx, __func__);
    return ctx->detach_resource(region, flush, unordered, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::detach_external_resources(
      Context ctx, ExternalResources resources, const bool flush /*= true*/,
      const bool unordered /*= false*/, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DETACH_EXTERNAL_RESOURCES_CALL> call(
        prov, ctx, __func__);
    return ctx->detach_resources(resources, flush, unordered, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::progress_unordered_operations(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_PROGRESS_UNORDERED_CALL> call(ctx, __func__);
    ctx->progress_unordered_operations();
  }

  LEGION_DISABLE_DEPRECATED_WARNINGS

  //--------------------------------------------------------------------------
  PhysicalRegion Runtime::attach_hdf5(
      Context ctx, const char* file_name, LogicalRegion handle,
      LogicalRegion parent, const std::map<FieldID, const char*>& field_map,
      LegionFileMode mode)
  //--------------------------------------------------------------------------
  {
    AttachLauncher launcher(LEGION_EXTERNAL_HDF5_FILE, handle, parent);
    launcher.attach_hdf5(file_name, field_map, mode);
    PhysicalRegion region = attach_external_resource(ctx, launcher);
    if (launcher.mapped)
      remap_region(ctx, region);
    return region;
  }

  //--------------------------------------------------------------------------
  void Runtime::detach_hdf5(Context ctx, PhysicalRegion region)
  //--------------------------------------------------------------------------
  {
    detach_external_resource(ctx, region);
  }

  //--------------------------------------------------------------------------
  PhysicalRegion Runtime::attach_file(
      Context ctx, const char* file_name, LogicalRegion handle,
      LogicalRegion parent, const std::vector<FieldID>& field_vec,
      LegionFileMode mode)
  //--------------------------------------------------------------------------
  {
    AttachLauncher launcher(LEGION_EXTERNAL_POSIX_FILE, handle, parent);
    launcher.attach_file(file_name, field_vec, mode);
    PhysicalRegion region = attach_external_resource(ctx, launcher);
    if (launcher.mapped)
      remap_region(ctx, region);
    return region;
  }

  LEGION_REENABLE_DEPRECATED_WARNINGS

  //--------------------------------------------------------------------------
  void Runtime::detach_file(Context ctx, PhysicalRegion region)
  //--------------------------------------------------------------------------
  {
    detach_external_resource(ctx, region);
  }

  //--------------------------------------------------------------------------
  void Runtime::issue_copy_operation(Context ctx, const CopyLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ISSUE_COPY_OPERATION_CALL> call(
        launcher.provenance, ctx, __func__);
    ctx->issue_copy(launcher, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::issue_copy_operation(
      Context ctx, const IndexCopyLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ISSUE_COPY_OPERATION_CALL> call(
        launcher.provenance, ctx, __func__);
    ctx->issue_copy(launcher, call);
  }

  //--------------------------------------------------------------------------
  Predicate Runtime::create_predicate(
      Context ctx, const Future& f, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PREDICATE_CALL> call(prov, ctx, __func__);
    return ctx->create_predicate(f, call);
  }

  //--------------------------------------------------------------------------
  Predicate Runtime::predicate_not(
      Context ctx, const Predicate& p, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_PREDICATE_NOT_CALL> call(prov, ctx, __func__);
    return ctx->predicate_not(p, call);
  }

  //--------------------------------------------------------------------------
  Predicate Runtime::predicate_and(
      Context ctx, const Predicate& p1, const Predicate& p2,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    PredicateLauncher launcher(true /*and*/);
    launcher.add_predicate(p1);
    launcher.add_predicate(p2);
    if (provenance != nullptr)
      launcher.provenance.assign(provenance);
    return create_predicate(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  Predicate Runtime::predicate_or(
      Context ctx, const Predicate& p1, const Predicate& p2,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    PredicateLauncher launcher(false /*and*/);
    launcher.add_predicate(p1);
    launcher.add_predicate(p2);
    if (provenance != nullptr)
      launcher.provenance.assign(provenance);
    return create_predicate(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  Predicate Runtime::create_predicate(
      Context ctx, const PredicateLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PREDICATE_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->create_predicate(launcher, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::get_predicate_future(
      Context ctx, const Predicate& p, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PREDICATE_CALL> call(prov, ctx, __func__);
    return ctx->get_predicate_future(p, call);
  }

  //--------------------------------------------------------------------------
  Lock Runtime::create_lock(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_LOCK_CALL> call(ctx, __func__);
    return ctx->create_lock();
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_lock(Context ctx, Lock l)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DESTROY_LOCK_CALL> call(ctx, __func__);
    ctx->destroy_lock(l);
  }

  //--------------------------------------------------------------------------
  Grant Runtime::acquire_grant(
      Context ctx, const std::vector<LockRequest>& requests)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ACQUIRE_GRANT_CALL> call(ctx, __func__);
    return ctx->acquire_grant(requests);
  }

  //--------------------------------------------------------------------------
  void Runtime::release_grant(Context ctx, Grant grant)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_RELEASE_GRANT_CALL> call(ctx, __func__);
    ctx->release_grant(grant);
  }

  //--------------------------------------------------------------------------
  PhaseBarrier Runtime::create_phase_barrier(Context ctx, unsigned arrivals)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_PHASE_BARRIER_CALL> call(ctx, __func__);
    return ctx->create_phase_barrier(arrivals);
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DESTROY_PHASE_BARRIER_CALL> call(ctx, __func__);
    ctx->destroy_phase_barrier(pb);
  }

  //--------------------------------------------------------------------------
  PhaseBarrier Runtime::advance_phase_barrier(Context ctx, PhaseBarrier pb)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ADVANCE_PHASE_BARRIER_CALL> call(ctx, __func__);
    return ctx->advance_phase_barrier(pb);
  }

  //--------------------------------------------------------------------------
  DynamicCollective Runtime::create_dynamic_collective(
      Context ctx, unsigned arrivals, ReductionOpID redop,
      const void* init_value, size_t init_size)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CREATE_DYNAMIC_COLLECTIVE_CALL> call(
        ctx, __func__);
    return ctx->create_dynamic_collective(
        arrivals, redop, init_value, init_size);
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_dynamic_collective(Context ctx, DynamicCollective dc)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DESTROY_DYNAMIC_COLLECTIVE_CALL> call(
        ctx, __func__);
    ctx->destroy_dynamic_collective(dc);
  }

  //--------------------------------------------------------------------------
  void Runtime::arrive_dynamic_collective(
      Context ctx, DynamicCollective dc, const void* buffer, size_t size,
      unsigned count)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ARRIVE_DYNAMIC_COLLECTIVE_CALL> call(
        ctx, __func__);
    ctx->arrive_dynamic_collective(dc, buffer, size, count);
  }

  //--------------------------------------------------------------------------
  void Runtime::defer_dynamic_collective_arrival(
      Context ctx, DynamicCollective dc, const Future& f, unsigned count)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_DEFER_DYNAMIC_COLLECTIVE_CALL> call(
        ctx, __func__);
    ctx->defer_dynamic_collective_arrival(dc, f, count);
  }

  //--------------------------------------------------------------------------
  Future Runtime::get_dynamic_collective_result(
      Context ctx, DynamicCollective dc, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_DYNAMIC_COLLECTIVE_CALL> call(
        prov, ctx, __func__);
    return ctx->get_dynamic_collective_result(dc, call);
  }

  //--------------------------------------------------------------------------
  DynamicCollective Runtime::advance_dynamic_collective(
      Context ctx, DynamicCollective dc)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ADVANCE_DYNAMIC_COLLECTIVE_CALL> call(
        ctx, __func__);
    return ctx->advance_dynamic_collective(dc);
  }

  //--------------------------------------------------------------------------
  void Runtime::issue_acquire(Context ctx, const AcquireLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ISSUE_ACQUIRE_CALL> call(
        launcher.provenance, ctx, __func__);
    ctx->issue_acquire(launcher, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::issue_release(Context ctx, const ReleaseLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ISSUE_RELEASE_CALL> call(
        launcher.provenance, ctx, __func__);
    ctx->issue_release(launcher, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::issue_mapping_fence(Context ctx, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ISSUE_MAPPING_FENCE_CALL> call(
        prov, ctx, __func__);
    return ctx->issue_mapping_fence(call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::issue_execution_fence(Context ctx, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ISSUE_EXECUTION_FENCE_CALL> call(
        prov, ctx, __func__);
    return ctx->issue_execution_fence(call);
  }

  //--------------------------------------------------------------------------
  void Runtime::begin_trace(
      Context ctx, TraceID tid, bool logical_only /*= false*/,
      bool static_trace, const std::set<RegionTreeID>* trees, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_BEGIN_TRACE_CALL> call(prov, ctx, __func__);
    ctx->begin_trace(
        tid, logical_only, static_trace, trees, false /*dep*/, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::end_trace(Context ctx, TraceID tid, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_END_TRACE_CALL> call(prov, ctx, __func__);
    ctx->end_trace(tid, false /*deprecated*/, call);
  }

  //--------------------------------------------------------------------------
  void Runtime::begin_static_trace(
      Context ctx, const std::set<RegionTreeID>* managed)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_BEGIN_TRACE_CALL> call(ctx, __func__);
    ctx->begin_trace(
        0, true /*logical only*/, true /*static*/, managed, true /*deprecated*/,
        nullptr /*provenance*/);
  }

  //--------------------------------------------------------------------------
  void Runtime::end_static_trace(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_END_TRACE_CALL> call(ctx, __func__);
    ctx->end_trace(0, true /*deprecated*/, nullptr /*provenance*/);
  }

  //--------------------------------------------------------------------------
  TraceID Runtime::generate_dynamic_trace_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_trace_id();
  }

  //--------------------------------------------------------------------------
  TraceID Runtime::generate_library_trace_ids(const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_trace_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ TraceID Runtime::generate_static_trace_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_trace_id();
  }

  //--------------------------------------------------------------------------
  void Runtime::complete_frame(Context ctx, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_COMPLETE_FRAME_CALL> call(prov, ctx, __func__);
    ctx->complete_frame(call);
  }

  //--------------------------------------------------------------------------
  FutureMap Runtime::execute_must_epoch(
      Context ctx, const MustEpochLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_MUST_EPOCH_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->execute_must_epoch(launcher, call);
  }

  //--------------------------------------------------------------------------
  Future Runtime::select_tunable_value(
      Context ctx, TunableID tid, MapperID mid, MappingTagID tag,
      const void* args, size_t argsize)
  //--------------------------------------------------------------------------
  {
    TunableLauncher launcher(tid, mid, tag);
    launcher.arg = UntypedBuffer(args, argsize);
    return select_tunable_value(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  Future Runtime::select_tunable_value(
      Context ctx, const TunableLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_SELECT_TUNABLE_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->select_tunable_value(launcher, call);
  }

  //--------------------------------------------------------------------------
  int Runtime::get_tunable_value(
      Context ctx, TunableID tid, MapperID mid, MappingTagID tag)
  //--------------------------------------------------------------------------
  {
    TunableLauncher launcher(tid, mid, tag);
    Future f = select_tunable_value(ctx, launcher);
    return f.get_result<int>();
  }

  //--------------------------------------------------------------------------
  const Task* Runtime::get_local_task(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOCAL_TASK_CALL> call(ctx, __func__);
    return ctx->get_task();
  }

  //--------------------------------------------------------------------------
  void* Runtime::get_local_task_variable_untyped(
      Context ctx, LocalVariableID id)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_LOCAL_VARIABLE_CALL> call(ctx, __func__);
    return ctx->get_local_task_variable(id);
  }

  //--------------------------------------------------------------------------
  void Runtime::set_local_task_variable_untyped(
      Context ctx, LocalVariableID id, const void* value,
      void (*destructor)(void*))
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_SET_LOCAL_VARIABLE_CALL> call(ctx, __func__);
    ctx->set_local_task_variable(id, value, destructor);
  }

  //--------------------------------------------------------------------------
  static Memory find_memory_by_kind(Context ctx, Memory::Kind kind, bool value)
  //--------------------------------------------------------------------------
  {
    Machine machine = Realm::Machine::get_machine();
    Machine::MemoryQuery finder(machine);
    const Processor exec_proc = ctx->get_executing_processor();
    finder.best_affinity_to(exec_proc);
    finder.only_kind(kind);
    if (finder.count() == 0)
    {
      finder = Machine::MemoryQuery(machine);
      finder.has_affinity_to(exec_proc);
      finder.only_kind(kind);
    }
    if (finder.count() == 0)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Unable to find associated " << kind << " memory kind for "
            << exec_proc << " when performed an (Untyped)Deferred"
            << (value ? "Value" : "Buffer") << " creation in " << *ctx << ".";
      error.raise();
    }
    return finder.first();
  }

  //--------------------------------------------------------------------------
  UntypedDeferredValue Runtime::allocate_deferred_value(
      Context ctx, const DeferredValueRequest& request)
  //--------------------------------------------------------------------------
  {
    // Don't check against implicit_context here because this method might
    // be called from OpenMP processors which don't have implicit_context set
    if ((ctx == nullptr) || ((Internal::implicit_context != nullptr) &&
                             (Internal::implicit_context != ctx)))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error
          << "Invalid task context passed to Runtime::allocate_deferred_value";
      error.raise();
    }
    unsigned long long start = 0;
    Internal::AutoProvenance prov;
    // Only do timing on the main thread
    if ((Internal::implicit_context != nullptr) &&
        ctx->begin_runtime_call(
            Internal::RUNTIME_ALLOCATE_DEFERRED_VALUE_CALL, prov))
      start = Realm::Clock::current_time_in_nanoseconds();
    // Figure out the memory
    const Memory memory =
        request.is_exact ?
            request.memory.exact :
            find_memory_by_kind(ctx, request.memory.kind, true /*value*/);
    // Create the instance layout
    const Realm::Point<1, coord_t> zero(0);
    Realm::IndexSpace<1, coord_t> bounds = Realm::Rect<1, coord_t>(zero, zero);
    const std::vector<size_t> field_sizes(1, request.field_size);
    Realm::InstanceLayoutConstraints constraints(field_sizes, 0 /*blocking*/);
    int dim_order[1];
    dim_order[0] = 0;
    Realm::InstanceLayoutGeneric* layout =
        Realm::InstanceLayoutGeneric::choose_instance_layout(
            bounds, constraints, dim_order);
    layout->alignment_reqd = request.alignment;
    Internal::RtEvent use_event;
    UntypedDeferredValue result;
    result.instance = ctx->create_task_local_instance(
        memory, *layout, request.can_fail, request.escaping, use_event);
    delete layout;
    if (result.instance.exists() && (request.initial_value != nullptr))
    {
      const Processor exec_proc = ctx->get_executing_processor();
      Machine machine = Realm::Machine::get_machine();
      if (machine.has_affinity(exec_proc, memory))
      {
        if (use_event.exists())
          use_event.wait();
        // Has affinity so we shold jsut be able to memcpy this
        void* ptr =
            result.instance.pointer_untyped(0 /*offset*/, request.field_size);
        std::memcpy(ptr, request.initial_value, request.field_size);
      }
      else
      {
        Realm::ProfilingRequestSet no_requests;
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(result.instance, 0 /*field id*/, request.field_size);
        const Internal::RtEvent wait_on(bounds.fill(
            dsts, no_requests, request.initial_value, request.field_size,
            use_event));
        if (wait_on.exists())
          wait_on.wait();
      }
    }
    else if (use_event.exists())
      use_event.wait();
    if (Internal::implicit_context != nullptr)
      ctx->end_runtime_call(
          Internal::RUNTIME_ALLOCATE_DEFERRED_VALUE_CALL, prov, start,
          (start == 0) ? 0 : Realm::Clock::current_time_in_nanoseconds());
    return result;
  }

  //--------------------------------------------------------------------------
  void Runtime::destroy_deferred_value(
      Context ctx, UntypedDeferredValue& value, Realm::Event precondition)
  //--------------------------------------------------------------------------
  {
    // Don't check against implicit_context here because this method might
    // be called from OpenMP processors which don't have implicit_context set
    if ((ctx == nullptr) || ((Internal::implicit_context != nullptr) &&
                             (Internal::implicit_context != ctx)))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Invalid task context passed to Runtime::destroy_deferred_value";
      error.raise();
    }
    unsigned long long start = 0;
    Internal::AutoProvenance prov;
    // Only do timing on the main thread
    if ((Internal::implicit_context != nullptr) &&
        ctx->begin_runtime_call(
            Internal::RUNTIME_DESTROY_DEFERRED_VALUE_CALL, prov))
      start = Realm::Clock::current_time_in_nanoseconds();
    // Don't trust events passed in by users to be safe from poison
    if (precondition.exists())
      ctx->destroy_task_local_instance(
          value.instance,
          Internal::RtEvent(Realm::Event::ignorefaults(precondition)));
    else
      ctx->destroy_task_local_instance(
          value.instance, Internal::RtEvent::NO_RT_EVENT);
    value.instance = Realm::RegionInstance::NO_INST;
    if (Internal::implicit_context != nullptr)
      ctx->end_runtime_call(
          Internal::RUNTIME_DESTROY_DEFERRED_VALUE_CALL, prov, start,
          (start == 0) ? 0 : Realm::Clock::current_time_in_nanoseconds());
  }

  //--------------------------------------------------------------------------
  template<typename T>
  UntypedDeferredBuffer<T> Runtime::allocate_deferred_buffer(
      Context ctx, const DeferredBufferRequest& request)
  //--------------------------------------------------------------------------
  {
    // Don't check against implicit_context here because this method might
    // be called from OpenMP processors which don't have implicit_context set
    if ((ctx == nullptr) || ((Internal::implicit_context != nullptr) &&
                             (Internal::implicit_context != ctx)))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error
          << "Invalid task context passed to Runtime::allocate_deferred_buffer";
      error.raise();
    }
    unsigned long long start = 0;
    Internal::AutoProvenance prov;
    // Only do timing on the main thread
    if ((Internal::implicit_context != nullptr) &&
        ctx->begin_runtime_call(
            Internal::RUNTIME_ALLOCATE_DEFERRED_BUFFER_CALL, prov))
      start = Realm::Clock::current_time_in_nanoseconds();
    // Figure out the domain
    Domain domain;
    if (!request.is_value)
    {
      const int dim = request.bounds.name.get_dim();
      switch (dim)
      {
#define DIMFUNC(DIM)                                                \
  case DIM:                                                         \
    {                                                               \
      DomainT<DIM, coord_t> realm_is;                               \
      runtime->get_index_space_domain(                              \
          request.bounds.name, &realm_is,                           \
          Internal::NT_TemplateHelper::encode_tag<DIM, coord_t>()); \
      domain = realm_is;                                            \
      break;                                                        \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error
                << "Unsupported dimension " << dim << " for "
                << "Runtime::allocate_deferred_buffer. This probably means you "
                << "need to build Legion with support for more dimensions.";
            error.raise();
          }
      }
    }
    else
      domain = request.bounds.value;
    if ((domain.get_dim() < 1) || (LEGION_MAX_DIM < domain.get_dim()))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Domain for a deferred buffer creation in " << *ctx << " has "
            << domain.get_dim() << "dimensions which are not in the range [1,"
            << LEGION_MAX_DIM << "].";
      error.raise();
    }
    else if (
        !request.dim_order.empty() &&
        (request.dim_order.size() != size_t(domain.get_dim())))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Specified dimension ordering for a deferred buffer creation in "
            << *ctx << " has " << request.dim_order.size() << " dimensions "
            << "but the domain of the request has " << domain.get_dim()
            << " dimensions. The number of dimensions must match.";
      error.raise();
    }
    if (!domain.dense())
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Specified non-dense domain for deferred buffer creation in "
            << *ctx << ". Deferred buffers must always be dense currently.";
      error.raise();
    }
    // Figure out the memory
    const Memory memory =
        request.is_exact ?
            request.memory.exact :
            find_memory_by_kind(ctx, request.memory.kind, false /*value*/);
    const std::vector<size_t> field_sizes(1, request.field_size);
    Realm::InstanceLayoutConstraints constraints(field_sizes, 0 /*blocking*/);
    Realm::InstanceLayoutGeneric* layout = nullptr;
    switch (domain.get_dim())
    {
#define DIMFUNC(DIM)                                                       \
  case DIM:                                                                \
    {                                                                      \
      const DomainT<DIM, T> bounds = domain;                               \
      int dim_order[DIM];                                                  \
      if (request.dim_order.empty())                                       \
      {                                                                    \
        for (int i = 0; i < DIM; i++) dim_order[i] = DIM - (i + 1);        \
      }                                                                    \
      else                                                                 \
      {                                                                    \
        for (int i = 0; i < DIM; i++) dim_order[i] = request.dim_order[i]; \
      }                                                                    \
      layout = Realm::InstanceLayoutGeneric::choose_instance_layout(       \
          bounds, constraints, dim_order);                                 \
      break;                                                               \
    }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        std::abort();
    }
    layout->alignment_reqd = request.alignment;
    Internal::RtEvent use_event;
    UntypedDeferredBuffer<T> result;
    result.instance = ctx->create_task_local_instance(
        memory, *layout, request.can_fail, request.escaping, use_event);
    delete layout;
    if (result.instance.exists())
    {
      result.field_size = request.field_size;
      result.dims = domain.get_dim();
      if (request.initial_value != nullptr)
      {
        Realm::ProfilingRequestSet no_requests;
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(result.instance, 0 /*field id*/, request.field_size);
        switch (domain.get_dim())
        {
#define DIMFUNC(DIM)                                                      \
  case DIM:                                                               \
    {                                                                     \
      const Rect<DIM, T> bounds = domain;                                 \
      const Internal::RtEvent wait_on(bounds.fill(                        \
          dsts, no_requests, request.initial_value, request.field_size)); \
      if (wait_on.exists())                                               \
        wait_on.wait();                                                   \
      break;                                                              \
    }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
      }
      else if (use_event.exists())
        use_event.wait();
    }
    if (Internal::implicit_context != nullptr)
      ctx->end_runtime_call(
          Internal::RUNTIME_ALLOCATE_DEFERRED_BUFFER_CALL, prov, start,
          (start == 0) ? 0 : Realm::Clock::current_time_in_nanoseconds());
    return result;
  }

  // Intantiate this for the types of coordinates that Realm supports
  template UntypedDeferredBuffer<int> Runtime::allocate_deferred_buffer<int>(
      Context ctx, const DeferredBufferRequest&);
  template UntypedDeferredBuffer<unsigned> Runtime::allocate_deferred_buffer<
      unsigned>(Context ctx, const DeferredBufferRequest&);
  template UntypedDeferredBuffer<long long> Runtime::allocate_deferred_buffer<
      long long>(Context ctx, const DeferredBufferRequest&);

  //--------------------------------------------------------------------------
  template<typename T>
  void Runtime::destroy_deferred_buffer(
      Context ctx, UntypedDeferredBuffer<T>& buffer, Realm::Event precondition)
  //--------------------------------------------------------------------------
  {
    // Don't check against implicit_context here because this method might
    // be called from OpenMP processors which don't have implicit_context set
    if ((ctx == nullptr) || ((Internal::implicit_context != nullptr) &&
                             (Internal::implicit_context != ctx)))
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error
          << "Invalid task context passed to Runtime::destroy_deferred_buffer";
      error.raise();
    }
    unsigned long long start = 0;
    Internal::AutoProvenance prov;
    // Only do timing on the main thread
    if ((Internal::implicit_context != nullptr) &&
        ctx->begin_runtime_call(
            Internal::RUNTIME_DESTROY_DEFERRED_BUFFER_CALL, prov))
      start = Realm::Clock::current_time_in_nanoseconds();
    // Don't trust events passed in by users to be safe from poison
    if (precondition.exists())
      ctx->destroy_task_local_instance(
          buffer.instance,
          Internal::RtEvent(Realm::Event::ignorefaults(precondition)));
    else
      ctx->destroy_task_local_instance(
          buffer.instance, Internal::RtEvent::NO_RT_EVENT);
    buffer.instance = Realm::RegionInstance::NO_INST;
    buffer.field_size = 0;
    buffer.dims = 0;
    if (Internal::implicit_context != nullptr)
      ctx->end_runtime_call(
          Internal::RUNTIME_DESTROY_DEFERRED_BUFFER_CALL, prov, start,
          (start == 0) ? 0 : Realm::Clock::current_time_in_nanoseconds());
  }

  // Intantiate this for the types of coordinates that Realm supports
  template void Runtime::destroy_deferred_buffer<int>(
      Context ctx, UntypedDeferredBuffer<int>&, Realm::Event);
  template void Runtime::destroy_deferred_buffer<unsigned>(
      Context ctx, UntypedDeferredBuffer<unsigned>&, Realm::Event);
  template void Runtime::destroy_deferred_buffer<long long>(
      Context ctx, UntypedDeferredBuffer<long long>&, Realm::Event);

  //--------------------------------------------------------------------------
  Future Runtime::get_current_time(Context ctx, Future precondition)
  //--------------------------------------------------------------------------
  {
    TimingLauncher launcher(LEGION_MEASURE_SECONDS);
    launcher.add_precondition(precondition);
    return issue_timing_measurement(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  Future Runtime::get_current_time_in_microseconds(Context ctx, Future pre)
  //--------------------------------------------------------------------------
  {
    TimingLauncher launcher(LEGION_MEASURE_MICRO_SECONDS);
    launcher.add_precondition(pre);
    return issue_timing_measurement(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  Future Runtime::get_current_time_in_nanoseconds(Context ctx, Future pre)
  //--------------------------------------------------------------------------
  {
    TimingLauncher launcher(LEGION_MEASURE_NANO_SECONDS);
    launcher.add_precondition(pre);
    return issue_timing_measurement(ctx, launcher);
  }

  //--------------------------------------------------------------------------
  Future Runtime::issue_timing_measurement(
      Context ctx, const TimingLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ISSUE_TIMING_MEASUREMENT_CALL> call(
        launcher.provenance, ctx, __func__);
    return ctx->issue_timing_measurement(launcher, call);
  }

  //--------------------------------------------------------------------------
  /*static*/ long long Runtime::get_zero_time(void)
  //--------------------------------------------------------------------------
  {
    return Realm::Clock::get_zero_time();
  }

  //--------------------------------------------------------------------------
  void Runtime::push_exception_handler(Context ctx, ExceptionHandlerID handler)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_PUSH_EXCEPTION_HANDLER_CALL> call(ctx, __func__);
    ctx->push_exception_handler(handler);
  }

  //--------------------------------------------------------------------------
  Future Runtime::pop_exception_handler(Context ctx, const char* provenance)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_POP_EXCEPTION_HANDLER_CALL> call(
        provenance, ctx, __func__);
    return ctx->pop_exception_handler(call);
  }

  //--------------------------------------------------------------------------
  Mapping::Mapper* Runtime::get_mapper(
      Context ctx, MapperID id, Processor target)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_MAPPER_CALL> call(ctx, __func__);
    if (target.exists())
      return runtime->get_mapper(id, target);
    else
      return runtime->get_mapper(id, ctx->get_executing_processor());
  }

  //--------------------------------------------------------------------------
  Mapping::MapperContext Runtime::begin_mapper_call(
      Context ctx, MapperID id, Processor target)
  //--------------------------------------------------------------------------
  {
    // Cannot use auto-call here for profiling
    if (ctx != Internal::implicit_context)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Invalid task context passed to Runtime::begin_mapper_call";
      error.raise();
    }
    if (target.exists())
      return runtime->begin_mapper_call(id, target, ctx->owner_task);
    else
      return runtime->begin_mapper_call(
          id, ctx->get_executing_processor(), ctx->owner_task);
  }

  //--------------------------------------------------------------------------
  void Runtime::end_mapper_call(Mapping::MapperContext ctx)
  //--------------------------------------------------------------------------
  {
    runtime->end_mapper_call(ctx);
  }

  //--------------------------------------------------------------------------
  Processor Runtime::get_executing_processor(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_EXECUTING_PROCESSOR_CALL> call(
        ctx, __func__);
    return ctx->get_executing_processor();
  }

  //--------------------------------------------------------------------------
  const Task* Runtime::get_current_task(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_GET_CURRENT_TASK_CALL> call(ctx, __func__);
    return ctx->get_task();
  }

  //--------------------------------------------------------------------------
  size_t Runtime::query_available_memory(
      Context ctx, Memory target, bool escaping)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_QUERY_AVAILABLE_MEMORY_CALL> call(ctx, __func__);
    return ctx->query_available_memory(target, escaping);
  }

  //--------------------------------------------------------------------------
  void Runtime::release_memory_pool(Context ctx, Memory target)
  //--------------------------------------------------------------------------
  {
    ctx->release_memory_pool(target);
  }

  //--------------------------------------------------------------------------
  void Runtime::raise_region_exception(
      Context ctx, PhysicalRegion region, bool nuclear)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_RAISE_REGION_EXCEPTION_CALL> call(ctx, __func__);
    ctx->raise_region_exception(region, nuclear);
  }

  //--------------------------------------------------------------------------
  void Runtime::yield(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_YIELD_CALL> call(ctx, __func__);
    ctx->yield();
  }

  //--------------------------------------------------------------------------
  void Runtime::record_asynchronous_effect(
      Context ctx, Realm::Event effect, const char* provenance)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_ASYNC_EFFECT_CALL> call(ctx, __func__);
    ctx->record_asynchronous_effect(Internal::ApEvent(effect), provenance);
  }

  //--------------------------------------------------------------------------
  void Runtime::concurrent_task_barrier(Context ctx)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CONCURRENT_TASK_BARRIER_CALL> call(
        ctx, __func__);
    ctx->concurrent_task_barrier();
  }

  //--------------------------------------------------------------------------
  void Runtime::start_profiling_range(Context ctx)
  //--------------------------------------------------------------------------
  {
    ctx->start_profiling_range();
  }

  //--------------------------------------------------------------------------
  void Runtime::stop_profiling_range(Context ctx, const char* provenance)
  //--------------------------------------------------------------------------
  {
    ctx->stop_profiling_range(provenance);
  }

  //--------------------------------------------------------------------------
  bool Runtime::is_MPI_interop_configured(void)
  //--------------------------------------------------------------------------
  {
    return runtime->is_MPI_interop_configured();
  }

  //--------------------------------------------------------------------------
  const std::map<int, AddressSpace>& Runtime::find_forward_MPI_mapping(void)
  //--------------------------------------------------------------------------
  {
    return runtime->find_forward_MPI_mapping();
  }

  //--------------------------------------------------------------------------
  const std::map<AddressSpace, int>& Runtime::find_reverse_MPI_mapping(void)
  //--------------------------------------------------------------------------
  {
    return runtime->find_reverse_MPI_mapping();
  }

  //--------------------------------------------------------------------------
  int Runtime::find_local_MPI_rank(void)
  //--------------------------------------------------------------------------
  {
    return runtime->find_local_MPI_rank();
  }

  //--------------------------------------------------------------------------
  Mapping::MapperRuntime* Runtime::get_mapper_runtime(void)
  //--------------------------------------------------------------------------
  {
    return runtime->get_mapper_runtime();
  }

  //--------------------------------------------------------------------------
  MapperID Runtime::generate_dynamic_mapper_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_mapper_id();
  }

  //--------------------------------------------------------------------------
  MapperID Runtime::generate_library_mapper_ids(const char* name, size_t cnt)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_mapper_ids(name, cnt);
  }

  //--------------------------------------------------------------------------
  /*static*/ MapperID Runtime::generate_static_mapper_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_mapper_id();
  }

  //--------------------------------------------------------------------------
  void Runtime::add_mapper(
      MapperID map_id, Mapping::Mapper* mapper, Processor proc)
  //--------------------------------------------------------------------------
  {
    runtime->add_mapper(map_id, mapper, proc);
  }

  //--------------------------------------------------------------------------
  void Runtime::replace_default_mapper(Mapping::Mapper* mapper, Processor proc)
  //--------------------------------------------------------------------------
  {
    runtime->replace_default_mapper(mapper, proc);
  }

  //--------------------------------------------------------------------------
  ProjectionID Runtime::generate_dynamic_projection_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_projection_id();
  }

  //--------------------------------------------------------------------------
  ProjectionID Runtime::generate_library_projection_ids(
      const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_projection_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ ProjectionID Runtime::generate_static_projection_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_projection_id();
  }

  //--------------------------------------------------------------------------
  void Runtime::register_projection_functor(
      ProjectionID pid, ProjectionFunctor* func, bool silence_warnings,
      const char* warning_string)
  //--------------------------------------------------------------------------
  {
    runtime->register_projection_functor(
        pid, func, true /*need zero check*/, silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::preregister_projection_functor(
      ProjectionID pid, ProjectionFunctor* func)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::preregister_projection_functor(pid, func);
  }

  //--------------------------------------------------------------------------
  /*static*/ ProjectionFunctor* Runtime::get_projection_functor(
      ProjectionID pid)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::get_projection_functor(pid);
  }

  //--------------------------------------------------------------------------
  ShardingID Runtime::generate_dynamic_sharding_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_sharding_id();
  }

  //--------------------------------------------------------------------------
  ShardingID Runtime::generate_library_sharding_ids(
      const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_sharding_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ ShardingID Runtime::generate_static_sharding_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_sharding_id();
  }

  //--------------------------------------------------------------------------
  void Runtime::register_sharding_functor(
      ShardingID sid, ShardingFunctor* functor, bool silence_warnings,
      const char* warning_string)
  //--------------------------------------------------------------------------
  {
    runtime->register_sharding_functor(
        sid, functor, true /*need zero check*/, silence_warnings,
        warning_string);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::preregister_sharding_functor(
      ShardingID sid, ShardingFunctor* func)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::preregister_sharding_functor(sid, func);
  }

  //--------------------------------------------------------------------------
  /*static*/ ShardingFunctor* Runtime::get_sharding_functor(ShardingID sid)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::get_sharding_functor(sid);
  }

  //--------------------------------------------------------------------------
  ConcurrentID Runtime::generate_dynamic_concurrent_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_concurrent_id();
  }

  //--------------------------------------------------------------------------
  ConcurrentID Runtime::generate_library_concurrent_ids(
      const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_concurrent_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ ConcurrentID Runtime::generate_static_concurrent_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_concurrent_id();
  }

  //--------------------------------------------------------------------------
  void Runtime::register_concurrent_coloring_functor(
      ConcurrentID cid, ConcurrentColoringFunctor* functor,
      bool silence_warnings, const char* warning_string)
  //--------------------------------------------------------------------------
  {
    runtime->register_concurrent_functor(
        cid, functor, silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::preregister_concurrent_coloring_functor(
      ConcurrentID cid, ConcurrentColoringFunctor* functor)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::preregister_concurrent_functor(cid, functor);
  }

  //--------------------------------------------------------------------------
  /*static*/ ConcurrentColoringFunctor*
      Runtime::get_concurrent_coloring_functor(ConcurrentID cid)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::get_concurrent_functor(cid);
  }

  //--------------------------------------------------------------------------
  ExceptionHandlerID Runtime::generate_dynamic_exception_handler_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_exception_handler_id();
  }

  //--------------------------------------------------------------------------
  ExceptionHandlerID Runtime::generate_library_exception_handler_ids(
      const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_exception_handler_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ ExceptionHandlerID Runtime::generate_static_exception_handler_id(
      void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_exception_handler_id();
  }

  //--------------------------------------------------------------------------
  void Runtime::register_exception_handler(
      ExceptionHandlerID hid, ExceptionHandler* handler)
  //--------------------------------------------------------------------------
  {
    runtime->register_exception_handler(hid, handler);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::preregister_exception_handler(
      ExceptionHandlerID hid, ExceptionHandler* handler)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::preregister_exception_handler(hid, handler);
  }

  //--------------------------------------------------------------------------
  /*static*/ ExceptionHandler* Runtime::get_exception_handler(
      ExceptionHandlerID hid)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::get_exception_handler(hid);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_semantic_information(
      TaskID task_id, SemanticTag tag, const void* buffer, size_t size,
      bool is_mut, bool local)
  //--------------------------------------------------------------------------
  {
    runtime->attach_semantic_information(
        task_id, tag, buffer, size, is_mut, !local);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_semantic_information(
      IndexSpace handle, SemanticTag tag, const void* buffer, size_t size,
      bool is_mut)
  //--------------------------------------------------------------------------
  {
    runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_semantic_information(
      IndexPartition handle, SemanticTag tag, const void* buffer, size_t size,
      bool is_mut)
  //--------------------------------------------------------------------------
  {
    runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_semantic_information(
      FieldSpace handle, SemanticTag tag, const void* buffer, size_t size,
      bool is_mut)
  //--------------------------------------------------------------------------
  {
    runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_semantic_information(
      FieldSpace handle, FieldID fid, SemanticTag tag, const void* buffer,
      size_t size, bool is_mut)
  //--------------------------------------------------------------------------
  {
    runtime->attach_semantic_information(
        handle, fid, tag, buffer, size, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_semantic_information(
      LogicalRegion handle, SemanticTag tag, const void* buffer, size_t size,
      bool is_mut)
  //--------------------------------------------------------------------------
  {
    runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_semantic_information(
      LogicalPartition handle, SemanticTag tag, const void* buffer, size_t size,
      bool is_mut)
  //--------------------------------------------------------------------------
  {
    runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_name(
      TaskID task_id, const char* name, bool is_mutable, bool local_only)
  //--------------------------------------------------------------------------
  {
    Runtime::attach_semantic_information(
        task_id, LEGION_NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mutable,
        local_only);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_name(IndexSpace handle, const char* name, bool is_mut)
  //--------------------------------------------------------------------------
  {
    Runtime::attach_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_name(IndexPartition handle, const char* name, bool ism)
  //--------------------------------------------------------------------------
  {
    Runtime::attach_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, name, strlen(name) + 1, ism);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_name(FieldSpace handle, const char* name, bool is_mut)
  //--------------------------------------------------------------------------
  {
    Runtime::attach_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mut);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_name(
      FieldSpace handle, FieldID fid, const char* name, bool is_mutable)
  //--------------------------------------------------------------------------
  {
    Runtime::attach_semantic_information(
        handle, fid, LEGION_NAME_SEMANTIC_TAG, name, strlen(name) + 1,
        is_mutable);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_name(LogicalRegion handle, const char* name, bool ism)
  //--------------------------------------------------------------------------
  {
    Runtime::attach_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, name, strlen(name) + 1, ism);
  }

  //--------------------------------------------------------------------------
  void Runtime::attach_name(LogicalPartition handle, const char* name, bool m)
  //--------------------------------------------------------------------------
  {
    Runtime::attach_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, name, strlen(name) + 1, m);
  }

  //--------------------------------------------------------------------------
  bool Runtime::retrieve_semantic_information(
      TaskID task_id, SemanticTag tag, const void*& result, size_t& size,
      bool can_fail, bool wait_until)
  //--------------------------------------------------------------------------
  {
    return runtime->retrieve_semantic_information(
        task_id, tag, result, size, can_fail, wait_until);
  }

  //--------------------------------------------------------------------------
  bool Runtime::retrieve_semantic_information(
      IndexSpace handle, SemanticTag tag, const void*& result, size_t& size,
      bool can_fail, bool wait_until)
  //--------------------------------------------------------------------------
  {
    return runtime->retrieve_semantic_information(
        handle, tag, result, size, can_fail, wait_until);
  }

  //--------------------------------------------------------------------------
  bool Runtime::retrieve_semantic_information(
      IndexPartition handle, SemanticTag tag, const void*& result, size_t& size,
      bool can_fail, bool wait_until)
  //--------------------------------------------------------------------------
  {
    return runtime->retrieve_semantic_information(
        handle, tag, result, size, can_fail, wait_until);
  }

  //--------------------------------------------------------------------------
  bool Runtime::retrieve_semantic_information(
      FieldSpace handle, SemanticTag tag, const void*& result, size_t& size,
      bool can_fail, bool wait_until)
  //--------------------------------------------------------------------------
  {
    return runtime->retrieve_semantic_information(
        handle, tag, result, size, can_fail, wait_until);
  }

  //--------------------------------------------------------------------------
  bool Runtime::retrieve_semantic_information(
      FieldSpace handle, FieldID fid, SemanticTag tag, const void*& result,
      size_t& size, bool can_fail, bool wait_until)
  //--------------------------------------------------------------------------
  {
    return runtime->retrieve_semantic_information(
        handle, fid, tag, result, size, can_fail, wait_until);
  }

  //--------------------------------------------------------------------------
  bool Runtime::retrieve_semantic_information(
      LogicalRegion handle, SemanticTag tag, const void*& result, size_t& size,
      bool can_fail, bool wait_until)
  //--------------------------------------------------------------------------
  {
    return runtime->retrieve_semantic_information(
        handle, tag, result, size, can_fail, wait_until);
  }

  //--------------------------------------------------------------------------
  bool Runtime::retrieve_semantic_information(
      LogicalPartition part, SemanticTag tag, const void*& result, size_t& size,
      bool can_fail, bool wait_until)
  //--------------------------------------------------------------------------
  {
    return runtime->retrieve_semantic_information(
        part, tag, result, size, can_fail, wait_until);
  }

  //--------------------------------------------------------------------------
  void Runtime::retrieve_name(TaskID task_id, const char*& result)
  //--------------------------------------------------------------------------
  {
    const void* dummy_ptr;
    size_t dummy_size;
    Runtime::retrieve_semantic_information(
        task_id, LEGION_NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
    static_assert(sizeof(dummy_ptr) == sizeof(result));
    memcpy(&result, &dummy_ptr, sizeof(result));
  }

  //--------------------------------------------------------------------------
  void Runtime::retrieve_name(IndexSpace handle, const char*& result)
  //--------------------------------------------------------------------------
  {
    const void* dummy_ptr;
    size_t dummy_size;
    Runtime::retrieve_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
    static_assert(sizeof(dummy_ptr) == sizeof(result));
    memcpy(&result, &dummy_ptr, sizeof(result));
  }

  //--------------------------------------------------------------------------
  void Runtime::retrieve_name(IndexPartition handle, const char*& result)
  //--------------------------------------------------------------------------
  {
    const void* dummy_ptr;
    size_t dummy_size;
    Runtime::retrieve_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
    static_assert(sizeof(dummy_ptr) == sizeof(result));
    memcpy(&result, &dummy_ptr, sizeof(result));
  }

  //--------------------------------------------------------------------------
  void Runtime::retrieve_name(FieldSpace handle, const char*& result)
  //--------------------------------------------------------------------------
  {
    const void* dummy_ptr;
    size_t dummy_size;
    Runtime::retrieve_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
    static_assert(sizeof(dummy_ptr) == sizeof(result));
    memcpy(&result, &dummy_ptr, sizeof(result));
  }

  //--------------------------------------------------------------------------
  void Runtime::retrieve_name(
      FieldSpace handle, FieldID fid, const char*& result)
  //--------------------------------------------------------------------------
  {
    const void* dummy_ptr;
    size_t dummy_size;
    Runtime::retrieve_semantic_information(
        handle, fid, LEGION_NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false,
        false);
    static_assert(sizeof(dummy_ptr) == sizeof(result));
    memcpy(&result, &dummy_ptr, sizeof(result));
  }

  //--------------------------------------------------------------------------
  void Runtime::retrieve_name(LogicalRegion handle, const char*& result)
  //--------------------------------------------------------------------------
  {
    const void* dummy_ptr;
    size_t dummy_size;
    Runtime::retrieve_semantic_information(
        handle, LEGION_NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
    static_assert(sizeof(dummy_ptr) == sizeof(result));
    memcpy(&result, &dummy_ptr, sizeof(result));
  }

  //--------------------------------------------------------------------------
  void Runtime::retrieve_name(LogicalPartition part, const char*& result)
  //--------------------------------------------------------------------------
  {
    const void* dummy_ptr;
    size_t dummy_size;
    Runtime::retrieve_semantic_information(
        part, LEGION_NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
    static_assert(sizeof(dummy_ptr) == sizeof(result));
    memcpy(&result, &dummy_ptr, sizeof(result));
  }

  //--------------------------------------------------------------------------
  void Runtime::print_once(Context ctx, FILE* f, const char* message)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_PRINT_ONCE_CALL> call(ctx, __func__);
    ctx->print_once(f, message);
  }

  //--------------------------------------------------------------------------
  void Runtime::log_once(Context ctx, Realm::LoggerMessage& message)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_LOG_ONCE_CALL> call(ctx, __func__);
    ctx->log_once(message);
  }

  //--------------------------------------------------------------------------
  /*static*/ const char* Runtime::get_legion_version(void)
  //--------------------------------------------------------------------------
  {
    static const char* legion_runtime_version = LEGION_VERSION;
    return legion_runtime_version;
  }

  //--------------------------------------------------------------------------
  /*static*/ int Runtime::start(
      int argc, char** argv, bool background, bool default_mapper, bool filter,
      bool init_network, bool configure_machine)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::start(
        argc, argv, background, default_mapper, filter, init_network,
        configure_machine);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::initialize(
      int* argc, char*** argv, bool filter, bool parse, bool init_network,
      bool configure_machine)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::initialize(
        argc, argv, parse, filter, init_network, configure_machine);
  }

  //--------------------------------------------------------------------------
  /*static*/ int Runtime::wait_for_shutdown(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::wait_for_shutdown();
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::set_return_code(int return_code)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::set_return_code(return_code);
  }

  //--------------------------------------------------------------------------
  Future Runtime::launch_top_level_task(const TaskLauncher& launcher)
  //--------------------------------------------------------------------------
  {
    return runtime->launch_top_level_task(launcher);
  }

  //--------------------------------------------------------------------------
  Context Runtime::begin_implicit_task(
      TaskID top_task_id, MapperID top_mapper_id, Processor::Kind proc_kind,
      const char* task_name, bool control_replicable,
      unsigned shard_per_address_space, int shard_id, DomainPoint point)
  //--------------------------------------------------------------------------
  {
    return runtime->begin_implicit_task(
        top_task_id, top_mapper_id, proc_kind, task_name, control_replicable,
        shard_per_address_space, shard_id, point);
  }

  //--------------------------------------------------------------------------
  void Runtime::unbind_implicit_task_from_external_thread(Context ctx)
  //--------------------------------------------------------------------------
  {
    runtime->unbind_implicit_task_from_external_thread(ctx);
  }

  //--------------------------------------------------------------------------
  void Runtime::bind_implicit_task_to_external_thread(Context ctx)
  //--------------------------------------------------------------------------
  {
    runtime->bind_implicit_task_to_external_thread(ctx);
  }

  //--------------------------------------------------------------------------
  void Runtime::finish_implicit_task(Context ctx, Realm::Event effects)
  //--------------------------------------------------------------------------
  {
    runtime->finish_implicit_task(ctx, Internal::ApEvent(effects));
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::set_top_level_task_id(TaskID top_id)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::set_top_level_task_id(top_id);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::set_top_level_task_mapper_id(MapperID mapper_id)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::set_top_level_task_mapper_id(mapper_id);
  }

  //--------------------------------------------------------------------------
  /*static*/ size_t Runtime::get_maximum_dimension(void)
  //--------------------------------------------------------------------------
  {
    return LEGION_MAX_DIM;
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::configure_MPI_interoperability(int rank)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::configure_MPI_interoperability(rank);
  }

  //--------------------------------------------------------------------------
  /*static*/ LegionHandshake Runtime::create_external_handshake(
      bool init_in_ext, int ext_participants, int legion_participants)
  //--------------------------------------------------------------------------
  {
    if (ext_participants != 1)
    {
      Fatal fatal;
      fatal << "Legion does not currently support creating handshake with a "
            << "value for 'external_participants' different than '1'.";
      fatal.raise();
    }
    if (legion_participants != 1)
    {
      Fatal fatal;
      fatal << "Legion does not currently support creating handshake with a "
            << "value for 'legion_participants' different than '1'.";
      fatal.raise();
    }
    LegionHandshake result(new Internal::LegionHandshakeImpl(init_in_ext));
    Internal::Runtime::register_handshake(result);
    return result;
  }

  //--------------------------------------------------------------------------
  /*static*/ MPILegionHandshake Runtime::create_handshake(
      bool init_in_MPI, int mpi_participants, int legion_participants)
  //--------------------------------------------------------------------------
  {
    if (mpi_participants != 1)
    {
      Fatal fatal;
      fatal << "Legion does not currently support creating handshake with a "
            << "value for 'mpi_participants' different than '1'.";
      fatal.raise();
    }
    if (legion_participants != 1)
    {
      Fatal fatal;
      fatal << "Legion does not currently support creating handshake with a "
            << "value for 'legion_participants' different than '1'.";
      fatal.raise();
    }
    MPILegionHandshake result(new Internal::LegionHandshakeImpl(init_in_MPI));
    Internal::Runtime::register_handshake(result);
    return result;
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::register_reduction_op(
      ReductionOpID redop_id, ReductionOp* redop, SerdezInitFunc init_func,
      SerdezFoldFunc fold_func, bool permit_duplicates)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::register_reduction_op(
        redop_id, redop, init_func, fold_func, permit_duplicates);
  }

  //--------------------------------------------------------------------------
  /*static*/ const ReductionOp* Runtime::get_reduction_op(
      ReductionOpID redop_id)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::get_reduction_op(redop_id);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::register_custom_serdez_op(
      CustomSerdezID serdez_id, SerdezOp* serdez_op, bool permit_duplicates)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::register_serdez_op(
        serdez_id, serdez_op, permit_duplicates);
  }

  //--------------------------------------------------------------------------
  /*static*/ const SerdezOp* Runtime::get_serdez_op(CustomSerdezID serdez_id)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::get_serdez_op(serdez_id);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::add_registration_callback(
      RegistrationCallback callback, bool dedup, size_t dedup_tag)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::add_registration_callback(callback, dedup, dedup_tag);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::add_registration_callback(
      RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
      bool dedup, size_t dedup_tag)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::add_registration_callback(
        callback, buffer, dedup, dedup_tag);
  }

  //--------------------------------------------------------------------------
  void Runtime::perform_registration_callback(
      RegistrationCallback callback, bool global, bool deduplicate,
      size_t dedup_tag)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::perform_dynamic_registration_callback(
        callback, global, deduplicate, dedup_tag);
  }

  //--------------------------------------------------------------------------
  void Runtime::perform_registration_callback(
      RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
      bool global, bool deduplicate, size_t dedup_tag)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::perform_dynamic_registration_callback(
        callback, buffer, global, deduplicate, dedup_tag);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::set_registration_callback(
      RegistrationCallback callback)
  //--------------------------------------------------------------------------
  {
    Internal::Runtime::add_registration_callback(callback, true /*dedup*/, 0);
  }

  //--------------------------------------------------------------------------
  /*static*/ const InputArgs& Runtime::get_input_args(void)
  //--------------------------------------------------------------------------
  {
    if (!Internal::Runtime::runtime_started)
    {
      Error error(LEGION_STARTUP_EXCEPTION);
      error << "Illegal call to 'Runtime::get_input_args' before the "
            << "runtime is started";
      error.raise();
    }
    return Internal::runtime->input_args;
  }

  //--------------------------------------------------------------------------
  /*static*/ bool Runtime::has_runtime(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::runtime_started;
  }

  //--------------------------------------------------------------------------
  /*static*/ Runtime* Runtime::get_runtime(Processor p)
  //--------------------------------------------------------------------------
  {
    if (!Internal::Runtime::runtime_started)
    {
      Error error(LEGION_STARTUP_EXCEPTION);
      error << "Illegal call to 'Runtime::get_runtime' before the "
            << "runtime is started";
      error.raise();
    }
    return Internal::runtime->external;
  }

  //--------------------------------------------------------------------------
  /*static*/ bool Runtime::has_context(void)
  //--------------------------------------------------------------------------
  {
    return (Internal::implicit_context != nullptr);
  }

  //--------------------------------------------------------------------------
  /*static*/ Context Runtime::get_context(void)
  //--------------------------------------------------------------------------
  {
    return Internal::implicit_context;
  }

  //--------------------------------------------------------------------------
  /*static*/ const Task* Runtime::get_context_task(Context ctx)
  //--------------------------------------------------------------------------
  {
    return ctx->get_owner_task();
  }

  //--------------------------------------------------------------------------
  TaskID Runtime::generate_dynamic_task_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_task_id();
  }

  //--------------------------------------------------------------------------
  TaskID Runtime::generate_library_task_ids(const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_task_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ TaskID Runtime::generate_static_task_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_task_id();
  }

  //--------------------------------------------------------------------------
  ReductionOpID Runtime::generate_dynamic_reduction_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_reduction_id();
  }

  //--------------------------------------------------------------------------
  ReductionOpID Runtime::generate_library_reduction_ids(
      const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_reduction_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ ReductionOpID Runtime::generate_static_reduction_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_reduction_id();
  }

  //--------------------------------------------------------------------------
  CustomSerdezID Runtime::generate_dynamic_serdez_id(void)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_dynamic_serdez_id();
  }

  //--------------------------------------------------------------------------
  CustomSerdezID Runtime::generate_library_serdez_ids(
      const char* name, size_t count)
  //--------------------------------------------------------------------------
  {
    return runtime->generate_library_serdez_ids(name, count);
  }

  //--------------------------------------------------------------------------
  /*static*/ CustomSerdezID Runtime::generate_static_serdez_id(void)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::generate_static_serdez_id();
  }

  //--------------------------------------------------------------------------
  VariantID Runtime::register_task_variant(
      const TaskVariantRegistrar& registrar, const CodeDescriptor& realm_desc,
      const void* user_data /*= nullptr*/, size_t user_len /*= 0*/,
      size_t return_type_size /*=MAX_RETURN_SIZE*/,
      VariantID vid /*= AUTO_GENERATE_ID*/,
      bool has_return_type_size /*= true*/)
  //--------------------------------------------------------------------------
  {
    return runtime->register_variant(
        registrar, user_data, user_len, realm_desc, return_type_size,
        has_return_type_size, vid);
  }

  //--------------------------------------------------------------------------
  /*static*/ VariantID Runtime::preregister_task_variant(
      const TaskVariantRegistrar& registrar, const CodeDescriptor& realm_desc,
      const void* user_data /*= nullptr*/, size_t user_len /*= 0*/,
      const char* task_name /*= nullptr*/, VariantID vid /*=AUTO_GENERATE_ID*/,
      size_t return_type_size /*=MAX_RETURN_SIZE*/,
      bool has_return_type_size /*=true*/, bool check_task_id /*=true*/)
  //--------------------------------------------------------------------------
  {
    // Make a copy of the descriptor here
    return Internal::Runtime::preregister_variant(
        registrar, user_data, user_len, realm_desc, return_type_size,
        has_return_type_size, task_name, vid, check_task_id);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::legion_task_preamble(
      const void* data, size_t datalen, Processor p, const Task*& task,
      const std::vector<PhysicalRegion>*& reg, Context& ctx, Runtime*& runtime)
  //--------------------------------------------------------------------------
  {
    // Read the context out of the buffer
    legion_assert(datalen == sizeof(Context));
    ctx = *((const Context*)data);
    task = ctx->get_task();
    const Processor exec_proc = Processor::get_executing_processor();
    legion_assert(exec_proc.exists());
    reg = &ctx->begin_task(exec_proc);
    runtime = Internal::runtime->external;
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::legion_task_postamble(
      Context ctx, const void* retvalptr, size_t retvalsize, bool owned,
      Realm::RegionInstance inst, const void* metadataptr, size_t metadatasize)
  //--------------------------------------------------------------------------
  {
    ctx->end_task(
        retvalptr, retvalsize, owned, inst, nullptr /*functor*/,
        nullptr /*resource*/, nullptr /*freefunc*/, metadataptr, metadatasize,
        Internal::ApEvent::NO_AP_EVENT);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::legion_task_postamble(
      Context ctx, FutureFunctor* callback_functor, bool owned)
  //--------------------------------------------------------------------------
  {
    ctx->end_task(
        nullptr, 0, owned, Realm::RegionInstance::NO_INST, callback_functor,
        nullptr /*resource*/, nullptr /*freefunc*/, nullptr, 0,
        Internal::ApEvent::NO_AP_EVENT);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::legion_task_postamble(
      Context ctx, const void* ptr, size_t size, bool owned,
      const Realm::ExternalInstanceResource& resource,
      void (*freefunc)(const Realm::ExternalInstanceResource&),
      const void* metadataptr, size_t metadatasize)
  //--------------------------------------------------------------------------
  {
    ctx->end_task(
        ptr, size, owned, Realm::RegionInstance::NO_INST, nullptr /*functor*/,
        &resource, freefunc, metadataptr, metadatasize,
        Internal::ApEvent::NO_AP_EVENT);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::legion_task_postamble(
      Context ctx, const Domain& domain, bool take_ownership,
      const void* metadataptr, size_t metadatasize)
  //--------------------------------------------------------------------------
  {
    if (!domain.dense())
    {
      if (!take_ownership)
      {
        SparsityReferenceHelper functor(domain);
        Internal::NT_TemplateHelper::demux<SparsityReferenceHelper>(
            domain.is_type, &functor);
      }
      Domain* copy = new Domain(domain);
      Realm::ExternalMemoryResource resource(copy, sizeof(domain));
      Runtime::legion_task_postamble(
          ctx, copy, sizeof(domain), true /*owned*/, resource,
          SparsityReferenceHelper::deletion_function, metadataptr,
          metadatasize);
    }
    else
      Runtime::legion_task_postamble(
          ctx, &domain, sizeof(domain), false /*owned*/,
          Realm::RegionInstance::NO_INST, metadataptr, metadatasize);
  }

  //--------------------------------------------------------------------------
  ShardID Runtime::get_shard_id(Context ctx, bool I_know_what_I_am_doing)
  //--------------------------------------------------------------------------
  {
    if (!I_know_what_I_am_doing)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "User does not know what they are doing asking for a shard ID";
      error.raise();
    }
    const Task* task = get_local_task(ctx);
    return task->get_shard_id();
  }

  //--------------------------------------------------------------------------
  size_t Runtime::get_num_shards(Context ctx, bool I_know_what_I_am_doing)
  //--------------------------------------------------------------------------
  {
    if (!I_know_what_I_am_doing)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "User does not know what they are doing asking for the "
            << "number of shards";
      error.raise();
    }
    const Task* task = get_local_task(ctx);
    return task->get_total_shards();
  }

  //--------------------------------------------------------------------------
  Future Runtime::consensus_match(
      Context ctx, const void* input, void* output, size_t num_elements,
      size_t element_size, const char* prov)
  //--------------------------------------------------------------------------
  {
    AutoCall<Internal::RUNTIME_CONSENSUS_MATCH_CALL> call(prov, ctx, __func__);
    return ctx->consensus_match(
        input, output, num_elements, element_size, call);
  }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::enable_profiling(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::disable_profiling(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  /*static*/ void Runtime::dump_profiling(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  LayoutConstraintID Runtime::register_layout(
      const LayoutConstraintRegistrar& registrar)
  //--------------------------------------------------------------------------
  {
    return runtime->register_layout(registrar, LEGION_AUTO_GENERATE_ID);
  }

  //--------------------------------------------------------------------------
  void Runtime::release_layout(LayoutConstraintID layout_id)
  //--------------------------------------------------------------------------
  {
    runtime->release_layout(layout_id);
  }

  //--------------------------------------------------------------------------
  /*static*/ LayoutConstraintID Runtime::preregister_layout(
      const LayoutConstraintRegistrar& registrar, LayoutConstraintID layout_id)
  //--------------------------------------------------------------------------
  {
    return Internal::Runtime::preregister_layout(registrar, layout_id);
  }

  //--------------------------------------------------------------------------
  FieldSpace Runtime::get_layout_constraint_field_space(
      LayoutConstraintID layout_id)
  //--------------------------------------------------------------------------
  {
    return runtime->get_layout_constraint_field_space(layout_id);
  }

  //--------------------------------------------------------------------------
  void Runtime::get_layout_constraints(
      LayoutConstraintID layout_id, LayoutConstraintSet& layout_constraints)
  //--------------------------------------------------------------------------
  {
    runtime->get_layout_constraints(layout_id, layout_constraints);
  }

  //--------------------------------------------------------------------------
  const char* Runtime::get_layout_constraints_name(LayoutConstraintID id)
  //--------------------------------------------------------------------------
  {
    return runtime->get_layout_constraints_name(id);
  }

}  // namespace Legion
