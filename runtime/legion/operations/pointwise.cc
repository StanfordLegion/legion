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

#include "legion/operations/pointwise.h"
#include "legion/analysis/logical.h"
#include "legion/analysis/projection.h"
#include "legion/api/functors_impl.h"
#include "legion/nodes/index.h"
#include "legion/operations/collective.h"
#include "legion/operations/copy.h"
#include "legion/operations/fill.h"
#include "legion/tasks/task.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Pointwise Dependence
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointwiseDependence::PointwiseDependence(void)
      : context_index(0), unique_id(0), kind(LAST_OP_KIND), region_index(0),
        domain(nullptr), projection(nullptr), sharding(nullptr), sharding_id(0),
        sharding_domain(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PointwiseDependence::PointwiseDependence(const LogicalUser& user)
      : context_index(user.ctx_index), unique_id(user.uid),
        kind(user.op->get_operation_kind()), region_index(user.idx),
        domain(user.shard_proj->domain),
        projection(user.shard_proj->projection),
        sharding(
            (user.shard_proj->sharding == nullptr) ?
                nullptr :
                user.shard_proj->sharding->functor),
        sharding_id(
            (user.shard_proj->sharding == nullptr) ?
                0 :
                user.shard_proj->sharding->sharding_id),
        sharding_domain(user.shard_proj->sharding_domain)
    //--------------------------------------------------------------------------
    {
      domain->add_base_expression_reference(POINTWISE_DEPENDENCE_REF);
      if (sharding_domain != nullptr)
        sharding_domain->add_base_expression_reference(
            POINTWISE_DEPENDENCE_REF);
    }

    //--------------------------------------------------------------------------
    PointwiseDependence::PointwiseDependence(const PointwiseDependence& rhs)
      : context_index(rhs.context_index), unique_id(rhs.unique_id),
        kind(rhs.kind), region_index(rhs.region_index), domain(rhs.domain),
        projection(rhs.projection), sharding(rhs.sharding),
        sharding_id(rhs.sharding_id), sharding_domain(rhs.sharding_domain)
    //--------------------------------------------------------------------------
    {
      if (domain != nullptr)
        domain->add_base_expression_reference(POINTWISE_DEPENDENCE_REF);
      if (sharding_domain != nullptr)
        sharding_domain->add_base_expression_reference(
            POINTWISE_DEPENDENCE_REF);
    }

    //--------------------------------------------------------------------------
    PointwiseDependence::PointwiseDependence(PointwiseDependence&& rhs) noexcept
      : context_index(rhs.context_index), unique_id(rhs.unique_id),
        kind(rhs.kind), region_index(rhs.region_index), domain(rhs.domain),
        projection(rhs.projection), sharding(rhs.sharding),
        sharding_id(rhs.sharding_id), sharding_domain(rhs.sharding_domain)
    //--------------------------------------------------------------------------
    {
      // Move references to ourselves
      rhs.domain = nullptr;
      rhs.sharding_domain = nullptr;
    }

    //--------------------------------------------------------------------------
    PointwiseDependence::~PointwiseDependence(void)
    //--------------------------------------------------------------------------
    {
      if ((domain != nullptr) &&
          domain->remove_base_expression_reference(POINTWISE_DEPENDENCE_REF))
        delete domain;
      if ((sharding_domain != nullptr) &&
          sharding_domain->remove_base_expression_reference(
              POINTWISE_DEPENDENCE_REF))
        delete sharding_domain;
    }

    //--------------------------------------------------------------------------
    PointwiseDependence& PointwiseDependence::operator=(
        const PointwiseDependence& rhs)
    //--------------------------------------------------------------------------
    {
      if ((domain != nullptr) &&
          domain->remove_base_expression_reference(POINTWISE_DEPENDENCE_REF))
        delete domain;
      if ((sharding_domain != nullptr) &&
          sharding_domain->remove_base_expression_reference(
              POINTWISE_DEPENDENCE_REF))
        delete sharding_domain;
      context_index = rhs.context_index;
      unique_id = rhs.unique_id;
      kind = rhs.kind;
      region_index = rhs.region_index;
      domain = rhs.domain;
      projection = rhs.projection;
      sharding = rhs.sharding;
      sharding_id = rhs.sharding_id;
      sharding_domain = rhs.sharding_domain;
      if (domain != nullptr)
        domain->add_base_expression_reference(POINTWISE_DEPENDENCE_REF);
      if (sharding_domain != nullptr)
        sharding_domain->add_base_expression_reference(
            POINTWISE_DEPENDENCE_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    PointwiseDependence& PointwiseDependence::operator=(
        PointwiseDependence&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if ((domain != nullptr) &&
          domain->remove_base_expression_reference(POINTWISE_DEPENDENCE_REF))
        delete domain;
      if ((sharding_domain != nullptr) &&
          sharding_domain->remove_base_expression_reference(
              POINTWISE_DEPENDENCE_REF))
        delete sharding_domain;
      context_index = rhs.context_index;
      unique_id = rhs.unique_id;
      kind = rhs.kind;
      region_index = rhs.region_index;
      domain = rhs.domain;
      projection = rhs.projection;
      sharding = rhs.sharding;
      sharding_id = rhs.sharding_id;
      sharding_domain = rhs.sharding_domain;
      // Just move over the references
      rhs.domain = nullptr;
      rhs.sharding_domain = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PointwiseDependence::matches(const LogicalUser& user) const
    //--------------------------------------------------------------------------
    {
      legion_assert(user.shard_proj != nullptr);
      if (context_index != user.ctx_index)
        return false;
      if (region_index != user.idx)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    void PointwiseDependence::find_dependences(
        const RegionRequirement& req,
        const std::vector<LogicalRegion>& point_regions,
        std::map<LogicalRegion, std::vector<DomainPoint> >& dependences) const
    //--------------------------------------------------------------------------
    {
      projection->find_inversions(
          kind, unique_id, region_index, req, domain, point_regions,
          dependences);
    }

    //--------------------------------------------------------------------------
    void PointwiseDependence::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(context_index);
      rez.serialize(unique_id);
      rez.serialize(kind);
      rez.serialize(region_index);
      rez.serialize(domain->handle);
      rez.serialize(projection->projection_id);
      rez.serialize(sharding_id);
      if (sharding_domain != nullptr)
        rez.serialize(sharding_domain->handle);
      else
        rez.serialize(IndexSpace::NO_SPACE);
    }

    //--------------------------------------------------------------------------
    void PointwiseDependence::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if ((domain != nullptr) &&
          domain->remove_base_expression_reference(POINTWISE_DEPENDENCE_REF))
        delete domain;
      if ((sharding_domain != nullptr) &&
          sharding_domain->remove_base_expression_reference(
              POINTWISE_DEPENDENCE_REF))
        delete sharding_domain;
      derez.deserialize(context_index);
      derez.deserialize(unique_id);
      derez.deserialize(kind);
      derez.deserialize(region_index);
      IndexSpace handle;
      derez.deserialize(handle);
      domain = runtime->get_node(handle);
      domain->add_base_expression_reference(POINTWISE_DEPENDENCE_REF);
      ProjectionID pid;
      derez.deserialize(pid);
      projection = runtime->find_projection_function(pid);
      derez.deserialize(sharding_id);
      sharding = runtime->find_sharding_functor(sharding_id);
      derez.deserialize(handle);
      if (handle.exists())
      {
        sharding_domain = runtime->get_node(handle);
        sharding_domain->add_base_expression_reference(
            POINTWISE_DEPENDENCE_REF);
      }
      else
        sharding_domain = nullptr;
    }

    /////////////////////////////////////////////////////////////
    // PointwiseAnalyzable
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    void PointwiseAnalyzable<OP>::activate(void)
    //--------------------------------------------------------------------------
    {
      OP::activate();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void PointwiseAnalyzable<OP>::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      pointwise_dependences.clear();
      OP::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    bool PointwiseAnalyzable<OP>::is_pointwise_analyzable(void) const
    //--------------------------------------------------------------------------
    {
      return runtime->enable_pointwise_analysis;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void PointwiseAnalyzable<OP>::register_pointwise_dependence(
        unsigned idx, const LogicalUser& previous)
    //--------------------------------------------------------------------------
    {
      legion_assert(previous.shard_proj != nullptr);
      std::vector<PointwiseDependence>& dependences =
          pointwise_dependences[idx];
      for (PointwiseDependence& dependence : dependences)
        if (dependence.matches(previous))
          return;
      dependences.emplace_back(PointwiseDependence(previous));
      if (this->tracing)
      {
        legion_assert(this->trace != nullptr);
        this->trace->record_pointwise_dependence(
            previous.op, previous.gen, this, this->gen, idx,
            dependences.back());
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void PointwiseAnalyzable<OP>::replay_pointwise_dependences(
        std::map<unsigned, std::vector<PointwiseDependence> >& dependences)
    //--------------------------------------------------------------------------
    {
      legion_assert(pointwise_dependences.empty());
      pointwise_dependences.swap(dependences);
    }

    // Explicit instantiations
    template class PointwiseAnalyzable<CollectiveViewCreator<Operation> >;
    template class PointwiseAnalyzable<CopyOp>;
    template class PointwiseAnalyzable<FillOp>;
    template class PointwiseAnalyzable<CollectiveViewCreator<TaskOp> >;

  }  // namespace Internal
}  // namespace Legion
