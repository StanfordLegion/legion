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

#ifndef __LEGION_POINTWISE_H__
#define __LEGION_POINTWISE_H__

#include "legion/operations/operation.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct PointwiseDependence
     * This data structure help record the necessary information for
     * capturing and reporting pointwise dependences
     */
    struct PointwiseDependence {
    public:
      PointwiseDependence(void);
      PointwiseDependence(const LogicalUser& previous);
      PointwiseDependence(const PointwiseDependence& rhs);
      PointwiseDependence(PointwiseDependence&& rhs) noexcept;
      ~PointwiseDependence(void);
    public:
      PointwiseDependence& operator=(const PointwiseDependence& rhs);
      PointwiseDependence& operator=(PointwiseDependence&& rhs) noexcept;
    public:
      bool matches(const LogicalUser& user) const;
      void find_dependences(
          const RegionRequirement& req,
          const std::vector<LogicalRegion>& point_regions,
          std::map<LogicalRegion, std::vector<DomainPoint> >& dependences)
          const;
    public:
      void serialize(Serializer& rez) const;
      void deserialize(Deserializer& derez);
    public:
      // Previous operation context index
      uint64_t context_index;
      // Previous operation unique ID
      UniqueID unique_id;
      // Previous operation kind
      OpKind kind;
      // Previous operation region index
      unsigned region_index;
      // Projection information of previous point-wise operation
      IndexSpaceNode* domain;
      ProjectionFunction* projection;
      ShardingFunctor* sharding;
      ShardingID sharding_id;
      IndexSpaceNode* sharding_domain;
    };

    /**
     * \class PointwiseAnalyzable
     */
    template<typename OP>
    class PointwiseAnalyzable : public OP {
    public:
      template<typename... Args>
      PointwiseAnalyzable(Args&&... args) : OP(std::forward<Args>(args)...)
      { }
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual bool is_pointwise_analyzable(void) const override;
      virtual void register_pointwise_dependence(
          unsigned idx, const LogicalUser& previous) override;
      virtual void replay_pointwise_dependences(
          std::map<unsigned, std::vector<PointwiseDependence> >& dependences)
          override;
    protected:
      // Map from region requirement indexes to the point-wise dependences
      // we'll need to compute for that region requirement
      std::map<unsigned, std::vector<PointwiseDependence> >
          pointwise_dependences;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_POINTWISE_H__
