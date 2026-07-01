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

#ifndef __LEGION_ARGUMENT_MAP_IMPL_H__
#define __LEGION_ARGUMENT_MAP_IMPL_H__

#include "legion/api/argument_map.h"
#include "legion/kernel/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ArgumentMapImpl
     * An argument map implementation that provides
     * the backing store for an argument map handle.
     * Argument maps maintain pairs of domain points
     * and task arguments.  To make re-use of argument
     * maps efficient with small deltas, argument map
     * implementations provide a nice versionining system
     * with all argument map implementations sharing
     * a single backing store to de-duplicate domain
     * points and values.
     */
    class ArgumentMapImpl : public Collectable,
                            public Heapify<ArgumentMapImpl, SHORT_LIFETIME> {
    public:
      ArgumentMapImpl(void);
      ArgumentMapImpl(const FutureMap& rhs);
      ArgumentMapImpl(const ArgumentMapImpl& impl) = delete;
      ~ArgumentMapImpl(void);
    public:
      ArgumentMapImpl& operator=(const ArgumentMapImpl& rhs) = delete;
    public:
      bool has_point(const DomainPoint& point);
      void set_point(
          const DomainPoint& point, const UntypedBuffer& arg, bool replace);
      void set_point(const DomainPoint& point, const Future& f, bool replace);
      bool remove_point(const DomainPoint& point);
      UntypedBuffer get_point(const DomainPoint& point);
    public:
      FutureMap freeze(InnerContext* ctx, Provenance* provenance);
      void unfreeze(void);
    private:
      FutureMap future_map;
      std::map<DomainPoint, Future> arguments;
      std::set<RtEvent> point_set_deletion_preconditions;
      IndexSpaceNode* point_set;
      unsigned dimensionality;
      unsigned dependent_futures;  // number of futures with producer ops
      bool update_point_set;
      bool equivalent;  // argument and future_map the same
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_ARGUMENT_MAP_IMPL_H__
