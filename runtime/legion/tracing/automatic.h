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

#ifndef __LEGION_AUTOMATIC_TRACING_H__
#define __LEGION_AUTOMATIC_TRACING_H__

#include "legion/tracing/recognizer.h"

namespace Legion {
  namespace Internal {

    /**
     * \class AutoTracing
     * The auto-tracing class provides an overload of the
     * add_to_dependence_queue method that will hook in the auto tracing
     * infrastructure and use the trace recognizer to see if we can find
     * traces that we will try to replay.
     */
    template<typename T>
    class AutoTracing : public T {
    public:
      template<typename... Args>
      AutoTracing(const Mapper::ContextConfigOutput& config, Args&&... args)
        : T(config, std::forward<Args>(args)...), recognizer(this, config),
          opidx(0)
      { }
    public:
      virtual bool add_to_dependence_queue(
          Operation* op,
          const std::vector<StaticDependence>* dependences = nullptr,
          bool unordered = false, bool outermost = true) override;
      // If the application performs a blocking operation, we need to know
      // about that, so override TaskContext::record_blocking_call().
      virtual void record_blocking_call(
          uint64_t future_coordinate, bool invalidate_trace = true) override;
      virtual void end_task(
          const void* res, size_t res_size, bool owned, PhysicalInstance inst,
          FutureFunctor* callback_functor,
          const Realm::ExternalInstanceResource* resource,
          void (*freefunc)(const Realm::ExternalInstanceResource&),
          const void* metadataptr, size_t metadatasize,
          ApEvent effects) override;
    private:
      TraceRecognizer recognizer;
      uint64_t opidx;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_AUTOMATIC_TRACING_H__
