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

#include "legion/tracing/automatic.h"
#include "legion/contexts/replicate.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Auto Tracing
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    bool AutoTracing<T>::add_to_dependence_queue(
        Operation* op, const std::vector<StaticDependence>* dependences,
        bool unordered, bool outermost)
    //--------------------------------------------------------------------------
    {
      // If we're unordered or inside an explicit trace then pass through
      // Note that we might set outermost to false if we're being flushed
      // from the trace cache so set it back to true for the context
      if (unordered || (this->current_trace != nullptr) || !outermost ||
          this->task_executed)
        return T::add_to_dependence_queue(
            op, dependences, unordered, true /*outermost*/);
      else if (op->record_trace_hash(this->recognizer, this->opidx))
      {
        this->opidx++;
        return true;
      }
      else
      {
        // Increment the current trace blocking index so we know
        // when we need to flush operations under blocking calls
        this->current_trace_blocking_index = this->next_blocking_index;
        return T::add_to_dependence_queue(op, dependences);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void AutoTracing<T>::record_blocking_call(
        uint64_t blocking_index, bool invalidate_trace)
    //--------------------------------------------------------------------------
    {
      // Check to see if the blocking operation happens for any operation
      // that occurs inside of the range of operations that we are buffering
      if ((blocking_index != InnerContext::NO_BLOCKING_INDEX) &&
          (this->current_trace == nullptr) &&
          (this->current_trace_blocking_index <= blocking_index))
      {
        // Handling waits from the application is very similar
        // to the case in add_to_dependence_queue when we encounter an
        // operation that is not traceable. We interrupt traces in
        // the identifier, and flush the watcher and replayer. We identify
        // whether a wait is coming from the application by seeing if the
        // future being waited on has a valid coordinate.
        this->recognizer.record_operation_untraceable(nullptr, this->opidx);
        this->current_trace_blocking_index = this->next_blocking_index;
      }
      // Need to also do whatever the base context was going to do.
      T::record_blocking_call(blocking_index, invalidate_trace);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void AutoTracing<T>::end_task(
        const void* res, size_t res_size, bool owned,
        PhysicalInstance deferred_result_instance,
        FutureFunctor* callback_functor,
        const Realm::ExternalInstanceResource* resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&),
        const void* metadataptr, size_t metadatasize, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Flush any buffered operations
      this->recognizer.record_operation_untraceable(nullptr, opidx++);
      T::end_task(
          res, res_size, owned, deferred_result_instance, callback_functor,
          resource, freefunc, metadataptr, metadatasize, effects);
    }

    template class AutoTracing<InnerContext>;
    template class AutoTracing<ReplicateContext>;

  }  // namespace Internal
}  // namespace Legion
