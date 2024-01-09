/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_AUTO_TRACE_H__
#define __LEGION_AUTO_TRACE_H__

#include "legion.h"
#include "legion/legion_context.h"
#include "legion/legion_utilities.h"

namespace Legion {
  namespace Internal {

    // TODO (rohany): Can add hashing methods here for hashing? I don't think that
    //  I want to do this on the Operations themselves because i'm not going to has
    //  everything about each operation, but just fields that are necessary for tracing.
    class TraceHashHelper {
    public:
      TraceHashHelper();
      Murmur3Hasher::Hash hash(Operation* op);
      void hash(TaskOp* op);
      void hash(FillOp* op);
      void hash(FenceOp* op);
      void hash(CopyOp* op);
      // TODO (rohany): AllReduceOp, DiscardOp.
      void hash(const RegionRequirement& req);
      void hash(const LogicalRegion& region);
    private:
      Murmur3Hasher hasher;
    };

    template <typename T>
    class AutomaticTracingContext : public T {
    public:
      // TODO (rohany): I'm not sure of the C++-ism to declare this constructor
      //  here and then implement it somewhere else.
      template <typename ... Args>
      AutomaticTracingContext(Args&& ... args) : T(std::forward<Args>(args) ... ) {}
    public:
      bool add_to_dependence_queue(Operation *op,
                                   bool unordered = false,
                                   bool outermost = true) override;
    private:
      TraceHashHelper hasher;
    };

    // Utility functions.
    bool is_operation_traceable(Operation* op);


    // TODO (rohany): Can we move these declarations to another file?

    template <typename T>
    bool AutomaticTracingContext<T>::add_to_dependence_queue(Operation* op, bool unordered, bool outermost) {
      // TODO (rohany): unordered operations should always be forwarded to the underlying context and not buffered up.

      auto kind = op->get_operation_kind();
      auto kind_str = Operation::get_string_rep(kind);
      auto prov = op->get_provenance();
      auto memo = op->get_memoizable();
      if (prov) {
        std::cout << "Op: " << std::string(kind_str) << " at " << std::string(prov->human_str()) << " " << (memo != nullptr) << std::endl;
      } else {
        std::cout << "Op: " << std::string(kind_str) << " " << (memo != nullptr) << std::endl;
      }

      if (is_operation_traceable(op)) {
        auto hash = TraceHashHelper{}.hash(op);
        std::cout << hash.x << " " << hash.y << std::endl;
      }

      return T::add_to_dependence_queue(op, unordered, outermost);
    }


  };
};

#endif // __LEGION_AUTO_TRACE_H__
