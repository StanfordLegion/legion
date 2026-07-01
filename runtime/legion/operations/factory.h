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

#ifndef __LEGION_OPERATION_FACTORY_H__
#define __LEGION_OPERATION_FACTORY_H__

#include "legion/kernel/allocation.h"

namespace Legion {
  namespace Internal {

    /**
     * The operation factory class helps with the creation and
     * reuse of operations. It will
     */
    template<typename OP, typename WRAP = OP, bool CAN_DELETE = false>
    class OperationFactory {
    public:
      OperationFactory(void) { }
      OperationFactory(const OperationFactory& rhs) = delete;
      ~OperationFactory(void);
    public:
      OperationFactory& operator=(const OperationFactory& rhs) = delete;
    public:
      void create(OP*& op);
      void recycle(OP* op);
    private:
      std::vector<OP*> available;
    };

    /*
     * Specialization for when CAN_DELETE is true
     */
    template<typename OP, typename WRAP>
    class OperationFactory<OP, WRAP, true> {
    public:
      OperationFactory(void) { }
      OperationFactory(const OperationFactory& rhs) = delete;
      ~OperationFactory(void);
    public:
      OperationFactory& operator=(const OperationFactory& rhs) = delete;
    public:
      void create(OP*& op);
      void recycle(OP* op);
    private:
      std::deque<OP*> available;
    };

    /*
     * Specialization for when OP == WRAP
     */
    template<typename OP>
    class OperationFactory<OP, OP, false> {
    public:
      OperationFactory(void) { }
      OperationFactory(const OperationFactory& rhs) = delete;
      ~OperationFactory(void);
    public:
      OperationFactory& operator=(const OperationFactory& rhs) = delete;
    public:
      void create(OP*& op);
      void recycle(OP* op);
    private:
      std::vector<OP*> available;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_OPERATION_FACTORY_H__
