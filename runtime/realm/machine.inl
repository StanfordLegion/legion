/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// nop, but helps IDEs
#include "realm/runtime.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine::ProcessorQuery

  inline bool Machine::ProcessorQuery::operator==(const Machine::ProcessorQuery& compare_to) const
  {
    return (impl == compare_to.impl);
  }

  inline bool Machine::ProcessorQuery::operator!=(const Machine::ProcessorQuery& compare_to) const
  {
    return (impl != compare_to.impl);
  }

  inline Machine::ProcessorQuery::iterator Machine::ProcessorQuery::begin(void) const
  {
    return iterator(*this, first());
  }

  inline Machine::ProcessorQuery::iterator Machine::ProcessorQuery::end(void) const
  {
    return iterator(*this, Processor::NO_PROC);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine::MemoryQuery

  inline bool Machine::MemoryQuery::operator==(const Machine::MemoryQuery& compare_to) const
  {
    return (impl == compare_to.impl);
  }

  inline bool Machine::MemoryQuery::operator!=(const Machine::MemoryQuery& compare_to) const
  {
    return (impl != compare_to.impl);
  }

  inline Machine::MemoryQuery::iterator Machine::MemoryQuery::begin(void) const
  {
    return iterator(*this, first());
  }

  inline Machine::MemoryQuery::iterator Machine::MemoryQuery::end(void) const
  {
    return iterator(*this, Memory::NO_MEMORY);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineQueryIterator<QT,RT>

  template <typename QT, typename RT>
  inline MachineQueryIterator<QT,RT>::MachineQueryIterator(const QT& _query, RT _result)
    : query(_query), result(_result)
  {}

  template <typename QT, typename RT>
  inline MachineQueryIterator<QT,RT>::MachineQueryIterator(const MachineQueryIterator<QT,RT>& copy_from)
    : query(copy_from.query), result(copy_from.result)
  {}

  template <typename QT, typename RT>
  inline MachineQueryIterator<QT,RT>::~MachineQueryIterator(void)
  {}
	
  template <typename QT, typename RT>
  inline MachineQueryIterator<QT,RT>& MachineQueryIterator<QT,RT>::operator=(const MachineQueryIterator<QT,RT>& copy_from)
  {
    query = copy_from.query;
    result = copy_from.result;
    return *this;
  }
	
  template <typename QT, typename RT>
  inline bool MachineQueryIterator<QT,RT>::operator==(const MachineQueryIterator<QT,RT>& compare_to) const
  {
    return (query == compare_to.query) && (result == compare_to.result);
  }

  template <typename QT, typename RT>
  inline bool MachineQueryIterator<QT,RT>::operator!=(const MachineQueryIterator<QT,RT>& compare_to) const
  {
    return (query != compare_to.query) || (result != compare_to.result);
  }
	
  template <typename QT, typename RT>
  inline RT MachineQueryIterator<QT,RT>::operator*(void)
  {
    return result;
  }

  template <typename QT, typename RT>
  inline const RT *MachineQueryIterator<QT,RT>::operator->(void)
  {
    return &result;
  }
	
  template <typename QT, typename RT>
  inline MachineQueryIterator<QT,RT>& MachineQueryIterator<QT,RT>::operator++(/*prefix*/)
  {
    result = query.next(result);
    return *this;
  }

  template <typename QT, typename RT>
  inline MachineQueryIterator<QT,RT> MachineQueryIterator<QT,RT>::operator++(int/*postfix*/)
  {
    MachineQueryIterator<QT,RT> orig(*this);
    result = query.next(result);
    return orig;
  }

  template <typename QT, typename RT>
  inline MachineQueryIterator<QT,RT>::operator bool(void) const
  {
    return result.exists();
  }


}; // namespace Realm

