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

#include "legion/api/exception.h"
#include "legion/kernel/runtime.h"
#include "legion/contexts/inner.h"
#include "legion/operations/operation.h"
#include "legion/utilities/provenance.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Exception
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  Exception::Exception(ExceptionType t)
    : type(t), heap_buffer(nullptr), heap_size(0)
  //--------------------------------------------------------------------------
  {
    clear();
    std::ostream stream(this);
    switch (type)
    {
      case LEGION_APPLICATION_EXCEPTION:
        {
          stream << "LEGION APPLICATION EXCEPTION: ";
          break;
        }
      case LEGION_INTERFACE_EXCEPTION:
        {
          stream << "LEGION API USAGE EXCEPTION: ";
          break;
        }
      case LEGION_DYNAMIC_TYPE_EXCEPTION:
        {
          stream << "LEGION DYNAMIC TYPE EXCEPTION: ";
          break;
        }
      case LEGION_PROGRAMMING_MODEL_EXCEPTION:
        {
          stream << "LEGION PROGRAMMING MODEL EXCEPTION: ";
          break;
        }
      case LEGION_MAPPER_EXCEPTION:
        {
          stream << "LEGION MAPPER EXCEPTION: ";
          break;
        }
      case LEGION_RESOURCE_EXCEPTION:
        {
          stream << "LEGION RESOURCE EXCEPTION: ";
          break;
        }
      case LEGION_STARTUP_EXCEPTION:
        {
          stream << "LEGION STARTUP EXCEPTION: ";
          break;
        }
      case LEGION_FATAL_EXCEPTION:
        {
          stream << "LEGION INTERNAL FATAL ERROR: ";
          break;
        }
      case LEGION_WARNING_EXCEPTION:
        {
          stream << "LEGION WARNING: ";
          break;
        }
      default:
        std::abort();
    }
  }

  //--------------------------------------------------------------------------
  Exception::Exception(Exception&& rhs)
    : type(rhs.type), heap_buffer(rhs.heap_buffer), heap_size(rhs.heap_size)
  //--------------------------------------------------------------------------
  {
    if (heap_buffer != nullptr)
    {
      const size_t size = rhs.pptr() - heap_buffer;
      setp(heap_buffer + size, heap_buffer + heap_size);
      setg(heap_buffer, heap_buffer, heap_buffer + heap_size);
      rhs.heap_buffer = nullptr;
      rhs.heap_size = 0;
      rhs.clear();
    }
    else
    {
      std::memcpy(stack_buffer, rhs.stack_buffer, STACK_SIZE);
      const size_t size = rhs.pptr() - rhs.stack_buffer;
      setp(stack_buffer + size, stack_buffer + STACK_SIZE);
      setg(stack_buffer, stack_buffer, stack_buffer + STACK_SIZE);
      rhs.clear();
    }
  }

  //--------------------------------------------------------------------------
  Exception::~Exception(void)
  //--------------------------------------------------------------------------
  {
    if (heap_buffer != nullptr)
      std::free(heap_buffer);
  }

  //--------------------------------------------------------------------------
  void Exception::clear(void)
  //--------------------------------------------------------------------------
  {
    if (heap_buffer != nullptr)
    {
      setp(heap_buffer, heap_buffer + heap_size);
      setg(heap_buffer, heap_buffer, heap_buffer + heap_size);
    }
    else
    {
      setp(stack_buffer, stack_buffer + STACK_SIZE);
      setg(stack_buffer, stack_buffer, stack_buffer + STACK_SIZE);
    }
  }

  //--------------------------------------------------------------------------
  void Exception::record_backtrace(const Realm::Backtrace& backtrace)
  //--------------------------------------------------------------------------
  {
    std::ostream stream(this);
    stream << "\n-----------------------------------\n";
    stream << "Backtrace:\n";
    stream << backtrace;
  }

  //--------------------------------------------------------------------------
  size_t Exception::size(void) const
  //--------------------------------------------------------------------------
  {
    return (pptr() - data());
  }

  //--------------------------------------------------------------------------
  const char* Exception::data(void) const
  //--------------------------------------------------------------------------
  {
    if (heap_buffer != nullptr)
      return heap_buffer;
    else
      return stack_buffer;
  }

  //--------------------------------------------------------------------------
  Exception::int_type Exception::overflow(int_type c)
  //--------------------------------------------------------------------------
  {
    size_t curlen = size();
    size_t offset = (gptr() - data());
    if (heap_buffer != nullptr)
    {
      // Double the size of the heap buffer
      legion_assert(curlen == heap_size);
      heap_size *= 2;
      heap_buffer = (char*)std::realloc(heap_buffer, heap_size);
    }
    else
    {
      // Switch to heap buffer
      heap_size = STACK_SIZE * 2;
      heap_buffer = (char*)std::malloc(heap_size);
      legion_assert(heap_buffer != nullptr);
      std::memcpy(heap_buffer, stack_buffer, STACK_SIZE);
    }
    if (c >= 0)
      heap_buffer[curlen++] = c;
    setp(heap_buffer + curlen, heap_buffer + heap_size);
    setg(heap_buffer, heap_buffer + offset, heap_buffer + heap_size);
    return 0;
  }

  //--------------------------------------------------------------------------
  Error::Error(ExceptionType t)
    : exception(t), stream(&exception), raised(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Error::~Error(void)
  //--------------------------------------------------------------------------
  {
    if (!raised)
      std::abort();
  }

  //--------------------------------------------------------------------------
  [[noreturn]] void Error::raise(void)
  //--------------------------------------------------------------------------
  {
    legion_assert(!raised);
    raised = true;
    Realm::Backtrace backtrace;
    backtrace.capture_backtrace(2 /*to skip*/);
    Internal::Runtime::raise_exception(std::move(exception), backtrace);
  }

  //--------------------------------------------------------------------------
  Fatal::Fatal(void)
    : exception(LEGION_FATAL_EXCEPTION), stream(&exception), raised(false)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Fatal::~Fatal(void)
  //--------------------------------------------------------------------------
  {
    if (!raised)
      std::abort();
  }

  //--------------------------------------------------------------------------
  [[noreturn]] void Fatal::raise(void)
  //--------------------------------------------------------------------------
  {
    legion_assert(!raised);
    raised = true;
    Realm::Backtrace backtrace;
    backtrace.capture_backtrace(2 /*to skip*/);
    Internal::Runtime::raise_exception(std::move(exception), backtrace);
  }

  //--------------------------------------------------------------------------
  Warning::Warning(void)
    : exception(LEGION_WARNING_EXCEPTION), stream(&exception), active(true)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Warning::~Warning(void)
  //--------------------------------------------------------------------------
  {
    if (active)
      std::abort();
  }

  //--------------------------------------------------------------------------
  void Warning::raise(void)
  //--------------------------------------------------------------------------
  {
    if (active)
    {
      active = false;
      Realm::Backtrace backtrace;
      if ((Internal::runtime != nullptr) &&
          Internal::runtime->warnings_backtrace)
        backtrace.capture_backtrace(2 /*to skip*/);
      Internal::Runtime::raise_warning(std::move(exception), backtrace);
    }
  }

}  // namespace Legion
