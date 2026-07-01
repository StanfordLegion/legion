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

#ifndef __LEGION_EXCEPTION_H__
#define __LEGION_EXCEPTION_H__

#include "legion/api/types.h"

namespace Legion {

  enum ExceptionType {
    LEGION_APPLICATION_EXCEPTION,
    LEGION_INTERFACE_EXCEPTION,
    LEGION_DYNAMIC_TYPE_EXCEPTION,
    LEGION_PROGRAMMING_MODEL_EXCEPTION,
    LEGION_MAPPER_EXCEPTION,
    LEGION_RESOURCE_EXCEPTION,
    LEGION_STARTUP_EXCEPTION,
    LEGION_FATAL_EXCEPTION,
    LEGION_WARNING_EXCEPTION,
  };

  /**
   * \class Exception
   * This class allows for users to package up any kind of data and
   * for raising an exception. It implements the std::streambuf
   * interface so that you can make input/output streams for accessing
   * it. You can also get access to the underlying buffer of data for
   * interpreting the data directly in case you serialized up data directly
   * (useful if you've done something like pickled a python exception).
   * DO NOT THROW THIS LIKE A C++ EXCEPTION OR YOUR PROGRAM WILL CRASH.
   */
  class Exception : public std::streambuf {
  public:
    Exception(ExceptionType type);
    Exception(const Exception& rhs) = delete;
    Exception(Exception&& rhs);
    ~Exception(void);
  public:
    Exception& operator=(const Exception& rhs) = delete;
    Exception& operator=(Exception&& rhs) = delete;
  public:
    void clear(void);  // Will invalidate any streams
    void record_backtrace(const Realm::Backtrace& backtrace);
  public:
    size_t size(void) const;
    const char* data(void) const;
    // Can convert implicitly to a std::stringview
    inline operator std::string_view(void) const
    {
      return std::string_view(data(), size());
    }
  protected:
    virtual int_type overflow(int_type c) override;
  public:
    const ExceptionType type;
  private:
    static constexpr size_t STACK_SIZE = 128;
    static_assert(sizeof(char) == sizeof(uint8_t));
    char stack_buffer[STACK_SIZE];
    char* heap_buffer;
    size_t heap_size;
  };

  /**
   * \class Error
   * This is a wrapper class for an exception that helps build an
   * error exception via a std::ostream.
   */
  class Error {
  public:
    Error(ExceptionType type = LEGION_APPLICATION_EXCEPTION);
    Error(const Error& rhs) = delete;
    Error(Error&& rhs) = delete;
    ~Error(void);
  public:
    Error& operator=(const Error& rhs) = delete;
    Error& operator=(Error&& rhs) = delete;
  public:
    template<typename T>
    inline Error& operator<<(T&& value)
    {
      stream << std::forward<T>(value);
      return *this;
    }
    [[noreturn]] void raise(void);
  public:
    Exception exception;
    std::ostream stream;
  private:
    bool raised;
  };

  /**
   * \class Fatal
   * This is a wrapper class for an exception that helps build a
   * fatal exception via a std::ostream
   */
  class Fatal {
  public:
    Fatal(void);
    Fatal(const Fatal& rhs) = delete;
    Fatal(Fatal&& rhs) = delete;
    ~Fatal(void);
  public:
    Fatal& operator=(const Fatal& rhs) = delete;
    Fatal& operator=(Fatal&& rhs) = delete;
  public:
    template<typename T>
    inline Fatal& operator<<(T&& value)
    {
      stream << std::forward<T>(value);
      return *this;
    }
    [[noreturn]] void raise(void);
  public:
    Exception exception;
    std::ostream stream;
  private:
    bool raised;
  };

  /**
   * \class Warning
   * This is a wrapper class for an exception that helps build a
   * warning exception via a std::ostream
   */
  class Warning {
  public:
    Warning(void);
    Warning(const Warning& rhs) = delete;
    Warning(Warning&& rhs) = delete;
    ~Warning(void);
  public:
    Warning& operator=(const Warning& rhs) = delete;
    Warning& operator=(Warning&& rhs) = delete;
  public:
    template<typename T>
    inline Warning& operator<<(T&& value)
    {
      stream << std::forward<T>(value);
      return *this;
    }
    void raise(void);
  public:
    Exception exception;
    std::ostream stream;
    bool active;
  };

  /**
   * \class ExceptionHandler
   * Both applications and mappers can register exception handlers
   * with Legion to aid in modifying or rewriting error messages.
   * In the case of exceptions that are not warnings the exception
   * will always be raised and can only be handled by
   */
  class ExceptionHandler {
  public:
    virtual ~ExceptionHandler(void) { }
    // Whether this exception handler can handle exceptions or
    // is only capable of handling warnings
    virtual bool can_handle(ExceptionType type) const { return false; }
    // Handle the kind of the exception and return if the exception
    // was successfaully handled
    virtual bool handle_exception(
        Exception& exception, const std::string_view& provenance,
        const Realm::Backtrace& backtrace)
    {
      return false;
    }
  };

}  // namespace Legion

#endif  // __LEGION_EXCEPTION_H__
