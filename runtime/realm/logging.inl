/* Copyright 2016 Stanford University, NVIDIA Corporation
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

// inlined methods for Realm logging

// nop, but helps IDEs
#include "logging.h"

#include <assert.h>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Logger

  inline const std::string& Logger::get_name(void) const
  {
    return name;
  }

  inline Logger::LoggingLevel Logger::get_level(void) const
  {
    return log_level;
  }

  inline LoggerMessage Logger::spew(void)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_SPEW) ||  // static early out
       (log_level > LEVEL_SPEW))                  // dynamic early out
      return LoggerMessage();
   
    return LoggerMessage(this, true, LEVEL_SPEW);
  }

  inline LoggerMessage Logger::debug(void)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_DEBUG) ||  // static early out
       (log_level > LEVEL_DEBUG))                  // dynamic early out
      return LoggerMessage();
   
    return LoggerMessage(this, true, LEVEL_DEBUG);
  }

  inline LoggerMessage Logger::info(void)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_INFO) ||  // static early out
       (log_level > LEVEL_INFO))                  // dynamic early out
      return LoggerMessage();
   
    return LoggerMessage(this, true, LEVEL_INFO);
  }

  inline LoggerMessage Logger::print(void)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_PRINT) ||  // static early out
       (log_level > LEVEL_PRINT))                  // dynamic early out
      return LoggerMessage();
   
    return LoggerMessage(this, true, LEVEL_PRINT);
  }

  inline LoggerMessage Logger::warning(void)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_WARNING) ||  // static early out
       (log_level > LEVEL_WARNING))                  // dynamic early out
      return LoggerMessage();
   
    return LoggerMessage(this, true, LEVEL_WARNING);
  }

  inline LoggerMessage Logger::error(void)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_ERROR) ||  // static early out
       (log_level > LEVEL_ERROR))                  // dynamic early out
      return LoggerMessage();
   
    return LoggerMessage(this, true, LEVEL_ERROR);
  }

  inline LoggerMessage Logger::fatal(void)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_FATAL) ||  // static early out
       (log_level > LEVEL_FATAL))                  // dynamic early out
      return LoggerMessage();
   
    return LoggerMessage(this, true, LEVEL_FATAL);
  }

  // use this only if you want a dynamic level for some reason
  inline LoggerMessage Logger::newmsg(LoggingLevel level)
  {
    if((REALM_LOGGING_MIN_LEVEL > level) ||
       (log_level > level))
      return LoggerMessage();
   
    return LoggerMessage(this, true, level);
  }

  // old printf-style interface
  inline void Logger::spew(const char *fmt, ...)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_SPEW) ||  // static early out
       (log_level > LEVEL_SPEW))                  // dynamic early out
      return;
   
    va_list args;
    va_start(args, fmt);
    spew().vprintf(fmt, args);
    va_end(args);
  }

  inline void Logger::debug(const char *fmt, ...)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_DEBUG) ||  // static early out
       (log_level > LEVEL_DEBUG))                  // dynamic early out
      return;
   
    va_list args;
    va_start(args, fmt);
    debug().vprintf(fmt, args);
    va_end(args);
  }

  inline void Logger::info(const char *fmt, ...)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_INFO) ||  // static early out
       (log_level > LEVEL_INFO))                  // dynamic early out
      return;
   
    va_list args;
    va_start(args, fmt);
    info().vprintf(fmt, args);
    va_end(args);
  }

  inline void Logger::print(const char *fmt, ...)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_PRINT) ||  // static early out
       (log_level > LEVEL_PRINT))                  // dynamic early out
      return;
   
    va_list args;
    va_start(args, fmt);
    print().vprintf(fmt, args);
    va_end(args);
  }

  inline void Logger::warning(const char *fmt, ...)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_WARNING) ||  // static early out
       (log_level > LEVEL_WARNING))                  // dynamic early out
      return;
   
    va_list args;
    va_start(args, fmt);
    warning().vprintf(fmt, args);
    va_end(args);
  }

  inline void Logger::error(const char *fmt, ...)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_ERROR) ||  // static early out
       (log_level > LEVEL_ERROR))                  // dynamic early out
      return;
   
    va_list args;
    va_start(args, fmt);
    error().vprintf(fmt, args);
    va_end(args);
  }

  inline void Logger::fatal(const char *fmt, ...)
  {
    if((REALM_LOGGING_MIN_LEVEL > LEVEL_FATAL) ||  // static early out
       (log_level > LEVEL_FATAL))                  // dynamic early out
      return;
   
    va_list args;
    va_start(args, fmt);
    fatal().vprintf(fmt, args);
    va_end(args);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LoggerMessage

  // default constructor makes an inactive message
  inline LoggerMessage::LoggerMessage(void)
    : logger(0), active(false), level(Logger::LEVEL_NONE), oss(0)
  {}

  inline LoggerMessage::LoggerMessage(Logger *_logger, bool _active, Logger::LoggingLevel _level)
    : logger(_logger), active(_active), level(_level), oss(0)
  {
    if(active)
      oss = new std::ostringstream;
  }

  inline LoggerMessage::LoggerMessage(const LoggerMessage& to_copy)
    : logger(to_copy.logger), active(to_copy.active), level(to_copy.level), oss(0)
  {
    if(active)
      oss = new std::ostringstream;
  }

  inline LoggerMessage::~LoggerMessage(void)
  {
    if(active) {
      logger->log_msg(level, oss->str());
      delete oss;
    }
  }
      
  template <typename T>
  inline LoggerMessage& LoggerMessage::operator<<(const T& val)
  {
    // send through to normal ostringstream formatting routines if active
    if(active)
      (*oss) << val;
    return *this;
  }

  inline bool LoggerMessage::is_active(void) const
  {
    return active;
  }

  inline std::ostream& LoggerMessage::get_stream(void)
  {
    assert(active);
    return (*oss);
  }

}; // namespace Realm
