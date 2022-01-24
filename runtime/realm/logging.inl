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

// inlined methods for Realm logging

// nop, but helps IDEs
#include "realm/logging.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// static checks in here require these to be #define's instead of the enums
//  defined in logging.h - we'll push/pop these to avoid polluting
//  other namespaces
#pragma push_macro("LEVEL_SPEW")
#pragma push_macro("LEVEL_DEBUG")
#pragma push_macro("LEVEL_INFO")
#pragma push_macro("LEVEL_PRINT")
#pragma push_macro("LEVEL_WARNING")
#pragma push_macro("LEVEL_ERROR")
#pragma push_macro("LEVEL_FATAL")
#pragma push_macro("LEVEL_NONE")
// in case they were already defined...
#undef LEVEL_SPEW
#undef LEVEL_DEBUG
#undef LEVEL_INFO
#undef LEVEL_PRINT
#undef LEVEL_WARNING
#undef LEVEL_ERROR
#undef LEVEL_FATAL
#undef LEVEL_NONE
#define LEVEL_SPEW 0
#define LEVEL_DEBUG 1
#define LEVEL_INFO 2
#define LEVEL_PRINT 3
#define LEVEL_WARNING 4
#define LEVEL_ERROR 5
#define LEVEL_FATAL 6
#define LEVEL_NONE 7

#if REALM_LOGGING_MIN_LEVEL <= LEVEL_SPEW
#define REALM_LOGGING_DO_SPEW
#endif
#if REALM_LOGGING_MIN_LEVEL <= LEVEL_DEBUG
#define REALM_LOGGING_DO_DEBUG
#endif
#if REALM_LOGGING_MIN_LEVEL <= LEVEL_INFO
#define REALM_LOGGING_DO_INFO
#endif
#if REALM_LOGGING_MIN_LEVEL <= LEVEL_PRINT
#define REALM_LOGGING_DO_PRINT
#endif
#if REALM_LOGGING_MIN_LEVEL <= LEVEL_WARNING
#define REALM_LOGGING_DO_WARNING
#endif
#if REALM_LOGGING_MIN_LEVEL <= LEVEL_ERROR
#define REALM_LOGGING_DO_ERROR
#endif
#if REALM_LOGGING_MIN_LEVEL <= LEVEL_FATAL
#define REALM_LOGGING_DO_FATAL
#endif
#pragma pop_macro("LEVEL_SPEW")
#pragma pop_macro("LEVEL_DEBUG")
#pragma pop_macro("LEVEL_INFO")
#pragma pop_macro("LEVEL_PRINT")
#pragma pop_macro("LEVEL_WARNING")
#pragma pop_macro("LEVEL_ERROR")
#pragma pop_macro("LEVEL_FATAL")
#pragma pop_macro("LEVEL_NONE")

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

  inline bool Logger::want_spew(void) const
  {
#ifndef REALM_LOGGING_DO_SPEW
    return false;
#else
    return (log_level <= LEVEL_SPEW);
#endif
  }

  inline bool Logger::want_debug(void) const
  {
#ifndef REALM_LOGGING_DO_DEBUG
    return false;
#else
    return (log_level <= LEVEL_DEBUG);
#endif
  }

  inline bool Logger::want_info(void) const
  {
#ifndef REALM_LOGGING_DO_INFO
    return false;
#else
    return (log_level <= LEVEL_INFO);
#endif
  }

  inline bool Logger::want_print(void) const
  {
#ifndef REALM_LOGGING_DO_PRINT
    return false;
#else
    return (log_level <= LEVEL_PRINT);
#endif
  }

  inline bool Logger::want_warning(void) const
  {
#ifndef REALM_LOGGING_DO_WARNING
    return false;
#else
    return (log_level <= LEVEL_WARNING);
#endif
  }

  inline bool Logger::want_error(void) const
  {
#ifndef REALM_LOGGING_DO_ERROR
    return false;
#else
    return (log_level <= LEVEL_ERROR);
#endif
  }

  inline bool Logger::want_fatal(void) const
  {
#ifndef REALM_LOGGING_DO_FATAL
    return false;
#else
    return (log_level <= LEVEL_FATAL);
#endif
  }

  inline LoggerMessage Logger::spew(void)
  {
#ifndef REALM_LOGGING_DO_SPEW           // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_SPEW)          // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(this, true, LEVEL_SPEW);
#endif
  }
  
  inline LoggerMessage Logger::spew(LoggerMessageID messageID)
  {
#ifndef REALM_LOGGING_DO_SPEW            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_SPEW)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(messageID, this, true, LEVEL_SPEW);
#endif
  }
  
  inline LoggerMessage Logger::debug(void)
  {
#ifndef REALM_LOGGING_DO_DEBUG            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_DEBUG)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(this, true, LEVEL_DEBUG);
#endif
  }
  
  inline LoggerMessage Logger::debug(LoggerMessageID messageID)
  {
#ifndef REALM_LOGGING_DO_DEBUG            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_DEBUG)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(messageID, this, true, LEVEL_DEBUG);
#endif
  }
  
  inline LoggerMessage Logger::info(void)
  {
#ifndef REALM_LOGGING_DO_INFO            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_INFO)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(this, true, LEVEL_INFO);
#endif
  }
  
  inline LoggerMessage Logger::info(LoggerMessageID messageID)
  {
#ifndef REALM_LOGGING_DO_INFO             // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_INFO)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(messageID, this, true, LEVEL_INFO);
#endif
  }
  
  inline LoggerMessage Logger::print(void)
  {
#ifndef REALM_LOGGING_DO_PRINT           // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_PRINT)          // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(this, true, LEVEL_PRINT);
#endif
  }
  
  inline LoggerMessage Logger::print(LoggerMessageID messageID)
  {
#ifndef REALM_LOGGING_DO_PRINT            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_PRINT)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(messageID, this, true, LEVEL_PRINT);
#endif
  }
  
  inline LoggerMessage Logger::warning(void)
  {
#ifndef REALM_LOGGING_DO_WARNING            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_WARNING)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(this, true, LEVEL_WARNING);
#endif
  }
  
  inline LoggerMessage Logger::warning(LoggerMessageID messageID)
  {
#ifndef REALM_LOGGING_DO_WARNING            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_WARNING)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(messageID, this, true, LEVEL_WARNING);
#endif
  }
  
  inline LoggerMessage Logger::error(void)
  {
#ifndef REALM_LOGGING_DO_ERROR            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_ERROR)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(this, true, LEVEL_ERROR);
#endif
  }
  
  inline LoggerMessage Logger::error(LoggerMessageID messageID)
  {
#ifndef REALM_LOGGING_DO_ERROR            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_ERROR)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(messageID, this, true, LEVEL_ERROR);
#endif
  }
  
  inline LoggerMessage Logger::fatal(void)
  {
#ifndef REALM_LOGGING_DO_FATAL            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_FATAL)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(this, true, LEVEL_FATAL);
#endif
  }
  
  inline LoggerMessage Logger::fatal(LoggerMessageID messageID)
  {
#ifndef REALM_LOGGING_DO_FATAL            // static early out
    return LoggerMessage();
#else
    if(log_level > LEVEL_FATAL)           // dynamic early out
      return LoggerMessage();
    
    return LoggerMessage(messageID, this, true, LEVEL_FATAL);
#endif
  }
  
  // use this only if you want a dynamic level for some reason
  inline LoggerMessage Logger::newmsg(LoggingLevel level)
  {
#ifndef REALM_LOGGING_DO_FATAL            // static early out (fatal is highest)
    return LoggerMessage();
#else
    if((REALM_LOGGING_MIN_LEVEL > level) ||
       (log_level > level))
    return LoggerMessage();
    
    return LoggerMessage(this, true, level);
#endif
  }
  
  
  // append a URL to the format string that links to the online message documentation
  inline const char *formatLink(const char *type, LoggerMessageID messageID,
				char *buffer, size_t maxlen)
  {
      const char *legionURL = "http://legion.stanford.edu/messages";
      snprintf(buffer, maxlen,
	       "\nFor more information see:\n%s/%s_code.html#%s_code_%d\n",
	      legionURL, type, type, int(messageID));
      return buffer;
  }
  
  // old printf-style interface
  inline void Logger::spew(const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_SPEW              // static early out
    if(log_level > LEVEL_SPEW)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    spew().vprintf(fmt, args);
    va_end(args);
#endif
  }
    
    static const int LENGTH_LONGEST_FORMAT_STRING = 8 * 1024;
  
  inline void Logger::spew(LoggerMessageID messageID, const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_SPEW              // static early out
    if(log_level > LEVEL_SPEW)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    char buffer[LENGTH_LONGEST_FORMAT_STRING];
    formatLink("spew", messageID, buffer, LENGTH_LONGEST_FORMAT_STRING);
    spew().vprintf("spew", messageID, fmt, args) << buffer;
    va_end(args);
#endif
  }
  
  inline void Logger::debug(const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_DEBUG              // static early out
    if(log_level > LEVEL_DEBUG)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    debug().vprintf(fmt, args);
    va_end(args);
#endif
  }
  
  inline void Logger::debug(LoggerMessageID messageID, const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_DEBUG              // static early out
    if(log_level > LEVEL_DEBUG)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    char buffer[LENGTH_LONGEST_FORMAT_STRING];
    formatLink("debug", messageID, buffer, LENGTH_LONGEST_FORMAT_STRING);
    debug().vprintf("debug", messageID, fmt, args) << buffer;
    va_end(args);
#endif
  }
  
  inline void Logger::info(const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_INFO              // static early out
    if(log_level > LEVEL_INFO)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    info().vprintf(fmt, args);
    va_end(args);
#endif
  }
  
  inline void Logger::info(LoggerMessageID messageID, const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_INFO              // static early out
    if(log_level > LEVEL_INFO)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    char buffer[LENGTH_LONGEST_FORMAT_STRING];
    formatLink("info", messageID, buffer, LENGTH_LONGEST_FORMAT_STRING);
    info().vprintf("info", messageID, fmt, args) << buffer;
    va_end(args);
#endif
  }
  
  inline void Logger::print(const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_PRINT              // static early out
    if(log_level > LEVEL_PRINT)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    print().vprintf(fmt, args);
    va_end(args);
#endif
  }
  
  inline void Logger::print(LoggerMessageID messageID, const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_PRINT              // static early out
    if(log_level > LEVEL_PRINT)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    char buffer[LENGTH_LONGEST_FORMAT_STRING];
    formatLink("print", messageID, buffer, LENGTH_LONGEST_FORMAT_STRING);
    print().vprintf("print", messageID, fmt, args) << buffer;
    va_end(args);
#endif
  }
  
  inline void Logger::warning(const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_WARNING              // static early out
    if(log_level > LEVEL_WARNING)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    warning().vprintf(fmt, args);
    va_end(args);
#endif
  }
  
  inline void Logger::warning(LoggerMessageID messageID, const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_WARNING              // static early out
    if(log_level > LEVEL_WARNING)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    char buffer[LENGTH_LONGEST_FORMAT_STRING];
    formatLink("warning", messageID, buffer, LENGTH_LONGEST_FORMAT_STRING);
    warning().vprintf("warning", messageID, fmt, args) << buffer;
    va_end(args);
#endif
  }
  
  inline void Logger::error(const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_ERROR              // static early out
    if(log_level > LEVEL_ERROR)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    error().vprintf(fmt, args);
    va_end(args);
#endif
  }
  
  inline void Logger::error(LoggerMessageID messageID, const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_ERROR              // static early out
    if(log_level > LEVEL_ERROR)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    char buffer[LENGTH_LONGEST_FORMAT_STRING];
    formatLink("error", messageID, buffer, LENGTH_LONGEST_FORMAT_STRING);
    error().vprintf("error", messageID, fmt, args) << buffer;
    va_end(args);
#endif
  }
  
  inline void Logger::fatal(const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_FATAL              // static early out
    if(log_level > LEVEL_FATAL)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    fatal().vprintf(fmt, args);
    va_end(args);
#endif
  }
  
  inline void Logger::fatal(LoggerMessageID messageID, const char *fmt, ...)
  {
#ifdef REALM_LOGGING_DO_FATAL              // static early out
    if(log_level > LEVEL_FATAL)            // dynamic early out
      return;
    
    va_list args;
    va_start(args, fmt);
    char buffer[LENGTH_LONGEST_FORMAT_STRING];
    formatLink("fatal", messageID, buffer, LENGTH_LONGEST_FORMAT_STRING);
    fatal().vprintf("fatal", messageID, fmt, args) << buffer;
    va_end(args);
#endif
  }
  
  ////////////////////////////////////////////////////////////////////////
  //
  // class LoggerMessage
  
  // default constructor makes an inactive message
  inline LoggerMessage::LoggerMessage(void)
    : messageID(RESERVED_LOGGER_MESSAGE_ID), logger(0), active(false), level(Logger::LEVEL_NONE)
  {}
  
  inline LoggerMessage::LoggerMessage(Logger *_logger, bool _active, Logger::LoggingLevel _level)
    : messageID(RESERVED_LOGGER_MESSAGE_ID), logger(_logger), active(_active), level(_level)
  {
    if(active)
      stream.construct(buffer.construct());
  }
  
  inline LoggerMessage::LoggerMessage(LoggerMessageID messageID, Logger *_logger, bool _active, Logger::LoggingLevel _level)
    : messageID(messageID), logger(_logger), active(_active), level(_level)
  {
    if(active)
      stream.construct(buffer.construct());
  }
  
  inline LoggerMessage::LoggerMessage(const LoggerMessage& to_copy)
    : messageID(to_copy.messageID), logger(to_copy.logger), active(to_copy.active), level(to_copy.level)
  {
    if(active)
      stream.construct(buffer.construct());
  }
  
  inline LoggerMessage::~LoggerMessage(void)
  {
    if(active) {
      logger->log_msg(level, buffer->data(), buffer->size());
      active = false;
    }
  }
  
  template <typename T>
  inline LoggerMessage& LoggerMessage::operator<<(const T& val)
  {
    // send through to normal ostringstream formatting routines if active
    if(active)
      get_stream() << val;
    return *this;
  }
  
  inline bool LoggerMessage::is_active(void) const
  {
    return active;
  }
  
  inline std::ostream& LoggerMessage::get_stream(void)
  {
#ifdef DEBUG_REALM
    assert(active);
#endif
    return *stream;
  }
  

}; // namespace Realm
