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

// Realm logging infrastructure

#include "realm/logging.h"

#include "realm/network.h"

#include "realm/cmdline.h"
#include "realm/timers.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <set>
#include <map>

#ifdef REALM_ON_WINDOWS
#include <windows.h>
#include <processthreadsapi.h>
#endif

namespace Realm {

  class LoggerFileStream : public LoggerOutputStream {
  public:
    LoggerFileStream(FILE *_f, bool _close_file, bool _include_timestamp)
      : f(_f), close_file(_close_file), include_timestamp(_include_timestamp)
    {}

    virtual ~LoggerFileStream(void)
    {
      if(close_file)
	fclose(f);
    }

    virtual void log_msg(Logger::LoggingLevel level, const char *name, const char *msgdata, size_t msglen)
    {
      // build message string, including prefix
      static const int MAXLEN = 4096;
      char buffer[MAXLEN];
      int pfxlen;
      if(include_timestamp) {
	// special case - log messages before we've agreed on common time base
	//  across all nodes show as 0.0
	double now;
	if(Clock::get_zero_time() != 0)
	  now = Clock::current_time();
	else
	  now = 0;
	pfxlen = snprintf(buffer, MAXLEN - 2, "[%d - %lx] %11.6f {%d}{%s}: ",
			  Network::my_node_id,
#ifdef REALM_ON_WINDOWS
			  GetCurrentThreadId(),
#else
			  (unsigned long)pthread_self(),
#endif
			  now, level, name);
      } else
	pfxlen = snprintf(buffer, MAXLEN - 2, "[%d - %lx] {%d}{%s}: ",
			  Network::my_node_id,
#ifdef REALM_ON_WINDOWS
			  GetCurrentThreadId(),
#else
			  (unsigned long)pthread_self(),
#endif
			  level, name);

      // would simply concatenating this message overflow the buffer?
      if((pfxlen + msglen) >= MAXLEN)
      {
        // if this is an error or a warning, print out the
        // whole message no matter what
        if ((level == Logger::LEVEL_FATAL) ||
            (level == Logger::LEVEL_ERROR) || (level == Logger::LEVEL_WARNING))
        {
          const size_t full_len = pfxlen + msglen + 1;
          char *full_buffer = (char*)malloc(full_len);
          memcpy(full_buffer, buffer, pfxlen);
          memcpy(full_buffer + pfxlen, msgdata, msglen);
          full_buffer[pfxlen + msglen] = '\n';

          write(full_buffer, full_len);

          free(full_buffer);
          return;
        } else {
                // less critical messages are truncated
                msglen = MAXLEN - pfxlen - 1;
        }
      }
      memcpy(buffer + pfxlen, msgdata, msglen);
      buffer[pfxlen + msglen] = '\n';
      size_t total_len = pfxlen + msglen + 1;

      write(buffer, total_len);
    }

    virtual void flush()
    {
      AutoLock<> al(mutex);
      fflush(f);
    }

  protected:
    virtual void write(const char *buffer, size_t len)
    {
      AutoLock<> al(mutex);
#ifndef NDEBUG
      size_t amt =
#endif
      fwrite(buffer, 1, len, f);
      assert(amt == len);
    }

    FILE *f;
    bool close_file, include_timestamp;
    Mutex mutex;
  };

  class LoggerConfig {
  protected:
    LoggerConfig(void);
    ~LoggerConfig(void);

  public:
    static LoggerConfig *get_config(void);

    static void flush_all_streams(void);

    void read_command_line(std::vector<std::string>& cmdline);
    void set_default_output(LoggerOutputStream *s);
    void set_logger_output(const std::string& name, LoggerOutputStream *s);

    // either configures a logger right away or remembers it to config once
    //   we know the desired settings
    void configure(Logger *logger);

    void remove_logger(Logger *logger);

  protected:
    bool parse_level_argument(const std::string& s);

    bool cmdline_read;
    Logger::LoggingLevel default_level, stderr_level;
    bool include_timestamp;
    std::map<std::string, Logger::LoggingLevel> category_levels;
    std::string cats_enabled;
    std::set<Logger *> pending_configs;
    LoggerOutputStream *stream, *stderr_stream, *default_output;
    std::map<std::string, LoggerOutputStream *> logger_output;
  };

  LoggerConfig::LoggerConfig(void)
    : cmdline_read(false)
    , default_level(Logger::LEVEL_PRINT)
    , stderr_level(Logger::LEVEL_ERROR)
    , include_timestamp(true)
    , stream(0)
    , stderr_stream(0)
    , default_output(0)
  {}

  LoggerConfig::~LoggerConfig(void)
  {
    delete stream;
  }

  /*static*/ LoggerConfig *LoggerConfig::get_config(void)
  {
    static LoggerConfig cfg;
    return &cfg;
  }

  /*static*/ void LoggerConfig::flush_all_streams(void)
  {
    LoggerConfig *cfg = get_config();
    if(cfg->stream)
      cfg->stream->flush();
  }

  template <>
  bool convert_integer_cmdline_argument<Logger::LoggingLevel>(const std::string& s, Logger::LoggingLevel& target)
  {
    // match strings first
    if(s == "spew") {
      target = Logger::LEVEL_SPEW;
      return true;
    }
    if(s == "debug") {
      target = Logger::LEVEL_DEBUG;
      return true;
    }
    if(s == "info") {
      target = Logger::LEVEL_INFO;
      return true;
    }
    if(s == "print") {
      target = Logger::LEVEL_PRINT;
      return true;
    }
    if((s == "warning") || (s == "warn")) {
      target = Logger::LEVEL_WARNING;
      return true;
    }
    if(s == "error") {
      target = Logger::LEVEL_ERROR;
      return true;
    }
    if(s == "fatal") {
      target = Logger::LEVEL_FATAL;
      return true;
    }
    if(s == "none") {
      target = Logger::LEVEL_NONE;
      return true;
    }

    // try to decode an integer between LEVEL_SPEW and LEVEL_NONE
    errno = 0;  // no errors from before
    char *pos;
    long v = strtol(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0) && 
       (v >= Logger::LEVEL_SPEW) && (v <= Logger::LEVEL_NONE)) {
      target = static_cast<Logger::LoggingLevel>(v);
      return true;
    } else 
      return false;
  }

  bool LoggerConfig::parse_level_argument(const std::string& s)
  {
    const char *p1 = s.c_str();

    while(true) {
      // skip commas
      while(*p1 == ',') p1++;
      if(!*p1) break;

      // numbers may be preceeded by name= to specify a per-category level
      std::string catname;
      if(!isdigit(*p1)) {
        const char *p2 = p1;
        while(*p2 != '=') {
          if(!*p2) {
            fprintf(stderr, "ERROR: category name in -level must be followed by =\n");
            return false;
          }
          p2++;
        }
        catname.assign(p1, p2 - p1);
        p1 = p2 + 1;
      }

      // levels are small integers or words - scan forward to the first thing
      //  that's not a digit or a number - should be, or \0
      Logger::LoggingLevel lvl = Logger::LEVEL_SPEW;
      const char *p2 = p1;
      while(*p2 && isalnum(*p2)) p2++;
      if((!*p2 || (*p2 == ',')) &&
         convert_integer_cmdline_argument(std::string(p1, p2-p1), lvl)) {
        if(catname.empty()) {
          if(lvl < Logger::REALM_LOGGING_MIN_LEVEL)
            fprintf(stderr, "WARNING: requested default logger level of %d is below compile-time minimum (%d) - not all logging output will be visible\n",
                    lvl, Logger::REALM_LOGGING_MIN_LEVEL);
          default_level = lvl;
        } else {
          if(lvl < Logger::REALM_LOGGING_MIN_LEVEL)
            fprintf(stderr, "WARNING: requested logger level for category '%s' of %d is below compile-time minimum (%d) - not all logging output will be visible\n",
                    catname.c_str(), lvl, Logger::REALM_LOGGING_MIN_LEVEL);
          category_levels[catname] = lvl;
        }

        p1 = p2;
        continue;
      }

      fprintf(stderr, "ERROR: logger level malformed or out of range: '%s'\n", p1);
      return false;
    }

    return true;
  }

  void LoggerConfig::read_command_line(std::vector<std::string>& cmdline)
  {
    std::string logname;

    bool ok = CommandLineParser()
      .add_option_string("-cat", cats_enabled)
      .add_option_string("-logfile", logname)
      .add_option_method("-level", this, &LoggerConfig::parse_level_argument)
      .add_option_int("-errlevel", stderr_level)
      .add_option_int("-logtime", include_timestamp)
      .parse_command_line(cmdline);

    if(!ok) {
      fprintf(stderr, "couldn't parse logger config options\n");
      exit(1);
    }

    // TODO: can't currently change streams (they are cached in loggers)
    if(stream)
      return;

    // lots of choices for log output
    if(logname.empty()) {
      // the gasnet UDP job spawner (amudprun) seems to buffer stdout, so make stderr the default
#ifdef GASNET_CONDUIT_UDP
      stream = new LoggerFileStream(stderr, false, include_timestamp);
#else
      stream = new LoggerFileStream(stdout, false, include_timestamp);
#endif
    } else if(logname == "stdout") {
      stream = new LoggerFileStream(stdout, false, include_timestamp);
    } else if(logname == "stderr") {
      stream = new LoggerFileStream(stderr, false, include_timestamp);
    } else {
      // we're going to open a file, but key off a + for appending and
      //  look for a % for node number insertion
      bool append = false;
      size_t start = 0;

      if(logname[0] == '+') {
        append = true;
        start++;
      }

      FILE *f = 0;
      size_t pct = logname.find_first_of('%', start);
      if(pct == std::string::npos) {
        // no node number - everybody uses the same file
        if(Network::max_node_id > 0) {
          if(!append) {
            if(Network::my_node_id == 0)
              fprintf(stderr, "WARNING: all ranks are logging to the same output file - appending is forced and output may be jumbled\n");
            append = true;
          }
        }
        const char *fn = logname.c_str() + start;
        f = fopen(fn, append ? "a" : "w");
        if(!f) {
          fprintf(stderr, "could not open log file '%s': %s\n", fn, strerror(errno));
          exit(1);
        }
      } else {
        // replace % with node number
        char filename[256];
        sprintf(filename, "%.*s%d%s",
                (int)(pct - start), logname.c_str() + start, Network::my_node_id, logname.c_str() + pct + 1);

        f = fopen(filename, append ? "a" : "w");
        if(!f) {
          fprintf(stderr, "could not open log file '%s': %s\n", filename, strerror(errno));
          exit(1);
        }
      }
      // TODO: consider buffering in some cases?
      setbuf(f, 0); // disable output buffering
      stream = new LoggerFileStream(f, true, include_timestamp);

      // when logging to a file, also sent critical-enough messages to stderr
      if(stderr_level < Logger::LEVEL_NONE)
        stderr_stream = new LoggerFileStream(stderr, false, include_timestamp);
    }

    atexit(LoggerConfig::flush_all_streams);

    cmdline_read = true;
    if(!pending_configs.empty()) {
      for(std::set<Logger *>::iterator it = pending_configs.begin();
          it != pending_configs.end();
          it++)
        configure(*it);
      pending_configs.clear();
    }
  }

  void LoggerConfig::set_default_output(LoggerOutputStream *s)
  {
    // must be called before command line is parsed
    assert(!cmdline_read);
    default_output = s;
  }

  void LoggerConfig::set_logger_output(const std::string& name, LoggerOutputStream *s)
  {
    // must be called before command line is read
    assert(!cmdline_read);
    logger_output[name] = s;
  }

  void LoggerConfig::configure(Logger *logger)
  {
    // if we haven't read the command line yet, remember this for later
    if(!cmdline_read) {
      pending_configs.insert(logger);
      return;
    }

    // see if this logger is one of the categories we want
    if(!cats_enabled.empty()) {
      bool found = false;
      const char *p = cats_enabled.c_str();
      int l = logger->get_name().length();
      const char *n = logger->get_name().c_str();
      while(*p) {
        if(((p[l] == '\0') || (p[l] == ',')) && !strncmp(p, n, l)) {
          found = true;
          break;
        }
        // skip to after next comma
        while(*p && (*p != ',')) p++;
        while(*p && (*p == ',')) p++;
      }
      if(!found) {
        //printf("'%s' not in '%s'\n", n, cats_enabled);
        return;
      }
    }

    // see if the level for this category has been customized
    Logger::LoggingLevel level = default_level;
    {
      std::map<std::string, Logger::LoggingLevel>::const_iterator it = category_levels.find(logger->get_name());
      if(it != category_levels.end())
        level = it->second;
    }

    // figure out which stream(s) to use for this logger
    {
      std::map<std::string, LoggerOutputStream *>::const_iterator it = logger_output.find(logger->get_name());
      if(it != logger_output.end()) {
        // specific stream for this module
        logger->add_stream(it->second, level,
                           false,  /* don't delete */
                           false); /* don't flush each write */
      } else if(default_output != 0) {
        // app-provided default stream for this module
        logger->add_stream(default_output, level,
                           false,  /* don't delete */
                           false); /* don't flush each write */

      } else {
        // give this logger a copy of the global stream
        logger->add_stream(stream, level,
                           false,  /* don't delete */
                           false); /* don't flush each write */

        // also use the stderr_stream, if present
        // make sure not to log at a level noisier than requested for this category
        if(stderr_stream)
          logger->add_stream(stderr_stream, 
                             ((level > stderr_level) ? level : stderr_level),
                             false, /* don't delete */
                             true); /* flush each write */
      }
    }

    logger->configure_done();
  }

  void LoggerConfig::remove_logger(Logger *logger)
  {
    // if we haven't read the command line yet, don't try to configure the
    //  logger that was removed
    if(!cmdline_read)
      pending_configs.erase(logger);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct DelayedMessage

  struct DelayedMessage {
    DelayedMessage *next_msg;
    Logger::LoggingLevel level;
    size_t msglen;
    // string data stored immediately after the fixed-size structure
    char *msgdata() { return reinterpret_cast<char *>(this + 1); }
    void operator delete(void *p) {
      // Since this object has tail-padding, use an unsized delete to avoid
      // upsetting allocators that check sizes.
      ::operator delete(p);
    }
  };


  ////////////////////////////////////////////////////////////////////////
  //
  // class Logger

  Logger::Logger(const std::string& _name)
    : name(_name), log_level(LEVEL_SPEW)
    , configured(false)
    , delayed_message_head(0)
    , delayed_message_tail(&delayed_message_head)
  {
    LoggerConfig::get_config()->configure(this);
  }

  Logger::~Logger(void)
  {
    // tell the logger config we no longer exist
    LoggerConfig::get_config()->remove_logger(this);

    // go through our streams and delete any we're supposed to
    for(std::vector<LogStream>::iterator it = streams.begin();
        it != streams.end();
        it++)
      if(it->delete_when_done)
        delete it->s;

    streams.clear();

    // if for some reason we never got configured, delete any delayed messages
    while(delayed_message_head != 0) {
      DelayedMessage *next = delayed_message_head->next_msg;
      delete delayed_message_head;
      delayed_message_head = next;
    }
  }

  /*static*/ void Logger::configure_from_cmdline(std::vector<std::string>& cmdline)
  {
    LoggerConfig::get_config()->read_command_line(cmdline);
  }

  /*static*/ void Logger::set_default_output(LoggerOutputStream *s)
  {
    LoggerConfig::get_config()->set_default_output(s);
  }

  /*static*/ void Logger::set_logger_output(const std::string& name, LoggerOutputStream *s)
  {
    LoggerConfig::get_config()->set_logger_output(name, s);
  }

  void Logger::log_msg(LoggingLevel level, const char *msgdata, size_t msglen)
  {
    // if we're not configured yet, delay the message
    if(!configured) {
      size_t bytes = sizeof(DelayedMessage) + msglen;
      void *ptr = malloc(bytes);
      assert(ptr != 0);
      DelayedMessage *d = new(ptr) DelayedMessage;
      d->next_msg = 0;
      d->level = level;
      d->msglen = msglen;
      memcpy(d->msgdata(), msgdata, msglen);
      *delayed_message_tail = d;
      delayed_message_tail = &(d->next_msg);
      return;
    }

    // no logging of empty messages
    if(msglen == 0)
      return;

    // go through all the streams
    for(std::vector<LogStream>::iterator it = streams.begin();
              it != streams.end();
              it++) {
      if(level < it->min_level)
              continue;

      it->s->log_msg(level, name.c_str(), msgdata, msglen);

      if(it->flush_each_write)
        it->s->flush();
    }
  }

  void Logger::add_stream(LoggerOutputStream *s, LoggingLevel min_level,
                          bool delete_when_done, bool flush_each_write)
  {
    LogStream ls;
    ls.s = s;
    ls.min_level = min_level;
    ls.delete_when_done = delete_when_done;
    ls.flush_each_write = flush_each_write;
    streams.push_back(ls);
  }

  void Logger::configure_done(void)
  {
    configured = true;

    // compute the minimum logging level we're interested in
    log_level = LEVEL_NONE;
    for(std::vector<LogStream>::iterator it = streams.begin();
	it != streams.end();
	it++)
      if(it->min_level < log_level)
	log_level = it->min_level;

    // and now handle any delayed messages
    while(delayed_message_head != 0) {
      DelayedMessage *next = delayed_message_head->next_msg;
      if(delayed_message_head->level >= log_level)
	log_msg(delayed_message_head->level,
		delayed_message_head->msgdata(),
		delayed_message_head->msglen);
      // was allocated with malloc, not new
      delayed_message_head->~DelayedMessage();
      free(delayed_message_head);
      delayed_message_head = next;
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LoggerMessage
    

  LoggerMessage& LoggerMessage::vprintf(const char *typeName, LoggerMessageID messageID, const char *fmt, va_list args)
  {
    if(active) {
      static const int MAXLEN = 4096;
       char msg[MAXLEN] = {0};
       if(messageID != RESERVED_LOGGER_MESSAGE_ID) {
          sprintf(msg, "[%s %d] ", typeName, messageID);
       }
       int prefixLength = strlen(msg);
      int full = prefixLength + vsnprintf(msg + prefixLength, MAXLEN - prefixLength, fmt, args);
      // If this is an error or a warning, print out the full string
      // no matter what
      if((full >= MAXLEN) && ((level == Logger::LEVEL_FATAL) || 
          (level == Logger::LEVEL_ERROR) || (level == Logger::LEVEL_WARNING))) {
         char *full_msg;
         if(messageID == RESERVED_LOGGER_MESSAGE_ID) {
            full_msg = (char*)malloc(full+1);
            vsnprintf(full_msg, full+1, fmt, args);
         } else {
            const int MAX_LENGTH_MESSAGE_ID = 16;
            full_msg = (char*)malloc(full+1+MAX_LENGTH_MESSAGE_ID+2);
            sprintf(full_msg, "[%d] ", messageID);
            vsnprintf(full_msg + strlen(full_msg), full+1, fmt, args);
         }
         get_stream() << full_msg;
        free(full_msg);
      } else {
        get_stream() << msg;
      }
    }
    return *this;
  }
    
    LoggerMessage& LoggerMessage::vprintf(const char *fmt, va_list args)
    {
        return vprintf(NULL, RESERVED_LOGGER_MESSAGE_ID, fmt, args);
    }

}; // namespace Realm
