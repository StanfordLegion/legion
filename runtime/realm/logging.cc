/* Copyright 2015 Stanford University
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

#include "logging.h"

#ifdef SHARED_LOWLEVEL
#define gasnet_mynode() 0
#define gasnet_nodes() 1
#else
#include "activemsg.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <fstream>
#include <set>

namespace Realm {

  class LoggerConfig {
  protected:
    LoggerConfig(void);
    ~LoggerConfig(void);

  public:
    static LoggerConfig *get_config(void);

    static void flush_all_streams(void);

    void read_command_line(int argc, const char *argv[]);

    // either configures a logger right away or remembers it to config once
    //   we know the desired settings
    void configure(Logger *logger);

  protected:
    bool cmdline_read;
    Logger::LoggingLevel level;
    char *cats_enabled;
    std::set<Logger *> pending_configs;
    bool delete_stream;
    std::ostream *stream;  // everybody uses the same one right now
  };

  LoggerConfig::LoggerConfig(void)
    : cmdline_read(false), level(Logger::LEVEL_PRINT), cats_enabled(0), delete_stream(false), stream(0)
  {}

  LoggerConfig::~LoggerConfig(void)
  {
    if(cats_enabled)
      free(cats_enabled);

    // only if we opened the file...
    if(delete_stream)
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

  void LoggerConfig::read_command_line(int argc, const char *argv[])
  {
    const char *logname = 0;

    for(int i = 1; i < argc-1; i++) {
      if(!strcmp(argv[i], "-level")) {
	int l = atoi(argv[++i]);
	assert((l >= 0) && (l <= Logger::LEVEL_NONE));
	level = (Logger::LoggingLevel)l;
	continue;
      }

      if(!strcmp(argv[i], "-cat")) {
	if(cats_enabled)
	  free(cats_enabled);
	cats_enabled = strdup(argv[++i]);
	continue;
      }

      if(!strcmp(argv[i], "-logfile")) {
	logname = argv[++i];
	continue;
      }
    }

    // lots of choices for log output
    if(!logname || !strcmp(logname, "stdout")) {
      stream = &std::cout;
      delete_stream = false;
    } else if(!strcmp(logname, "stderr")) {
      stream = &std::cerr;
      delete_stream = false;
    } else {
      // we're going to open a file, but key off a + for appending and
      //  look for a % for node number insertion
      std::ios_base::openmode mode = std::ios_base::out;

      if(*logname == '+') {
	mode |= std::ios_base::app;
	logname++;
      }

      const char *pos = strchr(logname, '%');
      if(pos == 0) {
	// no node number - everybody uses the same file
	if(gasnet_nodes() > 1) {
	  if(gasnet_mynode() == 1)
	    fprintf(stderr, "WARNING: all ranks are logging to the same output file - appending is forced and output may be jumbled\n");
	  mode |= std::ios_base::app;
	}
	stream = new std::fstream(logname, mode);
      } else {
	// replace % with node number
	char filename[256];
	sprintf(filename, "%.*s%d%s", (int)(pos - logname), logname, gasnet_mynode(), pos+1);

	stream = new std::fstream(filename, mode);
      }
      assert(stream->good());
      delete_stream = true;
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

  void LoggerConfig::configure(Logger *logger)
  {
    // if we haven't read the command line yet, remember this for later
    if(!cmdline_read) {
      pending_configs.insert(logger);
      return;
    }

    // see if this logger is one of the categories we want
    if(cats_enabled && (cats_enabled[0] != '*')) {
      bool found = false;
      const char *p = cats_enabled;
      int l = logger->get_name().length();
      const char *n = logger->get_name().c_str();
      while(*p) {
	if(((p[l] == '\0') || (p[l] == ',')) && !strcmp(p, n)) {
	  found = true;
	  break;
	}
	// skip to after next comma
	while(*p && (*p != ',')) p++;
	while(*p && (*p == ',')) p++;
      }
      if(!found) {
	printf("'%s' not in '%s'\n", n, cats_enabled);
	return;
      }
    }

    // give this logger a copy of the global stream
    logger->add_stream(stream, level, 
		       false,  /* don't delete */
		       false); /* don't flush each write */
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Logger

  Logger::Logger(const std::string& _name)
    : name(_name), log_level(LEVEL_NONE)
  {
    LoggerConfig::get_config()->configure(this);
  }

  Logger::~Logger(void)
  {
    // go through our streams and delete any we're supposed to
    for(std::vector<LogStream>::iterator it = streams.begin();
	it != streams.end();
	it++)
      if(it->delete_when_done)
	delete it->s;

    streams.clear();
  }

  /*static*/ void Logger::configure_from_cmdline(int argc, const char *argv[])
  {
    LoggerConfig::get_config()->read_command_line(argc, argv);
  }

  void Logger::log_msg(LoggingLevel level, const std::string& msg)
  {
    // no logging of empty messages
    if(msg.length() == 0)
      return;

    // build message string, including prefix
    static const int MAXLEN = 4096;
    char buffer[MAXLEN];
    int len = snprintf(buffer, MAXLEN - 2, "[%d - %lx] {%d}{%s}: ",
		       gasnet_mynode(), (unsigned long)pthread_self(),
		       level, name.c_str());
    int amt = msg.length();
    if((len + amt) >= MAXLEN)
      amt = MAXLEN - 2 - len;
    memcpy(buffer + len, msg.data(), amt);
    len += amt;
    buffer[len++] = '\n';
    buffer[len] = 0;
    
    // go through all the streams
    for(std::vector<LogStream>::iterator it = streams.begin();
	it != streams.end();
	it++) {
      if(level < it->min_level)
	continue;

      it->s->write(buffer, len);

      if(it->flush_each_write)
	it->s->flush();
    }
  }

  void Logger::add_stream(std::ostream *s, LoggingLevel min_level,
			  bool delete_when_done, bool flush_each_write)
  {
    LogStream ls;
    ls.s = s;
    ls.min_level = min_level;
    ls.delete_when_done = delete_when_done;
    ls.flush_each_write = flush_each_write;
    streams.push_back(ls);

    // update our logging level if needed
    if(log_level > min_level)
      log_level = min_level;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LoggerMessage

  LoggerMessage& LoggerMessage::vprintf(const char *fmt, va_list args)
  {
    if(active) {
      char msg[256];
      vsnprintf(msg, 256, fmt, args);
      oss << msg;
    }
    return *this;
  }

}; // namespace Realm
