/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// command-line processing helpers

#include "cmdline.h"

#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CommandLineParser

  CommandLineParser::CommandLineParser(void)
  {}

  CommandLineParser::~CommandLineParser(void)
  {
    for(std::vector<CommandLineOption *>::iterator it = options.begin();
	it != options.end();
	it++)
      delete (*it);

    options.clear();
  }

  bool CommandLineParser::parse_command_line(std::vector<std::string>& cmdline)
  {
    std::vector<std::string>::iterator pos = cmdline.begin();

    while(pos != cmdline.end()) {
      // try each option in turn
      std::vector<CommandLineOption *>::const_iterator it = options.begin();
      while((it != options.end()) && !((*it)->match(*pos)))
	it++;

      if(it == options.end()) {
	// not recognized - skip it and move on
	pos++;
	continue;
      }

      // parse the argument (if any)
      std::vector<std::string>::iterator arg = pos + 1;
      bool ok = (*it)->parse_argument(cmdline, arg);

      if(ok) {
	if((*it)->keep_arg())
	  pos = arg;  // 'arg' was left pointing at spot _after_ last argument
	else
	  pos = cmdline.erase(pos);
      } else
	return false;
    }

    // got all the way through without errors
    return true;
  }

  bool CommandLineParser::parse_command_line(int argc, const char *argv[])
  {
    int pos = 0;

    while(pos < argc) {
      std::string s(argv[pos]);
      // try each option in turn
      std::vector<CommandLineOption *>::const_iterator it = options.begin();
      while((it != options.end()) && !((*it)->match(s)))
	it++;

      if(it == options.end()) {
	// not recognized - skip it and move on
	pos++;
	continue;
      }

      // parse the argument (if any)
      ++pos;
      bool ok = (*it)->parse_argument(pos, argc, argv);
      if(!ok)
	return false;
    }

    // got all the way through without errors
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntegerCommandLineOption<T>

  // level of indirection allows us to just specialize the string parsing part
  template <>
  bool convert_integer_cmdline_argument<int>(const std::string& s, int& target)
  {
    errno = 0;  // no errors from before
    char *pos;
    long v = strtol(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0) && (v >= INT_MIN) && (v <= INT_MAX)) {
      target = v;
      return true;
    } else 
      return false;
  }

  template <>
  bool convert_integer_cmdline_argument<unsigned int>(const std::string& s, unsigned int& target)
  {
    errno = 0;  // no errors from before
    char *pos;
    unsigned long v = strtoul(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0) && (v <= UINT_MAX)) {
      target = v;
      return true;
    } else 
      return false;
  }

  template <>
  bool convert_integer_cmdline_argument<long>(const std::string& s, long& target)
  {
    errno = 0;  // no errors from before
    char *pos;
    target = strtol(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0)) {
      return true;
    } else 
      return false;
  }

  template <>
  bool convert_integer_cmdline_argument<unsigned long>(const std::string& s, unsigned long& target)
  {
    errno = 0;  // no errors from before
    char *pos;
    target = strtoul(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0)) {
      return true;
    } else 
      return false;
  }

  template <>
  bool convert_integer_cmdline_argument<bool>(const std::string& s, bool& target)
  {
    errno = 0;  // no errors from before
    char *pos;
    int v = strtol(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0) && (v >= 0) && (v <= 1)) {
      target = (v != 0);
      return true;
    } else 
      return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class StringCommandLineOption

  StringCommandLineOption::StringCommandLineOption(const std::string& _optname,
						   bool _keep,
						   std::string& _target)
    : CommandLineOption(_optname, _keep)
    , target(_target)
  {}

  bool StringCommandLineOption::parse_argument(std::vector<std::string>& cmdline,
					       std::vector<std::string>::iterator& pos)
  {
    // requires an additional argument
    if(pos == cmdline.end()) return false;

    // parse into a copy to avoid corrupting the value on failure
    target = *pos;

    if(keep) {
      ++pos;
    } else {
      pos = cmdline.erase(pos);
    }
    
    return true;
  }

  bool StringCommandLineOption::parse_argument(int& pos, int argc, const char *argv[])
  {
    // requires an additional argument
    if(pos >= argc) return false;

    // parse into a copy to avoid corrupting the value on failure
    target = argv[pos];

    // always keep
    ++pos;
    
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class BooleanCommandLineOption

  BooleanCommandLineOption::BooleanCommandLineOption(const std::string& _optname,
						     bool _keep,
						     bool& _target)
    : CommandLineOption(_optname, _keep)
    , target(_target)
  {}

  bool BooleanCommandLineOption::parse_argument(std::vector<std::string>& cmdline,
						std::vector<std::string>::iterator& pos)
  {
    // nothing to parse - all we care about is presence
    target = true;
    return true;
  }

  bool BooleanCommandLineOption::parse_argument(int& pos, int argc, const char *argv[])
  {
    // nothing to parse - all we care about is presence
    target = true;
    return true;
  }


}; // namespace Realm
