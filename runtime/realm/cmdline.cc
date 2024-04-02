/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "realm/realm_config.h"

#include "realm/cmdline.h"

#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <string.h>

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

  RealmStatus CommandLineParser::parse_command_line_v2(std::vector<std::string> &cmdline)
  {
    RealmStatus status = REALM_SUCCESS;
    std::vector<std::string>::iterator pos = cmdline.begin();

    while(pos != cmdline.end()) {
      // try each option in turn
      std::vector<CommandLineOption *>::const_iterator it = options.begin();
      while((it != options.end()) && !((*it)->match(*pos)))
	it++;

      if(it == options.end()) {
        // not recognized - skip it and move on
        status = REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS;
        pos++;
        continue;
      }

      // parse the argument (if any)
      std::vector<std::string>::iterator arg = pos + 1;
      RealmStatus parse_status =
          static_cast<RealmStatus>((*it)->parse_argument(cmdline, arg));

      if(parse_status == REALM_SUCCESS ||
         parse_status == REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS) {
        if((*it)->keep_arg())
          pos = arg; // 'arg' was left pointing at spot _after_ last argument
        else
          pos = cmdline.erase(pos);
      } else
        return parse_status;
    }

    // got all the way through without errors
    return status;
  }

  RealmStatus CommandLineParser::parse_command_line_v2(int argc, const char *argv[])
  {
    RealmStatus status = REALM_SUCCESS;
    int pos = 0;

    while(pos < argc) {
      std::string s(argv[pos]);
      // try each option in turn
      std::vector<CommandLineOption *>::const_iterator it = options.begin();
      while((it != options.end()) && !((*it)->match(s)))
	it++;

      if(it == options.end()) {
        // not recognized - skip it and move on
        status = REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS;
        pos++;
        continue;
      }

      // parse the argument (if any)
      ++pos;
      RealmStatus parse_status =
          static_cast<RealmStatus>((*it)->parse_argument(pos, argc, argv));
      if(parse_status != REALM_SUCCESS &&
         parse_status != REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS)
        return parse_status;
    }

    // got all the way through without errors
    return status;
  }

  RealmStatus CommandLineParser::parse_command_line_v2(int argc, char *argv[])
  {
    // add the const ourselves - we're not going to modify the data (or add
    //  any strings to the array)
    return parse_command_line_v2(argc, const_cast<const char **>(argv));
  }

  bool CommandLineParser::parse_command_line(std::vector<std::string> &cmdline)
  {
    RealmStatus status = parse_command_line_v2(cmdline);
    return (status == REALM_SUCCESS || status == REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS)
               ? true
               : false;
  }

  bool CommandLineParser::parse_command_line(int argc, const char *argv[])
  {
    RealmStatus status = parse_command_line_v2(argc, argv);
    return (status == REALM_SUCCESS || status == REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS)
               ? true
               : false;
  }

  bool CommandLineParser::parse_command_line(int argc, char *argv[])
  {
    RealmStatus status = parse_command_line_v2(argc, argv);
    return (status == REALM_SUCCESS || status == REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS)
               ? true
               : false;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class IntegerCommandLineOption<T>

  // level of indirection allows us to just specialize the string parsing part
  template <>
  int convert_integer_cmdline_argument<int>(const std::string &s, int &target)
  {
    errno = 0;  // no errors from before
    char *pos;
    long v = strtol(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0) && (v >= INT_MIN) && (v <= INT_MAX)) {
      target = v;
      return REALM_SUCCESS;
    } else
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER;
  }

  template <>
  int convert_integer_cmdline_argument<unsigned int>(const std::string &s,
                                                     unsigned int &target)
  {
    errno = 0;  // no errors from before
    char *pos;
    unsigned long v = strtoul(s.c_str(), &pos, 10);
    // if s == "-1", strtoul convert it to ULONG_MAX, however, since
    // target is an unsigned int, it should be UINT_MAX
    if (v == ULONG_MAX) {
      v = UINT_MAX;
    }
    if((errno == 0) && (*pos == 0) && (v <= UINT_MAX)) {
      target = v;
      return REALM_SUCCESS;
    } else
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER;
  }

  template <>
  int convert_integer_cmdline_argument<long>(const std::string &s, long &target)
  {
    errno = 0;  // no errors from before
    char *pos;
    target = strtol(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0)) {
      return REALM_SUCCESS;
    } else
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER;
  }

  template <>
  int convert_integer_cmdline_argument<unsigned long>(const std::string &s,
                                                      unsigned long &target)
  {
    errno = 0;  // no errors from before
    char *pos;
    target = strtoul(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0)) {
      return REALM_SUCCESS;
    } else
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER;
  }

  template <>
  int convert_integer_cmdline_argument<long long>(const std::string &s, long long &target)
  {
    errno = 0;  // no errors from before
    char *pos;
    target = strtoll(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0)) {
      return REALM_SUCCESS;
    } else
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER;
  }

  template <>
  int convert_integer_cmdline_argument<unsigned long long>(const std::string &s,
                                                           unsigned long long &target)
  {
    errno = 0;  // no errors from before
    char *pos;
    target = strtoull(s.c_str(), &pos, 10);
    if ((errno == 0) && (*pos == 0)) {
      return REALM_SUCCESS;
    }
    else
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER;
  }

  template <>
  int convert_integer_cmdline_argument<bool>(const std::string &s, bool &target)
  {
    errno = 0;  // no errors from before
    char *pos;
    int v = strtol(s.c_str(), &pos, 10);
    if((errno == 0) && (*pos == 0) && (v >= 0) && (v <= 1)) {
      target = (v != 0);
      return REALM_SUCCESS;
    } else
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntegerUnitsCommandLineOption<T>

  int convert_integer_units_cmdline_argument(const char *s, char default_unit,
                                             bool binary, double &value)
  {
    errno = 0;  // no errors from before
    char *pos;
    // parse as floating point to allow things like 3.5g
    value = strtod(s, &pos);
    if(errno != 0)
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER_UNIT;
    char unit = tolower(*pos ? *pos++ : default_unit);
    switch(unit) {
    case 'k': value *= (binary ? 1024 : 1000); break;
    case 'm': value *= (binary ? 1048576 : 1000000); break;
    case 'g': value *= (binary ? 1073741824 : 1000000000); break;
    case 't': value *= (binary ? 1099511627776LL : 1000000000000LL); break;
    case 0:
    case 'b': break;
    default:
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER_UNIT;
    }
    // allow a trailing 'b' so that things like 'kb' work
    if(*pos && ((unit == 'b') || (tolower(*pos) != 'b')))
      return REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER_UNIT;

    return REALM_SUCCESS;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class StringCommandLineOption

  StringCommandLineOption::StringCommandLineOption(const std::string& _optname,
						   bool _keep,
						   std::string& _target)
    : CommandLineOption(_optname, _keep)
    , target_str(&_target)
    , target_array(0)
    , target_arrlen(0)
  {}

  StringCommandLineOption::StringCommandLineOption(const std::string& _optname,
						   bool _keep,
						   char *_target,
						   size_t _maxlen)
    : CommandLineOption(_optname, _keep)
    , target_str(0)
    , target_array(_target)
    , target_arrlen(_maxlen)
  {}

  int StringCommandLineOption::parse_argument(std::vector<std::string> &cmdline,
                                              std::vector<std::string>::iterator &pos)
  {
    // requires an additional argument
    if(pos == cmdline.end())
      return REALM_ARGUMENT_ERROR_MISSING_INPUT;

    if(target_str)
      *target_str = *pos;
    if(target_array) {
      // check length first to avoid corrupting the value on failure
      if(pos->size() >= target_arrlen)
        return REALM_ARGUMENT_ERROR_OUTPUT_STRING_TOO_SHORT;
      strcpy(target_array, pos->c_str());
    }

    if(keep) {
      ++pos;
    } else {
      pos = cmdline.erase(pos);
    }

    return REALM_SUCCESS;
  }

  int StringCommandLineOption::parse_argument(int &pos, int argc, const char *argv[])
  {
    // requires an additional argument
    if(pos >= argc)
      return REALM_ARGUMENT_ERROR_MISSING_INPUT;

    if(target_str)
      *target_str = argv[pos];
    if(target_array) {
      // check length first to avoid corrupting the value on failure
      if(strlen(argv[pos]) >= target_arrlen)
        return REALM_ARGUMENT_ERROR_OUTPUT_STRING_TOO_SHORT;
      strcpy(target_array, argv[pos]);
    }

    // always keep
    ++pos;

    return REALM_SUCCESS;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class StringListCommandLineOption

  StringListCommandLineOption::StringListCommandLineOption(const std::string& _optname,
							   bool _keep,
							   std::vector<std::string>& _target)
    : CommandLineOption(_optname, _keep)
    , target(_target)
  {}

  int StringListCommandLineOption::parse_argument(std::vector<std::string> &cmdline,
                                                  std::vector<std::string>::iterator &pos)
  {
    // requires an additional argument
    if(pos == cmdline.end())
      return REALM_ARGUMENT_ERROR_MISSING_INPUT;

    target.push_back(*pos);

    if(keep) {
      ++pos;
    } else {
      pos = cmdline.erase(pos);
    }

    return REALM_SUCCESS;
  }

  int StringListCommandLineOption::parse_argument(int &pos, int argc, const char *argv[])
  {
    // requires an additional argument
    if(pos >= argc)
      return REALM_ARGUMENT_ERROR_MISSING_INPUT;

    target.push_back(argv[pos]);

    // always keep
    ++pos;

    return REALM_SUCCESS;
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

  int BooleanCommandLineOption::parse_argument(std::vector<std::string> &cmdline,
                                               std::vector<std::string>::iterator &pos)
  {
    // nothing to parse - all we care about is presence
    target = true;
    return REALM_SUCCESS;
  }

  int BooleanCommandLineOption::parse_argument(int &pos, int argc, const char *argv[])
  {
    // nothing to parse - all we care about is presence
    target = true;
    return REALM_SUCCESS;
  }


}; // namespace Realm
