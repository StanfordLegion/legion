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

// command-line processing helpers

// nop, but helps IDEs
#include "cmdline.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CommandLineParser

  template <typename T>
  CommandLineParser& CommandLineParser::add_option_int(const std::string& optname,
						       T& target,
						       bool keep /*= false*/)
  {
    options.push_back(new IntegerCommandLineOption<T>(optname, keep, target));
    return *this;
  }

  template <typename T>
  CommandLineParser& CommandLineParser::add_option_string(const std::string& optname,
							  T& target,
							  bool keep /*= false*/)
  {
    options.push_back(new StringCommandLineOption(optname, keep, target));
    return *this;
  }

  inline CommandLineParser& CommandLineParser::add_option_bool(const std::string& optname,
							       bool& target,
							       bool keep /*= false*/)
  {
    options.push_back(new BooleanCommandLineOption(optname, keep, target));
    return *this;
  }

  template <typename T>
  CommandLineParser& CommandLineParser::add_option_method(const std::string& optname,
							  T *target,
							  bool (T::*method)(const std::string&),
							  bool keep /*= false*/)
  {
    options.push_back(new MethodCommandLineOption<T>(optname, keep, target, method));
    return *this;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CommandLineOption

  inline CommandLineOption::CommandLineOption(const std::string& _optname, bool _keep)
    : optname(_optname), keep(_keep)
  {}
  
  inline CommandLineOption::~CommandLineOption(void)
  {}

  inline bool CommandLineOption::match(const std::string& s)
  {
    return optname == s;
  }

  inline bool CommandLineOption::keep_arg(void) const
  {
    return keep;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntegerCommandLineOption<T>

  // level of indirection allows us to just specialize the string parsing part
  template <typename T>
  bool convert_integer_cmdline_argument(const std::string& s, T& target);

  template <>
  bool convert_integer_cmdline_argument<int>(const std::string& s, int& target);

  template <>
  bool convert_integer_cmdline_argument<unsigned int>(const std::string& s, unsigned int& target);

  template <>
  bool convert_integer_cmdline_argument<unsigned long>(const std::string& s, unsigned long& target);
  
  template <>
  bool convert_integer_cmdline_argument<long long>(const std::string& s, long long& target);

  template <>
  bool convert_integer_cmdline_argument<bool>(const std::string& s, bool& target);

  template <typename T>
  IntegerCommandLineOption<T>::IntegerCommandLineOption(const std::string& _optname,
							bool _keep,
							T& _target)
    : CommandLineOption(_optname, _keep)
    , target(_target)
  {}
    
  template <typename T>
  bool IntegerCommandLineOption<T>::parse_argument(std::vector<std::string>& cmdline,
						   std::vector<std::string>::iterator& pos)
  {
    // requires an additional argument
    if(pos == cmdline.end()) return false;

    // parse into a copy to avoid corrupting the value on failure
    T val;
    if(convert_integer_cmdline_argument(*pos, val)) {
      target = val;
      if(keep) {
	++pos;
      } else {
	pos = cmdline.erase(pos);
      }
      return true;
    } else
      return false;
  }

  template <typename T>
  bool IntegerCommandLineOption<T>::parse_argument(int& pos, int argc,
						   const char *argv[])
  {
    // requires an additional argument
    if(pos >= argc) return false;

    // parse into a copy to avoid corrupting the value on failure
    T val;
    if(convert_integer_cmdline_argument(argv[pos], val)) {
      target = val;
      // can't update this array - have to keep
      ++pos;
      return true;
    } else
      return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class MethodCommandLineOption<T>

  template <typename T>
  MethodCommandLineOption<T>::MethodCommandLineOption(const std::string& _optname,
						      bool _keep,
						      T *_target,
						      bool (T::*_method)(const std::string&))
    : CommandLineOption(_optname, _keep)
    , target(_target), method(_method)
  {}
    
  template <typename T>
  bool MethodCommandLineOption<T>::parse_argument(std::vector<std::string>& cmdline,
						  std::vector<std::string>::iterator& pos)
  {
    // requires an additional argument
    if(pos == cmdline.end()) return false;

    // call method - true means it parsed ok
    if((target->*method)(*pos)) {
      if(keep) {
	++pos;
      } else {
	pos = cmdline.erase(pos);
      }
      return true;
    } else
      return false;
  }

  template <typename T>
  bool MethodCommandLineOption<T>::parse_argument(int& pos, int argc,
						  const char *argv[])
  {
    // requires an additional argument
    if(pos >= argc) return false;

    // call method - true means it parsed ok
    if((target->*method)(argv[pos])) {
      // can't update this array - have to keep
      ++pos;
      return true;
    } else
      return false;
  }


}; // namespace Realm
