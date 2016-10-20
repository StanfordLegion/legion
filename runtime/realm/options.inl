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


namespace Realm {

  // command line options will override those in config file as
  // config file parsing is done in the parse_configfile_option functions
  // and command line is done in parse_command_line

  template <typename T>
  OptionParser& OptionParser::add_option_int(const std::string& optname,
                               T& target,
                               bool keep /*= false*/)
  {
    CommandLineParser::add_option_int(optname, target, keep);
    int x = static_cast<int>(target); 
    parse_configfile_option(optname, x);
    target = static_cast<T>(x);
    return *this;
  }


  template <typename T>
  OptionParser& OptionParser::add_option_string(const std::string& optname,
                              T& target,
                              bool keep /*= false*/)
  {
    CommandLineParser::add_option_string(optname, target, keep);
    parse_configfile_option(optname, target);
    return *this;
  }

  inline OptionParser& OptionParser::add_option_bool(const std::string& optname,
                                   bool& target,
                                   bool keep /*= false*/)
  {
    CommandLineParser::add_option_bool(optname, target, keep);
    parse_configfile_option(optname, target);
    return *this;
  }

  template <typename T>
  OptionParser& OptionParser::add_option_method(const std::string& optname,
                              T *target,
                              bool (T::*method)(const std::string&),
                              bool keep /*= false*/)
  {
    CommandLineParser::add_option_method(optname, target, method, keep);
    std::string str;
    parse_configfile_option(optname, str);
    (target->*method)(str);

    return *this;
  }

}; // namespace Realm
