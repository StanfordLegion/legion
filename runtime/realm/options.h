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

#ifndef REALM_CONFIGFILE_H
#define REALM_CONFIGFILE_H

#include "cmdline.h"

#ifdef REALM_USE_LUAJIT

#include <string>
#include <vector>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

}

namespace Realm {

  class OptionParser : public CommandLineParser {
  public:
     OptionParser(std::vector<std::string>& cmdline);
    ~OptionParser(void);

    template <typename T>
    OptionParser& add_option_int(const std::string& optname, T& target, bool keep = false);

    template <typename T>
    OptionParser& add_option_string(const std::string& optname, T& target, bool keep = false);

    OptionParser& add_option_bool(const std::string& optname, bool& target, bool keep = false);

    template <typename T>
    OptionParser& add_option_method(const std::string& optname, T *target,
                     bool (T::*method)(const std::string&), bool keep = false);

    void parse_configfile_option(const std::string& optname, bool& target);
    void parse_configfile_option(const std::string& optname, int& target);
    void parse_configfile_option(const std::string& optname, std::string& target);

  protected:
    lua_State *luaState;
  };


}; // namespace Realm

#include "options.inl"

#else

// no luajit just use CommandLineParser
namespace Realm {
  class OptionParser : public CommandLineParser {
  public: 
     OptionParser(std::vector<std::string>& cmdline) {} 
  };
}; 

#endif

#endif

