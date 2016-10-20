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


#include "options.h"
#include "logging.h"

namespace Realm {

  Logger log_config("config");


  // convert command line options to lua variable names
  // -level -> opt_level
  // -ll:cpu -> opt_ll_cpu
  // etc
  std::string luaName(const std::string& optname)
  {
    std::string name = optname;

    // replace : w/ _
    size_t cidx = name.find(":");
    if (cidx != std::string::npos) {
      name.replace(cidx, 1, "_");
    }
    return "opt_" + name.substr(1); // replace dash w/ opt_
  }

  OptionParser::OptionParser(std::vector<std::string>& cmdline)
  {
    std::string luaConfig; 
    CommandLineParser cpf;
    cpf.add_option_string("-ll:lua_config", luaConfig);
    cpf.parse_command_line(cmdline);
   
    luaState = luaL_newstate();
    if(!luaState) {
      log_config.fatal("could not get lua state");
    }
 
    // Open standard libraries 
    luaL_openlibs(luaState); 
    // Load config file
    if(!luaConfig.empty()) { 
      if (luaL_loadfile(luaState, luaConfig.c_str()) == 0) {
        int ret = lua_pcall(luaState, 0, 0, 0);
        if (ret != 0) {
          log_config.fatal(lua_tostring(luaState, -1));
        }
      } else {
          log_config.fatal("invalid lua config file");
      }
    } else {
      luaConfig = "config.lua";
      log_config.info("using default config.lua");
      if (luaL_loadfile(luaState, luaConfig.c_str()) == 0) {
        int ret = lua_pcall(luaState, 0, 0, 0);
        if (ret != 0) {
          log_config.fatal(lua_tostring(luaState, -1));
        }
      } else {
         log_config.info("invalid or missing config.lua");
         luaState = NULL;
      }
    }
  }

  OptionParser::~OptionParser(void) 
  {
     if (luaState) lua_close(luaState); 
  }

  void OptionParser::parse_configfile_option(const std::string& optname, int& target) 
  {
    if (luaState) {
      std::string name = luaName(optname);
      lua_getglobal(luaState, name.c_str());
      if (lua_isnumber(luaState, -1 )) {
        target = lua_tointeger(luaState, -1 );
      }
      lua_settop(luaState, 0);
    }
  }

  void OptionParser::parse_configfile_option(const std::string& optname, bool& target)
  {
    if (luaState) {
      std::string name = luaName(optname);
      lua_getglobal(luaState, name.c_str());
      // accept 0/1 or true/false
      if (lua_isnumber(luaState, -1 )) {
        if (lua_tointeger(luaState, -1 ) == 1) {
          target = true;
        }
      } else if (lua_isboolean(luaState, -1 )) {
        if (lua_toboolean(luaState, -1 ) == 1) {
          target = true;
        }
      }
      lua_settop(luaState, 0);
    }
  }

  void OptionParser::parse_configfile_option(const std::string& optname, std::string& target)
  {
    if (luaState) {
      std::string name = luaName(optname);
      lua_getglobal(luaState, name.c_str());
      const char *str = lua_tostring(luaState, -1);
      if (str != NULL) {
        target = std::string(str);
      }
      lua_settop(luaState, 0);
    }
  } 

}; // namespace Realm
