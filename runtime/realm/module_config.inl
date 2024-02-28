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

// model configure for Realm

#ifndef REALM_MODULECONFIG_INL
#define REALM_MODULECONFIG_INL

// nop, but helpful for IDEs
#include "realm/module_config.h"

#include "realm/logging.h"

namespace Realm {

  REALM_INTERNAL_API_EXTERNAL_LINKAGE
  extern Logger log_moduleconfig;

  ////////////////////////////////////////////////////////////////////////
  //
  // class ModuleConfig
  //

  template <typename T>
  bool ModuleConfig::set_property(const std::string name, T value)
  {
    std::unordered_map<std::string, void* const>::iterator it = config_map.find(name);
    if (it == config_map.end()) {
      log_moduleconfig.error("Module %s does not have the configuration: %s", module_name.c_str(), name.c_str());
      return false;
    } else {
      *reinterpret_cast<T*>(it->second) = value;
      return true;
    }
  }

  template <typename T>
  bool ModuleConfig::get_property(const std::string name, T &value) const
  {
    std::unordered_map<std::string, void* const>::const_iterator it = config_map.find(name);
    if (it == config_map.cend()) {
      log_moduleconfig.error("Module %s does not have the configuration: %s", module_name.c_str(), name.c_str());
      value = 0;
      return false;
    } else {
      value = *reinterpret_cast<T*>(it->second);
      return true;
    }
  }

  template <typename T>
  bool ModuleConfig::get_resource(const std::string name, T &value) const
  {
    if (!resource_discover_finished) {
      log_moduleconfig.error("Module %s can not detect resources.",
                             module_name.c_str());
      return false;
    }

    std::unordered_map<std::string, void* const>::const_iterator it = resource_map.find(name);
    if (it == resource_map.cend()) {
      log_moduleconfig.error("Module %s does not have the resource: %s", module_name.c_str(), name.c_str());
      return false;
    } else {
      value = *reinterpret_cast<T*>(it->second);
      return true;
    }
  }

};

#endif // ifndef REALM_MODULECONFIG_INL
