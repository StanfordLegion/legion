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

#ifndef REALM_MODULECONFIG_H
#define REALM_MODULECONFIG_H

#include "realm/realm_config.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace Realm {

  class REALM_PUBLIC_API ModuleConfig {
  protected:
    ModuleConfig(const std::string &name);
  public:
    virtual ~ModuleConfig(void);
    template <typename T>
    bool set_property(const std::string name, T value);
    template <typename T>
    bool get_property(const std::string name, T &value) const;
    void finish_configure(void);
    const std::string& get_name(void) const;
    template <typename T>
    bool get_resource(const std::string name, T &value) const;
    virtual void configure_from_cmdline(std::vector<std::string>& cmdline);

  protected:
    // these maps store a map between configuration name and address of configurations,
    //  so that we can access these configurations using the name
    std::unordered_map<std::string, void* const> config_map;

    // these maps store a map between resource name and address of resource,
    //  so that we can access these resources using the name
    std::unordered_map<std::string, void *const> resource_map;

    std::string module_name;

    bool resource_discover_finished = false;

    bool finish_configured = false;
  };

}; // namespace Realm

#endif // ifndef REALM_MODULECONFIG_H

#include "realm/module_config.inl"
