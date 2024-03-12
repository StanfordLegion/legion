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

#include "realm/module_config.h"

namespace Realm {

  Logger log_moduleconfig("module_config");

  ////////////////////////////////////////////////////////////////////////
  //
  // class ModuleConfig
  //

  ModuleConfig::ModuleConfig(const std::string &name)
    : module_name(name)
  {}

  ModuleConfig::~ModuleConfig(void)
  {}

  void ModuleConfig::finish_configure(void)
  {
    assert(finish_configured == false);
    finish_configured = true;
  }

  const std::string& ModuleConfig::get_name(void) const
  {
    return module_name;
  }

  void ModuleConfig::configure_from_cmdline(std::vector<std::string>& cmdline)
  {
    log_moduleconfig.error("Module %s does not have implemented for configure_cmdline size_t", module_name.c_str());
    abort();
  }

};
