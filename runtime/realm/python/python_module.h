/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#ifndef REALM_PYTHON_MODULE_H
#define REALM_PYTHON_MODULE_H

#include "realm/realm_config.h"
#include "realm/module.h"

#include <set>

namespace Realm {

  namespace Python {

    // our interface to the rest of the runtime
    class PythonModule : public Module {
    protected:
      PythonModule(void);

    public:
      virtual ~PythonModule(void);

      // Request that the named Python module be imported
      static void import_python_module(const char *module_name);

      static Module *create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline);

      // do any general initialization - this is called after all configuration is
      //  complete
      virtual void initialize(RuntimeImpl *runtime);

      // create any memories provided by this module (default == do nothing)
      //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
      //virtual void create_memories(RuntimeImpl *runtime);

      // create any processors provided by the module (default == do nothing)
      //  (each new ProcessorImpl should use a Processor from
      //   RuntimeImpl::next_local_processor_id)
      virtual void create_processors(RuntimeImpl *runtime);

      // create any DMA channels provided by the module (default == do nothing)
      //virtual void create_dma_channels(RuntimeImpl *runtime);

      // create any code translators provided by the module (default == do nothing)
      //virtual void create_code_translators(RuntimeImpl *runtime);

      // clean up any common resources created by the module - this will be called
      //  after all memories/processors/etc. have been shut down and destroyed
      virtual void cleanup(void);

    public:
      static std::vector<std::string> extra_import_modules;

      int cfg_num_python_cpus;
      bool cfg_use_numa;
      size_t cfg_stack_size;
#ifdef REALM_USE_OPENMP
      int cfg_pyomp_threads;
#endif
      std::vector<std::string> cfg_import_modules;
      std::vector<std::string> cfg_init_scripts;

      std::set<int> active_numa_domains;
    };

  }; // namespace Python

}; // namespace Realm

#endif
