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

// Realm modules

#ifndef REALM_MODULE_H
#define REALM_MODULE_H

#include "realm/realm_config.h"

// to provide Realm functionality via the module interface, you need to:
//
// 1) define a subclass of Realm::Module, implementing all the methods
// 2) include the following in your .cc file:
//      REGISTER_REALM_MODULE(YourModuleClassName);
// 3) add your module to REALM_MODULES in the Makefile

#include <vector>
#include <string>

namespace Realm {

  class RuntimeImpl;
  class NetworkModule;

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE Module {
  protected:
    // can only construct subclasses of Module
    Module(const std::string& _name);

  public:
    virtual ~Module(void);

    const std::string& get_name(void) const;

    // all subclasses should define this (static) method - its responsibilities are:
    // 1) parse command line arguments in 'cmdline', removing any recognized arguments
    // 2) do any necessary system discovery
    // 3) do NOT actually allocate any resources yet
    // 4) register machine notification callbacks, if desired
    // 5) create an instance of a Module subclass and return it
    //
    // if the module cannot possibly be used in the current run (e.g. the CUDA module when
    //   no GPUs are present), 'create_module' can just return a null pointer
    //
    // static Module *create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline);

    // do any general initialization - this is called after all configuration is
    //  complete
    virtual void initialize(RuntimeImpl *runtime);

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    virtual void create_memories(RuntimeImpl *runtime);

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    virtual void create_processors(RuntimeImpl *runtime);

    // create any DMA channels provided by the module (default == do nothing)
    virtual void create_dma_channels(RuntimeImpl *runtime);

    // create any code translators provided by the module (default == do nothing)
    virtual void create_code_translators(RuntimeImpl *runtime);

    // if a module has to do cleanup that involves sending messages to other
    //  nodes, this must be done in the pre-detach cleanup
    virtual void pre_detach_cleanup(void);

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    virtual void cleanup(void);

  protected:
    std::string name;
  };

  // module-specific information can be attached to processors/memories/etc.
  //  as a sort of "mix-in", providing module-specific data and/or methods
  // finding module-specific info is based on knowing the type and finding it
  //  via dynamic casts
  class ModuleSpecificInfo {
  public:
    ModuleSpecificInfo();
    virtual ~ModuleSpecificInfo();

    ModuleSpecificInfo *next;
  };

  // helper class for module registration - supports modules compiled in statically
  //  and the other for dynamically-loaded modules

  class ModuleRegistrar {
  public:
    ModuleRegistrar(RuntimeImpl *_runtime);

    // called by the runtime during init - these may change the command line!
    void create_network_modules(std::vector<NetworkModule *>& modules,
				int *argc, const char ***argv);

    // called by the runtime during init
    void create_static_modules(std::vector<std::string>& cmdline,
			       std::vector<Module *>& modules);

    // called by the runtime during init
    void create_dynamic_modules(std::vector<std::string>& cmdline,
				std::vector<Module *>& modules);

    // called by runtime after all modules have been cleaned up
    void unload_module_sofiles(void);

    // TODO: consider some sort of "priority" scheme to order modules' inits?
    class StaticRegistrationBase {
    public:
      StaticRegistrationBase() : next(0) {}
      virtual Module *create_module(RuntimeImpl *runtime,
				    std::vector<std::string>& cmdline) const = 0;
      StaticRegistrationBase *next;
    };
    template <typename T>
    class StaticRegistration : public StaticRegistrationBase {
    public:
      StaticRegistration(void)
      {
	ModuleRegistrar::add_static_registration(this);
      }

      virtual Module *create_module(RuntimeImpl *runtime,
				    std::vector<std::string>& cmdline) const
      {
	return T::create_module(runtime, cmdline);
      }
    };

    // called by the module registration helpers
    static void add_static_registration(StaticRegistrationBase *reg);

    static bool check_symbol_visibility(void);

    // similar constructs for network modules
    class NetworkRegistrationBase {
    public:
      NetworkRegistrationBase() : next(0) {}
      virtual NetworkModule *create_network_module(RuntimeImpl *runtime,
						   int *argc,
						   const char ***argv) const = 0;
      NetworkRegistrationBase *next;
    };
    template <typename T>
    class NetworkRegistration : public NetworkRegistrationBase {
    public:
      NetworkRegistration(void)
      {
	ModuleRegistrar::add_network_registration(this);
      }

      virtual NetworkModule *create_network_module(RuntimeImpl *runtime,
						   int *argc,
						   const char ***argv) const
      {
	return T::create_network_module(runtime, argc, argv);
      }
    };

    // called by the module registration helpers
    static void add_network_registration(NetworkRegistrationBase *reg);

  protected:
    RuntimeImpl *runtime;
    std::vector<void *> sofile_handles;
  };
	
  // macros used within a module when being built as a dynamic shared object
#define REGISTER_REALM_MODULE_DYNAMIC(classname) \
  extern "C" { \
    REALM_INTERNAL_API_EXTERNAL_LINKAGE char realm_module_version[] = REALM_VERSION; \
    REALM_INTERNAL_API_EXTERNAL_LINKAGE Realm::Module *create_realm_module(Realm::RuntimeImpl *runtime, std::vector<std::string>& cmdline) { return classname::create_module(runtime, cmdline); } \
  }
#define REGISTER_REALM_NETWORK_MODULE_DYNAMIC(classname) \
  extern "C" { \
    REALM_INTERNAL_API_EXTERNAL_LINKAGE char realm_module_version[] = REALM_VERSION; \
    REALM_INTERNAL_API_EXTERNAL_LINKAGE Realm::NetworkModule *create_realm_network_module(Realm::RuntimeImpl *runtime, int *argc, const char ***argv) { return classname::create_network_module(runtime, argc, argv); } \
  }

}; // namespace Realm

#endif // ifndef REALM_MODULE_H

