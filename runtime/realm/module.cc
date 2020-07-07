/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#include "realm/realm_config.h"

#define REALM_MODULE_REGISTRATION_STATIC
#include "realm/module.h"

#include "realm/logging.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifdef REALM_USE_DLFCN
#include <dlfcn.h>
#endif

// TODO: replace this with Makefile (or maybe cmake) magic that adapts automatically
//  to the build-system-controlled list of statically-linked Realm modules
#include "realm/runtime_impl.h"
#include "realm/numa/numa_module.h"
#ifdef REALM_USE_OPENMP
#include "realm/openmp/openmp_module.h"
#endif
#include "realm/procset/procset_module.h"
#ifdef REALM_USE_PYTHON
#include "realm/python/python_module.h"
#endif
#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#endif
#ifdef REALM_USE_LLVM
#include "realm/llvmjit/llvmjit_module.h"
#endif
#ifdef REALM_USE_HDF5
#include "realm/hdf5/hdf5_module.h"
#endif
#ifdef REALM_USE_GASNET1
#include "realm/gasnet1/gasnet1_module.h"
#endif
#if defined REALM_USE_MPI
#include "realm/mpi/mpi_module.h"
#endif
#ifdef REALM_USE_ACCELERATOR
#include "realm/accelerator/accelerator_module.h"
#endif

namespace Realm {

  Logger log_module("module");

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class Module
  //

  Module::Module(const std::string& _name)
    : name(_name)
  {
    log_module.debug() << "module " << name << " created";
  }

  Module::~Module(void)
  {
    log_module.debug() << "module " << name << " destroyed";
  }

  const std::string& Module::get_name(void) const
  {
    return name;
  }

  void Module::initialize(RuntimeImpl *runtime)
  {
    log_module.debug() << "module " << name << " initialize";
  }

  void Module::create_memories(RuntimeImpl *runtime)
  {
    log_module.debug() << "module " << name << " create_memories";
  }

  void Module::create_processors(RuntimeImpl *runtime)
  {
    log_module.debug() << "module " << name << " create_processors";
  }
  
  void Module::create_dma_channels(RuntimeImpl *runtime)
  {
    log_module.debug() << "module " << name << " create_dma_channels";
  }
  
  void Module::create_code_translators(RuntimeImpl *runtime)
  {
    log_module.debug() << "module " << name << " create_code_translators";
  }

  void Module::cleanup(void)
  {
    log_module.debug() << "module " << name << " cleanup";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ModuleRegistrar
  //

  namespace {
    ModuleRegistrar::StaticRegistrationBase *static_modules_head = 0;
    ModuleRegistrar::StaticRegistrationBase **static_modules_tail = &static_modules_head;
  };

  ModuleRegistrar::ModuleRegistrar(RuntimeImpl *_runtime)
    : runtime(_runtime)
  {}

  // called by the runtime during init
  void ModuleRegistrar::create_static_modules(std::vector<std::string>& cmdline,
					      std::vector<Module *>& modules)
  {
    // just iterate over the static module list, trying to create each module
    for(const StaticRegistrationBase *sreg = static_modules_head;
	sreg;
	sreg = sreg->next) {
      Module *m = sreg->create_module(runtime, cmdline);
      if(m)
	modules.push_back(m);
    }
  }


#ifdef REALM_USE_DLFCN
  // accepts a colon-separated list of so files to try to load
  static void load_module_list(const char *sonames,
			       RuntimeImpl *runtime,
			       std::vector<std::string>& cmdline,
			       std::vector<void *>& handles,
			       std::vector<Module *>& modules)
  {
    // null/empty strings are nops
    if(!sonames || !*sonames) return;

    const char *p1 = sonames;
    while(true) {
      // skip leading colons
      while(*p1 == ':') p1++;
      if(!*p1) break;

      const char *p2 = p1 + 1;
      while(*p2 && (*p2 != ':')) p2++;

      char filename[1024];
      strncpy(filename, p1, p2 - p1);

      // no leftover errors from anybody else please...
      assert(dlerror() == 0);

      // open so file, resolving all symbols but not polluting global namespace
      void *handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);

      if(handle != 0) {
	// this file should have a "create_realm_module" symbol
	void *sym = dlsym(handle, "create_realm_module");

	if(sym != 0) {
	  // TODO: hold onto the handle even if it doesn't create a module?
	  handles.push_back(handle);

	  Module *m = ((Module *(*)(RuntimeImpl *, std::vector<std::string>&))dlsym)(runtime, cmdline);
	  if(m)
	    modules.push_back(m);
	} else {
	  log_module.error() << "symbol 'create_realm_module' not found in " << filename;
#ifndef NDEBUG
	  int ret =
#endif
	    dlclose(handle);
	  assert(ret == 0);
	}
      } else {
	log_module.error() << "could not load " << filename << ": " << dlerror();
      }

      if(!*p2) break;
      p1 = p2 + 1;
    }
  }
#endif

  // called by the runtime during init
  void ModuleRegistrar::create_dynamic_modules(std::vector<std::string>& cmdline,
					       std::vector<Module *>& modules)
  {
    // dynamic modules are requested in one of two ways:
    // 1) REALM_DYNAMIC_MODULES=sonames environment variable
    // 2) "-ll:module sonames" on command line
    // in both cases, 'sonames' is a colon-separate listed of .so files that should be

    // loading modules can also monkey with the cmdline, so do a pass first where we pull
    //  out all the name we want to load
    std::vector<std::string> sonames_list;

    {
      const char *e = getenv("REALM_DYNAMIC_MODULES");
      if(e)
	sonames_list.push_back(std::string(e));
    }

    {
      std::vector<std::string>::iterator it = cmdline.begin();
      while(it != cmdline.end()) {
	if(*it != "-ll:module") {
	  it++;
	  continue;
	}

	// eat this argument and move the next one to sonames_list
	it = cmdline.erase(it);
	assert(it != cmdline.end());
	sonames_list.push_back(*it);
	it = cmdline.erase(it);
      }
    }

#ifdef REALM_USE_DLFCN
    for(std::vector<std::string>::const_iterator it = sonames_list.begin();
	it != sonames_list.end();
	it++)
      load_module_list(it->c_str(),
		       runtime, cmdline, sofile_handles, modules);
#else
    if(!sonames_list.empty()) {
      log_module.error() << "loading of dynamic Realm modules requested, but REALM_USE_DLFCN=0!";
      exit(1);
    }
#endif
  }

  // called by runtime after all modules have been cleaned up
  void ModuleRegistrar::unload_module_sofiles(void)
  {
#ifdef REALM_USE_DLFCN
    while(!sofile_handles.empty()) {
      void *handle = sofile_handles.back();
      sofile_handles.pop_back();

#ifndef NDEBUG
      int ret =
#endif
	dlclose(handle);
      assert(ret == 0);
    }
#endif
  }

  // called by the module registration helpers
  /*static*/ void ModuleRegistrar::add_static_registration(StaticRegistrationBase *reg)
  {
    // done during init, so single-threaded
    *static_modules_tail = reg;
    static_modules_tail = &(reg->next);
  }
  
}; // namespace Realm
