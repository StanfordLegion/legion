/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "realm/hdf5/hdf5_module.h"

#include "realm/hdf5/hdf5_internal.h"

#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"
#include "realm/inst_impl.h"

namespace Realm {

  Logger log_hdf5("hdf5");


  namespace HDF5 {

    static HDF5Module *hdf5mod = 0;

    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5Module

    HDF5Module::HDF5Module(void)
      : Module("hdf5")
      , cfg_showerrors(true)
      , version_major(0)
      , version_minor(0)
      , version_rel(0)
      , threadsafe(false)
      , hdf5mem(0)
    {
    }
      
    HDF5Module::~HDF5Module(void)
    {}

    /*static*/ Module *HDF5Module::create_module(RuntimeImpl *runtime,
						 std::vector<std::string>& cmdline)
    {
      // create a module to fill in with stuff - we'll delete it if HDF5 is
      //  disabled
      HDF5Module *m = new HDF5Module;

      // first order of business - read command line parameters
      {
	CommandLineParser cp;

	cp.add_option_bool("-hdf5:showerrors", m->cfg_showerrors);
	
	bool ok = cp.parse_command_line(cmdline);
	if(!ok) {
	  log_hdf5.fatal() << "error reading HDF5 command line parameters";
	  assert(false);
	}
      }

      {
	herr_t err = H5open();
	if(err < 0) {
	  log_hdf5.warning() << "unable to open HDF5 library - result = " << err;
	  delete m;
	  return 0;
	}
      }

      // ask for the version number and whether the library is thread-safe
      {
	herr_t err;
	err = H5get_libversion(&m->version_major,
			       &m->version_minor,
			       &m->version_rel);
	assert(err == 0);
#ifdef HDF5_HAS_IS_LIBRARY_THREADSAFE
	err = H5is_library_threadsafe(&m->threadsafe);
	assert(err == 0);
#else
#ifdef H5_HAVE_THREADSAFE
	m->threadsafe = true;
#else
	m->threadsafe = false;
#endif
#endif
	log_hdf5.info() << "HDF5 library initialized: version " << m->version_major
			<< '.' << m->version_minor << '.' << m->version_rel
			<< (m->threadsafe ? " (thread-safe)" : " (NOT thread-safe)");
      }

      hdf5mod = m; // hack for now
      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void HDF5Module::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);

#if 0
      // memory allocations are performed here
      for(std::map<int, void *>::iterator it = HDF5_mem_bases.begin();
	  it != HDF5_mem_bases.end();
	  ++it) {
	void *base = HDF5sysif_alloc_mem(it->first,
					 cfg_HDF5_mem_size_in_mb << 20,
					 cfg_pin_memory);
	if(!base) {
	  log_hdf5.fatal() << "allocation of " << cfg_HDF5_mem_size_in_mb << " MB in HDF5 node " << it->first << " failed!";
	  assert(false);
	}
	it->second = base;
      }
#endif
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void HDF5Module::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);

      Memory m = runtime->next_local_memory_id();
      hdf5mem = new HDF5Memory(m);
      runtime->add_memory(hdf5mem);
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void HDF5Module::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);
    }
    
    // create any DMA channels provided by the module (default == do nothing)
    void HDF5Module::create_dma_channels(RuntimeImpl *runtime)
    {
      Module::create_dma_channels(runtime);

      //runtime->add_dma_channel(new HDF5WriteChannel(hdf5mem));
      //runtime->add_dma_channel(new HDF5ReadChannel(hdf5mem));
    }

    // create any code translators provided by the module (default == do nothing)
    void HDF5Module::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void HDF5Module::cleanup(void)
    {
      Module::cleanup();

      herr_t err = H5close();
      if(err < 0)
	log_hdf5.warning() << "unable to close HDF5 library - result = " << err;
#if 0
      // free our allocations here
      for(std::map<int, void *>::iterator it = HDF5_mem_bases.begin();
	  it != HDF5_mem_bases.end();
	  ++it) {
	bool ok = HDF5sysif_free_mem(it->first, it->second,
				     cfg_HDF5_mem_size_in_mb << 20);
	if(!ok)
	  log_hdf5.error() << "failed to free memory in HDF5 node " << it->first << ": ptr=" << it->second;
      }
#endif
    }


  }; // namespace HDF5

}; // namespace Realm
