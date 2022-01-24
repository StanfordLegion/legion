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

#include "realm/hdf5/hdf5_module.h"

#include "realm/hdf5/hdf5_internal.h"
#include "realm/hdf5/hdf5_access.h"

#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"
#include "realm/inst_impl.h"

#include <map>

namespace Realm {

  Logger log_hdf5("hdf5");


  namespace HDF5 {

    static HDF5Module *hdf5mod = 0;

    namespace Config {
      size_t max_open_files = 0;
      bool force_read_write = false;
    };
    
    struct HDF5OpenFile {
      hid_t file_id;
      int usage_count;
    };

    // files are indexed by filename and writeable-ness
    typedef std::map<std::pair<std::string, bool>, HDF5OpenFile> HDF5FileCache;
    HDF5FileCache file_cache;

    
    ////////////////////////////////////////////////////////////////////////
    //
    // class HDF5Dataset

    HDF5Dataset::HDF5Dataset()
    {}

    HDF5Dataset::~HDF5Dataset()
    {}

    /*static*/ HDF5Dataset *HDF5Dataset::open(const char *filename,
					      const char *dsetname,
					      bool read_only)
    {
      // strip off any filename prefix starting with a colon (e.g. rank=nn:)
      {
        const char *pos = strchr(filename, ':');
        if(pos) filename = pos + 1;
      }

      // find or open the file
      bool open_as_rw = !read_only || Config::force_read_write;
      std::pair<std::string, bool> key(filename, open_as_rw);
      HDF5FileCache::iterator it = file_cache.find(key);
      if(it == file_cache.end()) {
	struct HDF5OpenFile f;
	CHECK_HDF5( f.file_id = H5Fopen(filename,
					(open_as_rw ? H5F_ACC_RDWR :
					              H5F_ACC_RDONLY),
					H5P_DEFAULT) );
	log_hdf5.info() << "H5Fopen(\"" << filename << "\", "
			<< (open_as_rw ? "H5F_ACC_RDWR" :
			                 "H5F_ACC_RDONLY")
			<< ") = " << f.file_id;
	if(f.file_id < 0) return 0;
	f.usage_count = 0;
	it = file_cache.insert(std::make_pair(key, f)).first;
      }

      // open dataset within file, following group path if any /'s are present
      hid_t loc_id = it->second.file_id;
      // leading slash in dataset path is optional - ignore if present
      const char *curpos = dsetname;
      if(*curpos == '/') curpos++;
      while(true) {
	const char *pos = strchr(curpos, '/');
	if(!pos) break;
	char grpname[256];
	size_t len = pos-curpos;
	assert(len < 256);
	strncpy(grpname, curpos, len);
	grpname[len] = 0;
	hid_t grp_id;
	CHECK_HDF5( grp_id = H5Gopen2(loc_id, grpname, H5P_DEFAULT) );
	log_hdf5.info() << "H5Gopen2(" << loc_id << ", \"" << grpname << "\") = " << grp_id;
	if(loc_id != it->second.file_id)
	  CHECK_HDF5( H5Gclose(loc_id) );
	if(grp_id < 0)
	  return 0;
	loc_id = grp_id;
	curpos = pos + 1;
      }
      hid_t dset_id;
      CHECK_HDF5( dset_id = H5Dopen2(loc_id, curpos, H5P_DEFAULT) );
      log_hdf5.info() << "H5Dopen2(" << it->second.file_id << ", \"" << dsetname << "\") = " << dset_id;
      if(loc_id != it->second.file_id)
	CHECK_HDF5( H5Gclose(loc_id) );
      if(dset_id < 0)
	return 0;

      // get and cache the datatype
      hid_t dtype_id;
      CHECK_HDF5( dtype_id = H5Dget_type(dset_id) );

      // get and cache the bounds, checking that the dimension matches
      hid_t dspace_id;
      CHECK_HDF5( dspace_id = H5Dget_space(dset_id) );
      int ndims = H5Sget_simple_extent_ndims(dspace_id);
      if((ndims < 0) || (ndims > MAX_DIM)) {
	log_hdf5.error() << "dataset dimension out of range: file=" << filename
			 << " dset=" << dsetname << " dim=" << ndims;
	CHECK_HDF5( H5Sclose(dspace_id) );
	CHECK_HDF5( H5Tclose(dtype_id) );
	CHECK_HDF5( H5Dclose(dset_id) );
	return 0;
      }

      HDF5Dataset *dset = new HDF5Dataset;
      dset->file_id = it->second.file_id;
      dset->dset_id = dset_id;
      dset->dtype_id = dtype_id;
      dset->dspace_id = dspace_id;
      dset->read_only = read_only;
      dset->ndims = ndims;
      // since HDF5 supports growable datasets, we care about the maxdims
      CHECK_HDF5( H5Sget_simple_extent_dims(dspace_id, 0, dset->dset_size) );

      // increment the usage count on the file
      it->second.usage_count++;
      return dset;
    }

    void HDF5Dataset::flush()
    {
      CHECK_HDF5( H5Fflush(file_id, H5F_SCOPE_GLOBAL) );
    }

    void HDF5Dataset::close()
    {
      // find our file in the cache
      HDF5FileCache::iterator it = file_cache.begin();
      while((it != file_cache.end()) && (it->second.file_id != file_id)) ++it;
      assert(it != file_cache.end());

      log_hdf5.info() << "H5Dclose(" << dset_id << ")";
      CHECK_HDF5( H5Sclose(dspace_id) );
      CHECK_HDF5( H5Tclose(dtype_id) );
      CHECK_HDF5( H5Dclose(dset_id) );

      // decrement usage count of file and consider closing it
      it->second.usage_count--;
      if((it->second.usage_count == 0) &&
	 (file_cache.size() > Config::max_open_files)) {
	log_hdf5.info() << "H5Fclose(" << it->second.file_id << ")";
	CHECK_HDF5( H5Fclose(it->second.file_id) );
	file_cache.erase(it);
      } else {
	// if we're not going to close it, but we did writes, a flush is good
	CHECK_HDF5( H5Fflush(it->second.file_id, H5F_SCOPE_GLOBAL) );
      }

      // done with this object now
      delete this;
    }


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

	cp.add_option_bool("-hdf5:showerrors", m->cfg_showerrors)
	  .add_option_int("-hdf5:openfiles", Config::max_open_files)
	  .add_option_bool("-hdf5:forcerw", Config::force_read_write);
	
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

      runtime->add_dma_channel(new HDF5Channel(&runtime->bgwork));
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

      // close any files left open in the cache
      for(HDF5FileCache::iterator it = file_cache.begin();
	  it != file_cache.end();
	  ++it) {
	if(it->second.usage_count > 0)
	  log_hdf5.warning() << "nonzero usage count on file \"" << it->first.first << "\": " << it->second.usage_count;
	log_hdf5.info() << "H5Fclose(" << it->second.file_id << ")";
	CHECK_HDF5( H5Fclose(it->second.file_id) );
      }
      file_cache.clear();
	
      herr_t err = H5close();
      if(err < 0)
	log_hdf5.warning() << "unable to close HDF5 library - result = " << err;
    }


  }; // namespace HDF5

}; // namespace Realm
