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

// HDF5-specific instance layouts and accessors

#include "realm/hdf5/hdf5_access.h"
#include "realm/deppart/inst_helper.h"
#include "realm/machine.h"
#include "realm/network.h"
#include "realm/id.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

namespace Realm {

  extern Logger log_hdf5;

  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalHDF5Resource
  //

  ExternalHDF5Resource::ExternalHDF5Resource()
  {}

  ExternalHDF5Resource::ExternalHDF5Resource(const std::string& _filename,
					     bool _read_only)
    : filename(_filename)
    , read_only(_read_only)
  {}

  // returns the suggested memory in which this resource should be created
  Memory ExternalHDF5Resource::suggested_memory() const
  {
    NodeID target_node = Network::my_node_id;

    const char *name = filename.c_str();
    if(!strncmp(name, "rank=", 5)) {
      const char *pos;
      errno = 0;
      long val = strtol(name+5, (char **)&pos, 10);
      if((errno == 0) && (val >= 0) && (val <= Network::max_node_id) && (*pos == ':')) {
	target_node = val;
      } else {
	log_hdf5.warning() << "ill-formed rank prefix in filename: \"" << filename << "\"";
      }
    }

    // look up the HDF5 memory on the target node
    // kinda hacky, but create a proxy processor ID on the target node
    Processor proxy = ID::make_processor(target_node, 0).convert<Processor>();
    Memory memory = Machine::MemoryQuery(Machine::get_machine())
      .same_address_space_as(proxy)
      .only_kind(Memory::HDF_MEM)
      .first();

    return memory;
  }

  ExternalInstanceResource *ExternalHDF5Resource::clone(void) const
  {
    return new ExternalHDF5Resource(filename, read_only);
  }

  void ExternalHDF5Resource::print(std::ostream& os) const
  {
    os << "hdf5(filename='" << filename << "'";
    if(read_only)
      os << ", readonly";
    os << ")";
  }

  /*static*/ Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalHDF5Resource> ExternalHDF5Resource::serdez_subclass;


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance
  //

  template <int N, typename T>
  /*static*/ Event RegionInstance::create_hdf5_instance(RegionInstance& inst,
							const char *file_name,
							const IndexSpace<N,T>& space,
							const std::vector<HDF5FieldInfo<N,T> >& field_infos,
							bool read_only,
							const ProfilingRequestSet& prs,
							Event wait_on /*= Event::NO_EVENT*/)
  {
    // construct an instance layout for the new instance
    InstanceLayout<N,T> *layout = new InstanceLayout<N,T>;
    layout->bytes_used = 0;
    layout->alignment_reqd = 0;  // no allocation being made
    layout->space = space;
    layout->piece_lists.resize(field_infos.size());

    int idx = 0;
    for(typename std::vector<HDF5FieldInfo<N,T> >::const_iterator it = field_infos.begin();
	it != field_infos.end();
	++it) {
      FieldID id = it->field_id;
      InstanceLayoutGeneric::FieldLayout& fl = layout->fields[id];
      fl.list_idx = idx;
      fl.rel_offset = 0;
      fl.size_in_bytes = it->field_size;

      // create a single piece (for non-empty index spaces)
      if(!space.empty()) {
	HDF5LayoutPiece<N,T> *hlp = new HDF5LayoutPiece<N,T>;
	hlp->bounds = space.bounds;
	hlp->dsetname = it->dataset_name;
	for(int j = 0; j < N; j++)
          hlp->offset[j] = it->offset[j];
	for(int j = 0; j < N; j++)
	  hlp->dim_order[j] = it->dim_order[j];
	layout->piece_lists[idx].pieces.push_back(hlp);
      }
      idx++;
    }

    ExternalHDF5Resource res(file_name, read_only);
    return create_external_instance(inst,
				    res.suggested_memory(),
				    layout, res, prs, wait_on);
  }

#define DOIT(N,T) \
  template Event RegionInstance::create_hdf5_instance<N,T>(RegionInstance&, \
							      const char *, \
							      const IndexSpace<N,T>&, \
							      const std::vector<RegionInstance::HDF5FieldInfo<N,T> >&, \
							      bool, \
							      const ProfilingRequestSet&, \
							      Event);
  FOREACH_NT(DOIT)
#undef DOIT

  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, HDF5LayoutPiece<N,T> > HDF5LayoutPiece<N,T>::serdez_subclass;

#define DOIT(N,T) \
  template class HDF5LayoutPiece<N,T>;
  FOREACH_NT(DOIT)
#undef DOIT
  
}; // namespace Realm
