/* Copyright 2019 Stanford University, NVIDIA Corporation
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

namespace Realm {

  template <int N, typename T>
  /*static*/ Event RegionInstance::create_hdf5_instance(RegionInstance& inst,
							const char *file_name,
							const IndexSpace<N,T>& space,
							const std::vector<RegionInstance::HDF5FieldInfo<N,T> >& field_infos,
							bool read_only,
							const ProfilingRequestSet& prs,
							Event wait_on /*= Event::NO_EVENT*/)
  {
    // look up the local HDF5 memory
    Memory memory = Machine::MemoryQuery(Machine::get_machine())
      .local_address_space()
      .only_kind(Memory::HDF_MEM)
      .first();
    assert(memory.exists());
    
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
	hlp->filename = file_name;
	hlp->dsetname = it->dataset_name;
	hlp->offset = it->offset;
	for(int j = 0; j < N; j++)
	  hlp->dim_order[j] = it->dim_order[j];
	hlp->read_only = read_only;
	layout->piece_lists[idx].pieces.push_back(hlp);
      }
      idx++;
    }

    // and now create the instance using this layout
    return create_instance(inst, memory, layout, prs, wait_on);
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

}; // namespace Realm
