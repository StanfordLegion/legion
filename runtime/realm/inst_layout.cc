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

// Layout descriptors for Realm RegionInstances

#include "realm/inst_layout.h"
#include "realm/deppart/inst_helper.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayoutConstraints

  InstanceLayoutConstraints::InstanceLayoutConstraints(const std::map<FieldID, size_t>& field_sizes,
						       size_t block_size)
  {
    // use the field sizes to generate "offsets" as unique IDs
    switch(block_size) {
    case 0:
      {
	// SOA - each field is its own "group"
	field_groups.resize(field_sizes.size());
	size_t i = 0;
	for(std::map<FieldID, size_t>::const_iterator it = field_sizes.begin();
	    it != field_sizes.end();
	    ++it, ++i) {
	  field_groups[i].resize(1);
	  field_groups[i][0].field_id = it->first;
	  field_groups[i][0].fixed_offset = false;
	  field_groups[i][0].offset = 0;
	  field_groups[i][0].size = it->second;
	  field_groups[i][0].alignment = it->second; // natural alignment 
	}
	break;
      }

    case 1:
      {
	// AOS - all field_groups in same group
	field_groups.resize(1);
	field_groups[0].resize(field_sizes.size());
	size_t i = 0;
	for(std::map<FieldID, size_t>::const_iterator it = field_sizes.begin();
	    it != field_sizes.end();
	    ++it, ++i) {
	  field_groups[0][i].field_id = it->first;
	  field_groups[0][i].fixed_offset = false;
	  field_groups[0][i].offset = 0;
	  field_groups[0][i].size = it->second;
	  field_groups[0][i].alignment = it->second; // natural alignment 
	}
	break;
      }

    default:
      {
	// hybrid - blech
	assert(0);
      }
    }
  }

  InstanceLayoutConstraints::InstanceLayoutConstraints(const std::vector<FieldID>& field_ids,
						       const std::vector<size_t>& field_sizes,
						       size_t block_size)
  {
    switch(block_size) {
    case 0:
      {
	// SOA - each field is its own "group"
	field_groups.resize(field_sizes.size());
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  field_groups[i].resize(1);
	  field_groups[i][0].field_id = field_ids[i];
	  field_groups[i][0].fixed_offset = false;
	  field_groups[i][0].offset = 0;
	  field_groups[i][0].size = field_sizes[i];
	  field_groups[i][0].alignment = field_sizes[i]; // natural alignment 
	}
	break;
      }

    case 1:
      {
	// AOS - all field_groups in same group
	field_groups.resize(1);
	field_groups[0].resize(field_sizes.size());
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  field_groups[0][i].field_id = field_ids[i];
	  field_groups[0][i].fixed_offset = false;
	  field_groups[0][i].offset = 0;
	  field_groups[0][i].size = field_sizes[i];
	  field_groups[0][i].alignment = field_sizes[i]; // natural alignment 
	}
	break;
      }

    default:
      {
	// hybrid - blech
	assert(0);
      }
    }
  }

  InstanceLayoutConstraints::InstanceLayoutConstraints(const std::vector<size_t>& field_sizes,
						       size_t block_size)
  {
    // use the field sizes to generate "offsets" as unique IDs
    switch(block_size) {
    case 0:
      {
	// SOA - each field is its own "group"
	field_groups.resize(field_sizes.size());
	size_t offset = 0;
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  field_groups[i].resize(1);
	  field_groups[i][0].field_id = FieldID(offset);
	  field_groups[i][0].fixed_offset = false;
	  field_groups[i][0].offset = 0;
	  field_groups[i][0].size = field_sizes[i];
	  field_groups[i][0].alignment = field_sizes[i]; // natural alignment 
	  offset += field_sizes[i];
	}
	break;
      }

    case 1:
      {
	// AOS - all field_groups in same group
	field_groups.resize(1);
	field_groups[0].resize(field_sizes.size());
	size_t offset = 0;
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  field_groups[0][i].field_id = FieldID(offset);
	  field_groups[0][i].fixed_offset = false;
	  field_groups[0][i].offset = 0;
	  field_groups[0][i].size = field_sizes[i];
	  field_groups[0][i].alignment = field_sizes[i]; // natural alignment 
	  offset += field_sizes[i];
	}
	break;
      }

    default:
      {
	// hybrid - blech
	assert(0);
      }
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayout<N,T>

  // this currently lives in inst_impl.cc - would be nice to move it back here
#if 0
  template <int N, typename T>
  /*static*/ Serialization::PolymorphicSerdezSubclass<InstanceLayoutGeneric, InstanceLayout<N,T> > InstanceLayout<N,T>::serdez_subclass;

#define DOIT(N,T) \
  template class InstanceLayout<N,T>;
  FOREACH_NT(DOIT)
#endif

}; // namespace Realm

