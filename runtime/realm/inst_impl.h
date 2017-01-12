/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// RegionInstance implementations for Realm

#ifndef REALM_INST_IMPL_H
#define REALM_INST_IMPL_H

#include "instance.h"
#include "id.h"

#include "activemsg.h"

#include "rsrv_impl.h"
#include "metadata.h"

namespace Realm {

    class RegionInstanceImpl {
    public:
      RegionInstanceImpl(RegionInstance _me, IndexSpace _is, Memory _memory, off_t _offset, size_t _size, 
			 ReductionOpID _redopid,
			 const DomainLinearization& _linear, size_t _block_size, size_t _elmt_size, 
			 const std::vector<size_t>& _field_sizes,
			 const ProfilingRequestSet &reqs,
			 off_t _count_offset = -1, off_t _red_list_size = -1, 
			 RegionInstance _parent_inst = RegionInstance::NO_INST);

      // when we auto-create a remote instance, we don't know region/offset/linearization
      RegionInstanceImpl(RegionInstance _me, Memory _memory);

      ~RegionInstanceImpl(void);

#ifdef POINTER_CHECKS
      void verify_access(unsigned ptr);
      const ElementMask& get_element_mask(void);
#endif
      void get_bytes(int index, off_t byte_offset, void *dst, size_t size);
      void put_bytes(int index, off_t byte_offset, const void *src, size_t size);

#if 0
      static Event copy(RegionInstance src, 
			RegionInstance target,
			IndexSpace isegion,
			size_t elmt_size,
			size_t bytes_to_copy,
			Event after_copy = Event::NO_EVENT);
#endif

      bool get_strided_parameters(void *&base, size_t &stride,
				  off_t field_offset);

      Event request_metadata(void) { return metadata.request_data(ID(me).instance.owner_node, me.id); }

      void record_instance_usage(void);
      void finalize_instance(void);

    public: //protected:
      friend class RegionInstance;

      RegionInstance me;
      Memory memory; // not part of metadata because it's determined from ID alone
      // Profiling info only needed on creation node
      ProfilingRequestSet requests;
      ProfilingMeasurementCollection measurements;
      ProfilingMeasurements::InstanceTimeline timeline;

      class Metadata : public MetadataBase {
      public:
	void *serialize(size_t& out_size) const;
	void deserialize(const void *in_data, size_t in_size);

	IndexSpace is;
	off_t alloc_offset;
	size_t size;
	ReductionOpID redopid;
	off_t count_offset;
	off_t red_list_size;
	size_t block_size, elmt_size;
	std::vector<size_t> field_sizes;
	RegionInstance parent_inst;
	DomainLinearization linearization;
      };

      Metadata metadata;

      static const unsigned MAX_LINEARIZATION_LEN = 32;

      ReservationImpl lock;
    };

    // helper function to figure out which field we're in
    void find_field_start(const std::vector<size_t>& field_sizes, off_t byte_offset,
			  size_t size, off_t& field_start, int& field_size);
    
}; // namespace Realm

#endif // ifndef REALM_INST_IMPL_H
