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

#ifndef REALM_HDF5_INTERNAL_H
#define REALM_HDF5_INTERNAL_H

#include "realm/mem_impl.h"

#include "realm/transfer/lowlevel_dma.h"

#include <hdf5.h>

#define CHECK_HDF5(cmd) \
  do { \
    herr_t res = (cmd); \
    if(res < 0) { \
      fprintf(stderr, "HDF5 error on %s:\n", #cmd); \
      H5Eprint2(H5E_DEFAULT, stderr); \
      assert(0); \
    } \
  } while(0)

namespace Realm {

  namespace HDF5 {

    class HDF5Memory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      HDF5Memory(Memory _me);

      virtual ~HDF5Memory(void);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);
      virtual int get_home_node(off_t offset, size_t size);

    public:
      struct HDFMetadata {
        int lo[3];
        hsize_t dims[3];
        int ndims;
        hid_t file_id;
	std::map<size_t, hid_t> dataset_ids;
	std::map<size_t, hid_t> datatype_ids;
      };
      std::map<RegionInstance, HDFMetadata *> hdf_metadata;
    };

    class HDF5WriteChannel : public MemPairCopierFactory {
    public:
      HDF5WriteChannel(HDF5Memory *_mem);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);
#endif

    protected:
      HDF5Memory *mem;
    };

    class HDF5ReadChannel : public MemPairCopierFactory {
    public:
      HDF5ReadChannel(HDF5Memory *_mem);

      virtual bool can_perform_copy(Memory src_mem, Memory dst_mem,
				    ReductionOpID redop_id, bool fold);

#ifdef OLD_COPIERS
      virtual MemPairCopier *create_copier(Memory src_mem, Memory dst_mem,
					   ReductionOpID redop_id, bool fold);
#endif

    protected:
      HDF5Memory *mem;
    };

#ifdef OLD_COPIERS
    class HDF5WriteCopier : public MemPairCopier {
    public:
      HDF5WriteCopier(MemoryImpl *_src_impl, HDF5Memory *_mem);
      virtual ~HDF5WriteCopier(void);

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					OASVec &oas_vec);

      // used by HDF5InstPairCopier<HDF5WriteCopier>
      void transfer_data(hid_t dset_id, hid_t dtype_id,
			 hid_t mspace_id, hid_t dspace_id,
			 void *data_ptr);

    protected:
      MemoryImpl *src_impl;
      HDF5Memory *mem;
    };

    class HDF5ReadCopier : public MemPairCopier {
    public:
      HDF5ReadCopier(HDF5Memory *_mem, MemoryImpl *_dst_impl);
      virtual ~HDF5ReadCopier(void);

      virtual InstPairCopier *inst_pair(RegionInstance src_inst, RegionInstance dst_inst,
					OASVec &oas_vec);

      // used by HDF5InstPairCopier<HDF5ReadCopier>
      void transfer_data(hid_t dset_id, hid_t dtype_id,
			 hid_t mspace_id, hid_t dspace_id,
			 void *data_ptr);

    protected:
      HDF5Memory *mem;
      MemoryImpl *dst_impl;
    };

    template <typename T>
    class HDF5InstPairCopier : public InstPairCopier {
    public:
      HDF5InstPairCopier(T *_mpc, RegionInstanceImpl *_local_impl, HDF5Memory::HDFMetadata *_md,
			 OASVec& _oas_vec);
      virtual ~HDF5InstPairCopier(void);
    public:
      virtual bool copy_all_fields(Domain d);

      virtual void copy_field(off_t src_index, off_t dst_index, off_t elem_count,
                              unsigned offset_index);

      virtual void copy_all_fields(off_t src_index, off_t dst_index, off_t elem_count);

      virtual void flush(void);

    protected:
      T *mpc;
      RegionInstanceImpl *local_impl;
      HDF5Memory::HDFMetadata *md;
      OASVec& oas_vec;
    };
#endif

  }; // namespace HDF5

}; // namespace Realm

#endif
