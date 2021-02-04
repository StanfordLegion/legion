/* Copyright 2021 Stanford University, NVIDIA Corporation
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

    class HDF5Dataset {
    public:
      static HDF5Dataset *open(const char *filename,
			       const char *dsetname,
			       bool read_only);
      void flush();
      void close();

    protected:
      HDF5Dataset();
      ~HDF5Dataset();

    public:
      hid_t file_id, dset_id, dtype_id;
      int ndims;
      static const int MAX_DIM = 16;
      hsize_t dset_size[MAX_DIM];
      bool read_only;
    };

    class HDF5Memory : public MemoryImpl {
    public:
      static const size_t ALIGNMENT = 256;

      HDF5Memory(Memory _me);

      virtual ~HDF5Memory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
							  bool need_alloc_result,
							  bool poisoned,
							  TimeLimit work_until);

      virtual void release_storage_immediate(RegionInstanceImpl *inst,
					     bool poisoned,
					     TimeLimit work_until);
    };

  }; // namespace HDF5

}; // namespace Realm

#endif
