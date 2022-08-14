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

#ifndef REALM_HDF5_INTERNAL_H
#define REALM_HDF5_INTERNAL_H

#include "realm/mem_impl.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"

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
      hid_t file_id, dset_id, dtype_id, dspace_id;
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

      // HDF5Memory supports ExternalHDF5Resource
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t& inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

    };

    class HDF5Request : public Request {
    public:
      void *mem_base; // could be source or dest
      hid_t dataset_id, datatype_id;
      hid_t mem_space_id, file_space_id;
    };
    class HDF5Channel;

    class AddressInfoHDF5 : public TransferIterator::AddressInfoCustom {
    public:
      virtual int set_rect(const RegionInstanceImpl *inst,
                           const InstanceLayoutPieceBase *piece,
                           size_t field_size, size_t field_offset,
                           int ndims,
                           const int64_t lo[/*ndims*/],
                           const int64_t hi[/*ndims*/],
                           const int order[/*ndims*/]);

      //hid_t dset_id;
      //hid_t dtype_id;
      //FieldID field_id;  // used to cache open datasets
      const std::string *filename;
      const std::string *dsetname;
      //std::vector<hsize_t> dset_bounds;
      std::vector<hsize_t> offset; // start location in dataset
      std::vector<hsize_t> extent; // xfer dimensions in memory and dataset
    };

    class HDF5XferDes : public XferDes {
    public:
      HDF5XferDes(uintptr_t _dma_op, Channel *_channel,
		  NodeID _launch_node, XferDesID _guid,
		  const std::vector<XferDesPortInfo>& inputs_info,
		  const std::vector<XferDesPortInfo>& outputs_info,
		  int _priority,
                  const void *_fill_data, size_t _fill_size);

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

      virtual bool request_available();
      virtual Request* dequeue_request();
      virtual void enqueue_request(Request* req);

      bool progress_xd(HDF5Channel *channel, TimeLimit work_until);

    private:
      bool req_in_use;
      HDF5Request hdf5_req;
      typedef std::pair<const std::string *, const std::string *> DatasetMapKey;
      typedef std::map<DatasetMapKey, HDF5Dataset *> DatasetMap;
      DatasetMap datasets;
      static const size_t MAX_FILL_SIZE_IN_BYTES = 65536;
    };

    // single channel handles both HDF5 reads and writes
    class HDF5Channel : public SingleXDQChannel<HDF5Channel, HDF5XferDes> {
    public:
      HDF5Channel(BackgroundWorkManager *bgwork);
      ~HDF5Channel();

      // handle HDF5 requests in order - no concurrency
      static const bool is_ordered = true;

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);
    };

  }; // namespace HDF5

}; // namespace Realm

#endif
