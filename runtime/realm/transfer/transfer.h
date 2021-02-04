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

// data transfer (a.k.a. dma) engine for Realm

#ifndef REALM_TRANSFER_H
#define REALM_TRANSFER_H

#include "realm/event.h"
#include "realm/memory.h"
#include "realm/indexspace.h"

namespace Realm {

  // the data transfer engine has too much code to have it all be templated on the
  //  type of IndexSpace that is driving the transfer, so we need a widget that
  //  can hold an arbitrary IndexSpace and dispatch based on its type

  class XferDes;
  class AddressList;

  class TransferIterator {
  public:
    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);

    virtual ~TransferIterator(void);

    // must be called (and waited on) before iteration is possible
    virtual Event request_metadata(void);

    // specify the xd port used for indirect address flow control, if any
    virtual void set_indirect_input_port(XferDes *xd, int port_idx,
					 TransferIterator *inner_iter);

    virtual void reset(void) = 0;
    virtual bool done(void) = 0;

    // flag bits to control iterators
    enum {
      SRC_PARTIAL_OK   = (1 << 0),
      SRC_LINES_OK     = (1 << 1),
      SRC_PLANES_OK    = (1 << 2),
      SRC_FLAGMASK     = 0xff,

      DST_PARTIAL_OK   = (1 << 8),
      DST_LINES_OK     = (1 << 9),
      DST_PLANES_OK    = (1 << 10),
      DST_FLAGMASK     = 0xff00,

      PARTIAL_OK       = SRC_PARTIAL_OK | DST_PARTIAL_OK,
      LINES_OK         = SRC_LINES_OK   | DST_LINES_OK,
      PLANES_OK        = SRC_PLANES_OK  | DST_PLANES_OK,
    };

    struct AddressInfo {
      size_t base_offset;
      size_t bytes_per_chunk; // multiple of sizeof(T) unless PARTIAL_OK
      size_t num_lines;   // guaranteed to be 1 unless LINES_OK (i.e. 2D)
      size_t line_stride;
      size_t num_planes;  // guaranteed to be 1 unless PLANES_OK (i.e. 3D)
      size_t plane_stride;
    };

#ifdef REALM_USE_HDF5
    typedef unsigned long long hsize_t;
    struct AddressInfoHDF5 {
      //hid_t dset_id;
      //hid_t dtype_id;
      FieldID field_id;  // used to cache open datasets
      const std::string *filename;
      const std::string *dsetname;
      std::vector<hsize_t> dset_bounds;
      std::vector<hsize_t> offset; // start location in dataset
      std::vector<hsize_t> extent; // xfer dimensions in memory and dataset
    };
#endif

    // if a step is tentative, it must either be confirmed or cancelled before
    //  another one is possible
    virtual size_t step(size_t max_bytes, AddressInfo& info, unsigned flags,
			bool tentative = false) = 0;
#ifdef REALM_USE_HDF5
    virtual size_t step_hdf5(size_t max_bytes, AddressInfoHDF5& info,
			     bool tentative = false);
#endif
    virtual void confirm_step(void) = 0;
    virtual void cancel_step(void) = 0;

    virtual bool get_addresses(AddressList &addrlist) = 0;
  };

  class TransferDomain {
  protected:
    TransferDomain(void);

  public:
    template <typename S>
    static TransferDomain *deserialize_new(S& deserializer);
    
    template <int N, typename T>
    static TransferDomain *construct(const IndexSpace<N,T>& is);

    virtual TransferDomain *clone(void) const = 0;

    virtual ~TransferDomain(void);

    virtual Event request_metadata(void) = 0;

    virtual size_t volume(void) const = 0;

    virtual TransferIterator *create_iterator(RegionInstance inst,
					      RegionInstance peer,
					      const std::vector<FieldID>& fields,
					      const std::vector<size_t>& fld_offsets,
					      const std::vector<size_t>& fld_sizes) const = 0;

    virtual void print(std::ostream& os) const = 0;
  };

}; // namespace Realm

#include "realm/transfer/transfer.inl"

#endif // ifndef REALM_TRANSFER_H
