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

// data transfer (a.k.a. dma) engine for Realm

#ifndef REALM_TRANSFER_H
#define REALM_TRANSFER_H

#include "realm/event.h"
#include "realm/memory.h"
#include "realm/indexspace.h"

#ifdef USE_HDF
#include "realm/hdf5/hdf5_internal.h"
#endif

namespace Realm {

  // the data transfer engine has too much code to have it all be templated on the
  //  type of IndexSpace that is driving the transfer, so we need a widget that
  //  can hold an arbitrary IndexSpace and dispatch based on its type

  class TransferIterator {
  public:
    template <typename S>
    static TransferIterator *deserialize_new(S& deserializer);

    virtual ~TransferIterator(void);

    // must be called (and waited on) before iteration is possible
    virtual Event request_metadata(void);

    virtual void reset(void) = 0;
    virtual bool done(void) = 0;

    // flag bits to control iterators
    enum {
      PARTIAL_OK   = (1 << 0),
      LINES_OK     = (1 << 1),
      PLANES_OK    = (1 << 2),
    };

    struct AddressInfo {
      size_t base_offset;
      size_t bytes_per_chunk; // multiple of sizeof(T) unless PARTIAL_OK
      size_t num_lines;   // guaranteed to be 1 unless LINES_OK (i.e. 2D)
      size_t line_stride;
      size_t num_planes;  // guaranteed to be 1 unless PLANES_OK (i.e. 3D)
      size_t plane_stride;
    };

#ifdef USE_HDF
    struct AddressInfoHDF5 {
      //hid_t dset_id;
      //hid_t dtype_id;
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
#ifdef USE_HDF
    virtual size_t step(size_t max_bytes, AddressInfoHDF5& info,
			bool tentative = false);
#endif
    virtual void confirm_step(void) = 0;
    virtual void cancel_step(void) = 0;
  };

  template <typename S>
  inline bool serialize(S& serializer, const TransferIterator& ti)
  {
    return Serialization::PolymorphicSerdezHelper<TransferIterator>::serialize(serializer, ti);
  }

  template <typename S>
  /*static*/ inline TransferIterator *TransferIterator::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<TransferIterator>::deserialize_new(deserializer);
  }

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
					      const std::vector<FieldID>& fields) const = 0;

    virtual void print(std::ostream& os) const = 0;
  };

  inline std::ostream& operator<<(std::ostream& os, const TransferDomain& td)
  {
    td.print(os); return os;
  }

  template <typename S>
  inline bool serialize(S& serializer, const TransferDomain& ci)
  {
    return Serialization::PolymorphicSerdezHelper<TransferDomain>::serialize(serializer, ci);
  }

  template <typename S>
  /*static*/ inline TransferDomain *TransferDomain::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<TransferDomain>::deserialize_new(deserializer);
  }

  class TransferPlan {
  protected:
    // subclasses constructed in plan_* calls below
    TransferPlan(void);

  public:
    virtual ~TransferPlan(void);

    static bool plan_copy(std::vector<TransferPlan *>& plans,
			  const std::vector<CopySrcDstField> &srcs,
			  const std::vector<CopySrcDstField> &dsts,
			  ReductionOpID redop_id = 0, bool red_fold = false);

    static bool plan_fill(std::vector<TransferPlan *>& plans,
			  const std::vector<CopySrcDstField> &dsts,
			  const void *fill_value, size_t fill_value_size);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority) = 0;
  };

}; // namespace Realm

#endif // ifndef REALM_TRANSFER_H
