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

// data transfer (a.k.a. dma) engine for Realm

// nop, but useful for IDEs
#include "realm/transfer/transfer.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIterator
  //

  template <typename S>
  bool serialize(S& serializer, const TransferIterator& ti)
  {
    return Serialization::PolymorphicSerdezHelper<TransferIterator>::serialize(serializer, ti);
  }

  template <typename S>
  /*static*/ TransferIterator *TransferIterator::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<TransferIterator>::deserialize_new(deserializer);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDomain
  //

  inline std::ostream& operator<<(std::ostream& os, const TransferDomain& td)
  {
    td.print(os);
    return os;
  }

  template <typename S>
  bool serialize(S& serializer, const TransferDomain& ci)
  {
    return Serialization::PolymorphicSerdezHelper<TransferDomain>::serialize(serializer, ci);
  }

  template <typename S>
  /*static*/ TransferDomain *TransferDomain::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<TransferDomain>::deserialize_new(deserializer);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct XferDesPortInfo
  //

  template <typename S>
  inline bool serialize(S& s, const XferDesPortInfo& i)
  {
    return ((s << i.port_type) &&
	    (s << i.peer_guid) &&
	    (s << i.peer_port_idx) &&
	    (s << i.indirect_port_idx) &&
	    (s << i.mem) &&
	    (s << i.inst) &&
	    (s << i.ib_offset) &&
	    (s << i.ib_size) &&
	    (s << *i.iter) &&
	    (s << i.serdez_id));
  }

  template <typename S>
  inline bool deserialize(S& s, XferDesPortInfo& i)
  {
    if(!((s >> i.port_type) &&
	 (s >> i.peer_guid) &&
	 (s >> i.peer_port_idx) &&
	 (s >> i.indirect_port_idx) &&
	 (s >> i.mem) &&
	 (s >> i.inst) &&
	 (s >> i.ib_offset) &&
	 (s >> i.ib_size)))
      return false;
    i.iter = TransferIterator::deserialize_new(s);
    if(!i.iter) return false;
    if(!((s >> i.serdez_id)))
      return false;
    return true;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferGraph::XDTemplate
  //

  /*static*/ inline TransferGraph::XDTemplate::IO TransferGraph::XDTemplate::mk_inst(RegionInstance _inst,
										     unsigned _fld_start, unsigned _fld_count)
  {
    IO io;
    io.iotype = IO_INST;
    io.inst.inst = _inst;
    io.inst.fld_start = _fld_start;
    io.inst.fld_count = _fld_count;
    return io;
  }

  /*static*/ inline TransferGraph::XDTemplate::IO TransferGraph::XDTemplate::mk_indirect(unsigned _ind_idx, unsigned _port,
											 RegionInstance _inst,
											 unsigned _fld_start, unsigned _fld_count)
  {
    IO io;
    io.iotype = IO_INDIRECT_INST;
    io.indirect.ind_idx = _ind_idx;
    io.indirect.port = _port;
    io.indirect.inst = _inst;
    io.indirect.fld_start = _fld_start;
    io.indirect.fld_count = _fld_count;
    return io;
  }
  
  /*static*/ inline TransferGraph::XDTemplate::IO TransferGraph::XDTemplate::mk_edge(unsigned _edge)
  {
    IO io;
    io.iotype = IO_EDGE;
    io.edge = _edge;
    return io;
  }
  
  /*static*/ inline TransferGraph::XDTemplate::IO TransferGraph::XDTemplate::mk_fill(unsigned _fill_start, unsigned _fill_size, size_t _fill_total)
  {
    IO io;
    io.iotype = IO_FILL_DATA;
    io.fill.fill_start = _fill_start;
    io.fill.fill_size = _fill_size;
    io.fill.fill_total = _fill_total;
    return io;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDesc
  //

  template <int N, typename T>
  TransferDesc::TransferDesc(IndexSpace<N,T> _is,
			     const std::vector<CopySrcDstField> &_srcs,
			     const std::vector<CopySrcDstField> &_dsts,
			     const std::vector<const typename CopyIndirection<N,T>::Base *> &_indirects,
			     const ProfilingRequestSet &requests)
    : refcount(1)
    , deferred_analysis(this)
    , srcs(_srcs)
    , dsts(_dsts)
    , prs(requests)
    , analysis_complete(false)
    , fill_data(0)
    , fill_size(0)
  {
    domain = TransferDomain::construct(_is);

    indirects.resize(_indirects.size());
    for(size_t i = 0; i < _indirects.size(); i++)
      indirects[i] = _indirects[i]->create_info(_is);

    check_analysis_preconditions();
  }

  inline void TransferDesc::add_reference()
  {
    refcount.fetch_add_acqrel(1);
  }

  inline void TransferDesc::remove_reference()
  {
    int prev = refcount.fetch_sub_acqrel(1);
    if(prev == 1)
      delete this;
  }


}; // namespace Realm
