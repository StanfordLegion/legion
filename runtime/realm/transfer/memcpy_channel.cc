/* Copyright 2024 Stanford University
 * Copyright 2024 Los Alamos National Laboratory
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

#include "realm/realm_config.h"

#ifdef REALM_ON_WINDOWS
#define NOMINMAX
#endif

#include "realm/transfer/channel_common.h"
#include "realm/transfer/memcpy_channel.h"
#include "realm/transfer/transfer.h"
#include "realm/utils.h"

#include <algorithm>

TYPE_IS_SERIALIZABLE(Realm::XferDesKind);

namespace Realm {

  extern Logger log_new_dma;
  extern Logger log_request;
  extern Logger log_xd;
  extern Logger log_xd_ref;

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemcpyXferDes
  //

  MemcpyXferDes::MemcpyXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                               XferDesID _guid,
                               const std::vector<XferDesPortInfo> &inputs_info,
                               const std::vector<XferDesPortInfo> &outputs_info,
                               int _priority)
    : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
              _priority, 0, 0)
    , memcpy_req_in_use(false)
  {
    kind = XFER_MEM_CPY;

    // scan input and output ports to see if any use serdez ops
    has_serdez = false;
    for(size_t i = 0; i < inputs_info.size(); i++) {
      if(inputs_info[i].serdez_id != 0) {
        has_serdez = true;
      }
    }
    for(size_t i = 0; i < outputs_info.size(); i++) {
      if(outputs_info[i].serdez_id != 0) {
        has_serdez = true;
      }
    }

    // ignore requested max_nr and always use 1
    memcpy_req.xd = this;
  }

  long MemcpyXferDes::get_requests(Request **requests, long nr)
  {
    MemcpyRequest **reqs = (MemcpyRequest **)requests;
    // allow 2D and 3D copies
    unsigned flags = (TransferIterator::LINES_OK | TransferIterator::PLANES_OK);
    long new_nr = default_get_requests(requests, nr, flags);
    for(long i = 0; i < new_nr; i++) {
      bool src_is_serdez = (input_ports[reqs[i]->src_port_idx].serdez_op != 0);
      bool dst_is_serdez = (output_ports[reqs[i]->dst_port_idx].serdez_op != 0);
      if(!src_is_serdez && dst_is_serdez) {
        // source offset is determined later - not safe to call get_direct_ptr now
        reqs[i]->src_base = 0;
      } else {
        reqs[i]->src_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(
            reqs[i]->src_off, reqs[i]->nbytes);
        assert(reqs[i]->src_base != 0);
      }
      if(src_is_serdez && !dst_is_serdez) {
        // dest offset is determined later - not safe to call get_direct_ptr now
        reqs[i]->dst_base = 0;
      } else {
        reqs[i]->dst_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(
            reqs[i]->dst_off, reqs[i]->nbytes);
        assert(reqs[i]->dst_base != 0);
      }
    }
    return new_nr;
  }

  void MemcpyXferDes::notify_request_read_done(Request *req)
  {
    default_notify_request_read_done(req);
  }

  void MemcpyXferDes::notify_request_write_done(Request *req)
  {
    default_notify_request_write_done(req);
  }

  void MemcpyXferDes::flush() {}

  bool MemcpyXferDes::request_available() { return !memcpy_req_in_use; }

  Request *MemcpyXferDes::dequeue_request()
  {
    assert(!memcpy_req_in_use);
    memcpy_req_in_use = true;
    memcpy_req.is_read_done = false;
    memcpy_req.is_write_done = false;
    // memcpy request is handled in-thread, so no need to mess with refcount
    return &memcpy_req;
  }

  void MemcpyXferDes::enqueue_request(Request *req)
  {
    assert(memcpy_req_in_use);
    assert(req == &memcpy_req);
    memcpy_req_in_use = false;
  }

  bool MemcpyXferDes::progress_xd(MemcpyChannel *channel, TimeLimit work_until)
  {
    if(has_serdez) {
      Request *rq;
      bool did_work = false;
      do {
        long count = get_requests(&rq, 1);
        if(count > 0) {
          channel->submit(&rq, count);
          did_work = true;
        } else {
          break;
        }
      } while(!work_until.is_expired());

      return did_work;
    }

    // fast path - assumes no serdez
    bool did_work = false;
    ReadSequenceCache rseqcache(this, 2 << 20); // flush after 2MB
    WriteSequenceCache wseqcache(this, 2 << 20);

    while(true) {
      size_t min_xfer_size = 4096; // TODO: make controllable
      size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);

      if(max_bytes == 0) {
        break;
      }

      XferPort *in_port = 0, *out_port = 0;
      size_t in_span_start = 0, out_span_start = 0;
      if(input_control.current_io_port >= 0) {
        in_port = &input_ports[input_control.current_io_port];
        in_span_start = in_port->local_bytes_total;
      }
      if(output_control.current_io_port >= 0) {
        out_port = &output_ports[output_control.current_io_port];
        out_span_start = out_port->local_bytes_total;
      }

      size_t total_bytes = 0;
      if(in_port != 0) {
        if(out_port != 0) {
          // input and output both exist - transfer what we can
          log_xd.info() << "memcpy chunk: min=" << min_xfer_size << " max=" << max_bytes;

          uintptr_t in_base =
              reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
          uintptr_t out_base =
              reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

          while(total_bytes < max_bytes) {
            AddressListCursor &in_alc = in_port->addrcursor;
            AddressListCursor &out_alc = out_port->addrcursor;

            uintptr_t in_offset = in_alc.get_offset();
            uintptr_t out_offset = out_alc.get_offset();

            // the reported dim is reduced for partially consumed address
            //  ranges - whatever we get can be assumed to be regular
            int in_dim = in_alc.get_dim();
            int out_dim = out_alc.get_dim();

            size_t bytes = 0;
            size_t bytes_left = max_bytes - total_bytes;
            // memcpys don't need to be particularly big to achieve
            //  peak efficiency, so trim to something that takes
            //  10's of us to be responsive to the time limit
            bytes_left = std::min(bytes_left, size_t(256 << 10));

            if(in_dim > 0) {
              if(out_dim > 0) {
                size_t icount = in_alc.remaining(0);
                size_t ocount = out_alc.remaining(0);

                // contig bytes is always the min of the first dimensions
                size_t contig_bytes = std::min(std::min(icount, ocount), bytes_left);

                // catch simple 1D case first
                if((contig_bytes == bytes_left) ||
                   ((contig_bytes == icount) && (in_dim == 1)) ||
                   ((contig_bytes == ocount) && (out_dim == 1))) {
                  bytes = contig_bytes;
                  memcpy_1d(out_base + out_offset, in_base + in_offset, bytes);
                  in_alc.advance(0, bytes);
                  out_alc.advance(0, bytes);
                } else {
                  // grow to a 2D copy
                  int id;
                  int iscale;
                  uintptr_t in_lstride;
                  if(contig_bytes < icount) {
                    // second input dim comes from splitting first
                    id = 0;
                    in_lstride = contig_bytes;
                    size_t ilines = icount / contig_bytes;
                    if((ilines * contig_bytes) != icount) {
                      in_dim = 1; // leftover means we can't go beyond this
                    }
                    icount = ilines;
                    iscale = contig_bytes;
                  } else {
                    assert(in_dim > 1);
                    id = 1;
                    icount = in_alc.remaining(id);
                    in_lstride = in_alc.get_stride(id);
                    iscale = 1;
                  }

                  int od;
                  int oscale;
                  uintptr_t out_lstride;
                  if(contig_bytes < ocount) {
                    // second output dim comes from splitting first
                    od = 0;
                    out_lstride = contig_bytes;
                    size_t olines = ocount / contig_bytes;
                    if((olines * contig_bytes) != ocount) {
                      out_dim = 1; // leftover means we can't go beyond this
                    }
                    ocount = olines;
                    oscale = contig_bytes;
                  } else {
                    assert(out_dim > 1);
                    od = 1;
                    ocount = out_alc.remaining(od);
                    out_lstride = out_alc.get_stride(od);
                    oscale = 1;
                  }

                  size_t lines =
                      std::min(std::min(icount, ocount), bytes_left / contig_bytes);

                  // see if we need to stop at 2D
                  if(((contig_bytes * lines) == bytes_left) ||
                     ((lines == icount) && (id == (in_dim - 1))) ||
                     ((lines == ocount) && (od == (out_dim - 1)))) {
                    bytes = contig_bytes * lines;
                    memcpy_2d(out_base + out_offset, out_lstride, in_base + in_offset,
                              in_lstride, contig_bytes, lines);
                    in_alc.advance(id, lines * iscale);
                    out_alc.advance(od, lines * oscale);
                  } else {
                    uintptr_t in_pstride;
                    if(lines < icount) {
                      // third input dim comes from splitting current
                      in_pstride = in_lstride * lines;
                      size_t iplanes = icount / lines;
                      // check for leftovers here if we go beyond 3D!
                      icount = iplanes;
                      iscale *= lines;
                    } else {
                      id++;
                      assert(in_dim > id);
                      icount = in_alc.remaining(id);
                      in_pstride = in_alc.get_stride(id);
                      iscale = 1;
                    }

                    uintptr_t out_pstride;
                    if(lines < ocount) {
                      // third output dim comes from splitting current
                      out_pstride = out_lstride * lines;
                      size_t oplanes = ocount / lines;
                      // check for leftovers here if we go beyond 3D!
                      ocount = oplanes;
                      oscale *= lines;
                    } else {
                      od++;
                      assert(out_dim > od);
                      ocount = out_alc.remaining(od);
                      out_pstride = out_alc.get_stride(od);
                      oscale = 1;
                    }

                    size_t planes = std::min(std::min(icount, ocount),
                                             (bytes_left / (contig_bytes * lines)));
                    bytes = contig_bytes * lines * planes;
                    memcpy_3d(out_base + out_offset, out_lstride, out_pstride,
                              in_base + in_offset, in_lstride, in_pstride, contig_bytes,
                              lines, planes);
                    in_alc.advance(id, planes * iscale);
                    out_alc.advance(od, planes * oscale);
                  }
                }
              } else {
                // scatter adddress list
                assert(0);
              }
            } else {
              if(out_dim > 0) {
                // gather address list
                assert(0);
              } else {
                // gather and scatter
                assert(0);
              }
            }

#ifdef DEBUG_REALM
            assert(bytes <= bytes_left);
#endif
            total_bytes += bytes;

            // stop if it's been too long, but make sure we do at least the
            //  minimum number of bytes
            if((total_bytes >= min_xfer_size) && work_until.is_expired()) {
              break;
            }
          }
        } else {
          // input but no output, so skip input bytes
          total_bytes = max_bytes;
          in_port->addrcursor.skip_bytes(total_bytes);
        }
      } else {
        if(out_port != 0) {
          // output but no input, so skip output bytes
          total_bytes = max_bytes;
          out_port->addrcursor.skip_bytes(total_bytes);
        } else {
          // skipping both input and output is possible for simultaneous
          //  gather+scatter
          total_bytes = max_bytes;
        }
      }

      // memcpy is always immediate, so handle both skip and copy with the
      //  same code
      rseqcache.add_span(input_control.current_io_port, in_span_start, total_bytes);
      in_span_start += total_bytes;
      wseqcache.add_span(output_control.current_io_port, out_span_start, total_bytes);
      out_span_start += total_bytes;

      bool done = record_address_consumption(total_bytes, total_bytes);

      did_work = true;

      if(done || work_until.is_expired()) {
        break;
      }
    }

    rseqcache.flush();
    wseqcache.flush();

    return did_work;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemcpyChannel
  //

  static void
  enumerate_remote_shared_mems(std::vector<Memory> &mems,
                               const std::unordered_map<realm_id_t, SharedMemoryInfo>
                                   &remote_shared_memory_mappings)
  {
    if(remote_shared_memory_mappings.empty()) {
      return;
    }
    size_t idx = 0;
    mems.resize(remote_shared_memory_mappings.size(), Memory::NO_MEMORY);
    for(std::unordered_map<realm_id_t, SharedMemoryInfo>::const_iterator it =
            remote_shared_memory_mappings.begin();
        it != remote_shared_memory_mappings.end(); ++it) {
      Memory m;
      m.id = it->first;
      mems[idx++] = m;
    }
  }

  MemcpyChannel::MemcpyChannel(BackgroundWorkManager *_bgwork, const Node *_node,
                               const std::unordered_map<realm_id_t, SharedMemoryInfo>
                                   &remote_shared_memory_mappings,
                               NodeID _my_node_id)
    : SingleXDQChannel<MemcpyChannel, MemcpyXferDes>(_bgwork, XFER_MEM_CPY,
                                                     "memcpy channel")
    , node(_node)
  {
    // cbs = (MemcpyRequest**) calloc(max_nr, sizeof(MemcpyRequest*));
    unsigned bw = 128000;         // HACK - estimate at 128 GB/s
    unsigned latency = 100;       // HACK - estimate at 100ns
    unsigned frag_overhead = 100; // HACK - estimate at 100ns

    std::vector<Memory> local_cpu_mems;
    enumerate_local_cpu_memories(node, local_cpu_mems);
    std::vector<Memory> remote_shared_mems;

    if(!remote_shared_memory_mappings.empty()) {
      enumerate_remote_shared_mems(remote_shared_mems, remote_shared_memory_mappings);
    }

    add_path(local_cpu_mems, local_cpu_mems, bw, latency, frag_overhead, XFER_MEM_CPY)
        .set_max_dim(3)
        .allow_serdez();

    if(remote_shared_mems.size() > 0) {
      add_path(local_cpu_mems, remote_shared_mems, bw, latency, frag_overhead,
               XFER_MEM_CPY)
          .set_max_dim(3);
    }

    xdq.add_to_manager(_bgwork);
  }

  MemcpyChannel::~MemcpyChannel()
  {
    // free(cbs);
  }

  uint64_t MemcpyChannel::supports_path(
      ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
      CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
      const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
      XferDesKind *kind_ret /*= 0*/, unsigned *bw_ret /*= 0*/, unsigned *lat_ret /*= 0*/)
  {
    // simultaneous serialization/deserialization not
    //  allowed anywhere right now
    if((src_serdez_id != 0) && (dst_serdez_id != 0)) {
      return 0;
    }

    // fall through to normal checks
    return Channel::supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                  redop_id, total_bytes, src_frags, dst_frags, kind_ret,
                                  bw_ret, lat_ret);
  }

  XferDes *
  MemcpyChannel::create_xfer_des(uintptr_t dma_op, NodeID launch_node, XferDesID guid,
                                 const std::vector<XferDesPortInfo> &inputs_info,
                                 const std::vector<XferDesPortInfo> &outputs_info,
                                 int priority, XferDesRedopInfo redop_info,
                                 const void *fill_data, size_t fill_size,
                                 size_t fill_total)
  {
    assert(redop_info.id == 0);
    assert(fill_size == 0);
    return new MemcpyXferDes(dma_op, this, launch_node, guid, inputs_info, outputs_info,
                             priority);
  }

  long MemcpyChannel::submit(Request **requests, long nr)
  {
    MemcpyRequest **mem_cpy_reqs = (MemcpyRequest **)requests;
    for(long i = 0; i < nr; i++) {
      MemcpyRequest *req = mem_cpy_reqs[i];
      // handle 1-D, 2-D, and 3-D in a single loop
      switch(req->dim) {
      case Request::DIM_1D:
        assert(req->nplanes == 1);
        assert(req->nlines == 1);
        break;
      case Request::DIM_2D:
        assert(req->nplanes == 1);
        break;
      case Request::DIM_3D:
        // nothing to check
        break;
      default:
        assert(0);
      }
      size_t rewind_src = 0;
      size_t rewind_dst = 0;
      XferDes::XferPort *in_port = &req->xd->input_ports[req->src_port_idx];
      XferDes::XferPort *out_port = &req->xd->output_ports[req->dst_port_idx];
      const CustomSerdezUntyped *src_serdez_op = in_port->serdez_op;
      const CustomSerdezUntyped *dst_serdez_op = out_port->serdez_op;
      if(src_serdez_op && !dst_serdez_op) {
        // we manage write_bytes_total, write_seq_{pos,count}
        req->write_seq_pos = out_port->local_bytes_total;
      }
      if(!src_serdez_op && dst_serdez_op) {
        // we manage read_bytes_total, read_seq_{pos,count}
        req->read_seq_pos = in_port->local_bytes_total;
      }
      {
        char *wrap_buffer = 0;
        bool wrap_buffer_malloced = false;
        const size_t ALLOCA_LIMIT = 4096;
        const char *src_p = (const char *)(req->src_base);
        char *dst_p = (char *)(req->dst_base);
        for(size_t j = 0; j < req->nplanes; j++) {
          const char *src = src_p;
          char *dst = dst_p;
          for(size_t i = 0; i < req->nlines; i++) {
            if(src_serdez_op) {
              if(dst_serdez_op) {
                // serialization AND deserialization
                assert(0);
              } else {
                // serialization
                size_t field_size = src_serdez_op->sizeof_field_type;
                size_t num_elems = req->nbytes / field_size;
                assert((num_elems * field_size) == req->nbytes);
                size_t maxser_size = src_serdez_op->max_serialized_size;
                size_t max_bytes = num_elems * maxser_size;
                // ask the dst iterator (which should be a
                //  WrappingFIFOIterator for enough space to write all the
                //  serialized data in the worst case
                TransferIterator::AddressInfo dst_info;
                size_t bytes_avail =
                    out_port->iter->step(max_bytes, dst_info, 0, true /*tentative*/);
                size_t bytes_used;
                if(bytes_avail == max_bytes) {
                  // got enough space to do it all in one go
                  void *dst =
                      out_port->mem->get_direct_ptr(dst_info.base_offset, bytes_avail);
                  assert(dst != 0);
                  bytes_used = src_serdez_op->serialize(src, field_size, num_elems, dst);
                  if(bytes_used == max_bytes) {
                    out_port->iter->confirm_step();
                  } else {
                    out_port->iter->cancel_step();
                    bytes_avail = out_port->iter->step(bytes_used, dst_info, 0,
                                                       false /*!tentative*/);
                    assert(bytes_avail == bytes_used);
                  }
                } else {
                  // we didn't get the worst case amount, but it might be
                  //  enough
                  void *dst =
                      out_port->mem->get_direct_ptr(dst_info.base_offset, bytes_avail);
                  assert(dst != 0);
                  size_t elems_done = 0;
                  size_t bytes_left = bytes_avail;
                  bytes_used = 0;
                  while((elems_done < num_elems) && (bytes_left >= maxser_size)) {
                    size_t todo =
                        std::min(num_elems - elems_done, bytes_left / maxser_size);
                    size_t amt = src_serdez_op->serialize(((const char *)src) +
                                                              (elems_done * field_size),
                                                          field_size, todo, dst);
                    assert(amt <= bytes_left);
                    elems_done += todo;
                    bytes_left -= amt;
                    dst = ((char *)dst) + amt;
                    bytes_used += amt;
                  }
                  if(elems_done == num_elems) {
                    // we ended up getting all we needed without wrapping
                    if(bytes_used == bytes_avail) {
                      out_port->iter->confirm_step();
                    } else {
                      out_port->iter->cancel_step();
                      bytes_avail = out_port->iter->step(bytes_used, dst_info, 0,
                                                         false /*!tentative*/);
                      assert(bytes_avail == bytes_used);
                    }
                  } else {
                    // did we get lucky and finish on the wrap boundary?
                    if(bytes_left == 0) {
                      out_port->iter->confirm_step();
                    } else {
                      // need a temp buffer to deal with wraparound
                      if(!wrap_buffer) {
                        if(maxser_size > ALLOCA_LIMIT) {
                          wrap_buffer_malloced = true;
                          wrap_buffer = (char *)malloc(maxser_size);
                        } else {
                          wrap_buffer = (char *)alloca(maxser_size);
                        }
                      }
                      while((elems_done < num_elems) && (bytes_left > 0)) {
                        // serialize one element into our buffer
                        size_t amt = src_serdez_op->serialize(
                            ((const char *)src) + (elems_done * field_size), wrap_buffer);
                        if(amt < bytes_left) {
                          memcpy(dst, wrap_buffer, amt);
                          bytes_left -= amt;
                          dst = ((char *)dst) + amt;
                        } else {
                          memcpy(dst, wrap_buffer, bytes_left);
                          out_port->iter->confirm_step();
                          if(amt > bytes_left) {
                            size_t amt2 = out_port->iter->step(amt - bytes_left, dst_info,
                                                               0, false /*!tentative*/);
                            assert(amt2 == (amt - bytes_left));
                            void *dst =
                                out_port->mem->get_direct_ptr(dst_info.base_offset, amt2);
                            assert(dst != 0);
                            memcpy(dst, wrap_buffer + bytes_left, amt2);
                          }
                          bytes_left = 0;
                        }
                        elems_done++;
                        bytes_used += amt;
                      }
                      // if we still finished with bytes left over, give
                      //  them back to the iterator
                      if(bytes_left > 0) {
                        assert(elems_done == num_elems);
                        out_port->iter->cancel_step();
                        size_t amt = out_port->iter->step(bytes_used, dst_info, 0,
                                                          false /*!tentative*/);
                        assert(amt == bytes_used);
                      }
                    }

                    // now that we're after the wraparound, any remaining
                    //  elements are fairly straightforward
                    if(elems_done < num_elems) {
                      size_t max_remain = ((num_elems - elems_done) * maxser_size);
                      size_t amt = out_port->iter->step(max_remain, dst_info, 0,
                                                        true /*tentative*/);
                      assert(amt == max_remain); // no double-wrap
                      void *dst =
                          out_port->mem->get_direct_ptr(dst_info.base_offset, amt);
                      assert(dst != 0);
                      size_t amt2 = src_serdez_op->serialize(
                          ((const char *)src) + (elems_done * field_size), field_size,
                          num_elems - elems_done, dst);
                      bytes_used += amt2;
                      if(amt2 == max_remain) {
                        out_port->iter->confirm_step();
                      } else {
                        out_port->iter->cancel_step();
                        size_t amt3 =
                            out_port->iter->step(amt2, dst_info, 0, false /*!tentative*/);
                        assert(amt3 == amt2);
                      }
                    }
                  }
                }
                assert(bytes_used <= max_bytes);
                if(bytes_used < max_bytes) {
                  rewind_dst += (max_bytes - bytes_used);
                }
                out_port->local_bytes_total += bytes_used;
              }
            } else {
              if(dst_serdez_op) {
                // deserialization
                size_t field_size = dst_serdez_op->sizeof_field_type;
                size_t num_elems = req->nbytes / field_size;
                assert((num_elems * field_size) == req->nbytes);
                size_t maxser_size = dst_serdez_op->max_serialized_size;
                size_t max_bytes = num_elems * maxser_size;
                // ask the srct iterator (which should be a
                //  WrappingFIFOIterator for enough space to read all the
                //  serialized data in the worst case
                TransferIterator::AddressInfo src_info;
                size_t bytes_avail =
                    in_port->iter->step(max_bytes, src_info, 0, true /*tentative*/);
                size_t bytes_used;
                if(bytes_avail == max_bytes) {
                  // got enough space to do it all in one go
                  const void *src =
                      in_port->mem->get_direct_ptr(src_info.base_offset, bytes_avail);
                  assert(src != 0);
                  bytes_used =
                      dst_serdez_op->deserialize(dst, field_size, num_elems, src);
                  if(bytes_used == max_bytes) {
                    in_port->iter->confirm_step();
                  } else {
                    in_port->iter->cancel_step();
                    bytes_avail = in_port->iter->step(bytes_used, src_info, 0,
                                                      false /*!tentative*/);
                    assert(bytes_avail == bytes_used);
                  }
                } else {
                  // we didn't get the worst case amount, but it might be
                  //  enough
                  const void *src =
                      in_port->mem->get_direct_ptr(src_info.base_offset, bytes_avail);
                  assert(src != 0);
                  size_t elems_done = 0;
                  size_t bytes_left = bytes_avail;
                  bytes_used = 0;
                  while((elems_done < num_elems) && (bytes_left >= maxser_size)) {
                    size_t todo =
                        std::min(num_elems - elems_done, bytes_left / maxser_size);
                    size_t amt = dst_serdez_op->deserialize(
                        ((char *)dst) + (elems_done * field_size), field_size, todo, src);
                    assert(amt <= bytes_left);
                    elems_done += todo;
                    bytes_left -= amt;
                    src = ((const char *)src) + amt;
                    bytes_used += amt;
                  }
                  if(elems_done == num_elems) {
                    // we ended up getting all we needed without wrapping
                    if(bytes_used == bytes_avail) {
                      in_port->iter->confirm_step();
                    } else {
                      in_port->iter->cancel_step();
                      bytes_avail = in_port->iter->step(bytes_used, src_info, 0,
                                                        false /*!tentative*/);
                      assert(bytes_avail == bytes_used);
                    }
                  } else {
                    // did we get lucky and finish on the wrap boundary?
                    if(bytes_left == 0) {
                      in_port->iter->confirm_step();
                    } else {
                      // need a temp buffer to deal with wraparound
                      if(!wrap_buffer) {
                        if(maxser_size > ALLOCA_LIMIT) {
                          wrap_buffer_malloced = true;
                          wrap_buffer = (char *)malloc(maxser_size);
                        } else {
                          wrap_buffer = (char *)alloca(maxser_size);
                        }
                      }
                      // keep a snapshot of the iterator in cse we don't wrap after all
                      Serialization::DynamicBufferSerializer dbs(64);
                      dbs << *(in_port->iter);
                      memcpy(wrap_buffer, src, bytes_left);
                      // get pointer to data on other side of wrap
                      in_port->iter->confirm_step();
                      size_t amt = in_port->iter->step(max_bytes - bytes_avail, src_info,
                                                       0, true /*tentative*/);
                      // it's actually ok for this to appear to come up short - due to
                      //  flow control we know we won't ever actually wrap around
                      // assert(amt == (max_bytes - bytes_avail));
                      const void *src =
                          in_port->mem->get_direct_ptr(src_info.base_offset, amt);
                      assert(src != 0);
                      memcpy(wrap_buffer + bytes_left, src, maxser_size - bytes_left);
                      src = ((const char *)src) + (maxser_size - bytes_left);

                      while((elems_done < num_elems) && (bytes_left > 0)) {
                        // deserialize one element from our buffer
                        amt = dst_serdez_op->deserialize(
                            ((char *)dst) + (elems_done * field_size), wrap_buffer);
                        if(amt < bytes_left) {
                          // slide data, get a few more bytes
                          memmove(wrap_buffer, wrap_buffer + amt, maxser_size - amt);
                          memcpy(wrap_buffer + maxser_size, src, amt);
                          bytes_left -= amt;
                          src = ((const char *)src) + amt;
                        } else {
                          // update iterator to say how much wrapped data was actually
                          // used
                          in_port->iter->cancel_step();
                          if(amt > bytes_left) {
                            size_t amt2 = in_port->iter->step(amt - bytes_left, src_info,
                                                              0, false /*!tentative*/);
                            assert(amt2 == (amt - bytes_left));
                          }
                          bytes_left = 0;
                        }
                        elems_done++;
                        bytes_used += amt;
                      }
                      // if we still finished with bytes left, we have
                      //  to restore the iterator because we
                      //  can't double-cancel
                      if(bytes_left > 0) {
                        assert(elems_done == num_elems);
                        delete in_port->iter;
                        Serialization::FixedBufferDeserializer fbd(dbs.get_buffer(),
                                                                   dbs.bytes_used());
                        in_port->iter = TransferIterator::deserialize_new(fbd);
                        in_port->iter->cancel_step();
                        size_t amt2 = in_port->iter->step(bytes_used, src_info, 0,
                                                          false /*!tentative*/);
                        assert(amt2 == bytes_used);
                      }
                    }

                    // now that we're after the wraparound, any remaining
                    //  elements are fairly straightforward
                    if(elems_done < num_elems) {
                      size_t max_remain = ((num_elems - elems_done) * maxser_size);
                      size_t amt = in_port->iter->step(max_remain, src_info, 0,
                                                       true /*tentative*/);
                      assert(amt == max_remain); // no double-wrap
                      const void *src =
                          in_port->mem->get_direct_ptr(src_info.base_offset, amt);
                      assert(src != 0);
                      size_t amt2 = dst_serdez_op->deserialize(
                          ((char *)dst) + (elems_done * field_size), field_size,
                          num_elems - elems_done, src);
                      bytes_used += amt2;
                      if(amt2 == max_remain) {
                        in_port->iter->confirm_step();
                      } else {
                        in_port->iter->cancel_step();
                        size_t amt3 =
                            in_port->iter->step(amt2, src_info, 0, false /*!tentative*/);
                        assert(amt3 == amt2);
                      }
                    }
                  }
                }
                assert(bytes_used <= max_bytes);
                if(bytes_used < max_bytes) {
                  rewind_src += (max_bytes - bytes_used);
                }
                in_port->local_bytes_total += bytes_used;
              } else {
                // normal copy
                memcpy(dst, src, req->nbytes);
              }
            }
            if(req->dim == Request::DIM_1D) {
              break;
            }
            // serdez cases update src/dst directly
            // NOTE: this looks backwards, but it's not - a src serdez means it's the
            //  destination that moves unpredictably
            if(!dst_serdez_op) {
              src += req->src_str;
            }
            if(!src_serdez_op) {
              dst += req->dst_str;
            }
          }
          if((req->dim == Request::DIM_1D) || (req->dim == Request::DIM_2D)) {
            break;
          }
          // serdez cases update src/dst directly - copy back to src/dst_p
          src_p = (dst_serdez_op ? src : src_p + req->src_pstr);
          dst_p = (src_serdez_op ? dst : dst_p + req->dst_pstr);
        }
        // clean up our wrap buffer, if we malloc'd it
        if(wrap_buffer_malloced) {
          free(wrap_buffer);
        }
      }
      if(src_serdez_op && !dst_serdez_op) {
        // we manage write_bytes_total, write_seq_{pos,count}
        req->write_seq_count = out_port->local_bytes_total - req->write_seq_pos;
        if(rewind_dst > 0) {
          // log_request.print() << "rewind dst: " << rewind_dst;
          // if we've finished iteration, it's too late to rewind the
          //  conservative count, so decrement the number of write bytes
          //  pending (we know we can't drive it to zero) as well
          if(req->xd->iteration_completed.load()) {
            int64_t prev = req->xd->bytes_write_pending.fetch_sub(rewind_dst);
            assert((prev > 0) && (static_cast<size_t>(prev) > rewind_dst));
          }
          out_port->local_bytes_cons.fetch_sub(rewind_dst);
        }
      } else {
        assert(rewind_dst == 0);
      }
      if(!src_serdez_op && dst_serdez_op) {
        // we manage read_bytes_total, read_seq_{pos,count}
        req->read_seq_count = in_port->local_bytes_total - req->read_seq_pos;
        if(rewind_src > 0) {
          // log_request.print() << "rewind src: " << rewind_src;
          in_port->local_bytes_cons.fetch_sub(rewind_src);
        }
      } else {
        assert(rewind_src == 0);
      }
      req->xd->notify_request_read_done(req);
      req->xd->notify_request_write_done(req);
    }
    return nr;
  }
}; // namespace Realm
