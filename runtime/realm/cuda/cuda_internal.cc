/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#include "realm/cuda/cuda_internal.h"
#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_access.h"
#include "realm/cuda/cuda_memcpy.h"

namespace Realm {

  extern Logger log_xd;

  namespace Cuda {

    extern Logger log_gpu;
    extern Logger log_stream;
    extern Logger log_gpudma;

    typedef int (*PFN_cudaLaunchKernel)(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, CUstream stream);
    typedef int (*PFN_cudaGetFuncBySymbol)(void *functionPtr,
                                           const void *symbolPtr);

    ////////////////////////////////////////////////////////////////////////
    //
    // class CudaDeviceMemoryInfo

    CudaDeviceMemoryInfo::CudaDeviceMemoryInfo(CUcontext _context)
      : context(_context)
      , gpu(0)
    {
      // see if we can match this context to one of our GPU objects - handle
      //  the case where the cuda module didn't load though
      CudaModule *mod = get_runtime()->get_module<CudaModule>("cuda");
      if(mod) {
        for(std::vector<GPU *>::const_iterator it = mod->gpus.begin();
            it != mod->gpus.end();
            ++it)
          if((*it)->context == _context) {
            gpu = *it;
            break;
          }
      }
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class MemSpecificCudaArray
    //

    MemSpecificCudaArray::MemSpecificCudaArray(CUarray _array)
      : array(_array)
    {}

    MemSpecificCudaArray::~MemSpecificCudaArray()
    {
      assert(array == 0);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class AddressInfoCudaArray

    int AddressInfoCudaArray::set_rect(const RegionInstanceImpl *inst,
                                       const InstanceLayoutPieceBase *piece,
                                       size_t field_size, size_t field_offset,
                                       int ndims,
                                       const int64_t lo[/*ndims*/],
                                       const int64_t hi[/*ndims*/],
                                       const int order[/*ndims*/])
    {
      assert(ndims <= 3);
      const MemSpecificCudaArray *ms = inst->metadata.find_mem_specific<MemSpecificCudaArray>();
      assert(ms);
      array = ms->array;
      dim = ndims;
      pos[0] = (ndims >= 1) ? lo[0] * field_size : 0; // x is measured in bytes
      pos[1] = (ndims >= 2) ? lo[1] : 0;
      pos[2] = (ndims >= 3) ? lo[2] : 0;
      width_in_bytes = field_size;
      height = 1;
      depth = 1;
      // can only handle non-trivial dimensions in ascending order
      int ok_dims = 0;
      int prev_dim = -1;
      while(ok_dims < ndims) {
        int di = order[ok_dims];
        if(hi[di] != lo[di]) {
          if(di <= prev_dim) break;
          prev_dim = di;
          switch(di) {
          case 0: width_in_bytes *= (hi[0] - lo[0] + 1); break;
          case 1: height = hi[1] - lo[1] + 1; break;
          case 2: depth = hi[2] - lo[2] + 1; break;
          default: assert(0);
          }
        }
        ok_dims++;
      }

      return ok_dims;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUXferDes

    static GPU *mem_to_gpu(const MemoryImpl *mem)
    {
      if(ID(mem->me).is_memory()) {
        // might not be a GPUFBMemory...
        const GPUFBMemory *fbmem = dynamic_cast<const GPUFBMemory *>(mem);
        if(fbmem)
          return fbmem->gpu;

        // see if it has CudaDeviceMemoryInfo with a valid gpu
        const CudaDeviceMemoryInfo *cdm = mem->find_module_specific<CudaDeviceMemoryInfo>();
        if(cdm && cdm->gpu)
          return cdm->gpu;

        // not a gpu-associated memory
        return 0;
      } else {
        // is it an FBIBMemory?
        const GPUFBIBMemory *ibmem = dynamic_cast<const GPUFBIBMemory *>(mem);
        if(ibmem)
          return ibmem->gpu;

        // not a gpu-associated memory
        return 0;
      }
    }

    GPUXferDes::GPUXferDes(uintptr_t _dma_op, Channel *_channel,
                           NodeID _launch_node, XferDesID _guid,
                           const std::vector<XferDesPortInfo>& inputs_info,
                           const std::vector<XferDesPortInfo>& outputs_info,
                           int _priority)
      : XferDes(_dma_op, _channel, _launch_node, _guid,
                inputs_info, outputs_info,
                _priority, 0, 0)
    {
      kind = XFER_GPU_IN_FB; // TODO: is this needed at all?

      src_gpus.resize(inputs_info.size(), 0);
      for(size_t i = 0; i < input_ports.size(); i++) {
        src_gpus[i] = mem_to_gpu(input_ports[i].mem);
        // sanity-check
	if(input_ports[i].mem->kind == MemoryImpl::MKIND_GPUFB)
          assert(src_gpus[i]);
      }

      dst_gpus.resize(outputs_info.size(), 0);
      dst_is_ipc.resize(outputs_info.size(), false);
      for(size_t i = 0; i < output_ports.size(); i++) {
        dst_gpus[i] = mem_to_gpu(output_ports[i].mem);
	if(output_ports[i].mem->kind == MemoryImpl::MKIND_GPUFB) {
          // sanity-check
          assert(dst_gpus[i]);
        } else {
          // assume a memory owned by another node is ipc
          if(NodeID(ID(output_ports[i].mem->me).memory_owner_node()) != Network::my_node_id)
            dst_is_ipc[i] = true;
        }
      }
    }
	
    long GPUXferDes::get_requests(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    static GPUStream *select_stream(GPU* dst, GPU* src, const GPU::CudaIpcMapping *dst_mapping = 0)
    {
      if (src) {
        if (dst == src) {
          return src->get_next_d2d_stream();
        }
        else if (dst_mapping != NULL) {
          return src->cudaipc_streams[dst_mapping->owner];
        }
        else if (dst == NULL) {
          return src->device_to_host_stream;
        }
        return src->peer_to_peer_streams[dst->info->index];
      }
      assert(dst != NULL);
      return dst->host_to_device_stream;
    }

    static void get_nonaffine_strides(size_t &pitch, size_t &height, AddressInfoCudaArray &ainfo, AddressListCursor &alc, size_t bytes)
    {
      bool shape_ok = false;
      pitch = 1;  // Set a default for pitch and height
      height = 1; // Don't set these to zero, as the driver will patch a pitch that may be
                  // non-sensical
      if(ainfo.dim <= 2) {
        if((alc.get_dim() == 1) ||
            (alc.remaining(0) >= bytes)) {
          // contiguous input range
          pitch = ainfo.width_in_bytes;
          height = ainfo.height;
          alc.advance(0, bytes);
        } else {
          // width has to match
          assert(alc.remaining(0) == ainfo.width_in_bytes);
          assert(alc.remaining(1) >= ainfo.height);
          pitch = alc.get_stride(1);
          height = ainfo.height;
          alc.advance(1, ainfo.height);
        }
        shape_ok = true;
      } else {
        if((alc.get_dim() == 1) ||
            (alc.remaining(0) >= bytes)) {
          // contiguous input range
          pitch = ainfo.width_in_bytes;
          height = ainfo.height;
          alc.advance(0, bytes);
          shape_ok = true;
        } else {
          // if it's not contiguous, width must be exactly what
          // we need for either 1 or 2 leading dimensions
          if(alc.remaining(0) == ainfo.width_in_bytes) {
            if((alc.get_dim() == 2) || (alc.remaining(1) >= (ainfo.height * ainfo.depth))) {
              // input dim 1 covers output 1 and 2
              pitch = alc.get_stride(1);
              height = ainfo.height;
              alc.advance(1, (ainfo.height * ainfo.depth));
              shape_ok = true;
            } else {
              // for a full 3 dimensions, we need need dim 1 to
              // match exactly AND the stride for dim 2 has to
              // be a multiple of dim 1's stride due to
              // cuMemcpy3D restrictions
              if((alc.remaining(1) == ainfo.height) &&
                  (alc.get_dim() >= 3) &&
                  (alc.remaining(2) >= ainfo.depth) &&
                  ((alc.get_stride(2) % alc.get_stride(1)) == 0)) {
                pitch = alc.get_stride(1);
                height = (alc.get_stride(2) / alc.get_stride(1));
                alc.advance(2, ainfo.depth);
                shape_ok = true;
              }
            }
          } else {
            if((alc.remaining(0) == (ainfo.width_in_bytes * ainfo.height)) &&
                (alc.remaining(1) >= ainfo.depth) &&
                ((alc.get_stride(1) % ainfo.width_in_bytes) == 0)) {
              pitch = ainfo.width_in_bytes;
              height = (alc.get_stride(1) / ainfo.width_in_bytes);
              alc.advance(1, ainfo.depth);
              shape_ok = true;
            }
          }
        }
      }
      if(!shape_ok) {
        log_gpudma.fatal() << "array copy shape mismatch: alc="
                        << alc << " ainfo="
                        << ainfo.width_in_bytes << "x"
                        << ainfo.height << "x"
                        << ainfo.depth;
        abort();
      }
    }

    static bool needs_transpose(size_t in_lstride, size_t in_pstride, size_t out_lstride,
                                size_t out_pstride)
    {
      return in_lstride > in_pstride || out_lstride > out_pstride;
    }

    // Calculates the maximum alignment native type alignment the GPU supports that will
    // work with the given size.
    static size_t calculate_type_alignment(size_t v)
    {
      // We don't need a full log2 here
      for(size_t a = 16; a > 1; a >>= 1) {
        if((v & (a - 1)) == 0)
          return a;
      }
      return 1; // Unfortunately this can only be byte aligned :(
    }

    static size_t populate_affine_copy_info(AffineCopyInfo<3> &copy_infos,
                                            size_t &min_align,
                                            MemcpyTransposeInfo<size_t> &transpose_info,
                                            AddressListCursor &in_alc, uintptr_t in_base,
                                            GPU *in_gpu, AddressListCursor &out_alc,
                                            uintptr_t out_base, GPU *out_gpu,
                                            size_t bytes_left)
    {
      AffineCopyPair<3> &copy_info = copy_infos.subrects[copy_infos.num_rects++];
      uintptr_t in_offset = in_alc.get_offset();
      uintptr_t out_offset = out_alc.get_offset();
      // the reported dim is reduced for partially consumed address
      // ranges - whatever we get can be assumed to be regular
      int in_dim = in_alc.get_dim();
      int out_dim = out_alc.get_dim();
      size_t icount = in_alc.remaining(0);
      size_t ocount = out_alc.remaining(0);
      // contig bytes is always the min of the first dimensions
      size_t contig_bytes = std::min(std::min(icount, ocount), bytes_left);

      log_gpudma.info() << "IN: " << in_dim << ' ' << icount << ' ' << in_offset << ' '
                        << contig_bytes;
      log_gpudma.info() << "OUT: " << out_dim << ' ' << ocount << ' ' << out_offset << ' '
                        << contig_bytes;

      assert(in_dim > 0);
      assert(out_dim > 0);

      copy_info.src.addr = static_cast<uintptr_t>(in_base + in_offset);
      copy_info.dst.addr = static_cast<uintptr_t>(out_base + out_offset);
      copy_info.extents[1] = 1;
      copy_info.extents[2] = 1;
      min_align = std::min(min_align, calculate_type_alignment(copy_info.src.addr));
      min_align = std::min(min_align, calculate_type_alignment(copy_info.dst.addr));

      // Calculate the minimum alignment for contig bytes
      min_align = std::min(min_align, calculate_type_alignment(contig_bytes));

      // catch simple 1D case first
      if((contig_bytes == bytes_left) || ((contig_bytes == icount) && (in_dim == 1)) ||
         ((contig_bytes == ocount) && (out_dim == 1))) {
        copy_info.extents[0] = contig_bytes;
        copy_info.src.strides[0] = contig_bytes;
        copy_info.dst.strides[0] = contig_bytes;
        copy_info.volume = contig_bytes;

        in_alc.advance(0, contig_bytes);
        out_alc.advance(0, contig_bytes);
        return contig_bytes;
      }

      // grow to a 2D copy
      int id;
      size_t iscale;
      uintptr_t in_lstride;
      if(contig_bytes < icount) {
        // second input dim comes from splitting first
        id = 0;
        in_lstride = contig_bytes;
        size_t ilines = icount / contig_bytes;
        if((ilines * contig_bytes) != icount)
          in_dim = 1; // leftover means we can't go beyond this
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
      size_t oscale;
      uintptr_t out_lstride;
      if(contig_bytes < ocount) {
        // second output dim comes from splitting first
        od = 0;
        out_lstride = contig_bytes;
        size_t olines = ocount / contig_bytes;
        if((olines * contig_bytes) != ocount)
          out_dim = 1; // leftover means we can't go beyond this
        ocount = olines;
        oscale = contig_bytes;
      } else {
        assert(out_dim > 1);
        od = 1;
        ocount = out_alc.remaining(od);
        out_lstride = out_alc.get_stride(od);
        oscale = 1;
      }

      size_t lines = std::min(std::min(icount, ocount), bytes_left / contig_bytes);

      // *_lstride is the number of bytes for each line, so recalculate
      // the minimum alignment to make sure the alignment matches the
      // byte alignment across all lines.
      min_align = std::min(min_align, calculate_type_alignment(in_lstride));
      min_align = std::min(min_align, calculate_type_alignment(out_lstride));

      // see if we need to stop at 2D
      if(((contig_bytes * lines) == bytes_left) ||
         ((lines == icount) && (id == (in_dim - 1))) ||
         ((lines == ocount) && (od == (out_dim - 1)))) {
        copy_info.src.strides[0] = in_lstride;
        copy_info.src.strides[1] = lines;
        copy_info.dst.strides[0] = out_lstride;
        copy_info.dst.strides[1] = lines;
        copy_info.extents[0] = contig_bytes;
        copy_info.extents[1] = lines;
        copy_info.volume = lines * contig_bytes;

        in_alc.advance(id, lines * iscale);
        out_alc.advance(od, lines * oscale);
        return lines * contig_bytes;
      }

      // Grow to a 3D copy
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

      const size_t planes =
          std::min(std::min(icount, ocount), (bytes_left / (contig_bytes * lines)));

      if(needs_transpose(in_lstride, in_pstride, out_lstride, out_pstride)) {
        transpose_info.src = static_cast<uintptr_t>(in_base + in_offset);
        transpose_info.dst = static_cast<uintptr_t>(out_base + out_offset);

        transpose_info.src_strides[0] = in_lstride;
        transpose_info.src_strides[1] = in_pstride;

        transpose_info.dst_strides[0] = out_lstride;
        transpose_info.dst_strides[1] = out_pstride;

        transpose_info.extents[0] = contig_bytes;
        transpose_info.extents[1] = lines;
        transpose_info.extents[2] = planes;
        copy_infos.num_rects--;
      } else {
        copy_info.dst.strides[0] = out_lstride;
        copy_info.dst.strides[1] = out_pstride / out_lstride;

        copy_info.extents[0] = contig_bytes;
        copy_info.extents[1] = lines;
        copy_info.extents[2] = planes;

        copy_info.src.strides[0] = in_lstride;
        copy_info.src.strides[1] = in_pstride / in_lstride;

        copy_info.volume = planes * lines * contig_bytes;
      }

      in_alc.advance(id, planes * iscale);
      out_alc.advance(od, planes * oscale);
      return planes * lines * contig_bytes;
    }

    bool GPUXferDes::progress_xd(GPUChannel *channel, TimeLimit work_until)
    {
      // Mininum amount to transfer in a single quantum before returning in order to
      // ensure forward progress
      // TODO: make controllable
      const size_t MIN_XFER_SIZE = 4 << 20;
      // Maximum amount to transfer in a single quantum in order to ensure other requests
      // have a chance to make forward progress.  This should be large enough that the
      // overhead of splitting the copy shouldn't be noticable in terms of latency (4GiB
      // should be good here for most purposes)
      // TODO: make controllable
      const size_t flow_control_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
      ReadSequenceCache rseqcache(this, 2 << 20);
      WriteSequenceCache wseqcache(this, 2 << 20);
      GPUStream *stream = 0;
      size_t total_bytes = 0;

      AffineCopyInfo<3> copy_infos;
      CUDA_MEMCPY3D cuda_copy;
      MemcpyTransposeInfo<size_t> transpose_copy;

      memset(&cuda_copy, 0, sizeof(cuda_copy));
      memset(&transpose_copy, 0, sizeof(transpose_copy));
      memset(&copy_infos, 0, sizeof(copy_infos));

      // The general algorithm here can be described in three loops:
      // 1) Outer loop - iterates over all the addresses for each request.  This typically
      // corresponds to each rectangle in an index space transfer. 2) Batch loop - Map the
      // address list that can be a mix of different rectangle sizes to a batch of copies
      // that can be pushed in a single launch (either kernel or cuMemcpy call)
      //   2.a) At this point advancing the address list commits us to submitting the copy
      //   in #3, thus flow control happens here.
      // 3) Copy loop  - Based on the batch, descide the best copy to push.

      // 1) Outer loop - iterate over all the addresses for each request
      while (true) {
        XferPort *in_port = 0, *out_port = 0;
        size_t in_span_start = 0, out_span_start = 0;
        GPU *in_gpu = 0, *out_gpu = 0;
        const GPU::CudaIpcMapping *out_mapping = 0;
        uintptr_t in_base = 0, out_base = 0;
        bool out_is_ipc = false;

        const InstanceLayoutPieceBase *in_nonaffine, *out_nonaffine;

        if (((total_bytes >= MIN_XFER_SIZE) && work_until.is_expired()) || (total_bytes >= flow_control_bytes)) {
          log_gpudma.info() << "Flow control hit, copied " << total_bytes << " leave the rest for later!";
          break;
        }

        const size_t max_bytes = get_addresses(MIN_XFER_SIZE, &rseqcache,
                                                       in_nonaffine, out_nonaffine);
        if (max_bytes == 0) {
          break;
        }

        if (input_control.current_io_port >= 0) {
          in_port = &input_ports[input_control.current_io_port];
          in_span_start = in_port->local_bytes_total;
          in_gpu = src_gpus[input_control.current_io_port];
        }
        if (output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
          out_gpu = dst_gpus[output_control.current_io_port];
          out_is_ipc = dst_is_ipc[output_control.current_io_port];
        }

        if (in_port == 0 || out_port == 0) {
          if (in_port) {
            in_port->addrcursor.skip_bytes(max_bytes);
            rseqcache.add_span(input_control.current_io_port, in_span_start,
                               max_bytes);
          } else if (out_port) {
            out_port->addrcursor.skip_bytes(max_bytes);
          } else {
            wseqcache.add_span(output_control.current_io_port, out_span_start,
                               max_bytes);
          }
          // No in- or out- port here, so skip these bytes and move on to the next!
          continue;
        }

        assert(in_port && out_port);
        if (!in_nonaffine) {
          in_base = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
        }

        if (!out_nonaffine) {
          if (out_is_ipc) {
            out_mapping = in_gpu->find_ipc_mapping(out_port->mem->me);
            assert(out_mapping);
            out_base = out_mapping->local_base;
          } else {
            out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));
          }
        }

        stream = select_stream(out_gpu, in_gpu, out_mapping);
        assert(stream != NULL);

        std::string memcpy_kind;
        if(in_gpu) {
          if(out_gpu == in_gpu) {
            memcpy_kind = "d2d";
          } else if(out_mapping) {
            memcpy_kind = "ipc";
          } else if(!out_gpu) {
            memcpy_kind = "d2h";
          } else {
            memcpy_kind = "p2p";
          }
        } else {
          memcpy_kind = "h2d";
        }

        AutoGPUContext agc(stream->get_gpu());
        size_t copy_info_total = 0;
        size_t min_align = 16;  // Hope for the highest type alignment we can get, 16 bytes
        copy_infos.num_rects = 0;
        size_t bytes_left = std::min(flow_control_bytes - total_bytes, max_bytes);

        if(cuda_copy.WidthInBytes != 0) {
          memset(&cuda_copy, 0, sizeof(cuda_copy));
        }
        if(transpose_copy.src != 0) {
          memset(&transpose_copy, 0, sizeof(transpose_copy));
        }

        // 2) Batch loop - Collect all the rectangles for this inport/outport pair by
        // iterating the address list cursor for each and figure out what copy we can do
        // that best fits the layout of the source and destinations
        while (bytes_left > 0 && copy_infos.num_rects < AffineCopyInfo<3>::MAX_NUM_RECTS) {
          AddressListCursor &in_alc = in_port->addrcursor;
          AddressListCursor &out_alc = out_port->addrcursor;
          if (!in_nonaffine && !out_nonaffine) {
            log_gpudma.info() << "Affine -> Affine";
            // limit transfer size for host<->device copies
            // this is because CUDA stages these copies through a staging buffer and is a blocking call.
            // Thus to limit the amount of time spent within the cuda driver and to allow us to time out early if needed,
            // split these larger copies into smaller ones ourselves
            if(!in_gpu || (!out_gpu && !out_is_ipc)) {
              bytes_left = std::min(bytes_left, (size_t)(4U << 20U));
            }

            const size_t bytes_to_copy = populate_affine_copy_info(
                copy_infos, min_align, transpose_copy, in_alc, in_base, in_gpu, out_alc,
                out_base, out_gpu, bytes_left);
            // Either src or dst can't be accessed with a kernel, so just break out and
            // perform a standard cuMemcpy

            if(transpose_copy.extents[0] != 0 || (in_gpu == NULL) ||
               !in_gpu->can_access_peer(out_gpu)) {
              break;
            }

            log_gpudma.info() << "\tAdded " << bytes_to_copy
                              << " Bytes left= " << (bytes_left - bytes_to_copy);
            assert(bytes_to_copy <= bytes_left);
            copy_info_total += bytes_to_copy;
            bytes_left -= bytes_to_copy;
          } else {  // Non-affine transfers
            AddressInfoCudaArray ainfo;

            if (in_nonaffine) {
              assert(!out_nonaffine);
              log_gpudma.info() << "Array -> Affine";
              size_t bytes = in_port->iter->step_custom(bytes_left, ainfo,
                                                 false);
              if(bytes == 0) break;  // flow control or end of array
              cuda_copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
              cuda_copy.srcArray = ainfo.array;
              cuda_copy.srcXInBytes = ainfo.pos[0];
              cuda_copy.srcY = ainfo.pos[1];
              cuda_copy.srcZ = ainfo.pos[2];
              cuda_copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
              cuda_copy.dstDevice =
                  static_cast<CUdeviceptr>(out_base + out_alc.get_offset());
              get_nonaffine_strides(cuda_copy.dstPitch, cuda_copy.dstHeight, ainfo,
                                 out_alc, bytes);
            } else {
              assert(!in_nonaffine);
              log_gpudma.info() << "Affine -> Array";
              size_t bytes = out_port->iter->step_custom(bytes_left, ainfo,
                                                  false);
              if(bytes == 0) break;  // flow control or end of array
              cuda_copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
              cuda_copy.dstArray = ainfo.array;
              cuda_copy.dstXInBytes = ainfo.pos[0];
              cuda_copy.dstY = ainfo.pos[1];
              cuda_copy.dstZ = ainfo.pos[2];
              cuda_copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
              cuda_copy.srcDevice =
                  static_cast<CUdeviceptr>(in_base + in_alc.get_offset());
              get_nonaffine_strides(cuda_copy.srcPitch, cuda_copy.srcHeight, ainfo,
                                 in_alc, bytes);
            }
            cuda_copy.WidthInBytes = ainfo.width_in_bytes;
            cuda_copy.Height = ainfo.height;
            cuda_copy.Depth = ainfo.depth;
            assert(bytes_left >= cuda_copy.WidthInBytes * cuda_copy.Height * cuda_copy.Depth);
            bytes_left -= cuda_copy.WidthInBytes * cuda_copy.Height * cuda_copy.Depth;
            break;    // TODO: buffer these non-affine transfers
          }

        }
        // 3) Copy loop - Actually perform the copies enumerated earlier and track their
        // completion This logic will determine which path was ultimately chosen based on
        // the enumeration logic and should only be one launch API call, regardless of the
        // size of the batch.  These copies *must* be submitted and cannot be interrupted,
        // as we've already updated the addresslistcursor and committed to submitting them
        size_t bytes_to_fence = 0;
        if (cuda_copy.WidthInBytes != 0) {
          // First the non-affine copies
          const size_t bytes = cuda_copy.WidthInBytes * cuda_copy.Height * cuda_copy.Depth;
          CHECK_CU(CUDA_DRIVER_FNPTR(cuMemcpy3DAsync)(&cuda_copy, stream->get_stream()));
          bytes_to_fence += bytes;
        }

        if(in_gpu && in_gpu->can_access_peer(out_gpu) && transpose_copy.extents[0] != 0 &&
           transpose_copy.extents[0] <= CUDA_MAX_FIELD_BYTES) {
          stream->get_gpu()->launch_transpose_kernel(transpose_copy, min_align, stream);
          bytes_to_fence += transpose_copy.extents[0] * transpose_copy.extents[1] *
                            transpose_copy.extents[2];
        } else if(transpose_copy.extents[0] != 0) {
          CUDA_MEMCPY2D d2_copy_info;
          memset(&d2_copy_info, 0, sizeof(d2_copy_info));
          d2_copy_info.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
          d2_copy_info.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
          d2_copy_info.srcPitch = transpose_copy.src_strides[0];
          d2_copy_info.dstPitch = transpose_copy.dst_strides[0];
          d2_copy_info.WidthInBytes = transpose_copy.extents[0];
          d2_copy_info.Height = transpose_copy.extents[1];

          size_t planes = transpose_copy.extents[2];
          size_t act_planes = 0;
          while(act_planes < planes) {
            if(!stream->ok_to_submit_copy(d2_copy_info.WidthInBytes * d2_copy_info.Height,
                                          this)) {
              break;
            }

            d2_copy_info.dstDevice =
                transpose_copy.dst + act_planes * transpose_copy.dst_strides[1];
            d2_copy_info.srcDevice =
                transpose_copy.src + act_planes * transpose_copy.src_strides[1];

            log_gpudma.info() << "\t Launching 2D CE transpose srcPitch="
                              << d2_copy_info.srcPitch
                              << " dstpitch=" << d2_copy_info.dstPitch
                              << " WidthInBytes=" << d2_copy_info.WidthInBytes
                              << " Height=" << d2_copy_info.Height
                              << " dstDevice=" << d2_copy_info.dstDevice
                              << " srcDevice=" << d2_copy_info.srcDevice
                              << " memcpy_kind=" << memcpy_kind;

            CHECK_CU(
                CUDA_DRIVER_FNPTR(cuMemcpy2DAsync)(&d2_copy_info, stream->get_stream()));
            act_planes++;
          }
          // transpose sets up all 3 dims for extents.
          bytes_to_fence += transpose_copy.extents[0] * transpose_copy.extents[1] *
                            transpose_copy.extents[2];
        }

        if (copy_infos.num_rects > 1) {
          assert((in_gpu != NULL) && in_gpu->can_access_peer(out_gpu));
          // Adjust all the rectangles' sizes to account for the element size based on the calculated alignment
          for (size_t i = 0; (min_align > 1) && (i < copy_infos.num_rects); i++) {
            copy_infos.subrects[i].dst.strides[0] /= min_align;
            copy_infos.subrects[i].src.strides[0] /= min_align;
            copy_infos.subrects[i].extents[0]     /= min_align;
            copy_infos.subrects[i].volume /= min_align;
          }
          // TODO: add some heuristics here, like if some rectangles are very large, do a
          // cuMemcpy instead, possibly utilizing the copy engines or better optimized
          // kernels
          log_gpudma.info() << "\tLaunching kernel for rects=" << copy_infos.num_rects
                            << " bytes=" << copy_info_total
                            << " out_is_ipc=" << out_is_ipc;
          stream->get_gpu()->launch_batch_affine_kernel(
              &copy_infos, 3, min_align, copy_info_total / min_align, stream);
          bytes_to_fence += copy_info_total;
        } else if (copy_infos.num_rects == 1) {
          // Then the affine copies to/from the device
          // We default to cuMemcpy3D here since that's the most efficient for this copy
          AffineCopyPair<3> &copy_info = copy_infos.subrects[0];
          assert(copy_infos.num_rects == 1);

          cuda_copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
          cuda_copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
          cuda_copy.WidthInBytes = copy_info.extents[0];
          cuda_copy.Height = copy_info.extents[1];
          cuda_copy.Depth = copy_info.extents[2];

          cuda_copy.srcPitch = copy_info.src.strides[0];
          cuda_copy.srcHeight = copy_info.src.strides[1];
          cuda_copy.dstPitch = copy_info.dst.strides[0];
          cuda_copy.dstHeight = copy_info.dst.strides[1];

          cuda_copy.dstDevice = copy_info.dst.addr;
          cuda_copy.srcDevice = copy_info.src.addr;

          log_gpudma.info() << "\tLaunching 3D CE "
                            << "xd=" << std::hex << guid << std::dec << " bytes="
                            << copy_info.extents[0] * copy_info.extents[1] *
                                   copy_info.extents[2]
                            << " srcPitch=" << cuda_copy.srcPitch
                            << " srcHeight=" << cuda_copy.srcHeight
                            << " srcDevice=" << cuda_copy.srcDevice
                            << " dstPitch=" << cuda_copy.dstPitch
                            << " dstHeight=" << cuda_copy.dstHeight
                            << " dstDevice=" << cuda_copy.dstDevice
                            << " memcpy_kind=" << memcpy_kind;

          CHECK_CU(CUDA_DRIVER_FNPTR(cuMemcpy3DAsync)(&cuda_copy, stream->get_stream()));

          bytes_to_fence += copy_info.extents[0] * copy_info.extents[1] * copy_info.extents[2];
        }

        if(bytes_to_fence > 0) {
          assert(bytes_to_fence <= max_bytes);
          add_reference(); // released by transfer completion)
          log_gpudma.info() << "gpu memcpy fence: stream=" << stream << " xd=" << std::hex
                            << guid << std::dec << " bytes=" << bytes_to_fence;

          stream->add_notification(new GPUTransferCompletion(
              this, input_control.current_io_port, in_span_start, bytes_to_fence,
              output_control.current_io_port, out_span_start, bytes_to_fence));
          record_address_consumption(bytes_to_fence, bytes_to_fence);
          total_bytes += bytes_to_fence;
        }
      }

      rseqcache.flush();
      wseqcache.flush();

      return total_bytes > 0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUIndirectXferDes

    template <int N>
    static MemcpyIndirectInfo<N>
    make_indirect_info(const TransferIterator::AddressInfo &addr_info, size_t bytes,
                       uintptr_t in_base, uintptr_t out_base, uintptr_t src_ind_base,
                       uintptr_t dst_ind_base, bool do_scatter = false)
    {
      MemcpyIndirectInfo<N> memcpy_info;
      memset(&memcpy_info, 0, sizeof(MemcpyIndirectInfo<N>));
      memcpy_info.src_ind = src_ind_base;
      memcpy_info.dst_ind = dst_ind_base;
      memcpy_info.src.addr = in_base;
      memcpy_info.dst.addr = out_base;
      memcpy_info.field_size = addr_info.bytes_per_chunk;
      assert(memcpy_info.field_size);
      memcpy_info.volume = bytes / memcpy_info.field_size;
      assert(memcpy_info.volume);
      if(do_scatter) {
        memcpy_info.dst.addr += addr_info.base_offset;
      } else {
        memcpy_info.src.addr += addr_info.base_offset;
      }
      return memcpy_info;
    }

    static void launch_indirect_copy(GPU *gpu, GPUStream *stream, size_t addr_size,
                                     uintptr_t in_base, uintptr_t out_base,
                                     uintptr_t src_ind_base, uintptr_t dst_ind_base,
                                     size_t bytes,
                                     const TransferIterator::AddressInfo &addr_info)
    {
      bool do_scatter = (dst_ind_base != 0);
      if(addr_info.num_lines == 0 && addr_info.num_planes == 0) {
        MemcpyIndirectInfo<1> memcpy_info = make_indirect_info<1>(
            addr_info, bytes, in_base, out_base, src_ind_base, dst_ind_base, do_scatter);
        gpu->launch_indirect_copy_kernel(&memcpy_info, 1, addr_size,
                                         memcpy_info.field_size, memcpy_info.volume,
                                         stream);
      } else if(addr_info.num_planes == 0) {
        MemcpyIndirectInfo<2> memcpy_info = make_indirect_info<2>(
            addr_info, bytes, in_base, out_base, src_ind_base, dst_ind_base, do_scatter);
        if(do_scatter)
          memcpy_info.dst.strides[0] = addr_info.num_lines;
        else
          memcpy_info.src.strides[0] = addr_info.num_lines;
        gpu->launch_indirect_copy_kernel(&memcpy_info, 2, addr_size,
                                         memcpy_info.field_size, memcpy_info.volume,
                                         stream);
      } else {
        MemcpyIndirectInfo<3> memcpy_info = make_indirect_info<3>(
            addr_info, bytes, in_base, out_base, src_ind_base, dst_ind_base, do_scatter);
        if(do_scatter) {
          memcpy_info.dst.strides[0] = addr_info.num_lines;
          memcpy_info.dst.strides[1] = addr_info.num_planes;
        } else {
          memcpy_info.src.strides[0] = addr_info.num_lines;
          memcpy_info.src.strides[1] = addr_info.num_planes;
        }
        gpu->launch_indirect_copy_kernel(&memcpy_info, 3, addr_size,
                                         memcpy_info.field_size, memcpy_info.volume,
                                         stream);
      }
    }

    GPUIndirectXferDes::GPUIndirectXferDes(
        uintptr_t _dma_op, Channel *_channel, NodeID _launch_node, XferDesID _guid,
        const std::vector<XferDesPortInfo> &inputs_info,
        const std::vector<XferDesPortInfo> &outputs_info, int _priority,
        XferDesRedopInfo _redop_info)
      : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
                _priority, 0, 0)
    {
      kind = XFER_GPU_IN_FB;

      src_gpus.resize(inputs_info.size(), 0);
      for(size_t i = 0; i < input_ports.size(); i++) {
        src_gpus[i] = mem_to_gpu(input_ports[i].mem);
        if(input_ports[i].mem->kind == MemoryImpl::MKIND_GPUFB)
          assert(src_gpus[i]);
      }

      dst_gpus.resize(outputs_info.size(), 0);
      dst_is_ipc.resize(outputs_info.size(), false);
      for(size_t i = 0; i < output_ports.size(); i++) {
        dst_gpus[i] = mem_to_gpu(output_ports[i].mem);
        if(output_ports[i].mem->kind == MemoryImpl::MKIND_GPUFB) {
          assert(dst_gpus[i]);
        } else {
          if(NodeID(ID(output_ports[i].mem->me).memory_owner_node()) !=
             Network::my_node_id)
            dst_is_ipc[i] = true;
        }
      }
    }

    long GPUIndirectXferDes::get_requests(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    bool GPUIndirectXferDes::progress_xd(GPUIndirectChannel *channel,
                                         TimeLimit work_until)
    {
      bool did_work = false;
      // TODO: add span
      ReadSequenceCache rseqcache(this, 2 << 20);

      while(true) {
        size_t min_xfer_size = 4 << 20;
        const InstanceLayoutPieceBase *in_nonaffine, *out_nonaffine;

        size_t in_span_start = 0, out_span_start = 0;
        XferPort *in_port = 0, *out_port = 0;
        GPU *in_gpu = 0, *out_gpu = 0;
        bool out_is_ipc = false;

        if(input_control.current_io_port >= 0) {
          in_port = &input_ports[input_control.current_io_port];
          in_gpu = this->src_gpus[input_control.current_io_port];
          in_span_start = in_port->local_bytes_total;
        }

        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_gpu = this->dst_gpus[output_control.current_io_port];
          out_is_ipc = this->dst_is_ipc[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
        }

        assert(in_port != nullptr);
        assert(out_port != nullptr);

        size_t max_bytes =
            get_addresses(min_xfer_size, &rseqcache, in_nonaffine, out_nonaffine);
        log_gpudma.info() << "cuda gather/scatter copy"
                          << " xd=" << std::hex << guid << std::dec
                          << " min_xfer_size=" << min_xfer_size
                          << " max_bytes=" << max_bytes << " xd=" << std::hex
                          << out_port->peer_guid << std::dec;

        if(max_bytes == 0) {
          break;
        }

        size_t total_bytes = 0;

        uintptr_t in_base = 0;
        if(!in_nonaffine) {
          in_base = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
        }

        uintptr_t out_base = 0;
        const GPU::CudaIpcMapping *out_mapping = 0;
        if(!out_nonaffine) {
          if(out_is_ipc) {
            out_mapping = in_gpu->find_ipc_mapping(out_port->mem->me);
            assert(out_mapping);
            out_base = out_mapping->local_base;
          } else {
            out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));
          }
        }

        TransferIterator::AddressInfo addr_info;
        memset(&addr_info, 0, sizeof(TransferIterator::AddressInfo));

        size_t addr_size = 0;

        AddressListCursor &in_alc = in_port->addrcursor;
        AddressListCursor &out_alc = out_port->addrcursor;

        size_t write_ind_bytes = 0;
        uintptr_t dst_ind_base = 0;
        if(out_port->indirect_port_idx >= 0) {
          // scatter
          out_port->iter->step(max_bytes, addr_info, 0, 0);
          addr_size = out_port->iter->get_address_size();
          write_ind_bytes = (max_bytes / addr_info.bytes_per_chunk) * addr_size;

          dst_ind_base = reinterpret_cast<uintptr_t>(
              input_ports[out_port->indirect_port_idx].mem->get_direct_ptr(
                  out_port->iter->get_base_offset(), 0));

          dst_ind_base += (out_alc.get_offset() / addr_info.bytes_per_chunk) * addr_size;
        } else {
          out_base += out_alc.get_offset();
        }

        size_t read_ind_bytes = 0;
        uintptr_t src_ind_base = 0;
        if(in_port->indirect_port_idx >= 0) {
          // gather
          in_port->iter->step(max_bytes, addr_info, 0, 0);
          addr_size = in_port->iter->get_address_size();
          read_ind_bytes = (max_bytes / addr_info.bytes_per_chunk) * addr_size;

          src_ind_base = reinterpret_cast<uintptr_t>(
              input_ports[in_port->indirect_port_idx].mem->get_direct_ptr(
                  in_port->iter->get_base_offset(), 0));

          src_ind_base += (in_alc.get_offset() / addr_info.bytes_per_chunk) * addr_size;
        } else {
          in_base += in_alc.get_offset();
        }

        log_gpudma.info() << "cuda gathe/scatter bytes_per_chunk="
                          << addr_info.bytes_per_chunk
                          << " num_lines=" << addr_info.num_lines
                          << " line_stride=" << addr_info.line_stride
                          << " num_planes=" << addr_info.num_planes
                          << " plane_stride=" << addr_info.plane_stride
                          << " addr_size=" << addr_size
                          << " base_offset=" << in_port->iter->get_base_offset();

        auto stream = select_stream(out_gpu, in_gpu, out_mapping);
        AutoGPUContext agc(stream->get_gpu());

        // We can't do gather-scatter yet.
        assert(!(out_port->indirect_port_idx >= 0 && in_port->indirect_port_idx >= 0));
        assert(!in_nonaffine && !out_nonaffine);

        log_gpudma.info() << "\t launching cuda gather/scatter "
                          << "xd=" << std::hex << guid << std::dec
                          << " bytes=" << max_bytes << " addr_size=" << addr_size;
        launch_indirect_copy(in_gpu, stream, addr_size, in_base, out_base, src_ind_base,
                             dst_ind_base, max_bytes, addr_info);
        in_alc.advance(0, max_bytes);
        out_alc.advance(0, max_bytes);

        // TODO(apryakhin@): Add control flow
        total_bytes += max_bytes;
        size_t bytes_to_fence = total_bytes;
        if((total_bytes >= min_xfer_size) && work_until.is_expired()) {
          break;
        }

        if(bytes_to_fence > 0) {
          add_reference();
          log_gpudma.info() << "cuda gather/scatter fence: stream=" << stream << " "
                            << " xd=" << std::hex << guid << std::dec
                            << " bytes=" << bytes_to_fence
                            << " in_spart_start=" << in_span_start
                            << " out_span_start=" << out_span_start;

          stream->add_notification(new GPUIndirectTransferCompletion(
              this, input_control.current_io_port, in_span_start, total_bytes,
              output_control.current_io_port, out_span_start, bytes_to_fence,
              in_port->indirect_port_idx, 0, read_ind_bytes, out_port->indirect_port_idx,
              0, write_ind_bytes));
        }

        if(total_bytes > 0) {
          did_work = true;
          bool done = record_address_consumption(total_bytes, total_bytes);
          if(done || work_until.is_expired()) {
            break;
          }
        }
      }

      rseqcache.flush();
      return did_work;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUIndirectChannel

    static bool supports_scatter_gather_path(ChannelCopyInfo channel_copy_info,
                                             NodeID node)
    {
      // TODO(apryakhin): remove this condition when it's supported.
      // is_directt=false means we gather/scatter to the buffer which
      // won't be correct if ib smaller that overall size of
      // gather/scatter
      if(channel_copy_info.addr_size != sizeof(size_t) ||
         channel_copy_info.is_ranges == true) {
        return false;
      }

      if(channel_copy_info.src_mem.kind() != Memory::GPU_FB_MEM ||
         channel_copy_info.dst_mem.kind() != Memory::GPU_FB_MEM ||
         channel_copy_info.ind_mem.kind() != Memory::GPU_FB_MEM) {
        return false;
      }
      // TODO(apryakhin@): Consider checking for gpu access
      return true;
    }

    GPUIndirectChannel::GPUIndirectChannel(GPU *_src_gpu, XferDesKind _kind,
                                           BackgroundWorkManager *bgwork)
      : SingleXDQChannel<GPUIndirectChannel, GPUIndirectXferDes>(
            bgwork, _kind,
            stringbuilder() << "cuda channel (gpu=" << _src_gpu->info->index
                            << " kind=" << (int)_kind << ")")
    {
      src_gpu = _src_gpu;

      // switch out of ordered mode if multi-threaded dma is requested
      if(_src_gpu->module->config->cfg_multithread_dma)
        xdq.ordered_mode = false;

      std::vector<Memory> local_gpu_mems;
      local_gpu_mems.push_back(src_gpu->fbmem->me);
      if(src_gpu->fb_ibmem)
        local_gpu_mems.push_back(src_gpu->fb_ibmem->me);

      std::vector<Memory> peer_gpu_mems;
      peer_gpu_mems.insert(peer_gpu_mems.end(), src_gpu->peer_fbs.begin(),
                           src_gpu->peer_fbs.end());
      for(std::vector<GPU::CudaIpcMapping>::const_iterator it =
              src_gpu->cudaipc_mappings.begin();
          it != src_gpu->cudaipc_mappings.end(); ++it)
        peer_gpu_mems.push_back(it->mem);

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node &n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end(); ++it) {
        CudaDeviceMemoryInfo *cdm = (*it)->find_module_specific<CudaDeviceMemoryInfo>();
        if(!cdm)
          continue;
        if(cdm->context == src_gpu->context) {
          local_gpu_mems.push_back((*it)->me);
        } else {
          // if the other context is associated with a gpu and we've got peer
          //  access, use it
          // TODO: add option to enable peer access at this point?  might be
          //  expensive...
          if(cdm->gpu && (src_gpu->info->peers.count(cdm->gpu->info->device) > 0))
            peer_gpu_mems.push_back((*it)->me);
        }
      }

      switch(_kind) {
      case XFER_GPU_SC_IN_FB:
      {
        // self-path
        unsigned bw = 200000;   // HACK - estimate at 200 GB/s
        unsigned latency = 250; // HACK - estimate at 250 ns
        // TODO(apriakhin): Consider tweaking this value.
        unsigned frag_overhead = 200; // HACK - estimate at 2 us

        add_path(local_gpu_mems, local_gpu_mems, bw, latency, frag_overhead,
                 XFER_GPU_SC_IN_FB)
            .allow_redops()
            .set_max_dim(3);

        break;
      }

      case XFER_GPU_SC_PEER_FB:
      {
        // just do paths to peers - they'll do the other side
        unsigned bw = 50000;          // HACK - estimate at 50 GB/s
        unsigned latency = 1000;      // HACK - estimate at 1 us
        unsigned frag_overhead = 200; // HACK - estimate at 2 us

        add_path(local_gpu_mems, peer_gpu_mems, bw, latency, frag_overhead,
                 XFER_GPU_SC_PEER_FB)
            .set_max_dim(3);

        break;
      }

      default:
        assert(0);
      }
    }

    GPUIndirectChannel::~GPUIndirectChannel() {}

    Memory GPUIndirectChannel::suggest_ib_memories(Memory memory) const
    {
      if(memory.kind() != Memory::GPU_FB_MEM ||
         node != NodeID(ID(memory).memory_owner_node())) {
        Node &n = get_runtime()->nodes[node];
        for(std::vector<IBMemory *>::const_iterator it = n.ib_memories.begin();
            it != n.ib_memories.end(); ++it) {
          if((*it)->lowlevel_kind == Memory::GPU_FB_MEM) {
            return (*it)->me;
          }
        }
      }
      return Memory::NO_MEMORY;
    }

    bool GPUIndirectChannel::needs_wrapping_iterator() const { return true; }

    uint64_t GPUIndirectChannel::supports_path(
        ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
        CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
        const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
        XferDesKind *kind_ret /*= 0*/, unsigned *bw_ret /*= 0*/,
        unsigned *lat_ret /*= 0*/)
    {
      if(!supports_scatter_gather_path(channel_copy_info, node))
        return 0;
      return Channel::supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                    redop_id, total_bytes, src_frags, dst_frags, kind_ret,
                                    bw_ret, lat_ret);
    }

    XferDes *GPUIndirectChannel::create_xfer_des(
        uintptr_t dma_op, NodeID launch_node, XferDesID guid,
        const std::vector<XferDesPortInfo> &inputs_info,
        const std::vector<XferDesPortInfo> &outputs_info, int priority,
        XferDesRedopInfo redop_info, const void *fill_data, size_t fill_size,
        size_t fill_total)
    {
      // assert(redop_info.id == 0);
      assert(fill_size == 0);
      return new GPUIndirectXferDes(dma_op, this, launch_node, guid, inputs_info,
                                    outputs_info, priority, redop_info);
    }

    long GPUIndirectChannel::submit(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUIndirectRemoteChannelInfo
    //

    GPUIndirectRemoteChannelInfo::GPUIndirectRemoteChannelInfo(
        NodeID _owner, XferDesKind _kind, uintptr_t _remote_ptr,
        const std::vector<Channel::SupportedPath> &_paths)
      : SimpleRemoteChannelInfo(_owner, _kind, _remote_ptr, _paths)
    {}

    RemoteChannel *GPUIndirectRemoteChannelInfo::create_remote_channel()
    {
      GPUIndirectRemoteChannel *rc = new GPUIndirectRemoteChannel(remote_ptr);
      rc->node = owner;
      rc->kind = kind;
      rc->paths.swap(paths);
      return rc;
    }

    template <typename S>
    bool GPUIndirectRemoteChannelInfo::serialize(S &serializer) const
    {
      return ((serializer << owner) && (serializer << kind) &&
              (serializer << remote_ptr) && (serializer << paths));
    }

    template <typename S>
    /*static*/ RemoteChannelInfo *
    GPUIndirectRemoteChannelInfo::deserialize_new(S &deserializer)
    {
      NodeID owner;
      XferDesKind kind;
      uintptr_t remote_ptr;
      std::vector<Channel::SupportedPath> paths;

      if((deserializer >> owner) && (deserializer >> kind) &&
         (deserializer >> remote_ptr) && (deserializer >> paths)) {
        return new GPUIndirectRemoteChannelInfo(owner, kind, remote_ptr, paths);
      } else {
        return 0;
      }
    }

    /*static*/ Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                        GPUIndirectRemoteChannelInfo>
        GPUIndirectRemoteChannelInfo::serdez_subclass;

    RemoteChannelInfo *GPUIndirectChannel::construct_remote_info() const
    {
      return new GPUIndirectRemoteChannelInfo(node, kind,
                                              reinterpret_cast<uintptr_t>(this), paths);
    }
    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUIndirectRemoteChannel
    //

    GPUIndirectRemoteChannel::GPUIndirectRemoteChannel(uintptr_t _remote_ptr)
      : RemoteChannel(_remote_ptr)
    {}

    Memory GPUIndirectRemoteChannel::suggest_ib_memories(Memory memory) const
    {
      if(memory.kind() != Memory::GPU_FB_MEM ||
         node != NodeID(ID(memory).memory_owner_node())) {
        Node &n = get_runtime()->nodes[node];
        for(std::vector<IBMemory *>::const_iterator it = n.ib_memories.begin();
            it != n.ib_memories.end(); ++it) {
          if((*it)->lowlevel_kind == Memory::GPU_FB_MEM) {
            return (*it)->me;
          }
        }
      }
      return Memory::NO_MEMORY;
    }

    uint64_t GPUIndirectRemoteChannel::supports_path(
        ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
        CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
        const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
        XferDesKind *kind_ret /*= 0*/, unsigned *bw_ret /*= 0*/,
        unsigned *lat_ret /*= 0*/)
    {
      if(!supports_scatter_gather_path(channel_copy_info, node))
        return 0;
      return Channel::supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                    redop_id, total_bytes, src_frags, dst_frags, kind_ret,
                                    bw_ret, lat_ret);
    }

    bool GPUIndirectRemoteChannel::needs_wrapping_iterator() const { return true; }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUChannel

    GPUChannel::GPUChannel(GPU *_src_gpu, XferDesKind _kind,
                           BackgroundWorkManager *bgwork)
        : SingleXDQChannel<GPUChannel, GPUXferDes>(
              bgwork, _kind,
              stringbuilder() << "cuda channel (gpu=" << _src_gpu->info->index
                              << " kind=" << (int)_kind << ")") {
      src_gpu = _src_gpu;

      // switch out of ordered mode if multi-threaded dma is requested
      if (_src_gpu->module->config->cfg_multithread_dma)
        xdq.ordered_mode = false;

      std::vector<Memory> local_gpu_mems;
      local_gpu_mems.push_back(src_gpu->fbmem->me);
      if (src_gpu->fb_ibmem)
        local_gpu_mems.push_back(src_gpu->fb_ibmem->me);

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node &n = get_runtime()->nodes[Network::my_node_id];
      for (std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
           it != n.memories.end(); ++it) {
        CudaDeviceMemoryInfo *cdm =
            (*it)->find_module_specific<CudaDeviceMemoryInfo>();
        if (!cdm)
          continue;
        if (cdm->context == src_gpu->context) {
          local_gpu_mems.push_back((*it)->me);
        }
      }

      std::vector<Memory> mapped_cpu_mems;
      mapped_cpu_mems.insert(mapped_cpu_mems.end(),
                             src_gpu->pinned_sysmems.begin(),
                             src_gpu->pinned_sysmems.end());
      // treat managed memory as usually being on the host as well
      mapped_cpu_mems.insert(mapped_cpu_mems.end(),
                             src_gpu->managed_mems.begin(),
                             src_gpu->managed_mems.end());

      switch (_kind) {
      case XFER_GPU_TO_FB: {
        unsigned bw = src_gpu->info->pci_bandwidth;
        unsigned latency = 1000;       // HACK - estimate at 1 us
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(mapped_cpu_mems, local_gpu_mems, bw, latency, frag_overhead,
                 XFER_GPU_TO_FB)
            .set_max_dim(2); // D->H cudamemcpy3d is unrolled into 2d copies

        break;
      }

      case XFER_GPU_FROM_FB: {
        unsigned bw =
            src_gpu->info->pci_bandwidth; // HACK - estimate at 10 GB/s
        unsigned latency = 1000;          // HACK - estimate at 1 us
        unsigned frag_overhead = 2000;    // HACK - estimate at 2 us

        add_path(local_gpu_mems, mapped_cpu_mems, bw, latency, frag_overhead,
                 XFER_GPU_FROM_FB)
            .set_max_dim(2); // H->D cudamemcpy3d is unrolled into 2d copies

        break;
      }

      case XFER_GPU_IN_FB: {
        // self-path
        unsigned bw =
            src_gpu->info->logical_peer_bandwidth[_src_gpu->info->index];
        unsigned latency =
            src_gpu->info->logical_peer_latency[_src_gpu->info->index];
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(local_gpu_mems, local_gpu_mems, bw, latency, frag_overhead,
                 XFER_GPU_IN_FB)
            .set_max_dim(3);

        break;
      }

      case XFER_GPU_PEER_FB: {
        // just do paths to peers - they'll do the other side
        for (GPU *peer_gpu : src_gpu->module->gpus) {
          // Skip paths to the same GPU
          if (peer_gpu == src_gpu)
            continue;
          if (src_gpu->info->peers.find(peer_gpu->info->index) !=
              src_gpu->info->peers.end()) {
            unsigned bw = static_cast<unsigned>(
                src_gpu->info->logical_peer_bandwidth[peer_gpu->info->index]);
            unsigned latency = static_cast<unsigned>(
                src_gpu->info->logical_peer_latency[peer_gpu->info->index]);
            unsigned frag_overhead = 2000; // HACK - estimate at 2 us
            if (peer_gpu->fbmem != nullptr) {
              add_path(local_gpu_mems, peer_gpu->fbmem->me, bw, latency,
                       frag_overhead, XFER_GPU_PEER_FB)
                  .set_max_dim(3);
            }
            if (peer_gpu->fb_ibmem != nullptr) {
              add_path(local_gpu_mems, peer_gpu->fb_ibmem->me, bw, latency,
                       frag_overhead, XFER_GPU_PEER_FB)
                  .set_max_dim(3);
            }
            // Add paths for peer managed memories
            for (Realm::Memory mem : peer_gpu->managed_mems) {
              bw = src_gpu->info->pci_bandwidth; // HACK - be pessimestic with
                                                 // managed memory
              latency = 2000;                    // HACK - estimate at 2 us
              add_path(local_gpu_mems, mem, bw, latency, frag_overhead,
                       XFER_GPU_PEER_FB)
                  .set_max_dim(3);
            }
          }
        }

        for (const GPU::CudaIpcMapping &mapping : src_gpu->cudaipc_mappings) {
          unsigned bw = src_gpu->info->pci_bandwidth;
          unsigned latency = 1000;       // HACK - estimate at 1 us
          unsigned frag_overhead = 2000; // HACK - estimate at 2 us
          if (mapping.src_gpu != nullptr) {
            bw = static_cast<unsigned>(
                src_gpu->info
                    ->logical_peer_bandwidth[mapping.src_gpu->info->index]);
            latency = static_cast<unsigned>(
                src_gpu->info
                    ->logical_peer_latency[mapping.src_gpu->info->index]);
          }
          add_path(local_gpu_mems, mapping.mem, bw, latency, frag_overhead,
                   XFER_GPU_PEER_FB)
              .set_max_dim(3);
        }

        // Add paths for GPU dynamic memories
        Node &node = get_runtime()->nodes[Network::my_node_id];
        for (MemoryImpl *memImpl : node.memories) {
          if (memImpl->get_kind() == Realm::Memory::GPU_DYNAMIC_MEM) {
            CudaDeviceMemoryInfo *cdm =
                memImpl->find_module_specific<CudaDeviceMemoryInfo>();
            if (cdm->gpu == src_gpu) {
              continue;
            }
            if (src_gpu->info->peers.find(cdm->gpu->info->index) ==
                src_gpu->info->peers.end()) {
              continue;
            }
            unsigned bw = static_cast<unsigned>(
                src_gpu->info->logical_peer_bandwidth[cdm->gpu->info->index]);
            unsigned latency = static_cast<unsigned>(
                src_gpu->info->logical_peer_latency[cdm->gpu->info->index]);
            unsigned frag_overhead = 2000; // HACK - estimate at 2 us
            add_path(local_gpu_mems, memImpl->me, bw, latency, frag_overhead,
                     XFER_GPU_PEER_FB)
                .set_max_dim(3);
          }
        }

        break;
      }

      default:
        assert(0);
      }
    }

    GPUChannel::~GPUChannel()
    {
    }

    XferDes *GPUChannel::create_xfer_des(uintptr_t dma_op,
                                         NodeID launch_node,
                                         XferDesID guid,
                                         const std::vector<XferDesPortInfo>& inputs_info,
                                         const std::vector<XferDesPortInfo>& outputs_info,
                                         int priority,
                                         XferDesRedopInfo redop_info,
                                         const void *fill_data,
                                         size_t fill_size,
                                         size_t fill_total)
    {
      assert(redop_info.id == 0);
      assert(fill_size == 0);
      return new GPUXferDes(dma_op, this, launch_node, guid,
                            inputs_info, outputs_info,
                            priority);
    }

    long GPUChannel::submit(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUCompletionEvent

      void GPUCompletionEvent::request_completed(void)
      {
	req->xd->notify_request_read_done(req);
	req->xd->notify_request_write_done(req);
      }

      ////////////////////////////////////////////////////////////////////////
      //
      // class GPUIndirectTransferCompletion

      GPUIndirectTransferCompletion::GPUIndirectTransferCompletion(
          XferDes *_xd, int _read_port_idx, size_t _read_offset, size_t _read_size,
          int _write_port_idx, size_t _write_offset, size_t _write_size,
          int _read_ind_port_idx, size_t _read_ind_offset, size_t _read_ind_size,
          int _write_ind_port_idx, size_t _write_ind_offset, size_t _write_ind_size)
        : xd(_xd)
        , read_port_idx(_read_port_idx)
        , read_offset(_read_offset)
        , read_size(_read_size)
        , read_ind_port_idx(_read_ind_port_idx)
        , read_ind_offset(_read_ind_offset)
        , read_ind_size(_read_ind_size)
        , write_port_idx(_write_port_idx)
        , write_offset(_write_offset)
        , write_size(_write_size)
        , write_ind_port_idx(_write_ind_port_idx)
        , write_ind_offset(_write_ind_offset)
        , write_ind_size(_write_ind_size)
      {}

      void GPUIndirectTransferCompletion::request_completed(void)
      {

        log_gpudma.info() << "gpu gather complete: xd=" << std::hex << xd->guid
                          << std::dec << " read=" << read_port_idx << "/" << read_offset
                          << " write=" << write_port_idx << "/" << write_offset
                          << " bytes=" << write_size;

        if(read_ind_port_idx >= 0) {
          XferDes::XferPort &iip = xd->input_ports[read_ind_port_idx];
          xd->update_bytes_read(read_ind_port_idx, iip.local_bytes_total, read_ind_size);
          iip.local_bytes_total += read_ind_size;
        }

        if(write_ind_port_idx >= 0) {
          XferDes::XferPort &iip = xd->input_ports[write_ind_port_idx];
          xd->update_bytes_read(write_ind_port_idx, iip.local_bytes_total,
                                write_ind_size);
          iip.local_bytes_total += write_ind_size;
        }

        if(read_port_idx >= 0) {
          xd->update_bytes_read(read_port_idx, read_offset, read_size);
        }
        if(write_port_idx >= 0) {
          xd->update_bytes_write(write_port_idx, write_offset, write_size);
        }

        xd->update_progress();
        xd->remove_reference();
        // TODO(apryakhin@): Do we need to update this?
        delete this;
      }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUTransferCompletion

    GPUTransferCompletion::GPUTransferCompletion(XferDes *_xd,
                                                 int _read_port_idx,
                                                 size_t _read_offset,
                                                 size_t _read_size,
                                                 int _write_port_idx,
                                                 size_t _write_offset,
                                                 size_t _write_size)
      : xd(_xd)
      , read_port_idx(_read_port_idx)
      , read_offset(_read_offset)
      , read_size(_read_size)
      , write_port_idx(_write_port_idx)
      , write_offset(_write_offset)
      , write_size(_write_size)
    {}

    void GPUTransferCompletion::request_completed(void)
    {
      log_gpudma.info() << "gpu memcpy complete: xd=" << std::hex << xd->guid << std::dec
                        << " read=" << read_port_idx << "/" << read_offset
                        << " write=" << write_port_idx << "/" << write_offset
                        << " bytes=" << write_size;
      if(read_port_idx >= 0)
        xd->update_bytes_read(read_port_idx, read_offset, read_size);
      if(write_port_idx >= 0)
        xd->update_bytes_write(write_port_idx, write_offset, write_size);
      xd->remove_reference();
      delete this;  // TODO: recycle these!
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUfillXferDes

    GPUfillXferDes::GPUfillXferDes(uintptr_t _dma_op, Channel *_channel,
                                   NodeID _launch_node, XferDesID _guid,
                                   const std::vector<XferDesPortInfo>& inputs_info,
                                   const std::vector<XferDesPortInfo>& outputs_info,
                                   int _priority,
                                   const void *_fill_data, size_t _fill_size,
                                   size_t _fill_total)
      : XferDes(_dma_op, _channel, _launch_node, _guid,
                inputs_info, outputs_info,
                _priority, _fill_data, _fill_size)
    {
      kind = XFER_GPU_IN_FB;

      // no direct input data for us, but we know how much data to produce
      //  (in case the output is an intermediate buffer)
      assert(input_control.control_port_idx == -1);
      input_control.current_io_port = -1;
      input_control.remaining_count = _fill_total;
      input_control.eos_received = true;

      // cuda memsets are ideally 8/16/32 bits, so try to _reduce_ the fill
      //  size if there's duplication
      if((fill_size > 1) && (memcmp(fill_data,
                                    static_cast<char *>(fill_data) + 1,
                                    fill_size - 1) == 0))
        reduced_fill_size = 1;  // can use memset8
      else if((fill_size > 2) && ((fill_size >> 1) == 0) &&
              (memcmp(fill_data,
                      static_cast<char *>(fill_data) + 2,
                      fill_size - 2) == 0))
        reduced_fill_size = 2;  // can use memset16
      else if((fill_size > 4) && ((fill_size >> 2) == 0) &&
              (memcmp(fill_data,
                      static_cast<char *>(fill_data) + 4,
                      fill_size - 4) == 0))
        reduced_fill_size = 4;  // can use memset32
      else
        reduced_fill_size = fill_size; // will have to do it in pieces
    }

    long GPUfillXferDes::get_requests(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    bool GPUfillXferDes::progress_xd(GPUfillChannel *channel,
                                     TimeLimit work_until)
    {
      bool did_work = false;
      ReadSequenceCache rseqcache(this, 2 << 20);
      WriteSequenceCache wseqcache(this, 2 << 20);

      while(true) {
        size_t min_xfer_size = 4096;  // TODO: make controllable
        size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
        if(max_bytes == 0)
          break;

        XferPort *out_port = 0;
        size_t out_span_start = 0;
        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
        }

        bool done = false;

        size_t total_bytes = 0;
        if(out_port != 0) {
          // input and output both exist - transfer what we can
          log_xd.info() << "gpufill chunk: min=" << min_xfer_size
                        << " max=" << max_bytes;

          uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

          AutoGPUContext agc(channel->gpu);
          GPUStream *stream = channel->gpu->get_next_d2d_stream();

          while(total_bytes < max_bytes) {
            AddressListCursor& out_alc = out_port->addrcursor;

            uintptr_t out_offset = out_alc.get_offset();

            // the reported dim is reduced for partially consumed address
            //  ranges - whatever we get can be assumed to be regular
            int out_dim = out_alc.get_dim();

            // fast paths for 8/16/32 bit memsets exist for 1-D and 2-D
            switch(reduced_fill_size) {
            case 1: {
              // memset8
              uint8_t fill_u8;
              memcpy(&fill_u8, fill_data, 1);
              if(out_dim == 1) {
                size_t bytes = out_alc.remaining(0);
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD8Async)
                          (CUdeviceptr(out_base + out_offset),
                           fill_u8,
                           bytes,
                           stream->get_stream()) );
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t bytes = out_alc.remaining(0);
                size_t lines = out_alc.remaining(1);
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD2D8Async)
                          (CUdeviceptr(out_base + out_offset),
                           out_alc.get_stride(1),
                           fill_u8,
                           bytes, lines,
                           stream->get_stream()) );
                out_alc.advance(1, lines);
                total_bytes += bytes * lines;
              }
              break;
            }

            case 2: {
              // memset16
              uint16_t fill_u16;
              memcpy(&fill_u16, fill_data, 2);
              if(out_dim == 1) {
                size_t bytes = out_alc.remaining(0);
#ifdef DEBUG_REALM
                assert((bytes & 1) == 0);
#endif
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD16Async)
                          (CUdeviceptr(out_base + out_offset),
                           fill_u16,
                           bytes >> 1,
                           stream->get_stream()) );
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t bytes = out_alc.remaining(0);
                size_t lines = out_alc.remaining(1);
#ifdef DEBUG_REALM
                assert((bytes & 1) == 0);
                assert((out_alc.get_stride(1) & 1) == 0);
#endif
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD2D16Async)
                          (CUdeviceptr(out_base + out_offset),
                           out_alc.get_stride(1),
                           fill_u16,
                           bytes >> 1, lines,
                           stream->get_stream()) );
                out_alc.advance(1, lines);
                total_bytes += bytes * lines;
              }
              break;
            }

            case 4: {
              // memset32
              uint32_t fill_u32;
              memcpy(&fill_u32, fill_data, 4);
              if(out_dim == 1) {
                size_t bytes = out_alc.remaining(0);
#ifdef DEBUG_REALM
                assert((bytes & 3) == 0);
#endif
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD32Async)
                          (CUdeviceptr(out_base + out_offset),
                           fill_u32,
                           bytes >> 2,
                           stream->get_stream()) );
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t bytes = out_alc.remaining(0);
                size_t lines = out_alc.remaining(1);
#ifdef DEBUG_REALM
                assert((bytes & 3) == 0);
                assert((out_alc.get_stride(1) & 3) == 0);
#endif
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD2D32Async)
                          (CUdeviceptr(out_base + out_offset),
                           out_alc.get_stride(1),
                           fill_u32,
                           bytes >> 2, lines,
                           stream->get_stream()) );
                out_alc.advance(1, lines);
                total_bytes += bytes * lines;
              }
              break;
            }

            default: {
              // more general approach - use strided 2d copies to fill the first
              //  line, and then we can use logarithmic doublings to deal with
              //  multiple lines and/or planes
              size_t bytes = out_alc.remaining(0);
              size_t elems = bytes / reduced_fill_size;
#ifdef DEBUG_REALM
              assert((bytes % reduced_fill_size) == 0);
#endif
              // these strided 2d copies have very poor performance, so don't
              //  do more than ~32kb worth of them and switch to the repeated
              //  doubling of them for very long lines
              size_t fill_elems = ((bytes <= 32768)            ? elems :
                                   (reduced_fill_size > 32768) ? 1 :
                                                                 (32768 / reduced_fill_size));

              size_t partial_bytes = 0;
              if((reduced_fill_size & 3) == 0) {
                // 32-bit partial fills allowed
                while(partial_bytes <= (reduced_fill_size - 4)) {
                  uint32_t fill_u32;
                  memcpy(&fill_u32,
                         reinterpret_cast<const uint8_t *>(fill_data) + partial_bytes,
                         4);
                  CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD2D32Async)
                            (CUdeviceptr(out_base + out_offset + partial_bytes),
                             reduced_fill_size,
                             fill_u32,
                             1 /*"width"*/, fill_elems /*"height"*/,
                             stream->get_stream()) );
                  partial_bytes += 4;
                }
              }
              if((reduced_fill_size & 1) == 0) {
                // 16-bit partial fills allowed
                while(partial_bytes <= (reduced_fill_size - 2)) {
                  uint16_t fill_u16;
                  memcpy(&fill_u16,
                         reinterpret_cast<const uint8_t *>(fill_data) + partial_bytes,
                         2);
                  CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD2D16Async)
                            (CUdeviceptr(out_base + out_offset + partial_bytes),
                             reduced_fill_size,
                             fill_u16,
                             1 /*"width"*/, fill_elems /*"height"*/,
                             stream->get_stream()) );
                  partial_bytes += 2;
                }
              }
              // leftover or unaligned bytes are done 8 bits at a time
              while(partial_bytes < reduced_fill_size) {
                uint8_t fill_u8;
                memcpy(&fill_u8,
                       reinterpret_cast<const uint8_t *>(fill_data) + partial_bytes,
                       1);
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD2D8Async)
                          (CUdeviceptr(out_base + out_offset + partial_bytes),
                           reduced_fill_size,
                           fill_u8,
                           1 /*"width"*/, fill_elems /*"height"*/,
                           stream->get_stream()) );
                partial_bytes += 1;
              }

              while(fill_elems < elems) {
                size_t todo = std::min(fill_elems, elems - fill_elems);
                CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyAsync)
                          (CUdeviceptr(out_base + out_offset +
                                       (fill_elems * reduced_fill_size)),
                           CUdeviceptr(out_base + out_offset),
                           todo * reduced_fill_size,
                           stream->get_stream()) );
                fill_elems += todo;
              }

              if(out_dim == 1) {
                // all done
                out_alc.advance(0, bytes);
                total_bytes += bytes;
              } else {
                size_t lines = out_alc.remaining(1);
                size_t lstride = out_alc.get_stride(1);

                CUDA_MEMCPY2D copy2d;
                copy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                copy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                copy2d.srcDevice = CUdeviceptr(out_base + out_offset);
                copy2d.srcHost = 0;
                copy2d.srcPitch = lstride;
                copy2d.srcY = 0;
                copy2d.srcXInBytes = 0;
                copy2d.dstHost = 0;
                copy2d.dstPitch = lstride;
                copy2d.dstY = 0;
                copy2d.dstXInBytes = 0;
                copy2d.WidthInBytes = bytes;

                size_t lines_done = 1;  // first line already valid
                while(lines_done < lines) {
                  size_t todo = std::min(lines_done, lines - lines_done);
                  copy2d.dstDevice = CUdeviceptr(out_base + out_offset +
                                                 (lines_done * lstride));
                  copy2d.Height = todo;
                  CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpy2DAsync)
                            (&copy2d, stream->get_stream()) );
                  lines_done += todo;
                }

                if(out_dim == 2) {
                  out_alc.advance(1, lines);
                  total_bytes += bytes * lines;
                } else {
                  size_t planes = out_alc.remaining(2);
                  size_t pstride = out_alc.get_stride(2);

                  // logarithmic version requires that pstride be a multiple of
                  //  lstride
                  if((pstride % lstride) == 0) {
                    CUDA_MEMCPY3D copy3d;

                    copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                    copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                    copy3d.srcDevice = CUdeviceptr(out_base + out_offset);
                    copy3d.srcHost = 0;
                    copy3d.srcPitch = lstride;
                    copy3d.srcHeight = pstride / lstride;
                    copy3d.srcY = 0;
                    copy3d.srcZ = 0;
                    copy3d.srcXInBytes = 0;
                    copy3d.srcLOD = 0;
                    copy3d.dstHost = 0;
                    copy3d.dstPitch = lstride;
                    copy3d.dstHeight = pstride / lstride;
                    copy3d.dstY = 0;
                    copy3d.dstZ = 0;
                    copy3d.dstXInBytes = 0;
                    copy3d.dstLOD = 0;
                    copy3d.WidthInBytes = bytes;
                    copy3d.Height = lines;

                    size_t planes_done = 1;  // first plane already valid
                    while(planes_done < planes) {
                      size_t todo = std::min(planes_done, planes - planes_done);
                      copy3d.dstDevice = CUdeviceptr(out_base + out_offset +
                                                     (planes_done * pstride));
                      copy3d.Depth = todo;
                      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpy3DAsync)
                                (&copy3d, stream->get_stream()) );
                      planes_done += todo;
                    }

                    out_alc.advance(2, planes);
                    total_bytes += bytes * lines * planes;
                  } else {
                    // plane-at-a-time fallback - can reuse most of copy2d
                    //  setup above
                    copy2d.Height = lines;

                    for(size_t p = 1; p < planes; p++) {
                      copy2d.dstDevice = CUdeviceptr(out_base + out_offset +
                                                     (p * pstride));
                      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpy2DAsync)
                                (&copy2d, stream->get_stream()) );
                    }
                  }
                }
              }
              break;
            }
            }

            // stop if it's been too long, but make sure we do at least the
            //  minimum number of bytes
            if((total_bytes >= min_xfer_size) && work_until.is_expired()) break;
          }

          // however many fills/copies we submitted, put in a single fence that
          //  will tell us that they're all done
          add_reference(); // released by transfer completion
          stream->add_notification(new GPUTransferCompletion(this,
                                                             -1, 0, 0,
                                                             output_control.current_io_port,
                                                             out_span_start,
                                                             total_bytes));
	  out_span_start += total_bytes;
        }

        done = record_address_consumption(total_bytes, total_bytes);

        did_work = true;

        if(done || work_until.is_expired())
          break;
      }

      rseqcache.flush();

      return did_work;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUfillChannel

    GPUfillChannel::GPUfillChannel(GPU *_gpu, BackgroundWorkManager *bgwork)
      : SingleXDQChannel<GPUfillChannel,GPUfillXferDes>(bgwork,
                                                        XFER_GPU_IN_FB,
                                                        stringbuilder() << "cuda fill channel (gpu=" << _gpu->info->index << ")")
      , gpu(_gpu)
    {
      std::vector<Memory> local_gpu_mems;
      local_gpu_mems.push_back(gpu->fbmem->me);

      // look for any other local memories that belong to our context
      const Node& n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end();
          ++it) {
        CudaDeviceMemoryInfo *cdm = (*it)->find_module_specific<CudaDeviceMemoryInfo>();
        if(!cdm) continue;
        if(cdm->context != gpu->context) continue;
        local_gpu_mems.push_back((*it)->me);
      }

      unsigned bw = gpu->info->logical_peer_bandwidth[gpu->info->index];
      unsigned latency = gpu->info->logical_peer_latency[gpu->info->index];
      unsigned frag_overhead = 2000;  // HACK - estimate at 2 us

      add_path(Memory::NO_MEMORY, local_gpu_mems,
               bw, latency, frag_overhead, XFER_GPU_IN_FB)
        .set_max_dim(2);

      xdq.add_to_manager(bgwork);
    }

    XferDes *GPUfillChannel::create_xfer_des(uintptr_t dma_op,
                                             NodeID launch_node,
                                             XferDesID guid,
                                             const std::vector<XferDesPortInfo>& inputs_info,
                                             const std::vector<XferDesPortInfo>& outputs_info,
                                             int priority,
                                             XferDesRedopInfo redop_info,
                                             const void *fill_data,
                                             size_t fill_size,
                                             size_t fill_total)
    {
      assert(redop_info.id == 0);
      return new GPUfillXferDes(dma_op, this, launch_node, guid,
                                inputs_info, outputs_info,
                                priority,
                                fill_data, fill_size, fill_total);
    }

    long GPUfillChannel::submit(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceXferDes

    GPUreduceXferDes::GPUreduceXferDes(uintptr_t _dma_op, Channel *_channel,
                                       NodeID _launch_node, XferDesID _guid,
                                       const std::vector<XferDesPortInfo>& inputs_info,
                                       const std::vector<XferDesPortInfo>& outputs_info,
                                       int _priority,
                                       XferDesRedopInfo _redop_info)
      : XferDes(_dma_op, _channel, _launch_node, _guid,
                inputs_info, outputs_info,
                _priority, 0, 0)
      , redop_info(_redop_info)
    {
      kind = XFER_GPU_IN_FB;
      redop = get_runtime()->reduce_op_table.get(redop_info.id, 0);
      kernel = 0;
      kernel_host_proxy = nullptr;
      assert(redop);

      GPU *gpu = checked_cast<GPUreduceChannel *>(channel)->gpu;

      // select reduction kernel now - translate to CUfunction if possible
      void *host_proxy =
          (redop_info.is_fold
               ? (redop_info.is_exclusive ? redop->cuda_fold_excl_fn
                                          : redop->cuda_fold_nonexcl_fn)
               : (redop_info.is_exclusive ? redop->cuda_apply_excl_fn
                                          : redop->cuda_apply_nonexcl_fn));
#ifdef REALM_USE_CUDART_HIJACK
      // we have the host->device mapping table for functions
      kernel = gpu->lookup_function(host_proxy);
#else
      if (redop->cudaGetFuncBySymbol_fn != 0) {
        // we can ask the runtime to perform the mapping for us
        gpu->push_context();
        CHECK_CUDART(reinterpret_cast<PFN_cudaGetFuncBySymbol>(
            redop->cudaGetFuncBySymbol_fn)((void **)&kernel, host_proxy));
        gpu->pop_context();
      } else {
        // no way to ask the runtime to perform the mapping, so we'll have
        //  to actually launch the kernels with the runtime API using the launch
        //  kernel function provided
        kernel_host_proxy = host_proxy;
        assert(redop->cudaLaunchKernel_fn != 0);
      }
#endif

      stream = gpu->get_next_d2d_stream();
    }

    long GPUreduceXferDes::get_requests(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    bool GPUreduceXferDes::progress_xd(GPUreduceChannel *channel,
                                     TimeLimit work_until)
    {
      bool did_work = false;
      ReadSequenceCache rseqcache(this, 2 << 20);
      ReadSequenceCache wseqcache(this, 2 << 20);

      const size_t in_elem_size = redop->sizeof_rhs;
      const size_t out_elem_size = (redop_info.is_fold ? redop->sizeof_rhs : redop->sizeof_lhs);
      assert(redop_info.in_place);  // TODO: support for out-of-place reduces

      struct KernelArgs {
        uintptr_t dst_base, dst_stride;
        uintptr_t src_base, src_stride;
        uintptr_t count;
      };
      KernelArgs *args = 0; // allocate on demand
      size_t args_size = sizeof(KernelArgs) + redop->sizeof_userdata;

      while(true) {
        size_t min_xfer_size = 4096;  // TODO: make controllable
        size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
        if(max_bytes == 0)
          break;

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

        // have to count in terms of elements, which requires redoing some math
        //  if in/out sizes do not match
        size_t max_elems;
        if(in_elem_size == out_elem_size) {
          max_elems = max_bytes / in_elem_size;
        } else {
          max_elems = std::min(input_control.remaining_count / in_elem_size,
                               output_control.remaining_count / out_elem_size);
          if(in_port != 0) {
            max_elems = std::min(max_elems,
                                 in_port->addrlist.bytes_pending() / in_elem_size);
            if(in_port->peer_guid != XFERDES_NO_GUID) {
              size_t read_bytes_avail = in_port->seq_remote.span_exists(in_port->local_bytes_total,
                                                                        (max_elems * in_elem_size));
              max_elems = std::min(max_elems,
                                   (read_bytes_avail / in_elem_size));
            }
          }
          if(out_port != 0) {
            max_elems = std::min(max_elems,
                                 out_port->addrlist.bytes_pending() / out_elem_size);
            // no support for reducing into an intermediate buffer
            assert(out_port->peer_guid == XFERDES_NO_GUID);
          }
        }

        size_t total_elems = 0;
        if(in_port != 0) {
          if(out_port != 0) {
            // input and output both exist - transfer what we can
            log_xd.info() << "gpureduce chunk: min=" << min_xfer_size
                          << " max_elems=" << max_elems;

            uintptr_t in_base = reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
            uintptr_t out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

            while(total_elems < max_elems) {
              AddressListCursor& in_alc = in_port->addrcursor;
              AddressListCursor& out_alc = out_port->addrcursor;

              uintptr_t in_offset = in_alc.get_offset();
              uintptr_t out_offset = out_alc.get_offset();

              // the reported dim is reduced for partially consumed address
              //  ranges - whatever we get can be assumed to be regular
              int in_dim = in_alc.get_dim();
              int out_dim = out_alc.get_dim();

              // the current reduction op interface can reduce multiple elements
              //  with a fixed address stride, which looks to us like either
              //  1D (stride = elem_size), or 2D with 1 elem/line

              size_t icount = in_alc.remaining(0) / in_elem_size;
              size_t ocount = out_alc.remaining(0) / out_elem_size;
              size_t istride, ostride;
              if((in_dim > 1) && (icount == 1)) {
                in_dim = 2;
                icount = in_alc.remaining(1);
                istride = in_alc.get_stride(1);
              } else {
                in_dim = 1;
                istride = in_elem_size;
              }
              if((out_dim > 1) && (ocount == 1)) {
                out_dim = 2;
                ocount = out_alc.remaining(1);
                ostride = out_alc.get_stride(1);
              } else {
                out_dim = 1;
                ostride = out_elem_size;
              }

              size_t elems_left = max_elems - total_elems;
              size_t elems = std::min(std::min(icount, ocount), elems_left);
              assert(elems > 0);

              // allocate kernel arg structure if this is our first call
              if(!args) {
                args = static_cast<KernelArgs *>(alloca(args_size));
                if(redop->sizeof_userdata)
                  memcpy(args+1, redop->userdata, redop->sizeof_userdata);
              }

              args->dst_base = out_base + out_offset;
              args->dst_stride = ostride;
              args->src_base = in_base + in_offset;
              args->src_stride = istride;
              args->count = elems;

              size_t threads_per_block = 256;
              size_t blocks_per_grid = 1 + ((elems - 1) / threads_per_block);

              {
                AutoGPUContext agc(channel->gpu);

                if (kernel != 0) {
                  void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, args,
                                   CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                                   CU_LAUNCH_PARAM_END};

                  CHECK_CU(CUDA_DRIVER_FNPTR(cuLaunchKernel)(
                      kernel, blocks_per_grid, 1, 1, threads_per_block, 1, 1,
                      0 /*sharedmem*/, stream->get_stream(), 0 /*params*/,
                      extra));
                } else {
                  void *params[] = {&args->dst_base, &args->dst_stride,
                                    &args->src_base, &args->src_stride,
                                    &args->count,    args + 1};
                  assert(redop->cudaLaunchKernel_fn != 0);
                  CHECK_CUDART(reinterpret_cast<PFN_cudaLaunchKernel>(
                      redop->cudaLaunchKernel_fn)(
                      kernel_host_proxy, dim3(blocks_per_grid, 1, 1),
                      dim3(threads_per_block, 1, 1), params, 0 /*sharedMem*/,
                      stream->get_stream()));
                }

                // insert fence to track completion of reduction kernel
                add_reference(); // released by transfer completion
                stream->add_notification(new GPUTransferCompletion(this,
                                                                   input_control.current_io_port,
                                                                   in_span_start,
                                                                   elems * in_elem_size,
                                                                   output_control.current_io_port,
                                                                   out_span_start,
                                                                   elems * out_elem_size));
              }

              in_span_start += elems * in_elem_size;
              out_span_start += elems * out_elem_size;

              in_alc.advance(in_dim-1,
                             elems * ((in_dim == 1) ? in_elem_size : 1));
              out_alc.advance(out_dim-1,
                              elems * ((out_dim == 1) ? out_elem_size : 1));

#ifdef DEBUG_REALM
              assert(elems <= elems_left);
#endif
              total_elems += elems;

              // stop if it's been too long, but make sure we do at least the
              //  minimum number of bytes
              if(((total_elems * in_elem_size) >= min_xfer_size) &&
                 work_until.is_expired()) break;
            }
          } else {
            // input but no output, so skip input bytes
            total_elems = max_elems;
            in_port->addrcursor.skip_bytes(total_elems * in_elem_size);

            rseqcache.add_span(input_control.current_io_port,
                               in_span_start, total_elems * in_elem_size);
            in_span_start += total_elems * in_elem_size;
          }
        } else {
          if(out_port != 0) {
            // output but no input, so skip output bytes
            total_elems = max_elems;
            out_port->addrcursor.skip_bytes(total_elems * out_elem_size);

            wseqcache.add_span(output_control.current_io_port,
                               out_span_start, total_elems * out_elem_size);
            out_span_start += total_elems * out_elem_size;
          } else {
            // skipping both input and output is possible for simultaneous
            //  gather+scatter
            total_elems = max_elems;
          }
        }

        bool done = record_address_consumption(total_elems * in_elem_size,
                                               total_elems * out_elem_size);

        did_work = true;

        if(done || work_until.is_expired())
          break;
      }

      rseqcache.flush();
      wseqcache.flush();

      return did_work;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceChannel

    GPUreduceChannel::GPUreduceChannel(GPU *_gpu, BackgroundWorkManager *bgwork)
      : SingleXDQChannel<GPUreduceChannel,GPUreduceXferDes>(bgwork,
                                                            XFER_GPU_IN_FB,
                                                            stringbuilder() << "cuda reduce channel (gpu=" << _gpu->info->index << ")")
      , gpu(_gpu)
    {
      std::vector<Memory> local_gpu_mems;
      local_gpu_mems.push_back(gpu->fbmem->me);
      if(gpu->fb_ibmem)
        local_gpu_mems.push_back(gpu->fb_ibmem->me);

      std::vector<Memory> peer_gpu_mems;
      peer_gpu_mems.insert(peer_gpu_mems.end(),
                           gpu->peer_fbs.begin(),
                           gpu->peer_fbs.end());

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node& n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end();
          ++it) {
        CudaDeviceMemoryInfo *cdm = (*it)->find_module_specific<CudaDeviceMemoryInfo>();
        if(!cdm) continue;
        if(cdm->context == gpu->context) {
          local_gpu_mems.push_back((*it)->me);
        } else {
          // if the other context is associated with a gpu and we've got peer
          //  access, use it
          // TODO: add option to enable peer access at this point?  might be
          //  expensive...
          if(cdm->gpu && (gpu->info->peers.count(cdm->gpu->info->index) > 0))
            peer_gpu_mems.push_back((*it)->me);
        }
      }

      std::vector<Memory> mapped_cpu_mems;
      mapped_cpu_mems.insert(mapped_cpu_mems.end(),
                             gpu->pinned_sysmems.begin(),
                             gpu->pinned_sysmems.end());
      // treat managed memory as usually being on the host as well
      mapped_cpu_mems.insert(mapped_cpu_mems.end(),
                             gpu->managed_mems.begin(),
                             gpu->managed_mems.end());

      // intra-FB reduction
      {
        unsigned bw = gpu->info->logical_peer_bandwidth[gpu->info->index];
        unsigned latency = gpu->info->logical_peer_latency[gpu->info->index];
        unsigned frag_overhead = 2000;  // HACK - estimate at 2 us

        add_path(local_gpu_mems,
                 local_gpu_mems,
                 bw, latency, frag_overhead, XFER_GPU_IN_FB)
          .allow_redops();
      }

      // zero-copy to FB (no need for intermediate buffer in FB)
      {
        unsigned bw = gpu->info->pci_bandwidth;
        unsigned latency = 1000;  // HACK - estimate at 1 us
        unsigned frag_overhead = 2000;  // HACK - estimate at 2 us

        add_path(mapped_cpu_mems,
                 local_gpu_mems,
                 bw, latency, frag_overhead, XFER_GPU_TO_FB)
          .allow_redops();
      }

      // unlike normal cuda p2p copies where we want to push from the source,
      //  reductions are always sent to the destination memory's gpu to keep the
      //  RMW loop as tight as possible
      {
        unsigned bw = 50000;  // HACK - estimate at 50 GB/s
        unsigned latency = 2000;  // HACK - estimate at 1 us
        unsigned frag_overhead = 2000;  // HACK - estimate at 2 us

        add_path(peer_gpu_mems,
                 local_gpu_mems,
                 bw, latency, frag_overhead, XFER_GPU_PEER_FB)
          .allow_redops();
      }

      xdq.add_to_manager(bgwork);
    }

    /*static*/ bool GPUreduceChannel::is_gpu_redop(ReductionOpID redop_id)
    {
      if(redop_id == 0)
        return false;

      ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      assert(redop);

      // there's four different kernels, but they should be all or nothing, so
      //  just check one
      if(!redop->cuda_apply_excl_fn)
        return false;

      return true;
    }

    uint64_t GPUreduceChannel::supports_path(ChannelCopyInfo channel_copy_info,
                                             CustomSerdezID src_serdez_id,
                                             CustomSerdezID dst_serdez_id,
                                             ReductionOpID redop_id,
                                             size_t total_bytes,
                                             const std::vector<size_t> *src_frags,
                                             const std::vector<size_t> *dst_frags,
                                             XferDesKind *kind_ret /*= 0*/,
                                             unsigned *bw_ret /*= 0*/,
                                             unsigned *lat_ret /*= 0*/)
    {
      // first check that we have a reduction op (if not, we want the cudamemcpy
      //   path to pick this up instead) and that it has cuda kernels available
      if(!is_gpu_redop(redop_id))
        return 0;

      // then delegate to the normal supports_path logic
      return Channel::supports_path(channel_copy_info,
                                    src_serdez_id, dst_serdez_id, redop_id,
                                    total_bytes, src_frags, dst_frags,
                                    kind_ret, bw_ret, lat_ret);
    }

    RemoteChannelInfo *GPUreduceChannel::construct_remote_info() const
    {
      return new GPUreduceRemoteChannelInfo(node, kind,
                                            reinterpret_cast<uintptr_t>(this),
                                            paths);
    }

    XferDes *GPUreduceChannel::create_xfer_des(uintptr_t dma_op,
                                               NodeID launch_node,
                                               XferDesID guid,
                                               const std::vector<XferDesPortInfo>& inputs_info,
                                               const std::vector<XferDesPortInfo>& outputs_info,
                                               int priority,
                                               XferDesRedopInfo redop_info,
                                               const void *fill_data,
                                               size_t fill_size,
                                               size_t fill_total)
    {
      assert(fill_size == 0);
      return new GPUreduceXferDes(dma_op, this, launch_node, guid,
                                  inputs_info, outputs_info,
                                  priority,
                                  redop_info);
    }

    long GPUreduceChannel::submit(Request** requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceRemoteChannelInfo
    //

    GPUreduceRemoteChannelInfo::GPUreduceRemoteChannelInfo(NodeID _owner,
                                                           XferDesKind _kind,
                                                           uintptr_t _remote_ptr,
                                                           const std::vector<Channel::SupportedPath>& _paths)
      : SimpleRemoteChannelInfo(_owner, _kind, _remote_ptr, _paths)
    {}

    RemoteChannel *GPUreduceRemoteChannelInfo::create_remote_channel()
    {
      GPUreduceRemoteChannel *rc = new GPUreduceRemoteChannel(remote_ptr);
      rc->node = owner;
      rc->kind = kind;
      rc->paths.swap(paths);
      return rc;
    }

    // these templates can go here because they're only used by the helper below
    template <typename S>
    bool GPUreduceRemoteChannelInfo::serialize(S& serializer) const
    {
      return ((serializer << owner) &&
              (serializer << kind) &&
              (serializer << remote_ptr) &&
              (serializer << paths));
    }

    template <typename S>
    /*static*/ RemoteChannelInfo *GPUreduceRemoteChannelInfo::deserialize_new(S& deserializer)
    {
      NodeID owner;
      XferDesKind kind;
      uintptr_t remote_ptr;
      std::vector<Channel::SupportedPath> paths;

      if((deserializer >> owner) &&
         (deserializer >> kind) &&
         (deserializer >> remote_ptr) &&
         (deserializer >> paths)) {
        return new GPUreduceRemoteChannelInfo(owner, kind, remote_ptr, paths);
      } else {
        return 0;
      }
    }

    /*static*/ Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                        GPUreduceRemoteChannelInfo> GPUreduceRemoteChannelInfo::serdez_subclass;


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUreduceRemoteChannel
    //

    GPUreduceRemoteChannel::GPUreduceRemoteChannel(uintptr_t _remote_ptr)
      : RemoteChannel(_remote_ptr)
    {}

    uint64_t GPUreduceRemoteChannel::supports_path(ChannelCopyInfo channel_copy_info,
                                                   CustomSerdezID src_serdez_id,
                                                   CustomSerdezID dst_serdez_id,
                                                   ReductionOpID redop_id,
                                                   size_t total_bytes,
                                                   const std::vector<size_t> *src_frags,
                                                   const std::vector<size_t> *dst_frags,
                                                   XferDesKind *kind_ret /*= 0*/,
                                                   unsigned *bw_ret /*= 0*/,
                                                   unsigned *lat_ret /*= 0*/)
    {
      // check first that we have a reduction op (if not, we want the cudamemcpy
      //   path to pick this up instead) and that it has cuda kernels available
      if(!GPUreduceChannel::is_gpu_redop(redop_id))
        return 0;

      // then delegate to the normal supports_path logic
      return Channel::supports_path(channel_copy_info,
                                    src_serdez_id, dst_serdez_id, redop_id,
                                    total_bytes, src_frags, dst_frags,
                                    kind_ret, bw_ret, lat_ret);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUReplHeapListener
    //

    GPUReplHeapListener::GPUReplHeapListener(CudaModule *_module)
      : module(_module)
    {}

    void GPUReplHeapListener::chunk_created(void *base, size_t bytes)
    {
      if(!module->gpus.empty()) {
	log_gpu.info() << "registering replicated heap chunk: base=" << base
                       << " size=" << bytes;

	CUresult ret;
	{
	  AutoGPUContext agc(module->gpus[0]);
	  ret = CUDA_DRIVER_FNPTR(cuMemHostRegister)(base, bytes,
						     CU_MEMHOSTREGISTER_PORTABLE |
						     CU_MEMHOSTREGISTER_DEVICEMAP);
	}
	if(ret != CUDA_SUCCESS) {
	  log_gpu.fatal() << "failed to register replicated heap chunk: base=" << base
			  << " size=" << bytes << " ret=" << ret;
	  abort();
	}
      }
    }

    void GPUReplHeapListener::chunk_destroyed(void *base, size_t bytes)
    {
      if(!module->gpus.empty()) {
	log_gpu.info() << "unregistering replicated heap chunk: base=" << base
                       << " size=" << bytes;

	CUresult ret;
	{
	  AutoGPUContext agc(module->gpus[0]);
	  ret = CUDA_DRIVER_FNPTR(cuMemHostUnregister)(base);
	}
	if(ret != CUDA_SUCCESS)
	  log_gpu.warning() << "failed to unregister replicated heap chunk: base=" << base
			    << " size=" << bytes << " ret=" << ret;
      }
    }


  }; // namespace Cuda

}; // namespace Realm
