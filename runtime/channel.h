/* Copyright 2015 Stanford University
 * Copyright 2015 Los Alamos National Laboratory
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
#ifndef LOWLEVEL_CHANNEL
#define LOWLEVEL_CHANNEL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/syscall.h>
#ifdef USE_DISK
#include <linux/aio_abi.h>
#endif  /*USE_DISK*/
#include <map>
#include <vector>
#include <deque>
#include <queue>
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include "lowlevel.h"
#include "lowlevel_dma.h"

#ifdef USE_CUDA
#include "realm/cuda/cuda_module.h"
#endif


namespace LegionRuntime{
  namespace LowLevel{
    class XferDes;
    class Channel;
    
#ifdef USE_CUDA
    typedef Realm::Cuda::GPU GPU;
    typedef Realm::Cuda::GPUFBMemory GPUFBMemory;
#endif

    class Buffer {
    public:
      enum MemoryKind {
        MKIND_CPUMEM,
        MKIND_GPUFB,
        MKIND_DISK
      };

      enum {
        MAX_SERIALIZATION_LEN = 5 * sizeof(int64_t) / sizeof(int) + RegionInstanceImpl::MAX_LINEARIZATION_LEN
      };

      Buffer(void)
            : alloc_offset(0), is_ib(false), block_size(0), elmt_size(0),
              buf_size(0), linearization(), memory(Memory::NO_MEMORY) {}

      Buffer(RegionInstanceImpl::Metadata* metadata, Memory _memory)
            : alloc_offset(metadata->alloc_offset),
              is_ib(false), block_size(metadata->block_size), elmt_size(metadata->elmt_size),
              buf_size(metadata->size), linearization(metadata->linearization),
              memory(_memory){}

      Buffer(off_t _alloc_offset, bool _is_ib,
             int _block_size, int _elmt_size, size_t _buf_size,
             DomainLinearization _linearization, Memory _memory)
            : alloc_offset(_alloc_offset),
              is_ib(_is_ib), block_size(_block_size), elmt_size(_elmt_size),
              buf_size(_buf_size), linearization(_linearization),
              memory(_memory){}

      Buffer& operator=(const Buffer& other)
      {
        alloc_offset = other.alloc_offset;
        is_ib = other.is_ib;
        block_size = other.block_size;
        elmt_size = other.elmt_size;
        buf_size = other.buf_size;
        linearization = other.linearization;
        memory = other.memory;
        return *this;
      }


      ~Buffer() {
      }

      // Note that we don't serialize memory in current implementation
      // User has to manually set memory after deserialize
      void serialize(int* data) const
      {
        int64_t* data64 = (int64_t*) data;
        *data64 = alloc_offset; data64++;
        *data64 = is_ib; data64++;
        *data64 = block_size; data64++;
        *data64 = elmt_size; data64++;
        *data64 = buf_size; data64++;
        linearization.serialize((int*)data64);
      }

      void deserialize(const int* data)
      {
        int64_t* cur = (int64_t*) data;
        alloc_offset = *cur; cur++;
        is_ib = *cur; cur++;
        block_size = *cur; cur++;
        elmt_size = *cur; cur++;
        buf_size = *cur; cur++;
        linearization.deserialize((int*)cur);
      }

      enum DimensionKind {
        DIM_X, // first logical index space dimension
        DIM_Y, // second logical index space dimension
        DIM_Z, // ...
        DIM_F, // field dimension
        INNER_DIM_X, // inner dimension for tiling X
        OUTER_DIM_X, // outer dimension for tiling X
        INNER_DIM_Y, // ...
        OUTER_DIM_Y,
        INNER_DIM_Z,
        OUTER_DIM_Z,
        INNER_DIM_F,
        OUTER_DIM_F,
      };

      // std::vector<size_t> field_ordering;
      // std::vector<size_t> field_sizes;
      // std::vector<DimensionKind> dimension_ordering;
      // std::vector<size_t> dim_size;
      off_t alloc_offset;
      bool is_ib;
      size_t block_size, elmt_size;
      //int inner_stride[3], outer_stride[3], inner_dim_size[3];

      //MemoryKind memory_kind;

      // buffer size of this intermediate buffer.
      // 0 indicates this buffer is large enough to hold
      // entire data set.
      // A number smaller than bytes_total means we need
      // to reuse the buffer.
      size_t buf_size;

      DomainLinearization linearization;

      // The memory instance on which this buffer relies
      Memory memory;
    };

    class Request {
    public:
      // a pointer to the owning xfer descriptor
      // this should set at Request creation
      XferDes* xd;
      // a flag indicating whether this request read has been done
      bool is_read_done;
      // a flag indicating whether this request write has been done
      bool is_write_done;
    };
#ifdef USE_DISK
    class DiskReadRequest : public Request {
    public:
      int fd;
      uint64_t dst_buf;
      int64_t src_offset;
      uint64_t nbytes;
    };

    class DiskWriteRequest : public Request {
    public:
      int fd;
      uint64_t src_buf;
      int64_t dst_offset;
      uint64_t nbytes;
    };
#endif /*USE_DISK*/
    class MemcpyRequest : public Request {
    public:
      char *src_buf, *dst_buf;
      size_t nbytes;
      //std::deque<Copy_1D*> copies_1D;
      //std::deque<Copy_2D*> copies_2D;
      //long num_flying_aios;
    };

    class GASNetReadRequest : public Request {
    public:
      char *dst_buf;
      off_t src_offset;
      size_t nbytes;
    };

    class GASNetWriteRequest : public Request {
    public:
      char *src_buf;
      off_t dst_offset;
      size_t nbytes;
    };

    class RemoteWriteRequest : public Request {
    public:
      gasnet_node_t dst_node;
      char *src_buf, *dst_buf;
      size_t nbytes;
    };

#ifdef USE_CUDA
    class GPUCompletionEvent : public Realm::Cuda::GPUCompletionNotification {
    public:
      GPUCompletionEvent(void) {triggered = false;}
      void request_completed(void) {triggered = true;}
      void reset(void) {triggered = false;}
      bool has_triggered(void) {return triggered;}
    private:
      bool triggered;
    };

    class GPUtoFBRequest : public Request {
    public:
      const char* src;
      off_t dst_offset;
      off_t src_stride, dst_stride;
      off_t src_height, dst_height;
      size_t nbytes_per_line, height, depth;
      GPUCompletionEvent event;
    };

    class GPUfromFBRequest : public Request {
    public:
      off_t src_offset;
      char* dst;
      off_t src_stride, dst_stride;
      off_t src_height, dst_height;
      size_t nbytes_per_line, height, depth;
      GPUCompletionEvent event;
    };

    class GPUinFBRequest : public Request {
    public:
      off_t src_offset, dst_offset;
      off_t src_stride, dst_stride;
      off_t src_height, dst_height;
      size_t nbytes_per_line, height, depth;
      GPUCompletionEvent event;
    };

    class GPUpeerFBRequest : public Request {
    public:
      off_t src_offset, dst_offset;
      off_t src_stride, dst_stride;
      off_t src_height, dst_height;
      size_t nbytes_per_line, height, depth;
      GPU* dst_gpu;
      GPUCompletionEvent event;
    };

#endif

#ifdef USE_HDF
    class HDFReadRequest : public Request {
    public:
      hid_t dataset_id, mem_type_id;
      char* dst;
      hid_t mem_space_id, file_space_id;
      size_t nbytes;
      pthread_rwlock_t *rwlock;
      HDFMemory* hdf_memory;
    };

    class HDFWriteRequest : public Request {
    public:
      hid_t dataset_id, mem_type_id;
      char* src;
      hid_t mem_space_id, file_space_id;
      size_t nbytes;
      pthread_rwlock_t *rwlock;
      HDFMemory* hdf_memory;
    };
#endif

    typedef class Layouts::XferOrder XferOrder;
    class XferDesFence : public Realm::Operation::AsyncWorkItem {
    public:
      XferDesFence(Realm::Operation *op) : Realm::Operation::AsyncWorkItem(op) {}
      virtual void request_cancellation(void) {
    	// ignored for now
      }
      virtual void print(std::ostream& os) const { os << "XferDesFence"; }
    };

    class MaskEnumerator;

    class XferDes {
    public:
      enum XferKind {
        XFER_NONE,
        XFER_DISK_READ,
        XFER_DISK_WRITE,
        XFER_SSD_READ,
        XFER_SSD_WRITE,
        XFER_GPU_TO_FB,
        XFER_GPU_FROM_FB,
        XFER_GPU_IN_FB,
        XFER_GPU_PEER_FB,
        XFER_MEM_CPY,
        XFER_GASNET_READ,
        XFER_GASNET_WRITE,
        XFER_REMOTE_WRITE,
        XFER_HDF_READ,
        XFER_HDF_WRITE,
        XFER_FILE_READ,
        XFER_FILE_WRITE
      };
    public:
      // a pointer to the DmaRequest that contains this XferDes
      DmaRequest* dma_request;
      // a boolean indicating if we have marked started
      bool mark_start;
      // ID of the node that launches this XferDes
      gasnet_node_t launch_node;
      uint64_t /*bytes_submit, */bytes_read, bytes_write, bytes_total;
      uint64_t pre_bytes_write;
      uint64_t next_bytes_read;
      // Domain that is to be copied
      Domain domain;
      // source and destination buffer
      Buffer src_buf, dst_buf;
      // vector that contains the set of fields that needs
      // to be transferred
      // std::vector<size_t> field_set;
      std::vector<OffsetsAndSize> oas_vec;
      // maximum size for a single request
      uint64_t max_req_size;
      // priority of the containing XferDes
      int priority;
      // current, previous and next XferDes in the chain, XFERDES_NO_GUID
      // means this XferDes is the first/last one.
      XferDesID guid, pre_xd_guid, next_xd_guid;
      // XferKind of the Xfer Descriptor
      XferKind kind;
      // XferOrder of the Xfer Descriptor
      XferOrder::Type order;
      // channel this XferDes describes

      // map from unique id to request class, this map only keeps
       std::map<int64_t, uint64_t> segments_read, segments_write;
      // queue that contains all available free requests
      std::queue<Request*> available_reqs;
      enum {
        XFERDES_NO_GUID = 0
      };
      Channel* channel;
      // event is triggered when the XferDes is completed
      XferDesFence* complete_fence;
      // xd_lock is designed to provide thread-safety for
      // SIMULTANEOUS invocation to get_requests,
      // notify_request_read_done, and notify_request_write_done
      pthread_mutex_t xd_lock, update_read_lock, update_write_lock;
    public:
      XferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
              XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
              bool _mark_start, const Buffer& _src_buf, const Buffer& _dst_buf,
              const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
              uint64_t _max_req_size, int _priority,
              XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence)
        : dma_request(_dma_request), mark_start(_mark_start), launch_node(_launch_node),
          bytes_read(0), bytes_write(0), bytes_total(0), pre_bytes_write(0), next_bytes_read(0),
          domain(_domain), src_buf(_src_buf), dst_buf(_dst_buf), oas_vec(_oas_vec),
          max_req_size(_max_req_size), priority(_priority),
          guid(_guid), pre_xd_guid(_pre_xd_guid), next_xd_guid(_next_xd_guid),
          kind (_kind), order(_order), channel(NULL), complete_fence(_complete_fence)
      {
        pthread_mutex_init(&xd_lock, NULL);
        pthread_mutex_init(&update_read_lock, NULL);
        pthread_mutex_init(&update_write_lock, NULL);
      }

      virtual ~XferDes() {
        pthread_mutex_destroy(&xd_lock);
        pthread_mutex_destroy(&update_read_lock);
        pthread_mutex_destroy(&update_write_lock);
      };

      virtual long get_requests(Request** requests, long nr) = 0;

      bool simple_get_mask_request(off_t &src_start, off_t &dst_start, size_t &nbytes,
                                   MaskEnumerator* me,
                                   unsigned &offset_idx, coord_t available_slots);

      template<unsigned DIM>
      bool simple_get_request(off_t &src_start, off_t &dst_start, size_t &nbytes,
                              Layouts::GenericLayoutIterator<DIM>* li,
                              unsigned &offset_idx, coord_t available_slots);

      template<unsigned DIM>
      bool simple_get_request_2d(off_t &src_start, off_t &dst_start,
                                 off_t &src_stride, off_t &dst_stride,
                                 size_t &nbytes_per_line, size_t &nlines,
                                 Layouts::GenericLayoutIterator<DIM>* li,
                                 unsigned &offset_idx, coord_t available_slots);

      template<unsigned DIM>
      bool simple_get_request_3d(off_t &src_start, off_t &dst_start,
                                 off_t &src_stride, off_t &dst_stride,
                                 off_t &src_height, off_t &dst_height,
                                 size_t &nbytes_per_line, size_t &height, size_t &depth,
                                 Layouts::GenericLayoutIterator<DIM>* li,
                                 unsigned &offset_idx, coord_t available_slots);

#ifdef DEADCODE_USE_GENERIC_ITERATOR
      template<unsigned DIM>
      bool simple_get_request(off_t &src_start, off_t &dst_start, size_t &nbytes,
                              Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dsi,
                              Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dso,
                              Rect<1> &irect, Rect<1> &orect,
                              int &done, int &offset_idx, int &block_start, int &total, int available_slots,
                              bool disable_batch = false);
#endif
      void simple_update_bytes_read(int64_t offset, uint64_t size);
      void simple_update_bytes_write(int64_t offset, uint64_t size);

#ifdef USE_XFERDES_ITER
      virtual bool has_next_request() {
        return iterator->has_next();
      }

      // returns the size of this request
      virtual uint64_t gen_next_request(Request* request) {
        uint64_t ret = 0;
        request->copies_2D.clear();
        request->copies_1D.clear();
        int64_t src_offset, dst_offset;
        uint64_t space_left = max_req_size, nbytes;
        size_t field_id;
        std::map<Copy_2D*, size_t> copy_to_field;
        while (iterator->get_next(space_left, src_offset, dst_offset, nbytes, field_id)) {
          // see if we can aggregate this with previous copy structs
          bool aggregated = false;
          for (std::map<Copy_2D*, size_t>::iterator iter = copy_to_field.begin(); iter != copy_to_field.end(); iter++)
            if (iter->second == field_id && iter->first->nbytes == nbytes){
              Copy_2D *copy = iter->first;
              if (copy->nlines == 1) {
                copy->src_stride = src_offset - copy->src_offset;
                copy->dst_stride = dst_offset - copy->dst_offset;
                copy->nlines ++;
                aggregated = true;
                break;
              }
              else if (src_offset == copy->src_stride * copy->nlines + copy->src_offset
                    && dst_offset == copy->dst_stride * copy->nlines + copy->dst_offset) {
                copy->nlines ++;
                aggregated = true;
                break;
              }
            }
          if (!aggregated) {
            Copy_2D *copy = copy_pool_2d.get_one();
            request->copies_2D.push_back(copy);
            copy->src_offset = src_offset;
            copy->dst_offset = dst_offset;
            copy->nlines = 1;
            copy->nbytes = nbytes;
            copy_to_field.insert(std::pair<Copy_2D*, size_t>(copy, field_id));
          }
          ret += nbytes;
        }
        return ret;
      }
#endif

      bool is_completed() {
        return ((bytes_write == bytes_total)&&(next_xd_guid == XFERDES_NO_GUID || next_bytes_read == bytes_total));
      }

      void mark_completed();

      void update_pre_bytes_write(size_t new_val) {
        pthread_mutex_lock(&update_write_lock);
        if (pre_bytes_write < new_val)
          pre_bytes_write = new_val;
        /*uint64_t old_val = pre_bytes_write;
        while (old_val < new_val) {
          pre_bytes_write.compare_exchange_strong(old_val, new_val);
        }*/
        pthread_mutex_unlock(&update_write_lock);
      }

      void update_next_bytes_read(size_t new_val) {
        pthread_mutex_lock(&update_read_lock);
        if (next_bytes_read < new_val)
          next_bytes_read = new_val;
        /*uint64_t old_val = next_bytes_read;
        while (old_val < new_val) {
          next_bytes_read.compare_exchange_strong(old_val, new_val);
        }*/
        pthread_mutex_unlock(&update_read_lock);
      }

      gasnet_node_t find_execution_node() {
        // For now, we think the node that contains the src_buf is the execution node
        return ID(src_buf.memory).memory.owner_node;
      }

      virtual void notify_request_read_done(Request* req) = 0;
      /*{
        // notify previous XferDes that there are more bytes read
        req->is_read_done = true;
        // add read_done segments into segments_read
        for (std::deque<Copy_1D*>::const_iterator it = req->copies_1D.begin(); it != req->copies_1D.end(); it++) {
          segments_read.insert(std::pair<int64_t, uint64_t>((*it)->src_offset, (*it)->nbytes));
        }
        for (std::deque<Copy_2D*>::const_iterator it = req->copies_2D.begin(); it != req->copies_2D.end(); it++) {
          // check if we could reduce 2D -> 1D case
          if ((*it)->src_stride == (*it)->nbytes) {
            segments_read.insert(std::pair<int64_t, uint64_t>((*it)->src_offset, (*it)->nbytes * (*it)->nlines));
          }
          else {
            for (int i = 0; i < (*it)->nlines; i++)
              segments_read.insert(std::pair<int64_t, uint64_t>((*it)->src_offset + (*it)->src_stride * i, (*it)->nbytes));
          }
        }
        // clear countinous finished segments and updates bytes
        std::map<int64_t, uint64_t>::iterator it;
        bool updated = false;
        while (true) {
          it = segments_read.find(bytes_read);
          if (it == segments_read.end())
            break;
          updated = true;
          bytes_read += it->second;
          segments_read.erase(it);
        }
        if (pre_XferDes != NULL && updated) {
          pre_XferDes->update_next_bytes_read(bytes_read);
        }
      }*/

      virtual void notify_request_write_done(Request* req) = 0;
      /*{
        req->is_write_done = true;
        // add write_done segments into segments_written
        for (std::deque<Copy_1D*>::const_iterator it = req->copies_1D.begin(); it != req->copies_1D.end(); it++) {
          segments_written.insert(std::pair<int64_t, uint64_t>((*it)->dst_offset, (*it)->nbytes));
        }
        for (std::deque<Copy_2D*>::const_iterator it = req->copies_2D.begin(); it != req->copies_2D.end(); it++) {
          if ((*it)->dst_stride == (*it)->nbytes) {
            segments_written.insert(std::pair<int64_t, uint64_t>((*it)->dst_offset, (*it)->nbytes * (*it)->nlines));
          }
          else {
            for (int i = 0; i < (*it)->nlines; i++)
              segments_written.insert(std::pair<int64_t, uint64_t>((*it)->dst_offset + (*it)->dst_stride * i, (*it)->nbytes));
          }
        }

        // add this request back to available_reqs queue
        copy_pool_1d.free_multiple(req->copies_1D);
        copy_pool_2d.free_multiple(req->copies_2D);
        req->copies_1D.clear();
        req->copies_2D.clear();
        available_reqs.push(req);
        // clear countinous finished segments and updates bytes
        std::map<int64_t, uint64_t>::iterator it;
        bool updated = false;
        while (true) {
          it = segments_written.find(bytes_write);
          if (it == segments_written.end())
            break;
          updated = true;
          bytes_write += it-> second;
          segments_written.erase(it);
        }
        // notify next XferDes that there are more bytes written
        if (next_XferDes != NULL && updated) {
          next_XferDes->update_pre_bytes_write(bytes_write);
        }
      }*/

      virtual void flush() = 0;
    };

    /*
    class GenericBlockIterator {
    public:
      GenericBlockIterator(int in_lo, int in_hi, int in_block_size, int in_offset_size)
      : lo(in_lo), hi(in_hi), block_size(in_block_size), offset_size(in_offset_size) {
        done = 0;
        offset_idx = 0;
        any_left = true;
        block_start = lo;
      }

      bool step(int n) {
        if (done + n == hi - lo + 1) {
          offset_idx ++;
          if (offset_idx >= offset_size)
            any_left = false;
          done = block_start - lo;
        }
        else if ((done + n) % block_size == 0) {
          offset_idx =
        }
        else {
          done += n;
        }
        return any_left;
      }

      operator bool(void) const {return any_left;}
    public:
      int done, offset_idx, lo, hi, offset_size, block_size, block_start;
      bool any_left;
    };
    */

    template<unsigned DIM>
    class MemcpyXferDes : public XferDes {
    public:
      MemcpyXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                    XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                    bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
                    const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                    uint64_t max_req_size, long max_nr, int _priority,
                    XferOrder::Type _order, XferDesFence* _complete_fence);

      ~MemcpyXferDes()
      {
        // clear available_reqs
        while (!available_reqs.empty()) {
          available_reqs.pop();
        }
        if (DIM == 0) {
          delete me;
        } else {
          delete li;
        }
        free(requests);
        // trigger complete event
        //if (complete_event.exists()) {
          //get_runtime()->get_genevent_impl(complete_event)->trigger(complete_event.gen, gasnet_mynode());
        //}
        // If src_buf is intermediate buffer,
        // we need to free the buffer
        if (src_buf.is_ib) {
          get_runtime()->get_memory_impl(src_buf.memory)->free_bytes(src_buf.alloc_offset, src_buf.buf_size);
        }
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      MemcpyRequest* requests;
      //std::map<int64_t, uint64_t> segments_read, segments_write;
      Layouts::GenericLayoutIterator<DIM>* li;
      MaskEnumerator* me;
      unsigned offset_idx;
      const char *src_buf_base, *dst_buf_base;
    };

    template<unsigned DIM>
    class GASNetXferDes : public XferDes {
    public:
      GASNetXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                    XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                    bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
                    const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                    uint64_t _max_req_size, long max_nr, int _priority,
                    XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);

      ~GASNetXferDes()
      {
        // clear available_reqs
        while (!available_reqs.empty()) {
          available_reqs.pop();
        }
        if (DIM == 0) {
          delete me;
        } else {
          delete li;
        }
        free(requests);
        // trigger completion event
        //if (complete_event.exists()) {
          //get_runtime()->get_genevent_impl(complete_event)->trigger(complete_event.gen, gasnet_mynode());
        //}
        // If src_buf is intermediate buffer,
        // we need to free the buffer
        if (src_buf.is_ib) {
          get_runtime()->get_memory_impl(src_buf.memory)->free_bytes(src_buf.alloc_offset, src_buf.buf_size);
        }
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      Request* requests;
      //std::map<int64_t, uint64_t> segments_read, segments_write;
      Layouts::GenericLayoutIterator<DIM>* li;
      MaskEnumerator* me;
      unsigned offset_idx;
      const char *buf_base;
    };

    template<unsigned DIM>
    class RemoteWriteXferDes : public XferDes {
    public:
      RemoteWriteXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                         XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                         bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
                         const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                         uint64_t max_req_size, long max_nr, int _priority,
                         XferOrder::Type _order, XferDesFence* _complete_fence);

      ~RemoteWriteXferDes()
      {
        // clear available_reqs
        while (!available_reqs.empty()) {
          available_reqs.pop();
        }
        if (DIM == 0) {
          delete me;
        } else {
          delete li;
        }
        free(requests);
        //if (complete_event.exists()) {
          //get_runtime()->get_genevent_impl(complete_event)->trigger(complete_event.gen, gasnet_mynode());
        //}
        // If src_buf is intermediate buffer,
        // we need to free the buffer
        if (src_buf.is_ib) {
          get_runtime()->get_memory_impl(src_buf.memory)->free_bytes(src_buf.alloc_offset, src_buf.buf_size);
        }
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      RemoteWriteRequest* requests;
      //std::map<int64_t, uint64_t> segments_read, segments_write;
      Layouts::GenericLayoutIterator<DIM>* li;
      MaskEnumerator* me;
      unsigned offset_idx;
      const char *src_buf_base, *dst_buf_base;
      MemoryImpl *dst_mem_impl;
    };

#ifdef USE_DISK
    template<unsigned DIM>
    class DiskXferDes : public XferDes {
    public:
      DiskXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                  XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                  bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
                  const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                  uint64_t _max_req_size, long max_nr, int _priority,
                  XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);

      ~DiskXferDes() {
        // clear available_reqs
        while (!available_reqs.empty()) {
          available_reqs.pop();
        }
        if (DIM == 0) {
          delete me;
        } else {
          delete li;
        }
        free(requests);
        // trigger complete event
        //if (complete_event.exists()) {
          //get_runtime()->get_genevent_impl(complete_event)->trigger(complete_event.gen, gasnet_mynode());
        //}
        // If src_buf is intermediate buffer,
        // we need to free the buffer
        if (src_buf.is_ib) {
          get_runtime()->get_memory_impl(src_buf.memory)->free_bytes(src_buf.alloc_offset, src_buf.buf_size);
        }
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      int fd;
      Request* requests;
      //std::map<int64_t, uint64_t> segments_read, segments_write;
      Layouts::GenericLayoutIterator<DIM>* li;
      MaskEnumerator* me;
      unsigned offset_idx;
      const char *buf_base;
    };
#endif /*USE_DISK*/
#ifdef USE_CUDA
    template<unsigned DIM>
    class GPUXferDes : public XferDes {
    public:
      GPUXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                 XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                 bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
                 const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                 uint64_t _max_req_size, long max_nr, int _priority,
                 XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);
      ~GPUXferDes()
      {
        // clear available_reqs
        while (!available_reqs.empty()) {
          Request* req = available_reqs.front();
          delete req;
          available_reqs.pop();
        }
        if (DIM == 0) {
          delete me;
        } else {
          delete li;
        }
        //free(requests);
        // trigger complete event
        //if (complete_event.exists()) {
          //get_runtime()->get_genevent_impl(complete_event)->trigger(complete_event.gen, gasnet_mynode());
        //}
        // If src_buf is intermediate buffer,
        // we need to free the buffer
        if (src_buf.is_ib) {
          get_runtime()->get_memory_impl(src_buf.memory)->free_bytes(src_buf.alloc_offset, src_buf.buf_size);
        }
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      //Request* requests;
      //std::map<int64_t, uint64_t> segments_read, segments_write;
      Layouts::GenericLayoutIterator<DIM>* li;
      MaskEnumerator* me;
      unsigned offset_idx;
      char *src_buf_base;
      char *dst_buf_base;
      GPU *dst_gpu, *src_gpu;
    };
#endif

#ifdef USE_HDF
    template<unsigned DIM>
    class HDFXferDes : public XferDes {
    public:
      HDFXferDes(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                 XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                 bool mark_started,
                 RegionInstance inst, const Buffer& _src_buf, const Buffer& _dst_buf,
                 const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                 uint64_t _max_req_size, long max_nr, int _priority,
                 XferOrder::Type _order, XferKind _kind, XferDesFence* _complete_fence);
      ~HDFXferDes()
      {
        // clear available_reqs
        while (!available_reqs.empty()) {
          available_reqs.pop();
        }
        free(requests);
        delete hli;
        // trigger complete event
        //if (complete_event.exists()) {
          //get_runtime()->get_genevent_impl(complete_event)->trigger(complete_event.gen, gasnet_mynode());
        //}
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
      void flush();

    private:
      Request* requests;
      char *buf_base;
      HDFMemory::HDFMetadata *hdf_metadata;
      std::vector<OffsetsAndSize>::iterator fit;
      //GenericPointInRectIterator<DIM>* pir;
      //GenericLinearSubrectIterator<Mapping<DIM, 1> >* lsi;
      Layouts::HDFLayoutIterator<DIM>* hli;
    };
#endif

    class Channel {
    public:
      virtual ~Channel() {};
    public:
      // the kind of XferDes this channel can accept
      XferDes::XferKind kind;
      /*
       * Submit nr asynchronous requests into the channel instance.
       * This is supposed to be a non-blocking function call, and
       * should immediately return the number of requests that are
       * successfully submitted.
       */
      virtual long submit(Request** requests, long nr) = 0;

      /*
       *
       */
      virtual void pull() = 0;

      /*
       * Return the number of slots that are available for
       * submitting requests
       */
      virtual long available() = 0;
    protected:
      // std::deque<Copy_1D> copies_1D;
      // std::deque<Copy_2D> copies_2D;
    };

    class MemcpyChannel;

    class MemcpyThread {
    public:
      MemcpyThread(MemcpyChannel* _channel) : channel(_channel) {}
      void thread_loop();
      static void* start(void* arg);
      void stop();
    private:
      MemcpyChannel* channel;
      std::deque<MemcpyRequest*> thread_queue;
    };

    class MemcpyChannel : public Channel {
    public:
      MemcpyChannel(long max_nr);
      ~MemcpyChannel();
      void stop();
      void get_request(std::deque<MemcpyRequest*>& thread_queue);
      void return_request(std::deque<MemcpyRequest*>& thread_queue);
      long submit(Request** requests, long nr);
      void pull();
      long available();
      bool is_stopped;
    private:
      std::deque<MemcpyRequest*> pending_queue, finished_queue;
      pthread_mutex_t pending_lock, finished_lock;
      pthread_cond_t pending_cond;
      long capacity;
      bool sleep_threads;
      //std::vector<MemcpyRequest*> available_cb;
      //MemcpyRequest** cbs;
    };

    class GASNetChannel : public Channel {
    public:
      GASNetChannel(long max_nr, XferDes::XferKind _kind);
      ~GASNetChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      long capacity;
    };

    class RemoteWriteChannel : public Channel {
    public:
      RemoteWriteChannel(long max_nr);
      ~RemoteWriteChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
      void notify_completion() {capacity ++;}
    private:
      long capacity;
    };

#ifdef USE_DISK
    class DiskChannel : public Channel {
    public:
      DiskChannel(long max_nr, XferDes::XferKind _kind);
      ~DiskChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      aio_context_t ctx;
      long capacity;
      std::vector<struct iocb*> available_cb;
      struct iocb* cb;
      struct iocb** cbs;
      struct io_event* events;
      //std::deque<Copy_1D*>::iterator iter_1d;
      //std::deque<Copy_2D*>::iterator iter_2d;
      //uint64_t cur_line;
    };
#endif /*USE_DISK*/
    
#ifdef USE_CUDA
    class GPUChannel : public Channel {
    public:
      GPUChannel(GPU* _src_gpu, long max_nr, XferDes::XferKind _kind);
      ~GPUChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      GPU* src_gpu;
      long capacity;
      std::deque<Request*> pending_copies;
    };
#endif

#ifdef USE_HDF
    class HDFChannel : public Channel {
    public:
      HDFChannel(long max_nr, XferDes::XferKind _kind);
      ~HDFChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      long capacity;
    };
#endif

    class ChannelManager {
    public:
      ChannelManager(void) {
        memcpy_channel = NULL;
        gasnet_read_channel = gasnet_write_channel = NULL;
        remote_write_channel = NULL;
#ifdef USE_DISK
        disk_read_channel = NULL;
        disk_write_channel = NULL;
#endif /*USE_DISK*/
#ifdef USE_HDF
        hdf_read_channel = NULL;
        hdf_write_channel = NULL;
#endif
      }
      ~ChannelManager(void) {
        if (memcpy_channel)
          delete memcpy_channel;
        if (gasnet_read_channel)
          delete gasnet_read_channel;
        if (gasnet_write_channel)
          delete gasnet_write_channel;
        if (remote_write_channel)
          delete remote_write_channel;
#ifdef USE_DISK
        if (disk_read_channel)
          delete disk_read_channel;
        if (disk_write_channel)
          delete disk_write_channel;
#endif /*USE_DISK*/
#ifdef USE_CUDA
        std::map<GPU*, GPUChannel*>::iterator it;
        for (it = gpu_to_fb_channels.begin(); it != gpu_to_fb_channels.end(); it++) {
          delete it->second;
        }
        for (it = gpu_from_fb_channels.begin(); it != gpu_from_fb_channels.end(); it++) {
          delete it->second;
        }
        for (it = gpu_in_fb_channels.begin(); it != gpu_in_fb_channels.end(); it++) {
          delete it->second;
        }
        for (it = gpu_peer_fb_channels.begin(); it != gpu_peer_fb_channels.end(); it++) {
          delete it->second;
        }
#endif
      }
      MemcpyChannel* create_memcpy_channel(long max_nr) {
        assert(memcpy_channel == NULL);
        memcpy_channel = new MemcpyChannel(max_nr);
        return memcpy_channel;
      }
      GASNetChannel* create_gasnet_read_channel(long max_nr) {
        assert(gasnet_read_channel == NULL);
        gasnet_read_channel = new GASNetChannel(max_nr, XferDes::XFER_GASNET_READ);
        return gasnet_read_channel;
      }
      GASNetChannel* create_gasnet_write_channel(long max_nr) {
        assert(gasnet_write_channel == NULL);
        gasnet_write_channel = new GASNetChannel(max_nr, XferDes::XFER_GASNET_WRITE);
        return gasnet_write_channel;
      }
      RemoteWriteChannel* create_remote_write_channel(long max_nr) {
        assert(remote_write_channel == NULL);
        remote_write_channel = new RemoteWriteChannel(max_nr);
        return remote_write_channel;
      }
#ifdef USE_DISK
      DiskChannel* create_disk_read_channel(long max_nr) {
        assert(disk_read_channel == NULL);
        disk_read_channel = new DiskChannel(max_nr, XferDes::XFER_DISK_READ);
        return disk_read_channel;
      }
      DiskChannel* create_disk_write_channel(long max_nr) {
        assert(disk_write_channel == NULL);
        disk_write_channel = new DiskChannel(max_nr, XferDes::XFER_DISK_WRITE);
        return disk_write_channel;
      }
#endif /*USE_DISK*/
#ifdef USE_CUDA
      GPUChannel* create_gpu_to_fb_channel(long max_nr, GPU* src_gpu) {
        gpu_to_fb_channels[src_gpu] = new GPUChannel(src_gpu, max_nr, XferDes::XFER_GPU_TO_FB);
        return gpu_to_fb_channels[src_gpu];
      }
      GPUChannel* create_gpu_from_fb_channel(long max_nr, GPU* src_gpu) {
        gpu_from_fb_channels[src_gpu] = new GPUChannel(src_gpu, max_nr, XferDes::XFER_GPU_FROM_FB);
        return gpu_from_fb_channels[src_gpu];
      }
      GPUChannel* create_gpu_in_fb_channel(long max_nr, GPU* src_gpu) {
        gpu_in_fb_channels[src_gpu] = new GPUChannel(src_gpu, max_nr, XferDes::XFER_GPU_IN_FB);
        return gpu_in_fb_channels[src_gpu];
      }
      GPUChannel* create_gpu_peer_fb_channel(long max_nr, GPU* src_gpu) {
        gpu_peer_fb_channels[src_gpu] = new GPUChannel(src_gpu, max_nr, XferDes::XFER_GPU_PEER_FB);
        return gpu_peer_fb_channels[src_gpu];
      }
#endif
#ifdef USE_HDF
      HDFChannel* create_hdf_read_channel(long max_nr) {
        assert(hdf_read_channel == NULL);
        hdf_read_channel = new HDFChannel(max_nr, XferDes::XFER_HDF_READ);
        return hdf_read_channel;
      }
      HDFChannel* create_hdf_write_channel(long max_nr) {
        assert(hdf_write_channel == NULL);
        hdf_write_channel = new HDFChannel(max_nr, XferDes::XFER_HDF_WRITE);
        return hdf_write_channel;
      }
#endif
      MemcpyChannel* get_memcpy_channel() {
        return memcpy_channel;
      }
      GASNetChannel* get_gasnet_read_channel() {
        return gasnet_read_channel;
      }
      GASNetChannel* get_gasnet_write_channel() {
        return gasnet_write_channel;
      }
      RemoteWriteChannel* get_remote_write_channel() {
        return remote_write_channel;
      }
#ifdef USE_DISK
      DiskChannel* get_disk_read_channel() {
        return disk_read_channel;
      }
      DiskChannel* get_disk_write_channel() {
        return disk_write_channel;
      }
#endif /*USE_DISK*/
#ifdef USE_CUDA
      GPUChannel* get_gpu_to_fb_channel(GPU* gpu) {
        std::map<GPU*, GPUChannel*>::iterator it;
        it = gpu_to_fb_channels.find(gpu);
        assert(it != gpu_to_fb_channels.end());
        return (it->second);
      }
      GPUChannel* get_gpu_from_fb_channel(GPU* gpu) {
        std::map<GPU*, GPUChannel*>::iterator it;
        it = gpu_from_fb_channels.find(gpu);
        assert(it != gpu_from_fb_channels.end());
        return (it->second);
      }
      GPUChannel* get_gpu_in_fb_channel(GPU* gpu) {
        std::map<GPU*, GPUChannel*>::iterator it;
        it = gpu_in_fb_channels.find(gpu);
        assert(it != gpu_in_fb_channels.end());
        return (it->second);
      }
      GPUChannel* get_gpu_peer_fb_channel(GPU* gpu) {
        std::map<GPU*, GPUChannel*>::iterator it;
        it = gpu_peer_fb_channels.find(gpu);
        assert(it != gpu_peer_fb_channels.end());
        return (it->second);
      }
#endif
#ifdef USE_HDF
      HDFChannel* get_hdf_read_channel() {
        return hdf_read_channel;
      }
      HDFChannel* get_hdf_write_channel() {
        return hdf_write_channel;
      }
#endif
    public:
      MemcpyChannel* memcpy_channel;
      GASNetChannel *gasnet_read_channel, *gasnet_write_channel;
      RemoteWriteChannel* remote_write_channel;
#ifdef USE_DISK
      DiskChannel *disk_read_channel, *disk_write_channel;
#endif /*USE_DISK*/
#ifdef USE_CUDA
      std::map<GPU*, GPUChannel*> gpu_to_fb_channels, gpu_in_fb_channels, gpu_from_fb_channels, gpu_peer_fb_channels;
#endif
#ifdef USE_HDF
      HDFChannel *hdf_read_channel, *hdf_write_channel;
#endif
    };

    class CompareXferDes {
    public:
      bool operator() (XferDes* a, XferDes* b) {
        if(a->priority == b->priority)
          return (a < b);
        else 
          return (a->priority < b->priority);
      }
    };
    //typedef std::priority_queue<XferDes*, std::vector<XferDes*>, CompareXferDes> PriorityXferDesQueue;
    typedef std::set<XferDes*, CompareXferDes> PriorityXferDesQueue;

    class XferDesQueue;
    class DMAThread {
    public:
      DMAThread(long _max_nr, XferDesQueue* _xd_queue, std::vector<Channel*>& _channels) {
        for (std::vector<Channel*>::iterator it = _channels.begin(); it != _channels.end(); it ++) {
          channel_to_xd_pool[*it] = new PriorityXferDesQueue;
        }
        xd_queue = _xd_queue;
        max_nr = _max_nr;
        is_stopped = false;
        requests = (Request**) calloc(max_nr, sizeof(Request*));
        sleep = false;
        pthread_mutex_init(&enqueue_lock, NULL);
        pthread_cond_init(&enqueue_cond, NULL);
      }
      DMAThread(long _max_nr, XferDesQueue* _xd_queue, Channel* _channel) {
        channel_to_xd_pool[_channel] = new PriorityXferDesQueue;
        xd_queue = _xd_queue;
        max_nr = _max_nr;
        is_stopped = false;
        requests = (Request**) calloc(max_nr, sizeof(Request*));
        sleep = false;
        pthread_mutex_init(&enqueue_lock, NULL);
        pthread_cond_init(&enqueue_cond, NULL);
      }
      ~DMAThread() {
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        for (it = channel_to_xd_pool.begin(); it != channel_to_xd_pool.end(); it++) {
          delete it->second;
        }
        free(requests);
        pthread_mutex_destroy(&enqueue_lock);
        pthread_cond_destroy(&enqueue_cond);
      }
      void dma_thread_loop();
      // Thread start function that takes an input of DMAThread
      // instance, and start to execute the requests from XferDes
      // by using its channels.
      static void* start(void* arg) {
        DMAThread* dma_thread = (DMAThread*) arg;
        dma_thread->dma_thread_loop();
        return NULL;
      }

      void stop() {
        pthread_mutex_lock(&enqueue_lock);
        is_stopped = true;
        pthread_cond_signal(&enqueue_cond);
        pthread_mutex_unlock(&enqueue_lock);
      }
    public:
      pthread_mutex_t enqueue_lock;
      pthread_cond_t enqueue_cond;
      std::map<Channel*, PriorityXferDesQueue*> channel_to_xd_pool;
      bool sleep;
      bool is_stopped;
    private:
      // maximum allowed num of requests for a single
      long max_nr;
      Request** requests;
      XferDesQueue* xd_queue;
    };

    struct NotifyXferDesCompleteMessage {
      struct RequestArgs {
        XferDesFence* fence;
      };

      static void handle_request(RequestArgs args)
      {
        args.fence->mark_finished(true/*successful*/);
      }

      typedef ActiveMessageShortNoReply<XFERDES_NOTIFY_COMPLETION_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(gasnet_node_t target, XferDesFence* fence)
      {
        RequestArgs args;
        args.fence = fence;
        Message::request(target, args);
      }
    };

    struct XferDesRemoteWriteMessage {
      struct RequestArgs : public BaseMedium {
        char* dst_buf;
        RemoteWriteRequest* req;
        gasnet_node_t sender;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<XFERDES_REMOTEWRITE_MSGID,
                                         RequestArgs,
                                         handle_request> Message;

      static void send_request(gasnet_node_t target, char* dst_buf,
                               char* src_buf, size_t nbytes, RemoteWriteRequest* req)
      {
        RequestArgs args;
        args.dst_buf = dst_buf;
        args.req = req;
        args.sender = gasnet_mynode();
        //TODO: need to ask Sean what payload mode we should use
        Message::request(target, args, src_buf, nbytes, PAYLOAD_KEEP, dst_buf);
      }
    };

    struct XferDesRemoteWriteAckMessage {
      struct RequestArgs {
        RemoteWriteRequest* req;
      };

      static void handle_request(RequestArgs args);
      typedef ActiveMessageShortNoReply<XFERDES_REMOTEWRITE_ACK_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(gasnet_node_t target, RemoteWriteRequest* req)
      {
        RequestArgs args;
        args.req = req;
        Message::request(target, args);
      }
    };

    struct XferDesCreateMessage {
      struct RequestArgs : public BaseMedium {
        RegionInstance inst;
        Memory src_mem, dst_mem;
        XferDesFence* fence;
      };

      // TODO: replace with new serialization stuff
      struct Payload {
        DmaRequest* dma_request;
        gasnet_node_t launch_node;
        XferDesID guid, pre_xd_guid, next_xd_guid;
        uint64_t max_req_size;
        long max_nr;
        int priority;
        XferOrder::Type order;
        XferDes::XferKind kind;
        Domain domain;
        int src_buf_bits[Buffer::MAX_SERIALIZATION_LEN], dst_buf_bits[Buffer::MAX_SERIALIZATION_LEN];
        size_t oas_vec_size; // as long as it needs to be
        OffsetsAndSize oas_vec_start;
        const OffsetsAndSize &oas_vec(int idx) const { return *((&oas_vec_start)+idx); }
        OffsetsAndSize &oas_vec(int idx) { return *((&oas_vec_start)+idx); }
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<XFERDES_CREATE_MSGID,
                                         RequestArgs,
                                         handle_request> Message;

      static void send_request(gasnet_node_t target, DmaRequest* dma_request, gasnet_node_t launch_node,
                               XferDesID guid, XferDesID pre_xd_guid, XferDesID next_xd_guid,
                               const Buffer& src_buf, const Buffer& dst_buf,
                               const Domain& domain, const std::vector<OffsetsAndSize>& oas_vec,
                               uint64_t max_req_size, long max_nr, int priority,
                               XferOrder::Type order, XferDes::XferKind kind,
                               XferDesFence* fence, RegionInstance inst = RegionInstance::NO_INST)
      {
        size_t payload_size = sizeof(Payload) + sizeof(OffsetsAndSize) * oas_vec.size();
        Payload *payload = (Payload*) malloc(payload_size);
        payload->dma_request = dma_request;
        payload->launch_node = launch_node;
        payload->guid = guid;
        payload->pre_xd_guid = pre_xd_guid;
        payload->next_xd_guid = next_xd_guid;
        payload->max_req_size = max_req_size;
        payload->max_nr = max_nr;
        payload->priority = priority;
        payload->order = order;
        payload->kind = kind;
        payload->domain = domain;
        src_buf.serialize(payload->src_buf_bits);
        dst_buf.serialize(payload->dst_buf_bits);
        payload->oas_vec_size = oas_vec.size();
        for (unsigned i = 0; i < oas_vec.size(); i++)
          payload->oas_vec(i) = oas_vec[i];
        RequestArgs args;
        args.inst = inst;
        args.src_mem = src_buf.memory;
        args.dst_mem = dst_buf.memory;
        args.fence = fence;
        Message::request(target, args, payload, payload_size, PAYLOAD_FREE);
      }
    };

    struct XferDesDestroyMessage {
      struct RequestArgs {
        XferDesID guid;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<XFERDES_DESTROY_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(gasnet_node_t target, XferDesID guid)
      {
        RequestArgs args;
        args.guid = guid;
        Message::request(target, args);
      }
    };

    struct UpdateBytesWriteMessage {
      struct RequestArgs {
        XferDesID guid;
        uint64_t bytes_write;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<XFERDES_UPDATE_BYTES_WRITE_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(gasnet_node_t target, XferDesID guid, uint64_t bytes_write)
      {
        RequestArgs args;
        args.guid = guid;
        args.bytes_write = bytes_write;
        Message::request(target, args);
      }
    };

    struct UpdateBytesReadMessage {
      struct RequestArgs {
        XferDesID guid;
        uint64_t bytes_read;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<XFERDES_UPDATE_BYTES_READ_MSGID,
                                        RequestArgs,
                                        handle_request> Message;

      static void send_request(gasnet_node_t target, XferDesID guid, uint64_t bytes_read)
      {
        RequestArgs args;
        args.guid = guid;
        args.bytes_read = bytes_read;
        Message::request(target, args);
      }
    };

    class XferDesQueue {
    public:
      struct XferDesWithUpdates{
        XferDesWithUpdates(void): xd(NULL), pre_bytes_write(0) {}
        XferDes* xd;
        uint64_t pre_bytes_write;
      };
      enum {
        NODE_BITS = 16,
        INDEX_BITS = 32
      };
      XferDesQueue(Realm::CoreReservationSet& crs)
      : core_rsrv("DMA request queue", crs, Realm::CoreReservationParameters()) {
        pthread_mutex_init(&queues_lock, NULL);
        pthread_rwlock_init(&guid_lock, NULL);
        // reserve the first several guid
        next_to_assign_idx = 10;
        num_threads = 0;
        num_memcpy_threads = 0;
        dma_threads = NULL;
      }

      ~XferDesQueue() {
        // clean up the priority queues
        pthread_mutex_lock(&queues_lock);
        std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
        for (it2 = queues.begin(); it2 != queues.end(); it2++) {
          delete it2->second;
        }
        pthread_mutex_unlock(&queues_lock);
        pthread_mutex_destroy(&queues_lock);
        pthread_rwlock_destroy(&guid_lock);
      }

      XferDesID get_guid(gasnet_node_t execution_node)
      {
        // GUID rules:
        // First NODE_BITS indicates which node will execute this xd
        // Next NODE_BITS indicates on which node this xd is generated
        // Last INDEX_BITS means a unique idx, which is used to resolve conflicts
        XferDesID idx = __sync_fetch_and_add(&next_to_assign_idx, 1);
        return (((XferDesID)execution_node << (NODE_BITS + INDEX_BITS)) | ((XferDesID)gasnet_mynode() << INDEX_BITS) | idx);
      }

      void update_pre_bytes_write(XferDesID xd_guid, uint64_t bytes_write)
      {
        gasnet_node_t execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == gasnet_mynode()) {
          pthread_rwlock_wrlock(&guid_lock);
          std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(xd_guid);
          if (it != guid_to_xd.end()) {
            if (it->second.xd != NULL) {
              it->second.xd->update_pre_bytes_write(bytes_write);
            } else {
              if (bytes_write > it->second.pre_bytes_write)
                it->second.pre_bytes_write = bytes_write;
            }
          } else {
            XferDesWithUpdates xd_struct;
            xd_struct.pre_bytes_write = bytes_write;
            guid_to_xd[xd_guid] = xd_struct;
          }
          pthread_rwlock_unlock(&guid_lock);
        }
        else {
          // send a active message to remote node
          UpdateBytesWriteMessage::send_request(execution_node, xd_guid, bytes_write);
        }
      }

      void update_next_bytes_read(XferDesID xd_guid, uint64_t bytes_read)
      {
        gasnet_node_t execution_node = xd_guid >> (NODE_BITS + INDEX_BITS);
        if (execution_node == gasnet_mynode()) {
          pthread_rwlock_rdlock(&guid_lock);
          std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(xd_guid);
          if (it == guid_to_xd.end()) {
            // This means this update goes slower than future updates, which marks
            // completion of xfer des (ID = xd_guid). In this case, it is safe to return
            pthread_rwlock_unlock(&guid_lock);
            return;
          }
          assert(it != guid_to_xd.end());
          assert(it->second.xd != NULL);
          it->second.xd->update_next_bytes_read(bytes_read);
          pthread_rwlock_unlock(&guid_lock);
        }
        else {
          // send a active message to remote node
          UpdateBytesReadMessage::send_request(execution_node, xd_guid, bytes_read);
        }
      }

      void register_dma_thread(DMAThread* dma_thread)
      {
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        pthread_mutex_lock(&queues_lock);
        for(it = dma_thread->channel_to_xd_pool.begin(); it != dma_thread->channel_to_xd_pool.end(); it++) {
          channel_to_dma_thread[it->first] = dma_thread;
          queues[it->first] = new PriorityXferDesQueue;
        }
        pthread_mutex_unlock(&queues_lock);
      }

      void destroy_xferDes(XferDesID guid) {
        pthread_rwlock_wrlock(&guid_lock);
        std::map<XferDesID, XferDesWithUpdates>::iterator it = guid_to_xd.find(guid);
        assert(it != guid_to_xd.end());
        assert(it->second.xd != NULL);
        XferDes* xd = it->second.xd;
        guid_to_xd.erase(it);
        delete xd;
        pthread_rwlock_unlock(&guid_lock);
      }

      void enqueue_xferDes_local(XferDes* xd) {
        pthread_rwlock_wrlock(&guid_lock);
        std::map<XferDesID, XferDesWithUpdates>::iterator git = guid_to_xd.find(xd->guid);
        if (git != guid_to_xd.end()) {
          // xerDes_queue has received updates of this xferdes
          // need to integrate these updates into xferdes
          assert(git->second.xd == NULL);
          xd->update_pre_bytes_write(git->second.pre_bytes_write);
          git->second.xd = xd;
          git->second.pre_bytes_write = 0;
        } else {
          XferDesWithUpdates xd_struct;
          xd_struct.xd = xd;
          guid_to_xd[xd->guid] = xd_struct;
        }
        pthread_rwlock_unlock(&guid_lock);
        std::map<Channel*, DMAThread*>::iterator it;
        it = channel_to_dma_thread.find(xd->channel);
        assert(it != channel_to_dma_thread.end());
        DMAThread* dma_thread = it->second;
        pthread_mutex_lock(&dma_thread->enqueue_lock);
        pthread_mutex_lock(&queues_lock);
        std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
        it2 = queues.find(xd->channel);
        assert(it2 != queues.end());
        // push ourself into the priority queue
        it2->second->insert(xd);
        pthread_mutex_unlock(&queues_lock);
        if (dma_thread->sleep) {
          dma_thread->sleep = false;
          pthread_cond_signal(&dma_thread->enqueue_cond);
        }
        pthread_mutex_unlock(&dma_thread->enqueue_lock);
      }

      bool dequeue_xferDes(DMAThread* dma_thread, bool wait_on_empty) {
        pthread_mutex_lock(&dma_thread->enqueue_lock);
        std::map<Channel*, PriorityXferDesQueue*>::iterator it;
        if (wait_on_empty) {
          bool empty = true;
          for(it = dma_thread->channel_to_xd_pool.begin(); it != dma_thread->channel_to_xd_pool.end(); it++) {
            pthread_mutex_lock(&queues_lock);
            std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
            it2 = queues.find(it->first);
            assert(it2 != queues.end());
            if (it2->second->size() > 0)
              empty = false;
            pthread_mutex_unlock(&queues_lock);
          }

          if (empty && !dma_thread->is_stopped) {
            dma_thread->sleep = true;
            pthread_cond_wait(&dma_thread->enqueue_cond, &dma_thread->enqueue_lock);
          }
        }

        for(it = dma_thread->channel_to_xd_pool.begin(); it != dma_thread->channel_to_xd_pool.end(); it++) {
          pthread_mutex_lock(&queues_lock);
          std::map<Channel*, PriorityXferDesQueue*>::iterator it2;
          it2 = queues.find(it->first);
          assert(it2 != queues.end());
          it->second->insert(it2->second->begin(), it2->second->end());
          it2->second->clear();
          pthread_mutex_unlock(&queues_lock);
        }
        pthread_mutex_unlock(&dma_thread->enqueue_lock);
        return true;
      }

      void start_worker(int count, int max_nr, ChannelManager* channel_manager);

      void stop_worker();

    protected:
      std::map<Channel*, DMAThread*> channel_to_dma_thread;
      std::map<Channel*, PriorityXferDesQueue*> queues;
      std::map<XferDesID, XferDesWithUpdates> guid_to_xd;
      pthread_mutex_t queues_lock;
      pthread_rwlock_t guid_lock;
      XferDesID next_to_assign_idx;
      Realm::CoreReservation core_rsrv;
      int num_threads, num_memcpy_threads;
      DMAThread** dma_threads;
      MemcpyThread** memcpy_threads;
      std::vector<Realm::Thread*> worker_threads;
    };

    XferDesQueue* get_xdq_singleton();
#ifdef USE_CUDA
    void register_gpu_in_dma_systems(GPU* gpu);
#endif
    void start_channel_manager(int count, int max_nr, Realm::CoreReservationSet& crs);
    void stop_channel_manager();
    template<unsigned DIM>
    void create_xfer_des(DmaRequest* _dma_request, gasnet_node_t _launch_node,
                         XferDesID _guid, XferDesID _pre_xd_guid, XferDesID _next_xd_guid,
                         bool mark_started, const Buffer& _src_buf, const Buffer& _dst_buf,
                         const Domain& _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                         uint64_t _max_req_size, long max_nr, int _priority,
                         XferOrder::Type _order, XferDes::XferKind _kind,
                         XferDesFence* _complete_fence, RegionInstance inst = RegionInstance::NO_INST);

    void destroy_xfer_des(XferDesID _guid);
  }  // namespace LowLevel
} // namespace LegionRuntime
#endif

#ifdef COPY_POOL
struct Copy_1D {
   int64_t src_offset, dst_offset;
   uint64_t nbytes;
 };

 struct Copy_2D {
   int64_t src_offset, dst_offset, src_stride, dst_stride;
   uint64_t nbytes, nlines;
 };

 template<class T>
 class CopyPool {
 public:
   CopyPool(long in_num) {
     num = in_num;
     arr = (T*) calloc(num, sizeof(T));
     for (long i = 0; i < num; i++)
       queue.push(&arr[i]);
   }

   T* get_one() {
     assert(!queue.empty());
     T* ret = queue.front();
     queue.pop();
     return ret;
   }

   void free_one(T* elmnt) {
     queue.push(elmnt);
   }

   void free_multiple(std::deque<T*> list) {
     typename std::deque<T*>::iterator it;
     for (it = list.begin(); it != list.end(); it++) {
       free_one(*it);
     }
   }

   ~CopyPool() {
     free(arr);
   }
 private:
   long num;
   T* arr;
   std::queue<T*> queue;
 };

 static CopyPool<Copy_1D> copy_pool_1d(1000);
 static CopyPool<Copy_2D> copy_pool_2d(1000);

#endif

/*
    class MemcpyThread {
    public:
      MemcpyThread() {
        num_pending_reqs = 0;
        pthread_mutex_init(&submit_lock, NULL);
        pthread_mutex_init(&pull_lock, NULL);
        pthread_cond_init(&condvar, NULL);
      }
      ~MemcpyThread() {
        pthread_mutex_destroy(&submit_lock);
        pthread_mutex_destroy(&pull_lock);
        pthread_cond_destroy(&condvar);
      }
      long submit(MemcpyRequest** requests, long nr) {
        pthread_mutex_lock(&submit_lock);
        for (int i = 0; i < nr; i++) {
          pending_queue.push(requests[i]);
        }
        if (num_pending_reqs == 0 && nr > 0) {
          pthread_cond_signal(&condvar);
        }
        num_pending_reqs += nr;
        pthread_mutex_unlock(&submit_lock);
        return nr;
      }

      long pull(MemcpyRequest** requests, long nr) {
        pthread_mutex_lock(&pull_lock);
        long np = 0;
        while (np < nr && !finished_queue.empty()) {
          requests[np] = finished_queue.front();
          finished_queue.pop();
          np++;
        }
        pthread_mutex_unlock(&pull_lock);
        return np;
      }
      void work();
      static void* start(void* arg);
    private:
      std::queue<MemcpyRequest*> pending_queue;
      std::queue<MemcpyRequest*> finished_queue;
      long num_pending_reqs;
	  pthread_mutex_t submit_lock, pull_lock;
      pthread_cond_t condvar;
    };

      // TODO: for now, iterator always follows FIFO order on destination side,
      // should also implement FIFO order on source side
      class XferDesIterator {
        class XferPoint {
        public:
          DomainPoint point;
          size_t f;

          void translate_from_vector_position(Buffer* buffer, std::vector<size_t> pos) {

            DomainPoint base_point;
            for (int i = 0; i < MAX_DIM; i++) {
              base_point.x[i] = 1;
              pos[i] = 0;
            }
            for (int i = 0; i < buffer->dimension_ordering.size(); i++) {
              switch (Buffer::to_full_dimension(buffer->dimension_ordering[i])) {
                case Buffer::DIM_X:
                  x[0] += base_point.x[0] * pos[i];
                  base_point.x[0] *= buffer->block_size[i];
                  break;
                case Buffer::DIM_Y:
                  x[1] += base_point.x[1] * pos[i];
                  base_point.x[1] *= buffer->block_size[i];
                  break;
                case Buffer::DIM_Z:
                  x[2] += base_point.x[2] * pos[i];
                  base_point.x[2] *= buffer->block_size[i];
                  break;
                case Buffer::DIM_F:
                  f = pos[i];
                  break;
                default:
                  assert(0);
              }
            }
          }

          void translate_to_vector_position(Buffer* buffer, std::vector<size_t> pos) {
            pos.clear();
            Point base_point; base_point.x = base_point.y = base_point.z = base_point.f = 1;
            for (int i = 0; i < buffer->dimension_ordering.size(); i++) {
              switch (Buffer::to_full_dimension(buffer->dimension_ordering[i])) {
                case Buffer::DIM_X:
                  pos.push_back((point.x / base_point.x) % buffer->block_size[i]);
                  base_point.x *= buffer->block_size[i];
                  break;
                case Buffer::DIM_Y:
                  pos.push_back((point.y / base_point.y) % buffer->block_size[i]);
                  base_point.y *= buffer->block_size[i];
                  break;
                case Buffer::DIM_Z:
                  pos.push_back((point.z / base_point.z) % buffer->block_size[i]);
                  base_point.z *= buffer->block_size[i];
                  break;
                case Buffer::DIM_F:
                  assert(base_point.f == 1);
                  pos.push_back(point.f);
                  base_point.f *= buffer->block_size[i];
                  break;
                default:
                  assert(0);
              }
            }
          }

        };

        //calc the location of current position in  buffer
        static off_t calc_mem_loc(XferPoint pos, Buffer* buffer) {
          off_t offset = buffer->alloc_offset;
          int index = 0, field_start = 0;
          for (int i = 0; i < pos.f; i++)
            field_start += buffer->field_sizes[i];
          for (int i = 0; i < pos.point.dim; i++) {
            index += (pos.point.point_data[i] / buffer->inner_dim_size[i]) * buffer->outer_stride[i]
                      + (pos.point.point_data[i] % buffer->inner_dim_size[i]) * buffer->inner_stride[i];
          }

          offset += (index / buffer->block_size) * buffer->block_size * buffer->elmt_size +
                    field_start * buffer->block_size +
                    (index % buffer->block_size) * buffer->field_sizes[pos.f];
          return offset;
        }

        int get_rect_size(Buffer::DimensionKind kind) {
          switch (Buffer::to_full_dimension(kind)) {
            case Buffer::DIM_X:
              return rect_data[3] - rect_data[0];
            case Buffer::DIM_Y:
              return rect_data[4] - rect_data[1];
            case Buffer::DIM_Z:
              return rect_data[5] - rect_data[2];
            case Buffer::DIM_F:
              assert(0);
              return 0;
            default:
              assert(0);
          }
          return 0;
        }

        XferDesIterator(Buffer *_src_buffer, Buffer *_dst_buffer, std::vector<size_t> _field_set, int* _rect_data, uint64_t _min_req_size, uint64_t _max_req_size) {
          src_buffer = _src_buffer;
          dst_buffer = _dst_buffer;
          field_set = _field_set;
          for (int i = 0; i < 6; i++) rect_data[i] = _rect_data[i];
          min_req_size = _min_req_size;
          max_req_size = _max_req_size;
          max_field_size = 0;
          base_req_size = 1;
          for (int i = 0; i < field_set.size(); i++)
            if (field_set[i] > max_field_size)
              max_field_size = field_set[i];
          start_idx = 0;
          while (start_idx < src_buffer->dimension_ordering.size()
                 && start_idx < dst_buffer->dimension_ordering.size()) {
            Buffer::DimensionKind kind = src_buffer->dimension_ordering[start_idx];
            if (kind != dst_buffer->dimension_ordering[start_idx])
              break;
            if (src_buffer->dim_size[start_idx] != dst_buffer->dim_size[start_idx])
              break;
            if (kind == Buffer::DIM_F && !Buffer::same_field_ordering(*src_buffer, *dst_buffer))
              break;
            // TODO: the following check may need changes
            if (kind != Buffer::DIM_F && get_rect_size(kind) != src_buffer->block_size[start_idx])
              break;
            if (kind == Buffer::DIM_F) {
              int sum_field_size = 0;
              for (int i = 0; i < field_set.size(); i++)
                sum_field_size += field_set[i];
              if (sum_field_size * base_req_size > max_req_size)
                break;
              else {
                max_field_size = sum_field_size;
                all_field_size = max_field_size;
                aggregate_all_fields = true;
              }
            }
            else {
              int new_base_req_size = base_req_size * src_buffer->block_size[start_idx];
              if (max_field_size * new_base_req_size > max_req_size)
                break;
              else
                base_req_size = new_base_req_size;
            }
            start_idx ++;
          }

          // computing the beginning position
          // TODO: compute pos
        }
        // pointers to src and dst  buffers
        Buffer *src_buffer, *dst_buffer;
        const size_t ALL_FIELDS = 66666;
        XferPoint cur_pos;
        std::vector<size_t> field_set;
        size_t field_idx;
        Domain domain;
        bool aggregate_all_fields = false;
        uint64_t min_req_size, max_req_size;
        int start_idx, base_req_size, max_field_size, all_field_size;
      public:
        // returns whether current pos is available or not
        // moves pos to the next available position in buffer under the restriction rect_data and field_set
        // if current position is available, then do nothing
        bool move_to_next_available_pos(Buffer *buffer) {
          // TODO: this should be replaced by rect.contains()
          if (domain.contains(cur_pos.point) && cur_pos.f < field_set.size())
            return true;

        }

        bool has_next() {

        }

        uint64_t min(uint64_t a, uint64_t b, uint64_t c) {
          uint64_t ret = a;
          if (ret > b) ret = b;
          if (ret > c) ret = c;
          return ret;
        }

        bool get_next(uint64_t max_size, int64_t &src_offset, int64_t &dst_offset, uint64_t &nbytes, size_t &field_id) {
          assert(move_to_next_available_pos(dst_buffer));
          Buffer::DimensionKind src_kind = src_buffer->dimension_ordering[start_idx];
          Buffer::DimensionKind dst_kind = dst_buffer->dimension_ordering[start_idx];
          uint64_t field_size;
          if (dst_kind == Buffer::DIM_F) {
            // TODO: consider merging different fields
            assert(!aggregate_all_fields);
            nbytes = dst_buffer->metadata.field_sizes[cur_pos.f] * base_req_size;
            field_id = cur_pos.f;
            if (nbytes > max_size)
              return false;
            src_offset = calc_mem_loc(cur_pos, src_buffer);
            dst_offset = calc_mem_loc(cur_pos, dst_buffer);
            cur_pos.f
            move_to_next_available_pos(dst_buffer);
          }
          // This is the case src and dst has the same inner dimension, we get a chance to batch multiple elements
          else if (Buffer::to_full_dimension(src_kind) == Buffer::to_full_dimension(dst_kind)) {
            if (aggregate_all_fields) {
              field_size = all_field_size;
              field_id = ALL_FIELDS;
            }
            else {
              field_size = dst_buffer->metadata.field_sizes[point.f];
              field_id = point.f;
            }
            uint64_t num_batch_elemnts = min(max_size/field_size, dst_buffer->block_size[start_idx] - pos[start_idx], src_buffer->block_size[start_idx] - pos[start_idx]);
            if (num_batch_elemnts == 0)
              return false;
            point.translate_to_vector_position(src_buffer, src_pos);
            nbytes = field_size * base_req_size * num_batch_elemnts;
            src_offset = calc_mem_loc(src_pos, src_buffer);
            dst_offset = calc_mem_loc(pos, dst_buffer);
            pos[start_idx] += num_batch_elemnts;
            move_to_next_available_pos(pos, dst_buffer);
          }
          // This is the case we can only do a single element
          else {
            point.translate_from_vector_position(dst_buffer, pos);
            if (aggregate_all_fields) {
              field_size = all_field_size;
              field_id = ALL_FIELDS;
            }
            else {
              field_size = dst_buffer->metadata.field_sizes[point.f];
              field_id = point.f;
            }
            nbytes = field_size * base_req_size;
            if (nbytes > max_size)
              return false;
            point.translate_to_vector_position(src_buffer, src_pos);
            src_offset = calc_mem_loc(src_pos, src_buffer);
            dst_offset = calc_mem_loc(pos, dst_buffer);
            pos[start_idx] ++;
            move_to_next_available_pos(pos, dst_buffer);
          }
          return true;
        }
      };
 */
