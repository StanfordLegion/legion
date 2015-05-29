#ifndef LOWLEVEL_CHANNEL
#define LOWLEVEL_CHANNEL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
#include <map>
#include <vector>
#include <deque>
#include <queue>
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include "lowlevel.h"

namespace LegionRuntime{
  namespace LowLevel{
    class XferDes;
    class Channel;

    struct OffsetsAndSize {
      off_t src_offset, dst_offset;
      int size;
    };

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

    class Buffer {
    public:
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

      enum MemoryKind {
        MKIND_CPUMEM,
        MKIND_GPUFB,
        MKIND_DISK
      };
      // std::vector<size_t> field_ordering;
      std::vector<size_t> field_sizes;
      //std::vector<DimensionKind> dimension_ordering;
      //std::vector<size_t> dim_size;
      off_t alloc_offset;
      bool is_ib;
      int block_size, elmt_size;
      //int inner_stride[3], outer_stride[3], inner_dim_size[3];

      MemoryKind memory_kind;

      DomainLinearization linearization;

      // buffer size of this intermediate buffer.
      // 0 indicates this buffer is large enough to hold
      // entire data set.
      // A number smaller than bytes_total means we need
      // to reuse the buffer.
      uint64_t buf_size;
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

    class MemcpyRequest : public Request {
    public:
      char *src_buf, *dst_buf;
      size_t nbytes;
      //std::deque<Copy_1D*> copies_1D;
      //std::deque<Copy_2D*> copies_2D;
      //long num_flying_aios;
    };

    class XferDes {
    public:
      enum XferKind {
        XFER_DISK_READ,
        XFER_DISK_WRITE,
        XFER_SSD_READ,
        XFER_SSD_WRITE,
        XFER_GPU_READ,
        XFER_GPU_WRITE,
        XFER_MEM_CPY
      };
      enum XferOrder {
        SRC_FIFO,
        DST_FIFO,
        ANY_ORDER
      };
    protected:
      uint64_t /*bytes_submit, */bytes_read, bytes_write, bytes_total;
      uint64_t pre_bytes_write;
      uint64_t next_bytes_read;
      // Domain that is to be copied
      Domain domain;
      // source and destination buffer
      Buffer *src_buf, *dst_buf;
      // map from unique id to request class, this map only keeps
      // std::map<int64_t, uint64_t> segments_read, segments_written;
      // queue that contains all available free requests
      std::queue<Request*> available_reqs;
      // vector that contains the set of fields that needs
      // to be transferred
      // std::vector<size_t> field_set;
      std::vector<OffsetsAndSize> oas_vec;
      // minimum and maximum size for a single request
      uint64_t min_req_size, max_req_size;
      // previous and next XferDes in the chain, NULL is this XferDes is
      // the first one.
      XferDes *pre_XferDes, *next_XferDes;
    public:
      // XferKind of the Xfer Descriptor
      XferKind kind;
      // XferOrder of the Xfer Descriptor
      XferOrder order;
      // channel this XferDes describes
      Channel* channel;
    public:
      virtual ~XferDes() {};

      virtual long get_requests(Request** requests, long nr) = 0;

      template<unsigned DIM>
      bool simple_get_request(off_t &src_start, off_t &dst_start, size_t &nbytes,
                              Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dsi,
                              Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dso,
                              Rect<1> &irect, Rect<1> &orect,
                              int &done, int &offset_idx, int &block_start, int &total, int available_slots);

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

      bool is_done() {
        //TODO: replaced by other method
        return bytes_write == bytes_total;
      }

      void update_pre_XferDes(XferDes* _pre_XferDes) {
        pre_XferDes = _pre_XferDes;
      }

      void update_next_XferDes(XferDes* _next_XferDes) {
        next_XferDes = _next_XferDes;
      }

      void update_pre_bytes_write(size_t new_val) {
        pre_bytes_write = new_val;
      }

      void update_next_bytes_read(size_t new_val) {
        next_bytes_read = new_val;
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
      MemcpyXferDes(Channel* _channel,
                    bool has_pre_XferDes, bool has_next_XferDes,
                    Buffer* _src_bufsize, Buffer* _dst_bufsize,
                    char *_src_mem_base, char *_dst_mem_base,
                    Domain _domain, const std::vector<OffsetsAndSize>& _oas_vec,
                    uint64_t _bytes_total, uint64_t max_req_size, long max_nr,
                    XferOrder _order);

      ~MemcpyXferDes()
      {
        free(requests);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);

    private:
      MemcpyRequest* requests;
      std::map<int64_t, uint64_t> segments_read, segments_write;
      Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > *dso, *dsi;
      int done, offset_idx, block_start, total;
      Rect<1> orect, irect;
      char *src_mem_base, *dst_mem_base;
    };

    template<unsigned DIM>
    class DiskWriteXferDes : public XferDes {
    public:
      DiskWriteXferDes(Channel* _channel, int _fd,
                       bool has_pre_XferDes, bool has_next_XferDes,
                       Buffer* _src_buf, Buffer* _dst_buf,
                       uint64_t _bytes_total, long max_nr);

      ~DiskWriteXferDes()
      {
        free(requests);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);

    private:
      int fd;
      DiskWriteRequest* requests;
      std::map<int64_t, uint64_t> segments_read, segments_write;
      Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > dso, dsi;
      int done, offset_idx, block_start;
      Rect<1> orect, irect;
    };

    template<unsigned DIM>
    class DiskReadXferDes : public XferDes {
    public:
        DiskReadXferDes(Channel* _channel, int _fd,
                        bool has_pre_XferDes, bool has_next_XferDes,
                        Buffer* _src_buf, Buffer* _dst_buf,
                        uint64_t _bytes_total, long max_nr);

      ~DiskReadXferDes()
      {
        free(requests);
      }

      long get_requests(Request** requests, long nr);
      void notify_request_read_done(Request* req);
      void notify_request_write_done(Request* req);
    private:
      int fd;
      DiskReadRequest* requests;
      std::map<int64_t, uint64_t> segments_read, segments_write;
      Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> > dso, dsi;
      int done, offset_idx, block_start;
      Rect<1> orect, irect;
    #ifdef DEADCODE
    private:
      long min(long a, long b)
      {
        return (a<b) ? a : b;
      }

      long num_available_reqs()
      {
        assert(bytes_submit <= pre_bytes_write);
        long ret = min((pre_bytes_write - bytes_submit) / req_nbytes, (dst_bufsize + next_bytes_read - bytes_submit) / req_nbytes);
        //ret = min(ret, (bytes_total - bytes_submit) / req_nbytes);
        ret = min(ret, available_reqs.size());
        return ret;
      }
    #endif
    };

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

    class MemcpyThread;

    class MemcpyChannel : public Channel {
    public:
      MemcpyChannel(long max_nr, MemcpyThread* _worker);
      ~MemcpyChannel();
      long submit(Request** requests, long nr);
      void pull();
      long available();
    private:
      long capacity;
      MemcpyThread* worker;
      std::vector<MemcpyRequest*> available_cb;
      MemcpyRequest** cbs;
    };

    class DiskReadChannel : public Channel {
    public:
      DiskReadChannel(long max_nr);
      ~DiskReadChannel();
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

    class DiskWriteChannel : public Channel {
    public:
      DiskWriteChannel(long max_nr);
      ~DiskWriteChannel();
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

    class DMAThread {
    public:
      DMAThread(long _max_nr) {
        max_nr = _max_nr;
        is_stopped = false;
        requests = (Request**) calloc(max_nr, sizeof(Request*));
        pthread_mutex_init(&channel_lock, NULL);
        pthread_mutex_init(&xferDes_lock, NULL);
        channel_queue.clear();
        xferDes_queue.clear();
      }
      ~DMAThread() {
        free(requests);
        pthread_mutex_destroy(&channel_lock);
        pthread_mutex_destroy(&xferDes_lock);
      }
      // Add a channel into the DMAThread instance.
      // Invocation of this function indicates the DMAThread
      // instance keeps charge of this channel
      void add_channel(Channel* channel) {
        pthread_mutex_lock(&channel_lock);
        channel_queue.push_back(channel);
        pthread_mutex_unlock(&channel_lock);
      }
      // Add a XferDes into the DMAThread instance.
      // The DMAThread will start to polling requests from
      // this XferDes and perform IO
      void add_xferDes(XferDes* xferDes) {
        pthread_mutex_lock(&xferDes_lock);
        xferDes_queue.push_back(xferDes);
        pthread_mutex_unlock(&xferDes_lock);
      }
      // Thread start function that takes an input of DMAThread
      // instance, and start to execute the requests from XferDes
      // by using its channels.
      static void* start(void* arg);

      void stop() {
        is_stopped = true;
      }
    private:
      // maximum allowed num of requests for a single
      long max_nr;
      bool is_stopped;
      Request** requests;
      pthread_mutex_t channel_lock, xferDes_lock;
      std::vector<Channel*> channel_queue;
      std::vector<XferDes*> xferDes_queue;
    };

  }  // namespace LowLevel
} // namespace LegionRuntime
#endif


/*
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
