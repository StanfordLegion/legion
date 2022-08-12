/* Copyright 2022 Stanford University
 * Copyright 2022 Los Alamos National Laboratory
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

#include "realm/transfer/channel_disk.h"
#include "realm/transfer/lowlevel_dma.h"

#ifdef REALM_ON_WINDOWS
#include <windows.h>
#include <io.h>

static int open(const char *filename, int flags, int mode)
{
  int fd = -1;
  int ret = _sopen_s(&fd, filename, flags, -SH_DENYNO, mode);
  return (ret < 0) ? ret : fd;
}

#define close _close

static int fsync(int fd)
{
  // TODO: is there a way to limit to just the specified file descriptor?
  _flushall();
  return 0;
}
#endif

namespace Realm {

    FileXferDes::FileXferDes(uintptr_t _dma_op, Channel *_channel,
			     NodeID _launch_node, XferDesID _guid,
			     const std::vector<XferDesPortInfo>& inputs_info,
			     const std::vector<XferDesPortInfo>& outputs_info,
			     int _priority)
      : XferDes(_dma_op, _channel, _launch_node, _guid,
		inputs_info, outputs_info,
		_priority, 0, 0)
    {
      RegionInstance inst;
      if((inputs_info.size() == 1) &&
	 (input_ports[0].mem->kind == MemoryImpl::MKIND_FILE)) {
	kind = XFER_FILE_READ;
	inst = inputs_info[0].inst;
	assert(inst.exists());
      } else if((outputs_info.size() == 1) &&
		(output_ports[0].mem->kind == MemoryImpl::MKIND_FILE)) {
	kind = XFER_FILE_WRITE;
	inst = outputs_info[0].inst;
	assert(inst.exists());
      } else {
	assert(0 && "neither source nor dest of FileXferDes is file!?");
      }
	
      RegionInstanceImpl *impl = get_runtime()->get_instance_impl(inst);
      file_info = static_cast<FileMemory::OpenFileInfo *>(impl->metadata.mem_specific);

      const int max_nr = 10; // FIXME
      file_reqs = (FileRequest*) calloc(max_nr, sizeof(DiskRequest));
      for (int i = 0; i < max_nr; i++) {
        file_reqs[i].xd = this;
        available_reqs.push(&file_reqs[i]);
      }
    }
    
    long FileXferDes::get_requests(Request** requests, long nr)
    {
      FileRequest** reqs = (FileRequest**) requests;
      long new_nr = default_get_requests(requests, nr);
      switch (kind) {
        case XFER_FILE_READ:
        {
          for (long i = 0; i < new_nr; i++) {
	    reqs[i]->fd = file_info->fd;
            reqs[i]->file_off = reqs[i]->src_off + file_info->offset;
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
	    reqs[i]->mem_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off,
											reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
          }
          break;
        }
        case XFER_FILE_WRITE:
        {
          for (long i = 0; i < new_nr; i++) {
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
	    reqs[i]->mem_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off,
										       reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
	    reqs[i]->fd = file_info->fd;
            reqs[i]->file_off = reqs[i]->dst_off + file_info->offset;
          }
          break;
        }
        default:
          assert(0);
      }
      return new_nr;
    }

    bool FileXferDes::progress_xd(FileChannel *channel,
				  TimeLimit work_until)
    {
      Request *rq;
      bool did_work = false;
      do {
	long count = get_requests(&rq, 1);
	if(count > 0) {
	  channel->submit(&rq, count);
	  did_work = true;
	} else
	  break;
      } while(!work_until.is_expired());

      return did_work;
    }

    void FileXferDes::notify_request_read_done(Request* req)
    {
      default_notify_request_read_done(req);
    }

    void FileXferDes::notify_request_write_done(Request* req)
    {
      default_notify_request_write_done(req);
    }

    void FileXferDes::flush()
    {
      fsync(file_info->fd);
    }

    DiskXferDes::DiskXferDes(uintptr_t _dma_op, Channel *_channel,
			     NodeID _launch_node, XferDesID _guid,
			     const std::vector<XferDesPortInfo>& inputs_info,
			     const std::vector<XferDesPortInfo>& outputs_info,
			     int _priority)
      : XferDes(_dma_op, _channel, _launch_node, _guid,
		inputs_info, outputs_info,
		_priority, 0, 0)
      , fd(-1) // defer file open
    {
      if((inputs_info.size() >= 1) &&
	 (input_ports[0].mem->kind == MemoryImpl::MKIND_DISK)) {
	kind = XFER_DISK_READ;
	// all input ports should agree on which fd they target
	fd = ((Realm::DiskMemory*)(input_ports[0].mem))->fd;
	for(size_t i = 1; i < input_ports.size(); i++)
	  assert(input_ports[i].mem == input_ports[0].mem);
      } else if((outputs_info.size() >= 1) &&
		(output_ports[0].mem->kind == MemoryImpl::MKIND_DISK)) {
	kind = XFER_DISK_WRITE;
	// all output ports should agree on which fd they target
	fd = ((Realm::DiskMemory*)(output_ports[0].mem))->fd;
	for(size_t i = 1; i < output_ports.size(); i++)
	  assert(output_ports[i].mem == output_ports[0].mem);
      } else {
	assert(0 && "neither source nor dest of DiskXferDes is disk!?");
      }

      const int max_nr = 10; // FIXME
      disk_reqs = (DiskRequest*) calloc(max_nr, sizeof(DiskRequest));
      for (int i = 0; i < max_nr; i++) {
        disk_reqs[i].xd = this;
        disk_reqs[i].fd = fd;
        available_reqs.push(&disk_reqs[i]);
      }
    }
    
    long DiskXferDes::get_requests(Request** requests, long nr)
    {
      DiskRequest** reqs = (DiskRequest**) requests;
      long new_nr = default_get_requests(requests, nr);
      switch (kind) {
        case XFER_DISK_READ:
        {
          for (long i = 0; i < new_nr; i++) {
            reqs[i]->disk_off = /*src_buf.alloc_offset +*/ reqs[i]->src_off;
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
	    reqs[i]->mem_base = output_ports[reqs[i]->dst_port_idx].mem->get_direct_ptr(reqs[i]->dst_off,
											reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
          }
          break;
        }
        case XFER_DISK_WRITE:
        {
          for (long i = 0; i < new_nr; i++) {
            //reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
	    reqs[i]->mem_base = input_ports[reqs[i]->src_port_idx].mem->get_direct_ptr(reqs[i]->src_off,
										       reqs[i]->nbytes);
	    assert(reqs[i]->mem_base != 0);
            reqs[i]->disk_off = /*dst_buf.alloc_offset +*/ reqs[i]->dst_off;
          }
          break;
        }
        default:
          assert(0);
      }
      return new_nr;
    }

    bool DiskXferDes::progress_xd(DiskChannel *channel,
				  TimeLimit work_until)
    {
      Request *rq;
      bool did_work = false;
      do {
	long count = get_requests(&rq, 1);
	if(count > 0) {
	  channel->submit(&rq, count);
	  did_work = true;
	} else
	  break;
      } while(!work_until.is_expired());

      return did_work;
    }

    void DiskXferDes::notify_request_read_done(Request* req)
    {
      default_notify_request_read_done(req);
    }

    void DiskXferDes::notify_request_write_done(Request* req)
    {
      default_notify_request_write_done(req);
    }

    void DiskXferDes::flush()
    {
      fsync(fd);
    }

    FileChannel::FileChannel(BackgroundWorkManager *bgwork)
      : SingleXDQChannel<FileChannel, FileXferDes>(bgwork,
						   XFER_NONE /*FIXME*/,
						   "file channel")
    {
      unsigned bw = 10; // HACK - estimate 10 MB/s
      unsigned latency = 10000; // HACK - estimate 10 us
      unsigned frag_overhead = 10000; // HACK - estimate 10 us

      // all local cpu memories are valid sources/dests
      std::vector<Memory> local_cpu_mems;
      MemcpyChannel::enumerate_local_cpu_memories(local_cpu_mems);

      add_path(Memory::FILE_MEM, false,
               local_cpu_mems,
               bw, latency, frag_overhead, XFER_FILE_READ);

      add_path(local_cpu_mems,
               Memory::FILE_MEM, false,
               bw, latency, frag_overhead, XFER_FILE_WRITE);
    }

    FileChannel::~FileChannel()
    {
    }

    XferDes *FileChannel::create_xfer_des(uintptr_t dma_op,
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
      return new FileXferDes(dma_op, this, launch_node, guid,
			     inputs_info, outputs_info,
			     priority);
    }

    long FileChannel::submit(Request** requests, long nr)
    {
      AsyncFileIOContext* aio_ctx = AsyncFileIOContext::get_singleton();
      for (long i = 0; i < nr; i++) {
        FileRequest* req = (FileRequest*) requests[i];
	// no serdez support
	assert(req->xd->input_ports[req->src_port_idx].serdez_op == 0);
	assert(req->xd->output_ports[req->dst_port_idx].serdez_op == 0);
        switch (req->xd->kind) {
          case XFER_FILE_READ:
            aio_ctx->enqueue_read(req->fd, req->file_off,
                                  req->nbytes, req->mem_base, req);
            break;
          case XFER_FILE_WRITE:
            aio_ctx->enqueue_write(req->fd, req->file_off,
                                   req->nbytes, req->mem_base, req);
            break;
          default:
            assert(0);
        }
      }
      return nr;
    }

    DiskChannel::DiskChannel(BackgroundWorkManager *bgwork)
      : SingleXDQChannel<DiskChannel, DiskXferDes>(bgwork,
						   XFER_NONE /*FIXME*/,
						   "disk channel")
    {
      unsigned bw = 10; // HACK - estimate 10 MB/s
      unsigned latency = 10000; // HACK - estimate 10 us
      unsigned frag_overhead = 10000; // HACK - estimate 10 us

      // all local cpu memories are valid sources/dests
      std::vector<Memory> local_cpu_mems;
      MemcpyChannel::enumerate_local_cpu_memories(local_cpu_mems);

      add_path(Memory::DISK_MEM, false,
               local_cpu_mems,
               bw, latency, frag_overhead, XFER_DISK_READ);

      add_path(local_cpu_mems,
               Memory::DISK_MEM, false,
               bw, latency, frag_overhead, XFER_DISK_WRITE);
    }

    DiskChannel::~DiskChannel()
    {
    }

    XferDes *DiskChannel::create_xfer_des(uintptr_t dma_op,
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
      return new DiskXferDes(dma_op, this, launch_node, guid,
			     inputs_info, outputs_info,
			     priority);
    }

    long DiskChannel::submit(Request** requests, long nr)
    {
      AsyncFileIOContext* aio_ctx = AsyncFileIOContext::get_singleton();
      for (long i = 0; i < nr; i++) {
        DiskRequest* req = (DiskRequest*) requests[i];
	// no serdez support
	assert(req->xd->input_ports[req->src_port_idx].serdez_op == 0);
	assert(req->xd->output_ports[req->dst_port_idx].serdez_op == 0);
        switch (req->xd->kind) {
          case XFER_DISK_READ:
            aio_ctx->enqueue_read(req->fd, req->disk_off,
                                  req->nbytes, req->mem_base, req);
            break;
          case XFER_DISK_WRITE:
            aio_ctx->enqueue_write(req->fd, req->disk_off,
                                   req->nbytes, req->mem_base, req);
            break;
          default:
            assert(0);
        }
      }
      return nr;
    }

}; // namespace Realm
