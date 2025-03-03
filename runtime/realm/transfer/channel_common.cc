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

#include "realm/transfer/channel_common.h"
#include "realm/mem_impl.h"
#include "realm/transfer/ib_memory.h"
#include "realm/runtime_impl.h"
#include <algorithm>

namespace Realm {

  // fast memcpy stuff - uses std::copy instead of memcpy to communicate
  //  alignment guarantees to the compiler
  template <typename T>
  static void memcpy_1d_typed(uintptr_t dst_base, uintptr_t src_base, size_t bytes)
  {
    std::copy(reinterpret_cast<const T *>(src_base),
              reinterpret_cast<const T *>(src_base + bytes),
              reinterpret_cast<T *>(dst_base));
  }

  template <typename T>
  static void memcpy_2d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
                              uintptr_t src_base, uintptr_t src_lstride, size_t bytes,
                              size_t lines)
  {
    for(size_t i = 0; i < lines; i++) {
      std::copy(reinterpret_cast<const T *>(src_base),
                reinterpret_cast<const T *>(src_base + bytes),
                reinterpret_cast<T *>(dst_base));
      // manual strength reduction
      src_base += src_lstride;
      dst_base += dst_lstride;
    }
  }

  template <typename T>
  static void memcpy_3d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
                              uintptr_t dst_pstride, uintptr_t src_base,
                              uintptr_t src_lstride, uintptr_t src_pstride, size_t bytes,
                              size_t lines, size_t planes)
  {
    // adjust plane stride amounts to account for line strides (so we don't have
    //  to subtract the line strides back out in the loop)
    uintptr_t dst_pstride_adj = dst_pstride - (lines * dst_lstride);
    uintptr_t src_pstride_adj = src_pstride - (lines * src_lstride);

    for(size_t j = 0; j < planes; j++) {
      for(size_t i = 0; i < lines; i++) {
        std::copy(reinterpret_cast<const T *>(src_base),
                  reinterpret_cast<const T *>(src_base + bytes),
                  reinterpret_cast<T *>(dst_base));
        // manual strength reduction
        src_base += src_lstride;
        dst_base += dst_lstride;
      }
      src_base += src_pstride_adj;
      dst_base += dst_pstride_adj;
    }
  }

  template <typename T>
  static void memset_1d_typed(uintptr_t dst_base, size_t bytes, T val)
  {
    std::fill(reinterpret_cast<T *>(dst_base), reinterpret_cast<T *>(dst_base + bytes),
              val);
  }

  template <typename T>
  static void memset_2d_typed(uintptr_t dst_base, uintptr_t dst_lstride, size_t bytes,
                              size_t lines, T val)
  {
    for(size_t i = 0; i < lines; i++) {
      std::fill(reinterpret_cast<T *>(dst_base), reinterpret_cast<T *>(dst_base + bytes),
                val);
      // manual strength reduction
      dst_base += dst_lstride;
    }
  }

  template <typename T>
  static void memset_3d_typed(uintptr_t dst_base, uintptr_t dst_lstride,
                              uintptr_t dst_pstride, size_t bytes, size_t lines,
                              size_t planes, T val)
  {
    // adjust plane stride amounts to account for line strides (so we don't have
    //  to subtract the line strides back out in the loop)
    uintptr_t dst_pstride_adj = dst_pstride - (lines * dst_lstride);

    for(size_t j = 0; j < planes; j++) {
      for(size_t i = 0; i < lines; i++) {
        std::fill(reinterpret_cast<T *>(dst_base),
                  reinterpret_cast<T *>(dst_base + bytes), val);
        // manual strength reduction
        dst_base += dst_lstride;
      }
      dst_base += dst_pstride_adj;
    }
  }

  // need types with various powers-of-2 size/alignment - we have up to
  //  uint64_t as builtins, but we need trivially-copyable 16B and 32B things
  struct dummy_16b_t {
    uint64_t a, b;
  };
  struct dummy_32b_t {
    uint64_t a, b, c, d;
  };
  REALM_ALIGNED_TYPE_CONST(aligned_16b_t, dummy_16b_t, 16);
  REALM_ALIGNED_TYPE_CONST(aligned_32b_t, dummy_32b_t, 32);

  void memcpy_1d(uintptr_t dst_base, uintptr_t src_base, size_t bytes)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (src_base - 1) & (bytes - 1));
// define DEBUG_MEMCPYS
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memcpy_1d: dst=" << dst_base << " src=" << src_base
                   << std::dec << " bytes=" << bytes << " align=" << (alignment & 31);
#endif
    // TODO: consider jump table approach?
    if((alignment & 31) == 31)
      memcpy_1d_typed<aligned_32b_t>(dst_base, src_base, bytes);
    else if((alignment & 15) == 15)
      memcpy_1d_typed<aligned_16b_t>(dst_base, src_base, bytes);
    else if((alignment & 7) == 7)
      memcpy_1d_typed<uint64_t>(dst_base, src_base, bytes);
    else if((alignment & 3) == 3)
      memcpy_1d_typed<uint32_t>(dst_base, src_base, bytes);
    else if((alignment & 1) == 1)
      memcpy_1d_typed<uint16_t>(dst_base, src_base, bytes);
    else
      memcpy_1d_typed<uint8_t>(dst_base, src_base, bytes);
  }

  void memcpy_2d(uintptr_t dst_base, uintptr_t dst_lstride, uintptr_t src_base,
                 uintptr_t src_lstride, size_t bytes, size_t lines)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (dst_lstride - 1) & (src_base - 1) &
                          (src_lstride - 1) & (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memcpy_2d: dst=" << dst_base << "+" << dst_lstride
                   << " src=" << src_base << "+" << src_lstride << std::dec
                   << " bytes=" << bytes << " lines=" << lines
                   << " align=" << (alignment & 31);
#endif
    // TODO: consider jump table approach?
    if((alignment & 31) == 31)
      memcpy_2d_typed<aligned_32b_t>(dst_base, dst_lstride, src_base, src_lstride, bytes,
                                     lines);
    else if((alignment & 15) == 15)
      memcpy_2d_typed<aligned_16b_t>(dst_base, dst_lstride, src_base, src_lstride, bytes,
                                     lines);
    else if((alignment & 7) == 7)
      memcpy_2d_typed<uint64_t>(dst_base, dst_lstride, src_base, src_lstride, bytes,
                                lines);
    else if((alignment & 3) == 3)
      memcpy_2d_typed<uint32_t>(dst_base, dst_lstride, src_base, src_lstride, bytes,
                                lines);
    else if((alignment & 1) == 1)
      memcpy_2d_typed<uint16_t>(dst_base, dst_lstride, src_base, src_lstride, bytes,
                                lines);
    else
      memcpy_2d_typed<uint8_t>(dst_base, dst_lstride, src_base, src_lstride, bytes,
                               lines);
  }

  void memcpy_3d(uintptr_t dst_base, uintptr_t dst_lstride, uintptr_t dst_pstride,
                 uintptr_t src_base, uintptr_t src_lstride, uintptr_t src_pstride,
                 size_t bytes, size_t lines, size_t planes)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment =
        ((dst_base - 1) & (dst_lstride - 1) & (dst_pstride - 1) & (src_base - 1) &
         (src_lstride - 1) & (src_pstride - 1) & (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memcpy_3d: dst=" << dst_base << "+" << dst_lstride
                   << "+" << dst_pstride << " src=" << src_base << "+" << src_lstride
                   << "+" << src_pstride << std::dec << " bytes=" << bytes
                   << " lines=" << lines << " planes=" << planes
                   << " align=" << (alignment & 31);
#endif
    // performance optimization for intel (and probably other) cpus: walk
    //  destination addresses as linearly as possible, even if that messes up
    //  the source address pattern (probably because writebacks are more
    //  expensive than cache fills?)
    if(dst_pstride < dst_lstride) {
      // switch lines and planes
      std::swap(dst_pstride, dst_lstride);
      std::swap(src_pstride, src_lstride);
      std::swap(planes, lines);
    }
    // TODO: consider jump table approach?
    if((alignment & 31) == 31)
      memcpy_3d_typed<aligned_32b_t>(dst_base, dst_lstride, dst_pstride, src_base,
                                     src_lstride, src_pstride, bytes, lines, planes);
    else if((alignment & 15) == 15)
      memcpy_3d_typed<aligned_16b_t>(dst_base, dst_lstride, dst_pstride, src_base,
                                     src_lstride, src_pstride, bytes, lines, planes);
    else if((alignment & 7) == 7)
      memcpy_3d_typed<uint64_t>(dst_base, dst_lstride, dst_pstride, src_base, src_lstride,
                                src_pstride, bytes, lines, planes);
    else if((alignment & 3) == 3)
      memcpy_3d_typed<uint32_t>(dst_base, dst_lstride, dst_pstride, src_base, src_lstride,
                                src_pstride, bytes, lines, planes);
    else if((alignment & 1) == 1)
      memcpy_3d_typed<uint16_t>(dst_base, dst_lstride, dst_pstride, src_base, src_lstride,
                                src_pstride, bytes, lines, planes);
    else
      memcpy_3d_typed<uint8_t>(dst_base, dst_lstride, dst_pstride, src_base, src_lstride,
                               src_pstride, bytes, lines, planes);
  }

  void memset_1d(uintptr_t dst_base, size_t bytes, const void *fill_data,
                 size_t fill_size)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memset_1d: dst=" << dst_base << std::dec
                   << " bytes=" << bytes << " align=" << (alignment & 31);
#endif
    // alignment must be at least as good as fill size to use memset
    // TODO: consider jump table approach?
    if((fill_size == 32) && ((alignment & 31) == 31))
      memset_1d_typed<aligned_32b_t>(dst_base, bytes,
                                     *reinterpret_cast<const aligned_32b_t *>(fill_data));
    else if((fill_size == 16) && ((alignment & 15) == 15))
      memset_1d_typed<aligned_16b_t>(dst_base, bytes,
                                     *reinterpret_cast<const aligned_16b_t *>(fill_data));
    else if((fill_size == 8) && ((alignment & 7) == 7))
      memset_1d_typed<uint64_t>(dst_base, bytes,
                                *reinterpret_cast<const uint64_t *>(fill_data));
    else if((fill_size == 4) && ((alignment & 3) == 3))
      memset_1d_typed<uint32_t>(dst_base, bytes,
                                *reinterpret_cast<const uint32_t *>(fill_data));
    else if((fill_size == 2) && ((alignment & 1) == 1))
      memset_1d_typed<uint16_t>(dst_base, bytes,
                                *reinterpret_cast<const uint16_t *>(fill_data));
    else if(fill_size == 1)
      memset_1d_typed<uint8_t>(dst_base, bytes,
                               *reinterpret_cast<const uint8_t *>(fill_data));
    else {
      // fallback based on memcpy
      memcpy_2d(dst_base, fill_size, reinterpret_cast<uintptr_t>(fill_data), 0, fill_size,
                bytes / fill_size);
    }
  }

  void memset_2d(uintptr_t dst_base, uintptr_t dst_lstride, size_t bytes, size_t lines,
                 const void *fill_data, size_t fill_size)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment = ((dst_base - 1) & (dst_lstride - 1) & (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memset_2d: dst=" << dst_base << "+" << dst_lstride
                   << std::dec << " bytes=" << bytes << " lines=" << lines
                   << " align=" << (alignment & 31);
#endif
    // alignment must be at least as good as fill size to use memset
    // TODO: consider jump table approach?
    if((fill_size == 32) && ((alignment & 31) == 31))
      memset_2d_typed<aligned_32b_t>(dst_base, dst_lstride, bytes, lines,
                                     *reinterpret_cast<const aligned_32b_t *>(fill_data));
    else if((fill_size == 16) && ((alignment & 15) == 15))
      memset_2d_typed<aligned_16b_t>(dst_base, dst_lstride, bytes, lines,
                                     *reinterpret_cast<const aligned_16b_t *>(fill_data));
    else if((fill_size == 8) && ((alignment & 7) == 7))
      memset_2d_typed<uint64_t>(dst_base, dst_lstride, bytes, lines,
                                *reinterpret_cast<const uint64_t *>(fill_data));
    else if((fill_size == 4) && ((alignment & 3) == 3))
      memset_2d_typed<uint32_t>(dst_base, dst_lstride, bytes, lines,
                                *reinterpret_cast<const uint32_t *>(fill_data));
    else if((fill_size == 2) && ((alignment & 1) == 1))
      memset_2d_typed<uint16_t>(dst_base, dst_lstride, bytes, lines,
                                *reinterpret_cast<const uint16_t *>(fill_data));
    else if(fill_size == 1)
      memset_2d_typed<uint8_t>(dst_base, dst_lstride, bytes, lines,
                               *reinterpret_cast<const uint8_t *>(fill_data));
    else {
      // fallback based on memcpy
      memcpy_3d(dst_base, fill_size, dst_lstride, reinterpret_cast<uintptr_t>(fill_data),
                0, 0, fill_size, bytes / fill_size, lines);
    }
  }

  void memset_3d(uintptr_t dst_base, uintptr_t dst_lstride, uintptr_t dst_pstride,
                 size_t bytes, size_t lines, size_t planes, const void *fill_data,
                 size_t fill_size)
  {
    // by subtracting 1 from bases, strides, and lengths, we get LSBs set
    //  based on the common alignment of every parameter in the copy
    unsigned alignment =
        ((dst_base - 1) & (dst_lstride - 1) & (dst_pstride - 1) & (bytes - 1));
#ifdef DEBUG_MEMCPYS
    log_xd.print() << std::hex << "memset_3d: dst=" << dst_base << "+" << dst_lstride
                   << "+" << dst_pstride << std::dec << " bytes=" << bytes
                   << " lines=" << lines << " planes=" << planes
                   << " align=" << (alignment & 31);
#endif
    // alignment must be at least as good as fill size to use memset
    // TODO: consider jump table approach?
    if((fill_size == 32) && ((alignment & 31) == 31))
      memset_3d_typed<aligned_32b_t>(dst_base, dst_lstride, dst_pstride, bytes, lines,
                                     planes,
                                     *reinterpret_cast<const aligned_32b_t *>(fill_data));
    else if((fill_size == 16) && ((alignment & 15) == 15))
      memset_3d_typed<aligned_16b_t>(dst_base, dst_lstride, dst_pstride, bytes, lines,
                                     planes,
                                     *reinterpret_cast<const aligned_16b_t *>(fill_data));
    else if((fill_size == 8) && ((alignment & 7) == 7))
      memset_3d_typed<uint64_t>(dst_base, dst_lstride, dst_pstride, bytes, lines, planes,
                                *reinterpret_cast<const uint64_t *>(fill_data));
    else if((fill_size == 4) && ((alignment & 3) == 3))
      memset_3d_typed<uint32_t>(dst_base, dst_lstride, dst_pstride, bytes, lines, planes,
                                *reinterpret_cast<const uint32_t *>(fill_data));
    else if((fill_size == 2) && ((alignment & 1) == 1))
      memset_3d_typed<uint16_t>(dst_base, dst_lstride, dst_pstride, bytes, lines, planes,
                                *reinterpret_cast<const uint16_t *>(fill_data));
    else if(fill_size == 1)
      memset_3d_typed<uint8_t>(dst_base, dst_lstride, dst_pstride, bytes, lines, planes,
                               *reinterpret_cast<const uint8_t *>(fill_data));
    else {
      // fallback based on memcpy
      for(size_t p = 0; p < planes; p++)
        memcpy_3d(dst_base + (p * dst_pstride), fill_size, dst_lstride,
                  reinterpret_cast<uintptr_t>(fill_data), 0, 0, fill_size,
                  bytes / fill_size, lines);
    }
  }

  void enumerate_local_cpu_memories(const Node *node, std::vector<Memory> &mems)
  {
    for(std::vector<MemoryImpl *>::const_iterator it = node->memories.begin();
        it != node->memories.end(); ++it) {
      if(((*it)->lowlevel_kind == Memory::SYSTEM_MEM) ||
         ((*it)->lowlevel_kind == Memory::REGDMA_MEM) ||
         ((*it)->lowlevel_kind == Memory::Z_COPY_MEM) ||
         ((*it)->lowlevel_kind == Memory::SOCKET_MEM) ||
         ((*it)->lowlevel_kind == Memory::GPU_MANAGED_MEM)) {
        mems.push_back((*it)->me);
      }
    }

    for(std::vector<IBMemory *>::const_iterator it = node->ib_memories.begin();
        it != node->ib_memories.end(); ++it) {
      if(((*it)->lowlevel_kind == Memory::SYSTEM_MEM) ||
         ((*it)->lowlevel_kind == Memory::REGDMA_MEM) ||
         ((*it)->lowlevel_kind == Memory::Z_COPY_MEM) ||
         ((*it)->lowlevel_kind == Memory::SOCKET_MEM) ||
         ((*it)->lowlevel_kind == Memory::GPU_MANAGED_MEM)) {
        mems.push_back((*it)->me);
      }
    }
  }
}; // namespace Realm
