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

#ifndef REALM_CACHING_ALLOCATOR_H
#define REALM_CACHING_ALLOCATOR_H

#include "realm/atomics.h"
#include "realm/compiler_support.h"
#include "realm/lists.h"
#include "realm/mutex.h"

#include <memory>

namespace Realm {

template <typename T, size_t N> class CachingAllocator {
public:
  typedef T value_type;
  enum { BLOCK_SIZE = N };

private:
  struct Block {
    struct Chunk {
      alignas(T) char data[sizeof(T)];
      IntrusiveListLink<Chunk> chunk_link;
      // Index in the block list.
      // TODO: remove if the blocks are aligned properly
      size_t idx;
      REALM_PMTA_DEFN(Chunk, IntrusiveListLink<Chunk>, chunk_link);
    };

    atomic<ssize_t> num_alloced_chunks;
    Chunk chunks[BLOCK_SIZE];
    atomic<Chunk *> free_chunk_head;

    IntrusiveListLink<Block> block_link;
    REALM_PMTA_DEFN(Block, IntrusiveListLink<Block>, block_link);
    struct BlockList
        : public IntrusiveList<Block, REALM_PMTA_USE(Block, block_link),
                               Mutex> {
      ~BlockList(void) {
        // This should only be called on process exit, and is only here to clean
        // up asan.
        Block *ptr = this->head.next;
        while (ptr != nullptr) {
          Block *p = ptr->block_link.next;
          ptr->block_link.next = nullptr;
          delete ptr;
          ptr = p;
        }
        this->head.next = nullptr;
      }
    };

    Block() : num_alloced_chunks(0), free_chunk_head(&chunks[0]), block_link() {
      for (size_t i = 0; i < BLOCK_SIZE - 1; i++) {
        chunks[i].chunk_link.next = &chunks[i + 1];
        chunks[i].idx = i;
      }
      chunks[BLOCK_SIZE - 1].idx = BLOCK_SIZE - 1;
      chunks[BLOCK_SIZE - 1].chunk_link.next = nullptr;
    }

    ~Block() {
      assert((num_alloced_chunks.load() == 0) &&
             "Not all chunks have been freed!");
      // If we're part of the global list, make sure to clean up the rest of the
      // list (only done on process exit)
      if (block_link.next != nullptr) {
        delete block_link.next;
      }
    }

    void *alloc_obj() {
      Chunk *old_head = nullptr;
      ssize_t expected_full_size = BLOCK_SIZE;

      if (num_alloced_chunks.compare_exchange(expected_full_size, 0)) {
        // All full up, the block is now flagged for reclaimation
        return nullptr;
      } else {
        assert((expected_full_size >= 0) &&
               "Tried to allocate from a block marked for reclaimation!");
      }

      old_head = free_chunk_head.load_acquire();
      assert((old_head != nullptr) && "Non-empty block with no free chunks!");

      // Only the owning thread can pop off the free list, so no ABA issue here
      // and we know from num_alloced_chunks that there must at least be one
      // item in the block (old_head cannot be null)
      Chunk * next = nullptr;
      do {
        next = old_head->chunk_link.next;
      } while (!free_chunk_head.compare_exchange_weak(old_head, next));

      old_head->chunk_link.next = nullptr;
      num_alloced_chunks.fetch_add_acqrel(1);

      return &old_head->data;
    }
    bool free_obj(void *p) {
      Chunk *chunk = get_chunk_from_obj(p);
      Chunk *old_head = free_chunk_head.load_acquire();
      // Multiple threads can push onto the free list at once
      do {
        chunk->chunk_link.next = old_head;
      } while (!free_chunk_head.compare_exchange_weak(old_head, chunk));
      // If this block was flagged for reclaimation, then its allocated chunks
      // are negative and it'll be empty if instead of zero, the number of
      // allocated blocks is -BLOCK_SIZE.  Tell the caller so it can append the
      // block to the global block list
      return (num_alloced_chunks.fetch_sub_acqrel(1) - 1) == -BLOCK_SIZE;
    }

    static Chunk *get_chunk_from_obj(void *p) {
      return reinterpret_cast<Chunk *>(reinterpret_cast<uintptr_t>(p) -
                                       offsetof(Chunk, data));
    }

    static Block *get_block_from_obj(void *p) {
      // TODO: replace with pointer arithmetic using the alignment of the block
      // class.
      Chunk *chunk = get_chunk_from_obj(p);
      return reinterpret_cast<Block *>(
          reinterpret_cast<uintptr_t>(chunk - chunk->idx) -
          offsetof(Block, chunks));
    }
  };

  static typename Block::BlockList free_blocks;

  // When a thread exits, we want to make sure it's current block is marked for
  // reclaimation rather than deleted
  static void release_block(Block *blk) {
    ssize_t old_num = blk->num_alloced_chunks.load();
    while (old_num > 0 &&
           !blk->num_alloced_chunks.compare_exchange_weak(old_num, -old_num))
      ;
    // Delete the block if there aren't any outstanding references
    if (old_num == 0) {
      delete blk;
    }
  }

public:
  static void *alloc_obj() {
    // We use "thread_local" (since C++11) instead of REALM_THREAD_LOCAL here to
    // leverage C++ TLS, which works with C++ constructors and destructors
    static thread_local std::unique_ptr<Block, decltype(&release_block)>
        current_block(nullptr, release_block);
    void *obj = nullptr;

    if (current_block != nullptr) {
      obj = current_block->alloc_obj();
    }

    if (obj == nullptr) {
      Block *newblk = free_blocks.pop_front();
      if (newblk == nullptr) {
        newblk = new (std::nothrow) Block;
      }
      if (newblk != nullptr) {
        obj = newblk->alloc_obj();
        assert((obj != nullptr) && "Newly acquired block can't allocate!");
        current_block.release();
        current_block.reset(newblk);
      }
    }

    return obj;
  }
  static void free_obj(void *p) {
    Block *block = Block::get_block_from_obj(p);
    if (block->free_obj(p)) {
      // This block is empty and ready for reuse!
      // Push to the front for better cache locality!
      block->num_alloced_chunks.store_release(0);
      free_blocks.push_front(block);
    }
  }
};

template <typename T, size_t N>
typename CachingAllocator<T, N>::Block::BlockList
    CachingAllocator<T, N>::free_blocks;

} // namespace Realm

#endif // ifndef REALM_CACHING_ALLOCATOR_H