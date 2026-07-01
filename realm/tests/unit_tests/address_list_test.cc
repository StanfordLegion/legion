/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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

#include "realm/runtime_impl.h"
#include "realm/transfer/address_list.h"
#include <gtest/gtest.h>
#include <cstdlib>

using namespace Realm;

namespace {

  constexpr size_t kStride = 8;
  constexpr size_t kBytes = 1024;

  struct MockHeap {
    void *alloc_obj(std::size_t bytes, std::size_t align = 16)
    {
      void *ptr = nullptr;
#ifdef REALM_ON_WINDOWS
      ptr = _aligned_malloc(bytes, align);
#else
      int ret = posix_memalign(&ptr, align, bytes);
      if(ret != 0)
        ptr = nullptr;
#endif
      assert(ptr != nullptr);
      return ptr;
    }

    void free_obj(void *ptr)
    {
#ifdef REALM_ON_WINDOWS
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
  };

  static void make_1d_entry(AddressList &alist, size_t bytes, int payload = 0)
  {
    size_t *e = alist.begin_entry(1);
    ASSERT_NE(e, nullptr);
    e[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(bytes, 1);
    alist.commit_entry(1, bytes);
  }

  TEST(AddressListTests, AdvanceWithFieldsBasic)
  {
    AddressList addrlist;

    std::vector<int> fields = {100, 101, 102, 103};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, fields.data(), fields.size());
    addrlist.attach_field_block(fb);

    size_t *entry = addrlist.begin_entry(1);
    ASSERT_NE(entry, nullptr);
    entry[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(kBytes, 1);
    addrlist.commit_entry(1, kBytes);

    ASSERT_NE(fb, nullptr);
    ASSERT_EQ(fb->count, fields.size());
    ASSERT_NE(fb->fields, nullptr);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);

    EXPECT_EQ(addrlist.bytes_pending(), kBytes * fields.size());

    ASSERT_EQ(cursor.remaining_fields(), fields.size());
    cursor.advance(0, 128, 1);
    EXPECT_EQ(addrlist.bytes_pending(), kBytes * fields.size() - 128);
    ASSERT_EQ(cursor.remaining_fields(), fields.size());

    cursor.advance(0, 128, 1);
    EXPECT_EQ(addrlist.bytes_pending(), kBytes * fields.size() - 256);
    ASSERT_EQ(cursor.remaining_fields(), fields.size());

    cursor.advance(0, kBytes - 256, 1);
    EXPECT_EQ(addrlist.bytes_pending(), kBytes * (fields.size() - 1));
    ASSERT_EQ(cursor.remaining_fields(), fields.size() - 1);

    cursor.advance(0, kBytes, fields.size() - 1);
    EXPECT_EQ(addrlist.bytes_pending(), 0);
    // Fields should be reset once we finished the rect entry
    ASSERT_EQ(cursor.remaining_fields(), fields.size());

    heap.free_obj(fb);
  }

  TEST(AdvanceTests, WithFields_PartialDoesNotIncrementField)
  {
    AddressList al;
    std::vector<int> ids = {1, 2, 3, 4};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, ids.data(), ids.size());
    al.attach_field_block(fb);
    make_1d_entry(al, kBytes, ids.size());

    AddressListCursor cur;
    cur.set_addrlist(&al);

    // small amt < full rect: field stays at 0
    cur.advance(0, 128, 1);
    EXPECT_EQ(cur.fields_data(), fb->fields + 0);
    EXPECT_EQ(al.bytes_pending(), kBytes * ids.size() - 128);
    ASSERT_EQ(cur.remaining_fields(), ids.size());

    heap.free_obj(fb);
  }

  TEST(AdvanceTests, WithFields_FullRectSingleField)
  {
    AddressList al;
    std::vector<int> ids = {1, 2, 3, 4};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, ids.data(), ids.size());
    al.attach_field_block(fb);
    make_1d_entry(al, kBytes, ids.size());

    AddressListCursor cur;
    cur.set_addrlist(&al);

    // full rect for 1 field
    cur.advance(0, kBytes, 1);
    EXPECT_EQ(cur.fields_data(), fb->fields + 1);
    // entry still pending until all 4 fields done
    EXPECT_EQ(al.bytes_pending(), kBytes * ids.size() - kBytes);
    ASSERT_EQ(cur.remaining_fields(), ids.size() - 1);

    heap.free_obj(fb);
  }

  TEST(AdvanceTests, WithFields_FullRectMultipleFieldsAtOnce)
  {
    AddressList al;
    std::vector<int> ids = {1, 2, 3, 4};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, ids.data(), ids.size());
    al.attach_field_block(fb);
    make_1d_entry(al, kBytes, ids.size());

    AddressListCursor cur;
    cur.set_addrlist(&al);

    // consume 2 fields in one go
    cur.advance(0, kBytes, 2);
    EXPECT_EQ(cur.fields_data(), fb->fields + 2);
    EXPECT_EQ(al.bytes_pending(), kBytes * ids.size() - 2 * kBytes);
    ASSERT_EQ(cur.remaining_fields(), ids.size() - 2);

    heap.free_obj(fb);
  }

  TEST(AdvanceTests, WithFields_ConsumeAllFieldsAtOnce)
  {
    AddressList al;
    std::vector<int> ids = {1, 2, 3, 4};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, ids.data(), ids.size());
    al.attach_field_block(fb);
    make_1d_entry(al, kBytes, ids.size());

    AddressListCursor cur;
    cur.set_addrlist(&al);

    // consume all fields => entry consumed
    cur.advance(0, kBytes, ids.size());
    EXPECT_EQ(al.bytes_pending(), 0);
    EXPECT_EQ(cur.fields_data(), fb->fields);
    ASSERT_EQ(cur.remaining_fields(), ids.size());

    heap.free_obj(fb);
  }

  TEST(AdvanceTests, WithFields_SequentialFullRect)
  {
    AddressList al;
    std::vector<int> ids = {1, 2, 3, 4};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, ids.data(), ids.size());
    al.attach_field_block(fb);
    make_1d_entry(al, kBytes, ids.size());

    AddressListCursor cur;
    cur.set_addrlist(&al);

    // call advance(full,1) four times
    for(int i = 0; i < (int)ids.size(); i++) {
      EXPECT_EQ(cur.fields_data(), (i < (int)ids.size() ? fb->fields + i : fb->fields));
      cur.advance(0, kBytes, 1);
    }
    EXPECT_EQ(al.bytes_pending(), 0);

    heap.free_obj(fb);
  }

  TEST(AdvanceTests, MultiDim_WithFields)
  {
    AddressList al;
    std::vector<int> ids = {7, 8, 9};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, ids.data(), ids.size());
    al.attach_field_block(fb);

    size_t *entry = al.begin_entry(3);
    entry[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(kBytes, 3);
    entry[AddressList::SLOT_BASE] = 0;            // base offset
    entry[AddressList::DIM_SLOTS * 1] = 8;        // dim1 count
    entry[AddressList::DIM_SLOTS * 1 + 1] = 1024; // dim1 stride
    entry[AddressList::DIM_SLOTS * 2] = 2;        // dim2 count
    entry[AddressList::DIM_SLOTS * 2 + 1] = 8192; // dim2 stride
    const size_t volume = kBytes * 8 * 2;
    al.commit_entry(3, volume);

    AddressListCursor cur;
    cur.set_addrlist(&al);

    EXPECT_EQ(al.bytes_pending(), volume * ids.size());
    cur.advance(2, 2);
    EXPECT_EQ(al.bytes_pending(), volume * (ids.size() - 1));

    cur.advance(2, 2);
    EXPECT_EQ(al.bytes_pending(), volume * (ids.size() - 2));

    cur.advance(2, 2);
    EXPECT_EQ(al.bytes_pending(), volume * (ids.size() - 3));

    heap.free_obj(fb);
  }

  TEST(AdvanceTests, MultiDim_WithFieldsSingleAdvance)
  {
    AddressList al;
    std::vector<int> ids = {7, 8, 9};
    MockHeap heap;
    auto *fb = FieldBlock::create(heap, ids.data(), ids.size());
    al.attach_field_block(fb);

    size_t *entry = al.begin_entry(3);
    entry[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(kBytes, 3);
    entry[AddressList::SLOT_BASE] = 0;            // base offset
    entry[AddressList::DIM_SLOTS * 1] = 8;        // dim1 count
    entry[AddressList::DIM_SLOTS * 1 + 1] = 1024; // dim1 stride
    entry[AddressList::DIM_SLOTS * 2] = 2;        // dim2 count
    entry[AddressList::DIM_SLOTS * 2 + 1] = 8192; // dim2 stride
    const size_t volume = kBytes * 8 * 2;
    al.commit_entry(3, volume);

    AddressListCursor cur;
    cur.set_addrlist(&al);

    EXPECT_EQ(al.bytes_pending(), volume * ids.size());
    cur.advance(2, 2, ids.size());
    EXPECT_EQ(al.bytes_pending(), 0);

    heap.free_obj(fb);
  }

  TEST(AddressListTests, Basic1DEntryNoPayload)
  {
    AddressList addrlist;
    const int dim = 1;

    size_t *entry = addrlist.begin_entry(dim);
    ASSERT_NE(entry, nullptr);
    entry[0] = AddressList::pack_entry_header(kBytes, dim);
    addrlist.commit_entry(dim, kBytes);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);

    EXPECT_EQ(cursor.get_dim(), 1);
    EXPECT_EQ(cursor.remaining(0), kBytes);
    EXPECT_EQ(cursor.get_offset(), 0);

    cursor.advance(0, kStride);
    EXPECT_EQ(cursor.remaining(0), kBytes - kStride);

    cursor.skip_bytes(kStride);
    EXPECT_EQ(cursor.remaining(0), kBytes - 2 * kStride);

    cursor.advance(0, kBytes - 2 * kStride);
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, Multiple1DEntries)
  {
    AddressList addrlist;
    const size_t entries = 10;

    for(size_t i = 0; i < entries; ++i) {
      size_t *entry = addrlist.begin_entry(1);
      ASSERT_NE(entry, nullptr);
      entry[0] = AddressList::pack_entry_header(kBytes, 1);
      addrlist.commit_entry(1, kBytes);
    }

    EXPECT_EQ(addrlist.bytes_pending(), entries * kBytes);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);
    for(size_t i = 0; i < entries; ++i) {
      EXPECT_EQ(cursor.remaining(0), kBytes);
      cursor.advance(0, kBytes);
    }
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, AppendComplex3DEntry)
  {
    AddressList addrlist;

    std::unordered_map<int, std::pair<size_t, size_t>> count_strides;
    count_strides[1] = {8, 1024};
    count_strides[2] = {2, 8192};
    bool commited = addrlist.append_entry(3, kBytes, kBytes * 8 * 2, /*base_offset=*/16,
                                          count_strides);
    ASSERT_TRUE(commited);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);

    EXPECT_EQ(cursor.remaining(0), kBytes);
    EXPECT_EQ(cursor.remaining(1), 8);
    EXPECT_EQ(cursor.remaining(2), 2);
    EXPECT_EQ(cursor.get_offset(), 16);

    cursor.advance(2, 2);
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, Complex3DEntry)
  {
    AddressList addrlist;

    size_t *entry = addrlist.begin_entry(3);
    entry[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(kBytes, 3);
    entry[AddressList::SLOT_BASE] = 0;            // base offset
    entry[AddressList::DIM_SLOTS * 1] = 8;        // dim1 count
    entry[AddressList::DIM_SLOTS * 1 + 1] = 1024; // dim1 stride
    entry[AddressList::DIM_SLOTS * 2] = 2;        // dim2 count
    entry[AddressList::DIM_SLOTS * 2 + 1] = 8192; // dim2 stride
    const size_t volume = kBytes * 8 * 2;
    addrlist.commit_entry(3, volume);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);

    EXPECT_EQ(cursor.remaining(0), kBytes);
    EXPECT_EQ(cursor.remaining(1), 8);
    EXPECT_EQ(cursor.remaining(2), 2);
    EXPECT_EQ(cursor.get_offset(), 0);

    cursor.advance(2, 2);
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, WraparoundBufferSafety)
  {
    const size_t max_entries = 16;
    AddressList addrlist(max_entries);

    size_t successful = 0;
    while(true) {
      size_t *entry = addrlist.begin_entry(1);
      if(!entry)
        break;
      entry[0] = AddressList::pack_entry_header(kBytes, 1);
      addrlist.commit_entry(1, kBytes);
      successful++;
    }

    EXPECT_GT(successful, 0);
    EXPECT_LE(successful, max_entries);

    AddressListCursor cursor;
    cursor.set_addrlist(&addrlist);
    for(size_t i = 0; i < successful; ++i) {
      cursor.advance(0, kBytes);
    }
    EXPECT_EQ(addrlist.bytes_pending(), 0);
  }

  TEST(AddressListTests, FullFieldBytes1D)
  {
    AddressList addrlist;

    // Create a simple 1-D entry with kBytes contiguous bytes
    size_t *entry = addrlist.begin_entry(1);
    ASSERT_NE(entry, nullptr);
    entry[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(kBytes, 1);
    addrlist.commit_entry(1, kBytes);

    // full_field_bytes should return the contiguous byte count for 1-D entries
    EXPECT_EQ(addrlist.full_field_bytes(), kBytes);
  }

  TEST(AddressListTests, FullFieldBytes3D)
  {
    AddressList addrlist;

    // Build a 3-D entry: dim0 has kBytes contiguous bytes, dim1 has 8 elements,
    // and dim2 has 2 elements. The expected volume (per field) is kBytes * 8 * 2.
    size_t *entry = addrlist.begin_entry(3);
    ASSERT_NE(entry, nullptr);
    entry[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(kBytes, 3);
    entry[AddressList::SLOT_BASE] = 0;     // base offset (unused by full_field_bytes)
    entry[AddressList::DIM_SLOTS * 1] = 8; // dim1 count
    entry[AddressList::DIM_SLOTS * 1 + 1] =
        1024;                              // dim1 stride (unused by full_field_bytes)
    entry[AddressList::DIM_SLOTS * 2] = 2; // dim2 count
    entry[AddressList::DIM_SLOTS * 2 + 1] =
        8192; // dim2 stride (unused by full_field_bytes)
    const size_t expected_volume = kBytes * 8 * 2;
    addrlist.commit_entry(3, expected_volume);

    // full_field_bytes should compute the volume across all dimensions
    EXPECT_EQ(addrlist.full_field_bytes(), expected_volume);
  }

  TEST(AddressListTests, WithFields_2D_PartialLine_NoFieldAdvance)
  {
    constexpr size_t NUM_BYTES = 64;
    constexpr size_t NUM_LINES = 3;
    AddressList al;
    std::vector<int> f = {1, 2, 3};
    MockHeap h;
    FieldBlock *fb = FieldBlock::create(h, f.data(), f.size());
    al.attach_field_block(fb);

    size_t *e = al.begin_entry(2);
    e[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(NUM_BYTES, 2);
    e[AddressList::DIM_SLOTS * 1] = NUM_LINES;
    e[AddressList::DIM_SLOTS * 1 + 1] = NUM_BYTES;
    al.commit_entry(2, NUM_BYTES * NUM_LINES);

    AddressListCursor c;
    c.set_addrlist(&al);

    // Consume half a line – must NOT change field index
    c.advance(0, NUM_BYTES / 2, 1);
    EXPECT_EQ(c.fields_data(), fb->fields);
    EXPECT_EQ(al.bytes_pending(), NUM_BYTES * NUM_LINES * f.size() - NUM_BYTES / 2);

    h.free_obj(fb);
  }

  TEST(AddressListTests, WithFields_PartialThenFullRect)
  {
    constexpr size_t NUM_BYTES = 32;
    AddressList al;
    std::vector<int> f = {0, 1, 2};
    MockHeap h;
    FieldBlock *fb = FieldBlock::create(h, f.data(), f.size());
    al.attach_field_block(fb);
    make_1d_entry(al, NUM_BYTES);

    AddressListCursor cur;
    cur.set_addrlist(&al);

    // (1) move full rect for two fields at once
    cur.advance(0, NUM_BYTES, 2);
    EXPECT_EQ(cur.fields_data(), fb->fields + 2);
    EXPECT_EQ(al.bytes_pending(), NUM_BYTES * (f.size() - 2));

    // (2) move half-rect – must stay on field #2
    cur.advance(0, NUM_BYTES / 2, 1);
    EXPECT_EQ(cur.fields_data(), fb->fields + 2);
    EXPECT_EQ(al.bytes_pending(), NUM_BYTES * (f.size() - 2) - NUM_BYTES / 2);

    // (3) finish the rect – now field pointer wraps to beginning
    cur.advance(0, NUM_BYTES / 2, 1);
    EXPECT_EQ(cur.fields_data(), fb->fields);
    EXPECT_EQ(al.bytes_pending(), 0u);

    h.free_obj(fb);
  }

  TEST(AddressListTests, WithFields_3D_PlaneWisePartial)
  {
    constexpr size_t NUM_BYTES = 16;
    constexpr size_t NUM_LINES = 2;
    constexpr size_t NUM_PLANES = 3;
    AddressList al;
    std::vector<int> f = {9, 9};
    MockHeap h;
    FieldBlock *fb = FieldBlock::create(h, f.data(), f.size());
    al.attach_field_block(fb);

    size_t *e = al.begin_entry(3);
    e[AddressList::SLOT_HEADER] = AddressList::pack_entry_header(NUM_BYTES, 3);
    e[AddressList::DIM_SLOTS * 1] = NUM_LINES;
    e[AddressList::DIM_SLOTS * 1 + 1] = NUM_BYTES;
    e[AddressList::DIM_SLOTS * 2] = NUM_PLANES;
    e[AddressList::DIM_SLOTS * 2 + 1] = NUM_BYTES * NUM_LINES;
    al.commit_entry(3, NUM_BYTES * NUM_LINES * NUM_PLANES);

    AddressListCursor c;
    c.set_addrlist(&al);

    // consume one plane at a time – still field 0
    for(size_t p = 0; p < NUM_PLANES; ++p) {
      c.advance(2, 1);
    }

    EXPECT_EQ(al.bytes_pending(), NUM_BYTES * NUM_LINES * NUM_PLANES * (f.size() - 1));
    EXPECT_EQ(c.fields_data(), fb->fields + 1);

    // now full rect for one field
    c.advance(2, NUM_PLANES, 1);
    EXPECT_EQ(al.bytes_pending(), 0u);
    EXPECT_EQ(c.fields_data(), fb->fields);

    h.free_obj(fb);
  }

} // namespace
