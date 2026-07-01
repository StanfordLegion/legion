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

#include "realm/transfer/address_list.h"
#include "realm/cuda/cuda_memcpy.h"
#include "realm/cuda/cuda_internal.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <cstdlib>
#include <numeric>

using namespace Realm;
using namespace Realm::Cuda;

namespace {

  // === Utility helpers for the tests
  // ==============================================================

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

  static void append_entry_1d(AddressList &al, size_t contig_bytes, size_t base = 0,
                              bool wrap_mode = true)
  {
    bool ret =
        al.append_entry(/*dims=*/1, contig_bytes, contig_bytes, base, {}, wrap_mode);
    assert(ret);
  }

  static void append_entry_2d(AddressList &al, size_t contig_bytes, size_t lines,
                              size_t base = 0, bool wrap_mode = true)
  {
    std::unordered_map<int, std::pair<size_t, size_t>> count_strides;
    // dim 1 : {count, stride}
    count_strides[1] = {lines, contig_bytes};
    bool ret = al.append_entry(/*dims=*/2, contig_bytes, contig_bytes * lines, base,
                               count_strides, wrap_mode);
    assert(ret);
  }

  static void append_entry_3d(AddressList &al, size_t contig_bytes, size_t lines,
                              size_t planes, size_t base = 0, bool wrap_mode = true)
  {
    std::unordered_map<int, std::pair<size_t, size_t>> count_strides;
    count_strides[1] = {lines, contig_bytes};
    count_strides[2] = {planes, contig_bytes * lines};
    bool ret = al.append_entry(/*dims=*/3, contig_bytes, contig_bytes * lines * planes,
                               base, count_strides, wrap_mode);
    assert(ret);
  }

} // anonymous namespace

// ================================================================================================
//                                          TESTS
// ================================================================================================

TEST(ReadAddressEntryTests, FastPath1D_NoFields)
{
  constexpr size_t CONTIG = 64;
  AddressList in_al, out_al;
  append_entry_1d(in_al, CONTIG);
  append_entry_1d(out_al, CONTIG);

  AddressListCursor in_cur, out_cur;
  in_cur.set_addrlist(&in_al);
  out_cur.set_addrlist(&out_al);

  AffineCopyInfo<3> copy_infos{};
  MemcpyTransposeInfo<size_t> transpose_info{};
  size_t min_align = 16;
  size_t fields_tot = 0;
  const size_t bytes_left = CONTIG;
  const size_t max_fields = 8;

  size_t ret_bytes = GPUXferDes::read_address_entry(
      copy_infos, min_align, transpose_info, in_cur,
      /*in_base=*/0, out_cur, /*out_base=*/0, bytes_left, max_fields, fields_tot);

  ASSERT_EQ(ret_bytes, CONTIG);
  ASSERT_EQ(copy_infos.num_rects, 1u);
  const auto &ci = copy_infos.subrects[0];
  EXPECT_EQ(ci.extents[0], CONTIG);
  EXPECT_EQ(ci.volume, CONTIG);
  EXPECT_EQ(ci.src.strides[0], CONTIG);
  EXPECT_EQ(ci.dst.strides[0], CONTIG);
  EXPECT_EQ(fields_tot, 1u);
  EXPECT_EQ(in_al.bytes_pending(), 0u);
  EXPECT_EQ(out_al.bytes_pending(), 0u);
}

TEST(ReadAddressEntryTests, TwoDimensionalSplit_FirstDim)
{
  constexpr size_t CONTIG = 32;
  constexpr size_t LINES = 4;
  const size_t kTotalBytes = CONTIG * LINES;

  AddressList in_al, out_al;
  append_entry_2d(in_al, CONTIG, LINES);
  append_entry_2d(out_al, CONTIG, LINES);

  AddressListCursor in_cur, out_cur;
  in_cur.set_addrlist(&in_al);
  out_cur.set_addrlist(&out_al);

  AffineCopyInfo<3> copy_infos{};
  MemcpyTransposeInfo<size_t> transpose_info{};
  size_t min_align = 16;
  size_t fields_tot = 0;
  const size_t max_fields = 8;

  size_t ret_bytes =
      GPUXferDes::read_address_entry(copy_infos, min_align, transpose_info, in_cur, 0,
                                     out_cur, 0, kTotalBytes, max_fields, fields_tot);

  ASSERT_EQ(ret_bytes, kTotalBytes);
  ASSERT_EQ(copy_infos.num_rects, 1u);
  const auto &ci = copy_infos.subrects[0];
  EXPECT_EQ(ci.extents[0], CONTIG);
  EXPECT_EQ(ci.extents[1], LINES);
  EXPECT_EQ(ci.extents[2], 1u);
  EXPECT_EQ(ci.volume, kTotalBytes);
  EXPECT_EQ(ci.src.strides[0], CONTIG);
  EXPECT_EQ(ci.src.strides[1], LINES);
  EXPECT_EQ(ci.dst.strides[0], CONTIG);
  EXPECT_EQ(ci.dst.strides[1], LINES);
  EXPECT_EQ(fields_tot, 1); // TODO: fix
}

TEST(ReadAddressEntryTests, ThreeDimensional_NoTranspose)
{
  constexpr size_t CONTIG = 16;
  constexpr size_t LINES = 4;
  constexpr size_t PLANES = 2;
  const size_t kTotalBytes = CONTIG * LINES * PLANES;

  AddressList in_al, out_al;
  append_entry_3d(in_al, CONTIG, LINES, PLANES);
  append_entry_3d(out_al, CONTIG, LINES, PLANES);

  AddressListCursor in_cur, out_cur;
  in_cur.set_addrlist(&in_al);
  out_cur.set_addrlist(&out_al);

  AffineCopyInfo<3> copy_infos{};
  MemcpyTransposeInfo<size_t> transpose_info{};
  size_t min_align = 16;
  size_t fields_tot = 0;

  const size_t max_fields = 8;
  size_t ret_bytes =
      GPUXferDes::read_address_entry(copy_infos, min_align, transpose_info, in_cur, 0,
                                     out_cur, 0, kTotalBytes, max_fields, fields_tot);

  ASSERT_EQ(ret_bytes, kTotalBytes);
  ASSERT_EQ(copy_infos.num_rects, 1u);
  const auto &ci = copy_infos.subrects[0];
  EXPECT_EQ(ci.extents[0], CONTIG);
  EXPECT_EQ(ci.extents[1], LINES);
  EXPECT_EQ(ci.extents[2], PLANES);
  EXPECT_EQ(ci.volume, kTotalBytes);
  EXPECT_EQ(ci.src.strides[0], CONTIG);
  EXPECT_EQ(ci.src.strides[1], LINES);
  EXPECT_EQ(ci.dst.strides[0], CONTIG);
  EXPECT_EQ(ci.dst.strides[1], LINES);
  EXPECT_EQ(fields_tot, 1);
}

TEST(ReadAddressEntryTests, FieldBlock_LimitedTransfer)
{
  constexpr size_t CONTIG = 32;
  constexpr int kTotalFields = 5;
  constexpr int kMaxFieldsPerXfer = 2;

  AddressList in_al, out_al;
  // Attach field blocks BEFORE appending the entry so that the byte accounting
  // inside AddressList includes all fields.
  std::vector<int> field_ids(kTotalFields);
  std::iota(field_ids.begin(), field_ids.end(), 0);

  MockHeap heap;
  auto *fb_in = FieldBlock::create(heap, field_ids.data(), field_ids.size());
  auto *fb_out = FieldBlock::create(heap, field_ids.data(), field_ids.size());

  in_al.attach_field_block(fb_in);
  out_al.attach_field_block(fb_out);

  append_entry_1d(in_al, CONTIG);
  append_entry_1d(out_al, CONTIG);

  AddressListCursor in_cur, out_cur;
  in_cur.set_addrlist(&in_al);
  out_cur.set_addrlist(&out_al);

  AffineCopyInfo<3> copy_infos{};
  MemcpyTransposeInfo<size_t> transpose_info{};
  size_t min_align = 16;
  size_t fields_tot = 0;

  size_t ret_bytes = GPUXferDes::read_address_entry(
      copy_infos, min_align, transpose_info, in_cur, 0, out_cur, 0, CONTIG * kTotalFields,
      kMaxFieldsPerXfer, fields_tot);

  ASSERT_EQ(copy_infos.num_rects, 1u);
  const auto &ci = copy_infos.subrects[0];

  EXPECT_EQ(fields_tot, static_cast<size_t>(kMaxFieldsPerXfer));
  EXPECT_EQ(ci.src.num_fields, static_cast<size_t>(kMaxFieldsPerXfer));
  EXPECT_EQ(ci.dst.num_fields, static_cast<size_t>(kMaxFieldsPerXfer));
  EXPECT_EQ(ret_bytes, CONTIG * kMaxFieldsPerXfer);
  EXPECT_EQ(in_al.bytes_pending(), CONTIG * (kTotalFields - kMaxFieldsPerXfer));
  EXPECT_EQ(out_al.bytes_pending(), CONTIG * (kTotalFields - kMaxFieldsPerXfer));

  heap.free_obj(fb_in);
  heap.free_obj(fb_out);
}

// ─────────────────────────────────────────────────────────────────────────────
//  (A)  Partial-field loop: move at most Fmax fields per iteration until done
// ─────────────────────────────────────────────────────────────────────────────
TEST(ReadAddressEntry, Loop_PartialFieldConsumption)
{
  constexpr size_t C = 64; // bytes per rectangle
  constexpr int Ftot = 7;  // total number of fields
  constexpr int Fmax = 3;  // copy at most this many per iteration

  // Build identical src/dst AddressLists and attach FieldBlocks
  AddressList in_al, out_al;

  std::vector<int> ids(Ftot);
  std::iota(ids.begin(), ids.end(), 0);
  MockHeap h;
  auto *fb_in = FieldBlock::create(h, ids.data(), Ftot);
  auto *fb_out = FieldBlock::create(h, ids.data(), Ftot);
  in_al.attach_field_block(fb_in);
  out_al.attach_field_block(fb_out);

  append_entry_1d(in_al, C);
  append_entry_1d(out_al, C);

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);

  size_t bytes_moved_total = 0;
  int fields_remaining = Ftot;

  while(in_al.bytes_pending() > 0) {
    AffineCopyInfo<3> info{};
    MemcpyTransposeInfo<size_t> tr{};
    size_t min_align = 16, fields_total = 0;

    // Allow up to `C * Fmax` bytes this round — enough for at most Fmax fields
    size_t moved = GPUXferDes::read_address_entry(info, min_align, tr, ic, 0, oc, 0,
                                                  C * Fmax, Fmax, fields_total);

    ASSERT_GT(moved, 0u);
    EXPECT_EQ(fields_total, static_cast<size_t>(std::min(Fmax, fields_remaining)));

    bytes_moved_total += moved;
    fields_remaining -= static_cast<int>(fields_total);
  }

  EXPECT_EQ(bytes_moved_total, C * Ftot);
  EXPECT_EQ(fields_remaining, 0);
  EXPECT_EQ(in_al.bytes_pending(), 0u);
  EXPECT_EQ(out_al.bytes_pending(), 0u);

  h.free_obj(fb_in);
  h.free_obj(fb_out);
}

// ─────────────────────────────────────────────────────────────────────────────
//  (B)  Partial-rect loop: move *bytes_left_step* at a time within ONE field
// ─────────────────────────────────────────────────────────────────────────────
TEST(ReadAddressEntry, Loop_PartialRectConsumption)
{
  constexpr size_t C = 64;    // full rectangle size
  constexpr size_t STEP = 16; // copy only 16 bytes per iteration

  AddressList in_al, out_al;
  append_entry_1d(in_al, C);
  append_entry_1d(out_al, C);

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);

  size_t bytes_moved_total = 0;

  while(in_al.bytes_pending() > 0) {
    AffineCopyInfo<3> info{};
    MemcpyTransposeInfo<size_t> tr{};
    size_t min_align = 16, fields_total = 0;

    size_t moved = GPUXferDes::read_address_entry(info, min_align, tr, ic, 0, oc, 0,
                                                  STEP, // bytes_left -> only STEP bytes
                                                  8,    // max_xfer_fields
                                                  fields_total);

    ASSERT_GT(moved, 0u);
    EXPECT_LE(moved, STEP);
    EXPECT_EQ(fields_total, 1u); // only one field in play

    bytes_moved_total += moved;
  }

  EXPECT_EQ(bytes_moved_total, C); // whole rectangle eventually moved
  EXPECT_EQ(in_al.bytes_pending(), 0u);
  EXPECT_EQ(out_al.bytes_pending(), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
//  (C)  3-D partial–rect loop : copy one plane at a time
// ─────────────────────────────────────────────────────────────────────────────
TEST(ReadAddressEntry, Loop_PartialRectConsumption3D)
{
  constexpr size_t CONTIG = 16;           // bytes per contiguous chunk (X)
  constexpr size_t LINES = 4;             // Y
  constexpr size_t PLANES = 5;            // Z   → total volume = 16 * 4 * 5 = 320
  constexpr size_t STEP = CONTIG * LINES; // 64 bytes ⇒ one full plane

  AddressList in_al, out_al;
  append_entry_3d(in_al, CONTIG, LINES, PLANES);
  append_entry_3d(out_al, CONTIG, LINES, PLANES);

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);

  size_t total_moved = 0;

  while(in_al.bytes_pending() > 0) {
    AffineCopyInfo<3> info{};
    MemcpyTransposeInfo<size_t> tr{};
    size_t min_align = 16, fields_total = 0;

    size_t moved = GPUXferDes::read_address_entry(info, min_align, tr, ic, 0, oc, 0,
                                                  STEP, // allow only one plane’s bytes
                                                  8,    // unlimited fields (1 here)
                                                  fields_total);

    ASSERT_GT(moved, 0u);
    EXPECT_LE(moved, STEP);     // never more than a plane
    EXPECT_EQ(fields_total, 1); // TODO: fix
    total_moved += moved;
  }

  EXPECT_EQ(total_moved, CONTIG * LINES * PLANES); // all bytes copied
  EXPECT_EQ(in_al.bytes_pending(), 0u);
  EXPECT_EQ(out_al.bytes_pending(), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
//  2-D entry: source row 64 B, destination row 32 B
//  bytes_left = 64  → contig_bytes becomes 32 (< icount 64) so id = od = 0 path
// ─────────────────────────────────────────────────────────────────────────────
TEST(ReadAddressEntry, TwoDimensional_SplitDim0_OneShot)
{
  constexpr size_t SRC_CONTIG = 64; // src row width  (icount starts at 64)
  constexpr size_t DST_CONTIG = 32; // dst row width  (ocount 32)
  constexpr size_t LINES = 2;       // second dimension
  constexpr size_t BYTES_LEFT = 64; // > dst row, < src row  ⇒ split path

  // Build AddressLists
  AddressList in_al, out_al;
  append_entry_2d(in_al, SRC_CONTIG, LINES);  // 64-byte rows
  append_entry_2d(out_al, DST_CONTIG, LINES); // 32-byte rows

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);

  AffineCopyInfo<3> info{};
  MemcpyTransposeInfo<size_t> tr{};
  size_t min_align = 16, fields_total = 0;

  // Call read_address_entry – should copy *one* 32-byte line
  size_t moved = GPUXferDes::read_address_entry(info, min_align, tr, ic, 0, oc, 0,
                                                BYTES_LEFT,       // flow-control limit
                                                8 /*max fields*/, // no FieldBlocks
                                                fields_total);

  // ─── Checks ───────────────────────────────────────────────────────────────
  EXPECT_EQ(moved, SRC_CONTIG); // 64 bytes copied
  EXPECT_EQ(info.num_rects, 1u);

  const auto &rect = info.subrects[0];
  EXPECT_EQ(rect.extents[0], DST_CONTIG); // contig reduced to 32
  EXPECT_EQ(rect.extents[1], LINES);

  EXPECT_EQ(rect.src.strides[0], DST_CONTIG);
  EXPECT_EQ(rect.src.strides[1], LINES);
  EXPECT_EQ(rect.dst.strides[0], DST_CONTIG);
  EXPECT_EQ(rect.dst.strides[1], LINES);

  EXPECT_EQ(rect.volume, SRC_CONTIG);
  EXPECT_EQ(fields_total, 1u); // TODO: fix

  // Remaining bytes:  (64-32) * 2 lines = 64 bytes still pending
  EXPECT_EQ(in_al.bytes_pending(), (SRC_CONTIG - DST_CONTIG) * LINES);
  EXPECT_EQ(out_al.bytes_pending(), 0);
}

TEST(ReadAddressEntryTests, SrcWithFields_DstNoFields_Partial)
{
  constexpr size_t kContig = 64;
  constexpr int kSrcFields = 5;
  constexpr int kMoveFields = 2;
  constexpr int kLeft = 3;
  constexpr size_t kBytesLeft = kContig * 5; // allow exactly one field worth of bytes

  // Build address lists
  AddressList in_al, out_al;

  // Attach field block to **source** only
  std::vector<int> src_field_ids(kSrcFields);
  std::iota(src_field_ids.begin(), src_field_ids.end(), 0);
  MockHeap heap;
  auto *fb_src = FieldBlock::create(heap, src_field_ids.data(), src_field_ids.size());
  in_al.attach_field_block(fb_src);

  append_entry_1d(in_al, kContig);
  append_entry_1d(out_al, kContig * kMoveFields + kLeft, 0);

  size_t bytes_left =
      std::min(kBytesLeft, std::min(in_al.bytes_pending(), out_al.bytes_pending()));

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);

  AffineCopyInfo<3> infos{};
  MemcpyTransposeInfo<size_t> tr{};
  size_t min_align = 16, fields_total = 0;

  const size_t max_fields = 8; // not the limiting factor here

  EXPECT_EQ(ic.get_offset(), 0);
  EXPECT_EQ(oc.get_offset(), 0);

  {
    size_t moved = GPUXferDes::read_address_entry(infos, min_align, tr, ic, 0, oc, 0,
                                                  bytes_left, max_fields, fields_total);
    // Expectations
    ASSERT_EQ(infos.num_rects, 1u);
    const auto &ci = infos.subrects[0];
    EXPECT_EQ(ci.extents[0], kContig);
    EXPECT_EQ(ci.volume, kContig);

    // Only one field can be moved because bytes_left == contig
    EXPECT_EQ(fields_total, kMoveFields);
    EXPECT_EQ(ci.src.num_fields, kMoveFields);
    EXPECT_EQ(ci.dst.num_fields, 0u);
    EXPECT_EQ(moved, kContig * kMoveFields);

    // Pending bytes: src started with 3*64, dst with 64
    EXPECT_EQ(in_al.bytes_pending(), kContig * (kSrcFields - kMoveFields));
    EXPECT_EQ(out_al.bytes_pending(), kLeft);

    EXPECT_EQ(ic.get_offset(), 0);
    EXPECT_EQ(oc.get_offset(), kContig * kMoveFields);
  }

  infos.num_rects = 0;
  bytes_left =
      std::min(kBytesLeft, std::min(in_al.bytes_pending(), out_al.bytes_pending()));

  {
    size_t moved = GPUXferDes::read_address_entry(infos, min_align, tr, ic, 0, oc, 0,
                                                  kLeft, max_fields, fields_total);

    // Expectations
    ASSERT_EQ(infos.num_rects, 1u);
    const auto &ci = infos.subrects[0];
    EXPECT_EQ(ci.extents[0], kLeft);
    EXPECT_EQ(ci.volume, kLeft);

    EXPECT_EQ(fields_total, 1);
    EXPECT_EQ(ci.src.num_fields, 1);
    EXPECT_EQ(ci.dst.num_fields, 0u);
    EXPECT_EQ(moved, kLeft);

    EXPECT_EQ(in_al.bytes_pending(), kContig * (kSrcFields - kMoveFields) - kLeft);
    EXPECT_EQ(out_al.bytes_pending(), 0);

    EXPECT_EQ(ic.get_offset(), kLeft);
    EXPECT_EQ(oc.get_offset(), 0); // WrapAround
    EXPECT_EQ(oc.remaining(0), kContig * kMoveFields + kLeft);
  }

  heap.free_obj(fb_src);
}

// ─────────────────────────────────────────────────────────────────────────────
//  2-D branch with field blocks on both sides, bytes_left moves two fields
// ─────────────────────────────────────────────────────────────────────────────
TEST(ReadAddressEntry, FieldBlock_2D_MoveTwoFields)
{
  constexpr size_t CONTIG = 32;
  constexpr size_t LINES = 4; // volume per field = 128
  constexpr int FIELDS = 4;
  constexpr size_t BYTES_LEFT = CONTIG * LINES * 2; // allow exactly 2 fields (256)

  AddressList in_al, out_al;
  std::vector<int> ids(FIELDS);
  std::iota(ids.begin(), ids.end(), 0);
  MockHeap h;
  auto *fb_in = FieldBlock::create(h, ids.data(), ids.size());
  auto *fb_out = FieldBlock::create(h, ids.data(), ids.size());
  in_al.attach_field_block(fb_in);
  out_al.attach_field_block(fb_out);

  append_entry_2d(in_al, CONTIG, LINES);
  append_entry_2d(out_al, CONTIG, LINES);

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);
  AffineCopyInfo<3> info{};
  MemcpyTransposeInfo<size_t> tr{};
  size_t min_align = 16, fields_total = 0;

  size_t moved = GPUXferDes::read_address_entry(
      info, min_align, tr, ic, 0, oc, 0, BYTES_LEFT, /*max_fields=*/8, fields_total);

  // We should hit the 2-D branch (one rectangle) and move exactly two fields
  ASSERT_EQ(info.num_rects, 1u);
  const auto &ci = info.subrects[0];
  EXPECT_EQ(ci.extents[0], CONTIG);
  EXPECT_EQ(ci.extents[1], LINES);
  EXPECT_EQ(ci.volume, CONTIG * LINES);
  EXPECT_EQ(fields_total, 2u);
  EXPECT_EQ(ci.src.num_fields, 2u);
  EXPECT_EQ(ci.dst.num_fields, 2u);
  EXPECT_EQ(moved, BYTES_LEFT);

  const size_t BYTES_PER_FIELD = CONTIG * LINES; // 128
  EXPECT_EQ(in_al.bytes_pending(), BYTES_PER_FIELD * (FIELDS - 2));
  EXPECT_EQ(out_al.bytes_pending(), BYTES_PER_FIELD * (FIELDS - 2));

  h.free_obj(fb_in);
  h.free_obj(fb_out);
}

// ─────────────────────────────────────────────────────────────────────────────
//  3-D branch with destination field block only, moves three fields
// ─────────────────────────────────────────────────────────────────────────────
TEST(ReadAddressEntry, DstFieldBlock_3D_MoveThreeFields)
{
  constexpr size_t CONTIG = 16;
  constexpr size_t LINES = 2;
  constexpr size_t PLANES = 3; // volume per field = 96
  constexpr size_t DST_FIELDS = 5;
  constexpr size_t MOVE_FIELDS = 3;
  constexpr size_t BYTES_LEFT =
      CONTIG * LINES * PLANES * MOVE_FIELDS; // 288 bytes → 3 fields

  AddressList in_al, out_al;
  std::vector<int> ids(DST_FIELDS);
  std::iota(ids.begin(), ids.end(), 0);
  MockHeap h;
  auto *fb_dst = FieldBlock::create(h, ids.data(), ids.size());
  out_al.attach_field_block(fb_dst);

  append_entry_3d(in_al, CONTIG, LINES, PLANES * MOVE_FIELDS); // large enough incoming IB
  append_entry_3d(out_al, CONTIG, LINES, PLANES);

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);
  AffineCopyInfo<3> info{};
  MemcpyTransposeInfo<size_t> tr{};
  size_t min_align = 16, fields_total = 0;

  size_t moved = GPUXferDes::read_address_entry(
      info, min_align, tr, ic, 0, oc, 0, BYTES_LEFT, /*max_fields=*/8, fields_total);

  // Expect 3-D branch (one rect) transferring three destination fields
  ASSERT_EQ(info.num_rects, 1u);
  const auto &ci = info.subrects[0];
  EXPECT_EQ(ci.extents[0], CONTIG);
  EXPECT_EQ(ci.extents[1], LINES);
  EXPECT_EQ(ci.extents[2], PLANES);
  EXPECT_EQ(ci.volume, CONTIG * LINES * PLANES);
  EXPECT_EQ(fields_total, 3u);
  EXPECT_EQ(ci.src.num_fields, 0u);
  EXPECT_EQ(ci.dst.num_fields, 3u);
  EXPECT_EQ(moved, BYTES_LEFT);

  const size_t VOL = CONTIG * LINES * PLANES; // 96
  EXPECT_EQ(in_al.bytes_pending(), 0u);
  EXPECT_EQ(out_al.bytes_pending(), VOL * (DST_FIELDS - 3));

  h.free_obj(fb_dst);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mis-aligned base addresses must force min_align down to 1
// ─────────────────────────────────────────────────────────────────────────────
TEST(ReadAddressEntry, MisalignedBase_ByteAlignment)
{
  constexpr size_t CONTIG = 64; // 64-byte 1-D rectangle (16-aligned size)
  // 1-byte-mis-aligned bases
  constexpr uintptr_t IN_BASE = 1;  // 0x…01
  constexpr uintptr_t OUT_BASE = 3; // 0x…03

  AddressList in_al, out_al;
  append_entry_1d(in_al, CONTIG);
  append_entry_1d(out_al, CONTIG);

  AddressListCursor ic, oc;
  ic.set_addrlist(&in_al);
  oc.set_addrlist(&out_al);

  AffineCopyInfo<3> info{};
  MemcpyTransposeInfo<size_t> tr{};
  size_t min_align = 16; // starts optimistic
  size_t fields_total = 0;

  size_t moved =
      GPUXferDes::read_address_entry(info, min_align, tr, ic, IN_BASE, // <-- mis-aligned
                                     oc, OUT_BASE,                     // <-- mis-aligned
                                     CONTIG,                           // bytes_left
                                     /*max_fields*/ 8, fields_total);

  ASSERT_EQ(moved, CONTIG);      // full rectangle copied
  ASSERT_EQ(info.num_rects, 1u); // fast path 1-D
  EXPECT_EQ(min_align, 1u);      // **critical check**
  EXPECT_EQ(fields_total, 1u);
  EXPECT_EQ(in_al.bytes_pending(), 0u);
  EXPECT_EQ(out_al.bytes_pending(), 0u);
}

TEST(ReadAddressEntryTests, DISABLED_Misaligned3D_NoFieldBlock)
{
  constexpr size_t CONTIG = 48; // not 16-byte aligned
  constexpr size_t LINES = 3;
  constexpr size_t PLANES = 2;
  AddressList src_al, dst_al;
  append_entry_3d(src_al, CONTIG, LINES, PLANES);
  append_entry_3d(dst_al, CONTIG, LINES, PLANES);

  AddressListCursor sc, dc;
  sc.set_addrlist(&src_al);
  dc.set_addrlist(&dst_al);

  AffineCopyInfo<3> ci{};
  MemcpyTransposeInfo<size_t> tr{};
  size_t min_align = 16;
  size_t fields_tot = 0;

  // this *must* return 0 because CONTIG is not 16-aligned
  size_t moved = GPUXferDes::read_address_entry(ci, min_align, tr, sc, 0, dc, 0,
                                                CONTIG * LINES * PLANES,
                                                /*max_fields*/ 1, fields_tot);
  EXPECT_EQ(moved, 0u);
}
