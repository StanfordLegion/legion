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

#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include "test_common.h"
#include <gtest/gtest.h>
#include <cstdlib>

using namespace Realm;

// Base class for parameterized test cases
struct BaseIDIndexedIteratorTestCaseData {
  virtual ~BaseIDIndexedIteratorTestCaseData() = default;
  virtual int get_dim() const = 0;
};

template <int N>
struct IDIndexedIteratorTestCaseData {
  // The iteration space
  Rect<N, int> domain;
  // The expected subrects from get_addresses (for uniform fields it's just the full
  // domain)
  std::vector<Rect<N, int>> expected;
  // Dimension traversal order
  std::vector<int> dim_order;
  // The list of field IDs (all fields have the same uniform size)
  std::vector<FieldID> fields;
  // Uniform field size in bytes
  size_t field_size;
};

template <int N>
struct WrappedIDIndexedIteratorTestCaseData : public BaseIDIndexedIteratorTestCaseData {
  IDIndexedIteratorTestCaseData<N> data;
  explicit WrappedIDIndexedIteratorTestCaseData(IDIndexedIteratorTestCaseData<N> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class IDIndexedFieldsIteratorGetAddressesTest
  : public ::testing::TestWithParam<BaseIDIndexedIteratorTestCaseData *> {
protected:
  void TearDown() override { delete GetParam(); }
};

struct MockHeap : public ReplicatedHeap {
  void *alloc_obj(std::size_t bytes, std::size_t align = 16) override
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

  void free_obj(void *ptr) override
  {
#ifdef REALM_ON_WINDOWS
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
};

template <int N>
void run_uniform_test_case(const IDIndexedIteratorTestCaseData<N> &tc)
{
  using T = int;

  // Create an instance with one entry per field, all uniformly sized
  std::vector<size_t> field_sizes(tc.fields.size(), tc.field_size);
  RegionInstanceImpl *inst_impl =
      create_inst<N, T>(nullptr, tc.domain, tc.fields, field_sizes);

  MockHeap mock_heap;

  // Build the IDIndexedFieldsIterator
  auto it = std::make_unique<IDIndexedFieldsIterator<N, T>>(
      tc.dim_order.data(), tc.fields, tc.field_size, inst_impl, tc.domain, &mock_heap);

  const InstanceLayoutPieceBase *nonaffine = nullptr;
  AddressList addrlist;
  AddressListCursor cursor;

  // Invoke get_addresses
  bool ok = it->get_addresses(addrlist, nonaffine);
  ASSERT_TRUE(ok) << "get_addresses() failed";
  ASSERT_TRUE(it->done()) << "Iterator should be done after get_addresses";

  // Check total bytes pending
  size_t total_volume = 0;
  for(auto &r : tc.expected)
    total_volume += r.volume();
  size_t expected_bytes = total_volume * tc.field_size * tc.fields.size();
  ASSERT_EQ(addrlist.bytes_pending(), expected_bytes) << "bytes_pending mismatch";

  // If 1D, walk through with the cursor and drain the bytes
  cursor.set_addrlist(&addrlist);
  if(expected_bytes > 0 && cursor.get_dim() == 1) {
    int dim = cursor.get_dim() - 1;
    for(size_t f = 0; f < tc.fields.size(); f++) {
      for(auto &r : tc.expected) {
        size_t rem = cursor.remaining(dim);
        ASSERT_EQ(rem, r.volume() * tc.field_size)
            << "Unexpected remaining bytes for field " << f;
        cursor.advance(dim, rem);
      }
    }
    ASSERT_EQ(addrlist.bytes_pending(), 0u) << "All bytes should have been consumed";
  }
}

TEST_P(IDIndexedFieldsIteratorGetAddressesTest, Base)
{
  BaseIDIndexedIteratorTestCaseData const *base = GetParam();
  dispatch_for_dimension(
      base->get_dim(),
      [&](auto Dim) {
        constexpr int N = Dim;
        auto const &tc =
            static_cast<WrappedIDIndexedIteratorTestCaseData<N> const *>(base)->data;
        run_uniform_test_case(tc);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(UniformFieldsCases, IDIndexedFieldsIteratorGetAddressesTest,
                         ::testing::Values(
                             // 1D: empty domain
                             new WrappedIDIndexedIteratorTestCaseData<1>({
                                 /* domain */ Rect<1, int>::make_empty(),
                                 /* expected */ {},
                                 /* dim_order */ {0},
                                 /* fields */ {0},
                                 /* field_size */ sizeof(int),
                             }),
                             // 1D: single field, full domain
                             new WrappedIDIndexedIteratorTestCaseData<1>({
                                 /* domain */ Rect<1, int>(0, 14),
                                 /* expected */ {Rect<1, int>(0, 14)},
                                 /* dim_order */ {0},
                                 /* fields */ {0},
                                 /* field_size */ sizeof(int),
                             }),
                             // 1D: two uniform fields, full domain
                             new WrappedIDIndexedIteratorTestCaseData<1>({
                                 /* domain */ Rect<1, int>(0, 14),
                                 /* expected */ {Rect<1, int>(0, 14)},
                                 /* dim_order */ {0},
                                 /* fields */ {0, 1},
                                 /* field_size */ sizeof(int),
                             }),
                             // 2D: single field, full rectangular domain
                             new WrappedIDIndexedIteratorTestCaseData<2>({
                                 /* domain */ Rect<2, int>({0, 0}, {10, 10}),
                                 /* expected */ {Rect<2, int>({0, 0}, {10, 10})},
                                 /* dim_order */ {0, 1},
                                 /* fields */ {0},
                                 /* field_size */ sizeof(int),
                             }),
                             // 2D: single field, reverse dims
                             new WrappedIDIndexedIteratorTestCaseData<2>({
                                 /* domain */ Rect<2, int>({0, 0}, {10, 10}),
                                 /* expected */ {Rect<2, int>({0, 0}, {10, 10})},
                                 /* dim_order */ {1, 0},
                                 /* fields */ {0},
                                 /* field_size */ sizeof(int),
                             }),
                             // 3D: single field, small cube
                             new WrappedIDIndexedIteratorTestCaseData<3>({
                                 /* domain */ Rect<3, int>({0, 0, 0}, {1, 1, 1}),
                                 /* expected */ {Rect<3, int>({0, 0, 0}, {1, 1, 1})},
                                 /* dim_order */ {0, 1, 2},
                                 /* fields */ {0},
                                 /* field_size */ sizeof(int),
                             })));
