#include "realm/transfer/transfer.h"
#include <memory>
#include <vector>
#include <gtest/gtest.h>

using namespace Realm;

template <int DIM>
struct TypeWrapper {
  static constexpr int value = DIM;
};

template <typename TypeWrapper>
class IndexSpaceIteratorTest : public ::testing::Test {
public:
  static constexpr int DIM = TypeWrapper::value;
};

template <int DIM>
struct IndexSpaceIteratorTestCase {
  Rect<DIM> domain;
  Rect<DIM> restrictions;
  std::vector<Rect<DIM>> rects;
  std::vector<Rect<DIM>> expected;
};

template <int DIM>
std::vector<IndexSpaceIteratorTestCase<DIM>> GetTestCases()
{
  if constexpr(DIM == 1) {
    return {
        // Full 1D no rects
        {/*bounds=*/{Rect<1>(0, 10)},
         /*restrictions=*/{Rect<1>(0, 10)},
         /*rects=*/{},
         /*exp_rects=*/{Rect<1>(0, 10)}},

        // Full 1D iteration
        {/*bounds=*/{Rect<1>(0, 10)},
         /*restrictions=*/{Rect<1>(0, 10)},
         /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
         /*exp_rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)}},

        // Full 1D iteration with restrictions
        {/*bounds=*/{Rect<1>(0, 10)},
         /*restrictions=*/{Rect<1>(4, 8)},
         /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
         /*exp_rects=*/{Rect<1>(4, 6), Rect<1>(8, 8)}},

        // 1D Empty bounds
        {/*bounds=*/{Rect<1>(1, 0)},
         /*restrictions=*/{Rect<1>(4, 8)},
         /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
         /*exp_rects=*/{}},

        // 1D empty restrictions
        {/*bounds=*/{Rect<1>(0, 10)},
         /*restrictions=*/{Rect<1>(1, 0)},
         /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
         /*exp_rects=*/{}},
    };
  } else if constexpr(DIM == 2) {
    return {
        // Full 2D domain
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*restrictions=*/Rect<2>({0, 0}, {10, 10}),
            /*rects=*/{Rect<2>({0, 0}, {10, 10})},
            /*expected=*/{Rect<2>({0, 0}, {10, 10})},
        },

        // Restricted 2D domain
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*restrictions=*/Rect<2>({2, 2}, {8, 8}),
            /*rects=*/{Rect<2>({0, 0}, {10, 10})},
            /*expected=*/{Rect<2>({2, 2}, {8, 8})},
        },

        // Sparse 2D domain
        {
            /*domain=*/Rect<2>({0, 0}, {10, 10}),
            /*restrictions=*/Rect<2>({3, 3}, {7, 7}),
            /*rects=*/
            {
                Rect<2>({1, 1}, {2, 2}),
                Rect<2>({3, 3}, {4, 4}),
                Rect<2>({6, 6}, {8, 8}),
            },
            /*expected=*/
            {
                Rect<2>({3, 3}, {4, 4}),
                Rect<2>({6, 6}, {7, 7}),
            },
        },
    };
  } else if constexpr(DIM == 3) {
    return {
        // Full 3D domain
        {
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*restrictions=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*expected=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
        },

        // Restricted 3D domain
        {
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*restrictions=*/Rect<3>({2, 2, 2}, {8, 8, 8}),
            /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
            /*expected=*/{Rect<3>({2, 2, 2}, {8, 8, 8})},
        },

        // Sparse 3D domain
        {
            /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
            /*restrictions=*/Rect<3>({3, 3, 3}, {7, 7, 7}),
            /*rects=*/
            {
                Rect<3>({1, 1, 1}, {2, 2, 2}),
                Rect<3>({3, 3, 3}, {4, 4, 4}),
                Rect<3>({6, 6, 6}, {8, 8, 8}),
            },
            /*expected=*/
            {
                Rect<3>({3, 3, 3}, {4, 4, 4}),
                Rect<3>({6, 6, 6}, {7, 7, 7}),
            },
        },
    };
  }
  return {};
}

TYPED_TEST_SUITE_P(IndexSpaceIteratorTest);

TYPED_TEST_P(IndexSpaceIteratorTest, HandlesVariousCases)
{
  using T = int;
  constexpr int N = TypeParam::value;
  auto test_cases = GetTestCases<N>();
  for(const auto &test_case : test_cases) {
    NodeSet subscribers;
    SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
    SparsityMapPublicImpl<N, T> *local_impl = nullptr;
    SparsityMap<N, T>::ImplLookup::get_impl_ptr =
        [&](const SparsityMap<N, T> &map) -> SparsityMapPublicImpl<N, T> * {
      if(local_impl == nullptr) {
        local_impl = new SparsityMapImpl<N, T>(handle, subscribers);
      }
      return local_impl;
    };

    IndexSpace<N, T> domain = test_case.domain;

    if(!test_case.rects.empty()) {
      SparsityMapImpl<N, T> *impl =
          reinterpret_cast<SparsityMapImpl<N, T> *>(handle.impl());
      impl->set_contributor_count(1);
      impl->contribute_dense_rect_list(test_case.rects, true);
      domain.sparsity = handle;
    }

    size_t index = 0;
    for(IndexSpaceIterator<N, T> it(domain, test_case.restrictions); it.valid;
        it.step()) {
      EXPECT_TRUE(index < test_case.expected.size());
      ASSERT_EQ(it.rect.lo, test_case.expected[index].lo);
      ASSERT_EQ(it.rect.hi, test_case.expected[index].hi);
      index++;
    }

    ASSERT_EQ(index, test_case.expected.size());

    delete local_impl;
  }
}

REGISTER_TYPED_TEST_SUITE_P(IndexSpaceIteratorTest, HandlesVariousCases);

using TestTypes = ::testing::Types<TypeWrapper<1>
#if REALM_MAX_DIM > 1
                                   ,
                                   TypeWrapper<2>
#endif
#if REALM_MAX_DIM > 2
                                   ,
                                   TypeWrapper<3>
#endif
                                   >;

INSTANTIATE_TYPED_TEST_SUITE_P(AllDimensions, IndexSpaceIteratorTest, TestTypes);
