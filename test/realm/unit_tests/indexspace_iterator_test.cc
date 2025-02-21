#include "realm/indexspace.h"
#include "realm/deppart/sparsity_impl.h"
#include <memory>
#include <vector>
#include <gtest/gtest.h>

using namespace Realm;

template <int N>
struct ItTestCaseData {
  Rect<N> domain;
  Rect<N> restrictions;
  std::vector<Rect<N>> rects;
  std::vector<Rect<N>> expected;
};

struct BaseItTestCaseData {
  virtual ~BaseItTestCaseData() = default;
  virtual int get_dim() const = 0;
};

template <int N>
struct WrappedItTestCaseData : public BaseItTestCaseData {
  ItTestCaseData<N> data;
  explicit WrappedItTestCaseData(ItTestCaseData<N> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class IndexSpaceItTest : public ::testing::TestWithParam<BaseItTestCaseData *> {
protected:
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_test_case(const ItTestCaseData<N> &test_case)
{
  using T = int;
  NodeSet subscribers;

  std::unique_ptr<SparsityMapImpl<N, T>> impl = std::make_unique<SparsityMapImpl<N, T>>(
      (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>(), subscribers);
  SparsityMapPublicImpl<N, T> *local_impl = nullptr;

  if(!test_case.rects.empty()) {
    impl->set_contributor_count(1);
    impl->contribute_dense_rect_list(test_case.rects, true);
    local_impl = reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());
  }

  size_t index = 0;
  for(IndexSpaceIterator<N, T> it(test_case.domain, test_case.restrictions, local_impl);
      it.valid; it.step()) {
    EXPECT_TRUE(index < test_case.expected.size());
    ASSERT_EQ(it.rect.lo, test_case.expected[index].lo);
    ASSERT_EQ(it.rect.hi, test_case.expected[index].hi);
    index++;
  }

  ASSERT_EQ(index, test_case.expected.size());
}

template <typename Func, size_t... Is>
void dispatch_for_dimension(int dim, Func &&func, std::index_sequence<Is...>)
{
  (
      [&] {
        if(dim == static_cast<int>(Is + 1)) {
          func(std::integral_constant<int, Is + 1>{});
        }
      }(),
      ...);
}

TEST_P(IndexSpaceItTest, Base)
{
  const BaseItTestCaseData *base_test_case = GetParam();

  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        constexpr int N = Dim;
        auto &test_case =
            static_cast<const WrappedItTestCaseData<N> *>(base_test_case)->data;
        run_test_case(test_case);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(
    IndexSpaceItCases, IndexSpaceItTest,
    ::testing::Values(new WrappedItTestCaseData<1>(
                          // Full 1D iteration
                          {/*bounds=*/{Rect<1>(0, 10)},
                           /*restrictions=*/{Rect<1>(0, 10)},
                           /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
                           /*exp_rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)}}),

                      new WrappedItTestCaseData<1>(
                          // Full 1D no rects
                          {/*bounds=*/{Rect<1>(0, 10)},
                           /*restrictions=*/{Rect<1>(0, 10)},
                           /*rects=*/{},
                           /*exp_rects=*/{Rect<1>(0, 10)}}),

                      new WrappedItTestCaseData<1>(
                          // Full 1D iteration with restrictions
                          {/*bounds=*/{Rect<1>(0, 10)},
                           /*restrictions=*/{Rect<1>(4, 8)},
                           /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
                           /*exp_rects=*/{Rect<1>(4, 6), Rect<1>(8, 8)}}),

                      new WrappedItTestCaseData<1>(
                          // 1D Empty bounds
                          {/*bounds=*/{Rect<1>(1, 0)},
                           /*restrictions=*/{Rect<1>(4, 8)},
                           /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
                           /*exp_rects=*/{}}),

                      new WrappedItTestCaseData<1>(
                          // 1D empty restrictions
                          {/*bounds=*/{Rect<1>(0, 10)},
                           /*restrictions=*/{Rect<1>(1, 0)},
                           /*rects=*/{Rect<1>(0, 2), Rect<1>(4, 6), Rect<1>(8, 10)},
                           /*exp_rects=*/{}}),

                      new WrappedItTestCaseData<2>(
                          // Full 2D domain
                          {
                              /*domain=*/Rect<2>({0, 0}, {10, 10}),
                              /*restrictions=*/Rect<2>({0, 0}, {10, 10}),
                              /*rects=*/{Rect<2>({0, 0}, {10, 10})},
                              /*expected=*/{Rect<2>({0, 0}, {10, 10})},
                          }),

                      new WrappedItTestCaseData<2>(
                          // Restricted 2D domain
                          {
                              /*domain=*/Rect<2>({0, 0}, {10, 10}),
                              /*restrictions=*/Rect<2>({2, 2}, {8, 8}),
                              /*rects=*/{Rect<2>({0, 0}, {10, 10})},
                              /*expected=*/{Rect<2>({2, 2}, {8, 8})},
                          }),

                      new WrappedItTestCaseData<2>(
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
                          }),

                      new WrappedItTestCaseData<3>(
                          // Full 3D domain
                          {
                              /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
                              /*restrictions=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
                              /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
                              /*expected=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
                          }),

                      new WrappedItTestCaseData<3>(
                          // Restricted 3D domain
                          {
                              /*domain=*/Rect<3>({0, 0, 0}, {10, 10, 10}),
                              /*restrictions=*/Rect<3>({2, 2, 2}, {8, 8, 8}),
                              /*rects=*/{Rect<3>({0, 0, 0}, {10, 10, 10})},
                              /*expected=*/{Rect<3>({2, 2, 2}, {8, 8, 8})},
                          }),

                      new WrappedItTestCaseData<3>(
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
                          })));
