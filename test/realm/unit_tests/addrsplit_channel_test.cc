#include "realm/transfer/addrsplit_channel.h"
#include <tuple>
#include <gtest/gtest.h>
#include <cstring>
#include <vector>

using namespace Realm;

template <int N, typename T>
struct AddressSplitXferDescTestCase {
  std::vector<size_t> src_strides;
  std::vector<size_t> src_extents;
  std::vector<size_t> dst_strides;
  std::vector<size_t> dst_extents;
  int expected_iterations = 1;
  size_t bytes_per_element = 4;
  ;
  std::vector<IndexSpace<N, T>> spaces;
};

template <typename Param>
struct AddressSplitTest : public ::testing::Test {
  static std::vector<AddressSplitXferDescTestCase<std::tuple_element_t<0, Param>::value,
                                                  std::tuple_element_t<1, Param>>>
      _test_cases_;
};

TYPED_TEST_CASE_P(AddressSplitTest);

TYPED_TEST_P(AddressSplitTest, ProgressXD)
{
  constexpr int N = std::tuple_element_t<0, TypeParam>::value;
  using T = std::tuple_element_t<1, TypeParam>;

  for(const auto &test_case : AddressSplitTest<TypeParam>::_test_cases_) {
    auto factory = new AddressSplitXferDesFactory<N, T>(test_case.bytes_per_element,
                                                        test_case.spaces);
    factory->release();
  }
}

REGISTER_TYPED_TEST_CASE_P(AddressSplitTest, ProgressXD);

typedef ::testing::Types<std::tuple<std::integral_constant<int, 1>, int>,
                         std::tuple<std::integral_constant<int, 2>, long long>>
    MyTypes;

INSTANTIATE_TYPED_TEST_SUITE_P(My, AddressSplitTest, MyTypes);

template <>
std::vector<AddressSplitXferDescTestCase<1, int>>
    AddressSplitTest<std::tuple<std::integral_constant<int, 1>, int>>::_test_cases_ = {
        AddressSplitXferDescTestCase<1, int>{.src_strides = {4},
                                             .src_extents = {4},
                                             .dst_strides = {4},
                                             .dst_extents = {4},
                                             .spaces = {IndexSpace<1, int>{}}}};

template <>
std::vector<AddressSplitXferDescTestCase<2, long long>> AddressSplitTest<
    std::tuple<std::integral_constant<int, 2>, long long>>::_test_cases_ = {
    AddressSplitXferDescTestCase<2, long long>{.src_strides = {4, 16},
                                               .src_extents = {4, 4},
                                               .dst_strides = {4, 16},
                                               .dst_extents = {4, 4},
                                               .spaces = {IndexSpace<2, long long>{}}}};
