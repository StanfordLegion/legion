#include "realm/transfer/transfer.h"
#include "realm/inst_layout.h"
#include <memory>
#include <gtest/gtest.h>

using namespace Realm;

template <int N, typename T>
static InstanceLayout<N, T> *create_layout(const Rect<N, T> &bounds,
                                           const std::vector<int> &dim_order,
                                           size_t bytes_per_element = 8)
{
  InstanceLayout<N, T> *inst_layout = new InstanceLayout<N, T>();
  InstanceLayoutGeneric::FieldLayout field_layout;
  field_layout.list_idx = 0;
  field_layout.rel_offset = 0;
  field_layout.size_in_bytes = bytes_per_element;

  AffineLayoutPiece<N, T> *affine_piece = new AffineLayoutPiece<N, T>();
  affine_piece->bounds = bounds;
  affine_piece->offset = 0;
  affine_piece->strides[dim_order[0]] = bytes_per_element;
  size_t mult = affine_piece->strides[dim_order[0]];
  for(int i = 1; i < N; i++) {
    int d = dim_order[i];
    affine_piece->strides[d] = (bounds.hi[d - 1] - bounds.lo[d - 1] + 1) * mult;
    mult *= (bounds.hi[d - 1] - bounds.lo[d - 1] + 1);
  }

  inst_layout->space = bounds;
  inst_layout->fields[0] = field_layout;
  inst_layout->piece_lists.resize(1);
  inst_layout->piece_lists[0].pieces.push_back(affine_piece);

  return inst_layout;
}

// TODO(apryakhin@): Move to utils
static inline RegionInstance make_inst(int owner = 0, int creator = 0, int mem_idx = 0,
                                       int inst_idx = 0)
{
  return ID::make_instance(owner, creator, mem_idx, inst_idx).convert<RegionInstance>();
}

template <int N, typename T>
static RegionInstanceImpl *
create_inst(const Rect<N, T> &bounds, const std::vector<int> &dim_order,
            size_t bytes_per_element = 8, RegionInstance inst = make_inst())
{
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, dim_order, bytes_per_element);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  NodeSet ns;
  impl->metadata.mark_valid(ns);
  return impl;
}

template <typename PointType>
struct PointTraits;

template <int N, typename T>
struct PointTraits<Realm::Point<N, T>> {
  static constexpr int DIM = N;
  using value_type = T;
};

template <typename PointType>
class IndexSpaceIteratorParamTest : public ::testing::Test {
protected:
  static constexpr int N = PointTraits<PointType>::DIM;
  using T = typename PointTraits<PointType>::value_type;
  constexpr static size_t elem_size = 8;
};

TYPED_TEST_SUITE_P(IndexSpaceIteratorParamTest);

template <int N, typename T>
static std::vector<Rect<N, T>> create_rects(int num_rects, int gap, int start = 0)
{
  std::vector<Rect<N, T>> rect_list;
  int index = start;
  for(int i = 0; i < num_rects; i++) {
    Point<N, T> lo_point = Point<N, T>(index);
    Point<N, T> hi_point = Point<N, T>(index + 1);
    index += gap;
    rect_list.emplace_back(Rect<N, T>(lo_point, hi_point));
  }
  return rect_list;
}

TYPED_TEST_P(IndexSpaceIteratorParamTest, StepSparse)
{
  using T = typename TestFixture::T;
  constexpr int N = TestFixture::N;
  constexpr int gap = 3;
  constexpr size_t num_rects = 3;
  Rect<N, T> domain = Rect<N, T>(TypeParam(0), TypeParam(16));
  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();

  std::vector<Rect<N, T>> rects = create_rects<N, T>(num_rects, gap);
  NodeSet subscribers;
  auto impl = std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers);
  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(rects, /*disjoint=*/true);
  SparsityMapPublicImpl<N, T> *public_impl = impl.get();

  size_t index = 0;
  for(IndexSpaceIterator<N, T> it(domain, domain, public_impl); it.valid; it.step()) {
    ASSERT_TRUE(index < rects.size());
    ASSERT_EQ(it.rect.lo, rects[index].lo);
    ASSERT_EQ(it.rect.hi, rects[index].hi);
    index++;
  }
}

TYPED_TEST_P(IndexSpaceIteratorParamTest, StepDense)
{
  constexpr int N = TestFixture::N;
  using T = typename TestFixture::T;
  Rect<N, T> domain = Rect<N, T>(TypeParam(4), TypeParam(16));
  std::vector<Rect<N, T>> rects{domain};

  size_t index = 0;
  for(IndexSpaceIterator<N, T> it(domain); it.valid; it.step()) {
    ASSERT_TRUE(index < rects.size());
    ASSERT_EQ(it.rect.lo, rects[index].lo);
    ASSERT_EQ(it.rect.hi, rects[index].hi);
    index++;
  }
}

REGISTER_TYPED_TEST_SUITE_P(IndexSpaceIteratorParamTest, StepDense, StepSparse);

template <typename T, int... Ns>
auto GeneratePointTypes(std::integer_sequence<int, Ns...>)
{
  return ::testing::Types<Realm::Point<Ns + 1, T>...>{};
}

template <typename T>
auto GeneratePointTypesForAllDims()
{
  return GeneratePointTypes<T>(std::make_integer_sequence<int, REALM_MAX_DIM>{});
}

#define INSTANTIATE_TEST_TYPES(BASE_TYPE, SUFFIX)                                        \
  using N##SUFFIX = decltype(GeneratePointTypesForAllDims<BASE_TYPE>());                 \
  INSTANTIATE_TYPED_TEST_SUITE_P(SUFFIX##Type, IndexSpaceIteratorParamTest, N##SUFFIX)

INSTANTIATE_TEST_TYPES(int, Int);
// TODO(apryakhin@): Consider enabling if needed
// INSTANTIATE_TEST_TYPES(long long, LongLong);
