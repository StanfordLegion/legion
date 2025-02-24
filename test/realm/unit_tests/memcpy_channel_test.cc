#include "realm/bgwork.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/memcpy_channel.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace Realm;

static inline Memory make_mem(int idx, int node_id)
{
  return ID::make_memory(idx, node_id).convert<Memory>();
}

struct SupportsPathTestCase {
  int src_mem_id;
  int src_node_id;
  int dst_mem_id;
  int dst_node_id;
  bool is_remote_shared;
  bool is_local;
  uint64_t expected_cost;
  ReductionOpID redop_id = 0;
  bool src_serdez_id = 0;
  bool dst_serdez_id = 0;
  std::vector<size_t> src_frags;
  std::vector<size_t> dst_frags;
};

class SupportsPathTest : public ::testing::TestWithParam<SupportsPathTestCase> {
protected:
  MemoryImpl *create_memory(int id, int node_id, std::vector<std::byte> buffer)
  {
    return new LocalCPUMemory(make_mem(id, node_id), buffer.size(), 0, Memory::SYSTEM_MEM,
                              buffer.data());
  }

  Node node;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
};

TEST_P(SupportsPathTest, CheckSupportsPath)
{
  SupportsPathTestCase test_case = GetParam();
  constexpr size_t bytes = 16;

  std::vector<std::byte> src_buffer(bytes);
  MemoryImpl *src_mem =
      create_memory(test_case.src_mem_id, test_case.src_node_id, src_buffer);
  std::vector<std::byte> dst_buffer(bytes);
  MemoryImpl *dst_mem =
      create_memory(test_case.dst_mem_id, test_case.dst_node_id, dst_buffer);

  node.memories.push_back(src_mem);

  if(test_case.is_remote_shared) {
    remote_shared_memory_mappings.insert({dst_mem->me.id, SharedMemoryInfo()});
  }

  if(test_case.is_local) {
    node.memories.push_back(dst_mem);
  }

  BackgroundWorkManager *bgwork = new BackgroundWorkManager();
  std::unique_ptr<Channel> channel(
      new MemcpyChannel(bgwork, &node, remote_shared_memory_mappings));

  uint64_t cost = channel->supports_path(
      ChannelCopyInfo(src_mem->me, dst_mem->me), test_case.src_serdez_id,
      test_case.dst_serdez_id, test_case.redop_id, bytes,
      (test_case.src_frags.empty() ? nullptr : &test_case.src_frags),
      (test_case.dst_frags.empty() ? nullptr : &test_case.dst_frags));

  ASSERT_EQ(cost, test_case.expected_cost);

  if(!test_case.is_local) {
    delete dst_mem;
  }

  channel->shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    SupportsPathTests, SupportsPathTest,
    ::testing::Values(
        // Case 1: Local memory copy (expected cost: 100)
        SupportsPathTestCase{0, 0, 1, 0, false, true, 100},

        // Case 2: Remote shared memory copy (expected cost: 100)
        SupportsPathTestCase{0, 0, 1, 1, true, false, 100},

        // Case 3: Unreachable memory (expected cost: 0)
        SupportsPathTestCase{0, 0, 2, 2, false, false, 0},

        // Case 4: Another local memory (expected cost: 100)
        SupportsPathTestCase{0, 0, 3, 0, false, true, 100},

        // Case 5: Another remote shared memory (expected cost: 100)
        SupportsPathTestCase{0, 0, 4, 1, true, false, 100},

        // Case 6: Completely disconnected memory (expected cost: 0)
        SupportsPathTestCase{0, 0, 5, 2, false, false, 0},

        // Case 7: Local memory with reduction
        SupportsPathTestCase{0, 0, 1, 0, false, true, 0,
                             /*redop_id=*/1},

        // Case 8: Local memory with both src serdez
        SupportsPathTestCase{0, 0, 1, 0, false, true, 100,
                             /*redop_id=*/0, /*src_serdez_id=*/1,
                             /*dst_serdez_id=*/0},

        // Case 9: Local memory with dst serdez
        SupportsPathTestCase{0, 0, 1, 0, false, true, 100,
                             /*redop_id=*/0, /*src_serdez_id=*/0,
                             /*dst_serdez_id=*/1},

        // Case 10: Local memory with both src/dst serdez
        SupportsPathTestCase{0, 0, 1, 0, false, true, 0,
                             /*redop_id=*/0, /*src_serdez_id=*/1,
                             /*dst_serdez_id=*/1},

        // Case 11: Local memory with src frags
        SupportsPathTestCase{0, 0, 1, 0, false, true, 1000,
                             /*redop_id=*/0, /*src_serdez_id=*/0,
                             /*dst_serdez_id=*/0, /*src_frags=*/{10}},

        // Case 12: Local memory with src and dst frags
        SupportsPathTestCase{0, 0, 1, 0, false, true, 2000,
                             /*redop_id=*/0, /*src_serdez_id=*/0,
                             /*dst_serdez_id=*/0, /*src_frags=*/{10}, /*dst_frags=*/{20}}

        ));

template <int N, typename T>
static InstanceLayout<N, T> *create_layout(Rect<N, T> bounds,
                                           const std::vector<FieldID> &field_ids,
                                           const std::vector<size_t> &field_sizes)
{
  InstanceLayout<N, T> *inst_layout = new InstanceLayout<N, T>();

  inst_layout->piece_lists.resize(field_ids.size());

  for(int i = 0; i < field_ids.size(); i++) {
    InstanceLayoutGeneric::FieldLayout field_layout;
    field_layout.list_idx = i;
    field_layout.rel_offset = 0;
    field_layout.size_in_bytes = field_sizes[i];

    AffineLayoutPiece<N, T> *affine_piece = new AffineLayoutPiece<N, T>();
    affine_piece->bounds = bounds;
    affine_piece->offset = 0;
    affine_piece->strides[0] = field_sizes[i];
    size_t mult = affine_piece->strides[0];
    for(int i = 1; i < N; i++) {
      affine_piece->strides[i] = (bounds.hi[i - 1] - bounds.lo[i - 1] + 1) * mult;
      mult *= (bounds.hi[i - 1] - bounds.lo[i - 1] + 1);
    }

    inst_layout->space = bounds;
    inst_layout->fields[field_ids[i]] = field_layout;
    inst_layout->piece_lists[i].pieces.push_back(affine_piece);
  }

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
create_inst(Rect<N, T> bounds, const std::vector<FieldID> &field_ids,
            const std::vector<size_t> &field_sizes, RegionInstance inst = make_inst())
{
  InstanceLayout<N, T> *inst_layout = create_layout(bounds, field_ids, field_sizes);
  RegionInstanceImpl *impl = new RegionInstanceImpl(inst, inst.get_location());
  impl->metadata.layout = inst_layout;
  impl->metadata.inst_offset = 0;
  return impl;
}

template <int N>
struct MemcpyXferTestCaseData {
  Rect<N> domain;
  std::vector<Rect<N>> rects;
  std::vector<int> dim_order;
  std::vector<FieldID> field_ids;
  std::vector<size_t> field_offsets;
  std::vector<size_t> field_sizes;
  std::vector<int> src_buffer;
  std::vector<int> exp_buffer;
};

struct BaseMemcpyXferTestCaseData {
  virtual ~BaseMemcpyXferTestCaseData() = default;
  virtual int get_dim() const = 0;
};

template <int N>
struct WrapedXferDesTestData : public BaseMemcpyXferTestCaseData {
  MemcpyXferTestCaseData<N> data;
  explicit WrapedXferDesTestData(MemcpyXferTestCaseData<N> d)
    : data(std::move(d))
  {}
  int get_dim() const override { return N; }
};

class MemcpyChannelTest : public ::testing::TestWithParam<BaseMemcpyXferTestCaseData *> {
protected:
  void TearDown() override { delete GetParam(); }
};

template <int N>
void run_test_case(const MemcpyXferTestCaseData<N> &test_case)
{
  using T = int;

  XferDesID guid = 0;
  int priority = 0;
  NodeID owner = 0;
  Node node_data;
  std::unordered_map<realm_id_t, SharedMemoryInfo> remote_shared_memory_mappings;
  XferDesRedopInfo redop_info;
  std::unique_ptr<BackgroundWorkManager> bgwork =
      std::make_unique<BackgroundWorkManager>();
  std::unique_ptr<MemcpyChannel> channel = std::make_unique<MemcpyChannel>(
      bgwork.get(), &node_data, remote_shared_memory_mappings);

  NodeSet subscribers;

  SparsityMap<N, T> handle = (ID::make_sparsity(0, 0, 0)).convert<SparsityMap<N, T>>();
  std::unique_ptr<SparsityMapImpl<N, T>> impl =
      std::make_unique<SparsityMapImpl<N, T>>(handle, subscribers);
  SparsityMapPublicImpl<N, T> *local_impl =
      reinterpret_cast<SparsityMapPublicImpl<N, T> *>(impl.get());

  impl->set_contributor_count(1);
  impl->contribute_dense_rect_list(test_case.rects, true);
  IndexSpace<N, T> domain = test_case.domain;

  TransferIteratorIndexSpace<N, T> *src_it = new TransferIteratorIndexSpace<N, T>(
      domain,
      create_inst<N, T>(test_case.domain, test_case.field_ids, test_case.field_sizes),
      test_case.dim_order.data(), test_case.field_ids, test_case.field_offsets,
      test_case.field_sizes, 0, local_impl);

  std::vector<XferDesPortInfo> inputs_info;
  std::vector<XferDesPortInfo> outputs_info;

  std::unique_ptr<MemcpyXferDes> xfer_desc(dynamic_cast<MemcpyXferDes *>(
      channel->create_xfer_des(0, owner, guid, inputs_info, outputs_info, priority,
                               redop_info, nullptr, 0, 0)));

  xfer_desc->input_ports.resize(1);
  XferDes::XferPort &input_port = xfer_desc->input_ports[0];

  std::vector<int> src_buffer = test_case.src_buffer;
  std::unique_ptr<LocalCPUMemory> input_mem =
      std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, src_buffer.size() * sizeof(int),
                                       0, Memory::SYSTEM_MEM, src_buffer.data());
  input_port.mem = input_mem.get();
  input_port.peer_port_idx = 0;
  input_port.iter = src_it;
  input_port.addrcursor.set_addrlist(&input_port.addrlist);

  TransferIteratorIndexSpace<N, T> *dst_it = new TransferIteratorIndexSpace<N, T>(
      domain,
      create_inst<N, T>(test_case.domain, test_case.field_ids, test_case.field_sizes),
      test_case.dim_order.data(), test_case.field_ids, test_case.field_offsets,
      test_case.field_sizes, 0, local_impl);

  xfer_desc->output_ports.resize(1);
  XferDes::XferPort &output_port = xfer_desc->output_ports[0];
  size_t total_dst_size = src_buffer.size();
  std::vector<int> dst_buffer(total_dst_size, 77);
  std::unique_ptr<LocalCPUMemory> output_mem =
      std::make_unique<LocalCPUMemory>(Memory::NO_MEMORY, total_dst_size * sizeof(int), 0,
                                       Memory::SYSTEM_MEM, dst_buffer.data());
  output_port.mem = output_mem.get();
  output_port.peer_port_idx = 0;
  output_port.iter = dst_it;

  output_port.addrcursor.set_addrlist(&output_port.addrlist);

  while(xfer_desc->progress_xd(channel.get(), TimeLimit::relative(10000000))) {
  }

  for(size_t i = 0; i < total_dst_size; i++) {
    EXPECT_EQ(dst_buffer[i], test_case.exp_buffer[i]);
  }

  channel->shutdown();
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

TEST_P(MemcpyChannelTest, Base)
{
  const BaseMemcpyXferTestCaseData *base_test_case = GetParam();

  dispatch_for_dimension(
      base_test_case->get_dim(),
      [&](auto Dim) {
        constexpr int N = Dim;
        auto &test_case =
            static_cast<const WrapedXferDesTestData<N> *>(base_test_case)->data;
        run_test_case(test_case);
      },
      std::make_index_sequence<REALM_MAX_DIM>{});
}

INSTANTIATE_TEST_SUITE_P(MemcpyChannelCases, MemcpyChannelTest,
                         ::testing::Values(
                             // Case 1: All points are mergeable
                             new WrapedXferDesTestData<1>({
                                 /*domain=*/{Rect<1>(1, 0)},
                                 /*rects=*/{Rect<1>(1, 0)},
                                 /*dim_order=*/{0},
                                 /*field_ids=*/{0, 1},
                                 /*field_offsets=*/{0, 0},
                                 /*field_sizes=*/{sizeof(int), sizeof(long long)},
                                 {},
                                 {},
                             }),

                             new WrapedXferDesTestData<1>(
                                 // Empty 1D domain
                                 {/*domain=*/{Rect<1>(1, 0)},
                                  /*rects=*/{},
                                  /*dim_order=*/{0},
                                  /*field_ids=*/{0},
                                  /*field_offsets=*/{0},
                                  /*field_sizes=*/{sizeof(int)},
                                  {}}),

                             new WrapedXferDesTestData<1>(
                                 // Sparse 1D rects
                                 {
                                     /*domain=*/{Rect<1>(0, 8)},
                                     /*rects=*/{Rect<1>(0, 1), Rect<1>(5, 6)},
                                     /*dim_order=*/{0},
                                     /*field_ids=*/{0},
                                     /*field_offsets=*/{0},
                                     /*field_sizes=*/{sizeof(int)},
                                     {0, 1, 2, 3, 5, 6, 7},
                                     {0, 1, 77, 77, 77, 6, 7},
                                 }),

                             new WrapedXferDesTestData<1>(
                                 // Dense 1D rects
                                 {
                                     /*domain=*/{Rect<1>(0, 3)},
                                     /*rects=*/{Rect<1>(0, 3)},
                                     /*dim_order=*/{0},
                                     /*field_ids=*/{0},
                                     /*field_offsets=*/{0},
                                     /*field_sizes=*/{sizeof(int)},
                                     {0, 1, 2, 3},
                                     {0, 1, 2, 3},
                                 }),

                             new WrapedXferDesTestData<2>(
                                 // Full 2D dense reverse dims
                                 {
                                     /*domain=*/Rect<2>({0, 0}, {1, 1}),
                                     /*rects*/ {Rect<2>({0, 0}, {1, 1})},
                                     /*dim_order=*/{1, 0},
                                     /*field_ids=*/{0},
                                     /*field_offsets=*/{0},
                                     /*field_sizes=*/{sizeof(int)},
                                     {0, 1, 2, 3},
                                     {0, 1, 2, 3},
                                 }),

                             new WrapedXferDesTestData<2>(
                                 // Full 2D sparse
                                 {
                                     /*domain=*/Rect<2>({0, 0}, {3, 1}),
                                     /*rects*/
                                     {Rect<2>({0, 0}, {0, 0}), Rect<2>({3, 0}, {3, 0}),
                                      Rect<2>({0, 1}, {0, 1}), Rect<2>({3, 1}, {3, 1})},
                                     /*dim_order=*/{0, 1},
                                     /*field_ids=*/{0},
                                     /*field_offsets=*/{0},
                                     /*field_sizes=*/{sizeof(int)},
                                     {0, 1, 2, 3, 4, 5, 6, 7},
                                     {0, 77, 77, 3, 4, 77, 77, 7},
                                 }),

                             new WrapedXferDesTestData<2>(
                                 // Full 2D dense
                                 {
                                     /*domain=*/Rect<2>({0, 0}, {1, 1}),
                                     /*rects*/ {Rect<2>({0, 0}, {1, 1})},
                                     /*dim_order=*/{0, 1},
                                     /*field_ids=*/{0},
                                     /*field_offsets=*/{0},
                                     /*field_sizes=*/{sizeof(int)},
                                     {0, 1, 2, 3},
                                     {0, 1, 2, 3},
                                 }),

                             new WrapedXferDesTestData<3>(
                                 // Full 3D domain
                                 {
                                     /*domain=*/Rect<3>({0, 0, 0}, {1, 1, 1}),
                                     /*rects=*/{Rect<3>({0, 0, 0}, {1, 1, 1})},
                                     /*dim_order=*/{0, 1, 2},
                                     /*field_ids=*/{0},
                                     /*field_offsets=*/{0},
                                     /*field_sizes=*/{sizeof(int)},
                                     {0, 1, 2, 3, 4, 5, 6, 7},
                                     {0, 1, 2, 3, 4, 5, 6, 7},
                                 }),

                             new WrapedXferDesTestData<3>(
                                 // Ful 3d domain with reverse dims
                                 {
                                     /*domain=*/Rect<3>({0, 0, 0}, {1, 1, 1}),
                                     /*rects=*/{Rect<3>({0, 0, 0}, {1, 1, 1})},
                                     /*dim_order=*/{2, 1, 0},
                                     /*field_ids=*/{0},
                                     /*field_offsets=*/{0},
                                     /*field_sizes=*/{sizeof(int)},
                                     {0, 1, 2, 3, 4, 5, 6, 7},
                                     {0, 1, 2, 3, 4, 5, 6, 7},
                                 })));
