#include "realm/hardware_topology.h"

#include <gtest/gtest.h>

using namespace Realm;

class SyntheticTopologyTest
  : public ::testing::TestWithParam<std::tuple<std::vector<HardwareTopology::Proc>,
                                               std::vector<HardwareTopology::MemoryInfo>,
                                               size_t, unsigned, unsigned, bool>> {
protected:
  std::vector<HardwareTopology::Proc> param_logical_cores;
  std::vector<HardwareTopology::MemoryInfo> param_memories;
  size_t param_host_memory;
  unsigned param_num_physical_cores;
  unsigned param_num_numa_domains;
  bool param_hyperthread;
  HardwareTopology topology;

  void SetUp() override
  {
    const auto &params = GetParam();
    param_logical_cores = std::get<0>(params);
    param_memories = std::get<1>(params);
    param_host_memory = std::get<2>(params);
    param_num_physical_cores = std::get<3>(params);
    param_num_numa_domains = std::get<4>(params);
    param_hyperthread = std::get<5>(params);
    topology = HardwareTopology(param_logical_cores, param_memories, param_host_memory);
  }

  const HardwareTopology::Proc *get_proc_by_id(HardwareTopology::ProcID proc_id)
  {
    for(const HardwareTopology::Proc &proc : param_logical_cores) {
      if(proc.id == proc_id) {
        return &proc;
      }
    }
    return nullptr;
  }
};

TEST_P(SyntheticTopologyTest, SystemMemory)
{
  EXPECT_EQ(topology.system_memory(), param_host_memory);
}

TEST_P(SyntheticTopologyTest, NumLogicalCores)
{
  EXPECT_EQ(topology.num_logical_cores(), param_logical_cores.size());
}

TEST_P(SyntheticTopologyTest, NumPhysicalCores)
{
  EXPECT_EQ(topology.num_physical_cores(), param_num_physical_cores);
}

TEST_P(SyntheticTopologyTest, NumNumaDomains)
{
  EXPECT_EQ(topology.num_numa_domains(), param_num_numa_domains);
}

TEST_P(SyntheticTopologyTest, IterateLogicalCores)
{
  std::set<HardwareTopology::ProcID> logical_cores;
  topology.get_logical_cores(logical_cores);

  EXPECT_EQ(logical_cores.size(), param_logical_cores.size());
  for(const HardwareTopology::Proc &proc : param_logical_cores) {
    EXPECT_TRUE(logical_cores.find(proc.id) != logical_cores.end());
  }
}

TEST_P(SyntheticTopologyTest, HasProc)
{
  for(const HardwareTopology::Proc &proc : param_logical_cores) {
    EXPECT_TRUE(topology.has_processor(proc.id));
  }
}

TEST_P(SyntheticTopologyTest, HasNumaDomainWithProcessors)
{
  for(unsigned i = 0; i < param_num_numa_domains; i++) {
    EXPECT_TRUE(topology.numa_domain_has_processors(i));
  }
}

TEST_P(SyntheticTopologyTest, GetProcsByDomain)
{
  for(unsigned i = 0; i < param_num_numa_domains; i++) {
    const std::set<HardwareTopology::ProcID> &proc_ids =
        topology.get_processors_by_domain(i);
    for(const HardwareTopology::ProcID proc_id : proc_ids) {
      const HardwareTopology::Proc *proc = get_proc_by_id(proc_id);
      EXPECT_EQ(proc->id, proc_id);
      EXPECT_EQ(proc->domain, i);
    }
  }
}

TEST_P(SyntheticTopologyTest, GetProcsShareALU)
{
  for(HardwareTopology::Proc &proc : param_logical_cores) {
    const std::set<HardwareTopology::ProcID> &proc_ids =
        topology.get_processors_share_alu(proc.id);
    EXPECT_EQ(proc_ids, proc.shares_alu);
  }
}

TEST_P(SyntheticTopologyTest, GetProcsShareFPU)
{
  for(HardwareTopology::Proc &proc : param_logical_cores) {
    const std::set<HardwareTopology::ProcID> &proc_ids =
        topology.get_processors_share_fpu(proc.id);
    EXPECT_EQ(proc_ids, proc.shares_fpu);
  }
}

TEST_P(SyntheticTopologyTest, GetProcsShareLDST)
{
  for(HardwareTopology::Proc &proc : param_logical_cores) {
    const std::set<HardwareTopology::ProcID> &proc_ids =
        topology.get_processors_share_ldst(proc.id);
    EXPECT_EQ(proc_ids, proc.shares_ldst);
  }
}

TEST_P(SyntheticTopologyTest, GetKernelProcIds)
{
  for(HardwareTopology::Proc &proc : param_logical_cores) {
    const std::set<HardwareTopology::ProcID> &proc_ids =
        topology.get_kernel_processor_ids(proc.id);
    EXPECT_EQ(proc_ids, proc.kernel_proc_ids);
  }
}

TEST_P(SyntheticTopologyTest, DistributeProcsAcrossDomains)
{
  std::vector<HardwareTopology::ProcID> procs =
      topology.distribute_processors_across_domains();
  int prev_domain = -1;
  for(HardwareTopology::ProcID proc_id : procs) {
    const HardwareTopology::Proc &proc = param_logical_cores[proc_id];
    if(param_num_numa_domains > 1) {
      EXPECT_TRUE(proc.domain != prev_domain);
    } else {
      if(prev_domain != -1) {
        EXPECT_TRUE(proc.domain == prev_domain);
      }
    }
    prev_domain = proc.domain;
  }
}
// the parameters are listed as:
// 1. a vector of all procs
// 2. a vector of all memories
// 3. host memory size
// 4. number of physical cores // used for verification
// 5. number of numa domains // used for verification
// 6. wether hyperthread is enabled
INSTANTIATE_TEST_SUITE_P(
    SyntheticTopologyTestInstances, SyntheticTopologyTest,
    ::testing::Values(
        // 1 numa, 2 ht
        std::make_tuple(std::vector<HardwareTopology::Proc>{{0, 0, {0}, {4}, {4}, {4}},
                                                            {1, 0, {1}, {5}, {5}, {5}},
                                                            {2, 0, {2}, {6}, {6}, {6}},
                                                            {3, 0, {3}, {7}, {7}, {7}},
                                                            {4, 0, {4}, {0}, {0}, {0}},
                                                            {5, 0, {5}, {1}, {1}, {1}},
                                                            {6, 0, {6}, {2}, {2}, {2}},
                                                            {7, 0, {7}, {3}, {3}, {3}}},
                        std::vector<HardwareTopology::MemoryInfo>{{1024, 0}}, 1024, 4, 1,
                        true),
        // 1 numa, 4 ht
        std::make_tuple(
            std::vector<HardwareTopology::Proc>{
                {0, 0, {0}, {2, 4, 6}, {2, 4, 6}, {2, 4, 6}},
                {1, 0, {1}, {3, 5, 7}, {3, 5, 7}, {3, 5, 7}},
                {2, 0, {2}, {0, 4, 6}, {0, 4, 6}, {0, 4, 6}},
                {3, 0, {3}, {1, 5, 7}, {1, 5, 7}, {1, 5, 7}},
                {4, 0, {4}, {0, 2, 6}, {0, 2, 6}, {0, 2, 6}},
                {5, 0, {5}, {1, 3, 7}, {1, 3, 7}, {1, 3, 7}},
                {6, 0, {6}, {0, 2, 4}, {0, 2, 4}, {0, 2, 4}},
                {7, 0, {7}, {1, 3, 5}, {1, 3, 5}, {1, 3, 5}}},
            std::vector<HardwareTopology::MemoryInfo>{{1024, 0}}, 1024, 2, 1, true),
        // 2 numa, 2 ht
        std::make_tuple(std::vector<HardwareTopology::Proc>{{0, 0, {0}, {2}, {2}, {2}},
                                                            {1, 0, {1}, {3}, {3}, {3}},
                                                            {2, 0, {2}, {0}, {0}, {0}},
                                                            {3, 0, {3}, {1}, {1}, {1}},
                                                            {4, 1, {4}, {6}, {6}, {6}},
                                                            {5, 1, {5}, {7}, {7}, {7}},
                                                            {6, 1, {6}, {4}, {4}, {4}},
                                                            {7, 1, {7}, {5}, {5}, {5}}},
                        std::vector<HardwareTopology::MemoryInfo>{{4096, 1}, {2048, 1}},
                        6144, 4, 2, true),
        // 1 numa, the first 4 cores have hyperthread, the last 4 cores do not have
        // hyperthread
        std::make_tuple(std::vector<HardwareTopology::Proc>{{0, 0, {0}, {2}, {2}, {2}},
                                                            {1, 0, {1}, {3}, {3}, {3}},
                                                            {2, 0, {2}, {0}, {0}, {0}},
                                                            {3, 0, {3}, {1}, {1}, {1}},
                                                            {4, 0, {4}, {}, {}, {}},
                                                            {5, 0, {5}, {}, {}, {}},
                                                            {6, 0, {6}, {}, {}, {}},
                                                            {7, 0, {7}, {}, {}, {}}},
                        std::vector<HardwareTopology::MemoryInfo>{{1024, 0}}, 1024, 6, 1,
                        true)));