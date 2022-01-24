/* Copyright 2022 Stanford University
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

#include "legion_interop.h"

#include "legion.h"
#include "cub/cub.cuh"

using namespace Legion;

typedef int V_ID;
typedef long long E_ID;
const int BLKSIZE = 512;
const int MAX_NUM_BLOCKS = 32767;

struct NodeStruct {
  E_ID index;
  V_ID degree;
};

struct EdgeStruct {
  V_ID src, dst;
};

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;                                                
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;                                               
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >; 

__global__
void load_kernel(V_ID left_bound,
                 V_ID right_bound,
                 V_ID cur_start_vtx,
                 const float* ro_dists,
                 float* ws)
{
  const V_ID tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid + cur_start_vtx <= right_bound)
  {
    V_ID cur_vtx = tid + cur_start_vtx;
    ws[cur_vtx] = ro_dists[cur_vtx];
  }
}

__global__
void pr_lb_kernel(V_ID left_bound,
                  V_ID right_bound,
                  V_ID cur_start_vtx,
                  E_ID idx_offset,
                  E_ID *col_idxs,
                  V_ID *degrees,
                  V_ID *srcs,
                  V_ID *dsts,
                  const float* ro_dists,
                  float* rw_dists)
{
  typedef cub::BlockScan<E_ID, BLKSIZE> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ float pr[BLKSIZE];
  __shared__ E_ID blkEdgeStart;
  const V_ID tid = blockIdx.x * blockDim.x + threadIdx.x;
  const V_ID blkVtxStart = blockIdx.x * blockDim.x + cur_start_vtx;
  E_ID myNumEdges = 0, scratchOffset, totalNumEdges = 0;
  V_ID myDegree = 0;
  if (tid + cur_start_vtx <= right_bound)
  {
    V_ID cur_vtx = tid + cur_start_vtx;
    E_ID start_col_idx, end_col_idx = col_idxs[cur_vtx - left_bound];
    myDegree = degrees[cur_vtx - left_bound];
    if (cur_vtx == left_bound)
      start_col_idx = idx_offset;
    else
    {
      start_col_idx = col_idxs[cur_vtx - left_bound - 1];
    }
    myNumEdges = end_col_idx - start_col_idx;
    if (threadIdx.x == 0)
      blkEdgeStart = start_col_idx;
  }
  pr[threadIdx.x] = 0;

  __syncthreads();
  BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
  E_ID done = 0;

  while (totalNumEdges >0)
  {
    if (threadIdx.x < totalNumEdges)
    {
      float src_pr = ro_dists[srcs[blkEdgeStart + done + threadIdx.x - idx_offset]];
      atomicAdd(pr + dsts[blkEdgeStart + done + threadIdx.x - idx_offset] - blkVtxStart, src_pr);
    }
    done += BLKSIZE;
    totalNumEdges -= (totalNumEdges > BLKSIZE) ? BLKSIZE : totalNumEdges;
  }
  __syncthreads();

  float my_pr = pr[threadIdx.x];
  if (myDegree != 0)
    my_pr = my_pr / myDegree;
  rw_dists[tid + cur_start_vtx - left_bound] = my_pr;
}

void pagerank(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime)
{
  assert(regions.size() == 5);
  assert(task->regions.size() == 5);
  const AccessorRO<NodeStruct, 1> acc_nodes(regions[0], FID_DATA);
  const AccessorRO<EdgeStruct, 1> acc_edges(regions[1], FID_DATA);
  const AccessorRO<float, 1> acc_pr_old(regions[2], FID_DATA);
  const AccessorWO<float, 1> acc_pr_new(regions[3], FID_DATA);
  const AccessorWO<float, 1> acc_workspace(regions[4], FID_DATA);
  Rect<1> rect_node, rect_edge, rect_pr_old, rect_pr_new, rect_workspace;
  rect_node = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_edge = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_pr_old = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_pr_new = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  rect_workspace = runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  const NodeStruct* node_ptr = acc_nodes.ptr(rect_node.lo);
  const EdgeStruct *edge_ptr = acc_edges.ptr(rect_edge.lo);
  const float *pr_old_ptr = acc_pr_old.ptr(rect_pr_old.lo);
  float *pr_new_ptr = acc_pr_new.ptr(rect_pr_new.lo);
  float *workspace = acc_workspace.ptr(rect_workspace.lo);
  V_ID left_bound = rect_node.lo[0];
  V_ID right_bound = rect_node.hi[0];
  E_ID idx_offset = rect_edge.lo[0];
  V_ID cur_vtx = left_bound;
  E_ID* col_idxs = (E_ID*) node_ptr;
  V_ID* degrees = (V_ID*)(col_idxs + rect_node.hi[0] - rect_node.lo[0] + 1);
  V_ID* srcs = (V_ID*) edge_ptr;
  V_ID* dsts = (V_ID*) (srcs + rect_edge.hi[0] - rect_edge.lo[0] + 1);

  //cudaMemcpy(workspace, pr_old_ptr, sizeof(float) * (rect_workspace.hi[0]-rect_workspace.lo[0]+1), cudaMemcpyHostToDevice);
  //float* fb_pr_ptr;
  //cudaMalloc(&fb_pr_ptr, sizeof(float) * (right_bound - left_bound + 1));
  //cudaDeviceSynchronize();
  //double cp_3 = Realm::Clock::current_time_in_microseconds();
  while (cur_vtx <= right_bound)
  {
    int num_blocks = (right_bound - cur_vtx + BLKSIZE) / BLKSIZE;
    if (num_blocks > MAX_NUM_BLOCKS)
      num_blocks = MAX_NUM_BLOCKS;
    load_kernel<<<num_blocks, BLKSIZE>>>(left_bound, right_bound, cur_vtx, pr_old_ptr, workspace);
    cur_vtx += num_blocks * BLKSIZE;
  }
  cur_vtx = left_bound;
  while (cur_vtx <= right_bound)
  {
    int num_blocks = (right_bound - cur_vtx + BLKSIZE) / BLKSIZE;
    if (num_blocks > MAX_NUM_BLOCKS)
      num_blocks = MAX_NUM_BLOCKS;
    pr_lb_kernel<<<num_blocks, BLKSIZE>>>(left_bound, right_bound, cur_vtx,
                                          idx_offset, col_idxs, degrees, srcs, dsts,
                                          workspace, workspace);
    cur_vtx += num_blocks * BLKSIZE;
  }
  cudaDeviceSynchronize();
  //double cp_4 = Realm::Clock::current_time_in_microseconds();
  //printf("time = %.2fus col_idxs(%llx) degrees(%llx) srcs(%llx) dsts(%llx) ws(%llx)\n", cp_4 - cp_3, col_idxs, degrees, srcs, dsts, workspace);
}

__global__
void init_kernel(V_ID left_bound,
                 V_ID right_bound,
                 V_ID cur_start_vtx,
                 float* rw_dists,
                 float value)
{
  const V_ID tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid + cur_start_vtx <= right_bound)
  {
    V_ID cur_vtx = tid + cur_start_vtx;
    rw_dists[cur_vtx] = value;
  }
}


void init_pr_score(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const AccessorWO<float, 1> acc_pr_new(regions[0], FID_DATA);
  Rect<1> rect_node, rect_edge, rect_pr_old, rect_pr_new, rect_workspace;
  rect_pr_new = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  float *pr_new_ptr = acc_pr_new.ptr(rect_pr_new.lo);
  V_ID left_bound = rect_node.lo[0];
  V_ID right_bound = rect_node.hi[0];
  V_ID cur_vtx = left_bound;
  float value = 1.0f / (right_bound - left_bound + 1);
  while (cur_vtx <= right_bound)
  {
    int num_blocks = (right_bound - cur_vtx + BLKSIZE) / BLKSIZE;
    if (num_blocks > MAX_NUM_BLOCKS)
      num_blocks = MAX_NUM_BLOCKS;
    init_kernel<<<num_blocks, BLKSIZE>>>(left_bound, right_bound, cur_vtx, pr_new_ptr, value);
    cur_vtx += num_blocks * BLKSIZE;
  }
}

void register_tasks()
{
  {
    TaskVariantRegistrar registrar(TID_F, "pagerank");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<pagerank>(registrar, "pagerank");
  }
  {
    TaskVariantRegistrar registrar(TID_F2, "init_pr_score");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<init_pr_score>(registrar, "init_pr_score");
  }
}
