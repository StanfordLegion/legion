/* Copyright 2022 Stanford University, NVIDIA Corporation
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

namespace Legion {
  namespace Mapping {

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ void DefaultMapper::default_decompose_points(
                           const DomainT<DIM,coord_t> &point_space,
                           const std::vector<Processor> &targets,
                           const Point<DIM,coord_t> &num_blocks,
                           bool recurse, bool stealable,
                           std::vector<TaskSlice> &slices)
    //--------------------------------------------------------------------------
    {
      Point<DIM,coord_t> zeroes;
      for (int i = 0; i < DIM; i++)
        zeroes[i] = 0;
      Point<DIM,coord_t> ones;
      for (int i = 0; i < DIM; i++)
        ones[i] = 1;
      Point<DIM,coord_t> num_points = 
        point_space.bounds.hi - point_space.bounds.lo + ones;
      Rect<DIM,coord_t> blocks(zeroes, num_blocks - ones);
      size_t next_index = 0;
      slices.reserve(blocks.volume());
      for (PointInRectIterator<DIM> pir(blocks); pir(); pir++) {
        Point<DIM,coord_t> block_lo = *pir;
        Point<DIM,coord_t> block_hi = *pir + ones;
        Point<DIM,coord_t> slice_lo =
          num_points * block_lo / num_blocks + point_space.bounds.lo;
        Point<DIM,coord_t> slice_hi = 
          num_points * block_hi / num_blocks + point_space.bounds.lo - ones;
        // Construct a new slice space based on the new bounds 
        // and any existing sparsity map, tighten if necessary
        DomainT<DIM,coord_t> slice_space;
        slice_space.bounds.lo = slice_lo;
        slice_space.bounds.hi = slice_hi;
        slice_space.sparsity = point_space.sparsity;
        if (!slice_space.dense())
          slice_space = slice_space.tighten();
        if (slice_space.volume() > 0) {
          TaskSlice slice;
          slice.domain = slice_space;
          slice.proc = targets[next_index++ % targets.size()];
          slice.recurse = recurse;
          slice.stealable = stealable;
          slices.push_back(slice);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ void DefaultMapper::default_decompose_points(
                               const LegionRuntime::Arrays::Rect<DIM> &rect,
                               const std::vector<Processor> &targets,
                               const LegionRuntime::Arrays::Point<DIM> &blocks,
                               bool recurse, bool stealable,
                               std::vector<TaskSlice> &slices)
    //--------------------------------------------------------------------------
    {
      const Domain dom_rect = Domain::from_rect<DIM>(rect);
      const DomainT<DIM,coord_t> point_space = dom_rect;

      const DomainPoint dom_point = DomainPoint::from_point<DIM>(blocks);
      const Point<DIM,coord_t> num_blocks = dom_point;

      default_decompose_points(point_space, targets, num_blocks, 
                               recurse, stealable, slices);
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ Point<DIM,coord_t> DefaultMapper::default_select_num_blocks( 
                                             long long int factor, 
                                             const Rect<DIM,coord_t> &to_factor)
    //--------------------------------------------------------------------------
    {
      if (factor == 1)
      {
        Point<DIM,coord_t> ones;
        for (int i = 0; i < DIM; i++)
          ones[i] = 1;
        return ones;
      }

      // Fundamental theorem of arithmetic time!
      const unsigned num_primes = 32;
      const long long int primes[num_primes] = { 2, 3, 5, 7, 11, 13, 17, 19, 
                                        23, 29, 31, 37, 41, 43, 47, 53,
                                        59, 61, 67, 71, 73, 79, 83, 89,
                                        97, 101, 103, 107, 109, 113, 127, 131 };
      // Increase the size of the prime number table if you ever hit this
      assert(factor <= (primes[num_primes-1] * primes[num_primes-1]));
      // Factor into primes
      std::vector<int> prime_factors;
      for (unsigned idx = 0; idx < num_primes; idx++)
      {
        const long long int prime = primes[idx];
        if ((prime * prime) > factor)
          break;
        while ((factor % prime) == 0)
        {
          prime_factors.push_back(prime);
          factor /= prime;
        }
        if (factor == 1)
          break;
      }
      if (factor > 1)
        prime_factors.push_back(factor);
      // Assign prime factors onto the dimensions for the target rect
      // from the largest primes down to the smallest. The goal here
      // is to assign all of the elements (in factor) while
      // maintaining a block size that is as square as possible.
      long long int result[DIM];
      for (int i = 0; i < DIM; i++)
        result[i] = 1;
      double dim_chunks[DIM];
      for (int i = 0; i < DIM; i++)
        dim_chunks[i] = ((to_factor.hi[i] - to_factor.lo[i]) + 1);
      for (int idx = prime_factors.size()-1; idx >= 0; idx--)
      {
        // Find the dimension with the biggest dim_chunk 
        int next_dim = -1;
        double max_chunk = -1;
        for (int i = 0; i < DIM; i++)
        {
          if (dim_chunks[i] > max_chunk)
          {
            max_chunk = dim_chunks[i];
            next_dim = i;
          }
        }
        const long long int next_prime = prime_factors[idx];

        result[next_dim] *= next_prime;
        dim_chunks[next_dim] /= next_prime;
      }
      return Point<DIM,coord_t>(result);
    }

    //--------------------------------------------------------------------------
    template<bool IS_SRC>
    void DefaultMapper::default_create_copy_instance(MapperContext ctx,
                         const Copy &copy, const RegionRequirement &req, 
                         unsigned idx, std::vector<PhysicalInstance> &instances)
    //--------------------------------------------------------------------------
    {
      // See if we have all the fields covered
      std::set<FieldID> missing_fields = req.privilege_fields;
      for (std::vector<PhysicalInstance>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        it->remove_space_fields(missing_fields);
        if (missing_fields.empty())
          break;
      }
      if (missing_fields.empty())
        return;
      // If we still have fields, we need to make an instance
      // We clearly need to take a guess, let's see if we can find
      // one of our instances to use.
      Memory target_memory = default_policy_select_target_memory(ctx,
                                        copy.parent_task->current_proc,
                                        req);
      bool force_new_instances = false;
      LayoutConstraintID our_layout_id = 
       default_policy_select_layout_constraints(ctx, target_memory, 
                                                req, COPY_MAPPING,
                                                true/*needs check*/, 
                                                force_new_instances);
      LayoutConstraintSet creation_constraints = 
                  runtime->find_layout_constraints(ctx, our_layout_id);
      creation_constraints.add_constraint(
          FieldConstraint(missing_fields,
                          false/*contig*/, false/*inorder*/));
      instances.resize(instances.size() + 1);
      if (!default_make_instance(ctx, target_memory, 
            creation_constraints, instances.back(), 
            COPY_MAPPING, force_new_instances, true/*meets*/, req))
      {
        // If we failed to make it that is bad
        fprintf(stderr,"Default mapper failed allocation for "
                       "%s region requirement %d of explicit "
                       "region-to-region copy operation in task %s "
                       "(ID %lld) in memory " IDFMT " for processor "
                       IDFMT ". This means the working set of your "
                       "application is too big for the allotted "
                       "capacity of the given memory under the default "
                       "mapper's mapping scheme. You have three "
                       "choices: ask Realm to allocate more memory, "
                       "write a custom mapper to better manage working "
                       "sets, or find a bigger machine. Good luck!",
                       IS_SRC ? "source" : "destination", idx, 
                       copy.parent_task->get_task_name(),
                       copy.parent_task->get_unique_id(),
		       target_memory.id,
		       copy.parent_task->current_proc.id);
        assert(false);
      }
    }

  };
};
