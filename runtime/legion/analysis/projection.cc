/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "legion/analysis/projection.h"
#include "legion/contexts/replicate.h"
#include "legion/kernel/runtime.h"
#include "legion/api/functors_impl.h"
#include "legion/nodes/region.h"
#include "legion/utilities/privileges.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // ProjectionInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionInfo::ProjectionInfo(
        const RegionRequirement* req, IndexSpaceNode* launch_space,
        ShardingFunction* func, IndexSpace shard_space)
      : projection(nullptr), projection_type(LEGION_SINGULAR_PROJECTION),
        projection_space(nullptr), sharding_function(func),
        sharding_space(nullptr)
    //--------------------------------------------------------------------------
    {
      if (launch_space == nullptr)
        return;

      if (shard_space.exists())
        sharding_space = runtime->get_node(shard_space);
      else if (func != nullptr)
      {
        sharding_space = launch_space;
      }

      // There is special logic here to handle the case of singular region
      // requirements with sharding functions which we still want to treat as
      // projection region requirements for the logical analysis
      // Should always have a launch space with a sharding function
      legion_assert((func == nullptr) || (launch_space != nullptr));
      if (req->handle_type == LEGION_SINGULAR_PROJECTION)
      {
        if (func != nullptr)
        {
          // Treat single region requirements with sharding functions
          // as projections with the identity functor
          projection = runtime->find_projection_function(0 /*identity*/);
          projection_type = LEGION_REGION_PROJECTION;
          projection_space = launch_space;
        }
        else
        {
          projection = nullptr;
          projection_type = req->handle_type;
          projection_space = nullptr;
        }
      }
      else
      {
        projection = runtime->find_projection_function(req->projection);
        projection_type = req->handle_type;
        projection_space = launch_space;
      }
    }

    //--------------------------------------------------------------------------
    bool ProjectionInfo::is_complete_projection(
        RegionTreeNode* node, const LogicalUser& user) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_projecting());
      return projection->is_complete(node, user.op, user.idx, projection_space);
    }

    /////////////////////////////////////////////////////////////
    // ShardedColorMap
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardID ShardedColorMap::at(LegionColor color) const
    //--------------------------------------------------------------------------
    {
      std::unordered_map<LegionColor, ShardID>::const_iterator finder =
          color_shards.find(color);
      legion_assert(finder != color_shards.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void ShardedColorMap::pack(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(color_shards.size());
      for (const std::pair<const LegionColor, ShardID>& it : color_shards)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardedColorMap::pack_empty(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    /*static*/ ShardedColorMap* ShardedColorMap::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_colors;
      derez.deserialize(num_colors);
      if (num_colors == 0)
        return nullptr;
      std::unordered_map<LegionColor, ShardID> color_shards;
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        LegionColor color;
        derez.deserialize(color);
        derez.deserialize(color_shards[color]);
      }
      return new ShardedColorMap(std::move(color_shards));
    }

    /////////////////////////////////////////////////////////////
    // ProjectionNode
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void ProjectionNode::IntervalTree::add_child(LegionColor color)
    //--------------------------------------------------------------------------
    {
      add_range(color, color + 1);
    }

    //--------------------------------------------------------------------------
    void ProjectionNode::IntervalTree::remove_child(LegionColor color)
    //--------------------------------------------------------------------------
    {
      // Don't contain the color
      if (ranges.empty())
        return;
      std::map<LegionColor, LegionColor>::iterator finder =
          ranges.upper_bound(color);
      // Don't contain the color
      if (finder == ranges.begin())
        return;
      // Get the interval right before
      finder = std::prev(finder);
      // Don't have the color in the previous interval
      if (finder->second <= color)
        return;
      // Start a new range if this wasn't the last color in the range
      if ((color + 1) < finder->second)
        ranges[color + 1] = finder->second;
      // See if it is at the start of the range in which case this is easy
      if (finder->first == color)
        // Remove the old range
        ranges.erase(finder);
      else
        // Update the range to end at the color which was removed
        finder->second = color;
    }

    //--------------------------------------------------------------------------
    void ProjectionNode::IntervalTree::add_range(
        LegionColor start, LegionColor stop)
    //--------------------------------------------------------------------------
    {
      if (!ranges.empty())
      {
        // Find the first range that might contain the start point
        std::map<LegionColor, LegionColor>::iterator next =
            ranges.upper_bound(start);
        if (next != ranges.begin())
        {
          std::map<LegionColor, LegionColor>::iterator prev = std::prev(next);
          // Note that prev->first <= start because of upper_bound
          // Start is contained in this interval so we can merge it
          if (start < prev->second)
          {
            // Interval contained in already existing range
            if (stop <= prev->second)
              return;
            start = prev->first;
            ranges.erase(prev);
          }
        }
        // Now merge forwards with any future ranges
        while ((next != ranges.end()) && (next->first <= stop))
        {
          if (stop < next->second)
            stop = next->second;
          std::map<LegionColor, LegionColor>::iterator to_delete = next++;
          ranges.erase(to_delete);
        }
      }
      // Add this range to the tree
      ranges[start] = stop;
    }

    //--------------------------------------------------------------------------
    bool ProjectionNode::IntervalTree::has_child(LegionColor color) const
    //--------------------------------------------------------------------------
    {
      if (ranges.empty())
        return false;
      // Find the first interval that starts after the color
      std::map<LegionColor, LegionColor>::const_iterator finder =
          ranges.upper_bound(color);
      // If it's the first interval, then we can't contain the color
      if (finder == ranges.begin())
        return false;
      // Otherwise step backwards to get the interval right before
      finder = std::prev(finder);
      // See if that interval includes it or not
      // Note that (finder->first <= color) is already implied by upper_bound
      return (color < finder->second);
    }

    //--------------------------------------------------------------------------
    void ProjectionNode::IntervalTree::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(ranges.size());
      for (const std::pair<const LegionColor, LegionColor>& it : ranges)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionNode::IntervalTree::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_ranges;
      derez.deserialize(num_ranges);
      for (unsigned idx = 0; idx < num_ranges; idx++)
      {
        LegionColor start, end;
        derez.deserialize(start);
        derez.deserialize(end);
        add_range(start, end);
      }
    }

#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
    //--------------------------------------------------------------------------
    ProjectionNode::ShardSet::ShardSet(void) : size(0), max(MAX_VALUES)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ProjectionNode::ShardSet::~ShardSet(void)
    //--------------------------------------------------------------------------
    {
      if (max > MAX_VALUES)
        free(set.buffer);
    }

    //--------------------------------------------------------------------------
    void ProjectionNode::ShardSet::insert(ShardID shard, unsigned total_shards)
    //--------------------------------------------------------------------------
    {
      if (max == MAX_VALUES)
      {
        if (size == 0)
          set.values[size++] = shard;
        else if (!std::binary_search(
                     &set.values[0], &set.values[size], shard,
                     std::less<ShardID>()))
        {
          if (size == MAX_VALUES)
          {
            // Need to increase to intermediate size
            unsigned new_max = max * 2;
            unsigned bit_max =
                (((total_shards + 7) / 8) + sizeof(ShardID) - 1) /
                sizeof(ShardID);
            if (new_max < bit_max)
            {
              // Going to sparse buffer representation
              max = new_max;
              ShardID* buffer = (ShardID*)malloc(new_max * sizeof(ShardID));
              for (unsigned idx = 0; idx < MAX_VALUES; idx++)
                buffer[idx] = set.values[idx];
              buffer[size++] = shard;
              std::sort(buffer, buffer + size, std::less<ShardID>());
              set.buffer = buffer;
            }
            else
            {
              size = bit_max;
              max = total_shards + 1;
              ShardID* buffer = (ShardID*)malloc(bit_max * sizeof(ShardID));
              constexpr size_t power = STATIC_LOG2(sizeof(ShardID));
              for (unsigned idx = 0; idx < MAX_VALUES; idx++)
              {
                unsigned index = set.values[idx] >> power;
                buffer[idx] |=
                    (1U << (set.values[index] & ((1U << power) - 1)));
              }
              unsigned index = shard >> power;
              buffer[index] |= (1U << (shard & ((1U << power) - 1)));
              set.buffer = buffer;
            }
          }
          else
          {
            set.values[size++] = shard;
            std::sort(&set.values[0], &set.values[size], std::less<ShardID>());
          }
        }
      }
      else if (total_shards < max)
      {
        // We're already in a bitmask mode
        constexpr size_t power = STATIC_LOG2(sizeof(ShardID));
        unsigned index = shard >> power;
        set.buffer[index] |= (1U << (shard & ((1U << power) - 1)));
      }
      else if (!std::binary_search(
                   &set.buffer[0], &set.buffer[size], shard,
                   std::less<ShardID>()))
      {
        if (size < max)
        {
          set.buffer[size++] = shard;
          std::sort(set.buffer, set.buffer + size, std::less<ShardID>());
        }
        else
        {
          unsigned new_max = max * 2;
          unsigned bit_max = (((total_shards + 7) / 8) + sizeof(ShardID) - 1) /
                             sizeof(ShardID);
          if (new_max < bit_max)
          {
            // Going to sparse buffer representation
            max = new_max;
            ShardID* buffer = (ShardID*)malloc(new_max * sizeof(ShardID));
            for (unsigned idx = 0; idx < size; idx++)
              buffer[idx] = set.buffer[idx];
            buffer[size++] = shard;
            std::sort(buffer, buffer + size, std::less<ShardID>());
            free(set.buffer);
            set.buffer = buffer;
          }
          else
          {
            max = total_shards + 1;
            ShardID* buffer = (ShardID*)malloc(bit_max * sizeof(ShardID));
            constexpr size_t power = STATIC_LOG2(sizeof(ShardID));
            for (unsigned idx = 0; idx < size; idx++)
            {
              unsigned index = set.buffer[idx] >> power;
              buffer[index] |= (1U << (set.buffer[idx] & ((1U << power) - 1)));
            }
            unsigned index = shard >> power;
            buffer[index] |= (1U << (shard & ((1U << power) - 1)));
            free(set.buffer);
            set.buffer = buffer;
            size = bit_max;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    ShardID ProjectionNode::ShardSet::find_nearest_shard(
        ShardID local, unsigned total_shards) const
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      if (max == MAX_VALUES)
      {
        if (size == 1)
          return set.values[0];
        return find_nearest(local, total_shards, &set.values[0], size);
      }
      else if (total_shards < max)
      {
        constexpr size_t power = STATIC_LOG2(sizeof(ShardID));
        const unsigned bit_max =
            (((total_shards + 7) / 8) + sizeof(ShardID) - 1) / sizeof(ShardID);
        // Find the next set bit both above and below
        ShardID lower = 0;
        {
          int index = local >> power;
          int offset = local & ((1U << power) - 1);
          // Shouldn't exist in the set
          legion_assert(!(set.buffer[index] & (1U << offset)));
          offset--;
          while (true)
          {
            while (offset >= 0)
            {
              if (set.buffer[index] & (1U << offset))
              {
                lower = (index << power) + offset;
                break;
              }
              else
                offset--;
            }
            if (offset < 0)
            {
              offset = ((1U << power) - 1);
              // Handle wrap around case
              if (--index < 0)
                index = bit_max - 1;
            }
            else
              break;
          }
        }
        ShardID upper = 0;
        {
          unsigned index = local >> power;
          unsigned offset = (local & ((1U << power) - 1)) + 1;
          while (true)
          {
            while (offset < (1U << power))
            {
              if (set.buffer[index] & (1U << offset))
              {
                upper = (index << power) + offset;
                break;
              }
              else
                offset++;
            }
            if (offset == (1U << power))
            {
              offset = 0;
              // Handle wrap around case
              if (++index == bit_max)
                index = 0;
            }
            else
              break;
          }
        }
        unsigned lower_dist = find_distance(lower, local, total_shards);
        unsigned upper_dist = find_distance(local, upper, total_shards);
        if (lower_dist < upper_dist)
          return lower;
        else
          return upper;
      }
      else
        return find_nearest(local, total_shards, set.buffer, size);
    }

    //--------------------------------------------------------------------------
    ShardID ProjectionNode::ShardSet::find_nearest(
        ShardID local, unsigned total_shards, const ShardID* buffer,
        unsigned buffer_size) const
    //--------------------------------------------------------------------------
    {
      // Find the upper bound if it exists
      unsigned upper = 0;
      unsigned count = buffer_size;
      while (count > 0)
      {
        unsigned step = count / 2;
        if (local >= buffer[upper + step])
        {
          upper = step + 1;
          count -= step + 1;
        }
        else
          count = step;
      }
      legion_assert(upper <= buffer_size);
      // Check to see if the upper bound exists or not
      unsigned lower = 0;
      if (upper == buffer_size)
      {
        legion_assert(buffer[upper - 1] < local);
        lower = buffer_size - 1;
        upper = 0;
      }
      else if (upper == 0)
      {
        legion_assert(local < buffer[0]);
        lower = buffer_size - 1;
      }
      else
      {
        lower = buffer[upper - 1];
        legion_assert(buffer[lower] < local);
        legion_assert(local < buffer[upper]);
      }
      // Figure out which of the two is closer
      unsigned lower_dist = find_distance(buffer[lower], local, total_shards);
      unsigned upper_dist = find_distance(local, buffer[upper], total_shards);
      if (lower_dist < upper_dist)
        return buffer[lower];
      else
        return buffer[upper];
    }

    //--------------------------------------------------------------------------
    /*static*/ unsigned ProjectionNode::ShardSet::find_distance(
        ShardID one, ShardID two, unsigned total_shards)
    //--------------------------------------------------------------------------
    {
      unsigned abs_diff = (one < two) ? (two - one) : (one - two);
      // closest distance with wrap around
      if (abs_diff < (total_shards / 2))
        return abs_diff;
      else
        return (total_shards - abs_diff);
    }

    //--------------------------------------------------------------------------
    void ProjectionNode::ShardSet::serialize(
        Serializer& rez, unsigned total_shards) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(max);
      if (max == MAX_VALUES)
      {
        rez.serialize(size);
        for (unsigned idx = 0; idx < size; idx++)
          rez.serialize(set.values[idx]);
      }
      else if (total_shards < max)
      {
        for (unsigned idx = 0; idx < size; idx++)
          rez.serialize(set.buffer[idx]);
      }
      else
      {
        rez.serialize(size);
        for (unsigned idx = 0; idx < size; idx++)
          rez.serialize(set.buffer[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionNode::ShardSet::deserialize(
        Deserializer& derez, unsigned total_shards)
    //--------------------------------------------------------------------------
    {
      unsigned dmax;
      derez.deserialize(dmax);
      if (dmax == MAX_VALUES)
      {
        unsigned dsize;
        derez.deserialize(dsize);
        for (unsigned idx = 0; idx < dsize; idx++)
        {
          ShardID shard;
          derez.deserialize(shard);
          insert(shard, total_shards);
        }
      }
      else if (total_shards < dmax)
      {
        if (total_shards < max)
        {
          for (unsigned idx = 0; idx < size; idx++)
          {
            ShardID bits;
            derez.deserialize(bits);
            set.buffer[idx] |= bits;
          }
        }
        else
        {
          unsigned bit_max = (((total_shards + 7) / 8) + sizeof(ShardID) - 1) /
                             sizeof(ShardID);
          ShardID* buffer = (ShardID*)malloc(bit_max * sizeof(ShardID));
          for (unsigned idx = 0; idx < bit_max; idx++)
            derez.deserialize(buffer[idx]);
          legion_assert(max < bit_max);
          constexpr size_t power = STATIC_LOG2(sizeof(ShardID));
          if (max == MAX_VALUES)
          {
            for (unsigned idx = 0; idx < size; idx++)
            {
              unsigned index = set.values[idx] >> power;
              buffer[index] |= (1U << (set.values[idx] & ((1U << power) - 1)));
            }
          }
          else
          {
            for (unsigned idx = 0; idx < size; idx++)
            {
              unsigned index = set.buffer[idx] >> power;
              buffer[index] |= (1U << (set.buffer[idx] & ((1U << power) - 1)));
            }
            free(set.buffer);
          }
          max = total_shards + 1;
          size = bit_max;
          set.buffer = buffer;
        }
      }
      else
      {
        unsigned dsize;
        derez.deserialize(dsize);
        for (unsigned idx = 0; idx < dsize; idx++)
        {
          ShardID shard;
          derez.deserialize(shard);
          insert(shard, total_shards);
        }
      }
    }
#endif  // LEGION_NAME_BASED_CHILDREN_SHARDS

    /////////////////////////////////////////////////////////////
    // ProjectionRegion
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionRegion::ProjectionRegion(RegionNode* node) : region(node)
    //--------------------------------------------------------------------------
    {
      region->add_base_gc_ref(PROJECTION_REF);
    }

    //--------------------------------------------------------------------------
    ProjectionRegion::~ProjectionRegion(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const LegionColor, ProjectionPartition*>& it :
           local_children)
        if (it.second->remove_reference())
          delete it.second;
      if (region->remove_base_gc_ref(PROJECTION_REF))
        delete region;
    }

    //--------------------------------------------------------------------------
    bool ProjectionRegion::is_disjoint(void) const
    //--------------------------------------------------------------------------
    {
      if (local_children.size() > 1)
        return false;
      if (!local_children.empty() &&
          !local_children.begin()->second->is_disjoint())
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionRegion::is_leaves_only(void) const
    //--------------------------------------------------------------------------
    {
      if (shard_users.empty())
      {
        for (const std::pair<const LegionColor, ProjectionPartition*>& it :
             local_children)
          if (!it.second->is_leaves_only())
            return false;
        return true;
      }
      else
        return local_children.empty();
    }

    //--------------------------------------------------------------------------
    bool ProjectionRegion::is_unique_shards(void) const
    //--------------------------------------------------------------------------
    {
      if (shard_users.size() > 1)
        return false;
      for (const std::pair<const LegionColor, ProjectionPartition*>& it :
           local_children)
        if (!it.second->is_unique_shards())
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionRegion::interferes(
        ProjectionNode* other, ShardID local_shard, bool& dominates) const
    //--------------------------------------------------------------------------
    {
      ProjectionRegion* rhs = legion_safe_cast<ProjectionRegion*>(other);
      legion_assert(region == rhs->region);
      return has_interference(rhs, local_shard, dominates);
    }

    //--------------------------------------------------------------------------
    bool ProjectionRegion::pointwise_dominates(
        const ProjectionNode* other) const
    //--------------------------------------------------------------------------
    {
      const ProjectionRegion* rhs =
          legion_safe_cast<const ProjectionRegion*>(other);
      legion_assert(region == rhs->region);
      return has_pointwise_dominance(rhs);
    }

    //--------------------------------------------------------------------------
    void ProjectionRegion::extract_shard_summaries(
        bool supports_name_based, ShardID local_shard, size_t total_shards,
        std::map<LogicalRegion, RegionSummary>& region_summaries,
        std::map<LogicalPartition, PartitionSummary>& partition_summaries) const
    //--------------------------------------------------------------------------
    {
      legion_assert(
          region_summaries.find(region->handle) == region_summaries.end());
      RegionSummary& summary = region_summaries[region->handle];
      summary.users = shard_users;
      for (const std::pair<const LegionColor, ProjectionPartition*>& it :
           local_children)
      {
        summary.children.add_child(it.first);
        it.second->extract_shard_summaries(
            supports_name_based, local_shard, total_shards, region_summaries,
            partition_summaries);
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionRegion::update_shard_summaries(
        bool supports_name_based, ShardID local_shard, size_t total_shards,
        std::map<LogicalRegion, RegionSummary>& region_summaries,
        std::map<LogicalPartition, PartitionSummary>& partition_summaries)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          region_summaries.find(region->handle) != region_summaries.end());
      RegionSummary& summary = region_summaries[region->handle];
      shard_users.swap(summary.users);
      shard_children.swap(summary.children);
      if (supports_name_based && !shard_children.empty())
      {
        // Check to see if we have a child but haven't recorded a local
        // one in which case we need to make a child for the partition
        // locally and unpack it so we can have the knowledge of which
        // shards know about the subregions of the partition
        legion_assert(shard_children.ranges.size() == 1);
        std::map<LegionColor, LegionColor>::const_iterator it =
            shard_children.ranges.begin();
        // Should only be one color
        legion_assert((it->first + 1) == it->second);
        if (local_children.empty())
          local_children[it->first] =
              new ProjectionPartition(region->get_child(it->first));
        else
          legion_assert(local_children.find(it->first) != local_children.end());
        legion_assert(local_children.size() == 1);
      }
      // Remove all our local children from the shard children
      for (const std::pair<const LegionColor, ProjectionPartition*>& it :
           local_children)
      {
        shard_children.remove_child(it.first);
        it.second->update_shard_summaries(
            supports_name_based, local_shard, total_shards, region_summaries,
            partition_summaries);
      }
    }

    //--------------------------------------------------------------------------
    bool ProjectionRegion::has_interference(
        ProjectionRegion* other, ShardID local_shard, bool& dominates) const
    //--------------------------------------------------------------------------
    {
      // If either one has more than one shard ID then we're done
      if ((shard_users.size() > 1) || (other->shard_users.size() > 1))
        return true;
      if (!shard_users.empty() && (shard_users.back() != local_shard))
        return true;
      if (!other->shard_users.empty() &&
          (other->shard_users.back() != local_shard))
        return true;
      // If either has any shard children we're immediately interfering
      if (!shard_children.empty() || !other->shard_children.empty())
        return true;
      // If we have different numbers of partitions then we are definitely
      // going to be interfering on something
      if (local_children.size() != other->local_children.size())
        return true;
      for (const std::pair<const LegionColor, ProjectionPartition*>& it :
           local_children)
      {
        std::unordered_map<LegionColor, ProjectionPartition*>::const_iterator
            finder = other->local_children.find(it.first);
        if (finder == other->local_children.end())
          return true;
        if (it.second->has_interference(finder->second, local_shard, dominates))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool ProjectionRegion::has_pointwise_dominance(
        const ProjectionRegion* other) const
    //--------------------------------------------------------------------------
    {
      if (other->shard_users.empty())
      {
        legion_assert(!other->local_children.empty());
        if (!shard_users.empty())
          return false;
        for (const std::pair<const LegionColor, ProjectionPartition*>& it :
             other->local_children)
        {
          std::unordered_map<LegionColor, ProjectionPartition*>::const_iterator
              finder = local_children.find(it.first);
          if (finder == local_children.end())
            return false;
          if (!finder->second->has_pointwise_dominance(it.second))
            return false;
        }
        return true;
      }
      else
      {
        // Would violate name-based analysis
        legion_assert(other->local_children.empty());
        // If we don't have any other local children then we can do
        // pointwise analysis regardless of where the shards are
        return local_children.empty();
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionRegion::add_user(ShardID user)
    //--------------------------------------------------------------------------
    {
      if (std::binary_search(shard_users.begin(), shard_users.end(), user))
        return;
      shard_users.emplace_back(user);
      std::sort(shard_users.begin(), shard_users.end());
    }

    //--------------------------------------------------------------------------
    void ProjectionRegion::add_child(ProjectionPartition* child)
    //--------------------------------------------------------------------------
    {
      LegionColor color = child->partition->row_source->color;
      if (local_children.insert(std::make_pair(color, child)).second)
        child->add_reference();
    }

    /////////////////////////////////////////////////////////////
    // ProjectionPartition
    /////////////////////////////////////////////////////////////

#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
    //--------------------------------------------------------------------------
    ProjectionPartition::ProjectionPartition(
        PartitionNode* node, ShardedColorMap* map)
      : partition(node), name_based_children_shards(map)
    //--------------------------------------------------------------------------
    {
      partition->add_base_gc_ref(PROJECTION_REF);
      if (name_based_children_shards != nullptr)
        name_based_children_shards->add_reference();
    }
#else
    //--------------------------------------------------------------------------
    ProjectionPartition::ProjectionPartition(PartitionNode* node)
      : partition(node)
    //--------------------------------------------------------------------------
    {
      partition->add_base_gc_ref(PROJECTION_REF);
    }
#endif

    //--------------------------------------------------------------------------
    ProjectionPartition::~ProjectionPartition(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const LegionColor, ProjectionRegion*>& it :
           local_children)
        if (it.second->remove_reference())
          delete it.second;
      if (partition->remove_base_gc_ref(PROJECTION_REF))
        delete partition;
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      if ((name_based_children_shards != nullptr) &&
          name_based_children_shards->remove_reference())
        delete name_based_children_shards;
#endif
    }

    //--------------------------------------------------------------------------
    bool ProjectionPartition::is_disjoint(void) const
    //--------------------------------------------------------------------------
    {
      if (!partition->row_source->is_disjoint(false /*from app*/))
        return false;
      for (const std::pair<const LegionColor, ProjectionRegion*>& it :
           local_children)
        if (!it.second->is_disjoint())
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionPartition::is_leaves_only(void) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const LegionColor, ProjectionRegion*>& it :
           local_children)
        if (!it.second->is_leaves_only())
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionPartition::is_unique_shards(void) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const LegionColor, ProjectionRegion*>& it :
           local_children)
        if (!it.second->is_unique_shards())
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionPartition::interferes(
        ProjectionNode* other, ShardID local_shard, bool& dominates) const
    //--------------------------------------------------------------------------
    {
      ProjectionPartition* rhs = legion_safe_cast<ProjectionPartition*>(other);
      legion_assert(partition == rhs->partition);
      return has_interference(rhs, local_shard, dominates);
    }

    //--------------------------------------------------------------------------
    bool ProjectionPartition::pointwise_dominates(
        const ProjectionNode* other) const
    //--------------------------------------------------------------------------
    {
      const ProjectionPartition* rhs =
          legion_safe_cast<const ProjectionPartition*>(other);
      legion_assert(partition == rhs->partition);
      return has_pointwise_dominance(rhs);
    }

    //--------------------------------------------------------------------------
    void ProjectionPartition::extract_shard_summaries(
        bool supports_name_based, ShardID local_shard, size_t total_shards,
        std::map<LogicalRegion, RegionSummary>& region_summaries,
        std::map<LogicalPartition, PartitionSummary>& partition_summaries) const
    //--------------------------------------------------------------------------
    {
      legion_assert(
          partition_summaries.find(partition->handle) ==
          partition_summaries.end());
      PartitionSummary& summary = partition_summaries[partition->handle];
      for (const std::pair<const LegionColor, ProjectionRegion*>& it :
           local_children)
      {
        summary.children.add_child(it.first);
        it.second->extract_shard_summaries(
            supports_name_based, local_shard, total_shards, region_summaries,
            partition_summaries);
      }
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      if (supports_name_based)
      {
        // Record that we know about all of our local children
        for (const std::pair<const LegionColor, ProjectionRegion*>& it :
             local_children)
          summary.disjoint_complete_child_shards[it.first].insert(
              local_shard, total_shards);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void ProjectionPartition::update_shard_summaries(
        bool supports_name_based, ShardID local_shard, size_t total_shards,
        std::map<LogicalRegion, RegionSummary>& region_summaries,
        std::map<LogicalPartition, PartitionSummary>& partition_summaries)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          partition_summaries.find(partition->handle) !=
          partition_summaries.end());
      PartitionSummary& summary = partition_summaries[partition->handle];
      shard_children.swap(summary.children);
      // Remove all our local children from the shard children
      for (const std::pair<const LegionColor, ProjectionRegion*>& it :
           local_children)
      {
        shard_children.remove_child(it.first);
        it.second->update_shard_summaries(
            supports_name_based, local_shard, total_shards, region_summaries,
            partition_summaries);
      }
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      if (supports_name_based &&
          (local_children.size() < partition->get_num_children()))
      {
        // If necessary create a sharded color map for any children
        // which we don't know about locally so we know the nearest
        // shard which does have some information about it
        std::unordered_map<LegionColor, ShardID> nearest_shards;
        for (const std::pair<const LegionColor, ShardSet>& it :
             summary.disjoint_complete_child_shards)
        {
          if (local_children.find(it.first) != local_children.end())
            continue;
          nearest_shards[it.first] =
              it->second.find_nearest_shard(local_shard, total_shards);
        }
        // Now we can make our ShardedColorMap and save it
        legion_assert(name_based_children_shards == nullptr);
        name_based_children_shards =
            new ShardedColorMap(std::move(nearest_shards));
        name_based_children_shards->add_reference();
      }
#endif  // LEGION_NAME_BASED_CHILDREN_SHARDS
    }

    //--------------------------------------------------------------------------
    bool ProjectionPartition::has_interference(
        ProjectionPartition* other, ShardID local_shard, bool& dominates) const
    //--------------------------------------------------------------------------
    {
      if (partition->row_source->is_disjoint(false /*from app*/))
      {
        // Disjoint partition, check all the children against each other
        for (const std::pair<const LegionColor, ProjectionRegion*>& it :
             local_children)
        {
          std::unordered_map<LegionColor, ProjectionRegion*>::const_iterator
              finder = other->local_children.find(it.first);
          if (finder == other->local_children.end())
          {
            // Check to see if there is a remote shard with that child
            if (other->shard_children.has_child(it.first))
              return true;
            // Otherwise this does not dominate us
            dominates = false;
          }
          else if (it.second->has_interference(
                       finder->second, local_shard, dominates))
            return true;
        }
        // Check in the opposite direction too
        for (const std::pair<const LegionColor, ProjectionRegion*>& it :
             other->local_children)
        {
          std::unordered_map<LegionColor, ProjectionRegion*>::const_iterator
              finder = local_children.find(it.first);
          if (finder == local_children.end())
          {
            if (shard_children.has_child(it.first))
              return true;
          }
          // Else we've already done interferences with ourself locally
        }
        return false;
      }
      else
      {
        // Aliased partition
        // If either has any shard children we're immediately interfering
        if (!shard_children.empty() || !other->shard_children.empty())
          return true;
        // If we have different numbers of partitions then we are definitely
        // going to be interfering on something
        if (local_children.size() != other->local_children.size())
          return true;
        for (const std::pair<const LegionColor, ProjectionRegion*>& it :
             local_children)
        {
          std::unordered_map<LegionColor, ProjectionRegion*>::const_iterator
              finder = other->local_children.find(it.first);
          if (finder == other->local_children.end())
            return true;
          if (it.second->has_interference(
                  finder->second, local_shard, dominates))
            return true;
        }
        return false;
      }
    }

    //--------------------------------------------------------------------------
    bool ProjectionPartition::has_pointwise_dominance(
        const ProjectionPartition* other) const
    //--------------------------------------------------------------------------
    {
      // Should be disjoint or would violate name-based-self-analysis
      legion_assert(partition->row_source->is_disjoint(false /*from app*/));
      for (const std::pair<const LegionColor, ProjectionRegion*>& it :
           other->local_children)
      {
        std::unordered_map<LegionColor, ProjectionRegion*>::const_iterator
            finder = local_children.find(it.first);
        if (finder == local_children.end())
          return false;
        if (!finder->second->has_pointwise_dominance(it.second))
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void ProjectionPartition::add_child(ProjectionRegion* child)
    //--------------------------------------------------------------------------
    {
      LegionColor color = child->region->row_source->color;
      if (local_children.insert(std::make_pair(color, child)).second)
        child->add_reference();
    }

    /////////////////////////////////////////////////////////////
    // ProjectionSummary
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(
        const ProjectionInfo& proj_info, ProjectionNode* node, Operation* op,
        unsigned index, const RegionRequirement& req, LogicalState* state)
      : owner(state), domain(proj_info.projection_space),
        projection(proj_info.projection), sharding(proj_info.sharding_function),
        sharding_domain(proj_info.sharding_space),
        arglen(req.projection_args_size),
        args((arglen > 0) ? malloc(arglen) : nullptr), tree(node),
        exchange(nullptr),
        // Special case here: if we can't prove its disjoint by the region tree
        // but we know that all the regions are writing and the projection
        // function is not invertible then the user is guaranteeing use that all
        // the point in this projection function are disjoint from each other
        disjoint(
            (IS_WRITE(req) && !proj_info.projection->is_invertible) ||
            tree->is_disjoint()),
        complete(proj_info.projection->is_complete(
            state->owner, op, index, proj_info.projection_space)),
        permits_name_based_self_analysis(disjoint && tree->is_leaves_only()),
        unique_shard_users(
            true) /* no shard in non-control-replicated context */
    //--------------------------------------------------------------------------
    {
      legion_assert(proj_info.is_projecting());
      legion_assert(tree != nullptr);
      if (domain != nullptr)
        domain->add_base_gc_ref(PROJECTION_REF);
      if (sharding_domain != nullptr)
        sharding_domain->add_base_gc_ref(PROJECTION_REF);
      tree->add_reference();
      if (arglen > 0)
        memcpy(args, req.projection_args, arglen);
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(
        const ProjectionInfo& proj_info, ProjectionNode* node, Operation* op,
        unsigned index, const RegionRequirement& req, LogicalState* state,
        bool dis, bool unique)
      : owner(state), domain(proj_info.projection_space),
        projection(proj_info.projection), sharding(proj_info.sharding_function),
        sharding_domain(proj_info.sharding_space),
        arglen(req.projection_args_size),
        args((arglen > 0) ? malloc(arglen) : nullptr), tree(node),
        exchange(nullptr),
        // Special case here: if we can't prove its disjoint by the region tree
        // but we know that all the regions are writing and the projection
        // function is not invertible then the user is guaranteeing use that all
        // the point in this projection function are disjoint from each other
        disjoint(
            dis || (IS_WRITE(req) && !proj_info.projection->is_invertible)),
        complete(proj_info.projection->is_complete(
            state->owner, op, index, proj_info.projection_space)),
        permits_name_based_self_analysis(disjoint), unique_shard_users(unique)
    //--------------------------------------------------------------------------
    {
      legion_assert(proj_info.is_projecting());
      legion_assert(tree != nullptr);
      if (domain != nullptr)
        domain->add_base_gc_ref(PROJECTION_REF);
      if (sharding_domain != nullptr)
        sharding_domain->add_base_gc_ref(PROJECTION_REF);
      tree->add_reference();
      if (arglen > 0)
        memcpy(args, req.projection_args, arglen);
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(
        const ProjectionInfo& proj_info, ProjectionNode* node, Operation* op,
        unsigned index, const RegionRequirement& req, LogicalState* state,
        ReplicateContext* context)
      : owner(state), domain(proj_info.projection_space),
        projection(proj_info.projection), sharding(proj_info.sharding_function),
        sharding_domain(proj_info.sharding_space),
        arglen(req.projection_args_size),
        args((arglen > 0) ? malloc(arglen) : nullptr), tree(node),
        exchange(new ProjectionTreeExchange(
            tree, context, COLLECTIVE_LOC_50, disjoint,
            permits_name_based_self_analysis, unique_shard_users)),
        // Special case here: if we can't prove its disjoint by the region tree
        // but we know that all the regions are writing and the projection
        // function is not invertible then the user is guaranteeing use that all
        // the point in this projection function are disjoint from each other
        disjoint(
            (IS_WRITE(req) && !proj_info.projection->is_invertible) ||
            tree->is_disjoint()),
        complete(proj_info.projection->is_complete(
            state->owner, op, index, proj_info.projection_space)),
        permits_name_based_self_analysis(true),
        unique_shard_users(tree->is_unique_shards())
    //--------------------------------------------------------------------------
    {
      legion_assert(proj_info.is_projecting());
      legion_assert(tree != nullptr);
      exchange->perform_collective_async();
      if (domain != nullptr)
        domain->add_base_gc_ref(PROJECTION_REF);
      if (sharding_domain != nullptr)
        sharding_domain->add_base_gc_ref(PROJECTION_REF);
      tree->add_reference();
      if (arglen > 0)
        memcpy(args, req.projection_args, arglen);
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::~ProjectionSummary(void)
    //--------------------------------------------------------------------------
    {
      owner->remove_projection_summary(this);
      if ((domain != nullptr) && domain->remove_base_gc_ref(PROJECTION_REF))
        delete domain;
      if ((sharding_domain != nullptr) &&
          sharding_domain->remove_base_gc_ref(PROJECTION_REF))
        delete sharding_domain;
      if (exchange != nullptr)
      {
        exchange->perform_collective_wait(true /*block*/);
        delete exchange;
      }
      if (tree->remove_reference())
        delete tree;
      if (args != nullptr)
        free(args);
    }

    //--------------------------------------------------------------------------
    bool ProjectionSummary::matches(
        const ProjectionInfo& rhs, const RegionRequirement& req) const
    //--------------------------------------------------------------------------
    {
      if (domain != rhs.projection_space)
        return false;
      if (projection != rhs.projection)
        return false;
      if (sharding != rhs.sharding_function)
        return false;
      if (sharding_domain != rhs.sharding_space)
        return false;
      size_t proj_arglen;
      const void* projection_args = req.get_projection_args(&proj_arglen);
      if (arglen != proj_arglen)
        return false;
      if ((arglen > 0) && (memcmp(args, projection_args, arglen) != 0))
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionSummary::is_disjoint(void)
    //--------------------------------------------------------------------------
    {
      if (exchange != nullptr)
      {
        exchange->perform_collective_wait(true /*block*/);
        delete exchange;
        exchange = nullptr;
      }
      return disjoint;
    }

    //--------------------------------------------------------------------------
    bool ProjectionSummary::can_perform_name_based_self_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (exchange != nullptr)
      {
        exchange->perform_collective_wait(true /*block*/);
        delete exchange;
        exchange = nullptr;
      }
      return permits_name_based_self_analysis;
    }

    //--------------------------------------------------------------------------
    bool ProjectionSummary::has_unique_shard_users(void)
    //--------------------------------------------------------------------------
    {
      if (exchange != nullptr)
      {
        exchange->perform_collective_wait(true /*block*/);
        delete exchange;
        exchange = nullptr;
      }
      return unique_shard_users;
    }

    //--------------------------------------------------------------------------
    ProjectionNode* ProjectionSummary::get_tree(void)
    //--------------------------------------------------------------------------
    {
      if (exchange != nullptr)
      {
        exchange->perform_collective_wait(true /*block*/);
        delete exchange;
        exchange = nullptr;
      }
      return tree;
    }

    /////////////////////////////////////////////////////////////
    // Projection Tree Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionTreeExchange::ProjectionTreeExchange(
        ProjectionNode* n, ReplicateContext* ctx, CollectiveIndexLocation loc,
        bool& dis, bool& name_based, bool& unique)
      : AllGatherCollective<false>(
            ctx, ctx->get_next_collective_index(loc, true /*logical*/)),
        node(n), disjoint(dis), permits_name_based_self_analysis(name_based),
        unique_shards(unique), leaves_only(node->is_leaves_only())
    //--------------------------------------------------------------------------
    {
      // Extract our local summaries
      node->extract_shard_summaries(
          disjoint && leaves_only, local_shard, ctx->total_shards,
          region_summaries, partition_summaries);
    }

    //--------------------------------------------------------------------------
    ProjectionTreeExchange::~ProjectionTreeExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ProjectionTreeExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<bool>(disjoint);
      rez.serialize<bool>(leaves_only);
      rez.serialize<bool>(unique_shards);
      rez.serialize<size_t>(region_summaries.size());
      for (const std::pair<const LogicalRegion, RegionSummary>& it :
           region_summaries)
      {
        rez.serialize(it.first);
        it.second.children.serialize(rez);
        rez.serialize(it.second.users.size());
        for (unsigned idx = 0; idx < it.second.users.size(); idx++)
          rez.serialize(it.second.users[idx]);
      }
      rez.serialize<size_t>(partition_summaries.size());
      for (const std::pair<const LogicalPartition, PartitionSummary>& it :
           partition_summaries)
      {
        rez.serialize(it.first);
        it.second.children.serialize(rez);
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
        if (disjoint && leaves_only)
        {
          rez.serialize<size_t>(
              it.second.disjoint_complete_child_shards.size());
          for (const std::pair<const LegionColor, ShardSet>& cit :
               it.second.disjoint_complete_child_shards)
          {
            rez.serialize(cit.first);
            cit.second.serialize(rez, context->total_shards);
          }
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionTreeExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      bool dis;
      derez.deserialize<bool>(dis);
      if (!dis)
        disjoint = false;
      bool leaves;
      derez.deserialize<bool>(leaves);
      if (!leaves)
        leaves_only = false;
      bool unique;
      derez.deserialize<bool>(unique);
      if (!unique)
        unique_shards = false;
      size_t num_regions;
      derez.deserialize(num_regions);
      for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
      {
        LogicalRegion region;
        derez.deserialize(region);
        RegionSummary& summary = region_summaries[region];
        summary.children.deserialize(derez);
        size_t num_users;
        derez.deserialize(num_users);
        if (num_users > 0)
        {
          if (summary.users.empty())
          {
            summary.users.resize(num_users);
            for (unsigned idx2 = 0; idx2 < num_users; idx2++)
              derez.deserialize(summary.users[idx2]);
          }
          else
          {
            for (unsigned idx2 = 0; idx2 < num_users; idx2++)
            {
              ShardID user;
              derez.deserialize(user);
              if (std::binary_search(
                      summary.users.begin(), summary.users.end(), user))
                continue;
              summary.users.emplace_back(user);
              std::sort(summary.users.begin(), summary.users.end());
            }
          }
          if (unique_shards && (summary.users.size() > 1))
            unique_shards = false;
        }
        if (leaves_only && !summary.users.empty() && !summary.children.empty())
          leaves_only = false;
      }
      size_t num_partitions;
      derez.deserialize(num_partitions);
      for (unsigned idx = 0; idx < num_partitions; idx++)
      {
        LogicalPartition partition;
        derez.deserialize(partition);
        PartitionSummary& summary = partition_summaries[partition];
        summary.children.deserialize(derez);
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
        if (dis && leaves)
        {
          size_t num_children;
          derez.deserialize(num_children);
          for (unsigned child = 0; child < num_children; child++)
          {
            LegionColor color;
            derez.deserialize(color);
            if (disjoint && leaves_only)
              summary.disjoint_complete_child_shards[color].deserialize(
                  derez, context->total_shards);
            else
            {
              // Temporary one just to unpack correctly
              ShardSet temporary;
              temporary.deserialize(derez, context->total_shards);
            }
          }
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ProjectionTreeExchange::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      permits_name_based_self_analysis = disjoint && leaves_only;
      // Update our local summaries
      node->update_shard_summaries(
          disjoint && leaves_only, local_shard, context->total_shards,
          region_summaries, partition_summaries);
      region_summaries.clear();
      partition_summaries.clear();
      return RtEvent::NO_RT_EVENT;
    }

  }  // namespace Internal
}  // namespace Legion
