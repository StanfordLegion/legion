/* Copyright 2013 Stanford University
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


#include "legion.h"
#include "legion_ops.h"
#include "region_tree.h"
#include "legion_utilities.h"
#include "legion_logging.h"
#include "legion_profiling.h"

namespace LegionRuntime {
  namespace HighLevel {
  
    // Extern declarations for loggers
    extern Logger::Category log_run;
    extern Logger::Category log_task;
    extern Logger::Category log_region;
    extern Logger::Category log_index;
    extern Logger::Category log_field;
    extern Logger::Category log_inst;
    extern Logger::Category log_spy;
    extern Logger::Category log_garbage;
    extern Logger::Category log_leak;
    extern Logger::Category log_variant;

    // Inline functions for dependence analysis

    //--------------------------------------------------------------------------
    static inline DependenceType check_for_anti_dependence(const RegionUsage &u1,
                                                           const RegionUsage &u2,
                                                           DependenceType actual)
    //--------------------------------------------------------------------------
    {
      // Check for WAR or WAW with write-only
      if (IS_READ_ONLY(u1))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(HAS_WRITE(u2)); // We know at least req1 or req2 is a writers, so if req1 is not...
#endif
        return ANTI_DEPENDENCE;
      }
      else
      {
        if (IS_WRITE_ONLY(u2))
        {
          // WAW with a write-only
          return ANTI_DEPENDENCE;
        }
        else
        {
          // This defaults to whatever the actual dependence is
          return actual;
        }
      }
    }

    //--------------------------------------------------------------------------
    static inline DependenceType check_dependence_type(const RegionUsage &u1,
                                                       const RegionUsage &u2)
    //--------------------------------------------------------------------------
    {
      // Two readers are never a dependence
      if (IS_READ_ONLY(u1) && IS_READ_ONLY(u2))
      {
        return NO_DEPENDENCE;
      }
      else if (IS_REDUCE(u1) && IS_REDUCE(u2))
      {
        // If they are the same kind of reduction, no dependence, otherwise true dependence
        if (u1.redop == u2.redop)
        {
          return NO_DEPENDENCE;
        }
        else
        {
          return TRUE_DEPENDENCE;
        }
      }
      else
      {
        // Everything in here has at least one right
#ifdef DEBUG_HIGH_LEVEL
        assert(HAS_WRITE(u1) || HAS_WRITE(u2));
#endif
        // If anything exclusive 
        if (IS_EXCLUSIVE(u1) || IS_EXCLUSIVE(u1))
        {
          return check_for_anti_dependence(u1,u2,TRUE_DEPENDENCE/*default*/);
        }
        // Anything atomic (at least one is a write)
        else if (IS_ATOMIC(u1) || IS_ATOMIC(u2))
        {
          // If they're both atomics, return an atomic dependence
          if (IS_ATOMIC(u1) && IS_ATOMIC(u2))
          {
            return check_for_anti_dependence(u1,u2,ATOMIC_DEPENDENCE/*default*/); 
          }
          // If the one that is not an atomic is a read, we're also ok
          else if ((!IS_ATOMIC(u1) && IS_READ_ONLY(u1)) ||
                   (!IS_ATOMIC(u2) && IS_READ_ONLY(u2)))
          {
            return NO_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(u1,u2,TRUE_DEPENDENCE/*default*/);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(u1) || IS_SIMULT(u2))
        {
          return check_for_anti_dependence(u1,u2,SIMULTANEOUS_DEPENDENCE/*default*/);
        }
        else if (IS_RELAXED(u1) && IS_RELAXED(u2))
        {
          // TODO: Make this truly relaxed, right now it is the same as simultaneous
          return check_for_anti_dependence(u1,u2,SIMULTANEOUS_DEPENDENCE/*default*/);
          // This is what it should be: return NO_DEPENDENCE;
          // What needs to be done:
          // - RegionNode::update_valid_instances needs to allow multiple outstanding writers
          // - RegionNode needs to detect relaxed case and make copies from all 
          //              relaxed instances to non-relaxed instance
        }
        // We should never make it here
        assert(false);
        return NO_DEPENDENCE;
      }
    }

    //--------------------------------------------------------------------------
    static inline bool perform_dependence_check(const LogicalUser &prev,
                                                const LogicalUser &next)
    //--------------------------------------------------------------------------
    {
      bool mapping_dependence = false;
      DependenceType dtype = check_dependence_type(prev.usage, next.usage);
      switch (dtype)
      {
        case NO_DEPENDENCE:
          break;
        case TRUE_DEPENDENCE:
        case ANTI_DEPENDENCE:
        case ATOMIC_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          {
            next.op->add_mapping_dependence(next.idx, prev, dtype);
            mapping_dependence = true;
            break;
          }
        default:
          assert(false);
      }
      return mapping_dependence;
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeForest::RegionTreeForest(HighLevelRuntime *rt)
      : runtime(rt),
#ifdef LOW_LEVEL_LOCKS
        context_lock(Lock::create_lock()),
        creation_lock(Lock::create_lock())
#else
        context_lock(ImmovableLock(true/*initialize*/)),
        creation_lock(ImmovableLock(true/*initialize*/))
#endif
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      lock_held = false;
#endif
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::~RegionTreeForest(void)
    //--------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      context_lock.destroy_lock();
      creation_lock.destroy_lock();
#else
      context_lock.destroy();
      creation_lock.destroy();
#endif
#ifdef DEBUG_HIGH_LEVEL
      if (!escaped_users.empty())
      {
        log_leak(LEVEL_WARNING,"Leaking escaped users (runtime bug).");
      }
      if (!escaped_copies.empty())
      {
        log_leak(LEVEL_WARNING,"Leaking escaped copies (runtime bug).");
      }
#endif
      // Now we need to go through and delete all the things that we've created
      for (std::map<IndexSpace,IndexSpaceNode*>::iterator it = index_nodes.begin();
            it != index_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<IndexPartition,IndexPartNode*>::iterator it = index_parts.begin();
            it != index_parts.end(); it++)
      {
        delete it->second;
      }
      for (std::map<FieldSpace,FieldSpaceNode*>::iterator it = field_nodes.begin(); 
            it != field_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<LogicalRegion,RegionNode*>::iterator it = region_nodes.begin();
            it != region_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<LogicalPartition,PartitionNode*>::iterator it = part_nodes.begin();
            it != part_nodes.end(); it++)
      {
        delete it->second;
      }
      for (std::map<InstanceKey,InstanceView*>::iterator it = views.begin();
            it != views.end(); it++)
      {
        delete it->second;
      }
      for (std::map<UniqueManagerID,InstanceManager*>::iterator it = managers.begin();
            it != managers.end(); it++)
      {
        delete it->second;
      }
      for (std::map<InstanceKey,ReductionView*>::iterator it = reduc_views.begin();
            it != reduc_views.end(); it++)
      {
        delete it->second;
      }
      for (std::map<UniqueManagerID,ReductionManager*>::iterator it = reduc_managers.begin();
            it != reduc_managers.end(); it++)
      {
        delete it->second;
      }
      if (!created_index_trees.empty())
      {
        for (std::list<IndexSpace>::const_iterator it = created_index_trees.begin();
              it != created_index_trees.end(); it++)
        {
          log_leak(LEVEL_WARNING,"The index space tree rooted at index space %x was not deleted",
                                  it->id);
        }
      }
      if (!created_field_spaces.empty())
      {
        for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
              it != created_field_spaces.end(); it++)
        {
          log_leak(LEVEL_WARNING,"The field space %x was not deleted", it->id);
        }
      }
      if (!created_region_trees.empty())
      {
        for (std::list<LogicalRegion>::const_iterator it = created_region_trees.begin();
              it != created_region_trees.end(); it++)
        {
          log_leak(LEVEL_WARNING,"The region tree rooted at logical region (%d,%x,%x) was not deleted",
                    it->tree_id, it->index_space.id, it->field_space.id);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::lock_context(bool exclusive /*= true*/)
    //--------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      if (exclusive)
      {
        Event lock_event = context_lock.lock(0,true/*exclusive*/);
        lock_event.wait();
      }
      else
      {
        Event lock_event = context_lock.lock(1,false/*exclusive*/);
        lock_event.wait();
      }
#else
      context_lock.lock();
#endif
#ifdef DEBUG_HIGH_LEVEL
      lock_held = true;
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unlock_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      lock_held = false;
#endif
      context_lock.unlock();
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void RegionTreeForest::assert_locked(void)
    //--------------------------------------------------------------------------
    {
      assert(lock_held);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::assert_not_locked(void)
    //--------------------------------------------------------------------------
    {
      assert(!lock_held);
    }
#endif

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_index_path(IndexSpace parent, IndexSpace child,
                                      std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *child_node = get_node(child); 
      path.push_back(child_node->color);
      if (parent == child) 
        return true; // Early out
      IndexSpaceNode *parent_node = get_node(parent);
      while (parent_node != child_node)
      {
        if (parent_node->depth >= child_node->depth)
          return false;
        if (child_node->parent == NULL)
          return false;
        path.push_back(child_node->parent->color);
        path.push_back(child_node->parent->parent->color);
        child_node = child_node->parent->parent;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_partition_path(IndexSpace parent, IndexPartition child,
                                      std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *child_node = get_node(child);
      path.push_back(child_node->color);
      if (child_node->parent == NULL)
        return false;
      return compute_index_path(parent, child_node->parent->handle, path);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_disjoint(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->disjoint;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_overlapping(LogicalRegion h1, LogicalRegion h2)
    //--------------------------------------------------------------------------
    {
      // If the tree ids are different or the field spaces are different
      // then we are done easily
      if (h1.get_field_space() != h2.get_field_space())
        return false;
      if (h1.get_tree_id() != h2.get_tree_id())
        return false;
      std::vector<Color> path;
      if (compute_index_path(h1.get_index_space(), h2.get_index_space(), path))
        return true;
      path.clear();
      if (compute_index_path(h2.get_index_space(), h1.get_index_space(), path))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_space(Domain domain)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Create a new index space node and put it on the list
      create_node(domain, NULL/*parent*/, 0/*color*/, true/*add*/);
      created_index_trees.push_back(domain.get_index_space());
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_space(IndexSpace space, bool finalize, 
                               const std::vector<ContextID> &deletion_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *target_node = get_node(space);
      // First destroy all the logical regions trees that use this index space  
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        // Mark that this region has been destroyed
        deleted_regions.push_back((*it)->handle);
        destroy_node(*it, true/*top*/, finalize, deletion_contexts);
        // Also check to see if the handle was in the created list
        for (std::list<LogicalRegion>::iterator cit = created_region_trees.begin();
              cit != created_region_trees.end(); cit++)
        {
          if ((*cit) == ((*it)->handle))
          {
            created_region_trees.erase(cit);
            // No longer need to mark this as deleted since it was made here
            deleted_regions.pop_back();
            break;
          }
        }
      }
      target_node->logical_nodes.clear();
      // Now we delete the index space and its subtree
      deleted_index_spaces.push_back(target_node->handle);
      destroy_node(target_node, true/*top*/, finalize);
      // Also check to see if this is one of our created regions in which case
      // we need to remove it from that list
      for (std::list<IndexSpace>::iterator it = created_index_trees.begin();
            it != created_index_trees.end(); it++)
      {
        if ((*it) == space)
        {
          created_index_trees.erase(it);
          // No longer need to mark this as deleted since it was made here
          deleted_index_spaces.pop_back();
          break;
        }
      }
#ifdef DYNAMIC_TESTS
      // Filter out any dynamic tests that we might have asked for
      for (std::list<DynamicSpaceTest>::iterator it = dynamic_space_tests.begin();
            it != dynamic_space_tests.end(); /*nothing*/)
      {
        if ((it->left == space) || (it->right == space))
          it = dynamic_space_tests.erase(it);
        else
          it++;
      }
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_destroyed_regions(IndexSpace space, std::vector<LogicalRegion> &new_deletions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *target_node = get_node(space);
      // First destroy all the logical regions trees that use this index space  
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        new_deletions.push_back((*it)->handle);
      }
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint,
                                int color, const std::map<Color,Domain> &coloring, Domain color_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *parent_node = get_node(parent);
      Color part_color;
      if (color < 0)
        part_color = parent_node->generate_color();
      else
        part_color = unsigned(color);
      IndexPartNode *new_part = create_node(pid, parent_node, part_color, color_space, disjoint, true/*add*/);
#ifdef LEGION_SPY
      LegionSpy::log_index_partition(parent.id, pid, disjoint, part_color);
#endif
#ifdef DYNAMIC_TESTS
      std::vector<IndexSpaceNode*> children; 
#endif
      // Now do all of the child nodes
      for (std::map<Color,Domain>::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        Domain domain = it->second;
        domain.get_index_space(true/*create if necessary*/);
#ifdef DYNAMIC_TESTS
        IndexSpaceNode *child = 
#endif
        create_node(domain, new_part, it->first, true/*add*/);
#ifdef DYNAMIC_TESTS
        children.push_back(child);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_index_subspace(pid, domain.get_index_space().id, it->first);
#endif
      }
#ifdef DYNAMIC_TESTS
      bool notify_runtime = false;
      for (std::map<Color,IndexPartNode*>::const_iterator it = parent_node->valid_map.begin();
            it != parent_node->valid_map.end(); it++)
      {
        if (it->first == part_color)
          continue;
        // Otherwise add a disjointness test
        notify_runtime = true;
        dynamic_part_tests.push_back(DynamicPartTest(parent_node, part_color, it->first));
        DynamicPartTest &test = dynamic_part_tests.back();
        // Add the left children
        for (std::vector<IndexSpaceNode*>::const_iterator lit = children.begin();
              lit != children.end(); lit++)
        {
          test.add_child_space(true/*left*/,(*lit)->handle);
        }
        // Add the right children
        for (std::map<Color,IndexSpaceNode*>::const_iterator rit = it->second->valid_map.begin();
              rit != it->second->valid_map.end(); rit++)
        {
          test.add_child_space(false/*left*/,rit->second->handle);
        }
      }
      // Now do they dynamic tests between all the children if the partition is not disjoint
      if (!disjoint && (children.size() > 1))
      {
        notify_runtime = true; 
        for (std::vector<IndexSpaceNode*>::const_iterator it1 = children.begin();
              it1 != children.end(); it1++)
        {
          if ((*it1)->node_destroyed)
            continue;
          for (std::vector<IndexSpaceNode*>::const_iterator it2 = children.begin();
                it2 != it1; it2++)
          {
            if ((*it2)->node_destroyed)
              continue;
            dynamic_space_tests.push_back(DynamicSpaceTest(new_part, (*it1)->color, (*it1)->handle, (*it2)->color, (*it2)->handle));
          }
        }
      }
      if (notify_runtime)
        runtime->request_dynamic_tests(this);
#endif
      return part_color;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition pid, bool finalize, 
                                const std::vector<ContextID> &deletion_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *target_node = get_node(pid);
      // First destroy all of the logical region trees that use this index space
      for (std::list<PartitionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        deleted_partitions.push_back((*it)->handle);
        destroy_node(*it, true/*top*/, finalize, deletion_contexts);
      }
      target_node->logical_nodes.clear();
      // Now we delete the index partition
      deleted_index_parts.push_back(target_node->handle);
      destroy_node(target_node, true/*top*/, finalize);
#ifdef DYNAMIC_TESTS
      Color target_color = target_node->color;
      // Go through and remove any dynamic tests involving this partition
      for (std::list<DynamicPartTest>::iterator it = dynamic_part_tests.begin();
            it != dynamic_part_tests.end(); /*nothing*/)
      {
        if ((it->c1 == target_color) || (it->c2 == target_color))
          it = dynamic_part_tests.erase(it);
        else
          it++;
      }
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_destroyed_partitions(IndexPartition pid, std::vector<LogicalPartition> &new_deletions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *target_node = get_node(pid);
      for (std::list<PartitionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        new_deletions.push_back((*it)->handle);
      }
    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_index_partition(IndexSpace parent, Color color, bool can_create)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (can_create)
        assert(lock_held);
#endif
      if (!has_node(parent))
        return 0;
      IndexSpaceNode *parent_node = get_node(parent);
      if (can_create)
      {
        // No need to grab anymore locks
        if (!parent_node->has_child(color))
          return 0;
        IndexPartNode *index_node = parent_node->get_child(color);
        return index_node->handle;
      }
      else
      {
        // Need to hold the creation lock to read this
        AutoLock c_lock(creation_lock);
        if (!parent_node->has_child(color))
          return 0;
        IndexPartNode *index_node = parent_node->get_child(color);
        return index_node->handle;
      }
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_index_subspace(IndexPartition p, Color color, bool can_create)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (can_create)
        assert(lock_held);
#endif
      if (!has_node(p))
        return IndexSpace::NO_SPACE;
      IndexPartNode *parent_node = get_node(p);
      if (can_create)
      {
        if (!parent_node->has_child(color))
          return IndexSpace::NO_SPACE;
        IndexSpaceNode *index_node = parent_node->get_child(color);
        return index_node->handle;
      }
      else
      {
        // Need to hold the creation lock to read this
        AutoLock c_lock(creation_lock);
        if (!parent_node->has_child(color))
          return IndexSpace::NO_SPACE;
        IndexSpaceNode *index_node = parent_node->get_child(color);
        return index_node->handle;
      }
    }

    //--------------------------------------------------------------------------
    Domain RegionTreeForest::get_index_space_domain(IndexSpace handle, bool can_create)
    //--------------------------------------------------------------------------
    {
      if (!has_node(handle))
        return Domain::NO_DOMAIN;
      IndexSpaceNode *node = get_node(handle);
      return node->domain;
    }

    //--------------------------------------------------------------------------
    Domain RegionTreeForest::get_index_partition_color_space(IndexPartition p, bool can_create)
    //--------------------------------------------------------------------------
    {
      if (!has_node(p))
        return Domain::NO_DOMAIN;
      IndexPartNode *node = get_node(p);
      return node->color_space;
    }

    //--------------------------------------------------------------------------
    int RegionTreeForest::get_index_space_color(IndexSpace handle, bool can_create)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (can_create)
        assert(lock_held);
#endif
      if (can_create)
      {
        IndexSpaceNode *index_node = get_node(handle);
        return index_node->color;
      }
      else
      {
        AutoLock c_lock(creation_lock);
        if (!has_node(handle))
          return -1;
        IndexSpaceNode *index_node = get_node(handle);
        return index_node->color;
      }
    }

    //--------------------------------------------------------------------------
    int RegionTreeForest::get_index_partition_color(IndexPartition handle, bool can_create)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (can_create)
        assert(lock_held);
#endif
      if (can_create)
      {
        IndexPartNode *index_node = get_node(handle);
        return index_node->color;
      }
      else
      {
        AutoLock c_lock(creation_lock);
        if (!has_node(handle))
          return -1;
        IndexPartNode *index_node = get_node(handle);
        return index_node->color;
      }
    }
    
    //--------------------------------------------------------------------------
    void RegionTreeForest::create_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(field_nodes.find(space) == field_nodes.end());
#endif
      create_node(space);
      // Add this to the list of created field spaces
      created_field_spaces.insert(space);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace space, bool finalize,
                                const std::vector<ContextID> &deletion_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *target_node = get_node(space);
      // Need to delete all the regions that use this field space
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        // Mark that this region has been destroyed
        deleted_regions.push_back((*it)->handle);
        destroy_node(*it, true/*top*/, finalize, deletion_contexts);
        // Also check to see if the handle was in the created list
        for (std::list<LogicalRegion>::iterator cit = created_region_trees.begin();
              cit != created_region_trees.end(); cit++)
        {
          if ((*cit) == ((*it)->handle))
          {
            created_region_trees.erase(cit);
            // No longer need to mark this as deleted since it was made here
            deleted_regions.pop_back();
            break;
          }
        }
      }      
      deleted_field_spaces.push_back(space);
      destroy_node(target_node);
      // Check to see if it was on the list of our created field spaces
      // in which case we need to remove it
      std::set<FieldSpace>::iterator finder = created_field_spaces.find(space);
      if (finder != created_field_spaces.end())
      {
        created_field_spaces.erase(finder);
        deleted_field_spaces.pop_back();
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_destroyed_regions(FieldSpace space, std::vector<LogicalRegion> &new_deletions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *target_node = get_node(space);
      for (std::list<RegionNode*>::const_iterator it = target_node->logical_nodes.begin();
            it != target_node->logical_nodes.end(); it++)
      {
        new_deletions.push_back((*it)->handle);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::allocate_fields(FieldSpace space, const std::map<FieldID,size_t> &field_allocations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(space)->allocate_fields(field_allocations);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_fields(FieldSpace space, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(space)->free_fields(to_free);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(field_nodes.find(space) != field_nodes.end());
#endif
      return get_node(space)->has_field(fid);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_field_size(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(field_nodes.find(space) != field_nodes.end());
#endif
      return get_node(space)->get_field_size(fid);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_region(LogicalRegion handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(region_nodes.find(handle) == region_nodes.end());
#endif
      RegionNode *top_node = create_node(handle, NULL/*parent*/, true/*add*/);
      created_region_trees.push_back(handle);
      top_node->initialize_physical_context(ctx, false/*clear*/, FieldMask(FIELD_ALL_ONES), true/*top*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_region(LogicalRegion handle, bool finalize, 
                                const std::vector<ContextID> &deletion_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Mark this as one of the deleted regions
      deleted_regions.push_back(handle);
      // Check to see if the region node has been made if, it hasn't been
      // made, then we don't need to worry about deleting anything
      if (has_node(handle))
        destroy_node(get_node(handle), true/*top*/, finalize, deletion_contexts);
      for (std::list<LogicalRegion>::iterator it = created_region_trees.begin();
            it != created_region_trees.end(); it++)
      {
        if ((*it) == handle)
        {
          created_region_trees.erase(it);
          // We don't need to mark it as deleted anymore since we created it
          deleted_regions.pop_back();
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_partition(LogicalPartition handle, bool finalize, 
                                const std::vector<ContextID> &deletion_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      deleted_partitions.push_back(handle);
      // Check to see if it even exists, if it doesn't then
      // we don't need to worry about deleting it
      if (has_node(handle))
        destroy_node(get_node(handle), true/*top*/, finalize, deletion_contexts);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_region_partition(LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Check to see if has already been instantiated, if it has
      // then we can just return it, otherwise we need to make the new node
      RegionNode *parent_node = get_node(parent);
      LogicalPartition result(parent.tree_id, handle, parent.field_space);
      if (!has_node(result))
      {
        create_node(result, parent_node, true/*add*/);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_partition_subregion(LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Check to see if has already been instantiated, if it has
      // then we can just return it, otherwise we need to make the new node
      PartitionNode *parent_node = get_node(parent);
      LogicalRegion result(parent.tree_id, handle, parent.field_space);
      if (!has_node(result))
      {
        create_node(result, parent_node, true/*add*/);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_region_subcolor(LogicalRegion parent, Color c, bool can_create)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (can_create)
        assert(lock_held);
#endif
      if (can_create)
      {
        // Check to see if has already been instantiated, if it has
        // then we can just return it, otherwise we need to make the new node
        RegionNode *parent_node = get_node(parent);
        IndexPartNode *index_node = parent_node->row_source->get_child(c);
        LogicalPartition result(parent.tree_id, index_node->handle, parent.field_space);
        if (!has_node(result))
        {
          create_node(result, parent_node, true/*add*/);
        }
        return result;
      }
      else
      {
        const IndexSpace &space = parent.get_index_space();
        if (!has_node(space))
          return LogicalPartition::NO_PART;
        IndexSpaceNode *parent_node = get_node(space);
        // Need to hold the creation lock to read this
        AutoLock c_lock(creation_lock);
        if (!parent_node->has_child(c))
          return LogicalPartition::NO_PART;
        IndexPartNode *index_node = parent_node->get_child(c);
        return LogicalPartition(parent.tree_id, index_node->handle, parent.field_space);
      }
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_partition_subcolor(LogicalPartition parent, Color c, bool can_create)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (can_create)
        assert(lock_held);
#endif
      if (can_create)
      {
        // Check to see if has already been instantiated, if it has
        // then we can just return it, otherwise we need to make the new node
        PartitionNode *parent_node = get_node(parent);
        IndexSpaceNode *index_node = parent_node->row_source->get_child(c);
        LogicalRegion result(parent.tree_id, index_node->handle, parent.field_space);
        if (!has_node(result))
        {
          create_node(result, parent_node, true/*add*/);
        }
        return result;
      }
      else
      {
        const IndexPartition &part = parent.get_index_partition(); 
        if (!has_node(part))
          return LogicalRegion::NO_REGION;
        IndexPartNode *parent_node = get_node(part);
        // Need the creation lock to read the color maps
        AutoLock c_lock(creation_lock);
        if (!parent_node->has_child(c))
          return LogicalRegion::NO_REGION;
        IndexSpaceNode *index_node = parent_node->get_child(c);
        return LogicalRegion(parent.tree_id, index_node->handle, parent.field_space);
      }
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_partition_subtree(IndexPartition handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      LogicalPartition result(tid, handle, space);
      if (!has_node(result))
      {
        get_node(result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_region_subtree(IndexSpace handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      LogicalRegion result(tid, handle, space);
      if (!has_node(result))
      {
        get_node(result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_partition_parent(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      PartitionNode *node = get_node(handle);
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_logical_context(LogicalRegion handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      get_node(handle)->initialize_logical_context(ctx);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_region(RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_space = get_node(az.start.field_space);
      FieldMask user_mask = field_space->get_field_mask(az.fields);
      // Handle the special case of when there are no field allocated yet
      if (!user_mask)
        user_mask = FieldMask(FIELD_ALL_ONES);
      // Build the logical user and then do the traversal
      LogicalUser user(az.op, az.idx, user_mask, az.usage);
      // Now do the traversal
      RegionNode *start_node = get_node(az.start);
      start_node->register_logical_region(user, az);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_index_space_deletion(ContextID ctx, IndexSpace sp, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexSpaceNode *index_node = get_node(sp);
      FieldMask deletion_mask(FIELD_ALL_ONES);
      // Perform the deletion registration across all instances
      for (std::list<RegionNode*>::const_iterator it = index_node->logical_nodes.begin();
            it != index_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }
    
    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_index_part_deletion(ContextID ctx, IndexPartition part, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      IndexPartNode *index_node = get_node(part);
      FieldMask deletion_mask(FIELD_ALL_ONES);
      // Perform the deletion registration across all instances
      for (std::list<PartitionNode*>::const_iterator it = index_node->logical_nodes.begin();
            it != index_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_field_space_deletion(ContextID ctx, FieldSpace sp, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(sp);
      FieldMask deletion_mask(FIELD_ALL_ONES);
      // Perform the deletion operation across all instances
      for (std::list<RegionNode*>::const_iterator it = field_node->logical_nodes.begin();
            it != field_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_field_deletion(ContextID ctx, FieldSpace sp, const std::set<FieldID> &to_free, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(sp);
      // Get the mask for the single field
      FieldMask deletion_mask = field_node->get_field_mask(to_free);
      // Perform the deletion across all the instances
      for (std::list<RegionNode*>::const_iterator it = field_node->logical_nodes.begin();
            it != field_node->logical_nodes.end(); it++)
      {
        (*it)->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_region_deletion(ContextID ctx, LogicalRegion handle, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldMask deletion_mask(FIELD_ALL_ONES);
      get_node(handle)->register_deletion_operation(ctx, op, deletion_mask); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_partition_deletion(ContextID ctx, LogicalPartition handle, DeletionOperation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldMask deletion_mask(FIELD_ALL_ONES);
      get_node(handle)->register_deletion_operation(ctx, op, deletion_mask);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::initialize_physical_context(const RegionRequirement &req, unsigned idx, InstanceRef ref, 
                                                              UniqueID uid, ContextID ctx, Event term_event, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(req.handle_type == SINGULAR);
#endif
      
      if (!ref.is_virtual_ref())
      {
        // Initialize the physical context
        RegionNode *top_node = get_node(req.region);
        FieldSpaceNode *field_node = get_node(req.region.field_space);
        FieldMask priv_mask = field_node->get_field_mask(req.privilege_fields);
        FieldMask init_mask = field_node->get_field_mask(req.instance_fields);
        top_node->initialize_physical_context(ctx, true/*clear*/, init_mask, true/*top*/);
        // Make a physical user for this task
        PhysicalUser user(priv_mask, RegionUsage(req), term_event, term_event, idx);
        // Do different things depending on whether this a normal instance or a reduction instance
        if (ref.is_reduction_ref())
        {
          ReductionManager *clone_manager = ref.get_manager()->as_reduction_manager()->clone_manager();
          ReductionView *clone_view = create_reduction_view(clone_manager, top_node, true/*make local*/);
          clone_view->add_valid_reference();
          RegionTreeNode::PhysicalState &state = top_node->physical_states[ctx];
          state.reduction_views[clone_view] = user.field_mask;
          return clone_view->add_init_user(uid, user);
        }
        else
        {
          // Now go through and make a new InstanceManager and InstanceView for the
          // top level region and put them at the top of the tree
          InstanceManager *clone_manager = ref.get_manager()->as_instance_manager()->clone_manager(user.field_mask, field_node);
          InstanceView *clone_view = create_instance_view(clone_manager, NULL/*no parent*/, top_node, true/*make local*/);
          clone_view->add_valid_reference();
          // Update the state of the top level node 
          RegionTreeNode::PhysicalState &state = top_node->physical_states[ctx];
          state.valid_views[clone_view] = user.field_mask;
          return clone_view->add_init_user(uid, user, point);  
        }
      }
      else
      {
        // Virtual reference so no need to do anything
        return ref;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::map_region(RegionMapper &rm, LogicalRegion start_region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node((rm.req.handle_type == SINGULAR) || (rm.req.handle_type == REG_PROJECTION)
                                                            ? rm.req.region.field_space : rm.req.partition.field_space);
      FieldMask field_mask = field_node->get_field_mask(rm.req.instance_fields);
      PhysicalUser user(field_mask, RegionUsage(rm.req), rm.single_term, rm.multi_term, rm.idx);
      RegionNode *top_node = get_node(start_region);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id(), top_node, rm.ctx, true/*premap*/, rm.sanitizing, false/*closing*/, FIELD_ALL_ONES, field_mask);
#endif
      top_node->register_physical_region(user, rm);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id(), top_node, rm.ctx, false/*premap*/, rm.sanitizing, false/*closing*/, FIELD_ALL_ONES, field_mask);
#endif
    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::close_to_instance(const InstanceRef &ref, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(rm.req.region.field_space);
      FieldMask field_mask = field_node->get_field_mask(rm.req.instance_fields);
      PhysicalUser user(field_mask, RegionUsage(rm.req), rm.single_term, rm.multi_term, rm.idx);
      RegionNode *close_node = get_node(rm.req.region);
      PhysicalCloser closer(user, rm, close_node, false/*leave open*/); 
      closer.add_upper_target(ref.view->as_instance_view());
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.upper_targets.back()->logical_region == close_node);
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id(), close_node, rm.ctx, true/*premap*/, false/*sanitizing*/, true/*closing*/, FIELD_ALL_ONES, field_mask);
#endif
      close_node->issue_final_close_operation(rm, user, closer);
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.success);
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id(), close_node, rm.ctx, false/*premap*/, false/*sanitizing*/, true/*closing*/, FIELD_ALL_ONES, field_mask);
#endif
      // Now get the event for when the close is done
      return ref.view->as_instance_view()->get_valid_event(field_mask);
    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::close_to_reduction(const InstanceRef &ref, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(rm.req.region.field_space);
      FieldMask field_mask = field_node->get_field_mask(rm.req.instance_fields);
      // Make the region usage Read-Write-Exclusive to guarantee we close everything up
      PhysicalUser user(field_mask, RegionUsage(READ_WRITE, EXCLUSIVE, 0/*redop*/), rm.single_term, rm.multi_term, rm.idx);
      RegionNode *close_node = get_node(rm.req.region);
      ReductionCloser closer(user, rm, close_node, ref.view->as_reduction_view());
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.target->logical_region == close_node);
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id(), close_node, rm.ctx, true/*premap*/, false/*sanitizing*/, true/*closing*/, FIELD_ALL_ONES, field_mask);
#endif
      close_node->issue_final_reduction_operation(rm, user, closer);
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.success);
      TreeStateLogger::capture_state(runtime, &rm.req, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id(), close_node, rm.ctx, false/*premap*/, false/*sanitizing*/, true/*closing*/, FIELD_ALL_ONES, field_mask);
#endif
      return ref.view->as_reduction_view()->get_valid_event(field_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(const RegionRequirement &req,
        const std::vector<FieldID> &new_fields, ContextID ctx, bool new_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(req.handle_type == SINGULAR);
#endif
      // Compute the field mask to be used
      FieldSpaceNode *field_node = get_node(req.region.field_space);
      FieldMask invalidate_mask = field_node->get_field_mask(new_fields);
      if (!new_only)
        invalidate_mask |= field_node->get_field_mask(req.privilege_fields);
      // If no invalidate mask, then we're done
      if (!invalidate_mask)
        return;
      // Otherwise get the region node and do the invalidation
      RegionNode *top_node = get_node(req.region);
      top_node->recursive_invalidate_views(ctx, invalidate_mask, !new_only/*last use*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(LogicalRegion handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Only do this if the node actually exists
      if (has_node(handle))
      {
        RegionNode *top_node = get_node(handle);
        top_node->recursive_invalidate_views(ctx, FieldMask(FIELD_ALL_ONES), true/*last use*/);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(LogicalPartition handle, ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Only do this if the node actually exists
      if (has_node(handle))
      {
        PartitionNode *top_node = get_node(handle);
        top_node->recursive_invalidate_views(ctx, FieldMask(FIELD_ALL_ONES), true/*last use*/);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(LogicalRegion handle, ContextID ctx, 
                                        const std::vector<FieldID> &fields, bool last_use)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      if (has_node(handle))
      {
        FieldSpaceNode *field_node = get_node(handle.field_space);
        FieldMask invalidate_mask = field_node->get_field_mask(fields);
        RegionNode *top_node = get_node(handle);
        top_node->recursive_invalidate_views(ctx, invalidate_mask, last_use);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::merge_field_context(LogicalRegion handle, ContextID outer_ctx, ContextID inner_ctx,
                                                const std::vector<FieldID> &merge_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      if (merge_fields.empty())
        return;
      FieldSpaceNode *field_node = get_node(handle.field_space);
      FieldMask merge_mask = field_node->get_field_mask(merge_fields);
      RegionNode *pivot_node = get_node(handle); 
      // No need to initialize things on the way down since that's already
      // been handled by the outer context
      // Check to see if we need to fill exclusive up from this node
      if (pivot_node->parent != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(pivot_node->parent != NULL);
#endif
        pivot_node->parent->fill_exclusive_context(outer_ctx, merge_mask, pivot_node->row_source->color);
      }
      // Now we can do the merge
      pivot_node->merge_physical_context(outer_ctx, inner_ctx, merge_mask);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_forest_shape_size(const std::vector<IndexSpaceRequirement> &indexes,
                                                              const std::vector<FieldSpaceRequirement> &fields,
                                                              const std::vector<RegionRequirement> &regions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Find the sets of trees we need to send
      // Go through and mark all the top nodes we need to send this tree
      for (std::vector<IndexSpaceRequirement>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
      {
        if (it->privilege != NO_MEMORY)
        {
          IndexSpaceNode *node = get_node(it->handle);
          node->mark_node(true/*recurse*/);
        }
      }
      for (std::vector<FieldSpaceRequirement>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (it->privilege != NO_MEMORY)
        {
          FieldSpaceNode *node = get_node(it->handle);
          send_field_nodes.insert(node);
        }
      }
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (it->privilege != NO_ACCESS)
        {
          if ((it->handle_type == SINGULAR) || (it->handle_type == REG_PROJECTION))
          {
            RegionNode *node = get_node(it->region);
            node->mark_node(true/*recurse*/);
            // Also do the field spaces and the index spaces
            FieldSpaceNode *fnode = get_node(it->region.field_space);
            send_field_nodes.insert(fnode);
            IndexSpaceNode *inode = get_node(it->region.index_space);
            inode->mark_node(true/*recurse*/);
          }
          else
          {
            PartitionNode *node = get_node(it->partition);
            node->mark_node(true/*recurse*/);
            node->parent->mark_node(false/*recurse*/);
            // Also do the field spaces and the index spaces
            FieldSpaceNode *fnode = get_node(it->partition.field_space);
            send_field_nodes.insert(fnode);
            IndexPartNode *inode = get_node(it->partition.index_partition);
            inode->mark_node(true/*recurse*/);
            inode->parent->mark_node(false/*recurse*/);
          }
        }
      }
      // Now find the tops of the trees to send
      for (std::vector<IndexSpaceRequirement>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
      {
        if (it->privilege != NO_MEMORY)
        {
          IndexSpaceNode *node = get_node(it->handle);
          send_index_nodes.insert(node->find_top_marked());
        }
      }
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
        if (it->privilege != NO_ACCESS)
        {
          if ((it->handle_type == SINGULAR) || (it->handle_type == REG_PROJECTION))
          {
            RegionNode *node = get_node(it->region);
            send_logical_nodes.insert(node->find_top_marked());
            IndexSpaceNode *inode = get_node(it->region.index_space);
            send_index_nodes.insert(inode->find_top_marked());
          }
          else
          {
            PartitionNode *node = get_node(it->partition);
            send_logical_nodes.insert(node->find_top_marked());
            IndexPartNode *inode = get_node(it->partition.index_partition);
            send_index_nodes.insert(inode->find_top_marked());
          }
        }
      }

      size_t result = 3*sizeof(size_t);  // number of top nodes for each type 
      // Now we have list of unique nodes to send, so compute the sizes
      for (std::set<IndexSpaceNode*>::const_iterator it = send_index_nodes.begin();
            it != send_index_nodes.end(); it++)
      {
        result += (*it)->compute_tree_size(false/*returning*/);
      }
      for (std::set<FieldSpaceNode*>::const_iterator it = send_field_nodes.begin();
            it != send_field_nodes.end(); it++)
      {
        result += (*it)->compute_node_size();
      }
      for (std::set<RegionNode*>::const_iterator it = send_logical_nodes.begin();
            it != send_logical_nodes.end(); it++)
      {
        result += (*it)->compute_tree_size(false/*returning*/);
      }

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_forest_shape(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(send_index_nodes.size());
      for (std::set<IndexSpaceNode*>::const_iterator it = send_index_nodes.begin();
            it != send_index_nodes.end(); it++)
      {
        (*it)->serialize_tree(rez,false/*returning*/);
      }
      rez.serialize(send_field_nodes.size());
      for (std::set<FieldSpaceNode*>::const_iterator it = send_field_nodes.begin();
            it != send_field_nodes.end(); it++)
      {
        (*it)->serialize_node(rez);
      }
      rez.serialize(send_logical_nodes.size());
      for (std::set<RegionNode*>::const_iterator it = send_logical_nodes.begin();
            it != send_logical_nodes.end(); it++)
      {
        (*it)->serialize_tree(rez,false/*returning*/);
      }
      send_index_nodes.clear();
      send_field_nodes.clear();
      send_logical_nodes.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_forest_shape(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t num_index_trees, num_field_nodes, num_logical_trees;
      derez.deserialize(num_index_trees);
      for (unsigned idx = 0; idx < num_index_trees; idx++)
      {
        top_index_trees.push_back(IndexSpaceNode::deserialize_tree(derez, NULL/*parent*/, this, false/*returning*/));
      }
      derez.deserialize(num_field_nodes);
      for (unsigned idx = 0; idx < num_field_nodes; idx++)
      {
        FieldSpaceNode::deserialize_node(derez, this);
      }
      derez.deserialize(num_logical_trees);
      for (unsigned idx = 0; idx < num_logical_trees; idx++)
      {
        top_logical_trees.push_back(RegionNode::deserialize_tree(derez, NULL/*parent*/, this, false/*returning*/));
      }
    }

    //--------------------------------------------------------------------------
    FieldMask RegionTreeForest::compute_field_mask(const RegionRequirement &req, SendingMode mode, 
                                                    FieldSpaceNode *field_node) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      std::set<FieldID> packing_fields;
      switch (mode)
      {
        case PHYSICAL:
          {
            packing_fields.insert(req.instance_fields.begin(), req.instance_fields.end());
            break;
          }
        case PRIVILEGE:
          {
            packing_fields = req.privilege_fields;
            break;
          }
        case DIFF:
          {
            packing_fields = req.privilege_fields;
            for (std::vector<FieldID>::const_iterator it = req.instance_fields.begin();
                  it != req.instance_fields.end(); it++)
            {
              packing_fields.erase(*it);
            }
            break;
          }
        default:
          assert(false); // should never get here
      }
      return field_node->get_field_mask(packing_fields);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_size(const RegionRequirement &req, ContextID ctx, SendingMode mode)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      // Field mask for packing is based on the computed packing fields 
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return 0;
      size_t result = 0;
      if ((req.handle_type == SINGULAR) || (req.handle_type == REG_PROJECTION))
      {
        RegionNode *top_node = get_node(req.region);
        std::set<InstanceManager*> needed_managers;
        result += top_node->compute_state_size(ctx, packing_mask, 
                              needed_managers, unique_reductions, 
                              false/*mark invalid views*/, true/*recurse*/);
        // Now for each of the managers get the set of required views
        for (std::set<InstanceManager*>::const_iterator it = needed_managers.begin();
              it != needed_managers.end(); it++)
        {
          unique_managers.insert(*it);
          (*it)->find_views_from(req.region, unique_views, ordered_views, packing_mask);
        }
      }
      else
      {
        PartitionNode *top_node = get_node(req.partition);
        // Pack the parent state without recursing
        std::set<InstanceManager*> needed_managers;
        result += top_node->parent->compute_state_size(ctx, packing_mask, 
                                        needed_managers, unique_reductions, 
                                        false/*mark invalid views*/, false/*recurse*/);
        result += top_node->compute_state_size(ctx, packing_mask,
                                        needed_managers, unique_reductions, 
                                        false/*mark invalid views*/, true/*recurse*/);
        for (std::set<InstanceManager*>::const_iterator it = needed_managers.begin();
              it != needed_managers.end(); it++)
        {
          unique_managers.insert(*it);
          (*it)->find_views_from(top_node->parent->handle, unique_views, ordered_views, 
                                  packing_mask, top_node->row_source->color);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::post_compute_region_tree_state_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Go through all the managers and views and compute the size needed to move them
      size_t result = 0;
      result += (3*sizeof(size_t)); // number of managers, reduction managers, and number of views
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        result += (*it)->compute_send_size();
      }
      for (std::set<ReductionManager*>::const_iterator it = unique_reductions.begin();
            it != unique_reductions.end(); it++)
      {
        result += (*it)->compute_send_size();
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(unique_views.size() == ordered_views.size());
#endif
      for (std::map<InstanceView*,FieldMask>::const_iterator it = unique_views.begin();
            it != unique_views.end(); it++)
      {
        result += it->first->compute_send_size(it->second);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_pack_region_tree_state(Serializer &rez, unsigned long num_ways /*= 1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(unique_managers.size());
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        (*it)->pack_manager_send(rez, num_ways);
      }
      rez.serialize(unique_reductions.size());
      for (std::set<ReductionManager*>::const_iterator it = unique_reductions.begin();
            it != unique_reductions.end(); it++)
      {
        (*it)->pack_manager_send(rez, num_ways);
      }
      rez.serialize(unique_views.size());
      // Now do these in order!  Very important to do them in order!
      for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
            it != ordered_views.end(); it++)
      {
        std::map<InstanceView*,FieldMask>::const_iterator finder = unique_views.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != unique_views.end());
#endif
        (*it)->pack_view_send(finder->second, rez);
      }
      unique_managers.clear();
      unique_reductions.clear();
      unique_views.clear();
      ordered_views.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state(const RegionRequirement &req, ContextID ctx, 
                                                  SendingMode mode, Serializer &rez
#ifdef DEBUG_HIGH_LEVEL
                                                  , unsigned idx, const char *task_name
                                                  , unsigned uid
#endif
                                                  )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Get the field mask for what we're packing
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      // Field mask for packing is based on the privilege fields
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      if ((req.handle_type == SINGULAR) || (req.handle_type == REG_PROJECTION))
      {
        RegionNode *top_node = get_node(req.region);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, uid, top_node, ctx, true/*pack*/, true/*send*/, packing_mask, packing_mask);
#endif
        top_node->pack_physical_state(ctx, packing_mask, rez, false/*invalidate views*/, true/*recurse*/);
      }
      else
      {
        PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, uid, top_node, ctx, true/*pack*/, true/*send*/, packing_mask, packing_mask);
#endif
        top_node->parent->pack_physical_state(ctx, packing_mask, rez, false/*invalidate views*/, false/*recurse*/);
        top_node->pack_physical_state(ctx, packing_mask, rez, false/*invalidate views*/, true/*recurse*/);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_unpack_region_tree_state(Deserializer &derez, unsigned long split_factor /*= 1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t num_managers;
      derez.deserialize(num_managers);
      for (unsigned idx = 0; idx < num_managers; idx++)
      {
        InstanceManager::unpack_manager_send(this, derez, split_factor); 
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      for (unsigned idx = 0; idx < num_reductions; idx++)
      {
        ReductionManager::unpack_manager_send(this, derez, split_factor);
      }
      size_t num_views;
      derez.deserialize(num_views);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        InstanceView::unpack_view_send(this, derez);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_state(const RegionRequirement &req, ContextID ctx, SendingMode mode, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                                    , unsigned idx, const char *task_name
                                                    , unsigned uid
#endif
        )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask unpacking_mask = compute_field_mask(req, mode, field_node);
      if (!unpacking_mask)
        return;
      if ((req.handle_type == SINGULAR) || (req.handle_type == REG_PROJECTION))
      {
        RegionNode *top_node = get_node(req.region);
        top_node->initialize_physical_context(ctx, false/*clear*/, unpacking_mask, true/*top*/);
        top_node->unpack_physical_state(ctx, derez, true/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, uid, top_node, ctx, false/*pack*/, true/*send*/, unpacking_mask, unpacking_mask);
#endif
      }
      else
      {
        PartitionNode *top_node = get_node(req.partition);
        top_node->parent->initialize_physical_context(ctx, false/*clear*/, unpacking_mask, true/*top*/);
        top_node->parent->unpack_physical_state(ctx, derez, false/*recurse*/);
        top_node->unpack_physical_state(ctx, derez, true/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, uid, top_node->parent, ctx, false/*pack*/, true/*send*/, unpacking_mask, unpacking_mask);
#endif
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_reference_size(InstanceRef ref)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // For right now we're not even going to bother hooking these up to real references
      // since you shouldn't be able to remove it remotely anyway
      size_t result = 0;
      result += sizeof(ref.ready_event);
      result += sizeof(ref.required_lock);
      result += sizeof(ref.location);
      result += sizeof(ref.instance);
      result += sizeof(ref.copy);
      // Don't need to send the view since we should never need it remotely
      // It certainly can't be removed remotely
      result += sizeof(UniqueManagerID);
      result += sizeof(LogicalRegion);
      // Check to see if we need to add this manager to the set of managers to be sent
      if (ref.get_manager() != NULL)
      {
        if (ref.is_reduction_ref())
          unique_reductions.insert(ref.get_manager()->as_reduction_manager());
        else
          unique_managers.insert(ref.get_manager()->as_instance_manager());
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_reference(const InstanceRef &ref, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(!ref.is_virtual_ref());
      assert(ref.view != NULL);
#endif
      rez.serialize(ref.ready_event);
      rez.serialize(ref.required_lock);
      rez.serialize(ref.location);
      rez.serialize(ref.instance);
      rez.serialize(ref.copy);
      if (ref.get_manager() != NULL)
        rez.serialize(ref.get_manager()->get_unique_id());
      else
        rez.serialize<UniqueManagerID>(0);
      rez.serialize(ref.view->logical_region->handle);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::unpack_reference(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      Event ready_event;
      derez.deserialize(ready_event);
      Lock req_lock;
      derez.deserialize(req_lock);
      Memory location;
      derez.deserialize(location);
#ifdef DEBUG_HIGH_LEVEL
      assert(location.exists());
#endif
      PhysicalInstance inst;
      derez.deserialize(inst);
      bool copy;
      derez.deserialize(copy);
      UniqueManagerID uid;
      derez.deserialize(uid);
      LogicalRegion handle;
      derez.deserialize(handle);
      if (uid == 0)
        return InstanceRef(NULL, handle, ready_event, location, inst, copy, req_lock);
      else
      {
        PhysicalManager *manager = find_manager(uid);
#ifdef DEBUG_HIGH_LEVEL
        assert(manager != NULL);
#endif
        return InstanceRef(manager, handle, ready_event, location, inst, copy, req_lock);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_reference_size_return(InstanceRef ref)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t result = 0;
      // Only sending back things required for removing a reference
      result += sizeof(ref.ready_event);
      result += sizeof(ref.copy);
      result += sizeof(UniqueManagerID);
      result += sizeof(LogicalRegion);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_reference_return(InstanceRef ref, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(ref.ready_event);
      rez.serialize(ref.copy);
      if (ref.view != NULL)
      {
        rez.serialize(ref.view->get_manager()->get_unique_id());
        rez.serialize(ref.view->logical_region->handle);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(ref.handle != LogicalRegion::NO_REGION);
        assert(ref.manager != NULL);
#endif
        rez.serialize(ref.manager->get_unique_id());
        rez.serialize(ref.handle);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_and_remove_reference(Deserializer &derez, UniqueID uid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      Event ready_event;
      derez.deserialize(ready_event);
      bool copy;
      derez.deserialize(copy);
      UniqueManagerID mid;
      derez.deserialize(mid);
      LogicalRegion handle;
      derez.deserialize(handle);
      PhysicalView *view = find_view(InstanceKey(mid, handle));
      if (copy)
        view->remove_copy(ready_event, false/*strict*/);
      else
        view->remove_user(uid, 1/*number of references*/, false/*strict*/);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_updates_return(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Go through all our top trees and find the created partitions
      for (std::list<IndexSpaceNode*>::const_iterator it = top_index_trees.begin();
            it != top_index_trees.end(); it++)
      {
        (*it)->find_new_partitions(new_index_part_nodes);
      }

      // Then compute the size of the computed partitions, the created nodes,
      // and the handles for the deleted nodes
      size_t result = 0;
      result += sizeof(size_t); // number of new index partitions
      result += (new_index_part_nodes.size() * sizeof(IndexSpace)); // parent handles
      for (std::vector<IndexPartNode*>::const_iterator it = new_index_part_nodes.begin();
            it != new_index_part_nodes.end(); it++)
      {
        result += (*it)->compute_tree_size(true/*returning*/);
      }

      // Pack up the created nodes
      result += sizeof(size_t);
      for (std::list<IndexSpace>::const_iterator it = created_index_trees.begin();
            it != created_index_trees.end(); it++)
      {
        result += get_node(*it)->compute_tree_size(true/*returning*/);
      }
      result += sizeof(size_t);
      for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        result += get_node(*it)->compute_node_size();
      }
      result += sizeof(size_t);
      for (std::list<LogicalRegion>::const_iterator it = created_region_trees.begin();
            it != created_region_trees.end(); it++)
      {
        result += get_node(*it)->compute_tree_size(true/*returning*/);
      }

      // Pack up the Field Space nodes which have modified fields
      result += sizeof(size_t); // number of field spaces with new fields 
      for (std::map<FieldSpace,FieldSpaceNode*>::const_iterator it = field_nodes.begin();
            it != field_nodes.end(); it++)
      {
        // Make sure it isn't a created node that we already sent back
        if (created_field_spaces.find(it->first) == created_field_spaces.end())
        {
          if (it->second->has_modifications())
          {
            send_field_nodes.insert(it->second);
            result += sizeof(it->first);
            result += it->second->compute_field_return_size();
          }
        }
      }

      // Now pack up any deleted things
      result += sizeof(size_t); // num deleted index spaces
      result += (deleted_index_spaces.size() * sizeof(IndexSpace));
      result += sizeof(size_t); // num deleted index parts
      result += (deleted_index_parts.size() * sizeof(IndexPartition));
      result += sizeof(size_t); // num deleted field spaces
      result += (deleted_field_spaces.size() * sizeof(FieldSpace));
      result += sizeof(size_t); // num deleted regions
      result += (deleted_regions.size() * sizeof(LogicalRegion));
      result += sizeof(size_t); // num deleted partitions
      result += (deleted_partitions.size() * sizeof(LogicalPartition));

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_updates_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Pack up any created partitions
      rez.serialize(new_index_part_nodes.size());
      for (std::vector<IndexPartNode*>::const_iterator it = new_index_part_nodes.begin();
            it != new_index_part_nodes.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->parent != NULL);
#endif
        rez.serialize((*it)->parent->handle);
        (*it)->serialize_tree(rez,true/*returning*/);
      }

      // Pack up any created nodes
      rez.serialize(created_index_trees.size());
      for (std::list<IndexSpace>::const_iterator it = created_index_trees.begin();
            it != created_index_trees.end(); it++)
      {
        get_node(*it)->serialize_tree(rez,true/*returning*/);
      }
      rez.serialize(created_field_spaces.size());
      for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        get_node(*it)->serialize_node(rez);
      }
      rez.serialize(created_region_trees.size());
      for (std::list<LogicalRegion>::const_iterator it = created_region_trees.begin();
            it != created_region_trees.end(); it++)
      {
        get_node(*it)->serialize_tree(rez,true/*returning*/);
      }

      // Pack up any field space nodes which had modifications
      rez.serialize(send_field_nodes.size());
      for (std::set<FieldSpaceNode*>::const_iterator it = send_field_nodes.begin();
            it != send_field_nodes.end(); it++)
      {
        rez.serialize((*it)->handle);
        (*it)->serialize_field_return(rez);
      }

      // Finally send back the names of everything that has been deleted
      rez.serialize(deleted_index_spaces.size());
      for (std::list<IndexSpace>::const_iterator it = deleted_index_spaces.begin();
            it != deleted_index_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_index_parts.size());
      for (std::list<IndexPartition>::const_iterator it = deleted_index_parts.begin();
            it != deleted_index_parts.end(); it++)
      {
        rez.serialize(*it);
      } 
      rez.serialize(deleted_field_spaces.size());
      for (std::list<FieldSpace>::const_iterator it = deleted_field_spaces.begin();
            it != deleted_field_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_regions.size());
      for (std::list<LogicalRegion>::const_iterator it = deleted_regions.begin();
            it != deleted_regions.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_partitions.size());
      for (std::list<LogicalPartition>::const_iterator it = deleted_partitions.begin();
            it != deleted_partitions.end(); it++)
      {
        rez.serialize(*it);
      }
      // Now we can clear all these things since they've all been sent back
      created_index_trees.clear();
      deleted_index_spaces.clear();
      deleted_index_parts.clear();
      //created_field_spaces.clear(); // still need this for packing created state
      deleted_field_spaces.clear();
      created_region_trees.clear();
      deleted_regions.clear();
      deleted_partitions.clear();
      // Clean up our state from sending
      send_index_nodes.clear();
      send_field_nodes.clear();
      send_logical_nodes.clear();
      new_index_part_nodes.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_updates_return(Deserializer &derez,
                                const std::vector<ContextID> &enclosing_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Unpack new partitions
      size_t new_index_part_nodes;
      derez.deserialize(new_index_part_nodes);
      for (unsigned idx = 0; idx < new_index_part_nodes; idx++)
      {
        IndexSpace parent_space;
        derez.deserialize(parent_space);
        IndexSpaceNode *parent_node = get_node(parent_space);
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_node != NULL);
#endif
        IndexPartNode::deserialize_tree(derez, parent_node, this, true/*returning*/);
      }

      // Unpack created nodes
      size_t new_index_trees;
      derez.deserialize(new_index_trees);
      for (unsigned idx = 0; idx < new_index_trees; idx++)
      {
        created_index_trees.push_back(IndexSpaceNode::deserialize_tree(derez, NULL, this, true/*returning*/)->handle); 
      }
      size_t new_field_nodes;
      derez.deserialize(new_field_nodes);
      for (unsigned idx = 0; idx < new_field_nodes; idx++)
      {
        created_field_spaces.insert(FieldSpaceNode::deserialize_node(derez, this)->handle);
      }
      size_t new_logical_trees;
      derez.deserialize(new_logical_trees);
      for (unsigned idx = 0; idx < new_logical_trees; idx++)
      {
        created_region_trees.push_back(RegionNode::deserialize_tree(derez, NULL, this, true/*returning*/)->handle);
      }
      
      // Unpack field spaces with created fields
      size_t modified_field_spaces;
      derez.deserialize(modified_field_spaces);
      for (unsigned idx = 0; idx < modified_field_spaces; idx++)
      {
        FieldSpace handle;
        derez.deserialize(handle);
        get_node(handle)->deserialize_field_return(derez);
      }
      
      // Unpack everything that was deleted
      size_t num_deleted_index_nodes;
      derez.deserialize(num_deleted_index_nodes);
      for (unsigned idx = 0; idx < num_deleted_index_nodes; idx++)
      {
        IndexSpace handle;
        derez.deserialize(handle);
        destroy_index_space(handle, true/*finalize*/, enclosing_contexts);
      }
      size_t num_deleted_index_parts;
      derez.deserialize(num_deleted_index_parts);
      for (unsigned idx = 0; idx < num_deleted_index_parts; idx++)
      {
        IndexPartition handle;
        derez.deserialize(handle);
        destroy_index_partition(handle, true/*finalize*/, enclosing_contexts);
      }
      size_t num_deleted_field_nodes;
      derez.deserialize(num_deleted_field_nodes);
      for (unsigned idx = 0; idx < num_deleted_field_nodes; idx++)
      {
        FieldSpace handle;
        derez.deserialize(handle);
        destroy_field_space(handle, true/*finalize*/, enclosing_contexts);
      }
      size_t num_deleted_regions;
      derez.deserialize(num_deleted_regions);
      for (unsigned idx = 0; idx < num_deleted_regions; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        destroy_region(handle, true/*finalize*/, enclosing_contexts);
      }
      size_t num_deleted_partitions;
      derez.deserialize(num_deleted_partitions);
      for (unsigned idx = 0; idx < num_deleted_partitions; idx++)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        destroy_partition(handle, true/*finalize*/, enclosing_contexts);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
                                                              ContextID ctx, bool overwrite, SendingMode mode
#ifdef DEBUG_HIGH_LEVEL
                                                              , const char *task_name
                                                              , unsigned uid
#endif
                                                              )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return 0;
      size_t result = 0;
      if (overwrite)
      {
#ifdef DEBUG_HIGH_LEVEl
        assert(req.handle_type == SINGULAR);
#endif
        // Pack the entire state of the tree
        RegionNode *top_node = get_node(req.region);
        std::set<InstanceManager*> needed_managers;
        result += top_node->compute_state_size(ctx, packing_mask,
                              needed_managers, unique_reductions, 
                              true/*mark invalide views*/, true/*recurse*/);
        for (std::set<InstanceManager*>::const_iterator it = needed_managers.begin();
              it != needed_managers.end(); it++)
        {
          unique_managers.insert(*it);
          (*it)->find_views_from(req.region, unique_views, ordered_views, packing_mask);
        }
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, idx, task_name, uid, top_node, ctx, true/*pack*/, false/*send*/, packing_mask, packing_mask);
#endif
      }
      else
      {
        if (diff_region_maps.find(idx) == diff_region_maps.end())
          diff_region_maps[idx] = std::vector<RegionNode*>();
        if (diff_part_maps.find(idx) == diff_part_maps.end())
          diff_part_maps[idx] = std::vector<PartitionNode*>();
        std::vector<RegionNode*> &diff_regions = diff_region_maps[idx];
        std::vector<PartitionNode*> &diff_partitions = diff_part_maps[idx];
        
        if ((req.handle_type == SINGULAR) || (req.handle_type == REG_PROJECTION))
        {
          RegionNode *top_node = get_node(req.region);
#ifdef DEBUG_HIGH_LEVEL
          TreeStateLogger::capture_state(runtime, idx, task_name, uid, top_node, ctx, true/*pack*/, false/*send*/, packing_mask, packing_mask);
#endif
          std::set<InstanceManager*> needed_managers;
          result += top_node->compute_diff_state_size(ctx, packing_mask,
                          needed_managers, unique_reductions, diff_regions, diff_partitions, 
                          true/*invalidate views*/, true/*recurse*/);
          for (std::set<InstanceManager*>::const_iterator it = needed_managers.begin();
                it != needed_managers.end(); it++)
          {
            unique_managers.insert(*it);
            (*it)->find_views_from(req.region, unique_views, ordered_views, packing_mask);
          }
        }
        else
        {
          PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
          TreeStateLogger::capture_state(runtime, idx, task_name, uid, top_node, ctx, true/*pack*/, false/*return*/, packing_mask, packing_mask);
#endif
          std::set<InstanceManager*> needed_managers;
          result += top_node->compute_diff_state_size(ctx, packing_mask,
                          needed_managers, unique_reductions, diff_regions, diff_partitions, 
                          true/*invalidate views*/, true/*recurse*/);
          for (std::set<InstanceManager*>::const_iterator it = needed_managers.begin();
                it != needed_managers.end(); it++)
          {
            unique_managers.insert(*it);
            (*it)->find_views_from(top_node->parent->handle, unique_views, ordered_views, packing_mask);
          }
          // Also need to invalidate the valid views of the parent
#ifdef DEBUG_HIGH_LEVEL
          assert(top_node->parent != NULL);
#endif
          top_node->parent->mark_invalid_instance_views(ctx, packing_mask, false/*recurse*/);
        }
        result += 2*sizeof(size_t); // number of regions and partitions
        result += (diff_regions.size() * sizeof(LogicalRegion));
        result += (diff_partitions.size() * sizeof(LogicalPartition));
      }
      // Update the vector indicating which view to overwrite
      {
        unsigned idx = overwrite_views.size();
        overwrite_views.resize(ordered_views.size());
        for (/*nothing*/; idx < overwrite_views.size(); idx++)
          overwrite_views[idx] = overwrite;
      } 
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::post_partition_state_return(const RegionRequirement &req, ContextID ctx, SendingMode mode)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(req.handle_type == PART_PROJECTION);
      assert(IS_WRITE(req)); // should only need to do this for write requirements
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
      assert(top_node->parent != NULL); 
#endif
      // Mark all the nodes in the parent invalid
      top_node->parent->mark_invalid_instance_views(ctx, packing_mask, false/*recurse*/);
      // Now do the rest of the tree
      top_node->mark_invalid_instance_views(ctx, packing_mask, true/*recurse*/);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::post_compute_region_tree_state_return(bool created_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
      assert(unique_views.size() == ordered_views.size());
#endif
      // First filter out all the instances that are remote since they
      // already exist on the parent node.
      {
        std::vector<InstanceManager*> to_delete;
        for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
              it != unique_managers.end(); it++)
        {
          if ((*it)->remote)
            to_delete.push_back(*it);
        }
        for (std::vector<InstanceManager*>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          unique_managers.erase(*it);
        }
      }
      // This is the set of managers which can have their remote fractions sent back.  Either
      // they are in the set of unique managers being sent back or they're remote.  No matter
      // what they must NOT have any valid views here to send them back
      for (std::map<UniqueManagerID,InstanceManager*>::const_iterator it = managers.begin();
            it != managers.end(); it++)
      {
        if (((it->second->is_remote() && !it->second->is_returned()) 
              || (unique_managers.find(it->second) != unique_managers.end()))
            && (it->second->is_valid_free()))
        {
          returning_managers.push_back(it->second);
          it->second->find_user_returns(returning_views);
        }
      }
      // Note we compute the reduction views returning before filtering.  This is because
      // ReductionViews don't have epochs and therefore any created users must be sent back
      for (std::map<UniqueManagerID,ReductionManager*>::const_iterator it = reduc_managers.begin();
            it != reduc_managers.end(); it++)
      {
        if ((it->second->is_remote() && !it->second->is_returned())
              || (unique_reductions.find(it->second) != unique_reductions.end()))
        {
          if (it->second->is_valid_free())
            returning_reductions.push_back(it->second);
          it->second->find_user_returns(returning_reduc_views); 
        }
      }
      // Filter out all the reduction instances that are remote too
      {
        std::vector<ReductionManager*> to_delete;
        for (std::set<ReductionManager*>::const_iterator it = unique_reductions.begin();
              it != unique_reductions.end(); it++)
        {
          if ((*it)->remote)
            to_delete.push_back(*it);
        }
        for (std::vector<ReductionManager*>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          unique_reductions.erase(*it);
        }
      }
      // Now we can actually compute the size of the things being returned
      size_t result = 0; 
      // First compute the size of the created managers going back
      result += sizeof(size_t); // number of created instances
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        result += (*it)->compute_return_size();  
      }
      // Number of created reduction managers going back
      result += sizeof(size_t);
      for (std::set<ReductionManager*>::const_iterator it = unique_reductions.begin();
            it != unique_reductions.end(); it++)
      {
        result += (*it)->compute_return_size();
      }
#ifdef DEBUG_HIGH_LEVEL
        assert(ordered_views.size() == unique_views.size());
        assert(ordered_views.size() == overwrite_views.size());
#endif
      // There is a nice property here if we know all the state is created state.
      // Since created state only gets passed back at the end of a task there is no need to
      // pass back any of the state/users in the InstanceViews since they all
      // have to have completed prior task which is returning the state completing.
      if (created_only)
      {
        result += sizeof(size_t); // number of non-local views 
        std::vector<InstanceView*> actual_views;
        for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
              it != ordered_views.end(); it++)
        {
          if ((*it)->local_view)
          {
            actual_views.push_back(*it);
            result += (*it)->compute_simple_return();
          }
        }
        ordered_views = actual_views; // hopefully just a copy of STL meta-data
      }
      else
      {
        result += sizeof(size_t); // number of unique views
        // Now pack up the instance views that need to be send back for the updated state
        unsigned idx = 0;
        for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
              it != ordered_views.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(unique_views.find(*it) != unique_views.end());
#endif
          result += (*it)->compute_return_state_size(unique_views[*it], overwrite_views[idx++],
                                                     escaped_users, escaped_copies);
        }
        
        // Now we do the parts that are going to be send back in the end_pack_region_tree_state_return
        result += sizeof(size_t); // number of returning views
        for (std::vector<InstanceView*>::const_iterator it = returning_views.begin();
              it != returning_views.end(); it++)
        {
          std::map<InstanceView*,FieldMask>::const_iterator finder = unique_views.find(*it);
          result += (*it)->compute_return_users_size(escaped_users, escaped_copies,
                                        (finder != unique_views.end()), finder->second);
        }
      }
      // No matter what we have to send back the returning reduction views
      result += sizeof(size_t); // number of returning reduction views
      for (std::vector<ReductionView*>::const_iterator it = returning_reduc_views.begin();
            it != returning_reduc_views.end(); it++)
      {
        result += (*it)->compute_return_size(escaped_users, escaped_copies);
      }

      result += sizeof(size_t); // number of returning managers
      result += (returning_managers.size() * (sizeof(UniqueManagerID) + sizeof(InstFrac)));
      result += sizeof(size_t); // number of returning reduction managers
      result += (returning_reductions.size() * (sizeof(UniqueManagerID) + sizeof(InstFrac)));

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_pack_region_tree_state_return(Serializer &rez, bool created_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(unique_managers.size());
      for (std::set<InstanceManager*>::const_iterator it = unique_managers.begin();
            it != unique_managers.end(); it++)
      {
        (*it)->pack_manager_return(rez);
      }
      unique_managers.clear();

      rez.serialize(unique_reductions.size());
      for (std::set<ReductionManager*>::const_iterator it = unique_reductions.begin();
            it != unique_reductions.end(); it++)
      {
        (*it)->pack_manager_return(rez);
      }
      unique_reductions.clear();

      if (created_only)
      {
        rez.serialize(ordered_views.size());
        for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
              it != ordered_views.end(); it++)
        {
          (*it)->pack_simple_return(rez); 
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(ordered_views.size() == unique_views.size());
        assert(ordered_views.size() == overwrite_views.size());
#endif
        rez.serialize(unique_views.size());
        unsigned idx = 0;
        for (std::vector<InstanceView*>::const_iterator it = ordered_views.begin();
              it != ordered_views.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(unique_views.find(*it) != unique_views.end());
#endif
          (*it)->pack_return_state(unique_views[*it], overwrite_views[idx++], rez);
        }
      }
      unique_views.clear();
      ordered_views.clear();
      overwrite_views.clear();

      rez.serialize(returning_reduc_views.size());
      for (std::vector<ReductionView*>::const_iterator it = returning_reduc_views.begin();
            it != returning_reduc_views.end(); it++)
      {
        (*it)->pack_view_return(rez);
      }
      returning_reduc_views.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_region_tree_state_return(const RegionRequirement &req, unsigned idx, 
            ContextID ctx, bool overwrite, SendingMode mode, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      if (overwrite)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(req.handle_type == SINGULAR);
#endif
        // Pack the entire state of the tree
        RegionNode *top_node = get_node(req.region);
        // In the process of traversing invalidate any views which are no longer valid for any fields
        // so we can know which physical instances no longer have any valid views and can therefore
        // be sent back to their owner to maybe be garbage collected.
        top_node->pack_physical_state(ctx, packing_mask, rez, true/*invalidate_views*/, true/*recurse*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(diff_region_maps.find(idx) != diff_region_maps.end());
        assert(diff_part_maps.find(idx) != diff_part_maps.end());
#endif
        std::vector<RegionNode*> &diff_regions = diff_region_maps[idx];
        std::vector<PartitionNode*> &diff_partitions = diff_part_maps[idx];
        rez.serialize(diff_regions.size());
        for (std::vector<RegionNode*>::const_iterator it = diff_regions.begin();
              it != diff_regions.end(); it++)
        {
          rez.serialize((*it)->handle);
          (*it)->pack_diff_state(ctx, packing_mask, rez);
        }
        rez.serialize(diff_partitions.size());
        for (std::vector<PartitionNode*>::const_iterator it = diff_partitions.begin();
              it != diff_partitions.end(); it++)
        {
          rez.serialize((*it)->handle);
          (*it)->pack_diff_state(ctx, packing_mask, rez);
        }
        diff_regions.clear();
        diff_partitions.clear();
        // Invalidate any parent views of a partition node
        if (req.handle_type == PART_PROJECTION)
        {
          PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
          assert(top_node->parent != NULL);
#endif
          std::map<ContextID,RegionTreeNode::PhysicalState>::iterator finder = 
            top_node->parent->physical_states.find(ctx);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != top_node->parent->physical_states.end());
#endif
          top_node->parent->invalidate_instance_views(finder->second, packing_mask, false/*clean*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::post_partition_pack_return(const RegionRequirement &req, ContextID ctx, SendingMode mode)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(req.handle_type == PART_PROJECTION);
      assert(IS_WRITE(req)); // should only need to do this for write requirements
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask packing_mask = compute_field_mask(req, mode, field_node);
      if (!packing_mask)
        return;
      PartitionNode *top_node = get_node(req.partition);
#ifdef DEBUG_HIGH_LEVEL
      assert(top_node->parent != NULL); 
#endif
      // first invalidate the parent views
      top_node->parent->mark_invalid_instance_views(ctx, packing_mask, false/*clean*/);
      // Now recursively do the rest
      top_node->recursive_invalidate_views(ctx, packing_mask, false/*last use*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::end_pack_region_tree_state_return(Serializer &rez, bool created_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      if (!created_only)
      {
        // Views first so we can't accidentally reclaim something prematurely
        rez.serialize(returning_views.size());
        for (std::vector<InstanceView*>::const_iterator it = returning_views.begin();
              it != returning_views.end(); it++)
        {
          (*it)->pack_return_users(rez);
        }
      }
      returning_views.clear();
      rez.serialize(returning_managers.size());
      for (std::vector<InstanceManager*>::const_iterator it = returning_managers.begin();
            it != returning_managers.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->is_valid_free());
#endif
        rez.serialize((*it)->unique_id);
        (*it)->pack_remote_fraction(rez);
      }
      returning_managers.clear();
      rez.serialize(returning_reductions.size());
      for (std::vector<ReductionManager*>::const_iterator it = returning_reductions.begin();
            it != returning_reductions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->is_valid_free());
#endif
        rez.serialize((*it)->unique_id);
        (*it)->pack_remote_fraction(rez);
      }
      returning_reductions.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_unpack_region_tree_state_return(Deserializer &derez, bool created_only)
    //--------------------------------------------------------------------------
    {
      // First unpack all the new InstanceManagers that came back
      size_t num_new_managers;
      derez.deserialize(num_new_managers);
      for (unsigned idx = 0; idx < num_new_managers; idx++)
      {
        InstanceManager::unpack_manager_return(this, derez);
      }
      // Unpack all the new ReductionManagers that came back 
      size_t num_new_reductions;
      derez.deserialize(num_new_reductions);
      for (unsigned idx = 0; idx < num_new_reductions; idx++)
      {
        ReductionManager::unpack_manager_return(this, derez);
      }
      // Now unpack all the InstanceView objects that are returning
      size_t num_returning_views;
      derez.deserialize(num_returning_views);
      if (created_only)
      {
        for (unsigned idx = 0; idx < num_returning_views; idx++)
        {
          InstanceView::unpack_simple_return(this, derez);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < num_returning_views; idx++)
        {
          InstanceView::unpack_return_state(this, derez); 
        }
      }
      size_t num_returning_reduc_views;
      derez.deserialize(num_returning_reduc_views);
      for (unsigned idx = 0; idx < num_returning_reduc_views; idx++)
      {
        ReductionView::unpack_view_return(this, derez);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_region_tree_state_return(const RegionRequirement &req, ContextID ctx,
                                                            bool overwrite, SendingMode mode, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                                            , unsigned ridx, const char *task_name, unsigned uid
#endif
                                                            )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(req.parent.field_space);
      FieldMask unpacking_mask = compute_field_mask(req, mode, field_node);
      if (!unpacking_mask)
        return;
      if (overwrite)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(req.handle_type == SINGULAR);
#endif
        // Re-initialize the state and then unpack the state
        RegionNode *top_node = get_node(req.region);
        top_node->initialize_physical_context(ctx, true/*clear*/, unpacking_mask, false/*top is already set*/);
        top_node->unpack_physical_state(ctx, derez, true/*recurse*/); 
#ifdef DEBUG_HIGH_LEVEL
        TreeStateLogger::capture_state(runtime, ridx, task_name, uid, top_node, ctx, false/*pack*/, false/*send*/, unpacking_mask, unpacking_mask);
#endif
        // We also need to update the field states of the parent
        // partition so that it knows that this region is open
        if (top_node->parent != NULL)
        {
          std::map<ContextID,RegionTreeNode::PhysicalState>::iterator finder = top_node->parent->physical_states.find(ctx);
          if (finder != top_node->parent->physical_states.end())
          {
            RegionTreeNode::FieldState new_state(GenericUser(unpacking_mask, RegionUsage(req)), unpacking_mask, top_node->row_source->color);
            top_node->parent->upgrade_new_field_state(finder->second, new_state, true/*add state*/);
          }
        }
      }
      else
      {
        size_t num_diff_regions;
        derez.deserialize(num_diff_regions);
        for (unsigned idx = 0; idx < num_diff_regions; idx++)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          get_node(handle)->unpack_diff_state(ctx, derez);
        }
        size_t num_diff_partitions;
        derez.deserialize(num_diff_partitions);
        for (unsigned idx = 0; idx < num_diff_partitions; idx++)
        {
          LogicalPartition handle;
          derez.deserialize(handle);
          get_node(handle)->unpack_diff_state(ctx, derez);
        }
#ifdef DEBUG_HIGH_LEVEL
        if ((req.handle_type == SINGULAR) || (req.handle_type == REG_PROJECTION))
        {
          RegionNode *top_node = get_node(req.region);
          TreeStateLogger::capture_state(runtime, ridx, task_name, uid, top_node, ctx, false/*pack*/, false/*send*/, unpacking_mask, unpacking_mask);
        }
        else
        {
          PartitionNode *top_node = get_node(req.partition);
          TreeStateLogger::capture_state(runtime, ridx, task_name, uid, top_node, ctx, false/*pack*/, false/*send*/, unpacking_mask, unpacking_mask);
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::end_unpack_region_tree_state_return(Deserializer &derez, bool created_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // First unpack the views
      if (!created_only)
      {
        size_t num_returning_views;
        derez.deserialize(num_returning_views);
        for (unsigned idx = 0; idx < num_returning_views; idx++)
        {
          InstanceView::unpack_return_users(this, derez); 
        }
      }
      size_t num_returning_managers;
      derez.deserialize(num_returning_managers);
      for (unsigned idx = 0; idx < num_returning_managers; idx++)
      {
        UniqueManagerID mid;
        derez.deserialize(mid);
#ifdef DEBUG_HIGH_LEVEL
        assert(managers.find(mid) != managers.end());
#endif
        managers[mid]->unpack_remote_fraction(derez);
      }
      size_t num_returning_reductions;
      derez.deserialize(num_returning_reductions);
      for (unsigned idx = 0; idx < num_returning_reductions; idx++)
      {
        UniqueManagerID mid;
        derez.deserialize(mid);
#ifdef DEBUG_HIGH_LEVEL
        assert(reduc_managers.find(mid) != reduc_managers.end());
#endif
        reduc_managers[mid]->unpack_remote_fraction(derez);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_created_field_state_return(LogicalRegion handle,
              const std::vector<FieldID> &packing_fields, ContextID ctx
#ifdef DEBUG_HIGH_LEVEL
              , const char *task_name, unsigned uid
#endif
              )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(handle.field_space);
      FieldMask packing_mask = field_node->get_field_mask(packing_fields);
      size_t result = 0;
      result += sizeof(packing_mask);
      if (!packing_mask)
        return result;
      RegionNode *top_node = get_node(handle);
      std::set<InstanceManager*> needed_managers;
      result += top_node->compute_state_size(ctx, packing_mask,
                            needed_managers, unique_reductions, 
                            true/*mark invalid views*/, true/*recursive*/);
      for (std::set<InstanceManager*>::const_iterator it = needed_managers.begin();
            it != needed_managers.end(); it++)
      {
        unique_managers.insert(*it);
        (*it)->find_views_from(handle, unique_views, ordered_views, packing_mask);
      }
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, handle, task_name, uid, top_node, ctx, true/*pack*/, 0/*shift*/, packing_mask, packing_mask);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_created_field_state_return(LogicalRegion handle,
      const std::vector<FieldID> &packing_fields, ContextID ctx, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(handle.field_space);
      FieldMask packing_mask = field_node->get_field_mask(packing_fields);
      rez.serialize(packing_mask);
      if (!packing_mask)
        return;
      RegionNode *top_node = get_node(handle);
      top_node->pack_physical_state(ctx, packing_mask, rez, true/*invalidate views*/, true/*recurse*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_created_field_state_return(LogicalRegion handle,
                      ContextID ctx, Deserializer &derez 
#ifdef DEBUG_HIGH_LEVEL
                      , const char *task_name, unsigned uid
#endif
                      )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(handle.field_space);
      // Get the amount to shift the field mask by
      unsigned shift = field_node->get_shift_amount();
      FieldMask unpacking_mask;
      derez.deserialize(unpacking_mask);
      if (!unpacking_mask)
        return;
      unpacking_mask.shift_left(shift);
      // Re-initialize the state, note we have to go up the tree if there is no parent
      RegionNode *top_node = get_node(handle);
      top_node->initialize_physical_context(ctx, false/*clear*/, unpacking_mask, true/*top for these fields*/);
      // If this is not the top of the context, we need to fill in the rest of the
      // context marking that this is open in EXCLUSIVE mode
      if (top_node->parent != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(top_node->parent != NULL);
#endif
        top_node->parent->fill_exclusive_context(ctx, unpacking_mask, top_node->row_source->color);
      }
      // Finally we can do the unpack operation
      top_node->unpack_physical_state(ctx, derez, true/*recurse*/, shift);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, handle, task_name, uid, top_node, ctx, false/*pack*/, shift, unpacking_mask, unpacking_mask);
#endif
    }
    
    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_created_state_return(LogicalRegion handle, ContextID ctx
#ifdef DEBUG_HIGH_LEVEL
                                                          , const char *task_name
                                                          , unsigned uid
#endif
                                                          )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(handle.field_space);
      FieldMask packing_mask = field_node->get_field_mask();
      size_t result = 0;
      result += sizeof(packing_mask);
      if (!packing_mask)
        return result;
      // Check to see if this field space node has created nodes
      // If it does we have to send them back in a separate way since they
      // may have to be shifted when they get returned
      result += sizeof(bool); // whether it needs two unpacks
      if (!field_node->created_fields.empty())
      {
        std::vector<FieldID> new_fields(field_node->created_fields.begin(),
                                        field_node->created_fields.end());
        result += compute_created_field_state_return(handle, new_fields, ctx
#ifdef DEBUG_HIGH_LEVEL
                                                      , task_name, uid
#endif
                                                    );
        packing_mask -= field_node->get_created_field_mask();
        result += sizeof(packing_mask); // need to send the second mask
      }
      // Now pack the remaining fields
      if (!packing_mask)
        return result;
      RegionNode *top_node = get_node(handle);
      std::set<InstanceManager*> needed_managers;
      result += top_node->compute_state_size(ctx, packing_mask,
                            needed_managers, unique_reductions, 
                            true/*mark invalid views*/, true/*recursive*/);
      for (std::set<InstanceManager*>::const_iterator it = needed_managers.begin();
            it != needed_managers.end(); it++)
      {
        unique_managers.insert(*it);
        (*it)->find_views_from(handle, unique_views, ordered_views, packing_mask);
      }
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, handle, task_name, uid, top_node, ctx, true/*pack*/, 0/*shift*/, packing_mask, packing_mask);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_created_state_return(LogicalRegion handle, ContextID ctx, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldSpaceNode *field_node = get_node(handle.field_space);
      FieldMask packing_mask = field_node->get_field_mask();
      rez.serialize(packing_mask);
      if (!packing_mask)
        return;
      if (!field_node->created_fields.empty())
      {
        rez.serialize(true);
        std::vector<FieldID> new_fields(field_node->created_fields.begin(),
                                        field_node->created_fields.end());
        pack_created_field_state_return(handle, new_fields, ctx, rez);
        packing_mask -= field_node->get_created_field_mask();
        rez.serialize(packing_mask);
      }
      else
        rez.serialize(false);
      if (!packing_mask)
        return;
      RegionNode *top_node = get_node(handle);
      top_node->pack_physical_state(ctx, packing_mask, rez, true/*invalidate views*/, true/*recurse*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_created_state_return(LogicalRegion handle, ContextID ctx, Deserializer &derez
#ifdef DEBUG_HIGH_LEVEL
                                                      , const char *task_name
                                                      , unsigned uid
#endif
                                                      )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      FieldMask unpacking_mask;
      derez.deserialize(unpacking_mask);
      if (!unpacking_mask)
        return;
      // Initialize the state before unpacking anything
      RegionNode *top_node = get_node(handle);
      top_node->initialize_physical_context(ctx, false/*clear*/, unpacking_mask, true/*top*/);
      bool has_created_fields;
      derez.deserialize(has_created_fields);
      if (has_created_fields)
      {
        unpack_created_field_state_return(handle, ctx, derez
#ifdef DEBUG_HIGH_LEVEL
                                          , task_name, uid
#endif
                                          );
        // Unpack the mask for the remaining fields
        derez.deserialize(unpacking_mask);
      }
      if (!unpacking_mask)
        return;
      top_node->unpack_physical_state(ctx, derez, true/*recurse*/);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, handle, task_name, uid, top_node, ctx, false/*pack*/, 0/*shift*/, unpacking_mask, unpacking_mask);
#endif
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::compute_leaked_return_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      size_t result = 0;
      result += sizeof(size_t); // number of escaped users 
      result += (escaped_users.size() * (sizeof(EscapedUser) + sizeof(unsigned)));
      result += sizeof(size_t); // number of escaped copy users
      result += (escaped_copies.size() * sizeof(EscapedCopy));
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::pack_leaked_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      rez.serialize(escaped_users.size());
      for (std::map<EscapedUser,unsigned>::const_iterator it = escaped_users.begin();
            it != escaped_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(escaped_copies.size());
      for (std::set<EscapedCopy>::const_iterator it = escaped_copies.begin();
            it != escaped_copies.end(); it++)
      {
        rez.serialize(*it);
      }
      escaped_users.clear();
      escaped_copies.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unpack_leaked_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lock_held);
#endif
      // Note in some case the leaked references were remove by
      // the user who copied them back earlier and removed them
      // explicitly.  In this case we ignore any references which
      // may try to be pulled twice.
      size_t num_escaped_users;
      derez.deserialize(num_escaped_users);
      for (unsigned idx = 0; idx < num_escaped_users; idx++)
      {
        EscapedUser user;
        derez.deserialize(user);
        unsigned references;
        derez.deserialize(references);
        PhysicalView *view = find_view(user.view_key);
        view->remove_user(user.user, references, false/*strict*/);
      }
      size_t num_escaped_copies;
      derez.deserialize(num_escaped_copies);
      for (unsigned idx = 0; idx < num_escaped_copies; idx++)
      {
        EscapedCopy copy;
        derez.deserialize(copy);
        PhysicalView *view = find_view(copy.view_key);
        view->remove_copy(copy.copy_event, false/*strict*/);
      }
    }

#ifdef DYNAMIC_TESTS
    //--------------------------------------------------------------------------
    bool RegionTreeForest::fix_dynamic_test_set(void)
    //--------------------------------------------------------------------------
    {
      ghost_space_tests.insert(ghost_space_tests.end(),
          dynamic_space_tests.begin(),dynamic_space_tests.end());
      ghost_part_tests.insert(ghost_part_tests.end(),
          dynamic_part_tests.begin(),dynamic_part_tests.end());
      dynamic_space_tests.clear();
      dynamic_part_tests.clear();
      return (!ghost_space_tests.empty() || !ghost_part_tests.empty());
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_dynamic_tests(void)
    //--------------------------------------------------------------------------
    {
      for (std::list<DynamicSpaceTest>::iterator it = ghost_space_tests.begin();
            it != ghost_space_tests.end(); /*nothing*/)
      {
        if (it->perform_test())
          it++;
        else
          it = ghost_space_tests.erase(it);
      }
      for (std::list<DynamicPartTest>::iterator it = ghost_part_tests.begin();
            it != ghost_part_tests.end(); /*nothing*/)
      {
        if (it->perform_test())
          it++;
        else
          it = ghost_part_tests.erase(it);
      }
    }
    
    //--------------------------------------------------------------------------
    void RegionTreeForest::publish_dynamic_test_results(void)
    //--------------------------------------------------------------------------
    {
      for (std::list<DynamicSpaceTest>::const_iterator it = ghost_space_tests.begin();
            it != ghost_space_tests.end(); it++)
      {
        it->publish_test();
      }
      for (std::list<DynamicPartTest>::const_iterator it = ghost_part_tests.begin();
            it != ghost_part_tests.end(); it++)
      {
        it->publish_test();
      }
      ghost_space_tests.clear();
      ghost_part_tests.clear();
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::DynamicSpaceTest::DynamicSpaceTest(IndexPartNode *par,
        Color one, IndexSpace l, Color two, IndexSpace r)
      : parent(par), c1(one), c2(two), left(l), right(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::DynamicSpaceTest::perform_test(void)
    //--------------------------------------------------------------------------
    {
      const LowLevel::ElementMask &left_mask = left.get_valid_mask();
      const LowLevel::ElementMask &right_mask = right.get_valid_mask();
      LowLevel::ElementMask::OverlapResult result = 
        left_mask.overlaps_with(right_mask);
      return (result == LowLevel::ElementMask::OVERLAP_NO);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::DynamicSpaceTest::publish_test(void) const
    //--------------------------------------------------------------------------
    {
      parent->add_disjoint(c1,c2);
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::DynamicPartTest::DynamicPartTest(IndexSpaceNode *par,
        Color one, Color two)
      : parent(par), c1(one), c2(two)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::DynamicPartTest::add_child_space(bool l, IndexSpace space) 
    //--------------------------------------------------------------------------
    {
      if (l)
        left.push_back(space);
      else
        right.push_back(space);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::DynamicPartTest::perform_test(void)
    //--------------------------------------------------------------------------
    {
      // TODO: A Better way to do this is to bitwise-union everything on
      // the left and the right, and then do a intersection between left
      // and right to test for non-empty.
      for (std::vector<IndexSpace>::const_iterator lit = left.begin();
            lit != left.end(); lit++)
      {
        const LowLevel::ElementMask &left_mask = lit->get_valid_mask();
        for (std::vector<IndexSpace>::const_iterator rit = right.begin();
              rit != right.end(); rit++)
        {
          const LowLevel::ElementMask &right_mask = rit->get_valid_mask();
          LowLevel::ElementMask::OverlapResult result = 
            left_mask.overlaps_with(right_mask);
          // If it's anything other than overlap-no, then we don't know
          // that it is disjoint
          if (result != LowLevel::ElementMask::OVERLAP_NO)
            return false;
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::DynamicPartTest::publish_test(void) const
    //--------------------------------------------------------------------------
    {
      parent->add_disjoint(c1,c2);
    }
#endif

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(Domain d, IndexPartNode *parent,
                                        Color c, bool add)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *result = new IndexSpaceNode(d, parent, c, add, this);
      IndexSpace sp = d.get_index_space();
      AutoLock c_lock(creation_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      assert(index_nodes.find(sp) == index_nodes.end());
#endif
      index_nodes[sp] = result;
      if (parent != NULL)
        parent->add_child(sp, result);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, IndexSpaceNode *parent,
                                        Color c, Domain color_space, bool dis, bool add)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *result = new IndexPartNode(p, parent, c, color_space, dis, add, this);
      AutoLock c_lock(creation_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      assert(index_parts.find(p) == index_parts.end());
#endif
      index_parts[p] = result;
      if (parent != NULL)
        parent->add_child(p, result);
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace sp)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *result = new FieldSpaceNode(sp, this);
      AutoLock c_lock(creation_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      assert(field_nodes.find(sp) == field_nodes.end());
#endif
      field_nodes[sp] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::create_node(LogicalRegion r, PartitionNode *par, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!has_node(r));
      if (par != NULL)
      {
        assert(r.field_space == par->handle.field_space);
        assert(r.tree_id == par->handle.tree_id);
      }
#endif
      IndexSpaceNode *row_src = get_node(r.index_space);
      FieldSpaceNode *col_src = NULL;
      // Should only have a column source if we're the top of the tree
      if (par == NULL)
        col_src = get_node(r.field_space);

      RegionNode *result = new RegionNode(r, par, row_src, col_src, add, this);
      AutoLock c_lock(creation_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      region_nodes[r] = result;
      if (col_src != NULL)
        col_src->add_instance(result);
      row_src->add_instance(result);
      if (par != NULL)
        par->add_child(r, result);
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::create_node(LogicalPartition p, RegionNode *par, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!has_node(p));
      if (par != NULL)
      {
        assert(p.field_space == par->handle.field_space);
        assert(p.tree_id == par->handle.tree_id);
      }
#endif
      IndexPartNode *row_src = get_node(p.index_partition);
      PartitionNode *result = new PartitionNode(p, par, row_src, add, this);
      AutoLock c_lock(creation_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      part_nodes[p] = result;
      row_src->add_instance(result);
      if (par != NULL)
        par->add_child(p, result);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(IndexSpaceNode *node, bool top, bool finalize)
    //--------------------------------------------------------------------------
    {
      // We can only do this if we're finalizing
      if (top && finalize && (node->parent != NULL))
        node->parent->remove_child(node->color);
      // destroy any child nodes that haven't already been destroyed, then do ourselves
      for (std::map<Color,IndexPartNode*>::const_iterator it = node->valid_map.begin();
            it != node->valid_map.end(); it++)
      {
        destroy_node(it->second, false/*top*/, finalize);
      }
      // Can only clear the valid map if we're finalizing this deletion
      if (finalize)
        node->valid_map.clear();
      // Don't actually destroy anything, just mark destroyed, when the
      // destructor is called we'll decide if we want to do anything
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(IndexPartNode *node, bool top, bool finalize)
    //--------------------------------------------------------------------------
    {
      // We can only do this if we're finalizing
      if (top && finalize && (node->parent != NULL))
        node->parent->remove_child(node->color);
      // destroy any child nodes, then do ourselves
      for (std::map<Color,IndexSpaceNode*>::const_iterator it = node->valid_map.begin();
            it != node->valid_map.end(); it++)
      {
        destroy_node(it->second, false/*top*/, finalize);
      }
      // Can only clear the map if we're finalizing this deletion
      if (finalize)
        node->valid_map.clear();
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(FieldSpaceNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(field_nodes.find(node->handle) != field_nodes.end());
#endif
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(RegionNode *node, bool top, bool finalize,
                                const std::vector<ContextID> &deletion_contexts)
    //--------------------------------------------------------------------------
    {
      // Can only do this if we're finalizing the deletion
      if (top && finalize && (node->parent != NULL))
        node->parent->remove_child(node->row_source->color);
      // Now destroy our children
      for (std::map<Color,PartitionNode*>::const_iterator it = node->valid_map.begin();
            it != node->valid_map.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(has_node(it->second->handle));
#endif
        destroy_node(it->second, false/*top*/, finalize, deletion_contexts);
      }
      // Clear our valid map since we no longer have any valid children
      if (finalize)
      {
        node->valid_map.clear();
        // If we're finalizing then we can just free up all 
        // the valid instances and delete the contexts
        for (std::map<ContextID,RegionTreeNode::PhysicalState>::iterator it = 
              node->physical_states.begin(); it != node->physical_states.end(); it++)
        {
          node->invalidate_instance_views(it->second,FieldMask(FIELD_ALL_ONES), false/*clean*/);
          node->invalidate_reduction_views(it->second,FieldMask(FIELD_ALL_ONES));
        }
        node->physical_states.clear();
      }
      else
      {
        // Remove any references to valid views that we have in
        // the deletion contexts
        for (std::vector<ContextID>::const_iterator it = deletion_contexts.begin();
              it != deletion_contexts.end(); it++)
        {
          std::map<ContextID,RegionTreeNode::PhysicalState>::iterator finder = 
            node->physical_states.find(*it);
          if (finder != node->physical_states.end())
          {
            node->invalidate_instance_views(finder->second,FieldMask(FIELD_ALL_ONES), false/*clean*/);
            node->invalidate_reduction_views(finder->second,FieldMask(FIELD_ALL_ONES));
            node->physical_states.erase(finder);
          }
        }
      }
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_node(PartitionNode *node, bool top, bool finalize,
                                const std::vector<ContextID> &deletion_contexts)
    //--------------------------------------------------------------------------
    {
      // Can only do this if we're finalizing the deletion
      if (top && finalize && (node->parent != NULL))
        node->parent->remove_child(node->row_source->color);
      for (std::map<Color,RegionNode*>::const_iterator it = node->valid_map.begin();
            it != node->valid_map.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(has_node(it->second->handle));
#endif
        destroy_node(it->second, false/*top*/, finalize, deletion_contexts);
      }
      if (finalize)
      {
        node->valid_map.clear();
        // Since we're finalizing, we can remove all the contexts that
        // we no longer need, which is all of them.
        node->physical_states.clear();
      }
      else
      {
        // No valid views here, so we can just clear out our deletion states
        for (std::vector<ContextID>::const_iterator it = deletion_contexts.begin();
              it != deletion_contexts.end(); it++)
        {
          std::map<ContextID,RegionTreeNode::PhysicalState>::iterator finder = 
            node->physical_states.find(*it);
          if (finder != node->physical_states.end())
          {
            node->physical_states.erase(finder);
          }
        }
      }
      node->mark_destroyed();
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexSpace space) const
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(creation_lock);
      return (index_nodes.find(space) != index_nodes.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexPartition part) const
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(creation_lock);
      return (index_parts.find(part) != index_parts.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(FieldSpace space) const
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(creation_lock);
      return (field_nodes.find(space) != field_nodes.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalRegion handle, bool strict /*= true*/) const
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(creation_lock);
      if (region_nodes.find(handle) != region_nodes.end())
        return true;
      else if (!strict)
      {
        // Otherwise check to see if we could make it
        if (index_nodes.find(handle.index_space) == index_nodes.end())
          return false;
        if (field_nodes.find(handle.field_space) == field_nodes.end())
          return false;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalPartition handle, bool strict /*= true*/) const
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(creation_lock);
      if (part_nodes.find(handle) != part_nodes.end())
        return true;
      else if (!strict)
      {
        // Otherwise check to see if we could make it
        if (index_parts.find(handle.index_partition) == index_parts.end())
          return false;
        if (field_nodes.find(handle.field_space) == field_nodes.end())
          return false;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::get_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(creation_lock);
      std::map<IndexSpace,IndexSpaceNode*>::const_iterator it = index_nodes.find(space);
      if (it == index_nodes.end())
      {
        log_index(LEVEL_ERROR,"Unable to find entry for index space %x.  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", space.id);
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_INVALID_INDEX_SPACE_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::get_node(IndexPartition part)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(creation_lock);
      std::map<IndexPartition,IndexPartNode*>::const_iterator it = index_parts.find(part);
      if (it == index_parts.end())
      {
        log_index(LEVEL_ERROR,"Unable to find entry for index partition %d.  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", part);
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_INVALID_INDEX_PART_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::get_node(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      std::map<FieldSpace,FieldSpaceNode*>::const_iterator it = field_nodes.find(space);
      if (it == field_nodes.end())
      {
        log_field(LEVEL_ERROR,"Unable to find entry for field space %x.  This means it has either been "
                              "deleted or the appropriate privileges are not being requested.", space.id); 
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_INVALID_FIELD_SPACE_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::get_node(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalRegion,RegionNode*>::const_iterator it = region_nodes.find(handle);
      if (it == region_nodes.end())
      {
        IndexSpaceNode *index_node = get_node(handle.index_space);
        if (index_node == NULL)
        {
          log_region(LEVEL_ERROR,"Unable to find entry for logical region (%x,%x,%x).  This means it has either been "
                                "deleted or the appropriate privileges are not being requested.", 
                                handle.tree_id,handle.index_space.id,handle.field_space.id);
#ifdef DEBUG_HIGH_LEVEl
          assert(false);
#endif
          exit(ERROR_INVALID_REGION_ENTRY);
        }
        return index_node->instantiate_region(handle.tree_id, handle.field_space);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::get_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalPartition,PartitionNode*>::const_iterator it = part_nodes.find(handle);
      if (it == part_nodes.end())
      {
        IndexPartNode *index_node = get_node(handle.index_partition);
        if (index_node == NULL)
        {
          log_region(LEVEL_ERROR,"Unable to find entry for logical partition (%x,%x,%x).  This means it has either been "
                                "deleted or the appropriate privileges are not being requested.", 
                                handle.tree_id,handle.index_partition,handle.field_space.id);
#ifdef DEBUG_HIGH_LEVEl
          assert(false);
#endif
          exit(ERROR_INVALID_PARTITION_ENTRY);
        }
        return index_node->instantiate_partition(handle.tree_id, handle.field_space);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionTreeForest::create_instance_view(InstanceManager *manager, InstanceView *parent, 
                                                          RegionNode *reg, bool making_local)
    //--------------------------------------------------------------------------
    {
      InstanceKey key(manager->get_unique_id(), reg->handle);
      std::map<InstanceKey,InstanceView*>::const_iterator finder = views.find(key);
      InstanceView *result = NULL;
      if (finder == views.end())
      {
        result = new InstanceView(manager, parent, reg, this, making_local);
        views[key] = result;
        manager->add_view(result);
      }
      else
      {
        // This case only occurs when a child view has removed itself
        // from the parent because it no longer had any valid information
        // so add it back to the parent and return it
        result = finder->second;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      
      // If there is a parent, tell the parent that it has a child
      if (parent != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(reg->parent != NULL);
#endif
        parent->add_child_view(reg->parent->row_source->color,
                               reg->row_source->color, result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    InstanceManager* RegionTreeForest::create_instance_manager(Memory location, PhysicalInstance inst,
                      const std::map<FieldID,Domain::CopySrcDstField> &infos,
                      FieldSpace fsp, const FieldMask &field_mask, bool remote, bool clone,
                      UniqueManagerID mid /*= 0*/)
    //--------------------------------------------------------------------------
    {
      if (mid == 0)
        mid = runtime->get_unique_manager_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(managers.find(mid) == managers.end());
#endif
      InstanceManager *result = new InstanceManager(location, inst, infos, fsp, field_mask,
                                                    this, mid, remote, clone);
      managers[mid] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionView* RegionTreeForest::create_reduction_view(ReductionManager *manager, 
                                                          RegionNode *reg, bool made_local)
    //--------------------------------------------------------------------------
    {
      InstanceKey key(manager->get_unique_id(), reg->handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(reduc_views.find(key) == reduc_views.end());
#endif
      ReductionView *result = new ReductionView(this, reg, made_local, manager);
      reduc_views[key] = result;
      manager->set_view(result);
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionManager* RegionTreeForest::create_reduction_manager(Memory location, PhysicalInstance inst,
                        ReductionOpID redop, const ReductionOp *op, bool remote, bool clone,
                        Domain sparse_domain /*= Domain::NO_DOMAIN*/, UniqueManagerID mid /*= 0*/)
    //--------------------------------------------------------------------------
    {
      if (mid == 0)
        mid = runtime->get_unique_manager_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(reduc_managers.find(mid) == reduc_managers.end());
#endif
      if (sparse_domain.exists())
      {
        ReductionManager *result = new ListReductionManager(this, mid, remote, clone, location,
                                                            inst, redop, op, sparse_domain);
        reduc_managers[mid] = result;
        return result;
      }
      else
      {
        ReductionManager *result = new FoldReductionManager(this, mid, remote, clone, location,
                                                            inst, redop, op);
        reduc_managers[mid] = result;
        return result;
      }
    }

    //--------------------------------------------------------------------------
    PhysicalView* RegionTreeForest::find_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      std::map<InstanceKey,InstanceView*>::const_iterator finder = views.find(key);
      if (finder != views.end())
        return finder->second;
      std::map<InstanceKey,ReductionView*>::const_iterator finder2 = reduc_views.find(key);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder2 != reduc_views.end()); 
#endif
      return finder2->second;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* RegionTreeForest::find_manager(UniqueManagerID mid) const
    //--------------------------------------------------------------------------
    {
      std::map<UniqueManagerID,InstanceManager*>::const_iterator finder = managers.find(mid);
      if (finder != managers.end())
        return finder->second;
      std::map<UniqueManagerID,ReductionManager*>::const_iterator finder2 = reduc_managers.find(mid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder2 != reduc_managers.end());
#endif
      return finder2->second;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      return ((views.find(key) != views.end()) || (reduc_views.find(key) != reduc_views.end()));
    }

    //--------------------------------------------------------------------------
    template<typename T>
    Color RegionTreeForest::generate_unique_color(const std::map<Color,T> &current_map)
    //--------------------------------------------------------------------------
    {
      if (current_map.empty())
        return runtime->get_start_color();
      unsigned stride = runtime->get_color_modulus();
      typename std::map<Color,T>::const_reverse_iterator rlast = current_map.rbegin();
      Color result = rlast->first + stride;
#ifdef DEBUG_HIGH_LEVEL
      assert(current_map.find(result) == current_map.end());
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionTreeForest::find_instance_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      std::map<InstanceKey,InstanceView*>::const_iterator finder = views.find(key);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != views.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    InstanceManager* RegionTreeForest::find_instance_manager(UniqueManagerID mid) const
    //--------------------------------------------------------------------------
    {
      std::map<UniqueManagerID,InstanceManager*>::const_iterator finder = managers.find(mid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != managers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_instance_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      return (views.find(key) != views.end());
    }

    //--------------------------------------------------------------------------
    ReductionView* RegionTreeForest::find_reduction_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      std::map<InstanceKey,ReductionView*>::const_iterator finder = reduc_views.find(key);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != reduc_views.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    ReductionManager* RegionTreeForest::find_reduction_manager(UniqueManagerID mid) const
    //--------------------------------------------------------------------------
    {
      std::map<UniqueManagerID,ReductionManager*>::const_iterator finder = reduc_managers.find(mid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != reduc_managers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_reduction_view(InstanceKey key) const
    //--------------------------------------------------------------------------
    {
      return (reduc_views.find(key) != reduc_views.end());
    }

    /////////////////////////////////////////////////////////////
    // Index Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(Domain d, IndexPartNode *par, Color c, bool add, RegionTreeForest *ctx)
      : domain(d), handle(d.get_index_space()), depth((par == NULL) ? 0 : par->depth+1),
        color(c), parent(par), context(ctx), added(add), marked(false), 
        destroy_index_space(false), node_destroyed(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::~IndexSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      if (destroy_index_space)
      {
        // We were the owner so tell the low-level runtime we're done
        handle.destroy();
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      // Note this can be called twice, once when the deletion is first
      // performed and again when deletions are flushed
      node_destroyed = true;
      // If we were the owners of this index space mark that we can free
      // the index space when our destructor is called
      if (added)
      {
        destroy_index_space = true;
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_child(IndexPartition handle, IndexPartNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->color) == color_map.end());
#endif
      color_map[node->color] = node;
      valid_map[node->color] = node;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      // only ever remove things from the valid map
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_map.find(c) != valid_map.end());
#endif
      valid_map.erase(c);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::has_child(Color c) const
    //--------------------------------------------------------------------------
    {
      std::map<Color,IndexPartNode*>::const_iterator finder = color_map.find(c);
      return (finder != color_map.end());
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexSpaceNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::are_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (c1 == c2) 
        return false;
      if (disjoint_subsets.find(std::pair<Color,Color>(c1,c2)) !=
          disjoint_subsets.end())
        return true;
      else if (disjoint_subsets.find(std::pair<Color,Color>(c2,c1)) !=
               disjoint_subsets.end())
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (disjoint_subsets.find(std::pair<Color,Color>(c2,c1)) == 
          disjoint_subsets.end())
        disjoint_subsets.insert(std::pair<Color,Color>(c1,c2));
    }

    //--------------------------------------------------------------------------
    Color IndexSpaceNode::generate_color(void)
    //--------------------------------------------------------------------------
    {
      return context->generate_unique_color<IndexPartNode*>(color_map);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        assert((*it) != inst);
      }
#endif
      logical_nodes.push_back(inst);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      for (std::list<RegionNode*>::iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it) == inst)
        {
          logical_nodes.erase(it);
          return;
        }
      }
      assert(false); // should never get here
    }

    //--------------------------------------------------------------------------
    RegionNode* IndexSpaceNode::instantiate_region(RegionTreeID tid, FieldSpace fid)
    //--------------------------------------------------------------------------
    {
      LogicalRegion target(tid, handle, fid);
      // Check to see if we already have one made
      for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle == target)
          return *it;
      }
      // Otherwise we're going to need to make it, first make the parent
      PartitionNode *target_parent = NULL;
      if (parent != NULL)
        target_parent = parent->instantiate_partition(tid, fid);
      return context->create_node(target, target_parent, true/*add*/); 
    }

    //--------------------------------------------------------------------------
    size_t IndexSpaceNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0; 
      result += sizeof(bool);
      result += sizeof(domain);
      if (returning || marked)
      {
        result += sizeof(color);
        result += sizeof(size_t); // number of children
        result += sizeof(size_t); // number disjoint subsets
        result += (disjoint_subsets.size() * 2 * sizeof(Color));
        // Do all the children
        for (std::map<Color,IndexPartNode*>::const_iterator it = 
              valid_map.begin(); it != valid_map.end(); it++)
          result += it->second->compute_tree_size(returning);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(domain);
        rez.serialize(color);
        rez.serialize(valid_map.size());
        for (std::map<Color,IndexPartNode*>::const_iterator it = 
              valid_map.begin(); it != valid_map.end(); it++)
        {
          it->second->serialize_tree(rez, returning);
        }
        rez.serialize(disjoint_subsets.size());
        for (std::set<std::pair<Color,Color> >::const_iterator it =
              disjoint_subsets.begin(); it != disjoint_subsets.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
        rez.serialize(domain);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ IndexSpaceNode* IndexSpaceNode::deserialize_tree(Deserializer &derez, IndexPartNode *parent,
                                  RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool need_unpack;
      derez.deserialize(need_unpack);
      Domain domain;
      derez.deserialize(domain);
      if (need_unpack)
      {
        Color color;
        derez.deserialize(color);
        IndexSpaceNode *result_node = context->create_node(domain, parent, color, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          IndexPartNode::deserialize_tree(derez, result_node, context, returning);
        }
        size_t num_disjoint;
        derez.deserialize(num_disjoint);
        for (unsigned idx = 0; idx < num_disjoint; idx++)
        {
          Color c1, c2;
          derez.deserialize(c1);
          derez.deserialize(c2);
          result_node->add_disjoint(c1, c2);
        }
        return result_node;
      }
      else
      {
        return context->get_node(domain.get_index_space());
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,IndexPartNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexSpaceNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked); // we should only be here if this is marked
#endif
      if ((parent == NULL) || (!parent->marked))
        return const_cast<IndexSpaceNode*>(this);
      return parent->find_top_marked();
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::find_new_partitions(std::vector<IndexPartNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!added);
#endif
      for (std::map<Color,IndexPartNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(IndexPartition p, IndexSpaceNode *par, Color c, 
                                  Domain cspace, bool dis, bool add, RegionTreeForest *ctx)
      : handle(p), depth((par == NULL) ? 0 : par->depth+1),
        color(c), color_space(cspace), parent(par), context(ctx), 
        disjoint(dis), added(add), marked(false), node_destroyed(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexPartNode::~IndexPartNode(void)
    //--------------------------------------------------------------------------
    {
      // In the future we may want to reclaim partition handles here
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      // Note this can be called twice, once when the deletion is first
      // performed and again when deletions are flushed
      node_destroyed = true;
      added = false;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_child(IndexSpace handle, IndexSpaceNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->color) == color_map.end());
#endif
      color_map[node->color] = node;
      valid_map[node->color] = node;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      // only remove things from valid map
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_map.find(c) != valid_map.end());
#endif
      valid_map.erase(c);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::has_child(Color c) const
    //--------------------------------------------------------------------------
    {
      std::map<Color,IndexSpaceNode*>::const_iterator finder = color_map.find(c);
      return (finder != color_map.end());
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::are_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return false;
      if (disjoint)
        return true;
      if (disjoint_subspaces.find(std::pair<Color,Color>(c1,c2)) !=
          disjoint_subspaces.end())
        return true;
      if (disjoint_subspaces.find(std::pair<Color,Color>(c2,c1)) !=
          disjoint_subspaces.end())
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      if (disjoint_subspaces.find(std::pair<Color,Color>(c2,c1)) ==
          disjoint_subspaces.end())
        disjoint_subspaces.insert(std::pair<Color,Color>(c1,c2));
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_instance(PartitionNode *inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      for (std::list<PartitionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        assert((*it) != inst);
      }
#endif
      logical_nodes.push_back(inst);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::remove_instance(PartitionNode *inst)
    //--------------------------------------------------------------------------
    {
      for (std::list<PartitionNode*>::iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it) == inst)
        {
          logical_nodes.erase(it);
          return;
        }
      }
      assert(false); // should never get here
    }

    //--------------------------------------------------------------------------
    PartitionNode* IndexPartNode::instantiate_partition(RegionTreeID tid, FieldSpace fid)
    //--------------------------------------------------------------------------
    {
      LogicalPartition target(tid, handle, fid);
      for (std::list<PartitionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle == target)
          return *it;
      }
      // Otherwise we're going to need to make it
#ifdef DEBUG_HIGH_LEVEL
      // This requires that there always be at least part of the region
      // tree local.  This might not always be true.
      assert(parent != NULL);
#endif
      RegionNode *target_parent = parent->instantiate_region(tid, fid);
      return context->create_node(target, target_parent, true/*add*/);
    }

    //--------------------------------------------------------------------------
    size_t IndexPartNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(bool);
      if (returning || marked)
      {
        result += sizeof(handle);
        result += sizeof(color);
        result += sizeof(color_space);
        result += sizeof(disjoint);
        result += sizeof(size_t); // number of children
        for (std::map<Color,IndexSpaceNode*>::const_iterator it = 
              valid_map.begin(); it != valid_map.end(); it++)
          result += it->second->compute_tree_size(returning);
        result += sizeof(size_t); // number of disjoint children
        result += (disjoint_subspaces.size() * 2 * sizeof(Color));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(handle > 0);
#endif
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(handle);
        rez.serialize(color);
        rez.serialize(color_space);
        rez.serialize(disjoint);
        rez.serialize(valid_map.size());
        for (std::map<Color,IndexSpaceNode*>::const_iterator it = 
              valid_map.begin(); it != valid_map.end(); it++)
          it->second->serialize_tree(rez, returning);
        rez.serialize(disjoint_subspaces.size());
        for (std::set<std::pair<Color,Color> >::const_iterator it =
              disjoint_subspaces.begin(); it != disjoint_subspaces.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::deserialize_tree(Deserializer &derez, IndexSpaceNode *parent,
                                RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool needs_unpack;
      derez.deserialize(needs_unpack);
      if (needs_unpack)
      {
        IndexPartition handle;
        derez.deserialize(handle);
#ifdef DEBUG_HIGH_LEVEL
        assert(handle > 0);
#endif
        Color color;
        derez.deserialize(color);
        Domain domain;
        derez.deserialize(domain);
        bool disjoint;
        derez.deserialize(disjoint);
        IndexPartNode *result = context->create_node(handle, parent, color, domain, disjoint, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          IndexSpaceNode::deserialize_tree(derez, result, context, returning);
        }
        size_t num_disjoint;
        derez.deserialize(num_disjoint);
        for (unsigned idx = 0; idx < num_disjoint; idx++)
        {
          Color c1, c2;
          derez.deserialize(c1);
          derez.deserialize(c2);
          result->add_disjoint(c1,c2);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,IndexSpaceNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      } 
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked);
      assert(parent != NULL);
#endif
      return parent->find_top_marked(); 
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::find_new_partitions(std::vector<IndexPartNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
      // See if we're new, if so we're done
      if (added)
      {
        IndexPartNode *copy = const_cast<IndexPartNode*>(this);
        new_parts.push_back(copy);
        return;
      }
      // Otherwise continue
      for (std::map<Color,IndexSpaceNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx)
      : handle(sp), context(ctx), total_index_fields(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::~FieldSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      // In the future we may want to reclaim field space names here
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      // Intentionally do nothing
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::allocate_fields(const std::map<FieldID,size_t> &field_allocations)
    //--------------------------------------------------------------------------
    {
      FieldMask update_mask;
      for (std::map<FieldID,size_t>::const_iterator it = field_allocations.begin();
            it != field_allocations.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        if (total_index_fields >= MAX_FIELDS)
        {
          log_field(LEVEL_ERROR,"Exceeded maximum number of allocated fields for a field space %d. "  
                                "Change 'MAX_FIELDS' at the top of legion_types.h and recompile.", MAX_FIELDS);
          assert(false);
          exit(ERROR_MAX_FIELD_OVERFLOW);
        }
        assert(fields.find(it->first) == fields.end());
#endif
        update_mask.set_bit<FIELD_SHIFT,FIELD_MASK>(total_index_fields);
        fields[it->first] = FieldInfo(it->second,total_index_fields++);
        created_fields.push_back(it->first);
      }
      // Tell the top of each region tree to update the top mask for each of its contexts
      for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        (*it)->update_top_mask(update_mask);
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_fields(const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      for (std::set<FieldID>::const_iterator it = to_free.begin();
            it != to_free.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(fields.find(*it) != fields.end());
#endif
        // Let's not actually erase it
        //fields.erase(*it);
        deleted_fields.push_back(*it);
        // Check to see if we created it
        for (std::list<FieldID>::iterator cit = created_fields.begin();
              cit != created_fields.end(); cit++)
        {
          if ((*cit) == (*it))
          {
            created_fields.erase(cit);
            // No longer needs to be marked deleted
            deleted_fields.pop_back();
            break;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      return (fields.find(fid) != fields.end());
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::get_field_size(FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.find(fid) != fields.end());
#endif
      return fields[fid].field_size;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::is_set(FieldID fid, const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != fields.end());
#endif
      return mask.is_set<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        assert((*it) != inst);
      }
#endif
      logical_nodes.push_back(inst);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::remove_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      for (std::list<RegionNode*>::iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it) == inst)
        {
          logical_nodes.erase(it);
          return;
        }
      }
      assert(false); // should never get here
    }

    //--------------------------------------------------------------------------
    InstanceManager* FieldSpaceNode::create_instance(Memory location, Domain domain,
                        const std::vector<FieldID> &new_fields, size_t blocking_factor)
    //--------------------------------------------------------------------------
    {
      InstanceManager *result = NULL;
      if (new_fields.size() == 1)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(new_fields.back());
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif

        PhysicalInstance inst = domain.create_instance(location, finder->second.field_size);
        if (inst.exists())
        {
          std::map<FieldID,Domain::CopySrcDstField> field_infos;
          field_infos[new_fields.back()] = Domain::CopySrcDstField(inst, 0, finder->second.field_size);
          result = context->create_instance_manager(location, inst, field_infos, handle, get_field_mask(new_fields),
                                                    false/*remote*/, false/*clone*/);
#ifdef LEGION_PROF
          {
            std::map<FieldID,size_t> inst_fields;
            inst_fields[new_fields.back()] = finder->second.field_size;
            LegionProf::register_instance_creation(inst.id, result->get_unique_id(), location.id, 
                                0/*redop*/, 1/* blocking factor*/, inst_fields);
          }
#endif
        }
      }
      else
      {
        std::vector<size_t> field_sizes;
        // Figure out the size of each element
        for (unsigned idx = 0; idx < new_fields.size(); idx++)
        {
          std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(new_fields[idx]);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != fields.end());
#endif
          field_sizes.push_back(finder->second.field_size);
        }
        // Now try and make the instance
        PhysicalInstance inst = domain.create_instance(location, field_sizes, blocking_factor);
        if (inst.exists())
        {
          std::map<FieldID,Domain::CopySrcDstField> field_infos;
          unsigned accum_offset = 0;
#ifdef DEBUG_HIGH_LEVEL
          assert(field_sizes.size() == new_fields.size());
#endif
          for (unsigned idx = 0; idx < new_fields.size(); idx++)
          {
            field_infos[new_fields[idx]] = Domain::CopySrcDstField(inst, accum_offset, field_sizes[idx]);
            accum_offset += field_sizes[idx];
          }
          result = context->create_instance_manager(location, inst, field_infos, handle, get_field_mask(new_fields),
                                                    false/*remote*/, false/*clone*/);
#ifdef LEGION_PROF
          {
            std::map<FieldID,size_t> inst_fields;
            for (unsigned idx = 0; idx < new_fields.size(); idx++)
            {
              inst_fields[new_fields[idx]] = field_sizes[idx];
            }
            LegionProf::register_instance_creation(inst.id, result->get_unique_id(), location.id, 0/*redop*/,
                                                  blocking_factor, inst_fields);
          }
#endif
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::compute_node_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(handle);
      result += sizeof(size_t); // number of fields
      result += (fields.size() * (sizeof(FieldID) + sizeof(FieldInfo)));
      return result;;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::serialize_node(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(handle);
      rez.serialize(fields.size());
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ FieldSpaceNode* FieldSpaceNode::deserialize_node(Deserializer &derez, RegionTreeForest *context)
    //--------------------------------------------------------------------------
    {
      FieldSpace handle;
      derez.deserialize(handle);
      FieldSpaceNode *result = context->create_node(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      unsigned max_id = 0;
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        FieldInfo info;
        derez.deserialize(info);
        result->fields[fid] = info;
        if (info.idx > max_id)
          max_id = info.idx;
      }
      // Ignore segmentation for now
      result->total_index_fields = max_id + 1;
      // No shift since this field space is just returning
      result->shift = 0;
#ifdef DEBUG_HIGH_LEVEL
      result->sanity_check();
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_modifications(void) const
    //--------------------------------------------------------------------------
    {
      // Has modifications if it has fields that have been created locally
      // and not destroyed while here.  Fields that were passed here and
      // then deleted will get passed back by the enclosing task context.
      std::set<FieldID> del_fields(deleted_fields.begin(),deleted_fields.end());
      for (std::list<FieldID>::const_iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        if (del_fields.find(*it) == del_fields.end())
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::compute_field_return_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0; 
      result += sizeof(size_t); // number of created fields
      result += (created_fields.size() * (sizeof(FieldID) + sizeof(FieldInfo)));
      result += sizeof(size_t); // number of deleted fields
      result += (deleted_fields.size() * sizeof(FieldID)); 
      return result;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::serialize_field_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(created_fields.size());
      for (std::list<FieldID>::const_iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        rez.serialize(*it);
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        rez.serialize(finder->second);
      }
      // Don't clear created fields here, we still need it for finding the created state
      rez.serialize(deleted_fields.size());
      for (std::list<FieldID>::const_iterator it = deleted_fields.begin();
            it != deleted_fields.end(); it++)
      {
        rez.serialize(*it);
      }
      deleted_fields.clear();
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::deserialize_field_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      {
        size_t num_new_fields;
        derez.deserialize(num_new_fields);
        std::map<FieldID,size_t> new_fields;
        std::map<unsigned/*idx*/,FieldID> old_field_indexes;
        for (unsigned idx = 0; idx < num_new_fields; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          FieldInfo info;
          derez.deserialize(info);
          new_fields[fid] = info.field_size;
          old_field_indexes[info.idx] = fid;
        }
        // Rather than doing the standard allocation procedure, we instead
        // allocated all the fields so that they are all a constant shift
        // offset from their original index.  As a result when we unpack the
        // physical state for the created fields, we need only apply a shift
        // to the FieldMasks rather than rebuilding them all from scratch.
#ifdef DEBUG_HIGH_LEVEL
        assert(!new_fields.empty());
        assert(new_fields.size() == old_field_indexes.size());
#endif
        unsigned first_index = old_field_indexes.begin()->first;
#ifdef DEBUG_HIGH_LEVEL
        assert(total_index_fields >= first_index);
#endif
        shift = total_index_fields - first_index;
        FieldMask update_mask;
        for (std::map<unsigned,FieldID>::const_iterator it = old_field_indexes.begin();
              it != old_field_indexes.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(fields.find(it->second) == fields.end());
#endif
          update_mask |= (1 << (it->first + shift));
          fields[it->second] = FieldInfo(new_fields[it->second], it->first + shift);
          unsigned new_total_index_fields = it->first+shift+1;
#ifdef DEBUG_HIGH_LEVEL
          assert(new_total_index_fields > total_index_fields);
#endif
          total_index_fields = new_total_index_fields;
          created_fields.push_back(it->second);
        }
#ifdef DEBUG_HIGH_LEVEL
        if (total_index_fields >= MAX_FIELDS)
        {
          log_field(LEVEL_ERROR,"Exceeded maximum number of allocated fields for a field space %d when unpacking. "  
                                "Change 'MAX_FIELDS' at the top of legion_types.h and recompile.", MAX_FIELDS);
          assert(false);
          exit(ERROR_MAX_FIELD_OVERFLOW);
        }
#endif
        for (std::list<RegionNode*>::const_iterator it = logical_nodes.begin();
              it != logical_nodes.end(); it++)
        {
          (*it)->update_top_mask(update_mask);
        }
      }
      {
        size_t num_deleted_fields;
        derez.deserialize(num_deleted_fields);
        std::set<FieldID> del_fields;
        for (unsigned idx = 0; idx < num_deleted_fields; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          del_fields.insert(fid);
        }
        free_fields(del_fields);
      }
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(const std::vector<FieldID> &mask_fields) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::vector<FieldID>::const_iterator it = mask_fields.begin();
            it != mask_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(const std::set<FieldID> &mask_fields) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::set<FieldID>::const_iterator it = mask_fields.begin();
            it != mask_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(void) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(it->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_created_field_mask(void) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      for (std::list<FieldID>::const_iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        result.set_bit<FIELD_SHIFT,FIELD_MASK>(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    char* FieldSpaceNode::to_string(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!!mask);
#endif
      char *result = (char*)malloc(MAX_FIELDS*4); 
      bool first = true;
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (is_set(it->first, mask))
        {
          if (first)
          {
            sprintf(result,"%d",it->first);
            first = false;
          }
          else
          {
            char temp[8];
            sprintf(temp,",%d",it->first);
            strcat(result, temp);
          }
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(!first); // we should have written something
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::sanity_check(void) const
    //--------------------------------------------------------------------------
    {
      std::set<unsigned> indexes;
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        assert(indexes.find(it->second.idx) == indexes.end());
        indexes.insert(it->second.idx);
      }
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeNode::RegionTreeNode(RegionTreeForest *ctx)
      : context(ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_region(const LogicalUser &user, RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!az.path.empty());
      assert(color_match(az.path.back()));
#endif
      if (logical_states.find(az.ctx) == logical_states.end())
        logical_states[az.ctx] = LogicalState();
      
      LogicalState &state = logical_states[az.ctx];
      if (az.path.size() == 1)
      {
        az.path.pop_back();
        // We've arrived where we're going, go through and do the dependence analysis
        FieldMask dominator_mask = perform_dependence_checks(user, state.curr_epoch_users, user.field_mask);
        FieldMask non_dominated_mask = user.field_mask - dominator_mask;
        // For the fields that weren't dominated, we have to check those fields against the prev_epoch users
        if (!!non_dominated_mask)
          perform_dependence_checks(user, state.prev_epoch_users, non_dominated_mask);
        // Update the dominated fields 
        if (!!dominator_mask)
        {
          // Dominator mask is not empty
          // Mask off all the dominated fields from the prev_epoch_users
          // Remove any prev_epoch_users that were totally dominated
          for (std::list<LogicalUser>::iterator it = state.prev_epoch_users.begin();
                it != state.prev_epoch_users.end(); /*nothing*/)
          {
            it->field_mask -= dominator_mask;
            if (!it->field_mask)
              it = state.prev_epoch_users.erase(it); // empty so we can erase it
            else
              it++; // still has non-dominated fields
          }
          // Mask off all dominated fields from curr_epoch_users, and move them
          // to prev_epoch_users.  If all fields masked off, then remove them
          // from curr_epoch_users.
          for (std::list<LogicalUser>::iterator it = state.curr_epoch_users.begin();
                it != state.curr_epoch_users.end(); /*nothing*/)
          {
            FieldMask local_dom = it->field_mask & dominator_mask;
            if (!!local_dom)
            {
              // Move a copy over to the previous epoch users for the
              // fields that were dominated
              state.prev_epoch_users.push_back(*it);
              state.prev_epoch_users.back().field_mask = local_dom;
            }
            // Update the field mask with the non-dominated fields
            it->field_mask -= dominator_mask;
            if (!it->field_mask)
              it = state.curr_epoch_users.erase(it); // empty so we can erase it
            else
              it++; // Not empty so keep going
          }
        }
        // Close up any partitions which we might have dependences on below
        LogicalCloser closer(user, az.ctx, state.prev_epoch_users, are_closing_partition());
        siphon_open_children(closer, state, user, user.field_mask);
        // Add ourselves to the current epoch
        state.curr_epoch_users.push_back(user);
      }
      else
      {
        // Not there yet
        az.path.pop_back();
        Color next_child = az.path.back();
        // Perform the checks on the current users and the epoch users since we're still traversing
        perform_dependence_checks(user, state.curr_epoch_users, user.field_mask);
        perform_dependence_checks(user, state.prev_epoch_users, user.field_mask);
        
        LogicalCloser closer(user, az.ctx, state.prev_epoch_users, are_closing_partition());
        bool open_only = siphon_open_children(closer, state, user, user.field_mask, next_child);
        // Now we can continue the traversal, figure out if we need to just continue
        // or whether we can do an open operation
        RegionTreeNode *child = get_tree_child(next_child);
        if (open_only)
          child->open_logical_tree(user, az);
        else
          child->register_logical_region(user, az);
      }
      // Everything below here is only a performance optimization to keep the lists
      // from growing too big.  Turn it off for LegionSpy where we want to see
      // all the mapping dependences no matter what.
#ifndef LEGION_SPY
#if 0
      // This is the more complicated version that is currently wrong because
      // we can't guarantee that operations in the list are in program order
      //
      // Check to see if any of the sets of users are too large, if they are,
      // then resize them.  Yay for Nyquist, once we get 2X tasks, drop back to X tasks
      // Only do this if we're not doing LegionSpy in which case we probably want
      // to see all of the dependences anyway
      unsigned max_window_size = HighLevelRuntime::get_max_context_users();
      unsigned max_list_size = 2*max_window_size;
      if (state.curr_epoch_users.size() >= max_list_size)
        down_sample_list(state.curr_epoch_users, max_window_size);
      if (state.prev_epoch_users.size() >= max_list_size)
        down_sample_list(state.prev_epoch_users, max_window_size);
#else
      // The simpler way of shortening the lists by filtering out any tasks that have
      // already mapped and therefore will never have any dependences performed on them
      unsigned max_filter_size = HighLevelRuntime::get_max_filter_size();
      if (state.curr_epoch_users.size() >= max_filter_size)
        filter_user_list(state.curr_epoch_users);
      if (state.prev_epoch_users.size() >= max_filter_size)
        filter_user_list(state.prev_epoch_users);
#endif
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::open_logical_tree(const LogicalUser &user, RegionAnalyzer &az)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!az.path.empty());
      assert(color_match(az.path.back()));
#endif
      // If a state doesn't exist yet, create it
      if (logical_states.find(az.ctx) == logical_states.end())
        logical_states[az.ctx] = LogicalState();
      LogicalState &state = logical_states[az.ctx];
      if (az.path.size() == 1)
      {
        // We've arrived wehere we're going, add ourselves as a user
        state.curr_epoch_users.push_back(user);
        az.path.pop_back();
      }
      else
      {
        az.path.pop_back();
        Color next_child = az.path.back();
        std::vector<FieldState> new_states;
        new_states.push_back(FieldState(user, user.field_mask, next_child));
        merge_new_field_states(state, new_states, false/*add states*/);
        // Then continue the traversal
        RegionTreeNode *child_node = get_tree_child(next_child);
        child_node->open_logical_tree(user, az);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_logical_tree(LogicalCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_states.find(closer.ctx) != logical_states.end());
#endif
      LogicalState &state = logical_states[closer.ctx];
      // Register any dependences we have here
      FieldMask dominator_mask = perform_dependence_checks(closer.user, state.curr_epoch_users, 
                                                            closing_mask, closer.closing_partition);
      FieldMask non_dominator_mask = closing_mask - dominator_mask;
      if (!!non_dominator_mask)
        perform_dependence_checks(closer.user, state.prev_epoch_users, non_dominator_mask, closer.closing_partition);
      // Now get the epoch users that we need to send back
      for (std::list<LogicalUser>::iterator it = state.curr_epoch_users.begin();
            it != state.curr_epoch_users.end(); /*nothing*/)
      {
        // Now check for field disjointness
        if (closing_mask * it->field_mask)
        {
          it++;
          continue;
        }
        // Now figure out how to split this user to send the part
        // corresponding to the closing mask back to the parent
        closer.epoch_users.push_back(*it);
        closer.epoch_users.back().field_mask &= closing_mask;
        // Remove the closed set of fields from this user
        it->field_mask -= closing_mask;
        // If it's empty, remove it from the list
        if (!it->field_mask)
          it = state.curr_epoch_users.erase(it);
        else
          it++;
      }
      // Also go through and mask out any users in the prev_epoch_users list
      for (std::list<LogicalUser>::iterator it = state.prev_epoch_users.begin();
            it != state.prev_epoch_users.end(); /*nothing*/)
      {
        if (closing_mask * it->field_mask)
        {
          it++;
          continue;
        }
        // If this has one of the fields that wasn't dominated, include it
        if (!!non_dominator_mask && !(non_dominator_mask * it->field_mask))
        {
          closer.epoch_users.push_back(*it);
          closer.epoch_users.back().field_mask &= non_dominator_mask;
        }
        it->field_mask -= closing_mask;
        if (!it->field_mask)
          it = state.prev_epoch_users.erase(it);
        else
          it++;
      }
      // Now we need to traverse any open children 
      siphon_open_children(closer, state, closer.user, closing_mask);
    }

    //--------------------------------------------------------------------------
    FieldMask RegionTreeNode::perform_dependence_checks(const LogicalUser &user, 
        const std::list<LogicalUser> &users, const FieldMask &user_mask, bool closing_partition/*= false*/)
    //--------------------------------------------------------------------------
    {
      FieldMask dominator_mask = user_mask;
      for (std::list<LogicalUser>::const_iterator it = users.begin();
            it != users.end(); it++)
      {
        // Special case for closing partition, if we already have a user then we can ignore
        // it because we have over-approximated our set of regions by saying we're using a
        // partition.  This occurs whenever an index space task says its using a partition,
        // but might only use a subset of the regions in the partition, and then also has
        // a region requirement for another one of the regions in the partition.
        if (closing_partition && (it->op == user.op) && (it->gen == user.gen))
          continue;
        // Check to see if things are disjoint
        if (user_mask * it->field_mask)
          continue;
        if (!perform_dependence_check(*it, user))
        {
          // There wasn't a dependence so remove the bits from the
          // dominator mask
          dominator_mask -= it->field_mask;
        }
      }
      return dominator_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_state(GenericState &gstate,
                          const FieldState &new_state, bool add_state)
    //--------------------------------------------------------------------------
    {
      bool added = false;
      for (std::list<FieldState>::iterator it = gstate.field_states.begin();
            it != gstate.field_states.end(); it++)
      {
        if (it->overlap(new_state))
        {
          it->merge(new_state);
          added = true;
          break;
        }
      }
      if (!added)
        gstate.field_states.push_back(new_state);

      if (add_state)
        gstate.added_states.push_back(new_state);
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_field_states(gstate);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_states(GenericState &gstate,
                          std::vector<FieldState> &new_states, bool add_states)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < new_states.size(); idx++)
      {
        const FieldState &next = new_states[idx];
        merge_new_field_state(gstate, next, add_states);        
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_field_states(gstate);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::upgrade_new_field_state(GenericState &gstate,
                                                 const FieldState &new_state, bool add_state)
    //--------------------------------------------------------------------------
    {
      // First go through and remove any FieldStates that are being upgraded  
      for (std::list<FieldState>::iterator it = gstate.field_states.begin();
            it != gstate.field_states.end(); /*nothing*/)
      {
        // See if the new_state overlaps or is still valid after upgrading
        if (it->overlap(new_state) || it->upgrade(new_state))
          it++; // still valid
        else
          it = gstate.field_states.erase(it); // otherwise remove it
      }
      // now do the merge 
      merge_new_field_state(gstate, new_state, add_state);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::sanity_check_field_states(const GenericState &gstate) 
    //--------------------------------------------------------------------------
    {
      // For every child and every field, it should only be open in one mode
      std::map<Color,FieldMask> previous_children;
      for (std::list<FieldState>::const_iterator fit = gstate.field_states.begin();
            fit != gstate.field_states.end(); fit++)
      {
        FieldMask actually_valid;
        for (std::map<Color,FieldMask>::const_iterator it = fit->open_children.begin();
              it != fit->open_children.end(); it++)
        {
          actually_valid |= it->second;
          if (previous_children.find(it->first) == previous_children.end())
          {
            previous_children[it->first] = it->second;
          }
          else
          {
            FieldMask &previous = previous_children[it->first];
            assert(!(previous & it->second));
            previous |= it->second;
          }
        }
        // Valid fields should line up
        assert(actually_valid == fit->valid_fields);
      }
      // Also check that for each field it is either only open in one mode
      // or two children in different modes are disjoint
      for (std::list<FieldState>::const_iterator it1 = gstate.field_states.begin();
            it1 != gstate.field_states.end(); it1++)
      {
        for (std::list<FieldState>::const_iterator it2 = gstate.field_states.begin();
              it2 != gstate.field_states.end(); it2++)
        {
          if (it1 == it2) // No need to do comparisons if they are the same field state
            continue;
          const FieldState &f1 = *it1;
          const FieldState &f2 = *it2;
          for (std::map<Color,FieldMask>::const_iterator cit1 = f1.open_children.begin();
                cit1 != f1.open_children.end(); cit1++)
          {
            for (std::map<Color,FieldMask>::const_iterator cit2 = f2.open_children.begin();
                  cit2 != f2.open_children.end(); cit2++)
            {
              
              // Disjointness check on fields
              if (cit1->second * cit2->second)
                continue;
              Color c1 = cit1->first;
              Color c2 = cit2->first;
              // Some aliasing in the fields, so do the check for child disjointness
              assert(c1 != c2);
              assert(are_children_disjoint(c1, c2));
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::down_sample_list(std::list<LogicalUser> &users, unsigned max_users)
    //--------------------------------------------------------------------------
    {
      EpochOperation *epoch_op = context->runtime->get_available_epoch();
      // Get a mask containing the fields for all of the users we are about to pull off
      FieldMask epoch_mask;
      std::list<LogicalUser>::iterator it = users.begin();
      // need to lock the epoch_op context when doing this so it can't trigger
      epoch_op->start_analysis();
      epoch_op->start_down_sample();
      while (users.size() >= max_users)
      {
        LogicalUser user = *it;
        it = users.erase(it);
        if (user.op == epoch_op)
          continue;
        // If we added a dependence, update the mask 
        if (user.op->add_waiting_dependence(epoch_op, user.idx, user.gen))
        {
          epoch_mask |= user.field_mask;
          epoch_op->increment_dependence_count();
        }
      }
      epoch_op->end_down_sample();
      epoch_op->finish_analysis();
      // Now push the new logical user onto the front of the list of users
      users.push_front(LogicalUser(epoch_op, 0/*idx*/, epoch_mask, RegionUsage(READ_WRITE,EXCLUSIVE,0)));
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_user_list(std::list<LogicalUser> &users)
    //--------------------------------------------------------------------------
    {
      for (std::list<LogicalUser>::iterator it = users.begin();
            it != users.end(); /*nothing*/)
      {
        if (it->op->has_mapped(it->gen))
          it = users.erase(it);
        else
          it++;
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::perform_close_operations(TreeCloser &closer,
        const GenericUser &user, const FieldMask &closing_mask, FieldState &state,
        int next_child, bool allow_next_child, bool upgrade_next_child, bool permit_leave_open,
        std::vector<FieldState> &new_states, FieldMask &already_open)
    //--------------------------------------------------------------------------
    {
      std::vector<Color> to_delete;
      bool success = true;
      bool leave_open = permit_leave_open ? closer.leave_children_open() : false;
      // Go through and close all the children which we overlap with
      // and aren't the next child that we're going to use
      for (std::map<Color,FieldMask>::iterator it = state.open_children.begin();
            it != state.open_children.end(); it++)
      {
        // Check field disjointnes
        if (it->second * closing_mask)
          continue;
        // Check for same child, only allow upgrades in some cases
        // such as read-only -> exclusive.  This is calling context
        // sensitive hence the parameter.
        if (allow_next_child && (next_child >= 0) && (next_child == int(it->first)))
        {
          FieldMask open_users = it->second & closing_mask;
          already_open |= open_users;
          if (upgrade_next_child)
          {
            it->second -= open_users;
            if (!it->second)
              to_delete.push_back(it->first);
          }
          continue;
        }
        // Check for child disjointness 
        if ((next_child >= 0) && are_children_disjoint(it->first, unsigned(next_child)))
          continue;
        // Now we need to close this child 
        FieldMask close_mask = it->second & closing_mask;
        RegionTreeNode *child_node = get_tree_child(it->first);
        // Check to see if the closer is ready to do the close
        if (!closer.closing_state(state))
        {
          success = false;
          break;
        }
        closer.close_tree_node(child_node, close_mask);
        // Remove the close fields
        it->second -= close_mask;
        if (!it->second)
          to_delete.push_back(it->first);
        // If we're allowed to leave this open, add a new state for the current user
        if (leave_open)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(IS_READ_ONLY(user.usage)); // user should be read-only if we're leaving open
#endif
          new_states.push_back(FieldState(user, close_mask, it->first));
        }
      }
      // Remove the children that can be deleted
      for (std::vector<Color>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        state.open_children.erase(*it);
      }
      // Now we need to rebuild the valid fields mask
      FieldMask next_valid;
      for (std::map<Color,FieldMask>::const_iterator it = state.open_children.begin();
            it != state.open_children.end(); it++)
      {
        next_valid |= it->second;
      }
      state.valid_fields = next_valid;

      // Return if the close operation was a success
      return success;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::siphon_open_children(TreeCloser &closer, GenericState &state, 
          const GenericUser &user, const FieldMask &current_mask, int next_child /*= -1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_field_states(state);
#endif
      FieldMask open_mask = current_mask;
      std::vector<FieldState> new_states;

      closer.pre_siphon();

      // Go through and see which partitions we need to close
      for (std::list<FieldState>::iterator it = state.field_states.begin();
            it != state.field_states.end(); /*nothing*/)
      {
        // Check for field disjointness in which case we can continue
        if (it->valid_fields * current_mask)
        {
          it++;
          continue;
        }
        // Now check the state 
        switch (it->open_state)
        {
          case OPEN_READ_ONLY:
            {
              if (IS_READ_ONLY(user.usage))
              {
                // Everything is read-only
                // See if the partition that we want is already open
                if ((next_child >= 0) && 
                    (it->open_children.find(unsigned(next_child)) != it->open_children.end()))
                {
                  // Remove the overlap fields from that partition that
                  // overlap with our own from the open mask
                  open_mask -= (it->open_children[unsigned(next_child)] & current_mask);
                }
                it++;
              }
              else 
              {
                // Not read-only
                // Close up all the open partitions except the one
                // we want to go down, make a new state to be added
                // containing the fields that are still open
                // We need an upgrade if we're transitioning from read-only to some kind of write
                bool needs_upgrade = HAS_WRITE(user.usage);
                FieldMask already_open;
                bool success = perform_close_operations(closer, user, current_mask, *it,
                                                        next_child, true/*allow next child*/, needs_upgrade, 
                                                        false/*permit leave open*/, new_states, already_open);
                if (!success) // make sure the close worked
                  return false;
                // Update the open mask
                open_mask -= already_open;
                if (needs_upgrade)
                  new_states.push_back(FieldState(user, already_open, next_child));
                // See if there are still any valid fields open
                if (!(it->still_valid()))
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          case OPEN_READ_WRITE:
            {
              // Close up any open partitions that conflict with ours
              FieldMask already_open;
              bool success = perform_close_operations(closer, user, current_mask, *it,
                                                      next_child, true/*allow same child*/, false/*needs upgrade*/, 
                                                      IS_READ_ONLY(user.usage)/*permit leave open*/, new_states, already_open);
              if (!success)
                return false;
              // Update the open mask
              open_mask -= already_open;
              // See if this entry is still valid
              if (!(it->still_valid()))
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_SINGLE_REDUCE:
            {
              // Check to see if we have a child we want to go down
              if (next_child >= 0)
              {
                // There are four cases here:
                //   1. Same reduction, same child -> everything stays the same
                //   2. Same reduction, different child -> go to MULTI_REDUCE
                //   3. Diff operation, same child -> go to READ_WRITE
                //   4. Diff operation, diff child -> close everything up
                if (IS_REDUCE(user.usage) && (it->redop == user.usage.redop))
                {
                  std::vector<Color> to_delete;
                  // Go through all the children and see if there is any overlap
                  for (std::map<Color,FieldMask>::iterator cit = it->open_children.begin();
                        cit != it->open_children.end(); cit++)
                  {
                    FieldMask already_open = cit->second & current_mask;
                    // If disjoint, then nothing to do
                    if (!already_open || are_children_disjoint(cit->first, unsigned(next_child)))
                      continue;
                    // Remove the already open fields from this open_mask since
                    // either they are already open for the right child or we're
                    // going to mark them open in a new FieldState.
                    open_mask -= already_open;
                    if (cit->first != unsigned(next_child))
                    {
                      // Different child, so we need to create a new
                      // FieldState in MULTI_REDUCE mode with two children open
                      FieldState new_state(user, already_open, cit->first);
                      // Add the next_child as well
                      new_state.open_children[unsigned(next_child)] = already_open;
                      new_state.open_state = OPEN_MULTI_REDUCE;
                      new_states.push_back(new_state); 
                      
                      // Update the current child and state
                      cit->second -= already_open;
                      if (!cit->second)
                        to_delete.push_back(cit->first);
                      it->valid_fields -= already_open;
                    }
                    // Otherwise everything just stays in SINGLE_REDUCE MODE
                  }
                  // Now delete any open children that have empty valid fields
                  for (std::vector<Color>::const_iterator cit = to_delete.begin();
                        cit != to_delete.end(); cit++)
                  {
                    std::map<Color,FieldMask>::iterator finder = it->open_children.find(*cit);
#ifdef DEBUG_HIGH_LEVEL
                    assert(finder != it->open_children.end());
                    assert(!finder->second);
#endif
                    it->open_children.erase(finder);
                  }
                }
                else
                {
                  FieldMask already_open;
                  bool success = perform_close_operations(closer, user, current_mask, *it,
                                                          next_child, true/*allow same child*/, true/*needs upgrade*/,
                                                          false/*permit leave open*/, new_states, already_open);
                  if (!success)
                    return false;
                  open_mask -= already_open;
                  if (!!already_open)
                  {
                    // Create a new FieldState open in whatever mode is appropriate
                    // based on the usage
                    FieldState new_state(user, already_open, unsigned(next_child));
                    // Note, if it is another reduction in the same child, we actually need to open it
                    // in read-write mode since now there is dirty data below
                    if (IS_REDUCE(user.usage))
                      new_state.open_state = OPEN_READ_WRITE;
                    new_states.push_back(new_state);
                  }
                }
              }
              else
              {
                // Close up all the open children
                FieldMask already_open;
                bool success = perform_close_operations(closer, user, current_mask, *it,
                                                        next_child, true/*allow same child*/, false/*needs upgrade*/,
                                                        false/*permit leave open*/, new_states, already_open);
                if (!success)
                  return false;
                open_mask -= already_open;
              }
              // Now See if the current FieldState is still valid
              if (!(it->still_valid()))
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_MULTI_REDUCE:
            {
              // See if this is a reduction of the same kind
              if (IS_REDUCE(user.usage) && (user.usage.redop == it->redop))
              {
                if (next_child >= 0)
                {
                  std::map<Color,FieldMask>::const_iterator finder = 
                          it->open_children.find(unsigned(next_child));
                  if (finder != it->open_children.end())
                  {
                    // Remove the overlap fields from that partition that
                    // overlap with our own from the open mask
                    FieldMask child_open_mask = finder->second;
                    open_mask -= (child_open_mask & current_mask);
                  }
                }
                it++;
              }
              else
              {
                // Need to close up the open fields since we're going to have to do
                // an open anyway
                FieldMask already_open;
                bool success = perform_close_operations(closer, user, current_mask, *it, 
                                                        next_child, false/*allow same child*/, false/*needs upgrade*/,
                                                        false/*permit leave open*/, new_states, already_open);
#ifdef DEBUG_HIGH_LEVEL
                assert(!already_open); // should never have any fields that are already open
#endif
                if (!success)
                  return false;
                if (!(it->still_valid()))
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          default:
            assert(false);
        }
      }
      // Create a new state for the open mask
      if ((next_child >= 0) && !!open_mask)
        new_states.push_back(FieldState(user, open_mask, next_child));
      // Merge the new field states into the old field states
      merge_new_field_states(state, new_states, true/*add states*/);
        
      closer.post_siphon();

      return (open_mask == current_mask);
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::FieldState::FieldState(void)
      : open_state(NOT_OPEN), redop(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::FieldState::FieldState(const GenericUser &user)
    //--------------------------------------------------------------------------
    {
      redop = 0;
      if (IS_READ_ONLY(user.usage))
        open_state = OPEN_READ_ONLY;
      else if (IS_WRITE(user.usage))
        open_state = OPEN_READ_WRITE;
      else if (IS_REDUCE(user.usage))
      {
        open_state = OPEN_SINGLE_REDUCE;
        redop = user.usage.redop;
      }
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::FieldState::FieldState(const GenericUser &user, const FieldMask &mask, Color next)
    //--------------------------------------------------------------------------
    {
      redop = 0;
      if (IS_READ_ONLY(user.usage))
        open_state = OPEN_READ_ONLY;
      else if (IS_WRITE(user.usage))
        open_state = OPEN_READ_WRITE;
      else if (IS_REDUCE(user.usage))
      {
        open_state = OPEN_SINGLE_REDUCE;
        redop = user.usage.redop;
      }
      valid_fields = mask;
      open_children[next] = mask;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::FieldState::still_valid(void) const
    //--------------------------------------------------------------------------
    {
      return (!open_children.empty() && (!!valid_fields));
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::FieldState::overlap(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      if (redop != rhs.redop)
        return false;
      if (redop == 0)
        return (open_state == rhs.open_state);
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((open_state == OPEN_SINGLE_REDUCE) || (open_state == OPEN_MULTI_REDUCE));
        assert((rhs.open_state == OPEN_SINGLE_REDUCE) || (rhs.open_state == OPEN_MULTI_REDUCE));
#endif
        // Only support merging reduction fields with exactly the same masks
        // which should be single fields for reductions
        return (valid_fields == rhs.valid_fields);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::merge(const FieldState &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields |= rhs.valid_fields;
      for (std::map<Color,FieldMask>::const_iterator it = rhs.open_children.begin();
            it != rhs.open_children.end(); it++)
      {
        if (open_children.find(it->first) == open_children.end())
        {
          open_children[it->first] = it->second;
        }
        else
        {
          open_children[it->first] |= it->second;
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == rhs.redop);
#endif
      if (redop > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!open_children.empty());
#endif
        // For reductions, handle the case where we need to merge reduction modes
        if (open_children.size() == 1)
          open_state = OPEN_SINGLE_REDUCE;
        else
          open_state = OPEN_MULTI_REDUCE;
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::FieldState::upgrade(const FieldState &rhs)
    //--------------------------------------------------------------------------
    {
      // Quick disjointness test to see if we even need to do anything
      if (valid_fields * rhs.valid_fields)
        return true;
#ifdef DEBUG_HIGH_LEVEL
      // This is the only kind of upgrade that we currently support
      assert((this->open_state == OPEN_READ_ONLY) && (rhs.open_state == OPEN_READ_WRITE));
#endif
      // Clean out the fields for which rhs is valid
      clear(rhs.valid_fields);
      return (!!valid_fields); // return whether we still have valid fields
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::clear(const FieldMask &init_mask)
    //--------------------------------------------------------------------------
    {
      valid_fields -= init_mask;
      if (!valid_fields)
      {
        open_children.clear();
      }
      else
      {
        std::vector<Color> to_delete;
        for (std::map<Color,FieldMask>::iterator it = open_children.begin();
              it != open_children.end(); it++)
        {
          it->second -= init_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<Color>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          open_children.erase(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeNode::FieldState::compute_state_size(const FieldMask &pack_mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid_fields * pack_mask));
#endif
      size_t result = 0;
      result += sizeof(open_state);
      result += sizeof(redop);
      result += sizeof(size_t); // number of partitions to pack
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += (sizeof(it->first) + sizeof(it->second));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::pack_physical_state(const FieldMask &pack_mask, Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid_fields * pack_mask));
#endif
      rez.serialize(open_state);
      rez.serialize(redop);
      // find the number of partitions to pack
      size_t num_children = 0;
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_children++;
      }
      rez.serialize(num_children);
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        rez.serialize(it->first);
        FieldMask open_mask = it->second & pack_mask;
        rez.serialize(open_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::unpack_physical_state(Deserializer &derez, unsigned shift /*= 0*/)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(open_state);
      derez.deserialize(redop);
      size_t num_children;
      derez.deserialize(num_children);
      for (unsigned idx = 0; idx < num_children; idx++)
      {
        Color c;
        derez.deserialize(c);
        FieldMask open_mask;
        derez.deserialize(open_mask);
        if (shift > 0)
          open_mask.shift_left(shift);
        open_children[c] = open_mask;
        // Rebuild the valid fields mask as we're doing this
        valid_fields |= open_mask;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::FieldState::print_state(TreeStateLogger *logger, FieldMask capture_mask) const
    //--------------------------------------------------------------------------
    {
      switch (open_state)
      {
        case NOT_OPEN:
          {
            logger->log("Field State: NOT OPEN (%ld)", open_children.size());
            break;
          }
        case OPEN_READ_WRITE:
          {
            logger->log("Field State: OPEN READ WRITE (%ld)", open_children.size());
            break;
          }
        case OPEN_READ_ONLY:
          {
            logger->log("Field State: OPEN READ-ONLY (%ld)", open_children.size());
            break;
          }
        case OPEN_SINGLE_REDUCE:
          {
            logger->log("Field State: OPEN SINGLE REDUCE Mode %d (%ld)", redop, open_children.size());
            break;
          }
        case OPEN_MULTI_REDUCE:
          {
            logger->log("Field State: OPEN MULTI REDUCE Mode %d (%ld)", redop, open_children.size());
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      for (std::map<Color,FieldMask>::const_iterator it = open_children.begin();
            it != open_children.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        char *mask_buffer = overlap.to_string();
        logger->log("Color %d   Mask %s", it->first, mask_buffer);
        free(mask_buffer);
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::PhysicalState::clear_state(const FieldMask &init_mask)
    //--------------------------------------------------------------------------
    {
      for (std::list<FieldState>::iterator it = field_states.begin();
            it != field_states.end(); /*nothing*/)
      {
        it->clear(init_mask);
        if (it->still_valid())
          it++;
        else
          it = field_states.erase(it);
      }
      for (std::list<FieldState>::iterator it = added_states.begin();
            it != added_states.end(); /*nothing*/)
      {
        it->clear(init_mask);
        if (it->still_valid())
          it++;
        else
          it = added_states.erase(it);
      }
      {
        std::vector<InstanceView*> to_delete;
        for (std::map<InstanceView*,FieldMask>::iterator it = valid_views.begin();
              it != valid_views.end(); it++)
        {
          it->second -= init_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<InstanceView*>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          // Remove the reference, we can add it back later if it gets put back on
          (*it)->remove_valid_reference();
          valid_views.erase(*it);
        }
      }
      {
        std::vector<InstanceView*> to_delete;
        for (std::map<InstanceView*,FieldMask>::iterator it = added_views.begin();
              it != added_views.end(); it++)
        {
          it->second -= init_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<InstanceView*>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          added_views.erase(*it);
        }
      }
      dirty_mask -= init_mask;
      top_mask -= init_mask;
    }

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion r, PartitionNode *par, IndexSpaceNode *row_src,
                           FieldSpaceNode *col_src, bool add, RegionTreeForest *ctx)
      : RegionTreeNode(ctx), handle(r), parent(par), 
        row_source(row_src), column_source(col_src), added(add), marked(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------
    {
      // In the future we may want to reclaim region tree IDs here
    }

    //--------------------------------------------------------------------------
    void RegionNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      added = false;
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_child(LogicalPartition handle, PartitionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->row_source->color) == color_map.end());
#endif
      color_map[node->row_source->color] = node;
      valid_map[node->row_source->color] = node;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::has_child(Color c) const
    //--------------------------------------------------------------------------
    {
      std::map<Color,PartitionNode*>::const_iterator finder = color_map.find(c);
      return (finder != color_map.end());
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
      // If it doesn't exist yet, we have to make it
      std::map<Color,PartitionNode*>::const_iterator finder = color_map.find(c);
      if (finder == color_map.end())
      {
        IndexPartNode *index_child = row_source->get_child(c);
        LogicalPartition child_handle(handle.tree_id, index_child->handle, handle.field_space);
        return context->create_node(child_handle, this, true/*add*/);  
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void RegionNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      // only ever remove things from the valid map
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_map.find(c) != valid_map.end());
#endif
      valid_map.erase(c);
    }

    //--------------------------------------------------------------------------
    void RegionNode::initialize_logical_context(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,LogicalState>::iterator finder = logical_states.find(ctx);
      if (finder != logical_states.end())
      {
        LogicalState &state = finder->second;
        state.field_states.clear();
        state.added_states.clear();
        state.curr_epoch_users.clear();
        state.prev_epoch_users.clear();
        // Now do any valid children
        for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->initialize_logical_context(ctx);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::register_deletion_operation(ContextID ctx, DeletionOperation *op,
                                                  const FieldMask &deletion_mask)
    //--------------------------------------------------------------------------
    {
      // If we don't even have a logical state then neither 
      // do any of our children so we're done
      if (logical_states.find(ctx) != logical_states.end())
      {
        const LogicalState &state = logical_states[ctx];
        for (std::list<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
              it != state.curr_epoch_users.end(); it++)
        {
          // Check for field disjointness
          if (it->field_mask * deletion_mask)
            continue;
          op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
        }
        for (std::list<LogicalUser>::const_iterator it = state.prev_epoch_users.begin();
              it != state.prev_epoch_users.end(); it++)
        {
          // Check for field disjointness
          if (it->field_mask * deletion_mask)
            continue;
          op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
        }
      }
      // Do any children
      for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::initialize_physical_context(ContextID ctx, bool clear, const FieldMask &init_mask, bool top)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(ctx);
      if (finder != physical_states.end())
      {
        PhysicalState &state = finder->second;
#ifdef DEBUG_HIGH_LEVEL
        // If this ever fails we have aliasing between physical contexts which will be unresolvable
        //assert(state.context_top == top); 
#endif
        if (clear)
          state.clear_state(init_mask);
        // Put this after clear state since we just reset the bits for the top mask
        if (top)
          state.top_mask |= init_mask;

        // Only do children if we ourselves had a state
        for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->initialize_physical_context(ctx, clear, init_mask);
        }
      }
      else if (top)
      {
        physical_states[ctx] = PhysicalState(init_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::fill_exclusive_context(ContextID ctx, const FieldMask &fill_mask, Color open_child)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      FieldState to_add;
      to_add.valid_fields = fill_mask;
      to_add.open_state = RegionTreeNode::OPEN_READ_WRITE;
      to_add.redop = 0;
      to_add.open_children[open_child] = fill_mask;
      merge_new_field_state(state, to_add, true/*add state*/);
      if (parent != NULL)
      {
        state.top_mask -= fill_mask; 
        parent->fill_exclusive_context(ctx, fill_mask, row_source->color);
      }
      else
      {
        state.top_mask |= fill_mask;
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::merge_physical_context(ContextID outer_ctx, ContextID inner_ctx, const FieldMask &merge_mask)
    //--------------------------------------------------------------------------
    {
      // If there is no inner state then there's nothing to do in this tree
      if (physical_states.find(inner_ctx) == physical_states.find(inner_ctx))
        return;
      PhysicalState &inner_state = physical_states[inner_ctx];
      if (physical_states.find(outer_ctx) == physical_states.find(outer_ctx))
        physical_states[outer_ctx] = PhysicalState();
      PhysicalState &outer_state = physical_states[outer_ctx];
      // No need to explicitly update the dirty mask, that will happen as we update
      // the valid views
      // Update the valid views
      for (std::map<InstanceView*,FieldMask>::const_iterator it = inner_state.valid_views.begin();
            it != inner_state.valid_views.end(); it++)
      {
        FieldMask overlap = it->second & merge_mask;
        if (!overlap)
          continue;
        update_valid_views(outer_ctx, overlap, !!(overlap & inner_state.dirty_mask), it->first);
      }
      // Now do the open children
      for (std::list<FieldState>::const_iterator it = inner_state.field_states.begin();
            it != inner_state.field_states.end(); it++)
      {
        FieldState copy = *it;
        copy.valid_fields &= merge_mask;
        if (!copy.valid_fields)
          continue;
        std::vector<Color> to_delete;
        for (std::map<Color,FieldMask>::iterator cit = copy.open_children.begin();
              cit != copy.open_children.end(); cit++)
        {
          cit->second &= merge_mask;
          if (!cit->second)
            to_delete.push_back(cit->first);
        }
        // Delete any empty children
        for (std::vector<Color>::const_iterator cit = to_delete.begin();
              cit != to_delete.end(); cit++)
        {
          copy.open_children.erase(*cit);
        }
        if (copy.open_children.empty())
          continue;
        // Otherwise add the new state
        merge_new_field_state(outer_state, copy, true/*add states*/);
      }
      // finally continue the traversal
      for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->merge_physical_context(outer_ctx, inner_ctx, merge_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::register_physical_region(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
#endif
      if (physical_states.find(rm.ctx) == physical_states.end())
        physical_states[rm.ctx] = PhysicalState(); 

      PhysicalState &state = physical_states[rm.ctx];

      // Flush any incompatible reductions that don't mesh with the current user
      // e.g. different type of reduction or read/write
      flush_reductions(user, rm);

      if (rm.path.size() == 1)
      {
        // We've arrived
        rm.path.pop_back();
        if (rm.sanitizing)
        {
          // Figure out if we need to siphon the children here.
          // Read-only and reduce need to siphon since they can
          // have many simultaneous mapping operations happening which
          // will need to be merged later.
          if (IS_READ_ONLY(user.usage) || IS_REDUCE(user.usage))
          {
            PhysicalCloser closer(user, rm, this, IS_READ_ONLY(user.usage));
            siphon_open_children(closer, state, user, user.field_mask);
            // Make sure that the close operation succeeded
            if (!closer.success)
            {
              rm.success = false;
              return;
            }
          }

          // If we're sanitizing, get views for all of the regions with
          // valid data and make them valid here, but only if we're not
          // doing a reduction in which case we only have a single view
          // of an instance from any point.
          if (!IS_REDUCE(user.usage))
            pull_valid_views(rm.ctx, user.field_mask);
          rm.success = true;
        }
        else
        {
          // Separate paths for reductions and non-reductions
          if (!IS_REDUCE(user.usage))
          {
            // Map the region if we can
            InstanceView *new_view = map_physical_region(user, rm);
            // Check to see if the mapping was successful, if not we 
            // can just return
            if (new_view == NULL)
            {
              rm.success = false;
              return;
            }
            
            // If we mapped the region close up any partitions below that
            // might have valid data that we need for our instance
            PhysicalCloser closer(user, rm, this, IS_READ_ONLY(user.usage));
            closer.add_upper_target(new_view);
            closer.targets_selected = true;
            // If we're writing mark the dirty bits in the closer
            if (IS_WRITE(user.usage))
              closer.dirty_mask = user.field_mask;
            siphon_open_children(closer, state, user, user.field_mask);
#ifdef DEBUG_HIGH_LEVEL
            assert(closer.success);
            assert(state.valid_views.find(new_view) != state.valid_views.end());
#endif
            // Note that when the siphon operation is done it will automatically
            // update the set of valid instances
            // Now add our user and get the resulting reference back
            rm.result = new_view->add_user(rm.uid, user, rm.task->index_point);
            rm.success = true;
          }
          else
          {
            ReductionView *new_view = map_reduction_region(user, rm);
            if (new_view == NULL)
            {
              rm.success = false;
              return;
            }
            // Still need to close things up, but not to the reduction
            // instance.  Instead we close things up to all the pre-existing
            // valid instances.
            // Get the existing valid instances
            std::list<std::pair<InstanceView*,FieldMask> > valid_views;
            find_valid_instance_views(rm.ctx, valid_views, user.field_mask, 
                                      user.field_mask, true/*needs space*/);
#if 0
            if (valid_views.empty())
            {
              log_inst(LEVEL_ERROR,"Reduction operation %d on region %d of task %s (ID %d) is not fully initialized",
                                    user.usage.redop, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_UNINITIALIZED_REDUCTION);
            }
#endif
            PhysicalCloser closer(user, rm, this, false/*keep open*/);
            if (!valid_views.empty())
            {
              for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it = valid_views.begin();
                    it != valid_views.end(); it++)
              {
                closer.add_upper_target(it->first);
              }
              closer.targets_selected = true;
            }
            // This will update the valid instances with the new views
            siphon_open_children(closer, state, user, user.field_mask);
#ifdef DEBUG_HIGH_LEVEL
            assert(closer.success);
#endif
            // Now update the valid reduction views
            update_reduction_views(rm.ctx, user.field_mask, new_view);
            // Get the result and mark it as a success
            rm.result = new_view->add_user(rm.uid, user);
            rm.success = true;
          }
        }
      }
      else
      {
        // Not there yet, keep going
        rm.path.pop_back();
        Color next_part = rm.path.back();
        // Close up any partitions that might have data that we need
        PhysicalCloser closer(user, rm, this, IS_READ_ONLY(user.usage));
        bool open_only = siphon_open_children(closer, state, user, user.field_mask, next_part);
        // Check to see if we failed the close
        if (!closer.success)
        {
          rm.success = false;
          return;
        }
        PartitionNode *child = get_child(next_part);
        if (open_only)
          child->open_physical_tree(user, rm);
        else
          child->register_physical_region(user, rm);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::open_physical_tree(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
#endif
      if (physical_states.find(rm.ctx) == physical_states.end())
        physical_states[rm.ctx] = PhysicalState();
      PhysicalState &state = physical_states[rm.ctx];
      if (rm.path.size() == 1)
      {
        // We've arrived where we're going
        rm.path.pop_back();
        if (rm.sanitizing)
        {
          // Only find the valid views if we're not doing a reduction
          if (!IS_REDUCE(user.usage))
            pull_valid_views(rm.ctx, user.field_mask);
          rm.success = true;
        }
        else
        {
          if (!IS_REDUCE(user.usage))
          {
            InstanceView *new_view = map_physical_region(user, rm);
            if (new_view == NULL)
            {
              rm.success = false;
              return;
            }

            // No need to perform any close operations since this
            // was an open operation.  Dirty determined by the kind of task
            update_valid_views(rm.ctx, user.field_mask, HAS_WRITE(user.usage), new_view);
            // Add our user and get the reference back
            rm.result = new_view->add_user(rm.uid, user, rm.task->index_point);
            rm.success = true;
          }
          else
          {
            ReductionView *new_view = map_reduction_region(user, rm);
            if (new_view == NULL)
            {
              rm.success = false;
              return;
            }

            // No need to perform any close operations
            update_reduction_views(rm.ctx, user.field_mask, new_view);
            rm.result = new_view->add_user(rm.uid, user);
            rm.success = true;
          }
        }
      }
      else
      {
        rm.path.pop_back();
        Color next_part = rm.path.back();
        // Update the field states
        std::vector<FieldState> new_states;
        new_states.push_back(FieldState(user, user.field_mask, next_part));
        merge_new_field_states(state, new_states, true/*add states*/);
        // Continue the traversal
        PartitionNode *child_node = get_child(next_part);
        child_node->open_physical_tree(user, rm);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(closer.rm.ctx);
      if (finder == physical_states.end())
        return;
      PhysicalState &state = finder->second;
      closer.pre_region(row_source->color);
      // Figure out if we have dirty data.  If we do, issue copies back to
      // each of the target instances specified by the closer.  Note we
      // don't need to issue copies if the target view is already in
      // the list of currently valid views.  Then
      // perform the close operation on each of our open partitions that
      // interfere with the closing mask.
      // If there are any dirty fields we have to copy them back
      FieldMask dirty_fields = state.dirty_mask & closing_mask;
      if (!!dirty_fields)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!state.valid_views.empty());
#endif
        for (std::vector<InstanceView*>::const_iterator it = closer.lower_targets.begin();
              it != closer.lower_targets.end(); it++)
        {
          std::map<InstanceView*,FieldMask>::const_iterator finder = state.valid_views.find(*it);
          // Check to see if it is already a valid instance for some fields
          if (finder == state.valid_views.end())
          {
            issue_update_copies(*it, closer.rm, dirty_fields); 
          }
          else
          {
            // Only need to issue update copies for dirty fields we are not currently valid for 
            FieldMask diff_fields = dirty_fields - finder->second;
            if (!!diff_fields)
              issue_update_copies(*it, closer.rm, diff_fields);
          }
        }
      }
      // Now we need to close up any open children that we have open
      // Create a new closer object corresponding to this node
      PhysicalCloser next_closer(closer, this);
      siphon_open_children(next_closer, state, closer.user, closing_mask);
#ifdef DEBUG_HIGH_LEVEL
      assert(next_closer.success);
#endif
      // Finally appy any reductions that we might have back to the target instances
      FieldMask reduc_fields = state.reduction_mask & closing_mask;
      if (!!reduc_fields)
      {
        // Issue the necessary reductions operations to each of the target instances
        for (std::vector<InstanceView*>::const_iterator it = closer.lower_targets.begin();
              it != closer.lower_targets.end(); it++)
        {
          issue_update_reductions(*it, reduc_fields, closer.rm);
        }
        // Invalidate the reduction views we just reduced back
        invalidate_reduction_views(state, reduc_fields);
      }

      // Need to update was dirty with whether any of our sub-children were dirty
      closer.dirty_mask |= (dirty_fields | reduc_fields | next_closer.dirty_mask);

      if (!closer.leave_open)
        invalidate_instance_views(state, closing_mask, true/*clean*/);
      else // Still clean out the dirty bits since we copied everything back
        state.dirty_mask -= closing_mask; 

      closer.post_region();
    }

    //--------------------------------------------------------------------------
    void RegionNode::close_reduction_tree(ReductionCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(closer.rm.ctx);
      if (finder == physical_states.end())
        return;
      // Issue any reductions to our target instance
      issue_update_reductions(closer.target, closing_mask, closer.rm);
      // Continue the close operation
      PhysicalState &state = finder->second;
      siphon_open_children(closer, state, closer.user, closing_mask);
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.success);
#endif
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_children_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_closing_partition(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_tree_child(Color c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    Color RegionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    bool RegionNode::color_match(Color c)
    //--------------------------------------------------------------------------
    {
      return (c == row_source->color);
    }
#endif

    //--------------------------------------------------------------------------
    void RegionNode::update_valid_views(ContextID ctx, const FieldMask &valid_mask, 
                                        bool dirty, InstanceView *new_view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
      assert(new_view->logical_region == this);
#endif
      PhysicalState &state = physical_states[ctx];
      // Add our reference first in case the new view is also currently in
      // the list of valid views.  We don't want it to be prematurely deleted
      new_view->add_valid_reference();
      if (dirty)
      {
        invalidate_instance_views(state, valid_mask, false/*clean*/);
        state.dirty_mask |= valid_mask;
      }
      FieldMask new_fields;
      std::map<InstanceView*,FieldMask>::iterator finder = state.valid_views.find(new_view);
      if (finder == state.valid_views.end())
      {
        // New valid view, update everything accordingly
        state.valid_views[new_view] = valid_mask;
        new_fields = valid_mask;
      }
      else
      {
        // It already existed update the valid mask
        new_fields = valid_mask - finder->second;
        finder->second |= valid_mask;
        // Remove the reference that we added since it already was referenced
        new_view->remove_valid_reference();
      }
      if (!!new_fields)
      {
        // Also handle this for the added views
        if (state.added_views.find(new_view) == state.added_views.end())
        {
          state.added_views[new_view] = new_fields;
        }
        else
        {
          state.added_views[new_view] |= new_fields;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::update_valid_views(ContextID ctx, const FieldMask &valid_mask,
                const FieldMask &dirty_mask, const std::vector<InstanceView*> &new_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif     
      PhysicalState &state = physical_states[ctx];
      // Add our references first to avoid any premature free operations
      for (std::vector<InstanceView*>::const_iterator it = new_views.begin();
            it != new_views.end(); it++)
      {
        (*it)->add_valid_reference();
      }
      if (!!dirty_mask)
      {
        invalidate_instance_views(state, dirty_mask, false/*clean*/);
        state.dirty_mask |= dirty_mask;
      }
      for (std::vector<InstanceView*>::const_iterator it = new_views.begin();
            it != new_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->logical_region == this);
#endif
        FieldMask new_fields;
        std::map<InstanceView*,FieldMask>::iterator finder = state.valid_views.find(*it);
        if (finder == state.valid_views.end())
        {
          // New valid view, update everything accordingly
          state.valid_views[*it] = valid_mask;
          new_fields = valid_mask;
        }
        else
        {
          // It already existed update the valid mask
          new_fields = valid_mask - finder->second;
          finder->second |= valid_mask;
          // Remove the reference that we added since it already was referenced
          (*it)->remove_valid_reference();
        }
        if (!!new_fields)
        {
          // Also handle this for the added views
          if (state.added_views.find(*it) == state.added_views.end())
          {
            state.added_views[*it] = new_fields;
          }
          else
          {
            state.added_views[*it] |= new_fields;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::update_reduction_views(ContextID ctx, const FieldMask &valid_mask, ReductionView *new_view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif     
      PhysicalState &state = physical_states[ctx];
      std::map<ReductionView*,FieldMask>::iterator finder = state.reduction_views.find(new_view);
      FieldMask new_fields;
      if (finder == state.reduction_views.end())
      {
        new_view->add_valid_reference();
        state.reduction_views[new_view] = valid_mask;
        new_fields = valid_mask;
      }
      else
      {
        new_fields = valid_mask - finder->second;
        finder->second |= valid_mask;
      }
      state.reduction_mask |= valid_mask;
      if (!!new_fields)
      {
        if (state.added_reductions.find(new_view) == state.added_reductions.end())
        {
          state.added_reductions[new_view] = new_fields;
        }
        else
        {
          state.added_reductions[new_view] |= new_fields;
        }
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionNode::map_physical_region(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
      // Get the list of valid regions for fields we want to use 
      std::list<std::pair<InstanceView*,FieldMask> > valid_instances;
      find_valid_instance_views(rm.ctx, valid_instances, user.field_mask, user.field_mask, true/*needs space*/);
      // Ask the mapper for the list of memories of where to create the instance
      std::map<Memory,bool> valid_memories;
      for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        Memory m = it->first->get_location();
        if (valid_memories.find(m) == valid_memories.end())
          valid_memories[m] = !(user.field_mask - it->second);
        else if (!valid_memories[m])
          valid_memories[m] = !(user.field_mask - it->second);
        // Otherwise we already have an instance in this memory that
        // dominates all the fields in which case we don't care
      }
      // Ask the mapper what to do
      std::vector<Memory> chosen_order;
      std::set<FieldID> additional_fields;
      bool enable_WAR = false;
      bool notify_result = false;
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        notify_result = rm.mapper->map_task_region(rm.task, rm.target, rm.tag, rm.inline_mapping, (!rm.target.exists()),
                                   rm.req, rm.idx, valid_memories, chosen_order, additional_fields, enable_WAR);
      }
      // Filter out any memories that are not visible from the target processor
      // if there is a processor that we're targeting (e.g. never do this for premaps)
      if (!chosen_order.empty() && rm.target.exists())
      {
        Machine *machine = Machine::get_machine();
        const std::set<Memory> &visible_memories = machine->get_visible_memories(rm.target);
        std::vector<Memory> filtered_memories;
        filtered_memories.reserve(chosen_order.size());
        for (std::vector<Memory>::const_iterator it = chosen_order.begin();
              it != chosen_order.end(); it++)
        {
          if (visible_memories.find(*it) != visible_memories.end())
            filtered_memories.push_back(*it);
          else
          {
            log_region(LEVEL_WARNING,"WARNING: Mapper specified memory %x which is not visible from processor %x when mapping "
                                      "region %d of task %s (ID %d)!  Removing memory from the chosen ordering!", 
                                      it->id, rm.target.id, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id());
          }
        }
        chosen_order = filtered_memories;
      }
      if (valid_memories.empty() && chosen_order.empty())
      {
        log_region(LEVEL_ERROR,"Illegal mapper output, no memories specified for first instance of region %d of task %s (ID %d)",
                                rm.idx, rm.task->variants->name, rm.task->get_unique_task_id());
        assert(false);
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      // Check to see if the mapper requested any additional fields in this
      // instance.  If it did, then re-run the computation to get the list
      // of valid instances with the right set of fields
      std::vector<FieldID> new_fields = rm.req.instance_fields;
      if (!additional_fields.empty())
      {
        FieldSpaceNode *field_node = context->get_node(rm.req.region.field_space);
        // Update the list of new_fields that will be needed and get the mask
        // including the original fields with the instance fields
        new_fields.insert(new_fields.end(),additional_fields.begin(),additional_fields.end());
        // Get the additional field mask
        FieldMask additional_mask = field_node->get_field_mask(new_fields);
        // Now recompute the set of available instances
        valid_instances.clear();
        find_valid_instance_views(rm.ctx, valid_instances, additional_mask, additional_mask, true/*needs space*/);
        valid_memories.clear();
        // Recompute the set of valid memories to help our search, we've already filtered
        // for the additional fields, they don't need to be valid fields as well.
        for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
               valid_instances.begin(); it != valid_instances.end(); it++)
        {
          Memory m = it->first->get_location();
          if (valid_memories.find(m) == valid_memories.end())
            valid_memories[m] = !(user.field_mask - it->second);
          else if (!valid_memories[m])
            valid_memories[m] = !(user.field_mask - it->second);
          // Otherwise we already have an instance in this memory that
          // dominates all the fields in which case we don't care
        } 
      }
      InstanceView *result = NULL;
      FieldMask needed_fields; 
      // Go through each of the memories provided by the mapper
      for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
            mit != chosen_order.end(); mit++)
      {
        // See if it has any valid instances
        if (valid_memories.find(*mit) != valid_memories.end())
        {
          // Already have a valid instance with at least a few valid fields, figure
          // out if it has all or some of the fields valid
          if (valid_memories[*mit])
          {
            // We've got an instance with all the valid fields, go find it
            for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              if (it->first->get_location() != (*mit))
                continue;
              if (!(user.field_mask - it->second))
              {
                // Check to see if have any WAR dependences
                // in which case we'll skip it for a something better
                if (enable_WAR && HAS_WRITE(rm.req) && it->first->has_war_dependence(user))
                  continue;
                // No WAR problems, so it it is good
                result = it->first;
                // No need to set needed fields since everything is valid
                break;
              }
            }
            // If we found a good instance break, otherwise go onto
            // the partial instances
            if (result != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!needed_fields);
#endif
              break;
            }
          }
          // Do this if we couldn't find a better choice
          // Note we can't do this in the read-only case because we might end up issuing
          // multiple copies to the same location.
          if (!IS_READ_ONLY(user.usage))
          {
            // These are instances which have space for all the required fields
            // but only a subset of those fields contain valid data.
            // Find the valid instance with the most valid fields to use.
            int covered_fields = -1;
            for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              if (it->first->get_location() != (*mit))
                continue;
              int cf = FieldMask::pop_count(it->second);
              if (cf > covered_fields)
              {
                // Check to see if we have any WAR dependences which might disqualify us
                if (enable_WAR && HAS_WRITE(rm.req) && it->first->has_war_dependence(user))
                  continue;
                covered_fields = cf;
                result = it->first;
                needed_fields = user.field_mask - it->second; 
              }
            }
            // If we got a good one break out, otherwise we'll try to make a new instance
            if (result != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!!needed_fields);
#endif
              break;
            }
          }
        }
        // If it didn't find a valid instance, try to make one
        result = create_instance(*mit, rm, new_fields); 
        if (result != NULL)
        {
          // We successfully made an instance
          needed_fields = user.field_mask;
          break;
        }
      }
      // Figure out if successfully got an instance that we needed
      // and we still need to issue any copies to get up to date data
      // for any fields and this isn't a write-only region
      if (result != NULL && !!needed_fields && (user.usage.privilege != WRITE_ONLY))
        issue_update_copies(result, rm, needed_fields); 

      // If the mapper asked to be notified of the result, tell it
      // Note we only need to tell if it succeeded, otherwise it will
      // get told by the notify_failed_mapping call.
      if (notify_result && (result != NULL))
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        rm.mapper->notify_mapping_result(rm.task, rm.target, rm.req, rm.idx, rm.inline_mapping,
                                          result->get_manager()->get_location());
      }

      return result;
    }

    //--------------------------------------------------------------------------
    ReductionView* RegionNode::map_reduction_region(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(user.usage.redop != 0);
      assert(user.usage.redop == rm.req.redop);
#endif
      // Get the list of valid reduction instances we can use
      std::list<ReductionView*> valid_views;
      find_valid_reduction_views(rm.ctx, valid_views, user.field_mask); 

      // Ask the mapper for the list of memories of where to create the instance
      std::map<Memory,bool> valid_memories;
      for (std::list<ReductionView*>::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        Memory m = (*it)->get_location();
        valid_memories[m] = true; // always valid instance for all fields since this is a reduction
      }
      // Ask the mapper what to do
      std::vector<Memory> chosen_order;
      std::set<FieldID> additional_fields;
      bool enable_WAR = false;
      bool notify_result = false;
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        notify_result = rm.mapper->map_task_region(rm.task, rm.target, rm.tag, rm.inline_mapping, (!rm.target.exists()),
                                   rm.req, rm.idx, valid_memories, chosen_order, additional_fields, enable_WAR);
      }
      if (!chosen_order.empty() && rm.target.exists())
      {
        Machine *machine = Machine::get_machine();
        const std::set<Memory> &visible_memories = machine->get_visible_memories(rm.target);
        std::vector<Memory> filtered_memories;
        filtered_memories.reserve(chosen_order.size());
        for (std::vector<Memory>::const_iterator it = chosen_order.begin();
              it != chosen_order.end(); it++)
        {
          if (visible_memories.find(*it) != visible_memories.end())
            filtered_memories.push_back(*it);
          else
          {
            log_region(LEVEL_WARNING,"WARNING: Mapper specified memory %x which is not visible from processor %x when mapping "
                                      "region %d of task %s (ID %d)!  Removing memory from the chosen ordering!", 
                                      it->id, rm.target.id, rm.idx, rm.task->variants->name, rm.task->get_unique_task_id());
          }
        }
        chosen_order = filtered_memories;
      }
      if (valid_memories.empty() && chosen_order.empty())
      {
        log_region(LEVEL_ERROR,"Illegal mapper output, no memories specified for first instance of region %d of task %s (ID %d)",
                                rm.idx, rm.task->variants->name, rm.task->get_unique_task_id());
        assert(false);
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      ReductionView *result = NULL;
      // Go through each of the valid memories and see if we can either find
      // a reduction instance or we can make one
      for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
            mit != chosen_order.end(); mit++)
      {
        if (valid_memories.find(*mit) != valid_memories.end())
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(valid_memories[*mit]);
#endif
          // We've got a valid instance, let's go find it
          for (std::list<ReductionView*>::const_iterator it = valid_views.begin();
                it != valid_views.end(); it++)
          {
            if ((*it)->get_location() == *mit)
            {
              result = *it;
              break;
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(result != NULL);
#endif
          // We've found the instance that we want
          break;
        }
        else
        {
          // Try making a reduction instance in this memory
          result = create_reduction(*mit, rm);
          if (result != NULL)
            break;
        }
      }

      // If the mapper asked to be notified of the result, tell it
      // Note we only need to tell if it succeeded, otherwise it will
      // get told by the notify_failed_mapping call.
      if (notify_result && (result != NULL))
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        rm.mapper->notify_mapping_result(rm.task, rm.target, rm.req, rm.idx, rm.inline_mapping,
                                          result->get_manager()->get_location());
      }

      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::issue_update_copies(InstanceView *dst, RegionMapper &rm, FieldMask copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!!copy_mask);
      assert(dst->logical_region == this);
#endif
      // Get the list of valid regions for all the fields we need to do the copy for
      std::list<std::pair<InstanceView*,FieldMask> > valid_instances;
      find_valid_instance_views(rm.ctx, valid_instances, copy_mask, copy_mask, false/*needs space*/);

      // To facilitate optimized copies in the low-level runtime, we gather all the information
      // needed to issue gather copies from multiple instances into the data structures below,
      // we then issue the copy when we're done and update the destination instance.
      std::set<Event> preconditions;
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<std::pair<InstanceView*,FieldMask> > src_instances;
      std::vector<Domain::CopySrcDstField> dst_fields;

      // No need to call the mapper if there is only one valid instance
      if (valid_instances.size() == 1)
      {
        FieldMask op_mask = copy_mask & valid_instances.back().second;
        if (!!op_mask)
        {
          InstanceView *src = valid_instances.back().first;
          // No need to do anything if src and destination are the same
          if (src != dst)
          {
            src->copy_from(rm, op_mask, preconditions, src_fields);
            dst->copy_to(rm, op_mask, preconditions, dst_fields);
            src_instances.push_back(valid_instances.back());
          }
        }
      }
      else if (!valid_instances.empty())
      {
        bool copy_ready = false;
        // Ask the mapper to put everything in order
        std::set<Memory> available_memories;
        for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          available_memories.insert(it->first->get_location());  
        }
        std::vector<Memory> chosen_order;
        {
          DetailedTimer::ScopedPush sp(TIME_MAPPER);
          AutoLock m_lock(rm.mapper_lock);
          rm.mapper->rank_copy_sources(available_memories, dst->get_location(), chosen_order);
        }
        for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
              !copy_ready && mit != chosen_order.end(); mit++)
        {
          available_memories.erase(*mit); 
          // Go through all the valid instances and issue copies from instances
          // in the given memory
          for (std::list<std::pair<InstanceView*,FieldMask> >::iterator it = valid_instances.begin();
                it != valid_instances.end(); /*nothing*/)
          {
            if ((*mit) != it->first->get_location())
            {
              it++;
              continue;
            }
            // Check to see if there are valid fields in the copy mask
            FieldMask op_mask = copy_mask & it->second;
            if (!!op_mask)
            {
              // No need to do anything if they are the same instance
              if (dst != it->first)
              {
                it->first->copy_from(rm, op_mask, preconditions, src_fields);
                dst->copy_to(rm, op_mask, preconditions, dst_fields);
                src_instances.push_back(std::pair<InstanceView*,FieldMask>(it->first, op_mask));
              }
              // update the copy mask
              copy_mask -= op_mask;
              // Check for the fast out
              if (!copy_mask)
              {
                copy_ready = true;
                break;
              }
              // Issue the copy, so no longer need to consider it
              it = valid_instances.erase(it);
            }
            else
              it++;
          }
        }
        // Now do any remaining memories not handled by the mapper in some order
        for (std::set<Memory>::const_iterator mit = available_memories.begin();
              !copy_ready && mit != available_memories.end(); mit++)
        {
          for (std::list<std::pair<InstanceView*,FieldMask> >::iterator it = valid_instances.begin();
                it != valid_instances.end(); /*nothing*/)
          {
            if ((*mit) != it->first->get_location())
            {
              it++;
              continue;
            }
            // Check to see if there are valid fields in the copy mask
            FieldMask op_mask = copy_mask & it->second;
            if (!!op_mask)
            {
              if (dst != it->first)
              {
                it->first->copy_from(rm, op_mask, preconditions, src_fields);
                dst->copy_to(rm, op_mask, preconditions, dst_fields);
                src_instances.push_back(std::pair<InstanceView*,FieldMask>(it->first, op_mask));
              }
              // update the copy mask
              copy_mask -= op_mask;
              // Check for the fast out
              if (!copy_mask)
              {
                copy_ready = true;
                break;
              }
              // Issue the copy, so no longer need to consider it
              it = valid_instances.erase(it);
            }
            else
              it++;
          }
        }
      }
      // Otherwise there were no valid instances so this is a valid copy

      // Now issue the copy operation to the low-level runtime
      if (!src_fields.empty())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!dst_fields.empty());
        assert(src_fields.size() == dst_fields.size());
#endif
        Event copy_pre = Event::merge_events(preconditions);
#ifdef LEGION_SPY
        if (!copy_pre.exists())
        {
          UserEvent new_copy_pre = UserEvent::create_user_event();
          new_copy_pre.trigger();
          copy_pre = new_copy_pre;
        }
        LegionSpy::log_event_dependences(preconditions, copy_pre);
#endif
        Event copy_post = row_source->domain.copy(src_fields, dst_fields, copy_pre);
        // If we have a post-copy event, add the necessary references
        if (copy_post.exists())
        {
          for (std::vector<std::pair<InstanceView*,FieldMask> >::const_iterator it = src_instances.begin();
                it != src_instances.end(); it++)
          {
            rm.source_copy_instances.push_back(it->first->add_copy_user(0/*redop*/, copy_post, it->second, true/*reading*/));
          }
        }
#ifdef LEGION_SPY
        if (!copy_post.exists())
        {
          UserEvent new_copy_post = UserEvent::create_user_event();
          new_copy_post.trigger();
          copy_post = new_copy_post;
          // If we're doing legion spy add the references anyway
          for (std::vector<std::pair<InstanceView*,FieldMask> >::const_iterator it = src_instances.begin();
                it != src_instances.end(); it++)
          {
            rm.source_copy_instances.push_back(it->first->add_copy_user(0/*redop*/, copy_post, it->second, true/*reading*/));
          }
        }
#endif
        FieldMask update_mask;
        for (std::vector<std::pair<InstanceView*,FieldMask> >::const_iterator it = src_instances.begin();
              it != src_instances.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!!it->second);
#endif
          update_mask |= it->second;
#ifdef LEGION_SPY
          {
            FieldSpaceNode *field_node = context->get_node(handle.field_space);
            char *string_mask = field_node->to_string(it->second);
            LegionSpy::log_copy_operation(it->first->manager->get_unique_id(), dst->manager->get_unique_id(),
                                          handle.index_space.id, handle.field_space.id, handle.tree_id,
                                          copy_pre, copy_post, string_mask);
            free(string_mask);
          }
#endif
        }
        // Now update the valid event for the fields that were the target of the copy
        dst->update_valid_event(copy_post, update_mask);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::issue_update_reductions(PhysicalView *target, const FieldMask &mask, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[rm.ctx];
      // Go through all of our reduction instances and issue reductions to
      // the target instance
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
            it != state.reduction_views.end(); it++)
      {
        FieldMask copy_mask = mask & it->second;
        if (!!copy_mask)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!(it->second - copy_mask)); // all fields in the reduction instance should be used
#endif
          // Then we have a reduction to perform
          it->first->perform_reduction(target, copy_mask, rm);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::mark_invalid_instance_views(ContextID ctx, const FieldMask &invalid_mask, bool recurse)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(ctx);
      if (finder != physical_states.end())
      {
        PhysicalState &state = finder->second;
        for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
              it != state.valid_views.end(); it++)
        {
          FieldMask diff = it->second - invalid_mask;
          // only mark it as to be invalidated if all the fields will no longer be valid
          if (!diff)
            it->first->mark_to_be_invalidated();
        }

        if (recurse)
        {
          for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
                it != valid_map.end(); it++)
          {
            it->second->mark_invalid_instance_views(ctx, invalid_mask, recurse);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::recursive_invalidate_views(ContextID ctx, const FieldMask &invalid_mask, bool last_use)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(ctx);
      if (finder != physical_states.end())
      {
        
        // If it's the last use we can delete the context
        if (last_use)
        {
          finder->second.clear_state(FieldMask(FIELD_ALL_ONES));
          physical_states.erase(finder);
        }
        else
        {
          // Always need to correctly invalidate the views so we don't
          // accidentally leak any references to valid views
          invalidate_instance_views(finder->second, invalid_mask, false/*clean*/);
          invalidate_reduction_views(finder->second, invalid_mask);
        }
        for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->recursive_invalidate_views(ctx, invalid_mask, last_use);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::invalidate_instance_views(PhysicalState &state, const FieldMask &invalid_mask, bool clean)
    //--------------------------------------------------------------------------
    {
      std::vector<InstanceView*> to_delete;
      for (std::map<InstanceView*,FieldMask>::iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<InstanceView*>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        (*it)->remove_valid_reference();
        state.valid_views.erase(*it);
      }
      // Note we don't need to worry about invalidating the added_views map
      // because we only use added_views for sending back things that can
      // be merged, and therefore they are monotonically increasing in state
      // and will never call invalidate instance views.
      if (clean)
        state.dirty_mask -= invalid_mask;
    }

    //--------------------------------------------------------------------------
    void RegionNode::invalidate_reduction_views(PhysicalState &state, const FieldMask &invalid_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<ReductionView*> to_delete;
      for (std::map<ReductionView*,FieldMask>::iterator it = state.reduction_views.begin();
            it != state.reduction_views.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<ReductionView*>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        (*it)->remove_valid_reference();
        state.reduction_views.erase(*it);
      }
      state.reduction_mask -= invalid_mask;
      // See note above in invalidate_instance_views about why we don't have to
      // invalidate the added_reductions map.
    }

    //--------------------------------------------------------------------------
    void RegionNode::find_valid_instance_views(ContextID ctx, 
            std::list<std::pair<InstanceView*,FieldMask> > &valid_views,
            const FieldMask &valid_mask, const FieldMask &field_mask, bool needs_space)             
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      // If we can go up the tree, go up first
      FieldMask up_mask = valid_mask - (state.dirty_mask | state.top_mask);
      if (!!up_mask && (parent != NULL))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent->parent != NULL);
#endif
        parent->parent->find_valid_instance_views(ctx, valid_views, 
                                        up_mask, field_mask, needs_space);
        // Convert everything coming back down
        const Color rp = parent->row_source->color;
        const Color rc = row_source->color;
        for (std::list<std::pair<InstanceView*,FieldMask> >::iterator it =
              valid_views.begin(); it != valid_views.end(); it++)
        {
          it->first = it->first->get_subview(rp,rc);
#ifdef DEBUG_HIGH_LEVEL
          assert(it->first->logical_region == this);
#endif
        }
      }
      // Now figure out which of our valid views we can add
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        // If we need the physical instance to be at least as big as
        // the needed fields, check that first
        if (needs_space && !!(field_mask - it->first->get_physical_mask()))
          continue;
        // If we're looking for instances with space we want the instances
        // even if they have no valid fields, otherwise if we're not looking
        // for instances with enough space, we can exit out early if
        // they don't have any valid fields.
        FieldMask overlap = valid_mask & it->second;
        if (!needs_space && !overlap)
          continue;
#ifdef DEBUG_HIGH_LEVEL
        assert(it->first->logical_region == this);
#endif
        valid_views.push_back(std::pair<InstanceView*,FieldMask>(it->first,overlap));
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::find_valid_reduction_views(ContextID ctx, 
          std::list<ReductionView*> &valid_views, const FieldMask &valid_mask)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      // See if we can continue going up the tree
      if ((state.dirty_mask * valid_mask) && (state.top_mask * valid_mask))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent != NULL);
        assert(parent->parent != NULL);
#endif
        parent->parent->find_valid_reduction_views(ctx, valid_views, valid_mask);
      }
      // Now figure out which of our valid views we can add
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
            it != state.reduction_views.end(); it++)
      {
        FieldMask uncovered = valid_mask - it->second;
        // If the fields we need are fully covered, than this is a valid reduction
        if (!uncovered)
        {
          valid_views.push_back(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* RegionNode::create_instance(Memory location, RegionMapper &rm, const std::vector<FieldID> &new_fields) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.req.instance_fields.empty());
#endif
      // Ask the mapper what the blocking factor should be
      // Find the maximum value that can be returned
      size_t blocking_factor = handle.index_space.get_valid_mask().get_num_elmts();
      // Only need to do this if there is more than one field
      if (rm.req.instance_fields.size() > 1);
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        blocking_factor = rm.mapper->select_region_layout(rm.task, rm.target, rm.req, 
                                                          rm.idx, location, blocking_factor);
      }
      FieldSpaceNode *field_node = context->get_node(handle.field_space);
      // Now get the field Mask and see if we can make the instance
      InstanceManager *manager = field_node->create_instance(location, row_source->domain, 
                                                              new_fields, blocking_factor);
      // See if we made the instance
      InstanceView *result = NULL;
      if (manager != NULL)
      {
        // Made the instance, now make a view for it from this region
        result = context->create_instance_view(manager, NULL/*no parent*/, this, true/*make local*/);
#ifdef DEBUG_HIGH_LEVEL
        assert(result != NULL); // should never happen
#endif
#ifdef LEGION_SPY
        LegionSpy::log_physical_instance(manager->get_instance().id, location.id, handle.index_space.id,
                                          handle.field_space.id, handle.tree_id);
        LegionSpy::log_instance_manager(manager->get_instance().id, manager->get_unique_id());
#endif
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionView* RegionNode::create_reduction(Memory location, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.req.instance_fields.empty());
      assert(rm.req.redop > 0);
#endif
      // Find the reduction operation for this instance
      const ReductionOp *op = HighLevelRuntime::get_reduction_op(rm.req.redop);
      // Check to see if this reduction operation is foldable, if it is, ask the
      // mapper which kind of reduction instance it would like to make
      bool reduction_list = true; // default if there is no fold function
      if (op->is_foldable)
      {
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        AutoLock m_lock(rm.mapper_lock);
        reduction_list = rm.mapper->select_reduction_layout(rm.task, rm.target, rm.req,
                                                            rm.idx, location);
      }
      ReductionManager *manager = NULL;
      if (reduction_list)
      {
        // We need a new index space for handling the sparse reductions
        // TODO: allow users to specify the max number of reductions.  Otherwise
        // for right now we'll just over approximate with the number of elements
        // in the handle index space since ideally reduction lists are sparse
        // and will have less than one reduction per point.
        Domain ptr_space = Domain(IndexSpace::create_index_space(handle.index_space.get_valid_mask().get_num_elmts()));
        std::vector<size_t> element_sizes;
        element_sizes.push_back(sizeof(ptr_t)); // pointer types
        element_sizes.push_back(op->sizeof_rhs);
        // Don't give the reduction op here since this is a list instance and we
        // don't want to initialize any of the fields
        PhysicalInstance inst = ptr_space.create_instance(location, element_sizes, 1/*true list*/);
        if (inst.exists())
        {
          manager = context->create_reduction_manager(location, inst, rm.req.redop, op,
                                          false/*remote*/, false/*clone*/, ptr_space);
#ifdef LEGION_PROF
          {
            std::map<FieldID,size_t> inst_fields;
            inst_fields[0] = op->sizeof_rhs;
            LegionProf::register_instance_creation(inst.id, manager->get_unique_id(), location.id,
                rm.req.redop, 1/*blocking size*/, inst_fields);
          }
#endif
        }
      }
      else
      {
        // Easy case, there is only one field and it is the size of the RHS of the reduction
        // operation.  Note this is true regardless of how many fields are required of the instance.
        PhysicalInstance inst = row_source->domain.create_instance(location, op->sizeof_rhs, rm.req.redop);
        // Make the new manager
        if (inst.exists())
        {
          manager = context->create_reduction_manager(location, inst, rm.req.redop, op, 
                                          false/*remote*/, false/*clone*/); 
#ifdef LEGION_PROF
          {
            std::map<FieldID,size_t> inst_fields;
            inst_fields[0] = op->sizeof_rhs;
            LegionProf::register_instance_creation(inst.id, manager->get_unique_id(), location.id,
                rm.req.redop, 0/*blocking size*/, inst_fields);
          }
#endif
        }
      }
      ReductionView *result = NULL;
      if (manager != NULL)
      {
        result = context->create_reduction_view(manager, this, true/*made local*/);
#ifdef DEBUG_HIGH_LEVEL
        assert(result != NULL); // should never happen
#endif
#ifdef LEGION_SPY
        LegionSpy::log_physical_reduction(manager->get_instance().id, location.id, handle.index_space.id,
                                          handle.field_space.id, handle.tree_id, manager->is_foldable(),
                                          manager->get_pointer_space().exists() ? manager->get_pointer_space().get_index_space().id : 0);
        LegionSpy::log_reduction_manager(manager->get_instance().id, manager->get_unique_id());
#endif
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::issue_final_close_operation(RegionMapper &rm, const PhysicalUser &user, PhysicalCloser &closer)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[rm.ctx];
      // First copy any dirty data from instances at this level into the target instances
      // since they are the only regions that will remain live after this task is finished.
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.targets_selected);
#endif
      for (std::vector<InstanceView*>::const_iterator it = closer.upper_targets.begin();
            it != closer.upper_targets.end(); it++)
      {
        // Figure out which fields in the target overlap with the dirty data in this state
        FieldMask dirty_overlap = user.field_mask & state.dirty_mask; 
        // Remove any fields for which the target instance is already valid
        std::map<InstanceView*,FieldMask>::const_iterator finder = state.valid_views.find(*it);
        if (finder != state.valid_views.end())
          dirty_overlap -= finder->second;
        // If there are no fields that are dirty than we are done
        if (!dirty_overlap)
          continue;
        // Otherwise issue the copies to update the fields
        issue_update_copies(*it, rm, dirty_overlap);
      }
      // Now close up any open subtrees to the set of target regions
      siphon_open_children(closer, state, user, user.field_mask);
    }

    //--------------------------------------------------------------------------
    void RegionNode::issue_final_reduction_operation(RegionMapper &rm, const PhysicalUser &user, ReductionCloser &closer)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[rm.ctx];
      // Then issue all the update reductions from the current reduction instances to the target instance
      issue_update_reductions(closer.target, state.reduction_mask, rm);
      siphon_open_children(closer, state, user, user.field_mask);
    }

    //--------------------------------------------------------------------------
    void RegionNode::pull_valid_views(ContextID ctx, const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
      std::list<std::pair<InstanceView*,FieldMask> > new_valid_views;
      find_valid_instance_views(ctx, new_valid_views, field_mask, field_mask, false/*needs space*/);
      for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
            new_valid_views.begin(); it != new_valid_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->first->logical_region == this);
#endif
        update_valid_views(ctx, it->second, false/*dirty*/, it->first);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::flush_reductions(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(rm.ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[rm.ctx];
      // Go through the list of reduction views and see if there are any
      // that don't mesh with current user and therefore need to be flushed
      FieldMask flush_mask;
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
            it != state.reduction_views.end(); it++)
      {
        // Skip reductions that have already been flushed
        if (!(it->second - flush_mask))
          continue;
        FieldMask reduc_mask = user.field_mask & it->second;
        if (!!reduc_mask)
        {
         // Get the list of valid physical instances for which we need to issue reductions
          std::list<std::pair<InstanceView*,FieldMask> > valid_views;
          find_valid_instance_views(rm.ctx, valid_views, reduc_mask, reduc_mask, true/*need space*/);
          // Go through the list of valid instances and find the ones that have all
          // the required fields for the reduction mask
          std::vector<InstanceView*> update_views;
          for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it = valid_views.begin();
                it != valid_views.end(); it++)
          {
            FieldMask remaining = reduc_mask - it->second;
            if (!remaining)
            {
              issue_update_reductions(it->first, reduc_mask, rm);
              // Add it to the list of updated views
              update_views.push_back(it->first);
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(!update_views.empty()); // should have issued our updates to some place
          // Note this might not be the case if none of our targets all the needed fields
          // A TODO in the future might be to fix this if we ever encounter it
#endif
          // Update the valid instance views, mark that they are dirty since we
          // performed reductions to the flush fields
          update_valid_views(rm.ctx, reduc_mask, reduc_mask, update_views);
          // We'll invalidate the reduction views after we've done
          // iterating over all the reduction views since we don't 
          // want to mess up our iterator here.
          // Now we also have to invalidate the reduction views that we just flushed
          flush_mask |= reduc_mask;
        }
      }
      if (!!flush_mask)
        invalidate_reduction_views(state, flush_mask);
    } 

    //--------------------------------------------------------------------------
    void RegionNode::update_top_mask(FieldMask allocated_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent == NULL); // this method should only be called on top nodes
#endif
      // Iterate over all the contexts and update their top masks
      for (std::map<ContextID,PhysicalState>::iterator it = physical_states.begin();
            it != physical_states.end(); it++)
      {
        it->second.top_mask |= allocated_mask;
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(bool);
      result += sizeof(handle);
      if (returning || marked)
      {
        result += sizeof(size_t); // number of children
        for (std::map<Color,PartitionNode*>::const_iterator it = 
              valid_map.begin(); it != valid_map.end(); it++)
        {
          result += it->second->compute_tree_size(returning);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(handle);
        rez.serialize(valid_map.size());
        for (std::map<Color,PartitionNode*>::const_iterator it =
              valid_map.begin(); it != valid_map.end(); it++)
        {
          it->second->serialize_tree(rez, returning);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
        rez.serialize(handle);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ RegionNode* RegionNode::deserialize_tree(Deserializer &derez, PartitionNode *parent,
                                        RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool needs_unpack;
      derez.deserialize(needs_unpack);
      LogicalRegion handle;
      derez.deserialize(handle);
      if (needs_unpack)
      {
        RegionNode *result = context->create_node(handle, parent, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          PartitionNode::deserialize_tree(derez, result, context, returning); 
        }
        return result;
      }
      else
      {
        return context->get_node(handle);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked);
#endif
      if ((parent == NULL) || (!parent->marked))
        return const_cast<RegionNode*>(this);
      return parent->find_top_marked();
    }

    //--------------------------------------------------------------------------
    void RegionNode::find_new_partitions(std::vector<PartitionNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!added); // shouldn't be here if this is true
#endif
      for (std::map<Color,PartitionNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                          std::set<InstanceManager*> &unique_managers,
                                          std::set<ReductionManager*> &unique_reductions,
                                          bool mark_invalid_views, bool recurse) 
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      result += sizeof(state.dirty_mask);
      result += sizeof(size_t); // number of valid views
      // Find the InstanceViews that need to be sent
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(InstanceKey);
        result += sizeof(it->second);
        unique_managers.insert(it->first->get_instance_manager());
        if (mark_invalid_views && !(it->second - pack_mask))
          it->first->mark_to_be_invalidated();
      }
      result += sizeof(state.reduction_mask);
      result += sizeof(size_t); // number of reduction views
      // Find the ReductionViews that need to be sent
      // Note we currently don't go up the tree to find reduction views, so there
      // may be some valid reduction views that are not visible when mapped remotely.
      // It seems like the cost of implementing this is high relative to the minimal
      // cost of creating a new reduction instance which can be merged easily with
      // other reduction instances.
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
            it != state.reduction_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(InstanceKey);
        result += sizeof(it->second);
        unique_reductions.insert(it->first->manager);
        if (mark_invalid_views && !(it->second - pack_mask))
          it->first->mark_to_be_invalidated();
      }
      result += sizeof(size_t); // number of open partitions
      // Now go through and find any FieldStates that need to be sent
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        result += it->compute_state_size(pack_mask);
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_state_size(ctx, overlap, 
                          unique_managers, unique_reductions, mark_invalid_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::pack_physical_state(ContextID ctx, const FieldMask &pack_mask,
                                          Serializer &rez, bool invalidate_views, bool recurse) 
    //--------------------------------------------------------------------------
    {
      PhysicalState &state = physical_states[ctx];
      // count the number of valid views
      size_t num_valid_views = 0;
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_valid_views++;
      }
      // pack the valid views
      FieldMask dirty_overlap = state.dirty_mask & pack_mask;
      rez.serialize(dirty_overlap);
      rez.serialize(num_valid_views);
      if (num_valid_views > 0)
      {
        for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
              it != state.valid_views.end(); it++)
        {
          FieldMask overlap = it->second & pack_mask;
          if (!overlap)
            continue;
          rez.serialize(it->first->get_key());
          rez.serialize(overlap);
        }
        if (invalidate_views)
        {
          invalidate_instance_views(state, pack_mask, false/*clean*/);
        }
      }
      // count the number of reduction views
      size_t num_valid_reductions = 0;
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
            it != state.reduction_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_valid_reductions++;
      }
      // pack the reduction views
      FieldMask reduction_overlap = state.reduction_mask & pack_mask;
      rez.serialize(reduction_overlap);
      rez.serialize(num_valid_reductions);
      if (num_valid_reductions > 0)
      {
        for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
              it != state.reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & pack_mask;
          if (!overlap)
            continue;
          rez.serialize(it->first->get_key());
          rez.serialize(overlap);
        }
        if (invalidate_views)
        {
          invalidate_reduction_views(state, pack_mask);
        }
      }
      // count the number of open partitions
      size_t num_open_parts = 0;
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_open_parts++;
      }
      rez.serialize(num_open_parts);
      if (num_open_parts > 0)
      {
        // Now go through the field states
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          it->pack_physical_state(pack_mask, rez);
          if (recurse)
          {
            for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                  pit != it->open_children.end(); pit++)
            {
              FieldMask overlap = pit->second & pack_mask;
              if (!overlap)
                continue;
              color_map[pit->first]->pack_physical_state(ctx, overlap, rez, invalidate_views, true/*recurse*/);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift /*= 0*/)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      // Unpack the dirty mask and the valid instances
      FieldMask update_dirty_mask;
      derez.deserialize(update_dirty_mask);
      state.dirty_mask |= update_dirty_mask;
      size_t num_valid_views;
      derez.deserialize(num_valid_views);
      for (unsigned idx = 0; idx < num_valid_views; idx++)
      {
        InstanceKey key;
        derez.deserialize(key);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        if (shift > 0)
          valid_mask.shift_left(shift);
        InstanceView *new_view = context->find_instance_view(key);
        if (state.valid_views.find(new_view) == state.valid_views.end())
        {
          state.valid_views[new_view] = valid_mask;
          new_view->add_valid_reference();
        }
        else
          state.valid_views[new_view] |= valid_mask;
      }
      // Unpack the reduction mask and the valid reduction instances
      FieldMask update_reduction_mask;
      derez.deserialize(update_reduction_mask);
      state.reduction_mask |= update_reduction_mask;
      size_t num_reduc_views;
      derez.deserialize(num_reduc_views);
      for (unsigned idx = 0; idx < num_reduc_views; idx++)
      {
        InstanceKey key;
        derez.deserialize(key);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        if (shift > 0)
          valid_mask.shift_left(shift);
        ReductionView *new_view = context->find_reduction_view(key);
        if (state.reduction_views.find(new_view) == state.reduction_views.end())
        {
          state.reduction_views[new_view] = valid_mask;
          new_view->add_valid_reference();
        }
        else
          state.reduction_views[new_view] |= valid_mask;
      }
      // Unpack the field states for child partitions
      size_t num_open_parts;
      derez.deserialize(num_open_parts);
      std::vector<FieldState> new_field_states(num_open_parts);
      for (unsigned idx = 0; idx < num_open_parts; idx++)
      {
        new_field_states[idx].unpack_physical_state(derez, shift);
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator it = new_field_states[idx].open_children.begin();
                it != new_field_states[idx].open_children.end(); it++)
          {
            color_map[it->first]->unpack_physical_state(ctx, derez, true/*recurse*/, shift);
          }
        }
      }
      // Now merge the field states into the existing state
      merge_new_field_states(state, new_field_states, false/*add states*/);
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::compute_diff_state_size(ContextID ctx, const FieldMask &pack_mask,
                                              std::set<InstanceManager*> &unique_managers,
                                              std::set<ReductionManager*> &unique_reductions,
                                              std::vector<RegionNode*> &diff_regions,
                                              std::vector<PartitionNode*> &diff_partitions,
                                              bool invalidate_views, bool recurse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      // Find the set of views that need to be sent back
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
            it != state.valid_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        // Get the unique manager for which we will need to send views
        unique_managers.insert(it->first->get_instance_manager());
      }
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
            it != state.reduction_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        unique_reductions.insert(it->first->get_reduction_manager());
      }
      if (!state.added_views.empty() || !state.added_reductions.empty() || !state.added_states.empty())
      {
        diff_regions.push_back(this);
        // Get the size of data that needs to be send back for the diff
        result += sizeof(size_t); // number of unique views to be sent back
        for (std::map<InstanceView*,FieldMask>::const_iterator it = state.added_views.begin();
              it != state.added_views.end(); it++)
        {
          if (it->second * pack_mask)
            continue;
          result += sizeof(InstanceKey);
          result += sizeof(it->second);
        }
        result += sizeof(size_t); // number of unique reduction views to be sent back
        for (std::map<ReductionView*,FieldMask>::const_iterator it = state.added_reductions.begin();
              it != state.added_reductions.end(); it++)
        {
          if (it->second * pack_mask)
            continue;
          result += sizeof(InstanceKey);
          result += sizeof(it->second);
        }
        result += sizeof(size_t); // number of new states
        for (std::list<FieldState>::const_iterator it = state.added_states.begin();
              it != state.added_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          result += it->compute_state_size(pack_mask);
        }
      }
      if (invalidate_views)
      {
        invalidate_instance_views(state, pack_mask, false/*clean*/);
        invalidate_reduction_views(state, pack_mask);
      }
      // Now do any open children
      if (recurse)
      {
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_diff_state_size(ctx, overlap, 
                          unique_managers, unique_reductions, diff_regions, diff_partitions, 
                          invalidate_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::pack_diff_state(ContextID ctx, const FieldMask &pack_mask,
                                     Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t num_added_views = 0;
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.added_views.begin();
            it != state.added_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_added_views++;
      }
      rez.serialize(num_added_views);
      for (std::map<InstanceView*,FieldMask>::const_iterator it = state.added_views.begin();
            it != state.added_views.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        rez.serialize(it->first->get_key());
        rez.serialize(it->second & pack_mask);
      }
      size_t num_added_reductions = 0;
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.added_reductions.begin();
            it != state.added_reductions.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        num_added_reductions++;
      }
      rez.serialize(num_added_reductions);
      for (std::map<ReductionView*,FieldMask>::const_iterator it = state.added_reductions.begin();
            it != state.added_reductions.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        rez.serialize(it->first->get_key());
        rez.serialize(it->second & pack_mask);
      }
      size_t num_added_states = 0;
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_added_states++;
      }
      rez.serialize(num_added_states);
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        it->pack_physical_state(pack_mask, rez);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::unpack_diff_state(ContextID ctx, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Check to see if the physical state exists
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t num_added_views;
      derez.deserialize(num_added_views);
      for (unsigned idx = 0; idx < num_added_views; idx++)
      {
        InstanceKey key;
        derez.deserialize(key);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        InstanceView *new_view = context->find_instance_view(key);
        if (state.valid_views.find(new_view) == state.valid_views.end())
        {
          state.valid_views[new_view] = valid_mask;
          new_view->add_valid_reference();
        }
        else
          state.valid_views[new_view] |= valid_mask;
        // Also put it on the added list 
        if (state.added_views.find(new_view) == state.added_views.end())
          state.added_views[new_view] = valid_mask;
        else
          state.added_views[new_view] |= valid_mask;
      }
      size_t num_added_reductions;
      derez.deserialize(num_added_reductions);
      for (unsigned idx = 0; idx < num_added_reductions; idx++)
      {
        InstanceKey key;
        derez.deserialize(key);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        ReductionView *new_view = context->find_reduction_view(key);
        if (state.reduction_views.find(new_view) == state.reduction_views.end())
        {
          state.reduction_views[new_view] = valid_mask;
          new_view->add_valid_reference();
        }
        else
          state.reduction_views[new_view] |= valid_mask;
        // Update the reduction mask
        state.reduction_mask |= valid_mask;
        // Also put it on the added list
        if (state.added_reductions.find(new_view) == state.added_reductions.end())
          state.added_reductions[new_view] = valid_mask;
        else
          state.added_reductions[new_view] |= valid_mask;
      }
      size_t num_added_states;
      derez.deserialize(num_added_states);
      std::vector<FieldState> new_states(num_added_states);
      for (unsigned idx = 0; idx < num_added_states; idx++)
      {
        new_states[idx].unpack_physical_state(derez);
      }
      merge_new_field_states(state, new_states, true/*add states*/);
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_physical_context(ContextID ctx, TreeStateLogger *logger, FieldMask capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Region Node (%x,%d,%d) Color %d at depth %d", 
          handle.index_space.id, handle.field_space.id,handle.tree_id,
          row_source->color, logger->get_depth());
      logger->down();
      std::map<Color,FieldMask> to_traverse;
      if (physical_states.find(ctx) != physical_states.end())
      {
        PhysicalState &state = physical_states[ctx];
        // Dirty Mask
        {
          FieldMask overlap = state.dirty_mask & capture_mask;
          char *dirty_buffer = overlap.to_string();
          logger->log("Dirty Mask: %s",dirty_buffer);
          free(dirty_buffer);
        }
        // Valid Views
        {
          unsigned num_valid = 0;
          for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin(); 
                it != state.valid_views.end(); it++)
          {
            if (it->second * capture_mask)
              continue;
            num_valid++;
          }
          logger->log("Valid Instances (%d)", num_valid);
          logger->down();
          for (std::map<InstanceView*,FieldMask>::const_iterator it = state.valid_views.begin();
                it != state.valid_views.end(); it++)
          {
            FieldMask overlap = it->second & capture_mask;
            if (!overlap)
              continue;
            char *valid_mask = overlap.to_string();
            logger->log("Instance %x   Memory %x   Mask %s",
                it->first->get_instance().id, it->first->get_location().id, valid_mask);
            free(valid_mask);
          }
          logger->up();
        }
        // Valid Reduction Views
        {
          unsigned num_valid = 0;
          for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
                it != state.reduction_views.end(); it++)
          {
            if (it->second * capture_mask)
              continue;
            num_valid++;
          }
          logger->log("Valid Reduction Instances (%d)", num_valid);
          logger->down();
          for (std::map<ReductionView*,FieldMask>::const_iterator it = state.reduction_views.begin();
                it != state.reduction_views.end(); it++)
          {
            FieldMask overlap = it->second & capture_mask;
            if (!overlap)
              continue;
            char *valid_mask = overlap.to_string();
            logger->log("Reduction Instance %x   Memory %x  Mask %s",
                it->first->get_instance().id, it->first->get_location().id, valid_mask);
            free(valid_mask);
          }
          logger->up();
        }
        // Open Field States 
        {
          logger->log("Open Field States (%ld)", state.field_states.size());
          logger->down();
          for (std::list<FieldState>::const_iterator it = state.field_states.begin();
                it != state.field_states.end(); it++)
          {
            it->print_state(logger, capture_mask);
            if (it->valid_fields * capture_mask)
              continue;
            for (std::map<Color,FieldMask>::const_iterator cit = it->open_children.begin();
                  cit != it->open_children.end(); cit++)
            {
              FieldMask overlap = cit->second & capture_mask;
              if (!overlap)
                continue;
              if (to_traverse.find(cit->first) == to_traverse.end())
                to_traverse[cit->first] = overlap;
              else
                to_traverse[cit->first] |= overlap;
            }
          }
          logger->up();
        }
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");

      for (std::map<Color,FieldMask>::const_iterator it = to_traverse.begin();
            it != to_traverse.end(); it++)
      {
        if (color_map.find(it->first) != color_map.end())
          color_map[it->first]->print_physical_context(ctx, logger, it->second);
      }

      logger->up();
    }

    /////////////////////////////////////////////////////////////
    // Partition Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(LogicalPartition p, RegionNode *par, 
                      IndexPartNode *row_src, bool add, RegionTreeForest *ctx)
      : RegionTreeNode(ctx), handle(p), parent(par), row_source(row_src), 
        disjoint(row_src->disjoint), added(add), marked(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PartitionNode::~PartitionNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PartitionNode::mark_destroyed(void)
    //--------------------------------------------------------------------------
    {
      added = false;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_child(LogicalRegion handle, RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(node->row_source->color) == color_map.end());
#endif
      color_map[node->row_source->color] = node;
      valid_map[node->row_source->color] = node;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::has_child(Color c) const
    //--------------------------------------------------------------------------
    {
      std::map<Color,RegionNode*>::const_iterator finder = color_map.find(c);
      return (finder != color_map.end());
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
      // If it doesn't exist then we have to make the node
      std::map<Color,RegionNode*>::const_iterator finder = color_map.find(c);
      if (finder == color_map.end())
      {
        IndexSpaceNode *index_child = row_source->get_child(c);
        LogicalRegion child_handle(handle.tree_id, index_child->handle, handle.field_space);
        return context->create_node(child_handle, this, true/*add*/);
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      // only ever remove things from the valid map
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_map.find(c) != valid_map.end());
#endif
      valid_map.erase(c);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::initialize_logical_context(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,LogicalState>::iterator finder = logical_states.find(ctx);
      if (finder != logical_states.end())
      {
        LogicalState &state = finder->second;
        state.field_states.clear();
        state.added_states.clear();
        state.curr_epoch_users.clear();
        state.prev_epoch_users.clear();
        // Now do any children
        for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->initialize_logical_context(ctx);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::register_deletion_operation(ContextID ctx, DeletionOperation *op,
                                                    const FieldMask &deletion_mask)
    //--------------------------------------------------------------------------
    {
      // If we don't even have a logical state then neither 
      // do any of our children so we're done
      if (logical_states.find(ctx) != logical_states.end())
      {
        const LogicalState &state = logical_states[ctx];
        for (std::list<LogicalUser>::const_iterator it = state.curr_epoch_users.begin();
              it != state.curr_epoch_users.end(); it++)
        {
          // Check for field disjointness
          if (it->field_mask * deletion_mask)
            continue;
          op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
        }
        for (std::list<LogicalUser>::const_iterator it = state.prev_epoch_users.begin();
              it != state.prev_epoch_users.end(); it++)
        {
          // Check for field disjointness
          if (it->field_mask * deletion_mask)
            continue;
          op->add_mapping_dependence(0/*idx*/, *it, TRUE_DEPENDENCE);
        }
      }
      // Do any children
      for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->register_deletion_operation(ctx, op, deletion_mask);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::initialize_physical_context(ContextID ctx, bool clear, const FieldMask &init_mask)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(ctx);
      if (finder != physical_states.end())
      {
        PhysicalState &state = finder->second;
#ifdef DEBUG_HIGH_LEVEL
        assert(state.valid_views.empty());
        assert(state.added_views.empty());
#endif
        if (clear)
          state.clear_state(init_mask);
        // Only handle children if we ourselves already had a state
        for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->initialize_physical_context(ctx, clear, init_mask, false/*top*/); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::fill_exclusive_context(ContextID ctx, const FieldMask &fill_mask, Color open_child)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      FieldState to_add;
      to_add.valid_fields = fill_mask;
      to_add.open_state = RegionTreeNode::OPEN_READ_WRITE;
      to_add.redop = 0;
      to_add.open_children[open_child] = fill_mask;
      merge_new_field_state(state, to_add, true/*add state*/);
      // We better have a parent since we haven't found the top of the context yet
      // We also better not be the top of the context
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent->fill_exclusive_context(ctx, fill_mask, row_source->color);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::merge_physical_context(ContextID outer_ctx, ContextID inner_ctx, const FieldMask &merge_mask)
    //--------------------------------------------------------------------------
    {
      // If there is no inner state then there's nothing to do in this tree
      if (physical_states.find(inner_ctx) == physical_states.find(inner_ctx))
        return;
      PhysicalState &inner_state = physical_states[inner_ctx];
      if (physical_states.find(outer_ctx) == physical_states.find(outer_ctx))
        physical_states[outer_ctx] = PhysicalState();
      PhysicalState &outer_state = physical_states[outer_ctx];
      // Do the open children
      for (std::list<FieldState>::const_iterator it = inner_state.field_states.begin();
            it != inner_state.field_states.end(); it++)
      {
        FieldState copy = *it;
        copy.valid_fields &= merge_mask;
        if (!copy.valid_fields)
          continue;
        std::vector<Color> to_delete;
        for (std::map<Color,FieldMask>::iterator cit = copy.open_children.begin();
              cit != copy.open_children.end(); cit++)
        {
          cit->second &= merge_mask;
          if (!cit->second)
            to_delete.push_back(cit->first);
        }
        // Delete any empty children
        for (std::vector<Color>::const_iterator cit = to_delete.begin();
              cit != to_delete.end(); cit++)
        {
          copy.open_children.erase(*cit);
        }
        if (copy.open_children.empty())
          continue;
        // Otherwise add the new state
        merge_new_field_state(outer_state, copy, true/*add states*/);
      }
      // finally continue the traversal
      for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->merge_physical_context(outer_ctx, inner_ctx, merge_mask);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::register_physical_region(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
#endif
      if (physical_states.find(rm.ctx) == physical_states.end())
        physical_states[rm.ctx] = PhysicalState();

      PhysicalState &state = physical_states[rm.ctx];
      if (rm.path.size() == 1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(rm.sanitizing); // This should only be the end if we're sanitizing
#endif
        rm.path.pop_back();
        // If we're doing a write where each sub-task is going to get an
        // independent region in the partition, then we're done.  Otherwise
        // for read-only and reduce, we need to siphon all the open children.
        if (IS_READ_ONLY(user.usage) || IS_REDUCE(user.usage))
        {
          // If the partition is disjoint sanitize each of the children seperately
          // otherwise, we only need to do this one time
          if (disjoint)
          {
            for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
                  it != valid_map.end(); it++)
            {
              PhysicalCloser closer(user, rm, parent, IS_READ_ONLY(user.usage));
              siphon_open_children(closer, state, user, user.field_mask, it->first);
              if (!closer.success)
              {
                rm.success = false;
                return;
              }
            }
          }
          else
          {
            PhysicalCloser closer(user, rm, parent, IS_READ_ONLY(user.usage));
            siphon_open_children(closer, state, user, user.field_mask);
            if (!closer.success)
            {
              rm.success = false;
              return;
            }
          }
        }

        if (!IS_REDUCE(user.usage))
          parent->pull_valid_views(rm.ctx, user.field_mask);
        // No need to close anything up here since we were just sanitizing
        rm.success = true;
      }
      else
      {
        rm.path.pop_back();
        Color next_reg = rm.path.back();
        // Close up any regions which might contain data we need
        // and then continue the traversal
        // Use the parent node as the target of any close operations
        PhysicalCloser closer(user, rm, parent, IS_READ_ONLY(user.usage));
        // Since we're already down the partition, mark it as such before traversing
        closer.partition_color = row_source->color;
        closer.partition_valid = true;
        bool open_only = siphon_open_children(closer, state, user, user.field_mask, next_reg);
        // Check to see if the close was successful  
        if (!closer.success)
        {
          rm.success = false;
          return;
        }
        RegionNode *child = get_child(next_reg);
        if (open_only)
          child->open_physical_tree(user, rm);
        else
          child->register_physical_region(user, rm);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::open_physical_tree(const PhysicalUser &user, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!rm.path.empty());
      assert(rm.path.back() == row_source->color);
#endif
      if (physical_states.find(rm.ctx) == physical_states.end())
        physical_states[rm.ctx] = PhysicalState();
      PhysicalState &state = physical_states[rm.ctx];
      if (rm.path.size() == 1)
      {
        rm.path.pop_back();
#ifdef DEBUG_HIGH_LEVEL
        assert(rm.sanitizing); // should only end on a partition if sanitizing
#endif
        if (!IS_REDUCE(user.usage))
          parent->pull_valid_views(rm.ctx, user.field_mask);
        rm.success = true;
      }
      else
      {
        rm.path.pop_back();
        Color next_region = rm.path.back();
        // Update the field states
        std::vector<FieldState> new_states;
        new_states.push_back(FieldState(user, user.field_mask, next_region));
        merge_new_field_states(state, new_states, true/*add states*/);
        // Continue the traversal
        RegionNode *child_node = get_child(next_region);
        child_node->open_physical_tree(user, rm); 
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::close_physical_tree(PhysicalCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!closer.partition_valid);
#endif
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(closer.rm.ctx);
      if (finder == physical_states.end())
        return;
      PhysicalState &state = finder->second; 
      // Mark the closer with the color of the partition that we're closing
      // so we know how to convert InstanceViews later.  Then
      // figure out which of our open children we need to close.  If we do
      // need to issue a close to any of them, update the target_views with
      // new views corresponding to the logical region we're going to be closing.
      closer.pre_partition(row_source->color);
      siphon_open_children(closer, state, closer.user, closing_mask);
      closer.post_partition();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::close_reduction_tree(ReductionCloser &closer, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(closer.rm.ctx);
      if (finder == physical_states.end())
        return;
      PhysicalState &state = finder->second;
      siphon_open_children(closer, state, closer.user, closing_mask);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_children_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return false;
      return (disjoint || row_source->are_disjoint(c1, c2));
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_closing_partition(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_tree_child(Color c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    Color PartitionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    bool PartitionNode::color_match(Color c)
    //--------------------------------------------------------------------------
    {
      return (c == row_source->color);
    }
#endif

    //--------------------------------------------------------------------------
    size_t PartitionNode::compute_tree_size(bool returning) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(bool);
      if (returning || marked)
      {
        result += sizeof(handle);
        result += sizeof(size_t); // number of children
        for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          result += it->second->compute_tree_size(returning);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::serialize_tree(Serializer &rez, bool returning)
    //--------------------------------------------------------------------------
    {
      if (returning || marked)
      {
        rez.serialize(true);
        rez.serialize(handle);
        rez.serialize(valid_map.size());
        for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->serialize_tree(rez, returning);
        }
        marked = false;
      }
      else
      {
        rez.serialize(false);
      }
      if (returning)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added);
#endif
        added = false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PartitionNode::deserialize_tree(Deserializer &derez,
                      RegionNode *parent, RegionTreeForest *context, bool returning)
    //--------------------------------------------------------------------------
    {
      bool needs_unpack;
      derez.deserialize(needs_unpack);
      if (needs_unpack)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        PartitionNode *result = context->create_node(handle, parent, returning);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          RegionNode::deserialize_tree(derez, result, context, returning);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::mark_node(bool recurse)
    //--------------------------------------------------------------------------
    {
      marked = true;
      if (recurse)
      {
        for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->mark_node(true/*recurse*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::find_top_marked(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(marked);
      assert(parent != NULL);
#endif
      return parent->find_top_marked();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::find_new_partitions(std::vector<PartitionNode*> &new_parts) const
    //--------------------------------------------------------------------------
    {
      // If we're the top of a new partition tree, put ourselves on the list and return
      if (added)
      {
        PartitionNode *copy = const_cast<PartitionNode*>(this);
        new_parts.push_back(copy);
        return;
      }
      for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
            it != valid_map.end(); it++)
      {
        it->second->find_new_partitions(new_parts);
      }
    }

    //--------------------------------------------------------------------------
    size_t PartitionNode::compute_state_size(ContextID ctx, const FieldMask &pack_mask,
                                              std::set<InstanceManager*> &unique_managers, 
                                              std::set<ReductionManager*> &unique_reductions, 
                                              bool mark_invalid_views, bool recurse)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      result += sizeof(size_t); // number of field states
      // Can ignore the dirty and mask and valid instances here since they don't mean anything
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        result += it->compute_state_size(pack_mask);
        // Traverse any open partitions below
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_state_size(ctx, overlap, 
                        unique_managers, unique_reductions, mark_invalid_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::pack_physical_state(ContextID ctx, const FieldMask &pack_mask,
                                            Serializer &rez, bool invalidate_views, bool recurse)
    //--------------------------------------------------------------------------
    {
      PhysicalState &state = physical_states[ctx];
      size_t num_field_states = 0;
      for (std::list<FieldState>::const_iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_field_states++;
      }
      rez.serialize(num_field_states);
      if (num_field_states > 0)
      {
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          it->pack_physical_state(pack_mask, rez);
          if (recurse)
          {
            for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                  pit != it->open_children.end(); pit++)
            {
              FieldMask overlap = pit->second & pack_mask;
              if (!overlap)
                continue;
              color_map[pit->first]->pack_physical_state(ctx, overlap, rez, invalidate_views, true/*recurse*/);
            } 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::unpack_physical_state(ContextID ctx, Deserializer &derez, bool recurse, unsigned shift)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t num_field_states;
      derez.deserialize(num_field_states);
      std::vector<FieldState> new_field_states(num_field_states);
      for (unsigned idx = 0; idx < num_field_states; idx++)
      {
        new_field_states[idx].unpack_physical_state(derez, shift);
        if (recurse)
        {
          for (std::map<Color,FieldMask>::const_iterator it = new_field_states[idx].open_children.begin();
                it != new_field_states[idx].open_children.end(); it++)
          {
            color_map[it->first]->unpack_physical_state(ctx, derez, true/*recurse*/, shift);
          }
        }
      }
      merge_new_field_states(state, new_field_states, true/*add states*/);
    }

    //--------------------------------------------------------------------------
    size_t PartitionNode::compute_diff_state_size(ContextID ctx, const FieldMask &pack_mask,
                                          std::set<InstanceManager*> &unique_managers,
                                          std::set<ReductionManager*> &unique_reductions,
                                          std::vector<RegionNode*> &diff_regions,
                                          std::vector<PartitionNode*> &diff_partitions,
                                          bool invalidate_views, bool recurse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t result = 0;
      if (!state.added_states.empty())
      {
        diff_partitions.push_back(this);
        // Get the size of data that needs to be send back for the diff
        result += sizeof(size_t); // number of new states
        for (std::list<FieldState>::const_iterator it = state.added_states.begin();
              it != state.added_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          result += it->compute_state_size(pack_mask);
        }
      }
      // Now do any open children
      if (recurse)
      {
        for (std::list<FieldState>::const_iterator it = state.field_states.begin();
              it != state.field_states.end(); it++)
        {
          if (it->valid_fields * pack_mask)
            continue;
          for (std::map<Color,FieldMask>::const_iterator pit = it->open_children.begin();
                pit != it->open_children.end(); pit++)
          {
            FieldMask overlap = pit->second & pack_mask;
            if (!overlap)
              continue;
            result += color_map[pit->first]->compute_diff_state_size(ctx, overlap, 
                          unique_managers, unique_reductions, diff_regions, diff_partitions, 
                          invalidate_views, true/*recurse*/);
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::pack_diff_state(ContextID ctx, const FieldMask &pack_mask, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_states.find(ctx) != physical_states.end());
#endif
      PhysicalState &state = physical_states[ctx];
      size_t num_added_states = 0;
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        num_added_states++;
      }
      rez.serialize(num_added_states);
      for (std::list<FieldState>::const_iterator it = state.added_states.begin();
            it != state.added_states.end(); it++)
      {
        if (it->valid_fields * pack_mask)
          continue;
        it->pack_physical_state(pack_mask, rez);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::unpack_diff_state(ContextID ctx, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if (physical_states.find(ctx) == physical_states.end())
        physical_states[ctx] = PhysicalState();
      PhysicalState &state = physical_states[ctx];
      size_t num_added_states;
      derez.deserialize(num_added_states);
      std::vector<FieldState> new_states(num_added_states);
      for (unsigned idx = 0; idx < num_added_states; idx++)
      {
        new_states[idx].unpack_physical_state(derez);
      }
      merge_new_field_states(state, new_states, true/*add states*/);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::mark_invalid_instance_views(ContextID ctx, const FieldMask &mask, bool recurse)
    //--------------------------------------------------------------------------
    {
      if (recurse && (physical_states.find(ctx) != physical_states.end()))
      {
        for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->mark_invalid_instance_views(ctx, mask, recurse);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::recursive_invalidate_views(ContextID ctx, const FieldMask &mask, bool last_use)
    //--------------------------------------------------------------------------
    {
      std::map<ContextID,PhysicalState>::iterator finder = physical_states.find(ctx);
      if (finder != physical_states.end())
      {
        if (last_use)
          physical_states.erase(finder);
        for (std::map<Color,RegionNode*>::const_iterator it = valid_map.begin();
              it != valid_map.end(); it++)
        {
          it->second->recursive_invalidate_views(ctx, mask, last_use);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_physical_context(ContextID ctx, TreeStateLogger *logger, FieldMask capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (%d,%d,%d) Color %d disjoint %d at depth %d",
          handle.index_partition, handle.field_space.id, handle.tree_id, 
          row_source->color, disjoint, logger->get_depth());
      logger->down();
      std::map<Color,FieldMask> to_traverse;
      if (physical_states.find(ctx) != physical_states.end())
      {
        PhysicalState &state = physical_states[ctx];
        // Open Field States
        {
          logger->log("Open Field States (%ld)", state.field_states.size()); 
          logger->down();
          for (std::list<FieldState>::const_iterator it = state.field_states.begin();
                it != state.field_states.end(); it++)
          {
            it->print_state(logger, capture_mask);
            if (it->valid_fields * capture_mask)
              continue;
            for (std::map<Color,FieldMask>::const_iterator cit = it->open_children.begin();
                  cit != it->open_children.end(); cit++)
            {
              FieldMask overlap = cit->second & capture_mask;
              if (!overlap)
                continue;
              if (to_traverse.find(cit->first) == to_traverse.end())
                to_traverse[cit->first] = overlap;
              else
                to_traverse[cit->first] |= overlap;
            }
          }
          logger->up();
        }
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");

      // Now do all the children
      for (std::map<Color,FieldMask>::const_iterator it = to_traverse.begin();
            it != to_traverse.end(); it++)
      {
        if (color_map.find(it->first) != color_map.end())
          color_map[it->first]->print_physical_context(ctx, logger, it->second);
      }

      logger->up();
    }

    /////////////////////////////////////////////////////////////
    // Physical Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalManager::PhysicalManager(RegionTreeForest *ctx, UniqueManagerID mid,
                      bool rem, bool cl, Memory loc, PhysicalInstance inst)
      : context(ctx), references(0), unique_id(mid), remote(rem), clone(cl),
        remote_frac(Fraction<unsigned long>(0,1)), local_frac(Fraction<unsigned long>(1,1)), 
        location(loc), instance(inst)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalManager::~PhysicalManager(void)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Instance Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(Memory m, PhysicalInstance inst, 
            const std::map<FieldID,Domain::CopySrcDstField> &infos, FieldSpace fsp,
            const FieldMask &mask, RegionTreeForest *ctx, UniqueManagerID mid, bool rem, bool cl)
      : PhysicalManager(ctx, mid, rem, cl, m, inst),
        fspace(fsp), allocated_fields(mask), field_infos(infos)
    //--------------------------------------------------------------------------
    {
      // If we're not remote, make the lock
      if (!remote)
        lock = Lock::create_lock();
      else
        lock = Lock::NO_LOCK;
    }

    //--------------------------------------------------------------------------
    InstanceManager::~InstanceManager(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (!remote && !clone && instance.exists())
      {
        log_leak(LEVEL_WARNING,"Leaking physical instance %x in memory %x",
                    instance.id, location.id);
      }
      if (remote && !remote_frac.is_empty())
      {
        log_leak(LEVEL_WARNING,"Leaking remote fraction (%ld/%ld) of instance %x "
                    "in memory %x (runtime bug)", remote_frac.get_num(),
                    remote_frac.get_denom(), instance.id, location.id);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void InstanceManager::add_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      references++;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::remove_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references > 0);
#endif
      references--;
      if (references == 0)
        garbage_collect();
    }

    //--------------------------------------------------------------------------
    void InstanceManager::add_view(InstanceView *view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view->manager == this);
      assert(view != NULL);
#endif
      all_views.push_back(view);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(const FieldMask &field_mask,
                           std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
      int found = 0;
#endif
      FieldSpaceNode *field_space = context->get_node(fspace);
      for (std::map<FieldID,Domain::CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        if (field_space->is_set(it->first, field_mask))
        {
          fields.push_back(it->second);
#ifdef DEBUG_HIGH_LEVEL
          found++;
#endif
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(found == FieldMask::pop_count(field_mask));
#endif
    }

    //--------------------------------------------------------------------------
    void InstanceManager::find_info(FieldID fid, std::vector<Domain::CopySrcDstField> &sources)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(field_infos.find(fid) != field_infos.end());
      assert(instance.exists());
#endif
      sources.push_back(field_infos[fid]);
    }

    //--------------------------------------------------------------------------
    InstanceManager* InstanceManager::clone_manager(const FieldMask &mask, FieldSpaceNode *field_space) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,Domain::CopySrcDstField> new_infos;
      for (std::map<FieldID,Domain::CopySrcDstField>::const_iterator it =
            field_infos.begin(); it != field_infos.end(); it++)
      {
        if (field_space->is_set(it->first, mask))
        {
          new_infos.insert(*it);
        }
      }
      InstanceManager *clone = context->create_instance_manager(location, instance, new_infos,
                                                    fspace, mask, false/*remote*/, true/*clone*/);
#ifdef LEGION_SPY
      LegionSpy::log_instance_manager(instance.id, clone->get_unique_id());
#endif
      return clone;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::find_views_from(LogicalRegion handle, std::map<InstanceView*,FieldMask> &unique_views,
                  std::vector<InstanceView*> &ordered_views, const FieldMask &packing_mask, int filter /*= -1*/)
    //--------------------------------------------------------------------------
    {
      // Go through our views and see if we can find the one we want to pack from
      InstanceView *target = NULL;
      for (std::vector<InstanceView*>::const_iterator it = all_views.begin();
            it != all_views.end(); it++)
      {
        if ((*it)->get_handle() == handle)
        {
          target = *it;
          break;
        }
      }
      // Check to see if we found it
      if (target != NULL)
      {
        // It does exist, so get the required views from this perspective
        target->find_required_views(unique_views, ordered_views, packing_mask, filter);
      }
      else
      {
        // Otherwise, this is easy, all the views have to be sent, so add them
        // Note: they are already in a good order for serialization
        for (std::vector<InstanceView*>::const_iterator it = all_views.begin();
              it != all_views.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert((*it) != NULL);
#endif
          std::map<InstanceView*,FieldMask>::iterator finder = unique_views.find(*it);
          if (finder != unique_views.end())
          {
            // Update the mask
            finder->second |= packing_mask;
          }
          else
          {
            ordered_views.push_back(*it);
            unique_views[*it] = packing_mask;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t InstanceManager::compute_send_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0; 
      result += sizeof(unique_id);
      result += sizeof(local_frac);
      result += sizeof(location);
      result += sizeof(instance);
      result += sizeof(lock);
      result += sizeof(FieldSpace);
      result += sizeof(allocated_fields);
      result += sizeof(size_t);
      result += (field_infos.size() * (sizeof(FieldID) + sizeof(Domain::CopySrcDstField)));
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::pack_manager_send(Serializer &rez, unsigned long num_ways)
    //--------------------------------------------------------------------------
    {
      rez.serialize(unique_id);
      InstFrac to_take = local_frac.get_part(num_ways);
#ifdef DEBUG_HIGH_LEVEL
      assert(!to_take.is_empty());
#endif
      local_frac.subtract(to_take);
#ifdef DEBUG_HIGH_LEVEL
      assert(!local_frac.is_empty()); // this is really bad if it happens
#endif
      rez.serialize(to_take);
      rez.serialize(location);
      rez.serialize(instance);
      rez.serialize(lock);
      rez.serialize(fspace);
      rez.serialize(allocated_fields);
      rez.serialize(field_infos.size());
      for (std::map<FieldID,Domain::CopySrcDstField>::const_iterator it =
            field_infos.begin(); it != field_infos.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/void InstanceManager::unpack_manager_send(RegionTreeForest *context,
                      Deserializer &derez, unsigned long split_factor)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      InstFrac remote_frac;
      derez.deserialize(remote_frac);
      // Scale the remote fraction by the split factor
      remote_frac.divide(split_factor);
      Memory location;
      derez.deserialize(location);
      PhysicalInstance inst;
      derez.deserialize(inst);
      Lock lock;
      derez.deserialize(lock);
      FieldSpace fsp;
      derez.deserialize(fsp);
      FieldMask alloc_fields;
      derez.deserialize(alloc_fields);
      std::map<FieldID,Domain::CopySrcDstField> field_infos;
      size_t num_infos;
      derez.deserialize(num_infos);
      for (unsigned idx = 0; idx < num_infos; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        Domain::CopySrcDstField info;
        derez.deserialize(info);
        field_infos[fid] = info;
      }
      InstanceManager *result = context->create_instance_manager(location, inst, 
                  field_infos, fsp, alloc_fields, true/*remote*/, false/*clone*/, mid);
      // Set the remote fraction and scale it by the split factor
      result->remote_frac = remote_frac;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::find_user_returns(std::vector<InstanceView*> &returning_views) const
    //--------------------------------------------------------------------------
    {
      // Find all our views that either haven't been returned or need to be
      // returned because they have added users
      for (std::vector<InstanceView*>::const_iterator it = all_views.begin();
            it != all_views.end(); it++)
      {
        if ((*it)->local_view || (*it)->has_added_users()) 
        {
          returning_views.push_back(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t InstanceManager::compute_return_size(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Should only happen to non-remote, non-clone managers
      assert(!remote && !clone);
      assert(local_frac.is_whole());
      assert(instance.exists());
#endif
      size_t result = 0;
      result += sizeof(unique_id);
      result += sizeof(location);
      result += sizeof(instance);
      result += sizeof(lock);
      result += sizeof(fspace);
      result += sizeof(size_t); // number of allocated fields
      result += (field_infos.size() * (sizeof(FieldID) + sizeof(Domain::CopySrcDstField)));
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::pack_manager_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(unique_id);
      rez.serialize(location);
      rez.serialize(instance);
      rez.serialize(lock);
      rez.serialize(fspace);
      rez.serialize(field_infos.size());
      for (std::map<FieldID,Domain::CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      // Finally mark this manager as remote since it has now been sent
      // back and should no longer be allowed to be deleted from this point
      remote = true;
      // Mark also that we still hold half the remote part, the other
      // half part will be sent back to enclosing context.  Note this is
      // done implicitly (see below in unpack_manager_return).
      remote_frac = InstFrac(1,2);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceManager::unpack_manager_return(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      Memory location;
      derez.deserialize(location);
      PhysicalInstance inst;
      derez.deserialize(inst);
      Lock lock;
      derez.deserialize(lock);
      FieldSpace fsp;
      derez.deserialize(fsp);
      size_t num_infos;
      derez.deserialize(num_infos);
      std::map<FieldID,Domain::CopySrcDstField> field_infos;
      std::vector<FieldID> fields(num_infos);
      for (unsigned idx = 0; idx < num_infos; idx++)
      {
        derez.deserialize(fields[idx]);
        Domain::CopySrcDstField info;
        derez.deserialize(info);
#ifdef DEBUG_HIGH_LEVEL
        assert(field_infos.find(fields[idx]) == field_infos.end());
#endif
        field_infos[fields[idx]] = info;
      }
      FieldSpaceNode *field_node = context->get_node(fsp);
      FieldMask allocated_fields = field_node->get_field_mask(fields);
      // Now make the instance manager
      InstanceManager *result = context->create_instance_manager(location, inst, field_infos, 
                                  fsp, allocated_fields, false/*remote*/, false/*clone*/, mid);
      // Mark that we only have half the local frac since the other half is still
      // on the original node.  It's also possible that the other half will be unpacked
      // later in this process and we'll be whole again.
      result->local_frac = InstFrac(1,2);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::pack_remote_fraction(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remote);
      assert(is_valid_free());
      assert(!remote_frac.is_empty());
#endif
      InstFrac return_frac = remote_frac;
      rez.serialize(return_frac);
      remote_frac.subtract(return_frac);
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_frac.is_empty());
#endif
    }

    //--------------------------------------------------------------------------
    void InstanceManager::unpack_remote_fraction(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!local_frac.is_whole());
#endif
      InstFrac return_frac;
      derez.deserialize(return_frac);
      local_frac.add(return_frac);
      if (local_frac.is_whole())
        garbage_collect();
    }

    //--------------------------------------------------------------------------
    void InstanceManager::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
#ifndef DISABLE_GC
      if (!remote && !clone && (references == 0) && local_frac.is_whole())
      {
        log_garbage(LEVEL_DEBUG,"Garbage collecting physical instance %x in memory %x",instance.id, location.id);
        instance.destroy();
        lock.destroy_lock();
        instance = PhysicalInstance::NO_INST;
        lock = Lock::NO_LOCK;
#ifdef LEGION_PROF
        LegionProf::register_instance_deletion(unique_id);
#endif
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::is_valid_free(void) const
    //--------------------------------------------------------------------------
    {
      bool result = true;
      for (std::vector<InstanceView*>::const_iterator it = all_views.begin();
            it != all_views.end(); it++)
      {
        if ((*it)->is_valid_view())
        {
          result = false;
          break;
        }
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Reduction Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionManager::ReductionManager(RegionTreeForest *ctx, UniqueManagerID mid, 
                                        bool rem, bool cl, Memory loc, 
                                        PhysicalInstance inst, ReductionOpID r, const ReductionOp *o)
      : PhysicalManager(ctx, mid, rem, cl, loc, inst), 
        redop(r), op(o), view(NULL)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(redop > 0);
#endif
    }

    //--------------------------------------------------------------------------
    ReductionManager::~ReductionManager(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (!remote && !clone && instance.exists())
      {
        log_leak(LEVEL_WARNING,"Leaking reduction instance %x in memory %x",
                    instance.id, location.id);
      }
      if (remote && !remote_frac.is_empty())
      {
        log_leak(LEVEL_WARNING,"Leaking remote fraction (%ld/%ld) of reduction instance %x "
                    "in memory %x (runtime bug)", remote_frac.get_num(),
                    remote_frac.get_denom(), instance.id, location.id);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void ReductionManager::set_view(ReductionView *v)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view == NULL);
#endif
      view = v;
    }

    //--------------------------------------------------------------------------
    void ReductionManager::add_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      references++;
    }

    //--------------------------------------------------------------------------
    void ReductionManager::remove_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references > 0);
#endif
      references--;
      if (references == 0)
        garbage_collect();
    }

    //--------------------------------------------------------------------------
    size_t ReductionManager::compute_send_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(unique_id);
      result += sizeof(local_frac);
      result += sizeof(location);
      result += sizeof(instance);
      result += sizeof(redop);
      result += sizeof(Domain);
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
#endif
      result += view->compute_send_size();
      return result;
    }

    //--------------------------------------------------------------------------
    void ReductionManager::pack_manager_send(Serializer &rez, unsigned long num_ways)
    //--------------------------------------------------------------------------
    {
      rez.serialize(unique_id);
      InstFrac to_take = local_frac.get_part(num_ways);
#ifdef DEBUG_HIGH_LEVEL
      assert(!to_take.is_empty());
#endif
      local_frac.subtract(to_take);
      rez.serialize(to_take);
      rez.serialize(location);
      rez.serialize(instance);
      rez.serialize(redop);
      rez.serialize(get_pointer_space());
      view->pack_view_send(rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionManager::unpack_manager_send(RegionTreeForest *context,
                        Deserializer &derez, unsigned long split_factor)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      InstFrac remote_frac;
      derez.deserialize(remote_frac);
      // Scale the remote fraction by the split factor
      remote_frac.divide(split_factor);
      Memory location;
      derez.deserialize(location);
      PhysicalInstance inst;
      derez.deserialize(inst);
      ReductionOpID redop;
      derez.deserialize(redop);
      const ReductionOp *op = HighLevelRuntime::get_reduction_op(redop);
      Domain pointer_space;
      derez.deserialize(pointer_space);
      ReductionManager *result = context->create_reduction_manager(location, inst, redop, op, 
                                              true/*remote*/, false/*clone*/, pointer_space, mid);
      result->remote_frac = remote_frac;
      // Now unpack the View
      ReductionView::unpack_view_send(context, derez);
    }

    //--------------------------------------------------------------------------
    size_t ReductionManager::compute_return_size(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!remote && !clone);
      assert(local_frac.is_whole());
      assert(instance.exists());
#endif
      size_t result = 0;
      result += sizeof(unique_id);
      result += sizeof(location);
      result += sizeof(instance);
      result += sizeof(redop);
      result += sizeof(Domain);
      // no need to pack the view, it will get sent back seperately
      return result;
    }

    //--------------------------------------------------------------------------
    void ReductionManager::pack_manager_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(unique_id);
      rez.serialize(location);
      rez.serialize(instance);
      rez.serialize(redop);
      rez.serialize(get_pointer_space());
      // Now mark this manager as remote, and reduce its remote fraction to 1/2
      remote = true;
      remote_frac = InstFrac(1,2);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionManager::unpack_manager_return(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      Memory location;
      derez.deserialize(location);
      PhysicalInstance inst;
      derez.deserialize(inst);
      ReductionOpID redop;
      derez.deserialize(redop);
      const ReductionOp *op = HighLevelRuntime::get_reduction_op(redop);
      Domain pointer_space;
      derez.deserialize(pointer_space);
      ReductionManager *result = context->create_reduction_manager(location, inst, redop, op,
                                      false/*remote*/, false/*clone*/, pointer_space, mid);
      // Set the local fraction to 1/2 since the other half is still remote
      result->local_frac = InstFrac(1,2);
    }

    //--------------------------------------------------------------------------
    void ReductionManager::pack_remote_fraction(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remote);
      assert(is_valid_free());
      assert(!remote_frac.is_empty());
#endif
      InstFrac return_frac = remote_frac;
      rez.serialize(return_frac);
      remote_frac.subtract(return_frac);
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_frac.is_empty());
#endif
    }

    //--------------------------------------------------------------------------
    void ReductionManager::unpack_remote_fraction(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!local_frac.is_whole());
#endif
      InstFrac return_frac;
      derez.deserialize(return_frac);
      local_frac.add(return_frac);
      if (local_frac.is_whole())
        garbage_collect();
    }

    //--------------------------------------------------------------------------
    void ReductionManager::find_user_returns(std::vector<ReductionView*> &returning_views) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
#endif
      if (view->local_view || view->has_added_users())
        returning_views.push_back(view);
    }

    //--------------------------------------------------------------------------
    bool ReductionManager::is_valid_free(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
#endif
      return !view->is_valid_view();
    }

    //--------------------------------------------------------------------------
    void ReductionManager::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      if (!remote && !clone && (references == 0) && local_frac.is_whole())
      {
        log_garbage(LEVEL_DEBUG,"Garbage collecting reduction instance %x in memory %x",instance.id, location.id);
        instance.destroy();
        instance = PhysicalInstance::NO_INST;
#ifdef LEGION_PROF
        LegionProf::register_instance_deletion(unique_id);
#endif
      }
    }

    /////////////////////////////////////////////////////////////
    // List Reduction Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ListReductionManager::ListReductionManager(RegionTreeForest *ctx, UniqueManagerID mid, 
                                                bool remote, bool clone, Memory loc, 
                                                PhysicalInstance inst, ReductionOpID redop, 
                                                const ReductionOp *op, Domain space)
      : ReductionManager(ctx, mid, remote, clone, loc, inst, redop, op), ptr_space(space)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ListReductionManager::~ListReductionManager(void)
    //--------------------------------------------------------------------------
    {
      // If this is the last version of the manager, destroy the pointer index space
      if (!remote && !clone)
        ptr_space.get_index_space().destroy();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> ListReductionManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      // TODO: Implement this
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> ListReductionManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    ReductionManager* ListReductionManager::clone_manager(void) const
    //--------------------------------------------------------------------------
    {
      ReductionManager *result = context->create_reduction_manager(location, instance, 
                                            redop, op, remote, true/*clone*/, ptr_space);
#ifdef LEGION_SPY
      LegionSpy::log_reduction_manager(instance.id, result->get_unique_id());
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void ListReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      // Assume that it's all the fields for right now
      // but offset by the pointer size
      fields.push_back(Domain::CopySrcDstField(instance, sizeof(ptr_t), op->sizeof_rhs));
    }

    //--------------------------------------------------------------------------
    Event ListReductionManager::issue_reduction(const std::vector<Domain::CopySrcDstField> &src_fields,
                                                const std::vector<Domain::CopySrcDstField> &dst_fields,
                                                Domain space, Event precondition, bool reduction_fold)
    //--------------------------------------------------------------------------
    {
      Domain::CopySrcDstField idx_field(instance, 0/*offset*/, sizeof(ptr_t));
      return space.copy_indirect(idx_field, src_fields, dst_fields, precondition, redop, reduction_fold);
    }

    /////////////////////////////////////////////////////////////
    // Fold Reduction Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FoldReductionManager::FoldReductionManager(RegionTreeForest *ctx, UniqueManagerID mid, 
                                                bool remote, bool clone, Memory loc, 
                                                PhysicalInstance inst, ReductionOpID redop, 
                                                const ReductionOp *op)
      : ReductionManager(ctx, mid, remote, clone, loc, inst, redop, op)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FoldReductionManager::~FoldReductionManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> FoldReductionManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> FoldReductionManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // Should never be called 
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    ReductionManager* FoldReductionManager::clone_manager(void) const
    //--------------------------------------------------------------------------
    {
      return context->create_reduction_manager(location, instance, redop, op, remote, true/*clone*/);
    }

    //--------------------------------------------------------------------------
    void FoldReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      // Assume that its all the fields for now
      // until we find a different way to do reductions on a subset of fields
      fields.push_back(Domain::CopySrcDstField(instance, 0/*offset*/, op->sizeof_rhs));
    }
    
    //--------------------------------------------------------------------------
    Event FoldReductionManager::issue_reduction(const std::vector<Domain::CopySrcDstField> &src_fields,
                                                const std::vector<Domain::CopySrcDstField> &dst_fields,
                                                Domain space, Event precondition, bool reduction_fold)
    //--------------------------------------------------------------------------
    {
      return space.copy(src_fields, dst_fields, precondition, redop, reduction_fold);
    }

    /////////////////////////////////////////////////////////////
    // Instance Key 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceKey::InstanceKey(void)
      : mid(0), handle(LogicalRegion::NO_REGION)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceKey::InstanceKey(UniqueManagerID id, LogicalRegion hand)
      : mid(id), handle(hand)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool InstanceKey::operator==(const InstanceKey &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((mid == rhs.mid) && (handle == rhs.handle));
    }

    //--------------------------------------------------------------------------
    bool InstanceKey::operator<(const InstanceKey &rhs) const
    //--------------------------------------------------------------------------
    {
      if (mid < rhs.mid)
        return true;
      else if (mid > rhs.mid)
        return false;
      else
        return (handle < rhs.handle);
    }

    /////////////////////////////////////////////////////////////
    // Physical View 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    PhysicalView::PhysicalView(RegionTreeForest *ctx, RegionNode *logical, bool made_local)
      : context(ctx), logical_region(logical), valid_references(0), local_view(made_local)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalView::~PhysicalView(void)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Instance View 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(InstanceManager *man, InstanceView *par, 
                               RegionNode *reg, RegionTreeForest *ctx, bool made_local)
      : PhysicalView(ctx, reg, made_local),
        manager(man), parent(par), active_children(0), to_be_invalidated(false)
    //--------------------------------------------------------------------------
    { 
    }

    //--------------------------------------------------------------------------
    InstanceView::~InstanceView(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (!manager->is_remote() && !manager->is_clone())
      {
        if (valid_references > 0)
          log_leak(LEVEL_WARNING,"Instance View for Instace %x from Logical Region (%x,%d,%d) still has %d valid references",
              manager->get_instance().id, logical_region->handle.index_space.id, logical_region->handle.field_space.id,
              logical_region->handle.tree_id, valid_references);
#ifdef DEBUG_HIGH_LEVEL
        assert(users.empty());
#endif
        if (!added_users.empty())
        {
          log_leak(LEVEL_WARNING,"Instance View for Instance %x from Logical Region (%x,%d,%d) still has %ld added users",
              manager->get_instance().id, logical_region->handle.index_space.id, logical_region->handle.field_space.id,
              logical_region->handle.tree_id, added_users.size());
          for (std::map<UniqueID,ReferenceTracker>::const_iterator it = added_users.begin();
                it != added_users.end(); it++)
          {
            log_leak(LEVEL_WARNING,"Instance View for Instance %x has user %d with %d references",
                manager->get_instance().id, it->first, it->second.get_reference_count());
          }
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    InstanceView* InstanceView::get_subview(Color pc, Color rc)
    //--------------------------------------------------------------------------
    {
      std::pair<Color,Color> key(pc,rc);
      if (children.find(key) == children.end())
      {
        // If it doesn't exist yet, make it, otherwise re-use it
        PartitionNode *pnode = logical_region->get_child(pc);
        RegionNode *rnode = pnode->get_child(rc);
        InstanceView *subview = context->create_instance_view(manager, this, rnode, true/*make local*/); 
        return subview;
      }
      return children[key];
    }

    //--------------------------------------------------------------------------
    void InstanceView::add_child_view(Color pc, Color rc, InstanceView *child)
    //--------------------------------------------------------------------------
    {
      ChildKey key(pc,rc);
      std::map<ChildKey,InstanceView*>::iterator finder = children.find(key);
      if (finder == children.end())
      {
        std::set<InstanceView*> aliased_set;
        // First go through all our current children and see if any of them alias
        // with the new child
        IndexSpaceNode *index_node = logical_region->row_source;
        for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          if (it->first.first != pc)
          {
            // Different partitions, see if they are disjoint or not
            if (!index_node->are_disjoint(it->first.first, pc))
            {
              aliased_set.insert(it->second);
              aliased_children[it->second].insert(child);
            }
          }
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(it->first.second != rc); // should never have the same key
#endif
            // Same partition, see if it is disjoint or if siblings are
            IndexPartNode *index_part = index_node->get_child(pc);
            if (!index_part->are_disjoint(it->first.second, rc))
            {
              aliased_set.insert(it->second);
              aliased_children[it->second].insert(child);
            }
          }
        }
        aliased_children[child] = aliased_set;
        
        // Then add the child
#ifdef DEBUG_HIGH_LEVEL
        assert(children.find(key) == children.end());
#endif
        children[key] = child;
        active_map[child] = true;
        active_children++;
      }
      else
      {
        // Mark the entry as being active again
        std::map<InstanceView*,bool>::iterator active_finder = active_map.find(finder->second);
#ifdef DEBUG_HIGH_LEVEL
        assert(active_finder != active_map.end());
#endif
        if (!active_finder->second)
        {
          active_children++;
          active_finder->second = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_child_view(const ChildKey &key)
    //--------------------------------------------------------------------------
    {
      std::map<ChildKey,InstanceView*>::iterator finder = children.find(key); 
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != children.end());
      assert(active_children > 0);
#endif
      std::map<InstanceView*,bool>::iterator active_finder = active_map.find(finder->second);
#ifdef DEBUG_HIGH_LEVEL
      assert(active_finder != active_map.end());
#endif
      // Mark that the entry as no longer active
      // TODO: hack for correctness, don't actually mark it as invalid
      //active_finder->second = false;
      //active_children--;
    }

    //--------------------------------------------------------------------------
    InstanceRef InstanceView::add_user(UniqueID uid, const PhysicalUser &user, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      // Find any dependences above or below for a specific user 
      std::set<Event> wait_on;
      if (parent != NULL)
        parent->find_dependences_above(wait_on, user, this);
      bool all_dominated = find_dependences_below(wait_on, user);
      Event wait_event = Event::merge_events(wait_on);
#ifdef LEGION_SPY
      if (!wait_event.exists())
      {
        UserEvent new_wait = UserEvent::create_user_event();
        new_wait.trigger();
        wait_event = new_wait;
      }
      LegionSpy::log_event_dependences(wait_on, wait_event);
#endif
      // Update the list of users
      bool use_single;
      std::map<UniqueID,ReferenceTracker>::iterator it = added_users.find(uid);
      if (it == added_users.end())
      {
        if (added_users.empty())
          check_state_change(true/*adding*/);
        added_users[uid] = ReferenceTracker(1, point);
        use_single = true;
      }
      else
      {
        use_single = it->second.add_reference(1, point);
      }
      // If we dominated everything then we can simply update the valid event,
      // otherwise we need to update the list of epoch users
      if (all_dominated && IS_WRITE(user.usage))
      {
        if (use_single)
          update_valid_event(user.single_term, user.field_mask);
        else
          update_valid_event(user.multi_term, user.field_mask);
      }
      else
      {
        std::pair<UniqueID,unsigned> key(uid,user.idx);
        std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::iterator finder = epoch_users.find(key);
        if (finder == epoch_users.end())
          epoch_users[key] = user;
#ifdef DEBUG_HIGH_LEVEL
        else // otherwise these all better be the same
        {
          assert(finder->second.idx == user.idx);
          assert(finder->second.usage == user.usage);
          assert(finder->second.field_mask == user.field_mask);
        }
#endif
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_state();
#endif
      return InstanceRef(wait_event, manager->get_location(), manager->get_instance(),
                          this, false/*copy*/, (IS_ATOMIC(user.usage) ? manager->get_lock() : Lock::NO_LOCK));
    }

    //--------------------------------------------------------------------------
    InstanceRef InstanceView::add_init_user(UniqueID uid, const PhysicalUser &user, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      // nothing to wait on since we're first
      Event wait_event = Event::NO_EVENT;
#ifdef LEGION_SPY
      if (!wait_event.exists())
      {
        UserEvent new_wait = UserEvent::create_user_event();
        new_wait.trigger();
        wait_event = new_wait;
      }
#endif
      check_state_change(true/*adding*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(added_users.find(uid) == added_users.end());
#endif
      added_users[uid] = ReferenceTracker(1, point);
      epoch_users[std::pair<UniqueID,unsigned>(uid,user.idx)] = user;
      return InstanceRef(wait_event, manager->get_location(), manager->get_instance(),
                          this, false/*copy*/, (IS_ATOMIC(user.usage) ? manager->get_lock() : Lock::NO_LOCK));
    }

    //--------------------------------------------------------------------------
    InstanceRef InstanceView::add_copy_user(ReductionOpID redop, 
                                            Event copy_term, const FieldMask &mask, bool reading)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(added_copy_users.find(copy_term) == added_copy_users.end());
      assert(!reading || (redop == 0));
#endif
      if (added_copy_users.empty())
        check_state_change(true/*adding*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(added_copy_users.find(copy_term) == added_copy_users.end());
#endif
      added_copy_users[copy_term] = redop;
      // If we're writing and this isn't a reduction op, then we can set
      // a new valid event
      if (!reading && (redop == 0))
        update_valid_event(copy_term, mask);
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(epoch_copy_users.find(copy_term) == epoch_copy_users.end());
#endif
        epoch_copy_users[copy_term] = mask;
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_state();
#endif
      return InstanceRef(copy_term, manager->get_location(), manager->get_instance(),
                          this, true/*copy*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_user(UniqueID uid, unsigned refs, bool force)
    //--------------------------------------------------------------------------
    {
      // deletions should only come out of the added users
      std::map<UniqueID,ReferenceTracker>::iterator it = added_users.find(uid);
      if ((it == added_users.end()) && !force)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_users.end());
#endif
      if (it->second.remove_reference(refs))
      {
#ifndef LEGION_SPY
        // Delete all the indexes for this unique ID
        {
          std::vector<unsigned> indexes_to_delete;
          for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
                it != epoch_users.end(); it++)
          {
            if (it->first.first == uid)
              indexes_to_delete.push_back(it->first.second);
          }
          for (std::vector<unsigned>::const_iterator it = indexes_to_delete.begin();
                it != indexes_to_delete.end(); it++)
          {
            epoch_users.erase(std::pair<UniqueID,unsigned>(uid,*it));
          }
        }
#else
        // If we're doing legion spy debugging, then keep it in the epoch users
        // and move it over to the deleted users 
        if (!force)
        {
          // Shouldn't need this now since we never delete points
          //it->second.use_multi = true; // everything use multi for safety in deleted users
          deleted_users.insert(*it);
        }
        else
        {
          std::vector<unsigned> indexes_to_delete;
          for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
                it != epoch_users.end(); it++)
          {
            if (it->first.first == uid)
              indexes_to_delete.push_back(it->first.second);
          }
          for (std::vector<unsigned>::const_iterator it = indexes_to_delete.begin();
                it != indexes_to_delete.end(); it++)
          {
            epoch_users.erase(std::pair<UniqueID,unsigned>(uid,*it));
          }
        }
#endif
        added_users.erase(it);
        if (added_users.empty())
          check_state_change(false/*adding*/);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_copy(Event copy_e, bool force)
    //--------------------------------------------------------------------------
    {
      // deletions should only come out of the added users
      std::map<Event,ReductionOpID>::iterator it = added_copy_users.find(copy_e);
      if ((it == added_copy_users.end()) && !force)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_copy_users.end());
#endif
#ifndef LEGION_SPY
      epoch_copy_users.erase(copy_e);
#else
      // If we're doing legion spy then don't keep it in the epoch users
      // and move it over to the deleted users
      if (!force)
        deleted_copy_users.insert(*it);
      else
        epoch_copy_users.erase(copy_e);
#endif
      added_copy_users.erase(it);
      if (added_copy_users.empty())
        check_state_change(false/*adding*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::add_valid_reference(void)
    //--------------------------------------------------------------------------
    {
      if (valid_references == 0)
        check_state_change(true/*adding*/);
      valid_references++;
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_valid_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_references > 0);
#endif
      valid_references--;
      to_be_invalidated = false;
      if (valid_references == 0)
        check_state_change(false/*adding*/);
    }

    //--------------------------------------------------------------------------
    void InstanceView::mark_to_be_invalidated(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_references > 0);
#endif
      to_be_invalidated = true;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::is_valid_view(void) const
    //--------------------------------------------------------------------------
    {
      return (!to_be_invalidated && (valid_references > 0));
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_war_dependence(const PhysicalUser &user) const
    //--------------------------------------------------------------------------
    {
      // Right now we'll just look for anything which might be reading this
      // instance that might cause a dependence.  A future optimization is
      // to check for things like simultaneous reductions which should be ok.
      if ((parent != NULL) && parent->has_war_dependence_above(user))
        return true;
      return has_war_dependence_below(user);
    }

    //--------------------------------------------------------------------------
    void InstanceView::copy_to(RegionMapper &rm, const FieldMask &copy_mask,
                               std::set<Event> &preconditions,
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
      find_copy_preconditions(preconditions, true/*writing*/, 0/*no reduction*/, copy_mask);
      manager->compute_copy_offsets(copy_mask, dst_fields);
    }

    //--------------------------------------------------------------------------
    bool InstanceView::reduce_to(ReductionOpID redop, 
                                 const FieldMask &copy_mask, std::set<Event> &preconditions,
                                 std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(redop > 0);
#endif
      find_copy_preconditions(preconditions, true/*writing*/, redop, copy_mask);
      manager->compute_copy_offsets(copy_mask, dst_fields);
      return false; // not a fold
    }

    //--------------------------------------------------------------------------
    void InstanceView::copy_from(RegionMapper &rm, const FieldMask &copy_mask,
                                 std::set<Event> &preconditions,
                                 std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      find_copy_preconditions(preconditions, false/*writing*/, 0/*no reduction*/, copy_mask);
      manager->compute_copy_offsets(copy_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    Event InstanceView::get_valid_event(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      std::set<Event> wait_on;
#ifdef DEBUG_HIGH_LEVEL
      bool dominated = 
#endif
      find_dependences_below(wait_on, true/*writing*/, 0, mask);
#ifdef DEBUG_HIGH_LEVEL
      assert(dominated);
#endif
      Event result = Event::merge_events(wait_on);
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(wait_on, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_copy_preconditions(std::set<Event> &wait_on, bool writing, 
                            ReductionOpID redop, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // Find any dependences above or below for a copy reader
      if (parent != NULL)
        parent->find_dependences_above(wait_on, writing, redop, mask, this);
      find_dependences_below(wait_on, writing, redop, mask);
    }

    //--------------------------------------------------------------------------
    void InstanceView::check_state_change(bool adding)
    //--------------------------------------------------------------------------
    {
      // Only add or remove references if the manager is remote,
      // otherwise it doesn't matter since the instance can't be collected anyway
      if (!manager->is_remote())
      {
        // This is the actual garbage collection case
        if ((valid_references == 0) && users.empty() && added_users.empty() &&
            copy_users.empty() && added_copy_users.empty())
        {
          if (adding)
            manager->add_reference();
          else
            manager->remove_reference();
        }
      }
      // Also check to see if we can remove this instance view from its
      // parent view.  Same condition as garbage collection plus we need
      // to not have any children (who could also still be valid)
      if ((parent != NULL) && (active_children == 0) && (valid_references == 0)
          && users.empty() && added_users.empty() && copy_users.empty()
          && added_copy_users.empty())
      {
        Color pc = logical_region->parent->row_source->color;
        Color rc = logical_region->row_source->color;
        if (adding)
          parent->add_child_view(pc, rc, this);
        else
          parent->remove_child_view(ChildKey(pc,rc));
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_dependences_above(std::set<Event> &wait_on, const PhysicalUser &user, 
                                              InstanceView* child)
    //--------------------------------------------------------------------------
    {
      find_local_dependences(wait_on, user);
      if (parent != NULL)
        parent->find_dependences_above(wait_on, user, this);
      // Also need to find any dependences in aliased children below
#ifdef DEBUG_HIGH_LEVEL
      assert(aliased_children.find(child) != aliased_children.end());
#endif
      const std::set<InstanceView*> &aliases = aliased_children[child];
      for (std::set<InstanceView*>::const_iterator it = aliases.begin();
            it != aliases.end(); it++)
      {
        std::map<InstanceView*,bool>::const_iterator finder = active_map.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != active_map.end());
#endif
        // Skip any aliased children that are not active
        if (!finder->second)
          continue;
        (*it)->find_dependences_below(wait_on, user);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_dependences_above(std::set<Event> &wait_on, bool writing, 
                                            ReductionOpID redop, const FieldMask &mask, InstanceView *child)
    //--------------------------------------------------------------------------
    {
      find_local_dependences(wait_on, writing, redop, mask);
      if (parent != NULL)
        parent->find_dependences_above(wait_on, writing, redop, mask, this);
      // Also need to find any dependences in aliased children below
#ifdef DEBUG_HIGH_LEVEL
      assert(aliased_children.find(child) != aliased_children.end());
#endif
      const std::set<InstanceView*> &aliases = aliased_children[child];
      for (std::set<InstanceView*>::const_iterator it = aliases.begin();
            it != aliases.end(); it++)
      {
        std::map<InstanceView*,bool>::const_iterator finder = active_map.find(*it);
        // Skip any aliased children that are not active
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != active_map.end());
#endif
        if (!finder->second)
          continue;
        (*it)->find_dependences_below(wait_on, writing, redop, mask);
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_dependences_below(std::set<Event> &wait_on, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      bool all_dominated = find_local_dependences(wait_on, user);
      for (std::map<InstanceView*,bool>::const_iterator it = active_map.begin();
            it != active_map.end(); it++)
      {
        // Skip any children which are not active
        if (!it->second)
          continue;
        bool dominated = it->first->find_dependences_below(wait_on, user);
        all_dominated = all_dominated && dominated;
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_dependences_below(std::set<Event> &wait_on, bool writing,
                                          ReductionOpID redop, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      bool all_dominated = find_local_dependences(wait_on, writing, redop, mask);
      for (std::map<InstanceView*,bool>::const_iterator it = active_map.begin();
            it != active_map.end(); it++)
      {
        // Skip any children which are not active
        if (!it->second)
          continue;
        bool dominated = it->first->find_dependences_below(wait_on, writing, redop, mask);
        all_dominated = all_dominated && dominated;
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_local_dependences(std::set<Event> &wait_on, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      // Can only dominate everyone if user is exclusive coherence, otherwise
      // we don't want to dominate everyone so somebody after us with the same 
      // coherence can com in later and run at the same time as us
      bool all_dominated = IS_EXCLUSIVE(user.usage);
      // Find any valid events we need to wait on
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        // Check for field disjointness
        if (!(it->second * user.field_mask))
        {
          wait_on.insert(it->first);
        }
      }
      // Also need to go through any aliased events
      for (std::vector<AliasedEvent>::const_iterator it = aliased_valid_events.begin();
            it != aliased_valid_events.end(); it++)
      {
        if (!(it->second * user.field_mask))
        {
          wait_on.insert(it->first);
        }
      }
      
      // Go through all of the current epoch users and see if we have any dependences
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        // Check for field disjointness 
        if (!(it->second.field_mask * user.field_mask))
        {
          DependenceType dtype = check_dependence_type(it->second.usage, user.usage);
          bool has_dependence = false;
          switch (dtype)
          {
            // Atomic and simultaneous are not dependences here since we know that they
            // are using the same physical instance
            case NO_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                all_dominated = false;
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Has a dependence, figure out which event to add
                has_dependence = true;
                break;
              }
            default:
              assert(false); // should never get here
          }
          if (has_dependence)
          {
            std::map<UniqueID,ReferenceTracker>::const_iterator finder = users.find(it->first.first);
            if (finder == users.end())
            {
              finder = added_users.find(it->first.first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_users.end());
#endif
#else
              if (finder == added_users.end())
              {
                finder = deleted_users.find(it->first.first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_users.end());
#endif
              }
#endif
            }
            if (finder->second.use_single_event())
              wait_on.insert(it->second.single_term);
            else
              wait_on.insert(it->second.multi_term);
          }
        }
      }
      // Check the aliased users too
      for (std::vector<AliasedUser>::const_iterator it = aliased_users.begin();
            it != aliased_users.end(); it++)
      {
        if (!(it->valid_mask * user.field_mask))
        {
          DependenceType dtype = check_dependence_type(it->user.usage, user.usage);
          switch (dtype)
          {
            // Atomic and simultaneous are not dependences here since we know that they
            // are using the same physical instance
            case NO_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                all_dominated = false;
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Has a dependence, figure out which event to add
                if (it->use_single)
                  wait_on.insert(it->user.single_term);
                else
                  wait_on.insert(it->user.multi_term);
                break;
              }
            default:
              assert(false); // should never get here
          }
        }
      }

      if (IS_READ_ONLY(user.usage))
      {
        // Wait for all reduction copy operations to finish
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          // Check for disjointnes on fields
          if (!(it->second * user.field_mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
#endif
            }
            if (finder->second != 0)
              wait_on.insert(finder->first);
            else
              all_dominated = false;
          }
        }
        // Also need to check the aliased reduction copy users
        for (std::vector<AliasedCopyUser>::const_iterator it = aliased_copy_users.begin();
              it != aliased_copy_users.end(); it++)
        {
          if (!(it->valid_mask * user.field_mask))
          {
            if (it->redop != 0)
              wait_on.insert(it->ready_event);
          }
        }
      }
      else if (IS_REDUCE(user.usage))
      {
        // Wait on all read operations and reductions of a different type
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          // Check for disjointnes on fields
          if (!(it->second * user.field_mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.end();
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
#endif
            }
            if (finder->second != user.usage.redop)
              wait_on.insert(finder->first);
            else
              all_dominated = false;
          }
        }
        // Also do any aliased reductions
        for (std::vector<AliasedCopyUser>::const_iterator it = aliased_copy_users.begin();
              it != aliased_copy_users.end(); it++)
        {
          if (!(it->valid_mask * user.field_mask))
          {
            if (it->redop != user.usage.redop)
              wait_on.insert(it->ready_event);
          }
        }
      }
      else
      {
        // Wait until all copy operations are done
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          // Check for disjointnes on fields
          if (!(it->second * user.field_mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
#endif
            }
            wait_on.insert(finder->first);
          }
        }
        // Also do any aliased copies
        for (std::vector<AliasedCopyUser>::const_iterator it = aliased_copy_users.begin();
              it != aliased_copy_users.end(); it++)
        {
          if (!(it->valid_mask * user.field_mask))
            wait_on.insert(it->ready_event);
        }
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::find_local_dependences(std::set<Event> &wait_on, bool writing,
                                            ReductionOpID redop, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      bool all_dominated = true;
      // Find any valid events we need to wait on
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        // Check for field disjointness
        if (!(it->second * mask))
        {
          wait_on.insert(it->first);
        }
      }
      // Also need to check any aliased users
      for (std::vector<AliasedEvent>::const_iterator it = aliased_valid_events.begin();
            it != aliased_valid_events.end(); it++)
      {
        if (!(it->second * mask))
        {
          wait_on.insert(it->first);
        }
      }

      if (writing)
      {
        // Record dependences on all users except ones with the same non-zero redop id
        for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
              it != epoch_users.end(); it++)
        {
          // check for field disjointness
          if (!(it->second.field_mask * mask))
          {
            if ((redop != 0) && (it->second.usage.redop == redop))
              all_dominated = false;
            else
            {
              std::map<UniqueID,ReferenceTracker>::const_iterator finder = users.find(it->first.first);
              if (finder == users.end())
              {
                finder = added_users.find(it->first.first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != added_users.end());
#endif
#else
                if (finder == added_users.end())
                {
                  finder = deleted_users.find(it->first.first);
#ifdef DEBUG_HIGH_LEVEL
                  assert(finder != deleted_users.end());
#endif
                }
#endif
              }
              // Pass in an empty domain point
              if (finder->second.use_single_event())
                wait_on.insert(it->second.single_term);
              else
                wait_on.insert(it->second.multi_term);
            }
          }
        }
        // Handle the aliased users
        for (std::vector<AliasedUser>::const_iterator it = aliased_users.begin();
              it != aliased_users.end(); it++)
        {
          if (!(it->valid_mask * mask))
          {
            if ((redop != it->user.usage.redop))
            {
              if (it->use_single)
                wait_on.insert(it->user.single_term);
              else
                wait_on.insert(it->user.multi_term);
            }
          }
        }
        // Also handle the copy users
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          if (!(it->second * mask))
          {
            if (redop != 0)
            {
              // If we're doing a reduction, see if they can happen in parallel
              std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
              if (finder == copy_users.end())
              {
                finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != added_copy_users.end());
#endif
#else
                if (finder == added_copy_users.end())
                {
                  finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                  assert(finder != deleted_copy_users.end());
#endif
                }
#endif
              }
              if (finder->second == redop)
                all_dominated = false;
              else
                wait_on.insert(it->first);
            }
            else
              wait_on.insert(it->first);
          }
        }
        // Handle the aliased copy users
        for (std::vector<AliasedCopyUser>::const_iterator it = aliased_copy_users.begin();
              it != aliased_copy_users.end(); it++)
        {
          if (!(it->valid_mask * mask))
          {
            if (redop != 0)
            {
              if (it->redop != redop)
                wait_on.insert(it->ready_event);
            }
            else
              wait_on.insert(it->ready_event);
          }
        }
      }
      else
      {
        // We're reading, find any users or copies that have a write that we need to wait for
        for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
              it != epoch_users.end(); it++)
        {
          if (!(it->second.field_mask * mask))
          {
            if (HAS_WRITE(it->second.usage))
            {
              std::map<UniqueID,ReferenceTracker>::const_iterator finder = users.find(it->first.first);
              if (finder == users.end())
              {
                finder = added_users.find(it->first.first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != added_users.end());
#endif
#else
                if (finder == added_users.end())
                {
                  finder = deleted_users.find(it->first.first);
#ifdef DEBUG_HIGH_LEVEL
                  assert(finder != deleted_users.end());
#endif
                }
#endif
              }
              // Pass in an empty domain point since this is a copy
              if (finder->second.use_single_event())
                wait_on.insert(it->second.single_term);
              else
                wait_on.insert(it->second.multi_term);
            }
            else
              all_dominated = false;
          }
        }
        for (std::vector<AliasedUser>::const_iterator it = aliased_users.begin();
              it != aliased_users.end(); it++)
        {
          if (!(it->valid_mask * mask))
          {
            if (HAS_WRITE(it->user.usage))
            {
              if (it->use_single)
                wait_on.insert(it->user.single_term);
              else
                wait_on.insert(it->user.multi_term);
            }
          }
        }
        // Also see if we have any copy users in non-reduction mode
        for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
              it != epoch_copy_users.end(); it++)
        {
          if (!(it->second * mask))
          {
            std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
            if (finder == copy_users.end())
            {
              finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != added_copy_users.end());
#endif
#else
              if (finder == added_copy_users.end())
              {
                finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
                assert(finder != deleted_copy_users.end());
#endif
              }
#endif
            }
            if (finder->second == 0)
              all_dominated = false;
            else
              wait_on.insert(it->first);
          }
        }
        // Handle the aliased copy users
        for (std::vector<AliasedCopyUser>::const_iterator it = aliased_copy_users.begin();
              it != aliased_copy_users.end(); it++)
        {
          if (!(it->valid_mask * mask))
          {
            if (it->redop != 0)
              wait_on.insert(it->ready_event);
          }
        }
      }
      return all_dominated;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_war_dependence_above(const PhysicalUser &user) const
    //--------------------------------------------------------------------------
    {
      if (has_local_war_dependence(user))
        return true;
      else if (parent != NULL)
        return parent->has_war_dependence_above(user);
      return false;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_war_dependence_below(const PhysicalUser &user) const
    //--------------------------------------------------------------------------
    {
      if (has_local_war_dependence(user))
        return true;
      for (std::map<InstanceView*,bool>::const_iterator it = active_map.begin();
            it != active_map.end(); it++)
      {
        // Skip any children which are not active
        if (!it->second)
          continue;
        if (it->first->has_war_dependence_below(user))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_local_war_dependence(const PhysicalUser &user) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(HAS_WRITE(user.usage));
#endif
      // If there is anyone who matches on this mask, then there is
      // a WAR dependence
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        // Check for field disjointness 
        if (!(it->second.field_mask * user.field_mask))
        {
          DependenceType dtype = check_dependence_type(it->second.usage, user.usage);
          switch (dtype)
          {
            case ANTI_DEPENDENCE:
              return true;
            case NO_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
            case TRUE_DEPENDENCE:
              continue;
            default:
              assert(false); // should never get here
          }
        }
      }
      // Check the aliased users too
      for (std::vector<AliasedUser>::const_iterator it = aliased_users.begin();
            it != aliased_users.end(); it++)
      {
        if (!(it->valid_mask * user.field_mask))
        {
          DependenceType dtype = check_dependence_type(it->user.usage, user.usage);
          switch (dtype)
          {
            case ANTI_DEPENDENCE:
              return true;
            case NO_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
            case TRUE_DEPENDENCE:
              continue;
            default:
              assert(false); // should never get here
          }
        }
      }
      // Only need to check masks here since we know the user is already
      // a writer of the instance
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (!(it->second * user.field_mask))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void InstanceView::update_valid_event(Event new_valid, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // Go through all the epoch users and remove ones from the new valid mask
      remove_invalid_users(epoch_users, mask);
      remove_invalid_elements<Event>(epoch_copy_users, mask);

      // Then update the set of valid events
      remove_invalid_elements<Event>(valid_events, mask);
      std::map<Event,FieldMask>::iterator finder = valid_events.find(new_valid);
      if (finder == valid_events.end())
        valid_events[new_valid] = mask;
      else
        finder->second |= mask;

      // Do it for all children regardless of whether they are active or not
      for (std::map<std::pair<Color,Color>,InstanceView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->second->update_valid_event(new_valid, mask);
      }
      
      // TODO: This is a hack to make sure we get the right answer, but once
      // we make this child view active it will never be unactive.  This will
      // always be correct, but at some point we should be able to deactivate it
      if (parent != NULL)
      {
        Color pc = logical_region->parent->row_source->color;
        Color rc = logical_region->row_source->color;
        parent->add_child_view(pc, rc, this);
      }
    }
    
    //--------------------------------------------------------------------------
    template<typename T>
    void InstanceView::remove_invalid_elements(std::map<T,FieldMask> &elements,
                                                      const FieldMask &new_mask)
    //--------------------------------------------------------------------------
    {
      typename std::vector<T> to_delete;
      for (typename std::map<T,FieldMask>::iterator it = elements.begin();
            it != elements.end(); it++)
      {
        it->second -= new_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (typename std::vector<T>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        elements.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::remove_invalid_users(std::map<std::pair<UniqueID,unsigned>,PhysicalUser> &filter_users,
                                            const FieldMask &new_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<UniqueID,unsigned> > to_delete;
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::iterator it = filter_users.begin();
            it != filter_users.end(); it++)
      {
        it->second.field_mask -= new_mask;
        if (!it->second.field_mask)
          to_delete.push_back(it->first);
      }
      for (std::vector<std::pair<UniqueID,unsigned> >::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        filter_users.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_send_size(const FieldMask &pack_mask)
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      result += sizeof(manager->unique_id);
      result += sizeof(parent->logical_region->handle);
      result += sizeof(logical_region->handle);
      result += sizeof(size_t); // number of valid events
      packing_sizes[0] = 0;
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(it->first);
        result += sizeof(it->second);
        packing_sizes[0]++;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(needed_trackers.empty());
#endif
      result += sizeof(size_t); // number of users
      packing_sizes[1] = 0;
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (it->second.field_mask * pack_mask)
          continue;
        result += sizeof(it->first);
        result += sizeof(it->second);
        needed_trackers[it->first.first] = -1;
        packing_sizes[1]++;
      }
      result += sizeof(size_t); // number of needed trackers
      for (std::map<UniqueID,int>::iterator it = needed_trackers.begin();
            it != needed_trackers.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->second == -1);
#endif
        std::map<UniqueID,ReferenceTracker>::const_iterator finder = users.find(it->first);
        if (finder == users.end())
        {
          finder = added_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_users.end());
#endif
          it->second = 1;
#else
          if (finder == added_users.end())
          {
            finder = deleted_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != deleted_users.end());
#endif
            it->second = 2;
          }
          else
            it->second = 1;
#endif
        }
        else
          it->second = 0;
        result += sizeof(it->first);
        result += finder->second.compute_tracker_send();
#ifdef DEBUG_HIGH_LEVEL
        assert(it->second != -1);
#endif
      }
      result += sizeof(size_t); // number of copy users
      packing_sizes[2] = 0;
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        result += sizeof(it->first);
        result += sizeof(it->second);
        result += sizeof(ReductionOpID);
        packing_sizes[2]++;
      }
      result += sizeof(size_t); // number of aliased users
      // Also need to pack any aliased users that we have here as well
      // as the ones that we've computed we need to send
      for (std::vector<AliasedUser>::const_iterator it = aliased_users.begin();
            it != aliased_users.end(); it++)
      {
        if (it->valid_mask * pack_mask)
          continue;
        packing_aliased_users.push_back(AliasedUser(it->valid_mask & pack_mask, it->user, it->use_single));
      }
      result += (packing_aliased_users.size() * sizeof(AliasedUser));
      result += sizeof(size_t); // number of aliased copy users
      for (std::vector<AliasedCopyUser>::const_iterator it = aliased_copy_users.begin();
            it != aliased_copy_users.end(); it++)
      {
        if (it->valid_mask * pack_mask)
          continue;
        packing_aliased_copy_users.push_back(AliasedCopyUser(it->valid_mask & pack_mask, it->ready_event, it->redop));
      }
      result += (packing_aliased_copy_users.size() * sizeof(AliasedCopyUser));
      result += sizeof(size_t); // number of aliased valid events
      for (std::vector<AliasedEvent>::const_iterator it = aliased_valid_events.begin();
            it != aliased_valid_events.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        packing_aliased_valid_events.push_back(AliasedEvent(it->first, it->second & pack_mask));
      }
      result += (packing_aliased_valid_events.size() * sizeof(AliasedEvent));
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_view_send(const FieldMask &pack_mask, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->unique_id);
      if (parent != NULL)
        rez.serialize(parent->logical_region->handle);
      else
        rez.serialize<LogicalRegion>(LogicalRegion::NO_REGION);
      rez.serialize(logical_region->handle);
      rez.serialize(packing_sizes[0]);
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        rez.serialize(it->first);
        rez.serialize(overlap);
      }
      rez.serialize(packing_sizes[1]);
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        FieldMask overlap = it->second.field_mask & pack_mask;
        if (!overlap)
          continue;
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(needed_trackers.size());
      for (std::map<UniqueID,int>::const_iterator it = needed_trackers.begin();
            it != needed_trackers.end(); it++)
      {
        rez.serialize(it->first);
        switch (it->second)
        {
          case 0:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(users.find(it->first) != users.end());
#endif
              users[it->first].pack_tracker_send(rez);
            }
          case 1:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(added_users.find(it->first) != added_users.end());
#endif
              added_users[it->first].pack_tracker_send(rez);
              break;
            }
#ifdef LEGION_SPY
          case 2:
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(deleted_users.find(it->first) != deleted_users.end());
#endif
              deleted_users[it->first].pack_tracker_send(rez);
              break;
            }
#endif
          default:
            assert(false); // bad location
        }
      }
      needed_trackers.clear();
      rez.serialize(packing_sizes[2]);
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        rez.serialize(it->first);
        rez.serialize(overlap);
        std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
        if (finder == copy_users.end())
        {
          finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_copy_users.end());
#endif
#else
          if (finder == added_copy_users.end())
          {
            finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != deleted_copy_users.end());
#endif
          }
#endif
        }
        rez.serialize(finder->second);
      }
      pack_vector<AliasedUser>(packing_aliased_users, rez);
      pack_vector<AliasedCopyUser>(packing_aliased_copy_users, rez);
      pack_vector<AliasedEvent>(packing_aliased_valid_events, rez);
      packing_aliased_users.clear();
      packing_aliased_copy_users.clear();
      packing_aliased_valid_events.clear();
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_view_send(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      InstanceManager *manager = context->find_instance_manager(mid);
      LogicalRegion parent_region_handle;
      derez.deserialize(parent_region_handle);
      // Note that if we only sent a part of the instance view tree then
      // its possible that the parent wasn't sent, so check to see if we
      // have it before asking to find it
      InstanceKey parent_key(manager->unique_id, parent_region_handle);
      InstanceView *parent = ((parent_region_handle == LogicalRegion::NO_REGION) ? NULL :
                              context->has_instance_view(parent_key) ? context->find_instance_view(parent_key) : NULL);
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionNode *reg_node = context->get_node(handle);

      InstanceView *result = context->create_instance_view(manager, parent, reg_node, false/*make local*/);
      // Now unpack everything
      size_t num_valid_events;
      derez.deserialize(num_valid_events);
      for (unsigned idx = 0; idx < num_valid_events; idx++)
      {
        Event valid_event;
        derez.deserialize(valid_event);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        result->valid_events[valid_event] = valid_mask;
      }
      size_t num_users;
      derez.deserialize(num_users);
      for (unsigned idx = 0; idx < num_users; idx++)
      {
        std::pair<UniqueID,unsigned> key;
        derez.deserialize(key);
        PhysicalUser user;
        derez.deserialize(user);
        result->epoch_users[key] = user;
      }
      size_t num_trackers;
      derez.deserialize(num_trackers);
      for (unsigned idx = 0; idx < num_trackers; idx++)
      {
        UniqueID uid;
        derez.deserialize(uid);
        result->users[uid].unpack_tracker_send(derez);
      }
      size_t num_copy_users;
      derez.deserialize(num_copy_users);
      for (unsigned idx = 0; idx < num_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        FieldMask copy_mask;
        derez.deserialize(copy_mask);
        ReductionOpID redop;
        derez.deserialize(redop);
        result->epoch_copy_users[copy_event] = copy_mask;
        result->copy_users[copy_event] = redop;
      }
      unpack_vector<AliasedUser>(result->aliased_users, derez);
      unpack_vector<AliasedCopyUser>(result->aliased_copy_users, derez);
      unpack_vector<AliasedEvent>(result->aliased_valid_events, derez);
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_required_views(std::map<InstanceView*,FieldMask> &unique_views,
            std::vector<InstanceView*> &ordered_views, const FieldMask &packing_mask, int filter)
    //--------------------------------------------------------------------------
    {
      // First add ourselves
      std::map<InstanceView*,FieldMask>::iterator finder = unique_views.find(this);
      if (finder != unique_views.end())
      {
        finder->second |= packing_mask;
      }
      else
      {
        unique_views[this] = packing_mask;
        ordered_views.push_back(this);
      }

      // Then compute any aliased users from above
      if (parent != NULL)
      {
        // Find any aliased users from above
        parent->find_aliased_above(packing_aliased_users, packing_aliased_copy_users,
                                    packing_aliased_valid_events, packing_mask, this);
      }

      // Finaly find users below, doing different things if we are filtered or not.
      // If we are filtered, add users in any filtered children that are not disjoint
      // with the target child to the aliased set.
      if (filter > -1)
      {
        Color child_filter = unsigned(filter);
        // Get all the children in the path that we want, otherwise check to see
        // if they are aliased, if they are add them to our aliased set
        for (std::map<ChildKey,InstanceView*>::const_iterator it = children.begin();
              it != children.end(); it++)
        {
          std::map<InstanceView*,bool>::const_iterator finder = active_map.find(it->second);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != active_map.end());
#endif
          // Skip any children which are not active
          if (!finder->second)
            continue;
          if (it->first.first == child_filter)
          {
            it->second->find_required_below(unique_views, ordered_views, packing_mask);
          }
          else
          {
            // Otherwise check to see if it is aliased
            if (!logical_region->row_source->are_disjoint(child_filter, it->first.first))
            {
              it->second->find_aliased_below(packing_aliased_users, packing_aliased_copy_users,
                                              packing_aliased_valid_events, packing_mask);
            }
          }
        }
      }
      else
      {
        // Get all the children
        for (std::map<InstanceView*,bool>::const_iterator it = active_map.begin();
              it != active_map.end(); it++)
        {
          // Skip any children which are not active
          if (!it->second)
            continue;
          it->first->find_required_below(unique_views, ordered_views, packing_mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_required_below(std::map<InstanceView*,FieldMask> &unique_views,
            std::vector<InstanceView*> &ordered_views, const FieldMask &packing_mask)
    //--------------------------------------------------------------------------
    {
      // First add ourselves
      std::map<InstanceView*,FieldMask>::iterator finder = unique_views.find(this);
      if (finder != unique_views.end())
      {
        finder->second |= packing_mask;
      }
      else
      {
        unique_views[this] = packing_mask;
        ordered_views.push_back(this);
      }
      // Now do any children that we might have
      for (std::map<InstanceView*,bool>::const_iterator it = active_map.begin();
            it != active_map.end(); it++)
      {
        // Skip children which are not active
        if (!it->second)
          continue;
        it->first->find_required_below(unique_views, ordered_views, packing_mask);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_aliased_above(std::vector<AliasedUser> &add_aliased_users,
                                          std::vector<AliasedCopyUser> &add_aliased_copies,
                                          std::vector<AliasedEvent> &add_aliased_events,
                                          const FieldMask &packing_mask, InstanceView *child_source)
    //--------------------------------------------------------------------------
    {
      // If we have a parent, go up first
      if (parent != NULL)
        parent->find_aliased_above(add_aliased_users, add_aliased_copies,
                                    add_aliased_events, packing_mask, this);

      // Now do ourselves
      find_aliased_local(add_aliased_users, add_aliased_copies,
                          add_aliased_events, packing_mask);

#ifdef DEBUG_HIGH_LEVEL
      assert(aliased_children.find(child_source) != aliased_children.end());
#endif
      // Now do any children which are aliased with the child source
      const std::set<InstanceView*> &aliases = aliased_children[child_source];
      for (std::set<InstanceView*>::const_iterator it = aliases.begin();
            it != aliases.end(); it++)
      {
        std::map<InstanceView*,bool>::const_iterator finder = active_map.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != active_map.end());
#endif
        // Skip any children which are not active
        if (!finder->second)
          continue;
        (*it)->find_aliased_below(add_aliased_users, add_aliased_copies,
                                  add_aliased_events, packing_mask);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_aliased_below(std::vector<AliasedUser> &add_aliased_users, 
                                          std::vector<AliasedCopyUser> &add_aliased_copies,
                                          std::vector<AliasedEvent> &add_aliased_events,
                                          const FieldMask &packing_mask)
    //--------------------------------------------------------------------------
    {
      // Do ourselves first, then do all our children
      find_aliased_local(add_aliased_users, add_aliased_copies,
                          add_aliased_events, packing_mask);

      for (std::map<InstanceView*,bool>::const_iterator it = active_map.begin();
            it != active_map.end(); it++)
      {
        // Skip any children which are not active
        if (!it->second)
          continue;
        it->first->find_aliased_below(add_aliased_users, add_aliased_copies,
                                        add_aliased_events, packing_mask);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::find_aliased_local(std::vector<AliasedUser> &add_aliased_users, 
                                          std::vector<AliasedCopyUser> &add_aliased_copies,
                                          std::vector<AliasedEvent> &add_aliased_events,
                                          const FieldMask &packing_mask)
    //--------------------------------------------------------------------------
    {
      // Go through all our different users and valid events and find the ones
      // that overlap with the packing mask
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (!(it->second.field_mask * packing_mask))
        {
          // Find it and add it to the list of aliased users
          std::map<UniqueID,ReferenceTracker>::const_iterator finder = users.find(it->first.first);
          if (finder == users.end())
          {
            finder = added_users.find(it->first.first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != added_users.end());
#endif
#else
            if (finder == added_users.end())
            {
              finder = deleted_users.find(it->first.first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != deleted_users.end());
#endif
            }
#endif
          }
          // Pass in an empty domain point for the comparison
          add_aliased_users.push_back(AliasedUser(it->second.field_mask & packing_mask, it->second, 
                                                  finder->second.use_single_event()));
        }
      }
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (!(it->second * packing_mask))
        {
          // Find it and add it to the list of aliased copy users
          std::map<Event,ReductionOpID>::const_iterator finder = copy_users.find(it->first);
          if (finder == copy_users.end())
          {
            finder = added_copy_users.find(it->first);
#ifndef LEGION_SPY
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != added_copy_users.end());
#endif
#else
            if (finder == added_copy_users.end())
            {
              finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != deleted_copy_users.end());
#endif
            }
#endif
          }
          add_aliased_copies.push_back(AliasedCopyUser(it->second & packing_mask, finder->first, finder->second)); 
        }
      }
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        if (!(it->second * packing_mask))
        {
          add_aliased_events.push_back(AliasedEvent(it->first, it->second & packing_mask));
        }
      }

      // Note we also have to do all the aliased users that we have here since 
      // they might also be aliased from somewhere else
      for (std::vector<AliasedUser>::const_iterator it = aliased_users.begin();
            it != aliased_users.end(); it++)
      {
        if (!(it->valid_mask * packing_mask))
        {
          add_aliased_users.push_back(AliasedUser(it->valid_mask & packing_mask, it->user, it->use_single));
        }
      }
      for (std::vector<AliasedCopyUser>::const_iterator it = aliased_copy_users.begin();
            it != aliased_copy_users.end(); it++)
      {
        if (!(it->valid_mask * packing_mask))
        {
          add_aliased_copies.push_back(AliasedCopyUser(it->valid_mask & packing_mask, it->ready_event, it->redop));
        }
      }
      for (std::vector<AliasedEvent>::const_iterator it = aliased_valid_events.begin();
            it != aliased_valid_events.end(); it++)
      {
        if (!(it->second * packing_mask))
        {
          add_aliased_events.push_back(AliasedEvent(it->first, it->second & packing_mask));
        }
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceView::has_added_users(void) const
    //--------------------------------------------------------------------------
    {
      return (!added_users.empty() || !added_copy_users.empty());
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_return_state_size(const FieldMask &pack_mask, bool overwrite,
            std::map<EscapedUser,unsigned> &escaped_users, std::set<EscapedCopy> &escaped_copies)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!manager->is_clone());
      sanity_check_state();
#endif
      // Pack all the valid events, the epoch users, and the epoch copy users,
      // also pack any added users that need to be sent back
      for (unsigned idx = 0; idx < 5; idx++)
        packing_sizes[idx] = 0;
      size_t result = 0;
      // Check to see if we have been returned before
      result += sizeof(bool); // local return
      result += sizeof(bool); //overwrite
      if (local_view)
      {
        result += sizeof(manager->unique_id);
        result += sizeof(LogicalRegion); // parent handle
        result += sizeof(logical_region->handle);
      }
      else
      {
        result += sizeof(manager->unique_id);
        result += sizeof(logical_region->handle);
      }
      result += sizeof(FieldMask);
      result += sizeof(size_t); // number of returning valid events
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        packing_sizes[0]++;
        result += sizeof(it->first);
        result += sizeof(it->second);
      }
      result += sizeof(size_t); // number of epoch users
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (it->second.field_mask * pack_mask)
          continue;
        // Only pack these up if we're overwriting or its new
        std::map<UniqueID,ReferenceTracker>::const_iterator finder = added_users.find(it->first.first);
        if (!overwrite && (finder == added_users.end())
#ifdef LEGION_SPY
              && (deleted_users.find(it->first.first) == deleted_users.end())
#endif
            )
          continue;
        packing_sizes[1]++;
        result += sizeof(it->first);
        result += sizeof(it->second);
        result += sizeof(bool); // returning
        // See if it is an added user that we need to get
        if (finder != added_users.end())
        {
          packing_sizes[3]++;
#ifdef LEGION_SPY
          result += sizeof(bool);
#endif
          result += sizeof(finder->first);
          result += finder->second.compute_tracker_return();
          // Add it to the list of escaped users
          escaped_users[EscapedUser(get_key(), it->first.first)] = finder->second.get_reference_count();
        }
#ifdef LEGION_SPY
        // make sure it is not a user before sending it back
        else if (users.find(it->first.first) == users.end())
        {
          finder = deleted_users.find(it->first.first);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != deleted_users.end());
#endif
          packing_sizes[3]++;
          result += sizeof(bool);
          result += sizeof(finder->first);
          result += finder->second.compute_tracker_return();
        }
#endif
      }
      result += sizeof(size_t); // number of epoch copy users
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        std::map<Event,ReductionOpID>::const_iterator finder = added_copy_users.find(it->first);
        // Only pack this up if we're overwriting or its new
        if (!overwrite && (finder == added_copy_users.end())
#ifdef LEGION_SPY
              && (deleted_copy_users.find(it->first) == deleted_copy_users.end())
#endif
            )
          continue;
        packing_sizes[2]++;
        result += sizeof(it->first);
        result += sizeof(it->second);
        result += sizeof(bool); // returning
        if (finder != added_copy_users.end())
        {
          packing_sizes[4]++;
#ifdef LEGION_SPY
          result += sizeof(bool);
#endif
          result += sizeof(finder->first);
          result += sizeof(finder->second);
          // Add it to the list of escaped copies
          escaped_copies.insert(EscapedCopy(get_key(), finder->first));
        }
#ifdef LEGION_SPY
        // make sure it is not a user before sending it back
        else if(copy_users.find(it->first) == copy_users.end())
        {
          finder = deleted_copy_users.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != deleted_copy_users.end());
#endif
          packing_sizes[4]++;
          result += sizeof(bool);
          result += sizeof(finder->first);
          result += sizeof(finder->second);
        }
#endif
      }
      result += sizeof(size_t); // number of added users
      result += sizeof(size_t); // number of added copy users that needs to be sent back

      return result; 
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_return_users_size(std::map<EscapedUser,unsigned> &escaped_users,
                std::set<EscapedCopy> &escaped_copies, bool already_returning, const FieldMask &returning_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!manager->is_clone());
#endif
      // Zero out the packing sizes
      size_t result = 0;
      // Check to see if we've returned before 
      result += sizeof(local_view);
      if (local_view && !already_returning)
      {
        result += sizeof(manager->unique_id);
        result += sizeof(LogicalRegion); // parent handle
        result += sizeof(logical_region->handle);
      }
      else
      {
        result += sizeof(manager->unique_id);
        result += sizeof(logical_region->handle);
      }
      result += sizeof(size_t); // number of added users
      packing_sizes[5] = added_users.size();
      if (already_returning)
      {
        // Find the set of added users not in the epoch users
        for (std::map<UniqueID,ReferenceTracker>::const_iterator it = added_users.begin();
              it != added_users.end(); it++)
        {
          bool has_epoch_user = false;
          for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator eit = epoch_users.begin();
                eit != epoch_users.end(); eit++)
          {
            if (eit->first.first != it->first)
              continue;
            if (eit->second.field_mask * returning_mask)
              continue;
            has_epoch_user = true;;
            break;
          }
          if (has_epoch_user)
            packing_sizes[5]--;
          else
            result += it->second.compute_tracker_return();
        }
      }
      result += (packing_sizes[5] * sizeof(UniqueID));
      result += sizeof(size_t); // number of added copy users
      packing_sizes[6] = added_copy_users.size();
      if (already_returning)
      {
        // Find the set of added copy users not in the epoch users
        // that are going to be send back
        for (std::map<Event,ReductionOpID>::const_iterator it = added_copy_users.begin();
              it != added_copy_users.end(); it++)
        {
          std::map<Event,FieldMask>::const_iterator finder = epoch_copy_users.find(it->first);
          if ((finder != epoch_copy_users.end()) &&
              !(finder->second * returning_mask))
          {
            packing_sizes[6]--;
          }
        }
      }
      result += (packing_sizes[6] * (sizeof(Event) + sizeof(ReductionOpID)));
      // Update the esacped references
      for (std::map<UniqueID,ReferenceTracker>::const_iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        escaped_users[EscapedUser(get_key(), it->first)] = it->second.get_reference_count();
      }
      for (std::map<Event,ReductionOpID>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        escaped_copies.insert(EscapedCopy(get_key(), it->first));
      }

      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_return_state(const FieldMask &pack_mask, bool overwrite, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_state();
#endif
      rez.serialize(local_view);
      rez.serialize(overwrite);
      if (local_view)
      {
        rez.serialize(manager->unique_id);
        if (parent == NULL)
          rez.serialize(LogicalRegion::NO_REGION);
        else
          rez.serialize(parent->logical_region->handle);
        rez.serialize(logical_region->handle);
        // Mark that this is no longer a local view since it has been returned
        local_view = false;
      }
      else
      {
        rez.serialize(manager->unique_id);
        rez.serialize(logical_region->handle);
      }
      // A quick thought about not about packing field masks here.  Even though there may be fields
      // that have been created that haven't been returned, we know that none of these
      // fields are newly created fields, because the only states returned by this series
      // of function calls is based on fields dictated by region requirements which means
      // that they had to be created in the parent context and therefore already exist
      // in the owning node.
      rez.serialize(pack_mask);
      rez.serialize(packing_sizes[0]); // number of returning valid events
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(packing_sizes[1]); // number of returning epoch users 
      std::set<UniqueID> return_add_users;
#ifdef LEGION_SPY
      std::set<UniqueID> return_deleted_users;
#endif
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        if (it->second.field_mask * pack_mask)
          continue;
        std::map<UniqueID,ReferenceTracker>::const_iterator finder = added_users.find(it->first.first);
        if (!overwrite && (finder == added_users.end())
#ifdef LEGION_SPY
              && (deleted_users.find(it->first.first) == deleted_users.end())
#endif
            )
          continue;
        rez.serialize(it->first);
        rez.serialize(it->second);
        if (finder != added_users.end())
        {
          return_add_users.insert(it->first.first);
          rez.serialize(true); // has returning
        }
#ifdef LEGION_SPY
        else if (deleted_users.find(it->first.first) != deleted_users.end())
        {
          return_deleted_users.insert(it->first.first);
          rez.serialize(true); // has returning
        }
#endif
        else
          rez.serialize(false); // has returning
      }
#ifdef DEBUG_HIGH_LEVEL
#ifndef LEGION_SPY
      assert(return_add_users.size() == packing_sizes[3]);
#else
      assert((return_add_users.size() + return_deleted_users.size()) == packing_sizes[3]);
#endif
#endif
      rez.serialize(packing_sizes[2]);
      std::vector<Event> return_copy_users;
#ifdef LEGION_SPY
      std::vector<Event> return_deleted_copy_users;
#endif
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        if (it->second * pack_mask)
          continue;
        std::map<Event,ReductionOpID>::const_iterator finder = added_copy_users.find(it->first);
        if (!overwrite && (finder == added_copy_users.end())
#ifdef LEGION_SPY
              && (deleted_copy_users.find(it->first) == deleted_copy_users.end())
#endif
            )
          continue;
        rez.serialize(it->first);
        rez.serialize(it->second);
        if (finder != added_copy_users.end())
        {
          return_copy_users.push_back(it->first);
          rez.serialize(true); // has returning
        }
#ifdef LEGION_SPY
        else if (deleted_copy_users.find(it->first) != deleted_copy_users.end())
        {
          return_deleted_copy_users.push_back(it->first);
          rez.serialize(true); // has returning
        }
#endif
        else
          rez.serialize(false); // has returning
      }
#ifdef DEBUG_HIGH_LEVEL
#ifndef LEGION_SPY
      assert(return_copy_users.size() == packing_sizes[4]);
#else
      assert((return_copy_users.size() + return_deleted_copy_users.size()) == packing_sizes[4]);
#endif
#endif
      rez.serialize(packing_sizes[3]);
      if (packing_sizes[3] > 0)
      {
        for (std::set<UniqueID>::const_iterator it = return_add_users.begin();
              it != return_add_users.end(); it++)
        {
          std::map<UniqueID,ReferenceTracker>::iterator finder = added_users.find(*it);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_users.end());
#endif
#ifdef LEGION_SPY
          rez.serialize<bool>(true);
#endif
          rez.serialize(*it);
          finder->second.pack_tracker_return(rez);
          // Remove it from the added users and put it in the users
#ifdef DEBUG_HIGH_LEVEL
          assert(users.find(*it) == users.end());
#endif
          users.insert(*finder);
          added_users.erase(finder);
        }
#ifdef LEGION_SPY
        for (std::set<UniqueID>::const_iterator it = return_deleted_users.begin();
              it != return_deleted_users.end(); it++)
        {
          rez.serialize<bool>(false);
          rez.serialize(*it);
          deleted_users[*it].pack_tracker_return(rez);
        }
#endif
      }
      rez.serialize(packing_sizes[4]);
      if (packing_sizes[4] > 0)
      {
        for (std::vector<Event>::const_iterator it = return_copy_users.begin();
              it != return_copy_users.end(); it++)
        {
          std::map<Event,ReductionOpID>::iterator finder = added_copy_users.find(*it);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != added_copy_users.end());
#endif
#ifdef LEGION_SPY
          rez.serialize<bool>(true);
#endif
          rez.serialize(*it);
          rez.serialize(finder->second);
          // Remove it from the list of added copy users and make it a user
#ifdef DEBUG_HIGH_LEVEL
          assert(copy_users.find(*it) == copy_users.end());
#endif
          copy_users.insert(*finder);
          added_copy_users.erase(finder);
        }
#ifdef LEGION_SPY
        for (std::vector<Event>::const_iterator it = return_deleted_copy_users.begin();
              it != return_deleted_copy_users.end(); it++)
        {
          rez.serialize<bool>(false);
          rez.serialize(*it);
          rez.serialize(deleted_copy_users[*it]);
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_return_users(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(local_view);
      if (local_view)
      {
        rez.serialize(manager->unique_id);
        if (parent == NULL)
          rez.serialize(LogicalRegion::NO_REGION);
        else
          rez.serialize(parent->logical_region->handle);
        rez.serialize(logical_region->handle);
        // Mark that this is no longer a local view since it has been returned
        local_view = false;
      }
      else
      {
        rez.serialize(manager->unique_id);
        rez.serialize(logical_region->handle);
      }    
#ifdef DEBUG_HIGH_LEVEL
      assert(added_users.size() == packing_sizes[5]);
#endif
      rez.serialize(added_users.size());
      for (std::map<UniqueID,ReferenceTracker>::iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        rez.serialize(it->first);
        it->second.pack_tracker_return(rez);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(added_copy_users.size() == packing_sizes[6]);
#endif
      rez.serialize(added_copy_users.size());
      for (std::map<Event,ReductionOpID>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      // Now move everything over to the users and copy users
      users.insert(added_users.begin(), added_users.end());
      added_users.clear();
      copy_users.insert(added_copy_users.begin(), added_copy_users.end());
      added_copy_users.clear();
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_return_state(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      bool local_view, overwrite;
      derez.deserialize(local_view);
      derez.deserialize(overwrite);
      InstanceView *result = NULL;
      if (local_view)
      {
        UniqueManagerID mid;
        LogicalRegion parent, handle;
        derez.deserialize(mid);
        derez.deserialize(parent);
        derez.deserialize(handle);
        // See if the view already exists (another child could have already have made it and returned it)
        InstanceKey key(mid, handle);
        if (context->has_instance_view(key))
        {
          result = context->find_instance_view(key);
        }
        else
        {
          InstanceManager *manager = context->find_instance_manager(mid);
          RegionNode *node = context->get_node(handle);
          InstanceView *parent_node = ((parent == LogicalRegion::NO_REGION) ? NULL : context->find_instance_view(InstanceKey(mid,parent)));
          result = context->create_instance_view(manager, parent_node, node, true/*made local*/);
        }
      }
      else
      {
        // This better already exist
        UniqueManagerID mid;
        LogicalRegion handle;
        derez.deserialize(mid);
        derez.deserialize(handle);
        result = context->find_instance_view(InstanceKey(mid, handle));
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      FieldMask unpack_mask;
      derez.deserialize(unpack_mask);
      // If we're overwriting, clear stuff out
      if (overwrite)
      {
        result->remove_invalid_elements<Event>(result->valid_events, unpack_mask); 
        result->remove_invalid_users(result->epoch_users, unpack_mask);
        result->remove_invalid_elements<Event>(result->epoch_copy_users, unpack_mask);
      }
      // Now we can unpack everything and add it
      size_t new_valid_events;
      derez.deserialize(new_valid_events);
      for (unsigned idx = 0; idx < new_valid_events; idx++)
      {
        Event valid_event;
        derez.deserialize(valid_event);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        if (result->valid_events.find(valid_event) == result->valid_events.end())
          result->valid_events[valid_event] = valid_mask;
        else
          result->valid_events[valid_event] |= valid_mask;
      }
      size_t new_epoch_users;
      derez.deserialize(new_epoch_users);
      for (unsigned idx = 0; idx < new_epoch_users; idx++)
      {
        std::pair<UniqueID,unsigned> key;
        derez.deserialize(key);
        PhysicalUser user;
        derez.deserialize(user);
        bool has_returning;
        derez.deserialize(has_returning);
        // It's possible that epoch users were removed locally while we were
        // remote, in which case if this user isn't marked as returning
        // we should check to make sure that there is a user before putting
        // it in the set of epoch users.  Note we don't need to do this for LEGION_SPY
        // since we know that the user already exists in one of the sets of users
        // (possibly the deleted ones).
#ifndef LEGION_SPY
        if (!has_returning && (result->users.find(key.first) == result->users.end())
            && (result->added_users.find(key.first) == result->added_users.end()))
          continue;
#ifdef DEBUG_HIGH_LEVEL
        if (result->epoch_users.find(key) != result->epoch_users.end())
        {
          assert(result->epoch_users[key].usage == user.usage);
          assert(result->epoch_users[key].single_term == user.single_term);
          assert(result->epoch_users[key].multi_term == user.multi_term);
          assert(result->epoch_users[key].idx == user.idx);
        }
#endif
#endif
        std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::iterator finder = result->epoch_users.find(key);
        if (finder == result->epoch_users.end())
          result->epoch_users[key] = user;
        else
          finder->second.field_mask |= user.field_mask;
      }
      size_t new_epoch_copy_users;
      derez.deserialize(new_epoch_copy_users);
      for (unsigned idx = 0; idx < new_epoch_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        FieldMask valid_mask;
        derez.deserialize(valid_mask);
        bool has_returning;
        derez.deserialize(has_returning);
        // See the note above about users.  The same thing applies here
#ifndef LEGION_SPY
        if (!has_returning && (result->copy_users.find(copy_event) == result->copy_users.end())
            && (result->added_copy_users.find(copy_event) == result->added_copy_users.end()))
          continue;
#endif
        std::map<Event,FieldMask>::iterator finder = result->epoch_copy_users.find(copy_event); 
        if (finder == result->epoch_copy_users.end())
          result->epoch_copy_users[copy_event] = valid_mask;
        else
          finder->second |= valid_mask;
      }
      size_t new_added_users;
      derez.deserialize(new_added_users);
      if (result->added_users.empty() && (new_added_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < new_added_users; idx++)
      {
#ifdef LEGION_SPY
        bool is_added;
        derez.deserialize(is_added);
#endif
        UniqueID uid;
        derez.deserialize(uid);
#ifdef LEGION_SPY
        if (is_added)
          result->added_users[uid].unpack_tracker_return(derez);
        else
          result->deleted_users[uid].unpack_tracker_return(derez);
#else
        result->added_users[uid].unpack_tracker_return(derez);
#endif
      }
      size_t new_added_copy_users;
      derez.deserialize(new_added_copy_users);
      if (result->added_copy_users.empty() && (new_added_copy_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < new_added_copy_users; idx++)
      {
#ifdef LEGION_SPY
        bool is_added;
        derez.deserialize(is_added);
#endif
        Event copy_event;
        derez.deserialize(copy_event);
        ReductionOpID redop;
        derez.deserialize(redop);
#ifdef DEBUG_HIGH_LEVEL
        if (result->added_copy_users.find(copy_event) != result->added_copy_users.end())
          assert(result->added_copy_users[copy_event] == redop);
#endif
#ifdef LEGION_SPY
        if (is_added) {
#endif
        result->added_copy_users[copy_event] = redop;
#ifdef LEGION_SPY
        }
        else
          result->deleted_copy_users[copy_event] = redop;
#endif
      }
#ifdef DEBUG_HIGH_LEVEL
      result->sanity_check_state();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_return_users(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      bool local_view;
      derez.deserialize(local_view);
      InstanceView *result = NULL;
      if (local_view)
      {
        UniqueManagerID mid;
        LogicalRegion parent, handle;
        derez.deserialize(mid);
        derez.deserialize(parent);
        derez.deserialize(handle);
        // See if the view already exists (another child could have already have made it and returned it)
        InstanceKey key(mid, handle);
        if (context->has_instance_view(key))
        {
          result = context->find_instance_view(key);
        }
        else
        {
          InstanceManager *manager = context->find_instance_manager(mid);
          RegionNode *node = context->get_node(handle);
          InstanceView *parent_node = ((parent == LogicalRegion::NO_REGION) ? NULL : context->find_instance_view(InstanceKey(mid,parent)));
          result = context->create_instance_view(manager, parent_node, node, true/*made local*/);
        }
      }
      else
      {
        // This better already exist
        UniqueManagerID mid;
        LogicalRegion handle;
        derez.deserialize(mid);
        derez.deserialize(handle);
        result = context->find_instance_view(InstanceKey(mid, handle));
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      size_t num_added_users;
      derez.deserialize(num_added_users);
      if (result->added_users.empty() && (num_added_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < num_added_users; idx++)
      {
        UniqueID uid;
        derez.deserialize(uid);
        result->added_users[uid].unpack_tracker_return(derez);
      }
      size_t num_added_copy_users;
      derez.deserialize(num_added_copy_users);
      if (result->added_copy_users.empty() && (num_added_copy_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < num_added_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        ReductionOpID redop;
        derez.deserialize(redop);
        result->added_copy_users[copy_event] = redop;
      }
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::compute_simple_return(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(local_view);
#endif
      size_t result = 0;
      result += sizeof(manager->unique_id);
      result += sizeof(LogicalRegion); // parent handle
      result += sizeof(logical_region->handle);
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::pack_simple_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->unique_id);
      if (parent == NULL)
        rez.serialize(LogicalRegion::NO_REGION);
      else
        rez.serialize(parent->logical_region->handle);
      rez.serialize(logical_region->handle);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::unpack_simple_return(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      LogicalRegion handle, parent_handle;
      derez.deserialize(mid);
      derez.deserialize(parent_handle);
      derez.deserialize(handle);
      // Check to see if it has already been created
      if (!context->has_instance_view(InstanceKey(mid, handle)))
      {
        // Then we need to make it
        InstanceManager *manager = context->find_instance_manager(mid);
        InstanceView *parent = NULL;
        if (!(parent_handle == LogicalRegion::NO_REGION))
          parent = context->find_instance_view(InstanceKey(mid, parent_handle));
        RegionNode *node = context->get_node(handle);
        context->create_instance_view(manager, parent, node, true/*made local*/);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ void InstanceView::pack_vector(const std::vector<T> &to_pack, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(to_pack.size());
      for (typename std::vector<T>::const_iterator it = to_pack.begin();
            it != to_pack.end(); it++)
      {
        rez.serialize<T>(*it);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ void InstanceView::unpack_vector(std::vector<T> &target, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_elems;
      derez.deserialize<size_t>(num_elems);
      size_t old_size = target.size();
      target.resize(old_size + num_elems);
      for (unsigned idx = old_size; idx < target.size(); idx++)
      {
        derez.deserialize<T>(target[idx]);
      }
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void InstanceView::sanity_check_state(void) const
    //--------------------------------------------------------------------------
    {
      // Big sanity check
      // Each valid event should have exactly one valid field

      // This is not a valid check if we can have tasks that map in parallel but may
      // issue multiple copies to update the same data.  See the comment in map_physical_region.
      // We can now have multiple valid events for
      // a field if two people in read-only mode issue the same copy to update a field
      // in parallel.  All the rest of the code for finding valid events will record
      // all the valid events for a field as a prerequisite.
#if 0
      FieldMask valid_shadow;
      for (std::map<Event,FieldMask>::const_iterator it = valid_events.begin();
            it != valid_events.end(); it++)
      {
        assert(!(valid_shadow & it->second));
        valid_shadow |= it->second;
      }
#endif
      // There should be an entry for each epoch user
      for (std::map<std::pair<UniqueID,unsigned>,PhysicalUser>::const_iterator it = epoch_users.begin();
            it != epoch_users.end(); it++)
      {
        assert((users.find(it->first.first) != users.end())
               || (added_users.find(it->first.first) != added_users.end())
#ifdef LEGION_SPY
               || (deleted_users.find(it->first.first) != deleted_users.end())
#endif
               );
      }
      // Same thing for epoch copy users
      for (std::map<Event,FieldMask>::const_iterator it = epoch_copy_users.begin();
            it != epoch_copy_users.end(); it++)
      {
        assert((copy_users.find(it->first) != copy_users.end())
               || (added_copy_users.find(it->first) != added_copy_users.end())
#ifdef LEGION_SPY
               || (deleted_copy_users.find(it->first) != deleted_copy_users.end())
#endif
            );
      }
    }
#endif

    //--------------------------------------------------------------------------
    bool InstanceView::ReferenceTracker::add_reference(unsigned refs, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      references += refs;
      points.insert(point);
      return (points.size() <= 1);
    }

    //--------------------------------------------------------------------------
    bool InstanceView::ReferenceTracker::remove_reference(unsigned refs) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references >= refs);
#endif
      references -= refs;
      return (references == 0);
    }

    //--------------------------------------------------------------------------
    bool InstanceView::ReferenceTracker::use_single_event(void) const
    //--------------------------------------------------------------------------
    {
      // Only use the single event in the case that we have multiple points
      // that are accessing the region
      if (points.size() <= 1)
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    unsigned InstanceView::ReferenceTracker::get_reference_count(void) const
    //--------------------------------------------------------------------------
    {
      return references;
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::ReferenceTracker::compute_tracker_send(void) const
    //--------------------------------------------------------------------------
    {
      // Only need to send the points that are valid
      // If we already have more than two, we only need to send two since
      // that is sufficient to trigger the use of the multi event
      size_t result = sizeof(size_t);
      size_t num_points = points.size();
      if (num_points > 2)
        num_points = 2;
      result += (num_points * sizeof(DomainPoint));
      // No need to send the references since they can't be removed remotely anyway
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::ReferenceTracker::pack_tracker_send(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      size_t num_points = points.size();
      if (num_points > 2)
        num_points = 2;
      rez.serialize(num_points);
      std::set<DomainPoint>::const_iterator it = points.begin();
      while (num_points > 0)
      {
        rez.serialize(*it);
        it++;
        num_points--;
      }
    }

    //--------------------------------------------------------------------------
    void InstanceView::ReferenceTracker::unpack_tracker_send(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_points;
      derez.deserialize(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint p;
        derez.deserialize(p);
        points.insert(p);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(num_points == points.size());
#endif
    }

    //--------------------------------------------------------------------------
    size_t InstanceView::ReferenceTracker::compute_tracker_return(void) const
    //--------------------------------------------------------------------------
    {
      // Only need to send back at most two points
      size_t result = sizeof(size_t);
      size_t num_points = points.size();
      if (num_points > 2)
        num_points = 2;
      result += (num_points * sizeof(DomainPoint));
      result += sizeof(references);
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceView::ReferenceTracker::pack_tracker_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      size_t num_points = points.size();
      if (num_points > 2)
        num_points = 2;
      rez.serialize(num_points);
      std::set<DomainPoint>::const_iterator it = points.begin();
      while (num_points > 0)
      {
        rez.serialize(*it);
        it++;
        num_points--;
      }
      rez.serialize(references);
    }

    //--------------------------------------------------------------------------
    void InstanceView::ReferenceTracker::unpack_tracker_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_points;
      derez.deserialize(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint p;
        derez.deserialize(p);
        points.insert(p);
      }
      unsigned new_references;
      derez.deserialize(new_references);
      references += new_references;
    }

    /////////////////////////////////////////////////////////////
    // Reduction View 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(RegionTreeForest *ctx, RegionNode *logical,
                                  bool made_local, ReductionManager *man)
      : PhysicalView(ctx, logical, made_local), manager(man), to_be_invalidated(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReductionView::~ReductionView(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (!manager->is_remote() && !manager->is_clone())
      {
        if (valid_references > 0)
          log_leak(LEVEL_WARNING,"Reduction View for Reduction Instace %x from Logical Region (%x,%d,%d) still has %d valid references",
              manager->get_instance().id, logical_region->handle.index_space.id, logical_region->handle.field_space.id,
              logical_region->handle.tree_id, valid_references);
      }
#endif
    }

    //--------------------------------------------------------------------------
    InstanceRef ReductionView::add_user(UniqueID uid, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(user.usage.redop == manager->redop);
#endif
      // Always should add to the added users
      std::map<UniqueID,TaskUser>::iterator finder = added_users.find(uid);
      if (finder != added_users.end())
      {
        // Already exists so update the reference count
        finder->second.references++;
        finder->second.use_multi = true;
      }
      else
      {
        if (added_users.empty())
          check_state_change(true/*adding*/);
        // Doesn't exist yet, so make one
        added_users[uid] = TaskUser(user, 1/*num refs*/);
      }
      // Find the event precondition for using this instance 
      // Have to wait for all previous readers to be done
      std::set<Event> preconditions;
      find_read_preconditions(copy_users, user.field_mask, preconditions);
      find_read_preconditions(added_copy_users, user.field_mask, preconditions);
#ifdef LEGION_SPY
      find_read_preconditions(deleted_copy_users, user.field_mask, preconditions);
#endif
      Event wait_event = Event::merge_events(preconditions);
#ifdef LEGION_SPY
      if (!wait_event.exists())
      {
        UserEvent new_wait = UserEvent::create_user_event();
        new_wait.trigger();
        wait_event = new_wait;
      }
      LegionSpy::log_event_dependences(preconditions, wait_event);
#endif
      return InstanceRef(wait_event, manager->get_location(), manager->get_instance(), this);
    }

    //--------------------------------------------------------------------------
    InstanceRef ReductionView::add_init_user(UniqueID uid, const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(user.usage.redop == manager->redop);
#endif
      check_state_change(true/*adding*/);
      added_users[uid] = TaskUser(user, 1/*num refs*/);
      Event wait_event = Event::NO_EVENT;
#ifdef LEGION_SPY
      if (!wait_event.exists())
      {
        UserEvent new_wait = UserEvent::create_user_event();
        new_wait.trigger();
        wait_event = new_wait;
      }
#endif
      return InstanceRef(wait_event, manager->get_location(), manager->get_instance(), this);
    }

    //--------------------------------------------------------------------------
    InstanceRef ReductionView::add_copy_user(ReductionOpID redop, Event copy_term, const FieldMask &mask, bool reading)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop);
      assert(added_copy_users.find(copy_term) == added_copy_users.end());
#endif
      if (added_copy_users.empty())
        check_state_change(true/*adding*/);
      added_copy_users[copy_term] = CopyUser(reading, mask);
      return InstanceRef(copy_term, manager->get_location(), manager->get_instance(), this, true/*copy*/);
    }

    //--------------------------------------------------------------------------
    void ReductionView::remove_user(UniqueID uid, unsigned refs, bool force)
    //--------------------------------------------------------------------------
    {
      // deletions should only come out of the added users
      std::map<UniqueID,TaskUser>::iterator it = added_users.find(uid);
      if ((it == added_users.end()) && !force)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_users.end());
      assert(it->second.references > 0);
#endif
      it->second.references--;
      if (it->second.references == 0)
      {
#ifdef LEGION_SPY
        if (!force)
        {
          it->second.use_multi = true; // use multi for deleted users to be safe
          deleted_users.insert(*it);
        }
#endif 
        added_users.erase(it);
        if (added_users.empty())
          check_state_change(false/*adding*/);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::remove_copy(Event copy, bool force)
    //--------------------------------------------------------------------------
    {
      std::map<Event,CopyUser>::iterator it = added_copy_users.find(copy);
      if ((it == added_copy_users.end()) && !force)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(it != added_copy_users.end());
#endif
#ifdef LEGION_SPY
      if (!force)
        deleted_copy_users.insert(*it);
#endif
      added_copy_users.erase(it);
      if (added_copy_users.empty())
        check_state_change(false/*adding*/);
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_valid_reference(void)
    //--------------------------------------------------------------------------
    {
      if (valid_references == 0)
        check_state_change(true/*adding*/);
      valid_references++;
    }

    //--------------------------------------------------------------------------
    void ReductionView::remove_valid_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_references > 0);
#endif
      valid_references--;
      to_be_invalidated = false;
      if (valid_references == 0)
        check_state_change(false/*adding*/);
    }

    //--------------------------------------------------------------------------
    bool ReductionView::has_added_users(void) const
    //--------------------------------------------------------------------------
    {
      return (!added_users.empty() || !added_copy_users.empty());
    }

    //--------------------------------------------------------------------------
    bool ReductionView::is_valid_view(void) const
    //--------------------------------------------------------------------------
    {
      return (!to_be_invalidated && (valid_references > 0));
    }

    //--------------------------------------------------------------------------
    size_t ReductionView::compute_send_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;  
      result += sizeof(manager->unique_id);
      result += sizeof(logical_region->handle);
      result += 2*sizeof(size_t); // number of users and copy users
      result += ((users.size() + added_users.size()
#ifdef LEGION_SPY
                  + deleted_users.size()
#endif
                  ) * (sizeof(UniqueID) + sizeof(TaskUser)));
      result += ((copy_users.size() + added_copy_users.size()
#ifdef LEGION_SPY
                  + deleted_copy_users.size()
#endif
                  ) * (sizeof(Event) + sizeof(CopyUser)));
      return result;
    }

    //--------------------------------------------------------------------------
    void ReductionView::pack_view_send(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->unique_id);
      rez.serialize(logical_region->handle);
      rez.serialize<size_t>(users.size() 
                            + added_users.size()
#ifdef LEGION_SPY
                            + deleted_users.size()
#endif
                            );
      for (std::map<UniqueID,TaskUser>::const_iterator it = users.begin();
            it != users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      for (std::map<UniqueID,TaskUser>::const_iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#ifdef LEGION_SPY
      for (std::map<UniqueID,TaskUser>::const_iterator it = deleted_users.begin();
            it != deleted_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#endif
      rez.serialize<size_t>(copy_users.size() 
                            + added_copy_users.size()
#ifdef LEGION_SPY
                            + deleted_copy_users.size()
#endif
                            );
      for (std::map<Event,CopyUser>::const_iterator it = copy_users.begin();
            it != copy_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      for (std::map<Event,CopyUser>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#ifdef LEGION_SPY
      for (std::map<Event,CopyUser>::const_iterator it = deleted_copy_users.begin();
            it != deleted_copy_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::unpack_view_send(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      LogicalRegion handle;
      derez.deserialize(handle);
      ReductionManager *manager = context->find_reduction_manager(mid);
      RegionNode *reg_node = context->get_node(handle);
      ReductionView *result = context->create_reduction_view(manager, reg_node, false/*made local*/);
      size_t num_users;
      derez.deserialize(num_users);
      for (unsigned idx = 0; idx < num_users; idx++)
      {
        UniqueID uid;
        derez.deserialize(uid);
        TaskUser user;
        derez.deserialize(user);
        result->users[uid] = user;
      }
      size_t num_copiers;
      derez.deserialize(num_copiers);
      for (unsigned idx = 0; idx < num_copiers; idx++)
      {
        Event event;
        derez.deserialize(event);
        CopyUser user;
        derez.deserialize(user);
        result->copy_users[event] = user;
      }
    }

    //--------------------------------------------------------------------------
    size_t ReductionView::compute_return_size(std::map<EscapedUser,unsigned> &escaped_users,
                                              std::set<EscapedCopy> &escaped_copies) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!manager->is_clone());
#endif
      size_t result = 0;
      result += sizeof(manager->unique_id);
      result += sizeof(logical_region->handle);
      // Need to send back all the added users, and record them as escaped
      result += sizeof(size_t); // number of added users
      result += (added_users.size() * (sizeof(UniqueID) + sizeof(TaskUser)));
      result += sizeof(size_t); // number of added copy users
      result += (added_copy_users.size() * (sizeof(Event) + sizeof(CopyUser)));
#ifdef LEGION_SPY
      result += sizeof(size_t); // number of deleted users
      result += (deleted_users.size() * (sizeof(UniqueID) + sizeof(TaskUser)));
      result += sizeof(size_t); // number of deleted users
      result += (deleted_copy_users.size() * (sizeof(Event) + sizeof(CopyUser)));
#endif
      // Record the escapees
      for (std::map<UniqueID,TaskUser>::const_iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        escaped_users[EscapedUser(get_key(), it->first)] = it->second.references;
      }
      for (std::map<Event,CopyUser>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        escaped_copies.insert(EscapedCopy(get_key(), it->first));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ReductionView::pack_view_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->unique_id);
      rez.serialize(logical_region->handle);
      rez.serialize(added_users.size());
      for (std::map<UniqueID,TaskUser>::const_iterator it = added_users.begin();
            it != added_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(added_copy_users.size());
      for (std::map<Event,CopyUser>::const_iterator it = added_copy_users.begin();
            it != added_copy_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#ifdef LEGION_SPY
      rez.serialize(deleted_users.size());
      for (std::map<UniqueID,TaskUser>::const_iterator it = deleted_users.begin();
            it != deleted_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(deleted_copy_users.size());
      for (std::map<Event,CopyUser>::const_iterator it = deleted_copy_users.begin();
            it != deleted_copy_users.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#endif
      // Move everything over to the users and copy users
      users.insert(added_users.begin(), added_users.end());
      added_users.clear();
      copy_users.insert(added_copy_users.begin(), added_copy_users.end());
      added_copy_users.clear();
#ifdef LEGION_SPY
      users.insert(deleted_users.begin(), deleted_users.end());
      deleted_users.clear();
      copy_users.insert(deleted_copy_users.begin(), deleted_copy_users.end());
      deleted_copy_users.clear();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::unpack_view_return(RegionTreeForest *context, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UniqueManagerID mid;
      derez.deserialize(mid);
      LogicalRegion handle;
      derez.deserialize(handle);
      ReductionView *result = NULL;
      InstanceKey key(mid, handle);
      if (context->has_reduction_view(key))
      {
        result = context->find_reduction_view(key);
      }
      else
      {
        ReductionManager *manager = context->find_reduction_manager(mid);
        RegionNode *node = context->get_node(handle);
        result = context->create_reduction_view(manager, node, true/*made local*/);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      size_t num_added_users;
      derez.deserialize(num_added_users);
      if (result->added_users.empty() && (num_added_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < num_added_users; idx++)
      {
        UniqueID uid;
        derez.deserialize(uid);
        TaskUser user;
        derez.deserialize(user);
        std::map<UniqueID,TaskUser>::iterator finder = result->added_users.find(uid);
        if (finder == result->added_users.end())
          result->added_users[uid] = user;
        else
        {
          finder->second.references += user.references;
          finder->second.use_multi = true;
        }
      }
      size_t num_added_copy_users;
      derez.deserialize(num_added_copy_users);
      if (result->added_copy_users.empty() && (num_added_copy_users > 0))
        result->check_state_change(true/*adding*/);
      for (unsigned idx = 0; idx < num_added_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        CopyUser user;
        derez.deserialize(user);
#ifdef DEBUG_HIGH_LEVEL
        assert(result->added_copy_users.find(copy_event) == result->added_copy_users.end());
#endif
        result->added_copy_users[copy_event] = user;
      }
#ifdef LEGION_SPY
      size_t num_deleted_users;
      derez.deserialize(num_deleted_users);
      for (unsigned idx = 0; idx < num_deleted_users; idx++)
      {
        UniqueID uid;
        derez.deserialize(uid);
        TaskUser user;
        derez.deserialize(user);
        // Doesn't matter if we overwrite it since it was deleted
        // which means that there were no valid references
        result->deleted_users[uid] = user;
      }
      size_t num_deleted_copy_users;
      derez.deserialize(num_deleted_copy_users);
      for (unsigned idx = 0; idx < num_deleted_copy_users; idx++)
      {
        Event copy_event;
        derez.deserialize(copy_event);
        CopyUser user;
        derez.deserialize(user);
        result->deleted_copy_users[copy_event] = user;
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool ReductionView::reduce_to(ReductionOpID redop, const FieldMask &reduce_mask, std::set<Event> &preconditions,
                                  std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop); // better be the same reduction operation
#endif
      // Since we're writing, we only need to wait for any copy-readers to avoid WAR
      // dependences, otherwise everything else can happen in parallel
      find_read_preconditions(copy_users, reduce_mask, preconditions);
      find_read_preconditions(added_copy_users, reduce_mask, preconditions);
#ifdef LEGION_SPY
      find_read_preconditions(deleted_copy_users, reduce_mask, preconditions);
#endif
      // Then update the destination fields for this copy
      manager->find_field_offsets(reduce_mask, dst_fields);
      return true; // reduction fold
    }

    //--------------------------------------------------------------------------
    void ReductionView::reduce_from(ReductionOpID redop, const FieldMask &reduce_mask, std::set<Event> &preconditions,
                                    std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop); // better be the same reduction operation
#endif
      // Since we're reading, we need to get preconditions from all task that are
      // reducing and all copy writers
      find_preconditions(users, reduce_mask, preconditions);
      find_preconditions(added_users, reduce_mask, preconditions);
      find_write_preconditions(copy_users, reduce_mask, preconditions);
      find_write_preconditions(added_copy_users, reduce_mask, preconditions);
#ifdef LEGION_SPY
      find_preconditions(deleted_users, reduce_mask, preconditions);
      find_write_preconditions(deleted_copy_users, reduce_mask, preconditions);
#endif
      // Then update the source fields for this copy
      manager->find_field_offsets(reduce_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::perform_reduction(PhysicalView *dst, const FieldMask &reduce_mask, RegionMapper &rm)
    //--------------------------------------------------------------------------
    {
      std::set<Event> preconditions;
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      bool fold = dst->reduce_to(manager->redop, reduce_mask, preconditions, dst_fields);
      this->reduce_from(manager->redop, reduce_mask, preconditions, src_fields);
      // Issue the reduction operation
      Event reduce_pre = Event::merge_events(preconditions);
#ifdef LEGION_SPY
      if (!reduce_pre.exists())
      {
        UserEvent new_reduce_pre = UserEvent::create_user_event();
        new_reduce_pre.trigger();
        reduce_pre = new_reduce_pre;
      }
      LegionSpy::log_event_dependences(preconditions, reduce_pre);
#endif
      Event reduce_post = manager->issue_reduction(src_fields, dst_fields, 
                      logical_region->row_source->domain, reduce_pre, fold);
#ifdef LEGION_SPY
      if (!reduce_post.exists())
      {
        UserEvent new_reduce_post = UserEvent::create_user_event();
        new_reduce_post.trigger();
        reduce_post = new_reduce_post;
      }
      // Log the reduction operation
      {
        FieldSpaceNode *field_node = context->get_node(logical_region->handle.field_space);
        char *string_mask = field_node->to_string(reduce_mask);
        LegionSpy::log_reduction_operation(this->manager->get_unique_id(), dst->get_manager()->get_unique_id(),
                                           logical_region->handle.index_space.id, logical_region->handle.field_space.id,
                                           logical_region->handle.tree_id, reduce_pre, reduce_post, manager->redop, string_mask);
        free(string_mask);
      }
#endif
      // Add reduction references if necessary
      if (reduce_post.exists())
      {
        rm.source_copy_instances.push_back(dst->add_copy_user(manager->redop, reduce_post, reduce_mask, false/*reading*/));
        rm.source_copy_instances.push_back(this->add_copy_user(manager->redop, reduce_post, reduce_mask, true/*reading*/));
      }
    }

    //--------------------------------------------------------------------------
    Event ReductionView::get_valid_event(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      std::set<Event> preconditions;
      // Get the event for when all the tasks and writers are done
      find_preconditions(users, mask, preconditions);
      find_preconditions(added_users, mask, preconditions);
      find_write_preconditions(copy_users, mask, preconditions);
      find_write_preconditions(added_copy_users, mask, preconditions);
#ifdef LEGION_SPY
      find_preconditions(deleted_users, mask, preconditions);
      find_write_preconditions(deleted_copy_users, mask, preconditions);
#endif
      Event result = Event::merge_events(preconditions);
#ifdef LEGION_SPY
      if (!result.exists());
      {
        UserEvent new_result = UserEvent::create_user_event();
        new_result.trigger();
        result = new_result;
      }
      LegionSpy::log_event_dependences(preconditions, result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void ReductionView::mark_to_be_invalidated(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid_references > 0);
#endif
      to_be_invalidated = true;
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_preconditions(const std::map<UniqueID,TaskUser> &current_users,
                                           const FieldMask &reduce_mask, std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      for (std::map<UniqueID,TaskUser>::const_iterator it = current_users.begin();
            it != current_users.end(); it++)
      {
        if (!(reduce_mask * it->second.mask))
        {
          if (it->second.use_multi)
            preconditions.insert(it->second.user.multi_term);
          else
            preconditions.insert(it->second.user.single_term);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_read_preconditions(const std::map<Event,CopyUser> &current_users,
                                                const FieldMask &reduce_mask, std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      for (std::map<Event,CopyUser>::const_iterator it = current_users.begin();
            it != current_users.end(); it++)
      {
        if ((it->second.reading) && (!(reduce_mask * it->second.mask)))
        {
          preconditions.insert(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_write_preconditions(const std::map<Event,CopyUser> &current_users,
                                                 const FieldMask &reduce_mask, std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      for (std::map<Event,CopyUser>::const_iterator it = current_users.begin();
            it != current_users.end(); it++)
      {
        if ((!it->second.reading) && (!(reduce_mask * it->second.mask)))
        {
          preconditions.insert(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::check_state_change(bool adding)
    //--------------------------------------------------------------------------
    {
      if (!manager->is_remote())
      {
        if ((valid_references == 0) && users.empty() && added_users.empty() &&
            copy_users.empty() && added_copy_users.empty())
        {
          if (adding)
            manager->add_reference();
          else
            manager->remove_reference();
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Instance Ref 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(void)
      : ready_event(Event::NO_EVENT), required_lock(Lock::NO_LOCK),
        location(Memory::NO_MEMORY), instance(PhysicalInstance::NO_INST),
        copy(false), is_reduction(false), view(NULL), manager(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(Event ready, Memory loc, PhysicalInstance inst,
                             PhysicalView *v, bool c /*= false*/, Lock lock /*= Lock::NO_LOCK*/)
      : ready_event(ready), required_lock(lock), location(loc), 
        instance(inst), copy(c), view(v), handle(LogicalRegion::NO_REGION)
    //--------------------------------------------------------------------------
    {
      if (view == NULL)
      {
        is_reduction = false;
        manager = NULL;
      }
      else
      {
        is_reduction = view->is_reduction_view();
        manager = view->get_manager();
      }
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(PhysicalManager *m, LogicalRegion h, Event ready, Memory loc,
                             PhysicalInstance inst, bool c /*= false*/, Lock lock /*= Lock::NO_LOCK*/)
      : ready_event(ready), required_lock(lock), location(loc), instance(inst),
        copy(c), view(NULL), handle(h), manager(m)
    //--------------------------------------------------------------------------
    {
      if (manager == NULL)   
        is_reduction = false;
      else
        is_reduction = manager->is_reduction_manager();
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_reference(UniqueID uid, bool strict)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
#endif
      // Remove the reference and set the view to NULL so
      // we can't accidentally remove the reference again
      if (copy)
        view->remove_copy(ready_event, strict);
      else
        view->remove_user(uid, 1/*single reference*/, strict);
      view = NULL;
    }

    /////////////////////////////////////////////////////////////
    // Generic User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GenericUser::GenericUser(const FieldMask &m, const RegionUsage &u)
      : field_mask(m), usage(u)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(GeneralizedOperation *o, unsigned id, const FieldMask &m, const RegionUsage &u)
      : GenericUser(m, u), op(o), idx(id), gen(o->get_gen())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool LogicalUser::operator==(const LogicalUser &rhs) const
    //--------------------------------------------------------------------------
    {
      if (field_mask != rhs.field_mask)
        return false;
      if (usage != rhs.usage)
        return false;
      if (op != rhs.op)
        return false;
      if (idx != rhs.idx)
        return false;
      if (gen != rhs.gen)
        return false;
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Physical User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const FieldMask &m, const RegionUsage &u, Event single, Event multi, unsigned id)
      : GenericUser(m, u), single_term(single), multi_term(multi), idx(id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PhysicalUser::operator==(const PhysicalUser &rhs) const
    //--------------------------------------------------------------------------
    {
      if (field_mask != rhs.field_mask)
        return false;
      if (usage != rhs.usage)
        return false;
      if (single_term != rhs.single_term)
        return false;
      if (multi_term != rhs.multi_term)
        return false;
      if (idx != rhs.idx)
        return false;
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Escaped User 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool EscapedUser::operator==(const EscapedUser &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((view_key == rhs.view_key) && (user == rhs.user));
    }

    //--------------------------------------------------------------------------
    bool EscapedUser::operator<(const EscapedUser &rhs) const
    //--------------------------------------------------------------------------
    {
      if (view_key < rhs.view_key)
        return true;
      else if (!(view_key == rhs.view_key)) // therefore greater than
        return false;
      else
        return (user < rhs.user);
    }

    /////////////////////////////////////////////////////////////
    // Escaped Copy 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool EscapedCopy::operator==(const EscapedCopy &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((view_key == rhs.view_key) && (copy_event == rhs.copy_event));
    }

    //--------------------------------------------------------------------------
    bool EscapedCopy::operator<(const EscapedCopy &rhs) const
    //--------------------------------------------------------------------------
    {
      if (view_key < rhs.view_key)
        return true;
      else if (!(view_key == rhs.view_key)) // therefore greater than
        return false;
      else
        return (copy_event < rhs.copy_event);
    }

    /////////////////////////////////////////////////////////////
    // Logical Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(const LogicalUser &u, ContextID c, std::list<LogicalUser> &users, bool closing_part)
      : user(u), ctx(c), epoch_users(users), closing_partition(closing_part)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::pre_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::post_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    bool LogicalCloser::closing_state(const RegionTreeNode::FieldState &state)
    //--------------------------------------------------------------------------
    {
      // Always continue with the closing
      return true;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      node->close_logical_tree(*this, closing_mask);
    }

    //--------------------------------------------------------------------------
    bool LogicalCloser::leave_children_open(void) const
    //--------------------------------------------------------------------------
    {
      // Never leave anything open when doing a logical close
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Physical Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalUser &u, RegionMapper &r,
                                    RegionNode *ct, bool lo)
      : user(u), rm(r), close_target(ct), leave_open(lo), 
        targets_selected(false), success(true), partition_valid(false)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(close_target != NULL);
#endif
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalCloser &rhs, RegionNode *ct)
      : user(rhs.user), rm(rhs.rm), close_target(ct), leave_open(rhs.leave_open),
        targets_selected(rhs.targets_selected), success(true), partition_valid(false)
    //--------------------------------------------------------------------------
    {
      if (targets_selected)
      {
        upper_targets = rhs.lower_targets;
        for (std::vector<InstanceView*>::const_iterator it = upper_targets.begin();
              it != upper_targets.end(); it++)
        {
          (*it)->add_valid_reference();
        }
      }
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::~PhysicalCloser(void)
    //--------------------------------------------------------------------------
    {
      // Remove valid references from any physical targets
      for (std::vector<InstanceView*>::const_iterator it = upper_targets.begin();
            it != upper_targets.end(); it++)
      {
        (*it)->remove_valid_reference();
      }
    }
    
    //--------------------------------------------------------------------------
    void PhysicalCloser::pre_siphon(void)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::post_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have any dirty instance views to update
      // target region node with
      if (!upper_targets.empty())
      {
        close_target->update_valid_views(rm.ctx, user.field_mask, dirty_mask, upper_targets);    
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalCloser::closing_state(const RegionTreeNode::FieldState &state)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to select our targets
      if (!targets_selected && ((state.open_state == RegionTreeNode::OPEN_READ_WRITE) ||
                                (state.open_state == RegionTreeNode::OPEN_SINGLE_REDUCE) ||
                                (state.open_state == RegionTreeNode::OPEN_MULTI_REDUCE)))
      {
        // We're going to need to issue a close so make some targets
         
        // First get the list of valid instances
        std::list<std::pair<InstanceView*,FieldMask> > valid_views;
        close_target->find_valid_instance_views(rm.ctx, valid_views, user.field_mask, 
                                                user.field_mask, true/*needs space*/);
        // Get the set of memories for which we have valid instances
        std::set<Memory> valid_memories;
        for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it =
              valid_views.begin(); it != valid_views.end(); it++)
        {
          valid_memories.insert(it->first->get_location());
        }
        // Now ask the mapper what it wants to do
        bool create_one = true;
        std::set<Memory>    to_reuse;
        std::vector<Memory> to_create;
        {
          DetailedTimer::ScopedPush sp(TIME_MAPPER);
          AutoLock m_lock(rm.mapper_lock);
          rm.mapper->rank_copy_targets(rm.task, rm.target, rm.tag, rm.inline_mapping, rm.req, 
                                        rm.idx, valid_memories, to_reuse, to_create, create_one);
        }
        if (to_reuse.empty() && to_create.empty())
        {
          log_region(LEVEL_ERROR,"Invalid mapper output for rank copy targets.  Must specify at least one target memory.");
#ifdef DEBUG_HIGH_LEVEl
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
        // Now process the results
        // First see if we should re-use any instances
        for (std::set<Memory>::const_iterator mit = to_reuse.begin();
              mit != to_reuse.end(); mit++)
        {
          // Make sure it was a valid choice 
          if (valid_memories.find(*mit) == valid_memories.end())
            continue;
          InstanceView *best = NULL;
          FieldMask best_mask;
          unsigned num_valid_fields = 0;
          for (std::list<std::pair<InstanceView*,FieldMask> >::const_iterator it = valid_views.begin();
                it != valid_views.end(); it++)
          {
            if (it->first->get_location() != (*mit))
              continue;
            unsigned valid_fields = FieldMask::pop_count(it->second);
            if (valid_fields > num_valid_fields)
            {
              num_valid_fields = valid_fields;
              best = it->first;
              best_mask = it->second;
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(best != NULL);
#endif
          // Update any of the fields that are different from the current valid mask
          FieldMask need_update = user.field_mask - best_mask;
          if (!!need_update)
            close_target->issue_update_copies(best, rm, need_update);
          add_upper_target(best);
        }
        // Now see if we want to try to create any new instances
        for (std::vector<Memory>::const_iterator it = to_create.begin();
              it != to_create.end(); it++)
        {
          // Try making an instance in the memory 
          InstanceView *new_view = close_target->create_instance(*it, rm, rm.req.instance_fields);
          if (new_view != NULL)
          {
            // Update all the fields
            close_target->issue_update_copies(new_view, rm, user.field_mask);
            add_upper_target(new_view);
            // If we were only supposed to make one, then we're done
            if (create_one)
              break;
          }
        }
        targets_selected = true;
        // See if we succeeded in making a target instance
        if (upper_targets.empty())
        {
          // We failed, have to try again later
          success = false;
          return false;
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lower_targets.empty());
#endif
      // Convert the upper InstanceViews to the lower instance views
      node->close_physical_tree(*this, closing_mask); 
    }

    //--------------------------------------------------------------------------
    bool PhysicalCloser::leave_children_open(void) const
    //--------------------------------------------------------------------------
    {
      return leave_open;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::pre_region(Color region_color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lower_targets.empty());
      assert(partition_valid);
#endif
      for (std::vector<InstanceView*>::const_iterator it = upper_targets.begin();
            it != upper_targets.end(); it++)
      {
        lower_targets.push_back((*it)->get_subview(partition_color,region_color));
#ifdef DEBUG_HIGH_LEVEL
        assert(lower_targets.back() != NULL);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::post_region(void)
    //--------------------------------------------------------------------------
    {
      lower_targets.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::pre_partition(Color pc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!partition_valid);
#endif
      partition_color = pc;
      partition_valid = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::post_partition(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partition_valid);
#endif
      partition_valid = false;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::add_upper_target(InstanceView *target)
    //--------------------------------------------------------------------------
    {
      targets_selected = true;
      target->add_valid_reference();
      upper_targets.push_back(target);
    }

    /////////////////////////////////////////////////////////////
    // Reduction Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionCloser::ReductionCloser(const PhysicalUser &u, RegionMapper &r, RegionNode *close, ReductionView *t)
      : user(u), rm(r), close_target(close), target(t), success(true)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(target != NULL);
#endif
      // Add a valid reference to the target
      target->add_valid_reference();
    }

    //--------------------------------------------------------------------------
    ReductionCloser::~ReductionCloser(void)
    //--------------------------------------------------------------------------
    {
      // Remove the valid reference from the target
      target->remove_valid_reference();
    }

    //--------------------------------------------------------------------------
    void ReductionCloser::pre_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void ReductionCloser::post_siphon(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    bool ReductionCloser::closing_state(const RegionTreeNode::FieldState &state)
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    void ReductionCloser::close_tree_node(RegionTreeNode *node, const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      node->close_reduction_tree(*this, closing_mask);
    }

    //--------------------------------------------------------------------------
    bool ReductionCloser::leave_children_open(void) const
    //--------------------------------------------------------------------------
    {
      // Never leave children open when doing a reduction close
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Region Analyzer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionAnalyzer::RegionAnalyzer(ContextID ctx_id, GeneralizedOperation *o, unsigned id, const RegionRequirement &req)
      : ctx(ctx_id), op(o), idx(id), start(req.parent), usage(RegionUsage(req)) 
    //--------------------------------------------------------------------------
    {
      // Copy the fields from the region requirement
      fields.resize(req.privilege_fields.size());
      unsigned i = 0;
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        fields[i++] = *it;
      }
    }

    /////////////////////////////////////////////////////////////
    // Region Analyzer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionMapper::RegionMapper(Task *t, UniqueID u, ContextID c, unsigned id, const RegionRequirement &r, Mapper *m,
#ifdef LOW_LEVEL_LOCKS
                                Lock m_lock,
#else
                                ImmovableLock m_lock,
#endif
                                Processor tar, Event single, Event multi, MappingTagID tg, bool sanit,
                                bool in_map, std::vector<InstanceRef> &source_copy)
      : ctx(c), sanitizing(sanit), inline_mapping(in_map), success(false), idx(id), req(r), task(t), uid(u),
        mapper_lock(m_lock), mapper(m), tag(tg), target(tar), single_term(single), multi_term(multi),
        source_copy_instances(source_copy), result(InstanceRef())
    //--------------------------------------------------------------------------
    {
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

