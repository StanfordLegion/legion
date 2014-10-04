/* Copyright 2014 Stanford University
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
#include "runtime.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "region_tree.h"
#include "legion_spy.h"
#include "legion_logging.h"
#include "legion_profiling.h"
#include "interval_tree.h"
#include "rectangle_set.h"

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

    /////////////////////////////////////////////////////////////
    // Region Tree Forest 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeForest::RegionTreeForest(Runtime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
      this->forest_lock = Reservation::create_reservation();
      this->lookup_lock = Reservation::create_reservation();
      this->distributed_lock = Reservation::create_reservation();
#ifdef DYNAMIC_TESTS
      this->dynamic_lock = Reservation::create_reservation();
#endif
#ifdef DEBUG_PERF
      this->perf_trace_lock = Reservation::create_reservation();
      int max_local_id = 1;
      int local_space = 
        Machine::get_executing_processor().address_space();
      const std::set<Processor> &procs = runtime->machine->get_all_processors();
      for (std::set<Processor>::const_iterator it = procs.begin();
            it != procs.end(); it++)
      {
        if (local_space == it->address_space())
        {
          int local = it->local_id();
          if (local > max_local_id)
            max_local_id = local;
        }
      }
      // Reserve enough space for traces for each processor
      traces.resize(max_local_id+1);
#endif
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::RegionTreeForest(const RegionTreeForest &rhs)
      : runtime(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::~RegionTreeForest(void)
    //--------------------------------------------------------------------------
    {
      forest_lock.destroy_reservation();
      forest_lock = Reservation::NO_RESERVATION;
      lookup_lock.destroy_reservation();
      lookup_lock = Reservation::NO_RESERVATION;
      distributed_lock.destroy_reservation();
      distributed_lock = Reservation::NO_RESERVATION;
#ifdef DYNAMIC_TESTS
      dynamic_lock.destroy_reservation();
      dynamic_lock = Reservation::NO_RESERVATION;
#endif
#ifdef DEBUG_PERF
      perf_trace_lock.destroy_reservation();
      perf_trace_lock = Reservation::NO_RESERVATION;
#endif
    }

    //--------------------------------------------------------------------------
    RegionTreeForest& RegionTreeForest::operator=(const RegionTreeForest &rhs)
    //--------------------------------------------------------------------------
    {
      // should never happen
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_space(const Domain &domain) 
    //--------------------------------------------------------------------------
    {
      create_node(domain, NULL/*parent*/, 0/*color*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_space(const Domain &hull,
                                              const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      // Note that it is safe that we do this in two passes
      // because we haven't given back the handle yet for
      // the index space so no one actually knows it exists yet.
      IndexSpaceNode *node = create_node(hull, NULL/*parent*/, 0/*color*/);
      node->update_component_domains(domains);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_partition(IndexPartition pid,
        IndexSpace parent, bool disjoint, 
        int color, const std::map<Color,Domain> &coloring, Domain color_space)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *parent_node = get_node(parent);
      Color part_color;
      if (color < 0)
        part_color = parent_node->generate_color();
      else
        part_color = unsigned(color);
      IndexPartNode *new_part = create_node(pid, parent_node, part_color,
                                    color_space, disjoint);
#ifdef LEGION_SPY
      LegionSpy::log_index_partition(parent.id, pid, disjoint, part_color);
#endif
#ifdef DYNAMIC_TESTS
      std::vector<IndexSpaceNode*> children; 
#endif
      // Now do all the child nodes
      for (std::map<Color,Domain>::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        if (it->first == UINT_MAX)
        {
          log_index(LEVEL_ERROR,"Invalid child color UINT_MAX specified "
                                "for create index partition.  All colors "
                                "must be between 0 and UINT_MAX-1");
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_PARTITION_COLOR);
        }
        Domain domain = it->second;
        domain.get_index_space(true/*create if necessary*/);
#ifdef DYNAMIC_TESTS
        IndexSpaceNode *child = 
#endif
        create_node(domain, new_part, it->first);
#ifdef DYNAMIC_TESTS
        children.push_back(child);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_index_subspace(pid, 
            domain.get_index_space().id, it->first);
#endif
      } 
#ifdef DYNAMIC_TESTS
      if (Runtime::dynamic_independence_tests)
      {
        parent_node->add_disjointness_tests(new_part, children); 
        AutoLock d_lock(dynamic_lock);
        if (!disjoint && (children.size() > 1))
        {
          for (std::vector<IndexSpaceNode*>::const_iterator it1 = 
              children.begin(); it1 != children.end(); it1++)
          {
            for (std::vector<IndexSpaceNode*>::const_iterator it2 = 
                  children.begin(); it2 != it1; it2++)
            {
              dynamic_space_tests.push_back(
                  DynamicSpaceTest(new_part, *it1, *it2));
            }
          }
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_index_partition(IndexPartition pid,
        IndexSpace parent, bool disjoint, int color,
        const std::map<Color,Domain> &convex_hulls, Domain color_space,
        const std::map<Color,std::set<Domain> > &component_domains)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *parent_node = get_node(parent);
      Color part_color;
      if (color < 0)
        part_color = parent_node->generate_color();
      else
        part_color = unsigned(color);
      IndexPartNode *new_part = create_node(pid, parent_node, part_color,
                                    color_space, disjoint);
#ifdef LEGION_SPY
      LegionSpy::log_index_partition(parent.id, pid, disjoint, part_color);
#endif
#ifdef DYNAMIC_TESTS
      std::vector<IndexSpaceNode*> children; 
#endif
      // Now do all the child nodes
      std::map<Color,std::set<Domain> >::const_iterator comp_it = 
        component_domains.begin();
      for (std::map<Color,Domain>::const_iterator it = convex_hulls.begin();
            it != convex_hulls.end(); it++, comp_it++)
      {
        if (it->first == UINT_MAX)
        {
          log_index(LEVEL_ERROR,"Invalid child color UINT_MAX specified "
                                "for create index partition.  All colors "
                                "must be between 0 and UINT_MAX-1");
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_PARTITION_COLOR);
        }
        Domain hull = it->second;
        hull.get_index_space(true/*create if necessary*/);
        IndexSpaceNode *child = create_node(hull, new_part, it->first);
        child->update_component_domains(comp_it->second);
#ifdef DYNAMIC_TESTS
        children.push_back(child);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_index_subspace(pid, 
            hull.get_index_space().id, it->first);
#endif
      }
#ifdef DYNAMIC_TESTS
      if (Runtime::dynamic_independence_tests)
      {
        parent_node->add_disjointness_tests(new_part, children); 
        AutoLock d_lock(dynamic_lock);
        if (!disjoint && (children.size() > 1))
        {
          for (std::vector<IndexSpaceNode*>::const_iterator it1 = 
              children.begin(); it1 != children.end(); it1++)
          {
            for (std::vector<IndexSpaceNode*>::const_iterator it2 = 
                  children.begin(); it2 != it1; it2++)
            {
              dynamic_space_tests.push_back(
                  DynamicSpaceTest(new_part, *it1, *it2));
            }
          }
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::destroy_index_space(IndexSpace handle,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      node->destroy_node(source);
      // Return true if this is a top level node
      return (node->parent == NULL);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_index_partition(IndexPartition handle,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      node->destroy_node(source);
    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_index_partition(IndexSpace parent,
                                                         Color color)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *parent_node = get_node(parent);
      IndexPartNode *child_node = parent_node->get_child(color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_index_subspace(IndexPartition parent,
                                                    Color color)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *parent_node = get_node(parent);
      IndexSpaceNode *child_node = parent_node->get_child(color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_multiple_domains(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      return node->has_component_domains();
    }

    //--------------------------------------------------------------------------
    Domain RegionTreeForest::get_index_space_domain(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      return node->domain;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_index_space_domains(IndexSpace handle,
                                                   std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      if (node->has_component_domains())
      {
        const std::set<Domain> &comp_domains = node->get_component_domains();
        domains.insert(domains.end(), comp_domains.begin(), comp_domains.end());
      }
      else
        domains.push_back(node->domain);
    }

    //--------------------------------------------------------------------------
    Domain RegionTreeForest::get_index_partition_color_space(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(p);
      return node->color_space;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_index_space_partition_colors(IndexSpace sp,
                                                        std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(sp);
      node->get_colors(colors); 
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_index_partition_disjoint(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(p);
      return node->disjoint;
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::get_index_space_color(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      return node->color;
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::get_index_partition_color(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      return node->color;
    }

    //--------------------------------------------------------------------------
    IndexSpace RegionTreeForest::get_parent_index_space(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      return (node->parent != NULL);
    }

    //--------------------------------------------------------------------------
    IndexPartition RegionTreeForest::get_parent_index_partition(
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      if (node->parent == NULL)
      {
        log_run(LEVEL_ERROR,"Parent index partition requested for "
                            "index space " IDFMT " with no parent. Use "
                            "has_parent_index_partition to check "
                            "before requesting a parent.", handle.id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PARENT_REQUEST);
      }
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    IndexSpaceAllocator* RegionTreeForest::get_index_space_allocator(
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      return node->get_allocator();
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_domain_volume(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      if (node->domain.get_dim() == 0)
      {
        const LowLevel::ElementMask &mask = 
          node->domain.get_index_space().get_valid_mask();
        return mask.get_num_elmts();
      }
      else
        return node->domain.get_volume();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_field_space(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      create_node(handle);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_field_space(FieldSpace handle,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->destroy_node(source);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::allocate_field(FieldSpace handle, size_t field_size,
                                          FieldID fid, bool local)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      if (local && node->has_field(fid))
        return true;
      node->allocate_field(fid, field_size, local);
      return false;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_field(FieldSpace handle, FieldID fid, 
                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->free_field(fid, source);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::allocate_fields(FieldSpace handle, 
                                           const std::vector<size_t> &sizes,
                                           const std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(sizes.size() == fields.size());
#endif
      // We know that none of these field allocations are local
      FieldSpaceNode *node = get_node(handle);
      for (unsigned idx = 0; idx < fields.size(); idx++)
      {
        node->allocate_field(fields[idx], sizes[idx], false/*local*/);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::free_fields(FieldSpace handle,
                                       const std::set<FieldID> &to_free,
                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      for (std::set<FieldID>::const_iterator it = to_free.begin();
            it != to_free.end(); it++)
      {
        node->free_field(*it, source);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::allocate_field_index(FieldSpace handle, 
                                                size_t field_size, FieldID fid,
                                                unsigned index, 
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->allocate_field_index(fid, field_size, source, index);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::allocate_field_indexes(FieldSpace handle,
                                        const std::vector<FieldID> &fields,
                                        const std::vector<size_t> &sizes,
                                        const std::vector<unsigned> &indexes,
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.size() == sizes.size());
      assert(fields.size() == indexes.size());
#endif
      FieldSpaceNode *node = get_node(handle);
      for (unsigned idx = 0; idx < fields.size(); idx++)
      {
        unsigned index = indexes[idx];
        node->allocate_field_index(fields[idx], sizes[idx], source, index);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_all_fields(FieldSpace handle, 
                                          std::set<FieldID> &to_set)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->get_all_fields(to_set);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::get_all_regions(FieldSpace handle,
                                           std::set<LogicalRegion> &regions)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->get_all_regions(regions);
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_field_size(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      if (!node->has_field(fid))
      {
        log_run(LEVEL_ERROR,"FieldSpace %x has no field %d", handle.id, fid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_FIELD_ID);
      }
      return node->get_field_size(fid);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::create_logical_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // We do need the autolock here since we're going to be making
      // region tree nodes
      AutoLock f_lock(forest_lock,1,false/*exclusive*/);
      RegionNode *result = create_node(handle, NULL/*parent*/);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::destroy_logical_region(LogicalRegion handle,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      node->destroy_node(source);
      // Return true if this was a top-level region
      return (node->parent == NULL);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::destroy_logical_partition(LogicalPartition handle,
                                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PartitionNode *node = get_node(handle);
      node->destroy_node(source);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_logical_partition(
                                    LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalPartition(parent.tree_id, handle, parent.field_space);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_logical_partition_by_color(
                                                LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(forest_lock,1,false/*exclusive*/);
      RegionNode *parent_node = get_node(parent);
      IndexPartNode *index_node = parent_node->row_source->get_child(c);
      LogicalPartition result(parent.tree_id, index_node->handle, 
                              parent.field_space);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_logical_partition_by_tree(
                      IndexPartition handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalPartition(tid, handle, space);
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_logical_subregion(
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalRegion(parent.tree_id, handle, parent.field_space);
    }
    
    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_logical_subregion_by_color(
                                              LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(forest_lock,1,false/*exclusive*/);
      PartitionNode *parent_node = get_node(parent);
      IndexSpaceNode *index_node = parent_node->row_source->get_child(c);
      LogicalRegion result(parent.tree_id, index_node->handle,
                           parent.field_space);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_logical_subregion_by_tree(
                          IndexSpace handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalRegion(tid, handle, space);
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::get_logical_region_color(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(forest_lock,1,false/*exclusive*/);
      RegionNode *node = get_node(handle);
      return node->row_source->color;
    }

    //--------------------------------------------------------------------------
    Color RegionTreeForest::get_logical_partition_color(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(forest_lock,1,false/*exclusive*/);
      PartitionNode *node = get_node(handle);
      return node->row_source->color;
    }

    //--------------------------------------------------------------------------
    LogicalRegion RegionTreeForest::get_parent_logical_region(
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      PartitionNode *node = get_node(handle);
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      return (node->parent != NULL);
    }

    //--------------------------------------------------------------------------
    LogicalPartition RegionTreeForest::get_parent_logical_partition(
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      if (node->parent == NULL)
      {
        log_run(LEVEL_ERROR,"Parent logical partition requested for "
                            "logical region (" IDFMT ",%x,%d) with no parent. "
                            "Use has_parent_logical_partition to check "
                            "before requesting a parent.", 
                            handle.index_space.id,
                            handle.field_space.id,
                            handle.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PARENT_REQUEST);
      }
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    size_t RegionTreeForest::get_domain_volume(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      Domain d = node->get_domain();
      if (d.get_dim() == 0)
      {
        const LowLevel::ElementMask &mask = 
          d.get_index_space().get_valid_mask();
        return mask.get_num_elmts();
      }
      else
        return d.get_volume();
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_dependence_analysis(RegionTreeContext ctx,
                                                  Operation *op, unsigned idx,
                                                  RegionRequirement &req,
                                                  RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      begin_perf_trace(REGION_DEPENDENCE_ANALYSIS);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx.exists());
#endif
      RegionNode *parent_node = get_node(req.parent);
      
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      // Then compute the logical user
      LogicalUser user(op, idx, RegionUsage(req), user_mask); 
      
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), true/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
      // Finally do the traversal, note that we don't need to hold the
      // context lock since the runtime guarantees that all dependence
      // analysis for a single context are performed in order
      parent_node->register_logical_node(ctx.get_id(), user, 
                                         path, op->already_traced());
      // Now check to see if we have any simultaneous restrictions
      // we need to check
      if (req.restricted)
      {
        RestrictedTraverser traverser(ctx.get_id(), path);
        traverser.traverse(parent_node);
        // Check to see if there was user-level software
        // coherence for all of our fields.
        FieldMask restricted_mask = user_mask - traverser.get_coherence_mask();
        // If none of our fields are still restricted
        // then we can remove the restricted field on
        // our region requirement.  Otherwise we keep
        // the restriction.
        if (!restricted_mask)
          req.restricted = false; 
      }
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &req, idx, op->get_logging_name(),
                                     op->get_unique_op_id(), parent_node,
                                     ctx.get_id(), false/*before*/, 
                                     false/*premap*/,
                                     false/*closing*/, true/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
#ifdef DEBUG_PERF
      end_perf_trace(Runtime::perf_trace_tolerance);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::perform_fence_analysis(RegionTreeContext ctx,
                                                  Operation *fence,
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Register dependences for this fence on all users in the tree
      RegionNode *top_node = get_node(handle);
      LogicalRegistrar<true> registrar(ctx.get_id(), fence, 
                                       FieldMask(FIELD_ALL_ONES));
      top_node->visit_node(&registrar);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_destroy_index_space(RegionTreeContext ctx,
                                                       IndexSpace handle,
                                                       Operation *op,
                                                       LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *delete_node = get_node(handle);
      // Because we lazily instantiate the region tree, we need to do
      // tree comparisons from the the source nodes at the top of the tree
      IndexSpaceNode *top_index = delete_node;
      while (top_index->parent != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(top_index->parent->parent != NULL);
#endif
        top_index = top_index->parent->parent;
      }
      if (top_index->has_instance(region.get_tree_id()))
      {
        RegionNode *start_node = get_node(region);
        RegionTreePath path;
        initialize_path(delete_node,start_node->row_source,path);
        LogicalPathRegistrar reg(ctx.get_id(), op, 
                                 FieldMask(FIELD_ALL_ONES), path);
        reg.traverse(start_node); 
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_destroy_index_partition(
                                                          RegionTreeContext ctx,
                                                          IndexPartition handle,
                                                          Operation *op,
                                                          LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *delete_node = get_node(handle);
      // Because we lazily instantiate the region tree, we need to do
      // tree comparisons from the the source nodes at the top of the tree
      IndexSpaceNode *top_index = delete_node->parent;
      while (top_index->parent != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(top_index->parent->parent != NULL);
#endif
        top_index = top_index->parent->parent;
      }
      if (top_index->has_instance(region.get_tree_id()))
      {
        RegionNode *start_node = get_node(region);
        RegionTreePath path;
        initialize_path(delete_node,start_node->row_source,path);
        LogicalPathRegistrar reg(ctx.get_id(), op, 
                                 FieldMask(FIELD_ALL_ONES), path);
        reg.traverse(start_node);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_destroy_field_space(RegionTreeContext ctx,
                                                       FieldSpace handle,
                                                       Operation *op,
                                                       LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *delete_node = get_node(handle);
      if (delete_node->has_instance(region.get_tree_id()))
      {
        RegionNode *start_node = get_node(region);
        LogicalRegistrar<false> registrar(ctx.get_id(), op, 
                                          FieldMask(FIELD_ALL_ONES));
        start_node->visit_node(&registrar);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_destroy_fields(RegionTreeContext ctx,
                                                  FieldSpace handle,
                                            const std::set<FieldID> &to_delete,
                                                  Operation *op,
                                                  LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *delete_node = get_node(handle);
      if (delete_node->has_instance(region.get_tree_id()))
      {
        RegionNode *start_node = get_node(region);
        LogicalRegistrar<false> registrar(ctx.get_id(), op, 
                                        delete_node->get_field_mask(to_delete));
        start_node->visit_node(&registrar);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_destroy_logical_region(RegionTreeContext ctx,
                                                          LogicalRegion handle,
                                                          Operation *op,
                                                          LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (handle.get_tree_id() == region.get_tree_id())
      {
        RegionNode *start_node = get_node(region);
        RegionNode *delete_node = get_node(handle);
        RegionTreePath path;
        initialize_path(delete_node->row_source,start_node->row_source,path);
        LogicalPathRegistrar reg(ctx.get_id(), op, 
                                 FieldMask(FIELD_ALL_ONES), path);
        reg.traverse(start_node);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::analyze_destroy_logical_partition(
                                                        RegionTreeContext ctx,
                                                        LogicalPartition handle,
                                                        Operation *op,
                                                        LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (handle.get_tree_id() == region.get_tree_id())
      {
        RegionNode *start_node = get_node(region);
        PartitionNode *delete_node = get_node(handle);
        RegionTreePath path;
        initialize_path(delete_node->row_source,start_node->row_source,path);
        LogicalPathRegistrar reg(ctx.get_id(), op, 
                                 FieldMask(FIELD_ALL_ONES), path);
        reg.traverse(start_node);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_logical_context(RegionTreeContext ctx,
                                                      LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // For now we don't need to do anything here assuming we 
      // always clean up after ourselves by calling invalidate
      // logical context after we are done with a context
      // In debug mode we'll do a check just to make sure this is true
#ifdef DEBUG_HIGH_LEVEL
      RegionNode *top_node = get_node(handle); 
      LogicalInitializer init(ctx.get_id());
      top_node->visit_node(&init);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_logical_context(RegionTreeContext ctx,
                                                      LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *top_node = get_node(handle);
      LogicalInvalidator invalidator(ctx.get_id());
      top_node->visit_node(&invalidator);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::acquire_user_coherence(RegionTreeContext ctx,
                                                  LogicalRegion handle,
                                                const std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      FieldMask user_mask = node->column_source->get_field_mask(fields);
      node->acquire_user_coherence(ctx.get_id(), user_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::release_user_coherence(RegionTreeContext ctx,
                                                  LogicalRegion handle,
                                                const std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      FieldMask user_mask = node->column_source->get_field_mask(fields);
      node->release_user_coherence(ctx.get_id(), user_mask);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::premap_physical_region(RegionTreeContext ctx,
                                                  RegionTreePath &path,
                                                  RegionRequirement &req,
                                                  Mappable *mappable,
                                                  SingleTask *parent_ctx,
                                                  Processor local_proc
#ifdef DEBUG_HIGH_LEVEL
                                                  , unsigned index
                                                  , const char *log_name
                                                  , UniqueID uid
#endif
                                                  )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      begin_perf_trace(PREMAP_PHYSICAL_REGION_ANALYSIS);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx.exists());
#endif
      RegionNode *parent_node = get_node(req.parent);
      // Don't need to initialize the path since that was done
      // in the logical traversal.
      // Construct a premap traversal object
      FieldMask user_mask = 
        parent_node->column_source->get_field_mask(req.privilege_fields);
      MappableInfo info(ctx.get_id(), mappable, local_proc, req, user_mask); 
      PremapTraverser traverser(path, &info);
      // Mark that we are beginning the premapping
      UserEvent premap_event = 
        parent_ctx->begin_premapping(req.parent.tree_id, user_mask);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     parent_node, ctx.get_id(), 
                                     true/*before*/, true/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
      const bool result = traverser.traverse(parent_node);
#ifdef DEBUG_HIGH_LEVEL
      if (result)
      {
        TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                       parent_node, ctx.get_id(), 
                                       false/*before*/, true/*premap*/, 
                                       false/*closing*/, false/*logical*/,
                                       FieldMask(FIELD_ALL_ONES), user_mask);
      }
#endif
      // Indicate that we are done premapping
      parent_ctx->end_premapping(req.parent.tree_id, premap_event);
      // If we are restricted, prune out any instances which do
      // not have fully valid data
      if (req.restricted)
      {
        std::vector<Memory> to_delete;
        for (std::map<Memory,bool>::const_iterator it = 
              req.current_instances.begin(); it != 
              req.current_instances.end(); it++)
        {
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<Memory>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
          req.current_instances.erase(*it);
      }
#ifdef DEBUG_PERF
      end_perf_trace(Runtime::perf_trace_tolerance);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    MappingRef RegionTreeForest::map_physical_region(RegionTreeContext ctx,
                                                     RegionTreePath &path,
                                                     RegionRequirement &req,
                                                     unsigned index,
                                                     Mappable *mappable,
                                                     Processor local_proc,
                                                     Processor target_proc
#ifdef DEBUG_HIGH_LEVEL
                                                     , const char *log_name
                                                     , UniqueID uid
#endif
                                                     )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      begin_perf_trace(MAP_PHYSICAL_REGION_ANALYSIS);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx.exists());
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *child_node = get_node(req.region);
      FieldMask user_mask = 
        child_node->column_source->get_field_mask(req.privilege_fields);
      // Construct the mappable info
      MappableInfo info(ctx.get_id(), mappable, local_proc, req, user_mask);
      // Get the start node
      RegionTreeNode *start_node = child_node;
      for (unsigned idx = 0; idx < (path.get_path_length()-1); idx++)
        start_node = start_node->get_parent();
      // Construct the traverser
      MappingTraverser traverser(path, &info, RegionUsage(req),
                                 user_mask, target_proc, index);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     start_node, ctx.get_id(), 
                                     true/*before*/, false/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
      bool result = traverser.traverse(start_node);
#ifdef DEBUG_PERF
      end_perf_trace(Runtime::perf_trace_tolerance);
#endif
      if (result)
        return traverser.get_instance_ref();
      else
        return MappingRef();
    }

    //--------------------------------------------------------------------------
    MappingRef RegionTreeForest::remap_physical_region(RegionTreeContext ctx,
                                                       RegionRequirement &req,
                                                       unsigned index,
                                                       const InstanceRef &ref
#ifdef DEBUG_HIGH_LEVEL
                                                       , const char *log_name
                                                       , UniqueID uid
#endif
                                                       )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      begin_perf_trace(REMAP_PHYSICAL_REGION_ANALYSIS);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx.exists());
      assert(req.handle_type == SINGULAR);
#endif
      // Reductions don't need any update fields
      if (IS_REDUCE(req))
      {
        return MappingRef(ref.get_handle().get_view(), FieldMask());
      }
      RegionNode *target_node = get_node(req.region);
      FieldMask user_mask = 
        target_node->column_source->get_field_mask(req.privilege_fields);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     target_node, ctx.get_id(), 
                                     true/*before*/, false/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
      MaterializedView *view = 
        ref.get_handle().get_view()->as_instance_view()->as_materialized_view();
      FieldMask needed_mask;
      target_node->remap_region(ctx.get_id(), view, user_mask, needed_mask);
#ifdef DEBUG_PERF
      end_perf_trace(Runtime::perf_trace_tolerance);
#endif
      return MappingRef(view, needed_mask);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::register_physical_region(
                                                        RegionTreeContext ctx,
                                                        const MappingRef &ref,
                                                        RegionRequirement &req,
                                                        unsigned index,
                                                        Mappable *mappable,
                                                        Processor local_proc,
                                                        Event term_event
#ifdef DEBUG_HIGH_LEVEL
                                                        , const char *log_name
                                                        , UniqueID uid
                                                        , RegionTreePath &path
#endif
                                                        ) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      begin_perf_trace(REGISTER_PHYSICAL_REGION_ANALYSIS);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx.exists());
      assert(req.handle_type == SINGULAR);
      assert(ref.has_ref());
#endif
      RegionNode *child_node = get_node(req.region);
      FieldMask user_mask = 
        child_node->column_source->get_field_mask(req.privilege_fields);
      // Construct the mappable info
      MappableInfo info(ctx.get_id(), mappable, local_proc, req, user_mask);
      // Construct the user
      PhysicalUser user(RegionUsage(req), user_mask, term_event);
      LogicalView *view = ref.get_view();
      InstanceRef result = child_node->register_region(&info, user, 
                                                       view, ref.get_mask());
#ifdef DEBUG_HIGH_LEVEL 
      RegionTreeNode *start_node = child_node;
      for (unsigned idx = 0; idx < (path.get_path_length()-1); idx++)
        start_node = start_node->get_parent();
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     start_node, ctx.get_id(), 
                                     false/*before*/, false/*premap*/, 
                                     false/*closing*/, false/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
#ifdef DEBUG_PERF
      end_perf_trace(Runtime::perf_trace_tolerance);
#endif
      return result;
    }
    
    //--------------------------------------------------------------------------
    InstanceRef RegionTreeForest::initialize_physical_context(
                                                RegionTreeContext ctx,
                                                const RegionRequirement &req,
                                                PhysicalManager *manager,
                                                Event term_event,
                                                Processor local_proc,
                                                unsigned depth,
                             std::map<PhysicalManager*,LogicalView*> &top_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *top_node = get_node(req.region);
      PhysicalUser user(RegionUsage(req), 
          top_node->column_source->get_field_mask(req.privilege_fields),
          term_event);
      LogicalView *new_view = NULL;
      if (manager->is_reduction_manager())
      {
        std::map<PhysicalManager*,LogicalView*>::const_iterator finder = 
          top_views.find(manager);
        if (finder == top_views.end())
        {
          new_view = manager->as_reduction_manager()->create_view();
          top_views[manager] = new_view;
        }
        else
          new_view = finder->second;
      }
      else
      {
        InstanceManager *inst_manager = manager->as_instance_manager();
#ifdef DEBUG_HIGH_LEVEL
        assert(inst_manager != NULL);
#endif
        std::map<PhysicalManager*,LogicalView*>::const_iterator finder = 
          top_views.find(manager);
        MaterializedView *top_view = NULL;
        if (finder == top_views.end())
        {
          top_view = inst_manager->create_top_view(depth);
          top_views[manager] = top_view;
        }
        else
          top_view = finder->second->as_instance_view()->as_materialized_view();
#ifdef DEBUG_HIGH_LEVEL
        assert(top_view != NULL);
#endif
        // Now walk from the top view down to the where the 
        // node is that we're initializing
        // First compute the path
        std::vector<Color> path;
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        compute_index_path(inst_manager->region_node->row_source->handle,
                           top_node->row_source->handle, path);
#ifdef DEBUG_HIGH_LEVEL
        assert(result);
        assert(!path.empty());
#endif
        // Note we don't need to traverse the last element
        for (int idx = int(path.size())-2; idx >= 0; idx--)
          top_view = top_view->get_materialized_subview(path[idx]);
        // Once we've made it down to the child we are done
#ifdef DEBUG_HIGH_LEVEL
        assert(top_view->logical_node == top_node);
#endif
        new_view = top_view;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(new_view != NULL);
#endif
      // It's actually incorrect to do full initialization of the
      // region tree here in case we have multiple region requirements
      // that overlap with each other.
      // Now seed the top node
      return top_node->seed_state(ctx.get_id(), user, new_view, local_proc);
    }
    
    //--------------------------------------------------------------------------
    void RegionTreeForest::invalidate_physical_context(RegionTreeContext ctx,
                                                       LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode *top_node = get_node(handle);
      PhysicalInvalidator invalidator(ctx.get_id());
      top_node->visit_node(&invalidator);
    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::close_physical_context(RegionTreeContext ctx,
                                                  RegionRequirement &req,
                                                  Mappable *mappable,
                                                  SingleTask *parent_ctx,
                                                  Processor local_proc,
                                                  const InstanceRef &ref
#ifdef DEBUG_HIGH_LEVEL
                                                  , unsigned index
                                                  , const char *log_name
                                                  , UniqueID uid
#endif
                                                  )
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(req.handle_type == SINGULAR);
#endif
      RegionNode *top_node = get_node(req.region);
      FieldMask user_mask = 
        top_node->column_source->get_field_mask(req.privilege_fields);
      PhysicalUser user(RegionUsage(req), user_mask, Event::NO_EVENT);      
      MappableInfo info(ctx.get_id(), mappable, local_proc, req, user_mask);
      // We serialize close operations with premappings
      UserEvent close_term = 
        parent_ctx->begin_premapping(req.region.tree_id, user_mask);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     top_node, ctx.get_id(), 
                                     true/*before*/, false/*premap*/, 
                                     true/*closing*/, false/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
      Event result = top_node->close_state(&info, user, ref);
#ifdef DEBUG_HIGH_LEVEL
      TreeStateLogger::capture_state(runtime, &req, index, log_name, uid,
                                     top_node, ctx.get_id(), 
                                     false/*before*/, false/*premap*/, 
                                     true/*closing*/, false/*logical*/,
                                     FieldMask(FIELD_ALL_ONES), user_mask);
#endif
      parent_ctx->end_premapping(req.region.tree_id, close_term);
      return result;
    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::copy_across(Mappable *mappable,
                                        Processor local_proc,
                                        RegionTreeContext src_ctx,
                                        RegionTreeContext dst_ctx,
                                        RegionRequirement &src_req,
                                        const RegionRequirement &dst_req,
                                        const InstanceRef &dst_ref,
                                        Event pre)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_PERF
      begin_perf_trace(COPY_ACROSS_ANALYSIS);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(src_req.handle_type == SINGULAR);
      assert(dst_req.handle_type == SINGULAR);
      assert(dst_ref.has_ref());
      assert(src_req.instance_fields.size() == dst_req.instance_fields.size());
#endif     
      MaterializedView *dst_view = dst_ref.get_handle().get_view()->
                              as_instance_view()->as_materialized_view();
      // Find the valid instance views for the source and then sort them
      std::map<MaterializedView*,FieldMask> src_instances;
      std::map<CompositeView*,FieldMask> composite_instances;
      RegionNode *src_node = get_node(src_req.region);
      FieldMask src_mask = 
        src_node->column_source->get_field_mask(src_req.privilege_fields);

      RegionNode *dst_node = get_node(dst_req.region);
      FieldMask dst_mask = 
        dst_node->column_source->get_field_mask(dst_req.privilege_fields);
      MappableInfo info(src_ctx.get_id(), mappable, 
                        local_proc, src_req, src_mask);
      src_node->find_copy_across_instances(&info, dst_view,
                                           src_instances, composite_instances);
      // Now is where things get tricky, since we don't have any correspondence
      // between fields in the two different requirements we can't use our 
      // normal copy routines. Instead we'll issue copies one field at a time
      std::set<Event> result_events;
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      Event dst_pre = dst_ref.get_ready_event(); 
      Event precondition = Event::merge_events(pre, dst_pre);
      // Also compute all of the source preconditions for each instance
      std::map<MaterializedView*,std::map<Event,FieldMask> > src_preconditions;
      for (std::map<MaterializedView*,FieldMask>::const_iterator it = 
            src_instances.begin(); it != src_instances.end(); it++)
      {
        std::map<Event,FieldMask> &preconditions = src_preconditions[it->first];
        it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                           it->second, preconditions);
      }
      for (unsigned idx = 0; idx < src_req.instance_fields.size(); idx++)
      {
        src_fields.clear();
        dst_fields.clear();
        unsigned src_index = src_node->column_source->get_field_index(
                                            src_req.instance_fields[idx]);
        bool found = false;
        // Iterate through the instances and see if we can find
        // a materialized views for the source field
        for (std::map<MaterializedView*,FieldMask>::const_iterator sit = 
              src_instances.begin(); sit != src_instances.end(); sit++)
        {
          if (sit->second.is_set(src_index))
          {
            // Compute the src_dst fields  
            sit->first->copy_field(src_req.instance_fields[idx], src_fields);
            dst_view->copy_field(dst_req.instance_fields[idx], dst_fields);
            // Compute the event preconditions
            std::set<Event> preconditions;
            preconditions.insert(precondition);
            // Find the source and destination preconditions
            for (std::map<Event,FieldMask>::const_iterator it = 
                  src_preconditions[sit->first].begin(); it !=
                  src_preconditions[sit->first].end(); it++)
            {
              if (it->second.is_set(src_index))
                preconditions.insert(it->first);
            }
            // Now we've got all the preconditions so we can actually
            // issue the copy operation
            Event copy_pre = Event::merge_events(preconditions);
            Event copy_post = dst_node->perform_copy_operation(
                                  copy_pre, src_fields, dst_fields);
            // Register the users of the post condition
            FieldMask local_src; local_src.set_bit(src_index);
            sit->first->add_copy_user(0/*redop*/, copy_post,
                                      local_src, true/*reading*/,
                                      info.local_proc);
            // No need to register a user for the destination because
            // we've already mapped it.
            result_events.insert(copy_post);
            found = true;
            break;
          }
        }
        if (!found)
        {
          // Check the composite instances
          for (std::map<CompositeView*,FieldMask>::const_iterator it = 
                composite_instances.begin(); it != 
                composite_instances.end(); it++)
          {
            if (it->second.is_set(src_index))
            {
              it->first->issue_composite_copies_across(&info, dst_view,
                                          src_req.instance_fields[idx],
                                          dst_req.instance_fields[idx],
                                          precondition, result_events);
              found = true;
              break;
            }
          }
        }
        // If we still didn't find it then there are no valid
        // instances for the data yet so we're done anyway
      }
      Event result = Event::merge_events(result_events);
#ifdef DEBUG_PERF
      end_perf_trace(Runtime::perf_trace_tolerance);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Event RegionTreeForest::copy_across(RegionTreeContext src_ctx, 
                                        RegionTreeContext dst_ctx,
                                        const RegionRequirement &src_req,
                                        const RegionRequirement &dst_req,
                                        const InstanceRef &src_ref,
                                        const InstanceRef &dst_ref,
                                        Event precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      begin_perf_trace(COPY_ACROSS_ANALYSIS);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(src_req.handle_type == SINGULAR);
      assert(dst_req.handle_type == SINGULAR);
      assert(src_ref.has_ref());
      assert(dst_ref.has_ref());
#endif
      // We already have the events for using the physical instances
      // All we need to do is get the offsets for performing the copies
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      MaterializedView *src_view = src_ref.get_handle().get_view()->
                              as_instance_view()->as_materialized_view();
      MaterializedView *dst_view = dst_ref.get_handle().get_view()->
                              as_instance_view()->as_materialized_view();
      src_view->manager->compute_copy_offsets(src_req.instance_fields, 
                                              src_fields);
      dst_view->manager->compute_copy_offsets(dst_req.instance_fields, 
                                              dst_fields);

      Event copy_pre = Event::merge_events(src_ref.get_ready_event(),
                                           dst_ref.get_ready_event(),
                                           precondition);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      if (!copy_pre.exists())
      {
        UserEvent new_copy_pre = UserEvent::create_user_event();
        new_copy_pre.trigger();
        copy_pre = new_copy_pre;
      }
#endif
#ifdef LEGION_LOGGING
      {
        Processor exec_proc = Machine::get_executing_processor();
        LegionLogging::log_event_dependence(exec_proc, 
            src_ref.get_ready_event(), copy_pre);
        LegionLogging::log_event_dependence(exec_proc,
            dst_ref.get_ready_event(), copy_pre);
        LegionLogging::log_event_dependence(exec_proc,
            precondition, copy_pre);
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(src_ref.get_ready_event(), copy_pre);
      LegionSpy::log_event_dependence(dst_ref.get_ready_event(), copy_pre);
      LegionSpy::log_event_dependence(precondition, copy_pre);
#endif
      RegionNode *dst_node = get_node(dst_req.region);
      // See if we have component domains or whether there is only
      // one domain for which the copy needs to be issued
      Event result;
      if (dst_node->has_component_domains())
      {
        const std::set<Domain> &component_domains = 
                                  dst_node->get_component_domains();
        std::set<Event> result_events;
        for (std::set<Domain>::const_iterator it = component_domains.begin();
              it != component_domains.end(); it++)
        {
          result_events.insert(it->copy(src_fields, dst_fields, copy_pre));
        }
        result = Event::merge_events(result_events);
      }
      else
      {
        Domain copy_domain = dst_node->get_domain();
        result = copy_domain.copy(src_fields, dst_fields, copy_pre);  
      }
      // Note we don't need to add the copy users because
      // we already mapped these regions as part of the CopyOp.
#ifdef LEGION_SPY
      if (!result.exists())
      {
        UserEvent new_result = UserEvent::create_user_event();
        new_result.trigger();
        result = new_result;
      }
      {
        RegionNode *src_node = get_node(src_req.region);
        FieldMask src_mask = 
          src_node->column_source->get_field_mask(src_req.privilege_fields);
        FieldMask dst_mask = 
          dst_node->column_source->get_field_mask(dst_req.privilege_fields);
        char *field_mask = src_node->column_source->to_string(src_mask);
        LegionSpy::log_copy_operation(src_view->manager->get_instance().id,
                                      dst_view->manager->get_instance().id,
                                      src_node->handle.index_space.id,
                                      src_node->handle.field_space.id,
                                      src_node->handle.tree_id,
                                      copy_pre, result, src_req.redop,
                                      field_mask);
        free(field_mask);
      }
#endif
#ifdef DEBUG_PERF
      end_perf_trace(Runtime::perf_trace_tolerance);
#endif
      // No need to add copy users since we added them when we
      // mapped this copy operation
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_physical_state(RegionTreeContext ctx,
                                               const RegionRequirement &req,
                                               UniqueID unique_id,
                                               AddressSpaceID target,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *top_node;
      FieldSpaceNode *field_node;
      if (req.handle_type == PART_PROJECTION)
      {
        // Handling partition projection
        PartitionNode *part_node = get_node(req.partition);
        // Send the shape of the tree
        // Index space tree first
        part_node->row_source->send_node(target, true/*up*/, true/*down*/);
        // The field space
        part_node->column_source->send_node(target);
        // Finally need the top node in our tree
        part_node->send_node(target);
        top_node = part_node;
        field_node = part_node->column_source;
      }
      else
      {
        // We're dealing with region nodes
        RegionNode *reg_node = get_node(req.region);
        // First send the shape of the tree
        // Index space tree first
        reg_node->row_source->send_node(target, true/*up*/, true/*down*/);
        // Then field space
        reg_node->column_source->send_node(target);
        // Finally need the top node of our region tree
        reg_node->send_node(target);
        top_node = reg_node; 
        field_node = reg_node->column_source;
      }
      // Invalidate if it is a singlular region requirement and writes
      bool invalidate = (req.handle_type == SINGULAR) && IS_WRITE(req);
      // Construct a traverser to send the state
      FieldMask send_mask = field_node->get_field_mask(req.privilege_fields);
      StateSender sender(ctx.get_id(), unique_id, target, 
          needed_views, needed_managers, send_mask, invalidate);
      // Now we're ready to send the state
      top_node->visit_node(&sender);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_tree_shape(const IndexSpaceRequirement &req,
                                           AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(req.handle);
      node->send_node(target, true/*up*/, true/*down*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_tree_shape(const RegionRequirement &req,
                                           AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (req.handle_type == PART_PROJECTION)
      {
        PartitionNode *part_node = get_node(req.partition);
        // Index space tree first
        part_node->row_source->send_node(target, true/*up*/, true/*down*/);
        // Then the field space
        part_node->column_source->send_node(target);
        // Finally the top node of the region tree
        part_node->send_node(target);
      }
      else
      {
        RegionNode *reg_node = get_node(req.region);
        // Index space tree first
        reg_node->row_source->send_node(target, true/*up*/, true/*down*/);
        // Then the field space
        reg_node->column_source->send_node(target);
        // Finally the top node of the region tree
        reg_node->send_node(target);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_tree_shape(IndexSpace handle, 
                                           AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = get_node(handle);
      node->send_node(target, true/*up*/, true/*down*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_tree_shape(FieldSpace handle,
                                           AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *node = get_node(handle);
      node->send_node(target);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_tree_shape(LogicalRegion handle,
                                           AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RegionNode *node = get_node(handle);
      node->row_source->send_node(target, true/*up*/, true/*down*/);
      node->column_source->send_node(target);
      node->send_node(target);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_back_physical_state(RegionTreeContext ctx,
                                                  RegionTreeContext rem_ctx,
                                                  RegionTreePath &path,
                                                  const RegionRequirement &req,
                                                  AddressSpaceID target,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *top_node;
      FieldSpaceNode *field_node;
      if (req.handle_type == PART_PROJECTION)
      {
        PartitionNode *part_node = get_node(req.partition);
        // Send any region tree shape information
        // Index space tree first
        part_node->row_source->send_node(target, true/*up*/, true/*down*/);
        // The the field space
        part_node->column_source->send_node(target);
        // Finally the top node of our region tree
        part_node->send_node(target);
        top_node = part_node;
        field_node = part_node->column_source;
      }
      else
      {
        RegionNode *reg_node = get_node(req.region);
        // Send any region tree shape information
        // Index space tree first
        reg_node->row_source->send_node(target, true/*up*/, true/*down*/);
        // Then field space
        reg_node->column_source->send_node(target);
        // Finally the top node of our region tree
        reg_node->send_node(target);
        top_node = reg_node;
        field_node = reg_node->column_source;
      }

      // Get the field mask for the return
      FieldMask return_mask = field_node->get_field_mask(req.privilege_fields);
      // See if we need to do a merge traversal for any regions which
      // were initially a projection requirement
      if (path.get_path_length() > 1)
      {
        // If we're here it's because we had a projection requirement
        PathReturner returner(path, ctx.get_id(), rem_ctx, target, 
                              return_mask, needed_managers);
        // Get the start node
        RegionTreeNode *start_node = top_node;
        for (unsigned idx = 0; idx < (path.get_path_length()-1); idx++)
          start_node = start_node->get_parent();
        returner.traverse(start_node);
      }
      // If we were writing then we will need to invalidate any nodes
      // below here
      const bool invalidate = IS_WRITE(req);
      // Send back the rest of the nodes
      StateReturner returner(ctx.get_id(), rem_ctx, target, invalidate, 
                             return_mask, needed_managers);
      top_node->visit_node(&returner);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_remote_references(
       const std::set<PhysicalManager*> &needed_managers, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // See which physical managers actually need to send their
      // remote references
      std::deque<DistributedID> send_ids;
      for (std::set<PhysicalManager*>::const_iterator it = 
            needed_managers.begin(); it != needed_managers.end(); it++)
      {
        if ((*it)->send_remote_reference(target))
          send_ids.push_back((*it)->did);
      }
      if (!send_ids.empty())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          size_t num_views = 0;
          rez.serialize(num_views);
          rez.serialize(send_ids.size());
          for (unsigned idx = 0; idx < send_ids.size(); idx++)
            rez.serialize(send_ids[idx]);
        }
        runtime->send_remote_references(target, rez);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::send_remote_references(
                           const std::map<LogicalView*,FieldMask> &needed_views,
       const std::set<PhysicalManager*> &needed_managers, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // See which physical managers actually need to send their
      // remote references
      std::deque<DistributedID> send_ids;
      for (std::set<PhysicalManager*>::const_iterator it = 
            needed_managers.begin(); it != needed_managers.end(); it++)
      {
        if ((*it)->send_remote_reference(target))
          send_ids.push_back((*it)->did);
      }
      if (!send_ids.empty() || !needed_views.empty())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(needed_views.size());
          for (std::map<LogicalView*,FieldMask>::const_iterator it = 
                needed_views.begin(); it != needed_views.end(); it++)
          {
            rez.serialize(it->first->find_distributed_id(target));
          }
          rez.serialize(send_ids.size());
          for (unsigned idx = 0; idx < send_ids.size(); idx++)
            rez.serialize(send_ids[idx]);
        }
        runtime->send_remote_references(target, rez);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::handle_remote_references(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_views;
      derez.deserialize(num_views);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        LogicalView *view = find_view(did);
        view->add_held_remote_reference();
      }
      size_t num_managers;
      derez.deserialize(num_managers);
      for (unsigned idx = 0; idx < num_managers; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        PhysicalManager *manager = find_manager(did);
        manager->add_held_remote_reference();
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::check_context_state(RegionTreeContext ctx)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeID,RegionNode*> trees;
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        trees = tree_nodes;
      }
      LogicalInitializer log_init(ctx.get_id());
      PhysicalInitializer phy_init(ctx.get_id());
      for (std::map<RegionTreeID,RegionNode*>::const_iterator it = 
            trees.begin(); it != trees.end(); it++)
      {
        it->second->visit_node(&log_init);
        it->second->visit_node(&phy_init);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::create_node(Domain d, 
                                                  IndexPartNode *parent,
                                                  Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, CREATE_NODE_CALL);
#endif
      IndexSpaceNode *result = new IndexSpaceNode(d, parent, c, this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      IndexSpace sp = d.get_index_space();
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpace,IndexSpaceNode*>::const_iterator it =
          index_nodes.find(sp);
        if (it != index_nodes.end())
        {
          delete result;
          return it->second;
        }
        index_nodes[sp] = result;
      }
      if (parent != NULL)
        parent->add_child(result);
      
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* RegionTreeForest::create_node(IndexPartition p, 
                                                 IndexSpaceNode *parent,
                                                 Color c, Domain color_space,
                                                 bool disjoint)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, CREATE_NODE_CALL);
#endif
      IndexPartNode *result = new IndexPartNode(p, parent, c, color_space,
                                                disjoint, this);
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
      assert(result != NULL);
#endif
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition,IndexPartNode*>::const_iterator it =
          index_parts.find(p);
        if (it != index_parts.end())
        {
          delete result;
          return it->second;
        }
        index_parts[p] = result;
      }
      if (parent != NULL)
        parent->add_child(result);
      
      return result;
    }
 
    //--------------------------------------------------------------------------
    FieldSpaceNode* RegionTreeForest::create_node(FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, CREATE_NODE_CALL);
#endif
      FieldSpaceNode *result = new FieldSpaceNode(space, this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      // Hold the lookup lock while modifying the lookup table
      AutoLock l_lock(lookup_lock);
      std::map<FieldSpace,FieldSpaceNode*>::const_iterator it =
        field_nodes.find(space);
      if (it != field_nodes.end())
      {
        delete result;
        return it->second;
      }
      field_nodes[space] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::create_node(LogicalRegion r, 
                                              PartitionNode *parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, CREATE_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      if (parent != NULL)
      {
        assert(r.field_space == parent->handle.field_space);
        assert(r.tree_id == parent->handle.tree_id);
      }
#endif
      IndexSpaceNode *row_src = get_node(r.index_space);
      FieldSpaceNode *col_src = get_node(r.field_space);
      RegionNode *result = new RegionNode(r, parent, row_src, 
                                          col_src, this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      // Special case here in case multiple clients attempt to
      // make the node at the same time
      {
        // Hold the lookup lock when modifying the lookup table
        AutoLock l_lock(lookup_lock);
        // Check to see if it already exists
        std::map<LogicalRegion,RegionNode*>::const_iterator it =
          region_nodes.find(r);
        if (it != region_nodes.end())
        {
          // It already exists, delete our copy and return
          // the one that has already been made
          delete result;
          return it->second;
        }
        // Now make sure that the node has the right number of contexts
        // before we put it on the map and release the look-up lock.
        // Note that this is only safe because the runtime ups the 
        // total_context count in allocate_context before calling the
        // resize_node_contexts funciton.
        result->reserve_contexts(runtime->get_context_count());
        // Now we can add it to the map
        region_nodes[r] = result;
        // If this is a top level region add it to the collection
        // of top level tree IDs
        if (parent == NULL)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(tree_nodes.find(r.tree_id) == tree_nodes.end());
#endif
          tree_nodes[r.tree_id] = result;
        }
      }
      // Now we can make the other ways of accessing the node available
      if (parent == NULL)
        col_src->add_instance(result);
      row_src->add_instance(result);
      if (parent != NULL)
        parent->add_child(result);
      
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::create_node(LogicalPartition p,
                                                 RegionNode *parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, CREATE_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
      assert(p.field_space == parent->handle.field_space);
      assert(p.tree_id = parent->handle.tree_id);
#endif
      IndexPartNode *row_src = get_node(p.index_partition);
      FieldSpaceNode *col_src = get_node(p.field_space);
      PartitionNode *result = new PartitionNode(p, parent, row_src, 
                                                col_src, this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      // Special case here in case multiple clients attempt
      // to make the node at the same time
      {
        // Hole the lookup lock when modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<LogicalPartition,PartitionNode*>::const_iterator it =
          part_nodes.find(p);
        if (it != part_nodes.end())
        {
          // It already exists, delete our copy and
          // return the one that has already been made
          delete result;
          return it->second;
        }
        // Now make sure that the node has the right number of contexts
        // before we put it on the map and release the look-up lock.
        // Note that this is only safe because the runtime ups the 
        // total_context count in allocate_context before calling the
        // resize_node_contexts funciton.
        result->reserve_contexts(runtime->get_context_count());
        // Now we can put the node in the map
        part_nodes[p] = result;
      }
      // Now we can make the other ways of accessing the node available
      row_src->add_instance(result);
      parent->add_child(result);
      
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* RegionTreeForest::get_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, GET_NODE_CALL);
#endif
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/); 
      std::map<IndexSpace,IndexSpaceNode*>::const_iterator it = 
        index_nodes.find(space);
      if (it == index_nodes.end())
      {
        log_index(LEVEL_ERROR,"Unable to find entry for index space " IDFMT "."
                              "This is either a runtime bug, or requires "
                              "Legion fences if index space names are being "
                              "returned out of the context in which they are "
                              "created.",
                              space.id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_PERF
      PerfTracer tracer(this, GET_NODE_CALL);
#endif
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<IndexPartition,IndexPartNode*>::const_iterator it =
        index_parts.find(part);
      if (it == index_parts.end())
      {
        log_index(LEVEL_ERROR,"Unable to find entry for index partition %x. "
                              "This is either a runtime bug, or requires "
                              "Legion fences if index partition names are "
                              "being returned out of the context in which "
                              "they are created.", part);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_PERF
      PerfTracer tracer(this, GET_NODE_CALL);
#endif
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<FieldSpace,FieldSpaceNode*>::const_iterator it = 
        field_nodes.find(space);
      if (it == field_nodes.end())
      {
        log_field(LEVEL_ERROR,"Unable to find entry for field space %x.  This "
                              "is either a runtime bug, or requires Legion "
                              "fences if field space names are being returned "
                              "out of the context in which they are created.",
                              space.id);
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_PERF
      PerfTracer tracer(this, GET_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      if (!has_node(handle.index_space))
      {
        log_region(LEVEL_ERROR,"Unable to find index space entry " IDFMT " for "
                               "logical region. This is either a runtime bug "
                               "or requires Legion fences if names are being "
                               "returned out of the context in which they are "
                               "being created.", handle.index_space.id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_REGION_ENTRY);
      }
      if (!has_node(handle.field_space))
      {
        log_region(LEVEL_ERROR,"Unable to find field space entry %x for "
                               "logical region. This is either a runtime bug "
                               "or requires Legion fences if names are being "
                               "returned out of the context in which they are "
                               "being created.", handle.field_space.id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_REGION_ENTRY);
      }
      if (!has_tree(handle.tree_id))
      {
        log_region(LEVEL_ERROR,"Unable to find region tree ID %x for "
                               "logical region. This is either a runtime bug "
                               "or requires Legion fences if names are being "
                               "returned out of the context in which they are "
                               "being created.", handle.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_REGION_ENTRY);
      }
#endif
      // Check to see if the node already exists
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<LogicalRegion,RegionNode*>::const_iterator it = 
          region_nodes.find(handle);
        if (it != region_nodes.end())
          return it->second;
      }
      // Otherwise it hasn't been made yet, so make it
      IndexSpaceNode *index_node = get_node(handle.index_space);
#ifdef DEBUG_HIGH_LEVEL
      assert(index_node->parent != NULL);
#endif
      LogicalPartition parent_handle(handle.tree_id, index_node->parent->handle,
                                     handle.field_space);
      // Note this request can recursively build more nodes, but we
      // are guaranteed that the top level node exists
      PartitionNode *parent = get_node(parent_handle);
      // Now make our node and then return it
      return create_node(handle, parent);
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionTreeForest::get_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, GET_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      if (!has_node(handle.index_partition))
      {
        log_region(LEVEL_ERROR,"Unable to find index partition entry %x for "
                               "logical partition.  This is either a runtime "
                               "bug or requires Legion fences if names are "
                               "being returned out of the context in which "
                               "they are being created.", 
                               handle.index_partition);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PARTITION_ENTRY);
      }
      if (!has_node(handle.field_space))
      {
        log_region(LEVEL_ERROR,"Unable to find field space entry %x for "
                               "logical partition.  This is either a runtime "
                               "bug or requires Legion fences if names are "
                               "being returned out of the context in which "
                               "they are being created.", 
                               handle.field_space.id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PARTITION_ENTRY);
      }
      if (!has_tree(handle.tree_id))
      {
        log_region(LEVEL_ERROR,"Unable to find region tree ID entry %x for "
                               "logical partition.  This is either a runtime "
                               "bug or requires Legion fences if names are "
                               "being returned out of the context in which "
                               "they are being created.", 
                               handle.tree_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PARTITION_ENTRY);
      }
#endif
      // Check to see if the node already exists
      {
        AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
        std::map<LogicalPartition,PartitionNode*>::const_iterator it =
          part_nodes.find(handle);
        if (it != part_nodes.end())
          return it->second;
      }
      // Otherwise it hasn't been made yet so make it
      IndexPartNode *index_node = get_node(handle.index_partition);
#ifdef DEBUG_HIGH_LEVEL
      assert(index_node->parent != NULL);
#endif
      LogicalRegion parent_handle(handle.tree_id, index_node->parent->handle,
                                  handle.field_space);
      // Note this request can recursively build more nodes, but we
      // are guaranteed that the top level node exists
      RegionNode *parent = get_node(parent_handle);
      // Now create our node and return it
      return create_node(handle, parent);
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionTreeForest::get_tree(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      std::map<RegionTreeID,RegionNode*>::const_iterator it = 
        tree_nodes.find(tid);
      if (it == tree_nodes.end())
      {
        log_region(LEVEL_ERROR,"Unable to find top-level tree entry for "
                               "region tree %d.  This is either a runtime "
                               "bug or requires Legion fences if names are "
                               "being returned out fo the context in which"
                               "they are being created.", tid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_TREE_ENTRY);
      }
      return it->second;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexSpace space) const
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (index_nodes.find(space) != index_nodes.end());
    }
    
    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(IndexPartition part) const
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (index_parts.find(part) != index_parts.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(FieldSpace space) const
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (field_nodes.find(space) != field_nodes.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      // Reflect that we can build these nodes whenever this is true
      return (has_node(handle.index_space) && has_node(handle.field_space) &&
              has_tree(handle.tree_id));
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_node(LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      // Reflect that we can build these nodes whenever this is true
      return (has_node(handle.index_partition) && has_node(handle.field_space)
              && has_tree(handle.tree_id));
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_tree(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      return (tree_nodes.find(tid) != tree_nodes.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      if (!has_node(space))
        return false;
      return get_node(space)->has_field(fid);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_disjoint(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = get_node(handle);
      return node->disjoint;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_disjoint(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return is_disjoint(handle.get_index_partition());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_disjoint(IndexSpace parent, IndexSpace child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, ARE_DISJOINT_CALL);
#endif
      std::vector<Color> path;
      if (compute_index_path(parent, child, path))
        return false;
      // Now check for a common ancestor and see if the
      // children are disjoint
      IndexSpaceNode *sp_one = get_node(parent);
      IndexSpaceNode *sp_two = get_node(child);
      // Bring them up to the same minimum depth
      unsigned depth = sp_one->depth;
      if (sp_two->depth < depth)
        depth = sp_two->depth;
      while (sp_one->depth > depth)
        sp_one = sp_one->parent->parent;
      while (sp_two->depth > depth)
        sp_two = sp_two->parent->parent;
      // Now we're at the same depth, we know they can't
      // equal or else there would have been a path
#ifdef DEBUG_HIGH_LEVEL
      assert(sp_one != sp_two);
      assert(sp_one->depth == sp_two->depth);
#endif
      while (sp_one->depth > 0)
      {
        // Check for a common partition
        if (sp_one->parent == sp_two->parent)
          return sp_one->parent->are_disjoint(sp_one->color, sp_two->color);
        // Check for a common new space
        if (sp_one->parent->parent == sp_two->parent->parent)
          return sp_one->parent->parent->are_disjoint(sp_one->parent->color,
                                                      sp_two->parent->color);
        // Otherwise advance everything
        sp_one = sp_one->parent->parent;
        sp_two = sp_two->parent->parent;
      }
      // Otherwise they're not even in the same tree
      // which guarantees disjointness
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_disjoint(IndexSpace parent, IndexPartition child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, ARE_DISJOINT_CALL);
#endif
      std::vector<Color> path;
      if (compute_partition_path(parent, child, path))
        return false;
      IndexPartNode *part_node = get_node(child);
      // Do a little check for a path between the partitions
      // parent and the parent node
      if (compute_index_path(parent, part_node->parent->handle, path))
      {
        path.pop_back(); // pop off the parent node's color
        return part_node->parent->are_disjoint(part_node->color,path.back());
      }
      if (compute_index_path(part_node->parent->handle, parent, path))
      {
        path.pop_back(); // pop off the parent node's color
        return part_node->parent->are_disjoint(part_node->color,path.back());
      }
      // Now check for a common ancestor and see if the
      // children are disjoint
      IndexSpaceNode *sp_one = get_node(parent);
      IndexSpaceNode *sp_two = part_node->parent;
      // Bring them up to the same minimum depth
      unsigned depth = sp_one->depth;
      if (sp_two->depth < depth)
        depth = sp_two->depth;
      while (sp_one->depth > depth)
        sp_one = sp_one->parent->parent;
      while (sp_two->depth > depth)
        sp_two = sp_two->parent->parent;
#ifdef DEBUG_HIGH_LEVEL
      assert(sp_one != sp_two);
      assert(sp_one->depth == sp_two->depth);
#endif
      while (sp_one->depth > 0)
      {
        // Check for a common partition
        if (sp_one->parent == sp_two->parent)
          return sp_one->parent->are_disjoint(sp_one->color, sp_two->color);
        // Check for a common new space
        if (sp_one->parent->parent == sp_two->parent->parent)
          return sp_one->parent->parent->are_disjoint(sp_one->parent->color,
                                                      sp_two->parent->color);
        // Otherwise advance everything
        sp_one = sp_one->parent->parent;
        sp_two = sp_two->parent->parent;
      }
      // Otherwise they are not in the same tree
      // and therefore by definition disjoint
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::are_compatible(IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *left_node = get_node(left);
      IndexSpaceNode *right_node = get_node(right);
      if (left_node->domain.get_dim() != right_node->domain.get_dim())
        return false;
      else if (left_node->domain.get_dim() == 0)
      {
        const LowLevel::ElementMask &left_mask = 
          left_node->handle.get_valid_mask();
        const LowLevel::ElementMask &right_mask = 
          right_node->handle.get_valid_mask();
        return (left_mask.get_num_elmts() == right_mask.get_num_elmts());
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::is_dominated(IndexSpace src, IndexSpace dst)
    //--------------------------------------------------------------------------
    {
      // Check to see if dst is dominated by source
#ifdef DEBUG_HIGH_LEVEL
      assert(are_compatible(src, dst));
#endif
      IndexSpaceNode *src_node = get_node(src);
      IndexSpaceNode *dst_node = get_node(dst);
      return src_node->dominates(dst_node);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_index_path(IndexSpace parent, 
                                    IndexSpace child, std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, COMPUTE_PATH_CALL);
#endif
      IndexSpaceNode *child_node = get_node(child); 
      path.push_back(child_node->color);
      if (parent == child) 
        return true; // Early out
      IndexSpaceNode *parent_node = get_node(parent);
      while (parent_node != child_node)
      {
        if (parent_node->depth >= child_node->depth)
        {
          path.clear();
          return false;
        }
        if (child_node->parent == NULL)
        {
          path.clear();
          return false;
        }
        path.push_back(child_node->parent->color);
        path.push_back(child_node->parent->parent->color);
        child_node = child_node->parent->parent;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::compute_partition_path(IndexSpace parent, 
                                IndexPartition child, std::vector<Color> &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, COMPUTE_PATH_CALL);
#endif
      IndexPartNode *child_node = get_node(child);
      path.push_back(child_node->color);
      if (child_node->parent == NULL)
      {
        path.clear();
        return false;
      }
      return compute_index_path(parent, child_node->parent->handle, path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexSpace child, IndexSpace parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexPartition child, 
                                           IndexSpace parent, 
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexSpace child,
                                           IndexPartition parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexPartition child,
                                           IndexPartition parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
      initialize_path(get_node(child), get_node(parent), path);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::initialize_path(IndexTreeNode *child,
                                           IndexTreeNode *parent,
                                           RegionTreePath &path)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(child->depth >= parent->depth);
#endif
      path.initialize(parent->depth, child->depth);
      while (child != parent)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(child->depth > 0);
#endif
        path.register_child(child->depth-1,child->color);
        child = child->get_parent();
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::register_physical_manager(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(managers.find(manager->did) == managers.end());
#endif
      managers[manager->did] = manager;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unregister_physical_manager(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(managers.find(did) != managers.end());
#endif
      managers.erase(did);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::register_logical_view(DistributedID did,
                                                 LogicalView *view)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(views.find(did) == views.end()); 
#endif
      views[did] = view;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::unregister_logical_view(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(views.find(did) != views.end());
#endif
      views.erase(did);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_manager(DistributedID did) const
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock,1,false/*exclusive*/);
      return (managers.find(did) != managers.end());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeForest::has_view(DistributedID did) const
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock,1,false/*exclusive*/);
      return (views.find(did) != views.end());
    }

    //--------------------------------------------------------------------------
    PhysicalManager* RegionTreeForest::find_manager(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock,1,false/*exclusive*/);
      std::map<DistributedID,PhysicalManager*>::const_iterator finder = 
        managers.find(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != managers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    LogicalView* RegionTreeForest::find_view(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_lock,1,false/*exclusive*/);
      std::map<DistributedID,LogicalView*>::const_iterator finder = 
        views.find(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != views.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    Color RegionTreeForest::generate_unique_color(
                                          const std::map<Color,T> &current_map)
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
    void RegionTreeForest::resize_node_contexts(unsigned total_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this, RESIZE_CONTEXTS_CALL);
#endif
      // We're only reading the maps of nodes, so we only need read permissions
      AutoLock l_lock(lookup_lock,1,false/*exclusive*/);
      for (std::map<LogicalRegion,RegionNode*>::const_iterator it = 
            region_nodes.begin(); it != region_nodes.end(); it++)
      {
        it->second->reserve_contexts(total_contexts);
      }
      for (std::map<LogicalPartition,PartitionNode*>::const_iterator it = 
            part_nodes.begin(); it != part_nodes.end(); it++)
      {
        it->second->reserve_contexts(total_contexts);
      }
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void RegionTreeForest::dump_logical_state(LogicalRegion region,
                                              ContextID ctx)
    //--------------------------------------------------------------------------
    {
      TreeStateLogger dump_logger; 
      assert(region_nodes.find(region) != region_nodes.end());
      region_nodes[region]->dump_logical_context(ctx, &dump_logger,
                                                 FieldMask(FIELD_ALL_ONES));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::dump_physical_state(LogicalRegion region,
                                               ContextID ctx)
    //--------------------------------------------------------------------------
    {
      TreeStateLogger dump_logger;
      assert(region_nodes.find(region) != region_nodes.end());
      region_nodes[region]->dump_physical_context(ctx, &dump_logger,
                                                  FieldMask(FIELD_ALL_ONES));
    }
#endif

#ifdef DYNAMIC_TESTS
    //--------------------------------------------------------------------------
    bool RegionTreeForest::perform_dynamic_tests(unsigned num_tests)
    //--------------------------------------------------------------------------
    {
      std::deque<DynamicSpaceTest> space_tests;
      std::deque<DynamicPartTest> part_tests;
      bool result;
      // Pull some tests off the queues
      {
        AutoLock d_lock(dynamic_lock);
        for (unsigned idx = 0; (idx < num_tests) &&
              !dynamic_space_tests.empty(); idx++)
        {
          space_tests.push_back(dynamic_space_tests.front());
          dynamic_space_tests.pop_front();
        }
        for (unsigned idx = 0; (idx < num_tests) &&
              !dynamic_part_tests.empty(); idx++)
        {
          part_tests.push_back(dynamic_part_tests.front());
          dynamic_part_tests.pop_front();
        }
        result = (!dynamic_space_tests.empty() ||
                  !dynamic_part_tests.empty());
      }
      for (std::deque<DynamicSpaceTest>::iterator it = space_tests.begin();
            it != space_tests.end(); it++)
      {
        it->perform_test();
      }
      for (std::deque<DynamicPartTest>::iterator it = part_tests.begin();
            it != part_tests.end(); it++)
      {
        it->perform_test();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::add_disjointness_test(const DynamicPartTest &test)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(dynamic_lock);
      dynamic_part_tests.push_back(test);
    }

    //--------------------------------------------------------------------------
    /*static*/ bool RegionTreeForest::are_disjoint(const Domain &left,
                                                   const Domain &right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(left.get_dim() == right.get_dim());
#endif
      bool disjoint = true;
      switch (left.get_dim())
      {
        case 0:
          {
            const LowLevel::ElementMask &left_mask = 
                                       left.get_index_space().get_valid_mask();
            const LowLevel::ElementMask &right_mask = 
                                       right.get_index_space().get_valid_mask();
            LowLevel::ElementMask::OverlapResult result = 
                                            left_mask.overlaps_with(right_mask);
            if (result != LowLevel::ElementMask::OVERLAP_NO)
              disjoint = false;
            break;
          }
        case 1:
          {
            Rect<1> leftr = left.get_rect<1>();
            Rect<1> rightr = right.get_rect<1>();
            if (leftr.overlaps(rightr))
              disjoint = false;
            break;
          }
        case 2:
          {
            Rect<2> leftr = left.get_rect<2>();
            Rect<2> rightr = right.get_rect<2>();
            if (leftr.overlaps(rightr))
              disjoint = false;
            break;
          }
        case 3:
          {
            Rect<3> leftr = left.get_rect<3>();
            Rect<3> rightr = right.get_rect<3>();
            if (leftr.overlaps(rightr))
              disjoint = false;
            break;
          }
        default:
          assert(false);
      }
      return disjoint;
    }

    //--------------------------------------------------------------------------
    /*static*/ bool RegionTreeForest::are_disjoint(IndexSpaceNode *left,
                                                   IndexSpaceNode *right)
    //--------------------------------------------------------------------------
    {
      bool disjoint = true;
      if (left->has_component_domains())
      {
        const std::set<Domain> &left_domains = left->get_component_domains();
        if (right->has_component_domains())
        {
          const std::set<Domain> &right_domains = 
                                              right->get_component_domains();
          // Double Loop
          for (std::set<Domain>::const_iterator lit = left_domains.begin();
                disjoint && (lit != left_domains.end()); lit++)
          {
            for (std::set<Domain>::const_iterator rit = right_domains.begin();
                  disjoint && (rit != right_domains.end()); rit++)
            {
              disjoint = RegionTreeForest::are_disjoint(*lit, *rit);
            }
          }
        }
        else
        {
          // Loop over left components
          for (std::set<Domain>::const_iterator it = left_domains.begin();
                disjoint && (it != left_domains.end()); it++)
          {
            disjoint = RegionTreeForest::are_disjoint(*it, right->domain);
          }
        }
      }
      else
      {
        if (right->has_component_domains())
        {
          const std::set<Domain> &right_domains = 
                                              right->get_component_domains();
          // Loop over right components
          for (std::set<Domain>::const_iterator it = right_domains.begin();
                disjoint && (it != right_domains.end()); it++)
          {
            disjoint = RegionTreeForest::are_disjoint(left->domain, *it);
          }
        }
        else
        {
          // No Loops
          disjoint = RegionTreeForest::are_disjoint(left->domain,right->domain);
        }
      }
      return disjoint;
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::DynamicSpaceTest::DynamicSpaceTest(IndexPartNode *par,
                                                         IndexSpaceNode *l, 
                                                         IndexSpaceNode *r)
      : parent(par), left(l), right(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::DynamicSpaceTest::perform_test(void) const
    //--------------------------------------------------------------------------
    {
      if (RegionTreeForest::are_disjoint(left, right))
        parent->add_disjoint(left->color, right->color);
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::DynamicPartTest::DynamicPartTest(IndexSpaceNode *par,
                                                       IndexPartNode *l, 
                                                       IndexPartNode *r)
      : parent(par), left(l), right(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::DynamicPartTest::add_child_space(bool l, 
                                                           IndexSpaceNode *node) 
    //--------------------------------------------------------------------------
    {
      if (l)
        left_spaces.push_back(node);
      else
        right_spaces.push_back(node);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::DynamicPartTest::perform_test(void) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<IndexSpaceNode*>::const_iterator lit = 
            left_spaces.begin(); lit != left_spaces.end(); lit++)
      {
        for (std::vector<IndexSpaceNode*>::const_iterator rit = 
              right_spaces.begin(); rit != right_spaces.end(); rit++)
        {
          if (!RegionTreeForest::are_disjoint(*lit, *rit))
            return;
        }
      }
      // If we made it here then they are disjoint
      parent->add_disjoint(left->color, right->color);
    }
#endif // DYNAMIC_TESTS

#ifdef DEBUG_PERF
    //--------------------------------------------------------------------------
    void RegionTreeForest::record_call(int kind, unsigned long long time)
    //--------------------------------------------------------------------------
    {
      Processor p = Machine::get_executing_processor();
      traces[p.local_id()].record_call(kind, time);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::begin_perf_trace(int kind)
    //--------------------------------------------------------------------------
    {
      Processor p = Machine::get_executing_processor();
      unsigned long long start = TimeStamp::get_current_time_in_micros();
      assert(p.local_id() < traces.size());
      traces[p.local_id()] = PerfTrace(kind, start);
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::end_perf_trace(unsigned long long tolerance)
    //--------------------------------------------------------------------------
    {
      Processor p = Machine::get_executing_processor();
      unsigned long long stop = TimeStamp::get_current_time_in_micros();
      PerfTrace &trace = traces[p.local_id()];
      unsigned long long diff = stop - trace.start;
      if (diff >= tolerance)
      {
        AutoLock t_lock(perf_trace_lock);
        trace.report_trace(diff);
      }
    }

    //--------------------------------------------------------------------------
    RegionTreeForest::PerfTrace::PerfTrace(int k, unsigned long long s)
      : tracing(true), kind(k), start(s)
    //--------------------------------------------------------------------------
    {
      // Allocate space for all of the calls
      for (unsigned idx = 0; idx < NUM_CALL_KIND; idx++)
        records.push_back(CallRecord(idx));
    }

    //--------------------------------------------------------------------------
    void RegionTreeForest::PerfTrace::report_trace(unsigned long long diff)
    //--------------------------------------------------------------------------
    {
      // Print out the kind of trace 
      switch (kind)
      {
        case REGION_DEPENDENCE_ANALYSIS:
          {
            fprintf(stdout,"REGION DEPENDENCE ANALYSIS: %lld us\n",diff);
            break;
          }
        case PREMAP_PHYSICAL_REGION_ANALYSIS:
          {
            fprintf(stdout,"PREMAP PHYSICAL REGION ANALYSIS: %lld us\n",diff);
            break;
          }
        case MAP_PHYSICAL_REGION_ANALYSIS:
          {
            fprintf(stdout,"MAP PHYSICAL REGION ANALYSIS: %lld us\n",diff);
            break;
          }
        case REMAP_PHYSICAL_REGION_ANALYSIS:
          {
            fprintf(stdout,"REMAP PHYSICAL REGION ANALYSIS: %lld us\n",diff);
            break;
          }
        case REGISTER_PHYSICAL_REGION_ANALYSIS:
          {
            fprintf(stdout,"REGISTER PHYSICAL REGION ANALYSIS: %lld us\n",diff);
            break;
          }
        case COPY_ACROSS_ANALYSIS:
          {
            fprintf(stdout,"COPY ACROSS ANALYSIS: %lld us\n",diff);
            break;
          }
        default:
          assert(false);
      }
      // Record all the call records which have a non-zero call count
      // Keep them in order from largest to smallest using a simple
      // insertion sort
      std::list<unsigned> record_indexes;
      for (unsigned idx = 0; idx < records.size(); idx++)
      {
        if (records[idx].count > 0)
        {
          bool inserted = false;
          for (std::list<unsigned>::iterator it = record_indexes.begin();
                it != record_indexes.end(); it++)
          {
            if (records[idx].total_time > records[*it].total_time)
            {
              record_indexes.insert(it, idx);
              inserted = true;
              break;
            }
          }
          if (!inserted)
            record_indexes.push_back(idx);
        }
      }

      // Then print out all the records
      for (std::list<unsigned>::const_iterator it = record_indexes.begin();
            it != record_indexes.end(); it++)
      {
        // Print out the kind of call record
        const CallRecord &rec = records[*it];
        switch (rec.kind)
        {
          case CREATE_NODE_CALL:
            {
              fprintf(stdout,"  Create Node Call:\n");
              break;
            }
          case GET_NODE_CALL:
            {
              fprintf(stdout,"  Get Node Call:\n");
              break;
            }
          case ARE_DISJOINT_CALL:
            {
              fprintf(stdout,"  Are Disjoint Call:\n");
              break;
            }
          case COMPUTE_PATH_CALL:
            {
              fprintf(stdout,"  Compute Path Call:\n");
              break;
            }
          case RESIZE_CONTEXTS_CALL:
            {
              fprintf(stdout,"  Resize Contexts Call:\n");
              break;
            }
          case CREATE_INSTANCE_CALL:
            {
              fprintf(stdout,"  Create Instance Call:\n");
              break;
            }
          case CREATE_REDUCTION_CALL:
            {
              fprintf(stdout,"  Create Reduction Call:\n");
              break;
            }
          case PERFORM_PREMAP_CLOSE_CALL:
            {
              fprintf(stdout,"  Perform Premap Close Call:\n");
              break;
            }
          case MAPPING_TRAVERSE_CALL:
            {
              fprintf(stdout,"  Mapping Traverse Call:\n");
              break;
            }
          case MAP_PHYSICAL_REGION_CALL:
            {
              fprintf(stdout,"  Map Physical Region Call:\n");
              break;
            }
          case MAP_REDUCTION_REGION_CALL:
            {
              fprintf(stdout,"  Map Reduction Region Call:\n");
              break;
            }
          case RESERVE_CONTEXTS_CALL:
            {
              fprintf(stdout,"  Reserve Contexts Call:\n");
              break;
            }
          case ACQUIRE_PHYSICAL_STATE_CALL:
            {
              fprintf(stdout,"  Acquire Physical State Call:\n");
              break;
            }
          case RELEASE_PHYSICAL_STATE_CALL:
            {
              fprintf(stdout,"  Release Physical State Call:\n");
              break;
            }
          case REGISTER_LOGICAL_NODE_CALL:
            {
              fprintf(stdout,"  Register Logical Node Call:\n");
              break;
            }
          case OPEN_LOGICAL_NODE_CALL:
            {
              fprintf(stdout,"  Open Logical Node Call:\n");
              break;
            }
          case CLOSE_LOGICAL_NODE_CALL:
            {
              fprintf(stdout,"  Close Logical Node Call:\n");
              break;
            }
          case SIPHON_LOGICAL_CHILDREN_CALL:
            {
              fprintf(stdout,"  Siphon Logical Children Call:\n");
              break;
            }
          case PERFORM_LOGICAL_CLOSE_CALL:
            {
              fprintf(stdout,"  Perform Logical Close Call:\n");
              break;
            }
          case RECORD_CLOSE_CALL:
            {
              fprintf(stdout,"  Record Close Call:\n");
              break;
            }
          case UPDATE_CLOSE_CALL:
            {
              fprintf(stdout,"  Update Close Call:\n");
              break;
            }
          case ADVANCE_FIELD_CALL:
            {
              fprintf(stdout,"  Advance Field Call:\n");
              break;
            }
          case FILTER_PREV_EPOCH_CALL:
            {
              fprintf(stdout,"  Filter Previous Epoch Call:\n");
              break;
            }
          case FILTER_CURR_EPOCH_CALL:
            {
              fprintf(stdout,"  Filter Current Epoch Call:\n");
              break;
            }
          case FILTER_CLOSE_CALL:
            {
              fprintf(stdout,"  Filter Close Call:\n");
              break;
            }
          case INITIALIZE_LOGICAL_CALL:
            {
              fprintf(stdout,"  Initialize Logical Call:\n");
              break;
            }
          case INVALIDATE_LOGICAL_CALL:
            {
              fprintf(stdout,"  Invalidate Logical Call:\n");
              break;
            }
          case REGISTER_LOGICAL_DEPS_CALL:
            {
              fprintf(stdout,"  Register Logical Dependences Call:\n");
              break;
            }
          case CLOSE_PHYSICAL_NODE_CALL:
            {
              fprintf(stdout,"  Close Physical Node Call:\n");
              break;
            }
          case SELECT_CLOSE_TARGETS_CALL:
            {
              fprintf(stdout,"  Select Close Targets Call:\n");
              break;
            }
          case SIPHON_PHYSICAL_CHILDREN_CALL:
            {
              fprintf(stdout,"  Siphon Physical Children Call:\n");
              break;
            }
          case CLOSE_PHYSICAL_CHILD_CALL:
            {
              fprintf(stdout,"  Close Physical Child Call:\n");
              break;
            }
          case FIND_VALID_INSTANCE_VIEWS_CALL:
            {
              fprintf(stdout,"  Find Valid Instance Views Call:\n");
              break;
            }
          case FIND_VALID_REDUCTION_VIEWS_CALL:
            {
              fprintf(stdout,"  Find Valid Reduction Views Call:\n");
              break;
            }
          case PULL_VALID_VIEWS_CALL:
            {
              fprintf(stdout,"  Pull Valid Views Call:\n");
              break;
            }
          case FIND_COPY_ACROSS_INSTANCES_CALL:
            {
              fprintf(stdout,"  Find Copy Across Instances Call:\n");
              break;
            }
          case ISSUE_UPDATE_COPIES_CALL:
            {
              fprintf(stdout,"  Issue Update Copies Call:\n");
              break;
            }
          case ISSUE_UPDATE_REDUCTIONS_CALL:
            {
              fprintf(stdout,"  Issue Update Reductions Call:\n");
              break;
            }
          case PERFORM_COPY_DOMAIN_CALL:
            {
              fprintf(stdout,"  Perform Copy Domain Call:\n");
              break;
            }
          case INVALIDATE_INSTANCE_VIEWS_CALL:
            {
              fprintf(stdout,"  Invalidate Instance Views Call:\n");
              break;
            }
          case INVALIDATE_REDUCTION_VIEWS_CALL:
            {
              fprintf(stdout,"  Invalidate Reduction Views Call:\n");
              break;
            }
          case UPDATE_VALID_VIEWS_CALL:
            {
              fprintf(stdout,"  Update Valid Views Call:\n");
              break;
            }
          case UPDATE_REDUCTION_VIEWS_CALL:
            {
              fprintf(stdout,"  Update Reduction Views Call:\n");
              break;
            }
          case FLUSH_REDUCTIONS_CALL:
            {
              fprintf(stdout,"  Flush Reductions Call:\n");
              break;
            }
          case INITIALIZE_PHYSICAL_STATE_CALL:
            {
              fprintf(stdout,"  Initialize Physical State Call:\n");
              break;
            }
          case INVALIDATE_PHYSICAL_STATE_CALL:
            {
              fprintf(stdout,"  Invalidate Physical State Call:\n");
              break;
            }
          case PERFORM_DEPENDENCE_CHECKS_CALL:
            {
              fprintf(stdout,"  Perform Dependence Checks Call:\n");
              break;
            }
          case PERFORM_CLOSING_CHECKS_CALL:
            {
              fprintf(stdout,"  Perform Closing Checks Call:\n");
              break;
            }
          case REMAP_REGION_CALL:
            {
              fprintf(stdout,"  Remap Region Call:\n");
              break;
            }
          case REGISTER_REGION_CALL:
            {
              fprintf(stdout,"  Register Region Call:\n");
              break;
            }
          case CLOSE_PHYSICAL_STATE_CALL:
            {
              fprintf(stdout,"  Close Physical State Call:\n");
              break;
            }
          case GARBAGE_COLLECT_CALL:
            {
              fprintf(stdout,"  Garbage Collect Call:\n");
              break;
            }
          case NOTIFY_INVALID_CALL:
            {
              fprintf(stdout,"  Notify Invalid Call:\n");
              break;
            }
          case GET_RECYCLE_EVENT_CALL:
            {
              fprintf(stdout,"  Get Recycle Event Call:\n");
              break;
            }
          case DEFER_COLLECT_USER_CALL:
            {
              fprintf(stdout,"  Defer Collect User Call:\n");
              break;
            }
          case GET_SUBVIEW_CALL:
            {
              fprintf(stdout,"  Get Subview Call:\n");
              break;
            }
          case COPY_FIELD_CALL:
            {
              fprintf(stdout,"  Copy-Field Call:\n");
              break;
            }
          case COPY_TO_CALL:
            {
              fprintf(stdout,"  Copy-To Call:\n");
              break;
            }
          case REDUCE_TO_CALL:
            {
              fprintf(stdout,"  Reduce-To Call:\n");
              break;
            }
          case COPY_FROM_CALL:
            {
              fprintf(stdout,"  Copy-From Call:\n");
              break;
            }
          case REDUCE_FROM_CALL:
            {
              fprintf(stdout,"  Reduce-From Call:\n");
              break;
            }
          case HAS_WAR_DEPENDENCE_CALL:
            {
              fprintf(stdout,"  Has WAR Dependence Call:\n");
              break;
            }
          case ACCUMULATE_EVENTS_CALL:
            {
              fprintf(stdout,"  Accumulate Events Call:\n");
              break;
            }
          case ADD_COPY_USER_CALL:
            {
              fprintf(stdout,"  Add Copy User Call:\n");
              break;
            }
          case ADD_USER_CALL:
            {
              fprintf(stdout,"  Add User Call:\n");
              break;
            }
          case ADD_USER_ABOVE_CALL:
            {
              fprintf(stdout,"  Add User Above Call:\n");
              break;
            }
          case ADD_LOCAL_USER_CALL:
            {
              fprintf(stdout,"  Add Local User Call:\n");
              break;
            }
          case FIND_COPY_PRECONDITIONS_CALL:
            {
              fprintf(stdout,"  Find Copy Preconditions Call:\n");
              break;
            }
          case FIND_COPY_PRECONDITIONS_ABOVE_CALL:
            {
              fprintf(stdout,"  Find Copy Preconditions Above Call:\n");
              break;
            }
          case FIND_LOCAL_COPY_PRECONDITIONS_CALL:
            {
              fprintf(stdout,"  Find Local Copy Preconditions Call:\n");
              break;
            }
          case HAS_WAR_DEPENDENCE_ABOVE_CALL:
            {
              fprintf(stdout,"  Has WAR Dependence Above Call:\n");
              break;
            }
          case UPDATE_VERSIONS_CALL:
            {
              fprintf(stdout,"  Update Versions Call:\n");
              break;
            }
          case CONDENSE_USER_LIST_CALL:
            {
              fprintf(stdout,"  Condense User List Call:\n");
              break;
            }
          case PERFORM_REDUCTION_CALL:
            {
              fprintf(stdout,"  Perform Reduction Call:\n");
              break;
            }
          default:
            assert(false);
        }
        // Print out the statistics
        fprintf(stdout,"    Total calls: %d\n", rec.count);
        fprintf(stdout,"    Total time: %lld us\n", rec.total_time);
        fprintf(stdout,"    Avg time: %lld us\n", rec.total_time/rec.count);
        fprintf(stdout,"    Max time: %lld us\n", rec.max_time);
        fprintf(stdout,"    Min time: %lld us\n", rec.min_time);
      }
      fflush(stdout);
    }
#endif

    /////////////////////////////////////////////////////////////
    // Index Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTreeNode::IndexTreeNode(void)
      : depth(0), color(0), context(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTreeNode::IndexTreeNode(Color c, unsigned d, RegionTreeForest *ctx)
      : depth(d), color(c), context(ctx), 
        node_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx != NULL);
#endif
    }

    //--------------------------------------------------------------------------
    IndexTreeNode::~IndexTreeNode(void)
    //--------------------------------------------------------------------------
    {
      node_lock.destroy_reservation();
      node_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    /*static*/ bool IndexTreeNode::compute_intersections(
        const std::set<Domain> &left, const std::set<Domain> &right, 
        std::set<Domain> &result_domains)
    //--------------------------------------------------------------------------
    {
      for (std::set<Domain>::const_iterator lit = left.begin();
            lit != left.end(); lit++)
      {
        for (std::set<Domain>::const_iterator rit = right.begin();
              rit != right.end(); rit++)
        {
          Domain result;
          if (compute_intersection(*lit, *rit, result))
            result_domains.insert(result);
        }
      }
      return !result_domains.empty();
    }

    //--------------------------------------------------------------------------
    /*static*/ bool IndexTreeNode::compute_intersections(
        const std::set<Domain> &left, const Domain &right, 
        std::set<Domain> &result_domains)
    //--------------------------------------------------------------------------
    {
      for (std::set<Domain>::const_iterator it = left.begin();
            it != left.end(); it++)
      {
        Domain result;
        if (compute_intersection(*it, right, result))
          result_domains.insert(result);
      }
      return !result_domains.empty();
    }

    //--------------------------------------------------------------------------
    /*static*/ bool IndexTreeNode::compute_intersection(const Domain &left,
                                                        const Domain &right,
                                                        Domain &result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(left.get_dim() == right.get_dim());
#endif
      bool non_empty = false;
      switch (left.get_dim())
      {
        case 0:
          {
            const LowLevel::ElementMask &left_mask = 
                                    left.get_index_space().get_valid_mask();
            const LowLevel::ElementMask &right_mask = 
                                    right.get_index_space().get_valid_mask();
            LowLevel::ElementMask intersection = left_mask & right_mask;
            if (!!intersection)
            {
              non_empty = true;
              result = Domain(IndexSpace::create_index_space(intersection));
            }
            break;
          }
        case 1:
          {
            Rect<1> leftr = left.get_rect<1>();
            Rect<1> rightr = right.get_rect<1>();
            Rect<1> temp = leftr.intersection(rightr);
            if (temp.volume() > 0)
            {
              non_empty = true;
              result = Domain::from_rect<1>(temp);
#ifdef SHARED_LOWLEVEL
              result.get_index_space(true/*create if needed*/);
#endif
            }
            break;
          }
        case 2:
          {
            Rect<2> leftr = left.get_rect<2>();
            Rect<2> rightr = right.get_rect<2>();
            Rect<2> temp = leftr.intersection(rightr);
            if (temp.volume() > 0)
            {
              non_empty = true;
              result = Domain::from_rect<2>(temp);
#ifdef SHARED_LOWLEVEL
              result.get_index_space(true/*create if needed*/);
#endif
            }
            break;
          }
        case 3:
          {
            Rect<3> leftr = left.get_rect<3>();
            Rect<3> rightr = right.get_rect<3>();
            Rect<3> temp = leftr.intersection(rightr);
            if (temp.volume() > 0)
            {
              non_empty = true;
              result = Domain::from_rect<3>(temp);
#ifdef SHARED_LOWLEVEL
              result.get_index_space(true/*create if needed*/);
#endif
            }
            break;
          }
        default:
          assert(false);
      }
      return non_empty;
    }

    //--------------------------------------------------------------------------
    /*static*/ bool IndexTreeNode::compute_dominates(
            const std::set<Domain> &left_set, const std::set<Domain> &right_set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!left_set.empty());
      assert(!right_set.empty());
#endif
      // Check to see if the left set of domains dominates the right set
      bool dominates = false;
      // Handle the easy case for dimension zero
      Domain left = *(left_set.begin());
      if (left.get_dim() == 0)
      {
        // Union left and right together and then test
        LowLevel::ElementMask left_mask, right_mask;
        bool first = true;
        for (std::set<Domain>::const_iterator it = left_set.begin();
              it != left_set.end(); it++)
        {
          if (first)
          {
            left_mask = it->get_index_space().get_valid_mask();
            first = false;
          }
          else
            left_mask |= it->get_index_space().get_valid_mask();
        }
        first = true;
        for (std::set<Domain>::const_iterator it = right_set.begin();
              it != right_set.end(); it++)
        {
          if (first)
          {
            right_mask = it->get_index_space().get_valid_mask();
            first = false;
          }
          else
            right_mask |= it->get_index_space().get_valid_mask();
        }
        LowLevel::ElementMask diff = right_mask - left_mask;
        if (!diff)
          dominates = true;
      }
      else if (left_set.size() == 1)
      {
        // This is the easy case where we only have a single domain on the left
        switch (left.get_dim())
        {
          case 1:
            {
              Rect<1> leftr = left.get_rect<1>();
              dominates = true;
              for (std::set<Domain>::const_iterator it = right_set.begin();
                    it != right_set.end(); it++)
              {
                Rect<1> right = it->get_rect<1>(); 
                if ((right.intersection(leftr)) != right)
                {
                  dominates = false;
                  break;
                }
              }
              break;
            }
          case 2:
            {
              Rect<2> leftr = left.get_rect<2>();
              dominates = true;
              for (std::set<Domain>::const_iterator it = right_set.begin();
                    it != right_set.end(); it++)
              {
                Rect<2> right = it->get_rect<2>(); 
                if ((right.intersection(leftr)) != right)
                {
                  dominates = false;
                  break;
                }
              }
              break;
            }
          case 3:
            {
              Rect<3> leftr = left.get_rect<3>();
              dominates = true;
              for (std::set<Domain>::const_iterator it = right_set.begin();
                    it != right_set.end(); it++)
              {
                Rect<3> right = it->get_rect<3>(); 
                if ((right.intersection(leftr)) != right)
                {
                  dominates = false;
                  break;
                }
              }
              break;
            }
          default:
            assert(false);
        }
      }
      else
      {
        // This is the hard case where we have multiple domains on the left
        switch (left.get_dim())
        {
          case 1:
            {
              // Construct an interval tree for the left set
              // and then check to see if all the intervals within
              // the right set are dominated by an interval in the tree
              IntervalTree<int,true/*discrete*/> intervals;
              for (std::set<Domain>::const_iterator it = left_set.begin();
                    it != left_set.end(); it++)
              {
                Rect<1> left_rect = it->get_rect<1>();
                intervals.insert(left_rect.lo[0], left_rect.hi[0]);
              }
              dominates = true;
              for (std::set<Domain>::const_iterator it = right_set.begin();
                    it != right_set.end(); it++)
              {
                Rect<1> right_rect = it->get_rect<1>();
                if (!intervals.dominates(right_rect.lo[0], right_rect.hi[0]))
                {
                  dominates = false;
                  break;
                }
              }
              break;
            }
          case 2:
            {
              RectangleSet<int,true/*discrete*/> rectangles;
              for (std::set<Domain>::const_iterator it = left_set.begin();
                    it != left_set.end(); it++)
              {
                Rect<2> left_rect = it->get_rect<2>();
                rectangles.add_rectangle(left_rect.lo[0], left_rect.lo[1],
                                         left_rect.hi[0], left_rect.hi[1]);
              }
              dominates = true;
              for (std::set<Domain>::const_iterator it = right_set.begin();
                    it != right_set.end(); it++)
              {
                Rect<2> right_rect = it->get_rect<2>();
                if (!rectangles.covers(right_rect.lo[0], right_rect.lo[1],
                                       right_rect.hi[0], right_rect.hi[1]))
                {
                  dominates = false;
                  break;
                }
              }
              break;
            }
          case 3:
            {
              // TODO: Improve this terrible approximation
              dominates = true;
              for (std::set<Domain>::const_iterator rit = right_set.begin();
                    (rit != right_set.end()) && dominates; rit++)
              {
                Rect<3> right_rect = rit->get_rect<3>();
                bool has_dominator = false;
                // See if any of the rectangles on the left dominate it
                for (std::set<Domain>::const_iterator lit = left_set.begin();
                      lit != left_set.end(); lit++)
                {
                  Rect<3> left_rect = lit->get_rect<3>();
                  if (right_rect.intersection(left_rect) == right_rect)
                  {
                    has_dominator = true;
                    break;
                  }
                }
                if (!has_dominator)
                {
                  dominates = false;
                  break;
                }
              }
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
      return dominates;
    }


    /////////////////////////////////////////////////////////////
    // Index Space Node 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(Domain d, IndexPartNode *par, Color c,
                                   RegionTreeForest *ctx)
      : IndexTreeNode(c, (par == NULL) ? 0 : par->depth+1, ctx),
        domain(d), handle(d.get_index_space()), parent(par), allocator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::IndexSpaceNode(const IndexSpaceNode &rhs)
      : IndexTreeNode(), domain(Domain::NO_DOMAIN), 
        handle(IndexSpace::NO_SPACE), parent(NULL), allocator(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode::~IndexSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      if (allocator != NULL)
      {
        free(allocator);
        allocator = NULL;
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode& IndexSpaceNode::operator=(const IndexSpaceNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* IndexSpaceNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::has_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return (color_map.find(c) != color_map.end());
    }

    //--------------------------------------------------------------------------
    IndexPartNode* IndexSpaceNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_child(IndexPartNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(child->color) == color_map.end());
#endif
      color_map[child->color] = child;
      valid_map[child->color] = child;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      valid_map.erase(c);
    }

    //--------------------------------------------------------------------------
    size_t IndexSpaceNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return valid_map.size();
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::are_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      // Quick out
      if (c1 == c2)
        return false;
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      if (disjoint_subsets.find(std::pair<Color,Color>(c1,c2)) !=
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
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      disjoint_subsets.insert(std::pair<Color,Color>(c1,c2));
      disjoint_subsets.insert(std::pair<Color,Color>(c2,c1));
    }

    //--------------------------------------------------------------------------
    Color IndexSpaceNode::generate_color(void)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return context->generate_unique_color(color_map);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::get_colors(std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::map<Color,IndexPartNode*>::const_iterator it = 
            valid_map.begin(); it != valid_map.end(); it++)
      {
        colors.insert(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_nodes.find(inst) == logical_nodes.end());
#endif
      logical_nodes.insert(inst);
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::has_instance(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle.get_tree_id() == tid)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_creation_source(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      creation_set.set_bit(source);
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->remove_child(color);
      AutoLock n_lock(node_lock);
      destruction_set.set_bit(source);
      for (std::set<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        (*it)->destroy_node(source);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::has_component_domains(void) const
    //--------------------------------------------------------------------------
    {
      return !component_domains.empty();
    }

    //--------------------------------------------------------------------------
    void IndexSpaceNode::update_component_domains(const std::set<Domain> &doms)
    //--------------------------------------------------------------------------
    {
      component_domains.insert(doms.begin(), doms.end());
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& IndexSpaceNode::get_component_domains(void) const
    //--------------------------------------------------------------------------
    {
      return component_domains;
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::intersects_with(IndexSpaceNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.has_intersects;
      }
      std::set<Domain> intersect;
      bool result, own_results;
      if (component_domains.empty())
      { 
#ifndef SHARED_LOWLEVEL
        own_results = (domain.get_dim() == 0);
#else
        own_results = true;
#endif
        if (other->has_component_domains())
          result = compute_intersections(other->get_component_domains(),
                                         domain, intersect);
        else
        {
          Domain inter;
          result = compute_intersection(domain, other->domain, inter);
          if (result)
            intersect.insert(inter);
        }
      }
      else
      {
#ifndef SHARED_LOWLEVEL
        own_results = false;
#else
        own_results = true;
#endif
        if (other->has_component_domains())
          result = compute_intersections(component_domains,
                        other->get_component_domains(), intersect);
        else
          result = compute_intersections(component_domains,
                                         other->domain, intersect); 
      }
      AutoLock n_lock(node_lock);
      if (result)
      {
        // Check to make sure we didn't lose the race
        if (intersections.find(other) == intersections.end())
        {
          intersections[other] = IntersectInfo(true/*has intersects*/,
                                               own_results, intersect);
        }
      }
      else
        intersections[other] = IntersectInfo(false/*has intersects*/, 
                                             false/*own*/);
      return result;
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::intersects_with(IndexPartNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.has_intersects;
      }
      // Build up the set of domains for the partition
      std::set<Domain> other_domains, intersect;
      other->get_subspace_domains(other_domains);
      bool result, own_results;
      if (component_domains.empty())
      {
        result = compute_intersections(other_domains, domain, intersect);
        // If the dim is zero then we own the index space results
#ifndef SHARED_LOWLEVEL
        own_results = (domain.get_dim() == 0);
#else
        own_results = true;
#endif
      }
      else
      {
        result = compute_intersections(component_domains, other_domains,
                                       intersect);
#ifndef SHARED_LOWLEVEL
        own_results = false;
#else
        own_results = true;
#endif
      }
      AutoLock n_lock(node_lock);
      if (result)
      {
        // Check to make sure we didn't lose the race
        if (intersections.find(other) == intersections.end())
        {
          intersections[other] = IntersectInfo(true/*has intersects*/,
                                               own_results, intersect);
        }
      }
      else
        intersections[other] = IntersectInfo(false/*has intersects*/,
                                             false/*own*/);
      return result;
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& IndexSpaceNode::get_intersection_domains(
                                                          IndexSpaceNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.intersections;
      }
      std::set<Domain> intersect;
      bool result, own_results;
      if (component_domains.empty())
      { 
#ifndef SHARED_LOWLEVEL
        own_results = (domain.get_dim() == 0);
#else
        own_results = true;
#endif
        if (other->has_component_domains())
          result = compute_intersections(other->get_component_domains(),
                                         domain, intersect);
        else
        {
          Domain inter;
          result = compute_intersection(domain, other->domain, inter);
          if (result)
            intersect.insert(inter);
        }
      }
      else
      {
#ifndef SHARED_LOWLEVEL
        own_results = false;
#else
        own_results = true;
#endif
        if (other->has_component_domains())
          result = compute_intersections(component_domains,
                        other->get_component_domains(), intersect);
        else
          result = compute_intersections(component_domains,
                                         other->domain, intersect); 
      }
      AutoLock n_lock(node_lock);
      if (result)
      {
        // Check again to make sure we didn't lose the race
        if (intersections.find(other) == intersections.end())
        {
          intersections[other] = IntersectInfo(true/*has intersects*/,
                                               own_results, intersect);
        }
      }
      else
        intersections[other] = IntersectInfo(false/*has intersects*/, 
                                             false/*own*/);
      return intersections[other].intersections;
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& IndexSpaceNode::get_intersection_domains(
                                                           IndexPartNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.intersections;
      }
      // Build up the set of domains for the partition
      std::set<Domain> other_domains, intersect;
      other->get_subspace_domains(other_domains);
      bool result, own_results;
      if (component_domains.empty())
      {
        result = compute_intersections(other_domains, domain, intersect);
        // If the dim is zero then we own the index space results
#ifndef SHARED_LOWLEVEL
        own_results = (domain.get_dim() == 0);
#else
        own_results = true;
#endif
      }
      else
      {
        result = compute_intersections(component_domains, other_domains,
                                       intersect);
#ifndef SHARED_LOWLEVEL
        own_results = false;
#else
        own_results = true;
#endif
      }
      AutoLock n_lock(node_lock);
      if (result)
      {
        // Check again to make sure we didn't lose the race
        if (intersections.find(other) == intersections.end())
        {
          intersections[other] = IntersectInfo(true/*has intersects*/,
                                               own_results, intersect);
        }
      }
      else
        intersections[other] = IntersectInfo(false/*has intersects*/,
                                             false/*own*/);
      return intersections[other].intersections;
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::dominates(IndexSpaceNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(other);
        if (finder != dominators.end())
          return finder->second;
      }
      bool result;
      if (component_domains.empty())
      {
        if (other->has_component_domains())
        {
          std::set<Domain> local;
          local.insert(domain);
          result = compute_dominates(local, other->get_component_domains());
        }
        else
        {
          std::set<Domain> left, right;
          left.insert(domain);
          right.insert(other->domain);
          result = compute_dominates(left, right);
        }
      }
      else
      {
        if (other->has_component_domains())
          result = compute_dominates(component_domains,   
                                     other->get_component_domains()); 
        else
        {
          std::set<Domain> other_doms;
          other_doms.insert(other->domain);
          result = compute_dominates(component_domains, other_doms);
        }
      }
      AutoLock n_lock(node_lock);
      dominators[other] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceNode::dominates(IndexPartNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(other);
        if (finder != dominators.end())
          return finder->second;
      }
      bool result;
      std::set<Domain> other_doms;
      other->get_subspace_domains(other_doms);
      if (component_domains.empty())
      {
        std::set<Domain> local;
        local.insert(domain);
        result = compute_dominates(local, other_doms);
      }
      else
        result = compute_dominates(component_domains, other_doms);
      AutoLock n_lock(node_lock);
      dominators[other] = result;
      return result;
    }

#ifdef DYNAMIC_TESTS
    //--------------------------------------------------------------------------
    void IndexSpaceNode::add_disjointness_tests(IndexPartNode *child,
                                  const std::vector<IndexSpaceNode*> &children)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::map<Color,IndexPartNode*>::const_iterator it = 
            valid_map.begin(); it != valid_map.end(); it++)
      {
        if (it->second == child)
          continue;
        it->second->add_disjointness_tests(child, children);
      }
    }
#endif

    //--------------------------------------------------------------------------
    void IndexSpaceNode::send_node(AddressSpaceID target, bool up, bool down)
    //--------------------------------------------------------------------------
    {
      // Go up first so we know those nodes will be there
      if (up && (parent != NULL))
        parent->send_node(target, true/*up*/, false/*down*/);
      // Check to see if our creation set includes the target
      std::map<Color,IndexPartNode*> valid_copy;
      {
        AutoLock n_lock(node_lock);
        if (!creation_set.is_set(target))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(domain);
            if (parent != NULL)
              rez.serialize(parent->handle);
            else
              rez.serialize(0);
            rez.serialize(color);
            rez.serialize(component_domains.size());
            for (std::set<Domain>::const_iterator it = 
                  component_domains.begin(); it != 
                  component_domains.end(); it++)
            {
              rez.serialize(*it);
            }
          }
          context->runtime->send_index_space_node(target, rez); 
          creation_set.set_bit(target);
        }
        if (!destruction_set.is_set(target))
        {
          // Now we need to send a destruction
          context->runtime->send_index_space_destruction(handle, target);
          destruction_set.set_bit(target);
        }
        // If we need to go down, make a copy of the valid children
        if (down)
          valid_copy = valid_map;
      }
      if (down)
      {
        for (std::map<Color,IndexPartNode*>::const_iterator it = 
              valid_copy.begin(); it != valid_copy.end(); it++)
        {
          it->second->send_node(target, false/*up*/, true/*down*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceNode::handle_node_creation(
        RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Domain domain;
      derez.deserialize(domain);
      IndexPartition parent;
      derez.deserialize(parent);
      Color color;
      derez.deserialize(color);
      size_t components;
      derez.deserialize(components);
      std::set<Domain> component_domains;
      for (unsigned idx = 0; idx < components; idx++)
      {
        Domain component;
        derez.deserialize(component);
        component_domains.insert(component);
      }
      IndexPartNode *parent_node = NULL;
      if (parent > 0)
      {
        parent_node = context->get_node(parent);
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_node != NULL);
#endif
      }
      IndexSpaceNode *node = context->create_node(domain, parent_node, color);
#ifdef DEBUG_HIGH_LEVEL
      assert(node != NULL);
#endif
      node->add_creation_source(source);
      if (components > 0)
        node->update_component_domains(component_domains);
    }

    //--------------------------------------------------------------------------
    IndexSpaceAllocator* IndexSpaceNode::get_allocator(void)
    //--------------------------------------------------------------------------
    {
      if (allocator == NULL)
      {
        AutoLock n_lock(node_lock);
        if (allocator == NULL)
        {
          allocator = (IndexSpaceAllocator*)malloc(sizeof(IndexSpaceAllocator));
          *allocator = handle.create_allocator();
        }
      }
      return allocator;
    }

    /////////////////////////////////////////////////////////////
    // Index Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(IndexPartition p, IndexSpaceNode *par,
                                 Color c, Domain cspace, bool dis,
                                 RegionTreeForest *ctx)
      : IndexTreeNode(c, par->depth+1, ctx), handle(p), color_space(cspace),
        parent(par), disjoint(dis), has_complete(false)
    //--------------------------------------------------------------------------
    { 
    }

    //--------------------------------------------------------------------------
    IndexPartNode::IndexPartNode(const IndexPartNode &rhs)
      : IndexTreeNode(), handle(0), color_space(Domain::NO_DOMAIN),
        parent(NULL), disjoint(false), has_complete(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexPartNode::~IndexPartNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexPartNode& IndexPartNode::operator=(const IndexPartNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* IndexPartNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::has_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return (color_map.find(c) != color_map.end());
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexPartNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/); 
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c) != color_map.end());
#endif
      return color_map[c];
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_child(IndexSpaceNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(child->color) == color_map.end());
#endif
      color_map[child->color] = child;
      valid_map[child->color] = child;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      valid_map.erase(c);
      // Mark that any completeness computations we've done are no longer valid
      has_complete = false;
    }

    //--------------------------------------------------------------------------
    size_t IndexPartNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return valid_map.size();
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::are_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      if (c1 == c2)
        return false;
      if (disjoint)
        return true;
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      if (disjoint_subspaces.find(std::pair<Color,Color>(c1,c2)) !=
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
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(c1) != color_map.end());
      assert(color_map.find(c2) != color_map.end());
#endif
      disjoint_subspaces.insert(std::pair<Color,Color>(c1,c2));
      disjoint_subspaces.insert(std::pair<Color,Color>(c2,c1));
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::is_complete(void)
    //--------------------------------------------------------------------------
    {
      // If we've cached the value then we are good to go
      if (has_complete)
        return complete;
      // Otherwise compute it 
      std::set<Domain> parent_domains, child_domains;
      bool can_cache = false;
      if (parent->has_component_domains())
        parent_domains = parent->get_component_domains();
      else
      {
        parent_domains.insert(parent->domain);
        // We can cache the result if we know the domains
        // has dimension greater than zero indicating we have
        // a structured index space
        can_cache = (parent->domain.get_dim() > 0);
      }
      for (std::map<Color,IndexSpaceNode*>::const_iterator it = 
            color_map.begin(); it != color_map.end(); it++)
      {
        if (it->second->has_component_domains())
        {
          const std::set<Domain> &child_doms = 
                                      it->second->get_component_domains();
          child_domains.insert(child_doms.begin(), child_doms.end());
        }
        else
          child_domains.insert(it->second->domain);
      }
      bool result = compute_dominates(child_domains, parent_domains);
      if (can_cache)
      {
        complete = result;
        has_complete = true;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_instance(PartitionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_nodes.find(inst) == logical_nodes.end());
#endif
      logical_nodes.insert(inst);
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::has_instance(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<PartitionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle.get_tree_id() == tid)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::add_creation_source(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      creation_set.set_bit(source);
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->remove_child(color);
      AutoLock n_lock(node_lock);
      destruction_set.set_bit(source);
      for (std::set<PartitionNode*>::const_iterator it = logical_nodes.begin();
             it != logical_nodes.end(); it++)
      {
        (*it)->destroy_node(source);
      }
    }

    //--------------------------------------------------------------------------
    void IndexPartNode::get_subspace_domains(std::set<Domain> &subspaces)
    //--------------------------------------------------------------------------
    {
      for (std::map<Color,IndexSpaceNode*>::const_iterator it = 
            color_map.begin(); it != color_map.end(); it++)
      {
        if (it->second->has_component_domains())
        {
          const std::set<Domain> &components = 
                                            it->second->get_component_domains();
          subspaces.insert(components.begin(), components.end());
        }
        else
          subspaces.insert(it->second->domain);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::intersects_with(IndexSpaceNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.has_intersects;
      }
      std::set<Domain> local_domains, intersect;
      bool result, own_results;
      get_subspace_domains(local_domains);
      if (other->has_component_domains())
      {
        result = compute_intersections(local_domains, 
                                     other->get_component_domains(), intersect);
#ifndef SHARED_LOWLEVEL
        own_results = false;
#else
        own_results = true;
#endif
      }
      else
      {
        result = compute_intersections(local_domains, other->domain, intersect);
#ifndef SHARED_LOWLEVEL
        own_results = (other->domain.get_dim() == 0);
#else
        own_results = true;
#endif
      }
      AutoLock n_lock(node_lock);
      if (result)
        intersections[other] = IntersectInfo(true/*has intersects*/,
                                             own_results, intersect);
      else
        intersections[other] = IntersectInfo(false/*own intersects*/,
                                             false/*own*/);
      return result;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::intersects_with(IndexPartNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.has_intersects;
      }
      std::set<Domain> local_domains, other_domains, intersect;
      get_subspace_domains(local_domains);
      other->get_subspace_domains(other_domains);
      bool result = compute_intersections(local_domains, other_domains, 
                                          intersect);
#ifndef SHARED_LOWLEVEL
      bool own_results = (local_domains.begin()->get_dim() == 0);
#else
      bool own_results = true;
#endif
      AutoLock n_lock(node_lock);
      if (result)
        intersections[other] = IntersectInfo(true/*has intersects*/,
                                             own_results, intersect);
      else
        intersections[other] = IntersectInfo(false/*own intersects*/,
                                             false/*own*/);
      return result;
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& IndexPartNode::get_intersection_domains(
                                                          IndexSpaceNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.intersections;
      }
      std::set<Domain> local_domains, intersect;
      bool result, own_results;
      get_subspace_domains(local_domains);
      if (other->has_component_domains())
      {
        result = compute_intersections(local_domains, 
                                     other->get_component_domains(), intersect);
#ifndef SHARED_LOWLEVEL
        own_results = false;
#else
        own_results = true;
#endif
      }
      else
      {
        result = compute_intersections(local_domains, other->domain, intersect);
#ifndef SHARED_LOWLEVEL
        own_results = (other->domain.get_dim() == 0);
#else
        own_results = true;
#endif
      }
      AutoLock n_lock(node_lock);
      if (result)
        intersections[other] = IntersectInfo(true/*has intersects*/,
                                             own_results, intersect);
      else
        intersections[other] = IntersectInfo(false/*own intersects*/,
                                             false/*own*/);
      return intersections[other].intersections;
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& IndexPartNode::get_intersection_domains(
                                                           IndexPartNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,IntersectInfo>::const_iterator finder = 
          intersections.find(other);
        if (finder != intersections.end())
          return finder->second.intersections;
      }
      std::set<Domain> local_domains, other_domains, intersect;
      get_subspace_domains(local_domains);
      other->get_subspace_domains(other_domains);
      bool result = compute_intersections(local_domains, other_domains, 
                                          intersect);
#ifndef SHARED_LOWLEVEL
      bool own_results = (local_domains.begin()->get_dim() == 0);
#else
      bool own_results = true;
#endif
      AutoLock n_lock(node_lock);
      if (result)
        intersections[other] = IntersectInfo(true/*has intersects*/,
                                             own_results, intersect);
      else
        intersections[other] = IntersectInfo(false/*own intersects*/,
                                             false/*own*/);
      return intersections[other].intersections;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::dominates(IndexSpaceNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(other);
        if (finder != dominators.end())
          return finder->second;
      }
      std::set<Domain> local;
      get_subspace_domains(local);
      bool result;
      if (other->has_component_domains())
        result = compute_dominates(local, other->get_component_domains()); 
      else
      {
        std::set<Domain> other_doms;
        other_doms.insert(other->domain);
        result = compute_dominates(local, other_doms);
      }
      AutoLock n_lock(node_lock);
      dominators[other] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    bool IndexPartNode::dominates(IndexPartNode *other)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<IndexTreeNode*,bool>::const_iterator finder = 
          dominators.find(other);
        if (finder != dominators.end())
          return finder->second;
      }
      std::set<Domain> local, other_doms;
      get_subspace_domains(local);
      other->get_subspace_domains(other_doms);
      bool result = compute_dominates(local, other_doms);
      AutoLock n_lock(node_lock);
      dominators[other] = result;
      return result;
    }

#ifdef DYNAMIC_TESTS
    //--------------------------------------------------------------------------
    void IndexPartNode::add_disjointness_tests(IndexPartNode *child,
                                  const std::vector<IndexSpaceNode*> &children)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest::DynamicPartTest test(parent, child, this);
      for (std::vector<IndexSpaceNode*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        test.add_child_space(true/*left*/, *it);
      }
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        for (std::map<Color,IndexSpaceNode*>::const_iterator it =
              valid_map.begin(); it != valid_map.end(); it++)
        {
          test.add_child_space(false/*left*/, it->second); 
        }
      }
      context->add_disjointness_test(test);
    }
#endif

    //--------------------------------------------------------------------------
    void IndexPartNode::send_node(AddressSpaceID target, bool up, bool down)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      if (up)
        parent->send_node(target, true/*up*/, false/*down*/);
      std::map<Color,IndexSpaceNode*> valid_copy;
      {
        AutoLock n_lock(node_lock);
        if (!creation_set.is_set(target))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(color_space);
            rez.serialize(parent->handle); 
            rez.serialize(color);
            rez.serialize(disjoint);
          }
          context->runtime->send_index_partition_node(target, rez);
          creation_set.set_bit(target);
        }
        if (!destruction_set.is_set(target))
        {
          // Send the deletion notification
          context->runtime->send_index_partition_destruction(handle, target);
          destruction_set.set_bit(target);
        }
        if (down)
          valid_copy = valid_map;
      }
      if (down)
      {
        for (std::map<Color,IndexSpaceNode*>::const_iterator it = 
              valid_copy.begin(); it != valid_copy.end(); it++)
        {
          it->second->send_node(target, false/*up*/, true/*down*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartNode::handle_node_creation(
        RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      Domain color_space;
      derez.deserialize(color_space);
      IndexSpace parent;
      derez.deserialize(parent);
      Color color;
      derez.deserialize(color);
      bool disjoint;
      derez.deserialize(disjoint);
      IndexSpaceNode *parent_node = context->get_node(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_node != NULL);
#endif
      IndexPartNode *node = context->create_node(handle, parent_node, color,
                                color_space, disjoint);
#ifdef DEBUG_HIGH_LEVEL
      assert(node != NULL);
#endif
      node->add_creation_source(source);
    } 

    /////////////////////////////////////////////////////////////
    // Field Space Node 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(FieldSpace sp, RegionTreeForest *ctx)
      : handle(sp), context(ctx)
    //--------------------------------------------------------------------------
    {
      this->node_lock = Reservation::create_reservation();
      this->allocated_indexes = FieldMask();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(const FieldSpaceNode &rhs)
      : handle(FieldSpace::NO_SPACE), context(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::~FieldSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      node_lock.destroy_reservation();
      node_lock = Reservation::NO_RESERVATION;
      for (std::map<FIELD_TYPE,LegionContainer<LayoutDescription*,
            LAYOUT_DESCRIPTION_ALLOC>::deque>::iterator it =
            layouts.begin(); it != layouts.end(); it++)
      {
        LegionContainer<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::deque 
          &descs = it->second;
        for (unsigned idx = 0; idx < descs.size(); idx++)
          delete descs[idx];
      }
      layouts.clear();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode& FieldSpaceNode::operator=(const FieldSpaceNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::allocate_field(FieldID fid, size_t size, bool local)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.find(fid) == fields.end());
#endif
      // Find an index in which to allocate this field  
      unsigned index = allocate_index(local);
#ifdef DEBUG_HIGH_LEVEL
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        assert(it->second.destroyed || (it->second.idx != index));
      }
#endif
      fields[fid] = FieldInfo(size, index, local);
      // Send messages to all our subscribers telling them about the allocation
      // as long as it is not local.  Local fields get sent by the task contexts
      if (!local && !!creation_set)
      {
        for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
        {
          if (!creation_set.is_set(idx))
            continue;
          context->runtime->send_field_allocation(handle, fid, size,
                                                  index, idx);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::allocate_field_index(FieldID fid, size_t size,
                                          AddressSpaceID source, unsigned index)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // First check to see if we have already allocated this field.
      // If not do our own allocation
      unsigned our_index;
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
      if (finder == fields.end())
      {
        // Try to allocate in the same place
        our_index = allocate_index(false/*local*/, index);
#ifdef DEBUG_HIGH_LEVEL
        for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
              it != fields.end(); it++)
        {
          assert(it->second.destroyed || (it->second.idx != index));
        }
#endif
        fields[fid] = FieldInfo(size, our_index, false/*local*/);
        // If we haven't done the allocation already send updates to
        // all our subscribers telling them where we allocated the field
        // Note this includes sending it back to the source which sent
        // us the allocation in the first place
        if (!!creation_set)
        {
          for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
          {
            if (!creation_set.is_set(idx))
              continue;
            context->runtime->send_field_allocation(handle, fid, size,
                                                    our_index, idx);
          }
        }
      }
      else
      {
        our_index = finder->second.idx;
      }
      // Update our permutation transformer. Note we do this
      // no matter what and let the permutation transformer
      // keep track of whether or not it is an identity or not.
      std::map<AddressSpaceID,FieldPermutation>::iterator 
        trans_it = transformers.find(source);
      // Create the transformer if we need to
      if (trans_it == transformers.end())
      {
        transformers[source] = FieldPermutation();
        trans_it = transformers.find(source);
#ifdef DEBUG_HIGH_LEVEL
        assert(trans_it != transformers.end());
#endif
      }
      trans_it->second.send_to(index, our_index);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field(FieldID fid, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      std::map<FieldID,FieldInfo>::iterator finder = fields.find(fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != fields.end());
#endif
      // If we already destroyed the field then we are done
      if (finder->second.destroyed)
        return;
      // Tell all our subscribers that we've destroyed the field
      if (!!creation_set)
      {
        for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
        {
          if (idx == source)
            continue;
          if (!creation_set.is_set(idx))
            continue;
          context->runtime->send_field_destruction(handle, fid, idx);
        }
      }
      // Free the index
      free_index(finder->second.idx);
      // Mark the field destroyed
      finder->second.destroyed = true;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
      if (finder == fields.end())
        return false;
      // Make sure we haven't destroyed this field
      return (!finder->second.destroyed);
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::get_field_size(FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != fields.end());
#endif
      return finder->second.field_size;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_all_fields(std::set<FieldID> &to_set)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (!it->second.destroyed)
          to_set.insert(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_all_regions(std::set<LogicalRegion> &regions)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if (!(*it)->destruction_set)
          regions.insert((*it)->handle);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::add_instance(RegionNode *inst)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_nodes.find(inst) == logical_nodes.end());
#endif
      logical_nodes.insert(inst);
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_instance(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        if ((*it)->handle.get_tree_id() == tid)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::add_creation_source(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      creation_set.set_bit(source);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      destruction_set.set_bit(source);
      for (std::set<RegionNode*>::const_iterator it = logical_nodes.begin();
            it != logical_nodes.end(); it++)
      {
        (*it)->destroy_node(source);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::transform_field_mask(FieldMask &mask, 
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Need an exclusive lock since we might change the state
      // of the transformer.
      AutoLock n_lock(node_lock);
      std::map<AddressSpaceID,FieldPermutation>::iterator finder = 
        transformers.find(source);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != transformers.end());
#endif
      finder->second.permute(mask);
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(
                              const std::set<FieldID> &privilege_fields) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      FieldMask result;
      for (std::set<FieldID>::const_iterator it = privilege_fields.begin();
            it != privilege_fields.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        result.set_bit(finder->second.idx);
      }
#ifdef DEBUG_HIGH_LEVEL
      // Have a little bit of code for logging bit masks when requested
      if (Runtime::bit_mask_logging)
      {
        char *bit_string = result.to_string();
        fprintf(stderr,"%s\n",bit_string);
        free(bit_string);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned FieldSpaceNode::get_field_index(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != fields.end());
#endif
      return finder->second.idx;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_indexes(const std::set<FieldID> &needed,
                                     std::map<unsigned,FieldID> &indexes) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (std::set<FieldID>::const_iterator it = needed.begin();
            it != needed.end(); it++)
      {
        std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != fields.end());
#endif
        indexes[finder->second.idx] = *it;
      }
    }

    //--------------------------------------------------------------------------
    InstanceManager* FieldSpaceNode::create_instance(Memory location,
                                                     Domain domain,
                                       const std::set<FieldID> &create_fields,
                                                     size_t blocking_factor,
                                                     unsigned depth,
                                                     RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this->context, CREATE_INSTANCE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(!create_fields.empty());
#endif
      InstanceManager *result = NULL;
      if (create_fields.size() == 1)
      {
        FieldID fid = *create_fields.begin();
        size_t field_size;
        unsigned field_index;
        {
          // Need to hold the field lock when accessing field infos
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != fields.end());
#endif
          field_size = finder->second.field_size;
          field_index = finder->second.idx;
        }
        // First see if we can recycle a physical instance
        Event use_event = Event::NO_EVENT;
        PhysicalInstance inst = context->runtime->find_physical_instance(
                          location, field_size, domain, depth, use_event);
        // If we couldn't recycle one, then try making one
        if (!inst.exists())
          inst = domain.create_instance(location, field_size);
        if (inst.exists())
        {
          FieldMask inst_mask = get_field_mask(create_fields);
          // See if we can find a layout description object
          LayoutDescription *layout = 
            find_layout_description(inst_mask, domain, blocking_factor);
          if (layout == NULL)
          {
            // Now we need to make a layout
            std::vector<size_t> field_sizes(1);
            std::vector<unsigned> indexes(1);
            field_sizes[0] = field_size;
            indexes[0] = field_index;
            layout = create_layout_description(inst_mask, domain,
                                               blocking_factor, 
                                               create_fields,
                                               field_sizes,
                                               indexes);
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(layout != NULL);
#endif
          DistributedID did = context->runtime->get_available_distributed_id();
          result = legion_new<InstanceManager>(context, did, 
                                       context->runtime->address_space,
                                       context->runtime->address_space,
                                       location, inst, node, layout, 
                                       use_event, depth);
#ifdef DEBUG_HIGH_LEVEL
          assert(result != NULL);
#endif
#ifdef LEGION_PROF
          if (!use_event.exists())
          {
            std::map<FieldID,size_t> inst_fields;
            inst_fields[fid] = field_size;
            LegionProf::register_instance_creation(inst.id,
                location.id, 0/*redop*/, blocking_factor,
                inst_fields);
          }
#endif
        }
      }
      else
      {
        std::vector<size_t> field_sizes(create_fields.size());
        std::vector<unsigned> indexes(create_fields.size());
        // Figure out the size of each element
        {
          // Need to hold the lock when accessing field infos
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          unsigned idx = 0;
          for (std::set<FieldID>::const_iterator it = 
                create_fields.begin(); it != create_fields.end(); it++,idx++)
          {
            std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != fields.end());
#endif
            field_sizes[idx] = finder->second.field_size;
            indexes[idx] = finder->second.idx;
          }
        }
        // First see if we can recycle a physical instance
        Event use_event = Event::NO_EVENT;
        PhysicalInstance inst = context->runtime->find_physical_instance(
            location, field_sizes, domain, blocking_factor, depth, use_event);
        // If that didn't work, try making one
        if (!inst.exists())
          inst = domain.create_instance(location, field_sizes, blocking_factor);
        if (inst.exists())
        {
          FieldMask inst_mask = get_field_mask(create_fields);
          LayoutDescription *layout = 
            find_layout_description(inst_mask, domain, blocking_factor);
          if (layout == NULL)
          {
            // We couldn't find one so make one
            layout = create_layout_description(inst_mask, domain,
                                               blocking_factor,
                                               create_fields,
                                               field_sizes,
                                               indexes);
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(layout != NULL);
#endif
          DistributedID did = context->runtime->get_available_distributed_id();
          result = legion_new<InstanceManager>(context, did,
                                       context->runtime->address_space,
                                       context->runtime->address_space,
                                       location, inst, node, layout, 
                                       use_event, depth);
#ifdef DEBUG_HIGH_LEVEL
          assert(result != NULL);
#endif
#ifdef LEGION_PROF
          if (!use_event.exists())
          {
            std::map<unsigned,size_t> inst_fields;
            for (std::set<FieldID>::const_iterator it = 
                  create_fields.begin(); it != create_fields.end(); it++)
            {
              std::map<FieldID,FieldInfo>::const_iterator finder = 
                fields.find(*it);
#ifdef DEBUG_HIGH_LEVEL
              assert(finder != fields.end());
#endif
              inst_fields[*it] = finder->second.field_size;
            }
            LegionProf::register_instance_creation(inst.id, location.id,
                                0/*redop*/, blocking_factor, inst_fields);
          }
#endif
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionManager* FieldSpaceNode::create_reduction(Memory location,
                                                       Domain domain,
                                                       FieldID fid,
                                                       bool reduction_list,
                                                       RegionNode *node,
                                                       ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this->context, CREATE_REDUCTION_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(redop > 0);
#endif
      ReductionManager *result = NULL;
      // Find the reduction operation for this instance
      const ReductionOp *op = Runtime::get_reduction_op(redop);
      std::map<FieldID,FieldInfo>::const_iterator finder = fields.find(fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != fields.end());
#endif
      if (reduction_list)
      {
        // We need a new index space for handling the sparse reductions
        // TODO: allow users to specify the max number of reductions.  Otherwise
        // for right now we'll just over approximate with the number of elements
        // in the handle index space since ideally reduction lists are sparse
        // and will have less than one reduction per point.
        Domain ptr_space = Domain(IndexSpace::create_index_space(
            node->handle.get_index_space().get_valid_mask().get_num_elmts()));
        std::vector<size_t> element_sizes;
        element_sizes.push_back(sizeof(ptr_t)); // pointer types
        element_sizes.push_back(op->sizeof_rhs);
        // Don't give the reduction op here since this is a list instance and we
        // don't want to initialize any of the fields
        PhysicalInstance inst = 
          ptr_space.create_instance(location, element_sizes, 1/*true list*/);
        if (inst.exists())
        {
          DistributedID did = context->runtime->get_available_distributed_id();
          result = legion_new<ListReductionManager>(context, did,
                                            context->runtime->address_space,
                                            context->runtime->address_space, 
                                            location, inst, node, 
                                            redop, op, ptr_space);
#ifdef DEBUG_HIGH_LEVEL
          assert(result != NULL);
#endif
#ifdef LEGION_PROF
          {
            std::map<FieldID,size_t> inst_fields;
            inst_fields[fid] = op->sizeof_rhs;
            LegionProf::register_instance_creation(inst.id, location.id,
                redop, 1/*blocking factor*/, inst_fields);
          }
#endif
        }
      }
      else
      {
        // Ease case of making a foldable reduction
        PhysicalInstance inst = domain.create_instance(location, op->sizeof_rhs,
                                                        redop);
        if (inst.exists())
        {
          DistributedID did = context->runtime->get_available_distributed_id();
          result = legion_new<FoldReductionManager>(context, did,
                                            context->runtime->address_space,
                                            context->runtime->address_space, 
                                            location, inst, node, redop, op);
#ifdef DEBUG_HIGH_LEVEL
          assert(result != NULL);
#endif
#ifdef LEGION_PROF
          {
            std::map<FieldID,size_t> inst_fields;
            inst_fields[fid] = op->sizeof_rhs;
            LegionProf::register_instance_creation(inst.id, location.id,
                redop, 0/*blocking factor*/, inst_fields);
          }
#endif
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::find_layout_description(
        const FieldMask &mask, const Domain &domain, size_t blocking_factor)
    //--------------------------------------------------------------------------
    {
      uint64_t hash_key = mask.get_hash_key();
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      std::map<FIELD_TYPE,LegionContainer<LayoutDescription*,
        LAYOUT_DESCRIPTION_ALLOC>::deque>::const_iterator finder = 
                                                    layouts.find(hash_key);
      if (finder == layouts.end())
        return NULL;
      // First go through the existing descriptions and see if we find
      // one that matches the existing layout
      for (std::deque<LayoutDescription*>::const_iterator it = 
            finder->second.begin(); it != finder->second.end(); it++)
      {
        if ((*it)->match_layout(mask, domain, blocking_factor))
          return (*it);
      }
      return NULL;
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::create_layout_description(
        const FieldMask &mask, const Domain &domain, size_t blocking_factor,
                                     const std::set<FieldID> &create_fields,
                                     const std::vector<size_t> &field_sizes, 
                                     const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
      // Make the new field description and then register it
      LayoutDescription *result = new LayoutDescription(mask, domain,
                                                        blocking_factor, this);
      unsigned idx = 0;
      size_t accum_offset = 0;
      for (std::set<FieldID>::const_iterator it = create_fields.begin();
            it != create_fields.end(); it++, idx++)
      {
        result->add_field_info((*it), indexes[idx],
                               accum_offset, field_sizes[idx]);
        accum_offset += field_sizes[idx];
      }
      // Now we can register it
      return register_layout_description(result);
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::register_layout_description(
                                                      LayoutDescription *layout)
    //--------------------------------------------------------------------------
    {
      uint64_t hash_key = layout->allocated_fields.get_hash_key();
      AutoLock n_lock(node_lock);
      LegionContainer<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::deque
        &descs = layouts[hash_key];
      if (!descs.empty())
      {
        for (LegionContainer<LayoutDescription*,LAYOUT_DESCRIPTION_ALLOC>::
              deque::const_iterator it = descs.begin(); it != descs.end(); it++)
        {
          if (layout->match_layout(*it))
          {
            // Delete the layout we are trying to register
            // and return the matching one
            delete layout;
            return (*it);
          }
        }
      }
      // Otherwise we successfully registered it
      descs.push_back(layout);
      return layout;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // See if this is in our creation set, if not, send it and all the fields
      AutoLock n_lock(node_lock);
      if (!creation_set.is_set(target))
      {
        // First send the node info and then send all the fields
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
        }
        context->runtime->send_field_space_node(target, rez);
        // Send all the field allocations
        for (std::map<FieldID,FieldInfo>::const_iterator it = 
              fields.begin(); it != fields.end(); it++)
        {
          // No need to send it if it has been destroyed
          if (!it->second.destroyed)
          {
            context->runtime->send_field_allocation(handle, it->first,
                it->second.field_size, it->second.idx, target);
          }
        }
        // Finally add it to the creation set
        creation_set.set_bit(target);
      }
      // Send any deletions if necessary
      if (!destruction_set.is_set(target))
      {
        context->runtime->send_field_space_destruction(handle, target);
        destruction_set.set_bit(target);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNode::handle_node_creation(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldSpaceNode *node = context->create_node(handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(node != NULL);
#endif
      node->add_creation_source(source);
    }

    //--------------------------------------------------------------------------
    char* FieldSpaceNode::to_string(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(!!mask);
#endif
      char *result = (char*)malloc(MAX_FIELDS*4); 
      bool first = true;
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (mask.is_set(it->second.idx))
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
    void FieldSpaceNode::to_field_set(const FieldMask &mask,
                                      std::set<FieldID> &field_set) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(!!mask);
#endif
      for (std::map<FieldID,FieldInfo>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (mask.is_set(it->second.idx))
        {
          field_set.insert(it->first);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(!field_set.empty()); // we should have found something
#endif
    }

    //--------------------------------------------------------------------------
    unsigned FieldSpaceNode::allocate_index(bool local, int goal /*= -1*/)
    //--------------------------------------------------------------------------
    {
      // Assume we are already holding the node lock 

      // We do something intelligent here to try and maintain
      // identity permutations as long as possible.  First, if there
      // is a target goal try and allocate it there since it has already
      // been assigned there on a remote node.
      if ((goal >= 0) && !allocated_indexes.is_set(goal))
      {
        unsigned result = goal;
        allocated_indexes.set_bit(result);
        return result;
      }
      // Otherwise, try picking out an index along the stripe corresponding
      // to our runtime instance.
      unsigned tests = 0;
      unsigned offset = context->runtime->address_space;
      for (unsigned idx = offset;
            idx < MAX_FIELDS; idx += context->runtime->runtime_stride)
      {
        if (!allocated_indexes.is_set(idx))
        {
          allocated_indexes.set_bit(idx);
          return idx;
        }
        tests++;
      }
      offset++;
      if (offset == context->runtime->runtime_stride)
        offset = 0;
      // If our strip is full, go onto the next runtime from ours
      // continue doing this until we have tested all the points.
      // Walk these points backwards to avoid conflicts with remote
      // nodes doing their own allocation on their stipes.
      while (tests < MAX_FIELDS)
      {
        int target = MAX_FIELDS-1;
        // Find our stripe
        while (offset != unsigned(target%context->runtime->runtime_stride))
          target--;
        // Now we're on our stripe, so start searching
        while (target >= 0)
        {
          if (!allocated_indexes.is_set(target))
          {
            unsigned result = target;
            allocated_indexes.set_bit(result);
            return result;
          }
          target -= context->runtime->runtime_stride;
          tests++;
        }
        // Didn't find anything on this stripe, go to the next one
        offset++;
        if (offset == context->runtime->runtime_stride)
          offset = 0;
      }
      // If we make it here, the mask is full and we are out of allocations
      log_field(LEVEL_ERROR,"Exceeded maximum number of allocated fields for "
                            "field space %x.  Change MAX_FIELDS from %d and "
                            "related macros at the top of legion_types.h and "
                            "recompile.", handle.id, MAX_FIELDS);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_MAX_FIELD_OVERFLOW);
      return 0;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_index(unsigned index)
    //--------------------------------------------------------------------------
    {
      // Assume we are already holding the node lock
      allocated_indexes.unset_bit(index);
    }

    /////////////////////////////////////////////////////////////
    // Users and Info 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TreeCloseImpl::TreeCloseImpl(int c, const FieldMask &m, bool o)
      : Collectable(), target_child(c), leave_open(o), 
        remaining_logical(m), remaining_physical(m),
        tree_reservation(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TreeCloseImpl::TreeCloseImpl(const TreeCloseImpl &rhs)
      : Collectable(), target_child(-1), leave_open(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TreeCloseImpl::~TreeCloseImpl(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(tree_reservation.exists());
#endif
      tree_reservation.destroy_reservation();
      tree_reservation = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    TreeCloseImpl& TreeCloseImpl::operator=(const TreeCloseImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TreeCloseImpl::add_close_op(const FieldMask &close_mask,
                                     std::deque<CloseInfo> &needed_ops)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(tree_reservation);
      FieldMask overlap = close_mask & remaining_physical;
      if (!!overlap)
      {
        needed_ops.push_back(CloseInfo(overlap, this));
        remaining_physical -= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void TreeCloseImpl::return_close_op(const FieldMask &close_mask)
    //--------------------------------------------------------------------------
    {
      // Add the fields back into the set that need to be closed
      AutoLock t_lock(tree_reservation);
      remaining_physical |= close_mask;
    }

    //--------------------------------------------------------------------------
    TreeClose::TreeClose(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TreeClose::TreeClose(TreeCloseImpl *op)
      : impl(op)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    TreeClose::TreeClose(const TreeClose &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    TreeClose::~TreeClose(void)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && impl->remove_reference())
        legion_delete(impl);
      impl = NULL;
    }

    //--------------------------------------------------------------------------
    TreeClose& TreeClose::operator=(const TreeClose &rhs)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && impl->remove_reference())
        legion_delete(impl);
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    void TreeClose::add_close_op(const FieldMask &close_mask,
                                 std::deque<CloseInfo> &needed_ops)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_close_op(close_mask, needed_ops);
    }

    //--------------------------------------------------------------------------
    void TreeClose::return_close_op(const FieldMask &close_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->return_close_op(close_mask);
    }

    /////////////////////////////////////////////////////////////
    // Users and Info 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(void)
      : GenericUser(), op(NULL), idx(0), gen(0), timeout(TIMEOUT)
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
        , uid(0)
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(Operation *o, unsigned id, const RegionUsage &u,
                             const FieldMask &m)
      : GenericUser(u, m), op(o), idx(id), 
        gen(o->get_generation()), timeout(TIMEOUT)
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
        , uid(o->get_unique_op_id())
#endif
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(void)
      : GenericUser(), term_event(Event::NO_EVENT)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const RegionUsage &u, const FieldMask &m,
                               Event term, int c /*= -1*/)
      : GenericUser(u, m), term_event(term), child(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappableInfo::MappableInfo(ContextID c, Mappable *m, Processor p,
                               RegionRequirement &r, const FieldMask &k)
      : ctx(c), mappable(m), local_proc(p), req(r), traversal_mask(k)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // PathTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PathTraverser::PathTraverser(RegionTreePath &p)
      : path(p)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PathTraverser::PathTraverser(const PathTraverser &rhs)
      : path(rhs.path)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PathTraverser::~PathTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PathTraverser& PathTraverser::operator=(const PathTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PathTraverser::traverse(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      // Continue visiting nodes and then finding their children
      // until we have traversed the entire path.
      while (true)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(node != NULL);
#endif
        depth = node->get_depth();
        has_child = path.has_child(depth);
        if (has_child)
          next_child = path.get_child(depth);
        bool continue_traversal = node->visit_node(this);
        if (!continue_traversal)
          return false;
        if (!has_child)
          break;
        node = node->get_tree_child(next_child);
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // LogicalPathRegistrar
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::LogicalPathRegistrar(ContextID c, Operation *o,
                                       const FieldMask &m, RegionTreePath &p)
      : PathTraverser(p), ctx(c), field_mask(m), op(o)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::LogicalPathRegistrar(const LogicalPathRegistrar&rhs)
      : PathTraverser(rhs.path), ctx(0), field_mask(FieldMask()), op(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::~LogicalPathRegistrar(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar& LogicalPathRegistrar::operator=(
                                                const LogicalPathRegistrar &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool LogicalPathRegistrar::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences<false>(ctx, op, field_mask);
      if (!has_child)
      {
        // If we're at the bottom, fan out and do all the children
        LogicalRegistrar<false> registrar(ctx, op, field_mask);
        return node->visit_node(&registrar);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalPathRegistrar::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences<false>(ctx, op, field_mask);
      if (!has_child)
      {
        // If we're at the bottom, fan out and do all the children
        LogicalRegistrar<false> registrar(ctx, op, field_mask);
        return node->visit_node(&registrar);
      }
      return true;
    }


    /////////////////////////////////////////////////////////////
    // LogicalRegistrar
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    LogicalRegistrar<DOMINATE>::LogicalRegistrar(ContextID c, Operation *o,
                                                 const FieldMask &m)
      : ctx(c), field_mask(m), op(o)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    LogicalRegistrar<DOMINATE>::LogicalRegistrar(const LogicalRegistrar &rhs)
      : ctx(0), field_mask(FieldMask()), op(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    LogicalRegistrar<DOMINATE>::~LogicalRegistrar(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    LogicalRegistrar<DOMINATE>& LogicalRegistrar<DOMINATE>::operator=(
                                                    const LogicalRegistrar &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    bool LogicalRegistrar<DOMINATE>::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    bool LogicalRegistrar<DOMINATE>::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences<DOMINATE>(ctx, op, field_mask);
      return true;
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    bool LogicalRegistrar<DOMINATE>::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences<DOMINATE>(ctx, op, field_mask);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // LogicalInitializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalInitializer::LogicalInitializer(ContextID c)
      : ctx(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalInitializer::LogicalInitializer(const LogicalInitializer &rhs)
      : ctx(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalInitializer::~LogicalInitializer(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalInitializer& LogicalInitializer::operator=(
                                                  const LogicalInitializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool LogicalInitializer::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool LogicalInitializer::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_logical_state(ctx); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalInitializer::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_logical_state(ctx);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // LogicalInvalidator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalInvalidator::LogicalInvalidator(ContextID c)
      : ctx(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalInvalidator::LogicalInvalidator(const LogicalInvalidator &rhs)
      : ctx(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalInvalidator::~LogicalInvalidator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalInvalidator& LogicalInvalidator::operator=(
                                                  const LogicalInvalidator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool LogicalInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool LogicalInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_logical_state(ctx); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_logical_state(ctx);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // RestrictedTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RestrictedTraverser::RestrictedTraverser(ContextID c, RegionTreePath &path)
      : PathTraverser(path), ctx(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RestrictedTraverser::RestrictedTraverser(const RestrictedTraverser &rhs)
      : PathTraverser(rhs.path), ctx(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RestrictedTraverser::~RestrictedTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RestrictedTraverser& RestrictedTraverser::operator=(
                                                 const RestrictedTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool RestrictedTraverser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->record_user_coherence(ctx, coherence_mask);
      return true;
    }

    //--------------------------------------------------------------------------
    bool RestrictedTraverser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->record_user_coherence(ctx, coherence_mask);
      return true;
    }

    //--------------------------------------------------------------------------
    const FieldMask& RestrictedTraverser::get_coherence_mask(void) const
    //--------------------------------------------------------------------------
    {
      return coherence_mask;
    }

    /////////////////////////////////////////////////////////////
    // PhysicalInitializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalInitializer::PhysicalInitializer(ContextID c)
      : ctx(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalInitializer::PhysicalInitializer(const PhysicalInitializer &rhs)
      : ctx(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalInitializer::~PhysicalInitializer(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalInitializer& PhysicalInitializer::operator=(
                                                const PhysicalInitializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInitializer::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInitializer::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_physical_state(ctx);
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInitializer::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_physical_state(ctx);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // PhysicalInvalidator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalInvalidator::PhysicalInvalidator(ContextID c)
      : ctx(c), total(true), force(true), 
        invalid_mask(FieldMask(FIELD_ALL_ONES))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalInvalidator::PhysicalInvalidator(ContextID c, const FieldMask &m,
                                             const bool f)
      : ctx(c), total(false), force(f), invalid_mask(m)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalInvalidator::PhysicalInvalidator(const PhysicalInvalidator &rhs)
      : ctx(0), total(false), force(false), invalid_mask(FieldMask())
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalInvalidator::~PhysicalInvalidator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalInvalidator& PhysicalInvalidator::operator=(
                                                const PhysicalInvalidator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      if (total)
        node->invalidate_physical_state(ctx);
      else
        node->invalidate_physical_state(ctx, invalid_mask, force);
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      if (total)
        node->invalidate_physical_state(ctx);
      else
        node->invalidate_physical_state(ctx, invalid_mask, force);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // ReductionCloser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionCloser::ReductionCloser(ContextID c, ReductionView *t,
                                     const FieldMask &m, Processor local)
      : ctx(c), target(t), close_mask(m), local_proc(local)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReductionCloser::ReductionCloser(const ReductionCloser &rhs)
      : ctx(0), target(NULL), close_mask(FieldMask()), 
        local_proc(Processor::NO_PROC)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReductionCloser::~ReductionCloser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReductionCloser& ReductionCloser::operator=(const ReductionCloser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ReductionCloser::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    bool ReductionCloser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      std::map<ReductionView*,FieldMask> valid_reductions;
      {
        PhysicalState &state = 
          node->acquire_physical_state(ctx, true/*exclusive*/);
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              state.reduction_views.begin(); it != 
              state.reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & close_mask;
          if (!!overlap)
          {
            valid_reductions[it->first] = overlap;
            it->first->add_valid_reference();
          }
        }
        if (!valid_reductions.empty())
          node->invalidate_reduction_views(state, close_mask);
        node->release_physical_state(state);
      }
      if (!valid_reductions.empty())
      {
        node->issue_update_reductions(target, close_mask, 
                                      local_proc, valid_reductions);
        // Remove our valid references
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              valid_reductions.begin(); it != valid_reductions.end(); it++)
        {
          if (it->first->remove_valid_reference())
            legion_delete(it->first);
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool ReductionCloser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      std::map<ReductionView*,FieldMask> valid_reductions;
      {
        PhysicalState &state = 
          node->acquire_physical_state(ctx, true/*exclusive*/);
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              state.reduction_views.begin(); it != 
              state.reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & close_mask;
          if (!!overlap)
          {
            valid_reductions[it->first] = overlap;
            it->first->add_valid_reference();
          }
        }
        if (!valid_reductions.empty())
          node->invalidate_reduction_views(state, close_mask);
        node->release_physical_state(state);
      }
      if (!valid_reductions.empty())
      {
        node->issue_update_reductions(target, close_mask, 
                                      local_proc, valid_reductions);
        // Remove our valid references
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              valid_reductions.begin(); it != valid_reductions.end(); it++)
        {
          if (it->first->remove_valid_reference())
            legion_delete(it->first);
        }
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // PremapTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PremapTraverser::PremapTraverser(RegionTreePath &p, MappableInfo *i)
      : PathTraverser(p), info(i)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PremapTraverser::PremapTraverser(const PremapTraverser &rhs)
      : PathTraverser(rhs.path), info(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PremapTraverser::~PremapTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PremapTraverser& PremapTraverser::operator=(const PremapTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PremapTraverser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      return perform_close_operations(node, node->handle);
    }

    //--------------------------------------------------------------------------
    bool PremapTraverser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      return perform_close_operations(node, node->parent->handle);
    }

    //--------------------------------------------------------------------------
    bool PremapTraverser::perform_close_operations(RegionTreeNode *node, 
                                                   LogicalRegion closing_handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, PERFORM_PREMAP_CLOSE_CALL);
#endif
      // Check to see if we have any close operations to perform
      std::deque<CloseInfo> close_ops;
      path.get_close_operations(depth, close_ops);
      if (!close_ops.empty())
      {
        PhysicalCloser closer(info, false/*leave open*/, closing_handle);  
        PhysicalState &state = 
          node->acquire_physical_state(info->ctx, true/*exclusive*/);
        // Once we decide to make a composite instance for any of them
        // then we will make a composite instance for all of them
        bool create_composite = false;
        for (std::deque<CloseInfo>::iterator it = close_ops.begin();
              it != close_ops.end(); it++)
        {
          // Mark whether the closer is allowed to leave the child open
          closer.permit_leave_open = it->get_leave_open();
          // Handle a special case where we've arrive at our destination
          // node and it is a partition.  If we're permitted to leave
          // the partition open, then don't actually close the partition
          // but instead siphon each of its open children.  This handles
          // the case of leaving open a read-only partition.
          if (!has_child && !node->is_region() && it->get_leave_open())
          {
            // Release our state
            node->release_physical_state(state);
            RegionTreeNode *child = node->get_tree_child(it->get_child());
            RegionNode *child_node = static_cast<RegionNode*>(child);
            {
              // Make a new node for closing the child 
              PhysicalCloser child_closer(info, true/*leave open*/, 
                                          child_node->handle);
              // Acquire the child's state
              PhysicalState &child_state = 
                child_node->acquire_physical_state(info->ctx,true/*exclusive*/);
              // Now do the close operation
              if (!child_node->siphon_physical_children(child_closer, 
                                                        child_state,
                                                        it->close_mask, 
                                                        -1/*next child*/,
                                                        create_composite))
              {
                // If we failed, release the child state before returning
                child_node->release_physical_state(child_state);
                if (!create_composite)
                {
                  // We also need to pass back any close operations which we
                  // failed to perform so that someone else will try to perform
                  // them as well.
                  while (it != close_ops.end())
                  {
                    it->return_close_op();
                    it++;
                  }
                  return false;
                }
                else
                {
                  node->acquire_physical_state(state, true/*exclusive*/);
                  break;
                }
              }
              else // We succeeded
              {
                // Finally update the node's state
                child_closer.update_node_views(child_node, child_state);
                // Release the child's state
                child_node->release_physical_state(child_state);
              }
            }
            // Reacquire our state before continuing
            node->acquire_physical_state(state, true/*exclusive*/);
          }
          else
          {
            if (!node->close_physical_child(closer, state,
                                            it->close_mask,
                                            it->get_child(),
                                            (has_child ? int(next_child) : -1),
                                            create_composite))
            {
              if (!create_composite)
              {
                // If we failed, release the state before returning
                node->release_physical_state(state);
                // We also need to pass back any close operations which
                // we failed to perform so that someone else will try to
                // perform them as well.
                while (it != close_ops.end())
                {
                  it->return_close_op();
                  it++;
                }
                return false;
              }
              else
                break;
            }
          }
        }
        if (create_composite)
        {
          // Create a root composite node, no need to put it
          // in the map as no else will ever need to name it
          CompositeNode *root = 
            legion_new<CompositeNode>(node, ((CompositeNode*)NULL/*parent*/));
          FieldMask dirty_mask, complete_mask; 
          const bool capture_children = !node->is_region();
          std::map<Color,FieldMask> complete_children;
          CompositeCloser closer(info->ctx);
          for (std::deque<CloseInfo>::const_iterator it = close_ops.begin();
                it != close_ops.end(); it++)
          {
            closer.permit_leave_open = it->get_leave_open();
            Color child_to_close = it->get_child();
            FieldMask child_complete;
            node->close_physical_child(closer, root, state, 
                                       it->close_mask, child_to_close,
                                       (has_child ? int(next_child) : -1),
                                       dirty_mask, child_complete);
            if (!child_complete)
              continue;
            if (capture_children)
            {
              std::map<Color,FieldMask>::iterator finder = 
                complete_children.find(child_to_close);
              if (finder == complete_children.end())
                complete_children[child_to_close] = child_complete;
              else
                finder->second |= child_complete;
            }
            else
              complete_mask |= child_complete;
          }
          FieldMask capture_mask = dirty_mask;
          // If we are a region, then we can avoid closing any fields
          // for which we know we had complete children
          if (!capture_children)
          {
            // This is a region so we can avoid capturing data for
            // any fields which have complete children
            capture_mask -= complete_mask;
          }
          else if ((complete_children.size() == node->get_num_children()) &&
                    node->is_complete())
          {
            // Otherwise, this is a partition, so if we closed all
            // the children and this is a complete partition check to
            // see which fields we closed all the children so we can
            // count them as complete
            FieldMask complete_mask = capture_mask;
            for (std::map<Color,FieldMask>::const_iterator it = 
                  complete_children.begin(); it != 
                  complete_children.end(); it++)
            {
              complete_mask &= it->second;
            }
            if (!!complete_mask)
              capture_mask -= complete_mask;
          }
          // Update the root node with the appropriate information
          // and then create a composite view
          root->capture_physical_state(node, state, capture_mask, 
                                       closer, dirty_mask, complete_mask);
          closer.update_valid_views(state, root, dirty_mask);
        }
        FieldMask next_valid;
        for (std::map<Color,FieldMask>::const_iterator it = 
              state.children.open_children.begin(); it !=
              state.children.open_children.end(); it++)
        {
          next_valid |= it->second;
        }
        state.children.valid_fields = next_valid;
        // Update the node views and the dirty mask
        if (!create_composite)
          closer.update_node_views(node, state);
        node->release_physical_state(state);
      }
      // Flush any reduction operations
      node->flush_reductions(info->traversal_mask, 
                             info->req.redop, info);
      PhysicalState &state = 
        node->acquire_physical_state(info->ctx, true/*exclusive*/);
      // Update our physical state to indicate which child
      // we are opening and in which fields
      if (has_child)
      {
        state.children.valid_fields |= info->traversal_mask;
        std::map<Color,FieldMask>::iterator finder = 
          state.children.open_children.find(next_child);
        if (finder == state.children.open_children.end())
          state.children.open_children[next_child] = info->traversal_mask;
        else
          finder->second |= info->traversal_mask;
      }
      // Finally check to see if we arrived at our destination node
      // in which case we should pull down the valid instance views
      // to our node
      else if (!IS_REDUCE(info->req))
      {
        // If we're not doing a reduction, pull down all the valid views
        // and then record the valid physical instances unless we're
        // doing a reductions in which case it doesn't matter
        node->pull_valid_instance_views(state, info->traversal_mask);
        // Find the memories for all the instances and report
        // which memories have full instances and which ones
        // only have partial instances
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              state.valid_views.begin(); it != state.valid_views.end(); it++)
        {
          if (it->first->is_composite_view())
            continue;
          MaterializedView *cur_view = it->first->as_materialized_view();
          Memory mem = cur_view->get_location();
          std::map<Memory,bool>::iterator finder = 
            info->req.current_instances.find(mem);
          if ((finder == info->req.current_instances.end()) 
              || !finder->second)
          {
            bool full_instance = !(info->traversal_mask - it->second);
            if (finder == info->req.current_instances.end())
              info->req.current_instances[mem] = full_instance;
            else
              finder->second = full_instance;
          }
        }
        // Also set the maximum blocking factor for this region
        Domain node_domain = node->get_domain();
        if (node_domain.get_dim() == 0)
        {
          const LowLevel::ElementMask &mask = 
            node_domain.get_index_space().get_valid_mask();
          info->req.max_blocking_factor = mask.get_num_elmts();
        }
        else
          info->req.max_blocking_factor = node_domain.get_volume();
      }
      // Release the state
      node->release_physical_state(state);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // MappingTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingTraverser::MappingTraverser(RegionTreePath &p, MappableInfo *in,
                                       const RegionUsage &u, const FieldMask &m,
                                       Processor proc, unsigned idx)
      : PathTraverser(p), info(in), usage(u), user_mask(m), 
        target_proc(proc), index(idx)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MappingTraverser::MappingTraverser(const MappingTraverser &rhs)
      : PathTraverser(rhs.path), info(NULL), usage(RegionUsage()),
        user_mask(FieldMask()), target_proc(rhs.target_proc), index(rhs.index)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MappingTraverser::~MappingTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappingTraverser& MappingTraverser::operator=(const MappingTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      if (!has_child)
      {
        // Now we're ready to map this instance
        // Separate paths for reductions and non-reductions
        if (!IS_REDUCE(info->req))
        {
          // See if we can get or make a physical instance
          // that we can use
          return map_physical_region(node);
        }
        else
        {
          // See if we can make or use an existing reduction instance
          return map_reduction_region(node);
        }
      }
      else
      {
        // Still not there yet, traverse the node
        traverse_node(node);
        return true;
      } 
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      // Since we know we're mapping we know we won't ever stop
      // on a partition node
#ifdef DEBUG_HIGH_LEVEL
      assert(has_child);
#endif
      traverse_node(node);
      return true;
    }

    //--------------------------------------------------------------------------
    const MappingRef& MappingTraverser::get_instance_ref(void) const
    //--------------------------------------------------------------------------
    {
      return result;
    }

    //--------------------------------------------------------------------------
    void MappingTraverser::traverse_node(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, MAPPING_TRAVERSE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(has_child);
#endif
      PhysicalState &state = 
        node->acquire_physical_state(info->ctx, true/*exclusive*/);
      state.children.valid_fields |= info->traversal_mask;
      std::map<Color,FieldMask>::iterator finder = 
        state.children.open_children.find(next_child);
      if (finder == state.children.open_children.end())
        state.children.open_children[next_child] = info->traversal_mask;
      else
        finder->second |= info->traversal_mask;
      node->release_physical_state(state);
      // Flush any outstanding reduction operations
      node->flush_reductions(user_mask, 
                             info->req.redop, info);
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::map_physical_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, MAP_PHYSICAL_REGION_CALL);
#endif
      std::vector<Memory> &chosen_order = info->req.target_ranking;
      const std::set<FieldID> &additional_fields = info->req.additional_fields;
      // Clamp the selected blocking factor
      const size_t blocking_factor = 
        (info->req.blocking_factor <= info->req.max_blocking_factor) ? 
        info->req.blocking_factor : info->req.max_blocking_factor;
      // Filter out any memories that are not visible from 
      // the target processor if there is a processor that 
      // we're targeting (e.g. never do this for premaps)
      // We can also skip this if region requirement has a NO_ACCESS_FLAG
      if (!chosen_order.empty() && target_proc.exists() &&
          !(info->req.flags & NO_ACCESS_FLAG))
      {
        Machine *machine = Machine::get_machine();
        const std::set<Memory> &visible_memories = 
          machine->get_visible_memories(target_proc);
        std::vector<Memory> filtered_memories;
        filtered_memories.reserve(chosen_order.size());
        for (std::vector<Memory>::const_iterator it = chosen_order.begin();
              it != chosen_order.end(); it++)
        {
          if (visible_memories.find(*it) == visible_memories.end())
          {
            log_region(LEVEL_WARNING,"WARNING: Mapper specified memory " IDFMT 
                                     " which is not visible from processor "
                                     "" IDFMT " when mapping region %d of "
                                     "mappable (ID %lld)!  Removing memory "
                                     "from the chosen ordering!", it->id, 
                                     target_proc.id, index, 
                                     info->mappable->get_unique_mappable_id());
            continue;
          }
          // Also if we're restricted, remove any memories which
          // do not already contain valid memories.
          if (info->req.restricted && 
              (info->req.current_instances.find(*it) == 
                info->req.current_instances.end()))
          {
            log_region(LEVEL_WARNING,"WARNING: Mapper specified memory " IDFMT
                                     " for restricted region requirement "
                                     "when mapping region %d of mappable "
                                     "(ID %lld) on processor " IDFMT "!  "
                                     "Removing memory from the chosen "
                                     "ordering!", it->id, index, 
                                     info->mappable->get_unique_mappable_id(),
                                     target_proc.id);
            continue;
          }
          // Otherwise we can add it to the list of filtered memories
          filtered_memories.push_back(*it);
        }
        chosen_order = filtered_memories;
      }
      // Get the set of currently valid instances
      std::map<InstanceView*,FieldMask> valid_instances;
      // Check to see if the mapper requested any additional fields in this
      // instance.  If it did, then re-run the computation to get the list
      // of valid instances with the right set of fields
      std::set<FieldID> new_fields = info->req.privilege_fields;
      {
        PhysicalState &state = 
          node->acquire_physical_state(info->ctx, false/*exclusive*/);
        if (!additional_fields.empty())
        {
          new_fields.insert(additional_fields.begin(),
                               additional_fields.end());
          FieldMask additional_mask = 
            node->column_source->get_field_mask(new_fields);
          node->find_valid_instance_views(state, additional_mask,
                                          additional_mask, true/*space*/,
                                          valid_instances);
        }
        else
        {
          node->find_valid_instance_views(state, user_mask,
                                          user_mask, true/*space*/,
                                          valid_instances);
        }
        node->release_physical_state(state);
      }
      // Compute the set of valid memories and filter out instance which
      // do not have the proper blocking factor in the process
      std::map<Memory,bool> valid_memories;
      {
        std::vector<InstanceView*> to_erase;
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          // Remove any composite instances
          if (it->first->is_composite_view())
          {
            to_erase.push_back(it->first);
            if (it->first->remove_valid_reference())
              legion_delete(it->first->as_composite_view());
            continue;
          }
          MaterializedView *current_view = it->first->as_materialized_view();
          // For right now allow blocking factors that are greater
          // than or equal to the requested blocking factor
          size_t bf = current_view->get_blocking_factor();
          if (bf >= blocking_factor)
          {
            Memory m = current_view->get_location();
            if (valid_memories.find(m) == valid_memories.end())
              valid_memories[m] = !(user_mask - it->second);
            else if (!valid_memories[m])
              valid_memories[m] = !(user_mask - it->second);
            // Otherwise we already have an instance in this memory that
            // dominates all the fields in which case we don't care
          }
          else
          {
            to_erase.push_back(it->first);
            if (it->first->remove_valid_reference())
              legion_delete(it->first->as_materialized_view());
          }
        }
        for (std::vector<InstanceView*>::const_iterator it = to_erase.begin();
              it != to_erase.end(); it++)
          valid_instances.erase(*it);  
        to_erase.clear();
      }

      MaterializedView *chosen_inst = NULL;
      FieldMask needed_fields; 
      // Go through each of the memories provided by the mapper
      for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
            mit != chosen_order.end(); mit++)
      {
        // See if it has any valid instances
        if (valid_memories.find(*mit) != valid_memories.end())
        {
          // Already have a valid instance with at least a 
          // few valid fields, figure out if it has all or 
          // some of the fields valid
          if (valid_memories[*mit])
          {
            // We've got an instance with all the valid fields, go find it
            for (std::map<InstanceView*,FieldMask>::const_iterator it = 
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              // At this point we know everything is a materialized view
              MaterializedView *cur_view = it->first->as_materialized_view();
              if (cur_view->get_location() != (*mit))
                continue;
              if (!(user_mask - it->second))
              {
                // Check to see if have any WAR dependences
                // in which case we'll skip it for a something better
                if (info->req.enable_WAR_optimization && HAS_WRITE(info->req) 
                    && cur_view->has_war_dependence(usage, user_mask))
                  continue;
                // No WAR problems, so it it is good
                chosen_inst = cur_view;
                // No need to set needed fields since everything is valid
                break;
              }
            }
            // If we found a good instance break, otherwise go onto
            // the partial instances
            if (chosen_inst != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!needed_fields);
#endif
              break;
            }
          }
          // Do this if we couldn't find a better choice
          // Note we can't do this in the read-only case because we might 
          // end up issuing multiple copies to the same location.
          // On second thought this might be ok since they are both 
          // reading and anybody else who mutates this instance will
          // see both copies because of mapping dependences.
          // if (!IS_READ_ONLY(usage))
          {
            // These are instances which have space for all the required fields
            // but only a subset of those fields contain valid data.
            // Find the valid instance with the most valid fields to use.
            int covered_fields = -1;
            for (std::map<InstanceView*,FieldMask>::const_iterator it =
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              // At this point we know everything is a materialized view
              MaterializedView *cur_view = it->first->as_materialized_view();
              if (cur_view->get_location() != (*mit))
                continue;
              int cf = FieldMask::pop_count(it->second);
              if (cf > covered_fields)
              {
                // Check to see if we have any WAR dependences 
                // which might disqualify us
                if (info->req.enable_WAR_optimization && HAS_WRITE(info->req) 
                    && it->first->has_war_dependence(usage, user_mask))
                  continue;
                covered_fields = cf;
                chosen_inst = cur_view; 
                needed_fields = user_mask - it->second; 
              }
            }
            // If we got a good one break out, otherwise we'll try 
            // to make a new instance
            if (chosen_inst != NULL)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!!needed_fields);
#endif
              break;
            }
          }
        }
        // If we're restricted we can't make instances, so just keep going
        if (info->req.restricted)
          continue;
        // If it didn't find a valid instance, try to make one
        chosen_inst = node->create_instance(*mit, new_fields, 
                                            blocking_factor,
                                            info->mappable->get_depth());
        if (chosen_inst != NULL)
        {
          // We successfully made an instance
          needed_fields = user_mask;
          break;
        }
      }
      // Save our chosen instance if it exists in the mapping
      // reference and then return if we have an instance
      if (chosen_inst != NULL)
        result = MappingRef(chosen_inst, needed_fields);
      // Remove any valid references we are still holding
      // This has to go after we create the mapping reference to 
      // guarantee we hold a valid reference to the chosen instance
      // at all times
      RegionTreeNode::remove_valid_references(valid_instances);
      return (chosen_inst != NULL);
    }

    //--------------------------------------------------------------------------
    bool MappingTraverser::map_reduction_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(node->context, MAP_REDUCTION_REGION_CALL);
#endif
      std::vector<Memory> &chosen_order = info->req.target_ranking;
      // Filter out any memories that are not visible from 
      // the target processor if there is a processor that 
      // we're targeting (e.g. never do this for premaps)
      if (!chosen_order.empty() && target_proc.exists())
      {
        Machine *machine = Machine::get_machine();
        const std::set<Memory> &visible_memories = 
          machine->get_visible_memories(target_proc);
        std::vector<Memory> filtered_memories;
        filtered_memories.reserve(chosen_order.size());
        for (std::vector<Memory>::const_iterator it = chosen_order.begin();
              it != chosen_order.end(); it++)
        {
          if (visible_memories.find(*it) != visible_memories.end())
            filtered_memories.push_back(*it);
          else
          {
            log_region(LEVEL_WARNING,"WARNING: Mapper specified memory " IDFMT
                                     " which is not visible from processor "
                                     IDFMT " when mapping region %d of mappable"
                                     " (ID %lld)!  Removing memory from the "
                                     "chosen ordering!", it->id, 
                                     target_proc.id, index, 
                                     info->mappable->get_unique_mappable_id());
          }
        }
        chosen_order = filtered_memories;
      }

      std::set<ReductionView*> valid_views;
      {
        PhysicalState &state = 
          node->acquire_physical_state(info->ctx, false/*exclusive*/);
        node->find_valid_reduction_views(state, user_mask, valid_views);
        node->release_physical_state(state);
      }

      // Compute the set of valid memories
      std::set<Memory> valid_memories;
      for (std::set<ReductionView*>::const_iterator it = valid_views.begin();
            it != valid_views.end(); it++)
      {
        valid_memories.insert((*it)->get_location());
      }

      ReductionView *chosen_inst = NULL;
      // Go through each of the valid memories and see if we can either find
      // a reduction instance or we can make one
      for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
            mit != chosen_order.end(); mit++)
      {
        if (valid_memories.find(*mit) != valid_memories.end())
        {
          // We've got a valid instance, let's go find it
          for (std::set<ReductionView*>::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            if ((*it)->get_location() == *mit)
            {
              chosen_inst = *it;
              break;
            }
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(chosen_inst != NULL);
#endif
          // We've found the instance that we want
          break;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(info->req.privilege_fields.size() == 1);
#endif
          FieldID fid = *(info->req.privilege_fields.begin());
          // Try making a reduction instance in this memory
          chosen_inst = node->create_reduction(*mit, fid, 
                                               info->req.reduction_list,
                                               info->req.redop);
          if (chosen_inst != NULL)
            break;
        }
      }
      if (chosen_inst != NULL)
        result = MappingRef(chosen_inst,FieldMask());
      // Remove our valid references before we return
      RegionTreeNode::remove_valid_references(valid_views);
      return (chosen_inst != NULL);
    }

    /////////////////////////////////////////////////////////////
    // StateSender
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    StateSender::StateSender(ContextID c, UniqueID id, AddressSpaceID t,
                             std::map<LogicalView*,FieldMask> &views,
                             std::set<PhysicalManager*> &managers,
                             const FieldMask &mask, bool inv)
      : ctx(c), uid(id), target(t), needed_views(views),
        needed_managers(managers), send_mask(mask), invalidate(inv)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    StateSender::StateSender(const StateSender &rhs)
      : ctx(0), uid(0), target(0), 
        needed_views(rhs.needed_views), needed_managers(rhs.needed_managers),
        send_mask(FieldMask()), invalidate(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    StateSender::~StateSender(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    StateSender& StateSender::operator=(const StateSender &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool StateSender::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool StateSender::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      return node->send_state(ctx, uid, target, send_mask, invalidate,
                              needed_views, needed_managers);
    }

    //--------------------------------------------------------------------------
    bool StateSender::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      return node->send_state(ctx, uid, target, send_mask, invalidate,
                              needed_views, needed_managers); 
    }

    /////////////////////////////////////////////////////////////
    // PathReturner 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PathReturner::PathReturner(RegionTreePath &path, ContextID c,
                               RegionTreeContext remote, AddressSpaceID t,
                               const FieldMask &mask,
                               std::set<PhysicalManager*> &needed)
      : PathTraverser(path), ctx(c), remote_ctx(remote.get_id()),
        target(t), return_mask(mask), needed_managers(needed)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    PathReturner::PathReturner(const PathReturner &rhs)
      : PathTraverser(rhs.path), ctx(0), remote_ctx(0), 
        target(0), return_mask(FieldMask()), 
        needed_managers(rhs.needed_managers)
    //--------------------------------------------------------------------------
    {
      // should never get called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PathReturner::~PathReturner(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PathReturner& PathReturner::operator=(const PathReturner &rhs)
    //--------------------------------------------------------------------------
    {
      // should never get called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PathReturner::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      return node->send_back_state(ctx, remote_ctx,
                                   target, false/*invalidate*/, 
                                   return_mask, needed_managers);
    }

    //--------------------------------------------------------------------------
    bool PathReturner::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      return node->send_back_state(ctx, remote_ctx,
                                   target, false/*invalidate*/, 
                                   return_mask, needed_managers);
    }

    /////////////////////////////////////////////////////////////
    // StateReturner 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    StateReturner::StateReturner(ContextID c, RegionTreeContext remote,
                             AddressSpaceID t, bool inv, const FieldMask &mask,
                             std::set<PhysicalManager*> &needed)
      : ctx(c), remote_ctx(remote.get_id()), 
        target(t), invalidate(inv), return_mask(mask), needed_managers(needed)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    StateReturner::StateReturner(const StateReturner &rhs)
      : ctx(0), remote_ctx(0),
        target(0), invalidate(false), return_mask(FieldMask()),
        needed_managers(rhs.needed_managers)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    StateReturner::~StateReturner(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    StateReturner& StateReturner::operator=(const StateReturner &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool StateReturner::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool StateReturner::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      return node->send_back_state(ctx, remote_ctx,
                 target, invalidate, return_mask, needed_managers);
    }
    
    //--------------------------------------------------------------------------
    bool StateReturner::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      return node->send_back_state(ctx, remote_ctx,
                 target, invalidate, return_mask, needed_managers);
    }

    /////////////////////////////////////////////////////////////
    // States 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalState::LogicalState(void)
    //--------------------------------------------------------------------------
    {
      // Initialize the version of all fields to zero
      field_versions[0] = FieldMask(FIELD_ALL_ONES);
#ifdef LOGICAL_FIELD_TREE
      curr_epoch_users = new FieldTree<LogicalUser>(FIELD_ALL_ONES);
      prev_epoch_users = new FieldTree<LogicalUser>(FIELD_ALL_ONES);
#endif
    }

    //--------------------------------------------------------------------------
    LogicalState::LogicalState(const LogicalState &rhs)
    //--------------------------------------------------------------------------
    {
      // The only place this is called is when we are putting a 
      // new logical state in the list of logical states so we 
      // only need to copy over the field versions and make new
      // field trees if necessary.
      // Only copy over the field versions
      field_versions = rhs.field_versions;
#ifdef LOGICAL_FIELD_TREE
      curr_epoch_users = new FieldTree<LogicalUser>(FIELD_ALL_ONES);
      prev_epoch_users = new FieldTree<LogicalUser>(FIELD_ALL_ONES);
#endif
    }

    //--------------------------------------------------------------------------
    LogicalState::~LogicalState(void)
    //--------------------------------------------------------------------------
    {
#ifdef LOGICAL_FIELD_TREE
      delete curr_epoch_users;
      curr_epoch_users = NULL;
      delete prev_epoch_users;
      prev_epoch_users = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    LogicalState& LogicalState::operator=(const LogicalState &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LogicalState::reset(void)
    //--------------------------------------------------------------------------
    {
      field_versions.clear();
      field_versions[0] = FieldMask(FIELD_ALL_ONES);
      field_states.clear();
#ifndef LOGICAL_FIELD_TREE
      curr_epoch_users.clear();
      prev_epoch_users.clear();
#else
      // Free up the old field trees and make new ones
      delete curr_epoch_users;
      delete prev_epoch_users;
      curr_epoch_users = new FieldTree<LogicalUser>(FIELD_ALL_ONES);
      prev_epoch_users = new FieldTree<LogicalUser>(FIELD_ALL_ONES);
#endif
      close_operations.clear();
      user_level_coherence = FieldMask();
    }

    //--------------------------------------------------------------------------
    LogicalDepAnalyzer::LogicalDepAnalyzer(const LogicalUser &u,
                                           const FieldMask &check_mask,
                                           bool validates, bool trace)
      : user(u), validates_regions(validates), 
        tracing(trace), dominator_mask(check_mask)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    bool LogicalDepAnalyzer::analyze(LogicalUser &prev_user)
    //--------------------------------------------------------------------------
    {
      // Keep track of fields for which we have seen users
      observed_mask |= prev_user.field_mask;
      DependenceType dtype = check_dependence_type(prev_user.usage, user.usage);
      bool validate = validates_regions;
      switch (dtype)
      {
        case NO_DEPENDENCE:
          {
            // No dependence so remove bits from the dominator mask
            dominator_mask -= prev_user.field_mask;
            break;
          }
        case ANTI_DEPENDENCE:
        case ATOMIC_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          {
            // Mark that these kinds of dependences are not allowed
            // to validate region inputs
            validate = false;
            // No break so we register dependences just like
            // a true dependence
          }
        case TRUE_DEPENDENCE:
          {
#ifdef LEGION_LOGGING
            LegionLogging::log_mapping_dependence(
                Machine::get_executing_processor(),
                user.op->get_parent()->get_unique_task_id(),
                prev_user.uid, prev_user.idx, 
                user.uid, user.idx, dtype);
#endif
#ifdef LEGION_SPY
            LegionSpy::log_mapping_dependence(
                user.op->get_parent()->get_unique_task_id(),
                prev_user.uid, prev_user.idx, 
                user.uid, user.idx, dtype);
#endif
            // Do this after the logging since we might 
            // update the iterator.
            // If we can validate a region record which of our
            // predecessors regions we are validating, otherwise
            // just register a normal dependence
            if (validate)
            {
              if (user.op->register_region_dependence(user.idx, prev_user.op,
                                                      prev_user.gen, 
                                                      prev_user.idx, dtype))
              {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
                // Now we can prune it from the list
                return false;
#else
                return true;
#endif
              }
              else
              {
                // Reset the timeout and continue
                prev_user.timeout = LogicalUser::TIMEOUT;
                return true;
              }
            }
            else
            {
              if (user.op->register_dependence(user.idx, prev_user.op, 
                                               prev_user.gen, 
                                               prev_user.idx, dtype))
              {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
                // Now we can prune it from the list
                return false;
#else
                return true;
#endif
              }
              else
              {
                // Reset the timeout and continue
                prev_user.timeout = LogicalUser::TIMEOUT;
                return true;
              }
            }
            break;
          }
        default:
          assert(false); // should never get here
      }
      // When tracing we don't do timeouts because it is unsound
      if (tracing)
        return true;
      // See if the timeout has expired
      if (prev_user.timeout <= 0)
      {
        if (prev_user.op->is_operation_committed(prev_user.gen))
          return false;
        // Otherwise reset the timeout and keep it
        prev_user.timeout = LogicalUser::TIMEOUT;
        return true;
      }
      // Otherwise just decrement the timeout and keep it
      prev_user.timeout--;
      return true;
    }

    //--------------------------------------------------------------------------
    FieldMask LogicalDepAnalyzer::get_dominator_mask(void) const
    //--------------------------------------------------------------------------
    {
      // It is only sound to say that we dominated fields for which
      // we actually observed users, so intersect us with the 
      // observed mask.
      return (dominator_mask & observed_mask); 
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    LogicalOpAnalyzer<DOMINATE>::LogicalOpAnalyzer(Operation *o)
      : op(o)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    bool LogicalOpAnalyzer<DOMINATE>::analyze(LogicalUser &prev_user)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_mapping_dependence(
            Machine::get_executing_processor(),
            op->get_parent()->get_unique_task_id(),
            prev_user.uid, prev_user.idx, op->get_unique_op_id(),
            0/*idx*/, TRUE_DEPENDENCE);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_mapping_dependence(
            op->get_parent()->get_unique_task_id(),
            prev_user.uid, prev_user.idx, op->get_unique_op_id(),
            0/*idx*/, TRUE_DEPENDENCE);
#endif
      if (op->register_dependence(prev_user.op, prev_user.gen))
      {
        // Prune it from the list
        return false;
      }
      else if (DOMINATE)
        return false;
      // Check to see if the timeout has expired
      if (prev_user.timeout <= 0)
      {
        if (prev_user.op->is_operation_committed(prev_user.gen))
          return false;
        prev_user.timeout = LogicalUser::TIMEOUT;
        return true;
      }
      // Otherwise it can stay
      prev_user.timeout--;
      return true;
    }

    //--------------------------------------------------------------------------
    LogicalFilter::LogicalFilter(const FieldMask &mask,
                                 FieldTree<LogicalUser> *t /*= NULL*/)
      : filter_mask(mask), target(t), reinsert_count(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool LogicalFilter::analyze(LogicalUser &user)
    //--------------------------------------------------------------------------
    {
      if (target == NULL)
      {
        user.field_mask -= filter_mask;
        if (!user.field_mask)
        {
          // Otherwise remove the mapping reference
          // and then remove it from the list
          user.op->remove_mapping_reference(user.gen);
        }
        else
        {
          reinsert.push_back(user);
          reinsert_count++;
        }
        return false;
      }
      else
      {
        FieldMask overlap = user.field_mask & filter_mask;
        if (!!overlap)
        {
          // Insert a copy into the target
          LogicalUser copy = user;
          copy.field_mask = overlap;
          // Add a mapping reference before inserting
          copy.op->add_mapping_reference(copy.gen);
          target->insert(copy);
        }
        else
          return true; // No overlap so we can keep it
        // Remove any dominated fields
        user.field_mask -= filter_mask;
        if (!user.field_mask)
        {
          // Remove the mapping reference
          user.op->remove_mapping_reference(user.gen);
        }
        else
        {
          // Put it on the list to reinsert
          reinsert.push_back(user);
          reinsert_count++;
        }
        // Always remove it if we remove fields so that
        // it can be reinserted
        return false;
      }
    }

    //--------------------------------------------------------------------------
    void LogicalFilter::begin_node(FieldTree<LogicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Save the reinsert count from the next level up
      reinsert_stack.push_back(reinsert_count);
      reinsert_count = 0;
    }

    //--------------------------------------------------------------------------
    void LogicalFilter::end_node(FieldTree<LogicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Reinsert any users from this node that 
      for (unsigned idx = 0; idx < reinsert_count; idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!reinsert.empty());
#endif
        // Don't recurse when inserting as it may lead to 
        // long chains of unused nodes in the field tree
        node->insert(reinsert.back(), false/*recurse*/);
        reinsert.pop_back();
      }
      // Then restore the reinsert count from the next level up
      reinsert_count = reinsert_stack.back();
      reinsert_stack.pop_back();
    }

    //--------------------------------------------------------------------------
    bool LogicalFieldInvalidator::analyze(const LogicalUser &user)
    //--------------------------------------------------------------------------
    {
      user.op->remove_mapping_reference(user.gen);
      return false;
    }

    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(void)
      : acquired_count(0), exclusive(false), ctx(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(ContextID c)
      : acquired_count(0), exclusive(false), ctx(c)
    //--------------------------------------------------------------------------
    {
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    PhysicalState::PhysicalState(ContextID c, RegionTreeNode *n)
      : acquired_count(0), exclusive(false), ctx(c), node(n)
    //--------------------------------------------------------------------------
    { 
    }
#endif

    //--------------------------------------------------------------------------
    FieldState::FieldState(const GenericUser &user, const FieldMask &m, Color c)
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
      valid_fields = m;
      open_children[c] = m;
      rebuild_timeout = 1;
    }

    //--------------------------------------------------------------------------
    bool FieldState::overlaps(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      if (redop != rhs.redop)
        return false;
      if (redop == 0)
        return (open_state == rhs.open_state);
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((open_state == OPEN_SINGLE_REDUCE) ||
               (open_state == OPEN_MULTI_REDUCE));
        assert((rhs.open_state == OPEN_SINGLE_REDUCE) ||
               (rhs.open_state == OPEN_MULTI_REDUCE));
#endif
        // Only support merging reduction fields with exactly the
        // same mask which should be single fields for reductions
        return (valid_fields == rhs.valid_fields);
      }
    }

    //--------------------------------------------------------------------------
    void FieldState::merge(const FieldState &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields |= rhs.valid_fields;
      for (std::map<Color,FieldMask>::const_iterator it = 
            rhs.open_children.begin(); it != rhs.open_children.end(); it++)
      {
        std::map<Color,FieldMask>::iterator finder = 
          open_children.find(it->first);
        if (finder == open_children.end())
          open_children[it->first] = it->second;
        else
          finder->second |= it->second;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == rhs.redop);
#endif
      if (redop > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!open_children.empty());
#endif
        // For the reductions, handle the case where we need to merge
        // reduction modes
        if (open_children.size() == 1)
          open_state = OPEN_SINGLE_REDUCE;
        else
          open_state = OPEN_MULTI_REDUCE;
      }
    }

    //--------------------------------------------------------------------------
    void FieldState::print_state(TreeStateLogger *logger,
                                 const FieldMask &capture_mask) const
    //--------------------------------------------------------------------------
    {
      switch (open_state)
      {
        case NOT_OPEN:
          {
            logger->log("Field State: NOT OPEN (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_WRITE:
          {
            logger->log("Field State: OPEN READ WRITE (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_ONLY:
          {
            logger->log("Field State: OPEN READ-ONLY (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_SINGLE_REDUCE:
          {
            logger->log("Field State: OPEN SINGLE REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
            break;
          }
        case OPEN_MULTI_REDUCE:
          {
            logger->log("Field State: OPEN MULTI REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
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

    /////////////////////////////////////////////////////////////
    // Closers 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(ContextID c, const LogicalUser &u, bool val)
      : ctx(c), user(u), validates(val)
    //--------------------------------------------------------------------------
    {
    }

#ifdef LOGICAL_FIELD_TREE
    //--------------------------------------------------------------------------
    void LogicalCloser::begin_node(FieldTree<LogicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Save the reinsert count from the next level up
      reinsert_stack.push_back(reinsert_count);
      reinsert_count = 0;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::end_node(FieldTree<LogicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Reinsert any users from this node that we pulled out
      for (unsigned idx = 0; idx < reinsert_count; idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!reinsert.empty());
#endif
        // Don't recurse when reinserting into the tree
        // as it may lead to long chains of unused nodes.
        node->insert(reinsert.back(), false/*recurse*/);
        reinsert.pop_back();
      }
      // Then restore the reinsert count from the next level up
      reinsert_count = reinsert_stack.back();
      reinsert_stack.pop_back();
    } 

    //--------------------------------------------------------------------------
    bool LogicalCloser::analyze(LogicalUser &prev_user)
    //--------------------------------------------------------------------------
    {
      FieldMask overlap = local_closing_mask & prev_user.field_mask;
      if (!overlap)
        return true;
      closed_users.push_back(prev_user);
      // Update the field mask and privilege
      closed_users.back().field_mask = overlap;
      closed_users.back().usage.privilege = 
        (PrivilegeMode)((int)closed_users.back().privilege | PROMOTED);
      // Remove the close set of fields from this user
      prev_user.field_mask -= overlap;
      // If it's empty, remove it from the list and let
      // the mapping reference go up the tree with it.
      // Otherwise add a new mapping reference.
      if (!!prev_user.field_mask)
      {
        prev_user.op->add_mapping_reference(prev_user.gen);
        reinsert.push_back(prev_user);
        reinsert_count++;
      }
      return false; // don't keep it
    }
#endif

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(MappableInfo *in, bool open, LogicalRegion h)
      : info(in), handle(h), permit_leave_open(open), targets_selected(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::PhysicalCloser(const PhysicalCloser &rhs)
      : info(rhs.info), handle(rhs.handle), 
        permit_leave_open(rhs.permit_leave_open),
        upper_targets(rhs.get_lower_targets())
    //--------------------------------------------------------------------------
    {
      targets_selected = !upper_targets.empty(); 
      if (targets_selected)
      {
        for (std::vector<MaterializedView*>::const_iterator it = 
              upper_targets.begin(); it != upper_targets.end(); it++)
        {
          (*it)->add_valid_reference();
        }
      }
    }

    //--------------------------------------------------------------------------
    PhysicalCloser::~PhysicalCloser(void)
    //--------------------------------------------------------------------------
    {
      // Remove any valid references that we have from physical regions
      for (std::vector<MaterializedView*>::const_iterator it = 
            upper_targets.begin(); it != upper_targets.end(); it++)
      {
        if ((*it)->remove_valid_reference())
          legion_delete(*it);
      }
    }

    //--------------------------------------------------------------------------
    PhysicalCloser& PhysicalCloser::operator=(const PhysicalCloser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalCloser::needs_targets(void) const
    //--------------------------------------------------------------------------
    {
      return !targets_selected;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::add_target(MaterializedView *target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(target != NULL);
#endif
      target->add_valid_reference();
      upper_targets.push_back(target);
      targets_selected = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::close_tree_node(RegionTreeNode *node,
                                         const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
      // Lower the targets
      lower_targets.resize(upper_targets.size());
      for (unsigned idx = 0; idx < upper_targets.size(); idx++)
        lower_targets[idx] = 
          upper_targets[idx]->get_materialized_subview(node->get_color());

      // Close the node
      node->close_physical_node(*this, closing_mask);

      // Clear out the lowered targets
      lower_targets.clear();
    }

    //--------------------------------------------------------------------------
    const std::vector<MaterializedView*>& PhysicalCloser::
                                                  get_upper_targets(void) const
    //--------------------------------------------------------------------------
    {
      return upper_targets;
    }

    //--------------------------------------------------------------------------
    const std::vector<MaterializedView*>& PhysicalCloser::
                                                  get_lower_targets(void) const
    //--------------------------------------------------------------------------
    {
      return lower_targets;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::update_dirty_mask(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      dirty_mask |= mask;
    }

    //--------------------------------------------------------------------------
    const FieldMask& PhysicalCloser::get_dirty_mask(void) const
    //--------------------------------------------------------------------------
    {
      return dirty_mask;
    }

    //--------------------------------------------------------------------------
    void PhysicalCloser::update_node_views(RegionTreeNode *node,
                                           PhysicalState &state)
    //--------------------------------------------------------------------------
    {
      node->update_valid_views(state, info->traversal_mask,
                               dirty_mask, upper_targets);
    }

    //--------------------------------------------------------------------------
    CompositeCloser::CompositeCloser(ContextID c)
      : ctx(c), permit_leave_open(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeCloser::CompositeCloser(const CompositeCloser &rhs)
      : ctx(0), permit_leave_open(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeCloser::~CompositeCloser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeCloser& CompositeCloser::operator=(const CompositeCloser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    CompositeNode* CompositeCloser::get_composite_node(RegionTreeNode *node,
                                                       CompositeNode *parent)
    //--------------------------------------------------------------------------
    {
      std::map<RegionTreeNode*,CompositeNode*>::const_iterator finder = 
        constructed_nodes.find(node);
      if (finder != constructed_nodes.end())
        return finder->second;
      CompositeNode *result = legion_new<CompositeNode>(node, parent);
      constructed_nodes[node] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void CompositeCloser::update_reduction_views(ReductionView *view,
                                                 const FieldMask &valid_fields)
    //--------------------------------------------------------------------------
    {
      std::map<ReductionView*,FieldMask>::iterator finder = 
                                            reduction_views.find(view);
      if (finder != reduction_views.end())
        finder->second |= valid_fields;
      else
      {
        reduction_views[view] = valid_fields;
        view->add_valid_reference();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeCloser::update_valid_views(PhysicalState &state,
                                             CompositeNode *root,
                                             const FieldMask &closed_mask)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *node = root->logical_node;
      DistributedID did = 
                        node->context->runtime->get_available_distributed_id();
      CompositeView *composite_view = legion_new<CompositeView>(node->context, 
                                        did, 
                                        node->context->runtime->address_space,
                                        node, did, closed_mask);
      // Set the root value
      composite_view->add_root(root, closed_mask);
      // Update the reduction views for this level
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        composite_view->update_reduction_views(it->first, it->second);
      }
      // Now update the state of the node
      node->update_valid_views(state,closed_mask,true/*dirty*/,composite_view);
      // Now we can remove the valid references that we hold on the reductions
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        if (it->first->remove_valid_reference())
          legion_delete(it->first);
      }
    }

    //--------------------------------------------------------------------------
    template<bool FILTER>
    PhysicalDepAnalyzer<FILTER>::PhysicalDepAnalyzer(const PhysicalUser &u,
                                             const FieldMask &mask,
                                             RegionTreeNode *node,
                                             std::set<Event> &wait)
      : user(u), logical_node(node), wait_on(wait), reinsert_count(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<bool FILTER>
    const FieldMask& PhysicalDepAnalyzer<FILTER>::get_observed_mask(void) const
    //--------------------------------------------------------------------------
    {
      return observed;
    }

    //--------------------------------------------------------------------------
    template<bool FILTER>
    const FieldMask& PhysicalDepAnalyzer<FILTER>::get_non_dominated_mask(void)
                                                                        const
    //--------------------------------------------------------------------------
    {
      return non_dominated;
    }

    //--------------------------------------------------------------------------
    template<bool FILTER>
    bool PhysicalDepAnalyzer<FILTER>::analyze(PhysicalUser &prev_user)
    //--------------------------------------------------------------------------
    {
      if (FILTER)
        observed |= prev_user.field_mask;
      if (prev_user.term_event == user.term_event)
      {
        if (FILTER)
          non_dominated |= (prev_user.field_mask & user.field_mask);
        return true;;
      }
      if (user.child >= 0)
      {
        // Same child, already done the analysis
        if (user.child == prev_user.child)
        {
          if (FILTER)
            non_dominated |= (prev_user.field_mask & user.field_mask);
          return true;
        }
        // Disjoint children
        if ((prev_user.child >= 0) && 
            logical_node->are_children_disjoint(unsigned(user.child),
                                                unsigned(prev_user.child)))
        {
          if (FILTER)
            non_dominated |= (prev_user.field_mask & user.field_mask);
          return true;
        }
      }
      // Now we need to actually do a dependence check
      DependenceType dt = check_dependence_type(prev_user.usage, user.usage);
      switch (dt)
      {
        case NO_DEPENDENCE:
        case ATOMIC_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          {
            // No actualy dependence
            if (FILTER)
              non_dominated |= (prev_user.field_mask & user.field_mask);
            return true;
          }
        case TRUE_DEPENDENCE:
        case ANTI_DEPENDENCE:
          {
            // Actual dependence
            wait_on.insert(prev_user.term_event);
            break;
          }
        default:
          assert(false);
      }
      // If we made it here we have a true dependence, see if we are filtering
      if (FILTER)
      {
        FieldMask overlap = prev_user.field_mask & user.field_mask;
        filtered_users.push_back(prev_user);
        filtered_users.back().field_mask = overlap;
        prev_user.field_mask -= user.field_mask;
        // Save this one to be put back on the list if its mask is not empty
        if (!!prev_user.field_mask)
        {
          reinsert.push_back(prev_user);
          reinsert_count++;
        }
        return false; // don't keep this one since it has changed
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    template<bool FILTER>
    void PhysicalDepAnalyzer<FILTER>::begin_node(FieldTree<PhysicalUser> *node)
    //--------------------------------------------------------------------------
    {
      if (FILTER)
      {
        // Save the reinsert count from the next level up
        reinsert_stack.push_back(reinsert_count);
        reinsert_count = 0;
      }
    }

    //--------------------------------------------------------------------------
    template<bool FILTER>
    void PhysicalDepAnalyzer<FILTER>::end_node(FieldTree<PhysicalUser> *node)
    //--------------------------------------------------------------------------
    {
      if (FILTER)
      {
        // Reinsert any users from this node that 
        for (unsigned idx = 0; idx < reinsert_count; idx++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!reinsert.empty());
#endif
          // Don't recurse when inserting as it may lead to 
          // long chains of unused nodes in the field tree
          node->insert(reinsert.back(), false/*recurse*/);
          reinsert.pop_back();
        }
        // Then restore the reinsert count from the next level up
        reinsert_count = reinsert_stack.back();
        reinsert_stack.pop_back();
      }
    }

    //--------------------------------------------------------------------------
    template<bool FILTER>
    void PhysicalDepAnalyzer<FILTER>::insert_filtered_users(
                                                FieldTree<PhysicalUser> *target)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < filtered_users.size(); idx++)
      {
        target->insert(filtered_users[idx]);
      }
    }
    
    //--------------------------------------------------------------------------
    PhysicalFilter::PhysicalFilter(const FieldMask &mask)
      : filter_mask(mask), reinsert_count(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PhysicalFilter::analyze(PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      user.field_mask -= filter_mask;
      if (!!user.field_mask)
      {
        reinsert.push_back(user);
        reinsert_count++;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void PhysicalFilter::begin_node(FieldTree<PhysicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Save the reinsert count from the next level up
      reinsert_stack.push_back(reinsert_count);
      reinsert_count = 0;
    }

    //--------------------------------------------------------------------------
    void PhysicalFilter::end_node(FieldTree<PhysicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Reinsert any users from this node that 
      for (unsigned idx = 0; idx < reinsert_count; idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!reinsert.empty());
#endif
        // Don't recurse when inserting as it may lead to 
        // long chains of unused nodes in the field tree
        node->insert(reinsert.back(), false/*recurse*/);
        reinsert.pop_back();
      }
      // Then restore the reinsert count from the next level up
      reinsert_count = reinsert_stack.back();
      reinsert_stack.pop_back();
    }

    //--------------------------------------------------------------------------
    template<bool READING, bool REDUCE, bool TRACK, bool ABOVE>
    PhysicalCopyAnalyzer<READING,REDUCE,TRACK,ABOVE>::PhysicalCopyAnalyzer(
                                               const FieldMask &mask,
                                               ReductionOpID r,
                                               std::set<Event> &wait, 
                                               int c, 
                                               RegionTreeNode *node)
      : copy_mask(mask), redop(r), local_color(c), 
        logical_node(node), wait_on(wait)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!ABOVE || (local_color >= 0));
      assert(!ABOVE || (logical_node != NULL));
#endif
    }

    //--------------------------------------------------------------------------
    template<bool READING, bool REDUCE, bool TRACK, bool ABOVE>
    bool PhysicalCopyAnalyzer<READING,REDUCE,TRACK,ABOVE>::analyze(
                                                      const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      if (READING)
      {
        if (IS_READ_ONLY(user.usage))
        {
          if (TRACK)
            non_dominated |= (user.field_mask & copy_mask);
          return true;
        }
        // Note this is enough to guarantee
        if (ABOVE)
        {
          if (user.child == local_color)
          {
            if (TRACK)
              non_dominated |= (user.field_mask & copy_mask);
            return true;
          }
          if ((user.child >= 0) &&
              logical_node->are_children_disjoint(unsigned(local_color),
                                                  unsigned(user.child)))
          {
            if (TRACK)
              non_dominated |= (user.field_mask & copy_mask);
            return true;
          }
        }
        // Otherwise register a dependence
        wait_on.insert(user.term_event);
        return true;
      }
      else if (REDUCE)
      {
        if (IS_REDUCE(user.usage) && (user.usage.redop == redop))
        {
          if (TRACK)
            non_dominated |= (user.field_mask & copy_mask);
          return true;
        }
        if (ABOVE)
        {
          if (user.child == local_color)
          {
            if (TRACK)
              non_dominated |= (user.field_mask & copy_mask);
            return true;
          }
          if ((user.child >= 0) && 
              logical_node->are_children_disjoint(unsigned(local_color),
                                                  unsigned(user.child)))
          {
            if (TRACK)
              non_dominated |= (user.field_mask & copy_mask);
            return true;
          }
        }
        // Otherwise register a dependence
        wait_on.insert(user.term_event);
        return true;
      }
      else
      {
        if (ABOVE)
        {
          if (user.child == local_color)
          {
            if (TRACK)
              non_dominated |= (user.field_mask & copy_mask);
            return true;
          }
          if ((user.child >= 0) && 
              logical_node->are_children_disjoint(unsigned(local_color),
                                                  unsigned(user.child)))
          {
            if (TRACK)
              non_dominated |= (user.field_mask & copy_mask);
            return true;
          }
        }
        // Register a dependence
        wait_on.insert(user.term_event);
        return true;
      }
    }

    //--------------------------------------------------------------------------
    template<bool ABOVE>
    WARAnalyzer<ABOVE>::WARAnalyzer(int color/*=-1*/, 
                                    RegionTreeNode *node/*= NULL*/)
      : local_color(color), logical_node(node), has_war(false)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!ABOVE || (local_color >= 0));
      assert(!ABOVE || (logical_node != NULL));
#endif
    }

    //--------------------------------------------------------------------------
    template<bool ABOVE>
    bool WARAnalyzer<ABOVE>::analyze(const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      if (has_war)
        return true;
      if (ABOVE)
      {
        if (local_color == user.child)
          return true;
        if ((user.child >= 0) &&
            logical_node->are_children_disjoint(unsigned(local_color),
                                                unsigned(user.child)))
          return true;
      }
      has_war = IS_READ_ONLY(user.usage);
      return true;
    }

    //--------------------------------------------------------------------------
    PhysicalUnpacker::PhysicalUnpacker(FieldSpaceNode *node,
                                       AddressSpaceID src,
                                       std::map<Event,FieldMask> &events)
      : field_node(node), source(src), 
        deferred_events(events), reinsert_count(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PhysicalUnpacker::begin_node(FieldTree<PhysicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Translate the field mask
      field_node->transform_field_mask(node->local_mask, source);
      // Save the reinsert count from the next level up
      reinsert_stack.push_back(reinsert_count);
      reinsert_count = 0;
    }

    //--------------------------------------------------------------------------
    void PhysicalUnpacker::end_node(FieldTree<PhysicalUser> *node)
    //--------------------------------------------------------------------------
    {
      // Reinsert any users from this node that 
      for (unsigned idx = 0; idx < reinsert_count; idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!reinsert.empty());
#endif
        // Don't recurse when inserting as it may lead to 
        // long chains of unused nodes in the field tree
        node->insert(reinsert.back(), false/*recurse*/);
        reinsert.pop_back();
      }
      // Then restore the reinsert count from the next level up
      reinsert_count = reinsert_stack.back();
      reinsert_stack.pop_back();
    }

    //--------------------------------------------------------------------------
    bool PhysicalUnpacker::analyze(PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      // Note transforming the field mask doesn't change it's
      // precision unless it is a single field
      field_node->transform_field_mask(user.field_mask, source);
      if (FieldMask::pop_count(user.field_mask) == 1)
      {
        reinsert.push_back(user);
        reinsert_count++;
        return false;
      }
      else
        return true; // keep it
    }

    /////////////////////////////////////////////////////////////
    // Region Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeNode::RegionTreeNode(RegionTreeForest *ctx, 
                                   FieldSpaceNode *column_src)
      : context(ctx), column_source(column_src)
#ifdef DEBUG_HIGH_LEVEL
        , logical_state_size(0), physical_state_size(0)
#endif
    //--------------------------------------------------------------------------
    {
      this->node_lock = Reservation::create_reservation(); 
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::~RegionTreeNode(void)
    //--------------------------------------------------------------------------
    {
      node_lock.destroy_reservation();
      node_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::reserve_contexts(unsigned num_contexts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(this->context, RESERVE_CONTEXTS_CALL);
#endif
      // Hold the lock to prevent races on multiple people
      // trying to update the reserve size.
      // Also since deques don't copy objects when
      // appending new ones, we can add states without affecting the
      // already existing ones.
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_states.size() <= num_contexts);
      assert(physical_states.size() <= num_contexts);
      assert(logical_states.size() == physical_states.size());
      assert(num_contexts >= logical_states.size());
      assert(num_contexts >= physical_states.size());
#endif
      logical_states.append(num_contexts);
      physical_states.append(num_contexts);
#ifdef DEBUG_HIGH_LEVEL
      for (unsigned idx = physical_state_size; 
            idx < (physical_state_size+num_contexts); idx++)
        physical_states[idx] = PhysicalState(idx, this);
#else
      for (unsigned idx = physical_states.size()-num_contexts; 
            idx < physical_states.size(); idx++)
        physical_states[idx] = PhysicalState(idx);
#endif
#ifdef DEBUG_HIGH_LEVEL
      logical_state_size = logical_states.size();
      physical_state_size = physical_states.size();
#endif
    }

    //--------------------------------------------------------------------------
    LogicalState& RegionTreeNode::get_logical_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      return logical_states[ctx];
    }

    //--------------------------------------------------------------------------
    PhysicalState& RegionTreeNode::acquire_physical_state(ContextID ctx,
                                                          bool exclusive)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < physical_state_size);
#endif
      PhysicalState &result = physical_states[ctx];
      acquire_physical_state(result, exclusive);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::acquire_physical_state(PhysicalState &state,
                                                 bool exclusive)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ACQUIRE_PHYSICAL_STATE_CALL);
#endif
      Event wait_event = Event::NO_EVENT;
      {
        AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(state.node == this);
#endif
        // Check to see if it has already been acquired
        if (state.acquired_count > 0)
        {
          // If they are both not exclusive and there
          // is nobody in the queue ahead of us, then we can share
          // otherwise we need to wait
          if (state.requests.empty() && !exclusive && !state.exclusive)
          {
            // Update the acquisition count
            state.acquired_count++;
          }
          else
          {
            // Otherwise we ne need to wait 
            // Make a user event, put it on the list of pending
            // requests and then wait.  When we get woken up
            // we will have already been added to the list of 
            // acquistions by the thread who woke us up.
            UserEvent ready_event = UserEvent::create_user_event();
            state.requests.push_back(
                std::pair<UserEvent,bool>(ready_event, exclusive));
            // Can't go to sleep holding the lock so
            // set the wait_event then release the lock
            wait_event = ready_event;
          }
        }
        else
        {
          // Mark that we've acquired it in our mode
          state.acquired_count++;
          state.exclusive = exclusive;
        }
      }
      // See if we need to wait
      if (wait_event.exists())
        wait_event.wait(true/*block*/);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::release_physical_state(PhysicalState &state)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, RELEASE_PHYSICAL_STATE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(state.acquired_count > 0);
#endif
      state.acquired_count--;
      // Get the old exlusivity state
      bool result = state.exclusive;
      // Now see if we can wake any body up
      if ((state.acquired_count == 0) && !state.requests.empty())
      {
        // Set the new state, update the acquisition
        // count and then trigger the event
        state.exclusive = state.requests.front().second;
        state.acquired_count = 1;
        state.requests.front().first.trigger();
        state.requests.pop_front();
        // If it is not exclusive see how many other people
        // we can wake up
        if (!state.exclusive)
        {
          while (!state.requests.empty() &&
                 !state.requests.front().second)
          {
            state.acquired_count++;
            state.requests.front().first.trigger();
            state.requests.pop_front();
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_node(ContextID ctx,
                                               const LogicalUser &user,
                                               RegionTreePath &path,
                                               const bool already_traced)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REGISTER_LOGICAL_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];
      unsigned depth = get_depth();
      // Before we start, record the "before" versions
      // of all our fields
      record_field_versions(state,path,user.field_mask,depth,true/*before*/);
      if (!path.has_child(depth))
      {
        // Don't need to do these things if we've already traced
        if (!already_traced)
        {
          // We've arrived at our destination node
          FieldMask dominator_mask = perform_dependence_checks(user,
                    state.curr_epoch_users, user.field_mask, true/*validates*/);
          FieldMask non_dominated_mask = user.field_mask - dominator_mask;
          // For the fields that weren't dominated, we have to check
          // those fields against the previous epoch's users
          if (!!non_dominated_mask)
          {
            perform_dependence_checks(user,state.prev_epoch_users,
                                      non_dominated_mask, true/*validates*/);
          }
          // Update the dominated fields
          if (!!dominator_mask)
          {
            // Dominator mask is not empty
            // Mask off all the dominated fields from the previous set
            // of epoch users and remove any previous epoch users
            // that were totally dominated
            filter_prev_epoch_users(state, dominator_mask); 
            // Mask off all dominated fields from current epoch users and move
            // them to prev epoch users.  If all fields masked off, then remove
            // them from the list of current epoch users.
            filter_curr_epoch_users(state, dominator_mask); 
            // Finally remove any close operations which have now been dominated
            filter_close_operations(state, dominator_mask); 
            // Update the version IDs of any fields which we dominated
            advance_field_versions(state, dominator_mask); 
          }
        }
        // Now close up any children which we may have dependences on below
        LogicalCloser closer(ctx, user, true/*validates*/);
        // If we are in read-only or reduce mode then we need to record
        // the close operation so that it gets done by the first user
        // otherwise read-write will invalidate its state and write over
        // it so we don't need to have consensus over who does the close
        siphon_logical_children(closer, state, user.field_mask, 
            IS_READ_ONLY(user.usage) || IS_REDUCE(user.usage)/*record*/);
        // Update the list of closed users and close operations
        if (!closer.closed_users.empty())
        {
          // Add the closed users to the prev epoch users, we already
          // registered mapping dependences on them as part of the
          // closing process so we don't need to do it again
#ifndef LOGICAL_FIELD_TREE
          state.prev_epoch_users.insert(state.prev_epoch_users.end(),
                                        closer.closed_users.begin(),
                                        closer.closed_users.end());
#else
          for (std::deque<LogicalUser>::const_iterator it = 
                closer.closed_users.begin(); it != 
                closer.closed_users.end(); it++)
          {
            state.prev_epoch_users->insert(*it);
          }
#endif
        }
        if (!closer.close_operations.empty())
          update_close_operations(state, closer.close_operations);
        // Record any close operations that need to be done
        record_close_operations(state, path, user.field_mask, depth); 
        // No need to do this if we've already traced
        if (!already_traced)
        {
          // Record a mapping reference on this operation
          user.op->add_mapping_reference(user.gen);
#ifndef LOGICAL_FIELD_TREE
          // Add ourselves to the current epoch
          state.curr_epoch_users.push_back(user);
#else
          state.curr_epoch_users->insert(user);
#endif
        }
      }
      else
      {
        // No need to do this if we've already traced
        if (!already_traced)
        {
          // First perform dependence checks on the current and 
          // previous epoch users since we're still traversing
          perform_dependence_checks(user, state.curr_epoch_users, 
                                    user.field_mask, false/*validates*/);
          perform_dependence_checks(user, state.prev_epoch_users,
                                    user.field_mask, false/*validates*/);
        }
        // Otherwise see what we have to close up and
        // then continue the traversal
        // Close up any children that are open which we
        // will have a dependence on
        Color next_child = path.get_child(depth);
        LogicalCloser closer(ctx, user, false/*validates*/);
        // This also updates the new states
        bool open_only = siphon_logical_children(closer, state, 
                          user.field_mask, true/*record*/, next_child);

        // Filter all the close field users
        if (!!closer.closed_mask)
        {
          filter_prev_epoch_users(state, closer.closed_mask);
          filter_curr_epoch_users(state, closer.closed_mask);
          filter_close_operations(state, closer.closed_mask);
          // Advance the field versions for the closed fields
          advance_field_versions(state, closer.closed_mask);
        }

        if (!closer.closed_users.empty())
        {
          // Add the closed users to the prev epoch users, we already
          // registered mapping dependences on them as part of the
          // closing process so we don't need to do it again
#ifndef LOGICAL_FIELD_TREE
          state.prev_epoch_users.insert(state.prev_epoch_users.end(),
                                        closer.closed_users.begin(),
                                        closer.closed_users.end());
#else
          for (std::deque<LogicalUser>::const_iterator it = 
                closer.closed_users.begin(); it != 
                closer.closed_users.end(); it++)
          {
            state.prev_epoch_users->insert(*it);
          }
#endif
        }
        
        if (!closer.close_operations.empty())
          update_close_operations(state, closer.close_operations);
        // Also register any close operations which need to be done
        record_close_operations(state, path, user.field_mask, depth);
        
        RegionTreeNode *child = get_tree_child(next_child);
        if (open_only)
          child->open_logical_node(ctx, user, path, already_traced);
        else
          child->register_logical_node(ctx, user, path, already_traced);
      }
      record_field_versions(state,path,user.field_mask,depth,false/*before*/); 
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::open_logical_node(ContextID ctx,
                                             const LogicalUser &user,
                                             RegionTreePath &path,
                                             const bool already_traced)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, OPEN_LOGICAL_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];
      unsigned depth = get_depth();
      // Before we start, record the "before" versions
      // of all our fields
      record_field_versions(state,path,user.field_mask,depth,true/*before*/);
      if (!path.has_child(depth))
      {
        // No need to record ourselves if we've already traced
        if (!already_traced)
        {
          // We've arrived where we're going,
          // add ourselves as a user
          // Record a mapping reference on this operation
          user.op->add_mapping_reference(user.gen);
#ifndef LOGICAL_FIELD_TREE
          state.curr_epoch_users.push_back(user);
#else
          state.curr_epoch_users->insert(user);
#endif
        }
      }
      else
      {
        Color next_child = path.get_child(depth);
        // Update our field states
        merge_new_field_state(state, 
                              FieldState(user, user.field_mask, next_child));
#ifdef DEBUG_HIGH_LEVEL
        sanity_check_logical_state(state);
#endif
        // Then continue the traversal
        RegionTreeNode *child_node = get_tree_child(next_child);
        child_node->open_logical_node(ctx, user, path, already_traced);
      }
      record_field_versions(state,path,user.field_mask,depth,false/*before*/);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_logical_node(LogicalCloser &closer,
                                            const FieldMask &closing_mask,
                                            bool permit_leave_open)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, CLOSE_LOGICAL_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(closer.ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[closer.ctx];

      // Perform closing checks on both the current epoch users
      // as well as the previous epoch users
      perform_closing_checks(closer, state.curr_epoch_users, closing_mask);
      perform_closing_checks(closer, state.prev_epoch_users, closing_mask);
      
      // Filter out any close operations being done on the closed fields
      filter_close_operations(state, closing_mask);
      // Advance the versions of all the closed fields
      advance_field_versions(state, closing_mask);
      // Recursively traverse any open children and close them as well
      std::deque<FieldState> new_states;
      for (std::list<FieldState>::iterator it = state.field_states.begin();
            it != state.field_states.end(); /*nothing*/)
      {
        if (it->valid_fields * closing_mask)
        {
          it++;
          continue;
        }
        // Recursively perform any close operations
        FieldMask already_open;
        perform_close_operations(closer, closing_mask, *it, -1/*next child*/,
                                 false/*allow next*/, false/*upgrade*/,
                                 permit_leave_open,
                                 false/*record close operations*/,
                                 new_states, already_open);
        // Remove the state if it is now empty
        if (!it->valid_fields)
          it = state.field_states.erase(it);
        else
          it++;
      }
      // Merge any new field states
      merge_new_field_states(state, new_states);
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::siphon_logical_children(LogicalCloser &closer,
                                                 LogicalState &state,
                                                 const FieldMask &current_mask,
                                                 bool record_close_operations,
                                                 int next_child /*= -1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, SIPHON_LOGICAL_CHILDREN_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_logical_state(state);
#endif
      FieldMask open_mask = current_mask;
      std::deque<FieldState> new_states;

      for (std::list<FieldState>::iterator it = state.field_states.begin();
            it != state.field_states.end(); /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields * current_mask)
        {
          it++;
          continue;
        }
        // Now check the current state
        switch (it->open_state)
        {
          case OPEN_READ_ONLY:
            {
              std::map<Color,FieldMask>::const_iterator finder = 
                    it->open_children.find(unsigned(next_child));
              if (IS_READ_ONLY(closer.user.usage))
              {
                // Everything is read-only
                // See if the child that we want is already open
                if (next_child > -1)
                {
                  if (finder != it->open_children.end())
                  {
                    // Remove the child's open fields from the
                    // list of fields we need to open
                    open_mask -= finder->second;
                  }
                }
                it++;
              }
              else
              {
                // Not read-only
                // Close up all the open partitions except the one
                // we want to go down, make a new state to be added
                // containing the fields that are still open and mark
                // that we need an upgrade from read-only to some
                // kind of write operation. We only allow
                const bool needs_upgrade = HAS_WRITE(closer.user.usage);
                FieldMask already_open;
                perform_close_operations(closer, current_mask, *it, next_child,
                                         (it->open_children.size() == 1),
                                         needs_upgrade, 
                                         false/*permit leave open*/,
                                         record_close_operations,
                                         new_states, already_open);
                // Update the open mask
                open_mask -= already_open;
                if (needs_upgrade)
                  new_states.push_back(
                      FieldState(closer.user, already_open, next_child));
                // See if there are still any valid open fields
                if (!it->valid_fields)
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
              perform_close_operations(closer, current_mask, *it, next_child,
                                       true/*allow next*/,
                                       false/*needs upgrade*/,
                                       IS_READ_ONLY(closer.user.usage),
                                       record_close_operations,
                                       new_states, already_open);
              // Update the open mask
              open_mask -= already_open;
              if (!it->valid_fields)
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_SINGLE_REDUCE:
            {
              // Check to see if we have a child we want to go down
              if (next_child > -1)
              {
                // There are four cases here:
                //   1. Same reduction, same child -> everything stays the same
                //   2. Same reduction, different child -> go to MULTI_REDUCE
                //   3. Diff operation, same child -> go to READ_WRITE
                //   4. Diff operation, diff child -> close everything up
                if (IS_REDUCE(closer.user.usage) && 
                    (it->redop == closer.user.usage.redop))
                {
                  // Cases 1 and 2
                  bool needs_recompute = false;
                  std::vector<Color> to_delete;
                  // Go through all the children and see if there is any overlap
                  for (std::map<Color,FieldMask>::iterator cit = 
                        it->open_children.begin(); cit !=
                        it->open_children.end(); cit++)
                  {
                    FieldMask already_open = cit->second & current_mask;
                    // If disjoint children, nothing to do
                    if (!already_open || 
                        are_children_disjoint(cit->first, unsigned(next_child)))
                      continue;
                    // Remove the already open fields from this open_mask
                    // since either they are already open for the right child
                    // or we're going to mark them open in a new FieldState
                    open_mask -= already_open;
                    // Case 2
                    if (cit->first != unsigned(next_child))
                    {
                      // Different child so we need to create a new
                      // FieldState in MULTI_REDUCE mode with two
                      // children open
                      FieldState new_state(closer.user,already_open,cit->first);
                      // Add the next child as well
                      new_state.open_children[unsigned(next_child)] = 
                        already_open;
                      new_state.open_state = OPEN_MULTI_REDUCE;
                      new_states.push_back(new_state);
                      // Update the current child, mark that we need to
                      // recompute the valid fields for the state
                      cit->second -= already_open;
                      if (!cit->second)
                        to_delete.push_back(cit->first);
                      needs_recompute = true;
                    }
                    // Otherwise same child so case 1 and everything just
                    // stays in SINGLE_REDUCE_MODE
                  }
                  // See if we need to recompute any properties
                  // of the current state to see if they are still valid
                  if (needs_recompute)
                  {
                    // Remove all the empty children
                    for (std::vector<Color>::const_iterator cit = 
                          to_delete.begin(); cit != to_delete.end(); cit++)
                    {
                      std::map<Color,FieldMask>::iterator finder = 
                        it->open_children.find(*cit);
#ifdef DEBUG_HIGH_LEVEL
                      assert(finder != it->open_children.end());
                      assert(!finder->second);
#endif
                      it->open_children.erase(finder);
                    }
                    // Then recompute the valid mask for the current state
                    FieldMask new_valid_mask;
                    for (std::map<Color,FieldMask>::const_iterator cit = 
                          it->open_children.begin(); cit !=
                          it->open_children.end(); cit++)
                    {
#ifdef DEBUG_HIGH_LEVEL
                      assert(!!cit->second);
#endif
                      new_valid_mask |= cit->second;
                    }
                    // Update the valid mask on the field state, we'll
                    // check to see fi we need to delete it at the end
                    it->valid_fields = new_valid_mask;
                  }
                }
                else
                {
                  // Cases 3 and 4
                  FieldMask already_open;
                  perform_close_operations(closer, current_mask, *it, 
                                           next_child, 
                                           true/*allow next*/,
                                           true/*needs upgrade*/,
                                           false/*permit leave open*/,
                                           record_close_operations,
                                           new_states, already_open);
                  open_mask -= already_open;
                  if (!!already_open)
                  {
                    // Create a new FieldState open in whatever mode is
                    // appropriate based on the usage
                    FieldState new_state(closer.user, already_open, 
                                         unsigned(next_child));
                    // Note if it is another reduction in the same child
                    if (IS_REDUCE(closer.user.usage))
                      new_state.open_state = OPEN_READ_WRITE;
                    new_states.push_back(new_state);
                  }
                }
              }
              else
              {
                // Closing everything up, so just do it
                FieldMask already_open;
                perform_close_operations(closer, current_mask, *it, next_child,
                                         false/*allow next*/,
                                         false/*needs upgrade*/,
                                         false/*permit leave open*/,
                                         record_close_operations,
                                         new_states, already_open);
                open_mask -= already_open;
              }
              // Now see if the current field state is still valid
              if (!it->valid_fields)
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_MULTI_REDUCE:
            {
              // See if this reduction is a reduction of the same kind
              if (IS_REDUCE(closer.user.usage) &&
                  (closer.user.usage.redop == it->redop))
              {
                if (next_child > -1)
                {
                  std::map<Color,FieldMask>::const_iterator finder = 
                    it->open_children.find(unsigned(next_child));
                  if (finder != it->open_children.end())
                  {
                    // Already open, so remove the open fields
                    open_mask -= (finder->second & current_mask);
                  }
                }
                it++;
              }
              else
              {
                // Need to close up the open field since we're going
                // to have to do it anyway
                FieldMask already_open;
                perform_close_operations(closer, current_mask, *it, next_child,
                                         false/*allow next child*/,
                                         false/*needs upgrade*/,
                                         false/*permit leave open*/,
                                         record_close_operations,
                                         new_states, already_open);
#ifdef DEBUG_HIGH_LEVEL
                assert(!already_open); // should all be closed now
#endif
                if (!it->valid_fields)
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
      // If we had any fields that still need to be opened, create
      // a new field state and add it into the set of new states
      if ((next_child > -1) && !!open_mask)
        new_states.push_back(FieldState(closer.user, open_mask, next_child));
      merge_new_field_states(state, new_states);
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_logical_state(state);
#endif
      // Return true if we are opening all the fields
      return (open_mask == current_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_close_operations(LogicalCloser &closer,
                                            const FieldMask &closing_mask,
                                            FieldState &state,
                                            int next_child, 
                                            bool allow_next_child,
                                            bool upgrade_next_child,
                                            bool permit_leave_open,
                                            bool record_close_operations,
                                            std::deque<FieldState> &new_states,
                                            FieldMask &already_open)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_LOGICAL_CLOSE_CALL);
#endif
      // First, if we have a next child and we know all pairs of children
      // are disjoint, then we can skip a lot of this
      bool removed_fields = false;
      if ((next_child > -1) && are_all_children_disjoint())
      {
        // Check to see if we have anything to close
        std::map<Color,FieldMask>::iterator finder = 
            state.open_children.find(next_child);
        if (finder != state.open_children.end())
        {
          FieldMask close_mask = finder->second & closing_mask;
          if (!!close_mask)
          {
            if (allow_next_child)
            {
              already_open |= close_mask;
              if (upgrade_next_child)
              {
                finder->second -= close_mask;
                removed_fields = true;
                if (!finder->second)
                  state.open_children.erase(finder);
              }
            }
            else
            {
              // Otherwise we actually need to do the close
              RegionTreeNode *child_node = get_tree_child(finder->first);
              child_node->close_logical_node(closer, close_mask, 
                                             permit_leave_open);
              if (record_close_operations)
                closer.close_operations.push_back(TreeClose(
                      legion_new<TreeCloseImpl>(finder->first, close_mask, 
                                        permit_leave_open)));
              // Remove the closed fields
              finder->second -= close_mask;
              removed_fields = true;
              // Record any fields that we closed
              if (record_close_operations)
                closer.closed_mask |= close_mask;
              if (permit_leave_open)
              {
#ifdef DEBUG_HIGH_LEVEL
                assert(IS_READ_ONLY(closer.user.usage));
#endif
                new_states.push_back(FieldState(closer.user,
                                  close_mask, finder->first));
              }
              if (!finder->second)
                state.open_children.erase(finder);
            }
          }
          // Otherwise disjoint fields, nothing to do
        }
        // Otherwise it's closed so it doesn't matter
      }
      else
      {
        std::vector<Color> to_delete;
        // Go through and close all the children which we overlap with
        // and aren't the next child that we're going to use
        for (std::map<Color,FieldMask>::iterator it = 
              state.open_children.begin(); it != 
              state.open_children.end(); it++)
        {
          FieldMask close_mask = it->second & closing_mask;
          // check for field disjointness
          if (!close_mask)
            continue;
          // Check for same child, only allow upgrades in some cases
          // such as read-only -> exclusive.  This is calling context
          // sensitive hence the parameter.
          if (allow_next_child && (next_child > -1) && 
              (next_child == int(it->first)))
          {
            FieldMask open_fields = close_mask;
            already_open |= open_fields;
            if (upgrade_next_child)
            {
              it->second -= open_fields;
              removed_fields = true;
              if (!it->second)
                to_delete.push_back(it->first);
              // The upgraded field state gets added by the caller
            }
            continue;
          }
          // Check for child disjointness
          if ((next_child > -1) && 
              are_children_disjoint(it->first, unsigned(next_child)))
            continue;
          // Perform the close operation
          RegionTreeNode *child_node = get_tree_child(it->first);
          child_node->close_logical_node(closer, close_mask, permit_leave_open);
          if (record_close_operations)
            closer.close_operations.push_back(TreeClose(
                  legion_new<TreeCloseImpl>(it->first, close_mask, 
                                            permit_leave_open)));
          // Remove the close fields
          it->second -= close_mask;
          removed_fields = true;
          if (!it->second)
            to_delete.push_back(it->first);
          // Record any fields that we closed
          if (record_close_operations)
            closer.closed_mask |= close_mask;
          // If we're allowed to leave this open, add a new
          // state for the current user
          if (permit_leave_open)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(IS_READ_ONLY(closer.user.usage));
#endif
            new_states.push_back(FieldState(closer.user,close_mask,it->first));
          }
        }
        // Remove the children that can be deleted
        for (std::vector<Color>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          state.open_children.erase(*it);
        }
      }
      // See if it is time to rebuild the valid mask 
      if (removed_fields)
      {
        if (state.rebuild_timeout == 0)
        {
          // Rebuild the valid fields mask
          FieldMask new_valid_mask;
          for (std::map<Color,FieldMask>::const_iterator it = 
                state.open_children.begin(); it != 
                state.open_children.end(); it++)
          {
            new_valid_mask |= it->second;
          }
          state.valid_fields = new_valid_mask;    
          // Reset the timeout to the order of the number of open children
          state.rebuild_timeout = state.open_children.size();
        }
        else
          state.rebuild_timeout--;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_state(LogicalState &state,
                                               const FieldState &new_state)
    //--------------------------------------------------------------------------
    {
      for (std::list<FieldState>::iterator it = state.field_states.begin();
            it != state.field_states.end(); it++)
      {
        if (it->overlaps(new_state))
        {
          it->merge(new_state);
          return;
        }
      }
      // Otherwise just push it on the back
      state.field_states.push_back(new_state);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_states(LogicalState &state,
                                      const std::deque<FieldState> &new_states)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < new_states.size(); idx++)
      {
        const FieldState &next = new_states[idx];
        merge_new_field_state(state, next);
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check_logical_state(state);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_field_versions(LogicalState &state, 
                                               RegionTreePath &path,
                                               const FieldMask &field_mask,
                                               unsigned depth, bool before)
    //--------------------------------------------------------------------------
    {
      if (before)
      {
        for (std::map<VersionID,FieldMask>::const_iterator it = 
              state.field_versions.begin(); it != 
              state.field_versions.end(); it++)
        {
          FieldMask version_mask = field_mask & it->second;
          if (!!version_mask)
            path.record_before_version(depth, it->first, version_mask);
        }
      }
      else
      {
        for (std::map<VersionID,FieldMask>::const_iterator it = 
              state.field_versions.begin(); it != 
              state.field_versions.end(); it++)
        {
          FieldMask version_mask = field_mask & it->second;
          if (!!version_mask)
            path.record_after_version(depth, it->first, version_mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_close_operations(LogicalState &state,
                                                 RegionTreePath &path,
                                                 const FieldMask &field_mask,
                                                 unsigned depth)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, RECORD_CLOSE_CALL);
#endif
      for (std::map<Color,LegionContainer<TreeClose,TREE_CLOSE_ALLOC>::list>::
            const_iterator cit = state.close_operations.begin(); cit !=
            state.close_operations.end(); cit++)
      {
        for (LegionContainer<TreeClose,TREE_CLOSE_ALLOC>::list::const_iterator 
              it = cit->second.begin(); it != cit->second.end(); it++)
        {
          FieldMask close_mask = it->get_logical_mask() & field_mask;
          if (!!close_mask)
            path.record_close_operation(depth, *it, close_mask);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_close_operations(LogicalState &state,
                                   const std::deque<TreeClose> &new_close_infos)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, UPDATE_CLOSE_CALL);
#endif
      // Build a mask for each child of close operations that need to pruned
      std::map<Color,FieldMask> to_prune;
      for (std::deque<TreeClose>::const_iterator cit = new_close_infos.begin();
            cit != new_close_infos.end(); cit++)
      {
        Color child = cit->get_child();
        std::map<Color,FieldMask>::iterator finder = to_prune.find(child);
        if (finder == to_prune.end())
          to_prune[child] = cit->get_logical_mask();
        else
          finder->second |= cit->get_logical_mask();
      }
      // Prune out all the old close operations which are no longer relevant
      for (std::map<Color,FieldMask>::const_iterator pit = to_prune.begin();
            pit != to_prune.end(); pit++)
      {
        LegionContainer<TreeClose,TREE_CLOSE_ALLOC>::list &child_list = 
                                          state.close_operations[pit->first];
        for (std::list<TreeClose>::iterator it = child_list.begin();
              it != child_list.end(); /*nothing*/)
        {
          FieldMask &mask = it->get_logical_mask();
          mask -= pit->second;
          if (!mask)
            it = child_list.erase(it);
          else
            it++;
        }
      }
      // Now add in our child references
      for (std::deque<TreeClose>::const_iterator cit = new_close_infos.begin();
            cit != new_close_infos.end(); cit++)
      {
        state.close_operations[cit->get_child()].push_back(*cit);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::advance_field_versions(LogicalState &state,
                                                const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADVANCE_FIELD_CALL);
#endif
      std::map<VersionID,FieldMask> new_versions;
      std::vector<VersionID> to_delete;
      for (std::map<VersionID,FieldMask>::iterator it = 
            state.field_versions.begin(); it !=
            state.field_versions.end(); it++)
      {
        FieldMask overlap = it->second & field_mask;
        if (!!overlap)
        {
          it->second -= field_mask;
          new_versions[(it->first+1)] = overlap;
          if (!it->second)
            to_delete.push_back(it->first);
        }
      }
      // Delete any versions that need to be deleted before adding new versions
      for (std::vector<VersionID>::const_iterator it = 
            to_delete.begin(); it != to_delete.end(); it++)
      {
        state.field_versions.erase(*it);
      }
      // Update the versions with the new versions
      for (std::map<VersionID,FieldMask>::const_iterator it = 
            new_versions.begin(); it != new_versions.end(); it++)
      {
        std::map<VersionID,FieldMask>::iterator finder = 
          state.field_versions.find(it->first);
        if (finder == state.field_versions.end())
          state.field_versions.insert(*it);
        else
          finder->second |= it->second;
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_prev_epoch_users(LogicalState &state,
                                                 const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FILTER_PREV_EPOCH_CALL);
#endif
#ifndef LOGICAL_FIELD_TREE
      for (std::list<LogicalUser>::iterator it = 
            state.prev_epoch_users.begin(); it != 
            state.prev_epoch_users.end(); /*nothing*/)
      {
        it->field_mask -= field_mask;
        if (!it->field_mask)
        {
          // Remove the mapping reference
          it->op->remove_mapping_reference(it->gen);
          it = state.prev_epoch_users.erase(it); // empty so erase it
        }
        else
          it++; // still has non-dominated fields
      }
#else
      LogicalFilter filter(field_mask);
      state.prev_epoch_users->analyze<LogicalFilter>(field_mask, filter);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_curr_epoch_users(LogicalState &state,
                                                 const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FILTER_CURR_EPOCH_CALL);
#endif
#ifndef LOGICAL_FIELD_TREE
      for (std::list<LogicalUser>::iterator it = 
              state.curr_epoch_users.begin(); it !=
              state.curr_epoch_users.end(); /*nothing*/)
      {
        FieldMask local_dom = it->field_mask & field_mask;
        if (!!local_dom)
        {
          // Move a copy over to the previous epoch users for
          // the fields that were dominated
          state.prev_epoch_users.push_back(*it);
          state.prev_epoch_users.back().field_mask = local_dom;
          // Add a mapping reference
          it->op->add_mapping_reference(it->gen);
        }
        else
        {
          it++;
          continue;
        }
        // Update the field mask with the non-dominated fields
        it->field_mask -= field_mask;
        if (!it->field_mask)
        {
          // Remove the mapping reference
          it->op->remove_mapping_reference(it->gen);
          it = state.curr_epoch_users.erase(it); // empty so erase it
        }
        else
          it++; // not empty so keep going
      }
#else
      LogicalFilter filter(field_mask, state.prev_epoch_users);
      state.curr_epoch_users->analyze<LogicalFilter>(field_mask, filter);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::filter_close_operations(LogicalState &state,
                                                 const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FILTER_CLOSE_CALL);
#endif
      for (std::map<Color,LegionContainer<TreeClose,TREE_CLOSE_ALLOC>::list>::
            iterator cit = state.close_operations.begin(); cit !=
            state.close_operations.end(); cit++)
      {
        for (LegionContainer<TreeClose,TREE_CLOSE_ALLOC>::list::iterator it = 
              cit->second.begin(); it != cit->second.end(); /*nothing*/)
        {
          FieldMask &mask = it->get_logical_mask();
          mask -= field_mask;
          if (!mask)
            it = cit->second.erase(it);
          else
            it++;
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::sanity_check_logical_state(LogicalState &state)
    //--------------------------------------------------------------------------
    {
      // For every child and every field, it should only be open in one mode
      std::map<Color,FieldMask> previous_children;
      for (std::list<FieldState>::const_iterator fit = 
            state.field_states.begin(); fit != 
            state.field_states.end(); fit++)
      {
        FieldMask actually_valid;
        for (std::map<Color,FieldMask>::const_iterator it = 
              fit->open_children.begin(); it != 
              fit->open_children.end(); it++)
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
        // Actually valid should be greater than or equal
        assert(!(actually_valid - fit->valid_fields));
      }
      // Also check that for each field it is either only open in one mode
      // or two children in different modes are disjoint
      for (std::list<FieldState>::const_iterator it1 = 
            state.field_states.begin(); it1 != 
            state.field_states.end(); it1++)
      {
        for (std::list<FieldState>::const_iterator it2 = 
              state.field_states.begin(); it2 != 
              state.field_states.end(); it2++)
        {
          // No need to do comparisons if they are the same field state
          if (it1 == it2) 
            continue;
          const FieldState &f1 = *it1;
          const FieldState &f2 = *it2;
          for (std::map<Color,FieldMask>::const_iterator cit1 = 
                f1.open_children.begin(); cit1 != 
                f1.open_children.end(); cit1++)
          {
            for (std::map<Color,FieldMask>::const_iterator cit2 = 
                  f2.open_children.begin(); cit2 != 
                  f2.open_children.end(); cit2++)
            {
              
              // Disjointness check on fields
              if (cit1->second * cit2->second)
                continue;
              Color c1 = cit1->first;
              Color c2 = cit2->first;
              // Some aliasing in the fields, so do the check 
              // for child disjointness
              assert(c1 != c2);
              assert(are_children_disjoint(c1, c2));
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::initialize_logical_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, INITIALIZE_LOGICAL_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];
#ifdef DEBUG_HIGH_LEVEL
      // Technically these should already be empty
      assert(state.field_versions.size() == 1);
      assert(state.field_states.empty());
#ifndef LOGICAL_FIELD_TREE
      assert(state.curr_epoch_users.empty());
      assert(state.prev_epoch_users.empty());
#endif
      assert(state.close_operations.empty());
#endif
      state.reset();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_logical_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, INVALIDATE_LOGICAL_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];     
#ifndef LOGICAL_FIELD_TREE
      for (std::list<LogicalUser>::const_iterator it = 
            state.curr_epoch_users.begin(); it != 
            state.curr_epoch_users.end(); it++)
      {
        it->op->remove_mapping_reference(it->gen); 
      }
      for (std::list<LogicalUser>::const_iterator it = 
            state.prev_epoch_users.begin(); it != 
            state.prev_epoch_users.end(); it++)
      {
        it->op->remove_mapping_reference(it->gen); 
      }
#else
      LogicalFieldInvalidator invalidator;
      FieldMask all_ones(FIELD_ALL_ONES);
      state.curr_epoch_users->
        analyze<LogicalFieldInvalidator>(all_ones, invalidator);
      state.prev_epoch_users->
        analyze<LogicalFieldInvalidator>(all_ones, invalidator);
#endif
      state.reset();
    }

    //--------------------------------------------------------------------------
    template<bool DOMINATE>
    void RegionTreeNode::register_logical_dependences(ContextID ctx, 
                                    Operation *op, const FieldMask &field_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REGISTER_LOGICAL_DEPS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];
#ifndef LOGICAL_FIELD_TREE
      for (std::list<LogicalUser>::iterator it = 
            state.curr_epoch_users.begin(); it != 
            state.curr_epoch_users.end(); /*nothing*/)
      {
        if (!(it->field_mask * field_mask))
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_mapping_dependence(
              Machine::get_executing_processor(),
              op->get_parent()->get_unique_task_id(),
              it->uid, it->idx, op->get_unique_op_id(),
              0/*idx*/, TRUE_DEPENDENCE);
#endif
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
              op->get_parent()->get_unique_task_id(),
              it->uid, it->idx, op->get_unique_op_id(),
              0/*idx*/, TRUE_DEPENDENCE);
#endif
          // Do this after the logging since we 
          // are going to update the iterator
          if (op->register_dependence(it->op, it->gen))
          {
            // Prune it from the list
            it = state.curr_epoch_users.erase(it);
          }
          else if (DOMINATE)
            it = state.curr_epoch_users.erase(it);
          else
            it++;
        }
        else if (DOMINATE)
          it = state.curr_epoch_users.erase(it);
        else
          it++;
      }
      for (std::list<LogicalUser>::iterator it = 
            state.prev_epoch_users.begin(); it != 
            state.prev_epoch_users.end(); /*nothing*/)
      {
        if (!(it->field_mask * field_mask))
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_mapping_dependence(
              Machine::get_executing_processor(),
              op->get_parent()->get_unique_task_id(),
              it->uid, it->idx, op->get_unique_op_id(),
              0/*idx*/, TRUE_DEPENDENCE);
#endif
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(
              op->get_parent()->get_unique_task_id(),
              it->uid, it->idx, op->get_unique_op_id(), 0/*idx*/, 
              TRUE_DEPENDENCE);
#endif
          // Do this after the logging since we are going
          // to update the iterator
          if (op->register_dependence(it->op, it->gen))
          {
            // Prune it from the list
            it = state.prev_epoch_users.erase(it);
          }
          else if (DOMINATE)
            it = state.prev_epoch_users.erase(it);
          else
            it++;
        }
        else if (DOMINATE)
          it = state.prev_epoch_users.erase(it);
        else
          it++;
      }
#else
      LogicalOpAnalyzer<DOMINATE> analyzer(op);
      state.curr_epoch_users->analyze<LogicalOpAnalyzer<DOMINATE> >(
                                                          field_mask, analyzer);
      state.prev_epoch_users->analyze<LogicalOpAnalyzer<DOMINATE> >(
                                                          field_mask, analyzer);
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_user_coherence(ContextID ctx, 
                                               FieldMask &coherence_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];
      coherence_mask |= state.user_level_coherence;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::acquire_user_coherence(ContextID ctx, 
                                                const FieldMask &coherence_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];
      state.user_level_coherence |= coherence_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::release_user_coherence(ContextID ctx,
                                                const FieldMask &coherence_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < logical_state_size);
#endif
      LogicalState &state = logical_states[ctx];
      state.user_level_coherence -= coherence_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_physical_node(PhysicalCloser &closer,
                                             const FieldMask &closing_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, CLOSE_PHYSICAL_NODE_CALL);
#endif
      // Acquire the physical state of the node to close
      ContextID ctx = closer.info->ctx;
      // Figure out if we have dirty data.  If we do, issue copies back to
      // each of the target instances specified by the closer.  Note we
      // don't need to issue copies if the target view is already in
      // the list of currently valid views.  Then
      // perform the close operation on each of our open partitions that
      // interfere with the closing mask.
      // If there are any dirty fields we have to copy them back

      // Not we only need to do this operation for nodes which are
      // actually regions since they are the only ones that store
      // actual instances.
      
      FieldMask dirty_fields, reduc_fields;
      std::map<InstanceView*,FieldMask> valid_instances;
      std::map<ReductionView*,FieldMask> valid_reductions;
      {
        PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);
        dirty_fields = state.dirty_mask & closing_mask;
        reduc_fields = state.reduction_mask & closing_mask;
        if (is_region())
        {
          if (!!dirty_fields)
          {
            // Pull down instance views so we don't issue unnecessary copies
            pull_valid_instance_views(state, closing_mask);
#ifdef DEBUG_HIGH_LEVEL
            assert(!state.valid_views.empty());
#endif
            find_valid_instance_views(state, closing_mask, closing_mask, 
                                      false/*needs space*/, valid_instances);
          }
          if (!!reduc_fields)
          {
            for (std::map<ReductionView*,FieldMask>::const_iterator it = 
                  state.reduction_views.begin(); it != 
                  state.reduction_views.end(); it++)
            {
              FieldMask overlap = it->second & closing_mask;
              if (!!overlap)
              {
                valid_reductions[it->first] = overlap;
                it->first->add_valid_reference();
              }
            }
          }
        }
        // Invalidate any reduction views we are going to reduce back
        if (!!reduc_fields)
          invalidate_reduction_views(state, reduc_fields);
        release_physical_state(state);
      }
      if (is_region() && !!dirty_fields)
      {
        const std::vector<MaterializedView*> &targets = 
          closer.get_lower_targets();
        for (std::vector<MaterializedView*>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
        {
          std::map<InstanceView*,FieldMask>::const_iterator finder = 
            valid_instances.find(*it);
          // Check to see if it is already a valid instance for some fields
          if (finder == valid_instances.end())
          {
            issue_update_copies(closer.info, *it, 
                                dirty_fields, valid_instances);
          }
          else
          {
            // Only need to issue update copies for dirty fields for which
            // we are not currently valid
            FieldMask diff_fields = dirty_fields - finder->second;
            if (!!diff_fields)
              issue_update_copies(closer.info, *it, 
                                  diff_fields, valid_instances);
          }
        }
      }
      // Now we need to issue close operations for all our children
#ifdef DEBUG_HIGH_LEVEL
      assert(!closer.needs_targets());
#endif
      {
        PhysicalCloser next_closer(closer);
        PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);
        bool create_composite = false;
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        siphon_physical_children(next_closer, state, closing_mask,
                                 -1/*next child*/, create_composite);
#ifdef DEBUG_HIGH_LEVEL
        assert(result); // should always succeed since targets already exist
        assert(!create_composite);
#endif
        // Update the closer's dirty mask
        const FieldMask &dirty_below = next_closer.get_dirty_mask();
        closer.update_dirty_mask(dirty_fields | reduc_fields | dirty_below);

        if (!closer.permit_leave_open)
          invalidate_instance_views(state, closing_mask, 
                                    true/*clean*/, false/*force*/);
        else
        {
          // If we had any dirty fields below that are getting
          // flushed up past this level, then we have to invalidate
          // those instances here.
          if (!!dirty_below)
            invalidate_instance_views(state, dirty_below,
                                      true/*clean*/, false/*force*/);
          state.dirty_mask -= closing_mask;
        }
        // Finally release our hold on the state
        release_physical_state(state);
      }
      // Apply any reductions that we might have for the closing
      // fields back to the target instances
      // Again this only needs to be done for region nodes but
      // we should always invalidate reduction views
      if (is_region() && !!reduc_fields)
      {
        const std::vector<MaterializedView*> &targets = 
          closer.get_lower_targets();
        for (std::vector<MaterializedView*>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
        {
          issue_update_reductions(*it, reduc_fields,
                                  closer.info->local_proc, valid_reductions);
        }
      }
      // Remove any valid references we added to views
      remove_valid_references(valid_instances);
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        if (it->first->remove_valid_reference())
          legion_delete(it->first);
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::select_close_targets(PhysicalCloser &closer,
                                              const FieldMask &closing_mask,
                          const std::map<InstanceView*,FieldMask> &valid_views,
                          std::map<MaterializedView*,FieldMask> &update_views,
                                              bool &create_composite)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, SELECT_CLOSE_TARGETS_CALL);
#endif
      // First get the list of valid instances
      // Get the set of memories for which we have valid instances
      std::set<Memory> valid_memories;
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->is_composite_view())
          continue;
        MaterializedView *view = it->first->as_materialized_view();
        valid_memories.insert(view->get_location());
      }
      // Now ask the mapper what it wants to do
      bool create_one;
      std::set<Memory> to_reuse;
      std::vector<Memory> to_create;
      size_t blocking_factor = 1;
      size_t max_blocking_factor = 
        context->get_domain_volume(closer.handle.index_space); 
      create_composite = context->runtime->invoke_mapper_rank_copy_targets(
                                               closer.info->local_proc, 
                                               closer.info->mappable,
                                               closer.handle,
                                               valid_memories,
                                               true/*complete*/,
                                               max_blocking_factor,
                                               to_reuse,
                                               to_create,
                                               create_one,
                                               blocking_factor);
      // Filter out any re-use memories which are not in the list of
      // valid memories
      if (!to_reuse.empty())
      {
        std::vector<Memory> to_delete;
        for (std::set<Memory>::const_iterator it = to_reuse.begin();
              it != to_reuse.end(); it++)
        {
          if (valid_memories.find(*it) == valid_memories.end())
          {
            log_region(LEVEL_WARNING,"WARNING: memory " IDFMT " was specified "
                                     "to be reused in rank_copy_targets "
                                     "when closing mappable operation ID %lld, "
                                     "but no instance exists in that memory."
                                     "Memory " IDFMT " is being added to the "
                                     "set of create memories.", it->id,
                               closer.info->mappable->get_unique_mappable_id(),
                               it->id);
            // Add it to the list of memories to try creating
            to_create.push_back(*it);
            to_delete.push_back(*it);
          }
        }
        if (!to_delete.empty())
        {
          for (std::vector<Memory>::const_iterator it = to_delete.begin();
                it != to_delete.end(); it++)
          {
            to_reuse.erase(*it);
          }
        }
      }
      // If we're restricted, then we can't make any new instances
      if (closer.info->req.restricted && !to_create.empty())
      {
        log_region(LEVEL_WARNING,"WARNING: Mapper requested creation of new "
                                 "regions in rank_copy_targets when closing "
                                 "mappable operation ID %lld when requirement "
                                 "was restricted.  Request will be ignored.",
                             closer.info->mappable->get_unique_mappable_id());
        to_create.clear();
      }
      // See if the mapper gave us reasonable output
      if (!create_composite && to_reuse.empty() && to_create.empty())
      {
        log_region(LEVEL_ERROR,"Invalid mapper output for rank_copy_targets "
                               "when closing mappable operation ID %lld. "
                               "Must specify at least one target memory in "
                               "'to_reuse' or 'to_create'.",
                           closer.info->mappable->get_unique_mappable_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (create_composite)
      {
        // Return out early marking that we didn't make any instances
        return false;
      }
      else
      {
        // Now process the results
        // First see if we should re-use any instances
        for (std::set<Memory>::const_iterator mit = to_reuse.begin();
              mit != to_reuse.end(); mit++)
        {
          // Make sure it is a valid choice
          if (valid_memories.find(*mit) == valid_memories.end())
            continue;
          MaterializedView *best = NULL;
          FieldMask best_mask;
          int num_valid_fields = -1;
          for (std::map<InstanceView*,FieldMask>::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            if (it->first->is_composite_view())
              continue;
            MaterializedView *current = it->first->as_materialized_view();
            if (current->get_location() != (*mit))
              continue;
            int valid_fields = FieldMask::pop_count(it->second);
            if (valid_fields > num_valid_fields)
            {
              num_valid_fields = valid_fields;
              best = current;
              best_mask = it->second;
            }
          }
          if (best != NULL)
          {
            FieldMask need_update = closing_mask - best_mask;
            if (!!need_update)
              update_views[best] = need_update;
            // Add it to the list of close targets
            closer.add_target(best);
          }
        }
        // Now see if want to try to create any new instances
        for (unsigned idx = 0; idx < to_create.size(); idx++)
        {
          // Try making an instance in memory
          MaterializedView *new_view = 
            create_instance(to_create[idx], 
                            closer.info->req.privilege_fields, 
                            blocking_factor,
                            closer.info->mappable->get_depth());
          if (new_view != NULL)
          {
            // Update all the fields
            update_views[new_view] = closing_mask;
            closer.add_target(new_view);
            // If we only needed to make one, then we are done
            if (create_one)
              break;
          }
        }
      }
      // Check to see if have targets
      return (!closer.needs_targets());
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::siphon_physical_children(PhysicalCloser &closer,
                                              PhysicalState &state,
                                              const FieldMask &closing_mask,
                                              int next_child,
                                              bool &create_composite)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, SIPHON_PHYSICAL_CHILDREN_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      // First check, if all the fields are disjoint, then we're done
      if (state.children.valid_fields * closing_mask)
        return true;
      // Make a copy of the open children map since close_physical_child
      // will release our hold on the lock which may lead to someone
      // else invalidating our iterator.
      std::map<Color,FieldMask> open_copy = state.children.open_children;
      // Otherwise go through all of the children and 
      // see which ones we need to clean up
      for (std::map<Color,FieldMask>::iterator it = open_copy.begin();
            it != open_copy.end(); it++)
      {
        if (!close_physical_child(closer, state, closing_mask,
                             it->first, next_child, create_composite))
          return false;
      }
      // Rebuild the valid mask
      FieldMask next_valid;
      for (std::map<Color,FieldMask>::const_iterator it = 
            state.children.open_children.begin(); it !=
            state.children.open_children.end(); it++)
      {
        next_valid |= it->second;
      }
      state.children.valid_fields = next_valid;
      return true;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::close_physical_child(PhysicalCloser &closer,
                                              PhysicalState &state,
                                              const FieldMask &closing_mask,
                                              Color target_child,
                                              int next_child,
                                              bool &create_composite)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, CLOSE_PHYSICAL_CHILD_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      // See if we can find the child
      std::map<Color,FieldMask>::iterator finder = 
        state.children.open_children.find(target_child);
      if (finder == state.children.open_children.end())
        return true;
      // Check field disjointness
      if (finder->second * closing_mask)
        return true;
      // Check for child disjointness
      if ((next_child >= 0) && 
          are_children_disjoint(finder->first, unsigned(next_child)))
        return true;
      FieldMask close_mask = finder->second & closing_mask;
      // First check to see if the closer needs to make physical
      // instance targets in order to perform the close operation
      std::map<InstanceView*,FieldMask> space_views;
      std::map<InstanceView*,FieldMask> valid_views;
      std::map<MaterializedView*,FieldMask> update_views;
      if (closer.needs_targets())
      {
        // Have the closer make targets and return false indicating
        // we could not successfully perform the close operation
        // if he fails to make them. When making close targets pick
        // them for the full traversal mask so that other close
        // operations can reuse the same physical instances.
        find_valid_instance_views(state, closer.info->traversal_mask, 
                                  closer.info->traversal_mask,
                                  true/*needs space*/, space_views);
        // This doesn't matter so always mark it false for now
        if (!select_close_targets(closer, closer.info->traversal_mask, 
                          space_views, update_views, create_composite))
        {
          remove_valid_references(space_views);
          // We failed to close, time to return
          return false;
        }
        else
        {
          // We succeeded, so get the set of valid views
          // for issuing update copies
          find_valid_instance_views(state, closer.info->traversal_mask,
                                    closer.info->traversal_mask,
                                    false/*needs space*/, valid_views);
        }
      }
      // Need to get this value before the iterator is invalidated
      RegionTreeNode *child_node = get_tree_child(finder->first);
      // Now we need to actually close up this child
      // Mark that when we are done we will have successfully
      // closed up this child.  Do this now before we
      // release the lock and someone invalidates our iterator
      if (!closer.permit_leave_open)
      {
        finder->second -= close_mask;
        if (!finder->second)
          state.children.open_children.erase(finder);
      }
      // Release our lock on the current state before going down
      bool was_exclusive = release_physical_state(state);
      if (!update_views.empty())
      {
        // Issue any update copies, and then release any
        // valid view references that we are holding
        for (std::map<MaterializedView*,FieldMask>::const_iterator it =
              update_views.begin(); it != update_views.end(); it++)
        {
          issue_update_copies(closer.info, it->first, 
                              it->second, valid_views);
        }
        update_views.clear();
      }
      // Now remove any valid references that we have
      remove_valid_references(space_views);
      remove_valid_references(valid_views);
      // Now we're ready to perform the close operation
      closer.close_tree_node(child_node, close_mask);
      // Reacquire our lock on the state upon returning
      acquire_physical_state(state, was_exclusive);
      return true;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_physical_node(CompositeCloser &closer,
                                             CompositeNode *node,
                                             const FieldMask &closing_mask,
                                             FieldMask &dirty_mask,
                                             FieldMask &complete_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, CLOSE_PHYSICAL_NODE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(node->logical_node == this);
#endif
      // Tell the node to update its parent
      node->update_parent_info(closing_mask);
      // Acquire the state and save any pertinent information in the node
      PhysicalState &state = 
                acquire_physical_state(closer.ctx, true/*exlusive*/);
      // First close up any children below and see if we are flushing
      // back any dirty data which is complete
      FieldMask dirty_below, complete_below;
      siphon_physical_children(closer, node, state, closing_mask, 
                               dirty_below, complete_below);
      // Only capture state here for things in our close mask which we
      // don't have complete data for below.
      FieldMask capture_mask = closing_mask - complete_below;
      if (!!capture_mask)
        node->capture_physical_state(this, state, capture_mask,
                                     closer, dirty_mask, complete_mask);
      dirty_mask |= dirty_below;
      complete_mask |= complete_below;
      // clean up any instances here
      if (!closer.permit_leave_open)
        invalidate_instance_views(state, closing_mask, 
                                  true/*clean*/, false/*force*/);
      else
      {
        // If we had any dirty fields belows that are getting flushed up
        // past this level, then we have to invalidate them here.
        if (!!dirty_below)
          invalidate_instance_views(state, dirty_below,
                                    true/*clean*/, false/*force*/);
        state.dirty_mask -= closing_mask;
      }
      // No matter what invalidate the reduction views
      invalidate_reduction_views(state, closing_mask);
      release_physical_state(state);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_physical_children(CompositeCloser &closer,
                                                  CompositeNode *node,
                                                  PhysicalState &state,
                                                  const FieldMask &closing_mask,
                                                  FieldMask &dirty_mask,
                                                  FieldMask &complete_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, SIPHON_PHYSICAL_CHILDREN_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
      assert(node->logical_node == this);
#endif
      // First check, if all the fields are disjoint, then we're done
      if (state.children.valid_fields * closing_mask)
        return;
      // Make a copy of the open children map since close_physical_child
      // will release our hold on the lock which may lead to someone
      // else invalidating our iterator.
      std::map<Color,FieldMask> open_copy = state.children.open_children;
      // Keep track of two sets of fields
      // 1. The set of fields for which all children are complete
      // 2. The set of fields for which any children are complete
      // Optionally in the future we can make this more precise to 
      // track when we've written to enough children to dominate
      // this node and therefore be complete
      FieldMask all_children = closing_mask;
      FieldMask any_children;
      // Otherwise go through all of the children and 
      // see which ones we need to clean up
      for (std::map<Color,FieldMask>::iterator it = open_copy.begin();
            it != open_copy.end(); it++)
      {
        FieldMask overlap = it->second & closing_mask;
        all_children &= overlap;
        if (!overlap)
          continue;
        FieldMask child_complete;
        close_physical_child(closer, node, state, overlap, 
                             it->first, -1/*next child*/, 
                             dirty_mask, child_complete);
        all_children &= child_complete;
        any_children |= child_complete;
      }
      // If this is a region, if any of our children were complete
      // then we can also be considered complete
      if (is_region())
        complete_mask |= any_children;
      // Otherwise, if this is a partition, we have complete data
      // if all children were complete, we are complete, and 
      // we touched all of the children
      else if (!!all_children && is_complete() &&
               (open_copy.size() == get_num_children()))
        complete_mask |= all_children;
      // Rebuild the valid mask
      FieldMask next_valid;
      for (std::map<Color,FieldMask>::const_iterator it = 
            state.children.open_children.begin(); it !=
            state.children.open_children.end(); it++)
      {
        next_valid |= it->second;
      }
      state.children.valid_fields = next_valid;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_physical_child(CompositeCloser &closer,
                                              CompositeNode *node,
                                              PhysicalState &state,
                                              const FieldMask &closing_mask,
                                              Color target_child,
                                              int next_child,
                                              FieldMask &dirty_mask,
                                              FieldMask &complete_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, CLOSE_PHYSICAL_CHILD_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
      assert(node->logical_node == this);
#endif
      // See if we can find the child
      std::map<Color,FieldMask>::iterator finder = 
        state.children.open_children.find(target_child);
      if (finder == state.children.open_children.end())
        return;
      // Check field disjointness
      if (finder->second * closing_mask)
        return;
      // Check for child disjointness
      if ((next_child >= 0) && 
          are_children_disjoint(finder->first, unsigned(next_child)))
        return;
      FieldMask close_mask = finder->second & closing_mask;
      // Need to get this value before the iterator is invalidated
      RegionTreeNode *child_node = get_tree_child(finder->first);
      if (!closer.permit_leave_open)
      {
        finder->second -= close_mask;
        if (!finder->second)
          state.children.open_children.erase(finder);
      }
      // Release our lock on the current state before going down
      bool was_exclusive = release_physical_state(state);
      CompositeNode *child_composite = 
                                closer.get_composite_node(child_node, node);
      child_node->close_physical_node(closer, child_composite, 
                                      close_mask, dirty_mask, complete_mask);
      // Reacquire our lock on the state upon returning
      acquire_physical_state(state, was_exclusive);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::find_valid_instance_views(PhysicalState &state,
                                                   const FieldMask &valid_mask,
                                                   const FieldMask &space_mask,
                                                   bool needs_space,
                                 std::map<InstanceView*,FieldMask> &valid_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_VALID_INSTANCE_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      FieldMask up_mask = valid_mask - state.dirty_mask;
      RegionTreeNode *parent = get_parent();
      if ((!!up_mask || needs_space) && (parent != NULL))
      {
        // Acquire the parent nodes physical state in read-only mode
        PhysicalState &parent_state = 
          parent->acquire_physical_state(state.ctx, false/*exclusive*/);
        std::map<InstanceView*,FieldMask> local_valid;
        parent->find_valid_instance_views(parent_state, up_mask, space_mask, 
                                          needs_space, local_valid);
        // Release our hold on the parent state
        parent->release_physical_state(parent_state);
        // Get the subview for this level
        for (std::map<InstanceView*,FieldMask>::const_iterator it =
              local_valid.begin(); it != local_valid.end(); it++)
        {
          InstanceView *local_view = it->first->get_subview(get_color());
          valid_views[local_view] = it->second;
          // Add a valid reference
          local_view->add_valid_reference();
          // Then remove the valid reference from the parent view
          if (it->first->remove_valid_reference())
          {
            if (it->first->is_composite_view())
              legion_delete(it->first->as_composite_view());
            else
              legion_delete(it->first->as_materialized_view());
          }
        }
      }
      // Now figure out which of our valid views we can add
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            state.valid_views.begin(); it != state.valid_views.end(); it++)
      {
        // If we need the physical instances to be at least as big as the
        // needed fields, check that first
        if (needs_space)
        {
          if (it->first->is_composite_view())
            continue;
          MaterializedView *current = it->first->as_materialized_view();
          if (!!(space_mask - current->get_physical_mask()))
            continue;
        }
        // If we're looking for instances with space, we want the instances
        // even if they have no valid fields, otherwise if we're not looking
        // for instances with enough space, we can exit out early if they
        // don't have any valid fields
        FieldMask overlap = valid_mask & it->second;
        if (!needs_space && !overlap)
          continue;
        // Check to see if we need to merge the field views.
        std::map<InstanceView*,FieldMask>::iterator finder = 
          valid_views.find(it->first);
        if (finder == valid_views.end())
        {
          valid_views[it->first] = overlap;
          it->first->add_valid_reference();
        }
        else
          finder->second |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::remove_valid_references(
                           const std::map<InstanceView*,FieldMask> &valid_views)
    //--------------------------------------------------------------------------
    {
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_valid_reference())
        {
          if (it->first->is_composite_view())
            legion_delete(it->first->as_composite_view());
          else
            legion_delete(it->first->as_materialized_view());
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::find_valid_reduction_views(PhysicalState &state,
                                                    const FieldMask &valid_mask,
                                          std::set<ReductionView*> &valid_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_VALID_REDUCTION_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      // See if we can continue going up the tree
      if (state.dirty_mask * valid_mask)
      {
        RegionTreeNode *parent = get_parent();
        if (parent != NULL)
        {
          // Acquire the parent state in non-exclusive mode
          PhysicalState &parent_state = 
            parent->acquire_physical_state(state.ctx, false/*exclusive*/);
          parent->find_valid_reduction_views(parent_state, 
                                             valid_mask, valid_views);
          // Release the parent state
          parent->release_physical_state(parent_state);
        }
      }
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            state.reduction_views.begin(); it != 
            state.reduction_views.end(); it++)
      {
        FieldMask uncovered = valid_mask - it->second;
        if (!uncovered && (valid_views.find(it->first) == valid_views.end()))
        {
          valid_views.insert(it->first);
          it->first->add_valid_reference();
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::remove_valid_references(
                                    const std::set<ReductionView*> &valid_views)
    //--------------------------------------------------------------------------
    {
      for (std::set<ReductionView*>::const_iterator it = valid_views.begin();
            it != valid_views.end(); it++)
      {
        if ((*it)->remove_valid_reference())
          legion_delete(*it);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::pull_valid_instance_views(PhysicalState &state,
                                                   const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PULL_VALID_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      std::map<InstanceView*,FieldMask> new_valid_views;
      find_valid_instance_views(state, mask, mask, 
                                false/*needs space*/, new_valid_views);
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            new_valid_views.begin(); it != new_valid_views.end(); it++)
      {
        update_valid_views(state, it->second, false/*dirty*/, it->first);
      }
      // Remove the added valid references
      remove_valid_references(new_valid_views);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::find_pending_updates(PhysicalState &state,
                                              MaterializedView *target,
                                              FieldMask &needed_fields,
                                              std::set<Event> &pending_events) 
    //--------------------------------------------------------------------------
    {
      // Check out our fields first
      {
        std::map<InstanceView*,FieldMask>::const_iterator finder = 
          state.valid_views.find(target);
        if (finder != state.valid_views.end())
          needed_fields -= finder->second;
      }
      if (!!needed_fields)
      {
        // Go through and see if we have any pending updates at this level
        std::map<MaterializedView*,std::map<Event,FieldMask> >::iterator
          finder = state.pending_updates.find(target);
        if (finder != state.pending_updates.end())
        {
          std::map<Event,FieldMask> &pending = finder->second;
          if (!pending.empty())
          {
            for (std::map<Event,FieldMask>::const_iterator it = 
                  pending.begin(); it != pending.end(); it++)
            {
              if (needed_fields * it->second)
                continue;
              needed_fields -= it->second;
              pending_events.insert(it->first);
              if (!needed_fields)
                break;
            }
          }
        }
        // See if we still have any needed fields, if not, check the parent
        // excluding any dirty fields which we can't count going up
        RegionTreeNode *parent = get_parent();
        if (!!needed_fields && (parent != NULL) && target->has_parent_view())
        {
          FieldMask up_mask = needed_fields - state.dirty_mask;
          if (!!up_mask)
          {
            MaterializedView *parent_view = 
                                     target->get_materialized_parent_view();
            FieldMask copy_up = up_mask;
            PhysicalState &parent_state = 
              parent->acquire_physical_state(state.ctx, false/*exclusive*/);
            parent->find_pending_updates(parent_state, parent_view,
                                         copy_up, pending_events);
            parent->release_physical_state(parent_state);
            // Now remove any fields that were removed from the copy_up
            // mask from the set of needed fields
            needed_fields -= (up_mask - copy_up);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::find_copy_across_instances(MappableInfo *info,
                                                    MaterializedView *target,
                           std::map<MaterializedView*,FieldMask> &src_instances,
                        std::map<CompositeView*,FieldMask> &composite_instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_COPY_ACROSS_INSTANCES_CALL);
#endif
      std::map<InstanceView*,FieldMask> valid_views;
      PhysicalState &state = 
        acquire_physical_state(info->ctx, false/*exclusive*/);
      find_valid_instance_views(state, info->traversal_mask,
                                info->traversal_mask, false/*needs space*/,
                                valid_views);
      release_physical_state(state);
      // Now tease them apart into src and composite views and sort
      // them based on the target memory
      FieldMask copy_mask = info->traversal_mask;
      sort_copy_instances(info, target, copy_mask, valid_views,
                          src_instances, composite_instances);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::issue_update_copies(MappableInfo *info,
                                             MaterializedView *dst,
                                             FieldMask copy_mask,
                       const std::map<InstanceView*,FieldMask> &valid_instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ISSUE_UPDATE_COPIES_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(!!copy_mask);
      assert(dst->logical_node == this);
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        assert(it->first->logical_node == this);
      }
#endif
      // Quick check to see if we are done early
      {
        std::map<InstanceView*,FieldMask>::const_iterator finder = 
          valid_instances.find(dst);
        if ((finder != valid_instances.end()) &&
            !(copy_mask - finder->second))
          return;
      }

      // To facilitate optimized copies in the low-level runtime, 
      // we gather all the information needed to issue gather copies 
      // from multiple instances into the data structures below, we then 
      // issue the copy when we're done and update the destination instance.
      std::map<InstanceView*,FieldMask> copy_instances = valid_instances;
      std::map<MaterializedView*,FieldMask> src_instances;
      std::map<CompositeView*,FieldMask> composite_instances;
      // This call destroys copy_instances and also updates copy_mask
      sort_copy_instances(info, dst, copy_mask, copy_instances, 
                          src_instances, composite_instances);

      // Now we can issue the copy operation to the low-level runtime
      if (!src_instances.empty())
      {
        // Get all the preconditions for each of the different instances
        std::map<Event,FieldMask> preconditions;
        FieldMask update_mask; 
        for (std::map<MaterializedView*,FieldMask>::const_iterator it = 
              src_instances.begin(); it != src_instances.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!!it->second);
#endif
          it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                             it->second, preconditions);
          update_mask |= it->second;
        }
        // Now do the destination
        dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                     update_mask, preconditions);

        std::map<Event,FieldMask> postconditions;
        if (!has_component_domains())
        {
          std::set<Domain> copy_domain;
          copy_domain.insert(get_domain());
          issue_grouped_copies(info, dst, preconditions, update_mask,
                               copy_domain, src_instances, postconditions);
        }
        else
          issue_grouped_copies(info, dst, preconditions, update_mask,
                     get_component_domains(), src_instances, postconditions);

        // Tell the destination about all of the copies that were done
        for (std::map<Event,FieldMask>::const_iterator it = 
              postconditions.begin(); it != postconditions.end(); it++)
        {
          dst->add_copy_user(0/*redop*/, it->first, 
                             it->second, false/*reading*/,
                             info->local_proc);
        }
      }
      // If we still have fields that need to be updated and there
      // are composite instances then we need to issue updates copies
      // for those fields from the composite instances
      if (!composite_instances.empty())
      {
        for (std::map<CompositeView*,FieldMask>::const_iterator it = 
              composite_instances.begin(); it !=
              composite_instances.end(); it++)
        {
          it->first->issue_composite_copies(info, dst, it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::sort_copy_instances(MappableInfo *info,
                                             MaterializedView *dst,
                                             FieldMask &copy_mask,
                              std::map<InstanceView*,FieldMask> &copy_instances,
                           std::map<MaterializedView*,FieldMask> &src_instances,
                        std::map<CompositeView*,FieldMask> &composite_instances)
    //--------------------------------------------------------------------------
    {
      // No need to call the mapper if there is only one valid instance
      if (copy_instances.size() == 1)
      {
        const std::pair<InstanceView*,FieldMask> &src_info = 
          *(copy_instances.begin());
        FieldMask op_mask = copy_mask & src_info.second;
        if (!!op_mask)
        {
          InstanceView *src = src_info.first;
          // No need to do anything if src and destination are the same
          // Also check for the same instance which can occur in the case
          // of composite instances
          if (src != dst)
          {
            if (src->is_composite_view())
            {
              CompositeView *current = src->as_composite_view();
              composite_instances[current] = op_mask;
            }
            else
            {
              MaterializedView *current = src->as_materialized_view();
              // If they are the same instance (which can happen with
              // composite views) then we don't need to do it
              if (current->manager->get_instance() != 
                  dst->manager->get_instance())
                src_instances[current] = op_mask;
            }
          }
        }
      }
      else if (!copy_instances.empty())
      {
        bool copy_ready = false;
        // Ask the mapper to put everything in order
        std::set<Memory> available_memories;
        std::map<CompositeView*,FieldMask> available_composite;
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              copy_instances.begin(); it != copy_instances.end(); it++)
        {
          if (it->first->is_composite_view())
          {
            CompositeView *current = it->first->as_composite_view();
            std::map<CompositeView*,FieldMask>::iterator finder = 
              available_composite.find(current);
            if (finder == available_composite.end())
              available_composite[current] = it->second;
            else
              finder->second |= it->second;
          }
          else
          {
            available_memories.insert(
                it->first->as_materialized_view()->get_location());
          }
        }
        std::vector<Memory> chosen_order;
        context->runtime->invoke_mapper_rank_copy_sources(info->local_proc,
                                                          info->mappable,
                                                          available_memories,
                                                          dst->get_location(),
                                                          chosen_order);
        for (std::vector<Memory>::const_iterator mit = chosen_order.begin();
              !copy_ready && (mit != chosen_order.end()); mit++)
        {
          available_memories.erase(*mit);
          std::vector<InstanceView*> to_erase;
          // Go through all the valid instances and issue copies
          // from instances in the given memory
          for (std::map<InstanceView*,FieldMask>::const_iterator it = 
                copy_instances.begin(); it != copy_instances.end(); it++)
          {
            if (it->first->is_composite_view())
              continue;
            MaterializedView *current_view = it->first->as_materialized_view();
            if ((*mit) != current_view->get_location())
              continue;
            // Check to see if there are any valid fields in the copy mask
            FieldMask op_mask = copy_mask & it->second;
            if (!!op_mask)
            {
              // No need to do anything if they are the same instance
              if ((dst != current_view) || (dst->manager->get_instance() != 
                                    current_view->manager->get_instance()))
              {
                std::map<MaterializedView*,FieldMask>::iterator finder = 
                  src_instances.find(current_view);
                if (finder == src_instances.end())
                  src_instances[current_view] = op_mask;
                else
                  finder->second |= op_mask;
              }
              // Update the copy mask
              copy_mask -= op_mask;
              if (!copy_mask)
              {
                copy_ready = true;
                break;
              }
            }
            to_erase.push_back(it->first);
          }
          // Erase any instances we considered and used
          for (unsigned idx = 0; idx < to_erase.size(); idx++)
            copy_instances.erase(to_erase[idx]);
        }
        // Now do any remaining memories not put in order by the mapper
        for (std::set<Memory>::const_iterator mit = 
              available_memories.begin(); !copy_ready && (mit != 
              available_memories.end()); mit++)
        {
          std::vector<InstanceView*> to_erase;
          for (std::map<InstanceView*,FieldMask>::const_iterator it =
                copy_instances.begin(); it != copy_instances.end(); it++)
          {
            if (it->first->is_composite_view())
              continue;
            MaterializedView *current_view = it->first->as_materialized_view();
            if ((*mit) != current_view->get_location())
              continue;
            // Check to see if there are any valid fields in the copy mask
            FieldMask op_mask = copy_mask & it->second;
            if (!!op_mask)
            {
              // No need to do anything if they are the same instance
              if ((dst != current_view) || (dst->manager->get_instance() != 
                                     current_view->manager->get_instance()))
              {
                std::map<MaterializedView*,FieldMask>::iterator finder = 
                  src_instances.find(current_view);
                if (finder == src_instances.end())
                  src_instances[current_view] = op_mask;
                else
                  finder->second |= op_mask;
              }
              // Update the copy mask
              copy_mask -= op_mask;
              if (!copy_mask)
              {
                copy_ready = true;
                break;
              }
            }
            to_erase.push_back(it->first);
          }
          // Erase any instances we've checked
          for (unsigned idx = 0; idx < to_erase.size(); idx++)
            copy_instances.erase(to_erase[idx]);
        }
        // Lastly, if we are still not done, see if we have
        // any composite instances to issue copies from
        for (std::map<CompositeView*,FieldMask>::const_iterator cit = 
              available_composite.begin(); !copy_ready && (cit !=
              available_composite.end()); cit++)
        {
          FieldMask op_mask = copy_mask & cit->second;
          if (!!op_mask)
          {
            // No need to look for duplicates, we know this is
            // the first time this data structure can be touched
            composite_instances[cit->first] = op_mask;
            copy_mask -= op_mask;
            if (!copy_mask)
            {
              copy_ready = true;
              break;
            }
          }
        }
      }
      // Otherwise all the fields have no current data so they are
      // by defintiion up to date
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::issue_grouped_copies(MappableInfo *info,
                                                         MaterializedView *dst,
                                       std::map<Event,FieldMask> &preconditions, 
                                       const FieldMask &update_mask,
                                       const std::set<Domain> &copy_domains,
                     const std::map<MaterializedView*,FieldMask> &src_instances,
                                      std::map<Event,FieldMask> &postconditions)
    //--------------------------------------------------------------------------
    {
      // Now let's build maximal sets of fields which have
      // identical event preconditions. Use a list so our
      // iterators remain valid under insertion and push back
      std::list<PreconditionSet> precondition_sets;
      compute_precondition_sets(update_mask, preconditions, 
                                precondition_sets);
      // Now that we have our precondition sets, it's time
      // to issue the distinct copies to the low-level runtime
      // Issue a copy for each of the different precondition sets
      for (std::list<PreconditionSet>::iterator pit = 
            precondition_sets.begin(); pit != 
            precondition_sets.end(); pit++)
      {
        PreconditionSet &pre_set = *pit;
        // Build the src and dst fields vectors
        std::vector<Domain::CopySrcDstField> src_fields;
        std::vector<Domain::CopySrcDstField> dst_fields;
        std::map<MaterializedView*,FieldMask> update_views;
        for (std::map<MaterializedView*,FieldMask>::const_iterator it =
              src_instances.begin(); it != src_instances.end(); it++)
        {
          FieldMask op_mask = pre_set.pre_mask & it->second;
          if (!!op_mask)
          {
            it->first->copy_from(op_mask, src_fields);
            dst->copy_to(op_mask, dst_fields);
            update_views[it->first] = op_mask;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(!src_fields.empty());
        assert(!dst_fields.empty());
        assert(src_fields.size() == dst_fields.size());
#endif
        // Now that we've got our offsets ready, we
        // can now issue the copy to the low-level runtime
        Event copy_pre = Event::merge_events(pre_set.preconditions);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
        if (!copy_pre.exists())
        {
          UserEvent new_copy_pre = UserEvent::create_user_event();
          new_copy_pre.trigger();
          copy_pre = new_copy_pre;
        }
#endif
#ifdef LEGION_LOGGING
        LegionLogging::log_event_dependences(
            Machine::get_executing_processor(), 
            pre_set.preconditions, copy_pre);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_event_dependences(pre_set.preconditions, copy_pre);
#endif
        std::set<Event> post_events;
        for (std::set<Domain>::const_iterator it = copy_domains.begin();
              it != copy_domains.end(); it++)
        {
          post_events.insert(it->copy(src_fields, dst_fields, copy_pre));
        }
        Event copy_post = Event::merge_events(post_events);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
        if (!copy_post.exists())
        {
          UserEvent new_copy_post = UserEvent::create_user_event();
          new_copy_post.trigger();
          copy_post = new_copy_post;
        }
#endif
        // Save the copy post in the post conditions
        if (copy_post.exists())
        {
          // Register copy post with the source views
          // Note it is up to the caller to make sure the event
          // gets registered with the destination
          for (std::map<MaterializedView*,FieldMask>::const_iterator it = 
                update_views.begin(); it != update_views.end(); it++)
          {
            it->first->add_copy_user(0/*redop*/, copy_post,
                                     it->second, true/*reading*/,
                                     info->local_proc);
          }
          postconditions[copy_post] = pre_set.pre_mask;
        }
#if defined(LEGION_SPY) || defined(LEGION_LOGGING)
        IndexSpace copy_index_space = 
                        dst->logical_node->get_domain().get_index_space();
        for (std::map<MaterializedView*,FieldMask>::const_iterator it =
              update_views.begin(); it != update_views.end(); it++)
        {
#ifdef LEGION_LOGGING
          {
            std::set<FieldID> copy_fields;
            RegionNode *manager_node = dst->manager->region_node;
            manager_node->column_source->to_field_set(it->second,
                                                      copy_fields);
            LegionLogging::log_lowlevel_copy(
                Machine::get_executing_processor(),
                it->first->manager->get_instance(),
                dst->manager->get_instance(),
                copy_index_space,
                manager_node->column_source->handle,
                manager_node->handle.tree_id,
                copy_pre, copy_post, copy_fields, 0/*redop*/);
          }
#endif
#ifdef LEGION_SPY
          RegionNode *manager_node = dst->manager->region_node;
          char *string_mask = 
            manager_node->column_source->to_string(it->second);
          LegionSpy::log_copy_operation(
              it->first->manager->get_instance().id,
              dst->manager->get_instance().id,
              copy_index_space.id,
              manager_node->column_source->handle.id,
              manager_node->handle.tree_id, copy_pre, copy_post,
              0/*redop*/, string_mask);
          free(string_mask);
#endif
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::compute_precondition_sets(
        FieldMask update_mask, std::map<Event,FieldMask> &preconditions,
        std::list<PreconditionSet> &precondition_sets)
    //--------------------------------------------------------------------------
    {
      for (std::map<Event,FieldMask>::iterator pit = 
            preconditions.begin(); pit != preconditions.end(); pit++)
      {
        bool inserted = false;
        // Also keep track of which fields have updates
        // but don't have any preconditions
        update_mask -= pit->second;
        // Insert this event into the precondition sets 
        for (std::list<PreconditionSet>::iterator it = 
              precondition_sets.begin(); it != precondition_sets.end(); it++)
        {
          // Easy case, check for equality
          if (pit->second == it->pre_mask)
          {
            it->preconditions.insert(pit->first);
            inserted = true;
            break;
          }
          FieldMask overlap = pit->second & it->pre_mask;
          // Easy case, they are disjoint so keep going
          if (!overlap)
            continue;
          // Moderate case, we are dominated, split into two sets
          // reusing existing set and making a new set
          if (overlap == pit->second)
          {
            // Leave the existing set and make it the difference 
            it->pre_mask -= overlap;
            precondition_sets.push_back(PreconditionSet(overlap));
            PreconditionSet &last = precondition_sets.back();
            last.preconditions = it->preconditions;
            last.preconditions.insert(pit->first);
            inserted = true;
            break;
          }
          // Moderate case, we dominate the existing set
          if (overlap == it->pre_mask)
          {
            // Add ourselves to the existing set and then
            // keep going for the remaining fields
            it->preconditions.insert(pit->first);
            pit->second -= overlap;
            // Can't consider ourselves added yet
            continue;
          }
          // Hard case, neither dominates, compute three
          // distinct sets of fields, keep left one in
          // place and reduce scope, add new one at the
          // end for overlap, continue iterating for right one
          it->pre_mask -= overlap;
          const std::set<Event> &temp_preconditions = it->preconditions;
          it = precondition_sets.insert(it, PreconditionSet(overlap));
          it->preconditions = temp_preconditions;
          it->preconditions.insert(pit->first);
          pit->second -= overlap;
          continue;
        }
        if (!inserted)
        {
          precondition_sets.push_back(PreconditionSet(pit->second));
          PreconditionSet &last = precondition_sets.back();
          last.preconditions.insert(pit->first);
        }
      }
      // For any fields which need copies but don't have
      // any preconditions, but them in their own set.
      // Put it on the front because it is the copy with
      // no preconditions so it can start right away!
      if (!!update_mask)
        precondition_sets.push_front(PreconditionSet(update_mask));
    }

    //--------------------------------------------------------------------------
    Event RegionTreeNode::perform_copy_operation(Event precondition,
                         const std::vector<Domain::CopySrcDstField> &src_fields,
                         const std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
      if (has_component_domains())
      {
        const std::set<Domain> &component_domains = get_component_domains();
        std::set<Event> result_events;
        for (std::set<Domain>::const_iterator it = component_domains.begin();
              it != component_domains.end(); it++)
          result_events.insert(it->copy(src_fields, dst_fields, precondition));
        return Event::merge_events(result_events);
      }
      Domain copy_domain = get_domain();
      return copy_domain.copy(src_fields, dst_fields, precondition);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::issue_update_reductions(LogicalView *target,
                                                 const FieldMask &mask,
                                                 Processor local_proc,
                     const std::map<ReductionView*,FieldMask> &valid_reductions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ISSUE_UPDATE_REDUCTIONS_CALL);
#endif
      // Handle the special case where the target is a composite view
      if (!target->is_reduction_view())
      {
        InstanceView *inst_target = target->as_instance_view();
        if (inst_target->is_composite_view())
        {
          CompositeView *comp_target = inst_target->as_composite_view();
          // Save all the reductions to the composite target
          for (std::map<ReductionView*,FieldMask>::const_iterator it = 
                valid_reductions.begin(); it != valid_reductions.end(); it++)
          {
            FieldMask copy_mask = mask & it->second;
            if (!copy_mask)
              continue;
            comp_target->update_reduction_views(it->first, copy_mask);
          }
          // Once we're done saving the reductions we are finished
          return;
        }
      }
      // Go through all of our reduction instances and issue reductions
      // to the target instances
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        // Doesn't need to reduce to itself
        if (target == (it->first))
          continue;
        FieldMask copy_mask = mask & it->second;
        if (!!copy_mask)
        {
#ifdef DEBUG_HIGH_LEVEL
          // all fields in the reduction instances should be used
          assert(!(it->second - copy_mask));
#endif
          // Then we have a reduction to perform
          it->first->perform_reduction(target, copy_mask, local_proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_instance_views(PhysicalState &state,
                                                 const FieldMask &invalid_mask,
                                                 bool clean, bool force)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, INVALIDATE_INSTANCE_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      std::vector<InstanceView*> to_delete;
      for (std::map<InstanceView*,FieldMask>::iterator it = 
            state.valid_views.begin(); it != state.valid_views.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second && (force || !it->first->is_persistent()))
          to_delete.push_back(it->first);
      }
      for (std::vector<InstanceView*>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        if ((*it)->remove_valid_reference())
        {
          if ((*it)->is_composite_view())
            legion_delete((*it)->as_composite_view());
          else
            legion_delete((*it)->as_materialized_view());
        }
        state.valid_views.erase(*it);
      }
      if (clean)
        state.dirty_mask -= invalid_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_reduction_views(PhysicalState &state,
                                                  const FieldMask &invalid_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, INVALIDATE_REDUCTION_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      std::vector<ReductionView*> to_delete;
      for (std::map<ReductionView*,FieldMask>::iterator it = 
            state.reduction_views.begin(); it != 
            state.reduction_views.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<ReductionView*>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        if ((*it)->remove_valid_reference())
          legion_delete(*it);
        state.reduction_views.erase(*it);
      }
      state.reduction_mask -= invalid_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_valid_views(PhysicalState &state, 
                                            const FieldMask &valid_mask,
                                            bool dirty, InstanceView *new_view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, UPDATE_VALID_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
      if (!new_view->is_composite_view())
        assert(!(valid_mask - 
          new_view->as_materialized_view()->manager->layout->allocated_fields));
      assert(new_view->logical_node == this);
#endif
      // Add our reference first in case the new view is also currently in
      // the list of valid views.  We don't want it to be prematurely deleted
      new_view->add_valid_reference();
      if (dirty)
      {
        invalidate_instance_views(state, valid_mask, 
                                  false/*clean*/, false/*force*/);
        state.dirty_mask |= valid_mask;
      }
      std::map<InstanceView*,FieldMask>::iterator finder = 
        state.valid_views.find(new_view);
      if (finder == state.valid_views.end())
      {
        // New valid view, update everything accordingly
        state.valid_views[new_view] = valid_mask;
      }
      else
      {
        // It already existed update the valid mask
        finder->second |= valid_mask;
        // Remove the reference that we added since it already was referenced
        // Since we know it already had a reference no need to
        // check for the deletion condition
        new_view->remove_valid_reference();
      }
    } 

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_valid_views(PhysicalState &state,
                                            const FieldMask &valid_mask,
                                            const FieldMask &dirty_mask,
                                    const std::vector<InstanceView*> &new_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, UPDATE_VALID_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      // Add our references first to avoid any premature free operations
      for (std::vector<InstanceView*>::const_iterator it = new_views.begin();
            it != new_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->logical_node == this);
#endif
        (*it)->add_valid_reference();
      }
      if (!!dirty_mask)
      {
        invalidate_instance_views(state, dirty_mask, 
                                  false/*clean*/, false/*force*/);
        state.dirty_mask |= dirty_mask;
      }
      for (std::vector<InstanceView*>::const_iterator it = new_views.begin();
            it != new_views.end(); it++)
      {
        std::map<InstanceView*,FieldMask>::iterator finder = 
          state.valid_views.find(*it);
        if (finder == state.valid_views.end())
        {
          // New valid view, update everything accordingly
          state.valid_views[*it] = valid_mask;
        }
        else
        {
          // It already existed update the valid mask
          finder->second |= valid_mask;
          // Remove the reference that we added since it already was referenced
          // Since we know it already had a reference there is no
          // need to check for the deletion condition
          (*it)->remove_valid_reference();
        }
#ifdef DEBUG_HIGH_LEVEL
        finder = state.valid_views.find(*it);
        if (!(*it)->is_composite_view())
        assert(!(finder->second - 
            (*it)->as_materialized_view()->manager->layout->allocated_fields));
#endif
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_valid_views(PhysicalState &state,
                                            const FieldMask &valid_mask,
                                            const FieldMask &dirty_mask,
                                const std::vector<MaterializedView*> &new_views)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, UPDATE_VALID_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      // Add our references first to avoid any premature free operations
      for (std::vector<MaterializedView*>::const_iterator it = 
            new_views.begin(); it != new_views.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->logical_node == this);
#endif
        (*it)->add_valid_reference();
      }
      if (!!dirty_mask)
      {
        invalidate_instance_views(state, dirty_mask, 
                                  false/*clean*/, false/*force*/);
        state.dirty_mask |= dirty_mask;
      }
      for (std::vector<MaterializedView*>::const_iterator it = 
            new_views.begin(); it != new_views.end(); it++)
      {
        std::map<InstanceView*,FieldMask>::iterator finder = 
          state.valid_views.find(*it);
        if (finder == state.valid_views.end())
        {
          // New valid view, update everything accordingly
          state.valid_views[*it] = valid_mask;
        }
        else
        {
          // It already existed update the valid mask
          finder->second |= valid_mask;
          // Remove the reference that we added since it already was referenced
          // Since we know it already had a reference there is no
          // need to check for the deletion condition
          (*it)->remove_valid_reference();
        }
#ifdef DEBUG_HIGH_LEVEL
        finder = state.valid_views.find(*it);
        assert(!(finder->second - (*it)->manager->layout->allocated_fields));
#endif
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::update_reduction_views(PhysicalState &state,
                                                const FieldMask &valid_mask,
                                                ReductionView *new_view) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, UPDATE_REDUCTION_VIEWS_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(state.node == this);
#endif
      std::map<ReductionView*,FieldMask>::iterator finder = 
        state.reduction_views.find(new_view);
      if (finder == state.reduction_views.end())
      {
        new_view->add_valid_reference();
        state.reduction_views[new_view] = valid_mask;
      }
      else
      {
        finder->second |= valid_mask;
      }
      state.reduction_mask |= valid_mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::flush_reductions(const FieldMask &valid_mask,
                                          ReductionOpID redop,
                                          MappableInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FLUSH_REDUCTIONS_CALL);
#endif
      // Go through the list of reduction views and see if there are
      // any that don't mesh with the current user and therefore need
      // to be flushed.
      FieldMask flush_mask;
      std::map<InstanceView*,FieldMask> valid_views;
      std::map<ReductionView*,FieldMask> reduction_views;
      {
        PhysicalState &state = 
          acquire_physical_state(info->ctx, false/*exclusive*/);
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              state.reduction_views.begin(); it != 
              state.reduction_views.end(); it++)
        {
          // Skip reductions that have the same reduction op ID
          if (it->first->get_redop() == redop)
            continue;
          FieldMask overlap = valid_mask & it->second;
          if (!overlap)
            continue;
          flush_mask |= overlap; 
          reduction_views.insert(*it);
        }
        // Now get any physical instances to flush to
        if (!!flush_mask)
        {
          find_valid_instance_views(state, flush_mask, flush_mask, 
                                    false/*needs space*/, valid_views);
        }
        release_physical_state(state);
      }
      if (!!flush_mask)
      {
#ifdef DEBUG_HIGH_LEVEL
        FieldMask update_mask;
#endif
        // Iterate over all the valid instances and issue any reductions
        // to the target that need to be done
        for (std::map<InstanceView*,FieldMask>::iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = flush_mask & it->second; 
          issue_update_reductions(it->first, overlap, info->local_proc,
                                  reduction_views);
          // Save the overlap fields
          it->second = overlap;
#ifdef DEBUG_HIGH_LEVEL
          update_mask |= overlap;
#endif
        }
#ifdef DEBUG_HIGH_LEVEL
        // We should have issued reduction operations to at least
        // one place for every single reduction field.
        assert(update_mask == flush_mask);
#endif
        // Now update our physical state
        PhysicalState &state = 
          acquire_physical_state(info->ctx, true/*exclusive*/);
        // Update the valid views.  Don't mark them dirty since we
        // don't want to accidentally invalidate some of our other
        // instances which get updated later in the loop.  Note this
        // is safe since we're updating all the instances for each
        // reduction field.
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          update_valid_views(state, it->second, false/*dirty*/, it->first);
        }
        // Update the dirty mask since we didn't do it when updating
        // the valid instance views do it now
        state.dirty_mask |= flush_mask;
        // Then invalidate all the reduction views that we flushed
        invalidate_reduction_views(state, flush_mask);
        release_physical_state(state);
        // Release any valid view references we are holding
        remove_valid_references(valid_views);
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::initialize_physical_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, INITIALIZE_PHYSICAL_STATE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < physical_state_size);
#endif
      PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(!state.dirty_mask);
      assert(!state.reduction_mask);
      assert(!state.children.valid_fields);
      assert(state.children.open_children.empty());
      assert(state.valid_views.empty());
      assert(state.reduction_views.empty());
      // Should be one since we're using it
      assert(state.acquired_count == 1);
#endif
      state.dirty_mask = FieldMask();
      state.reduction_mask = FieldMask();
      state.children = ChildState();
      state.valid_views.clear();
      state.reduction_views.clear();
      release_physical_state(state);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_physical_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, INVALIDATE_PHYSICAL_STATE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < physical_state_size);
#endif
      PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);

      state.dirty_mask = FieldMask();
      state.reduction_mask = FieldMask();
      state.children = ChildState();
      state.pending_updates.clear();

      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            state.valid_views.begin(); it != state.valid_views.end(); it++)
      {
        if (it->first->remove_valid_reference())
        {
          if (it->first->is_composite_view())
            legion_delete(it->first->as_composite_view());
          else
            legion_delete(it->first->as_materialized_view());
        }
      }
      state.valid_views.clear();
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            state.reduction_views.begin(); it != 
            state.reduction_views.end(); it++)
      {
        if (it->first->remove_valid_reference())
          legion_delete(it->first);
      }
      state.reduction_views.clear();
      release_physical_state(state);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_physical_state(ContextID ctx,
                                                  const FieldMask &invalid_mask,
                                                  bool force)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, INVALIDATE_PHYSICAL_STATE_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx < physical_state_size);
#endif
      PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);

      invalidate_instance_views(state, invalid_mask, true/*clean*/, force);
      invalidate_reduction_views(state, invalid_mask);
      state.children.valid_fields -= invalid_mask;
      std::vector<Color> to_delete;
      for (std::map<Color,FieldMask>::iterator it = 
            state.children.open_children.begin(); it !=
            state.children.open_children.end(); it++)
      {
        it->second -= invalid_mask;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (unsigned idx = 0; idx < to_delete.size(); idx++)
        state.children.open_children.erase(to_delete[idx]);
      release_physical_state(state);
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::pack_send_state(ContextID ctx, Serializer &rez,
                             AddressSpaceID target, const FieldMask &send_mask,
                             std::map<LogicalView*,FieldMask> &needed_views,
                             std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      // Acquire our state in read only mode
      PhysicalState &state = acquire_physical_state(ctx, false/*exclusive*/);
      RezCheck z(rez);
      rez.serialize(state.dirty_mask & send_mask);
      rez.serialize(state.reduction_mask & send_mask);
      rez.serialize(state.children.open_children.size());
      for (std::map<Color,FieldMask>::const_iterator it = 
            state.children.open_children.begin(); it !=
            state.children.open_children.end(); it++)
      {
        FieldMask overlap = it->second & send_mask;
        rez.serialize(it->first);
        rez.serialize(overlap);
      }
      rez.serialize(state.valid_views.size());
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            state.valid_views.begin(); it != state.valid_views.end(); it++)
      {
        FieldMask overlap = it->second & send_mask;
        rez.serialize(overlap);
        if (!!overlap)
        {
          DistributedID did = it->first->send_state(target, overlap, 
                                                needed_views, needed_managers);
          rez.serialize(did);
        }
        else
        {
          DistributedID did = 0;
          rez.serialize(did); // empty did
        }
      }
      rez.serialize(state.reduction_views.size());
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            state.reduction_views.begin(); it != 
            state.reduction_views.end(); it++)
      {
        FieldMask overlap = it->second & send_mask;
        rez.serialize(overlap);
        if (!!overlap)
        {
          DistributedID did = it->first->send_state(target, overlap,
                                                needed_views, needed_managers);
          rez.serialize(did);
        }
        else
        {
          DistributedID did = 0;
          rez.serialize(did); // empty did
        }
      }
      bool result = !(send_mask * state.children.valid_fields);
      // Release our hold on this node
      release_physical_state(state);
      return result;
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::pack_send_back_state(ContextID ctx, Serializer &rez,
                              AddressSpaceID target, const FieldMask &send_mask,
                              std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      // This method is the same as the one above, but instead calls
      // send_back_state on instance and reduction views.  Since what is
      // sent is the same, we can use the same unpack_send_state method
      // when unpacking on the destination node.
      PhysicalState &state = acquire_physical_state(ctx, false/*exclusive*/); 
      RezCheck z(rez);
      rez.serialize(state.dirty_mask);
      rez.serialize(state.reduction_mask);
      rez.serialize(state.children.open_children.size());
      for (std::map<Color,FieldMask>::const_iterator it = 
            state.children.open_children.begin(); it !=
            state.children.open_children.end(); it++)
      {
        FieldMask overlap = it->second & send_mask;
        rez.serialize(it->first);
        rez.serialize(overlap);
      }
      rez.serialize(state.valid_views.size());
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            state.valid_views.begin(); it != state.valid_views.end(); it++)
      {
        FieldMask overlap = it->second & send_mask;
        rez.serialize(overlap);
        if (!!overlap)
        {
          DistributedID did = 
            it->first->send_back_state(target, overlap, needed_managers);
          rez.serialize(did);
        }
        else
        {
          DistributedID did = 0;
          rez.serialize(did); // empty did
        }
      }
      rez.serialize(state.reduction_views.size());
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            state.reduction_views.begin(); it != 
            state.reduction_views.end(); it++)
      {
        FieldMask overlap = it->second & send_mask;
        rez.serialize(overlap);
        if (!!overlap)
        {
          DistributedID did = 
            it->first->send_back_state(target, overlap, needed_managers);
          rez.serialize(did);
        }
        else
        {
          DistributedID did = 0;
          rez.serialize(did); // empty did
        }
      }
      bool result = !(send_mask * state.children.valid_fields);
      // Release our hold on the state
      release_physical_state(state);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::unpack_send_state(ContextID ctx, Deserializer &derez,
                                  FieldSpaceNode *column, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);
      DerezCheck z(derez);
      // Dirty mask
      FieldMask dirty_mask;
      derez.deserialize(dirty_mask);
      column->transform_field_mask(dirty_mask, source);
      state.dirty_mask |= dirty_mask;
      // Reduction mask
      FieldMask reduction_mask;
      derez.deserialize(reduction_mask);
      column->transform_field_mask(reduction_mask, source);
      state.reduction_mask |= reduction_mask;
      size_t num_open_children;
      derez.deserialize(num_open_children);
      for (unsigned idx = 0; idx < num_open_children; idx++)
      {
        Color child_color;
        derez.deserialize(child_color);
        FieldMask child_mask;
        derez.deserialize(child_mask);
        if (!!child_mask)
        {
          column->transform_field_mask(child_mask, source);
          std::map<Color,FieldMask>::iterator finder = 
            state.children.open_children.find(child_color);
          if (finder == state.children.open_children.end())
            state.children.open_children[child_color] = child_mask;
          else
            finder->second |= child_mask;
          state.children.valid_fields |= child_mask;
        }
      }
      size_t num_valid_views;
      derez.deserialize(num_valid_views);
      for (unsigned idx = 0; idx < num_valid_views; idx++)
      {
        FieldMask inst_mask;
        derez.deserialize(inst_mask);
        DistributedID did;
        derez.deserialize(did);
        if (!!inst_mask)
        {
          column->transform_field_mask(inst_mask, source);
          InstanceView *view = context->find_view(did)->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
          assert(view != NULL);
#endif
          // Dirty mask updates itself
          update_valid_views(state, inst_mask, false/*dirty*/, view);
        }
      }
      size_t num_reduc_views;
      derez.deserialize(num_reduc_views);
      for (unsigned idx = 0; idx < num_reduc_views; idx++)
      {
        FieldMask reduc_mask;
        derez.deserialize(reduc_mask);
        DistributedID did;
        derez.deserialize(did);
        if (!!reduc_mask)
        {
          column->transform_field_mask(reduc_mask, source);
          ReductionView *view = context->find_view(did)->as_reduction_view();
#ifdef DEBUG_HIGH_LEVEL
          assert(view != NULL);
#endif
          update_reduction_views(state, reduc_mask, view);
        }
      }
      // Release our hold on the physical state
      release_physical_state(state);
    }

#ifndef LOGICAL_FIELD_TREE
    //--------------------------------------------------------------------------
    template<typename ALLOC>
    FieldMask RegionTreeNode::perform_dependence_checks(
        const LogicalUser &user, std::list<LogicalUser, ALLOC> &prev_users,
        const FieldMask &check_mask, bool validates_regions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_DEPENDENCE_CHECKS_CALL);
#endif
      FieldMask dominator_mask = check_mask;
      // It's not actually sound to assume we dominate something
      // if we don't observe any users of those fields.  Therefore
      // also keep track of the fields that we observe.  We'll use this
      // at the end when computing the final dominator mask.
      FieldMask observed_mask;
      FieldMask user_check_mask = user.field_mask & check_mask;
      const bool tracing = user.op->is_tracing();
      for (std::list<LogicalUser>::iterator it = prev_users.begin();
            it != prev_users.end(); /*nothing*/)
      {
        if (!(user_check_mask * (it->field_mask & check_mask)))
        {
          observed_mask |= it->field_mask;
          DependenceType dtype = check_dependence_type(it->usage, user.usage);
          bool validate = validates_regions;
          switch (dtype)
          {
            case NO_DEPENDENCE:
              {
                // No dependence so remove bits from the dominator mask
                dominator_mask -= it->field_mask;
                break;
              }
            case ANTI_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
            case PROMOTED_DEPENDENCE:
              {
                // Mark that these kinds of dependences are not allowed
                // to validate region inputs
                validate = false;
                // No break so we register dependences just like
                // a true dependence
              }
            case TRUE_DEPENDENCE:
              {
#ifdef LEGION_LOGGING
                if (dtype != PROMOTED_DEPENDENCE)
                  LegionLogging::log_mapping_dependence(
                      Machine::get_executing_processor(),
                      user.op->get_parent()->get_unique_task_id(),
                      it->uid, it->idx, user.uid, user.idx, dtype);
#endif
#ifdef LEGION_SPY
                if (dtype != PROMOTED_DEPENDENCE)
                  LegionSpy::log_mapping_dependence(
                      user.op->get_parent()->get_unique_task_id(),
                      it->uid, it->idx, user.uid, user.idx, dtype);
#endif
                // Do this after the logging since we might 
                // update the iterator.
                // If we can validate a region record which of our
                // predecessors regions we are validating, otherwise
                // just register a normal dependence
                if (validate)
                {
                  if (user.op->register_region_dependence(user.idx, it->op, 
                                                          it->gen, it->idx,
                                                          dtype))
                  {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
                    // Now we can prune it from the list and continue
                    it = prev_users.erase(it);
#else
                    it++;
#endif
                    continue;
                  }
                  else
                  {
                    // hasn't commited, reset timeout and continue
                    it->timeout = LogicalUser::TIMEOUT;
                    it++;
                    continue;
                  }
                }
                else
                {
                  if (user.op->register_dependence(user.idx, it->op, it->gen,
                                                   it->idx, dtype))
                  {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
                    // Now we can prune it from the list and continue
                    it = prev_users.erase(it);
#else
                    it++;
#endif
                    continue;
                  }
                  else
                  {
                    // hasn't committed, reset timeout and continue
                    it->timeout = LogicalUser::TIMEOUT;
                    it++;
                    continue;
                  }
                }
                break;
              }
            default:
              assert(false); // should never get here
          }
        }
        // If we didn't register any kind of dependence, check
        // to see if the timeout has expired.  Note that it is
        // unsound to do this if we are tracing so don't perform
        // the check in that case.
        if (!tracing)
        {
          if (it->timeout <= 0)
          {
            // Timeout has expired.  Check whether the operation
            // has committed. If it has prune it from the list.
            // Otherwise reset its timeout and continue.
            if (it->op->is_operation_committed(it->gen))
            {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
              it = prev_users.erase(it);
#else
              // Can't prune things early for these cases
              it->timeout = LogicalUser::TIMEOUT;
              it++;
#endif
            }
            else
            {
              // Operation hasn't committed, reset timeout
              it->timeout = LogicalUser::TIMEOUT;
              it++;
            }
          }
          else
          {
            // Timeout hasn't expired, decrement it and continue
            it->timeout--;
            it++;
          }
        }
        else
          it++; // Tracing so no timeouts
      }
      // The result of this computation is the dominator mask.
      // It's only sound to say that we dominate fields that
      // we actually observed users for so intersect the dominator 
      // mask with the observed mask
      return (dominator_mask & observed_mask);
    }

    //--------------------------------------------------------------------------
    template<typename ALLOC>
    void RegionTreeNode::perform_closing_checks(
        LogicalCloser &closer, std::list<LogicalUser, ALLOC> &users, 
        const FieldMask &check_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_CLOSING_CHECKS_CALL);
#endif
      // Since we are performing a close operation on the region
      // tree data structure, we know that we need to register
      // mapping dependences on all of the operations in the 
      // current epoch since close operations must be serialized
      // with respect to mappings.  Finally we have to upgrade the
      // privilege to read-write to ensure that anyone that comes
      // later also records mapping dependences on the users.
      const FieldMask user_check_mask = closer.user.field_mask & check_mask; 
      for (std::list<LogicalUser>::iterator it = users.begin();
            it != users.end(); /*nothing*/)
      {
        FieldMask overlap = user_check_mask & it->field_mask;
        if (!overlap)
        {
          it++;
          continue;
        }
        DependenceType dtype = check_dependence_type(it->usage, 
                                                     closer.user.usage);
#ifdef LEGION_LOGGING
        if ((dtype != NO_DEPENDENCE) && (dtype != PROMOTED_DEPENDENCE))
          LegionLogging::log_mapping_dependence(
              Machine::get_executing_processor(),
              closer.user.op->get_parent()->get_unique_task_id(),
              it->uid, it->idx, closer.user.uid, closer.user.idx, dtype);
#endif
#ifdef LEGION_SPY
        if ((dtype != NO_DEPENDENCE) && (dtype != PROMOTED_DEPENDENCE))
          LegionSpy::log_mapping_dependence(
              closer.user.op->get_parent()->get_unique_task_id(),
              it->uid, it->idx, closer.user.uid, closer.user.idx, dtype);
#endif
        // Register the dependence 
        if (closer.validates)
        {
          if (closer.user.op->register_region_dependence(closer.user.idx, 
                                                         it->op, it->gen, 
                                                         it->idx, dtype))
          {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
            it = users.erase(it);
            continue;
#endif
          }
          else
          {
            // it hasn't committed, reset timeout
            it->timeout = LogicalUser::TIMEOUT;
          }
        }
        else
        {
          if (closer.user.op->register_dependence(closer.user.idx, it->op, 
                                                  it->gen, it->idx, dtype))
          {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
            it = users.erase(it);
            continue;
#endif
          }
          else
          {
            // it hasn't committed so reset the timeout
            it->timeout = LogicalUser::TIMEOUT;
          }
        }
        // Record that we closed this user
        // Update the field mask and the privilege
        closer.closed_users.push_back(*it);
        LogicalUser &closed_user = closer.closed_users.back();
        closed_user.field_mask = overlap;
        closed_user.usage.privilege = 
          (PrivilegeMode)((int)closed_user.usage.privilege | PROMOTED);
        // Remove the closed set of fields from this user
        it->field_mask -= overlap;
        // If it's empty, remove it from the list and let
        // the mapping reference go up the tree with it
        // Otherwise add a new mapping reference
        if (!it->field_mask)
          it = users.erase(it);
        else
        {
          it->op->add_mapping_reference(it->gen);
          it++;
        }
      }
    }

#else
    //--------------------------------------------------------------------------
    FieldMask RegionTreeNode::perform_dependence_checks(
        const LogicalUser &user, FieldTree<LogicalUser> *users,
        const FieldMask &check_mask, bool validates_regions)
    //--------------------------------------------------------------------------
    {
      FieldMask user_check_mask = user.field_mask & check_mask;
      if (!user_check_mask)
        return user_check_mask;
      LogicalDepAnalyzer analyzer(user, check_mask, 
                                  validates_regions, user.op->is_tracing());
      users->analyze<LogicalDepAnalyzer>(user_check_mask, analyzer);
      return analyzer.get_dominator_mask();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_closing_checks(
        LogicalCloser &closer, FieldTree<LogicalUser> *users,
        const FieldMask &check_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask user_check_mask = closer.user.field_mask & check_mask;
      if (!user_check_mask)
        return;
      closer.local_closing_mask = user_check_mask;
      closer.reinsert_count = 0;
      users->analyze<LogicalCloser>(user_check_mask, closer);
    }
#endif

    /////////////////////////////////////////////////////////////
    // Region Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(LogicalRegion r, PartitionNode *par,
                           IndexSpaceNode *row_src, FieldSpaceNode *col_src,
                           RegionTreeForest *ctx)
      : RegionTreeNode(ctx, col_src), handle(r), 
        parent(par), row_source(row_src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(const RegionNode &rhs)
      : RegionTreeNode(NULL, NULL), handle(LogicalRegion::NO_REGION), 
        parent(NULL), row_source(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionNode& RegionNode::operator=(const RegionNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::has_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return (color_map.find(c) != color_map.end());
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
      // check to see if we have it, if not try to make it
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<Color,PartitionNode*>::const_iterator finder = 
          color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      // If we get here we didn't immediately have it so try
      // to make it through the proper channels
      IndexPartNode *index_part = row_source->get_child(c);
#ifdef DEBUG_HIGH_LEVEL
      assert(index_part != NULL);
#endif
      LogicalPartition part_handle(handle.tree_id, index_part->handle,
                                   handle.field_space);
      return context->create_node(part_handle, this);
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_child(PartitionNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(child->row_source->color) == color_map.end());
#endif
      color_map[child->row_source->color] = child;
      valid_map[child->row_source->color] = child;
    }

    //--------------------------------------------------------------------------
    void RegionNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      valid_map.erase(c);
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_creation_source(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      creation_set.set_bit(source);
    }

    //--------------------------------------------------------------------------
    void RegionNode::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->remove_child(row_source->color);
      AutoLock n_lock(node_lock);
      destruction_set.set_bit(source);
    }

    //--------------------------------------------------------------------------
    unsigned RegionNode::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->depth;
    }

    //--------------------------------------------------------------------------
    unsigned RegionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_tree_child(Color c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_children_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_all_children_disjoint(void)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::is_region(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    RegionNode* RegionNode::as_region_node(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<RegionNode*>(this);
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::as_partition_node(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::visit_node(PathTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      return traverser->visit_region(this);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::visit_node(NodeTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal = traverser->visit_region(this);
      if (continue_traversal)
      {
        std::map<Color,PartitionNode*> children;
        // Need to hold the lock when reading from 
        // the color map or the valid map
        if (traverser->visit_only_valid())
        {
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          children = valid_map;
        }
        else
        {
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          children = color_map;
        }
        for (std::map<Color,PartitionNode*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          bool result = it->second->visit_node(traverser);
          continue_traversal = continue_traversal && result;
        }
      }
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::has_component_domains(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->has_component_domains();
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& RegionNode::get_component_domains(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_component_domains();
    }

    //--------------------------------------------------------------------------
    Domain RegionNode::get_domain(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->domain;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::is_complete(void)
    //--------------------------------------------------------------------------
    {
      // For now just assume that regions are never complete
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::intersects_with(RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other->is_region())
        return row_source->intersects_with(other->as_region_node()->row_source);
      else
        return row_source->intersects_with(
                      other->as_partition_node()->row_source);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::dominates(RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other->is_region())
        return row_source->dominates(other->as_region_node()->row_source);
      else
        return row_source->dominates(other->as_partition_node()->row_source);
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& RegionNode::get_intersection_domains(
                                                          RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other->is_region())
        return row_source->get_intersection_domains(
                                           other->as_region_node()->row_source);
      else
        return row_source->get_intersection_domains(
                                        other->as_partition_node()->row_source);
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_num_children();
    }

    //--------------------------------------------------------------------------
    MaterializedView* RegionNode::create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth)
    //--------------------------------------------------------------------------
    {
      InstanceManager *manager = column_source->create_instance(target_mem,
                                                        row_source->domain,
                                                        fields,
                                                        blocking_factor, 
                                                        depth, this);
      // See if we made the instance
      MaterializedView *result = NULL;
      if (manager != NULL)
      {
        result = manager->create_top_view(depth);
#ifdef DEBUG_HIGH_LEVEL
        assert(result != NULL);
#endif
#ifdef LEGION_LOGGING
        LegionLogging::log_physical_instance(
            Machine::get_executing_processor(),
            manager->get_instance(), manager->memory,
            handle.index_space, handle.field_space, handle.tree_id);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_physical_instance(manager->get_instance().id,
            manager->memory.id, handle.index_space.id,
            handle.field_space.id, handle.tree_id);
#endif
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionView* RegionNode::create_reduction(Memory target_mem, FieldID fid,
                                                bool reduction_list,
                                                ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      ReductionManager *manager = column_source->create_reduction(target_mem,
                                                          row_source->domain,
                                                          fid, reduction_list,
                                                          this, redop);
      ReductionView *result = NULL;
      if (manager != NULL)
      {
        result = manager->create_view(); 
#ifdef DEBUG_HIGH_LEVEL
        assert(result != NULL);
#endif
#ifdef LEGION_LOGGING
        LegionLogging::log_physical_instance(
            Machine::get_executing_processor(),
            manager->get_instance(), manager->memory,
            handle.index_space, handle.field_space, handle.tree_id,
            redop, !reduction_list, manager->get_pointer_space());
#endif
#ifdef LEGION_SPY
        Domain ptr_space = manager->get_pointer_space();
        LegionSpy::log_physical_reduction(manager->get_instance().id,
            manager->memory.id, handle.index_space.id,
            handle.field_space.id, handle.tree_id, !reduction_list,
            ptr_space.exists() ? ptr_space.get_index_space().id : 0);
#endif
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      bool send_deletion = false;
      {
        AutoLock n_lock(node_lock); 
        if (!creation_set.is_set(target))
        {
          continue_up = true;
          creation_set.set_bit(target);
        }
        if (!destruction_set.is_set(target))
        {
          send_deletion = true;
          destruction_set.set_bit(target);
        }
      }
      if (continue_up)
      {
        if (parent != NULL)
          parent->send_node(target);
        else
        {
          // We've made it to the top, send this node
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
          }
          context->runtime->send_logical_region_node(target, rez);
        }
      }
      if (send_deletion)
      {
        context->runtime->send_logical_region_destruction(handle, target);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_node_creation(RegionTreeForest *context,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);

      RegionNode *node = context->create_node(handle, NULL/*parent*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(node != NULL);
#endif
      node->add_creation_source(source);
    } 

    //--------------------------------------------------------------------------
    void RegionNode::remap_region(ContextID ctx, MaterializedView *view,
                                  const FieldMask &user_mask, 
                                  FieldMask &needed_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REMAP_REGION_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
      assert(ctx < physical_state_size);
#endif
      PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);
      // We've already pre-mapped so we've pulled down
      // all the valid instance views.  Check to see if we
      // the target views is already there with the right
      // set of valid fields.
      std::map<InstanceView*,FieldMask>::const_iterator finder = 
        state.valid_views.find(view);
      if (finder == state.valid_views.end())
        needed_mask = user_mask;
      else
        needed_mask = user_mask - finder->second;
      release_physical_state(state);
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionNode::register_region(MappableInfo *info,
                                            PhysicalUser &user,
                                            LogicalView *view,
                                            const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REGISTER_REGION_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(view != NULL);
#endif
      
      // This mirrors the if-else statement in MappingTraverser::visit_region
      // for handling the different instance and reduction cases
      if (!IS_REDUCE(info->req))
      {
        MaterializedView *new_view = 
          view->as_instance_view()->as_materialized_view();
        // If the user requested that this view become persistent make it so
        if (info->req.make_persistent)
          new_view->make_persistent();

        // Issue updates for any fields which needed to be brought up
        // to date with the current versions of those fields
        // (assuming we are not write discard)
        bool needs_update_views = true;
        if (!IS_WRITE_ONLY(info->req) && !!needed_fields) 
        {
          if (IS_READ_ONLY(info->req))
          {
            // Here we need to handle a special case for read-only
            // updates.  We know that multiple tasks can be mapping
            // in parallel and we don't want to issue duplicate
            // copies to the same target physical instance, so we
            // maintain a set of pending updates to physical instances.
            // We then check to see if our needed_fields set can be
            // reduced and only issue copies for the difference.
            std::set<Event> pending_events;
            FieldMask actually_needed = needed_fields;
            PhysicalState &state = 
              acquire_physical_state(info->ctx, true/*exclusive*/);
            // Find any pending updates to this instance view
            find_pending_updates(state, new_view, 
                                 actually_needed, pending_events);
            // If we still have some fields to update, mark
            // that we are updating them and get the valid
            // views with which to perform the updates
            if (!!actually_needed)
            {
              UserEvent our_pending_event = UserEvent::create_user_event();
              state.pending_updates[new_view][our_pending_event] = 
                                                          actually_needed;
              std::map<InstanceView*,FieldMask> valid_views;
              find_valid_instance_views(state, actually_needed, 
                  actually_needed, false/*needs space*/, valid_views);
              release_physical_state(state);
              issue_update_copies(info, new_view, 
                                  actually_needed, valid_views);
              remove_valid_references(valid_views);
              // Reacquire the state so we can update the
              // valid views and remove our pending event 
              state = acquire_physical_state(info->ctx, true/*exclusive*/);
              // Update our valid views so we don't duplicately issue copies
              update_valid_views(state, actually_needed, 
                                 false/*dirty*/, new_view);
              // Then trigger our pending event and remove it 
              our_pending_event.trigger();
              std::map<Event,FieldMask> &pending = 
                                            state.pending_updates[new_view];
              pending.erase(our_pending_event);
              if (pending.empty())
                state.pending_updates.erase(new_view);
            }
            // Now we're done so we can release the state
            release_physical_state(state);
            // Mark that we no longer need to update views
            // since they will be all up-to-date by the time
            // we return from waiting.
            needs_update_views = false;
            // If we have anything to wait for, do that now
            if (!pending_events.empty())
            {
              Event wait_for = Event::merge_events(pending_events);
              wait_for.wait(true/*block*/);
            }
          }
          else
          {
            // This is the normal non-read-only case
            PhysicalState &state = 
              acquire_physical_state(info->ctx, false/*exclusive*/);
            std::map<InstanceView*,FieldMask> valid_views;
            find_valid_instance_views(state, needed_fields, needed_fields, 
                                      false/*needs space*/, valid_views);
            release_physical_state(state);
            issue_update_copies(info, new_view, needed_fields, valid_views);
            remove_valid_references(valid_views);
          }
        }

        // If we mapped the region close up any partitions
        // below that might have valid data that we need for
        // this instance.  We only need to do this for 
        // non-read-only tasks, since the read-only close
        // operations happened during the pre-mapping step.
        PhysicalState &state = 
          acquire_physical_state(info->ctx, true/*exclusive*/);
        if (!IS_READ_ONLY(info->req))
        {
          if (IS_WRITE_ONLY(info->req))
          {
            // If we're write only then we can just
            // invalidate everything below and update
            // the valid instance views.  Note we
            // can't be holding the physical state lock
            // when going down the tree so release it 
            // and then reacquire it
            release_physical_state(state);
            PhysicalInvalidator invalidator(info->ctx, user.field_mask,
                                            false/*force invalidate*/);
            visit_node(&invalidator);
            // Re-acquire the physical state
            acquire_physical_state(state, true/*exclusive*/);
            update_valid_views(state, user.field_mask, 
                               true/*dirty*/, new_view);
          }
          else
          {
            PhysicalCloser closer(info, false/*leave open*/, handle);
            closer.add_target(new_view);
            // Mark the dirty mask with our bits since we're 
            closer.update_dirty_mask(user.field_mask);
            // writing and the closer will 
            bool create_composite = false;
#ifdef DEBUG_HIGH_LEVEL
            bool result = 
#endif
            siphon_physical_children(closer, state, user.field_mask,
                                      -1/*next child*/, create_composite);
#ifdef DEBUG_HIGH_LEVEL
            assert(result); // should always succeed
            assert(!create_composite);
#endif
            // Now update the valid views and the dirty mask
            closer.update_node_views(this, state);
          }
        }
        else if (needs_update_views)
        {
          // Otherwise just issue a non-dirty update for the user fields
          update_valid_views(state, user.field_mask,
                             false/*dirty*/, new_view);
        }
        // Release our hold on the state
        release_physical_state(state);
        // Flush any reductions that need to be flushed
        flush_reductions(user.field_mask,
                         info->req.redop, info);
        // Now add ourselves as a user of this region
        return new_view->add_user(user, info->local_proc);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        // should never have needed fields for reductions
        assert(!needed_fields); 
#endif
        ReductionView *new_view = view->as_reduction_view();
        // Flush any reductions that need to be flushed
        flush_reductions(user.field_mask,
                         info->req.redop, info);
        PhysicalState &state = 
          acquire_physical_state(info->ctx, true/*exclusive*/);
        // If there was a needed close for this reduction then
        // it was performed as part of the premapping operation
        update_reduction_views(state, user.field_mask, new_view);
        // Release our hold on the state
        release_physical_state(state);
        // Now we can add ourselves as a user of this region
        return new_view->add_user(user, info->local_proc);
      }
    }

    //--------------------------------------------------------------------------
    InstanceRef RegionNode::seed_state(ContextID ctx, PhysicalUser &user,
                                       LogicalView *new_view,
                                       Processor local_proc)
    //--------------------------------------------------------------------------
    {
      PhysicalState &state = acquire_physical_state(ctx, true/*exclusive*/);
      if (new_view->is_reduction_view())
      {
        ReductionView *view = new_view->as_reduction_view();
        update_reduction_views(state, user.field_mask, view);
        view->add_user(user, local_proc);
      }
      else
      {
        InstanceView *view = new_view->as_instance_view();
        update_valid_views(state, user.field_mask, 
                           HAS_WRITE(user.usage), view);
        view->add_user(user, local_proc);
      }
      release_physical_state(state);
      return InstanceRef(Event::NO_EVENT, 
                         Reservation::NO_RESERVATION, new_view);
    } 

    //--------------------------------------------------------------------------
    Event RegionNode::close_state(MappableInfo *info, PhysicalUser &user,
                                  const InstanceRef &target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, CLOSE_PHYSICAL_STATE_CALL);
#endif
      LogicalView *view = target.get_handle().get_view(); 
      if (view->is_reduction_view())
      {
        ReductionView *target_view = view->as_reduction_view();
        ReductionCloser closer(info->ctx, target_view, 
                               user.field_mask, info->local_proc);
        visit_node(&closer);
        InstanceRef result = target_view->add_user(user, info->local_proc);
#ifdef DEBUG_HIGH_LEVEL
        assert(result.has_ref());
#endif
        return result.get_ready_event();
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(info->ctx < physical_state_size);
#endif
        MaterializedView *target_view = 
                          view->as_instance_view()->as_materialized_view();
        PhysicalState &state = 
          acquire_physical_state(info->ctx, true/*exclusive*/);
        // First check to see if we are in the set of valid views, if
        // not then we need to issue updates for all of our fields
        std::map<InstanceView*,FieldMask>::const_iterator finder = 
          state.valid_views.find(target_view);
        if ((finder == state.valid_views.end()) || 
            !!(user.field_mask - finder->second))
        {
          FieldMask update_mask = user.field_mask;
          if (finder != state.valid_views.end())
            update_mask -= finder->second;
          std::map<InstanceView*,FieldMask> valid_views;
          find_valid_instance_views(state, update_mask, update_mask,
                                    false/*needs space*/, valid_views);
          release_physical_state(state);
          issue_update_copies(info, target_view, update_mask, valid_views);
          remove_valid_references(valid_views);
          acquire_physical_state(state, true/*exclusive*/); 
        }
        // Now do the close to this physical instance
        PhysicalCloser closer(info, false/*leave open*/, handle);
        closer.add_target(target_view);
        closer.update_dirty_mask(user.field_mask);
        bool create_composite = false;
#ifdef DEBUG_HIGH_LEVEL
        bool temp_result = 
#endif
        siphon_physical_children(closer, state, user.field_mask, 
                                 -1/*next child*/, create_composite); 
#ifdef DEBUG_HIGH_LEVEL
        assert(temp_result); // should always succeed
        assert(!create_composite);
#endif
        // Now update the valid views
        closer.update_node_views(this, state);
        // Release our hold on the physical state
        release_physical_state(state);
        flush_reductions(user.field_mask,
                         info->req.redop, info);
        // Get the resulting instance reference
        InstanceRef result = target_view->add_user(user, info->local_proc);
#ifdef DEBUG_HIGH_LEVEL
        assert(result.has_ref());
#endif
        return result.get_ready_event();
      }
    }

    //--------------------------------------------------------------------------
    bool RegionNode::send_state(ContextID ctx, UniqueID uid, 
                                AddressSpaceID target,
                                const FieldMask &send_mask, bool invalidate,
                                std::map<LogicalView*,FieldMask> &needed_views,
                                std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(uid);
        rez.serialize(handle);
        // Now pack up the rest of the state
        continue_traversal = pack_send_state(ctx, rez, target, send_mask, 
                                             needed_views, needed_managers);
      }
      // Send the message
      context->runtime->send_region_state(target, rez);
      // If we're supposed to invalidate the state, do that now
      if (invalidate)
        invalidate_physical_state(ctx, send_mask, false/*force*/);
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_send_state(RegionTreeForest *context,
                                          Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID uid;
      derez.deserialize(uid);
      LogicalRegion handle;
      derez.deserialize(handle);
      RemoteTask *remote_ctx = 
        context->runtime->find_or_init_remote_context(uid);
      RegionTreeContext ctx = remote_ctx->get_context();
      RegionNode *node = context->get_node(handle);
      // Now do the unpack
      node->unpack_send_state(ctx.get_id(), derez, node->column_source, source);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::send_back_state(ContextID ctx, ContextID remote_ctx,
                                    AddressSpaceID target,
                                    bool invalidate, const FieldMask &send_mask,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ctx);
        rez.serialize(handle);
        rez.serialize(invalidate);
        if (invalidate)
          rez.serialize(send_mask);
        continue_traversal = 
          pack_send_back_state(ctx, rez, target, send_mask, needed_managers);
      }
      context->runtime->send_back_region_state(target, rez);
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_send_back_state(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ContextID ctx;
      derez.deserialize(ctx);
      LogicalRegion handle;
      derez.deserialize(handle);
      bool invalidate;
      derez.deserialize(invalidate);
      RegionNode *node = context->get_node(handle);
      FieldMask invalidate_mask;
      if (invalidate)
      {
        derez.deserialize(invalidate_mask);
        // Transform field mask to our node 
        node->column_source->transform_field_mask(invalidate_mask, source);
        // Perform the invalidation
        node->invalidate_physical_state(ctx, invalidate_mask, false/*force*/);
      }
      
      // Note we don't need a separate routine for unpack send back state
      node->unpack_send_state(ctx, derez, node->column_source, source);
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_logical_context(ContextID ctx, 
                                           TreeStateLogger *logger,
                                           const FieldMask &capture_mask) 
    //--------------------------------------------------------------------------
    {
      logger->log("Region Node (" IDFMT ",%d,%d) Color %d at depth %d", 
          handle.index_space.id, handle.field_space.id,handle.tree_id,
          row_source->color, logger->get_depth());
      logger->down();
      std::map<Color,FieldMask> to_traverse;
#ifdef DEBUG_HIGH_LEVEL
      if (ctx < logical_state_size)
#else
      if (ctx < logical_states.size())
#endif
      {
        LogicalState &state = logical_states[ctx];
        print_logical_state(state, capture_mask, to_traverse, logger);  
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_logical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_physical_context(ContextID ctx, 
                                            TreeStateLogger *logger,
                                            const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Region Node (" IDFMT ",%d,%d) Color %d at depth %d", 
          handle.index_space.id, handle.field_space.id,handle.tree_id,
          row_source->color, logger->get_depth());
      logger->down();
      std::map<Color,FieldMask> to_traverse;
#ifdef DEBUG_HIGH_LEVEL
      if (ctx < physical_state_size)
#else
      if (ctx < physical_states.size())
#endif
      {
        PhysicalState &state = acquire_physical_state(ctx, false/*exclusive*/);
        print_physical_state(state, capture_mask, to_traverse, logger);
        release_physical_state(state);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionNode::print_logical_state(LogicalState &state,
                                         const FieldMask &capture_mask,
                                         std::map<Color,FieldMask> &to_traverse,
                                         TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      // Open Field States 
      {
        logger->log("Open Field States (%ld)", state.field_states.size());
        logger->down();
        for (std::list<FieldState>::const_iterator it = 
              state.field_states.begin(); it != 
              state.field_states.end(); it++)
        {
          it->print_state(logger, capture_mask);
          if (it->valid_fields * capture_mask)
            continue;
          for (std::map<Color,FieldMask>::const_iterator cit = 
                it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
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
    
    //--------------------------------------------------------------------------
    void RegionNode::print_physical_state(PhysicalState &state,
                                         const FieldMask &capture_mask,
                                         std::map<Color,FieldMask> &to_traverse,
                                         TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
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
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              state.valid_views.begin(); it != state.valid_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        logger->log("Valid Instances (%d)", num_valid);
        logger->down();
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              state.valid_views.begin(); it != state.valid_views.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          if (it->first->is_composite_view())
            continue;
          MaterializedView *current = it->first->as_materialized_view();
          char *valid_mask = overlap.to_string();
          logger->log("Instance " IDFMT "   Memory " IDFMT "   Mask %s",
                      current->manager->get_instance().id, 
                      current->manager->memory.id, valid_mask);
          free(valid_mask);
        }
        logger->up();
      }
      // Valid Reduction Views
      {
        unsigned num_valid = 0;
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              state.reduction_views.begin(); it != 
              state.reduction_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        logger->log("Valid Reduction Instances (%d)", num_valid);
        logger->down();
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              state.reduction_views.begin(); it != 
              state.reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          char *valid_mask = overlap.to_string();
          logger->log("Reduction Instance " IDFMT "   Memory " IDFMT 
                      "  Mask %s",
                      it->first->manager->get_instance().id, 
                      it->first->manager->memory.id, valid_mask);
          free(valid_mask);
        }
        logger->up();
      }
      // Open Children
      {
        logger->log("Open Children (%ld)", 
            state.children.open_children.size());
        logger->down();
        for (std::map<Color,FieldMask>::const_iterator it = 
              state.children.open_children.begin(); it !=
              state.children.open_children.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          char *mask_buffer = overlap.to_string();
          logger->log("Color %d   Mask %s", it->first, mask_buffer);
          free(mask_buffer);
          // Mark that we should traverse this child
          to_traverse[it->first] = overlap;
        }
        logger->up();
      }
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void RegionNode::dump_logical_context(ContextID ctx, 
                                          TreeStateLogger *logger,
                                          const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Region Node (" IDFMT ",%d,%d) Color %d at depth %d (%p)", 
          handle.index_space.id, handle.field_space.id,handle.tree_id,
          row_source->color, logger->get_depth(), this);
      logger->down();
      std::map<Color,FieldMask> to_traverse;
      if (ctx < logical_state_size)
        print_logical_state(logical_states[ctx], capture_mask,
                            to_traverse, logger);
      else
        logger->log("No state");
      logger->log("");
      if (!to_traverse.empty())
      {
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_logical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void RegionNode::dump_physical_context(ContextID ctx,
                                           TreeStateLogger *logger,
                                           const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Region Node (" IDFMT ",%d,%d) Color %d at depth %d (%p)", 
          handle.index_space.id, handle.field_space.id,handle.tree_id,
          row_source->color, logger->get_depth(), this);
      logger->down();
      std::map<Color,FieldMask> to_traverse;
      if (ctx < physical_state_size)
        print_physical_state(physical_states[ctx], capture_mask,
                             to_traverse, logger);
      else
        logger->log("No state");
      logger->log("");
      if (!to_traverse.empty())
      {
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,PartitionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }
#endif

    /////////////////////////////////////////////////////////////
    // Partition Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(LogicalPartition p, RegionNode *par,
                                 IndexPartNode *row_src, 
                                 FieldSpaceNode *col_src,
                                 RegionTreeForest *ctx)
      : RegionTreeNode(ctx, col_src), handle(p), parent(par),
        row_source(row_src), disjoint(row_src->disjoint)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(const PartitionNode &rhs)
      : RegionTreeNode(NULL, NULL), handle(LogicalPartition::NO_PART),
        parent(NULL), row_source(NULL), disjoint(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PartitionNode::~PartitionNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PartitionNode& PartitionNode::operator=(const PartitionNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::has_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      return (color_map.find(c) != color_map.end());
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::get_child(Color c)
    //--------------------------------------------------------------------------
    {
      // check to see if we have it, if not try to make it
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        std::map<Color,RegionNode*>::const_iterator finder = 
          color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      // If we get here we didn't immediately have it so try
      // to make it through the proper channels
      IndexSpaceNode *index_node = row_source->get_child(c);
#ifdef DEBUG_HIGH_LEVEL
      assert(index_node != NULL);
#endif
      LogicalRegion reg_handle(handle.tree_id, index_node->handle,
                               handle.field_space);
      return context->create_node(reg_handle, this);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_child(RegionNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(color_map.find(child->row_source->color) == color_map.end());
#endif
      color_map[child->row_source->color] = child;
      valid_map[child->row_source->color] = child;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::remove_child(Color c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      valid_map.erase(c);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_creation_source(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      creation_set.set_bit(source);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::destroy_node(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->remove_child(row_source->color);
      AutoLock n_lock(node_lock);
      destruction_set.set_bit(source);
    }

    //--------------------------------------------------------------------------
    unsigned PartitionNode::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->depth;
    }

    //--------------------------------------------------------------------------
    unsigned PartitionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_tree_child(Color c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_children_disjoint(Color c1, Color c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_all_children_disjoint(void)
    //--------------------------------------------------------------------------
    {
      return row_source->disjoint;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::is_region(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::as_region_node(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    PartitionNode* PartitionNode::as_partition_node(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<PartitionNode*>(this);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::visit_node(PathTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      return traverser->visit_partition(this);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::visit_node(NodeTraverser *traverser)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal = traverser->visit_partition(this);
      if (continue_traversal)
      {
        std::map<Color,RegionNode*> children;
        // Need to hold the lock when reading from 
        // the color map or the valid map
        if (traverser->visit_only_valid())
        {
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          children = valid_map;
        }
        else
        {
          AutoLock n_lock(node_lock,1,false/*exclusive*/);
          children = color_map;
        }
        for (std::map<Color,RegionNode*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          bool result = it->second->visit_node(traverser);
          continue_traversal = continue_traversal && result;
        }
      }
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::has_component_domains(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      return parent->has_component_domains();    
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& PartitionNode::get_component_domains(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      return parent->get_component_domains();
    }

    //--------------------------------------------------------------------------
    Domain PartitionNode::get_domain(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      return parent->get_domain();
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::is_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      return row_source->is_complete();
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::intersects_with(RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other->is_region())
        return row_source->intersects_with(other->as_region_node()->row_source);
      else
        return row_source->intersects_with(
                                        other->as_partition_node()->row_source);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::dominates(RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other->is_region())
        return row_source->dominates(other->as_region_node()->row_source);
      else
        return row_source->dominates(other->as_partition_node()->row_source);
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& PartitionNode::get_intersection_domains(
                                                          RegionTreeNode *other)
    //--------------------------------------------------------------------------
    {
      if (other->is_region())
        return row_source->get_intersection_domains(
                                           other->as_region_node()->row_source);
      else
        return row_source->get_intersection_domains(
                                        other->as_partition_node()->row_source);
    }

    //--------------------------------------------------------------------------
    size_t PartitionNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_num_children();
    }

    //--------------------------------------------------------------------------
    MaterializedView* PartitionNode::create_instance(Memory target_mem,
                                                const std::set<FieldID> &fields,
                                                size_t blocking_factor,
                                                unsigned depth)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      MaterializedView *result = parent->create_instance(target_mem, 
                                                         fields, 
                                                         blocking_factor,
                                                         depth);
      if (result != NULL)
      {
        result = result->get_materialized_subview(row_source->color);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionView* PartitionNode::create_reduction(Memory target_mem, 
                                              FieldID fid, bool reduction_list,
                                              ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      return parent->create_reduction(target_mem, fid, reduction_list, redop);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      bool send_deletion = false;
      {
        AutoLock n_lock(node_lock); 
        if (!creation_set.is_set(target))
        {
          continue_up = true;
          creation_set.set_bit(target);
        }
        if (!destruction_set.is_set(target))
        {
          send_deletion = true;
          destruction_set.set_bit(target);
        }
      }
      if (continue_up)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent != NULL);
#endif
        parent->send_node(target);
      }
      if (send_deletion)
      {
        context->runtime->send_logical_partition_destruction(handle, target);
      }
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::send_state(ContextID ctx, UniqueID uid,
                                   AddressSpaceID target,
                                   const FieldMask &send_mask, bool invalidate,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(uid);
        rez.serialize(handle);
        // Now pack up the rest of the state
        continue_traversal = pack_send_state(ctx, rez, target, send_mask,
                                             needed_views, needed_managers);
      }
      context->runtime->send_partition_state(target, rez);
      if (invalidate)
        invalidate_physical_state(ctx, send_mask, false/*force*/);
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PartitionNode::handle_send_state(RegionTreeForest *context,
                                         Deserializer &derez, 
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID uid;
      derez.deserialize(uid);
      LogicalPartition handle;
      derez.deserialize(handle);
      RemoteTask *remote_ctx = 
        context->runtime->find_or_init_remote_context(uid);
      RegionTreeContext ctx = remote_ctx->get_context();
      PartitionNode *node = context->get_node(handle);
      // Now do the unpack
      node->unpack_send_state(ctx.get_id(), derez, node->column_source, source);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::send_back_state(ContextID ctx, ContextID remote_ctx,
                                        AddressSpaceID target,
                                        bool invalidate, 
                                        const FieldMask &send_mask,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ctx);
        rez.serialize(handle);
        rez.serialize(invalidate);
        if (invalidate)
          rez.serialize(send_mask);
        // Now pack up the rest of the state
        continue_traversal = 
          pack_send_back_state(ctx, rez, target, send_mask, needed_managers);
      }
      context->runtime->send_back_partition_state(target, rez);
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PartitionNode::handle_send_back_state(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ContextID ctx;
      derez.deserialize(ctx);
      LogicalPartition handle;
      derez.deserialize(handle);
      bool invalidate;
      derez.deserialize(invalidate);
      FieldMask invalidate_mask;
      PartitionNode *node = context->get_node(handle);
      if (invalidate)
      {
        derez.deserialize(invalidate_mask);
        // Transform the field mask to our node 
        node->column_source->transform_field_mask(invalidate_mask, source);
        // Perform the invalidation 
        node->invalidate_physical_state(ctx, invalidate_mask, false/*force*/);
      }
      
      // Note we don't need a separate routine for unpack send back state
      node->unpack_send_state(ctx, derez, node->column_source, source);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_logical_context(ContextID ctx,
                                              TreeStateLogger *logger,
                                              const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (%d,%d,%d) Color %d disjoint %d at depth %d",
          handle.index_partition, handle.field_space.id, handle.tree_id, 
          row_source->color, disjoint, logger->get_depth());
      logger->down();
      std::map<Color,FieldMask> to_traverse;
#ifdef DEBUG_HIGH_LEVEL
      if (ctx < logical_state_size)
#else
      if (ctx < logical_states.size())
#endif
      {
        LogicalState &state = logical_states[ctx];
        print_logical_state(state, capture_mask, to_traverse, logger);    
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,RegionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_logical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_physical_context(ContextID ctx,
                                               TreeStateLogger *logger,
                                               const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (%d,%d,%d) Color %d disjoint %d at depth %d",
          handle.index_partition, handle.field_space.id, handle.tree_id, 
          row_source->color, disjoint, logger->get_depth());
      logger->down();
      std::map<Color,FieldMask> to_traverse;
#ifdef DEBUG_HIGH_LEVEL
      if (ctx < physical_state_size)
#else
      if (ctx < physical_states.size())
#endif
      {
        PhysicalState &state = acquire_physical_state(ctx, false/*exclusive*/);
        print_physical_state(state, capture_mask, to_traverse, logger);
        release_physical_state(state);    
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,RegionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->print_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::print_logical_state(LogicalState &state,
                                        const FieldMask &capture_mask,
                                        std::map<Color,FieldMask> &to_traverse,
                                        TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      // Open Field States
      {
        logger->log("Open Field States (%ld)", state.field_states.size()); 
        logger->down();
        for (std::list<FieldState>::const_iterator it = 
              state.field_states.begin(); it != 
              state.field_states.end(); it++)
        {
          it->print_state(logger, capture_mask);
          if (it->valid_fields * capture_mask)
            continue;
          for (std::map<Color,FieldMask>::const_iterator cit = 
                it->open_children.begin(); cit != 
                it->open_children.end(); cit++)
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

    //--------------------------------------------------------------------------
    void PartitionNode::print_physical_state(PhysicalState &state,
                                         const FieldMask &capture_mask,
                                         std::map<Color,FieldMask> &to_traverse,
                                         TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
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
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              state.valid_views.begin(); it != state.valid_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        logger->log("Valid Instances (%d)", num_valid);
        logger->down();
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              state.valid_views.begin(); it != state.valid_views.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          if (it->first->is_composite_view())
            continue;
          MaterializedView *current = it->first->as_materialized_view();
          char *valid_mask = overlap.to_string();
          logger->log("Instance " IDFMT "   Memory " IDFMT "   Mask %s",
                      current->manager->get_instance().id, 
                      current->manager->memory.id, valid_mask);
          free(valid_mask);
        }
        logger->up();
      }
      // Valid Reduction Views
      {
        unsigned num_valid = 0;
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              state.reduction_views.begin(); it != 
              state.reduction_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        logger->log("Valid Reduction Instances (%d)", num_valid);
        logger->down();
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              state.reduction_views.begin(); it != 
              state.reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          char *valid_mask = overlap.to_string();
          logger->log("Reduction Instance " IDFMT "   Memory " IDFMT 
                      "  Mask %s",
                      it->first->manager->get_instance().id, 
                      it->first->manager->memory.id, valid_mask);
          free(valid_mask);
        }
        logger->up();
      }
      // Open Children
      {
        logger->log("Open Children (%ld)", 
            state.children.open_children.size());
        logger->down();
        for (std::map<Color,FieldMask>::const_iterator it = 
              state.children.open_children.begin(); it !=
              state.children.open_children.end(); it++)
        {
          FieldMask overlap = it->second & capture_mask;
          if (!overlap)
            continue;
          char *mask_buffer = overlap.to_string();
          logger->log("Color %d   Mask %s", it->first, mask_buffer);
          free(mask_buffer);
          // Mark that we should traverse this child
          to_traverse[it->first] = overlap;
        }
        logger->up();
      }
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void PartitionNode::dump_logical_context(ContextID ctx,
                                             TreeStateLogger *logger,
                                             const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (%d,%d,%d) Color %d disjoint %d " 
                  "at depth %d (%p)",
          handle.index_partition, handle.field_space.id, handle.tree_id, 
          row_source->color, disjoint, logger->get_depth(), this);
      logger->down();
      std::map<Color,FieldMask> to_traverse;
      if (ctx < logical_state_size)
      {
        LogicalState &state = logical_states[ctx];
        print_logical_state(state, capture_mask, to_traverse, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,RegionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_logical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::dump_physical_context(ContextID ctx,
                                              TreeStateLogger *logger,
                                              const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      logger->log("Partition Node (%d,%d,%d) Color %d disjoint %d "
                  "at depth %d (%p)",
          handle.index_partition, handle.field_space.id, handle.tree_id, 
          row_source->color, disjoint, logger->get_depth(), this);
      logger->down();
      std::map<Color,FieldMask> to_traverse;
#ifdef DEBUG_HIGH_LEVEL
      if (ctx < physical_state_size)
#else
      if (ctx < physical_states.size())
#endif
      {
        PhysicalState &state = physical_states[ctx];
        print_physical_state(state, capture_mask, to_traverse, logger);
      }
      else
      {
        logger->log("No state");
      }
      logger->log("");
      if (!to_traverse.empty())
      {
        for (std::map<Color,FieldMask>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          std::map<Color,RegionNode*>::const_iterator finder = 
            color_map.find(it->first);
          if (finder != color_map.end())
            finder->second->dump_physical_context(ctx, logger, it->second);
        }
      }
      logger->up();
    }
#endif

    /////////////////////////////////////////////////////////////
    // RegionTreePath 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreePath::RegionTreePath(void) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::initialize(unsigned min, unsigned max)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min <= max);
#endif
      min_depth = min;
      max_depth = max;
      path.resize(max_depth+1);
      for (unsigned idx = 0; idx < path.size(); idx++)
        path[idx] = -1;
      close_ops.resize(max_depth+1);
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::register_child(unsigned depth, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      path[depth] = color;
    }

    //--------------------------------------------------------------------------
    bool RegionTreePath::has_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      return (path[depth] >= 0);
    }

    //--------------------------------------------------------------------------
    Color RegionTreePath::get_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      assert(has_child(depth));
#endif
      return Color(path[depth]);
    }

    //--------------------------------------------------------------------------
    unsigned RegionTreePath::get_path_length(void) const
    //--------------------------------------------------------------------------
    {
      return ((max_depth-min_depth)+1); 
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::record_close_operation(unsigned depth,
                                                const TreeClose &info,
                                                const FieldMask &close_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif     
      close_ops[depth].push_back(
          std::pair<TreeClose,FieldMask>(info, close_mask));
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::record_before_version(unsigned depth, VersionID vid,
                                               const FieldMask &version_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      // TODO: record version numbers
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::record_after_version(unsigned depth, VersionID vid,
                                              const FieldMask &version_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      // TODO: record version numbers
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::get_close_operations(unsigned depth,
                                              std::deque<CloseInfo> &needed_ops)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif     
      std::deque<std::pair<TreeClose,FieldMask> > &ops = close_ops[depth];
      for (std::deque<std::pair<TreeClose,FieldMask> >::iterator it = 
            ops.begin(); it != ops.end(); it++)
      {
        it->first.add_close_op(it->second, needed_ops);
      }
    }

    /////////////////////////////////////////////////////////////
    // Layout Description 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(const FieldMask &mask, const Domain &d,
                                         size_t bf, FieldSpaceNode *own)
      : allocated_fields(mask), blocking_factor(bf), 
        volume(compute_layout_volume(d)), owner(own)
    //--------------------------------------------------------------------------
    {
      offset_lock = Reservation::create_reservation();
    }

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(const LayoutDescription &rhs)
      : allocated_fields(rhs.allocated_fields), 
        blocking_factor(rhs.blocking_factor), 
        volume(rhs.volume), owner(rhs.owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LayoutDescription::~LayoutDescription(void)
    //--------------------------------------------------------------------------
    {
      memoized_offsets.clear();
      offset_lock.destroy_reservation();
      offset_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    LayoutDescription& LayoutDescription::operator=(
                                                   const LayoutDescription &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(const FieldMask &copy_mask,
                                                 PhysicalInstance instance,
                                   std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      uint64_t hash_key = copy_mask.get_hash_key();
      size_t added_offset_count = 0;
      bool found_in_cache = false;
      // First check to see if we've memoized this result 
      {
        AutoLock o_lock(offset_lock,1,false/*exclusive*/);
        std::map<FIELD_TYPE,std::vector<OffsetEntry> >::const_iterator
          finder = memoized_offsets.find(hash_key);
        if (finder != memoized_offsets.end())
        {
          for (std::vector<OffsetEntry>::const_iterator it = 
                finder->second.begin(); it != finder->second.end(); it++)
          {
            if (it->offset_mask == copy_mask)
            {
              fields.insert(fields.end(),it->offsets.begin(),it->offsets.end());
              found_in_cache = true;
              added_offset_count = it->offsets.size();
              break;
            }
          }
        }
      }
      if (found_in_cache)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(added_offset_count <= fields.size());
#endif
        // Go through and fill in all the annonymous instances
        for (unsigned idx = fields.size() - added_offset_count;
              idx < fields.size(); idx++)
        {
          fields[idx].inst = instance;
        }
        // Now we're done
        return;
      }
      // It is absolutely imperative that these infos be added in
      // the order in which they appear in the field mask so that 
      // they line up in the same order with the source/destination infos
      // (depending on the calling context of this function)
#ifdef DEBUG_HIGH_LEVEL
      int pop_count = 0;
#endif
      std::vector<Domain::CopySrcDstField> local;
      for (std::map<unsigned,FieldID>::const_iterator it = 
            field_indexes.begin(); it != field_indexes.end(); it++)
      {
        if (copy_mask.is_set(it->first))
        {
          std::map<FieldID,Domain::CopySrcDstField>::const_iterator finder = 
            field_infos.find(it->second);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != field_infos.end());
          pop_count++;
#endif
          fields.push_back(finder->second);
          // Because instances are annonymous in layout descriptions
          // we have to fill them in as we add them to fields
          fields.back().inst = instance;
          local.push_back(finder->second);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      // Make sure that we added exactly the number of infos as
      // there were fields set in the bit mask
      assert(pop_count == FieldMask::pop_count(copy_mask));
#endif
      // Add this to the results
      AutoLock o_lock(offset_lock);
      std::map<FIELD_TYPE,std::vector<OffsetEntry> >::iterator
        finder = memoized_offsets.find(hash_key);
      if (finder == memoized_offsets.end())
        memoized_offsets[hash_key].push_back(OffsetEntry(copy_mask,local));
      else
        finder->second.push_back(OffsetEntry(copy_mask,local));
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(
                                   const std::vector<FieldID> &copy_fields, 
                                   PhysicalInstance instance,
                                   std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      for (std::vector<FieldID>::const_iterator it = copy_fields.begin();
            it != copy_fields.end(); it++)
      {
        std::map<FieldID,Domain::CopySrcDstField>::const_iterator
          finder = field_infos.find(*it);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != field_infos.end());
#endif
        fields.push_back(finder->second);
        // Since instances are annonymous in layout descriptions we
        // have to fill them in when we add the field info
        fields.back().inst = instance;
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::add_field_info(FieldID fid, unsigned index,
                                           size_t offset, size_t field_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(field_infos.find(fid) == field_infos.end());
      assert(field_indexes.find(index) == field_indexes.end());
#endif
      // Use annonymous instances when creating these field infos since
      // we specifying layouts independently of any one instance
      field_infos[fid] = Domain::CopySrcDstField(PhysicalInstance::NO_INST,
                                                 offset, field_size);
      field_indexes[index] = fid;
#ifdef DEBUG_HIGH_LEVEL
      assert(offset_size_map.find(offset) == offset_size_map.end());
#endif
      offset_size_map[offset] = field_size;
    }

    //--------------------------------------------------------------------------
    const Domain::CopySrcDstField& LayoutDescription::find_field_info(
                                                              FieldID fid) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,Domain::CopySrcDstField>::const_iterator finder = 
        field_infos.find(fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != field_infos.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    size_t LayoutDescription::get_layout_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      // Add up all the field sizes
      for (std::map<FieldID,Domain::CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        result += (it->second.size);
      }
      result *= volume;
      return result;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_shape(const size_t field_size) const
    //--------------------------------------------------------------------------
    {
      if (field_infos.size() != 1)
        return false;
      if (field_infos.begin()->second.size != field_size)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_shape(const std::vector<size_t> &field_sizes,
                                        const size_t bf) const
    //--------------------------------------------------------------------------
    {
      if (field_sizes.size() != field_infos.size())
        return false;
      if (blocking_factor != bf)
        return false;
      unsigned offset;
      for (std::vector<size_t>::const_iterator it = field_sizes.begin();
            it != field_sizes.end(); it++)
      {
        std::map<unsigned,unsigned>::const_iterator finder = 
          offset_size_map.find(offset);
        // Check to make sure we have the same offset
        if (finder == offset_size_map.end())
          return false;
        // Check that the sizes are the same for the offset
        if (finder->second != (*it))
          return false;
        offset += (*it);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(const FieldMask &mask,
                                         const size_t vl, const size_t bf) const
    //--------------------------------------------------------------------------
    {
      if (blocking_factor != bf)
        return false;
      if (volume != vl)
        return false;
      if (allocated_fields != mask)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(const FieldMask &mask, const Domain &d,
                                         const size_t bf) const
    //--------------------------------------------------------------------------
    {
      return match_layout(mask, compute_layout_volume(d), bf);
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(LayoutDescription *rhs) const
    //--------------------------------------------------------------------------
    {
      return match_layout(rhs->allocated_fields, rhs->volume, 
                          rhs->blocking_factor);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::pack_layout_description(Serializer &rez,
                                                    AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      // Do a quick check to see if the target already has the layout
      // We don't need to hold a lock here since if we lose the race
      // we will just send the layout twice and everything will be
      // resolved on the far side
      if (known_nodes.is_set(target))
      {
        rez.serialize<bool>(true);
        // If it is already on the remote node, then we only
        // need to the necessary information to identify it
        rez.serialize(allocated_fields);
        rez.serialize(blocking_factor);
      }
      else
      {
        rez.serialize<bool>(false);
        rez.serialize(allocated_fields);
        rez.serialize(blocking_factor);
        rez.serialize<size_t>(field_infos.size());
#ifdef DEBUG_HIGH_LEVEL
        assert(field_infos.size() == field_indexes.size());
#endif
        for (std::map<unsigned,FieldID>::const_iterator it = 
              field_indexes.begin(); it != field_indexes.end(); it++)
        {
          std::map<FieldID,Domain::CopySrcDstField>::const_iterator finder = 
            field_infos.find(it->second);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != field_infos.end());
#endif
          rez.serialize(it->second);
          rez.serialize(finder->second.offset);
          rez.serialize(finder->second.size);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::unpack_layout_description(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_fields;
      derez.deserialize(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        unsigned index = owner->get_field_index(fid);
        field_indexes[index] = fid;
        unsigned offset, size;
        derez.deserialize(offset);
        derez.deserialize(size);
        field_infos[fid] = 
          Domain::CopySrcDstField(PhysicalInstance::NO_INST, offset, size);
#ifdef DEBUG_HIGH_LEVEL
        assert(offset_size_map.find(offset) == offset_size_map.end());
#endif
        offset_size_map[offset] = size;
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::update_known_nodes(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock here since races are alright
      known_nodes.set_bit(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutDescription* LayoutDescription::
      handle_unpack_layout_description(Deserializer &derez,
                                 AddressSpaceID source, RegionNode *region_node)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      bool has_local;
      derez.deserialize(has_local);
      FieldSpaceNode *field_space_node = region_node->column_source;
      LayoutDescription *result = NULL;
      FieldMask mask;
      derez.deserialize(mask);
      field_space_node->transform_field_mask(mask, source);
      size_t blocking_factor;
      derez.deserialize(blocking_factor);
      if (has_local)
      {
        // If we have a local layout, then we should be able to find it
        result = field_space_node->find_layout_description(mask,  
                                              region_node->get_domain(),
                                              blocking_factor);
      }
      else
      {
        // Otherwise create a new layout description, 
        // unpack it, and then try registering it with
        // the field space node
        result = new LayoutDescription(mask, region_node->get_domain(),
                                       blocking_factor, field_space_node);
        result->unpack_layout_description(derez);
        result = field_space_node->register_layout_description(result);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      // Record that the sender already has this layout
      // Only do this after we've registered the instance
      result->update_known_nodes(source);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ size_t LayoutDescription::compute_layout_volume(const Domain &d)
    //--------------------------------------------------------------------------
    {
      if (d.get_dim() == 0)
      {
        const LowLevel::ElementMask &mask = 
          d.get_index_space().get_valid_mask();
        return mask.get_num_elmts();
      }
      else
        return d.get_volume();
    }

    /////////////////////////////////////////////////////////////
    // PhysicalManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalManager::PhysicalManager(RegionTreeForest *ctx, DistributedID did,
                                     AddressSpaceID owner_space,
                                     AddressSpaceID local_space,
                                     Memory mem, PhysicalInstance inst)
      : DistributedCollectable(ctx->runtime, did, owner_space, local_space), 
        context(ctx), memory(mem), instance(inst)
    //--------------------------------------------------------------------------
    {
      context->register_physical_manager(this); 
    }

    //--------------------------------------------------------------------------
    PhysicalManager::~PhysicalManager(void)
    //--------------------------------------------------------------------------
    {
      context->unregister_physical_manager(this->did);
      if (owner && instance.exists())
      {
        log_leak(LEVEL_WARNING,"Leaking physical instance " IDFMT " in memory"
                               IDFMT "",
                               instance.id, memory.id);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_activate(void)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_new_remote(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
    }

    /////////////////////////////////////////////////////////////
    // InstanceManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(RegionTreeForest *ctx, DistributedID did,
                                     AddressSpaceID owner_space, 
                                     AddressSpaceID local_space,
                                     Memory mem, PhysicalInstance inst,
                                     RegionNode *node, 
                                     LayoutDescription *desc, Event u_event, 
                                     unsigned dep, bool persist)
      : PhysicalManager(ctx, did, owner_space, local_space, mem, inst), 
        region_node(node), layout(desc), use_event(u_event), 
        depth(dep), recycled(false), persistent(persist)
    //--------------------------------------------------------------------------
    {
      // Tell the runtime so it can update the per memory data structures
      context->runtime->allocate_physical_instance(this);
    }

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(const InstanceManager &rhs)
      : PhysicalManager(NULL, 0, 0, 0, Memory::NO_MEMORY,
                        PhysicalInstance::NO_INST), 
        region_node(NULL), layout(NULL), use_event(Event::NO_EVENT), depth(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InstanceManager::~InstanceManager(void)
    //--------------------------------------------------------------------------
    {
      // Tell the runtime this instance no longer exists
      // If we were the owner we already did this when we
      // garbage collected the physical instance
      if (!owner)
        context->runtime->free_physical_instance(this);
    }

    //--------------------------------------------------------------------------
    InstanceManager& InstanceManager::operator=(const InstanceManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      InstanceManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      InstanceManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
      assert(layout != NULL);
#endif
      const Domain::CopySrcDstField &info = layout->find_field_info(fid);
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> temp = 
        instance.get_accessor();
      return temp.get_untyped_field_accessor(info.offset, info.size);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::is_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    InstanceManager* InstanceManager::as_instance_manager(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<InstanceManager*>(this);
    }

    //--------------------------------------------------------------------------
    ReductionManager* InstanceManager::as_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    size_t InstanceManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(layout != NULL);
#endif
      return layout->get_layout_size();
    }

    //--------------------------------------------------------------------------
    void InstanceManager::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, GARBAGE_COLLECT_CALL);
#endif
      bool release_valid_views = true;
      if (owner)
      {
        // Always call this up front to see if we need to reclaim the
        // physical instance from the runtime because we recycled it.
        // Note we can do this here and not worry about a race with
        // notify_invalid because we are guaranteed they are called
        // sequentially by the state machine in the distributed
        // collectable implementation.
        bool reclaimed = context->runtime->reclaim_physical_instance(this);
        // Now tell the runtime that this instance will no longer exist
        context->runtime->free_physical_instance(this);
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
#endif
        // See if we should actually delete this instance
        // or whether we are trying to recycle it.  If we're
        // trying to recycle it, see if someone else has claimed it.
        // If not then take it back and delete it now to reclaim
        // the memory.
        if (!recycled || reclaimed)
        {
          // If either of these conditions were true, then we
          // should actually delete the physical instance.
          log_garbage(LEVEL_DEBUG,"Garbage collecting physical instance " IDFMT
                                " in memory " IDFMT " in address space %d",
                                instance.id, memory.id, owner_space);
#ifdef LEGION_PROF
          LegionProf::register_instance_deletion(instance.id);
#endif
#ifndef DISABLE_GC
          instance.destroy(use_event);
#endif
        }
        else // Otherwise it has been recycled, so don't release valid views
          release_valid_views = false;
        // Mark that this instance has been garbage collected
        instance = PhysicalInstance::NO_INST;
      }
      if (release_valid_views)
      {
        for (std::set<MaterializedView*>::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          if ((*it)->remove_resource_reference())
            legion_delete(*it);
        }
        valid_views.clear();
      }
    }

    //--------------------------------------------------------------------------
    void InstanceManager::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
      assert(!recycled);
#endif
    }

    //--------------------------------------------------------------------------
    void InstanceManager::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, NOTIFY_INVALID_CALL);
#endif
      // If we're the owner and we're now invalid and have no remote
      // references then we can tell the runtime that it is safe to 
      // recycle this physical instance.  Pass on the information to the
      // runtime and save the event that we should trigger to mark that
      // we're done using this physical instance.
      if (owner)
      {
        // Tell the runtime this instance is available for recycling
        // Note that doing this relies on the guarantee that notify_invalid
        // will always be called before garbage_collect, ensuring that
        // the instance is still valid.  This property is guaranteed by
        // the state machine in the distributed collectable implementation.
        context->runtime->recycle_physical_instance(this);
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
        assert(!recycled);
#endif
        // Mark that we are recycling this instance
        recycled = true;
      }
    }

    //--------------------------------------------------------------------------
    Event InstanceManager::get_recycle_event(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, GET_RECYCLE_EVENT_CALL);
#endif
      // Now do our operation, as we finish with each view we
      // can remove the resource reference that we are holding on it.
      // We accumulate the set of events representing all the users
      // still actively using the region from the given level. The
      // region can be recycled once they are all done using it.
      // Note we can do this without holding the lock because if we
      // get here, then we are ready to be reclaimed and no more
      // valid views will be added because we've already been 
      // garbage collected.
      std::set<Event> recycle_events;
      recycle_events.insert(use_event);
      for (std::set<MaterializedView*>::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        (*it)->accumulate_events(recycle_events);
        if ((*it)->remove_resource_reference())
          legion_delete(*it);
      }
      valid_views.clear();
      // Compute the recycle event
      Event recycle_event = Event::merge_events(recycle_events);
      return recycle_event;
    }

    //--------------------------------------------------------------------------
    MaterializedView* InstanceManager::create_top_view(unsigned depth)
    //--------------------------------------------------------------------------
    {
      DistributedID view_did = context->runtime->get_available_distributed_id();
      MaterializedView *result = legion_new<MaterializedView>(context, view_did,
                                                context->runtime->address_space,
                                                view_did, region_node, this,
                                            ((MaterializedView*)NULL/*parent*/),
                                                depth);
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(const FieldMask &copy_mask,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_mask, instance, fields);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(
                                  const std::vector<FieldID> &copy_fields,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_fields, instance, fields);
    }

    //--------------------------------------------------------------------------
    DistributedID InstanceManager::send_manager(AddressSpaceID target,
                                  std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      if (needed_managers.find(this) == needed_managers.end())
      {
        // Add ourselves to the needed managers 
        needed_managers.insert(this);
        // Now see if we need to send ourselves
        bool need_send;
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::set<AddressSpaceID>::const_iterator finder = 
            remote_spaces.find(target);
          need_send = (finder == remote_spaces.end());
        }
        // We'll handle the send of the remote reference
        // in send_remote_references
        if (!need_send)
          return did;
        // If we make it here then we need to be sent
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          pack_manager(rez, target);
        }
        // Now send the message
        context->runtime->send_instance_manager(target, rez);
        // Finally we can update our known nodes
        // It's only safe to do this after the message
        // has been sent
        layout->update_known_nodes(target);
      }
      // Otherwise there is nothing to do since we
      // have already been sent
      return did;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceManager::handle_send_manager(
          RegionTreeForest *context, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      // check to see if the forest already has the manager
      if (context->has_manager(did))
      {
        PhysicalManager *manager = context->find_manager(did);
        manager->update_remote_spaces(source);
#ifdef DEBUG_HIGH_LEVEL
        // If we're in debug mode do the unpack anyway to 
        // keep the deserializer happy
        InstanceManager::unpack_manager(derez, context, did, 
                                        source, false/*make*/);
#endif
      }
      else
      {
        InstanceManager *result = InstanceManager::unpack_manager(derez,
                                                          context, did, source);
        result->update_remote_spaces(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceManager::pack_manager(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(owner_space);
      rez.serialize(memory);
      rez.serialize(instance);
      rez.serialize(region_node->handle);
      rez.serialize(use_event);
      rez.serialize(depth);
      rez.serialize(persistent);
      layout->pack_layout_description(rez, target);
    }

    //--------------------------------------------------------------------------
    /*static*/ InstanceManager* InstanceManager::unpack_manager(
        Deserializer &derez, RegionTreeForest *context, 
        DistributedID did, AddressSpaceID source, bool make /*=true*/)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      Memory mem;
      derez.deserialize(mem);
      PhysicalInstance inst;
      derez.deserialize(inst);
      LogicalRegion handle;
      derez.deserialize(handle);
      Event use_event;
      derez.deserialize(use_event);
      unsigned depth;
      derez.deserialize(depth);
      bool persistent;
      derez.deserialize(persistent);
      RegionNode *node = context->get_node(handle);
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(derez, 
                                                            source, node);
#ifdef DEBUG_HIGH_LEVEL
      assert(layout != NULL);
#endif
      if (make)
        return legion_new<InstanceManager>(context, did, owner_space,
                                   context->runtime->address_space,
                                   mem, inst, node, layout,
                                   use_event, depth, persistent);
      else
        return NULL;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::add_valid_view(MaterializedView *view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view->depth == depth);
#endif
      // Add a resource reference so it can't be deleted
      view->add_resource_reference();
      bool remove_extra_reference = false;
      {
        AutoLock gc(gc_lock);
        if (valid_views.find(view) == valid_views.end())
          valid_views.insert(view);
        else
          remove_extra_reference = true;
      }
      if (remove_extra_reference && view->remove_resource_reference())
        legion_delete(view);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::remove_valid_view(MaterializedView *view)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(view->depth == depth);
#endif
      // Do nothing for right now
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::match_instance(size_t field_size, 
                                         const Domain &dom) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(layout != NULL);
#endif
      // First check to see if the domains are the same
      if (region_node->get_domain() != dom)
        return false;
      return layout->match_shape(field_size);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::match_instance(const std::vector<size_t> &field_sizes,
                                         const Domain &dom, 
                                         const size_t bf) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(layout != NULL);
#endif
      // First check to see if the domains are the same
      if (region_node->get_domain() != dom)
        return false;
      return layout->match_shape(field_sizes, bf);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::is_persistent(void) const
    //--------------------------------------------------------------------------
    {
      // No need to hold any locks as this is a monotonic variable
      return persistent;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::make_persistent(AddressSpaceID origin)
    //--------------------------------------------------------------------------
    {
      // Only need to do this if we weren't persistent to begin with
      if (!persistent)
      {
        persistent = true;
        // Now notify others about the update
        AutoLock gc(gc_lock,1,false/*exclusive*/);
        if (owner)
        {
          Serializer rez;
          rez.serialize(did);
          // If it is the owner send the update to all the remote address spaces
          for (std::set<AddressSpaceID>::const_iterator it = 
                remote_spaces.begin(); it != remote_spaces.end(); it++)
          {
            // Don't need to send it back to the origin
            if ((*it) != origin)
            {
              context->runtime->send_make_persistent((*it), rez);
            }
          }
        }
        else if (origin != owner_space)
        {
          // Not the owner, so send the update to the owner
          Serializer rez;
          rez.serialize(did);
          context->runtime->send_make_persistent(owner_space, rez);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceManager::handle_make_persistent(Deserializer &derez,
                               RegionTreeForest *context, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(context->has_manager(did));
#endif
      InstanceManager *manager = 
        context->find_manager(did)->as_instance_manager();
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      manager->make_persistent(source);
    }

    /////////////////////////////////////////////////////////////
    // ReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionManager::ReductionManager(RegionTreeForest *ctx, DistributedID did,
                                       AddressSpaceID owner_space, 
                                       AddressSpaceID local_space,
                                       Memory mem, PhysicalInstance inst, 
                                       RegionNode *node, ReductionOpID red, 
                                       const ReductionOp *o)
      : PhysicalManager(ctx, did, owner_space, local_space, mem, inst), 
        op(o), redop(red), region_node(node)
    //--------------------------------------------------------------------------
    { 
    }

    //--------------------------------------------------------------------------
    ReductionManager::~ReductionManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool ReductionManager::is_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    InstanceManager* ReductionManager::as_instance_manager(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    ReductionManager* ReductionManager::as_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<ReductionManager*>(this);
    }

    //--------------------------------------------------------------------------
    void ReductionManager::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
      if (owner)
      {
        context->runtime->free_physical_instance(this);
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(instance.exists());
#endif
        log_garbage(LEVEL_DEBUG,"Garbage collecting reduction instance " IDFMT
                                " in memory " IDFMT " in address space %d",
                                instance.id, memory.id, owner_space);
#ifdef LEGION_PROF
        LegionProf::register_instance_deletion(instance.id);
#endif
#ifndef DISABLE_GC
        instance.destroy();
#endif
        instance = PhysicalInstance::NO_INST;
      }
    }

    //--------------------------------------------------------------------------
    void ReductionManager::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // For right now we'll do nothing
      // There doesn't seem like much point in recycling reduction instances
    }

    //--------------------------------------------------------------------------
    DistributedID ReductionManager::send_manager(AddressSpaceID target,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      if (needed_managers.find(this) == needed_managers.end())
      {
        // Add ourselves to the needed managers
        needed_managers.insert(this);
        // Now see if we need to send ourselves
        bool need_send;
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::set<AddressSpaceID>::const_iterator finder = 
            remote_spaces.find(target);
          need_send = (finder == remote_spaces.end());
        }
        // We'll handle the send of the remote reference
        // in send_remote_references
        if (!need_send)
          return did;
        // If we make it here then we need to be sent
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          pack_manager(rez);
        }
        // Now send the message
        context->runtime->send_reduction_manager(target, rez);
      }
      return did;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionManager::handle_send_manager(
          RegionTreeForest *context, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);

      // check to see if the forest already has the manager
      if (context->has_manager(did))
      {
        PhysicalManager *manager = context->find_manager(did);
        manager->update_remote_spaces(source);
#ifdef DEBUG_HIGH_LEVEL
        // If we're in debug mode do the unpack anyway to 
        // keep the deserializer happy
        ReductionManager::unpack_manager(derez, context, did, false/*make*/);
#endif
      }
      else
      {
        ReductionManager *result = ReductionManager::unpack_manager(derez,
                                                            context, did);
        result->update_remote_spaces(source);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionManager::pack_manager(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(owner_space);
      rez.serialize(memory);
      rez.serialize(instance);
      rez.serialize(redop);
      rez.serialize(region_node->handle);
      rez.serialize(is_foldable());
      rez.serialize(get_pointer_space());
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionManager* ReductionManager::unpack_manager(
          Deserializer &derez, RegionTreeForest *context, 
          DistributedID did, bool make /*= true*/)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      Memory mem;
      derez.deserialize(mem);
      PhysicalInstance inst;
      derez.deserialize(inst);
      ReductionOpID redop;
      derez.deserialize(redop);
      LogicalRegion handle;
      derez.deserialize(handle);
      bool foldable;
      derez.deserialize(foldable);
      Domain ptr_space;
      derez.deserialize(ptr_space);

      RegionNode *node = context->get_node(handle);
      const ReductionOp *op = Runtime::get_reduction_op(redop);
      if (foldable)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!ptr_space.exists());
#endif
        if (make)
          return legion_new<FoldReductionManager>(context, did, owner_space,
                                          context->runtime->address_space,
                                          mem, inst, node, redop, op);
        else
          return NULL;
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(ptr_space.exists());
#endif
        if (make)
          return legion_new<ListReductionManager>(context, did, owner_space, 
                                        context->runtime->address_space,
                                        mem, inst, node, redop, op, ptr_space);
        else
          return NULL;
      }
    }

    //--------------------------------------------------------------------------
    ReductionView* ReductionManager::create_view(void)
    //--------------------------------------------------------------------------
    {
      DistributedID view_did = context->runtime->get_available_distributed_id();
      ReductionView *result = legion_new<ReductionView>(context, view_did,
                                                context->runtime->address_space,
                                                view_did, region_node,
                                                this);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // ListReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ListReductionManager::ListReductionManager(RegionTreeForest *ctx, 
                                               DistributedID did,
                                               AddressSpaceID owner_space, 
                                               AddressSpaceID local_space,
                                               Memory mem, 
                                               PhysicalInstance inst, 
                                               RegionNode *node,
                                               ReductionOpID red,
                                               const ReductionOp *o, Domain dom)
      : ReductionManager(ctx, did, owner_space, local_space, mem, 
                         inst, node, red, o), ptr_space(dom)
    //--------------------------------------------------------------------------
    {
      // Tell the runtime so it can update the per memory data structures
      context->runtime->allocate_physical_instance(this);
    }

    //--------------------------------------------------------------------------
    ListReductionManager::ListReductionManager(const ListReductionManager &rhs)
      : ReductionManager(NULL, 0, 0, 0, Memory::NO_MEMORY,
                         PhysicalInstance::NO_INST, NULL, 0, NULL),
        ptr_space(Domain::NO_DOMAIN)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ListReductionManager::~ListReductionManager(void)
    //--------------------------------------------------------------------------
    {
      // Free up our pointer space
      ptr_space.get_index_space().destroy();
      // Tell the runtime that this instance no longer exists
      // If we were the owner we already did this when we garbage
      // collected the physical instance
      if (!owner)
        context->runtime->free_physical_instance(this);
    }

    //--------------------------------------------------------------------------
    ListReductionManager& ListReductionManager::operator=(
                                                const ListReductionManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      ListReductionManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      // TODO: Implement this 
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      ListReductionManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    size_t ListReductionManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = op->sizeof_rhs;
      if (ptr_space.get_dim() == 0)
      {
        const LowLevel::ElementMask &mask = 
          ptr_space.get_index_space().get_valid_mask();
        result *= mask.get_num_elmts();
      }
      else
        result *= ptr_space.get_volume();
      return result;
    }
    
    //--------------------------------------------------------------------------
    bool ListReductionManager::is_foldable(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void ListReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      // Assume that it's all the fields for right now
      // but offset by the pointer size
      fields.push_back(
          Domain::CopySrcDstField(instance, sizeof(ptr_t), op->sizeof_rhs));
    }

    //--------------------------------------------------------------------------
    Event ListReductionManager::issue_reduction(
        const std::vector<Domain::CopySrcDstField> &src_fields,
        const std::vector<Domain::CopySrcDstField> &dst_fields,
        Domain space, Event precondition, bool reduction_fold, bool precise)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      if (precise)
      {
        Domain::CopySrcDstField idx_field(instance, 0/*offset*/, sizeof(ptr_t));
        return space.copy_indirect(idx_field, src_fields, dst_fields, 
                                   precondition, redop, reduction_fold);
      }
      else
      {
        // TODO: teach the low-level runtime how to issue
        // partial reduction copies from a given space
        assert(false);
        return Event::NO_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    Domain ListReductionManager::get_pointer_space(void) const
    //--------------------------------------------------------------------------
    {
      return ptr_space;
    }

    //--------------------------------------------------------------------------
    bool ListReductionManager::is_list_manager(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    ListReductionManager* ListReductionManager::as_list_manager(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<ListReductionManager*>(this);
    }

    //--------------------------------------------------------------------------
    FoldReductionManager* ListReductionManager::as_fold_manager(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // FoldReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FoldReductionManager::FoldReductionManager(RegionTreeForest *ctx, 
                                               DistributedID did,
                                               AddressSpaceID owner_space, 
                                               AddressSpaceID local_space,
                                               Memory mem,
                                               PhysicalInstance inst, 
                                               RegionNode *node,
                                               ReductionOpID red,
                                               const ReductionOp *o)
      : ReductionManager(ctx, did, owner_space, local_space, mem, 
                         inst, node, red, o)
    //--------------------------------------------------------------------------
    {
      // Tell the runtime so it can update the per memory data structures
      context->runtime->allocate_physical_instance(this);
    }

    //--------------------------------------------------------------------------
    FoldReductionManager::FoldReductionManager(const FoldReductionManager &rhs)
      : ReductionManager(NULL, 0, 0, 0, Memory::NO_MEMORY,
                         PhysicalInstance::NO_INST, NULL, 0, NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FoldReductionManager::~FoldReductionManager(void)
    //--------------------------------------------------------------------------
    {
      // Tell the runtime that this instance no longer exists
      // If we were the owner we already did this when we garbage
      // collected the physical instance
      if (!owner)
        context->runtime->free_physical_instance(this);
    }

    //--------------------------------------------------------------------------
    FoldReductionManager& FoldReductionManager::operator=(
                                                const FoldReductionManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      FoldReductionManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      FoldReductionManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    size_t FoldReductionManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = op->sizeof_rhs;
      const Domain &d = region_node->row_source->domain;
      if (d.get_dim() == 0)
      {
        const LowLevel::ElementMask &mask = 
          d.get_index_space().get_valid_mask();
        result *= mask.get_num_elmts();
      }
      else
        result *= d.get_volume();
      return result;
    }
    
    //--------------------------------------------------------------------------
    bool FoldReductionManager::is_foldable(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    void FoldReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      // Assume that its all the fields for now
      // until we find a different way to do reductions on a subset of fields
      fields.push_back(
          Domain::CopySrcDstField(instance, 0/*offset*/, op->sizeof_rhs));
    }

    //--------------------------------------------------------------------------
    Event FoldReductionManager::issue_reduction(
        const std::vector<Domain::CopySrcDstField> &src_fields,
        const std::vector<Domain::CopySrcDstField> &dst_fields,
        Domain space, Event precondition, bool reduction_fold, bool precise)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(instance.exists());
#endif
      // Doesn't matter if this one is precise or not
      return space.copy(src_fields, dst_fields, precondition, 
                        redop, reduction_fold);
    }

    //--------------------------------------------------------------------------
    Domain FoldReductionManager::get_pointer_space(void) const
    //--------------------------------------------------------------------------
    {
      return Domain::NO_DOMAIN;
    }

    //--------------------------------------------------------------------------
    bool FoldReductionManager::is_list_manager(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    ListReductionManager* FoldReductionManager::as_list_manager(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    FoldReductionManager* FoldReductionManager::as_fold_manager(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<FoldReductionManager*>(this);
    }

    /////////////////////////////////////////////////////////////
    // LogicalView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalView::LogicalView(RegionTreeForest *ctx, DistributedID did,
                             AddressSpaceID own_addr, DistributedID own_did,
                             RegionTreeNode *node)
      : HierarchicalCollectable(ctx->runtime, did, own_addr, own_did), 
        context(ctx), logical_node(node), 
        view_lock(Reservation::create_reservation()) 
    //--------------------------------------------------------------------------
    {
      context->register_logical_view(did, this);
    }

    //--------------------------------------------------------------------------
    LogicalView::~LogicalView(void)
    //--------------------------------------------------------------------------
    {
      view_lock.destroy_reservation();
      view_lock = Reservation::NO_RESERVATION;
      context->unregister_logical_view(this->did);
    }

    //--------------------------------------------------------------------------
    void LogicalView::defer_collect_user(Event term_event) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, DEFER_COLLECT_USER_CALL);
#endif
      // The runtime will add the gc reference to this view when necessary
      runtime->defer_collect_user(this, term_event);
    }
 
    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::handle_deferred_collect(LogicalView *view,
                                             const std::set<Event> &term_events)
    //--------------------------------------------------------------------------
    {
 #ifdef LEGION_PROF
      LegionProf::register_event(0, PROF_BEGIN_GC);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      0 /* no unique id */,
                                      BEGIN_GC);
#endif     
      view->collect_users(term_events);
      // Then remove the gc reference on the object
      if (view->remove_gc_reference())
      {
        if (view->is_reduction_view())
          legion_delete(view->as_reduction_view());
        else
        {
          InstanceView *inst_view = view->as_instance_view();
          if (inst_view->is_composite_view())
            legion_delete(inst_view->as_composite_view());
          else
            legion_delete(inst_view->as_materialized_view());
        }
      }
#ifdef LEGION_PROF
      LegionProf::register_event(0, PROF_END_GC);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      0 /* no unique id */,
                                      END_GC);
#endif
    }

    //--------------------------------------------------------------------------
    void LogicalView::send_back_user(const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(owner_did);
        rez.serialize(user.usage);
        rez.serialize(user.field_mask);
        rez.serialize(user.term_event);
        rez.serialize(user.child);
      }
      context->runtime->send_back_user(owner_addr, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::handle_send_back_user(RegionTreeForest *ctx,
                                                       Deserializer &derez,
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      PhysicalUser user;
      derez.deserialize(user.usage);
      derez.deserialize(user.field_mask);
      derez.deserialize(user.term_event);
      derez.deserialize(user.child);

      LogicalView *target_view = ctx->find_view(did);
      target_view->process_send_back_user(source, user);
    }

    /////////////////////////////////////////////////////////////
    // InstanceView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_proc, DistributedID own_did,
                               RegionTreeNode *node)
      : LogicalView(ctx, did, owner_proc, own_did, node)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceView::~InstanceView(void)
    //--------------------------------------------------------------------------
    {
      // Release any aliased distributed IDs we have 
      for (std::set<DistributedID>::const_iterator it = aliases.begin();
            it != aliases.end(); it++)
      {
        context->unregister_logical_view(*it);
        context->runtime->unregister_hierarchical_collectable(*it);
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceView::is_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    InstanceView* InstanceView::as_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<InstanceView*>(this);
    }

    //--------------------------------------------------------------------------
    ReductionView* InstanceView::as_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    void InstanceView::add_alias_did(DistributedID alias)
    //--------------------------------------------------------------------------
    {
      context->runtime->register_hierarchical_collectable(alias, this); 
      context->register_logical_view(alias, this);
      AutoLock v_lock(view_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(aliases.find(alias) == aliases.end());
#endif
      aliases.insert(alias);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_send_subscriber(
                                RegionTreeForest *context, Deserializer &derez,
                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID parent_did;
      derez.deserialize(parent_did);
      Color child_color;
      derez.deserialize(child_color);
      DistributedID did;
      derez.deserialize(did);
      DistributedID remote_did;
      derez.deserialize(remote_did);
      // First get the subview on this node
      InstanceView *parent_view = 
        context->find_view(parent_did)->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_view != NULL);
#endif
      InstanceView *child_view = parent_view->get_subview(child_color);
#ifdef DEBUG_HIGH_LEVEL
      assert(child_view != NULL);
#endif
      // Add the alias first for anyone who is coming later
      child_view->add_alias_did(did);
      // Then add the remote subscriber
      child_view->add_subscriber(source, remote_did);
      // Add a remote reference held by the view that sent this back
      child_view->add_remote_reference();
    }

    /////////////////////////////////////////////////////////////
    // MaterializedView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(
                               RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID own_addr, DistributedID own_did,
                               RegionTreeNode *node, InstanceManager *man,
                               MaterializedView *par, unsigned dep)
      : InstanceView(ctx, did, own_addr, own_did, node), 
        manager(man), parent(par), depth(dep)
#ifdef PHYSICAL_FIELD_TREE
        , curr_epoch_users(
            new FieldTree<PhysicalUser>(FieldMask(FIELD_ALL_ONES)))
        , prev_epoch_users(
            new FieldTree<PhysicalUser>(FieldMask(FIELD_ALL_ONES)))
#endif
    //--------------------------------------------------------------------------
    {
      // If we're the top of the tree and the owner make the instance lock
      if ((parent == NULL) && (owner_did == did))
        inst_lock = Reservation::create_reservation();
      else if (parent != NULL)
        inst_lock = parent->inst_lock;
      // Otherwise the instance lock will get filled in when we are unpacked
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      // Add a resource reference to the manager
      manager->add_resource_reference();
      // Initialize the current versions to zero
      current_versions[0] = FieldMask(FIELD_ALL_ONES);
    }

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(const MaterializedView &rhs)
      : InstanceView(NULL, 0, 0, 0, NULL),
        manager(NULL), parent(NULL), depth(0)
#ifdef PHYSICAL_FIELD_TREE
        , curr_epoch_users(NULL), prev_epoch_users(NULL)
#endif
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MaterializedView::~MaterializedView(void)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL && (owner_did == did))
        inst_lock.destroy_reservation();
      inst_lock = Reservation::NO_RESERVATION; 
      // Remove our references to the manager
      if (manager->remove_resource_reference())
        legion_delete(manager);
      // Remove our resource references on our children
      for (std::map<Color,MaterializedView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->second->remove_resource_reference())
          legion_delete(it->second);
      }
#ifdef PHYSICAL_FIELD_TREE
      delete curr_epoch_users;
      delete prev_epoch_users;
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView& MaterializedView::operator=(const MaterializedView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Memory MaterializedView::get_location(void) const
    //--------------------------------------------------------------------------
    {
      return manager->memory;
    }

    //--------------------------------------------------------------------------
    size_t MaterializedView::get_blocking_factor(void) const
    //--------------------------------------------------------------------------
    {
      return manager->layout->blocking_factor;
    } 

    //--------------------------------------------------------------------------
    const FieldMask& MaterializedView::get_physical_mask(void) const
    //--------------------------------------------------------------------------
    {
      return manager->layout->allocated_fields;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::is_composite_view(void) const
    //--------------------------------------------------------------------------
    {
      return false; 
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::as_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<MaterializedView*>(this);
    }

    //--------------------------------------------------------------------------
    CompositeView* MaterializedView::as_composite_view(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_parent_view(void) const
    //--------------------------------------------------------------------------
    {
      return (parent != NULL); 
    }

    //--------------------------------------------------------------------------
    InstanceView* MaterializedView::get_parent_view(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    InstanceView* MaterializedView::get_subview(Color c)
    //--------------------------------------------------------------------------
    {
      return get_materialized_subview(c);
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::get_materialized_subview(Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, GET_SUBVIEW_CALL);
#endif
      // This is the common case
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        std::map<Color,MaterializedView*>::const_iterator finder = 
                                                            children.find(c);
        if (finder != children.end())
          return finder->second;
      }
      DistributedID child_did = 
        context->runtime->get_available_distributed_id();
      DistributedID child_own_did = child_did;
      // See if the parent is remote, if it is then we need to also
      // be remote so we can send back information to the corresponding view
      if (did != owner_did)
        child_own_did = context->runtime->get_available_distributed_id();
      RegionTreeNode *child_node = logical_node->get_tree_child(c);
      MaterializedView *child_view = legion_new<MaterializedView>(context, 
                                child_did, owner_addr, child_own_did, 
                                child_node, manager, this, depth);
      child_view->add_resource_reference();
      // Retake the lock and try and add it, see if 
      // someone else added the child in the meantime
      {
        AutoLock v_lock(view_lock);
        std::map<Color,MaterializedView*>::const_iterator finder = 
                                                            children.find(c);
        if (finder != children.end())
        {
          // Guaranteed to succeed
          if (child_view->remove_resource_reference())
            legion_delete(child_view);
          if (child_own_did != child_did)
            context->runtime->free_distributed_id(child_own_did);
          return finder->second;
        }
        children[c] = child_view;
      }
      // If we are remote add a remote subscriber to the view on the owner node
      if (child_did != child_own_did)
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(owner_did);
          rez.serialize(c);
          rez.serialize(child_own_did);
          rez.serialize(child_did);
        }
        // Add a held remote reference
        child_view->add_held_remote_reference();
        // Now notify the owner view it has a remote subscriber
        context->runtime->send_subscriber(owner_addr, rez);
      }
      return child_view;
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::get_materialized_parent_view(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::add_subview(MaterializedView *view, Color c)
    //--------------------------------------------------------------------------
    {
      bool added = true;
      {
        AutoLock v_lock(view_lock);
        if (children.find(c) == children.end())
          children[c] = view;
        else
          added = false;
      }
      if (added)
        view->add_resource_reference();
      return added;
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::copy_field(FieldID fid,
                              std::vector<Domain::CopySrcDstField> &copy_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, COPY_FIELD_CALL);
#endif
      std::vector<FieldID> local_fields(1,fid);
      manager->compute_copy_offsets(local_fields, copy_fields); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_to(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, COPY_TO_CALL);
#endif
      manager->compute_copy_offsets(copy_mask, dst_fields);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_from(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, COPY_FROM_CALL);
#endif
      manager->compute_copy_offsets(copy_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::reduce_to(ReductionOpID redop, 
                                     const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REDUCE_TO_CALL);
#endif
      manager->compute_copy_offsets(copy_mask, dst_fields);
      return false; // not a fold
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_war_dependence(const RegionUsage &usage,
                                              const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, HAS_WAR_DEPENDENCE_CALL);
#endif
      // No WAR dependences for read-only or reduce 
      if (IS_READ_ONLY(usage) || IS_REDUCE(usage))
        return false;
      if ((parent != NULL) &&
          parent->has_war_dependence_above(usage, user_mask, 
                                           logical_node->get_color()))
        return true;
      // Do the local analysis
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
#ifndef PHYSICAL_FIELD_TREE
      for (std::list<PhysicalUser>::const_iterator it = 
            curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
      {
        if (user_mask * it->field_mask)
          continue;
        if (IS_READ_ONLY(it->usage))
          return true;
      }
      return false;
#else
      WARAnalyzer<false> analyzer;
      curr_epoch_users->analyze(user_mask, analyzer);
      return analyzer.has_war_dependence();
#endif
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::accumulate_events(std::set<Event> &all_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ACCUMULATE_EVENTS_CALL);
#endif
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      all_events.insert(event_references.begin(),event_references.end());
    } 

    //--------------------------------------------------------------------------
    PhysicalManager* MaterializedView::get_manager(void) const
    //--------------------------------------------------------------------------
    {
      return manager;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user(ReductionOpID redop, Event copy_term,
                                     const FieldMask &copy_mask, bool reading,
                                     Processor exec_proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_COPY_USER_CALL);
#endif
      // Quick test, we only need to do this if the copy_term event
      // exists, otherwise the user is already done
      if (copy_term.exists())
      {
        RegionUsage usage;
        usage.redop = redop;
        usage.prop = EXCLUSIVE;
        if (reading)
          usage.privilege = READ_ONLY;
        else if (redop > 0)
          usage.privilege = REDUCE;
        else
          usage.privilege = READ_WRITE;
        PhysicalUser user(usage, copy_mask, copy_term);
        if (parent != NULL)
        {
          // Save our color
          user.child = logical_node->get_color();
          parent->add_copy_user_above(user);
          // Restore the color
          user.child = -1;
        }
        add_local_copy_user(user);
        // Note we can ignore the wait on set here since we are just
        // registering the user and filtering previous users
        // Launch the garbage collection task
        defer_collect_user(user.term_event); 
      }
    }

    //--------------------------------------------------------------------------
    InstanceRef MaterializedView::add_user(PhysicalUser &user, 
                                           Processor exec_proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_USER_CALL);
#endif
      std::set<Event> wait_on_events;
      Event start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        wait_on_events.insert(start_use_event);
      if (parent != NULL)
      {
        // Set our color
        user.child = logical_node->get_color();
        parent->add_user_above(wait_on_events, user);
        // Restore the bottom color
        user.child = -1;
      }
      add_local_user<false>(wait_on_events, user);
      // Launch the garbage collection task, if it doesn't exist
      // then the user wasn't registered anyway, see add_local_user
      if (user.term_event.exists())
        defer_collect_user(user.term_event);
      // At this point tasks shouldn't be allowed to wait on themselves
#ifdef DEBUG_HIGH_LEVEL
      if (user.term_event.exists())
        assert(wait_on_events.find(user.term_event) == wait_on_events.end());
#endif
      // Make the instance ref
      Event ready_event = Event::merge_events(wait_on_events);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      if (!ready_event.exists())
      {
        UserEvent new_ready_event = UserEvent::create_user_event();
        new_ready_event.trigger();
        ready_event = new_ready_event;
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependences(
          Machine::get_executing_processor(), wait_on_events, ready_event);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(wait_on_events, ready_event);
#endif
      Reservation needed_lock = 
        IS_ATOMIC(user.usage) ? inst_lock : Reservation::NO_RESERVATION;
      return InstanceRef(ready_event, needed_lock, ViewHandle(this));
    }
 
    //--------------------------------------------------------------------------
    void MaterializedView::notify_activate(void)
    //--------------------------------------------------------------------------
    {
      // Add a gc reference to our manager
      manager->add_gc_reference();
      // Add a gc reference to our parent if we have one
      if (parent != NULL)
        parent->add_gc_reference();
    }

    //--------------------------------------------------------------------------
    void MaterializedView::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
      // No need to worry about handling the deletion case since
      // we know we also hold a resource reference and therefore
      // the manager won't be deleted until we are deleted at
      // the earliest
      manager->remove_gc_reference();
      // Also remove our parent gc reference if we have one
      if ((parent != NULL) && parent->remove_gc_reference())
        legion_delete(parent);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      if (depth == manager->depth)
      {
        manager->add_valid_reference();
        manager->add_valid_view(this);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      if (depth == manager->depth)
      {
        manager->remove_valid_view(this);
        if (manager->remove_valid_reference())
          legion_delete(manager);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::collect_users(const std::set<Event> &term_events)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock v_lock(view_lock);
        bool has_local_events = false;
        for (std::set<Event>::const_iterator it = term_events.begin();
              it != term_events.end(); it++)
        {
          std::set<Event>::iterator finder = event_references.find(*it);
          // If we find it, keep it in the set and prune it from this set
          // otherwise it has already been collected at this level so
          // we can remove it from the set
          if (finder != event_references.end())
          {
            event_references.erase(finder);
            has_local_events = true;
          }
        }
        // If we still have events then filter them
        if (has_local_events)
        {
          if (term_events.size() == 1)
            filter_local_users(*(term_events.begin()));
          else if (!term_events.empty())
            filter_local_users(term_events);
        }
      }
      if (parent != NULL)
        parent->collect_users(term_events);
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::process_send_back_user(AddressSpaceID source,
                                                  PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      // First we need to translate the field mask
      manager->region_node->column_source->transform_field_mask(
                                                    user.field_mask, source);
#ifdef DEBUG_HIGH_LEVEL
      assert(!subscribers.empty());
#endif
      std::set<Event> dummy_wait_on;
      // Only need to add this user locally, if there is a 
      // parent it will have sent the user back independently
      add_local_user<true>(dummy_wait_on, user);
      // The launch the garbage collection task
      defer_collect_user(user.term_event); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user_above(std::set<Event> &wait_on,
                                          PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_USER_ABOVE_CALL);
#endif
      if (parent != NULL)
      {
        // Save the child and replace with our child 
        int local_child = user.child;
        user.child = logical_node->get_color();
        parent->add_user_above(wait_on, user);
        // Restore the child
        user.child = local_child;
      }
      add_local_user<true>(wait_on, user);
      // No need to launch a collect user task, the child takes care of that
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user_above(PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
      {
        // Save the child and replace with our child 
        int local_child = user.child;
        user.child = logical_node->get_color();
        parent->add_copy_user_above(user);
        // Restore the child
        user.child = local_child;
      }
      add_local_copy_user(user);
      // No need to add a collect task, the child will do that for us
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_local_copy_user(PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      // Add this to the list of current epoch users,
      // this is safe to do because we previously filtered
      // the previous epoch usrs for these fields in the find
      // copy preconditions call
#ifndef PHYSICAL_FIELD_TREE
      curr_epoch_users.push_back(user);
#else
      curr_epoch_users->insert(user);
#endif
      event_references.insert(user.term_event);
      if (owner_did != did)
        send_back_user(user);
    }
 
    //--------------------------------------------------------------------------
    template<bool ABOVE>
    void MaterializedView::add_local_user(std::set<Event> &wait_on,
                                          const PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_LOCAL_USER_CALL);
#endif
      // Need the lock when doing this analysis
      AutoLock v_lock(view_lock);
#ifndef PHYSICAL_FIELD_TREE
      FieldMask non_dominated;
      FieldMask observed;
      std::deque<PhysicalUser> new_prev_users;
      for (std::list<PhysicalUser>::iterator it = curr_epoch_users.begin();
            it != curr_epoch_users.end(); /*nothing*/)
      {
        if (!ABOVE)
          observed |= it->field_mask;
        // If they are the same user then keep going
        if (user.term_event == it->term_event)
        {
          if (!ABOVE)
            non_dominated |= (user.field_mask & it->field_mask);
          it++;
          continue;
        }
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
        // We're about to do a bunch of expensive tests,
        // so first do something cheap to see if we can
        // prune the user out of the list.
        if (it->term_event.has_triggered())
        {
          it = curr_epoch_users.erase(it);
          continue;
        }
#endif
        if (ABOVE && (user.child >= 0))
        {
          // Same child, already done the analysis
          if (user.child == it->child)
          {
            it++;
            continue;
          }
          // Disjoint children, keep going
          if ((it->child >= 0) && 
              logical_node->are_children_disjoint(unsigned(user.child),
                                                  unsigned(it->child)))
          {
            it++;
            continue;
          }
        }
        FieldMask overlap = user.field_mask & it->field_mask;
        // Disjoint fields, keep going
        if (!overlap)
        {
          it++;
          continue;
        } 
        // Now we need to do a dependence analysis
        DependenceType dt = check_dependence_type(it->usage, user.usage);
        switch (dt)
        {
          case NO_DEPENDENCE:
          case ATOMIC_DEPENDENCE:
          case SIMULTANEOUS_DEPENDENCE:
            {
              // No actual dependence
              if (!ABOVE)
                non_dominated |= overlap;
              it++;
              continue;
            }
          case TRUE_DEPENDENCE:
          case ANTI_DEPENDENCE:
            {
              // Actual dependence
              wait_on.insert(it->term_event);
              // Move the user back to the previous epoch
              // if we're allowed to dominate it
              if (!ABOVE)
              {
                new_prev_users.push_back(*it);
                new_prev_users.back().field_mask = overlap;
                it->field_mask -= user.field_mask;
                if (!it->field_mask)
                  it = curr_epoch_users.erase(it);
                else
                  it++;
              }
              else
                it++;
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
      // It's only safe to dominate fields we observed 
      FieldMask dominated = (observed & (user.field_mask - non_dominated));
      // Update the non-dominated mask with what we
      // we're actually not-dominated by 
      if (!ABOVE)
        non_dominated = user.field_mask - dominated;
      // Filter any dominated users
      if (!ABOVE && !!dominated)
      {
        for (std::list<PhysicalUser>::iterator it = prev_epoch_users.begin();
              it != prev_epoch_users.end(); /*nothing*/)
        {
          it->field_mask -= dominated;
          if (!it->field_mask)
            it = prev_epoch_users.erase(it);
          else
            it++;
        }
        // If this is not read-only then update the versions of
        // the dominated fields.
        //if (!IS_READ_ONLY(user.usage))
        //  update_versions(dominated);
      }
      if (ABOVE || !!non_dominated)
      {
        for (std::list<PhysicalUser>::iterator it = 
              prev_epoch_users.begin(); it != 
              prev_epoch_users.end(); /*nothing*/)
        { 
          if (user.term_event == it->term_event)
          {
            it++;
            continue;
          }
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
          // Before we do a bunch of expensive operations,
          // do something cheap and check to see if the
          // event has already triggered.
          if (it->term_event.has_triggered())
          {
            it = prev_epoch_users.erase(it);
            continue;
          }
#endif
          if (ABOVE && (user.child >= 0))
          {
            if (user.child == it->child)
            {
              it++;
              continue;
            }
            if ((it->child >= 0) &&
                logical_node->are_children_disjoint(unsigned(user.child),
                                                    unsigned(it->child)))
            {
              it++;
              continue;
            }
          }
          if (!ABOVE && (it->field_mask * non_dominated))
          {
            it++;
            continue;
          }
          if (ABOVE && (it->field_mask * user.field_mask))
          {
            it++;
            continue;
          }
          // Now we need to do an actual dependence analysis
          DependenceType dt = check_dependence_type(it->usage, user.usage);
          switch (dt)
          {
            case NO_DEPENDENCE:
            case ATOMIC_DEPENDENCE:
            case SIMULTANEOUS_DEPENDENCE:
              {
                break;
              }
            case TRUE_DEPENDENCE:
            case ANTI_DEPENDENCE:
              {
                // Actual dependence
                wait_on.insert(it->term_event);
                break;
              }
            default:
              assert(false); // should never get here
          }
          // Update the user
          it++;
        }
      }
      // Add the new previous users to previous epoch list
      for (unsigned idx = 0; idx < new_prev_users.size(); idx++)
        prev_epoch_users.push_back(new_prev_users[idx]);
      // Add ourselves to the list of current users
      if (user.term_event.exists())
        curr_epoch_users.push_back(user);
      // See if we need to compress the lists
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
      if (Runtime::max_filter_size > 0)
      {
        if (prev_epoch_users.size() >= Runtime::max_filter_size)
          condense_user_list(prev_epoch_users, true/*previous*/);
        if (curr_epoch_users.size() >= Runtime::max_filter_size)
          condense_user_list(curr_epoch_users, false/*previous*/);
      }
#endif
#else
      // First do the analysis on the current epcoh users and filter
      // out any that can be sent back to the previous epoch users
      PhysicalDepAnalyzer<!ABOVE> curr_analyzer(user, user.field_mask,
                                                logical_node, wait_on);
      curr_epoch_users->analyze(user.field_mask, curr_analyzer);
      FieldMask non_dominated = curr_analyzer.get_non_dominated_mask();
      FieldMask dominated = ((user.field_mask - non_dominated) &
                              curr_analyzer.get_observed_mask());
      if (!ABOVE)
        non_dominated = user.field_mask - dominated;
      // Filter any dominated users from the previous users
      if (!ABOVE && !!dominated)
      {
        PhysicalFilter filter(dominated);
        prev_epoch_users->analyze<PhysicalFilter>(dominated, filter);
      }
      // If we didn't dominate all the field do another analysis
      // on the previous epoch users for the non-dominated fields
      if (ABOVE || !!non_dominated)
      {
        PhysicalDepAnalyzer<false> prev_analyzer(user, 
          (ABOVE ? user.field_mask : non_dominated), logical_node, wait_on);
        prev_epoch_users->analyze(non_dominated, prev_analyzer);
      }
      // Now we can add the filtered users into the previous users
      curr_analyzer.insert_filtered_users(prev_epoch_users);
      // Add our user to the list of current users
      curr_epoch_users->insert(user);
#endif
      // Only need to do this if termination event exists
      if (user.term_event.exists())
      {
        // Also update the event references
        event_references.insert(user.term_event);
        // Send back the user if necessary
        if (owner_did != did)
          send_back_user(user);
      }
    }
 
    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions(ReductionOpID redop, 
                                                   bool reading, 
                                                   const FieldMask &copy_mask,
                                       std::map<Event,FieldMask> &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_COPY_PRECONDITIONS_CALL);
#endif
      Event start_use_event = manager->get_use_event();
      if (start_use_event.exists())
      {
        std::map<Event,FieldMask>::iterator finder = 
          preconditions.find(start_use_event);
        if (finder == preconditions.end())
          preconditions[start_use_event] = copy_mask;
        else
          finder->second |= copy_mask;
      }
      if (parent != NULL)
        parent->find_copy_preconditions_above(logical_node->get_color(),
                                              redop, reading, copy_mask,
                                              preconditions);
      find_local_copy_preconditions<false>(-1/*child color*/, redop, reading,
                                           copy_mask, preconditions);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_above(Color child_color,
                                                         ReductionOpID redop,
                                                         bool reading,
                                                     const FieldMask &copy_mask,
                                       std::map<Event,FieldMask> &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_COPY_PRECONDITIONS_ABOVE_CALL);
#endif
      if (parent != NULL)
        parent->find_copy_preconditions_above(logical_node->get_color(),
                                              redop, reading, copy_mask,
                                              preconditions);
      find_local_copy_preconditions<true>(child_color, redop, reading,
                                          copy_mask, preconditions);
    }
    
    //--------------------------------------------------------------------------
    template<bool ABOVE>
    void MaterializedView::find_local_copy_preconditions(int local_color,
                                                         ReductionOpID redop,
                                                         bool reading,
                                                     const FieldMask &copy_mask,
                                       std::map<Event,FieldMask> &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_LOCAL_COPY_PRECONDITIONS_CALL);
#endif
      // Need the lock when doing this analysis
      AutoLock v_lock(view_lock);
#ifndef PHYSICAL_FIELD_TREE
      FieldMask non_dominated;
      FieldMask observed;
      std::deque<PhysicalUser> new_prev_users;
      for (std::list<PhysicalUser>::iterator it = curr_epoch_users.begin();
            it != curr_epoch_users.end(); /*nothing*/)
      {
        if (!ABOVE)
          observed |= it->field_mask;
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
        // We're about to do a bunch of expensive tests,
        // so first do something cheap to see if we can
        // prune the user out of the list.
        if (it->term_event.has_triggered())
        {
          it = curr_epoch_users.erase(it);
          continue;
        }
#endif
        if (ABOVE && (local_color >= 0))
        {
          // Same child, already done the analysis
          if (local_color == it->child)
          {
            it++;
            continue;
          }
          // Disjoint children, keep going
          if ((it->child >= 0) &&
              logical_node->are_children_disjoint(unsigned(local_color),
                                                  unsigned(it->child)))
          {
            it++;
            continue;
          }
        }
        FieldMask overlap = copy_mask & it->field_mask;
        // Disjoint fields, keep going
        if (!overlap)
        {
          it++;
          continue;
        }
        // Now we need to do a dependence analysis
        if (reading && IS_READ_ONLY(it->usage))
        {
          non_dominated |= overlap;
          it++;
          continue;
        }
        else if ((redop > 0) && IS_REDUCE(it->usage) && 
                 (it->usage.redop == redop))
        {
          non_dominated |= overlap;
          it++; 
          continue;
        }
        // If we get here then we have a dependence so record it
        std::map<Event,FieldMask>::iterator finder = 
          preconditions.find(it->term_event);
        if (finder == preconditions.end())
          preconditions[it->term_event] = overlap;
        else
          finder->second |= overlap;
        new_prev_users.push_back(*it);
        new_prev_users.back().field_mask = overlap;
        it->field_mask -= overlap;
        if (!it->field_mask)
          it = curr_epoch_users.erase(it);
        else
          it++;
      }
      // It's only safe to dominate fields that we observed
      FieldMask dominated = (observed & (copy_mask - non_dominated));
      // Update the non-dominated mask with what we
      // we're actually not-dominated by
      if (!ABOVE)
        non_dominated = copy_mask - dominated;
      // Filter any dominated users
      if (!ABOVE && !!dominated)
      {
        for (std::list<PhysicalUser>::iterator it = prev_epoch_users.begin();
              it != prev_epoch_users.end(); /*nothing*/)
        {
          it->field_mask -= dominated;
          if (!it->field_mask)
            it = prev_epoch_users.erase(it);
          else
            it++;
        }
        // If this is not read-only then update the versions of
        // the dominated fields.
        //if (!IS_READ_ONLY(user.usage))
        //  update_versions(dominated);
      }
      if (ABOVE || !!non_dominated)
      {
        for (std::list<PhysicalUser>::iterator it = 
              prev_epoch_users.begin(); it != 
              prev_epoch_users.end(); /*nothing*/)
        {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
          // Before we do a bunch of expensive operations,
          // do something cheap and check to see if the
          // event has already triggered.
          if (it->term_event.has_triggered())
          {
            it = prev_epoch_users.erase(it);
            continue;
          }
#endif
          if (ABOVE && (local_color >= 0))
          {
            if (local_color == it->child)
            {
              it++;
              continue;
            }
            if ((it->child >= 0) &&
                logical_node->are_children_disjoint(unsigned(local_color),
                                                    unsigned(it->child)))
            {
              it++;
              continue;
            }
          }
          FieldMask overlap = it->field_mask & 
                                      (ABOVE ? copy_mask : non_dominated);
          if (!overlap)
          {
            it++;
            continue;
          }
          if (reading && IS_READ_ONLY(it->usage))
          {
            it++;
            continue;
          }
          else if ((redop > 0) && IS_REDUCE(it->usage) &&
                   (it->usage.redop == redop))
          {
            it++;
            continue;
          }
          // Otherwise record the dependence
          std::map<Event,FieldMask>::iterator finder = 
            preconditions.find(it->term_event);
          if (finder == preconditions.end())
            preconditions[it->term_event] = overlap;
          else
            finder->second |= overlap;
          it++;
        }
      }
      // Add the new previous users to previous epoch list
      for (unsigned idx = 0; idx < new_prev_users.size(); idx++)
        prev_epoch_users.push_back(new_prev_users[idx]);
      // See if we need to compress the lists
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
      if (Runtime::max_filter_size > 0)
      {
        if (prev_epoch_users.size() >= Runtime::max_filter_size)
          condense_user_list(prev_epoch_users, true/*previous*/);
        if (curr_epoch_users.size() >= Runtime::max_filter_size)
          condense_user_list(curr_epoch_users, false/*previous*/);
      }
#endif
#else // PHYSICAL_FIELD_TREE
      assert(false); // TODO: Implement this for physical field tree
#endif
    }
    
#if 0
    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions(std::set<Event> &wait_on,
                                             bool writing, ReductionOpID redop,
                                             const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_COPY_PRECONDITIONS_CALL);
#endif
      if (parent != NULL)
        parent->find_copy_preconditions_above(logical_node->get_color(),
                                            wait_on, writing, redop, copy_mask);
      find_local_copy_preconditions<false>(wait_on, writing, redop,
                                           copy_mask, -1/*child color*/);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_above(Color child_color,
                                                     std::set<Event> &wait_on,
                                                     bool writing, 
                                                     ReductionOpID redop,
                                                     const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_COPY_PRECONDITIONS_ABOVE_CALL);
#endif
      if (parent != NULL)
        parent->find_copy_preconditions_above(logical_node->get_color(),
                                            wait_on, writing, redop, copy_mask);
      find_local_copy_preconditions<true>(wait_on, writing, redop,
                                          copy_mask, child_color);
    }

    //--------------------------------------------------------------------------
    template<bool ABOVE>
    void MaterializedView::find_local_copy_preconditions(
                                                     std::set<Event> &wait_on,
                                                     bool writing, 
                                                     ReductionOpID redop,
                                                     const FieldMask &copy_mask,
                                                     int local_color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_LOCAL_COPY_PRECONDITIONS_CALL);
#endif
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
#ifndef PHYSICAL_FIELD_TREE
      if (!writing)
      {
        // Only need to track dominated mask when not above
        FieldMask non_dominated;
        for (std::list<PhysicalUser>::const_iterator it = 
              curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
        {
          FieldMask overlap = it->field_mask & copy_mask;
          if (!overlap)
            continue;
          // Same child means we already did the analysis below
          if (ABOVE && (it->child == local_color))
            continue;
          if (IS_READ_ONLY(it->usage))
          {
            if (!ABOVE)
              non_dominated |= overlap;
            continue;
          }
          // Disjoint children don't have dependences
          if (ABOVE && (it->child >= 0) &&
              logical_node->are_children_disjoint(unsigned(local_color),
                                                  unsigned(it->child)))
            continue;
          // Otherwise register a dependence
          wait_on.insert(it->term_event);
        }
        // If we had fields we didn't dominate, need to analyze the
        // previous set of users as well.
        if (ABOVE || !!non_dominated)
        {
          for (std::list<PhysicalUser>::const_iterator it = 
                prev_epoch_users.begin(); it != prev_epoch_users.end(); it++)
          {
            if (ABOVE && (it->child == local_color))
              continue;
            if (ABOVE && (it->child >= 0) &&
                logical_node->are_children_disjoint(unsigned(local_color),
                                                    unsigned(it->child)))
              continue;
            if (IS_READ_ONLY(it->usage))
              continue;
            if (it->field_mask * (ABOVE ? copy_mask : non_dominated))
              continue;
            wait_on.insert(it->term_event);
          }
        }
      }
      else if (redop > 0)
      {
        // Only need to track non-dominated mask when not above
        FieldMask non_dominated;
        for (std::list<PhysicalUser>::const_iterator it = 
              curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
        {
          FieldMask overlap = it->field_mask & copy_mask;
          if (!overlap)
            continue;
          if (ABOVE && (it->child == local_color))
            continue;
          if (IS_REDUCE(it->usage) && (it->usage.redop == redop))
          {
            if (!ABOVE)
              non_dominated |= overlap;
            continue;
          }
          if (ABOVE && (it->child >= 0) &&
              logical_node->are_children_disjoint(unsigned(local_color),
                                                  unsigned(it->child)))
            continue;
          // Otherwise register a dependence
          wait_on.insert(it->term_event);
        }
        if (ABOVE || !!non_dominated)
        {
          for (std::list<PhysicalUser>::const_iterator it = 
                prev_epoch_users.begin(); it != prev_epoch_users.end(); it++)
          {
            if (ABOVE && (it->child == local_color))
              continue;
            if (ABOVE && (it->child >= 0) &&
                logical_node->are_children_disjoint(unsigned(local_color),
                                                    unsigned(it->child)))
              continue;
            if (IS_REDUCE(it->usage) && (it->usage.redop == redop))
              continue;
            if (it->field_mask * (ABOVE ? copy_mask : non_dominated))
              continue;
            wait_on.insert(it->term_event);
          }
        }
      }
      else
      {
        // Register dependences on everyone in with overlapping fields
        for (std::list<PhysicalUser>::const_iterator it = 
              curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
        {
          FieldMask overlap = it->field_mask & copy_mask;
          if (!overlap)
            continue;
          if (ABOVE && (it->child == local_color))
            continue;
          if (ABOVE && (it->child >= 0) &&
              logical_node->are_children_disjoint(unsigned(local_color),
                                                  unsigned(it->child)))
            continue;
          wait_on.insert(it->term_event);
        }
        // We know we dominated everyone if we're not above
        if (ABOVE)
        {
          for (std::list<PhysicalUser>::const_iterator it = 
                prev_epoch_users.begin(); it != prev_epoch_users.end(); it++)
          {
            if (it->field_mask * copy_mask)
              continue;
            if ((it->child >= 0) &&
                logical_node->are_children_disjoint(unsigned(local_color),
                                                    unsigned(it->child)))
              continue;
            wait_on.insert(it->term_event);
          }
        }
      }
#else
      if (!writing)
      {
        PhysicalCopyAnalyzer<true,false,!ABOVE,ABOVE> 
          copy_analyzer(copy_mask, 0, wait_on, local_color, logical_node);
        curr_epoch_users->analyze(copy_mask, copy_analyzer);
        const FieldMask &non_dominated = copy_analyzer.get_non_dominated_mask();
        if (ABOVE || !!non_dominated)
        {
          PhysicalCopyAnalyzer<true,false,false,ABOVE>
            prev_analyzer(copy_mask, 0, wait_on, local_color, logical_node);
          prev_epoch_users->analyze(
              (ABOVE ? copy_mask : non_dominated), prev_analyzer);
        }
      }
      else if (redop > 0)
      {
        PhysicalCopyAnalyzer<false,true,!ABOVE,ABOVE>
          copy_analyzer(copy_mask, redop, wait_on, local_color, logical_node);
        curr_epoch_users->analyze(copy_mask, copy_analyzer);
        const FieldMask &non_dominated = copy_analyzer.get_non_dominated_mask();
        if (ABOVE || !!non_dominated)
        {
          PhysicalCopyAnalyzer<false,true,false,ABOVE>
            prev_analyzer(copy_mask, redop, wait_on, local_color, logical_node);
          prev_epoch_users->analyze(
              (ABOVE ? copy_mask : non_dominated), prev_analyzer);
        }
      }
      else
      {
        PhysicalCopyAnalyzer<false,false,false,ABOVE>
          copy_analyzer(copy_mask, 0, wait_on, local_color, logical_node);
        curr_epoch_users->analyze(copy_mask, copy_analyzer);
        // We know we dominated everyone if we're not above
        if (ABOVE)
        {
          PhysicalCopyAnalyzer<false,false,false,ABOVE>
            prev_analyzer(copy_mask, 0, wait_on, local_color, logical_node);
          prev_epoch_users->analyze(copy_mask, prev_analyzer);
        }
      }
#endif
    }
#endif

    //--------------------------------------------------------------------------
    bool MaterializedView::has_war_dependence_above(const RegionUsage &usage,
                                                    const FieldMask &user_mask,
                                                    Color child_color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, HAS_WAR_DEPENDENCE_ABOVE_CALL);
#endif
      if ((parent != NULL) &&
          parent->has_war_dependence_above(usage, user_mask, 
                                           logical_node->get_color()))
        return true;
      int local_color = child_color;
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
#ifndef PHYSICAL_FIELD_TREE
      for (std::list<PhysicalUser>::const_iterator it = 
            curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
      {
        if (it->child == local_color)
          continue;
        if ((it->child >= 0) && 
            logical_node->are_children_disjoint(child_color, 
                                                unsigned(it->child)))
          continue;
        if (user_mask * it->field_mask)
          continue;
        if (IS_READ_ONLY(it->usage))
          return true;
      }
      return false; 
#else
      WARAnalyzer<true> analyzer(local_color, logical_node);
      curr_epoch_users->analyze(user_mask, analyzer);
      return analyzer.has_war_dependence();
#endif
    }

    //--------------------------------------------------------------------------
    void MaterializedView::update_versions(const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, UPDATE_VERSIONS_CALL);
#endif
      std::vector<VersionID> to_delete;
      std::map<VersionID,FieldMask> new_versions;
      for (std::map<VersionID,FieldMask>::iterator it = 
            current_versions.begin(); it != current_versions.end(); it++)
      {
        FieldMask overlap = it->second & update_mask;
        if (!!overlap)
        {
          new_versions[(it->first+1)] = overlap; 
          it->second -= update_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
      }
      for (std::vector<VersionID>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        current_versions.erase(*it);
      }
      for (std::map<VersionID,FieldMask>::const_iterator it = 
            new_versions.begin(); it != new_versions.end(); it++)
      {
        std::map<VersionID,FieldMask>::iterator finder = 
          current_versions.find(it->first);
        if (finder == current_versions.end())
          current_versions.insert(*it);
        else
          finder->second |= it->second;
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_local_users(Event term_event) 
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are in Legion Spy since we want to see
      // all of the dependences on an instance
#if !defined(LEGION_SPY) && !defined(LEGION_LOGGING)
      // should already be holding the lock when this is called
#ifndef PHYSICAL_FIELD_TREE
      for (std::list<PhysicalUser>::iterator it = curr_epoch_users.begin();
            it != curr_epoch_users.end(); /*nothing*/)
      {
        if (it->term_event == term_event)
          it = curr_epoch_users.erase(it);
        else
          it++;
      }
      for (std::list<PhysicalUser>::iterator it = prev_epoch_users.begin();
            it != prev_epoch_users.end(); /*nothing*/)
      {
        if (it->term_event == term_event)
          it = prev_epoch_users.erase(it);
        else
          it++;
      }
#else
      PhysicalEventFilter filter(term_event);
      curr_epoch_users->analyze<PhysicalEventFilter>(term_mask, filter);
      prev_epoch_users->analyze<PhysicalEventFilter>(term_mask, filter);
#endif
#endif
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_local_users(
                                             const std::set<Event> &term_events)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are in Legion Spy since we want to see
      // all of the dependences on an instance
#if !defined(LEGION_SPY) && !defined(LEGION_LOGGING)
      for (std::list<PhysicalUser>::iterator it = curr_epoch_users.begin();
            it != curr_epoch_users.end(); /*nothing*/)
      {
        if (term_events.find(it->term_event) != term_events.end())
          it = curr_epoch_users.erase(it);
        else
          it++;
      }
      for (std::list<PhysicalUser>::iterator it = prev_epoch_users.begin();
            it != prev_epoch_users.end(); /*nothing*/)
      {
        if (term_events.find(it->term_event) != term_events.end())
          it = prev_epoch_users.erase(it);
        else
          it++;
      }
#endif
    }

    //--------------------------------------------------------------------------
    template<typename ALLOC>
    void MaterializedView::condense_user_list(std::list<PhysicalUser,ALLOC> 
                                              &users, bool previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, CONDENSE_USER_LIST_CALL);
#endif
      // First try and regroup users with the same termination event
      // and privleges whose user masks have been split up for various reasons.
      // Also while scanning over the list, do a quick check for events
      // which have already triggered but haven't been collected yet.
      if (previous)
      {
        // Note storing pointers requires using an STL
        // list where nodes don't move
        std::map<Event,std::vector<PhysicalUser*> > earlier_users;
        for (std::list<PhysicalUser>::iterator it = users.begin();
              it != users.end(); /*nothing*/)
        {
          // Quick check for already triggered users
          if (it->term_event.has_triggered())
          {
            it = users.erase(it);
            continue;
          }
          std::map<Event,std::vector<PhysicalUser*> >::iterator 
            finder = earlier_users.find(it->term_event);
          if (finder == earlier_users.end())
          {
            // Haven't seen this event before, save its pointer
            // and then continue the traversal
            PhysicalUser *user = &(*it);
            earlier_users[it->term_event].push_back(user);
            it++;
          }
          else
          {
            // Otherwise, iterate over the current users and
            // see if we can find something with the same usage
            bool found = false;
            for (unsigned idx = 0; idx < finder->second.size(); idx++)
            {
              if ((it->usage == finder->second[idx]->usage) &&
                  (finder->second[idx]->child == it->child))
              {
                found = true;
                finder->second[idx]->field_mask |= it->field_mask;
                break;
              }
            }
            if (!found)
            {
              PhysicalUser *user =&(*it);
              finder->second.push_back(user);
              it++;
            }
            else
              it = users.erase(it);
          }
        }
      }
      else
      {
        // If this isn't previous, just do the check for triggered events
        for (std::list<PhysicalUser>::iterator it = users.begin();
              it != users.end(); /*nothing*/)
        {
          if (it->term_event.has_triggered())
            it = users.erase(it);
          else
            it++;
        }
      }

      // Now if that wasn't enough, go through all the users from children
      // and merge things with the same usage together.  This doesn't lose
      // very much precision because anyone who was waiting on us from 
      // below will still get precise waiters, and everyone checking from
      // this level would have registered on all the same children anyway.
      if (Runtime::enable_imprecise_filter &&
          (users.size() > (Runtime::max_filter_size/2)) && !children.empty())
      {
        for (std::list<PhysicalUser>::iterator it = users.begin();
              it != users.end(); /*nothing*/)
        {
          if (it->child >= 0)
          {
            // Iterate over the remaining elements looking for
            // users with the same privileges and field masks
            std::list<PhysicalUser>::iterator finder = it;
            finder++;
            std::set<Event> other_events;
            while (finder != users.end())
            {
              if ((it->child >= 0) && (it->usage == finder->usage) &&  
                  (it->field_mask == finder->field_mask))
              {
                // Can merge, reset the child information,
                // save the event, and remove from the list
                it->child = -1;
                other_events.insert(it->term_event);
                finder = users.erase(finder);
              }
              else
                finder++;
            }
            // See if we're doing a merge
            if (!other_events.empty())
            {
              // Add ourselves to the set
              other_events.insert(it->term_event);
              // Merge the events
              it->term_event = Event::merge_events(other_events);
              // Add a garbage collection task
              if (it->term_event.exists())
              {
                event_references.insert(it->term_event);
                defer_collect_user(it->term_event);
                it++;
              }
              else
                it = users.erase(it); // they all terminated
            }
            else
              it++;
          }
          else
            it++;
        }
      }

      // Finally, if that still didn't work, do something that might actually
      // harm precision: merge things from the first half of the list
      // which all shared the same privileges, regardless of mask or child.
      // To avoid merging things over and over again, we'll put the
      // results of the merge back at the end of the list.
      if (Runtime::enable_imprecise_filter &&
          (users.size() > (Runtime::max_filter_size/2)))
      {
        // Find the first element we won't let anyone merge
        std::list<PhysicalUser>::iterator end_user = users.begin();
        for (unsigned idx = 0; idx < (Runtime::max_filter_size/2); idx++)
          end_user++;
        unsigned difference = users.size() - (Runtime::max_filter_size/2);
        for (std::list<PhysicalUser>::iterator it = users.begin(); 
              (it != end_user) && (difference > 0); /*nothing*/)
        {
          std::list<PhysicalUser>::iterator finder = it;
          finder++;
          std::set<Event> to_merge;
          while ((difference > 0) && (finder != end_user))
          {
            // See if we found something to merge
            if (finder->usage == it->usage) 
            {
              to_merge.insert(finder->term_event);
              it->field_mask |= finder->field_mask;
              if (it->child != finder->child)
                it->child = -1;
              finder = users.erase(finder);
              difference--;
            }
            else
              finder++;
          }
          if (!to_merge.empty())
          {
            to_merge.insert(it->term_event);
            it->term_event = Event::merge_events(to_merge);
            if (it->term_event.exists())
            {
              // Add a garbage collection task
              event_references.insert(it->term_event);
              defer_collect_user(it->term_event);
              // Put the new item at the back of the list and
              // remove it from its current location.
              users.push_back(*it);
            }
            it = users.erase(it);
          }
          else // didn't find anything to merge with, keep going
            it++;
        }
      }
    }

    //--------------------------------------------------------------------------
    DistributedID MaterializedView::send_state(AddressSpaceID target,
                                               const FieldMask &send_mask,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      // If we've already packed this view then we are done with it
      // and everything above it
      std::map<LogicalView*,FieldMask>::iterator needed_finder = 
        needed_views.find(this);
      if (needed_finder == needed_views.end())
      {
        // Add ourselves to the needed views
        needed_views[this] = send_mask;
        // Always add a remote reference
        add_remote_reference();
        DistributedID parent_did = did;
        if (parent != NULL)
          parent_did = parent->send_state(target, send_mask, 
                                          needed_views,needed_managers);
        // Send the manager if it hasn't been sent and get it's ID
        DistributedID manager_did = 
          manager->send_manager(target, needed_managers);
        // Now see if we need to send ourselves
        DistributedID result = did;
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
            subscribers.find(target);
          // If we already have a remote view, we're done
          if (finder != subscribers.end())
            result = finder->second;
        }
        if (result != did)
        {
          // We already have a remote view so send the update
          send_updates(result, target, send_mask);
          return result;
        }
        // Otherwise if we make it here, we need to pack ourselves up
        // and send outselves to another node
        Serializer rez;
        DistributedID remote_did = 
          context->runtime->get_available_distributed_id();
        // If we don't have a parent save our did 
        // as the parent did which will tell the unpack
        // task that there is no parent
        if (parent == NULL)
          parent_did = remote_did;
        // Now pack up all the data, this has to be done
        // atomically to make sure that no one can add any
        // users while we are doing it, or tries to send
        // users to subscribers before we've actually sent
        // the subscriber.
        bool lost_race = false;
        {
          AutoLock v_lock(view_lock);
          // Check again to make sure we didn't lose the race
          {
            AutoLock gc(gc_lock,1,false/*exclusive*/);
            std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
              subscribers.find(target);
            if (finder != subscribers.end())
            {
              lost_race = true;
              result = finder->second;
            }
            else
              result = remote_did;
          }
          if (!lost_race)
          {
            {
              RezCheck z(rez);
              rez.serialize(remote_did);
              // Our processor and did as the owner
              rez.serialize(context->runtime->address_space);
              rez.serialize(did);
              rez.serialize(parent_did);
              rez.serialize(manager_did);
              rez.serialize(logical_node->get_color());
              rez.serialize(depth);
              pack_materialized_view(rez);
            }
            // Before sending the message, update the subscribers
            add_subscriber(target, result);
            // Now send the message
            context->runtime->send_materialized_view(target, rez);
          }
        }
        if (lost_race)
        {
          // Return the distributed ID
          context->runtime->free_distributed_id(remote_did);
          // Send the message
          send_updates(result, target, send_mask);
        }
        return result;
      }
      else
      {
        // Return the distributed ID of the view on the remote node
        DistributedID result;
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
            subscribers.find(target);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != subscribers.end());
#endif
          result = finder->second;
        }
        // See if we need to send an update
        FieldMask diff_mask = send_mask - needed_finder->second;
        if (!!diff_mask)
        {
          send_updates(result, target, diff_mask);
          // Update the mask with the fields we've sent
          needed_finder->second |= diff_mask;
        }
        return result;
      }
    }

    //--------------------------------------------------------------------------
    DistributedID MaterializedView::send_back_state(AddressSpaceID target,
                                                    const FieldMask &send_mask,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      if (owner_addr != target)
      {
#ifdef DEBUG_HIGH_LEVEL
        // If we're not remote and we need to be sent back
        // then we better be the owner
        assert(owner_did == did);
#endif
        DistributedID parent_did = did;
        if (parent != NULL)
          parent_did = parent->send_back_state(target, send_mask, 
                                               needed_managers);
        DistributedID manager_did = 
          manager->send_manager(target, needed_managers);
        DistributedID new_owner_did = 
          context->runtime->get_available_distributed_id();
        if (parent == NULL)
          parent_did = new_owner_did;
        // Need to hold the view lock when doing this so we
        // ensure that everything gets sent back properly
        bool return_new_did = false;
        {
          AutoLock v_lock(view_lock);
          // Recheck again here to avoid a race condition
          if (owner_addr == target)
          {
            return_new_did = true; 
          }
          else
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(new_owner_did);
              rez.serialize(parent_did);
              rez.serialize(manager_did);
              // Save our information so we can be added as a subscriber
              rez.serialize(did);
              rez.serialize(owner_addr);
              rez.serialize(logical_node->get_color());
              rez.serialize(depth);
              pack_materialized_view(rez);
            }
            // Before sending the message add resource reference that
            // will be held by the new owner
            add_resource_reference();
            // Add a remote reference that we hold on what we sent back
            add_held_remote_reference();
            context->runtime->send_back_materialized_view(target, rez);
            // Update our owner proc and did
            owner_addr = target;
            owner_did = new_owner_did;
          }
        }
        if (return_new_did)
          context->runtime->free_distributed_id(new_owner_did);
      }
      // Otherwise we're already remote from the target and therefore
      // all of our parents are also already remote so we are done
      return owner_did;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::is_persistent(void) const
    //--------------------------------------------------------------------------
    {
      return manager->is_persistent();
    }

    //--------------------------------------------------------------------------
    void MaterializedView::make_persistent(void)
    //--------------------------------------------------------------------------
    {
      manager->make_persistent(manager->local_space);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::pack_materialized_view(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // view_lock held from callers
      RezCheck z(rez);
      rez.serialize(inst_lock);
#ifndef PHYSICAL_FIELD_TREE
      rez.serialize(curr_epoch_users.size());
      for (std::list<PhysicalUser>::const_iterator it = 
            curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
      {
        rez.serialize(it->usage);
        rez.serialize(it->field_mask);
        rez.serialize(it->term_event);
        rez.serialize(it->child);
      }
      rez.serialize(prev_epoch_users.size());
      for (std::list<PhysicalUser>::const_iterator it = 
            prev_epoch_users.begin(); it != prev_epoch_users.end(); it++)
      {
        rez.serialize(it->usage);
        rez.serialize(it->field_mask);
        rez.serialize(it->term_event);
        rez.serialize(it->child);
      }
#else
      curr_epoch_users->pack_field_tree(rez);
      prev_epoch_users->pack_field_tree(rez);
#endif
      rez.serialize(current_versions.size());
      for (std::map<VersionID,FieldMask>::const_iterator it = 
            current_versions.begin(); it != current_versions.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::unpack_materialized_view(Deserializer &derez, 
                                                    AddressSpaceID source,
                                                    bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        Event lock_event = view_lock.acquire(0, true/*exclusive*/);
        lock_event.wait(true/*block*/);
      }
      DerezCheck z(derez);
      derez.deserialize(inst_lock);
      FieldSpaceNode *field_node = manager->region_node->column_source;
      std::map<Event,FieldMask> deferred_events;
#ifndef PHYSICAL_FIELD_TREE
      size_t num_current;
      derez.deserialize(num_current);
      for (unsigned idx = 0; idx < num_current; idx++)
      {
        curr_epoch_users.push_back(PhysicalUser());
        PhysicalUser &user = curr_epoch_users.back();
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        // Transform the field mask
        field_node->transform_field_mask(user.field_mask, source);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        // We only need to launch things once for each event
#ifdef DEBUG_HIGH_LEVEL
        assert(user.term_event.exists());
#endif
        event_references.insert(user.term_event);
        deferred_events[user.term_event] |= user.field_mask;
      }
      size_t num_previous;
      derez.deserialize(num_previous);
      for (unsigned idx = 0; idx < num_previous; idx++)
      {
        prev_epoch_users.push_back(PhysicalUser());
        PhysicalUser &user = prev_epoch_users.back();
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        // Transform the field mask
        field_node->transform_field_mask(user.field_mask, source);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        // We only need to have one reference for each event
#ifdef DEBUG_HIGH_LEVEL
        assert(user.term_event.exists());
#endif
        event_references.insert(user.term_event);
        deferred_events[user.term_event] |= user.field_mask;
      }
#else
      // Unpack the field trees
      curr_epoch_users->unpack_field_tree(derez);
      prev_epoch_users->unpack_field_tree(derez);
      // Now transform all the field masks and get the
      // set of events and field masks on which
      // to defer garbage collection.
      PhysicalUnpacker unpacker(field_node, source, deferred_events); 
      curr_epoch_users->analyze(FieldMask(FIELD_ALL_ONES), unpacker);
      prev_epoch_users->analyze(FieldMask(FIELD_ALL_ONES), unpacker);
#endif
      size_t num_versions;
      derez.deserialize(num_versions);
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        FieldMask version_mask;
        derez.deserialize(version_mask);
        // Transform the field mask
        field_node->transform_field_mask(version_mask, source);
        current_versions[vid] = version_mask;
      }
      // Now we can do the unlock
      if (need_lock)
        view_lock.release();
      // Now launch the waiting deferred events.  We wait until
      // here to do it so we don't need to hold the lock while unpacking
      Processor gc_proc = Machine::get_executing_processor();
      for (std::map<Event,FieldMask>::const_iterator it = 
            deferred_events.begin(); it != deferred_events.end(); it++)
      {
        defer_collect_user(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::send_updates(DistributedID remote_did,
                                        AddressSpaceID target,
                                        const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_did);
        rez.serialize(update_mask);
        std::deque<PhysicalUser> update_curr, update_prev;
        // Hold the view lock while iterating these data structures
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          for (std::list<PhysicalUser>::const_iterator it = 
                curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
          {
            FieldMask overlap = it->field_mask & update_mask;
            if (!overlap)
              continue;
            update_curr.push_back(*it);
            update_curr.back().field_mask = overlap;
          }
          for (std::list<PhysicalUser>::const_iterator it = 
                prev_epoch_users.begin(); it != prev_epoch_users.end(); it++)
          {
            FieldMask overlap = it->field_mask & update_mask;
            if (!overlap)
              continue;
            update_prev.push_back(*it);
            update_prev.back().field_mask = overlap;
          }
        }
        rez.serialize<size_t>(update_curr.size());
        for (std::deque<PhysicalUser>::const_iterator it = 
              update_curr.begin(); it != update_curr.end(); it++)
        {
          rez.serialize(it->usage);
          rez.serialize(it->field_mask);
          rez.serialize(it->term_event);
          rez.serialize(it->child);
        }
        rez.serialize<size_t>(update_prev.size());
        for (std::deque<PhysicalUser>::const_iterator it = 
              update_prev.begin(); it != update_prev.end(); it++)
        {
          rez.serialize(it->usage);
          rez.serialize(it->field_mask);
          rez.serialize(it->term_event);
          rez.serialize(it->child);
        }
      }
      runtime->send_materialized_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_updates(Deserializer &derez,
                                           AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_node = logical_node->column_source;
      FieldMask update_mask;
      derez.deserialize(update_mask);
      field_node->transform_field_mask(update_mask, source);
      // First do our unpacking
      size_t curr_updates_size;
      derez.deserialize(curr_updates_size);
      std::vector<PhysicalUser> curr_updates(curr_updates_size);
      for (unsigned idx = 0; idx < curr_updates_size; idx++)
      {
        PhysicalUser &user = curr_updates[idx];
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        field_node->transform_field_mask(user.field_mask, source);
      }
      size_t prev_updates_size;
      derez.deserialize(prev_updates_size);
      std::vector<PhysicalUser> prev_updates(prev_updates_size);
      for (unsigned idx = 0; idx < prev_updates_size; idx++)
      {
        PhysicalUser &user = prev_updates[idx];
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        field_node->transform_field_mask(user.field_mask, source);
      }
      {
        // This must be done atomically in order to be correct
        // because of the possibility of multiple updates to
        // the same fields being sent at the same time
        AutoLock v_lock(view_lock);
        // Go through and filter out any update fields from the
        // current and previous epoch users
        filter_list(curr_epoch_users, update_mask);
        curr_epoch_users.insert(curr_epoch_users.end(),
                                curr_updates.begin(), curr_updates.end());
        filter_list(prev_epoch_users, update_mask);
        prev_epoch_users.insert(prev_epoch_users.end(),
                                prev_updates.begin(), prev_updates.end());
        // Add the registered events
        for (std::vector<PhysicalUser>::const_iterator it = 
              curr_updates.begin(); it != curr_updates.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(it->term_event.exists());
#endif
          event_references.insert(it->term_event);
        }
        for (std::vector<PhysicalUser>::const_iterator it = 
              prev_updates.begin(); it != prev_updates.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(it->term_event.exists());
#endif
          event_references.insert(it->term_event);
        }
      }
      // Now we can add the new users to our set
      Processor curr_proc = Machine::get_executing_processor();
      for (std::vector<PhysicalUser>::const_iterator it =
            curr_updates.begin(); it != curr_updates.end(); it++)
      {
        defer_collect_user(it->term_event);
      }
      for (std::vector<PhysicalUser>::const_iterator it = 
            prev_updates.begin(); it != prev_updates.end(); it++)
      {
        defer_collect_user(it->term_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_materialized_view(
                                RegionTreeForest *context, Deserializer &derez,
                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner_addr;
      derez.deserialize(owner_addr);
      DistributedID owner_did;
      derez.deserialize(owner_did);
      DistributedID parent_did;
      derez.deserialize(parent_did);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      Color view_color;
      derez.deserialize(view_color);
      unsigned depth;
      derez.deserialize(depth);

      InstanceManager *manager = 
        context->find_manager(manager_did)->as_instance_manager();
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      MaterializedView *parent = NULL;
      // If the parent did is our did then that means we have no parent
      if (parent_did != did)
      {
        InstanceView *temp = context->find_view(parent_did)->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
        assert(temp != NULL);
        assert(!temp->is_composite_view());
#endif
        parent = temp->as_materialized_view();
      }
      // Make the new instance and unpack it
      MaterializedView *result;
      bool need_lock = false;
      if (parent != NULL)
      {
        // We can use the parent to find the logical node for this view
        RegionTreeNode *node = parent->logical_node->get_tree_child(view_color);
        result = legion_new<MaterializedView>(context, did, owner_addr, 
                                              owner_did, node, manager, 
                                              parent, depth);
        if (!parent->add_subview(result, view_color))
        {
          // The view already existed so add an alias
          result->set_no_free_did();
          legion_delete(result);
          result = parent->get_materialized_subview(view_color);
          result->add_alias_did(did);
          need_lock = true;
        }
      }
      else
      {
        // Use the manager region node for the logical node
        result = legion_new<MaterializedView>(context, did, owner_addr, 
                                      owner_did, manager->region_node, 
                                      manager, parent, depth);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->unpack_materialized_view(derez, source, need_lock);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_back_materialized_view(
                                RegionTreeForest *context, Deserializer &derez,
                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedID parent_did;
      derez.deserialize(parent_did);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      DistributedID sender_did;
      derez.deserialize(sender_did);
      AddressSpaceID sender_addr;
      derez.deserialize(sender_addr);
      Color view_color;
      derez.deserialize(view_color);
      unsigned depth;
      derez.deserialize(depth);

      InstanceManager *manager = 
        context->find_manager(manager_did)->as_instance_manager();
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      MaterializedView *parent = NULL;
      // If the parent did is our did then that means we have no parent
      if (parent_did != did)
      {
        InstanceView *temp = context->find_view(parent_did)->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
        assert(temp != NULL);
        assert(!temp->is_composite_view());
#endif
        parent = temp->as_materialized_view();
      }
      // Make the new instance and unpack it
      MaterializedView *result;
      bool need_lock = false;
      if (parent != NULL)
      {
        RegionTreeNode *node = 
          parent->logical_node->get_tree_child(view_color);
        result = legion_new<MaterializedView>(context, did, 
                                      context->runtime->address_space, did,
                                      node, manager, parent, depth);
        if (!parent->add_subview(result, view_color))
        {
          // The view already existed, so create an alias
          // Mark that we cannot reclaim the distributed ID on deletion
          result->set_no_free_did();
          legion_delete(result);
          result = parent->get_materialized_subview(view_color);
          result->add_alias_did(did);
          need_lock = true;
        }
      }
      else
      {
        result = legion_new<MaterializedView>(context, did,
                                      context->runtime->address_space, did,
                                      manager->region_node, manager, 
                                      parent, depth);
      }
      // Add the sender as a subscriber
      result->add_subscriber(sender_addr, sender_did);
      // Add a remote reference held by the view that sent this back
      result->add_remote_reference();
      // Unpack the rest of the state
      result->unpack_materialized_view(derez, source, need_lock);
    } 

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_updates(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      LogicalView *log_view = context->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(log_view != NULL);
      assert(!log_view->is_reduction_view());
#endif
      InstanceView *inst_view = log_view->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
      assert(!inst_view->is_composite_view());
#endif
      MaterializedView *view = inst_view->as_materialized_view();
      // In order to be done properly, this needs to be done atomically
      view->process_updates(derez, source);
    }

    //--------------------------------------------------------------------------
    template<typename ALLOC>
    /*static*/ void MaterializedView::filter_list(
         std::list<PhysicalUser,ALLOC> &user_list, const FieldMask &filter_mask)
    //--------------------------------------------------------------------------
    {
      for (typename std::list<PhysicalUser,ALLOC>::iterator it = 
            user_list.begin(); it != user_list.end(); /*nothing*/)
      {
        it->field_mask -= filter_mask;
        if (!it->field_mask)
          it = user_list.erase(it);
        else
          it++;
      }
    }

    /////////////////////////////////////////////////////////////
    // CompositeView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(RegionTreeForest *ctx, DistributedID did,
                              AddressSpaceID owner_proc, RegionTreeNode *node,
                              DistributedID owner_did, const FieldMask &mask,
                              CompositeView *par/*= NULL*/)
      : InstanceView(ctx, did, owner_proc, owner_did, node), 
        parent(par), valid_mask(mask)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    CompositeView::CompositeView(const CompositeView &rhs)
      : InstanceView(NULL, 0, 0, 0, NULL), parent(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeView::~CompositeView(void)
    //--------------------------------------------------------------------------
    {
      // Remove any resource references that we hold on child views
      for (std::map<Color,CompositeView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->second->remove_resource_reference())
          legion_delete(it->second);
      }
      children.clear();
      // Remove any references we have to our roots
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        if (it->first->remove_reference())
          legion_delete(it->first);
      }
      roots.clear();
      // Remove resource references on our valid reductions
      for (std::map<ReductionView*,ReduceInfo>::iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        if (it->first->remove_resource_reference())
          legion_delete(it->first);
      }
      valid_reductions.clear();
    }

    //--------------------------------------------------------------------------
    PhysicalManager* CompositeView::get_manager(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    bool CompositeView::is_persistent(void) const
    //--------------------------------------------------------------------------
    {
      // Composite views are never persistent
      return false;
    }

    //--------------------------------------------------------------------------
    void CompositeView::find_copy_preconditions(ReductionOpID redop, 
                                                bool reading,
                                                const FieldMask &copy_mask,
                                       std::map<Event,FieldMask> &preconditions)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void CompositeView::add_copy_user(ReductionOpID redop, Event copy_term,
                                      const FieldMask &mask, bool reading,
                                      Processor exec_proc)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    InstanceRef CompositeView::add_user(PhysicalUser &user,
                                        Processor exec_proc)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return InstanceRef();
    }

    //--------------------------------------------------------------------------
    bool CompositeView::reduce_to(ReductionOpID redop, 
                                  const FieldMask &reduce_mask,
                               std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_activate(void)
    //--------------------------------------------------------------------------
    {
      // Add a gc reference to our parent if we have one
      if (parent != NULL)
        parent->add_gc_reference();

      // Add gc references to all our reduction views
      for (std::map<ReductionView*,ReduceInfo>::iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        it->first->add_gc_reference();
      }

      // If we are the top level view, add gc references to all our instances
      if (parent == NULL)
      {
        for (std::map<CompositeNode*,FieldMask>::iterator it = 
              roots.begin(); it != roots.end(); it++)
        {
          it->first->add_gc_references();
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        for (std::map<CompositeNode*,FieldMask>::iterator it = 
              roots.begin(); it != roots.end(); it++)
        {
          it->first->remove_gc_references();
        }
      }

      for (std::map<ReductionView*,ReduceInfo>::iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        // No need to check for deletion condition since we hold resource refs
        it->first->remove_gc_reference(); 
      }

      if ((parent != NULL) && parent->remove_gc_reference())
        legion_delete(parent);
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<CompositeNode*,FieldMask>::iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        it->first->add_valid_references();
      }
      for (std::map<ReductionView*,ReduceInfo>::iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        it->first->add_valid_reference();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<CompositeNode*,FieldMask>::iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        it->first->remove_valid_references();
      }
      for (std::map<ReductionView*,ReduceInfo>::iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        // No need to check for deletion conditions since we hold resource refs
        it->first->remove_valid_reference();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::collect_users(const std::set<Event> &term_events) 
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void CompositeView::process_send_back_user(AddressSpaceID source,
                                               PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool CompositeView::is_composite_view(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    MaterializedView* CompositeView::as_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    CompositeView* CompositeView::as_composite_view(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<CompositeView*>(this);
    }

    //--------------------------------------------------------------------------
    bool CompositeView::has_parent_view(void) const
    //--------------------------------------------------------------------------
    {
      return (parent != NULL);
    }

    //--------------------------------------------------------------------------
    InstanceView* CompositeView::get_parent_view(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    void CompositeView::add_root(CompositeNode *root, const FieldMask &valid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid - valid_mask));
      // There should always be at most one root for each field
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        assert(it->second * valid);
      }
#endif
      std::map<CompositeNode*,FieldMask>::iterator finder = roots.find(root);
      if (finder == roots.end())
      {
        roots[root] = valid;
        // Add a reference for when we go to delete it
        root->add_reference();
      }
      else
        finder->second |= valid;
    }

    //--------------------------------------------------------------------------
    void CompositeView::update_reduction_views(ReductionView *view,
                                               const FieldMask &valid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid - valid_mask));
#endif
      // Handle a special case that occurs when flushing reductions
      if (logical_node == view->logical_node)
      {
        std::map<ReductionView*,ReduceInfo>::iterator finder = 
          valid_reductions.find(view);
        if (finder == valid_reductions.end())
        {
          view->add_resource_reference();
          view->add_valid_reference();
          if (logical_node->has_component_domains())
            valid_reductions[view] = ReduceInfo(valid, 
                                         logical_node->get_component_domains());
          else
            valid_reductions[view] = ReduceInfo(valid, 
                                                    logical_node->get_domain());
        }
        else
          finder->second.valid_fields |= valid;
        // Always update the reduction mask
        reduction_mask |= valid;
        return;
      }
      // See if we have an intersection with the reduction and if so save it 
      if (logical_node->intersects_with(view->logical_node))
      {
        std::map<ReductionView*,ReduceInfo>::iterator finder = 
          valid_reductions.find(view);
        if (finder == valid_reductions.end())
        {
          view->add_resource_reference();
          view->add_valid_reference();
          valid_reductions[view] = ReduceInfo(valid, 
                    logical_node->get_intersection_domains(view->logical_node));
        }
        else
          finder->second.valid_fields |= valid;
        // Always update the reduction mask
        reduction_mask |= valid;
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* CompositeView::get_subview(Color c)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        std::map<Color,CompositeView*>::const_iterator finder = 
                                                          children.find(c);
        if (finder != children.end())
          return finder->second;
      }
      DistributedID child_did =
        context->runtime->get_available_distributed_id();
      DistributedID child_own_did = child_did;
      if (did != owner_did)
        child_own_did = context->runtime->get_available_distributed_id();
      RegionTreeNode *child_node = logical_node->get_tree_child(c); 
      CompositeView *child_view = legion_new<CompositeView>(context, child_did, 
                                                    owner_addr, child_node, 
                                                    child_own_did, valid_mask,
                                                    this);
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        it->first->find_bounding_roots(child_view, it->second); 
      }
      child_view->add_resource_reference();
      for (std::map<ReductionView*,ReduceInfo>::const_iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        child_view->update_reduction_views(it->first, it->second.valid_fields);
      }
      // Retake the lock and try and add the child, see if
      // someone else added the child in the meantime
      {
        AutoLock v_lock(view_lock);
        std::map<Color,CompositeView*>::const_iterator finder = 
                                                          children.find(c);
        if (finder != children.end())
        {
          // Guaranteed to succeed
          if (child_view->remove_resource_reference())
            legion_delete(child_view);
          return finder->second;
        }
        children[c] = child_view;
      }
      if (child_did != child_own_did)
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(owner_did);
          rez.serialize(c);
          rez.serialize(child_own_did);
          rez.serialize(child_did);
        }
        // Add a held remote reference
        child_view->add_held_remote_reference();
        // Now notify the owner that it has a remote subscriber
        context->runtime->send_subscriber(owner_addr, rez);
      }
      return child_view;
    }

    //--------------------------------------------------------------------------
    bool CompositeView::add_subview(CompositeView *view, Color c)
    //--------------------------------------------------------------------------
    {
      bool added = true;
      {
        AutoLock v_lock(view_lock);
        if (children.find(c) == children.end())
          children[c] = view;
        else
          added = false;
      }
      if (added)
        view->add_resource_reference();
      return added;
    }

    //--------------------------------------------------------------------------
    void CompositeView::update_valid_mask(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      valid_mask |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeView::copy_to(const FieldMask &copy_mask, 
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void CompositeView::copy_from(const FieldMask &copy_mask, 
                               std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool CompositeView::has_war_dependence(const RegionUsage &usage, 
                                           const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    DistributedID CompositeView::send_state(AddressSpaceID target,
                                            const FieldMask &send_mask,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(send_mask - valid_mask));
#endif
      std::map<LogicalView*,FieldMask>::iterator needed_finder = 
        needed_views.find(this);
      if (needed_finder == needed_views.end())
      {
        needed_views[this] = send_mask;
        // Always add a remote reference
        add_remote_reference();
        DistributedID parent_did = did;
        if (parent != NULL)
        {
          // We never need to actually send any state here
          FieldMask empty_mask;
          parent_did = parent->send_state(target, empty_mask,
                                          needed_views, needed_managers);
        }
        // Now see if we need to send ourselves
        DistributedID result = did;
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
            subscribers.find(target);
          if (finder != subscribers.end())
            result = finder->second;
        }
        // If we already have a remote version, send any updates
        // and then we are done
        if (result != did)
        {
          send_updates(result, target, send_mask, 
                       needed_views, needed_managers); 
          return result;
        }
        // Otherwise if we make it here, we need to pack ourselves up
        // and send ourselves to another node
        Serializer rez;
        DistributedID remote_did = 
          context->runtime->get_available_distributed_id();
        // If we don't have a parent save our did as the
        // parent did which will tell the unpack task
        // that there is no parent
        if (parent == NULL)
          parent_did = remote_did;
        // Now pack up all the data, this has to be done
        // atomically to make sure that no one can add any
        // users while we are doing it, or tries to send
        // users to subscribers before we've actually sent
        // the subscriber.
        bool lost_race = false;
        {
          AutoLock v_lock(view_lock);
          {
            AutoLock gc(gc_lock,1,false/*exclusive*/);
            std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
              subscribers.find(target);
            if (finder != subscribers.end())
            {
              result = finder->second;
              lost_race = true;
            }
            else
              result = remote_did;
          }
          if (!lost_race)
          {
            {
              RezCheck z(rez);
              rez.serialize(remote_did);
              rez.serialize(did);
              rez.serialize(parent_did);
              // Our processor as the owner
              rez.serialize(context->runtime->address_space);
              rez.serialize(send_mask);
              if (logical_node->is_region())
              {
                rez.serialize<bool>(true);
                RegionNode *reg_node = logical_node->as_region_node();
                rez.serialize(reg_node->handle);
              }
              else
              {
                rez.serialize<bool>(false);
                PartitionNode *part_node = logical_node->as_partition_node();
                rez.serialize(part_node->handle);
              }
              pack_composite_view(rez, false/*send back*/, target, send_mask,
                                  needed_views, needed_managers);
            }
            // Before sending the message, update the subscribers
            add_subscriber(target, result);
            // Also record the state that we sent
            remote_state[target] = send_mask;
            // Now send the message
            context->runtime->send_composite_view(target, rez);
          }
        }
        if (lost_race)
        {
          // Free the distributed ID
          context->runtime->free_distributed_id(remote_did);
          // Sen the update
          send_updates(result, target, send_mask, 
                       needed_views, needed_managers); 
        }
        return result;
      }
      else
      {
        DistributedID result;
        // Return the distributed ID of the view on the remote node
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
            subscribers.find(target);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != subscribers.end());
#endif
          result = finder->second;
        }
        FieldMask diff_mask = send_mask - needed_finder->second;
        if (!!diff_mask)
        {
          send_updates(result, target, diff_mask, 
                       needed_views, needed_managers);
          needed_finder->second |= diff_mask;
        }
        return result;
      }
    }

    //--------------------------------------------------------------------------
    DistributedID CompositeView::send_back_state(AddressSpaceID target,
                                                 const FieldMask &send_mask,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(send_mask - valid_mask));
#endif
      if (owner_addr != target)
      {
#ifdef DEBUG_HIGH_LEVEL
        // If we're not remote and we need to be sent back
        // then we better be the owner
        assert(owner_did == did);
#endif
        DistributedID parent_did = did;
        if (parent != NULL)
          parent_did = parent->send_back_state(target, send_mask, 
                                               needed_managers);
        DistributedID new_owner_did = 
          context->runtime->get_available_distributed_id();
        // Need to hold the view lock when doing this so we
        // ensure that everything gets sent back properly
        bool return_new_did = false;
        {
          AutoLock v_lock(view_lock);
          // Recheck again here to avoid a race condition
          if (owner_addr == target)
          {
            return_new_did = true;
          }
          else
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(new_owner_did);
              // Save our information so we can be added as a subscriber
              rez.serialize(did);
              rez.serialize(parent_did);
              rez.serialize(owner_addr);
              rez.serialize(send_mask);
              if (logical_node->is_region())
              {
                rez.serialize<bool>(true);
                RegionNode *reg_node = logical_node->as_region_node();
                rez.serialize(reg_node->handle);
              }
              else
              {
                rez.serialize<bool>(false);
                PartitionNode *part_node = logical_node->as_partition_node();
                rez.serialize(part_node->handle);
              }
              std::map<LogicalView*,FieldMask> dummy_views;
              pack_composite_view(rez, true/*send back*/, target, send_mask,
                                  dummy_views, needed_managers);
            }
            // Before sending the message add resource reference
            // that will be held by the new owner
            add_resource_reference();
            // Add a remote reference that we hold on what we sent back
            add_held_remote_reference();
            context->runtime->send_back_composite_view(target, rez);
            // Update our owner proc and did
            owner_addr = target;
            owner_did = new_owner_did;
          }
        }
        if (return_new_did)
          context->runtime->free_distributed_id(new_owner_did);
      }
      return owner_did;
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_composite_copies(MappableInfo *info,
                                               MaterializedView *dst,
                                               const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(copy_mask - valid_mask));
#endif
      std::map<Event,FieldMask> preconditions;
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                   copy_mask, preconditions);
      // Iterate over all the roots and issue copies to update the 
      // target instance from this particular view
      std::map<Event,FieldMask> postconditions;
#ifdef DEBUG_HIGH_LEVEL
      FieldMask accumulate_mask;
#endif
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        it->first->issue_update_copies(info, dst, overlap, overlap,
                                       preconditions, postconditions);
#ifdef DEBUG_HIGH_LEVEL
        assert(overlap * accumulate_mask);
        accumulate_mask |= overlap;
#endif
      }
      // Fun trick here, use the precondition set routine to get the
      // sets of fields which all have the same precondition events
      std::list<PreconditionSet> postcondition_sets;
      RegionTreeNode::compute_precondition_sets(copy_mask,
                                                postconditions,
                                                postcondition_sets);
      // Now add all the post conditions for each of the
      // writes for fields with the same set of post condition events
      for (std::list<PreconditionSet>::iterator pit = 
            postcondition_sets.begin(); pit !=
            postcondition_sets.end(); pit++)
      {
        PreconditionSet &post_set = *pit;
        // Go through and issue any reductions for this set
        if (!valid_reductions.empty())
        {
          FieldMask reduce_overlap = reduction_mask & post_set.pre_mask;
          if (!!reduce_overlap)
          {
            flush_reductions(info, dst, reduce_overlap,
                             preconditions, post_set.preconditions);
          }
        }
        // If the set is empty, then we can skip it
        if (post_set.preconditions.empty())
          continue;
        // Compute the merge event
        Event postcondition = Event::merge_events(post_set.preconditions);
        dst->add_copy_user(0/*redop*/, postcondition,
                           post_set.pre_mask, false/*reading*/,
                           info->local_proc);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_composite_copies(MappableInfo *info,
                                               MaterializedView *dst,
                                               const FieldMask &copy_mask,
                                const std::map<Event,FieldMask> &preconditions,
                                      std::map<Event,FieldMask> &postconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(copy_mask - valid_mask));
#endif
      std::map<Event,FieldMask> local_postconditions;
#ifdef DEBUG_HIGH_LEVEL
      FieldMask accumulate_mask;
#endif
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        it->first->issue_update_copies(info, dst, overlap, overlap,
                                       preconditions, local_postconditions);
#ifdef DEBUG_HIGH_LEVEL
        assert(overlap * accumulate_mask);
        accumulate_mask |= overlap;
#endif
      }
      // Fun trick here, use the precondition set routine to get the
      // sets of fields which all have the same precondition events
      std::list<PreconditionSet> postcondition_sets;
      RegionTreeNode::compute_precondition_sets(copy_mask,
                                                local_postconditions,
                                                postcondition_sets);
      // Now add all the post conditions for each of the
      // writes for fields with the same set of post condition events
      for (std::list<PreconditionSet>::iterator pit = 
            postcondition_sets.begin(); pit !=
            postcondition_sets.end(); pit++)
      {
        PreconditionSet &post_set = *pit;
        // Go through and issue any reductions for this set
        if (!valid_reductions.empty())
        {
          FieldMask reduce_overlap = reduction_mask & post_set.pre_mask;
          if (!!reduce_overlap)
          {
            flush_reductions(info, dst, reduce_overlap,
                             preconditions, post_set.preconditions);
          }
        }
        // If the set is empty, then we can skip it
        if (post_set.preconditions.empty())
          continue;
        // Compute the merge event
        Event postcondition = Event::merge_events(post_set.preconditions);
        if (postcondition.exists())
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(postconditions.find(postcondition) == postconditions.end());
#endif
          postconditions[postcondition] = post_set.pre_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_composite_copies_across(MappableInfo *info,
                                                      MaterializedView *dst,
                                                      FieldID src_field,
                                                      FieldID dst_field,
                                                      Event precondition,
                                                std::set<Event> &postconditions)
    //--------------------------------------------------------------------------
    {
      unsigned src_index = 
        logical_node->column_source->get_field_index(src_field);
      std::set<Event> preconditions;
      // This includes the destination precondition
      preconditions.insert(precondition);
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        if (it->second.is_set(src_index))
        {
          it->first->issue_across_copies(info, dst, src_index, src_field, 
                                         dst_field, true/*need field*/,
                                         preconditions, postconditions);
          // We know there is at most one root here so
          // once we find it then we are done
          break;
        }
      }
      if (!valid_reductions.empty() && reduction_mask.is_set(src_index))
      {
        std::set<Event> reduce_preconditions = postconditions;
        reduce_preconditions.insert(precondition);
        FieldMask reduce_mask; reduce_mask.set_bit(src_index);
        for (std::map<ReductionView*,ReduceInfo>::const_iterator it = 
              valid_reductions.begin(); it != valid_reductions.end(); it++)
        {
          Event result = 
            it->first->perform_composite_across_reduction(dst, dst_field,
                                                          src_field, src_index,
                                                          info->local_proc,
                                                          reduce_preconditions,
                                                      it->second.intersections);
          if (result.exists())
            postconditions.insert(result);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::flush_reductions(MappableInfo *info,
                                         MaterializedView *dst,
                                         const FieldMask &event_mask,
                                 const std::map<Event,FieldMask> &preconditions,
                                         std::set<Event> &event_set)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(event_mask - valid_mask));
#endif
      // Get the set of reductions that need to be issued for the
      // fields in this precondition set
      std::map<ReductionView*,FieldMask> overlap_reductions;
      FieldMask overlap_mask;
      for (std::map<ReductionView*,ReduceInfo>::const_iterator it =
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        FieldMask overlap = it->second.valid_fields & event_mask;
        if (!overlap)
          continue;
        overlap_reductions[it->first] = overlap; 
        overlap_mask |= overlap;
      }
      if (!overlap_reductions.empty())
      {
        // Compute the preconditions for these fields
        // Make sure that all the events in the event_set are
        // included in the reduction preconditions because they
        // are copies are also writing to the fields.
        std::set<Event> reduce_preconditions = event_set;
        for (std::map<Event,FieldMask>::const_iterator it = 
              preconditions.begin(); it != preconditions.end(); it++)
        {
          if (it->second * overlap_mask)
            continue;
          reduce_preconditions.insert(it->first);
        }
        // Now issue all of the reductions to the target
        for (std::map<ReductionView*,FieldMask>::const_iterator it = 
              overlap_reductions.begin(); it != 
              overlap_reductions.end(); it++)
        {
          std::map<ReductionView*,ReduceInfo>::const_iterator finder =
            valid_reductions.find(it->first);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != valid_reductions.end());
#endif
          Event result = it->first->perform_composite_reduction(dst, 
                                          it->second, info->local_proc, 
                                          reduce_preconditions, 
                                          finder->second.intersections);
          // Add the result to the set of post conditions
          if (result.exists())
            event_set.insert(result);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::pack_composite_view(Serializer &rez, bool send_back,
                                            AddressSpaceID target,
                                            const FieldMask &pack_mask,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(pack_mask - valid_mask));
#endif
      RezCheck z(rez);
      // First pack up all of our roots necessary roots
      std::map<CompositeNode*,FieldMask> send_roots;
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        send_roots[it->first] = overlap;
      }
      rez.serialize<size_t>(send_roots.size());
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            send_roots.begin(); it != send_roots.end(); it++)
      {
        // Before we can pack the composite node, we first need to 
        // makes sure we have the necessary region tree shape in
        // place for the composite nodes. We already know the necessary
        // field space and logical region are there
        const bool is_region = it->first->logical_node->is_region();
        rez.serialize<bool>(is_region);
        if (is_region)
        {
          RegionNode *reg_node = it->first->logical_node->as_region_node();
          reg_node->row_source->send_node(target, true/*up*/, true/*down*/);
          rez.serialize(reg_node->handle);
        }
        else
        {
          PartitionNode *part_node = 
                              it->first->logical_node->as_partition_node(); 
          part_node->row_source->send_node(target, true/*up*/, true/*down*/);
          rez.serialize(part_node->handle);
        }
        // Now pack the name of the node
        it->first->pack_composite_node(rez, send_back, target, it->second,
                                       needed_views, needed_managers);
        rez.serialize(it->second);
      }
      // Now send any reduction views
      std::map<ReductionView*,FieldMask> send_reductions;
      for (std::map<ReductionView*,ReduceInfo>::const_iterator it = 
            valid_reductions.begin(); it != valid_reductions.end(); it++)
      {
        FieldMask overlap = it->second.valid_fields & pack_mask;
        if (!overlap)
          continue;
        send_reductions[it->first] = overlap;
      }
      rez.serialize<size_t>(send_reductions.size());
      if (!send_reductions.empty())
      {
        FieldMask send_reduction_mask = reduction_mask & pack_mask;
        rez.serialize(send_reduction_mask);
        for (std::map<ReductionView*,FieldMask>::iterator it = 
              send_reductions.begin(); it != send_reductions.end(); it++)
        {
          std::map<ReductionView*,ReduceInfo>::iterator finder = 
            valid_reductions.find(it->first);
          DistributedID red_did = 
            it->first->send_state(target, it->second,
                                  needed_views, needed_managers);
          rez.serialize(red_did);
          rez.serialize(it->second);
          rez.serialize<size_t>(finder->second.intersections.size());
          for (std::set<Domain>::const_iterator dit = 
                finder->second.intersections.begin(); dit !=
                finder->second.intersections.end(); dit++)
          {
            rez.serialize(*dit);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::unpack_composite_view(Deserializer &derez,
                                              AddressSpaceID source,
                                              bool send_back, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        Event lock_event = view_lock.acquire(0, true/*exclusive*/);
        lock_event.wait(true/*block*/);
      }
      DerezCheck z(derez);
      size_t num_roots;
      derez.deserialize(num_roots);
      FieldSpaceNode *field_node = logical_node->get_column_source();
      for (unsigned idx = 0; idx < num_roots; idx++)
      {
        bool is_region;
        derez.deserialize<bool>(is_region);
        RegionTreeNode *region_tree_node;
        if (is_region)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          region_tree_node = context->get_node(handle);
        }
        else
        {
          LogicalPartition handle;
          derez.deserialize(handle);
          region_tree_node = context->get_node(handle);
        }
        CompositeNode *new_root = legion_new<CompositeNode>(region_tree_node,
                                              ((CompositeNode*)NULL/*parent*/));
        new_root->add_reference();
        new_root->unpack_composite_node(derez, source);
        FieldMask &root_mask = roots[new_root];
        derez.deserialize(root_mask);
        field_node->transform_field_mask(root_mask, source);
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      if (num_reductions > 0)
      {
        FieldMask reduction_update;
        derez.deserialize(reduction_update);
        field_node->transform_field_mask(reduction_update, source);
        reduction_mask |= reduction_update;
        for (unsigned idx = 0; idx < num_reductions; idx++)
        {
          DistributedID red_did;
          derez.deserialize(red_did);
          LogicalView *log_view = context->find_view(red_did); 
#ifdef DEBUG_HIGH_LEVEL
          assert(log_view != NULL);
          assert(log_view->is_reduction_view());
#endif
          ReductionView *red_view = log_view->as_reduction_view();
          ReduceInfo &red_info = valid_reductions[red_view];
          FieldMask red_update;
          derez.deserialize(red_update);
          field_node->transform_field_mask(red_update, source);
          red_info.valid_fields |= red_update;
          size_t num_domains;
          derez.deserialize(num_domains);
          for (unsigned i = 0; i < num_domains; i++)
          {
            Domain dom;
            derez.deserialize(dom);
            red_info.intersections.insert(dom);
          }
        }
      }
      if (need_lock)
        view_lock.release();
    }

    //--------------------------------------------------------------------------
    void CompositeView::send_updates(DistributedID remote_did, 
                                     AddressSpaceID target, FieldMask send_mask,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      std::map<AddressSpaceID,FieldMask>::iterator finder = 
        remote_state.find(target);
      // Update our send mask and then mark that we sent any needed fields
      if (finder != remote_state.end())
      {
        send_mask -= finder->second;
        finder->second |= send_mask;
      }
      else
        remote_state[target] = send_mask;
      // If we have any field data to send, do that now
      if (!!send_mask)
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_did);
          rez.serialize(send_mask);
          pack_composite_view(rez, false/*send back*/, target, send_mask,
                              needed_views, needed_managers);
        }
        context->runtime->send_composite_update(target, rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_send_composite_view(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      DistributedID did;
      derez.deserialize(did);
      DistributedID owner_did;
      derez.deserialize(owner_did);
      DistributedID parent_did;
      derez.deserialize(parent_did);
      AddressSpaceID owner_addr;
      derez.deserialize(owner_addr);
      FieldMask valid_mask;
      derez.deserialize(valid_mask);
      // Transform below
      bool is_region;
      derez.deserialize<bool>(is_region);
      RegionTreeNode *logical_node;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        logical_node = context->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        logical_node = context->get_node(handle);
      }
      FieldSpaceNode *field_node = logical_node->column_source;
      field_node->transform_field_mask(valid_mask, source);
      CompositeView *parent = NULL;
      if (parent_did != did)
      {
        InstanceView *temp = context->find_view(parent_did)->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
        assert(temp != NULL);
        assert(temp->is_composite_view());
#endif
        parent = temp->as_composite_view();
      }
      CompositeView *result;
      bool need_lock = false;
      if (parent != NULL)
      {
        Color view_color = logical_node->get_color();
        result = legion_new<CompositeView>(context, did, owner_addr,
                                   logical_node, owner_did, valid_mask, parent);
        if (!parent->add_subview(result, view_color))
        {
          result->set_no_free_did();
          legion_delete(result);
          result = parent->get_subview(view_color)->as_composite_view();
          result->add_alias_did(did);
          result->update_valid_mask(valid_mask);
          need_lock = true;
        }
      }
      else
      {
        result = legion_new<CompositeView>(context, did, owner_addr, 
                                   logical_node, owner_did, valid_mask);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->unpack_composite_view(derez, source, 
                                    false/*send back*/, need_lock);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_send_back_composite_view(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedID sender_did;
      derez.deserialize(sender_did);
      DistributedID parent_did;
      derez.deserialize(parent_did);
      AddressSpaceID sender_addr;
      derez.deserialize(sender_addr);
      FieldMask valid_mask;
      derez.deserialize(valid_mask);
      // Transform below
      bool is_region;
      derez.deserialize<bool>(is_region);
      RegionTreeNode *logical_node;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        logical_node = context->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        logical_node = context->get_node(handle);
      }
      FieldSpaceNode *field_node = logical_node->column_source;
      field_node->transform_field_mask(valid_mask, source);
      CompositeView *parent = NULL;
      if (parent_did != did)
      {
        InstanceView *temp = context->find_view(parent_did)->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
        assert(temp != NULL);
        assert(temp->is_composite_view());
#endif
        parent = temp->as_composite_view();
      }
      CompositeView *result;
      bool need_lock = false;
      if (parent != NULL)
      {
        Color view_color = logical_node->get_color();
        result = legion_new<CompositeView>(context, did,
                                   context->runtime->address_space,
                                   logical_node, did, valid_mask, parent);
        if (!parent->add_subview(result, view_color))
        {
          // The view already existed so create an alias
          result->set_no_free_did();
          legion_delete(result);
          result = parent->get_subview(view_color)->as_composite_view();
          result->add_alias_did(did);
          result->update_valid_mask(valid_mask);
          need_lock = true;
        }
      }
      else
      {
        // Note we don't ever give a parent because we don't need to
        result = legion_new<CompositeView>(context, did,
                                   context->runtime->address_space,
                                   logical_node, did, valid_mask);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      // Add the sender as a subscriber
      result->add_subscriber(sender_addr, sender_did);
      // Add a remote reference held by the view that sent this back
      result->add_remote_reference();
      // Unpack the rest of the state
      result->unpack_composite_view(derez, source, 
                                    true/*send back*/, need_lock);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_send_composite_update(
          RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask valid_mask;
      derez.deserialize(valid_mask);
      // Transform below
      LogicalView *log_view = context->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(!log_view->is_reduction_view());
#endif
      InstanceView *inst_view = log_view->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
      assert(inst_view->is_composite_view());
#endif
      CompositeView *view = inst_view->as_composite_view();
      FieldSpaceNode *field_node = view->logical_node->column_source;
      field_node->transform_field_mask(valid_mask, source);
      view->update_valid_mask(valid_mask);
      view->unpack_composite_view(derez, source, 
                                  false/*send back*/, true/*need lock*/);
    }

    /////////////////////////////////////////////////////////////
    // CompositeNode 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(RegionTreeNode *logical, CompositeNode *par)
      : Collectable(), context(logical->context), logical_node(logical), 
        parent(par)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(const CompositeNode &rhs)
      : Collectable(), context(NULL), logical_node(NULL), parent(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeNode::~CompositeNode(void)
    //--------------------------------------------------------------------------
    {
      // Free up all our children
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        if (it->first->remove_reference())
          legion_delete(it->first);
      }
      open_children.clear();
      // Remove our resource references
      for (std::map<InstanceView*,FieldMask>::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_resource_reference())
        {
          if (it->first->is_composite_view())
            legion_delete(it->first->as_composite_view());
          else
            legion_delete(it->first->as_materialized_view());
        }
      }
      valid_views.clear();
    }

    //--------------------------------------------------------------------------
    CompositeNode& CompositeNode::operator=(const CompositeNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::capture_physical_state(RegionTreeNode *tree_node,
                                               PhysicalState &state,
                                               const FieldMask &capture_mask,
                                               CompositeCloser &closer,
                                               FieldMask &global_dirt,
                                               FieldMask &complete_mask)
    //--------------------------------------------------------------------------
    {
      // Capture the global dirt we are passing back
      global_dirt |= capture_mask & (state.dirty_mask | state.reduction_mask);
      // Also track the fields that we have dirty data for
      FieldMask local_dirty = capture_mask & state.dirty_mask;
      // Record the local dirty fields
      dirty_mask |= local_dirty;
      // Also mark that we are complete for these fields
      complete_mask |= local_dirty;
      // Don't just capture the valid views we have here, we need to
      // capture valid views for everyone above in the tree
      std::map<InstanceView*,FieldMask> instances;
      tree_node->find_valid_instance_views(state, capture_mask, capture_mask,
                                           false/*needs space*/, instances);
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        // Update the instance views
        update_instance_views(it->first, overlap);
      }
      for (std::map<ReductionView*,FieldMask>::const_iterator it = 
            state.reduction_views.begin(); it != 
            state.reduction_views.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        closer.update_reduction_views(it->first, overlap); 
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::update_parent_info(const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->update_child_info(this, capture_mask);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::update_child_info(CompositeNode *child, 
                                          const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      std::map<CompositeNode*,ChildInfo>::iterator finder = 
                                                    open_children.find(child); 
      if (finder == open_children.end())
      {
        // If we didn't find it, we have to make it
        // and determine if it is complete
        bool complete = child->logical_node->is_complete();
        open_children[child] = ChildInfo(complete, mask);
        // Add a reference to it
        child->add_reference();
      }
      else
        finder->second.open_fields |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::update_instance_views(InstanceView *view,
                                              const FieldMask &valid_mask)
    //--------------------------------------------------------------------------
    {
      std::map<InstanceView*,FieldMask>::iterator finder = 
        valid_views.find(view);
      if (finder == valid_views.end())
      {
        view->add_resource_reference();
        valid_views[view] = valid_mask;
      }
      else
        finder->second |= valid_mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::issue_update_copies(MappableInfo *info,
                                            MaterializedView *dst,
                                            FieldMask traversal_mask,
                                            const FieldMask &copy_mask,
                                      const std::map<Event,FieldMask> &preconds,
                                      std::map<Event,FieldMask> &postconditions)
    //--------------------------------------------------------------------------
    {
      // First check to see if any of our children are complete
      // If they are then we can skip issuing any copies from this level
      std::map<Event,FieldMask> dst_preconditions = preconds;
      if (!valid_views.empty())
      {
        // The fields we need to update are any in the traversal
        // mask plus any from the original copy for which we have dirty data
        FieldMask incomplete_mask = traversal_mask | (dirty_mask & copy_mask);
        // Otherwise, if none are complete, we need to issue update copies
        // from this level assuming we have instances that intersect
        bool already_valid = false;
        // Do a quick check to see if we are done early
        {
          std::map<InstanceView*,FieldMask>::const_iterator finder = 
            valid_views.find(dst);
          if ((finder != valid_views.end()) && 
              !(incomplete_mask - finder->second))
            already_valid = true;
        }
        if (!already_valid && !!incomplete_mask)
        {
          RegionTreeNode *target = dst->logical_node;
          std::map<InstanceView*,FieldMask> valid_instances;
          for (std::map<InstanceView*,FieldMask>::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            FieldMask overlap = incomplete_mask & it->second;
            if (!overlap)
              continue;
            valid_instances[it->first] = overlap;
          }
          std::map<MaterializedView*,FieldMask> src_instances;
          std::map<CompositeView*,FieldMask> composite_instances;
          // Note that this call destroys valid_instances 
          // and updates incomplete_mask
          target->sort_copy_instances(info, dst, incomplete_mask, 
                      valid_instances, src_instances, composite_instances);
          if (!src_instances.empty())
          {
            std::map<Event,FieldMask> update_preconditions;
            FieldMask update_mask;
            for (std::map<MaterializedView*,FieldMask>::const_iterator it =
                  src_instances.begin(); it != src_instances.end(); it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!!it->second);
#endif
              it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                             it->second, update_preconditions);
              update_mask |= it->second;
            }
            // Also get the set of destination preconditions
            for (std::map<Event,FieldMask>::const_iterator it = 
                  preconds.begin(); it != preconds.end(); it++)
            {
              FieldMask overlap = update_mask & it->second;
              if (!overlap)
                continue;
              std::map<Event,FieldMask>::iterator finder = 
                update_preconditions.find(it->first);
              if (finder == update_preconditions.end())
                update_preconditions[it->first] = overlap;
              else
                finder->second |= overlap;
            }
            // Now we have our preconditions so we can issue our copy
            std::map<Event,FieldMask> update_postconditions;
            RegionTreeNode::issue_grouped_copies(info, dst, 
                         update_preconditions, update_mask, 
                         find_intersection_domains(dst->logical_node),
                         src_instances, update_postconditions);
            // If we dominate the target, then we can remove
            // the update_mask fields from the traversal_mask
            if (dominates(dst->logical_node))
              traversal_mask -= update_mask;
            // Add all our updates to both the dst_preconditions
            // as well as the actual postconditions.  No need to
            // check for duplicates as we know all these events
            // are brand new and can't be anywhere else.
            if (!update_postconditions.empty())
            {
#ifdef DEBUG_HIGH_LEVEL
              for (std::map<Event,FieldMask>::const_iterator it = 
                    update_postconditions.begin(); it != 
                    update_postconditions.end(); it++)
              {
                assert(dst_preconditions.find(it->first) == 
                       dst_preconditions.end());
                assert(postconditions.find(it->first) == 
                       postconditions.end());
              }
#endif
              dst_preconditions.insert(update_postconditions.begin(),
                                       update_postconditions.end());
              postconditions.insert(update_postconditions.begin(),
                                    update_postconditions.end());
            }
          }
          // Now if we still have fields which aren't
          // updated then we need to see if we have composite
          // views for those fields
          if (!composite_instances.empty())
          {
            FieldMask update_mask;
            for (std::map<CompositeView*,FieldMask>::const_iterator it = 
                  composite_instances.begin(); it !=
                  composite_instances.end(); it++)
            {
              std::map<Event,FieldMask> postconds;
              it->first->issue_composite_copies(info, dst, it->second,
                                                preconds, postconds);
              update_mask |= it->second;
              if (!postconds.empty())
              {
#ifdef DEBUG_HIGH_LEVEL
                for (std::map<Event,FieldMask>::const_iterator it = 
                      postconds.begin(); it != postconds.end(); it++)
                {
                  assert(dst_preconditions.find(it->first) ==
                         dst_preconditions.end());
                  assert(postconditions.find(it->first) ==
                         postconditions.end());
                }
#endif
                dst_preconditions.insert(postconds.begin(), postconds.end());
                postconditions.insert(postconds.begin(), postconds.end());
              }
            }
            // If we dominate the logical node we can remove the
            // updated fields from the traversal mask
            if (dominates(dst->logical_node))
              traversal_mask -= update_mask;
          }
        }
      }

      // Now traverse any open children that intersect with the destination
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = copy_mask & it->second.open_fields;
        // If we have no fields in common or we don't intersect with
        // the child then we can skip traversing this child
        if (!overlap || !it->first->intersects_with(dst->logical_node))
          continue;
        // If we make it here then we need to traverse the child
        it->first->issue_update_copies(info, dst, traversal_mask, 
                                       overlap, dst_preconditions, 
                                       postconditions);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::issue_across_copies(MappableInfo *info,
                                            MaterializedView *dst,
                                            unsigned src_index,
                                            FieldID  src_field,
                                            FieldID  dst_field,
                                            bool    need_field,
                                            std::set<Event> &preconditions,
                                            std::set<Event> &postconditions)
    //--------------------------------------------------------------------------
    {
      std::set<Event> dst_preconditions = preconditions;
      if (!valid_views.empty())
      {
        bool incomplete = need_field || dirty_mask.is_set(src_index);
        if (incomplete)
        {
          FieldMask src_mask; src_mask.set_bit(src_index);
          std::map<InstanceView*,FieldMask> valid_instances;
          for (std::map<InstanceView*,FieldMask>::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            if (it->second.is_set(src_index))
              valid_instances[it->first] = src_mask;
          }
          std::map<MaterializedView*,FieldMask> src_instances;
          std::map<CompositeView*,FieldMask> composite_instances;
          dst->logical_node->sort_copy_instances(info, dst, src_mask,
                      valid_instances, src_instances, composite_instances);
          if (!src_instances.empty())
          {
            // There should be at most one of these
#ifdef DEBUG_HIGH_LEVEL
            assert(src_instances.size() == 1);
#endif
            MaterializedView *src = (src_instances.begin())->first;
            std::map<Event,FieldMask> src_preconditions;
            src->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                         src_mask, src_preconditions);
            for (std::map<Event,FieldMask>::const_iterator it = 
                  src_preconditions.begin(); it != 
                  src_preconditions.end(); it++)
            {
              preconditions.insert(it->first);
            }
            Event copy_pre = Event::merge_events(preconditions);
            std::set<Event> result_events;
            std::vector<Domain::CopySrcDstField> src_fields, dst_fields;
            src->copy_field(src_field, src_fields);
            dst->copy_field(dst_field, dst_fields);
            const std::set<Domain> &overlap_domains = 
              find_intersection_domains(dst->logical_node);
            for (std::set<Domain>::const_iterator it = overlap_domains.begin();
                  it != overlap_domains.end(); it++)
            {
              result_events.insert(it->copy(src_fields, dst_fields, copy_pre));
            }
            Event copy_post = Event::merge_events(result_events);
            if (copy_post.exists())
            {
              // Only need to record the source user as the destination
              // user will be recorded by the copy across operation
              src->add_copy_user(0/*redop*/, copy_post,
                                 src_mask, true/*reading*/,
                                 info->local_proc);
              // Also add the event to the dst_preconditions and 
              // our post conditions
              dst_preconditions.insert(copy_post);
              postconditions.insert(copy_post);
            }
            // If we dominate then we no longer need to get
            // updates unless they are dirty
            if (dominates(dst->logical_node))
              need_field = false;
          }
          else if (!composite_instances.empty())
          {
            // There should be at most one of these
#ifdef DEBUG_HIGH_LEVEL
            assert(composite_instances.size() == 1); 
#endif
            CompositeView *src = (composite_instances.begin())->first; 
            std::set<Event> postconds;
            Event pre = Event::merge_events(preconditions);
            src->issue_composite_copies_across(info, dst, src_field,
                                               dst_field, pre, postconds);
            if (!postconds.empty())
            {
              dst_preconditions.insert(postconds.begin(), postconds.end());
              postconditions.insert(postconds.begin(), postconds.end());
            }
            // If we dominate then we no longer need to get
            // updates unless they are dirty
            if (dominates(dst->logical_node))
              need_field = false;
          }
        }
      }
      // Now traverse any open children that intersect with the destination
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        if ((it->second.open_fields.is_set(src_index)) && 
            it->first->intersects_with(dst->logical_node))
        {
          it->first->issue_across_copies(info, dst, src_index,
                                         src_field, dst_field, need_field,
                                         dst_preconditions, postconditions);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool CompositeNode::intersects_with(RegionTreeNode *dst)
    //--------------------------------------------------------------------------
    {
      return logical_node->intersects_with(dst);
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& CompositeNode::find_intersection_domains(
                                                            RegionTreeNode *dst)
    //--------------------------------------------------------------------------
    {
      return logical_node->get_intersection_domains(dst);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::find_bounding_roots(CompositeView *target,
                                            const FieldMask &bounding_mask)
    //--------------------------------------------------------------------------
    {
      // See if we can fields with exactly one child that dominates
      FieldMask single, multi;
      std::map<CompositeNode*,FieldMask> dominators;
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = it->second.open_fields & bounding_mask;
        if (!overlap)
          continue;
        if (!it->first->dominates(target->logical_node))
          continue;
        std::map<CompositeNode*,FieldMask>::iterator finder = 
          dominators.find(it->first);
        if (finder == dominators.end())
          dominators[it->first] = overlap;
        else
          finder->second |= overlap;
        // Update the multi mask first 
        multi |= (single & overlap);
        // Now update the single mask
        single |= overlap;
      }
      // Subtract any fields from the multi mask from the single mask
      if (!!multi)
        single -= multi;
      // If we still have any single fields then go and issue them
      if (!!single)
      {
        for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
              dominators.begin(); it != dominators.end(); it++)
        {
          FieldMask overlap = single & it->second;
          if (!overlap)
            continue;
          it->first->find_bounding_roots(target, overlap);
        }
        // Now see if we have any leftover fields here
        FieldMask local_mask = bounding_mask - single;
        if (!!local_mask)
          target->add_root(this, local_mask);
      }
      else
      {
        // There were no single fields so add ourself
        target->add_root(this, bounding_mask);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::add_gc_references(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<InstanceView*,FieldMask>::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_gc_reference();
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->add_gc_references();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::remove_gc_references(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            valid_views.end(); it != valid_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->remove_gc_reference();
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->remove_gc_references();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::add_valid_references(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<InstanceView*,FieldMask>::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_valid_reference();
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->add_valid_references();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::remove_valid_references(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            valid_views.end(); it != valid_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->remove_valid_reference();
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->remove_valid_references();
      }
    }
    
    //--------------------------------------------------------------------------
    bool CompositeNode::dominates(RegionTreeNode *dst)
    //--------------------------------------------------------------------------
    {
      return logical_node->dominates(dst);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::pack_composite_node(Serializer &rez, bool send_back, 
                                            AddressSpaceID target,
                                            const FieldMask &pack_mask,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      FieldMask dirty_overlap = pack_mask & dirty_mask;
      rez.serialize(dirty_overlap);
      // Pack up all of our instances
      std::map<InstanceView*,FieldMask> send_valid;
      for (std::map<InstanceView*,FieldMask>::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        send_valid[it->first] = overlap;
      }
      rez.serialize<size_t>(send_valid.size());
      if (send_back)
      {
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              send_valid.begin(); it != send_valid.end(); it++)
        {
          DistributedID view_did = it->first->send_back_state(target, 
                                                  it->second, needed_managers);
          rez.serialize(view_did);
          rez.serialize(it->second);
        }
      }
      else
      {
        for (std::map<InstanceView*,FieldMask>::const_iterator it = 
              send_valid.begin(); it != send_valid.end(); it++)
        {
          DistributedID view_did = 
                  it->first->send_state(target, it->second,
                                        needed_views, needed_managers);
          rez.serialize(view_did);
          rez.serialize(it->second);
        }
      }
      // Then send all of our children
      std::map<CompositeNode*,FieldMask> send_children;
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = it->second.open_fields & pack_mask;
        if (!overlap)
          continue;
        send_children[it->first] = overlap;
      }
      rez.serialize<size_t>(send_children.size());
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            send_children.begin(); it != send_children.end(); it++)
      {
        std::map<CompositeNode*,ChildInfo>::const_iterator finder = 
          open_children.find(it->first);
        bool is_region = it->first->logical_node->is_region();
        rez.serialize<bool>(is_region);
        if (is_region)
        {
          RegionNode *reg_node = it->first->logical_node->as_region_node();
          rez.serialize(reg_node->handle);
        }
        else
        {
          PartitionNode *part_node = 
                              it->first->logical_node->as_partition_node();
          rez.serialize(part_node->handle);
        }
        it->first->pack_composite_node(rez, send_back, target, it->second,
                                       needed_views, needed_managers);
        rez.serialize<bool>(finder->second.complete);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::unpack_composite_node(Deserializer &derez,
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpaceNode *field_node = logical_node->get_column_source();
      FieldMask dirty_update;
      derez.deserialize(dirty_update);
      field_node->transform_field_mask(dirty_update, source);
      dirty_mask |= dirty_update;
      size_t num_views;
      derez.deserialize(num_views);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        LogicalView *log_view = context->find_view(view_did);
#ifdef DEBUG_HIGH_LEVEL
        assert(log_view != NULL);
        assert(!log_view->is_reduction_view());
#endif
        InstanceView *inst_view = log_view->as_instance_view();
        FieldMask &valid_mask = valid_views[inst_view];
        FieldMask update_mask;
        derez.deserialize(update_mask);
        field_node->transform_field_mask(update_mask, source);
        valid_mask |= update_mask;
      }
      size_t num_children;
      derez.deserialize(num_children);
      for (unsigned idx = 0; idx < num_children; idx++)
      {
        bool is_region;
        derez.deserialize<bool>(is_region);
        RegionTreeNode *region_tree_node;
        if (is_region)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          region_tree_node = context->get_node(handle);
        }
        else
        {
          LogicalPartition handle;
          derez.deserialize(handle);
          region_tree_node = context->get_node(handle);
        }
        CompositeNode *new_node = 
          legion_new<CompositeNode>(region_tree_node, this);
        new_node->add_reference();
        new_node->unpack_composite_node(derez, source); 
        ChildInfo &info = open_children[new_node];
        derez.deserialize<bool>(info.complete);
        derez.deserialize(info.open_fields);
        field_node->transform_field_mask(info.open_fields, source);
      }
    }
    
    /////////////////////////////////////////////////////////////
    // ReductionView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(RegionTreeForest *ctx, DistributedID did,
                                 AddressSpaceID own_addr, DistributedID own_did,
                                 RegionTreeNode *node, ReductionManager *man)
      : LogicalView(ctx, did, own_addr, own_did, node), manager(man)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      manager->add_resource_reference();
    }

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(const ReductionView &rhs)
      : LogicalView(NULL, 0, 0, 0, NULL), manager(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReductionView::~ReductionView(void)
    //--------------------------------------------------------------------------
    {
      if (manager->remove_resource_reference())
      {
        if (manager->is_list_manager())
          legion_delete(manager->as_list_manager());
        else
          legion_delete(manager->as_fold_manager());
      }
    }

    //--------------------------------------------------------------------------
    ReductionView& ReductionView::operator=(const ReductionView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReductionView::perform_reduction(LogicalView *target,
                                          const FieldMask &reduce_mask,
                                          Processor local_proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_REDUCTION_CALL);
#endif
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, reduce_mask, dst_fields);
      this->reduce_from(manager->redop, reduce_mask, src_fields);

      std::map<Event,FieldMask> preconditions;
      target->find_copy_preconditions(manager->redop, false/*reading*/, 
                                      reduce_mask, preconditions); 
      this->find_copy_preconditions(manager->redop, true/*reading*/, 
                                    reduce_mask, preconditions);
      std::set<Event> event_preconds;
      for (std::map<Event,FieldMask>::const_iterator it = preconditions.begin();
            it != preconditions.end(); it++)
      {
        event_preconds.insert(it->first);
      }
      Event reduce_pre = Event::merge_events(event_preconds); 
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      if (!reduce_pre.exists())
      {
        UserEvent new_reduce_pre = UserEvent::create_user_event();
        new_reduce_pre.trigger();
        reduce_pre = new_reduce_pre;
      }
      IndexSpace reduce_index_space;
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependences(
          Machine::get_executing_processor(), event_preconds, reduce_pre);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(event_preconds, reduce_pre);
#endif
      Event reduce_post; 
      if (logical_node->has_component_domains())
      {
        std::set<Event> post_events;
        const std::set<Domain> &component_domains = 
          logical_node->get_component_domains();
        for (std::set<Domain>::const_iterator it = 
              component_domains.begin(); it != component_domains.end(); it++)
        {
          Event post = manager->issue_reduction(src_fields, dst_fields,
                                                *it, reduce_pre, fold, 
                                                true/*precise*/);
          post_events.insert(post);
        }
        reduce_post = Event::merge_events(post_events);
#if defined(LEGION_SPY) || defined(LEGION_LOGGING)
        Domain domain = logical_node->get_domain();
        reduce_index_space = domain.get_index_space();
#endif
      }
      else
      {
        Domain domain = logical_node->get_domain();
        reduce_post = manager->issue_reduction(src_fields, dst_fields,
                                               domain, reduce_pre, fold,
                                               true/*precise*/);
#if defined(LEGION_SPY) || defined(LEGION_LOGGING)
        reduce_index_space = domain.get_index_space();
#endif
      }
#if defined(LEGION_SPY) || defined(LEGION_LOGGING)
      if (!reduce_post.exists())
      {
        UserEvent new_reduce_post = UserEvent::create_user_event();
        new_reduce_post.trigger();
        reduce_post = new_reduce_post;
      }
#endif
      target->add_copy_user(manager->redop, reduce_post,
                            reduce_mask, false/*reading*/, local_proc);
      this->add_copy_user(manager->redop, reduce_post,
                          reduce_mask, true/*reading*/, local_proc);
#ifdef LEGION_LOGGING
      {
        std::set<FieldID> reduce_fields;
        manager->region_node->column_source->to_field_set(reduce_mask,
                                                          reduce_fields);
        LegionLogging::log_lowlevel_copy(
            Machine::get_executing_processor(),
            manager->get_instance(),
            target->get_manager()->get_instance(),
            reduce_index_space,
            manager->region_node->column_source->handle,
            manager->region_node->handle.tree_id,
            reduce_pre, reduce_post, reduce_fields, manager->redop);
      }
#endif
#ifdef LEGION_SPY
      char *string_mask = 
        manager->region_node->column_source->to_string(reduce_mask);
      LegionSpy::log_copy_operation(manager->get_instance().id,
          target->get_manager()->get_instance().id, reduce_index_space.id,
          manager->region_node->column_source->handle.id,
          manager->region_node->handle.tree_id, reduce_pre, reduce_post,
          manager->redop, string_mask);
#endif
    } 

    //--------------------------------------------------------------------------
    Event ReductionView::perform_composite_reduction(MaterializedView *target,
                                                     const FieldMask &red_mask,
                                                     Processor local_proc,
                                                     const std::set<Event> &pre,
                                         const std::set<Domain> &reduce_domains)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_REDUCTION_CALL);
#endif
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, red_mask, dst_fields);
      this->reduce_from(manager->redop, red_mask, src_fields);

      std::map<Event,FieldMask> src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/,
                              red_mask, src_pre);
      std::set<Event> preconditions = pre;
      for (std::map<Event,FieldMask>::const_iterator it = src_pre.begin();
            it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      Event reduce_pre = Event::merge_events(preconditions); 
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      if (!reduce_pre.exists())
      {
        UserEvent new_reduce_pre = UserEvent::create_user_event();
        new_reduce_pre.trigger();
        reduce_pre = new_reduce_pre;
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependences(
          Machine::get_executing_processor(), preconditions, reduce_pre);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(preconditions, reduce_pre);
#endif
      std::set<Event> post_events;
      for (std::set<Domain>::const_iterator it = reduce_domains.begin();
            it != reduce_domains.end(); it++)
      {
        Event post = manager->issue_reduction(src_fields, dst_fields,
                                              *it, reduce_pre, fold,
                                              false/*precise*/);
        post_events.insert(post);
      }
      Event reduce_post = Event::merge_events(post_events);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post,
                    red_mask, true/*reading*/, local_proc);
#if defined(LEGION_SPY) || defined(LEGION_LOGGING)
      Domain domain = logical_node->get_domain();
      IndexSpace reduce_index_space = 
              target->logical_node->get_domain().get_index_space();
      if (!reduce_post.exists())
      {
        UserEvent new_reduce_post = UserEvent::create_user_event();
        new_reduce_post.trigger();
        reduce_post = new_reduce_post;
      }
#endif
#ifdef LEGION_LOGGING
      {
        std::set<FieldID> reduce_fields;
        manager->region_node->column_source->to_field_set(red_mask,
                                                          reduce_fields);
        LegionLogging::log_lowlevel_copy(
            Machine::get_executing_processor(),
            manager->get_instance(),
            target->get_manager()->get_instance(),
            reduce_index_space,
            manager->region_node->column_source->handle,
            manager->region_node->handle.tree_id,
            reduce_pre, reduce_post, reduce_fields, manager->redop);
      }
#endif
#ifdef LEGION_SPY
      char *string_mask = 
        manager->region_node->column_source->to_string(red_mask);
      LegionSpy::log_copy_operation(manager->get_instance().id,
          target->get_manager()->get_instance().id, reduce_index_space.id,
          manager->region_node->column_source->handle.id,
          manager->region_node->handle.tree_id, reduce_pre, reduce_post,
          manager->redop, string_mask);
#endif
      return reduce_post;
    }

    //--------------------------------------------------------------------------
    Event ReductionView::perform_composite_across_reduction(
        MaterializedView *target, FieldID dst_field, FieldID src_field,
                              unsigned src_index, Processor local_proc, 
                              const std::set<Event> &preconds,
                              const std::set<Domain> &reduce_domains)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_REDUCTION_CALL);
#endif
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      const bool fold = false;
      target->copy_field(dst_field, dst_fields);
      FieldMask red_mask; red_mask.set_bit(src_index);
      this->reduce_from(manager->redop, red_mask, src_fields);

      std::map<Event,FieldMask> src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/,
                              red_mask, src_pre);
      std::set<Event> preconditions = preconds;
      for (std::map<Event,FieldMask>::const_iterator it = src_pre.begin();
            it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      Event reduce_pre = Event::merge_events(preconditions); 
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      if (!reduce_pre.exists())
      {
        UserEvent new_reduce_pre = UserEvent::create_user_event();
        new_reduce_pre.trigger();
        reduce_pre = new_reduce_pre;
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependences(
          Machine::get_executing_processor(), preconditions, reduce_pre);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(preconditions, reduce_pre);
#endif
      std::set<Event> post_events;
      for (std::set<Domain>::const_iterator it = reduce_domains.begin();
            it != reduce_domains.end(); it++)
      {
        Event post = manager->issue_reduction(src_fields, dst_fields,
                                              *it, reduce_pre, fold,
                                              false/*precise*/);
        post_events.insert(post);
      }
      Event reduce_post = Event::merge_events(post_events);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post,
                    red_mask, true/*reading*/, local_proc);
#if defined(LEGION_SPY) || defined(LEGION_LOGGING)
      Domain domain = logical_node->get_domain();
      IndexSpace reduce_index_space = 
              target->logical_node->get_domain().get_index_space();
      if (!reduce_post.exists())
      {
        UserEvent new_reduce_post = UserEvent::create_user_event();
        new_reduce_post.trigger();
        reduce_post = new_reduce_post;
      }
#endif
#ifdef LEGION_LOGGING
      {
        std::set<FieldID> reduce_fields;
        manager->region_node->column_source->to_field_set(red_mask,
                                                          reduce_fields);
        LegionLogging::log_lowlevel_copy(
            Machine::get_executing_processor(),
            manager->get_instance(),
            target->get_manager()->get_instance(),
            reduce_index_space,
            manager->region_node->column_source->handle,
            manager->region_node->handle.tree_id,
            reduce_pre, reduce_post, reduce_fields, manager->redop);
      }
#endif
#ifdef LEGION_SPY
      char *string_mask = 
        manager->region_node->column_source->to_string(red_mask);
      LegionSpy::log_copy_operation(manager->get_instance().id,
          target->get_manager()->get_instance().id, reduce_index_space.id,
          manager->region_node->column_source->handle.id,
          manager->region_node->handle.tree_id, reduce_pre, reduce_post,
          manager->redop, string_mask);
#endif
      return reduce_post;
    }

    //--------------------------------------------------------------------------
    bool ReductionView::is_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    InstanceView* ReductionView::as_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    ReductionView* ReductionView::as_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<ReductionView*>(this);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* ReductionView::get_manager(void) const
    //--------------------------------------------------------------------------
    {
      return manager;
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_copy_preconditions(ReductionOpID redop,
                                                bool reading,
                                                const FieldMask &copy_mask,
                                      std::map<Event,FieldMask> &preconditions)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      if (reading)
      {
        // Register dependences on any reducers
        for (std::list<PhysicalUser>::const_iterator it = 
              reduction_users.begin(); it != reduction_users.end(); it++)
        {
          std::map<Event,FieldMask>::iterator finder = 
            preconditions.find(it->term_event);
          if (finder == preconditions.end())
            preconditions[it->term_event] = copy_mask;
          else
            finder->second |= copy_mask;
        }
      }
      else
      {
        // Register dependences on any readers
        for (std::list<PhysicalUser>::const_iterator it = reading_users.begin();
              it != reading_users.end(); it++)
        {
          std::map<Event,FieldMask>::iterator finder = 
            preconditions.find(it->term_event);
          if (finder == preconditions.end())
            preconditions[it->term_event] = copy_mask;
          else
            finder->second |= copy_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_copy_user(ReductionOpID redop, Event copy_term,
                                      const FieldMask &mask, bool reading,
                                      Processor exec_proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_COPY_USER_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop);
#endif
      // Quick test: only need to do this if copy term exists
      if (copy_term.exists())
      {
        AutoLock v_lock(view_lock);
        if (reading)
        { 
          PhysicalUser user(RegionUsage(READ_ONLY,EXCLUSIVE,0),mask,copy_term);
          reading_users.push_back(user);
          if (owner_did != did)
            send_back_user(user);
        }
        else
        { 
          PhysicalUser user(RegionUsage(REDUCE,EXCLUSIVE,redop),mask,copy_term);
          reduction_users.push_back(user);
          if (owner_did != did)
            send_back_user(user);
        }
        // Update the reference users
        event_references.insert(copy_term);
        // Launch the garbage collection task
        defer_collect_user(copy_term);
      }
    }

    //--------------------------------------------------------------------------
    InstanceRef ReductionView::add_user(PhysicalUser &user, Processor exec_proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_USER_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(IS_REDUCE(user.usage));
      assert(user.usage.redop == manager->redop);
#endif
      AutoLock v_lock(view_lock);
      // Wait on any readers currently reading the instance
      std::set<Event> wait_on;
      for (std::list<PhysicalUser>::const_iterator it = reading_users.begin();
            it != reading_users.end(); it++)
      {
        wait_on.insert(it->term_event);
      }
      // Only need to do this if we actually have a term event
      if (user.term_event.exists())
      {
        reduction_users.push_back(user);
        // Update the reference users
        event_references.insert(user.term_event);
        // Launch the garbage collection task
        defer_collect_user(user.term_event);
        if (owner_did != did)
          send_back_user(user);
      }
      // Return our result
      Event result = Event::merge_events(wait_on);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      if (!result.exists())
      {
        UserEvent new_result = UserEvent::create_user_event();
        new_result.trigger();
        result = new_result;
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependences(
          Machine::get_executing_processor(), wait_on, result);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(wait_on, result);
#endif
      return InstanceRef(result, 
                         Reservation::NO_RESERVATION, ViewHandle(this));
    }
 
    //--------------------------------------------------------------------------
    bool ReductionView::reduce_to(ReductionOpID redop, 
                                  const FieldMask &reduce_mask,
                              std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REDUCE_TO_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop);
#endif
      // Get the destination fields for this copy
      manager->find_field_offsets(reduce_mask, dst_fields);
      return manager->is_foldable();
    }
    
    //--------------------------------------------------------------------------
    void ReductionView::reduce_from(ReductionOpID redop,
                                    const FieldMask &reduce_mask,
                              std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REDUCE_FROM_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop);
#endif
      manager->find_field_offsets(reduce_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_activate(void)
    //--------------------------------------------------------------------------
    {
      manager->add_gc_reference();
    }

    //--------------------------------------------------------------------------
    void ReductionView::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
      // No need to check for deletion of the manager since
      // we know that we also hold a resource reference
      manager->remove_gc_reference();
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      manager->add_valid_reference();
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      manager->remove_valid_reference();
    }

    //--------------------------------------------------------------------------
    void ReductionView::collect_users(const std::set<Event> &term_events)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      bool has_local_references = false;
      for (std::set<Event>::const_iterator it = term_events.begin();
            it != term_events.end(); it++)
      {
        std::set<Event>::iterator finder = event_references.find(*it);
        if (finder != event_references.end())
        {
          event_references.erase(finder);
          has_local_references = true;
        }
      }
#if !defined(LEGION_SPY) && !defined(LEGION_LOGGING)
      if (has_local_references)
      {
        // Do not do this if we are in LegionSpy so we can see 
        // all of the dependences
        for (std::list<PhysicalUser>::iterator it = reduction_users.begin();
              it != reduction_users.end(); /*nothing*/)
        {
          if (term_events.find(it->term_event) != term_events.end())
            it = reduction_users.erase(it);
          else
            it++;
        }
        for (std::list<PhysicalUser>::iterator it = reading_users.begin();
              it != reading_users.end(); /*nothing*/)
        {
          if (term_events.find(it->term_event) != term_events.end())
            it = reading_users.erase(it);
          else
            it++;
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_send_back_user(AddressSpaceID source,
                                               PhysicalUser &user)
    //--------------------------------------------------------------------------
    {
      // First transform the user field mask
      manager->region_node->column_source->transform_field_mask(
                                                      user.field_mask, source);
      // Notify our subscribers and our owner if necessary
      AutoLock v_lock(view_lock);
      if (owner_did != did)
        send_back_user(user);
      if (IS_REDUCE(user.usage))
      {
        reduction_users.push_back(user);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(IS_READ_ONLY(user.usage));
#endif
        reading_users.push_back(user);
      }
      // Update the reference users
      event_references.insert(user.term_event);
      // Launch the garbage collection task
      defer_collect_user(user.term_event);
    }

    //--------------------------------------------------------------------------
    DistributedID ReductionView::send_state(AddressSpaceID target,
                                            const FieldMask &send_mask,
                                 std::map<LogicalView*,FieldMask> &needed_views,
                                 std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalView*,FieldMask>::iterator needed_finder = 
        needed_views.find(this);
      if (needed_finder == needed_views.end())
      {
        needed_views[this] = send_mask;
        // always add a remote reference
        add_remote_reference();
        DistributedID manager_did = 
          manager->send_manager(target, needed_managers);
        DistributedID result = did;
        // Now see if we need to send ourselves
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
            subscribers.find(target);
          // If we already have a remote subscriber, then we are done
          if (finder != subscribers.end())
            result = finder->second;
        }
        if (result != did)
        {
          // We already have a remote view so send the update
          send_updates(result, target, send_mask);
        }
        // Otherwise we need to send a copy remotely
        Serializer rez;
        DistributedID remote_did = 
          context->runtime->get_available_distributed_id();
        bool lost_race = false;
        {
          AutoLock v_lock(view_lock);
          // Check again to see if we lost the race
          {
            AutoLock gc(gc_lock,1,false/*exclusive*/);
            std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
              subscribers.find(target);
            if (finder != subscribers.end())
            {
              result = finder->second;
              lost_race = true;
            }
            else
              result = remote_did;
          }
          if (!lost_race)
          {
            {
              RezCheck z(rez);
              rez.serialize(result);
              // Our processor and did as the owner
              rez.serialize(context->runtime->address_space);
              rez.serialize(did);
              rez.serialize(manager_did);
              pack_reduction_view(rez);
            }
            // Before sending the message update the subscribers
            add_subscriber(target, result);
            context->runtime->send_reduction_view(target, rez);
          }
        }
        if (lost_race)
        {
          // Return the distributed id
          context->runtime->free_distributed_id(remote_did);
          // Send the update
          send_updates(result, target, send_mask);
        }
        return result;
      }
      else
      {
        // Otherwise there is nothing to do since we've already
        // been registered
        DistributedID result;
        {
          AutoLock gc(gc_lock,1,false/*exclusive*/);
          std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
            subscribers.find(target);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != subscribers.end());
#endif
          result = finder->second;
        }
        FieldMask diff_mask = send_mask - needed_finder->second;
        if (!!diff_mask)
        {
          send_updates(result, target, diff_mask);
          needed_finder->second |= diff_mask;
        }
        return result;
      }
    }

    //--------------------------------------------------------------------------
    DistributedID ReductionView::send_back_state(AddressSpaceID target,
                                                 const FieldMask &send_mask,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      if (owner_addr != target)
      {
#ifdef DEBUG_HIGH_LEVEL
        // If we're not remote and we need to be sent back
        // then we better be the owner
        assert(owner_did == did);
#endif
        DistributedID manager_did = 
          manager->send_manager(target, needed_managers);
        DistributedID new_owner_did = 
          context->runtime->get_available_distributed_id();
        Serializer rez;
        AutoLock v_lock(view_lock);
        {
          RezCheck z(rez);
          rez.serialize(new_owner_did);
          rez.serialize(manager_did);
          // Save our information so we can be added as a subscriber
          rez.serialize(did);
          rez.serialize(owner_addr);
          pack_reduction_view(rez);
        }
        // Before sending the message add resource reference that
        // will be held by the new owner on this view
        add_resource_reference();
        // Add a held remote reference on what we sent back
        add_held_remote_reference();
        context->runtime->send_back_reduction_view(target, rez);
        // Update our owner proc and did
        owner_addr = target;
        owner_did = new_owner_did;
      }
#ifdef DEBUG_HIGH_LEVEL
      else
      {
        // We better be holding some remote references
        // to guarantee that the owner is still there.
        assert(held_remote_references > 0);
      }
#endif
      return owner_did;
    }

    //--------------------------------------------------------------------------
    void ReductionView::pack_reduction_view(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // view lock held from callers
      RezCheck z(rez);
      rez.serialize(reduction_users.size());
      for (std::list<PhysicalUser>::const_iterator it = 
            reduction_users.begin(); it != reduction_users.end(); it++)
      {
        rez.serialize(it->usage);
        rez.serialize(it->field_mask);
        rez.serialize(it->term_event);
        rez.serialize(it->child);
      }
      rez.serialize(reading_users.size());
      for (std::list<PhysicalUser>::const_iterator it = 
            reading_users.begin(); it != reading_users.end(); it++)
      {
        rez.serialize(it->usage);
        rez.serialize(it->field_mask);
        rez.serialize(it->term_event);
        rez.serialize(it->child);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::unpack_reduction_view(Deserializer &derez, 
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      size_t num_reduction;
      derez.deserialize(num_reduction);
      FieldSpaceNode *field_node = manager->region_node->column_source;
      for (unsigned idx = 0; idx < num_reduction; idx++)
      {
        reduction_users.push_back(PhysicalUser());
        PhysicalUser &user = reduction_users.back();
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        // Transform the field mask
        field_node->transform_field_mask(user.field_mask, source);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        event_references.insert(user.term_event);
      }
      size_t num_reading;
      derez.deserialize(num_reading);
      for (unsigned idx = 0; idx < num_reading; idx++)
      {
        reading_users.push_back(PhysicalUser());
        PhysicalUser &user = reading_users.back();
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        // Transform the field mask
        field_node->transform_field_mask(user.field_mask, source);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        event_references.insert(user.term_event);
      }
      // Now launch the waiting deferred events.  We wait until
      // here to do it so we don't need to hold the lock while unpacking
      Processor gc_proc = Machine::get_executing_processor();
      for (std::list<PhysicalUser>::const_iterator it = 
            reduction_users.begin(); it != reduction_users.end(); it++)
      {
        defer_collect_user(it->term_event);
      }
      for (std::list<PhysicalUser>::const_iterator it = 
            reading_users.begin(); it != reading_users.end(); it++)
      {
        defer_collect_user(it->term_event);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::send_updates(DistributedID remote_did, 
                                     AddressSpaceID target,
                                     const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        std::deque<PhysicalUser> read_users, red_users;
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          for (std::list<PhysicalUser>::const_iterator it = 
                reduction_users.begin(); it != reduction_users.end(); it++)
          {
            FieldMask overlap = it->field_mask & update_mask;
            if (!overlap)
              continue;
            red_users.push_back(*it);
            red_users.back().field_mask = overlap;
          }
          for (std::list<PhysicalUser>::const_iterator it = 
                reading_users.begin(); it != reading_users.end(); it++)
          {
            FieldMask overlap = it->field_mask & update_mask;
            if (!overlap)
              continue;
            read_users.push_back(*it);
            read_users.back().field_mask = overlap;
          }
        }
        rez.serialize(remote_did);
        rez.serialize<size_t>(red_users.size());
        for (std::deque<PhysicalUser>::const_iterator it = red_users.begin();
              it != red_users.end(); it++)
        {
          rez.serialize(it->usage);
          rez.serialize(it->field_mask);
          rez.serialize(it->term_event);
          rez.serialize(it->child);
        }
        rez.serialize<size_t>(read_users.size());
        for (std::deque<PhysicalUser>::const_iterator it = read_users.begin();
              it != read_users.end(); it++)
        {
          rez.serialize(it->usage);
          rez.serialize(it->field_mask);
          rez.serialize(it->term_event);
          rez.serialize(it->child);
        }
      }
      runtime->send_reduction_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_updates(Deserializer &derez,
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_node = logical_node->column_source;
      // First unpack everything
      size_t num_red_users;
      derez.deserialize(num_red_users);
      std::vector<PhysicalUser> red_users(num_red_users);
      for (unsigned idx = 0; idx < num_red_users; idx++)
      {
        PhysicalUser &user = red_users[idx];
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        field_node->transform_field_mask(user.field_mask, source);
      }
      size_t num_read_users;
      derez.deserialize(num_read_users);
      std::vector<PhysicalUser> read_users(num_read_users);
      for (unsigned idx = 0; idx < num_read_users; idx++)
      {
        PhysicalUser &user = read_users[idx];
        derez.deserialize(user.usage);
        derez.deserialize(user.field_mask);
        derez.deserialize(user.term_event);
        derez.deserialize(user.child);
        field_node->transform_field_mask(user.field_mask, source);
      }
      // Now we can apply the updates
      {
        AutoLock v_lock(view_lock);
        reduction_users.insert(reduction_users.end(),
                               red_users.begin(), red_users.end());
        reading_users.insert(reading_users.end(),
                             read_users.begin(), read_users.end());
      }
      // Finally record that we are 
      Processor curr_proc = Machine::get_executing_processor();
      for (std::vector<PhysicalUser>::const_iterator it = 
            red_users.begin(); it != red_users.end(); it++)
      {
        defer_collect_user(it->term_event);
      }
      for (std::vector<PhysicalUser>::const_iterator it = 
            read_users.begin(); it != read_users.end(); it++)
      {
        defer_collect_user(it->term_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::handle_send_reduction_view(
        RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner_addr;
      derez.deserialize(owner_addr);
      DistributedID owner_did;
      derez.deserialize(owner_did);
      DistributedID manager_did;
      derez.deserialize(manager_did);

      ReductionManager *manager = 
        context->find_manager(manager_did)->as_reduction_manager();
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      ReductionView *result = legion_new<ReductionView>(context, did, 
                        owner_addr, owner_did, manager->region_node, manager);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->unpack_reduction_view(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::handle_send_back_reduction_view(
        RegionTreeForest *context, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      DistributedID send_did;
      derez.deserialize(send_did);
      AddressSpaceID send_addr;
      derez.deserialize(send_addr);
      
      ReductionManager *manager = 
        context->find_manager(manager_did)->as_reduction_manager();
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      ReductionView *result = legion_new<ReductionView>(context, did, 
                                    context->runtime->address_space, did,
                                    manager->region_node, manager);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      // Add the subscriber
      result->add_subscriber(send_addr, send_did);
      // Add a remote reference that is held by the person who sent us back
      result->add_remote_reference();
      // Now upack the view
      result->unpack_reduction_view(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::handle_send_update(RegionTreeForest *context,
                                                      Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      LogicalView *log_view = context->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(log_view != NULL);
      assert(log_view->is_reduction_view());
#endif
      ReductionView *view = log_view->as_reduction_view();
      view->process_updates(derez, source);
    }

    //--------------------------------------------------------------------------
    Memory ReductionView::get_location(void) const
    //--------------------------------------------------------------------------
    {
      return manager->memory;
    }

    //--------------------------------------------------------------------------
    ReductionOpID ReductionView::get_redop(void) const
    //--------------------------------------------------------------------------
    {
      return manager->redop;
    }

    /////////////////////////////////////////////////////////////
    // View Handle 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ViewHandle::ViewHandle(void)
      : view(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ViewHandle::ViewHandle(LogicalView *v)
      : view(v)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
        view->add_gc_reference();
    }

    //--------------------------------------------------------------------------
    ViewHandle::ViewHandle(const ViewHandle &rhs)
      : view(rhs.view)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
        view->add_gc_reference();
    }

    //--------------------------------------------------------------------------
    ViewHandle::~ViewHandle(void)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
      {
        if (view->remove_gc_reference())
        {
          if (view->is_reduction_view())
            legion_delete(view->as_reduction_view());
          else
          {
            InstanceView *inst_view = view->as_instance_view();
            if (inst_view->is_composite_view())
              legion_delete(inst_view->as_composite_view());
            else
              legion_delete(inst_view->as_materialized_view());
          }
        }
        view = NULL;
      }
    }

    //--------------------------------------------------------------------------
    ViewHandle& ViewHandle::operator=(const ViewHandle &rhs)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
      {
        if (view->remove_gc_reference())
        {
          if (view->is_reduction_view())
            legion_delete(view->as_reduction_view());
          else
          {
            InstanceView *inst_view = view->as_instance_view();
            if (inst_view->is_composite_view())
              legion_delete(inst_view->as_composite_view());
            else
              legion_delete(inst_view->as_materialized_view());
          }
        }
      }
      view = rhs.view;
      if (view != NULL)
        view->add_gc_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // MappingRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingRef::MappingRef(void)
      : view(NULL), needed_fields(FieldMask())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappingRef::MappingRef(LogicalView *v, const FieldMask &needed)
      : view(v), needed_fields(needed)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
        view->add_valid_reference();
    }

    //--------------------------------------------------------------------------
    MappingRef::MappingRef(const MappingRef &rhs)
      : view(rhs.view), needed_fields(rhs.needed_fields)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
        view->add_valid_reference();
    }

    //--------------------------------------------------------------------------
    MappingRef::~MappingRef(void)
    //--------------------------------------------------------------------------
    {
      if ((view != NULL) && view->remove_valid_reference())
      {
        if (view->is_reduction_view())
          legion_delete(view->as_reduction_view());
        else
        {
          InstanceView *inst_view = view->as_instance_view();
          if (inst_view->is_composite_view())
            legion_delete(inst_view->as_composite_view());
          else
            legion_delete(inst_view->as_materialized_view());
        }
      }
      view = NULL;
    }

    //--------------------------------------------------------------------------
    MappingRef& MappingRef::operator=(const MappingRef &rhs)
    //--------------------------------------------------------------------------
    {
      if ((view != NULL) && view->remove_valid_reference())
      {
        if (view->is_reduction_view())
          legion_delete(view->as_reduction_view());
        else
        {
          InstanceView *inst_view = view->as_instance_view();
          if (inst_view->is_composite_view())
            legion_delete(inst_view->as_composite_view());
          else
            legion_delete(inst_view->as_materialized_view());
        }
      }
      view = rhs.view;
      needed_fields = rhs.needed_fields;
      if (view != NULL)
        view->add_valid_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // InstanceRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(void)
      : ready_event(Event::NO_EVENT), 
        needed_lock(Reservation::NO_RESERVATION), handle(ViewHandle())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(Event ready, Reservation lock, const ViewHandle &h)
      : ready_event(ready), needed_lock(lock), handle(h)  
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Memory InstanceRef::get_memory(void) const
    //--------------------------------------------------------------------------
    {
      return handle.get_manager()->memory;
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
      InstanceRef::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      return handle.get_manager()->get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      InstanceRef::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return handle.get_manager()->get_field_accessor(fid);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::add_valid_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(has_ref());
#endif
      handle.get_view()->add_valid_reference();
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_valid_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(has_ref());
#endif
      handle.get_view()->remove_valid_reference();
    }

    //--------------------------------------------------------------------------
    void InstanceRef::pack_reference(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      bool has_reference = has_ref();
      rez.serialize(has_reference);
      if (has_reference)
      {
        rez.serialize(ready_event);
        rez.serialize(needed_lock);
        PhysicalManager *manager = handle.get_manager();
        std::set<PhysicalManager*> needed_managers;
        if (manager->is_reduction_manager())
        {
          ReductionManager *reduc_manager = manager->as_reduction_manager();
          // First send the tree infromation for the manager
          reduc_manager->region_node->row_source->
            send_node(target,true/*up*/,true/*down*/);
          reduc_manager->region_node->column_source->send_node(target);
          reduc_manager->region_node->send_node(target);
          DistributedID did = reduc_manager->
                                send_manager(target, needed_managers);
          rez.serialize(did);
        }
        else
        {
          InstanceManager *inst_manager = manager->as_instance_manager();
          // First send the tree infromation for the manager
          inst_manager->region_node->row_source->
            send_node(target,true/*up*/,true/*down*/);
          inst_manager->region_node->column_source->send_node(target);
          inst_manager->region_node->send_node(target);
          DistributedID did = inst_manager->
                                send_manager(target, needed_managers);
          rez.serialize(did);
        }
        // Now see if we need to send a remote reference
        if (manager->send_remote_reference(target))
          rez.serialize<bool>(true);
        else
          rez.serialize<bool>(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ InstanceRef InstanceRef::unpack_reference(Deserializer &derez,
                                                     RegionTreeForest *context,
                                                     unsigned depth)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      bool has_reference;
      derez.deserialize(has_reference);
      if (has_reference)
      {
        Event ready;
        derez.deserialize(ready);
        Reservation lock;
        derez.deserialize(lock);
        DistributedID did;
        derez.deserialize(did);
        PhysicalManager *manager = context->find_manager(did);
        if (manager->is_reduction_manager())
        {
          ReductionView *view = manager->as_reduction_manager()->create_view();
          bool add_remote_reference;
          derez.deserialize(add_remote_reference);
          if (add_remote_reference)
            manager->add_held_remote_reference();
          return InstanceRef(ready, lock, ViewHandle(view));
        }
        else
        {
          InstanceView *view = 
            manager->as_instance_manager()->create_top_view(depth);
          bool add_remote_reference;
          derez.deserialize(add_remote_reference);
          if (add_remote_reference)
            manager->add_held_remote_reference();
          return InstanceRef(ready, lock, ViewHandle(view));
        }
        
      }
      else
        return InstanceRef();
    } 

    /////////////////////////////////////////////////////////////
    // Legion Stack 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T, int MAX_SIZE, int INC_SIZE>
    LegionStack<T,MAX_SIZE,INC_SIZE>::LegionStack(void)
    //--------------------------------------------------------------------------
    {
      // Allocate the first entry
      ptr_buffer[0] = new T[INC_SIZE];
      buffer_size = 1;
      remaining = INC_SIZE;
    }

    //--------------------------------------------------------------------------
    template<typename T, int MAX_SIZE, int INC_SIZE>
    LegionStack<T,MAX_SIZE,INC_SIZE>::LegionStack(
                                    const LegionStack<T,MAX_SIZE,INC_SIZE> &rhs)
    //--------------------------------------------------------------------------
    {
      // Copy constructor should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T, int MAX_SIZE, int INC_SIZE>
    LegionStack<T,MAX_SIZE,INC_SIZE>::~LegionStack(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < buffer_size; idx++)
      {
        delete [] ptr_buffer[idx];
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, int MAX_SIZE, int INC_SIZE>
    LegionStack<T,MAX_SIZE,INC_SIZE>& LegionStack<T,MAX_SIZE,INC_SIZE>::
                          operator=(const LegionStack<T,MAX_SIZE,INC_SIZE> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, int MAX_SIZE, int INC_SIZE>
    T& LegionStack<T,MAX_SIZE,INC_SIZE>::operator[](unsigned int idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < MAX_SIZE);
#endif
      return ptr_buffer[idx/INC_SIZE][idx%INC_SIZE];
    }

    //--------------------------------------------------------------------------
    template<typename T, int MAX_SIZE, int INC_SIZE>
    void LegionStack<T,MAX_SIZE,INC_SIZE>::append(unsigned int append_count)
    //--------------------------------------------------------------------------
    {
      // Quick out if we have entires
      if (remaining >= append_count)
      {
        remaining -= append_count;
        return;
      }
      // If we make it here, we can subtract remaining from append count
      // to get the number we actually need to add
      append_count -= remaining;
      const int new_arrays = (append_count+INC_SIZE-1)/INC_SIZE;
#ifdef DEBUG_HIGH_LEVEL
      // If we fail this assertion, then we've run out of contexts
      assert((buffer_size+new_arrays) <= ((MAX_SIZE+INC_SIZE-1)/INC_SIZE));
#endif
      // Allocate new arrays
      for (unsigned idx = 0; idx < new_arrays; idx++)
        ptr_buffer[buffer_size+idx] = new T[INC_SIZE];
      remaining = append_count % INC_SIZE;
      buffer_size += new_arrays;
    }

    //--------------------------------------------------------------------------
    template<typename T, int MAX_SIZE, int INC_SIZE>
    size_t LegionStack<T,MAX_SIZE,INC_SIZE>::size(void) const
    //--------------------------------------------------------------------------
    {
      return ((buffer_size*INC_SIZE) +
                ((remaining == 0) ? 0 : (INC_SIZE-remaining)));
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

