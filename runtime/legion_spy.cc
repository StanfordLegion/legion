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
#include "region_tree.h"
#include "legion_spy.h"
#include "runtime.h"

namespace LegionRuntime {
  namespace HighLevel {

    //--------------------------------------------------------------------------
    TreeStateLogger::TreeStateLogger(void)
      : verbose(false), logical_only(false), physical_only(false),
        tree_state_log(NULL), depth(0), logger_lock(Reservation::NO_RESERVATION)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TreeStateLogger::TreeStateLogger(AddressSpaceID sid, bool verb,
        bool log_only, bool phy_only)
      : verbose(verb), logical_only(log_only), physical_only(phy_only),
        depth(0), logger_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      char file_name[100];
      sprintf(file_name,"region_tree_state_log_%x.log",sid);
      tree_state_log = fopen(file_name,"w");
      assert(tree_state_log != NULL);
      log("");
      log("Region Tree State Logger for Address Space %x",sid);
      log("");
    }
    
    //--------------------------------------------------------------------------
    TreeStateLogger::~TreeStateLogger(void)
    //--------------------------------------------------------------------------
    {
      if (tree_state_log != NULL)
        fclose(tree_state_log);
      else
        fflush(stdout);
      tree_state_log = NULL;
      if (logger_lock.exists())
        logger_lock.destroy_reservation();
      logger_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    void TreeStateLogger::log(const char *fmt, ...)
    //--------------------------------------------------------------------------
    {
      va_list args;
      va_start(args, fmt);
      println(fmt, args);
      va_end(args);
    }

    //--------------------------------------------------------------------------
    void TreeStateLogger::down(void)
    //--------------------------------------------------------------------------
    {
      depth++;
    }

    //--------------------------------------------------------------------------
    void TreeStateLogger::up(void)
    //--------------------------------------------------------------------------
    {
      assert(depth > 0);
      depth--;
    }

    //--------------------------------------------------------------------------
    void TreeStateLogger::start_block(const char *fmt, ...)
    //--------------------------------------------------------------------------
    {
      Event lock_event = logger_lock.acquire(0, true/*exclusive*/);
      lock_event.wait(true/*block*/);
      va_list args;
      va_start(args, fmt);
      vsnprintf(block_buffer, 127, fmt, args);
      va_end(args);
      char temp_buffer[128+32];
      sprintf(temp_buffer,"BEGIN: ");
      strcat(temp_buffer,block_buffer);
      log("");
      log("//////////////////////////////////////////////////////////////////");
      log(temp_buffer);
      log("/////////");
      log("");
    }
    
    //--------------------------------------------------------------------------
    void TreeStateLogger::finish_block(void)
    //--------------------------------------------------------------------------
    {
      char temp_buffer[128+32];
      sprintf(temp_buffer,"END: ");
      strcat(temp_buffer,block_buffer);
      log("");
      log("/////////");
      log(temp_buffer);
      log("//////////////////////////////////////////////////////////////////");
      log("");
      logger_lock.release();
    }

    //--------------------------------------------------------------------------
    /*static*/ void TreeStateLogger::capture_state(Runtime *rt, 
                  const RegionRequirement *req, unsigned idx, 
                  const char *task_name, long long uid, 
                  RegionTreeNode *node, ContextID ctx, bool before, 
                  bool pre_map, bool closing, bool logical, 
                  FieldMask capture_mask, FieldMask working_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (Runtime::logging_region_tree_state)
      {
        TreeStateLogger *logger = rt->get_tree_state_logger();
        assert(logger != NULL);
        if (logical && logger->physical_only)
          return;
        if (!logical && logger->logical_only)
          return;
        char *mask_string = working_mask.to_string();
        if (before)
        {
          if (logical)
          {
            if (req->handle_type == SINGULAR)
              logger->start_block("BEFORE ANALYZING REGION (%x,%d,%d) index "
                              "%d of %s (UID %lld) in context %d mask %s",
                    req->region.index_space.id, req->region.field_space.id, 
                    req->region.tree_id, idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("BEFORE ANALYZING PARTITION (%d,%d,%d) "
                        "index %d of %s (UID %lld) in context %d mask %s",
                req->partition.index_partition, req->partition.field_space.id, 
                req->partition.tree_id, idx, task_name, uid, ctx, mask_string);
          }
          else if (pre_map)
          {
            if (req->handle_type == SINGULAR)
              logger->start_block("BEFORE PRE-MAPPING REGION (%x,%d,%d) index "
                              "%d of %s (UID %lld) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, 
                  req->region.tree_id, idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("BEFORE PRE-MAPPING PARTITION (%d,%d,%d) "
                      "index %d of %s (UID %lld) in context %d mask %s",
                req->partition.index_partition, req->partition.field_space.id, 
                req->partition.tree_id, idx, task_name, uid, ctx, mask_string);
          }
          else
          {
            assert(req->handle_type == SINGULAR);
            if (closing)
              logger->start_block("BEFORE CLOSING REGION (%x,%d,%d) index %d "
                                "of %s (UID %lld) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, 
                  req->region.tree_id, idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("BEFORE MAPPING REGION (%x,%d,%d) index %d "
                                "of %s (UID %lld) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, 
                  req->region.tree_id, idx, task_name, uid, ctx, mask_string);
          }
        }
        else
        {
          if (logical)
          {
            if (req->handle_type == SINGULAR)
              logger->start_block("AFTER ANALYZING REGION (%x,%d,%d) index "
                              "%d of %s (UID %lld) in context %d mask %s",
                    req->region.index_space.id, req->region.field_space.id, 
                    req->region.tree_id, idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("AFTER ANALYZING PARTITION (%d,%d,%d) "
                        "index %d of %s (UID %lld) in context %d mask %s",
                req->partition.index_partition, req->partition.field_space.id, 
                req->partition.tree_id, idx, task_name, uid, ctx, mask_string);
          }
          else if (pre_map)
          {
            if (req->handle_type == SINGULAR)
              logger->start_block("AFTER PRE-MAPPING REGION (%x,%d,%d) index "
                            "%d of %s (UID %lld) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, 
                  req->region.tree_id, idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("AFTER PRE-MAPPING PARTITION (%d,%d,%d) "
                      "index %d of %s (UID %lld) in context %d mask %s",
                  req->partition.index_partition, req->partition.field_space.id, 
                  req->partition.tree_id, idx, task_name, uid, ctx, mask_string);
          }
          else
          {
            assert(req->handle_type == SINGULAR);
            if (closing)
              logger->start_block("AFTER CLOSING REGION (%x,%d,%d) index %d "
                                "of %s (UID %lld) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, 
                  req->region.tree_id, idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("AFTER MAPPING REGION (%x,%d,%d) index %d "
                                "of %s (UID %lld) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, 
                  req->region.tree_id, idx, task_name, uid, ctx, mask_string);
          }
        }
        free(mask_string);

        if (logical)
          node->print_logical_context(ctx, logger, capture_mask);
        else
          node->print_physical_context(ctx, logger, capture_mask);

        logger->finish_block();
      }
#endif
    }

    //--------------------------------------------------------------------------
    void TreeStateLogger::println(const char *fmt, va_list args)
    //--------------------------------------------------------------------------
    {
      if (tree_state_log != NULL)
      {
        for (unsigned idx = 0; idx < depth; idx++)
          fprintf(tree_state_log,"  ");
        vfprintf(tree_state_log,fmt,args);
        fprintf(tree_state_log,"\n");
        fflush(tree_state_log);
      }
      else
      {
        for (unsigned idx = 0; idx < depth; idx++)
          fprintf(stdout,"  ");
        vfprintf(stdout,fmt,args);
        fprintf(stdout,"\n");
        fflush(stdout);
      }
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

