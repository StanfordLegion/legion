/* Copyright 2015 Stanford University, NVIDIA Corporation
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
#include "legion_logging.h"
#include "runtime.h"

namespace LegionRuntime {
  namespace HighLevel {

    //--------------------------------------------------------------------------
    TreeStateLogger::TreeStateLogger(Processor local_proc)
      : depth(0)
    //--------------------------------------------------------------------------
    {
      char file_name[100];
      sprintf(file_name,"region_tree_state_log_%x.log",local_proc.id);
      tree_state_log = fopen(file_name,"w");
      assert(tree_state_log != NULL);
      log("");
      log("Region Tree State Logger for Processor %x",local_proc.id);
      log("");
    }
    
    //--------------------------------------------------------------------------
    TreeStateLogger::~TreeStateLogger(void)
    //--------------------------------------------------------------------------
    {
      assert(tree_state_log != NULL);
      fclose(tree_state_log);
      tree_state_log = NULL;
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
      va_list args;
      va_start(args, fmt);
      vsnprintf(block_buffer, 127, fmt, args);
      va_end(args);
      char temp_buffer[128+32];
      sprintf(temp_buffer,"BEGIN: ");
      strcat(temp_buffer,block_buffer);
      log("");
      log("/////////////////////////////////////////////////////////////////////");
      log(temp_buffer);
      log("/////////////////////////////////////////////////////////////////////");
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
      log("/////////////////////////////////////////////////////////////////////");
      log(temp_buffer);
      log("/////////////////////////////////////////////////////////////////////");
      log("");
    }

    //--------------------------------------------------------------------------
    /*static*/ void TreeStateLogger::capture_state(Runtime *rt, unsigned idx, 
                                const char *task_name, unsigned uid,
                                RegionNode *node, ContextID ctx, bool pack, bool send, 
                                FieldMask capture_mask, FieldMask working_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (Runtime::logging_region_tree_state)
      {
        char *field_mask = working_mask.to_string();
        TreeStateLogger *logger = rt->get_tree_state_logger();
        assert(logger != NULL);
        if (pack)
        {
          if (send)
            logger->start_block("PACK SEND of REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                node->handle.index_space.id, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, field_mask);
          else
            logger->start_block("PACK RETURN of REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                node->handle.index_space.id, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, field_mask);
        }
        else
        {
          if (send)
            logger->start_block("UNPACK SEND of REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                node->handle.index_space.id, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, field_mask);
          else
            logger->start_block("UNPACK RETURN of REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                node->handle.index_space.id, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, field_mask);
        }
        free(field_mask);

	assert(0 && "need to implement print_physical_context!");
        //node->print_physical_context(ctx, logger, capture_mask);

        logger->finish_block();
      }
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void TreeStateLogger::capture_state(Runtime *rt, unsigned idx, 
                                const char *task_name, unsigned uid,
                                PartitionNode *node, ContextID ctx, bool pack, bool send, 
                                FieldMask capture_mask, FieldMask working_mask)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_HIGH_LEVEL
      if (Runtime::logging_region_tree_state)
      {
        TreeStateLogger *logger = rt->get_tree_state_logger();
        assert(logger != NULL);
        char *mask_string = working_mask.to_string();
        if (pack)
        {
          if (send)
            logger->start_block("PACK SEND of PARTITION (%d,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                node->handle.index_partition, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, mask_string);
          else
            logger->start_block("PACK RETURN of PARTITION (%d,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                node->handle.index_partition, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, mask_string);
        }
        else
        {
          if (send)
            logger->start_block("UNPACK SEND of PARTITION (%d,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                node->handle.index_partition, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, mask_string);
          else
            logger->start_block("UNPACK RETURN of PARTITION (%d,%d,%d) index %d of task %s  (UID %d) in context %d mask %s",
                node->handle.index_partition, node->handle.field_space.id, node->handle.tree_id,
                idx, task_name, uid, ctx, mask_string);
        }
        free(mask_string);

        assert(0 && "need to implement print_physical_context!");
        //node->print_physical_context(ctx, logger, capture_mask);

        logger->finish_block();
      }
#endif     
    }

    //--------------------------------------------------------------------------
    /*static*/ void TreeStateLogger::capture_state(Runtime *rt, 
                  const RegionRequirement *req, unsigned idx, const char *task_name, 
                  unsigned uid, RegionNode *node, ContextID ctx, bool pre_map, 
                  bool sanitize, bool closing, 
                  FieldMask capture_mask, FieldMask working_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (Runtime::logging_region_tree_state)
      {
        TreeStateLogger *logger = rt->get_tree_state_logger();
        assert(logger != NULL);
        char *mask_string = working_mask.to_string();
        if (pre_map)
        {
          if (sanitize)
          {
            if (req->handle_type == SINGULAR)
              logger->start_block("BEFORE SANITIZING REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, req->region.tree_id,
                  idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("BEFORE SANITIZING PARTITION (%d,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->partition.index_partition, req->partition.field_space.id, req->partition.tree_id,
                  idx, task_name, uid, ctx, mask_string);
          }
          else
          {
            assert(req->handle_type == SINGULAR);
            if (closing)
              logger->start_block("BEFORE CLOSING REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, req->region.tree_id,
                  idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("BEFORE MAPPING REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, req->region.tree_id,
                  idx, task_name, uid, ctx, mask_string);
          }
        }
        else
        {
          if (sanitize)
          {
            if (req->handle_type == SINGULAR)
              logger->start_block("AFTER SANITIZING REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, req->region.tree_id,
                  idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("AFTER SANITIZING PARTITION (%d,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->partition.index_partition, req->partition.field_space.id, req->partition.tree_id,
                  idx, task_name, uid, ctx, mask_string);
          }
          else
          {
            assert(req->handle_type == SINGULAR);
            if (closing)
              logger->start_block("AFTER CLOSING REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, req->region.tree_id,
                  idx, task_name, uid, ctx, mask_string);
            else
              logger->start_block("AFTER MAPPING REGION (%x,%d,%d) index %d of task %s (UID %d) in context %d mask %s",
                  req->region.index_space.id, req->region.field_space.id, req->region.tree_id,
                  idx, task_name, uid, ctx, mask_string);
          }
        }
        free(mask_string);

        assert(0 && "need to implement print_physical_context!");
        //node->print_physical_context(ctx, logger, capture_mask);

        logger->finish_block();
      }
#endif
    }

    //--------------------------------------------------------------------------
    void TreeStateLogger::capture_state(Runtime *rt, LogicalRegion handle, 
                                        const char *task_name, unsigned uid,
                                        RegionNode *node, ContextID ctx, bool pack, 
                                        unsigned shift, FieldMask capture_mask, 
                                        FieldMask working_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (Runtime::logging_region_tree_state)
      {
        TreeStateLogger *logger = rt->get_tree_state_logger();
        assert(logger != NULL);
        char *mask_string = working_mask.to_string();
        if (pack)
          logger->start_block("PACK RETURN OF CREATED STATE for REGION (%x,%d,%d) of task %s (UID %d) in context %d mask %s",
              handle.index_space.id, handle.field_space.id, handle.tree_id, task_name, uid, ctx, mask_string);
        else
          logger->start_block("UNPACK RETURN OF CREATED STATE for REGION (%x,%d,%d) of task %s (UID %d) in context %d with shift %d mask %s",
              handle.index_space.id, handle.field_space.id, handle.tree_id, task_name, uid, ctx, shift, mask_string);
        free(mask_string);
        assert(0 && "need to implement print_physical_context!");
        //node->print_physical_context(ctx, logger, capture_mask);

        logger->finish_block();
      }
#endif
    }

    //--------------------------------------------------------------------------
    void TreeStateLogger::println(const char *fmt, va_list args)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < depth; idx++)
        fprintf(tree_state_log,"  ");
      vfprintf(tree_state_log,fmt,args);
      fprintf(tree_state_log,"\n");
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

