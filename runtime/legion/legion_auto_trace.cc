/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#include "legion/legion_auto_trace.h"
#include "legion/legion_auto_trace.inl"

namespace Legion {
  namespace Internal {

    TraceHashHelper::TraceHashHelper() :
      // TODO (rohany): Does precise need to be true here?
      hasher(false /* precise */) {}

    Murmur3Hasher::Hash TraceHashHelper::hash(Operation* op) {
      Operation::OpKind kind = op->get_operation_kind();
      // First just hash the operation kind.
      hasher.hash(kind);
      switch (kind) {
        case Operation::OpKind::TASK_OP_KIND: {
          TaskOp* task = dynamic_cast<TaskOp*>(op);
          switch (task->get_task_kind()) {
            case TaskOp::TaskKind::INDIVIDUAL_TASK_KIND:
            case TaskOp::TaskKind::INDEX_TASK_KIND: {
              this->hash(task);
              break;
            }
            default: {
              std::cout << "Attempting to has unexpected TaskOp kind: "
                        << int(task->get_task_kind())
                        << std::endl;
              assert(false);
            }
          }
          break;
        }
        case Operation::OpKind::FILL_OP_KIND: {
          this->hash(dynamic_cast<FillOp*>(op));
          break;
        }
        case Operation::OpKind::FENCE_OP_KIND: {
          this->hash(dynamic_cast<FenceOp*>(op));
          break;
        }
        case Operation::OpKind::COPY_OP_KIND: {
          this->hash(dynamic_cast<CopyOp*>(op));
          break;
        }
        default: {
          std::cout << "Attempting to hash an unsupported operation kind:"
                    << std::string(Operation::get_string_rep(kind))
                    << std::endl;
          assert(false);
        }
      };
      return hasher.get_hash();
    }

    void TraceHashHelper::hash(TaskOp* op) {
      hasher.hash(op->task_id);
      // Choosing to skip the fields indexes, futures, grants, wait_barriers, arrive_barriers, ...
      for (std::vector<RegionRequirement>::const_iterator it = op->regions.begin();
           it != op->regions.end(); it++) {
        this->hash(*it);
      }
      assert(op->output_regions.size() == 0);
      hasher.hash<bool>(op->is_index_space);
      if (op->is_index_space) {
        hasher.hash<bool>(op->concurrent_task);
        hasher.hash<bool>(op->must_epoch_task);
        hasher.hash(op->index_domain);
      }
    }

    void TraceHashHelper::hash(FillOp* op) {
      this->hash(op->requirement);
      // Choosing to skip the fields grants, wait_barriers, arrive_barriers.
      hasher.hash<bool>(op->is_index_space);
      if (op->is_index_space) {
        hasher.hash(op->index_domain);
      }
    }

    void TraceHashHelper::hash(FenceOp* op) {
      hasher.hash(op->get_fence_kind());
    }

    void TraceHashHelper::hash(CopyOp* op) {
      for (std::vector<RegionRequirement>::const_iterator it = op->src_requirements.begin();
           it != op->src_requirements.end(); it++) {
        this->hash(*it);
      }
      for (std::vector<RegionRequirement>::const_iterator it = op->dst_requirements.begin();
           it != op->dst_requirements.end(); it++) {
        this->hash(*it);
      }
      for (std::vector<RegionRequirement>::const_iterator it = op->src_indirect_requirements.begin();
           it != op->src_indirect_requirements.end(); it++) {
        this->hash(*it);
      }
      for (std::vector<RegionRequirement>::const_iterator it = op->dst_indirect_requirements.begin();
           it != op->dst_indirect_requirements.end(); it++) {
        this->hash(*it);
      }
      // Not including the fields grants, wait_barriers, arrive_barriers.
      hasher.hash<bool>(op->is_index_space);
      if (op->is_index_space) {
        hasher.hash(op->index_domain);
      }
    }

    void TraceHashHelper::hash(const RegionRequirement& req) {
      if (req.region.exists()) {
        hasher.hash<bool>(true); // is_reg
        this->hash(req.region);
      } else {
        hasher.hash<bool>(false); // is_reg
        hasher.hash(req.partition.get_index_partition().get_id());
        hasher.hash(req.partition.get_field_space().get_id());
        hasher.hash(req.partition.get_tree_id());
      }
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
           it != req.privilege_fields.end(); it++) {
        hasher.hash(*it);
      }
      for (std::vector<FieldID>::const_iterator it = req.instance_fields.begin();
           it != req.instance_fields.end(); it++) {
        hasher.hash(*it);
      }
      hasher.hash(req.privilege);
      hasher.hash(req.prop);
      this->hash(req.parent);
      hasher.hash(req.redop);
      // Excluding the fields: tag and flags.
      hasher.hash(req.handle_type);
      hasher.hash(req.projection);
    }

    void TraceHashHelper::hash(const LogicalRegion& region) {
      hasher.hash(region.get_index_space().get_id());
      hasher.hash(region.get_field_space().get_id());
      hasher.hash(region.get_tree_id());
    }

    bool is_operation_traceable(Operation* op) {
      MemoizableOp* memo = op->get_memoizable();
      if (memo == nullptr) {
        return false;
      }
      switch (memo->get_operation_kind()) {
        case Operation::OpKind::TASK_OP_KIND: {
          TaskOp* task = dynamic_cast<TaskOp*>(memo);
          // Tasks with output regions cannot be traced.
          if (task->output_regions.size() > 0) {
            return false;
          }
          return true;
          break;
        }
        default:
          return true;
      }
    }
  };
};
