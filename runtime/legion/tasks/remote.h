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

#ifndef __LEGION_REMOTE_TASK_H__
#define __LEGION_REMOTE_TASK_H__

#include "legion/operations/remote.h"
#include "legion/tasks/task.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RemoteTaskOp
     * This is a remote copy of a TaskOp to be used
     * for mapper calls and other operations
     */
    class RemoteTaskOp : public ExternalTask,
                         public RemoteOp {
    public:
      RemoteTaskOp(Operation* ptr, AddressSpaceID src);
      RemoteTaskOp(const RemoteTaskOp& rhs) = delete;
      virtual ~RemoteTaskOp(void);
    public:
      RemoteTaskOp& operator=(const RemoteTaskOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual int get_depth(void) const override;
      virtual bool has_parent_task(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
      virtual const char* get_task_name(void) const override;
      virtual Domain get_slice_domain(void) const override;
      virtual ShardID get_shard_id(void) const override;
      virtual size_t get_total_shards(void) const override;
      virtual DomainPoint get_shard_point(void) const override;
      virtual Domain get_shard_domain(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual ContextCoordinate get_task_tree_coordinate(void) const override
      {
        return ContextCoordinate(context_index, index_point);
      }
    public:
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REMOTE_TASK_H__
