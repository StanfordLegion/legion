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

#ifndef __LEGION_SERIALIZER_H__
#define __LEGION_SERIALIZER_H__

#include "legion/tools/profiler.h"

#ifdef LEGION_USE_ZLIB
#include <zlib.h>
// lp_fopen expects filename to be a std::string
#define lp_fopen(filename, mode) gzopen(filename.c_str(), mode)
#define lp_fwrite(f, data, num_bytes) gzwrite(f, data, num_bytes)
#define lp_fflush(f, mode) gzflush(f, mode)
#define lp_fclose(f) gzclose(f)
#else
// lp_fopen expects filename to be a std::string
#define lp_fopen(filename, mode) fopen(filename.c_str(), mode)
#define lp_fwrite(f, data, num_bytes) fwrite(data, num_bytes, 1, f)
#define lp_fflush(f, mode) fflush(f)
#define lp_fclose(f) fclose(f)
#endif

namespace Legion {
  namespace Internal {

    class LegionProfSerializer {
    public:
      LegionProfSerializer() { };
      virtual ~LegionProfSerializer() { };

      // You must override the following functions in your implementation
      virtual void serialize(const LegionProfiler::MapperName&) = 0;
      virtual void serialize(const LegionProfiler::MapperCallDesc&) = 0;
      virtual void serialize(const LegionProfiler::RuntimeCallDesc&) = 0;
      virtual void serialize(const LegionProfiler::MetaDesc&) = 0;
      virtual void serialize(const LegionProfiler::OpDesc&) = 0;
      virtual void serialize(const LegionProfiler::MaxDimDesc&) = 0;
      virtual void serialize(const LegionProfiler::RuntimeConfig&) = 0;
      virtual void serialize(const LegionProfiler::MachineDesc&) = 0;
      virtual void serialize(const LegionProfiler::ZeroTime&) = 0;
      virtual void serialize(const LegionProfiler::CalibrationErr&) = 0;
      virtual void serialize(const LegionProfiler::Provenance&) = 0;
      virtual void serialize(
          const LegionProfInstance::IndexSpacePointDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSpaceRectDesc&) = 0;
      virtual void serialize(
          const LegionProfInstance::IndexSpaceEmptyDesc&) = 0;
      virtual void serialize(const LegionProfInstance::FieldDesc&) = 0;
      virtual void serialize(const LegionProfInstance::FieldSpaceDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexPartDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexPartitionDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSpaceDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSubSpaceDesc&) = 0;
      virtual void serialize(const LegionProfInstance::LogicalRegionDesc&) = 0;
      virtual void serialize(
          const LegionProfInstance::PhysicalInstRegionDesc&) = 0;
      virtual void serialize(
          const LegionProfInstance::PhysicalInstLayoutDesc&) = 0;
      virtual void serialize(
          const LegionProfInstance::PhysicalInstDimOrderDesc&) = 0;
      virtual void serialize(
          const LegionProfInstance::PhysicalInstanceSpaces&) = 0;
      virtual void serialize(
          const LegionProfInstance::PhysicalInstanceUsage&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSpaceSizeDesc&) = 0;
      virtual void serialize(const LegionProfiler::TaskKind&) = 0;
      virtual void serialize(const LegionProfiler::TaskVariant&) = 0;
      virtual void serialize(const LegionProfInstance::OperationInstance&) = 0;
      virtual void serialize(const LegionProfInstance::MultiTask&) = 0;
      virtual void serialize(const LegionProfInstance::SliceOwner&) = 0;
      virtual void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::TaskInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::GPUTaskInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::MetaInfo&) = 0;
      virtual void serialize(const LegionProfInstance::TaskInfo&, bool) = 0;
      virtual void serialize(const LegionProfInstance::MetaInfo&) = 0;
      virtual void serialize(const LegionProfInstance::MessageInfo&) = 0;
      virtual void serialize(const LegionProfInstance::CopyInfo&) = 0;
      virtual void serialize(const LegionProfInstance::FillInfo&) = 0;
      virtual void serialize(const LegionProfInstance::InstTimelineInfo&) = 0;
      virtual void serialize(const LegionProfInstance::PartitionInfo&) = 0;
      virtual void serialize(const LegionProfInstance::MapperCallInfo&) = 0;
      virtual void serialize(const LegionProfInstance::RuntimeCallInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::ApplicationCallInfo&) = 0;
      virtual void serialize(const LegionProfInstance::AsyncEffectInfo&) = 0;
      virtual void serialize(const LegionProfInstance::GPUTaskInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::CopyInstInfo&,
          const LegionProfInstance::CopyInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::FillInstInfo&,
          const LegionProfInstance::FillInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::PartInstInfo&,
          const LegionProfInstance::PartitionInfo&) = 0;
      virtual void serialize(const LegionProfiler::ProcDesc&) = 0;
      virtual void serialize(const LegionProfiler::MemDesc&) = 0;
      virtual void serialize(const LegionProfiler::ProcMemDesc&) = 0;
      virtual void serialize(const LegionProfiler::Backtrace&) = 0;
      virtual void serialize(const LegionProfInstance::EventWaitInfo&) = 0;
      virtual void serialize(const LegionProfInstance::EventMergerInfo&) = 0;
      virtual void serialize(const LegionProfInstance::EventTriggerInfo&) = 0;
      virtual void serialize(const LegionProfInstance::EventPoisonInfo&) = 0;
      virtual void serialize(const LegionProfInstance::BarrierArrivalInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::ReservationAcquireInfo&) = 0;
      virtual void serialize(const LegionProfInstance::InstanceReadyInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::InstanceRedistrictInfo&) = 0;
      virtual void serialize(
          const LegionProfInstance::CompletionQueueInfo&) = 0;
      virtual void serialize(const LegionProfInstance::MakeValidInfo&) = 0;
      virtual void serialize(const LegionProfInstance::FetchMetadataInfo&) = 0;
      virtual void serialize(const LegionProfInstance::ProfTaskInfo&) = 0;
    };

    // This is the Internal Binary Format Serializer
    class LegionProfBinarySerializer : public LegionProfSerializer {
    public:
      LegionProfBinarySerializer(std::string filename);
      ~LegionProfBinarySerializer();

      void writePreamble();

      // Serialize Methods
      void serialize(const LegionProfiler::MapperName&) override;
      void serialize(const LegionProfiler::MapperCallDesc&) override;
      void serialize(const LegionProfiler::RuntimeCallDesc&) override;
      void serialize(const LegionProfiler::MetaDesc&) override;
      void serialize(const LegionProfiler::OpDesc&) override;
      void serialize(const LegionProfiler::MaxDimDesc&) override;
      void serialize(const LegionProfiler::RuntimeConfig&) override;
      void serialize(const LegionProfiler::MachineDesc&) override;
      void serialize(const LegionProfiler::ZeroTime&) override;
      void serialize(const LegionProfiler::CalibrationErr&) override;
      void serialize(const LegionProfiler::Provenance&) override;
      void serialize(const LegionProfInstance::IndexSpacePointDesc&) override;
      void serialize(const LegionProfInstance::IndexSpaceRectDesc&) override;
      void serialize(const LegionProfInstance::IndexSpaceEmptyDesc&) override;
      void serialize(const LegionProfInstance::FieldDesc&) override;
      void serialize(const LegionProfInstance::FieldSpaceDesc&) override;
      void serialize(const LegionProfInstance::IndexPartDesc&) override;
      void serialize(const LegionProfInstance::IndexPartitionDesc&) override;
      void serialize(const LegionProfInstance::IndexSpaceDesc&) override;
      void serialize(const LegionProfInstance::IndexSubSpaceDesc&) override;
      void serialize(const LegionProfInstance::LogicalRegionDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstRegionDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstLayoutDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstDimOrderDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstanceSpaces&) override;
      void serialize(const LegionProfInstance::PhysicalInstanceUsage&) override;
      void serialize(const LegionProfInstance::IndexSpaceSizeDesc&) override;
      void serialize(const LegionProfiler::TaskKind&) override;
      void serialize(const LegionProfiler::TaskVariant&) override;
      void serialize(const LegionProfInstance::OperationInstance&) override;
      void serialize(const LegionProfInstance::MultiTask&) override;
      void serialize(const LegionProfInstance::SliceOwner&) override;
      void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::TaskInfo&) override;
      void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::GPUTaskInfo&) override;
      void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::MetaInfo&) override;
      void serialize(const LegionProfInstance::TaskInfo&, bool) override;
      void serialize(const LegionProfInstance::MetaInfo&) override;
      void serialize(const LegionProfInstance::MessageInfo&) override;
      void serialize(const LegionProfInstance::CopyInfo&) override;
      void serialize(const LegionProfInstance::FillInfo&) override;
      void serialize(const LegionProfInstance::InstTimelineInfo&) override;
      void serialize(const LegionProfInstance::PartitionInfo&) override;
      void serialize(const LegionProfInstance::MapperCallInfo&) override;
      void serialize(const LegionProfInstance::RuntimeCallInfo&) override;
      void serialize(const LegionProfInstance::ApplicationCallInfo&) override;
      void serialize(const LegionProfInstance::AsyncEffectInfo&) override;
      void serialize(const LegionProfInstance::GPUTaskInfo&) override;
      void serialize(
          const LegionProfInstance::CopyInstInfo&,
          const LegionProfInstance::CopyInfo&) override;
      void serialize(
          const LegionProfInstance::FillInstInfo&,
          const LegionProfInstance::FillInfo&) override;
      void serialize(
          const LegionProfInstance::PartInstInfo&,
          const LegionProfInstance::PartitionInfo&) override;
      void serialize(const LegionProfiler::ProcDesc&) override;
      void serialize(const LegionProfiler::MemDesc&) override;
      void serialize(const LegionProfiler::ProcMemDesc&) override;
      void serialize(const LegionProfiler::Backtrace&) override;
      void serialize(const LegionProfInstance::EventWaitInfo&) override;
      void serialize(const LegionProfInstance::EventMergerInfo&) override;
      void serialize(const LegionProfInstance::EventTriggerInfo&) override;
      void serialize(const LegionProfInstance::EventPoisonInfo&) override;
      void serialize(const LegionProfInstance::BarrierArrivalInfo&) override;
      void serialize(
          const LegionProfInstance::ReservationAcquireInfo&) override;
      void serialize(const LegionProfInstance::InstanceReadyInfo&) override;
      void serialize(
          const LegionProfInstance::InstanceRedistrictInfo&) override;
      void serialize(const LegionProfInstance::CompletionQueueInfo&) override;
      void serialize(const LegionProfInstance::MakeValidInfo&) override;
      void serialize(const LegionProfInstance::FetchMetadataInfo&) override;
      void serialize(const LegionProfInstance::ProfTaskInfo&) override;
    private:
#ifdef LEGION_USE_ZLIB
      gzFile f;
#else
      FILE* f;
#endif
      enum LegionProfInstanceIDs {
        MESSAGE_DESC_ID,
        MAPPER_NAME_ID,
        MAPPER_CALL_DESC_ID,
        RUNTIME_CALL_DESC_ID,
        META_DESC_ID,
        OP_DESC_ID,
        PROC_DESC_ID,
        MEM_DESC_ID,
        MAX_DIM_DESC_ID,
        RUNTIME_CONFIG_ID,
        MACHINE_DESC_ID,
        TASK_KIND_ID,
        TASK_VARIANT_ID,
        OPERATION_INSTANCE_ID,
        MULTI_TASK_ID,
        SLICE_OWNER_ID,
        TASK_WAIT_INFO_ID,
        META_WAIT_INFO_ID,
        TASK_INFO_ID,
        META_INFO_ID,
        COPY_INFO_ID,
        FILL_INFO_ID,
        INST_TIMELINE_INFO_ID,
        PARTITION_INFO_ID,
        MESSAGE_INFO_ID,
        MAPPER_CALL_INFO_ID,
        RUNTIME_CALL_INFO_ID,
        APPLICATION_CALL_INFO_ID,
        ASYNC_EFFECT_INFO_ID,
        IMPLICIT_TASK_INFO_ID,
        GPU_TASK_INFO_ID,
        PROC_MEM_DESC_ID,
        INDEX_SPACE_POINT_ID,
        INDEX_SPACE_RECT_ID,
        INDEX_SPACE_EMPTY_ID,
        FIELD_ID,
        FIELD_SPACE_ID,
        INDEX_PART_ID,
        INDEX_PARTITION_ID,
        INDEX_SPACE_ID,
        INDEX_SUBSPACE_ID,
        LOGICAL_REGION_ID,
        PHYSICAL_INST_REGION_ID,
        PHYSICAL_INST_LAYOUT_ID,
        PHYSICAL_INST_LAYOUT_DIM_ID,
        PHYSICAL_INST_SPACES_ID,
        PHYSICAL_INST_USAGE_ID,
        INDEX_SPACE_SIZE_ID,
        INDEX_INST_INFO_ID,
        COPY_INST_INFO_ID,
        FILL_INST_INFO_ID,
        PARTITION_INST_INFO_ID,
        BACKTRACE_DESC_ID,
        EVENT_WAIT_INFO_ID,
        EVENT_MERGER_INFO_ID,
        EVENT_TRIGGER_INFO_ID,
        EVENT_POISON_INFO_ID,
        BARRIER_ARRIVAL_INFO_ID,
        RESERVATION_ACQUIRE_INFO_ID,
        INSTANCE_READY_INFO_ID,
        INSTANCE_REDISTRICT_INFO_ID,
        COMPLETION_QUEUE_INFO_ID,
        PROFTASK_INFO_ID,
        ZERO_TIME_ID,
        CALIBRATION_ERR_ID,
        PROVENANCE_ID,
        MAKE_VALID_INFO_ID,
        FETCH_METADATA_INFO_ID,
      };
    };

    // This is the Old ASCII Serializer
    class LegionProfASCIISerializer : public LegionProfSerializer {
    public:
      LegionProfASCIISerializer();
      ~LegionProfASCIISerializer();

      // Serialize Methods
      void serialize(const LegionProfiler::MapperName&) override;
      void serialize(const LegionProfiler::MapperCallDesc&) override;
      void serialize(const LegionProfiler::RuntimeCallDesc&) override;
      void serialize(const LegionProfiler::MetaDesc&) override;
      void serialize(const LegionProfiler::OpDesc&) override;
      void serialize(const LegionProfiler::MaxDimDesc&) override;
      void serialize(const LegionProfiler::RuntimeConfig&) override;
      void serialize(const LegionProfiler::MachineDesc&) override;
      void serialize(const LegionProfiler::ZeroTime&) override;
      void serialize(const LegionProfiler::CalibrationErr&) override;
      void serialize(const LegionProfiler::Provenance&) override;
      void serialize(const LegionProfInstance::IndexSpacePointDesc&) override;
      void serialize(const LegionProfInstance::IndexSpaceRectDesc&) override;
      void serialize(const LegionProfInstance::IndexSpaceEmptyDesc&) override;
      void serialize(const LegionProfInstance::FieldDesc&) override;
      void serialize(const LegionProfInstance::FieldSpaceDesc&) override;
      void serialize(const LegionProfInstance::IndexPartDesc&) override;
      void serialize(const LegionProfInstance::IndexPartitionDesc&) override;
      void serialize(const LegionProfInstance::IndexSpaceDesc&) override;
      void serialize(const LegionProfInstance::IndexSubSpaceDesc&) override;
      void serialize(const LegionProfInstance::LogicalRegionDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstRegionDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstLayoutDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstDimOrderDesc&) override;
      void serialize(
          const LegionProfInstance::PhysicalInstanceSpaces&) override;
      void serialize(const LegionProfInstance::PhysicalInstanceUsage&) override;
      void serialize(const LegionProfInstance::IndexSpaceSizeDesc&) override;
      void serialize(const LegionProfiler::TaskKind&) override;
      void serialize(const LegionProfiler::TaskVariant&) override;
      void serialize(const LegionProfInstance::OperationInstance&) override;
      void serialize(const LegionProfInstance::MultiTask&) override;
      void serialize(const LegionProfInstance::SliceOwner&) override;
      void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::TaskInfo&) override;
      void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::GPUTaskInfo&) override;
      void serialize(
          const LegionProfInstance::WaitInfo,
          const LegionProfInstance::MetaInfo&) override;
      void serialize(const LegionProfInstance::TaskInfo&, bool) override;
      void serialize(const LegionProfInstance::MetaInfo&) override;
      void serialize(const LegionProfInstance::MessageInfo&) override;
      void serialize(const LegionProfInstance::CopyInfo&) override;
      void serialize(const LegionProfInstance::FillInfo&) override;
      void serialize(const LegionProfInstance::InstTimelineInfo&) override;
      void serialize(const LegionProfInstance::PartitionInfo&) override;
      void serialize(const LegionProfInstance::MapperCallInfo&) override;
      void serialize(const LegionProfInstance::RuntimeCallInfo&) override;
      void serialize(const LegionProfInstance::ApplicationCallInfo&) override;
      void serialize(const LegionProfInstance::AsyncEffectInfo&) override;
      void serialize(const LegionProfInstance::GPUTaskInfo&) override;
      void serialize(
          const LegionProfInstance::CopyInstInfo&,
          const LegionProfInstance::CopyInfo&) override;
      void serialize(
          const LegionProfInstance::FillInstInfo&,
          const LegionProfInstance::FillInfo&) override;
      void serialize(
          const LegionProfInstance::PartInstInfo&,
          const LegionProfInstance::PartitionInfo&) override;
      void serialize(const LegionProfiler::ProcDesc&) override;
      void serialize(const LegionProfiler::MemDesc&) override;
      void serialize(const LegionProfiler::ProcMemDesc&) override;
      void serialize(const LegionProfiler::Backtrace&) override;
      void serialize(const LegionProfInstance::EventWaitInfo&) override;
      void serialize(const LegionProfInstance::EventMergerInfo&) override;
      void serialize(const LegionProfInstance::EventTriggerInfo&) override;
      void serialize(const LegionProfInstance::EventPoisonInfo&) override;
      void serialize(const LegionProfInstance::BarrierArrivalInfo&) override;
      void serialize(
          const LegionProfInstance::ReservationAcquireInfo&) override;
      void serialize(const LegionProfInstance::InstanceReadyInfo&) override;
      void serialize(
          const LegionProfInstance::InstanceRedistrictInfo&) override;
      void serialize(const LegionProfInstance::CompletionQueueInfo&) override;
      void serialize(const LegionProfInstance::MakeValidInfo&) override;
      void serialize(const LegionProfInstance::FetchMetadataInfo&) override;
      void serialize(const LegionProfInstance::ProfTaskInfo&) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SERIALIZER_H__
