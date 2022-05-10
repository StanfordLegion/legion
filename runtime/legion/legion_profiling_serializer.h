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
#ifndef __LEGION_PROFILING_SERIALIZER_H__
#define __LEGION_PROFILING_SERIALIZER_H__

#include <string>
#include <stdio.h>
#include "legion/legion_profiling.h"

#ifdef LEGION_USE_ZLIB
#include <zlib.h>
// lp_fopen expects filename to be a std::string
#define lp_fopen(filename, mode)      gzopen(filename.c_str(),mode)
#define lp_fwrite(f, data, num_bytes) gzwrite(f,data,num_bytes)
#define lp_fflush(f, mode)            gzflush(f,mode)
#define lp_fclose(f)                  gzclose(f)
#else
// lp_fopen expects filename to be a std::string
#define lp_fopen(filename, mode)      fopen(filename.c_str(),mode)
#define lp_fwrite(f, data, num_bytes) fwrite(data,num_bytes,1,f)
#define lp_fflush(f, mode)            fflush(f)
#define lp_fclose(f)                  fclose(f)
#endif

namespace Legion {
  namespace Internal { 
    class LegionProfSerializer {
    public:
      LegionProfSerializer() {};
      virtual ~LegionProfSerializer() {};

      virtual bool is_thread_safe(void) const = 0;
      // You must override the following functions in your implementation
      virtual void serialize(const LegionProfDesc::MapperCallDesc&) = 0;
      virtual void serialize(const LegionProfDesc::RuntimeCallDesc&) = 0;
      virtual void serialize(const LegionProfDesc::MetaDesc&) = 0;
      virtual void serialize(const LegionProfDesc::OpDesc&) = 0;
      virtual void serialize(const LegionProfDesc::MaxDimDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSpacePointDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSpaceRectDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSpaceEmptyDesc&) = 0;
      virtual void serialize(const LegionProfInstance::FieldDesc&) = 0;
      virtual void serialize(const LegionProfInstance::FieldSpaceDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexPartDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexPartitionDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSpaceDesc&) = 0;
      virtual void serialize(const LegionProfInstance::IndexSubSpaceDesc&) = 0;
      virtual void serialize(const LegionProfInstance::LogicalRegionDesc&) = 0;
      virtual void serialize(const LegionProfInstance::PhysicalInstRegionDesc&)
      = 0;
      virtual void serialize(const LegionProfInstance::PhysicalInstLayoutDesc&)
      = 0;
      virtual void serialize(const LegionProfInstance::PhysicalInstDimOrderDesc&)
      = 0;
      virtual void serialize(const LegionProfInstance::IndexSpaceSizeDesc&)
      = 0;
      virtual void serialize(const LegionProfInstance::TaskKind&) = 0;
      virtual void serialize(const LegionProfInstance::TaskVariant&) = 0;
      virtual void serialize(const LegionProfInstance::OperationInstance&) = 0;
      virtual void serialize(const LegionProfInstance::MultiTask&) = 0;
      virtual void serialize(const LegionProfInstance::SliceOwner&) = 0;
      virtual void serialize(const LegionProfInstance::WaitInfo,
                             const LegionProfInstance::TaskInfo&) = 0;
      virtual void serialize(const LegionProfInstance::WaitInfo,
                             const LegionProfInstance::GPUTaskInfo&) = 0;
      virtual void serialize(const LegionProfInstance::WaitInfo,
                             const LegionProfInstance::MetaInfo&) = 0;
      virtual void serialize(const LegionProfInstance::TaskInfo&) = 0;
      virtual void serialize(const LegionProfInstance::MetaInfo&) = 0;
      virtual void serialize(const LegionProfInstance::CopyInfo&) = 0;
      virtual void serialize(const LegionProfInstance::FillInfo&) = 0;
      virtual void serialize(const LegionProfInstance::InstCreateInfo&) = 0;
      virtual void serialize(const LegionProfInstance::InstUsageInfo&) = 0;
      virtual void serialize(const LegionProfInstance::InstTimelineInfo&) = 0;
      virtual void serialize(const LegionProfInstance::PartitionInfo&) = 0;
      virtual void serialize(const LegionProfInstance::MapperCallInfo&) = 0;
      virtual void serialize(const LegionProfInstance::RuntimeCallInfo&) = 0;
      virtual void serialize(const LegionProfInstance::GPUTaskInfo&) = 0;
      virtual void serialize(const LegionProfInstance::CopyInstInfo&,
                             const LegionProfInstance::CopyInfo&) = 0;
      virtual void serialize(const LegionProfInstance::ProcDesc&) = 0;
      virtual void serialize(const LegionProfInstance::MemDesc&) = 0;
      virtual void serialize(const LegionProfInstance::ProcMemDesc&) = 0;
#ifdef LEGION_PROF_SELF_PROFILE
      virtual void serialize(const LegionProfInstance::ProfTaskInfo&) = 0;
#endif
    };

    // This is the Internal Binary Format Serializer
    class LegionProfBinarySerializer: public LegionProfSerializer {
    public:
      LegionProfBinarySerializer(std::string filename);
      ~LegionProfBinarySerializer();

      void writePreamble();

      bool is_thread_safe(void) const { return false; }
      // Serialize Methods
      void serialize(const LegionProfDesc::MapperCallDesc&);
      void serialize(const LegionProfDesc::RuntimeCallDesc&);
      void serialize(const LegionProfDesc::MetaDesc&);
      void serialize(const LegionProfDesc::OpDesc&);
      void serialize(const LegionProfDesc::MaxDimDesc&);
      void serialize(const LegionProfInstance::IndexSpacePointDesc&);
      void serialize(const LegionProfInstance::IndexSpaceRectDesc&);
      void serialize(const LegionProfInstance::IndexSpaceEmptyDesc&);
      void serialize(const LegionProfInstance::FieldDesc&);
      void serialize(const LegionProfInstance::FieldSpaceDesc&);
      void serialize(const LegionProfInstance::IndexPartDesc&);
      void serialize(const LegionProfInstance::IndexPartitionDesc&);
      void serialize(const LegionProfInstance::IndexSpaceDesc&);
      void serialize(const LegionProfInstance::IndexSubSpaceDesc&);
      void serialize(const LegionProfInstance::LogicalRegionDesc&);
      void serialize(const LegionProfInstance::PhysicalInstRegionDesc&);
      void serialize(const LegionProfInstance::PhysicalInstLayoutDesc&);
      void serialize(const LegionProfInstance::PhysicalInstDimOrderDesc&);
      void serialize(const LegionProfInstance::IndexSpaceSizeDesc&);
      void serialize(const LegionProfInstance::TaskKind&);
      void serialize(const LegionProfInstance::TaskVariant&);
      void serialize(const LegionProfInstance::OperationInstance&);
      void serialize(const LegionProfInstance::MultiTask&);
      void serialize(const LegionProfInstance::SliceOwner&);
      void serialize(const LegionProfInstance::WaitInfo,
                     const LegionProfInstance::TaskInfo&);
      void serialize(const LegionProfInstance::WaitInfo,
                     const LegionProfInstance::GPUTaskInfo&);
      void serialize(const LegionProfInstance::WaitInfo,
                     const LegionProfInstance::MetaInfo&);
      void serialize(const LegionProfInstance::TaskInfo&);
      void serialize(const LegionProfInstance::MetaInfo&);
      void serialize(const LegionProfInstance::CopyInfo&);
      void serialize(const LegionProfInstance::FillInfo&);
      void serialize(const LegionProfInstance::InstCreateInfo&);
      void serialize(const LegionProfInstance::InstUsageInfo&);
      void serialize(const LegionProfInstance::InstTimelineInfo&);
      void serialize(const LegionProfInstance::PartitionInfo&);
      void serialize(const LegionProfInstance::MapperCallInfo&);
      void serialize(const LegionProfInstance::RuntimeCallInfo&);
      void serialize(const LegionProfInstance::GPUTaskInfo&);
      void serialize(const LegionProfInstance::CopyInstInfo&,
                     const LegionProfInstance::CopyInfo&);
      void serialize(const LegionProfInstance::ProcDesc&);
      void serialize(const LegionProfInstance::MemDesc&);
      void serialize(const LegionProfInstance::ProcMemDesc&);
#ifdef LEGION_PROF_SELF_PROFILE
      void serialize(const LegionProfInstance::ProfTaskInfo&);
#endif
    private:
#ifdef LEGION_USE_ZLIB
      gzFile f;
#else
      FILE *f;
#endif
      enum LegionProfInstanceIDs {
        MESSAGE_DESC_ID,
        MAPPER_CALL_DESC_ID,
        RUNTIME_CALL_DESC_ID,
        META_DESC_ID,
        OP_DESC_ID,
        PROC_DESC_ID,
        MEM_DESC_ID,
	MAX_DIM_DESC_ID,
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
        INST_CREATE_INFO_ID,
        INST_USAGE_INFO_ID,
        INST_TIMELINE_INFO_ID,
        PARTITION_INFO_ID,
        MESSAGE_INFO_ID,
        MAPPER_CALL_INFO_ID,
        RUNTIME_CALL_INFO_ID,
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
        INDEX_SPACE_SIZE_ID,
        INDEX_INST_INFO_ID,
        COPY_INST_INFO_ID,
#ifdef LEGION_PROF_SELF_PROFILE
        PROFTASK_INFO_ID
#endif
      };
    };

    // This is the Old ASCII Serializer
    class LegionProfASCIISerializer: public LegionProfSerializer {
    public:
      LegionProfASCIISerializer();
      ~LegionProfASCIISerializer();

      bool is_thread_safe(void) const { return true; }
      // Serialize Methods
      void serialize(const LegionProfDesc::MapperCallDesc&);
      void serialize(const LegionProfDesc::RuntimeCallDesc&);
      void serialize(const LegionProfDesc::MetaDesc&);
      void serialize(const LegionProfDesc::OpDesc&);
      void serialize(const LegionProfDesc::MaxDimDesc&);
      void serialize(const LegionProfInstance::IndexSpacePointDesc&);
      void serialize(const LegionProfInstance::IndexSpaceRectDesc&);
      void serialize(const LegionProfInstance::IndexSpaceEmptyDesc&);
      void serialize(const LegionProfInstance::FieldDesc&);
      void serialize(const LegionProfInstance::FieldSpaceDesc&);
      void serialize(const LegionProfInstance::IndexPartDesc&);
      void serialize(const LegionProfInstance::IndexPartitionDesc&);
      void serialize(const LegionProfInstance::IndexSpaceDesc&);
      void serialize(const LegionProfInstance::IndexSubSpaceDesc&);
      void serialize(const LegionProfInstance::LogicalRegionDesc&);
      void serialize(const LegionProfInstance::PhysicalInstRegionDesc&);
      void serialize(const LegionProfInstance::PhysicalInstLayoutDesc&);
      void serialize(const LegionProfInstance::PhysicalInstDimOrderDesc&);
      void serialize(const LegionProfInstance::IndexSpaceSizeDesc&);
      void serialize(const LegionProfInstance::TaskKind&);
      void serialize(const LegionProfInstance::TaskVariant&);
      void serialize(const LegionProfInstance::OperationInstance&);
      void serialize(const LegionProfInstance::MultiTask&);
      void serialize(const LegionProfInstance::SliceOwner&);
      void serialize(const LegionProfInstance::WaitInfo,
                     const LegionProfInstance::TaskInfo&);
      void serialize(const LegionProfInstance::WaitInfo,
                     const LegionProfInstance::GPUTaskInfo&);
      void serialize(const LegionProfInstance::WaitInfo,
                     const LegionProfInstance::MetaInfo&);
      void serialize(const LegionProfInstance::TaskInfo&);
      void serialize(const LegionProfInstance::MetaInfo&);
      void serialize(const LegionProfInstance::CopyInfo&);
      void serialize(const LegionProfInstance::FillInfo&);
      void serialize(const LegionProfInstance::InstCreateInfo&);
      void serialize(const LegionProfInstance::InstUsageInfo&);
      void serialize(const LegionProfInstance::InstTimelineInfo&);
      void serialize(const LegionProfInstance::PartitionInfo&);
      void serialize(const LegionProfInstance::MapperCallInfo&);
      void serialize(const LegionProfInstance::RuntimeCallInfo&);
      void serialize(const LegionProfInstance::GPUTaskInfo&);
      void serialize(const LegionProfInstance::CopyInstInfo&,
                     const LegionProfInstance::CopyInfo&);
      void serialize(const LegionProfInstance::ProcDesc&);
      void serialize(const LegionProfInstance::MemDesc&);
      void serialize(const LegionProfInstance::ProcMemDesc&);
#ifdef LEGION_PROF_SELF_PROFILE
      void serialize(const LegionProfInstance::ProfTaskInfo&);
#endif
    };
  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_PROFILING_SERIALIZER_H__
