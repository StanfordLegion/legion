#include "legion/legion_profiling_serializer.h"
#include "legion/runtime.h"
//#include "legion/legion_ops.h"
//#include "legion/legion_tasks.h"

#include <sstream>
#include <string>

// http://stackoverflow.com/questions/3553296/c-sizeof-single-struct-member
#define member_size(type, member) sizeof(((type *)0)->member)

namespace Legion {
  namespace Internal {

    extern Realm::Logger log_prof;

    //--------------------------------------------------------------------------
    LegionProfBinarySerializer::LegionProfBinarySerializer(std::string filename)
    //--------------------------------------------------------------------------
    {
      f = lp_fopen(filename, "wb");
      if (!f)
        REPORT_LEGION_ERROR(ERROR_INVALID_PROFILER_FILE,
            "Unable to open legion logfile %s for writing!", filename.c_str())
      writePreamble();
    }

    // Every legion prof instance that you want to serialize must be written 
    // in the preamble. The preamble defines the format that we'll use for 
    // the serialization.
    //
    // Each line defines the format of a different legion prof instance. 
    // The format is as follows:
    //
    //  <InstanceName> {id:<id>(, <field>:<type>:<sizeof(type)>)*}
    //
    // This can easily be parsed using a regex parser
    //
    // The end of the preamble is indicated by an empty line. After this 
    // empty line, the file will contain binary data
    
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::writePreamble() 
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "FileType: BinaryLegionProf v: 1.0" << std::endl;

      std::string delim = ", ";

      ss << "MapperCallDesc {" 
         << "id:" << MAPPER_CALL_DESC_ID            << delim
         << "kind:unsigned:"     << sizeof(unsigned) << delim
         << "name:string:" << "-1"
         << "}" << std::endl;

      ss << "RuntimeCallDesc {" 
         << "id:" << RUNTIME_CALL_DESC_ID           << delim
         << "kind:unsigned:"     << sizeof(unsigned) << delim
         << "name:string:" << "-1"
         << "}" << std::endl;

      ss << "MetaDesc {" 
         << "id:" << META_DESC_ID                   << delim
         << "kind:unsigned:"     << sizeof(unsigned) << delim
         << "message:bool:"      << sizeof(bool) << delim
         << "ordered_vc:bool:"   << sizeof(bool) << delim
         << "name:string:" << "-1"
         << "}" << std::endl;

      ss << "OpDesc {" 
         << "id:" << OP_DESC_ID                   << delim
         << "kind:unsigned:"   << sizeof(unsigned) << delim
         << "name:string:" << "-1"
         << "}" << std::endl;

      ss << "ProcDesc {" 
         << "id:" << PROC_DESC_ID                 << delim
         << "proc_id:ProcID:" << sizeof(ProcID)   << delim
         << "kind:ProcKind:"  << sizeof(ProcKind)
         << "}" << std::endl;

      ss << "MaxDimDesc {"
         << "id:" << MAX_DIM_DESC_ID                 << delim
         << "max_dim:maxdim:" << sizeof(unsigned)
         << "}" << std::endl;

      ss << "MemDesc {" 
         << "id:" << MEM_DESC_ID                               << delim
         << "mem_id:MemID:"                << sizeof(MemID)    << delim
         << "kind:MemKind:"                << sizeof(MemKind)  << delim
         << "capacity:unsigned long long:" << sizeof(unsigned long long)
         << "}" << std::endl;

      ss << "ProcMDesc {"
         << "id:" << PROC_MEM_DESC_ID                          << delim
         << "proc_id:ProcID:"              << sizeof(ProcID)   << delim
         << "mem_id:MemID:"                << sizeof(MemID)    << delim
         << "bandwidth:unsigned:"          << sizeof(unsigned) << delim
         << "latency:unsigned:"            << sizeof(unsigned)
         << "}" << std::endl;

      ss << "IndexSpacePointDesc {"
         << "id:" << INDEX_SPACE_POINT_ID                    << delim
         << "unique_id:IDType:"          << sizeof(IDType)   << delim
         << "dim:unsigned:"              << sizeof(unsigned) << delim
	 << "rem:point:"                 << sizeof(unsigned long long)
         << "}" << std::endl;

      ss << "IndexSpaceRectDesc {"
         << "id:" << INDEX_SPACE_RECT_ID                       << delim
         << "unique_id:IDType:"            << sizeof(IDType)   << delim
         << "dim:unsigned:"               << sizeof(unsigned)  << delim
         << "rem:array:"                  << sizeof(unsigned long long)
         << "}" << std::endl;

      ss << "IndexSpaceEmptyDesc {"
         << "id:" << INDEX_SPACE_EMPTY_ID                      << delim
         << "unique_id:IDType:"           << sizeof(unsigned long long)
	 << "}" << std::endl;

      ss << "FieldDesc {"
         << "id:" << FIELD_ID                                  << delim
	 << "unique_id:UniqueID:"         << sizeof(UniqueID)  << delim
	 << "field_id:unsigned:"          << sizeof(unsigned)  << delim
	 << "size:unsigned long long:"    << sizeof(unsigned long long) << delim
	 << "name:string:"                << "-1"
         << "}" << std::endl;

      ss << "FieldSpaceDesc {"
         << "id:" << FIELD_SPACE_ID                             << delim
	 << "unique_id:UniqueID:"          << sizeof(UniqueID)  << delim
	 << "name:string:"                 << "-1"
         << "}" << std::endl;

      ss << "PartDesc {"
         << "id:" << INDEX_PART_ID                              << delim
	 << "unique_id:UniqueID:"          << sizeof(UniqueID)  << delim
	 << "name:string:"                 << "-1"
         << "}" << std::endl;

      ss << "IndexSpaceDesc {"
         << "id:" << INDEX_SPACE_ID                             << delim
	 << "unique_id:UniqueID:"          << sizeof(UniqueID)  << delim
	 << "name:string:"                 << "-1"
         << "}" << std::endl;

      ss << "IndexSubSpaceDesc {"
         << "id:" << INDEX_SUBSPACE_ID                          << delim
	 << "parent_id:UniqueID:"          << sizeof(UniqueID)  << delim
	 << "unique_id:UniqueID:"          << sizeof(UniqueID)
         << "}" << std::endl;

      ss << "IndexPartitionDesc {"
         << "id:" << INDEX_PARTITION_ID                         << delim
	 << "parent_id:UniqueID:"          << sizeof(UniqueID)  << delim
	 << "unique_id:UniqueID:"          << sizeof(UniqueID)  << delim
	 << "disjoint:bool:"               << sizeof(bool)      << delim
	 << "point0:unsigned long long:"    << sizeof(unsigned long long)
         << "}" << std::endl;

      ss << "IndexSpaceSizeDesc {"
         << "id:" << INDEX_SPACE_SIZE_ID                         << delim
         << "unique_id:UniqueID:"            << sizeof(UniqueID)  << delim
         << "dense_size:unsigned long long:" << sizeof(unsigned long long)
         << delim
         << "sparse_size:unsigned long long:" << sizeof(unsigned long long) <<
        delim
         << "is_sparse:bool:"               << sizeof(bool)
         << "}" << std::endl;

      ss << "LogicalRegionDesc {"
         << "id:" << LOGICAL_REGION_ID                          << delim
	 << "ispace_id:IDType:"            << sizeof(IDType)    << delim
	 << "fspace_id:unsigned:"          << sizeof(unsigned)  << delim
	 << "tree_id:unsigned:"            << sizeof(unsigned)  << delim
	 << "name:string:"                 << "-1"
         << "}" << std::endl;

      ss << "PhysicalInstRegionDesc {"
         << "id:" << PHYSICAL_INST_REGION_ID                      << delim
	 << "op_id:UniqueID:"              << sizeof(UniqueID)    << delim
	 << "inst_id:InstID:"              << sizeof(InstID)      << delim
	 << "ispace_id:IDType:"            << sizeof(IDType)      << delim
	 << "fspace_id:unsigned:"          << sizeof(unsigned)    << delim
	 << "tree_id:unsigned:"            << sizeof(unsigned)
         << "}" << std::endl;

      ss << "PhysicalInstLayoutDesc {"
         << "id:" << PHYSICAL_INST_LAYOUT_ID                  << delim
         << "op_id:UniqueID:"           << sizeof(UniqueID)   << delim
         << "inst_id:InstID:"           << sizeof(InstID)     << delim
         << "field_id:unsigned:"        << sizeof(unsigned)   << delim
         << "fspace_id:unsigned:"       << sizeof(unsigned)   << delim
         << "has_align:bool:"           << sizeof(bool)       << delim
         << "eqk:unsigned:"             << sizeof(unsigned)   << delim
         << "align_desc:unsigned:"      << sizeof(unsigned)
         << "}" << std::endl;

      ss << "PhysicalInstDimOrderDesc {"
         << "id:" << PHYSICAL_INST_LAYOUT_DIM_ID              << delim
         << "op_id:UniqueID:"           << sizeof(UniqueID)   << delim
         << "inst_id:InstID:"           << sizeof(InstID)     << delim
         << "dim:unsigned:"             << sizeof(unsigned)   << delim
         << "dim_kind:unsigned:"        << sizeof(unsigned)
         << "}" << std::endl;

      ss << "TaskKind {" 
         << "id:" << TASK_KIND_ID                 << delim
         << "task_id:TaskID:"   << sizeof(TaskID) << delim
         << "name:string:"      << "-1"           << delim
         << "overwrite:bool:"   << sizeof(bool) 
         << "}" << std::endl;

      ss << "TaskVariant {"
         << "id:" << TASK_VARIANT_ID                     << delim
         << "task_id:TaskID:"       << sizeof(TaskID)    << delim
         << "variant_id:VariantID:" << sizeof(VariantID) << delim
         << "name:string:"          << "-1"
         << "}" << std::endl;

      ss << "OperationInstance {"
         << "id:" << OPERATION_INSTANCE_ID        << delim
         << "op_id:UniqueID:" << sizeof(UniqueID) << delim
         << "parent_id:UniqueID:" << sizeof(UniqueID) << delim
         << "kind:unsigned:"  << sizeof(unsigned) << delim
         << "provenance:string:" << "-1"
         << "}" << std::endl;

      ss << "MultiTask {"
         << "id:" << MULTI_TASK_ID                << delim
         << "op_id:UniqueID:" << sizeof(UniqueID) << delim
         << "task_id:TaskID:" << sizeof(TaskID)
         << "}" << std::endl;

      ss << "SliceOwner {"
         << "id:" << SLICE_OWNER_ID                   << delim
         << "parent_id:UniqueID:" << sizeof(UniqueID) << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)
         << "}" << std::endl;

      ss << "TaskWaitInfo {"
         << "id:" << TASK_WAIT_INFO_ID                       << delim
         << "op_id:UniqueID:"         << sizeof(UniqueID)    << delim
         << "task_id:TaskID:"         << sizeof(TaskID)      << delim
         << "variant_id:VariantID:"   << sizeof(VariantID)   << delim
         << "wait_start:timestamp_t:" << sizeof(timestamp_t) << delim
         << "wait_ready:timestamp_t:" << sizeof(timestamp_t) << delim
         << "wait_end:timestamp_t:"   << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "MetaWaitInfo {"
         << "id:" << META_WAIT_INFO_ID                       << delim
         << "op_id:UniqueID:"         << sizeof(UniqueID)    << delim
         << "lg_id:unsigned:"         << sizeof(unsigned)    << delim
         << "wait_start:timestamp_t:" << sizeof(timestamp_t) << delim
         << "wait_ready:timestamp_t:" << sizeof(timestamp_t) << delim
         << "wait_end:timestamp_t:"   << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "TaskInfo {"
         << "id:" << TASK_INFO_ID                         << delim
         << "op_id:UniqueID:"      << sizeof(UniqueID)    << delim
         << "task_id:TaskID:"      << sizeof(TaskID)      << delim
         << "variant_id:VariantID:"<< sizeof(VariantID)   << delim
         << "proc_id:ProcID:"      << sizeof(ProcID)      << delim
         << "create:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "start:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"    << sizeof(timestamp_t)
#ifdef LEGION_PROF_PROVENANCE
         << delim
         << "provenance:"          << sizeof(LgEvent) << delim
         << "finish_event:"        << sizeof(LgEvent)
#endif
         << "}" << std::endl;

      ss << "GPUTaskInfo {"
         << "id:" << GPU_TASK_INFO_ID                       << delim
         << "op_id:UniqueID:"        << sizeof(UniqueID)    << delim
         << "task_id:TaskID:"        << sizeof(TaskID)      << delim
         << "variant_id:VariantID:"   << sizeof(VariantID)    << delim
         << "proc_id:ProcID:"        << sizeof(ProcID)      << delim
         << "create:timestamp_t:"    << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"     << sizeof(timestamp_t) << delim
         << "start:timestamp_t:"     << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"      << sizeof(timestamp_t) << delim
         << "gpu_start:timestamp_t:" << sizeof(timestamp_t) << delim
         << "gpu_stop:timestamp_t:"  << sizeof(timestamp_t)
#ifdef LEGION_PROF_PROVENANCE
         << delim
         << "provenance:"            << sizeof(LgEvent) << delim
         << "finish_event:"          << sizeof(LgEvent)
#endif
         << "}" << std::endl;

      ss << "MetaInfo {"
         << "id:" << META_INFO_ID                         << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)     << delim
         << "lg_id:unsigned:"     << sizeof(unsigned)     << delim
         << "proc_id:ProcID:"     << sizeof(ProcID)       << delim
         << "create:timestamp_t:" << sizeof(timestamp_t)  << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t)  << delim
         << "start:timestamp_t:"  << sizeof(timestamp_t)  << delim
         << "stop:timestamp_t:"   << sizeof(timestamp_t)
#ifdef LEGION_PROF_PROVENANCE
         << delim
         << "provenance:"         << sizeof(LgEvent) << delim
         << "finish_event:"       << sizeof(LgEvent)
#endif
         << "}" << std::endl;

      ss << "CopyInfo {"
         << "id:" << COPY_INFO_ID                                    << delim
         << "op_id:UniqueID:"          << sizeof(UniqueID)           << delim
         << "src:MemID:"               << sizeof(MemID)              << delim
         << "dst:MemID:"               << sizeof(MemID)              << delim
         << "size:unsigned long long:" << sizeof(unsigned long long) << delim
         << "create:timestamp_t:"      << sizeof(timestamp_t)        << delim
         << "ready:timestamp_t:"       << sizeof(timestamp_t)        << delim
         << "start:timestamp_t:"       << sizeof(timestamp_t)        << delim
         << "stop:timestamp_t:"        << sizeof(timestamp_t)        << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent) << delim
         << "num_requests:unsigned:" << sizeof(unsigned)
#ifdef LEGION_PROF_PROVENANCE
         << delim
         << "provenance:"              << sizeof(LgEvent)
#endif
         << "}" << std::endl;

      ss << "CopyInstInfo {"
         << "id:" << COPY_INST_INFO_ID                           << delim
         << "op_id:UniqueID:"          << sizeof(UniqueID)       << delim
         << "src_inst:InstID:"         << sizeof(InstID)         << delim
         << "dst_inst:InstID:"         << sizeof(InstID)         << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)      << delim
         << "num_fields:unsigned:"       << sizeof(unsigned)     << delim
         << "request_type:unsigned:"     << sizeof(unsigned)     << delim
         << "num_hops:unsigned:"         << sizeof(unsigned)
         << "}" << std::endl;

      ss << "FillInfo {"
         << "id:" << FILL_INFO_ID                        << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)    << delim
         << "dst:MemID:"          << sizeof(MemID)       << delim
         << "create:timestamp_t:" << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "start:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"   << sizeof(timestamp_t)
#ifdef LEGION_PROF_PROVENANCE
         << delim
         << "provenance:"         << sizeof(LgEvent)
#endif
         << "}" << std::endl;

      ss << "InstCreateInfo {"
         << "id:" << INST_CREATE_INFO_ID                 << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)    << delim
         << "inst_id:InstID:"     << sizeof(InstID)      << delim
         << "create:timestamp_t:" << sizeof(timestamp_t)
#ifdef LEGION_PROF_PROVENANCE
         << delim
         << "provenance:"         << sizeof(LgEvent)
#endif
         << "}" << std::endl;

      ss << "InstUsageInfo {"
         << "id:" << INST_USAGE_INFO_ID                    << delim
         << "op_id:UniqueID:"          << sizeof(UniqueID) << delim
         << "inst_id:InstID:"          << sizeof(InstID)   << delim
         << "mem_id:MemID:"            << sizeof(MemID)    << delim
         << "size:unsigned long long:" << sizeof(unsigned long long)
         << "}" << std::endl;

      ss << "InstTimelineInfo {"
         << "id:" << INST_TIMELINE_INFO_ID                << delim
         << "op_id:UniqueID:"      << sizeof(UniqueID)    << delim
         << "inst_id:InstID:"      << sizeof(InstID)      << delim
         << "create:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "destroy:timestamp_t:" << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "PartitionInfo {"
         << "id:" << PARTITION_INFO_ID                          << delim
         << "op_id:UniqueID:"         << sizeof(UniqueID)       << delim
         << "part_op:DepPartOpKind:"  << sizeof(DepPartOpKind)  << delim
         << "create:timestamp_t:"     << sizeof(timestamp_t)    << delim
         << "ready:timestamp_t:"      << sizeof(timestamp_t)    << delim
         << "start:timestamp_t:"      << sizeof(timestamp_t)    << delim
         << "stop:timestamp_t:"       << sizeof(timestamp_t)
#ifdef LEGION_PROF_PROVENANCE
         << delim
         << "provenance:"             << sizeof(LgEvent)
#endif
         << "}" << std::endl;

      ss << "MapperCallInfo {"
         << "id:" << MAPPER_CALL_INFO_ID                          << delim
         << "kind:MappingCallKind:" << sizeof(MappingCallKind)    << delim
         << "op_id:UniqueID:"       << sizeof(UniqueID)           << delim
         << "start:timestamp_t:"    << sizeof(timestamp_t)        << delim
         << "stop:timestamp_t:"     << sizeof(timestamp_t)        << delim
         << "proc_id:ProcID:"       << sizeof(ProcID)
         << "}" << std::endl;

      ss << "RuntimeCallInfo {"
         << "id:" << RUNTIME_CALL_INFO_ID                      << delim
         << "kind:RuntimeCallKind:" << sizeof(RuntimeCallKind) << delim
         << "start:timestamp_t:"    << sizeof(timestamp_t)     << delim
         << "stop:timestamp_t:"     << sizeof(timestamp_t)     << delim
         << "proc_id:ProcID:"       << sizeof(ProcID)
         << "}" << std::endl;

#ifdef LEGION_PROF_SELF_PROFILE
      ss << "ProfTaskInfo {"
         << "id:" << PROFTASK_INFO_ID                        << delim
         << "proc_id:ProcID:"         << sizeof(ProcID)      << delim
         << "op_id:UniqueID:"         << sizeof(UniqueID)    << delim
         << "start:timestamp_t:"      << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"       << sizeof(timestamp_t)
         << "}" << std::endl;
#endif

      // An empty line indicates the end of the preamble.
      ss << std::endl;
      std::string preamble = ss.str();

      lp_fwrite(f, preamble.c_str(), strlen(preamble.c_str()));
    }


    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                         const LegionProfDesc::MapperCallDesc &mapper_call_desc)
    //--------------------------------------------------------------------------
    {
      int ID = MAPPER_CALL_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(mapper_call_desc.kind), 
                sizeof(mapper_call_desc.kind));
      lp_fwrite(f, mapper_call_desc.name, strlen(mapper_call_desc.name) + 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                       const LegionProfDesc::RuntimeCallDesc &runtime_call_desc)
    //--------------------------------------------------------------------------
    {
      int ID = RUNTIME_CALL_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(runtime_call_desc.kind), 
                sizeof(runtime_call_desc.kind));
      lp_fwrite(f, runtime_call_desc.name, strlen(runtime_call_desc.name) + 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                      const LegionProfDesc::MetaDesc& meta_desc)
    //--------------------------------------------------------------------------
    {
      int ID = META_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(meta_desc.kind), sizeof(meta_desc.kind));
      lp_fwrite(f, (char*)&(meta_desc.message), sizeof(meta_desc.message));
      lp_fwrite(f, (char*)&(meta_desc.ordered_vc),sizeof(meta_desc.ordered_vc));
      lp_fwrite(f, meta_desc.name, strlen(meta_desc.name) + 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                          const LegionProfDesc::OpDesc& op_desc)
    //--------------------------------------------------------------------------
    {
      int ID = OP_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(op_desc.kind), sizeof(op_desc.kind));
      lp_fwrite(f, op_desc.name, strlen(op_desc.name) + 1);
    }


    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                      const LegionProfDesc::MaxDimDesc
				      &max_dim_desc)
    //--------------------------------------------------------------------------
    {
      int ID = MAX_DIM_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(max_dim_desc.max_dim),
		sizeof(max_dim_desc.max_dim));

    }
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
               const LegionProfInstance::IndexSpacePointDesc &ispace_point_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_SPACE_POINT_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(ispace_point_desc.unique_id), 
                sizeof(ispace_point_desc.unique_id));
      lp_fwrite(f, (char*) &(ispace_point_desc.dim), 
                sizeof(ispace_point_desc.dim));
#define DIMFUNC(DIM) \
      lp_fwrite(f, (char*) &(ispace_point_desc.points[DIM-1]), \
                sizeof(ispace_point_desc.points[DIM-1]));
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                 const LegionProfInstance::IndexSpaceRectDesc &ispace_rect_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_SPACE_RECT_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(ispace_rect_desc.unique_id), 
                sizeof(ispace_rect_desc.unique_id));
      lp_fwrite(f, (char*) &(ispace_rect_desc.dim),
                sizeof(ispace_rect_desc.dim));
#define DIMFUNC(DIM) \
      lp_fwrite(f, (char*) &(ispace_rect_desc.rect_lo[DIM-1]), \
                sizeof(ispace_rect_desc.rect_lo[DIM-1]));
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
#define DIMFUNC(DIM) \
      lp_fwrite(f, (char*) &(ispace_rect_desc.rect_hi[DIM-1]), \
                sizeof(ispace_rect_desc.rect_hi[DIM-1]));
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
               const LegionProfInstance::IndexSpaceEmptyDesc &ispace_empty_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_SPACE_EMPTY_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(ispace_empty_desc.unique_id), 
                sizeof(ispace_empty_desc.unique_id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                const LegionProfInstance::FieldDesc &field_desc)
    //--------------------------------------------------------------------------
    {
      int ID = FIELD_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(field_desc.unique_id), 
                sizeof(field_desc.unique_id));
      lp_fwrite(f, (char*) &(field_desc.field_id), sizeof(field_desc.field_id));
      lp_fwrite(f, (char*) &(field_desc.size), sizeof(field_desc.size));
      lp_fwrite(f, field_desc.name, strlen(field_desc.name)+1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                     const LegionProfInstance::FieldSpaceDesc &field_space_desc)
    //--------------------------------------------------------------------------
    {
      int ID = FIELD_SPACE_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(field_space_desc.unique_id), 
                sizeof(field_space_desc.unique_id));
      lp_fwrite(f, field_space_desc.name, strlen(field_space_desc.name)+1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                       const LegionProfInstance::IndexPartDesc &index_part_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_PART_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(index_part_desc.unique_id), sizeof(UniqueID));
      lp_fwrite(f, index_part_desc.name, strlen(index_part_desc.name)+1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                     const LegionProfInstance::IndexSpaceDesc &index_space_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_SPACE_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(index_space_desc.unique_id), sizeof(UniqueID));
      lp_fwrite(f, index_space_desc.name, strlen(index_space_desc.name)+1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
               const LegionProfInstance::IndexSubSpaceDesc &index_subspace_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_SUBSPACE_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(index_subspace_desc.parent_id), sizeof(IDType));
      lp_fwrite(f, (char*)&(index_subspace_desc.unique_id), sizeof(IDType));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                  const LegionProfInstance::IndexPartitionDesc &index_part_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_PARTITION_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(index_part_desc.parent_id), sizeof(IDType));
      lp_fwrite(f, (char*)&(index_part_desc.unique_id), sizeof(IDType));
      lp_fwrite(f, (char*)&(index_part_desc.disjoint), sizeof(bool));
      lp_fwrite(f, (char*)&(index_part_desc.point), sizeof(LegionColor));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                           const LegionProfInstance::LogicalRegionDesc &lr_desc)
    //--------------------------------------------------------------------------
    {
      int ID = LOGICAL_REGION_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(lr_desc.ispace_id), sizeof(IDType));
      lp_fwrite(f, (char*)&(lr_desc.fspace_id), sizeof(unsigned));
      lp_fwrite(f, (char*)&(lr_desc.tree_id), sizeof(unsigned));
      lp_fwrite(f, lr_desc.name, strlen(lr_desc.name)+1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
           const LegionProfInstance::PhysicalInstRegionDesc &phy_instance_rdesc)
    //--------------------------------------------------------------------------
    {
      int ID = PHYSICAL_INST_REGION_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(phy_instance_rdesc.op_id), sizeof(UniqueID));
      lp_fwrite(f, (char*)&(phy_instance_rdesc.inst_id), sizeof(IDType));
      lp_fwrite(f, (char*)&(phy_instance_rdesc.ispace_id), sizeof(IDType));
      lp_fwrite(f, (char*)&(phy_instance_rdesc.fspace_id), sizeof(unsigned));
      lp_fwrite(f, (char*)&(phy_instance_rdesc.tree_id), sizeof(unsigned));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
           const LegionProfInstance::PhysicalInstDimOrderDesc
           &phy_instance_dim_order_rdesc)
    //--------------------------------------------------------------------------
    {
      int ID = PHYSICAL_INST_LAYOUT_DIM_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(phy_instance_dim_order_rdesc.op_id),
                sizeof(UniqueID));
      lp_fwrite(f, (char*)&(phy_instance_dim_order_rdesc.inst_id),
                sizeof(IDType));
      lp_fwrite(f, (char*)&(phy_instance_dim_order_rdesc.dim),
                sizeof(unsigned));
      lp_fwrite(f, (char*)&(phy_instance_dim_order_rdesc.k),
                sizeof(unsigned));
    }
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                const LegionProfInstance::PhysicalInstLayoutDesc 
                                                     &phy_instance_layout_rdesc)
    //--------------------------------------------------------------------------
    {
      int ID = PHYSICAL_INST_LAYOUT_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.op_id),sizeof(UniqueID));
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.inst_id),sizeof(InstID));
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.field_id),
                sizeof(unsigned));
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.fspace_id),
                sizeof(unsigned));
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.has_align),
                sizeof(bool));
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.eqk),
                sizeof(unsigned));
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.alignment),
                sizeof(unsigned));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                              const LegionProfInstance::IndexSpaceSizeDesc
                                                  &size_desc)
    //--------------------------------------------------------------------------
    {
      int ID = INDEX_SPACE_SIZE_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(size_desc.id),sizeof(UniqueID));
      lp_fwrite(f, (char*)&(size_desc.dense_size),sizeof(unsigned long long));
      lp_fwrite(f, (char*)&(size_desc.sparse_size),sizeof(unsigned long long));
      lp_fwrite(f, (char*)&(size_desc.is_sparse),sizeof(bool));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::TaskKind& task_kind)
    //--------------------------------------------------------------------------
    {
      int ID = TASK_KIND_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(task_kind.task_id), sizeof(task_kind.task_id));
      lp_fwrite(f, task_kind.name, strlen(task_kind.name) + 1);
      lp_fwrite(f, (char*)&(task_kind.overwrite), sizeof(task_kind.overwrite));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                            const LegionProfInstance::TaskVariant& task_variant)
    //--------------------------------------------------------------------------
    {
      int ID = TASK_VARIANT_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(task_variant.task_id),sizeof(task_variant.task_id));
      lp_fwrite(f, (char*)&(task_variant.variant_id), 
                sizeof(task_variant.variant_id));
      lp_fwrite(f, task_variant.name, strlen(task_variant.name) + 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                const LegionProfInstance::OperationInstance& operation_instance)
    //--------------------------------------------------------------------------
    {
      int ID = OPERATION_INSTANCE_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(operation_instance.op_id), 
                sizeof(operation_instance.op_id));
      lp_fwrite(f, (char*)&(operation_instance.parent_id),
                sizeof(operation_instance.parent_id));
      lp_fwrite(f, (char*)&(operation_instance.kind),
                sizeof(operation_instance.kind));
      if (operation_instance.provenance != NULL)
        lp_fwrite(f, operation_instance.provenance,
            strlen(operation_instance.provenance) + 1);
      else
        lp_fwrite(f, "", 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                const LegionProfInstance::MultiTask& multi_task)
    //--------------------------------------------------------------------------
    {
      int ID = MULTI_TASK_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(multi_task.op_id),   sizeof(multi_task.op_id));
      lp_fwrite(f, (char*)&(multi_task.task_id), sizeof(multi_task.task_id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                              const LegionProfInstance::SliceOwner& slice_owner)
    //--------------------------------------------------------------------------
    {
      int ID = SLICE_OWNER_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(slice_owner.parent_id), 
                sizeof(slice_owner.parent_id));
      lp_fwrite(f, (char*)&(slice_owner.op_id), sizeof(slice_owner.op_id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::WaitInfo wait_info, 
                                  const LegionProfInstance::TaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      int ID = TASK_WAIT_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(task_info.op_id),     sizeof(task_info.op_id));
      lp_fwrite(f, (char*)&(task_info.task_id),   sizeof(task_info.task_id));
      lp_fwrite(f, (char*)&(task_info.variant_id),sizeof(task_info.variant_id));
      lp_fwrite(f, (char*)&(wait_info.wait_start),sizeof(wait_info.wait_start));
      lp_fwrite(f, (char*)&(wait_info.wait_ready),sizeof(wait_info.wait_ready));
      lp_fwrite(f, (char*)&(wait_info.wait_end),  sizeof(wait_info.wait_end));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                              const LegionProfInstance::WaitInfo wait_info,
                              const LegionProfInstance::GPUTaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      int ID = TASK_WAIT_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(task_info.op_id),     sizeof(task_info.op_id));
      lp_fwrite(f, (char*)&(task_info.task_id),   sizeof(task_info.task_id));
      lp_fwrite(f, (char*)&(task_info.variant_id),sizeof(task_info.variant_id));
      lp_fwrite(f, (char*)&(wait_info.wait_start),sizeof(wait_info.wait_start));
      lp_fwrite(f, (char*)&(wait_info.wait_ready),sizeof(wait_info.wait_ready));
      lp_fwrite(f, (char*)&(wait_info.wait_end),  sizeof(wait_info.wait_end));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::WaitInfo wait_info, 
                                  const LegionProfInstance::MetaInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      int ID = META_WAIT_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(meta_info.op_id),     sizeof(meta_info.op_id));
      lp_fwrite(f, (char*)&(meta_info.lg_id),     sizeof(meta_info.lg_id));
      lp_fwrite(f, (char*)&(wait_info.wait_start),sizeof(wait_info.wait_start));
      lp_fwrite(f, (char*)&(wait_info.wait_ready),sizeof(wait_info.wait_ready));
      lp_fwrite(f, (char*)&(wait_info.wait_end),  sizeof(wait_info.wait_end));
    }
 
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::TaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      int ID = TASK_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(task_info.op_id),     sizeof(task_info.op_id));
      lp_fwrite(f, (char*)&(task_info.task_id),   sizeof(task_info.task_id));
      lp_fwrite(f, (char*)&(task_info.variant_id),sizeof(task_info.variant_id));
      lp_fwrite(f, (char*)&(task_info.proc_id),   sizeof(task_info.proc_id));
      lp_fwrite(f, (char*)&(task_info.create),    sizeof(task_info.create));
      lp_fwrite(f, (char*)&(task_info.ready),     sizeof(task_info.ready));
      lp_fwrite(f, (char*)&(task_info.start),     sizeof(task_info.start));
      lp_fwrite(f, (char*)&(task_info.stop),      sizeof(task_info.stop));
#ifdef LEGION_PROF_PROVENANCE
      lp_fwrite(f, (char*)&(task_info.provenance),sizeof(task_info.provenance));
      lp_fwrite(f, (char*)&(task_info.finish_event),
                                                sizeof(task_info.finish_event));
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                               const LegionProfInstance::GPUTaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      int ID = GPU_TASK_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(task_info.op_id),     sizeof(task_info.op_id));
      lp_fwrite(f, (char*)&(task_info.task_id),   sizeof(task_info.task_id));
      lp_fwrite(f, (char*)&(task_info.variant_id),sizeof(task_info.variant_id));
      lp_fwrite(f, (char*)&(task_info.proc_id),   sizeof(task_info.proc_id));
      lp_fwrite(f, (char*)&(task_info.create),    sizeof(task_info.create));
      lp_fwrite(f, (char*)&(task_info.ready),     sizeof(task_info.ready));
      lp_fwrite(f, (char*)&(task_info.start),     sizeof(task_info.start));
      lp_fwrite(f, (char*)&(task_info.stop),      sizeof(task_info.stop));
      lp_fwrite(f, (char*)&(task_info.gpu_start), sizeof(task_info.gpu_start));
      lp_fwrite(f, (char*)&(task_info.gpu_stop),  sizeof(task_info.gpu_stop));
#ifdef LEGION_PROF_PROVENANCE
      lp_fwrite(f, (char*)&(task_info.provenance),sizeof(task_info.provenance));
      lp_fwrite(f, (char*)&(task_info.finish_event),
                                                sizeof(task_info.finish_event));
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::MetaInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      int ID = META_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(meta_info.op_id),   sizeof(meta_info.op_id));
      lp_fwrite(f, (char*)&(meta_info.lg_id),   sizeof(meta_info.lg_id));
      lp_fwrite(f, (char*)&(meta_info.proc_id), sizeof(meta_info.proc_id));
      lp_fwrite(f, (char*)&(meta_info.create),  sizeof(meta_info.create));
      lp_fwrite(f, (char*)&(meta_info.ready),   sizeof(meta_info.ready));
      lp_fwrite(f, (char*)&(meta_info.start),   sizeof(meta_info.start));
      lp_fwrite(f, (char*)&(meta_info.stop),    sizeof(meta_info.stop));
#ifdef LEGION_PROF_PROVENANCE
      lp_fwrite(f, (char*)&(meta_info.provenance),sizeof(meta_info.provenance));
      lp_fwrite(f, (char*)&(meta_info.finish_event),
                                                sizeof(meta_info.finish_event));
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                          const LegionProfInstance::CopyInfo& copy_info)
    //--------------------------------------------------------------------------
    {
      int ID = COPY_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));

      lp_fwrite(f, (char*)&(copy_info.op_id),  sizeof(copy_info.op_id));
      lp_fwrite(f, (char*)&(copy_info.src),    sizeof(copy_info.src));
      lp_fwrite(f, (char*)&(copy_info.dst),    sizeof(copy_info.dst));
      lp_fwrite(f, (char*)&(copy_info.size),   sizeof(copy_info.size));
      lp_fwrite(f, (char*)&(copy_info.create), sizeof(copy_info.create));
      lp_fwrite(f, (char*)&(copy_info.ready),  sizeof(copy_info.ready));
      lp_fwrite(f, (char*)&(copy_info.start),  sizeof(copy_info.start));
      lp_fwrite(f, (char*)&(copy_info.stop),   sizeof(copy_info.stop));
      lp_fwrite(f, (char*)&(copy_info.fevent),sizeof(copy_info.fevent.id));
      lp_fwrite(f, (char*)&(copy_info.num_requests),   sizeof(copy_info.num_requests));
#ifdef LEGION_PROF_PROVENANCE
      lp_fwrite(f, (char*)&(copy_info.provenance),sizeof(copy_info.provenance));
#endif
    }

        //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::CopyInstInfo &copy_inst,
                                  const LegionProfInstance::CopyInfo& copy_info)
    //--------------------------------------------------------------------------
    {
      int ID = COPY_INST_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(copy_info.op_id),     sizeof(copy_info.op_id));
      lp_fwrite(f, (char*)&(copy_inst.src_inst_id),sizeof(copy_inst.src_inst_id));
      lp_fwrite(f, (char*)&(copy_inst.dst_inst_id),sizeof(copy_inst.dst_inst_id));
      lp_fwrite(f, (char*)&(copy_info.fevent),sizeof(copy_info.fevent.id));
      lp_fwrite(f, (char*)&(copy_inst.num_fields),sizeof(copy_inst.num_fields));
      lp_fwrite(f, (char*)&(copy_inst.request_type),sizeof(copy_inst.request_type));
      lp_fwrite(f, (char*)&(copy_inst.num_hops),sizeof(copy_inst.num_hops));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::FillInfo& fill_info)
    //--------------------------------------------------------------------------
    {
      int ID = FILL_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));

      lp_fwrite(f, (char*)&(fill_info.op_id),  sizeof(fill_info.op_id));
      lp_fwrite(f, (char*)&(fill_info.dst),    sizeof(fill_info.dst));
      lp_fwrite(f, (char*)&(fill_info.create), sizeof(fill_info.create));
      lp_fwrite(f, (char*)&(fill_info.ready),  sizeof(fill_info.ready));
      lp_fwrite(f, (char*)&(fill_info.start),  sizeof(fill_info.start));
      lp_fwrite(f, (char*)&(fill_info.stop),   sizeof(fill_info.stop));
#ifdef LEGION_PROF_PROVENANCE
      lp_fwrite(f, (char*)&(fill_info.provenance),sizeof(fill_info.provenance));
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                     const LegionProfInstance::InstCreateInfo& inst_create_info)
    //--------------------------------------------------------------------------
    {
      int ID = INST_CREATE_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(inst_create_info.op_id),   
                sizeof(inst_create_info.op_id));
      lp_fwrite(f, (char*)&(inst_create_info.inst_id), 
                sizeof(inst_create_info.inst_id));
      lp_fwrite(f, (char*)&(inst_create_info.create),  
                sizeof(inst_create_info.create));
#ifdef LEGION_PROF_PROVENANCE
      lp_fwrite(f, (char*)&(inst_create_info.provenance),
                sizeof(inst_create_info.provenance));
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                       const LegionProfInstance::InstUsageInfo& inst_usage_info)
    //--------------------------------------------------------------------------
    {
      int ID = INST_USAGE_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(inst_usage_info.op_id),   
                sizeof(inst_usage_info.op_id));
      lp_fwrite(f, (char*)&(inst_usage_info.inst_id), 
                sizeof(inst_usage_info.inst_id));
      lp_fwrite(f, (char*)&(inst_usage_info.mem_id),  
                sizeof(inst_usage_info.mem_id));
      lp_fwrite(f, (char*)&(inst_usage_info.size),    
                sizeof(inst_usage_info.size));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                 const LegionProfInstance::InstTimelineInfo& inst_timeline_info)
    //--------------------------------------------------------------------------
    {
      int ID = INST_TIMELINE_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(inst_timeline_info.op_id),   
                sizeof(inst_timeline_info.op_id));
      lp_fwrite(f, (char*)&(inst_timeline_info.inst_id), 
                sizeof(inst_timeline_info.inst_id));
      lp_fwrite(f, (char*)&(inst_timeline_info.create),  
                sizeof(inst_timeline_info.create));
      lp_fwrite(f, (char*)&(inst_timeline_info.ready),  
                sizeof(inst_timeline_info.ready));
      lp_fwrite(f, (char*)&(inst_timeline_info.destroy), 
                sizeof(inst_timeline_info.destroy));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                        const LegionProfInstance::PartitionInfo& partition_info)
    //--------------------------------------------------------------------------
    {
      int ID = PARTITION_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(partition_info.op_id),
                sizeof(partition_info.op_id));
      lp_fwrite(f, (char*)&(partition_info.part_op),
                sizeof(partition_info.part_op));
      lp_fwrite(f, (char*)&(partition_info.create),
                sizeof(partition_info.create));
      lp_fwrite(f, (char*)&(partition_info.ready),
                sizeof(partition_info.ready));
      lp_fwrite(f, (char*)&(partition_info.start),
                sizeof(partition_info.start));
      lp_fwrite(f, (char*)&(partition_info.stop),
                sizeof(partition_info.stop));
#ifdef LEGION_PROF_PROVENANCE
      lp_fwrite(f, (char*)&(partition_info.provenance),
                sizeof(partition_info.provenance));
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                     const LegionProfInstance::MapperCallInfo& mapper_call_info)
    //--------------------------------------------------------------------------
    {
      int ID = MAPPER_CALL_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(mapper_call_info.kind),    
                sizeof(mapper_call_info.kind));
      lp_fwrite(f, (char*)&(mapper_call_info.op_id),   
                sizeof(mapper_call_info.op_id));
      lp_fwrite(f, (char*)&(mapper_call_info.start),   
                sizeof(mapper_call_info.start));
      lp_fwrite(f, (char*)&(mapper_call_info.stop),    
                sizeof(mapper_call_info.stop));
      lp_fwrite(f, (char*)&(mapper_call_info.proc_id), 
                sizeof(mapper_call_info.proc_id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                   const LegionProfInstance::RuntimeCallInfo& runtime_call_info)
    //--------------------------------------------------------------------------
    {
      int ID = RUNTIME_CALL_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(runtime_call_info.kind),    
                sizeof(runtime_call_info.kind));
      lp_fwrite(f, (char*)&(runtime_call_info.start),   
                sizeof(runtime_call_info.start));
      lp_fwrite(f, (char*)&(runtime_call_info.stop),    
                sizeof(runtime_call_info.stop));
      lp_fwrite(f, (char*)&(runtime_call_info.proc_id), 
                sizeof(runtime_call_info.proc_id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                      const LegionProfInstance::ProcDesc& proc_desc)
    //--------------------------------------------------------------------------
    {
      int ID = PROC_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(proc_desc.proc_id), sizeof(proc_desc.proc_id));
      lp_fwrite(f, (char*)&(proc_desc.kind),    sizeof(proc_desc.kind));
    }
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                        const LegionProfInstance::MemDesc& mem_desc)
    //--------------------------------------------------------------------------
    {
      int ID = MEM_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(mem_desc.mem_id),   sizeof(mem_desc.mem_id));
      lp_fwrite(f, (char*)&(mem_desc.kind),     sizeof(mem_desc.kind));
      lp_fwrite(f, (char*)&(mem_desc.capacity), sizeof(mem_desc.capacity));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                          const LegionProfInstance::ProcMemDesc &pm)
    //--------------------------------------------------------------------------
    {
      int ID = PROC_MEM_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(pm.proc_id), sizeof(pm.proc_id));
      lp_fwrite(f, (char*) &(pm.mem_id), sizeof(pm.mem_id));
      lp_fwrite(f, (char*) &(pm.bandwidth), sizeof(pm.bandwidth));
      lp_fwrite(f, (char*) &(pm.latency), sizeof(pm.latency));
    }


#ifdef LEGION_PROF_SELF_PROFILE
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                          const LegionProfInstance::ProfTaskInfo& proftask_info)
    //--------------------------------------------------------------------------
    {
      int ID = PROFTASK_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(proftask_info.proc_id), 
                sizeof(proftask_info.proc_id));
      lp_fwrite(f, (char*)&(proftask_info.op_id), sizeof(proftask_info.op_id));
      lp_fwrite(f, (char*)&(proftask_info.start), sizeof(proftask_info.start));
      lp_fwrite(f, (char*)&(proftask_info.stop),  sizeof(proftask_info.stop));
    }
#endif

    //--------------------------------------------------------------------------
    LegionProfBinarySerializer::~LegionProfBinarySerializer()
    //--------------------------------------------------------------------------
    {
      lp_fflush(f, Z_FULL_FLUSH);
      lp_fclose(f);
    }



    ///////////////////////// LegionProfASCIISerializer ///////////////////////

    //--------------------------------------------------------------------------
    LegionProfASCIISerializer::LegionProfASCIISerializer()
    //--------------------------------------------------------------------------
    {
    }

    // Serialize Methods
    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
               const LegionProfInstance::IndexSpacePointDesc &ispace_point_desc)
    //--------------------------------------------------------------------------
    {
#if LEGION_MAX_DIM == 1
      log_prof.print("Index Space Point Desc  %llu %d %lld",
		     ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0]);
#elif LEGION_MAX_DIM == 2
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld",
		     ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1]);
#elif LEGION_MAX_DIM == 3
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld %lld",
		     ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1],
		     (long long)ispace_point_desc.points[2]);
#elif LEGION_MAX_DIM == 4
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld %lld %lld",
		     ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1],
		     (long long)ispace_point_desc.points[2],
                     (long long)ispace_point_desc.points[3]);
#elif LEGION_MAX_DIM == 5
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld %lld "
	             "%lld %lld",
		     ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1],
		     (long long)ispace_point_desc.points[2],
                     (long long)ispace_point_desc.points[3],
                     (long long)ispace_point_desc.points[4]);
#elif LEGION_MAX_DIM == 6
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld %lld "
                     "%lld %lld %lld", ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1],
		     (long long)ispace_point_desc.points[2],
                     (long long)ispace_point_desc.points[3],
                     (long long)ispace_point_desc.points[4],
                     (long long)ispace_point_desc.points[5]);
#elif LEGION_MAX_DIM == 7
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld %lld "
                     "%lld %lld %lld %lld", ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1],
		     (long long)ispace_point_desc.points[2],
                     (long long)ispace_point_desc.points[3],
                     (long long)ispace_point_desc.points[4],
                     (long long)ispace_point_desc.points[5],
                     (long long)ispace_point_desc.points[6]);
#elif LEGION_MAX_DIM == 8
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld %lld "
                     "%lld %lld %lld %lld %lld", ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1],
		     (long long)ispace_point_desc.points[2],
                     (long long)ispace_point_desc.points[3],
                     (long long)ispace_point_desc.points[4],
                     (long long)ispace_point_desc.points[5],
                     (long long)ispace_point_desc.points[6],
                     (long long)ispace_point_desc.points[7]);
#elif LEGION_MAX_DIM == 9
      log_prof.print("Index Space Point Desc  %llu %d %lld %lld %lld %lld %lld "
                     "%lld %lld %lld %lld", ispace_point_desc.unique_id,
		     ispace_point_desc.dim,
		     (long long)ispace_point_desc.points[0],
		     (long long)ispace_point_desc.points[1],
		     (long long)ispace_point_desc.points[2],
                     (long long)ispace_point_desc.points[3],
                     (long long)ispace_point_desc.points[4],
                     (long long)ispace_point_desc.points[5],
                     (long long)ispace_point_desc.points[6],
                     (long long)ispace_point_desc.points[7],
                     (long long)ispace_point_desc.points[8]);
#else
#error "Illegal LEGION_MAX_DIM"
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
               const LegionProfInstance::IndexSpaceEmptyDesc &ispace_empty_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Index Space Empty Desc %llu",
		     ispace_empty_desc.unique_id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                 const LegionProfInstance::IndexSpaceRectDesc &ispace_rect_desc)
    //--------------------------------------------------------------------------
    {
#if LEGION_MAX_DIM == 1
      log_prof.print("Index Space Rect Desc %llu %d %lld "
                     "%lld", ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     );
#elif LEGION_MAX_DIM == 2
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld "
                     "%lld", ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     );
#elif LEGION_MAX_DIM == 3
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld %lld "
                     "%lld %lld", ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_lo[2]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     (long long)(ispace_rect_desc.rect_hi[2])
		     );
#elif LEGION_MAX_DIM == 4
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld %lld "
                     "%lld %lld %lld %lld", ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_lo[2]),
                     (long long)(ispace_rect_desc.rect_lo[3]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     (long long)(ispace_rect_desc.rect_hi[2]),
                     (long long)(ispace_rect_desc.rect_hi[3])
		     );
#elif LEGION_MAX_DIM == 5
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld %lld "
                     "%lld %lld %lld %lld %lld %lld",
                     ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_lo[2]),
                     (long long)(ispace_rect_desc.rect_lo[3]),
                     (long long)(ispace_rect_desc.rect_lo[4]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     (long long)(ispace_rect_desc.rect_hi[2]),
                     (long long)(ispace_rect_desc.rect_hi[3]),
                     (long long)(ispace_rect_desc.rect_hi[4])
		     );
#elif LEGION_MAX_DIM == 6
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld %lld %lld "
                     "%lld %lld %lld %lld %lld %lld %lld",
                     ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_lo[2]),
                     (long long)(ispace_rect_desc.rect_lo[3]),
                     (long long)(ispace_rect_desc.rect_lo[4]),
                     (long long)(ispace_rect_desc.rect_lo[5]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     (long long)(ispace_rect_desc.rect_hi[2]),
                     (long long)(ispace_rect_desc.rect_hi[3]),
                     (long long)(ispace_rect_desc.rect_hi[4]),
                     (long long)(ispace_rect_desc.rect_hi[5])
		     );
#elif LEGION_MAX_DIM == 7
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld %lld %lld "
                     "%lld %lld %lld %lld %lld %lld %lld %lld %lld",
                     ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_lo[2]),
                     (long long)(ispace_rect_desc.rect_lo[3]),
                     (long long)(ispace_rect_desc.rect_lo[4]),
                     (long long)(ispace_rect_desc.rect_lo[5]),
                     (long long)(ispace_rect_desc.rect_lo[6]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     (long long)(ispace_rect_desc.rect_hi[2]),
                     (long long)(ispace_rect_desc.rect_hi[3]),
                     (long long)(ispace_rect_desc.rect_hi[4]),
                     (long long)(ispace_rect_desc.rect_hi[5]),
                     (long long)(ispace_rect_desc.rect_hi[6])
		     );
#elif LEGION_MAX_DIM == 8
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld %lld %lld "
                     "%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld "
                     "%lld", ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_lo[2]),
                     (long long)(ispace_rect_desc.rect_lo[3]),
                     (long long)(ispace_rect_desc.rect_lo[4]),
                     (long long)(ispace_rect_desc.rect_lo[5]),
                     (long long)(ispace_rect_desc.rect_lo[6]),
                     (long long)(ispace_rect_desc.rect_lo[7]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     (long long)(ispace_rect_desc.rect_hi[2]),
                     (long long)(ispace_rect_desc.rect_hi[3]),
                     (long long)(ispace_rect_desc.rect_hi[4]),
                     (long long)(ispace_rect_desc.rect_hi[5]),
                     (long long)(ispace_rect_desc.rect_hi[6]),
                     (long long)(ispace_rect_desc.rect_hi[7])
		     );
#elif LEGION_MAX_DIM == 9
      log_prof.print("Index Space Rect Desc %llu %d %lld %lld %lld %lld %lld "
                     "%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld "
                     "%lld %lld %lld", ispace_rect_desc.unique_id,
		     ispace_rect_desc.dim,
		     (long long)(ispace_rect_desc.rect_lo[0]),
		     (long long)(ispace_rect_desc.rect_lo[1]),
		     (long long)(ispace_rect_desc.rect_lo[2]),
                     (long long)(ispace_rect_desc.rect_lo[3]),
                     (long long)(ispace_rect_desc.rect_lo[4]),
                     (long long)(ispace_rect_desc.rect_lo[5]),
                     (long long)(ispace_rect_desc.rect_lo[6]),
                     (long long)(ispace_rect_desc.rect_lo[7]),
                     (long long)(ispace_rect_desc.rect_lo[8]),
		     (long long)(ispace_rect_desc.rect_hi[0]),
		     (long long)(ispace_rect_desc.rect_hi[1]),
		     (long long)(ispace_rect_desc.rect_hi[2]),
                     (long long)(ispace_rect_desc.rect_hi[3]),
                     (long long)(ispace_rect_desc.rect_hi[4]),
                     (long long)(ispace_rect_desc.rect_hi[5]),
                     (long long)(ispace_rect_desc.rect_hi[6]),
                     (long long)(ispace_rect_desc.rect_hi[7]),
                     (long long)(ispace_rect_desc.rect_hi[8])
		     );
#else
#error "Illegal LEGION_MAX_DIM"
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                const LegionProfInstance::FieldDesc &field_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Field Name Desc %llu %u %llu %s",
		     field_desc.unique_id, field_desc.field_id,
		     field_desc.size, field_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                     const LegionProfInstance::FieldSpaceDesc &field_space_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Field Space Name Desc %llu %s",
		     field_space_desc.unique_id,
		     field_space_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                       const LegionProfInstance::IndexPartDesc &index_part_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Index Part Name Desc %llu %s",
                     index_part_desc.unique_id,
                     index_part_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                     const LegionProfInstance::IndexSpaceDesc &index_space_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Index Space Name Desc %llu %s",
		     index_space_desc.unique_id,
		     index_space_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
               const LegionProfInstance::IndexSubSpaceDesc &index_subspace_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Index Sub Space Desc %llu %llu",
		     index_subspace_desc.parent_id,
		     index_subspace_desc.unique_id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                  const LegionProfInstance::IndexPartitionDesc &index_part_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Index Partition Desc %llu %llu %d %llu",
                     index_part_desc.parent_id,
                     index_part_desc.unique_id,
                     index_part_desc.disjoint,
                     index_part_desc.point);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                           const LegionProfInstance::LogicalRegionDesc &lr_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Logical Region Desc %llu %u %u %s",
		     lr_desc.ispace_id,
		     lr_desc.fspace_id,
		     lr_desc.tree_id,
		     lr_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
           const LegionProfInstance::PhysicalInstRegionDesc &phy_instance_rdesc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Physical Inst Region Desc " "%llu "  IDFMT " %llu %u %u",
		     phy_instance_rdesc.op_id,
		     phy_instance_rdesc.inst_id,
		     phy_instance_rdesc.ispace_id,
		     phy_instance_rdesc.fspace_id,
		     phy_instance_rdesc.tree_id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
           const LegionProfInstance::PhysicalInstDimOrderDesc
           &phy_instance_dim_order_rdesc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Physical Inst Dim Order Desc " "%llu " IDFMT " %u %u",
                     phy_instance_dim_order_rdesc.op_id,
                     phy_instance_dim_order_rdesc.inst_id,
                     phy_instance_dim_order_rdesc.dim,
                     phy_instance_dim_order_rdesc.k
                     );
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                 const LegionProfInstance::IndexSpaceSizeDesc
                                                  &size_desc)
    //--------------------------------------------------------------------------
    {

      log_prof.print("Index Space Size Desc " "%llu %llu %llu %u",
                     size_desc.id,
                     size_desc.dense_size,
                     size_desc.sparse_size,
                     size_desc.is_sparse
                     );
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
              const LegionProfInstance::PhysicalInstLayoutDesc
              &phy_instance_layout_rdesc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Physical Inst Layout Desc " "%llu " IDFMT " %u %u %u %u "
                     "%u",
                     phy_instance_layout_rdesc.op_id,
                     phy_instance_layout_rdesc.inst_id,
                     phy_instance_layout_rdesc.field_id,
                     phy_instance_layout_rdesc.fspace_id,
                     phy_instance_layout_rdesc.has_align,
                     phy_instance_layout_rdesc.eqk,
                     phy_instance_layout_rdesc.alignment
                     );
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                         const LegionProfDesc::MapperCallDesc &mapper_call_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mapper Call Desc %u %s", 
                      mapper_call_desc.kind, mapper_call_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                       const LegionProfDesc::RuntimeCallDesc &runtime_call_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Runtime Call Desc %u %s", 
                     runtime_call_desc.kind, runtime_call_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                      const LegionProfDesc::MetaDesc &meta_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Meta Desc %u %d %d %s", meta_desc.kind,
       meta_desc.message ? 1 : 0, meta_desc.ordered_vc ? 1 : 0, meta_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                          const LegionProfDesc::OpDesc &op_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Op Desc %u %s", op_desc.kind, op_desc.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                      const LegionProfDesc::MaxDimDesc
				      &max_dim_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Max Dim Desc %d",
                     max_dim_desc.max_dim);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::TaskKind &task_kind)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Task Kind %u %s %d", task_kind.task_id, 
                      task_kind.name, (task_kind.overwrite ? 1 : 0));
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                            const LegionProfInstance::TaskVariant& task_variant)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Task Variant %u %u %s", task_variant.task_id,
         task_variant.variant_id, task_variant.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                           const LegionProfInstance::OperationInstance& op_inst)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Operation %llu %llu %u %s", 
          op_inst.op_id, op_inst.parent_id, op_inst.kind,
          op_inst.provenance == NULL ? "" : op_inst.provenance);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                const LegionProfInstance::MultiTask& multi_task)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Multi %llu %u", 
                      multi_task.op_id, multi_task.task_id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::SliceOwner& slice_owner)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Slice Owner %llu %llu", 
                      slice_owner.parent_id, slice_owner.op_id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::WaitInfo wait_info, 
                                  const LegionProfInstance::TaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Task Wait Info %llu %u %u %llu %llu %llu",
                task_info.op_id, task_info.task_id, task_info.variant_id, 
                wait_info.wait_start, wait_info.wait_ready, wait_info.wait_end);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::WaitInfo wait_info,
                              const LegionProfInstance::GPUTaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Task Wait Info %llu %u %u %llu %llu %llu",
                task_info.op_id, task_info.task_id, task_info.variant_id,
                wait_info.wait_start, wait_info.wait_ready, wait_info.wait_end);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::WaitInfo wait_info, 
                                  const LegionProfInstance::MetaInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Meta Wait Info %llu %u %llu %llu %llu",
                   meta_info.op_id, meta_info.lg_id, wait_info.wait_start, 
                   wait_info.wait_ready, wait_info.wait_end);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::TaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_PROVENANCE
      log_prof.print("Prof Task Info %llu %u %u " IDFMT " %llu %llu %llu %llu "
                     IDFMT " " IDFMT "",
                     task_info.op_id, task_info.task_id, task_info.variant_id, 
                     task_info.proc_id, task_info.create, task_info.ready, 
                     task_info.start, task_info.stop, task_info.provenance.id,
                     task_info.finish_event.id);
#else
      log_prof.print("Prof Task Info %llu %u %u " IDFMT " %llu %llu %llu %llu",
                     task_info.op_id, task_info.task_id, task_info.variant_id, 
                     task_info.proc_id, task_info.create, task_info.ready, 
                     task_info.start, task_info.stop);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::GPUTaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_PROVENANCE
      log_prof.print("Prof GPU Task Info %llu %u %u " IDFMT
		     " %llu %llu %llu %llu %llu %llu " IDFMT " " IDFMT "",
                     task_info.op_id, task_info.task_id, task_info.variant_id,
                     task_info.proc_id, task_info.create, task_info.ready,
                     task_info.start, task_info.stop, task_info.gpu_start,
		     task_info.gpu_stop, task_info.provenance.id, 
                     task_info.finish_event.id);
#else
      log_prof.print("Prof GPU Task Info %llu %u %u " IDFMT
		     " %llu %llu %llu %llu %llu %llu",
                     task_info.op_id, task_info.task_id, task_info.variant_id,
                     task_info.proc_id, task_info.create, task_info.ready,
                     task_info.start, task_info.stop, task_info.gpu_start,
		     task_info.gpu_stop);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::MetaInfo& meta_info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_PROVENANCE
      log_prof.print("Prof Meta Info %llu %u " IDFMT " %llu %llu %llu %llu "
          IDFMT " " IDFMT "",
         meta_info.op_id, meta_info.lg_id, meta_info.proc_id,
         meta_info.create, meta_info.ready, meta_info.start, meta_info.stop,
         meta_info.provenance.id, meta_info.finish_event.id);
#else
      log_prof.print("Prof Meta Info %llu %u " IDFMT " %llu %llu %llu %llu",
         meta_info.op_id, meta_info.lg_id, meta_info.proc_id,
         meta_info.create, meta_info.ready, meta_info.start, meta_info.stop);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::CopyInfo& copy_info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_PROVENANCE
      log_prof.print("Prof Copy Info %llu " IDFMT " " IDFMT " %llu"
                     " %llu %llu %llu %llu " IDFMT " " IDFMT " %u",
                     copy_info.op_id, copy_info.src,
                     copy_info.dst, copy_info.size, copy_info.create,
                     copy_info.ready, copy_info.start, copy_info.stop,
                     copy_info.fevent.id,
                     copy_info.provenance.id,
                     copy_info.num_requests);
#else
      log_prof.print("Prof Copy Info %llu " IDFMT " " IDFMT " %llu"
                     " %llu %llu %llu %llu " IDFMT " %u", copy_info.op_id,
                     copy_info.src,
                     copy_info.dst, copy_info.size, copy_info.create,
                     copy_info.ready, copy_info.start, copy_info.stop,
                     copy_info.fevent.id,
                     copy_info.num_requests);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                          const LegionProfInstance::CopyInstInfo& copy_inst,
                          const LegionProfInstance::CopyInfo& copy_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Copy Inst Info %llu " IDFMT " " IDFMT " " IDFMT " %u %u %u",
                     copy_info.op_id, copy_inst.src_inst_id, copy_inst.dst_inst_id,
                     copy_info.fevent.id, copy_inst.num_fields, copy_inst.request_type, copy_inst.num_hops);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::FillInfo& fill_info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_PROVENANCE
      log_prof.print("Prof Fill Info %llu " IDFMT " %llu %llu %llu %llu " IDFMT,
        fill_info.op_id, fill_info.dst, fill_info.create, fill_info.ready, 
        fill_info.start, fill_info.stop, fill_info.provenance.id);
#else
      log_prof.print("Prof Fill Info %llu " IDFMT 
         " %llu %llu %llu %llu", fill_info.op_id, fill_info.dst, 
         fill_info.create, fill_info.ready, fill_info.start, fill_info.stop);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                     const LegionProfInstance::InstCreateInfo& inst_create_info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_PROVENANCE
      log_prof.print("Prof Inst Create %llu " IDFMT " %llu " IDFMT "", 
                     inst_create_info.op_id, inst_create_info.inst_id, 
                     inst_create_info.create, inst_create_info.provenance.id);
#else
      log_prof.print("Prof Inst Create %llu " IDFMT " %llu", 
                     inst_create_info.op_id, inst_create_info.inst_id, 
                     inst_create_info.create);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                       const LegionProfInstance::InstUsageInfo& inst_usage_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Inst Usage %llu " IDFMT " " IDFMT " %llu",
                     inst_usage_info.op_id, inst_usage_info.inst_id, 
                     inst_usage_info.mem_id, inst_usage_info.size);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                 const LegionProfInstance::InstTimelineInfo& inst_timeline_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Inst Timeline %llu " IDFMT " %llu %llu %llu",
         inst_timeline_info.op_id, inst_timeline_info.inst_id,
         inst_timeline_info.create, inst_timeline_info.ready,
         inst_timeline_info.destroy);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                        const LegionProfInstance::PartitionInfo& partition_info)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_PROVENANCE
      log_prof.print("Prof Partition Timeline %llu %d %llu %llu %llu %llu "
                     IDFMT "", partition_info.op_id, partition_info.part_op, 
                     partition_info.create, partition_info.create,
                     partition_info.start, partition_info.stop,
                     partition_info.provenance.id);
#else
      log_prof.print("Prof Partition Timeline %llu %d %llu %llu %llu %llu",
                     partition_info.op_id, partition_info.part_op, 
                     partition_info.create, partition_info.create,
                     partition_info.start, partition_info.stop);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                     const LegionProfInstance::MapperCallInfo& mapper_call_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mapper Call Info %u " IDFMT " %llu %llu %llu",
        mapper_call_info.kind, mapper_call_info.proc_id, mapper_call_info.op_id,
        mapper_call_info.start, mapper_call_info.stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                   const LegionProfInstance::RuntimeCallInfo& runtime_call_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Runtime Call Info %u " IDFMT " %llu %llu",
                     runtime_call_info.kind, runtime_call_info.proc_id, 
                     runtime_call_info.start, runtime_call_info.stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                 const LegionProfInstance::ProcDesc &proc_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Proc Desc " IDFMT " %d",
                     proc_desc.proc_id, proc_desc.kind);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                   const LegionProfInstance::MemDesc &mem_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mem Desc " IDFMT " %d %llu",
                      mem_desc.mem_id, mem_desc.kind, mem_desc.capacity);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                    const LegionProfInstance::ProcMemDesc &pm)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mem Proc Affinity Desc " IDFMT " " IDFMT " %u %u",
		     pm.proc_id, pm.mem_id, pm.bandwidth, pm.latency);
    }

#ifdef LEGION_PROF_SELF_PROFILE
    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                          const LegionProfInstance::ProfTaskInfo& proftask_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof ProfTask Info " IDFMT " %llu %llu %llu",
                     proftask_info.proc_id, proftask_info.op_id, 
                     proftask_info.start, proftask_info.stop);
    }
#endif

    //--------------------------------------------------------------------------
    LegionProfASCIISerializer::~LegionProfASCIISerializer()
    //--------------------------------------------------------------------------
    {
    }

  }; // namespace Internal
}; // namespace Legion

