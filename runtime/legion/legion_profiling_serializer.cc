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

      ss << "MapperName {"
         << "id:" << MAPPER_NAME_ID << delim
         << "mapper_id:MapperID:" << sizeof(MapperID) << delim
         << "mapper_proc:ProcID:" << sizeof(ProcID) << delim
         << "name:string:" << "-1"
         << "}" << std::endl;

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
         << "id:" << PROC_DESC_ID                     << delim
         << "proc_id:ProcID:"     << sizeof(ProcID)   << delim
         << "kind:ProcKind:"      << sizeof(ProcKind) << delim
         << "uuid_size:uuid_size:" << sizeof(unsigned) << delim
         << "cuda_device_uuid:uuid:"          << sizeof(char)
         << "}" << std::endl;

      ss << "MaxDimDesc {"
         << "id:" << MAX_DIM_DESC_ID                 << delim
         << "max_dim:maxdim:" << sizeof(unsigned)
         << "}" << std::endl;

      ss << "RuntimeConfig {"
         << "id:" << RUNTIME_CONFIG_ID << delim
         << "debug:bool:" << sizeof(bool) << delim
         << "spy:bool:" << sizeof(bool) << delim
         << "gc:bool:" << sizeof(bool) << delim
         << "inorder:bool:" << sizeof(bool) << delim
         << "safe_mapper:bool:" << sizeof(bool) << delim
         << "safe_runtime:bool:" << sizeof(bool) << delim
         << "safe_ctrlrepl:bool:" << sizeof(bool) << delim
         << "part_checks:bool:" << sizeof(bool) << delim
         << "bounds_checks:bool:" << sizeof(bool) << delim
         << "resilient:bool:" << sizeof(bool)
         << "}" << std::endl;

      ss << "MachineDesc {"
         << "id:" << MACHINE_DESC_ID                  << delim
         << "node_id:unsigned:"   << sizeof(unsigned) << delim
         << "num_nodes:unsigned:" << sizeof(unsigned) << delim
         << "version:unsigned:" << sizeof(unsigned)   << delim
         << "hostname:string:"    << "-1"             << delim
         << "host_id:unsigned long long:" << sizeof(unsigned long long) << delim
         << "process_id:unsigned:" << sizeof(unsigned)
         << "}" << std::endl;

      ss << "CalibrationErr {"
         << "id:" << CALIBRATION_ERR_ID                << delim
         << "calibration_err:long long:" << sizeof(long long)
         << "}" << std::endl;

      ss << "ZeroTime {"
         << "id:" << ZERO_TIME_ID  << delim
         << "zero_time:long long:" << sizeof(long long)
         << "}" << std::endl;

      ss << "Provenance {"
         << "id:" << PROVENANCE_ID << delim
         << "provenance:unsigned long long:" <<sizeof(unsigned long long) << delim
         << "prov:string:" << "-1"
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
         << "inst_uid:unsigned long long:" << sizeof(LgEvent)     << delim
	 << "ispace_id:IDType:"            << sizeof(IDType)      << delim
	 << "fspace_id:unsigned:"          << sizeof(unsigned)    << delim
	 << "tree_id:unsigned:"            << sizeof(unsigned)
         << "}" << std::endl;

      ss << "PhysicalInstLayoutDesc {"
         << "id:" << PHYSICAL_INST_LAYOUT_ID                  << delim
         << "inst_uid:unsigned long long:" << sizeof(LgEvent) << delim
         << "field_id:unsigned:"        << sizeof(unsigned)   << delim
         << "fspace_id:unsigned:"       << sizeof(unsigned)   << delim
         << "has_align:bool:"           << sizeof(bool)       << delim
         << "eqk:unsigned:"             << sizeof(unsigned)   << delim
         << "align_desc:unsigned:"      << sizeof(unsigned)
         << "}" << std::endl;

      ss << "PhysicalInstDimOrderDesc {"
         << "id:" << PHYSICAL_INST_LAYOUT_DIM_ID              << delim
         << "inst_uid:unsigned long long:" << sizeof(LgEvent) << delim
         << "dim:unsigned:"             << sizeof(unsigned)   << delim
         << "dim_kind:unsigned:"        << sizeof(unsigned)
         << "}" << std::endl;

      ss << "PhysicalInstanceUsage {"
         << "id:" << PHYSICAL_INST_USAGE_ID                   << delim
         << "inst_uid:unsigned long long:" << sizeof(LgEvent) << delim
         << "op_id:UniqueID:"           << sizeof(UniqueID)   << delim
         << "index_id:unsigned:"        << sizeof(unsigned)   << delim
         << "field_id:unsigned:"        << sizeof(unsigned)
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
         << "provenance:unsigned long long:" << sizeof(unsigned long long)
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
         << "wait_end:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "wait_event:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "MetaWaitInfo {"
         << "id:" << META_WAIT_INFO_ID                       << delim
         << "op_id:UniqueID:"         << sizeof(UniqueID)    << delim
         << "lg_id:unsigned:"         << sizeof(unsigned)    << delim
         << "wait_start:timestamp_t:" << sizeof(timestamp_t) << delim
         << "wait_ready:timestamp_t:" << sizeof(timestamp_t) << delim
         << "wait_end:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "wait_event:unsigned long long:" << sizeof(LgEvent)
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
         << "stop:timestamp_t:"    << sizeof(timestamp_t) << delim
         << "creator:unsigned long long:" << sizeof(LgEvent) << delim
         << "critical:unsigned long long:" << sizeof(LgEvent) << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "ImplicitTaskInfo {"
         << "id:" << IMPLICIT_TASK_INFO_ID                << delim
         << "op_id:UniqueID:"      << sizeof(UniqueID)    << delim
         << "task_id:TaskID:"      << sizeof(TaskID)      << delim
         << "variant_id:VariantID:"<< sizeof(VariantID)   << delim
         << "proc_id:ProcID:"      << sizeof(ProcID)      << delim
         << "create:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "start:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"    << sizeof(timestamp_t) << delim
         << "creator:unsigned long long:" << sizeof(LgEvent) << delim
         << "critical:unsigned long long:" << sizeof(LgEvent) << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
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
         << "gpu_stop:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "creator:unsigned long long:" << sizeof(LgEvent) << delim
         << "critical:unsigned long long:" << sizeof(LgEvent) << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "MetaInfo {"
         << "id:" << META_INFO_ID                         << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)     << delim
         << "lg_id:unsigned:"     << sizeof(unsigned)     << delim
         << "proc_id:ProcID:"     << sizeof(ProcID)       << delim
         << "create:timestamp_t:" << sizeof(timestamp_t)  << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t)  << delim
         << "start:timestamp_t:"  << sizeof(timestamp_t)  << delim
         << "stop:timestamp_t:"   << sizeof(timestamp_t)  << delim
         << "creator:unsigned long long:" << sizeof(LgEvent) << delim
         << "critical:unsigned long long:" << sizeof(LgEvent) << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "MessageInfo {"
         << "id:" << MESSAGE_INFO_ID                      << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)     << delim
         << "lg_id:unsigned:"     << sizeof(unsigned)     << delim
         << "proc_id:ProcID:"     << sizeof(ProcID)       << delim
         << "spawn:timestamp_t:"  << sizeof(timestamp_t)  << delim
         << "create:timestamp_t:" << sizeof(timestamp_t)  << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t)  << delim
         << "start:timestamp_t:"  << sizeof(timestamp_t)  << delim
         << "stop:timestamp_t:"   << sizeof(timestamp_t)  << delim
         << "creator:unsigned long long:" << sizeof(LgEvent) << delim
         << "critical:unsigned long long:" << sizeof(LgEvent) << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "CopyInfo {"
         << "id:" << COPY_INFO_ID                                    << delim
         << "op_id:UniqueID:"          << sizeof(UniqueID)           << delim
         << "size:unsigned long long:" << sizeof(unsigned long long) << delim
         << "create:timestamp_t:"      << sizeof(timestamp_t)        << delim
         << "ready:timestamp_t:"       << sizeof(timestamp_t)        << delim
         << "start:timestamp_t:"       << sizeof(timestamp_t)        << delim
         << "stop:timestamp_t:"        << sizeof(timestamp_t)        << delim
         << "creator:unsigned long long:" << sizeof(LgEvent)      << delim
         << "critical:unsigned long long:" << sizeof(LgEvent)        << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)          << delim
         << "collective:unsigned:"     << sizeof(CollectiveKind)
         << "}" << std::endl;

      ss << "CopyInstInfo {"
         << "id:" << COPY_INST_INFO_ID                           << delim
         << "src:MemID:"                 << sizeof(MemID)        << delim
         << "dst:MemID:"                 << sizeof(MemID)        << delim
         << "src_fid:unsigned:"          << sizeof(FieldID)      << delim
         << "dst_fid:unsigned:"          << sizeof(FieldID)      << delim
         << "src_inst:unsigned long long:"  << sizeof(LgEvent)   << delim
         << "dst_inst:unsigned long long:"  << sizeof(LgEvent)   << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)      << delim
         << "num_hops:unsigned:"       << sizeof(unsigned)           << delim
         << "indirect:bool:"             << sizeof(bool)
         << "}" << std::endl;

      ss << "FillInfo {"
         << "id:" << FILL_INFO_ID                        << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)    << delim
         << "size:unsigned long long:" << sizeof(unsigned long long) << delim
         << "create:timestamp_t:" << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "start:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "creator:unsigned long long:" << sizeof(LgEvent) << delim
         << "critical:unsigned long long:" << sizeof(LgEvent) << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "FillInstInfo {"
         << "id:" << FILL_INST_INFO_ID                           << delim
         << "dst:MemID:"                    << sizeof(MemID)     << delim
         << "fid:unsigned:"                 << sizeof(FieldID)   << delim
         << "dst_inst:unsigned long long:"  << sizeof(LgEvent)   << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "InstTimelineInfo {"
         << "id:" << INST_TIMELINE_INFO_ID                << delim
         << "inst_uid:unsigned long long:" << sizeof(LgEvent) << delim
         << "inst_id:InstID:"          << sizeof(InstID)   << delim
         << "mem_id:MemID:"            << sizeof(MemID)    << delim
         << "size:unsigned long long:" << sizeof(unsigned long long) << delim
         << "op_id:UniqueID:"       << sizeof(UniqueID) << delim
         << "create:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "destroy:timestamp_t:" << sizeof(timestamp_t) << delim
         << "creator:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "PartitionInfo {"
         << "id:" << PARTITION_INFO_ID                          << delim
         << "op_id:UniqueID:"         << sizeof(UniqueID)       << delim
         << "part_op:DepPartOpKind:"  << sizeof(DepPartOpKind)  << delim
         << "create:timestamp_t:"     << sizeof(timestamp_t)    << delim
         << "ready:timestamp_t:"      << sizeof(timestamp_t)    << delim
         << "start:timestamp_t:"      << sizeof(timestamp_t)    << delim
         << "stop:timestamp_t:"       << sizeof(timestamp_t)    << delim
         << "creator:unsigned long long:" << sizeof(LgEvent)    << delim
         << "critical:unsigned long long:" << sizeof(LgEvent)   << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "MapperCallInfo {"
         << "id:" << MAPPER_CALL_INFO_ID                          << delim
         << "mapper_id:MapperID:"   << sizeof(MapperID)           << delim
         << "mapper_proc:ProcID:"   << sizeof(ProcID)             << delim
         << "kind:MappingCallKind:" << sizeof(MappingCallKind)    << delim
         << "op_id:UniqueID:"       << sizeof(UniqueID)           << delim
         << "start:timestamp_t:"    << sizeof(timestamp_t)        << delim
         << "stop:timestamp_t:"     << sizeof(timestamp_t)        << delim
         << "proc_id:ProcID:"       << sizeof(ProcID)             << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "RuntimeCallInfo {"
         << "id:" << RUNTIME_CALL_INFO_ID                      << delim
         << "kind:RuntimeCallKind:" << sizeof(RuntimeCallKind) << delim
         << "start:timestamp_t:"    << sizeof(timestamp_t)     << delim
         << "stop:timestamp_t:"     << sizeof(timestamp_t)     << delim
         << "proc_id:ProcID:"       << sizeof(ProcID)          << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "ApplicationCallInfo {"
         << "id:" << APPLICATION_CALL_INFO_ID                  << delim
         << "provenance:unsigned long long:" << sizeof(unsigned long long) << delim
         << "start:timestamp_t:"    << sizeof(timestamp_t)     << delim
         << "stop:timestamp_t:"     << sizeof(timestamp_t)     << delim
         << "proc_id:ProcID:"       << sizeof(ProcID)          << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "BacktraceDesc {"
         << "id:" << BACKTRACE_DESC_ID                                       << delim
         << "backtrace_id:unsigned long long:" << sizeof(unsigned long long) << delim
         << "backtrace:string:" << "-1"
         << "}" << std::endl;

      ss << "EventWaitInfo {"
         << "id:" << EVENT_WAIT_INFO_ID                         << delim
         << "proc_id:ProcID:" << sizeof(ProcID)                 << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)     << delim
         << "wait_event:unsigned long long:" << sizeof(LgEvent) << delim
         << "backtrace_id:unsigned long long:" << sizeof(unsigned long long)
         << "}" << std::endl;

      ss << "EventMergerInfo {"
         << "id:" << EVENT_MERGER_INFO_ID                       << delim
         << "result:unsigned long long:" << sizeof(LgEvent)     << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)     << delim
         << "performed:timestamp_t:" << sizeof(timestamp_t)     << delim
         << "pre0:unsigned long long:" << sizeof(LgEvent)       << delim
         << "pre1:unsigned long long:" << sizeof(LgEvent)       << delim
         << "pre2:unsigned long long:" << sizeof(LgEvent)       << delim
         << "pre3:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "EventTriggerInfo {"
         << "id:" << EVENT_TRIGGER_INFO_ID                      << delim
         << "result:unsigned long long:" << sizeof(LgEvent)     << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)     << delim
         << "precondition:unsigned long long:" << sizeof(LgEvent) << delim
         << "performed:timestamp_t:" << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "EventPoisonInfo {"
         << "id:" << EVENT_POISON_INFO_ID                       << delim
         << "result:unsigned long long:" << sizeof(LgEvent)     << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)     << delim
         << "performed:timestamp_t:" << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "BarrierArrivalInfo {"
         << "id:" << BARRIER_ARRIVAL_INFO_ID                    << delim
         << "result:unsigned long long:" << sizeof(LgEvent)     << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)     << delim
         << "precondition:unsigned long long:" << sizeof(LgEvent) << delim
         << "performed:timestamp_t:" << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "ReservationAcquireInfo {"
         << "id:" << RESERVATION_ACQUIRE_INFO_ID                << delim
         << "result:unsigned long long:" << sizeof(LgEvent)     << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)     << delim
         << "precondition:unsigned long long:" << sizeof(LgEvent) << delim
         << "performed:timestamp_t:" << sizeof(timestamp_t)     << delim
         << "reservation:unsigned long long:" << sizeof(Reservation)
         << "}" << std::endl;

      ss << "InstanceReadyInfo {"
         << "id:" << INSTANCE_READY_INFO_ID                       << delim
         << "result:unsigned long long:" << sizeof(LgEvent)       << delim
         << "precondition:unsigned long long:" << sizeof(LgEvent) << delim
         << "inst_uid:unsigned long long:" << sizeof(LgEvent)     << delim
         << "performed:timestamp_t:" << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "CompletionQueueInfo {"
         << "id:" << COMPLETION_QUEUE_INFO_ID                   << delim
         << "result:unsigned long long:" << sizeof(LgEvent)     << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)     << delim
         << "performed:timestamp_t:" << sizeof(timestamp_t)     << delim
         << "pre0:unsigned long long:" << sizeof(LgEvent)       << delim
         << "pre1:unsigned long long:" << sizeof(LgEvent)       << delim
         << "pre2:unsigned long long:" << sizeof(LgEvent)       << delim
         << "pre3:unsigned long long:" << sizeof(LgEvent)
         << "}" << std::endl;

      ss << "ProfTaskInfo {"
         << "id:" << PROFTASK_INFO_ID                        << delim
         << "proc_id:ProcID:"         << sizeof(ProcID)      << delim
         << "op_id:UniqueID:"         << sizeof(UniqueID)    << delim
         << "start:timestamp_t:"      << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"       << sizeof(timestamp_t) << delim
         << "creator:unsigned long long:" << sizeof(LgEvent) << delim
         << "fevent:unsigned long long:" << sizeof(LgEvent)  << delim
         << "completion:bool:" << sizeof(bool)
         << "}" << std::endl;

      // An empty line indicates the end of the preamble.
      ss << std::endl;
      std::string preamble = ss.str();

      lp_fwrite(f, preamble.c_str(), strlen(preamble.c_str()));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfDesc::MapperName &mapper_name)
    //--------------------------------------------------------------------------
    {
      int ID = MAPPER_NAME_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(mapper_name.mapper_id),
          sizeof(mapper_name.mapper_id));
      lp_fwrite(f, (char*)&(mapper_name.mapper_proc),
          sizeof(mapper_name.mapper_proc));
      lp_fwrite(f, mapper_name.name, strlen(mapper_name.name) + 1);
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
                                    const LegionProfDesc::RuntimeConfig &config)
    //--------------------------------------------------------------------------
    {
      int ID = RUNTIME_CONFIG_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&config.debug, sizeof(config.debug));
      lp_fwrite(f, (char*)&config.spy, sizeof(config.spy));
      lp_fwrite(f, (char*)&config.gc, sizeof(config.gc));
      lp_fwrite(f, (char*)&config.inorder, sizeof(config.inorder));
      lp_fwrite(f, (char*)&config.safe_mapper, sizeof(config.safe_mapper));
      lp_fwrite(f, (char*)&config.safe_runtime, sizeof(config.safe_runtime));
      lp_fwrite(f, (char*)&config.safe_ctrlrepl, sizeof(config.safe_ctrlrepl));
      lp_fwrite(f, (char*)&config.part_checks, sizeof(config.part_checks));
      lp_fwrite(f, (char*)&config.bounds_checks, sizeof(config.bounds_checks));
      lp_fwrite(f, (char*)&config.resilient, sizeof(config.resilient));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                      const LegionProfDesc::MachineDesc
				      &machine_desc)
    //--------------------------------------------------------------------------
    {
      int ID = MACHINE_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(machine_desc.node_id),
		sizeof(machine_desc.node_id));
      lp_fwrite(f, (char*)&(machine_desc.num_nodes),
		sizeof(machine_desc.num_nodes));
      lp_fwrite(f, (char*)&(machine_desc.version),
                sizeof(machine_desc.version));
      lp_fwrite(f, machine_desc.process_info.hostname, strlen(machine_desc.process_info.hostname) + 1);
      lp_fwrite(f, (char*)&(machine_desc.process_info.hostid),
                sizeof(machine_desc.process_info.hostid));
      lp_fwrite(f, (char*)&(machine_desc.process_info.processid),
                sizeof(machine_desc.process_info.processid));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                      const LegionProfDesc::CalibrationErr
				      &calibration_err)
    //--------------------------------------------------------------------------
    {
      int ID = CALIBRATION_ERR_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(calibration_err.calibration_err),
		sizeof(calibration_err.calibration_err));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                         const LegionProfDesc::Provenance &prov)
    //--------------------------------------------------------------------------
    {
      int ID = PROVENANCE_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(prov.pid), sizeof(prov.pid));
      lp_fwrite(f, prov.provenance, prov.size + 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                      const LegionProfDesc::ZeroTime
				      &zero_time)
    //--------------------------------------------------------------------------
    {
      int ID = ZERO_TIME_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(zero_time.zero_time),
		sizeof(zero_time.zero_time));
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
      lp_fwrite(f, (char*)&(phy_instance_rdesc.inst_uid.id), 
                sizeof(phy_instance_rdesc.inst_uid.id));
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
      lp_fwrite(f, (char*)&(phy_instance_dim_order_rdesc.inst_uid.id),
                sizeof(phy_instance_dim_order_rdesc.inst_uid.id));
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
      lp_fwrite(f, (char*)&(phy_instance_layout_rdesc.inst_uid.id),
                sizeof(phy_instance_layout_rdesc.inst_uid.id));
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
                         const LegionProfInstance::PhysicalInstanceUsage &usage)
    //--------------------------------------------------------------------------
    {
      int ID = PHYSICAL_INST_USAGE_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(usage.inst_uid.id), sizeof(usage.inst_uid.id));
      lp_fwrite(f, (char*)&(usage.op_id), sizeof(UniqueID));
      lp_fwrite(f, (char*)&(usage.index), sizeof(unsigned));
      lp_fwrite(f, (char*)&(usage.field), sizeof(unsigned));
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
                                  const LegionProfDesc::TaskKind& task_kind)
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
                            const LegionProfDesc::TaskVariant& task_variant)
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
      lp_fwrite(f, (char*)&(operation_instance.provenance),
                sizeof(operation_instance.provenance));
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
      lp_fwrite(f, (char*)&(wait_info.wait_event), sizeof(wait_info.wait_event));
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
      lp_fwrite(f, (char*)&(wait_info.wait_event), sizeof(wait_info.wait_event));
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
      lp_fwrite(f, (char*)&(wait_info.wait_event), sizeof(wait_info.wait_event));
    }
 
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                   const LegionProfInstance::TaskInfo& task_info, bool implicit)
    //--------------------------------------------------------------------------
    {
      int ID = implicit ? IMPLICIT_TASK_INFO_ID : TASK_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(task_info.op_id),     sizeof(task_info.op_id));
      lp_fwrite(f, (char*)&(task_info.task_id),   sizeof(task_info.task_id));
      lp_fwrite(f, (char*)&(task_info.variant_id),sizeof(task_info.variant_id));
      lp_fwrite(f, (char*)&(task_info.proc_id),   sizeof(task_info.proc_id));
      lp_fwrite(f, (char*)&(task_info.create),    sizeof(task_info.create));
      lp_fwrite(f, (char*)&(task_info.ready),     sizeof(task_info.ready));
      lp_fwrite(f, (char*)&(task_info.start),     sizeof(task_info.start));
      lp_fwrite(f, (char*)&(task_info.stop),      sizeof(task_info.stop));
      lp_fwrite(f, (char*)&(task_info.creator),   sizeof(task_info.creator));
      lp_fwrite(f, (char*)&(task_info.critical),  sizeof(task_info.critical));
      lp_fwrite(f, (char*)&(task_info.finish_event),
                                                sizeof(task_info.finish_event));
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
      lp_fwrite(f, (char*)&(task_info.creator),   sizeof(task_info.creator));
      lp_fwrite(f, (char*)&(task_info.critical),  sizeof(task_info.critical));
      lp_fwrite(f, (char*)&(task_info.finish_event),
                                                sizeof(task_info.finish_event));
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
      lp_fwrite(f, (char*)&(meta_info.creator), sizeof(meta_info.creator));
      lp_fwrite(f, (char*)&(meta_info.critical),sizeof(meta_info.critical));
      lp_fwrite(f, (char*)&(meta_info.finish_event),
                                                sizeof(meta_info.finish_event));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                               const LegionProfInstance::MessageInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      int ID = MESSAGE_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(meta_info.op_id),   sizeof(meta_info.op_id));
      lp_fwrite(f, (char*)&(meta_info.lg_id),   sizeof(meta_info.lg_id));
      lp_fwrite(f, (char*)&(meta_info.proc_id), sizeof(meta_info.proc_id));
      lp_fwrite(f, (char*)&(meta_info.spawn),   sizeof(meta_info.spawn));
      lp_fwrite(f, (char*)&(meta_info.create),  sizeof(meta_info.create));
      lp_fwrite(f, (char*)&(meta_info.ready),   sizeof(meta_info.ready));
      lp_fwrite(f, (char*)&(meta_info.start),   sizeof(meta_info.start));
      lp_fwrite(f, (char*)&(meta_info.stop),    sizeof(meta_info.stop));
      lp_fwrite(f, (char*)&(meta_info.creator), sizeof(meta_info.creator));
      lp_fwrite(f, (char*)&(meta_info.critical),sizeof(meta_info.critical));
      lp_fwrite(f, (char*)&(meta_info.finish_event),
                                                sizeof(meta_info.finish_event));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                          const LegionProfInstance::CopyInfo& copy_info)
    //--------------------------------------------------------------------------
    {
      int ID = COPY_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));

      lp_fwrite(f, (char*)&(copy_info.op_id),  sizeof(copy_info.op_id));
      lp_fwrite(f, (char*)&(copy_info.size),   sizeof(copy_info.size));
      lp_fwrite(f, (char*)&(copy_info.create), sizeof(copy_info.create));
      lp_fwrite(f, (char*)&(copy_info.ready),  sizeof(copy_info.ready));
      lp_fwrite(f, (char*)&(copy_info.start),  sizeof(copy_info.start));
      lp_fwrite(f, (char*)&(copy_info.stop),   sizeof(copy_info.stop));
      lp_fwrite(f, (char*)&(copy_info.creator),sizeof(copy_info.creator));
      lp_fwrite(f, (char*)&(copy_info.critical),sizeof(copy_info.critical));
      lp_fwrite(f, (char*)&(copy_info.fevent),sizeof(copy_info.fevent.id));
      lp_fwrite(f, (char*)&(copy_info.collective),sizeof(copy_info.collective));
      for (std::vector<LegionProfInstance::CopyInstInfo>::const_iterator it =
           copy_info.inst_infos.begin(); it != copy_info.inst_infos.end(); it++)
        serialize(*it, copy_info);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                              const LegionProfInstance::CopyInstInfo &copy_inst,
                              const LegionProfInstance::CopyInfo &copy_info)
    //--------------------------------------------------------------------------
    {
      int ID = COPY_INST_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(copy_inst.src), sizeof(copy_inst.src));
      lp_fwrite(f, (char*)&(copy_inst.dst), sizeof(copy_inst.dst));
      lp_fwrite(f, (char*)&(copy_inst.src_fid), sizeof(copy_inst.src_fid));
      lp_fwrite(f, (char*)&(copy_inst.dst_fid), sizeof(copy_inst.dst_fid));
      lp_fwrite(f, (char*)&(copy_inst.src_inst_uid.id),
          sizeof(copy_inst.src_inst_uid.id));
      lp_fwrite(f, (char*)&(copy_inst.dst_inst_uid.id),
          sizeof(copy_inst.dst_inst_uid.id));
      lp_fwrite(f, (char*)&(copy_info.fevent),sizeof(copy_info.fevent.id));
      lp_fwrite(f, (char*)&(copy_inst.num_hops), sizeof(copy_inst.num_hops));
      lp_fwrite(f, (char*)&(copy_inst.indirect), sizeof(copy_inst.indirect));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::FillInfo& fill_info)
    //--------------------------------------------------------------------------
    {
      int ID = FILL_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(fill_info.op_id),  sizeof(fill_info.op_id));
      lp_fwrite(f, (char*)&(fill_info.size),   sizeof(fill_info.size));
      lp_fwrite(f, (char*)&(fill_info.create), sizeof(fill_info.create));
      lp_fwrite(f, (char*)&(fill_info.ready),  sizeof(fill_info.ready));
      lp_fwrite(f, (char*)&(fill_info.start),  sizeof(fill_info.start));
      lp_fwrite(f, (char*)&(fill_info.stop),   sizeof(fill_info.stop));
      lp_fwrite(f, (char*)&(fill_info.creator),sizeof(fill_info.creator));
      lp_fwrite(f, (char*)&(fill_info.critical),sizeof(fill_info.critical));
      lp_fwrite(f, (char*)&(fill_info.fevent), sizeof(fill_info.fevent.id));
      for (std::vector<LegionProfInstance::FillInstInfo>::const_iterator it =
           fill_info.inst_infos.begin(); it != fill_info.inst_infos.end(); it++)
        serialize(*it, fill_info);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                              const LegionProfInstance::FillInstInfo &fill_inst,
                              const LegionProfInstance::FillInfo &fill_info)
    //--------------------------------------------------------------------------
    {
      int ID = FILL_INST_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(fill_inst.dst), sizeof(fill_inst.dst));
      lp_fwrite(f, (char*)&(fill_inst.fid),    sizeof(fill_inst.fid));
      lp_fwrite(f, (char*)&(fill_inst.dst_inst_uid.id),
          sizeof(fill_inst.dst_inst_uid.id));
      lp_fwrite(f, (char*)&(fill_info.fevent),sizeof(fill_info.fevent.id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                 const LegionProfInstance::InstTimelineInfo& inst_timeline_info)
    //--------------------------------------------------------------------------
    {
      int ID = INST_TIMELINE_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(inst_timeline_info.inst_uid.id),
                sizeof(inst_timeline_info.inst_uid.id));
      lp_fwrite(f, (char*)&(inst_timeline_info.inst_id),
                sizeof(inst_timeline_info.inst_id));
      lp_fwrite(f, (char*)&(inst_timeline_info.mem_id),  
                sizeof(inst_timeline_info.mem_id));
      lp_fwrite(f, (char*)&(inst_timeline_info.size),    
                sizeof(inst_timeline_info.size));
      lp_fwrite(f, (char*)&(inst_timeline_info.op_id),
                sizeof(inst_timeline_info.op_id));
      lp_fwrite(f, (char*)&(inst_timeline_info.create),  
                sizeof(inst_timeline_info.create));
      lp_fwrite(f, (char*)&(inst_timeline_info.ready),  
                sizeof(inst_timeline_info.ready));
      lp_fwrite(f, (char*)&(inst_timeline_info.destroy), 
                sizeof(inst_timeline_info.destroy));
      lp_fwrite(f, (char*)&(inst_timeline_info.creator),
                sizeof(inst_timeline_info.creator));
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
      lp_fwrite(f, (char*)&(partition_info.creator),
                sizeof(partition_info.creator));
      lp_fwrite(f, (char*)&(partition_info.critical),
                sizeof(partition_info.critical));
      lp_fwrite(f, (char*)&(partition_info.fevent),
                sizeof(partition_info.fevent));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                     const LegionProfInstance::MapperCallInfo& mapper_call_info)
    //--------------------------------------------------------------------------
    {
      int ID = MAPPER_CALL_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(mapper_call_info.mapper),
                sizeof(mapper_call_info.mapper));
      lp_fwrite(f, (char*)&(mapper_call_info.mapper_proc),
                sizeof(mapper_call_info.mapper_proc));
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
      lp_fwrite(f, (char*)&(mapper_call_info.finish_event),
                sizeof(mapper_call_info.finish_event));
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
      lp_fwrite(f, (char*)&(runtime_call_info.finish_event),
                sizeof(runtime_call_info.finish_event));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
           const LegionProfInstance::ApplicationCallInfo& application_call_info)
    //--------------------------------------------------------------------------
    {
      int ID = APPLICATION_CALL_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(application_call_info.pid), 
                sizeof(application_call_info.pid));
      lp_fwrite(f, (char*)&(application_call_info.start),   
                sizeof(application_call_info.start));
      lp_fwrite(f, (char*)&(application_call_info.stop),    
                sizeof(application_call_info.stop));
      lp_fwrite(f, (char*)&(application_call_info.proc_id), 
                sizeof(application_call_info.proc_id));
      lp_fwrite(f, (char*)&(application_call_info.finish_event),
                sizeof(application_call_info.finish_event));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                      const LegionProfDesc::ProcDesc& proc_desc)
    //--------------------------------------------------------------------------
    {
      int ID = PROC_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(proc_desc.proc_id), sizeof(proc_desc.proc_id));
      lp_fwrite(f, (char*)&(proc_desc.kind),    sizeof(proc_desc.kind));
#ifdef LEGION_USE_CUDA
      unsigned uuid_size = Realm::Cuda::UUID_SIZE;
      lp_fwrite(f, (char*)&(uuid_size), sizeof(uuid_size));
      for (size_t i=0; i<Realm::Cuda::UUID_SIZE; i++) {
        lp_fwrite(f, (char*)&(proc_desc.cuda_device_uuid[i]),
            sizeof(char));
      }
#else
      unsigned uuid_size = 16;
      lp_fwrite(f, (char*)&(uuid_size), sizeof(uuid_size));
      char uuid_str[16] = {0};
      for (size_t i=0; i<uuid_size; i++) {
        lp_fwrite(f, (char*)&(uuid_str[i]),
            sizeof(char));
      }
#endif
    }
    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                        const LegionProfDesc::MemDesc& mem_desc)
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
                                          const LegionProfDesc::ProcMemDesc &pm)
    //--------------------------------------------------------------------------
    {
      int ID = PROC_MEM_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*) &(pm.proc_id), sizeof(pm.proc_id));
      lp_fwrite(f, (char*) &(pm.mem_id), sizeof(pm.mem_id));
      lp_fwrite(f, (char*) &(pm.bandwidth), sizeof(pm.bandwidth));
      lp_fwrite(f, (char*) &(pm.latency), sizeof(pm.latency));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                            const LegionProfDesc::Backtrace &bt)
    //--------------------------------------------------------------------------
    {
      int ID = BACKTRACE_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&bt.id, sizeof(bt.id));
      lp_fwrite(f, bt.backtrace, strlen(bt.backtrace) + 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                  const LegionProfInstance::EventWaitInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = EVENT_WAIT_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&info.proc_id, sizeof(info.proc_id));
      lp_fwrite(f, (char*)&info.fevent.id, sizeof(info.fevent.id));
      lp_fwrite(f, (char*)&info.event.id, sizeof(info.event.id));
      lp_fwrite(f, (char*)&info.backtrace_id, sizeof(info.backtrace_id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                const LegionProfInstance::EventMergerInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = EVENT_MERGER_INFO_ID;
      for (unsigned offset = 0; offset < info.preconditions.size(); offset += 4)
      {
        lp_fwrite(f, (char*)&ID, sizeof(ID));
        lp_fwrite(f, (char*)&info.result.id, sizeof(info.result.id));
        lp_fwrite(f, (char*)&info.fevent.id, sizeof(info.fevent.id));
        lp_fwrite(f, (char*)&info.performed, sizeof(info.performed));
        lp_fwrite(f, (char*)&info.preconditions[offset].id,
            sizeof (info.preconditions[offset].id));
        for (unsigned idx = 1; idx < 4; idx++)
        {
          if ((offset+idx) < info.preconditions.size())
            lp_fwrite(f, (char*)&info.preconditions[offset+idx].id,
                sizeof(info.preconditions[offset+idx].id));
          else
            lp_fwrite(f, (char*)&LgEvent::NO_LG_EVENT, 
                sizeof(LgEvent::NO_LG_EVENT)); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                               const LegionProfInstance::EventTriggerInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = EVENT_TRIGGER_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&info.result.id, sizeof(info.result.id));
      lp_fwrite(f, (char*)&info.fevent.id, sizeof(info.fevent.id));
      lp_fwrite(f, (char*)&info.precondition.id, sizeof(info.precondition.id));
      lp_fwrite(f, (char*)&info.performed, sizeof(info.performed));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                                const LegionProfInstance::EventPoisonInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = EVENT_POISON_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&info.result.id, sizeof(info.result.id));
      lp_fwrite(f, (char*)&info.fevent.id, sizeof(info.fevent.id));
      lp_fwrite(f, (char*)&info.performed, sizeof(info.performed));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                             const LegionProfInstance::BarrierArrivalInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = BARRIER_ARRIVAL_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&info.result.id, sizeof(info.result.id));
      lp_fwrite(f, (char*)&info.fevent.id, sizeof(info.fevent.id));
      lp_fwrite(f, (char*)&info.precondition.id, sizeof(info.precondition.id));
      lp_fwrite(f, (char*)&info.performed, sizeof(info.performed));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                         const LegionProfInstance::ReservationAcquireInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = RESERVATION_ACQUIRE_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&info.result.id, sizeof(info.result.id));
      lp_fwrite(f, (char*)&info.fevent.id, sizeof(info.fevent.id));
      lp_fwrite(f, (char*)&info.precondition.id, sizeof(info.precondition.id));
      lp_fwrite(f, (char*)&info.performed, sizeof(info.performed));
      lp_fwrite(f, (char*)&info.reservation.id, sizeof(info.reservation.id));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                              const LegionProfInstance::InstanceReadyInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = INSTANCE_READY_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&info.result.id, sizeof(info.result.id));
      lp_fwrite(f, (char*)&info.precondition.id, sizeof(info.precondition.id));
      lp_fwrite(f, (char*)&info.unique.id, sizeof(info.unique.id));
      lp_fwrite(f, (char*)&info.performed, sizeof(info.performed));
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                            const LegionProfInstance::CompletionQueueInfo &info)
    //--------------------------------------------------------------------------
    {
      int ID = EVENT_MERGER_INFO_ID;
      for (unsigned offset = 0; offset < info.preconditions.size(); offset += 4)
      {
        lp_fwrite(f, (char*)&ID, sizeof(ID));
        lp_fwrite(f, (char*)&info.result.id, sizeof(info.result.id));
        lp_fwrite(f, (char*)&info.fevent.id, sizeof(info.fevent.id));
        lp_fwrite(f, (char*)&info.performed, sizeof(info.performed));
        lp_fwrite(f, (char*)&info.preconditions[offset].id,
            sizeof (info.preconditions[offset].id));
        for (unsigned idx = 1; idx < 4; idx++)
        {
          if ((offset+idx) < info.preconditions.size())
            lp_fwrite(f, (char*)&info.preconditions[offset+idx].id,
                sizeof(info.preconditions[offset+idx].id));
          else
            lp_fwrite(f, (char*)&LgEvent::NO_LG_EVENT, 
                sizeof(LgEvent::NO_LG_EVENT)); 
        }
      }
    }

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
      lp_fwrite(f, (char*)&(proftask_info.creator),
                sizeof(proftask_info.creator));
      lp_fwrite(f, (char*)&(proftask_info.finish_event),
                                            sizeof(proftask_info.finish_event));
      lp_fwrite(f, (char*)&(proftask_info.completion),
                                            sizeof(proftask_info.completion));
    }

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
      log_prof.print("Physical Inst Region Desc "  IDFMT " %llu %u %u",
                     phy_instance_rdesc.inst_uid.id,
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
      log_prof.print("Physical Inst Dim Order Desc " IDFMT " %u %u",
                     phy_instance_dim_order_rdesc.inst_uid.id,
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
      log_prof.print("Physical Inst Layout Desc " IDFMT " %u %u %u %u "
                     "%u",
                     phy_instance_layout_rdesc.inst_uid.id,
                     phy_instance_layout_rdesc.field_id,
                     phy_instance_layout_rdesc.fspace_id,
                     phy_instance_layout_rdesc.has_align,
                     phy_instance_layout_rdesc.eqk,
                     phy_instance_layout_rdesc.alignment
                     );
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                         const LegionProfInstance::PhysicalInstanceUsage &usage)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Physical Inst Usage " IDFMT " %llu %u %u",
                      usage.inst_uid.id, usage.op_id, usage.index, usage.field);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfDesc::MapperName &mapper_name)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mapper Name %u " IDFMT "%s",
          mapper_name.mapper_id, mapper_name.mapper_proc, mapper_name.name);
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
                                    const LegionProfDesc::RuntimeConfig &config)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Runtime Config %d %d %d %d %d %d %d %d %d %d",
          config.debug ? 1 : 0, config.spy ? 1 : 0, config.gc ? 1 : 0,
          config.inorder ? 1 : 0, config.safe_mapper ? 1 : 0,
          config.safe_runtime ? 1 : 0, config.safe_ctrlrepl ? 1 : 0,
          config.part_checks ? 1 : 0, config.bounds_checks ? 1 : 0,
          config.resilient ? 1 : 0);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                      const LegionProfDesc::MachineDesc
				      &machine_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Machine Desc %d %d %d %s %llu %d",
                     machine_desc.node_id, machine_desc.num_nodes,
                     machine_desc.version,
                     machine_desc.process_info.hostname,
                     (unsigned long long)machine_desc.process_info.hostid,
                     machine_desc.process_info.processid);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                      const LegionProfDesc::CalibrationErr
				      &calibration_err)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Calibration Err %lld", calibration_err.calibration_err);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                      const LegionProfDesc::ZeroTime
				      &zero_time)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Zero Time %lld", zero_time.zero_time);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                         const LegionProfDesc::Provenance &prov)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Provenance %lld %s", prov.pid, prov.provenance);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfDesc::TaskKind &task_kind)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Task Kind %u %s %d", task_kind.task_id, 
                      task_kind.name, (task_kind.overwrite ? 1 : 0));
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                            const LegionProfDesc::TaskVariant& task_variant)
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
      log_prof.print("Prof Operation %llu %llu %u %lld", 
          op_inst.op_id, op_inst.parent_id, op_inst.kind, op_inst.provenance);
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
      log_prof.print("Prof Task Wait Info %llu %u %u %llu %llu %llu " IDFMT,
                task_info.op_id, task_info.task_id, task_info.variant_id, 
                wait_info.wait_start, wait_info.wait_ready,
                wait_info.wait_end, wait_info.wait_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::WaitInfo wait_info,
                              const LegionProfInstance::GPUTaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Task Wait Info %llu %u %u %llu %llu %llu " IDFMT,
                task_info.op_id, task_info.task_id, task_info.variant_id,
                wait_info.wait_start, wait_info.wait_ready,
                wait_info.wait_end, wait_info.wait_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::WaitInfo wait_info, 
                                  const LegionProfInstance::MetaInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Meta Wait Info %llu %u %llu %llu %llu " IDFMT,
                   meta_info.op_id, meta_info.lg_id, wait_info.wait_start, 
                   wait_info.wait_ready, wait_info.wait_end,
                   wait_info.wait_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                   const LegionProfInstance::TaskInfo& task_info, bool implicit)
    //--------------------------------------------------------------------------
    {
      if (implicit)
        log_prof.print("Prof Implicit Task Info %llu %u %u " IDFMT " %llu %llu "
                       "%llu %llu " IDFMT " " IDFMT " " IDFMT "",
                       task_info.op_id, task_info.task_id, task_info.variant_id,
                       task_info.proc_id, task_info.create, task_info.ready, 
                       task_info.start, task_info.stop, task_info.creator.id,
                       task_info.critical.id, task_info.finish_event.id);
      else
        log_prof.print("Prof Task Info %llu %u %u " IDFMT " %llu %llu %llu %llu"
                       " " IDFMT " " IDFMT " " IDFMT "",
                       task_info.op_id, task_info.task_id, task_info.variant_id,
                       task_info.proc_id, task_info.create, task_info.ready, 
                       task_info.start, task_info.stop, task_info.creator.id,
                       task_info.critical.id, task_info.finish_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::GPUTaskInfo& task_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof GPU Task Info %llu %u %u " IDFMT " %llu %llu %llu "
                     "%llu %llu %llu " IDFMT " " IDFMT " " IDFMT "",
                     task_info.op_id, task_info.task_id, task_info.variant_id,
                     task_info.proc_id, task_info.create, task_info.ready,
                     task_info.start, task_info.stop, task_info.gpu_start,
		     task_info.gpu_stop, task_info.creator.id, 
                     task_info.critical.id, task_info.finish_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::MetaInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Meta Info %llu %u " IDFMT " %llu %llu %llu %llu "
          IDFMT " " IDFMT " " IDFMT "",
         meta_info.op_id, meta_info.lg_id, meta_info.proc_id,
         meta_info.create, meta_info.ready, meta_info.start, meta_info.stop,
         meta_info.creator.id, meta_info.critical.id,
         meta_info.finish_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                               const LegionProfInstance::MessageInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Message Info %llu %u " IDFMT " %llu %llu %llu %llu "
          "%llu " IDFMT " " IDFMT " " IDFMT "",
         meta_info.op_id, meta_info.lg_id, meta_info.proc_id, meta_info.spawn,
         meta_info.create, meta_info.ready, meta_info.start, meta_info.stop,
         meta_info.creator.id, meta_info.critical.id,
         meta_info.finish_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::CopyInfo& copy_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Copy Info %llu %llu %llu %llu %llu %llu " IDFMT " "
                     IDFMT " " IDFMT " %u", copy_info.op_id, copy_info.size, 
                     copy_info.create, copy_info.ready, copy_info.start,
                     copy_info.stop, copy_info.creator.id,copy_info.critical.id,
                     copy_info.fevent.id, copy_info.collective);
      for (std::vector<LegionProfInstance::CopyInstInfo>::const_iterator it =
           copy_info.inst_infos.begin(); it != copy_info.inst_infos.end(); it++)
        serialize(*it, copy_info);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::CopyInstInfo& copy_inst,
                              const LegionProfInstance::CopyInfo &copy_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Copy Inst Info " IDFMT " " IDFMT " %d %d " IDFMT " " 
                     IDFMT " " IDFMT " %u %u", copy_inst.src, copy_inst.dst,
                     copy_inst.src_fid, copy_inst.dst_fid,
                     copy_inst.src_inst_uid.id, copy_inst.dst_inst_uid.id,
                     copy_info.fevent.id, copy_inst.num_hops,
                     copy_inst.indirect ? 1 : 0); 
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::FillInfo& fill_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Fill Info %llu %llu %llu %llu %llu %llu " IDFMT " "
          IDFMT " " IDFMT, fill_info.op_id, fill_info.size, fill_info.create, 
          fill_info.ready, fill_info.start, fill_info.stop,
          fill_info.creator.id, fill_info.critical.id, fill_info.fevent.id);
      for (std::vector<LegionProfInstance::FillInstInfo>::const_iterator it =
           fill_info.inst_infos.begin(); it != fill_info.inst_infos.end(); it++)
        serialize(*it, fill_info);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::FillInstInfo& fill_inst,
                              const LegionProfInstance::FillInfo &fill_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Fill Inst Info " IDFMT " %d " IDFMT " " IDFMT,
                      fill_inst.dst, fill_inst.fid, fill_inst.dst_inst_uid.id,
                      fill_info.fevent.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                 const LegionProfInstance::InstTimelineInfo& inst_timeline_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Inst Timeline " IDFMT " " IDFMT " " IDFMT 
                     " %llu %llu %llu %llu %llu",
         inst_timeline_info.inst_uid.id, inst_timeline_info.inst_id,
         inst_timeline_info.mem_id, inst_timeline_info.size,
         inst_timeline_info.op_id, inst_timeline_info.create,
         inst_timeline_info.ready, inst_timeline_info.destroy);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                        const LegionProfInstance::PartitionInfo& partition_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Partition Timeline %llu %d %llu %llu %llu %llu %llu "
                     IDFMT " " IDFMT "", partition_info.op_id, 
                     partition_info.part_op, partition_info.create,
                     partition_info.create, partition_info.start,
                     partition_info.stop, partition_info.creator.id,
                     partition_info.critical.id, partition_info.fevent.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                     const LegionProfInstance::MapperCallInfo& mapper_call_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mapper Call Info %u " IDFMT " %u " IDFMT 
                     " %llu %llu %llu " IDFMT,
        mapper_call_info.mapper, mapper_call_info.proc_id,
        mapper_call_info.kind, mapper_call_info.proc_id, mapper_call_info.op_id,
        mapper_call_info.start, mapper_call_info.stop, 
        mapper_call_info.finish_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                   const LegionProfInstance::RuntimeCallInfo& runtime_call_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Runtime Call Info %u " IDFMT " %llu %llu " IDFMT,
                     runtime_call_info.kind, runtime_call_info.proc_id, 
                     runtime_call_info.start, runtime_call_info.stop,
                     runtime_call_info.finish_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
           const LegionProfInstance::ApplicationCallInfo& application_call_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Application Call Info %llu " IDFMT 
                     " %llu %llu " IDFMT,
                     application_call_info.pid, application_call_info.proc_id,
                     application_call_info.start, application_call_info.stop,
                     application_call_info.finish_event.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                 const LegionProfDesc::ProcDesc &proc_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Proc Desc " IDFMT " %d",
                     proc_desc.proc_id, proc_desc.kind);
#ifdef LEGION_USE_CUDA
      if (proc_desc.kind == Processor::TOC_PROC) {
        char uuid_str[Realm::Cuda::UUID_SIZE];
        for (size_t i=0; i<Realm::Cuda::UUID_SIZE; i++) {
          sprintf(&uuid_str[i], "%x", proc_desc.cuda_device_uuid[i] & 0xFF);
        }

        log_prof.print("Prof CUDA Proc Desc %s", uuid_str);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                   const LegionProfDesc::MemDesc &mem_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mem Desc " IDFMT " %d %llu",
                      mem_desc.mem_id, mem_desc.kind, mem_desc.capacity);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                    const LegionProfDesc::ProcMemDesc &pm)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Mem Proc Affinity Desc " IDFMT " " IDFMT " %u %u",
		     pm.proc_id, pm.mem_id, pm.bandwidth, pm.latency);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                            const LegionProfDesc::Backtrace &bt)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Backtrace Desc %lld %s", bt.id, bt.backtrace);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::EventWaitInfo &info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Event Wait Info " IDFMT " " IDFMT " " IDFMT " %lld",
          info.proc_id, info.fevent.id, info.event.id, info.backtrace_id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                const LegionProfInstance::EventMergerInfo &info)
    //--------------------------------------------------------------------------
    {
      for (unsigned offset = 0; offset < info.preconditions.size(); offset += 4)
      {
        log_prof.print("Prof Event Merger Info " IDFMT " " IDFMT " %llu "
            IDFMT " " IDFMT " " IDFMT " " IDFMT, info.result.id, 
            info.fevent.id, info.performed, info.preconditions[offset].id,
            (offset+1) < info.preconditions.size() ? 
              info.preconditions[offset+1].id : 0,
            (offset+2) < info.preconditions.size() ? 
              info.preconditions[offset+2].id : 0,
            (offset+3) < info.preconditions.size() ? 
              info.preconditions[offset+3].id : 0);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                               const LegionProfInstance::EventTriggerInfo &info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Event Trigger Info " IDFMT " " IDFMT " " IDFMT 
                      " %lld", info.result.id, info.fevent.id,
                      info.precondition.id, info.performed);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                const LegionProfInstance::EventPoisonInfo &info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Event Poison Info " IDFMT " " IDFMT " %lld",
          info.result.id, info.fevent.id, info.performed);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                             const LegionProfInstance::BarrierArrivalInfo &info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Barrier Arrival Info " IDFMT " " IDFMT " " IDFMT
          " %lld", info.result.id, info.fevent.id, info.precondition.id,
          info.performed);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                         const LegionProfInstance::ReservationAcquireInfo &info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Reservation Acquire Info " IDFMT " " IDFMT " " IDFMT
          " %lld " IDFMT, info.result.id, info.fevent.id, info.precondition.id,
          info.performed, info.reservation.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                              const LegionProfInstance::InstanceReadyInfo &info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Instance Ready Info " IDFMT " " IDFMT " " IDFMT
          " %lld", info.result.id, info.precondition.id, info.unique.id,
          info.performed);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                            const LegionProfInstance::CompletionQueueInfo &info)
    //--------------------------------------------------------------------------
    {
      for (unsigned offset = 0; offset < info.preconditions.size(); offset += 4)
      {
        log_prof.print("Prof Completion Queue Info " IDFMT " " IDFMT " %llu "
            IDFMT " " IDFMT " " IDFMT " " IDFMT, info.result.id, 
            info.fevent.id, info.performed, info.preconditions[offset].id,
            (offset+1) < info.preconditions.size() ? 
              info.preconditions[offset+1].id : 0,
            (offset+2) < info.preconditions.size() ? 
              info.preconditions[offset+2].id : 0,
            (offset+3) < info.preconditions.size() ? 
              info.preconditions[offset+3].id : 0);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                          const LegionProfInstance::ProfTaskInfo& proftask_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof ProfTask Info " IDFMT " %llu %llu %llu " 
                     IDFMT " " IDFMT " %d", proftask_info.proc_id, 
                     proftask_info.op_id, proftask_info.start,
                     proftask_info.stop, proftask_info.creator.id,
                     proftask_info.finish_event.id,
                     proftask_info.completion ? 1 : 0);
    }

    //--------------------------------------------------------------------------
    LegionProfASCIISerializer::~LegionProfASCIISerializer()
    //--------------------------------------------------------------------------
    {
    }

  }; // namespace Internal
}; // namespace Legion

