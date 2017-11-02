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

      ss << "MessageDesc {" 
         << "id:" << MESSAGE_DESC_ID                << delim
         << "kind:unsigned:"     << sizeof(unsigned) << delim
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

      ss << "MemDesc {" 
         << "id:" << MEM_DESC_ID                               << delim
         << "mem_id:MemID:"                << sizeof(MemID)    << delim
         << "kind:MemKind:"                << sizeof(MemKind)  << delim
         << "capacity:unsigned long long:" << sizeof(unsigned long long)
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
         << "kind:unsigned:"  << sizeof(unsigned)
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
         << "variant_id:UniqueID:"    << sizeof(UniqueID)    << delim
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
         << "variant_id:UniqueID:" << sizeof(UniqueID)    << delim
         << "proc_id:ProcID:"      << sizeof(ProcID)      << delim
         << "create:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "start:timestamp_t:"   << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"    << sizeof(timestamp_t)
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
         << "stop:timestamp_t:"        << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "FillInfo {"
         << "id:" << FILL_INFO_ID                        << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)    << delim
         << "dst:MemID:"          << sizeof(MemID)       << delim
         << "create:timestamp_t:" << sizeof(timestamp_t) << delim
         << "ready:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "start:timestamp_t:"  << sizeof(timestamp_t) << delim
         << "stop:timestamp_t:"   << sizeof(timestamp_t)
         << "}" << std::endl;

      ss << "InstCreateInfo {"
         << "id:" << INST_CREATE_INFO_ID                 << delim
         << "op_id:UniqueID:"     << sizeof(UniqueID)    << delim
         << "inst_id:InstID:"     << sizeof(InstID)      << delim
         << "create:timestamp_t:" << sizeof(timestamp_t)
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
         << "}" << std::endl;

      ss << "MessageInfo {"
         << "id:" << MESSAGE_INFO_ID                           << delim
         << "kind:MessageKind:"  << sizeof(MessageKind)        << delim
         << "start:timestamp_t:" << sizeof(timestamp_t)        << delim
         << "stop:timestamp_t:"  << sizeof(timestamp_t)        << delim
         << "proc_id:ProcID:"    << sizeof(ProcID)
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
                                const LegionProfDesc::MessageDesc &message_desc)
    //--------------------------------------------------------------------------
    {
      // XXX: For now, we will assume little endian
      int ID = MESSAGE_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(message_desc.kind), sizeof(message_desc.kind));
      lp_fwrite(f, message_desc.name, strlen(message_desc.name) + 1);
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                         const LegionProfDesc::MapperCallDesc &mapper_call_desc)
    //--------------------------------------------------------------------------
    {
      int ID = MAPPER_CALL_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(mapper_call_desc.kind), sizeof(mapper_call_desc.kind));
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
                                      const LegionProfDesc::ProcDesc& proc_desc)
    //--------------------------------------------------------------------------
    {
      int ID = PROC_DESC_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(proc_desc.proc_id), sizeof(proc_desc.proc_id));
      lp_fwrite(f, (char*)&(proc_desc.kind),    sizeof(proc_desc.kind));
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

    // Serialize Methods

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
      lp_fwrite(f, (char*)&(operation_instance.kind),
                sizeof(operation_instance.kind));
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
    }

    //--------------------------------------------------------------------------
    void LegionProfBinarySerializer::serialize(
                            const LegionProfInstance::MessageInfo& message_info)
    //--------------------------------------------------------------------------
    {
      int ID = MESSAGE_INFO_ID;
      lp_fwrite(f, (char*)&ID, sizeof(ID));
      lp_fwrite(f, (char*)&(message_info.kind),   sizeof(message_info.kind));
      lp_fwrite(f, (char*)&(message_info.start),  sizeof(message_info.start));
      lp_fwrite(f, (char*)&(message_info.stop),   sizeof(message_info.stop));
      lp_fwrite(f, (char*)&(message_info.proc_id),sizeof(message_info.proc_id));
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
                                const LegionProfDesc::MessageDesc &message_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Message Desc %u %s", 
                      message_desc.kind, message_desc.name);
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
      log_prof.print("Prof Meta Desc %u %s", meta_desc.kind, meta_desc.name);
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
                                      const LegionProfDesc::ProcDesc &proc_desc)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Proc Desc " IDFMT " %d", 
                     proc_desc.proc_id, proc_desc.kind);
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
      log_prof.print("Prof Task Variant %u %lu %s", task_variant.task_id,
         task_variant.variant_id, task_variant.name);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                           const LegionProfInstance::OperationInstance& op_inst)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Operation %llu %u", op_inst.op_id, op_inst.kind);
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
      log_prof.print("Prof Task Wait Info %llu %u %lu %llu %llu %llu",
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
      log_prof.print("Prof Task Info %llu %u %lu " IDFMT " %llu %llu %llu %llu",
                     task_info.op_id, task_info.task_id, task_info.variant_id, 
                     task_info.proc_id, task_info.create, task_info.ready, 
                     task_info.start, task_info.stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::MetaInfo& meta_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Meta Info %llu %u " IDFMT " %llu %llu %llu %llu",
         meta_info.op_id, meta_info.lg_id, meta_info.proc_id,
         meta_info.create, meta_info.ready, meta_info.start, meta_info.stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::CopyInfo& copy_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Copy Info %llu " IDFMT " " IDFMT " %llu"
         " %llu %llu %llu %llu", copy_info.op_id, copy_info.src,
         copy_info.dst, copy_info.size, copy_info.create,
         copy_info.ready, copy_info.start, copy_info.stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                                  const LegionProfInstance::FillInfo& fill_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Fill Info %llu " IDFMT 
         " %llu %llu %llu %llu", fill_info.op_id, fill_info.dst, 
         fill_info.create, fill_info.ready, fill_info.start, fill_info.stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                     const LegionProfInstance::InstCreateInfo& inst_create_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Inst Create %llu " IDFMT " %llu", 
                     inst_create_info.op_id, inst_create_info.inst_id, 
                     inst_create_info.create);
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
      log_prof.print("Prof Inst Timeline %llu " IDFMT " %llu %llu",
         inst_timeline_info.op_id, inst_timeline_info.inst_id,
         inst_timeline_info.create, inst_timeline_info.destroy);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                        const LegionProfInstance::PartitionInfo& partition_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Partition Timeline %llu %d %llu %llu %llu %llu",
                     partition_info.op_id, partition_info.part_op, 
                     partition_info.create, partition_info.create,
                     partition_info.start, partition_info.stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfASCIISerializer::serialize(
                            const LegionProfInstance::MessageInfo& message_info)
    //--------------------------------------------------------------------------
    {
      log_prof.print("Prof Message Info %u " IDFMT " %llu %llu",
                     message_info.kind, message_info.proc_id, 
                     message_info.start, message_info.stop);
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

