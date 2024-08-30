/* Copyright 2024 Stanford University, NVIDIA Corporation
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
#include "legion/legion_ops.h"
#include "legion/region_tree.h"
#include "legion/legion_views.h"
#include "legion/legion_mapping.h"
#include "legion/mapper_manager.h"
#include "legion/legion_instances.h"

namespace Legion {
  namespace Mapping {

    /////////////////////////////////////////////////////////////
    // PhysicalInstance 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalInstance::PhysicalInstance(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalInstance::PhysicalInstance(PhysicalInstanceImpl i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      // By holding gc references, we prevent the data
      // structure from being collected, it doesn't change if 
      // the actual instance itself can be collected or not
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::MAPPER_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance::PhysicalInstance(PhysicalInstance &&rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      rhs.impl = NULL;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance::PhysicalInstance(const PhysicalInstance &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::MAPPER_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance::~PhysicalInstance(void)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && 
          impl->remove_base_gc_ref(Internal::MAPPER_REF))
        delete (impl);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance& PhysicalInstance::operator=(PhysicalInstance &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && 
          impl->remove_base_gc_ref(Internal::MAPPER_REF))
        delete (impl);
      impl = rhs.impl;
      rhs.impl = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance& PhysicalInstance::operator=(const PhysicalInstance &rhs)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && 
          impl->remove_base_gc_ref(Internal::MAPPER_REF))
        delete (impl);
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::MAPPER_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::operator<(const PhysicalInstance &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl < rhs.impl);
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::operator==(const PhysicalInstance &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl == rhs.impl);
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::operator!=(const PhysicalInstance &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl != rhs.impl);
    }

    //--------------------------------------------------------------------------
    std::size_t PhysicalInstance::hash(void) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        return std::hash<unsigned long long>{}(impl->did);
      else
        return std::hash<unsigned long long>{}(0);
    }

    //--------------------------------------------------------------------------
    Memory PhysicalInstance::get_location(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return Memory::NO_MEMORY;
      Internal::PhysicalManager *manager = impl->as_physical_manager();
      return manager->get_memory();
    }

    //--------------------------------------------------------------------------
    unsigned long PhysicalInstance::get_instance_id(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return 0;
      Internal::PhysicalManager *manager = impl->as_physical_manager();
      return manager->get_instance().id;
    }

    //--------------------------------------------------------------------------
    size_t PhysicalInstance::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return 0;
      return impl->as_physical_manager()->get_instance_size();
    }

    //--------------------------------------------------------------------------
    Domain PhysicalInstance::get_instance_domain(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return Domain::NO_DOMAIN;
      Domain domain;
      impl->instance_domain->get_domain(domain);
      return domain;
    }

    //--------------------------------------------------------------------------
    FieldSpace PhysicalInstance::get_field_space(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return FieldSpace::NO_SPACE;
      return impl->field_space_node->handle;
    }

    //--------------------------------------------------------------------------
    RegionTreeID PhysicalInstance::get_tree_id(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return 0;
      return impl->tree_id;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID PhysicalInstance::get_layout_id(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return 0;
      return impl->layout->constraints->layout_id;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::exists(bool strong_test /*= false*/) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return false;
      // Check to see if it still exists for now, maybe in the future
      // we could do a full check to see if it still exists on its owner node
      if (strong_test)
        assert(false); // implement this
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_normal_instance(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return false;
      return !impl->is_reduction_manager();
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_virtual_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->is_virtual_manager();
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_reduction_instance(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return false;
      return impl->is_reduction_manager();
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_external_instance(void) const
    //--------------------------------------------------------------------------
    {
      if ((impl == NULL) || !impl->is_physical_manager())
        return false;
      return impl->is_external_instance();
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalInstance PhysicalInstance::get_virtual_instance(void)
    //--------------------------------------------------------------------------
    {
      return PhysicalInstance(Internal::implicit_runtime->virtual_manager);
    }

    //--------------------------------------------------------------------------
    void PhysicalInstance::get_fields(std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_fields(fields);
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::has_field(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->has_field(fid);
    }

    //--------------------------------------------------------------------------
    void PhysicalInstance::has_fields(std::map<FieldID,bool> &fields) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        for (std::map<FieldID,bool>::iterator it = fields.begin();
              it != fields.end(); it++)
          it->second = false;
        return;
      }
      return impl->has_fields(fields);
    }

    //--------------------------------------------------------------------------
    void PhysicalInstance::remove_space_fields(std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return;
      impl->remove_space_fields(fields);
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::entails(const LayoutConstraintSet &constraint_set,
                               const LayoutConstraint **failed_constraint) const 
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->entails(constraint_set, failed_constraint);
    }

    //--------------------------------------------------------------------------
    /*friend*/ std::ostream& operator<<(std::ostream& os,
					const PhysicalInstance& p)
    //--------------------------------------------------------------------------
    {
      if (!p.impl->is_physical_manager())
        return os << Realm::RegionInstance::NO_INST;
      else
        return os << p.impl->as_physical_manager()->get_instance();
    }

    /////////////////////////////////////////////////////////////
    // CollectiveView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveView::CollectiveView(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CollectiveView::CollectiveView(CollectiveViewImpl i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->add_base_gc_ref(Internal::MAPPER_REF);
    }

    //--------------------------------------------------------------------------
    CollectiveView::CollectiveView(CollectiveView &&rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      rhs.impl = NULL;
    }

    //--------------------------------------------------------------------------
    CollectiveView::CollectiveView(const CollectiveView &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::MAPPER_REF);
    }

    //--------------------------------------------------------------------------
    CollectiveView::~CollectiveView(void)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) &&
          impl->remove_base_gc_ref(Internal::MAPPER_REF))
        delete impl;
    }

    //--------------------------------------------------------------------------
    CollectiveView& CollectiveView::operator=(CollectiveView &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) &&
          impl->remove_base_gc_ref(Internal::MAPPER_REF))
        delete impl;
      impl = rhs.impl;
      rhs.impl = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    CollectiveView& CollectiveView::operator=(const CollectiveView &rhs)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) &&
          impl->remove_base_gc_ref(Internal::MAPPER_REF))
        delete impl;
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::MAPPER_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::operator<(const CollectiveView &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl < rhs.impl);
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::operator==(const CollectiveView &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl == rhs.impl);
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::operator!=(const CollectiveView &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl != rhs.impl);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::find_instances_in_memory(Memory memory,
                                     std::vector<PhysicalInstance> &insts) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return;
      std::vector<Internal::PhysicalManager*> managers;
      impl->find_instances_in_memory(memory, managers);
      insts.reserve(insts.size() + managers.size());
      for (unsigned idx = 0; idx < managers.size(); idx++)
        insts.emplace_back(PhysicalInstance(managers[idx]));
    }

    //--------------------------------------------------------------------------
    void CollectiveView::find_instances_nearest_memory(Memory memory,
                     std::vector<PhysicalInstance> &insts, bool bandwidth) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return;
      std::vector<Internal::PhysicalManager*> managers;
      impl->find_instances_nearest_memory(memory, managers, bandwidth);
      insts.reserve(insts.size() + managers.size());
      for (unsigned idx = 0; idx < managers.size(); idx++)
        insts.emplace_back(PhysicalInstance(managers[idx]));
    }

    //--------------------------------------------------------------------------
    /*friend*/ std::ostream& operator<<(std::ostream& os,
					const CollectiveView &v)
    //--------------------------------------------------------------------------
    {
      if (v.impl == NULL)
        return os << "Empty Collective View";
      else
        return os << "Collective View " << std::hex << v.impl->did << std::dec;
    }

    /////////////////////////////////////////////////////////////
    // ProfilingRequest
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void ProfilingRequest::populate_realm_profiling_request(
					      Realm::ProfilingRequest& req)
    //--------------------------------------------------------------------------
    {
      for (std::set<ProfilingMeasurementID>::const_iterator it =
	     requested_measurements.begin();
	   it != requested_measurements.end();
	   ++it)
      {
	if((int)(*it) <= (int)(Realm::PMID_REALM_LAST))
	  req.add_measurement((Realm::ProfilingMeasurementID)(*it));
      }
    }

    /////////////////////////////////////////////////////////////
    // ProfilingResponse
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void ProfilingResponse::attach_realm_profiling_response(
					const Realm::ProfilingResponse& resp)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(realm_resp == NULL);
#endif
      realm_resp = &resp;
    }
    
    //--------------------------------------------------------------------------
    void ProfilingResponse::attach_overhead(
                                   ProfilingMeasurements::RuntimeOverhead *over)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(overhead == NULL);
#endif
      overhead = over; 
    }

    /////////////////////////////////////////////////////////////
    // Mapper 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mapper::Mapper(MapperRuntime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Mapper::~Mapper(void)
    //--------------------------------------------------------------------------
    {
    }

    LEGION_DISABLE_DEPRECATED_WARNINGS

    //--------------------------------------------------------------------------
    Mapper::PremapTaskInput::PremapTaskInput(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Mapper::PremapTaskInput::~PremapTaskInput(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Mapper::PremapTaskOutput::PremapTaskOutput(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Mapper::PremapTaskOutput::~PremapTaskOutput(void)
    //--------------------------------------------------------------------------
    {
    }

    LEGION_REENABLE_DEPRECATED_WARNINGS

    class AutoMapperCall {
    public:
      inline AutoMapperCall(MapperContext ctx, Internal::RuntimeCallKind kind);
      inline ~AutoMapperCall(void);
    public:
      const MapperContext ctx;
      const Internal::RuntimeCallKind kind;
    };

    //--------------------------------------------------------------------------
    inline AutoMapperCall::AutoMapperCall(MapperContext c,
                                          Internal::RuntimeCallKind k)
      : ctx(c), kind(k)
    //--------------------------------------------------------------------------
    {
      if (ctx != Internal::implicit_mapper_call)
      {
        static RUNTIME_CALL_DESCRIPTIONS(runtime_call_names);
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_CONTENT,
                      "Invalid mapper context passed to mapper runtime "
                      "call %s by mapper %s inside of mapper call %s. Mapper "
                      "contexts are only valid for the mapper call to which "
                      "they are passed. They cannot be stored beyond the "
                      "lifetime of the mapper call.", runtime_call_names[kind],
                      ctx->get_mapper_name(), ctx->get_mapper_call_name())
      }
      ctx->pause_mapper_call();
    }

    //--------------------------------------------------------------------------
    inline AutoMapperCall::~AutoMapperCall(void)
    //--------------------------------------------------------------------------
    {
      ctx->resume_mapper_call(kind);
    }

    /////////////////////////////////////////////////////////////
    // MapperRuntime
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperRuntime::MapperRuntime(Internal::Runtime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapperRuntime::~MapperRuntime(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_locked(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return ctx->is_locked();
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::lock_mapper(MapperContext ctx, bool read_only) const
    //--------------------------------------------------------------------------
    {
      ctx->lock_mapper(read_only);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::unlock_mapper(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      ctx->unlock_mapper();
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_reentrant(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return ctx->is_reentrant();
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::enable_reentrant(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      ctx->enable_reentrant();
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::disable_reentrant(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      ctx->disable_reentrant();
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::update_mappable_tag(MapperContext ctx,
                           const Mappable &mappable, MappingTagID new_tag) const
    //--------------------------------------------------------------------------
    {
      Mappable *map = const_cast<Mappable*>(&mappable);
      map->tag = new_tag;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::update_mappable_data(MapperContext ctx,
      const Mappable &mappable, const void *mapper_data, size_t data_size) const
    //--------------------------------------------------------------------------
    {
      Mappable *map = const_cast<Mappable*>(&mappable);
      // Free the old buffer if there is one
      if (map->mapper_data != NULL)
        free(map->mapper_data);
      map->mapper_data_size = data_size;
      if (data_size > 0)
      {
        map->mapper_data = malloc(data_size);
        memcpy(map->mapper_data, mapper_data, data_size);
      }
      else
        map->mapper_data = NULL;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::send_message(MapperContext ctx, Processor target,
          const void *message, size_t message_size, unsigned message_kind) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_SEND_MESSAGE_CALL);
      runtime->process_mapper_message(target, ctx->manager->mapper_id,
          ctx->manager->processor, message, message_size, message_kind);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::broadcast(MapperContext ctx, const void *message,
                    size_t message_size, unsigned message_kind, int radix) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_BROADCAST_CALL);
      runtime->process_mapper_broadcast(ctx->manager->mapper_id,
          ctx->manager->processor, message, message_size, message_kind,
          radix, 0/*index*/);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::pack_physical_instance(MapperContext ctx,
                                Serializer &rez, PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      // No need to even pause the mapper call here
      rez.serialize(instance.impl->did);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::unpack_physical_instance(MapperContext ctx,
                           Deserializer &derez, PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_UNPACK_INSTANCE_CALL);
      DistributedID did;
      derez.deserialize(did);
      Internal::RtEvent ready;
      instance.impl = runtime->find_or_request_instance_manager(did, ready);
      if (ready.exists())
        ready.wait();
    }

    //--------------------------------------------------------------------------
    MapperEvent MapperRuntime::create_mapper_event(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_CREATE_EVENT_CALL);
      MapperEvent result;
      result.impl = Internal::Runtime::create_rt_user_event();
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_mapper_event_triggered(MapperContext ctx,
                                                   MapperEvent event) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_HAS_TRIGGERED_CALL);
      return event.impl.has_triggered();
    }
    
    //--------------------------------------------------------------------------
    void MapperRuntime::trigger_mapper_event(MapperContext ctx, 
                                             MapperEvent event) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_TRIGGER_EVENT_CALL);
      Internal::RtUserEvent to_trigger = event.impl;
      if (to_trigger.exists())
        Internal::Runtime::trigger_event(to_trigger);
    }
    
    //--------------------------------------------------------------------------
    void MapperRuntime::wait_on_mapper_event(MapperContext ctx,
                                             MapperEvent event) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_WAIT_EVENT_CALL);
      Internal::RtEvent wait_on = event.impl;
      if (wait_on.exists())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    const ExecutionConstraintSet& MapperRuntime::find_execution_constraints(
                        MapperContext ctx, TaskID task_id, VariantID vid) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_EXECUTION_CONSTRAINTS_CALL);
      Internal::VariantImpl *impl = 
        runtime->find_variant_impl(task_id, vid, true/*can fail*/);
      if (impl == NULL)
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s requested execution "
                      "constraints for variant %d in mapper call %s, but "
                      "that variant does not exist.", ctx->get_mapper_name(),
                      vid, ctx->get_mapper_call_name())
      return impl->get_execution_constraints();
    }

    //--------------------------------------------------------------------------
    const TaskLayoutConstraintSet& MapperRuntime::find_task_layout_constraints(
                        MapperContext ctx, TaskID task_id, VariantID vid) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_TASK_LAYOUT_CONSTRAINTS_CALL);
      Internal::VariantImpl *impl = 
        runtime->find_variant_impl(task_id, vid, true/*can fail*/);
      if (impl == NULL)
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s requested task layout "
                      "constraints for variant %d in mapper call %s, but "
                      "that variant does not exist.", ctx->get_mapper_name(),
                      vid, ctx->get_mapper_call_name())
      return impl->get_layout_constraints();
    }

    //--------------------------------------------------------------------------
    const LayoutConstraintSet& MapperRuntime::find_layout_constraints(
                         MapperContext ctx, LayoutConstraintID layout_id) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_LAYOUT_CONSTRAINTS_CALL);
      Internal::LayoutConstraints *constraints = 
        runtime->find_layout_constraints(layout_id, true/*can fail*/);
      if (constraints == NULL)
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s requested layout "
                      "constraints for layout ID %ld in mapper call %s, but "
                      "that layout constraint ID is invalid.",
                      ctx->get_mapper_name(), layout_id,
                      ctx->get_mapper_call_name())
      return *constraints;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID MapperRuntime::register_layout(MapperContext ctx,
                                    const LayoutConstraintSet &constraints,
                                    FieldSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_REGISTER_LAYOUT_CALL);
      Internal::LayoutConstraints *cons = 
        runtime->register_layout(handle, constraints, false/*internal*/);
      return cons->layout_id;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_layout(MapperContext ctx, 
                                       LayoutConstraintID layout_id) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RELEASE_LAYOUT_CALL);
      runtime->release_layout(layout_id);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::do_constraints_conflict(MapperContext ctx,
                             LayoutConstraintID set1, LayoutConstraintID set2,
                             const LayoutConstraint **conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_CONSTRAINTS_CONFLICT_CALL);
      Internal::LayoutConstraints *c1 = 
        runtime->find_layout_constraints(set1, true/*can fail*/);
      Internal::LayoutConstraints *c2 = 
        runtime->find_layout_constraints(set2, true/*can fail*/);
      if ((c1 == NULL) || (c2 == NULL))
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s passed layout ID %ld "
                      "to conflict test in mapper call %s, but that layout ID "
                      "is invalid.", ctx->get_mapper_name(), 
                      (c1 == NULL) ? set1 : set2, ctx->get_mapper_call_name())
      const bool result = 
        c1->conflicts(c2, 0/*dont care about dimensions*/, conflict_constraint);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::do_constraints_entail(MapperContext ctx,
                           LayoutConstraintID source, LayoutConstraintID target,
                           const LayoutConstraint **failed_constraint) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_CONSTRAINTS_ENTAIL_CALL);
      Internal::LayoutConstraints *c1 = 
        runtime->find_layout_constraints(source, true/*can fail*/);
      Internal::LayoutConstraints *c2 = 
        runtime->find_layout_constraints(target, true/*can fail*/);
      if ((c1 == NULL) || (c2 == NULL))
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s passed layout ID %ld "
                      "to entailment test in mapper call %s, but that layout "
                      "ID is invalid.", ctx->get_mapper_name(), 
                      (c1 == NULL) ? source : target, 
                      ctx->get_mapper_call_name())
      const bool result = 
        c1->entails(c2, 0/*don't care about dimensions*/, failed_constraint);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::find_valid_variants(MapperContext ctx,TaskID task_id,
                                         std::vector<VariantID> &valid_variants,
                                         Processor::Kind kind) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_VALID_VARIANTS_CALL);
      Internal::TaskImpl *task_impl = 
        runtime->find_or_create_task_impl(task_id);
      task_impl->find_valid_variants(valid_variants, kind);
    }
    
    //--------------------------------------------------------------------------
    const char* MapperRuntime::find_task_variant_name(
                  MapperContext ctx, TaskID task_id, VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_TASK_VARIANT_NAME_CALL);
      Internal::VariantImpl *impl = 
        runtime->find_variant_impl(task_id, variant_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_leaf_variant(MapperContext ctx,
                                    TaskID task_id, VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_IS_LEAF_VARIANT_CALL);
      Internal::VariantImpl *impl =
        runtime->find_variant_impl(task_id, variant_id);
      return impl->is_leaf();
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_inner_variant(MapperContext ctx,
                                     TaskID task_id, VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_IS_INNER_VARIANT_CALL);
      Internal::VariantImpl *impl =
        runtime->find_variant_impl(task_id, variant_id);
      return impl->is_inner();
    }
    
    //--------------------------------------------------------------------------
    bool MapperRuntime::is_idempotent_variant(MapperContext ctx,
                                     TaskID task_id, VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_IS_IDEMPOTENT_VARIANT_CALL);
      Internal::VariantImpl *impl =
        runtime->find_variant_impl(task_id, variant_id);
      return impl->is_idempotent();
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_replicable_variant(MapperContext ctx,
                                     TaskID task_id, VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_IS_REPLICABLE_VARIANT_CALL);
      Internal::VariantImpl *impl =
        runtime->find_variant_impl(task_id, variant_id);
      return impl->is_replicable();
    }

    //--------------------------------------------------------------------------
    VariantID MapperRuntime::register_task_variant(MapperContext ctx,
                                  const TaskVariantRegistrar &registrar,
                                  const CodeDescriptor &realm_desc,
                                  const void *user_data, size_t user_len,
                                  size_t return_type_size,
                                  bool has_return_type) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_REGISTER_TASK_VARIANT_CALL);
      return runtime->register_variant(registrar, user_data,
                user_len, realm_desc, return_type_size, has_return_type);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::filter_variants(MapperContext ctx, const Task &task,
            const std::vector<std::vector<PhysicalInstance> > &chosen_instances,
                                        std::vector<VariantID> &variants) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FILTER_VARIANTS_CALL);
      std::map<LayoutConstraintID,Internal::LayoutConstraints*> layout_cache;
      for (std::vector<VariantID>::iterator var_it = variants.begin();
            var_it != variants.end(); /*nothing*/)
      {
        Internal::VariantImpl *impl = runtime->find_variant_impl(
            task.task_id, *var_it, true/*can_fail*/);
        // Not a valid variant
        if (impl == NULL)
        {
          var_it = variants.erase(var_it);
          continue;
        }
        const TaskLayoutConstraintSet &layout_constraints = 
                                        impl->get_layout_constraints();
        bool conflicts = false;
        for (std::multimap<unsigned,LayoutConstraintID>::const_iterator 
              lay_it = layout_constraints.layouts.begin(); 
              lay_it != layout_constraints.layouts.end(); lay_it++)
        {
          Internal::LayoutConstraints *constraints;
          std::map<LayoutConstraintID,Internal::LayoutConstraints*>
            ::const_iterator finder = layout_cache.find(lay_it->second);
          if (finder == layout_cache.end())
          {
            constraints = runtime->find_layout_constraints(lay_it->second);
            layout_cache[lay_it->second] = constraints;
          }
          else
            constraints = finder->second;
          const std::vector<PhysicalInstance> &instances = 
                                       chosen_instances[lay_it->first];
          for (unsigned idx = 0; idx < instances.size(); idx++)
          {
            Internal::InstanceManager *manager = instances[idx].impl;
            if (manager->conflicts(constraints,  NULL))
            {
              conflicts = true;
              break;
            }
            if (!constraints->specialized_constraint.is_virtual() &&
                (constraints->specialized_constraint.is_exact() ||
                 constraints->padding_constraint.delta.get_dim() > 0))
            {
              std::vector<LogicalRegion> regions_to_check(1,
                        task.regions[lay_it->first].region);
              Internal::PhysicalManager *phy = manager->as_physical_manager();
              if (!phy->meets_regions(regions_to_check,
                    constraints->specialized_constraint.is_exact(),
                    &constraints->padding_constraint.delta))
              {
                conflicts = true;
                break;
              }
            }
          }
          if (conflicts)
            break;
        }
        if (conflicts)
          var_it = variants.erase(var_it);
        else
          var_it++;
      }
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::filter_instances(MapperContext ctx, const Task &task,
                                         VariantID chosen_variant,
                  std::vector<std::vector<PhysicalInstance> > &chosen_instances,
                  std::vector<std::set<FieldID> > &missing_fields) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FILTER_INSTANCES_CALL);
      missing_fields.resize(task.regions.size());
      Internal::VariantImpl *impl = runtime->find_variant_impl(task.task_id, 
                                                     chosen_variant);
      const TaskLayoutConstraintSet &layout_constraints = 
                                        impl->get_layout_constraints();
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        if (idx >= chosen_instances.size())
          continue;
        std::vector<PhysicalInstance> &instances = chosen_instances[idx]; 
        // Iterate over the layout constraints and filter them
        // We know that instance constraints are complete (all dimensions
        // are fully constrainted), therefore we only need to test for conflicts
        for (std::multimap<unsigned,LayoutConstraintID>::const_iterator lay_it =
              layout_constraints.layouts.lower_bound(idx); lay_it != 
              layout_constraints.layouts.upper_bound(idx); lay_it++)
        {
          Internal::LayoutConstraints *constraints = 
            runtime->find_layout_constraints(lay_it->second);
          for (std::vector<PhysicalInstance>::iterator it = 
                instances.begin(); it != instances.end(); /*nothing*/)
          {
            Internal::InstanceManager *manager = it->impl;
            if (manager->conflicts(constraints, NULL))
              it = instances.erase(it);
            else if (!constraints->specialized_constraint.is_virtual() &&
                      (constraints->specialized_constraint.is_exact() ||
                       constraints->padding_constraint.delta.get_dim() > 0))
            {
              std::vector<LogicalRegion> regions_to_check(1,
                        task.regions[lay_it->first].region);
              Internal::PhysicalManager *phy = manager->as_physical_manager();
              if (!phy->meets_regions(regions_to_check,
                    constraints->specialized_constraint.is_exact(),
                    &constraints->padding_constraint.delta))
                it = instances.erase(it);
              else
                it++;
            }
            else
              it++;
          }
          if (instances.empty())
            break;
        }
        // Now figure out which fields are missing
        std::set<FieldID> &missing = missing_fields[idx];
        missing = task.regions[idx].privilege_fields;
        for (std::vector<PhysicalInstance>::const_iterator it = 
              instances.begin(); it != instances.end(); it++)
        {
          Internal::InstanceManager *manager = it->impl;
          manager->remove_space_fields(missing);
          if (missing.empty())
            break;
        }
      }
    }
    
    //--------------------------------------------------------------------------
    void MapperRuntime::filter_instances(MapperContext ctx, const Task &task,
                            unsigned index, VariantID chosen_variant,
                            std::vector<PhysicalInstance> &instances,
                            std::set<FieldID> &missing_fields) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_FILTER_INSTANCES_CALL);
      Internal::VariantImpl *impl = runtime->find_variant_impl(task.task_id, 
                                                     chosen_variant);
      const TaskLayoutConstraintSet &layout_constraints = 
                                        impl->get_layout_constraints();
      // Iterate over the layout constraints and filter them
      // We know that instance constraints are complete (all dimensions
      // are fully constrainted), therefore we only need to test for conflicts
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator lay_it =
            layout_constraints.layouts.lower_bound(index); lay_it != 
            layout_constraints.layouts.upper_bound(index); lay_it++)
      {
        Internal::LayoutConstraints *constraints = 
          runtime->find_layout_constraints(lay_it->second);
        for (std::vector<PhysicalInstance>::iterator it = 
              instances.begin(); it != instances.end(); /*nothing*/)
        {
          Internal::InstanceManager *manager = it->impl;
          if (manager->conflicts(constraints, NULL))
            it = instances.erase(it);
          else if (!constraints->specialized_constraint.is_virtual() &&
                    (constraints->specialized_constraint.is_exact() ||
                     constraints->padding_constraint.delta.get_dim() > 0))
          {
            std::vector<LogicalRegion> regions_to_check(1,
                      task.regions[lay_it->first].region);
            Internal::PhysicalManager *phy = manager->as_physical_manager();
            if (!phy->meets_regions(regions_to_check,
                  constraints->specialized_constraint.is_exact(),
                  &constraints->padding_constraint.delta))
              it = instances.erase(it);
            else
              it++;
          }
          else
            it++;
        }
        if (instances.empty())
          break;
      }
      // Now see which fields we are missing
      missing_fields = task.regions[index].privilege_fields;
      for (std::vector<PhysicalInstance>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        Internal::InstanceManager *manager = it->impl;
        manager->remove_space_fields(missing_fields);
        if (missing_fields.empty())
          break;
      }
    }

    //--------------------------------------------------------------------------
    static void check_region_consistency(MapperContext ctx,
                                         const char *call_name,
                                      const std::vector<LogicalRegion> &regions)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tree_id = 0;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!regions[idx].exists())
          continue;
        if (tree_id > 0)
        {
          RegionTreeID other_id = regions[idx].get_tree_id();
          if (other_id != tree_id)
            REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                          "Invalid region arguments passed to %s in "
                          "mapper call %s of mapper %s. All region arguments "
                          "must be from the same region tree (%d != %d).",
                          call_name, ctx->get_mapper_call_name(),
                          ctx->get_mapper_name(), tree_id, other_id)
        }
        else
          tree_id = regions[idx].get_tree_id();
      }
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "create_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to create_physical_instance "
                        "in unsupported mapper call %s in mapper %s", 
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_CREATE_PHYSICAL_INSTANCE_CALL);
      bool success = runtime->create_physical_instance(target_memory, 
        constraints, regions, result, ctx->manager->processor, acquire,
        priority, tight_region_bounds, unsat, footprint, 
        (ctx->operation == NULL) ? 0 : ctx->operation->get_unique_op_id());
      if (success && acquire)
        ctx->record_acquired_instance(result.impl, true/*created*/);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result,
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "create_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to create_physical_instance "
                        "in unsupported mapper call %s in mapper %s", 
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_CREATE_PHYSICAL_INSTANCE_CALL);
      Internal::LayoutConstraints *cons =
        runtime->find_layout_constraints(layout_id);
      bool success = runtime->create_physical_instance(target_memory, cons,
                      regions, result, ctx->manager->processor, acquire,
                      priority, tight_region_bounds, unsat, footprint,
                      (ctx->operation == NULL) ? 0 :
                        ctx->operation->get_unique_op_id());
      if (success && acquire)
        ctx->record_acquired_instance(result.impl, true/*created*/);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_or_create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "find_or_create_physical_instance",
                               regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_or_create_physical"
                        "_instance in unsupported mapper call %s in mapper %s",
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx,
          Internal::MAPPER_FIND_OR_CREATE_PHYSICAL_INSTANCE_CALL);
      bool success = runtime->find_or_create_physical_instance(target_memory,
                constraints, regions, result, created, ctx->manager->processor,
                acquire, priority, tight_region_bounds, unsat, footprint,
                (ctx->operation == NULL) ? 0 :
                 ctx->operation->get_unique_op_id());
      if (success && acquire)
        ctx->record_acquired_instance(result.impl, created);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_or_create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "find_or_create_physical_instance",
                               regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_or_create_physical"
                        "_instance in unsupported mapper call %s in mapper %s",
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx,
          Internal::MAPPER_FIND_OR_CREATE_PHYSICAL_INSTANCE_CALL);
      Internal::LayoutConstraints *cons =
        runtime->find_layout_constraints(layout_id);
      bool success = runtime->find_or_create_physical_instance(target_memory,
                 cons, regions, result, created, ctx->manager->processor,
                 acquire, priority, tight_region_bounds, unsat, footprint,
                 (ctx->operation == NULL) ? 0 : 
                  ctx->operation->get_unique_op_id());
      if (success && acquire)
        ctx->record_acquired_instance(result.impl, created);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_physical_instance(  
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result, bool acquire,
                                    bool tight_region_bounds) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      check_region_consistency(ctx, "find_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instance "
                        "in unsupported mapper call %s in mapper %s",
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_PHYSICAL_INSTANCE_CALL);
      bool success = runtime->find_physical_instance(target_memory, constraints,
                                 regions, result, acquire, tight_region_bounds);
      if (success && acquire)
        ctx->record_acquired_instance(result.impl, false/*created*/);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_physical_instance(  
                                MapperContext ctx, Memory target_memory,
                                LayoutConstraintID layout_id,
                                const std::vector<LogicalRegion> &regions,
                                PhysicalInstance &result, bool acquire,
                                bool tight_region_bounds) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      check_region_consistency(ctx, "find_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instance "
                        "in unsupported mapper call %s in mapper %s",
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_PHYSICAL_INSTANCE_CALL);
      Internal::LayoutConstraints *cons =
        runtime->find_layout_constraints(layout_id);
      bool success = runtime->find_physical_instance(target_memory, cons,
                          regions, result, acquire, tight_region_bounds);
      if (success && acquire)
        ctx->record_acquired_instance(result.impl, false/*created*/);
      return success;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::find_physical_instances(  
                                  MapperContext ctx, Memory target_memory,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  std::vector<PhysicalInstance> &results, 
                                  bool acquire, bool tight_region_bounds) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return;
      check_region_consistency(ctx, "find_physical_instances", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instances "
                        "in unsupported mapper call %s in mapper %s",
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_PHYSICAL_INSTANCES_CALL);
      const size_t initial_size = results.size();
      runtime->find_physical_instances(target_memory, constraints, regions, 
                                    results, acquire, tight_region_bounds);
      if ((initial_size < results.size()) && acquire)
      {
        for (unsigned idx = initial_size; idx < results.size(); idx++)
          ctx->record_acquired_instance(results[idx].impl, false/*created*/);
      }
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::find_physical_instances(  
                                  MapperContext ctx, Memory target_memory,
                                  LayoutConstraintID layout_id,
                                  const std::vector<LogicalRegion> &regions,
                                  std::vector<PhysicalInstance> &results, 
                                  bool acquire, bool tight_region_bounds) const
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return;
      check_region_consistency(ctx, "find_physical_instances", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instances "
                        "in unsupported mapper call %s in mapper %s",
                        ctx->get_mapper_call_name(), ctx->get_mapper_name());
        acquire = false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_FIND_PHYSICAL_INSTANCES_CALL);
      Internal::LayoutConstraints *cons =
        runtime->find_layout_constraints(layout_id);
      const size_t initial_size = results.size();
      runtime->find_physical_instances(target_memory, cons, regions, 
                              results, acquire, tight_region_bounds);
      if ((initial_size < results.size()) && acquire)
      {
        for (unsigned idx = initial_size; idx < results.size(); idx++)
          ctx->record_acquired_instance(results[idx].impl, false/*created*/);
      }
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::set_garbage_collection_priority(MapperContext ctx,
                     const PhysicalInstance &instance, GCPriority priority) const
    //--------------------------------------------------------------------------
    {
      Internal::InstanceManager *man = instance.impl;
      if (man->is_virtual_manager())
        return;
      AutoMapperCall call(ctx, Internal::MAPPER_SET_GC_PRIORITY_CALL);
      Internal::PhysicalManager *manager = man->as_physical_manager();
      // Ignore garbage collection priorities on external instances
      if (!manager->is_external_instance())
      {
        const Internal::RtEvent ready =
          manager->set_garbage_collection_priority(ctx->manager->mapper_id,
              ctx->manager->processor, runtime->address_space, priority);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
      else
        REPORT_LEGION_WARNING(LEGION_WARNING_EXTERNAL_GARBAGE_PRIORITY,
            "Ignoring request for mapper %s to set garbage collection "
            "priority on an external instance", ctx->get_mapper_name())
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_instance(MapperContext ctx,
                                         const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return false;
      }
      Internal::InstanceManager *man = instance.impl;
      // virtual instances are easy
      if (man->is_virtual_manager())
        return true;
      Internal::PhysicalManager *manager = man->as_physical_manager();
      // See if we already acquired it
      if (ctx->acquired_instances->find(manager) !=
          ctx->acquired_instances->end())
        return true;
      AutoMapperCall call(ctx, Internal::MAPPER_ACQUIRE_INSTANCE_CALL);
      if (manager->acquire_instance(Internal::MAPPING_ACQUIRE_REF))
      {
        ctx->record_acquired_instance(manager, false/*created*/);
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_instances(MapperContext ctx,
                            const std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return false;
      }
      // Quick fast path
      if (instances.size() == 1)
        return acquire_instance(ctx, instances[0]);
      AutoMapperCall call(ctx, Internal::MAPPER_ACQUIRE_INSTANCES_CALL);
      return ctx->perform_acquires(instances);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_and_filter_instances(MapperContext ctx,
                                    std::vector<PhysicalInstance> &instances,
                                    const bool filter_acquired_instances) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return false;
      }
      // Quick fast path
      if (instances.size() == 1)
      {
        bool result = acquire_instance(ctx, instances[0]);
        if (result)
        {
          if (filter_acquired_instances)
            instances.clear();
        }
        else
        {
          if (!filter_acquired_instances)
            instances.clear();
        }
        return result;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_ACQUIRE_AND_FILTER_INSTANCES_CALL);
      // Figure out which instances we need to acquire and sort by memories
      std::vector<unsigned> to_erase;
      const bool all_acquired =
        ctx->perform_acquires(instances, &to_erase, filter_acquired_instances);
      // Filter any invalid local instances
      if (!to_erase.empty())
      {
        // Erase from the back
        for (std::vector<unsigned>::const_reverse_iterator it =
              to_erase.rbegin(); it != to_erase.rend(); it++)
          instances.erase(instances.begin()+(*it));
        to_erase.clear();
      }
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_instances(MapperContext ctx,
              const std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_ACQUIRE_INSTANCES_CALL);
      // Figure out which instances we need to acquire and sort by memories
      bool all_acquired = true;
      for (std::vector<std::vector<PhysicalInstance> >::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        if (!ctx->perform_acquires(*it))
          all_acquired = false;
      }
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_and_filter_instances(MapperContext ctx,
                          std::vector<std::vector<PhysicalInstance> > &instances,
                          const bool filter_acquired_instances) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_ACQUIRE_AND_FILTER_INSTANCES_CALL);
      // Figure out which instances we need to acquire and sort by memories
      bool all_acquired = true;
      std::vector<unsigned> to_erase;
      for (std::vector<std::vector<PhysicalInstance> >::iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        if (!ctx->perform_acquires(*it, &to_erase, filter_acquired_instances))
        {
          all_acquired = false;
          // Erase from the back
          for (std::vector<unsigned>::const_reverse_iterator rit = 
                to_erase.rbegin(); rit != to_erase.rend(); rit++)
            it->erase(it->begin()+(*rit));
          to_erase.clear();
        }
      }
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_instance(MapperContext ctx, 
                                         const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_RELEASE_REQUEST,
                        "Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_RELEASE_INSTANCE_CALL);
      ctx->release_acquired_instance(instance.impl); 
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_instances(MapperContext ctx,
                           const std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_RELEASE_REQUEST,
                        "Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_RELEASE_INSTANCES_CALL);
      for (unsigned idx = 0; idx < instances.size(); idx++)
        ctx->release_acquired_instance(instances[idx].impl);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_instances(MapperContext ctx, 
             const std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_RELEASE_REQUEST,
                        "Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_RELEASE_INSTANCES_CALL);
      for (std::vector<std::vector<PhysicalInstance> >::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        for (unsigned idx = 0; idx < it->size(); idx++)
          ctx->release_acquired_instance((*it)[idx].impl);
      }
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::subscribe(MapperContext ctx,
                                  const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      if ((instance.impl == NULL) || instance.impl->is_virtual_manager())
        return false;
      AutoMapperCall call(ctx, Internal::MAPPER_SUBSCRIBE_INSTANCE_CALL);
      Internal::PhysicalManager *manager = instance.impl->as_physical_manager();
      const bool result = manager->register_deletion_subscriber(
          ctx->manager, true/*allow duplicates*/);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::unsubscribe(MapperContext ctx,
                                    const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      if ((instance.impl == NULL) || instance.impl->is_virtual_manager())
        return;
      AutoMapperCall call(ctx, Internal::MAPPER_UNSUBSCRIBE_INSTANCE_CALL);
      Internal::PhysicalManager *manager = instance.impl->as_physical_manager();
      manager->unregister_deletion_subscriber(ctx->manager);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::collect_instance(MapperContext ctx, 
                                         const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      if ((instance.impl == NULL) || instance.impl->is_virtual_manager() ||
          instance.impl->is_external_instance())
        return false;
      AutoMapperCall call(ctx, Internal::MAPPER_COLLECT_INSTANCE_CALL);
      Internal::PhysicalManager *manager = instance.impl->as_physical_manager();
      Internal::RtEvent collected;
      const bool result = manager->collect(collected);
      if (result)
      {
        // Tell the memory that the instance has been collected
        std::vector<Internal::PhysicalManager*> collected_instance(1, manager);
        manager->memory_manager->notify_collected_instances(collected_instance);
        // Wait for the collection to be done 
        collected.wait();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::collect_instances(MapperContext ctx,
                                const std::vector<PhysicalInstance> &instances,
                                std::vector<bool> &collected) const
    //--------------------------------------------------------------------------
    {
      collected.resize(instances.size(), false);
      if (instances.empty())
        return;
      AutoMapperCall call(ctx, Internal::MAPPER_COLLECT_INSTANCES_CALL);
      std::vector<Internal::RtEvent> wait_for;
      std::map<Internal::MemoryManager*,
        std::vector<Internal::PhysicalManager*> > to_notify;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        collected[idx] = false;
        Internal::InstanceManager *inst = instances[idx].impl;
        if ((inst == NULL) || inst->is_virtual_manager() || 
            inst->is_external_instance())
          continue;
        Internal::RtEvent instance_collected;
        Internal::PhysicalManager *manager = inst->as_physical_manager();
        if (manager->collect(instance_collected))
        {
          collected[idx] = true;
          to_notify[manager->memory_manager].push_back(manager);
          if (instance_collected.exists())
            wait_for.push_back(instance_collected);
        }
      }
      // Notify all the memory managers of the collection
      for (std::map<Internal::MemoryManager*,
            std::vector<Internal::PhysicalManager*> >::const_iterator it =
            to_notify.begin(); it != to_notify.end(); it++)
        it->first->notify_collected_instances(it->second);
      if (!wait_for.empty())
      {
        const Internal::RtEvent wait_on =
          Internal::Runtime::merge_events(wait_for);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_future(MapperContext ctx,
                                       const Future &future,Memory memory) const
    //--------------------------------------------------------------------------
    {
      if ((future.impl == NULL) || !memory.exists())
        return false;
      if (ctx->kind != Internal::MAP_TASK_CALL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire future request in unsupported mapper "
                        "call %s in mapper %s", ctx->get_mapper_call_name(),
                        ctx->get_mapper_name());
        return false;
      }
      AutoMapperCall call(ctx, Internal::MAPPER_ACQUIRE_FUTURE_CALL);
      return future.impl->find_or_create_application_instance(
                                memory, ctx->operation->get_unique_op_id()); 
    } 

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::create_index_space(MapperContext ctx,
                                                 const Domain &domain,
                                                 TypeTag type_tag,
                                                 const char *prov) const
    //--------------------------------------------------------------------------
    {
      if (type_tag == 0)
      {
        switch (domain.get_dim())
        {
#define DIMFUNC(DIM) \
          case DIM: \
            { \
              type_tag = \
                Internal::NT_TemplateHelper::encode_tag<DIM,coord_t>(); \
              break; \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            assert(false);
        }
      }
      AutoMapperCall call(ctx, Internal::MAPPER_CREATE_INDEX_SPACE_CALL);
      Internal::Provenance *provenance = NULL;
      if (prov != NULL)
        provenance = runtime->find_or_create_provenance(prov, strlen(prov));
      const IndexSpace result(runtime->get_unique_index_space_id(),
                    runtime->get_unique_index_tree_id(), type_tag);
      const DistributedID did = runtime->get_available_distributed_id();
      runtime->forest->create_index_space(result, &domain, did, provenance);
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      return result; 
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::create_index_space(MapperContext ctx,
           const std::vector<DomainPoint> &points, const char *provenance) const
    //--------------------------------------------------------------------------
    {
      switch (points[0].get_dim())
      {
#define DIMFUNC(DIM) \
      case DIM: \
        { \
          std::vector<Realm::Point<DIM,coord_t> > realm_points(points.size()); \
          for (unsigned idx = 0; idx < points.size(); idx++) \
            realm_points[idx] = Point<DIM,coord_t>(points[idx]); \
          DomainT<DIM,coord_t> realm_is( \
              (Realm::IndexSpace<DIM,coord_t>(realm_points))); \
          const Domain domain(realm_is); \
          return create_index_space(ctx, domain, \
           Internal::NT_TemplateHelper::encode_tag<DIM,coord_t>(),provenance); \
        }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::create_index_space(MapperContext ctx,
                 const std::vector<Domain> &rects, const char *provenance) const
    //--------------------------------------------------------------------------
    {
      switch (rects[0].get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            std::vector<Realm::Rect<DIM,coord_t> > realm_rects(rects.size()); \
            for (unsigned idx = 0; idx < rects.size(); idx++) \
              realm_rects[idx] = Rect<DIM,coord_t>(rects[idx]); \
            DomainT<DIM,coord_t> realm_is( \
                (Realm::IndexSpace<DIM,coord_t>(realm_rects))); \
            const Domain domain(realm_is); \
            return create_index_space(ctx, domain, \
                      Internal::NT_TemplateHelper::encode_tag<DIM,coord_t>(), \
                      provenance); \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::union_index_spaces(MapperContext ctx,
           const std::vector<IndexSpace> &sources, const char *provenance) const
    //--------------------------------------------------------------------------
    {
      if (sources.empty())
        return IndexSpace::NO_SPACE;
      AutoMapperCall call(ctx, Internal::MAPPER_UNION_INDEX_SPACES_CALL);
      bool none_exists = true;
      for (std::vector<IndexSpace>::const_iterator it = 
            sources.begin(); it != sources.end(); it++)
      {
        if (none_exists && it->exists())
          none_exists = false;
        if (sources[0].get_type_tag() != it->get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'union_index_spaces' "
                        "performed in mapper %s", ctx->get_mapper_name())
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace result(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), sources[0].get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      Internal::AutoProvenance prov(provenance);
      runtime->forest->create_union_space(result, did, prov, sources);
      if (runtime->legion_spy_enabled)
        Internal::LegionSpy::log_top_index_space(result.get_id(),
                    runtime->address_space, provenance);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::intersect_index_spaces(MapperContext ctx,
           const std::vector<IndexSpace> &sources, const char *provenance) const
    //--------------------------------------------------------------------------
    {
      if (sources.empty())
        return IndexSpace::NO_SPACE;
      AutoMapperCall call(ctx, Internal::MAPPER_INTERSECT_INDEX_SPACES_CALL);
      bool none_exists = true;
      for (std::vector<IndexSpace>::const_iterator it = 
            sources.begin(); it != sources.end(); it++)
      {
        if (none_exists && it->exists())
          none_exists = false;
        if (sources[0].get_type_tag() != it->get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'intersect_index_spaces' "
                        "performed in mapper %s", ctx->get_mapper_name())
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace result(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), sources[0].get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      Internal::AutoProvenance prov(provenance);
      runtime->forest->create_intersection_space(result, did, prov, sources);
      if (runtime->legion_spy_enabled)
        Internal::LegionSpy::log_top_index_space(result.get_id(),
                    runtime->address_space, provenance);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::subtract_index_spaces(MapperContext ctx,
                IndexSpace left, IndexSpace right, const char *provenance) const
    //--------------------------------------------------------------------------
    {
      if (!left.exists())
        return IndexSpace::NO_SPACE;
      AutoMapperCall call(ctx, Internal::MAPPER_SUBTRACT_INDEX_SPACES_CALL);
      if (right.exists() && left.get_type_tag() != right.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'create_difference_spaces' "
                        "performed in mapper %s", ctx->get_mapper_name())
      const IndexSpace result(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), left.get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      Internal::AutoProvenance prov(provenance);
      runtime->forest->create_difference_space(result, did, prov,
                                               left, right);
      if (runtime->legion_spy_enabled)
        Internal::LegionSpy::log_top_index_space(result.get_id(),
                    runtime->address_space, provenance);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_index_space_empty(MapperContext ctx,
                                             IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return true;
      AutoMapperCall call(ctx, Internal::MAPPER_INDEX_SPACE_EMPTY_CALL);
      Internal::IndexSpaceNode *node = runtime->forest->get_node(handle);
      return node->is_empty();
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::index_spaces_overlap(MapperContext ctx,
                                           IndexSpace one, IndexSpace two) const
    //--------------------------------------------------------------------------
    {
      if (!one.exists() || !two.exists())
        return false;
      AutoMapperCall call(ctx, Internal::MAPPER_INDEX_SPACES_OVERLAP_CALL);
      if (one.get_type_tag() != two.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'index_spaces_overlap' "
                        "performed in mapper %s", ctx->get_mapper_name())
      Internal::IndexSpaceNode *n1 = runtime->forest->get_node(one);
      Internal::IndexSpaceNode *n2 = runtime->forest->get_node(two);
      Internal::IndexSpaceExpression *overlap = 
        runtime->forest->intersect_index_spaces(n1, n2);
      return !overlap->is_empty();
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::index_space_dominates(MapperContext ctx,
                                        IndexSpace left, IndexSpace right) const
    //--------------------------------------------------------------------------
    {
      if (!left.exists())
        return true;
      if (!right.exists())
        return false;
      AutoMapperCall call(ctx, Internal::MAPPER_INDEX_SPACE_DOMINATES_CALL);
      if (left.get_type_tag() != right.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'index_spaces_dominates' "
                        "performed in mapper %s", ctx->get_mapper_name())
      Internal::IndexSpaceNode *n1 = runtime->forest->get_node(left);
      Internal::IndexSpaceNode *n2 = runtime->forest->get_node(right);
      Internal::IndexSpaceExpression *difference =
        runtime->forest->subtract_index_spaces(n1, n2);
      return difference->is_empty();
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_index_partition(MapperContext ctx,
                                            IndexSpace parent,Color color) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_HAS_INDEX_PARTITION_CALL);
      return runtime->has_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperRuntime::get_index_partition(MapperContext ctx,
                                                      IndexSpace parent, 
                                                      Color color) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_PARTITION_CALL);
      return runtime->get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::get_index_subspace(MapperContext ctx,
                                                 IndexPartition p,
                                                 Color c) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_SUBSPACE_CALL);
      Point<1,coord_t> color(c);
      return runtime->get_index_subspace(p, &color,
                    Internal::NT_TemplateHelper::encode_tag<1,coord_t>());
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::get_index_subspace(MapperContext ctx,
                                                 IndexPartition p, 
                                                 const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_SUBSPACE_CALL);
      IndexSpace result = IndexSpace::NO_SPACE;
      switch (color.get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            Point<DIM,coord_t> point(color); \
            result = runtime->get_index_subspace(p, &point, \
                Internal::NT_TemplateHelper::encode_tag<DIM,coord_t>()); \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_multiple_domains(MapperContext ctx,
                                             IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      // Never have multiple domains
      return false;
    }

    //--------------------------------------------------------------------------
    Domain MapperRuntime::get_index_space_domain(MapperContext ctx,
                                                 IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_SPACE_DOMAIN_CALL);
      Domain result = Domain::NO_DOMAIN;
      switch (Internal::NT_TemplateHelper::get_dim(handle.get_type_tag()))
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            DomainT<DIM,coord_t> realm_is; \
            const TypeTag tag =\
              Internal::NT_TemplateHelper::encode_tag<DIM,coord_t>(); \
            runtime->get_index_space_domain(handle, &realm_is, tag); \
            result = realm_is; \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::get_index_space_domains(MapperContext ctx,
                                            IndexSpace handle,
                                            std::vector<Domain> &domains) const
    //--------------------------------------------------------------------------
    {
      domains.push_back(get_index_space_domain(ctx, handle));
    }

    //--------------------------------------------------------------------------
    Domain MapperRuntime::get_index_partition_color_space(MapperContext ctx,
                                                        IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_PARTITION_CS_CALL);
      return runtime->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::get_index_partition_color_space_name(
                                     MapperContext ctx, IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_INDEX_PARTITION_CS_NAME_CALL);
      return runtime->get_index_partition_color_space_name(p);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::get_index_space_partition_colors(MapperContext ctx,
                               IndexSpace handle, std::set<Color> &colors) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_INDEX_SPACE_PARTITION_COLORS_CALL);
      runtime->get_index_space_partition_colors(handle, colors);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_index_partition_disjoint(MapperContext ctx,
                                                    IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_IS_INDEX_PARTITION_DISJOINT_CALL);
      return runtime->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_index_partition_complete(MapperContext ctx,
                                                    IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_IS_INDEX_PARTITION_COMPLETE_CALL);
      return runtime->is_index_partition_complete(p);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_index_space_color(MapperContext ctx,
                                               IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_SPACE_COLOR_CALL);
      Point<1,coord_t> point;
      runtime->get_index_space_color_point(handle, &point,
                Internal::NT_TemplateHelper::encode_tag<1,coord_t>());
      return point[0];
    }

    //--------------------------------------------------------------------------
    DomainPoint MapperRuntime::get_index_space_color_point(MapperContext ctx,
                                                       IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_SPACE_COLOR_POINT_CALL);
      return runtime->get_index_space_color_point(handle);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_index_partition_color(MapperContext ctx,
                                                   IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_PARTITION_COLOR_CALL);
      return runtime->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::get_parent_index_space(MapperContext ctx,
                                                   IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_PARENT_INDEX_SPACE_CALL);
      return runtime->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_parent_index_partition(MapperContext ctx,
                                                   IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_HAS_PARENT_INDEX_PARTITION_CALL);
      return runtime->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperRuntime::get_parent_index_partition(
                                    MapperContext ctx, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_PARENT_INDEX_PARTITION_CALL);
      return runtime->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    unsigned MapperRuntime::get_index_space_depth(MapperContext ctx,
                                                  IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_SPACE_DEPTH_CALL);
      return runtime->get_index_space_depth(handle);
    }

    //--------------------------------------------------------------------------
    unsigned MapperRuntime::get_index_partition_depth(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_INDEX_PARTITION_DEPTH_CALL);
      return runtime->get_index_partition_depth(handle);
    }

    //--------------------------------------------------------------------------
    size_t MapperRuntime::get_field_size(MapperContext ctx,
                                         FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_FIELD_SIZE_CALL);
      return runtime->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::get_field_space_fields(MapperContext ctx,
                          FieldSpace handle, std::vector<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_FIELD_SPACE_FIELDS_CALL);
      runtime->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition(MapperContext ctx,
                              LogicalRegion parent, IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_LOGICAL_PARTITION_CALL);
      return runtime->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition_by_color(
                       MapperContext ctx, LogicalRegion par, Color color) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_LOGICAL_PARTITION_BY_COLOR_CALL);
      return runtime->get_logical_partition_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition_by_color(
          MapperContext ctx, LogicalRegion par, const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_LOGICAL_PARTITION_BY_COLOR_CALL);
#ifdef DEBUG_LEGION
      assert((color.get_dim() == 0) || (color.get_dim() == 1));
#endif
      return runtime->get_logical_partition_by_color(par, color[0]);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition_by_tree(
                                                        MapperContext ctx,
                                                        IndexPartition part,
                                                        FieldSpace fspace, 
                                                        RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_LOGICAL_PARTITION_BY_TREE_CALL);
      return runtime->get_logical_partition_by_tree(part, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion(MapperContext ctx,
                                                       LogicalPartition parent,
                                                       IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_LOGICAL_SUBREGION_CALL);
      return runtime->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion_by_color(
                    MapperContext ctx, LogicalPartition par, Color color) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_LOGICAL_SUBREGION_BY_COLOR_CALL);
      Point<1,coord_t> point(color);
      return runtime->get_logical_subregion_by_color(par, &point,
          Internal::NT_TemplateHelper::encode_tag<1,coord_t>());
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion_by_color(
       MapperContext ctx, LogicalPartition par, const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_LOGICAL_SUBREGION_BY_COLOR_CALL);
      LogicalRegion result = LogicalRegion::NO_REGION;
      switch (color.get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            Point<DIM,coord_t> point(color); \
            result = runtime->get_logical_subregion_by_color(par, &point, \
                  Internal::NT_TemplateHelper::encode_tag<DIM,coord_t>()); \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion_by_tree(
                                      MapperContext ctx, IndexSpace handle, 
                                      FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_LOGICAL_SUBREGION_BY_TREE_CALL);
      return runtime->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_logical_region_color(MapperContext ctx,
                                                  LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_LOGICAL_REGION_COLOR_CALL);
      Point<1,coord_t> point;
      runtime->get_logical_region_color(handle, &point, 
            Internal::NT_TemplateHelper::encode_tag<1,coord_t>());
      return point[0];
    }

    //--------------------------------------------------------------------------
    DomainPoint MapperRuntime::get_logical_region_color_point(
                                  MapperContext ctx, LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_LOGICAL_REGION_COLOR_POINT_CALL);
      return runtime->get_logical_region_color_point(handle);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_logical_partition_color(MapperContext ctx,
                                                 LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_LOGICAL_PARTITION_COLOR_CALL);
      return runtime->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_parent_logical_region(MapperContext ctx,
                                                    LogicalPartition part) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_GET_PARENT_LOGICAL_REGION_CALL);
      return runtime->get_parent_logical_region(part);
    }
    
    //--------------------------------------------------------------------------
    bool MapperRuntime::has_parent_logical_partition(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_HAS_PARENT_LOGICAL_PARTITION_CALL);
      return runtime->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_parent_logical_partition(
                                      MapperContext ctx, LogicalRegion r) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx,
          Internal::MAPPER_GET_PARENT_LOGICAL_PARTITION_CALL);
      return runtime->get_parent_logical_partition(r);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        TaskID task_id, SemanticTag tag, const void *&result, size_t &size,
        bool can_fail, bool wait_until_ready) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return runtime->retrieve_semantic_information(task_id, tag,
					     result, size,
                                             can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        IndexSpace handle, SemanticTag tag, const void *&result, size_t &size,
        bool can_fail, bool wait_until_ready) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        IndexPartition handle, SemanticTag tag, const void *&result, 
        size_t &size, bool can_fail, bool wait_until_ready) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        FieldSpace handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        FieldSpace handle, FieldID fid, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return runtime->retrieve_semantic_information(handle, fid,
					     tag, result, size,
                                             can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        LogicalRegion handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        LogicalPartition handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, TaskID task_id,
                                      const char *&result) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_NAME_CALL);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(task_id, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, IndexSpace handle,
                                      const char *&result) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_NAME_CALL);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, 
                              IndexPartition handle, const char *&result) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_NAME_CALL);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, FieldSpace handle,
                                      const char *&result) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_NAME_CALL);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, FieldSpace handle,
                                      FieldID fid, const char *&result) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_NAME_CALL);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, fid, 
          LEGION_NAME_SEMANTIC_TAG, name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, 
                                LogicalRegion handle, const char *&result) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_NAME_CALL);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx,
                             LogicalPartition handle, const char *&result) const
    //--------------------------------------------------------------------------
    {
      AutoMapperCall call(ctx, Internal::MAPPER_RETRIEVE_NAME_CALL);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_MPI_interop_configured(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return runtime->is_MPI_interop_configured();
    }

    //--------------------------------------------------------------------------
    const std::map<int,AddressSpace>& MapperRuntime::find_forward_MPI_mapping(
                                                       MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return runtime->find_forward_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    const std::map<AddressSpace,int>& MapperRuntime::find_reverse_MPI_mapping(
                                                       MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return runtime->find_reverse_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    int MapperRuntime::find_local_MPI_rank(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return runtime->find_local_MPI_rank();
    }

    /////////////////////////////////////////////////////////////
    // AutoLock
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AutoLock::AutoLock(MapperContext c, LocalLock &r, int mode, bool excl)
      : Internal::AutoLock(mode, excl, r), ctx(c)
    //--------------------------------------------------------------------------
    {
      bool paused = false;
      if (exclusive)
      {
        Internal::RtEvent ready = local_lock.wrlock();
        while (ready.exists())
        {
          if (!paused)
          {
            ctx->manager->pause_mapper_call(ctx);
            paused = true;
          }
          ready.wait();
          ready = local_lock.wrlock();
        }
      }
      else
      {
        Internal::RtEvent ready = local_lock.rdlock();
        while (ready.exists())
        {
          if (!paused)
          {
            ctx->manager->pause_mapper_call(ctx);
            paused = true;
          }
          ready.wait();
          ready = local_lock.rdlock();
        }
      }
      held = true;
      Internal::local_lock_list = this;
      if (paused)
        ctx->manager->resume_mapper_call(ctx, Internal::MAPPER_AUTO_LOCK_CALL);
    }

    //--------------------------------------------------------------------------
    void AutoLock::reacquire(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!held);
      assert(Internal::local_lock_list == previous);
#endif
#ifdef DEBUG_REENTRANT_LOCKS
      if (previous != NULL)
        previous->check_for_reentrant_locks(&local_lock);
#endif
      bool paused = false;
      if (exclusive)
      {
        Internal::RtEvent ready = local_lock.wrlock();
        while (ready.exists())
        {
          if (!paused)
          {
            ctx->manager->pause_mapper_call(ctx);
            paused = true;
          }
          ready.wait();
          ready = local_lock.wrlock();
        }
      }
      else
      {
        Internal::RtEvent ready = local_lock.rdlock();
        while (ready.exists())
        {
          if (!paused)
          {
            ctx->manager->pause_mapper_call(ctx);
            paused = true;
          }
          ready.wait();
          ready = local_lock.rdlock();
        }
      }
      Internal::local_lock_list = this;
      held = true;
      if (paused)
        ctx->manager->resume_mapper_call(ctx, Internal::MAPPER_AUTO_LOCK_CALL);
    }

  }; // namespace Mapping
}; // namespace Legion

