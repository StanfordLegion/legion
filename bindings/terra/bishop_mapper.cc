/* Copyright 2015 Stanford University
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

#include "bishop_mapper.h"
#include "legion_c_util.h"
#include "utilities.h"

extern "C" {
#include "legion_c.h"
}

using namespace std;

LegionRuntime::Logger::Category log_bishop("bishop");

namespace LegionRuntime {

  namespace HighLevel {

#define RUN_ALL_TASK_RULES(CALLBACK) \
    for (unsigned i = 0; i < task_rules.size(); ++i)       \
    {                                                      \
      bishop_task_rule_t& rule = task_rules[i];            \
      if (rule.CALLBACK)                                   \
      {                                                    \
        legion_task_t task_ = CObjectWrapper::wrap(task);  \
        rule.CALLBACK(task_);                              \
      }                                                    \
    }                                                      \

#define RUN_ALL_REGION_RULES(CALLBACK)                               \
    for (unsigned i = 0; i < region_rules.size(); ++i)               \
    {                                                                \
      bishop_region_rule_t& rule = region_rules[i];                  \
      if (rule.CALLBACK)                                             \
      {                                                              \
        legion_task_t task_ = CObjectWrapper::wrap(task);            \
        for (unsigned idx = 0; idx < task->regions.size(); ++idx)    \
        {                                                            \
          legion_region_requirement_t req_ =                         \
            CObjectWrapper::wrap(&task->regions[idx]);               \
          rule.CALLBACK(task_, req_);                                \
        }                                                            \
      }                                                              \
    }                                                                \

    //--------------------------------------------------------------------------
		BishopMapper::BishopMapper(const std::vector<bishop_task_rule_t>& trules,
                               const std::vector<bishop_region_rule_t>& rrules,
                               Machine machine, HighLevelRuntime *runtime,
                               Processor local_proc)
			: DefaultMapper(machine, runtime, local_proc),
        task_rules(trules), region_rules(rrules)
    //--------------------------------------------------------------------------
		{
      log_bishop.info("bishop mapper created");
		}

    //--------------------------------------------------------------------------
		BishopMapper::~BishopMapper()
    //--------------------------------------------------------------------------
		{
      log_bishop.info("bishop mapper destroyed");
		}

    //--------------------------------------------------------------------------
    void BishopMapper::select_task_options(Task *task)
    //--------------------------------------------------------------------------
    {
      DefaultMapper::select_task_options(task);
      RUN_ALL_TASK_RULES(select_task_options);
    }

    //--------------------------------------------------------------------------
    void BishopMapper::slice_domain(const Task *task, const Domain &domain,
        std::vector<DomainSplit> &slices)
    //--------------------------------------------------------------------------
    {
      using namespace LegionRuntime::Arrays;
      log_bishop.info("[slice_domain] task %s", task->variants->name);
      DefaultMapper::slice_domain(task, domain, slices);
      for (unsigned i = 0; i < task_rules.size(); ++i)
      {
        bishop_task_rule_t& rule = task_rules[i];
        if (rule.select_target_for_point)
        {
          legion_task_t task_ = CObjectWrapper::wrap(const_cast<Task*>(task));
          // TODO: only supports 1D indexspace launch at the moment
          if (rule.matches(task_) && domain.get_dim() == 1)
          {
            slices.clear();
            Arrays::Rect<1> r = domain.get_rect<1>();
            for(Arrays::GenericPointInRectIterator<1> pir(r); pir; pir++)
            {
              legion_domain_point_t dp_ =
                CObjectWrapper::wrap(DomainPoint::from_point<1>(pir.p));
              Processor target =
                CObjectWrapper::unwrap(
                    rule.select_target_for_point(task_, dp_));
              Arrays::Rect<1> subrect(pir.p, pir.p);
              Mapper::DomainSplit ds(Domain::from_rect<1>(subrect), target,
                  false, false);
              slices.push_back(ds);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool BishopMapper::pre_map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_bishop.info("[pre_map_task] task %s", task->variants->name);
      bool result = DefaultMapper::pre_map_task(task);
      RUN_ALL_REGION_RULES(pre_map_task);
      return result;
    }

    //--------------------------------------------------------------------------
    void BishopMapper::select_task_variant(Task *task)
    //--------------------------------------------------------------------------
    {
      log_bishop.info("[select_task_variant] task %s", task->variants->name);
      DefaultMapper::select_task_variant(task);
      RUN_ALL_TASK_RULES(select_task_variant);
    }

    //--------------------------------------------------------------------------
    bool BishopMapper::map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_bishop.info("[map_task] task %s", task->variants->name);
      bool result = DefaultMapper::map_task(task);
      RUN_ALL_REGION_RULES(map_task);
      return result;
    }

    //--------------------------------------------------------------------------
    void BishopMapper::notify_mapping_result(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      DefaultMapper::notify_mapping_result(mappable);
    }

    //--------------------------------------------------------------------------
    void BishopMapper::notify_mapping_failed(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      DefaultMapper::notify_mapping_failed(mappable);
    }
	};
};
