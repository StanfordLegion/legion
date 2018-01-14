-- Copyright 2018 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

require('legionlib')
require('circuit_tasks')
require('circuit_defs')

binding = LegionLib:init_binding("circuit_tasks.lua")

binding:set_top_level_task_id(CIRCUIT_MAIN)
binding:register_single_task(CIRCUIT_MAIN, "circuit_main",
                             Processor.LOC_PROC, false)
binding:register_index_task(CALC_NEW_CURRENTS, "calculate_currents_task",
                            Processor.LOC_PROC, true)
binding:register_index_task(DISTRIBUTE_CHARGE, "distribute_charge_task",
                            Processor.LOC_PROC, true)
binding:register_index_task(UPDATE_VOLTAGES, "update_voltages_task",
                            Processor.LOC_PROC, true)
binding:register_reduction_op(REDUCE_ID, ReductionType.PLUS, PrimType.float)
binding:start(arg)
