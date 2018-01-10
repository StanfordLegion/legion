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
require('saxpy_task')

local binding = LegionLib:init_binding("saxpy_task.lua")
binding:set_top_level_task_id(TOP_LEVEL_TASK_ID)
binding:register_single_task(TOP_LEVEL_TASK_ID, "top_level_task",
                             Processor.LOC_PROC, false)
binding:register_single_task(TASKID_MAIN, "main_task",
                             Processor.LOC_PROC, false)
binding:register_index_task(TASKID_INIT_VECTORS, "init_vectors_task",
                            Processor.LOC_PROC, true)
binding:register_index_task(TASKID_ADD_VECTORS, "add_vectors_task",
                            Processor.LOC_PROC, true)
binding:start(arg)
