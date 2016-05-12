/* Copyright 2016 Stanford University
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

#ifndef __LUA_MAPPER_WRAPPER_H__
#define __LUA_MAPPER_WRAPPER_H__

#include <vector>
#include <string>

#include "legion.h"
#include "default_mapper.h"

extern "C"
{
#include "lua.h"
}

using namespace LegionRuntime::HighLevel;

class LuaMapperWrapper : public DefaultMapper
{
  public:
    LuaMapperWrapper(Machine machine, Processor locak, const char *name);
#if 0
  public:
    LuaMapperWrapper(const char*, Machine, HighLevelRuntime*, Processor);
    ~LuaMapperWrapper();

    virtual void slice_domain(const Task *task, const Domain &domain,
                              std::vector<DomainSplit> &slices);
    virtual bool map_task(Task *task);
    virtual bool map_inline(Inline *inline_operation);

    virtual void notify_mapping_failed(const Mappable *mappable);

  private:
    const std::string qualified_mapper_name;
    std::string mapper_name;
    lua_State* L;
#endif
};

#endif // __LUA_MAPPER_WRAPPER_H__
