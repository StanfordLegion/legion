/* Copyright 2023 Stanford University
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

#ifndef __REGENT_UTIL_H__
#define __REGENT_UTIL_H__

#include "legion_c_util.h"

#include "regent_tasks.h"

// These complain if not explicitly included in C mode.
extern "C"
{
#include "lua.h"
}

template< typename W, typename T>
void lua_push_opaque_object(lua_State* L, T obj)
{
  void* ptr = W::unwrap(obj);

  lua_newtable(L);
  lua_pushstring(L, "impl");
  lua_pushlightuserdata(L, ptr);
  lua_settable(L, -3);
}

template<typename T>
void lua_push_opaque_object_array(lua_State* L, T* objs, unsigned num_objs)
{
  lua_newtable(L);
  for(unsigned i = 0; i < num_objs; ++i)
  {
    lua_push_opaque_object<LegionRuntime::HighLevel::CObjectWrapper>(L, objs[i]);
    lua_pushinteger(L, i + 1);
    lua_insert(L, -2);
    lua_settable(L, -3);
  }
}

#define CHECK_LUA(EXP)                              \
  if ((EXP) != 0)                                   \
  {                                                 \
    fprintf(stderr,                                 \
        "error calling lua function : %s (%s:%d)\n", \
        lua_tostring(L, -1),                        \
        __FILE__, __LINE__);                        \
    exit(-1);                                       \
  }                                                 \

struct ObjectWrapper
{
  typedef LegionRuntime::HighLevel::Mapper::TaskSlice TaskSlice;

  static vector_legion_task_slice_t
  wrap(std::vector<TaskSlice>* slices)
  {
    vector_legion_task_slice_t slices_;
    slices_.impl = slices;
    return slices_;
  }

  static std::vector<TaskSlice>*
  unwrap(vector_legion_task_slice_t slices_)
  {
    return reinterpret_cast<std::vector<TaskSlice>*>(slices_.impl);
  }
};

#endif // __REGENT_UTIL_H__
