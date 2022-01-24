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

#include "mappers/debug_mapper.h"

namespace Legion {
  namespace Mapping {

    //--------------------------------------------------------------------------
    /*static*/ const char* DebugMapper::create_debug_name(Processor p)
    //--------------------------------------------------------------------------
    {
      const size_t buffer_size = 64;
      char *result = (char*)malloc(buffer_size*sizeof(char));
      snprintf(result, buffer_size-1,
                "Debug Mapper on Processor " IDFMT "", p.id);
      return result;
    }

    //--------------------------------------------------------------------------
    DebugMapper::DebugMapper(MapperRuntime *rt, Machine machine,Processor local,
                             const char *replay_file, const char *name)
      : ReplayMapper(rt, machine, local, replay_file, 
          (name == NULL) ? create_debug_name(local) : name)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DebugMapper::DebugMapper(const DebugMapper &rhs)
      : ReplayMapper(rhs.runtime, rhs.machine, rhs.local_proc, NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DebugMapper::~DebugMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DebugMapper& DebugMapper::operator=(const DebugMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

  }; // namespace Mapping
}; // namespace Legion

// EOF

