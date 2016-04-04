/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "replay_mapper.h"

namespace Legion {
  namespace Mapping {

    //--------------------------------------------------------------------------
    /*static*/ const char* ReplayMapper::create_replay_name(Processor p)
    //--------------------------------------------------------------------------
    {
      const size_t buffer_size = 64;
      char *result = (char*)malloc(buffer_size*sizeof(char));
      snprintf(result, buffer_size-1,
                "Replay Mapper on Processor " IDFMT "", p.id);
      return result;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReplayMapper(Machine m, Processor local,
                               const char *replay_file, const char *name)
      : machine(m), local_proc(local), 
        mapper_name((name == NULL) ? create_replay_name(local) : name)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReplayMapper(const ReplayMapper &rhs)
      : machine(rhs.machine), local_proc(rhs.local_proc), 
        mapper_name(rhs.mapper_name)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplayMapper::~ReplayMapper(void)
    //--------------------------------------------------------------------------
    {
      free(const_cast<char*>(mapper_name));
    }

    //--------------------------------------------------------------------------
    ReplayMapper& ReplayMapper::operator=(const ReplayMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    const char* ReplayMapper::get_mapper_name(void) const    
    //--------------------------------------------------------------------------
    {
      return mapper_name;
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel ReplayMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }

  }; // namespace Mapping 
}; // namespace Legion

// EOF

