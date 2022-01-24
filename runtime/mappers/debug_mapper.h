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

#ifndef __DEBUG_MAPPER_H__
#define __DEBUG_MAPPER_H__

#include "mappers/replay_mapper.h"

namespace Legion {
  namespace Mapping {

    // TODO: Finish implementing the debug mapper

    /**
     * \class DebugMapper
     * The debug mapper is an extension of the replay 
     * mapper. It will replay all the mapping decisions as
     * they were made in a previous run of the program.
     * Additionally, the debug mapper permits for the setting
     * of breakpoints and controlling of execution.
     */
    class DebugMapper : public ReplayMapper {
    public:
      DebugMapper(MapperRuntime *rt, Machine machine, Processor local, 
                  const char *replay_file, const char *mapper_name = NULL);
      DebugMapper(const DebugMapper &rhs);
      virtual ~DebugMapper(void);
    public:
      DebugMapper& operator=(const DebugMapper &rhs);
    public:
      static const char* create_debug_name(Processor p);
    };

  }; // namespace Mapping
}; // namespace Legion

#endif // __DEBUG_MAPPER_H__

