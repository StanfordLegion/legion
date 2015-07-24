/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#ifndef REALM_OPERATION_H
#define REALM_OPERATION_H

#include "lowlevel.h"
#include "realm/profiling.h"

namespace Realm {

  class Operation {
  public:
    Operation(void);
    Operation(const ProfilingRequestSet &reqs);
    virtual ~Operation(void);
  public:
    virtual void mark_ready(void);
    virtual void mark_started(void);
    virtual void mark_completed(void);
  public:
    inline bool perform_capture(void) const { return capture_timeline; }
  protected:
    void clear_profiling(void);
    void reconstruct_measurements();
    ProfilingMeasurements::OperationStatus status;
    ProfilingMeasurements::OperationTimeline timeline;
    ProfilingRequestSet requests; 
    ProfilingMeasurementCollection measurements;
    bool capture_timeline;
  };

};

#endif // REALM_OPERATION_H
