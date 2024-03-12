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

// NOP, but useful for IDE's
#include "realm/hip/hip_module.h"

#include "realm/runtime.h"

namespace Realm {
  
  namespace Hip {

    // a running task on a HIP processor is assigned a stream by Realm, and
    //  any work placed on this stream is automatically captured by the
    //  completion event for the task
    // when using the Hip runtime hijack, Realm will force work launched via
    //  the runtime API to use the task's stream, but without hijack, or for
    //  code that uses the HIP runtime API, the task must explicitly request
    //  the stream that is associated with the task and place work on it to
    //  avoid more expensive forms of completion detection for the task
    // NOTE: this function will return a null pointer if called outside of a
    //  task running on a HIP processor
    inline unifiedHipStream_t *get_task_hip_stream()
    {
      HipModule *mod = Runtime::get_runtime().get_module<HipModule>("hip");
      if(mod)
        return mod->get_task_hip_stream();
      else
        return 0;
    }

    // when Realm is not using the HIP runtime hijack to force work onto the
    //  task's stream, it conservatively uses a full context synchronization to
    //  make sure all device work launched by the task is captured by the task
    //  completion event - if a task uses `get_task_hip_stream` and places all
    //  work on that stream, this API can be used to tell Realm on a per-task
    //  basis that full context synchronization is not required
    inline void set_task_ctxsync_required(bool is_required)
    {
      HipModule *mod = Runtime::get_runtime().get_module<HipModule>("hip");
      if(mod)
        mod->set_task_ctxsync_required(is_required);
    }

  }; // namespace Hip

}; // namespace Realm
