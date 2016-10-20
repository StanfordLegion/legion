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

#include "legion_context.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Task Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(Runtime *rt)
      : runtime(rt), owner_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(const TaskContext &rhs)
      : runtime(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskContext& TaskContext::operator=(const TaskContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }
    
    /////////////////////////////////////////////////////////////
    // Top Level Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TopLevelContext::TopLevelContext(Runtime *rt)
      : TaskContext(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TopLevelContext::TopLevelContext(const TopLevelContext &rhs)
      : TaskContext(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TopLevelContext::~TopLevelContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TopLevelContext& TopLevelContext::operator=(const TopLevelContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }
    
    /////////////////////////////////////////////////////////////
    // Remote Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteContext::RemoteContext(Runtime *rt)
      : TaskContext(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteContext::RemoteContext(const RemoteContext &rhs)
      : TaskContext(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteContext::~RemoteContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteContext& RemoteContext::operator=(const RemoteContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Inline Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InlineContext::InlineContext(Runtime *rt)
      : TaskContext(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineContext::InlineContext(const InlineContext &rhs)
      : TaskContext(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InlineContext::~InlineContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineContext& InlineContext::operator=(const InlineContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

  };
};

// EOF

