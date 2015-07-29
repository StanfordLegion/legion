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

#ifndef LOWLEVEL_IMPL_H
#define LOWLEVEL_IMPL_H

// For doing bit masks for maximum number of nodes
#include "legion_types.h"
#include "legion_utilities.h"

#define NODE_MASK_TYPE uint64_t
#define NODE_MASK_SHIFT 6
#define NODE_MASK_MASK 0x3F

#ifndef MAX_NUM_THREADS
#define MAX_NUM_THREADS 32
#endif

#include "lowlevel.h"

#define NO_USE_REALMS_NODESET
#ifdef USE_REALMS_NODESET
#include "realm/dynamic_set.h"

namespace LegionRuntime {
  namespace LowLevel {
#if MAX_NUM_NODES <= 65536
    typedef DynamicSet<unsigned short> NodeSet;
#else
    // possibly unnecessary future-proofing...
    typedef DynamicSet<unsigned int> NodeSet;
#endif
  };
};
#else
namespace LegionRuntime {
  namespace LowLevel {
    typedef LegionRuntime::HighLevel::NodeSet NodeSet;
    //typedef LegionRuntime::HighLevel::BitMask<NODE_MASK_TYPE,MAX_NUM_NODES,
    //                                          NODE_MASK_SHIFT,NODE_MASK_MASK> NodeMask;
  };
};
namespace Realm {
  typedef LegionRuntime::HighLevel::NodeSet NodeSet;
};
#endif

#include "realm/operation.h"
#include "realm/dynamic_table.h"
#include "realm/id.h"
#include "realm/metadata.h"

#include <assert.h>

#include "activemsg.h"
#include <pthread.h>

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DECLARE(cur_thread);

extern pthread_key_t thread_timer_key;

#include <string.h>

#include <vector>
#include <deque>
#include <queue>
#include <set>
#include <list>
#include <map>
#include <aio.h>
#include <greenlet>

#if __cplusplus >= 201103L
#define typeof decltype
#endif

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)


namespace Realm {
  class Module;
  class Operation;
  class ProfilingRequestSet;
};

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"
#include "realm/machine_impl.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/inst_impl.h"
#include "realm/idx_impl.h"
#include "realm/runtime_impl.h"

// namespace importing for backwards compatibility
namespace LegionRuntime {
  namespace LowLevel {
    typedef Realm::ID ID;
    typedef Realm::EventWaiter EventWaiter;
    typedef Realm::EventImpl EventImpl;
    typedef Realm::GenEventImpl GenEventImpl;
    typedef Realm::BarrierImpl BarrierImpl;
    typedef Realm::ReservationImpl ReservationImpl;
    typedef Realm::MachineImpl MachineImpl;
    typedef Realm::ProcessorImpl ProcessorImpl;
    typedef Realm::ProcessorGroup ProcessorGroup;
    typedef Realm::Task Task;
    typedef Realm::MemoryImpl MemoryImpl;
    typedef Realm::MetadataBase MetadataBase;
    typedef Realm::RegionInstanceImpl RegionInstanceImpl;
    typedef Realm::IndexSpaceImpl IndexSpaceImpl;
    typedef Realm::Node Node;
  };
};

namespace LegionRuntime {
  namespace LowLevel {

    extern Logger::Category log_mutex;

#ifdef EVENT_TRACING
    // For event tracing
    struct EventTraceItem {
    public:
      enum Action {
        ACT_CREATE = 0,
        ACT_QUERY = 1,
        ACT_TRIGGER = 2,
        ACT_WAIT = 3,
      };
    public:
      unsigned time_units, event_id, event_gen, action;
    };
#endif

#ifdef LOCK_TRACING
    // For lock tracing
    struct LockTraceItem {
    public:
      enum Action {
        ACT_LOCAL_REQUEST = 0, // request for a lock where the owner is local
        ACT_REMOTE_REQUEST = 1, // request for a lock where the owner is not local
        ACT_FORWARD_REQUEST = 2, // for forwarding of requests
        ACT_LOCAL_GRANT = 3, // local grant of the lock
        ACT_REMOTE_GRANT = 4, // remote grant of the lock (change owners)
        ACT_REMOTE_RELEASE = 5, // remote release of a shared lock
      };
    public:
      unsigned time_units, lock_id, owner, action;
    };
#endif

    // prioritized list that maintains FIFO order within a priority level
    template <typename T>
    class pri_list : public std::list<T> {
    public:
      void pri_insert(T to_add) {
        // Common case: if the guy on the back has our priority or higher then just
        // put us on the back too.
        if (this->empty() || (this->back()->priority >= to_add->priority))
          this->push_back(to_add);
        else
        {
          // Uncommon case: go through the list until we find someone
          // who has a priority lower than ours.  We know they
          // exist since we saw them on the back.
          bool inserted = false;
          for (typename std::list<T>::iterator it = this->begin();
                it != this->end(); it++)
          {
            if ((*it)->priority < to_add->priority)
            {
              this->insert(it, to_add);
              inserted = true;
              break;
            }
          }
          // Technically we shouldn't need this, but just to be safe
          assert(inserted);
        }
      }
    };
     
  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
