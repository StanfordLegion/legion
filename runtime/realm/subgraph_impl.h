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

// Realm subgraph implementation

#ifndef REALM_SUBGRAPH_IMPL_H
#define REALM_SUBGRAPH_IMPL_H

#include "realm/subgraph.h"
#include "realm/id.h"
#include "realm/event_impl.h"

namespace Realm {

  struct SubgraphScheduleEntry {
    SubgraphDefinition::OpKind op_kind;
    unsigned op_index;
    std::vector<std::pair<unsigned, int> > preconditions;
    unsigned first_interp, num_interps;
    unsigned intermediate_event_base, intermediate_event_count;
    bool is_final_event;
  };

  class SubgraphImpl {
  public:
    SubgraphImpl();

    ~SubgraphImpl();

    void init(ID _me, int _owner);

    // compile/analyze the subgraph
    bool compile(void);

    void instantiate(const void *args, size_t arglen,
		     const ProfilingRequestSet& prs,
		     span<const Event> preconditions,
		     span<const Event> postconditions,
		     Event start_event, Event finish_event,
		     int priority_adjust);

    void destroy(void);

    class DeferredDestroy : public EventWaiter {
    public:
      void defer(SubgraphImpl *_subgraph, Event wait_on);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      SubgraphImpl *subgraph;
    };

  public:
    ID me;
    SubgraphImpl *next_free;
    SubgraphDefinition *defn;
    std::vector<SubgraphScheduleEntry> schedule;
    size_t num_intermediate_events, num_final_events, max_preconditions;

    DeferredDestroy deferred_destroy;
  };

  // active messages

  struct SubgraphInstantiateMessage {
    Subgraph subgraph;
    Event wait_on, finish_event;
    size_t arglen;
    int priority_adjust;

    static void handle_message(NodeID sender, const SubgraphInstantiateMessage& msg,
			       const void *data, size_t datalen);
  };

  struct SubgraphDestroyMessage {
    Subgraph subgraph;
    Event wait_on;

    static void handle_message(NodeID sender, const SubgraphDestroyMessage &msg,
			       const void *data, size_t datalen);
  };


}; // namespace Realm

#endif
