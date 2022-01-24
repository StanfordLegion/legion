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

// implementation sparsity maps

#ifndef REALM_DEPPART_SPARSITY_IMPL_H
#define REALM_DEPPART_SPARSITY_IMPL_H

#include "realm/indexspace.h"
#include "realm/sparsity.h"
#include "realm/id.h"

#include "realm/activemsg.h"
#include "realm/nodeset.h"
#include "realm/atomics.h"

namespace Realm {

  class PartitioningMicroOp;

  template <int N, typename T>
  class SparsityMapImpl : public SparsityMapPublicImpl<N,T> {
  public:
    SparsityMapImpl(SparsityMap<N,T> _me);

    // actual implementation - SparsityMapPublicImpl's version just calls this one
    Event make_valid(bool precise = true);

    static SparsityMapImpl<N,T> *lookup(SparsityMap<N,T> sparsity);

    // methods used in the population of a sparsity map

    // when we plan out a partitioning operation, we'll know how many
    //  different uops are going to contribute something (or nothing) to
    //  the sparsity map - once all of those contributions arrive, we can
    //  finalize the sparsity map
    void set_contributor_count(int count);

    void contribute_nothing(void);
    void contribute_dense_rect_list(const std::vector<Rect<N,T> >& rects,
                                    bool disjoint);
    void contribute_raw_rects(const Rect<N,T>* rects, size_t count,
			      size_t piece_count, bool disjoint,
                              size_t total_count);

    // adds a microop as a waiter for valid sparsity map data - returns true
    //  if the uop is added to the list (i.e. will be getting a callback at some point),
    //  or false if the sparsity map became valid before this call (i.e. no callback)
    bool add_waiter(PartitioningMicroOp *uop, bool precise);

    void remote_data_request(NodeID requestor, bool send_precise, bool send_approx);
    void remote_data_reply(NodeID requestor, bool send_precise, bool send_approx);

    SparsityMap<N,T> me;

    struct RemoteSparsityRequest {
      SparsityMap<N,T> sparsity;
      bool send_precise;
      bool send_approx;

      static void handle_message(NodeID sender,
				 const RemoteSparsityRequest &msg,
				 const void *data, size_t datalen);
    };

    struct RemoteSparsityContrib {
      SparsityMap<N,T> sparsity;
      size_t piece_count; // non-zero only on last piece of contribution
      bool disjoint; // if set, all rectangles (from this source and any other)
                     //   are known to be disjoint
      size_t total_count; // if non-zero, advertises the known total number of
                          //  recangles in the sparsity map

      static void handle_message(NodeID sender,
				 const RemoteSparsityContrib &msg,
				 const void *data, size_t datalen);
    };

    struct SetContribCountMessage {
      SparsityMap<N,T> sparsity;
      size_t count;

      static void handle_message(NodeID sender,
				 const SetContribCountMessage &msg,
				 const void *data, size_t datalen);
    };

  protected:
    void finalize(void);

    static ActiveMessageHandlerReg<RemoteSparsityRequest> remote_sparsity_request_reg;
    static ActiveMessageHandlerReg<RemoteSparsityContrib> remote_sparsity_contrib_reg;
    static ActiveMessageHandlerReg<SetContribCountMessage> set_contrib_count_msg_reg;

    atomic<int> remaining_contributor_count;
    atomic<int> total_piece_count, remaining_piece_count;
    Mutex mutex;
    std::vector<PartitioningMicroOp *> approx_waiters, precise_waiters;
    bool precise_requested, approx_requested;
    Event precise_ready_event, approx_ready_event;
    NodeSet remote_precise_waiters, remote_approx_waiters;
    NodeSet remote_sharers;
    size_t sizeof_precise;
  };

  // we need a type-erased wrapper to store in the runtime's lookup table
  class SparsityMapImplWrapper {
  public:
    static const ID::ID_Types ID_TYPE = ID::ID_SPARSITY;

    SparsityMapImplWrapper(void);
    ~SparsityMapImplWrapper(void);

    void init(ID _me, unsigned _init_owner);

    ID me;
    unsigned owner;
    SparsityMapImplWrapper *next_free;
    atomic<DynamicTemplates::TagType> type_tag;
    atomic<void *> map_impl;  // actual implementation

    // need a type-erased deleter
    typedef void(*Deleter)(void *);
    Deleter map_deleter;

    template <int N, typename T>
    SparsityMapImpl<N,T> *get_or_create(SparsityMap<N,T> me);

    void destroy(void);
  };


}; // namespace Realm

#endif // REALM_DEPPART_SPARSITY_IMPL_H

#include "realm/deppart/sparsity_impl.inl"
