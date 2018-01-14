/* Copyright 2018 Stanford University, NVIDIA Corporation
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
    void contribute_dense_rect_list(const std::vector<Rect<N,T> >& rects);
    void contribute_raw_rects(const Rect<N,T>* rects, size_t count, bool last);

    // adds a microop as a waiter for valid sparsity map data - returns true
    //  if the uop is added to the list (i.e. will be getting a callback at some point),
    //  or false if the sparsity map became valid before this call (i.e. no callback)
    bool add_waiter(PartitioningMicroOp *uop, bool precise);

    void remote_data_request(NodeID requestor, bool send_precise, bool send_approx);
    void remote_data_reply(NodeID requestor, bool send_precise, bool send_approx);

    SparsityMap<N,T> me;

  protected:
    void finalize(void);
    
    int remaining_contributor_count;
    GASNetHSL mutex;
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

    void init(ID _me, unsigned _init_owner);

    ID me;
    unsigned owner;
    SparsityMapImplWrapper *next_free;
    DynamicTemplates::TagType type_tag;
    void *map_impl;  // actual implementation

    template <int N, typename T>
    SparsityMapImpl<N,T> *get_or_create(SparsityMap<N,T> me);

    void destroy(void);
  };


  ///////////////////////////////
  //
  // active messages

  class FragmentAssembler {
  public:
    FragmentAssembler(void);
    ~FragmentAssembler(void);

    // returns a sequence ID that may not be unique, but hasn't been used in a 
    //   long time
    int get_sequence_id(void);

    // adds a fragment to the list, returning true if this is the last one from
    //  a sequence
    bool add_fragment(NodeID sender, int sequence_id, int sequence_count);

  protected:
    int next_sequence_id;
    GASNetHSL mutex; // protects the fragments map
    std::map<NodeID, std::map<int, int> > fragments;
  };

  struct RemoteSparsityRequestMessage {
    struct RequestArgs {
      NodeID sender;
      DynamicTemplates::TagType type_tag;
      ID::IDType sparsity_id;
      bool send_precise;
      bool send_approx;
    };

    struct DecodeHelper {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args);
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<REMOTE_SPARSITY_REQUEST_MSGID,
                                      RequestArgs,
                                      handle_request> Message;

    template <int N, typename T>
    static void send_request(NodeID target, SparsityMap<N,T> sparsity,
			     bool send_precise, bool send_approx);
  };

  struct SetContribCountMessage {
    struct RequestArgs {
      DynamicTemplates::TagType type_tag;
      ID::IDType sparsity_id;
      int count;
    };

    struct DecodeHelper {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args);
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<SET_CONTRIB_COUNT_MSGID,
                                      RequestArgs,
                                      handle_request> Message;

    template <int N, typename T>
    static void send_request(NodeID target, SparsityMap<N,T> sparsity, int count);
  };
    
  struct RemoteSparsityContribMessage {
    struct RequestArgs : public BaseMedium {
      NodeID sender;
      DynamicTemplates::TagType type_tag;
      ID::IDType sparsity_id;
      int sequence_id;
      int sequence_count;
    };

    struct DecodeHelper {
      template <typename NT, typename T>
      static void demux(const RequestArgs *args, const void *data, size_t datalen);
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<REMOTE_SPARSITY_CONTRIB_MSGID,
                                       RequestArgs,
                                       handle_request> Message;

    template <int N, typename T>
    static void send_request(NodeID target, SparsityMap<N,T> sparsity,
			     int sequence_id, int sequence_count,
			     const Rect<N,T> *rects, size_t count);
  };

}; // namespace Realm

#endif // REALM_DEPPART_SPARSITY_IMPL_H

#include "realm/deppart/sparsity_impl.inl"
