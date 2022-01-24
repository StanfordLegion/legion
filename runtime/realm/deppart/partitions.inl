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

// INCLUDED from partitions.h - DO NOT INCLUDE THIS DIRECTLY

// index space partitioning for Realm

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/deppart/partitions.h"


namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class PartitioningMicroOp
  //

  // helper code to ship a microop to another node
  template <typename T>
  /*static*/ void PartitioningMicroOp::forward_microop(NodeID target,
						       PartitioningOperation *op,
						       T *microop)
  {
    // we're going to ship microop elsewhere, which means we always need an
    //  AsyncMicroOp to track it
    AsyncMicroOp *async_microop = new AsyncMicroOp(op, microop);
    op->add_async_work_item(async_microop);

    // message size can be highly variable, so compute it before allocating
    Serialization::ByteCountSerializer bcs;
    bool ok = microop->serialize_params(bcs);
    assert(ok);
    ActiveMessage<RemoteMicroOpMessage<T> > msg(target, bcs.bytes_used());
    msg->operation = op;
    msg->async_microop = async_microop;
    ok = microop->serialize_params(msg);
    assert(ok);
    msg.commit();
  }


};

