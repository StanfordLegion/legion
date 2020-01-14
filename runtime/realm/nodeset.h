/* Copyright 2020 Stanford University, NVIDIA Corporation
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

// dynamic node set implementation for Realm

#ifndef REALM_NODESET_H
#define REALM_NODESET_H

#include "bitmask.h"
#include "realm_c.h"

#ifndef REALM_MAX_NUM_NODES
#define REALM_MAX_NUM_NODES         1024 // must be a power of 2
#endif

// The following macros are used in the NodeMask instantiation of BitMask
// If you change one you probably have to change the others too
#define REALM_NODE_MASK_NODE_TYPE           uint64_t
#define REALM_NODE_MASK_NODE_SHIFT          6
#define REALM_NODE_MASK_NODE_MASK           0x3F
#define REALM_NODE_MASK_NODE_ALL_ONES       0xFFFFFFFFFFFFFFFF
namespace Realm {
#if defined(__AVX__)
#if (REALM_MAX_NUM_NODES > 256)
    typedef AVXTLBitMask<REALM_MAX_NUM_NODES> NodeMask;
#elif (REALM_MAX_NUM_NODES > 128)
    typedef AVXBitMask<REALM_MAX_NUM_NODES> NodeMask;
#elif (REALM_MAX_NUM_NODES > 64)
    typedef SSEBitMask<REALM_MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<REALM_NODE_MASK_NODE_TYPE,REALM_MAX_NUM_NODES,
                    REALM_NODE_MASK_NODE_SHIFT,
                    REALM_NODE_MASK_NODE_MASK> NodeMask;
#endif
#elif defined(__SSE2__)
#if (REALM_MAX_NUM_NODES > 128)
    typedef SSETLBitMask<REALM_MAX_NUM_NODES> NodeMask;
#elif (REALM_MAX_NUM_NODES > 64)
    typedef SSEBitMask<REALM_MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<REALM_NODE_MASK_NODE_TYPE,REALM_MAX_NUM_NODES,
                    REALM_NODE_MASK_NODE_SHIFT,
                    REALM_NODE_MASK_NODE_MASK> NodeMask;
#endif
#elif defined(__ALTIVEC__)
#if (REALM_MAX_NUM_NODES > 128)
    typedef PPCTLBitMask<REALM_MAX_NUM_NODES> NodeMask;
#elif (REALM_MAX_NUM_NODES > 64)
    typedef PPCBitMask<REALM_MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<REALM_NODE_MASK_NODE_TYPE,REALM_MAX_NUM_NODES,
                    REALM_NODE_MASK_NODE_SHIFT,
                    REALM_NODE_MASK_NODE_MASK> NodeMask;
#endif
#else
#if (REALM_MAX_NUM_NODES > 64)
    typedef TLBitMask<REALM_NODE_MASK_NODE_TYPE,REALM_MAX_NUM_NODES,
                      REALM_NODE_MASK_NODE_SHIFT,
                      REALM_NODE_MASK_NODE_MASK> NodeMask;
#else
    typedef BitMask<REALM_NODE_MASK_NODE_TYPE,REALM_MAX_NUM_NODES,
                    REALM_NODE_MASK_NODE_SHIFT,
                    REALM_NODE_MASK_NODE_MASK> NodeMask;
#endif
#endif
    typedef IntegerSet<realm_address_space_t,NodeMask> NodeSet;
};
#undef REALM_NODE_MASK_NODE_SHIFT
#undef REALM_NODE_MASK_NODE_MASK

#endif // ifndef REALM_NODESET_H
