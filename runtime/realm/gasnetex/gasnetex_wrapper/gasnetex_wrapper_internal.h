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

#ifndef GASNETEX_WRAPPER_INTERNAL_H
#define GASNETEX_WRAPPER_INTERNAL_H

#ifndef GASNET_PAR
#if defined(GASNET_SEQ) || defined(GASNET_PARSYNC)
#error Realm requires GASNet-EX be used in parallel threading mode!
#else
#define GASNET_PAR
#endif
#endif
#include <gasnetex.h>

// there are two independent "version" that we may need to consider for
//  conditional compilation:
//
// 1) REALM_GEX_RELEASE refers to specific releases - GASNet-EX uses year.month
//      for major.minor, and we'll assume no more than 100 patch levels to
//      avoid conflicts, but there is no guarantee of chronological
//      monotonicity of behavior, so tests should be either equality against
//      a specific release or a bounded comparison when two or more consecutive
//      releases are of interest.  However, there should never be anything of
//      form: if REALM_GEX_RELEASE >= xyz
#define REALM_GEX_RELEASE                                                                \
  ((10000 * GASNET_RELEASE_VERSION_MAJOR) + (100 * GASNET_RELEASE_VERSION_MINOR) +       \
   GASNET_RELEASE_VERSION_PATCH)

// 2) REALM_GEX_API refers to versioning of the GASNet-EX specification -
//      currently this is defined in terms of major.minor, but we'll include
//      space for a patch level if that ever becomes important.  In contrast to
//      the release numbering, we will assume that the specification is
//      roughly monotonic in that a change is expected to remain in future
//      specs except for hopefully-uncommon cases where it changes again
#define REALM_GEX_API ((10000 * GEX_SPEC_VERSION_MAJOR) + (100 * GEX_SPEC_VERSION_MINOR))

#if REALM_GEX_API < 1200
#error Realm depends on GASNet-EX features that first appeared in the 0.12 spec, first available in the 2020.11.0 release.  For earlier versions of GASNet-EX, use the legacy API via the gasnet1 network layer.
#include <stop_compilation_due_to_gasnetex_version_mismatch>
#endif

// post 2020.11.0, GASNet has defines that say which operations are native
//  rather than emulated by their reference implementation - those defines
//  aren't there for 2020.11.0, but the only one that version has that we
//  care about is NPAM medium
#if REALM_GEX_RELEASE == 20201100
// NOTE: technically it's only native for the IBV/ARIES/SMP conduits,
//  but enable it as well on the MPI conduit so that we get more test
//  coverage of the code paths (and it's probably not making the MPI
//  conduit performance any worse)
#if defined(GASNET_CONDUIT_IBV) || defined(GASNET_CONDUIT_ARIES) ||                      \
    defined(GASNET_CONDUIT_SMP) || defined(GASNET_CONDUIT_MPI)
#define GASNET_NATIVE_NP_ALLOC_REQ_MEDIUM
#endif
#endif

// the GASNet-EX API defines the GEX_FLAG_IMMEDIATE flag to be a best-effort
//  thing, with calls that accept the flag still being allowed to block -
//  as of 2022.3.0, for any conduit other than aries "best effort" is actually
//  "no effort" for RMA operations and we want to avoid using them in
//  immediate-mode situations
// NOTE: as with the NPAM stuff above, we'll pretend that MPI honors it as
//  well so that we get code coverage in CI tests
#if defined(GASNET_CONDUIT_ARIES) || defined(GASNET_CONDUIT_MPI)
#define REALM_GEX_RMA_HONORS_IMMEDIATE_FLAG
#endif

// eliminate GASNet warnings for unused static functions
// REALM_ATTR_UNUSED(thing) - indicate that `thing` is unused
#define REALM_ATTR_UNUSED(thing) thing __attribute__((unused))

#include <gasnet_tools.h>
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning1) = (void *)
    _gasneti_threadkey_init;
REALM_ATTR_UNUSED(static const void *ignore_gasnet_warning2) = (void *)
    _gasnett_trace_printf_noop;

#define CHECK_GEX(cmd)                                                                   \
  do {                                                                                   \
    int ret = (cmd);                                                                     \
    if(ret != GASNET_OK) {                                                               \
      fprintf(stderr, "GEX: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret),       \
              gasnet_ErrorDesc(ret));                                                    \
      exit(1);                                                                           \
    }                                                                                    \
  } while(0)

struct gex_wrapper_handle_s;

namespace Realm {
  namespace GASNetEXHandlers {

    extern gex_AM_Entry_t handler_table[];
    extern size_t handler_table_size;

    void init_gex_handler_fnptr(struct gex_wrapper_handle_s *handle);
  }; // namespace GASNetEXHandlers
};   // namespace Realm

#endif