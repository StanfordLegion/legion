/* Copyright 2022 Argonne National Laboratory
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

#ifndef AM_MPI_H_INCLUDED
#define AM_MPI_H_INCLUDED

#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include <cstdio>
#include <cassert>
#include "realm/atomics.h"
#include "realm/activemsg.h"

#define AM_BUF_COUNT 128

// size of posted receives (i.e. header+payload limit that can be sent
//  eagerly)
#define AM_BUF_SIZE  4096

#define CHECK_MPI(cmd) do { \
  int ret = (cmd); \
  if(ret != MPI_SUCCESS) { \
    fprintf(stderr, "MPI: %s = %d\n", #cmd, ret); \
    exit(1); \
  } \
} while(0)

namespace Realm {
namespace MPI {

extern atomic<size_t> messages_sent, messages_rcvd;

struct AM_msg {
    int type;
    int msgid;
    int header_size;
    int payload_size;
    uintptr_t comp_ptr;
    char stuff[1];
};

void AM_Init(int *p_node_this, int *p_node_size);
void AM_Finalize();
void AM_init_long_messages(MPI_Win win, void *am_base,
			   Realm::IncomingMessageManager *message_manager);
void AMPoll();
void AMPoll_cancel();
void AMSend(int tgt, int msgid, int header_size, int payload_size, const char *header, const char *payload, int payload_lines, int payload_line_stride, int has_dest, MPI_Aint dest, void *remote_comp);

// must be defined by caller of AMSend
void AMComplete(void *remote_comp);

} /* namespace MPI */
} /* namespace Realm */

#endif /* AM_MPI_H_INCLUDED */
