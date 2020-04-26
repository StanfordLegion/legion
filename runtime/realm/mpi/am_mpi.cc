/* Copyright 2019 Argonne National Laboratory
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

#include "am_mpi.h"


static MPI_Win g_am_win = MPI_WIN_NULL;
static void *g_am_base = NULL;
static Realm::IncomingMessageManager *g_message_manager = NULL;
static __thread int thread_id = 0;
static __thread int am_seq = 0;
static Realm::atomic<unsigned int> num_threads(0);
static unsigned char buf_recv_list[AM_BUF_COUNT][1024];
static unsigned char *buf_recv = buf_recv_list[0];
static MPI_Request req_recv_list[AM_BUF_COUNT];
static int n_am_mult_recv = 5;
static int pre_initialized;
static int node_size;
static int node_this;
static MPI_Comm comm_medium;
int i_recv_list = 0;

namespace Realm {
namespace MPI {

#define AM_MSG_HEADER_SIZE 4 * sizeof(int)


void AM_Init(int *p_node_this, int *p_node_size)
{
    char *s;

    MPI_Initialized(&pre_initialized);
    if (pre_initialized) {
        int mpi_thread_model;
        MPI_Query_thread(&mpi_thread_model);
        assert(mpi_thread_model == MPI_THREAD_MULTIPLE);
    } else {
        int mpi_thread_model;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mpi_thread_model);
        assert(mpi_thread_model == MPI_THREAD_MULTIPLE);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_this);
    *p_node_size = node_size;
    *p_node_this = node_this;

    s = getenv("AM_MULT_RECV");
    if (s) {
        n_am_mult_recv = atoi(s);
    }
    for (int  i = 0; i<n_am_mult_recv; i++) {
        CHECK_MPI( MPI_Irecv(buf_recv_list[i], 1024, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv_list[i]) );
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_medium);
}

void AM_Finalize()
{
    MPI_Request req_final;

    CHECK_MPI( MPI_Ibarrier(MPI_COMM_WORLD, &req_final) );
    while (1) {
        int is_done;
        MPI_Status status;
        CHECK_MPI( MPI_Test(&req_final, &is_done, &status) );
        if (req_final == MPI_REQUEST_NULL) {
            break;
        }
        AMPoll();
    }

    AMPoll_cancel();
    CHECK_MPI( MPI_Comm_free(&comm_medium) );

    if (!pre_initialized) {
        MPI_Finalize();
    }
}

void AM_init_long_messages(MPI_Win win, void *am_base,
			   Realm::IncomingMessageManager *message_manager)
{
    g_am_win = win;
    g_am_base = am_base;
    g_message_manager = message_manager;
}

void AMPoll()
{
    struct AM_msg *msg;
    int tn_src;

    while (1) {
        int got_am;
        MPI_Status status;
        CHECK_MPI( MPI_Test(&req_recv_list[i_recv_list], &got_am, &status) );
        if (!got_am) {
            break;
        }
        msg = (struct AM_msg *) buf_recv;

        tn_src = status.MPI_SOURCE;

        char *header;
        char *payload;
	int payload_mode = PAYLOAD_NONE;
        if (msg->type == 0) {
            header = msg->stuff;
            payload = msg->stuff + msg->header_size;
	    payload_mode = PAYLOAD_COPY;
        } else if (msg->type == 1) {
            header = msg->stuff + 4;

            int msg_tag = *(int32_t *)(msg->stuff);
            payload = (char *) malloc(msg->payload_size);
	    payload_mode = PAYLOAD_FREE;
            CHECK_MPI( MPI_Recv(payload, msg->payload_size, MPI_BYTE, tn_src, msg_tag, comm_medium, &status) );
        } else if (msg->type == 2) {
            int offset = *(int32_t *)(msg->stuff);
            header = msg->stuff + 4;
            payload = (char *) g_am_base + offset;
	    payload_mode = PAYLOAD_KEEP;  // already in dest memory
        } else {
            assert(0 && "invalid message type");
            header = 0;
            payload = 0;
            msg->msgid = 0x7fff; // skips handler below
        }

        if (msg->msgid != 0x7fff) {
	    g_message_manager->add_incoming_message(tn_src, msg->msgid,
						    header, msg->header_size,
						    PAYLOAD_COPY,
						    payload, msg->payload_size,
						    payload_mode,
						    0, 0);
        }

        CHECK_MPI( MPI_Irecv(buf_recv_list[i_recv_list], 1024, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv_list[i_recv_list]) );
        i_recv_list = (i_recv_list + 1) % n_am_mult_recv;
        buf_recv = buf_recv_list[i_recv_list];
    }

}

void AMPoll_cancel()
{
    for (int  i = 0; i<n_am_mult_recv; i++) {
        CHECK_MPI( MPI_Cancel(&req_recv_list[i]) );
    }
}

void AMSend(int tgt, int msgid, int header_size, int payload_size, const char *header, const char *payload, int has_dest, MPI_Aint dest)
{
    char buf_send[1024];

    struct AM_msg *msg = (struct AM_msg *)(buf_send);
    msg->msgid = msgid;
    msg->header_size = header_size;
    msg->payload_size = payload_size;
    char *msg_header = msg->stuff;

    if (has_dest) {
        assert(g_am_win);
        CHECK_MPI( MPI_Put(payload, payload_size, MPI_BYTE, tgt, dest, payload_size, MPI_BYTE, g_am_win) );
        CHECK_MPI( MPI_Win_flush(tgt, g_am_win) );

        msg->type = 2;
        *((int32_t *) msg_header) = (int32_t) dest;
        memcpy(msg_header + 4, header, header_size);
        int n = AM_MSG_HEADER_SIZE + 4 + header_size;
        assert(tgt != node_this);
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, 0x1, MPI_COMM_WORLD) );
    } else if (AM_MSG_HEADER_SIZE + header_size + payload_size < 1024) {
        msg->type = 0;
        if (header_size > 0) {
            memcpy(msg_header, header, header_size);
        }
        if (payload_size > 0) {
            memcpy(msg_header + header_size, payload, payload_size);
        }
        int n = AM_MSG_HEADER_SIZE + header_size + payload_size;
        assert(tgt != node_this);
        CHECK_MPI( MPI_Send(buf_send, n, MPI_CHAR, tgt, 0x1, MPI_COMM_WORLD) );
    } else {
        msg->type = 1;
        int msg_tag = 0x0;
        if (thread_id == 0) {
            thread_id = num_threads.fetch_add_acqrel(1) + 1;
        }
        am_seq = (am_seq + 1) & 0x1f;
        msg_tag = (thread_id << 10) + am_seq;
        *((int32_t *) msg_header) = (int32_t) msg_tag;
        memcpy(msg_header + 4, header, header_size);
        int n = AM_MSG_HEADER_SIZE + 4 + header_size;
        assert(tgt != node_this);
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, 0x1, MPI_COMM_WORLD) );
        assert(tgt != node_this);
        CHECK_MPI( MPI_Send(payload, payload_size, MPI_BYTE, tgt, msg_tag, comm_medium) );

    }
}

} /* namespace MPI */
} /* namespace Realm */
