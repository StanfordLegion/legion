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

#include "am_mpi.h"

#define AM_MSG_HEADER_SIZE (4 * sizeof(int) + sizeof(uintptr_t))
#define AM_BUF_SIZE_W_HEADER  (AM_BUF_SIZE + AM_MSG_HEADER_SIZE)

// if defined, replaces use of MPI_Put with two-sided send/recv
//define REALM_MPI_USE_TWO_SIDED_ONLY

static MPI_Win g_am_win = MPI_WIN_NULL;
static void *g_am_base = NULL;
static Realm::IncomingMessageManager *g_message_manager = NULL;
static Realm::atomic<unsigned int> num_threads(0);
static unsigned char buf_recv_list[AM_BUF_COUNT][AM_BUF_SIZE_W_HEADER];
static unsigned char *buf_recv = buf_recv_list[0];
static MPI_Request req_recv_list[AM_BUF_COUNT];
static int n_am_mult_recv = 5;
static int pre_initialized;
static int node_size;
static int node_this;
static MPI_Comm comm_medium;
int i_recv_list = 0;
static const int TAG_COMMAND = 0x1;
static const int TAG_DATA_BASE = 0x2;

namespace Realm {
namespace MPI {

  atomic<unsigned> data_tags_used(0);
  atomic<size_t> messages_sent(0);
  atomic<size_t> messages_rcvd(0);


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
        CHECK_MPI( MPI_Irecv(buf_recv_list[i], AM_BUF_SIZE_W_HEADER, MPI_CHAR, MPI_ANY_SOURCE, TAG_COMMAND, MPI_COMM_WORLD, &req_recv_list[i]) );
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

static void incoming_message_handled(NodeID sender,
				     uintptr_t comp_ptr,
				     uintptr_t /*unused*/)
{
    struct AM_msg msg;
    msg.type = 3; // completion reply
    msg.msgid = 0x7fff; // no hander needed
    msg.header_size = 0;
    msg.payload_size = 0;
    msg.comp_ptr = comp_ptr;

    CHECK_MPI( MPI_Send(&msg, AM_MSG_HEADER_SIZE, MPI_CHAR,
			sender, TAG_COMMAND, MPI_COMM_WORLD) );
    messages_sent.fetch_add(1);
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
            int32_t msg_tag;
            memcpy(&msg_tag, msg->stuff, sizeof(int32_t));
            header = msg->stuff + 4;
            payload = (char *) malloc(msg->payload_size);
	    payload_mode = PAYLOAD_FREE;
            CHECK_MPI( MPI_Recv(payload, msg->payload_size, MPI_BYTE, tn_src, msg_tag, comm_medium, &status) );
        } else if (msg->type == 2) {
            int32_t offset;
            memcpy(&offset, msg->stuff, sizeof(int32_t));
            header = msg->stuff + 4;
            payload = (char *) g_am_base + offset;
#ifdef REALM_MPI_USE_TWO_SIDED_ONLY
            // if we weren't allowed to do an MPI_Put on the sender, post
            //  receives here to actually get the data
            int32_t msg_tag;
            memcpy(&msg_tag, msg->stuff + msg->header_size + 4, sizeof(int32_t));
            CHECK_MPI( MPI_Recv(payload, msg->payload_size, MPI_BYTE, tn_src, msg_tag, MPI_COMM_WORLD, &status) );
#endif
	    payload_mode = PAYLOAD_KEEP;  // already in dest memory
        } else if (msg->type == 3) {
	    AMComplete(reinterpret_cast<void *>(msg->comp_ptr));
	    header = 0;
	    payload = 0;
        } else {
            assert(0 && "invalid message type");
            header = 0;
            payload = 0;
            msg->msgid = 0x7fff; // skips handler below
        }

	uintptr_t completion = 0;  // do we need to send a completion here?
        if (msg->msgid != 0x7fff) {
	    bool handled = g_message_manager->add_incoming_message(tn_src, msg->msgid,
								   header, msg->header_size,
								   PAYLOAD_COPY,
								   payload, msg->payload_size,
								   payload_mode,
								   ((msg->comp_ptr != 0) ?
								      incoming_message_handled :
								      0),
								   msg->comp_ptr,
								   0,
								   TimeLimit());
	    if (handled)
	        completion = msg->comp_ptr;
        }

        CHECK_MPI( MPI_Irecv(buf_recv_list[i_recv_list], AM_BUF_SIZE_W_HEADER, MPI_CHAR, MPI_ANY_SOURCE, TAG_COMMAND, MPI_COMM_WORLD, &req_recv_list[i_recv_list]) );
        i_recv_list = (i_recv_list + 1) % n_am_mult_recv;
        buf_recv = buf_recv_list[i_recv_list];

	if (completion)
	  incoming_message_handled(tn_src, completion, 0);

        messages_rcvd.fetch_add(1);
    }

}

void AMPoll_cancel()
{
    for (int  i = 0; i<n_am_mult_recv; i++) {
        CHECK_MPI( MPI_Cancel(&req_recv_list[i]) );
    }
}

static int32_t generate_payload_tag()
{
  // wrap a monotonically-incrementing `data_tags_used` counter to the
  //  range [TAG_DATA_BASE,MPI_TAG_UB]
  unsigned count = data_tags_used.fetch_add(1);
  int32_t tag = (TAG_DATA_BASE + (count % (MPI_TAG_UB + 1 - TAG_DATA_BASE)));
  return tag;
}

void AMSend(int tgt, int msgid, int header_size, int payload_size, const char *header, const char *payload, int payload_lines, int payload_line_stride, int has_dest, MPI_Aint dest, void *remote_comp)
{
    char buf_send[AM_BUF_SIZE_W_HEADER];

    struct AM_msg *msg = (struct AM_msg *)(buf_send);
    msg->msgid = msgid;
    msg->header_size = header_size;
    msg->payload_size = payload_size;
    msg->comp_ptr = reinterpret_cast<uintptr_t>(remote_comp);
    char *msg_header = msg->stuff;

    if (has_dest) {
        assert(g_am_win);
#ifndef REALM_MPI_USE_TWO_SIDED_ONLY
	if (payload_lines > 1) {
	  int line_size = payload_size / payload_lines;
	  for (int i = 0; i < payload_lines; i++)
	    CHECK_MPI( MPI_Put(payload + (i * payload_line_stride), line_size, MPI_BYTE,
			       tgt, dest + (i * line_size), line_size, MPI_BYTE, g_am_win) );
	} else
	  CHECK_MPI( MPI_Put(payload, payload_size, MPI_BYTE, tgt, dest, payload_size, MPI_BYTE, g_am_win) );
        CHECK_MPI( MPI_Win_flush(tgt, g_am_win) );
#endif

        msg->type = 2;
        int32_t dest_as_int32 = dest;
        memcpy(msg_header, &dest_as_int32, sizeof(int32_t));
        memcpy(msg_header + 4, header, header_size);
        int n = AM_MSG_HEADER_SIZE + 4 + header_size;
#ifdef REALM_MPI_USE_TWO_SIDED_ONLY
        int32_t msg_tag = generate_payload_tag();
        memcpy(msg_header + 4 + header_size, &msg_tag, sizeof(int32_t));
        n += 4;
#endif
        assert(tgt != node_this);
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, TAG_COMMAND, MPI_COMM_WORLD) );
#ifdef REALM_MPI_USE_TWO_SIDED_ONLY
        // after the header (which contains the destination address, use
        //  old-school MPI_Send to ship the data over
        assert(payload_lines <= 1); // TODO: multi-line means sending line count/stride to receiver
        CHECK_MPI( MPI_Send(payload, payload_size, MPI_BYTE, tgt, msg_tag, MPI_COMM_WORLD) );
#endif
    } else if (header_size + payload_size <= AM_BUF_SIZE) {
        msg->type = 0;
        if (header_size > 0) {
            memcpy(msg_header, header, header_size);
        }
        if (payload_size > 0) {
	    if (payload_lines > 1) {
	      int line_size = payload_size / payload_lines;
	      for (int i = 0; i < payload_lines; i++)
		memcpy(msg_header + header_size + (i * line_size),
		       payload + (i * payload_line_stride), line_size);
	    } else
	      memcpy(msg_header + header_size, payload, payload_size);
        }
        int n = AM_MSG_HEADER_SIZE + header_size + payload_size;
        assert(tgt != node_this);
        CHECK_MPI( MPI_Send(buf_send, n, MPI_CHAR, tgt, TAG_COMMAND, MPI_COMM_WORLD) );
    } else {
        msg->type = 1;
        int32_t msg_tag = generate_payload_tag();
        memcpy(msg_header, &msg_tag, sizeof(int32_t));
        memcpy(msg_header + 4, header, header_size);
        int n = AM_MSG_HEADER_SIZE + 4 + header_size;
        assert(tgt != node_this);
        CHECK_MPI( MPI_Send(buf_send, n, MPI_BYTE, tgt, TAG_COMMAND, MPI_COMM_WORLD) );
        assert(tgt != node_this);
	assert(payload_lines <= 1); // not supporting 2d payloads here yet
        CHECK_MPI( MPI_Send(payload, payload_size, MPI_BYTE, tgt, msg_tag, comm_medium) );

    }

    messages_sent.fetch_add(1);
}

} /* namespace MPI */
} /* namespace Realm */
