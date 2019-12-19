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
static __thread int thread_id = 0;
static Realm::atomic<unsigned int> num_threads(0);
static unsigned char buf_recv_list[AM_BUF_COUNT][1024];
static unsigned char *buf_recv = buf_recv_list[0];
static MPI_Request req_recv_list[AM_BUF_COUNT];
static int n_am_mult_recv = 5;
static int node_size;
static int node_this;

namespace Realm {
namespace MPI {

#define AM_MSG_HEADER_SIZE 4 * sizeof(int)


void AM_Init(int *p_node_this, int *p_node_size)
{
    int mpi_thread_model;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mpi_thread_model);
    assert(mpi_thread_model == MPI_THREAD_MULTIPLE);
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_this);
    *p_node_size = node_size;
    *p_node_this = node_this;
}

void AM_Finalize()
{
    MPI_Request req_final;
    int ret;

    ret = MPI_Ibarrier(MPI_COMM_WORLD, &req_final);
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "MPI error in [Ibarrier(MPI_COMM_WORLD, &req_final)]\n");
        exit(-1);
    }
    while (1) {
        int is_done;
        MPI_Status status;
        ret = MPI_Test(&req_final, &is_done, &status);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Test(&req_final, &is_done, &status)]\n");
            exit(-1);
        }
        if (req_final == MPI_REQUEST_NULL) {
            break;
        }
        AMPoll();
    }

    AMPoll_cancel();

    MPI_Finalize();
}

void AM_init_long_messages(MPI_Win win, void *am_base)
{
    g_am_win = win;
    g_am_base = am_base;
}

void AMPoll()
{
    int i_got;
    int ret;
    struct AM_msg *msg;
    int tn_src;

    while (1) {
        int got_am;
        MPI_Status status;
        i_got = -1;
        ret = MPI_Testany(n_am_mult_recv, req_recv_list, &i_got, &got_am, &status);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Testany(n_am_mult_recv, req_recv_list, &i_got, &got_am, &status)]\n");
            exit(-1);
        }
        if (!got_am) {
            break;
        }
        buf_recv = buf_recv_list[i_got];

        msg = (struct AM_msg *) buf_recv;
        tn_src = status.MPI_SOURCE;

        char *header;
        char *payload;
        bool need_free = false;
        if (msg->type == 0) {
            header = msg->stuff;
            payload = msg->stuff + msg->header_size;
        } else if (msg->type == 1) {
            int msg_tag = *(int *)(msg->stuff);
            header = msg->stuff + sizeof(int);
            payload = (char *) malloc(msg->payload_size);
            need_free = true;
            ret = MPI_Recv(payload, msg->payload_size, MPI_BYTE, tn_src, msg_tag, MPI_COMM_WORLD, &status);
            if (ret != MPI_SUCCESS) {
                fprintf(stderr, "MPI error in [Recv(payload, msg->payload_size, MPI_BYTE, tn_src, msg_tag, MPI_COMM_WORLD, &status)]\n");
                exit(-1);
            }
        } else if (msg->type == 2) {
            int offset = *(int *)(msg->stuff);
            header = msg->stuff + sizeof(int);
            payload = (char *) g_am_base + offset;
        }

        Realm::ActiveMessageHandlerTable::MessageHandler handler = Realm::activemsg_handler_table.lookup_message_handler(msg->msgid);
        (*handler) (tn_src, header, payload, msg->payload_size);
        if (need_free) {
            free(payload);
        }
        ret = MPI_Irecv(buf_recv_list[i_got], 1024, MPI_CHAR, MPI_ANY_SOURCE, 0x1, MPI_COMM_WORLD, &req_recv_list[i_got]);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Irecv(buf_recv_list[i_got], 1024, MPI_CHAR, MPI_ANY_SOURCE, 0x1, MPI_COMM_WORLD, &req_recv_list[i_got])]\n");
            exit(-1);
        }
    }

}

void AMPoll_init()
{
    char *s;
    int ret;

    s = getenv("AM_MULT_RECV");
    if (s) {
        n_am_mult_recv = atoi(s);
    }
    for (int  i = 0; i<n_am_mult_recv; i++) {
        ret = MPI_Irecv(buf_recv_list[i], 1024, MPI_CHAR, MPI_ANY_SOURCE, 0x1, MPI_COMM_WORLD, &req_recv_list[i]);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Irecv(buf_recv_list[i], 1024, MPI_CHAR, MPI_ANY_SOURCE, 0x1, MPI_COMM_WORLD, &req_recv_list[i])]\n");
            exit(-1);
        }
    }
}

void AMPoll_cancel()
{
    int ret;

    for (int  i = 0; i<n_am_mult_recv; i++) {
        ret = MPI_Cancel(&req_recv_list[i]);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Cancel(&req_recv_list[i])]\n");
            exit(-1);
        }
    }
}

void AMSend(int tgt, int msgid, int header_size, int payload_size, const char *header, const char *payload, MPI_Aint dest)
{
    char buf_send[1024];
    int n;
    int ret;

    struct AM_msg *msg = (struct AM_msg *)(buf_send);
    msg->msgid = msgid;
    msg->header_size = header_size;
    msg->payload_size = payload_size;
    char *msg_header = msg->stuff;

    if (AM_MSG_HEADER_SIZE + header_size + payload_size < 1024) {
        msg->type = 0;
        memcpy(msg_header, header, header_size);
        if (payload_size > 0) {
            memcpy(msg_header + header_size, payload, payload_size);
        }
        n = AM_MSG_HEADER_SIZE + header_size + payload_size;
        assert(tgt != node_this);
        ret = MPI_Send(buf_send, n, MPI_CHAR, tgt, 0x1, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Send(buf_send, n, MPI_CHAR, tgt, 0x1, MPI_COMM_WORLD)]\n");
            exit(-1);
        }
    } else if (!dest) {
        msg->type = 1;
        int msg_tag = 0x0;
        if (thread_id == 0) {
            thread_id = num_threads.fetch_add_acqrel(1) + 1;
        }
        msg_tag = thread_id << 1;
        memcpy(msg_header, &msg_tag, 4);
        memcpy(msg_header + 4, header, header_size);
        n = AM_MSG_HEADER_SIZE + 4 + header_size;
        assert(tgt != node_this);
        ret = MPI_Send(buf_send, n, MPI_BYTE, tgt, 0x1, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Send(buf_send, n, MPI_BYTE, tgt, 0x1, MPI_COMM_WORLD)]\n");
            exit(-1);
        }
        assert(tgt != node_this);
        ret = MPI_Send(payload, payload_size, MPI_BYTE, tgt, msg_tag, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Send(payload, payload_size, MPI_BYTE, tgt, msg_tag, MPI_COMM_WORLD)]\n");
            exit(-1);
        }
    } else {
        ret = MPI_Win_lock(MPI_LOCK_SHARED, tgt, 0, g_am_win);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Win_lock(MPI_LOCK_SHARED, tgt, 0, g_am_win)]\n");
            exit(-1);
        }
        assert(g_am_win);
        ret = MPI_Put(payload, payload_size, MPI_BYTE, tgt, dest, payload_size, MPI_BYTE, g_am_win);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Put(payload, payload_size, MPI_BYTE, tgt, dest, payload_size, MPI_BYTE, g_am_win)]\n");
            exit(-1);
        }
        ret = MPI_Win_unlock(tgt, g_am_win);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Win_unlock(tgt, g_am_win)]\n");
            exit(-1);
        }

        msg->type = 2;
        memcpy(msg_header, &dest, 4);
        memcpy(msg_header + 4, header, header_size);
        n = AM_MSG_HEADER_SIZE + 4 + header_size;
        assert(tgt != node_this);
        ret = MPI_Send(buf_send, n, MPI_BYTE, tgt, 0x1, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
            fprintf(stderr, "MPI error in [Send(buf_send, n, MPI_BYTE, tgt, 0x1, MPI_COMM_WORLD)]\n");
            exit(-1);
        }
    }
}

} /* namespace MPI */
} /* namespace Realm */
