
/* Copyright 2023 NVIDIA Corporation
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

#include <stdlib.h>
#include <mpi.h>

#include "bootstrap.h"
#include "bootstrap_util.h"

static MPI_Comm bootstrap_comm = MPI_COMM_NULL;
static int realm_ucp_initialized_mpi = 0;

MPI_Op mpi_ops[REDUCTION_LAST] = {
  [REDUCTION_SUM] = MPI_SUM
};

static int bootstrap_mpi_barrier(struct bootstrap_handle *handle) {
  int status = MPI_SUCCESS;

  status = MPI_Barrier(bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "MPI_Barrier failed\n");

out:
  return status;
}

static int bootstrap_mpi_bcast(void *buf, int length, int root,
    struct bootstrap_handle *handle) {
  int status = MPI_SUCCESS;

  status = MPI_Bcast(buf, length, MPI_BYTE, root, bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "MPI_Bcast failed\n");

out:
  return status;
}

static int bootstrap_mpi_gather(const void *sendbuf, void *recvbuf, int length, int root,
    struct bootstrap_handle *handle) {
  int status = MPI_SUCCESS;

  status = MPI_Gather(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, root, bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "MPI_Gather failed\n");

out:
  return status;
}

static int bootstrap_mpi_allgather(const void *sendbuf, void *recvbuf, int length,
    struct bootstrap_handle *handle) {
  int status = MPI_SUCCESS;

  status = MPI_Allgather(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "MPI_Allgather failed\n");

out:
  return status;
}

static int bootstrap_mpi_alltoall(const void *sendbuf, void *recvbuf, int length,
    struct bootstrap_handle *handle) {
  int status = MPI_SUCCESS;

  status = MPI_Alltoall(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "MPI_Alltoall failed\n");

out:
  return status;
}

static int bootstrap_mpi_allreduce_ull(const void *sendbuf, void *recvbuf, int count,
    enum reduction_op op, struct bootstrap_handle *handle) {
  int status = MPI_SUCCESS;

  status = MPI_Allreduce(sendbuf, recvbuf, count, MPI_UNSIGNED_LONG_LONG,
      mpi_ops[op], bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "MPI_Allreduce ULL failed\n");

out:
  return status;
}

static int bootstrap_mpi_finalize(bootstrap_handle_t *handle) {
  int status = MPI_SUCCESS, finalized;

  /* Ensure user hasn't finalized MPI before finalizing UCX bootstrap */
  status = MPI_Finalized(&finalized);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "MPI_Finalized failed\n");

  if (finalized) {
    status = BOOTSTRAP_ERROR_INTERNAL;
    BOOTSTRAP_ERROR_PRINT("MPI is finalized\n");
    goto out;
  }

  status = MPI_Comm_free(&bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
      "Freeing bootstrap communicator failed\n");

  if (realm_ucp_initialized_mpi)
    MPI_Finalize();

out:
  return status;
}

int realm_ucp_bootstrap_plugin_init(void *mpi_comm, bootstrap_handle_t *handle) {
  int status = MPI_SUCCESS, initialized = 0, finalized = 0;
  MPI_Comm src_comm;

  if (NULL == mpi_comm)
    src_comm = MPI_COMM_WORLD;
  else
    src_comm = *((MPI_Comm *)mpi_comm);

  status = MPI_Initialized(&initialized);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, error,
      "MPI_Initialized failed\n");

  status = MPI_Finalized(&finalized);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, error,
      "MPI_Finalized failed\n");

  if (!initialized && !finalized) {
    int provided_level;
    // Legate collInit requires MPI_THREAD_MULTIPLE
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided_level);
    realm_ucp_initialized_mpi = 1;

    // Because MPI was not initialized, the only communicators that would
    // have been valid to pass are the predefined communicators
    if (src_comm != MPI_COMM_WORLD && src_comm != MPI_COMM_SELF) {
      status = BOOTSTRAP_ERROR_INTERNAL;
      BOOTSTRAP_ERROR_PRINT("Invalid communicator\n");
      goto error;
    }
  } else if (finalized) {
    status = BOOTSTRAP_ERROR_INTERNAL;
    BOOTSTRAP_ERROR_PRINT("MPI is finalized\n");
    goto error;
  }


  status = MPI_Comm_dup(src_comm, &bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, error,
      "Creating bootstrap communicator failed\n");

  status = MPI_Comm_rank(bootstrap_comm, &handle->pg_rank);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, error,
      "MPI_Comm_rank failed\n");

  status = MPI_Comm_size(bootstrap_comm, &handle->pg_size);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, error,
      "MPI_Comm_size failed\n");

  handle->barrier       = bootstrap_mpi_barrier;
  handle->bcast         = bootstrap_mpi_bcast;
  handle->gather        = bootstrap_mpi_gather;
  handle->allgather     = bootstrap_mpi_allgather;
  handle->alltoall      = bootstrap_mpi_alltoall;
  handle->allreduce_ull = bootstrap_mpi_allreduce_ull;
  handle->finalize      = bootstrap_mpi_finalize;

  goto out;

error:
  if (realm_ucp_initialized_mpi) {
    MPI_Finalize();
    realm_ucp_initialized_mpi = 0;
  }

out:
  return status;
}
