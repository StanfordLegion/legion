
/* Copyright 2024 NVIDIA Corporation
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
#include <limits.h>
#include <mpi.h>

#include "bootstrap.h"
#include "bootstrap_util.h"

#if !defined(REALM_MPI_HAS_COMM_SPLIT_TYPE)
#if (OMPI_MAJOR_VERSION*100 + OMPI_MINOR_VERSION) >= 107
#define REALM_MPI_HAS_COMM_SPLIT_TYPE 1
#endif
#endif

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

static int bootstrap_mpi_allgatherv(const void *sendbuf, void *recvbuf, int *sizes,
                                int *offsets, struct bootstrap_handle *handle)
{
  int status = MPI_SUCCESS;
  status = MPI_Allgatherv(sendbuf, sizes[handle->pg_rank], MPI_BYTE, recvbuf, sizes,
                          offsets, MPI_BYTE, bootstrap_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "MPI_Allgatherv failed\n");
out:
  return status;
}

static int populate_shared_ranks(bootstrap_handle_t *handle)
{
  int status = MPI_SUCCESS;
  MPI_Comm shared_comm = MPI_COMM_NULL;
  handle->shared_ranks = NULL;
  handle->num_shared_ranks = 0;
#if defined(REALM_MPI_HAS_COMM_SPLIT_TYPE)
  // This version is more accurate as it uses topology information rather than hostname,
  // but this is only available on certain versions of MPI.
  status = MPI_Comm_split_type(bootstrap_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                               &shared_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "MPI_Comm_split_type failed\n");
#else
  // This version hashes the name from MPI_Get_processor_name and splits the communicator
  // if they're the same.  This is highly inaccurate and has issues crossing container
  // boundaries and more exotic systems.
  char mpi_proc_name[MPI_MAX_PROCESSOR_NAME];
  int mpi_proc_name_len = sizeof(mpi_proc_name);
  unsigned mpi_name_hash = 5381;

  status = MPI_Get_processor_name(mpi_proc_name, &mpi_proc_name_len);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "MPI_Get_processor_name failed");

  for(size_t i = 0; i < mpi_proc_name_len; i++) {
    mpi_name_hash =
        (((mpi_name_hash << 5)) + mpi_name_hash) + (unsigned)mpi_proc_name[i];
  }

  // MPI_Comm_split requires the 'color' to be a positive int type, so zero out the high
  // order bits.
  mpi_name_hash &= INT_MAX;

  status = MPI_Comm_split(bootstrap_comm, (int)mpi_name_hash, handle->pg_rank, &shared_comm);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "MPI_Comm_split failed\n");
#endif

  status = MPI_Comm_size(shared_comm, &handle->num_shared_ranks);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                         "MPI_Comm_size failed\n");
  if(handle->num_shared_ranks > 0) {
    // Request the global rank ids from all the shared peers
    handle->shared_ranks =
        malloc(handle->num_shared_ranks * sizeof(handle->shared_ranks[0]));
    BOOTSTRAP_NULL_ERROR_JMP(handle->shared_ranks, status, BOOTSTRAP_ERROR_INTERNAL, out,
                             "Failed to allocate space for shared ranks\n");
    status = MPI_Allgather(&handle->pg_rank, 1, MPI_INT, handle->shared_ranks, 1, MPI_INT,
                           shared_comm);
    BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, out,
                            "MPI_Allgather failed\n");
  }

out:
  if(shared_comm != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&shared_comm);
  }
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

  if (handle->shared_ranks != NULL) {
    free(handle->shared_ranks);
  }

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

  status = populate_shared_ranks(handle);
  BOOTSTRAP_NE_ERROR_JMP(status, MPI_SUCCESS, BOOTSTRAP_ERROR_INTERNAL, error,
      "populate_shared_ranks failed\n");

  handle->barrier       = bootstrap_mpi_barrier;
  handle->bcast         = bootstrap_mpi_bcast;
  handle->gather        = bootstrap_mpi_gather;
  handle->allgather     = bootstrap_mpi_allgather;
  handle->alltoall      = bootstrap_mpi_alltoall;
  handle->allreduce_ull = bootstrap_mpi_allreduce_ull;
  handle->allgatherv    = bootstrap_mpi_allgatherv;
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
