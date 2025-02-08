#ifndef __UCCLAYER_H__
#define __UCCLAYER_H__

#include <cassert>
#include <memory>

// #include <ucp/api/ucp.h>

#include "bootstrap/bootstrap.h"
#include "oob_group_comm.h"

namespace Realm {
  namespace ucc {
    class UCCComm {
      int rank;
      int world_sz;
      std::unique_ptr<OOBGroupComm> oob_comm;

      ucc_lib_h lib;
      ucc_team_h team{};
      ucc_context_h context{};

      // Helper functions
      ucc_status_t init_lib();
      ucc_status_t create_context();
      ucc_status_t create_team();

      // Currently simply assert fails on non-success state.
      void ucc_check(const ucc_status_t &st);

      // Currently simply assert fails on non-success state.
      ucc_status_t ucc_collective(ucc_coll_args_t &coll_args, ucc_coll_req_h &req);

    public:
      // @brief Construct UCC group communicator
      UCCComm(int rank, int world_sz, bootstrap_handle_t *bh);

      // @brief Initialize ucc library, create ucc context and team.
      ucc_status_t init();

      inline int get_rank() { return rank; };
      inline int get_world_size() { return world_sz; };

      // @brief Broadcast message from the process with root rank to all other
      //        processes of the group
      // @param buffer Starting address of the buffer
      // @param count Number of elements in the buffer
      // @param datatype Type of each of the element in the buffer
      // @param root Rank of the root
      // @return UCC_OK on success, error otherwise.
      ucc_status_t UCC_Bcast(void *buffer, int count, ucc_datatype_t datatype, int root);

      // @brief Gather values from a group of processes at root.
      // @param sbuf Starting address of the send buffer
      // @param sendcount Number of elements in the send buffer
      // @param sendtype Type of each of the element in the send buffer
      // @param rbuf Address of the receive buffer
      // @param rececount Number of elements to be received from each process.
      // @param recvtype Type of the elements in receive buffer
      // @prama root Rank of the root of gather operations
      // @return UCC_OK on success, error otherwise.
      ucc_status_t UCC_Gather(void *sbuf, int sendcount, ucc_datatype_t sendtype,
                              void *rbuf, int recvcount, ucc_datatype_t recvtype,
                              int root);

      // @brief Gather values from a all processes in a group and distributes
      //        to all processes in the group.
      // @param sbuf Starting address of the send buffer
      // @param sendcount Number of elements in the send buffer
      // @param sendtype Type of each of the element in the send buffer
      // @param rbuf Address of the receive buffer
      // @param rececount Number of elements to be received from each process.
      // @param recvtype Type of the elements in receive buffer
      // @return UCC_OK on success, error otherwise.
      ucc_status_t UCC_Allgather(void *sbuf, int sendcount, ucc_datatype_t sendtype,
                                 void *rbuf, int recvcount, ucc_datatype_t recvtype);

      ucc_status_t UCC_Allreduce(void *sbuf, void *rbuf, int count,
                                 ucc_datatype_t datatype, ucc_reduction_op_t op);

      ucc_status_t UCC_Allgatherv(void *sbuf, int count, ucc_datatype_t sendtype,
                                  void *rbuf, const std::vector<int> &recvcounts,
                                  const std::vector<int> &displs,
                                  ucc_datatype_t recvtype);

      ucc_status_t UCC_Barrier();

      ucc_status_t UCC_Finalize();
    };
  } // namespace ucc
} // namespace Realm
#endif
