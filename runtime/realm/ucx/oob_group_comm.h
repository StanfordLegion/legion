#ifndef __OOB_GROUPCOMM_H__
#define __OOB_GROUPCOMM_H__

#include <memory>
#include <vector>

#include <ucc/api/ucc.h>

#include "bootstrap/bootstrap.h"

namespace Realm {
  namespace ucc {
    // @class OOBGroupComm
    // @brief Class  responsible for  out-of-band group communication  duing UCC
    //        initialization
    class OOBGroupComm {
      int rank_;
      int world_sz_;
      static bootstrap_handle_t boot_handle_;
      // boot_handle for allgather
    public:
      // @brief Constructor
      // @param r Rank of the process
      // @param ws World size
      // @param bh Ptr to p2p boot handle where custom all-gather present.
      OOBGroupComm(int r, int ws, bootstrap_handle_t *bh);

      // @brief Out-of-band AllGather collectives
      // @param sbuf Starting address of the send buffer
      // @param rbuf Starting address of the receive buffer
      // @param msglen Length of data received from any process
      // @param coll_info Ptr to group communicator used for the out-of-band
      //                  communication.
      // @return UCC_OK if success, UCC_ERR_LAST otherwise.
      static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                        void *coll_info, void **req);

      // @brief Out-of-band allGather test.
      //        This is  currently noops(or always succeed).  Since the custom
      //        p2p all-gather operation  is synchronous, there is  no need to
      //        check the check/test  the status of the operation.  In case of
      //        non-blocking   all-gather()   collecitve,   ucc   context/team
      //        creation api relies on this function to test the status of oob
      //        allgather
      // @param req pointer to the ucc collective request handle
      // @return UCC_OK - always succeed.
      static ucc_status_t oob_allgather_test(void *req);

      // @brief Out-of-band allGather test.
      //        This is currently noops(or always succeed).
      // @param req pointer to the ucc collective request handle
      // @return UCC_OK - always succeed.
      static ucc_status_t oob_allgather_free(void *req);

      int get_rank();
      int get_world_size();
      void *get_coll_info();
    };
  } // namespace ucc
} // namespace Realm

#endif
