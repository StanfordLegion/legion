#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#include "oob_group_comm.h"
#include "bootstrap/bootstrap.h"
#include "realm_defines.h"

namespace Realm {
  namespace ucc {
    bootstrap_handle OOBGroupComm::boot_handle_;

    OOBGroupComm::OOBGroupComm(int r, int ws, bootstrap_handle_t *bh)
      : rank_(r)
      , world_sz_(ws)
    {
      boot_handle_ = *bh;
    }

    ucc_status_t OOBGroupComm::oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                             void *coll_info, void **req)
    {

      ucc_status_t status{UCC_OK};
      int ret = boot_handle_.allgather(sbuf, rbuf, msglen, &boot_handle_);
      if(0 != ret) {
        std::cerr << "OOB-Allgather() error in allgather" << std::endl;
        status = UCC_ERR_LAST;
      }
      return status;
    }

    ucc_status_t OOBGroupComm::oob_allgather_test(void *req) { return UCC_OK; }

    ucc_status_t OOBGroupComm::oob_allgather_free(void *req)
    {
      return UCC_OK;
      ;
    }

    int OOBGroupComm::get_rank() { return rank_; }

    int OOBGroupComm::get_world_size() { return world_sz_; }

    void *OOBGroupComm::get_coll_info() { return this; }
  } // namespace ucc
} // namespace Realm
