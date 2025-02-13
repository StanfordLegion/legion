#include <algorithm>
#include <cstdint>
#include <unistd.h>
#include <cstring>
#include <iostream>

#include "logger.h"
#include "p2p_comm.h"

namespace p2p {
  P2PComm::P2PComm(const std::string &self, const std::vector<std::string> &peers,
                   const std::string &log_file)
    : self_(self)
    , peers_(peers)
  {
    std::sort(peers_.begin(), peers_.end());

    rank_ =
        std::distance(peers_.begin(),
                      std::find_if(peers_.begin(), peers_.end(),
                                   [id = self_](const auto &val) { return id == val; }));
    world_sz_ = peers_.size();

    p2p_log_ = p2p::Logger::getInstance(rank_);

    *p2p_log_ << "My Rank = " << rank_ << ", Team Size = " << world_sz_ << ".\n"
              << std::endl;
  }

  bool P2PComm::Init()
  {
    worker_ = std::make_unique<mesh::Worker>(self_, peers_);
    if(worker_->init()) {
      *p2p_log_ << "Failed to construct full mesh topology" << std::endl;
      return false;
    }

    return true;
  }

  int P2PComm::Allgather(void *sbuf, int sendcount, uint8_t sendtype, void *rbuf,
                         int recvcount, uint8_t recvtype)
  {
    for(const auto &dest : peers_) {
      if(worker_->send_buf(dest, const_cast<void *>(sbuf), sendcount)) {
        *p2p_log_ << "P2PComm-Allgather() : Send failed" << std::endl;
        return -1;
      } else {
        *p2p_log_ << "P2PComm-Allgather() : Sent " << sendcount << " bytes to " << dest
                  << "  succeeded" << std::endl;
      }
    }

    int ind{0};
    for(const auto &src : peers_) {
      uint8_t buf[recvcount];
      size_t recv_len;

      *p2p_log_ << "P2PComm-Allgather() : receiving " << recvcount << " bytes from "
                << src << std::endl;
      if(worker_->recv_buf(src, static_cast<void *>(buf), recvcount)) {
        *p2p_log_ << "P2PComm-Allgather() : Receive Failed" << std::endl;
        return -1;
      }

      std::memcpy(static_cast<uint8_t *>(rbuf) + ind++ * recvcount,
                  static_cast<uint8_t *>(buf), recvcount);
    }
    return 0;
  }
} // namespace p2p
