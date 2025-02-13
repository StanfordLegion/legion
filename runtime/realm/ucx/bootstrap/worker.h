#ifndef __WORKER_H__
#define __WORKER_H__

#include <cmath>
#include <unordered_map>
#include <queue>

#include "client.h"
#include "logger.h"
#include "server.h"
#include "types.h"

namespace mesh {
  // @class Worker
  // @brief Representation of rank/dask worker.

  // Worker class responsible for creating mesh  topology and all the p2p send
  // and receive of the information.
  class Worker {
    std::unique_ptr<mesh::Server> server_{nullptr};
    std::unique_ptr<mesh::Client> client_{nullptr};

    // Peers ip and port identity
    mesh::NodeIdent self_;
    std::unordered_map<std::string, mesh::NodeIdent> peers_;

    std::shared_ptr<p2p::Logger::p2p_log> p2p_log_{nullptr};

  public:
    Worker(const std::string &self, const std::vector<std::string> &peers);

    // @brief Establish the mesh-topology among all the member processes.
    int init();

    // @brief Shutdown the mesh topology
    int shutdown();

    int send_buf(const std::string &dst, void *buf, size_t len);

    int recv_buf(const std::string &src, void *buf, size_t len);
  };
} // namespace mesh

#endif
