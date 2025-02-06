#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <future>
#include <thread>

#include <arpa/inet.h>

#include "client.h"
#include "logger.h"
#include "server.h"
#include "types.h"
#include "worker.h"

namespace mesh {
  Worker::Worker(const std::string &self, const std::vector<std::string> &peers)
  {
    auto lamda = [](const std::string &str) {
      const auto &tmp_str = std::string(str);
      const auto &ind = tmp_str.find(':');
      const auto &ip = tmp_str.substr(0, ind);
      const auto &portstr = tmp_str.substr(ind + 1);
      const auto &port = static_cast<unsigned short>(std::stoul(portstr));
      return mesh::NodeIdent{tmp_str, ip, port};
    };

    p2p_log_ = p2p::Logger::getInstance();

    self_ = lamda(self);
    std::for_each(peers.begin(), peers.end(),
                  [this, lamda](const auto &p) { peers_[p] = lamda(p); });
  }

  int Worker::init()
  {
    server_ = std::make_unique<mesh::Server>(self_, peers_.size());
    client_ = std::make_unique<mesh::Client>(self_, peers_);

    // Start client and server asynchronously.
    auto fs = std::async(std::launch::async, &Server::start, std::ref(*server_));

    auto fc = std::async(std::launch::async, &Client::start, std::ref(*client_));

    if(fs.get() || fc.get()) {
      *p2p_log_ << "Failed to start the server" << std::endl;
      return -1;
    } else {
      *p2p_log_ << "\nAll peer-to-peer connections are established." << std::endl;
    }

    return 0;
  }

  int Worker::shutdown()
  {
    // Shutdown server and client
    auto ret_client = client_->shutdown();
    auto ret_server = server_->shutdown();
    if(ret_server != 0 || ret_client != 0) {
      *p2p_log_ << "Server stop status = " << ret_server
                << "Client stop status = " << ret_client << std::endl;
      return false;
    }
    return true;
  }

  int Worker::send_buf(const std::string &dst, void *buf, size_t len)
  {
    return client_->send_buf(dst, buf, len);
  }

  int Worker::recv_buf(const std::string &src, void *buf, size_t len)
  {
    return server_->recv_buf(src, buf, len);
  }
}; // namespace mesh
