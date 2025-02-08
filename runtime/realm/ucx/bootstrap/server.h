#ifndef __SERVER_H__
#define __SERVER_H__

#include <string>
#include <unordered_map>
#include <vector>

#include <sys/socket.h>

#include "logger.h"
#include "types.h"

namespace mesh {
  // @class Server
  // @brief Server in client-server architecture.
  class Server {
    mesh::NodeIdent self_;
    int grp_sz_; // size of the group
    int ai_family_{AF_INET};

    std::vector<int> socks_;
    // Map of receiving sockets
    std::unordered_map<std::string, int> recv_socks_;
    std::shared_ptr<p2p::Logger::p2p_log> p2p_log_{nullptr};

    void handle_client_(int client_sock);
    void set_sock_addr_(const char *address_str, struct sockaddr_in *saddr);
    int listen_();

  public:
    Server(mesh::NodeIdent, int group_sz);

    /**
     * @brief Start server on well know port.
     *        Accepts connection from all the clients.
     *        Store socket ids in recv_socks_ map.
     * @return Return 0 on success, -1 otherwise.
     */
    int start();

    /**
     * @brief Receives data from the source.
     *
     * @param src Source worker to receive data from.
     * @param buf Pointer to buffer to receive at.
     * @param len Size of the buffer.
     * @return Return 0 if success, -1 if fails.
     */
    int recv_buf(const std::string &src, void *buf, size_t len);

    /**
     * @brief close each connection connected to client on all the workers.
     * @return Return 0 on success, -1 otherwise
     */
    int shutdown();
  };
} // namespace mesh

#endif
