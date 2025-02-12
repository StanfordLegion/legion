#ifndef __CLIENT_H__
#define __CLIENT_H__

#include <thread>
#include <unordered_map>
#include <vector>
#include <netinet/in.h>

#include "logger.h"
#include "types.h"

namespace mesh {
  // @class Client
  // @brief Client in client-server architecture.
  class Client {
    mesh::NodeIdent self;
    std::unordered_map<std::string, mesh::NodeIdent> peers;
    int ai_family{AF_INET};

    // map of sending sockets
    std::unordered_map<std::string, int> send_sockets;

    // Independent threads for each client
    std::vector<std::thread> threads;

    std::shared_ptr<p2p::Logger::p2p_log> p2p_log{nullptr};

    void set_sock_addr_(const char *address_str, int port, struct sockaddr_in *saddr);
    int connect_to_server_(const std::string &ip, int port);

  public:
    Client(const mesh::NodeIdent &self,
           const std::unordered_map<std::string, mesh::NodeIdent> &servers);

    /**
     * @brief Connects to all the servers on all the workers and saves the
     *        Socket ids in send_socks_ map.
     * @return Return 0 on success, -1 otherwise.
     */
    int start();

    /**
     * @brief sends data to the destination.
     *
     * @param dst Destination worker to send data to.
     * @param buf Pointer to buffer to send data from.
     * @param len Size of the buffer.
     * @return Return 0 on success, -1 otherwise.
     */
    int send_buf(const std::string &dst, void *buf, size_t len);

    /**
     * @brief close connections to all servers on all the workers.
     * @return Return 0 on success, -1 otherwise
     */
    int shutdown();
  };

} // namespace mesh

#endif
