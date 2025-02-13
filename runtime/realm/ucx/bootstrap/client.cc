#include <cassert>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <thread>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "client.h"

namespace mesh {
  Client::Client(const mesh::NodeIdent &_self,
                 const std::unordered_map<std::string, mesh::NodeIdent> &_servers)
    : self(_self)
    , peers(_servers)
  {
    p2p_log = p2p::Logger::getInstance();
  }

  /**
   * Set an address for the server to listen on - INADDR_ANY on a well known port.
   */
  void Client::set_sock_addr_(const char *address_str, int port,
                              struct sockaddr_in *sa_in)
  {
    switch(ai_family) {
    case AF_INET:
      if(address_str != NULL) {
        if(inet_pton(AF_INET, address_str, &sa_in->sin_addr) <= 0) {
          std::cerr << "Failed to set input ip adress as listener" << std::endl;
          assert(false);
        }
      } else {
        sa_in->sin_addr.s_addr = INADDR_ANY;
      }
      sa_in->sin_family = AF_INET;
      sa_in->sin_port = htons(port);
      break;
    case AF_INET6:
      assert(false);
      break;
    default:
      std::cerr << "Invalid address family" << std::endl;
      assert(false);
      break;
    }
  }

  int Client::connect_to_server_(const std::string &ip, int port)
  {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in server_addr;
    set_sock_addr_(ip.data(), port, &server_addr);

    *p2p_log << "Connecting to server : " << ip << ":" << port << std::endl;
    // retry only if the server is not listening
    while(true) {
      int ret =
          connect(client_socket, (struct sockaddr *)&server_addr, sizeof(server_addr));
      if(ret < 0) {
        // server is not yet listening, try again.
        if(errno == ECONNREFUSED) {
          *p2p_log << "server is not yet listening. "
                   << "Sleep for a second and try again." << std::endl;
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        } else {
          *p2p_log << "Connection to " << ip << ":" << port << " failed!" << std::endl;
          return ret;
        }
      }
      break;
    }

    *p2p_log << "Connected to " << ip << ":" << port << std::endl;

    send_sockets[ip + +":" + std::to_string(port)] = client_socket;
    // close(client_socket);
    return 0;
  }

  int Client::start()
  {
    *p2p_log << "Total servers to connect to = " << peers.size() << std::endl;
    for(const auto &server : peers) {
      auto ipaddr = server.second.ipaddress;
      auto port = server.second.port;
      if(connect_to_server_(ipaddr, port)) {
        *p2p_log << "Failed to connect to " << ipaddr << ":" << port << std::endl;
        return -1;
      }
    }
    // At this point connected to all the servers.
    // Send to the server the id of the host, the client is running on

    auto key = self.ipaddress + ":" + std::to_string(self.port);
    key.resize(BUFSIZE); // always send BUFSIZE bytes!
    for(const auto &peer : send_sockets) {
      size_t sz = send(peer.second, static_cast<void *>(key.data()), key.size(), 0);
      if(sz != key.size()) {
        *p2p_log << "Failed to send the identity of the host" << std::endl;
        return -1;
      }
    }
    return 0;
  }

  int Client::send_buf(const std::string &dst, void *buf, size_t len)
  {
    while(len) {
      int sz = send(send_sockets[dst], buf, len, 0);
      if(sz < 0) {
        *p2p_log << "Failed to send the buffer to destination " << dst << std::endl;
        return -1;
      }

      buf = static_cast<char *>(buf) + sz;
      len -= sz;
    }
    return 0; // Success
  }

  int Client::shutdown()
  {
    int ret{0};
    for(const auto &s : send_sockets) {
      if((ret = close(s.second)) < 0)
        return ret;
    }
    return ret;
  }
} // namespace mesh
