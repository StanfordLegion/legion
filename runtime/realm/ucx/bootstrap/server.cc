#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "server.h"
#include "logger.h"
#include "types.h"

namespace mesh {
  Server::Server(mesh::NodeIdent self, int group_sz)
    : self_(self)
    , grp_sz_(group_sz)
  {
    p2p_log_ = p2p::Logger::getInstance();
  }

  /**
   * Set an address for the server to listen on - INADDR_ANY on a well known port.
   */
  void Server::set_sock_addr_(const char *address_str, struct sockaddr_in *sa_in)
  {
    switch(ai_family_) {
    case AF_INET:
      if(address_str != NULL) {
        if(inet_pton(AF_INET, address_str, &sa_in->sin_addr) <= 0) {
          perror("Failed to set input ip adress as listener");
        }
      } else {
        sa_in->sin_addr.s_addr = INADDR_ANY;
      }
      sa_in->sin_family = AF_INET;
      sa_in->sin_port = htons(self_.port);
      break;
    case AF_INET6:
      assert(false);
      break;
    default:
      std::cerr << "Invalid address family" << std::endl;
      break;
    }
  }

  int Server::listen_()
  {
    int server_socket = socket(ai_family_, SOCK_STREAM, 0);
    sockaddr_in serverAddr;
    set_sock_addr_(self_.ipaddress.data(), &serverAddr);

    int res = bind(server_socket, (struct sockaddr *)&serverAddr, sizeof(serverAddr));
    if(res) {
      *p2p_log_ << "Failed to bind " << res << std::endl;
      close(server_socket);
      return res;
    }
    int ret = listen(server_socket, 5);
    if(ret < 0) {
      close(server_socket);
      *p2p_log_ << "Failed to listen " << errno << std::endl;
      return ret;
    }

    *p2p_log_ << "Server listening on " << self_.ipaddress << ":" << self_.port
              << std::endl;

    while(socks_.size() != grp_sz_) {
      int client_socket = accept(server_socket, nullptr, nullptr);
      *p2p_log_ << "New client connected!" << std::endl;
      socks_.push_back(client_socket);
    }

    // No more listening, all the connections are established.
    close(server_socket);
    return 0;
  }

  int Server::start()
  {
    if(listen_()) {
      *p2p_log_ << "Unable to start the server" << std::endl;
      return -1;
    }
    // handshake - necessary to understand the host id of the other end of
    // every sockets.

    // TODO : Receive correct amount of bytes in multiple round-trips
    for(const auto &s : socks_) {
      std::string id(BUFSIZE, '\0');

      int exp_bytes{BUFSIZE};
      int sz = recv(s, static_cast<void *>(id.data()), BUFSIZE, 0);
      while(sz != exp_bytes) {
        if(sz < 0) {
          *p2p_log_ << "Failed to receive from socket " << s << std::endl;
          return -1;
        }
        exp_bytes -= sz;
        sz = recv(s, static_cast<char *>(id.data()) + BUFSIZE - exp_bytes, exp_bytes, 0);
      }

      // TODO : Remove trailing null characters
      auto remove_trailing_nulls = [](std::string &str) {
        // Erase trailing null characters
        size_t pos = str.find_last_not_of('\0');
        if(pos != std::string::npos) {
          // Erase everything after the last non-null character
          str.erase(pos + 1);
        } else { // If the string consists only of nulls, clear it.
          str.clear();
        }
      };

      remove_trailing_nulls(id);
      *p2p_log_ << "Exptected identity is received - " << id << "(" << id.size() << ")"
                << std::endl;
      recv_socks_[id] = s;
    }

    return 0;
  };

  int Server::recv_buf(const std::string &src, void *dst, size_t sz)
  {
    int exp_bytes = sz;

    int sock = recv_socks_[src];
    int cnt = recv(sock, dst, exp_bytes, 0);
    while(cnt != exp_bytes) {
      if(cnt < 0) {
        *p2p_log_ << "Failed to receive from socket " << sock << std::endl;
        return -1;
      }
      exp_bytes -= cnt;
      cnt = recv(sock, static_cast<char *>(dst) + sz - exp_bytes, exp_bytes, 0);
    }
    return 0;
  }

  int Server::shutdown()
  {
    int ret{0};
    for(const auto &c : recv_socks_) {
      if((ret = close(c.second)) < 0)
        return ret;
    }

    return ret;
  }
} // namespace mesh
