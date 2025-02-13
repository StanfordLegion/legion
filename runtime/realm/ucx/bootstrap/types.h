#ifndef __TYPES_H__
#define __TYPES_H__

#include <cstdio>
#include <string>

namespace mesh {
  const int BUFSIZE = 100;
  const std::string DEF_LOG("/tmp/p2p_bootstrap.log");
  struct NodeIdent {
    std::string id; // unique id of rank - <ip:port>
    std::string ipaddress;
    uint16_t port;
  };
} // namespace mesh

#endif
