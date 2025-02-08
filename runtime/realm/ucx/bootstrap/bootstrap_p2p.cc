#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <unistd.h>

#include "bootstrap.h"
#include "bootstrap_util.h"

#include "p2p_comm.h"
#include "worker.h"

static std::shared_ptr<p2p::P2PComm> p2p_comm{nullptr};

static int bootstrap_p2p_allgather(const void *sendbuf, void *recvbuf, int length,
                                   struct bootstrap_handle *handle)
{
  int status = 0;
  status =
      p2p_comm->Allgather(const_cast<void *>(sendbuf), length, 0, recvbuf, length, 0);
  BOOTSTRAP_NE_ERROR_JMP(status, 0, BOOTSTRAP_ERROR_INTERNAL, out,
                         "MPI_Allgather failed\n");
out:
  return status;
}

// Read the environment variables.
// WORKER_SELF_INFO -  The information about self id - <IP:PORT>
// WORKER_PEERS_INFO - List of information about all the workers in the system
//                     The information is delimited by ' ' (space character).
bool read_env(std::string &sstr, std::vector<std::string> &psstr)
{
  const char *peers_info = std::getenv("WORKER_PEERS_INFO");
  const char *self_info = std::getenv("WORKER_SELF_INFO");

  if(!peers_info || !self_info) {
    std::cerr << "The environment variable 'WORKER_SELF_INFO' and/or "
              << "'WORKER_PEERS_INFO' is not set." << std::endl;
    return false;
  }

  auto trimmer = [](const std::string &str) {
    // Find the first non-whitespace character from the start
    auto start = std::find_if_not(str.begin(), str.end(), ::isspace);

    // Find the first non-whitespace character from the end
    auto end = std::find_if_not(str.rbegin(), str.rend(), ::isspace).base();
    // Return the substring from start to end
    return (start < end) ? std::string(start, end) : "";
  };

  std::string env_self_id(self_info);
  std::string self_id = trimmer(env_self_id);

  sstr = self_id;

  auto split_in_words = [](const std::string &sentence) {
    std::vector<std::string> words;
    std::stringstream ss(sentence);
    std::string word;

    // Extract words from the stringstream
    while(ss >> word) {
      words.push_back(word);
    }

    return words;
  };

  const std::string env_peers_ids(peers_info);
  // std::vector<std::string> peers_list = split_in_words(env_peers_ids);
  psstr = split_in_words(env_peers_ids);
  //  psstr = peers_list;

  return true;
}

extern "C" int realm_ucp_bootstrap_plugin_init(void *arg, bootstrap_handle_t *handle)
{
  std::string self;
  std::vector<std::string> peers;
  if(!read_env(self, peers)) {
    std::cerr << "Failed to gather workers information " << std::endl;
    return -1;
  }

  p2p_comm = std::make_shared<p2p::P2PComm>(self, peers);
  if(!p2p_comm->Init()) {
    std::cerr << "Failed to initialize p2p comm" << std::endl;
    return -1;
  }

  handle->pg_rank = p2p_comm->get_rank();
  handle->pg_size = p2p_comm->get_world_size();

  handle->shared_ranks = nullptr;
  handle->barrier = nullptr;
  handle->bcast = nullptr;
  handle->gather = nullptr;
  handle->allgather = bootstrap_p2p_allgather;
  handle->alltoall = nullptr;
  handle->allreduce_ull = nullptr;
  handle->allgatherv = nullptr;
  handle->finalize = nullptr;
  return 0;
}
