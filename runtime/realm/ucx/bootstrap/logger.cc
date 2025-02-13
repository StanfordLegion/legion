#include <iostream>
#include <fstream>

#include "logger.h"
#include "types.h"

namespace p2p {
  std::shared_ptr<Logger::p2p_log> Logger::instance_;

  std::shared_ptr<Logger::p2p_log> Logger::getInstance(int rank)
  {
    if(instance_ == nullptr) {
      // Read from env varaible; if env is not set use default value.
      std::string logfile_name = mesh::DEF_LOG + std::string(".") + std::to_string(rank);
      instance_ = std::make_shared<Logger::p2p_log>(std::ofstream(logfile_name));
    }
    return instance_;
  }
} // namespace p2p
