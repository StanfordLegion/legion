#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <iostream>
#include <memory>
#include <fstream>

#define LOG_TAG() (std::string(__func__ + ":" + __LINE__ + " "))

namespace p2p {
  // @class Logger
  // @brief Singleton class for the logging per rank.
  //
  // The class is wrapper around the  std::ofstream. It logs the messages from
  // worker, client  and server  threads at  pre-configured tmp  location.  If
  // there are more than  one rank per node, it is ensured  that each rank has
  // its  own log  file.  This  avoid polluting  to  standard output/err  with
  // messeges otherwise needed for debugging.
  //
  // The other option to use is to  use the Realm's logger if it is reasonable
  // to instantiate the Realm's logger inside p2p bootstrap library.
  class Logger {
  public:
    using p2p_log = std::ofstream;

  private:
    static std::shared_ptr<p2p_log> instance_;
    // static p2p_log* instance;
    Logger() {} // Private constructor to prevent direct instantiation

  public:
    //		static p2p_log* getInstance(int rank=0);
    static std::shared_ptr<p2p_log> getInstance(int rank = 0);
  };
} // namespace p2p

#endif
