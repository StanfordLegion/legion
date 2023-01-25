#ifndef MULTIAFFINE_H
#define MULTIAFFINE_H

#include "realm.h"

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Realm::Processor::TASK_ID_FIRST_AVAILABLE+0,
  PTR_WRITE_TASK_BASE,
};

enum {
  FID_BASE = 44,
  FID_ADDR,
};

template <int N, typename T>
struct PtrWriteTaskArgs {
  Realm::IndexSpace<N,T> space;
  Realm::RegionInstance inst;
};

#endif
