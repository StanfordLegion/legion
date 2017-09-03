#ifndef CGSOLVER_H
#define CGSOLVER_H

#include <legion.h>

using namespace Legion;

template<typename FT, int N, typename T = coord_t>
using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t>
using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t>
using AccessorWD = FieldAccessor<WRITE_DISCARD,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t>
using AccessorRD = FieldAccessor<REDUCE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

struct GhostInfo {
  GhostInfo(void);

  enum GhostType {
    GHOST_UNKNOWN,
    GHOST_BOUNDARY,
    GHOST_LOCAL,
    GHOST_REMOTE
  };
  GhostType gtype;
  IndexSpace ispace;
  LogicalRegion lr_parent;   // parent shared region
  LogicalRegion lr_shared;   // actual subset of shared region we use
  LogicalRegion lr_ghost;
  PhaseBarrier pb_shared_ready;  // used to receive readiness of shared region
  PhaseBarrier pb_shared_done;   // used to signals that reading is done
};

inline GhostInfo::GhostInfo(void)
  : gtype(GHOST_UNKNOWN), ispace(IndexSpace::NO_SPACE), lr_shared(LogicalRegion::NO_REGION), lr_ghost(LogicalRegion::NO_REGION)
{}

struct BlockMetadata {
  Rect<3> bounds;  // subgrid bounds
  IndexSpace ispace;
  LogicalRegion lr_private;  // logical region containing private fields
  LogicalRegion lr_shared; // logical region shared with other shards
  int neighbors;
  PhaseBarrier pb_shared_ready;  // used to signal readiness of shared region
  PhaseBarrier pb_shared_done;   // used to receive signals that readers are done with previous version
  GhostInfo ghosts[3][2];
};

typedef std::map<Point<3>, BlockMetadata> MyBlockMap;

#endif
