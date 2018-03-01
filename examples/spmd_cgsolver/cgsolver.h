#ifndef CGSOLVER_H
#define CGSOLVER_H

#include <legion.h>

using namespace Legion;

typedef FieldAccessor<READ_ONLY,int,3,coord_t,Realm::AffineAccessor<int,3,coord_t> > AccessorROint;
typedef FieldAccessor<READ_WRITE,int,3,coord_t,Realm::AffineAccessor<int,3,coord_t> > AccessorRWint;
typedef FieldAccessor<WRITE_DISCARD,int,3,coord_t,Realm::AffineAccessor<int,3,coord_t> > AccessorWDint;

typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t> > AccessorROdouble;
typedef FieldAccessor<READ_WRITE,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t> > AccessorRWdouble;
typedef FieldAccessor<WRITE_DISCARD,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t> > AccessorWDdouble;

typedef FieldAccessor<READ_ONLY,PhaseBarrier,3,coord_t,Realm::AffineAccessor<PhaseBarrier,3,coord_t> > AccessorROpb;
typedef FieldAccessor<READ_WRITE,PhaseBarrier,3,coord_t,Realm::AffineAccessor<PhaseBarrier,3,coord_t> > AccessorRWpb;
typedef FieldAccessor<WRITE_DISCARD,PhaseBarrier,3,coord_t,Realm::AffineAccessor<PhaseBarrier,3,coord_t> > AccessorWDpb;

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
