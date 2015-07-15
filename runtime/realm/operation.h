
#ifndef REALM_OPERATION_H
#define REALM_OPERATION_H

#include "lowlevel.h"
#include "realm/profiling.h"

namespace Realm {

  class Operation {
  public:
    Operation(void);
    Operation(const ProfilingRequestSet &reqs);
    virtual ~Operation(void);
  public:
    virtual void mark_ready(void);
    virtual void mark_started(void);
    virtual void mark_completed(void);
  public:
    inline bool perform_capture(void) const { return capture_timeline; }
  protected:
    void clear_profiling(void);
    void reconstruct_measurements();
    ProfilingMeasurements::OperationStatus status;
    ProfilingMeasurements::OperationTimeline timeline;
    ProfilingRequestSet requests; 
    ProfilingMeasurementCollection measurements;
    bool capture_timeline;
  };

};

#endif // REALM_OPERATION_H
