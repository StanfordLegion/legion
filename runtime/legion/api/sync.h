/* Copyright 2026 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LEGION_SYNCHRONIZATION_H__
#define __LEGION_SYNCHRONIZATION_H__

#include "legion/api/types.h"

namespace Legion {

  /**
   * \class Lock
   * NOTE THIS IS NOT A NORMAL LOCK!
   * A lock is an atomicity mechanism for use with regions acquired
   * with simultaneous coherence in a deferred execution model.
   * Locks are light-weight handles that are created in a parent
   * task and can be passed to child tasks to guaranteeing atomic access
   * to a region in simultaneous mode.  Lock can be used to request
   * access in either exclusive (mode 0) or non-exclusive mode (any number
   * other than zero).  Non-exclusive modes are mutually-exclusive from each
   * other. While locks can be passed down the task tree, they should
   * never escape the context in which they are created as they will be
   * garbage collected when the task in which they were created is complete.
   *
   * There are two ways to use locks.  The first is to use the blocking
   * acquire and release methods on the lock directly.  Acquire
   * guarantees that the application will hold the lock when it
   * returns, but may result in stalls while some other task is holding the
   * lock.  The recommended way of using locks is to request
   * grants of a lock through the runtime interface and then pass
   * grants to launcher objects.  This ensures that the lock will be
   * held during the course of the operation while still avoiding
   * any stalls in the task's execution.
   * @see TaskLauncher
   * @see IndexTaskLauncher
   * @see CopyLauncher
   * @see InlineLauncher
   * @see Runtime
   */
  class Lock {
  public:
    Lock(void);
  protected:
    // Only the runtime is allowed to make non-empty locks
    FRIEND_ALL_RUNTIME_CLASSES
    Lock(Reservation r);
  public:
    bool operator<(const Lock& rhs) const;
    bool operator==(const Lock& rhs) const;
  public:
    void acquire(unsigned mode = 0, bool exclusive = true);
    void release(void);
  private:
    Reservation reservation_lock;
  };

  /**
   * \struct LockRequest
   * This is a helper class for requesting grants.  It
   * specifies the locks that are needed, what mode they
   * should be acquired in, and whether or not they
   * should be acquired in exclusive mode or not.
   */
  struct LockRequest {
  public:
    LockRequest(Lock l, unsigned mode = 0, bool exclusive = true);
  public:
    Lock lock;
    unsigned mode;
    bool exclusive;
  };

  /**
   * \class Grant
   * Grants are ways of naming deferred acquisitions and releases
   * of locks.  This allows the application to defer a lock
   * acquire but still be able to use it to specify which tasks
   * must run while holding the this particular grant of the lock.
   * Grants are created through the runtime call 'acquire_grant'.
   * Once a grant has been used for all necessary tasks, the
   * application can defer a grant release using the runtime
   * call 'release_grant'.
   * @see Runtime
   */
  class Grant {
  public:
    Grant(void);
    Grant(const Grant& g);
    ~Grant(void);
  protected:
    // Only the runtime is allowed to make non-empty grants
    FRIEND_ALL_RUNTIME_CLASSES
    explicit Grant(Internal::GrantImpl* impl);
  public:
    bool operator==(const Grant& g) const { return impl == g.impl; }
    bool operator<(const Grant& g) const { return impl < g.impl; }
    Grant& operator=(const Grant& g);
  protected:
    Internal::GrantImpl* impl;
  };

  /**
   * \class PhaseBarrier
   * Phase barriers are a synchronization mechanism for use with
   * regions acquired with simultaneous coherence in a deferred
   * execution model.  Phase barriers allow the application to
   * guarantee that a collection of tasks are all executing their
   * sub-tasks all within the same phase of computation.  Phase
   * barriers are light-weight handles that can be passed by value
   * or stored in data structures.  Phase barriers are made in
   * a parent task and can be passed down to any sub-tasks.  However,
   * phase barriers should not escape the context in which they
   * were created as the task that created them will garbage collect
   * their resources.
   *
   * Note that there are two ways to use phase barriers.  The first
   * is to use the blocking operations to wait for a phase to begin
   * and to indicate that the task has arrived at the current phase.
   * These operations may stall and block current task execution.
   * The preferred method for using phase barriers is to pass them
   * in as wait and arrive barriers for launcher objects which will
   * perform the necessary operations on barriers before an after
   * the operation is executed.
   * @see TaskLauncher
   * @see IndexTaskLauncher
   * @see CopyLauncher
   * @see InlineLauncher
   * @see Runtime
   */
  class PhaseBarrier {
  public:
    PhaseBarrier(void);
  protected:
    // Only the runtime is allowed to make non-empty phase barriers
    FRIEND_ALL_RUNTIME_CLASSES
    PhaseBarrier(Internal::ApBarrier b);
  public:
    bool operator<(const PhaseBarrier& rhs) const;
    bool operator==(const PhaseBarrier& rhs) const;
    bool operator!=(const PhaseBarrier& rhs) const;
  public:
    void arrive(unsigned count = 1);
    void wait(void);
    void alter_arrival_count(int delta);
    Realm::Barrier get_barrier(void) const { return phase_barrier; }
    bool exists(void) const;
  protected:
    Internal::ApBarrier phase_barrier;
    friend std::ostream& operator<<(std::ostream& os, const PhaseBarrier& pb);
  };

  /**
   * \class Collective
   * A DynamicCollective object is a special kind of PhaseBarrier
   * that is created with an associated reduction operation.
   * Arrivals on a dynamic collective can contribute a value to
   * each generation of the collective, either in the form of a
   * value or in the form of a future. The reduction operation is used
   * to reduce all the contributed values (which all must be of the same
   * type) to a common value. This value is returned in the form of
   * a future which applications can use as a normal future. Note
   * that unlike MPI collectives, collectives in Legion can
   * have different sets of producers and consumers and not
   * all producers need to contribute a value.
   */
  class DynamicCollective : public PhaseBarrier {
  public:
    DynamicCollective(void);
  protected:
    // Only the runtime is allowed to make non-empty dynamic collectives
    FRIEND_ALL_RUNTIME_CLASSES
    DynamicCollective(Internal::ApBarrier b, ReductionOpID redop);
  public:
    // All the same operations as a phase barrier
    void arrive(const void* value, size_t size, unsigned count = 1);
  protected:
    ReductionOpID redop;
  };

}  // namespace Legion

#include "legion/api/sync.inl"

#endif  // __LEGION_SYNCHRONIZATION_H__
