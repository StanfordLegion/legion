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

#ifndef __LEGION_INTEROP_H__
#define __LEGION_INTEROP_H__

#include "legion/api/types.h"

namespace Legion {

  /**
   * \class LegionHandshake
   * This class provides a light-weight synchronization primitive for
   * external applications to use when performing synchronization with
   * Legion tasks. It allows for control to be passed from the external
   * application into Legion and then for control to be handed back to
   * the external application from Legion. The user can configure which
   * direction occurs first when constructing the handshake object.
   * @see Runtime::create_external_handshake
   */
  class LegionHandshake : public Unserializable {
  public:
    LegionHandshake(void);
    LegionHandshake(const LegionHandshake& rhs);
    ~LegionHandshake(void);
  protected:
    Internal::LegionHandshakeImpl* impl;
  protected:
    // Only the runtime should be able to make these
    FRIEND_ALL_RUNTIME_CLASSES
    explicit LegionHandshake(Internal::LegionHandshakeImpl* impl);
  public:
    bool operator==(const LegionHandshake& h) const { return impl == h.impl; }
    bool operator<(const LegionHandshake& h) const { return impl < h.impl; }
    LegionHandshake& operator=(const LegionHandshake& rhs);
  public:
    /**
     * Non-blocking call to signal to Legion that this participant
     * is ready to pass control to Legion.
     */
    void ext_handoff_to_legion(void) const;
    /**
     * A blocking call that will cause this participant to wait
     * for all Legion participants to hand over control to the
     * external application.
     */
    void ext_wait_on_legion(void) const;
  public:
    /**
     * A non-blocking call to signal to the external application
     * that this participant is ready to pass control to to it.
     */
    void legion_handoff_to_ext(void) const;
    /**
     * A blocking call that will cause this participant to wait
     * for all external participants to hand over control to Legion.
     */
    void legion_wait_on_ext(void) const;
  public:
    /*
     * For asynchronous Legion execution, you can use these
     * methods to get a phase barrier associated with the
     * handshake object instead of blocking on the legion side
     */
    /**
     * Get the Legion phase barrier associated with waiting on the handshake
     */
    PhaseBarrier get_legion_wait_phase_barrier(void) const;
    /**
     * Get the Legion phase barrier associated with arriving on the handshake
     */
    PhaseBarrier get_legion_arrive_phase_barrier(void) const;
    /**
     * Advance the handshake associated with the Legion side
     */
    void advance_legion_handshake(void) const;
  };

  /**
   * \class MPILegionHandshake
   * This class is only here for legacy reasons. In general we encourage
   * users to use the generic LegionHandshake
   */
  class MPILegionHandshake : public LegionHandshake {
  public:
    MPILegionHandshake(void);
    MPILegionHandshake(const MPILegionHandshake& rhs);
    ~MPILegionHandshake(void);
  protected:
    // Only the runtime should be able to make these
    FRIEND_ALL_RUNTIME_CLASSES
    explicit MPILegionHandshake(Internal::LegionHandshakeImpl* impl);
  public:
    bool operator==(const MPILegionHandshake& h) const
    {
      return impl == h.impl;
    }
    bool operator<(const MPILegionHandshake& h) const { return impl < h.impl; }
    MPILegionHandshake& operator=(const MPILegionHandshake& rhs);
  public:
    /**
     * Non-blocking call to signal to Legion that this participant
     * is ready to pass control to Legion.
     */
    inline void mpi_handoff_to_legion(void) const { ext_handoff_to_legion(); }
    /**
     * A blocking call that will cause this participant to wait
     * for all Legion participants to hand over control to MPI.
     */
    inline void mpi_wait_on_legion(void) const { ext_wait_on_legion(); }
  public:
    /**
     * A non-blocking call to signal to MPI that this participant
     * is ready to pass control to MPI.
     */
    inline void legion_handoff_to_mpi(void) const { legion_handoff_to_ext(); }
    /**
     * A blocking call that will cause this participant to wait
     * for all MPI participants to hand over control to Legion.
     */
    inline void legion_wait_on_mpi(void) const { legion_wait_on_ext(); }
  };

}  // namespace Legion

#endif  // __LEGION_INTEROP_H__
