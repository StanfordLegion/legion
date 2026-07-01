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

#ifndef __LEGION_ARGUMENT_MAP_H__
#define __LEGION_ARGUMENT_MAP_H__

#include "legion/api/future_map.h"

namespace Legion {

  //==========================================================================
  //                    Pass-By-Value Argument Classes
  //==========================================================================

  /**
   * \class UntypedBuffer
   * A class for describing an untyped buffer value.  Note that untyped
   * buffers do not make copies of the data they point to.  Copies
   * are only made upon calls to the runtime to avoid double copying.
   * It is up to the user to make sure that the the memory described by
   * an untyped buffer is live throughout the duration of its lifetime.
   */
  class UntypedBuffer : public Unserializable {
  public:
    UntypedBuffer(void) : args(nullptr), arglen(0) { }
    UntypedBuffer(const void* arg, size_t argsize)
      : args(const_cast<void*>(arg)), arglen(argsize)
    { }
    UntypedBuffer(const UntypedBuffer& rhs) : args(rhs.args), arglen(rhs.arglen)
    { }
    UntypedBuffer(UntypedBuffer&& rhs) noexcept
      : args(rhs.args), arglen(rhs.arglen)
    { }
  public:
    inline size_t get_size(void) const { return arglen; }
    inline void* get_ptr(void) const { return args; }
  public:
    inline bool operator==(const UntypedBuffer& arg) const
    {
      return (args == arg.args) && (arglen == arg.arglen);
    }
    inline bool operator<(const UntypedBuffer& arg) const
    {
      return (args < arg.args) && (arglen < arg.arglen);
    }
    inline UntypedBuffer& operator=(const UntypedBuffer& rhs)
    {
      args = rhs.args;
      arglen = rhs.arglen;
      return *this;
    }
    inline UntypedBuffer& operator=(UntypedBuffer&& rhs) noexcept
    {
      args = rhs.args;
      arglen = rhs.arglen;
      return *this;
    }
  private:
    void* args;
    size_t arglen;
  };
  // This typedef is here for backwards compatibility since we
  // used to call an UntypedBuffer a TaskArgument
  typedef UntypedBuffer TaskArgument;

  /**
   * \class ArgumentMap
   * Argument maps provide a data structure for storing the task
   * arguments that are to be associated with different points in
   * an index space launch.  Argument maps are light-weight handle
   * to the actual implementation that uses a versioning system
   * to make it efficient to re-use argument maps over many task
   * calls, especially if there are very few changes applied to
   * the map between task call launches.
   */
  class ArgumentMap : public Unserializable {
  public:
    ArgumentMap(void);
    ArgumentMap(const FutureMap& rhs);
    ArgumentMap(const ArgumentMap& rhs);
    ArgumentMap(ArgumentMap&& rhs) noexcept;
    ~ArgumentMap(void);
  public:
    ArgumentMap& operator=(const FutureMap& rhs);
    ArgumentMap& operator=(const ArgumentMap& rhs);
    ArgumentMap& operator=(ArgumentMap&& rhs) noexcept;
    inline bool operator==(const ArgumentMap& rhs) const
    {
      return (impl == rhs.impl);
    }
    inline bool operator<(const ArgumentMap& rhs) const
    {
      return (impl < rhs.impl);
    }
    inline bool exists(void) const { return (impl != nullptr); }
  public:
    /**
     * Check to see if a point has an argument set
     * @param point the point to check
     * @return true if the point has a value already set
     */
    bool has_point(const DomainPoint& point);
    /**
     * Associate an argument with a domain point
     * @param point the point to associate with the untyped buffer
     * @param arg the untyped buffer
     * @param replace specify whether to overwrite an existing value
     */
    void set_point(
        const DomainPoint& point, const UntypedBuffer& arg,
        bool replace = true);
    /**
     * Associate a future with a domain point
     * @param point the point to associate with the untyped buffer
     * @param future the future argument
     * @param replace specify whether to overwrite an existing value
     */
    void set_point(
        const DomainPoint& point, const Future& f, bool replace = true);
    /**
     * Remove a point from the argument map
     * @param point the point to be removed
     * @return true if the point was removed
     */
    bool remove_point(const DomainPoint& point);
    /**
     * Get the untyped buffer for a point if it exists, otherwise
     * return an empty untyped buffer.
     * @param point the point to retrieve
     * @return a untyped buffer if the point exists otherwise
     *    an empty untyped buffer
     */
    UntypedBuffer get_point(const DomainPoint& point) const;
  public:
    /**
     * An older method for setting the point argument in
     * an argument map.
     * @param point the point to associate the untyped buffer
     * @param arg the argument
     * @param replace specify if the value should overwrite
     *    the existing value if it already exists
     */
    template<typename PT, unsigned DIM>
    inline void set_point_arg(
        const PT point[DIM], const UntypedBuffer& arg, bool replace = false);
    /**
     * An older method for removing a point argument from
     * an argument map.
     * @param point the point to remove from the map
     */
    template<typename PT, unsigned DIM>
    inline bool remove_point(const PT point[DIM]);
  private:
    FRIEND_ALL_RUNTIME_CLASSES
    Internal::ArgumentMapImpl* impl;
  private:
    explicit ArgumentMap(Internal::ArgumentMapImpl* i);
  };

}  // namespace Legion

#include "legion/api/argument_map.inl"

#endif  // __LEGION_ARGUMENT_MAP_H__
