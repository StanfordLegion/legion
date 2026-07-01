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

#ifndef __LEGION_FUTURE_MAP_H__
#define __LEGION_FUTURE_MAP_H__

#include "legion/api/future.h"

namespace Legion {

  /**
   * \class FutureMap
   * Future maps are the values returned from asynchronous index space
   * task launches.  Future maps store futures for each of the points
   * in the index space launch.  The application can either wait for
   * a point or choose to extract a future for the given point which
   * will be filled in when the task for that point completes.
   *
   * Future maps are handles that can be passes by value or stored in
   * data structures.  However, future maps should not escape the
   * context in which they are created as the runtime garbage collects
   * them after the enclosing task context completes execution.
   */
  class FutureMap : public Unserializable {
  public:
    FutureMap(void);
    FutureMap(const FutureMap& map);
    FutureMap(FutureMap&& map) noexcept;
    ~FutureMap(void);
  private:
    Internal::FutureMapImpl* impl;
  protected:
    // Only the runtime should be allowed to make these
    FRIEND_ALL_RUNTIME_CLASSES
    explicit FutureMap(Internal::FutureMapImpl* impl);
  public:
    inline bool exists(void) const { return (impl != nullptr); }
    inline bool operator==(const FutureMap& f) const { return impl == f.impl; }
    inline bool operator<(const FutureMap& f) const { return impl < f.impl; }
    inline Future operator[](const DomainPoint& point) const
    {
      return get_future(point);
    }
    FutureMap& operator=(const FutureMap& f);
    FutureMap& operator=(FutureMap&& f) noexcept;
    std::size_t hash(void) const;
  public:
    /**
     * Block until we can return the result for the
     * task executing for the given domain point.
     * @param point the point task to wait for
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with any warnings
     * @return the return value of the task
     */
    template<typename T>
    inline T get_result(
        const DomainPoint& point, bool silence_warnings = false,
        const char* warning_string = nullptr) const;
    /**
     * Non-blocking call that will return a future that
     * will contain the value from the given index task
     * point when it completes.
     * @param point the point task to wait for
     * @return a future for the index task point
     */
    Future get_future(const DomainPoint& point) const;
    /**
     * Blocking call that will return one the point
     * in the index space task has executed.
     * @param point the point task to wait for
     * @param silience_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with any warnings
     */
    void get_void_result(
        const DomainPoint& point, bool silence_warnings = false,
        const char* warning_string = nullptr) const;
  public:
    /**
     * An older method for getting the result of
     * a point in an index space launch that is
     * maintained for backwards compatibility.
     * @param point the index task point to get the return value from
     * @return the return value of the index task point
     */
    template<typename RT, typename PT, unsigned DIM>
    inline RT get_result(const PT point[DIM]) const;
    /**
     * An older method for getting a future corresponding
     * to a point in an index task launch.  This call is
     * non-blocking and actually waiting for the task to
     * complete will necessitate waiting on the future.
     * @param point the index task point to get the future for
     * @return a future for the point in the index task launch
     */
    template<typename PT, unsigned DIM>
    inline Future get_future(const PT point[DIM]) const;
    /**
     * An older method for performing a blocking wait
     * for a point in an index task launch.
     * @param point the point in the index task launch to wait for
     */
    template<typename PT, unsigned DIM>
    inline void get_void_result(const PT point[DIM]) const;
  public:
    /**
     * Wait for all the tasks in the index space launch of
     * tasks to complete before returning.
     * @param silence_warnings silience warnings for this blocking call
     * @param warning_string a string to be reported with any warnings
     */
    void wait_all_results(
        bool silence_warnings = false,
        const char* warning_string = nullptr) const;
  public:
    /**
     * This method will return the domain of points that can be
     * used to index into this future map.
     * @return domain of all points in the future map
     */
    Domain get_future_map_domain(void) const;
  };

}  // namespace Legion

#include "legion/api/future_map.inl"

#endif  // __LEGION_FUTURE_MAP_H__
