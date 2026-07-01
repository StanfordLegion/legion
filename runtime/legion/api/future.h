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

#ifndef __LEGION_FUTURE_H__
#define __LEGION_FUTURE_H__

#include "legion/api/physical_region.h"

namespace Legion {

  //==========================================================================
  //                          Future Value Classes
  //==========================================================================

  /**
   * \class Future
   * Futures are the objects returned from asynchronous task
   * launches.  Applications can wait on futures to get their values,
   * pass futures as arguments and preconditions to other tasks,
   * or use them to create predicates if they are boolean futures.
   * Futures are lightweight handles that can be passed by value
   * or stored in data structures.  However, futures should not
   * escape the context in which they are created as the runtime
   * garbage collects them after the enclosing task context
   * completes execution.
   *
   * Since futures can be the result of predicated tasks we also
   * provide a mechanism for checking whether the future contains
   * an empty result.  An empty future will be returned for all
   * futures which come from tasks which predicates that resolve
   * to false.
   */
  class Future : public Unserializable {
  public:
    Future(void);
    Future(const Future& f);
    Future(Future&& f) noexcept;
    ~Future(void);
  private:
    Internal::FutureImpl* impl;
  protected:
    // Only the runtime should be allowed to make these
    FRIEND_ALL_RUNTIME_CLASSES
    explicit Future(Internal::FutureImpl* impl);
  public:
    inline bool exists(void) const { return (impl != nullptr); }
    inline bool operator==(const Future& f) const { return impl == f.impl; }
    inline bool operator<(const Future& f) const { return impl < f.impl; }
    Future& operator=(const Future& f);
    Future& operator=(Future&& f) noexcept;
    std::size_t hash(void) const;
  public:
    /**
     * Wait on the result of this future.  Return
     * the value of the future as the specified
     * template type.
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with the warning
     * @return the value of the future cast as the template type
     */
    template<typename T>
    inline T get_result(
        bool silence_warnings = false,
        const char* warning_string = nullptr) const;
    /**
     * Block until the future completes.
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with the warning
     */
    void get_void_result(
        bool silence_warnings = false,
        const char* warning_string = nullptr) const;
    /**
     * Check to see if the future is empty.  The
     * user can specify whether to block and wait
     * for the future to complete first before
     * returning.  If the non-blocking version
     * of the call will return true, until
     * the future actually completes.
     * @param block indicate whether to block for the result
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with the warning
     */
    bool is_empty(
        bool block = false, bool silence_warnings = false,
        const char* warning_string = nullptr) const;
    /**
     * Check to see if the future is ready. This will return
     * true if the future can be used without blocking to wait
     * on the computation that the future represents, otherwise
     * it will return false.
     * @param subscribe ask for the payload to be brought here when ready
     */
    bool is_ready(bool subscribe = false) const;
  public:
    /**
     * Return a span object representing the data for the future.
     * The size of the future must be evenly divisible by sizeof(T).
     * The resulting span object is only good as long as the application
     * program maintains a handle to the future object that created it.
     * At the moment the privilege mode must be read-only; no other
     * values will be accepted. This call will not unpack data serialized
     * with the legion_serialize method.
     * @param memory the kind of memory for the allocation, the memory
     *    with the best affinity to the executing processor will be used
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with any warnings
     * @return a Span object representing the data for the future
     */
    template<typename T, PrivilegeMode PM = LEGION_READ_ONLY>
    Span<T, PM> get_span(
        Memory::Kind memory, bool silence_warnings = false,
        const char* warning_string = nullptr) const;

    /**
     * Return a pointer and optional size for the data for the future.
     * The pointer is only valid as long as the application program
     * maintains a handle to the future object that produced it. This call
     * will not deserialized data packed with the legion_serialize method.
     * @param memory the kind of memory for the allocation, the memory
     *    with the best affinity to the executing processor will be used
     * @param extent_in_bytes pointer to a location to write the future size
     * @param check_extent check that the extent matches the future size
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with any warnings
     * @return a const pointer to the future data in the specified memory
     */
    const void* get_buffer(
        Memory::Kind memory, size_t* extent_in_bytes = nullptr,
        bool check_extent = false, bool silence_warnings = false,
        const char* warning_string = nullptr) const;

    /**
     * Report an instantaneous set of available memories where instances
     * for the this future exist. These will only be memories local to
     * the current process in which the call is performed. The result of
     * this query might be come stale as soon as it is returned since it
     * is only a snapshot of the memories where the future has copies.
     */
    void get_memories(
        std::set<Memory>& memories, bool silence_warnings = false,
        const char* warning_string = nullptr) const;

    /**
     * Return a const reference to the future.
     * WARNING: these method is unsafe as the underlying
     * buffer containing the future result can be deleted
     * if the Future handle is lost even a reference
     * to the underlying buffer is maitained.  This
     * scenario can lead to seg-faults.  Use at your
     * own risk.  Note also that this call will not
     * properly deserialize buffers that were serialized
     * with a 'legion_serialize' method.
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with the warning
     */
    template<typename T>
    LEGION_DEPRECATED("Use 'Future::get_span' instead")
    inline const T& get_reference(
        bool silence_warnings = false,
        const char* warning_string = nullptr) const;
    /**
     * Return an untyped pointer to the
     * future result.  WARNING: this
     * method is unsafe for the same reasons
     * as get_reference.  It also will not
     * deserialize anything serialized with a
     * legion_serialize method.
     * @param silence_warnings silence any warnings for this blocking call
     * @param warning_string a string to be reported with the warning
     */
    LEGION_DEPRECATED("Use 'Future::get_buffer' instead")
    inline const void* get_untyped_pointer(
        bool silence_warnings = false,
        const char* warning_string = nullptr) const;

    /**
     * Return the number of bytes contained in the future.
     */
    size_t get_untyped_size(void) const;

    /**
     * Return a pointer to the metadata buffer for this future.
     * Unlike getting a buffer for the future which can exist on
     * any memory, the metadata is always guaranteed to be on the
     * host memory.
     * @param optional pointer to a place to write the size
     * @return a pointer to the buffer containing the metadata
     */
    const void* get_metadata(size_t* size = nullptr) const;
  public:
    // These methods provide partial support the C++ future interface
    template<typename T>
    inline T get(void);

    inline bool valid(void) const;

    inline void wait(void) const;
  public:
    /**
     * Allow users to generate their own futures. These
     * futures are guaranteed to always have completed
     * and to always have concrete values.
     */
    template<typename T>
    LEGION_DEPRECATED("Use the version without a runtime pointer argument")
    static inline Future from_value(Runtime* rt, const T& value);
    template<typename T>
    static inline Future from_value(const T& value);
    /**
     * If you are creating a future from a Domain then you need to
     * use this method to construct the future in a way to ensure
     * that the Domain maintains the right lifetime.
     */
    static Future from_domain(
        const Domain& d, bool take_ownership, const char* provenance = nullptr,
        bool shard_local = false);

    /**
     * Generates a future from an untyped pointer.  No
     * serialization is performed.
     */
    LEGION_DEPRECATED("Use the version without a runtime pointer argument")
    static Future from_untyped_pointer(
        Runtime* rt, const void* buffer, size_t bytes,
        bool take_ownership = false);
    static Future from_untyped_pointer(
        const void* buffer, size_t bytes, bool take_ownership = false,
        const char* provenance = nullptr, bool shard_local = false);
    static Future from_value(
        const void* buffer, size_t bytes, bool owned,
        const Realm::ExternalInstanceResource& resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&) = nullptr,
        const char* provenance = nullptr, bool shard_local = false);
  private:
    // This should only be available for accessor classes
    template<PrivilegeMode, typename, int, typename, typename, bool>
    friend class FieldAccessor;
    Realm::RegionInstance get_instance(
        Memory::Kind kind, size_t field_size, bool check_field_size,
        const char* warning_string, bool silence_warnings) const;
    void report_incompatible_accessor(
        const char* accessor_kind, Realm::RegionInstance instance) const;
  };

}  // namespace Legion

#include "legion/api/future.inl"

#endif  // __LEGION_FUTURE_H__
