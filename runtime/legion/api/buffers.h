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

#ifndef __LEGION_DEFERRED_BUFFERS_H__
#define __LEGION_DEFERRED_BUFFERS_H__

#include "legion/api/data.h"
#include "legion/api/values.h"

namespace Legion {

  /**
   * \struct DeferredBufferRequest
   * Configuration struct for creation of deferred buffers
   */
  struct DeferredBufferRequest {
  public:
    DeferredBufferRequest(void) = default;
    // Convenience constructors
    DeferredBufferRequest(
        Memory mem, const Domain& bounds, size_t size,
        size_t align = alignof(std::max_align_t),
        bool fortran_order_dims = false, const void* initial = nullptr,
        bool escaping = true);
    DeferredBufferRequest(
        Memory::Kind kind, const Domain& bounds, size_t size,
        size_t align = alignof(std::max_align_t),
        bool fortran_order_dims = false, const void* initial = nullptr,
        bool escaping = true);
    DeferredBufferRequest(
        Memory mem, const IndexSpace& space, size_t size,
        size_t align = alignof(std::max_align_t),
        bool fortran_order_dims = false, const void* initial = nullptr,
        bool escaping = true);
    DeferredBufferRequest(
        Memory::Kind kind, const IndexSpace& space, size_t size,
        size_t align = alignof(std::max_align_t),
        bool fortran_order_dims = false, const void* initial = nullptr,
        bool escaping = true);
  public:
    size_t field_size = 0;
    size_t alignment = alignof(std::max_align_t);
    const void* initial_value = nullptr;
    std::vector<DimensionKind> dim_order;
    union {
      Memory exact = Memory::NO_MEMORY;
      Memory::Kind kind;
    } memory;
    union {
      Domain value;
      IndexSpace name;
    } bounds = {Domain::NO_DOMAIN};
    bool is_exact = true;
    bool is_value = true;
    bool can_fail = false;
    bool escaping = true;
  };

  /**
   * \class DeferredBuffer
   * A deferred buffer is a local instance that can be made inside of a
   * task that will live just for lifetime of the task without needing to
   * be associated with a logical region. The runtime will automatically
   * reclaim the memory associated with it after the task is done. The task
   * must specify the kind of memory to use and the runtime will pick a
   * specific memory of that kind associated with current processor on
   * which the task is executing. Users can provide an optional
   * initialization value for the buffer. Users must guarantee that no
   * instances of the DeferredBuffer object live past the end of the
   * execution of a task. The user must also guarantee that DefferedBuffer
   * objects are not returned as the result of the task. The user can
   * control the layout of dimensions with the 'fortran_order_dims'
   * parameter. The default is C order dimensions (e.g. last changing
   * fastest), but can be switched to fortran order (e.g. first fastest).
   */
  template<
      typename T, int DIM, typename COORD_T = coord_t,
#ifdef LEGION_BOUNDS_CHECKS
      bool CHECK_BOUNDS = true>
#else
      bool CHECK_BOUNDS = false>
#endif
  class DeferredBuffer : public Unserializable {
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    DeferredBuffer(void);
  public:  // Constructors specifying a generic memory kind
    DeferredBuffer(
        Memory::Kind kind, const Domain& bounds,
        const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(),
        bool fortran_order_dims = false, bool escaping = true);
    DeferredBuffer(
        const Rect<DIM, COORD_T>& bounds, Memory::Kind kind,
        const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(),
        bool fortran_order_dims = false, bool escaping = true);
  public:  // Constructors specifying a specific memory
    DeferredBuffer(
        Memory memory, const Domain& bounds, const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(),
        bool fortran_order_dims = false, bool escaping = true);
    DeferredBuffer(
        const Rect<DIM, COORD_T>& bounds, Memory memory,
        const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(),
        bool fortran_order_dims = false, bool escaping = true);
  public:  // Constructors specifying a specific ordering
    DeferredBuffer(
        Memory::Kind kind, const Domain& bounds,
        std::array<DimensionKind, DIM> ordering,
        const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(), bool escaping = true);
    DeferredBuffer(
        const Rect<DIM, COORD_T>& bounds, Memory::Kind kind,
        std::array<DimensionKind, DIM> ordering,
        const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(), bool escaping = true);
    DeferredBuffer(
        Memory memory, const Domain& bounds,
        std::array<DimensionKind, DIM> ordering,
        const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(), bool escaping = true);
    DeferredBuffer(
        const Rect<DIM, COORD_T>& bounds, Memory memory,
        std::array<DimensionKind, DIM> ordering,
        const T* initial_value = nullptr,
        size_t alignment = std::alignment_of<T>(), bool escaping = true);
  public:
    __LEGION_CUDA_HD__
    inline T read(const Point<DIM, COORD_T>& p) const;
    __LEGION_CUDA_HD__
    inline void write(const Point<DIM, COORD_T>& p, T value) const;
    __LEGION_CUDA_HD__
    inline T* ptr(const Point<DIM, COORD_T>& p) const;
    __LEGION_CUDA_HD__
    inline T* ptr(const Rect<DIM, COORD_T>& r) const;  // must be dense
    __LEGION_CUDA_HD__
    inline T* ptr(const Rect<DIM, COORD_T>& r, size_t strides[DIM]) const;
    __LEGION_CUDA_HD__
    inline T& operator[](const Point<DIM, COORD_T>& p) const;
  public:
    inline bool exists(void) const { return instance.exists(); }
    inline void destroy(Realm::Event precondition = Realm::Event::NO_EVENT);
    __LEGION_CUDA_HD__
    inline Realm::RegionInstance get_instance(void) const;
    __LEGION_CUDA_HD__
    inline Rect<DIM, COORD_T> get_bounds(void) const;
  public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
  protected:
    friend class OutputRegion;
    friend class UntypedDeferredBuffer<COORD_T>;
    Realm::RegionInstance instance;
    Realm::AffineAccessor<T, DIM, COORD_T> accessor;
    std::array<DimensionKind, DIM> ordering;
    Rect<DIM, COORD_T> bounds;
    size_t alignment;
  };

  /**
   * \class UntypedDeferredBuffer
   * An untypeded deferred buffer is a type-erased representation
   * of a deferred buffer with the type of the field and the number
   * of dimensions erased.
   */
  template<typename COORD_T = coord_t>
  class UntypedDeferredBuffer : public Unserializable {
  private:
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  public:
    UntypedDeferredBuffer(void);
  public:  // Constructors specifying a generic memory kind
    UntypedDeferredBuffer(
        size_t field_size, int dims, Memory::Kind kind, const Domain& bounds,
        const void* initial_value = nullptr, size_t alignment = 16,
        bool fortran_order_dims = false, bool escaping = true);
    UntypedDeferredBuffer(
        size_t field_size, int dims, Memory::Kind kind, IndexSpace bounds,
        const void* initial_value = nullptr, size_t alignment = 16,
        bool fortran_order_dims = false, bool escaping = true);
  public:  // Constructors specifying a specific memory
    UntypedDeferredBuffer(
        size_t field_size, int dims, Memory memory, const Domain& bounds,
        const void* initial_value = nullptr, size_t alignment = 16,
        bool fortran_order_dims = false, bool escaping = true);
    UntypedDeferredBuffer(
        size_t field_size, int dims, Memory memory, IndexSpace bounds,
        const void* initial_value = nullptr, size_t alignment = 16,
        bool fortran_order_dims = false, bool escaping = true);
  public:
    template<typename T, int DIM>
    UntypedDeferredBuffer(const DeferredBuffer<T, DIM, COORD_T>& rhs);
  public:
    template<typename T, int DIM, bool BC>
    inline operator DeferredBuffer<T, DIM, COORD_T, BC>(void) const;
  public:
    inline bool exists(void) const { return instance.exists(); }
    void destroy(Realm::Event precondition = Realm::Event::NO_EVENT);
    inline Realm::RegionInstance get_instance(void) const { return instance; }
  private:
    static UntypedDeferredBuffer<COORD_T> allocate_buffer(
        const DeferredBufferRequest& request);
    static void report_nondense_rect(void);
    friend class Runtime;
    template<PrivilegeMode, typename, int, typename, typename, bool>
    friend class FieldAccessor;
    template<typename, bool, int, typename, typename, bool>
    friend class ReductionAccessor;
    template<typename, int, typename, bool>
    friend class DeferredBuffer;
    Realm::RegionInstance instance;
    size_t field_size;
    int dims;
  };

}  // namespace Legion

#include "legion/api/buffers.inl"

#endif  // __LEGION_DEFERRED_BUFFERS_H__
