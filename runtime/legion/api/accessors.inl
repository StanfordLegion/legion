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

// Included from accessors.h - do not include this directly

// Useful for IDEs
#include "legion/api/accessors.h"

namespace Legion {

  // Special namespace for providing multi-dimensional
  // array syntax on accessors
  namespace ArraySyntax {
    // A helper class for handling reductions
    template<typename A, typename FT, int N, typename T>
    class ReductionHelper {
    public:
      __LEGION_CUDA_HD__
      ReductionHelper(const A& acc, const Point<N>& p) : accessor(acc), point(p)
      { }
    public:
      __LEGION_CUDA_HD__
      inline void reduce(FT val) const { accessor.reduce(point, val); }
      __LEGION_CUDA_HD__
      inline void operator<<=(FT val) const { accessor.reduce(point, val); }
    public:
      const A& accessor;
      const Point<N, T> point;
    };

    template<typename FT, PrivilegeMode P>
    class AccessorRefHelper {
    public:
      AccessorRefHelper(const Realm::AccessorRefHelper<FT>& h) : helper(h) { }
    public:
      // read
      inline operator FT(void) const { return helper; }
      // writes
      inline AccessorRefHelper<FT, P>& operator=(const FT& newval)
      {
        helper = newval;
        return *this;
      }
      template<PrivilegeMode P2>
      inline AccessorRefHelper<FT, P>& operator=(
          const AccessorRefHelper<FT, P2>& rhs)
      {
        helper = rhs.helper;
        return *this;
      }
    protected:
      template<typename T, PrivilegeMode P2>
      friend class AccessorRefHelper;
      Realm::AccessorRefHelper<FT> helper;
    };

    template<typename FT>
    class AccessorRefHelper<FT, LEGION_READ_ONLY> {
    public:
      AccessorRefHelper(const Realm::AccessorRefHelper<FT>& h) : helper(h) { }
      // read
      inline operator FT(void) const { return helper; }
    private:
      // no writes allowed
      inline AccessorRefHelper<FT, LEGION_READ_ONLY>& operator=(
          const AccessorRefHelper<FT, LEGION_READ_ONLY>& rhs)
      {
        helper = rhs.helper;
        return *this;
      }
    protected:
      template<typename T, PrivilegeMode P2>
      friend class AccessorRefHelper;
      Realm::AccessorRefHelper<FT> helper;
    };

    // LEGION_NO_ACCESS means we dynamically check the privilege
    template<typename FT>
    class AccessorRefHelper<FT, LEGION_NO_ACCESS> {
    public:
      AccessorRefHelper(
          const Realm::AccessorRefHelper<FT>& h, FieldID fid,
          const DomainPoint& pt, PrivilegeMode p)
        : helper(h), point(pt), field(fid), privilege(p)
      { }
    public:
      // read
      inline operator FT(void) const
      {
        if ((privilege & LEGION_READ_PRIV) == 0)
          PhysicalRegion::fail_privilege_check(point, field, privilege);
        return helper;
      }
      // writes
      inline AccessorRefHelper<FT, LEGION_NO_ACCESS>& operator=(
          const FT& newval)
      {
        if ((privilege & LEGION_WRITE_PRIV) == 0)
          PhysicalRegion::fail_privilege_check(point, field, privilege);
        helper = newval;
        return *this;
      }
      template<PrivilegeMode P2>
      inline AccessorRefHelper<FT, LEGION_NO_ACCESS>& operator=(
          const AccessorRefHelper<FT, P2>& rhs)
      {
        if ((privilege & LEGION_WRITE_PRIV) == 0)
          PhysicalRegion::fail_privilege_check(point, field, privilege);
        helper = rhs.helper;
        return *this;
      }
    protected:
      template<typename T, PrivilegeMode P2>
      friend class AccessorRefHelper;
      Realm::AccessorRefHelper<FT> helper;
      DomainPoint point;
      FieldID field;
      PrivilegeMode privilege;
    };

    // A small helper class that helps provide some syntactic sugar for
    // indexing accessors like a multi-dimensional array for generic accessors
    template<typename A, typename FT, int N, typename T, int M, PrivilegeMode P>
    class GenericSyntaxHelper {
    public:
      GenericSyntaxHelper(const A& acc, const Point<M - 1, T>& p)
        : accessor(acc)
      {
        for (int i = 0; i < (M - 1); i++) point[i] = p[i];
      }
    public:
      inline GenericSyntaxHelper<A, FT, N, T, M + 1, P> operator[](T val)
      {
        point[M - 1] = val;
        return GenericSyntaxHelper<A, FT, N, T, M + 1, P>(accessor, point);
      }
    public:
      const A& accessor;
      Point<M, T> point;
    };
    // Specialization for M = N
    template<typename A, typename FT, int N, typename T, PrivilegeMode P>
    class GenericSyntaxHelper<A, FT, N, T, N, P> {
    public:
      GenericSyntaxHelper(const A& acc, const Point<N - 1, T>& p)
        : accessor(acc)
      {
        for (int i = 0; i < (N - 1); i++) point[i] = p[i];
      }
    public:
      inline AccessorRefHelper<FT, P> operator[](T val)
      {
        point[N - 1] = val;
        return accessor[point];
      }
    public:
      const A& accessor;
      Point<N, T> point;
    };
    // Further specialization for M = N and read-only
    template<typename A, typename FT, int N, typename T>
    class GenericSyntaxHelper<A, FT, N, T, N, LEGION_READ_ONLY> {
    public:
      GenericSyntaxHelper(const A& acc, const Point<N - 1, T>& p)
        : accessor(acc)
      {
        for (int i = 0; i < (N - 1); i++) point[i] = p[i];
      }
    public:
      inline AccessorRefHelper<FT, LEGION_READ_ONLY> operator[](T val)
      {
        point[N - 1] = val;
        return accessor[point];
      }
    public:
      const A& accessor;
      Point<N, T> point;
    };
    // Further specialization for M = N and reductions
    template<typename A, typename FT, int N, typename T>
    class GenericSyntaxHelper<A, FT, N, T, N, LEGION_REDUCE> {
    public:
      GenericSyntaxHelper(const A& acc, const Point<N - 1, T>& p)
        : accessor(acc)
      {
        for (int i = 0; i < (N - 1); i++) point[i] = p[i];
      }
    public:
      inline const ReductionHelper<A, FT, N, T> operator[](T val)
      {
        point[N - 1] = val;
        return ReductionHelper<A, FT, N, T>(accessor, point);
      }
    public:
      const A& accessor;
      Point<N, T> point;
    };

    // A small helper class that helps provide some syntactic sugar for
    // indexing accessors like a multi-dimensional array for affine accessors
    template<typename A, typename FT, int N, typename T, int M, PrivilegeMode P>
    class AffineSyntaxHelper {
    public:
      __LEGION_CUDA_HD__
      AffineSyntaxHelper(const A& acc, const Point<M - 1, T>& p) : accessor(acc)
      {
        for (int i = 0; i < (M - 1); i++) point[i] = p[i];
      }
    public:
      __LEGION_CUDA_HD__
      inline AffineSyntaxHelper<A, FT, N, T, M + 1, P> operator[](T val)
      {
        point[M - 1] = val;
        return AffineSyntaxHelper<A, FT, N, T, M + 1, P>(accessor, point);
      }
    public:
      const A& accessor;
      Point<M, T> point;
    };

    // Specialization for M = N
    template<typename A, typename FT, int N, typename T, PrivilegeMode P>
    class AffineSyntaxHelper<A, FT, N, T, N, P> {
    public:
      __LEGION_CUDA_HD__
      AffineSyntaxHelper(const A& acc, const Point<N - 1, T>& p) : accessor(acc)
      {
        for (int i = 0; i < (N - 1); i++) point[i] = p[i];
      }
    public:
      __LEGION_CUDA_HD__
      inline FT& operator[](T val)
      {
        point[N - 1] = val;
        return accessor[point];
      }
    public:
      const A& accessor;
      Point<N, T> point;
    };

    // Further specialization for M = N and read-only
    template<typename A, typename FT, int N, typename T>
    class AffineSyntaxHelper<A, FT, N, T, N, LEGION_READ_ONLY> {
    public:
      __LEGION_CUDA_HD__
      AffineSyntaxHelper(const A& acc, const Point<N - 1, T>& p) : accessor(acc)
      {
        for (int i = 0; i < (N - 1); i++) point[i] = p[i];
      }
    public:
      __LEGION_CUDA_HD__
      inline const FT& operator[](T val)
      {
        point[N - 1] = val;
        return accessor[point];
      }
    public:
      const A& accessor;
      Point<N, T> point;
    };

    // Further specialize for M = N and reductions
    template<typename A, typename FT, int N, typename T>
    class AffineSyntaxHelper<A, FT, N, T, N, LEGION_REDUCE> {
    public:
      __LEGION_CUDA_HD__
      AffineSyntaxHelper(const A& acc, const Point<N - 1, T>& p) : accessor(acc)
      {
        for (int i = 0; i < (N - 1); i++) point[i] = p[i];
      }
    public:
      __LEGION_CUDA_HD__
      inline const ReductionHelper<A, FT, N, T> operator[](T val)
      {
        point[N - 1] = val;
        return ReductionHelper<A, FT, N, T>(accessor, point);
      }
    public:
      const A& accessor;
      Point<N, T> point;
    };

    // Helper class for affine syntax that behaves like a
    // pointer/reference, but does dynamic privilege checks
    template<typename FT>
    class AffineRefHelper {
    public:
      __LEGION_CUDA_HD__
      AffineRefHelper(
          FT& r, FieldID fid, const DomainPoint& pt, PrivilegeMode p)
        : ref(r),
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
          point(pt), field(fid),
#endif
          privilege(p)
      { }
    public:
      // read
      __LEGION_CUDA_HD__
      inline operator const FT &(void) const
      {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        assert(privilege & LEGION_READ_PRIV);
#else
        if ((privilege & LEGION_READ_PRIV) == 0)
          PhysicalRegion::fail_privilege_check(point, field, privilege);
#endif
        return ref;
      }
      // writes
      __LEGION_CUDA_HD__
      inline AffineRefHelper<FT>& operator=(const FT& newval)
      {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        assert(privilege & LEGION_WRITE_PRIV);
#else
        if ((privilege & LEGION_WRITE_PRIV) == 0)
          PhysicalRegion::fail_privilege_check(point, field, privilege);
#endif
        ref = newval;
        return *this;
      }
      __LEGION_CUDA_HD__
      inline AffineRefHelper<FT>& operator=(const AffineRefHelper<FT>& rhs)
      {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        assert(privilege & LEGION_WRITE_PRIV);
#else
        if ((privilege & LEGION_WRITE_PRIV) == 0)
          PhysicalRegion::fail_privilege_check(point, field, privilege);
#endif
        ref = rhs.ref;
        return *this;
      }
    protected:
      FT& ref;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
      DomainPoint point;
      FieldID field;
#endif
      PrivilegeMode privilege;
    };

    // Further specialization for M = N and NO_ACCESS (dynamic privilege)
    template<typename A, typename FT, int N, typename T>
    class AffineSyntaxHelper<A, FT, N, T, N, LEGION_NO_ACCESS> {
    public:
      __LEGION_CUDA_HD__
      AffineSyntaxHelper(const A& acc, const Point<N - 1, T>& p) : accessor(acc)
      {
        for (int i = 0; i < (N - 1); i++) point[i] = p[i];
      }
    public:
      __LEGION_CUDA_HD__
      inline AffineRefHelper<FT> operator[](T val)
      {
        point[N - 1] = val;
        return accessor[point];
      }
    public:
      const A& accessor;
      Point<N, T> point;
    };
  };  // namespace ArraySyntax

  ////////////////////////////////////////////////////////////
  // Constructors for Generic Accessors
  ////////////////////////////////////////////////////////////

#define PHYSICAL_REGION_CONSTRUCTORS(PRIVILEGE, DIM, FIELD_CHECK)              \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, true /*generic accessor*/, check_field_size);        \
    if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                    \
            instance, fid, is.bounds))                                         \
      region.report_incompatible_accessor("GenericAccessor", instance, fid);   \
    accessor =                                                                 \
        Realm::GenericAccessor<FT, DIM, T>(instance, fid, is.bounds, offset);  \
  }                                                                            \
  /* with source bounds */                                                     \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, true /*generic accessor*/, check_field_size);        \
    if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                    \
            instance, fid, source_bounds))                                     \
      region.report_incompatible_accessor("GenericAccessor", instance, fid);   \
    accessor = Realm::GenericAccessor<FT, DIM, T>(                             \
        instance, fid, source_bounds, offset);                                 \
  }                                                                            \
  /* colocation regions */                                                     \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("GenericAccessor", fid);        \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, true /*generic accessor*/, check_field_size);      \
      if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                  \
              inst, fid, is.bounds))                                           \
        it->report_incompatible_accessor("GenericAccessor", inst, fid);        \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "GenericAccessor", fid, instance, inst, *start);                   \
      else                                                                     \
        instance = inst;                                                       \
    }                                                                          \
    accessor = Realm::GenericAccessor<FT, DIM, T>(instance, fid, offset);      \
  }                                                                            \
  /* colocation regions with source bounds */                                  \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("GenericAccessor", fid);        \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, true /*generic accessor*/, check_field_size);      \
      if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                  \
              inst, fid, source_bounds))                                       \
        it->report_incompatible_accessor("GenericAccessor", inst, fid);        \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "GenericAccessor", fid, instance, inst, *start);                   \
      else                                                                     \
        instance = inst;                                                       \
    }                                                                          \
    accessor = Realm::GenericAccessor<FT, DIM, T>(                             \
        instance, fid, source_bounds, offset);                                 \
  }

#define PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(PRIVILEGE, DIM, FIELD_CHECK)  \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &bounds,                            \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, true /*generic accessor*/, check_field_size);        \
    if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                    \
            instance, fid, bounds.bounds))                                     \
      region.report_incompatible_accessor("GenericAccessor", instance, fid);   \
    accessor = Realm::GenericAccessor<FT, DIM, T>(                             \
        instance, fid, bounds.bounds, offset);                                 \
  }                                                                            \
  /* with source bounds */                                                     \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid), bounds(source_bounds)                                        \
  {                                                                            \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &bounds,                            \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, true /*generic accessor*/, check_field_size);        \
    if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                    \
            instance, fid, source_bounds))                                     \
      region.report_incompatible_accessor("GenericAccessor", instance, fid);   \
    accessor = Realm::GenericAccessor<FT, DIM, T>(                             \
        instance, fid, source_bounds, offset);                                 \
    bounds.bounds = source_bounds.intersection(bounds.bounds);                 \
  }                                                                            \
  /* colocation regions */                                                     \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("GenericAccessor", fid);        \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, true /*generic accessor*/, check_field_size);      \
      if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                  \
              inst, fid, is.bounds))                                           \
        it->report_incompatible_accessor("GenericAccessor", inst, fid);        \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "GenericAccessor", fid, instance, inst, *start);                   \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
    }                                                                          \
    accessor = Realm::GenericAccessor<FT, DIM, T>(instance, fid, offset);      \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, bounds, Realm::ProfilingRequestSet()));                          \
    /* Defer delete the bounds when the task is done */                        \
    bounds.destroy(Processor::get_current_finish_event());                     \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
  }                                                                            \
  /* colocation regions with source bounds */                                  \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid), bounds(source_bounds)                                        \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("GenericAccessor", fid);        \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, true /*generic accessor*/, check_field_size);      \
      if (!Realm::GenericAccessor<FT, DIM, T>::is_compatible(                  \
              inst, fid, source_bounds))                                       \
        it->report_incompatible_accessor("GenericAccessor", inst, fid);        \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "GenericAccessor", fid, instance, inst, *start);                   \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
    }                                                                          \
    accessor = Realm::GenericAccessor<FT, DIM, T>(                             \
        instance, fid, source_bounds, offset);                                 \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, bounds, Realm::ProfilingRequestSet()));                          \
    /* Defer delete the bounds when the task is done */                        \
    bounds.destroy(Processor::get_current_finish_event());                     \
    /* Update the bounding box of the bounds with source bounds */             \
    bounds.bounds = source_bounds.intersection(bounds.bounds);                 \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
  }

  ////////////////////////////////////////////////////////////
  // Specializations for Generic Accessors
  ////////////////////////////////////////////////////////////

  // Read-only FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, N, false)
#endif
  public:
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    inline const ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>
        operator[](const Point<N, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_READ_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>, CB>,
        FT, N, T, 2, LEGION_READ_ONLY>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_READ_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>, CB>,
          FT, N, T, 2, LEGION_READ_ONLY>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-only FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, N, false)
#endif
  public:
    inline FT read(const Point<N, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return accessor.read(p);
    }
    inline const ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>
        operator[](const Point<N, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_READ_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>, true>,
        FT, N, T, 2, LEGION_READ_ONLY>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_READ_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_READ_ONLY>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
    FieldID field;
    DomainT<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-only FieldAccessor specialization
  // with N==1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, 1, false)
#endif
  public:
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    inline const ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>
        operator[](const Point<1, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>(accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-only FieldAccessor specialization
  // with N==1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, 1, false)
#endif
  public:
    inline FT read(const Point<1, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return accessor.read(p);
    }
    inline const ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>
        operator[](const Point<1, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_ONLY>(accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
    FieldID field;
    DomainT<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-write FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, N, T, Realm::GenericAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, N, false)
#endif
  public:
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<N, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_READ_WRITE, FT, N, T, Realm::GenericAccessor<FT, N, T>, CB>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_READ_WRITE, FT, N, T, Realm::GenericAccessor<FT, N, T>,
              CB>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    // No reductions since we can't handle atomicity correctly
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-write FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, N, T, Realm::GenericAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, N, false)
#endif
  public:
    inline FT read(const Point<N, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return accessor.read(p);
    }
    inline void write(const Point<N, T>& p, FT val) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<N, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_READ_WRITE, FT, N, T, Realm::GenericAccessor<FT, N, T>,
            true>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_READ_WRITE, FT, N, T, Realm::GenericAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    // No reductions since we can't handle atomicity correctly
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
    FieldID field;
    DomainT<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-write FieldAccessor specialization
  // with N==1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, 1, false)
#endif
  public:
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<1, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    // No reductions since we can't handle atomicity correctly
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-write FieldAccessor specialization
  // with N==1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, 1, false)
#endif
  public:
    inline FT read(const Point<1, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return accessor.read(p);
    }
    inline void write(const Point<1, T>& p, FT val) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<1, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    // No reduction since we can't handle atomicity correctly
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
    FieldID field;
    DomainT<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-discard FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, false)
#endif
  public:
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<N, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
            CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
              CB>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-discard FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, false)
#endif
  public:
    inline FT read(const Point<N, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return accessor.read(p);
    }
    inline void write(const Point<N, T>& p, FT val) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<N, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
            true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
    FieldID field;
    DomainT<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-discard FieldAccessor specialization with
  // N == 1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, false)
#endif
  public:
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<1, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-discard FieldAccessor specialization with
  // N == 1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, false)
#endif
  public:
    inline FT read(const Point<1, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
      return accessor.read(p);
    }
    inline void write(const Point<1, T>& p, FT val) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<1, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
    FieldID field;
    DomainT<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-only FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, false)
#endif
  public:
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<N, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
            CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
              CB>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-only FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, N, T, Realm::GenericAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_ONLY, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_ONLY, N, false)
#endif
  public:
    inline void write(const Point<N, T>& p, FT val) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<N, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
            true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::GenericAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
    FieldID field;
    DomainT<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-only FieldAccessor specialization with
  // N == 1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_ONLY, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_ONLY, 1, false)
#endif
  public:
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<1, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-only FieldAccessor specialization with
  // N == 1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, 1, T, Realm::GenericAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_ONLY, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_ONLY, 1, false)
#endif
  public:
    inline void write(const Point<1, T>& p, FT val) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD> operator[](
        const Point<1, T>& p) const
    {
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_WRITE_DISCARD>(
          accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
    FieldID field;
    DomainT<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

#undef PHYSICAL_REGION_CONSTRUCTORS
#undef PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS

  // Special namespace for providing bounds check help for affine accessors
  namespace AffineBounds {
    // A helper class for testing bounds for affine accessors
    // which might have a transform associated with them
    // We should never use the base version of this, only the specializations
    template<int N, typename T>
    class Tester {
    public:
      static_assert(N > 0, "Accessor DIM must be positive");
      static_assert(
          N <= LEGION_MAX_DIM, "Accessor DIM larger than LEGION_MAX_DIM");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      Tester(void) { }
      Tester(const DomainT<N, T>& b) : domain(b) { }
      Tester(const DomainT<N, T>& b, const Rect<N, T> s)
        : domain(intersect_with_rect(b, s))
      { }
      template<int M2>
      Tester(const DomainT<M2, T> b, const AffineTransform<M2, N, T> t)
        : domain(full_rect()), range(b), transform(t)
      {
        static_assert(M2 > 0, "Accessor DIM must be positive");
        static_assert(
            M2 <= LEGION_MAX_DIM, "Accessor DIM larger than LEGION_MAX_DIM");
      }
      template<int M2>
      Tester(
          const DomainT<M2, T> b, const Rect<N, T> s,
          const AffineTransform<M2, N, T> t)
        : domain(s), range(b), transform(t)
      {
        static_assert(M2 > 0, "Accessor DIM must be positive");
        static_assert(
            M2 <= LEGION_MAX_DIM, "Accessor DIM larger than LEGION_MAX_DIM");
      }
    public:
      __LEGION_CUDA_HD__
      inline bool contains(const Point<N, T>& point) const
      {
        // Check the domain first
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (!domain.dense())
          check_gpu_warning();
#endif
        if (!test_point(domain, point))
          return false;
        const int range_dim = range.get_dim();
        if (range_dim > 0)
        {
          assert(range_dim <= LEGION_MAX_DIM);
          // We have a transform so need to convert and perform the test
          switch (range_dim)
          {
#define DIMFUNC(DIM)                                  \
  case DIM:                                           \
    {                                                 \
      const DomainT<DIM, T> r = convert_range<DIM>(); \
      const AffineTransform<DIM, N, T> t = transform; \
      return test_point<DIM>(r, t[point]);            \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          }
        }
        return true;
      }
      __LEGION_CUDA_HD__
      inline bool contains_all(const Rect<N, T>& rect) const
      {
        if (rect.empty())
          return true;
        // Check the domain first
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // Can only test bounds on the GPU and not sparsity maps yet
        if (!domain.dense())
          check_gpu_warning();
        if (!domain.bounds.contains(rect))
          return false;
#else
        // On the host so we can check sparsity maps too
        if (!domain.contains_all(rect))
          return false;
#endif
        const int range_dim = range.get_dim();
        if (range_dim > 0)
        {
          assert(range_dim <= LEGION_MAX_DIM);
          // We have a transform so need to convert and perform the test
          switch (range_dim)
          {
#define DIMFUNC(DIM)                                          \
  case DIM:                                                   \
    {                                                         \
      const DomainT<DIM, T> r = convert_range<DIM>();         \
      const AffineTransform<DIM, N, T> t = transform;         \
      for (PointInRectIterator<N, T> itr(rect); itr(); itr++) \
        if (!test_point<DIM>(r, t[*itr]))                     \
          return false;                                       \
      break;                                                  \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          }
        }
        return true;
      }
    private:
      static inline DomainT<N, T> intersect_with_rect(
          const DomainT<N, T>& domain, const Rect<N, T>& rect)
      {
        DomainT<N, T> result = domain;
        result.bounds = domain.bounds.intersection(rect);
        return result;
      }
      static inline DomainT<N, T> full_rect(void)
      {
        DomainT<N, T> result = DomainT<N, T>::make_empty();
        for (int d = 0; d < N; d++)
        {
          result.bounds.lo[d] = std::numeric_limits<T>::min;
          result.bounds.hi[d] = std::numeric_limits<T>::max;
        }
        return result;
      }
      __LEGION_CUDA_HD__
      inline void check_gpu_warning(void) const
      {
        // We've turned this off for now since most users seems to
        // think it is too verbose, but we can renable it if needed
#if 0
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        bool need_warning = !bounds.dense();
        if (need_warning)
          printf("WARNING: GPU bounds check is imprecise!\n");
#endif
#endif
      }
      template<int M>
      __LEGION_CUDA_HD__ inline DomainT<M, T> convert_range(void) const
      {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // Can't see sparsity maps on the GPU
        const DomainT<M, T> result(range.bounds<M, T>());
#else
        const DomainT<M, T> result = range;
#endif
        return result;
      }
      template<int M>
      __LEGION_CUDA_HD__ static inline bool test_point(
          const DomainT<M, T>& domain, const Point<M, T>& point)
      {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // Can only test bounds on the GPU and not sparsity maps yet
        return domain.bounds.contains(point);
#else
        // Can test sparsity maps on the host
        return domain.contains(point);
#endif
      }
    private:
      DomainT<N, T> domain;  // always check for inputs
      Domain range;          // in case we have a transform
      DomainAffineTransform transform;
    };
  };  // namespace AffineBounds

  ////////////////////////////////////////////////////////////
  // Macros for PhysicalRegion Constructors with Affine Accessors
  ////////////////////////////////////////////////////////////

#define PHYSICAL_REGION_CONSTRUCTORS(PRIVILEGE, DIM, FIELD_CHECK)              \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, is.bounds))                                         \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor =                                                                 \
        Realm::AffineAccessor<FT, DIM, T>(instance, fid, is.bounds, offset);   \
  }                                                                            \
  /* With explicit bounds */                                                   \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, source_bounds))                                     \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, fid, source_bounds, offset);                                 \
  }                                                                            \
  /* With explicit transform */                                                \
  template<int M>                                                              \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const AffineTransform<M, DIM, T> transform,                              \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<M, T> is;                                                          \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,       \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid))             \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, offset);         \
  }                                                                            \
  /* With explicit transform and bounds */                                     \
  template<int M>                                                              \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const AffineTransform<M, DIM, T> transform,                              \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<M, T> is;                                                          \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,       \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid,              \
            source_bounds))                                                    \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, source_bounds,   \
        offset);                                                               \
  }                                                                            \
  /* colocation regions */                                                     \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    Rect<DIM, T> bounding_box;                                                 \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (it == start)                                                         \
        bounding_box = is.bounds;                                              \
      else                                                                     \
        bounding_box = bounding_box.union_bbox(is.bounds);                     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, bounding_box))                                      \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, fid, bounding_box, offset);                                  \
  }                                                                            \
  /* colocation regions ith explicit bounds */                                 \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, source_bounds))                                     \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, fid, source_bounds, offset);                                 \
  }                                                                            \
  /* colocation regions with explicit transform */                             \
  template<typename InputIterator, int M>                                      \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const AffineTransform<M, DIM, T> transform,                              \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<M, T> is;                                                        \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,     \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid))             \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, offset);         \
  }                                                                            \
  /* colocation regions with explicit transform and bounds */                  \
  template<typename InputIterator, int M>                                      \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const AffineTransform<M, DIM, T> transform,                              \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<M, T> is;                                                        \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,     \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid,              \
            source_bounds))                                                    \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, source_bounds,   \
        offset);                                                               \
  }

#define PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(PRIVILEGE, DIM, FIELD_CHECK)  \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, is.bounds))                                         \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor =                                                                 \
        Realm::AffineAccessor<FT, DIM, T>(instance, fid, is.bounds, offset);   \
    bounds = AffineBounds::Tester<DIM, T>(is);                                 \
  }                                                                            \
  /* With explicit bounds */                                                   \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, source_bounds))                                     \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, fid, source_bounds, offset);                                 \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                  \
  }                                                                            \
  /* With explicit transform */                                                \
  template<int M>                                                              \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const AffineTransform<M, DIM, T> transform,                              \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    DomainT<M, T> is;                                                          \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,       \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid))             \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, offset);         \
    bounds = AffineBounds::Tester<DIM, T>(is, transform);                      \
  }                                                                            \
  /* With explicit transform and bounds */                                     \
  template<int M>                                                              \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const AffineTransform<M, DIM, T> transform,                              \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    DomainT<M, T> is;                                                          \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,       \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid,              \
            source_bounds))                                                    \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, source_bounds,   \
        offset);                                                               \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, transform);       \
  }                                                                            \
  /* colocation regions */                                                     \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    Rect<DIM, T> bounding_box;                                                 \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (it == start)                                                         \
        bounding_box = is.bounds;                                              \
      else                                                                     \
        bounding_box = bounding_box.union_bbox(is.bounds);                     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, bounding_box))                                      \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, fid, bounding_box, offset);                                  \
    DomainT<DIM, T> is;                                                        \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, is, Realm::ProfilingRequestSet()));                              \
    /* Defer delete the bounds when the task is done */                        \
    is.destroy(Processor::get_current_finish_event());                         \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is);                                 \
  }                                                                            \
  /* colocation regions with explicit bounds */                                \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, fid, source_bounds))                                     \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, fid, source_bounds, offset);                                 \
    DomainT<DIM, T> is;                                                        \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, is, Realm::ProfilingRequestSet()));                              \
    /* Defer delete the bounds when the task is done */                        \
    is.destroy(Processor::get_current_finish_event());                         \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                  \
  }                                                                            \
  /* colocation regions with explicit transform */                             \
  template<typename InputIterator, int M>                                      \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const AffineTransform<M, DIM, T> transform,                              \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<M, T> is;                                                        \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,     \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid))             \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, offset);         \
    DomainT<DIM, T> is;                                                        \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, is, Realm::ProfilingRequestSet()));                              \
    /* Defer delete the bounds when the task is done */                        \
    is.destroy(Processor::get_current_finish_event());                         \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is, transform);                      \
  }                                                                            \
  /* colocation regions with explicit transform and bounds */                  \
  template<typename InputIterator, int M>                                      \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const AffineTransform<M, DIM, T> transform,                              \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid);         \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<M, T> is;                                                        \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,     \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "AffineAccessor", fid, instance, inst, *start);                    \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
    }                                                                          \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                     \
            instance, transform.transform, transform.offset, fid,              \
            source_bounds))                                                    \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);    \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                              \
        instance, transform.transform, transform.offset, fid, source_bounds,   \
        offset);                                                               \
    DomainT<DIM, T> is;                                                        \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, is, Realm::ProfilingRequestSet()));                              \
    /* Defer delete the bounds when the task is done */                        \
    is.destroy(Processor::get_current_finish_event());                         \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, transform);       \
  }

  ////////////////////////////////////////////////////////////
  // Macros UntypedDeferredValue/UntypedDeferredBuffer
  // Constructors with Affine Accessors
  ////////////////////////////////////////////////////////////

#define DEFERRED_VALUE_BUFFER_CONSTRUCTORS(DIM, FIELD_CHECK)                  \
  FieldAccessor(                                                              \
      const UntypedDeferredValue& value,                                      \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    Realm::Rect<DIM, T> source_bounds;                                        \
    /* Anything in range works for these bounds since we're */                \
    /* going to remap them to the origin */                                   \
    for (int i = 0; i < DIM; i++)                                             \
    {                                                                         \
      source_bounds.lo[i] = std::numeric_limits<T>::min();                    \
      source_bounds.hi[i] = std::numeric_limits<T>::max();                    \
    }                                                                         \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
  }                                                                           \
  FieldAccessor(                                                              \
      const UntypedDeferredValue& value, const Rect<DIM, T>& source_bounds,   \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
  }                                                                           \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, 0 /*field id*/, is.bounds))                             \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, 0 /*field id*/, is.bounds, offset);                         \
  }                                                                           \
  /* With explicit bounds */                                                  \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const Rect<DIM, T>& source_bounds,                                      \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, 0 /*field id*/, source_bounds))                         \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, 0 /*field id*/, source_bounds, offset);                     \
  }                                                                           \
  /* With explicit transform */                                               \
  template<int M>                                                             \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform.transform, transform.offset, 0 /*field id*/)) \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform.transform, transform.offset, 0 /*field id*/,      \
        offset);                                                              \
  }                                                                           \
  /* With explicit transform and bounds */                                    \
  template<int M>                                                             \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      const Rect<DIM, T>& source_bounds,                                      \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform.transform, transform.offset, 0 /*fid*/,       \
            source_bounds))                                                   \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform.transform, transform.offset, 0 /*fid*/,           \
        source_bounds, offset);                                               \
  }

#define DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(DIM, FIELD_CHECK)      \
  FieldAccessor(                                                              \
      const UntypedDeferredValue& value,                                      \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    Realm::Rect<DIM, T> source_bounds;                                        \
    /* Anything in range works for these bounds since we're */                \
    /* going to remap them to the origin */                                   \
    for (int i = 0; i < DIM; i++)                                             \
    {                                                                         \
      source_bounds.lo[i] = std::numeric_limits<T>::min();                    \
      source_bounds.hi[i] = std::numeric_limits<T>::max();                    \
    }                                                                         \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
    DomainT<1, T> is;                                                         \
    is.bounds.lo[0] = 0;                                                      \
    is.bounds.hi[0] = 0;                                                      \
    is.sparsity.id = 0;                                                       \
    AffineTransform<1, DIM, T> affine(transform, origin);                     \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, affine);         \
  }                                                                           \
  FieldAccessor(                                                              \
      const UntypedDeferredValue& value, const Rect<DIM, T>& source_bounds,   \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
    DomainT<1, T> is;                                                         \
    is.bounds.lo[0] = 0;                                                      \
    is.bounds.hi[0] = 0;                                                      \
    is.sparsity.id = 0;                                                       \
    AffineTransform<1, DIM, T> affine(transform, origin);                     \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, affine);         \
  }                                                                           \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, 0 /*field id*/, is.bounds))                             \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, 0 /*field id*/, is.bounds, offset);                         \
    bounds = AffineBounds::Tester<DIM, T>(is);                                \
  }                                                                           \
  /* With explicit bounds */                                                  \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const Rect<DIM, T>& source_bounds,                                      \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, 0 /*field id*/, source_bounds))                         \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, 0 /*field id*/, source_bounds, offset);                     \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                 \
  }                                                                           \
  /* With explicit transform */                                               \
  template<int M>                                                             \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform.transform, transform.offset, 0 /*field id*/)) \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform.transform, transform.offset, 0 /*field id*/,      \
        offset);                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is, transform);                     \
  }                                                                           \
  /* With explicit transform and bounds */                                    \
  template<int M>                                                             \
  FieldAccessor(                                                              \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      const Rect<DIM, T>& source_bounds,                                      \
      size_t actual_field_size = sizeof(FT),                                  \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,     \
      const char* warning_string = nullptr, size_t offset = 0)                \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<FT, DIM, T>::is_compatible(                    \
            instance, transform.transform, transform.offset, 0 /*fid*/,       \
            source_bounds))                                                   \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<FT, DIM, T>(                             \
        instance, transform.transform, transform.offset, 0 /*fid*/,           \
        source_bounds, offset);                                               \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, transform);      \
  }

  ////////////////////////////////////////////////////////////
  // Specializations for Affine Accessors
  ////////////////////////////////////////////////////////////

  // Read-only FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
    // Future accessor
    FieldAccessor(
        const Future& future, Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, N, T> transform;
      for (int i = 0; i < N; i++) transform[0][i] = 0;
      Realm::Point<1, T> origin(0);
      Realm::Rect<N, T> source_bounds;
      // Anything in range works for these bounds since we're
      // going to remap them to the origin
      for (int i = 0; i < N; i++)
      {
        source_bounds.lo[i] = std::numeric_limits<T>::min();
        source_bounds.hi[i] = std::numeric_limits<T>::max();
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
    }
    // Future accessor with explicit bounds
    FieldAccessor(
        const Future& future, const Rect<N, T> source_bounds,
        Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, N, T> transform;
      for (int i = 0; i < N; i++) transform[0][i] = 0;
      Realm::Point<1, T> origin(0);
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<N, T>& p) const
    {
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>,
        FT, N, T, 2, LEGION_READ_ONLY>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>,
          FT, N, T, 2, LEGION_READ_ONLY>(*this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-only FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
    // Future accessor
    FieldAccessor(
        const Future& future, Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, N, T> transform;
      for (int i = 0; i < N; i++) transform[0][i] = 0;
      Realm::Point<1, T> origin(0);
      Realm::Rect<N, T> source_bounds;
      // Anything in range works for these bounds since we're
      // going to remap them to the origin
      for (int i = 0; i < N; i++)
      {
        source_bounds.lo[i] = std::numeric_limits<T>::min();
        source_bounds.hi[i] = std::numeric_limits<T>::max();
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
      DomainT<1, T> is;
      is.bounds.lo[0] = 0;
      is.bounds.hi[0] = 0;
      is.sparsity.id = 0;
      AffineTransform<1, N, T> affine(transform, origin);
      bounds = AffineBounds::Tester<N, T>(is, source_bounds, affine);
    }
    // Future accessor with explicit bounds
    FieldAccessor(
        const Future& future, const Rect<N, T> source_bounds,
        Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, N, T> transform;
      for (int i = 0; i < N; i++) transform[0][i] = 0;
      Realm::Point<1, T> origin(0);
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
      DomainT<1, T> is;
      is.bounds.lo[0] = 0;
      is.bounds.hi[0] = 0;
      is.sparsity.id = 0;
      AffineTransform<1, N, T> affine(transform, origin);
      bounds = AffineBounds::Tester<N, T>(is, source_bounds, affine);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
#endif
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, true>,
        FT, N, T, 2, LEGION_READ_ONLY>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_READ_ONLY>(*this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-only FieldAccessor specialization
  // with N==1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
    // Future accessor
    FieldAccessor(
        const Future& future, Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, 1, T> transform;
      transform[0][0] = 0;
      Realm::Point<1, T> origin(0);
      Realm::Rect<1, T> source_bounds;
      // Anything in range works for these bounds since we're
      // going to remap them to the origin
      source_bounds.lo[0] = std::numeric_limits<T>::min();
      source_bounds.hi[0] = std::numeric_limits<T>::max();
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
    }
    // Future accessor with explicit bounds
    FieldAccessor(
        const Future& future, const Rect<1, T> source_bounds,
        Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, 1, T> transform;
      transform[0][0] = 0;
      Realm::Point<1, T> origin(0);
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<1, T>& p) const
    {
      return accessor[p];
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-only FieldAccessor specialization
  // with N==1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
    // Future accessor
    FieldAccessor(
        const Future& future, Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, 1, T> transform;
      transform[0][0] = 0;
      Realm::Point<1, T> origin(0);
      Realm::Rect<1, T> source_bounds;
      // Anything in range works for these bounds since we're
      // going to remap them to the origin
      source_bounds.lo[0] = std::numeric_limits<T>::min();
      source_bounds.hi[0] = std::numeric_limits<T>::max();
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
      DomainT<1, T> is;
      is.bounds.lo[0] = 0;
      is.bounds.hi[0] = 0;
      is.sparsity.id = 0;
      AffineTransform<1, 1, T> affine(transform, origin);
      bounds = AffineBounds::Tester<1, T>(is, source_bounds, affine);
    }
    // Future accessor with explicit bounds
    FieldAccessor(
        const Future& future, const Rect<1, T> source_bounds,
        Memory::Kind memkind = Memory::NO_MEMKIND,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      const Realm::RegionInstance instance = future.get_instance(
          memkind, actual_field_size, check_field_size, warning_string,
          silence_warnings);
      // This mapping ignores the input points and sends
      // everything to the 1-D origin
      Realm::Matrix<1, 1, T> transform;
      transform[0][0] = 0;
      Realm::Point<1, T> origin(0);
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform, origin, 0 /*field id*/, source_bounds))
        future.report_incompatible_accessor("AffineAccessor", instance);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform, origin, 0 /*field id*/, source_bounds, offset);
      DomainT<1, T> is;
      is.bounds.lo[0] = 0;
      is.bounds.hi[0] = 0;
      is.sparsity.id = 0;
      AffineTransform<1, 1, T> affine(transform, origin);
      bounds = AffineBounds::Tester<1, T>(is, source_bounds, affine);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
#endif
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor[p];
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-write FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-write FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
#endif
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, true>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-write FieldAccessor specialization
  // with N==1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-write FieldAccessor specialization
  // with N==1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
#endif
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-discard FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
            CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
              CB>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-discard FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
#endif
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
            true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-discard FieldAccessor specialization with
  // N == 1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-discard FieldAccessor specialization with
  // N == 1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
#endif
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-only FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
            CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
              CB>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-only FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
#endif
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
            true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-only FieldAccessor specialization with
  // N == 1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-only FieldAccessor specialization with
  // N == 1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, true)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, false)
    DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
#endif
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

#undef PHYSICAL_REGION_CONSTRUCTORS
#undef PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS
#undef DEFERRED_VALUE_BUFFER_CONSTRUCTORS
#undef DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS

#define PHYSICAL_REGION_CONSTRUCTORS(DIM, FIELD_CHECK)                        \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, is.bounds))                                        \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, is.bounds, offset);                                    \
  }                                                                           \
  /* With explicit bounds */                                                  \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, source_bounds))                                    \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, source_bounds, offset);                                \
  }                                                                           \
  /* With explicit transform */                                               \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const AffineTransform<M, DIM, T> transform,                             \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    DomainT<M, T> is;                                                         \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,      \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid))            \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, offset);        \
  }                                                                           \
  /* With explicit transform and bounds */                                    \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const AffineTransform<M, DIM, T> transform,                             \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    DomainT<M, T> is;                                                         \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,      \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid,             \
            source_bounds))                                                   \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, source_bounds,  \
        offset);                                                              \
  }                                                                           \
  /* colocation regions */                                                    \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, bool silence_warnings = false,                     \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    Rect<DIM, T> bounding_box;                                                \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (it == start)                                                        \
        bounding_box = is.bounds;                                             \
      else                                                                    \
        bounding_box = bounding_box.union_bbox(is.bounds);                    \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, bounding_box))                                     \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, bounding_box, offset);                                 \
  }                                                                           \
  /* colocation regions with explicit bounds */                               \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const Rect<DIM, T> source_bounds,                  \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, source_bounds))                                    \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, source_bounds, offset);                                \
  }                                                                           \
  /* colocation regions with explicit transform */                            \
  template<typename InputIterator, int M>                                     \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const AffineTransform<M, DIM, T> transform,        \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<M, T> is;                                                       \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,    \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid))            \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, offset);        \
  }                                                                           \
  /* colocation regions with explicit transform and bounds */                 \
  template<typename InputIterator, int M>                                     \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const AffineTransform<M, DIM, T> transform,        \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<M, T> is;                                                       \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,    \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid,             \
            source_bounds))                                                   \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, source_bounds,  \
        offset);                                                              \
  }

#define PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(DIM, FIELD_CHECK)            \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, is.bounds))                                        \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, is.bounds, offset);                                    \
    bounds = AffineBounds::Tester<DIM, T>(is);                                \
  }                                                                           \
  /* With explicit bounds */                                                  \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, source_bounds))                                    \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, source_bounds, offset);                                \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                 \
  }                                                                           \
  /* With explicit transform */                                               \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const AffineTransform<M, DIM, T> transform,                             \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    DomainT<M, T> is;                                                         \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,      \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid))            \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, offset);        \
    bounds = AffineBounds::Tester<DIM, T>(is, transform);                     \
  }                                                                           \
  /* With explicit transform and bounds */                                    \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const AffineTransform<M, DIM, T> transform,                             \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    DomainT<M, T> is;                                                         \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,      \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid,             \
            source_bounds))                                                   \
      region.report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, source_bounds,  \
        offset);                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, transform);      \
  }                                                                           \
  /* colocation regions*/                                                     \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, bool silence_warnings = false,                     \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    Rect<DIM, T> bounding_box;                                                \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                             \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (it == start)                                                        \
        bounding_box = is.bounds;                                             \
      else                                                                    \
        bounding_box = bounding_box.union_bbox(is.bounds);                    \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
      ises.emplace_back(is);                                                  \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, bounding_box))                                     \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, bounding_box, offset);                                 \
    DomainT<DIM, T> is;                                                       \
    /* The bounds are the union of the ises (need to be precise) */           \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(   \
        ises, is, Realm::ProfilingRequestSet()));                             \
    /* Defer delete the bounds when the task is done */                       \
    is.destroy(Processor::get_current_finish_event());                        \
    /* Make sure the bounds are ready before we return */                     \
    ready.wait();                                                             \
    bounds = AffineBounds::Tester<DIM, T>(is);                                \
  }                                                                           \
  /* colocation regions with explicit bounds */                               \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const Rect<DIM, T> source_bounds,                  \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                             \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
      ises.emplace_back(is);                                                  \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, fid, source_bounds))                                    \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, fid, source_bounds, offset);                                \
    DomainT<DIM, T> is;                                                       \
    /* The bounds are the union of the ises (need to be precise) */           \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(   \
        ises, is, Realm::ProfilingRequestSet()));                             \
    /* Defer delete the bounds when the task is done */                       \
    is.destroy(Processor::get_current_finish_event());                        \
    /* Make sure the bounds are ready before we return */                     \
    ready.wait();                                                             \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                 \
  }                                                                           \
  /* colocation regions with explicit transform */                            \
  template<typename InputIterator, int M>                                     \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const AffineTransform<M, DIM, T> transform,        \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                             \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<M, T> is;                                                       \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,    \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
      ises.emplace_back(is);                                                  \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid))            \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, offset);        \
    DomainT<DIM, T> is;                                                       \
    /* The bounds are the union of the ises (need to be precise) */           \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(   \
        ises, is, Realm::ProfilingRequestSet()));                             \
    /* Defer delete the bounds when the task is done */                       \
    is.destroy(Processor::get_current_finish_event());                        \
    /* Make sure the bounds are ready before we return */                     \
    ready.wait();                                                             \
    bounds = AffineBounds::Tester<DIM, T>(is, transform);                     \
  }                                                                           \
  /* colocation regions with explicit transform and bounds */                 \
  template<typename InputIterator, int M>                                     \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const AffineTransform<M, DIM, T> transform,        \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions("AffineAccessor", fid, true);  \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                             \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<M, T> is;                                                       \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,    \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "AffineAccessor", fid, instance, inst, *start, true);             \
      else                                                                    \
        instance = inst;                                                      \
      ises.emplace_back(is);                                                  \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, fid,             \
            source_bounds))                                                   \
      start->report_incompatible_accessor("AffineAccessor", instance, fid);   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, fid, source_bounds,  \
        offset);                                                              \
    DomainT<DIM, T> is;                                                       \
    /* The bounds are the union of the ises (need to be precise) */           \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(   \
        ises, is, Realm::ProfilingRequestSet()));                             \
    /* Defer delete the bounds when the task is done */                       \
    is.destroy(Processor::get_current_finish_event());                        \
    /* Make sure the bounds are ready before we return */                     \
    ready.wait();                                                             \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, transform);      \
  }

#define DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(DIM, FIELD_CHECK)        \
  ReductionAccessor(                                                          \
      const UntypedDeferredValue& value, bool silence_warnings = false,       \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    Realm::Rect<DIM, T> source_bounds;                                        \
    /* Anything in range works for these bounds since we're */                \
    /* going to remap them to the origin */                                   \
    for (int i = 0; i < DIM; i++)                                             \
    {                                                                         \
      source_bounds.lo[i] = std::numeric_limits<T>::min();                    \
      source_bounds.hi[i] = std::numeric_limits<T>::max();                    \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
  }                                                                           \
  ReductionAccessor(                                                          \
      const UntypedDeferredValue& value, const Rect<DIM, T>& source_bounds,   \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
  }                                                                           \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer, bool silence_warnings = false,  \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, 0 /*field id*/, is.bounds))                             \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, 0 /*field id*/, is.bounds, offset);                         \
  }                                                                           \
  /* With explicit bounds */                                                  \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const Rect<DIM, T>& source_bounds, bool silence_warnings = false,       \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, 0 /*field id*/, source_bounds))                         \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, 0 /*field id*/, source_bounds, offset);                     \
  }                                                                           \
  /* With explicit transform */                                               \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, 0 /*field id*/)) \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, 0 /*field id*/,      \
        offset);                                                              \
  }                                                                           \
  /* With explicit transform and bounds */                                    \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      const Rect<DIM, T>& source_bounds, bool silence_warnings = false,       \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, 0 /*field id*/,  \
            source_bounds))                                                   \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, 0 /*field id*/,      \
        source_bounds, offset);                                               \
  }

#define DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(             \
    DIM, FIELD_CHECK)                                                         \
  ReductionAccessor(                                                          \
      const UntypedDeferredValue& value, bool silence_warnings = false,       \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    Realm::Rect<DIM, T> source_bounds;                                        \
    /* Anything in range works for these bounds since we're */                \
    /* going to remap them to the origin */                                   \
    for (int i = 0; i < DIM; i++)                                             \
    {                                                                         \
      source_bounds.lo[i] = std::numeric_limits<T>::min();                    \
      source_bounds.hi[i] = std::numeric_limits<T>::max();                    \
    }                                                                         \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
    DomainT<1, T> is;                                                         \
    is.bounds.lo[0] = 0;                                                      \
    is.bounds.hi[0] = 0;                                                      \
    is.sparsity.id = 0;                                                       \
    AffineTransform<1, DIM, T> affine(transform, origin);                     \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, affine);         \
  }                                                                           \
  ReductionAccessor(                                                          \
      const UntypedDeferredValue& value, const Rect<DIM, T>& source_bounds,   \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    assert(!check_field_size || (actual_field_size == value.field_size()));   \
    const Realm::RegionInstance instance = value.instance;                    \
    /* This mapping ignores the input points and sends */                     \
    /* everything to the 1-D origin */                                        \
    Realm::Matrix<1, DIM, T> transform;                                       \
    for (int i = 0; i < DIM; i++) transform[0][i] = 0;                        \
    Realm::Point<1, T> origin(0);                                             \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform, origin, 0 /*field id*/, source_bounds))      \
      value.report_incompatible_accessor("AffineAccessor");                   \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform, origin, 0 /*field id*/, source_bounds, offset);  \
    DomainT<1, T> is;                                                         \
    is.bounds.lo[0] = 0;                                                      \
    is.bounds.hi[0] = 0;                                                      \
    is.sparsity.id = 0;                                                       \
    AffineTransform<1, DIM, T> affine(transform, origin);                     \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, affine);         \
  }                                                                           \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer, bool silence_warnings = false,  \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, 0 /*field id*/, is.bounds))                             \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, 0 /*field id*/, is.bounds, offset);                         \
    bounds = AffineBounds::Tester<DIM, T>(is);                                \
  }                                                                           \
  /* With explicit bounds */                                                  \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const Rect<DIM, T>& source_bounds, bool silence_warnings = false,       \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<DIM, T> is = instance.get_indexspace<DIM, T>();             \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, 0 /*field id*/, source_bounds))                         \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, 0 /*field id*/, source_bounds, offset);                     \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                 \
  }                                                                           \
  /* With explicit transform */                                               \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, 0 /*field id*/)) \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, 0 /*field id*/,      \
        offset);                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is, transform);                     \
  }                                                                           \
  /* With explicit transform and bounds */                                    \
  template<int M>                                                             \
  ReductionAccessor(                                                          \
      const UntypedDeferredBuffer<T>& buffer,                                 \
      const AffineTransform<M, DIM, T>& transform,                            \
      const Rect<DIM, T>& source_bounds, bool silence_warnings = false,       \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    const Realm::RegionInstance instance = buffer.instance;                   \
    const DomainT<M, T> is = instance.get_indexspace<M, T>();                 \
    if (!Realm::AffineAccessor<typename REDOP::RHS, DIM, T>::is_compatible(   \
            instance, transform.transform, transform.offset, 0 /*field id*/,  \
            source_bounds))                                                   \
      UntypedDeferredValue::report_incompatible_accessor(                     \
          "AffineAccessor", true);                                            \
    accessor = Realm::AffineAccessor<typename REDOP::RHS, DIM, T>(            \
        instance, transform.transform, transform.offset, 0 /*field id*/,      \
        source_bounds, offset);                                               \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds, transform);      \
  }

  // Reduce FieldAccessor specialization
  template<typename REDOP, bool EXCLUSIVE, int N, typename T, bool CB>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, N, T, Realm::AffineAccessor<typename REDOP::RHS, N, T>,
      CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(N, true)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(N, false)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<N, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<N, T>& p) const
    {
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::AffineAccessor<typename REDOP::RHS, N, T>, CB>,
        typename REDOP::RHS, N, T>
        operator[](const Point<N, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::AffineAccessor<typename REDOP::RHS, N, T>, CB>,
          typename REDOP::RHS, N, T>(*this, p);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::AffineAccessor<typename REDOP::RHS, N, T>, CB>,
        typename REDOP::RHS, N, T, 2, LEGION_REDUCE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::AffineAccessor<typename REDOP::RHS, N, T>, CB>,
          typename REDOP::RHS, N, T, 2, LEGION_REDUCE>(
          *this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<typename REDOP::RHS, N, T> accessor;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = N;
  };

  // Reduce ReductionAccessor specialization with bounds checks
  template<typename REDOP, bool EXCLUSIVE, int N, typename T>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, N, T, Realm::AffineAccessor<typename REDOP::RHS, N, T>,
      true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(N, true)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(N, false)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<N, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
#endif
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::AffineAccessor<typename REDOP::RHS, N, T>, true>,
        typename REDOP::RHS, N, T>
        operator[](const Point<N, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::AffineAccessor<typename REDOP::RHS, N, T>, true>,
          typename REDOP::RHS, N, T>(*this, p);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::AffineAccessor<typename REDOP::RHS, N, T>, true>,
        typename REDOP::RHS, N, T, 2, LEGION_REDUCE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::AffineAccessor<typename REDOP::RHS, N, T>, true>,
          typename REDOP::RHS, N, T, 2, LEGION_REDUCE>(
          *this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<typename REDOP::RHS, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = N;
  };

  // Reduce Field Accessor specialization with N==1
  // to avoid array ambiguity
  template<typename REDOP, bool EXCLUSIVE, typename T, bool CB>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, 1, T, Realm::AffineAccessor<typename REDOP::RHS, 1, T>,
      CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(1, true)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(1, false)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<1, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<1, T>& p) const
    {
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, 1, T,
            Realm::AffineAccessor<typename REDOP::RHS, 1, T>, CB>,
        typename REDOP::RHS, 1, T>
        operator[](const Point<1, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, 1, T,
              Realm::AffineAccessor<typename REDOP::RHS, 1, T>, CB>,
          typename REDOP::RHS, 1, T>(*this, p);
    }
  public:
    Realm::AffineAccessor<typename REDOP::RHS, 1, T> accessor;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = 1;
  };

  // Reduce Field Accessor specialization with N==1
  // to avoid array ambiguity and bounds checks
  template<typename REDOP, bool EXCLUSIVE, typename T>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, 1, T, Realm::AffineAccessor<typename REDOP::RHS, 1, T>,
      true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(1, true)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(1, false)
    DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<1, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
#endif
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, 1, T,
            Realm::AffineAccessor<typename REDOP::RHS, 1, T>, true>,
        typename REDOP::RHS, 1, T>
        operator[](const Point<1, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, 1, T,
              Realm::AffineAccessor<typename REDOP::RHS, 1, T>, true>,
          typename REDOP::RHS, 1, T>(*this, p);
    }
  public:
    Realm::AffineAccessor<typename REDOP::RHS, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = 1;
  };

#undef PHYSICAL_REGION_CONSTRUCTORS
#undef PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS

  ////////////////////////////////////////////////////////////
  // Specializations for Multi Affine Accessors
  ////////////////////////////////////////////////////////////

#define PHYSICAL_REGION_CONSTRUCTORS(PRIVILEGE, DIM, FIELD_CHECK)              \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(                \
            instance, fid, is.bounds))                                         \
      region.report_incompatible_accessor(                                     \
          "MultiAffineAccessor", instance, fid);                               \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(                         \
        instance, fid, is.bounds, offset);                                     \
  }                                                                            \
  /* With explicit bounds */                                                   \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(                \
            instance, fid, source_bounds))                                     \
      region.report_incompatible_accessor(                                     \
          "MultiAffineAccessor", instance, fid);                               \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(                         \
        instance, fid, source_bounds, offset);                                 \
  }                                                                            \
  /* colocation regions */                                                     \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("MultiAffineAccessor", fid);    \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "MultiAffineAccessor", fid, instance, inst, *start);               \
      else                                                                     \
        instance = inst;                                                       \
      if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(              \
              instance, fid, is.bounds))                                       \
        it->report_incompatible_accessor(                                      \
            "MultiAffineAccessor", instance, fid);                             \
    }                                                                          \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(instance, fid, offset);  \
  }                                                                            \
  /* colocation regions with explicit bounds */                                \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("MultiAffineAccessor", fid);    \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "MultiAffineAccessor", fid, instance, inst, *start);               \
      else                                                                     \
        instance = inst;                                                       \
      if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(              \
              instance, fid, source_bounds))                                   \
        it->report_incompatible_accessor(                                      \
            "MultiAffineAccessor", instance, fid);                             \
    }                                                                          \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(                         \
        instance, fid, source_bounds, offset);                                 \
  }

#define PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(PRIVILEGE, DIM, FIELD_CHECK)  \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(                \
            instance, fid, is.bounds))                                         \
      region.report_incompatible_accessor(                                     \
          "MultiAffineAccessor", instance, fid);                               \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(                         \
        instance, fid, is.bounds, offset);                                     \
    bounds = AffineBounds::Tester<DIM, T>(is);                                 \
  }                                                                            \
  /* With explicit bounds */                                                   \
  FieldAccessor(                                                               \
      const PhysicalRegion& region, FieldID fid,                               \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    DomainT<DIM, T> is;                                                        \
    const Realm::RegionInstance instance = region.get_instance_info(           \
        PRIVILEGE, fid, actual_field_size, &is,                                \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,     \
        silence_warnings, false /*generic accessor*/, check_field_size);       \
    if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(                \
            instance, fid, source_bounds))                                     \
      region.report_incompatible_accessor(                                     \
          "MultiAffineAccessor", instance, fid);                               \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(                         \
        instance, fid, source_bounds, offset);                                 \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                  \
  }                                                                            \
  /* colocation regions */                                                     \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      size_t actual_field_size = sizeof(FT),                                   \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("MultiAffineAccessor", fid);    \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "MultiAffineAccessor", fid, instance, inst, *start);               \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
      if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(              \
              instance, fid, is.bounds))                                       \
        it->report_incompatible_accessor(                                      \
            "MultiAffineAccessor", instance, fid);                             \
    }                                                                          \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(instance, fid, offset);  \
    DomainT<DIM, T> is;                                                        \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, is, Realm::ProfilingRequestSet()));                              \
    /* Defer delete the bounds when the task is done */                        \
    bounds.destroy(Processor::get_current_finish_event());                     \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is);                                 \
  }                                                                            \
  /* colocation with explicit bounds */                                        \
  template<typename InputIterator>                                             \
  FieldAccessor(                                                               \
      InputIterator start, InputIterator stop, FieldID fid,                    \
      const Rect<DIM, T> source_bounds, size_t actual_field_size = sizeof(FT), \
      bool check_field_size = FIELD_CHECK, bool silence_warnings = false,      \
      const char* warning_string = nullptr, size_t offset = 0)                 \
    : field(fid)                                                               \
  {                                                                            \
    static_assert(                                                             \
        std::is_same<                                                          \
            PhysicalRegion,                                                    \
            typename std::iterator_traits<InputIterator>::value_type>::value,  \
        "Input Iterators to FieldAccessors must be for PhysicalRegions");      \
    if (start == stop)                                                         \
      PhysicalRegion::empty_colocation_regions("MultiAffineAccessor", fid);    \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                              \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;           \
    for (InputIterator it = start; it != stop; it++)                           \
    {                                                                          \
      DomainT<DIM, T> is;                                                      \
      const Realm::RegionInstance inst = it->get_instance_info(                \
          PRIVILEGE, fid, actual_field_size, &is,                              \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,   \
          silence_warnings, false /*generic accessor*/, check_field_size);     \
      if (instance.exists() && (inst != instance))                             \
        it->report_colocation_violation(                                       \
            "MultiAffineAccessor", fid, instance, inst, *start);               \
      else                                                                     \
        instance = inst;                                                       \
      ises.emplace_back(is);                                                   \
      if (!Realm::MultiAffineAccessor<FT, DIM, T>::is_compatible(              \
              instance, fid, source_bounds))                                   \
        it->report_incompatible_accessor(                                      \
            "MultiAffineAccessor", instance, fid);                             \
    }                                                                          \
    accessor = Realm::MultiAffineAccessor<FT, DIM, T>(                         \
        instance, fid, source_bounds, offset);                                 \
    DomainT<DIM, T> is;                                                        \
    /* The bounds are the union of the ises (need to be precise) */            \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(    \
        ises, is, Realm::ProfilingRequestSet()));                              \
    /* Defer delete the bounds when the task is done */                        \
    bounds.destroy(Processor::get_current_finish_event());                     \
    /* Make sure the bounds are ready before we return */                      \
    ready.wait();                                                              \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                  \
  }

  // Read-only FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<N, T>& p) const
    {
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
            CB>,
        FT, N, T, 2, LEGION_READ_ONLY>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
              CB>,
          FT, N, T, 2, LEGION_READ_ONLY>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-only FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.prt(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
            true>,
        FT, N, T, 2, LEGION_READ_ONLY>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_READ_ONLY>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-only FieldAccessor specialization
  // with N==1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_ONLY, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<1, T>& p) const
    {
      return accessor[p];
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-only FieldAccessor specialization
  // with N==1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_READ_ONLY, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_ONLY, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_ONLY);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline const FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor[p];
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-write FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_WRITE, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
            CB>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_WRITE, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
              CB>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-write FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_READ_WRITE, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
            true>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_READ_WRITE, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
              true>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Read-write FieldAccessor specialization
  // with N==1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_READ_WRITE, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Read-write FieldAccessor specialization
  // with N==1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_READ_WRITE, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_READ_WRITE, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-discard FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
      CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T,
            Realm::MultiAffineAccessor<FT, N, T>, CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T,
              Realm::MultiAffineAccessor<FT, N, T>, CB>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-discard FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>,
      true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T,
            Realm::MultiAffineAccessor<FT, N, T>, true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T,
              Realm::MultiAffineAccessor<FT, N, T>, true>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-discard FieldAccessor specialization with
  // N == 1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>,
      CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-discard FieldAccessor specialization with
  // N == 1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_WRITE_DISCARD, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>,
      true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_ONLY);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-only FieldAccessor specialization
  template<typename FT, int N, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T,
            Realm::MultiAffineAccessor<FT, N, T>, CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T,
              Realm::MultiAffineAccessor<FT, N, T>, CB>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-only FieldAccessor specialization
  // with bounds checks
  template<typename FT, int N, typename T>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<
            LEGION_WRITE_DISCARD, FT, N, T,
            Realm::MultiAffineAccessor<FT, N, T>, true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<
              LEGION_WRITE_DISCARD, FT, N, T,
              Realm::MultiAffineAccessor<FT, N, T>, true>,
          FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Write-only FieldAccessor specialization with
  // N == 1 to avoid array ambiguity
  template<typename FT, typename T, bool CB>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(LEGION_WRITE_DISCARD, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Write-only FieldAccessor specialization with
  // N == 1 to avoid array ambiguity and bounds checks
  template<typename FT, typename T>
  class FieldAccessor<
      LEGION_WRITE_ONLY, FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    FieldAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(LEGION_WRITE_DISCARD, 1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_WRITE_DISCARD);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[1];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_READ_WRITE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, LEGION_READ_WRITE);
#endif
      return accessor[p];
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

#undef PHYSICAL_REGION_CONSTRUCTORS
#undef PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS

#define PHYSICAL_REGION_CONSTRUCTORS(DIM, FIELD_CHECK)                        \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::            \
            is_compatible(instance, fid, is.bounds))                          \
      region.report_incompatible_accessor(                                    \
          "MultiAffineAccessor", instance, fid);                              \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, is.bounds, offset);                                    \
  }                                                                           \
  /* With explicit bounds */                                                  \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::            \
            is_compatible(instance, fid, source_bounds))                      \
      region.report_incompatible_accessor(                                    \
          "MultiAffineAccessor", instance, fid);                              \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, source_bounds, offset);                                \
  }                                                                           \
  /* colocation regions */                                                    \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, bool silence_warnings = false,                     \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions(                               \
          "MultiAffineAccessor", fid, true);                                  \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "MultiAffineAccessor", fid, instance, inst, *start, true);        \
      else                                                                    \
        instance = inst;                                                      \
      if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::          \
              is_compatible(instance, fid, is.bounds))                        \
        it->report_incompatible_accessor(                                     \
            "MultiAffineAccessor", instance, fid);                            \
    }                                                                         \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, offset);                                               \
  }                                                                           \
  /* colocation with explicit bounds */                                       \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const Rect<DIM, T> source_bounds,                  \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions(                               \
          "MultiAffineAccessor", fid, true);                                  \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "MultiAffineAccessor", fid, instance, inst, *start, true);        \
      else                                                                    \
        instance = inst;                                                      \
      if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::          \
              is_compatible(instance, fid, source_bounds))                    \
        it->report_incompatible_accessor(                                     \
            "MultiAffineAccessor", instance, fid);                            \
    }                                                                         \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, source_bounds, offset);                                \
  }

#define PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(DIM, FIELD_CHECK)            \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = false)                                          \
    : field(fid)                                                              \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::            \
            is_compatible(instance, fid, is.bounds))                          \
      region.report_incompatible_accessor(                                    \
          "MultiAffineAccessor", instance, fid);                              \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, is.bounds, offset);                                    \
    bounds = AffineBounds::Tester<DIM, T>(is);                                \
  }                                                                           \
  /* With explicit bounds */                                                  \
  ReductionAccessor(                                                          \
      const PhysicalRegion& region, FieldID fid, ReductionOpID redop,         \
      const Rect<DIM, T> source_bounds, bool silence_warnings = false,        \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    DomainT<DIM, T> is;                                                       \
    const Realm::RegionInstance instance = region.get_instance_info(          \
        LEGION_REDUCE, fid, actual_field_size, &is,                           \
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,    \
        silence_warnings, false /*generic accessor*/, check_field_size,       \
        redop);                                                               \
    if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::            \
            is_compatible(instance, fid, source_bounds))                      \
      region.report_incompatible_accessor(                                    \
          "MultiAffineAccessor", instance, fid);                              \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, source_bounds, offset);                                \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                 \
  }                                                                           \
  /* colocation regions */                                                    \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, bool silence_warnings = false,                     \
      const char* warning_string = nullptr, size_t offset = 0,                \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = false)                                          \
    : field(fid)                                                              \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions(                               \
          "MultiAffineAccessor", fid, true);                                  \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                             \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "MultiAffineAccessor", fid, instance, inst, *start, true);        \
      else                                                                    \
        instance = inst;                                                      \
      if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::          \
              is_compatible(instance, fid, is.bounds))                        \
        it->report_incompatible_accessor(                                     \
            "MultiAffineAccessor", instance, fid);                            \
    }                                                                         \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, offset);                                               \
    DomainT<DIM, T> is;                                                       \
    /* The bounds are the union of the ises (need to be precise) */           \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(   \
        ises, is, Realm::ProfilingRequestSet()));                             \
    /* Defer delete the bounds when the task is done */                       \
    is.destroy(Processor::get_current_finish_event());                        \
    /* Make sure the bounds are ready before we return */                     \
    ready.wait();                                                             \
    bounds = AffineBounds::Tester<DIM, T>(is);                                \
  }                                                                           \
  /* colocation regions with explicit bounds */                               \
  template<typename InputIterator>                                            \
  ReductionAccessor(                                                          \
      InputIterator start, InputIterator stop, FieldID fid,                   \
      ReductionOpID redop, const Rect<DIM, T> source_bounds,                  \
      bool silence_warnings = false, const char* warning_string = nullptr,    \
      size_t offset = 0,                                                      \
      size_t actual_field_size = sizeof(typename REDOP::RHS),                 \
      bool check_field_size = FIELD_CHECK)                                    \
    : field(fid)                                                              \
  {                                                                           \
    static_assert(                                                            \
        std::is_same<                                                         \
            PhysicalRegion,                                                   \
            typename std::iterator_traits<InputIterator>::value_type>::value, \
        "Input Iterators to ReductionAccessors must be for PhysicalRegions"); \
    if (start == stop)                                                        \
      PhysicalRegion::empty_colocation_regions(                               \
          "MultiAffineAccessor", fid, true);                                  \
    std::vector<Realm::IndexSpace<DIM, T> > ises;                             \
    Realm::RegionInstance instance = Realm::RegionInstance::NO_INST;          \
    for (InputIterator it = start; it != stop; it++)                          \
    {                                                                         \
      DomainT<DIM, T> is;                                                     \
      const Realm::RegionInstance inst = it->get_instance_info(               \
          LEGION_REDUCE, fid, actual_field_size, &is,                         \
          Internal::NT_TemplateHelper::encode_tag<DIM, T>(), warning_string,  \
          silence_warnings, false /*generic accessor*/, check_field_size,     \
          redop);                                                             \
      if (instance.exists() && (inst != instance))                            \
        it->report_colocation_violation(                                      \
            "MultiAffineAccessor", fid, instance, inst, *start, true);        \
      else                                                                    \
        instance = inst;                                                      \
      if (!Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>::          \
              is_compatible(instance, fid, source_bounds))                    \
        it->report_incompatible_accessor(                                     \
            "MultiAffineAccessor", instance, fid);                            \
    }                                                                         \
    accessor = Realm::MultiAffineAccessor<typename REDOP::RHS, DIM, T>(       \
        instance, fid, source_bounds, offset);                                \
    DomainT<DIM, T> is;                                                       \
    /* The bounds are the union of the ises (need to be precise) */           \
    const Internal::LgEvent ready(Realm::IndexSpace<DIM, T>::compute_union(   \
        ises, is, Realm::ProfilingRequestSet()));                             \
    /* Defer delete the bounds when the task is done */                       \
    is.destroy(Processor::get_current_finish_event());                        \
    /* Make sure the bounds are ready before we return */                     \
    ready.wait();                                                             \
    bounds = AffineBounds::Tester<DIM, T>(is, source_bounds);                 \
  }

  // Reduce FieldAccessor specialization
  template<typename REDOP, bool EXCLUSIVE, int N, typename T, bool CB>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, N, T,
      Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<N, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<N, T>& p) const
    {
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      size_t strides[N];
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, CB>,
        typename REDOP::RHS, N, T>
        operator[](const Point<N, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, CB>,
          typename REDOP::RHS, N, T>(*this, p);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, CB>,
        typename REDOP::RHS, N, T, 2, LEGION_REDUCE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, CB>,
          typename REDOP::RHS, N, T, 2, LEGION_REDUCE>(
          *this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<typename REDOP::RHS, N, T> accessor;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = N;
  };

  // Reduce ReductionAccessor specialization with bounds checks
  template<typename REDOP, bool EXCLUSIVE, int N, typename T>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, N, T,
      Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<N, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      size_t strides[N];
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, true>,
        typename REDOP::RHS, N, T>
        operator[](const Point<N, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, true>,
          typename REDOP::RHS, N, T>(*this, p);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, N, T,
            Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, true>,
        typename REDOP::RHS, N, T, 2, LEGION_REDUCE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, N, T,
              Realm::MultiAffineAccessor<typename REDOP::RHS, N, T>, true>,
          typename REDOP::RHS, N, T, 2, LEGION_REDUCE>(
          *this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<typename REDOP::RHS, N, T> accessor;
    FieldID field;
    AffineBounds::Tester<N, T> bounds;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = N;
  };

  // Reduce Field Accessor specialization with N==1
  // to avoid array ambiguity
  template<typename REDOP, bool EXCLUSIVE, typename T, bool CB>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, 1, T,
      Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    __LEGION_CUDA_HD__
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<1, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<1, T>& p) const
    {
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      size_t strides[1];
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, 1, T,
            Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T>, CB>,
        typename REDOP::RHS, 1, T>
        operator[](const Point<1, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, 1, T,
              Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T>, CB>,
          typename REDOP::RHS, 1, T>(*this, p);
    }
  public:
    mutable Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T> accessor;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = 1;
  };

  // Reduce Field Accessor specialization with N==1
  // to avoid array ambiguity and bounds checks
  template<typename REDOP, bool EXCLUSIVE, typename T>
  class ReductionAccessor<
      REDOP, EXCLUSIVE, 1, T,
      Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    // No CUDA support due to PhysicalRegion constructor
    ReductionAccessor(void) { }
#ifdef LEGION_DEBUG
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
    PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
  public:
    __LEGION_CUDA_HD__
    inline void reduce(const Point<1, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      REDOP::template fold<EXCLUSIVE>(accessor[p], val);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains(p));
#else
      if (!bounds.contains(p))
        PhysicalRegion::fail_bounds_check(DomainPoint(p), field, LEGION_REDUCE);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r,
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      size_t strides[1];
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline typename REDOP::RHS* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(typename REDOP::RHS)) const
    {
      typename REDOP::RHS* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.contains_all(r));
      assert(result != nullptr);
#else
      if (!bounds.contains_all(r))
        PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      strides[0] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::ReductionHelper<
        ReductionAccessor<
            REDOP, EXCLUSIVE, 1, T,
            Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T>, true>,
        typename REDOP::RHS, 1, T>
        operator[](const Point<1, T>& p) const
    {
      return ArraySyntax::ReductionHelper<
          ReductionAccessor<
              REDOP, EXCLUSIVE, 1, T,
              Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T>, true>,
          typename REDOP::RHS, 1, T>(*this, p);
    }
  public:
    mutable Realm::MultiAffineAccessor<typename REDOP::RHS, 1, T> accessor;
    FieldID field;
    AffineBounds::Tester<1, T> bounds;
  public:
    typedef typename REDOP::RHS value_type;
    typedef typename REDOP::RHS& reference;
    typedef const typename REDOP::RHS& const_reference;
    static const int dim = 1;
  };

#undef PHYSICAL_REGION_CONSTRUCTORS
#undef PHYSICAL_REGION_CONSTRUCTORS_WITH_BOUNDS

  ////////////////////////////////////////////////////////////
  // Padding Accessor with Generic Accessors
  ////////////////////////////////////////////////////////////

  // Padding Accessor, generic, N, no bounds checks
  template<typename FT, int N, typename T, bool CB>
  class PaddingAccessor<FT, N, T, Realm::GenericAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      Domain outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, nullptr /*inner*/, outer_bounds,
          warning_string, silence_warnings, true /*generic*/, check_field_size);
      const Rect<N, T> bounds = outer_bounds;
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<N, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        PaddingAccessor<FT, N, T, Realm::GenericAccessor<FT, N, T>, CB>, FT, N,
        T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          PaddingAccessor<FT, N, T, Realm::GenericAccessor<FT, N, T>, CB>, FT,
          N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Padding Accessor, generic, N, bounds checks
  template<typename FT, int N, typename T>
  class PaddingAccessor<FT, N, T, Realm::GenericAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      Domain inner_bounds, outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, &inner_bounds, outer_bounds, warning_string,
          silence_warnings, true /*generic*/, check_field_size);
      inner = inner_bounds;
      outer = outer_bounds;
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, outer))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor = Realm::GenericAccessor<FT, N, T>(instance, fid, outer, offset);
    }
  public:
    inline FT read(const Point<N, T>& p) const
    {
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
      return accessor.read(p);
    }
    inline void write(const Point<N, T>& p, FT val) const
    {
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<N, T>& p) const
    {
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        PaddingAccessor<FT, N, T, Realm::GenericAccessor<FT, N, T>, true>, FT,
        N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          PaddingAccessor<FT, N, T, Realm::GenericAccessor<FT, N, T>, true>, FT,
          N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
    Rect<N, T> inner, outer;
    FieldID field;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Padding Accessor, generic, 1, no bounds checks
  template<typename FT, typename T, bool CB>
  class PaddingAccessor<FT, 1, T, Realm::GenericAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      Domain outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, nullptr /*inner*/, outer_bounds,
          warning_string, silence_warnings, true /*generic*/, check_field_size);
      const Rect<1, T> bounds = outer_bounds;
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
  public:
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<1, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Padding Accessor, generic, 1, bounds checks
  template<typename FT, typename T>
  class PaddingAccessor<FT, 1, T, Realm::GenericAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      Domain inner_bounds, outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, &inner_bounds, outer_bounds, warning_string,
          silence_warnings, true /*generic*/, check_field_size);
      inner = inner_bounds;
      outer = outer_bounds;
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, outer))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor = Realm::GenericAccessor<FT, 1, T>(instance, fid, outer, offset);
    }
  public:
    inline FT read(const Point<1, T>& p) const
    {
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
      return accessor.read(p);
    }
    inline void write(const Point<1, T>& p, FT val) const
    {
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<1, T>& p) const
    {
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
    Rect<1, T> inner, outer;
    FieldID field;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  ////////////////////////////////////////////////////////////
  // Padding Accessor with Affine Accessors
  ////////////////////////////////////////////////////////////

  // Padding Accessor, affine, N, no bounds checks
  template<typename FT, int N, typename T, bool CB>
  class PaddingAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      Domain outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, nullptr /*inner*/, outer_bounds,
          warning_string, silence_warnings, false /*generic*/,
          check_field_size);
      const Rect<N, T> bounds = outer_bounds;
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        PaddingAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>, FT, N,
        T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          PaddingAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>, FT, N,
          T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Padding Accessor, affine, N, bounds checks
  template<typename FT, int N, typename T>
  class PaddingAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T>, true> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      Domain inner_bounds, outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, &inner_bounds, outer_bounds, warning_string,
          silence_warnings, false /*generic*/, check_field_size);
      inner = inner_bounds;
      outer = outer_bounds;
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(instance, fid, outer))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, outer, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      const Rect<N, T> contained = outer.intersection(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(contained.volume() == r.volume());
#else
      if (contained.volume() < r.volume())
      {
        if (outer.contains(r.lo))
          PhysicalRegion::fail_padding_check(DomainPoint(r.lo), field);
        else
          PhysicalRegion::fail_padding_check(DomainPoint(r.hi), field);
      }
#endif
      const Rect<N, T> overlap = inner.overlaps(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(overlap.empty());
#else
      if (!overlap.empty())
        PhysicalRegion::fail_padding_check(DomainPoint(overlap.lo), field);
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      const Rect<N, T> contained = outer.intersection(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(contained.volume() == r.volume());
#else
      if (contained.volume() < r.volume())
      {
        if (outer.contains(r.lo))
          PhysicalRegion::fail_padding_check(DomainPoint(r.lo), field);
        else
          PhysicalRegion::fail_padding_check(DomainPoint(r.hi), field);
      }
#endif
      const Rect<N, T> overlap = inner.overlaps(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(overlap.empty());
#else
      if (!overlap.empty())
        PhysicalRegion::fail_padding_check(DomainPoint(overlap.lo), field);
#endif
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        PaddingAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T>, true>, FT, N,
        T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          PaddingAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T>, true>, FT,
          N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
    Rect<N, T> inner, outer;
    FieldID field;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Padding Accessor, affine, 1, no bounds checks
  template<typename FT, typename T, bool CB>
  class PaddingAccessor<FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      Domain outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, nullptr /*inner*/, outer_bounds,
          warning_string, silence_warnings, false /*generic*/,
          check_field_size);
      const Rect<1, T> bounds = outer_bounds;
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Padding Accessor, affine, 1, bounds checks
  template<typename FT, typename T>
  class PaddingAccessor<FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    PaddingAccessor(void) { }
    PaddingAccessor(
        const PhysicalRegion& region, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      Domain inner_bounds, outer_bounds;
      const Realm::RegionInstance instance = region.get_padding_info(
          fid, actual_field_size, &inner_bounds, outer_bounds, warning_string,
          silence_warnings, true /*generic*/, check_field_size);
      inner = inner_bounds;
      outer = outer_bounds;
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(instance, fid, outer))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, outer, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      return accessor.ptr(p);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
      const Rect<1, T> contained = outer.intersection(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(contained.volume() == r.volume());
#else
      if (contained.volume() < r.volume())
      {
        if (outer.contains(r.lo))
          PhysicalRegion::fail_padding_check(DomainPoint(r.lo), field);
        else
          PhysicalRegion::fail_padding_check(DomainPoint(r.hi), field);
      }
#endif
      const Rect<1, T> overlap = inner.overlaps(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(overlap.empty());
#else
      if (!overlap.empty())
        PhysicalRegion::fail_padding_check(DomainPoint(overlap.lo), field);
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      const Rect<1, T> contained = outer.intersection(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(contained.volume() == r.volume());
#else
      if (contained.volume() < r.volume())
      {
        if (outer.contains(r.lo))
          PhysicalRegion::fail_padding_check(DomainPoint(r.lo), field);
        else
          PhysicalRegion::fail_padding_check(DomainPoint(r.hi), field);
      }
#endif
      const Rect<1, T> overlap = inner.overlaps(r);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(overlap.empty());
#else
      if (!overlap.empty())
        PhysicalRegion::fail_padding_check(DomainPoint(overlap.lo), field);
#endif
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      return accessor[p];
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(outer.contains(p) && !inner.contains(p));
#else
      if (!outer.contains(p) || inner.contains(p))
        PhysicalRegion::fail_padding_check(DomainPoint(p), field);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
    Rect<1, T> inner, outer;
    FieldID field;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  ////////////////////////////////////////////////////////////
  // Multi Region Accessor with Generic Accessors
  ////////////////////////////////////////////////////////////
#ifdef LEGION_MULTI_REGION_ACCESSOR
  // Multi-Accessor, generic, N, bounds checks and/or privilege checks
  template<typename FT, int N, typename T, bool CB, bool CP, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::GenericAccessor<FT, N, T>, CB, CP, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = region_bounds[0].bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
        idx++;
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = region_bounds[0].bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx].bounds =
            source_bounds.intersection(region_bounds[idx].bounds);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
        idx++;
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      assert(regions.size() <= MR);
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = region_bounds[0].bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      assert(regions.size() <= MR);
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      region_bounds[0].bounds =
          source_bounds.intersection(region_bounds[0].bounds);
      Rect<N, T> bounds = region_bounds[0].bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx].bounds =
            source_bounds.intersection(region_bounds[idx].bounds);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    inline FT read(const Point<N, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if (CP && ((region_privileges[idx] & LEGION_READ_ONLY) == 0))
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
        found = true;
        break;
      }
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
      return accessor.read(p);
    }
    inline void write(const Point<N, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if (CP && ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0))
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
        found = true;
        break;
      }
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
      return accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_NO_ACCESS> operator[](
        const Point<N, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_NO_ACCESS>(
          accessor[p], field, DomainPoint(p), region_privileges[index]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::GenericAccessor<FT, N, T>, CB, CP, MR>,
        FT, N, T, 2, LEGION_NO_ACCESS>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::GenericAccessor<FT, N, T>, CB, CP, MR>,
          FT, N, T, 2, LEGION_NO_ACCESS>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    DomainT<N, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, generic, 1, bounds checks and/or privilege checks
  template<typename FT, typename T, bool CB, bool CP, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::GenericAccessor<FT, 1, T>, CB, CP, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = region_bounds[0].bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
        idx++;
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = region_bounds[0].bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx].bounds =
            source_bounds.intersection(region_bounds[idx].bounds);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
        idx++;
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      assert(regions.size() <= MR);
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = region_bounds[0].bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      assert(regions.size() <= MR);
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      region_bounds[0].bounds =
          source_bounds.intersection(region_bounds[0].bounds);
      Rect<1, T> bounds = region_bounds[0].bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx].bounds =
            source_bounds.intersection(region_bounds[idx].bounds);
        bounds = bounds.union_bbox(region_bounds[idx].bounds);
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
  public:
    inline FT read(const Point<1, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if (CP && ((region_privileges[idx] & LEGION_READ_ONLY) == 0))
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
        found = true;
        break;
      }
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
      return accessor.read(p);
    }
    inline void write(const Point<1, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if (CP && ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0))
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
        found = true;
        break;
      }
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
      return accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_NO_ACCESS> operator[](
        const Point<1, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
      return ArraySyntax::AccessorRefHelper<FT, LEGION_NO_ACCESS>(
          accessor[p], field, DomainPoint(p), region_privileges[index]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    DomainT<1, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Multi-Accessor, generic, N, no bounds, no privileges
  template<typename FT, int N, typename T, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::GenericAccessor<FT, N, T>, false, false, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          start->get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
        idx++;
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
      }
      if (!Realm::GenericAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    inline void write(const Point<N, T>& p, FT val) const
    {
      return accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<N, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::GenericAccessor<FT, N, T>, false, false, MR>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::GenericAccessor<FT, N, T>, false, false, MR>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::GenericAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, generic, 1, no bounds, no privileges
  template<typename FT, typename T, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::GenericAccessor<FT, 1, T>, false, false, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          start->get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
        idx++;
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, true /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
      }
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
  public:
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    inline void write(const Point<1, T>& p, FT val) const
    {
      return accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<1, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  ////////////////////////////////////////////////////////////
  // Multi Region Accessor with Affine Accessors
  ////////////////////////////////////////////////////////////

  // Multi-Accessor, affine, N, with privilege checks (implies bounds checks)
  template<typename FT, int N, typename T, bool CB, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::AffineAccessor<FT, N, T>, CB, true, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, transform);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, T> transform,
        const Rect<N, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<N, T>(is, source_bounds, transform);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<N, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, transform);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, T> transform,
        const Rect<N, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<N, T>(is, source_bounds, transform);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<N, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_READ_ONLY) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineRefHelper<FT> operator[](
        const Point<N, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return ArraySyntax::AffineRefHelper<FT>(
          accessor[p], field, DomainPoint(p), region_privileges[index]);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::AffineAccessor<FT, N, T>, CB, true, MR>,
        FT, N, T, 2, LEGION_NO_ACCESS>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::AffineAccessor<FT, N, T>, CB, true, MR>,
          FT, N, T, 2, LEGION_NO_ACCESS>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_REDUCE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<N, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, affine, 1, with privilege checks (implies bounds checks)
  template<typename FT, typename T, bool CB, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB, true, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, 1, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, transform);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, 1, T> transform,
        const Rect<1, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<1, T>(is, source_bounds, transform);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<1, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, 1, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, transform);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, 1, T> transform,
        const Rect<1, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<1, T>(is, source_bounds, transform);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<1, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_READ_ONLY) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineRefHelper<FT> operator[](
        const Point<1, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return ArraySyntax::AffineRefHelper<FT>(
          accessor[p], field, DomainPoint(p), region_privileges[index]);
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_REDUCE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<1, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Multi-Accessor, affine, N, bounds checks only
  template<typename FT, int N, typename T, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::AffineAccessor<FT, N, T>, true /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 0;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, transform);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, T> transform,
        const Rect<N, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<N, T>(is, source_bounds, transform);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<N, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, transform);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, T> transform,
        const Rect<N, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<N, T>(is, source_bounds, transform);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<N, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::AffineAccessor<FT, N, T>, true, false, MR>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::AffineAccessor<FT, N, T>, true, false, MR>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<N, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, affine, 1, bounds checks only
  template<typename FT, typename T, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 0;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, 1, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, transform);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, 1, T> transform,
        const Rect<1, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<1, T>(is, source_bounds, transform);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<1, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, 1, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, transform);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, transform);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, 1, T> transform,
        const Rect<1, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] =
          AffineBounds::Tester<1, T>(is, source_bounds, transform);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] =
            AffineBounds::Tester<1, T>(is, transform, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor[p];
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<1, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Multi-Accessor, affine, N, no bounds, no privileges
  template<typename FT, int N, typename T, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::AffineAccessor<FT, N, T>, false /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, N, T> transform,
        const Rect<N, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, N, T> transform,
        const Rect<N, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::AffineAccessor<FT, N, T>, false, false, MR>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::AffineAccessor<FT, N, T>, false, false, MR>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, affine, 1, no bounds, no privileges
  template<typename FT, typename T, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::AffineAccessor<FT, 1, T>, false /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, 1, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds);
    }
    template<int M, typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop,
        const AffineTransform<M, 1, T> transform,
        const Rect<1, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds);
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(instance, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, 1, T> transform, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
    template<int M>
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const AffineTransform<M, 1, T> transform,
        const Rect<1, T> source_bounds, FieldID fid,
        // The actual field size in case it is different from the
        // one being used in FT and we still want to check it
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid, bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  ////////////////////////////////////////////////////////////
  // Multi Region Accessor with Multi Affine Accessors
  ////////////////////////////////////////////////////////////

  // Multi-Accessor, multi affine, N, with privilege checks
  // (implies bounds checks)
  template<typename FT, int N, typename T, bool CB, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, CB, true, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_READ_ONLY) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineRefHelper<FT> operator[](
        const Point<N, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return ArraySyntax::AffineRefHelper<FT>(
          accessor[p], field, DomainPoint(p), region_privileges[index]);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, CB, true, MR>,
        FT, N, T, 2, LEGION_NO_ACCESS>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, CB, true, MR>,
          FT, N, T, 2, LEGION_NO_ACCESS>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_REDUCE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<N, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, multi affine, 1, with privilege checks
  // (implies bounds checks)
  template<typename FT, typename T, bool CB, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, CB, true, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_READ_ONLY) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineRefHelper<FT> operator[](
        const Point<1, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return ArraySyntax::AffineRefHelper<FT>(
          accessor[p], field, DomainPoint(p), region_privileges[index]);
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        if ((region_privileges[idx] & LEGION_REDUCE_PRIV) == 0)
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          // bounds checks are not precise for CUDA so keep going to
          // see if there is another region that has it with the privileges
          continue;
#else
          PhysicalRegion::fail_privilege_check(
              DomainPoint(p), field, region_privileges[idx]);
#endif
        }
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<1, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Multi-Accessor, multi affine, N, bounds checks only
  template<typename FT, int N, typename T, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, true /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 0;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<N, T>(is, source_bounds);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<N, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor[p];
    }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, true, false, MR>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, true, false, MR>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<N, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, multi affine, 1, bounds checks only
  template<typename FT, typename T, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, true /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 0;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
      total_regions = idx;
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        region_privileges[idx] = start->get_privilege();
        const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
      total_regions = idx;
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
      : field(fid), total_regions(regions.size())
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      region_privileges[0] = region.get_privilege();
      const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      region_bounds[0] = AffineBounds::Tester<1, T>(is, source_bounds);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        region_privileges[idx] = regions[idx].get_privilege();
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        region_bounds[idx] = AffineBounds::Tester<1, T>(is, source_bounds);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.read(p);
    }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const
    {
      int index = -1;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        index = idx;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(index >= 0);
#else
      if (index < 0)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      return accessor[p];
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      bool found = false;
      for (unsigned idx = 0; idx < total_regions; idx++)
      {
        if (!region_bounds[idx].contains(p))
          continue;
        found = true;
        break;
      }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(found);
#else
      if (!found)
        PhysicalRegion::fail_bounds_check(
            DomainPoint(p), field, region_privileges[0], true /*multi*/);
#endif
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
    FieldID field;
    PrivilegeMode region_privileges[MR];
    AffineBounds::Tester<1, T> region_bounds[MR];
    unsigned total_regions;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  // Multi-Accessor, multi affine, N, no bounds, no privileges
  template<typename FT, int N, typename T, int MR>
  class MultiRegionAccessor<
      FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, false /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<N, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<N, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<N, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<N, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor =
          Realm::MultiAffineAccessor<FT, N, T>(instance, fid, bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline ArraySyntax::AffineSyntaxHelper<
        MultiRegionAccessor<
            FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, false, false, MR>,
        FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          MultiRegionAccessor<
              FT, N, T, Realm::MultiAffineAccessor<FT, N, T>, false, false, MR>,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<N, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Multi-Accessor, multi affine, 1, no bounds, no privileges
  template<typename FT, typename T, int MR>
  class MultiRegionAccessor<
      FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T>, false /*check bounds*/,
      false /*check privileges*/, MR> {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    MultiRegionAccessor(void) { }
  public:
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
    template<typename InputIterator>
    MultiRegionAccessor(
        InputIterator start, InputIterator stop, const Rect<1, T> source_bounds,
        FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (start == stop)
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = *start;
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      unsigned idx = 1;
      while (++start != stop)
      {
        assert(idx < MR);
        const Realm::RegionInstance inst = start->get_instance_info(
            start->get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        idx++;
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
  public:
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = is.bounds;
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(is.bounds);
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
    MultiRegionAccessor(
        const std::vector<PhysicalRegion>& regions,
        const Rect<1, T> source_bounds, FieldID fid,
        size_t actual_field_size = sizeof(FT),
#ifdef LEGION_DEBUG
        bool check_field_size = true,
#else
        bool check_field_size = false,
#endif
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      if (regions.empty())
        return;
      DomainT<1, T> is;
      const PhysicalRegion& region = regions.front();
      const Realm::RegionInstance instance = region.get_instance_info(
          region.get_privilege(), fid, actual_field_size, &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/, check_field_size);
      Rect<1, T> bounds = source_bounds.intersection(is.bounds);
      for (unsigned idx = 1; idx < regions.size(); idx++)
      {
        const Realm::RegionInstance inst = regions[idx].get_instance_info(
            regions[idx].get_privilege(), fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
            silence_warnings, false /*generic accessor*/, check_field_size);
        if (inst != instance)
          region.report_incompatible_multi_accessor(idx, fid, instance, inst);
        bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
      }
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(instance, fid, bounds);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      return accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
    template<typename REDOP, bool EXCLUSIVE>
    __LEGION_CUDA_HD__ inline void reduce(
        const Point<1, T>& p, typename REDOP::RHS val) const
    {
      REDOP::template apply<EXCLUSIVE>(accessor[p], val);
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };
#endif  // LEGION_MULTI_REGION_ACCESSOR

  // A hidden class for users that really know what they are doing
  /**
   * \class UnsafeFieldAccessor
   * This is a class for getting access to region data without
   * privilege checks or bounds checks. Users should only use
   * this accessor if they are confident that they actually do
   * have their privileges and bounds correct
   */
  template<
      typename FT, int N, typename T = coord_t,
      typename A = Realm::GenericAccessor<FT, N, T> >
  class UnsafeFieldAccessor {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    UnsafeFieldAccessor(void) { }
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,

        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      DomainT<N, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, true /*generic accessor*/,
          false /*check field size*/);
      if (!A::is_compatible(instance, fid, is.bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor = A(instance, fid, is.bounds, offset);
    }
  public:
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<N, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
    inline ArraySyntax::GenericSyntaxHelper<
        UnsafeFieldAccessor<FT, N, T, A>, FT, N, T, 2, LEGION_READ_WRITE>
        operator[](T index) const
    {
      return ArraySyntax::GenericSyntaxHelper<
          UnsafeFieldAccessor<FT, N, T, A>, FT, N, T, 2, LEGION_READ_WRITE>(
          *this, Point<1, T>(index));
    }
  public:
    mutable A accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  template<typename FT, typename T>
  class UnsafeFieldAccessor<FT, 1, T, Realm::GenericAccessor<FT, 1, T> > {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    UnsafeFieldAccessor(void) { }
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      DomainT<1, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, true /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::GenericAccessor<FT, 1, T>::is_compatible(
              instance, fid, is.bounds))
        region.report_incompatible_accessor("GenericAccessor", instance, fid);
      accessor =
          Realm::GenericAccessor<FT, 1, T>(instance, fid, is.bounds, offset);
    }
  public:
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    inline ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE> operator[](
        const Point<1, T>& p) const
    {
      return ArraySyntax::AccessorRefHelper<FT, LEGION_READ_WRITE>(accessor[p]);
    }
  public:
    mutable Realm::GenericAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  template<typename FT, int N, typename T>
  class UnsafeFieldAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T> > {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    UnsafeFieldAccessor(void) { }
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      DomainT<N, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, is.bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor =
          Realm::AffineAccessor<FT, N, T>(instance, fid, is.bounds, offset);
    }
    // With explicit bounds
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const Rect<N, T> source_bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<N, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, fid, source_bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor =
          Realm::AffineAccessor<FT, N, T>(instance, fid, source_bounds, offset);
    }
    // With explicit transform
    template<int M>
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const AffineTransform<M, N, T> transform, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<M, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, offset);
    }
    // With explicit transform and bounds
    template<int M>
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const AffineTransform<M, N, T> transform,
        const Rect<N, T> source_bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<M, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, N, T>::is_compatible(
              instance, transform.transform, transform.offset, fid,
              source_bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, N, T>(
          instance, transform.transform, transform.offset, fid, source_bounds,
          offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      for (int i = 0; i < N; i++) strides[i] = accessor.strides[i] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline FT& operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          UnsafeFieldAccessor<FT, N, T, Realm::AffineAccessor<FT, N, T> >, FT,
          N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
  public:
    Realm::AffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Specialization for UnsafeFieldAccessor for dimension 1
  // to avoid ambiguity for array access
  template<typename FT, typename T>
  class UnsafeFieldAccessor<FT, 1, T, Realm::AffineAccessor<FT, 1, T> > {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    UnsafeFieldAccessor(void) { }
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      DomainT<1, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, is.bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor =
          Realm::AffineAccessor<FT, 1, T>(instance, fid, is.bounds, offset);
    }
    // With explicit bounds
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const Rect<1, T> source_bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<1, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, source_bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor =
          Realm::AffineAccessor<FT, 1, T>(instance, fid, source_bounds, offset);
    }
    // With explicit transform
    template<int M>
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const AffineTransform<M, 1, T> transform, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<M, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, offset);
    }
    // With explicit transform and bounds
    template<int M>
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const AffineTransform<M, 1, T> transform,
        const Rect<1, T> source_bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<M, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<M, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::AffineAccessor<FT, 1, T>::is_compatible(
              instance, transform.transform, transform.offset, fid,
              source_bounds))
        region.report_incompatible_accessor("AffineAccessor", instance, fid);
      accessor = Realm::AffineAccessor<FT, 1, T>(
          instance, transform.transform, transform.offset, fid, source_bounds,
          offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<1, T>& r, size_t field_size = sizeof(FT)) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<1, T>& r, size_t strides[1],
        size_t field_size = sizeof(FT)) const
    {
      strides[0] = accessor.strides[0] / field_size;
      return accessor.ptr(r.lo);
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
  public:
    Realm::AffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

  template<typename FT, int N, typename T>
  class UnsafeFieldAccessor<FT, N, T, Realm::MultiAffineAccessor<FT, N, T> > {
  private:
    static_assert(N > 0, "DIM must be positive");
    static_assert(N <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    UnsafeFieldAccessor(void) { }
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      DomainT<N, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, is.bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, N, T>(
          instance, fid, is.bounds, offset);
    }
    // With explicit bounds
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const Rect<N, T> source_bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<N, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<N, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::MultiAffineAccessor<FT, N, T>::is_compatible(
              instance, fid, source_bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, N, T>(
          instance, fid, source_bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<N, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<N, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<N, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Rect<N, T>& r, size_t field_size = sizeof(FT)) const
    {
      size_t strides[N];
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
      assert(Internal::is_dense_layout(r, strides, field_size));
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
      if (!Internal::is_dense_layout(r, strides, field_size))
        PhysicalRegion::fail_nondense_rect();
#endif
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(
        const Rect<N, T>& r, size_t strides[N],
        size_t field_size = sizeof(FT)) const
    {
      FT* result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(result != nullptr);
#else
      if (result == nullptr)
        PhysicalRegion::fail_rect_piece();
#endif
      for (int i = 0; i < N; i++) strides[i] /= field_size;
      return result;
    }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<N, T>& p) const { return accessor[p]; }
    __LEGION_CUDA_HD__
    inline FT& operator[](T index) const
    {
      return ArraySyntax::AffineSyntaxHelper<
          UnsafeFieldAccessor<FT, N, T, Realm::MultiAffineAccessor<FT, N, T> >,
          FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
    }
  public:
    mutable Realm::MultiAffineAccessor<FT, N, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = N;
  };

  // Specialization for UnsafeFieldAccessor for dimension 1
  // to avoid ambiguity for array access
  template<typename FT, typename T>
  class UnsafeFieldAccessor<FT, 1, T, Realm::MultiAffineAccessor<FT, 1, T> > {
  private:
    static_assert(std::is_integral<T>::value, "must be integral type");
  public:
    UnsafeFieldAccessor(void) { }
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        bool silence_warnings = false, const char* warning_string = nullptr,
        size_t offset = 0)
    {
      DomainT<1, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, is.bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(
          instance, fid, is.bounds, offset);
    }
    // With explicit bounds
    UnsafeFieldAccessor(
        const PhysicalRegion& region, FieldID fid,
        const Rect<1, T> source_bounds, bool silence_warnings = false,
        const char* warning_string = nullptr, size_t offset = 0)
    {
      DomainT<1, T> is;
      const Realm::RegionInstance instance = region.get_instance_info(
          LEGION_NO_ACCESS, fid, sizeof(FT), &is,
          Internal::NT_TemplateHelper::encode_tag<1, T>(), warning_string,
          silence_warnings, false /*generic accessor*/,
          false /*check field size*/);
      if (!Realm::MultiAffineAccessor<FT, 1, T>::is_compatible(
              instance, fid, source_bounds))
        region.report_incompatible_accessor(
            "MultiAffineAccessor", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT, 1, T>(
          instance, fid, source_bounds, offset);
    }
  public:
    __LEGION_CUDA_HD__
    inline FT read(const Point<1, T>& p) const { return accessor.read(p); }
    __LEGION_CUDA_HD__
    inline void write(const Point<1, T>& p, FT val) const
    {
      accessor.write(p, val);
    }
    __LEGION_CUDA_HD__
    inline FT* ptr(const Point<1, T>& p) const { return accessor.ptr(p); }
    __LEGION_CUDA_HD__
    inline FT& operator[](const Point<1, T>& p) const { return accessor[p]; }
  public:
    mutable Realm::MultiAffineAccessor<FT, 1, T> accessor;
  public:
    typedef FT value_type;
    typedef FT& reference;
    typedef const FT& const_reference;
    static const int dim = 1;
  };

}  // namespace Legion
