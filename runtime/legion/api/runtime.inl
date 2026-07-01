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

// Included from runtime.h - do not include this directly

// Useful for IDEs
#include "legion/api/runtime.h"

namespace Legion {

  /**
   * \struct SerdezRedopFns
   * Small helper class for storing instantiated templates
   */
  struct SerdezRedopFns {
  public:
    SerdezInitFunc init_fn;
    SerdezFoldFunc fold_fn;
  };

  /**
   * \class LegionSerialization
   * The Legion serialization class provides template meta-programming
   * help for returning complex data types from task calls.  If the
   * types have three special methods defined on them then we know
   * how to serialize the type for the runtime rather than just doing
   * a dumb bit copy.  This is especially useful for types which
   * require deep copies instead of shallow bit copies.  The three
   * methods which must be defined are:
   * size_t legion_buffer_size(void)
   * void legion_serialize(void *buffer)
   * void legion_deserialize(const void *buffer)
   *(optional) void legion_buffer_finalize(const void *buffer)
   */
  class LegionSerialization {
  public:
    // A helper method for getting access to the runtime's
    // end_task method with private access
    static inline void end_helper(
        Context ctx, const void* result, size_t result_size)
    {
      Runtime::legion_task_postamble(ctx, result, result_size, false /*owned*/);
    }
    static inline void end_helper(
        Context ctx, const void* result, size_t result_size,
        const Realm::ExternalInstanceResource& resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&))
    {
      Runtime::legion_task_postamble(
          ctx, result, result_size, true /*owned*/, resource, freefunc);
    }
    static inline Future from_value_helper(const void* value, size_t value_size)
    {
      return Future::from_untyped_pointer(value, value_size, false /*owned*/);
    }
    static inline Future from_value_helper(
        const void* value, size_t value_size,
        const Realm::ExternalInstanceResource& resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&))
    {
      return Future::from_value(
          value, value_size, true /*owned*/, resource, freefunc);
    }
    static void free_func(const Realm::ExternalInstanceResource& res)
    {
      const Realm::ExternalMemoryResource* resource =
          legion_safe_cast<const Realm::ExternalMemoryResource*>(&res);
      free((void*)resource->base);
    }
    template<void (*FINALIZE)(const void* buffer)>
    static void free_func_wrapper(const Realm::ExternalInstanceResource& res)
    {
      const Realm::ExternalMemoryResource* resource =
          legion_safe_cast<const Realm::ExternalMemoryResource*>(&res);
      void* buffer = (void*)resource->base;
      (*FINALIZE)(buffer);
      free(buffer);
    }

    // WARNING: There are two levels of SFINAE (substitution failure is
    // not an error) here.  Proceed at your own risk. First we have to
    // check to see if the type is a struct.  If it is then we check to
    // see if it has a 'legion_serialize' method.  We assume if there is
    // a 'legion_serialize' method there are also 'legion_buffer_size'
    // and 'legion_deserialize' methods.

    template<typename T, bool HAS_SERIALIZE, bool HAS_FINALIZE>
    struct NonPODSerializer {
      static inline void end_task(Context ctx, T* result)
      {
        size_t buffer_size = result->legion_buffer_size();
        if (buffer_size > 0)
        {
          void* buffer = malloc(buffer_size);
          result->legion_serialize(buffer);
          Realm::ExternalMemoryResource resource(buffer, buffer_size);
          end_helper(ctx, buffer, buffer_size, resource, free_func);
        }
        else
          end_helper(ctx, nullptr, 0);
      }
      static inline Future from_value(const T* value)
      {
        size_t buffer_size = value->legion_buffer_size();
        void* buffer = malloc(buffer_size);
        value->legion_serialize(buffer);
        Realm::ExternalMemoryResource resource(buffer, buffer_size);
        return from_value_helper(buffer, buffer_size, resource, free_func);
      }
      static inline T unpack(
          const Future& f, bool silence_warnings, const char* warning_string)
      {
        size_t size = 0;
        const void* result = f.get_buffer(
            Memory::SYSTEM_MEM, &size, false /*check size*/, silence_warnings,
            warning_string);
        T derez;
        derez.legion_deserialize(result);
        return derez;
      }
    };

    // Specialization for the case where we have a finalize method
    template<typename T>
    struct NonPODSerializer<T, true, true> {
      static inline void end_task(Context ctx, T* result)
      {
        size_t buffer_size = result->legion_buffer_size();
        if (buffer_size > 0)
        {
          void* buffer = malloc(buffer_size);
          result->legion_serialize(buffer);
          Realm::ExternalMemoryResource resource(buffer, buffer_size);
          end_helper(
              ctx, buffer, buffer_size, resource,
              free_func_wrapper<T::legion_buffer_finalize>);
        }
        else
          end_helper(ctx, nullptr, 0);
      }
      static inline Future from_value(const T* value)
      {
        size_t buffer_size = value->legion_buffer_size();
        void* buffer = malloc(buffer_size);
        value->legion_serialize(buffer);
        Realm::ExternalMemoryResource resource(buffer, buffer_size);
        return from_value_helper(
            buffer, buffer_size, resource,
            free_func_wrapper<T::legion_buffer_finalize>);
      }
      static inline T unpack(
          const Future& f, bool silence_warnings, const char* warning_string)
      {
        size_t size = 0;
        const void* result = f.get_buffer(
            Memory::SYSTEM_MEM, &size, false /*check size*/, silence_warnings,
            warning_string);
        T derez;
        derez.legion_deserialize(result);
        return derez;
      }
    };

    // Further specialization for deferred reductions
    template<typename REDOP, bool EXCLUSIVE, bool FINAL>
    struct NonPODSerializer<DeferredReduction<REDOP, EXCLUSIVE>, false, FINAL> {
      static inline void end_task(
          Context ctx, DeferredReduction<REDOP, EXCLUSIVE>* result)
      {
        static_assert(
            !IsSerdezType<typename REDOP::RHS>::value,
            "Legion does not currently support serialize/deserialize "
            "methods on types in DefrredReductions");
        result->finalize(ctx);
      }
      static inline Future from_value(
          const DeferredReduction<REDOP, EXCLUSIVE>* value)
      {
        // Should never be called
        std::abort();
      }
      static inline DeferredReduction<REDOP, EXCLUSIVE> unpack(
          const Future& f, bool silence_warnings, const char* warning)
      {
        // Should never be called
        std::abort();
      }
    };

    // Further specialization to see if this a deferred value
    template<typename T, bool FINAL>
    struct NonPODSerializer<DeferredValue<T>, false, FINAL> {
      static inline void end_task(Context ctx, DeferredValue<T>* result)
      {
        static_assert(
            !IsSerdezType<T>::value,
            "Legion does not currently support serialize/deserialize "
            "methods on types in DeferredValues");
        result->finalize(ctx);
      }
      static inline Future from_value(const DeferredValue<T>* value)
      {
        // Should never be called
        std::abort();
      }
      static inline DeferredValue<T> unpack(
          const Future& f, bool silence_warnings, const char* warning_string)
      {
        // Should never be called
        std::abort();
      }
    };

    // Another specialization for Untyped deferred buffers
    template<bool FINAL>
    struct NonPODSerializer<UntypedDeferredValue, false, FINAL> {
      static inline void end_task(Context ctx, UntypedDeferredValue* result)
      {
        result->finalize(ctx);
      }
      static inline Future from_value(const UntypedDeferredValue* value)
      {
        // should never be called
        std::abort();
      }
      static inline UntypedDeferredValue unpack(
          const Future& f, bool silence_warnings, const char* warning_string)
      {
        // should never be called
        std::abort();
      }
    };

    template<typename T, bool FINAL>
    struct NonPODSerializer<T, false, FINAL> {
      static inline void end_task(Context ctx, T* result)
      {
        end_helper(ctx, (void*)result, sizeof(T));
      }
      static inline Future from_value(const T* value)
      {
        return from_value_helper((const void*)value, sizeof(T));
      }
      static inline T unpack(
          const Future& f, bool silence_warnings, const char* warning_string)
      {
        size_t size = sizeof(T);
        const T* result = static_cast<const T*>(f.get_buffer(
            Memory::SYSTEM_MEM, &size, true /*check size*/, silence_warnings,
            warning_string));
        return *result;
      }
    };

    template<typename T>
    struct IsSerdezType {
      // We should never testing this against an unserialable type
      static_assert(
          !std::is_base_of<Unserializable, T>::value,
          "Attempting to serialize unserializable type when "
          "returning a value from the end of the task. Unserializable "
          "types are not permitted to be returned by value.");

      typedef char yes;
      typedef long no;

      template<typename C>
      static yes test(
          decltype(&C::legion_buffer_size), decltype(&C::legion_serialize),
          decltype(&C::legion_deserialize));
      template<typename C>
      static no test(...);

      static constexpr bool value =
          sizeof(test<T>(nullptr, nullptr, nullptr)) == sizeof(yes);
    };

    template<typename T>
    struct IsFinalizeType {
      typedef char yes;
      typedef long no;

      template<typename C>
      static yes test(decltype(&C::legion_buffer_finalize));
      template<typename C>
      static no test(...);

      static constexpr bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
    };

    template<typename T, bool IS_STRUCT>
    struct StructHandler {
      static inline void end_task(Context ctx, T* result)
      {
        // Otherwise this is a struct, so see if it has serialization methods
        NonPODSerializer<T, IsSerdezType<T>::value, IsFinalizeType<T>::value>::
            end_task(ctx, result);
      }
      static inline Future from_value(const T* value)
      {
        return NonPODSerializer<
            T, IsSerdezType<T>::value,
            IsFinalizeType<T>::value>::from_value(value);
      }
      static inline T unpack(
          const Future& f, bool silence_warnings, const char* warning_string)
      {
        return NonPODSerializer<
            T, IsSerdezType<T>::value, IsFinalizeType<T>::value>::
            unpack(f, silence_warnings, warning_string);
      }
    };
    // False case of template specialization
    template<typename T>
    struct StructHandler<T, false> {
      static inline void end_task(Context ctx, T* result)
      {
        end_helper(ctx, (void*)result, sizeof(T));
      }
      static inline Future from_value(const T* value)
      {
        return from_value_helper((const void*)value, sizeof(T));
      }
      static inline T unpack(
          const Future& f, bool silence_warnings, const char* warning_string)
      {
        size_t size = sizeof(T);
        const T* result = static_cast<const T*>(f.get_buffer(
            Memory::SYSTEM_MEM, &size, true /*check size*/, silence_warnings,
            warning_string));
        return *result;
      }
    };

    // Figure out whether this is a struct or not
    // and call the appropriate Finisher
    template<typename T>
    static inline void end_task(Context ctx, T* result)
    {
      StructHandler<T, std::is_class<T>::value>::end_task(ctx, result);
    }

    template<typename T>
    static inline Future from_value(const T* value)
    {
      return StructHandler<T, std::is_class<T>::value>::from_value(value);
    }

    template<typename T>
    static inline T unpack(
        const Future& f, bool silence_warnings, const char* warning_string)
    {
      return StructHandler<T, std::is_class<T>::value>::unpack(
          f, silence_warnings, warning_string);
    }

    // Some more help for reduction operations with RHS types
    // that have serialize and deserialize methods

    template<typename REDOP_RHS>
    static void serdez_redop_init(
        const ReductionOp* reduction_op, void*& ptr, size_t& size)
    {
      REDOP_RHS init_serdez;
      memcpy(&init_serdez, reduction_op->identity, reduction_op->sizeof_rhs);
      size_t new_size = init_serdez.legion_buffer_size();
      if (new_size > size)
      {
        size = new_size;
        ptr = realloc(ptr, size);
      }
      init_serdez.legion_serialize(ptr);
    }

    template<typename REDOP_RHS>
    static void serdez_redop_fold(
        const ReductionOp* reduction_op, void*& lhs_ptr, size_t& lhs_size,
        const void* rhs_ptr)
    {
      REDOP_RHS lhs_serdez, rhs_serdez;
      lhs_serdez.legion_deserialize(lhs_ptr);
      rhs_serdez.legion_deserialize(rhs_ptr);
      (reduction_op->cpu_fold_excl_fn)(
          &lhs_serdez, 0, &rhs_serdez, 0, 1, reduction_op->userdata);
      size_t new_size = lhs_serdez.legion_buffer_size();
      // Reallocate the buffer if it has grown
      if (new_size > lhs_size)
      {
        lhs_size = new_size;
        lhs_ptr = realloc(lhs_ptr, lhs_size);
      }
      // Now save the value
      lhs_serdez.legion_serialize(lhs_ptr);
    }

    template<typename REDOP_RHS, bool HAS_SERDEZ>
    struct SerdezRedopHandler {
      static inline void register_reduction(
          ReductionOp* redop, ReductionOpID redop_id, bool permit_duplicates)
      {
        Runtime::register_reduction_op(
            redop_id, redop, nullptr, nullptr, permit_duplicates);
      }
    };
    // True case of template specialization
    template<typename REDOP_RHS>
    struct SerdezRedopHandler<REDOP_RHS, true> {
      static inline void register_reduction(
          ReductionOp* redop, ReductionOpID redop_id, bool permit_duplicates)
      {
        Runtime::register_reduction_op(
            redop_id, redop, serdez_redop_init<REDOP_RHS>,
            serdez_redop_fold<REDOP_RHS>, permit_duplicates);
      }
    };

    template<typename REDOP_RHS, bool IS_STRUCT>
    struct StructRedopHandler {
      static inline void register_reduction(
          ReductionOp* redop, ReductionOpID redop_id, bool permit_duplicates)
      {
        Runtime::register_reduction_op(
            redop_id, redop, nullptr, nullptr, permit_duplicates);
      }
    };
    // True case of template specialization
    template<typename REDOP_RHS>
    struct StructRedopHandler<REDOP_RHS, true> {
      static inline void register_reduction(
          ReductionOp* redop, ReductionOpID redop_id, bool permit_duplicates)
      {
        SerdezRedopHandler<REDOP_RHS, IsSerdezType<REDOP_RHS>::value>::
            register_reduction(redop, redop_id, permit_duplicates);
      }
    };

    // Register reduction functions if necessary
    template<typename REDOP>
    static inline void register_reduction(
        ReductionOpID redop_id, bool permit_duplicates)
    {
      StructRedopHandler<
          typename REDOP::RHS, std::is_class<typename REDOP::RHS>::value>::
          register_reduction(
              Realm::ReductionOpUntyped::create_reduction_op<REDOP>(), redop_id,
              permit_duplicates);
    }

    template<typename T>
    struct HasSerdezBound {
      typedef char yes;
      typedef long no;

      template<typename C>
      static yes test(decltype(&C::legion_upper_bound_size));
      template<typename C>
      static no test(...);

      static constexpr bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
    };

    template<typename T, bool HAS_BOUND>
    struct SerdezBound {
      static constexpr size_t value = T::legion_upper_bound_size();
    };

    template<typename T>
    struct SerdezBound<T, false> {
      static constexpr size_t value = LEGION_MAX_RETURN_SIZE;
    };

    template<typename T>
    struct SizeBound {
      static constexpr size_t value = sizeof(T);
    };

    template<typename T>
    struct ReturnSize {
      static constexpr size_t value = std::conditional<
          IsSerdezType<T>::value, SerdezBound<T, HasSerdezBound<T>::value>,
          SizeBound<T>>::type::value;
    };

  };  // Serialization namespace

  // Specialization for Domain
  template<>
  inline void LegionSerialization::end_task<Domain>(Context ctx, Domain* result)
  {
    Runtime::legion_task_postamble(ctx, *result, true /*take ownership*/);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline T Future::get_result(
      bool silence_warnings, const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    // Unpack the value using LegionSerialization in case
    // the type has an alternative method of unpacking
    return LegionSerialization::unpack<T>(
        *this, silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  /*static*/ inline Future Future::from_value(Runtime* rt, const T& value)
  //--------------------------------------------------------------------------
  {
    static_assert(
        !std::is_base_of<Domain, T>::value,
        "Use Future::from_domain for returning domains in futures");
    return LegionSerialization::from_value(&value);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  /*static*/ inline Future Future::from_value(const T& value)
  //--------------------------------------------------------------------------
  {
    static_assert(
        !std::is_base_of<Domain, T>::value,
        "Use Future::from_domain for returning domains in futures");
    return LegionSerialization::from_value(&value);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::create_index_space(
      Context ctx, const Rect<DIM, T>& bounds, const char* provenance)
  //--------------------------------------------------------------------------
  {
    const Domain domain(bounds);
    return IndexSpaceT<DIM, T>(create_index_space(
        ctx, domain, Internal::NT_TemplateHelper::template encode_tag<DIM, T>(),
        provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::create_index_space(
      Context ctx, const DomainT<DIM, T>& bounds, const char* provenance,
      bool take_ownership)
  //--------------------------------------------------------------------------
  {
    const Domain domain(bounds);
    return IndexSpaceT<DIM, T>(create_index_space(
        ctx, domain, Internal::NT_TemplateHelper::template encode_tag<DIM, T>(),
        provenance, take_ownership));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::create_index_space(
      Context ctx, const Future& future, const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexSpaceT<DIM, T>(create_index_space(
        ctx, DIM, future,
        Internal::NT_TemplateHelper::template encode_tag<DIM, T>(),
        provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::create_index_space(
      Context ctx, const std::vector<Point<DIM, T>>& points,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    // C++ type system is dumb
    std::vector<Realm::Point<DIM, T>> realm_points(points.size());
    for (unsigned idx = 0; idx < points.size(); idx++)
      realm_points[idx] = points[idx];
    const DomainT<DIM, T> realm_is((Realm::IndexSpace<DIM, T>(realm_points)));
    const Domain domain(realm_is);
    return IndexSpaceT<DIM, T>(create_index_space(
        ctx, domain, Internal::NT_TemplateHelper::template encode_tag<DIM, T>(),
        provenance, true /*take ownership*/));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::create_index_space(
      Context ctx, const std::vector<Rect<DIM, T>>& rects,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    // C++ type system is dumb
    std::vector<Realm::Rect<DIM, T>> realm_rects(rects.size());
    for (unsigned idx = 0; idx < rects.size(); idx++)
      realm_rects[idx] = rects[idx];
    const DomainT<DIM, T> realm_is((Realm::IndexSpace<DIM, T>(realm_rects)));
    const Domain domain(realm_is);
    return IndexSpaceT<DIM, T>(create_index_space(
        ctx, domain, Internal::NT_TemplateHelper::template encode_tag<DIM, T>(),
        provenance, true /*take ownership*/));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::union_index_spaces(
      Context ctx, const std::vector<IndexSpaceT<DIM, T>>& spaces,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    std::vector<IndexSpace> handles(spaces.size());
    for (unsigned idx = 0; idx < spaces.size(); idx++)
      handles[idx] = spaces[idx];
    return IndexSpaceT<DIM, T>(union_index_spaces(ctx, handles, provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::intersect_index_spaces(
      Context ctx, const std::vector<IndexSpaceT<DIM, T>>& spaces,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    std::vector<IndexSpace> handles(spaces.size());
    for (unsigned idx = 0; idx < spaces.size(); idx++)
      handles[idx] = spaces[idx];
    return IndexSpaceT<DIM, T>(
        intersect_index_spaces(ctx, handles, provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::subtract_index_spaces(
      Context ctx, IndexSpaceT<DIM, T> left, IndexSpaceT<DIM, T> right,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexSpaceT<DIM, T>(subtract_index_spaces(
        ctx, IndexSpace(left), IndexSpace(right), provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_equal_partition(
      Context ctx, IndexSpaceT<DIM, T> parent,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, size_t granularity,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_equal_partition(
        ctx, IndexSpace(parent), IndexSpace(color_space), granularity, color,
        prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_weights(
      Context ctx, IndexSpaceT<DIM, T> parent,
      const std::map<Point<COLOR_DIM, COLOR_T>, int>& weights,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, size_t granularity,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    std::map<DomainPoint, int> untyped_weights;
    for (const std::pair<const Point<COLOR_DIM, COLOR_T>, int>& weight_pair :
         weights)
      untyped_weights[DomainPoint(weight_pair.first)] = weight_pair.second;
    return IndexPartitionT<DIM, T>(create_partition_by_weights(
        ctx, IndexSpace(parent), untyped_weights, IndexSpace(color_space),
        granularity, color, prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_weights(
      Context ctx, IndexSpaceT<DIM, T> parent,
      const std::map<Point<COLOR_DIM, COLOR_T>, size_t>& weights,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, size_t granularity,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    std::map<DomainPoint, size_t> untyped_weights;
    for (const std::pair<const Point<COLOR_DIM, COLOR_T>, size_t>& weight_pair :
         weights)
      untyped_weights[DomainPoint(weight_pair.first)] = weight_pair.second;
    return IndexPartitionT<DIM, T>(create_partition_by_weights(
        ctx, IndexSpace(parent), untyped_weights, IndexSpace(color_space),
        granularity, color, prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_weights(
      Context ctx, IndexSpaceT<DIM, T> parent, const FutureMap& weights,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, size_t granularity,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_partition_by_weights(
        ctx, IndexSpace(parent), weights, IndexSpace(color_space), granularity,
        color, prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_union(
      Context ctx, IndexSpaceT<DIM, T> parent, IndexPartitionT<DIM, T> handle1,
      IndexPartitionT<DIM, T> handle2,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_partition_by_union(
        ctx, IndexSpace(parent), IndexPartition(handle1),
        IndexPartition(handle2), IndexSpace(color_space), part_kind, color,
        prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_intersection(
      Context ctx, IndexSpaceT<DIM, T> parent, IndexPartitionT<DIM, T> handle1,
      IndexPartitionT<DIM, T> handle2,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_partition_by_intersection(
        ctx, IndexSpace(parent), IndexPartition(handle1),
        IndexPartition(handle2), IndexSpace(color_space), part_kind, color,
        prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_intersection(
      Context ctx, IndexSpaceT<DIM, T> parent,
      IndexPartitionT<DIM, T> partition, PartitionKind part_kind, Color color,
      bool safe, const char* prov)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_partition_by_intersection(
        ctx, IndexSpace(parent), IndexPartition(partition), part_kind, color,
        safe, prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_difference(
      Context ctx, IndexSpaceT<DIM, T> parent, IndexPartitionT<DIM, T> handle1,
      IndexPartitionT<DIM, T> handle2,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_partition_by_difference(
        ctx, IndexSpace(parent), IndexPartition(handle1),
        IndexPartition(handle2), IndexSpace(color_space), part_kind, color,
        prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  Color Runtime::create_cross_product_partitions(
      Context ctx, IndexPartitionT<DIM, T> handle1,
      IndexPartitionT<DIM, T> handle2,
      typename std::map<IndexSpaceT<DIM, T>, IndexPartitionT<DIM, T>>& handles,
      PartitionKind part_kind, Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    std::map<IndexSpace, IndexPartition> untyped_handles;
    for (const std::pair<const IndexSpaceT<DIM, T>, IndexPartitionT<DIM, T>>&
             handle_pair : handles)
      untyped_handles[handle_pair.first] = IndexPartition::NO_PART;
    Color result = create_cross_product_partitions(
        ctx, handle1, handle2, untyped_handles, part_kind, color, prov);
    for (std::pair<const IndexSpaceT<DIM, T>, IndexPartitionT<DIM, T>>&
             handle_pair : handles)
    {
      std::map<IndexSpace, IndexPartition>::const_iterator finder =
          untyped_handles.find(handle_pair.first);
      legion_assert(finder != untyped_handles.end());
      handle_pair.second = IndexPartitionT<DIM, T>(finder->second);
    }
    return result;
  }

  //--------------------------------------------------------------------------
  template<int DIM1, typename T1, int DIM2, typename T2>
  void Runtime::create_association(
      Context ctx, LogicalRegionT<DIM1, T1> domain,
      LogicalRegionT<DIM1, T1> domain_parent, FieldID domain_fid,
      IndexSpaceT<DIM2, T2> range, MapperID id, MappingTagID tag,
      UntypedBuffer marg, const char* provenance)
  //--------------------------------------------------------------------------
  {
    create_association(
        ctx, LogicalRegion(domain), LogicalRegion(domain_parent), domain_fid,
        IndexSpace(range), id, tag, marg, provenance);
  }

  //--------------------------------------------------------------------------
  template<int DIM1, typename T1, int DIM2, typename T2>
  void Runtime::create_bidirectional_association(
      Context ctx, LogicalRegionT<DIM1, T1> domain,
      LogicalRegionT<DIM1, T1> domain_parent, FieldID domain_fid,
      LogicalRegionT<DIM2, T2> range, LogicalRegionT<DIM2, T2> range_parent,
      FieldID range_fid, MapperID id, MappingTagID tag, UntypedBuffer marg,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    create_bidirectional_association(
        ctx, LogicalRegion(domain), LogicalRegion(domain_parent), domain_fid,
        LogicalRegion(range), LogicalRegion(range_parent), range_fid, id, tag,
        marg, provenance);
  }

  //--------------------------------------------------------------------------
  template<int DIM, int COLOR_DIM, typename T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_restriction(
      Context ctx, IndexSpaceT<DIM, T> parent,
      IndexSpaceT<COLOR_DIM, T> color_space,
      Transform<DIM, COLOR_DIM, T> transform, Rect<DIM, T> extent,
      PartitionKind part_kind, Color color, const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_restricted_partition(
        ctx, parent, color_space, &transform, sizeof(transform), &extent,
        sizeof(extent), part_kind, color, provenance, __func__, false));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_blockify(
      Context ctx, IndexSpaceT<DIM, T> parent, Point<DIM, T> blocking_factor,
      Color color, const char* provenance)
  //--------------------------------------------------------------------------
  {
    Point<DIM, T> origin;
    for (int i = 0; i < DIM; i++) origin[i] = 0;
    return create_partition_by_blockify<DIM, T>(
        ctx, parent, blocking_factor, origin, color, provenance);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_blockify(
      Context ctx, IndexSpaceT<DIM, T> parent, Point<DIM, T> blocking_factor,
      Point<DIM, T> origin, Color color, const char* provenance)
  //--------------------------------------------------------------------------
  {
    // Get the domain of the color space to partition
    const DomainT<DIM, T> parent_is = get_index_space_domain(parent);
    const Rect<DIM, T>& bounds = parent_is.bounds;
    if (bounds.empty())
      return IndexPartitionT<DIM, T>();
    // Compute the intended color space bounds
    Point<DIM, T> colors;
    for (int i = 0; i < DIM; i++)
      colors[i] = (((bounds.hi[i] - bounds.lo[i]) +  // -1 and +1 cancel out
                    blocking_factor[i]) /
                   blocking_factor[i]) -
                  1;
    Point<DIM, T> zeroes;
    for (int i = 0; i < DIM; i++) zeroes[i] = 0;
    // Make the color space
    IndexSpaceT<DIM, T> color_space =
        create_index_space(ctx, Rect<DIM, T>(zeroes, colors));
    // Now make the transform matrix
    Transform<DIM, DIM, T> transform;
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        if (i == j)
          transform[i][j] = blocking_factor[i];
        else
          transform[i][j] = 0;
    // And the extent
    Point<DIM, T> ones;
    for (int i = 0; i < DIM; i++) ones[i] = 1;
    const Rect<DIM, T> extent(origin, origin + blocking_factor - ones);
    // Then do the create partition by restriction call
    return IndexPartitionT<DIM, T>(create_restricted_partition(
        ctx, parent, color_space, &transform, sizeof(transform), &extent,
        sizeof(extent), LEGION_DISJOINT_KIND, color, provenance, __func__,
        true));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_domain(
      Context ctx, IndexSpaceT<DIM, T> parent,
      const std::map<Point<COLOR_DIM, COLOR_T>, DomainT<DIM, T>>& domains,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, bool perform_intersections,
      PartitionKind part_kind, Color color, const char* provenance,
      bool take_ownership)
  //--------------------------------------------------------------------------
  {
    std::map<DomainPoint, Domain> converted_domains;
    for (const std::pair<const Point<COLOR_DIM, COLOR_T>, DomainT<DIM, T>>&
             domain_pair : domains)
      converted_domains[DomainPoint(domain_pair.first)] =
          Domain(domain_pair.second);
    return IndexPartitionT<DIM, T>(create_partition_by_domain(
        ctx, IndexSpace(parent), converted_domains, IndexSpace(color_space),
        perform_intersections, part_kind, color, provenance, take_ownership));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_domain(
      Context ctx, IndexSpaceT<DIM, T> parent,
      const FutureMap& domain_future_map,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, bool perform_intersections,
      PartitionKind part_kind, Color color, const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_partition_by_domain(
        ctx, IndexSpace(parent), domain_future_map, IndexSpace(color_space),
        perform_intersections, part_kind, color, provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_rectangles(
      Context ctx, IndexSpaceT<DIM, T> parent,
      const std::map<Point<COLOR_DIM, COLOR_T>, std::vector<Rect<DIM, T>>>&
          rectangles,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, bool perform_intersections,
      PartitionKind part_kind, Color color, const char* provenance,
      bool collective)
  //--------------------------------------------------------------------------
  {
    if (collective)
    {
      std::map<DomainPoint, Future> futures;
      for (const std::pair<
               const Point<COLOR_DIM, COLOR_T>,
               typename std::vector<Rect<DIM, T>>>& rect : rectangles)
      {
        const DomainT<DIM, T> domain(rect.second);
        futures[DomainPoint(rect.first)] =
            Future::from_domain(Domain(domain), true /*take ownership*/);
      }
      FutureMap fm = construct_future_map(
          ctx, IndexSpace(color_space), futures, true /*collective*/,
          0 /*shard id*/, true /*implicit sharding*/, provenance);
      return IndexPartitionT<DIM, T>(create_partition_by_domain(
          ctx, IndexSpace(parent), fm, IndexSpace(color_space),
          perform_intersections, part_kind, color, provenance));
    }
    else
    {
      // Make realm index spaces for each of the points and then we can call
      // the base domain version of this method which takes ownership of the
      // sparsity maps that have been created
      std::map<DomainPoint, Domain> domains;
      for (const std::pair<
               const Point<COLOR_DIM, COLOR_T>, std::vector<Rect<DIM, T>>>&
               rect_pair : rectangles)
        domains[DomainPoint(rect_pair.first)] =
            DomainT<DIM, T>(rect_pair.second);
      return IndexPartitionT<DIM, T>(create_partition_by_domain(
          ctx, IndexSpace(parent), domains, IndexSpace(color_space),
          perform_intersections, part_kind, color, provenance,
          true /*take ownership*/));
    }
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_partition_by_field(
      Context ctx, LogicalRegionT<DIM, T> handle, LogicalRegionT<DIM, T> parent,
      FieldID fid, IndexSpaceT<COLOR_DIM, COLOR_T> color_space, Color color,
      MapperID id, MappingTagID tag, PartitionKind part_kind,
      UntypedBuffer marg, const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_partition_by_field(
        ctx, LogicalRegion(handle), LogicalRegion(parent), fid,
        IndexSpace(color_space), color, id, tag, part_kind, marg, provenance));
  }

  //--------------------------------------------------------------------------
  template<
      int DIM1, typename T1, int DIM2, typename T2, int COLOR_DIM,
      typename COLOR_T>
  IndexPartitionT<DIM2, T2> Runtime::create_partition_by_image(
      Context ctx, IndexSpaceT<DIM2, T2> handle,
      LogicalPartitionT<DIM1, T1> projection, LogicalRegionT<DIM1, T1> parent,
      FieldID fid,  // type: Point<DIM2,COORD_T2>
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, MapperID id, MappingTagID tag, UntypedBuffer marg,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM2, T2>(create_partition_by_image(
        ctx, IndexSpace(handle), LogicalPartition(projection),
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, color,
        id, tag, marg, provenance));
  }

  //--------------------------------------------------------------------------
  template<
      int DIM1, typename T1, int DIM2, typename T2, int COLOR_DIM,
      typename COLOR_T>
  IndexPartitionT<DIM2, T2> Runtime::create_partition_by_image_range(
      Context ctx, IndexSpaceT<DIM2, T2> handle,
      LogicalPartitionT<DIM1, T1> projection, LogicalRegionT<DIM1, T1> parent,
      FieldID fid,  // type: Point<DIM2,COORD_T2>
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, MapperID id, MappingTagID tag, UntypedBuffer marg,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM2, T2>(create_partition_by_image_range(
        ctx, IndexSpace(handle), LogicalPartition(projection),
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, color,
        id, tag, marg, provenance));
  }

  //--------------------------------------------------------------------------
  template<
      int DIM1, typename T1, int DIM2, typename T2, int COLOR_DIM,
      typename COLOR_T>
  IndexPartitionT<DIM1, T1> Runtime::create_partition_by_preimage(
      Context ctx, IndexPartitionT<DIM2, T2> projection,
      LogicalRegionT<DIM1, T1> handle, LogicalRegionT<DIM1, T1> parent,
      FieldID fid,  // type: Point<DIM2,COORD_T2>
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, MapperID id, MappingTagID tag, UntypedBuffer marg,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM1, T1>(create_partition_by_preimage(
        ctx, IndexPartition(projection), LogicalRegion(handle),
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, color,
        id, tag, marg, provenance));
  }

  //--------------------------------------------------------------------------
  template<
      int DIM1, typename T1, int DIM2, typename T2, int COLOR_DIM,
      typename COLOR_T>
  IndexPartitionT<DIM1, T1> Runtime::create_partition_by_preimage_range(
      Context ctx, IndexPartitionT<DIM2, T2> projection,
      LogicalRegionT<DIM1, T1> handle, LogicalRegionT<DIM1, T1> parent,
      FieldID fid,  // type: Rect<DIM2,COORD_T2>
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, MapperID id, MappingTagID tag, UntypedBuffer marg,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM1, T1>(create_partition_by_preimage_range(
        ctx, IndexPartition(projection), LogicalRegion(handle),
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, color,
        id, tag, marg, provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexPartitionT<DIM, T> Runtime::create_pending_partition(
      Context ctx, IndexSpaceT<DIM, T> parent,
      IndexSpaceT<COLOR_DIM, COLOR_T> color_space, PartitionKind part_kind,
      Color color, const char* prov)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(create_pending_partition(
        ctx, IndexSpace(parent), IndexSpace(color_space), part_kind, color,
        prov));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexSpaceT<DIM, T> Runtime::create_index_space_union(
      Context ctx, IndexPartitionT<DIM, T> parent,
      Point<COLOR_DIM, COLOR_T> color,
      const typename std::vector<IndexSpaceT<DIM, T>>& handles,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    std::vector<IndexSpace> untyped_handles(handles.size());
    for (unsigned idx = 0; idx < handles.size(); idx++)
      untyped_handles[idx] = handles[idx];
    return IndexSpaceT<DIM, T>(create_index_space_union_internal(
        ctx, IndexPartition(parent), &color, sizeof(color),
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>(),
        provenance, __func__, untyped_handles));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexSpaceT<DIM, T> Runtime::create_index_space_union(
      Context ctx, IndexPartitionT<DIM, T> parent,
      Point<COLOR_DIM, COLOR_T> color, IndexPartitionT<DIM, T> handle,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexSpaceT<DIM, T>(create_index_space_union_internal(
        ctx, IndexPartition(parent), &color, sizeof(color),
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>(),
        provenance, __func__, IndexPartition(handle)));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexSpaceT<DIM, T> Runtime::create_index_space_intersection(
      Context ctx, IndexPartitionT<DIM, T> parent,
      Point<COLOR_DIM, COLOR_T> color,
      const typename std::vector<IndexSpaceT<DIM, T>>& handles,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    std::vector<IndexSpace> untyped_handles(handles.size());
    for (unsigned idx = 0; idx < handles.size(); idx++)
      untyped_handles[idx] = handles[idx];
    return IndexSpaceT<DIM, T>(create_index_space_intersection_internal(
        ctx, IndexPartition(parent), &color, sizeof(color),
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>(),
        provenance, __func__, untyped_handles));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexSpaceT<DIM, T> Runtime::create_index_space_intersection(
      Context ctx, IndexPartitionT<DIM, T> parent,
      Point<COLOR_DIM, COLOR_T> color, IndexPartitionT<DIM, T> handle,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    return IndexSpaceT<DIM, T>(create_index_space_intersection_internal(
        ctx, IndexPartition(parent), &color, sizeof(color),
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>(),
        provenance, __func__, IndexPartition(handle)));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexSpaceT<DIM, T> Runtime::create_index_space_difference(
      Context ctx, IndexPartitionT<DIM, T> parent,
      Point<COLOR_DIM, COLOR_T> color, IndexSpaceT<DIM, T> initial,
      const typename std::vector<IndexSpaceT<DIM, T>>& handles,
      const char* provenance)
  //--------------------------------------------------------------------------
  {
    std::vector<IndexSpace> untyped_handles(handles.size());
    for (unsigned idx = 0; idx < handles.size(); idx++)
      untyped_handles[idx] = handles[idx];
    return IndexSpaceT<DIM, T>(create_index_space_difference_internal(
        ctx, IndexPartition(parent), &color, sizeof(color),
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>(),
        provenance, __func__, IndexSpace(initial), untyped_handles));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T> Runtime::get_index_partition(
      IndexSpaceT<DIM, T> parent, Color color)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(
        get_index_partition(IndexSpace(parent), color));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  bool Runtime::has_index_partition(IndexSpaceT<DIM, T> parent, Color color)
  //--------------------------------------------------------------------------
  {
    return has_index_partition(IndexSpace(parent), color);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexSpaceT<DIM, T> Runtime::get_index_subspace(
      IndexPartitionT<DIM, T> p, Point<COLOR_DIM, COLOR_T> color)
  //--------------------------------------------------------------------------
  {
    return IndexSpaceT<DIM, T>(get_index_subspace_internal(
        IndexPartition(p), &color,
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>()));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  bool Runtime::has_index_subspace(
      IndexPartitionT<DIM, T> p, Point<COLOR_DIM, COLOR_T> color)
  //--------------------------------------------------------------------------
  {
    return has_index_subspace_internal(
        IndexPartition(p), &color,
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>());
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  DomainT<DIM, T> Runtime::get_index_space_domain(IndexSpaceT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    DomainT<DIM, T> realm_is;
    get_index_space_domain_internal(
        handle, &realm_is, Internal::NT_TemplateHelper::encode_tag<DIM, T>());
    return realm_is;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  DomainT<COLOR_DIM, COLOR_T> Runtime::get_index_partition_color_space(
      IndexPartitionT<DIM, T> p)
  //--------------------------------------------------------------------------
  {
    DomainT<COLOR_DIM, COLOR_T> realm_is;
    get_index_partition_color_space_internal(
        p, &realm_is,
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>());
    return realm_is;
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  IndexSpaceT<COLOR_DIM, COLOR_T> Runtime::get_index_partition_color_space_name(
      IndexPartitionT<DIM, T> p)
  //--------------------------------------------------------------------------
  {
    return IndexSpaceT<COLOR_DIM, COLOR_T>(
        get_index_partition_color_space_name(p));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  Point<COLOR_DIM, COLOR_T> Runtime::get_index_space_color(
      IndexSpaceT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    Point<COLOR_DIM, COLOR_T> point;
    return get_index_space_color_internal(
        IndexSpace(handle), &point,
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>());
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexSpaceT<DIM, T> Runtime::get_parent_index_space(
      IndexPartitionT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    return IndexSpaceT<DIM, T>(get_parent_index_space(IndexPartition(handle)));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  IndexPartitionT<DIM, T> Runtime::get_parent_index_partition(
      IndexSpaceT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    return IndexPartitionT<DIM, T>(
        get_parent_index_partition(IndexSpace(handle)));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  bool Runtime::safe_cast(
      Context ctx, Point<DIM, T> point, LogicalRegionT<DIM, T> region)
  //--------------------------------------------------------------------------
  {
    return safe_cast_internal(
        ctx, LogicalRegion(region), &point,
        Internal::NT_TemplateHelper::encode_tag<DIM, T>(), __func__);
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T> Runtime::create_logical_region(
      Context ctx, IndexSpaceT<DIM, T> index, FieldSpace fields,
      bool task_local, const char* provenance)
  //--------------------------------------------------------------------------
  {
    return LogicalRegionT<DIM, T>(create_logical_region(
        ctx, IndexSpace(index), fields, task_local, provenance));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T> Runtime::get_logical_partition(
      LogicalRegionT<DIM, T> parent, IndexPartitionT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    return LogicalPartitionT<DIM, T>(
        get_logical_partition(LogicalRegion(parent), IndexPartition(handle)));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T> Runtime::get_logical_partition_by_color(
      LogicalRegionT<DIM, T> parent, Color color)
  //--------------------------------------------------------------------------
  {
    return LogicalPartitionT<DIM, T>(
        get_logical_partition_by_color(LogicalRegion(parent), color));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T> Runtime::get_logical_partition_by_tree(
      IndexPartitionT<DIM, T> handle, FieldSpace space, RegionTreeID tid)
  //--------------------------------------------------------------------------
  {
    return LogicalPartitionT<DIM, T>(
        get_logical_partition_by_tree(IndexPartition(handle), space, tid));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T> Runtime::get_logical_subregion(
      LogicalPartitionT<DIM, T> parent, IndexSpaceT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    return LogicalRegionT<DIM, T>(
        get_logical_subregion(LogicalPartition(parent), IndexSpace(handle)));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  LogicalRegionT<DIM, T> Runtime::get_logical_subregion_by_color(
      LogicalPartitionT<DIM, T> parent, Point<COLOR_DIM, COLOR_T> color)
  //--------------------------------------------------------------------------
  {
    return LogicalRegionT<DIM, T>(get_logical_subregion_by_color_internal(
        LogicalPartition(parent), &color,
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>()));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  bool Runtime::has_logical_subregion_by_color(
      LogicalPartitionT<DIM, T> parent, Point<COLOR_DIM, COLOR_T> color)
  //--------------------------------------------------------------------------
  {
    return has_logical_subregion_by_color_internal(
        LogicalPartition(parent), &color,
        Internal::NT_TemplateHelper::encode_tag<COLOR_DIM, COLOR_T>());
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T> Runtime::get_logical_subregion_by_tree(
      IndexSpaceT<DIM, T> handle, FieldSpace space, RegionTreeID tid)
  //--------------------------------------------------------------------------
  {
    return LogicalRegionT<DIM, T>(
        get_logical_subregion_by_tree(IndexSpace(handle), space, tid));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
  Point<COLOR_DIM, COLOR_T> Runtime::get_logical_region_color_point(
      LogicalRegionT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    return get_logical_region_color_point(LogicalRegion(handle));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalRegionT<DIM, T> Runtime::get_parent_logical_region(
      LogicalPartitionT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    return LogicalRegionT<DIM, T>(
        get_parent_logical_region(LogicalPartition(handle)));
  }

  //--------------------------------------------------------------------------
  template<int DIM, typename T>
  LogicalPartitionT<DIM, T> Runtime::get_parent_logical_partition(
      LogicalRegionT<DIM, T> handle)
  //--------------------------------------------------------------------------
  {
    return LogicalPartitionT<DIM, T>(
        get_parent_logical_partition(LogicalRegion(handle)));
  }

  //--------------------------------------------------------------------------
  template<typename T>
  void Runtime::fill_field(
      Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
      const T& value, Predicate pred)
  //--------------------------------------------------------------------------
  {
    fill_field(ctx, handle, parent, fid, &value, sizeof(T), pred);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  void Runtime::fill_fields(
      Context ctx, LogicalRegion handle, LogicalRegion parent,
      const std::set<FieldID>& fields, const T& value, Predicate pred)
  //--------------------------------------------------------------------------
  {
    fill_fields(ctx, handle, parent, fields, &value, sizeof(T), pred);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  T* Runtime::get_local_task_variable(Context ctx, LocalVariableID id)
  //--------------------------------------------------------------------------
  {
    return static_cast<T*>(get_local_task_variable_untyped(ctx, id));
  }

  //--------------------------------------------------------------------------
  template<typename T>
  void Runtime::set_local_task_variable(
      Context ctx, LocalVariableID id, const T* value,
      void (*destructor)(void*))
  //--------------------------------------------------------------------------
  {
    set_local_task_variable_untyped(ctx, id, value, destructor);
  }

  //--------------------------------------------------------------------------
  template<typename REDOP>
  /*static*/ void Runtime::register_reduction_op(
      ReductionOpID redop_id, bool permit_duplicates)
  //--------------------------------------------------------------------------
  {
    // We also have to check to see if there are explicit serialization
    // and deserialization methods on the RHS type for doing fold reductions
    LegionSerialization::register_reduction<REDOP>(redop_id, permit_duplicates);
  }

#ifdef LEGION_GPU_REDUCTIONS
  //--------------------------------------------------------------------------
  template<typename REDOP>
  /*static*/ void Runtime::preregister_gpu_reduction_op(ReductionOpID redop)
  //--------------------------------------------------------------------------
  {
    Runtime::register_reduction_op<REDOP>(redop, false /*permit duplicates*/);
  }
#endif  // LEGION_GPU_REDUCTIONS

  //--------------------------------------------------------------------------
  template<typename SERDEZ>
  /*static*/ void Runtime::register_custom_serdez_op(
      CustomSerdezID serdez_id, bool permit_duplicates)
  //--------------------------------------------------------------------------
  {
    Runtime::register_custom_serdez_op(
        serdez_id, Realm::CustomSerdezUntyped::create_custom_serdez<SERDEZ>(),
        permit_duplicates);
  }

  namespace Internal {
    // Wrapper class for old projection functions
    template<RegionProjectionFnptr FNPTR>
    class RegionProjectionWrapper : public ProjectionFunctor {
    public:
      RegionProjectionWrapper(void) : ProjectionFunctor() { }
      virtual ~RegionProjectionWrapper(void) { }
    public:
      virtual LogicalRegion project(
          Context ctx, Task* task, unsigned index, LogicalRegion upper_bound,
          const DomainPoint& point)
      {
        return (*FNPTR)(upper_bound, point, runtime);
      }
      virtual LogicalRegion project(
          Context ctx, Task* task, unsigned index, LogicalPartition upper_bound,
          const DomainPoint& point)
      {
        std::abort();
      }
      virtual bool is_exclusive(void) const { return false; }
    };
  };  // namespace Internal

  //--------------------------------------------------------------------------
  template<
      LogicalRegion (*PROJ_PTR)(LogicalRegion, const DomainPoint&, Runtime*)>
  /*static*/ ProjectionID Runtime::register_region_function(ProjectionID handle)
  //--------------------------------------------------------------------------
  {
    Runtime::preregister_projection_functor(
        handle, new Internal::RegionProjectionWrapper<PROJ_PTR>());
    return handle;
  }

  namespace Internal {
    // Wrapper class for old projection functions
    template<PartitionProjectionFnptr FNPTR>
    class PartitionProjectionWrapper : public ProjectionFunctor {
    public:
      PartitionProjectionWrapper(void) : ProjectionFunctor() { }
      virtual ~PartitionProjectionWrapper(void) { }
    public:
      virtual LogicalRegion project(
          Context ctx, Task* task, unsigned index, LogicalRegion upper_bound,
          const DomainPoint& point)
      {
        std::abort();
      }
      virtual LogicalRegion project(
          Context ctx, Task* task, unsigned index, LogicalPartition upper_bound,
          const DomainPoint& point)
      {
        return (*FNPTR)(upper_bound, point, runtime);
      }
      virtual bool is_exclusive(void) const { return false; }
    };
  };  // namespace Internal

  //--------------------------------------------------------------------------
  template<
      LogicalRegion (*PROJ_PTR)(LogicalPartition, const DomainPoint&, Runtime*)>
  /*static*/ ProjectionID Runtime::register_partition_function(
      ProjectionID handle)
  //--------------------------------------------------------------------------
  {
    Runtime::preregister_projection_functor(
        handle, new Internal::PartitionProjectionWrapper<PROJ_PTR>());
    return handle;
  }

  //--------------------------------------------------------------------------
  // Wrapper functions for high-level tasks
  //--------------------------------------------------------------------------

  /**
   * \class LegionTaskWrapper
   * This is a helper class that has static template methods for
   * wrapping Legion application tasks.  For all tasks we can make
   * wrappers both for normal execution and also for inline execution.
   */
  class LegionTaskWrapper {
  public:
    // Non-void return type for new legion task types
    template<
        typename T,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    static void legion_task_wrapper(
        const void*, size_t, const void*, size_t, Processor);
    template<
        typename T, typename UDT,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
            const UDT&)>
    static void legion_udt_task_wrapper(
        const void*, size_t, const void*, size_t, Processor);
  public:
    // Void return type for new legion task types
    template<void (*TASK_PTR)(
        const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    static void legion_task_wrapper(
        const void*, size_t, const void*, size_t, Processor);
    template<
        typename UDT, void (*TASK_PTR)(
                          const Task*, const std::vector<PhysicalRegion>&,
                          Context, Runtime*, const UDT&)>
    static void legion_udt_task_wrapper(
        const void*, size_t, const void*, size_t, Processor);
  public:
    // Do-it-yourself pre/post-ambles for code generators
    // These are deprecated and are just here for backwards compatibility
    static void legion_task_preamble(
        const void* data, size_t datalen, Processor p, const Task*& task,
        const std::vector<PhysicalRegion>*& ptr, Context& ctx,
        Runtime*& runtime);
    static void legion_task_postamble(
        Context ctx, const void* retvalptr = nullptr, size_t retvalsize = 0);
  };

  //--------------------------------------------------------------------------
  template<
      typename T,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  void LegionTaskWrapper::legion_task_wrapper(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Processor p)
  //--------------------------------------------------------------------------
  {
    // Assert that we are returning Futures or FutureMaps
    static_assert(
        !std::is_same<T, Future>::value,
        "Future types are not permitted as return types for Legion tasks");
    static_assert(
        !std::is_same<T, FutureMap>::value,
        "FutureMap types are not permitted as return types for Legion tasks");
    const Task* task;
    Context ctx;
    Runtime* rt;
    const std::vector<PhysicalRegion>* regions;
    Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

    // Invoke the task with the given context
    T return_value = (*TASK_PTR)(task, *regions, ctx, rt);

    // Send the return value back
    LegionSerialization::end_task<T>(ctx, &return_value);
  }

  //--------------------------------------------------------------------------
  template<void (*TASK_PTR)(
      const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  void LegionTaskWrapper::legion_task_wrapper(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Processor p)
  //--------------------------------------------------------------------------
  {
    const Task* task;
    Context ctx;
    Runtime* rt;
    const std::vector<PhysicalRegion>* regions;
    Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

    (*TASK_PTR)(task, *regions, ctx, rt);

    Runtime::legion_task_postamble(ctx);
  }

  //--------------------------------------------------------------------------
  template<
      typename T, typename UDT,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
          const UDT&)>
  void LegionTaskWrapper::legion_udt_task_wrapper(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Processor p)
  //--------------------------------------------------------------------------
  {
    // Assert that we are returning Futures or FutureMaps
    static_assert(
        !std::is_same<T, Future>::value,
        "Future types are not permitted as return types for Legion tasks");
    static_assert(
        !std::is_same<T, FutureMap>::value,
        "FutureMap types are not permitted as return types for Legion tasks");

    const Task* task;
    Context ctx;
    Runtime* rt;
    const std::vector<PhysicalRegion>* regions;
    Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

    const UDT* user_data = nullptr;
    static_assert(sizeof(user_data) == sizeof(userdata), "C++ is dumb");
    memcpy(&user_data, &userdata, sizeof(user_data));

    // Invoke the task with the given context
    T return_value = (*TASK_PTR)(task, *regions, ctx, rt, *user_data);

    // Send the return value back
    LegionSerialization::end_task<T>(ctx, &return_value);
  }

  //--------------------------------------------------------------------------
  template<
      typename UDT, void (*TASK_PTR)(
                        const Task*, const std::vector<PhysicalRegion>&,
                        Context, Runtime*, const UDT&)>
  void LegionTaskWrapper::legion_udt_task_wrapper(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Processor p)
  //--------------------------------------------------------------------------
  {
    const Task* task;
    Context ctx;
    Runtime* rt;
    const std::vector<PhysicalRegion>* regions;
    Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

    const UDT* user_data = nullptr;
    static_assert(sizeof(user_data) == sizeof(userdata), "C++ is dumb");
    memcpy(&user_data, &userdata, sizeof(user_data));

    (*TASK_PTR)(task, *regions, ctx, rt, *user_data);

    // Send an empty return value back
    Runtime::legion_task_postamble(ctx);
  }

  //--------------------------------------------------------------------------
  inline void LegionTaskWrapper::legion_task_preamble(
      const void* data, size_t datalen, Processor p, const Task*& task,
      const std::vector<PhysicalRegion>*& regionsptr, Context& ctx,
      Runtime*& runtime)
  //--------------------------------------------------------------------------
  {
    Runtime::legion_task_preamble(
        data, datalen, p, task, regionsptr, ctx, runtime);
  }

  //--------------------------------------------------------------------------
  inline void LegionTaskWrapper::legion_task_postamble(
      Context ctx, const void* retvalptr /*= nullptr*/,
      size_t retvalsize /*= 0*/)
  //--------------------------------------------------------------------------
  {
    Runtime::legion_task_postamble(ctx, retvalptr, retvalsize);
  }

  //--------------------------------------------------------------------------
  template<
      typename T,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  VariantID Runtime::register_task_variant(
      const TaskVariantRegistrar& registrar, VariantID vid)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<T, TASK_PTR>);
    return register_task_variant(
        registrar, desc, nullptr /*UDT*/, 0 /*sizeof(UDT)*/,
        LegionSerialization::ReturnSize<T>::value, vid);
  }

  //--------------------------------------------------------------------------
  template<
      typename T, typename UDT,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
          const UDT&)>
  VariantID Runtime::register_task_variant(
      const TaskVariantRegistrar& registrar, const UDT& user_data,
      VariantID vid)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(
        LegionTaskWrapper::legion_udt_task_wrapper<T, UDT, TASK_PTR>);
    return register_task_variant(
        registrar, desc, &user_data, sizeof(UDT),
        LegionSerialization::ReturnSize<T>::value, vid);
  }

  //--------------------------------------------------------------------------
  template<void (*TASK_PTR)(
      const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  VariantID Runtime::register_task_variant(
      const TaskVariantRegistrar& registrar, VariantID vid)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
    return register_task_variant(
        registrar, desc, nullptr /*UDT*/, 0 /*sizeof(UDT)*/, 0 /*return size*/,
        vid);
  }

  //--------------------------------------------------------------------------
  template<
      typename UDT, void (*TASK_PTR)(
                        const Task*, const std::vector<PhysicalRegion>&,
                        Context, Runtime*, const UDT&)>
  VariantID Runtime::register_task_variant(
      const TaskVariantRegistrar& registrar, const UDT& user_data,
      VariantID vid)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(
        LegionTaskWrapper::legion_udt_task_wrapper<UDT, TASK_PTR>);
    return register_task_variant(
        registrar, desc, &user_data, sizeof(UDT), 0 /*return size*/, vid);
  }

  //--------------------------------------------------------------------------
  template<
      typename T,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  /*static*/ VariantID Runtime::preregister_task_variant(
      const TaskVariantRegistrar& registrar,
      const char* task_name /*= nullptr*/, VariantID vid /*=AUTO_GENERATE_ID*/)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<T, TASK_PTR>);
    return preregister_task_variant(
        registrar, desc, nullptr /*UDT*/, 0 /*sizeof(UDT)*/, task_name, vid,
        LegionSerialization::ReturnSize<T>::value);
  }

  //--------------------------------------------------------------------------
  template<
      typename T, typename UDT,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
          const UDT&)>
  /*static*/ VariantID Runtime::preregister_task_variant(
      const TaskVariantRegistrar& registrar, const UDT& user_data,
      const char* task_name /*= nullptr*/, VariantID vid /*=AUTO_GENERATE_ID*/)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(
        LegionTaskWrapper::legion_udt_task_wrapper<T, UDT, TASK_PTR>);
    return preregister_task_variant(
        registrar, desc, &user_data, sizeof(UDT), task_name, vid,
        LegionSerialization::ReturnSize<T>::value);
  }

  //--------------------------------------------------------------------------
  template<void (*TASK_PTR)(
      const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  /*static*/ VariantID Runtime::preregister_task_variant(
      const TaskVariantRegistrar& registrar,
      const char* task_name /*= nullptr*/,
      const VariantID vid /*=AUTO_GENERATE_ID*/)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
    return preregister_task_variant(
        registrar, desc, nullptr /*UDT*/, 0 /*sizeof(UDT)*/, task_name, vid,
        0 /*return size*/);
  }

  //--------------------------------------------------------------------------
  template<
      typename UDT, void (*TASK_PTR)(
                        const Task*, const std::vector<PhysicalRegion>&,
                        Context, Runtime*, const UDT&)>
  /*static*/ VariantID Runtime::preregister_task_variant(
      const TaskVariantRegistrar& registrar, const UDT& user_data,
      const char* task_name /*= nullptr*/, VariantID vid /*=AUTO_GENERATE_ID*/)
  //--------------------------------------------------------------------------
  {
    CodeDescriptor desc(
        LegionTaskWrapper::legion_udt_task_wrapper<UDT, TASK_PTR>);
    return preregister_task_variant(
        registrar, desc, &user_data, sizeof(UDT), task_name, vid,
        0 /*return size*/);
  }

  //--------------------------------------------------------------------------
  template<
      typename T,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  /*static*/ TaskID Runtime::register_legion_task(
      TaskID id, Processor::Kind proc_kind, bool single, bool index,
      VariantID vid, TaskConfigOptions options, const char* task_name)
  //--------------------------------------------------------------------------
  {
    bool check_task_id = true;
    if (id == LEGION_AUTO_GENERATE_ID)
    {
      id = generate_static_task_id();
      check_task_id = false;
    }
    TaskVariantRegistrar registrar(id, task_name);
    registrar.set_leaf(options.leaf);
    registrar.set_inner(options.inner);
    registrar.set_idempotent(options.idempotent);
    registrar.add_constraint(ProcessorConstraint(proc_kind));
    CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<T, TASK_PTR>);
    preregister_task_variant(
        registrar, desc, nullptr /*UDT*/, 0 /*sizeof(UDT)*/, task_name, vid,
        LegionSerialization::ReturnSize<T>::value, check_task_id);
    return id;
  }

  //--------------------------------------------------------------------------
  template<void (*TASK_PTR)(
      const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
  /*static*/ TaskID Runtime::register_legion_task(
      TaskID id, Processor::Kind proc_kind, bool single, bool index,
      VariantID vid, TaskConfigOptions options, const char* task_name)
  //--------------------------------------------------------------------------
  {
    bool check_task_id = true;
    if (id == LEGION_AUTO_GENERATE_ID)
    {
      id = generate_static_task_id();
      check_task_id = false;
    }
    TaskVariantRegistrar registrar(id, task_name);
    registrar.set_leaf(options.leaf);
    registrar.set_inner(options.inner);
    registrar.set_idempotent(options.idempotent);
    registrar.add_constraint(ProcessorConstraint(proc_kind));
    CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
    preregister_task_variant(
        registrar, desc, nullptr /*UDT*/, 0 /*sizeof(UDT)*/, task_name, vid,
        0 /*return size*/, check_task_id);
    return id;
  }

  //--------------------------------------------------------------------------
  template<
      typename T, typename UDT,
      T (*TASK_PTR)(
          const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
          const UDT&)>
  /*static*/ TaskID Runtime::register_legion_task(
      TaskID id, Processor::Kind proc_kind, bool single, bool index,
      const UDT& user_data, VariantID vid, TaskConfigOptions options,
      const char* task_name)
  //--------------------------------------------------------------------------
  {
    bool check_task_id = true;
    if (id == LEGION_AUTO_GENERATE_ID)
    {
      id = generate_static_task_id();
      check_task_id = false;
    }
    TaskVariantRegistrar registrar(id, task_name);
    registrar.set_leaf(options.leaf);
    registrar.set_inner(options.inner);
    registrar.set_idempotent(options.idempotent);
    registrar.add_constraint(ProcessorConstraint(proc_kind));
    CodeDescriptor desc(
        LegionTaskWrapper::legion_udt_task_wrapper<T, UDT, TASK_PTR>);
    preregister_task_variant(
        registrar, desc, &user_data, sizeof(UDT), task_name, vid,
        LegionSerialization::ReturnSize<T>::value, check_task_id);
    return id;
  }

  //--------------------------------------------------------------------------
  template<
      typename UDT, void (*TASK_PTR)(
                        const Task*, const std::vector<PhysicalRegion>&,
                        Context, Runtime*, const UDT&)>
  /*static*/ TaskID Runtime::register_legion_task(
      TaskID id, Processor::Kind proc_kind, bool single, bool index,
      const UDT& user_data, VariantID vid, TaskConfigOptions options,
      const char* task_name)
  //--------------------------------------------------------------------------
  {
    bool check_task_id = true;
    if (id == LEGION_AUTO_GENERATE_ID)
    {
      id = generate_static_task_id();
      check_task_id = false;
    }
    TaskVariantRegistrar registrar(id, task_name);
    registrar.set_leaf(options.leaf);
    registrar.set_inner(options.inner);
    registrar.set_idempotent(options.idempotent);
    registrar.add_constraint(ProcessorConstraint(proc_kind));
    CodeDescriptor desc(
        LegionTaskWrapper::legion_udt_task_wrapper<UDT, TASK_PTR>);
    preregister_task_variant(
        registrar, desc, &user_data, sizeof(UDT), task_name, vid,
        0 /*return size*/, check_task_id);
    return id;
  }

  //--------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream& os, const LogicalRegion& lr)
  //--------------------------------------------------------------------------
  {
    Runtime* runtime = Runtime::get_runtime();
    const void* name = nullptr;
    size_t size = 0;
    if (runtime->retrieve_semantic_information(
            lr, LEGION_NAME_SEMANTIC_TAG, name, size, true /*can fail*/,
            false /*wait until ready*/))
    {
      std::string_view view(static_cast<const char*>(name), size);
      os << view;
    }
    else
      os << "(is=" << lr.get_index_space() << ", fs=" << lr.get_field_space()
         << ", root=" << lr.get_tree_id() << ")";
    return os;
  }

  //--------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream& os, const LogicalPartition& lp)
  //--------------------------------------------------------------------------
  {
    Runtime* runtime = Runtime::get_runtime();
    const void* name = nullptr;
    size_t size = 0;
    if (runtime->retrieve_semantic_information(
            lp, LEGION_NAME_SEMANTIC_TAG, name, size, true /*can fail*/,
            false /*wait until ready*/))
    {
      std::string_view view(static_cast<const char*>(name), size);
      os << view;
    }
    else
      os << "(is=" << lp.get_index_partition()
         << ", fs=" << lp.get_field_space() << ", root=" << lp.get_tree_id()
         << ")";
    return os;
  }

  //--------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream& os, const IndexSpace& is)
  //--------------------------------------------------------------------------
  {
    // Check to see if we can find the semantic information name
    Runtime* runtime = Runtime::get_runtime();
    const void* name = nullptr;
    size_t size = 0;
    if (runtime->retrieve_semantic_information(
            is, LEGION_NAME_SEMANTIC_TAG, name, size, true /*can fail*/,
            false /*wait until ready*/))
    {
      std::string_view view(static_cast<const char*>(name), size);
      os << view;
    }
    else
      os << "(id=" << is.get_id() << ", tree_id=" << is.get_tree_id() << ")";
    return os;
  }

  //--------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream& os, const IndexPartition& ip)
  //--------------------------------------------------------------------------
  {
    Runtime* runtime = Runtime::get_runtime();
    const void* name = nullptr;
    size_t size = 0;
    if (runtime->retrieve_semantic_information(
            ip, LEGION_NAME_SEMANTIC_TAG, name, size, true /*can fail*/,
            false /*wait until ready*/))
    {
      std::string_view view(static_cast<const char*>(name), size);
      os << view;
    }
    else
      os << "(id=" << ip.get_id() << ", tree_id=" << ip.get_tree_id() << ")";
    return os;
  }

  //--------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream& os, const FieldSpace& fs)
  //--------------------------------------------------------------------------
  {
    Runtime* runtime = Runtime::get_runtime();
    const void* name = nullptr;
    size_t size = 0;
    if (runtime->retrieve_semantic_information(
            fs, LEGION_NAME_SEMANTIC_TAG, name, size, true /*can fail*/,
            false /*wait unilt ready*/))
    {
      std::string_view view(static_cast<const char*>(name), size);
      os << view;
    }
    else
      os << "(id=" << fs.get_id() << ")";
    return os;
  }

}  // namespace Legion
