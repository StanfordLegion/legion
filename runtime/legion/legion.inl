/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// Included from legion.h - do not include this directly

// Useful for IDEs 
#include "legion.h"

#include <limits>

namespace Legion {

    /**
     * \struct SerdezRedopFns
     * Small helper class for storing instantiated templates
     */
    struct SerdezRedopFns {
    public:
      SerdezInitFnptr init_fn;
      SerdezFoldFnptr fold_fn;
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
     */
    class LegionSerialization {
    public:
      // A helper method for getting access to the runtime's
      // end_task method with private access
      static inline void end_helper(Runtime *rt, Context ctx,
          const void *result, size_t result_size, bool owned)
      {
        Runtime::legion_task_postamble(rt, ctx, result, result_size, owned);
      }
      static inline Future from_value_helper(Runtime *rt, 
          const void *value, size_t value_size, bool owned)
      {
        return rt->from_value(value, value_size, owned);
      }

      // WARNING: There are two levels of SFINAE (substitution failure is 
      // not an error) here.  Proceed at your own risk. First we have to 
      // check to see if the type is a struct.  If it is then we check to 
      // see if it has a 'legion_serialize' method.  We assume if there is 
      // a 'legion_serialize' method there are also 'legion_buffer_size'
      // and 'legion_deserialize' methods.
      
      template<typename T, bool HAS_SERIALIZE>
      struct NonPODSerializer {
        static inline void end_task(Runtime *rt, Context ctx,
                                    T *result)
        {
          size_t buffer_size = result->legion_buffer_size();
          if (buffer_size > 0)
          {
            void *buffer = malloc(buffer_size);
            result->legion_serialize(buffer);
            end_helper(rt, ctx, buffer, buffer_size, true/*owned*/);
            // No need to free the buffer, the Legion runtime owns it now
          }
          else
            end_helper(rt, ctx, NULL, 0, false/*owned*/);
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          size_t buffer_size = value->legion_buffer_size();
          void *buffer = malloc(buffer_size);
          value->legion_serialize(buffer);
          return from_value_helper(rt, buffer, buffer_size, true/*owned*/);
        }
        static inline T unpack(const Future &f, bool silence_warnings,
                               const char *warning_string)
        {
          const void *result = 
            f.get_untyped_pointer(silence_warnings, warning_string);
          T derez;
          derez.legion_deserialize(result);
          return derez;
        }
      };

      // Further specialization for deferred reductions
      template<typename REDOP, bool EXCLUSIVE>
      struct NonPODSerializer<DeferredReduction<REDOP,EXCLUSIVE>,false> {
        static inline void end_task(Runtime *rt, Context ctx,
                                    DeferredReduction<REDOP,EXCLUSIVE> *result)
        {
          result->finalize(rt, ctx);
        }
        static inline Future from_value(Runtime *rt, 
            const DeferredReduction<REDOP,EXCLUSIVE> *value)
        {
          // Should never be called
          assert(false);
          return from_value_helper(rt, (const void*)value,
            sizeof(DeferredReduction<REDOP,EXCLUSIVE>), false/*owned*/);
        }
        static inline DeferredReduction<REDOP,EXCLUSIVE> 
          unpack(const Future &f, bool silence_warnings, const char *warning)
        {
          // Should never be called
          assert(false);
          const void *result = f.get_untyped_pointer(silence_warnings, warning);
          return (*((const DeferredReduction<REDOP,EXCLUSIVE>*)result));
        }
      };

      // Further specialization to see if this a deferred value
      template<typename T>
      struct NonPODSerializer<DeferredValue<T>,false> {
        static inline void end_task(Runtime *rt, Context ctx,
                                    DeferredValue<T> *result)
        {
          result->finalize(rt, ctx);
        }
        static inline Future from_value(Runtime *rt, const DeferredValue<T> *value)
        {
          // Should never be called
          assert(false);
          return from_value_helper(rt, (const void*)value,
                                   sizeof(DeferredValue<T>), false/*owned*/);
        }
        static inline DeferredValue<T> unpack(const Future &f,
            bool silence_warnings, const char *warning_string)
        {
          // Should never be called
          assert(false);
          const void *result = 
            f.get_untyped_pointer(silence_warnings, warning_string);
          return (*((const DeferredValue<T>*)result));
        }
      }; 
      
      template<typename T>
      struct NonPODSerializer<T,false> {
        static inline void end_task(Runtime *rt, Context ctx, T *result)
        {
          end_helper(rt, ctx, (void*)result, sizeof(T), false/*owned*/);
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          return from_value_helper(rt, (const void*)value,
                                   sizeof(T), false/*owned*/);
        }
        static inline T unpack(const Future &f, bool silence_warnings,
                               const char *warning_string)
        {
          return f.get_reference<T>(silence_warnings, warning_string);
        }
      };

      template <typename T>
      struct IsSerdezType {
        typedef char yes; typedef long no;

        template <typename C>
        static yes test(decltype(&C::legion_buffer_size),
                        decltype(&C::legion_serialize),
                        decltype(&C::legion_deserialize));
        template <typename C> static no test(...);

        static constexpr bool value =
          sizeof(test<T>(nullptr, nullptr, nullptr)) == sizeof(yes);
      };

      template<typename T, bool IS_STRUCT>
      struct StructHandler {
        static inline void end_task(Runtime *rt, Context ctx, T *result)
        {
          // Otherwise this is a struct, so see if it has serialization methods
          NonPODSerializer<T,IsSerdezType<T>::value>::end_task(rt, ctx, result);
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          return NonPODSerializer<T,IsSerdezType<T>::value>::from_value(
                                                                  rt, value);
        }
        static inline T unpack(const Future &f, bool silence_warnings,
                               const char *warning_string)
        {
          return NonPODSerializer<T,IsSerdezType<T>::value>::unpack(f,
                                    silence_warnings, warning_string); 
        }
      };
      // False case of template specialization
      template<typename T>
      struct StructHandler<T,false> {
        static inline void end_task(Runtime *rt, Context ctx, T *result)
        {
          end_helper(rt, ctx, (void*)result, sizeof(T), false/*owned*/);
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          return from_value_helper(rt, (const void*)value, 
                                   sizeof(T), false/*owned*/);
        }
        static inline T unpack(const Future &f, bool silence_warnings,
                               const char *warning_string)
        {
          return f.get_reference<T>(silence_warnings, warning_string);
        }
      };

      // Figure out whether this is a struct or not 
      // and call the appropriate Finisher
      template<typename T>
      static inline void end_task(Runtime *rt, Context ctx, T *result)
      {
        StructHandler<T,std::is_class<T>::value>::end_task(rt, ctx, result);
      }

      template<typename T>
      static inline Future from_value(Runtime *rt, const T *value)
      {
        return StructHandler<T,std::is_class<T>::value>::from_value(rt, value);
      }

      template<typename T>
      static inline T unpack(const Future &f, bool silence_warnings,
                             const char *warning_string)
      {
        return StructHandler<T,std::is_class<T>::value>::unpack(f, 
                            silence_warnings, warning_string);
      }

      // Some more help for reduction operations with RHS types
      // that have serialize and deserialize methods

      template<typename REDOP_RHS>
      static void serdez_redop_init(const ReductionOp *reduction_op,
                              void *&ptr, size_t &size)
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
      static void serdez_redop_fold(const ReductionOp *reduction_op,
                                    void *&lhs_ptr, size_t &lhs_size,
                                    const void *rhs_ptr)
      {
        REDOP_RHS lhs_serdez, rhs_serdez;
        lhs_serdez.legion_deserialize(lhs_ptr);
        rhs_serdez.legion_deserialize(rhs_ptr);
        (reduction_op->cpu_fold_excl_fn)(&lhs_serdez, 0, &rhs_serdez, 0,
                                         1, reduction_op->userdata);
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
        static inline void register_reduction(ReductionOp *redop,
                                              ReductionOpID redop_id,
                                              bool permit_duplicates)
        {
          Runtime::register_reduction_op(redop_id, redop, NULL, NULL, 
                                         permit_duplicates);
        }
      };
      // True case of template specialization
      template<typename REDOP_RHS>
      struct SerdezRedopHandler<REDOP_RHS,true> {
        static inline void register_reduction(ReductionOp *redop,
                                              ReductionOpID redop_id,
                                              bool permit_duplicates)
        {
          Runtime::register_reduction_op(redop_id, redop,
              serdez_redop_init<REDOP_RHS>, 
              serdez_redop_fold<REDOP_RHS>, permit_duplicates);
        }
      };

      template<typename REDOP_RHS, bool IS_STRUCT>
      struct StructRedopHandler {
        static inline void register_reduction(ReductionOp *redop,
                                              ReductionOpID redop_id,
                                              bool permit_duplicates)
        {
          Runtime::register_reduction_op(redop_id, redop, NULL, NULL, 
                                         permit_duplicates);
        }
      };
      // True case of template specialization
      template<typename REDOP_RHS>
      struct StructRedopHandler<REDOP_RHS,true> {
        static inline void register_reduction(ReductionOp *redop,
                                              ReductionOpID redop_id,
                                              bool permit_duplicates)
        {
          SerdezRedopHandler<REDOP_RHS,IsSerdezType<REDOP_RHS>::value>::
            register_reduction(redop, redop_id, permit_duplicates);
        }
      };

      // Register reduction functions if necessary
      template<typename REDOP>
      static inline void register_reduction(ReductionOpID redop_id,
                                            bool permit_duplicates)
      {
        StructRedopHandler<typename REDOP::RHS, 
          std::is_class<typename REDOP::RHS>::value>::register_reduction(
              Realm::ReductionOpUntyped::create_reduction_op<REDOP>(),
              redop_id, permit_duplicates);
      }

      template<typename T>
      struct HasSerdezBound {
        typedef char yes; typedef long no;

        template <typename C>
        static yes test(decltype(&C::legion_upper_bound_size));
        template <typename C> static no test(...);

        static constexpr bool value =
          sizeof(test<T>(nullptr)) == sizeof(yes);
      };

      template<typename T, bool HAS_BOUND>
      struct SerdezBound {
        static constexpr size_t value = T::legion_upper_bound_size();
      };

      template<typename T>
      struct SerdezBound<T,false> {
        static constexpr size_t value = LEGION_MAX_RETURN_SIZE;
      };

      template<typename T>
      struct SizeBound {
        static constexpr size_t value = sizeof(T);
      };

      template<typename T>
      struct ReturnSize {
        static constexpr size_t value = 
          std::conditional<IsSerdezType<T>::value,
           SerdezBound<T,HasSerdezBound<T>::value>, SizeBound<T> >::type::value;
      };

    }; // Serialization namespace

    // Special namespace for providing multi-dimensional 
    // array syntax on accessors 
    namespace ArraySyntax {
      // A helper class for handling reductions
      template<typename A, typename FT, int N, typename T>
      class ReductionHelper {
      public:
        __CUDA_HD__
        ReductionHelper(const A &acc, const Point<N> &p)
          : accessor(acc), point(p) { }
      public:
        __CUDA_HD__
        inline void reduce(FT val) const
        {
          accessor.reduce(point, val);
        }
        __CUDA_HD__
        inline void operator<<=(FT val) const
        {
          accessor.reduce(point, val);
        }
      public:
        const A &accessor;
        const Point<N,T> point;
      };

      template<typename FT, PrivilegeMode P>
      class AccessorRefHelper {
      public:
        AccessorRefHelper(const Realm::AccessorRefHelper<FT> &h) : helper(h) { }
      public:
        // read
        inline operator FT(void) const { return helper; }
        // writes
        inline AccessorRefHelper<FT,P>& operator=(const FT &newval)
          { helper = newval; return *this; }
        template<PrivilegeMode P2>
        inline AccessorRefHelper<FT,P>& operator=(
                const AccessorRefHelper<FT,P2> &rhs)
          { helper = rhs.helper; return *this; }
      protected:
        template<typename T, PrivilegeMode P2>
        friend class AccessorRefHelper;
        Realm::AccessorRefHelper<FT> helper;
      };

      template<typename FT>
      class AccessorRefHelper<FT,LEGION_READ_ONLY> {
      public:
        AccessorRefHelper(const Realm::AccessorRefHelper<FT> &h) : helper(h) { }
        // read
        inline operator FT(void) const { return helper; }
      private:
        // no writes allowed
        inline AccessorRefHelper<FT,LEGION_READ_ONLY>& operator=(
                const AccessorRefHelper<FT,LEGION_READ_ONLY> &rhs)
          { helper = rhs.helper; return *this; }
      protected:
        template<typename T, PrivilegeMode P2>
        friend class AccessorRefHelper;
        Realm::AccessorRefHelper<FT> helper;
      };

      // LEGION_NO_ACCESS means we dynamically check the privilege
      template<typename FT>
      class AccessorRefHelper<FT,LEGION_NO_ACCESS> {
      public:
        AccessorRefHelper(const Realm::AccessorRefHelper<FT> &h, FieldID fid,
                          const DomainPoint &pt, PrivilegeMode p)
          : helper(h), point(pt), field(fid), privilege(p) { }
      public:
        // read
        inline operator FT(void) const 
          { 
            if ((privilege & LEGION_READ_PRIV) == 0)
              PhysicalRegion::fail_privilege_check(point, field, privilege);
            return helper; 
          }
        // writes
        inline AccessorRefHelper<FT,LEGION_NO_ACCESS>& operator=(
                const FT &newval)
          { 
            if ((privilege & LEGION_WRITE_PRIV) == 0)
              PhysicalRegion::fail_privilege_check(point, field, privilege);
            helper = newval; 
            return *this; 
          }
        template<PrivilegeMode P2>
        inline AccessorRefHelper<FT,LEGION_NO_ACCESS>& operator=(
                const AccessorRefHelper<FT,P2> &rhs)
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
      template<typename A, typename FT, int N, typename T, 
                int M, PrivilegeMode P>
      class GenericSyntaxHelper {
      public:
        GenericSyntaxHelper(const A &acc, const Point<M-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (M-1); i++)
            point[i] = p[i];
        }
      public:
        inline GenericSyntaxHelper<A,FT,N,T,M+1,P> operator[](T val)
        {
          point[M-1] = val;
          return GenericSyntaxHelper<A,FT,N,T,M+1,P>(accessor, point);
        }
      public:
        const A &accessor;
        Point<M,T> point;
      };
      // Specialization for M = N
      template<typename A, typename FT, int N, typename T, PrivilegeMode P>
      class GenericSyntaxHelper<A,FT,N,T,N,P> {
      public:
        GenericSyntaxHelper(const A &acc, const Point<N-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (N-1); i++)
            point[i] = p[i];
        }
      public:
        inline AccessorRefHelper<FT,P> operator[](T val)
        {
          point[N-1] = val;
          return accessor[point];
        }
      public:
        const A &accessor;
        Point<N,T> point;
      };
      // Further specialization for M = N and read-only
      template<typename A, typename FT, int N, typename T>
      class GenericSyntaxHelper<A,FT,N,T,N,LEGION_READ_ONLY> {
      public:
        GenericSyntaxHelper(const A &acc, const Point<N-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (N-1); i++)
            point[i] = p[i];
        }
      public:
        inline AccessorRefHelper<FT,LEGION_READ_ONLY> operator[](T val)
        {
          point[N-1] = val;
          return accessor[point];
        }
      public:
        const A &accessor;
        Point<N,T> point;
      };
      // Further specialization for M = N and reductions
      template<typename A, typename FT, int N, typename T>
      class GenericSyntaxHelper<A,FT,N,T,N,LEGION_REDUCE> {
      public:
        GenericSyntaxHelper(const A &acc, const Point<N-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (N-1); i++)
            point[i] = p[i];
        }
      public:
        inline const ReductionHelper<A,FT,N,T> operator[](T val)
        {
          point[N-1] = val;
          return ReductionHelper<A,FT,N,T>(accessor, point);
        }
      public:
        const A &accessor;
        Point<N,T> point;
      };

      // A small helper class that helps provide some syntactic sugar for
      // indexing accessors like a multi-dimensional array for affine accessors
      template<typename A, typename FT, int N, typename T, 
                int M, PrivilegeMode P>
      class AffineSyntaxHelper {
      public:
        __CUDA_HD__
        AffineSyntaxHelper(const A &acc, const Point<M-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (M-1); i++)
            point[i] = p[i];
        }
      public:
        __CUDA_HD__
        inline AffineSyntaxHelper<A,FT,N,T,M+1,P> operator[](T val)
        {
          point[M-1] = val;
          return AffineSyntaxHelper<A,FT,N,T,M+1,P>(accessor, point);
        }
      public:
        const A &accessor;
        Point<M,T> point;
      };

      // Specialization for M = N
      template<typename A, typename FT, int N, typename T, PrivilegeMode P>
      class AffineSyntaxHelper<A,FT,N,T,N,P> {
      public:
        __CUDA_HD__
        AffineSyntaxHelper(const A &acc, const Point<N-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (N-1); i++)
            point[i] = p[i];
        }
      public:
        __CUDA_HD__
        inline FT& operator[](T val)
        {
          point[N-1] = val;
          return accessor[point];
        }
      public:
        const A &accessor;
        Point<N,T> point;
      };

      // Further specialization for M = N and read-only
      template<typename A, typename FT, int N, typename T>
      class AffineSyntaxHelper<A,FT,N,T,N,LEGION_READ_ONLY> {
      public:
        __CUDA_HD__
        AffineSyntaxHelper(const A &acc, const Point<N-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (N-1); i++)
            point[i] = p[i];
        }
      public:
        __CUDA_HD__
        inline const FT& operator[](T val)
        {
          point[N-1] = val;
          return accessor[point];
        }
      public:
        const A &accessor;
        Point<N,T> point;
      }; 

      // Further specialize for M = N and reductions
      template<typename A, typename FT, int N, typename T>
      class AffineSyntaxHelper<A,FT,N,T,N,LEGION_REDUCE> {
      public:
        __CUDA_HD__
        AffineSyntaxHelper(const A &acc, const Point<N-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (N-1); i++)
            point[i] = p[i];
        }
      public:
        __CUDA_HD__
        inline const ReductionHelper<A,FT,N,T> operator[](T val)
        {
          point[N-1] = val;
          return ReductionHelper<A,FT,N,T>(accessor, point);
        }
      public:
        const A &accessor;
        Point<N,T> point;
      };

      // Helper class for affine syntax that behaves like a
      // pointer/reference, but does dynamic privilege checks
      template<typename FT>
      class AffineRefHelper {
      public:
        __CUDA_HD__
        AffineRefHelper(FT &r,FieldID fid,const DomainPoint &pt,PrivilegeMode p)
          : ref(r),
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
            point(pt), field(fid), 
#endif
            privilege(p) { }
      public:
        // read
        __CUDA_HD__
        inline operator const FT&(void) const
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
        __CUDA_HD__
        inline AffineRefHelper<FT>& operator=(const FT &newval)
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
        __CUDA_HD__
        inline AffineRefHelper<FT>& operator=(const AffineRefHelper<FT> &rhs)
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
        FT &ref;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
        DomainPoint point;
        FieldID field;
#endif
        PrivilegeMode privilege;
      };

      // Further specialization for M = N and NO_ACCESS (dynamic privilege)
      template<typename A, typename FT, int N, typename T>
      class AffineSyntaxHelper<A,FT,N,T,N,LEGION_NO_ACCESS> {
      public:
        __CUDA_HD__
        AffineSyntaxHelper(const A &acc, const Point<N-1,T> &p)
          : accessor(acc)
        {
          for (int i = 0; i < (N-1); i++)
            point[i] = p[i];
        }
      public:
        __CUDA_HD__
        inline AffineRefHelper<FT> operator[](T val)
        {
          point[N-1] = val;
          return accessor[point];
        }
      public:
        const A &accessor;
        Point<N,T> point;
      };
    };

    ////////////////////////////////////////////////////////////
    // Specializations for Generic Accessors
    ////////////////////////////////////////////////////////////

    // Read-only FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline FT read(const Point<N,T>& p) const 
        { 
          return accessor.read(p); 
        }
      inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY> 
          operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(
                                                          accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
            FieldAccessor<LEGION_READ_ONLY,FT,N,T,
               Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size, 
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid), bounds(source_bounds)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bounds);
      }
    public:
      inline FT read(const Point<N,T>& p) const 
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return accessor.read(p); 
        }
      inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>
          operator[](const Point<N,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(
                                                          accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
             Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
            FieldAccessor<LEGION_READ_ONLY,FT,N,T,
              Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
      FieldID field;
      DomainT<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline FT read(const Point<1,T>& p) const 
        { 
          return accessor.read(p); 
        }
      inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>
          operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(
                                                          accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size, 
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, source_bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bounds);
      }
    public:
      inline FT read(const Point<1,T>& p) const 
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return accessor.read(p); 
        }
      inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY> 
          operator[](const Point<1,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(
                                                          accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
      FieldID field;
      DomainT<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-write FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_READ_WRITE,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline FT read(const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE> 
          operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_READ_WRITE,FT,N,T,
             Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
            FieldAccessor<LEGION_READ_WRITE,FT,N,T,
              Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_WRITE>(
              *this, Point<1,T>(index));
      }
      // No reductions since we can't handle atomicity correctly
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-write FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_READ_WRITE,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid, actual_field_size, 
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid, actual_field_size, 
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bounds);
      }
    public:
      inline FT read(const Point<N,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return accessor.read(p); 
        }
      inline void write(const Point<N,T>& p, FT val) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE> 
          operator[](const Point<N,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_READ_WRITE,FT,N,T,
              Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
            FieldAccessor<LEGION_READ_WRITE,FT,N,T,
              Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_WRITE>(
              *this, Point<1,T>(index));
      }
      // No reductions since we can't handle atomicity correctly
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
      FieldID field;
      DomainT<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-write FieldAccessor specialization 
    // with N==1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_READ_WRITE,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline FT read(const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE> 
          operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
      // No reductions since we can't handle atomicity correctly
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-write FieldAccessor specialization 
    // with N==1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_READ_WRITE,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid, actual_field_size, 
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid, actual_field_size, 
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, source_bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bounds);
      }
    public:
      inline FT read(const Point<1,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return accessor.read(p); 
        }
      inline void write(const Point<1,T>& p, FT val) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE> 
          operator[](const Point<1,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
      // No reduction since we can't handle atomicity correctly
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
      FieldID field;
      DomainT<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-discard FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size, 
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline FT read(const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD> 
          operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-discard FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bounds);
      }
    public:
      inline FT read(const Point<N,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return accessor.read(p); 
        }
      inline void write(const Point<N,T>& p, FT val) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>
          operator[](const Point<N,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
      FieldID field;
      DomainT<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-discard FieldAccessor specialization with
    // N == 1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline FT read(const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD> 
          operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-discard FieldAccessor specialization with
    // N == 1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid,actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds.bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bounds);
      }
    public:
      inline FT read(const Point<1,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
          return accessor.read(p); 
        }
      inline void write(const Point<1,T>& p, FT val) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD> 
          operator[](const Point<1,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
      FieldID field;
      DomainT<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-only FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD> 
          operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-only FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,N,T,
                        Realm::GenericAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bound);
      }
    public:
      inline void write(const Point<N,T>& p, FT val) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>
          operator[](const Point<N,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::GenericAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
      FieldID field;
      DomainT<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-only FieldAccessor specialization with
    // N == 1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size, 
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
    public:
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>
          operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-only FieldAccessor specialization with
    // N == 1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,1,T,
                        Realm::GenericAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           bounds.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds.bounds, offset);
      }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &bounds, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           source_bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, source_bounds, offset);
        bounds.bounds = source_bounds.intersection(bounds.bounds);
      }
    public:
      inline void write(const Point<1,T>& p, FT val) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
          accessor.write(p, val); 
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>
          operator[](const Point<1,T>& p) const
        { 
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_WRITE_DISCARD>(
                                                              accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
      FieldID field;
      DomainT<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Special namespace for providing bounds check help for affine accessors
    namespace AffineBounds {
      // A helper class for testing bounds for affine accessors
      // which might have a transform associated with them
      // We should never use the base version of this, only the specializations
      template<int N, typename T>
      class Tester {
      public:
        Tester(void) : M(0) { }
        Tester(const DomainT<N,T> b) 
          : bounds(b), M(N), has_source(false), 
            has_transform(false) { }
        Tester(const DomainT<N,T> b, const Rect<N,T> s)
          : bounds(b), source(s), M(N), has_source(true), 
            has_transform(false) { }
        template<int M2>
        Tester(const DomainT<M2,T> b,
               const AffineTransform<M2,N,T> t) 
          : bounds(b), transform(t), M(M2), has_source(false), 
            has_transform(!t.is_identity())
        { 
          LEGION_STATIC_ASSERT(M2 <= LEGION_MAX_DIM,
              "Accessor DIM larger than LEGION_MAX_DIM");
        }
        template<int M2>
        Tester(const DomainT<M2,T> b, const Rect<N,T> s,
               const AffineTransform<M2,N,T> t) 
          : bounds(b), transform(t), source(s), M(M2), has_source(true), 
            has_transform(!t.is_identity())
        { 
          LEGION_STATIC_ASSERT(M2 <= LEGION_MAX_DIM,
              "Accessor DIM larger than LEGION_MAX_DIM");
        }
      public:
        __CUDA_HD__
        inline bool contains(const Point<N,T> &p) const
        {
          if (has_source && !source.contains(p))
            return false;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          check_gpu_warning();
          // Note that in CUDA this function is likely being inlined
          // everywhere and we can't afford to instantiate templates
          // for every single dimension so do things untyped
          if (!has_transform)
          {
            // This is imprecise because we can't see the 
            // Realm index space on the GPU
            const Rect<N,T> b = bounds.bounds<N,T>();
            return b.contains(p);
          }
          else
            return bounds.contains_bounds_only(transform[DomainPoint(p)]);
#else
          if (!has_transform)
          {
            const DomainT<N,T> b = bounds;
            return b.contains(p);
          }
          switch (M)
          {
#define DIMFUNC(DIM) \
            case DIM: \
              { \
                const DomainT<DIM,T> b = bounds; \
                const AffineTransform<DIM,N,T> t = transform; \
                return b.contains(t[p]); \
              }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              assert(false);
          }
          return false;
#endif
        }
        __CUDA_HD__
        inline bool contains_all(const Rect<N,T> &r) const
        {
          if (has_source && !source.contains(r))
            return false;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          check_gpu_warning();
          // Note that in CUDA this function is likely being inlined
          // everywhere and we can't afford to instantiate templates
          // for every single dimension so do things untyped
          if (has_transform)
          {
            for (PointInRectIterator<N,T> itr(r); itr(); itr++)
              if (!bounds.contains_bounds_only(transform[DomainPoint(*itr)]))
                return false;
            return true;
          }
          else
          {
            // This is imprecise because we can't see the 
            // Realm index space on the GPU
            const Rect<N,T> b = bounds.bounds<N,T>();
            return b.contains(r);
          }
#else
          if (!has_transform)
          {
            const DomainT<N,T> b = bounds;
            return b.contains_all(r);
          }
          // If we have a transform then we have to do each point separately
          switch (M)
          {
#define DIMFUNC(DIM) \
            case DIM: \
              { \
                const DomainT<DIM,T> b = bounds; \
                const AffineTransform<DIM,N,T> t = transform; \
                for (PointInRectIterator<N,T> itr(r); itr(); itr++) \
                  if (!b.contains(t[*itr])) \
                    return false; \
                return true; \
              }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              assert(false);
          }
          return false;
#endif
        }
      private:
        __CUDA_HD__
        inline void check_gpu_warning(void) const
        {
#if 0
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          bool need_warning = !bounds.dense();
          if (need_warning)
            printf("WARNING: GPU bounds check is imprecise!\n");
#endif
#endif
        }
      private:
        Domain bounds;
        DomainAffineTransform transform;
        Rect<N,T> source;
        int M;
        bool has_source;
        bool has_transform;
      };
    };

    // Some helper methods for accessors and deferred buffers
    namespace Internal {
      template<int N, typename T> __CUDA_HD__
      static inline bool is_dense_layout(const Rect<N,T> &bounds,
                                const size_t strides[N], size_t field_size)
      {
        ptrdiff_t exp_offset = field_size;
        int used_mask = 0; // keep track of the dimensions we've already matched
        static_assert((N <= (8*sizeof(used_mask))), "Mask dim exceeded");
        for (int i = 0; i < N; i++) {
          bool found = false;
          for (int j = 0; j < N; j++) {
            if ((used_mask >> j) & 1) continue;
            if (strides[j] != exp_offset) 
            {
              // Mask off any dimensions with stride 0
              if (strides[j] == 0)
              {
                if (bounds.lo[j] != bounds.hi[j])
                  return false;
                used_mask |= (1 << j);
                if (++i == N) 
                {
                  found = true;
                  break;
                }
              }
              continue;
            }
            found = true;
            // It's possible other dimensions can have the same strides if
            // there are multiple dimensions with extents of size 1. At most
            // one dimension can have an extent >1 though
            int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
            for (int k = j+1; k < N; k++) {
              if ((used_mask >> k) & 1) continue;
              if (strides[k] == exp_offset) {
                if (bounds.lo[k] < bounds.hi[k]) {
                  // if we already saw a non-trivial dimension this is bad
                  if (nontrivial >= 0)
                    return false;
                  else
                    nontrivial = k;
                }
                used_mask |= (1 << k);
                i++;
              }
            }
            used_mask |= (1 << j);
            if (nontrivial >= 0)
              exp_offset *= (bounds.hi[nontrivial] - bounds.lo[nontrivial] + 1);
            break;
          }
          if (!found)
            return false;
        }
        return true;
      }

      // Same method as above but for realm points from affine accessors
      template<int N, typename T> __CUDA_HD__
      static inline bool is_dense_layout(const Rect<N,T> &bounds,
                  const Realm::Point<N,size_t> &strides, size_t field_size)
      {
        size_t exp_offset = field_size;
        int used_mask = 0; // keep track of the dimensions we've already matched
        static_assert((N <= (8*sizeof(used_mask))), "Mask dim exceeded");
        for (int i = 0; i < N; i++) {
          bool found = false;
          for (int j = 0; j < N; j++) {
            if ((used_mask >> j) & 1) continue;
            if (strides[j] != exp_offset) 
            {
              // Mask off any dimensions with stride 0
              if (strides[j] == 0) 
              {
                if (bounds.lo[j] != bounds.hi[j])
                  return false;
                used_mask |= (1 << j);
                if (++i == N) 
                {
                  found = true;
                  break;
                }
              }
              continue;
            }
            found = true;
            // It's possible other dimensions can have the same strides if
            // there are multiple dimensions with extents of size 1. At most
            // one dimension can have an extent >1 though
            int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
            for (int k = j+1; k < N; k++) {
              if ((used_mask >> k) & 1) continue;
              if (strides[k] == exp_offset) {
                if (bounds.lo[k] < bounds.hi[k]) {
                  // if we already saw a non-trivial dimension this is bad
                  if (nontrivial >= 0)
                    return false;
                  else
                    nontrivial = k;
                }
                used_mask |= (1 << k);
                i++;
              }
            }
            used_mask |= (1 << j);
            if (nontrivial >= 0)
              exp_offset *= (bounds.hi[nontrivial] - bounds.lo[nontrivial] + 1);
            break;
          }
          if (!found)
            return false;
        }
        return true;
      }
    }

    ////////////////////////////////////////////////////////////
    // Macros UntypedDeferredValue/UntypedDeferredBuffer 
    // Constructors with Affine Accessors
    ////////////////////////////////////////////////////////////

#define DEFERRED_VALUE_BUFFER_CONSTRUCTORS(DIM, FIELD_CHECK)                  \
      FieldAccessor(const UntypedDeferredValue &value,                        \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        Realm::Rect<DIM,T> source_bounds;                                     \
        /* Anything in range works for these bounds since we're */            \
        /* going to remap them to the origin */                               \
        for (int i = 0; i < DIM; i++)                                         \
        {                                                                     \
          source_bounds.lo[i] = std::numeric_limits<T>::min();                \
          source_bounds.hi[i] = std::numeric_limits<T>::max();                \
        }                                                                     \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform, origin, 0/*field id*/, source_bounds))               \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<FT,DIM,T>(instance, transform,       \
                          origin, 0/*field id*/, source_bounds, offset);      \
      }                                                                       \
      FieldAccessor(const UntypedDeferredValue &value,                        \
                    const Rect<DIM,T> &source_bounds,                         \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform, origin, 0/*field id*/, source_bounds))               \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<FT,DIM,T>(instance, transform,       \
                          origin, 0/*field id*/, source_bounds, offset);      \
      }                                                                       \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
                                            0/*field id*/, is.bounds))        \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<FT,DIM,T>(instance, 0/*field id*/,            \
                                          is.bounds, offset);                 \
      }                                                                       \
      /* With explicit bounds */                                              \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    const Rect<DIM,T> &source_bounds,                         \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
                                        0/*field id*/, source_bounds))        \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<FT,DIM,T>(instance, 0/*field id*/,            \
                                          source_bounds, offset);             \
      }                                                                       \
      /* With explicit transform */                                           \
      template<int M>                                                         \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    const AffineTransform<M,DIM,T> &transform,                \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform.transform, transform.offset, 0/*field id*/))          \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<FT,DIM,T>(instance,                  \
            transform.transform, transform.offset, 0/*field id*/, offset);    \
      }                                                                       \
      /* With explicit transform and bounds */                                \
      template<int M>                                                         \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    const AffineTransform<M,DIM,T> &transform,                \
                    const Rect<DIM,T> &source_bounds,                         \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform.transform, transform.offset, 0/*fid*/, source_bounds))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<FT,DIM,T>(instance, transform.transform,      \
                      transform.offset, 0/*fid*/, source_bounds, offset);     \
      }

#define DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(DIM, FIELD_CHECK)      \
      FieldAccessor(const UntypedDeferredValue &value,                        \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        Realm::Rect<DIM,T> source_bounds;                                     \
        /* Anything in range works for these bounds since we're */            \
        /* going to remap them to the origin */                               \
        for (int i = 0; i < DIM; i++)                                         \
        {                                                                     \
          source_bounds.lo[i] = std::numeric_limits<T>::min();                \
          source_bounds.hi[i] = std::numeric_limits<T>::max();                \
        }                                                                     \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform, origin, 0/*field id*/, source_bounds))               \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<FT,DIM,T>(instance, transform,       \
                          origin, 0/*field id*/, source_bounds, offset);      \
        DomainT<1,T> is;                                                      \
        is.bounds.lo[0] = 0;                                                  \
        is.bounds.hi[0] = 0;                                                  \
        is.sparsity.id = 0;                                                   \
        AffineTransform<1,DIM,T> affine(transform, origin);                   \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds, affine);      \
      }                                                                       \
      FieldAccessor(const UntypedDeferredValue &value,                        \
                    const Rect<DIM,T> &source_bounds,                         \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform, origin, 0/*field id*/, source_bounds))               \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<FT,DIM,T>(instance, transform,       \
                          origin, 0/*field id*/, source_bounds, offset);      \
        DomainT<1,T> is;                                                      \
        is.bounds.lo[0] = 0;                                                  \
        is.bounds.hi[0] = 0;                                                  \
        is.sparsity.id = 0;                                                   \
        AffineTransform<1,DIM,T> affine(transform, origin);                   \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds, affine);      \
      }                                                                       \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
                                            0/*field id*/, is.bounds))        \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<FT,DIM,T>(instance, 0/*field id*/,            \
                                          is.bounds, offset);                 \
        bounds = AffineBounds::Tester<DIM,T>(is);                             \
      }                                                                       \
      /* With explicit bounds */                                              \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    const Rect<DIM,T> &source_bounds,                         \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
                                        0/*field id*/, source_bounds))        \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<FT,DIM,T>(instance, 0/*field id*/,            \
                                          source_bounds, offset);             \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds);              \
      }                                                                       \
      /* With explicit transform */                                           \
      template<int M>                                                         \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    const AffineTransform<M,DIM,T> &transform,                \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform.transform, transform.offset, 0/*field id*/))          \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<FT,DIM,T>(instance,                  \
            transform.transform, transform.offset, 0/*field id*/, offset);    \
        bounds = AffineBounds::Tester<DIM,T>(is, transform);                  \
      }                                                                       \
      /* With explicit transform and bounds */                                \
      template<int M>                                                         \
      FieldAccessor(const UntypedDeferredBuffer<T> &buffer,                   \
                    const AffineTransform<M,DIM,T> &transform,                \
                    const Rect<DIM,T> &source_bounds,                         \
                    size_t actual_field_size = sizeof(FT),                    \
                    bool check_field_size = FIELD_CHECK,                      \
                    bool silence_warnings = false,                            \
                    const char *warning_string = NULL,                        \
                    size_t offset = 0)                                        \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance,         \
              transform.transform, transform.offset, 0/*fid*/, source_bounds))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<FT,DIM,T>(instance, transform.transform,      \
                      transform.offset, 0/*fid*/, source_bounds, offset);     \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds, transform);   \
      }

    ////////////////////////////////////////////////////////////
    // Specializations for Affine Accessors
    ////////////////////////////////////////////////////////////

    // Read-only FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true) 
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const 
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, 
                           size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, size_t strides[N],
                           size_t field_size = sizeof(FT)) const
        {
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
            Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
            FieldAccessor<LEGION_READ_ONLY,FT,N,T,
               Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<N,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const 
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, 
                           size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, size_t strides[N],
                           size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
#endif
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
             Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
            FieldAccessor<LEGION_READ_ONLY,FT,N,T,
              Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const 
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r, 
                           size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r, size_t strides[1],
                           size_t field_size = sizeof(FT)) const
        {
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo); 
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid) 
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid) 
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid) 
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<1,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid) 
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const 
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r,
                           size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r, size_t strides[1],
                           size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
#endif
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor[p]; 
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-write FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_READ_WRITE,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true) 
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_WRITE,FT,N,T,
             Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
            FieldAccessor<LEGION_READ_WRITE,FT,N,T,
              Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_WRITE>(
              *this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-write FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_READ_WRITE,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform, 
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<N,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform, 
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
#endif
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_WRITE,FT,N,T,
              Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
            FieldAccessor<LEGION_READ_WRITE,FT,N,T,
               Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_WRITE>(
              *this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__ 
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-write FieldAccessor specialization 
    // with N==1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_READ_WRITE,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-write FieldAccessor specialization 
    // with N==1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_READ_WRITE,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<1,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
#endif
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-discard FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size, 
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
             Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-discard FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<N,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const 
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
#endif
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
             Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-discard FieldAccessor specialization with
    // N == 1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-discard FieldAccessor specialization with
    // N == 1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<1,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
#endif
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-only FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(N, false)
#endif
    public:
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
             Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-only FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<N,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,T> transform,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
    public:
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const 
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
#endif
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
             Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-only FieldAccessor specialization with
    // N == 1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS(1, false)
#endif
    public:
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-only FieldAccessor specialization with
    // N == 1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
        bounds = AffineBounds::Tester<1,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,1,T> transform,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<M,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
      DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
    public:
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
#endif
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

#undef DEFERRED_VALUE_BUFFER_CONSTRUCTORS
#undef DEFERRED_VALUE_BUFFER_CONSTRUCTORS_WITH_BOUNDS

#define DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(DIM, FIELD_CHECK)        \
      ReductionAccessor(const UntypedDeferredValue &value,                    \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        Realm::Rect<DIM,T> source_bounds;                                     \
        /* Anything in range works for these bounds since we're */            \
        /* going to remap them to the origin */                               \
        for (int i = 0; i < DIM; i++)                                         \
        {                                                                     \
          source_bounds.lo[i] = std::numeric_limits<T>::min();                \
          source_bounds.hi[i] = std::numeric_limits<T>::max();                \
        }                                                                     \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform, origin, 0/*field id*/, source_bounds))     \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(          \
          instance, transform, origin, 0/*field id*/, source_bounds, offset); \
      }                                                                       \
      ReductionAccessor(const UntypedDeferredValue &value,                    \
                        const Rect<DIM,T> &source_bounds,                     \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform, origin, 0/*field id*/, source_bounds))     \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance, \
            transform, origin, 0/*field id*/, source_bounds, offset);         \
      }                                                                       \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
                                          instance, 0/*field id*/, is.bounds))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance,          \
                                  0/*field id*/, is.bounds, offset);          \
      }                                                                       \
      /* With explicit bounds */                                              \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        const Rect<DIM,T> &source_bounds,                     \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
                                      instance, 0/*field id*/, source_bounds))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance,          \
                              0/*field id*/, source_bounds, offset);          \
      }                                                                       \
      /* With explicit transform */                                           \
      template<int M>                                                         \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        const AffineTransform<M,DIM,T> &transform,            \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform.transform, transform.offset, 0/*field id*/))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance, \
            transform.transform, transform.offset, 0/*field id*/, offset);    \
      }                                                                       \
      /* With explicit transform and bounds */                                \
      template<int M>                                                         \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        const AffineTransform<M,DIM,T> &transform,            \
                        const Rect<DIM,T> &source_bounds,                     \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform.transform, transform.offset,                \
              0/*field id*/, source_bounds))                                  \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance,          \
              transform.transform, transform.offset, 0/*field id*/,           \
              source_bounds, offset);                                         \
      }

#define DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(DIM,FIELD_CHECK) \
      ReductionAccessor(const UntypedDeferredValue &value,                    \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        Realm::Rect<DIM,T> source_bounds;                                     \
        /* Anything in range works for these bounds since we're */            \
        /* going to remap them to the origin */                               \
        for (int i = 0; i < DIM; i++)                                         \
        {                                                                     \
          source_bounds.lo[i] = std::numeric_limits<T>::min();                \
          source_bounds.hi[i] = std::numeric_limits<T>::max();                \
        }                                                                     \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform, origin, 0/*field id*/, source_bounds))     \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance, \
            transform, origin, 0/*field id*/, source_bounds, offset);         \
        DomainT<1,T> is;                                                      \
        is.bounds.lo[0] = 0;                                                  \
        is.bounds.hi[0] = 0;                                                  \
        is.sparsity.id = 0;                                                   \
        AffineTransform<1,DIM,T> affine(transform, origin);                   \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds, affine);      \
      }                                                                       \
      ReductionAccessor(const UntypedDeferredValue &value,                    \
                        const Rect<DIM,T> &source_bounds,                     \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        assert(!check_field_size || (actual_field_size == value.field_size)); \
        const Realm::RegionInstance instance = value.instance;                \
        /* This mapping ignores the input points and sends */                 \
        /* everything to the 1-D origin */                                    \
        Realm::Matrix<1,DIM,T> transform;                                     \
        for (int i = 0; i < DIM; i++)                                         \
          transform[0][i] = 0;                                                \
        Realm::Point<1,T> origin(0);                                          \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform, origin, 0/*field id*/, source_bounds))     \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredValue\n");      \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance, \
            transform, origin, 0/*field id*/, source_bounds, offset);         \
        DomainT<1,T> is;                                                      \
        is.bounds.lo[0] = 0;                                                  \
        is.bounds.hi[0] = 0;                                                  \
        is.sparsity.id = 0;                                                   \
        AffineTransform<1,DIM,T> affine(transform, origin);                   \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds, affine);      \
      }                                                                       \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
                                          instance, 0/*field id*/, is.bounds))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance,          \
                                  0/*field id*/, is.bounds, offset);          \
        bounds = AffineBounds::Tester<DIM,T>(is);                             \
      }                                                                       \
      /* With explicit bounds */                                              \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        const Rect<DIM,T> &source_bounds,                     \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<DIM,T> is = instance.get_indexspace<DIM,T>();           \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
                                      instance, 0/*field id*/, source_bounds))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance,          \
                              0/*field id*/, source_bounds, offset);          \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds);              \
      }                                                                       \
      /* With explicit transform */                                           \
      template<int M>                                                         \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        const AffineTransform<M,DIM,T> &transform,            \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform.transform, transform.offset, 0/*field id*/))\
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor = Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance, \
            transform.transform, transform.offset, 0/*field id*/, offset);    \
        bounds = AffineBounds::Tester<DIM,T>(is, transform);                  \
      }                                                                       \
      /* With explicit transform and bounds */                                \
      template<int M>                                                         \
      ReductionAccessor(const UntypedDeferredBuffer<T> &buffer,               \
                        const AffineTransform<M,DIM,T> &transform,            \
                        const Rect<DIM,T> &source_bounds,                     \
                        bool silence_warnings = false,                        \
                        const char *warning_string = NULL,                    \
                        size_t offset = 0,                                    \
                        size_t actual_field_size=sizeof(typename REDOP::RHS), \
                        bool check_field_size = FIELD_CHECK)                  \
      {                                                                       \
        const Realm::RegionInstance instance = buffer.instance;               \
        const DomainT<M,T> is = instance.get_indexspace<M,T>();               \
        if (!Realm::AffineAccessor<typename REDOP::RHS,DIM,T>::is_compatible( \
              instance, transform.transform, transform.offset,                \
              0/*field id*/, source_bounds))                                  \
        {                                                                     \
          fprintf(stderr,                                                     \
              "Incompatible AffineAccessor for UntypedDeferredBuffer\n");     \
          assert(false);                                                      \
        }                                                                     \
        accessor =                                                            \
          Realm::AffineAccessor<typename REDOP::RHS,DIM,T>(instance,          \
              transform.transform, transform.offset, 0/*field id*/,           \
              source_bounds, offset);                                         \
        bounds = AffineBounds::Tester<DIM,T>(is, source_bounds, transform);   \
      }

    // Reduce FieldAccessor specialization
    template<typename REDOP, bool EXCLUSIVE, int N, typename T, bool CB>
    class ReductionAccessor<REDOP,EXCLUSIVE,N,T,
                        Realm::AffineAccessor<typename REDOP::RHS,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<N,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,N,T> transform,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(instance, 
            transform.transform, transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,N,T> transform,
                        const Rect<N,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance,transform.transform,transform.offset,fid,source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(instance, 
            transform.transform, transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(N, true)
#else
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(N, false)
#endif
    public:
      __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<N,T>& p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r, 
              size_t strides[N], size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,N,
             T,Realm::AffineAccessor<typename REDOP::RHS,N,T>,CB>,
             typename REDOP::RHS,N,T>
               operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            N,T,Realm::AffineAccessor<typename REDOP::RHS,N,T>,CB>,
            typename REDOP::RHS,N,T>(*this, p);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,EXCLUSIVE,
         N,T,Realm::AffineAccessor<typename REDOP::RHS,N,T>,CB>,
         typename REDOP::RHS,N,T,2,LEGION_REDUCE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,
          EXCLUSIVE,N,T,Realm::AffineAccessor<typename REDOP::RHS,N,T>,CB>,
          typename REDOP::RHS,N,T,2,LEGION_REDUCE>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<typename REDOP::RHS,N,T> accessor;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = N;
    };

    // Reduce ReductionAccessor specialization with bounds checks
    template<typename REDOP, bool EXCLUSIVE, int N, typename T>
    class ReductionAccessor<REDOP,EXCLUSIVE,N,T,
                          Realm::AffineAccessor<typename REDOP::RHS,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<N,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,N,T> transform,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(
            instance, transform.transform, transform.offset, fid, offset);
        bounds = AffineBounds::Tester<N,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,N,T> transform,
                        const Rect<N,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance,transform.transform,transform.offset,fid,source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,N,T>(instance,
            transform.transform, transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(N, true)
#else
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(N, false)
#endif
    public:
      __CUDA_HD__ 
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_REDUCE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r,
              size_t strides[N],
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_REDUCE);
#endif
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,N,
             T,Realm::AffineAccessor<typename REDOP::RHS,N,T>,true>,
             typename REDOP::RHS,N,T>
               operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            N,T,Realm::AffineAccessor<typename REDOP::RHS,N,T>,true>,
            typename REDOP::RHS,N,T>(*this, p);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,EXCLUSIVE,
             N,T, Realm::AffineAccessor<typename REDOP::RHS,N,T>,true>,
             typename REDOP::RHS,N,T,2,LEGION_REDUCE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,
          EXCLUSIVE,N,T,Realm::AffineAccessor<typename REDOP::RHS,N,T>,true>,
          typename REDOP::RHS,N,T,2,LEGION_REDUCE>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::AffineAccessor<typename REDOP::RHS,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = N;
    };
    
    // Reduce Field Accessor specialization with N==1
    // to avoid array ambiguity
    template<typename REDOP, bool EXCLUSIVE, typename T, bool CB>
    class ReductionAccessor<REDOP,EXCLUSIVE,1,T,
                        Realm::AffineAccessor<typename REDOP::RHS,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(
            instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<1,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(
            instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,1,T> transform,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE,fid,sizeof(typename REDOP::RHS),
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              false/*check field size*/, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(
            instance, transform.transform, transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,1,T> transform,
                        const Rect<1,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance,transform.transform,transform.offset,fid,source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(instance, 
            transform.transform, transform.offset, fid, source_bounds, offset);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(1, true)
#else
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS(1, false)
#endif
    public:
      __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<1,T>& p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r, 
              size_t strides[1],
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,1,
             T,Realm::AffineAccessor<typename REDOP::RHS,1,T>,CB>,
             typename REDOP::RHS,1,T>
               operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            1,T,Realm::AffineAccessor<typename REDOP::RHS,1,T>,CB>,
            typename REDOP::RHS,1,T>(*this, p);
        }
    public:
      Realm::AffineAccessor<typename REDOP::RHS,1,T> accessor;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = 1;
    };

    // Reduce Field Accessor specialization with N==1
    // to avoid array ambiguity and bounds checks
    template<typename REDOP, bool EXCLUSIVE, typename T>
    class ReductionAccessor<REDOP,EXCLUSIVE,1,T,
                        Realm::AffineAccessor<typename REDOP::RHS,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(
            instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<1,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(
              instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
      // With explicit transform
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,1,T> transform,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(instance,
            transform.transform, transform.offset, fid, offset);
        bounds = AffineBounds::Tester<1,T>(is, transform);
      }
      // With explicit transform and bounds
      template<int M>
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const AffineTransform<M,1,T> transform,
                        const Rect<1,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
#ifdef DEBUG_LEGION
                        bool check_field_size = true
#else
                        bool check_field_size = false
#endif
                       )
        : field(fid)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::AffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance,transform.transform,transform.offset,fid,source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<typename REDOP::RHS,1,T>(instance, 
            transform.transform, transform.offset, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds, transform);
      }
    public:
#ifdef DEBUG_LEGION
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(1, true)
#else
      DEFERRED_VALUE_BUFFER_REDUCTION_CONSTRUCTORS_WITH_BOUNDS(1, false)
#endif
    public:
      __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r,
              size_t strides[1],
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
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,1,
             T,Realm::AffineAccessor<typename REDOP::RHS,1,T>,true>,
             typename REDOP::RHS,1,T>
               operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            1,T,Realm::AffineAccessor<typename REDOP::RHS,1,T>,true>,
            typename REDOP::RHS,1,T>(*this, p);
        }
    public:
      Realm::AffineAccessor<typename REDOP::RHS,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = 1;
    };

    ////////////////////////////////////////////////////////////
    // Specializations for Multi Affine Accessors
    ////////////////////////////////////////////////////////////

    // Read-only FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const 
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, 
                           size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, size_t strides[N],
                           size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
            Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
            FieldAccessor<LEGION_READ_ONLY,FT,N,T,
              Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const 
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, 
                           size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<N,T>& r, size_t strides[N],
                           size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.prt(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
             Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
            Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const 
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r, 
                           size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r, size_t strides[1],
                           size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid) 
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid) 
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_ONLY, fid, actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const 
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r,
                           size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline const FT* ptr(const Rect<1,T>& r, size_t strides[1],
                           size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_ONLY);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline const FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor[p]; 
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-write FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_READ_WRITE,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_WRITE,FT,N,T,
             Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
            FieldAccessor<LEGION_READ_WRITE,FT,N,T,
             Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_WRITE>(
              *this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-write FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_READ_WRITE,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_WRITE,FT,N,T,
            Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_READ_WRITE,FT,N,T,
           Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_WRITE>(
              *this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__ 
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-write FieldAccessor specialization 
    // with N==1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_READ_WRITE,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-write FieldAccessor specialization 
    // with N==1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_READ_WRITE,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_READ_WRITE, fid,actual_field_size,&is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-discard FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size, 
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
           Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
         FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
          Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-discard FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const 
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
         Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,
              LEGION_WRITE_DISCARD>(*this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-discard FieldAccessor specialization with
    // N == 1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-discard FieldAccessor specialization with
    // N == 1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_WRITE_DISCARD,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                               instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_READ_ONLY);
#endif
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-only FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
           Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
         FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
          Realm::MultiAffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_WRITE_DISCARD>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-only FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,N,T,
                        Realm::MultiAffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<N,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const 
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
         Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_WRITE_DISCARD>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<
          FieldAccessor<LEGION_WRITE_DISCARD,FT,N,T,
            Realm::MultiAffineAccessor<FT,N,T>,true>,FT,N,T,2,
              LEGION_WRITE_DISCARD>(*this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Write-only FieldAccessor specialization with
    // N == 1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Write-only FieldAccessor specialization with
    // N == 1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_WRITE_ONLY,FT,1,T,
                        Realm::MultiAffineAccessor<FT,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<1,T> source_bounds,
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false,
                    const char *warning_string = NULL,
                    size_t offset = 0)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_WRITE_DISCARD, fid, actual_field_size,
              &is,Internal::NT_TemplateHelper::encode_tag<1,T>(),warning_string, 
              silence_warnings, false/*generic accessor*/, check_field_size);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p),
                                              field, LEGION_WRITE_DISCARD);
#endif
          accessor.write(p, val); 
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[1];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T>& r, size_t strides[1],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_READ_WRITE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                              LEGION_READ_WRITE);
#endif
          return accessor[p]; 
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Reduce FieldAccessor specialization
    template<typename REDOP, bool EXCLUSIVE, int N, typename T, bool CB>
    class ReductionAccessor<REDOP,EXCLUSIVE,N,T,
                      Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<N,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<N,T>& p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          size_t strides[N];
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r, 
              size_t strides[N],
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,N,
             T,Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,CB>,
             typename REDOP::RHS,N,T>
               operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            N,T,Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,CB>,
            typename REDOP::RHS,N,T>(*this, p);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,EXCLUSIVE,
         N,T,Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,CB>,
         typename REDOP::RHS,N,T,2,LEGION_REDUCE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,
          EXCLUSIVE,N,T,Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,CB>,
          typename REDOP::RHS,N,T,2,LEGION_REDUCE>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<typename REDOP::RHS,N,T> accessor;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = N;
    };

    // Reduce ReductionAccessor specialization with bounds checks
    template<typename REDOP, bool EXCLUSIVE, int N, typename T>
    class ReductionAccessor<REDOP,EXCLUSIVE,N,T,
                    Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<N,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
        : field(fid)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>(
            instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<N,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__ 
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<N,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          size_t strides[N];
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_REDUCE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<N,T>& r,
              size_t strides[N],
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, 
                                              LEGION_REDUCE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,N,
             T,Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,true>,
             typename REDOP::RHS,N,T>
               operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            N,T,Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,true>,
            typename REDOP::RHS,N,T>(*this, p);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,EXCLUSIVE,
             N,T, Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,true>,
             typename REDOP::RHS,N,T,2,LEGION_REDUCE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<ReductionAccessor<REDOP,
          EXCLUSIVE,N,T,Realm::MultiAffineAccessor<typename REDOP::RHS,N,T>,
            true>,typename REDOP::RHS,N,T,2,LEGION_REDUCE>(
              *this, Point<1,T>(index));
      }
    public:
      Realm::MultiAffineAccessor<typename REDOP::RHS,N,T> accessor;
      FieldID field;
      AffineBounds::Tester<N,T> bounds;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = N;
    };
    
    // Reduce Field Accessor specialization with N==1
    // to avoid array ambiguity
    template<typename REDOP, bool EXCLUSIVE, typename T, bool CB>
    class ReductionAccessor<REDOP,EXCLUSIVE,1,T,
                      Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>,CB> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      __CUDA_HD__
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>(
            instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<1,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>(
            instance, fid, source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<1,T>& p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          size_t strides[1];
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r, 
              size_t strides[1],
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,1,
             T,Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>,CB>,
             typename REDOP::RHS,1,T>
               operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            1,T,Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>,CB>,
            typename REDOP::RHS,1,T>(*this, p);
        }
    public:
      Realm::MultiAffineAccessor<typename REDOP::RHS,1,T> accessor;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = 1;
    };

    // Reduce Field Accessor specialization with N==1
    // to avoid array ambiguity and bounds checks
    template<typename REDOP, bool EXCLUSIVE, typename T>
    class ReductionAccessor<REDOP,EXCLUSIVE,1,T,
                    Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>,true> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      // No CUDA support due to PhysicalRegion constructor
      ReductionAccessor(void) { }
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>(
            instance, fid, is.bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is);
      }
      // With explicit bounds
      ReductionAccessor(const PhysicalRegion &region, FieldID fid,
                        ReductionOpID redop, 
                        const Rect<1,T> source_bounds,
                        bool silence_warnings = false,
                        const char *warning_string = NULL,
                        size_t offset = 0,
                        size_t actual_field_size = sizeof(typename REDOP::RHS),
                        bool check_field_size = false)
        : field(fid)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
         region.get_instance_info(LEGION_REDUCE, fid, actual_field_size,
              &is, Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, false/*generic accessor*/, 
              check_field_size, redop);
        if (!Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>::is_compatible(
              instance, fid, source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>(
              instance, fid, source_bounds, offset);
        bounds = AffineBounds::Tester<1,T>(is, source_bounds);
      }
    public:
      __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          REDOP::template fold<EXCLUSIVE>(accessor[p], val);
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Point<1,T>& p) const
        { 
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains(p));
#else
          if (!bounds.contains(p)) 
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                              LEGION_REDUCE);
#endif
          return accessor.ptr(p); 
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r,
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          size_t strides[1];
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline typename REDOP::RHS* ptr(const Rect<1,T>& r,
              size_t strides[1],
              size_t field_size = sizeof(typename REDOP::RHS)) const
        {
          typename REDOP::RHS *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(bounds.contains_all(r));
          assert(result != NULL);
#else
          if (!bounds.contains_all(r)) 
            PhysicalRegion::fail_bounds_check(Domain(r), field, LEGION_REDUCE);
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          strides[0] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,1,
             T,Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>,true>,
             typename REDOP::RHS,1,T>
               operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::ReductionHelper<ReductionAccessor<REDOP,EXCLUSIVE,
            1,T,Realm::MultiAffineAccessor<typename REDOP::RHS,1,T>,true>,
            typename REDOP::RHS,1,T>(*this, p);
        }
    public:
      Realm::MultiAffineAccessor<typename REDOP::RHS,1,T> accessor;
      FieldID field;
      AffineBounds::Tester<1,T> bounds;
    public:
      typedef typename REDOP::RHS value_type;
      typedef typename REDOP::RHS& reference;
      typedef const typename REDOP::RHS& const_reference;
      static const int dim = 1;
    };

    ////////////////////////////////////////////////////////////
    // Multi Region Accessor with Generic Accessors
    ////////////////////////////////////////////////////////////

    // Multi-Accessor, generic, N, bounds checks and/or privilege checks
    template<typename FT, int N, typename T, bool CB, bool CP, int MR>
    class MultiRegionAccessor<FT,N,T,Realm::GenericAccessor<FT,N,T>,CB,CP,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        const PhysicalRegion &region = *start; 
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = region.get_instance_info(
          region_privileges[0], fid, actual_field_size, &region_bounds[0],
          Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
          silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = region_bounds[0].bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
          idx++; 
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = region.get_instance_info(
            region_privileges[0], fid, actual_field_size, &region_bounds[0],
            Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = region_bounds[0].bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx].bounds = 
            source_bounds.intersection(region_bounds[idx].bounds);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
          idx++; 
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        assert(regions.size() <= MR);
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege(); 
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, actual_field_size,
              &region_bounds[0], Internal::NT_TemplateHelper::encode_tag<N,T>(),
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        Rect<N,T> bounds = region_bounds[0].bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        assert(regions.size() <= MR);
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, actual_field_size,
              &region_bounds[0], Internal::NT_TemplateHelper::encode_tag<N,T>(),
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        region_bounds[0].bounds = 
          source_bounds.intersection(region_bounds[0].bounds); 
        Rect<N,T> bounds = region_bounds[0].bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx].bounds = 
            source_bounds.intersection(region_bounds[idx].bounds);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
    public:
      inline FT read(const Point<N,T>& p) const
        {
          bool found = false;
          for (unsigned idx = 0; idx < total_regions; idx++)
          {
            if (!region_bounds[idx].contains(p))
              continue;
            if (CP && ((region_privileges[idx] & LEGION_READ_ONLY) == 0))
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
            found = true;
            break;
          }
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                region_privileges[0], true/*multi*/);
          return accessor.read(p);
        }
      inline void write(const Point<N,T>& p, FT val) const
        {
          bool found = false;
          for (unsigned idx = 0; idx < total_regions; idx++)
          {
            if (!region_bounds[idx].contains(p))
              continue;
            if (CP && ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0))
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
            found = true;
            break;
          }
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                region_privileges[0], true/*multi*/);
          return accessor.write(p, val);
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_NO_ACCESS> 
          operator[](const Point<N,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                region_privileges[0], true/*multi*/);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_NO_ACCESS>(
              accessor[p], field, DomainPoint(p), region_privileges[index]);
        }
      inline ArraySyntax::GenericSyntaxHelper<MultiRegionAccessor<FT,N,T,
             Realm::GenericAccessor<FT,N,T>,CB,CP,MR>,FT,N,T,2,LEGION_NO_ACCESS>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<MultiRegionAccessor<FT,N,T,
            Realm::GenericAccessor<FT,N,T>,CB,CP,MR>,FT,N,T,2,LEGION_NO_ACCESS>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR]; 
      DomainT<N,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Multi-Accessor, generic, 1, bounds checks and/or privilege checks
    template<typename FT, typename T, bool CB, bool CP, int MR>
    class MultiRegionAccessor<FT,1,T,Realm::GenericAccessor<FT,1,T>,CB,CP,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        const PhysicalRegion &region = *start; 
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = region.get_instance_info(
            region_privileges[0], fid, actual_field_size, &region_bounds[0],
            Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = region_bounds[0].bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
          idx++; 
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        const PhysicalRegion &region = *start; 
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = region.get_instance_info(
            region_privileges[0], fid, actual_field_size, &region_bounds[0],
            Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = region_bounds[0].bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx].bounds = 
            source_bounds.intersection(region_bounds[idx].bounds);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
          idx++; 
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        assert(regions.size() <= MR);
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, actual_field_size,
              &region_bounds[0], Internal::NT_TemplateHelper::encode_tag<1,T>(),
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        Rect<1,T> bounds = region_bounds[0].bounds; 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
              region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        assert(regions.size() <= MR);
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, actual_field_size,
              &region_bounds[0], Internal::NT_TemplateHelper::encode_tag<1,T>(),
              warning_string, silence_warnings, true/*generic accessor*/, 
              check_field_size);
        region_bounds[0].bounds = 
          source_bounds.intersection(region_bounds[0].bounds);
        Rect<1,T> bounds = region_bounds[0].bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
            region_privileges[idx], fid, actual_field_size, &region_bounds[idx],
            Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
            silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx].bounds = 
            source_bounds.intersection(region_bounds[idx].bounds);
          bounds = bounds.union_bbox(region_bounds[idx].bounds);
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
    public:
      inline FT read(const Point<1,T>& p) const
        {
          bool found = false;
          for (unsigned idx = 0; idx < total_regions; idx++)
          {
            if (!region_bounds[idx].contains(p))
              continue;
            if (CP && ((region_privileges[idx] & LEGION_READ_ONLY) == 0))
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
            found = true;
            break;
          }
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                region_privileges[0], true/*multi*/);
          return accessor.read(p);
        }
      inline void write(const Point<1,T>& p, FT val) const
        {
          bool found = false;
          for (unsigned idx = 0; idx < total_regions; idx++)
          {
            if (!region_bounds[idx].contains(p))
              continue;
            if (CP && ((region_privileges[idx] & LEGION_WRITE_PRIV) == 0))
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                 region_privileges[idx]);
            found = true;
            break;
          }
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                region_privileges[0], true/*multi*/);
          return accessor.write(p, val);
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_NO_ACCESS> 
          operator[](const Point<1,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
                                region_privileges[0], true/*multi*/);
          return ArraySyntax::AccessorRefHelper<FT,LEGION_NO_ACCESS>(
              accessor[p], field, DomainPoint(p), region_privileges[index]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR]; 
      DomainT<1,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Multi-Accessor, generic, N, no bounds, no privileges
    template<typename FT, int N, typename T, int MR>
    class MultiRegionAccessor<FT,N,T,Realm::GenericAccessor<FT,N,T>,
                              false,false,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds; 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(start->get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
          idx++;
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds; 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
        }
        if (!Realm::GenericAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
    public:
      inline FT read(const Point<N,T>& p) const
        {
          return accessor.read(p);
        }
      inline void write(const Point<N,T>& p, FT val) const
        {
          return accessor.write(p, val);
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>
          operator[](const Point<N,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<MultiRegionAccessor<FT,N,T,
             Realm::GenericAccessor<FT,N,T>,false,false,MR>,
              FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<MultiRegionAccessor<FT,N,T,
                Realm::GenericAccessor<FT,N,T>,false,false,MR>,
                FT,N,T,2,LEGION_READ_WRITE>(
            *this, Point<1,T>(index));
      }
    public:
      mutable Realm::GenericAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Multi-Accessor, generic, 1, no bounds, no privileges
    template<typename FT, typename T, int MR>
    class MultiRegionAccessor<FT,1,T,Realm::GenericAccessor<FT,1,T>,
                              false,false,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds; 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(start->get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
          idx++;
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds; 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, true/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, true/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.intersection(is.bounds));
        }
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
    public:
      inline FT read(const Point<1,T>& p) const
        {
          return accessor.read(p);
        }
      inline void write(const Point<1,T>& p, FT val) const
        {
          return accessor.write(p, val);
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>
          operator[](const Point<1,T>& p) const
        { 
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
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
    class MultiRegionAccessor<FT,N,T,Realm::AffineAccessor<FT,N,T>,CB,true,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, 
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege(); 
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,N,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, transform);
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,N,T> transform,
                          const Rect<N,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<N,T>(is,source_bounds,transform);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<N,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,N,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, transform);
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,N,T> transform,
                          const Rect<N,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<N,T>(is,source_bounds,transform);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<N,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineRefHelper<FT>
                operator[](const Point<N,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return ArraySyntax::AffineRefHelper<FT>(accessor[p], field,
                            DomainPoint(p), region_privileges[index]);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
           Realm::AffineAccessor<FT,N,T>,CB,true,MR>,FT,N,T,2,LEGION_NO_ACCESS>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
          Realm::AffineAccessor<FT,N,T>,CB,true,MR>,FT,N,T,2,LEGION_NO_ACCESS>(
              *this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<N,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Multi-Accessor, affine, 1, with privilege checks (implies bounds checks)
    template<typename FT, typename T, bool CB, int MR>
    class MultiRegionAccessor<FT,1,T,Realm::AffineAccessor<FT,1,T>,CB,true,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, 
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege(); 
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,1,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, transform);
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,1,T> transform,
                          const Rect<1,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<1,T>(is,source_bounds,transform);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<1,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,1,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, transform);
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,1,T> transform,
                          const Rect<1,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<1,T>(is,source_bounds,transform);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<1,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineRefHelper<FT>
                operator[](const Point<1,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return ArraySyntax::AffineRefHelper<FT>(accessor[p], field,
                            DomainPoint(p), region_privileges[index]);
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<1,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Multi-Accessor, affine, N, bounds checks only
    template<typename FT, int N, typename T, int MR>
    class MultiRegionAccessor<FT,N,T,Realm::AffineAccessor<FT,N,T>,
                  true/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 0;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,N,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, transform);
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,N,T> transform,
                          const Rect<N,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<N,T>(is,source_bounds,transform);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<N,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege(); 
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,N,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, transform);
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,N,T> transform,
                          const Rect<N,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<N,T>(is,source_bounds,transform);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<N,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor[p];
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
             Realm::AffineAccessor<FT,N,T>,true,false,MR>,
             FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
              Realm::AffineAccessor<FT,N,T>,true,false,MR>,
              FT,N,T,2,LEGION_READ_WRITE>(*this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<N,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Multi-Accessor, affine, 1, bounds checks only
    template<typename FT, typename T, int MR>
    class MultiRegionAccessor<FT,1,T,Realm::AffineAccessor<FT,1,T>,
                  true/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 0;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,1,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, transform);
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,1,T> transform,
                          const Rect<1,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<1,T>(is,source_bounds,transform);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<1,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,1,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, transform);
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, transform);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,1,T> transform,
                          const Rect<1,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] =AffineBounds::Tester<1,T>(is,source_bounds,transform);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = 
            AffineBounds::Tester<1,T>(is, transform, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor[p];
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<1,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Multi-Accessor, affine, N, no bounds, no privileges
    template<typename FT, int N, typename T, int MR>
    class MultiRegionAccessor<FT,N,T,Realm::AffineAccessor<FT,N,T>,
          false/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,N,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,N,T> transform,
                          const Rect<N,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,N,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,N,T> transform,
                          const Rect<N,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        {
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        {
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
       Realm::AffineAccessor<FT,N,T>,false,false,MR>,FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
            Realm::AffineAccessor<FT,N,T>,false,false,MR>,
            FT,N,T,2,LEGION_READ_WRITE>(*this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Multi-Accessor, affine, 1, no bounds, no privileges
    template<typename FT, typename T, int MR>
    class MultiRegionAccessor<FT,1,T,Realm::AffineAccessor<FT,1,T>,
          false/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds);
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds);
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,1,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds);
      }
      template<int M, typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const AffineTransform<M,1,T> transform,
                          const Rect<1,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds);
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,fid,bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,1,T> transform,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds)) 
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
      template<int M>
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const AffineTransform<M,1,T> transform,
                          const Rect<1,T> source_bounds, FieldID fid, 
                          // The actual field size in case it is different from the
                          // one being used in FT and we still want to check it
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance,
              transform.transform, transform.offset, fid, bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        {
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        {
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
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
    class MultiRegionAccessor<FT,N,T,
                              Realm::MultiAffineAccessor<FT,N,T>,CB,true,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, 
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege(); 
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      } 
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineRefHelper<FT>
                operator[](const Point<N,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return ArraySyntax::AffineRefHelper<FT>(accessor[p], field,
                            DomainPoint(p), region_privileges[index]);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
       Realm::MultiAffineAccessor<FT,N,T>,CB,true,MR>,FT,N,T,2,LEGION_NO_ACCESS>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
          Realm::MultiAffineAccessor<FT,N,T>,CB,true,MR>,FT,N,T,2,
            LEGION_NO_ACCESS>(*this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<N,T> region_bounds[MR];
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
    class MultiRegionAccessor<FT,1,T,
                              Realm::MultiAffineAccessor<FT,1,T>,CB,true,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid, 
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege(); 
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline ArraySyntax::AffineRefHelper<FT>
                operator[](const Point<1,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return ArraySyntax::AffineRefHelper<FT>(accessor[p], field,
                            DomainPoint(p), region_privileges[index]);
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
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
              PhysicalRegion::fail_privilege_check(DomainPoint(p), field,
                                                   region_privileges[idx]);
#endif
            }
            found = true;
            break;
          }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(found);
#else
          if (!found)
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<1,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Multi-Accessor, multi affine, N, bounds checks only
    template<typename FT, int N, typename T, int MR>
    class MultiRegionAccessor<FT,N,T,Realm::MultiAffineAccessor<FT,N,T>,
                      true/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 0;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is); 
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<N,T>(is, source_bounds);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<N,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor[p];
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
             Realm::MultiAffineAccessor<FT,N,T>,true,false,MR>,
             FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
              Realm::MultiAffineAccessor<FT,N,T>,true,false,MR>,
              FT,N,T,2,LEGION_READ_WRITE>(*this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<N,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Multi-Accessor, multi affine, 1, bounds checks only
    template<typename FT, typename T, int MR>
    class MultiRegionAccessor<FT,1,T,Realm::MultiAffineAccessor<FT,1,T>,
                      true/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 0;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
        total_regions = idx;
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          region_privileges[idx] = start->get_privilege();
          const Realm::RegionInstance inst = start->get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
        total_regions = idx;
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is); 
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
        : field(fid), total_regions(regions.size())
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        region_privileges[0] = region.get_privilege();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region_privileges[0], fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        region_bounds[0] = AffineBounds::Tester<1,T>(is, source_bounds);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          region_privileges[idx] = regions[idx].get_privilege();
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                region_privileges[idx], fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          region_bounds[idx] = AffineBounds::Tester<1,T>(is, source_bounds);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          return accessor[p];
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
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
            PhysicalRegion::fail_bounds_check(DomainPoint(p), field,
                                region_privileges[0], true/*multi*/);
#endif
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
      FieldID field;
      PrivilegeMode region_privileges[MR];
      AffineBounds::Tester<1,T> region_bounds[MR];
      unsigned total_regions;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Multi-Accessor, multi affine, N, no bounds, no privileges
    template<typename FT, int N, typename T, int MR>
    class MultiRegionAccessor<FT,N,T,Realm::MultiAffineAccessor<FT,N,T>,
                    false/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<N,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<N,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<N,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<N,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T>& p) const
        {
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T>& p, FT val) const
        {
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T>& p) const
        { 
          return accessor[p]; 
        }
      __CUDA_HD__
      inline ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
        Realm::MultiAffineAccessor<FT,N,T>,false,false,MR>,FT,N,T,2,
          LEGION_READ_WRITE> operator[](T index) const
      {
        return ArraySyntax::AffineSyntaxHelper<MultiRegionAccessor<FT,N,T,
            Realm::MultiAffineAccessor<FT,N,T>,false,false,MR>,
            FT,N,T,2,LEGION_READ_WRITE>(*this, Point<1,T>(index));
      }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<N,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Multi-Accessor, multi affine, 1, no bounds, no privileges
    template<typename FT, typename T, int MR>
    class MultiRegionAccessor<FT,1,T,Realm::MultiAffineAccessor<FT,1,T>,
                    false/*check bounds*/,false/*check privileges*/,MR> {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      MultiRegionAccessor(void) { }
    public:
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds;
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
      template<typename InputIterator>
      MultiRegionAccessor(InputIterator start, InputIterator stop,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (start == stop)
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = *start;
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        unsigned idx = 1;
        while (++start != stop)
        {
          assert(idx < MR);
          const Realm::RegionInstance inst = start->get_instance_info(
                start->get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
          idx++;
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
    public:
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          FieldID fid, size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = is.bounds;
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(is.bounds);
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, 
                                                               fid, bounds)) 
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
      MultiRegionAccessor(const std::vector<PhysicalRegion> &regions,
                          const Rect<1,T> source_bounds, FieldID fid,
                          size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                          bool check_field_size = true,
#else
                          bool check_field_size = false,
#endif
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        if (regions.empty())
          return;
        DomainT<1,T> is;
        const PhysicalRegion &region = regions.front();
        const Realm::RegionInstance instance = 
          region.get_instance_info(region.get_privilege(), fid,
              actual_field_size, &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, check_field_size);
        Rect<1,T> bounds = source_bounds.intersection(is.bounds); 
        for (unsigned idx = 1; idx < regions.size(); idx++)
        {
          const Realm::RegionInstance inst = regions[idx].get_instance_info(
                regions[idx].get_privilege(), fid, actual_field_size, &is,
                Internal::NT_TemplateHelper::encode_tag<1,T>(), warning_string,
                silence_warnings, false/*generic accessor*/, check_field_size);
          if (inst != instance)
            region.report_incompatible_multi_accessor(idx, fid, instance, inst);
          bounds = bounds.union_bbox(source_bounds.inersection(is.bounds));
        }
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance,
                                                               fid, bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, bounds);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T>& p) const
        {
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<1,T>& p, FT val) const
        {
          return accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T>& p) const
        { 
          return accessor[p]; 
        }
      template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
      inline void reduce(const Point<1,T>& p, 
                         typename REDOP::RHS val) const
        { 
          REDOP::template apply<EXCLUSIVE>(accessor[p], val);
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // A hidden class for users that really know what they are doing
    /**
     * \class UnsafeFieldAccessor
     * This is a class for getting access to region data without
     * privilege checks or bounds checks. Users should only use
     * this accessor if they are confident that they actually do
     * have their privileges and bounds correct
     */
    template<typename FT, int N, typename T = coord_t,
             typename A = Realm::GenericAccessor<FT,N,T> >
    class UnsafeFieldAccessor {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      UnsafeFieldAccessor(void) { }
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,

                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/,
              false/*check field size*/);
        if (!A::is_compatible(instance, fid, is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = A(instance, fid, is.bounds, offset);
      }
    public:
      inline FT read(const Point<N,T> &p) const
        {
          return accessor.read(p);
        }
      inline void write(const Point<N,T> &p, FT val) const
        {
          accessor.write(p, val);
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>
              operator[](const Point<N,T> &p) const
        {
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
      inline ArraySyntax::GenericSyntaxHelper<UnsafeFieldAccessor<FT,N,T,A>,
                                                FT,N,T,2,LEGION_READ_WRITE>
          operator[](T index) const
        {
          return ArraySyntax::GenericSyntaxHelper<UnsafeFieldAccessor<FT,N,T,A>,
                          FT,N,T,2,LEGION_READ_WRITE>(*this, Point<1,T>(index));
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
    class UnsafeFieldAccessor<FT,1,T,Realm::GenericAccessor<FT,1,T> > {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      UnsafeFieldAccessor(void) { }
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings, true/*generic accessor*/,
              false/*check field size*/);
        if (!Realm::GenericAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                           is.bounds))
          region.report_incompatible_accessor("GenericAccessor", instance, fid);
        accessor = 
          Realm::GenericAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
    public:
      inline FT read(const Point<1,T> &p) const
        {
          return accessor.read(p);
        }
      inline void write(const Point<1,T> &p, FT val) const
        {
          accessor.write(p, val);
        }
      inline ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>
              operator[](const Point<1,T> &p) const
        {
          return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_WRITE>(
                                                          accessor[p]);
        }
    public:
      mutable Realm::GenericAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    template<typename FT, int N, typename T>
    class UnsafeFieldAccessor<FT, N, T, Realm::AffineAccessor<FT,N,T> > {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      UnsafeFieldAccessor(void) { }
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const Rect<N,T> source_bounds,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,N,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const AffineTransform<M,N,T> transform,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const AffineTransform<M,N,T> transform,
                          const Rect<N,T> source_bounds,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,N,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,N,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T> &p) const
        {
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T> &p, FT val) const
        {
          accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T> &p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          for (int i = 0; i < N; i++)
            strides[i] = accessor.strides[i] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T> &p) const
        {
          return accessor[p];
        }
      __CUDA_HD__
      inline FT& operator[](T index) const
        {
          return ArraySyntax::AffineSyntaxHelper<UnsafeFieldAccessor<FT,N,T,
                 Realm::AffineAccessor<FT,N,T> >,FT,N,T,2,LEGION_READ_WRITE>(
                *this, Point<1,T>(index));
        }
    public:
      Realm::AffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Specialization for UnsafeFieldAccessor for dimension 1 
    // to avoid ambiguity for array access
    template<typename FT, typename T>
    class UnsafeFieldAccessor<FT,1,T,Realm::AffineAccessor<FT,1,T> > {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      UnsafeFieldAccessor(void) { }
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          is.bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const Rect<1,T> source_bounds,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                          source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = 
          Realm::AffineAccessor<FT,1,T>(instance, fid, source_bounds, offset);
      }
      // With explicit transform
      template<int M>
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const AffineTransform<M,1,T> transform,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, offset);
      }
      // With explicit transform and bounds
      template<int M>
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const AffineTransform<M,1,T> transform,
                          const Rect<1,T> source_bounds,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<M,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<M,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::AffineAccessor<FT,1,T>::is_compatible(instance, 
              transform.transform, transform.offset, fid, source_bounds))
          region.report_incompatible_accessor("AffineAccessor", instance, fid);
        accessor = Realm::AffineAccessor<FT,1,T>(instance, transform.transform,
            transform.offset, fid, source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T> &p) const
        {
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T> &p, FT val) const
        {
          accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T> &p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T> &r, size_t field_size = sizeof(FT)) const
        {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(Internal::is_dense_layout(r, accessor.strides, field_size));
#else
          if (!Internal::is_dense_layout(r, accessor.strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<1,T> &r, size_t strides[1], 
                     size_t field_size = sizeof(FT)) const
        {
          strides[0] = accessor.strides[0] / field_size;
          return accessor.ptr(r.lo);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T> &p) const
        {
          return accessor[p];
        }
    public:
      Realm::AffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    }; 

    template<typename FT, int N, typename T>
    class UnsafeFieldAccessor<FT, N, T, Realm::MultiAffineAccessor<FT,N,T> > {
    private:
      static_assert(N > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      UnsafeFieldAccessor(void) { }
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,N,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const Rect<N,T> source_bounds,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<N,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<N,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::MultiAffineAccessor<FT,N,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                               instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,N,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T> &p) const
        {
          return accessor.read(p);
        }
      __CUDA_HD__
      inline void write(const Point<N,T> &p, FT val) const
        {
          accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT* ptr(const Point<N,T> &p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t field_size = sizeof(FT)) const
        {
          size_t strides[N];
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
          assert(Internal::is_dense_layout(r, strides, field_size));
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
          if (!Internal::is_dense_layout(r, strides, field_size))
          {
            fprintf(stderr, 
                "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_DENSE_RECTANGLE);
#endif
          }
#endif
          return result;
        }
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T>& r, size_t strides[N],
                     size_t field_size = sizeof(FT)) const
        {
          FT *result = accessor.ptr(r, strides);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
          assert(result != NULL);
#else
          if (result == NULL)
          {
            fprintf(stderr, "ERROR: Illegal request for pointer of rectangle "
                            "not contained within the bounds of a piece\n");
#ifdef DEBUG_LEGION
            assert(false);
#else
            exit(ERROR_NON_PIECE_RECTANGLE);
#endif
          }
#endif
          for (int i = 0; i < N; i++)
            strides[i] /= field_size;
          return result;
        }
      __CUDA_HD__
      inline FT& operator[](const Point<N,T> &p) const
        {
          return accessor[p];
        }
      __CUDA_HD__
      inline FT& operator[](T index) const
        {
          return ArraySyntax::AffineSyntaxHelper<UnsafeFieldAccessor<FT,N,T,
                 Realm::MultiAffineAccessor<FT,N,T> >,
                  FT,N,T,2,LEGION_READ_WRITE>(*this, Point<1,T>(index));
        }
    public:
      Realm::MultiAffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Specialization for UnsafeFieldAccessor for dimension 1 
    // to avoid ambiguity for array access
    template<typename FT, typename T>
    class UnsafeFieldAccessor<FT,1,T,Realm::MultiAffineAccessor<FT,1,T> > {
    private:
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      UnsafeFieldAccessor(void) { }
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               is.bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,1,T>(instance, fid, is.bounds, offset);
      }
      // With explicit bounds
      UnsafeFieldAccessor(const PhysicalRegion &region, FieldID fid,
                          const Rect<1,T> source_bounds,
                          bool silence_warnings = false,
                          const char *warning_string = NULL,
                          size_t offset = 0)
      {
        DomainT<1,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<1,T>(), 
              warning_string, silence_warnings,
              false/*generic accessor*/, false/*check field size*/);
        if (!Realm::MultiAffineAccessor<FT,1,T>::is_compatible(instance, fid, 
                                                               source_bounds))
          region.report_incompatible_accessor("MultiAffineAccessor", 
                                              instance, fid);
        accessor = Realm::MultiAffineAccessor<FT,1,T>(instance, fid, 
                                              source_bounds, offset);
      }
    public:
      __CUDA_HD__
      inline FT read(const Point<1,T> &p) const
        {
          return accessor.read(p); 
        }
      __CUDA_HD__
      inline void write(const Point<1,T> &p, FT val) const
        {
          accessor.write(p, val);
        }
      __CUDA_HD__
      inline FT* ptr(const Point<1,T> &p) const
        {
          return accessor.ptr(p);
        }
      __CUDA_HD__
      inline FT& operator[](const Point<1,T> &p) const
        {
          return accessor[p];
        }
    public:
      Realm::MultiAffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    /**
     * \class UnsafeSpanIterator
     * This is a hidden class analogous to the UnsafeFieldAccessor that
     * allows for traversals over spans of elements in compact instances
     */
    template<typename FT, int DIM, typename T = coord_t>
    class UnsafeSpanIterator {
    private:
      static_assert(DIM > 0, "DIM must be positive");
      static_assert(std::is_integral<T>::value, "must be integral type");
    public:
      UnsafeSpanIterator(void) { }
      UnsafeSpanIterator(const PhysicalRegion &region, FieldID fid,
                         bool privileges_only = true,
                         bool silence_warnings = false,
                         const char *warning_string = NULL,
                         size_t offset = 0)
        : piece_iterator(PieceIteratorT<DIM,T>(region, fid, privileges_only)),
          partial_piece(false)
      {
        DomainT<DIM,T> is;
        const Realm::RegionInstance instance = 
          region.get_instance_info(LEGION_NO_ACCESS, fid, sizeof(FT), &is,
              Internal::NT_TemplateHelper::encode_tag<DIM,T>(), warning_string,
              silence_warnings, false/*generic accessor*/, false/*check size*/);
        if (!Realm::MultiAffineAccessor<FT,DIM,T>::is_compatible(instance, fid, 
                                                                 is.bounds))
          region.report_incompatible_accessor("UnsafeSpanIterator", 
                                              instance, fid);
        accessor = 
          Realm::MultiAffineAccessor<FT,DIM,T>(instance, fid, is.bounds,offset);
        // initialize the first span
        step();
      }
    public:
      inline bool valid(void) const
        {
          return !current.empty();
        }
      inline bool step(void)
        {
          // Handle the remains of a partial piece if that is what we're doing
          if (partial_piece)
          {
            bool carry = false;
            for (int idx = 0; idx < DIM; idx++)
            {
              const int dim = dim_order[idx];
              if (carry || (dim == partial_step_dim))
              {
                if (partial_step_point[dim] < piece_iterator->hi[dim])
                {
                  partial_step_point[dim] += 1;
                  carry = false;
                  break;
                }
                // carry case so reset and roll-over
                partial_step_point[dim] = piece_iterator->lo[dim];
                carry = true;
              }
              // Skip any dimensions before the partial step dim
            }
            // Make the next span
            current = Span<FT,LEGION_READ_WRITE>(
              accessor.ptr(partial_step_point), current.size(), current.step());
            // See if we are done with this partial piece
            if (carry)
              partial_piece = false; 
            return true;
          }
          // clear this for the next iteration
          current = Span<FT,LEGION_READ_WRITE>(); 
          // Otherwise try to group as many rectangles together as we can
          while (piece_iterator.valid())
          {
            size_t strides[DIM];
            FT *ptr = accessor.ptr(*piece_iterator, strides); 
#ifdef DEBUG_LEGION
            // If we ever hit this it is a runtime error because the 
            // runtime should already be guaranteeing these rectangles
            // are inside of pieces for the instance
            assert(ptr != NULL);
#endif         
            // Find the minimum stride and see if this piece is dense
            size_t min_stride = SIZE_MAX;
            for (int dim = 0; dim < DIM; dim++)
              if (strides[dim] < min_stride)
                min_stride = strides[dim];
            if (Internal::is_dense_layout(*piece_iterator, strides, min_stride))
            {
              const size_t volume = piece_iterator->volume();
              if (!current.empty())
              {
                uintptr_t base = current.get_base();
                // See if we can append to the current span
                if ((current.step() == min_stride) &&
                    ((base + (current.size() * min_stride)) == uintptr_t(ptr)))
                  current = Span<FT,LEGION_READ_WRITE>(current.data(), 
                                  current.size() + volume, min_stride);
                else // Save this rectangle for the next iteration
                  break;
              }
              else // Start a new span
                current = Span<FT,LEGION_READ_WRITE>(ptr, volume, min_stride);
            }
            else
            {
              // Not a uniform stride, so go to the partial piece case
              if (current.empty())
              {
                partial_piece = true;
                // Compute the dimension order from smallest to largest
                size_t stride_floor = 0;
                for (int idx = 0; idx < DIM; idx++)
                {
                  int index = -1;
                  size_t local_min = SIZE_MAX;
                  for (int dim = 0; dim < DIM; dim++)
                  {
                    if (strides[dim] <= stride_floor)
                      continue;
                    if (strides[dim] < local_min)
                    {
                      local_min = strides[dim];
                      index = dim;
                    }
                  }
#ifdef DEBUG_LEGION
                  assert(index >= 0); 
#endif
                  dim_order[idx] = index;
                  stride_floor = local_min;
                }
                // See which dimensions we can handle at once and which ones
                // we are going to need to walk over
                size_t extent = 1;
                size_t exp_offset = min_stride;
                partial_step_dim = -1;
                for (int idx = 0; idx < DIM; idx++)
                {
                  const int dim = dim_order[idx];
                  if (strides[dim] == exp_offset)
                  {
                    size_t pitch =
                     ((piece_iterator->hi[dim] - piece_iterator->lo[dim]) + 1); 
                    exp_offset *= pitch;
                    extent *= pitch;
                  }
                  // First dimension that is not contiguous
                  partial_step_dim = dim;
                  break;
                }
#ifdef DEBUG_LEGION
                assert(partial_step_dim >= 0);
#endif
                partial_step_point = piece_iterator->lo;
                current = Span<FT,LEGION_READ_WRITE>(
                    accessor.ptr(partial_step_point), extent, min_stride);
              }
              // No matter what we are breaking out here
              break;
            }
            // Step the piece iterator for the next iteration
            piece_iterator.step();
          }
          return valid();
        }
    public:
      inline operator bool(void) const
        {
          return valid();
        }
      inline bool operator()(void) const
        {
          return valid();
        }
      inline Span<FT,LEGION_READ_WRITE> operator*(void) const
        {
          return current;
        }
      inline const Span<FT,LEGION_READ_WRITE>* operator->(void) const
        {
          return &current;
        }
      inline UnsafeSpanIterator<FT,DIM,T>& operator++(void)
        {
          step();
          return *this;
        }
      inline UnsafeSpanIterator<FT,DIM,T> operator++(int)
        {
          UnsafeSpanIterator<FT,DIM,T> result = *this;
          step();
          return result;
        }
    private:
      PieceIteratorT<DIM,T> piece_iterator;
      Realm::MultiAffineAccessor<FT,DIM,T> accessor;
      Span<FT,LEGION_READ_WRITE> current;
      Point<DIM,T> partial_step_point;
      int dim_order[DIM];
      int partial_step_dim;
      bool partial_piece;
    };

    //--------------------------------------------------------------------------
    template<typename T>
    inline DeferredValue<T>::DeferredValue(void)
      : instance(Realm::RegionInstance::NO_INST)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline DeferredValue<T>::DeferredValue(T initial_value, size_t alignment)
    //--------------------------------------------------------------------------
    {
      // Construct a Region of size 1 in the zero copy memory for now
      Machine machine = Realm::Machine::get_machine();
      Machine::MemoryQuery finder(machine);
      finder.has_affinity_to(Processor::get_executing_processor());
      finder.only_kind(Memory::Z_COPY_MEM);
      if (finder.count() == 0)
      {
        fprintf(stderr,"Deferred Values currently need a local allocation "
                       "of zero-copy memory to work correctly. Please provide "
                       "a non-zero amount with the -ll:zsize flag");
        assert(false);
      }
      const Memory memory = finder.first();
      const Realm::Point<1,coord_t> zero(0);
      Realm::IndexSpace<1,coord_t> bounds = Realm::Rect<1,coord_t>(zero, zero);
      const std::vector<size_t> field_sizes(1,sizeof(T));
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      int dim_order[1];
      dim_order[0] = 0;
      Realm::InstanceLayoutGeneric *layout = 
        Realm::InstanceLayoutGeneric::choose_instance_layout(bounds, 
            constraints, dim_order);
      layout->alignment_reqd = alignment;
      Runtime *runtime = Runtime::get_runtime();
      instance = runtime->create_task_local_instance(memory, layout);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible = 
        Realm::AffineAccessor<T,1,coord_t>::is_compatible(instance, 0); 
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      accessor = Realm::AffineAccessor<T,1,coord_t>(instance, 0/*field id*/);
      // Initialize the value
      accessor[zero] = initial_value;
    }

    //--------------------------------------------------------------------------
    template<typename T> __CUDA_HD__
    inline T DeferredValue<T>::read(void) const
    //--------------------------------------------------------------------------
    {
      return accessor.read(Point<1,coord_t>(0));
    }

    //--------------------------------------------------------------------------
    template<typename T> __CUDA_HD__
    inline void DeferredValue<T>::write(T value) const
    //--------------------------------------------------------------------------
    {
      accessor.write(Point<1,coord_t>(0), value);
    }

    //--------------------------------------------------------------------------
    template<typename T> __CUDA_HD__
    inline T* DeferredValue<T>::ptr(void) const
    //--------------------------------------------------------------------------
    {
      return accessor.ptr(Point<1,coord_t>(0));
    }

    //--------------------------------------------------------------------------
    template<typename T> __CUDA_HD__
    inline T& DeferredValue<T>::ref(void) const
    //--------------------------------------------------------------------------
    {
      return accessor[Point<1,coord_t>(0)];
    }

    //--------------------------------------------------------------------------
    template<typename T> __CUDA_HD__
    inline DeferredValue<T>::operator T(void) const
    //--------------------------------------------------------------------------
    {
      return accessor[Point<1,coord_t>(0)];
    }

    //--------------------------------------------------------------------------
    template<typename T> __CUDA_HD__
    inline DeferredValue<T>& DeferredValue<T>::operator=(T value)
    //--------------------------------------------------------------------------
    {
      accessor[Point<1,coord_t>(0)] = value;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void DeferredValue<T>::finalize(Runtime *runtime, Context ctx) const
    //--------------------------------------------------------------------------
    {
#if 0
      Runtime::legion_task_postamble(runtime, ctx,
                    accessor.ptr(Point<1,coord_t>(0)), sizeof(T),
                    true/*owner*/, instance, instance.get_location().kind());
#else
      Runtime::legion_task_postamble(runtime, ctx,
                    accessor.ptr(Point<1,coord_t>(0)), sizeof(T),
                    false/*owner*/, instance);
#endif
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool EXCLUSIVE>
    inline DeferredReduction<REDOP,EXCLUSIVE>::DeferredReduction(size_t align)
      : DeferredValue<typename REDOP::RHS>(REDOP::identity, align)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
    inline void DeferredReduction<REDOP,EXCLUSIVE>::reduce(
                                                typename REDOP::RHS value) const
    //--------------------------------------------------------------------------
    {
      REDOP::template fold<EXCLUSIVE>(
          this->accessor[Point<1,coord_t>(0)], value);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool EXCLUSIVE> __CUDA_HD__
    inline void DeferredReduction<REDOP,EXCLUSIVE>::operator<<=(
                                                typename REDOP::RHS value) const
    //--------------------------------------------------------------------------
    {
      REDOP::template fold<EXCLUSIVE>(
          this->accessor[Point<1,coord_t>(0)], value);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline UntypedDeferredValue::UntypedDeferredValue(
                                                    const DeferredValue<T> &rhs)
      : instance(rhs.instance), field_size(sizeof(T))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool EXCLUSIVE>
    inline UntypedDeferredValue::UntypedDeferredValue(
                                  const DeferredReduction<REDOP,EXCLUSIVE> &rhs)
      : instance(rhs.instance), field_size(sizeof(REDOP::RHS))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline UntypedDeferredValue::operator DeferredValue<T>(void) const
    //--------------------------------------------------------------------------
    {
      assert(field_size == sizeof(T));
      DeferredValue<T> result;
      result.instance = instance;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible = 
        Realm::AffineAccessor<T,1,coord_t>::is_compatible(instance, 0); 
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      result.accessor =
        Realm::AffineAccessor<T,1,coord_t>(instance, 0/*field id*/);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool EXCLUSIVE>
    inline UntypedDeferredValue::operator 
                                  DeferredReduction<REDOP,EXCLUSIVE>(void) const
    //--------------------------------------------------------------------------
    {
      assert(field_size == sizeof(REDOP::RHS));
      DeferredReduction<typename REDOP::RHS,EXCLUSIVE> result;
      result.instance = instance;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible = 
        Realm::AffineAccessor<typename REDOP::RHS,1,coord_t>::is_compatible(
                                                    instance, 0/*field id*/); 
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      result.accessor =
        Realm::AffineAccessor<typename REDOP::RHS,1,coord_t>(instance,
                                                             0/*field id*/);
      return result;
    }

    // DeferredBuffer without bounds checks
    template<typename FT, int N, typename T> 
#ifdef LEGION_BOUNDS_CHECKS
    class DeferredBuffer<FT,N,T,false> {
#else
    class DeferredBuffer<FT,N,T,true> {
#endif
    public:
      inline DeferredBuffer(void);
      // Memory kinds
      inline DeferredBuffer(Memory::Kind kind, 
                            const Domain &bounds,
                            const FT *initial_value = NULL,
                            size_t alignment = 16,
                            bool fortran_order_dims = false);
      inline DeferredBuffer(const Rect<N,T> &bounds, 
                            Memory::Kind kind,
                            const FT *initial_value = NULL,
                            size_t alignment = 16,
                            bool fortran_order_dims = false);
      // Explicit memory
      inline DeferredBuffer(Memory memory, 
                            const Domain &bounds,
                            const FT *initial_value = NULL,
                            size_t alignment = 16,
                            bool fortran_order_dims = false);
      inline DeferredBuffer(const Rect<N,T> &bounds, 
                            Memory memory,
                            const FT *initial_value = NULL,
                            size_t alignment = 16,
                            bool fortran_order_dims = false);
    public: // Explicit ordering
      inline DeferredBuffer(Memory::Kind kind,
                            const Domain &bounds,
                            std::array<DimensionKind,N> ordering,
                            const FT *initial_value = NULL,
                            size_t alignment = 16);
      inline DeferredBuffer(const Rect<N,T> &bounds,
                            Memory::Kind kind,
                            std::array<DimensionKind,N> ordering,
                            const FT *initial_value = NULL,
                            size_t alignment = 16);
      inline DeferredBuffer(Memory memory,
                            const Domain &bounds,
                            std::array<DimensionKind,N> ordering,
                            const FT *initial_value = NULL,
                            size_t alignment = 16);
      inline DeferredBuffer(const Rect<N,T> &bounds,
                            Memory memory,
                            std::array<DimensionKind,N> ordering,
                            const FT *initial_value = NULL,
                            size_t alignment = 16);
    protected:
      Memory get_memory_from_kind(Memory::Kind kind);
      void initialize_layout(size_t alignment, bool fortran_order_dims);
      void initialize(Memory memory,
                      DomainT<N,T> bounds,
                      const FT *initial_value);
    public:
      __CUDA_HD__
      inline FT read(const Point<N,T> &p) const;
      __CUDA_HD__
      inline void write(const Point<N,T> &p, FT value) const;
      __CUDA_HD__
      inline FT* ptr(const Point<N,T> &p) const;
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T> &r) const; // must be dense
      __CUDA_HD__
      inline FT* ptr(const Rect<N,T> &r, size_t strides[N]) const;
      __CUDA_HD__
      inline FT& operator[](const Point<N,T> &p) const;
    protected:
      Realm::RegionInstance instance;
      Realm::AffineAccessor<FT,N,T> accessor;
      std::array<DimensionKind,N> ordering;
      size_t alignment;
#ifndef LEGION_BOUNDS_CHECKS
      DomainT<N,T> bounds;
#endif
    };

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(void)
      : instance(Realm::RegionInstance::NO_INST)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(Memory::Kind kind, const Domain &space,
                             const FT *initial_value/* = NULL*/,
                             size_t alignment/* = 16*/,
                             bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory::Kind kind,
                             const FT *initial_value /*= NULL*/,
                             size_t alignment/* = 16*/,
                             bool fortran_order_dims /*= false*/)
    //--------------------------------------------------------------------------
    {
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(Memory memory, const Domain &space,
                             const FT *initial_value/* = NULL*/,
                             size_t alignment/* = 16*/,
                             bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory memory,
                             const FT *initial_value /*= NULL*/,
                             size_t alignment/* = 16*/,
                             bool fortran_order_dims /*= false*/)
    //--------------------------------------------------------------------------
    {
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(Memory::Kind kind, const Domain &space,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value/* = NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory::Kind kind,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value /*= NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(Memory memory, const Domain &space,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value/* = NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
           false
#else
            CB
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory memory,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value /*= NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    Memory DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
                          false
#else
                          CB
#endif
                          >::get_memory_from_kind(Memory::Kind kind)
    //--------------------------------------------------------------------------
    {
      // Construct an instance of the right size in the corresponding memory
      Machine machine = Realm::Machine::get_machine();
      Machine::MemoryQuery finder(machine);
      finder.best_affinity_to(Processor::get_executing_processor());
      finder.only_kind(kind);
      if (finder.count() == 0)
      {
        finder = Machine::MemoryQuery(machine);
        finder.has_affinity_to(Processor::get_executing_processor());
        finder.only_kind(kind);
      }
      if (finder.count() == 0)
      {
        fprintf(stderr,"DeferredBuffer unable to find a memory of kind %d",
                kind);
        assert(false);
      }
      return finder.first();
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    void DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
                        false
#else
                        CB
#endif
                        >::initialize_layout(size_t _alignment,
                                             bool fortran_order_dims)
    //--------------------------------------------------------------------------
    {
      if (fortran_order_dims)
      {
        for (int i = 0; i < N; i++)
          ordering[i] =
            static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
      }
      else
      {
        for (int i = 0; i < N; i++)
          ordering[i] =
            static_cast<DimensionKind>(
                static_cast<int>(LEGION_DIM_X) + N - (i + 1));
      }

      alignment = _alignment;
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    void DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
                        false
#else
                        CB
#endif
                        >::initialize(Memory memory,
                                      DomainT<N,T> bounds,
                                      const FT *initial_value)
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = Runtime::get_runtime();
      const std::vector<size_t> field_sizes(1,sizeof(FT));
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      int dim_order[N];
      for (int i = 0; i < N; ++i)
        dim_order[i] =
          static_cast<int>(ordering[i]) - static_cast<int>(LEGION_DIM_X);
      Realm::InstanceLayoutGeneric *layout = 
        Realm::InstanceLayoutGeneric::choose_instance_layout(
          bounds, constraints, dim_order);
      layout->alignment_reqd = alignment;
      instance = runtime->create_task_local_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests;
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, sizeof(FT));
        const Internal::LgEvent wait_on(
            bounds.fill(dsts, no_requests, initial_value, sizeof(FT)));
        if (wait_on.exists())
          wait_on.wait();
      }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible =
        Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
                                                     0/*fid*/,
                                                     bounds.bounds);
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      accessor = Realm::AffineAccessor<FT,N,T>(instance,
                                               0/*field id*/,
                                               bounds.bounds);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              > __CUDA_HD__
    inline FT DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              false 
#else
              CB
#endif
              >::read(const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      return accessor.read(p);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline void DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              false 
#else
              CB
#endif
              >::write(const Point<N,T> &p,
                                                    FT value) const
    //--------------------------------------------------------------------------
    {
      accessor.write(p, value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              false 
#else
              CB
#endif
              >::ptr(const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      return accessor.ptr(p);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              false 
#else
              CB
#endif   
              >::ptr(const Rect<N,T> &r) const
    //--------------------------------------------------------------------------
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(Internal::is_dense_layout(r, accessor.strides, sizeof(FT)));
#else
      if (!Internal::is_dense_layout(r, accessor.strides, sizeof(FT)))
      {
        fprintf(stderr, 
            "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
        assert(false);
#else
        exit(ERROR_NON_DENSE_RECTANGLE);
#endif
      }
#endif
      return accessor.ptr(r.lo);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
            > __CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              false 
#else
              CB
#endif          
            >::ptr(const Rect<N,T> &r, size_t strides[N]) const
    //--------------------------------------------------------------------------
    {
      for (int i = 0; i < N; i++)
        strides[i] = accessor.strides[i] / sizeof(FT);
      return accessor.ptr(r.lo);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifndef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline FT& DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              false 
#else
              CB
#endif
              >::operator[](const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      return accessor[p];
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB 
#else
            true
#endif
           >::DeferredBuffer(void)
      : instance(Realm::RegionInstance::NO_INST)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB 
#else
            true
#endif
           >::DeferredBuffer(Memory::Kind kind, const Domain &space,
                             const FT *initial_value/* = NULL*/,
                             size_t alignment/* = 16*/,
                             const bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB
#else
            true
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory::Kind kind,
                             const FT *initial_value /*= NULL*/,
                             size_t alignment/* = 16*/,
                             const bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB 
#else
            true
#endif
           >::DeferredBuffer(Memory memory, const Domain &space,
                             const FT *initial_value/* = NULL*/,
                             size_t alignment/* = 16*/,
                             const bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB
#else
            true
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory memory,
                             const FT *initial_value /*= NULL*/,
                             size_t alignment/* = 16*/,
                             const bool fortran_order_dims/* = false*/)
    //--------------------------------------------------------------------------
    {
      initialize_layout(alignment, fortran_order_dims);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB
#else
            true
#endif
           >::DeferredBuffer(Memory::Kind kind, const Domain &space,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value/* = NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB
#else
            true
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory::Kind kind,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value /*= NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      const Realm::Memory memory = get_memory_from_kind(kind);
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB
#else
            true
#endif
           >::DeferredBuffer(Memory memory, const Domain &space,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value/* = NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      if (!space.dense())
      {
        fprintf(stderr, "DeferredBuffer only allows a dense domain\n");
        assert(false);
      }
      initialize(memory, space, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    inline DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
            CB
#else
            true
#endif
           >::DeferredBuffer(const Rect<N,T> &rect, Memory memory,
                             std::array<DimensionKind,N> _ordering,
                             const FT *initial_value /*= NULL*/,
                             size_t _alignment/* = 16*/)
      : ordering(_ordering), alignment(_alignment)
    //--------------------------------------------------------------------------
    {
      initialize(memory, rect, initial_value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    Memory DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
                          CB
#else
                          true
#endif
                          >::get_memory_from_kind(Memory::Kind kind)
    //--------------------------------------------------------------------------
    {
      // Construct an instance of the right size in the corresponding memory
      Machine machine = Realm::Machine::get_machine();
      Machine::MemoryQuery finder(machine);
      finder.best_affinity_to(Processor::get_executing_processor());
      finder.only_kind(kind);
      if (finder.count() == 0)
      {
        finder = Machine::MemoryQuery(machine);
        finder.has_affinity_to(Processor::get_executing_processor());
        finder.only_kind(kind);
      }
      if (finder.count() == 0)
      {
        fprintf(stderr,"DeferredBuffer unable to find a memory of kind %d",
                kind);
        assert(false);
      }
      return finder.first();
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    void DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
                        CB
#else
                        true
#endif
                        >::initialize_layout(size_t _alignment,
                                             bool fortran_order_dims)
    //--------------------------------------------------------------------------
    {
      if (fortran_order_dims)
      {
        for (int i = 0; i < N; i++)
          ordering[i] =
            static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
      }
      else
      {
        for (int i = 0; i < N; i++)
          ordering[i] =
            static_cast<DimensionKind>(
                static_cast<int>(LEGION_DIM_X) + N - (i + 1));
      }

      alignment = _alignment;
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              >
    void DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
                        CB
#else
                        true
#endif
                        >::initialize(Memory memory,
                                      DomainT<N,T> domain,
                                      const FT *initial_value)
    //--------------------------------------------------------------------------
    {
      bounds = domain;
      Runtime *runtime = Runtime::get_runtime();
      const std::vector<size_t> field_sizes(1,sizeof(FT));
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      int dim_order[N];
      for (int i = 0; i < N; ++i)
        dim_order[i] =
          static_cast<int>(ordering[i]) - static_cast<int>(LEGION_DIM_X);
      Realm::InstanceLayoutGeneric *layout = 
        Realm::InstanceLayoutGeneric::choose_instance_layout(bounds, 
            constraints, dim_order);
      layout->alignment_reqd = alignment;
      instance = runtime->create_task_local_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, sizeof(FT));
        const Internal::LgEvent wait_on(
            bounds.fill(dsts, no_requests, initial_value, sizeof(FT)));
        if (wait_on.exists())
          wait_on.wait();
      }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible =
        Realm::AffineAccessor<FT,N,T>::is_compatible(instance,
                                                     0/*fid*/,
                                                     bounds.bounds);
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      accessor = Realm::AffineAccessor<FT,N,T>(instance,
                                               0/*field id*/,
                                               bounds.bounds);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB
#endif
              > __CUDA_HD__
    inline FT DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              CB
#else
              true 
#endif
              >::read(const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      assert(instance.exists());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.bounds.contains(p));
#else
      assert(bounds.contains(p));
#endif
      return accessor.read(p);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline void DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              CB 
#else
              true
#endif
              >::write(const Point<N,T> &p, FT value) const
    //--------------------------------------------------------------------------
    {
      assert(instance.exists());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.bounds.contains(p));
#else
      assert(bounds.contains(p));
#endif
      accessor.write(p, value);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              CB 
#else
              true
#endif
              >::ptr(const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      assert(instance.exists());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.bounds.contains(p));
#else
      assert(bounds.contains(p));
#endif
      return accessor.ptr(p);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              CB 
#else
              true
#endif   
              >::ptr(const Rect<N,T> &r) const
    //--------------------------------------------------------------------------
    {
      assert(instance.exists());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.bounds.contains(r));
      assert(Internal::is_dense_layout(r, accessor.strides, sizeof(FT)));
#else
      assert(bounds.contains_all(r));
      if (!Internal::is_dense_layout(r, accessor.strides, sizeof(FT)))
      {
        fprintf(stderr, 
            "ERROR: Illegal request for pointer of non-dense rectangle\n");
#ifdef DEBUG_LEGION
        assert(false);
#else
        exit(ERROR_NON_DENSE_RECTANGLE);
#endif
      }
#endif
      return accessor.ptr(r.lo);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
            > __CUDA_HD__
    inline FT* DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              CB 
#else
              true
#endif          
            >::ptr(const Rect<N,T> &r, size_t strides[N]) const
    //--------------------------------------------------------------------------
    {
      assert(instance.exists());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.bounds.contains(r));
#else
      assert(bounds.contains_all(r));
#endif
      for (int i = 0; i < N; i++)
        strides[i] = accessor.strides[i] / sizeof(FT);
      return accessor.ptr(r.lo);
    }

    //--------------------------------------------------------------------------
    template<typename FT, int N, typename T
#ifdef LEGION_BOUNDS_CHECKS
              , bool CB 
#endif
              > __CUDA_HD__
    inline FT& DeferredBuffer<FT,N,T,
#ifdef LEGION_BOUNDS_CHECKS
              CB 
#else
              true
#endif
              >::operator[](const Point<N,T> &p) const
    //--------------------------------------------------------------------------
    {
      assert(instance.exists());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      assert(bounds.bounds.contains(p));
#else
      assert(bounds.contains(p));
#endif
      return accessor[p];
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(void)
      : instance(Realm::RegionInstance::NO_INST), field_size(0), dims(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory::Kind memkind,
                                                    IndexSpace space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      Machine machine = Realm::Machine::get_machine();
      Machine::MemoryQuery finder(machine);
      const Processor exec_proc = Processor::get_executing_processor();
      finder.best_affinity_to(exec_proc);
      finder.only_kind(memkind);
      if (finder.count() == 0)
      {
        finder = Machine::MemoryQuery(machine);
        finder.has_affinity_to(exec_proc);
        finder.only_kind(memkind);
      }
      Runtime *runtime = Runtime::get_runtime();
      if (finder.count() == 0)
      {
        const char *mem_names[] = {
#define MEM_NAMES(name, desc) desc,
          REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
        };
        const char *proc_names[] = {
#define PROC_NAMES(name, desc) desc,
          REALM_PROCESSOR_KINDS(PROC_NAMES)
#undef PROC_NAMES
        };
        Context ctx = Runtime::get_context();
        const Task *task = runtime->get_local_task(ctx);
        fprintf(stderr,
            "Unable to find associated %s memory for %s processor when "
            "performing an UntypedBuffer creation in task %s (UID %lld)",
            mem_names[memkind], proc_names[exec_proc.kind()],
            task->get_task_name(), task->get_unique_id());
        assert(false);
      }
      const Memory memory = finder.first();
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Realm::InstanceLayoutGeneric *layout = NULL;
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds =                                   \
                      runtime->get_index_space_domain<DIM,T>(               \
                          IndexSpaceT<DIM,T>(space));                       \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      layout->alignment_reqd = alignment;
      instance = runtime->create_task_local_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds =                               \
                      runtime->get_index_space_domain<DIM,T>(             \
                          IndexSpaceT<DIM,T>(space));                     \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            assert(false);
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory::Kind memkind,
                                                    const Domain &space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      Machine machine = Realm::Machine::get_machine();
      Machine::MemoryQuery finder(machine);
      const Processor exec_proc = Processor::get_executing_processor();
      finder.best_affinity_to(exec_proc);
      finder.only_kind(memkind);
      if (finder.count() == 0)
      {
        finder = Machine::MemoryQuery(machine);
        finder.has_affinity_to(exec_proc);
        finder.only_kind(memkind);
      }
      Runtime *runtime = Runtime::get_runtime();
      if (finder.count() == 0)
      {
        const char *mem_names[] = {
#define MEM_NAMES(name, desc) desc,
          REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
        };
        const char *proc_names[] = {
#define PROC_NAMES(name, desc) desc,
          REALM_PROCESSOR_KINDS(PROC_NAMES)
#undef PROC_NAMES
        };
        Context ctx = Runtime::get_context();
        const Task *task = runtime->get_local_task(ctx);
        fprintf(stderr,
            "Unable to find associated %s memory for %s processor when "
            "performing an UntypedBuffer creation in task %s (UID %lld)",
            mem_names[memkind], proc_names[exec_proc.kind()],
            task->get_task_name(), task->get_unique_id());
        assert(false);
      }
      const Memory memory = finder.first();
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Realm::InstanceLayoutGeneric *layout = NULL;
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds = space;                            \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      layout->alignment_reqd = alignment;
      instance = runtime->create_task_local_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds = space;                        \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            assert(false);
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory memory,
                                                    IndexSpace space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Runtime *runtime = Runtime::get_runtime();
      Realm::InstanceLayoutGeneric *layout = NULL;
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds =                                   \
                      runtime->get_index_space_domain<DIM,T>(               \
                          IndexSpaceT<DIM,T>(space));                       \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      layout->alignment_reqd = alignment;
      instance = runtime->create_task_local_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds =                               \
                      runtime->get_index_space_domain<DIM,T>(             \
                          IndexSpaceT<DIM,T>(space));                     \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            assert(false);
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(size_t fs, int d,
                                                    Memory memory,
                                                    const Domain &space,
                                                    const void *initial_value,
                                                    size_t alignment,
                                                    bool fortran_order_dims)
      : field_size(fs), dims(d)
    //--------------------------------------------------------------------------
    {
      assert(dims > 0);
      assert(dims <= LEGION_MAX_DIM);
      const std::vector<size_t> field_sizes(1, field_size);
      Realm::InstanceLayoutConstraints constraints(field_sizes, 0/*blocking*/);
      Runtime *runtime = Runtime::get_runtime();
      Realm::InstanceLayoutGeneric *layout = NULL;
      switch (dims)
      {
#define DIMFUNC(DIM)                                                        \
        case DIM:                                                           \
          {                                                                 \
            const DomainT<DIM,T> bounds = space;                            \
            int dim_order[DIM];                                             \
            if (fortran_order_dims)                                         \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = i;                                           \
            }                                                               \
            else                                                            \
            {                                                               \
              for (int i = 0; i < DIM; i++)                                 \
                dim_order[i] = DIM - (i+1);                                 \
            }                                                               \
            layout = Realm::InstanceLayoutGeneric::choose_instance_layout(  \
                bounds, constraints, dim_order);                            \
            break;                                                          \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      layout->alignment_reqd = alignment;
      instance = runtime->create_task_local_instance(memory, layout);
      if (initial_value != NULL)
      {
        Realm::ProfilingRequestSet no_requests; 
        std::vector<Realm::CopySrcDstField> dsts(1);
        dsts[0].set_field(instance, 0/*field id*/, field_size);
        Internal::LgEvent wait_on;
        switch (dims)
        {
#define DIMFUNC(DIM)                                                      \
          case DIM:                                                       \
            {                                                             \
              const DomainT<DIM,T> bounds = space;                        \
              wait_on = Internal::LgEvent(                                \
              bounds.fill(dsts, no_requests, initial_value, field_size)); \
              break;                                                      \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            assert(false);
        }
        if (wait_on.exists())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T> template<typename FT, int DIM>
    UntypedDeferredBuffer<T>::UntypedDeferredBuffer(
                                            const DeferredBuffer<FT,DIM,T> &rhs)
      : instance(rhs.instance), field_size(sizeof(FT)), dims(DIM)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T> template<typename FT, int DIM, bool BC>
    inline UntypedDeferredBuffer<T>::operator 
                                         DeferredBuffer<FT,DIM,T,BC>(void) const
    //--------------------------------------------------------------------------
    {
      static_assert(0 < DIM, "Only positive dimensions allowed");
      static_assert(DIM <= LEGION_MAX_DIM, "Exceeded LEGION_MAX_DIM");
      assert(field_size == sizeof(FT));
      assert(dims == DIM);
      DeferredBuffer<FT,DIM,T> result;
      result.instance = instance;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool is_compatible = 
        Realm::AffineAccessor<FT,DIM,T>::is_compatible(instance, 0/*field id*/);
#endif
      assert(is_compatible);
#endif
      // We can make the accessor
      result.accessor = Realm::AffineAccessor<FT,DIM,T>(instance,0/*field id*/);
#ifdef LEGION_BOUNDS_CHECKS
      result.bounds = instance.template get_indexspace<DIM,T>();
#endif
      return result;
    }

#if 0
    //--------------------------------------------------------------------------
    template<typename T>
    inline void UntypedDeferredBuffer<T>::destroy(void)
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = Runtime::get_runtime();
      runtime->destroy_task_local_instance(instance);
      instance = Realm::RegionInstance::NO_INST;
      field_size = 0;
      dims = 0;
    }
#endif

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator==(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id != rhs.id)
        return false;
      if (tid != rhs.tid)
        return false;
#ifdef DEBUG_LEGION
      assert(type_tag == rhs.type_tag);
#endif
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator!=(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((id == rhs.id) && (tid == rhs.tid))
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator<(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id < rhs.id)
        return true;
      if (id > rhs.id)
        return false;
      return (tid < rhs.tid);
    }

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator>(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id > rhs.id)
        return true;
      if (id < rhs.id)
        return false;
      return (tid > rhs.tid);
    }

    //--------------------------------------------------------------------------
    inline int IndexSpace::get_dim(void) const
    //--------------------------------------------------------------------------
    {
      if (type_tag == 0) return 0;
      return Internal::NT_TemplateHelper::get_dim(type_tag);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T>::IndexSpaceT(IndexSpaceID id, IndexTreeID tid)
      : IndexSpace(id, tid, 
          Internal::NT_TemplateHelper::template encode_tag<DIM,T>()) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T>::IndexSpaceT(void)
     : IndexSpace(0,0,Internal::NT_TemplateHelper::template encode_tag<DIM,T>())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T>::IndexSpaceT(const IndexSpace &rhs)
      : IndexSpace(rhs.get_id(), rhs.get_tree_id(), rhs.get_type_tag())
    //--------------------------------------------------------------------------
    {
      Internal::NT_TemplateHelper::template check_type<DIM,T>(type_tag);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline IndexSpaceT<DIM,T>& IndexSpaceT<DIM,T>::operator=(
                                                          const IndexSpace &rhs)
    //--------------------------------------------------------------------------
    {
      id = rhs.get_id();
      tid = rhs.get_tree_id();
      type_tag = rhs.get_type_tag();
      Internal::NT_TemplateHelper::template check_type<DIM,T>(type_tag);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator==(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id != rhs.id)
        return false;
      if (tid != rhs.tid)
        return false;
#ifdef DEBUG_LEGION
      assert(type_tag == rhs.type_tag);
#endif
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator!=(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((id == rhs.id) && (tid == rhs.tid))
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator<(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id < rhs.id)
        return true;
      if (id > rhs.id)
        return false;
      return (tid < rhs.tid);
    }

    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator>(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id > rhs.id)
        return true;
      if (id < rhs.id)
        return false;
      return (tid > rhs.tid);
    }

    //--------------------------------------------------------------------------
    inline int IndexPartition::get_dim(void) const
    //--------------------------------------------------------------------------
    {
      if (type_tag == 0) return 0;
      return Internal::NT_TemplateHelper::get_dim(type_tag);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T>::IndexPartitionT(IndexPartitionID id,IndexTreeID tid)
      : IndexPartition(id, tid,
          Internal::NT_TemplateHelper::template encode_tag<DIM,T>())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T>::IndexPartitionT(void)
      : IndexPartition(0,0,
          Internal::NT_TemplateHelper::template encode_tag<DIM,T>())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T>::IndexPartitionT(const IndexPartition &rhs)
      : IndexPartition(rhs.get_id(), rhs.get_tree_id(), rhs.get_type_tag())
    //--------------------------------------------------------------------------
    {
      Internal::NT_TemplateHelper::template check_type<DIM,T>(type_tag);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T>& IndexPartitionT<DIM,T>::operator=(
                                                      const IndexPartition &rhs)
    //--------------------------------------------------------------------------
    {
      id = rhs.get_id();
      tid = rhs.get_tree_id();
      type_tag = rhs.get_type_tag();
      Internal::NT_TemplateHelper::template check_type<DIM,T>(type_tag);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator==(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id == rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator!=(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id != rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator<(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id < rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator>(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id > rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator==(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && (index_space == rhs.index_space) 
              && (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator!=(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      return (!((*this) == rhs));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator<(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_space < rhs.index_space)
          return true;
        else if (index_space != rhs.index_space) // therefore greater than
          return false;
        else
          return field_space < rhs.field_space;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T>::LogicalRegionT(RegionTreeID tid, 
                                          IndexSpace is, FieldSpace fs)
      : LogicalRegion(tid, is, fs)
    //--------------------------------------------------------------------------
    {
      Internal::NT_TemplateHelper::template check_type<DIM,T>(
                                            is.get_type_tag());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T>::LogicalRegionT(void)
       : LogicalRegion()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T>::LogicalRegionT(const LogicalRegion &rhs)
      : LogicalRegion(rhs.get_tree_id(), rhs.get_index_space(), 
                      rhs.get_field_space())
    //--------------------------------------------------------------------------
    {
      Internal::NT_TemplateHelper::template check_type<DIM,T>(
                                rhs.get_type_tag());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T>& LogicalRegionT<DIM,T>::operator=(
                                                       const LogicalRegion &rhs)
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.get_tree_id();
      index_space = rhs.get_index_space();
      field_space = rhs.get_field_space();
      Internal::NT_TemplateHelper::template check_type<DIM,T>(
                                rhs.get_type_tag());
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator==(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && 
              (index_partition == rhs.index_partition) && 
              (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator!=(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      return (!((*this) == rhs));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator<(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_partition < rhs.index_partition)
          return true;
        else if (index_partition > rhs.index_partition)
          return false;
        else
          return (field_space < rhs.field_space);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T>::LogicalPartitionT(RegionTreeID tid, 
                                              IndexPartition pid, FieldSpace fs)
      : LogicalPartition(tid, pid, fs)
    //--------------------------------------------------------------------------
    {
      Internal::NT_TemplateHelper::template check_type<DIM,T>(
                                            pid.get_type_tag());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T>::LogicalPartitionT(void)
      : LogicalPartition()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T>::LogicalPartitionT(const LogicalPartition &rhs)
      : LogicalPartition(rhs.get_tree_id(), rhs.get_index_partition(), 
                         rhs.get_field_space())
    //--------------------------------------------------------------------------
    {
      Internal::NT_TemplateHelper::template check_type<DIM,T>(
                                            rhs.get_type_tag());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T>& LogicalPartitionT<DIM,T>::operator=(
                                                    const LogicalPartition &rhs)
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.get_tree_id();
      index_partition = rhs.get_index_partition();
      field_space = rhs.get_field_space();
      Internal::NT_TemplateHelper::template check_type<DIM,T>(
                                            rhs.get_type_tag());
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator==(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl == rhs.impl);
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator<(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl < rhs.impl);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void ArgumentMap::set_point_arg(const PT point[DIM], 
                                           const UntypedBuffer &arg, 
                                           bool replace/*= false*/)
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM,
          "ArgumentMap DIM is larger than LEGION_MAX_DIM");  
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      set_point(dp, arg, replace);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline bool ArgumentMap::remove_point(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM,
          "ArgumentMap DIM is larger than LEGION_MAX_DIM");
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      return remove_point(dp);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator==(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        if (p.impl == NULL)
          return (const_value == p.const_value);
        else
          return false;
      }
      else
        return (impl == p.impl);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator<(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        if (p.impl == NULL)
          return (const_value < p.const_value);
        else
          return true;
      }
      else
        return (impl < p.impl);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator!=(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      return !(*this == p);
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator~(RegionFlags f)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(~unsigned(f));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator|(RegionFlags left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(unsigned(left) | unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator&(RegionFlags left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(unsigned(left) & unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator^(RegionFlags left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(unsigned(left) ^ unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator|=(RegionFlags &left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l |= r;
      return left = static_cast<RegionFlags>(l);
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator&=(RegionFlags &left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l &= r;
      return left = static_cast<RegionFlags>(l);
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator^=(RegionFlags &left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l ^= r;
      return left = static_cast<RegionFlags>(l);
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& RegionRequirement::add_field(FieldID fid, 
                                             bool instance/*= true*/)
    //--------------------------------------------------------------------------
    {
      privilege_fields.insert(fid);
      if (instance)
        instance_fields.push_back(fid);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& RegionRequirement::add_fields(
                      const std::vector<FieldID>& fids, bool instance/*= true*/)
    //--------------------------------------------------------------------------
    {
      privilege_fields.insert(fids.begin(), fids.end());
      if (instance)
        instance_fields.insert(instance_fields.end(), fids.begin(), fids.end());
      return *this;
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& RegionRequirement::add_flags(
                                                          RegionFlags new_flags)
    //--------------------------------------------------------------------------
    {
      flags |= new_flags;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline void StaticDependence::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      dependent_fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline IndexSpaceRequirement& TaskLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
      return index_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& TaskLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
      return region_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_field(unsigned idx, FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_predicate_false_future(Future f)
    //--------------------------------------------------------------------------
    {
      predicate_false_future = f;
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_predicate_false_result(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      predicate_false_result = arg;
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_independent_requirements(bool independent)
    //--------------------------------------------------------------------------
    {
      independent_requirements = independent;
    }

    //--------------------------------------------------------------------------
    inline IndexSpaceRequirement& IndexTaskLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
      return index_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& IndexTaskLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
      return region_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_predicate_false_future(Future f)
    //--------------------------------------------------------------------------
    {
      predicate_false_future = f;
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_predicate_false_result(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      predicate_false_result = arg;
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_independent_requirements(
                                                               bool independent)
    //--------------------------------------------------------------------------
    {
      independent_requirements = independent;
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_field(FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
      requirement.add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline unsigned CopyLauncher::add_copy_requirements(
                     const RegionRequirement &src, const RegionRequirement &dst)
    //--------------------------------------------------------------------------
    {
      unsigned result = src_requirements.size();
#ifdef DEBUG_LEGION
      assert(result == dst_requirements.size());
#endif
      src_requirements.push_back(src);
      dst_requirements.push_back(dst);
      return result;
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_src_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < src_requirements.size());
#endif
      src_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_dst_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < dst_requirements.size());
#endif
      dst_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_src_indirect_field(FieldID src_idx_field,
                            const RegionRequirement &req, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(req);
      src_indirect_requirements.back().add_field(src_idx_field, inst);
      src_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_dst_indirect_field(FieldID dst_idx_field,
                            const RegionRequirement &req, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(req);
      dst_indirect_requirements.back().add_field(dst_idx_field, inst);
      dst_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& CopyLauncher::add_src_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(req);
      src_indirect_is_range.push_back(range);
      return src_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& CopyLauncher::add_dst_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(req);
      dst_indirect_is_range.push_back(range);
      return dst_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline unsigned IndexCopyLauncher::add_copy_requirements(
                     const RegionRequirement &src, const RegionRequirement &dst)
    //--------------------------------------------------------------------------
    {
      unsigned result = src_requirements.size();
#ifdef DEBUG_LEGION
      assert(result == dst_requirements.size());
#endif
      src_requirements.push_back(src);
      dst_requirements.push_back(dst);
      return result;
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_src_field(unsigned idx,
                                                 FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < src_requirements.size());
#endif
      src_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_dst_field(unsigned idx,
                                                 FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < dst_requirements.size());
#endif
      dst_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_src_indirect_field(FieldID src_idx_field,
                              const RegionRequirement &r, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(r);
      src_indirect_requirements.back().add_field(src_idx_field, inst);
      src_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_dst_indirect_field(FieldID dst_idx_field,
                              const RegionRequirement &r, bool range, bool inst)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(r);
      dst_indirect_requirements.back().add_field(dst_idx_field, inst);
      dst_indirect_is_range.push_back(range);
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& IndexCopyLauncher::add_src_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      src_indirect_requirements.push_back(req);
      src_indirect_is_range.push_back(range);
      return src_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& IndexCopyLauncher::add_dst_indirect_field(
                                       const RegionRequirement &req, bool range)
    //--------------------------------------------------------------------------
    {
      dst_indirect_requirements.push_back(req);
      dst_indirect_is_range.push_back(range);
      return dst_indirect_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::set_argument(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      argument = arg;
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::set_future(Future f)
    //--------------------------------------------------------------------------
    {
      future = f;
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_wait_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      wait_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_arrival_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      arrive_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_arrival_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::set_argument(UntypedBuffer arg)
    //--------------------------------------------------------------------------
    {
      argument = arg;
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::set_future(Future f)
    //--------------------------------------------------------------------------
    {
      future = f;
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_wait_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      wait_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_arrival_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      arrive_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_wait_handshake(LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_arrival_handshake(
                                                      LegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_file(const char *name,
                                            const std::vector<FieldID> &fields,
                                            LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_POSIX_FILE);
#endif
      file_name = name;
      mode = m;
      file_fields = fields;
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_hdf5(const char *name,
                                const std::map<FieldID,const char*> &field_map,
                                LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_HDF5_FILE);
#endif
      file_name = name;
      mode = m;
      field_files = field_map;
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_array_aos(void *base, bool column_major,
                          const std::vector<FieldID> &fields, Memory mem,
                          const std::map<FieldID,size_t> *alignments /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      constraints.add_constraint(PointerConstraint(mem, uintptr_t(base)));
      if (mem.exists())
        constraints.add_constraint(MemoryConstraint(mem.kind()));
      constraints.add_constraint(
          FieldConstraint(fields, true/*contiugous*/, true/*inorder*/));
      const int dims = handle.get_index_space().get_dim();
      std::vector<DimensionKind> dim_order(dims+1);
      // Field dimension first for AOS
      dim_order[0] = LEGION_DIM_F;
      if (column_major)
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx+1] = (DimensionKind)(LEGION_DIM_X + idx); 
      }
      else
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx+1] = (DimensionKind)(LEGION_DIM_X + (dims-1) - idx);
      }
      constraints.add_constraint(
          OrderingConstraint(dim_order, false/*contiguous*/));
      if (alignments != NULL)
        for (std::map<FieldID,size_t>::const_iterator it = alignments->begin();
             it != alignments->end(); it++)
          constraints.add_constraint(
              AlignmentConstraint(it->first, LEGION_GE_EK, it->second));
      privilege_fields.insert(fields.begin(), fields.end());
    }
    
    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_array_soa(void *base, bool column_major,
                          const std::vector<FieldID> &fields, Memory mem,
                          const std::map<FieldID,size_t> *alignments /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      constraints.add_constraint(PointerConstraint(mem, uintptr_t(base)));
      if (mem.exists())
        constraints.add_constraint(MemoryConstraint(mem.kind()));
      constraints.add_constraint(
          FieldConstraint(fields, true/*contiguous*/, true/*inorder*/));
      const int dims = handle.get_index_space().get_dim();
      std::vector<DimensionKind> dim_order(dims+1);
      if (column_major)
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx] = (DimensionKind)(LEGION_DIM_X + idx); 
      }
      else
      {
        for (int idx = 0; idx < dims; idx++)
          dim_order[idx] = (DimensionKind)(LEGION_DIM_X + (dims-1) - idx);
      }
      // Field dimension last for SOA 
      dim_order[dims] = LEGION_DIM_F;
      constraints.add_constraint(
          OrderingConstraint(dim_order, false/*contiguous*/));
      if (alignments != NULL)
        for (std::map<FieldID,size_t>::const_iterator it = alignments->begin();
             it != alignments->end(); it++)
          constraints.add_constraint(
              AlignmentConstraint(it->first, LEGION_GE_EK, it->second));
      privilege_fields.insert(fields.begin(), fields.end());
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_file(LogicalRegion handle,
                                             const char *file_name,
                                             const std::vector<FieldID> &fields,
                                             LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_POSIX_FILE);
#endif
      if (handles.empty())
        mode = m;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
        assert(mode == m);
#endif
#endif
      handles.push_back(handle);
      file_names.push_back(file_name);
      if (!file_fields.empty())
      {
#ifdef DEBUG_LEGION
        assert(fields.size() == file_fields.size());
#ifndef NDEBUG
        for (unsigned idx = 0; idx < fields.size(); idx++)
          assert(file_fields[idx] == fields[idx]);
#endif
#endif
      }
      else
        file_fields = fields;
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_hdf5(LogicalRegion handle,
                                 const char *file_name,
                                 const std::map<FieldID,const char*> &field_map,
                                 LegionFileMode m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource == LEGION_EXTERNAL_HDF5_FILE);
#endif
      if (handles.empty())
        mode = m;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
        assert(mode == m);
#endif
#endif
      handles.push_back(handle);
      file_names.push_back(file_name);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool first = field_files.empty();
#endif
#endif
      for (std::map<FieldID,const char*>::const_iterator it =
            field_map.begin(); it != field_map.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(first || (field_files.find(it->first) != field_files.end()));
#endif
        field_files[it->first].push_back(it->second);
      }
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_array_aos(LogicalRegion handle,
              void *base, bool column_major, const std::vector<FieldID> &fields,
              Memory mem, const std::map<FieldID,size_t> *alignments)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      if (handles.empty())
      {
        constraints.add_constraint(
            FieldConstraint(fields, true/*contiugous*/, true/*inorder*/));
        const int dims = handle.get_index_space().get_dim();
        std::vector<DimensionKind> dim_order(dims+1);
        // Field dimension first for AOS
        dim_order[0] = LEGION_DIM_F;
        if (column_major)
        {
          for (int idx = 0; idx < dims; idx++)
            dim_order[idx+1] = (DimensionKind)(LEGION_DIM_X + idx); 
        }
        else
        {
          for (int idx = 0; idx < dims; idx++)
            dim_order[idx+1] = (DimensionKind)(LEGION_DIM_X + (dims-1) - idx);
        }
        constraints.add_constraint(
            OrderingConstraint(dim_order, false/*contiguous*/));
        if (alignments != NULL)
          for (std::map<FieldID,size_t>::const_iterator it =
                alignments->begin(); it != alignments->end(); it++)
            constraints.add_constraint(
                AlignmentConstraint(it->first, LEGION_GE_EK, it->second));
        privilege_fields.insert(fields.begin(), fields.end());
      }
#ifdef DEBUG_LEGION
      else
      {
        // Check that the fields are the same
        assert(fields.size() == privilege_fields.size());
        for (std::vector<FieldID>::const_iterator it =
              fields.begin(); it != fields.end(); it++)
          assert(privilege_fields.find(*it) != privilege_fields.end());
        // Check that the layouts are the same
        const OrderingConstraint &order = constraints.ordering_constraint;
        assert(order.ordering.front() == LEGION_DIM_F);
        const int dims = handle.get_index_space().get_dim();
        assert(dims == handles.back().get_index_space().get_dim());
        if (column_major)
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx+1] == ((DimensionKind)LEGION_DIM_X+idx));
        }
        else
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx+1] == 
                ((DimensionKind)(LEGION_DIM_X + (dims-1) - idx)));
        }
        // Check that the alignments are the same
        if (alignments != NULL)
        {
          assert(alignments->size() == 
                  constraints.alignment_constraints.size());
          unsigned index = 0;
          for (std::map<FieldID,size_t>::const_iterator it =
                alignments->begin(); it != alignments->end(); it++, index++)
          {
            const AlignmentConstraint &alignment = 
              constraints.alignment_constraints[index];
            assert(alignment.fid == it->first);
            assert(alignment.alignment == it->second);
          }
        }
      }
#endif
      handles.push_back(handle);
      pointers.emplace_back(PointerConstraint(mem, uintptr_t(base)));
    }

    //--------------------------------------------------------------------------
    inline void IndexAttachLauncher::attach_array_soa(LogicalRegion handle,
              void *base, bool column_major, const std::vector<FieldID> &fields,
              Memory mem, const std::map<FieldID,size_t> *alignments)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
      assert(resource == LEGION_EXTERNAL_INSTANCE);
#endif
      if (handles.empty())
      {
        constraints.add_constraint(
            FieldConstraint(fields, true/*contiguous*/, true/*inorder*/));
        const int dims = handle.get_index_space().get_dim();
        std::vector<DimensionKind> dim_order(dims+1);
        if (column_major)
        {
          for (int idx = 0; idx < dims; idx++)
            dim_order[idx] = (DimensionKind)(LEGION_DIM_X + idx); 
        }
        else
        {
          for (int idx = 0; idx < dims; idx++)
            dim_order[idx] = (DimensionKind)(LEGION_DIM_X + (dims-1) - idx);
        }
        // Field dimension last for SOA 
        dim_order[dims] = LEGION_DIM_F;
        constraints.add_constraint(
            OrderingConstraint(dim_order, false/*contiguous*/));
        if (alignments != NULL)
          for (std::map<FieldID,size_t>::const_iterator it =
                alignments->begin(); it != alignments->end(); it++)
            constraints.add_constraint(
                AlignmentConstraint(it->first, LEGION_GE_EK, it->second));
        privilege_fields.insert(fields.begin(), fields.end());
      }
#ifdef DEBUG_LEGION
      else
      {
        // Check that the fields are the same
        assert(fields.size() == privilege_fields.size());
        for (std::vector<FieldID>::const_iterator it =
              fields.begin(); it != fields.end(); it++)
          assert(privilege_fields.find(*it) != privilege_fields.end());
        // Check that the layouts are the same
        const OrderingConstraint &order = constraints.ordering_constraint;
        const int dims = handle.get_index_space().get_dim();
        assert(dims == handles.back().get_index_space().get_dim());
        if (column_major)
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx] == ((DimensionKind)LEGION_DIM_X+idx));
        }
        else
        {
          for (int idx = 0; idx < dims; idx++)
            assert(order.ordering[idx] == 
                ((DimensionKind)(LEGION_DIM_X + (dims-1) - idx)));
        }
        assert(order.ordering.back() == LEGION_DIM_F);
        // Check that the alignments are the same
        if (alignments != NULL)
        {
          assert(alignments->size() == 
                  constraints.alignment_constraints.size());
          unsigned index = 0;
          for (std::map<FieldID,size_t>::const_iterator it =
                alignments->begin(); it != alignments->end(); it++, index++)
          {
            const AlignmentConstraint &alignment = 
              constraints.alignment_constraints[index];
            assert(alignment.fid == it->first);
            assert(alignment.alignment == it->second);
          }
        }
      }
#endif
      handles.push_back(handle);
      pointers.emplace_back(PointerConstraint(mem, uintptr_t(base)));
    }

    //--------------------------------------------------------------------------
    inline void PredicateLauncher::add_predicate(const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      predicates.push_back(pred);
    }

    //--------------------------------------------------------------------------
    inline void TimingLauncher::add_precondition(const Future &f)
    //--------------------------------------------------------------------------
    {
      preconditions.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void MustEpochLauncher::add_single_task(const DomainPoint &point,
                                                   const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      single_tasks.push_back(launcher);
      single_tasks.back().point = point;
    }

    //--------------------------------------------------------------------------
    inline void MustEpochLauncher::add_index_task(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      index_tasks.push_back(launcher);
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                         add_constraint(const SpecializedConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                              add_constraint(const MemoryConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                            add_constraint(const OrderingConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                           add_constraint(const SplittingConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                               add_constraint(const FieldConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                           add_constraint(const DimensionConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                           add_constraint(const AlignmentConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                              add_constraint(const OffsetConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                             add_constraint(const PointerConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                                 add_constraint(const ISAConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                           add_constraint(const ProcessorConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                            add_constraint(const ResourceConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                              add_constraint(const LaunchConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                          add_constraint(const ColocationConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
             add_layout_constraint_set(unsigned index, LayoutConstraintID desc)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_layout_constraint(index, desc);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::set_leaf(bool is_leaf /*= true*/)
    //--------------------------------------------------------------------------
    {
      leaf_variant = is_leaf;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::set_inner(bool is_inner /*= true*/)
    //--------------------------------------------------------------------------
    {
      inner_variant = is_inner;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::set_idempotent(bool is_idemp/*= true*/)
    //--------------------------------------------------------------------------
    {
      idempotent_variant = is_idemp;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::set_replicable(bool is_repl/*= true*/)
    //--------------------------------------------------------------------------
    {
      replicable_variant = is_repl;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::add_generator_task(TaskID tid)
    //--------------------------------------------------------------------------
    {
      generator_tasks.insert(tid); 
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Future::get_result(bool silence_warnings,
                                const char *warning_string) const
    //--------------------------------------------------------------------------
    {
      // Unpack the value using LegionSerialization in case
      // the type has an alternative method of unpacking
      return 
        LegionSerialization::unpack<T>(*this, silence_warnings, warning_string);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline const T& Future::get_reference(bool silence_warnings,
                                          const char *warning_string) const
    //--------------------------------------------------------------------------
    {
      return *((const T*)get_untyped_result(silence_warnings, warning_string,
                                            true/*check size*/, sizeof(T)));
    }

    //--------------------------------------------------------------------------
    inline const void* Future::get_untyped_pointer(bool silence_warnings,
                                               const char *warning_string) const
    //--------------------------------------------------------------------------
    {
      return get_untyped_result(silence_warnings, warning_string, false);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Future::get(void)
    //--------------------------------------------------------------------------
    {
      return get_result<T>();
    }

    //--------------------------------------------------------------------------
    inline bool Future::valid(void) const
    //--------------------------------------------------------------------------
    {
      return (impl != NULL);
    }

    //--------------------------------------------------------------------------
    inline void Future::wait(void) const
    //--------------------------------------------------------------------------
    {
      get_void_result();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline Future Future::from_value(Runtime *rt, const T &value)
    //--------------------------------------------------------------------------
    {
      return LegionSerialization::from_value(rt, &value);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline Future Future::from_untyped_pointer(Runtime *rt,
							  const void *buffer,
							  size_t bytes)
    //--------------------------------------------------------------------------
    {
      return LegionSerialization::from_value_helper(rt, buffer, bytes,
						    false /*!owned*/);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T FutureMap::get_result(const DomainPoint &dp, bool silence_warnings,
                                   const char *warning_string) const
    //--------------------------------------------------------------------------
    {
      Future f = get_future(dp);
      return f.get_result<T>(silence_warnings, warning_string);
    }

    //--------------------------------------------------------------------------
    template<typename RT, typename PT, unsigned DIM>
    inline RT FutureMap::get_result(const PT point[DIM]) const
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM,
          "FutureMap DIM is larger than LEGION_MAX_DIM");
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      Future f = get_future(dp);
      return f.get_result<RT>();
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline Future FutureMap::get_future(const PT point[DIM]) const
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM,
          "FutureMap DIM is larger than LEGION_MAX_DIM");
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      return get_future(dp);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void FutureMap::get_void_result(const PT point[DIM]) const
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM,
          "FutureMap DIM is larger than LEGION_MAX_DIM");
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      Future f = get_future(dp);
      return f.get_void_result();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainT<DIM,T> PhysicalRegion::get_bounds(void) const
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> result;
      get_bounds(&result, Internal::NT_TemplateHelper::encode_tag<DIM,T>());
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalRegion::operator DomainT<DIM,T>(void) const
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> result;
      get_bounds(&result, Internal::NT_TemplateHelper::encode_tag<DIM,T>());
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalRegion::operator Rect<DIM,T>(void) const
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> result;
      get_bounds(&result, Internal::NT_TemplateHelper::encode_tag<DIM,T>());
#ifdef DEBUG_LEGION
      assert(result.dense());
#endif
      return result.bounds;
    }

    //--------------------------------------------------------------------------
    inline bool PieceIterator::valid(void) const
    //--------------------------------------------------------------------------
    {
      return (impl != NULL) && (index >= 0);
    }

    //--------------------------------------------------------------------------
    inline PieceIterator::operator bool(void) const
    //--------------------------------------------------------------------------
    {
      return valid();
    }

    //--------------------------------------------------------------------------
    inline bool PieceIterator::operator()(void) const
    //--------------------------------------------------------------------------
    {
      return valid();
    }

    //--------------------------------------------------------------------------
    inline const Domain& PieceIterator::operator*(void) const
    //--------------------------------------------------------------------------
    {
      return current_piece;
    }

    //--------------------------------------------------------------------------
    inline const Domain* PieceIterator::operator->(void) const
    //--------------------------------------------------------------------------
    {
      return &current_piece;
    }

    //--------------------------------------------------------------------------
    inline PieceIterator& PieceIterator::operator++(void)
    //--------------------------------------------------------------------------
    {
      step();
      return *this;
    }

    //--------------------------------------------------------------------------
    inline PieceIterator PieceIterator::operator++(int)
    //--------------------------------------------------------------------------
    {
      PieceIterator result = *this;
      step();
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool PieceIterator::operator<(const PieceIterator &rhs) const
    //--------------------------------------------------------------------------
    {
      if (impl < rhs.impl)
        return true;
      if (impl > rhs.impl)
        return false;
      if (index < rhs.index)
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    inline bool PieceIterator::operator==(const PieceIterator &rhs) const
    //--------------------------------------------------------------------------
    {
      if (impl != rhs.impl)
        return false;
      return index == rhs.index;
    }

    //--------------------------------------------------------------------------
    inline bool PieceIterator::operator!=(const PieceIterator &rhs) const
    //--------------------------------------------------------------------------
    {
      if (impl != rhs.impl)
        return true;
      return index != rhs.index;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T>::PieceIteratorT(void) : PieceIterator()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T>::PieceIteratorT(const PieceIteratorT &rhs)
      : PieceIterator(rhs), current_rect(rhs.current_rect)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T>::PieceIteratorT(PieceIteratorT &&rhs)
      : PieceIterator(rhs), current_rect(rhs.current_rect)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T>::PieceIteratorT(const PhysicalRegion &region,
        FieldID fid, bool privilege_only, bool silence_warn, const char *warn)
      : PieceIterator(region, fid, privilege_only, silence_warn, warn)
    //--------------------------------------------------------------------------
    {
      if (valid())
        current_rect = current_piece;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T>& PieceIteratorT<DIM,T>::operator=(
                                                      const PieceIteratorT &rhs)
    //--------------------------------------------------------------------------
    {
      PieceIterator::operator=(rhs);
      current_rect = rhs.current_rect;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T>& PieceIteratorT<DIM,T>::operator=(
                                                           PieceIteratorT &&rhs)
    //--------------------------------------------------------------------------
    {
      PieceIterator::operator=(rhs);
      current_rect = rhs.current_rect;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline bool PieceIteratorT<DIM,T>::step(void)
    //--------------------------------------------------------------------------
    {
      const bool result = PieceIterator::step();
      current_rect = current_piece;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline const Rect<DIM,T>& PieceIteratorT<DIM,T>::operator*(void) const
    //--------------------------------------------------------------------------
    {
      return current_rect;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline const Rect<DIM,T>* PieceIteratorT<DIM,T>::operator->(void) const
    //--------------------------------------------------------------------------
    {
      return &current_rect;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T>& PieceIteratorT<DIM,T>::operator++(void)
    //--------------------------------------------------------------------------
    {
      step();
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline PieceIteratorT<DIM,T> PieceIteratorT<DIM,T>::operator++(int)
    //--------------------------------------------------------------------------
    {
      PieceIteratorT<DIM,T> result = *this;
      step();
      return result;
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline SpanIterator<PM,FT,DIM,T>::SpanIterator(const PhysicalRegion &region,
                   FieldID fid, size_t actual_field_size, bool check_field_size, 
                   bool priv, bool silence_warnings, const char *warning_string)
      : piece_iterator(PieceIteratorT<DIM,T>(region, fid, priv)),
        partial_piece(false)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> is;
      const Realm::RegionInstance instance = 
        region.get_instance_info(PM, fid, actual_field_size, &is,
            Internal::NT_TemplateHelper::encode_tag<DIM,T>(), warning_string,
            silence_warnings, false/*generic accessor*/, check_field_size);
      if (!Realm::MultiAffineAccessor<FT,DIM,T>::is_compatible(instance, fid, 
                                                               is.bounds))
        region.report_incompatible_accessor("SpanIterator", instance, fid);
      accessor = Realm::MultiAffineAccessor<FT,DIM,T>(instance, fid, is.bounds);
      // initialize the first span
      step();
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline bool SpanIterator<PM,FT,DIM,T>::valid(void) const
    //--------------------------------------------------------------------------
    {
      return !current.empty();
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline bool SpanIterator<PM,FT,DIM,T>::step(void)
    //--------------------------------------------------------------------------
    {
      // Handle the remains of a partial piece if that is what we're doing
      if (partial_piece)
      {
        bool carry = false;
        for (int idx = 0; idx < DIM; idx++)
        {
          const int dim = dim_order[idx];
          if (carry || (dim == partial_step_dim))
          {
            if (partial_step_point[dim] < piece_iterator->hi[dim])
            {
              partial_step_point[dim] += 1;
              carry = false;
              break;
            }
            // carry case so reset and roll-over
            partial_step_point[dim] = piece_iterator->lo[dim];
            carry = true;
          }
          // Skip any dimensions before the partial step dim
        }
        // Make the next span
        current = Span<FT,PM>(accessor.ptr(partial_step_point),
                              current.size(), current.step());
        // See if we are done with this partial piece
        if (carry)
          partial_piece = false; 
        return true;
      }
      current = Span<FT,PM>(); // clear this for the next iteration
      // Otherwise try to group as many rectangles together as we can
      while (piece_iterator.valid())
      {
        size_t strides[DIM];
        FT *ptr = accessor.ptr(*piece_iterator, strides); 
#ifdef DEBUG_LEGION
        // If we ever hit this it is a runtime error because the 
        // runtime should already be guaranteeing these rectangles
        // are inside of pieces for the instance
        assert(ptr != NULL);
#endif         
        // Find the minimum stride and see if this piece is dense
        size_t min_stride = SIZE_MAX;
        for (int dim = 0; dim < DIM; dim++)
          if (strides[dim] < min_stride)
            min_stride = strides[dim];
        if (Internal::is_dense_layout(*piece_iterator, strides, min_stride))
        {
          const size_t volume = piece_iterator->volume();
          if (!current.empty())
          {
            uintptr_t base = current.get_base();
            // See if we can append to the current span
            if ((current.step() == min_stride) &&
                ((base + (current.size() * min_stride)) == uintptr_t(ptr)))
              current = 
                Span<FT,PM>(current.data(), current.size() + volume, min_stride);
            else // Save this rectangle for the next iteration
              break;
          }
          else // Start a new span
            current = Span<FT,PM>(ptr, volume, min_stride);
        }
        else
        {
          // Not a uniform stride, so go to the partial piece case
          if (current.empty())
          {
            partial_piece = true;
            // Compute the dimension order from smallest to largest
            size_t stride_floor = 0;
            for (int idx = 0; idx < DIM; idx++)
            {
              int index = -1;
              size_t local_min = SIZE_MAX;
              for (int dim = 0; dim < DIM; dim++)
              {
                if (strides[dim] <= stride_floor)
                  continue;
                if (strides[dim] < local_min)
                {
                  local_min = strides[dim];
                  index = dim;
                }
              }
#ifdef DEBUG_LEGION
              assert(index >= 0); 
#endif
              dim_order[idx] = index;
              stride_floor = local_min;
            }
            // See which dimensions we can handle at once and which ones
            // we are going to need to walk over
            size_t extent = 1;
            size_t exp_offset = min_stride;
            partial_step_dim = -1;
            for (int idx = 0; idx < DIM; idx++)
            {
              const int dim = dim_order[idx];
              if (strides[dim] == exp_offset)
              {
                size_t pitch =
                  ((piece_iterator->hi[dim] - piece_iterator->lo[dim]) + 1); 
                exp_offset *= pitch;
                extent *= pitch;
              }
              // First dimension that is not contiguous
              partial_step_dim = dim;
              break;
            }
#ifdef DEBUG_LEGION
            assert(partial_step_dim >= 0);
#endif
            partial_step_point = piece_iterator->lo;
            current = 
              Span<FT,PM>(accessor.ptr(partial_step_point), extent, min_stride);
          }
          // No matter what we are breaking out here
          break;
        }
        // Step the piece iterator for the next iteration
        piece_iterator.step();
      }
      return valid();
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline SpanIterator<PM,FT,DIM,T>::operator bool(void) const
    //--------------------------------------------------------------------------
    {
      return valid();
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline bool SpanIterator<PM,FT,DIM,T>::operator()(void) const
    //--------------------------------------------------------------------------
    {
      return valid();
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline const Span<FT,PM>& SpanIterator<PM,FT,DIM,T>::operator*(void) const
    //--------------------------------------------------------------------------
    {
      return current;
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline const Span<FT,PM>* SpanIterator<PM,FT,DIM,T>::operator->(void) const
    //--------------------------------------------------------------------------
    {
      return &current;
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline SpanIterator<PM,FT,DIM,T>& SpanIterator<PM,FT,DIM,T>::operator++(
                                                                           void)
    //--------------------------------------------------------------------------
    {
      step();
      return *this;
    }

    //--------------------------------------------------------------------------
    template<PrivilegeMode PM, typename FT, int DIM, typename T>
    inline SpanIterator<PM,FT,DIM,T> SpanIterator<PM,FT,DIM,T>::operator++(int)
    //--------------------------------------------------------------------------
    {
      SpanIterator<PM,FT,DIM,T> result = *this;
      step();
      return result;
    }

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    //--------------------------------------------------------------------------
    inline bool IndexIterator::has_next(void) const
    //--------------------------------------------------------------------------
    {
      return is_iterator.valid;
    }
    
    //--------------------------------------------------------------------------
    inline ptr_t IndexIterator::next(void)
    //--------------------------------------------------------------------------
    {
      if (!rect_iterator.valid)
        rect_iterator = 
          Realm::PointInRectIterator<1,coord_t>(is_iterator.rect);
      const ptr_t result = rect_iterator.p[0];
      rect_iterator.step();
      if (!rect_iterator.valid)
        is_iterator.step();
      return result;
    }

    //--------------------------------------------------------------------------
    inline ptr_t IndexIterator::next_span(size_t& act_count, size_t req_count)
    //--------------------------------------------------------------------------
    {
      if (rect_iterator.valid)
      {
        // If we have a rect iterator we just go to the end of the rectangle
        const ptr_t result = rect_iterator.p[0];
        const ptr_t last = is_iterator.rect.hi[0];
        act_count = (last.value - result.value) + 1;
        if (act_count <= req_count)
        {
          rect_iterator.valid = false;
          is_iterator.step();
        }
        else
	{
          rect_iterator.p[0] = result.value + req_count;
	  act_count = req_count;
	}
        return result;
      }
      else
      {
        // Consume the whole rectangle
        const ptr_t result = is_iterator.rect.lo[0];
        const ptr_t last = is_iterator.rect.hi[0];
        act_count = (last.value - result.value) + 1;
        if (act_count > req_count)
        {
          rect_iterator = 
            Realm::PointInRectIterator<1,coord_t>(is_iterator.rect);
          rect_iterator.p[0] = result.value + req_count;
	  act_count = req_count;
        }
        else
        {
          rect_iterator.valid = false;
          is_iterator.step();
        }
        return result;
      }
    }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::create_index_space(Context ctx, 
                              const Rect<DIM,T> &bounds, const char *provenance)
    //--------------------------------------------------------------------------
    {
      const Domain domain(bounds);
      return IndexSpaceT<DIM,T>(create_index_space(ctx, domain,
        Internal::NT_TemplateHelper::template encode_tag<DIM,T>(), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::create_index_space(Context ctx, 
                           const DomainT<DIM,T> &bounds, const char *provenance)
    //--------------------------------------------------------------------------
    {
      const Domain domain(bounds);
      return IndexSpaceT<DIM,T>(create_index_space(ctx, domain,
        Internal::NT_TemplateHelper::template encode_tag<DIM,T>(), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::create_index_space(Context ctx, 
                                   const Future &future, const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<DIM,T>(create_index_space(ctx, DIM, future,
        Internal::NT_TemplateHelper::template encode_tag<DIM,T>(), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::create_index_space(Context ctx,
               const std::vector<Point<DIM,T> > &points, const char *provenance)
    //--------------------------------------------------------------------------
    {
      // C++ type system is dumb
      std::vector<Realm::Point<DIM,T> > realm_points(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
        realm_points[idx] = points[idx];
      const DomainT<DIM,T> realm_is((Realm::IndexSpace<DIM,T>(realm_points)));
      const Domain domain(realm_is);
      return IndexSpaceT<DIM,T>(create_index_space(ctx, domain,
        Internal::NT_TemplateHelper::template encode_tag<DIM,T>(), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::create_index_space(Context ctx,
                 const std::vector<Rect<DIM,T> > &rects, const char *provenance)
    //--------------------------------------------------------------------------
    {
      // C++ type system is dumb
      std::vector<Realm::Rect<DIM,T> > realm_rects(rects.size());
      for (unsigned idx = 0; idx < rects.size(); idx++)
        realm_rects[idx] = rects[idx];
      const DomainT<DIM,T> realm_is((Realm::IndexSpace<DIM,T>(realm_rects)));
      const Domain domain(realm_is);
      return IndexSpaceT<DIM,T>(create_index_space(ctx, domain,
        Internal::NT_TemplateHelper::template encode_tag<DIM,T>(), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::union_index_spaces(Context ctx,
         const std::vector<IndexSpaceT<DIM,T> > &spaces, const char *provenance)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpace> handles(spaces.size());
      for (unsigned idx = 0; idx < spaces.size(); idx++)
        handles[idx] = spaces[idx];
      return IndexSpaceT<DIM,T>(union_index_spaces(ctx, handles, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::intersect_index_spaces(Context ctx,
         const std::vector<IndexSpaceT<DIM,T> > &spaces, const char *provenance)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpace> handles(spaces.size());
      for (unsigned idx = 0; idx < spaces.size(); idx++)
        handles[idx] = spaces[idx];
      return IndexSpaceT<DIM,T>(intersect_index_spaces(ctx,handles,provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::subtract_index_spaces(Context ctx,
      IndexSpaceT<DIM,T> left, IndexSpaceT<DIM,T> right, const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<DIM,T>(subtract_index_spaces(ctx, 
                              IndexSpace(left), IndexSpace(right), provenance));
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IndexPartition Runtime::create_index_partition(Context ctx,
      IndexSpace parent, const T& mapping, Color part_color /*= AUTO_GENERATE*/)
    //--------------------------------------------------------------------------
    {
      LegionRuntime::Arrays::Rect<T::IDIM> parent_rect = 
        get_index_space_domain(ctx, parent).get_rect<T::IDIM>();
      LegionRuntime::Arrays::Rect<T::ODIM> color_space = 
        mapping.image_convex(parent_rect);
      DomainPointColoring c;
      for (typename T::PointInOutputRectIterator pir(color_space); 
          pir; pir++) 
      {
        LegionRuntime::Arrays::Rect<T::IDIM> preimage = mapping.preimage(pir.p);
#ifdef DEBUG_LEGION
        assert(mapping.preimage_is_dense(pir.p));
#endif
        c[DomainPoint::from_point<T::IDIM>(pir.p)] =
          Domain::from_rect<T::IDIM>(preimage.intersection(parent_rect));
      }
      return create_index_partition(ctx, parent, 
              Domain::from_rect<T::ODIM>(color_space), c, 
              LEGION_DISJOINT_KIND, part_color);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_equal_partition(Context ctx,
                              IndexSpaceT<DIM,T> parent,
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              size_t granularity, Color color, const char *prov)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_equal_partition(ctx,
                                    IndexSpace(parent), IndexSpace(color_space),
                                    granularity, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_weights(Context ctx,
                          IndexSpaceT<DIM,T> parent,
                          const std::map<Point<COLOR_DIM,COLOR_T>,int> &weights,
                          IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                          size_t granularity, Color color, const char *prov)
    //--------------------------------------------------------------------------
    {
      std::map<DomainPoint,int> untyped_weights;
      for (typename std::map<Point<COLOR_DIM,COLOR_T>,int>::const_iterator it =
            weights.begin(); it != weights.end(); it++)
        untyped_weights[DomainPoint(it->first)] = it->second;
      return IndexPartitionT<DIM,T>(create_partition_by_weights(ctx,
                            IndexSpace(parent), untyped_weights, 
                            IndexSpace(color_space), granularity, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_weights(Context ctx,
                       IndexSpaceT<DIM,T> parent,
                       const std::map<Point<COLOR_DIM,COLOR_T>,size_t> &weights,
                       IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                       size_t granularity, Color color, const char *prov)
    //--------------------------------------------------------------------------
    {
      std::map<DomainPoint,size_t> untyped_weights;
      for (typename std::map<Point<COLOR_DIM,COLOR_T>,size_t>::const_iterator
            it = weights.begin(); it != weights.end(); it++)
        untyped_weights[DomainPoint(it->first)] = it->second;
      return IndexPartitionT<DIM,T>(create_partition_by_weights(ctx,
                            IndexSpace(parent), untyped_weights, 
                            IndexSpace(color_space), granularity, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_weights(Context ctx,
                                     IndexSpaceT<DIM,T> parent,
                                     const FutureMap &weights,
                                     IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                                     size_t granularity, Color color,
                                     const char *prov)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_partition_by_weights(ctx,
                            IndexSpace(parent), weights, 
                            IndexSpace(color_space), granularity, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_union(Context ctx,
                              IndexSpaceT<DIM,T> parent,
                              IndexPartitionT<DIM,T> handle1,
                              IndexPartitionT<DIM,T> handle2,
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              PartitionKind part_kind, Color color,
                              const char *prov)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_partition_by_union(ctx,
           IndexSpace(parent), IndexPartition(handle1),
           IndexPartition(handle2), IndexSpace(color_space),
           part_kind, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_intersection(
                              Context ctx,
                              IndexSpaceT<DIM,T> parent,
                              IndexPartitionT<DIM,T> handle1,
                              IndexPartitionT<DIM,T> handle2,
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              PartitionKind part_kind, Color color,
                              const char *prov)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_partition_by_intersection(ctx,
           IndexSpace(parent), IndexPartition(handle1),
           IndexPartition(handle2), IndexSpace(color_space),
           part_kind, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_intersection(
                                              Context ctx,
                                              IndexSpaceT<DIM,T> parent,
                                              IndexPartitionT<DIM,T> partition,
                                              PartitionKind part_kind, 
                                              Color color, bool safe,
                                              const char *prov)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_partition_by_intersection(ctx,
                        IndexSpace(parent), IndexPartition(partition),
                        part_kind, color, safe, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_difference(Context ctx,
                              IndexSpaceT<DIM,T> parent,
                              IndexPartitionT<DIM,T> handle1,
                              IndexPartitionT<DIM,T> handle2,
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              PartitionKind part_kind, Color color,
                              const char *prov)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_partition_by_difference(ctx,
           IndexSpace(parent), IndexPartition(handle1),
           IndexPartition(handle2), IndexSpace(color_space),
           part_kind, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    Color Runtime::create_cross_product_partitions(Context ctx,
                                      IndexPartitionT<DIM,T> handle1,
                                      IndexPartitionT<DIM,T> handle2,
                                      typename std::map<
                                        IndexSpaceT<DIM,T>,
                                        IndexPartitionT<DIM,T> > &handles,
                                      PartitionKind part_kind, Color color,
                                      const char *prov)
    //--------------------------------------------------------------------------
    {
      std::map<IndexSpace,IndexPartition> untyped_handles;
      for (typename std::map<IndexSpaceT<DIM,T>,
                             IndexPartitionT<DIM,T> >::const_iterator it =
            handles.begin(); it != handles.end(); it++)
        untyped_handles[it->first] = IndexPartition::NO_PART;
      Color result = create_cross_product_partitions(ctx, handle1, handle2, 
                                  untyped_handles, part_kind, color, prov);
      for (typename std::map<IndexSpaceT<DIM,T>,
                             IndexPartitionT<DIM,T> >::iterator it =
            handles.begin(); it != handles.end(); it++)
      {
        std::map<IndexSpace,IndexPartition>::const_iterator finder = 
          untyped_handles.find(it->first);
#ifdef DEBUG_LEGION
        assert(finder != untyped_handles.end());
#endif
        it->second = IndexPartitionT<DIM,T>(finder->second);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1, int DIM2, typename T2>
    void Runtime::create_association(Context ctx,
                                     LogicalRegionT<DIM1,T1> domain,
                                     LogicalRegionT<DIM1,T1> domain_parent,
                                     FieldID domain_fid,
                                     IndexSpaceT<DIM2,T2> range,
                                     MapperID id, MappingTagID tag,
                                     UntypedBuffer marg,
                                     const char *provenance)
    //--------------------------------------------------------------------------
    {
      create_association(ctx, LogicalRegion(domain),
          LogicalRegion(domain_parent), domain_fid,
          IndexSpace(range), id, tag, marg, provenance);
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1, int DIM2, typename T2>
    void Runtime::create_bidirectional_association(Context ctx,
                                      LogicalRegionT<DIM1,T1> domain,
                                      LogicalRegionT<DIM1,T1> domain_parent,
                                      FieldID domain_fid,
                                      LogicalRegionT<DIM2,T2> range,
                                      LogicalRegionT<DIM2,T2> range_parent,
                                      FieldID range_fid,
                                      MapperID id, MappingTagID tag,
                                      UntypedBuffer marg,
                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      create_bidirectional_association(ctx, LogicalRegion(domain),
                                       LogicalRegion(domain_parent), domain_fid,
                                       LogicalRegion(range),
                                       LogicalRegion(range_parent), 
                                       range_fid, id, tag, marg, provenance);
    }

    //--------------------------------------------------------------------------
    template<int DIM, int COLOR_DIM, typename T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_restriction(Context ctx,
                                      IndexSpaceT<DIM,T> parent,
                                      IndexSpaceT<COLOR_DIM,T> color_space,
                                      Transform<DIM,COLOR_DIM,T> transform,
                                      Rect<DIM,T> extent,
                                      PartitionKind part_kind, Color color,
                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_restricted_partition(ctx,
        parent, color_space, &transform, sizeof(transform), 
        &extent, sizeof(extent), part_kind, color, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_blockify(Context ctx,
                                      IndexSpaceT<DIM,T> parent,
                                      Point<DIM,T> blocking_factor,
                                      Color color, const char *provenance)
    //--------------------------------------------------------------------------
    {
      Point<DIM,T> origin; 
      for (int i = 0; i < DIM; i++)
        origin[i] = 0;
      return create_partition_by_blockify<DIM,T>(ctx, parent, blocking_factor,
                                                 origin, color, provenance);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_blockify(Context ctx,
                                      IndexSpaceT<DIM,T> parent,
                                      Point<DIM,T> blocking_factor,
                                      Point<DIM,T> origin,
                                      Color color, const char *provenance)
    //--------------------------------------------------------------------------
    {
      // Get the domain of the color space to partition
      const DomainT<DIM,T> parent_is = get_index_space_domain(parent);
      const Rect<DIM,T> &bounds = parent_is.bounds;
      if (bounds.empty())
        return IndexPartitionT<DIM,T>();
      // Compute the intended color space bounds
      Point<DIM,T> colors;
      for (int i = 0; i < DIM; i++)
        colors[i] = (((bounds.hi[i] - bounds.lo[i]) + // -1 and +1 cancel out
            blocking_factor[i]) / blocking_factor[i]) - 1; 
      Point<DIM,T> zeroes; 
      for (int i = 0; i < DIM; i++)
        zeroes[i] = 0;
      // Make the color space
      IndexSpaceT<DIM,T> color_space = create_index_space(ctx, 
                                    Rect<DIM,T>(zeroes, colors));
      // Now make the transform matrix
      Transform<DIM,DIM,T> transform;
      for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
          if (i == j)
            transform[i][j] = blocking_factor[i];
          else
            transform[i][j] = 0;
      // And the extent
      Point<DIM,T> ones;
      for (int i = 0; i < DIM; i++)
        ones[i] = 1;
      const Rect<DIM,T> extent(origin, origin + blocking_factor - ones);
      // Then do the create partition by restriction call
      return create_partition_by_restriction(ctx, parent, color_space,
            transform, extent, LEGION_DISJOINT_KIND, color, provenance);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_domain(
                                    Context ctx, IndexSpaceT<DIM,T> parent,
                                    const std::map<Point<COLOR_DIM,COLOR_T>,
                                                   DomainT<DIM,T> > &domains,
                                    IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                                    bool perform_intersections,
                                    PartitionKind part_kind, Color color,
                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      std::map<DomainPoint,Domain> converted_domains;
      for (typename std::map<Point<COLOR_DIM,COLOR_T>,DomainT<DIM,T> >::
            const_iterator it = domains.begin(); it != domains.end(); it++)
        converted_domains[DomainPoint(it->first)] = Domain(it->second);
      return IndexPartitionT<DIM,T>(create_partition_by_domain(ctx,
              IndexSpace(parent), converted_domains, IndexSpace(color_space),
              perform_intersections, part_kind, color, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_domain(
                                    Context ctx, IndexSpaceT<DIM,T> parent,
                                    const FutureMap &domain_future_map,
                                    IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                                    bool perform_intersections,
                                    PartitionKind part_kind, Color color,
                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_partition_by_domain(ctx,
              IndexSpace(parent), domain_future_map, IndexSpace(color_space),
              perform_intersections, part_kind, color, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_partition_by_field(Context ctx,
                                    LogicalRegionT<DIM,T> handle,
                                    LogicalRegionT<DIM,T> parent,
                                    FieldID fid,
                                    IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                                    Color color, MapperID id, MappingTagID tag,
                                    PartitionKind part_kind, UntypedBuffer marg,
                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_partition_by_field(ctx,
            LogicalRegion(handle), LogicalRegion(parent), fid, 
            IndexSpace(color_space), color, id, tag,part_kind,marg,provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1, int DIM2, typename T2,
             int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM2,T2> Runtime::create_partition_by_image(Context ctx,
                              IndexSpaceT<DIM2,T2> handle,
                              LogicalPartitionT<DIM1,T1> projection,
                              LogicalRegionT<DIM1,T1> parent,
                              FieldID fid, // type: Point<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              PartitionKind part_kind, Color color,
                              MapperID id, MappingTagID tag,
                              UntypedBuffer marg, const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM2,T2>(create_partition_by_image(ctx,
        IndexSpace(handle), LogicalPartition(projection),
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, 
        color, id, tag, marg, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1, int DIM2, typename T2,
             int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM2,T2> Runtime::create_partition_by_image_range(
                              Context ctx,
                              IndexSpaceT<DIM2,T2> handle,
                              LogicalPartitionT<DIM1,T1> projection,
                              LogicalRegionT<DIM1,T1> parent,
                              FieldID fid, // type: Point<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              PartitionKind part_kind, Color color,
                              MapperID id, MappingTagID tag,
                              UntypedBuffer marg, const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM2,T2>(create_partition_by_image_range(ctx,
        IndexSpace(handle), LogicalPartition(projection),
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, 
        color, id, tag, marg, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1, int DIM2, typename T2,
             int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM1,T1> Runtime::create_partition_by_preimage(Context ctx,
                              IndexPartitionT<DIM2,T2> projection,
                              LogicalRegionT<DIM1,T1> handle,
                              LogicalRegionT<DIM1,T1> parent,
                              FieldID fid, // type: Point<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              PartitionKind part_kind, Color color,
                              MapperID id, MappingTagID tag,
                              UntypedBuffer marg, const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM1,T1>(create_partition_by_preimage(ctx, 
        IndexPartition(projection), LogicalRegion(handle),
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, 
        color, id, tag, marg, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM1, typename T1, int DIM2, typename T2,
             int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM1,T1> Runtime::create_partition_by_preimage_range(
                              Context ctx,
                              IndexPartitionT<DIM2,T2> projection,
                              LogicalRegionT<DIM1,T1> handle,
                              LogicalRegionT<DIM1,T1> parent,
                              FieldID fid, // type: Rect<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                              PartitionKind part_kind, Color color,
                              MapperID id, MappingTagID tag,
                              UntypedBuffer marg, const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM1,T1>(create_partition_by_preimage_range(ctx,
        IndexPartition(projection), LogicalRegion(handle), 
        LogicalRegion(parent), fid, IndexSpace(color_space), part_kind, 
        color, id, tag, marg, provenance));
    } 

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexPartitionT<DIM,T> Runtime::create_pending_partition(Context ctx,
                         IndexSpaceT<DIM,T> parent,
                         IndexSpaceT<COLOR_DIM,COLOR_T> color_space,
                         PartitionKind part_kind, Color color, const char *prov)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(create_pending_partition(ctx,
          IndexSpace(parent), IndexSpace(color_space), part_kind, color, prov));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexSpaceT<DIM,T> Runtime::create_index_space_union(Context ctx,
                                IndexPartitionT<DIM,T> parent,
                                Point<COLOR_DIM,COLOR_T> color,
                                const typename std::vector<
                                  IndexSpaceT<DIM,T> > &handles,
                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpace> untyped_handles(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
        untyped_handles[idx] = handles[idx];
      return IndexSpaceT<DIM,T>(create_index_space_union_internal(ctx, 
            IndexPartition(parent), &color, 
            Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>(),
            untyped_handles, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexSpaceT<DIM,T> Runtime::create_index_space_union(Context ctx,
                                IndexPartitionT<DIM,T> parent,
                                Point<COLOR_DIM,COLOR_T> color,
                                IndexPartitionT<DIM,T> handle,
                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<DIM,T>(create_index_space_union_internal(ctx,
          IndexPartition(parent), &color, 
          Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>(),
          IndexPartition(handle), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexSpaceT<DIM,T> Runtime::create_index_space_intersection(Context ctx,
                                IndexPartitionT<DIM,T> parent,
                                Point<COLOR_DIM,COLOR_T> color,
                                const typename std::vector<
                                  IndexSpaceT<DIM,T> > &handles,
                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpace> untyped_handles(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
        untyped_handles[idx] = handles[idx];
      return IndexSpaceT<DIM,T>(create_index_space_intersection_internal(ctx,
            IndexPartition(parent), &color,
            Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>(), 
            untyped_handles, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexSpaceT<DIM,T> Runtime::create_index_space_intersection(Context ctx,
                                IndexPartitionT<DIM,T> parent,
                                Point<COLOR_DIM,COLOR_T> color,
                                IndexPartitionT<DIM,T> handle,
                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<DIM,T>(create_index_space_intersection_internal(ctx,
          IndexPartition(parent), &color, 
          Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>(),
          IndexPartition(handle), provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexSpaceT<DIM,T> Runtime::create_index_space_difference(Context ctx,
                                IndexPartitionT<DIM,T> parent,
                                Point<COLOR_DIM,COLOR_T> color,
                                IndexSpaceT<DIM,T> initial,
                                const typename std::vector<
                                  IndexSpaceT<DIM,T> > &handles,
                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpace> untyped_handles(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
        untyped_handles[idx] = handles[idx];
      return IndexSpaceT<DIM,T>(create_index_space_difference_internal(ctx,
            IndexPartition(parent), &color,
            Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>(), 
            IndexSpace(initial), untyped_handles, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T> Runtime::get_index_partition(
                                         IndexSpaceT<DIM,T> parent, Color color)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(
                          get_index_partition(IndexSpace(parent), color));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool Runtime::has_index_partition(IndexSpaceT<DIM,T> parent, Color color)
    //--------------------------------------------------------------------------
    {
      return has_index_partition(IndexSpace(parent), color);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexSpaceT<DIM,T> Runtime::get_index_subspace(IndexPartitionT<DIM,T> p,
                                         Point<COLOR_DIM,COLOR_T> color)
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<DIM,T>(get_index_subspace_internal(IndexPartition(p), 
        &color, Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>()));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    bool Runtime::has_index_subspace(IndexPartitionT<DIM,T> p, 
                                     Point<COLOR_DIM,COLOR_T> color)
    //--------------------------------------------------------------------------
    {
      return has_index_subspace_internal(IndexPartition(p), &color,
          Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainT<DIM,T> Runtime::get_index_space_domain(IndexSpaceT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM,T> realm_is;
      get_index_space_domain_internal(handle, &realm_is, 
          Internal::NT_TemplateHelper::encode_tag<DIM,T>());
      return realm_is;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    DomainT<COLOR_DIM,COLOR_T> 
              Runtime::get_index_partition_color_space(IndexPartitionT<DIM,T> p)
    //--------------------------------------------------------------------------
    {
      DomainT<COLOR_DIM, COLOR_T> realm_is;
      get_index_partition_color_space_internal(p, &realm_is, 
          Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>());
      return realm_is;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    IndexSpaceT<COLOR_DIM,COLOR_T> 
         Runtime::get_index_partition_color_space_name(IndexPartitionT<DIM,T> p)
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<COLOR_DIM,COLOR_T>(
                              get_index_partition_color_space_name(p));
    }

    //--------------------------------------------------------------------------
    template<unsigned DIM>
    IndexSpace Runtime::get_index_subspace(Context ctx, 
                IndexPartition p, LegionRuntime::Arrays::Point<DIM> color_point)
    //--------------------------------------------------------------------------
    {
      DomainPoint dom_point = DomainPoint::from_point<DIM>(color_point);
      return get_index_subspace(ctx, p, dom_point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    Point<COLOR_DIM,COLOR_T> Runtime::get_index_space_color(
                                                      IndexSpaceT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      Point<COLOR_DIM,COLOR_T> point;
      return get_index_space_color_internal(IndexSpace(handle), &point,
          Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceT<DIM,T> Runtime::get_parent_index_space(
                                                  IndexPartitionT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      return IndexSpaceT<DIM,T>(get_parent_index_space(IndexPartition(handle)));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartitionT<DIM,T> Runtime::get_parent_index_partition(
                                                      IndexSpaceT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      return IndexPartitionT<DIM,T>(get_parent_index_partition(
                                              IndexSpace(handle)));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool Runtime::safe_cast(Context ctx, Point<DIM,T> point, 
                            LogicalRegionT<DIM,T> region)
    //--------------------------------------------------------------------------
    {
      return safe_cast_internal(ctx, LogicalRegion(region), &point,
          Internal::NT_TemplateHelper::encode_tag<DIM,T>());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T> Runtime::create_logical_region(Context ctx,
                                   IndexSpaceT<DIM,T> index, FieldSpace fields,
                                   bool task_local, const char *provenance)
    //--------------------------------------------------------------------------
    {
      return LogicalRegionT<DIM,T>(create_logical_region(ctx, 
                            IndexSpace(index), fields, task_local, provenance));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T> Runtime::get_logical_partition(
                    LogicalRegionT<DIM,T> parent, IndexPartitionT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      return LogicalPartitionT<DIM,T>(get_logical_partition(
                LogicalRegion(parent), IndexPartition(handle)));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T> Runtime::get_logical_partition_by_color(
                                      LogicalRegionT<DIM,T> parent, Color color)
    //--------------------------------------------------------------------------
    {
      return LogicalPartitionT<DIM,T>(get_logical_partition_by_color(
                                        LogicalRegion(parent), color));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T> Runtime::get_logical_partition_by_tree(
              IndexPartitionT<DIM,T> handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return LogicalPartitionT<DIM,T>(get_logical_partition_by_tree(
                                  IndexPartition(handle), space, tid));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T> Runtime::get_logical_subregion(
                     LogicalPartitionT<DIM,T> parent, IndexSpaceT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      return LogicalRegionT<DIM,T>(get_logical_subregion(
                LogicalPartition(parent), IndexSpace(handle)));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    LogicalRegionT<DIM,T> Runtime::get_logical_subregion_by_color(
        LogicalPartitionT<DIM,T> parent, Point<COLOR_DIM,COLOR_T> color)
    //--------------------------------------------------------------------------
    {
      return LogicalRegionT<DIM,T>(get_logical_subregion_by_color_internal(
            LogicalPartition(parent), &color,
            Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>()));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    bool Runtime::has_logical_subregion_by_color(
        LogicalPartitionT<DIM,T> parent, Point<COLOR_DIM,COLOR_T> color)
    //--------------------------------------------------------------------------
    {
      return has_logical_subregion_by_color_internal(
          LogicalPartition(parent), &color,
          Internal::NT_TemplateHelper::encode_tag<COLOR_DIM,COLOR_T>());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T> Runtime::get_logical_subregion_by_tree(
                  IndexSpaceT<DIM,T> handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return LogicalRegionT<DIM,T>(get_logical_subregion_by_tree(
                                    IndexSpace(handle), space, tid));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T, int COLOR_DIM, typename COLOR_T>
    Point<COLOR_DIM,COLOR_T> Runtime::get_logical_region_color_point(
                                                   LogicalRegionT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      return get_logical_region_color_point(LogicalRegion(handle));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalRegionT<DIM,T> Runtime::get_parent_logical_region(
                                                LogicalPartitionT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      return LogicalRegionT<DIM,T>(get_parent_logical_region(
                                    LogicalPartition(handle)));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LogicalPartitionT<DIM,T> Runtime::get_parent_logical_partition(
                                                   LogicalRegionT<DIM,T> handle)
    //--------------------------------------------------------------------------
    {
      return LogicalPartitionT<DIM,T>(get_parent_logical_partition(
                                            LogicalRegion(handle)));
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void Runtime::fill_field(Context ctx, LogicalRegion handle,
                                      LogicalRegion parent, FieldID fid,
                                      const T &value, Predicate pred)
    //--------------------------------------------------------------------------
    {
      fill_field(ctx, handle, parent, fid, &value, sizeof(T), pred);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void Runtime::fill_fields(Context ctx, LogicalRegion handle,
                                       LogicalRegion parent, 
                                       const std::set<FieldID> &fields,
                                       const T &value, Predicate pred)
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
    void Runtime::set_local_task_variable(Context ctx, LocalVariableID id,
                                      const T* value, void (*destructor)(void*))
    //--------------------------------------------------------------------------
    {
      set_local_task_variable_untyped(ctx, id, value, destructor);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    /*static*/ void Runtime::register_reduction_op(ReductionOpID redop_id,
                                                   bool permit_duplicates)
    //--------------------------------------------------------------------------
    {
      // We also have to check to see if there are explicit serialization
      // and deserialization methods on the RHS type for doing fold reductions
      LegionSerialization::register_reduction<REDOP>(redop_id, 
                                                     permit_duplicates);
    }

#ifdef LEGION_GPU_REDUCTIONS
    //--------------------------------------------------------------------------
    template<typename REDOP>
    /*static*/ void Runtime::preregister_gpu_reduction_op(ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      Runtime::register_reduction_op<REDOP>(redop, false/*permit duplicates*/);
    }
#endif // LEGION_GPU_REDUCTIONS

    //--------------------------------------------------------------------------
    template<typename SERDEZ>
    /*static*/ void Runtime::register_custom_serdez_op(CustomSerdezID serdez_id,
                                                       bool permit_duplicates)
    //--------------------------------------------------------------------------
    {
      Runtime::register_custom_serdez_op(serdez_id,
        Realm::CustomSerdezUntyped::create_custom_serdez<SERDEZ>(), 
        permit_duplicates);
    }

    namespace Internal {
      // Wrapper class for old projection functions
      template<RegionProjectionFnptr FNPTR>
      class RegionProjectionWrapper : public ProjectionFunctor {
      public:
        RegionProjectionWrapper(void) 
          : ProjectionFunctor() { }
        virtual ~RegionProjectionWrapper(void) { }
      public:
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalRegion upper_bound,
                                      const DomainPoint &point)
        {
          return (*FNPTR)(upper_bound, point, runtime); 
        }
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalPartition upper_bound,
                                      const DomainPoint &point)
        {
          assert(false);
          return LogicalRegion::NO_REGION;
        }
        virtual bool is_exclusive(void) const { return false; }
      };
    };

    //--------------------------------------------------------------------------
    template<LogicalRegion (*PROJ_PTR)(LogicalRegion, const DomainPoint&,
                                       Runtime*)>
    /*static*/ ProjectionID Runtime::register_region_function(
                                                            ProjectionID handle)
    //--------------------------------------------------------------------------
    {
      Runtime::preregister_projection_functor(handle,
          new Internal::RegionProjectionWrapper<PROJ_PTR>());
      return handle;
    }

    namespace Internal {
      // Wrapper class for old projection functions
      template<PartitionProjectionFnptr FNPTR>
      class PartitionProjectionWrapper : public ProjectionFunctor {
      public:
        PartitionProjectionWrapper(void)
          : ProjectionFunctor() { }
        virtual ~PartitionProjectionWrapper(void) { }
      public:
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalRegion upper_bound,
                                      const DomainPoint &point)
        {
          assert(false);
          return LogicalRegion::NO_REGION;
        }
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalPartition upper_bound,
                                      const DomainPoint &point)
        {
          return (*FNPTR)(upper_bound, point, runtime);
        }
        virtual bool is_exclusive(void) const { return false; }
      };
    };

    //--------------------------------------------------------------------------
    template<LogicalRegion (*PROJ_PTR)(LogicalPartition, const DomainPoint&,
                                       Runtime*)>
    /*static*/ ProjectionID Runtime::register_partition_function(
                                                    ProjectionID handle)
    //--------------------------------------------------------------------------
    {
      Runtime::preregister_projection_functor(handle,
          new Internal::PartitionProjectionWrapper<PROJ_PTR>());
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
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*)>
      static void legion_task_wrapper(const void*, size_t, 
                                      const void*, size_t, Processor);
      template<typename T, typename UDT,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*, const UDT&)>
      static void legion_udt_task_wrapper(const void*, size_t, 
                                          const void*, size_t, Processor);
    public:
      // Void return type for new legion task types
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*)>
      static void legion_task_wrapper(const void*, size_t, 
                                      const void*, size_t, Processor);
      template<typename UDT,
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*, const UDT&)>
      static void legion_udt_task_wrapper(const void*, size_t, 
                                          const void*, size_t, Processor);

    public:
      // Do-it-yourself pre/post-ambles for code generators
      // These are deprecated and are just here for backwards compatibility
      static void legion_task_preamble(const void *data,
				       size_t datalen,
				       Processor p,
				       const Task *& task,
				       const std::vector<PhysicalRegion> *& ptr,
				       Context& ctx,
				       Runtime *& runtime);
      static void legion_task_postamble(Runtime *runtime, Context ctx,
					const void *retvalptr = NULL,
					size_t retvalsize = 0);
    };
    
    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    void LegionTaskWrapper::legion_task_wrapper(const void *args, 
                                                size_t arglen, 
                                                const void *userdata,
                                                size_t userlen,
                                                Processor p)
    //--------------------------------------------------------------------------
    {
      // Assert that we are returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value),
          "Future types are not permitted as return types for Legion tasks");
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value),
          "FutureMap types are not permitted as return types for Legion tasks");
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(T) <= LEGION_MAX_RETURN_SIZE,
          "Task return values must be less than or equal to "
          "LEGION_MAX_RETURN_SIZE bytes");
      const Task *task; Context ctx; Runtime *rt;
      const std::vector<PhysicalRegion> *regions;
      Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

      // Invoke the task with the given context
      T return_value = (*TASK_PTR)(task, *regions, ctx, rt);

      // Send the return value back
      LegionSerialization::end_task<T>(rt, ctx, &return_value);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*)>
    void LegionTaskWrapper::legion_task_wrapper(const void *args, 
                                                size_t arglen, 
                                                const void *userdata,
                                                size_t userlen,
                                                Processor p)
    //--------------------------------------------------------------------------
    {
      const Task *task; Context ctx; Runtime *rt;
      const std::vector<PhysicalRegion> *regions;
      Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

      (*TASK_PTR)(task, *regions, ctx, rt);

      Runtime::legion_task_postamble(rt, ctx);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    void LegionTaskWrapper::legion_udt_task_wrapper(const void *args,
                                                    size_t arglen, 
                                                    const void *userdata,
                                                    size_t userlen,
                                                    Processor p)
    //--------------------------------------------------------------------------
    {
      // Assert that we are returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value),
          "Future types are not permitted as return types for Legion tasks");
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value),
          "FutureMap types are not permitted as return types for Legion tasks");
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT((sizeof(T) <= LEGION_MAX_RETURN_SIZE) ||
         (std::is_class<T>::value && 
          LegionSerialization::IsSerdezType<T>::value),
         "Task return values must be less than or equal to "
          "LEGION_MAX_RETURN_SIZE bytes");

      const Task *task; Context ctx; Runtime *rt;
      const std::vector<PhysicalRegion> *regions;
      Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

      const UDT *user_data = NULL;
      static_assert(sizeof(user_data) == sizeof(userdata), "C++ is dumb");
      memcpy(&user_data, &userdata, sizeof(user_data));

      // Invoke the task with the given context
      T return_value = (*TASK_PTR)(task, *regions, ctx, rt, *user_data); 

      // Send the return value back
      LegionSerialization::end_task<T>(rt, ctx, &return_value);
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    void LegionTaskWrapper::legion_udt_task_wrapper(const void *args,
                                                    size_t arglen, 
                                                    const void *userdata,
                                                    size_t userlen,
                                                    Processor p)
    //--------------------------------------------------------------------------
    {
      const Task *task; Context ctx; Runtime *rt;
      const std::vector<PhysicalRegion> *regions;
      Runtime::legion_task_preamble(args, arglen, p, task, regions, ctx, rt);

      const UDT *user_data = NULL;
      static_assert(sizeof(user_data) == sizeof(userdata), "C++ is dumb");
      memcpy(&user_data, &userdata, sizeof(user_data));

      (*TASK_PTR)(task, *regions, ctx, rt, *user_data); 

      // Send an empty return value back
      Runtime::legion_task_postamble(rt, ctx);
    }

    //--------------------------------------------------------------------------
    inline void LegionTaskWrapper::legion_task_preamble(
                  const void *data,
		  size_t datalen,
		  Processor p,
		  const Task *& task,
		  const std::vector<PhysicalRegion> *& regionsptr,
		  Context& ctx,
		  Runtime *& runtime)
    //--------------------------------------------------------------------------
    {
      Runtime::legion_task_preamble(data, datalen, p, task, 
                                    regionsptr, ctx, runtime);
    }

    //--------------------------------------------------------------------------
    inline void LegionTaskWrapper::legion_task_postamble(
                  Runtime *runtime, Context ctx,
		  const void *retvalptr /*= NULL*/,
		  size_t retvalsize /*= 0*/)
    //--------------------------------------------------------------------------
    {
      Runtime::legion_task_postamble(runtime, ctx, retvalptr, retvalsize);
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    VariantID Runtime::register_task_variant(
                           const TaskVariantRegistrar &registrar, VariantID vid)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>);
      return register_task_variant(registrar, desc,NULL/*UDT*/,
          0/*sizeof(UDT)*/, LegionSerialization::ReturnSize<T>::value, vid);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    VariantID Runtime::register_task_variant(
     const TaskVariantRegistrar &registrar, const UDT &user_data, VariantID vid)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(
          LegionTaskWrapper::legion_udt_task_wrapper<T,UDT,TASK_PTR>);
      return register_task_variant(registrar, desc, &user_data, sizeof(UDT),
                            LegionSerialization::ReturnSize<T>::value, vid);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*)>
    VariantID Runtime::register_task_variant(
                           const TaskVariantRegistrar &registrar, VariantID vid)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
      return register_task_variant(registrar, desc, NULL/*UDT*/, 
                                   0/*sizeof(UDT)*/, 0/*return size*/, vid);
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    VariantID Runtime::register_task_variant(
     const TaskVariantRegistrar &registrar, const UDT &user_data, VariantID vid)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(
          LegionTaskWrapper::legion_udt_task_wrapper<UDT,TASK_PTR>);
      return register_task_variant(registrar, desc, &user_data, sizeof(UDT),
                                   0/*return size*/, vid);
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    /*static*/ VariantID Runtime::preregister_task_variant(
        const TaskVariantRegistrar &registrar, 
        const char *task_name /*= NULL*/, VariantID vid /*=AUTO_GENERATE_ID*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>);
      return preregister_task_variant(registrar, desc, NULL/*UDT*/, 
                                  0/*sizeof(UDT)*/, task_name, vid, 
                                  LegionSerialization::ReturnSize<T>::value);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    /*static*/ VariantID Runtime::preregister_task_variant(
                    const TaskVariantRegistrar &registrar, 
                    const UDT &user_data, const char *task_name /*= NULL*/,
                    VariantID vid /*=AUTO_GENERATE_ID*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(
          LegionTaskWrapper::legion_udt_task_wrapper<T,UDT,TASK_PTR>);
      return preregister_task_variant(registrar, desc, &user_data, sizeof(UDT),
                    task_name, vid, LegionSerialization::ReturnSize<T>::value);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*)>
    /*static*/ VariantID Runtime::preregister_task_variant(
        const TaskVariantRegistrar &registrar, const char *task_name /*= NULL*/,
        const VariantID vid /*=AUTO_GENERATE_ID*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
      return preregister_task_variant(registrar, desc, NULL/*UDT*/,
                0/*sizeof(UDT)*/, task_name, vid, 0/*return size*/);
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    /*static*/ VariantID Runtime::preregister_task_variant(
                    const TaskVariantRegistrar &registrar, 
                    const UDT &user_data, const char *task_name /*= NULL*/,
                    VariantID vid /*=AUTO_GENERATE_ID*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor desc(
          LegionTaskWrapper::legion_udt_task_wrapper<UDT,TASK_PTR>);
      return preregister_task_variant(registrar, desc, &user_data, sizeof(UDT),
                                      task_name, vid, 0/*return size*/);
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
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
      CodeDescriptor desc(LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>);
      preregister_task_variant(registrar, desc, NULL/*UDT*/, 0/*sizeof(UDT)*/,
      task_name, vid, LegionSerialization::ReturnSize<T>::value, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
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
      preregister_task_variant(registrar, desc, NULL/*UDT*/, 0/*sizeof(UDT)*/,
                               task_name, vid, 0/*return size*/, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    const UDT &user_data,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
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
          LegionTaskWrapper::legion_udt_task_wrapper<T,UDT,TASK_PTR>);
      preregister_task_variant(registrar, desc, &user_data, sizeof(UDT),
      task_name, vid, LegionSerialization::ReturnSize<T>::value, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    const UDT &user_data,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
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
          LegionTaskWrapper::legion_udt_task_wrapper<UDT,TASK_PTR>);
      preregister_task_variant(registrar, desc, &user_data, sizeof(UDT),
                               task_name, vid, 0/*return size*/, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator~(PrivilegeMode p)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(~unsigned(p));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator|(PrivilegeMode left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(unsigned(left) | unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator&(PrivilegeMode left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(unsigned(left) & unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator^(PrivilegeMode left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(unsigned(left) ^ unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator|=(PrivilegeMode &left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l |= r;
      return left = static_cast<PrivilegeMode>(l);
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator&=(PrivilegeMode &left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l &= r;
      return left = static_cast<PrivilegeMode>(l);
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator^=(PrivilegeMode &left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l ^= r;
      return left = static_cast<PrivilegeMode>(l);
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator~(AllocateMode a)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(~unsigned(a));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator|(AllocateMode left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(unsigned(left) | unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator&(AllocateMode left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(unsigned(left) & unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator^(AllocateMode left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(unsigned(left) ^ unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator|=(AllocateMode &left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l |= r;
      return left = static_cast<AllocateMode>(l);
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator&=(AllocateMode &left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l &= r;
      return left = static_cast<AllocateMode>(l);
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator^=(AllocateMode &left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l ^= r;
      return left = static_cast<AllocateMode>(l);
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const LogicalRegion& lr)
    //--------------------------------------------------------------------------
    {
      os << "LogicalRegion(" << lr.tree_id << "," 
         << lr.index_space << "," << lr.field_space << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os,const LogicalPartition& lp)
    //--------------------------------------------------------------------------
    {
      os << "LogicalPartition(" << lp.tree_id << "," 
         << lp.index_partition << "," << lp.field_space << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const IndexSpace& is)
    //--------------------------------------------------------------------------
    {
      os << "IndexSpace(" << is.id << "," << is.tid << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const IndexPartition& ip)
    //--------------------------------------------------------------------------
    {
      os << "IndexPartition(" << ip.id << "," << ip.tid << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const FieldSpace& fs)
    //--------------------------------------------------------------------------
    {
      os << "FieldSpace(" << fs.id << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const PhaseBarrier& pb)
    //--------------------------------------------------------------------------
    {
      os << "PhaseBarrier(" << pb.phase_barrier << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline size_t Unserializable<T>::legion_buffer_size(void)
    //--------------------------------------------------------------------------
    {
      const std::type_info &info = typeid(T);
      fprintf(stderr,"ERROR: Illegal attempt to serialize Legion type %s. "
          "Objects of type %s are not allowed to be passed by value into or "
          "out of tasks.\n", info.name(), info.name());
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline size_t Unserializable<T>::legion_serialize(void *buffer)
    //--------------------------------------------------------------------------
    {
      const std::type_info &info = typeid(T);
      fprintf(stderr,"ERROR: Illegal attempt to serialize Legion type %s. "
          "Objects of type %s are not allowed to be passed by value into or "
          "out of tasks.\n", info.name(), info.name());
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline size_t Unserializable<T>::legion_deserialize(const void *buffer)
    //--------------------------------------------------------------------------
    {
      const std::type_info &info = typeid(T);
      fprintf(stderr,"ERROR: Illegal attempt to deserialize Legion type %s. "
          "Objects of type %s are not allowed to be passed by value into or "
          "out of tasks.\n", info.name(), info.name());
      assert(false);
      return 0;
    }

}; // namespace Legion

// This is for backwards compatibility with the old namespace scheme
namespace LegionRuntime {
  namespace HighLevel {
    using namespace LegionRuntime::Arrays;

    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::IndexSpace IndexSpace;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::IndexPartition IndexPartition;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::FieldSpace FieldSpace;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::LogicalRegion LogicalRegion;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::LogicalPartition LogicalPartition;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::FieldAllocator FieldAllocator;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::UntypedBuffer TaskArgument;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ArgumentMap ArgumentMap;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Predicate Predicate;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Lock Lock;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::LockRequest LockRequest;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Grant Grant;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::PhaseBarrier PhaseBarrier;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::DynamicCollective DynamicCollective;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::RegionRequirement RegionRequirement;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::IndexSpaceRequirement IndexSpaceRequirement;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::FieldSpaceRequirement FieldSpaceRequirement;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Future Future;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::FutureMap FutureMap;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::TaskLauncher TaskLauncher;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::IndexLauncher IndexLauncher;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::InlineLauncher InlineLauncher;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::CopyLauncher CopyLauncher;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::PhysicalRegion PhysicalRegion;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::IndexIterator IndexIterator;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::IndexAllocator IndexAllocator;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::AcquireLauncher AcquireLauncher;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ReleaseLauncher ReleaseLauncher;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::TaskVariantRegistrar TaskVariantRegistrar;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::MustEpochLauncher MustEpochLauncher;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::MPILegionHandshake MPILegionHandshake;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Mappable Mappable;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Task Task;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Copy Copy;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::InlineMapping Inline;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Acquire Acquire;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Release Release;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Mapping::Mapper Mapper;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::InputArgs InputArgs;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::TaskConfigOptions TaskConfigOptions;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ProjectionFunctor ProjectionFunctor;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Runtime Runtime;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Runtime HighLevelRuntime; // for backwards compatibility
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ColoringSerializer ColoringSerializer;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::DomainColoringSerializer DomainColoringSerializer;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Serializer Serializer;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Deserializer Deserializer;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::TaskResult TaskResult;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::CObjectWrapper CObjectWrapper;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ISAConstraint ISAConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ProcessorConstraint ProcessorConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ResourceConstraint ResourceConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::LaunchConstraint LaunchConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ColocationConstraint ColocationConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::ExecutionConstraintSet ExecutionConstraintSet;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::SpecializedConstraint SpecializedConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::MemoryConstraint MemoryConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::FieldConstraint FieldConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::OrderingConstraint OrderingConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::SplittingConstraint SplittingConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::DimensionConstraint DimensionConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::AlignmentConstraint AlignmentConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::OffsetConstraint OffsetConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::PointerConstraint PointerConstraint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::LayoutConstraintSet LayoutConstraintSet;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::TaskLayoutConstraintSet TaskLayoutConstraintSet;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Runtime RealmRuntime;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Machine Machine;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Domain Domain;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::DomainPoint DomainPoint;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::RegionInstance PhysicalInstance;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Memory Memory;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Processor Processor;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::CodeDescriptor CodeDescriptor;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Event Event;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Event MapperEvent;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::UserEvent UserEvent;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Reservation Reservation;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Barrier Barrier;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_reduction_op_id_t ReductionOpID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::ReductionOpUntyped ReductionOp;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_custom_serdez_id_t CustomSerdezID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::CustomSerdezUntyped SerdezOp;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Machine::ProcessorMemoryAffinity ProcessorMemoryAffinity;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Realm::Machine::MemoryMemoryAffinity MemoryMemoryAffinity;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::CustomSerdezID, 
                     const Realm::CustomSerdezUntyped *> SerdezOpTable;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Realm::ReductionOpID, 
            const Realm::ReductionOpUntyped *> ReductionOpTable;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef void (*SerdezInitFnptr)(const Legion::ReductionOp*, 
                                    void *&, size_t&);
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef void (*SerdezFoldFnptr)(const Legion::ReductionOp*, void *&, 
                                    size_t&, const void*, bool);
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Realm::ReductionOpID, 
                     Legion::SerdezRedopFns> SerdezRedopTable;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_address_space_t AddressSpace;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_task_priority_t TaskPriority;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_color_t Color;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_field_id_t FieldID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_trace_id_t TraceID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_mapper_id_t MapperID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_context_id_t ContextID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_instance_id_t InstanceID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_index_space_id_t IndexSpaceID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_index_partition_id_t IndexPartitionID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_index_tree_id_t IndexTreeID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_field_space_id_t FieldSpaceID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_generation_id_t GenerationID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_type_handle TypeHandle;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_projection_id_t ProjectionID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_region_tree_id_t RegionTreeID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_distributed_id_t DistributedID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_address_space_t AddressSpaceID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_tunable_id_t TunableID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_mapping_tag_id_t MappingTagID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_semantic_tag_t SemanticTag;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_variant_id_t VariantID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_unique_id_t UniqueID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_version_id_t VersionID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_task_id_t TaskID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef ::legion_layout_constraint_id_t LayoutConstraintID;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::Color,Legion::ColoredPoints<ptr_t> > Coloring;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::Color,Legion::Domain> DomainColoring;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::Color,
                     std::set<Legion::Domain> > MultiDomainColoring;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::DomainPoint,
                     Legion::ColoredPoints<ptr_t> > PointColoring;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::DomainPoint,Legion::Domain> DomainPointColoring;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::DomainPoint,
                     std::set<Legion::Domain> > MultiDomainPointColoring;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef void (*RegistrationCallbackFnptr)(Realm::Machine machine, 
        Legion::Runtime *rt, const std::set<Legion::Processor> &local_procs);
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::LogicalRegion (*RegionProjectionFnptr)(
        Legion::LogicalRegion parent,
        const Legion::DomainPoint&, Legion::Runtime *rt);
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::LogicalRegion (*PartitionProjectionFnptr)(
        Legion::LogicalPartition parent, 
        const Legion::DomainPoint&, Legion::Runtime *rt);
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef bool (*PredicateFnptr)(const void*, size_t, 
        const std::vector<Legion::Future> futures);
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::ProjectionID,Legion::RegionProjectionFnptr> 
      RegionProjectionTable;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef std::map<Legion::ProjectionID,Legion::PartitionProjectionFnptr> 
      PartitionProjectionTable;
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef void (*RealmFnptr)(const void*,size_t,
                               const void*,size_t,Legion::Processor);
    LEGION_DEPRECATED("Use the Legion namespace instance instead.")
    typedef Legion::Internal::TaskContext* Context; 
  };

  // map old Logger::Category to new Realm::Logger
  namespace Logger {
    typedef Realm::Logger Category;
  };
};

