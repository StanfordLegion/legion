/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// NEVER INCLUDE THIS FILE DIRECTLY, INCLUDE legion_stl.h INSTEAD!

#include "legion/arrays.h"
#include "legion/accessor.h"
#include "legion/legion_utilities.h"

namespace Legion {
  namespace STL {

    template <typename ... Ts>
    struct phantom {
    };

    inline size_t get_serialized_size_(phantom<>)
    {
      return 0;
    }

    template<typename T1, typename ... Ts>
    inline size_t get_serialized_size_(phantom<T1, Ts ...>)
    {
      return sizeof(T1) + get_serialized_size_(phantom<Ts ...>());
    }

    template<typename ... Ts>
    inline size_t get_serialized_size()
    {
      return get_serialized_size_(phantom<Ts ...>());
    }

    inline void serialize_(void *buffer)
    {
    }

    template<typename T1, typename ... Ts>
    inline void serialize_(void *buffer, const T1 &t1, const Ts & ... ts)
    {
      *((T1 *)buffer) = t1;
      buffer = (void *)((char *)buffer + sizeof(T1));
      serialize_(buffer, ts ...);
    }

    template<typename ... Ts>
    inline void serialize(void *buffer, const Ts & ... ts)
    {
      serialize_(buffer, ts ...);
    }

    template<typename ... Us>
    inline std::tuple<Us ...>
    deserialize_(const void *buffer, phantom<>, std::tuple<Us ...> &&us)
    {
      return us;
    }

    template<typename T1, typename ... Ts, typename ... Us>
    inline std::tuple<Us ..., T1, Ts ...>
    deserialize_(const void *buffer, phantom<T1, Ts ...>,
                std::tuple<Us ...> &&us)
    {
      T1 t1 = *((T1 *)buffer);
      buffer = (void *)((char *)buffer + sizeof(T1));
      return deserialize_(buffer, phantom<Ts ...>(),
                          std::tuple_cat(us, std::make_tuple(t1)));
    }

    template<typename ... Ts>
    inline std::tuple<Ts ...> deserialize(const void *buffer)
    {
      return deserialize_(buffer, phantom<Ts ...>(), std::make_tuple());
    }

    template<typename ... Ts>
    inline TypedArgument<Ts ...>::TypedArgument(const Ts & ... ts)
    {
      buf_size = get_serialized_size<Ts ...>();
      buffer = malloc(buf_size);
      assert(buffer);
      serialize((char *)buffer, ts ...);
    }

    template<typename ... Ts>
    inline TypedArgument<Ts ...>::~TypedArgument()
    {
      free(buffer);
    }

    template<typename ... Ts>
    inline TypedArgument<Ts ...>::operator TaskArgument() const
    {
      return TaskArgument(buffer, buf_size);
    }

    template<typename ... Ts>
    inline size_t TypedArgument<Ts ...>::get_size() const
    {
      return buf_size;
    }

    template<typename ... Ts>
    inline void * TypedArgument<Ts ...>::get_ptr() const
    {
      return buffer;
    }

    template<typename T>
    inline size_t set<T>::legion_buffer_size(void) const
    {
      return (sizeof(size_t) + (sizeof(T) * this->size()));
    }

    template<typename T>
    inline void set<T>::legion_serialize(void *buffer) const
    {
      Serializer rez;
      rez.serialize<size_t>(this->size());
      for (typename std::set<T>::const_iterator it = 
            this->begin(); it != this->end(); it++)
        rez.serialize(*it);
#ifdef DEBUG_LEGION
      assert(rez.get_used_bytes() == legion_buffer_size());
#endif
      memcpy(buffer, rez.get_buffer(), rez.get_used_bytes());
    }

    template<typename T>
    inline void set<T>::legion_deserialize(const void *buffer)
    {
      const char *ptr = (const char*)buffer;
      size_t elements = *((const size_t*)ptr);
      ptr += sizeof(elements);
      Deserializer derez(ptr, elements * sizeof(T));
      for (unsigned idx = 0; idx < elements; idx++)
      {
        T elem;
        deserialize(elem);
        this->insert(elem);
      }
    }

    template<typename T1, typename T2>
    inline size_t map<T1,T2>::legion_buffer_size(void) const
    {
      return (sizeof(size_t) + (sizeof(T1) + sizeof(T2)) * this->size());
    }

    template<typename T1, typename T2>
    inline void map<T1,T2>::legion_serialize(void *buffer) const
    {
      Serializer rez;
      rez.serialize<size_t>(this->size());
      for (typename std::map<T1,T2>::const_iterator it = 
            this->begin(); it != this->end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
#ifdef DEBUG_LEGION
      assert(rez.get_used_bytes() == legion_buffer_size());
#endif
      memcpy(buffer, rez.get_buffer(), rez.get_used_bytes());
    }

    template<typename T1, typename T2>
    inline void map<T1,T2>::legion_deserialize(const void *buffer)
    {
      const char *ptr = (const char*)buffer;
      size_t elements = *((const size_t*)ptr);
      ptr += sizeof(elements);
      Deserializer derez(ptr, elements * (sizeof(T1) + sizeof(T2)));
      for (unsigned idx = 0; idx < elements; idx++)
      {
        T1 key;
        derez.deserialize(key);
        derez.deserialize((*this)[key]);
      }
    }

    template<typename T>
    inline size_t vector<T>::legion_buffer_size(void) const
    {
      return (sizeof(size_t) + (sizeof(T) * this->size()));
    }

    template<typename T>
    inline void vector<T>::legion_serialize(void *buffer) const
    {
      Serializer rez;
      rez.serialize<size_t>(this->size());
      for (typename std::vector<T>::const_iterator it = 
            this->begin(); it != this->end(); it++)
        rez.serialize(*it);
#ifdef DEBUG_LEGION
      assert(rez.get_used_bytes() == legion_buffer_size());
#endif
      memcpy(buffer, rez.get_buffer(), rez.get_used_bytes());
    }

    template<typename T>
    inline void vector<T>::legion_deserialize(const void *buffer)
    {
      const char *ptr = (const char*)buffer;
      size_t elements = *((const size_t*)ptr);
      ptr += sizeof(elements);
      Deserializer derez(ptr, elements * sizeof(T));
      this->resize(elements);
      for (unsigned idx = 0; idx < elements; idx++)
        derez.deserialize((*this)[idx]);
    }

#define GET_RAW_POINTERS(x)                                                   \
  std::vector<T##x *> ptrs_##x(task->regions[x].privilege_fields.size(),NULL);\
  ByteOffset offsets_##x[DIM##x];                                             \
  Detail::get_raw_pointers<T##x, DIM##x>(task->regions[x], regions[x],        \
                                         ptrs_##x, offsets_##x, runtime, ctx);

#define GET_DENSE_POINTERS(x)                                                 \
  std::vector<T##x *> ptrs_##x(task->regions[x].privilege_fields.size(),NULL);\
  ByteOffset offset_##x;                                                      \
  Detail::get_dense_pointers<T##x, DIM##x>(task->regions[x], regions[x],      \
                                           ptrs_##x, offset_##x, runtime, ctx);

    namespace Detail {
      template<int DIM>
      static inline void compare_offsets(const ByteOffset a[DIM], 
                                         const ByteOffset b[DIM])
      {
        for (unsigned idx = 0; idx < DIM; idx++)
          assert(a[idx] == b[idx]);
      }

      template<typename T, int DIM>
      static inline void get_raw_pointers(const RegionRequirement &req,
          const PhysicalRegion &region, std::vector<T*> &ptrs, 
          ByteOffset offsets[DIM], Runtime *runtime, Context ctx)
      {
        if ((req.privilege == NO_ACCESS) || !region.is_mapped())
          return;
        unsigned idx = 0;
        LegionRuntime::Arrays::Rect<DIM> region_bounds = 
          runtime->get_index_space_domain(req.region.get_index_space()).get_rect<DIM>();
        LegionRuntime::Arrays::Rect<DIM> actual_bounds;
        for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
              it != req.privilege_fields.end(); it++, idx++)
        {
          LegionRuntime::Accessor::RegionAccessor<
            LegionRuntime::Accessor::AccessorType::Generic,T> facc = 
            region.get_field_accessor(*it).typeify<T>();
          if (idx != 0)
          {
            ByteOffset temp_offsets[DIM];
            ptrs[idx] = facc.template raw_rect_ptr<DIM>(region_bounds, 
                                          actual_bounds, temp_offsets);
            assert(region_bounds == actual_bounds);
            compare_offsets<DIM>(offsets, temp_offsets);       
          }
          else
          {
            ptrs[idx] = facc.template raw_rect_ptr<DIM>(region_bounds, 
                                              actual_bounds, offsets);
            assert(region_bounds == actual_bounds);
          }
        }
      }

      template<typename T, int DIM>
      static inline void get_dense_pointers(const RegionRequirement &req,
          const PhysicalRegion &region, std::vector<T*> &ptrs, 
          ByteOffset &offset, Runtime *runtime, Context ctx)
      {
        if ((req.privilege == NO_ACCESS) || !region.is_mapped())
          return;
        unsigned idx = 0;
        LegionRuntime::Arrays::Rect<DIM> region_bounds = 
          runtime->get_index_space_domain(req.region.get_index_space()).get_rect<DIM>();
        LegionRuntime::Arrays::Rect<DIM> actual_bounds;
        for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
              it != req.privilege_fields.end(); it++, idx++)
        {
          LegionRuntime::Accessor::RegionAccessor<
            LegionRuntime::Accessor::AccessorType::Generic,T> facc = 
            region.get_field_accessor(*it).typeify<T>();
          if (idx != 0)
          {
            ByteOffset temp_offset;
            ptrs[idx] = facc.template raw_dense_ptr<DIM>(region_bounds, 
                                          actual_bounds, temp_offset);
            assert(region_bounds == actual_bounds);
            assert(temp_offset == offset);
          }
          else
          {
            ptrs[idx] = facc.template raw_dense_ptr<DIM>(region_bounds, 
                                                actual_bounds, offset);
            assert(region_bounds == actual_bounds);
          }
        }
      }
    };

    // 1-Region

    template<typename T0, int DIM0,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0);
    }

    template<typename T, typename T0, int DIM0, 
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0);
    }

    template<typename T0, int DIM0,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0);
    }

    template<typename T, typename T0, int DIM0, 
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0);
    }

    // 2-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1);
    }

    // 3-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2);
    }

    // 4-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3);
    }

    // 5-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4);
    }

    // 6-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5);
    }

    // 7-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6);
    }

    // 8-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7);
    }

    // 9-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8);
    }

    // 10-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8],
                  const std::vector<T9*>&, const ByteOffset[DIM9])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8, ptrs_9, offsets_9);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8],
               const std::vector<T9*>&, const ByteOffset[DIM9])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8, ptrs_9, offsets_9);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset,
                  const std::vector<T9*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8, ptrs_9, offset_9);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8, ptrs_9, offset_9);
    }

    // 11-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8],
                  const std::vector<T9*>&, const ByteOffset[DIM9],
                  const std::vector<T10*>&, const ByteOffset[DIM10])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8, ptrs_9, offsets_9,
                                 ptrs_10, offsets_10);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8],
               const std::vector<T9*>&, const ByteOffset[DIM9],
               const std::vector<T10*>&, const ByteOffset[DIM10])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8, ptrs_9, offsets_9,
                                        ptrs_10, offsets_10);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset,
                  const std::vector<T9*>&, const ByteOffset,
                  const std::vector<T10*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8, ptrs_9, offset_9,
                                 ptrs_10, offset_10);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8, ptrs_9, offset_9,
                                        ptrs_10, offset_10);
    }

    // 12-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8],
                  const std::vector<T9*>&, const ByteOffset[DIM9],
                  const std::vector<T10*>&, const ByteOffset[DIM10],
                  const std::vector<T11*>&, const ByteOffset[DIM11])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8, ptrs_9, offsets_9,
                                 ptrs_10, offsets_10, ptrs_11, offsets_11);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8],
               const std::vector<T9*>&, const ByteOffset[DIM9],
               const std::vector<T10*>&, const ByteOffset[DIM10],
               const std::vector<T11*>&, const ByteOffset[DIM11])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8, ptrs_9, offsets_9,
                                        ptrs_10, offsets_10, ptrs_11, offsets_11);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset,
                  const std::vector<T9*>&, const ByteOffset,
                  const std::vector<T10*>&, const ByteOffset,
                  const std::vector<T11*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8, ptrs_9, offset_9,
                                 ptrs_10, offset_10, ptrs_11, offset_11);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8, ptrs_9, offset_9,
                                        ptrs_10, offset_10, ptrs_11, offset_11);
    }

    // 13-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8],
                  const std::vector<T9*>&, const ByteOffset[DIM9],
                  const std::vector<T10*>&, const ByteOffset[DIM10],
                  const std::vector<T11*>&, const ByteOffset[DIM11],
                  const std::vector<T12*>&, const ByteOffset[DIM12])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8, ptrs_9, offsets_9,
                                 ptrs_10, offsets_10, ptrs_11, offsets_11,
                                 ptrs_12, offsets_12);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8],
               const std::vector<T9*>&, const ByteOffset[DIM9],
               const std::vector<T10*>&, const ByteOffset[DIM10],
               const std::vector<T11*>&, const ByteOffset[DIM11],
               const std::vector<T12*>&, const ByteOffset[DIM12])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8, ptrs_9, offsets_9,
                                        ptrs_10, offsets_10, ptrs_11, offsets_11,
                                        ptrs_12, offsets_12);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset,
                  const std::vector<T9*>&, const ByteOffset,
                  const std::vector<T10*>&, const ByteOffset,
                  const std::vector<T11*>&, const ByteOffset,
                  const std::vector<T12*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8, ptrs_9, offset_9,
                                 ptrs_10, offset_10, ptrs_11, offset_11,
                                 ptrs_12, offset_12);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8, ptrs_9, offset_9,
                                        ptrs_10, offset_10, ptrs_11, offset_11,
                                        ptrs_12, offset_12);
    }

    // 14-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12, typename T13, int DIM13,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8],
                  const std::vector<T9*>&, const ByteOffset[DIM9],
                  const std::vector<T10*>&, const ByteOffset[DIM10],
                  const std::vector<T11*>&, const ByteOffset[DIM11],
                  const std::vector<T12*>&, const ByteOffset[DIM12],
                  const std::vector<T13*>&, const ByteOffset[DIM13])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      GET_RAW_POINTERS(13);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8, ptrs_9, offsets_9,
                                 ptrs_10, offsets_10, ptrs_11, offsets_11,
                                 ptrs_12, offsets_12, ptrs_13, offsets_13);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12, typename T13, int DIM13,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8],
               const std::vector<T9*>&, const ByteOffset[DIM9],
               const std::vector<T10*>&, const ByteOffset[DIM10],
               const std::vector<T11*>&, const ByteOffset[DIM11],
               const std::vector<T12*>&, const ByteOffset[DIM12],
               const std::vector<T13*>&, const ByteOffset[DIM13])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      GET_RAW_POINTERS(13);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8, ptrs_9, offsets_9,
                                        ptrs_10, offsets_10, ptrs_11, offsets_11,
                                        ptrs_12, offsets_12, ptrs_13, offsets_13);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12, typename T13, int DIM13,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset,
                  const std::vector<T9*>&, const ByteOffset,
                  const std::vector<T10*>&, const ByteOffset,
                  const std::vector<T11*>&, const ByteOffset,
                  const std::vector<T12*>&, const ByteOffset,
                  const std::vector<T13*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      GET_DENSE_POINTERS(13);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8, ptrs_9, offset_9,
                                 ptrs_10, offset_10, ptrs_11, offset_11,
                                 ptrs_12, offset_12, ptrs_13, offset_13);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12, typename T13, int DIM13,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset,
               const std::vector<T13*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      GET_DENSE_POINTERS(13);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8, ptrs_9, offset_9,
                                        ptrs_10, offset_10, ptrs_11, offset_11,
                                        ptrs_12, offset_12, ptrs_13, offset_13);
    }

    // 15-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12, typename T13, int DIM13,
             typename T14, int DIM14,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8],
                  const std::vector<T9*>&, const ByteOffset[DIM9],
                  const std::vector<T10*>&, const ByteOffset[DIM10],
                  const std::vector<T11*>&, const ByteOffset[DIM11],
                  const std::vector<T12*>&, const ByteOffset[DIM12],
                  const std::vector<T13*>&, const ByteOffset[DIM13],
                  const std::vector<T14*>&, const ByteOffset[DIM14])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      GET_RAW_POINTERS(13);
      GET_RAW_POINTERS(14);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8, ptrs_9, offsets_9,
                                 ptrs_10, offsets_10, ptrs_11, offsets_11,
                                 ptrs_12, offsets_12, ptrs_13, offsets_13,
                                 ptrs_14, offsets_14);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12, typename T13, int DIM13,
                         typename T14, int DIM14,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8],
               const std::vector<T9*>&, const ByteOffset[DIM9],
               const std::vector<T10*>&, const ByteOffset[DIM10],
               const std::vector<T11*>&, const ByteOffset[DIM11],
               const std::vector<T12*>&, const ByteOffset[DIM12],
               const std::vector<T13*>&, const ByteOffset[DIM13],
               const std::vector<T14*>&, const ByteOffset[DIM14])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      GET_RAW_POINTERS(13);
      GET_RAW_POINTERS(14);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8, ptrs_9, offsets_9,
                                        ptrs_10, offsets_10, ptrs_11, offsets_11,
                                        ptrs_12, offsets_12, ptrs_13, offsets_13,
                                        ptrs_14, offsets_14);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12, typename T13, int DIM13,
             typename T14, int DIM14,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset,
                  const std::vector<T9*>&, const ByteOffset,
                  const std::vector<T10*>&, const ByteOffset,
                  const std::vector<T11*>&, const ByteOffset,
                  const std::vector<T12*>&, const ByteOffset,
                  const std::vector<T13*>&, const ByteOffset,
                  const std::vector<T14*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      GET_DENSE_POINTERS(13);
      GET_DENSE_POINTERS(14);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8, ptrs_9, offset_9,
                                 ptrs_10, offset_10, ptrs_11, offset_11,
                                 ptrs_12, offset_12, ptrs_13, offset_13,
                                 ptrs_14, offset_14);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12, typename T13, int DIM13,
                         typename T14, int DIM14,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset,
               const std::vector<T13*>&, const ByteOffset,
               const std::vector<T14*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      GET_DENSE_POINTERS(13);
      GET_DENSE_POINTERS(14);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8, ptrs_9, offset_9,
                                        ptrs_10, offset_10, ptrs_11, offset_11,
                                        ptrs_12, offset_12, ptrs_13, offset_13,
                                        ptrs_14, offset_14);
    }

    // 16-Region

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12, typename T13, int DIM13,
             typename T14, int DIM14, typename T15, int DIM15,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3],
                  const std::vector<T4*>&, const ByteOffset[DIM4],
                  const std::vector<T5*>&, const ByteOffset[DIM5],
                  const std::vector<T6*>&, const ByteOffset[DIM6],
                  const std::vector<T7*>&, const ByteOffset[DIM7],
                  const std::vector<T8*>&, const ByteOffset[DIM8],
                  const std::vector<T9*>&, const ByteOffset[DIM9],
                  const std::vector<T10*>&, const ByteOffset[DIM10],
                  const std::vector<T11*>&, const ByteOffset[DIM11],
                  const std::vector<T12*>&, const ByteOffset[DIM12],
                  const std::vector<T13*>&, const ByteOffset[DIM13],
                  const std::vector<T14*>&, const ByteOffset[DIM14],
                  const std::vector<T15*>&, const ByteOffset[DIM15])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      GET_RAW_POINTERS(13);
      GET_RAW_POINTERS(14);
      GET_RAW_POINTERS(15);
      (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                 ptrs_2, offsets_2, ptrs_3, offsets_3,
                                 ptrs_4, offsets_4, ptrs_5, offsets_5,
                                 ptrs_6, offsets_6, ptrs_7, offsets_7,
                                 ptrs_8, offsets_8, ptrs_9, offsets_9,
                                 ptrs_10, offsets_10, ptrs_11, offsets_11,
                                 ptrs_12, offsets_12, ptrs_13, offsets_13,
                                 ptrs_14, offsets_14, ptrs_15, offsets_15);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12, typename T13, int DIM13,
                         typename T14, int DIM14, typename T15, int DIM15,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3],
               const std::vector<T4*>&, const ByteOffset[DIM4],
               const std::vector<T5*>&, const ByteOffset[DIM5],
               const std::vector<T6*>&, const ByteOffset[DIM6],
               const std::vector<T7*>&, const ByteOffset[DIM7],
               const std::vector<T8*>&, const ByteOffset[DIM8],
               const std::vector<T9*>&, const ByteOffset[DIM9],
               const std::vector<T10*>&, const ByteOffset[DIM10],
               const std::vector<T11*>&, const ByteOffset[DIM11],
               const std::vector<T12*>&, const ByteOffset[DIM12],
               const std::vector<T13*>&, const ByteOffset[DIM13],
               const std::vector<T14*>&, const ByteOffset[DIM14],
               const std::vector<T15*>&, const ByteOffset[DIM15])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_RAW_POINTERS(0);
      GET_RAW_POINTERS(1);
      GET_RAW_POINTERS(2);
      GET_RAW_POINTERS(3);
      GET_RAW_POINTERS(4);
      GET_RAW_POINTERS(5);
      GET_RAW_POINTERS(6);
      GET_RAW_POINTERS(7);
      GET_RAW_POINTERS(8);
      GET_RAW_POINTERS(9);
      GET_RAW_POINTERS(10);
      GET_RAW_POINTERS(11);
      GET_RAW_POINTERS(12);
      GET_RAW_POINTERS(13);
      GET_RAW_POINTERS(14);
      GET_RAW_POINTERS(15);
      return (*PTR)(task, ctx, runtime, ptrs_0, offsets_0, ptrs_1, offsets_1,
                                        ptrs_2, offsets_2, ptrs_3, offsets_3,
                                        ptrs_4, offsets_4, ptrs_5, offsets_5,
                                        ptrs_6, offsets_6, ptrs_7, offsets_7,
                                        ptrs_8, offsets_8, ptrs_9, offsets_9,
                                        ptrs_10, offsets_10, ptrs_11, offsets_11,
                                        ptrs_12, offsets_12, ptrs_13, offsets_13,
                                        ptrs_14, offsets_14, ptrs_15, offsets_15);
    }

    template<typename T0, int DIM0, typename T1, int DIM1,
             typename T2, int DIM2, typename T3, int DIM3,
             typename T4, int DIM4, typename T5, int DIM5,
             typename T6, int DIM6, typename T7, int DIM7,
             typename T8, int DIM8, typename T9, int DIM9,
             typename T10, int DIM10, typename T11, int DIM11,
             typename T12, int DIM12, typename T13, int DIM13,
             typename T14, int DIM14, typename T15, int DIM15,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset,
                  const std::vector<T4*>&, const ByteOffset,
                  const std::vector<T5*>&, const ByteOffset,
                  const std::vector<T6*>&, const ByteOffset,
                  const std::vector<T7*>&, const ByteOffset,
                  const std::vector<T8*>&, const ByteOffset,
                  const std::vector<T9*>&, const ByteOffset,
                  const std::vector<T10*>&, const ByteOffset,
                  const std::vector<T11*>&, const ByteOffset,
                  const std::vector<T12*>&, const ByteOffset,
                  const std::vector<T13*>&, const ByteOffset,
                  const std::vector<T14*>&, const ByteOffset,
                  const std::vector<T15*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      GET_DENSE_POINTERS(13);
      GET_DENSE_POINTERS(14);
      GET_DENSE_POINTERS(15);
      (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                 ptrs_2, offset_2, ptrs_3, offset_3,
                                 ptrs_4, offset_4, ptrs_5, offset_5,
                                 ptrs_6, offset_6, ptrs_7, offset_7,
                                 ptrs_8, offset_8, ptrs_9, offset_9,
                                 ptrs_10, offset_10, ptrs_11, offset_11,
                                 ptrs_12, offset_12, ptrs_13, offset_13,
                                 ptrs_14, offset_14, ptrs_15, offset_15);
    }

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
                         typename T2, int DIM2, typename T3, int DIM3,
                         typename T4, int DIM4, typename T5, int DIM5,
                         typename T6, int DIM6, typename T7, int DIM7,
                         typename T8, int DIM8, typename T9, int DIM9,
                         typename T10, int DIM10, typename T11, int DIM11,
                         typename T12, int DIM12, typename T13, int DIM13,
                         typename T14, int DIM14, typename T15, int DIM15,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset,
               const std::vector<T4*>&, const ByteOffset,
               const std::vector<T5*>&, const ByteOffset,
               const std::vector<T6*>&, const ByteOffset,
               const std::vector<T7*>&, const ByteOffset,
               const std::vector<T8>*&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset,
               const std::vector<T13*>&, const ByteOffset,
               const std::vector<T14*>&, const ByteOffset,
               const std::vector<T15*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime)
    {
      GET_DENSE_POINTERS(0);
      GET_DENSE_POINTERS(1);
      GET_DENSE_POINTERS(2);
      GET_DENSE_POINTERS(3);
      GET_DENSE_POINTERS(4);
      GET_DENSE_POINTERS(5);
      GET_DENSE_POINTERS(6);
      GET_DENSE_POINTERS(7);
      GET_DENSE_POINTERS(8);
      GET_DENSE_POINTERS(9);
      GET_DENSE_POINTERS(10);
      GET_DENSE_POINTERS(11);
      GET_DENSE_POINTERS(12);
      GET_DENSE_POINTERS(13);
      GET_DENSE_POINTERS(14);
      GET_DENSE_POINTERS(15);
      return (*PTR)(task, ctx, runtime, ptrs_0, offset_0, ptrs_1, offset_1,
                                        ptrs_2, offset_2, ptrs_3, offset_3,
                                        ptrs_4, offset_4, ptrs_5, offset_5,
                                        ptrs_6, offset_6, ptrs_7, offset_7,
                                        ptrs_8, offset_8, ptrs_9, offset_9,
                                        ptrs_10, offset_10, ptrs_11, offset_11,
                                        ptrs_12, offset_12, ptrs_13, offset_13,
                                        ptrs_14, offset_14, ptrs_15, offset_15);
    }

#undef GET_RAW_POINTERS
#undef GET_DENSE_POINTERS
  }; 
};

