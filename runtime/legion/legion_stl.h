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

#ifndef __LEGION_STL_H__
#define __LEGION_STL_H__

#include <set>
#include <map>
#include <vector>
#include "legion.h"

namespace Legion {
  namespace STL {

    /*
     * Helpers for typed serialization.
     *
     * WARNING: Currently only supports POD types.
     */

    template<typename ... Ts>
    size_t get_serialized_size();

    template<typename ... Ts>
    void serialize(void *buffer, const Ts & ... ts);

    template<typename ... Ts>
    std::tuple<Ts ...> deserialize(const void *buffer);

    /*
     * A helper for building a typed TaskArgument.
     */
    template<typename ... Ts>
    class TypedArgument
    {
    public:
      TypedArgument(const Ts& ... ts);
      ~TypedArgument();

      operator TaskArgument() const;

      size_t get_size() const;
      void *get_ptr() const;

    private:
      void *buffer;
      size_t buf_size;
    };

    /*
     * Provide some wrappers for serializing and deserializing STL
     * data structures when returning them as results from Legion tasks
     */
    template<typename T>
    class set : public std::set<T> {
    public:
      inline size_t legion_buffer_size(void) const;
      inline void legion_serialize(void *buffer) const;
      inline void legion_deserialize(const void *buffer);
    };

    template<typename T1, typename T2>
    class map : public std::map<T1,T2> {
    public:
      inline size_t legion_buffer_size(void) const;
      inline void legion_serialize(void *buffer) const;
      inline void legion_deserialize(const void *buffer);
    };

    template<typename T>
    class vector : public std::vector<T> {
    public:
      inline size_t legion_buffer_size(void) const;
      inline void legion_serialize(void *buffer) const;
      inline void legion_deserialize(const void *buffer);
    };

    /*
     * These methods can be used to create a Legion task from a function
     * that simply wants vectors of pointers for each field and the associated
     * offsets for using the pointer to stride through an index space.
     * There are two primary methods: ones that use raw_rect_ptr underneath
     * and ones that use raw_dense_pointer underneath. The raw_rect_ptr methods
     * will work with any layout, whereas the raw_dense_ptr methods can only
     * be used with instances of contiguous data. There are two variants of
     * each method for void and non-void return types. All variants currently
     * are templated to specify the types of pointers and the number of dimensions
     * in the index space associated with each logical region. The vector
     * of pointers is in the order of the privilege fields for the region
     * requirement for the task.
     */

    // 1-Region

    template<typename T0, int DIM0,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, 
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T0, int DIM0,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, 
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 2-Regions

    template<typename T0, int DIM0, typename T1, int DIM1,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T0, int DIM0, typename T1, int DIM1,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, typename T1, int DIM1,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 3-Regions

    template<typename T0, int DIM0, typename T1, int DIM1, typename T2, int DIM2,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, typename T1, int DIM1, 
                         typename T2, int DIM2,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T0, int DIM0, typename T1, int DIM1, typename T2, int DIM2,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, typename T1, int DIM1, 
                         typename T2, int DIM2,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 4-Regions

    template<typename T0, int DIM0, typename T1, int DIM1, 
             typename T2, int DIM2, typename T3, int DIM3,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset[DIM0],
                  const std::vector<T1*>&, const ByteOffset[DIM1],
                  const std::vector<T2*>&, const ByteOffset[DIM2],
                  const std::vector<T3*>&, const ByteOffset[DIM3])>
    static void raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, typename T1, int DIM1, 
                         typename T2, int DIM2, typename T3, int DIM3,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset[DIM0],
               const std::vector<T1*>&, const ByteOffset[DIM1],
               const std::vector<T2*>&, const ByteOffset[DIM2],
               const std::vector<T3*>&, const ByteOffset[DIM3])>
    static T raw_rect_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T0, int DIM0, typename T1, int DIM1, 
             typename T2, int DIM2, typename T3, int DIM3,
      void (*PTR)(const Task*, Context, Runtime*,
                  const std::vector<T0*>&, const ByteOffset,
                  const std::vector<T1*>&, const ByteOffset,
                  const std::vector<T2*>&, const ByteOffset,
                  const std::vector<T3*>&, const ByteOffset)>
    static void raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    template<typename T, typename T0, int DIM0, typename T1, int DIM1, 
                         typename T2, int DIM2, typename T3, int DIM3,
      T (*PTR)(const Task*, Context, Runtime*, 
               const std::vector<T0*>&, const ByteOffset,
               const std::vector<T1*>&, const ByteOffset,
               const std::vector<T2*>&, const ByteOffset,
               const std::vector<T3*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 5-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 6-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);
    
    // 7-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);
    
    // 8-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 9-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 10-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 11-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 12-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 13-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 14-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset,
               const std::vector<T13*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 15-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset,
               const std::vector<T13*>&, const ByteOffset,
               const std::vector<T14*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

    // 16-Regions

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

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
               const std::vector<T8*>&, const ByteOffset,
               const std::vector<T9*>&, const ByteOffset,
               const std::vector<T10*>&, const ByteOffset,
               const std::vector<T11*>&, const ByteOffset,
               const std::vector<T12*>&, const ByteOffset,
               const std::vector<T13*>&, const ByteOffset,
               const std::vector<T14*>&, const ByteOffset,
               const std::vector<T15*>&, const ByteOffset)>
    static T raw_dense_task_wrapper(const Task *task, 
       const std::vector<PhysicalRegion>& regions, Context ctx, Runtime *runtime);

  }; // namespace STL
}; // namespace Legion

#include "legion/legion_stl.inl"

#endif // __LEGION_STL_H__
