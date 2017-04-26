/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_AGENCY_H__
#define __LEGION_AGENCY_H__

/**
 * \file legion_agency.h
 * Support for using Agency inside of Legion tasks
 */

#include "legion.h"
#include <agency/agency.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/sequenced_executor.hpp>
#include <agency/execution/executor/vector_executor.hpp>
// Agency doesn't know that we hijacked OpenMP so fake it
#ifndef _OPENMP
#define _OPENMP 201307
#endif
#include <agency/omp/execution.hpp>
#ifdef __CUDACC__
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#endif

namespace Legion {

  namespace Internal {
    using ExecutorVariant = agency::experimental::variant<
                              agency::sequenced_executor,
                              agency::vector_executor,
                              agency::omp::parallel_for_executor
#ifdef __CUDACC__
                              , agency::cuda::parallel_executor
#endif
                              >;
    template<typename T>
    using FutureVariant = agency::experimental::variant<
                agency::executor_future_t<agency::sequenced_executor,T>,
                agency::executor_future_t<agency::vector_executor,T>,
                agency::executor_future_t<agency::omp::parallel_for_executor,T>
#ifdef __CUDACC__
                , agency::executor_future_t<agency::cuda::parallel_executor,T> 
#endif
                >;
  };

  /**
   * \class LegionExecutorFuture
   * This class implements an executor future
   */
  template<class T>
  class LegionExecutorFuture {
  public:
    LegionExecutorFuture(void) = default;
    LegionExecutorFuture(LegionExecutorFuture&& f);
  public:
    inline LegionExecutorFuture<T>& operator=(LegionExecutorFuture<T>&& rhs);
  public:
    template<class E>
    void set_future(agency::executor_future_t<E,T> &&rhs);
    template<class E>
    agency::executor_future_t<E,T>* get_future(void);
  public:
    inline void wait(void) const;
    inline T get(void);
    inline bool valid(void) const;
  public:
    template<class... Args>
    static inline LegionExecutorFuture<T> make_ready(Args&&... args);
  protected:
    Internal::FutureVariant<T> chosen_future;
  };

  /**
   * \class LegionExecutor
   * This class wraps executors for many different processor
   * kinds such that the same executor can be used to implement
   * a task variant that can be run on many different processor types
   */
  template<class Shape = size_t, class Index = Shape>
  class LegionExecutor {
  public:
    using execution_category = agency::parallel_execution_tag;
    using shape_type = Shape;
    using index_type = Index;
    template<class T>
    using future = LegionExecutorFuture<T>;
  public:
    // Optional task name for error messages
    LegionExecutor(const char *task_name = NULL);
    // Optionally pass in the Task* for the executor to get error messages
    LegionExecutor(const Task *task);
    // Provide overrides for all the agency customization points such that 
    // we get the most basic functionality of each kind of base executor
  public: // Bulk methods
    template<class Function, class ResultFactory, class SharedFactory>
    inline typename std::result_of<ResultFactory()>::type
      bulk_sync_execute(Function f, shape_type shape, 
          ResultFactory result_factory, SharedFactory shared_factory);

    template<class Function, class ResultFactory, class SharedFactory>
    inline LegionExecutorFuture<typename std::result_of<ResultFactory()>::type>
      bulk_async_execute(Function f, shape_type shape, 
                         ResultFactory result_factory, 
                         SharedFactory shared_factory);
    
    template<class Function, class T, class ResultFactory, class SharedFactory>
    inline LegionExecutorFuture<typename std::result_of<ResultFactory()>::type>
      bulk_then_execute(Function f, shape_type shape, 
                        LegionExecutorFuture<T>& predecessor,
                        ResultFactory result_factory, 
                        SharedFactory shared_factory); 
  public: // Singular methods
    template<class Function>
    inline typename std::result_of<agency::detail::decay_t<Function>()>
      sync_execute(Function f);

    template<class Function>
    inline LegionExecutorFuture<
            typename std::result_of<agency::detail::decay_t<Function>()> >
      async_execute(Function f);

    template<class Function, class T>
    inline LegionExecutorFuture<
            typename std::result_of<agency::detail::decay_t<Function>()> >
      then_execute(Function f, LegionExecutorFuture<T>& predecessor);
  public: // Future methods
    template<class T, class... Args>
    inline LegionExecutorFuture<T> make_ready_future(Args&&... args);
  protected:
    inline void select_local_executor(Processor p, const char *task_name);
  protected:
    Internal::ExecutorVariant chosen_executor;
  };

}; // namespace Legion

#include "legion_agency.inl"

#endif // __LEGION_AGENCY_H__

