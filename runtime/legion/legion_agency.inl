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

namespace Legion {

    /////////////////////////////////////////////////////////////
    // Legion Executor Future 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<class T>
    inline LegionExecutorFuture<T>& LegionExecutorFuture<T>::operator=(
                                                  LegionExecutorFuture<T>&& rhs)
    //--------------------------------------------------------------------------
    {
      chosen_future = agency::experimental::visit([=](auto&& fut)
          {
            return std::move(fut);
          }, 
          rhs.chosen_future);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<class T> template<class E>
    inline void LegionExecutorFuture<T>::set_future(
                                           agency::executor_future_t<E,T> &&rhs)
    //--------------------------------------------------------------------------
    {
      chosen_future = std::move(rhs);
    }

    //--------------------------------------------------------------------------
    template<class T> template<class E>
    inline agency::executor_future_t<E,T>* 
                                       LegionExecutorFuture<T>::get_future(void)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](auto &fut)
          {
            return &fut;
          },
          chosen_future);
    }

    //--------------------------------------------------------------------------
    template<class T>
    inline void LegionExecutorFuture<T>::wait(void) const
    //--------------------------------------------------------------------------
    {
      agency::experimental::visit([=](const auto &fut)
          {
            fut.wait();
          },
          chosen_future);
    }

    //--------------------------------------------------------------------------
    template<class T>
    inline T LegionExecutorFuture<T>::get(void)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](auto &fut)
          {
            return fut.get();
          }, 
          chosen_future);
    }

    //--------------------------------------------------------------------------
    template<class T>
    inline bool LegionExecutorFuture<T>::valid(void) const
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](const auto &fut)
          {
            return fut.valid();
          },
          chosen_future);
    }

    //--------------------------------------------------------------------------
    template<class T> template<class... Args>
    /*static*/ inline LegionExecutorFuture<T> 
                             LegionExecutorFuture<T>::make_ready(Args&&... args)
    //--------------------------------------------------------------------------
    {
      LegionExecutor<> exec;
      return agency::make_ready_future<T>(exec, std::forward<Args>(args)...);
    }

    /////////////////////////////////////////////////////////////
    // Legion Executor 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<class Shape, class Index>
    LegionExecutor<Shape,Index>::LegionExecutor(const char *task_name/*NULL*/)
    //--------------------------------------------------------------------------
    {
      // Use the kind of our current Realm processor to
      // determine the kind of executor that we should use
      Processor p = Processor::get_executing_processor();
      select_local_executor(p, task_name);
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index>
    LegionExecutor<Shape,Index>::LegionExecutor(const Task *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(task != NULL);
#endif
      Processor p = Processor::get_executing_processor();
      select_local_executor(p, task->get_task_name());
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index>
    inline void LegionExecutor<Shape,Index>::select_local_executor(Processor p,
                                                          const char *task_name)
    //--------------------------------------------------------------------------
    {
      switch (p.kind())
      {
        case Processor::LOC_PROC:
          {
            // Assume our CPUs have vector intrinsics, if
            // not the vector executor just falls back to
            // normal sequential execution
            chosen_executor = agency::vector_executor();
            break;
          }
        case Processor::OMP_PROC:
          {
            chosen_executor = agency::omp::parallel_for_executor();
            break;
          }
        case Processor::TOC_PROC:
          {
#ifdef __CUDACC__
            chosen_executor = agency::cuda::parallel_executor();
#else
            // This is an error because we need to be compiled
            // with nvcc if we are going to be running on GPU processors
            if (task_name != NULL)
              fprintf(stderr,"ERROR: using a LegionExecutor in an Agency "
                             "variant of task %s on a GPU processor, but "
                             "file %s was not compiled with nvcc! Please "
                             "make sure that %s is compiled with nvcc and "
                             "then try again.", task_name, __FILE__, __FILE__);
            else
              fprintf(stderr,"ERROR: using a LegionExecutor in an Agency "
                             "variant of an unknown task on a GPU processor, "
                             "but file %s was not compiled with nvcc! Please "
                             "make sure that %s is compiled with nvcc and "
                             "then try again. Note you can figure out which "
                             "task is responsible for this error by using the "
                             "LegionExecutor(const Task*) constructor.", 
                             __FILE__, __FILE__);
            assert(false);
#endif
            break;
          }
        case Processor::IO_PROC:
          {
            // I/O processors should just be sequential
            chosen_executor = agency::sequenced_executor();
            break;
          }
        default:
          {
            if (task_name != NULL)
              fprintf(stderr,"WARNING: LegionExecutor used in an Agency "
                             "variant of task %s is being used on an unknown "
                             "processor type %d. Falling back to the "
                             "sequential executor.", task_name, p.kind());
            else
              fprintf(stderr,"WARNING: LegionExecutor used in an Agency "
                             "variant of unknown task in file %s is being "
                             "used on an unknown processor type %d. Falling "
                             "back to the sequential executor.", 
                             __FILE__, p.kind());
            chosen_executor = agency::sequenced_executor();
          }
      }
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index>
    template<class Function, class ResultFactory, class SharedFactory>
    inline typename std::result_of<ResultFactory()>::type
    LegionExecutor<Shape,Index>::bulk_sync_execute(Function f, shape_type shape,
          ResultFactory result_factory, SharedFactory shared_factory)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](auto& exec)
          {
            return agency::bulk_sync_execute(exec, f, shape, 
                            result_factory, shared_factory);
          },
          chosen_executor);
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index>
    template<class Function, class ResultFactory, class SharedFactory>
    inline LegionExecutorFuture<typename std::result_of<ResultFactory()>::type>
      LegionExecutor<Shape,Index>::bulk_async_execute(Function f, 
                                shape_type shape, ResultFactory result_factory, 
                                SharedFactory shared_factory)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](auto &exec)
          {
            LegionExecutorFuture<
                typename std::result_of<ResultFactory()>::type> result;
            result.template set_future<typename 
                std::remove_reference<decltype(exec)>::type>(
                      agency::bulk_async_execute(exec, f, shape, 
                                  result_factory, shared_factory));
            return result;
          },
          chosen_executor);
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index>
    template<class Function, class T, class ResultFactory, class SharedFactory>
    inline LegionExecutorFuture<typename std::result_of<ResultFactory()>::type>
      LegionExecutor<Shape,Index>::bulk_then_execute(Function f, 
                    shape_type shape, LegionExecutorFuture<T> &predecessor,
                    ResultFactory result_factory, SharedFactory shared_factory)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=, &predecessor](auto &exec)
          {
            auto *base_future = predecessor.template get_future<typename
                                std::remove_reference<decltype(exec)>::type>();
            LegionExecutorFuture<
                typename std::result_of<ResultFactory()>::type> result;
            result.template set_future<typename
                std::remove_reference<decltype(exec)>::type>(
                  agency::bulk_then_execute(exec, f, shape, *base_future,
                                            result_factory, shared_factory));
            return result;
          },
          chosen_executor);
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index> template<class Function>
    inline typename std::result_of<agency::detail::decay_t<Function>()>
      LegionExecutor<Shape,Index>::sync_execute(Function f)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](auto& exec)
          {
            return agency::sync_execute(exec, f);
          },
          chosen_executor);
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index> template<class Function>
    inline LegionExecutorFuture<
            typename std::result_of<agency::detail::decay_t<Function>()> >
      LegionExecutor<Shape,Index>::async_execute(Function f)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](auto &exec)
          {
            LegionExecutorFuture<typename 
              std::result_of<agency::detail::decay_t<Function>()> > result;
            result.template set_future<typename
                std::remove_reference<decltype(exec)>::type>(
                  agency::async_execute(exec, f));
            return result;
          },
          chosen_executor);
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index> template<class Function, class T>
    inline LegionExecutorFuture<
                        std::result_of<agency::detail::decay_t<Function>()> >
      LegionExecutor<Shape,Index>::then_execute(Function f,
                                           LegionExecutorFuture<T>& predecessor)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=, &predecessor](auto &exec)
          {
            auto *base_future = predecessor.template get_future<typename
                                std::remove_reference<decltype(exec)>::type>();
            LegionExecutorFuture<typename 
              std::result_of<agency::detail::decay_t<Function>()> > result;
            result.template set_future<typename
                std::remove_reference<decltype(exec)>::type>(
                  agency::then_execute(exec, f, *base_future));
            return result;
          },
          chosen_executor);
    }

    //--------------------------------------------------------------------------
    template<class Shape, class Index> template<class T, class... Args>
    inline LegionExecutorFuture<T> 
      LegionExecutor<Shape,Index>::make_ready_future(Args&&... args)
    //--------------------------------------------------------------------------
    {
      return agency::experimental::visit([=](auto &exec)
          {
            LegionExecutorFuture<T> result;
            result.template set_future<typename
                std::remove_reference<decltype(exec)>::type>(
                  agency::make_ready_future<T>(exec, 
                                std::forward<Args>(args)...));
            return result;
          },
          chosen_executor);
    }

}; // namespace Legion

