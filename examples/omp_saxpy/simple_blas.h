/* Copyright 2022 Stanford University
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

#ifndef _OMP_SAXPY_H_
#define _OMP_SAXPY_H_

#include <legion.h>

using namespace Legion;

extern int blas_thread_count;
extern bool blas_do_parallel;

template <typename T>
class BlasArrayRef {
public:
  static const FieldID DEFAULT_FID = 22;

  BlasArrayRef(LogicalRegion _region, FieldID _fid = DEFAULT_FID);
  BlasArrayRef(const BlasArrayRef<T>& copy_from);
  ~BlasArrayRef(void);

  static BlasArrayRef<T> create(Runtime *runtime, Context ctx, IndexSpace is,
				FieldID fid = DEFAULT_FID);

  void destroy(Runtime *runtime, Context ctx);

  void fill(Runtime *runtime, Context ctx, T fill_val);

  template <typename LT>
  void add_requirement(LT& launcher, PrivilegeMode mode,
		       CoherenceProperty prop = EXCLUSIVE) const;

protected:
  LogicalRegion region;
  FieldID fid;
};

template <typename T>
void axpy(Runtime *runtime, Context ctx,
	  T alpha, const BlasArrayRef<T>& x, BlasArrayRef<T> y,
	  IndexPartition distpart = IndexPartition::NO_PART);

template <typename T>
T dot(Runtime *runtime, Context ctx,
      const BlasArrayRef<T>& x, BlasArrayRef<T> y,
      IndexPartition distpart = IndexPartition::NO_PART);

template <typename T>
class BlasTaskImplementations {
public:
  TaskID axpy_task_id;
  TaskID dot_task_id;

  // performs _static_ registration of tasks at startup
  void preregister_tasks(void);

protected:
public: // these must be public as they are used as template arguments
  static void axpy_task_cpu(const Task *task,
			    const std::vector<PhysicalRegion> &regions,
			    Context ctx, Runtime *runtime);

  static T dot_task_cpu(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime);
};

// single-precision float

extern BlasTaskImplementations<float> blas_impl_s;

#include "simple_blas.inl"

#endif
