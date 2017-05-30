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

// data transfer (a.k.a. dma) engine for Realm

#ifndef REALM_TRANSFER_H
#define REALM_TRANSFER_H

#include "event.h"
#include "memory.h"
#include "indexspace.h"

namespace Realm {

  // the data transfer engine has too much code to have it all be templated on the
  //  type of ZIndexSpace that is driving the transfer, so we need a widget that
  //  can hold an arbitrary ZIndexSpace and dispatch based on its type

  class TransferDomain {
  protected:
    TransferDomain(void);

  public:
    static TransferDomain *construct(Domain d);

    virtual ~TransferDomain(void);
  };

  class TransferPlan {
  protected:
    // subclasses constructed in plan_* calls below
    TransferPlan(void);

  public:
    virtual ~TransferPlan(void);

    static bool plan_copy(std::vector<TransferPlan *>& plans,
			  const std::vector<CopySrcDstField> &srcs,
			  const std::vector<CopySrcDstField> &dsts,
			  ReductionOpID redop_id = 0, bool red_fold = false);

    static bool plan_fill(std::vector<TransferPlan *>& plans,
			  const std::vector<CopySrcDstField> &dsts,
			  const void *fill_value, size_t fill_value_size);

    virtual Event execute_plan(const TransferDomain *td,
			       const ProfilingRequestSet& requests,
			       Event wait_on, int priority) = 0;
  };

}; // namespace Realm

#endif // ifndef REALM_TRANSFER_H
