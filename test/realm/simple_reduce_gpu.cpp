#include "hip/hip_runtime.h"
#include "realm.h"

#include "simple_reduce.h"

extern void register_gpu_reduction(Realm::Runtime& realm,
                                   Realm::ReductionOpID redop_id);


void register_gpu_reduction(Realm::Runtime& realm,
                            Realm::ReductionOpID redop_id)
{
  realm.register_reduction(redop_id,
                           Realm::ReductionOpUntyped::create_reduction_op<ReductionOpMixedAdd>());
}

