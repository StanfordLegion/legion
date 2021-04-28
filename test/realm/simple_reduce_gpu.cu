#include "realm.h"

#include "simple_reduce.h"

extern void register_gpu_reduction(Realm::Runtime& realm,
                                   Realm::ReductionOpID redop_id);


void register_gpu_reduction(Realm::Runtime& realm,
                            Realm::ReductionOpID redop_id)
{
  realm.register_reduction<ReductionOpMixedAdd>(redop_id);
}

