#include "realm.h"
#include "realm/cuda/cuda_module.h"
#include "simple_reduce.h"

extern Realm::Logger log_app;

#if defined(REALM_USE_CUDA) && !defined(REALM_USE_CUDART_HIJACK)
static void check_cudart(cudaError_t err, const char *tok, const char *file, int line)
{
  if(err != cudaSuccess) {
    log_app.fatal("%s(%d): Error in %s = %d", file, line, tok, (int)err);
    abort();
  }
}

#define CHECK_CURT(x) check_cudart((x), #x, __FILE__, __LINE__)
#endif

void register_gpu_reduction(Realm::Runtime &realm, Realm::ReductionOpID redop_id)
{
  // Register the redop the normal way (this will _not_ register the cuda redops, we'll do
  // that later)
  if(!realm.register_reduction<ReductionOpMixedAdd>(redop_id)) {
    log_app.fatal("Failed to register reduction op");
  }

#if defined(REALM_USE_CUDA) && !defined(REALM_USE_CUDART_HIJACK)
  Realm::Cuda::CudaModule *cuda = realm.get_module<Realm::Cuda::CudaModule>("cuda");
  if(cuda != nullptr) {
    Realm::Machine::ProcessorQuery pq =
        Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
            .only_kind(Realm::Processor::TOC_PROC);
    std::vector<Realm::Cuda::CudaRedOpDesc> descs;
    for(Realm::Processor p : pq) {
      int devid = 0;
      Realm::Cuda::CudaRedOpDesc desc;
      if(!cuda->get_cuda_device_id(p, &devid)) {
        continue;
      }
      desc.proc = p;
      desc.redop_id = redop_id;
      CHECK_CURT(cudaSetDevice(devid));
      // Retrieve all the function pointers
      // Note: yes, this could be done with the Realm::Runtime::register_reduction
      // function, but we want to specifically test that the CUfunc_st* registration works
      // for those that can't get a host symbol from the runtime for their reduction
      // operators for whatever reason.  Rather than writing a specific cubin and loading
      // it with the driver api, we'll just use the cuda runtime's API to retrieve the
      // CUfunc by symbol and reuse the already defined reduction operators exposed by the
      // runtime
      CHECK_CURT(cudaGetFuncBySymbol(
          static_cast<cudaFunction_t *>(&desc.apply_excl),
          reinterpret_cast<const void *>(Realm::Cuda::ReductionKernels::apply_cuda_kernel<
                                         ReductionOpMixedAdd, true>)));
      CHECK_CURT(cudaGetFuncBySymbol(
          static_cast<cudaFunction_t *>(&desc.apply_nonexcl),
          reinterpret_cast<const void *>(Realm::Cuda::ReductionKernels::apply_cuda_kernel<
                                         ReductionOpMixedAdd, false>)));
      CHECK_CURT(cudaGetFuncBySymbol(
          static_cast<cudaFunction_t *>(&desc.fold_excl),
          reinterpret_cast<const void *>(Realm::Cuda::ReductionKernels::fold_cuda_kernel<
                                         ReductionOpMixedAdd, true>)));
      CHECK_CURT(cudaGetFuncBySymbol(
          static_cast<cudaFunction_t *>(&desc.fold_nonexcl),
          reinterpret_cast<const void *>(Realm::Cuda::ReductionKernels::fold_cuda_kernel<
                                         ReductionOpMixedAdd, false>)));
      descs.push_back(desc);
    }
    Realm::Event e = Realm::Event::NO_EVENT;
    if(!cuda->register_reduction(e, descs.data(), descs.size())) {
      assert(0 && "Failed to register reduction directly with cuda module");
    }
    e.wait();
  }
#endif
}
