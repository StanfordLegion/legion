#define __USE_GNU

#include "realm/cuda/cuda_internal.h"
#include "realm/cuda/cuda_module.h"

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <unordered_map>

// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface
// function For interposing dlopen(). Sell elf/dl-lib.c for the internal
// dlopen_mode interface function
extern "C" {
void *__libc_dlsym(void *map, const char *name);
}
extern "C" {
void *__libc_dlopen_mode(const char *name, int mode);
}

namespace Realm {

  namespace Cuda {

#if(CUDA_VERSION < 11030)
    typedef CUresult(CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr *dptr, size_t bytesize);
    typedef CUresult(CUDAAPI *PFN_cuMemFree_v3020)(CUdeviceptr dptr);
    typedef CUresult(CUDAAPI *PFN_cuLaunchKernel_v4000)(
        CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
    typedef CUresult(CUDAAPI *PFN_cuLaunchKernel_v7000_ptsz)(
        CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
    typedef CUresult(CUDAAPI *PFN_cuLaunchCooperativeKernel_v9000)(
        CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
    typedef CUresult(CUDAAPI *PFN_cuLaunchCooperativeKernel_v9000_ptsz)(
        CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
    typedef CUresult(CUDAAPI *PFN_cuMemcpyAsync_v4000)(CUdeviceptr dst, CUdeviceptr src,
                                                       size_t ByteCount,
                                                       CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpyAsync_v7000_ptsz)(CUdeviceptr dst,
                                                            CUdeviceptr src,
                                                            size_t ByteCount,
                                                            CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpy2DAsync_v3020)(const CUDA_MEMCPY2D *pCopy,
                                                         CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpy2DAsync_v7000_ptsz)(const CUDA_MEMCPY2D *pCopy,
                                                              CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpy3DAsync_v3020)(const CUDA_MEMCPY3D *pCopy,
                                                         CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpy3DAsync_v7000_ptsz)(const CUDA_MEMCPY3D *pCopy,
                                                              CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpyDtoHAsync_v3020)(void *dstHost,
                                                           CUdeviceptr srcDevice,
                                                           size_t ByteCount,
                                                           CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpyDtoHAsync_v7000_ptsz)(void *dstHost,
                                                                CUdeviceptr srcDevice,
                                                                size_t ByteCount,
                                                                CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpyHtoDAsync_v3020)(CUdeviceptr dstDevice,
                                                           const void *srcHost,
                                                           size_t ByteCount,
                                                           CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuMemcpyHtoDAsync_v7000_ptsz)(CUdeviceptr dstDevice,
                                                                const void *srcHost,
                                                                size_t ByteCount,
                                                                CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuEventRecord_v2000)(CUevent hEvent, CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuEventRecord_v7000_ptsz)(CUevent hEvent,
                                                            CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuEventRecordWithFlags_v11010)(CUevent hEvent,
                                                                 CUstream hStream,
                                                                 unsigned int flags);
    typedef CUresult(CUDAAPI *PFN_cuEventRecordWithFlags_v11010_ptsz)(CUevent hEvent,
                                                                      CUstream hStream,
                                                                      unsigned int flags);
    typedef CUresult(CUDAAPI *PFN_cuEventSynchronize_v2000)(CUevent hEvent);
    typedef CUresult(CUDAAPI *PFN_cuStreamSynchronize_v2000)(CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuStreamSynchronize_v7000_ptsz)(CUstream hStream);
    typedef CUresult(CUDAAPI *PFN_cuCtxSynchronize_v2000)(void);

    typedef enum CUdriverProcAddress_flags_enum
    {
      CU_GET_PROC_ADDRESS_DEFAULT = 0, /**< Default search mode for driver symbols. */
      CU_GET_PROC_ADDRESS_LEGACY_STREAM =
          1 << 0, /**< Search for legacy versions of driver symbols. */
      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM =
          1 << 1 /**< Search for per-thread versions of driver symbols. */
    } CUdriverProcAddress_flags;
#else
#include <cudaTypedefs.h>
#endif

    typedef enum CUHookSymbolEnum
    {
      CU_HOOK_GET_PROC_ADDRESS = 0,
      CU_HOOK_MEM_ALLOC_V2,
      CU_HOOK_MEM_FREE_V2,
      CU_HOOK_LAUNCH_KERNEL,
      CU_HOOK_LAUNCH_COOP_KERNEL,
      CU_HOOK_MEMCPY_ASYNC,
      CU_HOOK_MEMCPY2D_ASYNC_V2,
      CU_HOOK_MEMCPY3D_ASYNC_V2,
      CU_HOOK_MEMCPYD2H_ASYNC_V2,
      CU_HOOK_MEMCPYH2D_ASYNC_V2,
      CU_HOOK_EVENT_RECORD,
      CU_HOOK_EVENT_RECORD_FLAGS,
      CU_HOOK_EVENT_SYNC,
      CU_HOOK_STREAM_SYNC,
      CU_HOOK_CTX_SYNC,
      CU_HOOK_SYMBOLS,
    } CUHookSymbol;

// this map converts a enum to a string
#if(CUDA_VERSION < 11030)
    std::unordered_map<CUHookSymbol, const char *> cuhook_symbol2string = {
        {CU_HOOK_GET_PROC_ADDRESS, "cuGetProcAddress"},
        {CU_HOOK_MEM_ALLOC_V2, "cuMemAlloc_v2"},
        {CU_HOOK_MEM_FREE_V2, "cuMemFree_v2"},
        {CU_HOOK_LAUNCH_KERNEL, "cuLaunchKernel"},
        {CU_HOOK_LAUNCH_COOP_KERNEL, "cuLaunchCooperativeKernel"},
        {CU_HOOK_MEMCPY_ASYNC, "cuMemcpyAsync"},
        {CU_HOOK_MEMCPY2D_ASYNC_V2, "cuMemcpy2DAsync_v2"},
        {CU_HOOK_MEMCPY3D_ASYNC_V2, "cuMemcpy3DAsync_v2"},
        {CU_HOOK_MEMCPYD2H_ASYNC_V2, "cuMemcpyDtoHAsync_v2"},
        {CU_HOOK_MEMCPYH2D_ASYNC_V2, "cuMemcpyHtoDAsync_v2"},
        {CU_HOOK_EVENT_RECORD, "cuEventRecord"},
        {CU_HOOK_EVENT_RECORD_FLAGS, "cuEventRecordWithFlags"},
        {CU_HOOK_EVENT_SYNC, "cuEventSynchronize"},
        {CU_HOOK_STREAM_SYNC, "cuStreamSynchronize"},
        {CU_HOOK_CTX_SYNC, "cuCtxSynchronize"},
    };
#else
    std::unordered_map<CUHookSymbol, const char *> cuhook_symbol2string = {
        {CU_HOOK_GET_PROC_ADDRESS, "cuGetProcAddress"},
        {CU_HOOK_MEM_ALLOC_V2, "cuMemAlloc"},
        {CU_HOOK_MEM_FREE_V2, "cuMemFree"},
        {CU_HOOK_LAUNCH_KERNEL, "cuLaunchKernel"},
        {CU_HOOK_LAUNCH_COOP_KERNEL, "cuLaunchCooperativeKernel"},
        {CU_HOOK_MEMCPY_ASYNC, "cuMemcpyAsync"},
        {CU_HOOK_MEMCPY2D_ASYNC_V2, "cuMemcpy2DAsync"},
        {CU_HOOK_MEMCPY3D_ASYNC_V2, "cuMemcpy3DAsync"},
        {CU_HOOK_MEMCPYD2H_ASYNC_V2, "cuMemcpyDtoHAsync"},
        {CU_HOOK_MEMCPYH2D_ASYNC_V2, "cuMemcpyHtoDAsync"},
        {CU_HOOK_EVENT_RECORD, "cuEventRecord"},
        {CU_HOOK_EVENT_RECORD_FLAGS, "cuEventRecordWithFlags"},
        {CU_HOOK_EVENT_SYNC, "cuEventSynchronize"},
        {CU_HOOK_STREAM_SYNC, "cuStreamSynchronize"},
        {CU_HOOK_CTX_SYNC, "cuCtxSynchronize"},
    };
#endif

    // data structure that passed into callback functions
    struct CUHookStreamCallbackData {
      CUstream stream;
      CUevent event;
    };

    // callback function type
    typedef void (*CUHookCallbackFNPTR)(CUHookSymbol symbol, int version, void *data);

    // Main structure that gets initialized at library load time
    struct cuHookInfo {
      // callback function of each hooked CU function
      CUHookCallbackFNPTR callback_fnptr[CU_HOOK_SYMBOLS] = {nullptr};

      // wether debug is enabled or not
      bool debug_enabled = false;

      // stats info of each hooked CU function
      atomic<int> hooked_function_calls[CU_HOOK_SYMBOLS] = {atomic<int>(0)};

      // track the size of memory allocation
      size_t alloc_size = 0;

      cuHookInfo()
      {
        const char *envHookDebug;
        // Check environment for REALM_CUDAHOOK_DEBUG to facilitate debugging
        envHookDebug = getenv("REALM_CUDAHOOK_DEBUG");
        if(envHookDebug && envHookDebug[0] == '1') {
          debug_enabled = true;
          printf("[CUDAHOOK]: pid %6d >> CUDA HOOK Library loaded.\n", getpid());
        }
      }

      ~cuHookInfo()
      {
        if(debug_enabled) {
          int nb_hooked_calls = 0;
          for(int i = 0; i < CU_HOOK_SYMBOLS; i++) {
            nb_hooked_calls += hooked_function_calls[i].load();
          }
          // assert (hooked_function_calls[CU_HOOK_MEM_ALLOC_V2] ==
          // hooked_function_calls[CU_HOOK_MEM_FREE_V2]);
          printf(
              "[CUDAHOOK]: pid %6d >> CUDA HOOK Library unloaded, total calls %d, alloc "
              "size %lu\n",
              getpid(), nb_hooked_calls, alloc_size);
        }
      }
    };

    namespace ThreadLocal {
      static REALM_THREAD_LOCAL GPUProcessor *current_gpu_proc = nullptr;
      static REALM_THREAD_LOCAL std::unordered_map<
          CUstream, std::pair<CUHookSymbol, CUevent>> *cuhook_stream_status = nullptr;
      static REALM_THREAD_LOCAL int nb_hooked_functions_per_task = 0;
    }; // namespace ThreadLocal

    static bool fnptr_map_inited = false;
#if(CUDA_VERSION < 11030)
    static std::unordered_map<std::string, void *> fnptr_map;
#else
    static std::unordered_map<std::string, std::map<int, void *>> fnptr_map;
#endif

    static struct cuHookInfo cuhl;

    static inline void cudahook_print(const char *fmt, ...)
    {
      if(cuhl.debug_enabled) {
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
      }
    }

    static void *get_fnptr(const char *symbol, int cudaVersion, cuuint64_t flags);

    // We need to interpose dlsym since anyone using dlopen+dlsym to get the CUDA
    // driver symbols will bypass the hooking mechanism (this includes the CUDA
    // runtime). Its tricky though, since if we replace the real dlsym with ours, we
    // can't dlsym() the real dlsym. To get around that, call the 'private' libc
    // interface called __libc_dlsym to get the real dlsym.
    typedef void *(*fnDlsym)(void *, const char *);

    static void *real_dlsym(void *handle, const char *symbol)
    {
      static fnDlsym internal_dlsym =
          (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
      return (*internal_dlsym)(handle, symbol);
    }

#if(CUDA_VERSION < 12000)
    enum CUdriverProcAddressQueryResult
    {
      CU_GET_PROC_ADDRESS_SUCCESS = 0,
      CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
      CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
    };
#endif

    static CUresult CUDAAPI
    real_cuGetProcAddress(const char *symbol, void **hooked, int cudaVersion,
                          cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
    {
#ifdef REALM_CUDA_DYNAMIC_LOAD
      void *libcuda = dlopen("libcuda.so.1", RTLD_NOW);
      assert(libcuda != NULL);
#if(CUDA_VERSION >= 12000)
      static void *real_func = (void *)real_dlsym(libcuda, "cuGetProcAddress_v2");
#else
      static void *real_func = (void *)real_dlsym(libcuda, "cuGetProcAddress");
#endif
#else
#if(CUDA_VERSION >= 12000)
      static void *real_func = (void *)real_dlsym(RTLD_NEXT, "cuGetProcAddress_v2");
#else
      static void *real_func = (void *)real_dlsym(RTLD_NEXT, "cuGetProcAddress");
#endif
#endif
      assert(real_func != NULL);
      CUresult result = CUDA_SUCCESS;

#if(CUDA_VERSION >= 12000)
      result = ((CUresult CUDAAPI(*)(
          const char *symbol, void **hooked, int cudaVersion, cuuint64_t flags,
          CUdriverProcAddressQueryResult *symbolStatus))real_func)(
          symbol, hooked, cudaVersion, flags, symbolStatus);
      if(symbolStatus != nullptr) {
        assert(*symbolStatus == CU_GET_PROC_ADDRESS_SUCCESS);
      }
#else
      result = ((CUresult CUDAAPI(*)(const char *symbol, void **hooked, int cudaVersion,
                                     cuuint64_t flags))real_func)(symbol, hooked,
                                                                  cudaVersion, flags);
#endif
      return result;
    }

    // Interposed Functions

#if(CUDA_VERSION < 11030)
#define GENERATE_HOOKED_STREAM_FUNC(hook_symbol_enum, funcname, stream, event, params,   \
                                    ...)                                                 \
  template <typename PFN_TYPE, int cuda_version, int cuGetProcAddress_flags>             \
  static CUresult CUDAAPI funcname params                                                \
  {                                                                                      \
    static PFN_TYPE real_func = nullptr;                                                 \
    if(real_func == nullptr) {                                                           \
      std::string hook_symbol = cuhook_symbol2string[hook_symbol_enum];                  \
      if(cuGetProcAddress_flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) {      \
        hook_symbol += "_ptsz";                                                          \
      }                                                                                  \
      real_func =                                                                        \
          reinterpret_cast<PFN_TYPE>(real_dlsym(RTLD_NEXT, hook_symbol.c_str()));        \
    }                                                                                    \
    assert(real_func != nullptr);                                                        \
                                                                                         \
    if(cuhl.debug_enabled) {                                                             \
      cuhl.hooked_function_calls[hook_symbol_enum].fetch_add(1);                         \
    }                                                                                    \
    assert(cuhl.callback_fnptr[hook_symbol_enum] != nullptr);                            \
    CUHookStreamCallbackData cb_data = {stream, event};                                  \
    (cuhl.callback_fnptr[hook_symbol_enum])(hook_symbol_enum, cuda_version, &cb_data);   \
    CUresult result = real_func(__VA_ARGS__);                                            \
    return result;                                                                       \
  }
#else
#define GENERATE_HOOKED_STREAM_FUNC(hook_symbol_enum, funcname, stream, event, params,   \
                                    ...)                                                 \
  template <typename PFN_TYPE, int cuda_version, int cuGetProcAddress_flags>             \
  static CUresult CUDAAPI funcname params                                                \
  {                                                                                      \
    static PFN_TYPE real_func = nullptr;                                                 \
    if(real_func == nullptr) {                                                           \
      CUdriverProcAddressQueryResult symbolStatus =                                      \
          CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;                                          \
      CUresult PFN_result = real_cuGetProcAddress(                                       \
          cuhook_symbol2string[hook_symbol_enum], reinterpret_cast<void **>(&real_func), \
          cuda_version, cuGetProcAddress_flags, &symbolStatus);                          \
      assert(PFN_result == CUDA_SUCCESS);                                                \
    }                                                                                    \
    assert(real_func != nullptr);                                                        \
                                                                                         \
    if(cuhl.debug_enabled) {                                                             \
      cuhl.hooked_function_calls[hook_symbol_enum].fetch_add(1);                         \
    }                                                                                    \
    assert(cuhl.callback_fnptr[hook_symbol_enum] != nullptr);                            \
    CUHookStreamCallbackData cb_data = {stream, event};                                  \
    (cuhl.callback_fnptr[hook_symbol_enum])(hook_symbol_enum, cuda_version, &cb_data);   \
    CUresult result = real_func(__VA_ARGS__);                                            \
    return result;                                                                       \
  }
#endif

    template <typename PFN_TYPE, int cuda_version>
    static CUresult CUDAAPI hooked_cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
    {
      static PFN_TYPE real_func = nullptr;
      if(real_func == nullptr) {
#if(CUDA_VERSION < 11030)
        real_func = reinterpret_cast<PFN_TYPE>(real_dlsym(RTLD_NEXT, "cuMemAlloc_v2"));
#else
        CUdriverProcAddressQueryResult symbolStatus =
            CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        CUresult PFN_result = real_cuGetProcAddress(
            "cuMemAlloc", reinterpret_cast<void **>(&real_func), cuda_version,
            CU_GET_PROC_ADDRESS_DEFAULT, &symbolStatus);
        assert(PFN_result == CUDA_SUCCESS);
#endif
      }
      assert(real_func != nullptr);

      if(cuhl.debug_enabled) {
        cuhl.hooked_function_calls[CU_HOOK_MEM_ALLOC_V2].fetch_add(1);
        cuhl.alloc_size += bytesize;
      }
      CUresult result = real_func(dptr, bytesize);
      return result;
    }

    template <typename PFN_TYPE, int cuda_version>
    static CUresult CUDAAPI hooked_cuMemFree_v2(CUdeviceptr dptr)
    {
      static PFN_TYPE real_func = nullptr;
      if(real_func == nullptr) {
#if(CUDA_VERSION < 11030)
        real_func = reinterpret_cast<PFN_TYPE>(real_dlsym(RTLD_NEXT, "cuMemFree_v2"));
#else
        CUdriverProcAddressQueryResult symbolStatus =
            CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        CUresult PFN_result = real_cuGetProcAddress(
            "cuMemFree", reinterpret_cast<void **>(&real_func), cuda_version,
            CU_GET_PROC_ADDRESS_DEFAULT, &symbolStatus);
        assert(PFN_result == CUDA_SUCCESS);
#endif
      }
      assert(real_func != nullptr);

      if(cuhl.debug_enabled) {
        cuhl.hooked_function_calls[CU_HOOK_MEM_FREE_V2].fetch_add(1);
      }
      CUresult result = real_func(dptr);
      return result;
    }

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_LAUNCH_KERNEL, hooked_cuLaunchKernel, hStream,
                                nullptr,
                                (CUfunction f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ, unsigned int sharedMemBytes,
                                 CUstream hStream, void **kernelParams, void **extra),
                                f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                blockDimZ, sharedMemBytes, hStream, kernelParams, extra)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_LAUNCH_COOP_KERNEL,
                                hooked_cuLaunchCooperativeKernel, hStream, nullptr,
                                (CUfunction f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ, unsigned int sharedMemBytes,
                                 CUstream hStream, void **kernelParams),
                                f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                blockDimZ, sharedMemBytes, hStream, kernelParams)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_MEMCPY_ASYNC, hooked_cuMemcpyAsync, hStream,
                                nullptr,
                                (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                                 CUstream hStream),
                                dst, src, ByteCount, hStream)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_MEMCPY2D_ASYNC_V2, hooked_cuMemcpy2DAsync_v2,
                                hStream, nullptr,
                                (const CUDA_MEMCPY2D *pCopy, CUstream hStream), pCopy,
                                hStream)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_MEMCPY3D_ASYNC_V2, hooked_cuMemcpy3DAsync_v2,
                                hStream, nullptr,
                                (const CUDA_MEMCPY3D *pCopy, CUstream hStream), pCopy,
                                hStream)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_MEMCPYD2H_ASYNC_V2, hooked_cuMemcpyDtoHAsync_v2,
                                hStream, nullptr,
                                (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                 CUstream hStream),
                                dstHost, srcDevice, ByteCount, hStream)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_MEMCPYH2D_ASYNC_V2, hooked_cuMemcpyHtoDAsync_v2,
                                hStream, nullptr,
                                (CUdeviceptr dstDevice, const void *srcHost,
                                 size_t ByteCount, CUstream hStream),
                                dstDevice, srcHost, ByteCount, hStream)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_EVENT_RECORD, hooked_cuEventRecord, hStream,
                                hEvent, (CUevent hEvent, CUstream hStream), hEvent,
                                hStream)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_EVENT_RECORD_FLAGS, hooked_cuEventRecordWithFlags,
                                hStream, hEvent,
                                (CUevent hEvent, CUstream hStream, unsigned int flags),
                                hEvent, hStream, flags)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_EVENT_SYNC, hooked_cuEventSynchronize, nullptr,
                                hEvent, (CUevent hEvent), hEvent)

    GENERATE_HOOKED_STREAM_FUNC(CU_HOOK_STREAM_SYNC, hooked_cuStreamSynchronize, hStream,
                                nullptr, (CUstream hStream), hStream)

    template <typename PFN_TYPE, int cuda_version>
    static CUresult CUDAAPI hooked_cuCtxSynchronize(void)
    {
      static PFN_TYPE real_func = nullptr;
      if(real_func == nullptr) {
#if(CUDA_VERSION < 11030)
        real_func = reinterpret_cast<PFN_TYPE>(real_dlsym(RTLD_NEXT, "cuCtxSynchronize"));
#else
        CUdriverProcAddressQueryResult symbolStatus =
            CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        CUresult PFN_result = real_cuGetProcAddress("cuCtxSynchronize",
                                                    reinterpret_cast<void **>(&real_func),
                                                    cuda_version, 0, &symbolStatus);
        assert(PFN_result == CUDA_SUCCESS);
#endif
      }
      assert(real_func != nullptr);

      if(cuhl.debug_enabled) {
        cuhl.hooked_function_calls[CU_HOOK_CTX_SYNC].fetch_add(1);
      }
      assert(cuhl.callback_fnptr[CU_HOOK_CTX_SYNC] != nullptr);
      CUHookStreamCallbackData cb_data = {nullptr};
      (cuhl.callback_fnptr[CU_HOOK_CTX_SYNC])(CU_HOOK_CTX_SYNC, cuda_version, &cb_data);
      CUresult result = real_func();
      return result;
    }

    // template <typename PFN_TYPE, int cuda_version>
    // CUresult CUDAAPI hooked_cuStreamSynchronize(CUstream hStream)
    // {
    //   static PFN_TYPE real_func = nullptr;
    //   if (real_func == nullptr) {
    //     CUresult PFN_result = real_cuGetProcAddress("cuStreamSynchronize",
    //     reinterpret_cast<void**>(&real_func), cuda_version, 0); assert(PFN_result ==
    //     CUDA_SUCCESS);
    //   }
    //   assert(real_func != nullptr);

    //   if (cuhl.debug_enabled) {
    //     cuhl.hooked_function_calls[CU_HOOK_STREAM_SYNC]++;
    //   }
    //   if (cuhl.preHooks[CU_HOOK_STREAM_SYNC]) {
    //     ((CUresult CUDAAPI(*) (CUstream hStream, int version))
    //       cuhl.preHooks[CU_HOOK_STREAM_SYNC])(hStream, cuda_version);
    //   }
    //   CUresult result = real_func(hStream);
    //   if (cuhl.postHooks[CU_HOOK_STREAM_SYNC] && result == CUDA_SUCCESS) {
    //     ((CUresult CUDAAPI(*) (CUstream hStream, int version))
    //       cuhl.postHooks[CU_HOOK_STREAM_SYNC])(hStream, cuda_version);
    //   }
    //   return result;
    // }

    static CUresult CUDAAPI hooked_cuGetProcAddress_v11030(const char *symbol,
                                                           void **hooked, int cudaVersion,
                                                           cuuint64_t flags)
    {
      void *fnptr = get_fnptr(symbol, cudaVersion, flags);
      if(fnptr != nullptr) {
        *hooked = fnptr;
        return CUDA_SUCCESS;
      } else {
        // printf("cuGetProcAddress_v11030 %s\n", symbol);
        CUresult result =
            real_cuGetProcAddress(symbol, hooked, cudaVersion, flags, nullptr);
        return result;
      }
    }

    static CUresult CUDAAPI hooked_cuGetProcAddress_v12000(
        const char *symbol, void **hooked, int cudaVersion, cuuint64_t flags,
        CUdriverProcAddressQueryResult *symbolStatus)
    {
      void *fnptr = get_fnptr(symbol, cudaVersion, flags);
      if(fnptr != nullptr) {
        *hooked = fnptr;
        return CUDA_SUCCESS;
      } else {
        // printf("cuGetProcAddress_v12000 %s\n", symbol);
        CUresult result =
            real_cuGetProcAddress(symbol, hooked, cudaVersion, flags, symbolStatus);
        return result;
      }
    }

    static void init_fnptr(void)
    {
#if(CUDA_VERSION < 11030)
      fnptr_map = {
          {"cuMemAlloc_v2", (void *)hooked_cuMemAlloc_v2<PFN_cuMemAlloc_v3020, 3020>},
          {"cuMemFree_v2", (void *)hooked_cuMemFree_v2<PFN_cuMemFree_v3020, 3020>},
          {"cuLaunchKernel", (void *)hooked_cuLaunchKernel<PFN_cuLaunchKernel_v4000, 4000,
                                                           CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuLaunchKernel_ptsz",
           (void *)hooked_cuLaunchKernel<PFN_cuLaunchKernel_v7000_ptsz, 7000,
                                         CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuLaunchCooperativeKernel",
           (void *)hooked_cuLaunchCooperativeKernel<PFN_cuLaunchCooperativeKernel_v9000,
                                                    9000, CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuLaunchCooperativeKernel_ptsz",
           (void *)hooked_cuLaunchCooperativeKernel<
               PFN_cuLaunchCooperativeKernel_v9000_ptsz, 9000,
               CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuMemcpyAsync", (void *)hooked_cuMemcpyAsync<PFN_cuMemcpyAsync_v4000, 4000,
                                                         CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuMemcpyAsync_ptsz",
           (void *)hooked_cuMemcpyAsync<PFN_cuMemcpyAsync_v7000_ptsz, 7000,
                                        CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuMemcpy2DAsync_v2",
           (void *)hooked_cuMemcpy2DAsync_v2<PFN_cuMemcpy2DAsync_v3020, 3020,
                                             CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuMemcpy2DAsync_v2_ptsz",
           (void *)
               hooked_cuMemcpy2DAsync_v2<PFN_cuMemcpy2DAsync_v7000_ptsz, 7000,
                                         CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuMemcpy3DAsync_v2",
           (void *)hooked_cuMemcpy3DAsync_v2<PFN_cuMemcpy3DAsync_v3020, 3020,
                                             CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuMemcpy3DAsync_v2_ptsz",
           (void *)
               hooked_cuMemcpy3DAsync_v2<PFN_cuMemcpy3DAsync_v7000_ptsz, 7000,
                                         CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuMemcpyDtoHAsync_v2",
           (void *)hooked_cuMemcpyDtoHAsync_v2<PFN_cuMemcpyDtoHAsync_v3020, 3020,
                                               CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuMemcpyDtoHAsync_v2_ptsz",
           (void *)hooked_cuMemcpyDtoHAsync_v2<
               PFN_cuMemcpyDtoHAsync_v7000_ptsz, 7000,
               CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuMemcpyHtoDAsync_v2",
           (void *)hooked_cuMemcpyHtoDAsync_v2<PFN_cuMemcpyHtoDAsync_v3020, 3020,
                                               CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuMemcpyHtoDAsync_v2_ptsz",
           (void *)hooked_cuMemcpyHtoDAsync_v2<
               PFN_cuMemcpyHtoDAsync_v7000_ptsz, 7000,
               CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuEventRecord", (void *)hooked_cuEventRecord<PFN_cuEventRecord_v2000, 2000,
                                                         CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuEventRecord_ptsz",
           (void *)hooked_cuEventRecord<PFN_cuEventRecord_v7000_ptsz, 7000,
                                        CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuEventRecordWithFlags",
           (void *)hooked_cuEventRecordWithFlags<PFN_cuEventRecordWithFlags_v11010, 11010,
                                                 CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuEventRecordWithFlags_ptsz",
           (void *)hooked_cuEventRecordWithFlags<
               PFN_cuEventRecordWithFlags_v11010_ptsz, 11010,
               CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuEventSynchronize",
           (void *)hooked_cuEventSynchronize<PFN_cuEventSynchronize_v2000, 2000,
                                             CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuStreamSynchronize",
           (void *)hooked_cuStreamSynchronize<PFN_cuStreamSynchronize_v2000, 2000,
                                              CU_GET_PROC_ADDRESS_DEFAULT>},
          {"cuStreamSynchronize_ptsz",
           (void *)
               hooked_cuStreamSynchronize<PFN_cuStreamSynchronize_v7000_ptsz, 7000,
                                          CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>},
          {"cuCtxSynchronize",
           (void *)hooked_cuCtxSynchronize<PFN_cuCtxSynchronize_v2000, 2000>}};
#else
      fnptr_map = {
          {"cuGetProcAddress",
           {{11030, (void *)hooked_cuGetProcAddress_v11030},
            {12000, (void *)hooked_cuGetProcAddress_v12000}}},
          {"cuMemAlloc",
           {{3020, (void *)hooked_cuMemAlloc_v2<PFN_cuMemAlloc_v3020, 3020>}}},
          {"cuMemFree", {{3020, (void *)hooked_cuMemFree_v2<PFN_cuMemFree_v3020, 3020>}}},
          {"cuLaunchKernel",
           {{4000, (void *)hooked_cuLaunchKernel<PFN_cuLaunchKernel_v4000, 4000,
                                                 CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)
                 hooked_cuLaunchKernel<PFN_cuLaunchKernel_v7000_ptsz, 7000,
                                       CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuLaunchCooperativeKernel",
           {{9000,
             (void *)hooked_cuLaunchCooperativeKernel<PFN_cuLaunchCooperativeKernel_v9000,
                                                      9000, CU_GET_PROC_ADDRESS_DEFAULT>},
            {9000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)hooked_cuLaunchCooperativeKernel<
                 PFN_cuLaunchCooperativeKernel_v9000_ptsz, 9000,
                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuMemcpyAsync",
           {{4000, (void *)hooked_cuMemcpyAsync<PFN_cuMemcpyAsync_v4000, 4000,
                                                CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)
                 hooked_cuMemcpyAsync<PFN_cuMemcpyAsync_v7000_ptsz, 7000,
                                      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuMemcpy2DAsync",
           {{3020, (void *)hooked_cuMemcpy2DAsync_v2<PFN_cuMemcpy2DAsync_v3020, 3020,
                                                     CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)hooked_cuMemcpy2DAsync_v2<
                 PFN_cuMemcpy2DAsync_v7000_ptsz, 7000,
                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuMemcpy3DAsync",
           {{3020, (void *)hooked_cuMemcpy3DAsync_v2<PFN_cuMemcpy3DAsync_v3020, 3020,
                                                     CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)hooked_cuMemcpy3DAsync_v2<
                 PFN_cuMemcpy3DAsync_v7000_ptsz, 7000,
                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuMemcpyDtoHAsync",
           {{3020, (void *)hooked_cuMemcpyDtoHAsync_v2<PFN_cuMemcpyDtoHAsync_v3020, 3020,
                                                       CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)hooked_cuMemcpyDtoHAsync_v2<
                 PFN_cuMemcpyDtoHAsync_v7000_ptsz, 7000,
                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuMemcpyHtoDAsync",
           {{3020, (void *)hooked_cuMemcpyHtoDAsync_v2<PFN_cuMemcpyHtoDAsync_v3020, 3020,
                                                       CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)hooked_cuMemcpyHtoDAsync_v2<
                 PFN_cuMemcpyHtoDAsync_v7000_ptsz, 7000,
                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuEventRecord",
           {{2000, (void *)hooked_cuEventRecord<PFN_cuEventRecord_v2000, 2000,
                                                CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)
                 hooked_cuEventRecord<PFN_cuEventRecord_v7000_ptsz, 7000,
                                      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuEventRecordWithFlags",
           {{11010,
             (void *)hooked_cuEventRecordWithFlags<PFN_cuEventRecordWithFlags_v11010,
                                                   11010, CU_GET_PROC_ADDRESS_DEFAULT>},
            {11010 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)hooked_cuEventRecordWithFlags<
                 PFN_cuEventRecordWithFlags_v11010_ptsz, 11010,
                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuEventSynchronize",
           {{2000, (void *)hooked_cuEventSynchronize<PFN_cuEventSynchronize_v2000, 2000,
                                                     CU_GET_PROC_ADDRESS_DEFAULT>}}},
          {"cuStreamSynchronize",
           {{2000, (void *)hooked_cuStreamSynchronize<PFN_cuStreamSynchronize_v2000, 2000,
                                                      CU_GET_PROC_ADDRESS_DEFAULT>},
            {7000 + CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
             (void *)hooked_cuStreamSynchronize<
                 PFN_cuStreamSynchronize_v7000_ptsz, 7000,
                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>}}},
          {"cuCtxSynchronize",
           {{2000, (void *)hooked_cuCtxSynchronize<PFN_cuCtxSynchronize_v2000, 2000>}}}};
#endif
    }

    static void *get_fnptr(const char *symbol, int cuda_version, cuuint64_t flags)
    {
      if(fnptr_map_inited == false) {
        init_fnptr();
        fnptr_map_inited = true;
      }
      void *fnptr = nullptr;
#if(CUDA_VERSION < 11030)
      std::unordered_map<std::string, void *>::const_iterator fnptr_map_it =
          fnptr_map.find(symbol);
      if(fnptr_map_it != fnptr_map.cend()) {
        fnptr = fnptr_map_it->second;
      }
      return fnptr;
#else
      std::unordered_map<std::string, std::map<int, void *>>::const_iterator
          fnptr_map_it = fnptr_map.find(symbol);
      if(fnptr_map_it == fnptr_map.cend()) {
        return fnptr;
      } else {
        std::map<int, void *>::const_iterator version_fnptr_map_it =
            fnptr_map_it->second.find(cuda_version + flags);
        if(version_fnptr_map_it == fnptr_map_it->second.cend()) {
          // could not find it, let's get the first item if CU_GET_PROC_ADDRESS_DEFAULT,
          // otherwise, the last item
          if(fnptr_map_it->second.size() == 2) {
            cudahook_print("[CUDAHOOK]: Warning 2!, can not find %s, ver %d, flags %lu\n",
                           symbol, cuda_version, flags);
            assert(fnptr_map_it->second.begin()->first <
                   fnptr_map_it->second.rbegin()->first);
            if(flags == CU_GET_PROC_ADDRESS_DEFAULT) {
              fnptr = fnptr_map_it->second.begin()->second;
            } else {
              fnptr = fnptr_map_it->second.rbegin()->second;
            }
          } else {
            cudahook_print("[CUDAHOOK]: Warning 1!, can not find %s, ver %d, flags %lu\n",
                           symbol, cuda_version, flags);
            fnptr = fnptr_map_it->second.begin()->second;
          }
        } else {
          fnptr = version_fnptr_map_it->second;
        }
        cudahook_print("[CUDAHOOK]: redirect %s to %p, ver %d, flags %lu\n", symbol,
                       fnptr, cuda_version, flags);
        return fnptr;
      }
#endif
    }

    // this is the callback function of all hooked CUDA driver APIs
    // it will be called before each CUDA driver call. We keep track of cuda streams and
    // events
    static void cuhook_stream_callback(CUHookSymbol symbol, int version, void *data)
    {
      CUHookStreamCallbackData *cb_data =
          reinterpret_cast<CUHookStreamCallbackData *>(data);
      if(!ThreadLocal::current_gpu_proc) {
        cudahook_print(
            "[CUDAHOOK]: callback outside task, symbol %s, stream %p, event %p, "
            "version %d\n",
            cuhook_symbol2string[symbol], static_cast<void *>(cb_data->stream),
            static_cast<void *>(cb_data->stream), version);
        return;
      }

      cudahook_print("[CUDAHOOK]: callback symbol %s, stream %p, event %p, "
                     "version %d\n",
                     cuhook_symbol2string[symbol], static_cast<void *>(cb_data->stream),
                     static_cast<void *>(cb_data->stream), version);

      ThreadLocal::nb_hooked_functions_per_task++;
      switch(symbol) {
      case CU_HOOK_LAUNCH_KERNEL ... CU_HOOK_EVENT_RECORD_FLAGS:
      {
        ThreadLocal::cuhook_stream_status->operator[](cb_data->stream) =
            std::make_pair(symbol, cb_data->event);
        break;
      }
      case CU_HOOK_EVENT_SYNC:
      {
        std::unordered_map<CUstream, std::pair<CUHookSymbol, CUevent>>::iterator it;
        for(it = ThreadLocal::cuhook_stream_status->begin();
            it != ThreadLocal::cuhook_stream_status->end(); it++) {
          if(it->second.second == cb_data->event) {
            ThreadLocal::cuhook_stream_status->erase(it);
          }
        }
        break;
      }
      case CU_HOOK_STREAM_SYNC:
      {
        std::unordered_map<CUstream, std::pair<CUHookSymbol, CUevent>>::iterator it =
            ThreadLocal::cuhook_stream_status->find(cb_data->stream);
        if(it != ThreadLocal::cuhook_stream_status->end()) {
          ThreadLocal::cuhook_stream_status->erase(it);
        }
        break;
      }
      case CU_HOOK_CTX_SYNC:
      {
        ThreadLocal::cuhook_stream_status->clear();
        break;
      }
      default:
      {
        break;
      }
      }
    }

    // this function is called before existing a GPU task.
    static void cuhook_stream_sanity_check(CUstream current_task_stream)
    {
      if(ThreadLocal::cuhook_stream_status->size() == 0) {
        cudahook_print("[CUDAHOOK]: cuda stream sanity check: safe, nb_calls %d\n",
                       ThreadLocal::nb_hooked_functions_per_task);
      } else {
        // we remove streams that are realm's task streams
        std::unordered_map<CUstream, std::pair<CUHookSymbol, CUevent>>::iterator
            stream_it = ThreadLocal::cuhook_stream_status->find(current_task_stream);
        if(stream_it != ThreadLocal::cuhook_stream_status->end()) {
          ThreadLocal::cuhook_stream_status->erase(stream_it);
        }
        if(ThreadLocal::cuhook_stream_status->size() == 0) {
          cudahook_print("[CUDAHOOK]: cuda stream sanity check: safe, nb calls %d\n",
                         ThreadLocal::nb_hooked_functions_per_task);
        } else {
          printf("[CUDAHOOK]: cuda stream sanity check: unsafe, size %ld, nb calls %d\n",
                 ThreadLocal::cuhook_stream_status->size(),
                 ThreadLocal::nb_hooked_functions_per_task);
        }
      }
    }

  }; // namespace Cuda
};   // namespace Realm

// Driver API

extern "C" {

using namespace Realm::Cuda;

#if defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)
// undefine them here to avoid the redefinition error when compiling with --default-stream
// per-thread
#undef cuLaunchKernel
#undef cuMemcpyAsync
#undef cuMemcpy2DAsync_v2
#undef cuMemcpy3DAsync_v2
#undef cuMemcpyDtoHAsync_v2
#undef cuMemcpyHtoDAsync_v2
#undef cuEventRecord
#undef cuEventRecordWithFlags
#undef cuStreamSynchronize
#undef cuGetProcAddress
#endif

REALM_PUBLIC_API CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
  return hooked_cuMemAlloc_v2<PFN_cuMemAlloc_v3020, 3020>(dptr, bytesize);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr)
{
  return hooked_cuMemFree_v2<PFN_cuMemFree_v3020, 3020>(dptr);
}

REALM_PUBLIC_API CUresult CUDAAPI cuLaunchKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
{
  return hooked_cuLaunchKernel<PFN_cuLaunchKernel_v4000, 4000,
                               CU_GET_PROC_ADDRESS_DEFAULT>(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
      hStream, kernelParams, extra);
}

REALM_PUBLIC_API CUresult CUDAAPI cuLaunchKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
{
  return hooked_cuLaunchKernel<PFN_cuLaunchKernel_v7000_ptsz, 7000,
                               CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
      hStream, kernelParams, extra);
}

REALM_PUBLIC_API CUresult CUDAAPI cuLaunchCooperativeKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)
{
  return hooked_cuLaunchCooperativeKernel<PFN_cuLaunchCooperativeKernel_v9000, 9000,
                                          CU_GET_PROC_ADDRESS_DEFAULT>(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
      hStream, kernelParams);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                                                size_t ByteCount, CUstream hStream)
{
  return hooked_cuMemcpyAsync<PFN_cuMemcpyAsync_v4000, 4000, CU_GET_PROC_ADDRESS_DEFAULT>(
      dst, src, ByteCount, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                                     size_t ByteCount, CUstream hStream)
{
  return hooked_cuMemcpyAsync<PFN_cuMemcpyAsync_v7000_ptsz, 7000,
                              CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      dst, src, ByteCount, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                                     CUstream hStream)
{
  return hooked_cuMemcpy2DAsync_v2<PFN_cuMemcpy2DAsync_v3020, 3020,
                                   CU_GET_PROC_ADDRESS_DEFAULT>(pCopy, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy,
                                                          CUstream hStream)
{
  return hooked_cuMemcpy2DAsync_v2<PFN_cuMemcpy2DAsync_v7000_ptsz, 7000,
                                   CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      pCopy, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                                     CUstream hStream)
{
  return hooked_cuMemcpy3DAsync_v2<PFN_cuMemcpy3DAsync_v3020, 3020,
                                   CU_GET_PROC_ADDRESS_DEFAULT>(pCopy, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy,
                                                          CUstream hStream)
{
  return hooked_cuMemcpy3DAsync_v2<PFN_cuMemcpy3DAsync_v7000_ptsz, 7000,
                                   CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      pCopy, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost,
                                                       CUdeviceptr srcDevice,
                                                       size_t ByteCount, CUstream hStream)
{
  return hooked_cuMemcpyDtoHAsync_v2<PFN_cuMemcpyDtoHAsync_v3020, 3020,
                                     CU_GET_PROC_ADDRESS_DEFAULT>(dstHost, srcDevice,
                                                                  ByteCount, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void *dstHost,
                                                            CUdeviceptr srcDevice,
                                                            size_t ByteCount,
                                                            CUstream hStream)
{
  return hooked_cuMemcpyDtoHAsync_v2<PFN_cuMemcpyDtoHAsync_v7000_ptsz, 7000,
                                     CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      dstHost, srcDevice, ByteCount, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                                       const void *srcHost,
                                                       size_t ByteCount, CUstream hStream)
{
  return hooked_cuMemcpyHtoDAsync_v2<PFN_cuMemcpyHtoDAsync_v3020, 3020,
                                     CU_GET_PROC_ADDRESS_DEFAULT>(dstDevice, srcHost,
                                                                  ByteCount, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                            const void *srcHost,
                                                            size_t ByteCount,
                                                            CUstream hStream)
{
  return hooked_cuMemcpyHtoDAsync_v2<PFN_cuMemcpyHtoDAsync_v7000_ptsz, 7000,
                                     CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      dstDevice, srcHost, ByteCount, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream)
{
  return hooked_cuEventRecord<PFN_cuEventRecord_v2000, 2000, CU_GET_PROC_ADDRESS_DEFAULT>(
      hEvent, hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuEventRecord_ptsz(CUevent hEvent, CUstream hStream)
{
  return hooked_cuEventRecord<PFN_cuEventRecord_v7000_ptsz, 7000,
                              CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(hEvent,
                                                                             hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuEventRecordWithFlags(CUevent hEvent, CUstream hStream,
                                                         unsigned int flags)
{
  return hooked_cuEventRecordWithFlags<PFN_cuEventRecordWithFlags_v11010, 11010,
                                       CU_GET_PROC_ADDRESS_DEFAULT>(hEvent, hStream,
                                                                    flags);
}

REALM_PUBLIC_API CUresult CUDAAPI cuEventRecordWithFlags_ptsz(CUevent hEvent,
                                                              CUstream hStream,
                                                              unsigned int flags)
{
  return hooked_cuEventRecordWithFlags<PFN_cuEventRecordWithFlags_v11010_ptsz, 11010,
                                       CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      hEvent, hStream, flags);
}

REALM_PUBLIC_API CUresult CUDAAPI cuEventSynchronize(CUevent hEvent)
{
  return hooked_cuEventSynchronize<PFN_cuEventSynchronize_v2000, 2000,
                                   CU_GET_PROC_ADDRESS_DEFAULT>(hEvent);
}

REALM_PUBLIC_API CUresult CUDAAPI cuStreamSynchronize(CUstream hStream)
{
  return hooked_cuStreamSynchronize<PFN_cuStreamSynchronize_v2000, 2000,
                                    CU_GET_PROC_ADDRESS_DEFAULT>(hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream hStream)
{
  return hooked_cuStreamSynchronize<PFN_cuStreamSynchronize_v7000_ptsz, 7000,
                                    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM>(
      hStream);
}

REALM_PUBLIC_API CUresult CUDAAPI cuCtxSynchronize(void)
{
  return hooked_cuCtxSynchronize<PFN_cuCtxSynchronize_v2000, 2000>();
}

// After CUDA 11.3, cuGetProcAddress is used to get CUDA driver symbols, so we need to
// intercept it, and redirect the call to our intercepted functions.
#if(CUDA_VERSION >= 12000)

REALM_PUBLIC_API CUresult CUDAAPI
cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                 CUdriverProcAddressQueryResult *symbolStatus)
{
  return hooked_cuGetProcAddress_v12000(symbol, pfn, cudaVersion, flags, symbolStatus);
}

#else

REALM_PUBLIC_API CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn,
                                                   int cudaVersion, cuuint64_t flags)
{
  return hooked_cuGetProcAddress_v11030(symbol, pfn, cudaVersion, flags);
}

#endif

// Exposed API to cuda_module.cc

REALM_PUBLIC_API void cuhook_register_callback(void)
{
  for(int symbol = CU_HOOK_LAUNCH_KERNEL; symbol < CU_HOOK_SYMBOLS; symbol++) {
    cuhl.callback_fnptr[symbol] = cuhook_stream_callback;
  }
}

REALM_PUBLIC_API void cuhook_start_task(Realm::Cuda::GPUProcessor *gpu_proc)
{
  assert(ThreadLocal::cuhook_stream_status == nullptr);
  ThreadLocal::cuhook_stream_status =
      new std::unordered_map<CUstream, std::pair<CUHookSymbol, CUevent>>();
  ThreadLocal::nb_hooked_functions_per_task = 0;
  ThreadLocal::current_gpu_proc = gpu_proc;
}

REALM_PUBLIC_API void cuhook_end_task(CUstream current_task_stream)
{
  cuhook_stream_sanity_check(current_task_stream);
  ThreadLocal::current_gpu_proc = nullptr;
  delete ThreadLocal::cuhook_stream_status;
  ThreadLocal::cuhook_stream_status = nullptr;
}

REALM_PUBLIC_API void *dlsym(void *__restrict handle, const char *__restrict symbol)
{
  // Early out if not a CUDA driver symbol
  if(strncmp(symbol, "cu", 2) != 0) {
    return (real_dlsym(handle, symbol));
  }

  // we call dlsym on cuhook_register_callback/cuhook_start/end_task with the NULL handle,
  // so if LD_PRELOAD is not set the dlsym returns a null pointer.
  if((strcmp(symbol, "cuhook_register_callback") == 0)) {
    return (void *)cuhook_register_callback;
  } else if((strcmp(symbol, "cuhook_start_task") == 0)) {
    return (void *)cuhook_start_task;
  } else if((strcmp(symbol, "cuhook_end_task") == 0)) {
    return (void *)cuhook_end_task;
  }

  cudahook_print("[CUDAHOOK]: dlsym %s\n", symbol);

#if(CUDA_VERSION < 11030)
  void *fnptr = get_fnptr(symbol, 0, 0);
  if(fnptr != nullptr) {
    return fnptr;
  }
#else
#if(CUDA_VERSION >= 12000)
  if((strcmp(symbol, "cuGetProcAddress_v2") == 0) ||
     (strcmp(symbol, "cuGetProcAddress_v2_ptsz") == 0)) {
    CUresult (*cuGetProcAddress_ptr)(const char *symbol, void **hooked, int cudaVersion,
                                     cuuint64_t flags,
                                     CUdriverProcAddressQueryResult *symbolStatus);
    cuGetProcAddress_ptr = hooked_cuGetProcAddress_v12000;
#else
  if((strcmp(symbol, "cuGetProcAddress") == 0) ||
     (strcmp(symbol, "cuGetProcAddress_ptsz") == 0)) {
    CUresult (*cuGetProcAddress_ptr)(const char *symbol, void **hooked, int cudaVersion,
                                     cuuint64_t flags);
    cuGetProcAddress_ptr = hooked_cuGetProcAddress_v11030;
#endif
    return (void *)(cuGetProcAddress_ptr);
  }
#endif

  return (real_dlsym(handle, symbol));
}
};
