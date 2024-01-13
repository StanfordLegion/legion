/* Copyright 2023 Stanford University, NVIDIA Corporation
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

// little helper utilities for Realm code
// none of this is Realm-specific, but it's put in the Realm namespace to
//  reduce the chance of conflicts

#include "realm/utils.h"
#include "realm/logging.h"

#include <cstring> // strerror_*
#include <type_traits> // std::is_same

// on an x86 system with SSE4.2, we can use the builtin CRC32(C) instruction
#ifdef __SSE4_2__
#include <smmintrin.h>
#endif

#if defined(REALM_ON_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#elif defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#ifndef MSG_CMSG_CLOEXEC
#define MSG_CMSG_CLOEXEC 0
#endif

namespace Realm {

  Logger log_mailbox("mailbox");

  // dummy variable used to make sure utils.cc ends up in binary
  int force_utils_cc_linkage;

  // helper tables for fallback software computation
  static uint32_t crc32c_table[256] = {
    0x00000000,0xf26b8303,0xe13b70f7,0x1350f3f4,0xc79a971f,0x35f1141c,0x26a1e7e8,0xd4ca64eb,
    0x8ad958cf,0x78b2dbcc,0x6be22838,0x9989ab3b,0x4d43cfd0,0xbf284cd3,0xac78bf27,0x5e133c24,
    0x105ec76f,0xe235446c,0xf165b798,0x030e349b,0xd7c45070,0x25afd373,0x36ff2087,0xc494a384,
    0x9a879fa0,0x68ec1ca3,0x7bbcef57,0x89d76c54,0x5d1d08bf,0xaf768bbc,0xbc267848,0x4e4dfb4b,
    0x20bd8ede,0xd2d60ddd,0xc186fe29,0x33ed7d2a,0xe72719c1,0x154c9ac2,0x061c6936,0xf477ea35,
    0xaa64d611,0x580f5512,0x4b5fa6e6,0xb93425e5,0x6dfe410e,0x9f95c20d,0x8cc531f9,0x7eaeb2fa,
    0x30e349b1,0xc288cab2,0xd1d83946,0x23b3ba45,0xf779deae,0x05125dad,0x1642ae59,0xe4292d5a,
    0xba3a117e,0x4851927d,0x5b016189,0xa96ae28a,0x7da08661,0x8fcb0562,0x9c9bf696,0x6ef07595,
    0x417b1dbc,0xb3109ebf,0xa0406d4b,0x522bee48,0x86e18aa3,0x748a09a0,0x67dafa54,0x95b17957,
    0xcba24573,0x39c9c670,0x2a993584,0xd8f2b687,0x0c38d26c,0xfe53516f,0xed03a29b,0x1f682198,
    0x5125dad3,0xa34e59d0,0xb01eaa24,0x42752927,0x96bf4dcc,0x64d4cecf,0x77843d3b,0x85efbe38,
    0xdbfc821c,0x2997011f,0x3ac7f2eb,0xc8ac71e8,0x1c661503,0xee0d9600,0xfd5d65f4,0x0f36e6f7,
    0x61c69362,0x93ad1061,0x80fde395,0x72966096,0xa65c047d,0x5437877e,0x4767748a,0xb50cf789,
    0xeb1fcbad,0x197448ae,0x0a24bb5a,0xf84f3859,0x2c855cb2,0xdeeedfb1,0xcdbe2c45,0x3fd5af46,
    0x7198540d,0x83f3d70e,0x90a324fa,0x62c8a7f9,0xb602c312,0x44694011,0x5739b3e5,0xa55230e6,
    0xfb410cc2,0x092a8fc1,0x1a7a7c35,0xe811ff36,0x3cdb9bdd,0xceb018de,0xdde0eb2a,0x2f8b6829,
    0x82f63b78,0x709db87b,0x63cd4b8f,0x91a6c88c,0x456cac67,0xb7072f64,0xa457dc90,0x563c5f93,
    0x082f63b7,0xfa44e0b4,0xe9141340,0x1b7f9043,0xcfb5f4a8,0x3dde77ab,0x2e8e845f,0xdce5075c,
    0x92a8fc17,0x60c37f14,0x73938ce0,0x81f80fe3,0x55326b08,0xa759e80b,0xb4091bff,0x466298fc,
    0x1871a4d8,0xea1a27db,0xf94ad42f,0x0b21572c,0xdfeb33c7,0x2d80b0c4,0x3ed04330,0xccbbc033,
    0xa24bb5a6,0x502036a5,0x4370c551,0xb11b4652,0x65d122b9,0x97baa1ba,0x84ea524e,0x7681d14d,
    0x2892ed69,0xdaf96e6a,0xc9a99d9e,0x3bc21e9d,0xef087a76,0x1d63f975,0x0e330a81,0xfc588982,
    0xb21572c9,0x407ef1ca,0x532e023e,0xa145813d,0x758fe5d6,0x87e466d5,0x94b49521,0x66df1622,
    0x38cc2a06,0xcaa7a905,0xd9f75af1,0x2b9cd9f2,0xff56bd19,0x0d3d3e1a,0x1e6dcdee,0xec064eed,
    0xc38d26c4,0x31e6a5c7,0x22b65633,0xd0ddd530,0x0417b1db,0xf67c32d8,0xe52cc12c,0x1747422f,
    0x49547e0b,0xbb3ffd08,0xa86f0efc,0x5a048dff,0x8ecee914,0x7ca56a17,0x6ff599e3,0x9d9e1ae0,
    0xd3d3e1ab,0x21b862a8,0x32e8915c,0xc083125f,0x144976b4,0xe622f5b7,0xf5720643,0x07198540,
    0x590ab964,0xab613a67,0xb831c993,0x4a5a4a90,0x9e902e7b,0x6cfbad78,0x7fab5e8c,0x8dc0dd8f,
    0xe330a81a,0x115b2b19,0x020bd8ed,0xf0605bee,0x24aa3f05,0xd6c1bc06,0xc5914ff2,0x37faccf1,
    0x69e9f0d5,0x9b8273d6,0x88d28022,0x7ab90321,0xae7367ca,0x5c18e4c9,0x4f48173d,0xbd23943e,
    0xf36e6f75,0x0105ec76,0x12551f82,0xe03e9c81,0x34f4f86a,0xc69f7b69,0xd5cf889d,0x27a40b9e,
    0x79b737ba,0x8bdcb4b9,0x988c474d,0x6ae7c44e,0xbe2da0a5,0x4c4623a6,0x5f16d052,0xad7d5351
  };

  // accumulates a crc32c checksum
  //   initialization (traditionally to 0xFFFFFFFF) and finalization (by
  //   inverting) the accumulator is left to the caller
  uint32_t crc32c_accumulate(uint32_t accum_in, const void *data, size_t len)
  {
#ifdef __SSE4_2__
    uint32_t accum = accum_in;
    uintptr_t ptr = reinterpret_cast<uintptr_t>(data);
    size_t left = len;
    // handle misalignment, but prefer stepping in 32-bit chunks
    while((ptr & 3) && left) {
      accum = _mm_crc32_u8(accum, *reinterpret_cast<const uint8_t *>(ptr));
      ptr++;  left--;
    }
    while(left >= 4) {
      accum = _mm_crc32_u32(accum, *reinterpret_cast<const uint32_t *>(ptr));
      ptr+=4;  left-=4;
    }
    while(left) {
      accum = _mm_crc32_u8(accum, *reinterpret_cast<const uint8_t *>(ptr));
      ptr++;  left--;
    }

    // double-check with SW implementation
    uint32_t accum_sw = accum_in;
    for(size_t i = 0; i < len; i++) {
      uint8_t idx = (accum_sw & 0xff) ^ static_cast<const uint8_t *>(data)[i];
      accum_sw = (accum_sw >> 8) ^ crc32c_table[idx];
    }
    assert(accum == accum_sw);
    return accum;
#else
    // pure software fallback
    uint32_t accum = accum_in;
    for(size_t i = 0; i < len; i++) {
      uint8_t idx = (accum & 0xff) ^ static_cast<const uint8_t *>(data)[i];
      accum = (accum >> 8) ^ crc32c_table[idx];
    }
    return accum;
#endif
  }

#ifndef REALM_ON_WINDOWS
  template<typename T>
  struct ErrorHelper {
  public:
    static inline const char* process_error_message(T result, char *buffer)
    {
      // this is the version of strerror_r that returns an int so make
      // sure that it is not zero and then return the buffer
      assert(result == 0);
      return buffer;
    }
  };

  template<>
  struct ErrorHelper<char*> {
  public:
    static inline const char* process_error_message(char *result, char *buffer)
    {
      // this is the version of strerror_r that returns a string so use
      // that if it is not null
      return (result == nullptr) ? buffer : result;
    }
  };
#endif
  
  // allocation for thread-local error buffer for reporting error messages
  namespace ThreadLocal {
    REALM_THREAD_LOCAL char error_buffer[REALM_ERROR_BUFFER_SIZE];
  }

  const char* realm_strerror(int err)
  {
#ifdef REALM_ON_WINDOWS
    int result = strerror_s<REALM_ERROR_BUFFER_SIZE>(ThreadLocal::error_buffer, err);
    assert(result == 0);
    return ThreadLocal::error_buffer;
#else
    // Deal with the fact that strerror_r has two different possible
    // return types on different systems, call the right one based
    // on the return type and get the result
    auto result = strerror_r(err, ThreadLocal::error_buffer, REALM_ERROR_BUFFER_SIZE);
    // Return types should either be int or char*
    static_assert(
      std::is_same<decltype(result),int>::value ||
      std::is_same<decltype(result),char*>::value,
      "Unknown strerror_r return type");
    return ErrorHelper<decltype(result)>::process_error_message(result, ThreadLocal::error_buffer);
#endif
  }

  unsigned ctz(uint64_t v)
  {
#ifdef REALM_ON_WINDOWS
    unsigned long index;
#ifdef _WIN64
    if(_BitScanForward64(&index, v))
      return index;
#else
    unsigned v_lo = v;
    unsigned v_hi = v >> 32;
    if(_BitScanForward(&index, v_lo))
      return index;
    else if(_BitScanForward(&index, v_hi))
      return index + 32;
#endif
    else
      return 0;
#else
    return __builtin_ctzll(v);
#endif
  }
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
  static const char MAILBOX_PREFIX[] = "@realm_uds.";
#endif

  OsHandle ipc_mailbox_create(const std::string &name)
  {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, MAILBOX_PREFIX, sizeof(addr.sun_path) - 1);
    std::strncat(addr.sun_path, name.c_str(),
                 sizeof(addr.sun_path) - sizeof(MAILBOX_PREFIX));
    // Use the abstract namespace so we don't need to unlink the file system
    // https://man7.org/linux/man-pages/man7/unix.7.html
    addr.sun_path[0] = 0;
    // Use a DGRAM socket instead of a stream socket in order to create a connection-less,
    // bidirectional socket that mimics a mailbox
    OsHandle fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if(fd < 0) {
      log_mailbox.error("Failed to create socket! errno: %d\n", errno);
      return Realm::INVALID_OS_HANDLE;
    }
    if(bind(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0) {
      log_mailbox.error("Failed to bind socket! errno: %d\n", errno);
      close(fd);
      return Realm::INVALID_OS_HANDLE;
    }
    return fd;
#else
    // TODO: Add support for windows
    return Realm::INVALID_OS_HANDLE;
#endif
  }

  bool ipc_mailbox_send(OsHandle mailbox, const std::string &to,
                        const std::vector<OsHandle> &handles, const void *data,
                        size_t data_sz)
  {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
    struct msghdr msg;
    struct sockaddr_un addr;
    struct iovec in_band_msg;
    std::vector<char> control_buffer(CMSG_SPACE(handles.size() * sizeof(handles[0])), 0);

    memset(&addr, 0, sizeof(addr));
    memset(&in_band_msg, 0, sizeof(in_band_msg));
    memset(&msg, 0, sizeof(msg));

    std::strncpy(addr.sun_path, MAILBOX_PREFIX, sizeof(addr.sun_path) - 1);
    std::strncat(addr.sun_path, to.c_str(),
                 sizeof(addr.sun_path) - sizeof(MAILBOX_PREFIX));
    // Use the abstract namespace so we don't need to unlink the file system
    // https://man7.org/linux/man-pages/man7/unix.7.html
    addr.sun_path[0] = 0;
    addr.sun_family = AF_UNIX;

    msg.msg_name = &addr;
    msg.msg_namelen = sizeof(addr);

    if(data_sz > 0) {
      in_band_msg.iov_base = const_cast<void *>(data);
      in_band_msg.iov_len = data_sz;
      msg.msg_iov = &in_band_msg;
      msg.msg_iovlen = 1;
    }

    if(handles.size() > 0) {
      struct cmsghdr *cmsg = nullptr;
      msg.msg_control = control_buffer.data();
      msg.msg_controllen = control_buffer.size();

      cmsg = CMSG_FIRSTHDR(&msg);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;
      cmsg->cmsg_len = CMSG_LEN(sizeof(handles[0]) * handles.size());
      memcpy(CMSG_DATA(cmsg), handles.data(), handles.size() * sizeof(handles[0]));
    }

    if(sendmsg(mailbox, &msg, 0) > 0) {
      return true;
    }
    log_mailbox.error("Failed to send message: %d\n", errno);
#endif
    // TODO: Add support for windows
    return false;
  }

  bool ipc_mailbox_recv(OsHandle mailbox, const std::string &from,
                        std::vector<OsHandle> &handles, void *data, size_t &data_sz,
                        size_t max_data_sz)
  {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
    struct msghdr msg;
    struct sockaddr_un addr;
    struct cmsghdr *cmsg = nullptr;
    struct iovec in_band_msg;
    const size_t MAX_HANDLES = 1024; // TODO: implement MSG_PEEK
    std::vector<char> control_buffer(CMSG_SPACE(MAX_HANDLES * sizeof(handles[0])));

    memset(&addr, 0, sizeof(addr));
    memset(&in_band_msg, 0, sizeof(in_band_msg));
    memset(&msg, 0, sizeof(msg));

    std::strncpy(addr.sun_path, MAILBOX_PREFIX, sizeof(addr.sun_path) - 1);
    std::strncat(addr.sun_path, from.c_str(),
                 sizeof(addr.sun_path) - sizeof(MAILBOX_PREFIX));
    // Use the abstract namespace so we don't need to unlink the file system
    // https://man7.org/linux/man-pages/man7/unix.7.html
    addr.sun_path[0] = 0;
    addr.sun_family = AF_UNIX;

    in_band_msg.iov_base = data;
    in_band_msg.iov_len = max_data_sz;
    msg.msg_iov = &in_band_msg;
    msg.msg_iovlen = 1;

    msg.msg_control = control_buffer.data();
    msg.msg_controllen = control_buffer.size();
    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(MAX_HANDLES * sizeof(handles[0]));

    // Make sure any file descriptors are closed if we fork+exec
    ssize_t bytes = recvmsg(mailbox, &msg, MSG_CMSG_CLOEXEC);
    if(bytes < 0) {
      log_mailbox.error("Failed to recv message: %d", errno);
      return false;
    }

    // Make sure we got the entire message
    if((msg.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
      log_mailbox.error("Failed to recv the entire message!");
      return false;
    }

    if(msg.msg_controllen > 0) {
      const size_t clen = cmsg->cmsg_len - CMSG_LEN(0);
      handles.resize(clen / sizeof(handles[0]));
      memcpy(handles.data(), CMSG_DATA(cmsg), clen);
    } else {
      handles.clear();
    }
    data_sz = static_cast<size_t>(bytes);
    return true;
#else
    // TODO: Add support for windows
    return false;
#endif
  }

  void close_handle(OsHandle handle)
  {
    if(handle != Realm::INVALID_OS_HANDLE) {
#if defined(REALM_ON_WINDOWS)
      CloseHandle(handle);
#elif defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
      close(handle);
#endif
    }
  }
};
