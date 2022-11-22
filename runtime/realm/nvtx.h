/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#ifndef NVTX_H
#define NVTX_H

#include "realm/realm_config.h"

#include <map>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <vector>

namespace Realm {

  struct NvtxARGB {
    constexpr NvtxARGB(uint8_t red_, uint8_t green_, uint8_t blue_,
                       uint8_t alpha_ = 0xFF) noexcept
      : red{red_}
      , green{green_}
      , blue{blue_}
      , alpha{alpha_}
    {}
    constexpr uint32_t to_uint(void)
    {
      return uint32_t{alpha} << 24 | uint32_t{red} << 16 | uint32_t{green} << 8 |
             uint32_t{blue};
    }
    uint8_t const red{};
    uint8_t const green{};
    uint8_t const blue{};
    uint8_t const alpha{};
  };

  enum nvtx_color : uint32_t
  {
    white = NvtxARGB(255, 255, 255).to_uint(),
    red = NvtxARGB(255, 0, 0).to_uint(),
    green = NvtxARGB(0, 255, 0).to_uint(),
    blue = NvtxARGB(0, 0, 255).to_uint(),
    purple = NvtxARGB(128, 0, 128).to_uint(),
    lawn_green = NvtxARGB(124, 252, 0).to_uint(),
    cyan = NvtxARGB(0, 255, 255).to_uint(),
    maroon = NvtxARGB(128, 0, 0).to_uint(),
    navy = NvtxARGB(0, 0, 128).to_uint(),
    magenta = NvtxARGB(255, 0, 255).to_uint(),
    yellow = NvtxARGB(255, 255, 0).to_uint(),
    gray = NvtxARGB(128, 128, 128).to_uint(),
    teal = NvtxARGB(0, 128, 128).to_uint(),
    olive = NvtxARGB(128, 128, 0).to_uint(),
  };

  struct NvtxCategory {
    NvtxCategory(const std::string &category_name, uint32_t category_id, uint32_t color);
    const std::string name;
    nvtxEventAttributes_t nvtx_event;
  };

  struct nvtxScopedRange {
    nvtxScopedRange(NvtxCategory *category, char const *message, int32_t payload);
    nvtxScopedRange(const std::string &name, char const *message, int32_t payload);
    ~nvtxScopedRange();
  };

  static constexpr uint32_t nvtx_proc_starting_category_id = 1000;

  // called by each kernel thread to init thread local variables.
  void init_nvtx_thread(const char *thread_name);

  // called by RuntimeImpl::configure_from_command_line from the main thread to init nvtx
  //   and its thread local variables.
  void init_nvtx(std::vector<std::string> &nvtx_modules);

  // called by each kernel thread to delete thread local variables.
  void finalize_nvtx_thread(void);

  // called by RuntimeImpl::wait_for_shutdown from the main thread to finalize nvtx
  //   and delete its thread local variables.
  void finalize_nvtx(void);

  // TODO(@Wei Wu): template it wih type T for payload
  void nvtx_range_push(NvtxCategory *category, const char *message,
                       uint32_t color = nvtx_color::white, int32_t payload = 0);

  void nvtx_range_push(const std::string &name, const char *message,
                       uint32_t color = nvtx_color::white, int32_t payload = 0);

  void nvtx_range_pop(void);

  // TODO(@Wei Wu): template it wih type T for payload
  nvtxRangeId_t nvtx_range_start(NvtxCategory *category, const char *message,
                                 uint32_t color = nvtx_color::white, int32_t payload = 0);
  nvtxRangeId_t nvtx_range_start(const std::string &name, const char *message,
                                 uint32_t color = nvtx_color::white, int32_t payload = 0);

  void nvtx_range_end(nvtxRangeId_t id);

  // TODO(@Wei Wu): template it wih type T for payload
  void nvtx_mark(NvtxCategory *category, const char *message,
                 uint32_t color = nvtx_color::white, int32_t payload = 0);
  void nvtx_mark(const std::string &name, const char *message,
                 uint32_t color = nvtx_color::white, int32_t payload = 0);

}; // namespace Realm

#endif // NVTX_H