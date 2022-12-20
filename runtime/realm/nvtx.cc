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

#include "realm/nvtx.h"

#include <assert.h>
#include <iostream>
#ifdef REALM_ON_WINDOWS
#include <processthreadsapi.h>
#else
#include <unistd.h>
#endif

namespace Realm {

  struct realm_nvtx_domain {
    static constexpr char const *name{"Realm"};
  };

  struct nvtx_category_id_color {
    uint32_t id;
    uint32_t color;
  };

  static std::map<std::string, nvtx_category_id_color> nvtx_categories_predefined = {
      {"amsg", {1, nvtx_color::red}},
      {"bgwork", {2, nvtx_color::blue}},
#ifdef REALM_USE_CUDA
      {"cuda", {100, nvtx_color::green}},
#endif
#ifdef REALM_USE_HIP
      {"hip", {101, nvtx_color::purple}},
#endif
#ifdef REALM_USE_GASNET1
      {"gasnet1", {102, nvtx_color::lawn_green}},
#endif
#ifdef REALM_USE_GASNETEX
      {"gasnetex", {103, nvtx_color::cyan}},
#endif
#ifdef REALM_USE_MPI
      {"mpi", {104, nvtx_color::maroon}},
#endif
#ifdef REALM_USE_OPENMP
      {"openmp", {105, nvtx_color::navy}},
#endif
#ifdef REALM_USE_PYTHON
      {"python", {106, nvtx_color::magenta}},
#endif
  };

  REALM_THREAD_LOCAL std::map<std::string, NvtxCategory *> *nvtx_categories;

  static nvtxDomainHandle_t nvtxRealmDomain = nullptr;

  static std::vector<std::string> enabled_nvtx_modules;

  static inline NvtxCategory *find_category_by_name(const std::string &name)
  {
    std::map<std::string, NvtxCategory *>::iterator it = nvtx_categories->find(name);
    if(it != nvtx_categories->end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class nvtxCategory

  NvtxCategory::NvtxCategory(const std::string &category_name, uint32_t category_id,
                             uint32_t color)
    : name(category_name)
  {
    // name the category
    nvtxDomainNameCategoryA(nvtxRealmDomain, category_id, category_name.c_str());

    // create nvtx event attribute and set values
    memset(&nvtx_event, 0, NVTX_EVENT_ATTRIB_STRUCT_SIZE);
    nvtx_event.version = NVTX_VERSION;
    nvtx_event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_event.category = category_id;
    nvtx_event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_event.message.ascii = "";
    nvtx_event.payloadType = NVTX_PAYLOAD_TYPE_INT32;
    nvtx_event.payload.iValue = 0;
    nvtx_event.colorType = NVTX_COLOR_ARGB;
    nvtx_event.color = color;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class nvtxScopedRange

  nvtxScopedRange::nvtxScopedRange(NvtxCategory *category, char const *message,
                                   int32_t payload)
  {
    category->nvtx_event.message.ascii = message;
    category->nvtx_event.payload.iValue = payload;
    nvtxDomainRangePushEx(nvtxRealmDomain, &(category->nvtx_event));
  }

  nvtxScopedRange::nvtxScopedRange(const std::string &name, char const *message,
                                   int32_t payload)
  {
    NvtxCategory *category = find_category_by_name(name);
    if(category) {
      category->nvtx_event.message.ascii = message;
      category->nvtx_event.payload.iValue = payload;
      nvtxDomainRangePushEx(nvtxRealmDomain, &(category->nvtx_event));
    }
  }

  nvtxScopedRange::~nvtxScopedRange() { nvtxDomainRangePop(nvtxRealmDomain); }

  void init_nvtx_thread(const char *thread_name)
  {
#ifdef REALM_ON_WINDOWS
    nvtxNameOsThread(GetCurrentThreadId(), thread_name)
#else
    nvtxNameOsThread(gettid(), thread_name);
#endif

    nvtx_categories = new std::map<std::string, NvtxCategory *>();
    nvtx_categories->clear();

    if(enabled_nvtx_modules.size() == 1 and enabled_nvtx_modules[0] == "all") {
      // handle -ll:nvtx_modules all
      std::map<std::string, nvtx_category_id_color>::const_iterator it;
      for(it = nvtx_categories_predefined.cbegin();
          it != nvtx_categories_predefined.cend(); it++) {
        nvtx_categories->insert(std::pair<std::string, NvtxCategory *>(
            it->first, new NvtxCategory(it->first, it->second.id, it->second.color)));
      }
    } else {
      for(const std::string &name : enabled_nvtx_modules) {
        if(name == "all") {
          std::cerr << "If all specified, then no other modules are needed." << std::endl;
          abort();
        }
        std::map<std::string, nvtx_category_id_color>::const_iterator it =
            nvtx_categories_predefined.find(name);
        if(it == nvtx_categories_predefined.end()) {
          std::cerr << "Unable to find specified nvtx module: " << name << std::endl;
          abort();
        }
        nvtx_categories->insert(std::pair<std::string, NvtxCategory *>(
            name, new NvtxCategory(name, it->second.id, it->second.color)));
      }
    }
  }

  void finalize_nvtx_thread(void)
  {
    std::map<std::string, NvtxCategory *>::iterator it;
    for(it = nvtx_categories->begin(); it != nvtx_categories->end(); it++) {
      assert(it->second != nullptr);
      delete it->second;
    }
    delete nvtx_categories;
  }

  void init_nvtx(std::vector<std::string> &nvtx_modules)
  {
    enabled_nvtx_modules = nvtx_modules;
    nvtxInitialize(nullptr);
    nvtxRealmDomain = nvtxDomainCreateA(realm_nvtx_domain::name);
    init_nvtx_thread("MainThread");
  }

  void finalize_nvtx(void)
  {
    nvtxDomainDestroy(nvtxRealmDomain);
    finalize_nvtx_thread();
  }

  void nvtx_range_push(NvtxCategory *category, const char *message, uint32_t color,
                       int32_t payload)
  {
    uint32_t origin_color;
    if(color != nvtx_color::white) {
      origin_color = category->nvtx_event.color;
      category->nvtx_event.color = color;
    }
    category->nvtx_event.message.ascii = message;
    category->nvtx_event.payload.iValue = payload;
    nvtxDomainRangePushEx(nvtxRealmDomain, &(category->nvtx_event));
    if(color != nvtx_color::white) {
      category->nvtx_event.color = origin_color;
    }
  }

  void nvtx_range_push(const std::string &name, const char *message, uint32_t color,
                       int32_t payload)
  {
    NvtxCategory *category = find_category_by_name(name);
    if(category) {
      nvtx_range_push(category, message, color, payload);
    }
  }

  void nvtx_range_pop(void) { nvtxDomainRangePop(nvtxRealmDomain); }

  nvtxRangeId_t nvtx_range_start(NvtxCategory *category, const char *message,
                                 uint32_t color, int32_t payload)
  {
    uint32_t origin_color;
    if(color != nvtx_color::white) {
      origin_color = category->nvtx_event.color;
      category->nvtx_event.color = color;
    }
    category->nvtx_event.message.ascii = message;
    category->nvtx_event.payload.iValue = payload;
    nvtxRangeId_t id = nvtxDomainRangeStartEx(nvtxRealmDomain, &(category->nvtx_event));
    if(color != nvtx_color::white) {
      category->nvtx_event.color = origin_color;
    }
    return id;
  }

  nvtxRangeId_t nvtx_range_start(const std::string &name, const char *message,
                                 uint32_t color, int32_t payload)
  {
    NvtxCategory *category = find_category_by_name(name);
    if(category) {
      return nvtx_range_start(category, message, color, payload);
    } else {
      return 0;
    }
  }

  void nvtx_range_end(nvtxRangeId_t id) { nvtxDomainRangeEnd(nvtxRealmDomain, id); }

  void nvtx_mark(NvtxCategory *category, const char *message, uint32_t color,
                 int32_t payload)
  {
    uint32_t origin_color;
    if(color != nvtx_color::white) {
      origin_color = category->nvtx_event.color;
      category->nvtx_event.color = color;
    }
    category->nvtx_event.message.ascii = message;
    category->nvtx_event.color = color;
    category->nvtx_event.payload.iValue = payload;
    nvtxDomainMarkEx(nvtxRealmDomain, &(category->nvtx_event));
    if(color != nvtx_color::white) {
      category->nvtx_event.color = origin_color;
    }
  }

  void nvtx_mark(const std::string &name, const char *message, uint32_t color,
                 int32_t payload)
  {
    NvtxCategory *category = find_category_by_name(name);
    if(category) {
      nvtx_mark(category, message, color, payload);
    }
  }

}; // namespace Realm