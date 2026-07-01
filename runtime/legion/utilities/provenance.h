/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_PROVENANCE_H__
#define __LEGION_PROVENANCE_H__

#include <string_view>

#include "legion/kernel/garbage_collection.h"
#include "legion/kernel/runtime.h"

namespace Legion {
  namespace Internal {

    /**
     * \class Provenance
     */
    class Provenance : public Collectable {
    public:
      Provenance(ProvenanceID pid, const char* prov);
      Provenance(ProvenanceID pid, const void* buffer, size_t size);
      Provenance(ProvenanceID pid, const std::string& prov);
      Provenance(const Provenance& rhs) = delete;
      ~Provenance(void) { }
    public:
      Provenance& operator=(const Provenance& rhs) = delete;
    public:
      void initialize(void);
      bool parse_provenance_parts(void);
      void serialize(Serializer& rez) const;
      static void serialize_null(Serializer& rez);
      static Provenance* deserialize(Deserializer& derez);
    public:
      const ProvenanceID pid;
    public:
      std::string full;
      // Keep the human and machine parts of the provenance string
      std::string_view human, machine;
      // Useful for cases where interfaces want a string
      static constexpr std::string_view no_provenance = std::string_view();
    };

    /**
     * \class AutoProvenance
     * Make a provenance from a string if it exists
     * Reclaim references on the provenance at the end
     * of the scope so it will be cleaned up if needed
     */
    class AutoProvenance {
    public:
      AutoProvenance(void) : provenance(nullptr) { }
      AutoProvenance(const char* prov)
        : provenance(
              (prov == nullptr) ?
                  nullptr :
                  runtime->find_or_create_provenance(prov, strlen(prov)))
      { }
      AutoProvenance(const std::string& prov)
        : provenance(
              prov.empty() ?
                  nullptr :
                  runtime->find_or_create_provenance(prov.c_str(), prov.size()))
      { }
      AutoProvenance(Provenance* prov, bool has_ref) : provenance(prov)
      {
        if ((provenance != nullptr) && !has_ref)
          provenance->add_reference();
      }
      AutoProvenance(AutoProvenance&& rhs) = delete;
      AutoProvenance(const AutoProvenance& rhs) = delete;
      ~AutoProvenance(void)
      {
        if ((provenance != nullptr) && provenance->remove_reference())
          delete provenance;
      }
    public:
      AutoProvenance& operator=(AutoProvenance&& rhs) = delete;
      AutoProvenance& operator=(const AutoProvenance& rhs) = delete;
    public:
      inline operator Provenance*(void) const { return provenance; }
    protected:
      Provenance* const provenance;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PROVENANCE_H__
