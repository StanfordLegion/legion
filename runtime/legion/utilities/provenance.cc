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

#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Provenance
    /////////////////////////////////////////////////////////////

    /*static*/ constexpr std::string_view Provenance::no_provenance;

    //--------------------------------------------------------------------------
    Provenance::Provenance(ProvenanceID p, const char* prov)
      : pid(p), full(prov)
    //--------------------------------------------------------------------------
    {
      initialize();
    }

    //--------------------------------------------------------------------------
    Provenance::Provenance(ProvenanceID p, const void* buffer, size_t size)
      : pid(p), full((const char*)buffer, size)
    //--------------------------------------------------------------------------
    {
      initialize();
    }

    //--------------------------------------------------------------------------
    Provenance::Provenance(ProvenanceID p, const std::string& prov)
      : pid(p), full(prov)
    //--------------------------------------------------------------------------
    {
      initialize();
    }

    //--------------------------------------------------------------------------
    void Provenance::initialize(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!full.empty());
      if (!parse_provenance_parts())
      {
        // If we have a bracket assume this whole this is a JSON string
        // and therefore we're going to assume the whole thing is JSON
        // Otherwise if things don't parse then everything is the just
        // the human readable string.
        if (full[0] == '{')
          machine = std::string_view(full.c_str());
        else
          human = std::string_view(full.c_str());
      }
    }

    //--------------------------------------------------------------------------
    bool Provenance::parse_provenance_parts(void)
    //--------------------------------------------------------------------------
    {
      {
        size_t len = full.length();

        // shortest valid input: ["",{}]
        if (len < 7)
          return false;

        // must start with: ["
        if (full[0] != '[' || full[1] != '"')
          return false;

        // must end with: }]
        if (full[len - 2] != '}' || full[len - 1] != ']')
          return false;
      }

      unsigned human_size = 0;
      bool human_closed = false;
      std::string::iterator it = full.begin() + 2;
      for (; it != full.end(); it++)
      {
        if (*it == '\\')
        {
          // Remove the escape character
          it = full.erase(it);
          if (it == full.end())
            return false;
          switch (*it)
          {
            case '"':
            case '\\':
            case '/':
              break;
            case 'b':
              *it = '\b';
              break;
            case 'f':
              *it = '\f';
              break;
            case 'n':
              *it = '\n';
              break;
            case 'r':
              *it = '\r';
              break;
            case 't':
              *it = '\t';
              break;
            case 'u':
              return false;  // Unicode is unsupported
            default:
              return false;  // Bad escape
          }
          human_size++;
        }
        else if (*it == '"')
        {
          human_closed = true;
          break;
        }
        else
          human_size++;
      }

      if (!human_closed)
        return false;

      human = std::string_view(full.c_str() + 2, human_size);

      for (; it != full.end(); it++)
      {
        if (*it == '{')
        {
          size_t offset = std::distance(full.begin(), it);
          // Start from our current offset and go to the
          // end but don't include the closing ']'
          machine = std::string_view(
              full.c_str() + offset, full.length() - (offset + 1));
          return true;
        }
      }

      // machine part never opened
      return false;
    }

    //--------------------------------------------------------------------------
    void Provenance::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!full.empty());
      rez.serialize<size_t>(full.size());
      rez.serialize(full.c_str(), full.size() + 1 /*null terminator*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Provenance::serialize_null(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    /*static*/ Provenance* Provenance::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t length;
      derez.deserialize(length);
      if (length > 0)
      {
        Provenance* result = runtime->find_or_create_provenance(
            (const char*)derez.get_current_pointer(), length);
        derez.advance_pointer(length + 1 /*null terminator*/);
        return result;
      }
      else
        return nullptr;
    }

  }  // namespace Internal
}  // namespace Legion
