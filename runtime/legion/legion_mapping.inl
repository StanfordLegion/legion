/* Copyright 2016 Stanford University, NVIDIA Corporation
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

// Included from legion_mapping.h - do not include this directly

// Useful for IDEs 
#include "legion_mapping.h"

namespace Legion {

  namespace Mapping {

    //--------------------------------------------------------------------------
    inline ProfilingRequest::ProfilingRequest(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    inline ProfilingRequest::~ProfilingRequest(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template <typename T>
    inline ProfilingRequest &ProfilingRequest::add_measurement(void)
    //--------------------------------------------------------------------------
    {
      requested_measurements.insert((ProfilingMeasurementID)T::ID);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline ProfilingResponse::ProfilingResponse(void)
    //--------------------------------------------------------------------------
      : realm_resp(0)
    {
    }

    //--------------------------------------------------------------------------
    inline ProfilingResponse::~ProfilingResponse(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template <typename T>
    inline bool ProfilingResponse::has_measurement(void) const
    //--------------------------------------------------------------------------
    {
      if ((int)(T::ID) <= (int)(Realm::PMID_REALM_LAST))
      {
	if (realm_resp)
	  return realm_resp->template has_measurement<T>();
	else
	  return false;
      }
      else
      {
	// Legion-specific measurements handled here
	return false;
      }
    }

    //--------------------------------------------------------------------------
    template <typename T>
    inline T *ProfilingResponse::get_measurement(void) const
    //--------------------------------------------------------------------------
    {
      if ((int)(T::ID) <= (int)(Realm::PMID_REALM_LAST))
      {
	if (realm_resp)
	  return realm_resp->template get_measurement<T>();
	else
	  return 0;
      }
      else
      {
	// Legion-specific measurements handled here
	return 0;
      }
    }

  };
};

