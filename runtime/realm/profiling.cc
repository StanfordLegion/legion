/* Copyright 2015 Stanford University, NVIDIA Corporation
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

// implementation of profiling stuff for Realm

#include "profiling.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequest
  //

  ProfilingRequest::ProfilingRequest(Processor _response_proc, TaskFuncID _response_task_id)
    : response_proc(_response_proc), response_task_id(_response_task_id)
    , user_data(0), user_data_size(0)
  {}

  ProfilingRequest::ProfilingRequest(const ProfilingRequest& to_copy)
    : response_proc(to_copy.response_proc), response_task_id(to_copy.response_task_id)
    , user_data(0), user_data_size(to_copy.user_data_size)
    , requested_measurements(to_copy.requested_measurements)
  {
    if(to_copy.user_data) {
      user_data = malloc(to_copy.user_data_size);
      assert(user_data != 0);
      memcpy(user_data, to_copy.user_data, to_copy.user_data_size);
    }
  }

  ProfilingRequest::~ProfilingRequest(void)
  {
    if(user_data) {
      free(user_data);
      user_data = 0;
    }
  }

  ProfilingRequest& ProfilingRequest::add_user_data(const void *payload, size_t payload_size)
  {
    assert(user_data == 0);
    if(payload_size > 0) {
      user_data = malloc(payload_size);
      assert(user_data != 0);
      user_data_size = payload_size;
      memcpy(user_data, payload, payload_size);
    }
    return *this;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingRequestSet
  //

  ProfilingRequestSet::ProfilingRequestSet(void)
  {}

  ProfilingRequestSet::ProfilingRequestSet(const ProfilingRequestSet& to_copy)
  {
    // deep copy
    for(std::vector<ProfilingRequest *>::const_iterator it = to_copy.requests.begin();
	it != to_copy.requests.end();
	it++)
      requests.push_back(new ProfilingRequest(**it));
  }

  ProfilingRequestSet::~ProfilingRequestSet(void)
  {
    // destroy all of our set members
    for(std::vector<ProfilingRequest *>::iterator it = requests.begin();
	it != requests.end();
	it++)
      delete *it;
  }

  ProfilingRequest& ProfilingRequestSet::add_request(Processor response_proc, TaskFuncID response_task_id,
						     const void *payload /*= 0*/, size_t payload_size /*= 0*/)
  {
    ProfilingRequest *pr = new ProfilingRequest(response_proc, response_task_id);

    if(payload)
      pr->add_user_data(payload, payload_size);

    requests.push_back(pr);

    return *pr;
  }

  size_t ProfilingRequestSet::request_count(void) const
  {
    return requests.size();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingMeasurementCollection
  //

  ProfilingMeasurementCollection::ProfilingMeasurementCollection(void)
  {}

  ProfilingMeasurementCollection::~ProfilingMeasurementCollection(void)
  {
    // have to delete any serialized measurements we have
    for(std::map<ProfilingMeasurementID, MeasurementData>::iterator it = measurements.begin();
	it != measurements.end();
	it++)
      free(it->second.base);
  }

  void ProfilingMeasurementCollection::import_requests(const ProfilingRequestSet& prs)
  {
    // just iterate over all the individual requests and union the sets of measurements requested
    for(std::vector<ProfilingRequest *>::const_iterator it = prs.requests.begin();
	it != prs.requests.end();
	it++)
      requested_measurements.insert((*it)->requested_measurements.begin(),
				    (*it)->requested_measurements.end());
  }

  void ProfilingMeasurementCollection::send_responses(const ProfilingRequestSet& prs) const
  {
    // print raw data right now
#ifdef DEBUG_PROFILING
    printf("raw profiling results:\n");
    for(std::map<ProfilingMeasurementID, MeasurementData>::const_iterator it = measurements.begin();
	it != measurements.end();
	it++) {
      printf("[%d] = %zd (", (int)(it->first), it->second.size);
      for(size_t i = 0; i < it->second.size; i++)
	printf(" %02x", ((unsigned char *)(it->second.base))[i]);
      printf(" )\n");
    }
#endif

    for(std::vector<ProfilingRequest *>::const_iterator it = prs.requests.begin();
	it != prs.requests.end();
	it++) {
      const ProfilingRequest& pr = **it;

      // for each request, find the intersection of the measurements it wants and the ones we have
      std::set<ProfilingMeasurementID> ids;

      // at the very least, we need a count of measurements and the offset of the user data
      size_t bytes_needed = 2 * sizeof(int) + pr.user_data_size;

      for(std::set<ProfilingMeasurementID>::const_iterator it2 = pr.requested_measurements.begin();
	  it2 != pr.requested_measurements.end();
	  it2++) {
	std::map<ProfilingMeasurementID, MeasurementData>::const_iterator it3 = measurements.find(*it2);
	if(it3 == measurements.end()) continue;

	ids.insert(*it2);
	size_t msize = it3->second.size;
	// we'll pad each measurement to an 8 byte boundary
	size_t msize_padded = (msize + 7) & ~7ULL;

	bytes_needed += 2 * sizeof(int);  // to store ID and offset of data
	bytes_needed += msize_padded;     // to store actual data
      }

      char *payload = (char *)malloc(bytes_needed);
      assert(payload != 0);

      int count = ids.size();

      int *header = (int *)payload;  // first bunch of stuff is a big int array
      char *data = payload + (2 + 2 * count) * sizeof(int);

      *header++ = count;
      for(std::set<ProfilingMeasurementID>::const_iterator it2 = ids.begin();
	  it2 != ids.end();
	  it2++) {
	*header = (int)(*it2);
	*(header + count) = data - payload; // offset of data start
	header++;

	std::map<ProfilingMeasurementID, MeasurementData>::const_iterator it3 = measurements.find(*it2);
	assert(it3 != measurements.end());
	const MeasurementData& mdata(it3->second);
	memcpy(data, mdata.base, mdata.size);

	size_t msize_padded = (mdata.size + 7) & ~7ULL;
	data += msize_padded;
      }

      // offset of user data start is always provided (if it equals the response size, there's no data)
      *(header + count) = data - payload;
      if(pr.user_data_size > 0) {
	memcpy(data, pr.user_data, pr.user_data_size);
	data += pr.user_data_size;
      }

      assert((size_t)(data - payload) == bytes_needed);

      pr.response_proc.spawn(pr.response_task_id, payload, bytes_needed);

      free(payload);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ProfilingResponse
  //

  ProfilingResponse::ProfilingResponse(const void *_data, size_t _data_size)
    : data(static_cast<const char *>(_data)), data_size(_data_size)
  {
    const int *idata = static_cast<const int *>(_data);
    
    measurement_count = idata[0];
    ids = &(idata[1]);
    user_data_offset = idata[2 * measurement_count + 1];
  }

  ProfilingResponse::~ProfilingResponse(void)
  {
    // nothing to free - we didn't own the data
  }

  const void *ProfilingResponse::user_data(void) const
  {
    if(user_data_offset < data_size)
      return data + user_data_offset;
    else
      return 0;
  }

  size_t ProfilingResponse::user_data_size(void) const
  {
    return data_size - user_data_offset;
  }

  bool ProfilingResponse::find_id(int id, int& offset, int& size) const
  {
    // binary search on ids
    int lo = 0;
    int hi = measurement_count - 1;

    while(lo <= hi) {
      int mid = (lo + hi) >> 1;
      if (id < ids[mid]) {
	hi = mid - 1;
      } else if(id > ids[mid]) {
	lo = mid + 1;
      } else {
	offset = ids[mid + measurement_count];
	size = ids[mid + measurement_count + 1] - offset;
	return true;
      }
    }

    return false;
  }

}; // namespace Realm
