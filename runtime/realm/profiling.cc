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

  size_t ProfilingRequest::compute_size(void) const
  {
    size_t result = sizeof(response_proc) + sizeof(response_task_id) + 
                    (2 * sizeof(size_t)) + user_data_size + 
                    requested_measurements.size() * sizeof(ProfilingMeasurementID);
    return result;
  }

  void* ProfilingRequest::serialize(void *target) const
  {
    char *buffer = (char*)target;
    memcpy(buffer,&response_proc,sizeof(response_proc));
    buffer += sizeof(response_proc);
    memcpy(buffer,&response_task_id,sizeof(response_task_id));
    buffer += sizeof(response_task_id);
    memcpy(buffer,&user_data_size,sizeof(user_data_size));
    buffer += sizeof(user_data_size);
    if (user_data_size > 0) {
      memcpy(buffer,user_data,user_data_size);
      buffer += user_data_size;
    }
    size_t measurement_count = requested_measurements.size();
    memcpy(buffer,&measurement_count,sizeof(measurement_count));
    buffer += sizeof(measurement_count);
    for (std::set<ProfilingMeasurementID>::const_iterator it = 
          requested_measurements.begin(); it != requested_measurements.end(); it++)
    {
      const ProfilingMeasurementID &measurement = *it;
      memcpy(buffer,&measurement,sizeof(measurement));
      buffer += sizeof(measurement);
    }
    return buffer;
  }

  const void* ProfilingRequest::deserialize(const void *source)
  {
    // Already unpacked proc and task
    const char *buffer = (const char*)source;
    memcpy(&user_data_size,buffer,sizeof(user_data_size));
    buffer += sizeof(user_data_size);
    if (user_data_size > 0) {
      add_user_data(buffer, user_data_size);
      buffer += user_data_size;
    }
    size_t num_measurements;
    memcpy(&num_measurements,buffer,sizeof(num_measurements));
    buffer += sizeof(num_measurements);
    for (unsigned idx = 0; idx < num_measurements; idx++)
    {
      ProfilingMeasurementID measurement;
      memcpy(&measurement,buffer,sizeof(measurement));
      buffer += sizeof(measurement);
      switch (measurement)
      {
        case ProfilingMeasurements::OperationStatus::ID:
          {
            add_measurement<ProfilingMeasurements::OperationStatus>();
            break;
          }
        case ProfilingMeasurements::OperationTimeline::ID:
          {
            add_measurement<ProfilingMeasurements::OperationTimeline>();
            break;
          }
        case ProfilingMeasurements::OperationProcessorUsage::ID:
          {
            add_measurement<ProfilingMeasurements::OperationProcessorUsage>();
            break;
          }
        case ProfilingMeasurements::OperationMemoryUsage::ID:
          {
            add_measurement<ProfilingMeasurements::OperationMemoryUsage>();
            break;
          }
        case ProfilingMeasurements::InstanceTimeline::ID:
          {
            add_measurement<ProfilingMeasurements::InstanceTimeline>();
            break;
          }
        case ProfilingMeasurements::InstanceMemoryUsage::ID:
          {
            add_measurement<ProfilingMeasurements::InstanceMemoryUsage>();
            break;
          }
        default:
          assert(false);
      }
    }
    return buffer;
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

  bool ProfilingRequestSet::empty(void) const
  {
    return requests.empty();
  }

  void ProfilingRequestSet::clear(void)
  {
    for (std::vector<ProfilingRequest*>::iterator it = 
          requests.begin(); it != requests.end(); it++)
    {
      delete (*it);
    }
    requests.clear();
  }

  size_t ProfilingRequestSet::compute_size(void) const
  {
    size_t result = sizeof(size_t);
    for (std::vector<ProfilingRequest*>::const_iterator it = 
          requests.begin(); it != requests.end(); it++)
    {
      result += (*it)->compute_size();
    }
    return result;
  }

  void* ProfilingRequestSet::serialize(void *target) const
  {
    char *buffer = (char*)target;
    size_t num_requests = requests.size();
    memcpy(buffer,&num_requests,sizeof(num_requests));
    buffer += sizeof(num_requests);
    for (std::vector<ProfilingRequest*>::const_iterator it = 
          requests.begin(); it != requests.end(); it++)
    {
      buffer = (char*)((*it)->serialize(buffer));
    }
    return buffer;
  }

  const void* ProfilingRequestSet::deserialize(const void *source)
  {
    const char *buffer = (const char*)source;
    size_t num_reqs;
    memcpy(&num_reqs,buffer,sizeof(num_reqs));
    buffer += sizeof(num_reqs);
    for (unsigned idx = 0; idx < num_reqs; idx++)
    {
      Processor resp_proc;
      memcpy(&resp_proc,buffer,sizeof(resp_proc));
      buffer += sizeof(resp_proc);
      TaskFuncID resp_task;
      memcpy(&resp_task,buffer,sizeof(resp_task));
      buffer += sizeof(resp_task);
      ProfilingRequest *new_req = new ProfilingRequest(resp_proc, resp_task);
      buffer = (const char*)new_req->deserialize(buffer);
      requests.push_back(new_req);
    }
    return buffer;
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
