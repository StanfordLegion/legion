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

// Realm inter-node networking abstractions

// NOP but helpful for IDEs
#include "realm/network.h"

namespace Realm {

  namespace Network {

    // gets the network for a given node
    inline NetworkModule *get_network(NodeID node)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLILELY(single_network == 0)) {
      } else
#endif
	return single_network;
    }

    inline void barrier(void)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	single_network->barrier();
    }

    // collective communication across all nodes (TODO: subcommunicators?)
    template <typename T>
    inline T broadcast(NodeID root, T val)
    {
      T bval;
      broadcast(root, &val, &bval, sizeof(T));
      return bval;
    }

    template <typename T>
    inline void gather(NodeID root, T val, std::vector<T>& result)
    {
      result.resize(max_node_id + 1);
      gather(root, &val, &result[0], sizeof(T));
    }

    template <typename T>
    inline void gather(NodeID root, T val)  // for non-root participants
    {
      gather(root, &val, 0, sizeof(T));
    }
    
    inline void broadcast(NodeID root,
			  const void *val_in, void *val_out, size_t bytes)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	single_network->broadcast(root, val_in, val_out, bytes);
    }
    
    inline void gather(NodeID root,
		       const void *val_in, void *vals_out, size_t bytes)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	single_network->gather(root, val_in, vals_out, bytes);
    }
    
    inline ActiveMessageImpl *create_active_message_impl(NodeID target,
							 unsigned short msgid,
							 size_t header_size,
							 size_t max_payload_size,
							 const void *src_payload_addr,
							 size_t src_payload_lines,
							 size_t src_payload_line_stride,
							 void *storage_base,
							 size_t storage_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->create_active_message_impl(target,
							  msgid,
							  header_size,
							  max_payload_size,
							  src_payload_addr,
							  src_payload_lines,
							  src_payload_line_stride,
							  storage_base,
							  storage_size);
    }

    inline ActiveMessageImpl *create_active_message_impl(NodeID target,
							 unsigned short msgid,
							 size_t header_size,
							 size_t max_payload_size,
							 const void *src_payload_addr,
							 size_t src_payload_lines,
							 size_t src_payload_line_stride,
							 const RemoteAddress& dest_payload_addr,
							 void *storage_base,
							 size_t storage_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->create_active_message_impl(target,
							  msgid,
							  header_size,
							  max_payload_size,
							  src_payload_addr,
							  src_payload_lines,
							  src_payload_line_stride,
							  dest_payload_addr,
							  storage_base,
							  storage_size);
    }

    inline ActiveMessageImpl *create_active_message_impl(const NodeSet& targets,
							 unsigned short msgid,
							 size_t header_size,
							 size_t max_payload_size,
							 const void *src_payload_addr,
							 size_t src_payload_lines,
							 size_t src_payload_line_stride,
							 void *storage_base,
							 size_t storage_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->create_active_message_impl(targets,
							  msgid,
							  header_size,
							  max_payload_size,
							  src_payload_addr,
							  src_payload_lines,
							  src_payload_line_stride,
							  storage_base,
							  storage_size);
    }
    
    inline size_t recommended_max_payload(NodeID target,
					  bool with_congestion,
					  size_t header_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->recommended_max_payload(target,
						       with_congestion,
						       header_size);
    }

    inline size_t recommended_max_payload(const NodeSet& targets,
					  bool with_congestion,
					  size_t header_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->recommended_max_payload(targets,
						       with_congestion,
						       header_size);
    }

    inline size_t recommended_max_payload(NodeID target,
					  const RemoteAddress& dest_payload_addr,
					  bool with_congestion,
					  size_t header_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->recommended_max_payload(target,
						       dest_payload_addr,
						       with_congestion,
						       header_size);
    }

    inline size_t recommended_max_payload(NodeID target,
					  const void *data, size_t bytes_per_line,
					  size_t lines, size_t line_stride,
					  bool with_congestion,
					  size_t header_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->recommended_max_payload(target,
						       data, bytes_per_line,
						       lines, line_stride,
						       with_congestion,
						       header_size);
    }

    inline size_t recommended_max_payload(const NodeSet& targets,
					  const void *data, size_t bytes_per_line,
					  size_t lines, size_t line_stride,
					  bool with_congestion,
					  size_t header_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->recommended_max_payload(targets,
						       data, bytes_per_line,
						       lines, line_stride,
						       with_congestion,
						       header_size);
    }

    inline size_t recommended_max_payload(NodeID target,
					  const void *data, size_t bytes_per_line,
					  size_t lines, size_t line_stride,
					  const RemoteAddress& dest_payload_addr,
					  bool with_congestion,
					  size_t header_size)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
      } else
#endif
	return single_network->recommended_max_payload(target,
						       data, bytes_per_line,
						       lines, line_stride,
						       dest_payload_addr,
						       with_congestion,
						       header_size);
    }

  };


};
