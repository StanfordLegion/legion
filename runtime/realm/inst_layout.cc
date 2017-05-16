/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// Layout descriptors for Realm RegionInstances

#include "inst_layout.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayoutGeneric

  static InstanceLayoutGeneric *choose_instance_layout(Domain d,
						       const std::vector<size_t>& field_sizes,
						       size_t block_size)
  {
    // first, use the field sizes to generate "offsets" as unique IDs
    FieldGroups fields;
    switch(block_size) {
    case 0:
      {
	// SOA - each field is its own "group"
	fields.resize(field_sizes.size());
	size_t offset = 0;
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  fields[i].resize(1);
	  fields[i][0].offset = offset;
	  fields[i][0].size = field_sizes[i];
	  offset += field_sizes[i];
	}
	break;
      }

    case 1:
      {
	// AOS - all fields in same group
	fields.resize(1);
	fields[0].resize(field_sizes.size());
	size_t offset = 0;
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  fields[0][i].offset = offset;
	  fields[0][i].size = field_sizes[i];
	  offset += field_sizes[i];
	}
	break;
      }

    default:
      {
	// hybrid - blech
	assert(0);
      }
    }

    // now switch by domain dimensionality
    switch(d.get_dim()) {
    case 0: assert(0);

    case 1:
      {
	ZRect<1, coord_t> bounds;
      return
	for(std::vector<size_t>::const_iterator it = field_sizes.begin();
	    it != field_sizes.end();
	    ++it) {
	  
  }
    template <int N, typename T>
    static InstanceLayoutGeneric *choose_instance_layout(ZIndexSpace<N,T> is,
							 std::vector<std::map<FieldID,size_t> >& fields);
#ifdef REALM_USE_LEGION_LAYOUT_CONSTRAINTS
    template <int N, typename T>
    static InstanceLayoutGeneric *choose_instance_layout(ZIndexSpace<N,T> is,
                                                         const Legion::LayoutConstraintSet& lcs);
#endif

    size_t bytes_used;
    size_t alignment_reqd;
  };

  template <int N, typename T>
  class InstanceLayoutPiece {
  public:
    InstanceLayoutPiece(void);
    virtual ~InstanceLayoutPiece(void);

    enum LayoutType {
      InvalidLayoutType,
      AffineLayoutType,
    };

    LayoutType layout_type;
    ZRect<N,T> bounds;
  };

  template <int N, typename T>
  class AffineLayoutPiece : public InstanceLayoutPiece<N,T> {
  public:
    AffineLayoutPiece(void);

    ZPoint<N, size_t> strides;
    size_t offset;
  };

  template <int N, typename T>
  class InstancePieceList {
  public:
    InstancePieceList(void);
    ~InstancePieceList(void);

    const InstanceLayoutPiece<N,T> *find_piece(ZPoint<N,T> p) const;

    std::vector<InstanceLayoutPiece<N,T> *> pieces;
    // placeholder for lookup structure (e.g. K-D tree)
  };

  template <int N, typename T>
  class InstanceLayout : public InstanceLayoutGeneric {
  public:
    InstanceLayout(void);
    virtual ~InstanceLayout(void);

    // adjusts offsets of pieces to start from 'base_offset'
    virtual void relocate(size_t base_offset);

    // we optimize for fields being laid out similarly, and have fields
    //  indirectly reference a piece list
    struct FieldLayout {
      int list_idx;
      int rel_offset;
    };

    std::map<FieldID, FieldLayout> fields;
    std::vector<InstancePieceList<N,T> > piece_lists;
  };

}; // namespace Realm

#include "inst_layout.inl"

#endif // ifndef REALM_INST_LAYOUT_H


