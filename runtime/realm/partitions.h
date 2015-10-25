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

// index space partitioning for Realm

#ifndef REALM_PARTITIONS_H
#define REALM_PARTITIONS_H

#include "indexspace.h"
#include "sparsity.h"
#include "activemsg.h"
#include "id.h"

// NOTE: all these interfaces are templated, which means partitions.cc is going
//  to have to somehow know which ones to instantiate - we'll try to have a 
//  Makefile-based way to control this, but right now it's hardcoded at the
//  bottom of partitions.cc, so go there if you get link errors

namespace Realm {

  // although partitioning operations eventually generate SparsityMap's, we work with
  //  various intermediates that try to keep things from turning into one big bitmask

  // the CoverageCounter just counts the number of points that get added to it
  // it's not even smart enough to eliminate duplicates
  template <int N, typename T>
  class CoverageCounter {
  public:
    CoverageCounter(void);

    void add_point(const ZPoint<N,T>& p);

    void add_rect(const ZRect<N,T>& r);

    size_t get_count(void) const;

  protected:
    size_t count;
  };

  template <int N, typename T>
  class DenseRectangleList {
  public:
    DenseRectangleList(void);

    void add_point(const ZPoint<N,T>& p);

    void add_rect(const ZRect<N,T>& r);

    std::vector<ZRect<N,T> > rects;
  };

  /////////////////////////////////////////////////////////////////////////

  template <int N, typename T>
  class SparsityMapImpl : public SparsityMapPublicImpl<N,T> {
  public:
    SparsityMapImpl(void);

    static SparsityMapImpl<N,T> *lookup(SparsityMap<N,T> sparsity);

    // methods used in the population of a sparsity map

    // when we plan out a partitioning operation, we'll know how many
    //  different uops are going to contribute something (or nothing) to
    //  the sparsity map - once all of those contributions arrive, we can
    //  finalize the sparsity map
    void update_contributor_count(int delta = 1);

    void contribute_nothing(void);
    void contribute_dense_rect_list(const DenseRectangleList<N,T>& rects);

  protected:
    void finalize(void);
    
    int remaining_contributor_count;
    GASNetHSL mutex;
  };

  // we need a type-erased wrapper to store in the runtime's lookup table
  class SparsityMapImplWrapper {
  public:
    static const ID::ID_Types ID_TYPE = ID::ID_SPARSITY;

    SparsityMapImplWrapper(void);

    void init(ID _me, unsigned _init_owner);

    ID me;
    unsigned owner;
    SparsityMapImplWrapper *next_free;
    int dim;
    int idxtype; // captured via sizeof(T) right now
    void *map_impl;  // actual implementation

    template <int N, typename T>
    SparsityMapImpl<N,T> *get_or_create(void);

    void destroy(void);
  };


  /////////////////////////////////////////////////////////////////////////

  template <int N, typename T>
  class PartitioningMicroOp {
  public:
    virtual ~PartitioningMicroOp(void);

    virtual void execute(void) = 0;
  };

  template <int N, typename T, typename FT>
  class ByFieldMicroOp : public PartitioningMicroOp<N,T> {
  public:
    ByFieldMicroOp(ZIndexSpace<N,T> _parent_space, ZIndexSpace<N,T> _inst_space,
		   RegionInstance _inst, size_t _field_offset);
    virtual ~ByFieldMicroOp(void);

    void set_value_range(FT _lo, FT _hi);
    void set_value_set(const std::vector<FT>& _value_set);
    void add_sparsity_output(FT _val, SparsityMap<N,T> _sparsity);

    template <typename BM>
    void populate_bitmasks(std::map<FT, BM *>& bitmasks);

    virtual void execute(void);

  protected:
    ZIndexSpace<N,T> parent_space, inst_space;
    RegionInstance inst;
    size_t field_offset;
    bool value_range_valid, value_set_valid;
    FT range_lo, range_hi;
    std::set<FT> value_set;
    std::map<FT, SparsityMap<N,T> > sparsity_outputs;
  };

};

#endif // REALM_PARTITIONS_H

