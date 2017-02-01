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

// Included from legion.h - do not include this directly

// Useful for IDEs 
#include "legion.h"

namespace Legion {

    /**
     * \struct SerdezRedopFns
     * Small helper class for storing instantiated templates
     */
    struct SerdezRedopFns {
    public:
      SerdezInitFnptr init_fn;
      SerdezFoldFnptr fold_fn;
    };

    /**
     * \class LegionSerialization
     * The Legion serialization class provides template meta-programming
     * help for returning complex data types from task calls.  If the 
     * types have three special methods defined on them then we know
     * how to serialize the type for the runtime rather than just doing
     * a dumb bit copy.  This is especially useful for types which 
     * require deep copies instead of shallow bit copies.  The three
     * methods which must be defined are:
     * size_t legion_buffer_size(void)
     * void legion_serialize(void *buffer)
     * void legion_deserialize(const void *buffer)
     */
    class LegionSerialization {
    public:
      // A helper method for getting access to the runtime's
      // end_task method with private access
      static inline void end_helper(Runtime *rt, InternalContext ctx,
          const void *result, size_t result_size, bool owned)
      {
        ctx->end_task(result, result_size, owned);
      }
      static inline Future from_value_helper(Runtime *rt, 
          const void *value, size_t value_size, bool owned)
      {
        return rt->from_value(value, value_size, owned);
      }

      // WARNING: There are two levels of SFINAE (substitution failure is 
      // not an error) here.  Proceed at your own risk. First we have to 
      // check to see if the type is a struct.  If it is then we check to 
      // see if it has a 'legion_serialize' method.  We assume if there is 
      // a 'legion_serialize' method there are also 'legion_buffer_size'
      // and 'legion_deserialize' methods.
      
      template<typename T, bool HAS_SERIALIZE>
      struct NonPODSerializer {
        static inline void end_task(Runtime *rt, InternalContext ctx,
                                    T *result)
        {
          size_t buffer_size = result->legion_buffer_size();
          void *buffer = malloc(buffer_size);
          result->legion_serialize(buffer);
          end_helper(rt, ctx, buffer, buffer_size, true/*owned*/);
          // No need to free the buffer, the Legion runtime owns it now
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          size_t buffer_size = value->legion_buffer_size();
          void *buffer = malloc(buffer_size);
          value->legion_serialize(buffer);
          return from_value_helper(rt, buffer, buffer_size, true/*owned*/);
        }
        static inline T unpack(const void *result)
        {
          T derez;
          derez.legion_deserialize(result);
          return derez;
        }
      };

      template<typename T>
      struct NonPODSerializer<T,false> {
        static inline void end_task(Runtime *rt, InternalContext ctx,
                                    T *result)
        {
          end_helper(rt, ctx, (void*)result, sizeof(T), false/*owned*/);
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          return from_value_helper(rt, (const void*)value,
                                   sizeof(T), false/*owned*/);
        }
        static inline T unpack(const void *result)
        {
          return (*((const T*)result));
        }
      };

      template<typename T>
      struct HasSerialize {
        typedef char no[1];
        typedef char yes[2];

        struct Fallback { void legion_serialize(void *); };
        struct Derived : T, Fallback { };

        template<typename U, U> struct Check;

        template<typename U>
        static no& test_for_serialize(
                  Check<void (Fallback::*)(void*), &U::legion_serialize> *);

        template<typename U>
        static yes& test_for_serialize(...);

        static const bool value = 
          (sizeof(test_for_serialize<Derived>(0)) == sizeof(yes));
      };

      template<typename T, bool IS_STRUCT>
      struct StructHandler {
        static inline void end_task(Runtime *rt, 
                                    InternalContext ctx, T *result)
        {
          // Otherwise this is a struct, so see if it has serialization methods 
          NonPODSerializer<T,HasSerialize<T>::value>::end_task(rt, ctx, result);
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          return NonPODSerializer<T,HasSerialize<T>::value>::from_value(
                                                                  rt, value);
        }
        static inline T unpack(const void *result)
        {
          return NonPODSerializer<T,HasSerialize<T>::value>::unpack(result); 
        }
      };
      // False case of template specialization
      template<typename T>
      struct StructHandler<T,false> {
        static inline void end_task(Runtime *rt, InternalContext ctx, 
                                    T *result)
        {
          end_helper(rt, ctx, (void*)result, sizeof(T), false/*owned*/);
        }
        static inline Future from_value(Runtime *rt, const T *value)
        {
          return from_value_helper(rt, (const void*)value, 
                                   sizeof(T), false/*owned*/);
        }
        static inline T unpack(const void *result)
        {
          return (*((const T*)result));
        }
      };

      template<typename T>
      struct IsAStruct {
        typedef char no[1];
        typedef char yes[2];
        
        template <typename U> static yes& test_for_struct(int U:: *x);
        template <typename U> static no& test_for_struct(...);

        static const bool value = 
                        (sizeof(test_for_struct<T>(0)) == sizeof(yes));
      };

      // Figure out whether this is a struct or not 
      // and call the appropriate Finisher
      template<typename T>
      static inline void end_task(Runtime *rt, InternalContext ctx, T *result)
      {
        StructHandler<T,IsAStruct<T>::value>::end_task(rt, ctx, result);
      }

      template<typename T>
      static inline Future from_value(Runtime *rt, const T *value)
      {
        return StructHandler<T,IsAStruct<T>::value>::from_value(rt, value);
      }

      template<typename T>
      static inline T unpack(const void *result)
      {
        return StructHandler<T,IsAStruct<T>::value>::unpack(result);
      }

      // Some more help for reduction operations with RHS types
      // that have serialize and deserialize methods

      template<typename REDOP_RHS>
      static void serdez_redop_init(const ReductionOp *reduction_op,
                              void *&ptr, size_t &size)
      {
        REDOP_RHS init_serdez;
        reduction_op->init(&init_serdez, 1);
        size_t new_size = init_serdez.legion_buffer_size();
        if (new_size > size)
        {
          size = new_size;
          ptr = realloc(ptr, size);
        }
        init_serdez.legion_serialize(ptr);
      }

      template<typename REDOP_RHS>
      static void serdez_redop_fold(const ReductionOp *reduction_op,
                                    void *&lhs_ptr, size_t &lhs_size,
                                    const void *rhs_ptr)
      {
        REDOP_RHS lhs_serdez, rhs_serdez;
        lhs_serdez.legion_deserialize(lhs_ptr);
        rhs_serdez.legion_deserialize(rhs_ptr);
        reduction_op->fold(&lhs_serdez, &rhs_serdez, 1, true/*exclusive*/);
        size_t new_size = lhs_serdez.legion_buffer_size();
        // Reallocate the buffer if it has grown
        if (new_size > lhs_size)
        {
          lhs_size = new_size;
          lhs_ptr = realloc(lhs_ptr, lhs_size);
        }
        // Now save the value
        lhs_serdez.legion_serialize(lhs_ptr);
      }

      template<typename REDOP_RHS, bool HAS_SERDEZ>
      struct SerdezRedopHandler {
        static inline void register_reduction(SerdezRedopTable &table,
                                              ReductionOpID redop_id)
        {
          // Do nothing in the case where there are no serdez functions
        }
      };
      // True case of template specialization
      template<typename REDOP_RHS>
      struct SerdezRedopHandler<REDOP_RHS,true> {
        static inline void register_reduction(SerdezRedopTable &table,
                                              ReductionOpID redop_id)
        {
          // Now we can do the registration
          SerdezRedopFns &fns = table[redop_id];
          fns.init_fn = serdez_redop_init<REDOP_RHS>;
          fns.fold_fn = serdez_redop_fold<REDOP_RHS>;
        }
      };

      template<typename REDOP_RHS, bool IS_STRUCT>
      struct StructRedopHandler {
        static inline void register_reduction(SerdezRedopTable &table,
                                              ReductionOpID redop_id)
        {
          // Do nothing in the case where this isn't a struct
        }
      };
      // True case of template specialization
      template<typename REDOP_RHS>
      struct StructRedopHandler<REDOP_RHS,true> {
        static inline void register_reduction(SerdezRedopTable &table,
                                              ReductionOpID redop_id)
        {
          SerdezRedopHandler<REDOP_RHS,HasSerialize<REDOP_RHS>::value>::
            register_reduction(table, redop_id);
        }
      };

      // Register reduction functions if necessary
      template<typename REDOP>
      static inline void register_reduction(SerdezRedopTable &table,
                                            ReductionOpID redop_id)
      {
        StructRedopHandler<typename REDOP::RHS, 
          IsAStruct<typename REDOP::RHS>::value>::register_reduction(table, 
                                                                     redop_id);
      }

    };

    //--------------------------------------------------------------------------
    inline IndexSpace& IndexSpace::operator=(const IndexSpace &rhs)
    //--------------------------------------------------------------------------
    {
      id = rhs.id;
      tid = rhs.tid;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator==(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id != rhs.id)
        return false;
      if (tid != rhs.tid)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator!=(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((id == rhs.id) && (tid == rhs.tid))
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator<(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id < rhs.id)
        return true;
      if (id > rhs.id)
        return false;
      return (tid < rhs.tid);
    }

    //--------------------------------------------------------------------------
    inline bool IndexSpace::operator>(const IndexSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id > rhs.id)
        return true;
      if (id < rhs.id)
        return false;
      return (tid > rhs.tid);
    }

    //--------------------------------------------------------------------------
    inline IndexPartition& IndexPartition::operator=(const IndexPartition &rhs)
    //--------------------------------------------------------------------------
    {
      id = rhs.id;
      tid = rhs.tid;
      return *this;
    }
    
    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator==(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id != rhs.id)
        return false;
      if (tid != rhs.tid)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator!=(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((id == rhs.id) && (tid == rhs.tid))
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator<(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id < rhs.id)
        return true;
      if (id > rhs.id)
        return false;
      return (tid < rhs.tid);
    }

    //--------------------------------------------------------------------------
    inline bool IndexPartition::operator>(const IndexPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (id > rhs.id)
        return true;
      if (id < rhs.id)
        return false;
      return (tid > rhs.tid);
    }
    
    //--------------------------------------------------------------------------
    inline FieldSpace& FieldSpace::operator=(const FieldSpace &rhs)
    //--------------------------------------------------------------------------
    {
      id = rhs.id;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator==(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id == rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator!=(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id != rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator<(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id < rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator>(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id > rhs.id);
    }

    //--------------------------------------------------------------------------
    inline LogicalRegion& LogicalRegion::operator=(const LogicalRegion &rhs) 
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.tree_id;
      index_space = rhs.index_space;
      field_space = rhs.field_space;
      return *this;
    }
    
    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator==(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && (index_space == rhs.index_space) 
              && (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator!=(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      return (!((*this) == rhs));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator<(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_space < rhs.index_space)
          return true;
        else if (index_space != rhs.index_space) // therefore greater than
          return false;
        else
          return field_space < rhs.field_space;
      }
    }

    //--------------------------------------------------------------------------
    inline LogicalPartition& LogicalPartition::operator=(
                                                    const LogicalPartition &rhs)
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.tree_id;
      index_partition = rhs.index_partition;
      field_space = rhs.field_space;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator==(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && 
              (index_partition == rhs.index_partition) && 
              (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator!=(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      return (!((*this) == rhs));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator<(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_partition < rhs.index_partition)
          return true;
        else if (index_partition > rhs.index_partition)
          return false;
        else
          return (field_space < rhs.field_space);
      }
    }

    //--------------------------------------------------------------------------
    inline bool IndexAllocator::operator==(const IndexAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((index_space == rhs.index_space) && (allocator == rhs.allocator));
    }

    //--------------------------------------------------------------------------
    inline bool IndexAllocator::operator<(const IndexAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      if (allocator < rhs.allocator)
        return true;
      else if (allocator > rhs.allocator)
        return false;
      else
        return (index_space < rhs.index_space);
    }

    //--------------------------------------------------------------------------
    inline ptr_t IndexAllocator::alloc(unsigned num_elements /*= 1*/)
    //--------------------------------------------------------------------------
    {
      ptr_t result(allocator->alloc(num_elements));
      return result;
    }

    //--------------------------------------------------------------------------
    inline void IndexAllocator::free(ptr_t ptr, unsigned num_elements /*= 1*/)
    //--------------------------------------------------------------------------
    {
      allocator->free(ptr.value,num_elements);
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator==(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((field_space == rhs.field_space) && (runtime == rhs.runtime));
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator<(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      if (runtime < rhs.runtime)
        return true;
      else if (runtime > rhs.runtime)
        return false;
      else
        return (field_space < rhs.field_space);
    }

    //--------------------------------------------------------------------------
    inline FieldID FieldAllocator::allocate_field(size_t field_size, 
                                FieldID desired_fieldid /*= AUTO_GENERATE_ID*/,
                                CustomSerdezID serdez_id /*=0*/)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(parent, field_space, 
                                     field_size, desired_fieldid, 
                                     false/*local*/, serdez_id); 
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::free_field(FieldID id)
    //--------------------------------------------------------------------------
    {
      runtime->free_field(parent, field_space, id);
    }

    //--------------------------------------------------------------------------
    inline FieldID FieldAllocator::allocate_local_field(size_t field_size,
                                FieldID desired_fieldid /*= AUTO_GENERATE_ID*/,
                                CustomSerdezID serdez_id /*=0*/)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(parent, field_space,
                                     field_size, desired_fieldid, 
                                     true/*local*/, serdez_id);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::allocate_fields(
        const std::vector<size_t> &field_sizes,
        std::vector<FieldID> &resulting_fields, CustomSerdezID serdez_id /*=0*/)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(parent, field_space, 
                               field_sizes, resulting_fields, 
                               false/*local*/, serdez_id);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::free_fields(const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      runtime->free_fields(parent, field_space, to_free);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::allocate_local_fields(
        const std::vector<size_t> &field_sizes,
        std::vector<FieldID> &resulting_fields, CustomSerdezID serdez_id /*=0*/)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(parent, field_space, 
                               field_sizes, resulting_fields, 
                               true/*local*/, serdez_id);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void ArgumentMap::set_point_arg(const PT point[DIM], 
                                           const TaskArgument &arg, 
                                           bool replace/*= false*/)
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);  
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      set_point(dp, arg, replace);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline bool ArgumentMap::remove_point(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      return remove_point(dp);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator==(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        if (p.impl == NULL)
          return (const_value == p.const_value);
        else
          return false;
      }
      else
        return (impl == p.impl);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator<(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        if (p.impl == NULL)
          return (const_value < p.const_value);
        else
          return true;
      }
      else
        return (impl < p.impl);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator!=(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      return !(*this == p);
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator~(RegionFlags f)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(~unsigned(f));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator|(RegionFlags left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(unsigned(left) | unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator&(RegionFlags left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(unsigned(left) & unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator^(RegionFlags left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      return static_cast<RegionFlags>(unsigned(left) ^ unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator|=(RegionFlags &left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l |= r;
      return left = static_cast<RegionFlags>(l);
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator&=(RegionFlags &left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l &= r;
      return left = static_cast<RegionFlags>(l);
    }

    //--------------------------------------------------------------------------
    inline RegionFlags operator^=(RegionFlags &left, RegionFlags right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l ^= r;
      return left = static_cast<RegionFlags>(l);
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& RegionRequirement::add_field(FieldID fid, 
                                             bool instance/*= true*/)
    //--------------------------------------------------------------------------
    {
      privilege_fields.insert(fid);
      if (instance)
        instance_fields.push_back(fid);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& RegionRequirement::add_fields(
                      const std::vector<FieldID>& fids, bool instance/*= true*/)
    //--------------------------------------------------------------------------
    {
      privilege_fields.insert(fids.begin(), fids.end());
      if (instance)
        instance_fields.insert(instance_fields.end(), fids.begin(), fids.end());
      return *this;
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& RegionRequirement::add_flags(RegionFlags new_flags)
    //--------------------------------------------------------------------------
    {
      flags |= new_flags;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline void StaticDependence::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      dependent_fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline IndexSpaceRequirement& TaskLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
      return index_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& TaskLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
      return region_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_field(unsigned idx, FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_wait_handshake(MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_predicate_false_future(Future f)
    //--------------------------------------------------------------------------
    {
      predicate_false_future = f;
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_predicate_false_result(TaskArgument arg)
    //--------------------------------------------------------------------------
    {
      predicate_false_result = arg;
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::set_independent_requirements(bool independent)
    //--------------------------------------------------------------------------
    {
      independent_requirements = independent;
    }

    //--------------------------------------------------------------------------
    inline IndexSpaceRequirement& IndexTaskLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
      return index_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline RegionRequirement& IndexTaskLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
      return region_requirements.back();
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_wait_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_predicate_false_future(Future f)
    //--------------------------------------------------------------------------
    {
      predicate_false_future = f;
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_predicate_false_result(TaskArgument arg)
    //--------------------------------------------------------------------------
    {
      predicate_false_result = arg;
    }

    //--------------------------------------------------------------------------
    inline void IndexTaskLauncher::set_independent_requirements(
                                                               bool independent)
    //--------------------------------------------------------------------------
    {
      independent_requirements = independent;
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_field(FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
      requirement.add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline unsigned CopyLauncher::add_copy_requirements(
                     const RegionRequirement &src, const RegionRequirement &dst)
    //--------------------------------------------------------------------------
    {
      unsigned result = src_requirements.size();
#ifdef DEBUG_LEGION
      assert(result == dst_requirements.size());
#endif
      src_requirements.push_back(src);
      dst_requirements.push_back(dst);
      return result;
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_src_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < src_requirements.size());
#endif
      src_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_dst_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < dst_requirements.size());
#endif
      dst_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_wait_handshake(MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline unsigned IndexCopyLauncher::add_copy_requirements(
                     const RegionRequirement &src, const RegionRequirement &dst)
    //--------------------------------------------------------------------------
    {
      unsigned result = src_requirements.size();
#ifdef DEBUG_LEGION
      assert(result == dst_requirements.size());
#endif
      src_requirements.push_back(src);
      dst_requirements.push_back(dst);
      return result;
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_src_field(unsigned idx,
                                                 FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < src_requirements.size());
#endif
      src_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_dst_field(unsigned idx,
                                                 FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < dst_requirements.size());
#endif
      dst_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_wait_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexCopyLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_wait_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      assert(bar.exists());
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_wait_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::set_argument(TaskArgument arg)
    //--------------------------------------------------------------------------
    {
      argument = arg;
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::set_future(Future f)
    //--------------------------------------------------------------------------
    {
      future = f;
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_wait_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      wait_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_arrival_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      arrive_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_wait_handshake(MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void FillLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::set_argument(TaskArgument arg)
    //--------------------------------------------------------------------------
    {
      argument = arg;
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::set_future(Future f)
    //--------------------------------------------------------------------------
    {
      future = f;
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      fields.insert(fid);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_wait_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      wait_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_arrival_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      assert(pb.exists());
      arrive_barriers.push_back(pb);
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_wait_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void IndexFillLauncher::add_arrival_handshake(
                                                   MPILegionHandshake handshake)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_file(const char *name,
                                            const std::vector<FieldID> &fields,
                                            LegionFileMode m)
    //--------------------------------------------------------------------------
    {
      file_name = name;
      mode = m;
      file_fields = fields;
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::attach_hdf5(const char *name,
                                const std::map<FieldID,const char*> &field_map,
                                LegionFileMode m)
    //--------------------------------------------------------------------------
    {
      file_name = name;
      mode = m;
      field_files = field_map;
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::add_field_pointer(FieldID fid, void *ptr)
    //--------------------------------------------------------------------------
    {
      field_pointers[fid] = ptr;
    }

    //--------------------------------------------------------------------------
    inline void AttachLauncher::set_pitch(unsigned dim, size_t pitch)
    //--------------------------------------------------------------------------
    {
      if (pitches.size() <= dim)
        pitches.resize(dim+1, 0);
      pitches[dim] = pitch;
    }

    //--------------------------------------------------------------------------
    inline void PredicateLauncher::add_predicate(const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      predicates.push_back(pred);
    }

    //--------------------------------------------------------------------------
    inline void TimingLauncher::add_precondition(const Future &f)
    //--------------------------------------------------------------------------
    {
      preconditions.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void MustEpochLauncher::add_single_task(const DomainPoint &point,
                                                   const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      single_tasks.push_back(launcher);
      single_tasks.back().point = point;
    }

    //--------------------------------------------------------------------------
    inline void MustEpochLauncher::add_index_task(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      index_tasks.push_back(launcher);
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                         add_constraint(const SpecializedConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                              add_constraint(const MemoryConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                            add_constraint(const OrderingConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                           add_constraint(const SplittingConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                               add_constraint(const FieldConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                           add_constraint(const DimensionConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                           add_constraint(const AlignmentConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                              add_constraint(const OffsetConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::
                             add_constraint(const PointerConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                                 add_constraint(const ISAConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                           add_constraint(const ProcessorConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                            add_constraint(const ResourceConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                              add_constraint(const LaunchConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
                          add_constraint(const ColocationConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      execution_constraints.add_constraint(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline TaskVariantRegistrar& TaskVariantRegistrar::
             add_layout_constraint_set(unsigned index, LayoutConstraintID desc)
    //--------------------------------------------------------------------------
    {
      layout_constraints.add_layout_constraint(index, desc);
      return *this;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::set_leaf(bool is_leaf /*= true*/)
    //--------------------------------------------------------------------------
    {
      leaf_variant = is_leaf;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::set_inner(bool is_inner /*= true*/)
    //--------------------------------------------------------------------------
    {
      inner_variant = is_inner;
    }

    //--------------------------------------------------------------------------
    inline void TaskVariantRegistrar::set_idempotent(bool is_idemp/*= true*/)
    //--------------------------------------------------------------------------
    {
      idempotent_variant = is_idemp;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Future::get_result(bool silence_warnings) const
    //--------------------------------------------------------------------------
    {
      // Unpack the value using LegionSerialization in case
      // the type has an alternative method of unpacking
      return 
        LegionSerialization::unpack<T>(get_untyped_result(silence_warnings));
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline const T& Future::get_reference(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      return *((const T*)get_untyped_result(silence_warnings));
    }

    //--------------------------------------------------------------------------
    inline const void* Future::get_untyped_pointer(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      return get_untyped_result(silence_warnings);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline Future Future::from_value(Runtime *rt, const T &value)
    //--------------------------------------------------------------------------
    {
      return LegionSerialization::from_value(rt, &value);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline Future Future::from_untyped_pointer(Runtime *rt,
							  const void *buffer,
							  size_t bytes)
    //--------------------------------------------------------------------------
    {
      return LegionSerialization::from_value_helper(rt, buffer, bytes,
						    false /*!owned*/);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T FutureMap::get_result(const DomainPoint &dp, bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      Future f = get_future(dp);
      return f.get_result<T>(silence_warnings);
    }

    //--------------------------------------------------------------------------
    template<typename RT, typename PT, unsigned DIM>
    inline RT FutureMap::get_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      Future f = get_future(dp);
      return f.get_result<RT>();
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline Future FutureMap::get_future(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      return get_future(dp);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void FutureMap::get_void_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      Future f = get_future(dp);
      return f.get_void_result();
    }

    //--------------------------------------------------------------------------
    inline bool IndexIterator::has_next(void) const
    //--------------------------------------------------------------------------
    {
      return (!finished);
    }
    
    //--------------------------------------------------------------------------
    inline ptr_t IndexIterator::next(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!finished);
#endif
      ptr_t result = current_pointer;
      remaining_elmts--;
      if (remaining_elmts > 0)
      {
        current_pointer++;
      }
      else
      {
        finished = !(enumerator->get_next(current_pointer, remaining_elmts));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    inline ptr_t IndexIterator::next_span(size_t& act_count, size_t req_count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!finished);
#endif
      ptr_t result = current_pointer;
      // did we consume the entire span from the enumerator?
      if ((size_t)remaining_elmts <= req_count)
      {
	// yes, limit the actual count to what we had, and get the next span
	act_count = remaining_elmts;
	current_pointer += remaining_elmts;
        finished = !(enumerator->get_next(current_pointer, remaining_elmts));
      }
      else
      {
	// no, just return what was requested
	act_count = req_count;
	current_pointer += req_count;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IndexPartition Runtime::create_index_partition(Context ctx,
        IndexSpace parent, const T& mapping, int part_color /*= AUTO_GENERATE*/)
    //--------------------------------------------------------------------------
    {
      LegionRuntime::Arrays::Rect<T::IDIM> parent_rect = 
        get_index_space_domain(ctx, parent).get_rect<T::IDIM>();
      LegionRuntime::Arrays::Rect<T::ODIM> color_space = 
        mapping.image_convex(parent_rect);
      DomainPointColoring c;
      for (typename T::PointInOutputRectIterator pir(color_space); 
          pir; pir++) 
      {
        LegionRuntime::Arrays::Rect<T::IDIM> preimage = mapping.preimage(pir.p);
#ifdef DEBUG_LEGION
        assert(mapping.preimage_is_dense(pir.p));
#endif
        c[DomainPoint::from_point<T::IDIM>(pir.p)] =
          Domain::from_rect<T::IDIM>(preimage.intersection(parent_rect));
      }
      IndexPartition result = create_index_partition(ctx, parent, 
              Domain::from_rect<T::ODIM>(color_space), c, 
              DISJOINT_KIND, part_color);
#ifdef DEBUG_LEGION
      // We don't actually know if we're supposed to check disjointness
      // so if we're in debug mode then just do it.
      {
        std::set<DomainPoint> current_colors;  
        for (DomainPointColoring::const_iterator it1 = c.begin();
              it1 != c.end(); it1++)
        {
          current_colors.insert(it1->first);
          for (DomainPointColoring::const_iterator it2 = c.begin();
                it2 != c.end(); it2++)
          {
            if (current_colors.find(it2->first) != current_colors.end())
              continue;
            LegionRuntime::Arrays::Rect<T::IDIM> rect1 = 
              it1->second.get_rect<T::IDIM>();
            LegionRuntime::Arrays::Rect<T::IDIM> rect2 = 
              it2->second.get_rect<T::IDIM>();
            if (rect1.overlaps(rect2))
            {
              switch (it1->first.dim)
              {
                case 1:
                  fprintf(stderr, "ERROR: colors %d and %d of partition %d are "
                                  "not disjoint rectangles as they should be!",
                                   (int)(it1->first)[0],
                                   (int)(it2->first)[0], result.id);
                  break;
                case 2:
                  fprintf(stderr, "ERROR: colors (%d, %d) and (%d, %d) of "
                                  "partition %d are not disjoint rectangles "
                                  "as they should be!",
                                  (int)(it1->first)[0], (int)(it1->first)[1],
                                  (int)(it2->first)[0], (int)(it2->first)[1],
                                  result.id);
                  break;
                case 3:
                  fprintf(stderr, "ERROR: colors (%d, %d, %d) and (%d, %d, %d) "
                                  "of partition %d are not disjoint rectangles "
                                  "as they should be!",
                                  (int)(it1->first)[0], (int)(it1->first)[1],
                                  (int)(it1->first)[2], (int)(it2->first)[0],
                                  (int)(it2->first)[1], (int)(it2->first)[2],
                                  result.id);
                  break;
                default:
                  assert(false);
              }
              assert(false);
              exit(ERROR_DISJOINTNESS_TEST_FAILURE);
            }
          }
        }
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<unsigned DIM>
    IndexSpace Runtime::get_index_subspace(Context ctx, 
                IndexPartition p, LegionRuntime::Arrays::Point<DIM> color_point)
    //--------------------------------------------------------------------------
    {
      DomainPoint dom_point = DomainPoint::from_point<DIM>(color_point);
      return get_index_subspace(ctx, p, dom_point);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void Runtime::fill_field(Context ctx, LogicalRegion handle,
                                      LogicalRegion parent, FieldID fid,
                                      const T &value, Predicate pred)
    //--------------------------------------------------------------------------
    {
      fill_field(ctx, handle, parent, fid, &value, sizeof(T), pred);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void Runtime::fill_fields(Context ctx, LogicalRegion handle,
                                       LogicalRegion parent, 
                                       const std::set<FieldID> &fields,
                                       const T &value, Predicate pred)
    //--------------------------------------------------------------------------
    {
      fill_fields(ctx, handle, parent, fields, &value, sizeof(T), pred);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    /*static*/ void Runtime::register_reduction_op(ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
      {
        fprintf(stderr,"ERROR: ReductionOpID zero is reserved.\n");
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_RESERVED_REDOP_ID);
      }
      ReductionOpTable &red_table = Runtime::get_reduction_table(); 
      // Check to make sure we're not overwriting a prior reduction op 
      if (red_table.find(redop_id) != red_table.end())
      {
        fprintf(stderr,"ERROR: ReductionOpID %d has already been used " 
                       "in the reduction table\n",redop_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DUPLICATE_REDOP_ID);
      }
      red_table[redop_id] = 
        Realm::ReductionOpUntyped::create_reduction_op<REDOP>(); 
      // We also have to check to see if there are explicit serialization
      // and deserialization methods on the RHS type for doing fold reductions
      SerdezRedopTable &serdez_red_table = Runtime::get_serdez_redop_table();
      LegionSerialization::register_reduction<REDOP>(serdez_red_table,redop_id);
    }

    //--------------------------------------------------------------------------
    template<typename SERDEZ>
    /*static*/ void Runtime::register_custom_serdez_op(CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      if (serdez_id == 0)
      {
        fprintf(stderr,"ERROR: Custom Serdez ID zero is reserved.\n");
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_RESERVED_SERDEZ_ID);
      }
      SerdezOpTable &serdez_table = Runtime::get_serdez_table();
      // Check to make sure we're not overwriting a prior serdez op
      if (serdez_table.find(serdez_id) != serdez_table.end())
      {
        fprintf(stderr,"ERROR: CustomSerdezID %d has already been used "
                       "in the serdez operation table\n", serdez_id);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_DUPLICATE_SERDEZ_ID);
      }
      serdez_table[serdez_id] =
        Realm::CustomSerdezUntyped::create_custom_serdez<SERDEZ>();
    }

    namespace Internal {
      // Wrapper class for old projection functions
      template<RegionProjectionFnptr FNPTR>
      class RegionProjectionWrapper : public ProjectionFunctor {
      public:
        RegionProjectionWrapper(void) 
          : ProjectionFunctor() { }
        virtual ~RegionProjectionWrapper(void) { }
      public:
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalRegion upper_bound,
                                      const DomainPoint &point)
        {
          return (*FNPTR)(upper_bound, point, runtime); 
        }
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalPartition upper_bound,
                                      const DomainPoint &point)
        {
          assert(false);
          return LogicalRegion::NO_REGION;
        }
        virtual bool is_exclusive(void) const { return false; }
      };
    };

    //--------------------------------------------------------------------------
    template<LogicalRegion (*PROJ_PTR)(LogicalRegion, const DomainPoint&,
                                       Runtime*)>
    /*static*/ ProjectionID Runtime::register_region_function(
                                                            ProjectionID handle)
    //--------------------------------------------------------------------------
    {
      Runtime::preregister_projection_functor(handle,
          new Internal::RegionProjectionWrapper<PROJ_PTR>());
      return handle;
    }

    namespace Internal {
      // Wrapper class for old projection functions
      template<PartitionProjectionFnptr FNPTR>
      class PartitionProjectionWrapper : public ProjectionFunctor {
      public:
        PartitionProjectionWrapper(void)
          : ProjectionFunctor() { }
        virtual ~PartitionProjectionWrapper(void) { }
      public:
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalRegion upper_bound,
                                      const DomainPoint &point)
        {
          assert(false);
          return LogicalRegion::NO_REGION;
        }
        virtual LogicalRegion project(Context ctx, Task *task,
                                      unsigned index,
                                      LogicalPartition upper_bound,
                                      const DomainPoint &point)
        {
          return (*FNPTR)(upper_bound, point, runtime);
        }
        virtual bool is_exclusive(void) const { return false; }
      };
    };

    //--------------------------------------------------------------------------
    template<LogicalRegion (*PROJ_PTR)(LogicalPartition, const DomainPoint&,
                                       Runtime*)>
    /*static*/ ProjectionID Runtime::register_partition_function(
                                                    ProjectionID handle)
    //--------------------------------------------------------------------------
    {
      Runtime::preregister_projection_functor(handle,
          new Internal::PartitionProjectionWrapper<PROJ_PTR>());
      return handle;
    }

    //--------------------------------------------------------------------------
    // Wrapper functions for high-level tasks
    //--------------------------------------------------------------------------

    /**
     * \class LegionTaskWrapper
     * This is a helper class that has static template methods for 
     * wrapping Legion application tasks.  For all tasks we can make
     * wrappers both for normal execution and also for inline execution.
     */
    class LegionTaskWrapper {
    public: 
      // Non-void return type for new legion task types
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*)>
      static void legion_task_wrapper(const void*, size_t, 
                                      const void*, size_t, Processor);
      template<typename T, typename UDT,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*, const UDT&)>
      static void legion_udt_task_wrapper(const void*, size_t, 
                                          const void*, size_t, Processor);
    public:
      // Void return type for new legion task types
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*)>
      static void legion_task_wrapper(const void*, size_t, 
                                      const void*, size_t, Processor);
      template<typename UDT,
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*, const UDT&)>
      static void legion_udt_task_wrapper(const void*, size_t, 
                                          const void*, size_t, Processor);

    public:
      // Do-it-yourself pre/post-ambles for code generators
      static void legion_task_preamble(const void *data,
				       size_t datalen,
				       Processor p,
				       const Task *& task,
				       const std::vector<PhysicalRegion> *& regionsptr,
				       Context& ctx,
				       Runtime *& runtime);
      static void legion_task_postamble(Runtime *runtime, Context ctx,
					const void *retvalptr = NULL,
					size_t retvalsize = 0);
    };
    
    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    void LegionTaskWrapper::legion_task_wrapper(const void *args, 
                                                size_t arglen, 
                                                const void *userdata,
                                                size_t userlen,
                                                Processor p)
    //--------------------------------------------------------------------------
    {
      // Assert that we are returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(T) <= MAX_RETURN_SIZE);
      // Get the high level runtime
      Runtime *runtime = Runtime::get_runtime(p);
      // Read the context out of the buffer
#ifdef DEBUG_LEGION
      assert(arglen == sizeof(InternalContext));
#endif
      InternalContext ctx = *((const InternalContext*)args);

      const std::vector<PhysicalRegion> &regions = ctx->begin_task();

      // Invoke the task with the given context
      T return_value = 
        (*TASK_PTR)(ctx->get_task(), regions, ctx->as_context(), runtime);

      // Send the return value back
      LegionSerialization::end_task<T>(runtime, ctx, &return_value);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*)>
    void LegionTaskWrapper::legion_task_wrapper(const void *args, 
                                                size_t arglen, 
                                                const void *userdata,
                                                size_t userlen,
                                                Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      Runtime *runtime = Runtime::get_runtime(p);

      // Read the context out of the buffer
#ifdef DEBUG_LEGION
      assert(arglen == sizeof(InternalContext));
#endif
      InternalContext ctx = *((const InternalContext*)args);

      const std::vector<PhysicalRegion> &regions = ctx->begin_task(); 

      (*TASK_PTR)(ctx->get_task(), regions, ctx->as_context(), runtime);

      // Send an empty return value back
      ctx->end_task(NULL, 0, false);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    void LegionTaskWrapper::legion_udt_task_wrapper(const void *args,
                                                    size_t arglen, 
                                                    const void *userdata,
                                                    size_t userlen,
                                                    Processor p)
    //--------------------------------------------------------------------------
    {
      // Assert that we are returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(T) <= MAX_RETURN_SIZE);
      // Get the high level runtime
      Runtime *runtime = Runtime::get_runtime(p);

      // Read the context out of the buffer
#ifdef DEBUG_LEGION
      assert(arglen == sizeof(InternalContext));
#endif
      InternalContext ctx = *((const InternalContext*)args);

      const UDT *user_data = reinterpret_cast<const UDT*>(userdata);

      const std::vector<PhysicalRegion> &regions = ctx->begin_task(); 

      // Invoke the task with the given context
      T return_value = (*TASK_PTR)(ctx->get_task(), regions, 
                                   ctx->as_context(), runtime, *user_data);

      // Send the return value back
      LegionSerialization::end_task<T>(runtime, ctx, &return_value);
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    void LegionTaskWrapper::legion_udt_task_wrapper(const void *args,
                                                    size_t arglen, 
                                                    const void *userdata,
                                                    size_t userlen,
                                                    Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      Runtime *runtime = Runtime::get_runtime(p);

      // Read the context out of the buffer
#ifdef DEBUG_LEGION
      assert(arglen == sizeof(InternalContext));
#endif
      InternalContext ctx = *((const InternalContext*)args);

      const UDT *user_data = reinterpret_cast<const UDT*>(userdata);

      const std::vector<PhysicalRegion> &regions = ctx->begin_task(); 

      (*TASK_PTR)(ctx->get_task(), regions, 
                  ctx->as_context(), runtime, *user_data);

      // Send an empty return value back
      ctx->end_task(NULL, 0, false);
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    VariantID Runtime::register_task_variant(
                                          const TaskVariantRegistrar &registrar)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
           LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>);
      return register_variant(registrar, true, NULL/*UDT*/, 0/*sizeof(UDT)*/,
                              realm_desc);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    VariantID Runtime::register_task_variant(
                    const TaskVariantRegistrar &registrar, const UDT &user_data)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
           LegionTaskWrapper::legion_udt_task_wrapper<T,UDT,TASK_PTR>);
      return register_variant(registrar, true, &user_data, sizeof(UDT),
                              realm_desc);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*)>
    VariantID Runtime::register_task_variant(
                                          const TaskVariantRegistrar &registrar)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
            LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
      return register_variant(registrar, false, NULL/*UDT*/, 0/*sizeof(UDT)*/,
                              realm_desc);
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    VariantID Runtime::register_task_variant(
                    const TaskVariantRegistrar &registrar, const UDT &user_data)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
            LegionTaskWrapper::legion_udt_task_wrapper<UDT,TASK_PTR>);
      return register_variant(registrar, false, &user_data, sizeof(UDT),
                              realm_desc);
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    /*static*/ VariantID Runtime::preregister_task_variant(
        const TaskVariantRegistrar &registrar, const char *task_name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
          LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>);
      return preregister_variant(registrar, NULL/*UDT*/, 0/*sizeof(UDT)*/,
                                 realm_desc, true/*ret*/, task_name);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    /*static*/ VariantID Runtime::preregister_task_variant(
                    const TaskVariantRegistrar &registrar, 
                    const UDT &user_data, const char *task_name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
          LegionTaskWrapper::legion_udt_task_wrapper<T,UDT,TASK_PTR>);
      return preregister_variant(registrar, &user_data, sizeof(UDT),
                               realm_desc, true/*ret*/, task_name);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*)>
    /*static*/ VariantID Runtime::preregister_task_variant(
        const TaskVariantRegistrar &registrar, const char *task_name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
            LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
      return preregister_variant(registrar, NULL/*UDT*/,0/*sizeof(UDT)*/,
                             realm_desc, false/*ret*/, task_name);
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    /*static*/ VariantID Runtime::preregister_task_variant(
                    const TaskVariantRegistrar &registrar, 
                    const UDT &user_data, const char *task_name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      CodeDescriptor *realm_desc = new CodeDescriptor(
            LegionTaskWrapper::legion_udt_task_wrapper<UDT,TASK_PTR>);
      return preregister_variant(registrar, &user_data, sizeof(UDT),
                             realm_desc, false/*ret*/, task_name);
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
    //--------------------------------------------------------------------------
    {
      bool check_task_id = true;
      if (id == AUTO_GENERATE_ID)
      {
        id = generate_static_task_id();
        check_task_id = false;
      }
      TaskVariantRegistrar registrar(id, task_name);
      registrar.set_leaf(options.leaf);
      registrar.set_inner(options.inner);
      registrar.set_idempotent(options.idempotent);
      registrar.add_constraint(ProcessorConstraint(proc_kind));
      CodeDescriptor *realm_desc = new CodeDescriptor(
          LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>);
      preregister_variant(registrar, NULL/*UDT*/, 0/*sizeof(UDT)*/,
                          realm_desc, true/*ret*/, task_name, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
    //--------------------------------------------------------------------------
    {
      bool check_task_id = true;
      if (id == AUTO_GENERATE_ID)
      {
        id = generate_static_task_id();
        check_task_id = false;
      }
      TaskVariantRegistrar registrar(id, task_name);
      registrar.set_leaf(options.leaf);
      registrar.set_inner(options.inner);
      registrar.set_idempotent(options.idempotent);
      registrar.add_constraint(ProcessorConstraint(proc_kind));
      CodeDescriptor *realm_desc = new CodeDescriptor(
            LegionTaskWrapper::legion_task_wrapper<TASK_PTR>);
      preregister_variant(registrar, NULL/*UDT*/, 0/*sizeof(UDT)*/,
                          realm_desc, false/*ret*/, task_name, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename UDT,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, Runtime*, const UDT&)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    const UDT &user_data,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
    //--------------------------------------------------------------------------
    {
      bool check_task_id = true;
      if (id == AUTO_GENERATE_ID)
      {
        id = generate_static_task_id();
        check_task_id = false;
      }
      TaskVariantRegistrar registrar(id, task_name);
      registrar.set_leaf(options.leaf);
      registrar.set_inner(options.inner);
      registrar.set_idempotent(options.idempotent);
      registrar.add_constraint(ProcessorConstraint(proc_kind));
      CodeDescriptor *realm_desc = new CodeDescriptor(
            LegionTaskWrapper::legion_udt_task_wrapper<T,UDT,TASK_PTR>);
      preregister_variant(registrar, &user_data, sizeof(UDT),
                          realm_desc, true/*ret*/, task_name, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    template<typename UDT,
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, Runtime*, const UDT&)>
    /*static*/ TaskID Runtime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    const UDT &user_data,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
    //--------------------------------------------------------------------------
    {
      bool check_task_id = true;
      if (id == AUTO_GENERATE_ID)
      {
        id = generate_static_task_id();
        check_task_id = false;
      }
      TaskVariantRegistrar registrar(id, task_name);
      registrar.set_leaf(options.leaf);
      registrar.set_inner(options.inner);
      registrar.set_idempotent(options.idempotent);
      registrar.add_constraint(ProcessorConstraint(proc_kind));
      CodeDescriptor *realm_desc = new CodeDescriptor(
            LegionTaskWrapper::legion_udt_task_wrapper<UDT,TASK_PTR>);
      preregister_variant(registrar, &user_data, sizeof(UDT),
                          realm_desc, false/*ret*/, task_name, check_task_id);
      return id;
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator~(PrivilegeMode p)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(~unsigned(p));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator|(PrivilegeMode left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(unsigned(left) | unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator&(PrivilegeMode left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(unsigned(left) & unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator^(PrivilegeMode left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<PrivilegeMode>(unsigned(left) ^ unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator|=(PrivilegeMode &left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l |= r;
      return left = static_cast<PrivilegeMode>(l);
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator&=(PrivilegeMode &left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l &= r;
      return left = static_cast<PrivilegeMode>(l);
    }

    //--------------------------------------------------------------------------
    inline PrivilegeMode operator^=(PrivilegeMode &left, PrivilegeMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l ^= r;
      return left = static_cast<PrivilegeMode>(l);
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator~(AllocateMode a)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(~unsigned(a));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator|(AllocateMode left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(unsigned(left) | unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator&(AllocateMode left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(unsigned(left) & unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator^(AllocateMode left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      return static_cast<AllocateMode>(unsigned(left) ^ unsigned(right));
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator|=(AllocateMode &left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l |= r;
      return left = static_cast<AllocateMode>(l);
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator&=(AllocateMode &left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l &= r;
      return left = static_cast<AllocateMode>(l);
    }

    //--------------------------------------------------------------------------
    inline AllocateMode operator^=(AllocateMode &left, AllocateMode right)
    //--------------------------------------------------------------------------
    {
      unsigned l = static_cast<unsigned>(left);
      unsigned r = static_cast<unsigned>(right);
      l ^= r;
      return left = static_cast<AllocateMode>(l);
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const LogicalRegion& lr)
    //--------------------------------------------------------------------------
    {
      os << "LogicalRegion(" << lr.tree_id << "," 
         << lr.index_space << "," << lr.field_space << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const LogicalPartition& lp)
    //--------------------------------------------------------------------------
    {
      os << "LogicalPartition(" << lp.tree_id << "," 
         << lp.index_partition << "," << lp.field_space << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const IndexSpace& is)
    //--------------------------------------------------------------------------
    {
      os << "IndexSpace(" << is.id << "," << is.tid << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const IndexPartition& ip)
    //--------------------------------------------------------------------------
    {
      os << "IndexPartition(" << ip.id << "," << ip.tid << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const FieldSpace& fs)
    //--------------------------------------------------------------------------
    {
      os << "FieldSpace(" << fs.id << ")";
      return os;
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const PhaseBarrier& pb)
    //--------------------------------------------------------------------------
    {
      os << "PhaseBarrier(" << pb.phase_barrier << ")";
      return os;
    }

}; // namespace Legion

// This is for backwards compatibility with the old namespace scheme
namespace LegionRuntime {
  namespace HighLevel {
    using namespace LegionRuntime::Arrays;

    typedef Legion::IndexSpace IndexSpace;
    typedef Legion::IndexPartition IndexPartition;
    typedef Legion::FieldSpace FieldSpace;
    typedef Legion::LogicalRegion LogicalRegion;
    typedef Legion::LogicalPartition LogicalPartition;
    typedef Legion::IndexAllocator IndexAllocator;
    typedef Legion::FieldAllocator FieldAllocator;
    typedef Legion::TaskArgument TaskArgument;
    typedef Legion::ArgumentMap ArgumentMap;
    typedef Legion::Predicate Predicate;
    typedef Legion::Lock Lock;
    typedef Legion::LockRequest LockRequest;
    typedef Legion::Grant Grant;
    typedef Legion::PhaseBarrier PhaseBarrier;
    typedef Legion::DynamicCollective DynamicCollective;
    typedef Legion::RegionRequirement RegionRequirement;
    typedef Legion::IndexSpaceRequirement IndexSpaceRequirement;
    typedef Legion::FieldSpaceRequirement FieldSpaceRequirement;
    typedef Legion::Future Future;
    typedef Legion::FutureMap FutureMap;
    typedef Legion::TaskLauncher TaskLauncher;
    typedef Legion::IndexLauncher IndexLauncher;
    typedef Legion::InlineLauncher InlineLauncher;
    typedef Legion::CopyLauncher CopyLauncher;
    typedef Legion::PhysicalRegion PhysicalRegion;
    typedef Legion::IndexIterator IndexIterator;
    typedef Legion::AcquireLauncher AcquireLauncher;
    typedef Legion::ReleaseLauncher ReleaseLauncher;
    typedef Legion::TaskVariantRegistrar TaskVariantRegistrar;
    typedef Legion::MustEpochLauncher MustEpochLauncher;
    typedef Legion::MPILegionHandshake MPILegionHandshake;
    typedef Legion::Mappable Mappable;
    typedef Legion::Task Task;
    typedef Legion::Copy Copy;
    typedef Legion::InlineMapping Inline;
    typedef Legion::Acquire Acquire;
    typedef Legion::Release Release;
    typedef Legion::Mapping::Mapper Mapper;
    typedef Legion::InputArgs InputArgs;
    typedef Legion::TaskConfigOptions TaskConfigOptions;
    typedef Legion::ProjectionFunctor ProjectionFunctor;
    typedef Legion::Runtime Runtime;
    typedef Legion::Runtime HighLevelRuntime; // for backwards compatibility
    typedef Legion::ColoringSerializer ColoringSerializer;
    typedef Legion::DomainColoringSerializer DomainColoringSerializer;
    typedef Legion::Serializer Serializer;
    typedef Legion::Deserializer Deserializer;
    typedef Legion::TaskResult TaskResult;
    typedef Legion::CObjectWrapper CObjectWrapper;
    typedef Legion::ImmovableAutoLock AutoLock;
    typedef Legion::ISAConstraint ISAConstraint;
    typedef Legion::ProcessorConstraint ProcessorConstraint;
    typedef Legion::ResourceConstraint ResourceConstraint;
    typedef Legion::LaunchConstraint LaunchConstraint;
    typedef Legion::ColocationConstraint ColocationConstraint;
    typedef Legion::ExecutionConstraintSet ExecutionConstraintSet;
    typedef Legion::SpecializedConstraint SpecializedConstraint;
    typedef Legion::MemoryConstraint MemoryConstraint;
    typedef Legion::FieldConstraint FieldConstraint;
    typedef Legion::OrderingConstraint OrderingConstraint;
    typedef Legion::SplittingConstraint SplittingConstraint;
    typedef Legion::DimensionConstraint DimensionConstraint;
    typedef Legion::AlignmentConstraint AlignmentConstraint;
    typedef Legion::OffsetConstraint OffsetConstraint;
    typedef Legion::PointerConstraint PointerConstraint;
    typedef Legion::LayoutConstraintSet LayoutConstraintSet;
    typedef Legion::TaskLayoutConstraintSet TaskLayoutConstraintSet;
    typedef Realm::Runtime RealmRuntime;
    typedef Realm::Machine Machine;
    typedef Realm::Domain Domain;
    typedef Realm::DomainPoint DomainPoint;
    typedef Realm::IndexSpaceAllocator IndexSpaceAllocator;
    typedef Realm::RegionInstance PhysicalInstance;
    typedef Realm::Memory Memory;
    typedef Realm::Processor Processor;
    typedef Realm::CodeDescriptor CodeDescriptor;
    typedef Realm::Event Event;
    typedef Realm::Event MapperEvent;
    typedef Realm::UserEvent UserEvent;
    typedef Realm::Reservation Reservation;
    typedef Realm::Barrier Barrier;
    typedef ::legion_reduction_op_id_t ReductionOpID;
    typedef Realm::ReductionOpUntyped ReductionOp;
    typedef ::legion_custom_serdez_id_t CustomSerdezID;
    typedef Realm::CustomSerdezUntyped SerdezOp;
    typedef Realm::Machine::ProcessorMemoryAffinity ProcessorMemoryAffinity;
    typedef Realm::Machine::MemoryMemoryAffinity MemoryMemoryAffinity;
    typedef Realm::ElementMask::Enumerator Enumerator;
    typedef Realm::IndexSpace::FieldDataDescriptor FieldDataDescriptor;
    typedef std::map<CustomSerdezID, 
                     const Realm::CustomSerdezUntyped *> SerdezOpTable;
    typedef std::map<Realm::ReductionOpID, 
            const Realm::ReductionOpUntyped *> ReductionOpTable;
    typedef void (*SerdezInitFnptr)(const ReductionOp*, void *&, size_t&);
    typedef void (*SerdezFoldFnptr)(const ReductionOp*, void *&, size_t&,
                                    const void*, bool);
    typedef std::map<Realm::ReductionOpID, 
                     Legion::SerdezRedopFns> SerdezRedopTable;
    typedef ::legion_address_space_t AddressSpace;
    typedef ::legion_task_priority_t TaskPriority;
    typedef ::legion_color_t Color;
    typedef ::legion_field_id_t FieldID;
    typedef ::legion_trace_id_t TraceID;
    typedef ::legion_mapper_id_t MapperID;
    typedef ::legion_context_id_t ContextID;
    typedef ::legion_instance_id_t InstanceID;
    typedef ::legion_index_space_id_t IndexSpaceID;
    typedef ::legion_index_partition_id_t IndexPartitionID;
    typedef ::legion_index_tree_id_t IndexTreeID;
    typedef ::legion_field_space_id_t FieldSpaceID;
    typedef ::legion_generation_id_t GenerationID;
    typedef ::legion_type_handle TypeHandle;
    typedef ::legion_projection_id_t ProjectionID;
    typedef ::legion_region_tree_id_t RegionTreeID;
    typedef ::legion_distributed_id_t DistributedID;
    typedef ::legion_address_space_id_t AddressSpaceID;
    typedef ::legion_tunable_id_t TunableID;
    typedef ::legion_mapping_tag_id_t MappingTagID;
    typedef ::legion_semantic_tag_t SemanticTag;
    typedef ::legion_variant_id_t VariantID;
    typedef ::legion_unique_id_t UniqueID;
    typedef ::legion_version_id_t VersionID;
    typedef ::legion_task_id_t TaskID;
    typedef ::legion_layout_constraint_id_t LayoutConstraintID;
    typedef std::map<Color,Legion::ColoredPoints<ptr_t> > Coloring;
    typedef std::map<Color,Domain> DomainColoring;
    typedef std::map<Color,std::set<Domain> > MultiDomainColoring;
    typedef std::map<DomainPoint,Legion::ColoredPoints<ptr_t> > PointColoring;
    typedef std::map<DomainPoint,Domain> DomainPointColoring;
    typedef std::map<DomainPoint,std::set<Domain> > MultiDomainPointColoring;
    typedef void (*RegistrationCallbackFnptr)(Machine machine, 
        Runtime *rt, const std::set<Processor> &local_procs);
    typedef LogicalRegion (*RegionProjectionFnptr)(LogicalRegion parent, 
        const DomainPoint&, Runtime *rt);
    typedef LogicalRegion (*PartitionProjectionFnptr)(LogicalPartition parent, 
        const DomainPoint&, Runtime *rt);
    typedef bool (*PredicateFnptr)(const void*, size_t, 
        const std::vector<Future> futures);
    typedef std::map<ProjectionID,RegionProjectionFnptr> 
      RegionProjectionTable;
    typedef std::map<ProjectionID,PartitionProjectionFnptr> 
      PartitionProjectionTable;
    typedef void (*RealmFnptr)(const void*,size_t,
                               const void*,size_t,Processor);
    typedef Legion::Internal::TaskContext* Context; 
  };
};

