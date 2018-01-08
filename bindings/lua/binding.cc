/* Copyright 2018 Stanford University
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

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <string>

#include <luabind/luabind.hpp>
#include <luabind/operator.hpp>
#include <luabind/object.hpp>
extern "C" 
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <cstdio>
#include <cstring>
}

#include "legion.h"
#include "legion_types.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "accessor.h"

#ifdef WITH_TERRA
#include "terra.h"
#include "binding.h"
#endif

#ifdef PROF_BINDING
#include <time.h>
#endif

// should match with legionlib.lua
#define PRIM_TYPE_FLOAT 0
#define PRIM_TYPE_DOUBLE 1
#define PRIM_TYPE_INT 2

#define RED_TYPE_PLUS  100
#define RED_TYPE_MINUS 200
#define RED_TYPE_TIMES 300

#define DEF_VECTOR_TYPE(...)                                \
  class_<std::vector< __VA_ARGS__ > >("vec")                \
  .def(constructor<>())                                     \
  .def(constructor<int>())                                  \
  .def("push_back", &std::vector< __VA_ARGS__ >::push_back) \
  .def("at",                                                \
       (std::vector< __VA_ARGS__ >::reference               \
        (std::vector< __VA_ARGS__ >::*)                     \
        (std::vector< __VA_ARGS__ >::size_type))            \
       &std::vector< __VA_ARGS__ >::at)                     \
  .def("size", &std::vector< __VA_ARGS__ >::size)

#define DEF_SET_TYPE(...)                                               \
  class_<std::pair<std::set< __VA_ARGS__ >::iterator, bool> >("set_insert_return_type"), \
    class_<std::set< __VA_ARGS__ > >("set")                             \
    .def(constructor<>())                                               \
    .def("insert",                                                      \
         (std::pair<std::set< __VA_ARGS__ >::iterator, bool>            \
          (std::set< __VA_ARGS__ >::*)(const  __VA_ARGS__ &))           \
         &std::set< __VA_ARGS__ >::insert)                              \
    .def("erase",                                                       \
         (std::set< __VA_ARGS__ >::size_type                            \
          (std::set< __VA_ARGS__ >::*)(const  __VA_ARGS__ &))           \
         &std::set< __VA_ARGS__ >::erase)                               \
    .def("size", &std::set< __VA_ARGS__ >::size)

extern "C" int init(lua_State* L);

#ifdef WITH_TERRA
extern "C" void* create_index_iterator(struct TLogicalRegion);
extern "C" void destroy_index_iterator(void*);
extern "C" unsigned next(void*);
extern "C" int has_next(void*);
#endif

template<typename T>
inline static T convert(const void* ptr)
{
  return reinterpret_cast<T>(const_cast<void*>(ptr));
}

void greet()
{
  std::cout << "hello world!" << std::endl;
}

const LegionRuntime::HighLevel::Predicate& get_true_pred()
{
  return LegionRuntime::HighLevel::Predicate::TRUE_PRED;
}

const LegionRuntime::HighLevel::Predicate& get_false_pred()
{
  return LegionRuntime::HighLevel::Predicate::FALSE_PRED;
}

luabind::scope register_HighLevelRuntime()
{
  using namespace luabind;
  using namespace LegionRuntime::HighLevel;

  return
    class_<HighLevelRuntime>("HighLevelRuntime")
    .def("create_index_space",
         (IndexSpace(HighLevelRuntime::*)(Context, size_t))
         &HighLevelRuntime::create_index_space)
    .def("create_index_space",
         (IndexSpace(HighLevelRuntime::*)(Context, Domain))
         &HighLevelRuntime::create_index_space)
    .def("create_field_space", &HighLevelRuntime::create_field_space)
    .def("create_logical_region", &HighLevelRuntime::create_logical_region)
    .def("execute_task", 
        (Future(HighLevelRuntime::*)
         (Context, Processor::TaskFuncID,
          const std::vector<IndexSpaceRequirement>&,
          const std::vector<FieldSpaceRequirement>&,
          const std::vector<RegionRequirement>&,
          const TaskArgument&,
          const Predicate&,
          MapperID,
          MappingTagID))
        &HighLevelRuntime::execute_task)
    .def("execute_task", 
        (Future(HighLevelRuntime::*)
         (Context, const TaskLauncher&))
        &HighLevelRuntime::execute_task)
    .def("execute_index_space", 
        (FutureMap(HighLevelRuntime::*)
         (Context, const IndexLauncher&))
        &HighLevelRuntime::execute_index_space)
    .def("destroy_logical_region", &HighLevelRuntime::destroy_logical_region)
    .def("destroy_index_space", &HighLevelRuntime::destroy_index_space)
    .def("destroy_field_space", &HighLevelRuntime::destroy_field_space)
    .def("create_index_allocator", &HighLevelRuntime::create_index_allocator)
    .def("create_field_allocator", &HighLevelRuntime::create_field_allocator)
    .def("create_index_partition",
         (IndexPartition(HighLevelRuntime::*)
          (Context, IndexSpace, const Coloring&, bool, int))
         &HighLevelRuntime::create_index_partition)
    .def("create_index_partition",
         (IndexPartition(HighLevelRuntime::*)
          (Context, IndexSpace, Domain, const DomainColoring&, bool, int))
         &HighLevelRuntime::create_index_partition)
    .def("create_index_partition",
         (IndexPartition(HighLevelRuntime::*)
          (Context, IndexSpace, const Blockify<1>&, int))
         &HighLevelRuntime::create_index_partition)
    .def("create_index_partition",
         (IndexPartition(HighLevelRuntime::*)
          (Context, IndexSpace, const Blockify<2>&, int))
         &HighLevelRuntime::create_index_partition)
    .def("create_index_partition",
         (IndexPartition(HighLevelRuntime::*)
          (Context, IndexSpace, const Blockify<3>&, int))
         &HighLevelRuntime::create_index_partition)
    .def("get_index_partition_color_space",
         &HighLevelRuntime::get_index_partition_color_space)
    .def("get_index_subspace",
         (IndexSpace(HighLevelRuntime::*)(Context, IndexPartition, Color))
         &HighLevelRuntime::get_index_subspace)
    .def("get_logical_partition", &HighLevelRuntime::get_logical_partition)
    .def("get_logical_partition_by_tree",
         &HighLevelRuntime::get_logical_partition_by_tree)
    .def("get_logical_subregion_by_color",
         &HighLevelRuntime::get_logical_subregion_by_color)
    .def("create_argument_map", &HighLevelRuntime::create_argument_map)
    .def("map_region", 
         (PhysicalRegion(HighLevelRuntime::*)
          (Context, const RegionRequirement&, MapperID, MappingTagID))
         &HighLevelRuntime::map_region)
    .def("map_region", 
         (PhysicalRegion(HighLevelRuntime::*)
          (Context, unsigned, MapperID, MappingTagID))
         &HighLevelRuntime::map_region)
    .def("unmap_region", &HighLevelRuntime::unmap_region)
    .def("map_all_regions", &HighLevelRuntime::map_all_regions)
    .def("unmap_all_regions", &HighLevelRuntime::unmap_all_regions)
    .scope
    [
     def("set_top_level_task_id", &HighLevelRuntime::set_top_level_task_id)
     ]
    ;
}

namespace
{
  using namespace LegionRuntime::HighLevel;

  RegionRequirement make_region_requirement(void)
  {
    return RegionRequirement();
  }

  RegionRequirement
  make_region_requirement(LogicalRegion _handle,
                          const std::set<FieldID> &privilege_fields,
                          const std::vector<FieldID> &instance_fields,
                          PrivilegeMode _priv, CoherenceProperty _prop,
                          LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(_handle, privilege_fields, instance_fields,
                             _priv, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalPartition pid, ProjectionID _proj,
                          const std::set<FieldID> &privilege_fields,
                          const std::vector<FieldID> &instance_fields,
                          PrivilegeMode _priv, CoherenceProperty _prop,
                          LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(pid, _proj, privilege_fields, instance_fields,
                             _priv, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalRegion _handle, ProjectionID _proj,
                          const std::set<FieldID> &privilege_fields,
                          const std::vector<FieldID> &instance_fields,
                          PrivilegeMode _priv, CoherenceProperty _prop,
                          LogicalRegion _parent, MappingTagID _tag,
                          bool _verified)
  {
    return RegionRequirement(_handle, _proj, privilege_fields, instance_fields,
                             _priv, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalRegion _handle,
                          const std::set<FieldID> &privilege_fields,
                          const std::vector<FieldID> &instance_fields,
                          ReductionOpID op, CoherenceProperty _prop,
                          LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(_handle, privilege_fields, instance_fields,
                             op, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalPartition pid, ProjectionID _proj, 
                          const std::set<FieldID> &privilege_fields,
                          const std::vector<FieldID> &instance_fields,
                          ReductionOpID op, CoherenceProperty _prop,
                          LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(pid, _proj, privilege_fields, instance_fields,
                             op, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalRegion _handle, ProjectionID _proj,
                          const std::set<FieldID> &privilege_fields,
                          const std::vector<FieldID> &instance_fields,
                          ReductionOpID op, CoherenceProperty _prop,
                          LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(_handle, _proj, privilege_fields, instance_fields,
                             op, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalRegion _handle, PrivilegeMode _priv, 
                          CoherenceProperty _prop, LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(_handle, _priv, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalPartition pid, ProjectionID _proj,
                          PrivilegeMode _priv, CoherenceProperty _prop,
                          LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(pid, _proj, _priv, _prop, _parent, _tag, _verified);
  }  

  RegionRequirement
  make_region_requirement(LogicalRegion _handle, ReductionOpID op, 
                          CoherenceProperty _prop, LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(_handle, op, _prop, _parent, _tag, _verified);
  }

  RegionRequirement
  make_region_requirement(LogicalPartition pid, ProjectionID _proj,
                          ReductionOpID op, CoherenceProperty _prop,
                          LogicalRegion _parent,
                          MappingTagID _tag, bool _verified)
  {
    return RegionRequirement(pid, _proj, op, _prop, _parent, _tag, _verified);
  }  

  Point<1> make_point1(int x) { return Point<1>(x); }
  Point<2> make_point2(int x, int y)
  {
    int vals[2];
    vals[0] = x; vals[1] = y;
    return Point<2>(vals);
  }
  Point<3> make_point3(int x, int y, int z)
  {
    int vals[3];
    vals[0] = x; vals[1] = y; vals[2] = z;
    return Point<3>(vals);
  }

  template<int DIM>
  Rect<DIM> make_rect(Point<DIM>& low, Point<DIM>& high)
  {
    return Rect<DIM>(low, high);
  }

  Blockify<1> make_blockify(int x)
  {
    return Blockify<1>(make_point1(x));
  }
  Blockify<2> make_blockify(int x, int y)
  {
    return Blockify<2>(make_point2(x, y));
  }
  Blockify<3> make_blockify(int x, int y, int z)
  {
    return Blockify<3>(make_point3(x, y, z));
  }

  luabind::object get_coords_from_domain_point(lua_State* L,
                                               const DomainPoint& dp)
  {
    using namespace luabind;
    object obj = newtable(L);
    for (int i = 0; i < dp.dim; ++i)
    {
      obj[i + 1] = object(L, dp.point_data[i]);
    }
    return obj;
  }
}

#define ACCUMULATOR(NAME, T, U, OP1, OP2, ID)                           \
  class NAME {                                                          \
  public:                                                               \
  typedef T LHS, RHS;                                                   \
  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);       \
  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);      \
  static const T identity;                                              \
  };                                                                    \
                                                                        \
  const T NAME::identity = ID;                                          \
                                                                        \
  template <>                                                           \
  void NAME::apply<true>(LHS &lhs, RHS rhs)                             \
  {                                                                     \
    lhs OP2 rhs;                                                        \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void NAME::apply<false>(LHS &lhs, RHS rhs)                            \
  {                                                                     \
    U *target = (U *)&(lhs);                                            \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = oldval.as_T OP1 rhs;                                \
    } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void NAME::fold<true>(RHS &rhs1, RHS rhs2)                            \
  {                                                                     \
    rhs1 OP2 rhs2;                                                      \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void NAME::fold<false>(RHS &rhs1, RHS rhs2)                           \
  {                                                                     \
    U *target = (U *)&rhs1;                                             \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = oldval.as_T OP1 rhs2;                               \
    } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
  }                                                                     

ACCUMULATOR(PlusAccumulatorInt, int, int, +, +=, 0)
ACCUMULATOR(PlusAccumulatorDouble, double, size_t, +, +=, 0.0)
ACCUMULATOR(PlusAccumulatorFloat, float, int, +, +=, 0.0f)

ACCUMULATOR(MinusAccumulatorInt, int, int, -, -=, 0)
ACCUMULATOR(MinusAccumulatorDouble, double, size_t, -, -=, 0.0)
ACCUMULATOR(MinusAccumulatorFloat, float, int, -, -=, 0.0f)

ACCUMULATOR(TimesAccumulatorInt, int, int, *, *=, 0)
ACCUMULATOR(TimesAccumulatorDouble, double, size_t, *, *=, 0.0)
ACCUMULATOR(TimesAccumulatorFloat, float, int, *, *=, 0.0f)

luabind::scope register_highlevel()
{
  using namespace luabind;
  using namespace LegionRuntime;
  using namespace LegionRuntime::HighLevel;
  using namespace LegionRuntime::LowLevel;
  using namespace Accessor;
  using namespace AccessorType;

  return namespace_("HighLevel")
    [
     //
     // Classes
     //
     class_<RegionAccessor<Generic> >("RegionAccessorGeneric")
     .def("typeify_float", &RegionAccessor<Generic>::typeify<float>)
     .def("typeify_double", &RegionAccessor<Generic>::typeify<double>)
     .def("typeify_int", &RegionAccessor<Generic>::typeify<int>)
     .def("get_untyped_field_accessor",
          (RegionAccessor<Generic>
           (RegionAccessor<Generic>::*)(size_t, size_t))
          &RegionAccessor<Generic>::get_untyped_field_accessor)
     ,
     class_<RegionAccessor<Generic, float> >("RegionAccessorFloat")
     .def("read", &RegionAccessor<Generic, float>::read<ptr_t>)
     .def("write", &RegionAccessor<Generic, float>::write<ptr_t>)
     .def("read_at_point",
          &RegionAccessor<Generic, float>::read<const DomainPoint&>)
     .def("write_at_point",
          &RegionAccessor<Generic, float>::write<const DomainPoint&>)
     .def("reduce_plus",
          &RegionAccessor<Generic, float>::reduce<PlusAccumulatorFloat>)
     .def("reduce_minus",
          &RegionAccessor<Generic, float>::reduce<MinusAccumulatorFloat>)
     .def("reduce_times",
          &RegionAccessor<Generic, float>::reduce<TimesAccumulatorFloat>)
     .def("convert_plus",
          &RegionAccessor<Generic, float>::
          convert<AccessorType::ReductionFold<PlusAccumulatorFloat> >)
     .def("convert_minus",
          &RegionAccessor<Generic, float>::
          convert<AccessorType::ReductionFold<MinusAccumulatorFloat> >)
     .def("convert_times",
          &RegionAccessor<Generic, float>::
          convert<AccessorType::ReductionFold<TimesAccumulatorFloat> >)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<PlusAccumulatorFloat>, float> >("ReductionFoldFloatPlus")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<PlusAccumulatorFloat>, float>::reduce)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<MinusAccumulatorFloat>, float> >("ReductionFoldFloatMinus")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<MinusAccumulatorFloat>, float>::reduce)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<TimesAccumulatorFloat>, float> >("ReductionFoldFloatTimes")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<TimesAccumulatorFloat>, float>::reduce)
     ,
     class_<RegionAccessor<Generic, double> >("RegionAccessorDouble")
     .def("read", &RegionAccessor<Generic, double>::read<ptr_t>)
     .def("write", &RegionAccessor<Generic, double>::write<ptr_t>)
     .def("read_at_point",
          &RegionAccessor<Generic, double>::read<const DomainPoint&>)
     .def("write_at_point",
          &RegionAccessor<Generic, double>::write<const DomainPoint&>)
     .def("reduce_plus",
          &RegionAccessor<Generic, double>::reduce<PlusAccumulatorDouble>)
     .def("reduce_minus",
          &RegionAccessor<Generic, double>::reduce<MinusAccumulatorDouble>)
     .def("reduce_times",
          &RegionAccessor<Generic, double>::reduce<TimesAccumulatorDouble>)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<PlusAccumulatorDouble>, double> >("ReductionFoldDoublePlus")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<PlusAccumulatorDouble>, double>::reduce)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<MinusAccumulatorDouble>, double> >("ReductionFoldDoubleMinus")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<MinusAccumulatorDouble>, double>::reduce)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<TimesAccumulatorDouble>, double> >("ReductionFoldDoubleTimes")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<TimesAccumulatorDouble>, double>::reduce)
     ,
     class_<RegionAccessor<Generic, int> >("RegionAccessorInt")
     .def("read", &RegionAccessor<Generic, int>::read<ptr_t>)
     .def("write", &RegionAccessor<Generic, int>::write<ptr_t>)
     .def("read_at_point",
          &RegionAccessor<Generic, int>::read<const DomainPoint&>)
     .def("write_at_point",
          &RegionAccessor<Generic, int>::write<const DomainPoint&>)
     .def("reduce_plus",
          &RegionAccessor<Generic, int>::reduce<PlusAccumulatorInt>)
     .def("reduce_minus",
          &RegionAccessor<Generic, int>::reduce<MinusAccumulatorInt>)
     .def("reduce_times",
          &RegionAccessor<Generic, int>::reduce<TimesAccumulatorInt>)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<PlusAccumulatorInt>, int> >("ReductionFoldIntPlus")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<PlusAccumulatorInt>, int>::reduce)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<MinusAccumulatorInt>, int> >("ReductionFoldIntMinus")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<MinusAccumulatorInt>, int>::reduce)
     ,
     class_<RegionAccessor<AccessorType::ReductionFold<TimesAccumulatorInt>, int> >("ReductionFoldIntTimes")
     .def("reduce",
          &RegionAccessor<AccessorType::ReductionFold<TimesAccumulatorInt>, int>::reduce)
     ,
     class_<ArgumentMap>("ArgumentMap")
     .def("set_point", &ArgumentMap::set_point)
     .def("set_point_arg_I1D", &ArgumentMap::set_point_arg<int, 1>)
     .def("set_point_arg_I2D", &ArgumentMap::set_point_arg<int, 2>)
     .def("set_point_arg_I3D", &ArgumentMap::set_point_arg<int, 3>)
     ,
     class_<Coloring>("Coloring")
     .def(constructor<>())
     .def("at", &Coloring::operator[])
     .def("size", &Coloring::size)
     ,
     class_<ColoredPoints<ptr_t> >("ColoredPoints")
     .def(constructor<>())
     .def_readwrite("points", &ColoredPoints<ptr_t>::points)
     .def_readwrite("ranges", &ColoredPoints<ptr_t>::ranges)
     ,
     class_<FieldAllocator>("FieldAllocator")
     .def("allocate_field", &FieldAllocator::allocate_field)
     .def("free_field", &FieldAllocator::free_field)
     .def("allocate_fields", &FieldAllocator::allocate_fields)
     .def("free_fields", &FieldAllocator::free_fields)
     .def("get_field_space", &FieldAllocator::get_field_space)
     ,
     class_<FieldID>("FieldID")
     .scope[ DEF_VECTOR_TYPE(FieldID), DEF_SET_TYPE(FieldID) ]
     ,
     class_<FieldSpace>("FieldSpace")
     .def("get_id", &FieldSpace::get_id)
     ,
     class_<FieldSpaceRequirement>("FieldSpaceRequirement")
     .def(constructor<>())
     .def(constructor<FieldSpace, AllocateMode, bool>())
     .scope[ DEF_VECTOR_TYPE(FieldSpaceRequirement) ]
     ,
     class_<Future>("Future")
     .def("get_void_result", &Future::get_void_result)
     .def("get_result_int", &Future::get_result<int>)
     .def("get_result_double", &Future::get_result<double>)
     ,
     class_<FutureMap>("FutureMap")
     .def("wait_all_results", &FutureMap::wait_all_results)
     ,
     class_<IndexAllocator>("IndexAllocator")
     .def("alloc", &IndexAllocator::alloc)
     .def("free", &IndexAllocator::free)
     ,
     class_<IndexIterator>("IndexIterator")
     .def(constructor<LogicalRegion>())
     .def(constructor<IndexSpace>())
     .def(constructor<const IndexIterator&>())
     .def("has_next", &IndexIterator::has_next)
     .def("next", &IndexIterator::next)
     ,
     class_<IndexSpace>("IndexSpace")
     .def_readwrite("id", &IndexSpace::id)
     ,
     class_<IndexSpaceRequirement>("IndexSpaceRequirement")
     .def(constructor<>())
     .def(constructor<IndexSpace, AllocateMode, IndexSpace, bool>())
     .scope[ DEF_VECTOR_TYPE(IndexSpaceRequirement) ]
     ,
     class_<InputArgs>("InputArgs")
     ,
     class_<LogicalPartition>("LogicalPartition")
     .def("get_index_partition", &LogicalPartition::get_index_partition)
     .def("get_field_space", &LogicalPartition::get_field_space)
     .def("get_tree_id", &LogicalPartition::get_tree_id)
     ,
     class_<LogicalRegion>("LogicalRegion")
     .def("get_index_space", &LogicalRegion::get_index_space)
     .def("get_field_space", &LogicalRegion::get_field_space)
     .def("get_tree_id", &LogicalRegion::get_tree_id)
     ,
     class_<PhysicalRegion>("PhysicalRegion")
     .def("get_accessor", &PhysicalRegion::get_accessor)
     .def("get_field_accessor", &PhysicalRegion::get_field_accessor)
     .def("wait_until_valid", &PhysicalRegion::wait_until_valid)
     .def("is_valid", &PhysicalRegion::is_valid)
     .def("get_logical_region", &PhysicalRegion::get_logical_region)
     .scope[ DEF_VECTOR_TYPE(PhysicalRegion) ]
     ,
     class_<IndexLauncher>("IndexLauncher")
     .def(constructor<>())
     .def(constructor<Processor::TaskFuncID, Domain,
          TaskArgument, const ArgumentMap,
          Predicate, bool, MapperID, MappingTagID>())
     .def("add_region_requirement",
          &IndexLauncher::add_region_requirement)
     .def("add_index_requirement",
          &IndexLauncher::add_index_requirement)
     .def_readwrite("predicate",
                    &IndexLauncher::predicate)
     .def_readwrite("must_parallelism",
                    &IndexLauncher::must_parallelism)
     .def_readwrite("map_id",
                    &IndexLauncher::map_id)
     .def_readwrite("tag",
                    &IndexLauncher::tag)
     ,
     class_<TaskLauncher>("TaskLauncher")
     .def(constructor<>())
     .def(constructor<Processor::TaskFuncID, TaskArgument,
          Predicate, MapperID, MappingTagID>())
     .def("add_region_requirement",
          &TaskLauncher::add_region_requirement)
     .def("add_index_requirement",
          &TaskLauncher::add_index_requirement)
     .def_readwrite("predicate",
                    &TaskLauncher::predicate)
     .def_readwrite("map_id",
                    &TaskLauncher::map_id)
     .def_readwrite("tag",
                    &TaskLauncher::tag)
     ,
     class_<Predicate>("Predicate")
     .scope
     [
      def("TRUE_PRED", &get_true_pred)
      ,
      def("FALSE_PRED", &get_false_pred)
      ]
     ,
     class_<RegionRequirement>("RegionRequirement")
     .def("add_field", &RegionRequirement::add_field)
     .def_readwrite("redop", &RegionRequirement::redop)
     .def_readwrite("prop", &RegionRequirement::prop)
     .def_readwrite("privilege", &RegionRequirement::privilege)
     .scope[ def("make",
                 (RegionRequirement(*)(void))&make_region_requirement)
             ,
             def("make",
                 (RegionRequirement(*)
                  (LogicalRegion,
                   const std::set<FieldID>&,
                   const std::vector<FieldID>&,
                   PrivilegeMode, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make",
                 (RegionRequirement(*)
                  (LogicalPartition, ProjectionID,
                   const std::set<FieldID>&,
                   const std::vector<FieldID>&,
                   PrivilegeMode, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make",
                 (RegionRequirement(*)
                  (LogicalRegion, ProjectionID,
                   const std::set<FieldID>&,
                   const std::vector<FieldID>&,
                   PrivilegeMode, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make",
                 (RegionRequirement(*)
                  (LogicalRegion,
                   PrivilegeMode, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make",
                 (RegionRequirement(*)
                  (LogicalPartition, ProjectionID,
                   PrivilegeMode, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make_with_reduction_op",
                 (RegionRequirement(*)
                  (LogicalRegion,
                   const std::set<FieldID>&,
                   const std::vector<FieldID>&,
                   ReductionOpID, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make_with_reduction_op",
                 (RegionRequirement(*)
                  (LogicalPartition, ProjectionID,
                   const std::set<FieldID>&,
                   const std::vector<FieldID>&,
                   ReductionOpID, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make_with_reduction_op",
                 (RegionRequirement(*)
                  (LogicalRegion, ProjectionID,
                   const std::set<FieldID>&,
                   const std::vector<FieldID>&,
                   ReductionOpID, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make_with_reduction_op",
                 (RegionRequirement(*)
                  (LogicalRegion,
                   ReductionOpID, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)
             ,
             def("make_with_reduction_op",
                 (RegionRequirement(*)
                  (LogicalPartition, ProjectionID,
                   ReductionOpID, CoherenceProperty,
                   LogicalRegion, MappingTagID, bool))
                 &make_region_requirement)             
             ]
     .scope[ DEF_VECTOR_TYPE(RegionRequirement) ]
     ,
     class_<SingleTask>("SingleTask")
     ,
     class_<TaskArgument>("TaskArgument")
     .def(constructor<const void*, size_t>())
     .scope[ DEF_VECTOR_TYPE(TaskArgument) ]
     ,
     register_HighLevelRuntime()
     ,
     namespace_("Point")
     [
      class_<Point<1> >("_1")
      ,
      class_<Point<2> >("_2")
      ,
      class_<Point<3> >("_3")
      ,
      // .def(constructor<int>())
      // .def("at", &Point<1>::operator[])
      def("make",
          (Point<1>(*)(int))&make_point1)
      ,
      def("make",
          (Point<2>(*)(int, int))&make_point2)
      ,
      def("make",
          (Point<3>(*)(int, int, int))&make_point3)
      ]
     ,
     namespace_("Rect")
     [
      class_<Rect<1> >("_1")
      ,
      class_<Rect<2> >("_2")
      ,
      class_<Rect<3> >("_3")
      ,
      def("make",
          (Rect<1>(*)(Point<1>&, Point<1>&))&make_rect<1>)
      ,
      def("make",
          (Rect<2>(*)(Point<2>&, Point<2>&))&make_rect<2>)
      ,
      def("make",
          (Rect<3>(*)(Point<3>&, Point<3>&))&make_rect<3>)
      // .def(constructor<const Point<1>, const Point<1> >())
      // .def_readwrite("lo", &Rect<1>::lo)
      // .def_readwrite("hi", &Rect<1>::hi)
      ]
     ,
     namespace_("Blockify")
     [
      class_<Blockify<1> >("_1")
      // .def(constructor<Point<1> >())
      // .def("preimage", &Blockify<1>::preimage)
      ,
      class_<Blockify<2> >("_2")
      ,
      class_<Blockify<3> >("_3")
      ,
      def("make",
           (Blockify<1>(*)(int))make_blockify)
      ,
      def("make",
          (Blockify<2>(*)(int, int))make_blockify)
      ,
      def("make",
          (Blockify<3>(*)(int, int, int))make_blockify)
      ]
     // ,
     // namespace_("GenericPointInRectIterator")
     // [
     //  class_<GenericPointInRectIterator<1> >("_1")
     //  .def(constructor<const Rect<1> >())
     //  .def("step", &GenericPointInRectIterator<1>::step)
     //  .def_readwrite("p", &GenericPointInRectIterator<1>::p)
     //  .def_readwrite("any_left", &GenericPointInRectIterator<1>::any_left)
     //  ]
     ,
     
     // Enumerations
     //
     // PrivilegeMode
     class_<PrivilegeMode>("PrivilegeMode")
     .enum_("constants")
     [
      value("NO_ACCESS", NO_ACCESS)
      ,
      value("READ_ONLY", READ_ONLY)
      ,
      value("READ_WRITE", READ_WRITE)
      ,
      value("WRITE_ONLY", WRITE_ONLY)
      ,
      value("REDUCE", REDUCE)
      ]
     ,
     // AllocateMode
     class_<AllocateMode>("AllocateMode")
     .enum_("constants")
     [
      value("NO_MEMORY", NO_MEMORY)
      ,
      value("ALLOCABLE", ALLOCABLE)
      ,
      value("FREEABLE", FREEABLE)
      ,
      value("MUTABLE", MUTABLE)
      ,
      value("REGION_CREATION", REGION_CREATION)
      ,
      value("REGION_DELETION", REGION_DELETION)
      ,
      value("ALL_MEMORY", ALL_MEMORY)
      ]
     ,
     // CoherenceProperty
     class_<CoherenceProperty>("CoherenceProperty")
     .enum_("constants")
     [
      value("EXCLUSIVE", EXCLUSIVE)
      ,
      value("ATOMIC", ATOMIC)
      ,
      value("SIMULTANEOUS", SIMULTANEOUS)
      ,
      value("RELAXED", RELAXED)
      ]
     ];
}

luabind::scope register_lowlevel()
{
  using namespace luabind;
  using namespace LegionRuntime;
  using namespace LegionRuntime::LowLevel;

  return namespace_("LowLevel")
    [
     class_<Domain>("Domain")
     .def(constructor<IndexSpace>())
     .scope
     [
      def("from_rect",
          (Domain(*)(Arrays::Rect<1>))&Domain::from_rect<1>)
      ,
      def("from_rect",
          (Domain(*)(Arrays::Rect<2>))&Domain::from_rect<2>)
      ,
      def("from_rect",
          (Domain(*)(Arrays::Rect<3>))&Domain::from_rect<3>)
      ]
     ,
     class_<DomainPoint>("DomainPoint")
     .def(constructor<int>())
     .def_readwrite("dim", &DomainPoint::dim)
     .def("get_index", &DomainPoint::get_index)
     .def("get_coords",
          &get_coords_from_domain_point)
     .scope
     [
      def("from_point",
          (DomainPoint(*)(Arrays::Point<1>))&DomainPoint::from_point<1>)
      ,
      def("from_point",
          (DomainPoint(*)(Arrays::Point<2>))&DomainPoint::from_point<2>)
      ,
      def("from_point",
          (DomainPoint(*)(Arrays::Point<3>))&DomainPoint::from_point<3>)
      ]
     ,
     class_<Processor>("Processor")
     .enum_("constants")
     [
      value("TOC_PROC", Processor::TOC_PROC)
      ,
      value("LOC_PROC", Processor::LOC_PROC)
      ,
      value("UTIL_PROC", Processor::UTIL_PROC)
      ]
     ];
}

void print_ptr(ptr_t& ptr)
{
  std::cout << ptr.value << std::endl;
}

luabind::class_<ptr_t> register_ptr_t()
{
  using namespace luabind;

  return
    class_<ptr_t>("ptr_t")
    // constructors
    .def(constructor<>())
    .def(constructor<const ptr_t&>())
    .def(constructor<unsigned>())
    // operators
    .def(self + other<const ptr_t&>())
    .def(self + other<unsigned>())
    .def(self + other<int>())
    .def(self - other<const ptr_t&>())
    .def(self - other<unsigned>())
    .def(self - other<int>())
    .def(self == other<const ptr_t&>())
    .def(self < other<const ptr_t&>())
    // methods
    .def("is_null", &ptr_t::is_null)
    .def("nil", &ptr_t::nil)
    // methods with different names assigned
    .def("inc",
         (ptr_t&(ptr_t::*)(void))
         &ptr_t::operator++)
    .def("dec", 
         (ptr_t&(ptr_t::*)(void))
         &ptr_t::operator--)
    // fields
    .def_readwrite("value", &ptr_t::value)
    .scope
    [
     DEF_SET_TYPE(ptr_t),
     DEF_SET_TYPE(std::pair<ptr_t, ptr_t>),
     class_<std::pair<ptr_t, ptr_t> >("pair")
     .def(constructor<const ptr_t&, const ptr_t&>())
     ]
    ;
}

namespace BindingLib
{
  using namespace LegionRuntime::LowLevel;
  using namespace LegionRuntime::HighLevel;

  struct Utility
  {
    static IndexSpace make_index_space(unsigned id)
    {
      IndexSpace ispace;
      ispace.id = id;
      return ispace;
    }

    static FieldSpace make_field_space(FieldSpaceID id)
    {
      FieldSpace fspace(id);
      return fspace;
    }
    
    static LogicalRegion make_logical_region(RegionTreeID tree_id,
                                             unsigned ispace_id,
                                             FieldSpaceID fspace_id)
    {
      IndexSpace ispace = make_index_space(ispace_id);
      FieldSpace fspace = make_field_space(fspace_id);
      LogicalRegion region(tree_id, ispace, fspace);
      return region;
    }

    static LogicalPartition make_logical_partition(RegionTreeID tree_id,
                                                   unsigned int ipartition,
                                                   FieldSpaceID fspace_id)
    {
      FieldSpace fspace = make_field_space(fspace_id);
      LogicalPartition partition(tree_id, ipartition, fspace);
      return partition;
    }

  };
  

  void top_task_wrapper(const void *local_args, size_t local_len,
                        const std::vector<RegionRequirement> &reqs,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime);

  void task_wrapper(const void *local_args, size_t local_len,
                    const std::vector<RegionRequirement> &reqs,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime);

  void index_task_wrapper(const void *global_args, size_t global_len,
                          const void *local_args, size_t local_len,
                          const DomainPoint &point,
                          const std::vector<RegionRequirement> &reqs,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime);

#ifdef WITH_TERRA
  void terra_task_wrapper(const Task* task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, HighLevelRuntime *runtime);
#endif
  
  static const int TY_NIL    = 0x01; // nil
  static const int TY_BOOL   = 0x0b; 
  static const int TY_NUM    = 0x10; // numbers in lua are only doubles
  static const int TY_STR    = 0x20;
  static const int TY_PTRT   = 0x30; // ptr_t
  static const int TY_LREG   = 0x40; // LogicalRegion
  static const int TY_ISPACE = 0x50; // IndexSpace
  static const int TY_FSPACE = 0x60; // FieldSpace
  static const int TY_TBL    = 0xFF;
  
  static size_t calculate_table_object_size(luabind::object obj);
  static size_t calculate_object_size(luabind::object obj);

  static size_t calculate_table_object_size(luabind::object obj)
  {
    using namespace luabind;

    size_t size_of_object = sizeof(uint32_t); // # entries in object

    for (luabind::iterator it(obj); it != luabind::iterator(); ++it) 
    {
      if (type(it.key()) == LUA_TSTRING)
      {
        size_of_object += calculate_object_size(it.key());
        size_of_object += calculate_object_size(*it);
      }
      else if(type(it.key()) == LUA_TNUMBER)
      {
        size_of_object += sizeof(uint8_t);
        size_of_object += sizeof(uint32_t);
        size_of_object += calculate_object_size(*it);
      }
    }

    return size_of_object;
  }

  static size_t calculate_object_size(luabind::object obj)
  {
    using namespace luabind;

    size_t size_of_object = sizeof(uint8_t); // tag for data type

    if (type(obj) != LUA_TTABLE)
    {
      switch(type(obj))
      {
      case LUA_TNUMBER:
        {
          size_of_object += sizeof(double);
          break;
        }
      case LUA_TSTRING:
        {
          size_of_object += sizeof(uint32_t);
          size_of_object += object_cast<std::string>(obj).size();
          break;
        }
      case LUA_TBOOLEAN:
        {
          size_of_object += sizeof(bool);
          break;
        }
      case LUA_TUSERDATA:
        {
          try
          {
            object_cast<ptr_t>(obj);
            size_of_object += sizeof(ptr_t);
          }
          catch(cast_failed& e)
          {
            try
            {
              object_cast<LogicalRegion>(obj);
              size_of_object += sizeof(LogicalRegion);
            }
            catch(cast_failed& e)
            {
              try
              {
                object_cast<IndexSpace>(obj);
                size_of_object += sizeof(IndexSpace);
              }
              catch(cast_failed& e)
              {
                try
                {
                  object_cast<FieldSpace>(obj);
                  size_of_object += sizeof(FieldSpace);
                }
                catch(cast_failed& e)
                {
                }
              }
            }
          }
          break;
        }
      case LUA_TNIL:
        {
          break;
        }
      default: break;
      }
    }
    else size_of_object += calculate_table_object_size(obj);

    return size_of_object;
  }

  template<typename T>
  static char* serialize_number(char* buffer, T num);
  static char* serialize_str(char* buffer, const char* str);
  static char* serialize_table_object(char* buffer, luabind::object obj);
  static char* serialize_object(char* buffer, luabind::object obj);

  template<typename T>
  static const char* deserialize_number(T& num, const char* buffer);
  static const char* deserialize_str(char*& str, const char* buffer);
  static const char* deserialize_table_object(lua_State* L,
                                              luabind::object& obj,
                                              const char* buffer);
  static const char* deserialize_object(lua_State* L,
                                        luabind::object& obj,
                                        const char* buffer);

  template<typename T>
  inline static char* serialize_number(char* buffer, T num)
  {
    T* ptr = reinterpret_cast<T*>(buffer);
    *ptr = num;
    return buffer + sizeof(T);
  }

  inline static char* serialize_str(char* buffer, const char* str)
  {
    uint32_t len = strlen(str);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(buffer);
    *ptr = len;
    buffer += sizeof(uint32_t);
    strncpy(buffer, str, len);
    return buffer + len;
  }

  static char* serialize_table_object(char* buffer, luabind::object obj)
  {
    using namespace luabind;

    uint32_t* ptr_to_num_entries = reinterpret_cast<uint32_t*>(buffer);
    uint32_t num_entries = 0;
    buffer += sizeof(uint32_t);

    for (luabind::iterator it(obj); it != luabind::iterator(); ++it) 
    {
      if (type(it.key()) == LUA_TSTRING)
      {
        buffer = serialize_object(buffer, it.key());
        buffer = serialize_object(buffer, *it);
        num_entries++;
      }
      else if(type(it.key()) == LUA_TNUMBER)
      {
        buffer =
          serialize_number<uint8_t>(buffer, TY_NUM);
        buffer =
          serialize_number<uint32_t>(buffer,
                                     object_cast<uint32_t>(it.key()));
        buffer = serialize_object(buffer, *it);
        num_entries++;
      }
    }
    
    *ptr_to_num_entries = num_entries;

    return buffer;
  }

  static char* serialize_object(char* buffer, luabind::object obj)
  {
    using namespace luabind;

    uint8_t* ptr_to_tag = reinterpret_cast<uint8_t*>(buffer);
    buffer += sizeof(uint8_t);

    if (type(obj) != LUA_TTABLE)
    {
      switch(type(obj))
      {
      case LUA_TNUMBER:
        {
          *ptr_to_tag = TY_NUM;
          double num = object_cast<double>(obj);
          buffer = serialize_number<double>(buffer, num);
          break;
        }
      case LUA_TSTRING:
        {
          *ptr_to_tag = TY_STR;
          std::string str = object_cast<std::string>(obj);
          buffer = serialize_str(buffer, str.c_str());
          break;
        }
      case LUA_TBOOLEAN:
        {
          *ptr_to_tag = TY_BOOL;
          bool b = object_cast<bool>(obj);
          buffer = serialize_number<bool>(buffer, b);
          break;
        }
      case LUA_TUSERDATA:
        {
          try
          {
            ptr_t ptr = object_cast<ptr_t>(obj);
            *ptr_to_tag = TY_PTRT;
            memcpy(buffer, &ptr, sizeof(ptr_t));
            buffer += sizeof(ptr_t);
          }
          catch(cast_failed& e)
          {
            try
            {
              LogicalRegion lr = object_cast<LogicalRegion>(obj);
              *ptr_to_tag = TY_LREG;
              memcpy(buffer, &lr, sizeof(LogicalRegion));
              buffer += sizeof(LogicalRegion);
            }
            catch(cast_failed& e)
            {
              try
              {
                IndexSpace ispace = object_cast<IndexSpace>(obj);
                *ptr_to_tag = TY_ISPACE;
                memcpy(buffer, &ispace, sizeof(IndexSpace));
                buffer += sizeof(IndexSpace);
              }
              catch(cast_failed& e)
              {
                try
                {
                  FieldSpace fspace = object_cast<FieldSpace>(obj);
                  *ptr_to_tag = TY_FSPACE;
                  memcpy(buffer, &fspace, sizeof(FieldSpace));
                  buffer += sizeof(FieldSpace);
                }
                catch(cast_failed& e)
                {
                }
              }
            }
          }
          break;
        }
      case LUA_TNIL:
        {
          *ptr_to_tag = TY_NIL;
          break;
        }
      default: break;
      }
    }
    else
    {
      *ptr_to_tag = TY_TBL;
      buffer = serialize_table_object(buffer, obj);
    }

    return buffer;
  }

  template<typename T>
  inline static const char* deserialize_number(T& num, const char* buffer)
  {
    const T* ptr = reinterpret_cast<const T*>(buffer);
    num = *ptr;
    return buffer + sizeof(T);
  }

  inline static const char* deserialize_str(char*& str, const char* buffer)
  {
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(buffer);
    const uint32_t len = *ptr;
    buffer += sizeof(uint32_t);
    str = new char[len + 1];
    strncpy(str, buffer, len);
    str[len] = '\0';
    return buffer + len;
  }

  static const char* deserialize_table_object(lua_State* L,
                                              luabind::object& obj,
                                              const char* buffer)
  {
    using namespace luabind;

    const uint32_t* ptr_to_num_entries =
      reinterpret_cast<const uint32_t*>(buffer);
    uint32_t num_entries = *ptr_to_num_entries;
    buffer += sizeof(uint32_t);

    obj = luabind::newtable(L);

    while (num_entries > 0)
    {
      const uint8_t* ptr_to_tag =
        reinterpret_cast<const uint8_t*>(buffer);
      uint8_t tag = *ptr_to_tag;
      
      if (tag == TY_NUM)
      {
        buffer += sizeof(uint8_t);
        uint32_t key = 0;
        buffer = deserialize_number<uint32_t>(key, buffer);

        luabind::object value;
        buffer = deserialize_object(L, value, buffer);
        obj[key] = value;
      }
      else if(tag == TY_STR)
      {
        luabind::object key;
        buffer = deserialize_object(L, key, buffer);

        luabind::object value;
        buffer = deserialize_object(L, value, buffer);
        obj[key] = value;
      }
      else
      {
        std::cerr << "unknown key type" << std::endl;
      }

      num_entries--;
    }
    
    return buffer;
  }

  static const char* deserialize_object(lua_State* L,
                                        luabind::object& obj,
                                        const char* buffer)
  {
    if (!buffer) return buffer;
    
    using namespace luabind;

    const uint8_t* ptr_to_tag =
      reinterpret_cast<const uint8_t*>(buffer);
    uint8_t tag = *ptr_to_tag;
    buffer += sizeof(uint8_t);

    if (tag != TY_TBL)
    {
      switch(tag)
      {
      case TY_NUM:
        {
          double num = 0.0;
          buffer = deserialize_number<double>(num, buffer);
          obj = object(L, num);
          break;
        }
      case TY_STR:
        {
          char* str = 0;
          buffer = deserialize_str(str, buffer);
          obj = object(L, std::string(str));
          delete str;
          break;
        }
      case TY_BOOL:
        {
          bool b = false;
          buffer = deserialize_number<bool>(b, buffer);
          obj = object(L, b);
          break;
        }
      case TY_PTRT:
        {
          ptr_t* ptr  = new ptr_t;
          memcpy(ptr, buffer, sizeof(ptr_t));
          buffer += sizeof(ptr_t);
          obj = object(L, ptr);
          break;
        }
      case TY_LREG:
        {
          LogicalRegion* lr = new LogicalRegion;
          memcpy(lr, buffer, sizeof(LogicalRegion));
          buffer += sizeof(LogicalRegion);
          obj = object(L, lr);
          break;
        }
      case TY_ISPACE:
        {
          IndexSpace* ispace = new IndexSpace;
          memcpy(ispace, buffer, sizeof(IndexSpace));
          buffer += sizeof(IndexSpace);
          obj = object(L, ispace);
          break;
        }
      case TY_FSPACE:
        {
          FieldSpace* fspace = new FieldSpace;
          memcpy(fspace, buffer, sizeof(FieldSpace));
          buffer += sizeof(FieldSpace);
          obj = object(L, fspace);
          break;
        }
      case TY_NIL:
        {
          break;
        }
      default: break;
      }
    }
    else
    {
      buffer = deserialize_table_object(L, obj, buffer);
    }

    return buffer;
  }
  
  static void register_single_task(unsigned task_id,
                                   const char* task_name,
                                   Processor::Kind kind,
                                   bool is_leaf)
  {
    
    if (LegionRuntime::HighLevel::Runtime::legion_main_id == task_id)
      HighLevelRuntime::register_single_task<top_task_wrapper>
        (task_id, kind, is_leaf, task_name);
    else
      HighLevelRuntime::register_single_task<task_wrapper>
        (task_id, kind, is_leaf, task_name);
  }

  static void register_index_task(unsigned task_id,
                                  const char* task_name,
                                  Processor::Kind kind,
                                  bool is_leaf)
  {
    HighLevelRuntime::register_index_task<index_task_wrapper>
      (task_id, kind, is_leaf, task_name);
  }

#ifdef WITH_TERRA
  static void register_terra_task(unsigned task_id,
                                  Processor::Kind kind,
                                  bool single, bool index)
  {
    HighLevelRuntime::register_legion_task<terra_task_wrapper>
      (task_id, kind, single, index);
  }
#endif
  
  static void print(luabind::object obj)
  {
    using namespace std;
    using namespace luabind; 

    if (type(obj) == LUA_TTABLE) 
    { 
      cout << "{" << endl; 

      for (luabind::iterator it(obj); it != luabind::iterator(); ++it) 
      {
        if (type(it.key()) == LUA_TSTRING || type(it.key()) == LUA_TNUMBER)
        { 
          if (type(*it) != LUA_TTABLE)
          {
            if (type(*it) == LUA_TNUMBER)
              cerr << it.key() << " = " << *it << endl;
            else if (type(*it) == LUA_TSTRING)
              cerr << it.key() << " = " << *it << endl;
            else
            {
              try
              {
                ptr_t ptr = object_cast<ptr_t>(*it);
                cerr << it.key() << " = ptr_t(" << ptr.value << ")" << endl;
              }
              catch(luabind::cast_failed& e)
              {
                try
                {
                  object_cast<LogicalRegion>(*it);
                  cerr << it.key() << " = LogicalRegion" << endl;
                }
                catch(luabind::cast_failed& e)
                {
                  try
                  {
                    IndexSpace ispace = object_cast<IndexSpace>(*it);
                    cerr << it.key() << " = IndexSpace(" << ispace.id << ")" << endl;
                  }
                  catch(luabind::cast_failed& e)
                  {
                    try
                    {
                      FieldSpace fspace = object_cast<FieldSpace>(*it);
                      cerr << it.key() << " = FieldSpace(" << fspace.get_id() << ")" << endl;
                    }
                    catch(luabind::cast_failed& e)
                    {
                
                    }
                
                  }
                }
              }
            }
          }
          else 
          { 
            cerr << it.key() << " = " << endl;
            print(*it); 
          } 
        } 
      } 

      cout << "}" << endl; 
    } 
  }

  static TaskArgument make_task_argument(luabind::object obj)
  {
    char* buffer = 0;
    size_t size_of_buffer = 0;

    size_of_buffer = calculate_object_size(obj);
    buffer = new char[size_of_buffer];
    serialize_object(buffer, obj);

    TaskArgument tmp(buffer, size_of_buffer);

    return tmp;
  }

#ifdef WITH_TERRA  
  static TaskArgument make_terra_task_argument(size_t ptr,
                                               size_t terra_obj_size)
  {
    char* buffer = new char[terra_obj_size];
    memcpy(buffer, reinterpret_cast<void*>(ptr), terra_obj_size);
    TaskArgument tmp(buffer, terra_obj_size);
    return tmp;
  }
#endif

  static void delete_task_argument(TaskArgument& arg)
  {
    char* ptr = reinterpret_cast<char*>(arg.get_ptr());
    delete ptr;
  }

  static void start(luabind::object arg)
  {
    using namespace luabind;

    char* buffer = 0;
    size_t size_of_buffer = 0;

    size_of_buffer = calculate_object_size(arg);
    buffer = new char[size_of_buffer];
    serialize_object(buffer, arg);
    
    HighLevelRuntime::start(1, &buffer, true);
    HighLevelRuntime::wait_for_shutdown();

    delete buffer;
  }

  static lua_State* prepare_new_interpreter()
  {
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    init(L);
#ifdef WITH_TERRA
    terra_init(L);
#endif

    return L;
  }

  static void load_task_file(lua_State* L, const char* task_file_name)
  {
#ifdef WITH_TERRA
    int err = terra_dofile(L, task_file_name);
#else
    int err = luaL_dofile(L, task_file_name);
#endif
    if (err != 0)
    {
      fprintf(stderr, "error loading task file : %s\n",
              lua_tostring(L, -1));
      exit(-1);
    }
  }

  inline const char* get_task_name(Context ctx)
  {
    using namespace LegionRuntime::HighLevel;
  
    Task* task = reinterpret_cast<Task*>(ctx);
    return task->variants->name;
  }

#ifdef PROF_BINDING    
  inline static void report_time(const char* task_name,
                                 struct timespec ts1,
                                 struct timespec ts2,
                                 struct timespec ts3,
                                 struct timespec ts4,
                                 struct timespec ts5)
  {
#define DIFF_TS(ts2, ts1)                       \
    (1e+3 * (ts2.tv_sec - ts1.tv_sec)) +        \
      (1e-6 * (ts2.tv_nsec - ts1.tv_nsec))      

    double init_interpreter = DIFF_TS(ts2, ts1);
    double deserialization  = DIFF_TS(ts3, ts2);
    double compilation      = DIFF_TS(ts4, ts3);
    double task_execution   = DIFF_TS(ts5, ts4);
      
    fprintf(stderr,
            "[%s] init interpreter : %7.2f ms\n",
            task_name,
            init_interpreter);
    fprintf(stderr,
            "[%s] deserialization  : %7.2f ms\n",
            task_name,
            deserialization);
    fprintf(stderr,
            "[%s] compilation      : %7.2f ms\n",
            task_name,
            compilation);
    fprintf(stderr,
            "[%s] task execution   : %7.2f ms\n",
            task_name,
            task_execution);
  }
#endif
  
  void top_task_wrapper(const void *local_args, size_t local_len,
                        const std::vector<RegionRequirement> &reqs,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
  {
#ifdef PROF_BINDING    
    struct timespec ts1, ts2, ts3, ts4, ts5;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
#endif
    const char* task_name = 0;
    // initialize new interpreter
    lua_State* L = prepare_new_interpreter();
#ifdef PROF_BINDING    
    clock_gettime(CLOCK_MONOTONIC, &ts2);
#endif
    {
      // deserialize lua object
      InputArgs *inputs = convert<InputArgs*>(local_args);
      const char* buffer = static_cast<const char*>(inputs->argv[0]);
      luabind::object lua_obj;
      deserialize_object(L, lua_obj, buffer);
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts3);
#endif
      // get task file name and load it to interpreter
      // assume that the task file loads legion binding library
      std::string task_file_name =
        luabind::object_cast<std::string>(lua_obj["__task_file_name"]);
      load_task_file(L, task_file_name.c_str());
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts4);
#endif  
      // call task wrapper in lua side
      task_name = get_task_name(ctx);
      try
      {
        luabind::call_function<void>(L,
                                     "task_wrapper_in_lua",
                                     task_name,
                                     task_file_name,
                                     lua_obj,
                                     reqs, regions, ctx, runtime);
      }
      catch(luabind::error& e)
      {
        fprintf(stderr, "error running function '%s': %s\n",
                task_name, lua_tostring(L, -1));
        exit(-1);
      }
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts5);
#endif
    } // wrap with a block to call destructor of luabind::object class

    lua_close(L);

#ifdef PROF_BINDING    
    report_time(task_name, ts1, ts2, ts3, ts4, ts5);
#endif

  }

  void task_wrapper(const void *local_args, size_t local_len,
                    const std::vector<RegionRequirement> &reqs,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
  {
#ifdef PROF_BINDING    
    struct timespec ts1, ts2, ts3, ts4, ts5;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
#endif
    const char* task_name = 0;
    // initialize new interpreter
    lua_State* L = prepare_new_interpreter();

    {
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts2);
#endif
      // deserialize lua object
      const char* buffer = reinterpret_cast<const char*>(local_args);
      luabind::object lua_obj;
      deserialize_object(L, lua_obj, buffer);

#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts3);
#endif
      // get task file name and load it to interpreter
      // assume that the task file loads legion binding library
      std::string task_file_name =
        luabind::object_cast<std::string>(lua_obj["__task_file_name"]);
      load_task_file(L, task_file_name.c_str());
  
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts4);
#endif
      // call task wrapper in lua side
      task_name = get_task_name(ctx);
      try
      {
        luabind::call_function<void>(L,
                                     "task_wrapper_in_lua",
                                     task_name,
                                     task_file_name,
                                     lua_obj,
                                     reqs, regions, ctx, runtime);
      }
      catch(luabind::error& e)
      {
        fprintf(stderr, "error running function '%s': %s\n",
                task_name, lua_tostring(L, -1));
        exit(-1);
      }
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts5);
#endif
    }  // wrap with a block to call destructor of luabind::object class

    lua_close(L);

#ifdef PROF_BINDING    
    report_time(task_name, ts1, ts2, ts3, ts4, ts5);
#endif
  }

  void index_task_wrapper(const void *global_args, size_t global_len,
                          const void *local_args, size_t local_len,
                          const DomainPoint &point,
                          const std::vector<RegionRequirement> &reqs,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime)
  {
#ifdef PROF_BINDING    
    struct timespec ts1, ts2, ts3, ts4, ts5;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
#endif
    const char* task_name = 0;
    // initialize new interpreter
    lua_State* L = prepare_new_interpreter();

    {
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts2);
#endif
      // deserialize lua object
      const char* global_buffer = reinterpret_cast<const char*>(global_args);
      const char* local_buffer = reinterpret_cast<const char*>(local_args);
      luabind::object global_lua_obj;
      luabind::object local_lua_obj;
      deserialize_object(L, global_lua_obj, global_buffer);
      deserialize_object(L, local_lua_obj, local_buffer);

#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts3);
#endif
      // get task file name and load it to interpreter
      // assume that the task file loads legion binding library
      std::string task_file_name =
        luabind::object_cast<std::string>(global_lua_obj["__task_file_name"]);
      load_task_file(L, task_file_name.c_str());
  
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts4);
#endif
      // call task wrapper in lua side
      task_name = get_task_name(ctx);
      try
      {
        luabind::call_function<void>(L,
                                     "index_task_wrapper_in_lua",
                                     task_name,
                                     task_file_name,
                                     global_lua_obj,
                                     local_lua_obj,
                                     point,
                                     reqs, regions, ctx, runtime);
      }
      catch(luabind::error& e)
      {
        fprintf(stderr, "error running function '%s': %s\n",
                task_name, lua_tostring(L, -1));
        exit(-1);
      }
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts5);
#endif
    } // wrap with a block to call destructor of luabind::object class

    lua_close(L);

#ifdef PROF_BINDING    
    report_time(task_name, ts1, ts2, ts3, ts4, ts5);
#endif
  }

#ifdef WITH_TERRA
  typedef void (*terra_task_ptr)(void*, TTask, const TPhysicalRegion*,
                                 size_t, void*, void*);

  typedef void (*terra_index_task_ptr)(void*, void*,
                                       TTask, const TPhysicalRegion*,
                                       size_t, void*, void*);

  void terra_task_wrapper(const Task* task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, HighLevelRuntime *runtime)
  {
#ifdef PROF_BINDING    
    struct timespec ts1, ts2, ts3, ts4, ts5;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
#endif
    void* terra_task_arg = 
      (reinterpret_cast<char*>(task->args) + sizeof(void*));

    size_t num_regions = regions.size();
    TPhysicalRegion regions_array[num_regions];
    for(size_t i = 0; i < num_regions; ++i)
    {
      regions_array[i].rawptr =
        const_cast<PhysicalRegion*>(&regions[i]);
      regions_array[i].redop = task->regions[i].redop;
    }

    TTask task_wrapper;
    task_wrapper.rawptr = const_cast<Task*>(task);
    if (!task->is_index_space)
    {
      terra_task_ptr terra_task_fun =
        reinterpret_cast<terra_task_ptr>
        (*reinterpret_cast<void**>(task->args));
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts2);
#endif

      (*terra_task_fun)(terra_task_arg, task_wrapper,
                        regions_array, num_regions,
                        ctx, runtime);

#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts3);
#endif

    }
    else // task->is_index_space
    {
      terra_index_task_ptr terra_task_fun =
        reinterpret_cast<terra_index_task_ptr>
        (*reinterpret_cast<void**>(task->args));
#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts2);
#endif

      (*terra_task_fun)(terra_task_arg,
                        task->local_args,
                        task_wrapper, regions_array, num_regions,
                        ctx, runtime);

#ifdef PROF_BINDING    
      clock_gettime(CLOCK_MONOTONIC, &ts3);
#endif

    }

#ifdef PROF_BINDING    
    const char* task_name = get_task_name(ctx);
    report_time(task_name, ts1, ts1, ts2, ts2, ts3);
#endif
  }
#endif

  static void register_plus_reduction_for_float(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<PlusAccumulatorFloat>(op);
  }

  static void register_plus_reduction_for_double(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<PlusAccumulatorDouble>(op);
  }

  static void register_plus_reduction_for_int(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<PlusAccumulatorInt>(op);
  }    

  static void register_minus_reduction_for_float(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<MinusAccumulatorFloat>(op);
  }

  static void register_minus_reduction_for_double(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<MinusAccumulatorDouble>(op);
  }

  static void register_minus_reduction_for_int(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<MinusAccumulatorInt>(op);
  }    

  static void register_times_reduction_for_float(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<TimesAccumulatorFloat>(op);
  }

  static void register_times_reduction_for_double(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<TimesAccumulatorDouble>(op);
  }

  static void register_times_reduction_for_int(ReductionOpID op)
  {
    HighLevelRuntime::register_reduction_op<TimesAccumulatorInt>(op);
  }    
  

} // namespace BindingLib

int init(lua_State* L)
{
  using namespace luabind;
  using namespace LegionRuntime::HighLevel;

  open(L);

  module(L)
    [
     register_ptr_t()
     ,
     namespace_("LegionRuntime")
     [
      register_highlevel()
      ,
      register_lowlevel()
      ]
     ,
     namespace_("BindingLibInC")
     [
      def("register_single_task", &BindingLib::register_single_task)
      ,
      def("register_index_task", &BindingLib::register_index_task)
      ,
#ifdef WITH_TERRA
      def("register_terra_task", &BindingLib::register_terra_task)
      ,
#endif
      def("start", &BindingLib::start)
      ,
      def("make_task_argument", &BindingLib::make_task_argument)
      ,
#ifdef WITH_TERRA
      def("make_terra_task_argument", &BindingLib::make_terra_task_argument)
      ,
#endif
      def("delete_task_argument", &BindingLib::delete_task_argument)
      ,
      def("make_index_space", &BindingLib::Utility::make_index_space)
      ,
      def("make_field_space", &BindingLib::Utility::make_field_space)
      ,
      def("make_logical_region", &BindingLib::Utility::make_logical_region)
      ,
      def("make_logical_partition", &BindingLib::Utility::make_logical_partition)
      ,
      def("register_times_reduction_for_float",
          &BindingLib::register_times_reduction_for_float)
      ,
      def("register_times_reduction_for_double",
          &BindingLib::register_times_reduction_for_double)
      ,
      def("register_times_reduction_for_int",
          &BindingLib::register_times_reduction_for_int)
      ,
      def("register_minus_reduction_for_float",
          &BindingLib::register_minus_reduction_for_float)
      ,
      def("register_minus_reduction_for_double",
          &BindingLib::register_minus_reduction_for_double)
      ,
      def("register_minus_reduction_for_int",
          &BindingLib::register_minus_reduction_for_int)
      ,
      def("register_plus_reduction_for_float",
          &BindingLib::register_plus_reduction_for_float)
      ,
      def("register_plus_reduction_for_double",
          &BindingLib::register_plus_reduction_for_double)
      ,
      def("register_plus_reduction_for_int",
          &BindingLib::register_plus_reduction_for_int)
      ]
     ];

  return 0;
}

#ifdef WITH_TERRA

extern "C"
{
  void* create_index_iterator(struct TLogicalRegion region)
  {
    using namespace LegionRuntime::HighLevel;

    LogicalRegion logicalregion =
      BindingLib::Utility::make_logical_region(region.tree_id,
                                               region.index_space.id,
                                               region.field_space.id);
    IndexIterator* iterator = new IndexIterator(logicalregion);
    return iterator;
  }

  void destroy_index_iterator(void* _iterator)
  {
    IndexIterator* iterator = reinterpret_cast<IndexIterator*>(_iterator);
    delete iterator;
  }

  unsigned next(void* _iterator)
  {
    IndexIterator* iterator = reinterpret_cast<IndexIterator*>(_iterator);
    ptr_t ptr = iterator->next();
    return ptr.value;
  }

  int has_next(void* _iterator)
  {
    IndexIterator* iterator = reinterpret_cast<IndexIterator*>(_iterator);
    return iterator->has_next();
  }

  // simple PhysicalRegion wrapper in C

  void* create_terra_accessor(TPhysicalRegion region)
  {
    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;
    using namespace LegionRuntime::HighLevel;
    
    PhysicalRegion* physical_region =
      reinterpret_cast<PhysicalRegion*>(region.rawptr);
    RegionAccessor<Generic>* _accessor =
      new RegionAccessor<Generic>(physical_region->get_accessor());
    return _accessor;
  }
  
  void* create_terra_field_accessor(TPhysicalRegion region, unsigned field)
  {
    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;
    using namespace LegionRuntime::HighLevel;
    PhysicalRegion* physical_region =
      reinterpret_cast<PhysicalRegion*>(region.rawptr);
    RegionAccessor<Generic>* _accessor =
      new RegionAccessor<Generic>(physical_region->get_field_accessor(field));
    return _accessor;
  }

  void destroy_terra_accessor(void* _accessor)
  {
    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;
    delete reinterpret_cast<RegionAccessor<Generic>*>(_accessor);
  }

  // simple GenericAccessor wrapper in C
  void read_from_accessor(void* _accessor, unsigned ptr,
                          void* dst, size_t bytes)
  {
    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;
    RegionAccessor<Generic>* accessor =
      reinterpret_cast<RegionAccessor<Generic>*>(_accessor);
    accessor->read_untyped(ptr_t(ptr), dst, bytes);
  }
    
  void write_to_accessor(void* _accessor, unsigned ptr,
                         void* src, size_t bytes)
  {
    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;
    RegionAccessor<Generic>* accessor =
      reinterpret_cast<RegionAccessor<Generic>*>(_accessor);
    accessor->write_untyped(ptr_t(ptr), src, bytes);
  }

  void* create_terra_reducer(TPhysicalRegion region,
                             unsigned long long offset,
                             uint32_t redop,
                             uint32_t elem_type,
                             uint32_t red_type)
  {
#define MAKE_SAFE_REDUCER(VAR_NAME, ACCESSOR, REDUCER_TYPE, ELEM_TYPE) \
    RegionAccessor<ReductionFold<REDUCER_TYPE>, ELEM_TYPE>* VAR_NAME = \
      new RegionAccessor<ReductionFold<REDUCER_TYPE>, ELEM_TYPE>    \
      (accessor.typeify<ELEM_TYPE>().convert<ReductionFold<REDUCER_TYPE> >()) 
#define MAKE_UNSAFE_REDUCER(VAR_NAME, ACCESSOR, REDUCER_TYPE, ELEM_TYPE) \
    RegionAccessor<Generic, ELEM_TYPE>* VAR_NAME =  \
      new RegionAccessor<Generic, ELEM_TYPE>(accessor.typeify<ELEM_TYPE>())

    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;

    PhysicalRegion* physical_region =
      reinterpret_cast<PhysicalRegion*>(region.rawptr);
    RegionAccessor<Generic> accessor =
      physical_region->get_accessor().get_untyped_field_accessor(offset, 0);

    if (redop > 0)
    {
      switch (elem_type)
      {
      case PRIM_TYPE_FLOAT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, PlusAccumulatorFloat, float);
              return _reducer;
            }
          case RED_TYPE_MINUS:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, MinusAccumulatorFloat, float);
              return _reducer;
            }
          case RED_TYPE_TIMES:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, TimesAccumulatorFloat, float);
              return _reducer;
            }
          }
        }
      case PRIM_TYPE_DOUBLE:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, PlusAccumulatorDouble, double);
              return _reducer;
            }
          case RED_TYPE_MINUS:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, MinusAccumulatorDouble, double);
              return _reducer;
            }
          case RED_TYPE_TIMES:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, TimesAccumulatorDouble, double);
              return _reducer;
            }
          }
        }
      case PRIM_TYPE_INT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, PlusAccumulatorInt, int);
              return _reducer;
            }
          case RED_TYPE_MINUS:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, MinusAccumulatorInt, int);
              return _reducer;
            }
          case RED_TYPE_TIMES:
            {
              MAKE_SAFE_REDUCER(_reducer, accesor, TimesAccumulatorInt, int);
              return _reducer;
            }
          }
        }
      }
    }
    else
    {
      switch (elem_type)
      {
      case PRIM_TYPE_FLOAT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, PlusAccumulatorFloat, float);
              return _reducer;
            }
          case RED_TYPE_MINUS:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, MinusAccumulatorFloat, float);
              return _reducer;
            }
          case RED_TYPE_TIMES:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, TimesAccumulatorFloat, float);
              return _reducer;
            }
          }
        }
      case PRIM_TYPE_DOUBLE:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, PlusAccumulatorDouble, double);
              return _reducer;
            }
          case RED_TYPE_MINUS:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, MinusAccumulatorDouble, double);
              return _reducer;
            }
          case RED_TYPE_TIMES:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, TimesAccumulatorDouble, double);
              return _reducer;
            }
          }
        }
      case PRIM_TYPE_INT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, PlusAccumulatorInt, int);
              return _reducer;
            }
          case RED_TYPE_MINUS:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, MinusAccumulatorInt, int);
              return _reducer;
            }
          case RED_TYPE_TIMES:
            {
              MAKE_UNSAFE_REDUCER(_reducer, accesor, TimesAccumulatorInt, int);
              return _reducer;
            }
          }
        }
      }
    }
  }
  
  void reduce_terra_reducer_float(void* _reducer,
                                  uint32_t redop,
                                  uint32_t red_type,
                                  unsigned int ptr,
                                  float value)
  {
#define CAST_SAFE_REDUCER(REDUCER, REDUCER_TYPE) \
    (reinterpret_cast<RegionAccessor<ReductionFold<REDUCER_TYPE>, float>*>(REDUCER))
#define CAST_UNSAFE_REDUCER(REDUCER) \
    (reinterpret_cast<RegionAccessor<Generic, float>*>(REDUCER))

    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;

    if (redop > 0)
    {
      switch (red_type)
      {
      case RED_TYPE_PLUS:
        {
          CAST_SAFE_REDUCER(_reducer, PlusAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      case RED_TYPE_MINUS:
        {
          CAST_SAFE_REDUCER(_reducer, MinusAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      case RED_TYPE_TIMES:
        {
          CAST_SAFE_REDUCER(_reducer, TimesAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      }
    }
    else
    {
      switch (red_type)
      {
      case RED_TYPE_PLUS:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<PlusAccumulatorFloat>(ptr, value);
          return;
        }
      case RED_TYPE_MINUS:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<MinusAccumulatorFloat>(ptr, value);
          return;
        }
      case RED_TYPE_TIMES:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<TimesAccumulatorFloat>(ptr, value);
          return;
        }
      }
    }
  }

  void reduce_terra_reducer_double(void* _reducer,
                                   uint32_t redop,
                                   uint32_t red_type,
                                   unsigned int ptr,
                                   double value)
  {
#define CAST_SAFE_REDUCER(REDUCER, REDUCER_TYPE) \
    (reinterpret_cast<RegionAccessor<ReductionFold<REDUCER_TYPE>, float>*>(REDUCER))
#define CAST_UNSAFE_REDUCER(REDUCER) \
    (reinterpret_cast<RegionAccessor<Generic, float>*>(REDUCER))

    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;

    if (redop > 0)
    {
      switch (red_type)
      {
      case RED_TYPE_PLUS:
        {
          CAST_SAFE_REDUCER(_reducer, PlusAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      case RED_TYPE_MINUS:
        {
          CAST_SAFE_REDUCER(_reducer, MinusAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      case RED_TYPE_TIMES:
        {
          CAST_SAFE_REDUCER(_reducer, TimesAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      }
    }
    else
    {
      switch (red_type)
      {
      case RED_TYPE_PLUS:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<PlusAccumulatorFloat>(ptr, value);
          return;
        }
      case RED_TYPE_MINUS:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<MinusAccumulatorFloat>(ptr, value);
          return;
        }
      case RED_TYPE_TIMES:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<TimesAccumulatorFloat>(ptr, value);
          return;
        }
      }
    }
  }

  void reduce_terra_reducer_int(void* _reducer,
                                uint32_t redop,
                                uint32_t red_type,
                                unsigned int ptr,
                                int value)
  {
#define CAST_SAFE_REDUCER(REDUCER, REDUCER_TYPE) \
    (reinterpret_cast<RegionAccessor<ReductionFold<REDUCER_TYPE>, float>*>(REDUCER))
#define CAST_UNSAFE_REDUCER(REDUCER) \
    (reinterpret_cast<RegionAccessor<Generic, float>*>(REDUCER))

    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;

    if (redop > 0)
    {
      switch (red_type)
      {
      case RED_TYPE_PLUS:
        {
          CAST_SAFE_REDUCER(_reducer, PlusAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      case RED_TYPE_MINUS:
        {
          CAST_SAFE_REDUCER(_reducer, MinusAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      case RED_TYPE_TIMES:
        {
          CAST_SAFE_REDUCER(_reducer, TimesAccumulatorFloat)->reduce(ptr, value);
          return;
        }
      }
    }
    else
    {
      switch (red_type)
      {
      case RED_TYPE_PLUS:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<PlusAccumulatorFloat>(ptr, value);
          return;
        }
      case RED_TYPE_MINUS:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<MinusAccumulatorFloat>(ptr, value);
          return;
        }
      case RED_TYPE_TIMES:
        {
          CAST_UNSAFE_REDUCER(_reducer)->reduce<TimesAccumulatorFloat>(ptr, value);
          return;
        }
      }
    }
  }

  void destroy_terra_reducer(void* _reducer,
                             uint32_t redop,
                             uint32_t elem_type,
                             uint32_t red_type)
  {
#define DELETE_SAFE_REDUCER(REDUCER, REDUCER_TYPE, ELEM_TYPE)           \
    delete                                                              \
      (reinterpret_cast<RegionAccessor<ReductionFold<REDUCER_TYPE>, ELEM_TYPE>*>(REDUCER))
#define DELETE_UNSAFE_REDUCER(REDUCER, ELEM_TYPE)                       \
    delete (reinterpret_cast<RegionAccessor<Generic, ELEM_TYPE>*>(REDUCER))

    using namespace LegionRuntime::Accessor;
    using namespace LegionRuntime::Accessor::AccessorType;

    if (redop > 0)
    {
      switch (elem_type)
      {
      case PRIM_TYPE_FLOAT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              DELETE_SAFE_REDUCER(_reducer, PlusAccumulatorFloat, float);
              return;
            }
          case RED_TYPE_MINUS:
            {
              DELETE_SAFE_REDUCER(_reducer, MinusAccumulatorFloat, float);
              return;
            }
          case RED_TYPE_TIMES:
            {
              DELETE_SAFE_REDUCER(_reducer, TimesAccumulatorFloat, float);
              return;
            }
          }
        }
      case PRIM_TYPE_DOUBLE:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              DELETE_SAFE_REDUCER(_reducer, PlusAccumulatorDouble, double);
              return;
            }
          case RED_TYPE_MINUS:
            {
              DELETE_SAFE_REDUCER(_reducer, MinusAccumulatorDouble, double);
              return;
            }
          case RED_TYPE_TIMES:
            {
              DELETE_SAFE_REDUCER(_reducer, TimesAccumulatorDouble, double);
              return;
            }
          }
        }
      case PRIM_TYPE_INT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              DELETE_SAFE_REDUCER(_reducer, PlusAccumulatorInt, int);
              return;
            }
          case RED_TYPE_MINUS:
            {
              DELETE_SAFE_REDUCER(_reducer, MinusAccumulatorInt, int);
              return;
            }
          case RED_TYPE_TIMES:
            {
              DELETE_SAFE_REDUCER(_reducer, TimesAccumulatorInt, int);
              return;
            }
          }
        }
      }
    }
    else
    {
      switch (elem_type)
      {
      case PRIM_TYPE_FLOAT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              DELETE_UNSAFE_REDUCER(_reducer, float);
              return;
            }
          case RED_TYPE_MINUS:
            {
              DELETE_UNSAFE_REDUCER(_reducer, float);
              return;
            }
          case RED_TYPE_TIMES:
            {
              DELETE_UNSAFE_REDUCER(_reducer, float);
              return;
            }
          }
        }
      case PRIM_TYPE_DOUBLE:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              DELETE_UNSAFE_REDUCER(_reducer, double);
              return;
            }
          case RED_TYPE_MINUS:
            {
              DELETE_UNSAFE_REDUCER(_reducer, double);
              return;
            }
          case RED_TYPE_TIMES:
            {
              DELETE_UNSAFE_REDUCER(_reducer, double);
              return;
            }
          }
        }
      case PRIM_TYPE_INT:
        {
          switch (red_type)
          {
          case RED_TYPE_PLUS:
            {
              DELETE_UNSAFE_REDUCER(_reducer, int);
              return;
            }
          case RED_TYPE_MINUS:
            {
              DELETE_UNSAFE_REDUCER(_reducer, int);
              return;
            }
          case RED_TYPE_TIMES:
            {
              DELETE_UNSAFE_REDUCER(_reducer, int);
              return;
            }
          }
        }
      }
    }
  }


  // simple Task wrapper in C

  int get_index(void* _task)
  {
    using namespace LegionRuntime::HighLevel;
    Task* task = reinterpret_cast<Task*>(_task);
    return task->index_point.get_index();
  }
}


#endif
