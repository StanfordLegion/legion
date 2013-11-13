/* Copyright 2013 Stanford University and Los Alamos National Security, LLC
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

#include "foreign_cpp_region.h"

using namespace LegionRuntime::Accessor;

void foreign_initialize(HighLevelRuntime *runtime, Context ctx, PhysicalRegion region[1], ptr_t pointer)
{
  RegionAccessor<AccessorType::AOS<0>, intptr_t> accessor =
    region[0].get_accessor().typeify<intptr_t>().convert<AccessorType::AOS<0> >();
  accessor.write(pointer, (intptr_t)5);
}

void foreign_iterate(HighLevelRuntime *runtime, Context ctx, PhysicalRegion region[1])
{
  RegionAccessor<AccessorType::AOS<0>, intptr_t> accessor =
    region[0].get_accessor().typeify<intptr_t>().convert<AccessorType::AOS<0> >();
  IndexSpace ispace = region[0].get_logical_region().get_index_space();
  Domain domain = runtime->get_index_space_domain(ctx, ispace);
  for (Domain::DomainPointIterator point(domain); point; point++) {
    ptr_t pointer(point.p.get_index());
    accessor.write(pointer, accessor.read(pointer) + 20);
  }
}
