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

#include "foreign_cpp_region_fields.h"
#include "foreign_cpp_region_fields.lg.h"

using namespace LegionRuntime::Accessor;

void foreign_initialize(HighLevelRuntime *runtime, Context ctx,
                        PhysicalRegion regions[2],
                        ptr_t pointer)
{
  RegionAccessor<AccessorType::Generic, intptr_t> accessor_y =
    regions[1].get_field_accessor(FIELD_Y).typeify<intptr_t>();
  accessor_y.write(pointer, (intptr_t)2);
}

void foreign_iterate(HighLevelRuntime *runtime, Context ctx,
                     PhysicalRegion regions[3])
{
  RegionAccessor<AccessorType::Generic, intptr_t> accessor_x =
    regions[1].get_field_accessor(FIELD_X).typeify<intptr_t>();
  RegionAccessor<AccessorType::Generic, intptr_t> accessor_y =
    regions[0].get_accessor().typeify<intptr_t>();
  RegionAccessor<AccessorType::Generic, intptr_t> accessor_z =
    regions[2].get_field_accessor(FIELD_Z).typeify<intptr_t>();

  IndexSpace ispace = regions[0].get_logical_region().get_index_space();
  Domain domain = runtime->get_index_space_domain(ctx, ispace);
  for (Domain::DomainPointIterator point(domain); point; point++) {
    ptr_t pointer(point.p.get_index());
    accessor_x.write(pointer, 10);
    accessor_y.reduce<reduction_plus_intptr_t>(pointer, 20);
    accessor_z.write(pointer, 30);
  }
}
