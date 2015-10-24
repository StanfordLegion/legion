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

// NOTE: all these interfaces are templated, which means partitions.cc is going
//  to have to somehow know which ones to instantiate - we'll try to have a 
//  Makefile-based way to control this, but right now it's hardcoded at the
//  bottom of partitions.cc, so go there if you get link errors

namespace Realm {

};

#endif // REALM_PARTITIONS_H

