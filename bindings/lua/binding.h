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

#ifndef __BINDING_H__
#define __BINDING_H__

struct TIndexSpace
{
  unsigned id;
};

struct TFieldSpace
{
  unsigned id;
};

struct TLogicalRegion
{
  unsigned int tree_id;
  struct TIndexSpace index_space;
  struct TFieldSpace field_space;
};

struct TLogicalPartition
{
  unsigned int tree_id;
  unsigned int index_partition;
  struct TFieldSpace field_space;
};

struct TPhysicalRegion
{
  void* rawptr;
  unsigned int redop;
};

struct TTask
{
  void* rawptr;
};

#endif // __BINDING_H_
