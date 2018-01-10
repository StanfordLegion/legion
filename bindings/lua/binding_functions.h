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

#ifndef __BINDING_FUNCTIONS_H__
#define __BINDING_FUNCTIONS_H__

#include "binding.h"

// simple IndexIterator wrapper in C

void* create_index_iterator(struct TLogicalRegion region);
void destroy_index_iterator(void* _iterator);
unsigned next(void* _iterator);
int has_next(void* _iterator);

// simple PhysicalRegion wrapper in C

void* create_terra_accessor(struct TPhysicalRegion region);
void* create_terra_field_accessor(struct TPhysicalRegion region,
                                  unsigned field);
void destroy_terra_accessor(void* _accessor);

// simple GenericAccessor wrapper in C
void read_from_accessor(void* _accessor, unsigned ptr,
                        void* dst, unsigned long long bytes);
void write_to_accessor(void* _accessor, unsigned ptr,
                       void* src, unsigned long long bytes);

void* create_terra_reducer(struct TPhysicalRegion region,
                           unsigned long long offset,
                           unsigned int redop,
                           unsigned int elem_type,
                           unsigned int red_type);
void reduce_terra_reducer_float(void* _reducer,
                                unsigned int redop,
                                unsigned int red_type,
                                unsigned int ptr,
                                float value);
void reduce_terra_reducer_double(void* _reducer,
                                 unsigned int redop,
                                 unsigned int red_type,
                                 unsigned int ptr,
                                 double value);
void reduce_terra_reducer_int(void* _reducer,
                              unsigned int redop,
                              unsigned int red_type,
                              unsigned int ptr,
                              int value);
void destroy_terra_reducer(void* _reducer,
                           unsigned int redop,
                           unsigned int elem_type,
                           unsigned int red_type);

// simple Task wrapper in C
int get_index(void* _task);

#endif // __BINDING_FUNCTIONS_H__
