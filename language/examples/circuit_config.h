/* Copyright 2022 Stanford University
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

#ifndef __CIRCUIT_CONFIG_H__
#define __CIRCUIT_CONFIG_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Config {
  unsigned num_loops;
  unsigned num_pieces;
  unsigned pieces_per_superpiece;
  unsigned nodes_per_piece;
  unsigned wires_per_piece;
  unsigned pct_wire_in_piece;
  unsigned random_seed;
  unsigned steps;
  unsigned sync;
  unsigned prune;
  bool perform_checks;
  bool dump_values;
  double pct_shared_nodes;
  unsigned shared_nodes_per_piece;
  unsigned density;
  unsigned num_neighbors;
  int window;
} Config;

#ifdef __cplusplus
}
#endif

#endif // __CIRCUIT_CONFIG_H__
