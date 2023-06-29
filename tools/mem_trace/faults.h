/* Copyright 2023 Stanford University, NVIDIA Corporation
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

// helper defines/data structures for fault reporting/handling in Realm

#ifndef FAULTS_H
#define FAULTS_H

#include <vector>
#include <iostream>

#include <stdint.h>

class Backtrace {
public:
  Backtrace(void);
  ~Backtrace(void);

  Backtrace(const Backtrace& copy_from);
  Backtrace& operator=(const Backtrace& copy_from);

  bool operator==(const Backtrace& rhs) const;

  uintptr_t hash(void) const;

  bool empty(void) const;

  // attempts to prune this backtrace by removing frames that appear
  //  in the other one
  bool prune(const Backtrace &other);

  // captures the current back trace, skipping 'skip' frames, and optionally
  //   limiting the total depth - this isn't as simple as as stack walk any more,
  //   so you probably don't want to ask for these during any normal execution paths
  void capture_backtrace(int skip = 0, int max_depth = 0);

  // attempts to map the pointers in the back trace to symbol names - this can be
  //   much more expensive
  void lookup_symbols(void);

  friend std::ostream& operator<<(std::ostream& os, const Backtrace& bt);
  //static Backtrace *bt;
protected:
  uintptr_t compute_hash(int depth = 0) const;

  uintptr_t pc_hash; // used for fast comparisons
  std::vector<uintptr_t> pcs;
  std::vector<std::string> symbols;
};

#endif // FAULTS_H
