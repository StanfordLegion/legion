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

#include "faults.h"
#include <stdlib.h>
#include <alloca.h>
#include <assert.h>
#include <execinfo.h>
// #ifdef HAVE_CXXABI_H
#include <cxxabi.h>
// #endif
#include <iomanip>

//Backtrace* Backtrace::bt;

////////////////////////////////////////////////////////////////////////
//
// class Backtrace


Backtrace::Backtrace(void)
{}

Backtrace::~Backtrace(void)
{}

Backtrace::Backtrace(const Backtrace& copy_from)
  : pc_hash(copy_from.pc_hash), pcs(copy_from.pcs), symbols(copy_from.symbols)
{}

Backtrace& Backtrace::operator=(const Backtrace& copy_from)
{
  pc_hash = copy_from.pc_hash;
  pcs = copy_from.pcs;
  symbols = copy_from.symbols;
  return *this;
}

bool Backtrace::operator==(const Backtrace& rhs) const
{
  // two early outs - different numbers of frame or a different hash
  if(pc_hash != rhs.pc_hash) return false;
  if(pcs.size() != rhs.pcs.size()) return false;
  // for now, check all the pcs - a good hash should make this unnecessary
  for(size_t i = 0; i < pcs.size(); i++)
    if(pcs[i] != rhs.pcs[i]) {
      std::cerr << "Hash match, but PC mismatch: hash = " << pc_hash << std::endl;
      std::cerr << "First backtrace " << *this;
      std::cerr << "Second backtrace " << rhs;
      return false;
    }
  return true;
}

uintptr_t Backtrace::hash(void) const
{
  return pc_hash;
}

bool Backtrace::empty(void) const
{
  return pcs.empty();
}

// attempts to prune this backtrace by removing the frames from the other one
bool Backtrace::prune(const Backtrace &other)
{
  // start searching from the second PC - we don't want to prune the whole thing
  for(size_t i = 1; i < pcs.size(); i++) {
    bool match = true;
    for(size_t j = 0; (j < other.pcs.size()) && (j + i) < pcs.size(); j++)
      if(pcs[j + i] != other.pcs[j]) {
        match = false;
        break;
      }
    if(match) {
      pcs.resize(i);
      if(!symbols.empty())
        symbols.resize(i);
      // recompute the hash too
      pc_hash = compute_hash();
      return true;
    }
  }
  return false;
}

uintptr_t Backtrace::compute_hash(int depth /*= 0*/) const
{
  uintptr_t newhash = 0;
  int i = 0;
  for(std::vector<uintptr_t>::const_iterator it = pcs.begin();
      it != pcs.end();
      it++) {
    newhash = (newhash * 0x10021) ^ *it;
    if(++i == depth) break;
  }
  return newhash;
}

// captures the current back trace, skipping 'skip' frames, and optionally
//   limiting the total depth - this is fairly quick as it just walks the stack
//   and records pointers
void Backtrace::capture_backtrace(int skip /*= 0*/, int max_depth /*= 0*/)
{
  uintptr_t *rawptrs;
  // if we weren't given a max depth, pick 100 for now
  if(max_depth <= 0)
    max_depth = 100;

  // clamp the skip amount, add in one for this call
  if(skip <= 0)
    skip = 0;
  skip++;

  // allocate space for the result of backtrace(), including the stuff on
  //  the front we're going to skip
  assert(sizeof(void *) == sizeof(intptr_t));
  rawptrs = (uintptr_t *)alloca(sizeof(void *) * (max_depth + skip));
  int count = backtrace((void **)rawptrs, max_depth + skip);
  assert(count >= 0);

  pcs.clear();
  symbols.clear();

  if(count > skip)
  {
    pcs.insert(pcs.end(), rawptrs + skip, rawptrs + count);
  }


  // recompute the hash too
  pc_hash = compute_hash();
}

// attempts to map the pointers in the back trace to symbol names - this can be
//   more expensive
void Backtrace::lookup_symbols(void)
{
  // have we already done the lookup?
  if(!symbols.empty()) return;

  symbols.resize(pcs.size());

  for(size_t i = 0; i < pcs.size(); i++) {
    // try backtrace_symbols() first
    char **s = backtrace_symbols((void * const *)&(pcs[i]), 1);
    if(s) {
      symbols[i].assign(s[0]);
      free(s);
      continue;
    }
    symbols[i] = "unknown";
  }
}

std::ostream& operator<<(std::ostream& os, const Backtrace& bt)
{
  char *demangle_buffer = 0;
  size_t demangle_len = 0;

  os << "stack trace: " << bt.pcs.size() << " frames" << std::endl;
  for(size_t i = 0; i < bt.pcs.size(); i++) {
    os << "  [" << i << "] = ";
    if(!bt.symbols.empty() && !bt.symbols[i].empty()) {
      char *s = (char *)(bt.symbols[i].c_str());
      char *lp = s;
      bool print_raw = true;
// #ifdef HAVE_CXXABI_H
      while(*lp && (*lp != '(')) lp++;
      if(*lp && (lp[1] != '+')) {
        char *rp = ++lp;
        while(*rp && (*rp != '+') && (*rp != ')')) rp++;
        if(*rp) {
          char orig_rp = *rp;
          *rp = 0;
          int status = -4;
          char *result = abi::__cxa_demangle(lp, demangle_buffer, &demangle_len, &status);
          *rp = orig_rp;
          if(status == 0) {
            demangle_buffer = result;
            char orig_lp = *lp;
            *lp = 0;
            os << s << demangle_buffer << rp;
            print_raw = false;
            *lp = orig_lp;
          }
        }
      }
// #endif
      if(print_raw)
        os << bt.symbols[i];
    } else {
      os << std::hex << std::setfill('0') << std::setw(sizeof(uintptr_t)*2) << bt.pcs[i];
      os << std::dec << std::setfill(' ');
    }
    os << std::endl;
  }

  if(demangle_buffer)
    free(demangle_buffer);

  return os;
}
