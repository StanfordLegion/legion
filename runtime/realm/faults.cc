/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "realm/faults.h"
#include "realm/profiling.h"

#include <stdlib.h>
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
// FreeBSD defines alloca() in stdlib.h
#include <alloca.h>
#endif
#include <assert.h>
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#ifdef REALM_USE_CPPTRACE
#include <cpptrace/cpptrace.hpp>
#endif /* REALM_USE_CPPTRACE */
#include <execinfo.h>
#endif /* defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD) */
#ifdef REALM_HAVE_CXXABI_H
#include <cxxabi.h>
#endif
#include <iomanip>

#ifdef ERROR_CANCELLED
#undef ERROR_CANCELLED
#endif

namespace Realm {

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
    // if we weren't given a max depth, pick 100 for now
    if(max_depth <= 0)
      max_depth = 100;

    // clamp the skip amount, add in one for this call
    if(skip <= 0)
      skip = 0;
    skip++;

#ifdef REALM_USE_CPPTRACE
    cpptrace::raw_trace raw_trace = cpptrace::generate_raw_trace(skip, max_depth);
    pcs = std::move(raw_trace.frames);
#else
    // allocate space for the result of backtrace(), including the stuff on
    //  the front we're going to skip
    assert(sizeof(void *) == sizeof(intptr_t));
    pcs.clear();
    pcs.resize(max_depth + skip, 0);
#ifdef REALM_ON_WINDOWS
    int count = 0; // TODO: StackWalk appears to be the right API call?
#else
    int count = backtrace((void **)pcs.data(), max_depth + skip);
#endif
    assert(count >= 0);

    if(count > skip) {
      pcs.erase(pcs.begin() + count, pcs.end());
      pcs.erase(pcs.begin(), pcs.begin() + skip);
    } else {
      pcs.clear();
    }

#endif /* REALM_USE_CPPTRACE */

    symbols.clear();

    // recompute the hash too
    pc_hash = compute_hash();
  }

  // attempts to map the pointers in the back trace to symbol names - this can be
  //   more expensive
  void Backtrace::lookup_symbols(void)
  {
    // have we already done the lookup?
    if(!symbols.empty()) return;

    symbols.reserve(pcs.size());

#ifdef REALM_USE_CPPTRACE
    cpptrace::raw_trace raw_trace;
    raw_trace.frames = std::move(pcs);
    for(const cpptrace::stacktrace_frame &frame : raw_trace.resolve().frames) {
      // the resolved trace has more frames than the raw trace,
      // so we have to use push_back
      // we do not need "unknown symbol" here becasue frame.to_string
      // always returns something
      symbols.push_back(frame.to_string(true));
    }
    pcs = std::move(raw_trace.frames);
#else
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
    for(size_t i = 0; i < pcs.size(); i++) {
      // try backtrace_symbols() first
      char **s = backtrace_symbols((void * const *)&(pcs[i]), 1);
      if(s) {
        symbols.emplace_back(s[0]);
        free(s);
      } else {
        symbols.emplace_back("unknown symbol");
      }
    }
#endif /* defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD) */
#endif /* REALM_USE_CPPTRACE */
  }

  std::ostream& operator<<(std::ostream& os, const Backtrace& bt)
  {
    os << "stack trace: " << bt.symbols.size() << " frames" << std::endl;
#ifdef REALM_USE_CPPTRACE
    for(const std::string &sym : bt.symbols) {
      os << sym << std::endl;
    }
#else
    char *demangle_buffer = 0;
    size_t demangle_len = 0;

    for(size_t i = 0; i < bt.pcs.size(); i++) {
      os << "  [" << i << "] = ";
      if(!bt.symbols.empty() && !bt.symbols[i].empty()) {
	char *s = (char *)(bt.symbols[i].c_str());
        bool print_raw = true;
        char *lp = s;
#ifdef REALM_HAVE_CXXABI_H
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
#endif
        if(print_raw) {
          os << bt.symbols[i];
        }
      } else {
        os << std::hex << std::setfill('0') << std::setw(sizeof(uintptr_t)*2) << bt.pcs[i];
        os << std::dec << std::setfill(' ');
      }
      os << std::endl;
    }

    if(demangle_buffer)
      free(demangle_buffer);
#endif /* REALM_USE_CPPTRACE */

    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExecutionException

  ExecutionException::ExecutionException(int _error_code,
					 const void *_detail_data, size_t _detail_size,
					 bool capture_backtrace /*= true*/)
    : error_code(_error_code)
    , details(_detail_data, _detail_size)
  {
    if(capture_backtrace)
      backtrace.capture_backtrace(1); // skip this frame
  }
  
  ExecutionException::~ExecutionException(void) REALM_NOEXCEPT
  {}

  void ExecutionException::populate_profiling_measurements(ProfilingMeasurementCollection& pmc) const
  {
    if(!backtrace.empty() && 
       pmc.wants_measurement<ProfilingMeasurements::OperationBacktrace>()) {
      ProfilingMeasurements::OperationBacktrace b;
      b.backtrace = backtrace;
      // there's a good chance the profiler is not on the same node, so look up symbols
      //  now
      b.backtrace.lookup_symbols();
      pmc.add_measurement(b);
    }
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CancellationException

  CancellationException::CancellationException(void)
    : ExecutionException(Faults::ERROR_CANCELLED, 0, 0)
  {}

  const char *CancellationException::what(void) const REALM_NOEXCEPT
  {
    return "CancellationException";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PoisonedEventException

  PoisonedEventException::PoisonedEventException(Event _event)
    : ExecutionException(Faults::ERROR_POISONED_EVENT, &_event, sizeof(Event))
    , event(_event)
  {}

  const char *PoisonedEventException::what(void) const REALM_NOEXCEPT
  {
    return "PoisonedEventException";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ApplicationException

  ApplicationException::ApplicationException(int _error_code,
					     const void *_detail_data, size_t _detail_size)
    : ExecutionException(_error_code, _detail_data, _detail_size)
  {}

  const char *ApplicationException::what(void) const REALM_NOEXCEPT
  {
    return "ApplicationException";
  }


}; // namespace Realm
