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

#include "realm/faults.h"
#include "realm/profiling.h"
#ifdef REALM_USE_LIBDW
#include "realm/mutex.h"
#endif

#include <stdlib.h>
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS)
// FreeBSD defines alloca() in stdlib.h
#include <alloca.h>
#endif
#include <assert.h>
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#ifdef REALM_USE_UNWIND // enabled by default
#include <unwind.h>
#include <limits>
#endif /* REALM_USE_UNWIND */
#include <execinfo.h>
#ifdef REALM_USE_LIBDW
#include <dwarf.h>
#include <elfutils/libdw.h>
#include <elfutils/libdwfl.h>
#include <unistd.h>
#endif /* REALM_USE_LIBDW */
#endif /* defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD) */
#ifdef REALM_HAVE_CXXABI_H
#include <cxxabi.h>
#endif
#include <iomanip>

#ifdef ERROR_CANCELLED
#undef ERROR_CANCELLED
#endif

namespace Realm {

#ifdef REALM_USE_LIBDW
  static Mutex backtrace_mutex;
#endif

// unwind.h is not supported by Windows
#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#ifdef REALM_USE_UNWIND
  ////////////////////////////////////////////////////////////////////////
  //
  // class UnwindTrace

  class UnwindTrace {
  public:
    UnwindTrace(size_t _depth, uintptr_t *_rawptrs);

    size_t backtrace(void);

  private:
    ssize_t index;
    size_t depth;
    uintptr_t *rawptrs;

    static _Unwind_Reason_Code backtrace_trampoline(_Unwind_Context *ctx, void *self);

    _Unwind_Reason_Code get_backtrace_per_line(_Unwind_Context *ctx);
  };

  UnwindTrace::UnwindTrace(size_t _depth, uintptr_t *_rawptrs)
    : index(-1), depth(_depth), rawptrs(_rawptrs)
  {}

  size_t UnwindTrace::backtrace(void)
  {
    _Unwind_Backtrace(&this->backtrace_trampoline, this);
    if (index == -1) {
      // _Unwind_Backtrace has failed to obtain any backtraces
      return 0;
    } else {
      return static_cast<size_t>(index);
    }
  }

  /*static*/ _Unwind_Reason_Code UnwindTrace::backtrace_trampoline(_Unwind_Context *ctx, void *self) 
  {
    return (static_cast<UnwindTrace *>(self))->get_backtrace_per_line(ctx);
  }

  _Unwind_Reason_Code UnwindTrace::get_backtrace_per_line(_Unwind_Context *ctx) 
  {
    if (index >= 0 && static_cast<size_t>(index) >= depth)
      return _URC_END_OF_STACK;

    int ip_before_instruction = 0;
    uintptr_t ip = _Unwind_GetIPInfo(ctx, &ip_before_instruction);

    if (!ip_before_instruction) {
      // calculating 0-1 for unsigned, looks like a possible bug to sanitizers,
      // so let's do it explicitly:
      if (ip == 0) {
        ip = std::numeric_limits<uintptr_t>::max(); // set it to 0xffff... (as
                                                    // from casting 0-1)
      } else {
        ip -= 1; // else just normally decrement it (no overflow/underflow will
                  // happen)
      }
    }

    if (index >= 0) { // ignore first frame.
      rawptrs[index] = ip;
    }

    index += 1;
    return _URC_NO_REASON;
  }
#endif
#endif


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
#ifdef REALM_ON_WINDOWS
    int count = 0; // TODO: StackWalk appears to be the right API call?
#elif defined(REALM_USE_UNWIND)
    UnwindTrace unwind_trace((max_depth + skip), rawptrs);
    int count = unwind_trace.backtrace();
#else
    int count = backtrace((void **)rawptrs, max_depth + skip);
#endif
    assert(count >= 0);
    
    pcs.clear();
    symbols.clear();

    if(count > skip)
      pcs.insert(pcs.end(), rawptrs + skip, rawptrs + count);

    // recompute the hash too
    pc_hash = compute_hash();
  }

#ifdef REALM_USE_LIBDW
  static bool die_has_pc(Dwarf_Die *die, Dwarf_Addr pc) 
  {
    Dwarf_Addr low, high;

    // continuous range
    if (dwarf_hasattr(die, DW_AT_low_pc) && dwarf_hasattr(die, DW_AT_high_pc)) {
      if (dwarf_lowpc(die, &low) != 0) {
        return false;
      }
      if (dwarf_highpc(die, &high) != 0) {
        Dwarf_Attribute attr_mem;
        Dwarf_Attribute *attr = dwarf_attr(die, DW_AT_high_pc, &attr_mem);
        Dwarf_Word value;
        if (dwarf_formudata(attr, &value) != 0) {
          return false;
        }
        high = low + value;
      }
      return pc >= low && pc < high;
    }

    // non-continuous range.
    Dwarf_Addr base;
    ptrdiff_t offset = 0;
    while ((offset = dwarf_ranges(die, offset, &base, &low, &high)) > 0) {
      if (pc >= low && pc < high) {
        return true;
      }
    }
    return false;
  }

  static Dwarf_Die *find_fundie_by_pc(Dwarf_Die *parent_die, Dwarf_Addr pc,
                                      Dwarf_Die *result) 
  {
    if (dwarf_child(parent_die, result) != 0) {
      return 0;
    }

    Dwarf_Die *die = result;
    do {
      switch (dwarf_tag(die)) {
      case DW_TAG_subprogram:
      case DW_TAG_inlined_subroutine:
        if (die_has_pc(die, pc)) {
          return result;
        }
      };
      bool declaration = false;
      Dwarf_Attribute attr_mem;
      dwarf_formflag(dwarf_attr(die, DW_AT_declaration, &attr_mem),
                     &declaration);
      if (!declaration) {
        // let's be curious and look deeper in the tree,
        // function are not necessarily at the first level, but
        // might be nested inside a namespace, structure etc.
        Dwarf_Die die_mem;
        Dwarf_Die *indie = find_fundie_by_pc(die, pc, &die_mem);
        if (indie) {
          *result = die_mem;
          return result;
        }
      }
    } while (dwarf_siblingof(die, result) == 0);
    return 0;
  }
#endif

  // attempts to map the pointers in the back trace to symbol names - this can be
  //   more expensive
  void Backtrace::lookup_symbols(void)
  {
    // have we already done the lookup?
    if(!symbols.empty()) return;

    symbols.resize(pcs.size(), "unknown symbol");

#if defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD)
#ifdef REALM_USE_LIBDW
    filenames.resize(pcs.size(), "unknown file");
    line_numbers.resize(pcs.size(), 0);
    AutoLock<> al(backtrace_mutex);

    // Initialize Dwfl 
    Dwfl_Callbacks proc_callbacks;
    proc_callbacks.find_debuginfo = dwfl_standard_find_debuginfo,
    proc_callbacks.debuginfo_path = nullptr,
    proc_callbacks.find_elf = dwfl_linux_proc_find_elf;
    Dwfl *dwfl_handle = dwfl_begin(&proc_callbacks);
    assert(dwfl_handle != nullptr);

    // use the current process.
    dwfl_report_begin(dwfl_handle);
    int r = dwfl_linux_proc_report(dwfl_handle, getpid());
    dwfl_report_end(dwfl_handle, NULL, NULL);
    assert(r >= 0);

    for(size_t i = 0; i < pcs.size(); i++) {
      Dwarf_Addr trace_addr = static_cast<Dwarf_Addr>(Backtrace::pcs[i]);
      Dwfl_Module* mod = dwfl_addrmodule(dwfl_handle, trace_addr);
      if (mod) {
        const char *sym_name = dwfl_module_addrname(mod, trace_addr);
        if (sym_name) {
          symbols[i].assign(sym_name);
        }

        Dwarf_Addr mod_bias = 0;
        Dwarf_Die *cudie = dwfl_module_addrdie(mod, trace_addr, &mod_bias);

        if (!cudie) {
          while ((cudie = dwfl_module_nextcu(mod, cudie, &mod_bias))) {
            Dwarf_Die die_mem;
            Dwarf_Die *fundie = find_fundie_by_pc(cudie, trace_addr - mod_bias, &die_mem);
            if (fundie) {
              break;
            }
          }
        }
        if (!cudie) {
          continue;
        }
        Dwarf_Line *srcloc = dwarf_getsrc_die(cudie, trace_addr - mod_bias);
        if (srcloc) {
          const char *srcfile = dwarf_linesrc(srcloc, 0, 0);
          if (srcfile) {
            filenames[i].assign(srcfile);
          }
          int line = 0;
          dwarf_lineno(srcloc, &line);
          line_numbers[i] = line;
        }
      }
    }
    dwfl_end(dwfl_handle);
#else
    for(size_t i = 0; i < pcs.size(); i++) {
      // try backtrace_symbols() first
      char **s = backtrace_symbols((void * const *)&(pcs[i]), 1);
      if(s) {
	symbols[i].assign(s[0]);
	free(s);
      }
    }
#endif /* REALM_USE_LIBDW */
#endif /* defined(REALM_ON_LINUX) || defined(REALM_ON_MACOS) || defined(REALM_ON_FREEBSD) */
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
        bool print_raw = true;
#ifdef REALM_USE_LIBDW
#ifdef REALM_HAVE_CXXABI_H
        int status = -4;
        char *result = abi::__cxa_demangle(s, demangle_buffer, &demangle_len, &status);
        if (status == 0) {
	  demangle_buffer = result;
          os << demangle_buffer << " at ";
          print_raw = false;
        }
#endif
        // we can not demangle the object symbol, so print it directly
        if (print_raw) {
          os << s << " at ";
        }
        if (!bt.filenames[i].empty()) {
          os << bt.filenames[i] << ":" << bt.line_numbers[i] << " ";
        }
        os << "[" << bt.pcs[i] << "]";
#else
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
        if(print_raw)
	  os << bt.symbols[i];
#endif /* REALM_USE_LIBDW */
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
  
  ExecutionException::~ExecutionException(void) throw()
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

  const char *CancellationException::what(void) const throw()
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

  const char *PoisonedEventException::what(void) const throw()
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

  const char *ApplicationException::what(void) const throw()
  {
    return "ApplicationException";
  }


}; // namespace Realm
