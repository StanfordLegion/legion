#!/usr/bin/env python

# Copyright 2013 Stanford University and Los Alamos National Security, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

###
### Command Line Options Parser
###

import copy, argparse, os, sys

TARGET_EXE = 'exe'
TARGET_OBJ = 'obj'
TARGET_CPP = 'cpp'

LOG_NONE = 0
LOG_WARNING = 1
LOG_ALL = 2

def with_ext(filename, ext):
    return ''.join([os.path.splitext(filename)[0], ext])

class Options:
    def __init__(self, input_filenames, output_filename, target, debug,
                 pointer_checks, leaf_task_optimization, thread_count,
                 clean_first, save_temps,
                 search_path, legion_runtime_dir, log_level):
        self.input_filenames = input_filenames
        self.output_filename = output_filename
        self.target = target
        self.debug = debug
        self.pointer_checks = pointer_checks
        self.leaf_task_optimization = leaf_task_optimization
        self.thread_count = thread_count
        self.clean_first = clean_first
        self.save_temps = save_temps
        self.search_path = search_path
        self.legion_runtime_dir = legion_runtime_dir
        self.log_level = log_level
    def with_single_source(self, input_filename, output_filename, target):
        opts = copy.copy(self)
        opts.input_filenames = [input_filename]
        opts.output_filename = output_filename
        opts.target = target
        return opts
    def with_search_path(self, search_path):
        opts = copy.copy(self)
        opts.search_path = search_path
        return opts
    def allow_warning(self):
        return self.log_level >= LOG_WARNING
    def allow_all(self):
        return self.log_level >= LOG_ALL

def parse_options(argv = None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description = 'Legion compiler frontend')
    parser.add_argument('-S',
                        action = 'store_true',
                        help = 'compile to a C++ source file',
                        dest = 'compile_to_cpp')
    parser.add_argument('-c',
                        action = 'store_true',
                        help = 'compile to an object file',
                        dest = 'compile_to_obj')
    parser.add_argument('-o',
                        nargs = '?',
                        help = 'location for output file',
                        dest = 'output')
    parser.add_argument('-g',
                        action = 'store_true',
                        help = 'debugging symbols',
                        dest = 'debug')
    parser.add_argument('--pointer-checks',
                        action = 'store_true',
                        help = 'dynamic pointer checks',
                        dest = 'pointer_checks')
    parser.add_argument('--leaf-tasks',
                        action = 'store_true',
                        help = 'aggressive leaf task optimization (potentially unsafe)',
                        dest = 'leaf_task_optimization')
    parser.add_argument('-j',
                        nargs = '?',
                        type = int,
                        help = 'number threads used to compile',
                        dest = 'thread_count')
    parser.add_argument('--clean',
                        action = 'store_true',
                        help = 'recompile the Legion runtime ',
                        dest = 'clean_first')
    parser.add_argument('--save-temps',
                        action = 'store_true',
                        help = 'save any temporary files',
                        dest = 'save_temps')
    parser.add_argument('-v',
                        action = 'store_true',
                        help = 'display verbose output',
                        dest = 'verbose')
    parser.add_argument('-q',
                        action = 'store_true',
                        help = 'diplay no output',
                        dest = 'quiet')
    parser.add_argument('input',
                        nargs = '*',
                        help = 'input files')
    args = parser.parse_args(argv[1:])

    if len(args.input) == 0:
        parser.error('No input files')
    if len(args.input) != 1 and (args.compile_to_cpp or args.compile_to_obj):
        parser.error('Too many input files')
    if args.compile_to_cpp and args.compile_to_obj:
        parser.error('Multiple, conflicting targets')
    if args.verbose and args.quiet:
        parser.error('Multiple, conflicting verbosity options')
    if 'LG_RT_DIR' not in os.environ:
        parser.error('LG_RT_DIR variable is not defined')
    else:
        legion_runtime_dir = os.environ['LG_RT_DIR']

    target = TARGET_EXE
    if args.compile_to_cpp:
        target = TARGET_CPP
    elif args.compile_to_obj:
        target = TARGET_OBJ

    log_level = LOG_WARNING
    if args.verbose:
        log_level = LOG_ALL
    elif args.quiet:
        log_level = LOG_NONE

    output_filename = args.output
    if target == TARGET_CPP:
        if output_filename is None:
            output_filename = with_ext(os.path.basename(args.input[0]), '.lg.cc')
        header_filename = with_ext(output_filename, '.h')
        output_filename = (header_filename, output_filename)
    elif target == TARGET_OBJ:
        if output_filename is None:
            output_filename = with_ext(os.path.basename(args.input[0]), '.lg.o')
        header_filename = with_ext(output_filename, '.h')
        output_filename = (header_filename, output_filename)
    elif target == TARGET_EXE:
        if output_filename is None:
            output_filename = 'a.out'
    else:
        assert False

    search_path = tuple(os.path.abspath(os.path.dirname(arg)) for arg in args.input)

    return Options(
        input_filenames = args.input,
        output_filename = output_filename,
        target = target,
        debug = args.debug,
        pointer_checks = args.pointer_checks,
        leaf_task_optimization = args.leaf_task_optimization,
        thread_count = args.thread_count,
        clean_first = args.clean_first,
        save_temps = args.save_temps,
        search_path = search_path,
        legion_runtime_dir = legion_runtime_dir,
        log_level = log_level)

# Builds a fake set of options for use in the test suite and other
# scenarios where a full set of options might not be available.
def build_fake_options(filename, verbose):
    if filename is not None:
        search_path = (os.path.abspath(os.path.dirname(filename)),)
    else:
        search_path = (os.getcwd(),)

    if 'LG_RT_DIR' not in os.environ:
        legion_runtime_dir = None
    else:
        legion_runtime_dir = os.environ['LG_RT_DIR']

    return Options(
        input_filenames = [filename],
        output_filename = os.devnull,
        target = TARGET_CPP,
        debug = True,
        pointer_checks = False,
        leaf_task_optimization = False,
        thread_count = 1,
        clean_first = False,
        save_temps = False,
        search_path = search_path,
        legion_runtime_dir = legion_runtime_dir,
        log_level = LOG_ALL if verbose else LOG_NONE)
