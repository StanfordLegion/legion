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
### Driver
###

import itertools, optparse, os, platform, shutil, subprocess, sys, tempfile
from . import passes

def platform_cxx():
    if platform.system() == 'Darwin':
        return 'clang++'
    return 'g++'

def platform_cxxflags():
    # Currently all platforms need c++11. For backwards compatibility
    # with versions before c++11 was finalized, the c++0x flag is used
    # instead.
    if platform.system() == 'Darwin':
        return ['-std=c++0x', '-stdlib=libc++']
    return ['-std=c++0x']

def platform_ldflags():
    if platform.system() == 'Darwin':
        return ['-stdlib=libc++', '-lpthread']
    return ['-lrt', '-lpthread']

def compile_to_cpp(input_filename, output_filename, search_path):
    with open(input_filename, 'rb') as input_file:
        program = passes.parse(input_file)
    cpp_result_text = passes.compile(program, search_path)
    with open(output_filename, 'wb') as output_file:
        output_file.write(cpp_result_text)

def compile_to_obj(input_filename, output_filename, search_path,
                   legion_runtime_dir, stdout, stderr):
    # Dump C++ to a temp directory, and build
    temp_dir = tempfile.mkdtemp()
    temp_cpp_basename = 'input.cc'
    temp_cpp_path = os.path.join(temp_dir, temp_cpp_basename)
    try:
        compile_to_cpp(input_filename, temp_cpp_path, search_path)
        proc = subprocess.Popen(
            [platform_cxx()] +
            platform_cxxflags() +
            ['-I', legion_runtime_dir] +
            [option
             for path in search_path
             for option in ('-I', path)] +
            ['-c', temp_cpp_path,
             '-o', output_filename],
            stdout = stdout, stderr = stderr)
        output, errors = proc.communicate()
        if proc.returncode != 0:
            raise Exception(output, errors)
    finally:
        shutil.rmtree(temp_dir)

_legion_extensions = ('.lg',)

# This is the list of file extensions recognized as (preproccessed)
# C++ by the GNU compiler.
_cpp_extensions = ('.cc', '.cp', '.cxx', '.cpp', 'c++', '.C')

def unique_filenames(filenames, extension, existing):
    result = []
    existing = set(existing)
    for filename in filenames:
        base = os.path.splitext(filename)[0]

        attempt = ''.join([base, extension])
        if attempt not in existing:
            result.append(attempt)
            existing.add(attempt)
            continue

        for i in itertools.count(1):
            attempt = ''.join([base, str(i), extension])
            if attempt not in existing:
                result.append(attempt)
                existing.add(attempt)
                break

    return result, existing

def compile_to_exe(input_filenames, output_filename, thread_count,
                   clean_first, search_path, legion_runtime_dir, stdout, stderr):
    # Filter input files according to file type
    input_legion_filenames = [
        filename
        for filename in input_filenames
        if os.path.splitext(filename)[1] in _legion_extensions]

    input_cpp_filenames = [
        filename
        for filename in input_filenames
        if os.path.splitext(filename)[1] in _cpp_extensions]

    assert len(set(input_legion_filenames + input_cpp_filenames)) == \
        len(input_legion_filenames + input_cpp_filenames)
    assert set(input_legion_filenames + input_cpp_filenames) == \
        set(input_filenames)

    # Dump C++ files and Makefile to a temp directory, build, and copy back
    temp_dir = tempfile.mkdtemp()
    existing = set(['Makefile', 'output'])
    temp_legion_basenames, existing = unique_filenames(
        [os.path.basename(filename) for filename in input_legion_filenames],
        '.cc',
        existing)
    temp_legion_paths = [
        os.path.join(temp_dir, basename)
        for basename in temp_legion_basenames]
    temp_cpp_basenames, existing = unique_filenames(
        [os.path.basename(filename) for filename in input_cpp_filenames],
        '.cc',
        existing)
    temp_cpp_paths = [
        os.path.join(temp_dir, basename)
        for basename in temp_cpp_basenames]
    temp_output_basename = 'output'
    temp_output_path = os.path.join(temp_dir, temp_output_basename)
    makefile_template_path = os.path.join(legion_runtime_dir, '..', 'apps', 'Makefile.template')
    temp_makefile_path = os.path.join(temp_dir, 'Makefile')
    try:
        shutil.copy(makefile_template_path, temp_makefile_path)
        for input_filename, temp_cpp_path in \
                zip(input_cpp_filenames, temp_cpp_paths):
            shutil.copy(input_filename, temp_cpp_path)
        for input_filename, temp_cpp_path in \
                zip(input_legion_filenames, temp_legion_paths):
            compile_to_cpp(input_filename, temp_cpp_path, search_path)
        make_args = [
            '--file=%s' % temp_makefile_path,
            '--directory=%s' % temp_dir,
            'OUTFILE=%s' % temp_output_basename,
            'GEN_SRC=%s' % ' '.join(temp_legion_basenames + temp_cpp_basenames),
            'GCC=%s' % platform_cxx(),
            'CC_FLAGS=%s' % ' '.join(
                platform_cxxflags() +
                [option
                 for path in search_path
                 for option in ('-I', path)]),
            'LD_FLAGS=%s' % ' '.join(
                platform_ldflags()),
            'SHARED_LOWLEVEL=1',
            ] + (
            ['-j', str(thread_count)] if thread_count is not None else ['-j'])
        make_targets = []
        if clean_first:
            make_targets.append('clean')
        make_targets.append('all')
        proc = subprocess.Popen(
            ['make'] + make_args + make_targets,
            stdout = stdout, stderr = stderr)
        output, errors = proc.communicate()
        if proc.returncode != 0:
            raise Exception(output, errors)
        shutil.copy(temp_output_path, output_filename)
    finally:
        shutil.rmtree(temp_dir)

def driver(argv, verbose = True):
    parser = optparse.OptionParser(description = 'Compiles a Legion source file.')
    parser.add_option('-c', dest = 'compile_to_obj', action = 'store_true', default = False)
    parser.add_option('-S', dest = 'compile_to_cpp', action = 'store_true', default = False)
    parser.add_option('-o', dest = 'output', default = 'a.out')
    parser.add_option('-j', dest = 'thread_count', default = None, type = int)
    parser.add_option('--clean', dest = 'clean_first', action = 'store_true', default = False)
    (options, args) = parser.parse_args(argv[1:])
    if len(args) == 0:
        parser.error('No input files')
    if len(args) != 1 and (options.compile_to_cpp or options.compile_to_obj):
        parser.error('No input files')

    search_path = tuple(os.path.abspath(os.path.dirname(arg)) for arg in args)

    # For the duration of compilation, replace stdout and stderr with
    # the provided values.
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        if verbose:
            stdout, stderr = None, None
        if not verbose:
            sys.stdout = open('/dev/null', 'wb')
            sys.stderr = sys.stdout
            stdout, stderr = subprocess.PIPE, subprocess.STDOUT

        if options.compile_to_cpp:
            compile_to_cpp(
                input_filename = args[0],
                output_filename = options.output,
                search_path = search_path)
        elif options.compile_to_obj:
            if 'LG_RT_DIR' not in os.environ:
                parser.error('LG_RT_DIR variable is not defined')
            compile_to_obj(
                input_filename = args[0],
                output_filename = options.output,
                search_path = search_path,
                legion_runtime_dir = os.environ['LG_RT_DIR'],
                stdout = stdout,
                stderr = stderr)
        else:
            if 'LG_RT_DIR' not in os.environ:
                parser.error('LG_RT_DIR variable is not defined')
            compile_to_exe(
                input_filenames = args,
                output_filename = options.output,
                thread_count = options.thread_count,
                clean_first = options.clean_first,
                search_path = search_path,
                legion_runtime_dir = os.environ['LG_RT_DIR'],
                stdout = stdout,
                stderr = stderr)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
