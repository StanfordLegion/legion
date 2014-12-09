#!/usr/bin/env python

# Copyright 2014 Stanford University and Los Alamos National Security, LLC
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

import itertools, multiprocessing, os, platform, shutil, subprocess, sys, tempfile
from . import options, passes

class CompilerException (Exception):
    def __init__(self, stdout, stderr, saved_temps):
        Exception.__init__(self, stdout, stderr)
        self.stdout = stdout
        self.stderr = stderr
        self.saved_temps = saved_temps

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

def compile_to_cpp(opts):
    assert len(opts.input_filenames) == 1
    input_path = opts.input_filenames[0]
    output_h_path, output_cpp_path = opts.output_filename

    with open(input_path, 'rb') as input_file:
        program = passes.parse(input_file)
    cpp_header, cpp_source = passes.compile(program, opts)
    with open(output_h_path, 'wb') as output_file:
        output_file.write(cpp_header)
    with open(output_cpp_path, 'wb') as output_file:
        output_file.write(cpp_source)
    return []

def compile_to_obj(opts, stdout, stderr):
    assert len(opts.input_filenames) == 1
    input_path = opts.input_filenames[0]
    output_h_path, output_obj_path = opts.output_filename

    # Dump C++ to a temp directory, and build
    temp_dir = tempfile.mkdtemp()
    temp_cpp_basename = options.with_ext(os.path.basename(output_obj_path), '.cc')
    temp_cpp_path = os.path.join(temp_dir, temp_cpp_basename)
    temp_h_path = os.path.join(temp_dir, os.path.basename(output_h_path))

    # If the compiler failed to produce any temporary files, no reason
    # to keep around the temporary directory.
    try:
        compile_to_cpp(opts.with_single_source(
                input_path, (temp_h_path, temp_cpp_path), options.TARGET_CPP))
    except:
        shutil.rmtree(temp_dir)
        raise

    try:
        proc = subprocess.Popen(
            [platform_cxx()] +
            platform_cxxflags() +
            (['-g'] if opts.debug else []) +
            ['-I', opts.legion_runtime_dir] +
            [option
             for path in opts.search_path
             for option in ('-I', path)] +
            ['-c', temp_cpp_path,
             '-o', output_obj_path],
            stdout = stdout, stderr = stderr)
        output, errors = proc.communicate()
        if proc.returncode != 0:
            raise CompilerException(output, errors, [temp_dir] if opts.save_temps else [])
        os.rename(temp_h_path, output_h_path)
    finally:
        if not opts.save_temps:
            shutil.rmtree(temp_dir)
            temp_dir = None
    return [temp_dir] if temp_dir is not None else []

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

def compile_to_exe(opts, stdout, stderr):
    # Filter input files according to file type
    input_legion_filenames = [
        filename
        for filename in opts.input_filenames
        if os.path.splitext(filename)[1] in _legion_extensions]

    input_cpp_filenames = [
        filename
        for filename in opts.input_filenames
        if os.path.splitext(filename)[1] in _cpp_extensions]

    assert len(set(input_legion_filenames + input_cpp_filenames)) == \
        len(input_legion_filenames + input_cpp_filenames)
    assert set(input_legion_filenames + input_cpp_filenames) == \
        set(opts.input_filenames)

    # Dump C++ files and Makefile to a temp directory, build, and copy back
    temp_dir = tempfile.mkdtemp()
    existing = set(['Makefile', 'output'])
    temp_legion_basenames, existing = unique_filenames(
        [os.path.basename(filename) for filename in input_legion_filenames],
        '.lg.cc',
        existing)
    temp_legion_paths = [
        os.path.join(temp_dir, basename)
        for basename in temp_legion_basenames]
    temp_legion_h_paths = [
        options.with_ext(path, '.h')
        for path in temp_legion_paths]
    temp_cpp_basenames, existing = unique_filenames(
        [os.path.basename(filename) for filename in input_cpp_filenames],
        '.cc',
        existing)
    temp_cpp_paths = [
        os.path.join(temp_dir, basename)
        for basename in temp_cpp_basenames]
    temp_output_basename = 'output'
    temp_output_path = os.path.join(temp_dir, temp_output_basename)
    makefile_template_path = os.path.join(opts.legion_runtime_dir, '..', 'apps', 'Makefile.template')
    temp_makefile_path = os.path.join(temp_dir, 'Makefile')

    threads = []
    if opts.thread_count is not None:
        threads = [str(opts.thread_count)]
    else:
        try:
            threads = [str(multiprocessing.cpu_count())]
        except NotImplementedError:
            pass

    # If the compiler failed to produce any temporary files, no reason
    # to keep around the temporary directory.
    try:
        for input_filename, temp_h_path, temp_cpp_path in \
                zip(input_legion_filenames, temp_legion_h_paths, temp_legion_paths):
            compile_to_cpp(opts.with_single_source(
                    input_filename, (temp_h_path, temp_cpp_path), options.TARGET_CPP))
    except:
        shutil.rmtree(temp_dir)
        raise

    try:
        shutil.copy(makefile_template_path, temp_makefile_path)
        for input_filename, temp_cpp_path in \
                zip(input_cpp_filenames, temp_cpp_paths):
            shutil.copy(input_filename, temp_cpp_path)
        make_env = dict(
            os.environ.items() +
            [('OUTFILE', temp_output_basename),
             ('GEN_SRC', ' '.join(temp_legion_basenames + temp_cpp_basenames)),
             ('GCC', platform_cxx()),
             ('CC_FLAGS', ' '.join(
                        platform_cxxflags() +
                        [option
                         for path in opts.search_path
                         for option in ('-I', path)])),
            ('DEBUG', ('1' if opts.debug else '0')),
            ('SHARED_LOWLEVEL', '1')])
        make_args = [
            '--file=%s' % temp_makefile_path,
            '--directory=%s' % temp_dir,
            'LD_FLAGS=%s' % ' '.join(platform_ldflags()),
            '-j'] + threads
        make_targets = []
        if opts.clean_first:
            make_targets.append('clean')
        make_targets.append('all')
        for make_target in make_targets:
            proc = subprocess.Popen(
                ['make'] + make_args + [make_target],
                env = make_env,
                stdout = stdout, stderr = stderr)
            output, errors = proc.communicate()
            if proc.returncode != 0:
                raise CompilerException(output, errors, [temp_dir] if opts.save_temps else [])
        shutil.copy(temp_output_path, opts.output_filename)
    finally:
        if not opts.save_temps:
            shutil.rmtree(temp_dir)
            temp_dir = None
    return [temp_dir] if temp_dir is not None else []

def driver(argv = None):
    opts = options.parse_options(argv)

    if opts.allow_all():
        stdout, stderr = None, None
    else:
        stdout, stderr = subprocess.PIPE, subprocess.STDOUT

    if opts.target == options.TARGET_CPP:
        return compile_to_cpp(opts)
    elif opts.target == options.TARGET_OBJ:
        return compile_to_obj(opts, stdout, stderr)
    elif opts.target == options.TARGET_EXE:
        return compile_to_exe(opts, stdout, stderr)
    else:
        assert False
