#!/usr/bin/env python3

# Copyright 2022 Stanford University
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

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import subprocess
from pathlib import Path

def find_header(filename, search_dir):
    def try_prefix(filename, prefix_dir):
        legion_h_path = os.path.join(prefix_dir, filename)
        if os.path.exists(legion_h_path):
            return prefix_dir, legion_h_path

    # We should always be in an in-source build, so just find the file
    # relative to the source directory.
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    runtime_dir = os.path.join(root_dir, search_dir)
    result = try_prefix(filename, runtime_dir)
    if result:
        return result

    raise Exception('Unable to locate header file:' + filename + " in:" + runtime_dir)

def build_cffi(libname, defines_dir, output_dir, header_files, header_file_search_dirs, output_file):
    headers_prefix_dir = []
    headers_path = []
    assert len(header_files) == len(header_file_search_dirs)
    for (header_file, search_dir) in zip(header_files, header_file_search_dirs):
        prefix_dir, h_path = find_header(header_file, search_dir)
        headers_prefix_dir.append(prefix_dir)
        headers_path.append(h_path)

    if defines_dir is not None:
        # For CMake, need to be told where the defines directory is:
        build_flags = ['-I', defines_dir]
    else:
        # For Make, legion_defines.h is in the source directory:
        build_flags = ['-I', os.path.dirname(os.path.realpath(__file__))]

    # Check to see if the user specified a C compiler with the CC environment variable, if not assume there is a built-in C compiler
    compiler = os.getenv('CC', 'cc')

    header_cmd = []
    for header_prefix_dir in headers_prefix_dir:
        header_cmd.append('-I')
        header_cmd.append(header_prefix_dir)

    compiler_cmd = [compiler] + header_cmd + build_flags + ['-DLEGION_USE_PYTHON_CFFI', '-E', '-P', headers_path[0]]

    header = subprocess.check_output(compiler_cmd).decode('utf-8')

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'legion_cffi.py.in')) as f:
        content = f.read()
    content = content.format(header=repr(header), libname=repr(libname))

    if output_dir is None:
        output_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(output_dir, output_file), 'wb') as f:
        f.write(content.encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical', action="store_true", required=False)
    parser.add_argument('--libname', required=False)
    parser.add_argument('--defines-dir', required=False)
    parser.add_argument('--output-dir', required=False)
    args = parser.parse_args()

    if args.canonical:
        build_cffi(args.libname, args.defines_dir, args.output_dir, ['canonical_python.h', 'legion.h'], [os.path.join('bindings', 'python'), 'runtime'], 'legion_canonical_cffi.py')
    else:
        build_cffi(None, args.defines_dir, args.output_dir, ['legion.h'], ['runtime'], 'legion_builtin_cffi.py')
