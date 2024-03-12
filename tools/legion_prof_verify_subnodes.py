#!/usr/bin/env python3

# Copyright 2024 Stanford University, NVIDIA Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, subprocess, argparse
from pathlib import Path
import fnmatch
import shutil
import sys
import os

from typing import NoReturn

if __name__ == '__main__':
    class MyParser(argparse.ArgumentParser):
        def error(self, message: str) -> NoReturn:
            self.print_usage(sys.stderr)
            print('error: %s' % message, file=sys.stderr)
            print('hint: invoke %s -h for a detailed description of all arguments' % self.prog, file=sys.stderr)
            sys.exit(2)
    parser = MyParser(
        description='Legion Prof: application profiler')
    parser.add_argument(
        dest='filenames', nargs='+',
        help='input Legion Prof log filenames')
    parser.add_argument(
        '--outdir', dest='outdir', action='store',
        type=str,
        help='dir of output files')
    parser.add_argument(
        '--runtimedir', dest='runtimedir', action='store',
        type=str,
        help='root of legion runtime dir')
    parser.add_argument(
        '--nodes', dest='nodes', action='store',
        type=str,
        help='a list of nodes that will be visualized')
    parser.add_argument(
        '--rust', dest='rust', action='store_true',
        help='rust or python')
    parser.add_argument(
        '--rustexe', dest='rustdir', action='store',
        type=str,
        help='path to the legion_prof executable')
    parser.add_argument(
        '--cleanup', dest='cleanup', action='store_true',
        help='cleanup output dir')
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true',
        help='verbose')
    args = parser.parse_args()
    filenames = args.filenames
    outdir = args.outdir
    runtimedir = args.runtimedir
    nodes = args.nodes
    rust = args.rust
    rustdir = args.rustdir
    cleanup = args.cleanup
    verbose = args.verbose

    if runtimedir is None:
        runtime_dir = Path(os.getenv("LG_RT_DIR")).parent.absolute()
    else:
        runtime_dir = Path(runtimedir).parent.absolute()
   
    if outdir is None:
        outdir = os.getcwd()

    if rust:
        if rustdir is None:
            legion_prof_exe = "legion_prof"
        else:
            legion_prof_exe = rustdir
    else:
        legion_prof_exe = str(runtime_dir) + "/tools/legion_prof.py"

    # folders
    if rust:
        legion_prof_filter_input_folder = os.path.join(outdir, "legion_prof_filter_input_rs")
        legion_prof_no_filter_input_folder = os.path.join(outdir,"legion_prof_no_filter_input_rs")
    else:
        legion_prof_filter_input_folder = os.path.join(outdir, "legion_prof_filter_input")
        legion_prof_no_filter_input_folder = os.path.join(outdir,"legion_prof_no_filter_input")

    # parse logs with input filter
    filter_input_cmd = [legion_prof_exe, "--nodes", nodes, "--output", legion_prof_filter_input_folder]
    filter_input_cmd[5:5] = filenames
    if verbose:
        print(filter_input_cmd)
    subprocess.check_call(filter_input_cmd)

    # parse logs without input filter
    no_filter_input_cmd = [legion_prof_exe, "--nodes", nodes, "--output", legion_prof_no_filter_input_folder, "--no-filter-input"]
    no_filter_input_cmd[5:5] = filenames
    if verbose:
        print(no_filter_input_cmd)
    subprocess.check_call(no_filter_input_cmd)

    # check number of files under the tsv folder
    filter_input_tsv_folder = Path(legion_prof_filter_input_folder + "/tsv").absolute()
    filter_input_tsv_files = [name for name in os.listdir(filter_input_tsv_folder) if os.path.isfile(os.path.join(filter_input_tsv_folder, name))]

    no_filter_input_tsv_folder = Path(legion_prof_no_filter_input_folder + "/tsv").absolute()
    no_filter_input_tsv_files = [name for name in os.listdir(no_filter_input_tsv_folder) if os.path.isfile(os.path.join(no_filter_input_tsv_folder, name))]

    if len(filter_input_tsv_files) != len(no_filter_input_tsv_files):
        print(filter_input_tsv_files, len(filter_input_tsv_files))
        print(no_filter_input_tsv_files, len(no_filter_input_tsv_files))
        assert 0

    for filename in filter_input_tsv_files:
        if verbose:
            print("checking:", filename)
        if filename not in no_filter_input_tsv_files:
            print(filename, " is not exsited in ", legion_prof_no_filter_input_folder)
            assert 0
        
        filter_input_filename = str(filter_input_tsv_folder) + "/" + filename
        num_lines_filter_input = sum(1 for line in open(filter_input_filename))
        no_filter_input_filename = str(no_filter_input_tsv_folder) + "/" + filename
        num_lines_no_filter_input = sum(1 for line in open(no_filter_input_filename))
        if num_lines_filter_input != num_lines_no_filter_input:
            print(num_lines_filter_input, num_lines_no_filter_input)
            assert 0

    if cleanup:
        shutil.rmtree(legion_prof_filter_input_folder)
        shutil.rmtree(legion_prof_no_filter_input_folder)
        #shutil.rmtree(outdir)
