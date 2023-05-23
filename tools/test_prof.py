#!/usr/bin/env python3

# Copyright 2023 Stanford University, NVIDIA
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

import sys, subprocess, os
import shutil
import glob
from pathlib import Path

def compare_prof_results(verbose, py_exe_path, profile_dirs):
    cmd = ['diff', '-r', '-u'] + profile_dirs
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        assert 0

def run_prof(verbose, py_exe_path, legion_path, legion_prof_result_folder, prof_logs):
    result_dir = os.path.join(legion_prof_result_folder, 'legion_prof_py')
    is_existed = os.path.exists(result_dir)
    if is_existed:
        print("remove:", result_dir)
        shutil.rmtree(result_dir)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cmd = [
        py_exe_path,
        os.path.join(legion_path, 'tools', 'legion_prof.py'),
        '-o', result_dir,
    ] + prof_logs
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        assert 0

def run_prof_rs(verbose, legion_prof_rs, legion_prof_result_folder, prof_logs):
    result_dir = os.path.join(legion_prof_result_folder, 'legion_prof_rs')
    is_existed = os.path.exists(result_dir)
    if is_existed:
        print("remove:", result_dir)
        shutil.rmtree(result_dir)
    cmd = [legion_prof_rs, '-o', result_dir,] + prof_logs
    if verbose: print('Running', ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT)
    output, _ = proc.communicate()
    retcode = proc.wait()
    if retcode != 0:
        assert 0
    return result_dir

def run_prof_test(legion_path, test_path, tmp_path):
    test_full_path = os.path.join(legion_path, test_path)
    legion_prof_rs = os.path.join(tmp_path, 'bin', 'legion_prof')
    prof_logs = glob.glob(os.path.join(test_path, 'prof_*.gz'))
    
    legion_prof_result_folder = os.path.join(os.getcwd(), 'legion_prof_test')
    # if this path is existed, remove it
    if os.path.exists(legion_prof_result_folder):
      shutil.rmtree(legion_prof_result_folder)
    os.makedirs(legion_prof_result_folder)
    run_prof(True, sys.executable, legion_path, legion_prof_result_folder, prof_logs)
    run_prof_rs(True, legion_prof_rs, legion_prof_result_folder, prof_logs)
    compare_prof_results(True, sys.executable, [os.path.join(legion_prof_result_folder, 'legion_prof_py'), os.path.join(legion_prof_result_folder, 'legion_prof_rs')])
    # remove the test folder
    shutil.rmtree(legion_prof_result_folder)
