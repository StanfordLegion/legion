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

def run_prof(verbose, py_exe_path, legion_root, prof_logs):
    result_dir = 'legion_prof_py'
    is_existed = os.path.exists(result_dir)
    if is_existed:
        print("remove:", result_dir)
        shutil.rmtree(result_dir)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cmd = [
        py_exe_path,
        os.path.join(legion_root, 'tools', 'legion_prof.py'),
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

def run_prof_rs(verbose, legion_prof_rs, prof_logs):
    result_dir = 'legion_prof_rs'
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

legion_path = os.getenv('LEGION_DIR')
if legion_path is None:
    test_path = os.path.dirname(os.path.realpath(__file__))
    test_path = Path(test_path)
    legion_path = dir_path.parent.parent.absolute()
else:
    test_path = os.path.join(legion_path, 'examples', 'provenance')
tmp_path = os.getenv('TMP_DIR')
if legion_path is None:
    tmp_path = ''
legion_prof_rs = os.path.join(tmp_path, 'bin', 'legion_prof')

prof_logs = glob.glob(os.path.join(test_path, 'prof_*.gz'))
run_prof(True, sys.executable, legion_path, prof_logs)
run_prof_rs(True, legion_prof_rs, prof_logs)
compare_prof_results(True, sys.executable, ['legion_prof_py', 'legion_prof_rs'])
