import argparse
from distutils.core import setup
from distutils.command.build_py import build_py
import sys

import legion_cffi_build

# Hack: I can't get initialize/finalize_option to work, so just parse
# the arguments here...
parser = argparse.ArgumentParser()
parser.add_argument('--cmake-build-dir', required=False)
args, unknown = parser.parse_known_args()
sys.argv[1:] = unknown

class my_build_py(build_py):
    def run(self):
        if not self.dry_run:
            self.mkpath(self.build_lib)
            legion_cffi_build.build(args.cmake_build_dir, self.build_lib)
        build_py.run(self)

setup(name='legion',
      version='0.1',
      py_modules=[
          'legion',
      ],
      cmdclass={'build_py': my_build_py})
