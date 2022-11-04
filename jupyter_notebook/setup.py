import argparse
from distutils.core import setup
from distutils.command.build_py import build_py
import sys

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
        build_py.run(self)

setup(name='legion_jupyter',
      version='0.1',
      zip_safe= False,
      py_modules=[
          'install_jupyter',
          'legion_kernel_nocr',
      ],
      cmdclass={'build_py': my_build_py})