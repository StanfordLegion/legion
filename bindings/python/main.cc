/* Copyright 2019 Stanford University
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

#include "legion.h"
#include "realm/python/python_module.h"
#include "realm/python/python_source.h"

#include <libgen.h>

using namespace Legion;

enum TaskIDs {
  MAIN_TASK_ID = 1,
};

int main(int argc, char **argv)
{
#ifdef BINDINGS_AUGMENT_PYTHONPATH
  // Add the binary directory to PYTHONPATH. This is needed for
  // in-place builds to find legion.py.

  // Do this before any threads are spawned.
  {
    char *bin_path = strdup(argv[0]);
    assert(bin_path != NULL);
    char *bin_dir = dirname(bin_path);

    char *previous_python_path = getenv("PYTHONPATH");
    if (previous_python_path != 0) {
      size_t bufsize = strlen(previous_python_path) + strlen(bin_dir) + 2;
      char *buffer = (char *)calloc(bufsize, sizeof(char));
      assert(buffer != 0);

      // assert(strlen(previous_python_path) + strlen(bin_dir) + 2 < bufsize);
      // Concatenate bin_dir to the end of PYTHONPATH.
      bufsize--;
      strncat(buffer, previous_python_path, bufsize);
      bufsize -= strlen(previous_python_path);
      strncat(buffer, ":", bufsize);
      bufsize -= strlen(":");
      strncat(buffer, bin_dir, bufsize);
      bufsize -= strlen(bin_dir);
      setenv("PYTHONPATH", buffer, true /*overwrite*/);
    } else {
      setenv("PYTHONPATH", bin_dir, true /*overwrite*/);
    }

    free(bin_path);
  }
#endif

#ifdef BINDINGS_DEFAULT_MODULE
#define str(x) #x
  Realm::Python::PythonModule::import_python_module(str(BINDINGS_DEFAULT_MODULE));
#undef str
#else
  if (argc < 2 || argv[1][0] == '-') {
    fprintf(stderr, "usage: %s [<module_name>|<script_path>]\n", argv[0]);
    exit(1);
  }
#endif

  const char *module_name = argv[1];
  if (strrchr(module_name, '.') == NULL) {
    Realm::Python::PythonModule::import_python_module(module_name);
  } else {
    Realm::Python::PythonModule::import_python_module("legion");
  }

  Runtime::set_top_level_task_id(MAIN_TASK_ID);

  return Runtime::start(argc, argv);
}
