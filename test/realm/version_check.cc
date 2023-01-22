// Copyright 2023 Stanford University, NVIDIA Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "realm.h"

#include <stdio.h>
#include <string.h>

int main(int argc, const char *argv[])
{
  // we can use the REALM_VERSION to get the version string for the Realm
  //  header's we're using
  const char *header_version = REALM_VERSION;

  // and we can ask the runtime what version the library itself is
  const char *library_version = Realm::Runtime::get_library_version();

  printf("Realm version check: header='%s'\n                    library='%s'\n",
         header_version, library_version);

  if(strcmp(header_version, library_version)) {
    printf("VERSION MISMATCH!\n");
    return 1;
  }

  return 0;
}
