/* Copyright 2023 Stanford University, NVIDIA Corporation
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

// g++ -o mem_trace_csv mem_trace_csv.cc

#include <iostream>
#include <fstream>
#include <cstring>
#include <string.h>
#include <map>
#include "mem_trace.h"

typedef void* KEY;

int main(int argc, char *argv[])
{
  if(argc < 4)
  {
    std::cout << "USAGE: ./mem_trace_csv -i <input> -o <output> (-v)" << std::endl;
    return 1;
  }

  char *in = NULL;
  char *out = NULL;
  int verbose = 0;

  for(int i = 0; i < argc; i++)
  {
    if(strcmp(argv[i], "-i") == 0)
    {
      in = argv[i+1];
    }
    else if(strcmp(argv[i], "-o") == 0)
    {
      out = argv[i+1];
    }
    else if(strcmp(argv[i], "-v") == 0)
    {
      verbose = 1;
    }
  }

  FILE *fin = fopen(in, "rb");
  FILE *fout = fopen(out, "w");

  fprintf(fout, "start,stop,type,size,addr,hash\n");

  std::map<KEY, Alloc> allocations;

  Alloc alloc;
  while(fread(&alloc, sizeof(alloc), 1, fin) == 1)
  {
    if(alloc.kind == AllocKind::FREE_KIND)
    {
      auto allocation = allocations.find(alloc.ptr);
      if(allocation == allocations.end())
      {
        if(verbose)
          std::cout << "DANGLING POINTER: " << alloc.ptr << std::endl;
      }
      else
      {
        Alloc aalloc = allocation->second;
        fprintf(fout, "%f,%f,%d,%zu,%p,%zu\n",
                aalloc.time, alloc.time, aalloc.kind, aalloc.size, aalloc.ptr, aalloc.hash);
        allocations.erase(allocation);
      }
    }
    else
    {
      allocations.insert(std::pair<KEY,Alloc>(alloc.ptr, alloc));
    }
  }

  for(auto it = allocations.begin(); it != allocations.end(); ++it)
  {
    KEY ptr = it->first;
    Alloc aalloc = it->second;
    fprintf(fout, "%f,nan,%d,%zu,%p,%zu\n", aalloc.time, aalloc.kind, aalloc.size, ptr, aalloc.hash);
  }

  fclose(fin);
  fclose(fout);

  return 0;
}
