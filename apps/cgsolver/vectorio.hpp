/* Copyright 2016 Stanford University
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

#ifndef vectorio_hpp
#define vectorio_hpp

#include <cstdio>
#include <string>
#include <vector>

#include "mmio.h"

/* Read vector using Matrix Market file format */

template <typename T>
void ReadMatrixMarketVector(std::string filename, T* &rhs)
{
  /* Open the file */

  FILE *f = fopen(filename.c_str(), "r");
  if (f == NULL)
  {
    std::cout<<"failed to open file "<<filename<<std::endl;
  }

  /* Read the banner */

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0)
  {
    std::cout<<"failed to read banner in file "<<filename<<std::endl;
  }

  /* Determine the size */

  int status, nrows, ncols;

  if ((status = mm_read_mtx_array_size(f, &nrows, &ncols)) != 0)
  {
    std::cout<<"failed to read number of rows and columns in file "<<filename<<std::endl;
  }

  if (ncols != 1)
  {
    fclose(f);
    std::cout<<"number of columns is not 1 in file "<<filename<<std::endl;
  }

  /* Read values into vector */

  rhs = new T[nrows];

  int nval;
  double val;
  for (unsigned int n = 0; n < (unsigned int)nrows; n++)
  {
    nval = fscanf(f, "%lg\n", &val);
    if (nval < 1)
    {
      std::cout<<"failed to read dense line from file "<<filename<<std::endl;
    }
    rhs[n] = (T)val;
  }

  /* Close file and return */
  fclose(f);
  return;
}


/* Write vector using Matrix Market file format */

template <typename T>
void WriteMatrixMarketVector(std::vector<T> v, std::string filename)
{
  /* Open the file */

  FILE *f = fopen(filename.c_str(), "w");
  if (f == NULL)
  {
    std::cout<<"failed to open file "<<filename<<std::endl;
  }

  /* Setup matcode, write banner and size */

  MM_typecode matcode;

  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(f, matcode);
  mm_write_mtx_array_size(f, v.size(), 1);

  for(unsigned int n = 0; n < v.size(); n++)
    fprintf(f, "%e\n", v[n]);

  /* Close file and return */

  fclose(f);
  return;
}

#endif /* vectorio_hpp */
