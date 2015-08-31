/* Copyright 2015 Stanford University
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

#ifndef sparse_hpp
#define sparse_hpp

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "mmio.h"
#include <time.h>

/* Bubble sorting function for sparse matrix format conversion,
   used to sort the entries in one row of the matrix. */
template <typename T>
void BubbleSortRow(int max_nnonzeros, int nrows, int *ell_nnonzero, int *ell_col_ind, T *ell_val)
{
  for(int r=0; r<nrows; r++) {
    for(int i = ell_nnonzero[r]-1; i>0; i--){
      for(int j=0; j<i; j++){
        if(ell_col_ind[r*max_nnonzeros+j] > ell_col_ind[r*max_nnonzeros+j+1])
        {
          /* Swap the value and the column index */
          T dt = ell_val[r*max_nnonzeros+j]; 
          ell_val[r*max_nnonzeros+j] = ell_val[r*max_nnonzeros+j+1];
          ell_val[r*max_nnonzeros+j+1] = dt;

          int it = ell_col_ind[r*max_nnonzeros+j]; 
          ell_col_ind[r*max_nnonzeros+j] = ell_col_ind[r*max_nnonzeros+j+1]; 
          ell_col_ind[r*max_nnonzeros+j+1] = it;
        }
      }
    }
  }
  return;
}


/* Template class for a sparse matrix in ELL format */
template <typename T>
class ELLMatrix
{
  private:
    /* Memory for COO storage of a matrix (using int for compatibility with Matrix Market IO). */
    T *val;
    int *col_ind;
    int *row_ptr;
    
    /* Size of matrix and number of nonzero entries */
    int nrows;
    int ncols;
    int nnonzeros;
    int max_nnonzeros;

  public:
    int GetNumberRows (void) const;
    int GetNumberCols (void) const;
    int GetNumberNonZeros (void) const;
    const T * GetData (void) const;
    const int * GetColIdx (void) const;
    const int * GetRowPtr (void) const;
    int ReadMatrixMarketFile(std::string filename, int* &ell_nnonzero, int* &ell_col_ind, T* &ell_val);
    void COO2ELL(int* &ell_nnonzero, int* &ell_col_ind, T* &ell_val);
    ~ELLMatrix();

};

/* In place conversion of square matrix from COO to ELL format */

template <typename T>
void ELLMatrix<T>::COO2ELL(int* &ell_nnonzero, int* &ell_col_ind, T* &ell_val)
{
  ell_nnonzero = new int[nrows]();
  int* counter = new int[nrows]();

  /* Determine row lengths */
  for (int i = 0; i < nnonzeros; i++) ell_nnonzero[row_ptr[i]]++;
  
  // find max_nnonzero
  max_nnonzeros = ell_nnonzero[0];
  for(int i=1; i<nrows; i++){
  
  	if(max_nnonzeros < ell_nnonzero[i])
  		max_nnonzeros = ell_nnonzero[i];
  }
  
  // allocate memory
  ell_val = new T[max_nnonzeros*nrows]();
  ell_col_ind = new int[max_nnonzeros*nrows]();
  
  for (int init = 0; init < nnonzeros; init++)
  {
    int i = row_ptr[init]; 
    ell_val[i*max_nnonzeros+counter[i]] = val[init];
    ell_col_ind[i*max_nnonzeros+counter[i]] = col_ind[init];
    counter[i]++;
  }

   // sort based on column
   double t_start = Realm::Clock::current_time();

   BubbleSortRow(max_nnonzeros, nrows, ell_nnonzero, ell_col_ind, ell_val);

   double t_end = Realm::Clock::current_time();

   double time = (t_end - t_start) * 1e3;;
   
   std::cout<<"Sorting time="<<std::setprecision(10)<<time<<" ms"<<std::endl;

   delete[] counter;
}


template <typename T>
int ELLMatrix<T>::GetNumberRows(void) const
{
  return nrows;
}


template <typename T>
int ELLMatrix<T>::GetNumberCols(void) const
{
  return ncols;
}


template <typename T>
int ELLMatrix<T>::GetNumberNonZeros(void) const
{
  return nnonzeros;
}


template <typename T>
const T * ELLMatrix<T>::GetData(void) const
{
  return val;
}


template <typename T>
const int * ELLMatrix<T>::GetColIdx(void) const
{
  return col_ind;
}


template <typename T>
const int * ELLMatrix<T>::GetRowPtr(void) const
{
  return row_ptr;
}


template <typename T>
int ELLMatrix<T>::ReadMatrixMarketFile(std::string filename, int* &ell_nnonzero, int* &ell_col_ind, T* &ell_val)
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

  int status;
  nrows = 0;
  ncols = 0;
  nnonzeros = 0;

  if (mm_is_sparse(matcode))
  {
    if ((status = mm_read_mtx_crd_size(f, &nrows, &ncols, &nnonzeros)) != 0)
    {
      std::cout<<"failed to read coordinate size in file "<<filename<<std::endl;
    }
  }
  else
  {
    std::cout<<"matrix in file "<<filename<<" is not sparse"<<std::endl;
  }

  /* Allocate memory for data in COO format */
	
  if (mm_is_symmetric(matcode))
  {
    val = new T[2*nnonzeros];
    col_ind = new int[2*nnonzeros];
    row_ptr = new int[2*nnonzeros];
  }
  else
  {
    val = new T[nnonzeros];
    col_ind = new int[nnonzeros];
    row_ptr = new int[nnonzeros];
  }

  /* Read values from file */

  int i, j, nn = 0, ndiag = 0;
  double value;
  // Note: this part is changed, because it didn't work for matrices from
  // Darve's group.
  
  while(fscanf(f, "%d %d %lg", &i, &j, &value) != EOF){
    // Keep track of non-zero entries on the diagonal
    if (i == j) ndiag++;
    
    // Store the value, Matrix Market uses 1 based indexing so subtract 1
    val[nn] = value;
    col_ind[nn] = j - 1;
    row_ptr[nn] = i - 1;
    nn++;
    
    if (mm_is_symmetric(matcode) and i != j)
    {
      val[nn] = value;
      col_ind[nn] = i - 1;
      row_ptr[nn] = j - 1;
      nn++;
    }
    
  }
  
  /* Recompute the number of nonzeros for symmetric case */
  
  if (mm_is_symmetric(matcode))
  {
    nnonzeros = 2*nnonzeros - ndiag;
  }

  /* Close file */

  fclose(f);

  /* Convert COO to ELL format */

  if (nrows != ncols)
  {
    std::cout<<"no support for converting non-square COO matrix to CSR format"<<std::endl;
  }

  COO2ELL(ell_nnonzero, ell_col_ind, ell_val);

  if (val)  {
  	delete[] val;
  	val = NULL;
  }
  if (col_ind) {
  	delete[] col_ind;
  	col_ind = NULL;
  }  
  if (row_ptr) {
  	delete[] row_ptr;
  	row_ptr = NULL;
  }
  
  return(max_nnonzeros);

}

template <typename T>
ELLMatrix<T>::~ELLMatrix()
{
  /* Free heap memory */
  if (val)     delete[] val;
  if (col_ind) delete[] col_ind;
  if (row_ptr) delete[] row_ptr;
}

#endif /* sparse_hpp */
