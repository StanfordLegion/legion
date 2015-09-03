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

#ifndef params_hpp
#define params_hpp

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "sparse.hpp"
#include "vectorio.hpp"

template<typename T>
class Params{

	public:
	int nx;
	int nrows;
	int nonzeros;
	int max_nzeros;
	T *vals;
	int *col_ind;
	int *nzeros_per_row;
	T *rhs;
	T *exact;
	
	Params(void){};
	void Init(int nx);
	void InitMat(std::string matrix_file);
	void InitRhs(std::string rhs_file);
	~Params(void);
	void GenerateVals(void);
};

template<typename T>
void Params<T>::Init(int nx) {
	
		this-> nx = nx;
		nrows = nx * nx;
		nonzeros = nrows * 5;
		max_nzeros = 5;
		
		vals = new T[nrows*max_nzeros];
		rhs = new T[nrows];
		col_ind = new int[nrows*max_nzeros];
		nzeros_per_row = new int[nrows];
		exact = new T[nrows];
}

template<typename T>
void Params<T>::InitMat(std::string matrix_file) {

	vals = NULL;
	col_ind = NULL;
	nzeros_per_row = NULL;
	exact = NULL;
	rhs = NULL;

	ELLMatrix<T> spmatrix;
	max_nzeros = spmatrix.ReadMatrixMarketFile(matrix_file, nzeros_per_row, col_ind, vals);
	
	nrows = spmatrix.GetNumberRows();
	nonzeros = spmatrix.GetNumberNonZeros();

	std::cout<<"In params>> Number of maximum nonzeros in some rows = "<<max_nzeros<<std::endl;
}

template<typename T>
void Params<T>::InitRhs(std::string rhs_file) {

	rhs = NULL;
	ReadMatrixMarketVector<T>(rhs_file, rhs);
	
	return;
}

template<typename T>
Params<T>::~Params(void) {

	//std::cout<<"deleting params..."<<std::endl;
	if(vals) delete [] vals;
	if(col_ind) delete [] col_ind;
	if(nzeros_per_row) delete [] nzeros_per_row;
	if(rhs) delete [] rhs;
	if(exact) delete [] exact;
}

template<typename T>
void Params<T>::GenerateVals(void) {

	// compute dx
	T h = 1.0 / (nx+1);
	int idx;
	T x = 0.0;
	T y = 0.0;
	
	// i=0
	idx = 0;
	x = h;
	y = h;
	
	nzeros_per_row[idx] = 3;

	rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(x)/(h*h) - exp(-2.0*y)/(h*h);
	exact[idx] = exp(x)*exp(-2.0*y);	
	vals[idx*max_nzeros] = -4.0 / (h*h);
	vals[idx*max_nzeros+1] = 1.0 / (h*h);
	vals[idx*max_nzeros+2] = 1.0 / (h*h);
	vals[idx*max_nzeros+3] = 0.0;
	vals[idx*max_nzeros+4] = 0.0;
	
	col_ind[idx*max_nzeros] =  idx;
	col_ind[idx*max_nzeros+1] = idx + 1;
	col_ind[idx*max_nzeros+2] = idx + nx;
	col_ind[idx*max_nzeros+3] = 0;
	col_ind[idx*max_nzeros+4] = 0;
	
	for(int j=1; j < nx-1; j++) {

		idx = j;
		x += h;
		
		nzeros_per_row[idx] = 4;
		
		rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(x)/(h*h); 
		exact[idx] = exp(x)*exp(-2.0*y);	
		vals[idx*max_nzeros] = 1.0 / (h*h);
		vals[idx*max_nzeros+1] = -4.0 / (h*h);
		vals[idx*max_nzeros+2] = 1.0 / (h*h);
		vals[idx*max_nzeros+3] = 1.0 / (h*h);
		vals[idx*max_nzeros+4] = 0.0;
	
		col_ind[idx*max_nzeros] =  idx - 1;
		col_ind[idx*max_nzeros+1] = idx;
		col_ind[idx*max_nzeros+2] = idx + 1;
		col_ind[idx*max_nzeros+3] = idx + nx;
		col_ind[idx*max_nzeros+4] = 0;
	}
	
	idx = nx-1;
	x += h;

	
	nzeros_per_row[idx] = 3;

	rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(x)/(h*h) - exp(1.0)*exp(-2.0*y)/(h*h);	
	exact[idx] = exp(x)*exp(-2.0*y);	
	vals[idx*max_nzeros] = 1.0 / (h*h);
	vals[idx*max_nzeros+1] = -4.0 / (h*h);
	vals[idx*max_nzeros+2] = 1.0 / (h*h);
	vals[idx*max_nzeros+3] = 0.0;
	vals[idx*max_nzeros+4] = 0.0;
	
	col_ind[idx*max_nzeros] =  idx - 1;
	col_ind[idx*max_nzeros+1] = idx;
	col_ind[idx*max_nzeros+2] = idx + nx;
	col_ind[idx*max_nzeros+3] = 0;
	col_ind[idx*max_nzeros+4] = 0;
	
	// i=1:nx-2
	for(int i=1; i < nx-1; i++) {
		
		idx = i*nx;
		y += h;
		x = h;
		
		nzeros_per_row[idx] = 4;
	
		rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(-2.0*y)/(h*h);
		exact[idx] = exp(x)*exp(-2.0*y);	
		vals[idx*max_nzeros] = 1.0 / (h*h);
		vals[idx*max_nzeros+1] = -4.0 / (h*h);
		vals[idx*max_nzeros+2] = 1.0 / (h*h);
		vals[idx*max_nzeros+3] = 1.0 / (h*h);
		vals[idx*max_nzeros+4] = 0.0;
	
		col_ind[idx*max_nzeros] =  idx - nx;
		col_ind[idx*max_nzeros+1] = idx;
		col_ind[idx*max_nzeros+2] = idx + 1;
		col_ind[idx*max_nzeros+3] = idx + nx;
		col_ind[idx*max_nzeros+4] = 0;
		
		for(int j=1; j < nx-1; j++) {
			idx = i*nx+j;
			x += h;
		
			nzeros_per_row[idx] = 5;
	
			rhs[idx] = 5.0 * exp(x) * exp(-2.0*y);
			exact[idx] = exp(x)*exp(-2.0*y);	
			vals[idx*max_nzeros] = 1.0 / (h*h);
			vals[idx*max_nzeros+1] = 1.0 / (h*h);
			vals[idx*max_nzeros+2] = -4.0 / (h*h);
			vals[idx*max_nzeros+3] = 1.0 / (h*h);
			vals[idx*max_nzeros+4] = 1.0 / (h*h);
	
			col_ind[idx*max_nzeros] =  idx - nx;
			col_ind[idx*max_nzeros+1] = idx - 1;
			col_ind[idx*max_nzeros+2] = idx;
			col_ind[idx*max_nzeros+3] = idx + 1;
			col_ind[idx*max_nzeros+4] = idx + nx;
		}
		
		idx = (i+1)*nx - 1;
		x += h;

		nzeros_per_row[idx] = 4;
		
		rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(1.0)*exp(-2.0*y)/(h*h);	
		exact[idx] = exp(x)*exp(-2.0*y);	
		vals[idx*max_nzeros] = 1.0 / (h*h);
		vals[idx*max_nzeros+1] = 1.0 / (h*h);
		vals[idx*max_nzeros+2] = -4.0 / (h*h);
		vals[idx*max_nzeros+3] = 1.0 / (h*h);
		vals[idx*max_nzeros+4] = 0.0;
	
		col_ind[idx*max_nzeros] =  idx - nx;
		col_ind[idx*max_nzeros+1] = idx - 1;
		col_ind[idx*max_nzeros+2] = idx;
		col_ind[idx*max_nzeros+3] = idx + nx;
		col_ind[idx*max_nzeros+4] = 0;
		
	}
	
	// i=nx-1
	idx = (nx-1)*nx;
	y += h;
	x = h;
	
	nzeros_per_row[idx] = 3;

	rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(x)*exp(-2.0)/(h*h) - exp(-2.0*y)/(h*h);	
	exact[idx] = exp(x)*exp(-2.0*y);	
	vals[idx*max_nzeros] = 1.0 / (h*h);
	vals[idx*max_nzeros+1] = -4.0 / (h*h);
	vals[idx*max_nzeros+2] = 1.0 / (h*h);
	vals[idx*max_nzeros+3] = 0.0;
	vals[idx*max_nzeros+4] = 0.0;
	
	col_ind[idx*max_nzeros] =  idx - nx;
	col_ind[idx*max_nzeros+1] = idx;
	col_ind[idx*max_nzeros+2] = idx + 1;
	col_ind[idx*max_nzeros+3] = 0;
	col_ind[idx*max_nzeros+4] = 0;
	
	for(int j=1; j < nx-1; j++) {
		idx = (nx-1)*nx + j;
	    	x += h;
	
		nzeros_per_row[idx] = 4;
	
		rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(x)*exp(-2.0)/(h*h);	
		exact[idx] = exp(x) * exp(-2.0*y);	
		vals[idx*max_nzeros] = 1.0 / (h*h);
		vals[idx*max_nzeros+1] = 1.0 / (h*h);
		vals[idx*max_nzeros+2] = -4.0 / (h*h);
		vals[idx*max_nzeros+3] = 1.0 / (h*h);
		vals[idx*max_nzeros+4] = 0.0;
	
		col_ind[idx*max_nzeros] =  idx - nx;
		col_ind[idx*max_nzeros+1] = idx - 1;
		col_ind[idx*max_nzeros+2] = idx;
		col_ind[idx*max_nzeros+3] = idx + 1;
		col_ind[idx*max_nzeros+4] = 0;
	}
	
	idx = nx*nx -1;
	x += h;

	nzeros_per_row[idx] = 3;

	rhs[idx] = 5.0 * exp(x) * exp(-2.0*y) - exp(x)*exp(-2.0)/(h*h) - exp(1.0)*exp(-2.0*y)/(h*h);	
	exact[idx] = exp(x) * exp(-2.0*y);	
	vals[idx*max_nzeros] = 1.0 / (h*h);
	vals[idx*max_nzeros+1] = 1.0 / (h*h);
	vals[idx*max_nzeros+2] = -4.0 / (h*h);
	vals[idx*max_nzeros+3] = 0.0;
	vals[idx*max_nzeros+4] = 0.0;
	
	col_ind[idx*max_nzeros] =  idx - nx;
	col_ind[idx*max_nzeros+1] = idx - 1;
	col_ind[idx*max_nzeros+2] = idx;
	col_ind[idx*max_nzeros+3] = 0;
	col_ind[idx*max_nzeros+4] = 0;
	
	return;
}
#endif
	
