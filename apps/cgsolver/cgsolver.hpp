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

#ifndef cgsolver_hpp
#define cgsolver_hpp

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "cgmapper.hpp"
#include "ell_sparsematrix.hpp"
#include "legionvector.hpp"

template<typename T>
class CGSolver{

	private:
	int niter;
	T L2normr;
	
	public:
	bool Solve(const SpMatrix &A,
		   const Array<T> &b,
		   Array<T> &x,
		   int nitermax,
		   T threshold,
		   Context ctx,
		   HighLevelRuntime *runtime);

	int GetNumberIterations(void) { return niter;}
	T GetL2Norm(void) { return L2normr;}
};

template<typename T>
bool CGSolver<T>::Solve(const SpMatrix &A,
                   const Array<T> &b,
                   Array<T> &x,
                   int nitermax,
                   T threshold,
		   Context ctx, 
		   HighLevelRuntime *runtime)
{
        bool converged = false;
	assert(A.nrows == b.size);
	assert(b.size == x.size);
		   
	if(nitermax == -1) nitermax = A.nrows;

	Array<T> r_old(x.size, x.nparts, ctx, runtime);
	Array<T> p(x.size, x.nparts, ctx, runtime);
	Array<T> A_p(x.size, x.nparts, ctx, runtime);

        Predicate loop_pred = Predicate::TRUE_PRED;

	// Ap = A * x	
	spmv(A, x, A_p, loop_pred, ctx, runtime);
	std::cout<<"Ax = A * x is done."<<std::endl;

	// r_old = b - Ap
	subtract(b, A_p, r_old, T(1.0), ctx, runtime);
	std::cout<<"r = b - Ax is done."<<std::endl;

	// Initial norm
	const T L2normr0 = L2norm(r_old, ctx, runtime);
	std::cout<<"L2normr0 is done."<<std::endl;
	L2normr = L2normr0;

	// p = r_old
	copy(r_old, p, ctx, runtime);
	std::cout<<"copy is done."<<std::endl;

	niter = 0;
	//std::cout<<"Iteration"<<"    "<<"L2norm"<<std::endl;
	//std::cout<<niter<<"            "<<std::setprecision(16)<<L2normr<<std::endl;

        Future r2_old, pAp, alpha, r2_new, beta; 
#ifdef PREDICATED_EXECUTION
        std::deque<Future>  pending_norms;
        const int max_norm_depth = runtime->get_tunable_value(ctx, PREDICATED_TUNABLE);
#endif

	std::cout<<"Iterating..."<<std::endl;

	while(niter < nitermax){
		
		std::cout<<niter<<"            "<<L2normr<<std::endl;
		niter++;

		// Ap = A * p
		spmv(A, p, A_p, loop_pred, ctx, runtime);

		// r2 = r' * r
		r2_old = dot(r_old, r_old, loop_pred, r2_old, ctx, runtime);

		// pAp = p' * A * p
		pAp = dot(p, A_p, loop_pred, pAp, ctx, runtime);	

		// alpha = r2 / pAp
		alpha = compute_scalar<T>(r2_old, pAp, loop_pred, alpha, ctx, runtime);	
	
		// x = x + alpha * p
		add_inplace(x, p, alpha, loop_pred, ctx, runtime);
	
		// r_old = r_old - alpha * A_p
                subtract_inplace(r_old, A_p, alpha, loop_pred, ctx, runtime);

		r2_new = dot(r_old, r_old, loop_pred, r2_new, ctx, runtime);

		beta = compute_scalar<T>(r2_new, r2_old, loop_pred, beta, ctx, runtime);
	
		// p = r_old + beta*p
                axpy_inplace(r_old, p, beta, loop_pred, ctx, runtime);

#ifdef PREDICATED_EXECUTION
                Future norm = dot(r_old, r_old, loop_pred, 
                    pending_norms.empty() ? Future() : pending_norms.back(), ctx ,runtime);
                loop_pred = test_convergence(norm, L2normr0, threshold, 
                    loop_pred, ctx, runtime);
                pending_norms.push_back(norm);
                if (pending_norms.size() == max_norm_depth) {
                  // Pop the next future off the stack and wait for it
                  norm = pending_norms.front();
                  pending_norms.pop_front();
                  L2normr = sqrt(norm.get_result<double>());
                  converged = ((L2normr/L2normr0) < threshold);
                  if (converged) {
                    std::cout<<"Converged! :)"<<std::endl;
                    break;
                  }
                }
#else
		L2normr = L2norm(r_old, ctx, runtime);


		if(L2normr/ L2normr0 < threshold){
                  converged = true;
		  std::cout<<"Converged! :)"<<std::endl;
		  std::cout<<"Iteration"<<"    "<<"L2norm"<<std::endl;
		  std::cout<<niter<<"            "<<L2normr<<std::endl;		  
                  break;
                }
#endif
	}

	// destroy the objects
        r_old.DestroyArray(ctx, runtime);
        p.DestroyArray(ctx, runtime);
        A_p.DestroyArray(ctx, runtime);
	
        return converged;
} 	

#endif
