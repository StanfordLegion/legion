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

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <string>
#include "legion.h"

#include "params.hpp"
#include "legionvector.hpp"
#include "ell_sparsematrix.hpp"
#include "cgoperators.hpp"
#include "cgsolver.hpp"
#include "cgmapper.hpp"

using namespace LegionRuntime::HighLevel;

enum TaskIDs {
   TOP_LEVEL_TASK_ID = 0,
};

void top_level_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime){

    int64_t nx = 15; 
    int64_t nparts = 1;
    int iter_max = -1;
    std::string matrix_file;
    std::string rhs_file;
    bool inputmat = false;
    bool inputrhs = false;
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    
        // Parse command line arguments
        for (int i = 1; i < command_args.argc; i++)
        {
          if (!strcmp(command_args.argv[i], "-n"))
          {
            nx = atoi(command_args.argv[++i]);
            assert(nx > 0);
            continue;
          }
          if (!strcmp(command_args.argv[i], "-m"))
          {
            matrix_file = command_args.argv[++i];
            inputmat = true;
            continue;
          }
          if (!strcmp(command_args.argv[i], "-b"))
          {
            rhs_file = command_args.argv[++i];
            inputrhs = true;
            continue;
          }
          if (!strcmp(command_args.argv[i], "-max"))
          {
            iter_max = atoi(command_args.argv[++i]);
            assert(iter_max >= 0);
            continue;
          }
        }
   	
	// get naprts from the custom mapper
	nparts = runtime->get_tunable_value(ctx, SUBREGION_TUNABLE, 0);
	
	Params<double> params;
	if(!inputmat && !inputrhs) {
	
		params.Init(nx);
   		params.GenerateVals();
   	}
	else {	
	
		if(inputmat){
			assert(!matrix_file.empty());
			params.InitMat(matrix_file);
		}
		
		if(inputrhs){
			assert(!rhs_file.empty());
			params.InitRhs(rhs_file);
		}		
   	}
		
	std::cout<<"Problem generation is done. Some properties are as follows:"<<std::endl;
	std::cout<<"*******************************************************"<<std::endl;

	// report the problem size and memory usage 
	std::cout<<"SPARSE MATRIX STORAGE FORMAT = ELL"<<std::endl;
	std::cout<<"MATRIX DIMENSIONS="<<params.nrows<<"x"<<params.nrows<<std::endl;
	std::cout<<"MEMORY SPENT ON  NONZERO VALUES = "<<params.max_nzeros * params.nrows * sizeof(double) / 1e6 <<
	" Mb"<<std::endl;
	std::cout<<"MEMORY SPENT ON COLUMN INDEX OF NONZERO VALUES  = "<<params.max_nzeros * params.nrows * 
	sizeof(int) / 1e6 <<" Mb"<<std::endl;
	std::cout<<"MEMORY SPENT ON X VECTOR = "<<params.nrows * sizeof(double) / 1e6 <<" Mb"<<std::endl;
	std::cout<<"MEMORY SPENT ON RHS VECTOR = "<<params.nrows * sizeof(double) / 1e6 <<" Mb"<<std::endl;
	std::cout<<"*******************************************************"<<std::endl;
	std::cout<<std::endl;
	
	// build sparse matrix
	std::cout<<"Make sparse matrix..."<<std::endl;
	SpMatrix A(params.nrows, nparts, params.nonzeros, params.max_nzeros, ctx, runtime);
	A.BuildMatrix(params.vals, params.col_ind, params.nzeros_per_row, ctx, runtime);

	// build unknown vector   
	std::cout<<"Make  unknown vector x..."<<std::endl;	
	Array<double> x(params.nrows, nparts, ctx, runtime);
	x.Initialize(ctx, runtime);

	// build rhs vector
	std::cout<<"Make rhs vector..."<<std::endl;
	Array<double> b(params.nrows, nparts, ctx, runtime);
	
	if(inputmat && !inputrhs) {	
		// fill rhs using random x vector
		Array<double> x_rand(params.nrows, nparts, ctx, runtime);
		x_rand.RandomInit(ctx, runtime);
		Predicate loop_pred = Predicate::TRUE_PRED;
		spmv(A, x_rand, b, loop_pred, ctx, runtime);
	}
	else {
	    // otherwise use the rhs array in params
		b.Initialize(params.rhs, ctx, runtime);	
	}
		
	std::cout<<"Launch the CG solver..."<<std::endl;	
	std::cout<<std::endl;

	
	// run CG solver
        double t_start = Realm::Clock::current_time();

	CGSolver<double> cgsolver;
	bool result = cgsolver.Solve(A, b, x, iter_max, 1e-4, ctx, runtime);

        double t_end = Realm::Clock::current_time();

	if(result) {

          double time = (t_end - t_start) * 1e3;

          std::cout<<"Elapsed time="<<std::setprecision(10)<<time<<" ms"<<std::endl;
	}
	else {
          std::cout<<"NO CONVERGENCE! :("<<std::endl;
	}
	std::cout<<std::endl;

	//print the solution
	//std::cout<<"SOLUTION:"<<std::endl;
	//x.PrintVals(ctx, runtime);
	//x.GiveNorm(params.exact, ctx, runtime);

	// destroy the objects
	x.DestroyArray(ctx, runtime);
	b.DestroyArray(ctx, runtime);
	A.DestroySpMatrix(ctx, runtime);
	
	return;
}

int main(int argc, char **argv){

	HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
   	
	HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
       	Processor::LOC_PROC, true/*single*/, false/*index*/,
        AUTO_GENERATE_ID, TaskConfigOptions(), "top_level_task");
       	
        // Register the callback function for creating custom mapper
  	HighLevelRuntime::set_registration_callback(mapper_registration);
	
	RegisterVectorTask<double>();

	RegisterOperatorTasks<double>();	
 
  return HighLevelRuntime::start(argc, argv);
}
