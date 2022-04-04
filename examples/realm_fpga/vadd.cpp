/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

// TRIPCOUNT indentifier
const unsigned int c_min = 4096;
const unsigned int c_max = 4 * 1024 * 1024;

/*
    Vector Addition Kernel Implementation
    Arguments:
        in1   (input)     --> Input Vector1
        in2   (input)     --> Input Vector2
        out_r   (output)    --> Output Vector
        size  (input)     --> Size of Vector in Integer
*/

extern "C" {
void vadd(const int* in1, // Read-Only Vector 1
          const int* in2, // Read-Only Vector 2
          int* out_r,     // Output Result
          int size                 // Size in integer
          ) {
// Unoptimized vector addition kernel to increase the kernel execution time
// Large execution time required to showcase parallel execution of multiple
// compute units in this example.
vadd1:
    for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_min max = c_max
        out_r[i] = in1[i] + in2[i];
    }
}
}
