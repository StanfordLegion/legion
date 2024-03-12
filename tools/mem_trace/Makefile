# Copyright 2024 Stanford University, NVIDIA Corporation
# Copyright 2024 Los Alamos National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

all: mem_trace.so mem_trace_csv

mem_trace.so: mem_trace.cc faults.cc mem_trace.h faults.h
	g++ -shared -fPIC -o mem_trace.so mem_trace.cc faults.cc -lpthread -ldl -rdynamic -fpermissive -O3

mem_trace_csv: mem_trace_csv.cc mem_trace.h
	g++ -o mem_trace_csv -O3 mem_trace_csv.cc
