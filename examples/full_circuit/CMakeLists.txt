add_executable(full_circuit circuit.cc
circuit_cpu.cc
circuit_mapper.cc)
target_link_libraries(full_circuit legion-rt)
