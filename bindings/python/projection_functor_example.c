#include "legion.h"

void proj_functor(legion_runtime_t runtime,
                  legion_logical_partition_t parent,
                  legion_domain_point_t point,
                  legion_domain_t launch_domain)
{
  legion_projection_functor_logical_partition_print_arguments_2(parent);
}
