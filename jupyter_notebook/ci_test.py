import sys
major_version = sys.version_info[0]
minor_version = sys.version_info[1]
assert major_version >= 3, "Need Python3"

import pygion

import examples.domain as domain
import examples.domain_point as domain_point
import examples.domain_transform as domain_transform
import examples.future as future
import examples.hello as hello
import examples.index_launch as index_launch
import examples.ispace as ispace
import examples.layout as layout
import examples.method as method
import examples.must_epoch as must_epoch
import examples.partition as partition
import examples.partition_by_field as partition_by_field
import examples.partition_by_image as partition_by_image
import examples.partition_by_image_range as partition_by_image_range
import examples.partition_by_preimage as partition_by_preimage
import examples.partition_by_preimage_range as partition_by_preimage_range
import examples.partition_by_restriction as partition_by_restriction
import examples.reduction as reduction
import examples.region as region
import examples.region_fields as region_fields
import examples.return_region as return_region
import examples.single_launch as single_launch
import examples.struct as struct
import examples.trace as trace
import examples.tunable as tunable
import examples.types as types

if __name__ == '__main__':
    domain.main()
    pygion.execution_fence(block=True)

    domain_point.main()
    pygion.execution_fence(block=True)

    domain_transform.main()
    pygion.execution_fence(block=True)

    future.main()
    pygion.execution_fence(block=True)

    hello.main()
    pygion.execution_fence(block=True)

    index_launch.main()
    pygion.execution_fence(block=True)

    ispace.main()
    pygion.execution_fence(block=True)

    layout.main()
    pygion.execution_fence(block=True)

    method.main()
    pygion.execution_fence(block=True)

    must_epoch.main()
    pygion.execution_fence(block=True)

    partition.main()
    pygion.execution_fence(block=True)

    partition_by_field.main()
    pygion.execution_fence(block=True)

    partition_by_image.main()
    pygion.execution_fence(block=True)

    partition_by_image_range.main()
    pygion.execution_fence(block=True)

    partition_by_preimage.main()
    pygion.execution_fence(block=True)

    partition_by_preimage_range.main()
    pygion.execution_fence(block=True)

    partition_by_restriction.main()
    pygion.execution_fence(block=True)

    reduction.main()
    pygion.execution_fence(block=True)

    region.main()
    pygion.execution_fence(block=True)

    region_fields.main()
    pygion.execution_fence(block=True)

    return_region.main()
    pygion.execution_fence(block=True)

    single_launch.main()
    pygion.execution_fence(block=True)

    struct.main()
    pygion.execution_fence(block=True)

    trace.main()
    pygion.execution_fence(block=True)

    tunable.main()
    pygion.execution_fence(block=True)

    types.main()
    pygion.execution_fence(block=True)