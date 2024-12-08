use std::ffi::OsString;
use std::io;

use legion_prof_viewer::{nvtxw::NVTXW, parallel_data::ParallelDeferredDataSource};

use crate::backend::data_source::StateDataSource;
use crate::state::{State, TimestampDelta};

pub fn write(
    state: State,
    backend: Option<OsString>,
    output: OsString,
    force: bool,
    merge: Option<OsString>,
    zero_time: TimestampDelta,
) -> io::Result<()> {
    let writer = NVTXW::new(
        ParallelDeferredDataSource::new(StateDataSource::new(state)),
        backend,
        output,
        force,
        merge,
        zero_time.0,
    );
    writer.write()
}
