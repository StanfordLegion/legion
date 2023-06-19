use std::io;
use std::path::Path;

use legion_prof_viewer::{
    archive_data::DataSourceArchiveWriter, parallel_data::ParallelDeferredDataSource,
};

use crate::backend::data_source::StateDataSource;
use crate::state::State;

pub fn write(
    state: State,
    levels: u32,
    branch_factor: u64,
    path: impl AsRef<Path>,
    force: bool,
) -> io::Result<()> {
    let archive = DataSourceArchiveWriter::new(
        ParallelDeferredDataSource::new(StateDataSource::new(state)),
        levels,
        branch_factor,
        path,
        force,
    );
    archive.write()
}
