use std::io;
use std::path::Path;

use legion_prof_viewer::{
    duckdb_data::DataSourceDuckDBWriter, parallel_data::ParallelDeferredDataSource,
};

use crate::backend::data_source::StateDataSource;
use crate::state::State;

pub fn write(state: State, path: impl AsRef<Path>, force: bool) -> io::Result<()> {
    let duckdb = DataSourceDuckDBWriter::new(
        ParallelDeferredDataSource::new(StateDataSource::new(state)),
        path,
        force,
    );
    duckdb.write()
}
