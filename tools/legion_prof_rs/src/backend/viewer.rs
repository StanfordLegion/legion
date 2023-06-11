use legion_prof_viewer::{app, parallel_data::ParallelDeferredDataSource};

use crate::backend::data_source::StateDataSource;
use crate::state::State;

pub fn start(state: State) {
    app::start(vec![Box::new(ParallelDeferredDataSource::new(
        StateDataSource::new(state),
    ))]);
}
