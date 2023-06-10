use legion_prof_viewer::{app, deferred_data::DeferredDataSourceWrapper};

use crate::backend::data_source::StateDataSource;
use crate::state::State;

pub fn start(state: State) {
    app::start(vec![Box::new(DeferredDataSourceWrapper::new(
        StateDataSource::new(state),
    ))]);
}
