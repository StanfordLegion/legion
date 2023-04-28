use legion_prof_viewer::app;

use crate::backend::data_source::StateDataSource;
use crate::state::State;

pub fn start(state: State) {
    app::start(Box::new(StateDataSource::new(state)), None);
}
