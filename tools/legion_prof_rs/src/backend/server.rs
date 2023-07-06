use legion_prof_viewer::http::server::DataSourceHTTPServer;

use crate::backend::data_source::StateDataSource;
use crate::state::State;

pub fn start(state: State, host: &str, port: u16) {
    let server = DataSourceHTTPServer::new(
        host.to_string(),
        port,
        Box::new(StateDataSource::new(state)),
    );
    server.run().expect("failed to start server");
}
