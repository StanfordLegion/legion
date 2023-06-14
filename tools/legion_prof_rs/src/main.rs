use std::ffi::OsString;
use std::io;

use clap::Parser;

use rayon::prelude::*;

#[cfg(feature = "client")]
use legion_prof_viewer::{
    app, deferred_data::DeferredDataSource, http::client::HTTPClientDataSource,
};
#[cfg(feature = "client")]
use url::Url;

#[cfg(feature = "server")]
use legion_prof::backend::server;
#[cfg(feature = "viewer")]
use legion_prof::backend::viewer;
use legion_prof::backend::{analyze, trace_viewer, visualize};
use legion_prof::serialize::deserialize;
use legion_prof::spy;
use legion_prof::state::{Config, NodeID, Records, SpyState, State, Timestamp};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(required = true, help = "input Legion Prof log filenames")]
    filenames: Vec<OsString>,

    #[arg(
        short,
        long,
        default_value = "legion_prof",
        help = "output directory pathname"
    )]
    output: OsString,

    #[arg(
        long = "start-trim",
        help = "start time in microseconds to trim the profile"
    )]
    start_trim: Option<u64>,

    #[arg(
        long = "stop-trim",
        help = "stop time in microseconds to trim the profile"
    )]
    stop_trim: Option<u64>,

    #[arg(
        long = "message-threshold",
        default_value_t = 1000.0,
        help = "threshold for warning about message latencies in microseconds"
    )]
    message_threshold: f64,

    #[arg(
        long = "message-percentage",
        default_value_t = 5.0,
        help = "perentage of messages that must be over the threshold to trigger a warning"
    )]
    message_percentage: f64,

    #[arg(long, help = "a list of nodes that will be visualized")]
    nodes: Option<String>,

    #[arg(
        long = "no-filter-input",
        hide = true,
        help = "parse all log files, even when a subset of nodes are being shown (uses more memory)"
    )]
    no_filter_input: bool,

    #[arg(short, long, help = "overwrite output directory if it exists")]
    force: bool,

    #[arg(long, help = "connect viewer to the specified HTTP profile server")]
    attach: bool,

    #[arg(long, help = "start profile HTTP server")]
    serve: bool,

    #[arg(long, help = "start interactive profile viewer")]
    view: bool,

    #[arg(short, long, help = "print statistics")]
    statistics: bool,

    #[arg(short, long, help = "emit JSON for Google Trace Viewer")]
    trace: bool,

    #[arg(short, long, help = "print verbose profiling information")]
    verbose: bool,

    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "host to bind for HTTP server"
    )]
    host: String,

    #[arg(long, default_value_t = 8080, help = "port to bind for HTTP server")]
    port: u16,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let start_trim = cli.start_trim.map(Timestamp::from_us);
    let stop_trim = cli.stop_trim.map(Timestamp::from_us);
    let message_threshold = cli.message_threshold;
    let message_percentage = cli.message_percentage;

    let mut node_list: Vec<NodeID> = Vec::new();
    let mut filter_input = false;
    if let Some(nodes_str) = cli.nodes {
        node_list = nodes_str
            .split(",")
            .flat_map(|x| {
                let splits: Vec<_> = x.splitn(2, "-").map(|x| x.parse::<u64>().unwrap()).collect();
                if splits.len() == 2 {
                    (splits[0]..=splits[1]).into_iter().map(NodeID)
                } else {
                    (splits[0]..=splits[0]).into_iter().map(NodeID)
                }
            })
            .collect();
        filter_input = !cli.no_filter_input;
    }

    #[cfg(not(feature = "client"))]
    if cli.attach {
        panic!(
            "Legion Prof was not build with the \"client\" feature. \
                Rebuild with --features=client to enable."
        );
    }

    #[cfg(not(feature = "server"))]
    if cli.serve {
        panic!(
            "Legion Prof was not build with the \"server\" feature. \
                Rebuild with --features=server to enable."
        );
    }

    #[cfg(not(feature = "viewer"))]
    if cli.view {
        panic!(
            "Legion Prof was not build with the \"viewer\" feature. \
                Rebuild with --features=viewer to enable."
        );
    }

    if [cli.attach, cli.serve, cli.view, cli.statistics, cli.trace]
        .iter()
        .filter(|x| **x)
        .count()
        > 1
    {
        panic!(
            "Legion Prof takes at most one of --attach, --serve, --view, --statistics, or --trace"
        );
    }

    if cli.attach {
        #[cfg(feature = "client")]
        {
            let urls: Vec<_> = cli
                .filenames
                .into_iter()
                .map(|x| {
                    Url::parse(x.to_str().expect("URL contains invalid UTF-8"))
                        .expect("invalid profile URL")
                })
                .collect();
            let data_sources: Vec<_> = urls
                .into_iter()
                .map(|url| {
                    let data_source: Box<dyn DeferredDataSource> =
                        Box::new(HTTPClientDataSource::new(url));
                    data_source
                })
                .collect();
            app::start(data_sources);
        }
        return Ok(());
    }

    let records: Result<Vec<Records>, _> = cli
        .filenames
        .par_iter()
        .map(|filename| {
            println!("Reading log file {:?}...", filename);
            deserialize(filename, &node_list, filter_input).map_or_else(
                |_| spy::serialize::deserialize(filename).map(Records::Spy),
                |r| Ok(Records::Prof(r)),
            )
        })
        .collect();
    let mut state = State::default();
    state.visible_nodes = node_list;
    let mut spy_state = SpyState::default();
    if filter_input {
        println!("Filtering profiles to nodes: {:?}", state.visible_nodes);
    }
    for record in records? {
        match record {
            Records::Prof(r) => {
                println!("Matched {} objects", r.len());
                state.process_records(&r);
            }
            Records::Spy(r) => {
                println!("Matched {} objects", r.len());
                spy_state.process_spy_records(&r);
            }
        }
    }

    if !state.has_prof_data {
        println!("Nothing to do");
        return Ok(());
    }

    let mut have_alllogs = true;
    // if number of files
    if state.num_nodes > cli.filenames.len().try_into().unwrap() {
        println!("Warning: This run involved {:?} nodes, but only {:?} log files were provided. If --verbose is enabled, subsequent warnings may not indicate a true error.", state.num_nodes, cli.filenames.len());
        have_alllogs = false;
    }

    // check if subnodes is enabled and filter input is true
    if state.visible_nodes.len() < state.num_nodes.try_into().unwrap() && filter_input {
        println!("Warning: This run involved {:?} nodes, but only {:?} log files were used. If --verbose ie enabled, subsequent warnings may not indicate a true error.", state.num_nodes, state.visible_nodes.len());
        have_alllogs = false;
    }

    Config::set_config(filter_input, cli.verbose, have_alllogs);

    spy_state.postprocess_spy_records(&state);

    state.trim_time_range(start_trim, stop_trim);
    println!("Sorting time ranges");
    state.sort_time_range();
    state.check_message_latencies(message_threshold, message_percentage);
    state.filter_output();
    if cli.statistics {
        analyze::print_statistics(&state);
    } else if cli.trace {
        trace_viewer::emit_trace(&state, cli.output, cli.force)?;
    } else if cli.serve {
        #[cfg(feature = "server")]
        {
            state.assign_colors();
            server::start(state, &cli.host, cli.port);
        }
    } else if cli.view {
        #[cfg(feature = "viewer")]
        {
            state.assign_colors();
            viewer::start(state);
        }
    } else {
        state.assign_colors();
        visualize::emit_interactive_visualization(&state, &spy_state, cli.output, cli.force)?;
    }

    Ok(())
}
