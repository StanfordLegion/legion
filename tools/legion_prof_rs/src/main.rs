use std::io;

use rayon::prelude::*;

#[cfg(feature = "server")]
use legion_prof::backend::server;
#[cfg(feature = "viewer")]
use legion_prof::backend::viewer;
use legion_prof::backend::{analyze, trace_viewer, visualize};
use legion_prof::serialize::deserialize;
use legion_prof::spy;
use legion_prof::state::{Config, NodeID, Records, SpyState, State, Timestamp};

fn main() -> io::Result<()> {
    let matches = clap::App::new("Legion Prof")
        .about("Legion Prof: application profiler")
        .arg(
            clap::Arg::with_name("filenames")
                .help("input Legion Prof log filenames")
                .required(true)
                .multiple(true),
        )
        .arg(
            clap::Arg::with_name("output")
                .short("o")
                .long("output")
                .takes_value(true)
                .default_value("legion_prof")
                .help("output directory pathname"),
        )
        .arg(
            clap::Arg::with_name("start-trim")
                .long("start-trim")
                .takes_value(true)
                .help("start time in microseconds to trim the profile"),
        )
        .arg(
            clap::Arg::with_name("stop-trim")
                .long("stop-trim")
                .takes_value(true)
                .help("stop time in microseconds to trim the profile"),
        )
        .arg(
            clap::Arg::with_name("message-threshold")
                .long("message-threshold")
                .takes_value(true)
                .help("threshold for warning about message latencies in microseconds"),
        )
        .arg(
            clap::Arg::with_name("message-percentage")
                .long("message-percentage")
                .takes_value(true)
                .help("perentage of messages that must be over the threshold to trigger a warning"),
        )
        .arg(
            clap::Arg::with_name("nodes")
                .long("nodes")
                .takes_value(true)
                .help("a list of nodes that will be visualized"),
        )
        .arg(
            clap::Arg::with_name("no-filter-input")
                .long("no-filter-input")
                .hidden(true)
                .help("parse all log files, even when a subset of nodes are being shown (uses more memory)"),
        )
        .arg(
            clap::Arg::with_name("force")
                .short("f")
                .long("force")
                .help("overwrite output directory if it exists"),
        )
        .arg(
            clap::Arg::with_name("serve")
                .long("serve")
                .help("start profile HTTP server"),
        )
        .arg(
            clap::Arg::with_name("host")
                .long("host")
                .takes_value(true)
                .default_value("127.0.0.1")
                .help("host to bind for HTTP server"),
        )
        .arg(
            clap::Arg::with_name("port")
                .long("port")
                .takes_value(true)
                .default_value("8080")
                .help("port to bind for HTTP server"),
        )
        .arg(
            clap::Arg::with_name("statistics")
                .short("s")
                .long("statistics")
                .help("print statistics"),
        )
        .arg(
            clap::Arg::with_name("trace")
                .short("t")
                .long("trace-viewer")
                .help("emit JSON for Google Trace Viewer"),
        )
        .arg(
            clap::Arg::with_name("view")
                .long("view")
                .help("start interactive profile viewer"),
        )
        .arg(
            clap::Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("print verbose profiling information"),
        )
        .get_matches();

    let filenames = matches.values_of_os("filenames").unwrap();
    let output = matches.value_of_os("output").unwrap();
    let force = matches.is_present("force");
    let serve = matches.is_present("serve");
    let statistics = matches.is_present("stats");
    let trace = matches.is_present("trace");
    let view = matches.is_present("view");
    let verbose = matches.is_present("verbose");
    let start_trim = matches
        .value_of("start-trim")
        .map(|x| Timestamp::from_us(x.parse::<u64>().unwrap()));
    let stop_trim = matches
        .value_of("stop-trim")
        .map(|x| Timestamp::from_us(x.parse::<u64>().unwrap()));
    let message_threshold = matches
        .value_of("message-threshold")
        .map_or(1000.0, |x| x.parse::<f64>().unwrap());
    let message_percentage = matches
        .value_of("message-percentage")
        .map_or(5.0, |x| x.parse::<f64>().unwrap());
    let mut node_list: Vec<NodeID> = Vec::new();
    let mut filter_input = false;
    if let Some(nodes_str) = matches.value_of("nodes") {
        node_list = nodes_str
            .split(",")
            .map(|x| NodeID(x.parse::<u64>().unwrap()))
            .collect();
        filter_input = !matches.is_present("no-filter-input");
    }

    let host = matches.value_of("host").unwrap();
    let port = matches
        .value_of("port")
        .map(|x| x.parse::<u16>().unwrap())
        .unwrap();

    #[cfg(not(feature = "server"))]
    if serve {
        panic!(
            "Legion Prof was not build with the \"server\" feature. \
                Rebuild with --features=server to enable."
        );
    }

    #[cfg(not(feature = "viewer"))]
    if view {
        panic!(
            "Legion Prof was not build with the \"viewer\" feature. \
                Rebuild with --features=viewer to enable."
        );
    }

    let filenames: Vec<_> = filenames.collect();
    let records: Result<Vec<Records>, _> = filenames
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
    if state.num_nodes > filenames.len().try_into().unwrap() {
        println!("Warning: This run involved {:?} nodes, but only {:?} log files were provided. If --verbose is enabled, subsequent warnings may not indicate a true error.", state.num_nodes, filenames.len());
        have_alllogs = false;
    }

    // check if subnodes is enabled and filter input is true
    if state.visible_nodes.len() < state.num_nodes.try_into().unwrap() && filter_input {
        println!("Warning: This run involved {:?} nodes, but only {:?} log files were used. If --verbose ie enabled, subsequent warnings may not indicate a true error.", state.num_nodes, state.visible_nodes.len());
        have_alllogs = false;
    }

    Config::set_config(filter_input, verbose, have_alllogs);

    spy_state.postprocess_spy_records(&state);

    state.trim_time_range(start_trim, stop_trim);
    println!("Sorting time ranges");
    state.sort_time_range();
    state.check_message_latencies(message_threshold, message_percentage);
    state.filter_output();
    if statistics {
        analyze::print_statistics(&state);
    } else if trace {
        trace_viewer::emit_trace(&state, output, force)?;
    } else if serve {
        #[cfg(feature = "server")]
        {
            state.assign_colors();
            server::start(state, host, port);
        }
    } else if view {
        #[cfg(feature = "viewer")]
        {
            state.assign_colors();
            viewer::start(state);
        }
    } else {
        state.assign_colors();
        visualize::emit_interactive_visualization(&state, &spy_state, output, force)?;
    }

    Ok(())
}
