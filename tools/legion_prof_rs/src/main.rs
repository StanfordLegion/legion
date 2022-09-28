use std::io;

use rayon::prelude::*;

use legion_prof::analyze::print_statistics;
use legion_prof::serialize::deserialize;
use legion_prof::spy;
use legion_prof::state::{Records, State, Timestamp};
use legion_prof::trace_viewer::emit_trace;
use legion_prof::visualize::emit_interactive_visualization;

fn main() -> io::Result<()> {
    let matches = clap::App::new("Integrity Checker")
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
            clap::Arg::with_name("force")
                .short("f")
                .long("force")
                .help("overwrite output directory if it exists"),
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
        .get_matches();

    let filenames = matches.values_of_os("filenames").unwrap();
    let output = matches.value_of_os("output").unwrap();
    let force = matches.is_present("force");
    let statistics = matches.is_present("stats");
    let trace = matches.is_present("trace");
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

    let mut state = State::default();
    let filenames: Vec<_> = filenames.collect();
    let records: Result<Vec<Records>, _> = filenames
        .par_iter()
        .map(|filename| {
            println!("Reading log file {:?}...", filename);
            deserialize(filename).map_or_else(
                |_| spy::serialize::deserialize(filename).map(Records::Spy),
                |r| Ok(Records::Prof(r)),
            )
        })
        .collect();
    for record in records? {
        match record {
            Records::Prof(r) => {
                println!("Matched {} objects", r.len());
                state.process_records(&r);
            }
            Records::Spy(r) => {
                println!("Matched {} objects", r.len());
                state.process_spy_records(&r);
            }
        }
    }

    if !state.has_prof_data {
        println!("Nothing to do");
        return Ok(());
    }

    state.postprocess_spy_records();

    state.trim_time_range(start_trim, stop_trim);
    println!("Sorting time ranges");
    state.sort_time_range();
    state.check_message_latencies(message_threshold, message_percentage);
    if statistics {
        print_statistics(&state);
    } else if trace {
        emit_trace(&state, output, force)?;
    } else {
        state.assign_colors();
        emit_interactive_visualization(&state, output, force)?;
    }

    Ok(())
}
