use std::cmp::max;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::Path;

use serde::Serialize;
use serde_json;

use crate::state::{ProcEntry, State, Timestamp};

#[derive(Serialize, Copy, Clone)]
struct Event<'a> {
    name: &'a str,
    #[serde(rename = "cat")]
    category: &'a str,
    #[serde(rename = "ph")]
    phase: &'a str,
    #[serde(rename = "ts")]
    timestamp: f64,
    #[serde(rename = "dur")]
    duration: f64,
    pid: u64,
    tid: u64,
}

fn ts_to_us(ts: Timestamp) -> f64 {
    (ts.0 as f64) / 1e3
}

pub fn emit_trace<P: AsRef<Path>>(state: &mut State, path: P, force: bool) -> io::Result<()> {
    println!("Generating trace file {:?}", path.as_ref());

    let mut file = OpenOptions::new()
        .write(true)
        // If force is set, create and truncate.
        .create(true)
        .truncate(true)
        // If force is not set, create only if it does not exist.
        .create_new(!force)
        .open(path)?;
    write!(file, "[")?;

    let mut first = true;

    for proc in state.procs.values() {
        for point in &proc.time_points {
            if point.first {
                let (base, time_range, waiters) = proc.entry(point.entry);

                let name = match point.entry {
                    ProcEntry::Task(op_id) => {
                        let task = &proc.tasks.get(&op_id).unwrap();
                        let task_name = &state.task_kinds.get(&task.task_id).unwrap().name;
                        let variant_name = &state
                            .variants
                            .get(&(task.task_id, task.variant_id))
                            .unwrap()
                            .name;
                        match task_name {
                            Some(task_name) => format!("{} ({})", task_name, variant_name),
                            None => variant_name.clone(),
                        }
                    }
                    ProcEntry::MetaTask(op_id, variant_id, idx) => {
                        let task = &proc.meta_tasks.get(&(op_id, variant_id)).unwrap()[idx];
                        state
                            .meta_variants
                            .get(&task.variant_id)
                            .unwrap()
                            .name
                            .clone()
                    }
                    ProcEntry::MapperCall(idx) => state
                        .mapper_call_kinds
                        .get(&proc.mapper_calls[idx].kind)
                        .unwrap()
                        .name
                        .clone(),
                    ProcEntry::RuntimeCall(idx) => state
                        .runtime_call_kinds
                        .get(&proc.runtime_calls[idx].kind)
                        .unwrap()
                        .name
                        .clone(),
                };

                let default = Event {
                    name: "",
                    category: "task",
                    phase: "X",
                    timestamp: 0.0,
                    duration: 0.0,
                    pid: proc.node_id.0,
                    tid: proc.proc_in_node,
                };

                let mut start = time_range.start.unwrap();
                for wait in &waiters.wait_intervals {
                    if !first {
                        write!(file, ",")?;
                    }
                    first = false;
                    serde_json::to_writer(
                        &file,
                        &Event {
                            name: &name,
                            timestamp: ts_to_us(start),
                            duration: ts_to_us(wait.start - start),
                            ..default
                        },
                    )?;
                    write!(file, ",")?;
                    serde_json::to_writer(
                        &file,
                        &Event {
                            name: &format!("{} (waiting)", &name),
                            timestamp: ts_to_us(wait.start),
                            duration: ts_to_us(wait.ready - wait.start),
                            ..default
                        },
                    )?;
                    write!(file, ",")?;
                    serde_json::to_writer(
                        &file,
                        &Event {
                            name: &format!("{} (ready)", &name),
                            timestamp: ts_to_us(wait.ready),
                            duration: ts_to_us(wait.end - wait.ready),
                            ..default
                        },
                    )?;
                    start = max(start, wait.end);
                }
                if start < time_range.stop.unwrap() {
                    if !first {
                        write!(file, ",")?;
                    }
                    first = false;
                    serde_json::to_writer(
                        &file,
                        &Event {
                            name: &name,
                            timestamp: ts_to_us(start),
                            duration: ts_to_us(time_range.stop.unwrap() - start),
                            ..default
                        },
                    )?;
                }
            }
        }
    }

    writeln!(file, "]")?;

    Ok(())
}
