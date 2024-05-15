use std::cmp::max;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::Path;

use serde::Serialize;
use serde_json;

use crate::state::{Container, ProcEntryKind, State};

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

pub fn emit_trace<P: AsRef<Path>>(state: &State, path: P, force: bool) -> io::Result<()> {
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
        for point in proc.time_points(None) {
            if point.first {
                let entry = proc.entry(point.entry);
                let (time_range, waiters) = (&entry.time_range, &entry.waiters);

                let name = match entry.kind {
                    ProcEntryKind::Task(task_id, variant_id)
                    | ProcEntryKind::GPUKernel(task_id, variant_id) => {
                        let task_name = &state.task_kinds.get(&task_id).unwrap().name;
                        let variant_name =
                            &state.variants.get(&(task_id, variant_id)).unwrap().name;
                        match task_name {
                            Some(task_name) => format!("{} ({})", task_name, variant_name),
                            None => variant_name.clone(),
                        }
                    }
                    ProcEntryKind::MetaTask(variant_id) => {
                        state.meta_variants.get(&variant_id).unwrap().name.clone()
                    }
                    ProcEntryKind::MapperCall(_, _, kind) => {
                        state.mapper_call_kinds.get(&kind).unwrap().name.clone()
                    }
                    ProcEntryKind::RuntimeCall(kind) => {
                        state.runtime_call_kinds.get(&kind).unwrap().name.clone()
                    }
                    ProcEntryKind::ApplicationCall(prov) => {
                        state.provenances.get(&prov).unwrap().name.clone()
                    }
                    ProcEntryKind::ProfTask => "ProfTask".to_owned(),
                };

                let default = Event {
                    name: "",
                    category: "task",
                    phase: "X",
                    timestamp: 0.0,
                    duration: 0.0,
                    pid: proc.proc_id.node_id().0,
                    tid: proc.proc_id.proc_in_node(),
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
                            timestamp: start.to_us(),
                            duration: (wait.start - start).to_us(),
                            ..default
                        },
                    )?;
                    write!(file, ",")?;
                    serde_json::to_writer(
                        &file,
                        &Event {
                            name: &format!("{} (waiting)", &name),
                            timestamp: wait.start.to_us(),
                            duration: (wait.ready - wait.start).to_us(),
                            ..default
                        },
                    )?;
                    write!(file, ",")?;
                    serde_json::to_writer(
                        &file,
                        &Event {
                            name: &format!("{} (ready)", &name),
                            timestamp: (wait.ready).to_us(),
                            duration: (wait.end - wait.ready).to_us(),
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
                            timestamp: start.to_us(),
                            duration: (time_range.stop.unwrap() - start).to_us(),
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
