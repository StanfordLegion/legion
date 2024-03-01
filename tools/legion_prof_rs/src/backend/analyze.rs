use crate::state::{ProcEntryKind, ProcEntryStats, State, Timestamp};
use std::cmp::Reverse;
use std::collections::BTreeMap;

fn print_statistics(
    state: &State,
    statistics: &BTreeMap<ProcEntryKind, ProcEntryStats>,
    category: &str,
) {
    // Find the order to output these statistics in,
    // currently we'll do it by the max running time
    let mut ordering = BTreeMap::<Reverse<Timestamp>, Vec<ProcEntryKind>>::new();
    for (entry, stats) in statistics {
        // Probably will never have a collision at nanosecond granularities
        // but use a vector just to be safe
        ordering
            .entry(Reverse(stats.running_time))
            .or_insert(Vec::new())
            .push(*entry);
    }

    println!("");
    println!("  -------------------------");
    println!("  {}", category);
    println!("  -------------------------");
    for entries in ordering.values() {
        for entry in entries {
            println!();
            match entry {
                ProcEntryKind::Task(task_id, variant_id) => {
                    let key = (*task_id, *variant_id);
                    println!(
                        "      Task {} Variant {}",
                        state
                            .task_kinds
                            .get(&task_id)
                            .unwrap()
                            .name
                            .as_ref()
                            .unwrap(),
                        state.variants.get(&key).unwrap().name
                    );
                }
                ProcEntryKind::MetaTask(variant_id) => {
                    println!(
                        "      Meta-Task {}",
                        state.meta_variants.get(&variant_id).unwrap().name
                    );
                }
                ProcEntryKind::MapperCall(call_kind) => {
                    println!(
                        "      Mapper Call {}",
                        state.mapper_call_kinds.get(&call_kind).unwrap().name
                    );
                }
                ProcEntryKind::RuntimeCall(call_kind) => {
                    println!(
                        "      Runtime Call {}",
                        state.runtime_call_kinds.get(&call_kind).unwrap().name
                    );
                }
                ProcEntryKind::ProfTask => {
                    println!("       Profiler Response");
                }
                ProcEntryKind::GPUKernel(task_id, variant_id) => {
                    let key = (*task_id, *variant_id);
                    println!(
                        "      GPU Kernel for Task {} Variant {}",
                        state
                            .task_kinds
                            .get(&task_id)
                            .unwrap()
                            .name
                            .as_ref()
                            .unwrap(),
                        state.variants.get(&key).unwrap().name
                    );
                }
            }
            let threshold = Timestamp(1000000000); // 1e9 ns or 1e6 us
            let stats = statistics.get(&entry).unwrap();
            println!("          Invocations: {}", stats.invocations);
            if stats.total_time < threshold {
                println!("          Total time: {:.3} us", stats.total_time.to_us());
            } else {
                println!("          Total time: {:.3e} us", stats.total_time.to_us());
            }
            if stats.running_time < threshold {
                println!(
                    "          Running time: {:.3} us ({:.2}%)",
                    stats.running_time.to_us(),
                    100.0 * stats.running_time.to_us() / stats.total_time.to_us()
                );
            } else {
                println!(
                    "          Running time: {:.3e} us ({:.2}%)",
                    stats.running_time.to_us(),
                    100.0 * stats.running_time.to_us() / stats.total_time.to_us()
                );
            }
            if stats.next_mean < threshold.to_us() {
                println!("          Average time: {:.3} us", stats.next_mean);
            } else {
                println!("          Average time: {:.3e} us", stats.next_mean);
            }
            let stddev = if stats.invocations > 1 {
                let variance = stats.next_stddev / ((stats.invocations - 1) as f64);
                variance.sqrt()
            } else {
                0.0
            };
            if stddev < threshold.to_us() {
                println!("          Std Dev: {:.3} us", stddev);
            } else {
                println!("          Std Dev: {:.3e} us", stddev);
            }
            if stats.min_time < threshold {
                println!("          Min time: {:.3} us", stats.min_time.to_us());
            } else {
                println!("          Min time: {:.3e} us", stats.min_time.to_us());
            }
            if stats.max_time < threshold {
                println!("          Max time: {:.3} us", stats.max_time.to_us());
            } else {
                println!("          Max time: {:.3e} us", stats.max_time.to_us());
            }
        }
    }
}

pub fn analyze_statistics(state: &State) {
    let mut task_stats = BTreeMap::new();
    let mut runtime_stats = BTreeMap::new();
    let mut mapper_stats = BTreeMap::new();
    for proc in state.procs.values() {
        proc.accumulate_statistics(&mut task_stats, &mut runtime_stats, &mut mapper_stats);
    }
    print_statistics(state, &task_stats, "Task Statistics");
    print_statistics(state, &runtime_stats, "Runtime Statistics");
    print_statistics(state, &mapper_stats, "Mapper Statistics");
}
