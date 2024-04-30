# Legion Prof

Legion Prof is a profiler for [Legion](https://legion.stanford.edu)
applications that visualizes the tasks and other operations that occur
during a program's execution. Documentation for installing and using
the profiler is provided on the [Legion profiling
page](https://legion.stanford.edu/profiling).

## Quickstart

Always make sure that you build Legion in release mode (`DEBUG=0` or
`-DCMAKE_BUILD_TYPE=Release`) when profiling Legion
applications. Release mode provides a substantial (often factor of 5x
or larger) speedup over debug builds.

After the application has been built, run with:

```
./your_legion_app -lg:prof 1 -lg:prof_logfile prof_%.gz
```

The flag `-lg:prof 1` enables profiling, and `-lg:prof_logfile`
specifies the path where the corresponding log files will be
written. The character `%` will be replaced with the rank number in
multi-node runs (starting at 0).

Once you have a set of logs, you can install the profiler. If you do
not already have Rust installed, run:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

And then install the profiler with:

```
cargo install --all-features --locked legion_prof@0.YYMM.0
```

Where `YYMM` corresponds to the version of Legion you used in the
application (i.e., `YY.MM.0`).

**Important:** The version of the profiler **MUST** match the version
of Legion used in the application, or the profiler may be unable to
parse the logs.

If you are running an unreleased version of Legion (e.g., on the
`master` branch), then you can build with:

```
cargo install --all-features --locked --path legion/tools/legion_prof_rs
```

Once `legion_prof` is installed, run:

```
legion_prof --view prof_*.gz
```

For other modes of running Legion Prof, see the [full profiler
documentation](https://legion.stanford.edu/profiling).

## Development

We use standard Rust development practices in developing Legion Prof.

The code is formatted via `cargo fmt`, and should compile warning-free
at all times. Be sure when you are compiling to check with
`--all-features`:

```
cargo check --all-features
```

When developing, you may wish to build your local copy, which you can
do with:

```
cargo run --release --all-features -- --view ...
```

If you need to modify the `legion_prof_viewer` frontend as well, you
can modify your `Cargo.toml` to point to your local copy:

```toml
legion_prof_viewer = { path = ".../path/to/prof-viewer", optional = true }
```
