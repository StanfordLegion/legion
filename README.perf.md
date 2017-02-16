# Legion Performance Test Infrastructure

This document describes Legion's performance regression test
infrastructure. Before reading this document, we recommend you read
the documentation on the [test infrastructure for
correctness](README.test.md), because the performance infrastructure
makes significant use of the same tools.

As with correctness, the performance test infrastructure has a number
of components. The flow is roughly:

  * The `test.py` script again encapsulates all test logic.
  * Another script `perf.py` runs individual tests and uploads the
    resulting measurements.
  * Raw measurements are stored in the
    [`perf-data` repository](https://github.com/StanfordLegion/perf-data).
  * The script `tools/perf_chart.py` parses the raw measurements to
    produce a table aggregating the results, which it uploads.
  * An [HTML-based front-end](https://stanfordlegion.github.io/perf-frontend/perf_chart.html)
    renders the aggregated results.

This may seem complicated, but critically all of the moving parts are
configured explicitly, and versioned and stored in Git repositories
(including the data). There is no local configuration that is required
and no local state that can be corrupted or lost. In the past, as a
result of hard drive crashes on Sapling, we've lost both data and
locally modified installations of tools, making it impossible to
recover past versions of our performance infrastructure.

## Running Performance Tests

All tests run through the `test.py` script:

```
PERF_ACCESS_TOKEN=... DEBUG=0 ./test.py --test=perf
```

A couple of notes:

  * `PERF_ACCESS_TOKEN` is a [Github personal access
    token](https://github.com/settings/tokens). The token must be for
    a user account with write permission to the
    [`perf-data` repository](https://github.com/StanfordLegion/perf-data).
    We have a special Github account which we use for this purpose.
  * `DEBUG=0` is required to ensure performance.
  * `--test=perf` requests that the performance tests be run.

These performance tests are enabled in Gitlab, so they run
automatically on every commit. In addition to this, there is a cronjob
that runs nightly on Sapling to measure performance on a machine with
beefier hardware. Ideally these would also run via Gitlab, but proper
cluster management would be required and is beyond the scope of this
document.

### How to Add New Tests

Tests are in `test.py` under `run_test_perf`. The specific way in
which you add a new test depends on the kind of test you're adding:

  * **C++ test in the main Legion repository:** Add the test to
    `legion_cxx_perf_tests` at the top of the file.

  * **Regent test in the main Legion repository:** Add the test to
    `regent_perf_tests` at the top of the file.

  * **External C++ or Regent test:** Add code to the end of
      `run_test_perf` to clone, build, and run the
      repository. Remember to use `perf.py` to launch the test as
      described below.

## Capturing Measurements

## Measurement Storage and Aggregation

## Visualization Front-end
