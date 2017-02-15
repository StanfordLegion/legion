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
configured in human-readable text, and versioned and stored in Git
repositories (including the data). There is no local configuration
that is required and no local state that can be corrupted or
lost.

## Running Performance Tests

## Capturing Measurements

## Measurement Storage and Aggregation

## Visualization Front-end
