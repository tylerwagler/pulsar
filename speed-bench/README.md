## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

`pulsar-bench` is a development tool (`make pulsar-bench`; it is not part of
the release).  It measures plain (non-speculative) prefill and decode at a
series of context frontiers.  Run it as:

```
./pulsar-bench \
  -m /path/to/<container-dir> \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv speed-bench/gb10.csv
```

Without `--csv` the CSV goes to stdout.  `./pulsar-bench --help benchmark`
lists the sweep options.

Provide a PR including your numbers if your hardware was not already tested.
Call the benchmark CSV file something like `gb10.csv`, so that it is clear
what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/gb10.csv --title "GB10 t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/gb10_ts.svg`.
