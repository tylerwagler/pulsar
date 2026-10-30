#!/bin/bash
# A cold prefill sweep (2k -> 8k) with pulsar-bench after a page-cache drop.
#   WT=<built tree> MODEL=<container> OUT=<output dir> prof/clean_bench.sh
set -u
WT=${WT:?set WT to the built engine tree}
MODEL=${MODEL:?set MODEL to the model container}
OUT=${OUT:?set OUT to the output directory}
cd "$WT"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
./pulsar-bench -m "$MODEL" --prompt-file speed-bench/promessi_sposi.txt \
   --ctx-start 2048 --ctx-max 8192 --step-incr 6144 --gen-tokens 0 \
   --csv "$OUT/clean_prefill.csv" 2>"$OUT/clean_bench.log"
echo "exit=$?"
cat "$OUT/clean_prefill.csv"
