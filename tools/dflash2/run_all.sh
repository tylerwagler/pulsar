#!/bin/bash
# CPU replay of all four fixtures in parallel (12 threads each).
cd /home/claude/Projects/AI/temp/dflash2
PY=/home/claude/Projects/AI/temp/v41q/.venv/bin/python
mkdir -p results
run() {  # fixture ids prompt_len
  $PY replay.py run --fixture $1 --ids $2 --prompt-len $3 --taps taps/$1 --threads 12 --out results/$1.json \
    > results/$1.log 2>&1
}
run code p1/ours-td305-code-d6-off.ids 51 &
run prose p1/ours-td305-prose-d6-off.ids 31 &
run story p1/ours-td305-story-d6-off.ids 531 &
run edit taps/edit.ids 1476 &
wait
cat results/*.log
