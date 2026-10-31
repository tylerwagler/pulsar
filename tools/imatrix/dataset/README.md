# Imatrix calibration dataset

This directory contains DeepSeek-rendered chat prompts for collecting activation
statistics before building new low-bit checkpoints.

The tracked files are the dataset.  The generator still names the upstream ds4
source files this tree no longer has (and skips missing files silently), so
re-running it does not reproduce them:

```sh
python3 tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

Generated files:

- `prompts.jsonl`: structured records with messages and rendered prompt text.
- `rendered_prompts.txt`: all rendered prompts, separated by visible markers.
- `rendered_prompts_nothink.txt`: only prompts ending with `</think>`.
- `rendered_prompts_think.txt`: only prompts ending with `<think>`.
- `manifest.json`: counts, byte totals, and rough token estimate.

The renderer mirrors the server prompt shape:

```text
<｜begin▁of▁sentence｜>system<｜User｜>...<｜Assistant｜><think>
<｜begin▁of▁sentence｜>system<｜User｜>...<｜Assistant｜></think>
```

Some records include DSML tool schemas, sampled DSML tool calls, and tool-result
turns so the imatrix sees the same special-token patterns used by agent clients.
The corpus is provider-neutral and also includes language/prose rewriting,
summarization, copy-editing, extraction, multilingual translation, programming
prompts, Bash scripting, algorithm recall, `pulsar-eval` benchmark-reasoning
prompts, long-context code synthesis, agent transcript replay, log diagnosis,
prose fact recovery, delayed-constraint and small needle tasks, CUDA/C code
review tasks, and inference-specific debugging tasks.

For normal imatrix collection, use `rendered_prompts.txt` so calibration covers
both thinking and non-thinking modes. Split files are provided for ablations.
