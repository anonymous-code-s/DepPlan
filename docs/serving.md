# Qwen inference service

The serving configuration for our experiments uses one vLLM instance running Qwen2.5-32B-Instruct in FP16 with
tensor parallelism across four NVIDIA RTX 3090 GPUs (24 GB each). The launcher
sets `--dtype float16` and `--tensor-parallel-size 4` and exposes the model name
`Qwen2.5-32B-Instruct`, which matches `configs/qa.json`.

Prepare the Qwen2.5-32B-Instruct weights and a CUDA-compatible vLLM installation
in the serving environment. Set the local model directory and four device IDs:

```bash
export DEPPLAN_QWEN_MODEL=/path/to/Qwen2.5-32B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/serve_qwen.sh --dry-run
bash scripts/serve_qwen.sh
```

The dry run prints the server command. Running the launcher without `--dry-run`
starts an OpenAI-compatible server at
`http://127.0.0.1:8000/v1`. Configure a QA process with:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

`DEPPLAN_PYTHON` selects the Python executable containing vLLM;
`DEPPLAN_SERVE_HOST` and `DEPPLAN_SERVE_PORT` change the listening address.
The script requires exactly four distinct visible device IDs. Record the GPU
model, CUDA/PyTorch/vLLM versions, model revision and server command with each
new experiment.

Planning temperature 0.7, greedy answer decoding and seeds 42, 123 and 2026 are
supplied by the QA clients with each request.
