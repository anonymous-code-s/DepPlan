#!/usr/bin/env bash
# Serve the paper's Qwen backbone with FP16 weights and four-way TP.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: DEPPLAN_QWEN_MODEL=/path/to/Qwen2.5-32B-Instruct bash scripts/serve_qwen.sh [--dry-run]

Required: DEPPLAN_QWEN_MODEL, a prepared Qwen2.5-32B-Instruct model directory.
Optional: CUDA_VISIBLE_DEVICES (default 0,1,2,3; exactly four distinct devices),
          DEPPLAN_PYTHON (default python), DEPPLAN_SERVE_HOST (default 127.0.0.1),
          DEPPLAN_SERVE_PORT (default 8000).
--dry-run prints the command without importing vLLM or starting a server.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# > 1 )) || [[ ${1:-} != '' && ${1:-} != --dry-run ]]; then
  usage >&2
  exit 2
fi
: "${DEPPLAN_QWEN_MODEL:?Set DEPPLAN_QWEN_MODEL to the prepared Qwen2.5-32B-Instruct directory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a depplan_devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#depplan_devices[@]} != 4 )) || [[ $CUDA_VISIBLE_DEVICES == *, ]]; then
  printf '%s\n' 'CUDA_VISIBLE_DEVICES must select exactly four distinct GPUs.' >&2
  exit 2
fi
declare -A depplan_seen_devices=()
for depplan_device in "${depplan_devices[@]}"; do
  if [[ -z $depplan_device || $depplan_device == *[[:space:]]* || -n ${depplan_seen_devices[$depplan_device]:-} ]]; then
    printf '%s\n' 'CUDA_VISIBLE_DEVICES must select exactly four distinct GPUs.' >&2
    exit 2
  fi
  depplan_seen_devices["$depplan_device"]=1
done

depplan_command=(
  "${DEPPLAN_PYTHON:-python}" -m vllm.entrypoints.openai.api_server
  --model "$DEPPLAN_QWEN_MODEL"
  --served-model-name Qwen2.5-32B-Instruct
  --dtype float16
  --tensor-parallel-size 4
  --host "${DEPPLAN_SERVE_HOST:-127.0.0.1}"
  --port "${DEPPLAN_SERVE_PORT:-8000}"
)
if [[ ${1:-} == --dry-run ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${depplan_command[@]}"
  printf '\n'
  exit 0
fi
exec "${depplan_command[@]}"
