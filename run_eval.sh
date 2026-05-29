#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   TEST_MODEL="nvidia/nvidia-nemotron-nano-9b-v2" \
#   MODEL_NAME="nemotron-nano-9b-v2-my-run" \
#   JUDGE_MODEL="anthropic.claude-3-7-sonnet-20250219-v1:0" \
#   THREADS=6 ITERATIONS=1 VERBOSITY=INFO \
#   bash run_eval.sh
#

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Think-off preset: use local ReasonOff model at port 8005 when REASON_OFF=1
if [ "${REASON_OFF:-0}" = "1" ]; then
  export VLLM_PORT=${VLLM_PORT:-8005}
  export VLLM_MODEL_ID="${VLLM_MODEL_ID:-/localhome/local-hndo/Egde_model/rc_candidates_03-02-26/i-XLAM-CAL-ReasonOff-Step20-onlyTauBench-ckpt_t1_lr5e-6_kl1e-3_step_5}"
  export MODEL_NAME="${MODEL_NAME:-i-XLAM-ReasonOff}"
fi

TEST_MODEL=${TEST_MODEL:-nvidia/nvidia-nemotron-nano-9b-v2}
MODEL_NAME=${MODEL_NAME:-nemotron-nano-9b-v2-eval}
# Track whether JUDGE_MODEL was explicitly provided before applying the default
_JUDGE_EXPLICITLY_SET="${JUDGE_MODEL+set}"
JUDGE_MODEL=${JUDGE_MODEL:-anthropic.claude-3-7-sonnet-20250219-v1:0}
DEFAULT_JUDGE="anthropic.claude-3-7-sonnet-20250219-v1:0"
THREADS=${THREADS:-6}
ITERATIONS=${ITERATIONS:-1}
VERBOSITY=${VERBOSITY:-INFO}
VLLM_PORT=${VLLM_PORT:-8011}
VLLM_HOST=${VLLM_HOST:-localhost}
VLLM_MODEL_ID=${VLLM_MODEL_ID:-}

echo "[INFO] Starting EQBench3 rubric run"
echo "[INFO] Test model:    $TEST_MODEL"
echo "[INFO] Model name:    $MODEL_NAME"
echo "[INFO] Judge model:   $JUDGE_MODEL"
echo "[INFO] Threads:       $THREADS, Iterations: $ITERATIONS"

# Optional: quick vLLM config via env VLLM_PORT (e.g., 8009)
# Skip VLLM_PORT auto-config if TEST_API_URL was already explicitly provided by the caller.
if [ -n "${VLLM_PORT:-}" ] && [ -z "${TEST_API_URL:-}" ]; then
  export TEST_API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"
  export TEST_API_KEY="${TEST_API_KEY:-dummy-key}"
  # Only route judge to local vLLM if the caller asked for it (non-default judge or explicit URL)
  if [ -z "${JUDGE_API_URL:-}" ]; then
    if [ "${JUDGE_MODEL}" != "${DEFAULT_JUDGE}" ]; then
      export JUDGE_API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"
    else
      echo "[INFO] Leaving judge on external API because JUDGE_MODEL is the default Anthropic model."
    fi
  fi
  if [ -z "${JUDGE_API_KEY:-}" ] && [ "${JUDGE_MODEL}" != "${DEFAULT_JUDGE}" ]; then
    export JUDGE_API_KEY="dummy-key"
  fi
  echo "[INFO] Using vLLM at ${TEST_API_URL} (set by VLLM_HOST=${VLLM_HOST}, VLLM_PORT=${VLLM_PORT})"
  echo "[INFO] Judge API URL: ${JUDGE_API_URL:-<unset>}"
elif [ -n "${TEST_API_URL:-}" ]; then
  echo "[INFO] Using pre-set TEST_API_URL: ${TEST_API_URL}"
  echo "[INFO] Judge API URL: ${JUDGE_API_URL:-<unset, using external>}"
fi

# If a VLLM_MODEL_ID is provided, use it for the API model id while keeping a human-friendly MODEL_NAME unless overridden
if [ -n "${VLLM_MODEL_ID}" ]; then
  TEST_MODEL="$VLLM_MODEL_ID"
  echo "[INFO] Overriding TEST_MODEL with VLLM_MODEL_ID"
fi

# If JUDGE_MODEL was NOT explicitly set by the user and VLLM_MODEL_ID is provided, route judge to local vLLM too.
# If the user explicitly passed JUDGE_MODEL (even if it equals DEFAULT_JUDGE), respect it and keep the external judge.
if [ -n "${VLLM_MODEL_ID}" ] && [ "${JUDGE_MODEL}" = "${DEFAULT_JUDGE}" ] && [ "${_JUDGE_EXPLICITLY_SET}" != "set" ]; then
  JUDGE_MODEL="$VLLM_MODEL_ID"
  echo "[INFO] Overriding JUDGE_MODEL with VLLM_MODEL_ID to use local vLLM as judge"
fi

# If we just redirected the judge model to the local vLLM, also point its API URL/key there when still unset
if [ -n "${VLLM_PORT:-}" ] && [ "${JUDGE_MODEL}" = "${VLLM_MODEL_ID}" ]; then
  if [ -z "${JUDGE_API_URL:-}" ]; then
    export JUDGE_API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"
    echo "[INFO] Routing judge API to local vLLM because JUDGE_MODEL matches VLLM_MODEL_ID."
  fi
  if [ -z "${JUDGE_API_KEY:-}" ]; then
    export JUDGE_API_KEY="dummy-key"
  fi
fi

# ensure MODEL_NAME visible to child processes (CSV export step)
export MODEL_NAME

# 1) Ensure venv + deps
if [ ! -d .venv ]; then
  echo "[INFO] Creating virtualenv (.venv)"
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt >/dev/null

# 2) Check .env for API keys
if [ ! -f .env ]; then
  echo "[WARN] .env not found. Make sure TEST_API_URL/KEY and JUDGE_API_URL/KEY are set in env."
fi

# 3) Run rubric-only evaluation
python eqbench3.py \
  --test-model "$TEST_MODEL" \
  --model-name "$MODEL_NAME" \
  --judge-model "$JUDGE_MODEL" \
  --no-elo \
  --iterations "$ITERATIONS" \
  --threads "$THREADS" \
  --verbosity "$VERBOSITY"

# 4) Export results to CSV (overall 0-100 + per-criterion 0-10)
CSV_PATH=$(python - <<'PY'
import os, json, csv, sys
from pathlib import Path

model_name = Path('.').resolve().name  # unused, just to appease linters

RUNS = Path('eqbench3_runs.json')
if not RUNS.exists():
    print('')
    sys.exit(0)
data = json.loads(RUNS.read_text())

target_name = os.environ.get('MODEL_NAME') if 'MODEL_NAME' in os.environ else None
if not target_name:
    print('')
    sys.exit(0)

# pick the latest run for MODEL_NAME
candidates = []
for k,v in data.items():
    if isinstance(v,dict) and v.get('model_name') == target_name:
        candidates.append((v.get('end_time') or '', k))
if not candidates:
    print('')
    sys.exit(0)
run_key = sorted(candidates)[-1][1]
run = data[run_key]

# overall rubric 0-100
avg20 = (run.get('results') or {}).get('average_rubric_score')
avg100 = round(avg20*5, 2) if isinstance(avg20, (int,float)) else None

# per-criterion (0-10) from allowed + extras
allowed = {
  'demonstrated_empathy','pragmatic_ei','depth_of_insight','social_dexterity',
  'emotional_reasoning','message_tailoring','theory_of_mind','subtext_identification',
  'intellectual_grounding','correctness',
}
extra = {'validating','challenging'}
agg = {k:[0.0,0] for k in sorted(allowed|extra)}

sc_tasks = run.get('scenario_tasks') or {}
for iter_str, scen_map in sc_tasks.items():
    if not isinstance(scen_map, dict):
        continue
    for sid, t in scen_map.items():
        scores = t.get('rubric_scores')
        if not isinstance(scores, dict):
            continue
        for k in agg:
            v = scores.get(k)
            if isinstance(v,(int,float)):
                s,c = agg[k]
                agg[k] = [s+float(v), c+1]
per10 = {k:(round((s/c)/2,3) if c else None) for k,(s,c) in agg.items()}

out_dir = Path('results'); out_dir.mkdir(exist_ok=True)
csv_path = out_dir / f"rubric_{target_name}_{run_key}.csv"
with csv_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['metric','score','scale'])
    w.writerow(['average_rubric_score', avg100 if avg100 is not None else 'N/A', '0-100'])
    for k in sorted(per10.keys()):
        w.writerow([k, per10[k] if per10[k] is not None else 'N/A', '0-10'])

print(str(csv_path))
PY)

if [ -z "$CSV_PATH" ]; then
  echo "[ERROR] Could not locate run results to export CSV."
  exit 1
fi

# 5) Print summary
AVG_SCORE=$(awk -F, 'NR==2{print $2}' "$CSV_PATH" 2>/dev/null || echo "N/A")
echo "[DONE] Model: $MODEL_NAME"
echo "[DONE] Average Rubric (0-100): $AVG_SCORE"
echo "[DONE] CSV: $CSV_PATH"
