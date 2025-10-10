#!/usr/bin/env bash

# run_provider_sweeps.sh
#
# Purpose:
#   - Execute MedQA robustness runs across multiple providers/models.
#   - Providers run IN PARALLEL; within each provider, models run SEQUENTIALLY.
#   - Failures are logged per-run but do NOT stop other runs.
#   - Results are written under the repository's default `results/` directory by the app.
#
# What it runs:
#   OpenAI (model.max_send_messages=100):
#     - openai/gpt-4o-2024-08-06 (no reasoning parameter)
#     - openai/gpt-5-mini-2025-08-07 with reasoning_effort=low
#     - openai/gpt-5-mini-2025-08-07 with reasoning_effort=high
#     - openai/gpt-5-2025-08-07 with reasoning_effort=medium
#     - openai/gpt-5-2025-08-07 with NO reasoning (omit reasoning_effort)
#
#   Anthropic (model.max_send_messages=20):
#     - anthropic/claude-sonnet-4-5-20250929 (no reasoning)
#     - anthropic/claude-sonnet-4-20250514 (no reasoning)
#     (Extended thinking is OFF by default when not provided.)
#
#   xAI (model.max_send_messages=20):
#     - xai/grok-4-fast-non-reasoning
#     - xai/grok-4-0709 (no explicit flag to disable reasoning exposed; run as-is)
#
# Dataset scope:
#   - Entire dataset (run.n_rows=all)
#   - The codebase default uses the 8 specified follow-ups; no override is required
#
# Requirements:
#   - API keys are already available in this shell (OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY)
#   - `medqa-deep` entrypoint is on PATH (installed via this repo)
#
# Usage:
#   chmod +x run_provider_sweeps.sh
#   ./run_provider_sweeps.sh

set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$ROOT_DIR/logs/sweeps/$TIMESTAMP"
mkdir -p "$LOG_ROOT/openai" "$LOG_ROOT/anthropic" "$LOG_ROOT/xai" "$LOG_ROOT/together"

# Helper: run one command, append logs, never stop the script on failure
run_and_log() {
  local log_file="$1"; shift
  local provider_log="$1"; shift
  local cmd=("$@")
  # START/END + command stdout go to both per-run log and provider aggregate log
  echo "[$(date +%H:%M:%S)] START: ${cmd[*]}" | tee -a "$log_file" | tee -a "$provider_log" >/dev/null
  ("${cmd[@]}" 2>&1 | tee -a "$log_file" | tee -a "$provider_log" >/dev/null) || \
    echo "[$(date +%H:%M:%S)] FAIL: ${cmd[*]}" | tee -a "$log_file" | tee -a "$provider_log" >/dev/null
  echo "[$(date +%H:%M:%S)] END: ${cmd[*]}" | tee -a "$log_file" | tee -a "$provider_log" >/dev/null
}

run_openai_group() {
  local outdir="$LOG_ROOT/openai"
  local provider_log="$outdir/provider.log"
  local common=(medqa-deep run.n_rows=all model.max_send_messages=100)

  # 4o (no reasoning)
  run_and_log "$outdir/openai_gpt-4o-2024-08-06.log" "$provider_log" \
    "${common[@]}" model.id=openai/gpt-4o-2024-08-06

  # gpt-5-mini (low/high)
  run_and_log "$outdir/openai_gpt-5-mini-2025-08-07_low.log" "$provider_log" \
    "${common[@]}" model.id=openai/gpt-5-mini-2025-08-07 model.extra_kwargs.reasoning_effort=low
  run_and_log "$outdir/openai_gpt-5-mini-2025-08-07_high.log" "$provider_log" \
    "${common[@]}" model.id=openai/gpt-5-mini-2025-08-07 model.extra_kwargs.reasoning_effort=high

  # gpt-5 (medium and no-reasoning)
  run_and_log "$outdir/openai_gpt-5-2025-08-07_medium.log" "$provider_log" \
    "${common[@]}" model.id=openai/gpt-5-2025-08-07 model.extra_kwargs.reasoning_effort=medium
  run_and_log "$outdir/openai_gpt-5-2025-08-07_off.log" "$provider_log" \
    "${common[@]}" model.id=openai/gpt-5-2025-08-07
}

run_anthropic_group() {
  local outdir="$LOG_ROOT/anthropic"
  local provider_log="$outdir/provider.log"
  local common=(medqa-deep run.n_rows=all model.max_send_messages=20)

  # Sonnet 4.5 / 4 — no reasoning (omit extended thinking)
  run_and_log "$outdir/anthropic_claude-sonnet-4-5-20250929_off.log" "$provider_log" \
    "${common[@]}" model.id=anthropic/claude-sonnet-4-5-20250929
  run_and_log "$outdir/anthropic_claude-sonnet-4-20250514_off.log" "$provider_log" \
    "${common[@]}" model.id=anthropic/claude-sonnet-4-20250514
}

run_xai_group() {
  local outdir="$LOG_ROOT/xai"
  local provider_log="$outdir/provider.log"
  local common=(medqa-deep run.n_rows=all model.max_send_messages=20)

  # Fast non-reasoning variant
  run_and_log "$outdir/xai_grok-4-fast-non-reasoning.log" "$provider_log" \
    "${common[@]}" model.id=xai/grok-4-fast-non-reasoning

  # grok-4-0709 — no public param to force disable reasoning; running default
  run_and_log "$outdir/xai_grok-4-0709_off.log" "$provider_log" \
    "${common[@]}" model.id=xai/grok-4-0709
}

run_together_group() {
  local outdir="$LOG_ROOT/together"
  local provider_log="$outdir/provider.log"
  local common=(medqa-deep run.n_rows=all model.max_send_messages=20)

  # together_ai/openai/gpt-oss-20b
  run_and_log "$outdir/together_openai_gpt-oss-20b.log" "$provider_log" \
    "${common[@]}" model.id=together_ai/openai/gpt-oss-20b

  # openai/gpt-oss-120b
  run_and_log "$outdir/together_openai_gpt-oss-120b.log" "$provider_log" \
    "${common[@]}" model.id=openai/gpt-oss-120b

  # together_ai/Qwen/Qwen2.5-72B-Instruct
  run_and_log "$outdir/together_qwen2.5-72b-instruct.log" "$provider_log" \
    "${common[@]}" model.id=together_ai/Qwen/Qwen2.5-72B-Instruct
}

echo "Logs will be written to: $LOG_ROOT"

# Providers in parallel, models within each sequential
run_openai_group &
PID_OPENAI=$!
run_anthropic_group &
PID_ANTHROPIC=$!
run_xai_group &
PID_XAI=$!
run_together_group &
PID_TOGETHER=$!

wait $PID_OPENAI || true
wait $PID_ANTHROPIC || true
wait $PID_XAI || true
wait $PID_TOGETHER || true

echo "\n==== Sweep Completed ===="
echo "Log root: $LOG_ROOT"
echo "Failures (if any):"
grep -R "FAIL:" "$LOG_ROOT" || echo "No failures detected."

exit 0


