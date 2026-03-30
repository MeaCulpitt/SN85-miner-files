#!/bin/bash
# sn85_ralph.sh
#
# Autonomous three-loop Ralph script for SN85 Vidaio miner.
# Runs Claude Code on a schedule matching Bittensor tempo cadence.
#
# Usage:
#   ./sn85_ralph.sh                    # run all three loops continuously
#   ./sn85_ralph.sh --loop fast        # run only the fast loop
#   ./sn85_ralph.sh --loop medium      # run only the medium loop
#   ./sn85_ralph.sh --loop slow        # run only the slow loop (interactive)
#   ./sn85_ralph.sh --once             # run one iteration and exit
#   ./sn85_ralph.sh --dry-run          # show what would run without executing
#
# Loop cadences (from CLAUDE.md):
#   Fast loop:   every tempo (~12 min = 720s). Observation + alignment checks.
#   Medium loop: every 5 fast loops (~60 min). Triage and diagnosis.
#   Slow loop:   when medium loop produces QUALITY_LOW or CONFIG_ERROR. Interactive.
#
# Window gate: 5 fast loops must pass since last deployment before slow loop runs.
#
# Prerequisites:
#   - claude CLI installed and authenticated
#   - CLAUDE.md and registry/ in current directory
#   - ffmpeg with libvmaf for local scoring (optional but recommended)
#   - VIDAIO_REPO_PATH set if using PieAPP from validator source
#
# Logs: ./logs/fast_YYYYMMDD.log, ./logs/medium_YYYYMMDD.log, ./logs/slow_YYYYMMDD.log

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
FAST_INTERVAL=720          # seconds between fast loop iterations (~1 tempo)
MEDIUM_EVERY_N_FAST=5      # run medium loop after every N fast loop iterations
SLOW_CHECK_EVERY_N_FAST=5  # same cadence -- medium always triggers slow check

LOG_DIR="./logs"
REGISTRY_DIR="./registry"
PROGRESS_FILE="${REGISTRY_DIR}/progress.txt"
REGISTRY_FILE="${REGISTRY_DIR}/agent_registry.jsonl"

# ── Argument parsing ──────────────────────────────────────────────────────────
LOOP_MODE="all"
ONCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --loop) LOOP_MODE="$2"; shift 2 ;;
    --once) ONCE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$REGISTRY_DIR"

log() {
  local level="$1"
  local msg="$2"
  local ts
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$ts] [$level] $msg" | tee -a "${LOG_DIR}/${level}_$(date '+%Y%m%d').log"
}

check_prerequisites() {
  if ! command -v claude &>/dev/null; then
    echo "ERROR: claude CLI not found. Install Claude Code first."
    exit 1
  fi
  if [[ ! -f "CLAUDE.md" ]]; then
    echo "ERROR: CLAUDE.md not found. Run from your miner repo root."
    exit 1
  fi
  if [[ ! -f "${REGISTRY_DIR}/task_type.txt" ]]; then
    echo "ERROR: registry/task_type.txt not found."
    echo "Run: python registry/init_registry.py --task-type UPSCALING"
    exit 1
  fi
  if [[ ! -f "${PROGRESS_FILE}" ]]; then
    echo "ERROR: registry/progress.txt not found."
    echo "Run: python registry/init_registry.py"
    exit 1
  fi

  local task_type
  task_type=$(cat "${REGISTRY_DIR}/task_type.txt" 2>/dev/null | tr -d '[:space:]')
  log "main" "Task type: ${task_type}"
}

count_fast_loops_since_deploy() {
  if [[ ! -f "$PROGRESS_FILE" ]]; then echo 0; return; fi
  local last_deploy_line=0
  local line_num=0
  local fast_count=0
  while IFS= read -r line; do
    ((line_num++))
    if echo "$line" | grep -q "DEPLOYING"; then
      last_deploy_line=$line_num
      fast_count=0
    elif echo "$line" | grep -q "\] FAST"; then
      if [[ $last_deploy_line -gt 0 ]]; then
        ((fast_count++))
      fi
    fi
  done < "$PROGRESS_FILE"
  echo $fast_count
}

run_alignment_checks() {
  # Run scoring alignment checks before medium/slow loops
  local alignment_script="./scoring_alignment/sn85_vidaio_scoring_alignment.py"
  if [[ ! -f "$alignment_script" ]]; then
    log "main" "Alignment tests not found at $alignment_script -- skipping"
    return 0
  fi
  if python "$alignment_script" &>/dev/null; then
    log "main" "Alignment checks: PASS"
    return 0
  else
    log "main" "Alignment checks: FAIL -- check scoring_alignment logs"
    return 1
  fi
}

# ── Fast loop ─────────────────────────────────────────────────────────────────
run_fast_loop() {
  log "fast" "Starting fast loop iteration"

  # Run append_observation.py first (faster than full claude invocation)
  local gate_count
  gate_count=$(count_fast_loops_since_deploy)

  if [[ -f "${REGISTRY_DIR}/append_observation.py" ]]; then
    if [[ "$DRY_RUN" != true ]]; then
      python "${REGISTRY_DIR}/append_observation.py" \
        --round "$(date '+%s')" \
        2>>"${LOG_DIR}/fast_$(date '+%Y%m%d').log" || true
    fi
  fi

  local prompt
  prompt="@CLAUDE.md @registry/agent_registry.jsonl @registry/progress.txt

You are running the FAST LOOP only. Section 2 of CLAUDE.md defines this.
Steps:
  1. Read the last entry in registry/agent_registry.jsonl.
  2. Check on-chain weights for your UID if btcli is available.
  3. Verify last commit URL is reachable (curl -sI).
  4. Note window gate status: ${gate_count}/5 fast loops since last deployment.
  5. Append one FAST line to registry/progress.txt.
Do NOT run triage. Do NOT propose mutations. Do NOT deploy.
Output exactly one of:
  FAST_COMPLETE -- observation appended
  FAST_ANOMALY:<reason> -- score drop >15% or alignment check failed
  HOLD -- (note gate count in progress.txt but continue observing)"

  if [[ "$DRY_RUN" == true ]]; then
    log "fast" "[DRY RUN] gate=${gate_count}/5"
    echo "FAST_COMPLETE"
    return
  fi

  local result
  result=$(claude --print "$prompt" 2>>"${LOG_DIR}/fast_$(date '+%Y%m%d').log")
  echo "$result" >> "${LOG_DIR}/fast_$(date '+%Y%m%d').log"
  log "fast" "$(echo "$result" | tail -1)"
  echo "$result"
}

# ── Medium loop ───────────────────────────────────────────────────────────────
run_medium_loop() {
  log "medium" "Starting medium loop iteration"

  # Run alignment checks first -- fast and deterministic
  run_alignment_checks || {
    log "medium" "Alignment checks failed. Appending CONFIG_ERROR to progress.txt."
    echo "[$(date '+%Y-%m-%d %H:%M')] MEDIUM -- diagnosis=CONFIG_ERROR reason=alignment_checks_failed" \
      >> "$PROGRESS_FILE"
    echo "MEDIUM_CONFIG_ERROR:alignment_checks_failed"
    return
  }

  local task_type
  task_type=$(cat "${REGISTRY_DIR}/task_type.txt" 2>/dev/null | tr -d '[:space:]')

  local prompt
  prompt="@CLAUDE.md @registry/agent_registry.jsonl @registry/progress.txt @registry/shadow_accuracy.json

You are running the MEDIUM LOOP only for task_type=${task_type}.
Section 2 (Medium Loop) and Section 4 (Triage Protocol) of CLAUDE.md define this.
Steps:
  1. Run all 5 triage steps in order. Stop at first failure.
  2. Produce exactly one diagnosis.
  3. Append MEDIUM and DIAGNOSIS lines to registry/progress.txt.
  4. Update registry/shadow_accuracy.json with latest score estimate.
Do NOT deploy. Do NOT run sv push or any model changes.
Output exactly one of:
  MEDIUM_STABLE -- no action needed, continue monitoring
  MEDIUM_QUALITY_LOW -- PieAPP/VMAF below target, slow loop needed
  MEDIUM_CONFIG_ERROR:<step> -- triage found a fixable issue
  MEDIUM_CONTENT_LENGTH -- ContentLength.TEN upgrade recommended (Tier 0)
  MEDIUM_HOLD:<reason> -- window gate or low confidence
  PLATEAU_REACHED -- rank stable top 20 for 10+ rounds, all mutations rejected"

  if [[ "$DRY_RUN" == true ]]; then
    log "medium" "[DRY RUN] Would invoke claude for medium loop"
    echo "MEDIUM_STABLE"
    return
  fi

  local result
  result=$(claude --print "$prompt" 2>>"${LOG_DIR}/medium_$(date '+%Y%m%d').log")
  echo "$result" >> "${LOG_DIR}/medium_$(date '+%Y%m%d').log"
  log "medium" "$(echo "$result" | tail -1)"
  echo "$result"
}

# ── Slow loop (interactive) ───────────────────────────────────────────────────
run_slow_loop() {
  local trigger_reason="$1"
  log "slow" "Starting slow loop (trigger: ${trigger_reason}) -- INTERACTIVE"

  local gate_count
  gate_count=$(count_fast_loops_since_deploy)

  if [[ $gate_count -lt 5 ]]; then
    log "slow" "Window gate active: ${gate_count}/5 fast loops since last deployment. Skipping."
    echo "SLOW_HOLD:window_gate_${gate_count}_of_5"
    return
  fi

  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  SLOW LOOP -- mutation proposal (${trigger_reason})"
  echo "  Window gate: ${gate_count}/5 -- OPEN"
  echo "════════════════════════════════════════════════════════"
  echo ""
  echo "Claude Code will propose a model or config change."
  echo "You must approve before any deployment executes."
  echo ""

  if [[ "$DRY_RUN" == true ]]; then
    log "slow" "[DRY RUN] Would invoke claude interactively for slow loop"
    return
  fi

  local task_type
  task_type=$(cat "${REGISTRY_DIR}/task_type.txt" 2>/dev/null | tr -d '[:space:]')

  # Run interactively so human can approve
  claude \
    "@CLAUDE.md" \
    "@registry/agent_registry.jsonl" \
    "@registry/progress.txt" \
    "@registry/shadow_accuracy.json" \
    "@registry/mutations.jsonl" \
    "You are running the SLOW LOOP for task_type=${task_type}.
Trigger: ${trigger_reason}
Window gate: ${gate_count}/5 (OPEN)

Section 2 (Slow Loop) of CLAUDE.md defines this. Steps:
  1. Verify all slow loop preconditions.
  2. Run the attribution protocol (Section 5) explicitly. Show your working.
  3. Propose exactly ONE mutation with tier and expected score delta.
  4. For upscaling: run replay_buffer evaluation if local clips exist.
     python replay_buffer/example_eval.py --task upscaling --model-tag proposed
  5. For compression: compute expected score using scoring_function formula.
  6. If deploying: output the exact command for human review. WAIT for approval.
  7. After approval: append DEPLOYING to registry/progress.txt.
  8. Append outcome to registry/mutations.jsonl.
If any precondition fails: explain why and output SLOW_HOLD:<reason>.
If plateau confirmed: output PLATEAU_REACHED."

  local exit_code=$?
  log "slow" "Slow loop completed (exit code: $exit_code)"
}

# ── Main orchestrator ─────────────────────────────────────────────────────────
main() {
  check_prerequisites

  local task_type
  task_type=$(cat "${REGISTRY_DIR}/task_type.txt" 2>/dev/null | tr -d '[:space:]')

  log "main" "Starting SN85 Ralph loop"
  log "main" "Task type: ${task_type}, Mode: ${LOOP_MODE}"
  log "main" "Fast interval: ${FAST_INTERVAL}s, Medium every: ${MEDIUM_EVERY_N_FAST} fast loops"

  # Single loop mode
  if [[ "$LOOP_MODE" == "fast" ]]; then
    run_fast_loop; exit 0
  elif [[ "$LOOP_MODE" == "medium" ]]; then
    run_medium_loop; exit 0
  elif [[ "$LOOP_MODE" == "slow" ]]; then
    run_slow_loop "manual"; exit 0
  fi

  # Main orchestration loop
  local fast_iteration=0
  local plateau_reached=false

  while [[ "$plateau_reached" == false ]]; do
    ((fast_iteration++))
    log "main" "=== Iteration ${fast_iteration} ==="

    # ── Fast loop ──────────────────────────────────────────────────────────
    fast_result=$(run_fast_loop)
    fast_exit=$?

    if [[ $fast_exit -ne 0 ]]; then
      log "main" "Fast loop failed. Retrying in ${FAST_INTERVAL}s."
      sleep "$FAST_INTERVAL"
      continue
    fi

    # Check for immediate anomaly -- run medium loop now if so
    local run_medium=false
    local trigger_reason="scheduled"

    if echo "$fast_result" | grep -q "FAST_ANOMALY"; then
      log "main" "Anomaly detected -- triggering medium loop immediately"
      run_medium=true
      trigger_reason="anomaly:$(echo "$fast_result" | grep -o 'FAST_ANOMALY:[^[:space:]]*')"
    elif (( fast_iteration % MEDIUM_EVERY_N_FAST == 0 )); then
      log "main" "Scheduled medium loop (iteration $fast_iteration)"
      run_medium=true
    fi

    # ── Medium loop ────────────────────────────────────────────────────────
    if [[ "$run_medium" == true ]]; then
      medium_result=$(run_medium_loop)
      medium_exit=$?

      if [[ $medium_exit -ne 0 ]]; then
        log "main" "Medium loop error. Continuing."
      elif echo "$medium_result" | grep -q "PLATEAU_REACHED"; then
        log "main" "Plateau reached. Entering monitoring mode."
        plateau_reached=true
      elif echo "$medium_result" | grep -qE "MEDIUM_QUALITY_LOW|MEDIUM_CONFIG_ERROR|MEDIUM_CONTENT_LENGTH"; then
        # ── Slow loop ──────────────────────────────────────────────────────
        log "main" "Medium signals action needed. Running slow loop."
        run_slow_loop "${medium_result##*:}"
        slow_exit=$?

        # Brief pause after deployment to let EWMA settle
        if grep -q "DEPLOYING" "$PROGRESS_FILE" 2>/dev/null; then
          log "main" "Deployment recorded. Pausing 1 fast loop interval."
          sleep "$FAST_INTERVAL"
        fi
      fi
    fi

    if [[ "$plateau_reached" == true ]]; then
      break
    fi

    if [[ "$ONCE" == true ]]; then
      log "main" "--once flag set. Exiting."
      exit 0
    fi

    log "main" "Sleeping ${FAST_INTERVAL}s..."
    sleep "$FAST_INTERVAL"
  done

  # ── Plateau monitoring mode ────────────────────────────────────────────────
  log "main" "Entering plateau monitoring mode."
  echo ""
  echo "Plateau reached. Fast loop only. Watching for:"
  echo "  - scoring_function.py formula changes"
  echo "  - New content length support (20s+)"
  echo "  - VideoSchedulerConfig weight changes"
  echo ""

  while true; do
    fast_result=$(run_fast_loop)
    if echo "$fast_result" | grep -q "FAST_ANOMALY\|SCORING_CHANGED"; then
      log "main" "Change detected in monitoring mode. Re-run sn85_ralph.sh to restart."
      exit 0
    fi
    sleep "$FAST_INTERVAL"
  done
}

main "$@"
