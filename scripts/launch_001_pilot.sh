#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/cephfs/wenhaozhang/projects/agentic-credit-assignment
RUN_DIR=/cephfs/wenhaozhang/runs/agentic-credit-assignment/exp_001
mkdir -p "$RUN_DIR/logs"

setsid nohup "$PROJECT_ROOT/scripts/run_001_pilot.sh" </dev/null \
  >"$RUN_DIR/logs/launcher.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$RUN_DIR/launcher.pid"
printf 'started exp_001 pilot pid=%s\nstatus: tail -f %s/logs/train.log\n' "$pid" "$RUN_DIR"
