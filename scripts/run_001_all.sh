#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/cephfs/wenhaozhang/projects/agentic-credit-assignment}"
SLIME_ROOT="${SLIME_ROOT:-/root/ac-slime-src/slime}"
ENV_ROOT="${ENV_ROOT:-/root/micromamba/envs/ac-slime}"
MODEL_PATH="${MODEL_PATH:-/cephfs/wenhaozhang/models/Qwen3.5-2B}"
DATA_ROOT="${DATA_ROOT:-/cephfs/wenhaozhang/datasets/agentic-credit-assignment/alfworld/downloads}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/cephfs/wenhaozhang/runs/agentic-credit-assignment/001/${RUN_ID}}"
HOT_ROOT="${HOT_ROOT:-/localssd/agentic-credit-assignment/001/${RUN_ID}}"
CACHE_ROOT="${CACHE_ROOT:-/localssd/agentic-credit-assignment/cache}"
UPDATES="${UPDATES:-50}" TASKS_PER_UPDATE="${TASKS_PER_UPDATE:-4}"
GROUP_SIZE="${GROUP_SIZE:-16}" NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_TP_SIZE="${ACTOR_TP_SIZE:-2}"
ACTOR_CP_SIZE="${ACTOR_CP_SIZE:-4}"
TRAIN_ATTENTION_BACKEND="${TRAIN_ATTENTION_BACKEND:-auto}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
ARCHIVE_INTERVAL="${ARCHIVE_INTERVAL:-10}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
RESUME="${RESUME:-1}"
ARMS="${ARMS:-base grpo bottleneckflow}"
RUN_EVAL="${RUN_EVAL:-1}"
mkdir -p "${RUN_ROOT}/logs" "${HOT_ROOT}/manifests" "${HOT_ROOT}/checkpoints" "${CACHE_ROOT}"
STATUS_LOG="${RUN_ROOT}/logs/status.log"
CURRENT_STAGE="initializing"
FINAL_STATUS="FAILED"
: > "${STATUS_LOG}"
status() {
  printf '%s status=%s stage=%s %s\n' \
    "$(date --iso-8601=seconds)" "$1" "${CURRENT_STAGE}" "${2:-}" | tee -a "${STATUS_LOG}"
}
status RUNNING "pid=$$ run_root=${RUN_ROOT} hot_root=${HOT_ROOT}"
# A Megatron checkpoint can only resume under the same training schedule and
# batch semantics. Fail before allocating GPUs if a restart accidentally uses
# a different command. ARMS is intentionally excluded so a user may resume a
# single unfinished arm; SAVE_INTERVAL may also be changed safely.
RUN_SIGNATURE="schema=2;updates=${UPDATES};tasks=${TASKS_PER_UPDATE};group=${GROUP_SIZE};gpus=${NUM_GPUS};tp=${ACTOR_TP_SIZE};cp=${ACTOR_CP_SIZE};attention=${TRAIN_ATTENTION_BACKEND};eval=${RUN_EVAL};max_steps=${MAX_STEPS:-50};tokens_per_turn=${TOKENS_PER_TURN:-192};loss=per_token;log_probs_chunk=${LOG_PROBS_CHUNK_SIZE};allocator=expandable_segments;model=${MODEL_PATH}"
SIGNATURE_PATH="${RUN_ROOT}/run_signature.txt"
if [[ -f "${SIGNATURE_PATH}" ]]; then
  SAVED_SIGNATURE="$(<"${SIGNATURE_PATH}")"
  if [[ "${SAVED_SIGNATURE}" != "${RUN_SIGNATURE}" ]]; then
    echo "Resume configuration differs from ${SIGNATURE_PATH}" >&2
    echo "saved:  ${SAVED_SIGNATURE}" >&2
    echo "current: ${RUN_SIGNATURE}" >&2
    exit 2
  fi
else
  printf '%s\n' "${RUN_SIGNATURE}" > "${SIGNATURE_PATH}"
fi
export PATH="${ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${SLIME_ROOT}:/root/Megatron-LM:${PYTHONPATH:-}"
export HF_HOME="${CACHE_ROOT}/huggingface" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions" CUDA_DEVICE_MAX_CONNECTIONS=1
# NVLS is Hopper-specific and unavailable on the A100 experiment nodes. Keep
# regular NCCL collectives while disabling the unsupported multicast path.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/flashinfer"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export CC="${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}" NVCC_CCBIN="${CXX}"
# FlashInfer JIT links sampling kernels on their first use.  Conda's compiler
# uses a sysroot and otherwise misses the NVIDIA driver library from the host.
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu ${LDFLAGS:-}"
# JIT subprocesses run while slime's memory-saver preload hook is active. Put
# the Conda GCC 13 runtime first so that hook can resolve GLIBCXX_3.4.29.
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${ENV_ROOT}/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
# Kubernetes counts threads as PIDs. Ray + 8 SGLang engines + 8 training
# ranks can otherwise exhaust the pod's pids.max during first-use Triton JIT.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}" MAX_JOBS="${MAX_JOBS:-1}"
# Long agent trajectories make the log-prob workspace allocation pattern
# uneven. Expandable segments let PyTorch reuse reserved memory instead of
# failing on a small contiguous allocation near the GPU memory limit.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
python "${REPO_ROOT}/scripts/prepare_001_alfworld.py" --data-root "${DATA_ROOT}" --output-dir "${HOT_ROOT}/manifests" \
  --max-steps "${MAX_STEPS:-50}" --tokens-per-turn "${TOKENS_PER_TURN:-192}"
GPU_MONITOR_PID=""
PID_MONITOR_PID=""
stop_ray() {
  ray stop --force >/dev/null 2>&1 || true
  # `ray stop --force` returns before all children have necessarily exited.
  # Do not let SGLang/actor threads from one arm leak into the next arm.
  for _ in $(seq 1 60); do
    if ! pgrep -f 'raylet|gcs_server|dashboard.py' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Ray processes did not exit within 60 seconds" >&2
  return 1
}
start_ray() {
  stop_ray
  ray start --head --node-ip-address 127.0.0.1 --num-cpus "${RAY_CPUS:-32}" \
    --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=127.0.0.1 >/dev/null
}
cleanup() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${PID_MONITOR_PID}" ]]; then
    kill "${PID_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${PID_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
  stop_ray || true
}
on_exit() {
  local exit_code=$?
  cleanup
  if [[ "${FINAL_STATUS}" == "COMPLETED" && "${exit_code}" == 0 ]]; then
    status COMPLETED "exit_code=0"
  else
    status FAILED "exit_code=${exit_code}; inspect ${RUN_ROOT}/logs/launcher.log and the current arm log"
  fi
}
trap on_exit EXIT
stop_ray
# Keep objective utilization evidence for every stage. GPU utilization is
# sampled rather than inferred from a single nvidia-smi snapshot.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv -l 5 > "${RUN_ROOT}/logs/gpu_usage.csv" 2> "${RUN_ROOT}/logs/gpu_monitor.err" &
GPU_MONITOR_PID=$!
# Record the actual pod PID/thread pressure so future failures are diagnosable.
(
  while true; do
    printf '%s,%s,%s\n' "$(date --iso-8601=seconds)" \
      "$(cat /sys/fs/cgroup/pids.current 2>/dev/null || echo NA)" \
      "$(awk '/^max / {print $2}' /sys/fs/cgroup/pids.events 2>/dev/null || echo NA)"
    sleep 5
  done
) > "${RUN_ROOT}/logs/pid_usage.csv" &
PID_MONITOR_PID=$!
source "${REPO_ROOT}/scripts/models/qwen3.5-2B.sh"

run_arm() {
  local method="$1" rollouts="$2" arm_root="${RUN_ROOT}/$1"
  CURRENT_STAGE="${method}:prepare"
  status RUNNING "preparing arm=${method}"
  mkdir -p "${arm_root}/audit" "${arm_root}/checkpoints" "${arm_root}/debug"
  local archive_checkpoint_root="${arm_root}/checkpoints"
  local checkpoint_root="${HOT_ROOT}/checkpoints/${method}"
  mkdir -p "${checkpoint_root}/rollout" "${archive_checkpoint_root}/rollout" \
    "${archive_checkpoint_root}/partial"
  local load_path="${MODEL_PATH}"
  local resume_step=""
  local num_rollout_arg="${rollouts}"
  local eval_only_recovery=0
  local checkpoint_args=()
  local resume_root=""
  local tracker=""
  if [[ "${RESUME}" == 1 ]]; then
    local root candidate
    for root in "${checkpoint_root}" "${archive_checkpoint_root}"; do
      [[ -d "${root}" ]] || continue
      while IFS= read -r candidate; do
        if [[ -f "${root}/iter_$(printf '%07d' "${candidate}")/.metadata" ]] && \
           [[ -f "${root}/rollout/global_dataset_state_dict_${candidate}.pt" ]] && \
           { [[ -z "${resume_step}" ]] || (( candidate > resume_step )); }; then
          resume_step="${candidate}"
          resume_root="${root}"
          break
        fi
      done < <(find "${root}" -maxdepth 1 -type d -name 'iter_*' -printf '%f\n' \
        | sed 's/^iter_//' | awk '{print $1 + 0}' | sort -nr)
    done
  fi
  if [[ "${RESUME}" == 1 && -n "${resume_step}" ]]; then
    tracker="${resume_root}/latest_checkpointed_iteration.txt"
    # Megatron writes its tracker before slime finishes the ALFWorld sampling
    # cursor. Power loss in that window leaves a model-only checkpoint. Scan
    # backwards and select the newest *jointly* durable boundary instead of
    # trusting the tracker blindly.
    if [[ -f "${tracker}" ]]; then
      local tracked_step
      tracked_step="$(tr -d '[:space:]' < "${tracker}")"
      if [[ "${tracked_step}" =~ ^[0-9]+$ ]] && [[ "${resume_step}" != "${tracked_step}" ]]; then
        echo "${method}: latest checkpoint ${tracked_step} is partial; falling back to ${resume_step}"
      fi
    fi
    # The final checkpoint is written before final evaluation. If DONE is
    # absent, either the evaluation already landed (only the marker was lost),
    # or we load the final weights in slime's eval-only mode. Never overwrite a
    # valid final checkpoint merely to recover the evaluation.
    if (( resume_step >= rollouts - 1 )); then
      if [[ "${RUN_EVAL}" == 0 || \
            ( -f "${arm_root}/eval_valid_seen.json" && -f "${arm_root}/eval_valid_unseen.json" ) ]]; then
        echo "${method}: final checkpoint and required outputs exist; restoring missing DONE marker"
        touch "${arm_root}/DONE"
        return 0
      fi
      eval_only_recovery=1
      num_rollout_arg=0
    fi
    if [[ -n "${resume_step}" ]]; then
      load_path="${resume_root}"
      checkpoint_args=(--ckpt-step "${resume_step}")
      printf '{"time":"%s","arm":"%s","checkpoint":%s,"next_rollout":%s,"eval_only":%s}\n' \
        "$(date --iso-8601=seconds)" "${method}" "${resume_step}" "$((resume_step + 1))" "${eval_only_recovery}" \
        >> "${arm_root}/resume_events.jsonl"
      if [[ "${eval_only_recovery}" == 1 ]]; then
        echo "${method}: final checkpoint ${resume_step} exists; recovering final evaluation only"
      else
        echo "${method}: resuming from rollout ${resume_step}; next rollout is $((resume_step + 1))"
      fi
    fi
  fi

  archive_checkpoint_step() {
    local step="$1" iter_name source_iter destination_iter staging_root
    iter_name="iter_$(printf '%07d' "${step}")"
    source_iter="${checkpoint_root}/${iter_name}"
    destination_iter="${archive_checkpoint_root}/${iter_name}"
    [[ -f "${source_iter}/.metadata" ]] || return 1
    [[ -f "${checkpoint_root}/rollout/global_dataset_state_dict_${step}.pt" ]] || return 1
    if [[ -f "${destination_iter}/.metadata" && \
          -f "${archive_checkpoint_root}/rollout/global_dataset_state_dict_${step}.pt" ]]; then
      return 0
    fi
    if [[ -e "${destination_iter}" ]]; then
      mv "${destination_iter}" \
        "${archive_checkpoint_root}/partial/${iter_name}_$(date +%Y%m%d_%H%M%S)"
    fi
    staging_root="${archive_checkpoint_root}/.staging_${iter_name}_$$"
    rm -rf "${staging_root}"
    mkdir -p "${staging_root}/rollout"
    cp -a "${source_iter}" "${staging_root}/${iter_name}"
    cp -a "${checkpoint_root}/rollout/global_dataset_state_dict_${step}.pt" \
      "${staging_root}/rollout/"
    [[ -f "${staging_root}/${iter_name}/.metadata" ]] || return 1
    mv "${staging_root}/${iter_name}" "${destination_iter}"
    mv "${staging_root}/rollout/global_dataset_state_dict_${step}.pt" \
      "${archive_checkpoint_root}/rollout/global_dataset_state_dict_${step}.pt"
    rmdir "${staging_root}/rollout" "${staging_root}"
    printf '%s\n' "${step}" > "${archive_checkpoint_root}/.latest_checkpointed_iteration.tmp"
    mv "${archive_checkpoint_root}/.latest_checkpointed_iteration.tmp" \
      "${archive_checkpoint_root}/latest_checkpointed_iteration.txt"
    status RUNNING "archived arm=${method} checkpoint=${step} local_ssd->cephfs"
  }

  checkpoint_archiver() {
    local last_archived="" step
    while true; do
      while IFS= read -r step; do
        (( (step + 1) % ARCHIVE_INTERVAL == 0 )) || continue
        [[ "${step}" == "${last_archived}" ]] && continue
        if archive_checkpoint_step "${step}"; then
          last_archived="${step}"
        else
          status WARNING "checkpoint ${step} is not jointly complete yet; archive will retry"
        fi
      done < <(find "${checkpoint_root}" -maxdepth 1 -type f -name latest_checkpointed_iteration.txt \
        -exec sh -c 'tr -d "[:space:]" < "$1"' _ {} \; 2>/dev/null)
      sleep 15
    done
  }
  local runtime_config="${HOT_ROOT}/runtime_${method}.json"
  python "${REPO_ROOT}/scripts/make_001_runtime_config.py" \
    --base "${REPO_ROOT}/configs/001_slime.yaml" --output "${runtime_config}" \
    --method "${method}" --audit-dir "${arm_root}/audit" \
    --max-steps "${MAX_STEPS:-50}" --tokens-per-turn "${TOKENS_PER_TURN:-192}"
  local debug_flag=()
  [[ "${method}" == base && "${eval_only_recovery}" == 0 ]] && debug_flag=(--debug-rollout-only)
  local replay_args=()
  if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" && "${eval_only_recovery}" == 0 ]]; then
    replay_args=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}" --debug-train-only)
  fi
  local eval_args=()
  if [[ "${RUN_EVAL}" == 1 ]]; then
    eval_args=(
      --eval-prompt-data valid_seen "${HOT_ROOT}/manifests/valid_seen.jsonl" valid_unseen "${HOT_ROOT}/manifests/valid_unseen.jsonl"
      --eval-interval "${rollouts}" --skip-eval-before-train --n-samples-per-eval-prompt 1
      --eval-temperature 0 --eval-top-p 1 --eval-max-response-len 32768
    )
  fi
  export ACA_MAX_STEPS="${MAX_STEPS:-50}" ACA_TOKENS_PER_TURN="${TOKENS_PER_TURN:-192}"
  local attempt_log="${RUN_ROOT}/logs/${method}.log"
  printf '\n===== attempt %s arm=%s load=%s checkpoint=%s =====\n' \
    "$(date --iso-8601=seconds)" "${method}" "${load_path}" "${resume_step:-base}" >> "${attempt_log}"
  CURRENT_STAGE="${method}:running"
  status RUNNING "starting arm=${method} load=${load_path} checkpoint=${resume_step:-base} save=${checkpoint_root}"
  checkpoint_archiver &
  local archiver_pid=$!
  # Submit from /tmp: the repository intentionally contains CephFS symlinks to
  # models/datasets/runs, which Ray must not scan as a working-dir package.
  pushd /tmp >/dev/null
  set +e
  RAY_ADDRESS=auto python "${SLIME_ROOT}/train.py" "${MODEL_ARGS[@]}" \
    --actor-num-nodes 1 --actor-num-gpus-per-node "${NUM_GPUS}" --rollout-num-gpus "${NUM_GPUS}" --colocate \
    --hf-checkpoint "${MODEL_PATH}" --load "${load_path}" "${checkpoint_args[@]}" \
    --save "${checkpoint_root}" --save-interval "${SAVE_INTERVAL}" \
    --prompt-data "${HOT_ROOT}/manifests/train.jsonl" --input-key prompt \
    --custom-generate-function-path agentic_credit_assignment.slime.rollout.generate \
    --rollout-sample-filter-path agentic_credit_assignment.slime.credit.process_groups \
    --custom-advantage-function-path agentic_credit_assignment.slime.credit.token_advantages \
    --custom-eval-rollout-log-function-path agentic_credit_assignment.slime.logging.log_eval \
    --custom-config-path "${runtime_config}" \
    --num-rollout "${num_rollout_arg}" --rollout-batch-size "${TASKS_PER_UPDATE}" --n-samples-per-prompt "${GROUP_SIZE}" --rollout-shuffle \
    --rollout-max-context-len 32768 --rollout-max-response-len 32768 --rollout-temperature 0.7 --rollout-top-p 1 \
    --router-policy consistent_hashing \
    "${eval_args[@]}" \
    --global-batch-size "$((TASKS_PER_UPDATE * GROUP_SIZE))" \
    --advantage-estimator grpo --kl-coef 0 --kl-loss-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --tensor-model-parallel-size "${ACTOR_TP_SIZE}" --sequence-parallel \
    --pipeline-model-parallel-size 1 --context-parallel-size "${ACTOR_CP_SIZE}" --expert-model-parallel-size 1 \
    --calculate-per-token-loss \
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" \
    --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu 8192 --recompute-granularity selective \
    --attention-dropout 0 --hidden-dropout 0 --attention-backend "${TRAIN_ATTENTION_BACKEND}" \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.55 --sglang-max-running-requests 64 \
    --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
    --use-tensorboard --tb-project-name aca001 --tb-experiment-name "${RUN_ID}_${method}" \
    --save-debug-rollout-data "${arm_root}/debug/rollout_{rollout_id}.pt" \
    "${debug_flag[@]}" "${replay_args[@]}" 2>&1 | tee -a "${attempt_log}"
  local train_exit=${PIPESTATUS[0]}
  set -e
  popd >/dev/null
  kill "${archiver_pid}" >/dev/null 2>&1 || true
  wait "${archiver_pid}" >/dev/null 2>&1 || true
  if (( train_exit != 0 )); then
    status FAILED "arm=${method} train_exit=${train_exit} log=${attempt_log}"
    return "${train_exit}"
  fi
  if [[ "${method}" != "base" ]]; then
    local final_step
    final_step="$(tr -d '[:space:]' < "${checkpoint_root}/latest_checkpointed_iteration.txt")"
    archive_checkpoint_step "${final_step}"
  fi
  if [[ -d "/tmp/tensorboard_log/aca001/${RUN_ID}_${method}" ]]; then
    mkdir -p "${arm_root}/tensorboard"
    cp -a "/tmp/tensorboard_log/aca001/${RUN_ID}_${method}/." "${arm_root}/tensorboard/"
  fi
  touch "${arm_root}/DONE"
  status RUNNING "arm=${method} completed"
}
for arm in ${ARMS}; do
  if [[ "${RESUME}" == 1 && -f "${RUN_ROOT}/${arm}/DONE" ]]; then
    echo "${arm}: DONE exists, skipping"
    continue
  fi
  # Each arm gets a fresh Ray cluster. This is deliberate: serving actors own
  # thousands of threads and Ray does not guarantee they are gone when one
  # train.py client exits.
  start_ray
  case "${arm}" in
    base) run_arm base 1 ;;
    grpo) run_arm grpo "${UPDATES}" ;;
    bottleneckflow) run_arm bottleneckflow "${UPDATES}" ;;
    *) echo "unknown arm: ${arm}" >&2; exit 2 ;;
  esac
  stop_ray
done
if [[ "${RUN_EVAL}" == 1 ]]; then
  python "${REPO_ROOT}/scripts/aggregate_001.py" --run-root "${RUN_ROOT}"
fi
touch "${RUN_ROOT}/EXPERIMENT_DONE"
FINAL_STATUS="COMPLETED"
echo "run_root=${RUN_ROOT}"
