#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/cephfs/wenhaozhang/projects/agentic-credit-assignment
VERL_ROOT="$PROJECT_ROOT/third_party/verl-agent"
RUN_DIR=/cephfs/wenhaozhang/runs/agentic-credit-assignment/exp_001
PYTHON=/root/micromamba/envs/ac-verl-agent/bin/python
MODEL=/cephfs/wenhaozhang/models/Qwen2.5-1.5B-Instruct

TRAIN_SIZE=2
GROUP_SIZE=8
VAL_SIZE=14
MAX_STEPS=30
TRAINING_STEPS=1
SEED=0
LR=1e-6

mkdir -p "$RUN_DIR"/{logs,checkpoints,metrics,samples,artifacts,data}
exec > >(tee "$RUN_DIR/logs/train.log") 2>&1

# A pilot is a fresh, inspectable miniature run. Reuse the fixed artifact
# paths, but never mix traces from an older schema/attempt into this run.
: > "$RUN_DIR/metrics/train_metrics.jsonl"
: > "$RUN_DIR/metrics/gpu_memory.csv"
: > "$RUN_DIR/metrics/eval_metrics.json"
: > "$RUN_DIR/samples/rollouts.jsonl"
: > "$RUN_DIR/samples/graphs.jsonl"

finish() {
  exit_code=$?
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  "$PYTHON" "$PROJECT_ROOT/scripts/experiment_001_metadata.py" finish --run-dir "$RUN_DIR" --exit-code "$exit_code" || true
  "$PYTHON" "$PROJECT_ROOT/scripts/experiment_001_metadata.py" analyze --run-dir "$RUN_DIR" || true
  case "$exit_code" in
    0) status=COMPLETED ;;
    130|143) status=STOPPED ;;
    *) status=FAILED ;;
  esac
  printf '[exp_001] exit_code=%s status=%s\n' "$exit_code" "$status"
  exit "$exit_code"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap finish EXIT

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing environment Python: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Missing model: $MODEL" >&2
  exit 2
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "Pilot requires exactly 8 visible GPUs; found $GPU_COUNT" >&2
  exit 2
fi

"$PYTHON" "$PROJECT_ROOT/scripts/experiment_001_metadata.py" init \
  --run-dir "$RUN_DIR" --mode pilot --model "$MODEL" \
  --project-root "$PROJECT_ROOT" --verl-root "$VERL_ROOT" \
  --train-size "$TRAIN_SIZE" --group-size "$GROUP_SIZE" --val-size "$VAL_SIZE" \
  --max-steps "$MAX_STEPS" --training-steps "$TRAINING_STEPS" \
  --seed "$SEED" --learning-rate "$LR"

"$PYTHON" "$PROJECT_ROOT/scripts/prepare_001_placeholders.py" \
  --output-dir "$RUN_DIR/data" --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"

"$PYTHON" "$PROJECT_ROOT/scripts/monitor_gpu_memory.py" \
  --output "$RUN_DIR/metrics/gpu_memory.csv" --interval 2 &
GPU_MONITOR_PID=$!

export PYTHONPATH="$PROJECT_ROOT/src:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HYDRA_FULL_ERROR=1
export ALFWORLD_DATA=/cephfs/wenhaozhang/datasets/agentic-credit-assignment/alfworld/downloads
export MPLCONFIGDIR="$RUN_DIR/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

cd "$VERL_ROOT"
"$PYTHON" -m recipe.GraphGPO.main_graphgpo \
  algorithm.adv_estimator=bottleneckflow \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="$RUN_DIR/data/train.parquet" \
  data.val_files="$RUN_DIR/data/test.parquet" \
  data.train_batch_size="$TRAIN_SIZE" \
  data.val_batch_size="$VAL_SIZE" \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed="$SEED" \
  env.max_steps="$MAX_STEPS" \
  env.history_length=2 \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.logger='[console]' \
  trainer.project_name=agentic-credit-assignment \
  trainer.experiment_name=exp_001_pilot \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps="$TRAINING_STEPS" \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.rollout_data_dir=null \
  ray_init.num_cpus=24 \
  aca.enabled=True \
  aca.train_metrics_file="$RUN_DIR/metrics/train_metrics.jsonl" \
  aca.eval_metrics_file="$RUN_DIR/metrics/eval_metrics.json" \
  aca.rollout_file="$RUN_DIR/samples/rollouts.jsonl" \
  aca.graph_file="$RUN_DIR/samples/graphs.jsonl"
