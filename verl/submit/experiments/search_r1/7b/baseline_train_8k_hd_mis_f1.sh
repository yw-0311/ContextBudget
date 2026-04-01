set -x

ulimit -n 65535

ACTOR_MODEL_PATH=""
PY_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --actor_model_path=*)
      ACTOR_MODEL_PATH="${arg#*=}"
      ;;
    *)
      PY_ARGS+=("$arg")
      ;;
  esac
done
if [[ -n "$ACTOR_MODEL_PATH" ]]; then
  ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH%/}"
  BASE_MODEL="$ACTOR_MODEL_PATH"
  echo "✅ override BASE_MODEL: $BASE_MODEL"
fi


# verl_nebula/
PROJECT_DIR="$(pwd)"

# export SWANLAB_API_KEY="jPvsjubltwXfluQst9wVs"

# Setup custom metrics (automatically detects metrics from reward function)
echo "Setting up custom metrics for automatic detection..."
bash "../scripts/setup_custom_metrics.sh"
DATA_DIR="/data/oss_bucket_0/shiyi/data/train_data/nq_hotpotqa_train_multi_1"

CONFIG_PATH="$PROJECT_DIR/submit/experiments/search_r1/configs"
TOOL_CONFIG="$CONFIG_PATH/tools/search_tool_config.yaml"
REWARD_FUNCTION_PATH="$PROJECT_DIR/submit/experiments/search_r1/ferret/reward_score"

PROJECT_NAME="BASELINE_V3"
EXPERIMENT_NAME="BASELINE_7B_RL_F1_RW_BS128_8K_MIS"

export TENSORBOARD_DIR="/data/oss_bucket_0/shiyi/tensorboard_dir/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# SAVE_DIR="/data/oss_bucket_0/Users/heran/experiments/${EXPERIMENT_NAME}/"
SAVE_DIR="/data/oss_bucket_0/shiyi/experiments/${PROJECT_NAME}/${EXPERIMENT_NAME}/"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH/train" \
    --config-name="8_grpo_8k" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH/search_r1_format.py" \
    custom_reward_function.name=compute_score_multi_answer \
    algorithm.rollout_is=True \
    algorithm.rollout_is_threshold=1.001 \
    algorithm.rollout_is_threshold_lower=0.999 \
    algorithm.rollout_is_level=geometric \
    algorithm.rollout_is_mode=mask \
    algorithm.rollout_is_veto_threshold=1e-4 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.dynamic_max_model_len_list=[8192] \
    +trainer.validation_data_dir="$TENSORBOARD_DIR/eva_rollout/" \
    +trainer.rollout_data_dir="$TENSORBOARD_DIR/rollout_with_mask/" \
    +custom_reward_function.reward_kwargs.structure_format_score=0.0 \
    +custom_reward_function.reward_kwargs.final_format_score=0.0 \
    +custom_reward_function.reward_kwargs.retrieval_score=0 \
    +custom_reward_function.reward_kwargs.format_score=0 \
    +custom_reward_function.reward_kwargs.score=1.0 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    critic.model.path=$BASE_MODEL \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$SAVE_DIR \
    data.train_files="$DATA_DIR/search_r1_processed/train.parquet" \
    data.val_files="$DATA_DIR/search_r1_processed/train.parquet"  \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"