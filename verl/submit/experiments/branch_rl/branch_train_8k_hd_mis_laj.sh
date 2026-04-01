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
DATA_DIR="/data/oss_bucket_0/shiyi/data/search_r1"

CONFIG_PATH="$PROJECT_DIR/submit/experiments/search_r1/configs"
TOOL_CONFIG="$CONFIG_PATH/tools/search_tool_config.yaml"
REWARD_FUNCTION_PATH="$PROJECT_DIR/submit/experiments/search_r1/ferret/reward_score"

# BASE_MODEL="/data/oss_bucket_0/shiyi/model/Qwen3-30B-A3B-Instruct-2507"
# BASE_MODEL="dfs://na63dfssearch3--cn-zhangjiakou/nebula_model/nebula_internal/Qwen__Qwen3-30B-A3B/main_ad44e777bcd18fa416d9da3bd8f70d33ebb85d39_na63_d416045d5b40/ckpt_base"
# BASE_MODEL="/data/oss_bucket_0/shiyi/model/Qwen2.5-3B"

PROJECT_NAME="BRANCH_V2"
EXPERIMENT_NAME="BRANCH_30B_RL_LAJ_RW_BS128_8K_HD_MIS"

export TENSORBOARD_DIR="/data/oss_bucket_0/shiyi/tensorboard_dir/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# SAVE_DIR="/data/oss_bucket_0/Users/heran/experiments/${EXPERIMENT_NAME}/"
SAVE_DIR="/data/oss_bucket_0/shiyi/experiments/${PROJECT_NAME}/${EXPERIMENT_NAME}/"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH/train" \
    --config-name="8_grpo_8k" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH/search_r1_format_llm_judge.py" \
    custom_reward_function.name=compute_score_em_with_judge \
    algorithm.rollout_is=True \
    algorithm.rollout_is_threshold=1.001 \
    algorithm.rollout_is_threshold_lower=0.999 \
    algorithm.rollout_is_level=geometric \
    algorithm.rollout_is_mode=mask \
    algorithm.rollout_is_veto_threshold=1e-4 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
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
    data.train_files="$DATA_DIR/asearcher_filtered_elastic.parquet" \
    data.val_files="$DATA_DIR/sampled_output_500_elastic_v2.parquet"  \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"