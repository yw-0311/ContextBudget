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

PROJECT_DIR="$(pwd)"

export SWANLAB_API_KEY="jPvsjubltwXfluQst9wVs"



# Setup custom metrics (automatically detects metrics from reward function)
echo "Setting up custom metrics for automatic detection..."
bash "../scripts/setup_custom_metrics.sh"

BASE_MODEL="/home/wy517954/model/Qwen/Qwen3-30B-A3B-Instruct-2507"

DATA_DIR="/data/oss_bucket_0/shiyi/data/search_r1"

CONFIG_PATH="$PROJECT_DIR/submit/experiments/search_r1/configs"
TOOL_CONFIG="$CONFIG_PATH/tools/search_tool_config.yaml"
REWARD_FUNCTION_PATH="$PROJECT_DIR/submit/experiments/search_r1/ferret/reward_score"

PROJECT_NAME="DEBUG_EVA_16K"
EXPERIMENT_NAME="BRANCH_30B_EM_RW_EVA"

export TENSORBOARD_DIR="/data/oss_bucket_0/shiyi/tensorboard_dir/${PROJECT_NAME}/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH/train" \
    --config-name="1_eva_16k" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    +trainer.validation_data_dir="$TENSORBOARD_DIR/eva_rollout_not_skip/" \
    trainer.log_val_generations=0 \
    custom_reward_function.path="$REWARD_FUNCTION_PATH/search_r1_format.py" \
    custom_reward_function.name=compute_score_em \
    +custom_reward_function.reward_kwargs.structure_format_score=0 \
    +custom_reward_function.reward_kwargs.final_format_score=0 \
    +custom_reward_function.reward_kwargs.retrieval_score=0 \
    +custom_reward_function.reward_kwargs.format_score=0 \
    +custom_reward_function.reward_kwargs.score=1.0 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    critic.model.path=$BASE_MODEL \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    data.train_files="$DATA_DIR/train_elastic.parquet" \
    data.val_files="$DATA_DIR/sampled_output_500_elastic.parquet"  \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
