set -x

PROJECT_DIR="$(pwd)"

# Setup custom metrics (automatically detects metrics from reward function)
echo "Setting up custom metrics for automatic detection..."
bash "../scripts/setup_custom_metrics.sh"


DATA_DIR="/data/oss_bucket_0/shiyi/data/train_data/nq_hotpotqa_train_multi_1"
BASE_MODEL="/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct"
# tensorboard event and rollout jsonl
export TENSORBOARD_DIR="/data/oss_bucket_0/shiyi/tensorboard_dir/${PROJECT_NAME}/${EXPERIMENT_NAME}"
# checkpoints
SAVE_DIR="/data/oss_bucket_0/shiyi/experiments/${EXPERIMENT_NAME}/"


CONFIG_PATH="$PROJECT_DIR/train/configs"
TOOL_CONFIG="$CONFIG_PATH/tools/search_tool_base_config.yaml"
REWARD_FUNCTION_PATH="$PROJECT_DIR/train/reward_score/"

PROJECT_NAME="DEBUG"
EXPERIMENT_NAME="DEBUG"


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH/train" \
    --config-name="1_grpo" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH/search_r1_format.py" \
    custom_reward_function.name=compute_score_em \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    +actor_rollout_ref.rollout.dynamic_max_model_len_list=[] \
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
    data.val_files="$DATA_DIR/test/sampled_output_500.parquet"  \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.rollout.multi_turn.enable_budget=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"