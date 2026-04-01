#!/bin/bash
set -e
set -x



# ========== 参数解析 ==========
SCRIPT_PATH=""
VERL_NODE_SIZE=""
WORLD_SIZE=""
EXPERIMENT_NAME=""
ACTOR_MODEL_PATH=""
JUDGE_MODEL_PATH=""

export RAY_worker_heartbeat_timeout_ms=60000000000000
export RAY_task_lease_timeout_ms=60000000000000

export RAY_worker_register_timeout_seconds=60000000000000  # Worker注册超时
export RAY_worker_starting_timeout_seconds=60000000000000   # Worker启动超时
export RAY_gcs_connect_timeout_seconds=60000000000000       # GCS连接超时
export RAY_gcs_rpc_server_reconnect_timeout_s=60000000000000

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --script_path=*) SCRIPT_PATH="${1#*=}" ;;
        --verl_node_size=*) VERL_NODE_SIZE="${1#*=}" ;;
        --world_size=*) WORLD_SIZE="${1#*=}" ;;
        --experiment_name=*) EXPERIMENT_NAME="${1#*=}" ;;
        --actor_model_path=*) ACTOR_MODEL_PATH="${1#*=}" ;;
        --judge_model_name_path=*) JUDGE_MODEL_PATH="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "❌ 必须指定 --experiment_name"
    exit 1
fi

SAVE_NAME="$(basename "$SCRIPT_PATH" .sh)"

echo "SCRIPT_PATH = $SCRIPT_PATH"
echo "VERL_NODE_SIZE = $VERL_NODE_SIZE"
echo "WORLD_SIZE = $WORLD_SIZE"
echo "EXPERIMENT_NAME = $EXPERIMENT_NAME"
echo "SAVE_NAME = $SAVE_NAME"
echo "JUDGE_MODEL_PATH = $JUDGE_MODEL_PATH"
echo "SEARCH_MODEL_PATH = $SEARCH_MODEL_PATH"

# ========== 通用参数 ==========
RAY_NODE_COUNT="$VERL_NODE_SIZE"
NEED_VLLM=$((WORLD_SIZE != VERL_NODE_SIZE))

# 日志目录
LOG_DIR="/data/oss_bucket_0/shiyi/log/$EXPERIMENT_NAME/$SAVE_NAME"
mkdir -p "$LOG_DIR"
export LOG_DIR="$LOG_DIR"
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export SAVE_NAME="$SAVE_NAME"


# ========== vLLM 相关配置 ==========
if [[ $NEED_VLLM -eq 1 ]]; then
    NUM_VLLM_PER_NODE=4
    VLLM_BASE_PORT=8000
    IP_FILE="$LOG_DIR/vllm_ip.txt"
    export IP_FILE="$IP_FILE"
    EXPECTED_VLLM_COUNT=$(( (WORLD_SIZE - VERL_NODE_SIZE) * NUM_VLLM_PER_NODE ))
fi

# ========== vLLM 节点启动 ==========
if [[ $NEED_VLLM -eq 1 ]] && [ "$RANK" -ge "$VERL_NODE_SIZE" ]; then
    IP=$(hostname -I | awk '{print $1}')
    NODE_IP_FILE="$LOG_DIR/vllm_ip_${RANK}.txt"
    : > "$NODE_IP_FILE"

    for i in $(seq 0 $((NUM_VLLM_PER_NODE - 1))); do
        PORT=$((VLLM_BASE_PORT + i))
        LOG_PATH="$LOG_DIR/vllm_server_${IP}_$PORT.log"
        GPU1=$((i * 2))
        GPU2=$((i * 2 + 1))

        echo "🚀 启动 vLLM 实例 $IP:$PORT 使用 GPU: $GPU1,$GPU2"

        CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 python3 -m sglang.launch_server \
            --model-path "$JUDGE_MODEL_PATH" \
            --served-model-name "q" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tp 2 \
            > "$LOG_PATH" 2>&1 &

        # 等待端口 Ready
        for j in {1..1200}; do
            if timeout 1 bash -c "</dev/tcp/$IP/$PORT" &>/dev/null; then
                echo "✅ vLLM 启动成功: $IP:$PORT"
                break
            fi
            sleep 1
        done

        if ! timeout 1 bash -c "</dev/tcp/$IP/$PORT" &>/dev/null; then
            echo "❌ vLLM 启动失败: $IP:$PORT"
            exit 1
        fi

        echo "$IP:$PORT" >> "$NODE_IP_FILE"
    done

    # 保持 vLLM 后台
    while :; do sleep 60; done

# ========== Rank 0 启动，先启动 Ray Head 再等待 vLLM 完成 ==========
elif [ "$RANK" -eq 0 ]; then
    # 启动 Ray Head 节点（先启动 GCS 服务）
    echo "🚀 启动 Ray Head 节点"
    ray start --head --dashboard-host=0.0.0.0 --port=6380 --num-cpus=16

    # 只有在需要 vLLM 时才清理日志
    if [[ $NEED_VLLM -eq 1 ]]; then
        echo "📝 初始化日志目录 & 清理"
        rm -f "$LOG_DIR"/vllm_ip_*.txt "$IP_FILE"
        find "$LOG_DIR" -maxdepth 1 -type f -name 'vllm_server_*.log' -delete

        # 等待所有 vLLM 节点写完 IP 文件
        while true; do
            shopt -s nullglob
            TOTAL_IP_COUNT=0
            files=("$LOG_DIR"/vllm_ip_*.txt)
            shopt -u nullglob

            if [ ${#files[@]} -eq 0 ]; then
                echo "⏳ 等待中... 当前没有检测到任何 vLLM IP 文件"
            else
                for f in "${files[@]}"; do
                    count=$(wc -l < "$f")
                    TOTAL_IP_COUNT=$((TOTAL_IP_COUNT + count))
                done
                echo "⏳ 等待中... 当前已检测到 $TOTAL_IP_COUNT / $EXPECTED_VLLM_COUNT 个实例"
            fi

            if [[ "$TOTAL_IP_COUNT" -ge "$EXPECTED_VLLM_COUNT" ]]; then
                echo "✅ 已检测到 $TOTAL_IP_COUNT 个 vLLM 实例 IP，开始合并"
                break
            fi
            sleep 5
        done

        # 合并 IP 文件
        : > "$IP_FILE"
        for f in "$LOG_DIR"/vllm_ip_*.txt; do
            cat "$f" >> "$IP_FILE"
        done

        LINE_COUNT=$(wc -l < "$IP_FILE")
        echo "✅ 合并完成，总计 $LINE_COUNT 个 vLLM 实例"

        if [[ "$LINE_COUNT" -lt "$EXPECTED_VLLM_COUNT" ]]; then
            echo "❌ 合并后的 IP 数量不足，期待 $EXPECTED_VLLM_COUNT，实际 $LINE_COUNT"
            exit 1
        fi

        echo "✅ vLLM 环境准备就绪"
    else
        echo "✅ 无需 vLLM 环境"
    fi

    # 等待所有 Ray 节点注册完成
    until NODE_COUNT=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(len(ray.nodes()))" 2>/dev/null) && [ "$NODE_COUNT" -eq "$RAY_NODE_COUNT" ]; do
        echo "⏳ 等待 Ray 节点注册中，当前: $NODE_COUNT / $RAY_NODE_COUNT"
        sleep 10
    done

    echo "✅ Ray 环境准备就绪，启动训练脚本:$SCRIPT_PATH,--actor_model_path=$ACTOR_MODEL_PATH"
    # 启动训练脚本
    bash "$SCRIPT_PATH" --actor_model_path=$ACTOR_MODEL_PATH

# ========== Rank 1~(VERL_NODE_SIZE - 1) Ray Worker ==========
else
    # echo "🚀 启动 server 在 Worker 节点"
    # bash submit/common/retrieval_launch.sh &
    echo "✅ 启动 Ray Worker 节点"
    ray start --address="$MASTER_ADDR:6380"
fi

# ========== GPU 显存监控 ==========
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
while true; do
    total_used=0
    total_total=0
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        read used total <<< $(nvidia-smi --id=$i --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr ',' ' ')
        total_used=$((total_used + used))
        total_total=$((total_total + total))
    done
    usage=$(( (total_used * 100) / total_total ))
    echo "🟢 当前 GPU 显存占用: $usage%"
    sleep 30
done