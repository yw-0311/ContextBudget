# EVA 使用说明

本目录包含EVA（Evaluation Agent）的基本使用脚本，用于评估MOQA和BC数据集。

## 目录结构

```
eva/
├── modules/                    # 核心模块
│   ├── eval_f1_offline.py     # 单文件评估
│   ├── eval_utils.py           # 评估工具函数
│   └── sglang_server_manager.py # SGLang服务器管理
├── experiments/                # 实验脚本（批量评测）
│   ├── infer_*.py             # 各种推理脚本
│   └── *.sh                   # 批量测试脚本
└── demo/                      # 基本使用脚本
    ├── inference.py           # 推理脚本
    ├── eval.py                # 评估脚本
    └── README.md              # 本文档
```

## 数据集

### MOQA数据集（NQ-HotpotQA）

MOQA（Multi-Object Question Answering）数据集，包含多个问题需要回答的场景。

**数据格式：**
- 每个样本包含多个问题
- 需要为每个问题生成答案
- 答案用分号分隔：`answer1; answer2; answer3`

**评估指标：**
- F1分数：基于token匹配的F1分数
- EM分数：精确匹配分数
- Token消耗：Dependent Cost和Peak Tokens

### BC数据集（Search-R1）

BC（Browser Chain）数据集，包含需要使用搜索工具的单问题场景。

**数据格式：**
- 每个样本包含一个问题
- 需要调用搜索工具获取信息
- 生成单个答案

**评估指标：**
- F1分数
- EM分数
- Token消耗

## 使用方法

### 1. 启动SGLang服务器

首先需要启动SGLang服务器：

```bash
# 使用8张GPU启动
python experiments/custom_server_mo.py /path/to/model server.log
```

### 2. 推理

#### MOQA数据集

```bash
python demo/inference.py \
    --model_path /path/to/model \
    --parquet /path/to/moqa_data.parquet \
    --out_jsonl moqa_results.jsonl \
    --sglang_url http://localhost:30000 \
    --max_model_len 8192 \
    --concurrency 4
```

#### BC数据集

```bash
python demo/inference.py \
    --model_path /path/to/model \
    --parquet /path/to/bc_data.parquet \
    --out_jsonl bc_results.jsonl \
    --sglang_url http://localhost:30000 \
    --max_model_len 8192 \
    --concurrency 4
```

### 3. 评估

```bash
# 评估MOQA结果
python demo/eval.py \
    --input_jsonl moqa_results.jsonl \
    --tokenizer_path /path/to/model

# 评估BC结果
python demo/eval.py \
    --input_jsonl bc_results.jsonl \
    --tokenizer_path /path/to/model
```

## 参数说明

### inference.py 参数

- `--model_path`: 模型路径（必需）
- `--parquet`: 数据文件路径（必需）
- `--out_jsonl`: 输出JSONL文件（必需）
- `--sglang_url`: SGLang服务器URL（默认: http://localhost:30000）
- `--tool_config_path`: 工具配置文件路径（可选）
- `--max_model_len`: 最大模型长度（默认: 8192）
- `--concurrency`: 并发数（默认: 4）

### eval.py 参数

- `--input_jsonl`: 输入JSONL文件（必需）
- `--tokenizer_path`: Tokenizer路径（必需）

## 输出格式

推理结果JSONL文件包含以下字段：

- `idx`: 样本索引
- `question`: 问题文本
- `prediction`: 完整预测
- `answer`: 提取的答案
- `num_turns`: 对话轮数
- `ground_truth`: 真实答案

## 评估结果

评估脚本会输出以下指标：

- **样本数**: 成功处理的样本数量
- **F1**: 平均F1分数 ± 标准误差
- **EM**: 平均精确匹配分数
- **Dependent Cost**: 平均assistant输出token数
- **Peak Tokens**: 平均单轮上下文token峰值
- **平均轮次**: 平均对话轮数

## 依赖

```bash
pip install pandas transformers
```

完整版本还需要：

```bash
pip install verl  # 从项目源码安装
```

## 注意事项

1. 确保SGLang服务器已启动并正常运行
2. 数据文件必须包含 `prompt` 字段（消息列表）和 `ground_truth` 字段
3. MOQA数据集需要处理多个问题，答案用分号分隔
4. BC数据集是单问题场景，需要使用搜索工具

## 完整示例

### MOQA完整流程

```bash
# 1. 启动服务器
python experiments/custom_server_mo.py /path/to/model server.log

# 2. 运行推理
python demo/inference.py \
    --model_path /path/to/model \
    --parquet /path/to/moqa_test.parquet \
    --out_jsonl moqa_results.jsonl \
    --sglang_url http://localhost:30000

# 3. 评估结果
python demo/eval.py \
    --input_jsonl moqa_results.jsonl \
    --tokenizer_path /path/to/model
```

### BC完整流程

```bash
# 1. 启动服务器
python experiments/custom_server_mo.py /path/to/model server.log

# 2. 运行推理
python demo/inference.py \
    --model_path /path/to/model \
    --parquet /path/to/bc_test.parquet \
    --out_jsonl bc_results.jsonl \
    --sglang_url http://localhost:30000

# 3. 评估结果
python demo/eval.py \
    --input_jsonl bc_results.jsonl \
    --tokenizer_path /path/to/model
```