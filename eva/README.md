# EVA 代码整理说明

## 目录结构

```
eva/
├── modules/                    # 核心模块（保留）
│   ├── eval_f1_offline.py     # 单文件评估
│   ├── eval_utils.py           # 评估工具函数
│   └── sglang_server_manager.py # SGLang服务器管理
│
├── experiments/                # 实验脚本（个人实验用）
│   ├── infer_*.py             # 推理脚本
│   ├── infer_*.sh             # 推理脚本
│   ├── test_eva_*.sh          # 批量测试脚本
│   ├── custom_server_*.py     # 服务器启动脚本
│   ├── eval_dir_to_csv.py     # 批量评估
│   └── eva.md                 # 实验记录
│
└── demo/                      # 基本使用（开源）
    ├── inference.py           # 简化的推理脚本
    ├── eval.py                # 简化的评估脚本
    └── README.md              # 使用说明
```

## 文件分类

### 保留的核心模块（modules/）

这些是评估的核心功能，用于开源：

- **eval_f1_offline.py**: 单文件评估脚本，计算F1分数和Token消耗
- **eval_utils.py**: 评估工具函数（F1计算、文本标准化等）
- **sglang_server_manager.py**: SGLang服务器管理器

### 实验脚本（experiments/）

这些是你个人实验用的脚本，包含了批量评测和各种配置：

- **推理脚本**:
  - `infer_nq_hotpotqa.py`: NQ-HotpotQA推理
  - `infer_bc.py`: BC数据集推理
  - `infer_idealab_nq_hotpotqa.py`: Idealab API推理
  - `infer_nq_hotpotqa_agentfold.py`: AgentFold推理

- **批量测试脚本**:
  - `test_eva_all_7b_tabel1.sh`: 7B模型批量测试
  - `test_eva_all_30b_tabel1.sh`: 30B模型批量测试
  - `test_eva_all_235b_tabel1.sh`: 235B模型批量测试
  - `test_eva_all_agentfold.sh`: AgentFold批量测试
  - `test_eva_all_235b_bc.sh`: BC数据集批量测试
  - `test_eva_bc_tabel3.sh`: BC数据集表格3测试

- **服务器脚本**:
  - `custom_server_mo.py`: 4GPU服务器启动
  - `custom_server.py`: 8GPU服务器启动
  - `sglang_server.sh`: SGLang服务器启动

- **其他**:
  - `eval_dir_to_csv.py`: 批量评估目录中的JSONL文件
  - `static_all.sh`: 静态评估所有结果
  - `eva.md`: 实验记录文档

### 基本使用（demo/）

简化版的使用脚本，用于开源和快速使用：

- **inference.py**: 简化的推理脚本
  - 支持MOQA和BC数据集
  - 不包含复杂的for循环
  - 简单的并发处理

- **eval.py**: 简化的评估脚本
  - 计算F1分数
  - 计算EM分数
  - 计算Token消耗

- **README.md**: 详细的使用说明
  - MOQA数据集使用方法
  - BC数据集使用方法
  - 参数说明和完整示例

## 删除的文件

以下文件已被删除，因为它们是无用的或重复的：

- `infer_gpt_oss.py`: GPT-OSS推理（不再使用）
- `infer_idealab.py`: Idealab简单测试（不再使用）
- `infer_online_web.py`: 在线Web推理（不再使用）
- `infer_bc_llj.py`: LLM Judge推理（不再使用）
- `infer_bc_llm_judge.py`: LLM Judge批量推理（不再使用）
- `openai_agent_loop.py`: OpenAI agent loop（不再使用）
- `test_parse.py`: 测试解析脚本（不再使用）
- `browser.py`: 浏览器工具（不再使用）
- `infer_bc.sh`: 重复的BC推理脚本
- `infer_bc_235.sh`: 重复的BC推理脚本
- `infer_nq_hotpotqa_single.sh`: 重复的NQ-HotpotQA推理脚本
- `run_llm_judge.sh`: LLM Judge脚本（不再使用）
- `infer_gpt_oss.sh`: GPT-OSS脚本（不再使用）
- `infer_online.sh`: 在线推理脚本（不再使用）
- `vllm.sh`: vLLM脚本（不再使用）
- `scripts/`: 原scripts目录，已合并到experiments/

## 数据集说明

### MOQA（NQ-HotpotQA）

- **描述**: 多问题数据集，每个样本包含多个问题需要回答
- **特点**: 
  - 需要为每个问题生成答案
  - 答案用分号分隔
  - 可能需要多轮对话
- **使用方法**: 参考 `demo/README.md` 中的MOQA示例

### BC（Search-R1）

- **描述**: 单问题数据集，需要使用搜索工具
- **特点**:
  - 单个问题
  - 需要调用搜索工具
  - 更简单直接
- **使用方法**: 参考 `demo/README.md` 中的BC示例

## 开源建议

开源时建议：

1. **包含**:
   - `modules/` 目录（核心功能）
   - `demo/` 目录（基本使用示例）
   - `README.md`（整体项目说明，需补充）

2. **不包含**:
   - `experiments/` 目录（个人实验脚本）
   - 任何包含个人路径的配置

3. **补充**:
   - 根目录 `README.md` 说明项目整体功能
   - 环境依赖说明
   - 安装步骤

## 使用流程

### 快速使用（MOQA/BC）

```bash
# 1. 启动服务器
python experiments/custom_server_mo.py /path/to/model server.log

# 2. 推理
python demo/inference.py \
    --model_path /path/to/model \
    --parquet /path/to/data.parquet \
    --out_jsonl results.jsonl

# 3. 评估
python demo/eval.py \
    --input_jsonl results.jsonl \
    --tokenizer_path /path/to/model
```

### 完整实验（experiments/）

参考 `experiments/eva.md` 中的实验记录，使用对应的脚本进行批量评测。