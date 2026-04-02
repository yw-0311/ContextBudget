# Data Generation Scripts

This directory contains scripts for generating and processing QA datasets for training and testing.

## Overview

The scripts are organized into two main workflows:

1. **Raw Data Generation** - Downloads and processes raw datasets from HuggingFace
2. **Data Processing** - Applies different prompt templates and agent configurations to the raw data

## Files

- `get_raw.sh` - Bash script to generate raw datasets
- `qa_search_train_merge_multi.py` - Generate training data from HuggingFace datasets
- `qa_search_test_merge_multi.py` - Generate test data from HuggingFace datasets
- `gen_main_train.py` - Process training data with different prompt templates
- `gen_main_test.py` - Process test data with different prompt templates

## Prerequisites

```bash
pip install datasets
```

## Usage

### Step 1: Generate Raw Data

Use the `get_raw.sh` script to download and process raw datasets:

```bash
cd /home/wy517954/code/ContextBudget/gen_data

# Use default batch size (3)
bash get_raw.sh

# Specify custom batch size
bash get_raw.sh --batch_size 5
```

This will:
- Create `data_all_raw_train/nq_hotpotqa_train_multi_*/train.parquet` for training
- Create `data_all_raw/nq_hotpotqa_train_multi_*/test.parquet` for testing

**Training Data Sources:** nq, hotpotqa
**Test Data Sources:** nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle

### Step 2: Process Training Data

Apply different prompt templates to the training data:

```bash
# Process all directories with default settings
python gen_main_train.py

# Sample 10% of data
python gen_main_train.py --sample_frac 0.1 --seed 42

# Sample exactly 256 examples
python gen_main_train.py --sample_n 256 --seed 42

# Process specific directories
python gen_main_train.py --dirs nq_hotpotqa_train_multi_1 nq_hotpotqa_train_multi_2

# Specify custom data root and output directory
python gen_main_train.py --data_root ./data_all_raw_train --out_root ./processed_data_train
```

Output structure:
```
processed_data_train/
├── nq_hotpotqa_train_multi_1/
│   ├── sampled_base/
│   │   └── train.parquet
│   ├── search_r1_processed/
│   │   └── train.parquet
│   └── elastic_processed/
│       └── train.parquet
```

### Step 3: Process Test Data

Apply different prompt templates to the test data:

```bash
# Process all directories with default settings
python gen_main_test.py

# Sample 10% of data
python gen_main_test.py --sample_frac 0.1 --seed 42

# Sample exactly 256 examples
python gen_main_test.py --sample_n 256 --seed 42

# Process specific directories
python gen_main_test.py --dirs nq_hotpotqa_train_multi_1 nq_hotpotqa_train_multi_2

# Specify custom data root and output directory
python gen_main_test.py --data_root ./data_all_raw --out_root ./processed_data
```

Output structure:
```
processed_data/
├── nq_hotpotqa_train_multi_1/
│   ├── sampled_base/
│   │   └── test.parquet
│   ├── search_r1_processed/
│   │   └── test.parquet
│   ├── elastic_processed/
│   │   └── test.parquet
│   └── mem1_processed/
│       └── test.parquet
```

## Prompt Templates

The scripts generate three different prompt variants:

### 1. Search R1 (tool_agent)
- System prompt: "You are Qwen, created by Alibaba Cloud."
- Uses `<information>`, `<search>`, `<answer>` tags
- Designed for search-based reasoning

### 2. Elastic (my_tool_agent)
- System prompt: "You are a research agent for long-running investigations."
- Uses `<context_commit>`, `<information>`, `<search>`, `<answer>` tags
- Designed for multi-step research with memory management

### 3. Mem1 (mem1)
- No system prompt (user-only)
- Uses `<information>`, `<search>`, `<answer>` tags
- FlashRAG-style format with persistent memory

## Command Line Arguments

### gen_main_train.py / gen_main_test.py

- `--data_root`: Directory containing raw data (default: `./data_all_raw_train` or `./data_all_raw`)
- `--dirs`: List of subdirectories to process (default: `nq_hotpotqa_train_multi_1`)
- `--out_root`: Output root directory (default: `./processed_data_train` or `./processed_data`)
- `--seed`: Random seed for deterministic sampling (default: 42)
- `--sample_frac`: Fraction of data to sample (<1.0 to take effect) (default: 1.0)
- `--sample_n`: Fixed number of examples to sample (higher priority than sample_frac) (default: None)
- `--num_proc`: Number of parallel processes for datasets.map (default: None)

### get_raw.sh

- `--batch_size`: Number of questions to process together (default: 3)

## Data Format

Each parquet file contains the following fields:

- `data_source`: Source dataset name (e.g., "nq", "hotpotqa")
- `prompt`: List of message dicts with `role` and `content`
- `question`: The question text
- `agent_name`: Agent type (e.g., "tool_agent", "my_tool_agent", "mem1")
- `ability`: Task type ("fact-reasoning")
- `reward_model`: Dictionary with `style` and `ground_truth`
- `extra_info`: Additional metadata (split, indices)

## Example Workflow

```bash
# 1. Generate raw data with batch size 5
bash get_raw.sh --batch_size 5

# 2. Process training data
python gen_main_train.py --sample_frac 0.1 --seed 42

# 3. Process test data
python gen_main_test.py --sample_frac 0.1 --seed 42

# 4. Copy processed data to storage
cp -r processed_data_train/nq_hotpotqa_train_multi_5 /path/to/train_data/
cp -r processed_data/nq_hotpotqa_train_multi_5 /path/to/test_data/
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'datasets'`
**Solution**: Install the required package: `pip install datasets`

**Issue**: `OSError: data_all_raw_train not found`
**Solution**: Run `bash get_raw.sh` first to generate the raw data

**Issue**: Memory errors with large datasets
**Solution**: Use `--sample_frac` or `--sample_n` to reduce the dataset size

## Notes

- The scripts use HuggingFace's `datasets` library for efficient data handling
- All sampling is deterministic when using the same seed
- The scripts automatically create necessary output directories
- Processed data is saved in parquet format for efficient loading