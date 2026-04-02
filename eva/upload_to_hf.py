#!/usr/bin/env python3
"""
Upload ContextBudget-MQ dataset to HuggingFace.
"""

from huggingface_hub import HfApi, login
from pathlib import Path
import sys
import os

# Configuration
REPO_ID = "watermaster-911/ContextBudget-MQ"
DATA_DIR = Path("/home/wy517954/code/ContextBudget/eva/data")

def main():
    print(f"[info] Uploading dataset to {REPO_ID}")
    print(f"[info] Data directory: {DATA_DIR}")

    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"[error] Data directory {DATA_DIR} does not exist")
        sys.exit(1)

    # Login to HuggingFace if token is provided
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            print(f"[info] Logged in to HuggingFace")
        except Exception as e:
            print(f"[warning] Failed to login: {e}")

    # Initialize HF API
    api = HfApi()

    # Check if repo exists, create if not
    try:
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type="dataset")
        print(f"[info] Repository {REPO_ID} already exists")
    except Exception as e:
        print(f"[info] Repository {REPO_ID} does not exist, creating...")
        try:
            api.create_repo(
                repo_id=REPO_ID,
                repo_type="dataset",
                public=True,
                exist_ok=True
            )
            print(f"[info] Created repository {REPO_ID}")
        except Exception as e:
            print(f"[error] Failed to create repository: {e}")
            sys.exit(1)

    # Upload all directories in data folder
    subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"[info] Found {len(subdirs)} directories to upload")

    for subdir in sorted(subdirs):
        print(f"\n[upload] Uploading {subdir.name}...")
        try:
            api.upload_folder(
                folder_path=str(subdir),
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo=subdir.name,
                commit_message=f"Upload {subdir.name}"
            )
            print(f"[success] Uploaded {subdir.name}")
        except Exception as e:
            print(f"[error] Failed to upload {subdir.name}: {e}")

    # Create README
    readme_content = """---
license: cc-by-4.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- context-budget
- multi-objective-qa
- retrieval-augmented-generation
- nq
- hotpotqa
- triviaqa
- popqa
---

# ContextBudget-MQ Dataset

Multi-objective QA dataset for evaluating context budget management in retrieval-augmented language models.

## Dataset Structure

The dataset contains multiple subdirectories for different target budgets (1, 2, 8, 16, 32) and BrowseComp-Plus evaluation:

- `nq_hotpotqa_train_multi_1/` - Target budget 1
- `nq_hotpotqa_train_multi_2/` - Target budget 2
- `nq_hotpotqa_train_multi_8/` - Target budget 8
- `nq_hotpotqa_train_multi_16/` - Target budget 16
- `nq_hotpotqa_train_multi_32/` - Target budget 32
- `processed_data_bc/` - BrowseComp-Plus evaluation data

Each directory contains processed versions:
- `elastic_processed/` - Processed with elastic search
- `mem1_processed/` - Processed with Mem1
- `sampled_base/` - Base sampled data
- `search_r1_processed/` - Processed with Search-R1

## Data Format

All data is stored in Parquet format with the following columns:
- `data_source` - Source dataset (nq, hotpotqa, triviaqa, popqa, etc.)
- `prompt` - Input prompt for the model
- `ability` - Task ability (fact-reasoning)
- `reward_model` - Reward model configuration
- `extra_info` - Additional metadata
- `question` - Original question
- `agent_name` - Agent type

## Usage

```python
from datasets import load_dataset

# Load test data for target 2
dataset = load_dataset("watermaster-911/ContextBudget-MQ", "nq_hotpotqa_train_multi_2", split="test")

# Or load BrowseComp-Plus data
dataset = load_dataset("watermaster-911/ContextBudget-MQ", "processed_data_bc", split="test")
```

## Citation

If you use this dataset, please cite:
```
@article{your_paper,
  title={Context Budget: Efficient Context Management for Retrieval-Augmented Language Models},
  author={Your Name},
  year={2025}
}
```
"""

    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add README"
        )
        print(f"\n[success] Uploaded README.md")
    except Exception as e:
        print(f"[error] Failed to upload README: {e}")

    print(f"\n[done] Upload complete! View at https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()