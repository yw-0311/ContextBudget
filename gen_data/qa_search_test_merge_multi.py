#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess the QA dataset to parquet format for testing
"""

import os
import argparse
import datasets


def make_prefix(dp, template_type):
    questions = dp['questions']

    if template_type == 'base':
        prefix = f"""You will answer multiple complex questions using iterative reasoning, summarization, and web search.

At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

1. Perform reasoning and update a cumulative, concise summary within <information> ... </information>. This acts as persistent memory and must include all essential information from previous <information> tags.

2. Then choose one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Always follow this structure after <information> or the initial questions: <information> ... </information><search> ... </search> or <information> ... </information><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {questions}\n"""
    else:
        raise NotImplementedError
    return prefix


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data_all_raw/nq_hotpotqa_train_multi_3')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--batch_size', type=int, default=3, help='Number of questions to process together')

    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:
        if data_source != 'strategyqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        else:
            dataset = datasets.load_dataset('json', data_files="/home/peterjin/mnt/data/strategyqa/test_correct.jsonl")

        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']

        def process_batch(examples, indices):
            processed_data = []

            questions = examples['question']
            new_questions = []
            for question in questions:
                if question[-1] != '?':
                    question += '?'
                new_questions.append(question)
            questions = new_questions

            golden_answers = examples['golden_answers']

            questions_str = '; '.join(questions)

            solution = {
                "target": golden_answers
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": questions_str,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': 'test',
                    'indices': indices,
                },
                "question": questions_str
            }

            processed_data.append(data)

            return processed_data

        processed_data = []
        if len(test_dataset) % args.batch_size != 0:
            test_dataset = test_dataset.select(range(len(test_dataset) - (len(test_dataset) % args.batch_size)))
        print(f"[{data_source}] length of test_dataset: {len(test_dataset)}")

        for i in range(0, len(test_dataset), args.batch_size):
            batch = test_dataset[i:i + args.batch_size]
            batch_indices = list(range(i, i + args.batch_size))
            processed_data.extend(process_batch(batch, batch_indices))

        test_dataset = datasets.Dataset.from_list(processed_data)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    ensure_dir(local_dir)
    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"[save] {os.path.join(local_dir, 'test.parquet')} (rows={len(all_test_dataset)})")
