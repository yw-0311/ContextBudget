#!/bin/bash

WORK_DIR=.

# Default batch size
BATCH_SIZE=3

export HF_ENDPOINT=https://hf-mirror.com

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Process multiple dataset search format train file
LOCAL_DIR=$WORK_DIR/data_all_raw_train/nq_hotpotqa_train_multi_${BATCH_SIZE}
DATA=nq,hotpotqa
python $WORK_DIR/qa_search_train_merge_multi.py --local_dir $LOCAL_DIR --data_sources $DATA --batch_size $BATCH_SIZE

# Process multiple dataset search format test file
LOCAL_DIR=$WORK_DIR/data_all_raw/nq_hotpotqa_train_multi_${BATCH_SIZE}
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/qa_search_test_merge_multi.py --local_dir $LOCAL_DIR --data_sources $DATA --batch_size $BATCH_SIZE