index_file=/data/oss_bucket_0/shiyi/data/wiki/e5_Flat.index
corpus_file=/data/oss_bucket_0/shiyi/data/wiki/wiki-18.jsonl
retriever_name=e5
retriever_path=/data/oss_bucket_0/shiyi/data/wiki/intfloat/e5-base-v2/


python retrieval/retrieval_server.py --index_path $index_file \
                                    --corpus_path $corpus_file \
                                    --topk 3 \
                                    --retriever_name $retriever_name \
                                    --retriever_model $retriever_path \
                                    --faiss_gpu
