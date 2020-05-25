#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME

start_time="20200510"
day_range=5
input_file=""
for i in $(seq 0 (($day_range-1)))
do
    tmp_file=`date -d "$start_time +${i}days" +"%Y%m%d"`
    input_file="$input_file#$tmp_file"
    echo "$tmp_file"
done

python create_training_tfrecord.py \
    --vocab_file="/search/odin/liruihong/word2vec_embedding/2000000-small.txt" \
    --stop_words_file="/search/odin/liruihong/word2vec_embedding/cn_stopwords.txt" \
    --max_seq_length=256 \
    --input_file="/search/odin/liruihong/article_data/article_5d" \
    --output_file="/search/odin/liruihong/NLPTool/datasets/article_5d_clean.tfrecord" 

