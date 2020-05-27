#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME

python keyword_pretraining.py \
    --config_file="/search/odin/liruihong/NLPTool/config_data/model_config.json" \
    --vocab_file="/search/odin/liruihong/word2vec_embedding/2000000-small.txt" \
    --stop_words_file="/search/odin/liruihong/word2vec_embedding/cn_stopwords.txt" \
    --embedding_table="/search/odin/liruihong/word2vec_embedding/2000000-small.txt" \
    --output_dir="/search/odin/liruihong/NLPTool/model_output/article_2d.epoch2" \
    --embedding_table_trainable=False \
    --embedding_size=200 \
    --max_seq_length=256 \
    --save_checkpoint_steps=100 \
    --do_train=False \
    --train_data="/search/odin/liruihong/NLPTool/datasets/article_2d.tfrecord" \
    --save_checkpoint_steps=100 \
    --num_warmup_steps=100 \
    --num_train_steps=25000 \
    --do_predict=True \
    --pred_data="/search/odin/liruihong/article_data/article_test"
