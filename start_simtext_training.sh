#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME

python similar_text.py \
    --config_file="/search/odin/liruihong/NLPTool/config_data/model_config.json" \
    --vocab_file="/search/odin/liruihong/word2vec_embedding/2000000-small.txt" \
    --stop_words_file="/search/odin/liruihong/word2vec_embedding/cn_stopwords.txt" \
    --embedding_table="/search/odin/liruihong/word2vec_embedding/2000000-small.txt" \
    --init_checkpoint="/search/odin/liruihong/NLPTool/model_output/article2d_small_epoch2_negsample/model.ckpt-20000" \
    --output_dir="/search/odin/liruihong/NLPTool/model_output/sentence_pair_epoch2_negsample" \
    --embedding_table_trainable=False \
    --embedding_size=200 \
    --max_seq_length=256 \
    --do_predict=False \
    --save_checkpoint_steps=2000 \
    --do_train=True \
    --do_encode=False  \
    --batch_size=32 \
    --train_data="/search/odin/liruihong/NLPTool/datasets/sentence_pairs.tfrecord" \
    --num_warmup_steps=1000 \
    --num_train_steps=30000
