#!/usr/bin/env python
# -*-encoding=utf-8-*-
import tensorflow as tf
import csv
import os
import collections
import numpy as np
import math
from model import modeling
from model.model_config import ModelConfig
from model import tokenization
from model import optimization
from model.util import create_initializer
from model.util import get_shape_list
from model.util import get_assignment_map_from_checkpoint
from util.data_util import InputExample 
from util.data_util import InputFeatures
from util.data_util import PairTextProcessor
from util.data_util import _truncate_seq_pair
from util.data_util import convert_single_example
from util.data_util import file_based_convert_pairexamples_to_features
from util.data_util import file_based_convert_examples_to_features
from util.data_util import create_examples_from_tsv_file
import logging
import pickle
import faiss
from sklearn.preprocessing import normalize
import threading
import zmq
import zmq.decorators as zmqd


def set_logger(name=None, verbose=False, handler=logging.StreamHandler()):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    #formatter = logging.Formatter('[%(levelname).1s%(name)s-%(asctime)s %(filename)s:%(funcName)s:%(lineno)3d] %(message)s', datefmt='%m-%d %H:%M:%S')
    formatter = logging.Formatter('[%(levelname).1s:%(name)s (%(asctime)s) %(funcName)s] %(message)s', datefmt='%m-%d %H:%M:%S')
    console_handler = handler
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def receiver_encode_input_fn_builder(receiver_sock,
                                     max_seq_length,
                                     tokenizer):
    def generate_fn():
        while True:
            events = receiver_sock.poll()
            if events:
                js_data = receiver_sock.recv_json()
                text = js_data["query"]
                example = InputExample(guid=0, text_a=text, text_b = None, label="1")
                feature = convert_single_example(10, example, ["1","2"], max_seq_length, tokenizer)
                input_ids = feature.input_ids
                input_mask = feature.input_mask
                segment_ids = feature.segment_ids
                tf.logging.info("[recv] %s"%(text))
                tf.logging.info("features:\n%s\n%s\n%s\n"%(input_ids, input_mask, segment_ids))
                yield {"input_ids_a":[input_ids], 
                    "input_mask_a":[input_mask], 
                    "segment_ids_a":[segment_ids]}

    def input_fn(params):
        data = tf.data.Dataset.from_generator(generate_fn, 
                                                 output_types={"input_ids_a":tf.int64,
                                                               "input_mask_a":tf.int64,
                                                               "segment_ids_a":tf.int64},
                                                 output_shapes={"input_ids_a":tf.TensorShape([None,max_seq_length]),
                                                               "input_mask_a":tf.TensorShape([None,max_seq_length]),
                                                               "segment_ids_a":tf.TensorShape([None,max_seq_length])})
        #data = data.batch(batch_size=16) 
        return data
    return input_fn


def create_encode_model(model_config,
                         is_training,
                         input_ids,
                         input_mask,
                         segment_ids,
                         embedding_table=None,
                         hidden_dropout_prob=0.1,
                         use_one_hot_embeddings=False):
    """Creates a classification model."""
    model = modeling.TextEncoder(
        config=model_config,
        is_training=is_training,
        input_ids=input_ids,
        embedding_table=embedding_table,
        input_mask=input_mask,
        token_type_ids=segment_ids)

    sequence_output = model.get_sequence_output()
    
    text_representation = tf.reduce_mean(sequence_output, axis=1)
    
    return text_representation  


def model_fn_builder(model_config,
                     init_checkpoint,
                     do_encode=False,
                     embedding_table_value=None,
                     embedding_table_trainable=True,
                     use_one_hot_embeddings=False):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids_a = features["input_ids_a"]
        input_mask_a = features["input_mask_a"]
        segment_ids_a = features["segment_ids_a"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        embedding_table = tf.get_variable("embedding_table",
                                          shape=[model_config.vocab_size, model_config.vocab_vec_size],
                                          trainable=embedding_table_trainable)

        def init_embedding_table(scoffold,sess):
            sess.run(embedding_table.initializer, {embedding_table.initial_value: embedding_table_value})

        if embedding_table_value is not None:
            scaffold = tf.train.Scaffold(init_fn=init_embedding_table)
        else:
            scaffold = None

        text_representation = create_encode_model(model_config,
                                                    is_training,
                                                    input_ids_a,
                                                    input_mask_a,
                                                    segment_ids_a,
                                                    embedding_table)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions={"text_representation": text_representation},
            prediction_hooks=None,
            scaffold=scaffold)
        return output_spec

    return model_fn


def load_embedding_table(embedding_table_file):
    # embedding_table = tokenization.load_embedding_table(embedding_table_file)
    vec_table = []
    with open(embedding_table_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        for i,line in enumerate(lines):
            if i == 0:
                continue
            items = line.rstrip().split(' ')
            word = items[0]
            vec = items[1:]
            vec_table.append(vec)
    embedding_table = np.asarray(vec_table)
    tf.logging.info("load embedding_table, shape is %s"%(str(embedding_table.shape)))
    return embedding_table


def create_estimator(model_config, init_checkpoint, model_dir, embedding_table_file):
    embedding_table = load_embedding_table(embedding_table_file)
    model_fn = model_fn_builder(
        model_config=model_config,
        init_checkpoint=init_checkpoint,
        do_encode=True,
        embedding_table_value=embedding_table)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=None)
    return estimator        


class WordVecTransformer(threading.Thread):
    def __init__(self, 
                 init_checkpoint, 
                 vocab_file,
                 stop_words_file,
                 config_file,
                 embedding_table_file,
                 index_vecs_file, 
                 index_data_file, 
                 listen_port=5022, 
                 logger=logging.StreamHandler()):
        super(WordVecTransformer, self).__init__()
        self.logger = logger
        self.port = listen_port
        self.vec_size = 2400
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")
        model_dir="./tmp"
        self.index_vecs = np.asarray(self.load_index_bin(index_vecs_file), dtype=np.float32)
        self.index_data = self.load_index_data(index_data_file)
        self.tokenizer = tokenization.Tokenizer(
            vocab_file=vocab_file, stop_words_file=stop_words_file, use_pos=False)
        self.model_config = ModelConfig.from_json_file(config_file)
        self.estimator = create_estimator(self.model_config, init_checkpoint, model_dir, embedding_table_file)
        self.build_index(self.index_vecs)
        self.logger.info("Finish WordVecTransFormer init.")

    def build_index(self, index_data):
        "normalize data"
        self.logger.info("Start build index")
        self.normalize_data = normalize(index_data)
        self.index = faiss.IndexFlatIP(self.vec_size)
        self.index.add(self.normalize_data)
        self.logger.info("Finish build index")

    def load_index_bin(self, index_bin_file):
        fp = open(index_bin_file, "rb")
        data = pickle.load(fp)
        fp.close()
        return data

    def load_index_data(self, index_data_file):
        index_data = []
        with open(index_data_file, "r") as fp:
            for line in fp:
                items = line.strip().split('\t')
                texts = items[0]
                docid = items[1]
                index_data.append((texts, docid))
        return index_data
    
    def run(self):
        self._run()


    @zmqd.context()
    @zmqd.socket(zmq.REP)
    def _run(self,_,receiver_sock):
        receiver_sock.bind('tcp://10.153.57.105:%d' % self.port)
        self.logger.info("bind wordtrans_server socket:%s:%d"%("tcp://10.153.57.105",self.port))
        input_fn = receiver_encode_input_fn_builder(receiver_sock, self.model_config.max_seq_length, self.tokenizer)
        result = self.estimator.predict(input_fn=input_fn)
        for item in result:
            """server job""" 
            text_embedding = item["text_representation"]
            text_embedding = normalize(np.asarray([text_embedding], dtype=np.float32))
            dis, idx = self.index.search(text_embedding, 10)
            res = []
            for i,d in enumerate(dis[0]):
                neighbor_text, docid = self.index_data[idx[0][i]]
                self.logger.debug("%s\t%s\t%s"%(neighbor_text,str(d),docid))
                self.logger.debug("sentrans_embeddings:\n%s"%(self.index_vecs[idx[0][i]]))
                self.logger.debug("norm_embeddings:\n%s"%(self.normalize_data[idx[0][i]]))
                res.append([neighbor_text, str(d), docid])
            receiver_sock.send_json({"wordtrans_res":res})


def start_server():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.logging.set_verbosity(tf.logging.DEBUG)
    init_checkpoint = "/search/odin/liruihong/NLPTool/model_output/sentence_pair_epoch2/model.ckpt-14000"
    vocab_file = "/search/odin/liruihong/word2vec_embedding/2000000-small.txt"
    stop_words_file = "/search/odin/liruihong/word2vec_embedding/cn_stopwords.txt"
    config_file = "/search/odin/liruihong/NLPTool/config_data/model_config.json"
    embedding_table_file = "/search/odin/liruihong/word2vec_embedding/2000000-small.txt"
    index_vecs_file = "/search/odin/liruihong/article_data/index_data/titles_5d_wordtrans.bin2"
    index_data_file = "/search/odin/liruihong/article_data/index_data/titles_5d.tsv"
    listen_port = 5022
    logger = set_logger(name="svc", verbose=True, handler=logging.StreamHandler())
    encoder_server = WordVecTransformer(init_checkpoint, 
                                        vocab_file,
                                        stop_words_file,
                                        config_file,
                                        embedding_table_file,
                                        index_vecs_file, 
                                        index_data_file, 
                                        listen_port=listen_port,
                                        logger=logger)

    encoder_server.start()
    encoder_server.join()

if __name__ == "__main__":
    start_server()
