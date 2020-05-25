#!/usr/bin/env python
# -*- encoding=utf-8 -*-
from __future__ import absolute_import
import tensorflow as tf
import os
from model import tokenization
from util import data_util


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("stop_words_file", None, 
                    "The stop words file")

flags.DEFINE_integer("max_seq_length", None, "max_seq_length")

def main(_):
    label_list = ["1","2"]
    tokenizer = tokenization.Tokenizer(
        vocab_file=FLAGS.vocab_file, stop_words_file=FLAGS.stop_words_file)
    input_files_all = FLAGS.input_file  
    input_files = input_files_all.split('#')
    train_examples = []
    for input_file in input_files:
        tmp_examples = data_util.create_examples_from_json_file(input_file)
        train_examples.extend(tmp_examples)
    train_file = FLAGS.output_file
    data_util.file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("max_seq_length")
    tf.app.run()
