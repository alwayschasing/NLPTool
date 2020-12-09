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

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "input_file")
flags.DEFINE_string("output_dir", None, "output_dir")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("vocab_file", None, "vocab_file")
flags.DEFINE_string("embedding_table", None, "embedding_table file")
flags.DEFINE_bool("embedding_table_trainable", True, "embedding_table_trainable")
flags.DEFINE_string("stop_words_file", None, "stop_words_file")
flags.DEFINE_string("init_checkpoint", None, "init_checkpoint")
flags.DEFINE_integer("max_seq_length", 256, "max_seq_length")
flags.DEFINE_string("config_file", None, "config_file")
flags.DEFINE_integer("embedding_size", 200, "word_vec embedding size")
flags.DEFINE_bool("do_train", False, "do_train")
flags.DEFINE_bool("do_eval", False, "do_eval")
flags.DEFINE_bool("do_predict", False, "do_predict")
flags.DEFINE_integer("batch_size", 32, "batch_size")
flags.DEFINE_integer("train_batch_size", 32, "train_batch_size")
flags.DEFINE_integer("eval_batch_size", 32, "eval_batch_size")
flags.DEFINE_integer("predict_batch_size", 32, "predict_batch_size")
flags.DEFINE_float("learning_rate", 5e-5, "learning_rate")
flags.DEFINE_integer("num_train_steps", 10, "num_train_steps")
flags.DEFINE_integer("num_train_epochs", 10, "num_train_steps")
flags.DEFINE_integer("num_warmup_steps", 10, "num_warmup_steps")
flags.DEFINE_float("warmup_proportion", 0.1, "warmup_proportion")
flags.DEFINE_integer("save_checkpoint_steps", 1000, "save_checkpoint_steps")
flags.DEFINE_string("train_data",None,"train_data")
flags.DEFINE_string("eval_data",None,"eval_data")
flags.DEFINE_string("pred_data",None,"pred_data")
flags.DEFINE_bool("do_encode",False,"whether in the sentence encode mode")
flags.DEFINE_string("encode_data",None,"encode_data")
flags.DEFINE_string("encode_output",None,"encode_output")



def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids_a": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask_a": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids_a": tf.FixedLenFeature([seq_length], tf.int64),
        "input_ids_b": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask_b": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids_b": tf.FixedLenFeature([seq_length], tf.int64),
        "label": tf.FixedLenFeature([], tf.float32),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        #d = d.apply(lambda record:_decode_record(record, name_to_features))
        d = d.map(lambda record:_decode_record(record, name_to_features))
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        #d = d.apply(
        #    tf.contrib.data.map_and_batch(
        #        lambda record: _decode_record(record, name_to_features),
        #        batch_size=batch_size,
        #        drop_remainder=drop_remainder))

        return d
    return input_fn

def file_based_encode_input_fn_builder(input_file,
                                       max_seq_length,
                                       tokenizer):
    examples = create_examples_from_tsv_file(input_file)
    input_features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, ["1","2"], max_seq_length, tokenizer)
        input_features.append(feature)

    def generate_fn():
        for feature in input_features:
            input_ids = feature.input_ids
            input_mask = feature.input_mask
            segment_ids = feature.segment_ids
            yield {"input_ids_a":input_ids, 
                   "input_mask_a":input_mask, 
                   "segment_ids_a":segment_ids}

    def input_fn(params):
        dataset = tf.data.Dataset.from_generator(generate_fn, 
                                                 output_types={"input_ids_a":tf.int64,
                                                               "input_mask_a":tf.int64,
                                                               "segment_ids_a":tf.int64},
                                                 output_shapes={"input_ids_a":tf.TensorShape([max_seq_length]),
                                                               "input_mask_a":tf.TensorShape([max_seq_length]),
                                                               "segment_ids_a":tf.TensorShape([max_seq_length])})
        
        d = dataset.batch(params["batch_size"])
        return d
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


def create_similar_model(model_config,
                         is_training,
                         input_ids_a,
                         input_mask_a,
                         segment_ids_a,
                         input_ids_b,
                         input_mask_b,
                         segment_ids_b,
                         label,
                         embedding_table=None,
                         hidden_dropout_prob=0.1,
                         use_one_hot_embeddings=False):
    """Creates a classification model."""
    model_a = modeling.TextEncoder(
        config=model_config,
        is_training=is_training,
        input_ids=input_ids_a,
        embedding_table=embedding_table,
        input_mask=input_mask_a,
        token_type_ids=segment_ids_a)

    model_b = modeling.TextEncoder(
        config=model_config,
        is_training=is_training,
        input_ids=input_ids_b,
        embedding_table=embedding_table,
        input_mask=input_mask_b,
        token_type_ids=segment_ids_b)

    sequence_output_a = model_a.get_sequence_output()
    sequence_output_b = model_b.get_sequence_output()

    seq_out_shape = get_shape_list(sequence_output_a)
    batch_size = seq_out_shape[0]
    
    text_representation_a = tf.reduce_mean(sequence_output_a, axis=1)
    text_representation_b = tf.reduce_mean(sequence_output_b, axis=1)
    
    with tf.variable_scope("loss"):
        normalize_a = tf.math.l2_normalize(text_representation_a,axis=-1)
        normalize_b = tf.math.l2_normalize(text_representation_b,axis=-1)
        # a_shape = get_shape_list(normalize_a)
        # b_shape = get_shape_list(normalize_b)
        # tf.logging.info("a_shape:%s, b_shape:%s"%(str(a_shape), str(b_shape)))
        # cosine_dist = tf.losses.cosine_distance(normalize_a, normalize_b, axis=-1)
        # label_shape = get_shape_list(label)
        # cosine_dist_shape = get_shape_list(cosine_dist)
        # tf.logging.info("label_shape:%s, cosine_dist_shape:%s"%(str(label_shape), str(cosine_dist_shape)))
        normalize_a = tf.reshape(normalize_a, shape=[batch_size, 1, -1])
        normalize_b = tf.reshape(normalize_b, shape=[batch_size, 1, -1])
        cosine = tf.matmul(normalize_a, normalize_b, transpose_b=True)
        cosine = tf.reshape(cosine, shape=[batch_size])
        # cosine = tf.tensordot(normalize_a, normalize_b, axes=[[1],[1]])
        loss = tf.reduce_sum(tf.losses.mean_squared_error(label, cosine))
        return (loss, cosine)  


def model_fn_builder(model_config,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
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

        if do_encode == False:
            input_ids_b = features["input_ids_b"]
            input_mask_b = features["input_mask_b"]
            segment_ids_b = features["segment_ids_b"]
            label = features["label"]

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

        if do_encode:
            text_representation = create_encode_model(model_config,
                                                       is_training,
                                                       input_ids_a,
                                                       input_mask_a,
                                                       segment_ids_a,
                                                       embedding_table)
        else:
            (total_loss, cosine) = create_similar_model(model_config,
                                                             is_training,
                                                             input_ids_a,
                                                             input_mask_a,
                                                             segment_ids_a,
                                                             input_ids_b,
                                                             input_mask_b,
                                                             segment_ids_b,
                                                             label,
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

        output_spec = None
        if do_encode == False:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                log_hook = tf.train.LoggingTensorHook({"total_loss":total_loss}, every_n_iter=100)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[log_hook],
                    scaffold=scaffold)
            elif mode == tf.estimator.ModeKeys.EVAL:
                pass
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"cosine":cosine},
                    prediction_hooks=None,
                    scaffold=scaffold)
        else:
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


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_encode:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' or `do_encode` must be True.")

    model_config = ModelConfig.from_json_file(FLAGS.config_file)
    if FLAGS.max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, model_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)
    processor = PairTextProcessor()
    tokenizer = tokenization.Tokenizer(
        vocab_file=FLAGS.vocab_file, stop_words_file=FLAGS.stop_words_file, use_pos=False)
    tf.logging.info("model_config vocab_size:%d, tokenizer.vocab_size:%d"%(model_config.vocab_size, tokenizer.vocab_size))
    assert(model_config.vocab_size == tokenizer.vocab_size)

    if FLAGS.embedding_table is not None:
        embedding_table = load_embedding_table(FLAGS.embedding_table)
    else:
        embedding_table = None

    assert(len(tokenizer.vocab) == embedding_table.shape[0])

    train_examples = None
    num_train_steps = FLAGS.num_train_steps
    num_warmup_steps = FLAGS.num_warmup_steps

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=6,
        log_step_count_steps=100)

    model_fn = model_fn_builder(
        model_config=model_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        do_encode=FLAGS.do_encode,
        embedding_table_value=embedding_table,
        embedding_table_trainable=FLAGS.embedding_table_trainable,
        use_one_hot_embeddings=False)

    params = {
        "batch_size":FLAGS.batch_size,
    }
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params)

    if FLAGS.do_train:
        train_file = FLAGS.train_data
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    elif FLAGS.do_eval:
        pass

    elif FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.pred_data)
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_pairexamples_to_features(
            predict_examples, FLAGS.max_seq_length, tokenizer, predict_file)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d ", num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn, hooks=None)
        output_predict_file = os.path.join(FLAGS.output_dir, "pred_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                text_representation = prediction["text_representation"]
                keyword_probs = prediction["keyword_probs"]
                input_ids = prediction["input_ids"]
                if i >= num_actual_predict_examples:
                    break

                sorted_keyword_probs = np.argsort(keyword_probs, axis=-1)
                top_keyword_ids = []
                top_keyword_probs = []
                for i in range(-1,-6,-1):
                    idx = sorted_keyword_probs[i]
                    top_keyword_ids.append(input_ids[idx])
                    top_keyword_probs.append(keyword_probs[idx])
                    
                top_keywords = tokenizer.convert_ids_to_tokens(top_keyword_ids)
                output_line = "\t".join(kw + ":" + str(prob) for kw,prob in zip(top_keywords, top_keyword_probs)) + "\n"
                writer.write(output_line)
                words = tokenizer.convert_ids_to_tokens(input_ids)
                check_line = "\t".join(w + ":" + str(prob) for w, prob in zip(words, keyword_probs)) + "\n"
                writer.write(check_line)
                num_written_lines += 1
        print("num_writen_lines:%d,num_actual_predict_examples:%d"%(num_written_lines, num_actual_predict_examples))
        assert num_written_lines == num_actual_predict_examples
    elif FLAGS.do_encode:
        encode_input_file = FLAGS.encode_data
        encode_input_fn = file_based_encode_input_fn_builder(
            input_file=encode_input_file,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer)

        output_file = FLAGS.encode_output
        wfp = open(output_file, "wb")
        result = estimator.predict(input_fn=encode_input_fn, hooks=None)
        text_embeddings = []
        for idx, item in enumerate(result):
            text_embeddings.append(item["text_representation"])
            if idx < 10:
                tf.logging.info("%s"%(item["text_representation"]))
        pickle.dump(text_embeddings, wfp)
        wfp.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="[%(asctime)s-%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("do_encode")
    tf.app.run()

