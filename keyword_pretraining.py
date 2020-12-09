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
from util.data_util import TextProcessor
from util.data_util import _truncate_seq_pair
from util.data_util import convert_single_example
from util.data_util import file_based_convert_examples_to_features
import logging

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
flags.DEFINE_bool("use_pos",False,"whether use part of speech feature")


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "keyword_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
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


def word_self_attention_layer(input_tensor,
                              input_mask,
                              num_attention_heads,
                              size_per_head,
                              hidden_size,
                              query_act=None,
                              key_act=None,
                              value_act=None,
                              attention_probs_dropout_prob=0.0,
                              initializer_range=0.02):
    """
    Args:
        input_tensor: Float Tensor of shape [batch_size, seq_length, hidden_size]
        input_mask: int Tensor of shape [batch_size, seq_length]
        hidden_size
        asster hidden_size == num_attention_heads * size_per_head
        size_per_head == word_embedding_size
    """
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor,
                                   [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    shape_list = get_shape_list(input_tensor, expected_rank=[2,3])
    batch_size = shape_list[0]
    seq_length = shape_list[1]

    query_layer = tf.layers.dense(
        inputs=input_tensor,
        units=hidden_size,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range)
    )
    
    key_layer = tf.layers.dense(
        inputs=input_tensor,
        units=hidden_size,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range)
    )
    
    value_layer = tf.layers.dense(
        inputs=input_tensor,
        units=hidden_size,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range)
    )
    query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, seq_length, size_per_head) 
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, seq_length, size_per_head)

    query_shape_list = get_shape_list(query_layer, expected_rank=4)
    tf.logging.info("query_layer shape: %s"%(str(query_shape_list)))
    # query shape [batch_size, seq_length, num_heads, size_per_head]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0/math.sqrt(float(size_per_head)))
    # attention_mask shape: [batch_size, seq_length, seq_length]
    attention_mask = modeling.create_attention_mask_from_input_mask(input_tensor, input_mask)
    # expand for multi heads, [batch_size, 1, seq_length, seq_length]
    attention_mask = tf.expand_dims(attention_mask, axis=[1]) 
    mask_adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    # attention_score: [batch_size, num_heads, seq_length, seq_length]
    attention_scores += mask_adder

    # attention_probs shape: [batch_size, num_heads, seq_length, seq_length]
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = modeling.dropout(attention_probs, attention_probs_dropout_prob)
    
    value_layer = tf.reshape(value_layer, [batch_size, seq_length, num_attention_heads, size_per_head])
    # value_layer shape : [batch_size, num_heads, seq_length, size_per_head]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    context_layer = tf.matmul(attention_probs, value_layer)
    # context_layer shape : [batch_size, seq_length, num_heads, size_per_head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    context_layer = tf.reshape(context_layer, [batch_size, seq_length, num_attention_heads*size_per_head])
    return context_layer
    

def create_model(model_config,
                 is_training,
                 input_ids,
                 input_mask,
                 keyword_mask,
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

    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    sequence_output = model.get_sequence_output()
    sequence_shape = get_shape_list(sequence_output, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]

    num_heads = model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    size_per_head = int(hidden_size/num_heads)
    
    prev_output = sequence_output
    for word_layer_idx in range(model_config.word_attn_layer_num):
        layer_input = prev_output
        with tf.variable_scope("word_attn_layer_%d"%(word_layer_idx)):
            attention_head = word_self_attention_layer(layer_input,
                                                        input_mask,
                                                        num_heads,
                                                        size_per_head,
                                                        hidden_size)

            attention_output = tf.layers.dense(
                attention_head,
                hidden_size,
                activation=None,
                kernel_initializer=create_initializer(0.02)
            )
            attention_output = modeling.dropout(attention_output, hidden_dropout_prob)
            prev_output = attention_output
            # attention_output = modeling.layer_norm(attention_output + layer_input)
    tf.logging.info("prev_output shape:%s"%(str(get_shape_list(prev_output))))
    # prev_output shape [batch_size, seq_length, hidden_size]
    # keyword_scores shape [batch_size, seq_length]
    keyword_scores = tf.layers.dense(
        prev_output,
        1,
        activation=None,
        kernel_initializer=create_initializer(0.02),
    )
    tf.logging.info("keyword_scores shape:%s"%(str(get_shape_list(keyword_scores))))
    keyword_scores = tf.reshape(keyword_scores, [batch_size, seq_length])
    # keyword_scores = tf.reduce_sum(prev_output)
    # keyword_scores shape: [batch_size, seq_length]
    mask_adder = (1.0 - tf.cast(input_mask, tf.float32)) * -10000.0
    keyword_scores += mask_adder
    kw_mask_adder = (1.0 - tf.cast(keyword_mask, tf.float32))*-10.0
    keyword_scores += kw_mask_adder
    keyword_probs = tf.nn.softmax(keyword_scores)
    tf.logging.info("mask_adder shape:%s"%(get_shape_list(mask_adder)))
    tf.logging.info("keyword_probs shape:%s"%(get_shape_list(mask_adder)))
    keyword_idx = tf.math.argmax(keyword_probs, axis=1) # [batch_size, 1]
    tf.logging.info("keyword_idx shape:%s"%(str(get_shape_list(keyword_idx))))
    onehot_vec = tf.one_hot(keyword_idx, depth=model_config.max_seq_length) # [batch_size, seq_length, 1]
    onehot_vec_shape = get_shape_list(onehot_vec)
    tf.logging.info("onehot_vec_shape:%s"%(onehot_vec_shape))
    keyword_weight = tf.reshape(onehot_vec, [batch_size, seq_length, 1])
    keyword_vec = tf.reduce_sum(keyword_weight * sequence_output, axis=1)
    tf.logging.info("kwyword_vec shape:%s"%(get_shape_list(keyword_vec)))

    negword_idx = tf.random.uniform(shape=[batch_size,1], minval=0, maxval=model_config.max_seq_length, dtype=tf.int32)
    negword_weights = tf.reshape(tf.one_hot(negword_idx, depth=model_config.max_seq_length), [batch_size, seq_length, 1])
    neg_vec_1 = tf.reduce_sum(negword_weights*sequence_output, axis=1)


    negword_idx = tf.random.uniform(shape=[batch_size,1], minval=0, maxval=model_config.max_seq_length, dtype=tf.int32)
    negword_weights = tf.reshape(tf.one_hot(negword_idx, depth=model_config.max_seq_length), [batch_size, seq_length, 1])
    neg_vec_2 = tf.reduce_sum(negword_weights*sequence_output, axis=1)
    

    with tf.variable_scope("loss"):
        keyword_representation = keyword_vec
        negword_representation = (neg_vec_1 + neg_vec_2)/2
        text_representation = tf.reduce_mean(sequence_output, axis=1)
        # cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)
        # loss = cosine_loss(keyword_representation, text_representation)
        normalize_a = tf.math.l2_normalize(keyword_representation,axis=-1)
        normalize_b = tf.math.l2_normalize(text_representation,axis=-1)
        normalize_c = tf.math.l2_normalize(negword_representation,axis=-1)
        
        loss_1 = tf.reduce_sum(tf.losses.cosine_distance(normalize_a, normalize_b, axis=-1))
        loss_2 = -tf.reduce_sum(tf.losses.cosine_distance(normalize_b, normalize_b, axis=-1))
        loss = loss_1 + loss_2
        return (loss, text_representation, keyword_probs)  


def model_fn_builder(model_config,
                     num_labels,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     embedding_table_value=None,
                     embedding_table_trainable=True,
                     use_one_hot_embeddings=False):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        keyword_mask = features["keyword_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

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

        (total_loss, text_representation, keyword_probs) = create_model(model_config,
                                                                       is_training,
                                                                       input_ids,
                                                                       input_mask,
                                                                       keyword_mask,
                                                                       segment_ids,
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
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            log_hook = tf.train.LoggingTensorHook({"total_loss":total_loss}, every_n_iter=10)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[log_hook],
                scaffold=scaffold)
        elif mode == tf.estimator.ModeKeys.EVAL:
            #def metric_fn(per_example_loss, label_ids, logits):
            #    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            #    accuracy = tf.metrics.accuracy(
            #        labels=label_ids, predictions=predictions)
            #    loss = tf.metrics.mean(values=per_example_loss)
            #    return {
            #        "eval_accuracy": accuracy,
            #        "eval_loss": loss,
            #    }

            #eval_metrics = (metric_fn,
            #                [per_example_loss, label_ids, logits])
            #predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            #eval_accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions)
            #eval_loss = tf.metrics.mean(values=per_example_loss)
            #output_spec = tf.estimator.EstimatorSpec(
            #    mode=mode,
            #    loss=total_loss,
            #    eval_metric_ops={"eval_accuracy":eval_accuracy, "eval_loss":eval_loss},
            #    scaffold=scaffold)
            pass
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"text_representation": text_representation, "input_ids":input_ids, "keyword_probs":keyword_probs},
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
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    model_config = ModelConfig.from_json_file(FLAGS.config_file)
    if FLAGS.max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, model_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = TextProcessor(labels=["1","2"])
    label_list = processor.get_labels()

    tokenizer = tokenization.Tokenizer(
        vocab_file=FLAGS.vocab_file, stop_words_file=FLAGS.stop_words_file, use_pos=FLAGS.use_pos)
    tf.logging.info("model_config vocab_size:%d, tokenizer.vocab_size:%d"%(model_config.vocab_size, tokenizer.vocab_size))
    assert(model_config.vocab_size == tokenizer.vocab_size)

    if FLAGS.embedding_table is not None:
        embedding_table = load_embedding_table(FLAGS.embedding_table)
    else:
        embedding_table = None

    assert(len(tokenizer.vocab) == embedding_table.shape[0])

    #train_examples = processor.get_train_examples(FLAGS.train_data)
    train_examples = None
    num_train_steps = FLAGS.num_train_steps
    num_warmup_steps = FLAGS.num_warmup_steps
    #if FLAGS.do_train:
    #    train_examples = processor.get_train_examples(FLAGS.train_data)
    #    num_train_steps = int(
    #        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    #    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=6,
        log_step_count_steps=100)

    model_fn = model_fn_builder(
        model_config=model_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
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
        #train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        #file_based_convert_examples_to_features(
        #    train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        #tf.logging.info("***** Running training *****")
        #tf.logging.info("  Num examples = %d", len(train_examples))
        #tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        #tf.logging.info("  Num steps = %d", num_train_steps)
        train_file = FLAGS.train_data
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        pass
        #eval_examples = processor.get_dev_examples(FLAGS.eval_data)
        #num_actual_eval_examples = len(eval_examples)
        #eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        #file_based_convert_examples_to_features(
        #    eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        #tf.logging.info("***** Running evaluation *****")
        #tf.logging.info(" Num examples = %d", num_actual_eval_examples)
        #tf.logging.info(" Batch size = %d", FLAGS.eval_batch_size)

        #eval_input_fn = file_based_input_fn_builder(
        #    input_file=eval_file,
        #    seq_length=FLAGS.max_seq_length,
        #    is_training=False,
        #    drop_remainder=False)

        #eval_steps = None
        #result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        #output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        #with tf.gfile.GFile(output_eval_file, "w") as writer:
        #    tf.logging.info("***** Eval results *****")
        #    for key in sorted(result.keys()):
        #        tf.logging.info("  %s = %s", key, str(result[key]))
        #        writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.pred_data)
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file)

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
                    
                #for i, idx in enumerate(sorted_keyword_probs):
                #    top_keyword_ids.append(input_ids[idx])
                #    top_keyword_probs.append(keyword_probs[idx])
                #    if i >= 5:
                #        break
                top_keywords = tokenizer.convert_ids_to_tokens(top_keyword_ids)
                output_line = "\t".join(kw + ":" + str(prob) for kw,prob in zip(top_keywords, top_keyword_probs)) + "\n"
                writer.write(output_line)
                words = tokenizer.convert_ids_to_tokens(input_ids)
                check_line = "\t".join(w + ":" + str(prob) for w, prob in zip(words, keyword_probs)) + "\n"
                writer.write(check_line)
                num_written_lines += 1
        print("num_writen_lines:%d,num_actual_predict_examples:%d"%(num_written_lines, num_actual_predict_examples))
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="[%(asctime)s-%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

