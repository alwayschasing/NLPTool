import tensorflow as tf
import collections
import csv
import json
import re
from model import tokenization

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    def get_train_examples(self, data_path):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_path):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_path):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class TextProcessor(DataProcessor):
    def __init__(self, labels=[]):
        self.labels=labels 
        
    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path),
            "dev_matched")

    def get_test_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), "test")

    def get_labels(self):
        """See base class."""
        #return ["1", "2", "3","4"]
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "0"
            text_a = tokenization.convert_to_unicode(line[0])

            if len(line) > 2:
                text_b = tokenization.convert_to_unicode(line[1])
            else:
                text_b = None
            if set_type == "test":
                label = "1"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[0: max_seq_length]

    tokens = []
    #segment_ids = []
    for token in tokens_a:
        tokens.append(token)
    #    #segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
    #        #segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    segment_ids = [0]*len(input_ids)
    if tokens_b:
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
        segment_ids_b = [1]*len(input_ids_b)
        input_ids += input_ids_b
        segment_ids += segment_ids_b

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x).encode("utf-8").decode("unicode_escape") for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        vocab_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tf.logging.info("ids_token: %s" % " ".join(vocab_tokens))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def create_examples_from_tsv_file(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    reader = csv.reader(fp, delimiter="\t")
    lines = []
    for line in reader:
        lines.append(line)

    examples = []
    for (i, line) in enumerate(lines):
        guid = "0"
        text_a = tokenization.convert_to_unicode(line[0])
        text_b = None
        label = "1"
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def create_examples_from_json_file(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    # reader = csv.reader(fp, delimiter="\t")
    lines = []
    for line in fp:
        lines.append(line)

    examples = []
    for (i, line) in enumerate(lines):
        guid = "0"
        json_data = json.loads(line.strip())
        title = re.sub("[\r\n]", " ", json_data["title"])
        content = re.sub("[\r\n]", " ", json_data["content"]) 
        text_a = tokenization.convert_to_unicode(title + " " + content)
        text_b = None
        label = "1"
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
