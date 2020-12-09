import tensorflow as tf
import collections
import csv
import json
import re
from tqdm import tqdm
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


class InputExample(object):
    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 keyword_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.keyword_mask = keyword_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class PairInputFeatures(object):
    def __init__(self,
                 input_ids_a,
                 input_mask_a,
                 segment_ids_a,
                 input_ids_b=None,
                 input_mask_b=None,
                 segment_ids_b=None,
                 label=None):
        # label: float
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.segment_ids_b = segment_ids_b
        self.label = label
    

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

            #if len(line) > 2:
            #    text_b = tokenization.convert_to_unicode(line[1])
            #else:
            #    text_b = None
            text_b = None
            if set_type == "test":
                label = "1"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PairTextProcessor(object):
    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path))

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path))

    def get_test_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), pred_mode=True)

    def get_encode_examples(self, data_path):
        return self._create_encode_examples(
            self._read_tsv(data_path))

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, pred_type=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i == 0:
            #    continue
            guid = "0"
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            if pred_type == True:
                label = 0.0
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    def _create_encode_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            #if i == 0:
            #    continue
            guid = "0"
            text_a = tokenization.convert_to_unicode(line[0])
            label = 0.0
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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

    tokens_a,pos_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    assert len(tokens_a) == len(pos_a)
    if example.text_b:
        tokens_b, pos_b = tokenizer.tokenize(example.text_b)

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
            pos_a = pos_a[0:max_seq_length]

    tokens = tokens_a
    #pos_mask = pos_a
    #segment_ids = []
    #for token in tokens_a:
    #   tokens.append(token)
    #    #segment_ids.append(0)

    if tokens_b:
        tokens.extend(tokens_b)
        #for token in tokens_b:
        #    tokens.append(token)
        #    segment_ids.append(1)
    assert len(tokens_a) == len(pos_a)
    input_ids,pos_mask = tokenizer.convert_tokens_to_ids(tokens_a, pos_a)
    segment_ids = [0]*len(input_ids)
    if tokens_b:
        input_ids_b, pos_mask_b = tokenizer.convert_tokens_to_ids(tokens_b)
        segment_ids_b = [1]*len(input_ids_b)
        input_ids += input_ids_b
        segment_ids += segment_ids_b
        pos_mask += pos_mask_b

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        pos_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(pos_mask) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        vocab_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tf.logging.info("ids_token: %s" % " ".join(vocab_tokens))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("pos_mask:%s" % " ".join([str(x) for x in pos_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        keyword_mask=pos_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_single_pairexample(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `PairInputFeatures`."""

    tokens_a, pos_a = tokenizer.tokenize(example.text_a)
    tokens_b, pos_b = tokenizer.tokenize(example.text_b)
    assert len(tokens_a) == len(pos_a)
    assert len(tokens_b) == len(pos_b)
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0: max_seq_length]
        pos_a = pos_a[0:max_seq_length]

    if len(tokens_b) > max_seq_length:
        tokens_b = tokens_b[0: max_seq_length]
        pos_b = pos_b[0:max_seq_length]

    input_ids_a,pos_mask_a = tokenizer.convert_tokens_to_ids(tokens_a, pos_a)
    input_ids_b,pos_mask_b = tokenizer.convert_tokens_to_ids(tokens_b, pos_b)

    segment_ids_a = [0]*len(input_ids_a)
    segment_ids_b = [0]*len(input_ids_b)
    input_mask_a = [1]*len(input_ids_a)
    input_mask_b = [1]*len(input_ids_b)

    # Zero-pad up to the sequence length.
    while len(input_ids_a) < max_seq_length:
        input_ids_a.append(0)
        input_mask_a.append(0)
        segment_ids_a.append(0)
        pos_mask_a.append(0)

    while len(input_ids_b) < max_seq_length:
        input_ids_b.append(0)
        input_mask_b.append(0)
        segment_ids_b.append(0)
        pos_mask_b.append(0)

    assert len(input_ids_a) == max_seq_length
    assert len(input_mask_a) == max_seq_length
    assert len(segment_ids_a) == max_seq_length
    assert len(pos_mask_a) == max_seq_length

    assert len(input_ids_b) == max_seq_length
    assert len(input_mask_b) == max_seq_length
    assert len(segment_ids_b) == max_seq_length
    assert len(pos_mask_b) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens_a: %s" % " ".join(
            [tokenization.printable_text(x).encode("utf-8").decode("unicode_escape") for x in tokens_a]))
        tf.logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
        vocab_tokens_a = tokenizer.convert_ids_to_tokens(input_ids_a)
        tf.logging.info("ids_token_a: %s" % " ".join(vocab_tokens_a))
        tf.logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
        tf.logging.info("pos_mask_a:%s" % " ".join([str(x) for x in pos_mask_a]))
        tf.logging.info("segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))
        tf.logging.info("tokens_b: %s" % " ".join(
            [tokenization.printable_text(x).encode("utf-8").decode("unicode_escape") for x in tokens_b]))
        tf.logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
        vocab_tokens_b = tokenizer.convert_ids_to_tokens(input_ids_b)
        tf.logging.info("ids_token_b: %s" % " ".join(vocab_tokens_b))
        tf.logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
        tf.logging.info("pos_mask_b:%s" % " ".join([str(x) for x in pos_mask_b]))
        tf.logging.info("segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))
        tf.logging.info("label: %s" % (str(example.label)))

    feature = PairInputFeatures(
        input_ids_a=input_ids_a,
        input_mask_a=input_mask_a,
        segment_ids_a=segment_ids_a,
        input_ids_b=input_ids_b,
        input_mask_b=input_mask_b,
        segment_ids_b=segment_ids_b,
        label=example.label)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    examples = tqdm(examples)
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
        features["keyword_mask"] = create_int_feature(feature.keyword_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_convert_pairexamples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    examples = tqdm(examples)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_pairexample(ex_index, example, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids_a"] = create_int_feature(feature.input_ids_a)
        features["input_mask_a"] = create_int_feature(feature.input_mask_a)
        features["segment_ids_a"] = create_int_feature(feature.segment_ids_a)
        features["input_ids_b"] = create_int_feature(feature.input_ids_b)
        features["input_mask_b"] = create_int_feature(feature.input_mask_b)
        features["segment_ids_b"] = create_int_feature(feature.segment_ids_b)
        features["label"] = tf.train.Feature(float_list=tf.train.FloatList(value=list([feature.label])))

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
    lines = tqdm(lines)
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


def create_pairexamples_from_tsv_file(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    reader = csv.reader(fp, delimiter="\t")
    lines = []
    for line in reader:
        lines.append(line)
    examples = []
    lines = tqdm(lines)
    for (i, line) in enumerate(lines):
        guid = "0"
        text_a = line[0]
        text_b = line[1]
        text_a = tokenization.convert_to_unicode(text_a)
        text_b = tokenization.convert_to_unicode(text_b)
        label = 0.0 
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    fp.close()
    return examples