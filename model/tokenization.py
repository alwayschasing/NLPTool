#!/usr/bin/env python
# -*-encoding=utf-8-*-
import jieba
import collections
import tensorflow as tf
import six
import csv


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type:%s"%(type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type:%s"%(type(text)))
    else:
        raise ValueError("Not running on Python2 or Python3")


def load_vocab(word_vocab_file, skip_head=True):
    vocab = collections.OrderedDict()
    index = 0

    skip_head = True
    with tf.gfile.GFile(word_vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            if skip_head:
                skip_head = False
                continue
            token = token.strip().split(' ')[0]
            vocab[token] = index
            index += 1
    tf.logging.info("vocab max index:%d"%(index-1))
    return vocab


def load_stop_words(stop_words_file):
    stop_words = set()
    index = 0
    with tf.gfile.GFile(stop_words_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            stop_words.add(token)
    return stop_words


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
    return output


class Tokenizer(object):
    def __init__(self, vocab_file, stop_words_file=None):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        if stop_words_file is None:
            self.stop_words = set()
        else:
            self.stop_words = load_stop_words(stop_words_file)
        self.vocab_size = len(self.vocab)

    def tokenize(self,text):
        split_tokens = []
        text = text.strip()
        words = jieba.cut(text)
        useful_words = self.filter_stop_words(words)
        return useful_words

    def filter_stop_words(self,words_list):
        useful_words = []
        for w in words_list:
            if w in self.stop_words:
                print(w.encode("utf-8").decode("unicode_escape"))
                continue
            useful_words.append(w)
        return useful_words

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


if __name__ == "__main__":
    vocab_file = "/search/odin/liruihong/word2vec_embedding/70000-small.txt"
    stop_words_file = "/search/odin/liruihong/word2vec_embedding/cn_stopwords.txt"
    tokenizer = Tokenizer(vocab_file, stop_words_file)
    test_file = "/search/odin/liruihong/article_data/article_test"
    res_file = "test_res"
    # fp = open(test_file,"r", encoding="utf-8") 
    fp = tf.gfile.GFile(test_file, "r")
    reader = csv.reader(fp, delimiter="\t", quotechar=None)
    wfp = open(res_file, "w", encoding="utf-8")
    # lines = fp.readlines()
    lines = []
    for line in reader:
        # print(line)
        lines.append(line)
    for idx,line in enumerate(lines):
        print(type(line))
        line = line[0]
        if idx > 10:
            break
        line = convert_to_unicode(line)
        line_words = tokenizer.tokenize(line)
        #print(line_words)
        res_line = "#".join(line_words) + "\n"
        ids = tokenizer.convert_tokens_to_ids(line_words) 
        words = tokenizer.convert_ids_to_tokens(ids)
        ids = [str(id) for id in ids]
        ids_line = "#".join(ids) + "\n"
        words_line = "#".join(words) + "\n"
        wfp.write(res_line)
        wfp.write(ids_line)
        wfp.write(words_line)
    wfp.close()
