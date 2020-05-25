from __future__ import absolute_import
import tensorflow as tf
import json
import copy


class ModelConfig(object):

    def __init__(self, 
                 vocab_size=None,
                 embedding_size=200,
                 hidden_size=2400,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.0,
                 token_type_size=2,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 max_seq_length=512,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 intermediate_size=1200,
                 hidden_act="gelu",
                 ):
        """模型配置
        hidden_size = embedding_size * num_attention_heads
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.token_type_size = token_type_size
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_length=max_seq_length
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

    @classmethod
    def from_dict(cls, json_object):
        config = ModelConfig()
        for key,value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as fp:
            text = fp.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
