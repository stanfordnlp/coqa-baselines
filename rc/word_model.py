# -*- coding: utf-8 -*-
"""
Module to handle word vectors and initializing embeddings.
"""

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from utils import constants as Constants
from utils.timer import Timer
from collections import Counter


################################################################################
# WordModel Class #
################################################################################

class GloveModel(object):

    def __init__(self, filename):
        self.word_vecs = {}
        self.vocab = []
        with open(filename, 'r') as input_file:
            for line in input_file.readlines():
                splitLine = line.split(' ')
                w = splitLine[0]
                self.word_vecs[w] = np.array([float(val) for val in splitLine[1:]])
                self.vocab.append(w)
        self.vector_size = len(self.word_vecs[w])

    def word_vec(self, word):
        return self.word_vecs[word]


class WordModel(object):
    """Class to get pretrained word vectors for a list of sentences. Can be used
    for any pretrained word vectors.
    """

    def __init__(self, embed_size=None, filename=None, embed_type='glove', top_n=None, additional_vocab=Counter()):
        if filename is None:
            if embed_size is None:
                raise Exception('Either embed_file or embed_size needs to be specified.')
            self.embed_size = embed_size
            self._model = None
        else:
            self.set_model(filename, embed_type)
            self.embed_size = self._model.vector_size

        # padding: 0
        self.vocab = {Constants._UNK_TOKEN: 1}
        if self._model is not None:
            for i, key in enumerate(self._model.vocab):
                if (top_n is not None) and (i >= top_n):
                    break
                self.vocab[key] = len(self.vocab) + 1

        n_added = 0
        for w, count in additional_vocab.most_common():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab) + 1
                n_added += 1
                if n_added <= 10:
                    print('Added word: {} (train_freq = {})'.format(w, count))
        print('Added {} words to the vocab in total.'.format(n_added))

        self.vocab_size = len(self.vocab) + 1
        self.word_vecs = np.random.rand(self.vocab_size, self.embed_size) * 0.2 - 0.1
        for word in self.vocab:
            idx = self.vocab[word]
            if word in self._model.vocab:
                self.word_vecs[idx] = self._model.word_vec(word)

    def set_model(self, filename, embed_type='glove'):
        timer = Timer('Load {}'.format(filename))
        if embed_type == 'glove':
            self._model = GloveModel(filename)
        else:
            self._model = KeyedVectors.load_word2vec_format(filename, binary=True
                                                            if embed_type == 'word2vec' else False)
        print('Embeddings: vocab = {}, embed_size = {}'.format(len(self._model.vocab), self._model.vector_size))
        timer.finish()

    def get_vocab(self):
        return self.vocab

    def get_word_vecs(self):
        return self.word_vecs
