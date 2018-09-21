# -*- coding: utf-8 -*-
"""
Module to handle word vectors and initializing embeddings.
"""

import numpy as np
import os
from gensim.models.keyedvectors import KeyedVectors

from .utils import constants as Constants
from .utils.timer import Timer
from collections import Counter


################################################################################
# Constants #
################################################################################

_SMALL_GLOVE = 'glove.6B.50d.txt'
_LARGE_GLOVE = 'glove.840B.300d.txt'
_WORD2VEC = 'GoogleNews-vectors-negative300.bin'
_FASTTEXT = 'crawl-300d-2M.vec'

################################################################################
# WordModel Class #
################################################################################


class WordModel(object):
    """Class to get pretrained word vectors for a list of sentences. Can be used
    for any pretrained word vectors.
    """

    def __init__(self, model, dataset=None, additional_vocab=Counter()):

        if model is None:
            self._model = None
        elif model == 'glove6b':
            self.set_model(_SMALL_GLOVE, dataset=dataset, binary=False)
        elif model == 'glove840b':
            self.set_model(_LARGE_GLOVE, dataset=dataset, binary=False)
        elif model == 'word2vec':
            self.set_model(_WORD2VEC, dataset=dataset, binary=True)
        elif model == 'fasttext':
            self.set_model(_FASTTEXT, dataset=dataset, binary=False)
        else:
            raise ValueError('embed_type = {} not recognized.'.format(model))

        # padding: 0
        self.vocab = {Constants._UNK_TOKEN: 1}
        for key in self._model.vocab:
            self.vocab[key] = len(self.vocab) + 1
        n_added = 0
        for w, count in additional_vocab.most_common():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab) + 1
                n_added += 1
                if n_added <= 10:
                    print('Added word: {} (train_freq = {})'.format(w, count))
        print('Added {} words to the vocab in total.'.format(n_added))

        self.embed_size = self.get_embed_size()
        self.vocab_size = len(self.vocab) + 1
        self.word_vecs = np.random.rand(self.vocab_size, self.embed_size) * 0.2 - 0.1
        for word in self.vocab:
            idx = self.vocab[word]
            if word in self._model.vocab:
                self.word_vecs[idx] = self._model.word_vec(word, use_norm=False)

    def set_model(self, filename, dataset=None, binary=False):
        if (dataset is not None) and (os.path.isfile(Constants._WORDVEC_DIR + dataset + '.' + filename)):
            filename = dataset + '.' + filename
        timer = Timer('Load {}'.format(filename))
        self._model = KeyedVectors.load_word2vec_format(Constants._WORDVEC_DIR + filename, binary=binary)
        timer.finish()

    def get_embed_size(self):
        if self._model is None:
            raise Exception('ERROR: Model not yet specified')
        return self._model.vector_size

    def get_vocab(self):
        return self.vocab

    def get_word_vecs(self):
        return self.word_vecs
