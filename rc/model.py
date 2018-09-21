import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .word_model import WordModel, CharModel
from .utils.eval_utils import compute_eval_metric
from .layers import multi_nll_loss
from .utils import constants as Constants
from collections import Counter
from .models.drqa import DrQA


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set):
        # Book-keeping.
        self.config = config
        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            print('Train vocab: {}'.format(len(train_set.vocab)))
            vocab = Counter()
            for w in train_set.vocab:
                if train_set.vocab[w] >= config['min_freq']:
                    vocab[w] = train_set.vocab[w]
            print('Pruned train vocab: {}'.format(len(vocab)))
            # Building network.
            word_model = WordModel(self.config['embed_type'],
                                   dataset=self.config['dataset'],
                                   additional_vocab=vocab)
            if self.config['char_embed'] > 0:
                char_model = CharModel(self.config['char_embed'], self.config['char_vocab'], train_set)
                self._init_new_network(train_set, word_model, char_model)
            else:
                self._init_new_network(train_set, word_model)

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()
        print('#Parameters = {}\n'.format(num_params))

        # Building optimizer.
        self._init_optimizer()

    def init_saved_network(self, saved_dir):
        # TO ADD: 'bow_layers', 'curriculum', 'reinforce_penalty',  'rnn_selector_hidden', 'bow_hidden_size',
        # 'rnn_sent_selector', 'dropout_bow'

        _OVERWRITTEN_ARGUMENTS = ['model', 'rnn_padding', 'embed_type', 'hidden_size', 'num_layers', 'rnn_type',
                                  'concat_rnn_layers', 'question_merge', 'use_qemb', 'f_qem', 'f_pos', 'f_ner',
                                  'sum_loss', 'doc_self_attn', 'char_vocab', 'max_word_len', 'char_embed',
                                  'char_layer', 'filter_height', 'resize_rnn_input', 'span_dependency',
                                  'fix_embeddings', 'dropout_rnn', 'dropout_emb', 'dropout_char',
                                  'dropout_ff', 'dropout_rnn_output', 'variational_dropout', 'word_dropout']

        # Load all saved fields.
        fname = os.path.join(Constants._RESULTS_DIR, saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.char_dict = saved_params['char_dict']
        self.word_dict = saved_params['word_dict']
        self.feature_dict = saved_params['feature_dict']
        self.config['num_features'] = len(self.feature_dict)
        self.state_dict = saved_params['state_dict']
        for k in _OVERWRITTEN_ARGUMENTS:
            if saved_params['config'][k] != self.config[k]:
                print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
                self.config[k] = saved_params['config'][k]

        if self.config['embed_type'] in ['word2vec', 'glove840b', 'fasttext']:
            embed_size = 300
        elif self.config['embed_type'] == 'glove6b':
            embed_size = 50
        else:
            raise ValueError('embed_type = {} not recognized.'.format(self.config['embed_type']))

        w_embedding = self._init_embedding(len(self.word_dict) + 1, embed_size)  # Should load saved below.

        if self.char_dict is None:
            self.network = DrQA(self.config, w_embedding)
        else:
            c_embedding = self._init_embedding(len(self.char_dict) + 1, self.config["char_embed"])
            self.network = DrQA(self.config, w_embedding, c_embedding)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_new_network(self, train_set, word_model, char_model=None):
        self.feature_dict = self._build_feature_dict(train_set)
        self.config['num_features'] = len(self.feature_dict)
        self.word_dict = word_model.get_vocab()
        w_embedding = self._init_embedding(word_model.vocab_size, word_model.embed_size,
                                           pretrained_vecs=word_model.get_word_vecs())

        if char_model is not None:
            self.char_dict = char_model.get_vocab()
            self.network = DrQA(self.config, w_embedding, self._init_embedding(
                                                              char_model.vocab_size, char_model.embed_size))
        else:
            self.char_dict = None
            self.network = DrQA(self.config, w_embedding)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings - character or word embeddings.
        """
        embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0,
                                 _weight=torch.from_numpy(pretrained_vecs).float()
                                 if pretrained_vecs is not None else None)
        return embedding

    def _build_feature_dict(self, train_set):
        feature_dict = {}
        if self.config['f_qem']:
            feature_dict['f_qem_cased'] = len(feature_dict)
            feature_dict['f_qem_uncased'] = len(feature_dict)

        if self.config['f_pos']:
            pos_tags = set()
            for ex in train_set:
                for context in ex['evidence']:
                    assert 'pos' in context
                    pos_tags |= set(context['pos'])
            print('{} pos tags: {}'.format(len(pos_tags), str(pos_tags)))
            for pos in pos_tags:
                feature_dict['f_pos={}'.format(pos)] = len(feature_dict)

        if self.config['f_ner']:
                ner_tags = set()
                for ex in train_set:
                    for context in ex['evidence']:
                        assert 'ner' in context
                        ner_tags |= set(context['ner'])
                print('{} ner tags: {}'.format(len(ner_tags), str(ner_tags)))
                for ner in ner_tags:
                    feature_dict['f_ner={}'.format(ner)] = len(feature_dict)

        print('# features: {}'.format(len(feature_dict)))
        return feature_dict

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])

    def predict(self, ex, update=True, out_predictions=False):
        # Train/Eval mode
        self.network.train(update)
        # Run forward
        res = self.network(ex)
        score_s, score_e = res['score_s'], res['score_e']

        span_loss_val, chunk_loss_val = -1, -1
        # Loss cannot be computed for test-time as we may not have targets
        if update:
            # Compute loss and accuracies
            span_loss = self.compute_span_loss(score_s, score_e, res['targets'])
            span_loss_val = span_loss.item()

            if 'chunk_loss' in res:
                loss = span_loss + res['chunk_loss']
                chunk_loss_val = res['chunk_loss'].item()
            else:
                loss = span_loss
                chunk_loss_val = 0.0

            # Clear gradients and run backward
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['grad_clipping'])

            # Update parameters
            self.optimizer.step()

        if (not update) or self.config['predict_train']:
            action_idxs = res.get('action_idxs', [0] * ex['batch_size'])
            predictions, spans = self.extract_predictions(ex, score_s, score_e, action_idxs)
            f1, em = self.evaluate_predictions(predictions, ex['answers'])
        else:
            f1 = em = 0.0
            predictions = None

        output = {'f1': f1, 'em': em,
                  'span_loss': span_loss_val,
                  'chunk_loss': chunk_loss_val,
                  'chunk_target_acc': res['chunk_target_acc'],
                  'chunk_any_acc': res['chunk_any_acc']}
        if out_predictions:
            output['predictions'] = predictions
            output['spans'] = spans
        return output

    def compute_span_loss(self, score_s, score_e, targets):
        assert targets.size(0) == score_s.size(0) == score_e.size(0)
        if self.config['sum_loss']:
            loss = multi_nll_loss(score_s, targets[:, :, 0]) + multi_nll_loss(score_e, targets[:, :, 1])
        else:
            loss = F.nll_loss(score_s, targets[:, 0]) + F.nll_loss(score_e, targets[:, 1])
        return loss

    def extract_predictions(self, ex, score_s, score_e, action_idxs):
        # Transfer to CPU/normal tensors for numpy ops (and convert log probabilities to probabilities)
        score_s = score_s.exp().squeeze()
        score_e = score_e.exp().squeeze()
        idx = torch.LongTensor([0])
        predictions = []
        spans = []
        for i, action_id in enumerate(action_idxs):
            if action_id >= 0:
                if self.config['predict_raw_text']:
                    prediction, span = self._scores_to_raw_text(ex['raw_evidence_text'][i][action_id],
                                                                ex['offsets'][i][action_id], score_s[idx], score_e[idx])
                else:
                    prediction, span = self._scores_to_text(ex['evidence_text'][i][action_id],
                                                            score_s[idx], score_e[idx])
                idx += 1
            else:
                prediction = ""
            predictions.append(prediction)
            spans.append(span)
        return predictions, spans

    def _scores_to_text(self, text, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return ' '.join(text[s_idx: e_idx + 1]), (int(s_idx), int(e_idx))

    def _scores_to_raw_text(self, raw_text, offsets, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return raw_text[offsets[s_idx][0]: offsets[e_idx][1]], (offsets[s_idx][0], offsets[e_idx][1])

    def evaluate_predictions(self, predictions, answers):
        f1_score = compute_eval_metric('f1', predictions, answers, dataset=self.config['dataset'])
        em_score = compute_eval_metric('em', predictions, answers, dataset=self.config['dataset'])
        return f1_score, em_score

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'word_dict': self.word_dict,
            'char_dict': self.char_dict,
            'feature_dict': self.feature_dict,
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(Constants._RESULTS_DIR, dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')
