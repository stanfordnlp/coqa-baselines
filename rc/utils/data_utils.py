# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import io
import torch
import numpy as np

from collections import Counter, defaultdict
from torch.utils.data import Dataset
from . import constants as Constants
from .timer import Timer


################################################################################
# Dataset Prep #
################################################################################

def prepare_datasets(config):
    train_set = None if config['trainset'] is None else CoQADataset(config['trainset'], config)
    dev_set = None if config['devset'] is None else CoQADataset(config['devset'], config)
    test_set = None if config['testset'] is None else CoQADataset(config['testset'], config)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}

################################################################################
# Dataset Classes #
################################################################################


class CoQADataset(Dataset):
    """SQuAD dataset."""

    def __init__(self, filename, config):
        timer = Timer('Load %s' % filename)
        self.filename = filename
        self.config = config
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        dataset = read_json(filename)
        for paragraph in dataset['data']:
            history = []
            for qas in paragraph['qas']:
                qas['paragraph_id'] = len(self.paragraphs)
                temp = []
                n_history = len(history) if config['n_history'] < 0 else min(config['n_history'], len(history))
                if n_history > 0:
                    for i, (q, a) in enumerate(history[-n_history:]):
                        d = n_history - i
                        temp.append('<Q{}>'.format(d))
                        temp.extend(q)
                        temp.append('<A{}>'.format(d))
                        temp.extend(a)
                temp.append('<Q>')
                temp.extend(qas['annotated_question']['word'])
                history.append((qas['annotated_question']['word'], qas['annotated_answer']['word']))
                qas['annotated_question']['word'] = temp
                self.examples.append(qas)
                question_lens.append(len(qas['annotated_question']['word']))
                paragraph_lens.append(len(paragraph['annotated_context']['word']))
                for w in qas['annotated_question']['word']:
                    self.vocab[w] += 1
                for w in paragraph['annotated_context']['word']:
                    self.vocab[w] += 1
                for w in qas['annotated_answer']['word']:
                    self.vocab[w] += 1
            self.paragraphs.append(paragraph)
        print('Load {} paragraphs, {} examples.'.format(len(self.paragraphs), len(self.examples)))
        print('Paragraph length: avg = %.1f, max = %d' % (np.average(paragraph_lens), np.max(paragraph_lens)))
        print('Question length: avg = %.1f, max = %d' % (np.average(question_lens), np.max(question_lens)))
        timer.finish()

    def __len__(self):
        return 50 if self.config['debug'] else len(self.examples)

    def __getitem__(self, idx):
        qas = self.examples[idx]
        paragraph = self.paragraphs[qas['paragraph_id']]
        question = qas['annotated_question']
        answers = [qas['answer']]
        if 'additional_answers' in qas:
            answers = answers + qas['additional_answers']

        sample = {'id': (paragraph['id'], qas['turn_id']),
                  'question': question,
                  'answers': answers,
                  'evidence': paragraph['annotated_context'],
                  'targets': qas['answer_span']}

        if self.config['predict_raw_text']:
            sample['raw_evidence'] = paragraph['context']
        return sample


################################################################################
# Read & Write Helper Functions #
################################################################################


def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with io.open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def log_json(data, filename, mode='w', encoding='utf-8'):
    with io.open(filename, mode, encoding=encoding) as outfile:
        outfile.write(json.dumps(data, indent=4, ensure_ascii=False))


def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_processed_file_contents(file_path, encoding="utf-8"):
    contents = get_file_contents(file_path, encoding=encoding)
    return contents.strip()

################################################################################
# DataLoader Helper Functions #
################################################################################


def sanitize_input(sample_batch, config, vocab, feature_dict, training=True):
    """
    Reformats sample_batch for easy vectorization.
    Args:
        sample_batch: the sampled batch, yet to be sanitized or vectorized.
        vocab: word embedding dictionary.
        feature_dict: the features we want to concatenate to our embeddings.
        train: train or test?
    """
    sanitized_batch = defaultdict(list)
    for ex in sample_batch:
        question = ex['question']['word']
        evidence = ex['evidence']['word']
        offsets = ex['evidence']['offsets']

        processed_q, processed_e = [], []
        for w in question:
            processed_q.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])
        for w in evidence:
            processed_e.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])

        # Append relevant index-structures to batch
        sanitized_batch['question'].append(processed_q)
        sanitized_batch['evidence'].append(processed_e)

        if config['predict_raw_text']:
            sanitized_batch['raw_evidence_text'].append(ex['raw_evidence'])
            sanitized_batch['offsets'].append(offsets)
        else:
            sanitized_batch['evidence_text'].append(evidence)

        # featurize evidence document:
        sanitized_batch['features'].append(featurize(ex['question'], ex['evidence'], feature_dict))
        sanitized_batch['targets'].append(ex['targets'])
        sanitized_batch['answers'].append(ex['answers'])
        if 'id' in ex:
            sanitized_batch['id'].append(ex['id'])
    return sanitized_batch


def vectorize_input(batch, config, training=True, device=None):
    """
    - Vectorize question and question mask
    - Vectorize evidence documents, mask and features
    - Vectorize target representations
    """
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch['question'])

    # Initialize all relevant parameters to None:
    targets = None

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for q in batch['question']])
    xq = torch.LongTensor(batch_size, max_q_len).fill_(0)
    xq_mask = torch.ByteTensor(batch_size, max_q_len).fill_(1)
    for i, q in enumerate(batch['question']):
        xq[i, :len(q)].copy_(torch.LongTensor(q))
        xq_mask[i, :len(q)].fill_(0)

    # Part 2: Document Words
    max_d_len = max([len(d) for d in batch['evidence']])
    xd = torch.LongTensor(batch_size, max_d_len).fill_(0)
    xd_mask = torch.ByteTensor(batch_size, max_d_len).fill_(1)
    xd_f = torch.zeros(batch_size, max_d_len, config['num_features']) if config['num_features'] > 0 else None

    # 2(a): fill up DrQA section variables
    for i, d in enumerate(batch['evidence']):
        xd[i, :len(d)].copy_(torch.LongTensor(d))
        xd_mask[i, :len(d)].fill_(0)
        if config['num_features'] > 0:
            xd_f[i, :len(d)].copy_(batch['features'][i])

    # Part 3: Target representations
    if config['sum_loss']:  # For sum_loss "targets" acts as a mask rather than indices.
        targets = torch.ByteTensor(batch_size, max_d_len, 2).fill_(0)
        for i, _targets in enumerate(batch['targets']):
            for s, e in _targets:
                targets[i, s, 0] = 1
                targets[i, e, 1] = 1
    else:
        targets = torch.LongTensor(batch_size, 2)
        for i, _target in enumerate(batch['targets']):
            targets[i][0] = _target[0]
            targets[i][1] = _target[1]

    torch.set_grad_enabled(training)
    example = {'batch_size': batch_size,
               'answers': batch['answers'],
               'xq': xq.to(device) if device else xq,
               'xq_mask': xq_mask.to(device) if device else xq_mask,
               'xd': xd.to(device) if device else xd,
               'xd_mask': xd_mask.to(device) if device else xd_mask,
               'xd_f': xd_f.to(device) if device else xd_f,
               'targets': targets.to(device) if device else targets}

    if config['predict_raw_text']:
        example['raw_evidence_text'] = batch['raw_evidence_text']
        example['offsets'] = batch['offsets']
    else:
        example['evidence_text'] = batch['evidence_text']
    return example


def featurize(question, document, feature_dict):
    doc_len = len(document['word'])
    features = torch.zeros(doc_len, len(feature_dict))
    q_cased_words = set([w for w in question['word']])
    q_uncased_words = set([w.lower() for w in question['word']])
    for i in range(doc_len):
        d_word = document['word'][i]
        if 'f_qem_cased' in feature_dict and d_word in q_cased_words:
            features[i][feature_dict['f_qem_cased']] = 1.0
        if 'f_qem_uncased' in feature_dict and d_word.lower() in q_uncased_words:
            features[i][feature_dict['f_qem_uncased']] = 1.0
        if 'pos' in document:
            f_pos = 'f_pos={}'.format(document['pos'][i])
            if f_pos in feature_dict:
                features[i][feature_dict[f_pos]] = 1.0
        if 'ner' in document:
            f_ner = 'f_ner={}'.format(document['ner'][i])
            if f_ner in feature_dict:
                features[i][feature_dict[f_ner]] = 1.0
    return features
