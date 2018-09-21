# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import os
import io
import torch
import torch.nn as nn
import numpy as np

from collections import Counter, defaultdict
from torch.utils.data import Dataset
from . import constants as Constants
from .timer import Timer


################################################################################
# Dataset Prep #
################################################################################

def prepare_datasets(config):
    """Eventually: will be more complex if we use more than the TriviaQA dataset.
    """
    train_set = CoQADataset(config['trainset'], config)
    dev_set = CoQADataset(config['devset'], config)
    test_set = CoQADataset(config['testset'], config)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}

################################################################################
# Dataset Classes #
################################################################################


class CoQADataset(Dataset):
    """SQuAD dataset."""

    def __init__(self, filename, config):
        timer = Timer('Load %s' % filename)
        self.root_dir = Constants._DATA_ROOTDIR + config['dataset'] + '/'
        self.filename = filename
        self.config = config
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        for fn in filename.split(';'):
            dataset = read_json(self.root_dir + fn)
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
        evidence = [paragraph['annotated_context']]

        targets = [qas['answer_span']]
        chunk_target = 0
        sample = {'id': (paragraph['id'], qas['turn_id']),
                  'question': question,
                  'answers': answers,
                  'evidence': evidence,
                  'targets': targets,
                  'chunk_target': chunk_target}

        if self.config['predict_raw_text']:
            sample['raw_evidence'] = [paragraph['context']]
        return sample


def log_error_ex(dirname, idx, correct, evidence, question, answers):
    data_file = os.path.join(Constants._RESULTS_DIR, dirname, Constants._ERROR_EXS)

    entry = {'idx': idx,
             'correct': correct,
             'question': question,
             'evidence': evidence,
             'answers': answers}

    feed = []
    if not os.path.isfile(data_file):
        feed += [entry]
    else:
        feed = read_json(data_file) + [entry]
    log_json(feed, data_file)

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


def sanitize_input(sample_batch, config, vocab, c_vocab, feature_dict, training=True):
    """
    Reformats sample_batch for easy vectorization.
    Args:
        sample_batch: the sampled batch, yet to be sanitized or vectorized.
        vocab: word embedding dictionary.
        c_vocab: character embedding dictionary (if applicable).
        feature_dict: the features we want to concatenate to our embeddings.
        train: train or test?
    """
    use_chars = config['char_embed'] > 0
    max_word_len = config['max_word_len']
    sanitized_batch = defaultdict(list)
    uniqueWords = {Constants._UNK_TOKEN: 0}  # For tracking unique words, used in character embedding layer.

    for ex in sample_batch:

        if training and ex['chunk_target'] == -1:  # If the target doesn't appear in the document, skip.
            continue

        question = ex['question']['word']
        evidence = [context['word'] for context in ex['evidence']]
        offsets = [context['offsets'] for context in ex['evidence']]

        processed_q, processed_e = [], []
        processed_qc, processed_ec, c_embed = [], [], []
        # Populate word and char index structures for question then document:
        for w in question:
            processed_q.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])
            if use_chars:
                w_chars = [c_vocab[c] if c in c_vocab else c_vocab[Constants._UNK_TOKEN] for c in w[:max_word_len]]
                w_str = str(w_chars)
                if w_str not in uniqueWords:
                    c_embed.append(w_chars)
                    uniqueWords[w_str] = len(uniqueWords)
                processed_qc.append(uniqueWords[w_str])

        for chunk in evidence:
            processed_chunk, processed_char_chunk = [], []
            for w in chunk:
                processed_chunk.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])
                if use_chars:
                    w_chars = [c_vocab[c] if c in c_vocab else c_vocab[Constants._UNK_TOKEN] for c in w[:max_word_len]]
                    w_str = str(w_chars)
                    if w_str not in uniqueWords:
                        c_embed.append(w_chars)
                        uniqueWords[w_str] = len(uniqueWords)
                    processed_char_chunk.append(uniqueWords[w_str])
            processed_e.append(processed_chunk)
            processed_ec.append(processed_char_chunk)

        # Append relevant index-structures to batch
        sanitized_batch['question'].append(processed_q)
        sanitized_batch['evidence'].append(processed_e)
        if use_chars:
            sanitized_batch['question_chars'].append(processed_qc)
            sanitized_batch['evidence_chars'].append(processed_ec)
            sanitized_batch['c_embed'] += c_embed

        if config['predict_raw_text']:
            sanitized_batch['raw_evidence_text'].append(ex['raw_evidence'])
            sanitized_batch['offsets'].append(offsets)
        else:
            sanitized_batch['evidence_text'].append(evidence)

        # featurize evidence document:
        sanitized_batch['features'].append([featurize(ex['question'], context, feature_dict)
                                            for context in ex['evidence']])

        sanitized_batch['chunk_targets'].append(ex['chunk_target'])
        sanitized_batch['targets'] += ex['targets']
        sanitized_batch['answers'].append(ex['answers'])
        if 'id' in ex:
            sanitized_batch['id'].append(ex['id'])
    return sanitized_batch


def vectorize_input(batch, config, training=True, device=None):
    """
    (1) Vectorize question and question mask
    (2) Vectorize evidence documents, mask and features
    (3) Vectorize character embeddings, and character layer lookup
    (4) Vectorize target representations
    """
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch['question'])
    num_chunks = [len(evidence) for evidence in batch['evidence']]
    total_num_chunks = sum(num_chunks)

    # Initialize all relevant parameters to None:
    targets = None

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for q in batch['question']])
    xq = torch.LongTensor(total_num_chunks, max_q_len).fill_(0)
    xq_mask = torch.ByteTensor(total_num_chunks, max_q_len).fill_(1)
    chunk_offset = 0
    for i, q in enumerate(batch['question']):
        xq[chunk_offset:chunk_offset + num_chunks[i], :len(q)].copy_(torch.LongTensor(q).expand(num_chunks[i], len(q)))
        xq_mask[chunk_offset:chunk_offset + num_chunks[i], :len(q)].fill_(0)
        chunk_offset += num_chunks[i]

    # Part 2: Document Words
    max_chunk_len = max([len(chunk) for evidence in batch['evidence'] for chunk in evidence])
    xd = torch.LongTensor(total_num_chunks, max_chunk_len).fill_(0)
    xd_mask = torch.ByteTensor(total_num_chunks, max_chunk_len).fill_(1)
    xd_nchunks = torch.LongTensor(num_chunks)
    xd_f = torch.zeros(total_num_chunks, max_chunk_len, config['num_features']) if config['num_features'] > 0 else None

    # 2(a): fill up DrQA section variables
    chunk_offset = 0
    for i, evidence in enumerate(batch['evidence']):
        for k, chunk in enumerate(evidence):
            xd[chunk_offset, :len(chunk)].copy_(torch.LongTensor(chunk))
            xd_mask[chunk_offset, :len(chunk)].fill_(0)
            if config['num_features'] > 0:
                xd_f[chunk_offset, :len(chunk)].copy_(batch['features'][i][k])
            chunk_offset += 1

    # Part 3: Character Embeddings and Character Layer
    if config['char_embed'] > 0:

        # Format character index representations, containing each unique word in batch encoded only once
        n_unique_words = len(batch['c_embed'])
        word_lengths = [len(w) for w in batch['c_embed']]
        max_w_len = max(word_lengths)
        c_emb = torch.LongTensor(n_unique_words, max_w_len).fill_(0)
        c_emb_mask = torch.LongTensor(n_unique_words, max_w_len).fill_(1)
        for i, len_w in enumerate(word_lengths):
            c_emb[i, :len_w].copy_(torch.LongTensor(batch['c_embed'][i]))
            c_emb_mask[i, :len_w].fill_(0)

        # Question word indices mapped to unique words processed from c_emb
        xqc = torch.LongTensor(batch_size, max_q_len).fill_(0)
        for qi, q in enumerate(batch['question_chars']):
            xqc[qi, :len(q)].copy_(torch.LongTensor(q))

        # Document word indices mapped to unique words processed from c_emb
        xdc = torch.LongTensor(total_num_chunks, max_chunk_len).fill_(0)
        chunk_offset = 0
        for di, d in enumerate(batch['evidence_chars']):  # For document (evidence) in batch
            for ci, chunk in enumerate(d):                # For chunk in document
                xdc[chunk_offset, :len(chunk)].copy_(torch.LongTensor(chunk))
                chunk_offset += 1

        # Lookup table between xdc/xqc and c_emb unique words. (Note index-0 is for 0-padding)
        c_layer_lookup = nn.Embedding(n_unique_words+1, config['char_embed'], padding_idx=0)

    # Part 4: Target representations
    if config['sum_loss']:  # For sum_loss "targets" acts as a mask rather than indices.
        targets = torch.ByteTensor(total_num_chunks, max_chunk_len, 2).fill_(0)
        for i, _targets in enumerate(batch['targets']):
            for s, e in _targets:
                targets[i, s, 0] = 1
                targets[i, e, 1] = 1
    else:
        targets = torch.LongTensor(batch['targets'])             # (total_num_chunks, 2)
    chunk_targets = torch.LongTensor(batch['chunk_targets'])     # (batch,)

    torch.set_grad_enabled(training)
    example = {'batch_size': batch_size,
               'answers': batch['answers'],
               'xq': xq.to(device) if device else xq,
               'xq_mask': xq_mask.to(device) if device else xq_mask,
               'xd': xd.to(device) if device else xd,
               'xd_mask': xd_mask.to(device) if device else xd_mask,
               'xd_nchunks': xd_nchunks.to(device) if device else xd_nchunks,
               'xd_f': xd_f.to(device) if device else xd_f,
               'targets': targets.to(device) if device else targets,
               'chunk_targets': chunk_targets.to(device) if device else chunk_targets}

    if config['char_embed'] > 0:
        example['c_emb'] = c_emb.to(device) if device else c_emb
        example['c_emb_mask'] = c_emb_mask.to(device) if device else c_emb_mask
        example['xqc'] = xqc.to(device) if device else xqc
        example['xdc'] = xdc.to(device) if device else xdc
        example['c_layer_lookup'] = c_layer_lookup.to(device) if device else c_layer_lookup

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
        if 'pos' in document['pos']:
            f_pos = 'f_pos={}'.format(document['pos'][i])
            if f_pos in feature_dict:
                features[i][feature_dict[f_pos]] = 1.0
        if 'ner' in document['ner']:
            f_ner = 'f_ner={}'.format(document['ner'][i])
            if f_ner in feature_dict:
                features[i][feature_dict[f_ner]] = 1.0
    return features
