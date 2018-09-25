import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SeqAttnMatch, StackedBRNN, LinearSeqAttn, BilinearSeqAttn
from .layers import weighted_avg, uniform_weights, dropout


class DrQA(nn.Module):
    """Network for the Document Reader module of DrQA."""
    _RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, config, w_embedding):
        """Configuration, word embeddings"""
        super(DrQA, self).__init__()
        # Store config
        self.config = config
        self.w_embedding = w_embedding
        input_w_dim = self.w_embedding.embedding_dim
        q_input_size = input_w_dim
        if self.config['fix_embeddings']:
            for p in self.w_embedding.parameters():
                p.requires_grad = False

        # Projection for attention weighted question
        if self.config['use_qemb']:
            self.qemb_match = SeqAttnMatch(input_w_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = input_w_dim + self.config['num_features']
        if self.config['use_qemb']:
            doc_input_size += input_w_dim

        # Project document and question to the same size as their encoders
        if self.config['resize_rnn_input']:
            self.doc_linear = nn.Linear(doc_input_size, config['hidden_size'], bias=True)
            self.q_linear = nn.Linear(input_w_dim, config['hidden_size'], bias=True)
            doc_input_size = q_input_size = config['hidden_size']

        # RNN document encoder
        self.doc_rnn = StackedBRNN(
            input_size=doc_input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout_rate=config['dropout_rnn'],
            dropout_output=config['dropout_rnn_output'],
            variational_dropout=config['variational_dropout'],
            concat_layers=config['concat_rnn_layers'],
            rnn_type=self._RNN_TYPES[config['rnn_type']],
            padding=config['rnn_padding'],
            bidirectional=True,
        )

        # RNN question encoder
        self.question_rnn = StackedBRNN(
            input_size=q_input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout_rate=config['dropout_rnn'],
            dropout_output=config['dropout_rnn_output'],
            variational_dropout=config['variational_dropout'],
            concat_layers=config['concat_rnn_layers'],
            rnn_type=self._RNN_TYPES[config['rnn_type']],
            padding=config['rnn_padding'],
            bidirectional=True,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * config['hidden_size']
        question_hidden_size = 2 * config['hidden_size']
        if config['concat_rnn_layers']:
            doc_hidden_size *= config['num_layers']
            question_hidden_size *= config['num_layers']

        if config['doc_self_attn']:
            self.doc_self_attn = SeqAttnMatch(doc_hidden_size)
            doc_hidden_size = doc_hidden_size + question_hidden_size

        # Question merging
        if config['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % config['question_merge'])
        if config['question_merge'] == 'self_attn':
            self.self_attn = LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        q_rep_size = question_hidden_size + doc_hidden_size if config['span_dependency'] else question_hidden_size
        self.end_attn = BilinearSeqAttn(
            doc_hidden_size,
            q_rep_size,
        )

    def forward(self, ex):
        """Inputs:
        xq = question word indices             (batch, max_q_len)
        xq_mask = question padding mask        (batch, max_q_len)
        xd = document word indices             (batch, max_d_len)
        xd_f = document word features indices  (batch, max_d_len, nfeat)
        xd_mask = document padding mask        (batch, max_d_len)
        targets = span targets                 (batch,)
        """

        # Embed both document and question
        xq_emb = self.w_embedding(ex['xq'])                         # (batch, max_q_len, word_embed)
        xd_emb = self.w_embedding(ex['xd'])                         # (batch, max_d_len, word_embed)

        shared_axes = [2] if self.config['word_dropout'] else []
        xq_emb = dropout(xq_emb, self.config['dropout_emb'], shared_axes=shared_axes, training=self.training)
        xd_emb = dropout(xd_emb, self.config['dropout_emb'], shared_axes=shared_axes, training=self.training)
        xd_mask = ex['xd_mask']
        xq_mask = ex['xq_mask']

        # Add attention-weighted question representation
        if self.config['use_qemb']:
            xq_weighted_emb = self.qemb_match(xd_emb, xq_emb, xq_mask)
            drnn_input = torch.cat([xd_emb, xq_weighted_emb], 2)
        else:
            drnn_input = xd_emb

        if self.config["num_features"] > 0:
            drnn_input = torch.cat([drnn_input, ex['xd_f']], 2)

        # Project document and question to the same size as their encoders
        if self.config['resize_rnn_input']:
            drnn_input = F.relu(self.doc_linear(drnn_input))
            xq_emb = F.relu(self.q_linear(xq_emb))
            if self.config['dropout_ff'] > 0:
                drnn_input = F.dropout(drnn_input, training=self.training)
                xq_emb = F.dropout(xq_emb, training=self.training)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, xd_mask)       # (batch, max_d_len, hidden_size)

        # Document self attention
        if self.config['doc_self_attn']:
            xd_weighted_emb = self.doc_self_attn(doc_hiddens, doc_hiddens, xd_mask)
            doc_hiddens = torch.cat([doc_hiddens, xd_weighted_emb], 2)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(xq_emb, xq_mask)
        if self.config['question_merge'] == 'avg':
            q_merge_weights = uniform_weights(question_hiddens, xq_mask)
        elif self.config['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens.contiguous(), xq_mask)
        question_hidden = weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, xd_mask)
        if self.config['span_dependency']:
            question_hidden = torch.cat([question_hidden, (doc_hiddens * start_scores.exp().unsqueeze(2)).sum(1)], 1)
        end_scores = self.end_attn(doc_hiddens, question_hidden, xd_mask)

        return {'score_s': start_scores,
                'score_e': end_scores,
                'targets': ex['targets']}
