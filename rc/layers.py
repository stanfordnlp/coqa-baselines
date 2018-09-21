import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# Modules #
################################################################################

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0,
                 dropout_output=False, variational_dropout=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False, bidirectional=True,
                 return_single_timestep=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.variational_dropout = variational_dropout
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.return_single_timestep = return_single_timestep
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else (2 * hidden_size if bidirectional else hidden_size)
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=bidirectional))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # Pad if we care or if its during eval.
        if self.padding or self.return_single_timestep or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            # Apply dropout to hidden input
            rnn_input = dropout(rnn_input, self.dropout_rate,
                                shared_axes=[1] if self.variational_dropout else [], training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)  # Concatenate hiddens at each timestep.
        else:
            output = outputs[-1]  # Take only hiddens after final layer (for all timesteps).

        # Dropout on output layer
        if self.dropout_output:
            output = dropout(output, self.dropout_rate,
                             shared_axes=[1] if self.variational_dropout else [], training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        # Sort x
        rnn_input = x.index_select(0, idx_sort)

        # Encode all layers
        outputs, single_outputs = [rnn_input], []
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                rnn_input = dropout(rnn_input, self.dropout_rate,
                                    shared_axes=[1] if self.variational_dropout else [], training=self.training)
            # Pack it
            rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True)
            # Run it
            rnn_output, (hn, _) = self.rnns[i](rnn_input)
            # Unpack it
            rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)[0]
            single_outputs.append(hn[-1])
            outputs.append(rnn_output)

        if self.return_single_timestep:
            output = single_outputs[-1]
        # Concat hidden layers or take final
        elif self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Unsort
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = dropout(output, self.dropout_rate,
                             shared_axes=[1] if self.variational_dropout else [], training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h  (document)
            y = batch * len2 * h  (question)
            y_mask = batch * len2 (question mask)
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # (batch, len1, len2)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())  # (batch, len1, len2)
        scores.masked_fill_(y_mask, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=-1)

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq                      # (batch, len2, h)


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1  (doc_hiddens)
        y = batch * h2        (question_hidden)
        x_mask = batch * len  (xd_mask)
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.masked_fill_(x_mask, -float('inf'))
        alpha = F.log_softmax(xWy, dim=-1)
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.masked_fill_(x_mask, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha

################################################################################
# Functional #
################################################################################


def dropout(x, drop_prob, shared_axes=[], training=False):
    if drop_prob == 0 or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


def multi_nll_loss(scores, target_mask):
    """
    Select actions with sampling at train-time, argmax at test-time:
    """
    scores = scores.exp()
    loss = 0
    for i in range(scores.size(0)):
        loss += torch.neg(torch.log(torch.masked_select(scores[i], target_mask[i]).sum() / scores[i].sum()))
    return loss


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    raise NotImplementedError


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
