import torch
import torch.nn as nn
from collections import namedtuple
import time
import numpy as np

# from embed_regularize import embedded_dropout
from .LockedDropout import LockedDropout
from .WeightDropout import WeightDrop


class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers,
                 lstm_dropout=0,
                 batch_first=False,):
        """
        Vanilla LSTM network. 1 or more layers of LSTM, followed by a FC layer
        to predict the next time step.

        Parameters
        ----------
        input_dim : int

        hidden_size : int

        num_layers : int

        lstm_dropout : int, optional
            Dropout rate for LSTM
        batch_first : bool, optional
            Default False
        """
        super(TimeSeriesLSTM, self).__init__()

        self.batch_first = batch_first
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=False,
                            dropout=lstm_dropout,
                            batch_first=batch_first,
                            bidirectional=False)
        # use lstm output to predict next time step,
        # output dim should match input_dim
        self.linear = nn.Linear(hidden_size, input_dim, bias=False)

    def forward(self, x):
        o, hc = self.lstm(x)

        if self.batch_first:
            batch_size, seq_len, hidden_size = o.shape
        else:
            # o.shape == (seq_len, batch_size, hidden_size*num_layers)
            seq_len, batch_size, hidden_size = o.shape

        # use last time step output for linear layer
        z = self.linear(o[-1, :, :].contiguous().view(batch_size, hidden_size))

        return z


class LastValueNet(nn.Module):
    def __init__(self, batch_first=False):
        """
        Always use the last value of input as prediction for the next.

        Parameters
        ----------
        batch_first : bool, optional

        """
        super(LastValueNet, self).__init__()
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first:
            # batch_size, seq_len, input_dim = x.shape
            x = x.permute([1, 0, 2])

        # pytorch default for LSTM
        seq_len, batch_size, input_dim = x.shape
        out = x[-1, :, :].clone()
        return out


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history.
    """
    # if type(h) == Variable:
    #     return Variable(h.data)
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class AWDRNN(nn.Module):

    def __init__(self,
                 rnn_type,
                 input_dim,
                 hidden_dim,
                 nlayers,
                 rnn_out_dropout=.5,
                 dropouth=.5,
                 wdrop=0,
                 tie_weights=False):
        """
        Adapted from awd-lstm-lm, with a few modifications.
        1. No embedding dropout

        Parameters
        ----------
        rnn_type : str
            Either of 'LSTM', 'GRU' or 'QRNN'
        input_dim : TYPE
            input dimension
        hidden_dim : TYPE
            hidden layer size
        nlayers : TYPE
            number of layers
        rnn_out_dropout : float, optional
            locked dropout, i.e. variational dropout rate. Applied to the
            output of the last RNN layer.
        dropouth : float, optional
            variational dropout between RNN layers
        wdrop : int, optional
            weight dropout rate for recurrent weights inside an RNN layer.
        tie_weights : bool, optional
            Default False. If True then for the last RNN layer, the hidden
            size/dim is set to input size/dim.
        """
        # super(AWDRNN, self).__init__()
        super().__init__()

        self.lockdrop = LockedDropout()
        # self.idrop = nn.Dropout(dropouti)
        # self.hdrop = nn.Dropout(dropouth)
        # self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Embedding(ntoken, input_dim)

        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(input_dim if l == 0 else hidden_dim,
                                       hidden_dim if l != nlayers - 1
                                       else (input_dim if tie_weights
                                             else hidden_dim),
                                       1,
                                       dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                             for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(input_dim if l == 0 else hidden_dim,
                                      hidden_dim if l != nlayers - 1
                                      else (input_dim if tie_weights
                                            else hidden_dim),
                                      1,
                                      dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                             for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=input_dim if l == 0
                                   else hidden_dim,
                                   hidden_size=(hidden_dim if l != nlayers - 1
                                                else (input_dim if tie_weights
                                                      else hidden_dim)),
                                   save_prev_x=True,
                                   zoneout=0,
                                   window=2 if l == 0 else 1,
                                   output_gate=True)
                         for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = nn.ModuleList(self.rnns)

        # self.decoder = nn.Linear(hidden_dim, ntoken)
        #
        # # Optionally tie weights as in:
        # # "Using the Output Embedding to Improve Language Models"
        # # (Press & Wolf 2016)
        # # https://arxiv.org/abs/1608.05859
        # # and
        # # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # # Language Modeling" (Inan et al. 2016)
        # # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     # if hidden_dim != input_dim:
        #     #    raise ValueError('When using the tied flag, hidden_dim must
        # be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        # self.init_weights()

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.rnn_out_dropout = rnn_out_dropout
        # self.dropouti = dropouti
        self.dropouth = dropouth
        # self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN':
            [r.reset() for r in self.rnns]

    # def init_weights(self):
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden=None, return_h=False):
        """
        Forward pass

        Parameters
        ----------
        x : TYPE
            default shape (seq_len, batch_size, x_dim)
        hidden : None, optional
            default shape (1, batch_size, hidden_dim), if weights are tied,
            shape becomes (1, batch_size, x_dim)
        return_h : bool, optional


        Returns
        -------
        TYPE
        """
        # no embedding dropout for time series
        # emb = embedded_dropout(self.encoder, x,
        #                        dropout=self.dropoute if self.training else 0)

        # emb = self.lockdrop(emb, self.dropouti)

        # hidden was not optional before. need to look into it.
        if hidden is None:
            _, batch_size, _ = x.shape
            hidden = self.init_hidden(batch_size)
            # print('hidden size = ', len(hidden), ', shape = ',
            #       hidden[0][0].shape)

        # raw_output = emb
        raw_output = x
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            # current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.rnn_out_dropout)
        outputs.append(output)

        # why reshape here? Original awd-lstm-lm feeds the output to a
        # softmax loss function.
        # result = output.view(output.size(0) * output.size(1), output.size(2))
        result = output
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        # weight = next(self.parameters()).data
        weight = next(self.parameters()).detach()
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz,
                                self.hidden_dim if l != self.nlayers - 1
                                else (self.input_dim if self.tie_weights
                                      else self.hidden_dim))
                     .zero_(),
                     weight.new(1, bsz,
                                self.hidden_dim if l != self.nlayers - 1
                                else (self.input_dim if self.tie_weights
                                      else self.hidden_dim))
                     .zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz,
                               self.hidden_dim if l != self.nlayers - 1
                               else (self.input_dim if self.tie_weights
                                     else self.hidden_dim))
                    .zero_()
                    for l in range(self.nlayers)]


class TimeSeriesAWDRNN(nn.Module):

    def __init__(self,
                 rnn_type,
                 input_dim,
                 hidden_dim,
                 nlayers,
                 rnn_out_dropout=.5,
                 dropouth=.5,
                 wdrop=0,
                 tie_weights=False,
                 fc_dropout=0.):
        """
        Use awd-lstm-lm network as the RNN core + linear head.

        Parameters
        ----------
        rnn_type : TYPE

        input_dim : TYPE

        hidden_dim : TYPE

        nlayers : TYPE

        rnn_out_dropout : float, optional

        dropouth : float, optional
            variational dropout rate between RNN layers
        wdrop : int, optional
            recurrent weight dropout rate
        tie_weights : bool, optional

        fc_dropout : float, optional
            Defult 0. Dropout rate between RNN and linear fully connected
            layer.

        """
        # python 3, same as super(TimeSeriesAWDRNN, self)
        super().__init__()

        self.fc_dropout = fc_dropout
        self.fc_input_dim = input_dim if tie_weights else hidden_dim

        self.awd = AWDRNN(rnn_type, input_dim, hidden_dim, nlayers,
                          rnn_out_dropout=rnn_out_dropout,
                          dropouth=dropouth,
                          wdrop=wdrop,
                          tie_weights=tie_weights)
        # for time series the output should have the same dimension as input
        self.linear = nn.Linear(self.fc_input_dim, input_dim, bias=False)

    def init_hidden(self, batch_size):
        return self.awd.init_hidden(batch_size)

    def forward(self, x, h, return_h=False):
        # forward through rnn layers
        out = self.awd(x, h, return_h=return_h)
        if return_h:
            o, hidden, raw_out, dropped_out = out
        else:
            o, hidden = out

        # o.shape == (seq_len, batch_size, dim)
        _, batch_size, hidden_size = o.shape

        # forward through linear layer.
        z = (self.linear(o[-1, :, :].contiguous()
                         .view(batch_size, self.fc_input_dim)))

        if return_h:
            return z, hidden, raw_out, dropped_out
        else:
            return z, hidden


AWDConfig = namedtuple('awdrnn_config',
                       field_names=['model',
                                    'lr',
                                    'epochs',
                                    'wdecay',
                                    'bptt',
                                    'batch_size',
                                    'clip',
                                    'alpha',
                                    'beta',
                                    'log_interval',
                                    'epoch_log_interval',
                                    'eval_batch_size'])


def get_batch(source, i, bptt, seq_len=None, evaluation=False):
    """
    Construct x, y dataset from a given sequence data. y is 1 timestep forward
    from x.

    Parameters
    ----------
    source : TYPE
        assume pytorch default batch shape (seq_len, batch_size, x_dim)
    i : TYPE
        sequence start position
    bptt : TYPE
        backprop through time steps
    seq_len : None, optional
        default timesteps
    evaluation : bool, optional
        Retired in pytorch 0.4

    Returns
    -------
    (x, y)
    """
    # esvhd: this mitigates the risk of drawing a larger seq_len
    # the max seq_len would be len(source) - 1 - 0
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    # data = Variable(source[i:i + seq_len], volatile=evaluation)
    # target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)

    return data, target


def train_awd_rnn(model, config, criterion, optimizer,
                  x_train, y_train, x_valid=None, y_valid=None):
    """
    Batch training for awd-lstm networks with x, y training data.

    Parameters
    ----------
    model : TYPE
        awd-lstm model
    config : TYPE
        config parameters for training, see `AWDConfig`.
    criterion : TYPE
        loss function
    x_train : TYPE
        shape == (seq_len, batch_size, input_dim)
    y_train : TYPE
        target, 1 timestep in the future, shape == (batch_size, input_dim)

    Returns
    -------
    None
    """
    # Turn on training mode which enables dropout.
    if config.model == 'QRNN':
        model.reset()

    hidden = model.init_hidden(config.batch_size)

    # standard pytorch rnn shape, index 1 is batch_size
    n_samples = x_train.shape[1]
    n_batches = n_samples // config.batch_size

    print('n_samples = %d, n_batches = %d' % (n_samples, n_batches))

    loss_hist_train = []
    loss_hist_valid = []

    total_loss = 0
    epoch_start_time = time.time()
    start_time = epoch_start_time
    allin_time = epoch_start_time
    # batch, i = 0, 0
    # while i < x_train.size(0) - 1 - 1:
    for epoch in range(1, config.epochs + 1):
        for batch in range(1, n_batches + 1):
            # print('epoch = ', epoch, ', batch = ', batch)
            batch_start_idx = (batch - 1) * config.batch_size
            batch_end_idx = batch_start_idx + config.batch_size

            bptt = (config.bptt if np.random.random() < 0.95
                    else config.bptt / 2.)
            # Prevent excessively small or negative sequence lengths
            # esvhd: stdev = 5
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long
            # sequence length resulting in OOM
            # esvhd: limit to 2x stdev
            # seq_len = min(seq_len, config.bptt + 10)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / config.bptt
            model.train()

            # don't need original awd-lstm-lm get_batch as data here is already
            # preprocessed into X, y form.
            # Plus for financial data, right now I don't have enough data to
            # randomize bptt timesteps. Maybe something for the future.

            # data, targets = get_batch(x_train, i, config.bptt,
            #                           seq_len=seq_len)
            data = x_train[:, batch_start_idx:batch_end_idx, :]
            # y.shape == (batch_size, x_dim)
            targets = y_train[batch_start_idx:batch_end_idx, :]

            # Starting each batch, we detach the hidden state from how it was
            # previously produced.
            # If we didn't, the model would try backpropagating all the way to
            # start of the dataset.
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden,
                                                           return_h=True)
            # original split cross softmax loss
            # raw_loss = criterion(model.decoder.weight,
            #                      model.decoder.bias, output, targets)
            # replace with time series loss
            raw_loss = criterion(output, targets)

            loss = raw_loss
            # Activiation Regularization
            if config.alpha:
                loss = loss + sum(config.alpha * dropped_rnn_h.pow(2).mean()
                                  for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if config.beta:
                loss = (loss +
                        sum(config.beta *
                            (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                            for rnn_h in rnn_hs[-1:]))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in
            # RNNs / LSTMs.
            if config.clip:
                torch.nn.utils.clip_grad_norm_(params, config.clip)
            optimizer.step()

            total_loss += raw_loss.detach()
            optimizer.param_groups[0]['lr'] = lr2

            # only raw_loss tracked, for comparison with valid or test sets.
            loss_hist_train.append(raw_loss.item())

            if config.log_interval > 0 and batch % config.log_interval == 0:
                # cur_loss = total_loss[0] / config.log_interval
                cur_loss = total_loss.item() / config.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
                      'ms/batch {:5.2f} | avg loss {:5.2f}'
                      .format(epoch, batch,
                              n_batches,
                              optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / config.log_interval,
                              cur_loss,
                              flush=True))
                total_loss = 0
                start_time = time.time()
            ###
        # print epoch data
        if (epoch % config.epoch_log_interval == 0 and
                x_valid is not None and
                y_valid is not None):
            # evaluate, show versus last training loss
            eval_loss = evaluate_rnn(model, config, criterion,
                                     x_valid, y_valid)

            loss_hist_valid.append(eval_loss.item())

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'last avg loss {:5.2f} | valid loss {:5.2f}'
                  .format(epoch,
                          (time.time() - epoch_start_time),
                          np.mean(loss_hist_train[-n_batches:]),
                          eval_loss))
            print('-' * 89, flush=True)
            # reset clock for next epoch
            epoch_start_time = time.time()

    # training finished, print final eval stats
    eval_loss = evaluate_rnn(model, config, criterion, x_valid, y_valid)
    print('-' * 89)
    print('| Training completed | time: {:5.2f}s | '
          'last avg loss {:5.2f} | final valid loss {:5.2f}'
          .format((time.time() - allin_time),
                  np.mean(loss_hist_train[-n_batches:]),
                  eval_loss))
    print('-' * 89, flush=True)

    return (loss_hist_train, loss_hist_valid)


def evaluate_rnn(model, config, loss_func, x_valid, y_valid):
    model.eval()
    if config.model == 'QRNN':
        model.reset()
    total_loss = 0

    # standard pytorch rnn shape, index 1 is batch_size
    n_samples = x_valid.shape[1]
    n_batches = n_samples // config.eval_batch_size

    hidden = model.init_hidden(config.eval_batch_size)

    for batch in range(1, n_batches + 1):
        batch_start_idx = (batch - 1) * config.batch_size
        batch_end_idx = batch_start_idx + config.batch_size

        data = x_valid[:, batch_start_idx:batch_end_idx, :]
        targets = y_valid[batch_start_idx:batch_end_idx, :]

        # hidden is passed to the next batch
        output, hidden = model(data, hidden, return_h=False)
        loss = loss_func(output, targets)
        total_loss += config.eval_batch_size * loss.detach()

        hidden = repackage_hidden(hidden)

    return total_loss / (n_batches * config.eval_batch_size)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AWDRNN('LSTM', 3, 4, 2, wdrop=0).to(device)

    # lstm input dim = (seq_len, batch, dims)
    x = torch.rand(5, 2, 3, device=device)

    print('X = ', x)

    h0 = model.init_hidden(2)
    print('h0 = ', h0)
    print('len(h0) = ', len(h0))
    print('h0 cuda=', h0[0][0].is_cuda)

    hidden = model.forward(x, h0)

    print(hidden)
