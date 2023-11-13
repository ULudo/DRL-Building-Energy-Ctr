import torch.nn as nn


class Summarizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2, recurrent_type='lstm'):

        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, observations, hidden=None, return_hidden=False):
        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary
        