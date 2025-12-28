from torch import nn


class RNNEncoder(nn.Module):
    def __init__(self,
                 proj_in_dim,
                 d_model = 0,
                 gru_model = None
                 ):
        super(RNNEncoder, self).__init__()
        # self.gru = GRUDecoder(neural_dim, n_units, n_days, n_classes,rnn_dropout,input_dropout,n_layers,patch_size,patch_stride)
        self.gru = gru_model
        self.proj = nn.Linear(proj_in_dim, d_model)

    def forward(self, x, day_idx, states=None, return_state=False):
        out = self.gru(x=x, day_idx=day_idx, states=states, return_state=return_state)
        if return_state:
            logits, st = out
            return self.proj(logits), st
        return self.proj(out)



