import torch 
from torch import nn

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits


import torch
import torch.nn as nn

class DayBiGRUEncoder(nn.Module):
    """
    Day-aware bidirectional GRU encoder -> (B, T_enc, d_model)
    Keeps the baseline's day-specific input alignment, but uses the notebook-style:
      - pack/pad for variable lengths
      - biGRU hidden states projected to Whisper d_model
    """

    def __init__(
        self,
        neural_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_days: int,
        d_model: int,
        rnn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        patch_size: int = 0,
        patch_stride: int = 0,
    ):
        super().__init__()
        self.out_ln = nn.LayerNorm(d_model)
        self.neural_dim = neural_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_days = n_days
        self.d_model = d_model

        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

        # --- Day-specific affine "alignment" layer ---
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(input_dropout)

        # More GPU/compile-friendly than ParameterList: (n_days, D, D) and (n_days, D)
        eye = torch.eye(neural_dim).unsqueeze(0).repeat(n_days, 1, 1)
        self.day_weights = nn.Parameter(eye)                    # (n_days, D, D)
        self.day_biases  = nn.Parameter(torch.zeros(n_days, neural_dim))  # (n_days, D)

        # --- Optional patching changes the GRU input size ---
        gru_input_dim = neural_dim
        if self.patch_size and self.patch_size > 0:
            gru_input_dim = neural_dim * self.patch_size

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )

        # Notebook-style: project hidden states to Whisper d_model
        self.proj = nn.Linear(2 * hidden_dim, d_model)

        # Learnable initial hidden state for biGRU: (num_layers * 2, 1, hidden_dim)
        self.h0 = nn.Parameter(torch.zeros(num_layers * 2, 1, hidden_dim))
        nn.init.xavier_uniform_(self.h0)

        # (Optional) init like baseline
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor, day_idx: torch.Tensor, lengths: torch.Tensor):
        """
        x:        (B, T, D) padded neural features
        day_idx:  (B,) day indices
        lengths:  (B,) true lengths AFTER patching if patching is enabled
                           (or raw lengths if patching disabled)
        returns:  (B, T_enc, d_model)
        """

        B, T, D = x.shape
        if day_idx.dim() == 0:
            day_idx = day_idx.unsqueeze(0)

        # --- Day-specific transform ---
        W = self.day_weights[day_idx]                 # (B, D, D)
        b = self.day_biases[day_idx].unsqueeze(1)     # (B, 1, D)

        # x @ W + b
        x = torch.einsum("btd,bde->bte", x, W) + b
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)

        # --- Optional patching/downsampling (matches baseline idea) ---
        if self.patch_size and self.patch_size > 0:
            # x: (B, T, D) -> (B, T_p, patch_size, D) -> (B, T_p, patch_size*D)
            x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
            x = x.contiguous().view(B, x.size(1), -1)  # (B, T_p, patch_size*D)

        # --- Pack/pad like notebook ---
        lengths = lengths.to(dtype=torch.long)
        lengths = torch.clamp(lengths, min=1)  # pack can't handle zeros
        lengths_cpu = lengths.detach().cpu()

        h0 = self.h0.to(device=x.device, dtype=x.dtype).expand(self.num_layers * 2, B, self.hidden_dim).contiguous()

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed, h0)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        out = self.proj(out)  # (B, T_enc, d_model)
        out = self.out_ln(out)
        return out


import torch
import torch.nn as nn

class DayBiGRUEncoderForPhonemes(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.phoneme_head = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.phoneme_head.weight)

    def forward(self, x, day_idx, lengths, return_features=False):
        feats = self.encoder(x, day_idx, lengths=lengths)     # (B, T, d_model)
        logits = self.phoneme_head(feats)                     # (B, T, n_classes)
        return (logits, feats) if return_features else logits

