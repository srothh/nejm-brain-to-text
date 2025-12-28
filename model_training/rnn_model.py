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
        self.proj_in = nn.Linear(2 * hidden_dim, d_model)
        self.proj = nn.Sequential(
            self.proj_in,
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable initial hidden state for biGRU: (num_layers * 2, 1, hidden_dim)
        self.h0 = nn.Parameter(torch.zeros(num_layers * 2, 1, hidden_dim))
        nn.init.xavier_uniform_(self.h0)

        # (Optional) init like baseline
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.proj_in.weight)

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

# Encoder Experiment results


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
                rnn_dropout=0.0,
                input_dropout=0.0,
                n_layers=5,
                patch_size=0,
                patch_stride=0,
                # New: post-RNN head (training improvement)
                head_type: str = "none",          # "none" | "resffn"
                head_num_blocks: int = 0,         # e.g., 1 or 2
                head_norm: str = "none",          # "bn" | "layernorm" | "rmsnorm" | "none"
                head_dropout: float = 0.0,
                head_activation: str = "gelu",
                # New: speckled masking (coordinated dropout)
                input_speckle_p: float = 0.0,
                input_speckle_mode: str = "feature",
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
        self.head_type = str(head_type)
        self.head_num_blocks = int(head_num_blocks)
        self.head_norm = str(head_norm)
        self.head_dropout = float(head_dropout)
        self.head_activation = str(head_activation)

        self.input_speckle_p = float(input_speckle_p)
        self.input_speckle_mode = str(input_speckle_mode)


        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

       # Day-specific affine parameters (vectorized, compile-friendly)
        self.day_weights = nn.Parameter(
            torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1)
        )  # (n_days, D, D)

        self.day_biases = nn.Parameter(
            torch.zeros(self.n_days, self.neural_dim)
        )  # (n_days, D)


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

        # Optional post-GRU head
        ht = self.head_type.lower()
        if ht == "none" or self.head_num_blocks <= 0:
            self.head = nn.Identity()
        elif ht in ("resffn", "ffn"):
            self.head = nn.Sequential(*[
                ResidualFFNBlock(
                    d=self.n_units,
                    norm_type=self.head_norm,
                    dropout=self.head_dropout,
                    activation=self.head_activation,
                )
                for _ in range(self.head_num_blocks)
            ])
        else:
            raise ValueError(f"Unknown head_type={self.head_type}. Use: none, resffn.")

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
        day_ids = day_idx.view(-1).long()  # (B,)

        day_weights = self.day_weights.index_select(0, day_ids)          # (B, D, D)
        day_biases  = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B, 1, D)

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

        # Speckled masking (training only)
        if self.training and self.input_speckle_p > 0:
            x = speckle_mask(x, self.input_speckle_p, self.input_speckle_mode)

        
        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Optional post-GRU head
        output = self.head(output)

        # Compute logits
        logits = self.out(output)

        
        if return_state:
            return logits, hidden_states
        
        return logits
        




import torch 
from torch import nn

class GRUEncoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                neural_dim,
                n_units,
                n_days,
                n_classes,
                rnn_dropout=0.0,
                input_dropout=0.0,
                n_layers=5,
                patch_size=0,
                patch_stride=0,
                # New: post-RNN head (training improvement)
                head_type: str = "none",          # "none" | "resffn"
                head_num_blocks: int = 0,         # e.g., 1 or 2
                head_norm: str = "none",          # "bn" | "layernorm" | "rmsnorm" | "none"
                head_dropout: float = 0.0,
                head_activation: str = "gelu",
                # New: speckled masking (coordinated dropout)
                input_speckle_p: float = 0.0,
                input_speckle_mode: str = "feature",
                 d_model = 0

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
        super(GRUEncoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.head_type = str(head_type)
        self.head_num_blocks = int(head_num_blocks)
        self.head_norm = str(head_norm)
        self.head_dropout = float(head_dropout)
        self.head_activation = str(head_activation)

        self.input_speckle_p = float(input_speckle_p)
        self.input_speckle_mode = str(input_speckle_mode)

        self.d_model = d_model
        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

       # Day-specific affine parameters (vectorized, compile-friendly)
        self.day_weights = nn.Parameter(
            torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1)
        )  # (n_days, D, D)

        self.day_biases = nn.Parameter(
            torch.zeros(self.n_days, self.neural_dim)
        )  # (n_days, D)


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

        # Optional post-GRU head
        ht = self.head_type.lower()
        if ht == "none" or self.head_num_blocks <= 0:
            self.head = nn.Identity()
        elif ht in ("resffn", "ffn"):
            self.head = nn.Sequential(*[
                ResidualFFNBlock(
                    d=self.n_units,
                    norm_type=self.head_norm,
                    dropout=self.head_dropout,
                    activation=self.head_activation,
                )
                for _ in range(self.head_num_blocks)
            ])
        else:
            raise ValueError(f"Unknown head_type={self.head_type}. Use: none, resffn.")

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)
        self.enc_proj = nn.Sequential(
            nn.Linear(self.n_units, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        # init
        for m in self.enc_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        self.enc_ln = nn.LayerNorm(self.d_model) 

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, lengths=None, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_ids = day_idx.view(-1).long()  # (B,)

        day_weights = self.day_weights.index_select(0, day_ids)          # (B, D, D)
        day_biases  = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B, 1, D)

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

        # Speckled masking (training only)
        if self.training and self.input_speckle_p > 0:
            x = speckle_mask(x, self.input_speckle_p, self.input_speckle_mode)

        
        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Optional post-GRU head
        output = self.head(output)
        enc = self.enc_ln(self.enc_proj(output))  # (B,T,d_model)
        if return_state:
            return enc, hidden_states
        return enc

        # Compute logits
        logits = self.out(output)

        
        if return_state:
            return logits, hidden_states
        
        return logits
        


class WhisperGRUEncoderForPhonemes(nn.Module):
    """
    Wraps a GRUDecoder configured as a Whisper encoder (returns d_model features),
    then adds a phoneme head for CTC pretraining.
    Keeps call signature: (x, day_idx, lengths) -> logits
    so you don't have to rewrite your trainer.
    """
    def __init__(self, encoder: nn.Module, d_model: int, n_classes: int = 41):
        super().__init__()
        self.encoder = encoder
        self.phoneme_head = nn.Linear(d_model, n_classes)
        nn.init.xavier_uniform_(self.phoneme_head.weight)

    def forward(self, x, day_idx, lengths=None, return_features=False):
        feats = self.encoder(x, day_idx)          # (B, T', d_model)
        logits = self.phoneme_head(feats)         # (B, T', n_phonemes)
        return (logits, feats) if return_features else logits

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class MyBatchNorm1d(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(d, eps=eps, momentum=momentum)

    def forward(self, x):
        if x.dim() == 3:  # (B,T,D)
            b, t, d = x.shape
            y = x.reshape(b * t, d)
            y = self.bn(y)
            return y.reshape(b, t, d)
        elif x.dim() == 2:  # (B,D)
            return self.bn(x)
        else:
            raise ValueError(f"MyBatchNorm1d expected 2D or 3D input, got shape={tuple(x.shape)}")

def build_time_norm(norm_type: str, d: int) -> nn.Module:
    norm_type = (norm_type or "none").lower()
    if norm_type == "bn":
        return MyBatchNorm1d(d)
    if norm_type == "layernorm":
        return nn.LayerNorm(d)
    if norm_type == "rmsnorm":
        return RMSNorm(d)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type={norm_type}. Use one of: bn, layernorm, rmsnorm, none.")

def get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation={name}. Use one of: gelu, relu, silu.")


def speckle_mask(x: torch.Tensor, p: float, mode: str) -> torch.Tensor:
    """
    Coordinated dropout / speckled masking.
    x: (B,T,D)
    mode:
      - 'feature': drop entire features across all timesteps (mask shape Bx1xD)
      - 'time':    drop entire timesteps across all features (mask shape BxTx1)
      - 'both':    elementwise (BxTxD)  (usually less stable; keep for ablation)
    """
    if p <= 0.0:
        return x
    mode = (mode or "feature").lower()
    B, T, D = x.shape
    if mode == "feature":
        mask = torch.rand(B, 1, D, device=x.device) < p
    elif mode == "time":
        mask = torch.rand(B, T, 1, device=x.device) < p
    elif mode == "both":
        mask = torch.rand(B, T, D, device=x.device) < p
    else:
        raise ValueError(f"Unknown speckle mode={mode}. Use: feature, time, both.")
    return x.masked_fill(mask, 0.0)


class ResidualFFNBlock(nn.Module):
    """
    Simple GPT-style MLP block without attention:
      x <- x + Dropout(Act(Linear(Norm(x))))
    Works on (B,T,D).
    """
    def __init__(self, d: int, norm_type: str, dropout: float, activation: str):
        super().__init__()
        self.norm = build_time_norm(norm_type, d)
        self.lin = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.lin.weight)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(self.norm(x))
        y = self.act(y)
        y = self.drop(y)
        return x + y


class ResLSTMSublayer(nn.Module):
    """
    One residual BiLSTM sublayer:
      - pre_norm: y = LSTM(norm(x)); x = x + dropout(y)
      - post_norm: y = LSTM(x); x = norm(x + dropout(y))
    """
    def __init__(
        self,
        d: int,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        norm_type: str = "bn",
        pre_norm: bool = False,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        assert d % 2 == 0, f"ResLSTM requires even d, got d={d}"
        self.pre_norm = bool(pre_norm)

        self.norm = build_time_norm(norm_type, d)
        self.residual_dropout = nn.Dropout(p=float(residual_dropout))

        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,
            num_layers=int(lstm_layers),
            dropout=float(lstm_dropout) if int(lstm_layers) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x_in = self.norm(x)
            y, _ = self.lstm(x_in)
            y = self.residual_dropout(y)
            return x + y
        else:
            y, _ = self.lstm(x)
            y = self.residual_dropout(y)
            return self.norm(x + y)


class ResLSTMBlock(nn.Module):
    """
    Notebook-style block = 2 residual BiLSTM sublayers.
    """
    def __init__(
        self,
        d: int,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        norm_type: str = "bn",
        pre_norm: bool = False,
        residual_dropout: float = 0.0,
        sublayers_per_block: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ResLSTMSublayer(
                d=d,
                lstm_layers=lstm_layers,
                lstm_dropout=lstm_dropout,
                norm_type=norm_type,
                pre_norm=pre_norm,
                residual_dropout=residual_dropout,
            )
            for _ in range(int(sublayers_per_block))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResLSTMDecoder(nn.Module):
    """
    Decoder:
      - day-specific affine
      - patching
      - projection to n_units
      - stack of ResLSTMBlocks
      - output head
    """
    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float,
        input_dropout: float,
        n_layers: int,          # kept for compatibility
        patch_size: int,
        patch_stride: int,
        # knobs
        reslstm_num_blocks: int = 1,
        reslstm_sublayers_per_block: int = 2,
        reslstm_lstm_layers: int = 2,
        reslstm_lstm_dropout: float = 0.1,
        reslstm_norm: str = "bn",
        reslstm_pre_norm: bool = False,
        reslstm_residual_dropout: float = 0.0,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

        # Day-specific affine
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(p=float(input_dropout))

        # en __init__
        self.day_weights = nn.Parameter(torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1))
        self.day_biases  = nn.Parameter(torch.zeros(self.n_days, self.neural_dim))


        # Patching => flatten => proj
        in_dim = neural_dim * self.patch_size if self.patch_size > 0 else neural_dim
        self.in_proj = nn.Linear(in_dim, n_units)
        nn.init.xavier_uniform_(self.in_proj.weight)

        # Stack blocks
        self.reslstm = nn.Sequential(*[
            ResLSTMBlock(
                d=n_units,
                lstm_layers=int(reslstm_lstm_layers),
                lstm_dropout=float(reslstm_lstm_dropout),
                norm_type=str(reslstm_norm),
                pre_norm=bool(reslstm_pre_norm),
                residual_dropout=float(reslstm_residual_dropout),
                sublayers_per_block=int(reslstm_sublayers_per_block),
            )
            for _ in range(int(reslstm_num_blocks))
        ])

        self.dropout = nn.Dropout(p=float(rnn_dropout))
        self.out = nn.Linear(n_units, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, features: torch.Tensor, day_indicies: torch.Tensor) -> torch.Tensor:
        # Vectorized day indexing (no .tolist() sync)
        day_ids = day_idx.view(-1).long()                       # (B,)
        W = self.day_weights.index_select(0, day_ids)           # (B,D,D)
        b = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B,1,D)

        x = torch.einsum("btd,bdk->btk", x, W) + b
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)  # si input_dropout>0


        # patching
        if self.patch_size > 0:
            ps = self.patch_size
            st = self.patch_stride
            x = x.unfold(dimension=1, size=ps, step=st)               # (B, T', C, ps)
            x = x.permute(0, 1, 3, 2).contiguous()                    # (B, T', ps, C)
            x = x.view(x.size(0), x.size(1), -1)                      # (B, T', ps*C)

        x = self.in_proj(x)                                           # (B, T', n_units)
        x = self.reslstm(x)                                           # (B, T', n_units)
        x = self.dropout(x)
        logits = self.out(x)                                          # (B, T', n_classes)
        return logits




