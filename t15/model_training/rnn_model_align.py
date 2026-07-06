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

        # Use ModuleList of single-layer GRUs (like bit_dann Transformer) so we can run
        # one full forward and optionally return an intermediate layer's output.
        layer_sizes = [self.input_size] + [self.n_units] * (self.n_layers - 1)
        self.gru_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=layer_sizes[i],
                    hidden_size=self.n_units,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                )
            )
        # Init: orthogonal for recurrent, xavier for input
        for layer in self.gru_layers:
            for name, param in layer.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, mapping legacy single self.gru keys to gru_layers.* if needed."""
        if any(k.startswith('gru.') and 'gru_layers' not in k for k in state_dict):
            # Legacy: gru.weight_ih_l0 -> gru_layers.0.weight_ih_l0 (each layer is 1-layer GRU so param is _l0)
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith('gru.') and ('weight_ih_l' in k or 'weight_hh_l' in k or 'bias_ih_l' in k or 'bias_hh_l' in k):
                    parts = k.split('.', 1)  # ['gru', 'weight_ih_l0']
                    rest = parts[1]  # 'weight_ih_l0'
                    layer_idx = int(rest.split('_l')[1])  # 0, 1, 2, ...
                    new_rest = rest.rsplit('_', 1)[0] + '_l0'  # weight_ih_l0 (same for layer 0; for layer 1: weight_ih_l1 -> weight_ih_l0)
                    new_sd['gru_layers.%d.%s' % (layer_idx, new_rest)] = v
                else:
                    new_sd[k] = v
            state_dict = new_sd
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x, day_idx, states=None, return_state=False, return_rep=False, rep_layer_idx=None):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x.
        return_rep (bool)  - if True, also return intermediate embedding for DANN (same API as bit_dann).
        rep_layer_idx (int|None) - which representation to return when return_rep=True:
            -1 = after day layer, before GRU (input to GRU);
            None = final GRU layer output (default);
            0 = GRU layer 1 output (after first GRU layer);
            1 = GRU layer 2 output (after second GRU layer);
            ...
            n_layers-1 = final GRU layer output (before projection to CTC classes).
        '''
        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        # Use stacked indexing (not list comp over day_idx) so torch.compile can trace this.
        stacked_w = torch.stack([self.day_weights[i] for i in range(self.n_days)], dim=0)
        stacked_b = torch.stack([self.day_biases[i] for i in range(self.n_days)], dim=0)
        day_weights = stacked_w[day_idx]
        day_biases = stacked_b[day_idx]

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

        if return_rep and rep_layer_idx == -1:
            rep_from_input = x  # [B, T, input_size] - before GRU

        if return_rep and rep_layer_idx is not None:
            if rep_layer_idx < -1 or rep_layer_idx > (self.n_layers - 1):
                raise ValueError(
                    f"rep_layer_idx must be in [-1, {self.n_layers - 1}] or None, got {rep_layer_idx}"
                )

        # Initial hidden states (one per layer)
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Full forward through GRU layers (same as bit_dann: one pass, optionally capture intermediate)
        current = x
        intermediate_rep = None
        hidden_list = []

        for i, layer in enumerate(self.gru_layers):
            h0_i = states[i : i + 1]  # [1, B, n_units]
            current, h_n = layer(current, h0_i)  # h_n: [1, B, n_units]
            hidden_list.append(h_n.squeeze(0))
            # 0-based indexing: rep_layer_idx=0 captures after first GRU layer (i=0)
            if return_rep and rep_layer_idx is not None and rep_layer_idx == i:
                intermediate_rep = current
            # Dropout between GRU layers (same as nn.GRU(..., dropout=rnn_dropout) with num_layers>1)
            if i < self.n_layers - 1 and self.rnn_dropout > 0:
                current = nn.functional.dropout(current, p=self.rnn_dropout, training=self.training)

        output = current
        hidden_states = torch.stack(hidden_list, dim=0)  # [n_layers, B, n_units]
        logits = self.out(output)

        if return_state and not return_rep:
            return logits, hidden_states
        if return_rep:
            if rep_layer_idx == -1:
                rep = rep_from_input
            elif rep_layer_idx is not None and rep_layer_idx >= 0 and intermediate_rep is not None:
                rep = intermediate_rep
            else:
                rep = output
            return logits, rep
        return logits

    def compute_length(self, n_time_steps):
        """CTC length after patching (same as baseline formula)."""
        return ((n_time_steps - self.patch_size) / self.patch_stride + 1).to(torch.int32)

