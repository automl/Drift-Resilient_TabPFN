import torch.nn as nn
from functools import partial


class MLP(nn.Module):
    """A simple MLP, consisting of fully connected layers and dropout layers with specified activations."""

    def __init__(self, n_inputs, n_outputs, hparams):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # set hyperparams
        self.hparams = hparams

        self.activation = hparams["activation"]

        if self.activation == "relu":
            self.activation = nn.ReLU()
            self.weight_init = partial(nn.init.kaiming_uniform_, nonlinearity="relu")
        elif self.activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
            self.weight_init = partial(
                nn.init.kaiming_uniform_, nonlinearity="leaky_relu"
            )
        elif self.activation == "prelu":
            self.activation = nn.PReLU()
            self.weight_init = partial(
                nn.init.kaiming_uniform_, nonlinearity="leaky_relu"
            )
        elif self.activation == "tanh":
            self.activation = nn.Tanh()
            self.weight_init = nn.init.xavier_normal_
        else:
            raise ValueError(f"Unsupported activation: {self.activation}.")

        self.use_batch_norm = hparams["use_batch_norm"]

        # Create a list to hold all the layers of the MLP
        self.model = nn.ModuleList()

        # Input layer
        self.model.append(nn.Linear(self.n_inputs, hparams["mlp_width"]))

        if self.use_batch_norm:
            self.model.append(nn.BatchNorm1d(hparams["mlp_width"]))

        self.model.append(self.activation)
        self.model.append(nn.Dropout(hparams["mlp_dropout"]))

        # Hidden layers
        for _ in range(hparams["mlp_hidden_layers"]):
            self.model.append(nn.Linear(hparams["mlp_width"], hparams["mlp_width"]))

            if self.use_batch_norm:
                self.model.append(nn.BatchNorm1d(hparams["mlp_width"]))

            self.model.append(self.activation)
            self.model.append(nn.Dropout(hparams["mlp_dropout"]))

        # Output layer
        self.model.append(nn.Linear(hparams["mlp_width"], self.n_outputs))

        # Convert ModuleList to Sequential
        self.model = nn.Sequential(*self.model)

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                self.weight_init(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
