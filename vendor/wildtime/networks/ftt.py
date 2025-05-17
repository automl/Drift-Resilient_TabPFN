from torch import nn
from pytorch_widedeep.models import FTTransformer


class FTT(nn.Module):
    """A simple MLP, consisting of fully connected layers and dropout layers with ReLU activations."""

    def __init__(
        self, column_idx, cat_embed_input, continuous_cols, n_outputs, hparams
    ):
        super(FTT, self).__init__()

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols

        hparams["mlp_hidden_dims"] = list(hparams["mlp_hidden_dims"]) + [n_outputs]

        self.hparams = hparams

        self.model = FTTransformer(
            column_idx=self.column_idx,
            cat_embed_input=self.cat_embed_input,
            continuous_cols=self.continuous_cols,
            **self.hparams,
        )

        """
        # Use the nn.Sequential container to create the model
        self.model = FTTransformer(
            categories=self.categories,             # tuple containing the number of unique values within each category
            num_continuous=self.num_continuous,     # number of continuous values
            dim=self.dim,                           # dimension, paper set at 32
            dim_out=self.dim_out,                   # binary prediction, but could be anything
            depth=self.depth,                       # depth, paper recommended 6
            heads=self.heads,                       # heads, paper recommends 8
            attn_dropout=self.attn_dropout,         # post-attention dropout
            ff_dropout=self.ff_dropout              # feed forward dropout
        )
        """

    def forward(self, x):
        return self.model(x)
