import functools
from functools import partial
from typing import Union, Optional, Tuple

import torch
from torch import nn
from torch.nn.modules.transformer import Module, Tensor

from tabpfn.model.multi_head_attention import MultiHeadAttention
from tabpfn.model.save_peak_mem_factor import support_save_peak_mem_factor
from tabpfn.model.mlp import MLP


class LayerNorm(torch.nn.LayerNorm):
    """
    Custom LayerNorm module that supports saving peak memory factor.

    This module extends the PyTorch LayerNorm implementation to handle FP16 inputs
    efficiently and support saving peak memory factor.

    Args:
        *args: Positional arguments passed to the base LayerNorm class.
        **kwargs: Keyword arguments passed to the base LayerNorm class.
    """

    @functools.wraps(torch.nn.LayerNorm.__init__)
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @support_save_peak_mem_factor
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the layer normalization.

        If the input is FP16 and the normalized shape is less than 512, the computation
        is forced to be in FP16 to improve performance.

        Args:
            x: The input tensor.

        Returns:
            The layer normalized tensor.
        """

        # this if statement has the following function:
        # torch.amp.autocast wants to run layer_norm in fp32
        # but that has very bad effects for our performance (up to 2x slower)
        # thus we force fp16, if the input to the module is fp16, which is the case if autocast is used.
        # WARNING: this could lead to instabilities for higher hidden sizes (> 512), thus we only do this for smaller hidden sizes

        if x.dtype == torch.float16 and sum(self.normalized_shape) < 512:
            with torch.amp.autocast("cuda" if x.is_cuda else "cpu", enabled=False):
                return super().forward(x)
        return super().forward(x)

    def forward(
        self,
        x: torch.Tensor,
        allow_inplace: bool = False,
        save_peak_mem_factor: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Perform layer normalization on the input tensor.

        Args:
            x: The input tensor.
            allow_inplace: Whether to allow in-place operations. Default is False.
            save_peak_mem_factor: The factor to save peak memory. Default is None.

        Returns:
            The layer normalized tensor.
        """
        x = self._compute(
            x, allow_inplace=allow_inplace, save_peak_mem_factor=save_peak_mem_factor
        )
        return x


class PerFeatureEncoderLayer(Module):
    """
    Transformer encoder layer that processes each feature block separately.

    This layer consists of multi-head attention between features, multi-head attention between items,
    and feedforward neural networks (MLPs). It supports various configurations and optimization options.

    Args:
        d_model: The dimensionality of the input and output embeddings.
        nhead: The number of attention heads.
        dim_feedforward: The dimensionality of the feedforward network. Default is None (2 * d_model).
        activation: The activation function to use in the MLPs. Default is "relu".
        layer_norm_eps: The epsilon value for layer normalization. Default is 1e-5.
        pre_norm: Whether to apply layer normalization before or after the attention and MLPs. Default is False.
        device: The device to use for the layer parameters. Default is None.
        dtype: The data type to use for the layer parameters. Default is None.
        recompute_attn: Whether to recompute attention during backpropagation. Default is False.
        second_mlp: Whether to include a second MLP in the layer. Default is False.
        layer_norm_with_elementwise_affine: Whether to use elementwise affine parameters in layer normalization. Default is False.
        zero_init: Whether to initialize the output of the MLPs to zero. Default is False.
        save_peak_mem_factor: The factor to save peak memory, only effective with post-norm. Default is None.
        attention_between_features: Whether to apply attention between feature blocks. Default is True.
        multiquery_item_attention: Whether to use multiquery attention for items. Default is False.
        multiquery_item_attention_for_test_set: Whether to use multiquery attention for the test set. Default is False.
        attention_init_gain: The gain value for initializing attention parameters. Default is 1.0.
        d_k: The dimensionality of the query and key vectors. Default is None (d_model // nhead).
        d_v: The dimensionality of the value vectors. Default is None (d_model // nhead).
        precomputed_kv: Precomputed key-value pairs for attention. Default is None.
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        recompute_attn: bool = False,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = False,
        save_peak_mem_factor: Optional[int] = None,
        attention_between_features: bool = True,
        multiquery_item_attention: bool = False,
        multiquery_item_attention_for_test_set: bool = False,
        attention_init_gain: float = 1.0,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        precomputed_kv: Union[
            None, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **extra_args,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert d_model % nhead == 0 or d_k is not None and d_v is not None
        assert not (
            multiquery_item_attention_for_test_set and multiquery_item_attention
        ), "Cannot use both multiquery_item_attention_for_test_set and multiquery_item_attention"
        if d_k is None:
            d_k = d_model // nhead
        if d_v is None:
            d_v = d_model // nhead
        if multiquery_item_attention:
            print("Using multiquery in item attention.")
        if attention_between_features:
            self.self_attn_between_features = MultiHeadAttention(
                d_model,
                d_model,
                d_k,
                d_v,
                nhead,
                device,
                dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
                init_gain=attention_init_gain,
            )
        else:
            self.self_attn_between_features = None

        if isinstance(precomputed_kv, tuple):
            precomputed_k, precomputed_v = precomputed_kv
            precomputed_kv = None
        else:
            assert precomputed_kv is None or isinstance(precomputed_kv, torch.Tensor)
            precomputed_k = precomputed_v = None
        self.self_attn_between_items = MultiHeadAttention(
            d_model,
            d_model,
            d_k,
            d_v,
            nhead,
            device,
            dtype,
            share_kv_across_n_heads=nhead if multiquery_item_attention else 1,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
            precomputed_k=precomputed_k,
            precomputed_v=precomputed_v,
            precomputed_kv=precomputed_kv,
            init_gain=attention_init_gain,
        )

        if dim_feedforward is None:
            dim_feedforward = 2 * d_model
        self.mlp = MLP(
            d_model,
            dim_feedforward,
            activation,
            device,
            dtype,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
        )

        self.layer_norms = nn.ModuleList(
            [
                LayerNorm(
                    d_model,
                    layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                    **factory_kwargs,
                )
                for _ in range(4 if second_mlp else 3)
            ]
        )

        if second_mlp:
            self.second_mlp = MLP(
                d_model,
                dim_feedforward,
                activation,
                device,
                dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
            )
        else:
            self.second_mlp = None

        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_peak_mem_factor = save_peak_mem_factor
        self.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )

    def __setstate__(self, state):
        state.setdefault("save_peak_mem_factor", None)
        super().__setstate__(state)

    def forward(
        self,
        state: Tensor,
        single_eval_pos: Optional[int] = None,
        cache_trainset_representation: bool = False,
        att_src: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            state: The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos: The position from which on everything is treated as test set. Default is None.
            cache_trainset_representation: Whether to cache the trainset representation.
                If single_eval_pos is set (> 0 and not None), create a cache of the trainset KV.
                This may require a lot of memory. Otherwise, use cached KV representations for inference.
                Default is False.
            att_src: The tensor to attend to from the final layer of the encoder. It has a shape of
                (batch_size, num_train_items, num_feature_blocks, d_model). This does not work with
                multiquery_item_attention_for_test_set and cache_trainset_representation at this point.
                Combining would be possible, however.
                Default is None.

        Returns:
            The transformer state passed through the encoder layer.
        """
        assert (
            len(state.shape) == 4
        ), "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
        if single_eval_pos is None:
            single_eval_pos = 0
        if cache_trainset_representation and not single_eval_pos:
            assert self.self_attn_between_items.has_cached_kv

        if att_src is not None:
            assert (
                not self.multiquery_item_attention_for_test_set
            ), "Not implemented yet."
            assert not cache_trainset_representation, "Not implemented yet."
            assert (
                not single_eval_pos
            ), "single_eval_pos should not be set, as the train representation is in att_src"

        def attn_between_features(x):
            return self.self_attn_between_features(
                x,
                save_peak_mem_factor=self.save_peak_mem_factor,
                add_input=True,
                allow_inplace=True,
            )

        def attn_between_items(x):
            # we need to transpose as self attention always treats dim -2 as the sequence dimension
            if self.multiquery_item_attention_for_test_set:
                if single_eval_pos < x.shape[1]:
                    new_x_test = self.self_attn_between_items(
                        x[:, single_eval_pos:].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2)
                        if single_eval_pos
                        else None,
                        save_peak_mem_factor=self.save_peak_mem_factor,
                        cache_kv=False,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=not single_eval_pos,
                        reuse_first_head_kv=True,
                    ).transpose(1, 2)
                else:
                    new_x_test = None

                if single_eval_pos:
                    new_x_train = self.self_attn_between_items(
                        x[:, :single_eval_pos].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2),
                        save_peak_mem_factor=self.save_peak_mem_factor,
                        cache_kv=cache_trainset_representation,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=False,
                    ).transpose(1, 2)
                else:
                    new_x_train = None
                return torch.cat(
                    [x_ for x_ in [new_x_train, new_x_test] if x_ is not None], dim=1
                )
            else:
                attention_src_x = None
                if att_src is not None:
                    attention_src_x = att_src.transpose(1, 2)
                elif single_eval_pos:
                    attention_src_x = x[:, :single_eval_pos].transpose(1, 2)

                return self.self_attn_between_items(
                    x.transpose(1, 2),
                    attention_src_x,
                    save_peak_mem_factor=self.save_peak_mem_factor,
                    cache_kv=cache_trainset_representation and single_eval_pos,
                    add_input=True,
                    allow_inplace=True,
                    use_cached_kv=cache_trainset_representation and not single_eval_pos,
                ).transpose(1, 2)

        mlp_save_peak_mem_factor = (
            self.save_peak_mem_factor * 8
            if self.save_peak_mem_factor is not None
            else None
        )

        sublayers = []
        if self.self_attn_between_features is not None:
            sublayers.append(attn_between_features)
        else:
            assert (
                state.shape[2] == 1
            ), "If there is no attention between features, the number of feature blocks must be 1."

        sublayers += [
            attn_between_items,
            partial(
                self.mlp,
                save_peak_mem_factor=mlp_save_peak_mem_factor
                if (
                    mlp_save_peak_mem_factor is not None
                    and state.numel() // state.shape[-1] // mlp_save_peak_mem_factor
                )
                > 32
                else None,  # this is a hot fix, since it seems the matmul kernels for small batch sizes yield different results...
                add_input=True,
                allow_inplace=True,
            ),
        ]

        if self.second_mlp is not None:
            sublayers.insert(
                1,
                partial(
                    self.second_mlp,
                    save_peak_mem_factor=mlp_save_peak_mem_factor,
                    add_input=True,
                    allow_inplace=True,
                ),
            )
        # TODO: it appears that memory overhead (~x2) is created before the first sublayer in inference with save_peak_mem_factor.
        for sublayer, layer_norm in zip(sublayers, self.layer_norms):
            if self.pre_norm:
                assert (
                    False
                ), "Pre-norm implementation is wrong, as the residual should never be layer normed here."
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=self.save_peak_mem_factor,
                )
            state = sublayer(state)
            if not self.pre_norm:
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=self.save_peak_mem_factor,
                )

        return state

    def empty_trainset_representation_cache(self):
        self.self_attn_between_items.empty_kv_cache()
        self.self_attn_between_features.empty_kv_cache()  # not necessary, but just in case
