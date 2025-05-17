from tabpfn.model import transformer


def build_model(
    criterion,
    encoder_generator,
    n_out,
    emsize=200,
    nhid=200,
    nlayers=6,
    nhead=2,
    y_encoder_generator=None,
    decoder_dict={},
    initializer=None,
    load_weights_from_this_state_dict=None,
    initialize_with_model=None,
    **model_extra_args,
):
    decoder_dict = decoder_dict if decoder_dict else {"standard": (None, n_out)}

    assert not model_extra_args.pop("use_zero_attention", False)
    assert not model_extra_args.pop("share_key_and_value_attention_proj", False)

    features_per_group = model_extra_args.pop("features_per_group", 1)
    assert model_extra_args.pop("use_flash_attention", True)
    assert not model_extra_args.pop("bias", False)

    model = transformer.PerFeatureTransformer(
        encoder=encoder_generator(features_per_group, emsize),
        nhead=nhead,
        ninp=emsize,
        nhid=nhid,
        nlayers=nlayers,
        y_encoder=y_encoder_generator(1, emsize),
        init_method=initializer,
        decoder_dict=decoder_dict,
        features_per_group=features_per_group,
        use_encoder_compression_layer=model_extra_args.pop(
            "use_encoder_compression_layer", False
        ),
        feature_positional_embedding=model_extra_args.pop(
            "feature_positional_embedding", None
        ),
        use_separate_decoder=model_extra_args.pop("use_separate_decoder", False),
        **model_extra_args,
    )

    model.criterion = criterion

    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    return model
