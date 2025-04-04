from plum import convert
import neuralprocesses.torch as nps  # This fixes inspection below.


__all__ = [
    "construct_convgnp",
    "construct_likelihood",
    "parse_transform",
    "register_model",
]

models = []  #: Registered models
def register_model(model):
    """Decorator to register a new model."""
    models.append(model)
    return model


def construct_likelihood(nps=nps, *, spec, dim_y, num_basis_functions, dtype):
    """Construct the likelihood.

    Args:
        nps (module): Appropriate backend-specific module.
        spec (str, optional): Specification. Must be one of `"het"`, `"lowrank"`,
            `"dense"`, `"spikes-beta"`, or `"bernoulli-gamma"`. Defaults to `"lowrank"`.
            Must be given as a keyword argument.
        dim_y (int): Dimensionality of the outputs. Must be given as a keyword argument.
        num_basis_functions (int): Number of basis functions for the low-rank
            likelihood. Must be given as a keyword argument.
        dtype (dtype): Data type. Must be given as a keyword argument.

    Returns:
        int: Number of channels that the likelihood requires.
        coder: Coder which can select a particular output channel. This coder may be
            `None`.
        coder: Coder.
    """
    if spec == "het":
        num_channels = 2 * dim_y
        selector = nps.SelectFromChannels(dim_y, dim_y)
        lik = nps.HeterogeneousGaussianLikelihood()
    elif spec == "lowrank":
        num_channels = (2 + num_basis_functions) * dim_y
        selector = nps.SelectFromChannels(dim_y, (num_basis_functions, dim_y), dim_y)
        lik = nps.LowRankGaussianLikelihood(num_basis_functions)
    elif spec == "dense":
        # This is intended to only work for global variables.
        num_channels = 2 * dim_y + dim_y * dim_y
        selector = None
        lik = nps.Chain(
            nps.Splitter(2 * dim_y, dim_y * dim_y),
            nps.Parallel(
                lambda x: x,
                nps.Chain(
                    nps.ToDenseCovariance(),
                    nps.DenseCovariancePSDTransform(),
                ),
            ),
            nps.DenseGaussianLikelihood(),
        )
    elif spec == "spikes-beta":
        num_channels = (2 + 3) * dim_y  # Alpha, beta, and three log-probabilities
        selector = nps.SelectFromChannels(dim_y, dim_y, dim_y, dim_y, dim_y)
        lik = nps.SpikesBetaLikelihood()
    elif spec == "bernoulli-gamma":
        num_channels = (2 + 2) * dim_y  # Shape, scale, and two log-probabilities
        selector = nps.SelectFromChannels(dim_y, dim_y, dim_y, dim_y)
        lik = nps.BernoulliGammaLikelihood()

    else:
        raise ValueError(f'Incorrect likelihood specification "{spec}".')
    return num_channels, selector, lik


def parse_transform(nps=nps, *, transform):
    """Construct the likelihood.

    Args:
        nps (module): Appropriate backend-specific module.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, `"softplus"`, or
            `"softplus_of_square"` for positive data or `(lower, upper)` for data in
            this open interval.

    Returns:
        coder: Transform.
    """
    if isinstance(transform, str) and transform.lower() in {"positive", "exp"}:
        transform = nps.Transform.exp()
    elif isinstance(transform, str) and transform.lower() == "softplus":
        transform = nps.Transform.softplus()
    elif isinstance(transform, str) and transform.lower() == "softplus_of_square":
        transform = nps.Chain(
            nps.Transform.signed_square(),
            nps.Transform.softplus(),
        )
    elif isinstance(transform, tuple):
        lower, upper = transform
        transform = nps.Transform.bounded(lower, upper)
    elif transform is not None:
        raise ValueError(f'Cannot parse value "{transform}" for `transform`.')
    else:
        transform = lambda x: x
    return transform






def _convgnp_init_dims(dim_yc, dim_yt, dim_y):
    # Make sure that `dim_yc` is initialised and a tuple.
    dim_yc = convert(dim_yc or dim_y, tuple)
    # Make sure that `dim_yt` is initialised.
    dim_yt = dim_yt or dim_y
    # `len(dim_yc)` is equal to the number of density channels.
    conv_in_channels = sum(dim_yc) + len(dim_yc)
    return dim_yc, dim_yt, conv_in_channels


def _convgnp_resolve_architecture(
    conv_arch,
    unet_channels,
    conv_channels,
    conv_receptive_field,
):
    if "unet" in conv_arch:
        conv_out_channels = unet_channels[0]
    elif "conv" in conv_arch:
        conv_out_channels = conv_channels
        if conv_receptive_field is None:
            raise ValueError("Must specify `conv_receptive_field`.")
    else:
        raise ValueError(f'Architecture "{conv_arch}" invalid.')
    return conv_out_channels


def _convgnp_construct_encoder_setconvs(
    nps,
    encoder_scales,
    dim_yc,
    disc,
    dtype=None,
    init_factor=1,
    encoder_scales_learnable=True,
):
    # Initialise scale.
    if encoder_scales is not None:
        encoder_scales = init_factor * encoder_scales
    else:
        encoder_scales = 1 / disc.points_per_unit
    # Ensure that there is one for every context set.
    if not isinstance(encoder_scales, (tuple, list)):
        encoder_scales = (encoder_scales,) * len(dim_yc)
    # Construct set convs.
    return nps.Parallel(
        *(
            nps.SetConv(s, dtype=dtype, learnable=encoder_scales_learnable)
            for s in encoder_scales
        )
    )


def _convgnp_assert_form_contexts(nps, dim_yc):
    if len(dim_yc) == 1:
        return nps.Chain(
            nps.SqueezeParallel(),
            nps.AssertNoParallel(),
        )
    else:
        return nps.AssertParallel(len(dim_yc))


def _convgnp_construct_decoder_setconv(
    nps,
    decoder_scale,
    disc,
    dtype=None,
    init_factor=1,
    decoder_scale_learnable=True,
):
    if decoder_scale is not None:
        decoder_scale = init_factor * decoder_scale
    else:
        decoder_scale = 1 / disc.points_per_unit
    return nps.SetConv(decoder_scale, dtype=dtype, learnable=decoder_scale_learnable)


def _convgnp_optional_division_by_density(nps, divide_by_density, epsilon):
    if divide_by_density:
        return nps.DivideByFirstChannel(epsilon=epsilon)
    else:
        return lambda x: x


@register_model
def construct_convgnp(
    dim_x=1,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    dim_aux_t=None,
    points_per_unit=64,
    margin=0.1,
    likelihood="lowrank",
    conv_arch="unet",
    unet_channels=(64,) * 6,
    unet_kernels=5,
    unet_strides=2,
    unet_activations=None, # ReLU is used when None
    unet_resize_convs=False,
    unet_resize_conv_interp_method="nearest",
    conv_receptive_field=None,
    conv_layers=6,
    conv_channels=64,
    num_basis_functions=64,
    dim_lv=0,
    lv_likelihood="het",
    encoder_scales=None,
    encoder_scales_learnable=True,
    decoder_scale=None,
    decoder_scale_learnable=True,
    aux_t_mlp_layers=(128,) * 3,
    divide_by_density=True,
    epsilon=1e-4,
    transform=None,
    dtype=None,
    nps=nps,
):
    """A Convolutional Gaussian Neural Process.

    Sets the attribute `receptive_field` to the receptive field of the model.

    Args:
        dim_x (int, optional): Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional): Dimensionality of the outputs of the
            context set. You should set this if the dimensionality of the outputs
            of the context set is not equal to the dimensionality of the outputs
            of the target set. You should also set this if you want to use multiple
            context sets. In that case, set this equal to a tuple of integers
            indicating the respective output dimensionalities.
        dim_yt (int, optional): Dimensionality of the outputs of the target set. You
            should set this if the dimensionality of the outputs of the target set is
            not equal to the dimensionality of the outputs of the context set.
        dim_aux_t (int, optional): Dimensionality of target-specific auxiliary
            variables.
        points_per_unit (float, optional): Density of the internal discretisation.
            Defaults to 64.
        margin (float, optional): Margin of the internal discretisation. Defaults to
            0.1.
        likelihood (str, optional): Likelihood. Must be one of `"het"`, `"lowrank"`,
            `"spikes-beta"`, or `"bernoulli-gamma"`. Defaults to `"lowrank"`.
        conv_arch (str, optional): Convolutional architecture to use. Must be one of
            `"unet[-res][-sep]"` or `"conv[-res][-sep]"`. Defaults to `"unet"`.
        unet_channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        unet_kernels (int or tuple[int], optional): Sizes of the kernels in the UNet.
            Defaults to 5.
        unet_strides (int or tuple[int], optional): Strides in the UNet. Defaults to 2.
        unet_activations (object or tuple[object], optional): Activation functions
            used by the UNet. If `None`, ReLUs are used.
        unet_resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions in the UNet. Defaults to `False`.
        unet_resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions in the UNet. Can be set to `"bilinear"`. Defaults
            to "nearest".
        conv_receptive_field (float, optional): Receptive field of the standard
            architecture. Must be specified if `conv_arch` is set to `"conv"`.
        conv_layers (int, optional): Layers of the standard architecture. Defaults to 8.
        conv_channels (int, optional): Channels of the standard architecture. Defaults
            to 64.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to `512`.
        dim_lv (int, optional): Dimensionality of the latent variable. Defaults to 0.
        lv_likelihood (str, optional): Likelihood of the latent variable. Must be one of
            `"het"` or `"lowrank"`. Defaults to `"het"`.
        encoder_scales (float or tuple[float], optional): Initial value for the length
            scales of the set convolutions for the context sets embeddings. Defaults
            to `1 / points_per_unit`.
        encoder_scales_learnable (bool, optional): Whether the encoder SetConv
            length scale(s) are learnable.
        decoder_scale (float, optional): Initial value for the length scale of the
            set convolution in the decoder. Defaults to `1 / points_per_unit`.
        decoder_scale_learnable (bool, optional): Whether the decoder SetConv
            length scale(s) are learnable.
        aux_t_mlp_layers (tuple[int], optional): Widths of the layers of the MLP
            for the target-specific auxiliary variable. Defaults to three layers of
            width 128.
        divide_by_density (bool, optional): Divide by the density channel. Defaults
            to `True`.
        epsilon (float, optional): Epsilon added by the set convolutions before
            dividing by the density channel. Defaults to `1e-4`.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, `"softplus"`, or
            `"softplus_of_square"` for positive data or `(lower, upper)` for data in
            this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: ConvGNP model.
    """
    dim_yc, dim_yt, conv_in_channels = _convgnp_init_dims(dim_yc, dim_yt, dim_y)

    # Construct likelihood of the encoder, which depends on whether we're using a
    # latent variable or not.
    if dim_lv > 0:
        lv_likelihood_in_channels, _, lv_likelihood = construct_likelihood(
            nps,
            spec=lv_likelihood,
            dim_y=dim_lv,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )
        encoder_likelihood = lv_likelihood
    else:
        encoder_likelihood = nps.DeterministicLikelihood()

    # Construct likelihood of the decoder.
    likelihood_in_channels, selector, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=dim_yt,
        num_basis_functions=num_basis_functions,
        dtype=dtype,
    )

    # Resolve the architecture.
    conv_out_channels = _convgnp_resolve_architecture(
        conv_arch,
        unet_channels,
        conv_channels,
        conv_receptive_field,
    )

    # If `dim_aux_t` is given, contruct an MLP which will use the auxiliary
    # information from the augmented inputs.
    if dim_aux_t:
        likelihood = nps.Augment(
            nps.Chain(
                nps.MLP(
                    in_dim=conv_out_channels + dim_aux_t,
                    layers=aux_t_mlp_layers,
                    out_dim=likelihood_in_channels,
                    dtype=dtype,
                ),
                likelihood,
            )
        )
        linear_after_set_conv = lambda x: x  # See the `else` clause below.
    else:
        # There is no auxiliary MLP available, so the CNN will have to produce the
        # right number of channels. In this case, however, it may be more efficient
        # to produce the right number of channels _after_ the set conv.
        if conv_out_channels < likelihood_in_channels:
            # Perform an additional linear layer _after_ the set conv.
            linear_after_set_conv = nps.Linear(
                in_channels=conv_out_channels,
                out_channels=likelihood_in_channels,
                dtype=dtype,
            )
        else:
            # Not necessary. Just let the CNN produce the right number of channels.
            conv_out_channels = likelihood_in_channels
            linear_after_set_conv = lambda x: x
        # Also assert that there is no augmentation given.
        likelihood = nps.Chain(nps.AssertNoAugmentation(), likelihood)

    # Construct the core CNN architectures for the encoder, which is only necessary
    # if we're using a latent variable, and for the decoder. First, we determine
    # how many channels these architectures should take in and produce.
    if dim_lv > 0:
        lv_in_channels = conv_in_channels
        lv_out_channels = lv_likelihood_in_channels
        in_channels = dim_lv
        out_channels = conv_out_channels  # These must be equal!
    else:
        in_channels = conv_in_channels
        out_channels = conv_out_channels  # These must be equal!
    if "unet" in conv_arch:
        if dim_lv > 0:
            lv_conv = nps.UNet(
                dim=dim_x,
                in_channels=lv_in_channels,
                out_channels=lv_out_channels,
                channels=unet_channels,
                kernels=unet_kernels,
                strides=unet_strides,
                activations=unet_activations,
                resize_convs=unet_resize_convs,
                resize_conv_interp_method=unet_resize_conv_interp_method,
                separable="sep" in conv_arch,
                residual="res" in conv_arch,
                dtype=dtype,
            )
        else:
            lv_conv = lambda x: x

        conv = nps.UNet(
            dim=dim_x,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=unet_channels,
            kernels=unet_kernels,
            strides=unet_strides,
            activations=unet_activations,
            resize_convs=unet_resize_convs,
            resize_conv_interp_method=unet_resize_conv_interp_method,
            separable="sep" in conv_arch,
            residual="res" in conv_arch,
            dtype=dtype,
        )
        receptive_field = conv.receptive_field / points_per_unit
    elif "conv" in conv_arch:
        if dim_lv > 0:
            lv_conv = nps.ConvNet(
                dim=dim_x,
                in_channels=lv_in_channels,
                out_channels=lv_out_channels,
                channels=conv_channels,
                num_layers=conv_layers,
                points_per_unit=points_per_unit,
                receptive_field=conv_receptive_field,
                separable="sep" in conv_arch,
                residual="res" in conv_arch,
                dtype=dtype,
            )
        else:
            lv_conv = lambda x: x

        conv = nps.ConvNet(
            dim=dim_x,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=conv_channels,
            num_layers=conv_layers,
            points_per_unit=points_per_unit,
            receptive_field=conv_receptive_field,
            separable="sep" in conv_arch,
            residual="res" in conv_arch,
            dtype=dtype,
        )
        receptive_field = conv_receptive_field
    else:
        raise ValueError(f'Architecture "{conv_arch}" invalid.')

    # Construct the discretisation, taking into account that the input to the UNet
    # must play nice with the halving layers.
    disc = nps.Discretisation(
        points_per_unit=points_per_unit,
        multiple=2**conv.num_halving_layers,
        margin=margin,
        dim=dim_x,
    )

    # Construct model.
    model = nps.Model(
        nps.FunctionalCoder(
            disc,
            nps.Chain(
                _convgnp_assert_form_contexts(nps, dim_yc),
                nps.PrependDensityChannel(),
                _convgnp_construct_encoder_setconvs(
                    nps,
                    encoder_scales,
                    dim_yc,
                    disc,
                    dtype,
                    encoder_scales_learnable=encoder_scales_learnable,
                ),
                _convgnp_optional_division_by_density(nps, divide_by_density, epsilon),
                nps.Concatenate(),
                lv_conv,
                encoder_likelihood,
            ),
        ),
        nps.Chain(
            conv,
            nps.RepeatForAggregateInputs(
                nps.Chain(
                    _convgnp_construct_decoder_setconv(
                        nps,
                        decoder_scale,
                        disc,
                        dtype,
                        decoder_scale_learnable=decoder_scale_learnable,
                    ),
                    linear_after_set_conv,
                    selector,  # Select the right target output.
                )
            ),
            likelihood,
            parse_transform(nps, transform=transform),
        ),
    )

    # Set attribute `receptive_field`.
    model.receptive_field = receptive_field

    return model
