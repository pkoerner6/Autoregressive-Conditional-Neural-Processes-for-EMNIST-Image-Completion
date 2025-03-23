from typing import Union

import lab as B
import numpy as np
from plum import Dispatcher
from wbml.util import inv_perm
import torch
import math

from neuralprocesses import _dispatch
from neuralprocesses.aggregate import Aggregate, AggregateInput
from neuralprocesses.datadims import data_dims
from neuralprocesses.mask import Masked
from neuralprocesses.numdata import num_data
from neuralprocesses.model import Model
from neuralprocesses.model.util import tile_for_sampling

__all__ = ["ar_predict", "ar_loglik"]


@_dispatch
def _determine_order(
    state: B.RandomState,
    xt: AggregateInput,
    yt: Union[Aggregate, None],
    order: str,
):
    dispatch = Dispatcher()

    # Compute the given ordering. This is what we'll start from.
    pairs = sum(
        [
            [(i_xt, i_out, i_x) for i_x in range(B.shape(xti, -1))]
            for i_xt, (xti, i_out) in enumerate(xt)
        ],
        [],
    )

    if order in {"random", "given"}:
        if order == "random":
            # Randomly permute.
            state, perm = B.randperm(state, B.dtype_int(xt), len(pairs))
            pairs = [pairs[i] for i in perm]

        # For every output, compute the inverse permutation.
        perms = [[] for _ in range(len(xt))]
        for i_xt, i_out, i_x in pairs:
            perms[i_xt].append(i_x)
        inv_perms = [inv_perm(perm) for perm in perms]

        @_dispatch
        def unsort(y: B.Numeric) -> Aggregate:
            # Create buckets for each output.
            buckets = [[] for _ in range(len(xt))]
            for i_y, (i_xt, _, _) in zip(range(B.shape(y, -1)), pairs):
                buckets[i_xt].append(y[..., i_y : i_y + 1])
            
            # Reorder buckets using inverse permutation.
            out_elems = []
            for i, (bucket, inv_perm_i) in enumerate(zip(buckets, inv_perms)):
                if bucket:  # If non-empty, reorder and concatenate.
                    bucket_sorted = [bucket[j] for j in inv_perm_i]
                    out_elems.append(B.concat(*bucket_sorted, axis=-1))
                else:
                    # Construct an empty tensor with the expected shape.
                    # For example, if the expected shape is the same as xt[i][0] but with last dim 0:
                    expected_shape = list(B.shape(xt[i][0]))
                    expected_shape[-1] = 0
                    # Ensure you create a torch tensor (instead of a numpy array).
                    # Adjust the device and dtype to match y.
                    empty_tensor = torch.zeros(*expected_shape, device=torch.device("cuda"), dtype=torch.get_default_dtype())
                    out_elems.append(empty_tensor)
            return Aggregate(*out_elems)

        return state, xt, yt, pairs, unsort



    elif order == "left-to-right":
        if len(xt) != 1:
            raise ValueError(f"Left-to-right ordering only works for a single output.")

        # Unpack the only output.
        xt, i_out = xt[0]
        if yt is not None:
            yt = yt[0]
        # Copy it, because we'll modify it.
        xt = B.identity(xt)
        if yt is not None:
            yt = B.identity(yt)

        # Sort the targets.
        xt_np = B.to_numpy(xt)  # Need to be NumPy, because we'll use `np.lexsort`.
        perms = [np.lexsort(batch[::-1, :]) for batch in xt_np]
        for i, perm in enumerate(perms):
            xt[i, :, :] = B.take(xt[i, :, :], perm, axis=-1)
            if yt is not None:
                yt[i, :, :] = B.take(yt[i, :, :], perm, axis=-1)

        # Compute the inverse permutations.
        inv_perms = [inv_perm(perm) for perm in perms]

        @_dispatch
        def unsort(y: B.Numeric) -> Aggregate:
            # Create buckets for each output.
            buckets = [[] for _ in range(len(xt))]
            for i_y, (i_xt, _, _) in zip(range(B.shape(y, -1)), pairs):
                buckets[i_xt].append(y[..., i_y : i_y + 1])
            
            # Reorder buckets using inverse permutation.
            out_elems = []
            for i, (bucket, inv_perm_i) in enumerate(zip(buckets, inv_perms)):
                if bucket:  # If non-empty, reorder and concatenate.
                    bucket_sorted = [bucket[j] for j in inv_perm_i]
                    out_elems.append(B.concat(*bucket_sorted, axis=-1))
                else:
                    # Construct an empty tensor with the expected shape.
                    # For example, if the expected shape is the same as xt[i][0] but with last dim 0:
                    expected_shape = list(B.shape(xt[i][0]))
                    expected_shape[-1] = 0
                    # Ensure you create a torch tensor (instead of a numpy array).
                    # Adjust the device and dtype to match y.
                    empty_tensor = torch.zeros(*expected_shape, device=torch.device("cuda"), dtype=torch.get_default_dtype())
                    out_elems.append(empty_tensor)
            return Aggregate(*out_elems)

        # Pack the one output again.
        xt = AggregateInput((xt, i_out))
        yt = Aggregate(yt)

        return state, xt, yt, pairs, unsort

    else:
        raise RuntimeError(f'Invalid ordering "{order}".')


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: AggregateInput,
    num_samples=50,
    order="random",
):
    """Autoregressive sampling.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (:class:`neuralprocesses.aggregrate.AggregateInput`): Inputs of the target
            set. This must be an aggregate of inputs.
        num_samples (int, optional): Number of samples to produce. Defaults to 50.
        order (str, optional): Order. Must be one of `"random"`, `"given"`, or
            `"left-to-right"`. Defaults to `"random"`.

    Returns:
        random state, optional: Random state.
        tensor: Marginal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
        tensor: `num_samples` noisy samples.
    """
    # Perform sorting.
    state, xt_ordered, _, pairs, unsort = _determine_order(state, xt, None, order)

    # Tile to produce multiple samples through batching.
    contexts = tile_for_sampling(contexts, num_samples)
    xt_ordered = tile_for_sampling(xt_ordered, num_samples)

    # Predict autoregressively. See also :func:`ar_loglik` below.
    contexts = list(contexts)  # Copy it so we can modify it.
    preds, yt = [], []
    for i_xt, i_out, i_x in pairs:
        xti, _ = xt_ordered[i_xt]

        # Make the selection of the particular input.
        xti = xti[..., i_x : i_x + 1]

        # Predict and sample.
        state, pred = model(state, contexts, AggregateInput((xti, i_out)))
        state, yti = pred.sample(state)
        yti = yti[0]  # It is an aggregate with one element.
        preds.append(pred)
        yt.append(yti)

        # Append to the context.
        xci, yci = contexts[i_out]
        contexts[i_out] = (
            B.concat(xci, xti, axis=-1),
            B.concat(yci, yti, axis=-1),
        )
    yt = unsort(B.concat(*yt, axis=-1))

    # Produce predictive statistics. The means and variance will be aggregates with
    # one element.
    m1 = B.mean(B.concat(*(p.mean[0] for p in preds), axis=-1), axis=0)
    m2 = B.mean(B.concat(*(p.var[0] + p.mean[0] ** 2 for p in preds), axis=-1), axis=0)
    mean, var = unsort(m1), unsort(m2 - m1**2)

    # Produce noiseless samples `ft` by passing the noisy samples through the model once
    # more.
    state, pred = model(state, contexts, xt)
    ft = pred.mean

    return state, mean, var, ft, yt


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: B.Numeric,
    **kw_args,
):
    # Run the model forward once to determine the number of outputs.
    # TODO: Is there a better way to do this?
    state, pred = model(state, contexts, xt)
    d = data_dims(xt)
    d_y = B.shape(pred.mean, -(d + 1))

    # Perform AR prediction.
    state, mean, var, ft, yt = ar_predict(
        state,
        model,
        contexts,
        AggregateInput(*((xt, i) for i in range(d_y))),
        **kw_args,
    )

    # Convert the outputs back from `Aggregate`s to a regular tensors.
    mean = B.concat(*mean, axis=-(d + 1))
    var = B.concat(*var, axis=-(d + 1))
    ft = B.concat(*ft, axis=-(d + 1))
    yt = B.concat(*yt, axis=-(d + 1))

    return state, mean, var, ft, yt


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    xc: B.Numeric,
    yc: B.Numeric,
    xt: B.Numeric,
    **kw_args,
):
    # Figure out out how many outputs there are.
    d = data_dims(xc)
    d_y = B.shape(yc, -(d + 1))

    def take(y, i):
        """Take the `i`th output."""
        colon = slice(None, None, None)
        return y[(Ellipsis, slice(i, i + 1)) + (colon,) * d]

    return ar_predict(
        state,
        model,
        [(xc, take(yc, i)) for i in range(d_y)],
        xt,
        **kw_args,
    )


@_dispatch
def ar_predict(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-1]))
    res = ar_predict(state, model, *args, **kw_args)
    state, res = res[0], res[1:]
    B.set_global_random_state(state)
    return res


@_dispatch
def _mask_nans(yc: B.Numeric):
    mask = ~B.isnan(yc)
    if B.any(~mask):
        yc = B.where(mask, yc, B.zero(yc))
        return Masked(yc, mask)
    else:
        return yc


@_dispatch
def _mask_nans(yc: Masked):
    return yc


@_dispatch
def _merge_ycs(yc1: B.Numeric, yc2: B.Numeric):
    return B.concat(yc1, yc2, axis=-1)


@_dispatch
def _merge_ycs(yc1: Masked, yc2: B.Numeric):
    with B.on_device(yc2):
        return _merge_ycs(yc1, Masked(yc2, B.ones(yc2)))


@_dispatch
def _merge_ycs(yc1: B.Numeric, yc2: Masked):
    with B.on_device(yc1):
        return _merge_ycs(Masked(yc1, B.ones(yc1)), yc2)


@_dispatch
def _merge_ycs(yc1: Masked, yc2: Masked):
    return Masked(
        _merge_ycs(yc1.y, yc2.y),
        _merge_ycs(yc1.mask, yc2.mask),
    )


@_dispatch
def _merge_contexts(xc1: B.Numeric, yc1, xc2: B.Numeric, yc2):
    xc_merged = B.concat(xc1, xc2, axis=-1)
    yc_merged = _merge_ycs(_mask_nans(yc1), _mask_nans(yc2))
    return xc_merged, yc_merged


# --- Helper functions for covariance estimation ---

def compute_cov_scm(samples):
    """
    Computes the standard sample covariance (unbiased) for samples of shape [n, D].
    """
    n = samples.shape[0]
    mu = torch.mean(samples, dim=0, keepdim=True)
    diff = samples - mu
    cov = torch.matmul(diff.t(), diff) / (n - 1)
    return cov

def compute_cov_bayesian(samples, psi, nu_p=2):
    """
    Computes a Bayesian covariance estimate.
    
    samples: Tensor of shape [n, D].
    psi: A [D, D] diagonal matrix of prior variances (if given as vector, convert to diag).
    nu_p: Extra degrees of freedom for the prior.
    """
    n, D = samples.shape
    sample_cov = compute_cov_scm(samples)
    # Set degrees of freedom: require nu > D + 1.
    nu = n + D + nu_p  
    alpha = (n - 1) / (n - 1 + nu)
    # psi is assumed to be a diagonal matrix.
    bayesian_cov = alpha * sample_cov + (1 - alpha) * psi
    return bayesian_cov

def compute_cov_shrinkage(samples):
    """
    Computes a Ledoit-Wolf shrinkage estimate for the covariance matrix.
    
    samples: Tensor of shape [n, D].
    """
    n, D = samples.shape
    mu = torch.mean(samples, dim=0, keepdim=True)
    X = samples - mu  # Center the data, shape [n, D]
    sample_cov = torch.matmul(X.t(), X) / (n - 1)
    target = torch.diag(torch.diag(sample_cov))
    # Estimate the variance of the covariance estimator:
    diff_cov = (torch.matmul(X.t(), X) / n) - sample_cov
    var_sample_cov = torch.mean(diff_cov ** 2)
    norm_squared = torch.sum((sample_cov - target) ** 2)
    shrinkage = torch.clamp(var_sample_cov / norm_squared, 0.0, 1.0)
    shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
    return shrunk_cov

@_dispatch
def ar_loglik(
    state: torch.Generator,  # Keep your original state type
    model: Model,
    contexts: list,
    xt: AggregateInput,
    yt: Aggregate,
    normalise=True,
    order="random",
    num_samples=50,
    covariance_est="bayesian",   # Choose: "scm", "bayesian", or "shrinkage"
    nu_p=100                  # Extra parameter for Bayesian estimation
):
    # print("Entering ar_loglik")
    
    # Generate autoregressive joint samples.
    state, mean, var, ft, samples = ar_predict(state, model, contexts, xt, num_samples=num_samples, order=order)
    # print("After ar_predict:")
    # for i, s in enumerate(samples):
    #     print(f"  samples[{i}] shape: {s.shape}")
    
    # Flatten each element of the samples Aggregate; each becomes [num_samples, d_i].
    samples_flat = []
    for i, s in enumerate(samples):
        # Skip any tensors with empty dimensions
        if 0 in s.shape:
            # print(f"Skipping empty tensor samples[{i}] with shape {s.shape}")
            continue
        
        # Check if the tensor has the correct first dimension
        if s.shape[0] != num_samples:
            # print(f"Warning: Tensor {i} has first dimension {s.shape[0]} instead of {num_samples}")
            continue
            
        # Reshape and add to the list
        samples_flat.append(s.reshape(s.shape[0], -1))
    
    # Convert any non-torch tensors
    samples_flat = [torch.as_tensor(s) if not isinstance(s, torch.Tensor) else s for s in samples_flat]
    
    # for i, s in enumerate(samples_flat):
    #     print(f"After flattening: samples_flat[{i}] shape: {s.shape}")
    
    # Skip concatenation if all tensors were empty
    if len(samples_flat) == 0:
        # print("All tensors are empty, returning zero loglikelihood")
        return state, torch.tensor(0.0, device=mean[0].device)
    
    # Concatenate all flattened sample tensors along the feature dimension.
    samples_vec = torch.cat(samples_flat, dim=-1)
    # print(f"Concatenated samples_vec shape: {samples_vec.shape}")
    
    # Flatten the observed targets, also skipping empty ones
    yt_flat = []
    for i, s in enumerate(yt):
        if 0 in s.shape:
            # print(f"Skipping empty tensor yt[{i}] with shape {s.shape}")
            continue
        yt_flat.append(s.reshape(-1))
    
    # Check if all observed targets are empty
    if len(yt_flat) == 0:
        # print("All targets are empty, returning zero loglikelihood")
        return state, torch.tensor(0.0, device=mean[0].device)
    
    # for i, s in enumerate(yt_flat):
    #     print(f"After flattening: yt_flat[{i}] shape: {s.shape}")
    
    observed_vec = torch.cat(yt_flat, dim=-1)
    # print(f"Concatenated observed_vec shape: {observed_vec.shape}")
    
    # Compute empirical mean across AR samples.
    mu_emp = torch.mean(samples_vec, dim=0)
    # print(f"Empirical mean shape: {mu_emp.shape}")
    
    # --- Covariance estimation ---
    if covariance_est == "scm":
        cov = compute_cov_scm(samples_vec)
        # print("Using SCM covariance")
    elif covariance_est == "bayesian":
        scm_cov = compute_cov_scm(samples_vec)
        psi = torch.diag(torch.diag(scm_cov))
        cov = compute_cov_bayesian(samples_vec, psi, nu_p=nu_p)
        # print("Using Bayesian covariance")
    elif covariance_est == "shrinkage":
        cov = compute_cov_shrinkage(samples_vec)
        # print("Using Shrinkage covariance")
    else:
        raise ValueError(f"Unknown covariance estimation method: {covariance_est}")
    
    # print(f"Covariance matrix shape before stabilization: {cov.shape}")
    # Symmetrize and add a small diagonal term for numerical stability.
    cov = 0.5 * (cov + cov.t()) + 1e-12 * torch.eye(cov.shape[0], device=cov.device)
    # print("Covariance matrix stabilized")
    # print("Covariance (first 5x5 block):")
    # print(cov[:5, :5])
    # -------------------------------
    
    # Construct a MultivariateNormal distribution with the estimated mean and covariance.
    try:
        mvn_dist = torch.distributions.MultivariateNormal(mu_emp, covariance_matrix=cov)
        logpdf = mvn_dist.log_prob(observed_vec)
    except Exception as e:
        # print("Error constructing multivariate normal distribution:", e)
        raise ValueError("Computed logpdf is NaN. Aborting execution.")
    
    # print(f"Final logpdf: {logpdf}")
    if normalise:
        logpdf = logpdf / mu_emp.shape[-1]
    # print("Exiting ar_loglik")
    return state, logpdf


@_dispatch
def ar_loglik(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: B.Numeric,
    yt: B.Numeric,
    **kw_args,
):
    return ar_loglik(
        state,
        model,
        contexts,
        AggregateInput(*((xt, i) for i in range(B.shape(yt, -2)))),
        Aggregate(*(yt[..., i : i + 1, :] for i in range(B.shape(yt, -2)))),
        **kw_args,
    )


@_dispatch
def ar_loglik(state: B.RandomState, model: Model, xc, yc, xt, yt, **kw_args):
    return ar_loglik(state, model, [(xc, yc)], xt, yt, **kw_args)


@_dispatch
def ar_loglik(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-2]))
    state, res = ar_loglik(state, model, *args, **kw_args)
    B.set_global_random_state(state)
    return res