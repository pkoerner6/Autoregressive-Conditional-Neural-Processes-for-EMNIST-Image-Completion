from typing import Union
import sys
import lab as B
import numpy as np
from plum import Dispatcher
from wbml.util import inv_perm

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

        @dispatch
        def unsort(y: B.Numeric) -> Aggregate:
            """Undo the sorting."""
            # Put every output in its bucket.
            buckets = [[] for _ in range(len(xt))]
            for i_y, (i_xt, _, _) in zip(range(B.shape(y, -1)), pairs):
                buckets[i_xt].append(y[..., i_y : i_y + 1])
            # Now sort the buckets.
            buckets = [[bucket[j] for j in p] for bucket, p in zip(buckets, inv_perms)]
            # Concatenate and return.
            return Aggregate(*(B.concat(*bucket, axis=-1) for bucket in buckets))

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

        @dispatch
        def unsort(z: B.Numeric) -> Aggregate:
            """Undo the sorting."""
            z = B.identity(z)  # Make a copy, because we'll modify it.
            for i, perm in enumerate(inv_perms):
                z[..., i, :, :] = B.take(z[..., i, :, :], perm, axis=-1)
            return Aggregate(z)

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


@_dispatch
def ar_loglik(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: AggregateInput,
    yt: Aggregate,
    normalise=False,
    order="random",
):
    """Autoregressive log-likelihood.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (:class:`neuralprocesses.aggregrate.AggregateInput`): Inputs of the target
            set. This must be an aggregate of inputs.
        yt (:class:`neuralprocesses.aggregrate.Aggregate`): Outputs of the target
            set. This must be an aggregate of outputs.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        order (str, optional): Order. Must be one of `"random"`, `"given"`, or
            `"left-to-right"`. Defaults to `"random"`.

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """
    state, xt, yt, pairs, _ = _determine_order(state, xt, yt, order)

    # Below, `i_x` will refer to the index of the inputs, and `i_xt` will refer to the
    # index of `xt`. Note that `i_xt` then does _not_ refer to the index of the output.
    # The index of the output `i_out` will be `i_out = xt[j][0]`.

    # Evaluate the log-likelihood autoregressively.
    # print("contexts: ", contexts[0][0].shape)
    # print("contexts points 0: ", contexts[0][0][0][0])
    # print("contexts points 1: ", contexts[0][0][0][1])


    # print("Number of pairs: ", len(pairs))
    contexts = list(contexts)  # Copy it so we can modify it.
    logpdfs = []
    for i_xt, i_out, i_x in pairs:
        # print("xt[i_xt] shape: ", xt[i_xt][0].shape)
        # print("yt[i_xt] shape: ", yt[i_xt].shape)
        xti, _ = xt[i_xt]
        yti = yt[i_xt]

        # Make the selection of the particular input.
        xti = xti[..., i_x : i_x + 1]
        yti = yti[..., i_x : i_x + 1]
        # print("xti shape: ", xti.shape)
        # print("yti shape: ", yti.shape)

        # Compute logpdf.
        state, pred = model(state, contexts, AggregateInput((xti, i_out)))
        # mean = pred.vectorised_normal.mean  # Extract mean (Dense object)
        # print("Predicted mean tensor shape: ", mean.mat.shape) # (b, c, n) B: batch size, c: dimensionality of the inputs/outputs or the number of channel, n: number of data points

        logpdfs.append(pred.logpdf(Aggregate(yti)))

        # Append to the context.
        xci, yci = contexts[i_out]
        # print("xci shape: ", xci.shape)
        # print("yci shape: ", yci.shape)
        # print("xti shape: ", xti.shape)
        # print("yti shape: ", yti.shape)

        contexts[i_out] = _merge_contexts(xci, yci, xti, yti)
        # sys.exit()
    logpdf = sum(logpdfs)

    if normalise:
        # Normalise by the number of targets.
        logpdf = logpdf / num_data(xt, yt)

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





# import torch
# from torch.distributions import Normal
# from einops import rearrange, repeat

# def ar_sample(
#         model, 
#         xc, 
#         yc, 
#         xt, 
#         num_samples=1, 
#         seed=None, 
#         smoothed=False, 
#         batch_size_targets=1
# ):
#     """
#     Performs autoregressive sampling using a model.
    
#     Args:
#         model: An instance of the model
#         xc: Context x values [batch_size, num_context_points, dim_x]
#         yc: Context y values [batch_size, num_context_points, dim_y]
#         xt: Target x values [batch_size, num_target_points, dim_x]
#         num_samples: Number of independent autoregressive samples to generate
#         seed: Random seed for reproducibility
#         smoothed: Whether to outputs smoothed samples too
#         batch_size_targets: Number of target points to process at once in each autoregressive step
        
#     Returns:
#         sampled_y: Sampled y values [num_samples, batch_size, num_target_points, dim_y]
#         sampled_y_noiseless: Noiseless y values [batch_size, num_target_points, dim_y]
#         pred_dists: list of torch.distribution.Normal objects for the autoregressively sampled y
#         smoothed_pred_dists: list of torch.distribution.Normal objects for the autoregressively smoothed sampled y
#     """
#     model.eval()
#     # Set random seed if provided
#     if seed is not None:
#         torch.manual_seed(seed)
#     else:
#         torch.manual_seed(42)
    
#     batch_size = xc.shape[0]
#     num_targets = xt.shape[1]
#     dim_y = yc.shape[-1]
    
#     # Ensure batch_size_targets is valid
#     batch_size_targets = min(batch_size_targets, num_targets)
    
#     # Initialize containers for results
#     use_x = repeat(xt.clone(), 'b n d -> s b n d', s=num_samples)
#     sampled_y = torch.zeros(num_samples, batch_size, num_targets, dim_y, device=xt.device)
    
#     # Generate multiple independent autoregressive samples
#     # Start with the original context for each sample
#     current_xc = repeat(xc.clone(), 'b n d -> (s b) n d', s=num_samples)
#     current_yc = repeat(yc.clone(), 'b n d -> (s b) n d', s=num_samples)
    
#     # Create random permutations for each sample
#     permutations = []
#     for s in range(num_samples):
#         perm = torch.randperm(num_targets)
#         permutations.append(perm)
    
#     # Track original indices for each sample to restore original order later
#     original_indices = torch.zeros(num_samples, num_targets, dtype=torch.long)
#     for s, perm in enumerate(permutations):
#         for i, p in enumerate(perm):
#             original_indices[s, p] = i
    
#     pred_dists = [None] * num_targets  # Initialize with None placeholders
    
#     # Process target points in batches according to random order for each sample
#     num_batches = (num_targets + batch_size_targets - 1) // batch_size_targets
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size_targets
#         end_idx = min((batch_idx + 1) * batch_size_targets, num_targets)
#         batch_size_current = end_idx - start_idx
        
#         # For each sample, get the current batch of points in its permuted order
#         current_xt_list = []
#         batch_indices = []  # Store the actual indices for each sample
        
#         for s in range(num_samples):
#             # Get the indices of the current batch in the permuted order for this sample
#             sample_indices = permutations[s][start_idx:end_idx]
#             batch_indices.append(sample_indices)
            
#             # Gather the corresponding target points
#             sample_xt = use_x[s:s+1, :, sample_indices, :]
#             current_xt_list.append(sample_xt)
        
#         # Concatenate across samples and reshape
#         current_xt = torch.cat(current_xt_list, dim=0)
#         current_xt = rearrange(current_xt, 's b n d -> (s b) n d')
        
#         # Get prediction distribution for the current batch of targets
#         pred_dist = model.predict(current_xc, current_yc, current_xt)
        
#         # Sample from the distribution
#         current_yt = pred_dist.sample()
        
#         # Store the sampled values and prediction distributions in the correct original positions
#         for s in range(num_samples):
#             for i, idx in enumerate(batch_indices[s]):
#                 pos = original_indices[s, idx]
#                 if start_idx <= pos < end_idx:  # This is a position in the current batch in original order
#                     # Store the prediction distribution for this position
#                     if pred_dists[pos] is None:
#                         # Extract the distribution for this specific target point
#                         mean = pred_dist.mean[s*batch_size:(s+1)*batch_size, i:i+1, :]
#                         stddev = pred_dist.stddev[s*batch_size:(s+1)*batch_size, i:i+1, :]
#                         pred_dists[pos] = Normal(mean, stddev)
                
#                 # Extract and store the sampled value for this sample
#                 s_idx = s * batch_size
#                 e_idx = (s + 1) * batch_size
#                 sampled_y[s, :, idx:idx+1, :] = rearrange(
#                     current_yt[s_idx:e_idx, i:i+1, :], 
#                     'b n d -> b n d'
#                 )
        
#         # Add the new points to the context
#         current_xc = torch.cat([current_xc, current_xt], dim=1)
#         current_yc = torch.cat([current_yc, current_yt], dim=1)
    
#     if not smoothed:
#         return sampled_y, None, pred_dists, None
        
#     # Get noiseless predictions using the final context for all target points in original order
#     smoothed_pred_dists = model.predict(current_xc, current_yc, rearrange(use_x, 's b n d -> (s b) n d'))
#     sampled_y_noiseless = rearrange(smoothed_pred_dists.mean, '(s b) n d -> s b n d', s=num_samples)
#     smoothed_pred_dists = [
#         Normal(
#             smoothed_pred_dists.mean[:, i:i+1, :], 
#             smoothed_pred_dists.stddev[:, i:i+1, :]
#         ) 
#         for i in range(use_x.size(2))
#     ]
    
#     return sampled_y, sampled_y_noiseless, pred_dists, smoothed_pred_dists



# def no_cheat_ar_log_likelihood(model, xc, yc, xt, yt, num_samples=20, seed=None, smooth=False, covariance_est="scm", nu_p=2, batch_size_targets=1):
#     """
#     Computes the log likelihood of the target points given the context using autoregressive sampling WITHOUT feeding target labels into context set.
#     In this sense the model does not cheat.
    
#     Args:
#         model: An instance of the LBANP model
#         xc: Context x values [batch_size, num_context_points, dim_x]
#         yc: Context y values [batch_size, num_context_points, dim_y]
#         xt: Target x values [batch_size, num_target_points, dim_x]
#         yt: Target y values [batch_size, num_target_points, dim_y]
#         num_samples: Number of samples to take for the AR sampling
#         seed: Random seed for reproducibility
#         smooth: Whether to use smooth or non smooth samples for covariance matrix estimation/likelihood calculation.
#         covariance_est: The method for estimating covariance matrix from samples. One of [scm, shrinkage, bayesian]
#         nu_p: The degrees of freedom for the inverse wishart prior for bayesian covariance estimation
#         batch_size_targets: Number of target points to process at once in each autoregressive step
        
#     Returns:
#         log_likelihood: Log likelihood of the target points given the context
#     """
#     model.eval()
#     # Set random seed if provided
#     if seed is not None:
#         torch.manual_seed(seed)
#     else:
#         torch.manual_seed(42)
    
#     sampled_y, sampled_y_noiseless, pred_dists, smoothed_pred_dists = ar_sample(model, xc, yc, xt, num_samples=num_samples, seed=seed, smoothed=smooth, batch_size_targets=batch_size_targets)
#     if smooth:
#         use_y = sampled_y_noiseless.clone() # shape (s, b, n, d) where s is num samples, b is batch size, n is target size, d is data dimension
#     else:
#         use_y = sampled_y.clone()
    
#     # Estimate mean across samples, or can use the model prediction
#     # for scm, we use empirical mean
#     mean_y = torch.mean(use_y, dim=0)  # shape (b, n, d)
#     batch_size, num_targets, dim_y = mean_y.shape

#     if covariance_est == "scm":
#         # just take sample covariance matrix (the unbiased version)
#         # uses torch.cov to compute sample covariance matrix for every batch and every data dimension
        
#         log_likelihood = torch.zeros(batch_size, device=use_y.device)
        
#         for b in range(batch_size):
#             # Reshape data for this batch to (num_samples, num_targets*dim_y)
#             # First, extract data for this batch
#             batch_data = use_y[:, b]  # shape (num_samples, num_targets, dim_y)
            
#             # Reshape to (num_samples, num_targets*dim_y)
#             batch_data_flat = batch_data.reshape(num_samples, -1)
            
#             # Compute sample covariance matrix using torch.cov
#             # torch.cov expects input of shape (features, observations)
#             # so we transpose batch_data_flat
#             cov_matrix = torch.cov(batch_data_flat.T)  # shape (num_targets*dim_y, num_targets*dim_y)
            
#             # Get mean for this batch
#             mean_vector = mean_y[b].reshape(-1)  # Flatten to (num_targets*dim_y)
            
#             # Ensure covariance matrix is symmetric and positive definite
#             cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
            
#             # Add small diagonal term for numerical stability
#             cov_matrix = cov_matrix + 1e-12 * torch.eye(cov_matrix.shape[0], device=use_y.device)
            
#             try:
#                 # Create multivariate normal distribution
#                 mvn_dist = torch.distributions.MultivariateNormal(mean_vector, cov_matrix)
                
#                 # Calculate log probability of ground truth
#                 yt_flat = yt[b].reshape(-1)  # Flatten to (num_targets*dim_y)
#                 log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # normalize by number of targets
#             except:
#                 # Handle numerical issues
#                 print("Warning: Covariance matrix is not positive definite. Using fallback method of only taking diagonal entries")
#                 # Fallback to diagonal covariance as approximation
#                 diag_cov = torch.diag(cov_matrix)
#                 mvn_dist = torch.distributions.MultivariateNormal(
#                     mean_vector, 
#                     torch.diag(torch.clamp(diag_cov, min=1e-6))
#                 )
#                 yt_flat = yt[b].reshape(-1)
#                 log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # normalize by number of targets
        
#         return log_likelihood

#     elif covariance_est == "bayesian":
#         # use bayesian covariance estimation with inverse wishart prior with scale matrix a diagonal matrix with variances given by
#         # model prediction on full target set (var_y)
#         # Get model prediction on full target set to estimate variances
#         pred_dist = model.predict(xc, yc, xt)
#         var_y = pred_dist.variance  # Shape: [batch_size, num_targets, dim_y]
        
#         if nu_p is None:
#             nu_p = 1
#         # Parameters for inverse Wishart prior
#         nu = num_samples + dim_y + nu_p  # Degrees of freedom (nu > dim_y + 1)
        
#         # Calculate log likelihood with Bayesian covariance estimation
#         log_likelihood = torch.zeros(batch_size, device=use_y.device)
        
#         for b in range(batch_size):
#             # Reshape data for this batch to (num_samples, num_targets*dim_y)
#             batch_data = use_y[:, b]  # shape (num_samples, num_targets, dim_y)
            
#             # Reshape to (num_samples, num_targets*dim_y)
#             batch_data_flat = batch_data.reshape(num_samples, -1)
            
#             # Compute sample covariance matrix using torch.cov
#             # torch.cov expects input of shape (features, observations)
#             sample_cov = torch.cov(batch_data_flat.T)  # shape (num_targets*dim_y, num_targets*dim_y)
            
#             # Create diagonal scale matrix from model's predicted variances
#             psi = torch.diag_embed(var_y[b].reshape(-1))  # Diagonal matrix of variances
            
#             # Bayesian estimate: weighted combination of sample covariance and prior
#             alpha = (num_samples - 1) / (num_samples - 1 + nu)
#             bayesian_cov = alpha * sample_cov + (1 - alpha) * psi
            
#             # Ensure symmetry and positive definiteness
#             bayesian_cov = 0.5 * (bayesian_cov + bayesian_cov.T)
#             bayesian_cov = bayesian_cov + 1e-12 * torch.eye(bayesian_cov.shape[0], device=use_y.device)
            
#             try:
#                 # Create multivariate normal distribution with Bayesian covariance
#                 mvn_dist = torch.distributions.MultivariateNormal(mean_y[b].reshape(-1), bayesian_cov)
                
#                 # Calculate log probability of ground truth
#                 yt_flat = yt[b].reshape(-1)
#                 log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # Normalize by number of targets
#             except:
#                 # Handle numerical issues
#                 print("Warning: Bayesian covariance matrix is not positive definite. Using fallback method.")
#                 # Fallback to diagonal covariance
#                 diag_cov = torch.diag(bayesian_cov)
#                 mvn_dist = torch.distributions.MultivariateNormal(
#                     mean_y[b].reshape(-1), 
#                     torch.diag(torch.clamp(diag_cov, min=1e-6))
#                 )
#                 yt_flat = yt[b].reshape(-1)
#                 log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets
        
#         return log_likelihood

#     elif covariance_est == "shrinkage":
#         # use ledoit-wolf shrinkage estimator
#         # Implement Ledoit-Wolf shrinkage estimator for covariance matrix
#         # This method combines the sample covariance matrix with a structured estimator
#         # to improve estimation when number of samples is limited
        
#         # Estimate mean across samples
#         mean_y = torch.mean(use_y, dim=0)  # shape (b, n, d)
        
#         # Center the data
#         centered_y = use_y - mean_y.unsqueeze(0)  # shape (s, b, n, d)
        
#         # Calculate log likelihood with shrinkage covariance estimation
#         log_likelihood = torch.zeros(batch_size, device=use_y.device)
        
#         for b in range(batch_size):
#             # Reshape centered data for this batch to (num_samples, num_targets*dim_y)
#             X = centered_y[:, b].reshape(num_samples, -1)  # shape (s, n*d)
            
#             # Sample covariance matrix
#             n = X.shape[0]
#             sample_cov = torch.matmul(X.t(), X) / (n - 1)  # shape (n*d, n*d)
            
#             # Target for shrinkage: diagonal matrix with average variance
#             target = torch.diag(torch.diag(sample_cov))
            
#             # Calculate optimal shrinkage intensity
#             # Formula based on Ledoit-Wolf method
#             var_sample_cov = torch.mean((torch.matmul(X.t(), X) / n - sample_cov) ** 2)
            
#             # Calculate Frobenius norm of difference between sample cov and target
#             cov_minus_target = sample_cov - target
#             norm_squared = torch.sum(cov_minus_target ** 2)
            
#             # Optimal shrinkage intensity (capped between 0 and 1)
#             shrinkage = torch.clamp(var_sample_cov / norm_squared, 0.0, 1.0)
            
#             # Apply shrinkage
#             shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
            
#             # Ensure symmetry and positive definiteness
#             shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.t())
#             shrunk_cov = shrunk_cov + 1e-6 * torch.eye(shrunk_cov.shape[0], device=use_y.device)
            
#             try:
#                 # Create multivariate normal distribution with shrunk covariance
#                 mvn_dist = torch.distributions.MultivariateNormal(mean_y[b].reshape(-1), shrunk_cov)
                
#                 # Calculate log probability of ground truth
#                 yt_flat = yt[b].reshape(-1)
#                 log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets  # Normalize by number of targets
#             except:
#                 # Handle numerical issues
#                 print("Warning: Shrunk covariance matrix is not positive definite. Using fallback method.")
#                 # Fallback to diagonal covariance
#                 diag_cov = torch.diag(shrunk_cov)
#                 mvn_dist = torch.distributions.MultivariateNormal(
#                     mean_y[b].reshape(-1), 
#                     torch.diag(torch.clamp(diag_cov, min=1e-6))
#                 )
#                 yt_flat = yt[b].reshape(-1)
#                 log_likelihood[b] = mvn_dist.log_prob(yt_flat) / num_targets
        
#         return log_likelihood
#     else:
#         raise NotImplementedError(f"{covariance_est} covariance estimation is not implemented for no_cheat_ar_log_likelihood")



# def ar_log_likelihood(model, xc, yc, xt, yt, seed=None):
#     """
#     Computes the log likelihood of the target points given the context using autoregressive sampling.
    
#     Args:
#         model: An instance of the LBANP model
#         xc: Context x values [batch_size, num_context_points, dim_x]
#         yc: Context y values [batch_size, num_context_points, dim_y]
#         xt: Target x values [batch_size, num_target_points, dim_x]
#         yt: Target y values [batch_size, num_target_points, dim_y]
#         seed: Random seed for reproducibility
        
#     Returns:
#         log_likelihood: Log likelihood of the target points given the context
#     """
#     model.eval()
#     # Set random seed if provided
#     if seed is not None:
#         torch.manual_seed(seed)
#     else:
#         torch.manual_seed(42)
    
#     batch_size = xc.shape[0]
#     num_targets = xt.shape[1]
#     dim_y = yc.shape[-1]
    
#     # Initialize containers for results
#     use_x = xt.clone()
#     log_likelihoods = torch.zeros(batch_size, num_targets, device=xt.device)
    
#     # Generate multiple independent autoregressive samples
#     # Start with the original context for each sample
#     current_xc = xc.clone()
#     current_yc = yc.clone()
    
#     # Process one target point at a time
#     for i in range(num_targets):
#         # Extract the current target point
#         current_xt = use_x[:, i:i+1, :]
#         current_yt = yt[:, i:i+1, :]
        
#         # Get prediction distribution for the current target
#         pred_dist = model.predict(current_xc, current_yc, current_xt)
        
#         # Get log likelihood of the target point
#         log_likelihood = pred_dist.log_prob(current_yt)
        
#         # Store the log likelihood
#         log_likelihoods[:, i] = log_likelihood[:,0,:].sum(-1)
        
#         # Add the new point to the context
#         current_xc = torch.cat([current_xc, current_xt], dim=1)
#         current_yc = torch.cat([current_yc, current_yt], dim=1)
    
#     log_likelihood = torch.mean(log_likelihoods, dim=-1)
#     return log_likelihood
    
        