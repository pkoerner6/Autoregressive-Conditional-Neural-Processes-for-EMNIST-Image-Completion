import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
import lab as B
import wbml.out as out
import experiment as exp
import torchvision.datasets as tvds


import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import neuralprocesses.torch as nps
from train import main
from ar import ar_loglik, ar_predict


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--ar", type=str, choices=["no_ar", "old_ar", "fixed_ar"])
parser.add_argument("--num_samples", type=int, default=100)
args = parser.parse_args()



def test_emnist():
    training_results_path = os.path.join('code', '_experiments')

    experiment = main(
        model=args.model,
        data="emnist",
        root=training_results_path,
        load=True,
    )
    print(f"Loading model from: {experiment['wd'].root}")

    model = experiment["model"]
    model.load_state_dict(
        torch.load(experiment["wd"].file("model-best.torch"), map_location="cpu", weights_only=False)["weights"]
    )

    if args.ar =="no_ar":
        nps_loglik = nps.loglik
    elif args.ar =="old_ar":
        nps_loglik = ar_loglik
    else:
        print("NOT IMPLEMENTED YET!!")
        sys.exit()

    gens_eval_instances = experiment["gens_eval"]()  # Get evaluation datasets
    for name, gen_eval in gens_eval_instances:
        print(f"Evaluating on {name} dataset with {len(gen_eval.data)} images.")

        with torch.no_grad():
            logliks = []

            for batch in tqdm(gen_eval.epoch(), total=gen_eval.num_batches):
                loglik = nps_loglik(
                    model, 
                    batch["contexts"], 
                    batch["xt"], 
                    batch["yt"], 
                    normalise=True
                )
                logliks.append(B.to_numpy(loglik))
                # print(B.to_numpy(loglik))

            logliks = B.concat(*logliks)
            out.kv(f"Loglik ({name})", exp.with_err(logliks, and_lower=True))

    return experiment, model


def create_masked_image(test_batch):
    context_coords = test_batch["contexts"][0][0][0].T
    context_pixels = test_batch["contexts"][0][1][0].squeeze(0)
    
    masked_image = torch.zeros((28, 28, 3))  # Shape: (28, 28, 3) for RGB
    masked_image[:, :, 2] = 1.0  # Full blue in the B channel

    # Convert coordinates from [-1,1] to pixel indices [0,27]
    pixel_x_context = torch.round((context_coords[:, 0] + 1) * 13.5).long()
    pixel_y_context = torch.round((context_coords[:, 1] + 1) * 13.5).long()
    pixel_x_context = torch.clamp(pixel_x_context, 0, 27)
    pixel_y_context = torch.clamp(pixel_y_context, 0, 27)

    for i in range(len(pixel_x_context)):
        grayscale_value = (context_pixels[i].item() + 0.5)  # Get grayscale intensity
        masked_image[pixel_y_context[i], pixel_x_context[i], :] = grayscale_value  # Replace blue pixel with grayscale

    return masked_image


def create_predicted_image(test_batch, model, masked_image):
    xt_single = test_batch["xt_all_non_context"].elements[0][0][0].T  # Extract missing pixel coordinates
    pixel_x = torch.round((xt_single[:, 0] + 1) * 13.5).long()
    pixel_y = torch.round((xt_single[:, 1] + 1) * 13.5).long()

    if args.ar =="no_ar":
        predict_func = nps.predict
    elif args.ar =="old_ar":
        predict_func = ar_predict
    else:
        print("NOT IMPLEMENTED YET!!")
        sys.exit()
    
    with torch.no_grad():
        mean, var, _, _ = predict_func(
            model,
            test_batch["contexts"],
            test_batch["xt_all_non_context"],
            num_samples=args.num_samples,
        )
    pred_normalized_pixel_mean = mean.elements[0][0, 0]
    pred_normalized_pixel_var = var.elements[0][0, 0]

    print("Min pred_normalized_pixel_mean:", pred_normalized_pixel_mean.min().item())
    print("Max pred_normalized_pixel_mean:", pred_normalized_pixel_mean.max().item())
    print("Min pred_normalized_pixel_var:", pred_normalized_pixel_var.min().item())
    print("Max pred_normalized_pixel_var:", pred_normalized_pixel_var.max().item())
    pred_pixel_mean = pred_normalized_pixel_mean + 0.5 # change back to 0-1 values
    pred_pixel_mean = pred_pixel_mean.view(-1) 
    pred_pixel_var = pred_normalized_pixel_var.view(-1) 

    # Convert coordinates from [-1,1] to pixel indices [0,27]
    pixel_x = torch.round((xt_single[:, 0] + 1) * 13.5).long()
    pixel_y = torch.round((xt_single[:, 1] + 1) * 13.5).long()
    assert pixel_x.min().item() == 0
    assert pixel_x.max().item() == 27
    assert pixel_y.min().item() == 0
    assert pixel_y.max().item() == 27

    # completed_image_mean = torch.clone(masked_image)
    # completed_image_var = torch.clone(masked_image)
    mean_image = torch.zeros((28, 28))
    var_image = torch.zeros((28, 28))

    # Fill in the model-predicted pixels
    for i in range(len(pixel_x)):
        grayscale_value_mean = (pred_pixel_mean[i].item())
        mean_image[pixel_y[i], pixel_x[i]] = grayscale_value_mean  # Add to completed image
        grayscale_value_var = (pred_pixel_var[i].item())
        var_image[pixel_y[i], pixel_x[i]] = grayscale_value_var  # Add to completed image

    return mean_image, var_image


def predict_entire_image(test_batch, model):
    if args.ar == "no_ar":
        predict_func = nps.predict
    elif args.ar == "old_ar":
        predict_func = ar_predict
    else:
        raise NotImplementedError("AR variant not implemented.")

    with torch.no_grad():
        mean, var, _, _ = predict_func(
            model,
            test_batch["contexts"],
            test_batch["xt_all"],
            num_samples=args.num_samples,
        )

    # Get predictions for a single image (assumes batch_size = 1)
    mean_flat = mean.elements[0][0, 0] + 0.5 # de-normalize from [-0.5, 0.5] → [0, 1]
    var_flat = var.elements[0][0, 0]

    mean_flat = mean.elements[0][0, 0]
    mean_min = mean_flat.min()
    mean_max = mean_flat.max()
    mean_flat = (mean_flat - mean_min) / (mean_max - mean_min + 1e-8)

    print(f"Mean - min: {mean_flat.min().item():.6f}, max: {mean_flat.max().item():.6f}")
    print(f"Variance - min: {var_flat.min().item():.6f}, max: {var_flat.max().item():.6f}")

    # Get coordinates to reshape flat vector to image
    coords = test_batch["xt_all"].elements[0][0][0].T
    pixel_x = torch.round((coords[:, 0] + 1) * 13.5).long()
    pixel_y = torch.round((coords[:, 1] + 1) * 13.5).long()
    pixel_x = torch.clamp(pixel_x, 0, 27)
    pixel_y = torch.clamp(pixel_y, 0, 27)

    # Create blank images
    mean_image = torch.zeros((28, 28))
    var_image = torch.zeros((28, 28))

    for i in range(len(pixel_x)):
        mean_image[pixel_y[i], pixel_x[i]] = mean_flat[i]
        var_image[pixel_y[i], pixel_x[i]] = var_flat[i]

    return mean_image, var_image



def plot_image_from_xt_yt(xt_all, yt_all, title="Reconstructed Image from xt_all and yt_all"):
    # De-normalize pixel values from [-0.5, 0.5] → [0, 1]
    pixel_values = yt_all[0] + 0.5  # shape: (784,)

    # Convert coordinates from [-1, 1] → [0, 27]
    pixel_x = torch.round((xt_all[0] + 1) * 13.5).long()
    pixel_y = torch.round((xt_all[1] + 1) * 13.5).long()

    # Clamp to avoid indexing errors
    pixel_x = torch.clamp(pixel_x, 0, 27)
    pixel_y = torch.clamp(pixel_y, 0, 27)

    # Reconstruct the image
    image = torch.zeros((28, 28))
    for x, y, val in zip(pixel_x, pixel_y, pixel_values):
        image[y, x] = val  # (y, x) is correct for image coordinates

    # Plot the image
    plt.imshow(image.numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def create_original_image_from_all(test_batch, plot=False):
    original_image = torch.zeros((28, 28)) # blank image

    all_pixels = test_batch["yt_all"].elements[0][0, 0].view(-1)
    all_coords = test_batch["xt_all"].elements[0][0][0].T

    # Convert coordinates from [-1,1] to pixel indices [0,27] with correct rounding
    pixel_x_all = torch.round((all_coords[:, 0] + 1) * 13.5).long()
    pixel_y_all = torch.round((all_coords[:, 1] + 1) * 13.5).long()

    # Place pixel values in the full image
    for i in range(len(pixel_x_all)):
        grayscale_value = (all_pixels[i].item() + 0.5)
        original_image[pixel_y_all[i], pixel_x_all[i]] = grayscale_value
    
    if plot:
        plt.imshow(original_image.numpy(), cmap="gray")
        plt.axis("off")
        plt.show()

    return original_image



if __name__ == "__main__":
    # print("\nEvaluating the model with old loglik")
    # experiment, model = test_emnist()


    training_results_path = os.path.join('code', '_experiments')
    experiment = main(
        model=args.model,
        data="emnist",
        root=training_results_path,
        load=True,
    )
    model = experiment["model"]
    model.load_state_dict(
        torch.load(experiment["wd"].file("model-best.torch"), map_location="cpu", weights_only=False)["weights"]
    )

    # Create label map for EMNIST
    label_map = {}
    with open("code/arnp/datasets/EMNIST/raw/emnist-balanced-mapping.txt", "r") as f:
        for line in f:
            label_id, ascii_code = map(int, line.strip().split())
            label_map[label_id] = chr(ascii_code)


    gens_eval_instances = experiment["gens_eval"]()  # Get evaluation datasets
    for name, gen_eval in gens_eval_instances:
        print(name)

        fixed_index = 10 # choose which batch to take

        for num_context in [1, 40, 200, 728]:
        # for num_context in [728]:
            test_batch = gen_eval.generate_batch(
                num_context=num_context, 
                num_target=784-num_context,
                fixed_index=fixed_index
            )
            label_id = test_batch["labels"][0].item()
            char = label_map[label_id]
            print(f"True label: {char} (label ID: {label_id})")

            # Reconstruct original full image
            original_image = create_original_image_from_all(test_batch)
            print("Finished creating original image")

            # Masked context image (with blue background)
            masked_image = create_masked_image(test_batch)
            print("Finished creating masked image")

            # Predict mean and variance from model
            mean_image, var_image = predict_entire_image(test_batch, model)
            # mean_image, var_image = create_predicted_image(test_batch, model, masked_image)

            # Plot original, context, mean, variance
            fig, axes = plt.subplots(4, 1, figsize=(3, 10))

            axes[0].imshow(original_image.numpy(), cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(masked_image.numpy())
            axes[1].set_title(f"Context ({num_context})")
            axes[1].axis("off")

            axes[2].imshow(mean_image.numpy(), cmap="gray")
            axes[2].set_title("Mean")
            axes[2].axis("off")

            axes[3].imshow(var_image.numpy(), cmap="gray")
            axes[3].set_title("Variance")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(f"figures/emnist_{name.replace(' ', '_').lower()}_context_{num_context}_{args.ar}.png", dpi=300, bbox_inches="tight")
            plt.close()


