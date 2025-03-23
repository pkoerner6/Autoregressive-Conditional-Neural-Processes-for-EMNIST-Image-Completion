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
from new_ar import ar_loglik as new_ar_loglik
from new_ar import ar_predict as new_ar_predict


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--ar", type=str, choices=["no_ar", "old_ar", "new_ar"], required=True)
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
    elif args.ar =="new_ar":
        nps_loglik = new_ar_loglik
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


def predict_entire_image(test_batch, model):
    if args.ar == "no_ar":
        predict_func = nps.predict
    elif args.ar == "old_ar":
        predict_func = ar_predict
    elif args.ar =="new_ar":
        predict_func = new_ar_predict
    else:
        raise NotImplementedError("AR variant not implemented.")

    single_context = [(xc[0:1], yc[0:1]) for xc, yc in test_batch["contexts"]]
    single_xt_all = nps.AggregateInput((test_batch["xt_all"].elements[0][0][0:1], 0))

    with torch.no_grad():
        mean, var, _, _ = predict_func(
            model,
            single_context, #test_batch["contexts"],
            single_xt_all, #test_batch["xt_all"],
            num_samples=args.num_samples,
        )

    # Get predictions for a single image (assumes batch_size = 1)
    mean_flat = mean.elements[0][0, 0] + 0.5 # de-normalize from [-0.5, 0.5] -> [0, 1]
    var_flat = var.elements[0][0, 0]

    print(f"Mean - min: {mean_flat.min().item():.6f}, max: {mean_flat.max().item():.6f}")
    print(f"Variance - min: {var_flat.min().item():.6f}, max: {var_flat.max().item():.6f}")

    # Get coordinates to reshape flat vector to image
    coords = test_batch["xt_all"].elements[0][0][0].T
    pixel_x = torch.round((coords[:, 0] + 1) * 13.5).long()
    pixel_y = torch.round((coords[:, 1] + 1) * 13.5).long()
    pixel_x = torch.clamp(pixel_x, 0, 27)
    pixel_y = torch.clamp(pixel_y, 0, 27)

    mean_image = torch.zeros((28, 28))
    var_image = torch.zeros((28, 28))
    for i in range(len(pixel_x)):
        mean_image[pixel_y[i], pixel_x[i]] = mean_flat[i]
        var_image[pixel_y[i], pixel_x[i]] = var_flat[i]

    return mean_image, var_image


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
    # print(f"\nEvaluating the model with {args.ar}")
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
    with open("code/arnp/datasets/EMNIST/others/emnist-balanced-mapping.txt", "r") as f:
        for line in f:
            label_id, ascii_code = map(int, line.strip().split())
            label_map[label_id] = chr(ascii_code)


    gens_eval_instances = experiment["gens_eval"]()  # Get evaluation datasets
    for name, gen_eval in gens_eval_instances:
        print(f"\n{name}")

        fixed_index = 1 # choose which batch to take

        num_context_list = [1, 50, 100, 200]

        fig, axes = plt.subplots(4, len(num_context_list), figsize=(4 * len(num_context_list), 12))

        for col, num_context in enumerate(num_context_list):
            print("Number of context pixels: ", num_context)
            test_batch = gen_eval.generate_batch(
                num_context=num_context, 
                num_target=784,
                fixed_index=fixed_index
            )
            label_id = test_batch["labels"][0].item()
            char = label_map[label_id]
            print(f"True label: {char} (label ID: {label_id})")

            # Generate all image components
            original_image = create_original_image_from_all(test_batch)
            masked_image = create_masked_image(test_batch)
            mean_image, var_image = predict_entire_image(test_batch, model)

            axes[0, col].imshow(original_image.numpy(), cmap="gray")
            axes[0, col].axis("off")

            axes[1, col].imshow(masked_image.numpy())
            axes[1, col].axis("off")

            axes[2, col].imshow(mean_image.numpy(), cmap="gray")
            axes[2, col].axis("off")

            axes[3, col].imshow(var_image.numpy(), cmap="gray")
            axes[3, col].axis("off")

            axes[0, col].set_title(f"{num_context}", fontsize=20)
        
        fig.text(0.5, 0.95, "Number of context points", ha="center", fontsize=24)

        fig.text(0.05, 0.8, "Original", ha="center", va="center", rotation=90, fontsize=20)
        fig.text(0.05, 0.57, "Masked", ha="center", va="center", rotation=90, fontsize=20)
        fig.text(0.05, 0.35, "Mean", ha="center", va="center", rotation=90, fontsize=20)
        fig.text(0.05, 0.13, "Variance", ha="center", va="center", rotation=90, fontsize=20)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"figures/emnist_{name.replace(' ', '_').lower()}_{args.model}_{args.ar}.png", dpi=300, bbox_inches="tight")
        plt.close()


