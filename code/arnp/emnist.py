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

            mean_loglik = B.mean(logliks) - 1.96 * B.std(logliks) / B.sqrt(len(logliks))
            print(f"Mean Log-likelihood on {name}: {mean_loglik:.4f}")
    
    return experiment, model


def create_original_image(test_batch):
    original_image = torch.zeros((28, 28)) # blank image

    # Extract single image's target pixel
    yt_single = test_batch["yt_all_other"].elements[0][0, 0].view(-1)
    context_pixels = test_batch["contexts"][0][1][0].squeeze(0)
    all_pixels = torch.cat([context_pixels, yt_single])

    xt_single = test_batch["xt_all_other"].elements[0][0][0].T
    context_coords = test_batch["contexts"][0][0][0].T
    all_coords = torch.cat([context_coords, xt_single]) 

    # Convert coordinates from [-1,1] to pixel indices [0,27] with correct rounding
    pixel_x_all = torch.round((all_coords[:, 0] + 1) * 13.5).long()
    pixel_y_all = torch.round((all_coords[:, 1] + 1) * 13.5).long()

    # Place pixel values in the full image
    for i in range(len(pixel_x_all)):
        grayscale_value = (all_pixels[i].item() + 0.5)
        original_image[pixel_y_all[i], pixel_x_all[i]] = grayscale_value

    num_context_pixels = test_batch["contexts"][0][1].shape[-1]  # Number of context pixels
    num_yt_all_other_pixels = test_batch["yt_all_other"].elements[0].shape[-1]  # Pixels in yt_all_other
    print(f"Number of context pixels: {num_context_pixels}, taget pixels: {num_yt_all_other_pixels}, total: {num_context_pixels+num_yt_all_other_pixels}")

    return original_image


def create_masked_image(test_batch):
    context_pixels = test_batch["contexts"][0][1][0].squeeze(0)
    context_coords = test_batch["contexts"][0][0][0].T
    
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
    xt_single = test_batch["xt_all_other"].elements[0][0][0].T  # Extract missing pixel coordinates
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
        mean, _, _, _ = predict_func(
            model,
            test_batch["contexts"],
            test_batch["xt_all_other"],
            num_samples=args.num_samples,
        )
    pred_normalized_pixels = mean.elements[0][0, 0]
    # TODO predicting normalized pixel values are not in the range [-0.5, 0.5]
    # assert pred_normalized_pixels.min().item() >= -0.5
    # assert pred_normalized_pixels.max().item() <= 0.5
    # print("pred_normalized_pixels: ", pred_normalized_pixels)
    print("Min pred_normalized_pixels:", pred_normalized_pixels.min().item())
    print("Max pred_normalized_pixels:", pred_normalized_pixels.max().item())
    pred_pixels = pred_normalized_pixels + 0.5 # change back to 0-1 values
    pred_pixels = pred_pixels.view(-1) 

    completed_image = torch.clone(masked_image)
    # Convert coordinates from [-1,1] to pixel indices [0,27]
    pixel_x = torch.round((xt_single[:, 0] + 1) * 13.5).long()
    pixel_y = torch.round((xt_single[:, 1] + 1) * 13.5).long()
    assert pixel_x.min().item() == 0
    assert pixel_x.max().item() == 27
    assert pixel_y.min().item() == 0
    assert pixel_y.max().item() == 27

    # Fill in the model-predicted pixels
    for i in range(len(pixel_x)):
        grayscale_value = (pred_pixels[i].item())
        completed_image[pixel_y[i], pixel_x[i], :] = grayscale_value  # Add to completed image

    return completed_image


def plot_image(image_tensor, title, ax):
    """Helper function to plot an image from a tensor."""
    image = image_tensor.squeeze(0).cpu().numpy() 
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def plot_original_masked_predicted(original_image, masked_image, completed_image, name):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_image(original_image, "Original Image (Full)", axes[0])
    print("After original_image")

    axes[1].imshow(masked_image.numpy(), cmap=None)
    axes[1].set_title("Masked Image") # (Blue = Pixels not in Context)
    axes[1].axis("off")
    print("After masked_image")

    # print(completed_image)
    print("Min:", completed_image.min().item())
    print("Max:", completed_image.max().item())
    axes[2].imshow(completed_image.numpy(), cmap=None)
    axes[2].set_title("Completed Image (Model Predictions)")
    axes[2].axis("off")
    print("After completed_image")

    plt.tight_layout()
    plt.savefig(f"figures/emnist_original_masked_predicted_{name}_{args.ar}.png", dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    print("\nEvaluating the model with old loglik")
    experiment, model = test_emnist()

    print("\nMaking predictions for two whole test images")

    gens_eval_instances = experiment["gens_eval"]()  # Get evaluation datasets
    for name, gen_eval in gens_eval_instances:
        print(name)
        test_batches = gen_eval.generate_batch() # Get single test batch
        # print(f"Number of images in one {name} test_batch: {test_batches['yt'].elements[0].shape[0]}")
        original_image = create_original_image(test_batches)
        print("Finished creating original image")
        masked_image = create_masked_image(test_batches)
        print("Finished creating masked image")
        completed_image = create_predicted_image(test_batches, model, masked_image)
        print("Finished completed masked image")
        # completed_image = masked_image
        plot_original_masked_predicted(original_image, masked_image, completed_image, name)

