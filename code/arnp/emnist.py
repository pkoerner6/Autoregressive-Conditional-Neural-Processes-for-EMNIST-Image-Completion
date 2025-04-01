import argparse
import matplotlib.pyplot as plt
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
from ar import ar_predict


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--ar", type=str, choices=["no_ar", "old_ar", "new_ar"], required=True)
parser.add_argument("--arch", type=str, choices=["unet", "unet-res"], required=True)
parser.add_argument("--num_samples", type=int, default=100)
args = parser.parse_args()



def test_emnist():
    training_results_path = os.path.join('code', '_experiments')

    experiment = main(
        model=args.model,
        data="emnist",
        root=training_results_path,
        load=True,
        arch=args.arch,
    )
    print(f"Loading model from: {experiment['wd'].root}")

    model = experiment["model"]
    model.load_state_dict(
        torch.load(experiment["wd"].file("model-best.torch"), map_location="cpu", weights_only=False)["weights"]
    )

    if args.ar =="no_ar":
        nps_loglik = nps.loglik
    elif args.ar =="old_ar":
        nps_loglik = nps.ar_loglik
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
    single_context = [(xc[0:1], yc[0:1]) for xc, yc in test_batch["contexts"]]
    single_xt_all = nps.AggregateInput((test_batch["xt_all"].elements[0][0][0:1], 0))

    if args.ar == "no_ar":
        with torch.no_grad():
            mean, var, ft, _ = nps.predict(
                model,
                single_context, 
                single_xt_all,
                num_samples=args.num_samples,
            )
            mean_flat = mean.elements[0][0, 0] + 0.5 # de-normalize from [-0.5, 0.5] -> [0, 1]
            var_flat = var.elements[0][0, 0]
            print(f"Mean - min: {mean_flat.min().item():.6f}, max: {mean_flat.max().item():.6f}")
            print(f"Variance - min: {var_flat.min().item():.6f}, max: {var_flat.max().item():.6f}")
    elif args.ar == "old_ar":
        with torch.no_grad():
            mean, var, ft, ft_var, _ = ar_predict(
                model,
                single_context, 
                single_xt_all,
                num_samples=args.num_samples,
                order="random"
            )
            mean_flat = mean.elements[0][0, 0] + 0.5 # de-normalize from [-0.5, 0.5] -> [0, 1]
            var_flat = var.elements[0][0, 0]
            smoothed_mean_flat = ft.elements[0][0, 0, 0] + 0.5 # de-normalize from [-0.5, 0.5] -> [0, 1] 
            print(f"Mean - min: {mean_flat.min().item():.6f}, max: {mean_flat.max().item():.6f}")
            print(f"Variance - min: {var_flat.min().item():.6f}, max: {var_flat.max().item():.6f}")
            print(f"smoothed mean - min: {smoothed_mean_flat.min().item():.6f}, max: {smoothed_mean_flat.max().item():.6f}")
    else:
        raise NotImplementedError("AR variant not implemented.")

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

    smoothed_mean_image = torch.zeros((28, 28))
    if args.ar == "old_ar":
        for i in range(len(pixel_x)):
            smoothed_mean_image[pixel_y[i], pixel_x[i]] = smoothed_mean_flat[i]

    return mean_image, var_image, smoothed_mean_image



if __name__ == "__main__":
    # print(f"\nEvaluating the model with {args.ar}")
    # experiment, model = test_emnist()


    training_results_path = os.path.join('code', '_experiments')
    experiment = main(
        model=args.model,
        data="emnist",
        root=training_results_path,
        load=True,
        arch=args.arch,
    )
    model = experiment["model"]
    model.load_state_dict(
        torch.load(experiment["wd"].file("model-best.torch"), map_location="cpu", weights_only=False)["weights"]
    )

    label_map = {}
    with open("code/arnp/datasets/EMNIST/others/emnist-balanced-mapping.txt", "r") as f:
        for line in f:
            label_id, ascii_code = map(int, line.strip().split())
            label_map[label_id] = chr(ascii_code)


    gens_eval_instances = experiment["gens_eval"]()  # Get evaluation datasets
    for name, gen_eval in gens_eval_instances:
        print(f"\n{name}")

        fixed_index = 3 # for regularized convcnp; 7 for old convcnp

        num_context_list = [1, 40, 200, 784]

        if args.ar == "old_ar":
            fig, axes = plt.subplots(4, len(num_context_list), figsize=(4 * len(num_context_list), 12))
        else: 
            fig, axes = plt.subplots(3, len(num_context_list), figsize=(4 * len(num_context_list), 9))

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
            masked_image = create_masked_image(test_batch)
            mean_image, var_image, smoothed_mean_image = predict_entire_image(test_batch, model)

            if args.ar == "old_ar":
                axes[0, col].imshow(masked_image.numpy())
                axes[0, col].axis("off")

                axes[1, col].imshow(mean_image.numpy(), cmap="gray")
                axes[1, col].axis("off")

                axes[2, col].imshow(var_image.numpy(), cmap="gray")
                axes[2, col].axis("off")

                axes[3, col].imshow(smoothed_mean_image.numpy(), cmap="gray")
                axes[3, col].axis("off")

                axes[0, col].set_title(f"{num_context}", fontsize=28)
            else:
                axes[0, col].imshow(masked_image.numpy())
                axes[0, col].axis("off")

                axes[1, col].imshow(mean_image.numpy(), cmap="gray")
                axes[1, col].axis("off")

                axes[2, col].imshow(var_image.numpy(), cmap="gray")
                axes[2, col].axis("off")

                axes[0, col].set_title(f"{num_context}", fontsize=28)
        
        fig.text(0.5, 0.95, "Number of context points", ha="center", fontsize=28)

        if args.ar == "old_ar":
            fig.text(0.05, 0.80, "Context", ha="center", va="center", rotation=90, fontsize=28) 
            fig.text(0.05, 0.57, "Mean", ha="center", va="center", rotation=90, fontsize=28) 
            fig.text(0.05, 0.35, "Variance", ha="center", va="center", rotation=90, fontsize=28) 
            fig.text(0.05, 0.13, "Smoothed", ha="center", va="center", rotation=90, fontsize=28)
        else:
            fig.text(0.05, 0.74, "Context", ha="center", va="center", rotation=90, fontsize=28)
            fig.text(0.05, 0.45, "Mean", ha="center", va="center", rotation=90, fontsize=28)
            fig.text(0.05, 0.16, "Variance", ha="center", va="center", rotation=90, fontsize=28)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"figures/{name.replace(' ', '_').lower()}_{args.model}_{args.ar}_{args.arch}.png", dpi=300, bbox_inches="tight")
        plt.close()


