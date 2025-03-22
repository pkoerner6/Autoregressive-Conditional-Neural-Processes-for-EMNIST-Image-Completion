import torch
import torchvision.datasets as tvds
import os
import sys
import lab as B
from neuralprocesses.aggregate import Aggregate, AggregateInput
from neuralprocesses.data.data import DataGenerator
from .util import register_data

import matplotlib.pyplot as plt

datasets_path = os.path.join('code', 'arnp', 'datasets')

class EMNIST(tvds.EMNIST):
    def __init__(self, train=True, class_range=[0, 10], device='cpu', download=False):
        super().__init__(datasets_path, train=train, split='balanced', download=download)

        self.data = self.data.unsqueeze(1).float().div(255).transpose(-1, -2).to(device)
        self.targets = self.targets.to(device)

        # Filter classes based on class_range
        idxs = []
        for c in range(class_range[0], class_range[1]): 
            idxs.append(torch.where(self.targets == c)[0])
        idxs = torch.cat(idxs)

        self.data = self.data[idxs]
        self.targets = self.targets[idxs]


    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class EmnistGenerator(DataGenerator):
    """EMNIST Meta-Regression Generator with proper data splits."""

    def __init__(
            self, 
            dtype, 
            seed=0, 
            num_tasks=2**14, 
            batch_size=16, 
            train=True,
            subset=None,  # "cv" or "eval" for non-overlapping validation/evaluation sets
            device="cpu",
            class_range=[0, 47]
        ):
        """
        Args:
            dtype (dtype): Data type to generate.
            seed (int, optional): Random seed. Defaults to 0.
            num_tasks (int, optional): Number of tasks per epoch. Defaults to 2^14.
            batch_size (int, optional): Batch size. Defaults to 16.
            train (bool, optional): Whether to load training or test data. Defaults to True.
            subset (str, optional): If "cv", loads first half of test set; 
                                    if "eval", loads second half. Defaults to None.
            device (str, optional): Device ("cpu" or "cuda"). Defaults to "cpu".
        """
        
        super().__init__(dtype=dtype, seed=seed, num_tasks=num_tasks, batch_size=batch_size, device=device)
        
        self.seed = seed
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
        self.data, self.targets = self.load_emnist(train, subset, device, class_range)


    def load_emnist(self, train, subset, device, class_range):
        """Loads the EMNIST dataset and splits the test set for cv and eval."""
        dataset = EMNIST(train=train, class_range=class_range, device=device, download=False)

        if not train and subset is not None:
            num_samples = len(dataset.data)

            # Generate a random permutation of indices
            indices = torch.randperm(num_samples)
            shuffled_data = dataset.data[indices]
            shuffled_targets = dataset.targets[indices]

            split_idx = num_samples // 2  # two equal halves, one for cv, one for eval

            if subset == "cv":
                data = shuffled_data[:split_idx]
                targets = shuffled_targets[:split_idx]
            elif subset == "eval":
                data = shuffled_data[split_idx:]
                targets = shuffled_targets[split_idx:]
            else:
                raise ValueError('Subset must be either "cv" or "eval" for test set.')

        else:
            data, targets = dataset.data, dataset.targets

        return data, targets

    
    def generate_batch(self):
        """Generates a batch for EMNIST Meta-Regression.

        Returns:
            dict: A batch with keys "contexts", "xt", and "yt".
        """
        with B.on_device(self.device):
            # Select a batch of images randomly
            indices = torch.randint(0, len(self.data), (self.batch_size,))
            images = self.data[indices].to(self.device)  # Shape: (batch_size, 1, 28, 28)

            # Flatten images to 1D (28x28 = 784)
            images = images.reshape(self.batch_size, 1, -1)  # (batch_size, 1, 784)

            # Generate normalized pixel coordinates in range [-1, 1]
            coords = torch.linspace(-1, 1, 28)  # Generate 1D grid
            row_grid, col_grid = torch.meshgrid(coords, coords, indexing="ij")
            pixel_coords = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=-1)  # Shape: (784, 2)
            pixel_coords = pixel_coords.to(self.device).expand(self.batch_size, -1, -1)  # (batch_size, 784, 2)

            # Randomly sample context and target indices
            num_context = torch.randint(3, 197, (1,)).item()  # Sample N from U[3,197)
            num_target = torch.randint(3, 200 - num_context, (1,)).item()  # Sample M from U[3, 200-N)

            all_indices = torch.randperm(28*28)  # Shuffle pixel indices
            context_indices = all_indices[:num_context]
            target_indices = all_indices[num_context : num_context + num_target]
            non_context_indices = all_indices[num_context:]

            # Extract context and target pixel coordinates
            xc = B.take(pixel_coords, context_indices, axis=1)
            xt = B.take(pixel_coords, target_indices, axis=1)

            # Extract pixel values and normalize to [-0.5, 0.5]
            yc = B.take(images, context_indices, axis=2) - 0.5  # (batch_size, 1, N)
            yt = B.take(images, target_indices, axis=2) - 0.5  # (batch_size, 1, M)

            # Extract remaining pixels for full image completion (excluding context pixels)
            xt_all_other = B.take(pixel_coords, non_context_indices, axis=1)  # (batch_size, remaining_pixels, 2)
            yt_all_other = B.take(images, non_context_indices, axis=2) - 0.5  # (batch_size, 1, remaining_pixels)

            # Reshape to match (*b, c, *n) convention
            xc = xc.permute(0, 2, 1)  # (batch_size, 2, N)  -- c=2 for (x, y)
            xt = xt.permute(0, 2, 1)  # (batch_size, 2, M)
            xt_all_other = xt_all_other.permute(0, 2, 1)  # (batch_size, 2, remaining_pixels)
            batch = {
                "contexts": [(xc, yc)],  # Ensure correct tensor format for context
                "xt": AggregateInput((xt, 0)),  # Ensure xt follows correct convention
                "yt": Aggregate(yt),  # Ensure yt follows correct convention
                "xt_all_other": AggregateInput((xt_all_other, 0)),  # All pixels except context
                "yt_all_other": Aggregate(yt_all_other),  # Ground truth values for all other pixels
            }  
            return batch


def setup(
        args, 
        config, 
        *, 
        num_tasks_train, 
        num_tasks_cv, 
        num_tasks_eval, 
        device, 
        visualize_images=False,
    ):
    config["default"]["rate"] = 1e-4
    config["default"]["epochs"] = 200
    config["dim_x"] = 2  # x-coordinates are 2D (pixel location)
    config["dim_y"] = 1  # Predicting single pixel intensity

    config["batch_size"] = args.batch_size

    config["image_size"] = 28  # EMNIST images are 28X28

    config["transform"] = (-0.5, 0.5)

    # Configure convolutional models:
    config["points_per_unit"] = 4
    config["margin"] = 1
    config["conv_receptive_field"] = 100
    config["unet_strides"] = (1,) + (2,) * 6
    config["unet_channels"] = (64,) * 7


    gen_train = EmnistGenerator(
        dtype=torch.float32, 
        seed=0, 
        num_tasks=num_tasks_train, 
        batch_size=args.batch_size, 
        train=True,
        subset=None,
        device=device,
        class_range=[0, 10]
    )
    gen_cv = lambda: EmnistGenerator(
        dtype=torch.float32, 
        seed=1, 
        num_tasks=num_tasks_cv, 
        batch_size=args.batch_size,
        train=False,
        subset="cv",
        device=device,
        class_range=[0, 10]
    )

    def gens_eval():
        return [
            ("EMNIST Seen (0-9)", EmnistGenerator(
                torch.float32, 
                seed=2, 
                batch_size=args.batch_size,
                num_tasks=num_tasks_eval, 
                train=False,
                subset="eval",
                device=device,
                class_range=[0, 10]
            )),
            ("EMNIST Unseen (10-46)", EmnistGenerator(
                torch.float32, 
                seed=3, 
                batch_size=args.batch_size,
                num_tasks=num_tasks_eval, 
                train=False,
                subset="eval",
                device=device,
                class_range=[10, 47]
            ))
        ]
    
    # The EMNIST balanced dataset contains 131,600 images
    print(f"Training Dataset Size: {len(gen_train.data)}") # 112,800, but only take 0-9 so only 24,000
    gen_cv_instance = gen_cv() 
    print(f"CV Dataset Size: {len(gen_cv_instance.data)}") # 9,400, but only take 0-9 so only 2000
    gens_eval_instances = gens_eval() 
    for name, gen_eval in gens_eval_instances:
        print(f"Test {name} Dataset Size: {len(gen_eval.data)}") # EMNIST Seen (0-9): 2000; EMNIST Unseen (10-46): 7400

    def visualize_emnist_image(generator, dataset_name):
        """Visualizes the first image from the dataset."""
        for i in range(1):
            image, label = generator.data[i].cpu(), generator.targets[i].cpu()
            image = image.squeeze(0)

            plt.imshow(image, cmap="gray")
            plt.title(f"{dataset_name} - Label: {label.item()}")
            plt.axis("off")
            plt.show()

    if visualize_images:
        visualize_emnist_image(gen_train, "Train Dataset")
        # EMNIST Seen (0-9) eval set
        eval_name, eval_gen = gens_eval_instances[0]
        visualize_emnist_image(eval_gen, eval_name)
        # EMNIST Unseen (10-46) eval set
        eval_name, eval_gen = gens_eval_instances[1]
        visualize_emnist_image(eval_gen, eval_name)

    # Print unique labels
    def print_dataset_labels(generator, dataset_name):
        """Print all unique labels in the dataset."""
        unique_labels = torch.unique(generator.targets).tolist()
        print(f"{dataset_name} contains {len(unique_labels)} unique labels: {unique_labels}")
    print_dataset_labels(gen_train, "Train Dataset")
    print_dataset_labels(gen_cv_instance, "CV Dataset")
    for name, gen_eval in gens_eval_instances:
        print_dataset_labels(gen_eval, "Test " + name)
    return gen_train, gen_cv, gens_eval

register_data("emnist", setup)



