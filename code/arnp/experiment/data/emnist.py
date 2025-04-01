import torch
import torchvision.datasets as tvds
import os
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
            class_range=[0, 47],
            training_epoch=None,
            max_epochs=None,
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
        self.training_epoch = training_epoch
        self.max_epochs = max_epochs
        
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

    
    def generate_batch(self, num_context=None, num_target=None, fixed_index=None):
        """Generates a batch for EMNIST Meta-Regression.

        Returns:
            dict: A batch with keys "contexts", "xt", and "yt".
        """
        
        with B.on_device(self.device):
            # Select a batch of images randomly
            if fixed_index is not None:
                indices = torch.tensor([fixed_index] * self.batch_size)
            else:
                indices = torch.randint(0, len(self.data), (self.batch_size,))

            labels = self.targets[indices] 

            images = self.data[indices].to(self.device)  # Shape: (batch_size, 1, 28, 28)

            # Flatten images to 1D (28x28 = 784)
            images = images.reshape(self.batch_size, 1, -1)  # (batch_size, 1, 784)

            # Generate normalized pixel coordinates in range [-1, 1]
            coords = torch.linspace(-1, 1, 28)  # Generate 1D grid
            row_grid, col_grid = torch.meshgrid(coords, coords, indexing="ij")
            pixel_coords = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=-1)  # Shape: (784, 2)
            pixel_coords = pixel_coords.to(self.device).expand(self.batch_size, -1, -1)  # (batch_size, 784, 2)

            if self.training_epoch is not None and num_target is None:
                progress = self.training_epoch / self.max_epochs

                # Linearly interpolate context and target sizes
                max_context = 397
                min_context = 3
                num_context = int(max_context * (1 - progress)) + min_context
                num_context = min(num_context, 784 - 3)

                max_target = 400
                min_target = 3
                num_target = int(min_target + progress * (max_target - min_target))
                num_target = min(num_target, 784 - num_context)  # ensure it fits
            else:
                if num_context is None:
                    num_context = torch.randint(3, 197, (1,)).item() 
                if num_target is None:
                    num_target = torch.randint(3, 200 - num_context, (1,)).item()

            all_indices = torch.randperm(28*28)  # Shuffle pixel indices
            context_indices = all_indices[:num_context]
            target_indices = all_indices[num_context : num_context + num_target]

            # Extract context and target pixel coordinates
            xc = B.take(pixel_coords, context_indices, axis=1)
            xt = B.take(pixel_coords, target_indices, axis=1)

            # Extract pixel values and normalize to [-0.5, 0.5]
            yc = B.take(images, context_indices, axis=2) - 0.5  # (batch_size, 1, N)
            yt = B.take(images, target_indices, axis=2) - 0.5  # (batch_size, 1, M)
            
            xt_all = pixel_coords.permute(0, 2, 1)  # (batch_size, 2, 784)
            yt_all = images - 0.5 

            # Reshape to match (*b, c, *n) convention
            xc = xc.permute(0, 2, 1)  # (batch_size, 2, N)  -- c=2 for (x, y)
            xt = xt.permute(0, 2, 1)  # (batch_size, 2, M)

            batch = {
                "contexts": [(xc, yc)],
                "xt": AggregateInput((xt, 0)),
                "yt": Aggregate(yt),
                "xt_all": AggregateInput((xt_all, 0)), 
                "yt_all": Aggregate(yt_all),   
                "labels": labels 
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
        training_epoch=None,
        max_epochs=None,
    ):
    config["default"]["rate"] = 1e-4
    config["default"]["epochs"] = 200
    config["dim_x"] = 2  # x-coordinates are 2D (pixel location)
    config["dim_y"] = 1  # Predicting single pixel intensity

    config["batch_size"] = args.batch_size

    config["image_size"] = 28  # EMNIST images are 28X28

    config["transform"] = None
    config["fix_noise"] = False

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
        class_range=[0, 10],
        training_epoch=training_epoch,
        max_epochs=max_epochs,
    )
    gen_cv = lambda: EmnistGenerator(
        dtype=torch.float32, 
        seed=1, 
        num_tasks=num_tasks_cv, 
        batch_size=args.batch_size,
        train=False,
        subset="cv",
        device=device,
        class_range=[0, 10],
        training_epoch=training_epoch,
        max_epochs=max_epochs,
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
                class_range=[0, 10],
                training_epoch=None,
                max_epochs=None,
            )),
            ("EMNIST Unseen (10-46)", EmnistGenerator(
                torch.float32, 
                seed=3, 
                batch_size=args.batch_size,
                num_tasks=num_tasks_eval, 
                train=False,
                subset="eval",
                device=device,
                class_range=[10, 47],
                training_epoch=None,
                max_epochs=None,
            ))
        ]
    
    if training_epoch==0:
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



