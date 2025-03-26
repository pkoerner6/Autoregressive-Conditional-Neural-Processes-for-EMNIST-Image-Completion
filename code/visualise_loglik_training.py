import argparse
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["convcnp", "convgnp"])
args = parser.parse_args()              

log_path = f"code/_experiments/emnist/{args.model}/unet/loglik/log_train_out.txt"

epochs = []
train_loglik = []
train_std = []
val_loglik = []
val_std = []

train_pattern = re.compile(r"Loglik \(T\):\s*([-\d.]+) \+-\s*([-\d.]+)")
val_pattern = re.compile(r"Loglik \(V\):\s*([-\d.]+) \+-\s*([-\d.]+)")
epoch_pattern = re.compile(r"Epoch (\d+)")

with open(log_path, "r") as f:
    current_epoch = None
    for line in f:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

        train_match = train_pattern.search(line)
        if train_match:
            train_loglik.append(float(train_match.group(1)))
            train_std.append(float(train_match.group(2)))

        val_match = val_pattern.search(line)
        if val_match:
            val_loglik.append(float(val_match.group(1)))
            val_std.append(float(val_match.group(2)))

plt.figure(figsize=(10, 6))
plt.errorbar(epochs[:len(train_loglik)], train_loglik, yerr=train_std, label="Train Loglik", fmt='-o')
plt.errorbar(epochs[:len(val_loglik)], val_loglik, yerr=val_std, label="Validation Loglik", fmt='-o')
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Log-Likelihood", fontsize=16)
plt.title(f"Training and Validation Log-Likelihood over Epochs for {args.model}", fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/training_validation_loglik_{args.model}.png", dpi=300, bbox_inches="tight")
# plt.show()


plt.figure(figsize=(10, 6))
plt.plot(epochs[:len(train_std)], train_std, label="Train Std (Variance)")
plt.plot(epochs[:len(val_std)], val_std, label="Validation Std (Variance)")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Log-Likelihood Std", fontsize=16)
plt.title(f"Log-Likelihood Variance over Epochs for {args.model}", fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/loglik_variance_only_{args.model}.png", dpi=300, bbox_inches="tight")
plt.show()


