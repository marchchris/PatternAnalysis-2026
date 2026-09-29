from datetime import datetime
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt

from config import LABEL_MAP, SEED, DATASET_ROOT, METADATA_PATH
from dataset import create_dataloaders
from modules import build_model

MODEL_NAME = "resnet18"
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

def run_epoch(model, loader, criterion, device, optimizer=None, epoch=None):
    """Train when an optimizer is provided, otherwise evaluate the model"""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0
    total_steps = len(loader)

    # validation and testing do not need gradient calculations
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader, start=1):
            # move images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # calculate loss

            # update weights if training model
            if training:
                loss.backward()
                optimizer.step()

            # weight by batch size so final batch is counted correctly
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += labels.size(0)

            if training and epoch is not None:
                print(
                    f"Epoch {epoch}/{EPOCHS} - "
                    f"Step {step}/{total_steps} "
                    f"({step / total_steps:.0%}) - "
                    f"Loss: {loss.item():.4f}",
                    end="\r",
                    flush=True,
                )

    return total_loss / total_images, total_correct / total_images

def save_training_plot(history, run_dir):
    """Creates and saves loss and accuracy plots of model"""
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [entry["val_loss"] for entry in history]
    train_accuracies = [entry["train_accuracy"] for entry in history]
    val_accuracies = [entry["val_accuracy"] for entry in history]

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5))

    # create loss plot
    loss_axis.plot(epochs, train_losses, label="Training")
    loss_axis.plot(epochs, val_losses, label="Validation")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)

    # create accuracy plot
    accuracy_axis.plot(epochs, train_accuracies, label="Training")
    accuracy_axis.plot(epochs, val_accuracies, label="Validation")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(True, alpha=0.3)

    # save plot under run directory
    figure.tight_layout()
    plot_path = run_dir / "training_history.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f"Saved training plot to: {plot_path}")

def main():
    # allow reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {MODEL_NAME} - Device: {device}")

    # get train, validation, and test data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=DATASET_ROOT,
        metadata_path=METADATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    # create model and move to GPU
    model = build_model(MODEL_NAME, num_classes=len(LABEL_MAP)).to(device)
    criterion = nn.CrossEntropyLoss() # use Cross Entropy Loss

    # use adam optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = []

    # main training loop
    print("\n--- Beginning Training ---")
    for epoch in range(1, EPOCHS + 1):
        # run training epoch
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, epoch=epoch
        )

        # run validation
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        # add current epoch results to history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%}"
        )

    print("\n--- Finished Training ---")

    # create a directory for this runs model and plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path("Results") / MODEL_NAME / f"{MODEL_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save model under the run directory
    model_path = run_dir / f"{MODEL_NAME}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # save training plot under the run directory
    save_training_plot(history, run_dir)


if __name__ == "__main__":
    main()