from datetime import datetime
from pathlib import Path
import torch
from torch import nn

from config import LABEL_MAP, SEED, DATASET_ROOT, METADATA_PATH
from dataset import create_dataloaders
from modules import build_model

MODEL_NAME = "convnext"
BATCH_SIZE = 128
EPOCHS = 30
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

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%}"
        )

    print("\n--- Finished Training ---")

    # create directory for model to be saved to
    results_dir = Path("Results") / MODEL_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    # save model under its directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"{MODEL_NAME}_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()