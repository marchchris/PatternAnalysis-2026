from datetime import datetime
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from config import LABEL_MAP, SEED, DATASET_ROOT, METADATA_PATH
from dataset import create_dataloaders
from modules import build_model

MODEL_NAME = "resnet18"
BATCH_SIZE = 512
EPOCHS = 2
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

def save_test_report(labels, predictions, ad_probabilities, test_loss, run_dir):
    """Calculate, save, and visualize test statistics with AD as the positive class."""
    # convert the original class labels into binary labels: NC = 0 and AD = 1
    positive_labels = [1 if label == LABEL_MAP["AD"] else 0 for label in labels]
    positive_predictions = [1 if pred == LABEL_MAP["AD"] else 0 for pred in predictions]

    # calculate classification metrics using the binary labels
    accuracy = accuracy_score(positive_labels, positive_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        positive_labels,
        positive_predictions,
        labels=[0, 1],
        zero_division=0,
    )
    # build a confusion matrix and calculate the ROC AUC from AD probabilities
    confusion = confusion_matrix(positive_labels, positive_predictions, labels=[0, 1])
    roc_auc = roc_auc_score(positive_labels, ad_probabilities)

    # format the test metrics as a readable report
    report_lines = [
        "\nTest Set Evaluation Report:",
        f"Test loss: {test_loss:.4f}",
        f"Accuracy: {accuracy:.4f}",
        f"Precision (AD positive): {precision[1]:.4f}",
        f"Recall (AD positive): {recall[1]:.4f}",
        f"F1 Score (AD positive): {f1[1]:.4f}",
        f"ROC AUC: {roc_auc:.4f}",
        f"Support: {support[0] + support[1]}",
        f"Negative support: {support[0]}",
        f"Positive support: {support[1]}\n",
    ]
    report_text = "\n".join(report_lines)

    # save the report to the current run directory and print it
    report_path = run_dir / "test_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Saved test statistics to: {report_path}")

    # create confusion matrix plot
    confusion_figure, confusion_axis = plt.subplots(figsize=(5, 4))
    confusion_image = confusion_axis.imshow(confusion, cmap="Blues")
    confusion_axis.set_title("Confusion Matrix")
    confusion_axis.set_xlabel("Predicted label")
    confusion_axis.set_ylabel("True label")
    confusion_axis.set_xticks([0, 1])
    confusion_axis.set_yticks([0, 1])
    confusion_axis.set_xticklabels(["NC", "AD"])
    confusion_axis.set_yticklabels(["NC", "AD"])

    # display each confusion-matrix count
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            value = confusion[row, col]
            color = "white" if value > confusion.max() / 2 else "black"
            confusion_axis.text(col, row, str(value), ha="center", va="center", color=color)

    # save the confusion matrix plot under the run directory
    confusion_figure.colorbar(confusion_image, ax=confusion_axis)
    confusion_figure.tight_layout()
    confusion_plot_path = run_dir / "confusion_matrix.png"
    confusion_figure.savefig(confusion_plot_path, dpi=150)
    plt.close(confusion_figure)
    print(f"Saved confusion matrix image to: {confusion_plot_path}")


def evaluate_test(model, test_loader, criterion, device, run_dir):
    """Evaluates the test set once after training without updating the model"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_ad_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # calculatee loss

            # softmax probabilities are used for AUC and confidence statistics
            probabilities = torch.softmax(outputs, dim=1)
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            all_ad_probabilities.extend(
                probabilities[:, LABEL_MAP["AD"]].cpu().tolist()
            )

    # save test statistics to run directory
    test_loss = total_loss / len(all_labels)
    save_test_report(all_labels, all_predictions, all_ad_probabilities, test_loss, run_dir)

def main():
    start_time = datetime.now()

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

    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    print(f"Training completed in {minutes}m {seconds}s")

    # evaluate the model on the test set
    evaluate_test(model, test_loader, criterion, device, run_dir)


if __name__ == "__main__":
    main()