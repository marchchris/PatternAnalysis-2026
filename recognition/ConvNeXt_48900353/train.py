from datetime import datetime
from pathlib import Path
from time import perf_counter
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset import create_dataloaders
from modules import build_model

DATASET_ROOT = "~/Documents/Datasets/ADNI/AD_NC"
SEED = 42
MODEL_NAME = "convnext"
BATCH_SIZE = 64
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
WARMUP_EPOCHS = 5

CLASS_NAMES = ["NC", "AD"]
LABEL_MAP = {"NC": 0, "AD": 1}
DATASET_SPLIT_NAMES = ["train", "test"]

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    epoch=None,
    scaler=None,
):
    """Train when an optimizer is provided, otherwise evaluate the model"""

    training = optimizer is not None
    amp_enabled = scaler is not None and scaler.is_enabled()
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

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(images) # forward pass
                loss = criterion(outputs, labels) # calculate loss

            # update weights if training model
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # weight by batch size so final batch is counted correctly
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += labels.size(0)

            if epoch is not None:
                phase = "Training" if training else "Validation"
                print(
                    f"{phase} - Epoch {epoch}/{EPOCHS} - "
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

def save_test_report(
    labels,
    predictions,
    ad_probabilities,
    test_loss,
    training_seconds,
    average_epoch_seconds,
    run_dir,
):
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
    training_minutes, training_remaining_seconds = divmod(
        int(training_seconds), 60
    )
    report_lines = [
        "\nModel Training Report:\n"
        f"Training time: {training_minutes}m {training_remaining_seconds}s\n",
        f"Average time per epoch: {average_epoch_seconds:.2f}s\n",

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

@torch.inference_mode()
def evaluate_test(
    model,
    test_loader,
    criterion,
    device,
    training_seconds,
    run_dir,
    average_epoch_seconds,
    scaler=None,
):
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
            amp_enabled = scaler is not None and scaler.is_enabled()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
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
    save_test_report(
        all_labels,
        all_predictions,
        all_ad_probabilities,
        test_loss,
        training_seconds,
        average_epoch_seconds,
        run_dir,
    )

def main():
    # allow reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {MODEL_NAME} training on: {torch.cuda.get_device_name(device)}")

    # get train, validation, and test data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=DATASET_ROOT,
        class_names=CLASS_NAMES,
        dataset_split_names=DATASET_SPLIT_NAMES,
        label_map=LABEL_MAP,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    # create model and move to GPU
    model = build_model(MODEL_NAME, num_classes=len(LABEL_MAP)).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # warm up linearly for five epochs, then decay with cosine annealing
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    cosine_steps = (EPOCHS - WARMUP_EPOCHS) * len(train_loader)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-6,
            ),
        ],
        milestones=[warmup_steps],
    )

    history = []
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None
    epoch_times = []

    # main training loop
    training_start_time = datetime.now()
    print("\n--- Beginning Training ---")
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = perf_counter()

        # run training epoch
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            epoch=epoch,
            scaler=scaler,
        )

        # run validation
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            scaler=scaler,
        )

        scheduler.step()
        epoch_seconds = perf_counter() - epoch_start_time
        epoch_times.append(epoch_seconds)

        # add current epoch results to history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        # if val accuracy is higher, store this model
        if best_model_state is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%} | "
            f"Time: {epoch_seconds:.2f}s"
        )

        # if validation stops improving, early stop
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs "
                "without validation accuracy improvement."
            )
            break

    print("\n--- Finished Training ---")
    training_elapsed = datetime.now() - training_start_time
    training_seconds = int(training_elapsed.total_seconds())
    average_epoch_seconds = sum(epoch_times) / len(epoch_times)

    # use the best validation checkpoint for saving and test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(
        f"Saving model from epoch {best_epoch} "
        f"with validation accuracy {best_val_accuracy:.2%}."
    )

    # create a directory for this runs model and plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path("Models") / MODEL_NAME / f"{MODEL_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save model under the run directory
    model_path = run_dir / f"{MODEL_NAME}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # save training plot under the run directory
    save_training_plot(history, run_dir)

    minutes, seconds = divmod(training_seconds, 60)
    print(f"Training completed in {minutes}m {seconds}s")

    # evaluate the model on the test set
    evaluate_test(
        model,
        test_loader,
        criterion,
        device,
        training_seconds,
        run_dir,
        average_epoch_seconds,
        scaler=scaler,
    )


if __name__ == "__main__":
    main()