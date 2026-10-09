"""Loads trained models and run inference on hold out images.

Run from the code/ directory:
python predict.py --model [model_name] --weights [saved_model_path]
"""

import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch

from config import DATASET_ROOT, LABEL_MAP, SEED
from dataset import create_splits, preprocess_image
from modules import build_model

def load_model(model_name, weights_path, device):
    """Load the saved model"""
    model = build_model(model_name, num_classes=len(LABEL_MAP))

    state_dict = torch.load(
        Path(weights_path).expanduser(), map_location="cpu", weights_only=True
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def predict_image(model, image_path, device):
    """Returns the preprocessed image and probabilities"""
    with Image.open(image_path) as image:
        image_tensor = preprocess_image(image) # preprocess the image

    with torch.inference_mode(): # disable gradient calculations
        logits = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    return image_tensor, probabilities

def main():
    # parse args from command line
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet18", "convnext"], required=True)
    parser.add_argument("--weights", required=True, help="Path to the saved .pth file")
    parser.add_argument("--image", help="Optional single image to predict")
    parser.add_argument("--num-images", type=int, default=9, help="Number of test examples")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output", default="Inferences/predictions.png", help="Output figure path")
    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("--num-images must be at least 1")

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the saved model
    model = load_model(args.model, args.weights, device)
    index_to_class = {index: name for name, index in LABEL_MAP.items()}
    print(f"Model: {args.model} - Device: {device}")

    if args.image:
        # use provided images if paths are provided
        imgs = [{"image_path": str(Path(args.image).expanduser()), "class_name": None}]
    else:
        # otherwise use images from test split
        _, _, test_frame = create_splits(
            Path(args.dataset_root).expanduser(),
            seed=SEED
        )

        # sample images from test split
        imgs = test_frame.sample(n=args.num_images, random_state=SEED).to_dict("records")

        # calculate how many rows and columns needed in output image
        columns = min(3, len(imgs))
        rows = (len(imgs) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False
        )

        correct = 0
        for index, img in enumerate(imgs):
            image_path = Path(img["image_path"])
            true_class = img["class_name"]

            # run inference on images
            image_tensor, probabilities = predict_image(model, image_path, device)

            # get model outputs
            predicted_index = probabilities.argmax().item()
            predicted_class = index_to_class[predicted_index]
            confidence = probabilities[predicted_index].item()
            nc_probability = probabilities[LABEL_MAP["NC"]].item()
            ad_probability = probabilities[LABEL_MAP["AD"]].item()

            # calculate number of correct predictions
            correct += int(predicted_class == true_class)

            # print each run of inference
            print(
                f"\nImage: {image_path}\n"
                f"True: {true_class or 'unknown'} | Predicted: {predicted_class} | "
                f"Confidence: {confidence:.2%}\n"
                f"P(NC): {nc_probability:.4f} | P(AD): {ad_probability:.4f}"
            )

            axis = axes[index // columns, index % columns]
            axis.imshow(image_tensor[0].numpy(), cmap="gray", vmin=0, vmax=1)
            colour = "green" if predicted_class == true_class else "red"

            axis.set_title(
                f"{image_path.name}\nTrue: {true_class or 'unknown'} | "
                f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}",
                color=colour,
                fontsize=10,
            )

        # only print test accuracy if using images from test set
        if not args.image:
            print(
                f"\nDisplayed-example accuracy: {correct}/{len(imgs)} "
                f"({correct / len(imgs):.2%})"
            )

        figure.suptitle(f"{args.model}: ADNI Predictions")
        figure.tight_layout(rect=(0, 0, 1, 0.96))

        # save image to specified path with current date and time appended to end of image name
        output_path = Path(args.output).expanduser()
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
        output_path = output_path.with_name( # change image name to have date and time appended
            f"{output_path.stem}{timestamp}{output_path.suffix}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"Saved predictions plot to: {output_path}")
                

if __name__ == "__main__":
    main()