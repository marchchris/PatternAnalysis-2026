# ADNI MRI Classifcation

## 1. Problem and Approach
This project investigates binary classifcation of two dimensional ADNI brain MRI images into Alzheimer's disease (AD) and congnitively normal (NC) classes. ConvNeXt-Tiny is the selected advanced model, and ResNet-18 provides a standard performance baseline. The engineering question is whether ConvNeXt improves prediction quality and the handling of uncertain cases enough to justify its increased computational cost. A successful model will achieve a minimum test accuracy of `0.80` for the ConvNeXt task, together with a significant increase in performance compared to the baseline model.

The pipeline associates each image with a patient using the `meta_data_with_label.json` file provided with the ADNI dataset. First, the training and testing image splits from the dataset are combined, then the patients are split into `70%` training, `20%` validation, and `10%` testing. The images are then placed into the split with their associated patient. This ensures that there is no patient leakage across the splits. Images are padded with black pixels to upscale them to `256 x 256` pixels, converted to three identical grayscale channels, and pixel values are normalized to values within `[0, 1]`. A selected model produces two class logits and is trained from random initalisation using cross-entropy loss and the AdamW optimizer. Validation runs after every epoch. After training, the final model weights are saved and the model is evaluated against the test set to measure the model's performance.

<p align="center">
  <img src="readme_imgs\comp3710flowchart.png" alt="Logo" width="500">
</p>

*Figure 1: The implemented data and model pipeline for this project.*

## 2. Feasibility Review

### User Need and Scope
The intended user this project is aimed towards is a researcher evaluating a machine learning model that could assist human reviewers in diagnosing Alzheimer's disease from MRI images. The project scope is limited to training and evaluating the models on the course-provided 2D ADNI MRI images, it will not combine multiple brain slices from the same patient to produce a prediction.

### Acceptance Criteria
1. Every included ADNI image has a valid mapping to a patient, and the training, validation, and testing set has no patient leakage.
2. ConvNeXt achieves a test accuracy of atleast `80%`.
3. Training fits within the selected GPU's memory budget, and uses a maximum of `90%` of the available memory.