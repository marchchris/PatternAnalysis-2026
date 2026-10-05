# ADNI MRI Classifcation

## 1. Problem and Approach
This project investigates binary classifcation of two dimensional ADNI brain MRI images into Alzheimer's disease (AD) and congnitively normal (NC) classes. ConvNeXt-Tiny is the selected advanced model, and ResNet-18 provides a standard performance baseline. The engineering question is whether ConvNeXt improves prediction quality and the handling of uncertain cases enough to justify its increased computational cost. A successful model will achieve a minimum test accuracy of `0.80` for the ConvNeXt task, together with a significant increase in performance compared to the baseline model.

The pipeline associates each image with a patient using the `meta_data_with_label.json` file provided with the ADNI dataset. First, the training and testing image splits from the dataset are combined, then the patients are split into `70%` training, `20%` validation, and `10%` testing. The images are then placed into the split with their associated patient. This ensures that there is no patient leakage across the splits. Images are padded with black pixels to upscale them to `256 x 256` pixels, converted to three identical grayscale channels, and pixel values are normalized to values within `[0, 1]`. A selected model produces two class logits and is trained from random initalisation using cross-entropy loss and the AdamW optimizer. Validation runs after every epoch. After training, the final model weights are saved and the model is evaluated against the test set to measure the model's performance. The `predict.py` script can then be used to load a saved model, and run inferencce on sampled test images from the training set or one supplied image. It prints the class probabilities of each prediction and saves an annotated figure of the models predictions.

<p align="center">
  <img src="readme_imgs\flowchart.png" alt="Flowchart" width="500">
</p>

*Figure 1: The implemented data and model pipeline for this project.*

## 2. Feasibility Review

### User Need and Scope
The intended user this project is aimed towards is a researcher evaluating a machine learning model that could assist human reviewers in diagnosing Alzheimer's disease from MRI images. The project scope is limited to training and evaluating the models on the course-provided 2D ADNI MRI images, it will not combine multiple brain slices from the same patient to produce a prediction.

### Acceptance Criteria
1. Every included ADNI image has a valid mapping to a patient, and the training, validation, and testing set has no patient leakage.
2. ConvNeXt achieves a test accuracy of atleast `80%`.
3. Training fits within the selected GPU's memory budget, and uses a maximum of `90%` of the available memory.

### Model Choice and Course Concepts
ConvNeXt-Tiny was selected for the hard-difficulty ConvNeXt-ADNI task, with ResNet-18 serving as a smaller CNN baseline. The Tiny variant was chosen to accommodate the memory limitations of the GPU used for training. ResNet-18 was selected as the baseline model for two reasons, the first being it is a familiar model that was previously used in Demo 2 (COMP3710 Teaching Team, 2026). The second reason being that is provides a meaningful comparison between an established residual CNN and a modernised convolutional architecture. In *A ConvNet for 2020s (Liu et al., 2022)*, ConvNeXt was developed by progressively modernising ResNet-50 with design ideas inspired by vision transformers. Their shared use of convolution and residual connections gives the comparison between ConvNeXt and ResNet a clear architectural basis. This will be useful for assessing whether ConvNeXt-Tiny offers improvements in classification peformance and prediction confidence that justify its additional computational cost.

Both models are trained from random initalisation with two outputs to distinguish the AD and NC classes using the provided ADNI 2D MRI images. They will both use the same preproccessing steps of first dividing patients into approximately `70%` training, `20%` validation, and `10%` testing, keeping each patient's scan images in one split to prevent data leakage. Images are then padded to `256 x 256`, converted to three identical grayscale channels, and pixel values are normalised to `[0, 1]`. If during training overfitting occurs, data augmentation to the training set will also be implemented.

### Preliminary Feasibility Evidence
The prelimiary investigations conducted in the jupyter notebooks under `investigations/` support the feasibility of preparing the ADNI dataset and running the proposed models on it. The image investigation identified the dataset contained `30520` grayscale JPEG images, all measuring 256 x 240 pixels, and the preprocessing investigation linked these images to 1526 scans from 680 unique patients. This consistent image format provides a clear basis for standardising the model inputs. The original training and test sets shared 216 patients, revealing there was patient leakage across the sets.

To address this, the preprocessing notebook investigated implementing a stratified split by patient, assigning each patient to one split. This results in `21140` training, `6420` validation, and `2960` test images, approximately matching the intended 70/20/10 proportions. The images were also broadly well balanced between classes, with `15660` NC images, and `14860` AD images.

To assess computational feasibility, an investigation was conducted using the MNIST dataset, with images upscaled to `256 x 256` pixels. After testing various batch sizes on an Nvidia RTX 5070 Ti, ResNet-18 supported a maximum batch size of `512`, while ConvNeXt-Tiny supported a batch size of up to `128` before exceeding the GPU's available memory.

These finding demonstrate a workable data preparation strategy and provide preliminary evidence that the proposed model configuration is practical on the available hardware.

### Risks, Budget, Next Experiment, and Fallback
The main risks this project currently has are:
  - The long model training times.
  - The A100 GPU's available VRAM.

Long training runs may prevent the models from completing enough epochs to converge before the assignment deadline. Memory limits may also restrict the model size, if the models that fit have insufficient capacity to learn the patterns in the training set, they could underfit.

The nexxt experiment will test difference ConvNeXt and ResNet model sizes and progressively larger batch sizes on the A100. This will establish which combinations fit in memory. If the preferred configurations exceed the memory limit, the fallback is to use smaller ConvNeXt and ResNet models with fewer parameters and reduce the batch size untill training fits with the A100's VRAM.


## References
- 1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S., Facebook, A., & Research. (2022). A ConvNet for the 2020s. https://arxiv.org/pdf/2201.03545
- 2. COMP3710 Teaching Team. (2026, August 26). Lab Demonstration 2 Pattern Recognition [PDF]. https://learn.uq.edu.au/ultra/courses/_206498_1/document/_14338370_1?view=content&state=view