# ADNI MRI Classifcation

## 1. Feasibility Review

### User Need and Scope
The intended user this project is aimed towards is a researcher evaluating a machine learning model that could assist human reviewers in diagnosing Alzheimer's disease from MRI images. The project scope is limited to training and evaluating the models on the course-provided 2D ADNI MRI images, it will not aggregate multiple brain slices from the same patient to produce a prediction.

### Acceptance Criteria
1. Every included ADNI image has a valid mapping to a patient, and the training, validation, and testing set has no patient leakage.
2. ConvNeXt achieves a test accuracy of atleast `80%`.
3. The ConvNeXt model achieves a higher test accuracy than the baseline ResNet model.
4. Training fits within the provided Nvidia A100 GPUs available memory.

### Model Choice and Course Concepts
The selected models are ConvNeXt, which serves as the advanced model, and ResNet-18, which serves as the smaller baseline. Both models will be initialised with random weights and trained from scratch. The ConvNeXt design will closely follow the architecture presented in *A ConvNet for the 2020s (Liu et al., 2022) [1]*, using stage depths of `(3, 3, 9, 3)` and channel dimensions of `(96, 192, 384, 768)`. The ResNet-18 design will similarly follow the architecture described in *Deep Residual Learning for Image Recognition (He et al., 2015)*, using `(2, 2, 2, 2)` residual blocks and channel dimensions of `(64, 128, 256, 512)`. ResNet-18 was chosen as the baseline because ConvNeXt was developed by progressively modernising the ResNet architecture, making ResNet a suitable reference point for evaluating the benefits of ConvNeXt's design changes.

There will be 2 main changes from the original architectures. Both model's input layers will be modified to accept a single channel instead of the original three RGB channels, and their classification layers be will reduced to only two outputs for AD and NC classification.

### Preliminary Feasibility Evidence
Preliminary data audits were conducted in jupyter notebooks to assess the feasibility of preparing the ADNI dataset and running the proposed models on it. It was identified that the dataset contained `30520` grayscale JPEG images, all measuring `256 x 240` pixels, and these images were linked to `1526` scans from `680` unique patients. This consistent image format already provided an effective dataset prior to any preprocessing. However,the original training and test sets shared `216` patients across them, revealing there was patient leakage across the sets.

To address this implementing stratified split by patient was investigated, assigning each patient to exactly one split. This resulted in `21140` training, `6420` validation, and `2960` test images, approximately matching the intended 70/20/10 proportions. The images were also broadly well balanced between classes, with `15660` NC images, and `14860` AD images.

### NOTE: COMPLETE COMPUTATION FEASIBILITY HERE

These finding demonstrate a workable data preparation strategy and provide preliminary evidence that the proposed model configuration is practical on the available hardware.

### Risks, Budget, Next Experiment, and Fallback
The main risks this project currently has are:
  - The long model training times.
  - The A100 GPU's available VRAM.

Long training runs may prevent the models from completing enough epochs to converge before the assignment deadline. Memory limits may also restrict the model size, if the models that fit have insufficient capacity to learn the patterns in the training set, they could underfit.

The next experiment will test difference ConvNeXt and ResNet model sizes and progressively larger batch sizes on the A100. This will establish which combinations fit in memory. If the preferred configurations exceed the memory limit, the fallback is to use smaller ConvNeXt and ResNet models with fewer parameters and reduce the batch size untill training fits with the A100's VRAM.

## 2. Problem and Approach
This project investigates binary classifcation of two dimensional ADNI brain MRI images into Alzheimer's disease (AD) and congnitively normal (NC) classes. ConvNeXt will be the advanced model, and ResNet-18 will be a standard baseline. The engineering question is whether ConvNeXt improves prediction quality and the handling of uncertain cases enough to justify its increased computational cost. A successful model will achieve a minimum test accuracy of `0.80` for the ConvNeXt task, paired with a significant enough increase in performance from the baseline model.


## References
- 1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S., Facebook, A., & Research. (2022). A ConvNet for the 2020s. https://arxiv.org/pdf/2201.03545
- 2. He, K., Zhang, X., Ren, S., & Sun, J. (2015, December 10). Deep residual learning for image recognition. arXiv. arXiv.Org. https://arxiv.org/abs/1512.03385
- 3. COMP3710 Teaching Team. (2026, August 26). Lab Demonstration 2 Pattern Recognition [PDF]. https://learn.uq.edu.au/ultra/courses/_206498_1/document/_14338370_1?view=content&state=view
- 5. Pytorch. (2024). ConvNeXt — Torchvision 0.28 documentation. Pytorch.Org. https://docs.pytorch.org/vision/0.28/models/convnext.html
- 6. ADNI. (n.d.). The anatomy of an ADNI table. ADNI Documentation. Retrieved https://adni.loni.usc.edu/quick-start-guide-asset/anatomy2.html

### Google AI Overview (Sep 26 9:56pm)

Prompt: "in python how to recruively get paths of all sub dir of a path"

```
Method 1: Using pathlib (Recommended for Python 3.4+)The pathlib module provides an object-oriented approach. You can use .rglob("*") to search recursively and filter for directories using .is_dir().pythonfrom pathlib import Path

# Define your starting directory
base_path = Path("/path/to/your/directory")

# Get paths of all subdirectories recursively
subdirectories = [str(p) for p in base_path.rglob("*") if p.is_dir()]

print(subdirectories)
```


### ChatGPT 5.6 Sol (Sep 26 10:51pm)

Prompt: "Give python regex for matching the patient id from this string."

"ADNI_T1_3T/ADNI_016_S_6381_MR_Accelerated_Sagittal_MPRAGE__br_raw_20180522132015417_85_S688087_I1000691.nii"

The patient id is 016_S_6381"

```
You can use this regex:

r"ADNI_(\d{3}_S_\d{4})"
```

### ChatGPT 5.6 Sol (Sep 29 3:00pm)

Prompt: "write function documentation for this function"

```
"""Build a classification model with a custom output layer.

Args:
    model_name: Model architecture to create: ``"resnet18"`` or
        ``"convnext"``.
    num_classes: Number of output classes.

Returns:
    A randomly initialized model configured for ``num_classes`` outputs.

Raises:
    ValueError: If ``model_name`` is not supported.
    """
```