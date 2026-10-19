## 1. Feasibility Review

### User Need and Scope
The intended user this project is aimed towards is a researcher evaluating a machine learning model that could assist human reviewers in diagnosing Alzheimer's disease from MRI images. The project scope is limited to training and evaluating the models on the course-provided 2D ADNI MRI images, it will not aggregate multiple brain slices from the same patient to produce a prediction.

### Acceptance Criteria
1. Every included ADNI image has a valid mapping to a patient, and the training, validation, and testing set has no patient leakage.
2. ConvNeXt achieves a test accuracy of atleast `80%`.
3. The ConvNeXt model achieves a higher test accuracy than the baseline ResNet-18 model.
4. Training fits within the provided Nvidia A100 GPUs available VRAM.
5. The model must automatically accept atleast 80% of test images with a prediction confidence of 90% or higher, referring the remaining images for human review.

### Model Choice and Course Concepts
The selected models are ConvNeXt, which serves as the advanced model, and ResNet-18, which serves as the smaller baseline. Both models will be initialised with random weights and trained from scratch. The ConvNeXt design will closely follow the architecture presented in *A ConvNet for the 2020s (Liu et al., 2022) [1]*, using stage depths of `(3, 3, 9, 3)` and channel dimensions of `(96, 192, 384, 768)`. The ResNet-18 design will similarly follow the architecture described in *Deep Residual Learning for Image Recognition (He et al., 2015)*, using `(2, 2, 2, 2)` residual blocks and channel dimensions of `(64, 128, 256, 512)`. ResNet-18 was chosen as the baseline because ConvNeXt was developed by progressively modernising the ResNet architecture, making ResNet a suitable reference point for evaluating the benefits of ConvNeXt's design changes.

There will be 2 main changes from the original architectures. Both model's input layers will be modified to accept a single channel instead of the original three RGB channels, and their classification layers be will reduced to only two outputs for AD and NC classification.

### Preliminary Feasibility Evidence
Preliminary data audits were conducted in jupyter notebooks to assess the feasibility of preparing the ADNI dataset and running the proposed models on it. It was identified that the dataset contained `30520` grayscale JPEG images, all measuring `256 x 240` pixels, and these images were linked to `1526` scans from `680` unique patients. This consistent image format already provided an effective dataset prior to any preprocessing. However,the original training and test sets shared `216` patients across them, revealing there was patient leakage across the sets.

To address this implementing stratified split by patient was investigated, assigning each patient to exactly one split. This resulted in `21140` training, `6420` validation, and `2960` test images, approximately matching the intended 70/20/10 proportions. The images were also broadly well balanced between classes, with `15660` NC images, and `14860` AD images.

Preliminary runs on an NVIDIA RTX 5070 Ti successfully began training ResNet-18 with a maximum tested batch size of `512` and ConvNeXt with a maximum tested batch size of `128`. Both configurations fit within the GPU's 16 GB of VRAM, indicating that they will fit within the larger VRAM capacity of the available NVIDIA A100 GPUs, especially given that smaller batch sizes will be used. These findings provide preliminary evidence that the proposed model configuration is computationally feasible on the available hardware.

### Risks, Budget, Next Experiment, and Fallback
The main risks this project currently has are:
  - The long model training times.
  - The A100 GPU's available VRAM.

Long training runs may prevent the models from completing enough epochs to converge before the assignment deadline. Memory limits may also restrict the model size, if the models that fit have insufficient capacity to learn the patterns in the training set, they could underfit.

The next experiment will test difference ConvNeXt and ResNet model adjusting the number of stages and channel dimensions. This will establish which combinations fit in memory. If the preferred configurations exceed the memory limit, the fallback is to use smaller ConvNeXt and ResNet models with fewer parameters and reduce the batch size untill training fits with the A100's VRAM.
