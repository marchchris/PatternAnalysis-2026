# ADNI Alzheimer's Disease Classification with ConvNeXt

## 1. Problem and Working Principles
This project implements a 2D convolutional network to classify ADNI brain MRI slices as Alzheimer's disease `AD` or cognitively normal `NC`. ConvNeXt is the main model, and ResNet-18 will provide a baseline. The engineering objective is to compare recognition performance, computational cost, and the reliability of unconfident predictions on ambiguous images that may need to be flagged for human review.

The data pipeline collects JPEG slices from the original dataset folders, groups them using the patient ID extracted from each filename, and assigns groups to approximately `70%` training, `20%` validation, and `10%` testing. Images are resized, converted to tensors, and standardised. Random augmentation is applied only to the training set to prevent overfitting. Each model learns spatial features through convolutional stages and residual connections, pools the final features, and outputs two classification logits. The epoch with the highest validation accuracy is saved for test evaluation. All reported scores describe individual slices, not aggregated patient diagnoses.

## 2. Model Architectures
### 2.1 ConvNeXt Implementation
ConvNeXt (Liu et al., 2022) improved on the ResNet architecture by modernising ResNet-50 using design ideas from Vision Transformers while mainting the simple convolutional structure of a traditional CNN.

Each block uses a `7 x 7` convolution to detect pattern in each feature channel separately. After normalisation, two linear layers mix information between channels, expanding their number by 4 and then reducing it back. GELU helps the model learn complex patterns. The result is scaled and added to the original input through a skip connection. During training, this added branch is sometimes dropped to help reduce overfitting.

| Component | Implementation | Output per image |
| --- | --- | --- |
| Input | Single channel image | `1 × 224 × 224` |
| Stem | `4 × 4` convolution, stride = 4 | `96 × 56 × 56` |
| Stage 1 | 3 blocks | `96 × 56 × 56` |
| Downsample 1 | `2 × 2` convolution = stride 2 | `192 × 28 × 28` |
| Stage 2 | 3 blocks | `192 × 28 × 28` |
| Downsample 2 | `2 × 2` convolution = stride 2 | `384 × 14 × 14` |
| Stage 3 | 9 blocks | `384 × 14 × 14` |
| Downsample 3 | `2 × 2` convolution = stride 2 | `768 × 7 × 7` |
| Stage 4 | 3 blocks | `768 × 7 × 7` |
| Pool and head | LayerNorm, linear `768 → 2` | 2 logits |

This design follows (Liu et al., 2022) configuration of `(3, 3, 9, 3)` blocks with `(96, 192, 384, 768)` channel dimensions giving approximately `27.8` million parameters. This project however changes the original design by reducing the stem down to 1 input channel and reducing the classification layer to 2 classes.

### 2.2 ResNet-18 Implementation
ResNet-18 will provide a suitable baseline comparison for the ConvNeXt network since it was the predecessor ConvNeXt was based off. Each `ResNetBlock` contains a `3 x 3` convolution, BatchNorm and ReLU followed by another `3 x 3` convolution and BatchNorm. The output is added to a skip connection and passed through ReLU. This allows the block to learn a residual correction while preserving a direct path for information and gradients.

| Component | Implementation | Output per image |
| --- | --- | --- |
| Input | Single channel image | `1 × 224 × 224` |
| Stem convolution | `7 × 7`, stride = 2 | `64 × 112 × 112` |
| Max pooling | `3 × 3`, stride = 2, padding = 1 | `64 × 56 × 56` |
| Stage 1 | 2 residual blocks, 64 channels | `64 × 56 × 56` |
| Stage 2 | 2 blocks | `128 × 28 × 28` |
| Stage 3 | 2 blocks | `256 × 14 × 14` |
| Stage 4 | 2 blocks | `512 × 7 × 7` |
| Pool and head | Adaptive average pool to `1 × 1`, linear `512 → 2` | 2 logits |

This design differs from the original (He et al., 2015) ResNet architecture by using a single input channel stem, adaptive global pooling and a 2 output classification layer. This gave a configuration of `(2, 2, 2, 2)` blocks with `(64, 128, 256, 512)` channel dimensions providing approximately `11.2` million parameters.

## 3. Dataset Preprocessing
### 3.1 Patient Splitting
The original provided ADNI dataset had patient leakage across the training and test sets. To fix this the original dataset was pooled together and resplit by patient ID provided by the images file names.

The filenames for example, `388206_78.jpeg` provided the patient ID `388206` and the slice index `78`. To avoid patient leakage, all slices were placed in the same split with their associated patient. To accomplish this, the `1526` unique patients were split into `70%` training, `20%` validation, and `10%` testing, then the images were assigned to the split of their corresponding patient ID.

This resulted the following splits below:

| Dataset Split | Total Images | Percentage | Subjects | NC Images | AD Images |
| ------------- | ------------ | ---------- | -------- | --------- | --------- |
| Train         | 21,360       | 69.99%     | 1,068    | 10,960    | 10,400    |
| Validation    | 6,100        | 19.99%     | 305      | 3,120     | 2,980     |
| Test          | 3,060        | 10.03%     | 153      | 1,580     | 1,480     |
| Total         | 30,520       | 100%       | 1,526    | 15,660    | 14,860    |

The dataset is approximately balanced with `NC` and `AD` nearly having equal class proportions.

### 3.2 Preprocessing Images
| Step                                                           | Reason                                                                                                              |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Resize to `224 x 224`                                          | All images in dataset are currently `256 x 240`. The standard ConvNeXt and Resnet designs used `224 x 224` images.  |
| Scale Pixel Values to `[0, 1]`                                 | Current pixel values are [0, 255], helps model training by avoiding large activations and gradients.                |
| Standardising using mean `0.1160`, standard deviation `0.2230` | Makes training set approximately mean `0` and standard deviation `1`.                                               |

All images are normalised using mean `0.1160` and standard deviation `0.2230` by: `x_normalised = (x - 0.1160) / 0.2230`. This mean and standard deviation was calculated using the training set and applied to training, validation, and test images. 

### 3.3 Dataset Augmentation
Both models suffered from overfitting to the training set quickly reaching `100%` training accuracy, to fix this a significant amount of data augmentation was needed to prevent the models from memorising the training images. The following random augmentation are applied to training images and reapplied each epoch. Validation and testing images used only the deterministic preprocessing described in section `3.2`.

| Augmentation | Parameters | Intended Effect |
| --- | --- | --- |
| Resize | `256 × 256` | Standardises the images before random cropping. |
| Random resized crop | Output `224 × 224`, area fraction `0.9–1.0`, aspect ratio `0.9–1.1` | Varies framing and scale, however can potentially remove anatomy |
| Horizontal flip | Probability `0.5` | Encourages robustness to left and right orientation changes. |
| Random affine | Rotation `±15°`, horizontal/vertical translation up to `10%`, scale `0.9–1.0` | Varies the position of the brain in the image, however can potentially remove anatomy. |
| Brightness/contrast jitter | Brightness `±0.1`, contrast `±0.15`, | Reduces dependence on intensity appearance. |
| Gaussian blur | Kernel `3 × 3`; sigma `0.1–1.0` | Varies sharpness of images, helps with potential differences in scan quality. |

Augmentation expands the appearances observed during learning without adding more patients. This helped the model generalise improving validation accuracy.

![Before and After of Training Images](images/augmentation.png)

## 4. Hyperparameters and Training Methodology


### 4.1 Configuration

| Parameter | ResNet-18 | ConvNeXt |
| --- | --- | --- |
| Initialization | Random weights | Random weights |
| Epochs | `200` | `200` |
| Batch size | `64` | `64` |
| Optimizer | AdamW | AdamW |
| Initial learning rate | `1e-3` | `1e-3` |
| Weight decay | `1e-4` | `0.05` |
| Drop Path | N/A | `0.1` |
| Loss | `CrossEntropyLoss(label_smoothing=0.05)` | `CrossEntropyLoss(label_smoothing=0.05)` |
| Learning rate schedule | `5` epoch linear warmup, then cosine decay toward `1e-6` | `5` epoch linear warmup, then cosine decay toward `1e-6` |
| Checkpoint selection | Highest validation accuracy | Highest validation accuracy |
| Random seed | `42` | `42` |
| DataLoader workers | `4` | `4` |

Validation is run at the end of every epoch where the highest validation accuracy is the model saved. This was to ensure the highest accuracy model was saved even if the models performancce degraded afterwards due to overfitting. `200` epochs were used to allow training time for the model to properly converge. A small batch size of `64` was chosen to lower memory usage and use stochastic noise during gradient calculations to potentially help the model escape poor local minima.

### 4.2 Regularisation
- ConvNext uses `0.1` drop path for stochastic depth regularisation.
- AdamW weight decay shrinks both models weights towards zero.
- Cross Entropy Loss Label smoothing to reduce over confidence.

### 4.3 Learning Rate Schedule
A combination of a linear warmup and cosine annealing was used to change learning rate during training. For the first `5` epochs the learning rate linearly increases up to the inital learning rate of `1e-3`. Then over the next `195` epochs, cosine annealing is used to progressively reduce the learning rate to `1e-6`. This helps the model achieve smooth convergence in late epoch learning.


![Plot of Learning rate vs epoch](images/learningrateplot.png)

## 5. ResNet-18 Model Results

For these results, both models were trained using a Nvidia RTX 5070 Ti. The test metrics were evaluated using the `10%` testing split.

### 5.1 Training History
The model was trained for `200` epochs which took approximately `37` minutes and `11.24` seconds per epoch. The training loss and accuracy history plots are shown below.

![ResNet-18 Training History](images/resnet_training_history.png)

The model saved was from epoch `167` which achieved the highest validation accuracy of `91.51%` indiciating the model is capable of generalising to unseen data. ResNet-18 demonstrated strong early learning performance quickly increasing to over `90%` training accuracy in less than `25` epochs. training accuracy. Validation accuracy improves significantly, reaching approximately `92%` towards the end of training, indicating strong generalisation to unseen data. During the early epochs, validation accuracy displays considerable fluctuations, with validation loss occasionally exceeding `1.0`, before becoming more stable as training continues. Despite this, training accuracy continues to increase towards `100%`, indicating overfitting without achieving further improvements on the validation set.


### 5.2 Test Results
The final model was evaluated on the hold out test set. The confusion matrix and test metrics below shows the model's performance on this unseen data.


![ResNet-18 Confusion Matrix](images/resnet_confusion_matrix.png)

| Metric | ResNet-18 |
| --- | --- |
| Accuracy | **91.24%** 
| F1 | **91.25%** |
| Precision (AD positive) | **88.26%** |
| Recall (AD positive) | **94.46%** |

The model achieved strong performance on the test set, with an accuracy of `91.24%`, and F1-score of `91.25%`. Importantly, the model achieved an AD recall of `94.46%` indicating that it successfully identified the majority of Alzheimer's cases. The model also achieved an AD precision of `88.26%` meaning majority of images predicted as Alzheimer's were correctly classified.

## 6. ConvNeXt Model Results
### 6.1 Training History
The model was trained for `200` epochs which took approximately `88` minutes and `26.46` seconds per epoch. The training loss and accuracy history plots are shown below.

![ConvNeXt Training History](images/convnext_training_history.png)

The model saved was from epoch `172` which achieved the highest validation accuracy of `92.16%`. Compared to ResNet-18, ConvNeXt demonstrates smoother validation performance following closely to training accuracy during earlier epochs. However, validation accuracy begins to plateu around epoch `120` where training loss continues to decrease and training accuracy approaches `100%`, creating a widening generalisation gap between training and validation performance. This again indicates the model overfitting later in training.

### 6.2 Test Results
![ConvNeXt Confusion Matrix](images/convnext_confusion_matrix.png)

| Metric | ConvNeXt |
| --- | --- |
| Accuracy | **91.63%** 
| F1-score | **91.61%** |
| Precision (AD positive) | **88.93%** |
| Recall (AD positive) | **94.46%** |

ConvNeXt achieved stronger performance on the test set compared to ResNet-18, with an accuracy of `91.63%` and F1-score of `91.61%`. The model achieved a higher AD precision of `88.93%` but had an equal AD recall of `94.46%` compared to ResNet-18.

ConvNeXt produced `12` fewer false positives to ResNet-18, demonstrating slightly better specificity while maintaining the same sensitivity.

## 7. Analysis of Results
### 7.1 Why Sensitivity Matters
High sensitivity is particularly important for this Alzheimer's disease classification task because the primary objective is to identify as many individuals with the disease as possible while minimising false negatives. A false negative means potentially delaying treatment, and early intervention. In contrast, a false positive may cause uncessary anxiety and additional diagnostic testing, this error can be investigated and corrected through further clinical assessments.

Therefore, the consequence of missing a true Alzheimer's case is more significant than incorrectly flagging a healthy individual for further investigation.

### 7.2 Comparison of Model Results
Comparing the performance of both models, ConvNeXt outperformed ResNet-18 across all evaluated test metrics, achieving higher acurracy, F1-score, precision and recall.

![Model ROC Curves](images/roc_curves.png)

The ROC curves demonstrate that both ResNet-18 and ConvNeXt achieve strong performance in distinguishing between Alzheimer's disease and normal brain images. However, ConvNeXt achieves a higher ROC-AUC of `96.05%` compared to ResNet-18's `94.53%`, indicating better overall performance across different classification thresholds. ConvNeXt generally achieves a higher sensitivity at lower false positive rates, suggesting that it is more effective at identifying Alzeheimer's cases while minimising the number of healthy individuals incorrectly classified as having the disease. This makes it a more promising model for preliminary Alzheimer's screening.

### 7.3 Comparison of Model Computational Efficiency
To measure the difference in inference time and GPU VRAM usage, both models ran inference on the same image from the test set, and only started measuring after the dataset loading and preprocessing was complete.


| Model     | Inference Time | Peak VRAM Usage |
| --------- | -------------- | --------------- |
| ResNet-18 | 0.1659s        | 74.86 MB        |
| ConvNeXt  | 0.2891s        | 154.38 MB       |

From the table it can be seen that both models ran inference on the same test image in less than half a second. ConvNeXt used over twice the amount of VRAM compared to ResNet-18, however this is still easily within the available VRAM of most modern devices. 

This shows that both models can be used in a clinical setting on a modern device, to receive results quickly. ResNet-18 however, is able to run inference faster and with less VRAM.

### 7.4 Final Result

Due to ConvNext performing better on the hold out test set, **ConvNeXt is the better performing model for this task**.

## 8. Usage Instructions
### 8.1 Environment Setup
First clone repository and install dependencies:
```bash
git clone https://github.com/marchchris/PatternAnalysis-2026.git
cd PatternAnalysis-2026/recognition/ConvNeXt_48900353
pip install -r requirements.txt
```

### 8.2 Training
Run training using with:

```bash
python train.py --dataset-root [DATASET ROOT] --model-name resnet18 --seed [SEED] --epochs [EPOCHS] --batch-size [BATCH SIZE] --learning-rate [LEARNING RATE] --weight-decay [WEIGHT DECAY] --warmup-epochs [NUM WARMUP EPOCHS] --num-workers [NUMB LOADER WORKERS]
```

| Optional argument | Description | Default |
| --- | --- | --- |
| `--dataset-root` | Path to the ADNI dataset root containing `train/` and `test/` | `~/Documents/Datasets/ADNI/AD_NC` |
| `--model-name` | Model architecture to train | `resnet18` or `convnext` |
| `--seed` | Random seed | `42` |
| `--epochs` | Number of training epochs | `200` |
| `--batch-size` | Number of images per batch | `64` |
| `--learning-rate` | Initial learning rate | `0.001` |
| `--weight-decay` | AdamW weight decay | `0.0001` |
| `--warmup-epochs` | Number of linear warmup epochs | `5` |
| `--num-workers` | Number of DataLoader worker processes | `4` |

Each run creates its own `Models/<model-name>/<model-name>_<timestamp>/` directory and stores `<model-name>.pth`, `training_history.png`, `confusion_matrix.png`, and `test_report.txt.` The saved weights correspond to the highest validation accuracy, which may occur before epoch 200.

An example of `test_report.txt` is:
```
Model Training Report:
Training time: 88m 12s

Average time per epoch: 26.46s

Test Set Evaluation Report:
Test loss: 0.3601
Accuracy: 0.9163
Precision (AD positive): 0.8893
Recall (AD positive): 0.9446
F1 Score (AD positive): 0.9161
ROC AUC: 0.9596
Support: 3060
Negative support: 1580
Positive support: 1480
```

### 8.3 Running Predictions

Run predictions using a saved model checkpoint with:

```bash
python predict.py --model convnext --weights Models/convnext/convnext_[TIMESTAMP]/convnext.pth
```

By default, the script selects `9` images from the test split, prints the
prediction details for each image, and saves a prediction visualisation to
`Inferences/predictions_[TIMESTAMP].png`. A different number of test images can
be selected with:

```bash
python predict.py --model resnet18 --weights Models/resnet18/resnet18_[TIMESTAMP]/resnet18.pth --num-images 12
```

An individual image can be supplied instead of sampling from the test split:

```bash
python predict.py --model convnext --weights Models/convnext/convnext_[TIMESTAMP]/convnext.pth --image path/to/image.jpeg
```

| Optional argument | Description | Default |
| --- | --- | --- |
| `--image` | Path to one image to predict instead of sampling test images | Test split |
| `--num-images` | Number of test images to sample when `--image` is not supplied | `9` |
| `--dataset-root` | Path to the ADNI dataset root | `~/Documents/Datasets/ADNI/AD_NC` |
| `--output` | Output path for the prediction visualisation | `Inferences/predictions.png` |

The required arguments are:

| Required argument | Description |
| --- | --- |
| `--model` | Model architecture used by the checkpoint: `resnet18` or `convnext` |
| `--weights` | Path to the saved `.pth` model checkpoint |

For every prediction, the script prints the predicted class, confidence,
class probabilities, prediction time, and peak VRAM usage. Prediction timing
and VRAM usage are measured during model evaluation after the image has been
loaded and preprocessed.

An example of the prediction visualisation saved from the trained ConvNeXt model:

![ConvNeXt Predictions](images/convnext_predictions.png)

## 9. Project Structure
```
PatternAnalysis-2026/recognition/
└── ConvNeXt_48900353/
    ├── dataset.py
    ├── modules.py
    ├── train.py
    ├── predict.py
    ├── requirements.txt
    ├── images/
    │   ├── 
    │   └── 
    └── README.md
```

## References
- 1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S., Facebook, A., & Research. (2022). A ConvNet for the 2020s. https://arxiv.org/pdf/2201.03545
- 2. He, K., Zhang, X., Ren, S., & Sun, J. (2015, December 10). Deep residual learning for image recognition. arXiv. arXiv.Org. https://arxiv.org/abs/1512.03385


## AI Usage Statement
- Chat GPT 6 was used for assisting with markdown syntax, table creation, spell check, and refinement of wording in the README, all actual content and information was written by myself.
- Github Copilot (Chat GPT 5.6) was used to assist in writing code documentation and comments, and minor code refactoring. All data loading, preprocessing, training and analysis code was written by myself.
