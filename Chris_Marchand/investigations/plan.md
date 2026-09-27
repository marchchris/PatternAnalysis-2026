# Investigation Results

## Data Investigation Results
- There are `30520` images in the entire dataset.
- All images in the dataset are `256 x 240` pixels and grayscale.
- Pixels values range from `0 to 255`.
- There are `2189` records in the `meta_data_with_label.json` file.
- There are `942` unique patients recorded in the meta data file.

After linking each image to the patient they were scanned from, it was discovered that is patient leakage between the train and test splits.

- There were `216` unique patients that were identified to be present in both the train and test splits.
- This resulted in `14380` images that belong to patients which were present in both splits.

Because nearly half of the dataset is affected by patient leakage, I will combine the existing train and test splits while preserving the separation between the `AC` and `NC` classes. I will then divide the patients into 70% training, 20% validation, and 10% test splits. Finally, I will move each image into the split assigned to its corresponding patient.

## Preprocessing Investigation Results

- Using `train_test_split` the patients were divided into 70% train, 20% validation and 10% test.
- After moving all the images into the split assigned to its corresponding patient the resulting dataset sizes were:

| Split | Images | Percentage of Dataset |
| --- | ---: | ---: |
| Training | 21140 | 69.27% |
| Validation | 6420 | 21.04% |
| Test | 2960 | 9.70% |
| **Total** | **30520** | **100%** |

Since some patients have a greater number of images to others, a perfect 70%, 20%, 10% was not possible. Despite this, simply splitting by patients rather than the images themselves, managed to achieve percentages that are reasonably close to the desired proportions.

The class distributions in each split were:

| Split | NC | AD | Total |
| --- | ---: | ---: | ---: |
| Training | 10660 | 10480 | 21140 |
| Validation | 3440 | 2980 | 6420 |
| Test | 1560 | 1400 | 2960 |

As shown, since each split was stratisfied, the class distributions in each split were nearly equal.

## ConvNeXt Model Investigation Results

- The model of ConvNeXt I chose to use was the tiny model from: https://docs.pytorch.org/vision/0.28/models/convnext.html, this is due to the memory limitations of my GPU.
- I stress tested the model using the MNIST dataset on my RTX 5070 TI. The maximum settings I was able to achieve without running out of memory was:
    - `128` Batch Size
    - `256 x 256` pixel image size.

According to this: https://www.geeksforgeeks.org/computer-vision/convnext/, the ConvNeXt archeticture downsamples the input image by a total factor of `32`. Therefore having an input image which is a multiple of `32` will produce whole numbers at each stage.

# Plans For The Project

## Preprocessing

1. The train and test splits will be combined but still seperated by their classes.
2. The patients will be split into `70%` training, `20%` validation, and `10%` test.
3. The images will be placed into the splits their associated patients are in.

Since the images are already `256x240` pixels and their edges are entirely black pixels, I will pad the images edges with 16 black pixels to upscale them to `256x256`.

4. Pad the images with black pixels to upscale them to `256 x 256` dimensions.
5. Convert the single channel grayscale image to 3 RGB channels.
6. Normalise pixel values from `0 - 255` to `0 - 1`.

Future Note: Will need to see if data augmentation is needed to prevent overfitting.

### Model Training

1. As a baseline, a small Resnet18 model will be used.
    - https://docs.pytorch.org/vision/main/models/resnet.html
2. After this baseline model has been trained and optimised. The tiny ConvNeXt model will be trained.
3. The ConvNeXt model will be fine tuned to achieve maximum performance, potentially using a larger variation if computation capabilities will allow it.
4. Potentially experiment with data augmentation if overfitting to the train set occurs.