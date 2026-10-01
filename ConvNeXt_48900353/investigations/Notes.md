### 1/10/2026 5:55pm
During attempt of training resnet18 model, after 5 epochs the model reached 100% training accuracy and began to plateu at a validation accuracy of approximately 72%.

This indicates overfitting to the training set. To address this I will introduce data augmentation to the training set.
```
--- Beginning Training ---
Epoch 01/100 | Train loss: 0.5095, accuracy: 73.88% | Val loss: 4.3196, accuracy: 46.42%
Epoch 02/100 | Train loss: 0.2169, accuracy: 91.14% | Val loss: 1.1156, accuracy: 67.12%
Epoch 03/100 | Train loss: 0.0489, accuracy: 98.67% | Val loss: 1.1133, accuracy: 71.84%
Epoch 04/100 | Train loss: 0.0092, accuracy: 99.94% | Val loss: 1.0112, accuracy: 70.87%
Epoch 05/100 | Train loss: 0.0030, accuracy: 100.00% | Val loss: 1.0526, accuracy: 71.68%
Epoch 06/100 | Train loss: 0.0016, accuracy: 100.00% | Val loss: 1.1029, accuracy: 72.43%
Epoch 07/100 | Train loss: 0.0012, accuracy: 100.00% | Val loss: 1.0628, accuracy: 72.48%
```