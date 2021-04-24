# Exercise multimodal encoder-decoder: RGB-D scene classification

## 1. RGB, HHA and Baseline RGBD Models

The following table shows train, validation and test average accuracies (and standard deviations) over 5 runs for each case (RGB, HHA and provided RGBD model). All the three models are trained using [OneCyle Training Policy](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR)
Hyperparams
- Max LR: 1e-3 for unimodal (1e-4 for RGBD)
- Scheduler: OneCycle Training
- Max Epochs: 10
- Batch Size: 16

| modality | set   | run 1 | run 2 | run 3 | run 4 | run 5 | avg   | std    |
|----------|-------|-------|-------|-------|-------|-------|-------|--------|
| rgb      | train | 86.98 | 86.71 | 86.98 | 87.38 | 87.92 | 87.19 | 0.4213 |
| rgb      | val   | 80.53 | 78.69 | 79.03 | 78.11 | 78.92 | 78.85 | 0.8714 |
| rgb      | test  | 73.85 | 73.66 | 74.23 | 72.65 | 74.54 | 73.39 | 1.0569 |
| hha      | train | 70.07 | 70.47 | 68.46 | 69.66 | 69.40 | 69.61 | 0.6812 |
| hha      | val   | 60.37 | 62.67 | 60.25 | 60.14 | 62.44 | 61.17 | 1.1323 |
| hha      | test  | 57.66 | 59.04 | 56.27 | 56.14 | 58.48 | 57.52 | 1.1592 |
| rgbd     | train | 92.75 | 93.96 | 94.23 | 94.09 | 93.15 | 93.64 | 0.5805 |
| rgbd     | val   | 81.68 | 82.37 | 77.88 | 80.53 | 81.22 | 80.74 | 1.5488 |
| rgbd     | test  | 74.61 | 77.76 | 76.12 | 77.38 | 75.74 | 76.32 | 1.1400 |

## 2. Improvements to Multimodal Architecture

### Fine-tuning more layers

We start by fine-tuning more layers in the baseline architecture and try two strategies

1. Same LR of 1e-4 for all trainable layers.

| Experiment | set   | run 1 | run 2 | run 3 | run 4 | run 5 | avg   | std    |
|------------|-------|-------|-------|-------|-------|-------|-------|--------|
| Same LR    | train | 94.78 | 94.50 | 95.57 | 93.91 | 94.77 | 94.71 | 0.5352 |
| Same LR    | val   | 81.34 | 80.30 | 77.07 | 82.03 | 81.34 | 80.42 | 1.7619 |
| Same LR    | test  | 77.50 | 76.37 | 75.24 | 77.06 | 76.24 | 76.48 | 0.7725 |

2. Discriminative LR of 1e-4 for classifier layer and 3e-5 for feature layers

|   Experiment      | set   | run 1 | run 2 | run 3 | run 4 | run 5 | avg   | std    |
|-------------------|-------|-------|-------|-------|-------|-------|-------|--------|
| Discriminative LR | train | 90.47 | 92.89 | 91.54 | 91.01 | 91.41 | 91.46 | 0.8045 |
| Discriminative LR | val   | 81.80 | 80.65 | 81.11 | 79.49 | 81.11 | 80.83 | 0.7649 |
| Discriminative LR | test  | 75.43 | 77.25 | 73.16 | 76.18 | 77.13 | 75.83 | 1.4912 |

### Changing Embeddings and Architecture

AlexNet is a 60M parameter model. Even though we train only a few hundred thousand parameters for our task, a forward pass is still very computationall expensive. Currently, there are much lighter and better pre-trained models available for feature extraction for eg. ResNet18(11M) and MobileNetV3(2.5M)


Also, the models we have looked at have been example of late fusion where features from RBG and HHA domain are fused at the very end stages of the network. We will also experiment with early fusion architectures. Even in early fusion, there are 2 categories: concatenation and addition. For computational reasons, we will not be running 5 runs for each experiment.

|Experiment - Late Fusion         | train | val   | test  |
|---------------------------------|-------|-------|-------|
| MobileNetV3 Concatenation       | 89.13 | 81.91 | 75.24 |
| MobileNetV3 Addition (Mean)     | 85.91 | 80.07 | 74.92 |


For early fusion, we concatenate the feature maps obtained from backbones (Cx7x7 maps) and process them through some convolutional layers.

|Experiment - Early Fusion        | train | val   | test  |
|---------------------------------|-------|-------|-------|
| MobileNetV3 Concatenation       | 93.15 | 82.14 | 77.25 |
| MobileNetV3 Addition (Mean)     | 90.20 | 83.18 | 76.62 |

### Experimental results and discussion

- Depth helps in classification as we saw a 3 percent increase on the test set after incorporating depth information along with RGB
- Fine-tuning AlexNet baseline model did not result in a significantly better performance
- Early fusion gave better results compared to Late fusion. Early fusion allows the network to learn better features.
- Changing the backbone and trying different fusion startegies did not result in a significant improvement over the baseline. One of the reasons behind this could be the simplicity of the fusion strategies that we have used. Better fusion strategies exist and should be explored in future work.

### Future work

- We would like to fuse features between RGB and Depth Encoder at multiple levels
- We also want to explore attention and transformer based networks to incorporate depth information in a better way than proposed by current architectures.

## 3. Team Members

- Aditya Rana
- Oscar Lorente
- Ian Riera
