# PsyNet: Self-supervised Approach to Object Localization Using Point Symmetric Transformation
Official Pytorch implementation of "PsyNet: Self-supervised Approach to Object Localization Using Point Symmetric Transformation" 

This implementation is based on these repos.
* [Pytorch Official ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)

## NOTE
![structure](./src/structure.png)

## Abstract
Existing co-localization techniques significantly lose performance over weakly or fully supervised methods in accuracy and inference time. In this paper, we overcome common drawbacks of co-localization techniques by utilizing self-supervised learning approach. The major technical contributions of the proposed method are two-fold. 1) We devise a new geometric transformation, namely point symmetric transformation and utilize its parameters as an artificial label for self-supervised learning. This new transformation can also play the role of region-drop based regularization. 2) We suggest a heat map extraction method for computing the heat map from the network trained by self-supervision, namely class-agnostic activation mapping. It is done by computing the spatial attention map. Based on extensive evaluations, we observe that the proposed method records new state-of-the-art performance in three fine-grained datasets for unsupervised object localization. Moreover, we show that the idea of the proposed method can be adopted in a modified manner to solve the weakly supervised object localization task. As a result, we outperform the current state-of-the-art technique in weakly supervised object localization by a significant gap.

## Todo
- [ ] Code Refactoring

## Requirement
  * python 3.6
  * pytorch 1.0.0 or 1.1.0
  * torchvision 0.2.2 or 0.3.0
  * tqdm
  * scipy
  * PIL
  * opencv-python (cv2)
  
## Data Preparation
-
   
## How to Run
### Arguments
-
   
### Train
-

### Test
-
   
## Results
-
## References
-
