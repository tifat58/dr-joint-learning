# dr-joint-learning
**Official Repository for DRG-Net**

Tusfiqur, Hasan Md, Duy MH Nguyen, Mai TN Truong, Triet A. Nguyen, Binh T. Nguyen, Michael Barz, Hans-Juergen Profitlich et al. "DRG-Net: interactive joint learning of multi-lesion segmentation and classification for diabetic retinopathy grading." arXiv preprint arXiv:2212.14615 (2022).
https://arxiv.org/abs/2212.14615

## Installation

Recommended environment:
- python 3.8+
- pytorch 1.7.1+
- torchvision 0.8.2+
- tqdm
- munch
- packaging
- tensorboard
- scikit-learn
- opencv-python
- pillow < 7
- segmentation-models-pytorch

To install the dependencies create a virtualenv and run:
```shell
$ pip install -r requirements.txt
```



**For classification**

- dr_classification folder contains scripts for dr grading classification task.
- Follow readme.md inside 'dr_classification' for instructions.


**For segmentation**

- Follow readme.md inside 'dr_segmentaion' instructions
