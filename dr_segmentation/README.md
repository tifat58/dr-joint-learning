To train the model for FGADR, run ```python train_fgadr.py --seed 765 --preprocess '2' --lesion 'EX'``` for training a UNet model to segment Hard Exudates lesion images with preprocessing method of Contrast Enhancement using random seed 765.

- ```lesion: MA, HE, EX, SE, SG```, SG for combined segmentation masks

The meaning of each preprocessing index is indicated in the following table.

| Preprocessing Index | Preprocessing Methods |
| :---: | :---: |
| '0' | None |
| '1' | Brightness Balance |
| '2' | Contrast Enhancement |
| '3' | Contrast Enhancement + Brightness Balance |
| '4' | Denoising |
| '5' | Denoising + Brightness Balance |
| '6' | Denoising + Contrast Enhancement |
| '7' | Denoising + Contrast Enhancement + Brightness Balance |

To evaluate the model on the test set, run ```python evaluate_model.py --seed 765 --preprocess '2' --lesion 'EX' --model results/models_ex/model.pth.tar``` for evaluating a saved UNet model checkpoint on MA under ```results/``` with preprocessing method of Contrast Enhancement using random seed 765. `results/models_ex/model.pth.tar` is the directory of the saved model checkpoint.

- traing scripts for other segmenation models and IDRID dataset will be updated soon' 
