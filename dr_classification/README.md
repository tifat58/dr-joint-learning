# Pytorch Classification

- A general, feasible and extensible framework for 2D image classification.



## Features

- Easy to configure (model, hyperparameters)
- Training progress monitoring and visualization
- Weighted sampling / weighted loss / kappa loss / focal loss for imbalance dataset
- Kappa metric for evaluating model on imbalance dataset
- Different learning rate schedulers and warmup support
- Data augmentation
- Multiple GPUs support



## Installation

Recommended environment:
- python 3.8+
- pytorch 1.7.1+
- torchvision 0.8.2+
- tqdm
- munch
- packaging
- tensorboard

To install the dependencies, run:
```shell
$ pip install -r requirements.txt
```



## How to use

**1. Use the following method to build your dataset on your server:**

- Dict-form dataset:

Define a dict as follows:

```python
your_data_dict = {
    'train': [
        ('path/to/image1', 0), # use int. to represent the class of images (start from 0)
        ('path/to/image2', 0),
        ('path/to/image3', 1),
        ('path/to/image4', 2),
        ...
    ],
    'test': [
        ('path/to/image5', 0),
        ...
    ],
    'val': [
        ('path/to/image6', 0),
        ...
    ]
}
```

Then use pickle to save it:

```python
import pickle
pickle.dump(your_data_dict, open('path/to/pickle/file', 'wb'))
```

Finally, replace the value of 'data_index' in BASIC_CONFIG in `configs/default.yaml` with 'path/to/pickle/file' and set 'data_path' as null.

**For FGADR dataset:**

- update 'fgadr_generate_pkl.py' file by changing variables paths in line 7-9:
    - image_dir = # fgadr image folder e.g. '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Original_Images'
    - groundtruth_file = # csv file path containing the groudturths e.g. '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/DR_Seg_Grading_Label.csv'
    - path_to_save_pkl_file = # path you want to save the dict file in pkl format e.g. '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/fgadr_pkl_file.pkl'
    
- And run 'fgadr_generate_pkl.py' 
```shell
$ python fgadr_generate_pkl.py
```

**2. Update your training configurations and hyperparameters in `configs/fgadr.yaml`.**
- specify the paths for variales in line 3-5
    - data_index: # specify path of the pickle file containing image paths for test train data
    - save_path:  #specify path to save best models
    - log_path:  # specify path to save log files

- you can also update the other hyperparameters if required. 

**3. Run to train:**

```shell
$ CUDA_VISIBLE_DEVICES=x python main.py -c configs/fgadr.yaml
```

Optional arguments:
```
-c yaml_file      Specify the config file (default: configs/default.yaml)
-o                Overwrite save_path and log_path without warning
-p                Print configs before training
```

**4. Monitor your training progress in website [127.0.0.1:6006](127.0.0.1:6006) by running:**

```shell
$ tensorborad --logdir=/path/to/your/log --port=6006
```

[Tips to use tensorboard on a remote server](https://blog.yyliu.net/remote-tensorboard/)
