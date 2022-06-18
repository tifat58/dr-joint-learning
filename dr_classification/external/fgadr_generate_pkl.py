import os
import pandas as pd
import pickle


# update the paths accordingly
image_dir = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/Processed-dataset/Original_Images'
groundtruth_file = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/DR_Seg_Grading_Label.csv'
path_to_save_pkl_file = '/mnt/sda/haal02-data/FGADR-Seg-Set/Seg-set/processed_fgadr_pkl_file_frist_half.pkl' # use this saved pkl file path in yaml config file


df = pd.read_csv(groundtruth_file)

train_df = df[0:821]
val_df = df[821:921]
test_df = df[921:]


# train data list
train_data_list = []
for idx, row in train_df.iterrows():
#     print(row)
    image_path = os.path.join(image_dir, row[0])
#     print(image_path)
    if os.path.isfile(image_path):
        img_tup = (image_path, int(row[1]))
        train_data_list.append(img_tup)
#         print(row[0], row[1])


# test and val data list
test_data_list = []
for idx, row in test_df.iterrows():
#     print(row)
    image_path = os.path.join(image_dir, row[0])
#     print(image_path)
    if os.path.isfile(image_path):
        img_tup = (image_path, int(row[1]))
        test_data_list.append(img_tup)
#         print(row[0], row[1])


# test and val data list
val_data_list = []
for idx, row in val_df.iterrows():
#     print(row)
    image_path = os.path.join(image_dir, row[0])
#     print(image_path)
    if os.path.isfile(image_path):
        img_tup = (image_path, int(row[1]))
        val_data_list.append(img_tup)
#         print(row[0], row[1])

data_dict = {
    'train' : train_data_list,
    'val' : val_data_list,
    'test' : test_data_list
}

pickle.dump(data_dict, open(path_to_save_pkl_file, 'wb'))

