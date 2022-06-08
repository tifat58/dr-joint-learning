import os
import pandas as pd
import pickle


# update the paths accordingly
image_dir = '/home/hnguyen/DATA/SSL-Medical/KOTORI/Eye_Dataset/FGADR/Seg-set/Original_Images'
groundtruth_file = '/home/hnguyen/DATA/SSL-Medical/KOTORI/Eye_Dataset/FGADR/Seg-set/DR_Seg_Grading_Label.csv'
path_to_save_pkl_file = '/home/hnguyen/DATA/SSL-Medical/KOTORI/Eye_Dataset/FGADR/Seg-set/fgadr_pkl_file.pkl' # use this saved pkl file path in yaml config file


df = pd.read_csv(groundtruth_file)

train_df = df[0:1500]
test_df = df[1500:]


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


data_dict = {
    'train' : train_data_list,
    'val' : test_data_list,
    'test' : test_data_list
}

pickle.dump(data_dict, open(path_to_save_pkl_file, 'wb'))

