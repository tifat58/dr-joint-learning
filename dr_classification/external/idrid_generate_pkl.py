import os
import pandas as pd
import pickle


# update the paths accordingly
train_image_dir = '/mnt/sda/haal02-data/IDRID-updated/Processed/DiseaseGrading/OriginalImages/TrainingSet'
train_groundtruth_file = '/mnt/sda/haal02-data/IDRID-updated/DiseaseGrading/Groundtruths/IDRiD_Disease_Grading_Training_Labels.csv'

test_image_dir = '/mnt/sda/haal02-data/IDRID-updated/Processed/DiseaseGrading/OriginalImages/TestingSet'
test_groundtruth_file = '/mnt/sda/haal02-data/IDRID-updated/DiseaseGrading/Groundtruths/IDRiD_Disease_Grading_Testing_Labels.csv'


path_to_save_pkl_file = '/mnt/sda/haal02-data/IDRID-updated/DiseaseGrading/Groundtruths/processed_idrid_pkl_file.pkl' # use this saved pkl file path in yaml config file


train_df = pd.read_csv(train_groundtruth_file)
test_df = pd.read_csv(test_groundtruth_file)



# train data list
train_data_list = []
for idx, row in train_df.iterrows():
#     print(row)
    image_path = os.path.join(train_image_dir, row[0]+'.jpg')
    # print(image_path)
    if os.path.isfile(image_path):
        img_tup = (image_path, int(row[1]))
        train_data_list.append(img_tup)
#         print(row[0], row[1])


# test and val data list
test_data_list = []
for idx, row in test_df.iterrows():
#     print(row)
    image_path = os.path.join(test_image_dir, row[0]+'.jpg')
#     print(image_path)
    if os.path.isfile(image_path):
        img_tup = (image_path, int(row[1]))
        test_data_list.append(img_tup)
#         print(row[0], row[1])


# test and val data list
val_data_list = []
for idx, row in test_df.iterrows():
#     print(row)
    image_path = os.path.join(test_image_dir, row[0]+'.jpg')
#     print(image_path)
    if os.path.isfile(image_path):
        img_tup = (image_path, int(row[1]))
        val_data_list.append(img_tup)
#         print(row[0], row[1])

print('Train data len: ', len(train_data_list))
print('Test data len: ', len(test_data_list))

data_dict = {
    'train' : train_data_list,
    'val' : val_data_list,
    'test' : test_data_list
}

pickle.dump(data_dict, open(path_to_save_pkl_file, 'wb'))

