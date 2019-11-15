from sklearn.model_selection import KFold,StratifiedKFold
import pandas as pd
import numpy as np
import os

PATH = '/media/jionie/my_disk/Kaggle/Cloud/input/understanding_cloud_organization'


train_csv = pd.read_csv(PATH+'/train.csv')
train_csv['ImageId'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[0])
train_csv['Label'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[1])
train_csv = train_csv.drop('Image_Label', axis=1)
train_csv = train_csv.fillna(-1)

train_df = train_csv.copy()
train_df = train_df[train_df['EncodedPixels']!=-1]

CLASSNAME_TO_CLASSNO = {
'Fish'   : 0,
'Flower' : 1,
'Gravel' : 2,
'Sugar'  : 3,
}
train_df['class_no']=train_df['Label'].map(CLASSNAME_TO_CLASSNO)

train_df_label = train_df[['ImageId', 'class_no']]
uid = list(train_df_label['ImageId'].unique())

image_class_dict = {}
class_list = []
id_list = []

for u in uid:
    classes = list(train_df_label[train_df_label['ImageId'] == u].class_no)
    class_one_hot = np.zeros(4).astype(np.int)
    for c in classes:
        class_one_hot[c] = 1
    class_string = ""
    for i in range(4):
        class_string += str(class_one_hot[i]) + ' '
    id_list.append(u)
    class_list.append(class_string[:-1])
    
image_class_dict['id'] = id_list
image_class_dict['class'] = class_list

image_class_df = pd.DataFrame.from_dict(image_class_dict)

image_class_df.to_csv("/media/jionie/my_disk/Kaggle/Cloud/input/understanding_cloud_organization/image_class.csv")