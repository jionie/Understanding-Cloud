from common import *


df = pd.read_csv('/root/share/project/kaggle/2019/cloud/data/train.csv').fillna('')
df[['image_id','name']]= df['Image_Label'].str.split('_', expand = True)
df['label'] = (df['EncodedPixels']!='').astype(np.int32)


d = pd.pivot_table(df, values = 'label', index=['image_id'], columns = 'name').reset_index()
d.to_csv('/root/share/project/kaggle/2019/cloud/data/train.more1.csv', index=False)


zz=0