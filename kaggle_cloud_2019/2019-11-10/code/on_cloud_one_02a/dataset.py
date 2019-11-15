from common import *
from kaggle import *


DATA_DIR = '/root/share/project/kaggle/2019/cloud/data'

class CloudDataset(Dataset):
    def __init__(self, split, csv, mode,
                 folder={'image': '1050x700', 'mask': '525x350'}, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.folder  = folder
        self.augment = augment

        self.uid = list(np.concatenate([np.load(DATA_DIR + '/split/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(DATA_DIR + '/%s'%f).fillna('') for f in csv])
        df = df_loc_by_list(df, 'Image_Label', [ u[0] + '_%s'%CLASSNO_TO_CLASSNAME[c]  for u in self.uid for c in [0,1,2,3] ])


        df[['image_id','class_name']]= df['Image_Label'].str.split('_', expand = True)
        df['class_no']=df['class_name'].map(CLASSNAME_TO_CLASSNO)
        df['encoded_pixel']=df['EncodedPixels']
        df['label'] = (df['EncodedPixels']!='').astype(np.int32)

        df = df[['image_id','class_no','class_name','label','encoded_pixel']]
        df_label = pd.pivot_table(df, values = 'label', index=['image_id'], columns = 'class_name').reset_index()

        self.df = df
        self.df_label = df_label
        self.num_image = len(self.uid)


    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        string += '\tmode    = %s\n'%self.mode
        string += '\tsplit   = %s\n'%self.split
        string += '\tcsv     = %s\n'%str(self.csv)
        string += '\tfolder  = %s\n'%self.folder
        string += '\tnum_image = %d\n'%self.num_image
        if self.mode == 'train':
            label = self.df_label[list(CLASSNAME_TO_CLASSNO.keys())].values

            num_image = len(label)
            num_pos = label.sum(0)
            num_neg = num_image - num_pos
            for c in range(NUM_CLASS):
                pos = num_pos[c]
                neg = num_neg[c]
                num = num_image
                string += '\t%16s   neg%d, pos%d = %5d  (%0.3f),  %5d  (%0.3f)\n'%(CLASSNO_TO_CLASSNAME[c], c,c,neg,neg/num,pos,pos/num)

        return string


    def __len__(self):
        return self.num_image


    def __getitem__(self, index):
        # print(index)
        image_id, folder = self.uid[index]
        #image = cv2.imread(DATA_DIR + '/image/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        image = cv2.imread(DATA_DIR + '/image/%s_%s/%s.png'%(folder, self.folder['image'], image_id[:-4]), cv2.IMREAD_COLOR)

        if self.mode == 'train':
            mask = cv2.imread(DATA_DIR +'/mask/%s_%s/%s.png'%(folder, self.folder['mask'], image_id[:-4]), cv2.IMREAD_UNCHANGED)
        else:
            #MASK_WIDTH =525
            #MASK_HEIGHT=350
            mask_width, mask_height = self.folder['mask'].split('_').split('x')[-2:]
            mask = np.zeros((int(mask_height),int(mask_width),4), np.uint8)

        image = image.astype(np.float32)/255
        mask  = mask.astype(np.float32)/255
        label = self.df_label.loc[self.df_label['image_id']==image_id][list(CLASSNAME_TO_CLASSNO.keys())].values[0]

        infor = Struct(
            index    = index,
            image_id = image_id,
        )

        if self.augment is None:
            return image, label, mask, infor
        else:
            return self.augment(image, label, mask, infor)



def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_label = []
    truth_mask  = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_label.append(batch[b][1])
        truth_mask.append(batch[b][2])
        infor.append(batch[b][3])

    input = np.stack(input)
    input = input[...,::-1].copy()
    input = input.transpose(0,3,1,2)

    truth_mask = np.stack(truth_mask)
    truth_mask = truth_mask.transpose(0,3,1,2) ## change to 0,1?

    truth_label = np.stack(truth_label)

    #----
    input = torch.from_numpy(input).float()
    truth_label = torch.from_numpy(truth_label).float()
    truth_mask = torch.from_numpy(truth_mask).float()


    #recompute
    if 1:
        m = truth_mask.view(batch_size,NUM_CLASS,-1).sum(-1)
        truth_label = (m>0).float()

    return input, truth_label, truth_mask, infor


##############################################################

def tensor_to_image(tensor):
    image = tensor.data.cpu().numpy()
    image = image.transpose(0,2,3,1)
    image = image[...,::-1]
    return image

def tensor_to_mask(tensor):
    mask = tensor.data.cpu().numpy()
    mask = mask.transpose(0,2,3,1)
    return mask


##############################################################

def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask  = cv2.flip(mask, 1)
    return image, mask

def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask  = cv2.flip(mask, 0)

    return image, mask



def do_random_crop(image, mask, w, h, mw,mh):
    height, width = image.shape[:2]
    height_mask, width_mask = mask.shape[:2]
    mask = cv2.resize( mask, dsize=(width,height), interpolation=cv2.INTER_LINEAR)

    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [y:y+h,x:x+w]

    mask  = cv2.resize( mask,  dsize=(mw,mh), interpolation=cv2.INTER_LINEAR)
    return image, mask


def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    height_mask, width_mask = mask.shape[:2]
    mask = cv2.resize( mask, dsize=(width,height), interpolation=cv2.INTER_LINEAR)

    x,y= 0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [y:y+h,x:x+w]

    #---
    if (w,h)!=(width,height):
        image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask,  dsize=(width_mask, height_mask), interpolation=cv2.INTER_LINEAR)

    return image, mask



def do_random_crop_rotate_rescale(image, mask, mode=['rotate','scale','shift']):
    height, width = image.shape[:2]
    height_mask, width_mask = mask.shape[:2]
    mask = cv2.resize( mask, dsize=(width,height), interpolation=cv2.INTER_LINEAR)

    dangle = 0
    dscale_x, dscale_y = 0,0
    dshift_x, dshift_y = 0,0

    if 'rotate' in mode:
        dangle = np.random.uniform(-30, 30)
    if 'scale' in mode:
        dscale_x, dscale_y = np.random.uniform(-1, 1, 2)*0.15
    if 'shift' in mode:
        dshift_x, dshift_y = np.random.uniform(-1, 1, 2)*0.10

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift_x*width, dshift_y*height

    src = np.array([[-width/2,-height/2],[ width/2,-height/2],[ width/2, height/2],[-width/2, height/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+width/2 +tx
    y = (src*[sin, cos]).sum(1)+height/2+ty
    src = np.column_stack([x,y])



    dst = np.array([[0,0],[width,0],[width,height],[0,height]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask = cv2.warpPerspective( mask, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    mask = cv2.resize( mask,  dsize=(width_mask, height_mask), interpolation=cv2.INTER_LINEAR)

    return image, mask

def do_random_log_contast(image, gain=[0.70, 1.30] ):
    gain = np.random.uniform(gain[0],gain[1],1)
    inverse = np.random.choice(2,1)

    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image,0,1)
    return image

# def do_random_noise(image, noise=8):
#     H,W = image.shape[:2]
#     image = image.astype(np.float32)
#     image = image + np.random.uniform(-1,1,(H,W,1))*noise
#     image = np.clip(image,0,255).astype(np.uint8)
#     return image
#
# ##---
# #https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# def do_random_contast(image):
#     beta=0
#     alpha=random.uniform(0.5, 2.0)
#     image = image.astype(np.float32) * alpha + beta
#     return image
#
# #----
#
#
# def do_random_cutout(image, mask):
#     height, width = image.shape[:2]
#
#     u0 = [0,1][np.random.choice(2)]
#     u1 = np.random.choice(width)
#
#     if u0 ==0:
#         x0,x1=0,u1
#     if u0 ==1:
#         x0,x1=u1,width
#
#     image[:,x0:x1]=0
#     mask [:,x0:x1]=0
#     return image,mask


# shuffle
def do_random_grid_shuffle(image, mask):
    height, width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    mask = cv2.resize( mask, dsize=(width,height), interpolation=cv2.INTER_LINEAR)

    x0,y0 = 0,0
    x1,y1 = width//2,height//2
    x2,y2 = width,height

    grid = [
        [(x0,y0),(x1,y1)],
        [(x1,y0),(x2,y1)],
        [(x0,y1),(x1,y2)],
        [(x1,y1),(x2,y2)],
    ]

    image_grid = []
    mask_grid  = []
    for (xx0,yy0),(xx1,yy1) in grid:
        image_grid.append(image[yy0:yy1,xx0:xx1])
        mask_grid.append(mask[yy0:yy1,xx0:xx1])

    s = np.arange(4)
    np.random.shuffle(s)

    image = np.hstack([
        np.vstack([image_grid[s[0]],image_grid[s[1]]]),
        np.vstack([image_grid[s[2]],image_grid[s[2]]]),
    ])
    mask = np.hstack([
        np.vstack([mask_grid[s[0]],mask_grid[s[1]]]),
        np.vstack([mask_grid[s[2]],mask_grid[s[2]]]),
    ])


    mask = cv2.resize( mask, dsize=(mask_width,mask_height), interpolation=cv2.INTER_LINEAR)
    return image,mask

##############################################################

def run_check_dataset():

    dataset = CloudDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_5546.npy',],
        #split   = ['by_random1/valid_small_fold_a0_120.npy',],
        augment = None,
    )

    # dataset = CloudDataset(
    #     mode    = 'test',
    #     csv     = ['sample_submission.csv',],
    #     split   = ['test_3698.npy',],
    #     augment = None,
    # )
    print(dataset)
    #exit(0)

    for n in range(0,len(dataset)):
        i = n #i = np.random.choice(len(dataset))

        image, label, mask, infor = dataset[i]
        overlay = draw_truth(image, label, mask, infor)

        #----
        print('%05d : %s'%(i, infor.image_id))
        print('label = %s'%str(label))
        print('')
        #image_show('image',image,0.5)
        image_show('overlay',overlay)
        cv2.waitKey(0)




def run_check_dataloader():

    dataset = CloudDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_5546.npy',],
        augment = None, #
    )
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = SequentialSampler(dataset),
        #sampler     = RandomSampler(dataset),
        batch_size  = 5,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    for t,(input, truth_label, truth_mask, infor) in enumerate(loader):

        print('----t=%d---'%t)
        print('')
        print(infor)
        print('input', input.shape)
        print('truth_label', truth_label.shape)
        print('truth_mask ', truth_mask.shape)
        print('')

        if 1:
            batch_size= len(infor)

            image       = tensor_to_image(input)
            truth_label = truth_label.data.cpu().numpy()
            truth_mask  = tensor_to_mask(truth_mask)

            for b in range(batch_size):

                overlay = draw_truth(image[b], truth_label[b], truth_mask[b], infor[b])

                #----
                print('%05d : %s'%(b, infor[b].image_id))
                print('label = %s'%str(truth_label[b]))
                print('')
                image_show('overlay',overlay)
                cv2.waitKey(0)



def run_check_augment():
    # 'image': '1050x700', 'mask': '525x350'
    def augment(image, label, mask, infor):
        if 0:
            #if np.random.rand()<0.5:  image, mask = do_flip_ud(image, mask)
            if np.random.rand()<0.5:  image, mask = do_flip_lr(image, mask)

        if 0:
            #image, mask = do_random_crop_rescale(image,mask, w=945, h=630)
            image, mask = do_random_crop_rotate_rescale(image, mask, mode=['rotate','scale','shift'])
            #image, mask = do_random_crop_rotate_rescale(image, mask, mode=['rotate'])
            #image, mask = do_random_crop_rotate_rescale(image, mask, mode=['scale' ])
            #image, mask = do_random_crop_rotate_rescale(image, mask, mode=['shift' ])

        if 0:
            image, mask = do_random_crop(image, mask, 512, 512, 128,128)

        if 0:
            image = do_random_log_contast(image, gain=[0.70, 1.50])

        if 1:
            image, mask = do_random_grid_shuffle(image, mask)

        #
        # image = do_random_noise(image, noise=16)

        return image, label, mask, infor


    dataset = CloudDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_5546.npy',],
        augment = None, #
    )
    print(dataset)


    for b in range(len(dataset)):
        image, label, mask, infor = dataset[b]
        overlay = draw_truth(image, label, mask, infor, None)

        #----
        print('%05d : %s'%(b, infor.image_id))
        print('label = %s'%str(label))
        print('')
        image_show('before',overlay)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, label1, mask1, infor1 =  augment(image.copy(), label.copy(), mask.copy(), infor)
                overlay1 = draw_truth(image1, label1, mask1, infor1, None)

                #----
                print('%05d : %s'%(b, infor1.image_id))
                print('label = %s'%str(label1))
                print(image1.shape, mask1.shape)
                print('')
                image_show('after',overlay1)
                cv2.waitKey(1)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_check_dataset()
    #run_check_dataloader()
    run_check_augment()





