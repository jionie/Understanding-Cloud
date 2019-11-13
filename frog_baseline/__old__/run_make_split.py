from common import *

if 1:
    folder = 'test'
    file = glob.glob('/root/share/project/kaggle/2019/cloud/data/image/%s/*.jpg'%(folder))
    print(len(file))

    all = []
    for f in file:
        image_id = f.split('/')[-1]
        all.append((image_id,folder))
    test=all
    np.save('/root/share/project/kaggle/2019/cloud/data/test_%d.npy'%(len(test)),test)



    exit(0)

if 0:
    folder = 'train'

    file = glob.glob('/root/share/project/kaggle/2019/cloud/data/image/%s/*.jpg'%(folder))
    print(len(file))

    all = []
    for f in file:
        image_id = f.split('/')[-1]
        all.append((image_id,folder))



    for s in range(3):
        valid = all[s*300:(s+1)*300]
        train = list(set(all)-set(valid))
        valid_small = valid[:120]

        np.save('/root/share/project/kaggle/2019/cloud/data/train_fold_a%d_%d.npy'%(s,len(train)),train)
        np.save('/root/share/project/kaggle/2019/cloud/data/valid_fold_a%d_%d.npy'%(s,len(valid)),valid)
        np.save('/root/share/project/kaggle/2019/cloud/data/valid_small_fold_a%d_%d.npy'%(s,len(valid_small)),valid_small)

    #
    # split = np.array(split)
    # train=split
    # np.save('/root/share/project/kaggle/2019/cloud/data/train_%d.npy'%(len(train)),train)

