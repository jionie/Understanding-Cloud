from common  import *
from dataset import *
from resnet  import *

from loss_more import *
####################################################################################################
#
# Feature Pyramid Networks for Object Detection- arXiv
# https://arxiv.org/abs/1612.03144
# Panoptic Feature Pyramid Networks - arXiv
# http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
# https://github.com/Angzz/panoptic-fpn-gluon/tree/master/gluoncv/model_zoo
#
####################################################################################################


def resize_like(x, reference, mode='bilinear'):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode=='bilinear':
            x = F.interpolate(x, size=reference.shape[2:],mode='bilinear',align_corners=False)
        if mode=='nearest':
            x = F.interpolate(x, size=reference.shape[2:],mode='nearest')
    return x


def upsize_add(x, lateral):
    x = resize_like(x, lateral, mode='nearest')  + lateral
    return x


def fuse(x, mode='cat'):
    batch_size,C0,H0,W0 = x[0].shape

    for i in range(1,len(x)):
        _,_,H,W = x[i].shape
        if (H,W)!=(H0,W0):
            x[i] = F.interpolate(x[i], size=(H0,W0), mode='bilinear', align_corners=False)

    if mode=='cat':
        return torch.cat(x,1)


class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, is_upsize=None, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.is_upsize = is_upsize
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group,out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        if self.is_upsize :
            x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=False)
        return x


#-------------------------------

class Net(nn.Module):
    def load_pretrain(self, skip=['logit.'], is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=4):
        super(Net, self).__init__()

        e = ResNet34()
        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  #dropped


        #---
        self.lateral0 = nn.Conv2d(512, 128,   kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d(256, 128,   kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(128, 128,   kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d( 64, 128,   kernel_size=1, padding=0, stride=1)


        self.top1 = nn.Sequential(
            ConvGnUp2d(128,128, True),
            ConvGnUp2d(128, 64, True),
            ConvGnUp2d( 64, 64, True),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d(128, 64, True),
            ConvGnUp2d( 64, 64, True),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d(128, 64, True),
        )
        self.logit = nn.Conv2d(64*3,num_class,kernel_size=1)


    def forward(self, x):
        batch_size,C,H,W = x.shape

        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        ##----
        #segment

        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #; print(t1.shape)
        t2 = self.top2(t2) #; print(t2.shape)
        t3 = self.top3(t3) #; print(t3.shape)

        x = fuse([t1,t2,t3], 'cat')
        logit = self.logit(x)

        #---
        probability_mask  = torch.sigmoid(logit)
        probability_label = F.adaptive_max_pool2d(probability_mask,1).view(batch_size,-1)
        return probability_label, probability_mask


# https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/models/encnet.py
#########################################################################

#focal loss
def criterion(probability_label, probability_mask, truth_label, truth_mask):

    #label
    p = torch.clamp(probability_label, 1e-9, 1-1e-9)
    t = truth_label
    loss_label = - t*torch.log(p) - 2*(1-t)*torch.log(1-p)
    loss_label = loss_label.mean()

    #mask
    w = probability_label.detach().view(-1,4,1,1)
    p = torch.clamp(probability_mask, 1e-9, 1-1e-9)
    t = truth_mask

    #loss_mask = - w*t*torch.log(p) - (1-w)*(1-t)*torch.log(1-p)
    #loss_mask = - t*torch.log(p) - (1-t)*torch.log(1-p)
    #loss_mask = loss_mask.mean()

    loss_mask = F.binary_cross_entropy(probability_mask,truth_mask, reduction='mean')


    return loss_label, loss_mask

# def safe_divide(n, d):
#     t = (n==0 )*(d==0)
#     divide = n/d
#     divide[safe_divide]=1
#     return divide

def metric (probability_label, probability_mask, truth_label, truth_mask, use_reject=True):

    threshold_label=0.60
    threshold_mask =0.30
    threshold_size =   1

    with torch.no_grad():
        batch_size,num_class=truth_label.shape

        probability = probability_label.view(batch_size,num_class)
        truth = truth_label.view(batch_size,num_class)

        #----
        lp = (probability>threshold_label).float()
        lt = (truth>0.5).float()
        num_tp = lt.sum(0)
        num_tn = batch_size-num_tp

        #----
        tp = ((lp + lt) == 2).float()  # True positives
        tn = ((lp + lt) == 0).float()  # True negatives
        tn = tn.sum(0)
        tp = tp.sum(0)


        #----------------------------------------------------------
        batch_size,num_class,H,W = truth_mask.shape

        probability = probability_mask.view(batch_size,num_class,-1)
        truth = truth_mask.view(batch_size,num_class,-1)

        #------
        mp = (probability>threshold_mask).float()
        mt = (truth>0.5).float()
        mt_sum = mt.sum(-1)
        mp_sum = mp.sum(-1)

        neg_index = (mt_sum==0).float()
        pos_index = 1-neg_index

        if use_reject:#get subset
            neg_index = neg_index*lp
            pos_index = pos_index*lp

        num_dn = neg_index.sum(0)
        num_dp = pos_index.sum(0)

        #------
        dn = (mp_sum < threshold_size).float()
        dp = 2*(mp*mt).sum(-1)/((mp+mt).sum(-1)+1e-12)
        dn = (dn*neg_index).sum(0)
        dp = (dp*pos_index).sum(0)

        #----
        all = torch.cat([
            tn,tp,num_tn,num_tp,
            dn,dp,num_dn,num_dp,
        ])
        all = all.data.cpu().numpy().reshape(-1,num_class)
        tn,tp,num_tn,num_tp, dn,dp,num_dn,num_dp = all

    return tn,tp,num_tn,num_tp, dn,dp,num_dn,num_dp


##############################################################################################
def make_dummy_data(batch_size=8):

    data = np.array([
        i+'.jpg' for i in [
            'a63867a','6aa2698','40f4a1b','50d3f97','4d7ab3d','540e758','c7f606d','a21d2e5','a47732c','5c9625b',
            'c0e8adb','3685cdd','73ad4aa','8d46873','2d9da73','c8c0269','ac8776b','d3cc177','a52d1e7','247f10c',
            'b734fad','50ebf1b','68562e6','85c4f1a','70448b3','b5908da','cbb861f','c72db3e','a5a4b72','e672f32',
            '4440011','548cb15','0745d08','1a3f6e6','a25e3ad','8e9d722','9075d29','ee4aeaf','6947f73','ba6522b',
            'bbf0cf8','5796eb0','93e44c0','82a29b3','a19b432','bf68101','541833b','f1e4864','c67f808','3a1d51d',
            'f203606','c0952e5','66aad93','d2c5a99','b991088','fce721b','4ac2abc','6c948cf','39c7521','efe1652',
            '0f1786f','b21394c','a152a41','2d4a01b','b54007a','5726c26','790f83a','12ae62b','3437a8c','4610b10',
            'c0dbe22','f13cbe0','df7fcd3','416aa8a','2366810','009e2f3','338b26d','52df6e7','3087f1e','32f591b',
            'fc25229','95ba48c','882a79a','c32bfd6','ac80e3b','bbec1a5','d9c6728','2107837','916f7ac','064dd48',
            'dd0f0b2','9a2d654','7ca1d0b','0741fda','8a7368e','28f3277','59c8593','b981995','540fde7','25f5f23',
        ]
    ])
    num_image = len(data)
    DATA_DIR = '/root/share/project/kaggle/2019/cloud/data'


    batch = []
    for b in range(0, batch_size):
        i = b%num_image
        image_id = data[i]

        image = cv2.imread(DATA_DIR + '/image/train_1050x700/%s.png'%(image_id[:-4]), cv2.IMREAD_COLOR)
        mask  = cv2.imread(DATA_DIR + '/mask/train_525x350/%s.png'%(image_id[:-4]), cv2.IMREAD_UNCHANGED)

        #image = cv2.resize(image, dsize=(width,height))
        image = image.astype(np.float32)/255
        mask  = mask.astype(np.float32)/255

        label = (mask.sum(0).sum(0)>0).astype(np.float32)

        infor = Struct(
            index    = b,
            image_id = image_id,
        )
        batch.append([image,label,mask,infor])

    input, truth_label, truth_mask, infor = null_collate(batch)
    input = input.cuda()
    truth_label = truth_label.cuda()
    truth_mask  = truth_mask.cuda()

    return input, truth_label, truth_mask, infor





#########################################################################
def run_check_basenet():
    net = Net()
    print(net)

    #---
    if 1:
        print(net)
        print('')

        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')

    net.load_pretrain()



def run_check_net():

    batch_size = 1
    C, H, W    = 3, 384, 512

    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = torch.from_numpy(input).float().cuda()

    net = Net().cuda()
    net.eval()

    with torch.no_grad():
        probability_label, probability_mask = net(input)

    print('')
    print('input: ',input.shape)
    print('probability_label: ',probability_label.shape)
    print('probability_mask: ',probability_mask.shape)
    #print(net)



def run_check_train():


    if 1:
        input, truth_label, truth_mask, infor = make_dummy_data(batch_size=6)
        batch_size, C, H, W  = input.shape

        print('input: ',input.shape)
        print('truth_label: ',truth_label.shape)
        print('(count)    : ',truth_label.sum(0))
        print('truth_mask: ',truth_mask.shape)
        print('')

    #---

    net = Net().cuda()
    net.load_pretrain(is_print=False)#

    net = net.eval()
    with torch.no_grad():
        probability_label, probability_mask   = net(input)
        probability_mask = resize_like(probability_mask, truth_mask, mode='bilinear')

        print('input: ',input.shape)
        print('probability_label: ',probability_label.shape)
        print('probability_mask: ',probability_mask.shape)
        print('')

        loss_label, loss_mask = criterion(probability_label, probability_mask, truth_label, truth_mask)
        tn,tp,num_tn,num_tp, dn,dp,num_dn,num_dp = metric (probability_label, probability_mask, truth_label, truth_mask)

        print('loss_label = %0.5f'%loss_label.item())
        print('loss_mask = %0.5f'%loss_mask.item())
        print('tn,tp = [%0.3f,%0.3f,%0.3f,%0.3f], [%0.3f,%0.3f,%0.3f,%0.3f] '%(*(tn/(num_tn+1e-8)),*(tp/(num_tp+1e-8))))
        print('tn,tp = [%0.3f,%0.3f,%0.3f,%0.3f], [%0.3f,%0.3f,%0.3f,%0.3f] '%(*(dn/(num_dn+1e-8)),*(dp/(num_dp+1e-8))))
        print('num_tn,num_tp = [%d,%d,%d,%d], [%d,%d,%d,%d] '%(*num_tn,*num_tp))
        print('num_dn,num_dp = [%d,%d,%d,%d], [%d,%d,%d,%d] '%(*num_dn,*num_dp))
        print('')
    #exit(0)

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =',batch_size)
    print('---------------------------------------------------------------------------------------------------------------')
    print('[iter ]     loss |          [tn1,2,3,4  : tp1,2,3,4]           |          [dn1,2,3,4  : dp1,2,3,4]            ')
    print('---------------------------------------------------------------------------------------------------------------')
          #[00040]  0.10762 | [1.00,1.00,1.00,1.00 : 1.00,1.00,1.00,1.00] | [1.00,1.00,1.00,1.00 : 0.89,0.94,0.97,0.96]


    i=0
    optimizer.zero_grad()
    while i<=60: #100

        net.train()
        optimizer.zero_grad()

        probability_label, probability_mask = net(input)
        probability_mask = resize_like(probability_mask, truth_mask, mode='bilinear')

        loss_label, loss_mask = criterion(probability_label, probability_mask, truth_label, truth_mask)
        #(loss_label + loss_mask).backward()
        (loss_mask).backward()
        optimizer.step()

        tn,tp,num_tn,num_tp, dn,dp,num_dn,num_dp = metric (probability_label, probability_mask, truth_label, truth_mask)
        if i%10==0:
            print(
                '[%05d] %8.5f, %8.5f | '%(i,loss_label.item(),loss_mask.item(),) + \
                '[%0.2f,%0.2f,%0.2f,%0.2f : %0.2f,%0.2f,%0.2f,%0.2f] | '%(*(tn/(num_tn+1e-8)),*(tp/(num_tp+1e-8))) + \
                '[%0.2f,%0.2f,%0.2f,%0.2f : %0.2f,%0.2f,%0.2f,%0.2f] '%(*(dn/(num_dn+1e-8)),*(dp/(num_dp+1e-8)))
            )
        i = i+1
    print('')


    #exit(0)
    if 1:

        #---
        image       = tensor_to_image(input)
        truth_mask  = tensor_to_mask(truth_mask)
        probability_mask  = tensor_to_mask(probability_mask)
        truth_label = truth_label.data.cpu().numpy()
        probability_label = probability_label.data.cpu().numpy()

        for b in range(batch_size):
            print('%2d ------ '%(b))
            result = draw_predict_result(
                image[b], truth_label[b], truth_mask[b], probability_label[b], probability_mask[b])

            image_show('result',result, resize=1)
            cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_basenet()
    #run_check_net()
    run_check_train()


    print('\nsucess!')


