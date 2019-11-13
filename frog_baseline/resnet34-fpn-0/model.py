from common  import *
from dataset import *
from resnet  import *


####################################################################################################
# Feature Pyramid Networks for Object Detection- arXiv
# https://arxiv.org/abs/1612.03144
# Panoptic Feature Pyramid Networks - arXiv
# http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
# https://github.com/Angzz/panoptic-fpn-gluon/tree/master/gluoncv/model_zoo
#########################################################################

class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group,out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral

def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x




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
            ConvGnUp2d(128,128),
            ConvGnUp2d(128, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d(128, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d(128, 64),
        )

        self.top4 = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit = nn.Conv2d(64,num_class,kernel_size=1)


    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = F.pad(x,[18,18,2,2],mode='constant', value=0) #pad = (left, right, top, down)

        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        ##----
        #classify
        # x = F.dropout(x4,0.5,training=self.training)
        # x = F.adaptive_avg_pool2d(x, 1)
        # x = self.feature(x)
        # logit_label = self.logit_label(x)

        ##----
        #segment

        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #; print(t1.shape)
        t2 = self.top2(t2) #; print(t2.shape)
        t3 = self.top3(t3) #; print(t3.shape)

        x = torch.cat([t1,t2,t3],1)
        x = self.top4(x)
        logit = self.logit(x)[:,:,1:350+1,10:525+10]
        return logit


# https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/models/encnet.py
#########################################################################

#focal loss
def criterion_mask(logit, truth, weight=None):
    if weight is None: weight=[1,1,1,1]
    weight = torch.FloatTensor(weight).to(truth.device).view(1,-1,1,1)

    batch_size,num_class,H,W = logit.shape

    logit = logit.view(batch_size,num_class,H,W , 1)
    truth = truth.view(batch_size,num_class,H,W , 1)
    # return F.cross_entropy(logit, truth, reduction='mean')

    l = torch.cat([ -logit,logit],-1)
    t = torch.cat([1-truth,truth],-1)
    log_p = -F.logsigmoid(l)
    p = torch.sigmoid(l)

    loss = (t*log_p).sum(-1)

    #---
    # if 1:#image based focusing
    #     probability = probability.view(batch_size,H*W,5)
    #     truth  = truth.view(batch_size,H*W,1)
    #     weight = weight.view(1,1,5)
    #
    #     alpha  = 2
    #     focal  = torch.gather(probability, dim=-1, index=truth.view(batch_size,H*W,1))
    #     focal  = (1-focal)**alpha
    #     focal_sum = focal.sum(dim=[1,2],keepdim=True)
    #     #focal_sum = focal.sum().view(1,1,1)
    #     weight = weight*focal/focal_sum.detach() *H*W
    #     weight = weight.view(-1,5)

    loss = loss*weight
    loss = loss.mean()
    return loss


#----
def probability_mask_to_label(probability):
    batch_size,num_class,H,W = probability.shape
    probability = F.adaptive_max_pool2d(probability,1).view(batch_size,-1)
    return probability


#----

def metric_label(probability, truth, threshold=0.5):
    batch_size=len(truth)

    with torch.no_grad():
        probability = probability.view(batch_size,4)
        truth = truth.view(batch_size,4)

        #----
        neg_index = (truth==0).float()
        pos_index = 1-neg_index
        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)

        #----
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives
        tn = tn.sum(0)
        tp = tp.sum(0)

        #----
        tn = tn.data.cpu().numpy()
        tp = tp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return tn,tp, num_neg,num_pos





def metric_mask(probability, truth, threshold=0.1, sum_threshold=1):

    with torch.no_grad():
        batch_size,num_class,H,W = truth.shape
        probability = probability.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        d_neg = (p_sum < sum_threshold).float()
        d_pos = 2*(p*t).sum(-1)/((p+t).sum(-1)+1e-12)

        neg_index = (t_sum==0).float()
        pos_index = 1-neg_index

        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)
        dn = (neg_index*d_neg).sum(0)
        dp = (pos_index*d_pos).sum(0)

        #----
        dn = dn.data.cpu().numpy()
        dp = dp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return dn,dp, num_neg,num_pos

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

        image = cv2.imread(DATA_DIR + '/image/train0.50/%s.png'%(image_id[:-4]), cv2.IMREAD_COLOR)
        mask  = cv2.imread(DATA_DIR + '/mask/train0.25/%s.png'%(image_id[:-4]), cv2.IMREAD_UNCHANGED)
        mask  = mask.astype(np.float32)/255
        image = image.astype(np.float32)/255

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
    C, H, W    = 3, 700, 1050

    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = torch.from_numpy(input).float().cuda()

    net = Net().cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ',input.shape)
    print('logit: ',logit.shape)
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
        logit_mask  = net(input)
        print('input: ',input.shape)
        print('logit_mask: ',logit_mask.shape)
        print('')

        loss = criterion_mask(logit_mask, truth_mask)


        probability_mask  = torch.sigmoid(logit_mask)
        probability_label = probability_mask_to_label(probability_mask)
        tn,tp, num_neg,num_pos = metric_label(probability_label, truth_label)
        dn,dp, num_neg,num_pos = metric_mask(probability_mask, truth_mask)


        print('loss = %0.5f'%loss.item())
        print('tn,tp = [%0.3f,%0.3f,%0.3f,%0.3f], [%0.3f,%0.3f,%0.3f,%0.3f] '%(*(tn/(num_neg+1e-8)),*(tp/(num_pos+1e-8))))
        print('tn,tp = [%0.3f,%0.3f,%0.3f,%0.3f], [%0.3f,%0.3f,%0.3f,%0.3f] '%(*(dn/(num_neg+1e-8)),*(dp/(num_pos+1e-8))))
        print('num_pos,num_neg = [%d,%d,%d,%d], [%d,%d,%d,%d] '%(*num_neg,*num_pos))
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
    while i<=50: #100

        net.train()
        optimizer.zero_grad()

        logit_mask  = net(input)
        loss = criterion_mask(logit_mask, truth_mask)

        probability_mask  = torch.sigmoid(logit_mask)
        probability_label = probability_mask_to_label(probability_mask)
        tn,tp, num_neg,num_pos = metric_label(probability_label, truth_label)
        dn,dp, num_neg,num_pos = metric_mask(probability_mask, truth_mask)

        (loss).backward()
        optimizer.step()

        if i%10==0:
            print('[%05d] %8.5f | [%0.2f,%0.2f,%0.2f,%0.2f : %0.2f,%0.2f,%0.2f,%0.2f] | [%0.2f,%0.2f,%0.2f,%0.2f : %0.2f,%0.2f,%0.2f,%0.2f] '%(
                i,
                loss.item(),
                *(tn/(num_neg+1e-8)),*(tp/(num_pos+1e-8)),
                *(dn/(num_neg+1e-8)),*(dp/(num_pos+1e-8)),
            ))
        i = i+1
    print('')


    #exit(0)
    if 1:
        #net.eval()
        logit_mask = net(input)
        probability_mask  = torch.sigmoid(logit_mask)
        probability_label = probability_mask_to_label(probability_mask)

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


