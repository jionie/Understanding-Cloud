from .include import *

# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def soft_dice_criterion(logit, truth):
    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)

    p = p*2-1
    t = t*2-1

    #non-empty
    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice
    return loss

def soft_dice1_criterion(logit, truth):
    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)

    p = p*2-1
    t = t*2-1

    #non-empty
    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  =  2*intersection/union

    eps  = 1e-12
    dice = torch.clamp(dice,eps,1-eps)


    loss = -torch.log(dice)
    return loss

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

    return tn, tp, num_neg, num_pos


def metric_mask(probability, truth, threshold=0.1, sum_threshold=1):
    
    with torch.no_grad():
        batch_size, num_class, H, W = truth.shape
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
