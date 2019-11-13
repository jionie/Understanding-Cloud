import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *


################################################################################################

def train_augment(image, label, mask, infor):
    # 'image': '1050x700', 'mask': '525x350'

    if np.random.rand()>0.5:
        image, mask = do_flip_lr(image, mask)
    if np.random.rand()>0.5:
        image, mask = do_flip_ud(image, mask)

    image, mask = random.choice([
        lambda image, mask : (image, mask),
        lambda image, mask : do_random_crop_rescale(image,mask, w=925, h=630),
        lambda image, mask : do_random_crop_rotate_rescale(image, mask, mode=['rotate']),
    ])(image, mask)


    image = cv2.resize(image,  dsize=(384, 576), interpolation=cv2.INTER_LINEAR)
    return image, label, mask, infor



def valid_augment(image, label, mask, infor):

    image = cv2.resize(image,  dsize=(384, 576), interpolation=cv2.INTER_LINEAR)
    return image, label, mask, infor



#------------------------------------
def do_valid(net, valid_loader, out_dir=None):

    valid_loss = np.zeros(18, np.float32)
    valid_num  = np.zeros_like(valid_loss)

    for t, (input, truth_label, truth_mask, infor) in enumerate(valid_loader):

        #if b==5: break
        batch_size = len(infor)

        net.eval()
        input = input.cuda()
        truth_label = truth_label.cuda()
        truth_mask  = truth_mask.cuda()

        with torch.no_grad():
            probability_label, probability_mask = data_parallel(net, input)
            probability_mask = resize_like(probability_mask, truth_mask, mode='bilinear')

            loss_label, loss_mask = criterion(probability_label, probability_mask, truth_label, truth_mask)
            tn,tp,num_tn,num_tp, dn,dp,num_dn,num_dp = metric (probability_label, probability_mask, truth_label, truth_mask)
        #---
        l = np.array([ loss_label.item()*batch_size, loss_mask.item()*batch_size, *tn, *tp, *dn, *dp ])
        n = np.array([ batch_size, batch_size, *num_tn, *num_tp, *num_dn, *num_dp ])
        valid_loss += l
        valid_num  += n

        #==========
        #dump results for debug
        if 0:
            image       = tensor_to_image(input)
            truth_mask  = tensor_to_mask(truth_mask)
            probability_mask  = tensor_to_mask(probability_mask)
            truth_label = truth_label.data.cpu().numpy()
            probability_label = probability_label.data.cpu().numpy()

            for b in range(batch_size):
                image_id = infor[b].image_id
                result = draw_predict_result(
                    image[b], truth_label[b], truth_mask[b], probability_label[b], probability_mask[b])

                image_show('result',result,resize=0.5)
                cv2.imwrite(out_dir +'/valid/%s.png'%image_id[:-4], result)
                cv2.waitKey(1)
                pass
        #==========

        #print(valid_loss)
        print('\r %8d /%d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/(valid_num+1e-8)

    #------
    test_pos_ratio = np.array(
        [NUM_TEST_POS[c][0]/NUM_TEST for c in list(CLASSNAME_TO_CLASSNO.keys())]
    )
    test_neg_ratio = 1-test_pos_ratio

    tn, tp, dn, dp = valid_loss[2:].reshape(-1,NUM_CLASS)
    kaggle = test_neg_ratio*tn + test_neg_ratio*(1-tn)*dn + test_pos_ratio*tp*dp
    kaggle = kaggle.mean()

    kaggle1 = test_neg_ratio*tn + test_pos_ratio*tp
    kaggle1 = kaggle1.mean()

    return valid_loss, (kaggle,kaggle1)





def run_train():
    out_dir = \
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-unet-fold_a2'
    initial_checkpoint = \
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-unet-fold_a2/checkpoint/00001500_model.pth'


    schduler = NullScheduler(lr=0.01)
    iter_accum = 2
    batch_size =16 #8

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = CloudDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['by_random1/train_fold_a2_5246.npy',],
        folder  = {'image': '1050x700', 'mask': '525x350'},
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    valid_dataset = CloudDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['by_random1/valid_fold_a2_300.npy',],
        folder  = {'image': '1050x700', 'mask': '525x350'},
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        # for k in list(state_dict.keys()):
        #      if any(s in k for s in ['logit',]): state_dict.pop(k, None)
        # net.load_state_dict(state_dict,strict=False)

        net.load_state_dict(state_dict,strict=True)  #True
    else:
        net.load_pretrain(is_print=False)


    log.write('net=%s\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0, weight_decay=0.0)

    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log    = 100
    iter_valid  = 250
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                    |------------------------------------------------- VALID------------------------------------------------------|---------------------- TRAIN/BATCH -----------------\n')
    log.write('rate    iter  epoch | kaggle      | loss                tn0,1,2,3 : tp0,1,2,3                     dn0,1,2,3 : dp0,1,2,3           | loss        dn : dp0,1,2,3           | time        \n')
    log.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
              #0.00000  28.0* 32.6 | 0.604,0.750 | 0.85,0.29  0.73 0.92 0.83 0.68: 0.54 0.70 0.60 0.84  0.00 0.00 0.00 0.00: 0.53 0.62 0.57 0.64 | 0.00,0.00  0.00: 0.00 0.00 0.00 0.00 | 0 hr 00 min

    def message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'):
        if mode==('print'):
            asterisk = ' '
            loss = batch_loss
        if mode==('log'):
            asterisk = '*' if iter in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f %5.1f%s %4.1f | '%(rate, iter/1000, asterisk, epoch,) +\
            '%0.3f,%0.3f | '%(*kaggle,) +\
            '%4.2f,%4.2f  %0.2f %0.2f %0.2f %0.2f: %0.2f %0.2f %0.2f %0.2f  %0.2f %0.2f %0.2f %0.2f: %0.2f %0.2f %0.2f %0.2f | '%(*valid_loss,) +\
            '%4.2f,%4.2f  %0.2f: %0.2f %0.2f %0.2f %0.2f |'%(*loss,) +\
            '%s' % (time_to_str((timer() - start_timer),'min'))

        return text

    #----
    kaggle = (0,0)
    valid_loss = np.zeros(18,np.float32)
    train_loss = np.zeros( 7,np.float32)
    batch_loss = np.zeros_like(valid_loss)
    iter = 0
    i    = 0



    start_timer = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        optimizer.zero_grad()
        for t, (input, truth_label, truth_mask, infor) in enumerate(train_loader):

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


            #if 0:
            if (iter % iter_valid==0):
                valid_loss, kaggle = do_valid(net, valid_loader, out_dir) #
                pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                log.write(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='log'))
                log.write('\n')

            #if 0:
            if iter in iter_save:
                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    pass



            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)

            net.train()
            input = input.cuda()
            truth_label = truth_label.cuda()
            truth_mask  = truth_mask.cuda()

            probability_label, probability_mask = data_parallel(net, input)
            probability_mask = resize_like(probability_mask, truth_mask, mode='bilinear')

            loss_label, loss_mask = criterion(probability_label, probability_mask, truth_label, truth_mask)

            #((loss_label+loss_mask )/iter_accum).backward()
            ((loss_mask )/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  --------
            tn,tp,num_tn,num_tp, dn,dp,num_dn,num_dp = metric (probability_label, probability_mask, truth_label, truth_mask, False)
            #print(num_tn,num_tp,num_dn,num_dp )
            l = np.array([ loss_label.item()*batch_size,loss_mask.item()*batch_size,dn.sum(),*dp ])
            n = np.array([ batch_size, batch_size, num_dn.sum(),*num_dp ])
            batch_loss      = l/(n+1e-8)
            sum_train_loss += l
            sum_train      += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum_train+1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0


            print('\r',end='',flush=True)
            print(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'), end='',flush=True)
            i=i+1


            # debug-----------------------------
            if 1:
                for di in range(3):
                    if (iter+di)%500==0:


                        image       = tensor_to_image(input)
                        truth_mask  = tensor_to_mask(truth_mask)
                        probability_mask  = tensor_to_mask(probability_mask)
                        truth_label = truth_label.data.cpu().numpy()
                        probability_label = probability_label.data.cpu().numpy()

                        for b in range(batch_size):
                            image_id = infor[b].image_id
                            result = draw_predict_result(
                                image[b], truth_label[b], truth_mask[b], probability_label[b], probability_mask[b])

                            image_show('result',result,resize=1)
                            cv2.imwrite(out_dir +'/train/%05d.png'%(di*100+b), result)
                            cv2.waitKey(1)
                            pass

        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

