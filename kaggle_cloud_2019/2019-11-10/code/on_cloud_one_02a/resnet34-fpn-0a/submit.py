import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *

from kaggle import *



######################################################################################


def do_evaluate_segmentation(net, test_dataset, augment=[], out_dir=None):

    test_loader = DataLoader(
        test_dataset,
        sampler     = SequentialSampler(test_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    #----



    test_num  = 0
    test_id   = []
    test_probability_label = [] # 8bit
    test_probability_mask  = [] # 8bit
    test_truth_label = [] # 8bit
    test_truth_mask  = [] # 8bit

    start_timer = timer()
    for t, (input, truth_label, truth_mask, infor) in enumerate(test_loader):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                p_label, p_mask  =  data_parallel(net,input)  #net(input)
                p_mask = resize_like(p_mask, truth_mask, mode='bilinear')

                probability_mask  = p_mask
                probability_label = p_label
                num_augment+=1

            if 'flip_lr' in augment:
                p_label, p_mask   = data_parallel(net,torch.flip(input,dims=[3]))
                p_mask  = resize_like(torch.flip(p_mask,dims=[3]), truth_mask, mode='bilinear')

                probability_mask  += p_mask
                probability_label += p_label
                num_augment+=1

            if 'flip_ud' in augment:
                p_label, p_mask   = data_parallel(net,torch.flip(input,dims=[2]))
                p_mask = resize_like(torch.flip(p_mask,dims=[2]), truth_mask, mode='bilinear')

                probability_mask  += p_mask
                probability_label += p_label
                num_augment+=1

            #---
            probability_mask  = probability_mask/num_augment
            probability_label = probability_label/num_augment

        #---
        batch_size  = len(infor)
        truth_label = truth_label.data.cpu().numpy().astype(np.uint8)
        truth_mask  = truth_mask.data.cpu().numpy().astype(np.uint8)
        probability_mask = (probability_mask.data.cpu().numpy()*255).astype(np.uint8)
        probability_label = (probability_label.data.cpu().numpy()*255).astype(np.uint8)

        test_id.extend([i.image_id for i in infor])
        test_truth_label.append(truth_label)
        test_truth_mask.append(truth_mask)
        test_probability_label.append(probability_label)
        test_probability_mask.append(probability_mask)
        test_num += batch_size

        #---
        print('\r %4d / %4d  %s'%(
             test_num, len(test_loader.dataset), time_to_str((timer() - start_timer),'min')
        ),end='',flush=True)

    assert(test_num == len(test_loader.dataset))
    print('')

    start_timer = timer()
    test_truth_label = np.concatenate(test_truth_label)
    test_truth_mask  = np.concatenate(test_truth_mask)
    test_probability_label = np.concatenate(test_probability_label)
    test_probability_mask = np.concatenate(test_probability_mask)
    print(time_to_str((timer() - start_timer),'sec'))

    return test_id, test_truth_label, test_truth_mask, test_probability_label, test_probability_mask


######################################################################################
def run_submit_segmentation(

):
    train_split = ['by_random1/valid_fold_a2_300.npy',]

    out_dir = \
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn1-fold_a2'
    initial_checkpoint = \
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn1-fold_a2/checkpoint/00022000_model.pth'


    ###############################################################3

    augment = ['null', 'flip_lr', 'flip_ud', ]  #['null', 'flip_lr', ]  #['null'] #
    mode    = 'valid' #'valid' # 'test'
    mode_folder = 'valid-tta' #tta  null

    #---

    ## setup
    os.makedirs(out_dir +'/submit/%s'%(mode_folder), exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset -------

    log.write('** dataset setting **\n')
    if mode == 'valid':
        test_dataset = CloudDataset(
            mode    = 'train',
            csv     = ['train.csv',],
            split   = train_split,
            folder  = {'image': '1050x700', 'mask': '525x350'},
            augment = None,
        )

    if mode == 'test':
        test_dataset = CloudDataset(
            mode    = 'test',
            csv     = ['sample_submission.csv',],
            split   = ['test_3698.npy',],
            folder  = {'image': '1050x700', 'mask': '525x350'},
            augment = None, #
        )

    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')
    #exit(0)


    ## start testing here! ##############################################
    #

    #---
    threshold_label = [ 0.60, 0.60, 0.60, 0.60,]
    threshold_mask  = [ 0.40, 0.40, 0.40, 0.40,]
    threshold_size  = [ 1, 1, 1, 1,]

    print('')
    log.write('submitting .... @ %s\n'%str(augment))
    log.write('initial_checkpoint  = %s\n'%initial_checkpoint)
    log.write('threshold_label = %s\n'%str(threshold_label))
    log.write('threshold_mask  = %s\n'%str(threshold_mask))
    log.write('threshold_mask  = %s\n'%str(threshold_size))
    log.write('\n')

    if 0: #save
        ## net ----------------------------------------
        log.write('** net setting **\n')
        net = Net().cuda()
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=True)

        image_id, truth_label, truth_mask, probability_label, probability_mask,  =\
            do_evaluate_segmentation(net, test_dataset, augment)

        if 1: #save
            write_list_to_file (out_dir + '/submit/%s/image_id.txt'%(mode_folder),image_id)
            np.savez_compressed(out_dir + '/submit/%s/probability_label.uint8.npz'%(mode_folder), probability_label)
            np.savez_compressed(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode_folder), probability_mask)
            if mode == 'valid':
                np.savez_compressed(out_dir + '/submit/%s/truth_label.uint8.npz'%(mode_folder), truth_label)
                np.savez_compressed(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode_folder), truth_mask)

        #exit(0)

    if 1:
        image_id = read_list_from_file(out_dir + '/submit/%s/image_id.txt'%(mode_folder))
        probability_label = np.load(out_dir + '/submit/%s/probability_label.uint8.npz'%(mode_folder))['arr_0']
        probability_mask  = np.load(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode_folder))['arr_0']
        if mode == 'valid':
            truth_label       = np.load(out_dir + '/submit/%s/truth_label.uint8.npz'%(mode_folder))['arr_0']
            truth_mask        = np.load(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode_folder))['arr_0']

    num_test= len(image_id)
    if 0: #show
        if mode == 'valid':
            folder='image/train_1050x700'
            for b in range(num_test):
                print(b, image_id[b])

                image = cv2.imread(DATA_DIR+'/%s/%s.png'%(folder,image_id[b][:-4]), cv2.IMREAD_COLOR)
                image = image.astype(np.float32)/255

                t_label = truth_label[b]
                t_mask = truth_mask[b]
                t_mask = t_mask.transpose(1,2,0)

                p_label = probability_label[b].astype(np.float32)/255
                p_mask = probability_mask[b].astype(np.float32)/255
                p_mask = p_mask.transpose(1,2,0)

                result = draw_predict_result(
                    image, t_label, t_mask, p_label,  p_mask,
                    threshold = [
                        threshold_label,
                        threshold_mask
                    ]
                )
                image_show('result',result)
                cv2.waitKey(0)


    # inspect here !!!  ###################

    if mode == 'valid':
        probability_label = (probability_label/255).astype(np.float32)
        probability_mask = (probability_mask/255).astype(np.float32)

        #---
        log.write('** all threshold **\n')
        result = compute_metric( probability_label, probability_mask, truth_label, truth_mask,
                        threshold=(
                            threshold_label,
                            threshold_mask,
                            threshold_size
                        ) )

        text = summarise_metric(result)
        log.write('\n%s'%(text))
        log.write('\n')

        #---
        log.write('** segmentation only **\n')
        result = compute_metric( probability_label, probability_mask, truth_label, truth_mask,
                        threshold=(
                            [-1,-1,-1,-1], #threshold_label
                            threshold_mask,
                            threshold_size
                        ) )

        text = summarise_metric(result)
        log.write('\n%s'%(text))
        log.write('\n')

        #---
        result = compute_label_metric(probability_label, truth_label)
        text   = summarise_label_metric(result)
        log.write('\n%s'%(text))
        log.write('\n')


    ###################

    if mode =='test':
        log.write('test submission .... @ %s\n'%str(augment))
        csv_file = out_dir +'/submit/%s/resnet34-fpn-xxx.csv'%(mode_folder)

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)

        image_id_class_id = []
        encoded_pixel = []
        for b in range(len(image_id)):
            for c in range(NUM_CLASS):
                image_id_class_id.append(image_id[b]+'_%s'%(CLASSNO_TO_CLASSNAME[c]))

                if predict_label[b,c]==0:
                    rle=''
                else:
                    rle = run_length_encode(predict_mask[b,c])
                encoded_pixel.append(rle)

        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['Image_Label', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)


        ## print statistics ----
        print('initial_checkpoint=%s'%initial_checkpoint)
        text = summarise_submission_csv(df)
        log.write('\n')
        log.write('%s'%(text))





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_submit_segmentation()
  
