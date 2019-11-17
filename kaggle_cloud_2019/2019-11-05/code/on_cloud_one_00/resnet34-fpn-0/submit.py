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
                logit =  data_parallel(net,input)  #net(input)
                probability = torch.sigmoid(logit)

                probability_mask  = probability
                probability_label = probability_mask_to_label(probability)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[3]))
                probability  = torch.sigmoid(torch.flip(logit,dims=[3]))

                probability_mask  += probability
                probability_label += probability_mask_to_label(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[2]))
                probability = torch.sigmoid(torch.flip(logit,dims=[2]))

                probability_mask += probability
                probability_label+= probability_mask_to_label(probability)
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
    train_split = ['by_random1/valid_fold_a0_300.npy',]

    out_dir = \
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn256-fold_a0_1'
    initial_checkpoint = \
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn256-fold_a0_1/checkpoint/00014000_model.pth'


    ###############################################################3

    augment = ['null', 'flip_lr', ]  #['null'] #
    mode = 'test' #'valid' # 'test'
    mode_folder = 'test-tta' #tta  null

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
            augment = None,
        )

    if mode == 'test':
        test_dataset = CloudDataset(
            mode    = 'test',
            csv     = ['sample_submission.csv',],
            split   = ['test_3698.npy',],
            augment = None, #
        )

    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')
    #exit(0)


    ## start testing here! ##############################################
    #

    #---
    threshold_label      = [ 0.90, 0.90, 0.90, 0.90,]
    threshold_mask_pixel = [ 0.40, 0.40, 0.40, 0.40,]
    threshold_mask_size  = [   1,   1,   1,   1,]


    if 0: #save
        ## net ----------------------------------------
        log.write('** net setting **\n')

        net = Net().cuda()
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=True)

        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        log.write('%s\n'%(type(net)))
        log.write('\n')


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
    # if 0: #show
    #     if mode == 'train':
    #         folder='train_images'
    #         for b in range(num_test):
    #             print(b, image_id[b])
    #             image=cv2.imread(DATA_DIR+'/%s/%s'%(folder,image_id[b]), cv2.IMREAD_COLOR)
    #             result = draw_predict_result(
    #                 image,
    #                 truth_label[b],
    #                 truth_mask[b],
    #                 probability_label[b].astype(np.float32)/255,
    #                 probability_mask[b].astype(np.float32)/255
    #             )
    #             image_show('result',result,0.5)
    #             cv2.waitKey(0)

    #----
    if 1: #decode

        # value = np.max(probability_mask,1,keepdims=True)
        # value = probability_mask*(value==probability_mask)
        pass



    # inspect here !!!  ###################
    print('')
    log.write('submitting .... @ %s\n'%str(augment))
    log.write('threshold_label = %s\n'%str(threshold_label))
    log.write('threshold_mask_pixel = %s\n'%str(threshold_mask_pixel))
    log.write('threshold_mask_size  = %s\n'%str(threshold_mask_size))
    log.write('\n')

    if mode == 'valid':

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)


        log.write('** threshold_label **\n')
        kaggle, result = compute_metric_label(truth_label, predict_label)
        text = summarise_metric_label(kaggle, result)
        log.write('\n%s'%(text))

        auc, result = compute_roc_label(truth_label, probability_label)
        text = summarise_roc_label(auc, result)
        log.write('\n%s'%(text))



        log.write('** threshold_pixel **\n')
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s'%(text))

        #-----

        log.write('** threshold_pixel + threshold_label **\n')
        predict_mask = predict_mask * predict_label.reshape(-1,4,1,1)
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s'%(text))

        #-----

        log.write('** threshold_pixel + threshold_label + threshold_size **\n')
        predict_mask = remove_small(predict_mask, threshold_mask_size)
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s'%(text))

    ###################

    if mode =='test':
        log.write('test submission .... @ %s\n'%str(augment))
        csv_file = out_dir +'/submit/%s/resnet34-fpn35.csv'%(mode_folder)

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

        exit(0)

        ## print statistics ----
        print('initial_checkpoint=%s'%initial_checkpoint)
        text = summarise_submission_csv(df)
        log.write('\n')
        log.write('%s'%(text))

        ##evalue based on probing results
        text = do_local_submit(image_id, predict_label,predict_mask)
        log.write('\n')
        log.write('%s'%(text))


        #--
        local_result = find_local_threshold(image_id, probability_label, cutoff=[90,0,550,110])
        threshold_label = [local_result[0][0],local_result[1][0],local_result[2][0],local_result[3][0]]
        log.write('test threshold_label=%s\n'%str(threshold_label))

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        text = do_local_submit(image_id, predict_label,predict_mask=None)
        log.write('\n')
        log.write('%s'%(text))



    exit(0)
 

'''

submitting .... @ ['null']
threshold_label = [0.75, 0.85, 0.5, 0.5]
threshold_mask_pixel = [0.4, 0.4, 0.4, 0.4]
threshold_mask_size  = [40, 40, 40, 40]

test submission .... @ ['null']
initial_checkpoint=/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0-mish1/checkpoint/00056000_model.pth

compare with LB probing ... 
		num_image =  1801(1801) 
		num  =  7204(7204) 

		pos1 =    83( 128)  0.648
		pos2 =     9(  43)  0.209
		pos3 =   589( 741)  0.795
		pos4 =   115( 120)  0.958

		neg1 =  1718(1673)  1.027   45
		neg2 =  1792(1758)  1.019   34
		neg3 =  1212(1060)  1.143  152
		neg4 =  1686(1681)  1.003    5
--------------------------------------------------
		neg  =  6408(6172)  1.038  236 


              defect1    defect2    defect3    defect4 
num_t     :     128         43        741        120   
num_p     :      83          9        589        115   
precision :     0.855      0.111      0.874      0.948 
recall    :     0.555      0.023      0.695      0.908 
tp        :      71          1        515        109   
fp        :      12          8         74          6   
kaggle625 :     0.947      0.972      0.726      0.968 
gain625   :      32.4       -7.4      247.9       62.1 

kaggle @1.00  = 0.93948 
kaggle @0.70  = 0.91049 
kaggle @0.65  = 0.90566 *
kaggle @0.62  = 0.90325 **
kaggle @0.60  = 0.90083 *
kaggle @0.50  = 0.89117 
kaggle @0.00  = 0.84287 

test threshold_label=[0.7254902, 0.95686275, 0.54901963, 0.61960787]

              defect1    defect2    defect3    defect4 
num_t     :     128         43        741        120   
num_p     :      89          0        548        110   
precision :     0.831      0.000      0.907      0.991 
recall    :     0.578      0.000      0.671      0.908 
tp        :      74          0        497        109   
fp        :      15          0         51          1   
kaggle625 :     0.946      0.976      0.733      0.971 
gain625   :      31.2        0.0      259.6       67.1 

kaggle @1.00  = 0.94184 
kaggle @0.70  = 0.91352 
kaggle @0.65  = 0.90880 *
kaggle @0.62  = 0.90644 **
kaggle @0.60  = 0.90408 *
kaggle @0.50  = 0.89464 
kaggle @0.00  = 0.84745 


'''


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_submit_segmentation()
  
