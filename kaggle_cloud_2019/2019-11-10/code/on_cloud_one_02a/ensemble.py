import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from kaggle import *





############



def run_test_ensemble_segmentation_only():
    dir=[
         # '/root/share/project/kaggle/2019/cloud/result/run1/resnet18-unet-fold_a0_1/submit/test-tta',
         # '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn256-fold_a0_1/submit/test-tta1',

        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn256-fold_a1/submit/test-tta',
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn256-fold_a2/submit/test-tta',
        '/root/share/project/kaggle/2019/cloud/result/run1/resnet34-fpn256-fold_a0_5/submit/test-tta'
    ]

    out_dir = '/root/share/project/kaggle/2019/cloud/result/ensmble/xxx2'

    ############################################################
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir+'/log.ensemble-seg.txt',mode='a')

    if 1:
        for t,d in enumerate(dir):
            print(t,d)
            image_id          = read_list_from_file(d +'/image_id.txt')
            probability_label = np.load(d +'/probability_label.uint8.npz')['arr_0']
            probability_mask  = np.load(d +'/probability_mask.uint8.npz')['arr_0']
            probability_label = probability_label.astype(np.float32) /255
            probability_mask  = probability_mask.astype(np.float32) /255

            if t==0:
                ensemble_label = probability_label
                ensemble_mask  = probability_mask
            else:
                ensemble_label += probability_label
                ensemble_mask  += probability_mask


        print('')
        num_ensemble = len(dir)
        probability_label = ensemble_label/num_ensemble
        probability_mask  = ensemble_mask/num_ensemble
        probability_label = (probability_label*255).astype(np.uint8)
        probability_mask  = (probability_mask*255).astype(np.uint8)

        #---
        if 0:
            write_list_to_file (out_dir + '/image_id.txt', image_id)
            np.savez_compressed(out_dir + '/probability_label.uint8.npz', probability_label)
            np.savez_compressed(out_dir + '/probability_mask.uint8.npz', probability_mask)


    #---
    threshold_label      = [ 0.60, 0.60, 0.60, 0.60,]
    threshold_mask_pixel = [ 0.30, 0.30, 0.30, 0.30,]
    threshold_mask_size  = [ 1,  1,  1,  1,]


    # threshold_label      = [ 0.825, 1.00, 0.525, 0.50,]
    # threshold_mask_pixel = [ 0.40, 0.40, 0.40, 0.40,]
    # threshold_mask_size  = [ 1,  1,  1,  1,]
    #---

    if 1:
    #if mode =='test':
        csv_file = out_dir +'/ensemble-xxx.csv'

        log.write('test submission .... @ ensmble\n')
        print('')
        log.write('threshold_label = %s\n'%str(threshold_label))
        log.write('threshold_mask_pixel = %s\n'%str(threshold_mask_pixel))
        log.write('threshold_mask_size  = %s\n'%str(threshold_mask_size))
        log.write('\n')

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)

        #-----
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
        #-----


        ## print statistics ----
        text = summarise_submission_csv(df)
        log.write('\n')
        log.write('%s'%(text))

        ##evalue based on probing results
        # text = do_local_submit(image_id, predict_label,predict_mask=None)
        # log.write('\n')
        # log.write('%s'%(text))


        #--
        # local_result = find_local_threshold(image_id, probability_label, cutoff=[100,0,575,110])
        # threshold_label = [local_result[0][0],local_result[1][0],local_result[2][0],local_result[3][0]]
        # log.write('test threshold_label=%s\n'%str(threshold_label))
        #
        # predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        # text = do_local_submit(image_id, predict_label,predict_mask=None)
        # log.write('\n')
        log.write('%s'%(text))


# main #################################################################
if __name__ == '__main__':
    #print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_test_ensemble_segmentation_only()

