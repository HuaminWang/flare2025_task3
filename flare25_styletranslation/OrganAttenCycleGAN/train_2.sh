# modas = ['OutPhase', 'C+Delay', 'C-pre', 'C+V', 'C+A', 'InPhase', 'DWI', 'T2WI']

export CUDA_VISIBLE_DEVICES=2
#python train.py --dataroot /home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/pet/ \
#--which_model_netD 'basic' --which_model_netG 'resnet_9blocks' --Net 'Unet' --model 'Organ_attention' \
#--lambda_A 10.0 --lambda_B 10.0 --SegLambda_B 5.0 --local_D_weight 0.5 --identity 0.5 --epoch_count 1 \
#--pool_size 50 --phase 'train' --niter 130 --niter_decay 15 --lr 1e-4 --display_freq 200 --update_html_freq 1000 --batchSize 2 \
#--checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/fdg_all' --gan_type 'lsgan' --data_aug \
#--display_freq 40 --organ_num 5 --target_domian fdg   #--partial_label

#python train.py --dataroot /home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri/ \
#--which_model_netD 'basic' --which_model_netG 'resnet_9blocks' --Net 'Unet' --model 'Organ_attention' \
#--lambda_A 10.0 --lambda_B 10.0 --SegLambda_B 5.0 --local_D_weight 0.5 --identity 0.5 --continue_train --which_epoch 124 --epoch_count 125 \
#--pool_size 50 --phase 'train' --niter 130 --niter_decay 15 --lr 1e-4 --display_freq 200 --update_html_freq 1000 --batchSize 2 \
#--checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/amos_all' --gan_type 'lsgan' --data_aug \
#--display_freq 40 --organ_num 14 --target_domian amos   #--partial_label
#
#python train.py --dataroot /home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri/ \
#--which_model_netD 'basic' --which_model_netG 'resnet_9blocks' --Net 'Unet' --model 'Organ_attention' \
#--lambda_A 10.0 --lambda_B 10.0 --SegLambda_B 5.0 --local_D_weight 0.5 --identity 0.5 --epoch_count 1 \
#--pool_size 50 --phase 'train' --niter 130 --niter_decay 15 --lr 1e-4 --display_freq 200 --update_html_freq 1000 --batchSize 2 \
#--checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/OutPhase_all' --gan_type 'lsgan' --data_aug \
#--display_freq 40 --organ_num 14 --target_domian OutPhase   #--partial_label

 python test.py --dataroot '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/pet' --phase train \
 --results_dir /home/data3/whm/dataset/flare25/fake_flare/fdg_all  \
 --Net 'Unet' --which_epoch 145 --organ_num 14 --G_A --target_domian fdg_all  \
 --checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/fdg_all'

 python test.py --dataroot '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri' --phase train \
 --results_dir /home/data3/whm/dataset/flare25/fake_flare/amos_all  \
 --Net 'Unet' --which_epoch 145 --organ_num 14 --G_A --target_domian amos_all  \
 --checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/amos_all'

 python test.py --dataroot '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri' --phase train \
 --results_dir /home/data3/whm/dataset/flare25/fake_flare/OutPhase  \
 --Net 'Unet' --which_epoch 145 --organ_num 14 --G_A --target_domian OutPhase  \
 --checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/OutPhase'
