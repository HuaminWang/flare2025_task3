# modas = ['OutPhase', 'C+Delay', 'C-pre', 'C+V', 'C+A', 'InPhase', 'DWI', 'T2WI']

export CUDA_VISIBLE_DEVICES=0
#python train.py --dataroot /home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/pet/ \
#--which_model_netD 'basic' --which_model_netG 'resnet_9blocks' --Net 'Unet' --model 'Organ_attention' \
#--lambda_A 10.0 --lambda_B 10.0 --SegLambda_B 5.0 --local_D_weight 0.5 --identity 0.5 --epoch_count 1 \
#--pool_size 50 --phase 'train' --niter 130 --niter_decay 15 --lr 1e-4 --display_freq 200 --update_html_freq 1000 --batchSize 2 \
#--checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/psma_all' --gan_type 'lsgan' --data_aug \
#--display_freq 40 --organ_num 5 --target_domian psma   #--partial_label
#
#python train.py --dataroot /home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri/ \
#--which_model_netD 'basic' --which_model_netG 'resnet_9blocks' --Net 'Unet' --model 'Organ_attention' \
#--lambda_A 10.0 --lambda_B 10.0 --SegLambda_B 5.0 --local_D_weight 0.5 --identity 0.5 --epoch_count 1 \
#--pool_size 50 --phase 'train' --niter 130 --niter_decay 15 --lr 1e-4 --display_freq 200 --update_html_freq 1000 --batchSize 2 \
#--checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/C+V_all' --gan_type 'lsgan' --data_aug \
#--display_freq 40 --organ_num 14 --target_domian C+V  #--partial_label
#
#python train.py --dataroot /home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri/ \
#--which_model_netD 'basic' --which_model_netG 'resnet_9blocks' --Net 'Unet' --model 'Organ_attention' \
#--lambda_A 10.0 --lambda_B 10.0 --SegLambda_B 5.0 --local_D_weight 0.5 --identity 0.5 --epoch_count 1 \
#--pool_size 50 --phase 'train' --niter 130 --niter_decay 15 --lr 1e-4 --display_freq 200 --update_html_freq 1000 --batchSize 2 \
#--checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/C+A_all' --gan_type 'lsgan' --data_aug \
#--display_freq 40 --organ_num 14 --target_domian C+A   #--partial_label
#
#
# python test.py --dataroot '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/pet/' --phase train \
# --results_dir /home/data3/whm/dataset/flare25/fake_flare/psma_all  \
# --Net 'Unet' --which_epoch 145 --organ_num 14 --G_A --target_domian psma_all  \
# --checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/psma_all'
#
# python test.py --dataroot '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri/' --phase train \
# --results_dir /home/data3/whm/dataset/flare25/fake_flare/C+V_all  \
# --Net 'Unet' --which_epoch 145 --organ_num 14 --G_A --target_domian C+V_all  \
# --checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/C+V_all'
#
# python test.py --dataroot '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri/' --phase train \
# --results_dir /home/data3/whm/dataset/flare25/fake_flare/C+A_all  \
# --Net 'Unet' --which_epoch 145 --organ_num 14 --G_A --target_domian C+A_all  \
# --checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/C+A_all'

 python test.py --dataroot '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/AllDataTrack/mri/' --phase train \
 --results_dir /home/data3/whm/dataset/flare25/fake_flare/OutPhase_all  \
 --Net 'Unet' --which_epoch 145 --organ_num 14 --G_A --target_domian OutPhase_all  \
 --checkpoints_dir '/home/data3/whm/code/flare25_styletranslation/OrganAttenCycleGAN/checkpoints/OutPhase_all'
