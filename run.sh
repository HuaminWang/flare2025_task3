# export CUDA_VISIBLE_DEVICES=1
#python inference.py --cfg ./configs/inference/inference_small_segnet.yaml 

# python inference.py --cfg ./configs/inference/inference_small_segnet.yaml
# python eval.py --cfg ./configs/eval/eval_small_segnet_mr.yaml
# python ./outputs/flare25_pet_lab_affine.py -raw_dir ./outputs/CT_MR_TJU_init1/PET_labelsVal_pred/o_13 -save_dir ./outputs/CT_MR_TJU_init1/PET_labelsVal_pred/o_4
# flare25/datasets/pseudolab/1adj_spacing/train_PET_unlabeled/CT_MR_TJU
# python ./outputs/flare25_pet_lab_affine.py -raw_dir ./datasets/pseudolab/1adj_spacing/train_PET_unlabeled/CT_MR_TJU/_13 -save_dir ./datasets/pseudolab/1adj_spacing/train_PET_unlabeled/CT_MR_TJU/o_4
# python ./outputs/flare25_pet_lab_affine.py -raw_dir ./datasets/pseudolab/1adj_spacing/train_PET_unlabeled/CT550/_13 -save_dir ./datasets/pseudolab/1adj_spacing/train_PET_unlabeled/CT550/o_4
# python eval.py --cfg ./configs/eval/eval_small_segnet_pet.yaml


# python ./preprocess/data_preprocess.py --cfg ./configs/preprocess/preprocess_step3_lld_fine.yaml
# python ./preprocess/data_preprocess.py --cfg ./configs/preprocess/preprocess_step1_CT.yaml
# python ./preprocess/data_preprocess.py --cfg ./configs/preprocess/preprocess_step3_amos_fine_vote.yaml
# python ./preprocess/data_preprocess.py --cfg ./configs/preprocess/preprocess_step3_lld_fine_vote.yaml
# python ./preprocess/data_preprocess.py --cfg ./configs/preprocess/preprocess_step3_pet_fine_vote.yaml # *2


export CUDA_VISIBLE_DEVICES=3
# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune3.yaml
# python ./preprocess/data_preprocess.py --cfg ./configs/preprocess/preprocess_step1_FakeMR.yaml
# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_pet_3.yaml
# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_pet_1.yaml

# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_pet_vote_01.yaml
python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_pet_vote_02.yaml
# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_mr_vote_01.yaml
# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_mr_vote_02.yaml
# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_mr_vote_03.yaml
# python train.py --cfg ./configs/train/train_small_segnet_fine_stage_fintune_mr_vote_04.yaml

