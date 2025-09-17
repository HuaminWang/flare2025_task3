# export CUDA_VISIBLE_DEVICES=1
#ln -s /workspace/inputs/ /workspace/datasets/inputs
python inference.py --cfg ./configs/inference/inference_small_segnet.yaml
python flare25_pet_lab_affine.py -raw_dir ./outputs/ -save_dir ./outputs/
