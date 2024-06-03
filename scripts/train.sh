export NCCL_P2P_DISABLE=1

# CelebAMask-HQ
# scratch
export OPENAI_LOGDIR='logs/train/celeba-scratch'
mpiexec -n 4 python image_train.py --data_dir data/CelebAMask-HQ --dataset_mode celeba \
    --lr 2e-5 --batch_size 10 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
    --use_scale_shift_norm True --use_checkpoint True --num_classes 19  --class_cond True --no_instance False \
    --save_interval 1000 --cond_diffuse True --gpus_per_node 4 --cond_opt discrete_zero_classwise --drop_rate 0.2 

# resume
export OPENAI_LOGDIR='logs/train/celeba-resume'
mpiexec -n 4 python image_train.py --data_dir data/CelebAMask-HQ --dataset_mode celeba \
    --lr 2e-5 --batch_size 10 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
    --use_scale_shift_norm True --use_checkpoint True --num_classes 19  --class_cond True --no_instance False \
    --resume_checkpoint checkpoints/celeba/model.pt --save_interval 1000 --cond_diffuse True --gpus_per_node 4 --cond_opt discrete_zero_classwise --drop_rate 0.2 


##################

# ADE20K
# scratch
export OPENAI_LOGDIR='logs/train/ade20k-scratch'
mpiexec -n 4 python image_train.py --data_dir data/ADE20K/ADEChallengeData2016 --dataset_mode ade20k \
    --lr 1e-4 --batch_size 10 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
    --use_scale_shift_norm True --use_checkpoint True --num_classes 151  --class_cond True --no_instance True \
    --save_interval 1000 --cond_diffuse True --gpus_per_node 4 --cond_opt discrete_zero_classwise --drop_rate 0.2 

# resume
export OPENAI_LOGDIR='logs/train/ade20k-resume'
mpiexec -n 4 python image_train.py --data_dir data/ADE20K/ADEChallengeData2016 --dataset_mode ade20k \
    --lr 2e-5 --batch_size 10 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
    --use_scale_shift_norm True --use_checkpoint True --num_classes 151  --class_cond True --no_instance True \
    --resume_checkpoint checkpoints/ade20k/model.pt --save_interval 1000 --cond_diffuse True --gpus_per_node 4 --cond_opt discrete_zero_classwise --drop_rate 0.2 

##################

# COCO
# scratch
export OPENAI_LOGDIR='logs/train/coco-scratch'
mpiexec -n 4 python image_train.py --data_dir data/coco --dataset_mode coco \
    --lr 1e-4 --batch_size 10 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
    --use_scale_shift_norm True --use_checkpoint True --num_classes 183  --class_cond True --no_instance False \
    --save_interval 1000 --cond_diffuse True --gpus_per_node 4 --cond_opt discrete_zero_classwise --drop_rate 0.2 

# resume
export OPENAI_LOGDIR='logs/train/coco-resume'
mpiexec -n 4 python image_train.py --data_dir data/coco --dataset_mode coco \
    --lr 2e-5 --batch_size 10 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
    --use_scale_shift_norm True --use_checkpoint True --num_classes 183  --class_cond True --no_instance False \
    --resume_checkpoint checkpoints/coco/model.pt --save_interval 1000 --cond_diffuse True --gpus_per_node 4 --cond_opt discrete_zero_classwise --drop_rate 0.2 
