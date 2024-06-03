export OPENAI_LOGDIR='logs/sample'
export NCCL_P2P_DISABLE=1

# CelebAMask-HQ
# 25-step Generation
mpiexec -n 4 python image_sample.py --data_dir data/celeba --dataset_mode celeba --attention_resolutions 32,16,8 \
--diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
--use_scale_shift_norm True --num_classes 19 --class_cond True --no_instance False --batch_size 10 --num_samples 2000 --model_path checkpoints/scdm_ade20k/model.pt \
--results_path results/celeba-25 --cond_diffuse True --cond_opt discrete_zero_classwise --seed 42 \
--gpus_per_node 4 --s 1.5 --dynamic_threshold 0.95 --extrapolation 0.8 --timestep_respacing ddim25;

# 1000-step Generation
mpiexec -n 4 python image_sample.py --data_dir data/celeba --dataset_mode celeba --attention_resolutions 32,16,8 \
--diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
--use_scale_shift_norm True --num_classes 19 --class_cond True --no_instance False --batch_size 10 --num_samples 2000 --model_path checkpoints/scdm_coco.pt \
--results_path results/celeba-1000 --cond_diffuse True --cond_opt discrete_zero_classwise --seed 42 \
--gpus_per_node 4 --s 1.5

##################

# ADE20K
# 25-step Generation
mpiexec -n 4 python image_sample.py --data_dir data/ade20k --dataset_mode ade20k --attention_resolutions 32,16,8 \
--diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
--use_scale_shift_norm True --num_classes 151 --class_cond True --no_instance True --batch_size 10 --num_samples 2000 --model_path checkpoints/scdm_ade20k/model.pt \
--results_path results/ade20k-25 --cond_diffuse True --cond_opt discrete_zero_classwise --seed 42 \
--gpus_per_node 4 --s 1.5 --dynamic_threshold 0.95 --extrapolation 0.8 --timestep_respacing ddim25;

# 1000-step Generation
mpiexec -n 4 python image_sample.py --data_dir data/ade20k --dataset_mode ade20k --attention_resolutions 32,16,8 \
--diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
--use_scale_shift_norm True --num_classes 151 --class_cond True --no_instance True --batch_size 10 --num_samples 2000 --model_path checkpoints/scdm_coco.pt \
--results_path results/ade20-1000 --cond_diffuse True --cond_opt discrete_zero_classwise --seed 42 \
--gpus_per_node 4 --s 1.5

##################

# COCO
# 25-step Generation
mpiexec -n 4 python image_sample.py --data_dir data/coco --dataset_mode coco --attention_resolutions 32,16,8 \
--diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
--use_scale_shift_norm True --num_classes 183 --class_cond True --no_instance False --batch_size 10 --num_samples 5000 --model_path checkpoints/scdm_ade20k/model.pt \
--results_path results/coco-25 --cond_diffuse True --cond_opt discrete_zero_classwise --seed 42 \
--gpus_per_node 4 --s 1.5 --dynamic_threshold 0.95 --extrapolation 0.8 --timestep_respacing ddim25;

# 1000-step Generation
mpiexec -n 4 python image_sample.py --data_dir data/coco --dataset_mode coco --attention_resolutions 32,16,8 \
--diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
--use_scale_shift_norm True --num_classes 183 --class_cond True --no_instance False --batch_size 10 --num_samples 5000 --model_path checkpoints/scdm_coco.pt \
--results_path results/coco-1000 --cond_diffuse True --cond_opt discrete_zero_classwise --seed 42 \
--gpus_per_node 4 --s 1.5
