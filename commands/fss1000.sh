python scripts/segmentation_train.py \
 --data_name FSS \
 --data_dir /images/PublicDataset/fewshot_datasets/ \
 --out_dir /home/staff/azad/deeplearning/afshin/MICCAI_2023/diffusion/MedSegDiff/output/fss \
 --image_size 256 \
 --num_channels 128 \
 --class_cond False \
 --num_res_blocks 2 \
 --num_heads 1 \
 --learn_sigma True \
 --use_scale_shift_norm False \
 --attention_resolutions 16 \
 --diffusion_steps 1000 \
 --noise_schedule linear \
 --rescale_learned_sigmas False \
 --rescale_timesteps False \
 --lr 1e-4 \
 --batch_size 8


python scripts/segmentation_sample.py \
 --data_name FSS \
 --data_dir /images/PublicDataset/fewshot_datasets/ \
 --out_dir /home/staff/azad/deeplearning/afshin/MICCAI_2023/diffusion/MedSegDiff/results/fss/ \
 --model_path /home/staff/azad/deeplearning/afshin/MICCAI_2023/diffusion/MedSegDiff/output/fss/savedmodel015000.pt \
 --image_size 256 \
 --num_channels 128 \
 --class_cond False \
 --num_res_blocks 2 \
 --num_heads 1 \
 --learn_sigma True \
 --use_scale_shift_norm False \
 --attention_resolutions 16 \
 --diffusion_steps 1000 \
 --noise_schedule linear \
 --rescale_learned_sigmas False \
 --rescale_timesteps False \
 --num_ensemble 5










best="--data_name FSS --data_dir /images/PublicDataset/fewshot_datasets/ --out_dir /home/staff/azad/deeplearning/afshin/MICCAI_2023/diffusion/MedSegDiff/tr_res/best/fss --image_size 256 --num_channels 512 --class_cond False --num_res_blocks 12 --num_heads 8 --learn_sigma True --use_scale_shift_norm True --attention_resolutions 24 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8"
