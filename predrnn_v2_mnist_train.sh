export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name asl \
    --train_data_paths /path/to/asl_dataset/train \
    --valid_data_paths /path/to/asl_dataset/valid \
    --save_dir checkpoints/asl_predrnn_v2 \
    --gen_frm_dir results/asl_predrnn_v2 \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \  # Update this based on ASL dataset image width
    --img_channel 1 \  # Update this based on ASL dataset image channels
    --input_length 30 \  # Update this based on your requirements
    --total_length 60 \  # Update this based on your requirements
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \  # Adjust batch size as needed
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
#    --pretrained_model ./checkpoints/asl_predrnn_v2/asl_model.ckpt
