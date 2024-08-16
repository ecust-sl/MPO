#!/bin/bash
seed=${RANDOM}
noamopt_warmup=1000

#RESUME=${'/home/shilei/project/R2GenRL/result/iu'}
python train_rl_iu.py \
    --image_dir /ailab/user/baichenjia/shilei/code/R2GenRL/data/iu_xray/images \
    --ann_path /ailab/user/baichenjia/shilei/code/R2GenRL/data/iu_xray/iu_annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 8 \
    --epochs 400 \
    --save_dir /ailab/user/baichenjia/shilei/code/R2GenRL/result/FusionModel/B1_RG \
    --step_size 1 \
    --gamma 0.8 \
    --seed 7580 \
    --topk 32 \
    --beam_size 3 \
    --log_period 100 \
    --monitor_metric BLEU_1 \
    --prefer_dim 2 \
    --atten_prefer_dim 128 \
    --num_layers 3 \
    --resume /ailab/user/baichenjia/shilei/code/R2GenRL/result/FusionModel/DCA_up_decoder4/model_best.pth
