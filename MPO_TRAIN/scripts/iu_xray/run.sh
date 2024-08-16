seed=${RANDOM}
noamopt_warmup=1000

RESUME=${'/home/shilei/project/R2GenRL/result/iu'}
python train.py \
    --image_dir /extra/shilei/dataset/physionet.org/files/iu_xray/images/ \
    --ann_path /extra/shilei/dataset/annotations/iu_annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 16 \
    --epochs 200 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --save_dir /home/shilei/project/R2GenRL/result/iu_xray/7580/BASE \
    --step_size 10 \
    --gamma 0.8 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --num_layers 3 \
    --seed 7580 \
    --topk 32 \
    --beam_size 3 \
    --log_period 100 \
#    --resume /home/shilei/project/R2GenCMN/results/iu_xray/model_best.pth \
