python main_test.py \
    --image_dir /ailab/user/baichenjia/shilei/code/R2GenRL/data/iu_xray/images \
    --ann_path /ailab/user/baichenjia/shilei/code/R2GenRL/data/iu_xray/iu_annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --batch_size 16 \
    --lr_ve 1e-6 \
    --lr_ed 1e-5 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 4 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir /ailab/user/baichenjia/shilei/code/R2GenTest/results/iu_xray \
    --log_period 50 \
    --prefer_dim 2 \
    #--load /ailab/user/baichenjia/shilei/code/R2GenRL/result/FusionModel/B1_RG_NO_CMN/model_best.pth