#seed=${RANDOM}

#mkdir -p result/mimic_cxr/base_cmn_rl/
#mkdir -p reslut/mimic_cxr/base_cmn_rl/

python train_rl.py \
--image_dir /extra/shilei/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/images \
--ann_path /extra/shilei/dataset/annotations/mimic_annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 6 \
--epochs 280 \
--save_dir /home/shilei/project/R2GenRL/result/mimic_cxr/7580/NEW_B1_F1 \
--record_dir /home/shilei/project/R2GenRL/result/mimic_cxr/7580/NEW_B1_F1/reports \
--step_size 1 \
--gamma 0.8 \
--seed 7580 \
--topk 32 \
--monitor_metric BLEU_4 \
--sc_eval_period 1000 \
--prefer_dim 3 \
--index_period 500 \
--num_layers 3 \
#--resume /home/shilei/project/R2GenRL/result/mimic_cxr/7580/DCA_B1/model_best.pth

# python train_rl.py \
# --image_dir data/mimic_cxr/images/ \
# --ann_path data/mimic_cxr/annotation.json \
# --dataset_name mimic_cxr \
# --max_seq_length 100 \
# --threshold 10 \
# --batch_size 6 \
# --epochs 50 \
# --save_dir results/mimic_cxr/base_cmn_rl/ \
# --record_dir records/mimic_cxr/base_cmn_rl/ \
# --step_size 1 \
# --gamma 0.8 \
# --seed ${seed} \
# --topk 32 \
# --sc_eval_period 3000 \
# --resume results/mimic_cxr/mimic_cxr_0.8_1_16_5e-5_1e-4_3_3_32_2048_512_30799/current_checkpoint.pth
