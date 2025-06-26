#!/bin/bash

s=$1
p=$2
i=$3
j=$4
g=$5

train="./dataset/nss$i/train/NS"
train+="_FullCV_$j.csv"
val="./dataset/nss$i/val/NS"
val+="_FullCV_$j.csv"
test="./dataset/nss$i/test/NS"
test+="_FullCV_$j.csv"
ckpt="./perm_results/soft_perm/nss$i/reg0.1/cv$j"
ckpt+="_ckpts_ds2s_h20"

rm -rf $ckpt/*
mkdir -p $ckpt/fold_0/
mkdir -p $ckpt/fold_0/perms
cp -r val_perms/$j/* chemprop/train/val_perm

python train.py \
    --data_path $train \
    --separate_val_path $val \
    --separate_test_path $test \
    --save_dir $ckpt \
    --gpu $g \
    --dataset_type regression \
    --metric mae \
    --extra_metrics rmse r2 \
    --reaction --reaction_mode reac_diff \
    --explicit_h \
    --use_lrl_network \
    --format perm \
    --perm_type soft \
    --similarity max \
    --gumbel_noise_factor 1.0 \
    --gumbel_temperature 0.1 \
    --sinkhorn_iters 10 \
    --perm_regularizer squared_fro_norm_inv \
    --perm_reg_lambda 0.1 \
    --loss_agg_func sum \
    --epochs 100 \
    --init_lr 0.0001\
    --max_lr 0.001 \
    --final_lr 0.0001 \
    --use_node_tf \
    --aggregation set2set \
    --step_aggregation set2set \
    --seed $s \
    --batch_size 8 \
    --ffn_hidden_size 20 \
    --hidden_size 20 \
    --pytorch_seed $p \
    # --use_indicator\
    # --use_node_tf\
    # --use_step_tf\
    # --step_tf NNAttention\
    # --aggregation set2set\

rm -rf chemprop/train/val_perm/*
